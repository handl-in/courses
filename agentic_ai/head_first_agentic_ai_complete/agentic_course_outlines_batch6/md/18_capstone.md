# Module 18 Outline — Capstone: Ship a Real Multi-Agent System

::: chapter-opener
<div class="module-num">MODULE 18 — OUTLINE</div>
<div class="module-title">Capstone: Ship a Real Multi-Agent System</div>
<div class="subtitle">Everything from Modules 1-17, applied to one real artifact.<br>Three implementations. Honest comparison. Framework or no framework.</div>
<div class="pages">Target length: ~42 pages (heaviest in the book)</div>
:::

## What this module is

The capstone. We build a production-grade multi-agent research system that applies every concept from the prior modules — agent loop, tools, MCP, model routing, context engineering, compression, memory, hallucination defense, reflection, guardrails, multi-agent architecture, transactional patterns, observability, eval. We build it three times: raw Anthropic SDK, LangGraph, and CrewAI. Same spec, three implementations. Then we benchmark and discuss honestly when to use which.

The system itself is real: an orchestrator + parallel research workers + critic + report writer that handles deep research queries with grounded citations, configurable depth, cost ceilings, full traces, and an eval harness fed from production traces.

By the end the reader has working code (or a working understanding of working code) for a system they could ship — and the framework knowledge to make a defensible architectural choice on their next project.

::: hook
"At every conference talk on multi-agent systems, someone asks the same question from the audience: 'so should we use a framework or not?' The honest answer is 'it depends, and here are the four things it depends on.' Let's actually answer the question by building the same system three ways and looking at the receipts."
:::

---

## Section 1 — The Spec (≈3 pages)

The system we're building. A deep research agent with these properties:

**Behavior.**
- Takes a research question (single sentence to a paragraph)
- Plans a research strategy (decomposes into 3-6 sub-questions)
- Dispatches parallel research workers to investigate each sub-question
- Each worker has access to web search and web fetch tools
- A critic agent reviews the gathered evidence for gaps and inconsistencies
- A report writer synthesizes findings into a final report with citations
- Total wall-clock budget: 5 minutes default
- Total cost budget: $5 default per research run
- Output: structured markdown report with inline citations

**Production-grade requirements.**
- OpenTelemetry instrumentation (Module 17)
- Cost limiter and loop detection (Module 17)
- Hallucination defense layer on the report (Module 10)
- Guardrails on input (topic check) and output (PII leak detection) (Module 12)
- Reflection pattern: critic feedback feeds back into report revision (Module 11)
- Eval harness with production-trace sampling (Module 17)
- Graceful failure: partial results > nothing if some workers fail (Module 14)
- Configurable concurrency limits and depth caps (Module 7)

**What we're NOT building.**
- Saga patterns (Module 16) — research is read-only; no compensations needed
- Long-term memory (Module 9) — single-session for the capstone; readers can extend
- Computer-use (Module 17) — out of scope
- A full UI — we ship a CLI + library; UI is left as exercise

This spec is small enough to actually build in a chapter, large enough to exercise the full stack. We won't be cute; we ship working code.

::: pullquote
A spec is a promise to your future self. The spec is what you build *to*; everything else is decoration. We're going to keep our eyes on this list.
:::

---

## Section 2 — Architecture Decisions (≈4 pages)

Before we code, we decide. This is where Modules 13's threshold test, 14's pattern selection, and 17's instrumentation choices show up.

**Multi-agent or single agent?** This task is the canonical Anthropic-research-system fit:
- Parallelizable (sub-questions are independent)
- High enough value (deep research output is worth $5+ in API cost)
- Latency budget is generous (5 minutes is fine)
- Sub-tasks need genuinely different focus (each sub-question has its own evidence)

Threshold test (Module 13 §8): 5/5. Multi-agent is justified.

**Which architecture (Module 14)?** Supervisor / orchestrator-worker. Same shape as Anthropic's published research system. Other options considered and rejected:
- Hierarchical: overkill at 4-6 workers
- Sequential pipeline: loses parallelism (the whole point)
- Group chat: cost explosion, no benefit on this task
- Pure debate: doesn't fit ("debate" makes sense for opinion synthesis; "research" is evidence aggregation)

We add a critic step at the end (Module 11's reflection / Module 14's debate-as-subpattern), but the outer architecture is supervisor.

**Which model where (Module 6)?**
- **Orchestrator**: Opus (planning is high-leverage; mistakes propagate)
- **Workers**: Sonnet (capable, faster, cheaper for fan-out)
- **Critic**: Opus (validation needs strong reasoning; cheap critics are useless — Module 16 §6)
- **Report writer**: Opus (final output quality is what users see)
- **Topic guardrail classifier**: Haiku (cheap, fast)

This routing trades cost vs quality where each one matters. Total expected cost: $1-3 per research depending on depth.

**Context strategy (Modules 7-8).**
- Workers: their own context window per sub-question (isolation)
- Workers return *structured findings* (not raw transcripts) to orchestrator
- Orchestrator never sees worker raw context — only synthesized structured outputs (avoids supervisor context bloat)
- Critic sees the report draft + the workers' structured findings (enough for cross-checking)
- Report writer sees structured findings + critic feedback

This is "isolate" + "compress" from Module 7's vocabulary.

**Memory (Module 9).** Single-session for the capstone. Episodic memory of the current research: what's been searched, what evidence has been found, what gaps remain.

**Hallucination defense (Module 10).** Citation enforcement — every claim in the report must cite a worker's finding. Report writer can't reference facts not in workers' outputs. Validate citations programmatically.

**Reflection (Module 11).** The critic step. Independent agent (different prompt, sees the workers' raw structured findings + the draft report) checks for: claims unsupported by evidence, missing perspectives, internal contradictions. Writer revises (max 1 revision round; quality saturates fast).

**Guardrails (Module 12).** Input: topic classifier (is this a research question? not a coding-help / personal-advice / off-topic query). Output: PII leak detection (the report shouldn't contain real-looking personal data).

**Cost ceiling (Module 17).** Per-research budget of $5 (configurable). Tracks running cost across all agents. Halts if exceeded. Returns partial results.

**Observability (Module 17).** Full OpenTelemetry instrumentation. Each agent invocation = a span. Each tool call = a span. Trace context propagates across sub-agent boundaries. Traces ship to Langfuse (chosen for self-hosting; LangSmith is a one-line swap).

::: brain
Our orchestrator uses Opus; workers use Sonnet. Why not Haiku for workers since there are 4-6 of them and cost matters most there?

(Workers do the actual research — searching, fetching, reading, synthesizing per sub-question. Quality of worker output gates everything downstream. Haiku-quality findings would propagate through the whole pipeline. The right place to economize is the topic classifier, not the workers. Cost-optimize where errors are recoverable; spend at quality-gating points.)
:::

---

## Section 3 — Implementation 1: Raw Anthropic SDK (≈8 pages)

The minimal-dependency version. We code it in full and walk through it.

**Project structure.**
```
research_capstone/
├── pyproject.toml
├── src/research_capstone/
│   ├── __init__.py
│   ├── orchestrator.py      ← supervisor agent
│   ├── workers.py           ← research workers
│   ├── critic.py            ← reflection critic
│   ├── writer.py            ← report writer
│   ├── tools.py             ← web_search, web_fetch
│   ├── guardrails.py        ← input/output guardrails
│   ├── instrumentation.py   ← OpenTelemetry setup
│   ├── budgets.py           ← cost limiter, loop detector
│   ├── eval/
│   │   ├── harness.py
│   │   └── sampler.py       ← production trace -> eval queue
│   └── cli.py
```

**The orchestrator.**

```python
# orchestrator.py (excerpt — full code in book)
from anthropic import AsyncAnthropic
from opentelemetry import trace as otel_trace

class ResearchOrchestrator:
    def __init__(
        self,
        client: AsyncAnthropic,
        budget_usd: float = 5.0,
        wall_clock_seconds: float = 300.0,
        max_workers: int = 6,
    ):
        self.client = client
        self.budget = CostLimiter(budget_usd)
        self.deadline = wall_clock_seconds
        self.max_workers = max_workers
        self.tracer = otel_trace.get_tracer(__name__)

    async def research(self, question: str) -> ResearchReport:
        with self.tracer.start_as_current_span("research.run") as span:
            span.set_attribute("research.question", question)
            
            # 1. Input guardrail
            await self._check_input(question)
            
            # 2. Plan
            plan = await self._plan(question)
            
            # 3. Dispatch workers in parallel
            findings = await self._dispatch_workers(plan)
            
            # 4. Critic
            critique = await self._critique(question, plan, findings)
            
            # 5. Writer (with optional revision based on critique)
            report = await self._write(question, findings, critique)
            
            # 6. Output guardrail
            await self._check_output(report)
            
            # 7. Return with full trace
            return report
```

We walk through every method. The plan tool's schema. How workers stream findings back as structured JSON. How the critic's structured output feeds the writer. How partial-failure is handled (workers fail; surviving workers' findings still flow forward).

**The worker.**

```python
# workers.py (excerpt)
class ResearchWorker:
    def __init__(self, client, sub_question: str, budget: CostLimiter, tracer):
        ...
    
    async def research(self) -> WorkerFinding:
        with self.tracer.start_as_current_span("worker.research") as span:
            span.set_attribute("worker.sub_question", self.sub_question)
            
            # Plan -> tool calls -> synthesize finding
            messages = [{"role": "user", "content": WORKER_PROMPT.format(
                sub_question=self.sub_question
            )}]
            
            for turn in range(self.max_turns):
                resp = await self.client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=4096,
                    tools=[WEB_SEARCH_TOOL, WEB_FETCH_TOOL, FINISH_TOOL],
                    messages=messages,
                )
                self.budget.add(resp.usage)  # propagates cost upward
                
                # Process tool calls or finish
                ...
            
            return WorkerFinding(
                sub_question=self.sub_question,
                claim=...,
                evidence=...,  # list of {source_url, snippet, retrieved_at}
                confidence=...,
            )
```

**The critic.**

```python
# critic.py (excerpt)
class Critic:
    async def critique(self, question, findings, draft_report) -> Critique:
        # Independent — uses Opus, different prompt, sees both workers' raw findings
        # AND the draft report; flags claims unsupported by findings, gaps, contradictions
        ...
```

**The writer.**

```python
# writer.py (excerpt)
class Writer:
    async def write(self, question, findings, critique=None) -> Report:
        # First pass: produce report citing findings
        # If critique provided and has issues: revision pass
        ...
```

**Wiring it together.** A short `main.py` shows the CLI usage. We run it on a sample question end to end and show a real trace tree, real cost, real output.

We also show what *fails*: pulling a typical edge case (a question that's mostly opinion, a question that has no good sources, a question where one worker times out) and walking through how the system handles each.

::: code-exercise
**Exercise 18.1 — Run the raw-SDK capstone.**

Clone the capstone repo (we provide it). Set your API key. Run on three questions:
1. "What were the major findings of the 2025 Apollo retrospective study on RLHF?"
2. "Compare the technical trade-offs between QUIC and HTTP/2 for low-latency video."
3. "What are the strongest current critiques of effective altruism as a movement?"

Inspect the traces (Langfuse dashboard). Note: total cost, wall-clock, which workers finished, which findings made it into the report, what the critic flagged.

Modify: change the orchestrator to use Sonnet instead of Opus. Re-run. Compare quality and cost.
:::

---

## Section 4 — Implementation 2: LangGraph (≈6 pages)

The same system in LangGraph. We don't repeat the conceptual content — we focus on what LangGraph adds and what it costs.

**What LangGraph adds:**

- Explicit state graph: nodes, edges, conditional routing
- Built-in checkpointing: pause, inspect, resume, time-travel debug
- LangGraph Studio: visualize the graph, set breakpoints (genuinely the best agent IDE in 2026 per the field consensus)
- Tight integration with LangSmith for observability (one env var)
- Native MCP support
- Reducer logic for merging concurrent state updates (matters for parallel workers)

**What it costs:**

- Mental model overhead — graph concepts, state schemas, reducers
- Lock-in to LangChain ecosystem
- More dependencies; bigger Docker images
- Closed-source self-hosting (LangSmith Enterprise-only for self-host)

**The implementation:**

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add

class ResearchState(TypedDict):
    question: str
    plan: dict
    findings: Annotated[list[WorkerFinding], add]  # reducer for parallel workers
    critique: dict | None
    report: str | None

def plan_node(state: ResearchState):
    plan = make_plan(state["question"])
    return {"plan": plan}

def worker_node(state: ResearchState, sub_q: str):
    finding = research_worker(sub_q)
    return {"findings": [finding]}  # gets merged via reducer

def critic_node(state: ResearchState):
    critique = critique_findings(state["question"], state["findings"])
    return {"critique": critique}

def writer_node(state: ResearchState):
    report = write_report(state["question"], state["findings"], state["critique"])
    return {"report": report}

graph = StateGraph(ResearchState)
graph.add_node("plan", plan_node)
graph.add_node("dispatch_workers", dispatch_workers_dynamic)
graph.add_node("critic", critic_node)
graph.add_node("writer", writer_node)

graph.set_entry_point("plan")
graph.add_edge("plan", "dispatch_workers")
graph.add_edge("dispatch_workers", "critic")
graph.add_edge("critic", "writer")
graph.add_edge("writer", END)

agent = graph.compile(checkpointer=PostgresSaver(...))
```

The graph is more explicit. State management is structured. Checkpointing is one line.

We walk through:
- Dynamic worker dispatch (LangGraph supports this via `Send` for fan-out)
- The reducer pattern for collecting parallel worker results
- LangGraph Studio screenshots showing breakpoint debugging
- The LangSmith trace for the same question we ran in Section 3

**When LangGraph is the right choice:**
- You want time-travel debugging (genuinely valuable for complex agents)
- You're already in LangChain (free integration win)
- You need first-class durable checkpointing
- You're building for a regulated industry (audit trail of every state transition)

**When raw SDK is the right choice:**
- Minimum dependencies
- The graph is simple enough that explicit nodes are overhead
- You want full control over the execution model
- You're shipping a library, not an application (smaller attack/dependency surface)

::: brain
LangGraph's biggest selling point is checkpointing — pause/resume/time-travel. For our research capstone, do we actually need it?

(For research that takes 2-5 minutes? Probably not. For agents that run for hours and need to survive process restarts? Critical. For our capstone the value of checkpointing is limited; we're using LangGraph here mainly to demonstrate the framework. For real production deployments, the question is: does your agent's lifetime exceed your process's lifetime? If yes, checkpointing is the answer; if no, you can skip the framework cost.)
:::

---

## Section 5 — Implementation 3: CrewAI (≈4 pages)

Same system, third time. CrewAI is the role-based / fastest-prototype framework. We build it; we discuss its trade-offs honestly.

**The implementation:**

```python
from crewai import Agent, Task, Crew, Process

orchestrator = Agent(
    role="Research Orchestrator",
    goal="Plan and synthesize comprehensive research on the user's question",
    backstory="An expert researcher who breaks complex questions into focused sub-questions",
    llm=opus_client,
)

worker_template = Agent(
    role="Research Worker",
    goal="Find evidence on a specific sub-question",
    backstory="A focused researcher who searches and reads sources to gather evidence",
    llm=sonnet_client,
    tools=[web_search_tool, web_fetch_tool],
)

critic = Agent(
    role="Research Critic",
    goal="Identify gaps and unsupported claims in research output",
    backstory="A rigorous reviewer who flags weaknesses",
    llm=opus_client,
)

writer = Agent(
    role="Report Writer",
    goal="Produce a clear, well-cited report",
    backstory="A skilled writer who synthesizes evidence into clear prose",
    llm=opus_client,
)

# Tasks define the work; CrewAI wires sequencing/parallelism via Process
plan_task = Task(description="Plan research strategy for: {question}", agent=orchestrator)
worker_tasks = [
    Task(description=f"Research: {sub_q}", agent=worker_template) for sub_q in sub_questions
]
critique_task = Task(description="Critique the gathered findings", agent=critic)
write_task = Task(description="Write the final report", agent=writer)

crew = Crew(
    agents=[orchestrator, worker_template, critic, writer],
    tasks=[plan_task, *worker_tasks, critique_task, write_task],
    process=Process.hierarchical,
    manager_llm=opus_client,
)

result = crew.kickoff(inputs={"question": "..."})
```

**What CrewAI does well:**
- Fastest prototyping in the field (working multi-agent system in <50 lines)
- Role-based abstraction is intuitive — non-engineers can read it
- Strong defaults; opinionated in helpful ways
- Growing ecosystem

**What CrewAI doesn't do as well:**
- Limited control over agent-to-agent communication (mediated through task outputs, not direct messaging)
- No built-in checkpointing for long-running workflows
- Coarse-grained error handling — failures are harder to recover from gracefully
- Less production-tested for highly-custom flows (per 2026 community comparisons)

**When CrewAI is the right choice:**
- Prototype phase, optimizing for time-to-working
- Workflow is mostly linear or hierarchical
- Team is small / mixed technical
- You're willing to migrate later if production-scale issues emerge

**When it isn't:**
- Custom control flow with cycles or conditional branching
- Production-grade durability with checkpoints needed
- Complex agent communication patterns
- Compliance / audit-trail requirements

::: gotcha
A common pattern: teams prototype on CrewAI, get something working in a day, then hit production-scale issues (no checkpointing, opaque failures, hard-to-customize flow) and migrate to LangGraph or raw SDK. This is fine — it's the right way to use CrewAI. The mistake is committing to CrewAI in production for a complex system without a migration path planned.
:::

---

## Section 6 — Honest Comparison (≈4 pages)

Three implementations, same spec. We compare honestly.

**Lines of code (approximate; varies by style):**

| Implementation | Core LOC | With instrumentation | With eval harness |
|---|---|---|---|
| Raw SDK | ~600 | ~900 | ~1,200 |
| LangGraph | ~400 | ~600 | ~900 |
| CrewAI | ~150 | ~350 | ~600 |

**Cost on the same 10 research queries (illustrative — re-run when drafting):**

| Implementation | Avg cost/research | Variance |
|---|---|---|
| Raw SDK | $1.83 | ±$0.42 |
| LangGraph | $1.91 | ±$0.45 |
| CrewAI | $2.27 | ±$0.78 |

The frameworks add some overhead (mostly framework-level prompts and orchestration calls); CrewAI's hierarchical process spawns extra coordination calls that show up in cost.

**Wall-clock latency:**

Comparable across all three; the wall-clock is dominated by the LLM calls and tool calls, not framework overhead. Differences are <5%.

**Quality on a 50-question eval set:**

Comparable. Same prompts in different framework wrappings produce comparable outputs. Quality differences come from the prompts and the routing, not the framework.

**Maintainability over 6 months:**

This is harder to measure quantitatively but matters most. The honest assessment:
- **Raw SDK**: easiest to debug; hardest to extend. Adding new agent types means writing new code; not generalizing existing patterns.
- **LangGraph**: best for evolving systems; the graph abstraction maps naturally to growing complexity. Steepest learning curve.
- **CrewAI**: easiest to evolve in shape (add agents, change tasks); but customizing existing agents' behavior beyond CrewAI's abstractions is awkward.

**The decision matrix:**

| If you... | Pick... |
|---|---|
| Want minimum dependencies and full control | Raw SDK |
| Are already on LangChain or need durable checkpointing or LangGraph Studio | LangGraph |
| Are prototyping and want time-to-working above all | CrewAI |
| Have multiple cycles, branching logic, regulated industry | LangGraph |
| Are shipping a library (not application) | Raw SDK |
| Have a team of mixed technical levels | CrewAI |
| Need to migrate later | Build clean abstractions on raw SDK or LangGraph |

::: pullquote
Frameworks are tools. Tools have shapes. Pick the tool whose shape fits your problem; don't try to fit your problem to a tool. The right framework for your next agent is whatever you can ship and operate confidently.
:::

---

## Section 7 — Production Concerns Across All Three (≈4 pages)

Things that have to be true regardless of which framework. Each implementation in this capstone has them; we cover what to verify and how.

**1. OpenTelemetry instrumentation.** Verify trace tree mirrors agent structure. Each LLM call → span. Each tool call → span. Trace context propagates across sub-agent boundaries.

**2. Cost ceiling.** Verify the per-research budget halts execution when exceeded. Test by setting budget = $0.50 and running — should halt cleanly with partial results.

**3. Loop detection.** Verify agents that get stuck (a tool that always fails) don't retry forever. Test with a deliberately-broken tool.

**4. Guardrails.** Verify input topic classifier rejects out-of-scope queries. Test with off-topic queries and adversarial prompts. Verify output PII detection blocks any leaked-looking PII.

**5. Hallucination defense.** Verify the citation checker catches claims without supporting evidence in workers' findings. Test by injecting a "fact" into the report writer's prompt that no worker found — citation check should flag it.

**6. Eval harness.** Verify production traces flow into the eval queue at 1% sampling + 100% on errors / negative feedback. Verify trajectory eval runs against the queue and produces actionable scores.

**7. Failure handling.** Verify that worker failures don't take down the whole research. Test by killing one worker mid-run — should continue with N-1 workers, surface the failure in the final report.

**Pre-launch checklist:**

- [ ] Trace tree readable for sample queries
- [ ] Cost ceiling tested and verified
- [ ] Loop detection tested with broken tool
- [ ] Guardrails tested with attack suite
- [ ] Hallucination defense tested with injected false claims
- [ ] Eval harness producing scores on sample data
- [ ] Worker failure tested
- [ ] Documentation: what the agent does, what its limits are, when to escalate to a human
- [ ] On-call runbook: what to do when alerts fire
- [ ] Rollback plan: how to revert to previous version

This is the same checklist that distinguishes "agent that works in demos" from "agent in production" from Module 17 — applied concretely.

::: code-exercise
**Exercise 18.2 — Ship the capstone (your version).**

Take whichever implementation you prefer. Customize it for a research domain you actually care about (your work, an industry you're learning, an academic field). Run it on 50 real questions. Iterate on prompts, routing, depth based on observed traces.

Then: pick one production concern from the checklist above and add it if it's missing. Most likely candidates: better drift monitoring, additional guardrail layers, expanded eval harness.

Deploy somewhere — even just locally with a CLI — that you can use it weekly for actual research. The capstone earns its name when it becomes a thing you use, not a thing you built.
:::

---

## Section 8 — What You Now Know (≈3 pages)

A retrospective. The reader has shipped a multi-agent research system that uses every concept in the book. What they now know:

**The conceptual stack** (Modules 1-3): workflows vs agents, the agent loop, when to reach for which.

**The building blocks** (Modules 4-9): tools that don't lie, MCP that scales, model routing for economics, context engineering, compression, memory.

**The reliability layer** (Modules 10-12): hallucinations and grounding, reflection, guardrails.

**The multi-agent layer** (Modules 13-15): when multi-agent earns its cost, the architectures that survived production, the swarms that mostly didn't.

**The transactional layer** (Module 16): when agents touch real-world state, sagas and compensations.

**The production layer** (Module 17): evals, observability, cost, failure.

**A working artifact** (Module 18): the research system itself, built three ways, ready to extend.

**What remains hard.** Honest list. Even after this book, the reader will still face:

- Eval design for fuzzy domains (creative tasks, judgment calls, cross-cultural sensitivity)
- The 1-2% of agent failures that resist all the defense layers
- The framework migration that comes when prototype-CrewAI hits production-scale issues
- The cost-quality trade-offs that don't have right answers, only tier choices
- The drift that happens when models update under you
- The new failure modes that emerge as the field evolves

**The honest framing:** this book gave you 80% of what you need. The other 20% is judgment that comes from shipping, breaking, fixing, shipping again. The book's job is to compress your learning curve, not eliminate it.

---

## Section 9 — Where to Go From Here (≈2 pages)

Pointers for the reader continuing the journey.

**Stay current on these specific things:**
- OpenTelemetry GenAI semantic conventions (still evolving as of 2026)
- The eval landscape (LLM-as-judge limitations are still being mapped)
- Multi-agent coordination protocols (A2A, MCP — the standards game)
- Production-survival case studies (what's worked, what hasn't, what changed)
- Anthropic / OpenAI / Google's published agent engineering posts (the sources of the actual numbers in this book)

**Build the next thing.** The capstone is research-domain. Apply the same patterns to:
- A coding agent for a narrow domain (a specific codebase, a specific stack)
- An ops/SRE agent (instrumented carefully, sandboxed heavily)
- A customer-facing support agent (with the full guardrail stack from Module 12)
- A data analysis agent (Pandas + viz tools)
- A document-processing pipeline (PDF, Word, structured extraction)

**Communities worth being in:**
- LangChain / LangGraph forums for ecosystem updates
- Anthropic Discord / forums for Claude-specific patterns
- The agent observability platforms' communities (Langfuse, Laminar) — practical operational discussions
- Papers Twitter / arXiv-sanity for research drift

**The honest goodbye.** You've finished the book. You can build the agents you set out to build. The field will move; the patterns in this book will become outdated in some specifics over the next 18 months. That's fine — the *judgment* you've built doesn't expire. Patterns shift; the discipline of asking "what's the actual failure mode?" and "what's this earning?" doesn't.

Go ship something. The agents you build will be better than the agents the field built last year, because you started with the lessons of last year. That's the whole point.

::: pullquote
The book ends. The work begins. You'll build agents better than the ones in this book — because the next year's lessons will be yours, not Sam's.
:::

---

## Recurring elements used in this module

- 1 chapter opener
- 1 hook
- 0 No Dumb Questions (capstone is integrative; questions live in earlier modules)
- 2 Brain Power prompts
- 0 Production Postmortems (the capstone IS the postmortem of the book)
- 1 Watch It! / Gotcha block
- 2 Code Exercises (run the capstone, ship your version)
- 2 Pullquotes
- 1 architecture diagram (the system)
- 1 framework decision matrix
- 1 Bullet Points recap

::: sam-arc
Sam ships the capstone. Three implementations live in three branches. The team picks LangGraph (already on LangChain elsewhere; LangSmith integration is one env var; checkpointing matters for the long-running deep-research tier). The system serves real customers. Sam keeps the raw-SDK version in the repo as the eval-comparison baseline. CrewAI version becomes the prototype platform for future agent ideas.

A year later, Sam reads this book again. Most of it is still right. Some of the platforms have shifted. Anthropic's published numbers got refined. PreFlect-style prospective reflection became standard. SagaLLM-style validation became table stakes. What didn't change: the discipline of asking what each pattern earns, of reading traces honestly, of measuring before optimizing, of refusing to add agents without evidence.

Sam closes the book. Goes to ship the next thing.

**Sam's arc, end of book: from frazzled staff engineer to systems thinker who builds working agents and operates them with honesty. That's the arc.**
:::

::: page-budget
S1 (Spec): 3p
S2 (Architecture decisions): 4p
S3 (Raw SDK implementation): 8p
S4 (LangGraph implementation): 6p
S5 (CrewAI implementation): 4p
S6 (Honest comparison): 4p
S7 (Production concerns): 4p
S8 (What you now know): 3p
S9 (Where to go from here): 2p
Recurring elements: 4p
TOTAL: ~42 pages
:::

::: sources
**Must verify when drafting:**

- Anthropic SDK current API and Python package versions
- LangGraph current API (StateGraph, Send for dynamic dispatch, checkpointers)
- LangSmith current pricing and self-host story
- CrewAI current API (Process types, hierarchical mode, manager_llm pattern)
- LangGraph Studio capabilities (the "best agent IDE" claim — verify scope)
- Langfuse self-host requirements (Postgres + ClickHouse)
- Real benchmark numbers — re-run the three implementations on the same 10-50 question test set with current models
- Anthropic model identifiers and pricing as of drafting
- AutoGen status (verify "maintenance mode" framing — if it's changed by drafting, update)
- Microsoft Agent Framework 1.0 GA (per articles, April 2026) — mention as alternative if relevant
- A2A protocol current state
- OpenAI Agents SDK current state (post-Swarm replacement)
- Gartner agentic AI cancellation prediction — verify the 40% / 2027 numbers

**Stable knowledge:**
- The architectural decisions section (the reasoning is durable even if specific framework versions change)
- The pre-launch checklist
- The decision matrix structure (specific frameworks may shift but the criteria are durable)
- The retrospective on the book's structure

**Cross-references:** essentially every prior module. We list:
- Module 1-3 (workflows, agent loop, patterns)
- Module 4-5 (tools, MCP)
- Module 6 (model routing)
- Module 7-8 (context, compression)
- Module 9 (memory)
- Module 10-12 (reliability)
- Module 13-15 (multi-agent)
- Module 16 (sagas)
- Module 17 (production)
:::

::: bullet-points
### Module 18 in eight bullets

(filled at draft time)

- The capstone is a deep-research multi-agent system: orchestrator + parallel workers + critic + writer
- We build it three times: raw SDK, LangGraph, CrewAI — same spec, three implementations
- Architecture decisions apply Module 13's threshold test, Module 14's pattern selection, Module 17's instrumentation
- Raw SDK: minimum dependencies, full control, easiest debug, hardest extend
- LangGraph: best for evolving production systems, durable checkpointing, LangSmith integration, steepest learning curve
- CrewAI: fastest prototyping, intuitive role-based, limits at custom complexity, expect migration eventually
- Quality is comparable across all three — frameworks differ on dev experience and operational concerns, not output quality
- Production concerns (instrumentation, ceilings, guardrails, eval harness) are framework-independent — every implementation needs them
:::
