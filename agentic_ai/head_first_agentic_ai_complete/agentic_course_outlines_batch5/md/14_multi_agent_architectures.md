# Module 14 Outline — Multi-Agent Architectures, Coded

::: chapter-opener
<div class="module-num">MODULE 14 — OUTLINE</div>
<div class="module-title">Multi-Agent Architectures, Coded</div>
<div class="subtitle">Five patterns. Same task four ways. Real benchmarks.<br>Plus the two patterns that "won" production in 2026 and the three that didn't.</div>
<div class="pages">Target length: ~38 pages (heavy code)</div>
:::

## What this module is

Module 13 established when multi-agent earns its cost. This module is the implementation: five canonical architectures, coded in raw Anthropic SDK plus LangGraph where it earns its place. We benchmark four of them on the same long-horizon research task to make the trade-offs concrete.

The patterns: **supervisor/orchestrator** (Anthropic's research system shape — the one that survived production), **hierarchical** (multi-level supervisors), **network/peer-to-peer** (free mesh — mostly didn't survive), **sequential pipeline** (handoffs between specialists), **debate/critique/judge** (adversarial cycles). Communication patterns: shared state vs message passing. Failure handling: agent failures, partial results, fallback strategies.

By the end the reader can: implement any of the five patterns, pick which one fits their task, instrument multi-agent for observability, and avoid the two architecture mistakes that account for most multi-agent disasters.

::: hook
"There are five canonical multi-agent architectures. Two of them survived production in 2026. Two of them survived as research demos. One of them ('group chat') survives mostly in conference talks and venture pitches. Let's build all five so you can pick from a position of knowledge — not aesthetics."
:::

---

## Section 1 — The Five Patterns at a Glance (≈3 pages)

A side-by-side. Each gets the deep treatment in its own section.

| Pattern | Coordination | Best For | Production Status (2026) |
|---|---|---|---|
| **Supervisor / Orchestrator-Worker** | Central planner dispatches to peers | Parallelizable exploration, research | ✅ Survived — dominant pattern |
| **Hierarchical** | Multiple supervisor levels | Very large systems, organizational mirroring | ✅ Survived — for >10-agent systems |
| **Sequential Pipeline (Handoffs)** | Linear chain, one-to-next | Domain routing, specialist relay | ✅ Survived — for routing-heavy domains |
| **Network / Peer-to-Peer (Group Chat)** | Free mesh, agents talk to each other | Brainstorming, debate, exploration | ⚠️ Mostly didn't survive — usually wrapped in a supervisor |
| **Debate / Critique / Judge** | Adversarial — one produces, others critique | High-stakes outputs, quality matters > speed | ✅ Survived — as a sub-pattern, not whole architecture |

The April 2026 "What Actually Survived in Production" article frames it cleanly: **graph (i.e. supervisor with explicit transitions) and hierarchy are the two patterns that earn their cost in production. Free mesh and pure swarms are theater outside controlled subroutines.**

We'll hold to that framing — without dismissing the other patterns, but being honest about where each lives in 2026.

::: pullquote
The patterns that ship are the ones with explicit control flow. The patterns that demo well are the ones with emergent behavior. These are not the same set.
:::

::: nodumbq
**Q: If supervisor and hierarchical are the survivors, why teach the others?**

Three reasons. (1) The "non-survivors" survive as *sub-patterns inside* survivors — group chat works fine as a brainstorming step inside a supervisor; debate cycles work fine as a critique step. (2) Different tasks fit different patterns; "what survived in general" doesn't mean "what's best for your specific case." (3) Knowing why the failed patterns failed is essential to recognizing when you're building one accidentally.
:::

---

## Section 2 — Pattern 1: Supervisor / Orchestrator-Worker (≈6 pages)

Anthropic's research system shape. The dominant production pattern. We code it twice — once in raw SDK, once in LangGraph — and discuss when to use which.

**The architecture:**

```
                    ┌──────────────┐
                    │  SUPERVISOR  │  ← Opus-class
                    └──────┬───────┘
                           │ plans, dispatches
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
     ┌──────────┐   ┌──────────┐   ┌──────────┐
     │ Worker 1 │   │ Worker 2 │   │ Worker 3 │  ← Sonnet-class, parallel
     └────┬─────┘   └────┬─────┘   └────┬─────┘
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                    ┌──────────────┐
                    │  SYNTHESIZER │  ← could be supervisor or separate
                    └──────────────┘
```

**Raw SDK implementation:**

```python
from anthropic import AsyncAnthropic

class SupervisorAgent:
    def __init__(
        self,
        client: AsyncAnthropic,
        worker_factory,  # callable(task_spec) -> WorkerAgent
        supervisor_model: str = "claude-opus-4-7",
        synth_model: str = "claude-opus-4-7",
        max_workers: int = 5,
    ):
        self.client = client
        self.worker_factory = worker_factory
        self.supervisor_model = supervisor_model
        self.synth_model = synth_model
        self.max_workers = max_workers

    async def run(self, query: str) -> AgentResult:
        # 1. Plan: supervisor decomposes
        plan = await self._plan(query)
        if len(plan.tasks) > self.max_workers:
            plan.tasks = plan.tasks[:self.max_workers]
        
        # 2. Dispatch in parallel
        worker_tasks = [
            self.worker_factory(task_spec).run(task_spec.prompt)
            for task_spec in plan.tasks
        ]
        worker_results = await asyncio.gather(*worker_tasks, return_exceptions=True)
        
        # 3. Handle partial failures
        successes = [r for r in worker_results if not isinstance(r, Exception)]
        failures = [(t, r) for t, r in zip(plan.tasks, worker_results) if isinstance(r, Exception)]
        
        # 4. Synthesize
        synthesis = await self._synthesize(query, plan, successes, failures)
        return synthesis

    async def _plan(self, query: str) -> Plan:
        resp = await self.client.messages.create(
            model=self.supervisor_model,
            max_tokens=2000,
            tools=[PLAN_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "create_plan"},
            messages=[{"role": "user", "content": SUPERVISOR_PLAN_PROMPT.format(query=query)}],
        )
        return parse_plan(resp)

    async def _synthesize(self, query, plan, successes, failures) -> AgentResult:
        resp = await self.client.messages.create(
            model=self.synth_model,
            max_tokens=4000,
            messages=[{"role": "user", "content": SYNTHESIS_PROMPT.format(
                query=query,
                successes=format_successes(successes),
                failures=format_failures(failures),
            )}],
        )
        return AgentResult(...)
```

We walk through every part. The plan tool's schema. The synthesis prompt's structure. How worker failures get represented to the synthesizer (don't hide them — let the synthesizer decide how to handle them).

**LangGraph implementation:**

```python
from langgraph.graph import StateGraph, END

class SupervisorState(TypedDict):
    query: str
    plan: Plan | None
    worker_outputs: dict[str, str]
    final_answer: str | None

def supervisor_node(state: SupervisorState) -> dict:
    plan = make_plan(state["query"])
    return {"plan": plan}

def worker_node(state: SupervisorState, worker_id: str) -> dict:
    task = state["plan"].tasks_by_id[worker_id]
    output = run_worker(task)
    return {"worker_outputs": {worker_id: output}}

def synthesizer_node(state: SupervisorState) -> dict:
    answer = synthesize(state["query"], state["plan"], state["worker_outputs"])
    return {"final_answer": answer}

graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("synth", synthesizer_node)
# Workers added dynamically based on plan
graph.add_edge("supervisor", "fan_out_workers")  # custom routing
graph.add_edge("fan_out_workers", "synth")
graph.add_edge("synth", END)
```

LangGraph earns its place when:
- You want explicit graph visualization for debugging
- You need durable state with checkpointing (resume after crash)
- You want time-travel debugging
- You're already in the LangChain ecosystem

Raw SDK earns its place when:
- You want minimal dependencies
- The graph is simple enough not to benefit from explicit nodes
- You want full control over the execution model

For this pattern, both are reasonable. We don't take a side; we show both.

::: code-exercise
**Exercise 14.1 — Supervisor on a research task.**

Build the supervisor pattern end-to-end (raw SDK version). Task: "Compare the carbon footprints of three given products and recommend the lowest." Your supervisor should:
- Decompose into 3 product-research workers + 1 comparison worker
- Run product workers in parallel
- Pass results to comparison worker
- Synthesize final recommendation

Measure: total tokens, total wall-clock, quality (vs a single-agent baseline answer).
:::

---

## Section 3 — Pattern 2: Hierarchical (≈4 pages)

Multi-level supervisors. Used when systems exceed ~5-10 agents and a flat supervisor becomes a bottleneck.

```
                  ┌──────────────┐
                  │  CHIEF       │  
                  └──────┬───────┘
                         │
              ┌──────────┼──────────┐
              ▼                     ▼
       ┌──────────┐          ┌──────────┐
       │ Manager A│          │ Manager B│
       └────┬─────┘          └────┬─────┘
            │                     │
       ┌────┼────┐           ┌────┼────┐
       ▼    ▼    ▼           ▼    ▼    ▼
      W1   W2   W3          W4   W5   W6
```

When this pattern earns its keep:
- Tasks naturally have multi-level decomposition (plan team → research teams → individual researchers)
- Single supervisor becomes a planning bottleneck or context overflow
- Different sub-systems need different supervisor expertise

When it doesn't:
- < 10 total agents — flat supervisor handles it
- Task doesn't naturally hierarchical
- Communication between branches is high (cross-cutting concerns kill the hierarchy advantage)

**Code sketch (we'll show the relevant differences from supervisor pattern, not full duplication):**

```python
class ChiefAgent(SupervisorAgent):
    """A supervisor whose 'workers' are themselves supervisor agents."""
    
    async def run(self, query: str) -> AgentResult:
        plan = await self._plan_at_chief_level(query)
        # Each "task" in the chief's plan is delegated to a manager (which is itself a supervisor)
        manager_tasks = [
            ManagerAgent(...).run(sub_query) for sub_query in plan.tasks
        ]
        results = await asyncio.gather(*manager_tasks)
        return await self._synthesize(query, results)
```

::: gotcha
Hierarchical patterns multiply the per-level translation loss. The chief gets a summary from each manager, who got summaries from each worker. By the time information reaches the chief, it's been compressed twice. For information-dense tasks, this is where hierarchical loses to flat-supervisor + better tools.
:::

::: brain
You have 12 sub-agents that need to coordinate. Flat supervisor (1 supervisor + 12 workers) or hierarchical (1 chief + 3 managers + 4 workers each)?

(Depends on whether the 12 workers can be naturally grouped. If yes — say, 3 categories of 4 workers each — hierarchical reduces the supervisor's planning load and lets each manager specialize. If no — 12 unrelated tasks — flat is simpler, and the supervisor handles each independently.)
:::

---

## Section 4 — Pattern 3: Sequential Pipeline / Handoff (≈4 pages)

Agents in a chain. Each handles its part, passes to the next. OpenAI Swarm's "handoff" pattern is the canonical implementation: an agent's tool returns another agent object, and control transfers.

```
[User Query] → [Triage Agent] → [Specialist A] → [Specialist B] → [Output]
```

**OpenAI Swarm-style implementation (using Anthropic SDK pattern):**

```python
@dataclass
class HandoffResult:
    next_agent: "Agent | None"
    final_output: str | None

class HandoffAgent:
    def __init__(self, client, name: str, instructions: str, tools, handoff_targets: list["HandoffAgent"]):
        self.client = client
        self.name = name
        self.instructions = instructions
        self.tools = tools
        # Add a "transfer_to" tool for each possible handoff target
        for target in handoff_targets:
            self.tools.register(make_handoff_tool(target))
    
    async def step(self, messages: list[dict]) -> HandoffResult:
        resp = await self.client.messages.create(
            model="claude-sonnet-4-6",
            system=self.instructions,
            tools=self.tools.schemas(),
            messages=messages,
        )
        for block in resp.content:
            if block.type == "tool_use" and block.name.startswith("transfer_to_"):
                target_name = block.name.replace("transfer_to_", "")
                return HandoffResult(next_agent=lookup_agent(target_name), final_output=None)
        # No handoff; this is the final response
        return HandoffResult(next_agent=None, final_output=extract_text(resp))


async def run_pipeline(start: HandoffAgent, query: str) -> str:
    current = start
    messages = [{"role": "user", "content": query}]
    for _ in range(10):  # cap handoffs
        result = await current.step(messages)
        if result.final_output is not None:
            return result.final_output
        current = result.next_agent
        messages.append({"role": "user", "content": f"[transferred from {current.name}]"})
    raise PipelineExhausted("too many handoffs")
```

When this pattern earns its keep:
- Domain routing problems (customer support: triage → billing/tech/refund/escalation)
- Compliance boundaries (different agents have different data access)
- Sequential refinement where each step has clear input/output

When it doesn't:
- Tasks where agents need each other's intermediate state (handoff loses context)
- Loops (each handoff costs tokens; circular pipelines compound fast)
- Cases where parallel exploration is the win (use supervisor instead)

::: postmortem
**The Handoff Loop That Wouldn't End**

A team's customer support pipeline: triage → billing → escalation → triage. Designed for the case where billing couldn't help and the user really needed re-triage. Worked in tests.

In production, a single confused customer query bounced between billing and triage 23 times before the system gave up. Each handoff added ~$0.04. The customer got a "we're having trouble" message. Total cost: $0.92 for one unresolved query. Multiplied by the % of edge cases: real money.

Fix: a max-handoffs counter (capped at 3). When exceeded, escalate to human. Better: the triage agent's prompt was updated to recognize "this is a re-routed query" and not re-route to the same destination.

**Lesson:** handoff loops are not theoretical. Cap them. Always cap them.
:::

---

## Section 5 — Pattern 4: Network / Peer-to-Peer (Group Chat) (≈4 pages)

The pattern that mostly didn't survive. Multiple agents in a shared conversation, taking turns. AutoGen's GroupChat is the canonical implementation.

```
        ┌──── Agent A ────┐
        │                 │
        │   [shared       │
        │    conversation]│
        │                 │
        └──── Agent B ─── Agent C
```

The mechanics: a "manager" picks who speaks next based on the conversation state. Each agent sees the full conversation. Each turn is a full LLM call with the accumulated history.

**The fundamental problem:** every agent reads everyone else's contributions every turn. A 4-agent debate over 5 rounds is 20 LLM calls minimum, with each call's input growing with the conversation. Quality often doesn't justify the cost.

The April 2026 "What Survived in Production" piece: free mesh survived mostly as a controlled subroutine inside a supervisor, not as the outer architecture. Group chat is fine as a phase ("brainstorming step within a larger plan"); it's expensive as the whole system.

When it does work:
- Genuinely needs multiple perspectives that benefit from seeing each other's reasoning
- High-stakes / quality-over-speed (legal review, scientific peer review)
- Bounded scope (3-5 agents, capped rounds)

When it doesn't:
- Customer-facing latency-sensitive flows
- Anything where a supervisor pattern would be cheaper
- Anything where one strong agent + good tools would suffice

We code a simple group-chat for completeness:

```python
class GroupChatManager:
    def __init__(self, agents: list[Agent], max_rounds: int = 5):
        self.agents = {a.name: a for a in agents}
        self.max_rounds = max_rounds
    
    async def run(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]
        for round_i in range(self.max_rounds):
            speaker_name = await self._select_speaker(messages)
            speaker = self.agents[speaker_name]
            response = await speaker.respond(messages)
            messages.append({"role": "assistant", "content": f"[{speaker.name}] {response}"})
            if "<terminate>" in response.lower():
                break
        return await self._synthesize(messages)
```

::: gotcha
The biggest group-chat anti-pattern: not capping rounds. A productive debate at round 3 becomes a circular argument by round 8. Always cap; then use a synthesizer to extract the conclusion.
:::

---

## Section 6 — Pattern 5: Debate / Critique / Judge (≈4 pages)

The adversarial pattern. One agent produces; one or more agents critique; a judge decides. Generalizes Module 11's reflection to multi-agent.

The shape:

```
[Producer Agent] → produces output
        │
        ▼
┌────────────────┐
│ Critic Agent A │  ──┐
└────────────────┘    │
┌────────────────┐    ├──→ [Judge Agent] → final output (or revise)
│ Critic Agent B │  ──┤
└────────────────┘    │
┌────────────────┐    │
│ Critic Agent C │  ──┘
└────────────────┘
```

When this pattern earns its keep:
- Output quality matters more than speed (legal contracts, code review, scientific writing)
- The producer is capable but variable
- The critics catch genuinely different issues
- The judge can act on critique decisively

```python
class DebateSystem:
    def __init__(self, producer, critics: list[Agent], judge, max_rounds: int = 2):
        self.producer = producer
        self.critics = critics
        self.judge = judge
        self.max_rounds = max_rounds

    async def run(self, task: str) -> str:
        draft = await self.producer.produce(task)
        for round_i in range(self.max_rounds):
            critiques = await asyncio.gather(*[
                c.critique(task, draft) for c in self.critics
            ])
            decision = await self.judge.decide(task, draft, critiques)
            if decision.accept:
                return draft
            draft = await self.producer.revise(task, draft, decision.feedback)
        return draft
```

**Critical design notes:**

- Critics must be different from producer (model, prompt, or both). Same-model critique is rationalization.
- Judge should be the strongest model (Opus-class). Judge errors propagate.
- Each critic gets a focused angle (security, performance, style) rather than "general critique."
- Cap rounds. Quality gains saturate at 2-3.

::: brain
You're building a code review agent system. You could implement it as: (a) supervisor with parallel security/performance/style workers, or (b) producer/critic/judge with the writer being a single agent and three critics.

Which is which? Are these the same thing in disguise?

(Almost the same. Both fan out parallel evaluation by aspect. The difference: (a) starts with a plan and dispatches workers to evaluate; (b) starts with a draft and dispatches critics to attack it. (a) is supervisor pattern; (b) is debate pattern. For code review specifically, (b) is usually better because the artifact-being-reviewed is a natural anchor, but (a) works fine too.)
:::

---

## Section 7 — Communication: Shared State vs Message Passing (≈3 pages)

Across all patterns, two communication models:

**Shared state.** All agents read/write to a common store (memory, database, scratchpad). Coordination via state. Fast within a process; needs locks across processes.

```python
class SharedState:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._data = {}
    
    async def write(self, key: str, value: Any):
        async with self._lock:
            self._data[key] = value
    
    async def read(self, key: str) -> Any:
        return self._data.get(key)
```

**Message passing.** Agents send explicit messages. No shared mutable state. Each agent's view is what it received.

```python
class MessageBus:
    async def send(self, from_agent, to_agent, message): ...
    async def receive(self, agent) -> Message: ...
```

Trade-offs:
- Shared state: simpler within a process; race conditions and lock contention at scale
- Message passing: more architecturally clean; higher overhead per coordination

Most production systems use a mix: shared state for the artifact being worked on, message passing for control flow.

The "message" itself is critical. Free-form text loses information across handoffs (the "translation/paraphrase loss" the 2026 production-survival analysis names). Structured messages (JSON with explicit fields) preserve more. The format you pick is part of your architecture.

::: pullquote
The hardest-to-debug bugs in multi-agent systems are communication bugs. The structure of messages between agents matters more than the structure of agents themselves.
:::

---

## Section 8 — Same Task, Four Ways: A Benchmark (≈4 pages)

The payoff. We define a representative task — a research-and-recommend task that's plausibly multi-agent — and implement it four ways. Measure tokens, latency, quality.

**Task:** "Research and recommend the best Python web framework for a team building a real-time collaboration app, considering performance, ecosystem, learning curve, and our existing TypeScript expertise."

**Four implementations:**

1. **Single agent** (Module 2 baseline) with web_search and web_fetch tools.
2. **Supervisor / orchestrator-worker.** Supervisor decomposes into 4 research workers (one per criterion) + 1 synthesizer.
3. **Sequential pipeline.** Triage → Performance researcher → Ecosystem researcher → Recommender.
4. **Debate.** Producer drafts a recommendation; 3 critics attack from different angles; judge decides.

Sample expected results table (illustrative — re-run when drafting):

| Approach | Avg Tokens | Avg Cost | p95 Latency | Quality Score |
|---|---|---|---|---|
| Single agent | 28K | $0.18 | 22s | 6.8/10 |
| Supervisor | 412K | $2.78 | 47s | 8.4/10 |
| Sequential pipeline | 95K | $0.61 | 38s | 7.6/10 |
| Debate | 178K | $1.21 | 52s | 8.1/10 |

The headlines:
- Supervisor wins on quality (matches Anthropic's findings) — at 15× the cost.
- Pipeline is a moderate quality bump for moderate cost; the right answer for moderate-stakes domain routing.
- Debate gets near-supervisor quality at 60% the cost — for tasks where the producer-critic loop fits.
- Single agent is the right answer for low-stakes; everyone else is paying premiums for premium quality.

**The lesson once again:** quality and cost both move with architecture. The "best" is task-value-dependent. There's no architecture that wins on every dimension.

::: code-exercise
**Exercise 14.2 — Build all four; benchmark.**

Implement all four architectures on the same task. Run each on 10 representative queries. Generate the same table format. Discuss with your team: which architecture would you ship for which use case?

(For most teams, the answer is: pipeline for routine tier, supervisor for premium tier, debate for compliance-required outputs. Single agent for everything else.)
:::

---

## Section 9 — Failure Handling in Multi-Agent (≈3 pages)

What happens when a sub-agent fails. Three strategies:

**1. Fail fast.** First worker failure aborts the whole task. Simple; high cost on transient failures.

**2. Retry the failed worker.** Other workers continue; failed worker retried independently. Better resilience; slight coordination complexity.

**3. Synthesize partial results.** Failed workers' contributions are noted as missing; synthesizer works with what's available.

```python
async def run_workers_with_resilience(self, plan: Plan) -> WorkerResults:
    workers = [self.worker_factory(t) for t in plan.tasks]
    results = await asyncio.gather(
        *[w.run() for w in workers],
        return_exceptions=True,
    )
    successes = []
    failures = []
    for task, result in zip(plan.tasks, results):
        if isinstance(result, Exception):
            failures.append((task, result))
        else:
            successes.append((task, result))
    return WorkerResults(successes=successes, failures=failures)
```

The synthesizer prompt should know about failures:

```python
SYNTHESIS_WITH_FAILURES = """Original task: {query}

Successful sub-results:
{successes}

Failed sub-tasks (could not be completed):
{failures}

Produce the best answer you can with the information available. Note explicitly
where information is missing due to failed sub-tasks; do not fill gaps with speculation.
"""
```

This is the right approach more often than people think. A research task with 4 successful workers and 1 failure usually still gives a useful answer with the missing information called out — better than blocking the whole task or spending another retry budget on a worker that may fail again.

---

## Section 10 — What's Next (≈1 page)

Module 15 covers swarms, stigmergy, and OpenAI Swarm-style handoff systems. Most of what's marketed as "swarm" is actually patterns covered in this module under different names; we'll separate what's real (handoff mechanics, stigmergy as a pattern) from what's theater.

Module 17 (production) covers the observability, eval, and operational concerns multi-agent makes urgent.

Module 18 (capstone) builds a real multi-agent research system applying everything from Modules 13-14.

---

## Recurring elements used in this module

- 1 chapter opener
- 1 hook
- 1 No Dumb Questions
- 2 Brain Power prompts
- 1 Production Postmortem (handoff loop)
- 2 Watch It! / Gotcha blocks
- 2 Code Exercises (supervisor on research, four-way benchmark)
- 2 Pullquotes
- 1 patterns-at-a-glance table
- 1 benchmark results table
- 5 architecture diagrams (one per pattern)
- 1 Bullet Points recap

::: sam-arc
Sam, having decided in Module 13 that the company's premium tier justifies multi-agent, comes to this module to pick the architecture. First instinct: group chat (it sounded most "agentic" in the demo). After reading Section 5, drops it. Builds supervisor pattern (Section 2). Then realizes the company's compliance review use case needs debate pattern (Section 6). Ships supervisor for premium research; debate for compliance reviews; pipeline for the existing customer support routing. Sam's arc this module: **architecture follows task, not aesthetic. Three patterns coexist in production because they each fit different jobs.**
:::

::: page-budget
S1 (Patterns at a glance): 3p
S2 (Supervisor): 6p
S3 (Hierarchical): 4p
S4 (Sequential pipeline): 4p
S5 (Network/group chat): 4p
S6 (Debate/critique/judge): 4p
S7 (Communication): 3p
S8 (Four-way benchmark): 4p
S9 (Failure handling): 3p
S10 (Next): 1p
Recurring elements: 2p
TOTAL: ~38 pages
:::

::: sources
**Must verify when drafting:**

- Anthropic's research system architecture (June 2025 post) — the canonical supervisor pattern
- Anthropic Sub-Agents API — model field options, mechanics
- LangGraph current API for StateGraph, supervisor patterns, subgraphs
- OpenAI Swarm framework (Oct 2024 release) — handoff mechanics, routine concept, current status
- AutoGen GroupChat — current state (article noted main repo is "maintenance-only" as of 2026 — verify)
- "Multi-Agent in Production in 2026: What Actually Survived" (Lanham, Apr 2026) — survival analysis framing
- Digital Applied Q2 2026 patterns taxonomy
- CrewAI current API — for the Section 7 anti-pattern reference
- Comet/Opik observability for multi-agent traces (Section 9 cross-reference forward to M17)
- Anthropic API current pricing for benchmark cost calculations
- A2A protocol (Google ADK) — mention as context, but don't deep-dive
- Real benchmark results — re-run the four-way comparison with current models when drafting

**Stable knowledge:**
- The five architecture patterns and their structural diagrams
- Communication models (shared state vs message passing)
- Failure handling strategies (fail fast / retry / partial synthesis)
- Trade-offs between patterns

**Cross-references:**
- Module 13 — when multi-agent is justified at all
- Module 11 — debate pattern is multi-agent reflection
- Module 15 — swarms (handoff in network form)
- Module 17 — observability
- Module 18 — capstone applies these
:::

::: bullet-points
### Module 14 in eight bullets

(filled at draft time)

- Five canonical patterns: supervisor, hierarchical, sequential pipeline, network/group chat, debate/critique
- Supervisor and hierarchical are the survivors of 2026 production reality
- Sequential pipeline (handoffs) survives for routing-heavy domains; cap handoffs to prevent loops
- Network / group chat mostly survives only as a controlled subroutine, not whole architecture
- Debate / critique / judge pattern earns its cost on quality-over-speed tasks
- Communication structure (shared state vs message passing, free-form vs structured messages) often matters more than agent structure
- Same task, four architectures, four different cost-quality profiles — pick by task value
- Failure handling: synthesize partial results is usually better than fail-fast or aggressive retry
:::
