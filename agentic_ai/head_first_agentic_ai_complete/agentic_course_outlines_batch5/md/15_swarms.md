# Module 15 Outline — Swarms and Emergent Behavior

::: chapter-opener
<div class="module-num">MODULE 15 — OUTLINE</div>
<div class="module-title">Swarms and Emergent Behavior</div>
<div class="subtitle">Most "swarms" are sequential pipelines with marketing.<br>Real swarm patterns exist — and have a tiny set of legitimate uses.</div>
<div class="pages">Target length: ~36 pages</div>
:::

## What this module is

Swarms are the most overhyped corner of multi-agent. The term evokes biological elegance — emergent intelligence from simple agents — but most production "swarm" systems are just the sequential pipelines from Module 14 with extra branding. This module separates the real patterns from the theater.

We cover OpenAI Swarm's actual mechanics (handoffs as control transfer, routines as structured prompts) and explain why it was explicitly released as a "reference implementation" not a production library. We code stigmergy patterns (agents leaving traces in shared environment) and explain when they're useful. We're honest about emergence: in agent systems, "emergent" is almost always the orchestrator doing visible work that observers project as emergent.

By the end the reader can: recognize when a system being marketed as "swarm" is actually one of Module 14's patterns, decide if their use case is one of the small set where genuine swarm coordination earns its cost, and implement the legitimate patterns (stigmergy through shared scratchpads, market/auction patterns for self-organizing) without the cargo-cult.

::: hook
"You'll see 'agent swarms' in conference talks. The architecture diagram has dozens of agents, arrows pointing everywhere, the word 'emergent' in italics. Then you read the code and it's a `for` loop calling agents in sequence. The 'swarm' is six prompts and a manager. Knowing what swarms actually are makes you immune to a category of pitch."
:::

---

## Section 1 — What "Swarm" Actually Means (≈4 pages)

The biological metaphor. A single ant follows pheromone trails and basic rules. A colony builds nests, farms fungus, wages war, adapts to disasters. The colony exhibits intelligence the individual ants don't. This is **stigmergy** — coordination via traces left in the environment.

The promise of agent swarms: many simple agents, no central coordinator, coordination emerges from environment-mediated interaction. Anyone selling this is selling a vision that's mostly aspirational.

**What gets called "swarm" in the agent world:**

1. **OpenAI Swarm framework** (Oct 2024). Lightweight handoff system. Despite the name, it's a *handoff pattern* — Module 14 Section 4. OpenAI explicitly released it as "less of a production library and more of a reference implementation."

2. **AutoGen Group Chat patterns.** Multiple agents in shared conversation. Already covered in Module 14 Section 5 as network/group chat.

3. **Stigmergy patterns** — agents writing to a shared scratchpad/blackboard, others reading and acting. This is the closest to the biological analogy and the one we'll deep-dive in this module.

4. **Market / auction patterns** — agents bid for tasks; tasks go to the highest bidder (or best-fit). Used in some research; rare in production.

5. **Decentralized peer networks** — agents communicate directly without orchestrator. Mostly research; production systems revert to having an orchestrator within months.

The 2026 production-survival analysis (Lanham, April 2026) was direct: "Swarm only for exploratory or research-mode systems where flow can't be pre-specified. Default to graph or hierarchy in production."

::: pullquote
The metaphor is biological; the implementations are usually not. When someone says "swarm," ask "where's the stigmergy?" If they can't show you a shared environment that mediates coordination, they mean "many agents," not "swarm."
:::

::: nodumbq
**Q: Is multi-agent the same as swarm?**

No. Multi-agent (Module 14) is about multiple coordinating agents — usually with a coordinator. Swarm specifically refers to systems where coordination is decentralized and often emergent. All swarms are multi-agent; not all multi-agent systems are swarms. Most production multi-agent systems are explicitly *not* swarms — they have a clear orchestrator.

**Q: If swarms are mostly hype, why have a whole module on them?**

Three reasons. (1) The hype is loud enough that you'll be asked about it — you should be able to talk about it accurately. (2) Some patterns within the swarm umbrella (stigmergy, market patterns) have legitimate uses you should know how to implement. (3) Recognizing what's not actually a swarm is the cheapest way to avoid building one accidentally.
:::

---

## Section 2 — OpenAI Swarm: Mechanics, Honestly (≈5 pages)

The framework named "Swarm" by OpenAI, deserves its own treatment — partly because of the name confusion, partly because the design choices are instructive.

OpenAI Swarm is built on two primitives:

**Routines.** Natural-language instructions paired with a list of tools. Maps to: a system prompt + functions. The model reads the routine, decides which functions to call, follows the steps.

**Handoffs.** A function that returns an `Agent` object instead of a string. Calling it transfers control to the returned agent.

```python
# Conceptual example — Swarm-style implementation in Anthropic SDK
@dataclass
class Agent:
    name: str
    instructions: str
    tools: list[Callable]

triage_agent = Agent(
    name="triage",
    instructions="Determine which specialist should handle this request.",
    tools=[transfer_to_billing, transfer_to_technical, transfer_to_refund],
)

billing_agent = Agent(
    name="billing",
    instructions="Handle billing questions. Escalate if you can't.",
    tools=[lookup_invoice, transfer_to_escalation],
)

def transfer_to_billing() -> Agent:
    return billing_agent

def transfer_to_technical() -> Agent:
    return technical_agent
```

Notice: this isn't a swarm in the biological sense. It's a state machine where each agent picks the next state. It's a sequential pipeline with branching. Module 14 Section 4 covered this.

**Why OpenAI named it "Swarm":** marketing aside, the framework supports lightweight, stateless multi-agent coordination without explicit graph definitions. The "swarm" framing emphasizes that the structure emerges from agents' handoff decisions rather than being pre-declared. This is a real design choice — but it's not stigmergy and it's not emergent intelligence.

**Why it's "experimental and not for production":** OpenAI's own framing. The framework lacks production essentials: durable state, observability, retry semantics, error handling primitives. It's a clean reference for the handoff pattern; it's not what you ship.

**What replaced it for production:** the OpenAI Agents SDK (released later) is the production-grade version, with clearer separation between agents-as-tools (composable) and handoffs (sequential).

::: gotcha
The Swarm framework's main repo is now in maintenance mode. Production deployments built on Swarm have largely migrated to Agents SDK or rebuilt the handoff pattern in their own code. If you're starting today, don't start with Swarm — implement the handoff pattern yourself or use a maintained framework.
:::

---

## Section 3 — Stigmergy: The Real Swarm Pattern (≈5 pages)

If OpenAI Swarm isn't really a swarm, what is?

**Stigmergy** is the biological pattern: agents leave traces in the environment; other agents read those traces and act. Termites build mounds without a blueprint or a leader. Ants build paths to food without communication beyond pheromone deposits. The coordination is *indirect*, mediated by environmental state.

In agent systems, stigmergy looks like:

```
[Agent A] writes to → [Shared Scratchpad / Blackboard / Vector Memory]
[Agent B] reads from / acts on traces / writes its own
[Agent C] reads from / acts on combined traces
... no agent talks to another agent directly ...
```

```python
class StigmergicScratchpad:
    """Shared environment that mediates agent coordination."""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._traces: list[Trace] = []
    
    async def deposit(self, agent_name: str, trace_type: str, content: dict):
        """Agent leaves a trace — task started, finding made, claim flagged, etc."""
        async with self._lock:
            self._traces.append(Trace(
                agent=agent_name,
                type=trace_type,
                content=content,
                timestamp=now(),
            ))
    
    async def read(self, filter_fn=None) -> list[Trace]:
        """Other agents read traces relevant to their work."""
        async with self._lock:
            return [t for t in self._traces if filter_fn is None or filter_fn(t)]


class StigmergicAgent:
    def __init__(self, name: str, scratchpad: StigmergicScratchpad, capabilities: list[str]):
        self.name = name
        self.scratchpad = scratchpad
        self.capabilities = capabilities
    
    async def act(self):
        # Read environment
        traces = await self.scratchpad.read()
        
        # Decide if there's work this agent can do
        unhandled = [t for t in traces if self._can_handle(t) and not self._already_handled(t)]
        if not unhandled:
            return
        
        # Take a task; mark it claimed
        task = unhandled[0]
        await self.scratchpad.deposit(self.name, "claimed", {"task_id": task.id})
        
        # Do work; deposit result
        result = await self._do(task)
        await self.scratchpad.deposit(self.name, "result", {"task_id": task.id, "result": result})
```

**When stigmergy is the right pattern:**

- Tasks decompose into independent units that can be picked up by any capable agent
- Agents have overlapping but not identical capabilities — flexibility about who does what
- The full set of subtasks isn't known upfront (agents can deposit new subtasks as they work)
- Coordination overhead of explicit messaging is high enough to justify environmental mediation

**Real production examples:**
- Distributed scraping/research systems where any agent can pick up any URL to fetch
- Issue triage systems where any agent can claim and resolve any issue
- Iterative document editing where multiple agents improve different sections without colliding

**When stigmergy is NOT the right pattern (most cases):**
- Task decomposition is known and stable (use supervisor)
- Agents have specialized roles that map cleanly to subtasks (use sequential pipeline)
- Scale is small enough that explicit coordination is cheap (use anything else)

::: brain
You're building a system where 10 agents need to collectively edit a long document. Each agent specializes in a different aspect (clarity, concision, factual accuracy, citations, formatting). Stigmergy or supervisor?

(Almost certainly supervisor. The agents have *specialized* roles that map cleanly to subtasks. Stigmergy would mean each agent reads the document and decides if there's work it can do — fine, but a supervisor that says "Clarity agent, your turn now" is simpler and gets the same result faster. Stigmergy is for when the *who-does-what* itself is the hard problem; with specialized roles, who-does-what is trivial.)
:::

::: code-exercise
**Exercise 15.1 — Build a stigmergic research system.**

5 research agents, all with web_search and read_url tools. A shared scratchpad. Initial seed: a research question and a list of starting URLs. Each agent's loop:
1. Read the scratchpad — what URLs have been fetched, what findings exist, what unresolved questions
2. Pick an action: fetch a new URL, follow up on a finding, ask a sub-question
3. Deposit results

Run for 5 minutes, observe coordination. Compare quality and cost vs the supervisor pattern from Module 14 on the same task.

(Expected outcome: stigmergy is more flexible but harder to control; supervisor is more predictable. The "right" choice depends on the task — and most teams should default to supervisor.)
:::

---

## Section 4 — Market and Auction Patterns (≈3 pages)

Another genuine swarm pattern: agents bid for tasks; tasks go to whoever bids best. Used in:
- Compute resource allocation in distributed systems
- Some research multi-agent setups for self-organizing capability matching
- Rare in production agent systems (most teams find supervisor is simpler)

The basic shape:

```python
class TaskMarket:
    async def post_task(self, task: Task) -> Bid:
        bids = await asyncio.gather(*[
            agent.bid(task) for agent in self.agents
        ])
        winning = max(bids, key=lambda b: b.score)
        return winning
    
class BiddingAgent:
    async def bid(self, task: Task) -> Bid:
        # Each agent self-assesses its fit for the task
        score = await self._estimate_fit(task)
        cost = await self._estimate_cost(task)
        return Bid(agent=self, score=score, cost=cost)
```

When this works:
- Heterogeneous agent pool with genuinely different capabilities
- Tasks come in with attributes that map to capabilities in non-obvious ways
- Self-assessment is roughly accurate

When it doesn't:
- Agents can't accurately self-assess (most LLM-based agents are bad at calibrated self-assessment)
- A supervisor with explicit routing rules is simpler
- The bidding overhead exceeds the value of better task placement

We don't expect most readers to build this. We cover it for completeness — and to demonstrate that "swarm" includes patterns beyond OpenAI Swarm's handoffs.

---

## Section 5 — When Emergence Is Real vs Theater (≈4 pages)

The "emergent intelligence" claim. Three categories:

**1. Genuine emergence: properties of the system not in any individual component.**

Examples in nature: an ant colony's optimal foraging path emerges from individual ants following pheromone gradients — no ant computes the path; the colony does, in aggregate.

In agent systems: extremely rare. The supervisor's plan is in the supervisor. The synthesis is in the synthesizer. The "emergent" output is usually traceable to a specific component.

**2. Apparent emergence: complex behavior from simple rules — but the complexity is in the rules, not their interaction.**

Example: a stigmergic system where agents follow simple "claim a trace if I can handle it; deposit result; move on" rules. Coordination looks emergent. Actually it's a queue with workers — emergence is doing little work in the explanation.

**3. Theater: the supervisor or aggregator is doing visible work, but it's framed as "emergent."**

Example: 10 agents in a group chat, manager picks who speaks, manager synthesizes. Output is described as "emergent from agent interaction." It's actually emerged from the manager's prompt.

The honest test: if you removed the orchestrator/supervisor/manager, does the system still produce coherent output? If no, the orchestrator was doing the work — it's not emergent.

::: pullquote
"Emergent" in agent systems is usually a marketing word for "I don't want to explain how the supervisor works."
:::

::: postmortem
**The "Emergent" System That Was a 50-Line Prompt**

A team gave a conference talk about their "emergent multi-agent research swarm." Diagram showed 7 agents with arrows in all directions. Phrase "emergent intelligence" appeared 5 times.

After the talk, an audience member asked to see the orchestration code. The "swarm" was: 7 prompts in a list, a manager that picked one to invoke each turn based on a 50-line prompt that read the conversation and decided. The 50-line prompt was where 95% of the system's behavior lived. The seven "specialist" agents were prompt templates with shared tools.

The audience member, a working ML engineer, said: "So it's a router with 7 destinations and the router does most of the work?" The presenter agreed.

**Lesson:** when you hear "emergent," look for the orchestrator. If there's an orchestrator, the orchestrator is the system — the agents are its tools. That's fine; it's just not emergence. Marketing it as emergence sets users up for disappointment.
:::

---

## Section 6 — Where Swarm Patterns Genuinely Win (≈3 pages)

The contrarian-to-the-contrarian section. Some workloads are legitimately swarm-shaped.

**1. Massively parallel exploration with deduplication.**

When you have a search space too large for centralized planning and where multiple agents finding the same thing is wasteful. Distributed crawlers, parallel hypothesis testing in research, computational drug discovery. Stigmergy through a shared "what's been explored" memory prevents redundant work.

**2. Long-lived agent populations with task discovery.**

When agents are persistent (not spawned per task) and tasks arrive continuously. Swarm-style "any agent can pick up any task it's qualified for" scales better than a supervisor that has to know about every task.

**3. Resilience to component failure.**

In a supervisor pattern, the supervisor is a single point of failure. In a swarm, individual agent failures don't bring down coordination — surviving agents see the same scratchpad and continue. For high-availability systems where downtime is expensive, this resilience matters.

**4. Heterogeneous, evolving capability sets.**

When you can't fully enumerate which agents exist or what they can do, market/auction patterns let agents self-select. Most production systems don't have this property; it's more common in research and in long-running internal tooling where capability evolves.

In practice: probably <5% of production agent systems are genuinely better as swarms. The other 95% should be supervisor or sequential pipeline. But that 5% exists, and the patterns matter when they do.

::: brain
Consider three systems: (a) a customer support routing system, (b) a distributed web crawler that builds a knowledge graph, (c) a code review system. Which is most swarm-shaped? Why?

(b is the strongest swarm fit — agents need to coordinate on what's been crawled, dedupe work via shared state, scale by adding more agents. a is sequential pipeline. c is supervisor or debate. The crawler benefits from genuine stigmergy — it's hard to centralize the planning of "what URL next" across millions of URLs.)
:::

---

## Section 7 — Building a Real Swarm (≈4 pages)

We code a stigmergic distributed-research system end-to-end. This is the swarm pattern that actually has production legs.

The task: given a research question, multiple agents collaborate to gather evidence by crawling links. Each agent works independently. Coordination is via a shared workspace.

```python
class ResearchWorkspace:
    """Shared environment for stigmergic coordination."""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._urls_to_visit: set[str] = set()
        self._urls_visited: set[str] = set()
        self._findings: list[Finding] = []
        self._questions: list[str] = []  # sub-questions agents have raised
    
    async def claim_url(self) -> str | None:
        async with self._lock:
            for url in list(self._urls_to_visit):
                if url not in self._urls_visited:
                    self._urls_visited.add(url)
                    self._urls_to_visit.discard(url)
                    return url
            return None
    
    async def add_finding(self, finding: Finding):
        async with self._lock:
            self._findings.append(finding)
    
    async def add_links(self, links: list[str]):
        async with self._lock:
            self._urls_to_visit.update(links - self._urls_visited)
    
    async def add_question(self, question: str):
        async with self._lock:
            self._questions.append(question)
    
    async def snapshot(self) -> dict:
        async with self._lock:
            return {
                "urls_pending": len(self._urls_to_visit),
                "urls_visited": len(self._urls_visited),
                "findings_count": len(self._findings),
                "open_questions": list(self._questions),
            }


class ResearcherAgent:
    def __init__(self, name: str, workspace: ResearchWorkspace, client: AsyncAnthropic):
        self.name = name
        self.workspace = workspace
        self.client = client

    async def run(self, original_question: str, max_iterations: int = 20):
        for _ in range(max_iterations):
            url = await self.workspace.claim_url()
            if url is None:
                # No work available — finish or yield to other agents
                snapshot = await self.workspace.snapshot()
                if snapshot["urls_pending"] == 0:
                    return  # done
                await asyncio.sleep(0.5)
                continue
            
            # Fetch and process
            content = await fetch_url(url)
            
            # Use LLM to extract findings + new links + new questions
            extracted = await self._extract(original_question, url, content)
            
            await self.workspace.add_finding(Finding(url=url, summary=extracted.summary))
            await self.workspace.add_links(extracted.relevant_links)
            for q in extracted.sub_questions:
                await self.workspace.add_question(q)
```

The full system: spawn 5-10 of these agents, all sharing one workspace, give them a starting URL, let them coordinate via the workspace.

We discuss the design choices:
- Lock granularity (per-collection vs whole-workspace)
- When agents should yield vs hammer the workspace
- How to detect "done" (no pending URLs for N seconds)
- How to bound the work (max URLs total, max time)

::: code-exercise
**Exercise 15.2 — Compare swarm vs supervisor on a research task.**

Take the research workspace + agents from Section 7. Run the same research task through:
1. Stigmergic swarm (5 agents, shared workspace)
2. Supervisor pattern (Module 14)

Measure: total tokens, total time, quality of final synthesis.

For most research tasks: supervisor wins on quality and tokens. Swarm wins on adaptability and parallelism scaling. Which one is "right" depends on the task properties.
:::

---

## Section 8 — Hype Glossary and What to Ask (≈3 pages)

A defensive glossary for vocabulary you'll encounter:

| Term | Often Means | Sometimes Means |
|---|---|---|
| Swarm | Multiple agents | Stigmergic coordination |
| Emergent | Output is unpredictable | Properties not present in components |
| Self-organizing | Has a manager that organizes them | Genuinely decentralized |
| Autonomous | Has tools and a loop | Unsupervised long-horizon operation |
| Multi-agent collective intelligence | A supervisor synthesizes outputs | Genuine collective reasoning |
| Distributed reasoning | Agents are in different processes | Reasoning that's actually distributed across components |
| Agent society | A few agents with personas | Persistent agents with evolving roles |

When someone pitches you a swarm, three questions to ask:

1. **Where's the orchestrator?** If there's one, the system is supervisor pattern in disguise. That's fine — but call it that.
2. **What's the shared environment that mediates coordination?** If there isn't one, you don't have stigmergy.
3. **What happens when one agent is removed?** If the system breaks, it's not a swarm — it's a coupled system.

Reasonable answers exist. Many "swarm" pitches don't survive these questions, and that's useful information.

::: pullquote
The vocabulary inflation in agent systems makes shopping hard. Anything-multi-agent gets called a swarm. Anything-multi-step gets called emergent. Anything-with-tools gets called autonomous. Translate every claim into the patterns you know — supervisor, pipeline, debate, stigmergy — and you'll see what's actually being offered.
:::

---

## Section 9 — What's Next (≈1 page)

Module 16 covers transactional / saga patterns for multi-agent — borrowing distributed-systems concepts (validation, rollback, compensating actions) for agent coordination. SagaLLM (PVLDB 2025) is the canonical reference; this addresses the "what if a sub-agent succeeds but a later one fails — how do you undo?" problem.

Module 17 covers production observability, eval, cost management — heavily relevant to multi-agent and swarm systems.

Module 18 (capstone) builds the multi-agent system. The swarm patterns from this module are unlikely to make the capstone (most production systems don't need them) but stigmergy as a sub-component is a strong candidate.

---

## Recurring elements used in this module

- 1 chapter opener
- 1 hook
- 2 No Dumb Questions
- 2 Brain Power prompts
- 1 Production Postmortem (the "emergent" 50-line prompt)
- 1 Watch It! / Gotcha block
- 2 Code Exercises (stigmergic research, swarm vs supervisor benchmark)
- 2 Pullquotes
- 1 hype-translation table
- 1 stigmergic system architecture diagram
- 1 Bullet Points recap

::: sam-arc
Sam, having shipped supervisor pattern (Module 14) for the premium tier, gets pitched at a conference: "your system would be even more powerful as a swarm." Sam asks the three questions from Section 8 ("where's the orchestrator? where's the shared environment? what if one agent dies?"). The pitcher answers two of three — there's no shared environment; coordination is via the manager. Sam concludes the pitch is for a renamed supervisor pattern, not a swarm. Doesn't switch architectures. Sam's arc this module: **vocabulary discipline beats vocabulary fashion. The right architecture for the task is whatever's the right architecture for the task — labels don't help.**
:::

::: page-budget
S1 (What swarm means): 4p
S2 (OpenAI Swarm honestly): 5p
S3 (Stigmergy): 5p
S4 (Market/auction): 3p
S5 (Emergence: real vs theater): 4p
S6 (Where swarms genuinely win): 3p
S7 (Building a real swarm): 4p
S8 (Hype glossary): 3p
S9 (Next): 1p
Recurring elements: 4p
TOTAL: ~36 pages
:::

::: sources
**Must verify when drafting:**

- OpenAI Swarm framework documentation and current status (Oct 2024 release; main repo status as of drafting)
- OpenAI Agents SDK (the production successor) — current API
- "Multi-Agent in Production in 2026: What Actually Survived" (Lanham, Apr 2026) — for the swarm-only-as-subroutine framing
- Digital Applied 2026 patterns taxonomy — "swarm and blackboard patterns are theoretical" claim
- Stigmergy in computer science — academic background (Beni, Wang 1989; ant colony optimization literature)
- Real production stigmergic systems — find concrete examples (distributed crawling, etc.)
- AutoGen GroupChat current status (article said maintenance-only as of 2026 — verify)
- Claim that <5% of production agent systems benefit from swarm — moderate this; it's an educated estimate

**Stable knowledge:**
- The biological metaphor and what stigmergy actually means
- The handoff mechanic (already established in Module 14)
- Three categories of "emergence" (genuine, apparent, theater)
- The hype glossary patterns
- Decision criteria for when swarm patterns earn their cost

**Cross-references:**
- Module 13 — multi-agent honest trade-offs
- Module 14 — the architectures swarms get conflated with
- Module 16 — transactional patterns for multi-agent
- Module 17 — observability for distributed agent systems
- Module 18 — capstone (swarm unlikely to make it; stigmergy possibly)
:::

::: bullet-points
### Module 15 in eight bullets

(filled at draft time)

- "Swarm" is an overused term; most "swarm" systems are supervisor pattern with marketing
- OpenAI Swarm is a handoff framework (Module 14 pattern) — explicitly a reference implementation, not production
- Stigmergy is the genuine swarm pattern: coordination via traces in shared environment
- Market / auction patterns let agents self-select for tasks; rare in production
- "Emergent" usually describes the orchestrator doing visible work; genuine emergence is rare
- Real swarm wins: massively parallel exploration, persistent agents with task discovery, resilience, evolving capabilities
- Build stigmergy with shared workspace + claim/deposit primitives; cap iterations and bound work
- Three questions to ask any swarm pitch: where's the orchestrator? where's the shared environment? what if one agent dies?
:::
