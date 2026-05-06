# Module 15 — Inter-Agent Communication

::: chapter-opener
<div class="module-num">MODULE 15</div>
<div class="module-title">Inter-Agent Communication</div>
<div class="subtitle">The architecture from M14 says where the agents sit. Communication is what flows between them.<br>Get the protocols wrong and the architecture doesn't matter.</div>
<div class="pages">~38 pages · the contracts, channels, and consensus mechanisms that distinguish working multi-agent systems from chaos</div>
:::

::: hook
The regulatory-research system shipped on a Tuesday. Sam's team had built it as orchestrator-worker, exactly per M14's framework: one orchestrator, four jurisdiction subagents (US, EU, APAC, Latin America), structured returns to a synthesizer.

By Friday it was producing weird outputs.

A compliance officer flagged the first case: a brief on cross-border data-flow regulation. The US subagent had returned that the relevant regulation "applied to all transfers of personal data exceeding 100 records." The EU subagent had returned that the *same* regulation "applied only to transfers exceeding 10,000 records, with exemptions for entities certified under the Adequacy Framework." The synthesizer, doing what it was prompted to do, had produced a single statement: "The regulation applies to transfers above a threshold; specific limits vary by jurisdiction."

That sentence was, technically, true. It was also useless. The compliance officer wanted to know which threshold applied to *their* situation. The synthesizer had papered over the contradiction by abstracting it.

Sam pulled the trace. Both subagents had grounded their findings in real sources — the US one had read the Cross-Border Data Flow Act; the EU one had read GDPR Article 45. They were reading *different regulations* and reporting them as the same one. The orchestrator's task description said "research the cross-border data-flow regulation," and each subagent had interpreted "the regulation" through the lens of its own jurisdiction.

The architecture was right. The communication was wrong. The subagents had no way to *coordinate* on what they meant by "the regulation." The structured returns were structured but the *content* of those structures had silently diverged.

Sam wrote the line: *"the architecture says where the agents sit. The communication says what flows between them. We got the architecture right and the communication wrong."*

The next thing Sam wrote was the title of a doc: *Inter-agent communication patterns for the regulatory-research system.* That doc became Module 15.
:::

---

## What this module is

M14 covered the four canonical multi-agent architectures: orchestrator-worker, hierarchical, peer collaboration, swarm. The architectures define *where the agents sit* — the topology, the roles, the dispatch. This module covers *what flows between them* — the contracts, channels, protocols, and consensus mechanisms that turn an architecture into a working system.

The cold open is a real failure mode. Two subagents producing structured outputs that *type-checked* but *contradicted* each other, with no mechanism to surface or resolve the contradiction. The architecture can't fix this on its own. Communication discipline can.

This module covers:

- **Section 1** — structured handoffs. The contract between agents that does more than just "return JSON"
- **Section 2** — shared state and blackboard patterns. When agents need to read/write coordinated state, with the concurrency disciplines that prevent corruption
- **Section 3** — voting and consensus mechanisms. When peer agents need to agree, including the underrated "tool-grounded consensus" pattern
- **Section 4** — message-passing and async coordination. Idempotency, deduplication, the patterns from distributed systems that apply
- **Section 5** — A2A and the cross-vendor protocols. Google's Agent2Agent standard, the relationship to MCP, when to use which
- **Section 6** — emergent-behavior controls. Budget caps, deadlock prevention, observable coordination — the disciplines that distinguish working multi-agent systems from chaos
- **Section 7** — Sam's regulatory-research system, post-fix
- **Section 8** — the framework

By the end of this module:

- You'll have working code for structured handoffs that do more than type-check — they enforce semantic agreement
- You'll know when blackboards beat point-to-point communication and when they don't
- You'll understand the four consensus mechanisms (majority voting, weighted voting, deliberation, tool-grounded) and which to use when
- You'll be able to design multi-agent systems whose communication patterns are debuggable, observable, and bounded

The connection to M14: M14 tells you which architecture. M15 tells you how to make that architecture actually work in production. Together they cover the multi-agent design space.

::: pullquote
The best multi-agent architecture in the world fails if the agents can't communicate cleanly. Structured returns aren't enough — they enforce types, not meaning. The communication discipline is what closes the gap between "the system runs" and "the system produces correct results."
:::

---

## Section 1 — Structured Handoffs Beyond Type Safety

M14's `WorkerTask` and `WorkerResult` Pydantic models are the starting point. They're necessary but not sufficient.

The Sam case from the cold open shows why. Both jurisdiction subagents returned valid `WorkerResult` objects with correctly-typed `findings` dicts. The Pydantic validation passed. The synthesizer accepted the results. The contradiction was invisible to the type system because the *meaning* of "the regulation" had drifted between subagents.

Effective handoffs do four things, not one:

**1. Type-check the structure** (Pydantic, JSON Schema)
**2. Anchor the semantics** (shared identifiers, explicit references)
**3. Surface uncertainty and disagreement** (confidence, obstacles, contested claims)
**4. Make context portable** (everything the receiver needs, with no implicit assumptions)

The Pydantic schemas from M14 handle (1). Sections 1.1-1.4 add the rest.

### 1.1 Anchor the semantics

The fix for Sam's contradiction wasn't a better type system. It was *grounding the references*. Each subagent's findings had to be tagged with the *specific regulation* it was about, not "the regulation" generically.

```python
from pydantic import BaseModel, Field
from typing import Literal


class RegulationReference(BaseModel):
    """A specific, identifiable regulation. Not 'the regulation' generically."""
    jurisdiction: Literal["US", "EU", "APAC", "LATAM"]
    short_name: str  # e.g., "GDPR Article 45"
    full_citation: str  # "Regulation (EU) 2016/679 Article 45"
    effective_date: str  # ISO date
    source_url: str | None = None


class JurisdictionFinding(BaseModel):
    finding_id: str  # globally unique
    regulation: RegulationReference  # WHICH regulation this finding is about
    claim: str  # what the finding says
    confidence: float = Field(ge=0.0, le=1.0)
    source_quote: str  # verbatim text from the source
    source_location: str  # paragraph, section, page

    # Cross-reference: does this finding contradict, complement, or extend
    # findings from other jurisdictions about adjacent regulations?
    relates_to: list[str] = []  # finding_ids from other jurisdictions
    relationship_type: Literal["contradicts", "complements", "extends", "independent"] | None = None
```

Two structural changes from a naive `findings: dict`:

**The `RegulationReference` is its own type.** Subagents can't drop free-text "the regulation" — they have to specify which one. Two subagents reporting on different regulations now produce findings that visibly reference different `RegulationReference` objects. The contradiction in Sam's cold open becomes visible at the schema layer.

**The `relates_to` and `relationship_type` fields make cross-jurisdiction relationships explicit.** A subagent that finds a regulation similar to one another subagent reported on flags the relationship: *"this complements US Cross-Border Data Flow Act," "this contradicts the threshold reported by the EU subagent," etc.* The synthesizer now has structured information about how findings relate, not just isolated structured returns.

This is the M10 lesson from hallucinations applied to inter-agent communication: **provenance and explicit identity prevent the model from silently conflating distinct things.**

### 1.2 Surface uncertainty and disagreement

A handoff that says "here's my finding" omits the most important production information: *how confident is this finding, and what couldn't I figure out?*

```python
class WorkerOutput(BaseModel):
    task_id: str
    status: Literal["complete", "partial", "blocked"]

    findings: list[JurisdictionFinding]

    # The "what I couldn't do" channel
    obstacles: list[str] = []
    open_questions: list[str] = []
    information_gaps: list[str] = []

    # Self-assessment
    overall_confidence: float = Field(ge=0.0, le=1.0)
    recommended_followups: list[str] = []
```

The discipline: workers must report what they didn't accomplish, not just what they did. A subagent that says "complete" with `obstacles=[]` and `confidence=1.0` has either done excellent work or hidden problems. A subagent that says "partial" with explicit `obstacles=["Could not find effective date for EU Adequacy Framework decision"]` is being honest about the limits of its work — and the synthesizer can act on that information.

Anthropic's Claude Code subagents documentation makes this explicit: *"If you can't complete the task, return partial results with explicit obstacles."* The "obstacles" channel is part of the contract, not an exception path.

### 1.3 Make context portable

The receiver of a handoff has no access to the sender's context. They see only what's in the handoff. So the handoff must be self-contained.

The mistake teams make: a worker's output assumes the synthesizer "knows" something — knows the user's original question, knows the worker's task scope, knows what the worker tried that didn't work. The worker's context has all this; the synthesizer's context has only the structured handoff.

The fix: every handoff includes the relevant context. Not the worker's full context (that defeats the purpose of context isolation), but the *specific framing the receiver needs*.

```python
class CompleteHandoff(BaseModel):
    # The output (Section 1.2)
    output: WorkerOutput

    # The context that travels with it
    original_task: WorkerTask  # what was asked
    interpretation: str  # how the worker understood the task
    scope_decisions: list[str] = []  # what the worker decided was in/out of scope
    methodology: str  # how the worker approached it (in 1-2 sentences)
    sources_consulted: list[str]  # references to what was read/queried
```

The `interpretation` field is doing important work. It surfaces the moment when a subagent might silently misinterpret the task. In Sam's regulatory case, each subagent's `interpretation` would have read something like *"researched cross-border data-flow regulation in [my jurisdiction]"* — making the divergence visible to the synthesizer rather than hidden in the findings.

When the synthesizer sees four `interpretation` strings that point to different actual regulations, that's a signal to surface the discrepancy to the user, not paper over it.

### 1.4 The handoff lifecycle

A handoff isn't a single message. It's a lifecycle:

**1. Dispatch** — orchestrator hands a task to worker (the M14 `WorkerTask`)
**2. Acknowledgment** — worker confirms receipt and intended interpretation
**3. Progress** (optional) — worker reports milestones for long-running work
**4. Return** — worker delivers the structured output (`CompleteHandoff` above)
**5. Reception** — orchestrator validates and integrates

The acknowledgment step is the underrated one. For high-stakes work, asking the worker to *restate the task in their own words before starting* catches misinterpretations early — much cheaper than discovering them after the worker has done the wrong work.

```python
async def dispatch_with_acknowledgment(
    worker_id: str,
    task: WorkerTask,
) -> WorkerOutput:
    # Step 1: send task
    # Step 2: get acknowledgment with interpretation
    ack = await worker_acknowledge(worker_id, task)

    # Optional: validate interpretation matches intent
    if not interpretation_matches(task, ack.interpretation):
        # Re-dispatch with clarifying context, or escalate to user
        clarified_task = await clarify_task(task, ack.interpretation)
        return await dispatch_with_acknowledgment(worker_id, clarified_task)

    # Step 3: actual execution; step 4: return; step 5: reception
    return await worker_execute(worker_id, task)
```

For Sam's regulatory case, this would have caught the problem at step 2: when each subagent's acknowledgment said "I will research [the specific regulation in my jurisdiction]," the orchestrator would have noticed the divergence — four different regulations being researched under one task description — and either clarified ("Yes, research each jurisdiction's analogous regulation") or split into four explicitly different tasks.

::: nodumbq
**Q: This adds a lot of overhead to handoffs. Isn't it overkill for routine work?**

For routine work, yes. The four-element handoff (semantic anchoring, uncertainty, portable context, lifecycle) earns its place when handoffs are *high-stakes* — when getting them wrong has real consequences. For low-stakes coordination (a worker that does a quick lookup and returns a value), simple Pydantic returns are fine. The discipline scales with the cost of misunderstanding. Sam's regulatory system is high-stakes; the customer-success bot's tool calls are not. Different handoffs need different rigor.

**Q: Doesn't the acknowledgment step double the latency?**

It adds a roundtrip, yes. For interactive systems, that's often unacceptable. For batch systems (research, analysis, planning), the roundtrip is rounding error and the misinterpretation prevention is worth it. For high-volume systems, you can use the acknowledgment pattern selectively — for the first dispatch of a new task type, then trust the pattern for repeated dispatches once you've validated worker interpretation is reliable.
:::

---

## Section 2 — Shared State and Blackboards

The handoff patterns from Section 1 are point-to-point: orchestrator to worker, worker back to orchestrator. They work well when communication is bilateral and structured.

Some multi-agent patterns require *shared state* — a place where multiple agents can read each other's work in progress, post findings as they're discovered, and converge on solutions incrementally. The classic name for this is the **blackboard architecture**, dating back to 1970s AI research. It's enjoyed a renaissance in 2024-2026 multi-agent LLM systems.

### What blackboard architecture is

A blackboard is a shared data store. Agents:

- **Read** the blackboard to see what other agents have contributed
- **Write** their own findings, claims, or partial solutions to the blackboard
- **Watch** the blackboard for changes and respond when relevant

Communication is *indirect*: agents don't message each other directly. They post to the shared store and read others' posts. The blackboard is the medium.

The bMAS paper (July 2025) showed blackboard-based LLM multi-agent systems outperforming static MAS designs by an average 4.33% on reasoning benchmarks while using fewer tokens. The architecture lets agents contribute *opportunistically* — whichever agent has relevant insight at any moment can post it; others read and build on it.

### When blackboards earn their place

Three cases:

**1. The right decomposition isn't known up front.** Orchestrator-worker requires the orchestrator to plan: "Worker A does X, Worker B does Y." For tasks where the structure emerges as you work — exploratory research, collaborative debugging, open-ended design — the blackboard's opportunistic contribution model handles the dynamism better.

**2. Agents naturally have overlapping but distinct expertise.** A team of specialists where any one of them might have the relevant insight at any moment. The blackboard lets each post when their expertise applies; the system doesn't have to predict ahead of time which specialist matters when.

**3. Iterative refinement of a shared artifact.** A document being drafted by multiple agents, each contributing a section. A plan being refined by multiple perspectives. The blackboard *is* the artifact, and the agents shape it together.

### When blackboards don't

- **When orchestrator-worker would have worked.** If the decomposition is clean, the centralized control of orchestrator-worker is easier to reason about.
- **High-volume routine work.** Blackboard's opportunistic model has overhead; for tasks where the structure is well-known, point-to-point handoffs are cheaper.
- **When concurrency is hard to manage.** Multiple agents writing to a blackboard simultaneously creates concurrency issues. Section 2.3 covers the discipline; if your team can't operate it, blackboards are a liability.

### A working blackboard implementation

```python
from datetime import datetime
from typing import Optional


class BlackboardEntry(BaseModel):
    entry_id: str
    author_agent: str  # which agent posted this
    timestamp: datetime
    section: str  # which part of the blackboard (see schema discussion below)
    content: dict
    references: list[str] = []  # other entry_ids this builds on
    confidence: float = Field(ge=0.0, le=1.0)
    revision_of: Optional[str] = None  # if this revises a prior entry


class Blackboard:
    """
    A simple in-memory blackboard. Production version would use a database
    with proper concurrency control (Section 2.3).
    """
    def __init__(self):
        self._entries: dict[str, BlackboardEntry] = {}
        self._lock = asyncio.Lock()
        self._subscribers: dict[str, list[callable]] = {}  # section -> callbacks

    async def post(self, entry: BlackboardEntry):
        async with self._lock:
            self._entries[entry.entry_id] = entry
            # Notify subscribers
            for subscriber in self._subscribers.get(entry.section, []):
                asyncio.create_task(subscriber(entry))

    async def read(
        self,
        *,
        section: str | None = None,
        author: str | None = None,
        since: datetime | None = None,
    ) -> list[BlackboardEntry]:
        async with self._lock:
            results = list(self._entries.values())
            if section:
                results = [e for e in results if e.section == section]
            if author:
                results = [e for e in results if e.author_agent == author]
            if since:
                results = [e for e in results if e.timestamp > since]
            return results

    def subscribe(self, section: str, callback: callable):
        """Register a callback that fires when entries are posted to a section."""
        self._subscribers.setdefault(section, []).append(callback)
```

The `Blackboard` class is the medium. Each agent reads relevant sections, performs its work, and posts back. The `subscribe` mechanism lets agents react to specific updates rather than polling.

### 2.1 Designing the blackboard schema

The single most important decision in a blackboard architecture is the *schema* — what sections exist, what writes go where, what each section contains.

The mindra production wisdom from December 2025: *"Design your blackboard schema so that different agents write to different, non-overlapping sections wherever possible."* Conflict-free design beats conflict-resolution code.

For a research-style blackboard:

```python
BLACKBOARD_SCHEMA = {
    "facts": {
        "description": "Established factual claims with sources",
        "writers": ["jurisdiction_subagents", "fact_checker"],
        "readers": ["all"],
    },
    "open_questions": {
        "description": "Questions that have surfaced but aren't yet answered",
        "writers": ["all"],
        "readers": ["all"],
    },
    "contradictions": {
        "description": "Conflicting claims that need adjudication",
        "writers": ["all"],
        "readers": ["orchestrator", "synthesizer"],
    },
    "synthesis_drafts": {
        "description": "Partial syntheses that the orchestrator builds incrementally",
        "writers": ["orchestrator", "synthesizer"],
        "readers": ["all"],
    },
    "metadata": {
        "description": "Coordination state: who's doing what, what's complete",
        "writers": ["orchestrator"],
        "readers": ["all"],
    },
}
```

Each section has clear writers and readers. Most writes are isolated to specific sections by author class. Conflicts are minimized structurally.

### 2.2 The blackboard cycle

A typical blackboard system follows a cycle:

1. **Orchestrator posts the goal** to the metadata section
2. **Specialists read** what's been posted; some find their expertise applies
3. **Specialists post** findings to relevant sections
4. **Other specialists read** new posts; build on or contradict them
5. **Orchestrator monitors** the contradictions section; adjudicates
6. **Synthesizer reads** facts + adjudicated contradictions; posts synthesis_drafts
7. **Cycle continues** until termination criteria met

The bMAS paper formalizes this: *"This iterative, context-driven orchestration allows the system to adapt collaboration patterns to each task instance, providing substantial flexibility when problem structures are ill-defined or evolve over time."*

Termination is the under-discussed part. A blackboard cycle can run forever if there's no clear stopping criterion. Common patterns:

- **Decider agent** explicitly signals termination based on blackboard state
- **Majority consensus** — agents vote that the synthesis is complete
- **Bounded iterations** — hard cap (e.g., 4 cycles per the bMAS implementation)
- **Convergence detection** — if the last N cycles produced no new entries, stop

For production, all four are typically used together: bounded iterations as a safety cap, convergence detection as the primary signal, decider agent as the explicit termination, majority consensus as a check on the decider.

### 2.3 Concurrency: the operational reality

Blackboards introduce concurrency problems that point-to-point handoffs avoid. Multiple agents writing simultaneously can:

- **Conflict on the same section.** Two agents posting findings about the same fact at once.
- **Race on revisions.** Agent A reads entry X, plans a revision; Agent B reads X first, posts a revision; Agent A's revision (based on the now-stale X) overwrites B's.
- **Deadlock through dependencies.** Agent A waits for Agent B's post; Agent B waits for Agent A's. Neither posts.

The disciplines:

**Schema-based isolation.** Different agents write to different sections by design. If specialist A writes only to `facts`, specialist B writes only to `synthesis_drafts`, they never conflict.

**Optimistic concurrency.** Each entry has a version number. When revising, the revising agent specifies which version it's revising. If the version has changed, the revision is rejected and the agent re-reads. Standard CAS pattern from distributed systems.

**Append-only patterns.** Don't update entries; post new ones that reference the old. The history is preserved; the latest is the authoritative. Trades storage for simplicity.

**Bounded iterations and timeouts.** Every blackboard operation has a timeout. Every cycle has a maximum duration. Deadlocks become timeouts; the system makes progress (possibly with degraded results) rather than hanging.

```python
async def post_with_optimistic_concurrency(
    blackboard: Blackboard,
    entry: BlackboardEntry,
    *,
    expected_predecessor_version: int | None = None,
):
    """Post with optimistic concurrency check."""
    if entry.revision_of and expected_predecessor_version is not None:
        async with blackboard._lock:
            existing = blackboard._entries.get(entry.revision_of)
            if existing and existing.version != expected_predecessor_version:
                raise ConcurrencyConflict(
                    f"Entry {entry.revision_of} has been modified "
                    f"(expected version {expected_predecessor_version}, "
                    f"actual {existing.version})"
                )
    await blackboard.post(entry)
```

::: gotcha
A common blackboard failure: agents that don't watch for relevance. The blackboard accumulates many entries; agents read everything every cycle. The token cost explodes; the agents lose the signal in noise. The fix: agents subscribe to *specific sections* relevant to their expertise, not the whole blackboard. The "monitor everything" pattern is exactly the M7 context-rot problem at the multi-agent layer.
:::

---

## Section 3 — Voting and Consensus

When multiple agents work on the same problem (peer collaboration from M14, or a blackboard with multiple specialists posting to the same section), the question becomes: *how do we decide which answer wins?*

Four canonical mechanisms.

### Mechanism 1: Majority voting

The simplest. N agents independently produce an answer; the most common answer wins.

```python
from collections import Counter


async def majority_vote(
    question: str,
    *,
    n_agents: int = 5,
    model: str = "claude-sonnet-4-6",
) -> dict:
    """Run N agents on the same question; majority wins."""
    answers = await asyncio.gather(*[
        run_agent(question, model=model, temperature=0.7)  # diversity from temperature
        for _ in range(n_agents)
    ])

    # Cluster similar answers; pick most common
    clustered = cluster_semantic_answers(answers)  # group by similarity
    counts = Counter(c.canonical_form for c in clustered)
    winner, count = counts.most_common(1)[0]

    return {
        "answer": winner,
        "vote_count": count,
        "total_agents": n_agents,
        "agreement_rate": count / n_agents,
        "all_answers": answers,
    }
```

Strengths: simple, robust against single-agent errors, the agreement_rate is itself signal (low agreement → low confidence).

Weaknesses: **all agents weighted equally**. A weak agent's vote counts as much as a strong one's. **Consensus on wrong answers is possible** — if the model has a systematic bias, all agents might produce the same wrong answer.

When to use: classification tasks, multiple-choice, structured outputs where "is the answer X or Y?" has a clean comparison.

### Mechanism 2: Weighted voting

Different agents have different weights based on their reliability or expertise.

```python
class WeightedAgent(BaseModel):
    agent_id: str
    weight: float
    expertise_areas: list[str]


async def weighted_vote(
    question: str,
    agents: list[WeightedAgent],
    relevant_expertise: str | None = None,
) -> dict:
    """Weight votes by agent reliability and expertise relevance."""
    answers = await asyncio.gather(*[
        run_agent_by_id(agent.agent_id, question)
        for agent in agents
    ])

    # Compute effective weights: base weight × expertise relevance
    weights = []
    for agent, answer in zip(agents, answers):
        weight = agent.weight
        if relevant_expertise and relevant_expertise in agent.expertise_areas:
            weight *= 1.5  # boost agents with relevant expertise
        weights.append(weight)

    # Sum weights by answer cluster
    clustered = cluster_semantic_answers(answers)
    weighted_counts = {}
    for cluster, weight in zip(clustered, weights):
        weighted_counts[cluster.canonical_form] = (
            weighted_counts.get(cluster.canonical_form, 0) + weight
        )

    winner = max(weighted_counts.items(), key=lambda x: x[1])
    return {
        "answer": winner[0],
        "weighted_score": winner[1],
        "total_weight": sum(weights),
        "confidence": winner[1] / sum(weights),
    }
```

Strengths: incorporates known reliability differences, handles heterogeneous agent quality.

Weaknesses: **the weights themselves can be wrong**. Bad weight calibration is worse than uniform weights — you've systematically tilted toward an unreliable agent.

When to use: when you have empirical evidence about agent reliability (from past evaluations), or genuine expertise differences (the medical agent's vote on medical questions outweighs the general agent's).

### Mechanism 3: Deliberation

Agents see each other's outputs and can revise their positions before a final vote. The 2024-2026 literature on multi-agent debate is essentially this.

```python
async def deliberation_vote(
    question: str,
    agents: list[str],
    rounds: int = 2,
) -> dict:
    """
    Round 1: each agent independent answer.
    Round 2+: each agent sees others' answers, can revise.
    Final: vote on revised answers.
    """
    # Round 1: independent answers
    current_answers = await asyncio.gather(*[
        run_agent(agent, question) for agent in agents
    ])

    # Subsequent rounds: revise with knowledge of others
    for round_num in range(rounds - 1):
        revised = await asyncio.gather(*[
            revise_answer(
                agent, question,
                my_answer=current_answers[i],
                other_answers=[a for j, a in enumerate(current_answers) if j != i],
            )
            for i, agent in enumerate(agents)
        ])
        current_answers = revised

    # Final vote on the deliberated answers
    return majority_vote_from_answers(current_answers)
```

Strengths: agents can correct each other; captures the M11 reflection benefit at the consensus layer.

Weaknesses: **convergence to majority can suppress correct minority views** — if 4 of 5 agents are wrong, the 5th may revise toward the wrong consensus. The classic groupthink failure mode applies. Also costs more (rounds × n_agents calls).

When to use: open-ended questions where reasoning matters, where seeing others' arguments has real value. Don't use when the right answer is unambiguous and you just need verification.

### Mechanism 4: Tool-grounded consensus

The Fastio framing from February 2026: *"Sometimes consensus is about fact, not opinion. If Agent A says X = 5 and Agent B says X = 7, you don't need a vote; you need a calculator."*

When agents disagree on something *verifiable*, don't vote — query a deterministic source.

```python
async def tool_grounded_consensus(
    question: str,
    agents: list[str],
    verification_tools: dict[str, callable],
) -> dict:
    """
    Run agents; if they disagree on facts, query tools for ground truth.
    """
    answers = await asyncio.gather(*[run_agent(agent, question) for agent in agents])

    # Extract claims from each answer
    claims = await extract_claims_from_answers(answers)

    # For each disputed claim, find a tool that can verify
    disputes = identify_disputes(claims)
    verifications = []
    for dispute in disputes:
        tool = find_relevant_tool(dispute, verification_tools)
        if tool:
            ground_truth = await tool(dispute.subject)
            verifications.append({
                "dispute": dispute,
                "ground_truth": ground_truth,
                "agents_aligned_with_truth": [
                    a for a in agents if a.claim_matches(ground_truth)
                ],
            })

    return {
        "agent_answers": answers,
        "verifications": verifications,
        "final_answer": construct_answer_from_verified_facts(answers, verifications),
    }
```

Strengths: **grounds consensus in objective reality rather than statistical agreement**. When the facts are checkable, this is strictly better than voting.

Weaknesses: requires verification tools for the relevant facts. For purely subjective questions ("is this design elegant?") no tool helps.

When to use: any disagreement that's verifiable. Math, logic, queries against authoritative sources, computations. *Always prefer this over voting when it applies.*

### Choosing the right mechanism

A decision tree:

1. **Is the disagreement about something verifiable?** → Tool-grounded consensus. Don't vote on facts.
2. **Are the agents heterogeneous in known reliability?** → Weighted voting.
3. **Does the question benefit from agents seeing each other's reasoning?** → Deliberation. Carefully — watch for groupthink.
4. **None of the above?** → Majority voting. Simple; robust against individual errors.

The Sam regulatory case from the cold open is a tool-grounded consensus problem dressed up as a contradiction. The subagents weren't disagreeing on opinion; they were grounded in *different actual regulations*. The fix wasn't to vote — it was to make explicit which regulation each was grounded in (Section 1's semantic anchoring), then surface the difference to the user. No vote needed; the underlying facts were verifiable through citation.

::: brain
A peer-collaboration system has three explorers each propose an architecture for a new feature. Two propose using a queue-based pattern; one proposes a synchronous RPC pattern. Should the system majority-vote, deliberate, or use tool-grounded consensus?

(Probably deliberation, possibly weighted voting if the explorers have known reliability differences. Definitely not naive majority voting — architecture is not a popularity contest. Tool-grounded consensus doesn't apply because there's no objective "correct architecture." The right move: round-2 revision where each explorer sees the others' arguments. If after revision two still favor queue-based and the third has updated reasons for synchronous RPC, the critic synthesizes the trade-off. The vote isn't the answer; the *reasoning* is.)
:::

---

## Section 4 — Message-Passing and Async Coordination

The patterns above are coordination *within* a request. Some multi-agent systems need coordination *across* requests, across time, across separate sessions. This is the territory where distributed-systems patterns from the past 30 years apply directly.

### 4.1 Idempotency

A message processed twice should produce the same result as processing once. This matters because in any real system, messages will sometimes be delivered twice.

```python
class IdempotencyKey(BaseModel):
    request_id: str
    operation: str
    timestamp: datetime


async def process_with_idempotency(
    message: Message,
    *,
    seen_keys: set[str],
):
    key = f"{message.request_id}:{message.operation}"
    if key in seen_keys:
        # Already processed; return cached result
        return get_cached_result(key)

    result = await actually_process(message)
    seen_keys.add(key)
    cache_result(key, result)
    return result
```

For LLM agent operations, idempotency is harder than for simple CRUD: an agent's response depends on context, and "processing twice" might involve a stale context. The discipline:

- **Tag every message with a unique request_id.** Agents check this before processing.
- **Cache results keyed by (request_id, operation).** Repeated requests return cached results.
- **For state-mutating operations**, idempotency is non-negotiable. M16's saga discipline relies on it.

### 4.2 Message ordering and causality

When agents send messages to each other asynchronously, ordering matters. Two patterns:

**FIFO per-source.** Messages from agent A to agent B are processed in order. Messages between *different* source-destination pairs may interleave. This is the cheapest discipline that maintains within-channel causality.

**Causal ordering.** Messages have explicit dependencies; the receiver waits until prerequisites have been processed before processing the dependent message.

```python
class CausalMessage(BaseModel):
    message_id: str
    sender: str
    receiver: str
    payload: dict
    depends_on: list[str] = []  # message_ids that must be processed first


async def process_causal_message(
    message: CausalMessage,
    *,
    processed_ids: set[str],
):
    # Wait for dependencies
    while not all(dep in processed_ids for dep in message.depends_on):
        await asyncio.sleep(0.1)

    # Now safe to process
    result = await handle_message(message)
    processed_ids.add(message.message_id)
    return result
```

For most multi-agent systems, FIFO per-source is sufficient. Causal ordering is for cases where messages have explicit dependencies that span agents.

### 4.3 Backpressure and flow control

If one agent produces messages faster than another can consume them, queues grow without bound. Backpressure is the discipline that prevents this.

Standard patterns from distributed systems apply:

- **Bounded queues.** When the queue is full, the producer blocks (or is rejected, or drops oldest). Producers can't outrun consumers indefinitely.
- **Rate limiting.** Producers are capped at N messages per second. Hard limit; doesn't depend on queue state.
- **Token-based flow control.** Consumers issue tokens; producers can only send when they hold a token. The consumer controls the rate.

For LLM agent systems specifically, backpressure usually shows up at the rate-limit boundary (Anthropic API rate limits) or at the cost-budget boundary (we don't want to run more than $X/minute of inference). Backpressure mechanisms tied to those budgets prevent runaway costs.

```python
class TokenBucket:
    """Simple token bucket for rate limiting agent dispatches."""
    def __init__(self, rate_per_second: float, burst: int):
        self.rate = rate_per_second
        self.burst = burst
        self.tokens = burst
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_refill = now
            if self.tokens < 1:
                # Wait until we have a token
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1
```

### 4.4 Timeouts and circuit breakers

Every cross-agent operation has a timeout. No exceptions. This is the single most important operational discipline in multi-agent systems.

```python
async def call_agent_with_timeout(
    agent_id: str,
    request: dict,
    *,
    timeout_seconds: float = 60,
) -> dict:
    try:
        return await asyncio.wait_for(
            call_agent(agent_id, request),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(f"Agent {agent_id} timed out after {timeout_seconds}s")
        return {"status": "timeout", "agent_id": agent_id}
```

The orchestrator handles timeouts as partial failures (Section 1's `status: "blocked"`) rather than crashes. The synthesis layer is built to accept partial results.

For repeated failures, **circuit breakers** prevent cascading degradation. If agent X has failed 5 times in 60 seconds, stop calling it; route around it for a cool-down period.

```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_seconds: float = 60):
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.failures = []
        self.opened_at: float | None = None

    def is_open(self) -> bool:
        # Clean up old failures
        now = time.monotonic()
        cutoff = now - self.recovery_seconds
        self.failures = [f for f in self.failures if f > cutoff]

        if self.opened_at and (now - self.opened_at) > self.recovery_seconds:
            # Half-open: allow one trial request
            self.opened_at = None
            self.failures.clear()
            return False

        return self.opened_at is not None

    def record_failure(self):
        self.failures.append(time.monotonic())
        if len(self.failures) >= self.failure_threshold:
            self.opened_at = time.monotonic()

    def record_success(self):
        self.failures.clear()
        self.opened_at = None
```

Standard circuit-breaker pattern, applied to agent calls.

::: pullquote
Multi-agent systems that don't bound timeouts, rate limits, and failure handling become operational nightmares. The discipline isn't optional; it's the difference between "the system works in dev" and "the system survives Friday afternoon production traffic."
:::

---

## Section 5 — A2A and the Cross-Vendor Protocols

A relatively recent development worth knowing about. Through 2024-2025, every multi-agent framework invented its own coordination protocol — LangChain had one, AutoGen another, CrewAI another. The result: agents from different frameworks couldn't easily communicate.

In April 2025, Google announced the **Agent2Agent (A2A) Protocol**, an open standard for inter-agent communication. By April 2026, it had been adopted by 50+ industry partners and donated (along with MCP) to the Agentic AI Foundation as community-governed standards.

### What A2A is

An HTTP/JSON-RPC protocol for agents to discover each other's capabilities and exchange structured messages. Key concepts:

- **Agent Cards** at `/.well-known/agent.json` — every A2A-compliant agent exposes a description of its capabilities at this well-known URL
- **Tasks** — units of work passed between agents, with explicit goals and outputs
- **Messages** — individual exchanges within a task
- **Event Queues** — temporary state holding for async coordination

A simplified A2A flow:

1. Agent A wants to delegate to Agent B
2. Agent A reads Agent B's agent card to understand B's capabilities
3. Agent A creates a Task with explicit objective
4. Agent A sends the Task to Agent B's HTTP endpoint
5. Agent B processes; sends back updates via SSE, polling, or webhooks
6. Agent B returns the final result

### A2A vs MCP

A common confusion. The two protocols address different things:

| Protocol | What it standardizes | Example |
|---|---|---|
| **MCP** (M5) | Agent-to-tool communication | Agent calls a database query tool |
| **A2A** | Agent-to-agent communication | One agent delegates to another |

They compose: An A2A-coordinated multi-agent system might use MCP for each agent's tool access. The agents talk to each other via A2A; each agent talks to its tools via MCP.

### When to use A2A

A2A earns its place when:

- **Cross-vendor coordination is needed.** Your orchestrator agent (built on Claude) needs to delegate to a specialist agent (built on a different model) maintained by a different team.
- **Agent discovery is dynamic.** New specialist agents come online; your orchestrator needs to discover and use them without hardcoded integration.
- **Long-running async coordination is required.** A2A's task-based model with event queues handles long workflows natively.

When *not* to use A2A:

- **Single-vendor systems.** If all your agents are built in one framework on Claude, framework-native coordination is simpler.
- **Synchronous request-response only.** A2A's async model adds overhead you might not need.
- **Operational maturity isn't there.** A2A introduces another protocol layer to operate, monitor, debug. For small teams, framework-native is often easier.

### Implementation note

```python
# Conceptual sketch — actual A2A SDK syntax varies
from a2a import A2AClient, AgentCard

# Discover an agent
agent_b_card = await fetch_agent_card("https://agents.example.com/specialist")
print(agent_b_card.capabilities)  # what can this agent do?

# Delegate a task
client = A2AClient("https://agents.example.com/specialist")
task = await client.create_task(
    objective="Analyze regulatory landscape for fintech in Q1 2026",
    context={"focus": "EU jurisdiction"},
)

# Stream updates
async for update in task.stream():
    print(update)

# Get final result
result = await task.wait_for_completion()
```

For most production multi-agent systems on Claude, you won't need A2A directly — the architectures from M14 (orchestrator-worker, peer collaboration) implemented in your own code are simpler. But if you're building cross-vendor systems or expecting your agents to be discoverable by external systems, A2A is the relevant standard.

::: nodumbq
**Q: Should I bet on A2A becoming the dominant standard?**

It's the most credible candidate. Google's backing, 50+ partners, donation to the Agentic AI Foundation, and complementary positioning to MCP (which is well-established) all suggest it has staying power. But "industry standard" status takes time. As of mid-2026, A2A is the right standard to watch and adopt for cross-vendor scenarios. For single-vendor work on Claude, framework-native coordination remains simpler and more flexible.

**Q: What about ACP (Agent Communication Protocol)?**

ACP is a third standard that's emerged for lightweight messaging — closer to a message-bus protocol than A2A's task-coordination protocol. The roles divide roughly: MCP for tools, A2A for full task delegation, ACP for lightweight async messaging. For most agent systems, MCP + A2A covers the territory; ACP is a more niche choice for high-volume async messaging needs.
:::

---

## Section 6 — Emergent-Behavior Controls

The hardest part of multi-agent operations. Emergent behaviors don't show up in any individual agent's logs; they emerge from interactions. The disciplines that prevent chaos:

### 6.1 Per-agent budget caps

Every agent in a multi-agent system has hard caps:

- **Maximum turns** (M2's discipline)
- **Maximum tokens consumed**
- **Maximum tool calls**
- **Maximum dollars spent**
- **Maximum wall-clock time**

These are non-negotiable. When the cap is hit, the agent returns whatever it has with status `partial` or `blocked`. The synthesis layer handles partial results.

```python
class AgentBudget(BaseModel):
    max_turns: int = 20
    max_input_tokens: int = 200_000
    max_output_tokens: int = 30_000
    max_tool_calls: int = 30
    max_dollars: float = 5.0
    max_seconds: float = 600.0


class BudgetTracker:
    """Track budget consumption; raise if exceeded."""
    def __init__(self, budget: AgentBudget):
        self.budget = budget
        self.turns = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.tool_calls = 0
        self.dollars = 0.0
        self.start_time = time.monotonic()

    def check_and_increment_turn(self):
        self.turns += 1
        if self.turns > self.budget.max_turns:
            raise BudgetExceeded("max_turns")
        if (time.monotonic() - self.start_time) > self.budget.max_seconds:
            raise BudgetExceeded("max_seconds")

    def record_usage(self, input_tokens: int, output_tokens: int, tool_calls: int):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.tool_calls += tool_calls
        # ... enforce caps; raise BudgetExceeded if any exceeded
```

The discipline: every agent loop checks the budget on every iteration. The first cap exceeded ends the agent. The orchestrator gets a partial result; synthesis proceeds with what's available.

### 6.2 Total system budget

Beyond per-agent caps, the orchestrator enforces a *system-wide* budget. The sum of agent costs + coordination overhead can't exceed a system-level cap.

This catches the case where many agents stay individually under their caps but collectively blow through your unit economics.

```python
class SystemBudget:
    def __init__(self, max_total_dollars: float, max_total_seconds: float):
        self.max_total_dollars = max_total_dollars
        self.max_total_seconds = max_total_seconds
        self.total_dollars = 0.0
        self.start_time = time.monotonic()
        self.lock = asyncio.Lock()

    async def reserve_for_agent(self, estimated_cost: float):
        async with self.lock:
            if self.total_dollars + estimated_cost > self.max_total_dollars:
                raise SystemBudgetExceeded(
                    f"Adding {estimated_cost:.2f} would exceed system cap "
                    f"of {self.max_total_dollars:.2f}"
                )
            self.total_dollars += estimated_cost

    async def report_actual(self, actual_cost: float, estimated_cost: float):
        async with self.lock:
            self.total_dollars += (actual_cost - estimated_cost)
```

This catches the case where the orchestrator's plan calls for 8 workers, each with $1 budget, plus synthesis costs — the system budget says "you have $10 total" and the orchestrator must trim its plan.

### 6.3 Deadlock prevention

Multi-agent systems can deadlock when agent A waits for agent B's output and agent B waits for agent A's. The standard disciplines:

**Acyclic dependency graphs.** No agent can depend on the output of an agent that depends on it. Statically enforceable in many architectures (orchestrator-worker is naturally acyclic; only the orchestrator depends on workers, never the reverse).

**Total timeouts.** Even if dependencies form cycles, every wait has a timeout. Deadlock becomes degraded performance rather than hung systems.

**Bounded retry counts.** If agent A times out waiting for agent B and retries N times, the retry stops. The orchestrator handles the failure.

For orchestrator-worker (the M14 default), deadlock prevention is mostly automatic — the architecture doesn't allow cycles. For peer collaboration with deliberation, bounded rounds prevent the agents from waiting on each other indefinitely. For blackboards, schema-based isolation prevents most cycle conditions; timeouts catch the rest.

### 6.4 Observable coordination

You cannot debug what you cannot see. Multi-agent systems require structured tracing of all coordination events:

- **Every dispatch** — orchestrator → worker, with task_id, timestamps, context size
- **Every return** — worker → orchestrator, with result, status, costs
- **Every blackboard write** — author, section, content size, timestamp
- **Every consensus event** — vote breakdown, deliberation rounds, ground-truth verifications
- **Every budget exceeded** — which agent, which cap, what state

The trace should be enough to reconstruct what happened. M17 will cover observability comprehensively; for M15, the principle: log every cross-agent event with enough structure to be queryable. Free-form logs aren't enough; structured events with consistent fields are.

```python
# Minimum coordination event log
class CoordinationEvent(BaseModel):
    event_id: str
    timestamp: datetime
    event_type: Literal[
        "dispatch", "return", "blackboard_post", "blackboard_read",
        "consensus_vote", "budget_exceeded", "timeout", "circuit_open",
    ]
    actor_agent: str
    target_agent: str | None = None
    correlation_id: str  # ties events together within a request
    payload: dict
```

When you ship multi-agent, ship the event log. The first time you debug a multi-agent failure without it, you'll wish you had it. The first time you debug with it, you'll wish you'd had it sooner.

::: postmortem
**The Multi-Agent System That Couldn't Be Debugged**

A team built a 6-agent collaborative content-generation system: writer, editor, fact-checker, formatter, SEO optimizer, final reviewer. Used a blackboard for coordination. Shipped to production.

For the first three weeks, it worked. Then customers started complaining about specific articles having factual errors that hadn't been caught. The team tried to debug. They couldn't.

Their logs showed each agent's outputs but not the coordination — which agent had read which other agent's blackboard entries, which entries had been revised by whom, which contradictions had been adjudicated by which agent. The trace was a flat sequence of LLM calls; the coordination structure was invisible.

It took two weeks of work to add structured coordination logging — by which point three more flawed articles had shipped. The retrofit revealed the actual problem: the fact-checker agent was *not* reading the writer's most recent revision before checking. It was checking an earlier draft. The blackboard schema let multiple writer revisions exist; the fact-checker had a bug where it always read the first one.

The team's two fixes:

1. **Coordination event logging from day one for any future system.**
2. **Blackboard schema constraint:** only the latest revision of any writer entry is visible to fact-checkers. Older revisions go to an `archive` section the fact-checker doesn't read.

Both fixes were obvious in retrospect. Both were *not obvious* until the team had structured coordination logs to expose the actual flow. The lesson: structured logging is not optional infrastructure; it's a prerequisite for operating multi-agent systems.
:::

---

## Section 7 — Sam's Regulatory-Research System, Post-Fix

Returning to the cold open. Sam rebuilt the regulatory-research system using the disciplines from this module:

### 7.1 Semantic anchoring (Section 1.1)

Each subagent's findings now reference a specific `RegulationReference`:

```python
class RegulationFinding(BaseModel):
    finding_id: str
    regulation: RegulationReference  # WHICH regulation; not "the regulation"
    jurisdiction: str
    claim: str
    confidence: float
    source_quote: str
    source_citation: str

    relates_to: list[str] = []  # finding_ids from other jurisdictions
    relationship_type: Literal["analogous", "contradicts", "complements", "independent"] | None = None
```

The orchestrator's task description was rewritten:

```
"Research the cross-border data-flow regulation."
```

became:

```
"Research the cross-border data-flow regulations applicable in your
jurisdiction. Each jurisdiction has its own analogous regulation; identify
the SPECIFIC regulation that applies in your jurisdiction and ground all
findings in that specific text. Tag findings with the regulation_reference
of what you actually researched."
```

Subagents now *can't* silently report on different regulations under one umbrella term. The schema makes the divergence visible.

### 7.2 Cross-jurisdiction relationship tagging (Section 1.1)

When a US subagent finds the Cross-Border Data Flow Act and the EU subagent finds GDPR Article 45, both reference the same *topic* (cross-border data flow) but different *regulations*. The new schema captures both:

```python
us_finding = RegulationFinding(
    finding_id="us-001",
    regulation=us_cbdfa_reference,  # specific
    jurisdiction="US",
    claim="Applies to transfers exceeding 100 records",
    relates_to=["eu-001"],
    relationship_type="analogous",  # similar topic, different specific rule
)

eu_finding = RegulationFinding(
    finding_id="eu-001",
    regulation=gdpr_45_reference,  # specific, different
    jurisdiction="EU",
    claim="Applies to transfers exceeding 10,000 records, with Adequacy exemptions",
    relates_to=["us-001"],
    relationship_type="analogous",
)
```

Both findings are correct; both are well-grounded; the synthesizer now sees that they're analogous (same topic) rather than contradictory (different topics).

### 7.3 Synthesis discipline

The synthesizer's prompt was rewritten to handle the analogous-but-different case:

```
When findings from multiple jurisdictions reference DIFFERENT regulations on
the same topic (relationship_type="analogous"), do NOT synthesize them into
a single statement. Present each jurisdiction's specific regulation
separately, with its own threshold/exemptions/details. The user needs to
know which regulation applies to their situation.

If the user's task description doesn't specify a jurisdiction, surface the
multi-jurisdictional landscape rather than picking one or averaging across.
```

The synthesizer that previously produced *"specific limits vary by jurisdiction"* now produces structured per-jurisdiction summaries.

### 7.4 Emergent-behavior controls (Section 6)

Each subagent has a hard budget: 30 turns max, 200K input tokens, $4 max. The system has a total budget: $25 per request, 8 minutes wall-clock max. Coordination events are logged structurally.

In the first week post-fix, the structured logs caught two cases of subagents hitting their budget cap mid-research. Both produced partial results with explicit obstacles ("Could not complete review of APAC jurisdictions; required additional research time"). The synthesizer flagged the gaps to the user. No more silent quality degradation.

### 7.5 Metrics

| Metric | Before fix | After fix |
|---|---|---|
| Briefs flagged for factual issues | 4 of 30 (13%) | 0 of 30 (0%) |
| Briefs with explicit gaps surfaced | 0 of 30 | 6 of 30 (20%) |
| Median latency | 6.2 minutes | 6.4 minutes |
| Median cost per brief | $14 | $15 |

The cost increase was minor; the latency increase was rounding error. The quality improvement was meaningful. More striking: the *gap surfacing* — 20% of briefs now explicitly mentioned what couldn't be researched, where before the system silently produced briefs that papered over gaps.

The compliance officer who'd flagged the original failure sent a follow-up note: *"This is the kind of brief I can actually use. When you say 'we couldn't determine X,' I know to investigate X separately. When you said 'specific limits vary by jurisdiction' before, I had to redo the research myself."*

Sam wrote one more lesson for the playbook: *"the synthesizer should never paper over disagreements. Visible disagreement is a feature; hidden disagreement is a bug."*

::: code-exercise
**Exercise 15.1 — Communication audit.**

Pick a multi-agent system you're working on (or a planned one). Audit its communication patterns:

1. **Are handoffs semantically anchored?** Do worker outputs reference specific identifiers (entities, regulations, files) rather than generic terms? Or does "the X" mean different things to different agents?
2. **Are uncertainty and obstacles surfaced?** Do workers have explicit channels for "what I couldn't do" — or does the schema only support "what I did"?
3. **Are concurrency disciplines in place?** If using shared state, is the schema partitioned by author? Are there optimistic concurrency checks?
4. **Are budgets enforced per-agent and system-wide?** What happens when a worker hits its cap? What happens when the system hits its cap?
5. **Are coordination events logged structurally?** Could you reconstruct "what happened" from logs alone, or would you need to read individual agent traces?

For any "no" answers, sketch the fix. The patterns in this module address each one specifically. The cost of adding them upfront is small; the cost of retrofitting after a production incident is substantial.
:::

---

## Section 8 — The Framework

Adding communication discipline to multi-agent systems:

**1. Structured handoffs go beyond Pydantic.** Type-check structure (necessary). Anchor semantics with explicit identifiers (essential for high-stakes work). Surface uncertainty and obstacles (operational discipline). Make context portable (no implicit assumptions).

**2. Use blackboards selectively.** When the right decomposition isn't known up front, when agents have overlapping expertise, when iterative refinement is the work. For cleanly-decomposable tasks, orchestrator-worker's point-to-point handoffs are simpler.

**3. Design blackboard schemas for conflict-free writes.** Different agents → different sections by author class. Optimistic concurrency for unavoidable conflicts. Append-only patterns where possible. Bounded iterations and timeouts.

**4. Pick the right consensus mechanism.** Tool-grounded for verifiable disagreements (always prefer this when applicable). Weighted voting for known-heterogeneous reliability. Deliberation for reasoning that benefits from seeing others. Majority voting for simple structured comparisons.

**5. Distributed-systems patterns apply.** Idempotency (every operation safe to retry). Bounded queues (backpressure). Timeouts on every cross-agent call (no exceptions). Circuit breakers for repeated failures.

**6. A2A for cross-vendor coordination.** Single-vendor multi-agent systems don't need it; cross-vendor systems benefit. MCP for agent-to-tool, A2A for agent-to-agent. Watch the standards space; both are now community-governed.

**7. Per-agent and system-wide budgets.** Hard caps on turns, tokens, tool calls, dollars, time. Sum-of-budgets bounded by system-level cap. Partial results when caps exceeded; never silent degradation.

**8. Structured coordination logs from day one.** Every dispatch, return, blackboard write, vote, budget event. Correlation IDs linking events within requests. Without this, multi-agent debugging is impossible.

**9. Visible disagreement is a feature.** Synthesizers should surface contradictions, not paper over them. Sam's regulatory case is the canonical lesson: hidden disagreement is the bug; visible disagreement is the system working correctly.

**10. The architecture (M14) and the communication (M15) compose.** Pick the right architecture for the work structure. Pick the right communication patterns for the coordination structure. Both have to be right; either alone isn't sufficient.

The discipline this enforces: **multi-agent systems are coordination systems first, model systems second.** The models are excellent. The architecture from M14 is sound. What turns these into production-grade systems is the communication discipline from M15. The patterns in this module aren't optional decoration; they're load-bearing infrastructure.

---

## Recap: Module 15 in eight bullets

::: bullet-points
- Structured handoffs need to do four things: type-check structure, anchor semantics with explicit identifiers, surface uncertainty and obstacles, make context portable. Pydantic alone is necessary but not sufficient. Sam's regulatory case showed the failure mode: type-checked outputs that silently referenced different regulations.
- Blackboard architecture earns its place when decomposition isn't known up front, when agents have overlapping expertise, or when iterative refinement is the work. Schema-design-for-conflict-avoidance beats conflict-resolution code. Bounded iterations + decider agents for termination.
- Four consensus mechanisms: tool-grounded (always prefer for verifiable disagreements), weighted voting (heterogeneous reliability), deliberation (reasoning benefits from seeing others; watch for groupthink), majority voting (simple structured comparisons). Sam's regulatory case was a tool-grounded problem dressed up as a vote — fix was anchoring, not voting.
- Distributed-systems patterns transfer directly: idempotency for retry safety, bounded queues for backpressure, timeouts on every cross-agent call (non-negotiable), circuit breakers for repeated failures. Multi-agent operations is distributed-systems operations.
- A2A protocol is the emerging cross-vendor standard for agent-to-agent communication. MCP for tools, A2A for agents, ACP for lightweight messaging. For single-vendor systems on Claude, framework-native coordination is simpler. Watch the standards space.
- Per-agent budgets (turns, tokens, tool calls, dollars, time) and system-wide budgets (sum-of-agents bounded). Hard caps. Partial results when caps exceeded. Never silent degradation.
- Structured coordination logs are not optional. Every dispatch, return, blackboard write, vote, budget event with correlation IDs. The first time you debug multi-agent without these, you'll wish you had them; the first time with them, you'll wish you'd had them sooner.
- Visible disagreement is a feature; hidden disagreement is a bug. Synthesizers should surface contradictions explicitly. Multi-agent systems that paper over conflicts produce statements that type-check but don't help users — Sam's "specific limits vary by jurisdiction" was technically true and operationally useless.
:::

---

::: sam-arc
**Sam, after the regulatory-research fix.**

A month after the fix shipped, Sam pulled up the dashboard. The regulatory-research system was now processing 60-80 briefs per week. Zero factual issues flagged in the latest 60. About 20% of briefs explicitly surfaced gaps ("could not determine X for jurisdiction Y; recommend separate research"). Compliance officers had stopped redoing the research themselves; the briefs were now actionable starting points rather than misleading summaries.

The platform team lead asked Sam what they'd learned beyond the technical fix. Sam thought about it.

"Two things. First, M14's architecture wasn't wrong. Orchestrator-worker was the right pattern. The bug was in the *communication* between the orchestrator and workers — specifically that 'the regulation' was an ambiguous reference that resolved differently for each worker. The architecture was correct; the communication discipline was missing.

"Second, the synthesizer's job changed. Before the fix, I thought of it as 'combine the workers' findings into one answer.' After the fix, I think of it as 'preserve the structure of disagreement when disagreement exists; surface gaps when gaps exist.' The synthesizer's job isn't to produce confident-sounding output. It's to be honest about what the workers found and didn't find."

The team lead nodded. "Document that for the playbook?"

Sam did. Three additions to the multi-agent design review checklist:

1. **Run M13 framework first.** Multi-agent earns its complexity? Yes/no.
2. **Run M14 framework next.** Which architecture? Pick one.
3. **Run M15 audit next.** Are handoffs semantically anchored? Are uncertainty channels in the schema? Are budgets bounded? Are coordination logs structured?

Three more agents went through this checklist over the next quarter. Each one shipped without the failure modes Sam had encountered. The patterns from M13-M14-M15 were now the platform's standard approach to multi-agent design.

Sam's arc this module: **the architecture says where the agents sit; the communication says what flows between them; both have to be right.** M2-M12 had been about each agent's individual rigor. M13 had been about *whether* to use multiple agents. M14 had been about *how to arrange* them. M15 was about *how they talk* — and how they don't, when communication discipline is missing. Together M13-M14-M15 formed the multi-agent triad: when, where, how.

Friday afternoon, looking at the dashboards. The deal-research workflow, the customer-success bot, the deal Q&A agent, the research-on-demand packets, the regulatory-research briefs — all running. All defensible. All operating with the disciplines from the first 15 modules. Three modules left in the book: operations (M16-17) and the capstone (M18).

The book had built one careful step at a time. The agent loop, the workflows, the tools, the protocols, the models, the context, the trajectories, the memory, the hallucinations, the reflection, the guardrails, the multi-agent decision, the multi-agent architecture, the multi-agent communication. Each module's discipline composed with all the others. Sam's systems weren't impressive because of any single module's pattern; they were impressive because they applied *all* of them, layer on layer, with the architectural restraint to use each layer only when it earned its place.

Sam closed the laptop. The hardest engineering work in agent systems, Sam had come to realize, wasn't building any one component well. It was the *discipline of when to use what* — when to add a layer, when to remove one, when to ship single-agent, when to escalate to multi-agent, when to add a guardrail, when to add reflection. The book was teaching that discipline as much as it was teaching the patterns themselves.

Three more modules. Then the system Sam had been building all along would be complete.
:::

---

## What's next

Module 16 covers sagas — the transactional safety pattern for agent systems that take real-world actions across multiple steps. When an agent has spent $50 sending three emails and the fourth one fails, what's the right behavior? The saga discipline answers that question and the broader class of "compensating actions for partial failures."

Module 17 covers observability for production agents. Distributed tracing for multi-agent systems. Cost dashboards. Quality monitoring. The instrumentation that turns "the system works in dev" into "the system survives at 3 a.m. on a Friday."

Module 18 is the capstone: a complete system pulling together every layer from M1-M17. Sam's deal-research workflow, customer-success bot, deal Q&A agent, research-on-demand packets, regulatory-research briefs — operating together as one production agent platform.

For now: take the framework from Section 8. If you have a multi-agent system in production, run the audit. If you're designing one, design with these patterns from day one. The single-agent disciplines from M2-M12 plus the multi-agent disciplines from M13-M15 are the foundation; M16-M17 are the operations layer; M18 is the integration. Six modules out of eighteen are about how to build them; the other twelve — including this one — are about how to think about them.

The architecture from M14 needs the communication from M15. Together they're the multi-agent half of the book. The first half (M2-M12) was building one agent well. The second half started with the gate (M13), proceeded to the architectures (M14), and now closes the design space (M15). What's left is making it all run reliably in production. M16 next.
