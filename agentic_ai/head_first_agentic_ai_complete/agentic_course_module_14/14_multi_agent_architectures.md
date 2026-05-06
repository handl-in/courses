# Module 14 — Multi-Agent Architectures

::: chapter-opener
<div class="module-num">MODULE 14</div>
<div class="module-title">Multi-Agent Architectures</div>
<div class="subtitle">For the cases that pass M13's framework, the actual architectures.<br>Four patterns, each with specific strengths, costs, and failure modes.</div>
<div class="pages">~38 pages · the architectures, with code, with honest trade-offs</div>
:::

::: hook
Wednesday afternoon. Sam was sketching the *one* multi-agent system the platform team had committed to building this quarter — the research-on-demand feature for the sales team. It was the rare system that passed M13's five-question framework cleanly:

- Q1 Parallelizable? **Yes.** Each target company researched independently.
- Q2 High-value outcome? **Yes.** Pre-call briefs feeding $50K-$500K deal conversations.
- Q3 Clear subagent boundaries? **Yes.** One subagent per company; bounded scope.
- Q4 Team capability? **Yes.** Sam's team had built operational infrastructure for the deal-research workflow already.
- Q5 Tried single-agent? **Yes** — and the single-agent version had a hard ceiling. AEs preparing for back-to-back customer calls would queue up 5-8 research requests; the single-agent system processed them sequentially over 20+ minutes. Multi-agent could process them in parallel in under 4.

The framework said multi-agent. Sam knew *that* much. What Sam didn't know was *which* multi-agent.

The literature was a mess. Sam pulled up bookmarks: orchestrator-worker. Hub-and-spoke. Swarm. Conductor. Mesh. Hierarchical. Pipeline. Flat. Tree. Three-explorers-plus-critic. Each with claimed strengths. Each with breathless production reports. Each with their own framework's marketing pitch.

Sam read through three architectures over coffee. Each one made sense in isolation. Each one sounded like the right answer when the author was advocating for it. By the third article, Sam had drawn a system on paper that combined elements from all three — and immediately recognized it as the multi-agent equivalent of the prompt accretion problem from M11. Patterns layered on patterns; complexity additive; design rationale lost.

Sam closed the laptop and wrote down the core question: *"for THIS specific task — parallel research over independent companies, synthesizing into briefs — what's the simplest architecture that works?"*

Module 14 was about answering that question. Not by surveying every architecture. By understanding the four canonical patterns deeply enough to recognize which one the task wants.
:::

---

## What this module is

M13 established when multi-agent earns its complexity. M14 covers the architectures themselves — for the cases that passed the framework. Four canonical patterns:

1. **Orchestrator-worker** (also called hub-and-spoke, conductor) — one central coordinator delegates to specialists. The pattern Anthropic's research system uses.
2. **Hierarchical** — orchestrator delegates to middle managers who delegate to workers. Tree-structured. Rare; specific use cases only.
3. **Peer collaboration** (also called debate, three-explorers-plus-critic) — multiple agents work on the same problem with different framings, then a judge synthesizes. Claude Code's Ultra Plan uses this.
4. **Swarm** (also called mesh, decentralized) — many agents working semi-independently with light coordination. Highest parallelism; hardest to debug.

This module:

- **Section 1** — terminology cleanup. The literature uses many names for the same patterns. We'll fix that.
- **Section 2** — orchestrator-worker in depth, with full code. The pattern most teams should reach for first.
- **Section 3** — hierarchical patterns. When the task tree has genuine depth.
- **Section 4** — peer collaboration. When parallel exploration with synthesis beats sequential reasoning.
- **Section 5** — swarm. The honest treatment of when this earns its place (rarely).
- **Section 6** — choosing among the four. A decision framework with worked examples.
- **Section 7** — Anthropic's first-party patterns: Claude Code subagents (the `code.claude.com/docs/sub-agents` design), Ultra Plan's three-explorers-plus-critic, Agent Teams.
- **Section 8** — Sam's research-on-demand system, post-decision.
- **Section 9** — the framework.

By the end of this module:

- You'll know the four canonical multi-agent patterns and what each is for
- You'll have working code for orchestrator-worker and peer collaboration (the two most useful)
- You'll know when each pattern's emergent failure modes show up and how to design against them
- You'll be able to use Anthropic's Claude Code subagents pattern directly when it fits

The discipline: each pattern earns its place when the task's structure matches it. The wrong pattern multiplies cost and complexity for no benefit; the right pattern handles parallelism, isolation, or capability heterogeneity that single-agent can't.

::: pullquote
The biggest mistake teams make in multi-agent design isn't picking the wrong architecture — it's picking *several* architectures, layering them, and ending up with a system whose design rationale has been diluted to nothing. Each pattern in this module is a coherent answer to a specific question. Pick the one that matches your question.
:::

---

## Section 1 — Terminology Cleanup

The multi-agent literature is full of overlapping names for similar patterns. Different authors call the same architecture different things; sometimes the same name covers fundamentally different patterns. Before we go further, the canonical names this module uses:

| Pattern | Also called | What it actually is |
|---|---|---|
| **Orchestrator-worker** | Hub-and-spoke, supervisor-worker, conductor, lead-and-subagents | One coordinator delegates to specialists; specialists return; coordinator synthesizes |
| **Hierarchical** | Tree, multi-level orchestrator | Orchestrator delegates to middle managers, who delegate to workers; tree of depth ≥ 3 |
| **Peer collaboration** | Debate, multi-agent reflection, three-explorers-plus-critic | Multiple agents tackle the same problem from different angles; a synthesizer combines |
| **Swarm** | Mesh, flat, decentralized | Many agents working semi-independently; coordination through shared state or light messaging |

Three patterns I'll *not* call multi-agent:

**Pipeline** is workflows (M3). One LLM call's output flows to the next. No independent decision loops.

**Single-agent with subtools** is M2 + M4. The agent calls tools that internally invoke other LLM calls. One decision loop; multiple capabilities.

**Single-agent with reflection** is M11. The agent generates output and critiques itself. One context; one model; structured prompting.

These are valuable patterns. They're not multi-agent in the M13/M14 sense — they don't have separate decision-making contexts that justify the coordination overhead.

The four patterns this module covers each have *separate context windows*, *independent decision loops*, and *explicit coordination cost*. They're multi-agent in the strict sense.

::: nodumbq
**Q: What about all the other "patterns" I see in the literature — blackboard, contract net, market-based, role-playing?**

Most of those are either (a) variations on the four canonical patterns with different terminology, (b) academic patterns that haven't materialized in production, or (c) coordination mechanisms that work *within* one of the four patterns. Blackboard, for example, is a way of implementing inter-agent communication — it's a coordination mechanism, not an architecture pattern. Role-playing (where each agent plays a different character) is usually peer collaboration with extra costume. The four patterns in this module cover the production landscape; everything else is detail.

**Q: Where does "MCP-coordinated multi-agent" fit?**

MCP (Module 5) is the protocol layer. It's how agents talk to tools and to each other. It's not an architecture pattern; it's the messaging substrate that any of the four patterns can use. An orchestrator-worker system can communicate via MCP. A swarm can. Hierarchical can. MCP is plumbing, not architecture.
:::

---

## Section 2 — Orchestrator-Worker

The pattern most teams should reach for first when multi-agent is justified. The Anthropic research system. The Claude Code main agent + sub-agents pattern. The pattern with the cleanest production track record.

### What it is

One orchestrator agent. N worker agents. The orchestrator:

- Receives the user's task
- Decomposes it into subtasks
- Spawns workers, one per subtask
- Provides each worker with bounded scope and clear output expectations
- Waits for workers to complete
- Synthesizes worker outputs into the final result

Workers:

- Run in their own context windows (separate from orchestrator)
- Have their own tool access (potentially different from orchestrator)
- Run their own decision loops (M2's hardened loop, scoped to the subtask)
- Return structured results to the orchestrator

The architecture is a star: orchestrator at the center, workers at the points. Workers don't communicate with each other; all coordination flows through the orchestrator.

### Why it works

Three properties make orchestrator-worker the production default:

**1. Easy to reason about.** The control flow is explicit and centralized. When something fails, you look at the orchestrator's plan and the worker's execution; the failure is in one or the other. Multi-agent debugging becomes single-agent debugging at two layers.

**2. Workers are independent.** A worker can fail, retry, get replaced, run on a different model — without affecting the others. This is the same property that makes microservices easier to operate than monoliths.

**3. Synthesis is well-bounded.** The orchestrator sees structured outputs from workers, not raw worker context. The synthesis problem is "given these N structured results, produce the final answer" — much easier than "given N parallel context windows, somehow combine them."

### Anatomy of an orchestrator-worker system

```python
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field
from typing import Literal
import asyncio

client = AsyncAnthropic()


# ============== Worker side ==============

class WorkerTask(BaseModel):
    task_id: str
    objective: str
    context: dict  # bounded scope provided by orchestrator
    expected_output_format: str  # what the orchestrator wants back


class WorkerResult(BaseModel):
    task_id: str
    status: Literal["success", "partial", "failed"]
    findings: dict
    confidence: float = Field(ge=0.0, le=1.0)
    obstacles: list[str] = []  # things that prevented completion


WORKER_SYSTEM_PROMPT = """
You are a specialized research worker. You've been given a focused task with
clear scope. Your job is to:

1. Use the tools available to investigate the specific objective.
2. Stay within the bounds of the context provided. Do not explore tangents.
3. Return structured findings matching the expected_output_format.
4. If you can't complete the task, return partial results with explicit obstacles.

You operate in your own context window — what you find here doesn't directly
reach the orchestrator until you return your structured result. Be thorough
within your scope; report concisely.
""".strip()


async def run_worker(task: WorkerTask, *, model: str = "claude-sonnet-4-6") -> WorkerResult:
    """
    Run one worker on one task. Returns structured result to orchestrator.
    Worker has its own context window — fresh, bounded, focused.
    """
    messages = [{
        "role": "user",
        "content": (
            f"<task>\n{task.objective}\n</task>\n\n"
            f"<context>\n{json.dumps(task.context, indent=2)}\n</context>\n\n"
            f"<output_format>\n{task.expected_output_format}\n</output_format>"
        ),
    }]

    # Run M2's hardened loop, scoped to this worker's context
    final_response = await run_agent_loop(
        messages=messages,
        system=WORKER_SYSTEM_PROMPT,
        tools=WORKER_TOOLS,  # bounded tool set for the worker's task
        model=model,
        max_turns=20,
        max_tokens_total=80_000,  # per-worker budget
    )

    return parse_worker_result(final_response, task.task_id)


# ============== Orchestrator side ==============

class OrchestrationPlan(BaseModel):
    overall_goal: str
    subtasks: list[WorkerTask]
    synthesis_strategy: str


ORCHESTRATOR_SYSTEM_PROMPT = """
You are an orchestrator coordinating research workers. Your job:

1. Analyze the user's request and decompose it into focused subtasks.
2. For each subtask, write a clear objective, scope, and expected output format.
3. Workers operate in isolated context windows — they need everything in the
   task description to work effectively. Don't assume they have your context.
4. After workers complete, synthesize their structured findings into the final
   answer for the user.

Decomposition discipline:
- Subtasks should be genuinely independent (one worker doesn't need another's
  output to start)
- Subtask scope should be focused (one company, one document, one question)
- Expected outputs should be structured (so synthesis is well-bounded)

Avoid:
- Spawning workers for trivial subtasks (orchestrator could handle them faster)
- Decomposing serial tasks into "subtasks" that actually depend on each other
- Workers with vague objectives ("research this thoroughly")
""".strip()


async def orchestrate(user_request: str, *, max_workers: int = 8) -> str:
    """
    Top-level orchestration. Plan, dispatch workers in parallel, synthesize.
    """
    # Step 1: Plan
    plan = await create_plan(user_request)
    if len(plan.subtasks) > max_workers:
        # Cap the parallelism; reduce to the most important subtasks
        plan.subtasks = await prune_subtasks(plan.subtasks, max_workers)

    # Step 2: Dispatch workers in parallel
    worker_results = await asyncio.gather(*[
        run_worker(task) for task in plan.subtasks
    ], return_exceptions=True)

    # Handle worker failures gracefully (worker error → partial result; not
    # whole-system failure)
    parsed_results = []
    for result in worker_results:
        if isinstance(result, Exception):
            parsed_results.append(WorkerResult(
                task_id="?", status="failed", findings={},
                confidence=0.0, obstacles=[str(result)],
            ))
        else:
            parsed_results.append(result)

    # Step 3: Synthesize
    final_response = await synthesize(
        original_request=user_request,
        plan=plan,
        worker_results=parsed_results,
    )
    return final_response


async def create_plan(user_request: str) -> OrchestrationPlan:
    response = await client.messages.create(
        model="claude-opus-4-7",  # Orchestration is high-stakes; use the best model
        max_tokens=4096,
        system=ORCHESTRATOR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_request}],
        tools=[{
            "name": "submit_plan",
            "description": "Submit your decomposition plan.",
            "input_schema": OrchestrationPlan.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": "submit_plan"},
    )
    block = next(b for b in response.content if b.type == "tool_use")
    return OrchestrationPlan(**block.input)


async def synthesize(
    original_request: str,
    plan: OrchestrationPlan,
    worker_results: list[WorkerResult],
) -> str:
    response = await client.messages.create(
        model="claude-opus-4-7",
        max_tokens=8192,
        system=(
            "You are synthesizing the findings of multiple research workers "
            "into a final response. Each worker investigated one aspect of "
            "the user's request. Your job is to combine their structured "
            "findings into a coherent answer. Note any gaps, conflicts, or "
            "areas where workers couldn't complete their tasks."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Original request: {original_request}\n\n"
                f"Plan: {plan.synthesis_strategy}\n\n"
                f"Worker results:\n{json.dumps([r.dict() for r in worker_results], indent=2)}\n\n"
                "Produce the final response for the user."
            ),
        }],
    )
    return response.content[0].text
```

Five things in this code worth noting:

**Bounded worker context.** Each worker sees only its task description, not the orchestrator's full reasoning or other workers' contexts. The `WorkerTask` is the contract: everything the worker needs is there.

**Structured returns.** Workers return `WorkerResult` (Pydantic-typed). The orchestrator doesn't have to parse free-form worker output; the structure is enforced.

**Parallel dispatch.** `asyncio.gather` runs all workers in parallel. The wall-clock time is roughly the time of the slowest worker, not the sum.

**Failure isolation.** `return_exceptions=True` means one worker's exception doesn't kill the whole orchestration. The orchestrator gets a `failed` result back and synthesizes around it.

**Model selection per role.** Orchestrator on Opus 4.7 (high-stakes synthesis). Workers default to Sonnet 4.6 (the right cost-quality balance for focused subtasks). M6's discipline applied to multi-agent.

### Where orchestrator-worker excels

- **Parallel research over independent entities.** The Anthropic research system. Sam's research-on-demand for sales.
- **Document analysis across many independent documents.** Legal review pipelines, contract analysis, due diligence.
- **Comparison shopping across many products or vendors.**
- **Verification or fact-checking across many independent claims.**

### Where it fails

- **Tightly interdependent tasks.** Workers can't do A without knowing what B found; the parallel structure breaks down.
- **Fuzzy decomposition.** When the orchestrator can't produce clean subtask boundaries up front, workers get vague mandates and produce vague results.
- **Synthesis-dominated tasks.** When 90% of the work is combining the workers' outputs (and the workers' work is trivial), the orchestrator's synthesis becomes the bottleneck and parallelism doesn't help.

::: gotcha
A common failure: orchestrator-worker systems where the orchestrator's plan changes mid-execution. Worker A returns; orchestrator decides to spawn three new workers; those return; orchestrator decides to spawn more. The pattern degrades into ad-hoc agent spawning with no clean structure. The fix: commit to a plan up front; let it run to completion; *then* decide if a second round is needed. Iterative planning sounds adaptive but usually produces chaos. Plan once, dispatch, synthesize, optionally iterate.
:::

---

## Section 3 — Hierarchical

The orchestrator-worker pattern extended to N levels. An orchestrator delegates to middle managers; middle managers delegate to workers. Tree-structured.

### When this earns its place

Rarely, in production. The cases:

**1. The task tree has genuine depth.** A research project with both broad themes and deep sub-investigations. "Analyze the regulatory landscape for fintech" might decompose into "US," "EU," "APAC" themes; each theme decomposes into "banking regulation," "data privacy," "consumer protection" sub-investigations. Three levels of decomposition with real distinct structure at each level.

**2. Different middle layers have genuinely different roles.** A coding orchestrator might delegate to a "backend architecture" middle manager and a "frontend architecture" middle manager. Each has different specialist workers (database, API, infrastructure for backend; React, styling, accessibility for frontend). The middle layer adds value because it provides domain framing the workers benefit from.

**3. Scaling parallelism beyond what one orchestrator can manage.** When the number of workers exceeds what one orchestrator can plan and synthesize coherently (typically >15-20), middle managers reduce the orchestrator's coordination load. This is rare; most production systems have <10 workers.

### When it doesn't

When orchestrator-worker would have done the job. The diagnostic: can you remove the middle layer and have the orchestrator delegate directly to workers? If yes, you don't need hierarchy. The middle layer's only justification is *adding genuine value* — domain expertise, decomposition the orchestrator can't do, scaling the orchestrator's coordination.

Most teams reaching for hierarchy are over-engineering. The Bouchard "multi-agent overengineering" warning from M13 applies doubly: hierarchy is multi-agent stacked on multi-agent. Each level adds coordination overhead. The cumulative cost is significant.

### Example structure

```python
async def hierarchical_orchestrate(user_request: str) -> str:
    # Level 0: Top orchestrator
    top_plan = await top_level_plan(user_request)

    # Level 1: Middle managers run in parallel, each handling their own subtree
    middle_results = await asyncio.gather(*[
        run_middle_manager(theme) for theme in top_plan.themes
    ])

    # Level 0 again: Top orchestrator synthesizes middle managers' outputs
    return await top_level_synthesize(user_request, middle_results)


async def run_middle_manager(theme: ThemeTask) -> ThemeResult:
    # Level 1: Middle manager plans its subtree
    middle_plan = await middle_level_plan(theme)

    # Level 2: Workers run in parallel under this middle manager
    worker_results = await asyncio.gather(*[
        run_worker(subtask) for subtask in middle_plan.subtasks
    ])

    # Level 1 again: Middle manager synthesizes its subtree's outputs
    return await middle_level_synthesize(theme, worker_results)
```

The structure mirrors orchestrator-worker, just nested. Each level has its own plan/dispatch/synthesize cycle.

### Where hierarchical excels

- **Large research projects with natural domain boundaries** (the regulatory landscape example)
- **Complex codebase analysis spanning frontend/backend/infrastructure**
- **Document analysis across genuinely different document categories**

### Where it fails

- **When a flat orchestrator-worker structure would have worked.** Most cases.
- **When the middle layer doesn't add domain value.** It's just routing; remove it.
- **Operational complexity.** Three levels of multi-agent operations is significantly harder than two. M16-17 territory but more so.

::: brain
Sam's research-on-demand system processes 5-8 target companies per AE request. Should it use orchestrator-worker or hierarchical?

(Orchestrator-worker. The companies don't have natural sub-themes that warrant middle managers. Each company is a worker. The orchestrator handles the 5-8 companies directly. Adding a middle layer — "manage the financial subagent for each company, manage the leadership subagent for each company" — would be hierarchical theater. The companies are independent; the orchestrator's coordination load (5-8 workers) is well within manageable bounds. Flat orchestrator-worker is the right call.)
:::

---

## Section 4 — Peer Collaboration

The pattern Anthropic's Claude Code Ultra Plan uses: three explorers and one critic. Multiple agents work on the *same* problem from different angles; a synthesizer combines.

### What it is

Unlike orchestrator-worker (where workers handle *different* parts of the task), peer collaboration has multiple agents handling the *same* part with different framings:

- **Explorer 1** approaches the problem from perspective A
- **Explorer 2** approaches the same problem from perspective B
- **Explorer 3** approaches the same problem from perspective C
- **Critic** evaluates the explorers' outputs and synthesizes a final answer

The MAR paper from M11's research is the academic version of this. Anthropic's Ultra Plan is the production version. The MindStudio writeup describes Anthropic's framing: *"three parallel exploration agents each independently attempt the problem, and a dedicated critic agent evaluates their outputs before producing a final answer."*

### Why three explorers?

The structural argument: a single agent commits to an approach early and defends it (the M11 blind-spot problem). Multiple agents working independently from different framings produce diverse approaches. The critic gets to evaluate genuinely different angles.

Three is a reasonable default — enough diversity to surface different approaches, few enough to keep cost manageable. Two is too few (no tiebreaker on disagreements). Five is usually too many (diminishing returns; the third, fourth, and fifth explorers tend to overlap).

### Production code

```python
class ExplorationResult(BaseModel):
    perspective: str
    approach: str
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    open_questions: list[str]


EXPLORER_PROMPTS = {
    "first_principles": (
        "Approach this problem from first principles. Don't pattern-match to "
        "similar problems you've seen. Reason from the fundamental constraints."
    ),
    "comparative": (
        "Approach this problem by analogy to similar problems you've seen "
        "before. What patterns from prior solutions apply here? What lessons "
        "transfer?"
    ),
    "constraint_focused": (
        "Approach this problem by enumerating the constraints first. What MUST "
        "be true of any solution? Then propose an approach that satisfies all "
        "constraints."
    ),
}


async def run_explorer(perspective: str, problem: str) -> ExplorationResult:
    """
    Single explorer agent. Independent context window. Specific perspective.
    """
    prompt = EXPLORER_PROMPTS[perspective]
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=(
            "You are an exploration agent solving a problem from a specific "
            "perspective. Other agents are working on the same problem from "
            "different angles. Your job is to provide YOUR perspective's best "
            "approach, not to converge with others. Be specific and confident "
            "about your view, while noting what you're uncertain about.\n\n"
            f"Your perspective: {prompt}"
        ),
        messages=[{"role": "user", "content": problem}],
        tools=[{
            "name": "submit_exploration",
            "description": "Submit your exploration result.",
            "input_schema": ExplorationResult.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": "submit_exploration"},
    )
    block = next(b for b in response.content if b.type == "tool_use")
    return ExplorationResult(**block.input)


class CriticAssessment(BaseModel):
    chosen_approach: str
    reasoning: str
    incorporates_perspectives: list[str]  # which explorers' insights are in the final
    rejected_approaches: list[str]
    rejected_reasons: list[str]


async def run_critic(
    problem: str,
    explorations: list[ExplorationResult],
) -> CriticAssessment:
    """
    Critic synthesizes the explorers' outputs into a final approach.
    """
    response = await client.messages.create(
        model="claude-opus-4-7",  # Stronger model for synthesis
        max_tokens=4096,
        system=(
            "You are a critic synthesizing multiple exploration agents' "
            "approaches to a problem. Each explorer worked independently from "
            "a specific perspective. Your job:\n\n"
            "1. Evaluate each exploration's strengths and weaknesses.\n"
            "2. Identify approaches that contradict each other and adjudicate.\n"
            "3. Synthesize a final approach that incorporates the best of "
            "the diverse perspectives.\n"
            "4. Be explicit about what you're rejecting and why."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Original problem:\n{problem}\n\n"
                f"Exploration results from {len(explorations)} agents:\n"
                f"{json.dumps([e.dict() for e in explorations], indent=2)}\n\n"
                "Provide your synthesized assessment."
            ),
        }],
        tools=[{
            "name": "submit_assessment",
            "description": "Submit your critical synthesis.",
            "input_schema": CriticAssessment.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": "submit_assessment"},
    )
    block = next(b for b in response.content if b.type == "tool_use")
    return CriticAssessment(**block.input)


async def peer_collaboration(problem: str) -> CriticAssessment:
    """
    Run 3 explorers in parallel; have a critic synthesize.
    """
    explorations = await asyncio.gather(*[
        run_explorer(perspective, problem)
        for perspective in EXPLORER_PROMPTS.keys()
    ])
    return await run_critic(problem, explorations)
```

The pattern is similar to orchestrator-worker but the *purpose* is different. Orchestrator-worker decomposes; peer collaboration multiplies perspectives. The critic isn't just synthesizing partitioned work — it's adjudicating between competing approaches.

### Where peer collaboration excels

- **Open-ended planning tasks.** Architecture decisions, system design, complex code refactoring strategies.
- **Tasks where pattern-matching is a risk.** First-principles thinking matters; multiple framings reduce the chance everyone falls into the same trap.
- **High-stakes decisions where being wrong is expensive.** Multiple perspectives + critic synthesis produces more reliable judgments than single-agent reasoning.

### Where it fails

- **Routine tasks.** The cost of 3 explorers + 1 critic isn't justified for "summarize this email."
- **Tasks with one clear right answer.** Multiple framings on a deterministic problem produce three roughly-identical outputs and a critic that has nothing to do.
- **Latency-sensitive applications.** 4 LLM calls (3 explorers + 1 critic) plus the synthesis adds up; for interactive use, this can push beyond acceptable thresholds.

### Anthropic's Ultra Plan as the canonical case

Claude Code's Ultra Plan applies this pattern to *implementation planning*: when you ask Claude Code to plan a complex refactor, three explorer agents independently propose plans, a critic synthesizes. The result is a plan that benefits from multiple framings (first-principles, comparative, constraint-focused) before code gets written.

The MindStudio framing: *"You get something closer to how a good engineering team actually works — multiple perspectives, then ruthless evaluation."* The pattern works because the alternative (one agent committing to a plan early) genuinely produces worse plans for genuinely hard problems.

The cost: 4 LLM calls vs 1, with the orchestrator on Opus 4.7 (high stakes). Roughly 4-5× the inference cost of single-agent planning. Justified for plans that gate significant downstream work; overkill for routine queries.

::: pullquote
Peer collaboration earns its place when committing to one approach early is the failure mode you're trying to prevent. For tasks where any reasonable approach works, it's expensive overkill. For tasks where the *choice of approach* is the hardest part, it's the right architecture.
:::

---

## Section 5 — Swarm

The honest treatment. Swarm is the architecture pattern that gets the most enthusiastic coverage and the least production validation.

### What it is

Many agents (often dozens, sometimes more) working semi-independently on related tasks. Coordination happens through:

- Shared state (a database, a file system, a message queue) that agents read and write
- Light messaging (broadcast announcements, request-response) without a central orchestrator
- Workspace isolation (git worktrees, containerized environments) so agents don't interfere

Examples that get cited:

- Code-modification swarms where each agent works on a separate branch
- Research swarms where each agent investigates a different aspect with no central coordinator
- Content-generation swarms producing variations in parallel

### What's true about swarm

It can produce high parallelism. When the work decomposes into many independent units, swarm processes them concurrently with no orchestrator bottleneck. For embarrassingly parallel tasks at scale, this is genuinely faster than orchestrator-worker.

It has emergent quality benefits in some research settings. Multiple agents finding different angles on a problem can produce better aggregate results than one agent's sequential exploration.

### What's also true about swarm

It's hard to debug. When something goes wrong in a swarm, "which agent failed" is often the wrong question — usually it's an emergent coordination failure across multiple agents. The trace is N parallel timelines with cross-references; debugging is genuinely harder than orchestrator-worker.

It has goal drift. Without a central anchor (the orchestrator), agents can drift from the original task. The digitalapplied taxonomy notes: *"Failure modes: goal drift without supervisor anchor; coordination dead-locks; debugging difficulty."*

It's expensive. Many agents means many LLM calls. The token cost compounds harder than orchestrator-worker (which has fewer agents that each do more work). Swarm with 30 agents at 30K tokens each is 900K tokens before you've done any synthesis.

It rarely earns its place over orchestrator-worker for the same task. The cases where swarm genuinely beats orchestrator-worker tend to be ones where:

1. The number of parallel units is genuinely huge (>20)
2. Each unit is genuinely independent (no synthesis required)
3. The orchestrator's coordination load would itself be the bottleneck

These conditions are rare in production.

### When swarm earns its place

Be honest: rarely. The cases:

**1. Massively parallel research with no synthesis requirement.** "Generate 50 product description variants" or "explore 100 hypothetical scenarios." The agents are doing the same kind of work; no synthesis; results aggregated by simple selection.

**2. Code modification across genuinely independent files.** A coding swarm that fixes 30 unrelated lint errors in 30 different files in parallel. Each agent has workspace isolation; no agent's change affects another's.

**3. Open-ended research where the right structure isn't known.** Some research tasks genuinely don't have a clean orchestrator-worker decomposition. Swarm can explore widely; the cost of redundancy is acceptable when you don't know what you're looking for.

For everything else, orchestrator-worker is better.

### Production swarm code (illustrative)

```python
async def swarm_explore(
    task: str,
    *,
    swarm_size: int = 20,
    diversity_seed: list[str] = None,
):
    """
    Swarm exploration. Many agents work in parallel on related variants of
    the task. Results aggregated, not synthesized.
    """
    diversity_seed = diversity_seed or [f"variation_{i}" for i in range(swarm_size)]

    # Each agent gets a slightly different framing
    agent_tasks = [
        f"{task}\n\nAdditional context: {seed}"
        for seed in diversity_seed
    ]

    # Spawn N agents in parallel; each runs independently
    results = await asyncio.gather(*[
        run_swarm_agent(agent_task) for agent_task in agent_tasks
    ], return_exceptions=True)

    # Aggregation, not synthesis: rank/dedupe/select
    valid_results = [r for r in results if not isinstance(r, Exception)]
    return aggregate_swarm_results(valid_results)


async def run_swarm_agent(task: str) -> dict:
    """One swarm agent. No coordinator; runs to completion independently."""
    # ... independent agent loop
```

The aggregation step (`aggregate_swarm_results`) does ranking/dedup/selection — *not* synthesis. The output is "the best of the swarm's outputs," not "a unified result combining the swarm's work." This is the structural difference: swarm produces variants; aggregation picks; orchestrator-worker produces parts; synthesis combines.

::: gotcha
The most common swarm anti-pattern: building swarm because "more agents = better results" without the structural justification. Teams ship a 20-agent swarm, see modest quality improvement over single-agent (which they could have gotten from M11 reflection at 5% the cost), and conclude swarm "works." It works in the sense that it doesn't crash; it doesn't work in the sense of justifying its cost. Before building swarm, prove that orchestrator-worker can't do the same task — with evidence, not intuition.
:::

---

## Section 6 — Choosing Among the Four

A practical decision framework. Run through it after M13's gate (multi-agent earns its complexity); use it to pick which multi-agent pattern.

### Question 1: What's the structure of the work?

**N independent subtasks with synthesis at the end** → Orchestrator-worker. The dominant case.

**One problem; multiple framings beneficial** → Peer collaboration. When committing to one approach early is the risk.

**Tree-structured task with genuine domain depth at multiple levels** → Hierarchical. Rare; verify the middle layer adds value.

**Massive parallelism with no central synthesis** → Swarm. Rarer still; verify orchestrator-worker doesn't fit.

### Question 2: What's the synthesis problem like?

**Combining structured outputs from different parts** → Orchestrator-worker handles this. The synthesizer reads N structured results and combines.

**Choosing among or merging competing approaches** → Peer collaboration. The critic adjudicates.

**Aggregating many similar outputs (rank, dedupe, select)** → Swarm. No synthesis; aggregation suffices.

**Synthesizing across multiple levels of nested results** → Hierarchical. Each level synthesizes its children.

### Question 3: What's the operational complexity tolerance?

**Low — small team, basic observability** → Orchestrator-worker. Simplest to operate.

**Medium — established multi-agent operations** → Orchestrator-worker or peer collaboration.

**High — distributed systems team, mature observability** → Hierarchical or swarm acceptable. But still verify the simpler patterns wouldn't work first.

### Worked examples

**Anthropic Research system:** Q1 says N independent subtasks. Q2 says structured synthesis. Q3 says they have the operational sophistication. → Orchestrator-worker. (And that's what they built.)

**Claude Code Ultra Plan:** Q1 says one problem with multiple framings beneficial. Q2 says merging competing approaches. → Peer collaboration. (And that's what they built.)

**Sam's research-on-demand:** Q1 says 5-8 independent companies. Q2 says structured synthesis (combine company briefs into a presentation packet). Q3 says small team, established operations. → Orchestrator-worker.

**A code-modification system fixing 30 unrelated lint errors:** Q1 says massive parallelism. Q2 says no synthesis (each fix is independent). Q3 depends. → Swarm if the operational maturity is there; otherwise, batch into orchestrator-worker chunks of 5-10 files at a time.

::: brain
A team is building a financial-research system: pulls financials for a target company, analyzes risks, summarizes management changes, evaluates competitive position, builds a final report. Run the framework.

(Q1: Independent subtasks (financials, risks, management, competitive position) with synthesis at end → orchestrator-worker. Q2: Combining structured outputs into a coherent report → orchestrator-worker. Q3: Match team's operational capability. The answer is orchestrator-worker. Don't reach for hierarchical because there are five subtasks; that's still flat. Don't reach for peer collaboration because each subtask has a clear answer; multiple framings on "what's the company's revenue?" produce three identical numbers.)
:::

---

## Section 7 — Anthropic's First-Party Multi-Agent Patterns

Anthropic ships multi-agent infrastructure in two products. Worth knowing the canonical implementations.

### Claude Code subagents

The pattern documented at `code.claude.com/docs/sub-agents`. From the docs:

> *"Subagents are specialized AI assistants that handle specific types of tasks... Each subagent runs in its own context window with a custom system prompt, specific tool access, and independent permissions."*

The orchestrator (the main Claude Code agent) delegates to subagents based on their descriptions. Each subagent has:

- A name and description (used by the orchestrator for routing)
- A system prompt (focused, scoped)
- A specific tool set (often more restrictive than the orchestrator)
- A permission mode (often more conservative)

Built-in subagents Claude Code ships:

- **Explore** — read-only agent on Haiku, optimized for fast file discovery and codebase search
- **Plan** — research agent used during plan mode
- **General-purpose** — capable agent for complex multi-step tasks requiring both exploration and action

Definition format (Markdown + YAML frontmatter):

```yaml
---
name: repo-explorer
description: Search unfamiliar codebases, map entry points, summarize architecture. Do not edit files.
tools: [Read, Grep, Glob]
disallowedTools: [Edit, Write, Bash]
model: haiku
permissionMode: plan
memory: project
---

Find the main app entry points, core data flow, and likely risk areas.
Return a short summary with file paths, key abstractions, and open questions.
```

The architecture is orchestrator-worker. The orchestrator is the main Claude Code session; subagents are workers with bounded scope. The discipline of "subagents cannot spawn other subagents" prevents infinite nesting and keeps the architecture flat.

Two key insights from the Claude Code design:

**Subagents as context-management primitive.** From the architecture guide: *"That isolation is valuable even if you never care about specialization. Large-file research, codebase exploration, and dependency review are verbose. If you keep them in the main session, you burn context budget quickly."* Subagents preserve the main session's context by handling verbose work in isolation.

**Subagents as cost control.** Read-heavy work goes to Haiku-based subagents; the main session stays on Sonnet/Opus. M6's per-step model selection applied to multi-agent.

### Claude Code Ultra Plan

Anthropic's three-explorers-plus-critic implementation, available on Ultra subscriptions. From the description:

> *"Ultra Plan mode uses multi-agent exploration and critique to create a detailed implementation plan. So it's not just one Claude thinking through your task, it's multiple agents exploring the problem, checking each other's work, and giving you the best possible plan before a single line of code gets written."*

The architecture is peer collaboration. Three exploration agents work the same problem from different angles; a critic synthesizes. The output: a detailed plan the user can review and edit before execution begins.

The critical insight from the buzzrag writeup: *"It addresses the oversight problem... It maintains AI autonomy while inserting a mandatory human checkpoint. The question is whether this actually scales."* Ultra Plan is *deliberately* slower than single-agent planning — the slowness is the feature. For high-stakes plans, the user reviews the result of multi-agent exploration rather than getting the first thing one agent thought of.

The cost: roughly 4-5× single-agent planning, plus the user's review time. Justified for plans that gate significant downstream work.

### Agent Teams

Anthropic's pattern for *peer coordination across separate sessions* (vs subagents which work within one session). The pubnub article frames the distinction:

> *"Subagents stay inside one session and report back to the parent. Agent teams add peer coordination across separate sessions... that coordination can use about 7x more tokens in plan-heavy workflows and comes with more operational overhead."*

Agent Teams uses MCP-based communication between agents in different sessions. The architecture is closer to swarm or mesh than orchestrator-worker — agents are peers, not strict hierarchical roles. The cost (7× tokens) reflects the coordination overhead.

When to use Agent Teams:

- Multiple specialists need to collaborate as peers (not delegate)
- The work spans multiple sessions over time
- Inter-session coordination is genuinely required

When not to:

- A single session with subagents would have worked (most cases)
- The "peers" are actually doing different parts of one workflow (orchestrator-worker fits)

### What this means for your design

If you're building on Claude Code, the subagents pattern is the canonical orchestrator-worker for that environment. Don't reinvent it.

If you're building agentic features in Claude Code Ultra Plan style, the three-explorers-plus-critic is the canonical peer collaboration pattern. Use it for high-stakes planning.

If you're building outside Claude Code (your own application using the Anthropic API), the patterns from Sections 2-5 are the implementation reference. The architectural shapes are the same; the framework infrastructure is yours to build.

---

## Section 8 — Sam's Research-on-Demand System

Returning to the cold open. Sam ran the Section 6 framework:

- **Q1:** N independent subtasks (one per company; 5-8 per AE request) → orchestrator-worker
- **Q2:** Structured synthesis into a unified packet → orchestrator-worker
- **Q3:** Team has multi-agent operational capability from the deal-research workflow → orchestrator-worker is operationally feasible

The framework said orchestrator-worker. Cleanly. Sam didn't need hierarchical (no domain-deep tree), didn't need peer collaboration (no contested-approach problem; each company is its own clean problem), didn't need swarm (8 companies isn't massive parallelism).

### The architecture

```python
class CompanyResearchTask(BaseModel):
    company_name: str
    deal_context: str
    research_focus: list[str]  # which sections matter most for this AE


class CompanyResearchResult(BaseModel):
    company_name: str
    profile: dict
    financial_summary: dict
    leadership_summary: dict
    competitive_position: dict
    key_risks: list[str]
    confidence: float
    obstacles: list[str] = []


class ResearchPacket(BaseModel):
    ae_id: str
    request_id: str
    company_briefs: list[dict]  # one per company
    cross_company_insights: list[str]  # patterns across the set
    preparation_priorities: list[str]  # what to focus on for each call


async def research_on_demand(
    ae_id: str,
    target_companies: list[str],
    deal_contexts: dict[str, str],  # company_name -> deal stage description
) -> ResearchPacket:
    """
    AE provides 5-8 target companies; system researches all in parallel,
    synthesizes into a packet for back-to-back call prep.
    """
    # Step 1: Plan
    plan = await create_research_plan(ae_id, target_companies, deal_contexts)

    # Step 2: Dispatch one company-researcher worker per company
    company_results = await asyncio.gather(*[
        run_company_researcher(task) for task in plan.company_tasks
    ], return_exceptions=True)

    # Step 3: Synthesize into research packet
    packet = await synthesize_packet(
        ae_id=ae_id,
        plan=plan,
        company_results=[r for r in company_results if not isinstance(r, Exception)],
        failures=[i for i, r in enumerate(company_results) if isinstance(r, Exception)],
    )

    return packet


async def run_company_researcher(
    task: CompanyResearchTask,
) -> CompanyResearchResult:
    """
    One worker. Researches one company. Independent context. M2 hardened loop.
    """
    return await run_agent_loop(
        system=COMPANY_RESEARCHER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _format_research_task(task)}],
        tools=COMPANY_RESEARCH_TOOLS,  # company_lookup, market_research, etc.
        model="claude-sonnet-4-6",
        max_turns=12,
        max_tokens_total=120_000,
    )
```

The structure is exactly orchestrator-worker:

- **Orchestrator** plans (which companies, what focus per company), dispatches in parallel, synthesizes into packet
- **Workers** are company-researchers, one per target company, with focused mandate
- **Synthesis** is structured — combine 5-8 `CompanyResearchResult` objects into one `ResearchPacket`

### What Sam preserved from M2-M12

Each worker is a fully-disciplined single agent:

- M2 hardened loop with `max_turns` and budget caps
- M4 well-designed tools (the existing `company_lookup`, `competitor_analysis`, `market_research` from earlier modules)
- M7 context engineering (the worker's system prompt is at the right altitude; tools are bounded)
- M9 memory (workers can read existing brief archives via the `list_briefs` / `get_brief` tools)
- M10 hallucination discipline (worker outputs are entity-verified before returning)
- M11 reflection (the synthesizer runs a final-pass critic before returning the packet)
- M12 guardrails (the orchestrator's input is filtered; workers' outputs are policy-checked)

The multi-agent architecture isn't *replacing* the single-agent disciplines. It's *adding* parallel coordination on top of fully-disciplined single agents.

### The metrics

Three months after rollout:

| Metric | Single-agent (sequential) | Multi-agent (parallel) |
|---|---|---|
| Median time for 5-company packet | 21 minutes | 3.2 minutes |
| Median time for 8-company packet | 34 minutes | 4.1 minutes |
| Tokens per 5-company packet | ~280K | ~480K |
| Tokens per 8-company packet | ~445K | ~770K |
| Cost per 5-company packet | $4.20 | $7.20 |
| Cost per 8-company packet | $6.70 | $11.55 |
| Quality (AE-rated rubric) | 7.6/10 | 8.1/10 |

The trade-offs:

- **Latency:** dramatically better. 21 minutes → 3 minutes for 5 companies. AEs prepping back-to-back calls can use the system *during* prep, not the night before.
- **Cost:** ~70% increase. Acceptable: each packet supports a $50K-$500K deal conversation; $7-12 per packet is rounding error.
- **Quality:** modestly better. The parallel context windows let each worker focus deeply on one company without competing for attention budget.

The AE team's response was unambiguous. The packet system rolled out company-wide within four weeks. AEs reported being able to prep for 6-8 calls in a morning where they'd previously prepped for 2-3.

Sam wrote one more lesson for the playbook: *"the right multi-agent pattern is the simplest one that handles the structure of the work. We had clean parallel structure; orchestrator-worker fit. Anything more sophisticated would have been over-engineering."*

::: code-exercise
**Exercise 14.1 — Pattern selection.**

Take a problem you're considering for multi-agent. Run the Section 6 framework:

1. **Q1: What's the structure of the work?** Independent subtasks? Multiple framings? Tree-structured? Massive parallelism?
2. **Q2: What's the synthesis problem?** Structured combination? Adjudication? Aggregation? Multi-level synthesis?
3. **Q3: What's your operational tolerance?** Match operational complexity to team capability.
4. **Pick one pattern.** Don't combine patterns; commit to the one that matches.
5. **Sketch the implementation.** Worker contract, orchestrator plan, synthesis approach.
6. **Cost estimate.** How many LLM calls? What models? What's the token multiplier vs single-agent?

If the framework points to "no clean pattern fits," that's signal to revisit M13's framework — multi-agent might not have been the right call after all. The patterns in M14 are deliberately distinct. If your problem doesn't fit one cleanly, the architecture probably needs to be simpler than multi-agent.
:::

---

## Section 9 — The Framework

Building multi-agent systems for the cases that pass M13's gate:

**1. Default to orchestrator-worker.** The clearest, most operable pattern. Reach for it first. It handles most multi-agent use cases that pass M13's framework.

**2. Use peer collaboration for one specific case.** When committing to one approach early is the failure mode (open-ended planning, architecture decisions, high-stakes design). Three explorers + one critic is the canonical structure.

**3. Reach for hierarchical only when the task tree has genuine depth.** Verify the middle layer adds domain value. Most cases that "feel hierarchical" are actually flat orchestrator-worker with extra steps.

**4. Reach for swarm rarely.** When parallelism is massive (>20 units), each unit is genuinely independent (no synthesis), and orchestrator-worker would itself bottleneck on coordination. These conditions are rare in production.

**5. Each agent is fully M2-M12 disciplined.** Multi-agent doesn't replace single-agent discipline; it adds coordination on top. Workers have hardened loops, well-designed tools, context engineering, hallucination discipline, guardrails. M2 through M12 still apply per agent.

**6. Bounded scope per worker.** Workers' tasks should be focused, with clear inputs, expected output formats, and budget caps. Workers shouldn't have to ask the orchestrator for clarification mid-execution.

**7. Structured returns, not free-form.** Workers return Pydantic-typed results. Synthesis is "given these N structured results, produce the final answer" — well-bounded.

**8. Failure isolation.** One worker's failure should produce a partial result, not crash the orchestration. `return_exceptions=True` in `asyncio.gather`; graceful synthesis around partial failures.

**9. Use Anthropic's first-party patterns when you can.** Claude Code subagents for that environment. Three-explorers-plus-critic for peer collaboration. Don't reinvent infrastructure that exists.

**10. Match operational complexity to team capability.** Multi-agent operations is harder than single-agent. M16-17 cover the operational disciplines, but the foundation is "your team has to be able to debug what you build."

The discipline this enforces: **multi-agent is a deliberate response to specific structural requirements, with the simplest pattern that handles those requirements.** Not the most sophisticated; the simplest. Sam's research-on-demand earned its complexity by passing M13's framework, then chose the simplest M14 pattern that handled the structure (orchestrator-worker), then implemented each agent with full M2-M12 discipline. That's the production multi-agent recipe.

---

## Recap: Module 14 in eight bullets

::: bullet-points
- Four canonical multi-agent patterns: orchestrator-worker (one coordinator + N workers + synthesis), hierarchical (orchestrator + middle managers + workers; tree depth ≥ 3), peer collaboration (multiple agents on same problem + critic synthesis), swarm (many agents with light coordination, no central synthesis).
- Orchestrator-worker is the production default. The Anthropic research system, the Claude Code subagents pattern, and most multi-agent systems that ship and survive use this architecture. Easy to reason about, workers are independent, synthesis is well-bounded.
- Hierarchical earns its place rarely. The middle layer must add domain value (genuine sub-orchestration ability), not just routing. Most "hierarchical" sketches are actually flat orchestrator-worker that should be flattened.
- Peer collaboration earns its place when committing to one approach early is the failure mode. Three explorers + one critic (Anthropic Ultra Plan's pattern) for high-stakes planning, architecture decisions, complex code refactoring. 4-5× cost; justified when the choice of approach is the hardest part.
- Swarm rarely earns its place. Massive parallelism (>20 units), each genuinely independent, no synthesis required, orchestrator-worker would bottleneck on coordination. These conditions are rare. When they apply, swarm with workspace isolation; otherwise orchestrator-worker.
- Each agent in a multi-agent system is fully M2-M12 disciplined. Multi-agent doesn't replace single-agent discipline; it adds coordination on top. Workers have hardened loops, well-designed tools, hallucination prevention, guardrails. The complexity is additive.
- Anthropic's first-party patterns: Claude Code subagents (orchestrator-worker with bounded subagents in their own contexts), Ultra Plan (peer collaboration with three explorers + critic), Agent Teams (peer coordination across separate sessions; ~7× cost). Use these when the environment supports them rather than reinventing.
- The right pattern is the simplest one that handles the structure of the work. Sam's research-on-demand had clean parallel structure → orchestrator-worker fit. Anything more sophisticated would have been over-engineering. The framework: Q1 work structure, Q2 synthesis problem, Q3 operational tolerance.
:::

---

::: sam-arc
**Sam, after the research-on-demand rollout.**

Friday afternoon, a month after launch. Sam pulled the metrics dashboard. The system was processing 200-300 packets per week. Median latency was holding at 3.2 minutes for 5-company packets. Cost per packet was $7-12. Quality, by the AE team's own rubric, was 8.1/10 — modestly better than the single-agent baseline.

The platform team lead pinged Sam: *"What did you learn from the build?"*

Sam thought about it. The biggest lesson wasn't technical. It was the discipline of *not getting fancy*. The literature had every pattern available — hierarchical, peer collaboration, swarm, hybrid combinations. Sam had read all of them. And the right answer for this specific task was the most basic multi-agent pattern: orchestrator-worker.

Sam typed back: *"The Section 6 framework picked orchestrator-worker. I went with it. Worked. Resisting the urge to layer in additional sophistication was harder than implementing the architecture itself."*

The team lead's reply: *"That's the lesson. Document it for the playbook."*

Sam wrote it up over the weekend. Three additions to the platform's design review checklist:

1. **For new multi-agent systems, run M13 framework first. If it doesn't pass, build single-agent.**
2. **For systems that pass M13, run M14 Section 6 framework. Pick one pattern; don't combine.**
3. **Implement workers with full M2-M12 discipline. Multi-agent is *additive* to single-agent rigor, not a replacement for it.**

Two new agents went through this checklist over the next quarter. One was building a contract-comparison system (compare a draft contract to template) — Q1 said "two-document comparison, not parallel exploration"; the framework said don't go multi-agent. The team built single-agent with good tools. Worked.

The other was building a regulatory-research system spanning multiple jurisdictions. Q1 said parallel exploration across independent jurisdictions; framework said orchestrator-worker. The team built it; it shipped on schedule.

Sam noticed the company's overall architecture was getting *simpler*, not more complex, even as more multi-agent systems were being built. The framework was acting as a filter — only the right systems passed; only the simplest patterns got chosen. The deal-research workflow, the customer-success bot, the deal Q&A agent, the research-on-demand packet system, the regulatory-research system. Each one fit-for-purpose. Each one defensible in code review. Each one handling its actual problem with the simplest architecture that worked.

Sam's arc this module: **the architecture matches the structure of the work, and that's the only thing that matters.** Not the architecture's sophistication. Not the agent count. Not the framework's recommendation alone. The structure of the work. Orchestrator-worker for parallel-with-synthesis. Peer collaboration for multiple-framings. Hierarchical for genuine tree depth. Swarm for massive-parallelism-no-synthesis. Match the pattern to the structure; everything else follows.

The book's second half was now well-launched. Two more architecture modules (M15 communication patterns), then operations (M16-17), then the capstone (M18). Six modules left. The single-agent foundations were solid. The multi-agent gate was working. The patterns themselves were now in hand. What remained was the connective tissue: how agents talk to each other (M15), how the system stays reliable in production (M16-17), and how it all comes together at the end (M18).
:::

---

## What's next

Module 15 covers inter-agent communication patterns: structured handoffs (the contract between orchestrator and worker), shared memory and blackboard patterns, voting and consensus mechanisms (when peer agents need to agree), message-passing protocols, and the emergent-behavior controls that distinguish working multi-agent systems from chaos.

After M15, the architecture arc closes. Module 16 covers sagas — the transactional safety pattern for agent systems that take real-world actions across multiple steps. Module 17 covers observability for production agents. Module 18 is the capstone: a complete system pulling together every layer from the book.

For now: take the framework from Section 6. If you have a multi-agent system in production or in design, run it through Q1-Q3 explicitly. If your system uses a pattern that doesn't match the structure of the work — orchestrator-worker for tree-structured tasks, hierarchical for flat tasks, swarm for tasks that need synthesis — the cost is real and the simplification is usually possible. Match the pattern to the structure.

The architectures from M14 are powerful tools when applied to the right problems. M13's gate ensures you reach for them only when justified. M14's selection ensures you reach for the *right one* when you do. Together they're the architectural discipline that turns multi-agent from "the trendy thing we're trying" into "the right answer for the problems that need it."
