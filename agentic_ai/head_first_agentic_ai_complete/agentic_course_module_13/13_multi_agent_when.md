# Module 13 — When Multi-Agent Earns Complexity

::: chapter-opener
<div class="module-num">MODULE 13</div>
<div class="module-title">When Multi-Agent Earns Complexity</div>
<div class="subtitle">Most production failures attributed to "the agent" are actually failures of an over-architected multi-agent system that didn't need to be one.<br>The first multi-agent design decision is whether to be multi-agent at all.</div>
<div class="pages">~38 pages · the architecture choice teams get wrong before they ever write code</div>
:::

::: hook
Monday morning. The platform team lead pinged Sam: *"I want you to design our next-gen customer support system. Take the rest of the quarter. Whatever architecture you think is right."*

Sam pulled up a blank doc and started sketching. The current customer-success bot was a single agent with twenty-something tools. Sam had spent the last several modules hardening it, and it was working well — but the architecture was starting to feel cluttered. The system prompt had grown to handle billing scenarios *and* technical support *and* account management *and* sales-handoff routing *and* compliance constraints. Each domain was its own ramp; the prompt had to teach the agent to navigate all of them.

Sam's first sketch was the obvious one: split the responsibilities. A *billing specialist* agent for billing questions. A *technical support specialist* agent for troubleshooting. An *account specialist* agent for plan management. A *sales-handoff agent* for the cases that needed human escalation. A *router agent* that classified incoming questions and routed to the right specialist. A *coordinator agent* that managed cross-domain conversations.

Six agents. Each focused. Each easier to maintain. Each easier to test. *That's* the future, Sam thought.

Sam was about to start building when something nagged at them. They opened the Anthropic engineering blog — the post on the multi-agent research system — and re-read it. The numbers landed differently this time:

- *15× more tokens than single-agent chat*
- *Best suited for tasks where the value of the outcome outweighs the expense*
- *Less effective for tightly interdependent tasks*
- *Coordination challenges and emergent behaviors*

Sam went back to the sketch and ran a thought experiment. A customer asks: *"My billing has a discrepancy that I think is related to the SSO integration we set up last month."*

The router classifies this as billing. Routes to billing specialist. Billing specialist sees "SSO integration" and realizes this is also a technical question. Hands off — or asks the technical specialist? Or does the coordinator agent re-route? Now there are two specialists looking at the same conversation, neither with full context. They both spawn tool calls. They both fetch the customer's account. The router is still in the loop. Six agents talking past each other for a question a single well-tooled agent answered in two turns yesterday.

Sam crossed out the six-agent sketch. Wrote at the top of the doc: *"the question isn't 'how do I split this into agents?' The question is 'does this task even WANT to be split?'"*

Module 13 was about that question.
:::

---

## What this module is

Modules 2 through 12 built one well-disciplined agent. Hardened loop, careful workflows, well-designed tools, secure protocols, right-sized models, engineered context, managed trajectories, persistent memory, grounded outputs, honest reflection, and boundary discipline. Sam's customer-success bot — singular — embodies all of it.

This module is the structural transition point. The book's second half is about *systems* of agents. M14 covers the specific multi-agent architectures (orchestrator-worker, peer collaboration, swarm patterns). M15 covers inter-agent communication and consensus. But before any of that — before sketching architectures or naming agents — there's a more important question.

**Should you be using multiple agents at all?**

The honest answer, supported by both Anthropic's own production experience and the broader 2025-2026 literature: *most of the time, no.* Most multi-agent systems in production are over-architected single-agent problems that ended up with coordination overhead instead of cleaner separation. The failure mode shows up as latency, cost, brittleness, and a debugging nightmare nobody anticipated.

This module covers:

- **Section 1** — the architecture spectrum: workflows → single-agent-with-tools → multi-agent. Where each lives, where each fails
- **Section 2** — the genuine reasons to go multi-agent (parallel exploration, capability heterogeneity, context isolation)
- **Section 3** — the false reasons (specialization theater, prompt complexity disguised as architecture, "it sounds more sophisticated")
- **Section 4** — Anthropic's research system as the canonical case study: when it earns its complexity, what it cost
- **Section 5** — the decision framework: a structured checklist for "do I need multi-agent?"
- **Section 6** — the cost reality: 15× tokens, coordination overhead, the emergent-behavior tax
- **Section 7** — when single-agent-with-many-tools beats multi-agent (most of the time)
- **Section 8** — Sam's customer support redesign, post-decision

By the end of this module:

- You'll have a clear framework for deciding when multi-agent earns its complexity
- You'll recognize the failure patterns of over-architected multi-agent systems
- You'll understand why "single agent with ten tools" is not multi-agent — and why that distinction is load-bearing
- You'll know when to push back when someone proposes splitting a working single-agent into multiple specialists

M14 will then cover *how* to build multi-agent systems for the cases where you've decided you need to. M15 will cover communication patterns. But M13's job is the harder one: convincing you that the fanciest architecture isn't usually the right one.

::: pullquote
The first multi-agent design decision is whether to be multi-agent at all. Most teams skip this decision. They reach for multi-agent because it sounds sophisticated, then discover six months later that they've built an expensive way to run one model.
:::

---

## Section 1 — The Architecture Spectrum

The Bouchard framing from January 2026 is the cleanest taxonomy I've seen. There's a spectrum of complexity, and your job is to **stay as far left on the spectrum as possible while still solving your problem**:

```
Workflow        Single agent     Multi-agent
(M3)            with tools       system
                (M2 + M4)        (M13+)
←───────────────────────────────────────→
Less complex                      More complex
Lower cost                        Higher cost
Easier to debug                  Harder to debug
More predictable                  More emergent
```

Each tier earns its place when the previous one stops working — *not before*.

**Level 1: Workflows.** A workflow is a sequence of LLM calls connected by code (M3 covered this). The control flow is deterministic. The LLM is making content decisions, not control-flow decisions. A research-brief workflow that does *retrieve → summarize → format* in fixed order is a workflow.

When workflows fail: when the right next step depends on what the model just discovered. A user asks "where is this function used?" — the answer requires running grep, then reading the matching files, then deciding which usages are relevant, then potentially running grep again. The control flow can't be predetermined.

**Level 2: Single agent with tools.** An agent loop with multiple tools. The model decides which tool to call, when, with what arguments. The control flow is decided at runtime by the model. M2's hardened loop is this. Sam's customer-success bot is this.

When single-agent fails: when the task genuinely requires either parallel exploration of separable subspaces, or significantly different capabilities for different sub-problems, or context windows that don't fit even with M7-M8 discipline.

**Level 3: Multi-agent.** Multiple agents, each with their own context window, possibly their own tools, possibly their own model. Coordination happens through structured handoffs, shared memory, or message passing. Anthropic's research system is this.

When multi-agent fails: when the task is tightly interdependent, when the cost can't justify 15× tokens, when latency budgets don't accommodate coordination overhead, when the team can't debug emergent behaviors.

### What "multi-agent" actually means

A common confusion: **a single agent with ten tools is not a multi-agent system.** Multiple capabilities ≠ multiple agents.

Bouchard's distinction is the right one:

- **A tool is a capability** — a calculator, a database query, a web browser, a validator, an API call.
- **An agent is a decision maker** — chooses which tools to use, when, with what context.

If your "specialist agents" are actually one model calling ten different APIs, you have one agent with ten tools — which is M2 territory, not M13. The architecture has the same complexity profile as the single-agent system; calling it "multi-agent" is marketing, not engineering.

What makes something *actually* multi-agent:

1. **Separate context windows.** Each agent has its own context that the others don't see in full. The lead agent passes structured instructions; the subagent works in isolation; the subagent returns structured results.

2. **Independent decision loops.** Each agent runs its own loop — its own model calls, its own tool selections, its own termination decisions. They're not all part of one giant orchestrated trajectory.

3. **Coordination cost.** There's actual communication overhead between agents (handoffs, shared state, message passing) that wouldn't exist in a single-agent design.

When all three are true, you have multi-agent. Otherwise, you have a single agent with structured tool use, which is fine and cheaper and easier to debug.

::: nodumbq
**Q: My system has a "router" that picks one of several "specialist" agents. Is that multi-agent?**

It depends. If the router is a separate LLM call that classifies, hands off, and never sees the specialist's reasoning, and each specialist runs its own independent loop with its own context — yes, that's multi-agent (a hierarchical pattern). If the "router" is just a tool the agent calls to decide what to do next, with everything happening in one loop and one context — no, that's single-agent. The distinction is whether there are genuinely separate decision-making contexts, or just structured branching within one.

**Q: How can I tell if my problem actually needs multi-agent, before I build it?**

Section 5 covers the decision framework formally. The short version: if you can describe your task as "do A, then B, then C" where each step depends on the previous one's output, you want a workflow or single-agent. If you can describe it as "explore N independent dimensions in parallel and synthesize the results," you might want multi-agent. The middle cases — most production tasks — usually want single-agent with good tools.
:::

---

## Section 2 — When Multi-Agent Genuinely Earns Its Place

Three reasons to go multi-agent. Anthropic's own engineering documentation and the broader 2025-2026 literature converge on these.

### Reason 1: Parallel exploration of separable subspaces

The Anthropic multi-agent research system's canonical use case: *"identifying board members from S&P 500 companies in the IT sector."* This task has structure that single-agent can't exploit:

- ~75 companies in the IT sector
- Each company's board can be researched independently
- Researching all 75 sequentially is slow
- Results must be synthesized at the end

A single-agent loop would process companies one at a time. A multi-agent system spawns 75 subagents — one per company — that run in parallel. Each has its own 200K context window for that company's research. The lead agent collects the results. Total wall-clock time: roughly the time of the slowest single subagent, not the sum of all of them.

This is what Anthropic's own data showed: **90.2% improvement on complex breadth-heavy research tasks** when multi-agent vs single-agent. The improvement isn't from "smarter agents" — it's from "agents that can explore in parallel where single-agent has to be sequential."

The key requirement: **the subspaces are genuinely separable.** Researching Company A's board doesn't require knowing what Company B's board looks like. There's no information that needs to flow between subagents during their execution. Each one can complete independently.

When this pattern fits:

- Research tasks across many independent entities
- Comparison shopping across many products
- Code search across many independent files (sometimes — there's a caveat coming)
- Document analysis across many independent documents
- Fact-checking many separable claims

When it doesn't fit:

- Tasks where subagent A needs subagent B's results to know what to look for
- Tasks where the right decomposition isn't known up front
- Tasks where the synthesis is the hard part (and the parallel work is trivial)
- Tasks where any subagent's failure invalidates all the others' work

### Reason 2: Capability heterogeneity

The O'Reilly piece (Feb 2026) framed this well: *"Genuine multi-agent value comes from heterogeneity. Different models with different capabilities operating at different price points for different subtasks."*

If your system genuinely needs:

- Opus 4.7 for high-stakes reasoning *and*
- Haiku 4.5 for high-volume cheap classification *and*
- A specialized model for embeddings *and*
- A code-specific model for code analysis

...and these *can't be unified* (one model handling all of them is too expensive or too imprecise), then multi-agent earns its place. The architecture isn't "multiple agents because we want sophistication" — it's "multiple agents because no single model is the right fit for every step."

Module 6 covered model selection. The advanced version: per-step model selection in a multi-agent system. The orchestrator runs on Opus 4.7 (high-stakes coordination). The fast classifier subagents run on Haiku 4.5 (cheap, parallel). The synthesis agent runs on Sonnet 4.6 (right balance). This is heterogeneous multi-agent, and it's a real design pattern.

When this pattern fits:

- Workloads with genuine cost-quality variance across subtasks
- Cases where one model's strengths don't span the whole task
- Long-running systems where single-model cost would be prohibitive

When it doesn't fit:

- Cases where a single Sonnet 4.6 agent handles all subtasks fine (most cases)
- Cases where the cost difference is rounding error
- Cases where the coordination overhead exceeds the model-selection savings

### Reason 3: Context isolation

The third reason, and the subtlest. Some tasks have content that *can't share a context window*:

- **Adversarial content.** A privileged agent shouldn't read untrusted email bodies (M12's dual-LLM pattern); a quarantined subagent processes them and returns structured summaries.
- **Compliance separation.** PHI data shouldn't enter the same context as marketing analysis. Subagents handle each domain in isolation; only sanitized summaries cross boundaries.
- **Cognitive load.** A subagent given only the relevant information for *its* subtask reasons better than one trying to maintain a giant context with everything every subagent might need.

The third bullet is most underrated. From the Anthropic engineering blog: *"subagents enabling the kind of scaling that a single agent cannot achieve"* refers in part to context isolation — each subagent gets a focused 200K window for its subtask, vs trying to fit everything into one window.

Sam's M7-M8 work was about managing context within a single agent. The multi-agent extension is: *don't manage one giant context; split into multiple focused contexts.* When the task naturally decomposes into subtasks with independent context needs, isolation wins.

When this pattern fits:

- Security boundaries (M12's dual-LLM pattern is a special case)
- Compliance domains that can't co-mingle
- Tasks where each subtask has very different relevant context

When it doesn't fit:

- Tasks where the relevant context overlaps heavily across subtasks
- Tasks where coordination requires lots of cross-context reference
- Cases where the M7-M8 disciplines would have handled the context size

::: brain
A team is building a system that needs to: (a) read 500 customer support tickets per day, (b) categorize each, (c) draft a response, (d) check for policy violations, (e) send. Single agent with five tools, or multi-agent with five specialists?

(Single agent. The flow is sequential — categorize then draft then check then send. There's no parallel exploration; each ticket goes through the steps in order. The "specialists" wouldn't have separate context needs; they'd all need the ticket content, the customer history, the policy. The "policy check" specialist might be a quick LLM-as-judge call, but that's a tool, not an agent. Going multi-agent here would add coordination overhead with no parallel-exploration benefit. M2 + M4 + M12 handles this. The temptation to "split into specialists" is exactly the trap M13 warns about.)
:::

---

## Section 3 — The False Reasons

If those are the genuine reasons, what are the *false* reasons teams reach for multi-agent? Three patterns I see repeatedly.

### False Reason 1: Specialization theater

The most common mistake. The reasoning sounds plausible: *"different concerns should be handled by different specialists."* The system gets sketched: a billing agent, a technical agent, an account agent, etc. Each "specialist" with its own focused prompt.

The problem: if these specialists handle the same conversation about the same customer with the same tools, they're not specialists — they're the same agent with branched prompts. The "specialization" exists in the team's mental model, not in the system's actual behavior. When a conversation crosses domains (which happens constantly in customer support), the multi-agent design adds coordination cost without adding capability.

The Bouchard piece called this out specifically:

> *"Request analysis. Content generation. Structure generation. Syntactic validation. Semantic validation. Spam prevention. Optimization. Security. Scoring. HTML normalization. Migration agents. Even comparison and analysis agents. On paper, it looked clean, specialists doing specialist work. But here, a single agent will work much better because the tasks are tightly coupled and sequential."*

The diagnostic: are your "specialists" doing fundamentally different *kinds* of reasoning, or the same kind of reasoning on different topics? If the latter, they're not specialists. They're prompt branches that didn't need to be agents.

### False Reason 2: Prompt complexity disguised as architecture

Sam's billing-vs-technical-vs-account temptation in the cold open was this. The system prompt had grown to handle multiple domains. The fix *seemed* to be splitting the prompt into multiple agents. But the underlying problem was prompt complexity, not architectural complexity.

The cleaner solution to a long system prompt is usually:

- **Break the prompt into modules and load relevant sections per turn** (M7's just-in-time pattern)
- **Use better tool descriptions** so the agent navigates the domains correctly
- **Add memory of customer-specific preferences** (M9) so the agent doesn't re-derive context
- **Prune the prompt** of accreted edge cases that no longer earn their place

A 4,000-token prompt that handles multiple domains well is *almost always* better than four 1,000-token prompts plus a router and coordinator. The single prompt has full context. The multi-agent version has handoff costs and information silos.

The diagnostic: is your urge to go multi-agent driven by "this prompt is too long" or by "these subtasks genuinely need parallel exploration / different capabilities / context isolation"? If the former, fix the prompt.

### False Reason 3: It sounds more sophisticated

Multi-agent is the current "exciting" architecture. Conference talks feature it. Vendors pitch it. Engineers want to build it. *"We're building a multi-agent system"* sounds more impressive than *"we're building a single agent with good tools."*

This is real and shouldn't be dismissed — engineers are humans, status incentives matter, careers are built. But the cost is real too. Production systems built for sophistication rather than fitness end up with coordination overhead, brittle handoffs, and debugging that nobody anticipated.

The honest framing from the netguru team:

> *"We learned to resist overengineering. Some tasks tempted us to split logic too early — adding agents where a smarter prompt or better tooling would've done the job. We now treat multi-agent setups as a response to complexity, not a default."*

The discipline: **multi-agent earns its place when single-agent fails.** Not before. The motivation should be specific failures of a single-agent design, not the abstract appeal of distributed architectures.

::: gotcha
A pattern I've seen many times: a team builds a multi-agent system, ships it, then over six months gradually consolidates it back to a single agent with tools. The intermediate version usually works *worse* than either endpoint — multi-agent overhead without multi-agent benefit. The lesson: getting the architecture right at the start saves the consolidation cost. Sketch the single-agent version first. If it works, ship it. Only escalate to multi-agent when single-agent demonstrably fails.
:::

---

## Section 4 — The Canonical Case: Anthropic's Research System

A worked example of multi-agent earning its complexity. From Anthropic's June 2025 engineering post and the follow-up production reports.

### What it does

Claude's Research feature (in Claude.ai) handles open-ended research queries: *"What are the major regulatory changes affecting fintech in Q1 2026?"* or *"Compare the compensation packages of CTOs at the top 50 cybersecurity companies."*

These are tasks with three characteristics:

1. **Breadth-heavy.** Many sub-questions to investigate
2. **Parallelizable.** Each sub-investigation is largely independent
3. **High-stakes outcomes.** A research report shapes downstream decisions; quality matters more than cost

This is the genuine fit for multi-agent. Section 2 Reason 1 (parallel exploration) applies cleanly.

### The architecture

From Anthropic's post and subsequent reporting:

- **Lead Researcher agent.** Receives the user's query, develops a research strategy, decides what subagents to spawn and what each should investigate. Saves its plan to memory (M9 cross-run memory) so it doesn't lose track when context fills.
- **Subagents.** Each handles a specific subtopic — a particular company, a particular time period, a particular technical detail. Has its own context window, its own tool access (web search, citation engines), its own iterative loop.
- **Citation/Review agent.** Verifies the synthesized output before final delivery. M11-style critic, lifted into a multi-agent role.

Each subagent is given (per Anthropic's writeup): *clear objectives, output formats, tool usage guidance, and task boundaries.* Without this detail, agents spawn excessive subagents for simple queries, conduct redundant searches, and fail to coordinate effectively. M3 workflow patterns and M11 reflection both apply at the multi-agent level.

### The numbers

Anthropic's published metrics:

- **90.2% improvement** on complex breadth-heavy research tasks vs single-agent
- **15× more tokens** than standard chat interactions
- **Improvement strongly linked to token usage and parallel context windows** — not "smarter" agents, but *more parallel work in independent contexts*

Each number tells you something:

The 90.2% improvement is *task-specific*. It's measured on tasks where multi-agent's strengths apply (breadth-heavy research). On other tasks, the improvement would be smaller or negative. Multi-agent isn't strictly better; it's better *for the right tasks*.

The 15× token cost is *significant*. That's the lower bound on cost; for serious research with many subagents and citation verification, real cost can be 20-30× single-agent. The economic question becomes: is the outcome's value worth this cost? For "research a deal target before a customer call," the answer is sometimes yes. For "answer a routine support question," the answer is no.

The "parallel context windows" framing is the deepest insight. Multi-agent's value isn't model capability; it's the architecture's ability to maintain *N independent 200K context windows in parallel*. A single-agent system has one 200K window. Multi-agent has 75 (or however many subagents). For breadth-heavy tasks, this multiplies the system's effective context by N — where any single-agent system, even with M7-M8 discipline, has just one window to spread across all topics.

### What it cost to build

Anthropic's post is direct about the production challenges:

> *"The transition from prototype to production revealed that prompt engineering becomes significantly more complex in multi-agent environments due to coordination challenges and emergent behaviors."*

Specific issues that took engineering effort to address:

- **Excessive subagent spawning.** Early versions spawned subagents for trivial queries. Required orchestration prompts that taught the lead agent when not to delegate.
- **Redundant searches.** Multiple subagents searching the same content. Required coordination via shared memory and explicit deduplication.
- **Coordination failures.** Subagents producing inconsistent or contradictory results. Required strict output formats and structured handoffs.
- **Reliability for production.** Checkpointing, retry logic, rainbow deployments. Module 16-17 territory; for multi-agent, all of it is harder.

The engineering cost of multi-agent is *not* "spend a few extra weeks on coordination." It's "build a substantial new operational layer for the multi-agent system." Anthropic's team had the resources to do this; many teams don't.

### Where it doesn't fit

Anthropic's own framing — *"less effective for tightly interdependent tasks such as coding"* — is the key constraint. Coding is the prototypical case where multi-agent struggles:

- A code change in one file often requires understanding many other files
- The right decomposition isn't known until you've explored
- Different parts of the codebase share variables, types, conventions that need to be coordinated
- A subagent's "fix" can break what another subagent is working on

For coding, *single-agent with smart tools* (Claude Code's design) outperforms most multi-agent attempts. The exception, which is interesting: Claude Code itself uses sub-agents for *certain bounded tasks* — read-only exploration with the Explore agent (Haiku-based, fast file discovery), specialized roles for parallel-explorable subtasks. The orchestrator stays single-agent; the parallelizable subtasks delegate. This is the heterogeneous-with-context-isolation pattern from Section 2 Reason 2 + Reason 3.

::: pullquote
Multi-agent is the right answer when the task has parallel exploration structure that single-agent can't exploit, capability heterogeneity that one model can't span, or context isolation requirements that coordination has to enforce. It's the wrong answer for everything else.
:::

---

## Section 5 — The Decision Framework

A structured checklist. Run through this before writing any multi-agent code.

### Question 1: Is the task parallelizable?

**Strong yes:** The task has N independent sub-investigations, each requiring its own context. Synthesis happens at the end. Examples: "Research these 50 companies." "Analyze these 200 documents." "Verify these 100 claims."

**Soft yes:** The task has some parallelizable parts and some sequential parts. The parallel parts dominate the cost. Examples: "Research a customer's account history across 5 sources, then write a brief."

**No:** The task is tightly sequential. Each step depends on the previous. Examples: "Triage this support ticket and respond." "Debug this code by exploring the call graph from this entry point."

If "no," stop. Multi-agent's primary advantage doesn't apply. Single-agent with workflow structure is better.

### Question 2: Does the value of the outcome justify 15× token cost?

**Yes:** The output drives a high-stakes decision. Examples: A research report informing an investment decision. A pre-call brief for an enterprise customer. An audit of regulatory exposure.

**No:** The output is routine, low-stakes, or high-volume. Examples: Customer support replies. Email categorization. Daily status summaries.

If "no," stop. The multi-agent premium isn't justified. M2 + M4 + M11 produces high-quality single-agent output at a fraction of the cost.

### Question 3: Can you specify clear subagent boundaries?

**Yes:** Each subagent has a focused, well-defined task with clear inputs and outputs. The orchestrator can write a one-paragraph mandate for each subagent. Subagents don't need to coordinate with each other during execution.

**Unclear:** You can imagine the subtasks but they're hard to specify precisely. The orchestrator might need to reshape mandates as it learns. Subagents would need to interrupt each other.

If "unclear," reconsider. Multi-agent systems where subagent boundaries are fuzzy produce coordination chaos. The cleaner version is usually a single agent that handles the exploration in a unified context.

### Question 4: Can your team operate the resulting system?

**Yes:** Your team has built and operated complex async systems before. You have observability for distributed traces. You have engineers who can debug multi-agent emergent behaviors.

**No:** Your team has shipped a few single-agent systems. Your observability is logs. Your debugging strategy is "read the trace."

If "no," reconsider. Multi-agent in production is operationally hard. Anthropic's team has resources most teams don't. The right call is often "ship single-agent, add observability infrastructure, then revisit multi-agent in 6 months."

### Question 5: Have you tried single-agent first?

**Yes, and it specifically failed because of [parallelism / capability / context isolation]:** You have evidence. Multi-agent is justified.

**No, but multi-agent seems right:** Stop. Build single-agent first. Identify the specific failure modes. *Then* decide if multi-agent addresses them.

If you skip this step, you'll over-architect. Almost guaranteed.

### The decision

If you answered "yes" to questions 1-4 and "yes" to question 5 with specific evidence:

→ **Multi-agent earns its place.** Proceed to M14 for the architectures.

If you answered "no" or "unclear" to any:

→ **Build single-agent.** Use M2's hardened loop, M3's workflow patterns where appropriate, M4-M12's discipline. Re-evaluate when you have specific evidence of single-agent failure.

::: brain
Run the framework on the Anthropic Research system. Q1: parallelizable (yes — research many independent subtopics). Q2: high-value (yes — research informs important decisions). Q3: clear boundaries (yes — one company per subagent). Q4: team capability (yes — Anthropic has the resources). Q5: tried single-agent (yes — single-agent had explicit limitations on breadth-heavy tasks). Multi-agent earns its place.

Now run it on Sam's customer support redesign. Q1: parallelizable (no — most support questions are sequential). Q2: high-value (no — most are routine). Q3: clear boundaries (unclear — domains overlap). Q4: team capability (yes — Sam can handle it). Q5: tried single-agent (yes, and it works). The framework says: don't go multi-agent. The single-agent design with good tools is the right architecture.
:::

---

## Section 6 — The Cost Reality

Worth a dedicated section because the cost picture is what most teams miss when they decide to go multi-agent.

### Direct costs

The 15× token multiplier is the headline. Where does it come from?

```
Single-agent task:
  Input tokens (prompt + tool results): ~30K total
  Output tokens (reasoning + response): ~5K
  Total: ~35K tokens

Multi-agent task (same outcome, different architecture):
  Lead agent input/output: ~10K
  6 subagents × ~30K input + ~10K output each: ~240K
  Coordination handoffs (lead reading subagent results): ~50K
  Final synthesis: ~10K
  Critic/review: ~30K
  Total: ~340K tokens
```

That's a 10× multiplier in this example. Anthropic's reported 15× is realistic for the research system specifically; some multi-agent designs hit 20-30×.

The dollar implication varies by model. With Sonnet 4.6 ($3/M input, $15/M output), 35K-token single-agent task costs roughly $0.30. The 340K-token multi-agent equivalent costs roughly $3-5 depending on input/output split.

For 1,000 daily executions: $300/day vs $4,500/day. Same outcome quality on the right tasks (90% improvement on Anthropic's benchmarks). But the cost difference is real.

### Latency costs

Multi-agent has parallelism, so wall-clock time often *improves* despite token costs. But not always:

- **Coordination overhead.** Lead agent waits for all subagents before synthesizing. Total time = max(subagent times) + coordination + synthesis. Often longer than single-agent's sequential time, despite parallelism.
- **Critical path dominance.** One slow subagent stalls the whole system. Without bounded latency per subagent, p99 latency can be much worse than single-agent.

The trade-off: multi-agent often has *better median latency* (parallelism wins) but *worse tail latency* (any subagent's slow path blocks everything).

For interactive applications (chat, support), tail latency matters more than median. For batch applications (research reports, overnight analysis), median wins. Different fits.

### Operational costs

Less visible but often dominant:

- **Observability.** Tracing a multi-agent execution requires distributed tracing. The trace shows N parallel timelines, handoffs between them, and synthesis. Standard log-based observability falls short.
- **Debugging.** When something goes wrong, "which agent failed" is the first question. Then "what did it see when it failed." Then "did the failure cascade through coordination." Multi-agent debugging is fundamentally harder.
- **Testing.** Single-agent unit tests check the agent's behavior. Multi-agent tests check coordination patterns, handoff fidelity, emergent behaviors. The test surface is much larger.
- **On-call burden.** When multi-agent fails in production, the recovery is more complex. Module 16-17 cover this; for now, the point is *multi-agent operations is more work than single-agent*.

The Anthropic post itself is honest about this: *"checkpointing, retry logic, and rainbow deployments for safety"* — production multi-agent requires substantial operational engineering.

### Emergent behavior costs

The hardest cost to anticipate. Multi-agent systems develop behaviors not present in any individual agent:

- **Cascading failures.** Subagent A fails; lead agent retries; retry produces different results that confuse subagent B's reasoning; subagent B fails differently than it would have alone.
- **Coordination drift.** Over many interactions, the agents develop a "style" of coordination that diverges from what the prompts specify. Hard to detect; harder to correct.
- **Echo chamber effects.** Subagents that read each other's outputs reinforce each other's biases. The synthesis amplifies what they all agreed on, even when they were all wrong in the same way.

The MAR paper from Module 11's research surfaced this: *"self reflections tend to repeat earlier misconceptions and do not introduce new reasoning paths on difficult examples."* Apply the same finding at the multi-agent level — agents in the same system have correlated blind spots.

::: postmortem
**The Multi-Agent System That Got Slower Over Time**

A team deployed a multi-agent customer-support system: a router agent, three specialist agents (billing, technical, account), and a coordinator that managed multi-domain conversations. It worked well at launch. Six months later, p50 latency had drifted from 4 seconds to 9 seconds. p99 from 12 to 30. Costs were 4× the original projection.

The team pulled the traces. The pattern:

- The router was making slightly different categorization decisions over time as the prompt accreted edge-case handling
- Categorization mistakes meant requests bounced between specialists more often
- The coordinator was spending more time mediating between specialists
- Each specialist's prompt had grown to handle the bounce cases, getting longer
- The longer prompts produced slower decisions, more handoffs, longer trajectories

The fix: consolidation. The team merged the three specialists into one agent with all the tools. Removed the router (the consolidated agent could classify itself). Removed the coordinator (no specialists to coordinate). The new single-agent system handled the same workload with median latency back to 3.2 seconds and costs at 30% of the multi-agent peak.

The original multi-agent design wasn't wrong at launch. It was wrong *for what the system became*. The drift compounded. Without active simplification, the system grew more complex monotonically until consolidation became the only fix.

**Lesson:** multi-agent designs that don't actively earn their complexity become liabilities over time. Periodic architecture review — "do we still need this to be multi-agent?" — is the only defense.
:::

---

## Section 7 — When Single-Agent-with-Many-Tools Wins

The pattern most teams should reach for first. A single agent with carefully designed tools, hardened loop, good context engineering, all the disciplines from M2-M12.

This is *not* a limitation. It's the right architecture for most production workloads.

### Why it usually wins

**Lower coordination overhead.** No handoffs. No shared state to manage. No multi-agent emergent behaviors. The agent has full context for every decision.

**Cheaper.** No token-multiplier penalty. A 35K-token execution is a 35K-token execution.

**Easier to debug.** One trace. One prompt. One model's behavior to understand. When something fails, the failure is local.

**Easier to evolve.** Adding a new capability is "add a new tool and update the prompt." Adding a capability to multi-agent is "decide which agent owns it, update that agent's prompt, update the coordinator's routing logic, possibly add new handoff patterns."

**More capable than people expect.** A modern Claude agent (Sonnet 4.6 or Opus 4.7) with 15-20 well-designed tools handles enormous complexity. The "ceiling" of single-agent is much higher than teams assume.

### Patterns that look multi-agent but are single-agent done well

**Pattern 1: Sub-tools that wrap LLM calls.** A "summarize" tool that internally calls an LLM. A "classify" tool that calls a cheaper model. The agent calls these like any other tool. Multiple LLM calls happen, but there's one decision-making agent. Heterogeneous-but-not-multi-agent.

```python
@tool
async def summarize_document(text: str) -> dict:
    """Summarize a long document. Returns a structured summary."""
    # Internal LLM call to a cheaper model
    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"Summarize:\n{text}"}],
    )
    return {"summary": response.content[0].text}


@tool
async def classify_content(text: str, categories: list[str]) -> dict:
    """Classify content into one of the provided categories."""
    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": f"Classify:\n{text}\n\nCategories: {categories}"}],
    )
    return {"category": response.content[0].text.strip()}
```

The orchestrating agent uses these tools alongside its other tools. From the agent's perspective, they're tools. Internally, they're LLM calls. Cost is heterogeneous (Haiku for cheap subtasks, Sonnet for the main agent). Architecture is single-agent.

**Pattern 2: Workflow steps wrapped as tools.** Pre-defined sequences of operations exposed as a single tool to the agent.

```python
@tool
async def research_company(company_name: str) -> dict:
    """Research a company. Internally orchestrates lookup, analysis, summary."""
    # This is M3's prompt-chaining workflow exposed as one tool
    profile = await fetch_company_profile(company_name)
    competitors = await analyze_competitors(profile)
    summary = await synthesize_summary(profile, competitors)
    return {"profile": profile, "competitors": competitors, "summary": summary}
```

The workflow runs deterministic internally; the agent calls it as one tool. M3 + M4 patterns combined.

**Pattern 3: M11's reflection within a single agent.** Generate, critique, revise — all within one agent's reasoning. The "critic" isn't a separate agent; it's a tool the agent calls or a structured prompt section. M11 territory, single-agent architecture.

### When this isn't enough

The cases (genuinely):

- **The exploration must be parallel** — Section 2 Reason 1
- **Multiple genuinely different models are required** — Section 2 Reason 2 (and where Pattern 1 above isn't enough)
- **Context boundaries must be enforced** — Section 2 Reason 3 (especially the M12 dual-LLM pattern when stakes warrant)

These are the cases where M14's multi-agent architectures earn their place. Everything else: single-agent.

::: nodumbq
**Q: Don't multi-agent systems scale better than single-agent?**

It depends on what you mean by "scale." Multi-agent scales better in *parallelism* — the ability to do N things simultaneously. Single-agent scales better in *throughput* — the cost-effective handling of many independent requests. Most production "scale" is throughput, not parallelism. A million customer support tickets per day is a throughput problem, and single-agent handles it cheaper. A research task that needs to investigate 100 things in parallel is a parallelism problem, and multi-agent handles it faster (at higher cost).

**Q: What about all the multi-agent frameworks (AutoGen, CrewAI, LangGraph multi-agent)? Don't they make it easy?**

Easier than implementing from scratch, yes. They handle some coordination boilerplate. But "easier to write the code" doesn't change the cost model, the latency profile, or the operational complexity. A LangGraph multi-agent system still costs 15× more tokens than a single-agent equivalent. The framework's value is reducing the cost of *building* the system; it doesn't reduce the cost of *running* the system. Tools matter, but the architectural decision matters more.
:::

---

## Section 8 — Sam's Customer Support Redesign

Returning to the cold open. Sam ran the framework from Section 5:

- **Q1 Parallelizable?** No. Most support tickets are sequential — read, classify, respond, possibly escalate.
- **Q2 High-value?** No. Most tickets are routine. The rare high-stakes ones (compliance, legal) get escalated to humans regardless.
- **Q3 Clear subagent boundaries?** Unclear. Domains overlap heavily; a billing question is often also a technical question.
- **Q4 Team can operate it?** Yes, but unnecessary capability.
- **Q5 Tried single-agent?** Yes — and it works (the M9-M12 hardened version).

The framework verdict: **don't go multi-agent.** Sam's six-agent sketch from Monday morning was specialization theater.

### What Sam built instead

A single-agent system with carefully designed tools and a structured prompt:

```python
SUPPORT_AGENT_TOOLS = [
    # Customer context (loaded by harness, not a tool)
    # ...

    # Information gathering
    fetch_account_state,
    fetch_billing_history,
    fetch_recent_tickets,
    search_knowledge_base,

    # Diagnostics (technical)
    check_integration_status,
    run_diagnostic_test,
    fetch_error_logs,

    # Account actions (with M12 confirmation gates)
    update_billing_method,  # Confirmation required
    apply_credit,            # Confirmation required + supervisor approval

    # Escalation
    create_handoff_to_human,
    create_handoff_to_sales,

    # Internal workflows (Pattern 2 from Section 7)
    investigate_billing_discrepancy,  # Wraps a M3 prompt-chain
    diagnose_integration_issue,       # Wraps a M3 routing workflow
    summarize_ticket_history,         # Wraps an LLM summarization

    # Reflection / verification
    verify_response_against_policy,   # M12 output check
]
```

The system prompt gives the agent its scope (customer success), its tone (helpful but bounded by policy), the structure of its tools, and explicit guidance on when to escalate.

### What Sam's design preserved from the multi-agent sketch

The instinct to "specialize" wasn't entirely wrong. It was the wrong *implementation*. Sam preserved the underlying separation:

- **Specialization in tools, not agents.** The diagnostic tools, the billing tools, the account tools — each is focused on its domain. The agent calls the right tool; no separate "agent" needed.
- **Specialization in prompts via structured guidance.** The system prompt has sections for billing scenarios, technical scenarios, account scenarios — structured guidance the agent uses to decide its approach. One prompt; multiple "modes" of operation.
- **Specialization in handoffs.** When the conversation needs a human, the agent picks the right handoff target (sales, technical lead, compliance) and creates a structured handoff. Single agent; multi-channel routing.

### What Sam's design avoided

- No router agent (the main agent handles classification implicitly)
- No coordinator agent (single context handles cross-domain conversations)
- No specialist agent silos (the agent has all tools and all guidance available)
- No coordination overhead between agents (because there are no other agents)
- No multi-agent emergent behaviors

### The metrics

Three months after rollout:

| Metric | Old single-agent | Multi-agent (sketch) | New single-agent |
|---|---|---|---|
| Median latency | 3.1s | (estimated 6-8s) | 2.4s |
| Cost per ticket | $0.04 | (estimated $0.50-0.80) | $0.04 |
| Resolution rate | 71% | (unknown) | 78% |
| Escalation accuracy | 89% | (unknown) | 94% |

The new single-agent design was *better* than the old single-agent design (which had ratty prompt accretion from M9-M12 work). The multi-agent design Sam almost built would have been worse on every measure that mattered to customers.

Sam wrote a one-line lesson: *"the best thing about the multi-agent system was the discipline I applied when deciding not to build it."*

::: code-exercise
**Exercise 13.1 — Multi-agent decision audit.**

Pick a system you're working on or considering. Run the Section 5 framework explicitly:

1. Write down the answers to Q1-Q5 with specific evidence.
2. If your answers say "single-agent," and you were considering multi-agent, document why you were tempted. Was it specialization theater? Prompt complexity? Sophistication appeal?
3. If your answers say "multi-agent," document the specific failure modes of single-agent that justify it. Which subtasks are parallelizable? Why isn't a single-agent's context window enough?
4. If you've already built a multi-agent system: imagine consolidating to single-agent. What would break? Are those breaks fundamental, or could they be handled with M2-M12 disciplines?

The honest answers to these questions matter more than any framework's verdict. The framework is a structured way to surface the honest answers.
:::

---

## Section 9 — The Framework

Adding architectural discipline before reaching for multi-agent:

**1. Default to single-agent.** Start with M2's hardened loop, M3's workflow patterns where appropriate, M4-M12's discipline. The architecture spectrum runs left-to-right; stay left as long as possible.

**2. Recognize the false reasons.** Specialization theater (different topics aren't different specializations). Prompt complexity disguised as architecture (long prompt fixes, not multi-agent). Sophistication appeal (resume value isn't system value).

**3. Recognize the genuine reasons.** Parallel exploration of separable subspaces. Capability heterogeneity that one model can't span. Context isolation that coordination must enforce. These are the cases — and only these — where multi-agent earns its place.

**4. Run the decision framework.** Five questions: parallelizable, high-value, clear boundaries, operable team, single-agent tried. All five "yes" → multi-agent. Any "no" or "unclear" → single-agent.

**5. Account for the full cost.** 15× tokens (sometimes 20-30×). Coordination latency. Operational complexity. Emergent behavior tax. The "easy multi-agent framework" doesn't change any of these.

**6. Use single-agent-with-many-tools patterns.** Sub-tools that wrap LLM calls (heterogeneous capability without multi-agent overhead). Workflow steps wrapped as tools (M3 + M4 + M5 inside what looks like a single tool). Reflection within a single agent (M11 without multi-agent infrastructure).

**7. Consolidate over time.** Multi-agent designs accrete. Periodic architecture review — "do we still need this to be multi-agent?" — is the only defense against consolidation pressure compounding into operational liability.

**8. When you do go multi-agent, do it deliberately.** M14 covers the architectures. The discipline from M2-M12 applies to every individual agent in a multi-agent system. The complexity is additive, not substitutional.

The discipline this enforces: **multi-agent is a deliberate response to specific failures of simpler architectures.** Not the default. Not the aspiration. The escalation when escalation is justified.

---

## Recap: Module 13 in eight bullets

::: bullet-points
- The architecture spectrum runs from workflows (M3) to single-agent-with-tools (M2 + M4) to multi-agent (M13+). Stay as far left as possible while still solving your problem. Each tier earns its place when the previous one demonstrably fails.
- Multi-agent is *not* "single agent with multiple capabilities." Multi-agent has separate context windows, independent decision loops, and explicit coordination overhead. A single agent with ten tools is M2 territory, not M13.
- Genuine reasons for multi-agent: parallel exploration of separable subspaces (Anthropic's 90.2% lift on breadth-heavy research), capability heterogeneity that one model can't span, context isolation that coordination must enforce.
- False reasons for multi-agent: specialization theater (different topics aren't different specializations), prompt complexity disguised as architecture (long prompts have prompt fixes, not architecture fixes), sophistication appeal (it sounds more impressive than it is).
- The Anthropic Research system is the canonical case where multi-agent earns its place: parallelizable (research many entities), high-value outcome, clear subagent boundaries, team capability for the operational complexity. Even Anthropic acknowledges multi-agent is "less effective for tightly interdependent tasks such as coding."
- The cost reality: 15× tokens minimum (sometimes 20-30×), coordination overhead, harder observability and debugging, operational complexity that scales with agent count, emergent behaviors that compound failures.
- Single-agent with carefully-designed tools handles most production workloads better than multi-agent. Sub-tools that wrap LLM calls, workflow steps wrapped as tools, M11 reflection within a single agent — these patterns capture multi-agent's apparent benefits without its actual costs.
- The decision framework: parallelizable + high-value + clear boundaries + operable team + single-agent tried-and-failed. All five "yes" → multi-agent. Any "no" → single-agent. Sam's customer support redesign hit "no" on three of five; the right answer was a better single-agent, not multi-agent.
:::

---

::: sam-arc
**Sam, after the redesign decision.**

Tuesday morning. Sam walked into the platform team standup with a one-page architecture doc instead of the multi-agent sketch they'd planned to build. The team lead read it.

"You're saying we don't need multi-agent."

"I'm saying multi-agent doesn't help us here. The Section 5 framework hits no on parallelism, no on cost-vs-stakes, and unclear on boundaries. The current single-agent works. The cleaner version of the current single-agent will work better."

"What about the prompt complexity? You said the prompt was getting unwieldy."

"It is. But that's a prompt problem with prompt fixes — modular sections loaded just-in-time, better tool descriptions, customer-specific memory loading. Not an architecture problem with architecture fixes."

The team lead read the doc again. "OK. Build it. But document the decision. I want this written up so the next team member who pitches multi-agent has to argue against the framework you used."

Sam wrote it up that afternoon. The decision doc became required reading for new platform team members. Sam noticed two things over the next quarter:

First: every time someone proposed multi-agent for a new system, they had to walk through the framework first. Most proposals didn't survive Q1 (parallelizable). The team's overall architecture got simpler over time, not more complex.

Second: the *one* multi-agent system the team did build during the quarter — a research-on-demand feature for the sales team — was deliberately multi-agent for the right reasons. Parallelizable across many target companies. High-value outcome. Clear subagent boundaries. Team capability for operations. It earned its complexity, and it shipped working.

Sam's arc this module: **the best architecture decision is often the one not to make.** M2-M12 had been about layering disciplines onto a single agent. M13 was about resisting the urge to layer additional structural complexity on top of the disciplines that were already working. The customer support system Sam built was a *consolidation* — pulling specialization back into a single agent with better tools, better prompts, better memory — rather than an *expansion* into specialists.

The book's first half ended with Sam shipping a single agent that worked. The book's second half started with Sam recognizing that "ship a system of agents" was usually the wrong follow-up question. The right question was: "what does this task actually need?"

Friday afternoon. The customer support redesign was running. The team had pushed Sam's framework into the platform documentation. The decision document was getting cited in design reviews across the company. Sam closed the laptop. Three more modules in the multi-agent arc — but Sam now knew when those modules' patterns *applied* and when they *didn't*. That was the M13 deliverable.
:::

---

## What's next

Module 14 covers the multi-agent architectures themselves: orchestrator-worker (the Anthropic research system pattern), peer collaboration (when multiple agents need to negotiate), hierarchical (lead → middle managers → workers), and swarm (many parallel agents with light coordination). Each architecture's strengths, costs, and the production patterns for making each work.

Module 15 covers inter-agent communication: structured handoffs, shared memory patterns, voting and consensus mechanisms, and the emergent-behavior controls that distinguish working multi-agent systems from chaos.

After M15, the architecture arc closes. M16-17 are operations (sagas for transactional safety, observability for production). M18 is the capstone.

For now: take the framework from Section 5. Run it on a system you're building or considering. The honest answer to those five questions is the most important architectural decision you'll make. Most of the time the answer is single-agent. When it isn't, M14-M15 await — but only when the answer is genuinely "yes." Multi-agent earns its complexity; until then, M2-M12 is enough.
