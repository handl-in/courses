# Module 13 Outline — Multi-Agent: The Honest Trade-offs

::: chapter-opener
<div class="module-num">MODULE 13 — OUTLINE</div>
<div class="module-title">Multi-Agent: The Honest Trade-offs</div>
<div class="subtitle">15× the tokens. 80% of performance variance from token use alone.<br>"Just add more agents" is the most expensive bug in the field.</div>
<div class="pages">Target length: ~36 pages</div>
:::

## What this module is

This is the contrarian module that earns the right to teach the next two. Multi-agent has the largest hype-to-reality gap in the agent field. Anthropic's own research published the canonical numbers: their multi-agent research system outperformed single-agent Opus by **90.2%** on internal evaluations — at **15× the tokens**, with token usage explaining **80% of performance variance**. That's the math everyone quotes. Almost nobody quotes the rest of the same paper: multi-agent systems are *less* effective for tightly-interdependent tasks like coding, the orchestration is hard, and Anthropic explicitly says they're "best suited for tasks where result value far exceeds cost."

This module covers when multi-agent genuinely wins, when it's expensive theater, and the decision framework for picking. It's the necessary throat-clearing before Modules 14 (architectures) and 15 (swarms). By the end the reader can defend — to a skeptical engineering manager — exactly why a particular task should or shouldn't be multi-agent.

::: hook
"You give the agent harder problems. It struggles. The thought arrives: 'maybe two agents would be better.' Then three. Then a supervisor coordinating five workers. By the time the architecture diagram has a dozen boxes, the bill has gone up 15× and the quality has gone up 7%. You needed better tools, not more agents."
:::

---

## Section 1 — The Numbers Everyone Should Have Memorized (≈4 pages)

The Anthropic multi-agent research system, June 2025 engineering blog. Internal eval. The findings:

- **+90.2%** improvement over single-agent Opus on the BrowseComp eval
- **~15× more tokens** vs standard chat use
- **~80% of performance variance** explained by token usage alone
- **Strongly linked to ability to spread reasoning across multiple independent context windows**
- **Less effective for tightly-interdependent tasks like coding** (where serial reasoning matters more than parallel breadth)
- **Best suited for tasks where result value far exceeds cost**

These are stunning numbers in both directions. If you cherry-pick the +90.2%, multi-agent is the future. If you cherry-pick the 15× cost, it's a disaster. Both are real. The answer is: it depends on what the task is worth.

A simple economic test: **if the task's value-of-success is greater than 15× the single-agent cost, multi-agent wins on expected value. Otherwise it loses.**

For Anthropic's research feature, this works: a deep research report a customer pays for, that takes a person hours to assemble manually, is easily worth $5-50 of API cost. For "summarize this email" it doesn't: the task value is sub-$0.01, and 15× still rounds to nothing.

::: pullquote
"Multi-agent uses 15× the tokens" is not an indictment. "Multi-agent uses 15× the tokens for a task that's worth less than the original 1×" is.
:::

::: nodumbq
**Q: Does the 15× number apply to all multi-agent systems, or just Anthropic's research one?**

It's a representative ratio for **orchestrator-worker** systems with parallel exploration. Different patterns have different multipliers. Sequential pipelines might be 2-4×. Debate patterns can be 5-10×. Group chats can be 10-30×. The 15× is a useful default to remember; the actual ratio depends on architecture, agent count, and how much the agents talk to each other.

**Q: Doesn't model improvement reduce this gap?**

It changes the absolute cost, not the multiplier. Better/cheaper models reduce the dollar cost of both single-agent and multi-agent setups proportionally. The 15× ratio is structural — it comes from agents independently exploring (which is the source of the gain) and accumulating context across coordination (which is the source of the cost).
:::

---

## Section 2 — Where Multi-Agent Genuinely Wins (≈5 pages)

Five categories where the math works.

**1. Parallelizable research / exploration.** The Anthropic case. One question, many independent sub-questions, each researchable in parallel. Token spend is high; quality gain is large. Worth it for high-value tasks.

Examples:
- "What are the top three competitors to X, with detailed analysis of pricing, feature gaps, and customer sentiment?"
- "Find primary sources on event Y from at least 5 different perspectives."
- Deep research reports, due-diligence packages, literature reviews.

**2. Genuinely specialized expertise.** Different sub-tasks need different model personas, tools, or context. A code-review system: security specialist + performance specialist + style specialist + correctness specialist. Each has a focused prompt and tool set. The orchestrator's job is to assign and merge.

When this isn't multi-agent: when "specialization" is just "different prompt for the same model with the same tools." That's a workflow (Module 3 routing or sectioning).

**3. Isolated context for security or scale.** A sub-agent processes untrusted external content (web pages, customer documents, emails). The sub-agent has minimal tools and can't escalate. The main agent never sees the untrusted content directly — only the sanitized output. This is sub-agent isolation as a security pattern (cross-references Module 5 MCP, Module 12 guardrails).

**4. Long-horizon tasks that exceed a single context window.** Even with 1M token contexts, some tasks accumulate so much state that fitting it in one window becomes impractical. Distributing state across sub-agents (each with their own window) is sometimes the only viable approach.

**5. Adversarial / critique cycles.** Debate, jury-of-many, writer/critic loops. The structural diversity of multiple agents catches errors that any single agent would miss. Module 11 covered intra-agent reflection; this is the inter-agent version.

::: brain
Look at the five categories. Which two are the strongest cost justifications? Which one is the most commonly **mis-applied** as multi-agent when it shouldn't be?

(Strongest: parallelizable research and isolated-context-for-security — the math is clear. Most mis-applied: "specialized expertise" — most teams call any prompt variation "specialization" and conclude they need multi-agent. They usually need a workflow.)
:::

---

## Section 3 — Where Multi-Agent Loses (≈5 pages)

The cases where adding agents makes things worse, not better.

**1. Tightly-interdependent tasks (coding being the canonical example).** Anthropic explicitly found multi-agent less effective for tasks like coding. Why: each step depends on the exact state of previous steps. Splitting across agents introduces translation loss between them. Coding agents work better as single agents with strong tools (Cursor, Claude Code) — these *use* sub-agents for narrow sub-tasks (search, file reads) but the main reasoning loop is one agent.

**2. Low task value.** Anything where the unit economics don't allow 15× the per-call cost. Customer-support classification at $0.0003 per call cannot become a $0.005 multi-agent system without breaking the business model.

**3. Latency-sensitive flows.** Multi-agent adds coordination overhead. Even with parallel execution, the "wait for all sub-agents to return" plus "synthesize" plus "possibly re-dispatch" is measured in tens of seconds, not single digits. For interactive flows where the user is waiting, this is unacceptable.

**4. Unbounded growth.** Sub-agents that spawn sub-agents that spawn sub-agents. Each layer adds cost. Without a hard cap, runaway is plausible. We covered the "sub-agent cascade that 10x'd the bill" postmortem in Module 7; the same dynamic at higher scope is worse here.

**5. Communication-fragile tasks.** When the natural decomposition has high information density between steps that doesn't compress well. The supervisor in a hierarchical pattern compresses each specialist's rich output into a summary for the next step, and meaningful information is lost in translation. (The April 2026 "Multi-Agent in Production" piece names this "translation/paraphrase loss at the center.")

**6. Single-skill tasks.** If the entire task could be done by one capable agent with good tools, adding agents introduces coordination overhead with no offsetting benefit. The coordination IS the cost; the cost has to be earned.

::: postmortem
**The Customer Service "Crew" That Ran 4× the Cost for Marginal Gain**

A team built a CrewAI-style multi-agent system for customer support: triage agent, billing specialist, technical specialist, escalation agent. Marketed as "specialized expert team."

Reality: the triage agent was 95% accurate. The "specialists" were just different prompts on the same model with the same tools. Most queries went through 2-3 agents. The supervisor's translation between agents lost detail (specialist outputs got compressed into one-line summaries for the next agent). End-to-end accuracy was 1.8 points higher than a single well-prompted agent — at 4× the cost and 3× the latency.

Fix: switched to single-agent with conditional routing within the agent (Module 3 routing pattern). Same tools. Better prompt for the routing logic. Quality matched the multi-agent version at 25% of the cost.

**Lesson:** "specialization" via different prompts on the same model is a workflow pattern, not multi-agent. Don't pay multi-agent costs for workflow benefits.
:::

::: gotcha
The "agent envy" anti-pattern: teams add agents because the architecture diagram looks more sophisticated, makes for better demos, sounds better in fundraising decks. None of these are reasons that survive contact with the bill.
:::

---

## Section 4 — The Decision Framework (≈5 pages)

When you should consider multi-agent, in order of decisiveness.

**Step 1: Can it be a single LLM call?** Most "AI features" can. Don't even use a workflow if you don't need to.

**Step 2: Can it be a workflow?** Reach for the five workflow patterns (Module 3) before reaching for an agent. Most multi-step tasks decompose this way.

**Step 3: Can it be a single agent with great tools?** Most "agent" use cases stop here. Coding agents (Cursor, Claude Code), customer support agents, research-lite agents — single agent + great tools is the default.

**Step 4: Does it meet ALL of the multi-agent criteria?**

- ☐ Task is parallelizable (sub-tasks are largely independent)
- ☐ Task value > 15× single-agent cost
- ☐ Latency budget allows multi-second coordination
- ☐ Sub-tasks need genuinely different tools/personas/contexts (not just different prompts)
- ☐ You have eval infrastructure to know whether multi-agent is actually winning
- ☐ You have observability to debug when it's not

If ANY of these are no: don't go multi-agent. The combination is what justifies the cost.

**Step 5: If yes, which architecture?** That's Module 14.

We provide a decision flowchart that the reader can apply to their own task:

```
┌────────────────────────────────────┐
│  Can a single LLM call do it?      │
└──────────┬──────────┬──────────────┘
       Yes │       No │
           ▼          ▼
    ┌─────────┐  ┌────────────────────┐
    │   DO    │  │  Decomposes into   │
    │  THAT   │  │  known steps?      │
    └─────────┘  └────┬──────────┬────┘
                  Yes │       No │
                      ▼          ▼
              ┌──────────┐  ┌────────────────────────┐
              │ WORKFLOW │  │  Tool selection IS the │
              │ (M3)     │  │  hard part?            │
              └──────────┘  └─────┬──────────┬───────┘
                              Yes │       No │
                                  ▼          ▼
                          ┌────────────┐  ┌─────────────┐
                          │ AGENT      │  │ Same answer:│
                          │ (M2)       │  │ AGENT (M2)  │
                          └────┬───────┘  └─────────────┘
                               │
                       Hits ceiling on
                       breadth/parallelism?
                               │
                            Yes ▼
                          ┌────────────┐
                          │ All M-A    │
                          │ criteria   │  ← Step 4
                          │ satisfied? │
                          └────┬───────┘
                            Yes ▼
                          ┌────────────┐
                          │MULTI-AGENT │  ← Module 14
                          └────────────┘
```

::: code-exercise
**Exercise 13.1 — Audit a multi-agent codebase.**

Given a sample multi-agent system (we provide one — the customer-support "crew" from the postmortem in Section 3), apply the decision framework. For each agent: is it earning its cost? What would replacing this agent with a workflow step look like? Estimate the cost reduction.

Bonus: estimate quality impact. (Often: minimal.)
:::

---

## Section 5 — Sub-Agents vs Multi-Agent (≈3 pages)

A distinction the field is converging on. Worth being precise about.

**Sub-agent.** A focused agent invoked by a parent agent for a narrow task, with isolated context, returning a result. Examples: Anthropic's Sub-Agents API, Cursor's "agent that reads files for me," Claude Code's task subagents. Sub-agents have:
- Limited tool set (often just one or two)
- Isolated context window (don't see parent's full context)
- Bounded lifespan (single task, return result)
- Parent treats them like an expensive function call

**Multi-agent system.** Multiple peer (or hierarchical) agents that coordinate over time, often with shared state, ongoing communication, mutual reasoning about each other's actions.

The key difference: sub-agents are a *context engineering* technique (Module 7's "isolate"). Multi-agent systems are an *architectural* commitment with all the coordination overhead.

Most production systems use sub-agents heavily and multi-agent sparingly. A coding agent might spawn 50 sub-agents in a session (each "go read this file and summarize it") without ever being a multi-agent system. Cursor and Claude Code do this — they look like single agents with great context engineering.

::: pullquote
Sub-agents are a feature you add to a single agent. Multi-agent is an architecture you commit to. Most teams need the former and accidentally build the latter.
:::

---

## Section 6 — When the Math Works: Doing It Right (≈4 pages)

If you've passed the decision framework, here's how to do multi-agent well.

**1. Plan the orchestrator carefully.** The orchestrator is where most multi-agent systems fail. It's not a thin wrapper — it's the most important agent in the system. It needs:
- Clear understanding of which sub-agents exist and what they do
- Ability to decompose tasks accurately
- Ability to handle sub-agent failures gracefully
- Synthesis logic that doesn't lose detail in translation

**2. Use the strongest model for the orchestrator.** Anthropic uses Opus as the lead, Sonnet as workers. Cheaper-on-leader / stronger-on-workers usually fails: the leader makes plan errors, propagates them, and you pay 15× to be wrong.

**3. Give sub-agents focused tool sets.** A sub-agent with the same tools as the parent has nothing distinctive. Each sub-agent should have a tool subset matched to its purpose.

**4. Cap depth and breadth.** Hard limits on:
- Number of sub-agent levels (typically 2-3)
- Number of sub-agents at each level
- Total budget per task (cost + tokens + wall-clock)
- Time per sub-agent

**5. Design the communication protocol.** What information goes between agents? Free-form text loses too much. Structured outputs (JSON with explicit fields) preserve more. Sub-agents return structured data; orchestrator decides what to do with it.

**6. Trace everything.** Module 17 covers observability in depth. Multi-agent systems are debugging hell without traces. Every sub-agent invocation, every message between agents, every state transition needs to be inspectable.

**7. Eval at the system level, not the agent level.** It's easy to test "is sub-agent X working?" It's much harder and much more important to test "is the system as a whole producing better results than a simpler alternative?" Without system-level eval, you'll over-engineer agents that don't move the metric you care about.

::: postmortem
**The Multi-Agent System That Was Better Than the Single-Agent (and Could Prove It)**

A research-tools company built a deep research agent. They had a working single-agent version. They built a multi-agent version (Anthropic-style: Opus orchestrator + 4-8 Sonnet workers).

Before deploying, they ran 100 representative research tasks through both. Multi-agent: 87% acceptable, $4.20 average cost. Single-agent: 71% acceptable, $0.40 average cost. Then they asked: of the 16 percentage points multi-agent won by, how many were on tasks where the customer would pay more for the better result?

Answer: most of them. Their pricing tier separated "quick research" ($1) from "deep research" ($25). Multi-agent for deep tier; single-agent for quick tier. Routing based on tier. Both stable in production for a year.

**Lesson:** the multi-agent decision is a tier decision. You can have both, route by task value, charge accordingly.
:::

---

## Section 7 — Anti-Patterns and Hype Patterns (≈4 pages)

Patterns we see often. Most are mistakes.

**The "Specialist" anti-pattern.** Same model, same tools, different prompts. Called "billing specialist" and "tech specialist." Should have been routing within one agent.

**The "Group Chat" anti-pattern.** Five agents arguing in a shared conversation. Tokens explode (each agent sees all others' messages every turn). Quality is rarely better than a single critic. Survives in research papers; rarely ships in production. (April 2026 "What Survived in Production" article: "Free mesh survived mostly as a controlled subroutine inside a supervisor, not as the outer architecture.")

**The "More Agents More Better" anti-pattern.** Linear belief that adding agents linearly improves quality. Reality: diminishing returns, then negative returns from coordination overhead.

**The "Demo-Driven Architecture" anti-pattern.** System designed to look impressive in a demo (many agents, lots of activity, visible coordination). Production cost-to-benefit doesn't survive scrutiny.

**The "Microservices Cargo Cult" anti-pattern.** "We have services-per-domain in our backend; we should have agents-per-domain in our agent system." This conflates organizational boundaries with architectural ones. Microservices solved a deployment / team-scaling problem; multi-agent solves a context-isolation / parallelism problem. They're different.

The hype patterns to recognize in conference talks and demos:

- "Our agent team..." (often: one agent)
- "We orchestrate dozens of agents..." (often: dozens of prompt templates)
- "Emergent intelligence from agent interaction..." (often: the supervisor reads all the worker outputs and synthesizes; "emergent" is doing a lot of work)

::: brain
The 90.2% Anthropic improvement number is real. So is the 15× cost. Why does almost every conference talk cite the first number and not the second?

(Survivorship bias and incentives. Talks that say "we used multi-agent and it worked great" sell better than "we tried multi-agent, the bill quintupled, we reverted to single agent." The latter is more common but less photogenic. As a builder, weight the bill-quintupled stories more heavily — they're the rare ones that get to be honest.)
:::

---

## Section 8 — A Threshold Test (≈2 pages)

Before reading Module 14 — try this on your own current or planned project.

**The 5-question test:**

1. What is one task instance worth, in dollars or business value?
2. What does the task currently cost (or would cost) as a single agent?
3. Is the ratio (1)/(2) > 15? If no, multi-agent is unlikely to pay back.
4. Is the task naturally parallelizable into independent sub-questions? If no, multi-agent gains will be smaller than the headline 90.2%.
5. Do you have eval infrastructure that can compare multi-agent vs single-agent on YOUR task (not Anthropic's)? If no, you'll never know if multi-agent is winning.

If you score 5/5: read Module 14, build it.
If 3-4/5: read Module 14 with skepticism; build a small comparison study first.
If <3/5: stay with single agent + great tools + great workflow patterns. Multi-agent will not earn its keep on your workload.

::: pullquote
The most expensive sentence in agent engineering is "let's add another agent." The cheapest is "we don't need another agent."
:::

---

## Section 9 — What's Next (≈1 page)

Module 14 codes the four multi-agent architectures (supervisor, hierarchical, network/peer, sequential pipeline, debate/critique) — same task, multiple ways. Now that we've established when multi-agent is worth it, Module 14 is about doing it well when it is.

Module 15 covers swarms — the most overhyped corner of the field. Stigmergy, OpenAI Swarm-style handoffs, when emergence is real, when it's theater.

Module 17 (production) covers the observability and eval infrastructure that makes multi-agent debuggable.

The capstone (Module 18) builds a multi-agent research system that's *actually* multi-agent — passing the threshold test deliberately.

---

## Recurring elements used in this module

- 1 chapter opener
- 1 hook
- 2 No Dumb Questions
- 2 Brain Power prompts
- 2 Production Postmortems (CrewAI customer support, the system that justified itself)
- 1 Watch It! / Gotcha block
- 1 Code Exercise (audit a multi-agent codebase)
- 3 Pullquotes
- 1 decision flowchart
- 1 Bullet Points recap

::: sam-arc
Sam, riding the high of shipping a working hardened/reflective/guarded agent (Modules 2-12), reads about Anthropic's multi-agent research system and proposes going multi-agent. CFO asks for a back-of-envelope cost estimate for the full company workload. Sam runs the numbers, applies the threshold test from Section 8, and discovers the company's main use case (customer support tickets) is unambiguously not a multi-agent fit — too low value-per-task, too tightly interdependent. Sam pivots: multi-agent for the new "deep analysis" premium tier; single agent for everything else. CFO buys it. Sam's arc this module: **the most expensive sentence is "let's add another agent" — and the second most expensive is "everyone else is doing it."**
:::

::: page-budget
S1 (The numbers): 4p
S2 (Where it wins): 5p
S3 (Where it loses): 5p
S4 (Decision framework): 5p
S5 (Sub-agents vs multi-agent): 3p
S6 (Doing it right): 4p
S7 (Anti-patterns): 4p
S8 (Threshold test): 2p
S9 (Next): 1p
Recurring elements: 3p
TOTAL: ~36 pages
:::

::: sources
**Must verify when drafting (the headline numbers are the most important to lock in):**

- Anthropic engineering blog "How We Built Our Multi-Agent Research System" (June 2025) — primary source for 90.2%, 15×, 80% of variance
- BrowseComp benchmark — verify the eval Anthropic used and the exact numbers
- Anthropic Sub-Agents API documentation — current model field options, mechanics
- Vellum's analysis of Anthropic's results — for the explicit "single-agent vs multi-agent for product descriptions" example
- ByteByteGo's writeup of Anthropic's system (Sep 2025) — corroborating source
- Anthropic API current pricing — to make the cost examples concrete
- "Multi-Agent in Production in 2026: What Actually Survived" (Lanham, Apr 2026) — for the supervisor/collaboration pattern survival analysis
- Digital Applied Q2 2026 architecture taxonomy — for the "graph and hierarchy survive; swarm/blackboard are theater" framing
- ZenML LLMOps database entry on Anthropic's research system — for production engineering details
- Per-pattern token multipliers (sequential 2-4×, debate 5-10×, group chat 10-30×) — verify or moderate these as estimates

**Stable knowledge:**
- The economic test (task value > 15× single-agent cost)
- The five categories where multi-agent wins
- The six categories where it loses
- The sub-agent vs multi-agent distinction
- The decision flowchart structure

**Cross-references:**
- Module 3 — workflow patterns are the alternative to multi-agent for most use cases
- Module 7 — sub-agent isolation as a context engineering technique
- Module 11 — debate/critique cycles are a multi-agent reflection variant
- Module 14 — codes the architectures
- Module 15 — swarms get the dedicated treatment
- Module 17 — observability is non-negotiable for multi-agent
- Module 18 — capstone applies these decisions to a real build
:::

::: bullet-points
### Module 13 in eight bullets

(filled at draft time)

- Anthropic's canonical numbers: +90.2% improvement at 15× tokens; 80% of variance from token use
- Multi-agent is best when result value far exceeds cost; explicitly worse for tightly-interdependent tasks like coding
- Five categories where it wins: parallelizable research, true specialization, isolated context for security, long-horizon, adversarial cycles
- Six categories where it loses: tight interdependence, low task value, latency-sensitive, unbounded growth, communication-fragile, single-skill
- Decision framework: single LLM call → workflow → single agent + tools → multi-agent (only if all criteria pass)
- Sub-agents are a context engineering feature; multi-agent is an architectural commitment — don't confuse them
- Doing multi-agent right: strong orchestrator model, focused sub-agent tools, hard caps, structured comms, system-level eval
- Threshold test: 5/5 = build it; <3/5 = stay with single agent + better tools
:::
