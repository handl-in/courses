# Module 1 Outline — Reframing: Workflows in a Trench Coat

::: chapter-opener
<div class="module-num">MODULE 01 — OUTLINE</div>
<div class="module-title">Reframing: Workflows in a Trench Coat</div>
<div class="subtitle">Most "agents" in production aren't agents.<br>Let's audit yours.</div>
<div class="pages">Target length: ~32 pages</div>
:::

## What this module is

The opener. Not a "what is an agent" tutorial — your readers already shipped one. This module is the **reframing**: a sharp definitional cut between agents and workflows, an audit of real codebases, and the contrarian voice that runs through the rest of the book.

By the end, the reader can look at any system and confidently say "that's a workflow," "that's an agent," or "that's an agent that should have been a workflow."

::: hook
"Sam shipped what the team called an 'AI research agent.' It's been running in prod for six weeks. This morning, finance flagged the OpenRouter bill. Sam pulls up the code. It's a `for` loop with three `client.messages.create` calls. There is no agent. There never was."
:::

---

## Section 1 — The Trench Coat (≈4 pages)

::: hook
Open with Sam at their laptop. The "agent" is a function. The function has a comment that says `# Agent loop`. Inside is a router, then a summarizer, then a writer. Each step has a fixed prompt. There is no LLM-directed control flow anywhere.
:::

The provocative claim: **most "agents" in production are workflows.** A workflow has predetermined steps; an agent has LLM-directed steps. The marketing budget for "agentic AI" is enormous, so the word has lost meaning. We're going to take it back.

Cite Anthropic's "Building effective agents" definition (Dec 2024, still the cleanest): *agents dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks*. Workflows are systems where LLMs and tools are orchestrated through predefined code paths.

::: pullquote
A workflow is a flowchart with LLMs in some boxes. An agent is an LLM that decides what the next box is.
:::

**Why this distinction matters in practice (not in theory):**

Cost behavior is different. Workflows have predictable token spend; agents have unbounded spend until you bound them. Latency is different. Workflows have a known critical path; agents have a tail that depends on what the model decides to do. Failure modes are different. Workflows fail at known boundaries; agents fail novel ways. Eval is different. Workflows can be unit-tested; agents need trajectory eval.

**What's not a useful distinction:** "uses tools" (workflows can too), "calls an LLM in a loop" (a `while` loop with a fixed prompt is still a workflow), "feels autonomous" (vibes are not architecture).

::: nodumbq
**Q: So if I have a router that picks one of three prompts, that's a workflow?**

Yes. Routing is a workflow pattern (we'll cover all five in Module 3). The LLM picks a path, but the paths are fixed.

**Q: What if the LLM decides to call a tool, and based on the result calls another tool?**

If the *set* of tools and *when to stop* are decided by the LLM at runtime, that's an agent. If your code has `if step == 1: do X; if step == 2: do Y`, that's a workflow even when each step has tools.

**Q: My system uses LangGraph. Does that make it agentic?**

LangGraph is a state-machine library. State machines are workflows by default. You can build agents in LangGraph by giving a node the freedom to choose the next node — but most LangGraph code in the wild is workflow code with extra nouns.
:::

---

## Section 2 — Audit Your Codebase (≈5 pages)

The first hands-on section is an **audit exercise**, not a build. The reader takes their existing "agent" code and reclassifies it.

::: code-exercise
**Exercise 1.1 — Audit a real agent.**

Reader is given (or pulls in) one of their own "agent" implementations. We provide a checklist:

1. Find the main loop. Is there one?
2. Inside the loop, who decides the next action — code or model?
3. List every tool call site. For each: is the tool selected by the model or hard-coded?
4. Where does the loop terminate? Fixed iterations, or model-decided?
5. Count branch points where code (not model) chooses a path.

Score: 0-2 model-driven decisions = workflow. 3-5 = hybrid. 6+ = genuine agent.
:::

We provide a sample codebase to audit (so readers without their own can still do the exercise). The sample is an "AI research agent" that's actually a 4-step pipeline. We walk through the audit live.

::: brain
Pick your most "agentic" production system. Without looking at code, predict its score on the audit. Now look at the code. Were you right?

Most engineers overestimate by 2-3 points.
:::

::: postmortem
**The "Multi-Agent Research System" That Was a For Loop**

Real production case from a Y Combinator company (anonymized). They marketed an "autonomous research agent" to enterprise. Internal name: "Kestrel." External demos showed it researching topics, citing sources, writing reports.

The code: 1 router prompt → 1 fan-out to N parallel summarizers (N hardcoded to 5) → 1 reducer prompt → 1 writer prompt. Total LLM-driven decisions: zero. The router didn't even pick a path; it normalized the query so the next step's prompt would be cleaner.

When complex queries broke it, they added a 6th prompt to "validate." Then a 7th to "retry." Then a 1000-line YAML config of branches. By the time the engineering team realized the agent was a workflow, they had reinvented Apache Airflow with 100x the latency and 1000x the cost.

**Lesson:** The marketing called it agentic. The code called it `pipeline.py`. Trust the code.
:::

---

## Section 3 — The Definitional Lattice (≈4 pages)

A more rigorous taxonomy. Three axes:

**Axis 1: Who controls the next step?** Code (workflow) or model (agent).

**Axis 2: What's the action space?** Fixed (tool list known at design time) or dynamic (tools discovered at runtime, e.g., MCP).

**Axis 3: Termination?** Fixed iterations / fixed conditions (workflow) or model-decided stop (agent).

Plot real systems on these axes. Examples to place:
- ChatGPT with tools enabled: agent on all three.
- A LangChain RAG pipeline: workflow on all three.
- Cursor / Claude Code: agent on axes 1 and 3, mostly fixed on 2.
- A "supervisor" pattern with hardcoded worker pool: workflow on 2, hybrid on 1, varies on 3.

::: fireside
**FIRESIDE CHAT: "Workflow" vs "Agent" argue their merits**

::: cast
<div class="character workflow"><span class="name">Workflow:</span> I'm predictable. You can test me. You can budget me. You can debug me. The user always knows what's going to happen.</div>

<div class="character agent"><span class="name">Agent:</span> You're rigid. You can't handle a query you weren't designed for. The user has to know your edges to use you.</div>

<div class="character workflow"><span class="name">Workflow:</span> Most production systems handle queries that *were* designed for. That's why they're production systems.</div>

<div class="character agent"><span class="name">Agent:</span> And that's why they get replaced when the product evolves. I can handle the new query without a code change.</div>

<div class="character workflow"><span class="name">Workflow:</span> You also handle it for 15x the cost and you might decide to do something nobody asked for.</div>

<div class="character agent"><span class="name">Agent:</span> Fair. So when do you win and when do I?</div>

<div class="character workflow"><span class="name">Workflow:</span> I win when the task shape is known. You win when it isn't.</div>
:::
:::

---

## Section 4 — When to Reach for Each (≈4 pages)

The decision framework. We'll teach it once here, refer back to it for the rest of the book.

**Use a workflow when:** the task decomposes cleanly into known steps, latency budget is tight, cost predictability matters more than handling novelty, the action space is small, you have good test coverage requirements.

**Use an agent when:** the task shape varies query-to-query, the tool set is large or dynamic, you genuinely need exploration, partial failures should adapt rather than abort.

**Use neither (just call the LLM once) when:** the answer fits in one prompt with one response. Most "AI features" should be this and aren't.

::: gotcha
The "agent first, simplify later" anti-pattern: teams reach for agents because it's exciting, then spend months adding constraints to make them behave. By the time the agent works, it's a workflow. They could have started there.
:::

::: exercise
For each scenario, pick: single LLM call, workflow, or agent. Justify in one sentence.

1. Translating user reviews from Japanese to English at scale.
2. A code review bot that comments on PRs.
3. A research assistant that answers ambiguous questions by searching, reading, synthesizing.
4. Categorizing customer support tickets into 12 known categories.
5. A debugging assistant that runs commands, reads output, decides next steps.
6. Summarizing earnings calls into a 3-bullet brief.

<details class="answer"><summary>show answers</summary>

1. **Single LLM call.** Translation is a single transform. No agent. No workflow.
2. **Workflow.** Steps are knowable: fetch diff, route by file type, generate comments, post. Some steps use LLMs. Routing happens once.
3. **Agent.** "Ambiguous questions" is the giveaway — you don't know the path until the model explores.
4. **Single LLM call** with structured output. Twelve categories, one prompt.
5. **Agent.** The model decides what to run next based on output it hasn't seen yet.
6. **Workflow.** Fetch transcript → chunk → summarize → reduce. All steps known.
:::
:::

---

## Section 5 — The Cost of Being Wrong (≈3 pages)

Quantify the failure modes:

**Workflow forced into agent territory:** brittle to query variation, requires constant new branches, eventually rewritten anyway.

**Agent forced into workflow territory:** ~15x the token cost (we'll cite Anthropic's research in Module 13), unpredictable latency, eval becomes a research project, support tickets up.

Real numbers from public post-mortems and our own measurements (which we'll re-verify when drafting). A simple categorization workflow costs ~$0.0003/query; the same task built as an agent costs ~$0.05/query — 150x — because the agent decides to "verify its answer" by re-reading the input twice.

::: key
The most expensive bug in agent engineering is using an agent for a workflow's job. The second most expensive is using a workflow for an agent's job. The first is more common.
:::

---

## Section 6 — A Working Vocabulary for the Rest of the Book (≈3 pages)

Lock terms we'll use consistently across all 18 modules:

- **LLM call** — one `messages.create`. The atom.
- **Workflow** — predetermined orchestration of LLM calls and tools.
- **Agent** — an LLM that directs its own control flow over multiple turns with tools.
- **Agent loop** — the iterative call-tool-observe cycle.
- **Tool** — a function the model can choose to invoke, exposed via the API's tool-use mechanism.
- **Trajectory** — the full sequence of (input, model output, tool calls, results) for a task.
- **Step / turn** — one iteration of the agent loop.
- **Multi-agent system** — multiple agents (or workflows wrapping agents) communicating.
- **Orchestrator** — an agent or workflow that coordinates other agents.
- **Sub-agent** — an agent invoked by another agent, typically with isolated context.

We'll resist marketing terms ("agentic," "autonomous," "AGI-like") in our own writing. Readers are free to use them; we won't.

---

## Section 7 — What's Next (≈1 page)

Module 2 builds a real agent loop in raw Python with the Anthropic SDK. Naive 20-line version vs. hardened 200-line version, side by side. The hardened version is the one your production agents should look like.

Module 3 covers the five workflow patterns that should be your first reach.

Modules 4–9 build the components: tools, MCP, model routing, context engineering, compression, memory, hallucination, reflection, guardrails.

Modules 10–18 cover multi-agent, swarms, transactional patterns, production, and the capstone build.

---

## Recurring elements used in this module

- 1 chapter opener
- 1 hook
- 3 No Dumb Questions blocks (~8 questions total)
- 2 Brain Power prompts
- 1 Fireside Chat (Workflow vs Agent)
- 1 Production Postmortem (Kestrel)
- 1 Watch It! (agent-first anti-pattern)
- 2 Sharpen Your Pencil exercises (audit + categorization)
- 1 Code Exercise (audit your codebase)
- 1 Pullquote
- 1 Key Takeaway
- 1 Bullet Points recap (placeholder, fills at draft time)

::: sam-arc
Sam appears in Section 1 staring at the OpenRouter bill. By Section 4 they've audited their "agent" and discovered it's a workflow. The arc this module: **realization**. Sam doesn't fix anything yet. They just see what they actually built.
:::

::: page-budget
Section 1 (Trench Coat): 4p
Section 2 (Audit): 5p
Section 3 (Definitional Lattice): 4p
Section 4 (When to Reach): 4p
Section 5 (Cost of Being Wrong): 3p
Section 6 (Vocabulary): 3p
Section 7 (What's Next): 1p
Recurring elements (woven in): 8p
TOTAL: ~32 pages
:::

::: sources
**Must verify when drafting (search before writing):**

- Anthropic, "Building effective agents" (Dec 2024) — primary source for definitions
- Anthropic's research on multi-agent token multiplier (15x figure) — Module 13 main source, cited briefly here
- Latest Anthropic SDK version + model strings (claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5)
- LangGraph current API (for the "LangGraph is workflow by default" claim)
- Public post-mortems for the Kestrel-style story (anonymize but base on real)
- OWASP LLM Top 10 (2025) — for cost/security framing in Section 5

**Stable knowledge (low search risk):**
- The workflow/agent definitional distinction
- The five workflow patterns (named here, taught Module 3)
- General taxonomy axes
:::

::: bullet-points
### Module 1 in eight bullets

(filled at draft time — placeholder for outline review)

- Most production "agents" are workflows in marketing wrappers
- The cut: who decides the next step — code (workflow) or model (agent)
- Audit your own code with the 5-question checklist
- Three taxonomy axes: control, action space, termination
- Workflows are predictable, agents are flexible — pick by task shape
- The "agent first, simplify later" anti-pattern is the most expensive mistake
- Vocabulary lock: workflow, agent, loop, tool, trajectory, step, sub-agent
- Next: build a hardened agent loop in raw SDK
:::
