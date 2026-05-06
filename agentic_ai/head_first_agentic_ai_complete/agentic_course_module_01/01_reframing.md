# Module 1 — Reframing: Workflows in a Trench Coat

::: chapter-opener
<div class="module-num">MODULE 1</div>
<div class="module-title">Reframing: Workflows in a Trench Coat</div>
<div class="subtitle">Most of what gets called an "agent" in 2026 is actually a workflow with better marketing.<br>Before we build agents, let's figure out what we're building.</div>
<div class="pages">~32 pages · the foundation</div>
:::

::: hook
A staff engineer stares at a Slack thread her CEO posted twenty minutes ago. The thread is three articles deep on "agentic AI transforming the enterprise." There's a quote about agents that book flights, write code, run companies. There's an architecture diagram with seven boxes labeled "Agent A," "Agent B," and so on, with arrows pointing in every direction.

At the bottom, the CEO has typed: *"Should we be doing more of this?"*

She has two open browser tabs. One is the Anthropic engineering blog from December 2024, where Erik Schluntz and Barry Zhang carefully define agents as systems "where LLMs dynamically direct their own processes and tool usage." The other is her team's actual codebase — a two-step LLM pipeline that classifies an incoming ticket, then routes it to one of three specialist prompts. It works well. Customers are happy. It's been running in production for six months without incident.

The Slack thread says her two-step pipeline is a "rudimentary agent." The CEO seems to think she should be building something with more arrows.

She closes the laptop. She wants to write back: *what we have is fine; the problem is the vocabulary.* But she doesn't know how to say that without sounding defensive.

This is the reframing problem. It's where this book begins.
:::

---

## What this module is

This module exists because the word "agent" is doing too much work. It's been stretched across an enormous category — anything from a single LLM call with a search tool, to a multi-agent collective that spans data centers — and the stretching has left engineers without a clean way to think about what they're actually building.

The goal here isn't to police vocabulary. It's to give you a working mental model that distinguishes the things that genuinely behave differently in production: what fails differently, what costs differently, what scales differently, what needs different debugging tools.

By the end of this module, you'll be able to:

- Distinguish workflows from agents using a single concrete criterion that survives contact with marketing material
- Recognize the five workflow patterns that solve most "agent" problems
- Identify the three properties that genuinely require an agent (not a workflow with an agent-shaped UI)
- Push back, with specifics, on a manager or stakeholder who insists on agents for a problem that doesn't need one

We're going to be contrarian in this module — not for sport, but because the field's vocabulary is currently incentivizing engineers to build the wrong thing. Once we've reframed, the rest of the book can talk straight about what agents *are*, what they're *for*, and how to build them when the answer is yes.

---

## Section 1 — A Definition That Holds Up

The most useful definition of "agent" comes from Anthropic itself, post-refinement of their original December 2024 essay. In a follow-up post on context engineering, they wrote:

> *"We've gravitated towards a simple definition for agents: LLMs autonomously using tools in a loop."*

Six words. Three load-bearing concepts. We're going to unpack each one because the loose use of any of them is where the vocabulary breaks down.

**"LLMs"** — plural is fine, but a system isn't more agentic just because there are more LLMs. Five LLMs in a fixed pipeline is a workflow with five steps. One LLM that decides what to do next is closer to an agent.

**"Autonomously"** — this is the discriminator. The model itself is choosing what happens next. Not a planner that the model drafted last Tuesday and now executes deterministically. Not an `if`/`else` over a routing classifier's output. The model, *during* the run, *deciding*.

**"Using tools"** — the model can affect the world. Read files, query APIs, execute code, navigate web pages, call other models. Without tools, the LLM is generating text that sits in a buffer; with tools, it's an active participant in a feedback loop with reality.

**"In a loop"** — the loop is what makes failure modes interesting. One pass through tool-call-and-observe is a single decision. Many passes is a trajectory, with all the compounding error, drift, and cost behavior that trajectories bring.

::: pullquote
An agent is an LLM autonomously using tools in a loop. Six words. If your system isn't all four — LLM, autonomy, tools, loop — it's a workflow that's borrowing the word "agent" for marketing reasons.
:::

Now look at the staff engineer's two-step pipeline from the cold open. Classify ticket → route to specialist prompt. Is it an agent?

- ✅ LLMs (two of them, even)
- ❌ Autonomously — the routing is `if classified=="billing" then call billing_prompt`. The model classifies; deterministic code routes.
- ❌ Tools — neither model uses tools; they emit text.
- ❌ Loop — one pass. Classify, route, respond, done.

It's not an agent. It's a workflow. A two-step prompt pipeline. And — this is the important part — that's *fine*. The system works. Customers are happy. Calling it an agent doesn't make it better; it makes it harder to talk about.

::: nodumbq
**Q: Doesn't the field disagree on this definition? I've seen people call workflows agents and vice versa.**

The field absolutely disagrees on this definition. People call workflows agents because "agent" sounds more impressive. People call agents workflows because they want to downplay risk. The definition we're using here is Anthropic's own post-refinement formulation; it's the closest thing the field has to a standard. We're using it because it draws a line that maps to actual differences in how these systems behave — failure modes, cost, debuggability — not because it's the only line you'll see drawn in the wild.

**Q: If my system has tool calls but the orchestration is hardcoded, is it an agent?**

No. That's a workflow with tools. Tools are necessary for an agent but not sufficient. The discriminator is whether the *model* is choosing what to do next. If a `for` loop is iterating over a fixed list of steps and each step calls a tool, that's a workflow even if every step has tool access.

**Q: Is "agent" a bad word, then? Should we stop using it?**

No, just use it precisely. When you have an LLM autonomously using tools in a loop, call it an agent. When you have a deterministic pipeline with LLM calls, call it a workflow. The book uses both words a lot; it tries hard not to mix them up.
:::

---

## Section 2 — The Augmented LLM: The Foundational Building Block

Before we get to workflows and agents, there's one layer underneath both: the augmented LLM. Anthropic's essay calls it "the basic building block of agentic systems." We'll call it that too, because the framing matters.

An augmented LLM is just a model with one or more of:

- **Retrieval** — the ability to pull in relevant context (RAG, semantic search, file reads)
- **Tools** — the ability to call functions and observe results
- **Memory** — the ability to retain information across turns or sessions

The augmented LLM is not an agent. It's a *capability layer*. A single API call with a search tool attached is an augmented LLM call. A workflow built on top might invoke that augmented call multiple times in a fixed pattern. An agent built on top might invoke it autonomously in a loop.

Why does this matter? Because most "AI features" you'll encounter — and most "AI features" you'll *build* — live at this layer. They're not workflows. They're not agents. They're a single LLM call, augmented appropriately, that does one thing well.

```python
from anthropic import Anthropic

client = Anthropic()

# Augmented LLM: a single call with retrieval and tools.
# Not a workflow, not an agent — just a capability.
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=[
        {
            "name": "search_internal_docs",
            "description": "Search the company's documentation.",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    ],
    messages=[
        {"role": "user", "content": "What's our policy on parental leave?"},
    ],
)
```

That's the building block. If you can stop here for your use case, *stop here*. Most user-facing AI features ship as augmented LLMs and never need anything more. The complexity tax of going further — workflows, then agents — is real. Anthropic's own framing is direct about this:

> *"When building applications with LLMs, we recommend finding the simplest solution possible, and only increasing complexity when needed. This might mean not building agentic systems at all."*

If you've internalized one thing from this module, let it be that quote. Most AI features don't need workflows. Most that do don't need agents. The hierarchy of complexity goes:

1. Single augmented LLM call
2. Workflow (orchestrated multi-call pipeline)
3. Agent (autonomous loop)
4. Multi-agent system (multiple agents coordinating)

You climb only when the level you're at can't do the job. Not when the architecture diagram looks more impressive.

::: gotcha
The single most expensive mistake in early-2026 agent engineering is reaching for an agent when an augmented LLM call would have done the job. The agent will work — sometimes — but at 10-50× the cost, with worse latency, more failure modes, and harder debugging. The "agent" version doesn't even necessarily produce better results; it produces *more variable* results that occasionally outperform on edge cases. For most use cases, that variance is a bug, not a feature.
:::

---

## Section 3 — Workflows: The Five Patterns That Cover 80% of Cases

Anthropic's "Building Effective Agents" essay catalogs five workflow patterns. They're worth memorizing because most "agent" projects could ship as one of them — faster, cheaper, more reliably.

We'll spend Module 3 on the deep treatment. Here, we're sketching them so you have the vocabulary by the end of this module.

### Pattern 1: Prompt chaining

A task is decomposed into a fixed sequence of LLM calls, each operating on the previous one's output. Optional programmatic checks ("gates") between calls verify intermediate state.

```
[Input] → [LLM Call 1] → [Gate?] → [LLM Call 2] → [Output]
```

**When to use:** the task can be cleanly decomposed into ordered subtasks, and trading some latency for better accuracy is worth it. Generating an outline → writing from the outline. Translating English → polishing the translation. Drafting a marketing email → rewriting it for a different audience.

### Pattern 2: Routing

An initial LLM (or classifier) decides which downstream prompt or model handles the request. Used for separation of concerns and for cost optimization.

```
[Input] → [Router LLM] → [one of: Specialist A, B, or C] → [Output]
```

**When to use:** distinct categories of input genuinely benefit from specialized handling, and the categories are stable enough that classification can be reliable. Anthropic's own example, refreshed for the current model lineup: route easy/common questions to Haiku 4.5; route hard ones to Sonnet 4.6 or Opus 4.7. Different specialist prompts for billing, technical, and account questions.

The staff engineer's pipeline from the cold open is exactly this pattern. It's a perfectly fine workflow.

### Pattern 3: Parallelization

A task is run by multiple LLM calls simultaneously, with their outputs aggregated. Two flavors:

- **Sectioning:** the task is split into independent subtasks; each runs in parallel; results are combined.
- **Voting:** the same task runs multiple times with different prompts or configurations; results are aggregated (majority vote, scoring, etc.).

```
              ┌─→ [LLM Call A] ─┐
[Input] ──────┼─→ [LLM Call B] ─┼─→ [Aggregator] → [Output]
              └─→ [LLM Call C] ─┘
```

**When to use:** subtasks are genuinely independent (sectioning) or you need diverse perspectives for confidence (voting). Reviewing code with separate prompts for security, performance, and style. Running content moderation in parallel with response generation. Aggregating multiple votes on a difficult classification.

### Pattern 4: Orchestrator-Workers

A central LLM dynamically decomposes the task and delegates to worker LLMs, then synthesizes their results. Looks like an agent — and is, sort of, depending on how dynamic the decomposition is.

```
                  ┌─→ [Worker A] ─┐
[Orchestrator] ───┼─→ [Worker B] ─┼─→ [Synthesis] → [Output]
                  └─→ [Worker C] ─┘
```

**When to use:** the structure of the work isn't known in advance. The orchestrator looks at the input and decides how to split it. This is the closest workflow pattern to "agentic" — and Anthropic's research system (covered in Module 13) is essentially a scaled-up version of this. We'll come back to it.

### Pattern 5: Evaluator-Optimizer

One LLM produces an output; another LLM evaluates it and provides feedback; the producer revises. Loop until the evaluator approves or a cap is hit.

```
[Input] → [Producer] → [Evaluator] → [pass? done : revise]
              ↑___________________________|
```

**When to use:** quality criteria are clear, iterative refinement provides measurable value, and the producer is variable enough that evaluation catches real issues. Literary translation with a separate evaluator for nuance. Code generation with a separate critic for correctness. Long-form writing with a separate editor.

This is the same shape as Module 11's reflection, by the way. Reflection in agents is the evaluator-optimizer workflow living *inside* an agent loop.

::: brain
Look at those five patterns. Pick a "AI feature" you've built, used, or seen demoed recently. Which pattern is it? Or is it actually just an augmented LLM call?

Most teams will find that what they've been calling an "agent" is actually one of these five patterns. The vocabulary upgrade — from "we built an agent" to "we built an evaluator-optimizer workflow" — is not a downgrade. It's a clarity gain. The team that knows which pattern they shipped can debug it, scale it, and improve it. The team that calls everything an agent debugs it by reading runes.
:::

---

## Section 4 — Agents: When the Loop Earns Its Cost

Now we can talk about agents honestly.

An agent — by the six-word definition — is an LLM autonomously using tools in a loop. Concretely, this means: at each turn, the model decides whether to call a tool, which tool to call, and with what arguments. It observes the result. It decides what to do next. It might call more tools. It might respond to the user. It might keep going for many turns.

Here's the minimal agent loop in code:

```python
from anthropic import Anthropic

client = Anthropic()
TOOLS = [...]  # tool schemas
TOOL_HANDLERS = {...}  # maps tool name to a Python callable

def run_agent(user_query: str, max_turns: int = 20) -> str:
    messages = [{"role": "user", "content": user_query}]

    for turn in range(max_turns):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            tools=TOOLS,
            messages=messages,
        )

        # If the model is done, return its final message.
        if response.stop_reason == "end_turn":
            return extract_text(response)

        # If the model called tools, execute them and feed results back.
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    handler = TOOL_HANDLERS[block.name]
                    result = handler(**block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })
            messages.append({"role": "user", "content": tool_results})
            continue

        # Anything else: bail.
        break

    return "Agent exhausted turn limit."
```

That's it. That's an agent. Module 2 hardens this loop — adds proper error handling, budget caps, observability, the works — but the *shape* doesn't change. Tool call, observe, decide, repeat.

What makes this fundamentally different from a workflow: at no point does the *programmer* decide what happens next. The programmer wires up the tools, sets the budget, and lets the model drive. The model can take three turns or thirty. It can call the same tool five times in a row. It can decide it's done before doing anything. It can decide it can't be done and explain why.

This flexibility is the agent's value proposition. It's also the agent's whole problem.

### When the loop earns its cost

The loop is expensive. Each turn is a full LLM call with growing context. Tool results add tokens. Multiple agents compound this. Anthropic's own research found that their multi-agent research system uses roughly **15× the tokens** of a single chat interaction (we'll cover the numbers in detail in Module 13).

So when does the loop earn its cost? Three properties — all three, not just one:

**Property 1: The number of steps can't be predicted in advance.** If you can write down the steps, write a workflow. If genuinely you can't — because it depends on what you find along the way — that's an agent shape.

A research task is a good example. The first source you find changes which sources you look for next. The user's follow-up question changes the plan. You can't write a fixed pipeline that handles "answer arbitrary research questions" the way you can write one that handles "translate this document and check the translation."

**Property 2: There's a feedback loop with the world that the model needs to drive.** The agent needs to see the result of its last action to decide the next one. A coding agent runs the test suite, reads the failures, decides what to fix, edits the code, runs again. The "decides" is doing work that no fixed pipeline could replicate without knowing every possible failure shape in advance.

**Property 3: The trust budget for the model is high enough.** The agent will operate for many turns, each turn with potential to compound an error. Anthropic's essay calls this out explicitly: *"The autonomous nature of agents means higher costs, and the potential for compounding errors. We recommend extensive testing in sandboxed environments, along with the appropriate guardrails."*

If your task fails any of these three — predictable steps, no real feedback loop, low trust budget — you don't have an agent shape. You have a workflow shape with an agent costume.

::: pullquote
Agents earn their cost on three properties together: unpredictable step count, real environmental feedback, sufficient trust budget. Two-out-of-three is a workflow. All three is an agent.
:::

### What agent shapes actually look like in production

A few real examples, with the property check:

**Coding agents (Cursor, Claude Code, Copilot agent mode).**
- Unpredictable steps? ✅ The number of edits to fix a bug isn't knowable upfront.
- Feedback loop? ✅ Tests run, errors return, agent reacts.
- Trust budget? ✅ User reviews each diff or approves the session.

Yes, agent.

**Customer support routing.**
- Unpredictable steps? ❌ Almost always: classify, route, respond.
- Feedback loop? ❌ The "feedback" is "did the user ask a follow-up?"
- Trust budget? Doesn't matter; first two failed.

Workflow (specifically, routing).

**Research-and-report (Anthropic's research feature).**
- Unpredictable steps? ✅ Sub-questions emerge as evidence is gathered.
- Feedback loop? ✅ Each search result reshapes the next search.
- Trust budget? ✅ User pays for a deep report and tolerates 5 minutes of work.

Yes, agent (and a multi-agent one — Module 13).

**E-commerce chatbot that answers FAQ.**
- Unpredictable steps? ❌ Match question, return answer.
- Feedback loop? ❌ One-shot.
- Trust budget? Doesn't matter.

Augmented LLM call. Maybe a workflow with a router.

::: nodumbq
**Q: My CEO/manager wants me to build "an agent." But I think the right thing is a workflow. How do I push back?**

Specifically. "What problem are we solving?" If the answer is "we want to be more agentic," the conversation is about marketing, not engineering. If the answer is a real problem, walk through the three properties: are the steps predictable? Is there a real feedback loop? Is the trust budget sufficient?

Then map the alternatives: augmented LLM call, then five workflow patterns, then agent. Show what each costs and what each delivers. The case for the simplest solution that works is almost always defensible on cost grounds alone. Most stakeholders care about cost; very few want to spend 15× more for the same outcome.

**Q: What if the answer is "the simplest solution doesn't work"?**

Then climb the ladder. But check carefully — sometimes "doesn't work" means "isn't sophisticated enough to demo well." Production cares about whether it serves users; demos care about whether it impresses observers. Build for production.
:::

---

## Section 5 — A Worked Example: Reframing a "Multi-Agent" System

The staff engineer from the cold open isn't fictional, exactly. The pattern is everywhere. Here's a slightly disguised version of a real conversation.

A team had built what they called a "multi-agent customer success system." Architecture diagram on the whiteboard had five boxes: an "Orchestrator Agent," a "Billing Agent," a "Technical Support Agent," an "Escalation Agent," and a "Sentiment Analysis Agent." Arrows in every direction.

The implementation, when you looked at the actual code:

```python
def handle_message(message: str) -> str:
    # "Sentiment Analysis Agent"
    sentiment = call_llm(SENTIMENT_PROMPT, message)

    # "Orchestrator Agent"
    category = call_llm(CATEGORY_PROMPT, message)

    if sentiment == "angry" or category == "complaint":
        # "Escalation Agent"
        return call_llm(ESCALATION_PROMPT, message)

    if category == "billing":
        # "Billing Agent"
        return call_llm(BILLING_PROMPT, message)

    if category == "technical":
        # "Technical Support Agent"
        return call_llm(TECHNICAL_PROMPT, message)

    return call_llm(GENERAL_PROMPT, message)
```

Let's count agents: zero. There are no agents in this system.

There are five LLM calls, orchestrated by Python `if`/`else` logic. The "Sentiment Analysis Agent" is one prompt. The "Orchestrator Agent" is another prompt. The router is a Python function. None of the LLMs autonomously use tools in a loop — none of them use tools at all.

What this system *is*, in workflow vocabulary: it's a parallelization workflow (sentiment + category run together) feeding into a routing workflow. Two well-understood patterns composed together. It's a fine system. It works.

What changed when the team reframed:

1. They stopped trying to "make the agents communicate better" (the agents were prompts; prompts don't communicate). They started thinking about prompt quality and routing accuracy, which were the actual problems.
2. They stopped trying to add observability for "agent-to-agent calls" (there were none). They added it for the LLM calls and the routing decisions, which were the things that could fail.
3. They stopped budgeting for "multi-agent token costs" (a 15× multiplier they didn't actually pay; their system was sequential workflow calls, not parallel research workers). They budgeted accurately and discovered they had headroom for a more capable router model.
4. The Slack thread to the CEO became "we ship a routing-and-parallelization workflow that handles 87% of tickets autonomously, with sentiment-aware escalation for the angry path." That sentence describes what they actually built, what it does, and why it works. The previous sentence ("we built a multi-agent system") described nothing.

::: postmortem
**The Re-architecture That Wasn't**

A team I'll call Acme spent six weeks "re-architecting their workflow into a true multi-agent system" at the request of their VP of engineering. They added an "agent loop" around their existing routing pipeline. They added tools the agents "could" use. They paid for sub-agent-style prompts that handed off to each other.

End-state behavior was identical to the original workflow. The orchestrator agent's loop ran exactly once per request (because the task didn't actually require iteration). The handoffs were structured the same way the routing had been. Latency went up 40%. Cost went up 3×. Quality went up 1.2 points on their internal eval — within noise.

Six weeks later, they reverted to the workflow. The VP asked what had gone wrong. The answer: nothing went wrong. The original pipeline was already the right shape for the problem. The "re-architecture" was vocabulary tourism.

The lesson isn't that multi-agent is bad. It's that *building the wrong shape for your problem* is bad, regardless of how the shape is labeled. If the problem is one-shot routing, build one-shot routing. If you call it an agent, you'll spend like one.
:::

---

## Section 6 — The Decision: A Field Guide

Here's the field guide you can carry into your next project meeting.

### Step 1 — Can a single LLM call do it?

Not "can it do it perfectly" — can it do it *well enough that users would pay for it*? If yes, ship that. Augmented with retrieval and tools as needed. Don't climb the ladder.

Examples that stop here: drafting an email, summarizing a document, classifying an intent, extracting structured data, answering a single factual question.

### Step 2 — Can a workflow do it?

If a single call isn't enough, the next question: can you write down the steps? If yes, you have a workflow. Pick the pattern:

- One step depends on the output of another? **Prompt chaining**.
- Different inputs need different handling? **Routing**.
- Subtasks are independent? **Parallelization** (sectioning).
- Need diverse perspectives? **Parallelization** (voting).
- Structure of the work depends on the input? **Orchestrator-workers**.
- Need iterative quality refinement? **Evaluator-optimizer**.

Most "AI features" stop here. Workflows are predictable, debuggable, cheap to scale, easy to monitor. They have a fixed cost per request that you can model.

### Step 3 — Are all three agent properties present?

If a workflow can't do it, check the three properties:

1. **Steps unpredictable in advance** — you genuinely don't know what the agent will do next.
2. **Feedback loop with the world** — the agent's next action depends on its last action's result.
3. **Trust budget sufficient** — you can tolerate the agent operating for many turns with potential for compounding error, and you have the eval/observability to know when it goes wrong.

Two out of three is not enough. All three. If you have all three: you have an agent shape.

### Step 4 — One agent or multiple?

We don't even ask this in Module 1. The default is one agent. Multi-agent has its own threshold (15× cost as the headline number from Anthropic's research) and we treat it in Modules 13-15. Don't add agents because the diagram looks impressive.

### The three questions to ask before "let's build an agent"

When someone proposes an agent — at your standup, in a roadmap doc, on the whiteboard — ask three questions:

1. **What's the augmented-LLM version of this?** Force the simplest case to be considered.
2. **Which workflow pattern fits, if any?** Force the workflow consideration before agent.
3. **Which of the three agent properties does the task require?** Force explicit articulation of why this needs an agent.

If the answer to question 1 is "an augmented LLM doesn't work because X," good — you've articulated a real requirement. If question 2 is "no workflow fits because Y," good. If question 3 produces all three properties cleanly, you have a defensible case for an agent.

If the answers are "an agent is what's modern" or "an agent will look better to investors" or "we want to be agentic" — those aren't requirements. They're decoration.

::: code-exercise
**Exercise 1.1 — Reframe an "agent" you've encountered.**

Pick an "AI agent" project you've built, used, or seen described in a blog post / case study / vendor pitch.

1. Write down what the system actually does, mechanically. (Not what it's called — what it does.)
2. Apply the six-word definition: is it an LLM autonomously using tools in a loop?
3. If not, which of the five workflow patterns is it?
4. If yes, which of the three agent properties does it satisfy? All three, or fewer?
5. Write one sentence describing the system in workflow/agent vocabulary that's accurate.

Goal: by the end of this exercise, you should be able to talk about the system without using the word "agent" until it's actually warranted.
:::

---

## Section 7 — What This Book Does (and Doesn't) Do

A short directional note before Module 2.

This book is about agents — the real shape, the autonomous-loop kind. We'll spend most of the book on them. But we got here by clearing away the things that *aren't* agents: augmented LLM calls, workflows, multi-LLM systems with deterministic orchestration. Those things are valuable; they're just not what this book is about.

What you'll see in the next 17 modules:

- **Module 2 hardens the agent loop.** All the things the minimal loop in Section 4 doesn't do: errors, budgets, instrumentation, partial failures.
- **Module 3 covers the five workflow patterns properly.** Because — even in this book — most of what you build will be workflows or augmented calls. Knowing the patterns well is non-negotiable.
- **Modules 4-9** are the building blocks: tools, MCP, model routing, context engineering, compression, memory.
- **Modules 10-12** are the reliability layer: hallucinations, reflection, guardrails.
- **Modules 13-15** are multi-agent: when it earns its cost, the architectures, the swarm patterns.
- **Modules 16-17** are operational: transactional safety (sagas), production observability and evals.
- **Module 18** is the capstone: build a real one, end to end.

What this book doesn't do:

- It doesn't sell agents. We're going to be honest about when they don't earn their cost.
- It doesn't favor a framework. We use Anthropic's SDK as the primary code surface (because the book is built around Claude), with LangGraph and CrewAI for comparison where they earn their place. The patterns are framework-neutral.
- It doesn't promise that agents will replace your team. They won't. Most of them won't even reliably replace one of your team's existing scripts without a lot of operational work.

What it tries to do is leave you, after Module 18, able to build agents that work — and able to refuse to build the ones that don't.

::: pullquote
Most "agent" projects in 2026 should not be agents. The discipline of figuring out which ones should — and shipping those well — is what this book is teaching.
:::

---

## Recap: Module 1 in eight bullets

::: bullet-points
- An agent is an LLM autonomously using tools in a loop. Six words. All four parts matter.
- The hierarchy of complexity goes: augmented LLM → workflow → agent → multi-agent. Climb only when the level you're at can't do the job.
- The augmented LLM (retrieval + tools + memory) is the foundational building block; most "AI features" should ship as augmented calls and stop there.
- Five workflow patterns cover most "agent" use cases: prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer.
- Agents earn their cost when three properties are all present: unpredictable steps, real feedback loop, sufficient trust budget. Two out of three is a workflow.
- Most "multi-agent systems" in production are workflows in agent costumes. Reframing reveals what to actually optimize.
- Four questions to ask before building an agent: what's the augmented-LLM version, which workflow pattern fits, which of the three agent properties applies, and why is none of the simpler options enough?
- The vocabulary upgrade — "we built an X workflow" instead of "we built an agent" — is a clarity gain, not a downgrade. Teams that know what they shipped can debug, scale, and improve it.
:::

---

::: sam-arc
**Sam, our recurring character, makes their first appearance.**

Sam is a staff engineer at a mid-sized SaaS company. They've been shipping production systems for ten years, the last two of which have involved increasing amounts of LLM glue. Last quarter they shipped a customer-support routing pipeline that's been running quietly and well — the same pipeline from this module's cold open.

This week, Sam's CEO posted the agentic AI Slack thread. Sam read this module — or its outline equivalent — and went into the next 1:1 with a clear answer.

"What we shipped last quarter is a routing workflow with sentiment-aware escalation. It works. It's cheap. It scales linearly. Most of what's described in those articles as 'agentic transformation' is the same shape we already shipped, with different vocabulary. Where we *would* benefit from a real agent — autonomous loop, tool use, unpredictable trajectories — is in the deal-research workflow that the AEs have been asking for. That one has the right shape. Want me to scope it?"

The CEO said yes.

Sam's arc this module: **the realization that workflows in a trench coat is most of the field, and that naming things accurately is the first engineering act.**

Sam will appear in every module. The arc compounds. By Module 18, Sam will have shipped a real multi-agent system — but only after building all the things that aren't agents first, and learning when not to.
:::

---

## What's next

Module 2 takes the minimal agent loop from Section 4 and hardens it into something you'd ship. Error handling, budget caps, partial failures, observability, the agent's relationship with its own context, and the operational instincts that turn "code that runs once successfully in a notebook" into "code that survives production." It's where the book stops talking about what agents are and starts talking about how to build them.

Before Module 2 — really sit with the field guide in Section 6. The next 17 modules assume you can apply it. If you can't, the rest of the book is going to feel like it's solving the wrong problems. The discrimination between augmented call, workflow, and agent is the foundation. Build on it.

Go ship something simple first. Then come back.
