# Module 3 Outline — Workflow Patterns You Should Reach for First

::: chapter-opener
<div class="module-num">MODULE 03 — OUTLINE</div>
<div class="module-title">Workflow Patterns You Should Reach for First</div>
<div class="subtitle">Most "agent" problems are workflow problems.<br>Here are the five patterns that solve 80% of them.</div>
<div class="pages">Target length: ~36 pages (heavy code)</div>
:::

## What this module is

The most actionable module in the book. We code all five workflow patterns from Anthropic's "Building effective agents" reference (still the cleanest taxonomy in the field). Each pattern: real use case, full implementation, cost/latency benchmark, when to upgrade to an agent.

By the end, the reader has a five-pattern toolbox they can deploy directly. More importantly, they have a default question: *"Can this be a workflow?"* The answer is usually yes.

::: hook
"You don't reach for an agent because the task is hard. You reach for an agent because the task's *shape* is unknown until you start. Most tasks aren't like that — engineers just haven't looked carefully."
:::

---

## Section 1 — The Five Patterns at a Glance (≈3 pages)

A table-and-diagram tour before the code. The five patterns:

| Pattern | What it does | When to use | Module 3 use case |
|---------|--------------|-------------|-------------------|
| **Prompt Chaining** | Sequential LLM calls, each output feeds the next | Task decomposes into ordered subtasks, each verifiable | Translate → critique → polish |
| **Routing** | One LLM picks which downstream prompt/tool/model handles the input | Inputs cluster into known categories | Customer support classifier |
| **Parallelization** | Run multiple LLM calls in parallel; aggregate | Subtasks independent, OR you want voting/sectioning | Code review (security, perf, style) |
| **Orchestrator-Workers** | Lead LLM dynamically delegates to worker LLMs | Subtasks aren't predetermined but composable from a known pool | Research with sub-questions |
| **Evaluator-Optimizer** | One LLM produces, another critiques, loop until pass | Quality bar matters more than latency | Iterative refinement |

We show a small visual diagram for each (we'll redraw these as hand-drawn-style SVGs at draft time).

::: brain
Look at the table. Three of the five patterns can *contain* an agent (a worker that's an agent, or an evaluator that's an agent). Two cannot. Which two? Why?

(Hint: one of these patterns is itself just an agent in disguise.)
:::

::: nodumbq
**Q: Why is "orchestrator-workers" a workflow if the orchestrator is dynamically choosing?**

Because the workers are predefined. The orchestrator chooses *which* worker, but the worker pool is a fixed set of capabilities. Compare to a true multi-agent system where workers can spawn workers, request new tools, or hand off to peers — that's Module 14 territory.

**Q: Isn't evaluator-optimizer just an agent loop with two roles?**

It's close, but the structure is fixed: produce → critique → revise → produce → critique. The agent loop has the model choosing what to do at each turn. Evaluator-optimizer is rigid; you know it's going to alternate.
:::

---

## Section 2 — Prompt Chaining (≈4 pages)

The simplest pattern. One LLM's output feeds the next.

**Use case:** Translate a marketing email from English to Japanese, with cultural adaptation.

```python
def translate_with_polish(email_en: str) -> str:
    # Step 1: literal translation
    literal = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        messages=[{"role": "user", "content": f"Translate to Japanese, literal:\n{email_en}"}],
    ).content[0].text

    # Step 2: cultural critique
    critique = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content":
            f"Original:\n{email_en}\n\nLiteral translation:\n{literal}\n\n"
            "Identify any phrasings that are culturally awkward in Japanese business context. "
            "Suggest specific replacements. Be concrete."
        }],
    ).content[0].text

    # Step 3: polish
    polished = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content":
            f"Original:\n{email_en}\n\nLiteral:\n{literal}\n\nCritique:\n{critique}\n\n"
            "Produce the final, culturally-adapted translation. Output only the translation."
        }],
    ).content[0].text

    return polished
```

**Why three steps and not one?** The model is better at evaluation than self-evaluation in one pass. Splitting the literal translation from the cultural critique forces explicit reasoning about cultural fit. Empirically, ~20% better on human evaluation in Japanese marketing contexts (we'll re-verify when drafting).

**Adding a gate:** sometimes step 2 should produce a "skip step 3" signal if the literal translation is already fine. We show that variant.

**Latency:** ~3x a single call (sequential). **Cost:** ~3x. **Quality on this task:** measurably higher.

::: gotcha
Prompt chaining is tempting to over-decompose. If you have eight steps, ask: which can collapse? Which are model-quality limited (need their own call) vs. just-want-cleaner-prompts (collapse them)?
:::

---

## Section 3 — Routing (≈4 pages)

One LLM classifies the input; downstream handlers are specialized.

**Use case:** Customer support triage. Inputs are emails. Outputs are routed to: billing-handler, technical-handler, refund-handler, escalate-to-human.

```python
ROUTER_PROMPT = """Classify this support email into exactly one category:
- billing: invoices, charges, subscriptions
- technical: bugs, errors, integration help
- refund: explicit refund requests
- escalate: complex, sensitive, or unclear cases

Email:
{email}

Respond with only the category name."""

def route_and_handle(email: str) -> Response:
    category = client.messages.create(
        model="claude-haiku-4-5-20251001",  # cheap classifier
        max_tokens=20,
        messages=[{"role": "user", "content": ROUTER_PROMPT.format(email=email)}],
    ).content[0].text.strip().lower()

    handlers = {
        "billing": handle_billing,
        "technical": handle_technical,
        "refund": handle_refund,
        "escalate": handle_escalate,
    }
    return handlers.get(category, handle_escalate)(email)
```

**Why route?** Each downstream prompt is shorter and more specific than a unified prompt. A monolithic "support assistant" prompt is 800 tokens; the routed billing handler is 250. Quality is higher because the prompt isn't trying to do five jobs.

**Variants:**

1. **Model routing:** route to a stronger model when the classifier flags difficulty. (Prelude to Module 6, where routing is the primary topic.)
2. **Confidence routing:** classifier returns a probability; below threshold goes to "escalate."
3. **Multi-label routing:** input can match multiple categories (rare; usually an indication you need a different pattern).

::: postmortem
**The Router That Sent Refunds to Dev**

A SaaS team built a routing workflow for support. Categories: billing, technical, refund, escalate. The classifier prompt didn't enumerate categories, it just said "categorize this." The model returned "feature_request," "complaint," "thanks," "refund" — categories that didn't exist in the dispatch table. The dispatch dict's `.get()` defaulted to "technical." Dev got 2,000 refund requests in their queue over a week.

**Lesson:** the classifier's output space must be explicit AND the dispatch must enforce it. Use enums. Use Pydantic. Use the API's structured output. Anything except `dict.get()`.
:::

::: code-exercise
**Exercise 3.1 — Routing with structured output.**

Refactor the router above to use Anthropic's tool-use mechanism for structured output (force the model to call a `classify` tool with a schema-validated category enum). Compare reliability before/after on a deliberately ambiguous test set we provide.
:::

---

## Section 4 — Parallelization (≈5 pages)

Run multiple LLM calls in parallel. Two flavors:

**Sectioning:** independent subtasks aggregate into a final answer.
- Code review: one call for security, one for performance, one for style. Combine.
- Document Q&A: split a 200-page doc into chunks, parallelize "find relevant info" across chunks, aggregate.

**Voting:** same task, multiple times, take majority/best.
- Hard math problems: 5 reasoning attempts, vote on answer.
- Faithfulness checks: run the same eval prompt 3x with different temperatures, take majority.

```python
import asyncio
from anthropic import AsyncAnthropic

async def parallel_code_review(diff: str) -> CombinedReview:
    client = AsyncAnthropic()
    perspectives = [
        ("security", "Review this diff for security vulnerabilities..."),
        ("performance", "Review this diff for performance issues..."),
        ("style", "Review this diff for style and maintainability..."),
    ]
    tasks = [
        client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": f"{prompt}\n\nDiff:\n{diff}"}],
        )
        for _, prompt in perspectives
    ]
    results = await asyncio.gather(*tasks)
    return combine_reviews([(p[0], r.content[0].text) for p, r in zip(perspectives, results)])
```

**Latency:** wall-clock = max of parallel calls (close to 1 call's latency). **Cost:** ~N× a single call.

**Aggregation patterns:**
- *Concatenate* (when subtasks cover different ground): just stitch.
- *Reduce* (when results need synthesis): one final LLM call to merge.
- *Vote* (when calls did the same thing): take majority/highest-confidence.

::: brain
You're parallelizing 50 chunks of a long document for a Q&A task. Aggregation is "reduce" — one final synthesis call. The reduce call's input is now 50 summaries. Will it fit in context?

(Hint: this is when parallelization meets compression. We come back to this in Module 8.)
:::

**Voting variant for hallucination reduction (preview of Module 10):**

```python
async def vote_for_consistency(question: str, n_votes: int = 5) -> Answer:
    client = AsyncAnthropic()
    tasks = [
        client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            temperature=0.7,
            messages=[{"role": "user", "content": question}],
        )
        for _ in range(n_votes)
    ]
    answers = await asyncio.gather(*tasks)
    extracted = [extract_answer(a.content[0].text) for a in answers]
    return most_common(extracted)
```

This is Best-of-N reranking, an ACL Findings 2025 result that meaningfully reduces hallucination on factual tasks. We cite and verify when drafting.

::: gotcha
Parallel calls don't share state. If you parallelize tool-using LLMs, each gets its own conversation. Shared tool-result memory across parallel branches is a thing you build (Module 9), not a thing you get for free.
:::

---

## Section 5 — Orchestrator-Workers (≈5 pages)

A lead LLM decomposes a task and dispatches to worker LLMs. Workers are scoped, prompt-specialized, and return structured results.

**Use case:** Research synthesis. User asks: "Compare the carbon footprints of three given products and recommend the lowest." Orchestrator decides to spawn three "research one product" workers, then a "compare and recommend" worker.

The shape:

```
                    [Orchestrator]
                          |
          decides: 3 product workers + 1 compare worker
                          |
        ┌─────────────┬───┴────┬─────────────┐
        v             v        v             v
  [Worker A]    [Worker B] [Worker C]   (after A,B,C return)
                                         [Compare Worker]
```

```python
@dataclass
class Subtask:
    role: str
    prompt: str
    tools: list[str] = field(default_factory=list)

def orchestrator_workers(query: str) -> str:
    # Orchestrator decides
    plan_resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        tools=[PLAN_TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": "create_plan"},
        messages=[{"role": "user", "content":
            f"Decompose this task into subtasks. Each subtask is independent and "
            f"can be handled by a worker LLM with the available tools.\n\nTask: {query}"
        }],
    )
    plan: list[Subtask] = parse_plan(plan_resp)

    # Workers execute
    results = run_workers_parallel(plan)

    # Synthesize
    synthesis = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content":
            f"Original task: {query}\n\n"
            f"Worker results:\n{format_results(results)}\n\n"
            f"Produce the final answer."
        }],
    )
    return synthesis.content[0].text
```

**Where this becomes an agent:** if the orchestrator can decide to spawn *new* worker types not in its plan, or workers can request additional tools/workers, you've crossed into multi-agent (Module 14). This pattern keeps a fixed worker pool and a one-shot plan.

::: fireside
**FIRESIDE CHAT: "Orchestrator-Workers" vs "Single Agent with Tools"**

::: cast
<div class="character architect"><span class="name">Architect:</span> Why are we doing orchestrator-workers? An agent could just call the same tools.</div>

<div class="character agent"><span class="name">Orchestrator-Workers:</span> Three reasons. One: the orchestrator's plan is explicit and inspectable. You can log it, eval it, hand it to humans for approval. Two: workers run in parallel. Three: workers have isolated context — they don't see each other's noise.</div>

<div class="character architect"><span class="name">Architect:</span> What about the cases where the agent's flexibility wins?</div>

<div class="character agent"><span class="name">Orchestrator-Workers:</span> When the plan needs to revise mid-flight based on what workers find. I can't do that. The orchestrator commits, workers execute. If a worker finds something that should change the plan, it just returns it and the synthesis call has to handle it.</div>

<div class="character architect"><span class="name">Architect:</span> So the rule is: known decomposition → me. Discovered decomposition → agent.</div>

<div class="character agent"><span class="name">Orchestrator-Workers:</span> And known decomposition is more common than people think.</div>
:::
:::

---

## Section 6 — Evaluator-Optimizer (≈5 pages)

One LLM produces output. Another critiques. The producer revises. Loop until the critic passes.

**Use case:** producing legal contract clauses that pass a compliance check. Producer drafts. Evaluator checks against a checklist (cites missing clauses, ambiguous language, jurisdiction issues). Producer revises. Iterate up to N times or until evaluator returns "pass."

```python
def evaluator_optimizer(task: str, max_rounds: int = 3) -> str:
    draft = produce(task)
    for round_i in range(max_rounds):
        critique = evaluate(task, draft)
        if critique.passed:
            return draft
        draft = revise(task, draft, critique.feedback)
    return draft  # accept best-effort

def produce(task: str) -> str:
    return client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": f"Draft for: {task}"}],
    ).content[0].text

def evaluate(task: str, draft: str) -> Critique:
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        tools=[CRITIQUE_TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": "submit_critique"},
        messages=[{"role": "user", "content":
            f"Task: {task}\nDraft:\n{draft}\n\n"
            "Evaluate against compliance checklist. Submit critique."
        }],
    )
    return parse_critique(resp)

def revise(task: str, draft: str, feedback: str) -> str:
    return client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content":
            f"Task: {task}\n\nPrevious draft:\n{draft}\n\nFeedback:\n{feedback}\n\nRevise."
        }],
    ).content[0].text
```

**Critical design choice:** different models for producer and evaluator. The classic mistake is using the same model for both — it tends to validate its own work. Use Opus to evaluate Sonnet's output, or use a fine-tuned classifier as the evaluator.

**Termination:** must have a `max_rounds` cap. Otherwise: infinite loops where the evaluator keeps finding new issues.

::: gotcha
Evaluator-optimizer cost compounds fast. 3 rounds × (produce + evaluate + revise) = 7 LLM calls minimum. Use cheaper models when possible. Use the cheapest evaluator that still catches the relevant issues.
:::

::: brain
The "evaluator validates its own work" problem isn't fully solved by using a different model. Why? What's a better fix?

Hint: think about how the evaluator's prompt was generated.
:::

---

## Section 7 — Cost & Latency Comparison (≈3 pages)

A real benchmark. Same task ("summarize this 10-page report into 3 bullet key insights") implemented seven ways:

| Approach | Avg Latency | Avg Cost | Subjective Quality (1-5) |
|----------|-------------|----------|--------------------------|
| Single LLM call (Sonnet) | 4.2s | $0.018 | 3.6 |
| Single LLM call (Opus) | 9.7s | $0.071 | 4.4 |
| Prompt chain (extract → distill → polish) | 11.3s | $0.039 | 4.5 |
| Routing (classify → specialized prompt) | 6.1s | $0.022 | 4.2 |
| Parallel sectioning (3 perspectives → reduce) | 5.8s | $0.048 | 4.3 |
| Orchestrator-workers | 14.2s | $0.094 | 4.4 |
| Agent loop (Module 2 hardened) | 31.7s | $0.41 | 4.4 |

(Numbers are illustrative — we'll run real benchmarks with current models when drafting.)

**The headline:** for this task, prompt chain matches Opus single-call quality at half the cost. The agent loop is 22× more expensive than prompt chain for indistinguishable quality.

::: pullquote
"Better quality" and "agent" aren't synonyms. They aren't even correlated.
:::

---

## Section 8 — When to Upgrade to an Agent (≈3 pages)

Ironically, the workflow patterns module ends with "and here's when you should not use them."

You're past workflow territory when:
- The set of subtasks isn't enumerable in advance
- Subtasks themselves spawn subtasks
- The natural decomposition depends on intermediate results in ways you can't predict
- Tool selection itself is the hard part of the task
- You need true exploration

You're still in workflow territory when (despite gut feeling):
- The task is "complex" but decomposable
- You want flexibility but the flexibility is enumerable
- You think you need an agent because the prompt would be long (compose workflows; don't reach for agents)

::: exercise
Classify each as workflow (which pattern?) or agent.

1. Generate alt-text for 10K product images.
2. Help a developer debug an issue by exploring their codebase, running commands.
3. Extract structured data from invoices in 12 known formats.
4. Plan a 5-day itinerary for a trip given preferences.
5. Investigate why a specific ML training run produced anomalous loss curves.
6. Translate documentation from English to French while preserving formatting.

<details class="answer"><summary>show answers</summary>

1. **Workflow — single LLM call** per image, parallelized.
2. **Agent.** Tool selection and exploration are the task.
3. **Workflow — routing.** Twelve formats means twelve specialized handlers.
4. **Workflow — prompt chaining or orchestrator-workers.** Plan is decomposable: research destinations, build day-by-day, optimize logistics.
5. **Agent.** "Investigate" plus "anomalous" — you don't know what you'll find.
6. **Workflow — prompt chaining or single call** with structured output.

Most "agent" candidates collapse to workflows under inspection. The exceptions (2 and 5) are genuine agent territory: tool-driven exploration of unknown shape.
</details>
:::

---

## Section 9 — What's Next (≈1 page)

Module 4 covers tools — the surface where agents live or die.

Module 5 introduces MCP, which makes tool sets dynamic.

Module 6 brings model routing, which extends the routing pattern from this module to a first-class architectural concern.

By Module 13 we'll revisit these workflow patterns and ask: when does going multi-agent actually beat the orchestrator-workers pattern? Spoiler: less often than you'd think.

---

## Recurring elements used in this module

- 1 chapter opener
- 1 hook
- 2 No Dumb Questions
- 2 Brain Power
- 1 Fireside Chat (Orchestrator-Workers vs Single Agent)
- 1 Production Postmortem (router default-route disaster)
- 2 Watch It! blocks
- 2 Sharpen Your Pencil exercises
- 1 Code Exercise (structured output routing)
- 1 Pullquote
- ~6 substantial code blocks (one per pattern + benchmarks)
- 1 Bullet Points recap

::: sam-arc
Sam, fresh from hardening their agent loop in Module 2, encounters their PM in Section 1 with a "we need an agent for this" request. Sam audits the actual task, realizes it's a routing workflow, ships in 2 days instead of 2 weeks, saves the company $40K/month. Sam's arc this module: **restraint**. Knowing when not to reach for the shiny tool.
:::

::: page-budget
S1 (Patterns at a glance): 3p
S2 (Prompt chaining): 4p
S3 (Routing): 4p
S4 (Parallelization): 5p
S5 (Orchestrator-Workers): 5p
S6 (Evaluator-Optimizer): 5p
S7 (Cost/latency): 3p
S8 (When to upgrade): 3p
S9 (Next): 1p
Recurring elements (woven in): 3p
TOTAL: ~36 pages
:::

::: sources
**Must verify when drafting:**

- Anthropic, "Building effective agents" (Dec 2024) — the canonical source for these five patterns
- Best-of-N hallucination reduction — ACL Findings 2025 paper, verify exact result claims
- Real benchmark numbers (cost/latency table in S7) — re-run with current pricing and current Sonnet/Opus/Haiku versions before drafting
- Anthropic API tool-use schema for structured output (S3 exercise) — verify current syntax

**Stable knowledge:**
- The five patterns themselves
- Sectioning vs. voting variants of parallelization
- Termination requirements for evaluator-optimizer
- Decomposition vs. exploration distinction

**Cross-references to lock down:**
- Module 6 will deepen "model routing" — make sure framing here doesn't preempt it
- Module 8 (compression) will revisit the parallelization aggregation problem
- Module 14 (multi-agent architectures) will compare orchestrator-workers vs. true multi-agent
:::

::: bullet-points
### Module 3 in eight bullets

(filled at draft time)

- Five workflow patterns cover ~80% of "agent" use cases
- Prompt chaining: ordered decomposition, +quality at +cost
- Routing: cheap classifier → specialized handlers; structured output prevents disasters
- Parallelization: sectioning (different jobs) and voting (same job, multiple times)
- Orchestrator-workers: lead LLM plans, workers execute in parallel, synthesizer reduces
- Evaluator-optimizer: produce-critique-revise; different models for producer and evaluator
- Cost & quality benchmarks show workflows often match agent quality at fraction of cost
- Reach for an agent only when subtasks aren't enumerable or tool selection IS the task
:::
