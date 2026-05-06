# Module 6 — Choosing Models / Routing as Architecture

::: chapter-opener
<div class="module-num">MODULE 6</div>
<div class="module-title">Choosing Models / Routing as Architecture</div>
<div class="subtitle">The model field in your API call is an architectural decision.<br>Most teams treat it as a default. The bill notices.</div>
<div class="pages">~38 pages · the cost-quality lever you've been ignoring</div>
:::

::: hook
Sam opened the MCP gateway's cost dashboard on a Tuesday morning. The deal-research server had been running across four teams for two weeks. The cost line was steeper than expected — roughly 2.4× the projection.

Sam clicked into the breakdown. By tool: company_lookup, competitor_analysis, market_research, brief_synthesizer, brief_evaluator. By model: 73% of spend on Opus 4.7. 24% on Sonnet 4.6. 3% on Haiku 4.5.

That distribution made sense for *some* of the calls. The brief synthesizer was Opus on purpose; that's where the quality mattered. The brief evaluator was Opus too — the rubric was nuanced. But why was company_lookup hitting Opus? It was a database lookup. Haiku could do it.

Sam pulled the trace for one Opus call to company_lookup. The reason was banal: Sam had written `model="claude-opus-4-7"` in the tool's wrapper code, copied from the synthesizer pattern, and never thought about it again. Five tools. All Opus. All set six weeks ago when "claude-opus-4-7" was the latest model and Sam had thought "use the best."

Sam ran the math. company_lookup was getting called ~600 times a day across four teams. The actual reasoning required was zero — the tool fetched data and returned it. Switching to Haiku 4.5 cost the company 80% less per call with no quality difference. Same for competitor_analysis (also a fetch). And for market_research (a search wrapper).

Just by changing the `model` parameter in three tools, Sam cut the deal-research system's daily cost by 41%. Quality stayed the same. Latency improved.

Sam wrote it up in Slack: *"Spent four hours moving three tools from Opus to Haiku. Saves us $1,800/month. The model parameter is an architectural decision; I'd been treating it like a default."*

A reply from the platform team lead: *"You should write that up. We have ten more agents in production that probably look the same."*
:::

---

## What this module is

The previous five modules handled `model="claude-sonnet-4-6"` (or whatever) as a constant. We picked it once, used it everywhere, moved on. That was deliberate — the pedagogy was about loops, workflows, tools, and protocols, not about model selection.

It's time to bring model selection forward. Sam's discovery is the rule, not the exception: most production agent systems pick a model early in development (usually whichever was newest or most capable at the time) and then never re-examine the choice. The cost is enormous, the quality gap is often nonexistent, and the fix is sometimes literally three lines of code.

This module:

- Walks through the current Claude lineup as of mid-2026 (Opus 4.7, Sonnet 4.6, Haiku 4.5) and what each is actually good at
- Builds the cost-quality framework: when each model earns its tier, when it doesn't
- Covers four routing architectures in code: static role-based, dynamic per-call, cascade with fallback, and adaptive routing based on observed results
- Treats model upgrades as an engineering concern: how to migrate, what to test, how to A/B safely
- Discusses cross-provider routing (when you'd use a non-Claude model in your agent system) honestly

By the end, the `model` parameter in every API call you write will be a deliberate choice you can justify, and you'll have the patterns to make those choices automatic across a system rather than per-developer.

---

## Section 1 — The Lineup, As It Stands

The Claude family is structurally simple by design: three tiers, each meaningfully different. As of May 2026, the current generation is:

| Model | API ID | Input $/M | Output $/M | Context | Max Output |
|---|---|---|---|---|---|
| Opus 4.7 | `claude-opus-4-7` | $5.00 | $25.00 | 1M | 128K |
| Opus 4.6 | `claude-opus-4-6` | $5.00 | $25.00 | 1M | 128K |
| Sonnet 4.6 | `claude-sonnet-4-6` | $3.00 | $15.00 | 1M | 128K |
| Haiku 4.5 | `claude-haiku-4-5` | $1.00 | $5.00 | 200K | 64K |

Three things worth being explicit about:

**Opus 4.7 vs Opus 4.6.** Same sticker price, but Opus 4.7 ships with a new tokenizer that produces up to 35% more tokens for the same input text. Effective cost can rise even though the rate card hasn't. If you're migrating from 4.6 to 4.7 specifically for the upgrade, benchmark cost-per-task, not cost-per-token.

**The 5×/3× ratio.** Each tier is roughly 3-5× the cost of the one below it on output tokens. This consistency is a feature: it makes back-of-envelope cost modeling straightforward. If you know your input cost on Sonnet, multiply by ~1.67 for Opus and divide by 3 for Haiku.

**Context windows differ.** Sonnet and Opus both support 1M-token context at standard pricing. Haiku is capped at 200K. If a task genuinely needs long-context reasoning, Haiku is out — period. If 200K suffices (which is almost always), Haiku stays in play.

### What each is for

The headlines from the benchmarks and the production reports converge on a clean picture:

**Haiku 4.5** — fast, cheap, surprisingly capable. SWE-bench Verified ~73%; SWE-bench Pro ~40%. That puts it ahead of GPT-5.1 on coding and within the noise of GPT-5.2 Codex. For routing, classification, extraction, summarization, simple lookups, structured output, and high-volume tasks where the reasoning is shallow, Haiku is the right default. The 5× cost gap below Sonnet matters at scale.

**Sonnet 4.6** — the workhorse. SWE-bench Verified 79.6%, within 1.2 points of Opus 4.6. 70% of developers in survey work prefer it for daily coding. For most production agent tasks — RAG responses, content generation, standard tool use, multi-step workflows — Sonnet is the right default. Anthropic's own guidance is "default to Sonnet; specialize away."

**Opus 4.7** — the precision tier. SWE-bench Verified ~80%; the strongest model on graduate-level reasoning (GPQA Diamond), the strongest on agentic desktop control (OSWorld), the strongest on multi-file refactoring and architectural decisions. Reach for Opus when the task genuinely benefits from depth: orchestrators in multi-agent systems, evaluators in evaluator-optimizer workflows, the few percent of work where being measurably better matters more than the 1.67× cost.

The cleanest one-line summary I've seen: **Sonnet covers most of what you do; Haiku covers most of what you do at volume; Opus covers the work where being right matters more than being cheap.**

::: pullquote
Default to Sonnet. Specialize Haiku for volume. Reserve Opus for the cases where the quality delta is measurable. The wrong model is a tax you pay forever.
:::

::: nodumbq
**Q: What about extended thinking? When does that change the calculus?**

Extended thinking (available on Sonnet 4.6 and Opus, not Haiku) lets the model reason longer before producing the final output. It improves quality on hard problems significantly but adds 5-30 seconds of latency and more output tokens (the thinking counts). Use cases: complex reasoning, debugging, mathematical work, anything where you'd otherwise pay a human to think for two minutes. Don't use it for: response generation, simple tool calls, classifications. The extended thinking decision is orthogonal to the model selection decision; you make both per-call.

**Q: Is there a fourth tier — something between Haiku and Sonnet?**

Anthropic ships three tiers. There isn't a 4.5 between Haiku 4.5 and Sonnet 4.6. The structural choice — three tiers spaced by 3-5× cost — is deliberate and unlikely to change. If you find yourself wishing for an in-between tier, the right move is usually prompt caching on Sonnet (which can cut Sonnet's effective cost to near-Haiku territory for cacheable workloads) rather than searching for a model that doesn't exist.
:::

---

## Section 2 — Cost-Quality Math, Specific

Sam's company_lookup case got a 5× cost reduction by switching one model parameter. Was that lucky, or is the gap that big in general?

It's that big in general. Concrete walkthrough on a representative agent workload.

A coding agent session: 50 prompts averaging 2,000 input tokens and 1,000 output tokens each. The cost per session at each tier:

- **All Opus 4.7:** (50 × 2K × $5/M) + (50 × 1K × $25/M) = $0.50 + $1.25 = **$1.75 per session**
- **All Sonnet 4.6:** (50 × 2K × $3/M) + (50 × 1K × $15/M) = $0.30 + $0.75 = **$1.05 per session**
- **All Haiku 4.5:** (50 × 2K × $1/M) + (50 × 1K × $5/M) = $0.10 + $0.25 = **$0.35 per session**

For 10,000 sessions a month: $17,500 / $10,500 / $3,500. The Opus-to-Haiku gap is $14,000/month for the same volume. The Opus-to-Sonnet gap is $7,000/month.

Now layer in routing. The empirical distribution that production systems converge on, when teams actually measure: roughly **60-70% of requests handled by Haiku, 25-30% by Sonnet, 3-5% by Opus.** Plug those proportions into the same workload:

- **Cascade routing (65/30/5):** $0.35×0.65 + $1.05×0.30 + $1.75×0.05 = $0.23 + $0.32 + $0.09 = **$0.64 per session**

That's 63% cheaper than all-Sonnet, 37% cheaper than even Sonnet, and 81% cheaper than all-Opus. For 10,000 sessions/month: $6,400 vs the $17,500 all-Opus baseline — over $11,000/month saved with no measurable quality difference if the routing is right.

The "if the routing is right" caveat is everything. Section 4 builds the routing patterns; Section 7 covers how to know whether your routing is right.

::: brain
A team is running 100,000 user sessions a month, all on Opus 4.7. They estimate 70% are simple Q&A that Haiku could handle. The cost-cutting opportunity in absolute terms?

(Math: 70K sessions × ($1.75 - $0.35) = $98,000/month, or roughly $1.2M/year. That's a senior-engineer-salary-and-a-half worth of savings sitting in the model parameter. Most teams don't quantify this. They should.)
:::

### What about prompt caching?

Prompt caching changes the math significantly for any workload with repeated prefixes — system prompts, document context, few-shot examples, tool definitions. Cached input tokens cost ~10% of the standard rate (90% savings on cache reads).

For an agent with a 5K-token system prompt and tool definitions, repeated across many turns: cache the prefix, and your effective input cost drops to roughly 10% on the cacheable portion. A Sonnet system at $3/M input becomes effectively $0.50/M for the cacheable parts plus full price for the variable parts.

This matters for the Haiku-vs-Sonnet decision specifically. A Sonnet workload with heavy caching can drop to near-Haiku effective cost while keeping Sonnet's reasoning quality. Before defaulting Haiku for a workload, check whether caching would close the gap; if it would, you might prefer Sonnet at cached rates.

The caching decision is orthogonal to the model decision but interacts with it. Module 8 covers context engineering and caching patterns in depth.

---

## Section 3 — When Each Model Earns Its Tier

A field guide for the model selection conversations you'll have on your team.

### Reach for Haiku 4.5 when:

- The task is **classification** — routing, sentiment, intent detection, content categorization
- The task is **extraction** — structured output from text, named entities, fact pulling
- The task is **summarization** of straightforward content (not nuanced editorial work)
- The task is **lookup or fetch** — wrapping an API or database call where reasoning is shallow
- The task is **per-step in a workflow** where another model handles the hard part (Sam's case — Haiku does the data fetch, Sonnet does the synthesis)
- **Volume matters** — high-throughput pipelines where 5× cost difference scales to real money
- **Latency matters** — Haiku's tokens-per-second is roughly 2× Sonnet's

### Reach for Sonnet 4.6 when:

- **Most production tool use** — Sonnet is the default for agent loops with tools
- **Generation tasks of moderate length** — drafting emails, blog posts, structured content
- **Multi-step reasoning** that doesn't require deep planning
- **Standard coding tasks** — Sonnet hits 79.6% on SWE-bench Verified, only 1.2 points behind Opus 4.6
- **Conversational agents** with reasonable complexity
- **Workflow steps in the middle of a pipeline** where neither shallow lookup (Haiku) nor deep synthesis (Opus) fits
- **The default**, when you don't have a specific reason to pick another tier

### Reach for Opus 4.7 when:

- **Multi-file architectural decisions** — refactoring, system design, complex code review
- **Graduate-level reasoning** — scientific analysis, formal proofs, intricate mathematical work
- **Orchestrators in multi-agent systems** — the planning layer where errors compound (Module 14)
- **Evaluator-optimizer evaluators** with nuanced criteria (Module 3 Section 6)
- **Long-horizon agentic coding** — the Anthropic-described use case for 4.7 specifically
- **Brand-voice or tone-critical content** where the difference between "good" and "great" affects revenue
- The task has **clear evidence in your evals** that Opus produces materially better results than Sonnet

That last bullet matters. Without an eval that demonstrates Opus-Sonnet quality difference for *your* task, you're paying 1.67× cost for theoretical benefit. Sometimes the benefit is real; sometimes it's noise. Measure.

::: gotcha
The most common Opus over-use pattern: a developer prototypes with Opus, gets good results, ships with Opus, never tries Sonnet. The mental model is "Opus is the safe choice; we can downgrade later." But "later" rarely comes; the system runs on Opus for months. The cheaper move: prototype with Sonnet, upgrade specific calls to Opus *if and when* you have evidence of a quality gap. Default down, specialize up.
:::

### What about Haiku 4.5 for coding?

The conventional wisdom — "Haiku is for classification and lookup, not for code" — is partially out of date. Haiku 4.5 hits 73% on SWE-bench Verified. That's better than GPT-5.1 and most open-source models, and within striking distance of Sonnet 4.6 on the easier portion of the benchmark.

The split: Haiku handles routine coding well — boilerplate generation, simple bug fixes, code formatting, syntax-level transformations. It struggles with multi-file changes, deep architectural reasoning, and tasks requiring careful attention to subtle constraints across a large context.

For *coding agents* specifically, the right pattern often is: Haiku for the routine 60% of operations, Sonnet for the rest, Opus reserved for the architectural moments. Module 14's coding-agent example will use exactly this distribution.

---

## Section 4 — Four Routing Architectures

The architectural question: how does your system *decide* which model to use? Four patterns, in increasing sophistication.

### Pattern 1: Static role-based routing

The simplest and the right starting point. Each role in your system has a fixed model. The orchestrator is Opus. The workers are Sonnet. The classifier is Haiku. Hardcoded; explicit; auditable.

```python
# config.py — model selection by role
MODEL_BY_ROLE = {
    "orchestrator": "claude-opus-4-7",
    "worker": "claude-sonnet-4-6",
    "synthesizer": "claude-opus-4-7",
    "router": "claude-haiku-4-5",
    "classifier": "claude-haiku-4-5",
    "evaluator": "claude-opus-4-7",
    "fetcher": "claude-haiku-4-5",
}


async def call_model(role: str, **kwargs):
    model = MODEL_BY_ROLE[role]
    return await client.messages.create(model=model, **kwargs)


# Usage in your agent system:
plan = await call_model("orchestrator", messages=[...], tools=...)
results = await asyncio.gather(*[
    call_model("worker", messages=[...]) for subtask in plan.subtasks
])
final = await call_model("synthesizer", messages=[...])
```

When this works:

- The roles map cleanly to complexity tiers
- The model choices are stable across requests
- You want changes to be one-place and reviewable

When this falls short:

- Some "worker" calls are simple and should use Haiku; some "fetcher" calls are complex and should use Sonnet
- Volume in one role swamps the others (you'd want different routing for that role specifically)
- The right model for a call depends on the *content* of the call, not just its role

For most systems, static role-based routing is the right v1. It captures 80% of the cost optimization gain with minimal complexity. Sam's deal-research system, after Sam's three-line fix in the cold open, became this pattern.

### Pattern 2: Per-call dynamic routing

A small router (Haiku) reads the request and picks the model for the actual work. The router itself is cheap; the savings come from getting the work-tier right.

```python
from enum import Enum
from pydantic import BaseModel, Field


class ModelTier(str, Enum):
    HAIKU = "claude-haiku-4-5"
    SONNET = "claude-sonnet-4-6"
    OPUS = "claude-opus-4-7"


class RoutingDecision(BaseModel):
    tier: ModelTier
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


ROUTING_SYSTEM_PROMPT = """
You are a model routing system. Given a user request, decide which model tier to use:

- HAIKU: simple lookups, classifications, extractions, formatting, routine summaries
- SONNET: standard reasoning, content generation, multi-step but not deeply nuanced tasks
- OPUS: complex multi-step reasoning, architectural decisions, expert-level analysis, evaluation with nuanced criteria

Default to SONNET. Pick HAIKU when the task is shallow. Pick OPUS only when there's a clear reasoning need.

Return your decision with confidence; low confidence (<0.7) should default to SONNET.
""".strip()


async def route_request(user_request: str) -> RoutingDecision:
    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=400,
        system=ROUTING_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_request}],
        tools=[{
            "name": "submit_routing",
            "description": "Submit your routing decision.",
            "input_schema": RoutingDecision.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": "submit_routing"},
    )
    block = next(b for b in response.content if b.type == "tool_use")
    decision = RoutingDecision(**block.input)
    if decision.confidence < 0.7:
        decision = RoutingDecision(
            tier=ModelTier.SONNET,
            reasoning=f"Low confidence on initial routing; defaulting to SONNET. ({decision.reasoning})",
            confidence=decision.confidence,
        )
    return decision


async def handle_with_routing(user_request: str) -> str:
    decision = await route_request(user_request)
    response = await client.messages.create(
        model=decision.tier.value,
        max_tokens=4096,
        messages=[{"role": "user", "content": user_request}],
    )
    return response.content[0].text
```

This is the routing workflow from Module 3 Section 3, applied at the model-selection layer. The router cost is small (Haiku is ~$0.001 per request); the work-tier savings are significant (Haiku-for-shallow vs Sonnet-for-everything is 3× cheaper).

When this works:

- Request types are heterogeneous and the router can reliably classify them
- The router cost is small relative to the work cost
- Audit-ability of routing decisions is acceptable (the decision is logged with reasoning)

When this falls short:

- Routing accuracy isn't high enough — wrong tier means worse output, not just slower
- The router itself becomes a bottleneck (latency adds up if every request hits the router)

For Sonnet/Haiku decisions, this is usually safe — getting it wrong rarely produces unusable output, just suboptimal cost. For Opus/Sonnet decisions, more care is needed; routing a complex request to Sonnet when Opus was warranted produces measurably worse work.

### Pattern 3: Cascade with confidence-based fallback

The premise: most tasks can be handled by a cheap model; some need a stronger model. Try the cheap one first; escalate when the result fails a quality check.

```python
async def cascade_completion(
    user_request: str,
    *,
    quality_check: callable,
    cascade: list[str] = None,
) -> tuple[str, str]:
    """
    Try each model in the cascade. Return as soon as quality_check passes.
    Returns (output, model_that_succeeded).
    """
    cascade = cascade or ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"]
    last_output = None
    for model in cascade:
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": user_request}],
        )
        output = response.content[0].text
        last_output = output
        if quality_check(output):
            return output, model
    return last_output, cascade[-1]
```

The `quality_check` is the load-bearing piece. Examples:

- **Schema validation:** for structured output, check that the response parses against your schema
- **Confidence floor:** for classification tasks, the model returns a confidence value; if < 0.8, escalate
- **Content checks:** specific things the response must contain (a JSON code block, a particular section header, a non-empty list)
- **A second-model evaluator:** a Haiku call that scores the output and decides if it's good enough

```python
def schema_quality_check(output: str) -> bool:
    """Does the output parse as valid JSON matching our expected schema?"""
    try:
        parsed = json.loads(output)
        ResultSchema.model_validate(parsed)
        return True
    except (json.JSONDecodeError, ValidationError):
        return False


async def cascade_with_schema(request: str):
    output, model = await cascade_completion(
        request,
        quality_check=schema_quality_check,
    )
    print(f"Succeeded on {model}")
    return output
```

When this works:

- The "cheap" model usually succeeds, and the cascade pays off statistically
- The quality check is fast and catches most failures
- Escalation is acceptable latency-wise

When this falls short:

- The quality check has too many false negatives (you escalate when you didn't need to)
- The cheap model fails *silently* — produces output that passes the check but is wrong (the worst case)
- The latency penalty when escalating happens often is unacceptable

The cascade pattern is best for tasks where there's a clean validation criterion. For tasks where "good output" is fuzzy and depends on what the user actually needed, cascading is harder to make work.

### Pattern 4: Adaptive routing based on observed results

The most sophisticated pattern. The router learns from past requests: if requests of a certain type kept failing on Sonnet, route similar future requests to Opus by default.

The simplest version is a static lookup table updated by your eval pipeline:

```python
# adaptive_routing.py — keep a learned per-pattern routing table
LEARNED_ROUTING: dict[str, str] = {
    # task_pattern → recommended model
    "code_review_python": "claude-sonnet-4-6",
    "code_review_typescript_complex": "claude-opus-4-7",
    "data_extraction_invoice": "claude-haiku-4-5",
    "data_extraction_legal_contract": "claude-sonnet-4-6",
    # ... etc, populated by analysis
}


def adaptive_route(task_pattern: str, default: str = "claude-sonnet-4-6") -> str:
    return LEARNED_ROUTING.get(task_pattern, default)
```

The richer version uses contextual signals — request length, presence of certain keywords, user history, time of day, recent failure rate — to dynamically pick the model. This is most useful at scale where the patterns are stable and the volume justifies the engineering.

For most teams, Patterns 1-3 cover the territory. Adaptive routing is overkill until you've exhausted the simpler patterns.

::: pullquote
Static role-based is the right v1. Per-call dynamic earns its complexity when request types are heterogeneous. Cascade earns its complexity when validation is cheap. Adaptive earns its complexity rarely — usually only at scale, after Patterns 1-3 are exhausted.
:::

---

## Section 5 — Routing Inside Agents and Workflows

So far the patterns have treated each request as one model call. Real systems have many model calls per request — every agent loop turn, every workflow step, every parallelization branch. Each is its own model selection decision.

The discipline: **assign the model at the smallest possible unit, not at the request level.**

In a workflow:

```python
async def deal_research_workflow(company: str, deal_type: str):
    # Routing: Haiku because it's a simple classification
    deal_path = await call_model("router", model="claude-haiku-4-5", ...)

    # Parallel research: Sonnet because synthesis-grade work
    target_research, competitor_research, market_research = await asyncio.gather(
        call_model("research", model="claude-sonnet-4-6", topic=company),
        call_model("research", model="claude-sonnet-4-6", topic=f"{company} competitors"),
        call_model("research", model="claude-sonnet-4-6", topic=f"{company} market"),
    )

    # Orchestration: Opus because plan errors propagate
    plan = await call_model("orchestrator", model="claude-opus-4-7",
                             research=[target_research, competitor_research, market_research])

    # Section drafts: Sonnet because focused work, lots of them
    sections = await asyncio.gather(*[
        call_model("section_writer", model="claude-sonnet-4-6", section=s) for s in plan.sections
    ])

    # Synthesis: Opus because cross-section coherence matters
    brief = await call_model("synthesizer", model="claude-opus-4-7", sections=sections)

    # Evaluation: Opus because nuanced criteria
    evaluation = await call_model("evaluator", model="claude-opus-4-7", brief=brief)

    return brief, evaluation
```

Six different model decisions in one workflow. Three Sonnet, two Opus, one Haiku. The Haiku call is cheap; the Sonnet calls are the bulk of the work; the Opus calls are reserved for the moments where error propagation or judgment quality matter.

In an agent loop:

The Module 2 hardened loop made one model selection at construction time. For most agents, that's the right call — the agent's role doesn't change mid-trajectory, so neither should its model. But two patterns warrant attention:

**Per-tool model overrides.** Some tools are genuinely simpler than the agent's general work. A "search the docs" tool that's used a hundred times in a long agent run could use Haiku for those calls specifically, while the agent's main reasoning stays on Sonnet. This is rare in practice; tools usually wrap deterministic operations and don't make their own LLM calls. But if you have an agent calling a "summarize this long page" sub-call, that sub-call doesn't need to be the same model as the agent.

**Cascading at the agent level.** If your agent has tasks that vary widely in complexity and you've already invested in cascade routing (Pattern 3), you can use the cascade outcome to pick the model for the rest of that agent run. Example: route incoming requests through Haiku first; if Haiku punts to escalation, run the rest of the agent on Sonnet for that user's session.

For 95% of cases, "pick a model per role / per workflow step" is the right discipline. The agent loop's model is a property of the agent's role, not a per-turn decision.

::: nodumbq
**Q: Should I keep the router as Haiku even if Haiku gets the routing wrong sometimes?**

If the routing wrongness is symmetric (Haiku occasionally routes simple tasks to Opus and complex tasks to Haiku at similar rates), you're paying small inefficiency on both sides — usually still net cheaper than always using Sonnet for routing. If the routing wrongness is asymmetric (Haiku routes too aggressively to Haiku, missing the cases that needed Sonnet), then yes, upgrade the router. The diagnostic: log routing decisions, manually review 100 of them weekly for a month. If Haiku's accuracy is >90%, keep it. If not, move the router to Sonnet.

**Q: What about "auto" mode where the model picks itself?**

Anthropic's recent extended-thinking-with-adaptive-depth on Sonnet 4.6 and Opus is a related idea, but it's about reasoning depth within one model, not which model to use. There isn't a current "Anthropic auto-routes for you" feature; you build the routing layer yourself. Some third-party providers offer cross-model auto-routing; treat it skeptically until you've measured against your own routing.
:::

---

## Section 6 — Model Migrations and Upgrades

Models change. Anthropic ships new versions roughly every quarter. When Opus 4.7 lands and your system is running on Opus 4.6, the question is: do you migrate, when, and how?

The discipline:

**Don't migrate by reflex.** "New model came out, let's switch" is the most common reason to ship a regression. The new model isn't strictly better — it's differently calibrated. Performance on your specific task may go up or down; the only way to know is to measure.

**Migrate per-call, with eval evidence.** If you have an eval that shows Opus 4.7 is better than Opus 4.6 for *your* synthesizer prompt, migrate the synthesizer. Don't migrate every call to Opus 4.7 just because it's newer. Different roles benefit differently.

**Watch for cost surprises.** Opus 4.7's tokenizer produces up to 35% more tokens than Opus 4.6 for the same text. Same sticker price; different effective bill. Cost-per-task is what matters, not cost-per-token.

**A/B in production with traffic splitting.** For high-stakes decisions, run 5-10% of traffic through the new model for a week. Compare quality (your eval metric), cost (per-task), and latency. Roll forward if the numbers justify it; roll back if they don't.

```python
import random


def select_model_with_ab(role: str, ab_split: dict[str, float]) -> str:
    """
    role -> ab_split is something like:
    {"synthesizer": {"claude-opus-4-6": 0.9, "claude-opus-4-7": 0.1}}
    """
    candidates = ab_split[role]
    r = random.random()
    cumulative = 0.0
    for model, weight in candidates.items():
        cumulative += weight
        if r < cumulative:
            return model
    return list(candidates.keys())[-1]  # fallback


# Usage
AB_CONFIG = {
    "synthesizer": {
        "claude-opus-4-6": 0.9,  # current production
        "claude-opus-4-7": 0.1,  # new model under evaluation
    },
}

model = select_model_with_ab("synthesizer", AB_CONFIG)
# Log which model was used so you can compare downstream metrics
```

**Deprecate old models on a schedule.** Anthropic deprecates old models periodically — Sonnet 4.5's 1M-token beta was retired April 2026 in favor of Sonnet 4.6's GA support. Track Anthropic's deprecation announcements; don't get caught by a model going away in production.

**Pin model versions, not aliases.** `claude-sonnet-4-6` is a specific version; `claude-sonnet` (if it existed) would be a moving target. Pin the exact version in your code; upgrade explicitly when you migrate. This is also what Anthropic recommends.

::: postmortem
**The Migration That Wasn't Tested**

A team running on Sonnet 4.5 saw the Sonnet 4.6 announcement and pushed a config change replacing the model ID. Both models priced the same; both showed similar benchmark scores. The team thought it was a no-op upgrade.

It wasn't. The team's system used a complex tool-calling pattern with custom output formats that Sonnet 4.5 had been carefully prompt-engineered for. Sonnet 4.6's slightly different output behavior caused 3% of calls to produce malformed tool calls — within noise on benchmarks, catastrophic in production for the workflows downstream of the malformed output.

The team rolled back within four hours, then spent two weeks re-tuning the prompts for 4.6 before migrating successfully.

**Lesson:** "no-op upgrade" doesn't exist for model migrations. New models are differently calibrated, even within the same name and price tier. Treat every migration as a behavior change. A/B in production. Roll forward only with evidence.
:::

---

## Section 7 — Knowing Your Routing Is Right

Routing decisions need feedback. Without evidence that your routing matches reality, you're guessing.

Three metrics to track per role/route:

**1. Cost per task.** What does one successful task cost on this routing config? Track this over time; deviations are signals (a regression, a new failure mode, drift in the upstream input distribution).

**2. Quality per route.** A task-appropriate eval metric: classification accuracy, schema-validation rate, human-judged quality, retention of structured output, whatever maps to your task. Quality should not vary by route except for tasks correctly routed to the cheaper tier.

**3. Routing accuracy.** When the router decides Haiku, would Sonnet have done meaningfully better? When the router decides Sonnet, was Opus warranted? You measure this by running periodic shadow evaluations: route as normal, but also run a sample on a stronger model and compare.

```python
async def shadow_eval(
    user_request: str,
    primary_model: str,
    shadow_model: str,
    sample_rate: float = 0.05,
):
    """
    Run the request on primary_model. Some fraction of the time, also run on
    shadow_model for comparison. Log both for offline analysis.
    """
    primary_result = await client.messages.create(
        model=primary_model,
        max_tokens=4096,
        messages=[{"role": "user", "content": user_request}],
    )

    if random.random() < sample_rate:
        shadow_result = await client.messages.create(
            model=shadow_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": user_request}],
        )
        # Log the pair for offline analysis
        log_shadow_comparison(
            request=user_request,
            primary_model=primary_model,
            primary_output=primary_result.content[0].text,
            shadow_model=shadow_model,
            shadow_output=shadow_result.content[0].text,
        )

    return primary_result.content[0].text
```

The shadow evaluation costs the price of the shadow calls (5% of traffic × shadow cost) but gives you ground truth on whether the primary routing is leaving quality on the table. If the shadow model is consistently producing better results, your primary routing is too aggressive on the cheap tier.

The opposite check is just as important. If your primary is Sonnet and a Haiku shadow produces equivalent results 60% of the time, you have a downgrade opportunity worth ~3× cost reduction on that 60%.

Both checks run in shadow — neither affects the user-facing path. This is the discipline that turns routing from "gut feeling" into "engineering decision with evidence."

::: code-exercise
**Exercise 6.1 — Audit your model spend.**

Pick an agent or workflow you've built. Go through every API call and:

1. Note the current model used.
2. Justify it in one sentence: why this tier and not the one below?
3. If you can't justify it, mark it for downgrade.
4. Estimate the per-call cost; multiply by your call volume.

Sum the savings on the calls you marked for downgrade. That's your potential. Then ship the downgrades for the easiest 2-3 cases (lookups, classifications, fetches), measure quality with a small eval, confirm no regression.

If you don't have an eval, write one: 20-50 representative tasks for the role you're downgrading, manual quality review of outputs at both tiers. The eval is the contract that lets you downgrade without flinching.
:::

---

## Section 8 — Cross-Provider Routing

A short, honest section. The book is built around Claude — but plenty of production agent systems include non-Claude models for specific tasks. When and why?

**Strong reasons to mix providers:**

- **Specialized capabilities.** GPT-5.4's built-in computer use is more mature than Claude's; some teams use GPT specifically for desktop automation tasks. Gemini 3.1's video understanding leads on certain benchmarks.
- **Cost arbitrage at extreme scale.** Open-source models (DeepSeek, Llama, Qwen) hosted on commodity inference providers can be substantially cheaper than commercial frontier models for certain tasks. At a billion calls a month, the math sometimes shifts.
- **Provider redundancy.** For mission-critical workloads, having a fallback to a different provider is operational hygiene. If Anthropic has an outage, your agent shouldn't be down.
- **Domain-specific fine-tuning.** Some tasks benefit from a fine-tuned smaller model that an open provider lets you train.

**Weak reasons to mix providers:**

- **"Everyone says X is better."** Per-task evals beat received wisdom. The frontier model lineups are very close on most tasks; benchmark differences within a tier are usually within the noise of your specific use case.
- **Fear of vendor lock-in for its own sake.** MCP and standardized protocols (Module 5) reduce lock-in significantly. The actual lock-in cost — switching from one Claude tier to another vs switching to GPT — is much lower than people assume.
- **Resume-driven development.** Adding a third provider because the team wants to learn it is real but should be acknowledged as a cost, not framed as a benefit.

The right design: **pick one provider as your primary**, route within their model family for cost-quality optimization, add cross-provider routing only when there's a specific capability or cost case that justifies it.

If you do go cross-provider: the OpenAI SDK's API shape is similar enough to Anthropic's that you can write a provider-agnostic wrapper. Many teams use the OpenAI SDK with `base_url` pointing at Anthropic, OpenAI, OpenRouter, or self-hosted endpoints. Frameworks like LangChain and LiteLLM provide cross-provider routing layers; for production, evaluate whether you need that abstraction or whether direct SDK calls are simpler.

This module is about Claude routing because that's where the highest-leverage optimization lives for most readers. Cross-provider routing matters; it's a smaller module's worth of content; we won't dwell.

---

## Section 9 — Putting It Together

A short framework, applying everything in this module:

**1. Audit current spend.** Pull cost-by-model from your observability. If you can't see it, fix that first — Module 17 covers observability properly. You can't optimize what you can't see.

**2. Map calls to roles.** Every model call has a role: orchestrator, worker, classifier, fetcher, synthesizer, evaluator. Name the roles in your system. There are usually 4-8 distinct roles.

**3. Pick a model per role.** Default Sonnet. Specialize Haiku for shallow work. Reserve Opus for nuanced judgment. Document the choice with a one-sentence justification per role.

**4. Implement static role-based routing.** Pattern 1 from Section 4. Single source of truth for model selection. Reviewable in code review.

**5. Measure quality per role.** A small eval (20-50 tasks) per role. Confirm the chosen model meets quality bar; confirm cheaper models don't. Document the eval as the routing decision's evidence.

**6. Add cascading or per-call routing only where needed.** If a single role's traffic is heterogeneous (some shallow, some deep), Patterns 2-3 from Section 4 earn their complexity. For uniform-traffic roles, static is enough.

**7. Run shadow evaluation continuously.** 5% sample rate on a shadow model gives you the running calibration you need. When patterns shift (input distribution drifts, the model gets a quiet upgrade, etc.), shadow eval catches it before it hits production quality.

**8. Treat upgrades as engineering events.** A/B with traffic splits, eval evidence, rollback plans. Don't migrate by reflex.

The discipline this enforces: the model parameter is no longer a default — it's a decision, owned by someone, justified in writing, validated by evidence.

---

## Recap: Module 6 in eight bullets

::: bullet-points
- The Claude lineup is three tiers spaced by 3-5× cost: Haiku 4.5, Sonnet 4.6, Opus 4.7. Each tier earns its price for a different shape of work; defaulting to one tier for everything overpays.
- Default to Sonnet for most production work; specialize Haiku for shallow/volume tasks; reserve Opus for nuanced judgment, orchestrator/synthesizer roles, and the small fraction of work where being measurably better matters.
- Production cascade distributions converge near 60-70% Haiku, 25-30% Sonnet, 3-5% Opus. Hitting that distribution typically cuts cost 50-80% vs all-Sonnet, with no measurable quality regression.
- Four routing architectures: static role-based (most teams' v1), per-call dynamic (heterogeneous requests), cascade with quality check (cheap-first with fallback), adaptive learned routing (rare, scale-dependent).
- Assign models at the smallest unit — per workflow step, per agent role, per tool — not per request. Sam's deal-research workflow has six different model decisions in one trajectory.
- Migrations are not no-ops. New model versions are differently calibrated; A/B in production with traffic splits, validate against your evals, roll forward only with evidence.
- Routing decisions need feedback. Track cost per task, quality per route, and routing accuracy via shadow evaluation (5% sample on a comparison model). Without measurement, you're guessing.
- Cross-provider routing has a place but is rarely the highest-leverage optimization. Pick a primary provider, optimize within their family first, add cross-provider only for specific capability or cost cases.
:::

---

::: sam-arc
**Sam, after the platform team review.**

The platform team lead's reply turned into a meeting. Sam walked the rest of the team through the audit — three tools moved from Opus to Haiku, $1,800/month saved, zero quality regression. The team lead asked Sam to do the same exercise across the company's other ten production agents.

Two weeks of audits later, the picture was clear. Of the ten agents reviewed: six had at least one role miscalibrated (usually a fetcher or classifier on Opus when Haiku was right), three were fine, and one had an *under*-routed evaluator that was on Sonnet when the nuance genuinely warranted Opus. Total monthly savings from the right-sizing: $14,000. Total quality regressions: zero. Total quality *improvements* from the one Sonnet-to-Opus upgrade: measurable on the eval.

Sam's company didn't change. Anthropic didn't change. The agents didn't change architecturally. The model parameter changed in eleven places. That was the whole intervention.

Sam wrote up a one-page playbook for the platform team: audit-template, eval-template, A/B-config-template, rollback-checklist. New agents going through review would now justify their model choices in a checklist; rejected reviews would call out unjustified Opus use.

Sam's arc this module: **measure before optimizing.** The cost optimization in agent systems isn't in the loop or the prompt or the framework. It's in the model parameter, examined honestly. Module 5's discipline (the integration is cheap, the trust layer is expensive) generalizes here: the *architecture* is cheap, the *model choices* compound into the bill. The architecture is what gets attention; the model choices is where the savings live.
:::

---

## What's next

Module 7 starts the context-engineering arc. We've been writing prompts and feeding them to the model without much thought to what's *in* the context window — system prompts, conversation history, tool results, retrieved documents, examples. As agents run longer and tool results pile up, the context becomes a resource that needs management. Anthropic's own framing — "context engineering as the natural progression of prompt engineering" — is the lens.

After Module 7 (context engineering principles), Module 8 covers compression and summarization (what to do when context overflows), and Module 9 covers memory across runs (state that persists between trajectories). By the end of that arc, you'll have a complete picture of what flows through your agents — what they read, what they remember, what they forget.

Sam's deal-research system, post-Module 6, is well-shaped, well-tooled, well-secured, and economically right-sized. By Module 9 it'll also be smart about what it loads into context. The book builds.

For now: take the audit exercise from Section 7. Find one Opus call that should be Sonnet. Find one Sonnet call that should be Haiku. Ship the changes. Watch the bill.
