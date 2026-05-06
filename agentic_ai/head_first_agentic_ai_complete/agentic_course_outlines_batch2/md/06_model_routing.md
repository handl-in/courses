# Module 6 Outline — Choosing Models: Routing as Architecture

::: chapter-opener
<div class="module-num">MODULE 06 — OUTLINE</div>
<div class="module-title">Choosing Models: Routing as Architecture</div>
<div class="subtitle">Stop using one model for everything.<br>Routing is now first-class agent design — not optimization.</div>
<div class="pages">Target length: ~36 pages</div>
:::

## What this module is

Most agent code uses one model — usually the strongest available — for every call. This is the simplest thing to write and almost always wrong. By 2026, **model routing is architectural**: which model handles which call is as fundamental a design decision as which tools live in your registry.

This module covers three layers of routing: **static** (hard-code per role/agent), **rule-based** (heuristics on prompt features), **classifier-based** (a small model decides). Plus the cascading pattern, the cost/quality tradeoff curves, when each approach earns its keep, and a hardened router we plug into the Module 2 agent.

By the end, the reader can take their existing single-model agent and route to a mix of Haiku/Sonnet/Opus (or equivalents) with a measured 40-70% cost reduction at equal or better quality — a number now well-documented in production case studies.

::: hook
"You're paying for a brain surgeon to read your text messages. The surgeon is great at brain surgery and overqualified for everything else. The router is the receptionist who knows when to call the surgeon and when to handle it themselves."
:::

---

## Section 1 — Why Model Routing Now Matters (≈3 pages)

The landscape has changed. In 2024, model choice was mostly: pick the strongest model you can afford. By 2026, the gap between tiers has both **shrunk** (smaller models are very capable) and **grown** (Opus/GPT-5 frontier models are genuinely better at hard reasoning).

Two facts that drive routing economics:

**Cost-per-token differs by 10-30×** between tiers (Haiku vs Opus, Sonnet vs Opus, etc.). A million tokens through Opus at frontier pricing is meaningfully different from a million through Haiku.

**Capability gap is task-dependent**, not uniform. Recent research ("Scaling Small Agents Through Strategy Auctions," 2026): small agents hit ~87% of large agents' performance on simple tasks but only ~21% on complex tasks. Model size is a per-task decision, not a global choice.

The architectural implication: an agent that uses one model for everything is leaving 40-70% cost on the table (a number consistently reported across LiteLLM, Portkey, OpenRouter, and Bifrost case studies in 2025-2026). And often quality on the table too — because the strong model is sometimes overkill (slower, more verbose) when a faster model would actually be better UX.

::: pullquote
"Use the strongest model" is a confession that you haven't measured your workload.
:::

::: nodumbq
**Q: Doesn't routing add latency? You're calling a classifier before every model call.**

Yes — typically 50-200ms (per Augment Code's 2026 routing guide). For short, latency-sensitive calls, that overhead matters. For agent calls (typically 1-30 seconds each), it's noise. We discuss when routing latency is unacceptable in Section 7.

**Q: My team uses one provider. Is routing only useful for multi-provider setups?**

No. Within Anthropic alone you have Haiku, Sonnet, Opus — three tiers with ~20× cost spread. Same within OpenAI. Routing within a single provider is the most common case.
:::

---

## Section 2 — Three Routing Strategies (≈4 pages)

A taxonomy. Each strategy fits different workloads.

**Static routing.** Per agent role, hardcode the model. The supervisor uses Opus, the workers use Sonnet, the validators use Haiku.

```python
ROLE_MODELS = {
    "supervisor": "claude-opus-4-7",
    "worker": "claude-sonnet-4-6",
    "validator": "claude-haiku-4-5-20251001",
    "summarizer": "claude-haiku-4-5-20251001",
}

def call(role: str, **kwargs):
    return client.messages.create(model=ROLE_MODELS[role], **kwargs)
```

When it works: predictable workloads where roles map cleanly to complexity.
When it breaks: complexity varies within a role (some validation tasks are trivial, some require deep reasoning).

**Rule-based routing.** Heuristics on the input itself.

```python
def pick_model(prompt: str, role: str) -> str:
    if role == "supervisor":
        return "claude-opus-4-7"
    if "analyze" in prompt.lower() or "compare" in prompt.lower():
        return "claude-sonnet-4-6"
    if len(prompt) > 5000:
        return "claude-sonnet-4-6"
    return "claude-haiku-4-5-20251001"
```

When it works: clear correlation between observable signals and model needs.
When it breaks: silently — the rules are wrong on edge cases and you don't notice until quality degrades.

**Classifier-based routing.** A small model (or a fine-tuned BERT-class classifier) predicts complexity and routes accordingly. This is what RouteLLM (LMSYS), vLLM Semantic Router (Red Hat), and most production gateways use.

```python
async def pick_model_classifier(prompt: str) -> str:
    classification = await complexity_classifier.predict(prompt)
    # classification.complexity ∈ ["simple", "moderate", "complex"]
    # classification.confidence ∈ [0, 1]
    if classification.confidence < 0.7:
        return "claude-sonnet-4-6"  # default to middle tier when uncertain
    return {
        "simple":   "claude-haiku-4-5-20251001",
        "moderate": "claude-sonnet-4-6",
        "complex":  "claude-opus-4-7",
    }[classification.complexity]
```

When it works: workloads with enough variance to make rules unreliable; high enough volume to justify classifier training.
When it breaks: classifier drift over time (workload changes, model evals shift), uncalibrated confidence scores.

::: brain
You have three routing strategies and 100,000 calls/day. Which would you start with? Why? When would you upgrade to the next?

(Hint: start static. Measure cost and quality per role. Add rules only when static breaks. Add a classifier only when rules can't capture the variance and the math justifies the engineering.)
:::

---

## Section 3 — The Cascade Pattern (≈4 pages)

The most production-ready routing pattern. Try the cheapest model first; escalate only if needed.

```python
async def cascade_call(prompt: str, tools: list, judge_threshold: float = 0.7) -> str:
    # Tier 1: try Haiku
    result_haiku = await call_model("claude-haiku-4-5-20251001", prompt, tools)
    
    # Quality gate: did it actually work?
    confidence = await judge_quality(prompt, result_haiku)
    if confidence >= judge_threshold:
        return result_haiku
    
    # Tier 2: escalate to Sonnet
    result_sonnet = await call_model("claude-sonnet-4-6", prompt, tools)
    confidence = await judge_quality(prompt, result_sonnet)
    if confidence >= judge_threshold:
        return result_sonnet
    
    # Tier 3: Opus
    return await call_model("claude-opus-4-7", prompt, tools)
```

The economics: if 70% of calls succeed at Tier 1, you pay (0.7 × Haiku) + (0.3 × Haiku + Sonnet) for an effective cost much closer to Haiku than Sonnet, and you pay full Opus only for the hardest 5-10%.

**Quality gates** — how do you know Tier 1 succeeded?
- Heuristic: did the response include all expected elements? (Length, format, mentioned all required entities.)
- Self-evaluation: ask the model "did you answer this confidently?"
- Judge model: a separate (cheap) call evaluates the response.
- Tool failure: if the model gave up and returned "I don't know," escalate.
- Schema validation: if structured output failed validation, escalate.

**Pitfalls of cascade:**
- Quality gate is itself an LLM call → adds cost and latency
- If the gate is wrong, you either escalate too much (cost) or not enough (quality)
- Calibrating threshold is empirical — measure on your workload

::: postmortem
**The Cascade That Cascaded Wrong**

A team built a cascade with Haiku → Sonnet → Opus. They used self-evaluation as the gate ("rate your confidence 1-10"). After deployment, they discovered Haiku was rating itself 9/10 on every response — including responses that were obviously wrong. The cascade never escalated. They were paying Haiku prices and getting Haiku quality, but had told leadership they had "Opus quality with cost optimization."

**Lesson:** never use the same model to evaluate its own output. Use a different (often stronger) model as the judge, OR use heuristics, OR use a fine-tuned classifier — but not self-evaluation.
:::

---

## Section 4 — Routing for Agents Specifically (≈4 pages)

Routing in chat is one thing. Routing in agents is harder because:

**1. Each turn has different needs.** The first turn (planning) is hard reasoning. Middle turns (executing) might be straightforward. The final turn (synthesis) is hard again. Static per-agent model choice can't capture this.

**2. Tool selection complexity varies.** "Should I use search or fetch?" is easy. "Given these 50 tools, which combination accomplishes the goal in the fewest steps?" is hard. Different turns need different reasoning depth.

**3. Context grows.** Token cost per turn rises monotonically. By turn 20, a single Opus call might cost $0.50. If you can drop to Sonnet for routine middle turns, you win big.

A pattern: **per-turn routing inside the agent loop.**

```python
class Agent:
    def run(self, query: str) -> AgentResult:
        messages = [{"role": "user", "content": query}]
        for turn in range(self.budget.max_turns):
            # Decide model for this turn
            model = self.router.pick(
                turn_index=turn,
                messages=messages,
                last_tool_result=self._last_tool_result(messages),
            )
            
            resp = self.client.messages.create(
                model=model,
                tools=self.tools.schemas(),
                messages=messages,
            )
            ...
```

The router can use signals like:
- Turn index (first/last turns get stronger models)
- Last tool result (errors → strong model to recover; successful retrieval → cheap model to summarize)
- Message count (later turns with more context might need stronger model OR might be routine)
- Tools called so far (specific tool patterns indicate planning vs execution)

::: code-exercise
**Exercise 6.1 — Per-turn router.**

Take the hardened agent from Module 2. Add a `Router` class. Implement static-by-default with rule overrides for first turn (Opus), error recovery turns (Opus), and routine middle turns (Sonnet or Haiku). Measure cost and task completion on a provided eval set.
:::

**A different pattern: sub-agent routing.** When the orchestrator-workers pattern (Module 3) is involved, route at the worker-spawn level. A worker spawned to do "fact extraction" gets Haiku; a worker spawned to do "comparative analysis" gets Sonnet; the orchestrator that decided what workers to spawn used Opus.

This is what Anthropic's Sub-Agents API supports natively (`model: sonnet | opus | haiku | <full-id> | inherit`).

---

## Section 5 — Building a Production Router (≈5 pages)

We design a `Router` class for the agent. Requirements:

- Pluggable strategy (static, rules, classifier, cascade)
- Per-call cost tracking
- Per-call latency tracking
- Quality monitoring (samples N% of cheap-route calls for judge evaluation)
- Drift detection (cheap-route quality scores over time)
- Fallback on errors (if Opus is rate-limited, fall back to Sonnet rather than fail)

```python
class RoutingStrategy(Protocol):
    def pick(self, context: RoutingContext) -> str: ...

class StaticStrategy:
    def __init__(self, role_model_map: dict[str, str]): ...
    def pick(self, context): return self.role_model_map[context.role]

class RuleStrategy:
    def __init__(self, rules: list[Rule], default: str): ...
    def pick(self, context):
        for rule in self.rules:
            if rule.matches(context):
                return rule.model
        return self.default

class ClassifierStrategy:
    def __init__(self, classifier, model_map, confidence_threshold=0.7): ...
    async def pick(self, context):
        c = await self.classifier.predict(context)
        if c.confidence < self.confidence_threshold:
            return self.model_map["default"]
        return self.model_map[c.complexity]

class CascadeStrategy:
    def __init__(self, tiers: list[str], judge): ...
    async def execute(self, context):
        for model in self.tiers:
            result = await call_model(model, context)
            if await self.judge.acceptable(context, result):
                return model, result
        return self.tiers[-1], result  # final tier wins by default

class Router:
    def __init__(self, strategy: RoutingStrategy, monitor: RouterMonitor): ...
    async def call(self, context: RoutingContext, **kwargs):
        model = await self._pick(context)
        with self.monitor.track(model, context):
            try:
                return await call_model(model, **kwargs)
            except RateLimitError:
                fallback = self._fallback_for(model)
                return await call_model(fallback, **kwargs)
```

We also build the `RouterMonitor` — Prometheus-style metrics for: model selection distribution, cost-per-route, escalation rate, judge-sampled quality scores per route, drift alarms.

::: gotcha
**Don't optimize too early.** Build static first. Measure. Only add complexity when measurement shows the simpler strategy is leaving real money on the table or hitting real quality issues. A team that ships a classifier-based router on day one usually spends three months tuning it before realizing their workload was simple enough that static would have done 90% as well.
:::

---

## Section 6 — Cost & Quality Measurement (≈4 pages)

You can't route what you don't measure. We build the measurement layer.

**Per-route cost tracking.** Every call, log: model, input tokens, output tokens, dollar cost (computed), wall-clock time. Aggregate by route, by hour, by user, by task type.

**Quality measurement.** Three approaches:

1. **Outcome metrics** (when available): did the agent complete the task? Did the user accept the output? Did downstream code run without errors?
2. **Sampling judge.** Take 1-5% of cheap-route calls, run them through a judge model (often Opus), score quality. Track over time.
3. **Side-by-side eval.** Periodically (weekly?) re-run a representative eval set with the cheap and expensive models. Compare. If the gap widens, the cheap route is degrading.

**Drift detection.** A classifier trained on Q3 data may not match Q1 workloads. A rule that worked in pilot might be wrong at scale. Alert on:
- Sudden change in route distribution (was 60% Haiku, now 90% — what changed?)
- Sudden change in escalation rate
- Drop in judge-sampled quality
- Increase in user-reported issues correlated with route

::: brain
You're routing 1M calls/day. Sampling 1% for judge evaluation = 10K judge calls/day at, say, $0.05 each = $500/day in eval costs.

Is this worth it? When would it not be?

(Worth it almost always. The 1% sample tells you whether the other 99% is degrading. The alternative is finding out from angry users.)
:::

---

## Section 7 — When NOT to Route (≈3 pages)

The contrarian section. Routing is great until it isn't.

**Don't route when:**
- Latency budget is tight (<1s end-to-end). Even 50ms classifier overhead is too much.
- Volume is low (<10K calls/day). The savings don't justify the complexity.
- Workload is uniform. If 95% of your calls are the same task type, just pick the right model and skip the router.
- You can't measure quality. If you have no signal whether the cheap route worked, you're flying blind. Build measurement first, then route.
- The cost difference doesn't matter. If your spend is $50/month, optimizing routing is engineering theater.

**Do route when:**
- Your bill is meaningful and your workload has variance
- You have clear quality signals (outcome metrics, judge feasibility)
- You're at scale where cost optimization compounds

Routing is a real tool for real cost problems. It's also where teams over-engineer when they should just be picking the right model once and shipping.

---

## Section 8 — Beyond Cost: Routing for Other Reasons (≈3 pages)

Cost is the obvious driver but not the only one.

**Latency routing.** When the user is waiting, prefer faster models. Haiku-class models often have 3-5× lower TTFT and are noticeably snappier. For interactive flows, route by latency budget regardless of cost.

**Capability routing.** Some tasks need multimodal (vision); route to a multimodal model. Some need very long context; route to a long-context model. Some need code generation; route to a code-specialized model.

**Provider redundancy routing.** When Provider A is down or rate-limiting, fall back to Provider B with comparable quality. This is a *resilience* concern, not a cost one. (LiteLLM, Portkey, OpenRouter all support this natively.)

**Compliance/locality routing.** EU users → EU-hosted models for GDPR. Sensitive content → on-prem model. PII detection → route around providers that log.

::: pullquote
The router is your traffic policy layer. Cost is one policy. Latency is another. Compliance is another. Don't conflate them.
:::

---

## Section 9 — What's Next (≈1 page)

Module 7 begins context engineering — once your agent is routing models well, the next leverage point is what each model sees in its context window.

Module 8 covers compression algorithms (ACON, anchored summarization, budget-aware compression) — directly relevant to cost, since smaller context = lower cost regardless of which model you route to.

Modules 9-12 build out memory, hallucination handling, reflection, and guardrails — each interacting with routing decisions.

By Module 14 (multi-agent architectures) routing is everywhere: per-agent, per-role, per-turn, per-task.

---

## Recurring elements used in this module

- 1 chapter opener
- 1 hook
- 2 No Dumb Questions
- 2 Brain Power prompts
- 1 Production Postmortem (cascade self-eval failure)
- 2 Watch It! / Gotcha blocks
- 1 Code Exercise (per-turn router)
- 2 Pullquotes
- ~6 substantial code blocks
- 1 Bullet Points recap

::: sam-arc
Sam looks at the OpenRouter bill again — Module 1's bill, but worse now that the agent has more tools. PM asks: "Can you cut costs without sacrificing quality?" Sam adds static routing in Section 2 (saves 40%). Tries a classifier in Section 3 and accidentally builds the self-evaluation cascade from Section 3's postmortem (loses quality, doesn't notice for three weeks). Eventually settles on cascade with judge sampling in Section 6. Sam's arc this module: **measure before optimizing, and never trust a model to grade itself**.
:::

::: page-budget
S1 (Why routing matters): 3p
S2 (Three strategies): 4p
S3 (Cascade): 4p
S4 (Agent-specific routing): 4p
S5 (Production router): 5p
S6 (Measurement): 4p
S7 (When not to): 3p
S8 (Beyond cost): 3p
S9 (Next): 1p
Recurring elements: 5p
TOTAL: ~36 pages
:::

::: sources
**Must verify when drafting (heavy verification needed — pricing and model specifics):**

- Current Anthropic model pricing (input/output per token for Haiku 4.5, Sonnet 4.6, Opus 4.7)
- Current OpenAI/Google/etc. pricing for cross-provider examples
- "Scaling Small Agents Through Strategy Auctions" (2026) — exact figures for the 87% / 21% claim
- RouteLLM (LMSYS) current API and capabilities
- vLLM Semantic Router (Iris v0.1, Jan 2026) — features, benchmarks
- LiteLLM, Portkey, OpenRouter, Bifrost — current feature comparison
- The "40-70% cost cut" headline — verify across multiple production case studies (Akshay Ghalme's 2026 piece is one source; find more)
- Augment Code's 2026 routing guide — for the 50-200ms classifier latency figure
- Anthropic Sub-Agents API model field options (sonnet, opus, haiku, full ID, inherit)

**Stable knowledge:**
- The cost-per-token tier structure (will exist in some form)
- Static vs rule-based vs classifier-based taxonomy
- Cascade pattern with quality gates
- Self-evaluation failure mode
- Drift detection principles

**Cross-reference checks:**
- Module 3's "routing" workflow pattern is single-prompt routing; this module's routing is model selection. Make the distinction explicit
- Module 14 will route per-agent in multi-agent systems; foreshadow but don't preempt
:::

::: bullet-points
### Module 6 in eight bullets

(filled at draft time)

- Model routing is now architecture, not optimization — driven by 10-30× cost spread between tiers
- Three strategies: static (per role), rule-based (heuristics), classifier-based (small model decides)
- The cascade pattern (Haiku → Sonnet → Opus with quality gates) is the most production-ready
- Quality gates can be heuristic, judge-model, or schema validation — never self-evaluation
- Per-turn routing inside the agent loop captures variance static routing misses
- Build static first. Measure. Add complexity only when measurement justifies it
- Don't route when latency-tight, low-volume, uniform workload, or no quality signal
- Routing serves cost, latency, capability, redundancy, and compliance — not all the same policy
:::
