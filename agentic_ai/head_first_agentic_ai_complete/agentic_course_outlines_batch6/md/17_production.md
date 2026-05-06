# Module 17 Outline — Production: Evals, Observability, Cost, Failure

::: chapter-opener
<div class="module-num">MODULE 17 — OUTLINE</div>
<div class="module-title">Production: Evals, Observability, Cost, Failure</div>
<div class="subtitle">Your agent is in production. Now what?<br>The honest answer: most of your work just started.</div>
<div class="pages">Target length: ~38 pages</div>
:::

## What this module is

Production is where the abstractions from Modules 1-16 hit reality. This module covers the four operational concerns that turn a working prototype into a system you trust: evaluation (does it actually work? how do you know?), observability (what's happening right now? what happened in that failed trace?), cost management (the bill is sustainable?), and failure handling (what happens when things break in ways you didn't anticipate?).

We cover: trajectory eval with LLM-as-judge, rubric-based eval, eval datasets mined from production traces, the OpenTelemetry GenAI semantic conventions that became the 2026 standard, the agent observability platforms and their honest trade-offs (LangSmith, Langfuse, Laminar, Arize Phoenix), distributed tracing across multi-agent systems, cost ceilings and runaway prevention, loop detection, drift monitoring, the Q1 2026 finding that **40% of agentic AI projects will be canceled by end of 2027 due to reliability concerns** (Gartner) — and what to do so yours isn't one of them. Plus a brief treatment of computer-use / GUI agents because the operational concerns there are uniquely brutal (OSWorld, AndroidWorld benchmarks).

By the end the reader can: instrument an agent system to OpenTelemetry standards, build a useful eval harness from production traces, set cost ceilings that prevent runaway, and read a failed trace well enough to actually debug it.

::: hook
"Your agent ships. The CEO loves the demo. Three weeks later, sales loops a customer ticket: 'agent gave me bad info on April 3rd around 2pm.' You go to find that conversation. Your logs are unstructured. Your trace IDs don't link to LLM calls. Cost is shown daily, not per-request. You spend two hours digging and find nothing. The customer leaves. The CEO asks why you can't reproduce it. This is the difference between a working agent and a production agent."
:::

---

## Section 1 — The Production Reality Check (≈3 pages)

Gartner's 2025 finding: over 40% of agentic AI projects will be canceled by the end of 2027 due to reliability concerns and unclear objectives. Take that as a baseline expectation. Most production agent deployments are going to struggle. The question is which side of the line yours ends up on.

Three operational properties separate the survivors from the cancellations:

**1. Measurable.** You can answer "is the agent working?" with numbers, not vibes. Specific quality metrics. Specific cost metrics. Specific latency metrics. Updated continuously, not after-the-fact.

**2. Debuggable.** When something breaks, you can find it. Trace IDs link agent decisions to LLM calls to tool invocations to outcomes. A failure surfaces with enough context to reproduce locally.

**3. Bounded.** Cost has ceilings. Loops have caps. Runaway scenarios have circuit breakers. The system fails safely, not catastrophically.

This module is how you build all three. Some of it is platform selection. Most of it is architectural commitments you make in code.

::: pullquote
Demos make agents look impressive. Production makes agents look fragile. The work between those two states isn't agent work — it's operations work that happens to apply to agents.
:::

::: nodumbq
**Q: My agent works in testing. Why do I need all this?**

Testing covers what you anticipated. Production reveals what you didn't. The gap between "works on test cases" and "works on the long tail of real user queries" is where most agent projects die. Observability and eval infrastructure exist to surface the gap quickly enough to close it.

**Q: How much of this do I need on day one?**

Trace IDs and cost tracking on day one. Eval harness and structured logging in week one. Drift monitoring and proper alerting by month one. The infrastructure compounds — early investment in tracing pays dividends every time you debug a production issue.
:::

---

## Section 2 — Evaluation: How You Know It Works (≈6 pages)

Eval is the foundation. Without it, every other change is a guess.

**Three layers of eval:**

**Layer 1: Component evals.** Individual prompts, individual tools, individual sub-agents. "Does the topic classifier correctly classify these 100 examples?" Easiest; least valuable for whole-system quality.

**Layer 2: Trajectory evals.** Whole agent runs on representative tasks. "Did the agent succeed at this multi-step booking flow?" Captures emergent issues component evals miss.

**Layer 3: System-level evals.** End-to-end on production-representative workloads. "Compared to last week, has overall quality changed?" The metric you actually optimize.

Most teams overinvest in Layer 1 and underinvest in Layer 3. Layer 1 evals are easy to write and easy to run; they catch the bugs you'd catch in code review anyway. Layer 3 is the hard work — and it's where production reality lives.

**LLM-as-judge for trajectory eval.**

```python
TRAJECTORY_JUDGE_PROMPT = """You are evaluating an AI agent's trajectory on a task.

TASK GIVEN TO AGENT:
{task}

AGENT'S TRAJECTORY (tools called, intermediate reasoning, final answer):
{trajectory}

EXPECTED OUTCOME (or rubric):
{expected_or_rubric}

Score the trajectory on these dimensions (1-5 each):
- task_completion: did the agent achieve the goal?
- efficiency: tools and steps were appropriate, not wasteful
- correctness: factual claims were accurate, citations valid
- safety: no policy violations, no harmful actions

For each dimension below 5, explain why.

Output JSON: {{"task_completion": 1-5, "efficiency": 1-5, "correctness": 1-5, "safety": 1-5,
              "explanations": {{"<dim>": "..."}}, "overall_pass": true/false}}"""
```

Caveats from the literature (IUI 2025 "No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding"):
- Judge LLMs have biases — calibrate against human-labeled gold sets
- Use a different model family from the agent under eval
- For high-stakes domains (medical, legal, financial), LLM-as-judge has documented limitations — use human evaluators for the critical slice

**Rubric-based eval.** When the success criteria are well-defined, write the rubric explicitly. Score against the rubric, not against a vague "is this good?"

```python
SUPPORT_RUBRIC = """A good support response has:
1. Acknowledges the user's specific issue (not a generic "sorry to hear that")
2. Provides a concrete next action or answer
3. Cites the relevant policy/doc when applicable
4. Doesn't fabricate features or policies
5. Doesn't promise things outside the agent's scope (refunds beyond limit, etc.)
6. Length appropriate to question complexity

Score each criterion 0/1 (binary). Total is sum / 6."""
```

Binary rubric scores aggregate cleaner than 1-5 scales — you can talk about pass rates without arguing about what "3 vs 4" means.

**Eval datasets from production traces.** The single highest-leverage practice. Capture production traces, sample, label, accumulate into a growing eval set.

```python
async def sample_for_eval(trace: Trace, sample_rate: float = 0.01) -> bool:
    if random.random() < sample_rate:
        await eval_queue.add(trace)
        return True
    # Always sample failures and edge cases
    if trace.had_error or trace.user_negative_feedback:
        await eval_queue.add(trace)
        return True
    return False
```

Production traces beat synthetic eval cases for one reason: they reflect actual user distributions. Your eval improves as your traffic grows. Most production-monitoring platforms (LangSmith, Langfuse) have built-in workflows for promoting traces to eval datasets.

::: gotcha
The single biggest eval anti-pattern: building an eval set early, then never updating it. Reality drifts; your eval doesn't. After 6 months, you're optimizing for problems users had in month 1 and missing the problems they have now. Treat eval datasets as living artifacts; rotate examples, add new ones from production, retire stale ones.
:::

::: brain
You add a new feature to your agent. Your eval shows quality went up 3 points. Should you ship it?

(Depends on what's in the eval set. If your eval is mostly the cases the new feature was designed for, the +3 is biased — you're optimizing the metric you measured, not quality. The right test is held-out evals (cases the feature designer didn't see) and A/B in production. +3 on the dev eval is a green light to A/B, not a green light to ship.)
:::

---

## Section 3 — Observability: OpenTelemetry and the GenAI Conventions (≈5 pages)

The 2026 standard. As of 2026 the industry is converging on OpenTelemetry (OTEL) for AI agent telemetry — collect once, route to any backend, no vendor lock-in.

**The OpenTelemetry GenAI semantic conventions.** Standard attribute names for LLM application spans:

```python
from opentelemetry import trace
from opentelemetry.semconv.ai import SpanAttributes  # GenAI semconv

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("agent.llm_call") as span:
    span.set_attribute(SpanAttributes.LLM_SYSTEM, "anthropic")
    span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, "claude-sonnet-4-6")
    span.set_attribute(SpanAttributes.LLM_REQUEST_MAX_TOKENS, 4096)
    span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
    span.set_attribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens)
    span.set_attribute(SpanAttributes.LLM_RESPONSE_FINISH_REASON, finish_reason)
    # ... call API ...
```

These are *standard*. Switching backends (Langfuse → Arize, or Datadog → Langfuse) doesn't require changing your instrumentation.

**What to instrument:**

For each agent invocation:
- A root span for the whole agent run, with attributes: agent name, query, user ID, session ID
- A child span for each LLM call, with the GenAI attributes above
- A child span for each tool call, with: tool name, input, output, duration, success/failure
- A child span for each sub-agent invocation
- A child span for each guardrail check
- A child span for each saga step (if applicable, from Module 16)

```
agent.run (root)
├── agent.plan (LLM call)
├── agent.tool_call.web_search
├── agent.tool_call.read_file
├── agent.subagent.researcher_a (recursive)
│   └── ... 
├── agent.guardrail.input_check
├── agent.synthesize (LLM call)
└── agent.guardrail.output_check
```

The trace tree should mirror the agent's logical structure. Done well, you can look at a trace and immediately see where time and tokens went.

**Trace context propagation across agent boundaries.** A subtle but critical detail. When agent A spawns agent B (via tool call, sub-agent, or message), the trace context must propagate or B's spans become orphans.

```python
async def call_subagent(self, subagent: Agent, query: str):
    # Capture current trace context
    ctx = trace.get_current_span().get_span_context()
    # Pass it explicitly or via baggage
    return await subagent.run(query, parent_context=ctx)
```

For multi-agent systems with concurrent sub-agents, this is non-negotiable. Every observability platform requires it; the agent code has to do it.

**Common GenAI semconv attributes** (verify exact names from OpenTelemetry docs):
- `gen_ai.system` — the model provider
- `gen_ai.request.model` — model identifier
- `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` — token counts
- `gen_ai.response.finish_reason` — why generation stopped
- `gen_ai.request.temperature`, `gen_ai.request.top_p` — sampling params

::: postmortem
**The Trace That Didn't Connect**

A team had OpenTelemetry instrumentation. Spans were being collected. The Langfuse dashboard showed traces. Things looked fine.

Then a production incident: an agent was claiming to call a tool, but the tool call wasn't appearing in traces. Investigation: the tool call lived in a different async context, and trace context wasn't propagated. The tool call was happening — there was just no trace evidence. Two hours of debugging on a "missing tool call" that was actually a missing trace span.

Fix: explicit context propagation in the agent's tool-call dispatcher. Every async tool call now wraps in a span tied to the parent agent span. Identical issue surfaces immediately in traces.

**Lesson:** missing trace context creates phantom problems. Test your tracing on async paths and sub-agent boundaries. The places where context is most likely to be lost are the places where bugs are hardest to debug without traces.
:::

---

## Section 4 — Choosing an Observability Platform (≈5 pages)

The honest landscape. We covered hints in earlier modules; this is the consolidated treatment.

**LangSmith (LangChain).** Closed source. Tightest integration with LangChain/LangGraph. LangGraph Studio is genuinely the best agent IDE in the field — visualize the graph, set breakpoints, modify state mid-run, resume from a checkpoint. Near-zero overhead in benchmarks. OpenTelemetry support added in March 2026. Pricing: developer free with 5k base traces/month; Plus $39/seat/month plus $0.50 per 1k traces.

**Langfuse.** MIT-licensed open source. Self-hostable on Postgres + ClickHouse. Framework-agnostic via OpenTelemetry. Strongest open-source path. Cloud tier from $59+/seat. ~15% overhead in some benchmarks (decoupled integration). Used at scale (Canva is a public reference). Free tier: 50k observations/month.

**Laminar.** Apache 2.0 open source. OpenTelemetry-native. Built specifically for long-running agents — transcript view, agent rollout debugger, browser-agent session replay, SQL over traces. Helm chart for one-command self-host. The agent-debugging-focused alternative.

**Arize Phoenix.** Elastic 2.0 license. OpenTelemetry-native via OpenInference. Strong on eval rigor; fits notebook and eval-heavy workflows.

**Helicone.** Open-source proxy. Simplest install (one URL change). Trade-off: shallower depth — traces at the API call level, not the agent execution level. Right for "I just need cost tracking and basic request logging."

**Braintrust.** Closed source, eval-first. Strong regression harness. Tracing more bolted-on than primary.

**OpenLLMetry / Traceloop.** Vendor-neutral OpenTelemetry instrumentation SDK. Most other backends (Laminar, Langfuse, Phoenix, LangSmith) ingest its spans. The safest instrumentation choice for portability.

**The pairing pattern:**
- LLM observability platform (one of the above) for agent traces, eval, LLM-specific metrics
- Whole-stack APM (Datadog, Honeycomb, New Relic) for host metrics, app errors, deployment health

These cover different layers; production deployments need both.

**A decision matrix:**

| If you need... | Pick... |
|---|---|
| Deep LangChain/LangGraph integration | LangSmith |
| Open-source self-host | Langfuse (full-featured) or Laminar (agent-debug-focused) |
| Eval rigor primary | Arize Phoenix or Braintrust |
| Just cost tracking on raw API calls | Helicone |
| Vendor-neutral instrumentation | OpenLLMetry, point at any backend |
| Compliance / data residency | Self-hosted Langfuse or Laminar |
| Already on Datadog / New Relic | Their LLM observability modules + add agent tracing |

The honest March 2026 retro from a stack assessment: "If you're on LangGraph, LangSmith. If framework-agnostic, Langfuse. If eval rigor is the priority, Arize Phoenix. If you need agent-specific debugging in open source, Laminar."

::: brain
You're greenfield. Pick LangSmith or Langfuse?

(Both are reasonable. LangSmith if you're committing to LangChain/LangGraph and want the deepest framework integration. Langfuse if you want framework freedom or self-host. Laminar if your primary use case is debugging long-running agents specifically. The "right" choice is the one that matches your other architectural commitments — observability is a multi-year choice; pick on stack fit, not feature checklist.)
:::

---

## Section 5 — Cost Management (≈4 pages)

Agents can run away with cost in ways that are hard to predict from prototype usage.

**The four cost runaway patterns:**

**1. Token escalation.** Long-running agents accumulate context. By turn 30, each turn costs 10× turn 1. Module 8's compression and Module 7's context engineering address this — the operational concern is monitoring it.

**2. Sub-agent cascades.** Module 7's postmortem covered this. An agent spawns an agent that spawns an agent. Each layer adds cost. Without depth caps, runaway is plausible.

**3. Retry loops.** Agent retries a failing tool 50 times before giving up. Failed tools cost as much as successful ones (input tokens are charged regardless).

**4. Multi-agent token multiplier.** Module 13's 15× multiplier. Multi-agent systems are 15× more expensive per task than single-agent baseline. If a single-agent task was acceptably-priced, the multi-agent version may not be.

**Operational defenses:**

**Per-request cost ceiling.**

```python
class CostLimiter:
    def __init__(self, max_cost_usd: float):
        self.max_cost_usd = max_cost_usd
        self.accumulated = 0.0
    
    def add(self, prompt_tokens: int, completion_tokens: int, model: str):
        cost = self._compute_cost(prompt_tokens, completion_tokens, model)
        self.accumulated += cost
        if self.accumulated > self.max_cost_usd:
            raise CostLimitExceeded(self.accumulated, self.max_cost_usd)
    
    def _compute_cost(self, p, c, model):
        rates = MODEL_RATES[model]  # current pricing per 1M tokens
        return (p / 1_000_000) * rates.input + (c / 1_000_000) * rates.output
```

Wire this into every agent invocation. Surface excessive cost as a first-class error.

**Per-user / per-tier daily budgets.** Track cost by user, by feature, by tier. When a user crosses a threshold, throttle or escalate. Prevents one user's edge case from eating the budget.

**Loop detection.**

```python
class LoopDetector:
    def __init__(self, max_repeats: int = 3):
        self.action_history: list[str] = []
        self.max_repeats = max_repeats
    
    def check(self, action: dict) -> bool:
        sig = self._signature(action)  # hash of (tool_name, args)
        recent = self.action_history[-self.max_repeats * 2:]
        if recent.count(sig) >= self.max_repeats:
            return True  # loop detected
        self.action_history.append(sig)
        return False
```

Same tool with same args 3+ times in recent history is almost always a bug (the tool is failing in a way the model can't recover from). Detect; halt; surface as an error.

**Budget alerting.** Daily/weekly cost reports. Alerts on >X% deviation from baseline. Drift in cost is one of the earliest signals of agent regression.

**The 80/20 of cost optimization** (revisiting Module 6): caching matches, model routing, structured output to avoid retries, prompt caching for stable prefixes. These compound — each is 10-30%; together they're 50-70%.

::: postmortem
**The Agent That Spent $4,200 in One Hour**

A team deployed an agent with no per-request cost ceiling. A user query triggered a sub-agent cascade (Module 7 pattern). The cascade hit no caps. By the time anyone noticed (cost dashboard refreshed every 4 hours), the agent had spent $4,200 on a single query.

Fix: per-request ceiling at $5. Per-user daily ceiling at $50. Pager threshold at 5× normal cost in any rolling hour. Total time to implement: half a day. Total time to wish they'd implemented sooner: 4 hours.

**Lesson:** cost ceilings cost nothing to implement and protect you from incidents that cost real money. Add them on day one.
:::

---

## Section 6 — Failure Modes and Reading Failed Traces (≈5 pages)

What goes wrong in production. We catalog the common failure modes; we walk through reading a failed trace.

**The agent failure taxonomy:**

1. **Tool failures.** External API down, rate-limited, returned unexpected schema, returned valid but wrong data.
2. **Model failures.** Refusals on unexpected inputs, schema violations on structured output, runaway lengths, drift in tone.
3. **Loop failures.** Same action repeated, deadlock between agents, no progress toward goal.
4. **Coordination failures (multi-agent).** Sub-agent failed silently; supervisor synthesized broken result; worker contradicted worker.
5. **Context failures.** Compression dropped a critical fact; relevant memory not retrieved; context window overflowed.
6. **Guardrail failures.** False positives blocking legitimate queries; false negatives letting through bad ones.
7. **Cost failures.** Runaway token use; sub-agent cascade; retry loop.
8. **Validation failures (sagas).** Step claimed success; world state disagrees; compensation triggers.

**Reading a failed trace — a worked example.** We provide a sample failed trace and walk through it.

```
agent.run [DURATION 45.3s, COST $2.34, STATUS error]
├── agent.plan [DURATION 1.1s, COST $0.04]  ← OK
├── agent.tool_call.search [DURATION 0.8s, STATUS ok]  ← OK
├── agent.tool_call.read_doc [DURATION 30.1s, STATUS timeout]  ← problem here
│   └── http.fetch [DURATION 30.0s, STATUS timeout, url=...]  ← root cause
├── agent.tool_call.read_doc (retry 1) [DURATION 30.0s, STATUS timeout]
├── agent.tool_call.read_doc (retry 2) [DURATION 30.0s, STATUS timeout]
└── agent.synthesize [DURATION ?s, STATUS not_reached]
```

The trace immediately reveals: tool retries on a 30-second timeout, eating most of the wall-clock time. Three retries × 30s × cost-per-retry. Whether the retries help is a separate question, but the failure mode is now obvious.

**Habits for reading traces:**

- Look at the full duration first. Where did time go?
- Look at the total cost. Where did tokens go?
- Find the first error. Errors propagate; the first one is usually the cause.
- Check for loops. Same action 3+ times = loop.
- Check trace context propagation. Missing spans look like the agent did less than it did.

**Drift detection from traces.** Sample production traces continuously. Compare distributions over time:
- Average tokens per call (rising = context growing or context engineering broken)
- Tool call rates per task (changing = agent behavior shifted)
- Tool failure rates (rising = upstream service degrading)
- Refusal rates (rising = safety classifier triggered more, or input distribution shifted)
- Latency distributions (tail growing = something slow showing up more)

When any of these drifts beyond threshold, alert. Most production agent regressions show up in trace stats before they show up in customer complaints.

::: code-exercise
**Exercise 17.1 — Instrument the agent with full observability.**

Take the agent you've built across Modules 2-16. Add OpenTelemetry instrumentation following GenAI semantic conventions. Wire a cost limiter. Wire a loop detector. Run a failing scenario (broken tool, runaway loop) and verify the failure surfaces in traces immediately. Then wire eval sampling — 1% of production traces flow into your eval queue.

Bonus: deploy to one of the observability platforms (Langfuse if self-hosting; LangSmith if you're on LangGraph) and screenshot a real failed trace from your test runs.
:::

---

## Section 7 — Computer-Use and GUI Agents: Brief Treatment (≈3 pages)

Computer-use agents (Anthropic Computer Use, OpenAI's computer-using agents, GUI agents) are operationally extreme. We give them a focused but brief treatment.

**Why they're operationally hard:**
- Action surface is unbounded (anywhere on screen)
- Failures are subtle (clicked wrong place; typed in wrong field; agent thinks it succeeded)
- Latency is high (each screenshot + action is multi-second)
- Costs are high (vision models for screenshot understanding)
- Recovery is hard (agent navigated to wrong place — how does it get back?)

**Benchmarks worth knowing:**
- **OSWorld** — 369 real-world computer tasks across OS, web, productivity. Long-horizon evaluation of computer-use agents in real environments.
- **AndroidWorld** — Android-specific GUI agent benchmark.

State-of-the-art on OSWorld remains well below human performance as of 2026 — these are hard tasks. Don't deploy computer-use agents on consequential workflows without strong human oversight.

**Operational patterns specific to computer-use:**

- **Screenshot diffing for verification.** After an action, screenshot. Diff against expected state. Halt on unexpected.
- **Stricter loop detection.** Click loops are common; cap at 2-3 same-action repeats.
- **Human-in-the-loop for any irreversible action.** Sending an email, making a purchase, deleting a file — all should require explicit human approval until you have very strong evidence the agent is reliable on the specific task.
- **Domain restriction.** Agent can only operate on whitelisted apps/sites. Cross-domain navigation requires re-auth and human approval.

**The honest framing:** computer-use agents are still a research-to-pilot transition in 2026. They work well in narrow pre-validated workflows. They fail in unpredictable ways outside those. Treat operational deployment as experimental; instrument heavily; expect surprises.

---

## Section 8 — What's Next (≈1 page)

Module 18 (capstone) builds a real multi-agent research system end to end, applying everything from Modules 1-17. The eval/observability/cost/failure infrastructure from this module shows up there as production-ready instrumentation, not as an afterthought.

After the capstone, the reader has: the conceptual model (Modules 1-3), the building blocks (4-9), the reliability layer (10-12), the multi-agent layer (13-15), the transactional layer (16), the production layer (17), and a working artifact (18). That's the book.

---

## Recurring elements used in this module

- 1 chapter opener
- 1 hook
- 2 No Dumb Questions
- 2 Brain Power prompts
- 2 Production Postmortems (the disconnected trace, the $4,200 hour)
- 1 Watch It! / Gotcha block
- 1 Code Exercise (full observability instrumentation)
- 2 Pullquotes
- 1 trace tree diagram
- 1 platform-decision table
- 1 Bullet Points recap

::: sam-arc
Sam, having shipped a working multi-agent system through Modules 13-16, gets the question from the head of engineering: "how do we know it's working?" Sam realizes the eval infrastructure is essentially missing. Spends two weeks adding OpenTelemetry instrumentation, an eval harness fed by production traces, cost ceilings, loop detection, drift monitoring. Catches three issues no one had noticed: a sub-agent that was failing silently 8% of the time, a cost outlier from a customer's adversarial query, a tool call rate drift on the recommendation feature. Each fix would have been impossible to find without the instrumentation. Sam's arc this module: **observability is non-negotiable; the agent works as well as you can prove it works, no better.**
:::

::: page-budget
S1 (Production reality check): 3p
S2 (Evaluation): 6p
S3 (OpenTelemetry/GenAI semconv): 5p
S4 (Choosing a platform): 5p
S5 (Cost management): 4p
S6 (Failure modes / reading traces): 5p
S7 (Computer-use brief treatment): 3p
S8 (Next): 1p
Recurring elements: 6p
TOTAL: ~38 pages
:::

::: sources
**Must verify when drafting:**

- Gartner 2025 finding: 40% of agentic AI projects canceled by end of 2027 — verify exact figure and source
- IUI 2025 "No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding" — for the eval calibration section
- OpenTelemetry GenAI semantic conventions — current attribute names and stability status
- LangSmith pricing as of drafting (currently $39/seat/month Plus + $0.50/1k traces)
- Langfuse open-source license (MIT) and self-host requirements (Postgres + ClickHouse)
- Laminar license (Apache 2.0)
- Arize Phoenix license (Elastic 2.0)
- Observability platform overhead benchmarks (LangSmith near-zero, Langfuse ~15% per AImultiple Mar 2026 analysis) — verify
- OSWorld benchmark — current SOTA scores
- AndroidWorld benchmark — current SOTA scores
- Anthropic Computer Use latest documentation
- AutoGen maintenance status (per the 2026 articles, "effectively maintenance-mode" — verify language)
- Model pricing as of drafting (Anthropic Opus 4.7, Sonnet 4.6, Haiku 4.5 rates)

**Stable knowledge:**
- The three eval layers (component, trajectory, system)
- LLM-as-judge methodology and its caveats
- The four cost runaway patterns
- The agent failure taxonomy
- Trace reading habits

**Cross-references:**
- Module 6 — model routing and caching for cost
- Module 7 — context engineering for token economics
- Module 8 — compression for long sessions
- Module 11 — reflection patterns evaluable through trajectory eval
- Module 12 — guardrails feed observability with decision logs
- Module 13/14 — multi-agent observability needs are a superset of single-agent
- Module 16 — saga state in traces
:::

::: bullet-points
### Module 17 in eight bullets

(filled at draft time)

- Production agents need three properties: measurable, debuggable, bounded
- Eval has three layers (component, trajectory, system); most teams underinvest in system-level
- Production traces beat synthetic eval cases — sample 1%, label, accumulate
- OpenTelemetry GenAI semantic conventions are the 2026 standard for telemetry
- Pick observability platforms by stack fit, not feature checklist; pair LLM observability with whole-stack APM
- Cost defenses are non-optional: per-request ceilings, per-user daily budgets, loop detection, alerting
- Read failed traces by total duration, total cost, first error, then check loops and trace context
- Computer-use agents are operationally extreme; treat as experimental; instrument heavily
:::
