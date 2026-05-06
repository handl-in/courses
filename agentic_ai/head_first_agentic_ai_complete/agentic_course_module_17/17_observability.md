# Module 17 — Observability for Production Agents

::: chapter-opener
<div class="module-num">MODULE 17</div>
<div class="module-title">Observability for Production Agents</div>
<div class="subtitle">Sagas (M16) make failures recoverable.<br>Observability makes failures visible — and diagnosable in minutes instead of hours.</div>
<div class="pages">~38 pages · the operational discipline that turns "agents that work" into "agents you can debug at 3 a.m."</div>
:::

::: hook
Wednesday morning, two months after the saga rollout from M16. Sam was reviewing the on-call queue. The structured escalations from compensation failures were arriving as designed — about 3 per month, exactly as the metrics from M16 had predicted. The system was catching real problems and routing them to humans before customers noticed.

But the on-call team was unhappy.

Sam pulled up the logs from yesterday's escalation. The saga had failed during a customer-credit application; specifically, `update_billing_record` step had returned a permanent error during compensation. The escalation contained the saga ID, the failed step, the error message ("billing record version conflict"), and a structured suggested-actions block.

What the on-call engineer had needed to figure out:

- *When did the original action sequence start?*
- *Which agent invoked the saga, in what conversation, after what reasoning?*
- *Was the version conflict caused by another agent acting concurrently? Or by a human in the admin panel?*
- *What did the agent's earlier tool calls show about the billing record state at start of the saga?*
- *Has this happened before? How often?*

The escalation contained none of that. It contained the saga's structured failure, which was useful for understanding *what* failed. It contained nothing about *why*. To answer "why," the engineer had spent forty minutes pulling correlated logs from six different services: the agent's reasoning trace, the orchestrator's saga state, the billing service's audit log, the admin panel's activity log, the conversation history, and the metrics dashboard. Each of those had a different format, a different timestamp scheme, and a different way of identifying the customer or the saga.

Forty minutes to diagnose one escalation. The team was hitting three escalations per month. That meant two hours per month per engineer just on diagnosis — not counting fix time, not counting the cognitive load of context-switching into trace archaeology mode.

Sam wrote it: *"M16 made failures recoverable. We need the failures to also be DIAGNOSABLE. The compensations are running; the on-call queue is working; the system is structurally sound. But every failure is still a forty-minute investigation. That's not sustainable."*

The next thing Sam wrote: a list of every system that should have been emitting structured traces at every step, and wasn't.
:::

---

## What this module is

Module 16 built the discipline that makes failures *recoverable*. This module is about the discipline that makes failures *visible* — and crucially, *diagnosable* without forty-minute archaeology expeditions.

The relationship matters. Sagas tell you *what* failed; observability tells you *why* and *what was happening when it did*. Without observability, sagas catch the symptom but the diagnosis still takes hours. With observability, the saga's escalation arrives in the on-call queue with the full context, and the diagnosis collapses from forty minutes to forty seconds.

The structural insight from the buildfastwithai writeup: *"In 2026, the competitive advantage in AI development has shifted from prompt engineering to observability — the ability to see exactly why an agent succeeded or failed in a multi-step task is now more valuable than the initial instruction."*

That's strong, but it's directionally right. Production agents fail in ways pure software doesn't:

- **Non-determinism.** Same prompt, different output. Reproducing an issue requires capturing the exact input, model, parameters, and tool results at the time of the call.
- **Multi-step reasoning.** A single user request triggers 10-50 LLM calls, each with their own context, tool calls, and decisions. Without trace continuity, the trajectory is invisible.
- **Multi-agent coordination.** When subagents run in parallel, you have N timelines that need to be correlated. Standard logs don't reconstruct the topology.
- **Token and cost economics.** Latency and cost track tokens, not request counts. A "slow request" can mean "the agent burned 80K tokens looping" — different problem than "the network was slow."
- **Emergent behaviors.** Multi-agent systems develop coordination patterns no individual agent created. These only show up in aggregate.

Standard application observability (logs, metrics, traces) needs adaptation to address these. The good news: by 2026 the adaptation is well-understood. OpenTelemetry's `gen_ai.*` semantic conventions landed as stable in early 2026. Anthropic's Managed Agents includes observability built in. The Agent SDK exports OTLP traces to any compatible backend. The infrastructure exists; the question is using it well.

This module:

- **Section 1** — what makes agent observability different from regular observability
- **Section 2** — the three pillars (logs, metrics, traces) for agent systems
- **Section 3** — distributed tracing across agents, with OpenTelemetry GenAI conventions
- **Section 4** — the metrics that matter: leading vs lagging, per-agent dimensions, what to alert on
- **Section 5** — structured event logging: every reasoning step, tool call, handoff as a queryable event
- **Section 6** — alerting patterns: catching degradation before it becomes failure
- **Section 7** — debugging multi-agent emergent behaviors
- **Section 8** — Anthropic's first-party observability: Claude Console, Managed Agents tracing, Agent SDK OTLP
- **Section 9** — Sam's customer-success bot post-observability rollout
- **Section 10** — the framework

By the end of this module:

- You'll know which signals to instrument and which to skip
- You'll have working examples of OpenTelemetry-based agent tracing
- You'll be able to design alert hierarchies that catch problems before customers do
- You'll know how to debug emergent multi-agent behaviors from production traces

This is the third operational discipline (after M12 guardrails and M16 sagas). M12 prevented bad inputs/outputs/actions; M16 made failures recoverable; M17 makes them diagnosable. Together they're the difference between agent systems that ship and agent systems that survive in production.

::: pullquote
The discipline that distinguishes "agents that work" from "agents you can debug at 3 a.m." isn't sophistication of model or cleverness of prompt. It's whether the system tells you what it was doing when it broke. Observability is the difference between "we'll figure out what happened" and "the trace shows exactly what happened."
:::

---

## Section 1 — Why Agent Observability Is Different

Standard application observability handles deterministic systems where the same input reliably produces the same output. Logs tell you what happened; metrics tell you how the system behaves over time; traces show the request flow. The discipline is mature, the tooling is excellent, and most teams know how to use it.

Agent systems break the determinism assumption. The Uptrace 2026 piece named it directly: *"The same prompt produces different outputs. You can't reproduce an issue without capturing the exact input, model parameters, and temperature at the time of the call."* That single property cascades into multiple observability requirements that don't apply to standard services.

### Five things that make agent observability harder

**1. Inputs and outputs are large and unstructured.** A standard API call has a JSON request and JSON response that fit in a log line. An LLM call has potentially 100K+ tokens of input context and a free-form response. You can't log them in full at production volume; you can't log them in summary and reproduce issues.

**2. Multi-step trajectories are the unit of work.** A standard request maps to one operation. An agent request maps to 5-50 LLM calls, tool executions, and decisions, all of which together form one user-perceived operation. Tracing each call independently doesn't show the trajectory; tracing the whole request as one span loses the steps.

**3. Tool calls span systems.** A single agent turn might query your database, hit three external APIs, call an MCP server, and process results. The trace needs to span all of those, with proper parent-child relationships, while still being readable as one logical agent operation.

**4. Cost is a first-class metric.** Token usage isn't an implementation detail; it's the cost driver and a leading indicator of problems. Tokens-per-request, tokens-per-step, tokens-per-tool-call are first-class signals, not derived metrics.

**5. Emergent behaviors only appear in aggregate.** A single agent's trace might look normal. The same agent across thousands of conversations might show a pattern — context bloat, increasing latency over time, gradual policy drift. Observability has to support both per-request debugging and aggregate analysis.

### What this means for instrumentation

Three implications:

**Capture inputs and outputs at the trace level, not the log level.** Logs are unstructured noise; traces are structured events. The full prompt, full response, full tool call payload should be on the trace, queryable by trace ID.

**Treat multi-step trajectories as first-class.** Every LLM call gets a child span; every tool call gets a child span; every saga step gets a child span. The root span represents the user-perceived operation. The trace tree shows the full trajectory.

**Standardize attributes across the stack.** OpenTelemetry's `gen_ai.*` semantic conventions exist exactly for this. `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` — standardized names that work across vendors.

The Uptrace piece is direct: *"OpenTelemetry instruments LLM applications by wrapping API calls in spans with standardized gen_ai.* attributes — model name, token counts, finish reason — defined by the OpenTelemetry GenAI semantic conventions."*

::: nodumbq
**Q: My team uses Datadog/Honeycomb/New Relic for the rest of our stack. Do I need a separate observability platform for agents?**

You probably don't. The 2026 landscape has converged on OpenTelemetry as the export protocol, and most major observability platforms support OTLP. Your existing platform almost certainly accepts OpenTelemetry traces with `gen_ai.*` attributes. You may need to set up dashboards and alerts specifically for agent metrics, but the data plumbing reuses what you have. The exception: if you need very deep LLM-specific features (prompt diffing, evaluation tracking, regression detection), tools like Langfuse layer on top of your existing observability rather than replacing it.

**Q: Anthropic's Managed Agents has observability built in. Do I still need OpenTelemetry?**

Yes, for systems that span beyond the agent itself. Managed Agents gives you excellent observability for the agent's reasoning, tool calls, and lifecycle events — within the Claude Console. For tracing that spans your application code, your downstream services, your sagas, and the agent, you need a unified backend. Most teams use both: Managed Agents Console for deep agent diagnostics, OpenTelemetry for the cross-system view. The Agent SDK exports OTLP directly so the trace tree is unified.
:::

---

## Section 2 — The Three Pillars for Agent Systems

Standard observability rests on three pillars: logs, metrics, traces. The same applies to agents, but each pillar has agent-specific patterns.

### Logs

The least-changed pillar. Agent systems still need standard application logs for non-LLM concerns: HTTP errors, database failures, infrastructure issues. The agent-specific addition: structured events at decision points (Section 5).

What changes for agents:

- **Don't log full prompts and responses to standard logs.** They're too large, too noisy, and they leak into log retention systems where you don't want them. Use traces instead.
- **Log the boundary events.** Saga start, saga complete, escalation created, agent invoked, agent completed. These are queryable, indexable, and small.
- **Use correlation IDs religiously.** Every log line for a given user request includes the same correlation ID, propagated from the entry point through every system the request touches.

### Metrics

The pillar where agent systems diverge most. Standard metrics — request counts, error rates, p50/p99 latency — apply but don't capture the LLM-specific dimensions.

Agent-specific metrics by category:

**Token metrics:**
- `gen_ai.usage.input_tokens` — input tokens per LLM call
- `gen_ai.usage.output_tokens` — output tokens per LLM call
- `gen_ai.usage.total_tokens_per_request` — sum across the trajectory
- Cost in dollars (derived: tokens × per-model rate)

**Latency metrics:**
- `gen_ai.client.operation.duration` — per-LLM-call latency
- `agent.trajectory.duration` — full trajectory latency
- `agent.tool_call.duration` — per-tool-call latency

**Loop and reasoning metrics:**
- `agent.turns_per_request` — how many model calls per user request
- `agent.tool_calls_per_request` — how many tool calls per user request
- `agent.context_size` — context tokens at start of each turn
- `agent.compaction_events` — how often the system compacted (M8)

**Action metrics:**
- `agent.action.attempted` — by action name and outcome (success/failure)
- `agent.saga.duration` — per-saga latency
- `agent.saga.compensation_invocations` — how often compensation ran
- `agent.saga.escalation_to_human` — count by reason

**Quality metrics (where measurable):**
- `agent.guardrail.blocks` — by category (Section 6, M12)
- `agent.refusal_rate` — model declining to respond
- `agent.confidence_distribution` — when the agent reports confidence

The metrics matter because they're how you alert (Section 6) and how you spot drift over time. Without them, you're flying blind on cost and quality.

### Traces

The pillar agent systems most depend on. The trace is what reconstructs the trajectory: every LLM call, tool call, saga step, handoff, all in their proper hierarchy.

OpenTelemetry's GenAI conventions define the standard span structure:

```python
# Pseudocode showing OpenTelemetry GenAI span shape
with tracer.start_as_current_span(
    "chat.claude-sonnet-4-6",
    attributes={
        "gen_ai.system": "anthropic",
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "claude-sonnet-4-6",
        "gen_ai.request.max_tokens": 4096,
        "gen_ai.request.temperature": 0.7,
    }
) as span:
    response = await client.messages.create(...)
    span.set_attribute("gen_ai.response.model", response.model)
    span.set_attribute("gen_ai.response.id", response.id)
    span.set_attribute("gen_ai.usage.input_tokens", response.usage.input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", response.usage.output_tokens)
    span.set_attribute("gen_ai.response.finish_reasons", [response.stop_reason])
```

Each LLM call is a span. Each tool call is a span. Each saga step is a span. They nest under the user-request root span.

The Anthropic Agent SDK's OpenTelemetry integration handles this automatically:

```bash
# From Anthropic's Agent SDK docs
export OTEL_EXPORTER_OTLP_ENDPOINT="https://your-otlp-endpoint"
export OTEL_SERVICE_NAME="my-customer-success-agent"
export OTEL_RESOURCE_ATTRIBUTES="deployment.environment=production,service.version=2.4.1"
```

With those environment variables set, the Agent SDK exports traces, metrics, and structured events to any OTLP-compatible backend. From the Anthropic docs: *"It records spans around each model request and tool execution, emits metrics for token and cost counters, and emits structured log events for prompts and tool results."*

The auto-instrumentation captures most of what you need. Custom instrumentation fills in the application-specific events: saga steps, business operations, domain-specific metrics.

::: brain
What's the right span structure for Sam's customer-credit application from M16's saga? Sketch the trace tree.

(Root span: `customer.support.request` (the user's ticket). Child: `agent.turn` (one model call analyzing the ticket). Child: `tool.fetch_account_state`. Child: `agent.turn` (model decides credit warranted). Child: `tool.apply_customer_credit` — which itself starts a saga. The saga is a child span: `saga.customer_credit`. Inside that: `saga.step.apply_credit`, `saga.step.update_billing_record`, `saga.step.send_confirmation_email`, `saga.step.schedule_followup`. If a step fails, sibling spans for compensations: `saga.compensation.update_billing_record` (failed), `saga.compensation.apply_credit`. Above the saga step spans, the underlying API calls to billing service, email service, etc. The trace tree is essentially a visual reconstruction of the full trajectory; one trace ID spans all of it; the on-call engineer reading the trace sees exactly what happened in what order.)
:::

---

## Section 3 — Distributed Tracing Across Agents

The single most important observability artifact for agent systems. A trace tells the story of one user request as it flows through the system, with every decision, tool call, and saga step in proper order.

For multi-agent systems, the trace gets more complex but more valuable. Multiple agents running in parallel create multiple branches in the trace tree; the structure shows orchestrator-worker handoffs, peer collaboration, sagas, and any retries.

### Building a multi-agent trace

The pattern: every agent's execution carries a parent trace context. When an orchestrator spawns a worker, the worker's trace is a child of the orchestrator's trace. The trace IDs propagate explicitly.

```python
from opentelemetry import trace
from opentelemetry.propagate import inject, extract


tracer = trace.get_tracer(__name__)


async def orchestrator(user_request: str) -> str:
    """Orchestrator with explicit trace context propagation to workers."""
    with tracer.start_as_current_span("agent.orchestrator") as span:
        span.set_attribute("agent.role", "orchestrator")
        span.set_attribute("user.request", user_request[:500])  # truncated

        plan = await create_plan(user_request)

        # Capture trace context to pass to workers
        carrier: dict[str, str] = {}
        inject(carrier)

        # Workers run in parallel; each carries the orchestrator's trace context
        worker_results = await asyncio.gather(*[
            run_worker_with_context(task, trace_carrier=carrier)
            for task in plan.subtasks
        ])

        return await synthesize(user_request, worker_results)


async def run_worker_with_context(task: WorkerTask, trace_carrier: dict[str, str]) -> dict:
    """Worker that joins the orchestrator's trace."""
    # Extract the parent context from the carrier
    parent_context = extract(trace_carrier)

    # Start a child span that's parented to the orchestrator
    with tracer.start_as_current_span(
        f"agent.worker.{task.task_id}",
        context=parent_context,
    ) as span:
        span.set_attribute("agent.role", "worker")
        span.set_attribute("worker.task_id", task.task_id)
        span.set_attribute("worker.objective", task.objective[:500])

        result = await execute_worker_loop(task)

        span.set_attribute("worker.status", result.status)
        span.set_attribute("worker.confidence", result.confidence)
        span.set_attribute("worker.tokens_used", result.tokens_used)

        return result.dict()
```

Three things in this code worth noting:

**Explicit context propagation.** `inject()` serializes the trace context into a carrier dict; `extract()` reconstructs it on the receiving side. This works whether workers are in the same process or remote.

**Per-agent attributes.** Every span has `agent.role` so you can filter traces by role. Multi-agent systems with many roles (planner, generator, evaluator, critic) become navigable.

**Bounded payloads.** `[:500]` truncations on the request and objective. Don't put unbounded text on spans; truncate explicitly. For full text capture, use a dedicated event field with size limits, or store in a separate system referenced by ID.

### Trace continuity across sagas

Sagas (M16) span multiple steps; the trace must continue across them. With Temporal, each activity is automatically a child span of its workflow. Hand-rolled sagas need explicit propagation:

```python
async def execute_saga_traced(saga: Saga) -> dict:
    """Saga execution with full trace coverage."""
    with tracer.start_as_current_span(f"saga.{saga.saga_type}") as saga_span:
        saga_span.set_attribute("saga.id", saga.saga_id)
        saga_span.set_attribute("saga.type", saga.saga_type)
        saga_span.set_attribute("saga.step_count", len(saga.steps))

        completed = []
        try:
            for i, step in enumerate(saga.steps):
                with tracer.start_as_current_span(
                    f"saga.step.{step.name}"
                ) as step_span:
                    step_span.set_attribute("saga.step.index", i)
                    step_span.set_attribute("saga.step.name", step.name)
                    step_span.set_attribute("saga.step.idempotency_key", step.idempotency_key)

                    result = await step.forward(saga.context)
                    step_span.set_attribute("saga.step.status", "completed")

                    saga.context[step.name] = result
                    completed.append(step)

            saga_span.set_attribute("saga.status", "success")
            return {"status": "success", "context": saga.context}

        except Exception as e:
            saga_span.set_attribute("saga.status", "compensating")
            saga_span.record_exception(e)

            # Compensate with traced spans
            for step in reversed(completed):
                with tracer.start_as_current_span(
                    f"saga.compensation.{step.name}"
                ) as comp_span:
                    try:
                        await step.compensate(saga.context)
                        comp_span.set_attribute("compensation.status", "completed")
                    except Exception as comp_error:
                        comp_span.set_attribute("compensation.status", "failed")
                        comp_span.record_exception(comp_error)
                        # Section 7 from M16: escalation
                        raise

            saga_span.set_attribute("saga.status", "compensated")
            return {"status": "compensated", "error": str(e)}
```

The trace now shows the full saga structure: forward steps that succeeded, the failure point, the compensations that ran. When the on-call engineer pulls the trace by saga ID, they see the entire story in one tree.

### Sampling for cost control

Production agent systems generate enormous trace volume. Every LLM call is a span; every tool call is a span; every multi-step trajectory has dozens of spans. At scale, the trace storage cost can rival the LLM inference cost.

The DEV.to OTel piece is direct: *"Trace volume can surprise you. Agents that loop — retries, multi-turn conversations — generate a lot of spans. Set up sampling early. A simple tail-based sampler that keeps error traces and samples 10% of success traces works well."*

The sampling discipline:

- **Keep all error traces.** These are the ones you actually need.
- **Keep all traces with anomalies.** Unusual token counts, unusual durations, unusual tool call patterns.
- **Sample successful traces at a low rate.** 10% works; 1% is enough for very high volumes.
- **Always keep traces for specific high-value operations.** Customer-impact transactions, financial operations, escalations.

```python
class TailSampler:
    """
    Tail-based sampling: decide what to keep AFTER the trace completes.
    Knows the outcome before deciding whether to retain.
    """
    def should_keep(self, trace: Trace) -> bool:
        # Always keep errors
        if trace.has_errors:
            return True

        # Always keep escalations
        if trace.has_attribute("saga.status", "escalated"):
            return True

        # Always keep anomalies
        if trace.tokens > self.token_anomaly_threshold:
            return True
        if trace.duration > self.duration_anomaly_threshold:
            return True

        # Always keep customer-impacting operations
        if trace.has_attribute("operation.customer_facing", True):
            return True

        # Sample a percentage of normal successful traces
        return random.random() < self.success_sample_rate  # e.g., 0.10
```

Without tail sampling, your trace storage costs grow linearly with traffic. With it, costs grow with anomaly volume — much slower in practice.

::: pullquote
The trace is the single most useful observability artifact for agent systems. Properly structured, one trace tells the story of one user request — the reasoning, the tool calls, the saga execution, the failures, the compensations. The on-call engineer reads the trace and sees what happened. Without it, the same diagnosis takes hours of correlated log archaeology.
:::

---

## Section 4 — The Metrics That Matter

Logs and traces are for debugging individual requests. Metrics are for spotting trends, alerting on degradation, and understanding aggregate behavior over time.

### The hierarchy of agent metrics

**Tier 1: Always alert on these.**
- **Error rate** — agents failing to complete requests
- **Saga escalation rate** — sagas escalating to humans (M16)
- **Guardrail block rate** — by category (M12); spike indicates either attack or behavior change
- **Cost per request** — tokens × rate; spike indicates context bloat or loops

**Tier 2: Watch for trends, alert on significant drift.**
- **Trajectory length** — turns per request, tool calls per request; drift indicates degraded reasoning
- **p99 latency** — for the full trajectory
- **Compensation invocation rate** — sagas hitting compensation paths (vs success path)
- **Refusal rate** — model declining to respond; drift indicates prompt issues
- **Token usage distribution** — input and output tokens per request; right-shifted distribution indicates context bloat

**Tier 3: Diagnostic, not alerting.**
- **Per-tool latency and error rate** — to identify slow tools
- **Per-model usage** — when using multiple models (M6)
- **Confidence distribution** — when the agent reports confidence

### Per-agent dimensions

For multi-agent systems, every metric should be dimensioned by agent role:

```python
# Example: emit metrics with dimensions
from opentelemetry import metrics

meter = metrics.get_meter(__name__)

agent_request_counter = meter.create_counter(
    "agent.request.count",
    description="Number of agent invocations",
    unit="1",
)

agent_token_counter = meter.create_counter(
    "agent.tokens.total",
    description="Total tokens consumed",
    unit="token",
)

agent_request_duration = meter.create_histogram(
    "agent.request.duration",
    description="Agent request duration",
    unit="ms",
)


async def record_agent_metrics(
    agent_role: str,
    model: str,
    duration_ms: float,
    input_tokens: int,
    output_tokens: int,
    status: str,
):
    attributes = {
        "agent.role": agent_role,
        "gen_ai.request.model": model,
        "agent.status": status,
    }

    agent_request_counter.add(1, attributes)
    agent_token_counter.add(input_tokens, {**attributes, "token.type": "input"})
    agent_token_counter.add(output_tokens, {**attributes, "token.type": "output"})
    agent_request_duration.record(duration_ms, attributes)
```

With these metrics, you can answer:

- *"What's the error rate for orchestrators vs workers?"*
- *"Which agent role uses the most tokens?"*
- *"Did our latest deployment change the worker's tool-call patterns?"*
- *"Is the compensation rate climbing for any specific saga type?"*

### The metrics that matter most

If you're starting from scratch, instrument these first:

1. **`agent.request.count`** by `agent.role` and `agent.status` — visibility into volume and success
2. **`agent.request.duration`** by `agent.role` — visibility into latency
3. **`gen_ai.usage.tokens`** by `gen_ai.request.model` and `token.type` — visibility into cost
4. **`agent.saga.outcome`** by `saga.type` — visibility into action durability (M16)
5. **`agent.guardrail.blocks`** by `block.category` — visibility into boundary events (M12)

Five metrics, dimensioned correctly, give you enough signal to operate the system. More metrics are better, but not at the cost of getting these five right.

::: gotcha
A common metric anti-pattern: high-cardinality dimensions. Adding `customer_id` or `request_id` as a metric dimension produces metric series that explode storage and break dashboards. Use them in *traces* (where they're indexed differently) or *logs* (where they're queryable). Never as metric labels. The right metric labels are: agent role, model, tool name, saga type, status — bounded vocabularies. Customer ID and request ID are infinite vocabularies.
:::

---

## Section 5 — Structured Event Logging

Beyond metrics and traces, agent systems benefit from a third class of observability data: structured events at decision points. The Claude Code hooks pattern (and the broader observability-via-hooks approach) shows what these look like.

### What events to capture

The decision points in agent execution where structured events earn their place:

- **Agent invocation start/end** — with the user request, the role, the model
- **LLM call start/end** — already covered by traces; events provide queryable summaries
- **Tool call start/end** — name, params, result, error
- **Saga events** — start, step start/end, compensation start/end, escalation
- **Handoff events** — orchestrator dispatching, worker returning, synthesizer receiving
- **Guardrail events** — block, allow with caveats, modify (PII redaction, etc.)
- **Memory events** — load, save, retrieval (M9)
- **Reflection events** — critique generated, revision applied (M11)

Each event has:

- A unique event ID
- A trace ID linking to the broader trace
- A timestamp
- A structured payload specific to the event type
- A correlation ID (saga ID, conversation ID, customer ID where applicable)

### Schema design

Structured events benefit from typed schemas:

```python
from pydantic import BaseModel
from typing import Literal
from datetime import datetime


class BaseAgentEvent(BaseModel):
    event_id: str
    event_type: str
    timestamp: datetime
    trace_id: str
    span_id: str
    agent_role: str
    correlation_ids: dict[str, str]  # e.g., {"saga_id": ..., "conversation_id": ...}


class ToolCallEvent(BaseAgentEvent):
    event_type: Literal["tool.call.start", "tool.call.complete", "tool.call.error"]
    tool_name: str
    tool_input: dict  # or reference if too large
    tool_output: dict | None = None
    tool_error: str | None = None
    duration_ms: float | None = None


class SagaEvent(BaseAgentEvent):
    event_type: Literal[
        "saga.start", "saga.step.complete", "saga.step.error",
        "saga.compensation.start", "saga.compensation.complete",
        "saga.escalation",
    ]
    saga_id: str
    saga_type: str
    step_name: str | None = None
    step_status: str | None = None
    error: str | None = None


class GuardrailEvent(BaseAgentEvent):
    event_type: Literal["guardrail.block", "guardrail.modify", "guardrail.allow"]
    layer: Literal["input", "output", "action"]  # M12 layers
    category: str  # the rule category that triggered
    severity: Literal["low", "medium", "high"]
    confidence: float
```

The events are queryable: *"show me all `saga.escalation` events in the last 24 hours, grouped by saga_type."* Or: *"show me all `guardrail.block` events with category='off_topic' for customer X."* Without structured events, the same query requires log parsing.

### Where events go

Three destinations:

**1. Trace backend.** Events attach to spans as span events. They're queryable through the trace UI, contextually linked to the trace they belong to.

**2. Event streaming platform.** For high-volume use cases, events go to Kafka, Kinesis, or equivalent for stream processing. This enables real-time alerting on event patterns.

**3. Searchable storage.** Events stored in a structured event store (Elasticsearch, ClickHouse, Snowflake) for ad-hoc analysis. *"How many compensation events happened across all saga types last quarter?"*

For most teams, the trace backend is enough — modern observability platforms (Honeycomb, Datadog, New Relic) handle structured events on spans well. Streaming and searchable storage add value for advanced use cases (real-time fraud detection on guardrail events, longitudinal analysis of agent behavior).

### The Claude Code hooks pattern

Anthropic's Claude Code instruments observability via hooks: lifecycle event handlers that fire at agent execution boundaries. The disler/claude-code-hooks-multi-agent-observability open-source project shows the pattern in action — every hook event becomes a structured event, exported to a backend, visualized in real-time.

Hook events Claude Code emits (per the docs):

- `PreToolUse` — before any tool call
- `PostToolUse` — after a tool call completes
- `PostToolUseFailure` — after a tool call fails
- `SubagentStop` — when a subagent completes
- `Stop` — when the main agent completes
- `PermissionRequest` — when the agent requests elevated permissions

For Claude Code-based agents, you get this instrumentation by writing hook scripts that emit structured events. For Agent SDK-based deployments, the OpenTelemetry integration covers most of the same ground via standard span events.

::: nodumbq
**Q: Don't traces already capture all this? Why do I need separate events?**

Traces capture the call structure and timing; events capture the semantically-meaningful business actions. A trace shows "tool call to apply_credit completed in 230ms"; an event captures "saga step apply_credit succeeded with idempotency key X for customer Y." The event is structured, queryable, and aggregatable in ways the trace span isn't. Most observability platforms support both — traces and events on traces — and the discipline is to use each for what it's best at: traces for trajectory understanding, events for business-level analysis.
:::

---

## Section 6 — Alerting Patterns

The point of metrics is to know when something's wrong. The discipline is alerting on the right signals at the right thresholds.

### Leading vs lagging indicators

The distinction matters more for agent systems than most.

**Lagging indicators** measure outcomes after they've happened: customer complaints, support tickets, payment errors. They're reliable but late — by the time you alert, the damage is done.

**Leading indicators** measure conditions that predict outcomes: token bloat, latency drift, error rate trends, guardrail block patterns. They're noisier but earlier — they let you fix problems before customers see them.

For agent systems specifically, leading indicators include:

| Leading indicator | What it predicts | Why |
|---|---|---|
| Rising token-per-request | Increased cost; possibly degraded quality | Context bloat; loops; over-tooling |
| Rising p99 latency | User experience degradation | Same causes; or model latency issues |
| Rising tool call count per request | Inefficiency; possibly wrong reasoning | Agent struggling to complete tasks |
| Rising guardrail block rate | Drift in user behavior or agent behavior | Either users testing edges or agent interpretation drifting |
| Falling confidence in agent outputs | Quality degradation | Agent uncertain on cases it used to handle |
| Rising compensation rate | Action durability degrading | Underlying services flaking |

Each one is alertable when it deviates significantly from baseline. The alert is "investigate," not "page someone." But over a few hours of drift, the leading indicator is what tells you the system is degrading before customers notice.

### Alert hierarchy

Production agent systems benefit from tiered alerting:

**Tier 1: Page immediately.**
- Error rate > X% for Y minutes
- Saga escalation rate > N per hour
- p99 latency > customer SLO threshold
- Customer-facing failures detected

**Tier 2: Notify on-call channel; investigate next business day if not actively impactful.**
- Token-per-request 50%+ above baseline
- Tool error rate for any single tool above N%
- Guardrail block rate for any category above 2× baseline
- Refusal rate above baseline

**Tier 3: Track on dashboard; investigate during regular operations review.**
- Per-tool latency drift
- Trajectory length distribution shifts
- Per-model usage shifts (might indicate routing issues)

The discipline of tiered alerts: Tier 1 is sacred (paging means waking someone up; reserve for actual problems). Tier 2 is useful (catches drift before it becomes Tier 1). Tier 3 is informational (useful for trending, not alerting).

### Anomaly detection on traces

Beyond threshold-based alerts, anomaly detection on individual traces catches outliers that don't fit aggregate patterns. The OpenObserve writeup on AI-powered observability frames it well: *"AI-powered RCA, log clustering, trace correlation, and LLM-powered anomaly detection."*

The pattern: a baseline of normal trace shapes (LLM call counts, tool call counts, token usage, structure) is computed. Traces that deviate significantly from baseline are flagged for investigation. This catches:

- **Reasoning loops.** A trace with 50 LLM calls when baseline is 5 — agent is stuck.
- **Excessive tool use.** A trace with 20 tool calls when baseline is 3 — possible reasoning failure.
- **Context bloat.** Token count 5× baseline — context isn't being compacted properly (M8).
- **Unusual paths.** Trace structure that doesn't match any baseline cluster — novel scenario worth reviewing.

For modest-scale agent systems, threshold-based alerting on aggregate metrics suffices. For high-scale or high-stakes systems, anomaly detection on traces catches the long tail of problems thresholds miss.

::: postmortem
**The Token Bloat That Hit a Cost Alert at 2 a.m.**

A team had a customer-support agent running on Sonnet 4.6. Their cost monitoring alerted when daily spend exceeded a threshold. One Tuesday, the cost alert fired at 2 a.m. — daily spend was 4× normal.

The on-call engineer pulled traces. The pattern: median trajectory length had drifted from 3 turns per request to 8 turns. Token usage per request had drifted from 12K to 38K. The cost spike was real.

Why? A deploy three days prior had updated a tool's description (a cosmetic change, the team thought) in a way that confused the agent. The agent started calling the tool, getting unexpected results, calling it again with different parameters, and gradually using more turns per request. The aggregate had drifted slowly enough that no single request looked anomalous, but the cumulative cost was 4× normal.

Two fixes:

1. **Token-per-request alert at Tier 2.** A 50%+ deviation from baseline trip a notification (not page). Would have caught this within hours of the deploy, not three days.

2. **Tool description rollback.** The cosmetic change was reverted; trajectory length returned to normal within an hour.

**Lesson:** cost alerts (lagging) catch problems after they've cost you money. Token-per-request drift (leading) catches the same problems hours earlier, when they're cheaper to fix. Instrument both; alert on the leading indicator at lower threshold than the lagging one.
:::

---

## Section 7 — Debugging Multi-Agent Emergent Behaviors

The hardest observability challenge. Multi-agent systems develop behaviors no individual agent created — coordination patterns, drift, echo chambers, cascading failures. These only show up in aggregate.

### Patterns that only emerge with multiple agents

**1. Cascading failures.** Worker A fails; orchestrator retries; retry has slightly different parameters; worker B (which had succeeded) now produces different output because the orchestrator's context changed. The original failure cascaded into a different failure pattern.

**2. Coordination drift.** Over thousands of interactions, the orchestrator and workers develop a "style" of communication that diverges from what the prompts specify. Workers start including more (or less) detail than expected; orchestrators start trusting (or distrusting) certain workers more. Hard to detect; harder to correct.

**3. Echo chamber effects.** Subagents that read each other's outputs reinforce each other's biases. The synthesizer amplifies what they all agreed on, even when they were all wrong in the same way (M11's MAR finding).

**4. Coordination deadlocks.** Agents waiting for each other in patterns that work in development but emerge under production conditions.

### Detecting emergent behaviors

The discipline: aggregate analysis on per-agent dimensions, looking for distributional shifts.

**Trajectory shape clustering.** Cluster traces by their structure (LLM call counts, tool calls, hierarchy depth). Healthy systems have a few stable clusters representing different operation types. Drift shows up as new clusters or shifting cluster sizes.

**Cross-agent correlation analysis.** Are workers that should be independent actually correlated? If subagent A's output predicts subagent B's output more than chance, you have echo chamber risk.

**Per-saga compensation patterns.** Are compensations clustering on certain saga types or steps? That's a leading indicator of underlying service degradation.

**Latency distribution drift.** Is p99 latency for orchestrator increasing while worker latencies are stable? Suggests orchestrator coordination overhead growing — possibly emergent.

### A diagnostic dashboard for multi-agent systems

The metrics that distinguish working multi-agent systems from emergent-failure-mode systems:

- **Worker independence:** Pearson correlation between sibling workers' outputs over time. Should be low (they're investigating different things). Rising correlation indicates echo chamber.
- **Trajectory diversity:** Number of distinct trace cluster shapes. Should be roughly stable. Rising diversity indicates novel scenarios; falling indicates collapse.
- **Coordination overhead:** Time spent in orchestrator synthesis vs workers doing actual work. Should be a stable ratio. Rising indicates coordination drift.
- **Error origination distribution:** Which agents produce errors. If errors shift from "any agent" to "always synthesizer," the synthesizer is becoming a bottleneck or has degraded.

These aren't alerts (most teams don't need to alert on them); they're dashboard metrics for periodic operational review. Once a quarter, the team reviews the dashboard, asks "are there any drift patterns?", and intervenes if needed.

::: brain
A multi-agent system's per-quarter review shows: trajectory diversity stable, coordination overhead stable, but worker independence has dropped from 0.15 to 0.42 over six months. What does this suggest, and what's the right intervention?

(Workers' outputs have become 3× more correlated than they were six months ago. Most likely causes: (1) the orchestrator's task descriptions to workers have drifted to be more similar, (2) workers are reading shared context that's biasing them similarly, (3) the synthesizer's preferences are leaking back into worker behavior through prompt updates. Intervention: audit recent prompt changes; check whether worker mandates have become more similar; consider re-establishing prompt diversity and explicitly instructing workers to investigate from different angles. The metric won't tell you the cause directly; it tells you to look.)
:::

---

## Section 8 — Anthropic's First-Party Observability

Anthropic ships observability infrastructure across two primary products. Worth dedicated treatment because they're the canonical implementations.

### Claude Console session tracing

For Claude Managed Agents (April 2026 GA), the Claude Console provides session-level observability. From Anthropic's docs:

> *"Claude Managed Agents provides observability tools in the Claude Console to help you monitor, debug, and understand your agent sessions. The Console provides a visual timeline view of your agent sessions."*

What's visible:

- **Session list** — all sessions with status, creation time, model
- **Tracing view** — chronological view of events (content, timestamps, token usage) within a session
- **Tool execution** — details of each tool call and its result

The interface is designed for debugging individual sessions. For "what happened in this specific saga that escalated to the on-call queue," the Claude Console gives you the agent's reasoning, tool calls, and decisions in a single timeline view.

The reviews from the buildfastwithai and unicodeveloper writeups are direct: *"Session tracing, integration analytics, and troubleshooting guidance are built into the Claude Console. You can see every tool call, every decision point, every failure mode."* For DIY agent setups, this kind of observability typically takes months to build; Managed Agents includes it.

### Claude Agent SDK OpenTelemetry export

For self-hosted agents using the Claude Agent SDK, OpenTelemetry export is built in. From the SDK docs:

> *"The Agent SDK can export this data as OpenTelemetry traces, metrics, and log events to any backend that accepts the OpenTelemetry Protocol (OTLP), such as Honeycomb, Datadog, Grafana, Langfuse, or a self-hosted collector."*

The integration is environment-variable-driven:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otlp.honeycomb.io"
export OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=$HC_API_KEY"
export OTEL_SERVICE_NAME="customer-success-agent"
export OTEL_RESOURCE_ATTRIBUTES="deployment.environment=production,service.version=2.4.1"
```

What gets exported:

- **Spans** around each model request and tool execution
- **Metrics** for token and cost counters
- **Structured log events** for prompts and tool results

The key advantage: the SDK does the instrumentation; your application code doesn't change. Whether you use Honeycomb, Datadog, Langfuse, or a self-hosted Tempo+Grafana stack, the same data flows through.

### Claude Code hooks for custom observability

For Claude Code-based agents, observability happens via hooks — lifecycle event handlers that fire at execution boundaries. The disler/claude-code-hooks-multi-agent-observability project demonstrates the pattern: every hook event becomes a structured event posted to a backend, visualized in real-time.

Hook events Claude Code emits:

- `PreToolUse` — before any tool call
- `PostToolUse` — after a tool call completes
- `PostToolUseFailure` — after a tool call fails
- `SubagentStop` — when a subagent completes
- `Stop` — when the main agent completes
- `PermissionRequest` — when the agent requests elevated permissions

For multi-agent observability specifically, the hooks pattern shines. Each agent emits events as it runs; the events flow to a central observability system; the dashboard shows the entire swarm in real-time.

### Choosing among Anthropic's observability options

For most production deployments, the stack is:

1. **Claude Managed Agents** for the agent runtime — observability built in via Console
2. **Anthropic Agent SDK** for self-hosted agents — OpenTelemetry export to your existing backend
3. **OpenTelemetry standard semantic conventions** for everything that spans your app and the agents

This gives you both the deep agent-specific debugging (Managed Agents Console) and the unified cross-system view (OpenTelemetry to a backend like Honeycomb or Datadog).

The specific recommendation for Sam's customer-success bot: run on Managed Agents (durability + saga support + observability), export OTLP to the team's existing Datadog deployment for cross-system correlation, use the Console for deep-dive debugging when escalations require it.

::: pullquote
The infrastructure for production agent observability is mature in 2026. Anthropic ships it built into Managed Agents; OpenTelemetry standardizes it across vendors; major observability platforms support agent-specific dashboards. The work for production teams isn't building observability — it's instrumenting deliberately and using what's there.
:::

---

## Section 9 — Sam's Customer-Success Bot, Post-Observability Rollout

Returning to the cold open. Sam's team had M16's saga discipline working: failures were recoverable, escalations were structured, the on-call queue was receiving them as designed. The diagnosis time was the gap. Sam applied the M17 patterns:

### The observability rollout

**1. Migrated the agent runtime to Managed Agents.** The customer-success bot moved from self-hosted Agent SDK to Managed Agents, gaining the Claude Console session tracing automatically.

**2. Wired OpenTelemetry through the entire stack.** Every service in the customer-success path — billing, email, ticketing, CRM — got OpenTelemetry instrumentation. Trace context propagation from the agent through every downstream call.

**3. Defined the trace tree explicitly.** Root span: customer support request. Children: agent turns, tool calls, saga executions. Spans for every step of every saga, with idempotency keys, status, attempts.

**4. Standardized correlation IDs.** Every observability artifact — traces, metrics, events, logs — included the same correlation IDs: customer_id, ticket_id, conversation_id, saga_id where applicable. Cross-system queries became instant.

**5. Implemented tail sampling.** All error traces kept; all escalation traces kept; all customer-facing operation traces kept; 10% of routine successful traces kept. Trace storage costs were 8% of what they would have been with full sampling.

**6. Built tier-based alerting.** Tier 1 paged on customer-facing failures and escalation rate. Tier 2 notified on token-per-request drift, p99 latency drift, tool error rates. Tier 3 went to a weekly dashboard review.

**7. Added the multi-agent diagnostic dashboard.** Even though the customer-success bot is single-agent, Sam added the diagnostic dashboard pattern in case the team built multi-agent systems later. The other teams' multi-agent systems immediately benefited.

### The escalation-to-resolution time

The metric that mattered most:

| Metric | Pre-observability | Post-observability |
|---|---|---|
| Median time from escalation arrival to root cause identified | ~38 minutes | ~4 minutes |
| Median time from root cause to fix (or fix-deferred decision) | ~25 minutes | ~12 minutes |
| Median total time on escalation per engineer | ~63 minutes | ~16 minutes |
| On-call engineer hours per month on agent escalations | ~8 hours | ~2 hours |

The gain wasn't preventing failures (M16 already did that); it was *diagnosing* them. The on-call engineer who used to spend 40 minutes pulling correlated logs from six services now pulled one trace, saw the entire trajectory, identified the root cause, and either applied a fix or filed a deferred-action ticket.

### The leading-indicator catches

In the first three months post-rollout, the Tier 2 alerts caught:

- **Two cases of token-per-request drift** — both due to prompt changes that increased context size; flagged within 6 hours; reverted before customer impact
- **One case of guardrail block rate spiking** — a new customer pattern (asking about competitor pricing in unusual phrasings); the input filter was tuned to handle the new pattern
- **Three cases of tool error rate elevation** — all transient infrastructure issues; alerted within minutes; resolved by tool team within an hour

None of these became customer-facing problems. Pre-observability, all three patterns would have manifested as customer complaints over a multi-day window.

Sam wrote one line for the playbook: *"M16 made failures recoverable. M17 made failures fast to diagnose. Together: failures stop being mysteries."*

::: code-exercise
**Exercise 17.1 — Observability audit.**

Pick an agent system in your stack. Run the audit:

1. **Trace coverage.** Can you reconstruct a single user request as a trace tree spanning the agent, tools, sagas, and downstream services? If not, what's missing?
2. **Correlation IDs.** Pick a recent customer issue. Could you query all observability artifacts (traces, logs, metrics, events) for that issue's correlation ID and get the full picture? If not, the IDs aren't propagating.
3. **Metrics dimensions.** Are your metrics dimensioned by `agent.role`, `gen_ai.request.model`, and other agent-specific dimensions? Or are they generic request-rate and error-rate metrics that don't tell you about agent behavior?
4. **Sampling.** Do you keep all error traces? All escalation traces? What percentage of successful traces? Calculate your storage costs vs your debugging usefulness.
5. **Alert hierarchy.** What pages on-call (Tier 1)? What notifies (Tier 2)? What goes to dashboards (Tier 3)? Are leading indicators in Tier 2 or only lagging indicators?
6. **Time-to-diagnosis.** Pick the last three escalations. How long did diagnosis take? If it's measured in tens of minutes per escalation, observability is the gap.

The goal is calibration. Most production agent systems have observability that worked in development and degrades silently as the system grows. The audit surfaces where the gaps are; the patterns from this module fix them.
:::

---

## Section 10 — The Framework

Adding observability discipline to production agent systems:

**1. Treat observability as core infrastructure, not an afterthought.** It's the difference between agents that ship and agents that survive. Budget engineering time for it from day one.

**2. Use OpenTelemetry as the foundation.** The `gen_ai.*` semantic conventions are stable as of early 2026. Auto-instrumentation exists for major LLM SDKs. Vendor-neutral export to any backend.

**3. Make traces the primary debugging artifact.** Every user request is a trace tree spanning agent reasoning, tool calls, saga steps, and downstream services. Properly structured, one trace tells the full story.

**4. Propagate correlation IDs religiously.** customer_id, ticket_id, conversation_id, saga_id — propagated through every observability artifact, queryable across systems. The on-call engineer should be able to pull everything related to one issue with one query.

**5. Standardize metric dimensions.** `agent.role`, `gen_ai.request.model`, `tool.name`, `saga.type`, `agent.status`. Bounded vocabularies. Never use unbounded values like customer_id as metric dimensions.

**6. Capture structured events at decision points.** Saga events, guardrail events, handoff events, memory events — typed schemas, queryable, aggregatable. They complement traces; they don't replace them.

**7. Implement tail sampling.** All errors, all escalations, all anomalies, all customer-facing operations. Sample successful routine traces at 10%. Storage costs become tractable.

**8. Tier alerts: page on critical, notify on drift, dashboard on diagnostics.** Tier 1 for customer-impacting failures. Tier 2 for leading indicators (token drift, latency drift, error rate trends). Tier 3 for periodic operational review.

**9. Build dashboards for emergent multi-agent behaviors.** Worker independence, trajectory diversity, coordination overhead, error origination distribution. Periodic review catches drift before it becomes failure.

**10. Use Anthropic's first-party observability when available.** Managed Agents Console for deep agent debugging; Agent SDK OTLP export for cross-system correlation; Claude Code hooks for custom event flows. Don't reinvent what's already shipped.

The discipline this enforces: **failures get visible before customers see them, and diagnosable in minutes when they happen.** Sam's customer-success bot reached this state through M16 (recoverable) + M17 (diagnosable). Every production agent system needs both layers; the question is whether you build them deliberately or discover them through painful incidents.

---

## Recap: Module 17 in eight bullets

::: bullet-points
- Agent observability is different from standard application observability. Five things make it harder: large unstructured inputs/outputs, multi-step trajectories as the unit of work, tool calls spanning systems, token economics as a first-class metric, emergent behaviors only visible in aggregate.
- The three pillars (logs, metrics, traces) all need agent-specific patterns. Traces are the most important — properly structured, one trace tells the story of one user request including reasoning, tool calls, and saga execution.
- OpenTelemetry's `gen_ai.*` semantic conventions are the de facto standard as of early 2026. Vendor-neutral export, auto-instrumentation packages, and stable conventions for model name, token counts, finish reason. Anthropic's Agent SDK exports OTLP directly.
- Distributed tracing across multi-agent systems requires explicit context propagation. Every agent's execution carries the parent trace context; workers' traces nest under orchestrators'; sagas span trace tree branches; the trace tree is the visual reconstruction of the full trajectory.
- The metrics that matter: error rate, escalation rate, guardrail block rate, cost per request (Tier 1 alerts); token-per-request drift, p99 latency, compensation rate, refusal rate (Tier 2 notifications); per-tool diagnostics and per-model usage (Tier 3 dashboard). Bounded dimensions only — never customer_id or request_id as metric labels.
- Structured events at decision points complement traces. Typed schemas for saga events, guardrail events, handoff events. Queryable and aggregatable in ways trace spans aren't. Anthropic's Claude Code hooks pattern shows the canonical implementation.
- Tail sampling controls trace storage costs while preserving debugging usefulness. Keep all errors, all escalations, all anomalies, all customer-facing operations. Sample 10% of routine successes. Storage costs grow with anomaly volume, not with traffic.
- Sam's customer-success bot post-rollout: median time from escalation to root cause dropped from 38 minutes to 4 minutes; on-call engineer hours per month on agent escalations dropped from 8 to 2. Leading indicators (Tier 2) caught 6+ issues in 3 months that would have become customer complaints pre-observability. The combination of M16 (recoverable failures) and M17 (diagnosable failures) is the foundation of production-grade agent operations.
:::

---

::: sam-arc
**Sam, after the observability rollout.**

Three months in. Sam was reviewing the on-call rotation metrics. The team had become noticeably less stressed about agent escalations. The pattern had inverted: escalations used to be dreaded ("oh god, another forty-minute trace dive"); now they were routine ("trace shows the issue, fix takes ten minutes, done").

The CTO asked Sam to present the work at the engineering all-hands. Sam thought about how to frame it. The honest framing wasn't *"we built better observability."* The honest framing was: *"we built the operational maturity that makes agent systems sustainable."*

Sam pulled up a slide showing four numbers across three months:

- **0 customer-facing incidents** caused by agent failures (M16 caught everything)
- **9 escalations to the on-call queue** (the structured failures M16 surfaced)
- **All 9 diagnosed in under 10 minutes each** (M17 made diagnosis fast)
- **6 leading-indicator alerts** caught issues before they became customer-impacting (M17 made drift visible)

The CTO's question: *"How does this scale to the next 10 agent systems we're going to build?"*

Sam thought about that. The patterns from M16 and M17 weren't customer-success-bot-specific. They were *agent-system-specific* — the operational discipline that turns any agent system from "works in dev" into "works in production at 3 a.m." The platform team had built the patterns; now they were reusable. The next agent system would inherit the saga discipline, the observability infrastructure, the alert hierarchy, the diagnostic dashboards. The cost of operating it would be the cost of building it well — not the cost of recovering from incidents nobody anticipated.

Sam said something like that. The CTO nodded. *"Make it the platform standard. Every agent system gets this from day one."*

That became the next platform documentation update: *Required for production agents — saga discipline (M16), observability instrumentation (M17), tier-based alerting, structured escalations.* Not optional. Not deferred until later. Required from day one.

Sam's arc this module: **the operational discipline that makes agents sustainable is the actual deliverable.** The agent's reasoning, the architecture, the prompts — those are necessary but not sufficient. M2 through M15 made the agent capable; M16 and M17 made the agent operable. The first half of the work made the agent *possible*; the second half made it *sustainable*. Both layers were the actual deliverable; the team had now built both.

One module left in the book — M18, the capstone. A complete system pulling together every layer Sam had learned over the seventeen modules. The full arc, from M2's hardened loop through M17's observability discipline. Sam closed the laptop. The platform was ready. The patterns were ready. The capstone would be the synthesis.
:::

---

## What's next

Module 18 is the capstone. A complete system pulling together every layer from the book — M2's hardened loop, M3's workflows, M4's tools, M5's MCP, M6's model selection, M7's context engineering, M8's compaction, M9's memory, M10's hallucination prevention, M11's reflection, M12's guardrails, M13's multi-agent decision, M14's architectures, M15's communication, M16's sagas, and M17's observability.

The capstone won't introduce new patterns — those are all in M2 through M17. The capstone will *synthesize* — show how the patterns compose into a complete production agent system, how the layers interact, where the trade-offs surface, and how to know when the system is ready for production.

Sam's customer-success bot, by M17's close, embodies most of the layers. The capstone takes a fresh problem (a deal-research-and-brief system) and walks through the full design from M2 through M17, showing how each layer's discipline contributes to the whole.

For now: take the audit exercise from Section 9. Pick a production agent system. Audit its trace coverage, correlation IDs, metric dimensions, sampling, alert hierarchy, and time-to-diagnosis. The patterns from M17 are mature; the infrastructure exists; the work for production teams isn't building observability but using what's available deliberately. Sam's team's escalation diagnosis time dropped 90% from instrumenting the patterns from this module. Yours probably can too.
