# Module 16 — Sagas: Transactional Safety for Agent Actions

::: chapter-opener
<div class="module-num">MODULE 16</div>
<div class="module-title">Sagas: Transactional Safety for Agent Actions</div>
<div class="subtitle">When agents take real-world actions, partial failures aren't bugs to fix — they're inevitable conditions to design for.<br>The saga pattern is how you stay sane.</div>
<div class="pages">~38 pages · the discipline that turns "agents that work" into "agents that survive Tuesday"</div>
:::

::: hook
Tuesday morning, 9:47 a.m. Sam was on a video call when the customer-success channel lit up. *"@sam — customer #2847 just told me they got a confusing email about a credit they never agreed to. I checked their account: $500 credit applied last night, no explanation in the ticket history."*

Sam pulled up the trace. The customer-success bot had handled a ticket from #2847 the night before about a billing dispute. The agent had researched, decided a $500 credit was warranted (within its authorized range, with M12 confirmation gate), gotten the customer's approval, and started the multi-step action sequence:

```
Step 1: apply_credit($500, customer=2847, ticket=87123) → SUCCESS at 23:14:22
Step 2: update_billing_record(credit_applied=500, ticket=87123) → SUCCESS at 23:14:24
Step 3: send_confirmation_email(customer=2847, ticket=87123, ...) → ?
```

Step 3 was the problem. The email service had been having issues that night — a regional outage Sam's monitoring had eventually flagged but not before this trace. The email send had returned a 500 error. The agent had retried twice (per its `max_retries=3` config from M2). Both retries also failed. The agent gave up and... did nothing. Logged the failure. Moved on to the next ticket in queue.

The customer's account now had:
- A $500 credit applied (real money)
- A billing record updated (state changed)
- No email sent (no confirmation)
- No follow-up scheduled (no rollback)
- No human flagged (no escalation)

The next morning the customer saw the credit on their account, didn't know why, called in confused. The CS rep had to dig through the ticket history to figure out what had happened.

Sam pulled the agent's logs and saw the same pattern across 14 other tickets that night. Not all of them had failures — most had completed cleanly. But for the ones with partial failures, the system had no plan. The agent's design assumed each step either succeeded or failed atomically. The reality was: step 1 could succeed, step 2 could succeed, step 3 could fail, and now the system was in a state nobody had explicitly designed for.

Sam wrote it: *"the agent loop assumes each turn is independent. The actions are not. We've been treating multi-step actions like they were single calls. We've been wrong."*
:::

---

## What this module is

Modules 2-15 built agents that decide well, communicate well, and constrain themselves well. This module is about what happens when those agents *act on the real world* and one of the actions fails partway through.

The problem is structural. An agent loop completes its turn or it doesn't — that's M2. But the *actions* the agent takes within a turn often span multiple operations: charge a card, update a ledger, send a notification, schedule a follow-up. Each of those operations can succeed or fail independently. The agent's reasoning treated them as one logical action; the systems treat them as separate calls; reality lives in the gap.

The saga pattern, well-established in distributed systems for fifteen+ years, is the canonical answer. Not the only answer — there are simpler patterns (atomic single-call operations) and more complex ones (formal two-phase commit). But sagas are where most production systems land, and they're where most agent systems should land too.

This module covers:

- **Section 1** — why agent action failures are different from agent reasoning failures
- **Section 2** — the saga pattern, classical version. Local transactions + compensating transactions + eventual consistency
- **Section 3** — orchestration vs choreography. The two coordination patterns and which fits agent systems
- **Section 4** — idempotency. The non-negotiable foundation
- **Section 5** — compensating transactions in agent contexts. Why "rollback" is the wrong mental model and what to use instead
- **Section 6** — durable state and checkpointing. Anthropic's Managed Agents pattern and the broader category
- **Section 7** — the saga pattern trap: compensation failures, the saga of sagas problem, and human escalation as the ultimate compensation
- **Section 8** — Sam's customer-success bot redesign with proper sagas
- **Section 9** — the framework

By the end of this module:

- You'll know which agent actions need saga discipline and which don't
- You'll have working code for orchestrated agent sagas with proper compensation
- You'll understand idempotency at a level deep enough to design it correctly the first time
- You'll know when to reach for production saga frameworks (Temporal, Step Functions, Anthropic Managed Agents) vs hand-rolling

This is the operational discipline that turns agents from "they work in dev" into "they work at 3 a.m. when something flakes." Module 17 will cover observability — how you see what happened. Module 16 is about ensuring that when something goes wrong, the system is in a recoverable state, not a broken one.

::: pullquote
The agent loop guarantees the *reasoning* completes or doesn't. It says nothing about the *actions* the reasoning takes. Treating multi-step actions as single logical operations is the most common reliability bug in production agent systems — and the one teams discover only after a customer notices.
:::

---

## Section 1 — Why Agent Action Failures Are Different

Three structural reasons agents need saga discipline that pure-text agents don't.

**1. Real-world side effects are durable.** When the agent calls `apply_credit(500)`, the credit is real. It exists in the billing system whether or not the rest of the trajectory completes. If the agent then crashes, the credit doesn't un-apply. Compare to a pure-reasoning agent — when it crashes mid-thought, nothing in the world has changed.

**2. Partial state isn't recoverable from logs.** When step 1 succeeds and step 3 fails, the system has a record of step 3's failure but no automatic plan for step 1's outcome. Without explicit saga design, the recovery path is "a human reads the logs and figures out what to do." For Sam's customer-success bot, that human was the CS rep digging through ticket history the next morning.

**3. Agent non-determinism amplifies the problem.** A microservice that retries a failed call sends the same request. An LLM agent that retries a failed step might phrase the request slightly differently — different parameters, different framing, different intent. Without idempotency keys, the retry can create *additional* state changes rather than completing the original one. Sam's bot retrying `apply_credit` without an idempotency key could have applied $1,500 instead of the intended $500.

The combination is what makes agent action failures distinctive. Real consequences + partial state + non-deterministic retries = a failure mode pure software doesn't have in the same shape. The saga pattern was designed for distributed systems facing #1 and #2; agent systems face all three, which means the disciplines have to be even more careful.

### What this module is not about

Saga is a distributed-systems pattern. This module covers it in the agent-specific context. We won't deep-dive:

- **The full distributed systems theory** — CAP, BASE vs ACID, the saga math. The references in the recap point to this; production teams should read at least one classical saga paper before deploying.
- **Specific framework tutorials** — Temporal, Step Functions, Camunda all have excellent docs. We'll mention them; you'll learn them from their docs.
- **Two-phase commit and pure ACID alternatives** — these rarely fit agent systems' interaction patterns.

What we will cover: the patterns specific to agent contexts, the mistakes specific to agent contexts, and the operational discipline specific to agent contexts. Sam's customer-success bot is the running case study because its failure mode is the canonical one.

::: nodumbq
**Q: Doesn't M12's confirmation gate handle this? The agent confirms before taking action.**

Confirmation gates handle the "should this action happen?" question. Sagas handle the "now that the action sequence is in flight, what happens if part of it fails?" question. Different concerns. Confirmation prevents bad actions from starting; sagas handle bad outcomes when correct actions encounter system failures. Both are needed; neither replaces the other.

**Q: Can I just make every multi-step action a single API call to a service that handles the saga internally?**

Sometimes yes — if your backend services already implement saga discipline, your agent can treat the composite operation as atomic from its perspective. But (a) most service backends don't have saga implementations, (b) agent flows often combine *multiple* services (your billing service, your email service, your CRM) where no single backend can wrap them, and (c) this just pushes the saga problem one layer down. Eventually somebody is implementing it. This module is for the cases where it's you.
:::

---

## Section 2 — The Saga Pattern, Classical Version

The classical saga pattern from Garcia-Molina and Salem's 1987 paper, refined through decades of distributed systems practice. The core mechanics:

**A saga is a sequence of local transactions, each with a compensating transaction, that together produce eventual consistency.**

Three properties make it work:

**1. Each step is a local transaction.** Step N either commits or aborts atomically within its service. There's no partial state within a single step — credit is either applied or not, billing is either updated or not. The saga's eventual consistency rests on each step's local atomicity.

**2. Each forward step has a compensating transaction.** If a later step fails, the orchestrator (or event chain) triggers compensations in reverse order. Compensation undoes the *business effect* of the forward step, not necessarily the database write itself.

**3. The saga commits when all forward steps succeed, or aborts when any step fails and all prior compensations succeed.**

```
Forward path:     Compensation path:
──────────────    ───────────────────
Step 1: apply_credit       ←  Compensation: reverse_credit
Step 2: update_billing     ←  Compensation: revert_billing_record
Step 3: send_email         ←  Compensation: send_correction_email
                              (no email to "unsend"; send a correction)
Step 4: schedule_followup  ←  Compensation: cancel_scheduled_followup
```

The ColdFusion blogpost on saga traps named the central insight: *"In a Distributed Saga, ACID guarantees are traded for BASE (Basically Available, Soft state, Eventual consistency)."*

You give up:
- **Atomicity** across the whole saga (only within steps)
- **Isolation** (intermediate states are visible to other observers)

You keep:
- **Atomicity within each step** (each local transaction commits or aborts)
- **Eventual consistency** (the saga drives toward a consistent business outcome)

For agent systems, this trade is almost always the right one. Pure ACID requires distributed transactions which require coordinated infrastructure your services often don't have. Sagas work with services as they are.

### A minimal saga implementation

```python
from typing import Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum


class StepState(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATED = "compensated"


@dataclass
class SagaStep:
    name: str
    forward: Callable[[dict], Awaitable[dict]]
    compensate: Callable[[dict], Awaitable[None]]
    state: StepState = StepState.PENDING
    forward_result: dict | None = None


@dataclass
class Saga:
    saga_id: str
    steps: list[SagaStep]
    context: dict = field(default_factory=dict)


async def execute_saga(saga: Saga) -> dict:
    """
    Execute a saga: run forward steps; on any failure, compensate in reverse.
    Returns the final context (with results from all completed steps).
    """
    completed: list[SagaStep] = []
    try:
        for step in saga.steps:
            step.forward_result = await step.forward(saga.context)
            saga.context[step.name] = step.forward_result
            step.state = StepState.COMPLETED
            completed.append(step)
        return {"status": "success", "context": saga.context}
    except Exception as e:
        # Compensate completed steps in reverse order
        for step in reversed(completed):
            try:
                await step.compensate(saga.context)
                step.state = StepState.COMPENSATED
            except Exception as comp_error:
                # Compensation failure: this is the saga trap (Section 7)
                step.state = StepState.FAILED
                # Critical: log and escalate; don't continue blindly
                await escalate_compensation_failure(saga, step, comp_error)
                raise SagaFailureError(saga, original_error=e, compensation_error=comp_error)
        return {"status": "compensated", "context": saga.context, "error": str(e)}
```

Three things to notice:

**Compensations run in reverse.** If step 3 fails, we compensate step 2 first, then step 1. This is the standard saga ordering — undo the most recent change first.

**Failed compensations are special.** They're not just regular errors. A failed compensation means the system is now in a state nobody expected and nobody can automatically recover from. Section 7 covers this in depth.

**The context is shared across steps.** Each step's result becomes part of the context the next step reads. This is convenient but it's also a coupling point: if step 1's output schema changes, all later steps that depend on it have to update.

::: brain
Sam's customer-success bot from the cold open. What's the saga structure for the credit-application action?

(Step 1: `apply_credit($500, customer=2847, idempotency_key=...)`. Compensation: `reverse_credit(credit_id, reason="saga_compensation")`. Step 2: `update_billing_record(...)`. Compensation: `revert_billing_record(...)`. Step 3: `send_confirmation_email(...)`. Compensation: `send_correction_email(reason="previous email may not have arrived; here's the actual state")` — note that the compensation is *not* "unsend the email" because that's impossible. It's a new corrective action. Step 4: `schedule_followup(...)`. Compensation: `cancel_scheduled_followup(...)`. The whole sequence is the saga. If any step fails, the prior steps' compensations run.)
:::

---

## Section 3 — Orchestration vs Choreography

Two ways to coordinate the steps. The choice matters.

**Orchestration** — a central coordinator (the orchestrator) explicitly drives each step in sequence. The orchestrator knows the saga's structure, calls services in order, handles failures, and triggers compensations. Most production systems use this.

**Choreography** — services emit events when their step completes; other services subscribe to relevant events and execute their own step in response. No central coordinator. The saga's structure is implicit in the event flows.

For agent systems, **orchestration is almost always the right call**. Three reasons:

**1. The agent's reasoning is the orchestrator.** When the agent decides "I'm going to apply a credit and notify the customer," it's already taking on the orchestrator's role. Adding choreography on top means the agent kicks off events and then waits for downstream events; the agent's role gets fragmented across multiple turns and event handlers. Orchestration keeps the saga's logical control where the agent's reasoning naturally is.

**2. Compensations are explicit.** Choreography depends on services *deciding* to emit compensation events when others fail. In practice this is fragile — services often don't know about all the downstream sagas they participate in, and getting compensation event flows right across organizations is an ongoing operational tax. Orchestration centralizes compensation logic.

**3. Debugging is tractable.** When an orchestrated saga fails, the orchestrator's logs tell you exactly what happened. When a choreographed saga fails, you're piecing together event traces from N services to figure out which event was missed or arrived in the wrong order. The kalviumlabs framing from M15 applies here — multi-agent debugging is hard enough; making it event-driven adds a layer of indirection nobody asked for.

The single argument for choreography in agent contexts: when the agent's role is to *initiate* an action and the rest unfolds across an organization's existing event infrastructure. *"Customer signs up; downstream services handle onboarding via events that already exist."* In that case, the agent's role isn't to orchestrate; it's to start the chain. The chain itself is choreography that already exists. The agent participates without owning.

### The Temporal pattern

Most production systems that orchestrate sagas use a workflow engine — Temporal is the most common in 2026, with AWS Step Functions and Camunda also widely used. The reason: hand-rolled orchestration usually doesn't survive long-running tasks, retries, persistence requirements, and observability needs. Workflow engines handle the durability, the retry logic, the timeout handling, the state persistence.

The Ajit Singh post is direct about this: *"In production, most teams settle on orchestration with Temporal, AWS Step Functions, Camunda, or Eventuate Tram. The hand-rolled choreography saga rarely survives contact with a real on-call rotation."*

A simplified Temporal-style saga in Python:

```python
from temporalio import workflow
from temporalio.exceptions import ActivityError


@workflow.defn
class CreditApplicationSaga:
    @workflow.run
    async def run(self, params: CreditApplicationParams) -> SagaResult:
        applied_credit_id = None
        billing_update_id = None
        email_id = None
        followup_id = None

        try:
            # Step 1: Apply credit
            applied_credit_id = await workflow.execute_activity(
                apply_credit_activity,
                params,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=1),
                ),
            )

            # Step 2: Update billing
            billing_update_id = await workflow.execute_activity(
                update_billing_activity,
                params,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            # Step 3: Send confirmation email
            email_id = await workflow.execute_activity(
                send_email_activity,
                params,
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(maximum_attempts=5),
            )

            # Step 4: Schedule follow-up
            followup_id = await workflow.execute_activity(
                schedule_followup_activity,
                params,
                start_to_close_timeout=timedelta(seconds=30),
            )

            return SagaResult(status="success", ...)

        except ActivityError as e:
            # Compensate in reverse order
            if email_id:
                # No "unsend" — send correction instead
                await workflow.execute_activity(
                    send_correction_email_activity,
                    params, email_id,
                )
            if billing_update_id:
                await workflow.execute_activity(
                    revert_billing_activity,
                    billing_update_id,
                )
            if applied_credit_id:
                await workflow.execute_activity(
                    reverse_credit_activity,
                    applied_credit_id,
                )
            return SagaResult(status="compensated", error=str(e), ...)
```

What Temporal gives you that hand-rolled doesn't:

- **Durability of workflow state.** If your worker crashes mid-saga, Temporal resumes from where it left off when a worker comes back online. The hand-rolled version loses state on crash.
- **Automatic retry policies.** Configurable backoff, max attempts, retry-only-on-specific-errors.
- **Heartbeating for long-running activities.** A task that should take 30 seconds but is taking 10 minutes can be cancelled and retried.
- **Workflow versioning.** When you update the saga code, in-flight workflows continue with their original logic; new workflows use the new code. Critical for production migrations.
- **Observable execution.** Every activity attempt, every retry, every compensation is logged with structured data.

For agent systems with serious operational requirements, Temporal-or-equivalent is the production answer. For prototypes and small-scale systems, hand-rolled with the patterns from Section 2 is acceptable. The maturity ladder from CloudOpsNow's saga guide (Beginner → Intermediate → Advanced) corresponds roughly to "hand-rolled with retries → workflow engine with monitoring → autonomous compensation with observability."

::: pullquote
For agent systems with real-world actions, orchestration with a workflow engine isn't optional once you cross a certain volume threshold. The hand-rolled version works in development and dies in production. Temporal, Step Functions, and Camunda exist because every team that built one of these without them eventually built one.
:::

---

## Section 4 — Idempotency

The non-negotiable foundation. Without idempotency, sagas can't retry; without retry, sagas can't survive the transient failures that are the whole reason sagas exist.

**An operation is idempotent if applying it once produces the same result as applying it N times.**

For Sam's `apply_credit`:

- **Non-idempotent:** Each call adds $500 to the customer's account. Three retries → $1,500 applied.
- **Idempotent (correct):** Each call with the same idempotency key applies $500 once. Three retries → $500 applied.

The key insight: idempotency isn't about the operation's *intent*; it's about its *effect*. The intent is "apply $500 credit"; the effect under retry must be "apply $500 once total, regardless of how many times you call."

### Idempotency keys

The standard pattern: every action gets a unique idempotency key. The receiving service checks if it's already processed that key; if so, returns the original result without re-executing. If not, executes and records the key.

```python
from uuid import uuid4
from hashlib import sha256


def generate_idempotency_key(saga_id: str, step_name: str, params: dict) -> str:
    """
    Generate a deterministic idempotency key for a saga step.
    Same saga + same step + same params → same key.
    """
    payload = f"{saga_id}:{step_name}:{json.dumps(params, sort_keys=True)}"
    return sha256(payload.encode()).hexdigest()


# In the activity (the actual operation):
async def apply_credit_activity(params: CreditParams) -> str:
    """Apply a credit to the customer's account; idempotent."""
    idempotency_key = generate_idempotency_key(
        saga_id=params.saga_id,
        step_name="apply_credit",
        params={"customer": params.customer_id, "amount": params.amount},
    )

    # Check if we've already processed this key
    existing = await idempotency_store.get(idempotency_key)
    if existing:
        return existing.result_id  # Return the original result

    # Acquire a lock on the key to prevent concurrent executions
    async with idempotency_store.lock(idempotency_key):
        # Double-check inside the lock
        existing = await idempotency_store.get(idempotency_key)
        if existing:
            return existing.result_id

        # Execute the actual operation
        credit_id = await billing_api.apply_credit(
            customer=params.customer_id,
            amount=params.amount,
        )

        # Record the result
        await idempotency_store.set(
            idempotency_key,
            IdempotencyRecord(
                key=idempotency_key,
                result_id=credit_id,
                created_at=datetime.utcnow(),
            ),
            ttl=timedelta(days=30),  # Keys must outlive the saga's possible retry window
        )

        return credit_id
```

Three properties of this implementation:

**Deterministic key generation.** Same saga + same step + same params → same key. This means retries within the saga produce the same key; the second attempt sees the first attempt's record and returns the original result.

**Lock around the check-and-execute.** Without the lock, two concurrent retries could both see "key not found" and both execute. The lock prevents the race.

**TTL on the records.** Idempotency records can't live forever (storage cost). They must outlive the saga's possible retry window — typically 30 days for human-driven workflows, hours for automated ones.

### What "same params" means

A subtle point. Two retries of `apply_credit` should produce the same key. But agents are non-deterministic; what if the agent retries with slightly different parameters?

```
Attempt 1: apply_credit(customer=2847, amount=500, reason="billing dispute - customer A")
Attempt 2: apply_credit(customer=2847, amount=500, reason="billing dispute - customer A (retry)")
```

If `reason` is part of the idempotency key calculation, these are different operations and the saga applies $1,000 instead of $500. If `reason` is excluded from the key, both attempts have the same key and only $500 is applied — correct, but the reason from attempt 1 (without the "retry" suffix) is what's recorded.

The discipline: **idempotency keys should be calculated from the operation's logical identity, not its full payload.** For `apply_credit`, the logical identity is `(customer, amount, source)` where source might be `"saga:S123:step:apply_credit"`. Cosmetic fields (reasons, descriptions, retry markers) are excluded from the key calculation.

For agent systems, this means the orchestrator generates the idempotency key from saga ID and step name (deterministic across retries), and the agent's reasoning *cannot* generate idempotency keys — that would defeat the purpose.

### Compensation idempotency

Often forgotten: compensations must also be idempotent. If a compensation fails partway through and gets retried, the retry must not double-apply the compensation.

```python
async def reverse_credit_activity(credit_id: str, saga_id: str) -> None:
    """Reverse a previously-applied credit; idempotent."""
    idempotency_key = generate_idempotency_key(
        saga_id=saga_id,
        step_name="reverse_credit",
        params={"credit_id": credit_id},
    )

    existing = await idempotency_store.get(idempotency_key)
    if existing:
        return  # Already reversed; safe to ignore the retry

    async with idempotency_store.lock(idempotency_key):
        existing = await idempotency_store.get(idempotency_key)
        if existing:
            return

        # Reverse the credit
        await billing_api.reverse_credit(credit_id)

        # Record the reversal
        await idempotency_store.set(
            idempotency_key,
            IdempotencyRecord(key=idempotency_key, result_id=credit_id),
            ttl=timedelta(days=30),
        )
```

Forward steps and compensations both need idempotency. Forgetting compensation idempotency is a common production bug — the system handles forward retries correctly but breaks during compensation retries.

::: gotcha
A subtle idempotency anti-pattern: using a timestamp in the idempotency key. *"key = `f"apply_credit:{customer_id}:{timestamp}"`."* The first attempt at 23:14:22 generates one key; the retry at 23:14:25 generates a different key; both apply $500. The "idempotency" is broken because the key isn't actually deterministic across retries. Use the saga ID and step name instead — those are stable.
:::

---

## Section 5 — Compensating Transactions in Agent Contexts

The classical saga literature talks about "compensating transactions" as if they're just rollbacks. They're not. Singh's post made the point sharply:

> *"Compensating transactions are not rollbacks. They are real business operations like refunds, restocks, and apology emails, written in normal code that can fail and must be retried."*

For agent systems specifically, this matters more than usual. Several common compensations have no clean rollback equivalent:

**Email send → ?** You can't unsend an email. Compensation: send a correction email explaining the situation.

**SMS sent → ?** Same as email. Compensation: send a correction SMS or escalate to human.

**External API call → ?** Depends on the API. Compensation: a separate "undo" API call if available; otherwise, escalate.

**Slack post → ?** Either delete the post (if recently sent) or follow up with a correction.

**Calendar event created → ?** Cancel the event with a notification.

The pattern: agent compensations are usually **forward-only operations that achieve the business goal of "undoing" the original action**, not literal reversals. A "rollback" mental model leads to disappointment; a "what would the customer experience as a clean correction" mental model leads to good design.

### The compensation design checklist

For every forward action your agent takes, design the compensation explicitly:

```python
@dataclass
class ActionDefinition:
    name: str
    forward_operation: str  # what it does
    compensation_strategy: Literal[
        "reverse_state_change",  # actual rollback (e.g., reverse_credit)
        "corrective_action",      # forward-only fix (e.g., send correction email)
        "escalate_to_human",       # no automated compensation possible
        "best_effort_then_escalate",  # try corrective action; escalate if it fails
    ]
    compensation_details: str  # specific implementation
    irreversibility: Literal["fully_reversible", "partially_reversible", "irreversible"]


CUSTOMER_SUCCESS_ACTIONS = [
    ActionDefinition(
        name="apply_credit",
        forward_operation="Add credit to customer's billing account",
        compensation_strategy="reverse_state_change",
        compensation_details="Call billing_api.reverse_credit(credit_id)",
        irreversibility="fully_reversible",
    ),
    ActionDefinition(
        name="update_billing_record",
        forward_operation="Modify customer's billing record (e.g., notes, status)",
        compensation_strategy="reverse_state_change",
        compensation_details="Restore original record from saga's saved snapshot",
        irreversibility="fully_reversible",
    ),
    ActionDefinition(
        name="send_confirmation_email",
        forward_operation="Send email confirming an action taken",
        compensation_strategy="corrective_action",
        compensation_details=(
            "Send correction email with subject 'Update on your previous request' "
            "explaining the actual current state of their account"
        ),
        irreversibility="irreversible",
    ),
    ActionDefinition(
        name="post_to_internal_slack",
        forward_operation="Post a notification to internal team channel",
        compensation_strategy="best_effort_then_escalate",
        compensation_details=(
            "Edit/delete original message if within edit window; "
            "post correction if not; flag a human if both fail"
        ),
        irreversibility="partially_reversible",
    ),
    ActionDefinition(
        name="schedule_followup",
        forward_operation="Create a scheduled followup task",
        compensation_strategy="reverse_state_change",
        compensation_details="Call followup_api.cancel(task_id)",
        irreversibility="fully_reversible",
    ),
]
```

Building this catalog for your agent's actions is a one-time design exercise, but it pays off for the lifetime of the agent. Without it, compensations get designed reactively when the first failure happens — which is also when the team is least equipped to design them well.

### When there is no compensation

Some actions genuinely cannot be undone. The classic example: a wire transfer to an external bank. Once the funds leave your control, no compensation reverses them.

For these actions, the saga discipline is different:

**1. Make them the *last* step.** If the irreversible action is step 5 of 5, then any failure before step 5 leaves the irreversible action not-yet-taken. The sage compensates the reversible steps and never reaches the irreversible one.

**2. Add aggressive pre-validation.** Since you can't undo, you must be sure before you act. Confirmation gates (M12), double-checks, dry-runs, human-in-the-loop approval — all justified for irreversible steps.

**3. When they fail, escalate to human immediately.** If the irreversible action itself fails, the system can't compensate the action and can't simply retry without potentially executing the action twice. Human judgment is the only safe path.

```python
@dataclass
class IrreversibleStep:
    """A step that can't be compensated; must be designed differently."""
    require_explicit_human_approval: bool = True
    require_dry_run_first: bool = True
    place_last_in_saga: bool = True
    on_failure: Literal["escalate_to_human"] = "escalate_to_human"


WIRE_TRANSFER_STEP = IrreversibleStep(
    require_explicit_human_approval=True,
    require_dry_run_first=True,
    place_last_in_saga=True,
    on_failure="escalate_to_human",
)
```

The honest framing: irreversible actions in agent systems should be rare. If your agent has many irreversible operations, the agent's design probably needs revision. Most "irreversible" actions can be designed to be reversible — *make a wire transfer to an internal escrow account, hold for 24 hours, release after confirmation* converts irreversible to "reversible within 24 hours." The design effort to add that reversibility is almost always worth it.

::: brain
A team's agent has these actions: send_email, apply_credit, update_billing, transfer_funds_to_customer_external_bank, post_to_slack, schedule_followup. Which of these are irreversible? How should the saga design accommodate them?

(transfer_funds_to_customer_external_bank is irreversible. The others are either reversible (apply_credit, update_billing, schedule_followup), partially reversible (post_to_slack — edit/delete window), or compensable via corrective action (send_email — send correction). The saga design: place the wire transfer as the last step; require explicit human approval before reaching it; dry-run validation before execution; on failure, escalate to human and don't retry. The other steps run normally with standard compensations.)
:::

---

## Section 6 — Durable State and Checkpointing

A saga is only as reliable as its persistence. If your worker crashes mid-saga and restarts, the saga must be able to resume from where it left off — not start over (which could double-apply effects) and not abandon (which leaves the system in a partial state).

The discipline: **durable state at every saga boundary.**

### What durable state needs to cover

Three things:

**1. The saga's current step.** Where in the sequence are we?

**2. Each completed step's result.** Including the idempotency key and the result ID, so retries see the original result.

**3. Pending compensations.** If the saga is in the compensation path, which compensations have been attempted and which still need to run.

```python
@dataclass
class SagaState:
    saga_id: str
    saga_type: str  # e.g., "customer_credit_application"
    status: Literal["running", "compensating", "completed", "compensated", "failed"]
    current_step: int
    step_results: dict[str, dict]  # step_name -> {"result": ..., "idempotency_key": ...}
    compensations_pending: list[str]  # step_names whose compensations need to run
    compensations_completed: list[str]
    created_at: datetime
    updated_at: datetime
    context: dict  # the saga's input parameters and accumulated state
```

This state must be persisted to durable storage (database, key-value store) at every transition. Workflow engines like Temporal handle this automatically; hand-rolled sagas need explicit persistence.

```python
async def execute_saga_with_persistence(
    saga: Saga,
    state_store: SagaStateStore,
) -> dict:
    """
    Execute a saga with durable state at every boundary.
    Resumable if the worker crashes.
    """
    # Try to load existing state (resumption case)
    state = await state_store.get(saga.saga_id)
    if state is None:
        # Fresh start
        state = SagaState(
            saga_id=saga.saga_id,
            saga_type=saga.saga_type,
            status="running",
            current_step=0,
            step_results={},
            compensations_pending=[],
            compensations_completed=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            context=saga.context,
        )
        await state_store.put(state)

    # Resume from current_step
    if state.status == "running":
        for i in range(state.current_step, len(saga.steps)):
            step = saga.steps[i]
            try:
                # Skip if already completed (idempotent execution)
                if step.name in state.step_results:
                    continue

                # Execute the step
                result = await step.forward(state.context)
                state.step_results[step.name] = {
                    "result": result,
                    "idempotency_key": step.idempotency_key,
                }
                state.context[step.name] = result
                state.current_step = i + 1
                state.updated_at = datetime.utcnow()

                # Persist state after each step
                await state_store.put(state)

            except Exception as e:
                # Move to compensation path
                state.status = "compensating"
                state.compensations_pending = list(reversed([
                    name for name in state.step_results.keys()
                ]))
                await state_store.put(state)
                break

    # Compensation path
    if state.status == "compensating":
        for step_name in list(state.compensations_pending):
            step = next(s for s in saga.steps if s.name == step_name)
            try:
                await step.compensate(state.context)
                state.compensations_completed.append(step_name)
                state.compensations_pending.remove(step_name)
                state.updated_at = datetime.utcnow()
                await state_store.put(state)
            except Exception as comp_error:
                # Compensation failure - the saga trap (Section 7)
                state.status = "failed"
                state.updated_at = datetime.utcnow()
                await state_store.put(state)
                await escalate_compensation_failure(state, step, comp_error)
                raise

        state.status = "compensated"
        state.updated_at = datetime.utcnow()
        await state_store.put(state)

    return {"status": state.status, "context": state.context}
```

The implementation is verbose because durability is. Every transition is `state.something_changed → await state_store.put(state)`. Without this, a crash between transitions loses progress.

### Anthropic's Managed Agents pattern

Anthropic's Managed Agents (April 2026 GA) handles much of this for you. From the Momentic writeup:

> *"Long-running tasks that need to survive crashes: Managed Agents. Sessions can run for hours, with checkpointing that resumes them if something dies."*

And from Anthropic's release notes:

> *"Anthropic introduces Managed Agents, a hosted Claude Platform service for long-horizon agent work with stable interfaces for sessions, harnesses, and sandboxes. It emphasizes durable state, safer tool access, and faster startup for reliable long-running tasks."*

Managed Agents provides:

- **Session state persistence.** The agent's full state is checkpointed; if a worker crashes, a new worker resumes from the checkpoint.
- **Tool execution sandboxing.** Tool calls run in isolated environments that survive worker restarts.
- **Built-in retry with exponential backoff.** Transient failures are handled automatically.
- **Observable execution.** Every checkpoint, every tool call, every retry is logged with structured data.

For agent systems with serious operational requirements, this is the production answer for individual agent durability. **It does not, however, automatically give you saga semantics across multiple agent actions.** The agent's *reasoning* is durable; its *actions on the world* still need explicit saga design.

The relationship: Managed Agents provides the harness durability (the agent's loop survives crashes); your saga design provides the action durability (the agent's effects on external systems are recoverable). Both are needed.

### Anthropic's Agent SDK file checkpointing

For local development and self-hosted deployments, Anthropic's Agent SDK ships file checkpointing. From the SDK docs:

> *"When file checkpointing is enabled, the SDK creates backups of files before they are modified. This allows programmatic restoration to any previous state in the conversation."*

And:

```python
# Rewind files to previous checkpoint
await client.rewind_files(checkpoint_id)
```

For coding agents specifically, this is a saga-like pattern at the file system level. If a sequence of file modifications fails partway through, `rewind_files(checkpoint_id)` restores the pre-saga state. The "compensation" is automatic; the agent doesn't need explicit compensation logic for file modifications.

This works because file modifications are uniquely amenable to checkpointing — the file system is local, the changes are visible, the rollback is mechanical. The same pattern doesn't generalize to external API calls (you can't checkpoint Stripe), but for agents that primarily modify files (Claude Code, code-modification agents), it captures most of the saga value with much less design effort.

::: nodumbq
**Q: Should I use Managed Agents or the Agent SDK for my system?**

It depends on the trade-off. Managed Agents handles infrastructure (sandboxing, durability, retry, observability) but you give up some control. Agent SDK gives you full control but you build the infrastructure. For long-running agents with real-world actions, Managed Agents is increasingly the right answer — the saga discipline you'd build hand-rolled is mostly the saga discipline Anthropic has already built. For prototypes, sensitive-data scenarios, or tightly controlled environments, Agent SDK gives you the flexibility you need. The choice isn't permanent; many teams prototype on Agent SDK and migrate specific workflows to Managed Agents as they mature.

**Q: What about Temporal vs Managed Agents?**

Temporal is workflow engine for general-purpose distributed sagas. Managed Agents is a hosted agent harness with built-in durability. They overlap in some places (both handle long-running execution with checkpointing) but the abstractions are different. Temporal: you write workflow code; it manages execution. Managed Agents: you define an agent and tools; the agent's reasoning manages execution. For pure agent workflows, Managed Agents is more natural. For workflows that combine LLM agents with traditional services in a heavyweight orchestration, Temporal still fits. Many production systems use both — Temporal for the cross-service saga, Managed Agents for the agent-driven steps within it.
:::

---

## Section 7 — The Saga Pattern Trap: Compensation Failures

The most insidious failure mode in saga design. The ColdFusion blogpost named it directly:

> *"You have implemented the Saga pattern (likely Choreography) to manage distributed transactions across your Order, Inventory, and Payment microservices. The standard rollback mechanism relies on Compensating Transactions. If Service A succeeds, Service B succeeds, but Service C fails, the Saga orchestrator (or event chain) commands Service B to undo its work. The trap lies in how we handle exceptions during that undo."*

Compensation can fail. When it does, the system is in a state nobody designed for.

### The failure modes

**Transient compensation failure.** The compensation should work but the underlying service is temporarily unavailable. The retry will likely succeed; the right discipline is bounded retry with exponential backoff.

**Permanent compensation failure.** The compensation cannot succeed regardless of retry. The underlying state is now permanently inconsistent. Examples:
- The `reverse_credit` API rejects the request because the credit has already been applied to a closed billing cycle
- The `cancel_followup` API can't cancel because the followup has already been started
- The `revert_billing_record` operation fails because the record was modified by another process

**The "saga of sagas" failure.** A compensation triggers its own saga that can also fail. If your `reverse_credit` compensation involves notifying accounting (which is itself a multi-step operation), and that notification fails partway through, you now have nested saga failures.

### The defenses

**1. Bounded retry with categorical handling.** Try the compensation N times with backoff. After N failures, escalate. Don't retry indefinitely.

```python
async def compensate_with_bounded_retry(
    step: SagaStep,
    context: dict,
    *,
    max_retries: int = 5,
    base_delay_seconds: float = 1.0,
) -> CompensationResult:
    last_error = None
    for attempt in range(max_retries):
        try:
            await step.compensate(context)
            return CompensationResult(success=True, attempts=attempt + 1)
        except RetryableCompensationError as e:
            last_error = e
            await asyncio.sleep(base_delay_seconds * (2 ** attempt))
        except PermanentCompensationError as e:
            # Don't retry; escalate immediately
            return CompensationResult(
                success=False,
                attempts=attempt + 1,
                error_type="permanent",
                error=str(e),
            )

    # Exceeded retry limit
    return CompensationResult(
        success=False,
        attempts=max_retries,
        error_type="exhausted_retries",
        error=str(last_error),
    )
```

**2. Categorical error handling.** Services should distinguish *transient* from *permanent* failures explicitly. The saga's retry logic respects the distinction.

**3. Human escalation as the terminal compensation.** When automated compensation fails, escalate to a human with full context. This is the saga's universal fallback.

```python
async def escalate_compensation_failure(
    saga_state: SagaState,
    failed_step: SagaStep,
    error: Exception,
) -> None:
    """
    The terminal compensation. When automated paths have all failed,
    a human gets a structured report and decides what to do.
    """
    escalation = CompensationEscalation(
        saga_id=saga_state.saga_id,
        saga_type=saga_state.saga_type,
        failed_compensation_step=failed_step.name,
        completed_steps=[
            s for s in saga_state.step_results
        ],
        completed_compensations=saga_state.compensations_completed,
        pending_compensations=saga_state.compensations_pending,
        error_summary=str(error),
        full_state_dump=saga_state,
        suggested_actions=_suggest_recovery_actions(saga_state, failed_step),
        urgency=_calculate_urgency(saga_state),
    )

    # Send to on-call queue
    await escalation_queue.send(escalation)

    # Log structured event
    await event_log.write(
        event_type="compensation_escalation",
        saga_id=saga_state.saga_id,
        details=escalation,
    )
```

The `suggested_actions` field matters. The on-call human reading this at 3 a.m. shouldn't have to figure out the saga's structure — they need a structured report saying "step 1 succeeded ($500 credit applied to customer 2847), step 2 succeeded (billing record updated), step 3 failed (email send), compensation 1 failed (reverse_credit API permanent error). Manual actions needed: (a) verify the credit is actually applied, (b) decide whether to keep the credit or manually reverse via accounting workflow, (c) send manual customer communication."

Without that structure, the saga's failure becomes the human's mystery to solve. With it, the saga's failure becomes the human's checklist.

::: postmortem
**The Compensation That Caused More Damage Than the Original Failure**

A team had a multi-step billing saga that included an email confirmation as step 4. When step 4 (send_confirmation) failed transiently, the team's saga compensated steps 1-3 (which had succeeded), including a `reverse_credit` operation in step 1's compensation.

Then the email service recovered. The original step 4 hadn't actually failed — it had timed out at the gateway but successfully sent the email. The customer received an email saying their credit was applied. Then ten seconds later, the saga's compensation reversed the credit, leaving the customer with: an email saying they had a credit, and no actual credit.

Three lessons:

1. **The timeout response was ambiguous.** The saga treated "timed out" as "definitely failed" when it was really "outcome unknown." The fix: distinguish timeouts from confirmed failures, and for ambiguous responses, attempt a status query before assuming failure.

2. **The compensation ran without verifying the original failure was real.** Compensations should verify the forward step's effect *before* compensating. If the credit can be queried and is in the expected state, the saga can proceed without compensation.

3. **The compensation itself notified the customer** — there should have been a final reconciliation step that compared "what we told the customer" vs "what we actually did" and surfaced any inconsistency to a human. Without that, the customer's email and account state diverged silently.

After the fix: the saga distinguishes "step 4 timed out" from "step 4 explicitly failed"; on timeout, runs a status check on the email service; if the email actually sent, treats the step as succeeded. The on-customer email was sent at the start of the saga rather than at the end, and the compensation includes "send correction email if confirmation was already sent."

**Lesson:** compensations are *new operations* with their own potential failures. Designing them as if they're free reversals leads to compounding failures. They need the same engineering rigor as forward operations — including idempotency, retry, escalation, and reconciliation with what the user has already been told.
:::

---

## Section 8 — Sam's Customer-Success Bot Redesign

Returning to the cold open. Sam rebuilt the customer-success bot's action handling using everything in this module:

### The architecture decisions

**1. All multi-step actions become explicit sagas.** The previous "agent calls tools in sequence" pattern is replaced with "agent invokes a saga which calls tools in sequence." The agent's reasoning still drives the work, but the orchestration of effects is structured.

**2. Use Anthropic Managed Agents for harness durability.** The agent's loop runs on Managed Agents infrastructure. If the worker crashes, the loop resumes. Reasoning is durable.

**3. Use Temporal for cross-service saga orchestration.** When the agent decides to apply a credit, it invokes a Temporal workflow that orchestrates the steps with proper compensation, idempotency, and durability.

**4. Idempotency keys derived from saga ID.** Every action gets a deterministic idempotency key. Retries are safe.

**5. Compensation strategy per action.** The action catalog (Section 5) defines compensation for every operation the agent can perform.

**6. Human escalation as terminal compensation.** Failures that exhaust automated recovery escalate to a structured on-call queue with full context.

### The saga itself

```python
from temporalio import workflow
from datetime import timedelta


@workflow.defn
class CustomerCreditSaga:
    """
    Saga for applying a credit to a customer's account with notification.
    Invoked by the customer-success agent when it decides a credit is warranted.
    """

    @workflow.run
    async def run(self, params: CreditApplicationParams) -> SagaResult:
        # Snapshot the customer's billing state for potential rollback
        billing_snapshot = await workflow.execute_activity(
            snapshot_billing_state,
            params.customer_id,
            start_to_close_timeout=timedelta(seconds=10),
        )

        # State tracking
        applied_credit_id: str | None = None
        billing_update_id: str | None = None
        confirmation_email_id: str | None = None
        followup_id: str | None = None

        try:
            # Step 1: Apply credit (reversible)
            applied_credit_id = await workflow.execute_activity(
                apply_credit_activity,
                params,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=1),
                    backoff_coefficient=2.0,
                ),
            )

            # Step 2: Update billing record (reversible via snapshot)
            billing_update_id = await workflow.execute_activity(
                update_billing_record_activity,
                params,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            # Step 3: Send confirmation email (irreversible — corrective only)
            confirmation_email_id = await workflow.execute_activity(
                send_confirmation_email_activity,
                params,
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(
                    maximum_attempts=5,
                    initial_interval=timedelta(seconds=2),
                ),
            )

            # Step 4: Schedule followup (reversible)
            followup_id = await workflow.execute_activity(
                schedule_followup_activity,
                params,
                start_to_close_timeout=timedelta(seconds=30),
            )

            return SagaResult(
                status="success",
                credit_id=applied_credit_id,
                billing_update_id=billing_update_id,
                email_id=confirmation_email_id,
                followup_id=followup_id,
            )

        except Exception as e:
            # Compensate in reverse order with bounded retry per compensation
            compensation_results = []

            if followup_id:
                result = await self._safe_compensate(
                    cancel_followup_activity,
                    followup_id,
                    step_name="schedule_followup",
                )
                compensation_results.append(result)

            if confirmation_email_id:
                # Email is irreversible; send a correction
                result = await self._safe_compensate(
                    send_correction_email_activity,
                    {**params.dict(), "original_email_id": confirmation_email_id},
                    step_name="confirmation_email",
                )
                compensation_results.append(result)

            if billing_update_id:
                result = await self._safe_compensate(
                    revert_billing_state_activity,
                    {"customer_id": params.customer_id, "snapshot": billing_snapshot},
                    step_name="update_billing_record",
                )
                compensation_results.append(result)

            if applied_credit_id:
                result = await self._safe_compensate(
                    reverse_credit_activity,
                    applied_credit_id,
                    step_name="apply_credit",
                )
                compensation_results.append(result)

            # Check if any compensation failed permanently
            failed_compensations = [
                r for r in compensation_results
                if not r.success and r.error_type == "permanent"
            ]

            if failed_compensations:
                # Escalate to human with full context
                await workflow.execute_activity(
                    escalate_compensation_failure_activity,
                    EscalationContext(
                        saga_id=workflow.info().workflow_id,
                        original_error=str(e),
                        failed_compensations=failed_compensations,
                        completed_steps=[
                            applied_credit_id, billing_update_id,
                            confirmation_email_id, followup_id,
                        ],
                        suggested_actions=_suggest_recovery(failed_compensations),
                    ),
                    start_to_close_timeout=timedelta(seconds=30),
                )
                return SagaResult(
                    status="escalated",
                    error=str(e),
                    failed_compensations=failed_compensations,
                )

            return SagaResult(
                status="compensated",
                error=str(e),
                compensation_results=compensation_results,
            )

    async def _safe_compensate(
        self,
        compensation_activity: Callable,
        params: any,
        step_name: str,
    ) -> CompensationResult:
        """Run a compensation with bounded retry and categorical error handling."""
        try:
            await workflow.execute_activity(
                compensation_activity,
                params,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(
                    maximum_attempts=5,
                    initial_interval=timedelta(seconds=2),
                    backoff_coefficient=2.0,
                ),
            )
            return CompensationResult(success=True, step_name=step_name)
        except PermanentCompensationError as e:
            return CompensationResult(
                success=False,
                step_name=step_name,
                error_type="permanent",
                error=str(e),
            )
        except Exception as e:
            return CompensationResult(
                success=False,
                step_name=step_name,
                error_type="exhausted_retries",
                error=str(e),
            )
```

The agent's reasoning code becomes much shorter:

```python
# Customer-success agent's tool: invokes the saga
@tool
async def apply_customer_credit(
    customer_id: str,
    amount: float,
    reason: str,
    ticket_id: str,
) -> dict:
    """
    Apply a credit to a customer's account. Invokes the credit application saga
    which handles the multi-step process with proper rollback on failure.

    Returns the saga's result. Successful sagas return {"status": "success", ...}.
    Failed sagas return {"status": "compensated", ...} (system is back to original state)
    or {"status": "escalated", ...} (system needs human intervention).
    """
    handle = await temporal_client.start_workflow(
        CustomerCreditSaga.run,
        CreditApplicationParams(
            customer_id=customer_id,
            amount=amount,
            reason=reason,
            ticket_id=ticket_id,
            saga_id=f"credit:{ticket_id}:{generate_short_id()}",
        ),
        id=f"credit_saga_{ticket_id}_{generate_short_id()}",
        task_queue="customer-success-sagas",
    )

    result = await handle.result()
    return result.dict()
```

The agent's prompt teaches it about the three possible saga outcomes:

```python
SYSTEM_PROMPT_WITH_SAGA = """
When you call apply_customer_credit, the operation runs as a saga that handles
multi-step actions atomically. Three possible outcomes:

- "success": all steps completed; the customer received the credit and was notified
- "compensated": at least one step failed; all completed steps were rolled back;
  the system is back to its original state. Inform the customer that the action
  could not be completed and offer to retry or escalate.
- "escalated": at least one compensation failed; the system is in an inconsistent
  state and a human is being notified. Inform the customer that there's been an
  issue and they'll be contacted directly.

Always read the saga's status and respond appropriately. Don't proceed as if the
operation succeeded when it did not.
""".strip()
```

### The before-and-after metrics

Three months after the rollout, Sam pulled the metrics:

| Metric | Pre-saga | Post-saga |
|---|---|---|
| Multi-step actions per day | ~340 | ~340 |
| Partial-state incidents (customer affected) | ~14 per month | 0 |
| Compensation failures requiring human escalation | N/A | ~3 per month |
| Median action latency | 1.2s | 1.4s |
| p99 action latency | 4.8s | 6.2s |
| Cost per action (infrastructure) | $0.012 | $0.018 |

The trade-offs:

- **Reliability** — partial-state incidents went from 14/month to 0. The 3/month escalations were now visible in a structured queue rather than discovered through customer complaints.
- **Latency** — modest increase from durability overhead. Acceptable for action operations (asynchronous from user perspective).
- **Cost** — 50% increase from Temporal infrastructure. Easily justified by the operational savings (no more 3 a.m. customer-confused-by-mystery-credit emergencies).

The customer success team's response: *"Customers stopped getting confused emails. Most importantly, the failures we couldn't avoid are now showing up where we can act on them — in our queue — instead of in customer complaints two days later."*

Sam wrote one line for the playbook: *"the saga isn't there to make actions succeed. It's there to make sure failures don't become customer mysteries."*

::: code-exercise
**Exercise 16.1 — Saga audit.**

Pick an agent in your stack that takes real-world actions (sending emails, modifying databases, calling external APIs in sequence). Run the audit:

1. **Catalog the actions.** What multi-step operations does the agent perform? Which steps are reversible? Which aren't?
2. **Identify partial-state risks.** For each multi-step action, what happens if step N fails after step N-1 succeeded? Is the system left in a recoverable state automatically, or does it require a human?
3. **Design the saga.** For each multi-step action, define the forward steps, the compensation for each, and the idempotency strategy.
4. **Implement durability.** Decide on a workflow engine (Temporal, Step Functions, Camunda) or hand-rolled implementation with state persistence. For agent harnesses, evaluate Anthropic Managed Agents.
5. **Test compensation paths.** Force failures at each step and verify the system returns to a clean state.
6. **Set up escalation.** Define the on-call queue for compensation failures with full context and suggested recovery actions.

The goal is calibration. Most agent systems with real-world actions have partial-state risks they haven't fully cataloged. The audit makes them visible; the patterns from this module make them addressable.
:::

---

## Section 9 — The Framework

Adding saga discipline to agent systems with real-world actions:

**1. Catalog every multi-step action.** What sequences of operations does your agent perform? Map them explicitly. Most teams underestimate how many of their actions are actually multi-step.

**2. Categorize each step's reversibility.** Fully reversible (state can be restored), partially reversible (corrective action possible), irreversible (no compensation possible). The categorization drives the saga design.

**3. Default to orchestration.** Central coordination beats event-driven choreography for agent systems. The agent's reasoning is the natural orchestrator role; aligning the saga's orchestration with the agent's reasoning keeps complexity bounded.

**4. Use a workflow engine in production.** Temporal, AWS Step Functions, Camunda, or Anthropic Managed Agents. Hand-rolled sagas don't survive on-call rotations. The infrastructure exists; use it.

**5. Idempotency keys derived from saga ID + step name.** Deterministic across retries. Excludes cosmetic fields. Forward steps and compensations both need them.

**6. Design compensations as forward operations.** "Rollback" is the wrong mental model for agent contexts. Compensations are corrective actions that achieve the business goal of "undoing" — sometimes by reversing state, sometimes by sending corrections, sometimes by escalating to humans.

**7. Plan for compensation failures.** Bounded retry, categorical error handling (transient vs permanent), human escalation as terminal compensation with structured context.

**8. Place irreversible steps last.** When you can't avoid an irreversible step, make it the last step in the saga. Earlier failures compensate cleanly without touching the irreversible action.

**9. Persist state at every transition.** Workflow engines do this automatically. Hand-rolled implementations need explicit checkpointing.

**10. Test the failure paths.** Forward paths succeed in development. Compensation paths only get tested if you force failures. A saga whose compensations have never been exercised in test is a saga whose compensations probably don't work.

The discipline this enforces: **multi-step actions are designed as transactions, not as sequences of independent operations.** The agent's reasoning is empowered to take complex actions; the saga ensures those actions succeed atomically or fail recoverably. Sam's customer-success bot needed both layers; production agent systems generally do.

---

## Recap: Module 16 in eight bullets

::: bullet-points
- Agent action failures are different from agent reasoning failures. Real-world side effects are durable; partial state isn't recoverable from logs alone; agent non-determinism amplifies retry problems. The saga pattern, well-established in distributed systems, is the canonical answer adapted for agent contexts.
- A saga is a sequence of local transactions, each with a compensating transaction, that together produce eventual consistency. ACID guarantees are traded for BASE — Basically Available, Soft state, Eventual consistency. For agent systems, this trade is almost always the right one.
- Orchestration beats choreography for agent sagas. The agent's reasoning is the natural orchestrator. Choreography fragments the saga's logical control across multiple turns and event handlers; orchestration keeps it where the agent's reasoning lives.
- Idempotency is non-negotiable. Every step and every compensation must be safely retriable. Idempotency keys derived from saga ID + step name (deterministic across retries), excluding cosmetic fields. Forward steps and compensations both need them.
- Compensations are not rollbacks. They're forward-only operations that achieve the business goal of "undoing" — sometimes by reversing state, sometimes by sending corrections (email already sent → send correction), sometimes by escalating. Design them with the same rigor as forward operations.
- Use a workflow engine in production. Temporal, AWS Step Functions, Camunda, or Anthropic Managed Agents. Hand-rolled sagas with proper persistence work for prototypes; they rarely survive on-call rotations. Anthropic's Managed Agents handles harness durability; explicit saga design handles cross-service action durability.
- Plan for compensation failures. Bounded retry with exponential backoff. Categorical error handling: transient vs permanent. Human escalation as terminal compensation, with structured context and suggested recovery actions. The escalation queue is where automated saga discipline meets human judgment.
- Sam's customer-success bot: pre-saga, ~14 partial-state incidents per month silently affecting customers; post-saga, 0 partial-state incidents and ~3/month structured escalations to the on-call queue. The saga isn't there to make actions succeed; it's there to make failures recoverable rather than mysterious.
:::

---

::: sam-arc
**Sam, after the customer-success bot's saga rollout.**

Three months after the redesign, the platform team's quarterly review came up. Sam presented the metrics dashboard. The 14-incidents-per-month → 0 number was the headline; the 3-escalations-per-month was the trailing detail. The CFO was at the meeting (the saga work had crossed her radar via the Temporal infrastructure cost review).

She asked: *"What's the actual return on the engineering investment? You spent a quarter on this. What did it produce?"*

Sam thought about it. The honest answer wasn't *"better customer experience,"* though that was true. The honest answer was *"the failures we couldn't avoid are now visible."* Pre-saga, partial-state failures became customer-confusion incidents that were detected through customer complaints, days after the fact, after the customer had already had a bad experience. Post-saga, the same failures became structured escalations to an on-call queue, addressed within an hour, often before the customer noticed anything.

The system hadn't gotten *more reliable* in the sense of fewer underlying failures. It had gotten *more transparent* in the sense of visible-and-actionable failures.

Sam said something like that. The CFO thought about it. *"You're saying we used to ship reliability bugs to customers, and now we ship them to the on-call queue. The bugs still happen. But the customer experience is consistent because we catch them before they become customer-facing."*

Yes, Sam said. That was it.

The CFO nodded. *"Make sure the next agent system has this from the start. Don't wait for the incident report."*

Sam wrote that down. The next entry in the platform team's design review checklist — *"For new agents that take real-world actions: catalog all multi-step operations, design saga + compensation per operation, define escalation queue with structured context, before launch."*

Sam's arc this module: **the saga isn't there to make actions succeed; it's there to make failures recoverable rather than mysterious.** Production agents will have failures — transient, permanent, novel, weird. The discipline isn't preventing failures; it's ensuring failures result in visible-and-actionable states rather than silent-and-confusing ones. The customer-success bot now had this. The next system would have it from the start.

Two modules left in the book — observability (M17) and the capstone (M18). Sam closed the laptop. The agent shipped. The customers were no longer confused. The on-call queue had three new structured escalations this month, and the team had handled all three in under an hour. That was the actual deliverable.
:::

---

## What's next

Module 17 covers observability for agent systems. The discipline of seeing what your system is actually doing in production — distributed tracing across agents, per-agent metrics, the alerting patterns that catch degradation before it becomes failure. Sam's saga work made failures *recoverable*; M17's observability work makes them *visible* before they cascade.

The relationship between M16 and M17: sagas ensure failures result in actionable states; observability ensures failures are detected before they spread. Both layers are needed. M16 was about building systems that survive failures gracefully; M17 is about building systems where you can see failures happening.

Module 18 is the capstone: a complete system pulling together every layer from the book — M2's hardened loop, M3's workflows, M4's tools, M5's MCP, M6's model selection, M7's context, M8's compaction, M9's memory, M10's hallucination prevention, M11's reflection, M12's guardrails, M13's multi-agent decision, M14's architectures, M15's communication, M16's sagas, and M17's observability. A system that demonstrates the whole arc.

For now: take the audit exercise from Section 8. Pick one agent that takes real-world actions. Catalog the multi-step operations. Identify partial-state risks. Design the saga. The patterns from this module are 50+ years of distributed systems wisdom applied to the new shape of agent systems. They work; they're production-tested; they turn agents from "they work in dev" into "they work at 3 a.m. when something flakes."
