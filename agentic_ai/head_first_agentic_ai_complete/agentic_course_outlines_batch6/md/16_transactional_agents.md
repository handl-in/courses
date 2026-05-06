# Module 16 Outline — Transactional Agents: Saga Patterns and State

::: chapter-opener
<div class="module-num">MODULE 16 — OUTLINE</div>
<div class="module-title">Transactional Agents: Saga Patterns and State</div>
<div class="subtitle">Your agent did three things successfully. The fourth failed.<br>Now what? Hint: distributed systems solved this in 1987.</div>
<div class="pages">Target length: ~36 pages</div>
:::

## What this module is

Multi-agent systems do many things, sometimes in parallel, often touching real-world state — files, APIs, databases, money, schedules. When step 4 of 7 fails, you have a problem distributed-systems engineers have been thinking about for decades: partial commits, inconsistent state, the need to undo work that's already been done.

This module borrows the **Saga pattern** (Garcia-Molina & Salem, 1987) and applies it to agents — specifically the SagaLLM architecture (Chang & Geng, Stanford, PVLDB 2025; arXiv:2503.11951) which extends sagas to LLM-based multi-agent planning. We cover the four foundational limitations SagaLLM identifies in current LLM-based planning systems (unreliable self-validation, context loss, lack of transactional safeguards, insufficient inter-agent coordination), the saga primitives (transaction agents, compensation agents, validation agents, persistent context), and code a working saga-enabled multi-agent system.

By the end the reader can: identify when their agent system needs transactional safety, design compensating actions for non-idempotent tool calls, structure persistent state with checkpoints, and implement an independent validation layer that's not the same model as the producer.

::: hook
"Your travel-booking agent reserved the flight, charged the card, booked the hotel, and then failed on the rental car because the customer's license had expired. The flight is non-refundable. The hotel cancellation window passed during the agent's third retry. You owe the customer $1,400 in compensation, and your agent has no idea any of this happened. This is not an LLM problem. This is a problem distributed databases solved in the 80s."
:::

---

## Section 1 — Why Agents Need Transactions (≈4 pages)

Most early agent demos didn't need transactions because they didn't touch real state. A research agent reads things and produces a report. Worst case: the report is wrong; you read it, discard it, try again. No state to undo.

But the moment your agent:

- Charges a card
- Sends an email
- Posts a public message
- Modifies a database
- Books a resource
- Triggers a webhook to a partner
- Files a ticket
- Updates a CRM record

You're in distributed-systems territory. Failures partway through a multi-step process leave the world in an inconsistent state, and you need a way to either complete or undo.

The four foundational limitations the SagaLLM paper identifies in current LLM-based planning systems:

**1. Unreliable self-validation.** Models asked "did your action succeed?" mostly say yes. Self-validation is the same trap from Module 11's reflection — confirmation bias dressed as verification. A specialized validation agent (different model, different prompt, structured rubric) catches what self-validation misses.

**2. Context loss / context narrowing.** Long agent trajectories drift. By step 8, the agent has forgotten constraints from step 1. SagaLLM's experiments show advanced LLMs (Claude 3.7, GPT-o1 in their tests) often "fixate on recent context while neglecting critical earlier constraints, particularly in reactive planning scenarios where models attempt to retroactively rewrite past actions rather than adapting from the current state."

**3. Lack of transactional safeguards.** No rollback when steps fail. No compensation for actions already taken. No checkpoint to resume from.

**4. Insufficient inter-agent coordination.** Agents don't track each other's state changes. Agent A reserves a resource; Agent B doesn't know it's reserved; Agent C tries to use it.

This module addresses all four — borrowing primitives that distributed systems already proved.

::: pullquote
"Your agent committed to a side effect" is the same problem as "your transaction committed to a database." The solutions are the same. The field just relearned them with new vocabulary.
:::

::: nodumbq
**Q: Can't I just retry the failed step?**

Sometimes. Idempotent operations (ones safe to repeat) can be retried. Non-idempotent operations (charging a card, sending an email) cannot — retrying causes double charges. The whole point of saga compensation is handling the non-idempotent case where retry isn't safe.

**Q: This sounds like overkill for my support chatbot.**

It is, for a support chatbot. Sagas are for agents that take consequential actions in the world. If your agent only reads things and produces text, you don't need this. If it writes/charges/sends/triggers — you do.
:::

---

## Section 2 — The Saga Pattern (Background, Brief) (≈3 pages)

A 5-minute distributed-systems primer. The reader doesn't need to be a database expert; they need vocabulary for the rest of the module.

A **saga** is a sequence of local transactions where each transaction has a compensating transaction that semantically undoes it. If T1, T2, T3 succeed and T4 fails, the system runs C3, C2, C1 (in reverse) to leave the world consistent.

```
T1 → T2 → T3 → T4 fails
         ↓
        C3 → C2 → C1 (compensate, reverse order)
```

Two flavors:
- **Choreography**: each service knows what comes next; events drive transitions
- **Orchestration**: a central coordinator drives the saga (more agent-like)

Why sagas instead of two-phase commit (the other distributed transaction protocol):
- Sagas don't require all participants to be available at the same moment for a global commit
- Sagas tolerate long-running operations (a flight booking takes seconds; 2PC would lock everyone out)
- Sagas relax atomicity for liveness — the trade you almost always want at the agent layer

What you give up:
- **Strict atomicity** — "all or nothing" becomes "all, or partial-with-compensation-attempted"
- **Isolation** — between T1 starting and the saga ending, intermediate states are visible
- You ensure **eventual consistency**, not strict ACID

For agents this is the right trade. The world doesn't pause while your agent thinks.

::: brain
A saga can fail to fully compensate (e.g., a refund API is down when you try to refund). What's the right behavior?

(Log the failed compensation as a durable record requiring manual intervention. Do not silently retry forever; do not pretend it succeeded; do not delete the record. The saga's job is to make failure *visible* and *recoverable*, not invisible. Operators close the loop on what the system couldn't.)
:::

---

## Section 3 — SagaLLM: Architecture and Primitives (≈5 pages)

The SagaLLM paper's contribution: applying these primitives to LLM multi-agent systems. The architecture has four primitive roles:

**1. Transaction agents.** Each does the actual work — books the flight, charges the card, sends the email. Wraps the side effect in a logged step.

**2. Compensation agents.** For each transaction agent, a paired agent that knows how to undo. If the booking agent reserved seat 14A, the compensation agent knows to call cancel-reservation with the reservation ID.

**3. Validation agents.** Independent agents (different model/prompt) that check whether each transaction actually achieved its goal. *Not* the same agent as the executor. SagaLLM's whole point: independent validation, not self-validation.

**4. Context management agents.** Maintain persistent state across the saga, prevent context loss/narrowing, ensure constraints from step 1 still apply at step 8.

The high-level flow:

```
        ┌──────────────────────────────┐
        │       SAGA ORCHESTRATOR      │
        └───┬───────────────┬──────────┘
            │               │
       plans saga    on each step:
       (T1...Tn)    [T_i] → [V_i] → ok? → next
                      │       │       
                      │       └─→ if fail → [C_(i-1) ... C_1]
                      ▼
                   [side effect]
                       ↓
                 [persistent log]
```

Each step is logged durably. State is checkpointed. Validation is independent. Failure triggers compensation. The whole thing has the audit trail of a database transaction with the planning flexibility of an LLM agent.

The SagaLLM paper notes their code is organized into three categories: **application interface** (what the user-facing agent calls), **SagaLLM core** (the saga primitives, logging, validation), and **LangGraph integration** (state graph wiring). We mirror that structure.

::: gotcha
"Compensation" is not "undo." Compensation is *semantic* undo — it cancels the effect, not the operation. You don't un-charge a card; you issue a refund. You don't un-send an email; you send a follow-up retraction. Designing compensations is the most subtle part of saga design — and the one LLMs are not good at on their own. Use code, not prompts, for compensation logic where you can.
:::

---

## Section 4 — Designing Compensating Actions (≈5 pages)

The hardest part. We give it the room it needs.

**Three categories of operations by compensability:**

**A. Naturally compensable.** The operation has a clear inverse. Reserve → cancel reservation. Charge → refund. Insert row → delete row. Send email → send retraction.

**B. Compensable with effort.** The operation has no native inverse, but a workflow that achieves the equivalent. Posted to a public channel? Edit/delete the post or post a correction. Triggered a partner webhook? Call their reverse endpoint or notify their team.

**C. Not compensable.** The operation cannot be undone. Sent a missile (literally — defense scenarios). Made an irreversible decision (deleted backup; burned bridge). Posted to a public feed that doesn't allow edits.

For Category A, compensation is trivial. For Category B, compensation is engineering. For Category C, you don't allow the saga to take that action without explicit confirmation that the operation is committed and unrecoverable.

**The design rule:** before adding a transaction step, write its compensation. If you can't write the compensation, you don't have a transaction step — you have a point of no return that needs human approval.

```python
@dataclass
class TransactionStep:
    name: str
    do: Callable[[State], TransactionResult]
    compensate: Callable[[State, TransactionResult], CompensationResult]
    validate: Callable[[State, TransactionResult], ValidationResult]
    
    # Hard rule: do and compensate must both be defined.
    # If you can't define compensate, this isn't a transaction step.
```

**Idempotency keys** — the under-appreciated tool. Many APIs support an `Idempotency-Key` header. Sending the same key twice gives you the same result without doing the work twice. Use idempotency keys on all outbound API calls in your saga so retries don't double-charge / double-send.

```python
async def charge_card(amount: int, idempotency_key: str) -> ChargeResult:
    return await stripe.charges.create(
        amount=amount,
        currency="usd",
        idempotency_key=idempotency_key,  # safe to retry
    )
```

**The "best-effort compensation" pattern.** Sometimes compensation itself can fail. Refund API down. Email retraction service offline. The saga should:
1. Attempt compensation
2. On failure, log durably to a "compensations needed" queue
3. Continue compensating other steps (don't block the whole rollback)
4. Surface the queue to operators

Never fail silently. Never retry forever. Make the failure visible and resolvable.

::: postmortem
**The Saga That Couldn't Refund**

A travel-booking agent system used Stripe for charges, a regional payment processor for some markets, and a third for refunds. When a booking saga failed in the third market, compensation tried to refund. The third processor was down. The saga's compensation code retried 5 times, then... silently logged "refund failed" and moved on.

Two weeks later, customers had received goods/services and partial bookings, were never charged successfully, and the saga thought everything was fine. Reconciliation against the actual state of the world found 47 incomplete sagas affecting $90K in disputed charges.

Fix: the "compensations needed" queue, monitored. After 3 retries, the saga marks the compensation as `requires_manual_intervention` and pages the on-call. No silent failures. No "best-effort and forget."

**Lesson:** compensation can fail. The saga must surface that, not hide it.
:::

---

## Section 5 — Persistent State and Checkpointing (≈4 pages)

Sagas need durable state. If the agent crashes mid-saga, the next process needs to know which steps completed, which compensations are needed, what the original plan was.

**Three things to persist:**

1. **The saga plan.** What steps will run, in what order, with what compensations.
2. **Step state.** For each step: not-started / in-progress / committed / failed / compensated.
3. **Checkpoints.** Snapshots of the agent's reasoning + tool results between steps, so a resumed saga can pick up with full context.

```python
class SagaStore:
    """Durable storage for saga state. Could be Postgres, DynamoDB, etcd, etc."""
    
    async def create_saga(self, plan: SagaPlan) -> str:
        saga_id = uuid.uuid4().hex
        await self.db.write(saga_id, SagaRecord(
            id=saga_id, plan=plan, state="planning_done", checkpoints=[],
        ))
        return saga_id
    
    async def record_step_start(self, saga_id, step_idx, idempotency_key):
        ...
    
    async def record_step_complete(self, saga_id, step_idx, result):
        ...
    
    async def record_step_failure(self, saga_id, step_idx, error):
        ...
    
    async def record_compensation(self, saga_id, step_idx, comp_result):
        ...
    
    async def checkpoint(self, saga_id, step_idx, agent_state):
        ...
    
    async def resume(self, saga_id) -> ResumePoint:
        record = await self.db.read(saga_id)
        return determine_resume_point(record)
```

**The resume protocol.** On startup, the saga orchestrator looks for in-progress sagas and either:
- Continues forward if the saga was making progress
- Triggers compensation if a failed step was recorded but compensation didn't complete
- Marks for manual review if state is ambiguous (e.g., step started but no completion or failure recorded — the worst case)

LangGraph's checkpointing is designed exactly for this — it ships with Postgres and SQLite checkpointers and supports time-travel debugging. Most production saga implementations either build on LangGraph's checkpointer or implement equivalent functionality.

::: brain
A saga's step started, made an external API call, then the process crashed. On restart, you don't know if the API call succeeded or failed. What do you do?

(Three options, each with trade-offs: (a) Use idempotency keys so you can safely retry — best when the API supports it. (b) Query the API for the resource state to determine what happened — works when the API has good state-query endpoints. (c) Mark for manual review — last resort. The right answer depends on which the API supports; design saga steps to use (a) when possible.)
:::

---

## Section 6 — Independent Validation (≈4 pages)

The most important non-obvious primitive in SagaLLM. Validation cannot be the same agent as execution.

The pattern:

```python
class ValidationAgent:
    """Independent agent that verifies a step actually did what it claimed."""
    
    def __init__(self, model="claude-opus-4-7"):  # different from executor
        self.client = AsyncAnthropic()
        self.model = model
    
    async def validate(self, step: TransactionStep, claimed_result: dict, world_state: dict) -> ValidationResult:
        prompt = VALIDATION_PROMPT.format(
            step_name=step.name,
            step_intended_outcome=step.intended_outcome,
            claimed_result=json.dumps(claimed_result),
            world_state_evidence=json.dumps(world_state),
        )
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        return parse_validation(resp.content[0].text)


VALIDATION_PROMPT = """You are a validation agent. Another agent claims to have completed a step.
Your job: verify, with evidence, that the step's intended outcome was actually achieved.

Step: {step_name}
Intended outcome: {step_intended_outcome}
Claimed result: {claimed_result}
World state evidence (queried independently): {world_state_evidence}

Verify rigorously. Do not assume the claim is true because the agent says so.
Look for concrete evidence. If the world state doesn't match the claim, flag it.

Output JSON:
{{"validated": true/false, "evidence": "...", "concerns": ["..."]}}"""
```

**Critical design notes:**

- The validation agent uses a *different* model from the executor (or at least a different family). Same model = confirmation bias.
- The validation agent has access to *independent* evidence — query the API yourself, don't trust the executor's claimed result.
- Validation is a saga step too. It can fail. Failed validation triggers compensation just like a failed transaction.
- Validation prompts should look for *evidence*, not vibes. "The booking confirmation includes seat 14A" is evidence; "the agent says it booked seat 14A" is not.

::: gotcha
A common mistake: making validation cheap by using a tiny model with a vague prompt. This produces a validator that approves everything (cheap and useless) or rejects everything (cheap and over-cautious). Validation is where you spend Opus-class compute, because validation errors propagate. Cheap validators give you the illusion of safety without the safety.
:::

---

## Section 7 — Building a Real Saga-Enabled Agent (≈5 pages)

End-to-end implementation. We pick a concrete domain — order processing for an e-commerce app — and build it.

The scenario: a customer places an order. The agent must:
1. Reserve inventory (T1: reserve / C1: release)
2. Charge the card (T2: charge / C2: refund)
3. Generate shipping label (T3: create label / C3: void label)
4. Notify warehouse (T4: send notification / C4: send cancellation notification)
5. Send customer confirmation email (T5: send / C5: send order-cancelled email)

If T3 fails (e.g., shipping API is down), compensation runs C2 (refund), then C1 (release inventory). T4 and T5 never ran, so no compensation for those.

```python
class OrderProcessingSaga:
    def __init__(self, store: SagaStore, executor: AsyncAnthropic, validator: ValidationAgent):
        self.store = store
        self.executor = executor
        self.validator = validator
        self.steps = self._build_steps()
    
    def _build_steps(self) -> list[TransactionStep]:
        return [
            TransactionStep(
                name="reserve_inventory",
                do=self._reserve_inventory,
                compensate=self._release_inventory,
                validate=lambda r, ws: self.validator.validate(...),
            ),
            TransactionStep(
                name="charge_card",
                do=self._charge_card,
                compensate=self._refund_card,
                validate=lambda r, ws: self.validator.validate(...),
            ),
            # ... etc
        ]
    
    async def run(self, order: Order) -> SagaResult:
        saga_id = await self.store.create_saga(plan=self._plan_for(order))
        completed = []
        try:
            for idx, step in enumerate(self.steps):
                idempotency_key = f"{saga_id}:{step.name}"
                await self.store.record_step_start(saga_id, idx, idempotency_key)
                
                # Execute
                result = await step.do(order, idempotency_key)
                await self.store.record_step_complete(saga_id, idx, result)
                
                # Validate independently
                validation = await step.validate(result, await self._query_world_state(order))
                if not validation.validated:
                    raise ValidationFailed(step=step, evidence=validation.concerns)
                
                completed.append((step, result))
            
            return SagaResult(success=True, saga_id=saga_id)
        
        except Exception as e:
            # Compensate in reverse
            comp_results = []
            for step, result in reversed(completed):
                try:
                    cr = await step.compensate(order, result)
                    await self.store.record_compensation(saga_id, step.name, cr)
                    comp_results.append(("ok", step.name, cr))
                except Exception as comp_err:
                    await self.store.record_compensation(saga_id, step.name, comp_err)
                    comp_results.append(("failed", step.name, comp_err))
                    # Continue with other compensations; don't block on one failure
            
            return SagaResult(success=False, saga_id=saga_id, error=e, compensations=comp_results)
```

The full implementation includes the LangGraph integration (the saga as a state graph with node = step), the persistent context layer that prevents context loss across long sagas, and the orchestrator agent that decides which sagas to run for which orders.

We walk through:
- How idempotency keys are generated (deterministic from saga_id + step name)
- How world-state queries work (don't trust the API response from the executor; re-query independently)
- How the orchestrator handles partial-completion-on-resume (the worst case)
- How human-in-the-loop fits in for non-compensable steps

::: code-exercise
**Exercise 16.1 — Build a saga-enabled travel booking agent.**

Implement the saga pattern for a travel booking flow: flight + hotel + rental car. Each has reserve and cancel APIs. Use idempotency keys. Add an independent validation agent. Persist state. Test by injecting failures at each step (random fault injection) and verify compensation runs correctly.

Bonus: simulate a compensation failure (e.g., one cancel API returns 500) and verify the system surfaces the issue rather than swallowing it.
:::

---

## Section 8 — When NOT to Use Sagas (≈3 pages)

The honest section. Sagas have real costs.

**1. Cost.** Each step has a transaction agent + validation agent + persistent state writes. For low-stakes flows, this is overkill.

**2. Latency.** Validation adds latency between steps. Persistent state adds I/O. A 5-step saga can be 30-60% slower than the equivalent un-sagaed flow.

**3. Complexity.** Sagas are a non-trivial commitment. Compensation logic doubles your code surface. State persistence adds infrastructure. Independent validation adds models.

**4. Compensation isn't always meaningful.** For some flows, "compensation" doesn't have a useful semantic. A research report saga that "rolls back" is just deleting the report — which the user can do themselves.

**The right scope for sagas:**
- Multi-step flows that touch real-world state (money, resources, communications)
- Flows where partial completion is worse than full failure (booking 3 of 4 things)
- Flows where audit trail matters (regulated industries, high-value transactions)

**Wrong scope:**
- Read-only flows (no state to undo)
- Single-step actions (no saga; just retry / fail)
- Internal-only flows where partial completion is benign
- Prototype phase (build it without sagas; add when stakes warrant)

::: pullquote
Sagas are insurance. You pay a premium per transaction. The premium is worth it when the cost of an inconsistent state exceeds the saga's overhead. Otherwise it's a tax on your hot path for problems you don't have.
:::

---

## Section 9 — What's Next (≈1 page)

Module 17 covers production observability — including how saga state shows up in traces. A well-instrumented saga is debuggable; a poorly-instrumented one is a black box. We cover OpenTelemetry GenAI semantic conventions, LangSmith/Langfuse for agent traces, and the eval infrastructure that catches saga regressions.

Module 18 (capstone) ships a multi-agent research system. The capstone doesn't strictly need sagas (research is mostly read-only), but the *pattern of independent validation* from this module shows up directly there.

For readers building consequential-action agents — payment, booking, communication, infrastructure — Module 16's patterns are non-negotiable. The capstone won't fully exercise them, but production deployments of the capstone in different domains will.

---

## Recurring elements used in this module

- 1 chapter opener
- 1 hook
- 2 No Dumb Questions
- 2 Brain Power prompts
- 1 Production Postmortem (the saga that couldn't refund)
- 1 Watch It! / Gotcha block
- 1 Code Exercise (travel booking saga)
- 2 Pullquotes
- 1 saga architecture diagram
- 1 Bullet Points recap

::: sam-arc
Sam, having shipped reflective + guarded + multi-agent systems through Modules 11-15, gets pulled into a different team's incident: their travel-booking agent left 12 customers in inconsistent states over the weekend. Sam reads SagaLLM, applies the architecture, builds out the order-processing saga (Section 7) for that team. Adds independent validation (Section 6) — catches that the existing executor was claiming successes that didn't match world state ~3% of the time. The team's "our agent works fine" turns into "our agent works fine *and we can prove it*." Sam's arc this module: **distributed systems thinking applied to agents — the field rediscovered the saga pattern, and the rediscovery was overdue.**
:::

::: page-budget
S1 (Why agents need transactions): 4p
S2 (Saga pattern background): 3p
S3 (SagaLLM architecture): 5p
S4 (Designing compensations): 5p
S5 (Persistent state): 4p
S6 (Independent validation): 4p
S7 (Building a saga-enabled agent): 5p
S8 (When NOT to use sagas): 3p
S9 (Next): 1p
Recurring elements: 2p
TOTAL: ~36 pages
:::

::: sources
**Must verify when drafting:**

- SagaLLM paper (Chang & Geng, Stanford, arXiv:2503.11951; PVLDB 2025) — primary source for the four limitations, the architecture, the validation methodology
- The empirical claim about Claude 3.7 and GPT-o1 fixating on recent context (verify exact phrasing and which models)
- Garcia-Molina & Salem 1987 — the original Saga paper (foundational citation)
- "Multi-LLM Agent Collaborative Intelligence: The Path to Artificial General Intelligence" (Chang, ACM Books 2025) — companion reference
- ALAS paper (Chang & Geng 2025) — companion paper with detailed algorithms (per arXiv abstract)
- LangGraph checkpointer documentation — Postgres/SQLite/in-memory variants
- Stripe idempotency key documentation — for the idempotency-key pattern
- Anthropic SDK current model identifiers (claude-opus-4-7, claude-sonnet-4-6) for code samples

**Stable knowledge:**
- Saga pattern primitives (transaction, compensation, validation)
- Three categories of compensability (A/B/C from Section 4)
- Idempotency keys as a defensive pattern
- ACID relaxation trade-offs

**Cross-references:**
- Module 4 — tool design with idempotency in mind
- Module 11 — independent validation is structural reflection
- Module 12 — guardrails as another defense layer alongside saga validation
- Module 13/14 — multi-agent systems are where sagas matter most
- Module 17 — observability for sagas
- Module 18 — capstone (validation pattern carries over even without full sagas)
:::

::: bullet-points
### Module 16 in eight bullets

(filled at draft time)

- Agents that touch real-world state need transactional safety; "research-only" agents don't
- The Saga pattern (Garcia-Molina 1987) provides primitives: transactions, compensations, eventual consistency
- SagaLLM (Chang & Geng 2025) extends sagas to LLM multi-agent: transaction agents + compensation agents + validation agents + persistent context
- Compensation is semantic undo, not literal undo; design compensations *before* writing transaction steps
- Use idempotency keys on outbound API calls so retries don't double-charge / double-send
- Persist saga state durably; design for crash-and-resume from any step
- Validation must be independent — different model from executor — or it's confirmation bias dressed as verification
- Sagas have real costs (latency, complexity, code surface); use them when partial completion is worse than full failure
:::
