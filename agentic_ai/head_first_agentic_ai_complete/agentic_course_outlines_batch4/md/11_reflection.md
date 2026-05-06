# Module 11 Outline — Reflection and Self-Correction

::: chapter-opener
<div class="module-num">MODULE 11 — OUTLINE</div>
<div class="module-title">Reflection and Self-Correction</div>
<div class="subtitle">Detection found the problem.<br>Reflection decides what to do about it — before the agent does it again.</div>
<div class="pages">Target length: ~36 pages</div>
:::

## What this module is

Reflection is the loop closure for agent reliability. Detection (Module 10) tells you something went wrong. Reflection turns that signal into action: replan, retry, abstain, escalate, or learn — depending on when the detection fires and what the agent's already committed to.

This module covers the two reflection paradigms the field converged on in 2025-2026: **retrospective** (act, observe failure, reflect, retry — the Reflexion / Self-Refine family) and **prospective** (criticize plans before execution — PreFlect, Feb 2026). We code both. We also cover the contrarian finding that reflection often *doesn't* help unless specifically architected for it (Kang et al., Oct 2025 — "Reflection steps during inference infrequently yield error correction unless specifically architected for multi-iteration or corrective reasoning"), and the certainty-guided suppression techniques that save 18-42% compute while preserving the benefit.

By the end the reader can: add reflection to the Module 10 agent without exploding cost or latency, decide when to reflect prospectively vs retrospectively, and avoid the "reflection theater" anti-pattern where the agent confidently confirms its own wrong answers.

::: hook
"Telling the model 'reflect on your answer' often makes things worse. The model rationalizes the answer it already gave instead of correcting it. Reflection done well is structural — different model, different prompt, different question, sometimes a different time. Reflection done badly is just paying twice for the same wrong answer."
:::

---

## Section 1 — The Two Paradigms (≈4 pages)

The PreFlect paper (Feb 2026) cleanly names the split:

**Retrospective reflection.** The agent acts, observes failure, then reflects and updates future trials. Reflexion (Shinn et al., 2023), Self-Refine (Madaan et al., 2023), MIRROR (Guo et al., 2025), SAMULE (Ge et al., 2025). The whole reflection-after-failure family.

```
[Plan] → [Act] → [Observe failure] → [Reflect] → [Retry with lesson] → ...
```

**Prospective reflection.** The agent generates a plan, then a critic inspects the plan *before* execution, flags issues, forces a revision. PreFlect (Feb 2026), AR (Wang et al., 2024).

```
[Draft plan] → [Critic inspects plan] → [Revise plan] → [Execute] → ...
```

Both have a place. Retrospective is necessary because some failures are unforeseeable and only visible after the fact. Prospective is cheaper because catching errors before execution avoids wasted tool calls, wasted tokens, wasted time.

The empirical finding from Reflexion (the founding paper): adding reflection to ReAct improved accuracy by ~17% on its eval. Subsequent work has both replicated and complicated this — sometimes reflection helps; sometimes it doesn't; the conditions matter.

::: pullquote
The question isn't "should I add reflection?" — it's "which reflection, when, with what critic, and how do I know it's actually correcting rather than rationalizing?"
:::

::: nodumbq
**Q: Isn't asking the model to "double-check" already reflection?**

A weak form. The Kang et al. Oct 2025 finding: reflection during inference rarely yields error correction unless the architecture is specifically designed for multi-iteration corrective reasoning. A "double-check the answer" prompt to the same model usually produces "yes that looks right" — confirmation bias, not correction.

**Q: When does reflection actually help vs. just costing more?**

When (1) the critic is meaningfully different from the producer (different model, different prompt, structured rubric), (2) the failure mode is detectable (incorrect tool use, missing requirements, internal contradictions), and (3) revision can act on the critique (the model can produce a different output, not just the same output with a defensive disclaimer). Without all three, reflection is theater.
:::

---

## Section 2 — Retrospective Reflection: Reflexion-Style (≈5 pages)

The original pattern, still useful and widely deployed.

The Reflexion architecture:

```
┌─────────────┐
│   ACTOR     │  ← ReAct-style agent
└──────┬──────┘
       │ executes trajectory
       ▼
┌─────────────┐
│  EVALUATOR  │  ← scores the trajectory (binary success/fail or scalar)
└──────┬──────┘
       │ if failed
       ▼
┌─────────────┐
│SELF-REFLECT │  ← generates verbal feedback: "what went wrong, how to fix"
└──────┬──────┘
       │ stores in memory
       ▼
┌─────────────┐
│  RETRY      │  ← actor re-runs, with reflection in context
└─────────────┘
```

```python
class ReflexionAgent:
    def __init__(self, actor, evaluator, reflector, max_trials: int = 3):
        self.actor = actor
        self.evaluator = evaluator
        self.reflector = reflector
        self.max_trials = max_trials
        self.reflections = []  # list of past reflections, persisted across trials
    
    async def run(self, task: str) -> AgentResult:
        for trial in range(self.max_trials):
            trajectory = await self.actor.execute(
                task=task,
                additional_context=self._format_reflections(),
            )
            evaluation = await self.evaluator.score(task, trajectory)
            
            if evaluation.success:
                return AgentResult(trajectory=trajectory, trials=trial + 1)
            
            # Failed — reflect and retry
            reflection = await self.reflector.reflect(
                task=task,
                trajectory=trajectory,
                failure=evaluation.failure_reason,
            )
            self.reflections.append(reflection)
        
        # Max trials reached without success
        return AgentResult(trajectory=trajectory, trials=self.max_trials, success=False)
    
    def _format_reflections(self) -> str:
        if not self.reflections:
            return ""
        return "Lessons from past attempts:\n" + "\n".join(
            f"- {r.lesson}" for r in self.reflections
        )
```

The reflection prompt is where the work happens:

```python
REFLECTION_PROMPT = """You are an autonomous agent that just attempted a task and failed.

TASK: {task}

YOUR ATTEMPT (trajectory):
{trajectory}

EVALUATOR FEEDBACK: {failure_reason}

Your job: explain WHY this attempt failed and what you would do DIFFERENTLY next time.
Be specific. Don't rationalize. Don't claim the attempt was actually correct.
Output a short lesson (1-3 sentences) that another version of you, retrying this task,
could use to avoid the same mistake.

LESSON:"""
```

**Critical design notes:**

- The evaluator must be external to the actor. Self-evaluation of trajectories is the same trap as self-evaluation of answers (Module 10).
- Reflections accumulate across trials in this trial. Don't dump them across all tasks ever — different tasks have different lessons.
- Cap trials. Reflexion's original work used 1-5 trials; gains saturate quickly.
- Most failures are caught in trial 2; trial 3+ has diminishing returns.

::: brain
Reflexion needs an evaluator that can score success/failure. For some tasks (math problems, code that runs tests) this is trivial. For open-ended tasks (writing, analysis) it's hard. How do you evaluate trajectory success when "success" itself is fuzzy?

(Three approaches: rubric-based LLM judge (define what success looks like; judge against rubric), human-in-the-loop sampling (human labels a fraction; train signal from that), or downstream proxy (did the output get used? edited? rejected?). Without an evaluator, retrospective reflection has no signal to learn from.)
:::

::: code-exercise
**Exercise 11.1 — Reflexion on a coding task.**

Build the Reflexion loop. Task: agent writes Python functions that should pass given test cases. Evaluator: runs the tests and reports pass/fail with the failing test's output. Reflector: explains what about the failure suggests changes.

Compare success rate over 3 trials with vs without reflection on a held-out set of 20 problems.
:::

---

## Section 3 — Self-Refine: Reflection Within a Single Run (≈4 pages)

Self-Refine (Madaan et al., 2023) is the sibling pattern: instead of multi-trial, do produce → critique → revise as a single iterative loop.

```python
async def self_refine(producer, critic, task: str, max_iters: int = 3) -> str:
    output = await producer.produce(task)
    for _ in range(max_iters):
        critique = await critic.critique(task, output)
        if critique.acceptable:
            return output
        output = await producer.revise(task, output, critique.feedback)
    return output
```

This is exactly the evaluator-optimizer workflow pattern from Module 3. We didn't call it "reflection" then because it wasn't named that way in the workflow patterns paper. It is reflection — a tightly-scoped intra-task version.

**When Self-Refine works:**
- Tasks with clear quality criteria (compliance, completeness, format)
- Producer is capable but variable (Sonnet-class)
- Critic is strong (Opus or specialized)
- Revision can meaningfully improve (more iterations don't always; gains often saturate at 2-3)

**When Self-Refine fails:**
- Producer can't produce *better* output, only different output (model ceiling reached)
- Critic and producer are the same model (rationalization, not refinement)
- Critique is too vague to act on

::: gotcha
Self-Refine often shows large gains in research papers and modest gains in production. The papers use carefully-crafted critic prompts on tasks where the critic genuinely catches errors. In production, generic "make this better" critics produce generic improvements that don't move quality metrics. The critic prompt is the entire game.
:::

---

## Section 4 — Prospective Reflection: PreFlect-Style (≈5 pages)

The newer pattern (PreFlect, Feb 2026). Catch errors *before* the agent commits to actions.

The architecture:

```
┌─────────────┐
│  PLANNER    │  ← agent generates initial plan (sequence of steps)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  CRITIC     │  ← inspects plan, flags issues (using distilled error patterns)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  REVISER    │  ← updates plan based on critique
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  EXECUTOR   │  ← runs the (revised) plan
└──────┬──────┘
       │ on unexpected deviation
       ▼
┌─────────────┐
│ RE-PLANNER  │  ← dynamic mid-execution plan update
└─────────────┘
```

The PreFlect insight: prospective reflection works best when the critic has access to **distilled planning errors** from past trajectories. Don't ask the critic to invent issues; give it a taxonomy of common failure modes (specific to the task domain) and ask it to check each.

```python
class PreFlectAgent:
    def __init__(self, planner, critic, executor, error_patterns: list[str]):
        self.planner = planner
        self.critic = critic
        self.executor = executor
        self.error_patterns = error_patterns  # distilled from history
    
    async def run(self, task: str) -> AgentResult:
        plan = await self.planner.plan(task)
        
        for revision_round in range(2):  # at most 2 critic passes
            critique = await self.critic.review(
                task=task, plan=plan, error_patterns=self.error_patterns,
            )
            if not critique.has_issues:
                break
            plan = await self.planner.revise(task, plan, critique)
        
        # Execute, with mid-flight re-planning if needed
        result = await self.executor.execute(plan, on_deviation=self._replan)
        return result
    
    async def _replan(self, current_plan, deviation):
        # Mid-execution: agent encountered something unexpected, replan from here
        return await self.planner.replan(current_plan, deviation)
```

**Where to get the error patterns:** mine your own production failures. Cluster them. Generate a taxonomy. The critic prompts include the taxonomy. Update periodically as new failure modes emerge.

Example taxonomy for a coding agent:
- "Plan calls a tool that requires authentication without first checking auth state"
- "Plan reads a file before checking if it exists"
- "Plan modifies state in a loop without considering rate limits"
- "Plan passes raw user input to an exec tool"

The critic checks the proposed plan against each pattern. If a match, flags + suggests fix.

::: postmortem
**The Critic That Caught the Production-DB Wipe**

A team built a database migration agent. Plans involved running schema changes. Initially deployed without prospective reflection. Within two weeks, an agent ran `DROP TABLE users` on production because the user said "delete the users table from staging" and the agent's plan didn't include "verify environment is staging before destructive operation."

Recovery: backups, panic, public apology, three-day incident.

Fix: prospective critic with a destructive-operations taxonomy. Every plan involving DROP/DELETE/TRUNCATE/UPDATE-without-WHERE is flagged. Critic forces an environment check as the first step. In the year since, zero production incidents from this class. The critic rejects ~3% of plans; ~90% of those would have been bugs.

**Lesson:** prospective reflection's value isn't catching average mistakes — it's catching the catastrophic ones before they execute. The math overwhelmingly favors having one.
:::

---

## Section 5 — Reflection That Doesn't Help (≈4 pages)

The contrarian section. Multiple 2025 papers (Kang et al. Oct, Ge et al. Mar, others) document that reflection often doesn't improve outcomes — and sometimes makes them worse.

**Failure mode 1: Confirmation bias.** Same model asked to reflect on its own answer mostly agrees with itself. The "reflection" is a rationalization. ~80% of self-reflective steps in the Kang et al. study produced no meaningful correction.

**Failure mode 2: Spurious revision.** The model changes a correct answer to an incorrect one because the critique prompt biased it toward "find something to change." Net negative.

**Failure mode 3: Reflection theater.** Long, eloquent self-critiques that don't change behavior. Token cost without quality gain. Common in production deployments where someone added "reflect carefully on your answer" to the prompt.

**Failure mode 4: Cost-benefit collapse.** Reflection adds 2-3× tokens per turn. If gains are <5% on a task you're shipping at scale, the math doesn't work.

The Kang et al. finding: certainty-guided reflection suppression saves 18-42% compute by *not* reflecting when the model's already confident. Confidence comes from entropy of token logits, semantic consistency across short samples, or rubric-based self-assessment with structured output.

```python
async def maybe_reflect(model_resp, critic, threshold: float = 0.85) -> str:
    confidence = await assess_confidence(model_resp)
    if confidence >= threshold:
        return model_resp.text  # skip reflection — model is already confident
    return await critic.critique_and_revise(model_resp.text)
```

When confidence is high, you save the reflection cost. When it's low, you reflect — which is when reflection actually helps.

::: brain
A skeptic could say: "if the model is wrong but confident, certainty-guided suppression skips the reflection that would have caught it." Is that a real concern?

(Yes, partially. Confident wrong answers are exactly the ones reflection would help most with. But: the data show that reflection doesn't reliably catch them either. So you're trading "reflect on everything for marginal-zero gain" against "reflect only when uncertain — when reflection actually helps." The latter is better economically and only slightly worse on quality.)
:::

::: gotcha
The single biggest reflection anti-pattern: adding "reflect on your answer before responding" to every prompt. This is reflection theater. Either build a structural reflection step (different model as critic, explicit revision call) or don't — but inline self-critique buried in a prompt rarely changes outcomes.
:::

---

## Section 6 — Reflection Granularity: Action vs Plan vs Trajectory (≈3 pages)

Reflection can fire at different levels. The right granularity depends on where errors originate.

**Action-level.** Reflect on each tool call before/after execution. Fine-grained, expensive, useful when individual actions are dangerous (the production-DB-wipe case).

**Plan-level.** Reflect on multi-step plans before any step executes (PreFlect). Cheaper than action-level, catches strategic errors.

**Trajectory-level.** Reflect after the whole task completes (Reflexion). Cheapest per-task, but errors only fixable on retry.

A practical mix:
- Plan-level critique on initial plan (catch strategic errors)
- Action-level critique on flagged dangerous actions only (catch catastrophic single-step errors)
- Trajectory-level reflection on failed runs to update procedural memory (Module 9) for future runs

```python
class LayeredReflection:
    async def before_plan(self, plan): return await self.plan_critic.review(plan)
    async def before_dangerous_action(self, action): return await self.action_critic.review(action)
    async def after_failed_trajectory(self, traj): return await self.trajectory_reflector.reflect(traj)
```

::: nodumbq
**Q: Won't all these reflection layers stack into massive cost overhead?**

Each layer has a different *frequency* and *gating*. Plan-level fires once per task. Action-level fires only on flagged actions (typically <5% of actions). Trajectory-level fires only on failed trajectories. Net overhead is usually 10-30% of base cost, not multiples — and the gains can be 2-5× on reliability metrics for high-stakes domains.
:::

---

## Section 7 — Reflection and Memory: Closing the Loop (≈3 pages)

Reflexion's real innovation wasn't the loop — it was *storing* the reflection so future runs benefit. This connects directly to Module 9's procedural and reflective memory.

Pattern:

```python
async def reflect_and_persist(reflector, memory, task, trajectory, failure):
    reflection = await reflector.reflect(task, trajectory, failure)
    
    # Decide: is this a one-off lesson or a generalizable pattern?
    if reflection.is_generalizable:
        # Add to procedural memory — applies to similar future tasks
        await memory.write_skill(Skill(
            trigger_pattern=reflection.applicable_to,
            instructions=reflection.lesson,
            examples=[reflection.failed_example],
        ))
    else:
        # Reflective memory — episodic with reflection tag
        await memory.write_reflective_episode(...)
```

Over time, the procedural memory becomes a library of "lessons learned" the agent automatically pulls in when starting similar tasks. This is the closest thing agents have to *learning* in production without retraining.

The "In Prospect and Retrospect" paper (Tan et al., ACL 2025) generalizes this into Reflective Memory Management — combining prospective reflection (summarize as you go) with retrospective reflection (refine retrieval based on what evidence got cited).

::: pullquote
A reflection that's not stored is a lesson learned in the moment and forgotten in the next call. Memory makes reflection cumulative.
:::

---

## Section 8 — Building It Into the Agent (≈4 pages)

We integrate reflection into the agent stack we've been building since Module 2.

```python
class ReflectiveAgent(Agent):  # extends Module 2-9 hardened agent
    def __init__(self, *args, plan_critic=None, action_critic=None,
                 trajectory_reflector=None, procedural_memory=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.plan_critic = plan_critic
        self.action_critic = action_critic
        self.trajectory_reflector = trajectory_reflector
        self.procedural_memory = procedural_memory
    
    async def run(self, query: str) -> AgentResult:
        # 1. Pull procedural memory: relevant lessons from past tasks
        skills = await self.procedural_memory.retrieve_skills_for(query)
        
        # 2. Generate initial plan
        plan = await self._generate_plan(query, skills)
        
        # 3. Prospective critique (PreFlect-style)
        if self.plan_critic:
            critique = await self.plan_critic.review(query, plan)
            if critique.has_issues:
                plan = await self._revise_plan(query, plan, critique)
        
        # 4. Execute, with action-level checks on flagged actions
        try:
            result = await super().run(query, plan=plan, action_check=self.action_critic)
        except AgentFailure as e:
            # 5. Trajectory-level reflection on failure
            if self.trajectory_reflector and self.allow_retry:
                lesson = await self.trajectory_reflector.reflect(query, e.trajectory, e.reason)
                if lesson.is_generalizable:
                    await self.procedural_memory.write_skill(lesson.to_skill())
                # Optional: retry with the lesson in context
                ...
            raise
        
        return result
```

We run a complete example: an agent solving a moderate research task. Show:
- Prospective critique flagging a missing verification step in the plan
- Action critique flagging an attempt to write to a production system
- (Optionally introduce a failure to demonstrate retrospective reflection + memory write)
- Future run: same task class triggers the stored skill, avoids the same mistake

::: code-exercise
**Exercise 11.2 — Add reflection to the production agent.**

Take the agent from Module 10 (with hallucination defense). Add: plan-level critic with a small error pattern taxonomy, action-level critic for destructive operations, retrospective reflection on failed trajectories with persistence to procedural memory.

Measure: gain in task success rate, latency overhead, cost overhead. Tune the gating thresholds (when to invoke each reflection layer) to match your stakes vs cost trade-off.
:::

---

## Section 9 — What's Next (≈1 page)

Module 12 covers **guardrails** — the policy enforcement layer. Reflection is one part of a defense-in-depth strategy; guardrails cover topic drift, scope violations, output policy compliance, and the security material that complements hallucination + reflection.

Module 14 (multi-agent) revisits reflection in the multi-agent setting: critic agents, debate patterns, judge cycles. The Reflexion-style pattern generalizes naturally to "one agent acts, another agent critiques."

Module 17 (production) covers the eval and observability needed to know whether your reflection is actually helping — not just whether it's running.

The capstone (Module 18) ships an agent with prospective + retrospective reflection wired in.

---

## Recurring elements used in this module

- 1 chapter opener
- 1 hook
- 2 No Dumb Questions
- 2 Brain Power prompts
- 1 Production Postmortem (production-DB wipe prevented)
- 2 Watch It! / Gotcha blocks
- 2 Code Exercises (Reflexion on coding, integrate into production agent)
- 2 Pullquotes
- 2 architecture diagrams (retrospective and prospective)
- 1 Bullet Points recap

::: sam-arc
Sam adds "reflect on your answer before responding" to the agent's system prompt. Latency goes up 30%, quality unchanged (Section 5's Failure Mode 3). Sam reads the Kang et al. paper, removes the inline reflection, and instead adds prospective plan critique with a small destructive-operations taxonomy. Catches the next would-be production DB issue in pre-execution review. Adds retrospective reflection that writes lessons to procedural memory; agent gradually gets better at the team's specific task patterns. Sam's arc this module: **structural reflection beats inline self-critique; the win is in the architecture, not the prompt**.
:::

::: page-budget
S1 (Two paradigms): 4p
S2 (Reflexion-style): 5p
S3 (Self-Refine): 4p
S4 (Prospective / PreFlect): 5p
S5 (Reflection that doesn't help): 4p
S6 (Granularity): 3p
S7 (Reflection + memory): 3p
S8 (Integration): 4p
S9 (Next): 1p
Recurring elements: 3p
TOTAL: ~36 pages
:::

::: sources
**Must verify when drafting:**

- Reflexion paper (Shinn et al., 2023) — primary citation; ReAct +17% accuracy claim
- Self-Refine paper (Madaan et al., 2023) — primary citation
- PreFlect paper (Wang et al., arXiv:2602.07187, Feb 2026) — for prospective reflection paradigm and architecture
- Kang et al. Oct 2025 finding on reflection efficacy — exact paper, exact numbers (18-42% compute savings via certainty-guided suppression)
- MIRROR paper (Guo et al., 2025) — multi-agent intra/inter reflection
- SAMULE (Ge et al., 2025) — fine-tuned reflection model
- AR (Wang et al., 2024) — real-time adaptation with backup moves
- RMM "In Prospect and Retrospect" (Tan et al., ACL 2025) — for memory-augmented reflection
- Agent-R (cited Semantic Scholar) — iterative self-training with reflection
- The "reflection during inference rarely yields error correction" finding — verify the exact source and conditions

**Stable knowledge:**
- Retrospective vs prospective taxonomy (now well-established)
- Confirmation bias as failure mode
- Plan-level vs action-level vs trajectory-level granularity
- Memory-augmented reflection as cumulative learning

**Cross-references:**
- Module 3 — Self-Refine = evaluator-optimizer workflow pattern; this module deepens it
- Module 9 — procedural memory is where lessons persist
- Module 10 — hallucination detection feeds reflection signals
- Module 12 — guardrails; reflection is one defense layer
- Module 14 — multi-agent reflection (debate, critique cycles)
:::

::: bullet-points
### Module 11 in eight bullets

(filled at draft time)

- Two paradigms: retrospective (act, fail, reflect, retry) and prospective (critique plan before execution)
- Reflexion-style is the foundational pattern; needs external evaluator and capped trials
- Self-Refine is the intra-task version; same pattern as evaluator-optimizer workflow
- PreFlect introduces prospective reflection with distilled error patterns
- Reflection often doesn't help: confirmation bias, spurious revision, reflection theater, cost-benefit collapse
- Certainty-guided suppression saves 18-42% compute by skipping reflection when model is already confident
- Granularity matters: plan-level, action-level, trajectory-level — mix based on stakes
- Reflection that's stored to procedural memory becomes cumulative learning across runs
:::
