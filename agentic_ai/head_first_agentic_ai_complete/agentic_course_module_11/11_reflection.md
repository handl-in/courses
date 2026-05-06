# Module 11 — Reflection and Self-Correction

::: chapter-opener
<div class="module-num">MODULE 11</div>
<div class="module-title">Reflection and Self-Correction</div>
<div class="subtitle">Asking the model to check its own work is one of the most effective things you can do.<br>Asking it the wrong way is one of the most expensive.</div>
<div class="pages">~38 pages · turning a single answer into a deliberate process</div>
:::

::: hook
The deal-research workflow's brief evaluator was producing weird verdicts. Sam noticed it on a Tuesday afternoon: the same brief, run through the evaluator twice in a row, came back with different scores. First time: 6.5/10, with feedback flagging "weak competitive analysis." Second time: 8/10, with no mention of competitive analysis at all.

This wasn't a hallucination problem from M10. The facts in the brief were grounded. The evaluator wasn't fabricating anything. It just couldn't make up its mind about what was good and what wasn't.

Sam ran the evaluator twenty times on the same brief. Scores ranged from 5 to 9. Standard deviation 1.3. The mean was 6.8, which sounded reasonable until Sam looked at the distribution: two clusters — one around 5.5, one around 7.5. The evaluator was reading the brief through *different lenses* on different runs. One lens noticed weak competitive analysis. Another lens noticed strong financial detail. Same brief; same evaluator; same prompt; different verdicts.

Worse: when Sam re-ran the brief generator with the 6.5/10 feedback, the *next* version of the brief got a 5/10 from the evaluator — because now the evaluator was reading through a third lens that flagged "lacks executive summary punch" (which the original 6.5 evaluation hadn't mentioned).

The evaluator-optimizer loop was thrashing. Each iteration was nominally improving the brief but the evaluator's own inconsistency meant the optimizer was chasing different targets each time. Output quality wasn't converging; it was wandering.

Sam looked at the evaluator's prompt: a single LLM call, asking for a score and feedback in one pass. No reflection. No self-check. The evaluator was an LLM call dressed up as a process — which made it about as reliable as any single LLM call.

Sam wrote the line: *"the evaluator from M3 needs the discipline from M11. Otherwise the optimization loop is just expensive noise."*
:::

---

## What this module is

Hallucinations (M10) are about generation-time correctness — what the model produces in the first place. This module is about *post-generation* discipline: catching mistakes after the fact, refining outputs through structured iteration, and adding the deliberate-thinking surfaces that turn a single LLM call into a more reliable process.

The pattern goes by many names — reflection, self-correction, self-critique, verbal reinforcement, iterative refinement. Reflexion (Shinn et al. 2023) gave the field its canonical framing: the agent produces output, evaluates the output, generates linguistic feedback, and incorporates that feedback into its next attempt. Self-Refine (Madaan et al. 2023) generalized this into a generate-critique-correct loop without external supervision. Anthropic's "think" tool and adaptive/interleaved thinking are the platform-native versions of the same idea — surfaces for the model to deliberate rather than just produce.

The structure of this module:

- **Section 1** — what reflection actually is and the three places you can apply it (within a single response, between agent turns, across runs)
- **Section 2** — Anthropic's first-party surfaces: extended thinking (adaptive), interleaved thinking, the "think" tool. When to use each
- **Section 3** — the core reflection loop pattern from Reflexion and Self-Refine, with production code
- **Section 4** — the evaluator-optimizer pattern from M3 lifted into the agent context, with the discipline that fixes Sam's thrashing evaluator
- **Section 5** — the limits of reflection: blind-spot reinforcement, diminishing returns, when reflection makes things worse
- **Section 6** — multi-agent reflection patterns (preview of M13) and when single-agent isn't enough
- **Section 7** — cost calibration: iteration caps, token budgets, the hard limits that prevent infinite loops
- **Section 8** — putting it together: Sam's evaluator rebuilt
- **Section 9** — the framework

By the end of this module:

- You'll know when reflection is high-leverage and when it's just expensive
- You'll have working code for the canonical reflection patterns
- You'll understand why naive reflection often fails (the blind-spot problem) and how to address it
- You'll be able to use Anthropic's first-party thinking surfaces correctly — adaptive vs interleaved vs the "think" tool

The connection to M10: hallucination prevention happens at generation time; reflection happens after. The agent that produced a hallucinated response in M10 could, with reflection, catch it before delivering it. The two are complementary.

::: pullquote
Reflection is one of the most effective techniques in the agent toolkit and one of the most commonly misapplied. Done well, it produces dramatic quality improvements at modest cost. Done poorly, it inflates token bills, increases latency, and sometimes produces *worse* output than no reflection at all.
:::

---

## Section 1 — What Reflection Actually Is

The word "reflection" is doing a lot of work in agent literature. Let me decompose it.

**Generation-time reflection.** The model deliberates before producing the final output. Chain-of-thought prompting is the simplest version: "think step by step." Anthropic's extended thinking is the platform-native version: the model reasons in dedicated thinking tokens before generating the user-facing response. Within a single API call.

**Post-generation reflection.** The model (or a separate critic model) evaluates the produced output and either revises it or signals that revision is needed. M10's iterative refinement (Technique 6) was a single-pass version of this. The full Reflexion / Self-Refine pattern is the multi-iteration version.

**Inter-turn reflection.** Between turns of an agent loop, the agent reflects on what just happened — what tool result came back, whether the plan is still working, whether to adjust. Anthropic's interleaved thinking is the platform-native version: thinking blocks *between* tool calls.

**Cross-run reflection.** After a session ends, the agent (or a meta-agent) reflects on what happened across the whole run. What worked? What didn't? Lessons get stored in memory (M9) for future sessions. ExpeL, AutoGuide, and similar 2024–2026 frameworks formalize this.

This module focuses on the first three. Cross-run reflection sits at the intersection of M9 (memory) and what's becoming a research field of its own (continual learning for agents); we'll touch it briefly in Section 6.

### What all four have in common

The shared structure: **the model is given a chance to evaluate its own work and adjust before that work becomes load-bearing for downstream decisions.** Without reflection, every model output goes directly to use. With reflection, there's a deliberate pause — a place to catch errors before they propagate.

The shared cost: **every reflection step is another LLM call (or extra tokens within one call).** Reflection is not free. The decision to reflect — and how much — is a cost-quality trade-off, exactly like the model selection trade-off from M6.

::: nodumbq
**Q: Isn't reflection just chain-of-thought with extra steps?**

Chain-of-thought is *one* form of generation-time reflection. The broader pattern includes things CoT doesn't cover: separate critic models that look at finished output (post-generation), thinking blocks between tool calls (inter-turn), and lessons stored across sessions (cross-run). CoT is the simplest case; reflection is the family.

**Q: What's the actual evidence that reflection works?**

Mixed, but real. Reflexion (Shinn et al. 2023) showed dramatic gains on HotPotQA and HumanEval. Anthropic's "think" tool eval showed the highest pass^1 score (0.812) on a retail benchmark vs no thinking. But — and this is the key caveat — reflection's gains depend heavily on the *quality of the reflection signal*. Bad evaluators produce bad reflections, which can make output *worse* (Section 5 covers this). Reflection is high-leverage when the conditions are right; expensive when they aren't.
:::

---

## Section 2 — Anthropic's First-Party Reflection Surfaces

Anthropic ships several thinking-related capabilities. They're related but distinct, and using the wrong one for your use case is a common mistake. Let me clarify.

### Adaptive thinking (Sonnet 4.6, Opus 4.6/4.7)

The default thinking mode on the current Claude lineup. The model dynamically decides when and how much to think based on two factors: the `effort` parameter you set (low/medium/high) and the query's complexity. On easy queries, the model responds directly without thinking. On hard queries, it thinks deeply.

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=8192,
    thinking={"type": "adaptive", "effort": "medium"},
    messages=[{"role": "user", "content": user_question}],
)
```

When to use:

- **You want generation-time deliberation but don't want to manage budget yourself.** The model calibrates depth.
- **Your workload has heterogeneous difficulty.** Some queries warrant deep thinking; others don't. Adaptive handles both correctly.
- **You're on Opus 4.7.** This is the only thinking mode supported on Opus 4.7 — the model always thinks adaptively.

The `effort` parameter is the main control. Anthropic's internal evaluations report adaptive thinking reliably outperforming the older fixed-budget extended thinking on the same workloads.

### Extended thinking with `budget_tokens` (legacy mode)

The older mode, still supported on Sonnet 4.6 and Opus 4.6 but deprecated for those. Replaced by adaptive on newer models.

```python
# Legacy pattern — prefer adaptive on current models
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=16384,
    thinking={"type": "enabled", "budget_tokens": 8000},
    messages=[{"role": "user", "content": user_question}],
)
```

When to use:

- **You need a hard ceiling on thinking costs.** `budget_tokens` is a strict cap; adaptive is a soft signal.
- **You're maintaining older code or running on older Claude versions.** Migrating to adaptive is the recommended path forward.

For new builds on current models, prefer adaptive. The exception is the rare case where you need exact cost predictability (regulated workloads, fixed-price contracts) and the soft `effort` signal isn't sufficient.

### Interleaved thinking

Thinking blocks *between tool calls* in an agent loop. After the agent receives a tool result, the model thinks before deciding the next action. This is structurally different from adaptive/extended thinking, which both happen in one shot before generating output.

```python
# Interleaved thinking — Sonnet 4.5 and earlier required a beta header.
# Opus 4.6/4.7 enable it automatically when adaptive thinking is on.
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=16384,
    thinking={"type": "adaptive", "effort": "high"},
    tools=tools,
    messages=messages,
)
```

When to use:

- **Long agent trajectories with many tool calls.** Interleaved thinking lets the agent re-plan after each tool result instead of grinding through a frozen plan.
- **Tasks where tool results can reveal new information that should change the plan.** Research, debugging, exploratory work.
- **High-uncertainty environments where the agent can't predict tool outcomes.** Each result deserves a moment of thought before the next action.

When *not* to use:

- **Latency-sensitive tasks with predictable tool sequences.** If the agent knows what it's doing turn-by-turn, interleaved thinking is overhead.
- **Simple workflows where the tool sequence is fixed.** A workflow from M3 doesn't benefit from interleaved thinking; an open-ended agent often does.

The Anthropic AWS blog post on interleaved thinking with Strands frames it well: *"Interleaved thinking expands on the model's ability to self-reflect, correct errors, and orchestrate a workflow of reasoning and tool use."* The mechanism is exactly the inter-turn reflection from Section 1, baked into the platform.

### The "think" tool

A different thing entirely. Not built-in thinking — *a tool* you give the agent that does nothing except give it a place to write down a thought.

```python
THINK_TOOL = {
    "name": "think",
    "description": (
        "Use this tool to think through a problem. The 'thought' parameter is "
        "for your own reasoning. It does not affect the user-visible output."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "Your reasoning, plan, or analysis."
            },
        },
        "required": ["thought"],
    },
}
```

The tool returns nothing useful. The point is that calling it forces the model to verbalize its reasoning explicitly, in a place the model has been trained to think hard about (tool call inputs).

Anthropic's research showed this simple pattern produced the highest pass^1 score (0.812) on a retail policy compliance benchmark. The reason: the model can think *anywhere* with adaptive thinking, but the "think" tool gives it a *deliberate*, *named* surface to think on, which seems to elicit better-structured reasoning than free-form thinking.

When to use:

- **Compliance-heavy tasks.** Policy checks, rule-following, multi-step reasoning under constraints.
- **Tasks where the model's reasoning needs to be auditable.** The tool calls are visible in the trace; thinking blocks are not always (some are summarized).
- **Workloads that don't have access to native thinking** (Haiku 4.5 doesn't support extended thinking; the "think" tool does the same job for it).

When not to use:

- **Tasks already covered by adaptive/interleaved thinking.** Adding the "think" tool on top is redundant and adds tool-selection overhead.

### Choosing among the four

| Surface | When | Cost | Best for |
|---|---|---|---|
| Adaptive thinking | Generation-time deliberation | Variable | Default for hard tasks on Sonnet 4.6 / Opus |
| Extended thinking (legacy) | Need strict cost ceiling | `budget_tokens` cap | Regulated workloads only |
| Interleaved thinking | Inter-turn deliberation in agent loops | Variable | Long agent trajectories with tools |
| "think" tool | Explicit auditable reasoning | One tool call | Haiku, compliance work, policy adherence |

For most agents on Sonnet 4.6 or Opus 4.7: adaptive thinking with appropriate effort. Interleaved thinking is auto-enabled for tool use. Add the "think" tool only when you specifically need auditable reasoning steps or you're on Haiku.

::: brain
A coding agent runs many tool calls per session — file reads, test runs, code edits. A customer-support agent answers one question per session. Both are on Sonnet 4.6. Which thinking surfaces should each get?

(Coding agent: adaptive thinking with high effort, plus interleaved thinking auto-enabled on Sonnet 4.6 — it benefits from re-planning after each test result. Customer-support agent: adaptive thinking with medium effort. The single-shot nature means interleaved thinking adds nothing; the per-question complexity warrants moderate generation-time deliberation. Don't crank both to maximum on every workload — the cost compounds and the quality gain plateaus.)
:::

---

## Section 3 — The Core Reflection Loop

The platform features in Section 2 give you thinking *within* a model call or between tool calls. Reflection as a *loop pattern* — generate → critique → revise → repeat — is something you build on top.

The canonical structure from Reflexion / Self-Refine:

```python
async def reflection_loop(
    initial_prompt: str,
    *,
    max_iterations: int = 3,
    quality_threshold: float = 0.85,
    model: str = "claude-sonnet-4-6",
) -> dict:
    """
    Generate → critique → revise → repeat, until the critic is satisfied
    or we hit the iteration cap.
    """
    history = []
    candidate = await generate(initial_prompt, model=model)

    for iteration in range(max_iterations):
        critique = await critique_response(initial_prompt, candidate, model=model)
        history.append({
            "iteration": iteration,
            "candidate": candidate,
            "critique": critique,
        })

        if critique.score >= quality_threshold and not critique.required_changes:
            return {
                "final": candidate,
                "iterations_used": iteration + 1,
                "history": history,
                "converged": True,
            }

        candidate = await revise(
            initial_prompt,
            previous=candidate,
            feedback=critique.feedback,
            model=model,
        )

    return {
        "final": candidate,
        "iterations_used": max_iterations,
        "history": history,
        "converged": False,
    }
```

Three pieces, each its own design decision:

### The generator

The model that produces the candidate output. Same as your normal generation step. Nothing special about it.

```python
async def generate(prompt: str, *, model: str) -> str:
    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
```

### The critic

The model that evaluates the candidate. This is where most reflection systems succeed or fail.

```python
from pydantic import BaseModel, Field


class Critique(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    strengths: list[str]
    weaknesses: list[str]
    required_changes: list[str]
    feedback: str


CRITIQUE_SYSTEM_PROMPT = """
You are a rigorous critic evaluating a candidate response to a user request.

Your job is to identify weaknesses precisely. Score the response from 0.0 to
1.0 against these criteria:
- Accuracy: are the claims correct and grounded?
- Completeness: does it address all parts of the request?
- Clarity: is it well-structured and readable?
- Actionability: can the user act on it?

Be specific. Vague feedback like "could be better" produces vague revisions.
Concrete feedback like "the second paragraph contradicts the first paragraph
on competitor pricing" produces targeted revisions.

If the response meets all criteria well, return high score and empty
required_changes. Do not invent flaws to look thorough.
""".strip()


async def critique_response(
    original_prompt: str,
    candidate: str,
    *,
    model: str,
) -> Critique:
    response = await client.messages.create(
        model=model,
        max_tokens=2048,
        system=CRITIQUE_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Original request:\n{original_prompt}\n\n"
                f"Candidate response:\n{candidate}\n\n"
                "Evaluate the candidate and return your critique as JSON."
            ),
        }],
        tools=[{
            "name": "submit_critique",
            "description": "Submit your evaluation.",
            "input_schema": Critique.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": "submit_critique"},
    )
    block = next(b for b in response.content if b.type == "tool_use")
    return Critique(**block.input)
```

Three things to notice:

**Structured output.** The critic returns a typed object, not free-form text. This forces it to commit to a score, list specific strengths/weaknesses, and identify discrete required changes. The structure constrains hand-waving.

**Specificity in the prompt.** The instruction "concrete feedback... produces targeted revisions" is doing real work. Vague critics produce vague feedback; vague feedback produces wandering revisions. Sam's evaluator from the cold open was producing exactly this kind of vague output.

**The "do not invent flaws" instruction.** Critics that are too aggressive produce *false negatives* — they reject candidates that were fine. The instruction calibrates the critic toward honesty, not maximum criticism.

### The reviser

The model that takes the critique and produces a revised candidate.

```python
REVISION_SYSTEM_PROMPT = """
You are revising a previous response based on critic feedback. Your job is
to address the specific weaknesses identified, while preserving what was good.

Do not rewrite from scratch unless the critic specifies. Targeted revision
beats wholesale replacement.
""".strip()


async def revise(
    original_prompt: str,
    previous: str,
    feedback: str,
    *,
    model: str,
) -> str:
    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        system=REVISION_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Original request:\n{original_prompt}\n\n"
                f"Previous response:\n{previous}\n\n"
                f"Critique feedback:\n{feedback}\n\n"
                "Produce a revised response addressing the feedback."
            ),
        }],
    )
    return response.content[0].text
```

The key discipline: **targeted revision over wholesale rewrite.** A reviser that throws away the previous candidate and starts over loses the parts that were good and burns tokens regenerating them. A reviser that targets the specific weaknesses preserves quality.

### Convergence and termination

Two termination conditions:

- **Quality threshold met.** The critic returns a score above the threshold and empty `required_changes`. Stop and return.
- **Iteration cap hit.** The loop has run `max_iterations` times. Stop and return whatever's in hand.

The `converged` field in the return value tells the caller which case happened. Non-converged outputs may need different handling — escalation to a human, fallback to a stronger model, or a clear "best effort" signal.

The Section 7 cost discussion will tell you how to set the iteration cap. For now: 2 or 3 is the right default. Diminishing returns set in fast.

::: pullquote
Targeted revision over wholesale rewrite. The reviser's job is to fix the specific weaknesses identified — not to throw away the previous output and start over. Wholesale rewrites lose the good parts and waste tokens.
:::

---

## Section 4 — The Evaluator-Optimizer Pattern, Hardened

M3 introduced the evaluator-optimizer workflow: generator produces output, evaluator scores it, generator revises until the evaluator is satisfied. Sam's brief evaluator was an instance of this pattern.

The cold open showed the failure mode: the *evaluator itself* was inconsistent. Same input, different verdicts. The optimizer was chasing a moving target.

The fix has three parts: making the evaluator more consistent, making its feedback more actionable, and adding the structural discipline that prevents the loop from degenerating.

### Part 1: Make the evaluator more consistent

The single biggest fix for evaluator inconsistency is **explicit, decomposed criteria**. Sam's evaluator was using a holistic prompt: "rate the brief 0-10 with feedback." That asks the model to compress many dimensions of quality into one score, which is exactly the kind of task where models produce different verdicts on different runs.

The fix: decompose into specific criteria, each scored independently, then combine.

```python
class BriefEvaluation(BaseModel):
    accuracy_score: float = Field(ge=0.0, le=1.0)
    accuracy_notes: str

    completeness_score: float = Field(ge=0.0, le=1.0)
    completeness_notes: str

    clarity_score: float = Field(ge=0.0, le=1.0)
    clarity_notes: str

    actionability_score: float = Field(ge=0.0, le=1.0)
    actionability_notes: str

    overall_score: float = Field(ge=0.0, le=1.0)
    required_changes: list[str]
    optional_improvements: list[str]


BRIEF_EVALUATOR_PROMPT = """
You are evaluating a deal-research brief produced for an account executive.
Score the brief on four independent criteria, then compute an overall score.

ACCURACY (0.0-1.0):
- Are all factual claims grounded in the research findings provided?
- Are sources cited correctly?
- Are numbers, dates, names accurate?

COMPLETENESS (0.0-1.0):
- Does it cover all required sections (Executive Summary, Financial Profile,
  Leadership, Market Position, Recent Developments, Key Risks)?
- Is each section adequately developed (150-300 words)?
- Are gaps in research findings explicitly noted?

CLARITY (0.0-1.0):
- Is it well-structured with clear section headers?
- Is the writing concise and direct?
- Could a busy reader extract the key points in 5 minutes?

ACTIONABILITY (0.0-1.0):
- Does it surface specific, useful information for a customer call?
- Are risks and opportunities flagged clearly?
- Could the AE walk into a call better-prepared after reading?

OVERALL: weighted average. Accuracy 35%, completeness 25%, clarity 20%,
actionability 20%.

REQUIRED CHANGES: things that MUST be fixed before the brief is shippable.
Be specific. "Improve competitive analysis" is not specific. "Section 4 lacks
discussion of how the target's pricing compares to the two named competitors"
is specific.

OPTIONAL IMPROVEMENTS: things that would improve the brief but are not blockers.
""".strip()
```

This single change — decomposing the score — improves consistency dramatically. Each sub-score has a narrower target. Combining them is deterministic. The variance between runs drops because no single dimension carries all the weight.

Sam re-ran the 20-call experiment with the decomposed evaluator. Standard deviation dropped from 1.3 to 0.4. The two-cluster bimodal distribution disappeared. Same brief now got consistently 7.0 ± 0.4.

### Part 2: Make feedback actionable

The second fix: **specific feedback over vague suggestions.** "Improve competitive analysis" is unactionable; the optimizer can't tell what to do. "Section 4 doesn't compare the target's pricing to the two named competitors mentioned in section 1" is actionable; the optimizer knows exactly what to fix.

The system prompt above already encodes this. The Pydantic schema enforces it: `required_changes` is a list of distinct items, each (per the prompt) phrased specifically.

The downstream effect: the optimizer's revisions converge faster because each iteration has clear targets. Sam's loop went from 4-5 iterations on average (often non-convergent) to 1-2 iterations (almost always convergent on iteration 2).

### Part 3: Prevent loop degeneration

A subtle failure mode of evaluator-optimizer loops: **the evaluator and optimizer collude**. After enough iterations, the optimizer learns the evaluator's idiosyncratic preferences (because both are LLMs with shared training distributions). The brief becomes increasingly tuned to the evaluator's quirks, scoring better and better on the evaluator while becoming *worse* in some dimensions the evaluator doesn't measure well.

The defense: **anchor the evaluator in external criteria, not just LLM judgment.**

For Sam's brief evaluator, the external anchors:

- **Schema validation.** All six required sections present? Each within the 150-300 word target? This is deterministic; the LLM evaluator can't drift on it.
- **Entity verification (M10).** All named entities (companies, people, dates, dollar figures) appear in the source research? Not in the source → fail.
- **Citation check.** All claims that should be cited have citations? Citations point to actual source material? Programmatic check.

The combined evaluator becomes:

```python
async def evaluate_brief(brief: str, source_research: str) -> dict:
    # Programmatic checks first — fast, deterministic
    schema_check = check_schema_compliance(brief)
    entity_check = await verify_entities_against_source(brief, source_research)
    citation_check = await verify_citations(brief, source_research)

    # If hard checks fail, no need for the LLM evaluator
    if not schema_check.passed:
        return {"score": 0.0, "feedback": schema_check.violations, "type": "structural"}
    if entity_check.unverified:
        return {"score": 0.3, "feedback": f"Unverified entities: {entity_check.unverified}",
                "type": "factual"}

    # LLM evaluator only for subjective quality
    llm_eval = await llm_evaluate(brief, source_research)
    return {
        "score": llm_eval.overall_score,
        "feedback": llm_eval.required_changes,
        "subscores": {
            "accuracy": llm_eval.accuracy_score,
            "completeness": llm_eval.completeness_score,
            "clarity": llm_eval.clarity_score,
            "actionability": llm_eval.actionability_score,
        },
        "type": "subjective",
    }
```

The LLM evaluator now only handles the genuinely subjective dimensions. Structure, facts, and citations get checked deterministically. The collusion failure mode goes away because the deterministic checks can't be gamed — either the section is there or it isn't, either the entity appears in the source or it doesn't.

::: gotcha
A specific mistake teams make: stacking a critic on top of a generator using the *same model*. Sonnet generates; Sonnet critiques. The shared training distribution means the critic has correlated blind spots with the generator. If the generator confabulates a particular kind of fact, the critic is also less likely to catch that kind of fabrication. The fix: use a *stronger* critic than generator (Opus critiquing Sonnet output), use a critic with *different* prompts that surface different perspectives, or — when stakes warrant — use multiple critics with different framings (Section 6).
:::

---

## Section 5 — The Limits of Reflection

Reflection works. Reflection also fails in specific, predictable ways. Knowing the failure modes is what separates "applies reflection" from "applies reflection well."

### Limit 1: The blind-spot problem

The most cited failure mode in the 2024–2026 literature. Single-agent reflection tends to repeat earlier misconceptions on hard examples. The MAR paper (December 2025) found this empirically: *"self reflections tend to repeat earlier misconceptions and do not introduce new reasoning paths on difficult examples."*

The mechanism: when the model produces an output and is then asked to critique it, the critique runs on the same context, with the same model, drawing on the same patterns. The model's blind spots — places where its training didn't distinguish a plausible answer from a correct one — are blind in *both* the generation and the critique. Asking the same model to find what it can't see doesn't help.

The empirical result: reflection produces meaningful gains on tasks where the model has the right knowledge but applied it sloppily; minimal gains on tasks where the model's underlying capability is the bottleneck. The 6.2-point gain MAR reported on HumanEval pass@1 (76.4 → 82.6) over single-agent Reflexion came from explicitly diversifying the critique perspectives — multiple agents with different reasoning personas, exactly to dodge the shared blind-spot problem.

The defenses:

- **Use a stronger model as critic.** Opus 4.7 critiquing Sonnet 4.6 catches blind spots Sonnet doesn't have. Cost goes up; quality follows. M6's stronger-as-critic pattern earns its place here.
- **Different prompts surface different perspectives.** If the generator was prompted as a "helpful assistant," prompt the critic as a "rigorous fact-checker" or "skeptical reader." Different framings activate different reasoning patterns.
- **External anchors (Section 4 Part 3).** Programmatic checks find what no LLM critic can find — structural problems, entity fabrication, citation gaps.
- **Multi-agent reflection (Section 6).** Multiple critics with different personas. Most expensive; highest ceiling.

### Limit 2: Diminishing returns

The 2026 design patterns guide cited in research observed: *"In the author's experience, returns diminish after two to three iterations for code generation and summarization tasks, though analytical or research-heavy domains sometimes benefit from a fourth pass."*

This matches Sam's own experience after rebuilding the evaluator. Iteration 1 typically produces meaningful improvement. Iteration 2 sometimes catches issues the first iteration missed. Iteration 3 occasionally helps. Iteration 4+ rarely helps and often makes things subtly worse (over-tuning, drift).

The implication: **set hard iteration caps.** Two iterations as the default. Three for analytical work. Don't let the loop run longer hoping for more improvement; set a hard ceiling and accept the output if it's not done by then.

### Limit 3: Reflection on fundamentally wrong outputs

Reflection improves *partially-correct* output toward more-correct output. It does not fix *fundamentally wrong* output.

If the generator produced a brief that's structured around the wrong company (entity confusion early in research), reflection iterates on the wrong-company brief. The critic might catch a typo or a clarity issue, but it can't say "this is about Mercer Industries when it should be about Mercer Healthcare" unless the critic has access to the right ground truth — and if it has that, you should have wired in entity verification (M10 Pattern A) at the entity stage rather than trying to catch it via reflection.

The implication: **reflection is not a substitute for upstream correctness.** The hallucination disciplines from M10, the tool design from M4, the context engineering from M7-M9 — those have to work first. Reflection is the layer above; it improves outputs that are mostly right and might be wrong in specific ways. It doesn't rescue outputs that are categorically wrong.

### Limit 4: Cost compounds

Every reflection iteration is another LLM call. A 3-iteration reflection loop costs roughly 3× the unreflected single call (one generation + two reflections + two revisions = five calls, but with shorter critique calls than full generations). For high-volume systems, this matters.

The cost calibration:

- **Cheap workloads with low stakes.** No reflection. Single-shot generation; accept the output as-is.
- **Medium workloads with medium stakes.** One iteration of reflection (generate → critique → revise once). Captures most of the benefit at modest cost.
- **High-stakes workloads.** Two or three iterations, with stronger critic models. Pay for the quality.

The mistake to avoid: applying reflection prophylactically. Section 7 expands on the cost calibration framework.

### Limit 5: When reflection makes things worse

The most concerning failure mode: reflection produces *worse* output than no reflection at all. This happens in specific cases:

- **The critic is biased toward criticism.** A critic prompted to "find every flaw" finds flaws even when the output is good. The reviser then changes things that didn't need changing, often in ways that introduce new problems.
- **The critic and reviser share idiosyncratic preferences.** Both prefer formal tone, both insert equivocation, both remove specific examples in favor of generalities. Multiple iterations make the output progressively more generic.
- **The reviser overcorrects.** Critique mentions one weakness; reviser rewrites that section completely, losing context the original had right.
- **The critic hallucinates flaws.** The critique points to a problem that isn't there; the reviser dutifully addresses it, introducing actual problems.

The defense: **measure end-to-end quality, not just consistency.** A reflection loop that produces consistent output isn't necessarily producing *good* output. Sam's hardened evaluator from Section 4 was much more consistent than the original — but the team also ran a quality eval (independent rubric, manual review) to verify the consistency was tracking quality, not just convergence. Without that check, the loop could have been confidently producing worse-and-worse output and the metrics would have shown improvement.

::: postmortem
**The Reflection Loop That Made Things Worse**

A team had a customer-email-drafting agent. They added a reflection step: the agent would draft an email, critique it, revise, and send. The critique prompt asked the model to "identify ways the email could be more empathetic and professional."

After deploying the reflection step, complaints from customer-facing reps went up. The emails were technically polished, but they had become *generic* — they no longer mentioned specific customer details that the original drafts had included. The team pulled traces.

What was happening: the critic model was suggesting changes like "could be more empathetic by acknowledging the customer's situation broadly" — generic, professional-sounding additions. The reviser implemented these by replacing customer-specific phrasing with the generic framings. After 2-3 iterations, every email read like it had been written by a corporate communications template.

Two fixes shipped:

1. **Critic prompt rewritten.** New instruction: "Identify weaknesses precisely. Do not suggest generic improvements. Specific feedback only — name the exact phrase that should change and why."

2. **Specificity preservation rule.** The reviser was instructed: "Do not replace specific customer details (names, situations, account specifics) with generic framings. Specificity is more empathetic than generic professionalism."

After the fixes, the rep complaints stopped. Email quality (measured by an independent rubric) recovered. The reflection loop now improved emails rather than genericizing them.

**Lesson:** A reflection loop is only as good as its critic. A vague or biased critic produces vague or biased revisions. Watch end-to-end quality, not just whether the loop converges. If the loop is converging on something worse than where it started, the critic is the problem.
:::

---

## Section 6 — Multi-Agent Reflection

The blind-spot problem from Section 5 motivated the field's move toward multi-agent reflection. Single-agent reflection has correlated blind spots with the generator. Multi-agent reflection uses multiple critics with different framings to catch what one critic misses.

This is a natural extension of M3's evaluator-optimizer pattern, but with multiple evaluators. We'll preview it briefly here; M13 covers multi-agent architectures in depth.

### The pattern

```python
async def multi_critic_reflection(
    candidate: str,
    original_prompt: str,
    *,
    critics: list[dict],  # [{name, prompt, model}]
) -> dict:
    """
    Run multiple critics in parallel, each with a different framing.
    Aggregate the critiques into combined feedback.
    """
    individual_critiques = await asyncio.gather(*[
        run_critic(candidate, original_prompt, **critic_config)
        for critic_config in critics
    ])

    # Aggregate: pool all required_changes, deduplicate, rank by frequency
    all_changes = []
    for critique in individual_critiques:
        all_changes.extend(critique.required_changes)
    aggregated = aggregate_changes(all_changes)

    return {
        "individual_critiques": individual_critiques,
        "aggregated_changes": aggregated,
        "consensus_score": sum(c.score for c in individual_critiques) / len(critics),
    }


CRITICS_FOR_BRIEF = [
    {
        "name": "factuality",
        "prompt": "Critique this brief focusing on factual accuracy and citation grounding.",
        "model": "claude-opus-4-7",
    },
    {
        "name": "audience_fit",
        "prompt": (
            "Critique this brief from the perspective of a busy AE who will read it "
            "in 5 minutes. Is it useful for prepping a customer call?"
        ),
        "model": "claude-sonnet-4-6",
    },
    {
        "name": "completeness",
        "prompt": (
            "Critique this brief for completeness. Are required sections present? "
            "Are gaps acknowledged? Are competitor analyses substantive?"
        ),
        "model": "claude-sonnet-4-6",
    },
]
```

Three critics, three perspectives:

- **Factuality** — strict on fabrication, citations, accuracy
- **Audience fit** — practical, "would the AE find this useful"
- **Completeness** — structural, all-bases-covered

The MAR paper's finding: this kind of multi-perspective critique consistently produces better revisions than single-perspective critique. The blind spots of any one critic are often visible to another.

### Cost vs benefit

Multi-critic reflection is expensive — N critic calls per iteration, each potentially using a stronger model. A 3-critic loop with 2 iterations runs 7+ calls per request (1 generation + 3 × 2 critiques + 2 revisions).

When justified:

- **High-stakes outputs** — legal briefs, medical recommendations, financial analyses
- **Outputs with multi-dimensional quality** — where "good" requires several different things simultaneously
- **Workflows where the cost of being wrong dwarfs the cost of computation**

When not justified:

- **High-volume routine outputs** — most customer support replies, standard summaries
- **Tasks where one critic's framing covers the relevant dimensions** — schema-validated extraction, simple Q&A
- **Latency-sensitive applications** — multi-critic adds 2-3× latency

The principle from Module 6 applies: match the verification cost to the stakes. Multi-critic reflection is the ceiling; most workloads don't need to reach for it.

### The judge model pattern

A variation that's become popular: instead of aggregating critiques programmatically, have a *judge* model synthesize the multiple critiques into unified feedback.

```python
async def judge_synthesize(
    candidate: str,
    individual_critiques: list,
    *,
    model: str = "claude-opus-4-7",
) -> str:
    response = await client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": (
                f"Candidate response:\n{candidate}\n\n"
                f"Critiques from multiple reviewers:\n"
                + "\n\n".join(f"=== {c.name} ===\n{c.feedback}" for c in individual_critiques)
                + "\n\n"
                "Synthesize these critiques into a single, coherent feedback document. "
                "Identify which concerns are most important. Resolve disagreements between "
                "reviewers — for example, if one reviewer says 'too detailed' and another "
                "says 'lacking specifics,' adjudicate. Output the unified feedback."
            ),
        }],
    )
    return response.content[0].text
```

The judge resolves contradictions between critics, prioritizes the most important issues, and produces a single coherent revision target. The MAR paper found judge-synthesized critiques produced better revisions than naively aggregated ones.

This is the multi-agent pattern getting close to a full multi-agent architecture — generator, multiple critics, a judge. Module 13 will formalize this.

::: nodumbq
**Q: Multi-agent reflection sounds expensive. When is it actually worth it?**

The math: a single-critic 2-iteration loop runs ~5 calls. A 3-critic 2-iteration loop with a judge runs ~10 calls. So roughly 2× cost for multi-agent. The quality difference: MAR reported a 6.2-point pass@1 gain on HumanEval; on subjective tasks like brief writing, the gains are usually 5-15% by independent rubric scoring. So 2× cost for ~10% quality lift is the typical trade-off. Worth it for high-stakes workloads (legal, financial); usually not for routine ones.

**Q: Can I get the multi-perspective benefit without multiple critics?**

Partially. Asking a single critic to evaluate from multiple framings in one call ("first evaluate for factual accuracy, then for audience fit, then for completeness, then synthesize") captures some of the benefit at lower cost. But the shared-context limitation (one model, one inference) means it's not as good as truly separate critic calls. It's a middle-ground option for cases where multi-agent is too expensive but single-agent is too narrow.
:::

---

## Section 7 — Cost Calibration: Hard Caps and Budgets

Reflection burns tokens. Without explicit caps, it can burn many tokens.

### The iteration cap

Hard limit on the number of generate-critique-revise cycles. The 2026 design patterns guide:

> *"Without a maximum iteration cap, reflection loops can cycle indefinitely, burning tokens without improving output. Returns diminish after two to three iterations for code generation and summarization tasks, though analytical or research-heavy domains sometimes benefit from a fourth pass."*

Defaults that work:

- **Routine outputs:** 1 iteration (generate → critique → revise once → ship). Captures most of the gain at minimum cost.
- **Standard production:** 2 iterations. The default for medium-stakes work.
- **High-stakes outputs:** 3 iterations max. Usually converges in 2; the third is a buffer.
- **Research / analysis:** 4 iterations. The exception case.

Anything beyond 4 is almost certainly diminishing-or-negative returns. If the loop hasn't converged by then, escalate (human review, stronger model, different approach).

### Token budget per cycle

Per-iteration token cap, separate from total. Prevents one runaway critique or revision from dominating the cost.

```python
ITERATION_TOKEN_BUDGET = 4096  # max output tokens per critic/reviser call


async def critique_with_budget(prompt: str, candidate: str, model: str) -> Critique:
    response = await client.messages.create(
        model=model,
        max_tokens=ITERATION_TOKEN_BUDGET,
        ...
    )
    return ...
```

A critic that produces a 10,000-token critique is signaling that something's wrong with the prompt design — the critic shouldn't need that many tokens to identify weaknesses. Cap it; if outputs get cut off, fix the prompt.

### Total budget per request

Cumulative ceiling across all iterations. Prevents the loop from running 4 iterations × multi-critic × stronger models from blowing past your unit-economics target.

```python
async def reflection_with_budget(
    initial_prompt: str,
    *,
    max_iterations: int = 2,
    total_token_budget: int = 32000,
    ...
) -> dict:
    tokens_used = 0
    iterations_completed = 0

    while iterations_completed < max_iterations and tokens_used < total_token_budget:
        critique = await critique_response(...)
        tokens_used += critique.usage_tokens

        if tokens_used >= total_token_budget:
            break  # budget exhausted; ship what we have

        if critique.score >= threshold:
            return {"final": candidate, "tokens_used": tokens_used, ...}

        candidate = await revise(...)
        tokens_used += revise.usage_tokens
        iterations_completed += 1
```

Two terminal conditions: convergence (good) and budget exhaustion (acceptable). Both produce a usable output. Neither lets the loop run wild.

### When to skip reflection entirely

Some workloads don't earn reflection. The decision criteria:

- **Cost of error is low.** The user can just ask again if the answer is bad.
- **Single-shot quality is already high.** If the unreflected baseline meets quality bar, reflection is overhead.
- **Latency budget is tight.** Sub-second response targets don't accommodate reflection.
- **Volume is high and stakes are low.** A high-volume customer-acknowledgment generator probably doesn't need reflection per request.

Sam's deal-research evaluator earned reflection because its outputs gated downstream optimizer iterations. The customer-success bot's individual responses generally didn't earn reflection — they're high-volume, conversational, with cheap-to-fix errors. Different workloads, different reflection budgets.

::: brain
You have an agent that generates legal contract summaries for review by lawyers. Average input length 50 pages; average output length 500 words; 100 requests per day. Should you add reflection? With what budget?

(Yes, definitely. High stakes (legal review), low volume (100/day means cost compounds slowly), high cost-of-error (a missed clause has six-figure consequences). Use 2-3 iterations of single-critic reflection with Opus as both generator and critic, plus entity verification on named clauses. Total cost per request might be 10× a single Sonnet call — but at 100 requests/day, that's still affordable, and the quality lift is worth the spend.)
:::

---

## Section 8 — Putting It Together: Sam's Evaluator Rebuilt

Returning to the cold open. Sam rebuilt the brief evaluator using everything in this module:

**Architecture:**

```python
async def evaluate_brief_v2(
    brief: str,
    source_research: str,
    *,
    max_revisions: int = 2,
) -> dict:
    # Programmatic structural checks (deterministic, fast)
    schema_check = check_schema_compliance(brief)
    if not schema_check.passed:
        return {
            "decision": "reject",
            "reason": "structural",
            "details": schema_check.violations,
        }

    # Entity and citation verification (M10 Pattern A)
    entity_check = await verify_entities_against_source(brief, source_research)
    if entity_check.unverified:
        return {
            "decision": "reject",
            "reason": "factual",
            "details": entity_check.unverified,
        }

    citation_check = await verify_citations(brief, source_research)
    if citation_check.unverified:
        return {
            "decision": "reject",
            "reason": "citations",
            "details": citation_check.unverified,
        }

    # LLM evaluator with decomposed criteria (Section 4 Part 1)
    llm_eval = await llm_evaluate_decomposed(brief, source_research)

    return {
        "decision": "accept" if llm_eval.overall_score >= 0.75 else "revise",
        "reason": "subjective_quality",
        "score": llm_eval.overall_score,
        "subscores": {
            "accuracy": llm_eval.accuracy_score,
            "completeness": llm_eval.completeness_score,
            "clarity": llm_eval.clarity_score,
            "actionability": llm_eval.actionability_score,
        },
        "required_changes": llm_eval.required_changes,
        "optional_improvements": llm_eval.optional_improvements,
    }
```

**The optimizer loop:**

```python
async def write_brief_with_reflection(
    research: dict,
    *,
    max_revisions: int = 2,
) -> dict:
    candidate = await generate_brief(research)
    history = []

    for iteration in range(max_revisions + 1):
        evaluation = await evaluate_brief_v2(candidate, research)
        history.append({"iteration": iteration, "evaluation": evaluation})

        if evaluation["decision"] == "accept":
            return {
                "final": candidate,
                "evaluation": evaluation,
                "iterations_used": iteration,
                "history": history,
                "converged": True,
            }

        if evaluation["decision"] == "reject":
            # Hard reject (structural / factual / citation) — fix the upstream issue
            return {
                "final": None,
                "evaluation": evaluation,
                "history": history,
                "converged": False,
                "escalation_needed": True,
            }

        # decision == "revise" — actionable subjective feedback
        if iteration < max_revisions:
            candidate = await revise_brief(
                research,
                previous=candidate,
                feedback=evaluation["required_changes"],
            )

    return {
        "final": candidate,
        "evaluation": evaluation,
        "iterations_used": max_revisions,
        "history": history,
        "converged": False,
    }
```

**What changed from the broken version:**

1. **Decomposed evaluator** — four sub-scores instead of one holistic score. Variance dropped from 1.3 to 0.4 standard deviation.
2. **Programmatic anchors** — schema, entity, and citation checks happen first. The LLM evaluator only handles subjective quality.
3. **Specific feedback discipline** — `required_changes` is a list of specific items, each phrased actionably.
4. **Iteration cap** — max 2 revisions. Convergence usually happens in 1.
5. **Three terminal states** — accept (ship), revise (iterate), reject (escalate). Clear handling for each.

**The before-and-after metrics** (Sam ran this on a sample of 50 briefs):

| Metric | Before | After |
|---|---|---|
| Evaluator standard deviation on same brief | 1.3 | 0.4 |
| Average iterations to convergence | 4.2 | 1.6 |
| Non-convergent runs (loop hits cap) | 35% | 4% |
| Quality (independent rubric, 0-10) | 6.4 | 7.8 |
| Tokens per brief end-to-end | 28K | 18K |
| End-to-end latency | 47s | 22s |

Better quality, fewer iterations, lower cost, faster latency. The hardened reflection loop wasn't *more* expensive than the broken one — it was *less* expensive, because it converged faster and didn't thrash.

Sam wrote a one-line lesson: *"a sloppy evaluator costs more than a rigorous one. The rigor is the optimization."*

::: code-exercise
**Exercise 11.1 — Reflection audit.**

Pick a workflow in your stack that has an evaluator-optimizer loop or any reflection step. Run the consistency test:

1. **Pick a representative input.** A real production input (or test input) that's representative of typical work.
2. **Run the evaluator 20 times** on the same candidate output. Plot the score distribution.
3. **If standard deviation is > 0.3 on a 0-1 scale**, you have an evaluator-consistency problem. Apply Section 4 Part 1 (decomposed criteria) and re-test.
4. **Look at the feedback the evaluator produces.** Is it specific or vague? Could a generator act on it concretely? If not, apply Section 4 Part 2 and re-test.
5. **Audit one converged loop end-to-end.** What does the brief look like at iteration 1 vs iteration 3? Is it actually better, or is it different in ways that don't matter?
6. **Measure the quality with an independent rubric.** Don't trust the evaluator to evaluate itself. Use a different framing or a stronger model.

Most teams ship their first reflection loop and never run this audit. The audit reveals exactly the failure modes from Section 5 (blind spots, generic genericization, criticism for criticism's sake). Once you see them, the fixes are straightforward.
:::

---

## Section 9 — The Framework

Adding reflection discipline to a system you're building:

**1. Diagnose whether you need reflection.** Cost of error vs cost of inference. Volume vs stakes. Single-shot quality vs reflection-improved quality. Don't add reflection prophylactically.

**2. Pick the right Anthropic surface.** Adaptive thinking for generation-time deliberation on hard tasks. Interleaved thinking for inter-turn deliberation in agent loops. The "think" tool for compliance work or Haiku-tier auditable reasoning. Match the surface to the workload.

**3. Build the core reflection loop with structured critics.** Pydantic schemas for critique output. Specific feedback discipline in the prompt. Targeted revision over wholesale rewrite. Iteration cap (default 2).

**4. Decompose evaluator criteria.** Single holistic scores produce inconsistent verdicts. Multiple sub-scores combined deterministically produce stable evaluation.

**5. Anchor in external criteria.** Programmatic checks for structure, entities, citations. The LLM evaluator handles only the genuinely subjective. Prevents collusion between evaluator and optimizer.

**6. Watch for the limit failure modes.** Blind-spot reinforcement (use stronger or differently-framed critics). Diminishing returns (cap iterations). Genericization (preserve specificity). Worse-than-no-reflection outcomes (measure end-to-end quality, not just convergence).

**7. Reach for multi-agent reflection only when warranted.** High-stakes, multi-dimensional quality, error costs that justify 2-3× compute. Most workloads don't need it.

**8. Cap costs explicitly.** Iteration limits. Per-call token budgets. Total budget per request. Reflection that runs unbounded is reflection that ships products at unit-economics-killing cost.

The discipline this enforces: **reflection is a deliberate intervention, not a default.** Applied to the right workloads with the right structure, it produces meaningful quality lift. Applied unreflectively, it inflates costs without proportional benefit — and sometimes makes things worse.

---

## Recap: Module 11 in eight bullets

::: bullet-points
- Reflection is the family of patterns that give the model a chance to evaluate its own work before that work becomes load-bearing. Includes generation-time (CoT, extended thinking), post-generation (Reflexion, Self-Refine), inter-turn (interleaved thinking), and cross-run (memory-backed lessons).
- Anthropic's first-party surfaces: adaptive thinking (default on Sonnet 4.6 / Opus), interleaved thinking (auto-enabled on Opus 4.7), the "think" tool (auditable reasoning, especially useful on Haiku). Pick the surface that matches your workload's structure.
- The core reflection loop: generate → critique → revise → repeat with iteration cap. Pydantic-typed critique output with specific feedback. Targeted revision over wholesale rewrite. Diminishing returns set in fast — 2 iterations is usually the right default.
- Evaluator-optimizer pattern hardening: decompose holistic scores into sub-criteria (drops variance ~3×), make feedback actionable (specific items, not vague suggestions), anchor in programmatic checks (entities, citations, schema) so the loop can't collude.
- Limits of reflection: blind-spot reinforcement (single-agent reflects on its own context), diminishing returns after 2-3 iterations, can't fix fundamentally-wrong output, and sometimes produces worse results (genericization, over-criticism). Watch for these.
- Multi-agent reflection (preview of M13) addresses blind-spot reinforcement at higher cost. Multiple critics with different framings, sometimes a judge model synthesizing. Justified for high-stakes outputs; overkill for routine ones.
- Cost calibration is non-negotiable. Iteration caps (2 default, 3 max for analytical, 4 absolute ceiling). Per-call token budgets. Total budget per request. Without caps, reflection burns tokens indefinitely.
- Sam's evaluator rebuild produced better quality at lower cost: variance dropped 3×, iterations dropped from 4.2 to 1.6, tokens dropped from 28K to 18K, quality went from 6.4/10 to 7.8/10. The rigor is the optimization.
:::

---

::: sam-arc
**Sam, after the evaluator rebuild.**

Two weeks of work. Tuesday afternoon when the broken evaluator was discovered to Friday two weeks later when the rebuilt version shipped.

The biggest learning wasn't a technical one. The biggest learning was that *reflection had been hiding the underlying problem*. The original evaluator's inconsistency had made the optimizer thrash, and the thrashing had been read by the team as "reflection just adds variance, what can you do." The fix wasn't to remove reflection. The fix was to make the evaluator deserve its role.

Sam thought about this on the walk home that Friday. The first ten modules of work — loops, workflows, tools, MCP, models, context, compaction, memory, hallucinations — had all been about *upstream correctness*. Make the things that produce output as correct as possible. Module 11 was the first module that was about *epistemic discipline* — knowing what you know, knowing what you don't, building processes that surface the difference.

The reflection loop, done right, was that discipline encoded in code. The evaluator wasn't perfect, so it was decomposed. The feedback wasn't actionable, so it was schema-constrained. The loop could collude with itself, so it was anchored in programmatic checks. Each fix was an admission that the previous version was making claims it couldn't back up — and a refactor that aligned the system's behavior with its actual capabilities.

Sam's arc this module: **the rigor is the optimization.** The hardened loop, the structured workflows, the careful tools, the compactions and memories — they all had this same shape. A version that worked sometimes, a discovery of why it was sometimes-wrong, and a refactor that made the discipline match the requirement. M11 was that pattern applied to the agent's own self-evaluation.

The deal-research workflow was now operating at a level Sam could defend in a code review: every model call had an explainable purpose, every reflection step had a defensible justification, every iteration cap had a reason. The system was no longer just *working*; it was *auditable*.

Two more reliability modules to go. Then the multi-agent arc. The book was building toward something Sam could see now: a system where every layer was honest about what it was doing, and the layers composed without gaps. M12 — guardrails at the boundaries — was next.
:::

---

## What's next

Module 12 is guardrails. Input filtering, output checks, topical and behavioral constraints. The cousin of Module 5's protocol-level security, applied at the per-call layer. The agent that's been told "don't fabricate citations" in M10 still needs an output-time check that *verifies* it didn't fabricate one. M12 is where those checks live — the boundary discipline that catches what the upstream layers missed.

After M12, the reliability arc closes (M10 hallucinations, M11 reflection, M12 guardrails). Then M13–15 take the system multi-agent (when complexity is justified, the architectures, the patterns). M16–17 are operations (sagas, observability). M18 is the capstone.

For now: take the audit exercise from Section 8. Run the 20-call consistency test on any reflection step in your system. The variance you see will tell you whether you're using reflection or just paying for it. The fix path is in Section 4. The rigor is the optimization — for evaluators, for loops, for the whole system.
