# Module 10 Outline — Hallucinations and Grounding

::: chapter-opener
<div class="module-num">MODULE 10 — OUTLINE</div>
<div class="module-title">Hallucinations and Grounding</div>
<div class="subtitle">Most agent failures look like hallucinations.<br>Most "hallucinations" are retrieval misses, compression losses, or ungrounded reasoning.<br>Let's pull them apart.</div>
<div class="pages">Target length: ~36 pages</div>
:::

## What this module is

"Hallucination" has become a catch-all word for any time an agent gets something wrong. The catch-all hides what's actually happening. This module separates the failure modes — factuality vs faithfulness, intrinsic vs extrinsic, retrieval-induced vs reasoning-induced, single-agent vs multi-agent communication hallucinations — and provides specific defenses for each.

We cover the detection methods that actually work in production: self-consistency (with semantic entropy), Best-of-N reranking with a faithfulness scorer, LLM-as-judge faithfulness, Cross-Layer Attention Probing (CLAP) for white-box, MetaQA for black-box. We also cover the defenses: grounding through retrieval, citation enforcement, abstention calibration, multi-agent verification cycles.

By the end the reader can: classify a hallucination they've seen, pick the right detection technique for their access pattern, and add hallucination defenses to the agent without nuking latency or cost.

::: hook
"Your agent confidently told a customer their order shipped on April 12. The order shipped on April 21. Was that a hallucination? Probably not — probably the model retrieved the wrong order, or your tool returned stale data, or the compressor dropped the date. 'Hallucination' is the symptom. The diagnosis is harder."
:::

---

## Section 1 — A Taxonomy That Actually Helps (≈4 pages)

The literature uses several axes. We pick the two that map cleanly to defenses.

**Axis 1: Factuality vs Faithfulness.**

- **Factuality** — does the output match the real world? "The Eiffel Tower is in London" is factually wrong.
- **Faithfulness** — does the output match the source/context the model was given? An agent given a document and asked to summarize can be faithful (matches the doc) and unfactual (the doc itself was wrong), or unfaithful (invents claims not in the doc).

Faithfulness is what you can engineer for. Factuality requires ground truth, which often doesn't exist at the moment of generation.

**Axis 2: Intrinsic vs Extrinsic.**

- **Intrinsic** — output is internally inconsistent. "Alice is older than Bob, who is older than Carol, who is older than Alice."
- **Extrinsic** — output contradicts external truth (or the provided source).

Intrinsic hallucinations are detectable from the output alone. Extrinsic ones require comparison.

A practical 2x2:

|              | Intrinsic                          | Extrinsic                        |
|--------------|------------------------------------|----------------------------------|
| Factuality   | Self-contradiction in claims       | Wrong about the world            |
| Faithfulness | Contradicts own reasoning chain    | Contradicts provided source      |

Most "agent hallucinated" reports are extrinsic-faithfulness: the agent gave the wrong answer based on what was retrieved/computed. Defenses live mostly in this quadrant.

::: pullquote
You can't engineer for factuality without ground truth. You can engineer for faithfulness with discipline. Pick the fight you can win.
:::

::: nodumbq
**Q: Are agents more or less hallucinatory than chat models?**

Different. Chat models hallucinate by inventing answers from training data when they don't know. Agents *retrieve* answers from tools — so they hallucinate less of the "made up out of nowhere" kind. But agents introduce two new kinds: (1) the tool returned wrong/stale data and the agent presented it confidently, and (2) the model misread or selectively cited the tool result. Total hallucination rate often goes down with agents; the *profile* changes.

**Q: I'm using RAG. Doesn't that solve hallucinations?**

It reduces them. The Vectara hallucination leaderboard tracks faithfulness on summarization with provided documents — even top models hallucinate ~1-3% of the time on grounded summaries. Add agent loops, multiple retrieved chunks, and tool results, and the rate goes up. RAG is a defense, not a cure.
:::

---

## Section 2 — Where Agent Hallucinations Actually Come From (≈4 pages)

Six concrete sources, with examples of each:

**1. Tool returned wrong data.** The model is faithful to the tool result, but the result was wrong. Stale cache. Wrong customer ID. Off-by-one. The model didn't hallucinate; it just trusted bad input.

**2. Retrieval miss.** The agent retrieved 5 documents; none contained the answer; the model synthesized something that sounded like an answer. (This is what most "RAG hallucinations" actually are.)

**3. Compression loss.** Discussed in Module 8. The fact was in the conversation, got compressed away, and now the model "remembers" a related but wrong fact.

**4. Tool selection error.** Model called `get_user_orders` when it should have called `get_pending_orders`. Got real data; wrong data.

**5. Confident extrapolation.** Model has 3 facts and infers a 4th that doesn't logically follow. "User is in EU, GDPR applies, therefore user is German." Reasoning hallucination.

**6. Communication hallucination (multi-agent).** Agent A hallucinates a fact, communicates it to Agent B as if certain, Agent B treats it as ground truth. Cited in the Sep 2025 survey "LLM-based Agents Suffer from Hallucinations" — this becomes a structural problem in multi-agent systems and we cover defenses in Module 14.

::: postmortem
**The "Hallucination" That Was a Stale Cache**

A team filed an angry bug: their support agent told a customer their package was at "facility XYZ" when it was actually at "facility ABC." Engineering blamed the model. Spent two weeks adding "more grounding."

Actual cause: the shipping API tool had a 1-hour cache. The package had moved 30 minutes ago. The model faithfully reported what the tool returned. The tool returned stale data. Zero engineering improvements would fix this; the cache was the bug.

**Lesson:** before assuming a hallucination is a model problem, trace upstream. If the input was wrong, the model isn't hallucinating — it's being faithful to wrong input.
:::

---

## Section 3 — Detection: Self-Consistency and Semantic Entropy (≈5 pages)

The first family of detection methods. The idea: if the model is confident, multiple samples converge. If it's hallucinating, samples diverge.

**Self-consistency (Wang et al., ICLR 2023; built upon ever since).** Sample N responses to the same prompt with non-zero temperature. Aggregate.

```python
async def self_consistency_check(client, prompt: str, n: int = 5, temperature: float = 0.7) -> ConsistencyResult:
    samples = await asyncio.gather(*[
        client.messages.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=500,
        ) for _ in range(n)
    ])
    answers = [extract_answer(s.content[0].text) for s in samples]
    
    # If they cluster tightly, model is consistent (likely correct)
    # If they diverge, model is uncertain (likely hallucinating)
    return analyze_consistency(answers)
```

For factual QA, the cluster modes is often the right answer. For complex reasoning, you want **semantic** consistency, not lexical — the answers should mean the same thing even if worded differently.

**Semantic entropy (Farquhar et al., 2024).** Group samples by semantic equivalence using NLI; entropy of the resulting clusters measures uncertainty.

```python
async def semantic_entropy(samples: list[str]) -> float:
    # Cluster semantically equivalent answers
    clusters = await cluster_by_nli_equivalence(samples)
    # Cluster sizes
    sizes = [len(c) for c in clusters]
    total = sum(sizes)
    probs = [s / total for s in sizes]
    return -sum(p * math.log(p) for p in probs)
```

Low entropy → model is confident (one cluster dominates). High entropy → model is uncertain (samples spread across many semantic groups).

**Cost vs benefit.** N=5 samples = 5x cost. Use selectively:
- High-stakes claims (medical, legal, financial)
- Low-confidence outputs (when entropy from a cheap proxy is already high)
- Sampling 1-5% of production traffic for monitoring (Module 17)

::: brain
You're sampling 5 times to detect hallucination. The 5 samples agree. Is the model right?

(Maybe. Self-consistency is a *necessary but not sufficient* condition for correctness. Models can be consistently wrong — particularly on plausible-looking but factually false claims that match training data patterns. Use self-consistency as a screen, not a proof.)
:::

---

## Section 4 — Detection: LLM-as-Judge for Faithfulness (≈4 pages)

When the agent has a source (retrieved doc, tool result, prior turn), faithfulness is checkable by another LLM. This is what production RAG eval frameworks use (Maxim, Vectara leaderboard, jumpcloud's grounding score).

```python
FAITHFULNESS_PROMPT = """You will be given:
1. A SOURCE: text the model was supposed to base its answer on.
2. An ANSWER: what the model produced.

Your task: identify EVERY claim in the ANSWER. For each claim, mark:
- SUPPORTED: explicitly stated in SOURCE
- INFERRED: reasonably implied by SOURCE
- UNSUPPORTED: not in SOURCE
- CONTRADICTED: SOURCE says otherwise

Output JSON: {"claims": [{"text": "...", "verdict": "..."}, ...], "score": <fraction supported>}

SOURCE:
{source}

ANSWER:
{answer}
"""

async def faithfulness_score(judge_client, source: str, answer: str) -> FaithfulnessReport:
    resp = await judge_client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1500,
        messages=[{"role": "user", "content": FAITHFULNESS_PROMPT.format(
            source=source, answer=answer,
        )}],
    )
    return parse_faithfulness(resp.content[0].text)
```

Caveats from research (IUI 2025, "No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding"):
- Judge models have biases (favor verbose answers, their own outputs)
- Use a *different* model as judge (different family if possible)
- Calibrate against human-labeled gold set; don't trust raw scores
- Judge fails on expert domains (clinical, legal) — escalate to humans there

::: gotcha
The classic failure: using the same model as both generator and judge. The judge consistently approves outputs that match its own biases. Use Opus as judge for Sonnet outputs, or a different family entirely (use an open-source judge for closed-model outputs).
:::

---

## Section 5 — Detection: Best-of-N Reranking (≈4 pages)

The pattern from the ACL Findings 2025 study cited in our Module 3 parallelization section. Generate N candidate responses; score each for faithfulness; return the highest-scoring.

```python
async def best_of_n_faithful(client, judge, prompt: str, source: str, n: int = 4) -> str:
    candidates = await asyncio.gather(*[
        client.messages.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=1000,
        ) for _ in range(n)
    ])
    candidate_texts = [c.content[0].text for c in candidates]
    
    # Score each for faithfulness against the source
    scores = await asyncio.gather(*[
        faithfulness_score(judge, source=source, answer=t) for t in candidate_texts
    ])
    
    # Return the most faithful
    best_idx = max(range(n), key=lambda i: scores[i].score)
    return candidate_texts[best_idx]
```

Empirical result (ACL Findings 2025): meaningful error rate reduction with N=4-8, no retraining required. Cost: N× generation + N× judge. Justified for high-stakes outputs; overkill for routine.

A variant that's cheaper: **rejection sampling.** Generate one. If it scores below threshold, generate another. Repeat up to N times.

```python
async def rejection_sample(client, judge, prompt, source, threshold=0.85, max_attempts=4):
    for attempt in range(max_attempts):
        resp = await client.messages.create(...)
        candidate = resp.content[0].text
        score = await faithfulness_score(judge, source, candidate)
        if score.score >= threshold:
            return candidate, score
    # Couldn't satisfy; return best-so-far + abstention signal
    return candidate, score
```

When threshold isn't met after max_attempts, the right move is often to *abstain* — return "I'm not confident enough to answer this" — rather than ship a low-confidence answer.

::: brain
Best-of-N picks the most faithful candidate. But what if all N candidates are unfaithful in the same way? (E.g., they all hallucinate the same plausible-sounding fact.)

(Best-of-N doesn't help against systematic biases — only against random sampling errors. Defenses against systematic hallucination need diverse generators (different prompts, different models) or external verification (retrieval, code execution).)
:::

---

## Section 6 — Detection: White-Box and Black-Box Internal Methods (≈3 pages)

When you have model internals (open-source models, or you're an inference provider): **Cross-Layer Attention Probing (CLAP)**, the Sep 2025 result. Train a lightweight classifier on the model's own attention patterns and hidden states. Predicts hallucinations from internal signals before/during generation, with reported gains of up to 11.7% over baseline non-hallucination rate using their CLAP-I strategy.

When you don't have internals (closed-source APIs): **MetaQA (ACM 2025)**. Apply small mutations to the prompt (rephrase, reorder details, change irrelevant tokens). If the model's answer changes meaningfully, low confidence. If the answer is stable across mutations, higher confidence. Black-box, no token probabilities required.

```python
async def metaqa_check(client, prompt: str, n_mutations: int = 5) -> StabilityScore:
    mutations = [generate_mutation(prompt) for _ in range(n_mutations)]
    answers = await asyncio.gather(*[
        client.messages.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": m}],
            temperature=0.0,
        ) for m in mutations
    ])
    return measure_stability([a.content[0].text for a in answers])
```

Most readers building agents on closed APIs will use MetaQA-style stability checks more than CLAP. We mention CLAP for completeness; reader who runs their own models can lean on it.

::: nodumbq
**Q: Why does temperature=0 in MetaQA when self-consistency uses temperature>0?**

Different goals. Self-consistency wants the model's *uncertainty across samples* — temperature gives it diversity. MetaQA wants the model's *robustness to input phrasing* — temperature should be 0 so any output variation comes from the prompt mutation, not random sampling.
:::

---

## Section 7 — Defenses: Grounding, Citation, Abstention (≈5 pages)

Detection is half the job. The other half: design the agent to hallucinate less in the first place.

**1. Grounding via retrieval.** Force the agent to base claims on retrieved sources. Standard RAG; covered in Module 9. Add: don't just retrieve — make the retrieved snippets *visible* in context with markers, and prompt the model to ground every claim.

```python
SYSTEM_GROUND = """When answering, cite specific evidence from the provided <sources>.
For every factual claim, include a citation like [src:doc_id:line].
If no source supports a claim, either omit the claim or explicitly mark it 'unverified'."""
```

**2. Citation enforcement.** Validate the model's citations programmatically. If it cites `[src:42:7]`, check that source 42 line 7 actually contains the claimed evidence. Mismatched citations are explicit hallucination signals.

```python
def validate_citations(answer: str, sources: dict[str, str]) -> CitationReport:
    citations = extract_citations(answer)  # parses [src:id:line] markers
    invalid = []
    for cit in citations:
        if cit.source_id not in sources:
            invalid.append((cit, "unknown source"))
            continue
        evidence = get_lines(sources[cit.source_id], cit.start, cit.end)
        if not claim_supported_by(cit.claim, evidence):
            invalid.append((cit, "evidence does not support claim"))
    return CitationReport(invalid=invalid, valid_count=len(citations) - len(invalid))
```

**3. Abstention as a feature, not a bug.** Train/prompt the model to say "I don't know" rather than guess. The HalluLens benchmark (ACL 2025) shows GPT-4o has ~45% hallucination rate "when not refusing" — meaning models that refuse less hallucinate more.

```
SYSTEM_ABSTAIN = """If the provided sources do not contain enough information to answer
the question, respond with: "I don't have enough information to answer this confidently."
Do not guess. Do not infer beyond what the sources directly support."""
```

This costs you: more "I don't know" responses. It buys you: fewer confidently-wrong responses, which damage trust much more than abstentions do.

**4. Calibrated uncertainty.** Have the model output a confidence score per claim, calibrated against actual correctness. Hard to do well, but high-leverage where it works (medical, legal).

::: pullquote
A confidently-wrong answer costs you a customer. An honest "I don't know" costs you a follow-up question. The math favors abstention almost always.
:::

::: postmortem
**The Agent That Made Up Refund Amounts**

A support agent integrated with a customer DB. Asked "what's the refund for order #12345?", it would happily produce a number — even when the order didn't exist. The number was always plausible-looking ($23.47, $156.99). It was always wrong.

Cause: tool returned `{"error": "order not found"}`. The agent's prompt didn't say what to do with errors. Model interpreted "tool result" as "ground truth" and synthesized a plausible refund anyway, because that's what answers looked like in training.

Fix: tool result schemas explicitly modeled "not found" as a first-class state with an instruction in the system prompt: "if tool result indicates not_found, respond that the order was not found. Do not produce a refund amount." Hallucination dropped from ~3% of refund queries to <0.05%.

**Lesson:** the model fills gaps. Don't leave gaps to be filled.
:::

---

## Section 8 — Building a Hallucination Defense Layer (≈4 pages)

We integrate everything into a `HallucinationDefense` component that wraps the agent's outputs.

```python
class HallucinationDefense:
    def __init__(
        self,
        judge_client,
        faithfulness_threshold: float = 0.85,
        consistency_threshold: float = 0.7,
        require_citations: bool = True,
    ):
        ...
    
    async def check(
        self,
        answer: str,
        sources: dict[str, str] | None = None,
        prompt: str | None = None,
    ) -> DefenseResult:
        results = []
        
        # Citation validation (cheap, deterministic)
        if self.require_citations and sources:
            results.append(("citations", validate_citations(answer, sources)))
        
        # Faithfulness check (one judge call)
        if sources:
            f_score = await faithfulness_score(self.judge_client, format_sources(sources), answer)
            results.append(("faithfulness", f_score))
        
        # Self-consistency (N+1 generation calls; expensive — gate this)
        if prompt and self._should_check_consistency(answer):
            consistency = await self_consistency_check(self.judge_client, prompt, n=3)
            results.append(("consistency", consistency))
        
        return DefenseResult(checks=results, verdict=self._aggregate(results))
    
    def _should_check_consistency(self, answer: str) -> bool:
        # Only run consistency on high-stakes answers
        return contains_specific_facts(answer) or contains_numbers(answer)
```

The defense returns a verdict: PASS, WARN (suspicious; surface to user with disclaimer), or FAIL (block; ask agent to retry or abstain).

The agent loop integrates:

```python
async def run(self, query):
    while not done:
        resp = await self.client.messages.create(...)
        if resp.stop_reason == "end_turn":
            answer = extract_answer(resp)
            defense = await self.defense.check(
                answer=answer,
                sources=self.collected_sources,
                prompt=query,
            )
            if defense.verdict == "FAIL":
                # Inject correction request
                messages.append({"role": "user", "content": format_defense_failure(defense)})
                continue
            return answer  # PASS or WARN (warning surfaced separately)
```

::: code-exercise
**Exercise 10.1 — Wire the defense layer.**

Take the agent from Module 9 (now with full memory). Add `HallucinationDefense`. Run on a test set with known correct answers. Measure: false-positive rate (defense flagged correct answers), false-negative rate (defense missed wrong answers), latency overhead, cost overhead.

Tune thresholds. The goal isn't zero false positives — it's an acceptable balance for the workload's stakes.
:::

---

## Section 9 — Multi-Agent Communication Hallucinations (Preview) (≈2 pages)

A preview of what Module 14 covers in depth.

In multi-agent systems, hallucinations compound:
- Agent A hallucinates fact X
- Agent A communicates X to Agent B as if certain
- Agent B treats X as ground truth, builds on it
- Agent C synthesizes B's output, presents to user

User sees confident output built on a hallucinated foundation. The Sep 2025 survey "LLM-based Agents Suffer from Hallucinations" calls this **communication hallucination** and identifies it as a primary failure mode in multi-agent systems.

Defenses (covered fully in Module 14):
- Each agent's outputs labeled with confidence
- Inter-agent messages include source citations
- Receiving agents apply faithfulness checks before treating input as fact
- Cross-checking agents (debate/critique patterns from Module 14)

For now: know that adding more agents doesn't reduce hallucinations — it can compound them if you don't engineer for the structural risk.

---

## Section 10 — What's Next (≈1 page)

Module 11 covers **reflection** — using detected hallucinations to make agents self-correct mid-trajectory. Reflection and hallucination detection are siblings: detection tells you something's wrong; reflection acts on it.

Module 12 covers **guardrails** — input/output policy enforcement that includes hallucination defense as one layer in a stack.

Module 14 covers multi-agent specifically; communication hallucinations get the full treatment there.

Module 17 (production) covers monitoring hallucination rates over time, drift detection, and the eval infrastructure that feeds back into defenses.

---

## Recurring elements used in this module

- 1 chapter opener
- 1 hook
- 2 No Dumb Questions
- 2 Brain Power prompts
- 2 Production Postmortems (stale cache, made-up refund amounts)
- 1 Watch It! / Gotcha block
- 1 Code Exercise (defense layer)
- 2 Pullquotes
- 1 2x2 taxonomy table
- ~6 substantial code blocks
- 1 Bullet Points recap

::: sam-arc
Sam's agent works well... until QA finds it occasionally invents customer data when tools fail. Sam reads this module, realizes the agent isn't "hallucinating" in any single way — it's doing all six (Section 2). Sam fixes the made-up-refund issue by tightening tool result handling (Section 7), adds citation enforcement for retrieved facts, gates Best-of-N for high-stakes answers, and ships abstention as a first-class response. Hallucination-related complaints drop ~80%. Sam's arc this module: **specificity beats vague worry — name the failure mode and the defense becomes obvious**.
:::

::: page-budget
S1 (Taxonomy): 4p
S2 (Sources): 4p
S3 (Self-consistency): 5p
S4 (LLM-as-judge): 4p
S5 (Best-of-N): 4p
S6 (White/black-box internal): 3p
S7 (Defenses): 5p
S8 (Defense layer): 4p
S9 (Multi-agent preview): 2p
S10 (Next): 1p
TOTAL: ~36 pages
:::

::: sources
**Must verify when drafting:**

- Vectara hallucination leaderboard — current numbers for top models on grounded summarization
- HalluLens (ACL 2025) — GPT-4o ~45% "when not refusing" claim
- "LLM-based Agents Suffer from Hallucinations" survey (arXiv:2509.18970, Sep 2025) — taxonomy and communication hallucination
- Self-consistency original paper (Wang et al., ICLR 2023)
- Semantic entropy paper (Farquhar et al., 2024)
- ACL Findings 2025 paper on Best-of-N faithfulness reranking — find exact citation
- "INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection" — for EigenScore variant
- Cross-Layer Attention Probing (CLAP) paper, Sep 2025 — exact gain numbers (11.7% claim)
- MetaQA paper (ACM 2025)
- "No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding" (IUI 2025)
- Mu-SHROOM (SemEval 2025), CCHall (ACL 2025) — for multilingual / multimodal hotspots
- Vectara grounding metric definition
- HaluAgent (cited in survey) — autonomous hallucination detection agent

**Stable knowledge:**
- Factuality vs faithfulness, intrinsic vs extrinsic taxonomy
- Self-consistency as a general principle
- LLM-as-judge methodology
- Abstention as defense

**Cross-references:**
- Module 8 (compression) — compression-induced hallucination is one of the six sources
- Module 9 (memory) — retrieval misses are a primary hallucination source
- Module 11 (reflection) — what to DO when defense flags a hallucination
- Module 12 (guardrails) — defense layer integrates into broader guardrail stack
- Module 14 (multi-agent) — communication hallucination in depth
- Module 17 (production) — monitoring and drift
:::

::: bullet-points
### Module 10 in eight bullets

(filled at draft time)

- Two axes that map to defenses: factuality vs faithfulness, intrinsic vs extrinsic
- Six sources of agent "hallucinations" — many aren't model hallucinations at all (stale data, retrieval miss, compression loss)
- Self-consistency with semantic entropy: cluster N samples; high entropy = uncertainty
- LLM-as-judge for faithfulness; never use the same model as judge as generator
- Best-of-N reranking: generate N, score each, return most faithful — cheap variant: rejection sampling
- White-box (CLAP) and black-box (MetaQA) internal detection methods exist; pick by access pattern
- Defenses: grounding via retrieval, citation enforcement, abstention as first-class response, calibrated uncertainty
- Multi-agent communication hallucinations compound — confidence labels and inter-agent verification needed
:::
