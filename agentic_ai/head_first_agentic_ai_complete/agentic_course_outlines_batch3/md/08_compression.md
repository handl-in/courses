# Module 8 Outline — Context Compression Algorithms

::: chapter-opener
<div class="module-num">MODULE 08 — OUTLINE</div>
<div class="module-title">Context Compression Algorithms</div>
<div class="subtitle">Four real algorithms, four working implementations,<br>real benchmarks. The thing that pays the rent.</div>
<div class="pages">Target length: ~38 pages (algorithm-heavy + code-heavy)</div>
:::

## What this module is

Compression is the load-bearing technique in production agent systems. Claude Code triggers auto-compact at ~95% of the 200K window. OpenAI Codex runs server-side compaction after every turn. Cursor truncates old history. Every coding agent compresses; the methods differ.

This module covers four production-relevant compression algorithms in code: **anchored iterative summarization**, **ACON's failure-driven optimization** (arXiv:2510.00615), **budget-aware compression**, and **distilled compressors** for cost. We benchmark all four against naive baselines (FIFO truncation, retrieval, LLMLingua) on a long-horizon agent task. We end by integrating the best-performing into the Module 7 Context Builder.

By the end the reader can pick the right compression algorithm for their workload, implement it, measure it, and avoid the most common failure mode (compressing away the one fact the agent needed).

::: hook
"Naive context compression is easy. Useful context compression is hard. The hard part isn't 'how do I shrink this' — it's 'how do I shrink this without losing the one fact the agent will need three turns from now that I have no way of predicting.'"
:::

---

## Section 1 — The Compression Problem, Stated Precisely (≈3 pages)

What we're optimizing for. Inputs:
- A growing message history (assistant turns + tool results)
- A token budget (e.g., "context must stay under 50K")
- The agent's task (which determines what's relevant)

Output: a compressed message sequence that:
1. Fits the budget
2. Preserves task-critical information
3. Doesn't break the model's reasoning

The hard constraint is #2. You don't know what's task-critical until the agent needs it, three turns from now. Compression is gambling: you bet that the discarded info won't be needed.

::: pullquote
Every compression decision is a bet. Good compression algorithms make better-calibrated bets, not lossless ones.
:::

**Three signal types you can preserve:**

- **Factual** — entities, numbers, IDs that must be exact. Hardest to compress safely.
- **Procedural** — what was tried, what worked, what failed. Compresses well into action-outcome pairs.
- **Reasoning** — why the agent made decisions. Often safe to drop after the decision was made.

The ACON paper (Kang et al., Oct 2025) names five signal types worth preserving: factual history, action-outcome relationships, evolving environment states, success preconditions, future decision cues. Naive summarization tends to keep the first two and drop the last three — which is exactly why agents fail after compression.

::: nodumbq
**Q: Can't I just use a longer-context model and skip compression?**

You can — at the cost of money and quality. A 1M context Sonnet call costs ~$3 per turn on input tokens alone. Across a 30-turn task that's $90, before output. Compression buys you 50-80% reduction in those costs (per ACON's 26-54% peak token reduction, plus what you save on every subsequent turn). And research consistently shows long-context degradation — models reason more reliably on focused 30K than dumped 500K.

**Q: Isn't this what providers do server-side already?**

Some do. Anthropic supports prompt caching which helps on repeated context, but that's caching, not compression. OpenAI Codex does server-side compaction. But in agent code where you control the message history, you compress better than the provider can — because you know what's task-relevant.
:::

---

## Section 2 — Naive Baselines and Their Failure Modes (≈4 pages)

Before the good algorithms, the bad ones — so we know what we're improving on. ACON benchmarks against these; we'll do the same.

**FIFO truncation.** Drop oldest turns until you fit the budget.

```python
def compress_fifo(messages: list[dict], target_tokens: int) -> list[dict]:
    while count_tokens(messages) > target_tokens and len(messages) > 1:
        messages.pop(0)  # drop oldest non-system message
    return messages
```

Failure mode: the original task description and any early-established facts (user identity, key constraints) get dropped first. The agent forgets what it's doing.

**Sliding window with system pin.** Pin the system prompt and original user query; FIFO the rest.

```python
def compress_window(messages: list[dict], target_tokens: int, pin_first_n: int = 2) -> list[dict]:
    pinned = messages[:pin_first_n]
    rest = messages[pin_first_n:]
    while count_tokens(pinned + rest) > target_tokens and len(rest) > 1:
        rest.pop(0)
    return pinned + rest
```

Failure mode: middle-turn facts get dropped. If turn 8 established something important and you're at turn 30, that's gone.

**Retrieval-based selection.** Embed past turns; retrieve the top-k most similar to current state.

```python
def compress_retrieval(messages, current_query, target_tokens, k=10):
    candidates = messages[1:-1]  # everything but first and last
    embeddings = embed_all(candidates)
    query_emb = embed(current_query)
    ranked = rank_by_similarity(candidates, embeddings, query_emb)
    selected = top_k_within_budget(ranked, k=k, budget=target_tokens)
    return [messages[0]] + selected + [messages[-1]]
```

Failure mode: agent state often doesn't look like the answer's content. "I need to confirm the order" doesn't retrieve "confirmed order #12345 yesterday."

**LLMLingua-style token pruning.** Encoder-only LM scores each token; drop low-importance tokens.

Failure mode: works well for static prompts, breaks for agent contexts where exact tokens (function call args, tool result values) matter and "low importance" tokens are often the critical ones.

::: postmortem
**The Agent That Compressed Away Its Customer ID**

A team built FIFO compression into their support agent. Worked fine in tests. In production, by turn 25 of a long debugging session, the agent forgot which customer it was helping. It started returning answers about the wrong account. Customer was furious. Compliance flagged it.

Cause: the customer ID was established in turn 1's tool result ("identified user as alice@example.com, account_id=12345"). FIFO dropped turn 1 around turn 20. Nothing in subsequent turns repeated it. By turn 25, the agent's context started in the middle of a debugging conversation with no anchor.

Fix: switched to compression that pins facts (Section 3) and writes IDs to working memory (Module 7) so they survive compression.

**Lesson:** FIFO without semantic awareness is dangerous. The "old" turns often contain the anchors.
:::

---

## Section 3 — Anchored Iterative Summarization (≈5 pages)

The first real compression algorithm. The pattern most production systems implement — Claude Code's auto-compact, OpenAI Codex's compaction, and similar — is some variant of this.

**The idea:** maintain a running summary of "everything before turn N" alongside the most recent K turns verbatim. As the budget tightens, increase the summary's coverage; shrink the verbatim window.

**The "anchored" variant:** explicitly extract and preserve a set of anchor facts (entities, IDs, constraints, decisions made) that always survive compression, separate from the prose summary.

```python
@dataclass
class CompressedHistory:
    anchors: dict[str, str]   # stable facts: user_id, task_id, key constraints
    summary: str              # prose summary of compressed turns
    recent: list[dict]        # K most recent turns verbatim

class AnchoredSummarizer:
    def __init__(self, llm, recent_window=4, summary_target=2000):
        self.llm = llm
        self.recent_window = recent_window
        self.summary_target = summary_target

    async def compress(self, messages: list[dict], target_tokens: int) -> CompressedHistory:
        # Always keep last N turns verbatim
        recent = messages[-self.recent_window:]
        to_summarize = messages[:-self.recent_window]

        # Extract anchors (entities, IDs, decisions)
        anchors = await self._extract_anchors(to_summarize)

        # Generate prose summary
        summary = await self._summarize(to_summarize, target=self.summary_target)

        return CompressedHistory(anchors=anchors, summary=summary, recent=recent)

    async def _extract_anchors(self, messages):
        prompt = ANCHOR_EXTRACTION_PROMPT.format(history=format_messages(messages))
        resp = await self.llm.complete(prompt, max_tokens=500)
        return parse_anchors(resp)  # returns dict like {"user_id": "12345", "task": "..."}

    async def _summarize(self, messages, target):
        prompt = SUMMARY_PROMPT.format(
            history=format_messages(messages),
            target_tokens=target,
        )
        return await self.llm.complete(prompt, max_tokens=target)

    def to_messages(self, compressed: CompressedHistory) -> list[dict]:
        anchor_block = "\n".join(f"- {k}: {v}" for k, v in compressed.anchors.items())
        return [
            {"role": "user", "content": (
                f"<context>\n"
                f"<anchors>\n{anchor_block}\n</anchors>\n"
                f"<summary>\n{compressed.summary}\n</summary>\n"
                f"</context>"
            )},
            *compressed.recent,
        ]
```

We walk through the prompts. Anchor extraction is a structured-output call (use tool-use for reliability — Module 4's structured-output pattern). Summary generation is straightforward but the prompt matters a lot — we'll show good vs bad summary prompts.

**The iterative part:** when the compressed result still doesn't fit, summarize again with a tighter target. Re-summarization is lossy — be careful — but sometimes necessary.

::: code-exercise
**Exercise 8.1 — Implement anchored summarization.**

Build the `AnchoredSummarizer` class. Test it on a provided 30-turn agent transcript. Verify: anchors preserved across compressions, recent window intact, summary fits target. Compare resulting context size vs naive FIFO.
:::

::: gotcha
The recent window matters more than people realize. Anthropic and OpenAI both default to keeping ~3-5 most recent turns verbatim. Drop below that and the agent loses local coherence (forgets what it just did). Go above 8-10 and you've barely compressed.
:::

---

## Section 4 — ACON: Failure-Driven Compression (≈6 pages)

The Oct 2025 paper (Kang et al., arXiv:2510.00615) introduced **Agent Context Optimization** — a meta-algorithm that *learns* good compression by analyzing failures. ACON is the most important compression-research result of 2025.

**The insight:** compression prompts can be optimized in natural-language space (no fine-tuning) by:

1. Run a task with full (uncompressed) context. Record success/failure.
2. Run the same task with current compression. Record success/failure.
3. Find tasks where full succeeded but compressed failed.
4. For each failure, ask a capable LLM: "what information did the compression drop that caused this failure?"
5. Update the compression prompt to preserve that class of information.
6. Re-run. The compression prompt now generates better summaries.

This is gradient-free (no model training), works with any API-accessible model, and ACON reports 26-54% peak token reduction while preserving task performance — across AppWorld, OfficeBench, and Multi-objective QA.

```python
class ACON:
    def __init__(self, base_compressor: Compressor, judge_llm, optimizer_llm):
        self.compressor = base_compressor
        self.judge = judge_llm
        self.optimizer = optimizer_llm
        self.compression_prompt = INITIAL_COMPRESSION_PROMPT

    async def optimize(self, train_tasks: list[Task], rounds: int = 5):
        for round_i in range(rounds):
            failures = []
            for task in train_tasks:
                # Run with full context (ground truth)
                full_result = await run_agent(task, compressor=NoOp())
                # Run with current compression
                comp_result = await run_agent(task, compressor=self._with_prompt())
                
                if full_result.succeeded and not comp_result.succeeded:
                    failures.append((task, full_result, comp_result))
            
            if not failures:
                break
            
            # Analyze failures
            analysis = await self._analyze_failures(failures)
            
            # Update prompt
            self.compression_prompt = await self._update_prompt(
                old_prompt=self.compression_prompt,
                analysis=analysis,
            )

    async def _analyze_failures(self, failures):
        prompts = []
        for task, full, comp in failures:
            prompts.append(FAILURE_ANALYSIS_PROMPT.format(
                task=task.description,
                full_trajectory=full.trajectory,
                comp_trajectory=comp.trajectory,
                comp_compressed=comp.compressed_history,
            ))
        analyses = await asyncio.gather(*[self.judge.complete(p) for p in prompts])
        return aggregate(analyses)

    async def _update_prompt(self, old_prompt, analysis):
        update_prompt = PROMPT_UPDATE_TEMPLATE.format(
            old=old_prompt,
            analysis=analysis,
        )
        return await self.optimizer.complete(update_prompt)
```

We walk through the actual ACON paper's prompts (or paraphrase if licensing constraints — verify when drafting). Key design decisions:
- Use a strong model as judge and optimizer (Opus / GPT-5)
- Train on ~50-100 representative tasks
- Optimize for 3-5 rounds; gains saturate
- Distill the optimized compressor into a smaller model for deployment (preserves 95% of accuracy at lower per-call cost)

**Why this matters beyond the numbers:** ACON proves that compression is a *learnable* problem in natural-language space. You don't need a research team or GPUs. You need an eval set and an optimizer model.

::: brain
ACON requires running every training task twice (with and without compression). For 100 training tasks averaging $0.50 each, that's $100 to optimize the prompt. Worth it?

(Almost always. The optimized prompt then runs on every production task, indefinitely. If you save $0.05 per production task and you have 10K+ tasks, the math is overwhelming. ACON's economics favor anyone with enough volume to need compression in the first place.)
:::

::: code-exercise
**Exercise 8.2 — Run ACON on your own task set.**

Implement the ACON optimization loop. Use a small synthetic eval set (10 tasks we provide). Optimize the compression prompt for 3 rounds. Compare task success rate before vs after.

Realistic expectation: with a small eval set you may not see big gains, but you'll understand the loop. The point is the *mechanism* — you can scale it later with real production tasks.
:::

---

## Section 5 — Budget-Aware Compression (≈4 pages)

The newer wrinkle (ContextBudget, 2025/2026 research). Most compression methods are *budget-free* — they compress to a fixed target. But agent context budgets vary across the trajectory:

- Early turns: lots of headroom; minimal compression needed
- Middle turns: tightening; compress historically
- Late turns: tight; aggressive compression of everything pre-recent

Budget-aware compression conditions every compression decision on remaining capacity. Reinforcement learning (GRPO-extension) can train this end-to-end, but a simpler rule-based version captures most of the win:

```python
class BudgetAwareCompressor:
    def __init__(self, base: AnchoredSummarizer, total_budget: int):
        self.base = base
        self.total_budget = total_budget

    async def compress(self, messages: list[dict], turn_idx: int) -> list[dict]:
        current_size = count_tokens(messages)
        headroom = self.total_budget - current_size
        
        if headroom > self.total_budget * 0.5:
            # Plenty of room — no compression
            return messages
        elif headroom > self.total_budget * 0.2:
            # Moderate — compress old turns, keep generous recent window
            return await self.base.compress(messages, recent_window=8)
        else:
            # Tight — aggressive compression
            return await self.base.compress(messages, recent_window=3, summary_target=1000)
```

We discuss the more sophisticated RL-trained version, where the agent itself learns *when* to compress and *what* to preserve given remaining budget. This is research-frontier (Hu et al. 2026); reader is unlikely to deploy it but should know it exists.

::: nodumbq
**Q: Why isn't this just "compress when context is full"?**

Because by the time context is full, you've already paid for all those tokens this turn. Budget-aware compression looks ahead: how many more turns might this task need? At current rate of growth, when will I run out? Compress proactively to leave room for the next 3 turns of expected growth, not reactively after every turn.
:::

---

## Section 6 — Distilled Compressors (≈3 pages)

Running compression with a strong LLM is expensive. ACON's distillation contribution: take an optimized strong-LLM compressor, generate (input, output) training pairs, fine-tune a small model on them. The smaller model preserves 95%+ of the strong model's compression quality at a fraction of the cost.

```python
# Conceptual distillation pipeline
def distill_compressor(strong_model_compressor, training_data):
    pairs = []
    for messages in training_data:
        compressed = strong_model_compressor(messages)
        pairs.append({"input": messages, "output": compressed})
    
    # Fine-tune a small model (Haiku-class, or even open-source 3B-7B) on pairs
    fine_tuned = fine_tune_model(base="haiku-class", data=pairs)
    return fine_tuned
```

When this matters: high-volume agents where compression overhead is itself non-trivial. If your agent runs 10K times/day and each compression costs $0.01 with Sonnet, that's $100/day in compression alone. Distill to Haiku-class (or a small open-source model you self-host), and that drops 5-10×.

We don't actually fine-tune in the exercise (out of scope), but we discuss the architecture and when to invest in it.

::: pullquote
ACON shows you can teach a small model to compress as well as a big model. This isn't a magic trick — it's the same teacher-student distillation that's worked across ML for a decade. It just took until 2025 to apply it to agent compression.
:::

---

## Section 7 — Benchmark: All Methods on a Real Task (≈5 pages)

The payoff section. We define a representative long-horizon agent task and benchmark every method. (Numbers are illustrative — we re-run with current models and current pricing when drafting.)

**Task:** a research agent answering 20 questions that each require 15-25 tool calls (search, fetch, read, synthesize). Each question's answer must be verifiable against a known correct answer.

**Methods compared:**
- No compression (baseline)
- FIFO truncation
- Sliding window with system pin
- Retrieval-based
- LLMLingua-style token pruning
- Anchored iterative summarization (Section 3)
- ACON-optimized summarization (Section 4)
- Budget-aware (Section 5)

**Metrics:**
- Average peak tokens per task
- Average task cost
- Task success rate (vs verified correct answers)
- p95 latency per turn
- Number of tasks where critical info was lost

Sample expected results (illustrative):

| Method | Peak tokens | Cost / task | Success | p95 latency | Info loss |
|--------|-------------|-------------|---------|-------------|-----------|
| No compression | 142K | $1.42 | 87% | 6.2s | 0% |
| FIFO | 50K | $0.31 | 53% | 4.1s | 38% |
| Sliding window | 50K | $0.32 | 61% | 4.2s | 31% |
| Retrieval | 50K | $0.34 | 64% | 4.4s | 27% |
| LLMLingua | 70K | $0.51 | 68% | 5.0s | 22% |
| Anchored summarization | 55K | $0.42 | 81% | 4.7s | 9% |
| ACON-optimized | 48K | $0.39 | 86% | 4.6s | 4% |
| Budget-aware | 52K | $0.41 | 85% | 4.5s | 5% |

**The headline:** ACON-optimized summarization recovers ~99% of no-compression task success at ~27% of the cost. That's the math that makes compression an unambiguous win when implemented well.

**The footnote:** every method has tasks where it fails. The right choice depends on your workload — read failure modes, not just averages.

::: brain
Look at the table. The naive baselines (FIFO, sliding window) are 50-65% success. The good algorithms (anchored, ACON, budget-aware) are 81-86%. Why such a gap?

(Because the naive methods drop information without considering relevance. The good methods preserve task-critical signals (anchors, action-outcome pairs, success preconditions) explicitly. The 30+ point success gap is the entire reason this module exists.)
:::

---

## Section 8 — Choosing a Method (≈3 pages)

Decision framework. Match algorithm to workload.

**Use anchored iterative summarization when:**
- You have moderate volume (100s-1000s of tasks/day)
- Tasks have clear "anchor" facts (IDs, constraints, decisions)
- You don't want to invest in eval infrastructure yet
- Default starting point for most teams

**Use ACON-optimized when:**
- You have an eval set (or can build one)
- Volume justifies the upfront optimization cost
- Failure modes from generic summarization are biting you
- You want measurable wins beyond default summarization

**Use budget-aware when:**
- Context budgets vary wildly across tasks
- Some tasks are short (no compression needed); others very long
- You're already running anchored or ACON; this is an upgrade

**Use distilled compressors when:**
- High volume (10K+ tasks/day)
- Compression cost itself is non-trivial
- You have ML infrastructure for fine-tuning

**Use naive baselines when:**
- Prototyping
- Workload is so simple you genuinely don't need compression yet
- You want a known-bad baseline to measure against

::: gotcha
Don't optimize compression in isolation. The right metric isn't "smallest context" — it's "task success per dollar." A method that gets you to 30K tokens but tanks success isn't winning. Always benchmark on the actual task, not just on token count.
:::

---

## Section 9 — Integrating With the Context Builder (≈3 pages)

We wire the compressor into the Module 7 Context Builder. The builder now calls `Compressor.compress()` with the appropriate budget, and the agent loop is unchanged from Module 2 — all the complexity hides in the builder.

```python
class ContextBuilder:
    def __init__(self, memory, compressor: Compressor, budget):
        self.memory = memory
        self.compressor = compressor
        self.budget = budget

    async def build(self, system, tools, working_memory, ...):
        # Compress working memory if needed
        if count_tokens(working_memory) > self.budget.working_target:
            working_memory = await self.compressor.compress(
                working_memory,
                target_tokens=self.budget.working_target,
            )
        
        # Retrieve memory (Module 9 fills this)
        retrieved = await self.memory.retrieve(...)
        
        return assemble_messages(system, tools, retrieved, working_memory)
```

We swap compressor implementations to compare. Same agent code; different `Compressor` instance; different cost-quality profile.

::: code-exercise
**Exercise 8.3 — Wire and benchmark.**

Take the Context Builder from Module 7. Plug in three compressors: `FIFOCompressor`, `AnchoredSummarizer`, and `ACONCompressor` (using a simple synthetic optimization). Run the same set of agent tasks under each. Generate a results table. Decide which to use for production.
:::

---

## Section 10 — What's Next (≈1 page)

Module 9 builds the **Memory System** — the storage layer that compression relies on. Once compression discards old turns, the facts in those turns need somewhere to live. That's memory.

Module 10 covers hallucination — relevant here because aggressive compression can introduce hallucinations (the model invents details about what was compressed away).

By Module 12 (guardrails) we'll add output checks that validate compressed-context outputs against original facts.

The capstone (Module 18) uses ACON-optimized compression in the production agent.

---

## Recurring elements used in this module

- 1 chapter opener
- 1 hook
- 2 No Dumb Questions
- 2 Brain Power prompts
- 1 Production Postmortem (compressed-away customer ID)
- 2 Watch It! / Gotcha blocks
- 3 Code Exercises (anchored, ACON, integrated benchmark)
- 2 Pullquotes
- 1 benchmark results table
- 1 Bullet Points recap

::: sam-arc
Sam's agent now has the Context Builder from Module 7 — but it's a passthrough. Long tasks are still expensive and slow. Sam ships FIFO compression first ("seemed simple"), then loses a customer ID and gets paged at 2am (the postmortem in Section 2). Sam then implements anchored summarization (much better), reads the ACON paper (Section 4), and runs the optimization loop on a small eval set. By the end of Section 9, Sam has a Compressor that's saving 70% of token cost at 99% of original task success. Sam's arc this module: **the right algorithm pays for the engineering time it took to find it**.
:::

::: page-budget
S1 (Problem stated): 3p
S2 (Naive baselines): 4p
S3 (Anchored summarization): 5p
S4 (ACON): 6p
S5 (Budget-aware): 4p
S6 (Distilled compressors): 3p
S7 (Benchmark): 5p
S8 (Choosing method): 3p
S9 (Integration): 3p
S10 (Next): 1p
TOTAL: ~37 pages
:::

::: sources
**Must verify when drafting (heavy verification — research-frontier):**

- ACON paper (Kang et al., arXiv:2510.00615, v2 Oct 17 2025) — primary source, all numbers
- ACON's exact baselines (FIFO, Retrieval, LLMLingua, naive prompting) and their reported scores
- ContextBudget / Budget-Aware Context Management (Hu et al., 2026) — for Section 5
- LLMLingua paper for the token-pruning baseline characterization
- Claude Code's auto-compact behavior (95% threshold, was 77-78% mid-2025)
- OpenAI Codex server-side compaction documentation
- Cursor's history truncation approach
- Morph FlashCompact 2026 comparison piece — for the "every coding agent compresses" framing
- Factory.ai's 36K-message benchmark
- Current Anthropic Sonnet 4.6 / Opus 4.7 input pricing for cost calculations
- Tiktoken or equivalent tokenizer for `count_tokens()` helper

**Stable knowledge:**
- The general compression problem statement
- The five signal types worth preserving (factual, action-outcome, environment, preconditions, decision cues — from ACON)
- Distillation as a general technique
- Decision frameworks for picking algorithms

**Open question for drafting:**
- ACON's prompts: paraphrase from the paper or get permission to reproduce verbatim? Lean toward paraphrase + cite.
- Budget-aware section: cite the GRPO-extension paper or just describe the approach? Probably describe + cite.

**Cross-references:**
- Module 7 introduces compression as one of four strategies; this module is the algorithm-deep version
- Module 9 builds the memory layer that compression discards into
- Module 10 (hallucination) covers compressed-context hallucination
- Module 17 (production) covers monitoring compression quality drift
:::

::: bullet-points
### Module 8 in eight bullets

(filled at draft time)

- Compression is a bet on what the agent won't need; good algorithms make better-calibrated bets
- ACON identifies five signal types worth preserving: factual, action-outcome, environment state, preconditions, decision cues
- Naive baselines (FIFO, sliding window, retrieval, LLMLingua) lose 30-50% task success — they drop the wrong things
- Anchored iterative summarization is the production default; pin entities/IDs/decisions, prose-summarize the rest
- ACON optimizes compression prompts in natural-language space using failure analysis — gradient-free, strong gains
- Budget-aware compression conditions decisions on remaining capacity; works with any base method
- Distilled compressors run a fine-tuned small model with ~95% of strong-model compression quality at lower cost
- Benchmark results: ACON-optimized recovers ~99% no-compression task success at ~27% the cost
:::
