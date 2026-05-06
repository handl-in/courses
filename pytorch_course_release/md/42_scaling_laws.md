# Module 42 — Scaling laws & compute economics

# _Scaling laws & compute economics:_ where to spend your FLOPs

_Part XI · Module 42 · April 2026 currency_

— from Chinchilla's 20:1 to LFM2.5's 80,000:1, why every frontier model is now overtrained, the three-way pretrain/RL/test-time compute tradeoff, and the economics of choosing what to scale

\--- 

Scaling laws are the closest thing this field has to a theory. From Kaplan et al. (2020) through Chinchilla (Hoffmann et al., 2022) through the inference-aware reformulations (Sardana/MosaicML 2024-2025) through the unified test-time compute scaling work (2026), each generation of laws answers a different version of the same question: _given a fixed compute budget, how should you spend it?_ The 2022 answer was "20 tokens per parameter, balance training compute between size and data." The 2026 answer is more interesting: _it depends on how much inference you'll do, what test-time compute strategy you'll use, what numerical precision you'll train in, and whether you're doing pretraining or RL post-training._

This module is foundational and threads through everything in M1-M41. Where M11 talked about training a model, M14 talked about precision, M27 about serving, M34 about reasoning, M39 about RL — this module is about _choosing how much of each_. By the end you'll know why Chinchilla's ratio is too aggressive for production deployment, why frontier 2026 models train at ratios 1000-4000× higher than Chinchilla-optimal, when test-time compute beats parameter scaling, what the RL scaling laws look like (log-linear in reasoning tokens), and how to reason about the economics of "should I train a bigger model or run my smaller one for longer at inference?"

> **★ KEY IDEA**  
>  Scaling laws went through three eras. **Era 1 (Kaplan 2020)** : scale parameters faster than data. **Era 2 (Chinchilla 2022)** : parameters and tokens should grow equally — 20 tokens per parameter at compute-optimal training. **Era 3 (Beyond Chinchilla, 2024-2026)** : inference cost matters — train smaller and longer when serving many requests. Production frontier models are now massively overtrained: Llama 3 8B at 1,875:1, Qwen3-0.6B at 60,000:1, **Liquid AI LFM2.5-350M at 80,000:1** (the public April 2026 record). The compute budget now splits three ways: **pretraining** (expanding base capability), **RL post-training** (M39 territory; ~20% of pretraining compute for DeepSeek-R1, scaling fast — projected to match pretraining by late 2026 per Epoch AI), **test-time compute** (M34 reasoning; smaller models with best-of-N or tree search outperform larger models on Pareto curves). RL scaling laws are **log-linear in reasoning tokens** : accuracy increases linearly with log of average tokens generated. Test-time scaling laws favor smaller-than-Chinchilla models for inference-heavy deployments; Sardana et al. validated up to 10,000:1 ratios. The key 2026 finding: **test-time scaling makes overtraining compute-optimal**. The three regimes interact: precision (M14, M41) modifies all three (Scaling Laws for Precision, Kumar et al.); data constraints (Muennighoff et al.) cap pretraining; RL recipes (DAPO/REINFORCE++) determine RL efficiency. _The trade is no longer "bigger or more data?" — it's "where on the pretrain × RL × inference compute manifold should I sit?"_

## Two new faces — closing Part XI

L

Scaling Law

"I'm an empirical formula relating compute, parameters, data, and quality. I rule training-budget decisions."

Give me your compute budget; I'll tell you how to spend it. Chinchilla taught me 20:1 for compute-optimal training. The Beyond-Chinchilla generation taught me that compute-optimal isn't deployment-optimal: if you're going to serve a model 1B times at inference, you want it smaller and trained longer, even if it costs more upfront. The reasoning-model era added another axis: I now decompose into **pretrain × RL post-train × test-time compute** , and the optimal mix depends on your task and traffic. _I'm not a single law anymore_ ; I'm a family of curves you have to navigate. By 2026, model families like Llama and Qwen ship at multiple sizes specifically because different deployment scales sit at different points on me.

$

Economist

"I translate FLOPs into dollars. Training cost vs inference cost vs RL cost — I track all three."

Compute-optimal isn't dollar-optimal. Training a Chinchilla-optimal 70B model costs ~$10M; if you serve 100M requests at inference, the inference compute totals $50M. Train a smaller 13B for 5× more tokens (still improves) and your training cost rises to $15M but inference drops to $10M — net win. **I'm what the Beyond-Chinchilla literature is really about**. In 2026, I add two more line items: **RL post-training compute** (DeepSeek-R1 spent $1M on RL, 20% of pretraining; ScaleRL and ProRL V2 push this much higher) and **test-time compute** (running a model in best-of-32 mode is 32× per-query, but lets you use a smaller model). Production AI economics is portfolio optimization across these three pools.

## Era 1: Kaplan (2020) — the original scaling laws

Kaplan et al. ("Scaling Laws for Neural Language Models," 2020) found that test loss decreases as a power law in three quantities: parameters N, training tokens D, and compute C. The relationships looked like:
    
    
    L(N) ≈ (N_c / N)^α_N      # loss decreases with model size
    L(D) ≈ (D_c / D)^α_D      # loss decreases with training tokens
    L(C) ≈ (C_c / C)^α_C      # loss decreases with compute

The exponents in Kaplan's measurements suggested that **parameters should grow faster than data**. For a fixed compute budget, the recommendation was to scale parameters aggressively — about 5× more than data per FLOP doubling. This is why GPT-3 had 175B parameters but was trained on only 300B tokens (a 1.7:1 ratio).

The Kaplan recommendations dominated 2020-2022. Models grew aggressively in parameter count; data scaling lagged. _Then Chinchilla came along and showed Kaplan was wrong about the exponents._

## Era 2: Chinchilla (2022) — the 20:1 ratio

Hoffmann et al. ("Training Compute-Optimal Large Language Models," 2022) ran a much more careful empirical study — training hundreds of models across the parameter/data plane, fitting the loss surface, and finding the compute-optimal frontier. Their conclusion was sharper than Kaplan's: **parameters and tokens should grow approximately equally**. For a fixed compute budget, the optimal split is roughly equal FLOPs spent on bigger models and on more data.

The practical translation: **~20 tokens per parameter** at compute-optimal training. A 70B model should be trained on ~1.4T tokens to be compute-optimal; conversely, GPT-3 (175B params, 300B tokens, 1.7:1 ratio) was massively undertrained — the same compute would have produced a much better model at ~30B params and 600B tokens, or vice versa.

To validate, DeepMind trained Chinchilla itself: **70B parameters, 1.4T tokens, beat GPT-3 (175B) on most evals using a fraction of the compute**. The result was decisive enough to overturn Kaplan's recommendations almost overnight.

### Why the 20:1 ratio mattered

The Chinchilla finding had two practical implications:

  * **Most models from the GPT-3 era were undertrained**. They had too many parameters relative to data; reducing parameter count and adding tokens would have produced better models.
  * **Compute could be redirected**. Instead of scaling to ever-larger parameter counts, the same compute spent on 20:1-balanced training would produce stronger models.

The 2022-2023 model wave (Llama 1, OPT, BLOOM, MPT, Falcon) explicitly adopted Chinchilla-aware ratios. Llama 1 (Touvron et al., February 2023) trained 7B/13B/33B/65B models on 1T-1.4T tokens — _much_ closer to compute-optimal than GPT-3.

## The Chinchilla Trap and Era 3: Beyond Chinchilla

By 2023, a problem became visible. Chinchilla-optimal training optimizes for _training compute_ only. It ignores inference cost. For a model that's deployed and queried billions of times, inference compute dominates training compute by orders of magnitude — a 70B Chinchilla-optimal model is expensive to run, and a smaller model trained for longer (well beyond 20:1) might serve the same accuracy at far lower deployment cost.

This is the **Chinchilla Trap** : training compute-optimal at deployment time means using a model larger than you should.

The Llama 1 paper (Touvron et al., 2023) flagged this explicitly — it noted that loss continues to decrease beyond Chinchilla-optimal ratios, and that smaller-but-longer-trained models would be cheaper to deploy. Llama 1's smallest 7B model was trained at **142:1 ratio** (1T tokens / 7B params) — 7× past Chinchilla's recommendation.

### Sardana et al. (MosaicML 2024): the inference-aware scaling law

The first systematic treatment of "Beyond Chinchilla" was Sardana et al.'s "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws" (2024, refined April 2025). They modified the Chinchilla loss surface to include inference cost, parameterized by expected number of inference requests. The headline finding:

**Researchers expecting reasonably large inference demand (~1B requests) should train models smaller and longer than Chinchilla-optimal** — substantially so. Their experiments validated up to **10,000 tokens per parameter ratios** , with quality continuing to improve well past Chinchilla's 20:1.

The argument is concrete economics. If a 70B Chinchilla-optimal model costs $10M to train and $50M to serve over its lifetime, vs a 13B-trained-for-5×-more-tokens that costs $15M to train and $10M to serve at the same quality, the smaller model wins by $35M. _For high-traffic deployments, training duration is the right knob to spend extra compute on._

### The Llama trajectory: explicit overtraining

Successive Llama generations went further past Chinchilla:

The overtraining trajectory: tokens-per-parameter ratios over time Model| Year| Params| Tokens| Ratio| Multiple of Chinchilla  
---|---|---|---|---|---  
GPT-3| 2020| 175B| 300B| 1.7:1| 0.085× (severely undertrained)  
Chinchilla| 2022| 70B| 1.4T| 20:1| 1× (the canonical reference)  
Llama 1 7B| Feb 2023| 7B| 1T| 142:1| 7×  
Llama 2 7B| Jul 2023| 7B| 2T| 284:1| 14×  
Llama 3 8B| Apr 2024| 8B| 15T| 1,875:1| 94×  
Qwen3-0.6B| Apr 2025| 0.6B| 36T| 60,000:1| 3,000×  
**Liquid AI LFM2.5-350M**| **Apr 2026**| **0.35B**| **28T**| **80,000:1**| **4,000×**  
  
The pattern: as deployment becomes the dominant cost, training ratios explode. LFM2.5-350M's **80,000:1** ratio is the public April 2026 record — a 350M-parameter model trained on 28 trillion tokens. The math: ~10²² FLOPs of pretraining, deployable at orders-of-magnitude lower inference cost than any Chinchilla-optimal model would offer at similar quality. _The Era 3 recipe is to train tiny models for absurdly long_ , when your deployment requires inference-cheap models.

### Data constraints and the limits of overtraining

Naïvely you'd think you could keep increasing tokens forever. Two realities push back:

  * **Data scarcity**. The web has a finite quantity of high-quality natural language. By 2025-2026, frontier teams hit walls — there isn't enough fresh data to train at 100,000:1 ratios for ever-bigger models. M35's data engineering becomes the bottleneck.
  * **Diminishing returns from repeated data** (Muennighoff et al., 2023, "Scaling Data-Constrained Language Models"): training for multiple epochs over the same data still helps, but with rapidly diminishing returns. They validated up to ~4 epochs as broadly useful; up to 1500 epochs at small scales but with sharply diminishing benefit. **The first 1-2 epochs over fresh data are worth ~the same as same-volume fresh data; beyond 4 epochs, additional training is mostly wasted.**
  * **Code data as a 2× multiplier** (same paper): mixing in 50% code data effectively doubles useful training tokens for natural language tasks; some tasks (state-tracking, reasoning) actively improve with code mix.

The April 2026 reality: LFM2.5-350M's 80,000:1 is achieved partly by aggressive data curation (M35's province) and partly by accepting some repetition. _How much further this can go before hitting hard data limits is an open empirical question._

## The three-pool decomposition: pretraining × RL × test-time compute

By 2026, the compute budget for a frontier model splits across three pools, not one:

The three compute pools — and how they trade off ① Pretraining compute Base capability Knob: tokens × params Chinchilla 20:1 = compute-opt. Beyond Chinchilla: 100-80,000:1 for inference-heavy deployment Limits: • Data scarcity (M35) • Diminishing returns past ~4 epochs • Loss of generalization at extremes Frontier 2026: 1e25-1e26 FLOPs ② RL post-training Reasoning + agentic Knob: rollouts × env interactions M39: distributed RLHF/GRPO M40: agentic RL environments Log-linear scaling in tokens Trajectory: • DeepSeek-R1: $1M, 20% of pretrain • ProRL V2 (Feb 2026): 700+ steps • ScaleRL (Oct 2025): frontier scale Projected: ≈ pretrain by late 2026 ③ Test-time compute Per-query reasoning Knob: tokens × samples per query M34: chain-of-thought, BoN, MCTS Smaller model + N samples can beat larger model on Pareto Tradeoffs: • Per-query cost scales linearly • Latency scales with depth • Overthinking degrades accuracy M37 calibration mandatory The unified question: where on the (pretrain, RL, test-time) compute manifold should I sit? Inference-heavy + low task complexity → small model, long pretrain, modest RL, minimal TTC Inference-light + high reasoning → moderate model, heavy RL, generous TTC (best-of-N, tree search) → "Test-time scaling makes overtraining compute-optimal" (2026): the three pools COUPLE

The three pools have substantively different scaling laws and substantively different costs:

  * **Pretraining** scales with parameters × tokens. Compute is loss-bound; you spend it once. Frontier 2026 pretraining: 10²⁵-10²⁶ FLOPs for top-tier models. Cost: $10M-$200M depending on size and hardware.
  * **RL post-training** scales with rollouts × environment interactions. Compute is reward-bound; the policy improves until it saturates the reward signal. Frontier 2026: comparable scale to pretraining or somewhat less; rapidly growing. DeepSeek-R1 spent ~$1M on RL (20% of pretraining); ProRL V2 (Feb 2026) ran 700+ stable RL steps; **Epoch AI's Q1 2026 estimate: reasoning training compute will converge with overall frontier compute by late 2026**.
  * **Test-time compute** scales with tokens × samples per query. Compute is per-query; total cost scales linearly with deployment volume. Each query at best-of-32 mode is 32× more expensive than greedy decoding, but accuracy improves substantially for hard tasks.

## RL scaling laws: log-linear in reasoning tokens

The Era-3 generation surfaced a specific RL scaling regularity. The "Scaling Reasoning Tokens via RL and Parallel Thinking" paper (Apr 2026, Seed-OSS-36B trained on competitive programming):

**During RL training, validation accuracy increases linearly with the logarithm of average generated reasoning tokens.** As successive RL checkpoints generate longer reasoning traces, accuracy follows a clear log-linear trend.
    
    
    # Empirical observation across RL checkpoints during training:
    accuracy = α * log(avg_reasoning_tokens) + β
    
    # Two ways to shift the trajectory (per the paper):
    #   1. Verification RL warmup: raises β (starting point higher)
    #   2. Randomized clipping: increases α (steeper slope)

The "Art of Scaling RL Compute" paper (Oct 2025) corroborated this at scale, validating frontier RL recipes (DAPO clipping, asymmetric loss aggregation, length interruptions to prevent runaway generation). The general pattern: _RL scaling is well-behaved and log-linear in reasoning tokens, similar to the test-time scaling laws but during training_.

The implication for compute economics: **RL training compute provides predictable returns, much like pretraining compute does**. You can budget RL compute based on target token-length goals, similar to budgeting pretraining tokens against target loss. This was unclear in early 2025; the 2025-2026 papers settled it.

## Test-time compute scaling: smaller can beat bigger

The "Inference Scaling Laws" paper (Snell et al., 2024, refined 2025) ran a careful empirical study: **given a compute budget, when is it better to run a smaller model with more inference compute vs a bigger model with less?**

Their finding, validated across model families (Pythia, Llemma, Llama): _smaller models combined with advanced inference algorithms (best-of-N, tree search, weighted voting) can be Pareto-optimal compared to scaling parameters_. Specific result: **Llemma-7B with their tree search consistently outperformed Llemma-34B across all tested inference strategies on the MATH benchmark** — a 5× smaller model winning by spending compute on inference instead of parameters.

Inference scaling: smaller model + N samples vs larger model accuracy → total compute (FLOPs, log scale) → 7B + best-of-N inference larger models, greedy decode Pareto crossover Llemma-7B with tree search beats Llemma-34B greedy on MATH at the same compute (Snell et al. 2024) → for hard reasoning tasks, inference compute > parameter scaling on Pareto

### The TTS trilemma: accuracy, consistency, efficiency

The "Art of Scaling Test-Time Compute" paper (December 2025) introduced the **test-time scaling trilemma** : you can optimize for at most two of accuracy, consistency, and efficiency.

  * **High accuracy + high consistency** (e.g., self-consistency with many samples + majority vote): expensive — many samples per query.
  * **High accuracy + high efficiency** (e.g., adaptive depth — short for easy, long for hard): inconsistent — same query may take different paths different times.
  * **High consistency + high efficiency** (greedy decoding): low accuracy on hard tasks.

Production deployment chooses two; the third is the cost.

### Compute-optimal TTC allocation

The follow-on work — Snell et al. (2024) for the foundational paper, and the April 2026 "Adaptive Test-Time Compute Allocation" paper — addressed the practical question: _given a query, how much inference compute should you spend?_

The 2026 framework: train a difficulty classifier (often tree-based, since allocation boundaries are non-linear) that predicts how much TTC the query needs; use a Lagrangian relaxation with a single dual variable λ controlling the accuracy-cost tradeoff. The result is principled adaptive allocation: easy queries use 1× compute, hard queries use 32× or more, average compute stays bounded.

The published results: **up to 70% accuracy gains with 39% token reduction** compared to flat-allocation baselines (Plan-and-Budget, May 2025; refined 2026). The technique works across DeepSeek-R1-Distill, QwQ-32B, OpenAI o4-mini.

### Overthinking: when more TTC hurts

Counter-finding from 2025-2026 literature: **longer chain-of-thought often degrades accuracy**. Models can overthink, generate unnecessarily long and tangential reasoning paths, and arrive at worse answers than they would with shorter reasoning.

  * Ghosal et al. (2025) and Hassid et al. (2025) showed that longer CoT chains often degrade accuracy on benchmark tasks.
  * Inverse scaling effects (Gema et al., 2025): more reasoning sometimes makes things worse on specific task types.
  * Plan-and-Budget (2025-2026): explicit token budget allocation produces 39% token reduction with accuracy improvements, showing the prior expense was largely waste.

The implication: _more test-time compute is not monotonically better_. Adaptive allocation (spend more on hard queries) and budgeted allocation (cap easy queries) both outperform flat scaling. M37's eval rigor matters here — measure accuracy as a function of TTC for your tasks; don't assume the relationship is monotonic.

## The unifying 2026 finding: TTS makes overtraining compute-optimal

The most important recent paper on scaling laws — "Test-Time Scaling Makes Overtraining Compute-Optimal" (April 2026) — unified the pretraining and inference scaling literature. The key insight: _pretraining and test-time compute decisions are coupled, not independent_.

The argument:

  1. Test-time scaling (best-of-N, tree search) requires sampling many times from the model.
  2. Sample quality matters: pass@k probability depends nonlinearly on per-sample accuracy.
  3. Smaller models are cheaper per sample but weaker per sample.
  4. The "right" pretraining decision depends on how the model will be used at inference: a model designed for many-sample TTC inference should be smaller (faster per sample) and trained longer (higher per-sample accuracy).

**Conclusion: when test-time scaling is part of deployment, the compute-optimal pretraining ratio shifts toward overtraining** — exactly the trajectory we've seen in production (Llama 3 → Qwen3 → LFM2.5).

The paper validated this empirically: across model sizes and pretraining schedules, models trained at higher tokens-per-parameter ratios have better TTC scaling curves. The 80,000:1 ratio of LFM2.5-350M is consistent with a deployment plan that includes substantial test-time compute.

## Scaling laws for precision (M14 + M41 connection)

Kumar et al. ("Scaling Laws for Precision," November 2024) added another axis: **numerical precision interacts with the scaling laws**. Their findings, validated across 465 pretraining runs and model sizes up to 1.7B trained on up to 26B tokens:

  * **Training in lower precision may be compute-optimal** for some configurations. The compute saved per FLOP at FP8 or NVFP4 (M41) can outweigh the slight quality loss for many tasks.
  * **Post-training quantization degradation scales with training tokens**. A model trained on 10T tokens loses more accuracy when post-training-quantized than the same model trained on 1T tokens — suggesting that inference-time precision changes interact with training-time pretraining decisions.
  * **A unified scaling formula** : predicts loss as a function of parameters, tokens, AND precision. Different parts of training/inference can be in different precisions; the formula composes.

The practical 2026 implication: **"how should I train" includes a precision dimension that interacts with the size and data dimensions**. M14 + M41 + this scaling work all compose. NVFP4 training at 80,000:1 ratios (LFM2.5 territory) is a coherent design point that exploits all three insights.

## The economics: real costs in 2026

Translating compute into dollars at April 2026 cloud prices:

Compute cost estimates, April 2026 Workload| Compute| Hardware| Approx cost  
---|---|---|---  
7B model, Chinchilla-optimal (140B tokens)| ~5e21 FLOPs| 8× H100 for 1 week| $10-30K  
7B model, Llama-3-style (15T tokens)| ~5e23 FLOPs| 64× H100 for 4 weeks| $300-800K  
70B model, Chinchilla-optimal (1.4T tokens)| ~6e23 FLOPs| 128× H100 for 4 weeks| $500K-1M  
70B model, overtrained (15T tokens)| ~6e24 FLOPs| 512× H100 for 4 weeks| $3-7M  
405B model (Llama-3.1 scale)| ~4e25 FLOPs| 5,120 Blackwell GPUs (10 min on B200/NVFP4 — MLPerf v5.1)| ~$10M with NVFP4 / Blackwell  
DeepSeek-R1 RL (reasoning)| ~6e23 FLOPs| 2T tokens of rollout| ~$1M (20% of pretraining)  
ProRL V2 1.5B reasoning (700+ steps)| ~few × 10²² FLOPs| 32-128× H100 for 2-4 weeks| $50-200K  
Inference (frontier model)| per-query 10⁹-10¹¹ FLOPs| Blackwell + NVFP4 (M41)| $0.02 / 1M tokens (GPT-OSS-120B benchmark)  
Inference at scale (1B requests, 70B model)| ~10²⁰ FLOPs| varies| ~$5-20M depending on TTC mode  
  
Three observations:

  * **The Blackwell+NVFP4 transition (M41) reshaped the math**. Llama 3.1 405B for $10M is a 2026 number; on Hopper FP8 it would be $30M+; on H100 BF16 it would be $80M+. Hardware progress changes which scaling law regime is economically reachable.
  * **Inference dominates training over deployment lifetime** for any model with substantial traffic. A model trained for $1M and serving $10M of inference makes training cost a 10% line item.
  * **RL post-training cost is rapidly catching up to pretraining**. ProRL V2 at $50-200K is small; ScaleRL at frontier scale (Oct 2025) is much larger; Epoch AI projects parity with overall frontier by late 2026.

## Decision framework: where to spend FLOPs

Putting it all together, the 2026 framework for "how should I spend my compute budget":

  1. **What's the deployment volume?** If high (millions+ requests), bias toward smaller models trained longer (Beyond Chinchilla; Liquid AI / Qwen3 territory). If low, Chinchilla-optimal or close.
  2. **Is reasoning required?** If yes, plan RL post-training (M39, M40) and test-time compute. If no, pretraining alone may suffice.
  3. **What hardware is available?** Blackwell (M41) lets you reach 4× more useful compute per dollar than Hopper. Adjust the entire budget accordingly.
  4. **What's the data ceiling?** If data-constrained (M35), training ratio is bounded; can't go to 80,000:1 if the data doesn't exist. Code mixing and data engineering matter.
  5. **What test-time strategy?** Greedy → larger model is better. Best-of-N or tree search → smaller-model-overtrained is better (the Snell et al. + 2026 unifying paper finding).
  6. **What's the precision plan?** NVFP4 training (M41) effectively multiplies your compute budget by 2-3× over FP8. Build that into the training ratio.

The compound recipe for a 2026 production deployment:

  * **Pretraining** : small-to-moderate model size (1B-30B), 1,000-10,000:1 token-to-parameter ratio, NVFP4 training on Blackwell, M35-quality data.
  * **RL post-training** : REINFORCE++ baseline or two-sided clipped GRPO (M39), agentic environments (M40) for the target task surface, 700+ stable steps if reasoning is the target.
  * **Test-time compute strategy** : adaptive allocation (Plan-and-Budget style); compute-optimal inference based on M37-grade evaluation of the accuracy-token curve for your task.
  * **Eval rigor** : M37 throughout. Without it, scaling-law decisions are unfalsifiable.

This composition of concepts from M11 → M14 → M27 → M34 → M35 → M37 → M39 → M40 → M41 with M42 sitting on top is the full 2026 frontier-class recipe. The course as a whole is the manual.

#### Q&A; — About scaling laws and compute economics **Q:** Why did Kaplan get the exponents wrong, and what does Chinchilla teach us about empirical methodology? **A:** Kaplan et al. (2020) used a smaller range of model sizes and a different fitting methodology — they fit power laws across smaller-scale data and extrapolated. Hoffmann et al. (Chinchilla, 2022) ran a much more thorough empirical study with hundreds of model-size × token-count points, fitting the joint loss surface directly rather than independent power laws. The Chinchilla methodology was more careful in three specific ways: (a) varied parameters and tokens jointly rather than fitting separately, (b) used a larger and more representative range of compute scales, (c) used a more honest evaluation of "compute-optimal" by accounting for compute spent on the experiment itself. _The general lesson: empirical laws extracted from limited data ranges can be wrong outside that range_. Chinchilla's frontier extends Kaplan's; LFM2.5's 80,000:1 extends Chinchilla's. There's no reason to think the current laws are the final word. **Q:** If Llama 3 8B at 1,875:1 already beats Chinchilla, why isn't every model at 80,000:1? **A:** Three obstacles. (1) **Data scarcity**. There isn't infinite high-quality data; LFM2.5-350M's 28T tokens required substantial M35-grade data engineering. Bigger models would need proportionally more data to maintain extreme ratios. (2) **Diminishing returns past ~4 epochs**. With limited unique data, you eventually hit a ceiling. (3) **Different deployment plans**. A frontier flagship model serving complex reasoning (TTC + tools + agents) optimizes differently than an edge-deployment small model. Frontier flagships in 2026 are at ~50-200:1 ratios — past Chinchilla but nowhere near the small-model extremes. The ratio is a deployment plan signal, not a universal "better." _The "right" ratio depends on the deployment, not on a fundamental property of training._ **Q:** How should I think about the RL/pretrain compute split for a new training project? **A:** The 2026 production answer: **pretrain enough to give the model a reasonable starting policy, then invest heavily in RL**. Specifically: (1) Pretraining should cover the data-and-architecture territory — enough tokens to be in the overtrained regime, M35-quality data, M41 precision. (2) **RL is where current capability gains are concentrated** : M39's REINFORCE++/GRPO + M40's environments + verifier engineering. (3) The DeepSeek-R1 ratio (RL = 20% of pretraining) was a 2024-2025 baseline; by 2026 it's commonly closer to 50% or matching pretraining for reasoning-focused models. ProRL V2 and ScaleRL show this trajectory. (4) For non-reasoning models (chatbots, classification, generation), RL post-training is much smaller — preference fine-tuning rather than capability building. _If you're building a reasoning model, plan for RL to dominate compute; if you're building a chat model, pretraining dominates._ **Q:** What's the actual relationship between Beyond Chinchilla overtraining and test-time scaling? **A:** They're complementary; the 2026 unification paper made this explicit. Beyond Chinchilla says: _train smaller models longer when you'll run them many times at inference_. TTC says: _spend more compute per inference query on smaller models for hard tasks_. Both biases push toward the same design choice: _smaller models trained for longer_. The unified "Test-Time Scaling Makes Overtraining Compute-Optimal" paper (April 2026) showed they couple: a model designed for many-sample TTC deployment benefits from extreme overtraining because the per-sample quality bonus from longer training compounds across the many samples taken at inference. _The Era 3 picture is consistent_ : small + overtrained + heavy TTC is the modern compute-optimal frontier for reasoning workloads. **Q:** When does it make sense to scale model size instead of training duration or TTC? **A:** Three specific cases. (1) **Capability frontier requires it** : some tasks (very long context, certain multimodal capabilities, certain reasoning) require model capacity that a small overtrained model can't provide. The frontier flagships exist for these tasks. (2) **Inference cost is not a constraint** : research labs running internal experiments, government/scientific computing, or high-value queries (medical, legal) where per-query cost is acceptable. (3) **TTC infrastructure is unavailable** : deployments that can't support best-of-N or tree search (latency-critical applications, edge deployment without compute headroom for multiple samples). For most production deployments outside these cases, smaller-and-overtrained beats larger-and-Chinchilla. _This is why frontier 2026 model families ship at multiple sizes_ : different deployment scenarios sit at different points on the scaling manifold. **Q:** What's the practical limit on overtraining? Will we see 1,000,000:1 ratios? **A:** Empirically uncertain; theoretical limits suggest yes for very small models. The Sardana et al. results validated up to 10,000:1; LFM2.5 hit 80,000:1 at 350M params. For 100M-class models, ratios above 100,000:1 might be tractable with 50-100T tokens of data. **Two factors will determine where this saturates** : (a) **data availability** (M35) — high-quality natural language tops out at ~50-100T tokens by current estimates; synthetic data extends this but with quality questions; (b) **diminishing returns** — at extreme ratios, the marginal token contributes very little. The "Test-Time Scaling Makes Overtraining Compute-Optimal" paper's empirical curves suggest gradual saturation, not a hard cliff. _Plausible 2027-2028 picture: small models at 200,000-1,000,000:1 ratios for edge deployment_ ; but high-quality data engineering becomes the bottleneck before the scaling laws themselves do. 

## Code Magnets: implement Chinchilla-aware compute budget split

You're writing a function that takes a compute budget C and returns the Chinchilla-optimal model size N and tokens D. Three magnets are wrong choices.

Arrange the magnets to compute the compute-optimal allocation.

def chinchilla_optimal(compute_C, ratio=20): # Chinchilla: C ≈ 6 * N * D, optimal at D ≈ ratio * N # Solve: 6 * N * (ratio * N) = C → N = sqrt(C / (6 * ratio)) N = (compute_C / (6 * ratio)) ** 0.5 N = compute_C / (6 * ratio) D = ratio * N D = compute_C / N return {"params_N": N, "tokens_D": D, "compute_C": compute_C} def beyond_chinchilla(compute_C, inference_requests): # Sardana et al.: optimal ratio shifts up with inference demand # Heuristic: tokens_per_param ∝ log10(inference_requests) * scale_factor ratio = 20 * max(1, (inference_requests / 1e9) ** 0.4) ratio = 20 return chinchilla_optimal(compute_C, ratio=ratio)

show solution
    
    
    def chinchilla_optimal(compute_C, ratio=20):
        # Chinchilla: C ≈ 6 * N * D, optimal at D ≈ ratio * N
        # Solve: 6 * N * (ratio * N) = C → N = sqrt(C / (6 * ratio))
        N = (compute_C / (6 * ratio)) ** 0.5
        D = ratio * N
        return {"params_N": N, "tokens_D": D, "compute_C": compute_C}
    
    def beyond_chinchilla(compute_C, inference_requests):
        # Sardana et al.: optimal ratio shifts up with inference demand
        ratio = 20 * max(1, (inference_requests / 1e9) ** 0.4)
        return chinchilla_optimal(compute_C, ratio=ratio)

The traps:

  * `N = compute_C / (6 * ratio)`: linear instead of square-root. Chinchilla's compute relation is C ≈ 6 × N × D, with D = ratio × N. Substituting: C ≈ 6 × N × (ratio × N) = 6 × ratio × N². Solving for N gives N = sqrt(C / (6 × ratio)) — square root, not linear. The wrong magnet would scale parameters proportionally to compute, the Kaplan-era recommendation that Chinchilla overturned. Square root is what makes parameters and tokens grow in balance: doubling compute multiplies both N and D by sqrt(2), not one by 2× and the other by 1×.
  * `D = compute_C / N`: this gives you tokens implied by C/N (a different formula entirely — it's the cost per parameter). The correct relationship from Chinchilla is D = ratio × N — tokens scale linearly with parameters at the Chinchilla-optimal ratio. The wrong magnet ignores the ratio entirely; it would always give a 1:1 tokens-per-parameter relationship after dividing through, which is far from any meaningful scaling regime.
  * `ratio = 20` in `beyond_chinchilla`: hard-codes the Chinchilla ratio regardless of inference demand. The whole point of `beyond_chinchilla` is to adjust the ratio based on expected inference volume — high inference → higher ratio (smaller model trained longer). The wrong magnet defeats the function's purpose; calling `beyond_chinchilla(C, 1e10)` would return the same allocation as `beyond_chinchilla(C, 1)`, missing the entire Sardana-et-al insight.

The pattern: **Chinchilla's compute identity is C = 6×N×D, optimal D = ratio×N → solve for N as sqrt(C / (6×ratio)) → tokens follow**. Beyond Chinchilla wraps this with an inference-aware ratio that increases with deployment volume. The most common bug — using linear scaling for the compute-to-parameters relationship — is exactly what Kaplan got wrong; Chinchilla's contribution was getting the exponent right, and the square root is the encoding of that.

## Who does what?

Match each scaling-law concept to its real role.

Concept

Real role

Kaplan (2020)

A. Original scaling laws; recommended scaling parameters faster than data; superseded by Chinchilla.

Chinchilla (2022)

B. 20:1 tokens-to-parameters at compute-optimal training; balanced parameter/data scaling.

Beyond Chinchilla (2024-2026)

C. Train smaller and longer when inference demand is high; validated up to 10,000:1 ratios.

Test-time compute (TTC)

D. Spend extra compute per query (best-of-N, tree search); smaller models can beat larger on Pareto.

RL log-linear scaling

E. Validation accuracy increases linearly with log of average reasoning tokens during RL training.

TTS trilemma

F. Test-time scaling can optimize at most two of: accuracy, consistency, efficiency.

Overthinking

G. Longer chain-of-thought sometimes degrades accuracy; more TTC isn't monotonically better.

show solution

**Kaplan** → A  
**Chinchilla** → B  
**Beyond Chinchilla** → C  
**Test-time compute** → D  
**RL log-linear scaling** → E  
**TTS trilemma** → F  
**Overthinking** → G 

The mental shortcut: _Kaplan was wrong on exponents, Chinchilla balanced them, Beyond Chinchilla overshot for inference, TTC adds an axis, RL log-scales, the trilemma constrains, overthinking limits_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team has $5M of compute budget for a new model. They expect ~100M inference requests over the model's lifetime. Walk through how to think about the size/training-tokens decision in 2026.

show answer

Step-by-step framework, applying Beyond Chinchilla:

(1) **Express the budget in FLOPs**. $5M at April 2026 cloud prices ($3-4/hour B200) and Blackwell+NVFP4 efficiency: roughly 5e23 to 1e24 FLOPs of training compute, depending on hardware utilization. Use 8e23 FLOPs as a planning estimate.

(2) **Apply Chinchilla as a baseline**. C = 6 × N × D, D = 20×N. So N ≈ sqrt(8e23 / 120) ≈ 8.2e10 parameters ≈ 80B model on 1.6T tokens. That's the compute-optimal training answer if inference volume were tiny.

(3) **Adjust for inference volume**. 100M requests is moderate — not the 1B+ that Sardana et al. flagged for extreme ratio shifts. Apply something like ratio = 20 × (1e8/1e9)^0.4 ≈ 20 × 0.4 ≈ ~8. But that gives lower ratio, not higher. Wait — Sardana's law goes the other way: _more_ inference → _higher_ ratio. Reread: at 1B requests, train smaller and longer than 20:1. At 100M, somewhere between Chinchilla and the Sardana extremes. Use ratio ≈ 100-500 as a planning value.

(4) **Recompute with adjusted ratio**. With ratio = 200: N ≈ sqrt(8e23 / (6 × 200)) ≈ 2.6e10 = 26B model on 5.2T tokens. With ratio = 500: N ≈ 1.6e10 = 16B model on 8T tokens.

(5) **Sanity-check against published models**. Llama 3 8B at 15T tokens (1875:1 ratio) was chosen for very-high-volume deployment. Our 100M-request scenario is much lower volume, so somewhere between Chinchilla's 20:1 and Llama 3's 1875:1 makes sense. **16B-30B trained on 5-10T tokens is the right ballpark**.

(6) **Adjust for other factors**. NVFP4 (M41) effectively multiplies the compute budget by 2-3× over FP8 — so the team can either train a larger model or extend training. Data availability (M35) caps how many tokens are practical. Reasoning-task focus would shift budget toward RL (M39) post-training and reduce pretraining size.

(7) **Reserve budget for RL and eval**. Production-grade 2026 deployment shouldn't spend 100% on pretraining — keep 20-30% for RL post-training and ongoing eval (M37). Final pretraining budget: $3.5M at the size/data point above.

The output of this analysis is _a deployment plan_ , not just a model size. The 2026 framework treats compute allocation as portfolio optimization across pretraining/RL/inference/eval.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why does test-time scaling (best-of-N, tree search) make small overtrained models more attractive than large compute-optimal models?

show answer

Two compounding effects from the unifying 2026 paper:

(1) **Per-sample quality matters at TTC**. Best-of-N with N=32 means generating 32 samples per query and choosing the best (or majority vote, or via verifier). Pass@k probability for a model with per-sample correctness p is 1 - (1-p)^k — strongly nonlinear in p. A small bump in per-sample quality compounds across many samples. **Overtrained small models have higher per-sample quality than Chinchilla-balanced larger models at the same per-sample compute cost** , because the extra training has refined the model's outputs without inflating per-token cost.

(2) **Per-sample compute matters for TTC budget**. If your inference budget is fixed (say, 32× the cost of one greedy decode), you can spend it on (a) one big-model greedy decode, (b) 32 small-model samples + voting, (c) intermediate splits. Smaller models mean more samples per fixed budget. The TTC literature (Snell et al. 2024 onward) consistently shows option (b) winning on hard tasks: 32 small-model samples + voting often beats one big-model decode.

The combined effect: **smaller-and-overtrained-and-many-samples is Pareto-superior to larger-and-fewer-samples for TTC-friendly tasks**. The 2026 unifying paper made this rigorous: pretraining decisions and inference strategy are coupled; the compute-optimal pretraining ratio depends on the planned TTC mode. If you'll run the model in best-of-N mode at deployment, train it smaller and longer than Chinchilla suggests; if you'll run it greedy, Chinchilla is closer to right.

The practical implication: _the question "what size model should I train?" has no answer without specifying the inference strategy_. M34 (reasoning), M37 (eval to determine TTC accuracy curves), and M42 (this scaling question) all compose into the deployment design.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** Walk through why "more reasoning tokens" can hurt accuracy (overthinking), even though the RL scaling laws say accuracy is log-linear in reasoning tokens.

show answer

The two findings are about different things and aren't actually in conflict:

(1) **RL scaling law (training-time)** : _during RL training_ , average reasoning-token length grows as the model improves, and validation accuracy grows linearly with log of length. This is a _cross-checkpoint_ observation: as RL training progresses, both reasoning length and accuracy climb together.

(2) **Overthinking (inference-time)** : at a _fixed checkpoint_ , prompting the model to generate longer reasoning often degrades accuracy. The relationship between reasoning length and accuracy at a single model checkpoint is non-monotonic.

The two are about different axes: training trajectory vs single-checkpoint inference behavior. The RL scaling law is "models that train longer have longer reasoning AND better accuracy." Overthinking is "for a given model, forcing it to reason longer doesn't always help."

Why does overthinking happen?

  * **Distractors accumulate** : longer chains have more tokens that can mention plausible-but-wrong facts, leading the reasoning astray.
  * **Confidence drift** : late tokens in long chains are conditioned on potentially incorrect early tokens; the conditional distribution loses calibration.
  * **Off-distribution territory** : training distributions have characteristic length distributions; forcing significantly longer reasoning pushes the model into regions it wasn't trained for.

The mitigation (Plan-and-Budget, 2025-2026): **adaptive token budgets per query**. Use a difficulty classifier to decide how many tokens to allow; easy queries get short budgets (preventing overthinking); hard queries get long budgets (allowing genuine deep reasoning). The reported result is up to 70% accuracy gains with 39% token reduction over flat budgets.

The reconciliation: _RL training improves the average accuracy at all length budgets, but the optimal length budget per query is task-dependent and bounded_. A model that's been trained to reason for longer can use those longer chains effectively when needed; forcing it to do so when not needed is wasteful and sometimes harmful.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Sketch the compute-economic argument for whether a startup should train its own 7B reasoning model or use API access to a frontier model. April 2026 numbers.

show answer

The framework: compare lifetime training+inference cost of self-trained model vs API access for the expected query volume.

**Self-trained 7B reasoning model (April 2026 numbers)** :

  * Pretraining: 7B at 1875:1 = 13T tokens. ~$300-800K on Blackwell + NVFP4 (M41).
  * RL post-training: REINFORCE++ on a custom domain (M39, M40). $50-200K depending on environment complexity and target capability.
  * Engineering: 2-4 engineers × 3-6 months for setup, eval, deployment. ~$300-600K loaded cost.
  * Inference at deployment: a 7B model on Blackwell + NVFP4 costs roughly $0.05-0.2 per 1M tokens at scale (4-10× cheaper than M41's $0.02 reference for GPT-OSS-120B because 7B is smaller).
  * **Total upfront cost: $700K - $1.6M; ongoing inference: ~$0.1/Mtoken**.

**Frontier API access (April 2026)** :

  * Claude Opus 4.6 / GPT-5.4 / Gemini 3.1 Pro at $3-15 per million output tokens (varies by provider and tier).
  * No upfront engineering cost beyond integration (~weeks of one engineer).
  * Quality typically higher than 7B for general tasks; comparable on narrow specialized tasks if the 7B has been RL'd effectively.

**The crossover analysis** :

(a) **Volume-dependent**. At 100M tokens of inference per month, frontier API costs $300K-$1.5M/month; self-trained costs ~$10K/month. Crossover happens when API costs exceed the self-train upfront in some payback period. At 100M tokens/month, API is more expensive after 1-2 months.

(b) **Domain-specificity**. If the task is narrow (medical billing, customer support for specific product, code completion in specific framework), a 7B + RL on M40-grade environment can match or exceed frontier on the specific task. General reasoning, broad knowledge, complex math — frontier wins.

(c) **Privacy / sovereignty**. Self-hosted models are required for regulated data (healthcare, finance, government). API isn't an option regardless of economics.

(d) **Speed of iteration**. APIs let you ship fast; self-trained models lock you into your training run for months. For early-stage startups, API access is usually the right starting point even when economics favor self-training long-term.

(e) **Vendor risk**. API providers can change pricing, discontinue models, or restrict use cases. Self-hosted models lock in your stack.

**The 2026 advice** : most startups should start with frontier API; switch to self-trained when (a) volume exceeds ~50M tokens/month sustained, (b) task is narrow enough that 7B can match frontier on it, AND (c) privacy/cost economics justify the upfront investment. The Cursor pattern (M40 territory) is exactly this: started on frontier APIs, now post-training their own models because volume justifies it. _The decision is portfolio-level, not all-or-nothing_ : many production systems use frontier APIs for some queries and self-trained models for others.

### What just happened?

  * Scaling laws went through three eras. **Era 1 (Kaplan 2020)** : scale parameters faster than data. **Era 2 (Chinchilla 2022)** : 20 tokens per parameter at compute-optimal training. **Era 3 (Beyond Chinchilla, 2024-2026)** : train smaller and longer when serving high inference volume.
  * The Chinchilla compute identity: **C ≈ 6 × N × D, optimal D = 20 × N → N = sqrt(C / 120)**. Square root, not linear.
  * **The Chinchilla Trap** : training compute-optimal at deployment time means using a model larger than you should for a given quality target. Llama 1 paper flagged this; Sardana et al. (2024-2025) formalized it.
  * The overtraining trajectory: GPT-3 1.7:1 → Llama 1 7B at 142:1 → Llama 2 at 284:1 → Llama 3 8B at 1,875:1 → Qwen3-0.6B at 60,000:1 → **Liquid AI LFM2.5-350M at 80,000:1** (April 2026 record).
  * **Compute splits three ways in 2026** : pretraining (base capability), RL post-training (M39, M40 — reasoning + agentic), test-time compute (M34 — best-of-N, tree search, adaptive allocation).
  * **RL training scaling laws are log-linear in reasoning tokens** : validation accuracy increases linearly with log of average generated tokens across RL checkpoints (Seed-OSS-36B paper, April 2026; Art of Scaling RL Compute, October 2025).
  * **RL post-training compute is rapidly catching pretraining** : DeepSeek-R1 at 20% of pretraining ($1M); ProRL V2 (Feb 2026) ran 700+ stable steps; **Epoch AI projects parity by late 2026**.
  * **Inference scaling laws (Snell et al. 2024 onward)** : smaller models combined with advanced inference algorithms (best-of-N, weighted voting, tree search) can be Pareto-optimal vs scaling parameters. Llemma-7B + tree search beat Llemma-34B on MATH at the same compute.
  * **TTS trilemma** : test-time scaling optimizes at most two of accuracy, consistency, efficiency. Production deployment chooses two.
  * **Overthinking** : longer CoT often degrades accuracy at a fixed model checkpoint. Plan-and-Budget (2025-2026) achieves 70% accuracy gain + 39% token reduction via adaptive budgeting.
  * **Test-Time Scaling Makes Overtraining Compute-Optimal (April 2026)** : unifying paper showing pretraining and inference scaling decisions COUPLE. A model designed for many-sample TTC inference benefits from extreme overtraining because per-sample quality bonuses compound across samples.
  * **Scaling Laws for Precision (Kumar et al. 2024)** : numerical precision interacts with scaling. Training in lower precision may be compute-optimal; post-training quantization degradation scales with training tokens. M14 + M41 + M42 compose.
  * **Data-constrained scaling (Muennighoff et al. 2023)** : ~4 epochs over fresh data ≈ same-volume fresh data; beyond that diminishing returns. Code mixing as 2× useful data multiplier. Caps the overtraining trajectory.
  * The 2026 economic ranges: **Llama 3.1 405B for ~$10M with NVFP4 + Blackwell** (10 minutes on 5,120 GPUs, MLPerf v5.1); inference at **$0.02 per million tokens** for GPT-OSS-120B; **RL post-training at $50K-$1M** depending on scale.
  * The reflex for 2026 training projects: ask "**volume? reasoning needed? hardware? data ceiling? TTC strategy? precision plan?** " — the six questions determine the compute split. The answer isn't "size of model" alone; it's a position on the (pretrain, RL, test-time) compute manifold.

Module 43 (next, if Part XI continues to its planned end) would tackle **embodied AI & robotics RL** — VLA (vision-language-action) models, RLinf real-world stack, sim-to-real transfer, the bridge between LLM RL and robotics. Closes Part XI; addresses the final gap from the analysis.
