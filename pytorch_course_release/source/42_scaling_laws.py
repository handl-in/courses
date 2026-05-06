#!/usr/bin/env python3
"""Module 42: Scaling laws & compute economics — full HF vibe, April 2026 current."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part XI · Module 42 · April 2026 currency</div>
  <h1 class="module-title"><em>Scaling laws &amp; compute economics:</em> where to spend your FLOPs</h1>
  <p class="module-sub">— from Chinchilla's 20:1 to LFM2.5's 80,000:1, why every frontier model is now overtrained, the three-way pretrain/RL/test-time compute tradeoff, and the economics of choosing what to scale</p>
</div>

<p>Scaling laws are the closest thing this field has to a theory. From Kaplan et al. (2020) through Chinchilla (Hoffmann et al., 2022) through the inference-aware reformulations (Sardana/MosaicML 2024-2025) through the unified test-time compute scaling work (2026), each generation of laws answers a different version of the same question: <em>given a fixed compute budget, how should you spend it?</em> The 2022 answer was "20 tokens per parameter, balance training compute between size and data." The 2026 answer is more interesting: <em>it depends on how much inference you'll do, what test-time compute strategy you'll use, what numerical precision you'll train in, and whether you're doing pretraining or RL post-training.</em></p>

<p>This module is foundational and threads through everything in M1-M41. Where M11 talked about training a model, M14 talked about precision, M27 about serving, M34 about reasoning, M39 about RL — this module is about <em>choosing how much of each</em>. By the end you'll know why Chinchilla's ratio is too aggressive for production deployment, why frontier 2026 models train at ratios 1000-4000× higher than Chinchilla-optimal, when test-time compute beats parameter scaling, what the RL scaling laws look like (log-linear in reasoning tokens), and how to reason about the economics of "should I train a bigger model or run my smaller one for longer at inference?"</p>

<div class="keyidea">
Scaling laws went through three eras. <strong>Era 1 (Kaplan 2020)</strong>: scale parameters faster than data. <strong>Era 2 (Chinchilla 2022)</strong>: parameters and tokens should grow equally — 20 tokens per parameter at compute-optimal training. <strong>Era 3 (Beyond Chinchilla, 2024-2026)</strong>: inference cost matters — train smaller and longer when serving many requests. Production frontier models are now massively overtrained: Llama 3 8B at 1,875:1, Qwen3-0.6B at 60,000:1, <strong>Liquid AI LFM2.5-350M at 80,000:1</strong> (the public April 2026 record). The compute budget now splits three ways: <strong>pretraining</strong> (expanding base capability), <strong>RL post-training</strong> (M39 territory; ~20% of pretraining compute for DeepSeek-R1, scaling fast — projected to match pretraining by late 2026 per Epoch AI), <strong>test-time compute</strong> (M34 reasoning; smaller models with best-of-N or tree search outperform larger models on Pareto curves). RL scaling laws are <strong>log-linear in reasoning tokens</strong>: accuracy increases linearly with log of average tokens generated. Test-time scaling laws favor smaller-than-Chinchilla models for inference-heavy deployments; Sardana et al. validated up to 10,000:1 ratios. The key 2026 finding: <strong>test-time scaling makes overtraining compute-optimal</strong>. The three regimes interact: precision (M14, M41) modifies all three (Scaling Laws for Precision, Kumar et al.); data constraints (Muennighoff et al.) cap pretraining; RL recipes (DAPO/REINFORCE++) determine RL efficiency. <em>The trade is no longer "bigger or more data?" — it's "where on the pretrain × RL × inference compute manifold should I sit?"</em>
</div>

<h2>Two new faces — closing Part XI</h2>

<div class="character" style="--c: #d4a017;">
  <div class="avatar" style="background: #d4a017; color: #fff;">L</div>
  <div>
    <p class="who">Scaling Law</p>
    <p class="name">"I'm an empirical formula relating compute, parameters, data, and quality. I rule training-budget decisions."</p>
    <p class="says">Give me your compute budget; I'll tell you how to spend it. Chinchilla taught me 20:1 for compute-optimal training. The Beyond-Chinchilla generation taught me that compute-optimal isn't deployment-optimal: if you're going to serve a model 1B times at inference, you want it smaller and trained longer, even if it costs more upfront. The reasoning-model era added another axis: I now decompose into <strong>pretrain × RL post-train × test-time compute</strong>, and the optimal mix depends on your task and traffic. <em>I'm not a single law anymore</em>; I'm a family of curves you have to navigate. By 2026, model families like Llama and Qwen ship at multiple sizes specifically because different deployment scales sit at different points on me.</p>
  </div>
</div>

<div class="character" style="--c: #c1502e;">
  <div class="avatar" style="background: #c1502e; color: #fff;">$</div>
  <div>
    <p class="who">Economist</p>
    <p class="name">"I translate FLOPs into dollars. Training cost vs inference cost vs RL cost — I track all three."</p>
    <p class="says">Compute-optimal isn't dollar-optimal. Training a Chinchilla-optimal 70B model costs ~$10M; if you serve 100M requests at inference, the inference compute totals $50M. Train a smaller 13B for 5× more tokens (still improves) and your training cost rises to $15M but inference drops to $10M — net win. <strong>I'm what the Beyond-Chinchilla literature is really about</strong>. In 2026, I add two more line items: <strong>RL post-training compute</strong> (DeepSeek-R1 spent $1M on RL, 20% of pretraining; ScaleRL and ProRL V2 push this much higher) and <strong>test-time compute</strong> (running a model in best-of-32 mode is 32× per-query, but lets you use a smaller model). Production AI economics is portfolio optimization across these three pools.</p>
  </div>
</div>

<h2>Era 1: Kaplan (2020) — the original scaling laws</h2>

<p>Kaplan et al. ("Scaling Laws for Neural Language Models," 2020) found that test loss decreases as a power law in three quantities: parameters N, training tokens D, and compute C. The relationships looked like:</p>

<pre><code>L(N) ≈ (N_c / N)^α_N      <span class="com"># loss decreases with model size</span>
L(D) ≈ (D_c / D)^α_D      <span class="com"># loss decreases with training tokens</span>
L(C) ≈ (C_c / C)^α_C      <span class="com"># loss decreases with compute</span></code></pre>

<p>The exponents in Kaplan's measurements suggested that <strong>parameters should grow faster than data</strong>. For a fixed compute budget, the recommendation was to scale parameters aggressively — about 5× more than data per FLOP doubling. This is why GPT-3 had 175B parameters but was trained on only 300B tokens (a 1.7:1 ratio).</p>

<p>The Kaplan recommendations dominated 2020-2022. Models grew aggressively in parameter count; data scaling lagged. <em>Then Chinchilla came along and showed Kaplan was wrong about the exponents.</em></p>

<h2>Era 2: Chinchilla (2022) — the 20:1 ratio</h2>

<p>Hoffmann et al. ("Training Compute-Optimal Large Language Models," 2022) ran a much more careful empirical study — training hundreds of models across the parameter/data plane, fitting the loss surface, and finding the compute-optimal frontier. Their conclusion was sharper than Kaplan's: <strong>parameters and tokens should grow approximately equally</strong>. For a fixed compute budget, the optimal split is roughly equal FLOPs spent on bigger models and on more data.</p>

<p>The practical translation: <strong>~20 tokens per parameter</strong> at compute-optimal training. A 70B model should be trained on ~1.4T tokens to be compute-optimal; conversely, GPT-3 (175B params, 300B tokens, 1.7:1 ratio) was massively undertrained — the same compute would have produced a much better model at ~30B params and 600B tokens, or vice versa.</p>

<p>To validate, DeepMind trained Chinchilla itself: <strong>70B parameters, 1.4T tokens, beat GPT-3 (175B) on most evals using a fraction of the compute</strong>. The result was decisive enough to overturn Kaplan's recommendations almost overnight.</p>

<h3>Why the 20:1 ratio mattered</h3>

<p>The Chinchilla finding had two practical implications:</p>

<ul>
  <li><strong>Most models from the GPT-3 era were undertrained</strong>. They had too many parameters relative to data; reducing parameter count and adding tokens would have produced better models.</li>
  <li><strong>Compute could be redirected</strong>. Instead of scaling to ever-larger parameter counts, the same compute spent on 20:1-balanced training would produce stronger models.</li>
</ul>

<p>The 2022-2023 model wave (Llama 1, OPT, BLOOM, MPT, Falcon) explicitly adopted Chinchilla-aware ratios. Llama 1 (Touvron et al., February 2023) trained 7B/13B/33B/65B models on 1T-1.4T tokens — <em>much</em> closer to compute-optimal than GPT-3.</p>

<h2>The Chinchilla Trap and Era 3: Beyond Chinchilla</h2>

<p>By 2023, a problem became visible. Chinchilla-optimal training optimizes for <em>training compute</em> only. It ignores inference cost. For a model that's deployed and queried billions of times, inference compute dominates training compute by orders of magnitude — a 70B Chinchilla-optimal model is expensive to run, and a smaller model trained for longer (well beyond 20:1) might serve the same accuracy at far lower deployment cost.</p>

<p>This is the <strong>Chinchilla Trap</strong>: training compute-optimal at deployment time means using a model larger than you should.</p>

<p>The Llama 1 paper (Touvron et al., 2023) flagged this explicitly — it noted that loss continues to decrease beyond Chinchilla-optimal ratios, and that smaller-but-longer-trained models would be cheaper to deploy. Llama 1's smallest 7B model was trained at <strong>142:1 ratio</strong> (1T tokens / 7B params) — 7× past Chinchilla's recommendation.</p>

<h3>Sardana et al. (MosaicML 2024): the inference-aware scaling law</h3>

<p>The first systematic treatment of "Beyond Chinchilla" was Sardana et al.'s "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws" (2024, refined April 2025). They modified the Chinchilla loss surface to include inference cost, parameterized by expected number of inference requests. The headline finding:</p>

<p><strong>Researchers expecting reasonably large inference demand (~1B requests) should train models smaller and longer than Chinchilla-optimal</strong> — substantially so. Their experiments validated up to <strong>10,000 tokens per parameter ratios</strong>, with quality continuing to improve well past Chinchilla's 20:1.</p>

<p>The argument is concrete economics. If a 70B Chinchilla-optimal model costs $10M to train and $50M to serve over its lifetime, vs a 13B-trained-for-5×-more-tokens that costs $15M to train and $10M to serve at the same quality, the smaller model wins by $35M. <em>For high-traffic deployments, training duration is the right knob to spend extra compute on.</em></p>

<h3>The Llama trajectory: explicit overtraining</h3>

<p>Successive Llama generations went further past Chinchilla:</p>

<div class="table-wrap">
<table>
<caption>The overtraining trajectory: tokens-per-parameter ratios over time</caption>
<thead><tr><th>Model</th><th>Year</th><th>Params</th><th>Tokens</th><th>Ratio</th><th>Multiple of Chinchilla</th></tr></thead>
<tbody>
<tr><td>GPT-3</td><td>2020</td><td>175B</td><td>300B</td><td>1.7:1</td><td>0.085× (severely undertrained)</td></tr>
<tr><td>Chinchilla</td><td>2022</td><td>70B</td><td>1.4T</td><td>20:1</td><td>1× (the canonical reference)</td></tr>
<tr><td>Llama 1 7B</td><td>Feb 2023</td><td>7B</td><td>1T</td><td>142:1</td><td>7×</td></tr>
<tr><td>Llama 2 7B</td><td>Jul 2023</td><td>7B</td><td>2T</td><td>284:1</td><td>14×</td></tr>
<tr><td>Llama 3 8B</td><td>Apr 2024</td><td>8B</td><td>15T</td><td>1,875:1</td><td>94×</td></tr>
<tr><td>Qwen3-0.6B</td><td>Apr 2025</td><td>0.6B</td><td>36T</td><td>60,000:1</td><td>3,000×</td></tr>
<tr><td><strong>Liquid AI LFM2.5-350M</strong></td><td><strong>Apr 2026</strong></td><td><strong>0.35B</strong></td><td><strong>28T</strong></td><td><strong>80,000:1</strong></td><td><strong>4,000×</strong></td></tr>
</tbody>
</table>
</div>

<p>The pattern: as deployment becomes the dominant cost, training ratios explode. LFM2.5-350M's <strong>80,000:1</strong> ratio is the public April 2026 record — a 350M-parameter model trained on 28 trillion tokens. The math: ~10²² FLOPs of pretraining, deployable at orders-of-magnitude lower inference cost than any Chinchilla-optimal model would offer at similar quality. <em>The Era 3 recipe is to train tiny models for absurdly long</em>, when your deployment requires inference-cheap models.</p>

<h3>Data constraints and the limits of overtraining</h3>

<p>Naïvely you'd think you could keep increasing tokens forever. Two realities push back:</p>

<ul>
  <li><strong>Data scarcity</strong>. The web has a finite quantity of high-quality natural language. By 2025-2026, frontier teams hit walls — there isn't enough fresh data to train at 100,000:1 ratios for ever-bigger models. M35's data engineering becomes the bottleneck.</li>
  <li><strong>Diminishing returns from repeated data</strong> (Muennighoff et al., 2023, "Scaling Data-Constrained Language Models"): training for multiple epochs over the same data still helps, but with rapidly diminishing returns. They validated up to ~4 epochs as broadly useful; up to 1500 epochs at small scales but with sharply diminishing benefit. <strong>The first 1-2 epochs over fresh data are worth ~the same as same-volume fresh data; beyond 4 epochs, additional training is mostly wasted.</strong></li>
  <li><strong>Code data as a 2× multiplier</strong> (same paper): mixing in 50% code data effectively doubles useful training tokens for natural language tasks; some tasks (state-tracking, reasoning) actively improve with code mix.</li>
</ul>

<p>The April 2026 reality: LFM2.5-350M's 80,000:1 is achieved partly by aggressive data curation (M35's province) and partly by accepting some repetition. <em>How much further this can go before hitting hard data limits is an open empirical question.</em></p>

<h2>The three-pool decomposition: pretraining × RL × test-time compute</h2>

<p>By 2026, the compute budget for a frontier model splits across three pools, not one:</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 380" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">The three compute pools — and how they trade off</text>

  <!-- Three boxes -->
  <g transform="translate(20, 50)">
    <rect x="0" y="0" width="220" height="200" fill="#fff5d8" stroke="#d4a017" stroke-width="2"/>
    <text x="110" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">① Pretraining compute</text>
    <text x="110" y="38" text-anchor="middle" font-size="9" fill="#6b5d4f">Base capability</text>

    <text x="20" y="64" font-size="9" font-weight="700" fill="#1a1612">Knob: tokens × params</text>
    <text x="20" y="80" font-size="9" fill="#1a1612">  Chinchilla 20:1 = compute-opt.</text>
    <text x="20" y="94" font-size="9" fill="#1a1612">  Beyond Chinchilla: 100-80,000:1</text>
    <text x="20" y="108" font-size="9" fill="#1a1612">  for inference-heavy deployment</text>

    <text x="20" y="132" font-size="9" font-weight="700" fill="#1a1612">Limits:</text>
    <text x="20" y="146" font-size="9" fill="#1a1612">  • Data scarcity (M35)</text>
    <text x="20" y="160" font-size="9" fill="#1a1612">  • Diminishing returns past ~4 epochs</text>
    <text x="20" y="174" font-size="9" fill="#1a1612">  • Loss of generalization at extremes</text>

    <text x="20" y="192" font-size="9" font-style="italic" fill="#c1502e">Frontier 2026: 1e25-1e26 FLOPs</text>
  </g>

  <g transform="translate(260, 50)">
    <rect x="0" y="0" width="220" height="200" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="110" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">② RL post-training</text>
    <text x="110" y="38" text-anchor="middle" font-size="9" fill="#6b5d4f">Reasoning + agentic</text>

    <text x="20" y="64" font-size="9" font-weight="700" fill="#1a1612">Knob: rollouts × env interactions</text>
    <text x="20" y="80" font-size="9" fill="#1a1612">  M39: distributed RLHF/GRPO</text>
    <text x="20" y="94" font-size="9" fill="#1a1612">  M40: agentic RL environments</text>
    <text x="20" y="108" font-size="9" fill="#1a1612">  Log-linear scaling in tokens</text>

    <text x="20" y="132" font-size="9" font-weight="700" fill="#1a1612">Trajectory:</text>
    <text x="20" y="146" font-size="9" fill="#1a1612">  • DeepSeek-R1: $1M, 20% of pretrain</text>
    <text x="20" y="160" font-size="9" fill="#1a1612">  • ProRL V2 (Feb 2026): 700+ steps</text>
    <text x="20" y="174" font-size="9" fill="#1a1612">  • ScaleRL (Oct 2025): frontier scale</text>

    <text x="20" y="192" font-size="9" font-style="italic" fill="#c1502e">Projected: ≈ pretrain by late 2026</text>
  </g>

  <g transform="translate(500, 50)">
    <rect x="0" y="0" width="220" height="200" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="110" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">③ Test-time compute</text>
    <text x="110" y="38" text-anchor="middle" font-size="9" fill="#6b5d4f">Per-query reasoning</text>

    <text x="20" y="64" font-size="9" font-weight="700" fill="#1a1612">Knob: tokens × samples per query</text>
    <text x="20" y="80" font-size="9" fill="#1a1612">  M34: chain-of-thought, BoN, MCTS</text>
    <text x="20" y="94" font-size="9" fill="#1a1612">  Smaller model + N samples can</text>
    <text x="20" y="108" font-size="9" fill="#1a1612">  beat larger model on Pareto</text>

    <text x="20" y="132" font-size="9" font-weight="700" fill="#1a1612">Tradeoffs:</text>
    <text x="20" y="146" font-size="9" fill="#1a1612">  • Per-query cost scales linearly</text>
    <text x="20" y="160" font-size="9" fill="#1a1612">  • Latency scales with depth</text>
    <text x="20" y="174" font-size="9" fill="#1a1612">  • Overthinking degrades accuracy</text>

    <text x="20" y="192" font-size="9" font-style="italic" fill="#c1502e">M37 calibration mandatory</text>
  </g>

  <!-- Bottom unification -->
  <g transform="translate(20, 280)">
    <rect x="0" y="0" width="700" height="80" fill="#fff8a8" stroke="#1a1612" stroke-width="2"/>
    <text x="350" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">The unified question: where on the (pretrain, RL, test-time) compute manifold should I sit?</text>
    <text x="20" y="44" font-size="10" fill="#1a1612">  Inference-heavy + low task complexity → small model, long pretrain, modest RL, minimal TTC</text>
    <text x="20" y="58" font-size="10" fill="#1a1612">  Inference-light + high reasoning → moderate model, heavy RL, generous TTC (best-of-N, tree search)</text>
    <text x="20" y="72" font-size="10" font-weight="700" fill="#1f5f5b">  → "Test-time scaling makes overtraining compute-optimal" (2026): the three pools COUPLE</text>
  </g>
</svg>
</div>

<p>The three pools have substantively different scaling laws and substantively different costs:</p>

<ul>
  <li><strong>Pretraining</strong> scales with parameters × tokens. Compute is loss-bound; you spend it once. Frontier 2026 pretraining: 10²⁵-10²⁶ FLOPs for top-tier models. Cost: $10M-$200M depending on size and hardware.</li>
  <li><strong>RL post-training</strong> scales with rollouts × environment interactions. Compute is reward-bound; the policy improves until it saturates the reward signal. Frontier 2026: comparable scale to pretraining or somewhat less; rapidly growing. DeepSeek-R1 spent ~$1M on RL (20% of pretraining); ProRL V2 (Feb 2026) ran 700+ stable RL steps; <strong>Epoch AI's Q1 2026 estimate: reasoning training compute will converge with overall frontier compute by late 2026</strong>.</li>
  <li><strong>Test-time compute</strong> scales with tokens × samples per query. Compute is per-query; total cost scales linearly with deployment volume. Each query at best-of-32 mode is 32× more expensive than greedy decoding, but accuracy improves substantially for hard tasks.</li>
</ul>

<h2>RL scaling laws: log-linear in reasoning tokens</h2>

<p>The Era-3 generation surfaced a specific RL scaling regularity. The "Scaling Reasoning Tokens via RL and Parallel Thinking" paper (Apr 2026, Seed-OSS-36B trained on competitive programming):</p>

<p><strong>During RL training, validation accuracy increases linearly with the logarithm of average generated reasoning tokens.</strong> As successive RL checkpoints generate longer reasoning traces, accuracy follows a clear log-linear trend.</p>

<pre><code><span class="com"># Empirical observation across RL checkpoints during training:</span>
accuracy = α * <span class="fn">log</span>(avg_reasoning_tokens) + β

<span class="com"># Two ways to shift the trajectory (per the paper):</span>
<span class="com">#   1. Verification RL warmup: raises β (starting point higher)</span>
<span class="com">#   2. Randomized clipping: increases α (steeper slope)</span></code></pre>

<p>The "Art of Scaling RL Compute" paper (Oct 2025) corroborated this at scale, validating frontier RL recipes (DAPO clipping, asymmetric loss aggregation, length interruptions to prevent runaway generation). The general pattern: <em>RL scaling is well-behaved and log-linear in reasoning tokens, similar to the test-time scaling laws but during training</em>.</p>

<p>The implication for compute economics: <strong>RL training compute provides predictable returns, much like pretraining compute does</strong>. You can budget RL compute based on target token-length goals, similar to budgeting pretraining tokens against target loss. This was unclear in early 2025; the 2025-2026 papers settled it.</p>

<h2>Test-time compute scaling: smaller can beat bigger</h2>

<p>The "Inference Scaling Laws" paper (Snell et al., 2024, refined 2025) ran a careful empirical study: <strong>given a compute budget, when is it better to run a smaller model with more inference compute vs a bigger model with less?</strong></p>

<p>Their finding, validated across model families (Pythia, Llemma, Llama): <em>smaller models combined with advanced inference algorithms (best-of-N, tree search, weighted voting) can be Pareto-optimal compared to scaling parameters</em>. Specific result: <strong>Llemma-7B with their tree search consistently outperformed Llemma-34B across all tested inference strategies on the MATH benchmark</strong> — a 5× smaller model winning by spending compute on inference instead of parameters.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 320" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Inference scaling: smaller model + N samples vs larger model</text>

  <!-- Pareto curve sketch -->
  <g transform="translate(60, 50)">
    <!-- axes -->
    <line x1="0" y1="200" x2="600" y2="200" stroke="#1a1612" stroke-width="1.5"/>
    <line x1="0" y1="200" x2="0" y2="20" stroke="#1a1612" stroke-width="1.5"/>
    <text x="-40" y="110" font-size="10" font-weight="700" fill="#1a1612" transform="rotate(-90 -40 110)">accuracy →</text>
    <text x="300" y="240" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">total compute (FLOPs, log scale) →</text>

    <!-- "small model with more samples" curve -->
    <path d="M 30 180 Q 100 130 200 90 T 400 50 T 580 35" stroke="#1f5f5b" stroke-width="2.5" fill="none"/>
    <text x="280" y="80" font-size="10" font-weight="700" fill="#1f5f5b">7B + best-of-N inference</text>

    <!-- "big model greedy" markers -->
    <circle cx="60" cy="170" r="5" fill="#c1502e"/>
    <circle cx="120" cy="155" r="5" fill="#c1502e"/>
    <circle cx="200" cy="135" r="5" fill="#c1502e"/>
    <circle cx="320" cy="115" r="5" fill="#c1502e"/>
    <circle cx="450" cy="100" r="5" fill="#c1502e"/>
    <text x="200" y="170" font-size="10" font-weight="700" fill="#c1502e">larger models, greedy decode</text>

    <!-- Crossover annotation -->
    <line x1="200" y1="90" x2="200" y2="135" stroke="#d4a017" stroke-width="1" stroke-dasharray="3 2"/>
    <text x="220" y="115" font-size="9" fill="#d4a017">Pareto crossover</text>

    <!-- key result -->
    <g transform="translate(20, 220)">
      <rect x="0" y="0" width="560" height="40" fill="#fff8a8" stroke="#1a1612" stroke-width="1"/>
      <text x="280" y="18" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Llemma-7B with tree search beats Llemma-34B greedy on MATH at the same compute (Snell et al. 2024)</text>
      <text x="280" y="34" text-anchor="middle" font-size="9" fill="#1a1612">→ for hard reasoning tasks, inference compute &gt; parameter scaling on Pareto</text>
    </g>
  </g>
</svg>
</div>

<h3>The TTS trilemma: accuracy, consistency, efficiency</h3>

<p>The "Art of Scaling Test-Time Compute" paper (December 2025) introduced the <strong>test-time scaling trilemma</strong>: you can optimize for at most two of accuracy, consistency, and efficiency.</p>

<ul>
  <li><strong>High accuracy + high consistency</strong> (e.g., self-consistency with many samples + majority vote): expensive — many samples per query.</li>
  <li><strong>High accuracy + high efficiency</strong> (e.g., adaptive depth — short for easy, long for hard): inconsistent — same query may take different paths different times.</li>
  <li><strong>High consistency + high efficiency</strong> (greedy decoding): low accuracy on hard tasks.</li>
</ul>

<p>Production deployment chooses two; the third is the cost.</p>

<h3>Compute-optimal TTC allocation</h3>

<p>The follow-on work — Snell et al. (2024) for the foundational paper, and the April 2026 "Adaptive Test-Time Compute Allocation" paper — addressed the practical question: <em>given a query, how much inference compute should you spend?</em></p>

<p>The 2026 framework: train a difficulty classifier (often tree-based, since allocation boundaries are non-linear) that predicts how much TTC the query needs; use a Lagrangian relaxation with a single dual variable λ controlling the accuracy-cost tradeoff. The result is principled adaptive allocation: easy queries use 1× compute, hard queries use 32× or more, average compute stays bounded.</p>

<p>The published results: <strong>up to 70% accuracy gains with 39% token reduction</strong> compared to flat-allocation baselines (Plan-and-Budget, May 2025; refined 2026). The technique works across DeepSeek-R1-Distill, QwQ-32B, OpenAI o4-mini.</p>

<h3>Overthinking: when more TTC hurts</h3>

<p>Counter-finding from 2025-2026 literature: <strong>longer chain-of-thought often degrades accuracy</strong>. Models can overthink, generate unnecessarily long and tangential reasoning paths, and arrive at worse answers than they would with shorter reasoning.</p>

<ul>
  <li>Ghosal et al. (2025) and Hassid et al. (2025) showed that longer CoT chains often degrade accuracy on benchmark tasks.</li>
  <li>Inverse scaling effects (Gema et al., 2025): more reasoning sometimes makes things worse on specific task types.</li>
  <li>Plan-and-Budget (2025-2026): explicit token budget allocation produces 39% token reduction with accuracy improvements, showing the prior expense was largely waste.</li>
</ul>

<p>The implication: <em>more test-time compute is not monotonically better</em>. Adaptive allocation (spend more on hard queries) and budgeted allocation (cap easy queries) both outperform flat scaling. M37's eval rigor matters here — measure accuracy as a function of TTC for your tasks; don't assume the relationship is monotonic.</p>

<h2>The unifying 2026 finding: TTS makes overtraining compute-optimal</h2>

<p>The most important recent paper on scaling laws — "Test-Time Scaling Makes Overtraining Compute-Optimal" (April 2026) — unified the pretraining and inference scaling literature. The key insight: <em>pretraining and test-time compute decisions are coupled, not independent</em>.</p>

<p>The argument:</p>

<ol>
  <li>Test-time scaling (best-of-N, tree search) requires sampling many times from the model.</li>
  <li>Sample quality matters: pass@k probability depends nonlinearly on per-sample accuracy.</li>
  <li>Smaller models are cheaper per sample but weaker per sample.</li>
  <li>The "right" pretraining decision depends on how the model will be used at inference: a model designed for many-sample TTC inference should be smaller (faster per sample) and trained longer (higher per-sample accuracy).</li>
</ol>

<p><strong>Conclusion: when test-time scaling is part of deployment, the compute-optimal pretraining ratio shifts toward overtraining</strong> — exactly the trajectory we've seen in production (Llama 3 → Qwen3 → LFM2.5).</p>

<p>The paper validated this empirically: across model sizes and pretraining schedules, models trained at higher tokens-per-parameter ratios have better TTC scaling curves. The 80,000:1 ratio of LFM2.5-350M is consistent with a deployment plan that includes substantial test-time compute.</p>

<h2>Scaling laws for precision (M14 + M41 connection)</h2>

<p>Kumar et al. ("Scaling Laws for Precision," November 2024) added another axis: <strong>numerical precision interacts with the scaling laws</strong>. Their findings, validated across 465 pretraining runs and model sizes up to 1.7B trained on up to 26B tokens:</p>

<ul>
  <li><strong>Training in lower precision may be compute-optimal</strong> for some configurations. The compute saved per FLOP at FP8 or NVFP4 (M41) can outweigh the slight quality loss for many tasks.</li>
  <li><strong>Post-training quantization degradation scales with training tokens</strong>. A model trained on 10T tokens loses more accuracy when post-training-quantized than the same model trained on 1T tokens — suggesting that inference-time precision changes interact with training-time pretraining decisions.</li>
  <li><strong>A unified scaling formula</strong>: predicts loss as a function of parameters, tokens, AND precision. Different parts of training/inference can be in different precisions; the formula composes.</li>
</ul>

<p>The practical 2026 implication: <strong>"how should I train" includes a precision dimension that interacts with the size and data dimensions</strong>. M14 + M41 + this scaling work all compose. NVFP4 training at 80,000:1 ratios (LFM2.5 territory) is a coherent design point that exploits all three insights.</p>

<h2>The economics: real costs in 2026</h2>

<p>Translating compute into dollars at April 2026 cloud prices:</p>

<div class="table-wrap">
<table>
<caption>Compute cost estimates, April 2026</caption>
<thead><tr><th>Workload</th><th>Compute</th><th>Hardware</th><th>Approx cost</th></tr></thead>
<tbody>
<tr><td>7B model, Chinchilla-optimal (140B tokens)</td><td>~5e21 FLOPs</td><td>8× H100 for 1 week</td><td>$10-30K</td></tr>
<tr><td>7B model, Llama-3-style (15T tokens)</td><td>~5e23 FLOPs</td><td>64× H100 for 4 weeks</td><td>$300-800K</td></tr>
<tr><td>70B model, Chinchilla-optimal (1.4T tokens)</td><td>~6e23 FLOPs</td><td>128× H100 for 4 weeks</td><td>$500K-1M</td></tr>
<tr><td>70B model, overtrained (15T tokens)</td><td>~6e24 FLOPs</td><td>512× H100 for 4 weeks</td><td>$3-7M</td></tr>
<tr><td>405B model (Llama-3.1 scale)</td><td>~4e25 FLOPs</td><td>5,120 Blackwell GPUs (10 min on B200/NVFP4 — MLPerf v5.1)</td><td>~$10M with NVFP4 / Blackwell</td></tr>
<tr><td>DeepSeek-R1 RL (reasoning)</td><td>~6e23 FLOPs</td><td>2T tokens of rollout</td><td>~$1M (20% of pretraining)</td></tr>
<tr><td>ProRL V2 1.5B reasoning (700+ steps)</td><td>~few × 10²² FLOPs</td><td>32-128× H100 for 2-4 weeks</td><td>$50-200K</td></tr>
<tr><td>Inference (frontier model)</td><td>per-query 10⁹-10¹¹ FLOPs</td><td>Blackwell + NVFP4 (M41)</td><td>$0.02 / 1M tokens (GPT-OSS-120B benchmark)</td></tr>
<tr><td>Inference at scale (1B requests, 70B model)</td><td>~10²⁰ FLOPs</td><td>varies</td><td>~$5-20M depending on TTC mode</td></tr>
</tbody>
</table>
</div>

<p>Three observations:</p>

<ul>
  <li><strong>The Blackwell+NVFP4 transition (M41) reshaped the math</strong>. Llama 3.1 405B for $10M is a 2026 number; on Hopper FP8 it would be $30M+; on H100 BF16 it would be $80M+. Hardware progress changes which scaling law regime is economically reachable.</li>
  <li><strong>Inference dominates training over deployment lifetime</strong> for any model with substantial traffic. A model trained for $1M and serving $10M of inference makes training cost a 10% line item.</li>
  <li><strong>RL post-training cost is rapidly catching up to pretraining</strong>. ProRL V2 at $50-200K is small; ScaleRL at frontier scale (Oct 2025) is much larger; Epoch AI projects parity with overall frontier by late 2026.</li>
</ul>

<h2>Decision framework: where to spend FLOPs</h2>

<p>Putting it all together, the 2026 framework for "how should I spend my compute budget":</p>

<ol>
  <li><strong>What's the deployment volume?</strong> If high (millions+ requests), bias toward smaller models trained longer (Beyond Chinchilla; Liquid AI / Qwen3 territory). If low, Chinchilla-optimal or close.</li>
  <li><strong>Is reasoning required?</strong> If yes, plan RL post-training (M39, M40) and test-time compute. If no, pretraining alone may suffice.</li>
  <li><strong>What hardware is available?</strong> Blackwell (M41) lets you reach 4× more useful compute per dollar than Hopper. Adjust the entire budget accordingly.</li>
  <li><strong>What's the data ceiling?</strong> If data-constrained (M35), training ratio is bounded; can't go to 80,000:1 if the data doesn't exist. Code mixing and data engineering matter.</li>
  <li><strong>What test-time strategy?</strong> Greedy → larger model is better. Best-of-N or tree search → smaller-model-overtrained is better (the Snell et al. + 2026 unifying paper finding).</li>
  <li><strong>What's the precision plan?</strong> NVFP4 training (M41) effectively multiplies your compute budget by 2-3× over FP8. Build that into the training ratio.</li>
</ol>

<p>The compound recipe for a 2026 production deployment:</p>

<ul>
  <li><strong>Pretraining</strong>: small-to-moderate model size (1B-30B), 1,000-10,000:1 token-to-parameter ratio, NVFP4 training on Blackwell, M35-quality data.</li>
  <li><strong>RL post-training</strong>: REINFORCE++ baseline or two-sided clipped GRPO (M39), agentic environments (M40) for the target task surface, 700+ stable steps if reasoning is the target.</li>
  <li><strong>Test-time compute strategy</strong>: adaptive allocation (Plan-and-Budget style); compute-optimal inference based on M37-grade evaluation of the accuracy-token curve for your task.</li>
  <li><strong>Eval rigor</strong>: M37 throughout. Without it, scaling-law decisions are unfalsifiable.</li>
</ul>

<p>This composition of concepts from M11 → M14 → M27 → M34 → M35 → M37 → M39 → M40 → M41 with M42 sitting on top is the full 2026 frontier-class recipe. The course as a whole is the manual.</p>

<div class="ndq">
<h4>About scaling laws and compute economics</h4>

<p class="q">Why did Kaplan get the exponents wrong, and what does Chinchilla teach us about empirical methodology?</p>
<p class="a">Kaplan et al. (2020) used a smaller range of model sizes and a different fitting methodology — they fit power laws across smaller-scale data and extrapolated. Hoffmann et al. (Chinchilla, 2022) ran a much more thorough empirical study with hundreds of model-size × token-count points, fitting the joint loss surface directly rather than independent power laws. The Chinchilla methodology was more careful in three specific ways: (a) varied parameters and tokens jointly rather than fitting separately, (b) used a larger and more representative range of compute scales, (c) used a more honest evaluation of "compute-optimal" by accounting for compute spent on the experiment itself. <em>The general lesson: empirical laws extracted from limited data ranges can be wrong outside that range</em>. Chinchilla's frontier extends Kaplan's; LFM2.5's 80,000:1 extends Chinchilla's. There's no reason to think the current laws are the final word.</p>

<p class="q">If Llama 3 8B at 1,875:1 already beats Chinchilla, why isn't every model at 80,000:1?</p>
<p class="a">Three obstacles. (1) <strong>Data scarcity</strong>. There isn't infinite high-quality data; LFM2.5-350M's 28T tokens required substantial M35-grade data engineering. Bigger models would need proportionally more data to maintain extreme ratios. (2) <strong>Diminishing returns past ~4 epochs</strong>. With limited unique data, you eventually hit a ceiling. (3) <strong>Different deployment plans</strong>. A frontier flagship model serving complex reasoning (TTC + tools + agents) optimizes differently than an edge-deployment small model. Frontier flagships in 2026 are at ~50-200:1 ratios — past Chinchilla but nowhere near the small-model extremes. The ratio is a deployment plan signal, not a universal "better." <em>The "right" ratio depends on the deployment, not on a fundamental property of training.</em></p>

<p class="q">How should I think about the RL/pretrain compute split for a new training project?</p>
<p class="a">The 2026 production answer: <strong>pretrain enough to give the model a reasonable starting policy, then invest heavily in RL</strong>. Specifically: (1) Pretraining should cover the data-and-architecture territory — enough tokens to be in the overtrained regime, M35-quality data, M41 precision. (2) <strong>RL is where current capability gains are concentrated</strong>: M39's REINFORCE++/GRPO + M40's environments + verifier engineering. (3) The DeepSeek-R1 ratio (RL = 20% of pretraining) was a 2024-2025 baseline; by 2026 it's commonly closer to 50% or matching pretraining for reasoning-focused models. ProRL V2 and ScaleRL show this trajectory. (4) For non-reasoning models (chatbots, classification, generation), RL post-training is much smaller — preference fine-tuning rather than capability building. <em>If you're building a reasoning model, plan for RL to dominate compute; if you're building a chat model, pretraining dominates.</em></p>

<p class="q">What's the actual relationship between Beyond Chinchilla overtraining and test-time scaling?</p>
<p class="a">They're complementary; the 2026 unification paper made this explicit. Beyond Chinchilla says: <em>train smaller models longer when you'll run them many times at inference</em>. TTC says: <em>spend more compute per inference query on smaller models for hard tasks</em>. Both biases push toward the same design choice: <em>smaller models trained for longer</em>. The unified "Test-Time Scaling Makes Overtraining Compute-Optimal" paper (April 2026) showed they couple: a model designed for many-sample TTC deployment benefits from extreme overtraining because the per-sample quality bonus from longer training compounds across the many samples taken at inference. <em>The Era 3 picture is consistent</em>: small + overtrained + heavy TTC is the modern compute-optimal frontier for reasoning workloads.</p>

<p class="q">When does it make sense to scale model size instead of training duration or TTC?</p>
<p class="a">Three specific cases. (1) <strong>Capability frontier requires it</strong>: some tasks (very long context, certain multimodal capabilities, certain reasoning) require model capacity that a small overtrained model can't provide. The frontier flagships exist for these tasks. (2) <strong>Inference cost is not a constraint</strong>: research labs running internal experiments, government/scientific computing, or high-value queries (medical, legal) where per-query cost is acceptable. (3) <strong>TTC infrastructure is unavailable</strong>: deployments that can't support best-of-N or tree search (latency-critical applications, edge deployment without compute headroom for multiple samples). For most production deployments outside these cases, smaller-and-overtrained beats larger-and-Chinchilla. <em>This is why frontier 2026 model families ship at multiple sizes</em>: different deployment scenarios sit at different points on the scaling manifold.</p>

<p class="q">What's the practical limit on overtraining? Will we see 1,000,000:1 ratios?</p>
<p class="a">Empirically uncertain; theoretical limits suggest yes for very small models. The Sardana et al. results validated up to 10,000:1; LFM2.5 hit 80,000:1 at 350M params. For 100M-class models, ratios above 100,000:1 might be tractable with 50-100T tokens of data. <strong>Two factors will determine where this saturates</strong>: (a) <strong>data availability</strong> (M35) — high-quality natural language tops out at ~50-100T tokens by current estimates; synthetic data extends this but with quality questions; (b) <strong>diminishing returns</strong> — at extreme ratios, the marginal token contributes very little. The "Test-Time Scaling Makes Overtraining Compute-Optimal" paper's empirical curves suggest gradual saturation, not a hard cliff. <em>Plausible 2027-2028 picture: small models at 200,000-1,000,000:1 ratios for edge deployment</em>; but high-quality data engineering becomes the bottleneck before the scaling laws themselves do.</p>
</div>

<h2>Code Magnets: implement Chinchilla-aware compute budget split</h2>

<p>You're writing a function that takes a compute budget C and returns the Chinchilla-optimal model size N and tokens D. Three magnets are wrong choices.</p>

<div class="magnets">
<p>Arrange the magnets to compute the compute-optimal allocation.</p>

<div class="magnet-pool">
  <span class="magnet">def chinchilla_optimal(compute_C, ratio=20):</span>
  <span class="magnet">    # Chinchilla: C ≈ 6 * N * D, optimal at D ≈ ratio * N</span>
  <span class="magnet">    # Solve: 6 * N * (ratio * N) = C → N = sqrt(C / (6 * ratio))</span>
  <span class="magnet">    N = (compute_C / (6 * ratio)) ** 0.5</span>
  <span class="magnet">    N = compute_C / (6 * ratio)</span>
  <span class="magnet">    D = ratio * N</span>
  <span class="magnet">    D = compute_C / N</span>
  <span class="magnet">    return {"params_N": N, "tokens_D": D, "compute_C": compute_C}</span>

  <span class="magnet">def beyond_chinchilla(compute_C, inference_requests):</span>
  <span class="magnet">    # Sardana et al.: optimal ratio shifts up with inference demand</span>
  <span class="magnet">    # Heuristic: tokens_per_param ∝ log10(inference_requests) * scale_factor</span>
  <span class="magnet">    ratio = 20 * max(1, (inference_requests / 1e9) ** 0.4)</span>
  <span class="magnet">    ratio = 20</span>
  <span class="magnet">    return chinchilla_optimal(compute_C, ratio=ratio)</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">def</span> <span class="fn">chinchilla_optimal</span>(compute_C, ratio=<span class="num">20</span>):
    <span class="com"># Chinchilla: C ≈ 6 * N * D, optimal at D ≈ ratio * N</span>
    <span class="com"># Solve: 6 * N * (ratio * N) = C → N = sqrt(C / (6 * ratio))</span>
    N = (compute_C / (<span class="num">6</span> * ratio)) ** <span class="num">0.5</span>
    D = ratio * N
    <span class="kw">return</span> {<span class="str">"params_N"</span>: N, <span class="str">"tokens_D"</span>: D, <span class="str">"compute_C"</span>: compute_C}

<span class="kw">def</span> <span class="fn">beyond_chinchilla</span>(compute_C, inference_requests):
    <span class="com"># Sardana et al.: optimal ratio shifts up with inference demand</span>
    ratio = <span class="num">20</span> * <span class="fn">max</span>(<span class="num">1</span>, (inference_requests / <span class="num">1e9</span>) ** <span class="num">0.4</span>)
    <span class="kw">return</span> <span class="fn">chinchilla_optimal</span>(compute_C, ratio=ratio)</code></pre>
<p>The traps:</p>
<ul>
  <li><code>N = compute_C / (6 * ratio)</code>: linear instead of square-root. Chinchilla's compute relation is C ≈ 6 × N × D, with D = ratio × N. Substituting: C ≈ 6 × N × (ratio × N) = 6 × ratio × N². Solving for N gives N = sqrt(C / (6 × ratio)) — square root, not linear. The wrong magnet would scale parameters proportionally to compute, the Kaplan-era recommendation that Chinchilla overturned. Square root is what makes parameters and tokens grow in balance: doubling compute multiplies both N and D by sqrt(2), not one by 2× and the other by 1×.</li>
  <li><code>D = compute_C / N</code>: this gives you tokens implied by C/N (a different formula entirely — it's the cost per parameter). The correct relationship from Chinchilla is D = ratio × N — tokens scale linearly with parameters at the Chinchilla-optimal ratio. The wrong magnet ignores the ratio entirely; it would always give a 1:1 tokens-per-parameter relationship after dividing through, which is far from any meaningful scaling regime.</li>
  <li><code>ratio = 20</code> in <code>beyond_chinchilla</code>: hard-codes the Chinchilla ratio regardless of inference demand. The whole point of <code>beyond_chinchilla</code> is to adjust the ratio based on expected inference volume — high inference → higher ratio (smaller model trained longer). The wrong magnet defeats the function's purpose; calling <code>beyond_chinchilla(C, 1e10)</code> would return the same allocation as <code>beyond_chinchilla(C, 1)</code>, missing the entire Sardana-et-al insight.</li>
</ul>
<p>The pattern: <strong>Chinchilla's compute identity is C = 6×N×D, optimal D = ratio×N → solve for N as sqrt(C / (6×ratio)) → tokens follow</strong>. Beyond Chinchilla wraps this with an inference-aware ratio that increases with deployment volume. The most common bug — using linear scaling for the compute-to-parameters relationship — is exactly what Kaplan got wrong; Chinchilla's contribution was getting the exponent right, and the square root is the encoding of that.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each scaling-law concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>Kaplan (2020)</div>
  <div>A. Original scaling laws; recommended scaling parameters faster than data; superseded by Chinchilla.</div>

  <div>Chinchilla (2022)</div>
  <div>B. 20:1 tokens-to-parameters at compute-optimal training; balanced parameter/data scaling.</div>

  <div>Beyond Chinchilla (2024-2026)</div>
  <div>C. Train smaller and longer when inference demand is high; validated up to 10,000:1 ratios.</div>

  <div>Test-time compute (TTC)</div>
  <div>D. Spend extra compute per query (best-of-N, tree search); smaller models can beat larger on Pareto.</div>

  <div>RL log-linear scaling</div>
  <div>E. Validation accuracy increases linearly with log of average reasoning tokens during RL training.</div>

  <div>TTS trilemma</div>
  <div>F. Test-time scaling can optimize at most two of: accuracy, consistency, efficiency.</div>

  <div>Overthinking</div>
  <div>G. Longer chain-of-thought sometimes degrades accuracy; more TTC isn't monotonically better.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>Kaplan</strong> → A<br>
<strong>Chinchilla</strong> → B<br>
<strong>Beyond Chinchilla</strong> → C<br>
<strong>Test-time compute</strong> → D<br>
<strong>RL log-linear scaling</strong> → E<br>
<strong>TTS trilemma</strong> → F<br>
<strong>Overthinking</strong> → G
</p>
<p>The mental shortcut: <em>Kaplan was wrong on exponents, Chinchilla balanced them, Beyond Chinchilla overshot for inference, TTC adds an axis, RL log-scales, the trilemma constrains, overthinking limits</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team has $5M of compute budget for a new model. They expect ~100M inference requests over the model's lifetime. Walk through how to think about the size/training-tokens decision in 2026.</p>
<details class="answer"><summary>show answer</summary>
<p>Step-by-step framework, applying Beyond Chinchilla:</p>
<p>(1) <strong>Express the budget in FLOPs</strong>. $5M at April 2026 cloud prices ($3-4/hour B200) and Blackwell+NVFP4 efficiency: roughly 5e23 to 1e24 FLOPs of training compute, depending on hardware utilization. Use 8e23 FLOPs as a planning estimate.</p>
<p>(2) <strong>Apply Chinchilla as a baseline</strong>. C = 6 × N × D, D = 20×N. So N ≈ sqrt(8e23 / 120) ≈ 8.2e10 parameters ≈ 80B model on 1.6T tokens. That's the compute-optimal training answer if inference volume were tiny.</p>
<p>(3) <strong>Adjust for inference volume</strong>. 100M requests is moderate — not the 1B+ that Sardana et al. flagged for extreme ratio shifts. Apply something like ratio = 20 × (1e8/1e9)^0.4 ≈ 20 × 0.4 ≈ ~8. But that gives lower ratio, not higher. Wait — Sardana's law goes the other way: <em>more</em> inference → <em>higher</em> ratio. Reread: at 1B requests, train smaller and longer than 20:1. At 100M, somewhere between Chinchilla and the Sardana extremes. Use ratio ≈ 100-500 as a planning value.</p>
<p>(4) <strong>Recompute with adjusted ratio</strong>. With ratio = 200: N ≈ sqrt(8e23 / (6 × 200)) ≈ 2.6e10 = 26B model on 5.2T tokens. With ratio = 500: N ≈ 1.6e10 = 16B model on 8T tokens.</p>
<p>(5) <strong>Sanity-check against published models</strong>. Llama 3 8B at 15T tokens (1875:1 ratio) was chosen for very-high-volume deployment. Our 100M-request scenario is much lower volume, so somewhere between Chinchilla's 20:1 and Llama 3's 1875:1 makes sense. <strong>16B-30B trained on 5-10T tokens is the right ballpark</strong>.</p>
<p>(6) <strong>Adjust for other factors</strong>. NVFP4 (M41) effectively multiplies the compute budget by 2-3× over FP8 — so the team can either train a larger model or extend training. Data availability (M35) caps how many tokens are practical. Reasoning-task focus would shift budget toward RL (M39) post-training and reduce pretraining size.</p>
<p>(7) <strong>Reserve budget for RL and eval</strong>. Production-grade 2026 deployment shouldn't spend 100% on pretraining — keep 20-30% for RL post-training and ongoing eval (M37). Final pretraining budget: $3.5M at the size/data point above.</p>
<p>The output of this analysis is <em>a deployment plan</em>, not just a model size. The 2026 framework treats compute allocation as portfolio optimization across pretraining/RL/inference/eval.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Why does test-time scaling (best-of-N, tree search) make small overtrained models more attractive than large compute-optimal models?</p>
<details class="answer"><summary>show answer</summary>
<p>Two compounding effects from the unifying 2026 paper:</p>
<p>(1) <strong>Per-sample quality matters at TTC</strong>. Best-of-N with N=32 means generating 32 samples per query and choosing the best (or majority vote, or via verifier). Pass@k probability for a model with per-sample correctness p is 1 - (1-p)^k — strongly nonlinear in p. A small bump in per-sample quality compounds across many samples. <strong>Overtrained small models have higher per-sample quality than Chinchilla-balanced larger models at the same per-sample compute cost</strong>, because the extra training has refined the model's outputs without inflating per-token cost.</p>
<p>(2) <strong>Per-sample compute matters for TTC budget</strong>. If your inference budget is fixed (say, 32× the cost of one greedy decode), you can spend it on (a) one big-model greedy decode, (b) 32 small-model samples + voting, (c) intermediate splits. Smaller models mean more samples per fixed budget. The TTC literature (Snell et al. 2024 onward) consistently shows option (b) winning on hard tasks: 32 small-model samples + voting often beats one big-model decode.</p>
<p>The combined effect: <strong>smaller-and-overtrained-and-many-samples is Pareto-superior to larger-and-fewer-samples for TTC-friendly tasks</strong>. The 2026 unifying paper made this rigorous: pretraining decisions and inference strategy are coupled; the compute-optimal pretraining ratio depends on the planned TTC mode. If you'll run the model in best-of-N mode at deployment, train it smaller and longer than Chinchilla suggests; if you'll run it greedy, Chinchilla is closer to right.</p>
<p>The practical implication: <em>the question "what size model should I train?" has no answer without specifying the inference strategy</em>. M34 (reasoning), M37 (eval to determine TTC accuracy curves), and M42 (this scaling question) all compose into the deployment design.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> Walk through why "more reasoning tokens" can hurt accuracy (overthinking), even though the RL scaling laws say accuracy is log-linear in reasoning tokens.</p>
<details class="answer"><summary>show answer</summary>
<p>The two findings are about different things and aren't actually in conflict:</p>
<p>(1) <strong>RL scaling law (training-time)</strong>: <em>during RL training</em>, average reasoning-token length grows as the model improves, and validation accuracy grows linearly with log of length. This is a <em>cross-checkpoint</em> observation: as RL training progresses, both reasoning length and accuracy climb together.</p>
<p>(2) <strong>Overthinking (inference-time)</strong>: at a <em>fixed checkpoint</em>, prompting the model to generate longer reasoning often degrades accuracy. The relationship between reasoning length and accuracy at a single model checkpoint is non-monotonic.</p>
<p>The two are about different axes: training trajectory vs single-checkpoint inference behavior. The RL scaling law is "models that train longer have longer reasoning AND better accuracy." Overthinking is "for a given model, forcing it to reason longer doesn't always help."</p>
<p>Why does overthinking happen?</p>
<ul>
  <li><strong>Distractors accumulate</strong>: longer chains have more tokens that can mention plausible-but-wrong facts, leading the reasoning astray.</li>
  <li><strong>Confidence drift</strong>: late tokens in long chains are conditioned on potentially incorrect early tokens; the conditional distribution loses calibration.</li>
  <li><strong>Off-distribution territory</strong>: training distributions have characteristic length distributions; forcing significantly longer reasoning pushes the model into regions it wasn't trained for.</li>
</ul>
<p>The mitigation (Plan-and-Budget, 2025-2026): <strong>adaptive token budgets per query</strong>. Use a difficulty classifier to decide how many tokens to allow; easy queries get short budgets (preventing overthinking); hard queries get long budgets (allowing genuine deep reasoning). The reported result is up to 70% accuracy gains with 39% token reduction over flat budgets.</p>
<p>The reconciliation: <em>RL training improves the average accuracy at all length budgets, but the optimal length budget per query is task-dependent and bounded</em>. A model that's been trained to reason for longer can use those longer chains effectively when needed; forcing it to do so when not needed is wasteful and sometimes harmful.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> Sketch the compute-economic argument for whether a startup should train its own 7B reasoning model or use API access to a frontier model. April 2026 numbers.</p>
<details class="answer"><summary>show answer</summary>
<p>The framework: compare lifetime training+inference cost of self-trained model vs API access for the expected query volume.</p>
<p><strong>Self-trained 7B reasoning model (April 2026 numbers)</strong>:</p>
<ul>
  <li>Pretraining: 7B at 1875:1 = 13T tokens. ~$300-800K on Blackwell + NVFP4 (M41).</li>
  <li>RL post-training: REINFORCE++ on a custom domain (M39, M40). $50-200K depending on environment complexity and target capability.</li>
  <li>Engineering: 2-4 engineers × 3-6 months for setup, eval, deployment. ~$300-600K loaded cost.</li>
  <li>Inference at deployment: a 7B model on Blackwell + NVFP4 costs roughly $0.05-0.2 per 1M tokens at scale (4-10× cheaper than M41's $0.02 reference for GPT-OSS-120B because 7B is smaller).</li>
  <li><strong>Total upfront cost: $700K - $1.6M; ongoing inference: ~$0.1/Mtoken</strong>.</li>
</ul>
<p><strong>Frontier API access (April 2026)</strong>:</p>
<ul>
  <li>Claude Opus 4.6 / GPT-5.4 / Gemini 3.1 Pro at $3-15 per million output tokens (varies by provider and tier).</li>
  <li>No upfront engineering cost beyond integration (~weeks of one engineer).</li>
  <li>Quality typically higher than 7B for general tasks; comparable on narrow specialized tasks if the 7B has been RL'd effectively.</li>
</ul>
<p><strong>The crossover analysis</strong>:</p>
<p>(a) <strong>Volume-dependent</strong>. At 100M tokens of inference per month, frontier API costs $300K-$1.5M/month; self-trained costs ~$10K/month. Crossover happens when API costs exceed the self-train upfront in some payback period. At 100M tokens/month, API is more expensive after 1-2 months.</p>
<p>(b) <strong>Domain-specificity</strong>. If the task is narrow (medical billing, customer support for specific product, code completion in specific framework), a 7B + RL on M40-grade environment can match or exceed frontier on the specific task. General reasoning, broad knowledge, complex math — frontier wins.</p>
<p>(c) <strong>Privacy / sovereignty</strong>. Self-hosted models are required for regulated data (healthcare, finance, government). API isn't an option regardless of economics.</p>
<p>(d) <strong>Speed of iteration</strong>. APIs let you ship fast; self-trained models lock you into your training run for months. For early-stage startups, API access is usually the right starting point even when economics favor self-training long-term.</p>
<p>(e) <strong>Vendor risk</strong>. API providers can change pricing, discontinue models, or restrict use cases. Self-hosted models lock in your stack.</p>
<p><strong>The 2026 advice</strong>: most startups should start with frontier API; switch to self-trained when (a) volume exceeds ~50M tokens/month sustained, (b) task is narrow enough that 7B can match frontier on it, AND (c) privacy/cost economics justify the upfront investment. The Cursor pattern (M40 territory) is exactly this: started on frontier APIs, now post-training their own models because volume justifies it. <em>The decision is portfolio-level, not all-or-nothing</em>: many production systems use frontier APIs for some queries and self-trained models for others.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>Scaling laws went through three eras. <strong>Era 1 (Kaplan 2020)</strong>: scale parameters faster than data. <strong>Era 2 (Chinchilla 2022)</strong>: 20 tokens per parameter at compute-optimal training. <strong>Era 3 (Beyond Chinchilla, 2024-2026)</strong>: train smaller and longer when serving high inference volume.</li>
  <li>The Chinchilla compute identity: <strong>C ≈ 6 × N × D, optimal D = 20 × N → N = sqrt(C / 120)</strong>. Square root, not linear.</li>
  <li><strong>The Chinchilla Trap</strong>: training compute-optimal at deployment time means using a model larger than you should for a given quality target. Llama 1 paper flagged this; Sardana et al. (2024-2025) formalized it.</li>
  <li>The overtraining trajectory: GPT-3 1.7:1 → Llama 1 7B at 142:1 → Llama 2 at 284:1 → Llama 3 8B at 1,875:1 → Qwen3-0.6B at 60,000:1 → <strong>Liquid AI LFM2.5-350M at 80,000:1</strong> (April 2026 record).</li>
  <li><strong>Compute splits three ways in 2026</strong>: pretraining (base capability), RL post-training (M39, M40 — reasoning + agentic), test-time compute (M34 — best-of-N, tree search, adaptive allocation).</li>
  <li><strong>RL training scaling laws are log-linear in reasoning tokens</strong>: validation accuracy increases linearly with log of average generated tokens across RL checkpoints (Seed-OSS-36B paper, April 2026; Art of Scaling RL Compute, October 2025).</li>
  <li><strong>RL post-training compute is rapidly catching pretraining</strong>: DeepSeek-R1 at 20% of pretraining ($1M); ProRL V2 (Feb 2026) ran 700+ stable steps; <strong>Epoch AI projects parity by late 2026</strong>.</li>
  <li><strong>Inference scaling laws (Snell et al. 2024 onward)</strong>: smaller models combined with advanced inference algorithms (best-of-N, weighted voting, tree search) can be Pareto-optimal vs scaling parameters. Llemma-7B + tree search beat Llemma-34B on MATH at the same compute.</li>
  <li><strong>TTS trilemma</strong>: test-time scaling optimizes at most two of accuracy, consistency, efficiency. Production deployment chooses two.</li>
  <li><strong>Overthinking</strong>: longer CoT often degrades accuracy at a fixed model checkpoint. Plan-and-Budget (2025-2026) achieves 70% accuracy gain + 39% token reduction via adaptive budgeting.</li>
  <li><strong>Test-Time Scaling Makes Overtraining Compute-Optimal (April 2026)</strong>: unifying paper showing pretraining and inference scaling decisions COUPLE. A model designed for many-sample TTC inference benefits from extreme overtraining because per-sample quality bonuses compound across samples.</li>
  <li><strong>Scaling Laws for Precision (Kumar et al. 2024)</strong>: numerical precision interacts with scaling. Training in lower precision may be compute-optimal; post-training quantization degradation scales with training tokens. M14 + M41 + M42 compose.</li>
  <li><strong>Data-constrained scaling (Muennighoff et al. 2023)</strong>: ~4 epochs over fresh data ≈ same-volume fresh data; beyond that diminishing returns. Code mixing as 2× useful data multiplier. Caps the overtraining trajectory.</li>
  <li>The 2026 economic ranges: <strong>Llama 3.1 405B for ~$10M with NVFP4 + Blackwell</strong> (10 minutes on 5,120 GPUs, MLPerf v5.1); inference at <strong>$0.02 per million tokens</strong> for GPT-OSS-120B; <strong>RL post-training at $50K-$1M</strong> depending on scale.</li>
  <li>The reflex for 2026 training projects: ask "<strong>volume? reasoning needed? hardware? data ceiling? TTC strategy? precision plan?</strong>" — the six questions determine the compute split. The answer isn't "size of model" alone; it's a position on the (pretrain, RL, test-time) compute manifold.</li>
</ul>
</div>

<p>Module 43 (next, if Part XI continues to its planned end) would tackle <strong>embodied AI &amp; robotics RL</strong> — VLA (vision-language-action) models, RLinf real-world stack, sim-to-real transfer, the bridge between LLM RL and robotics. Closes Part XI; addresses the final gap from the analysis.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">42</span>
  <span>Scaling laws &amp; compute economics</span>
</div>
"""

emit("42_scaling_laws", "Module 42 — Scaling laws & compute economics", BODY)
