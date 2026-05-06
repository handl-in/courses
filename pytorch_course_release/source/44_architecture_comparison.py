#!/usr/bin/env python3
"""Module 44: Comparing latest LLM & VLM architectures — full HF vibe, April 2026 current."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Bonus · Module 44 · April 2026 currency</div>
  <h1 class="module-title"><em>Comparing latest LLM &amp; VLM architectures:</em> the April 2026 zoo, taxonomized</h1>
  <p class="module-sub">— from DeepSeek V4's hybrid attention to Llama 4's iRoPE, why every flagship is now MoE, the three-era VLM taxonomy, and the architectural primitives that distinguish 2026 models from each other</p>
</div>

<p>You've spent 43 modules learning how to build models. This bonus module steps back to compare what's actually deployed in April 2026. Same pedagogy, different scope: instead of explaining one architecture, we taxonomize the family.</p>

<p>The 2026 model landscape has more entrants than any prior year. Closed-source: GPT-5.4 / 5.5, Claude Opus 4.6 / 4.7 / Sonnet 4.6, Gemini 3 / 3.1 Pro, Grok 4. Open-weight: Llama 4 Scout / Maverick, DeepSeek V4-Pro / V4-Flash, GLM-5 / 5.1, Qwen 3.5 / 3.6, Kimi K2.5 / K2.6, MiniMax M2.5, MiMo V2.5, Gemma 3, Nemotron Ultra. <em>Every frontier model is now MoE</em>. <em>Every flagship has 1M+ context window</em>. <em>The 25× price gap</em> between cheapest and most expensive frontier model is the defining 2026 economics.</p>

<p>This module taxonomizes them. The architectural primitives that matter: attention variants (full, GQA, sparse, hybrid), position encoding (RoPE, iRoPE, YaRN), MoE patterns (pure, alternating, shared experts), reasoning modes (separate vs unified switchable), and the VLM design eras. By the end you'll know <em>what makes DeepSeek V4 different from Llama 4 different from GLM-5 different from Qwen 3.6</em>, why all the closed-source flagships keep landing within 2.5 points of each other on benchmarks, and which architectural choices are converging vs diverging.</p>

<div class="keyidea">
The April 2026 LLM zoo has converged on three macro-patterns. <strong>(1) Mixture-of-Experts is universal</strong> at frontier scale — DeepSeek V4-Pro (1.6T/49B active), Llama 4 Maverick (400B/17B), GLM-5 (744B/40B), Qwen 3.6 (35B/3B), Kimi K2.6, MiniMax M2.5, MiMo V2.5 (1.02T/42B). Every flagship is sparse. <strong>(2) 1M+ token context is table stakes</strong> — Llama 4 Scout extends to 10M via iRoPE; DeepSeek V4 hits 1M via Hybrid Attention (CSA + HCA, 27% FLOPs / 10% KV vs V3.2); GPT-5.5 has 256K, Claude Opus 4.7 has 200K with 1M beta, Gemini 3.1 Pro has 1M. <strong>(3) Reasoning consolidates into the base model</strong> — DeepSeek V4 folds the R reasoning line into a single model with switchable Thinking / Non-Thinking modes; closed-source flagships do the same.

The architectural primitives that differentiate: <strong>Attention variants</strong> — full vs GQA vs sparse-content-based (DSA) vs hybrid CSA+HCA (DeepSeek V4) vs sliding-window. <strong>Position encoding</strong> — RoPE vs iRoPE (Llama 4: 3 RoPE layers + 1 NoPE layer interleaved, enables 10M context) vs YaRN scaling. <strong>MoE patterns</strong> — pure MoE (Scout, GLM-5, DeepSeek) vs MoE/dense alternating (Maverick) vs shared experts (DeepSeek lineage) vs different routing (top-k, expert choice). <strong>Optimizer</strong> — AdamW remains dominant; <strong>Muon</strong> (DeepSeek V4) is the first frontier replacement. <strong>Precision</strong> — NVFP4 training (M41) maturing; FP8 still dominant for closed-source.

For VLMs, the survey-level taxonomy (Vision-Language-Models-Overview, Feb 2026): <strong>three eras</strong>. <strong>Era 1</strong> — frozen vision encoder + frozen LLM + learnable connector (CLIP, BLIP, Flamingo, 2021-2023). <strong>Era 2</strong> — LLM trunk with vision as bolt-on adapter (LLaVA, Qwen2.5-VL, GPT-4V, 2023-2025). <strong>Era 3</strong> — single transformer trained from scratch on mixed-modality data; forks into <strong>3a (early fusion, text-out)</strong> used by Qwen3.5/3.6, Gemma 4, Gemini 3, GPT-5.4, Claude Opus 4.6, and <strong>3b (mixed-modality output)</strong> for models generating images/audio. <strong>Vision encoders</strong>: SigLIP 2 dominant (Qwen3-VL, Gemma 3); InternViT-6B for InternVL line; DINOv2+SigLIP fused (M43's OpenVLA). <strong>The benchmarks have saturated</strong> — MMMU-Pro spread between top four flagships is 2.4 points by April 2026 (within run-to-run noise). Differentiation moved to video (Gemini 3 leads), audio (Gemini 3), long-document OCR (Claude Opus 4.7), chart reasoning (GPT-5.5), code-with-vision (GPT-5.5).
</div>

<h2>Two new faces — taking inventory</h2>

<div class="character" style="--c: #d4a017;">
  <div class="avatar" style="background: #d4a017; color: #fff;">T</div>
  <div>
    <p class="who">Taxonomist</p>
    <p class="name">"I categorize the zoo. Same family tree, divergent traits — what distinguishes one species from another?"</p>
    <p class="says">Every frontier 2026 model is a transformer with attention, position encoding, normalization, MoE, training recipe. <em>The differences between them are which variants of each primitive they use.</em> Llama 4 Scout differs from DeepSeek V4-Pro because of: pure-MoE 16-expert vs alternating MoE+dense 256-expert; iRoPE positional with NoPE layers vs standard RoPE with hybrid-attention compression; 10M context via length generalization vs 1M context via FLOP reduction. Both work; both are excellent; the choices reflect different deployment priorities. <strong>Reading 2026 model cards is now an exercise in primitive identification</strong>: spot the attention variant, the MoE pattern, the position encoding, the optimizer, the precision. The base transformer is invariant; the variants tell the story.</p>
  </div>
</div>

<div class="character" style="--c: #c1502e;">
  <div class="avatar" style="background: #c1502e; color: #fff;">C</div>
  <div>
    <p class="who">Convergence</p>
    <p class="name">"Frontier models converge faster than press releases suggest. The architectural moves are public, then everyone copies."</p>
    <p class="says">DeepSeek R1 (Jan 2025) showed RL with verifiable rewards works → every lab adopted GRPO-family methods within 6 months. Llama 4 (April 2025) showed MoE at frontier scale → DeepSeek, GLM, Qwen, Kimi, MiniMax all shipped MoE flagships by end of 2025. NVIDIA's NVFP4 paper (Sep 2025) → MLPerf v5.1 sweep (Nov 2025) → DeepSeek V4 trained with FP4 quantization-aware (April 2026). <strong>I'm the gravity that pulls architectures toward each other</strong>. Public papers + open weights + ~6 months = convergence. The gap between "leading-edge novel architecture" and "everyone has it" is one model generation. <em>The 2.4-point spread between top flagships on MMMU-Pro is structural, not coincidental</em> — they're approximately the same architecture trained on approximately the same data.</p>
  </div>
</div>

<h2>The frontier zoo, April 2026</h2>

<p>Inventory first. The current generation of LLMs in production:</p>

<div class="table-wrap">
<table>
<caption>Frontier LLMs in production, April 2026</caption>
<thead><tr><th>Model</th><th>Source</th><th>Architecture</th><th>Context</th><th>Released</th></tr></thead>
<tbody>
<tr><td><strong>GPT-5.5</strong> (Codex)</td><td>OpenAI</td><td>Closed; reasoning unified</td><td>256K (1M in Codex)</td><td>March 2026</td></tr>
<tr><td><strong>Claude Opus 4.7</strong></td><td>Anthropic</td><td>Closed; Sonnet 4.6 sibling</td><td>200K (1M beta)</td><td>April 2026</td></tr>
<tr><td><strong>Gemini 3.1 Pro</strong></td><td>Google DeepMind</td><td>Closed; multimodal-native</td><td>1M</td><td>February 2026</td></tr>
<tr><td><strong>Grok 4</strong></td><td>xAI</td><td>Closed; multi-agent variant</td><td>~256K</td><td>Late 2025</td></tr>
<tr><td><strong>DeepSeek V4-Pro</strong></td><td>DeepSeek</td><td>Open MoE: 1.6T / 49B active</td><td>1M</td><td>April 24, 2026</td></tr>
<tr><td><strong>DeepSeek V4-Flash</strong></td><td>DeepSeek</td><td>Open MoE: 284B / 13B active</td><td>1M</td><td>April 24, 2026</td></tr>
<tr><td><strong>Llama 4 Scout</strong></td><td>Meta</td><td>Open MoE: 109B / 17B active, 16 experts</td><td>10M</td><td>April 2025</td></tr>
<tr><td><strong>Llama 4 Maverick</strong></td><td>Meta</td><td>Open MoE: 400B / 17B active, 128 experts (alternating)</td><td>1M</td><td>April 2025</td></tr>
<tr><td><strong>Llama 4 Behemoth</strong> (preview)</td><td>Meta</td><td>Open MoE teacher; ~2T total</td><td>—</td><td>Preview 2025</td></tr>
<tr><td><strong>GLM-5.1</strong></td><td>Z.ai (Zhipu)</td><td>Open MoE: 744B / 40B active, MIT licensed, DSA</td><td>~200K</td><td>Q1 2026</td></tr>
<tr><td><strong>Qwen 3.6</strong> (35B-A3B)</td><td>Alibaba</td><td>Open MoE: 35B / 3B active, Apache 2.0</td><td>~128K</td><td>Q1 2026</td></tr>
<tr><td><strong>Kimi K2.6</strong></td><td>Moonshot AI</td><td>Open MoE; agentic-focused</td><td>~128K</td><td>Q1 2026</td></tr>
<tr><td><strong>MiniMax M2.5</strong></td><td>MiniMax</td><td>Open-weight MoE</td><td>~256K</td><td>Feb 2026</td></tr>
<tr><td><strong>MiMo V2.5-Pro</strong></td><td>Xiaomi</td><td>Open MoE: 1.02T / 42B active, FP8 native</td><td>32K native</td><td>Q1 2026</td></tr>
<tr><td><strong>Nemotron Ultra 253B</strong></td><td>NVIDIA</td><td>Open dense reasoning model</td><td>~128K</td><td>2025</td></tr>
<tr><td><strong>Gemma 3 (27B)</strong></td><td>Google</td><td>Open dense; multilingual</td><td>~128K</td><td>2025</td></tr>
</tbody>
</table>
</div>

<p>Three observations from the inventory alone, before any architectural detail:</p>

<ul>
  <li><strong>Every frontier model is MoE except small "edge" models (Gemma 3, Nemotron 253B dense)</strong>. The dense-vs-MoE debate from 2023-2024 is over; MoE won at scale.</li>
  <li><strong>1M+ context is table stakes for new releases</strong>. Models without it (Qwen 3.6 at 128K, Kimi K2.6 at 128K) are explicitly trading context for cost or speed; new flagship releases extend further.</li>
  <li><strong>Open-weight has caught closed-source on coding benchmarks</strong>. GLM-5.1 leads SWE-bench Pro at 58.4%; DeepSeek V4-Pro matches Claude Opus 4.6 at 80.6% on SWE-bench Verified. The "two years behind" gap of 2024 is gone for coding; partial gaps remain for multimodal maturity and safety.</li>
</ul>

<h2>The architectural primitives that distinguish 2026 models</h2>

<p>Every transformer LLM in 2026 has the same skeleton from M11/M14: token embedding → stacked transformer blocks (attention + FFN with normalization) → output projection. The blocks have specific variant choices:</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 460" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Five primitives where 2026 LLMs differ — and what each lab chose</text>

  <!-- Attention variants -->
  <g transform="translate(20, 50)">
    <rect x="0" y="0" width="340" height="125" fill="#fff8a8" stroke="#1a1612" stroke-width="1.5"/>
    <text x="170" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">① Attention mechanism</text>
    <text x="15" y="42" font-size="9" font-weight="700" fill="#1a1612">Full attention:</text>
    <text x="120" y="42" font-size="9" fill="#1a1612">vanilla; quadratic in seq length</text>
    <text x="15" y="56" font-size="9" font-weight="700" fill="#1a1612">GQA / MQA:</text>
    <text x="120" y="56" font-size="9" fill="#1a1612">share K/V across query heads — Llama, GPT</text>
    <text x="15" y="70" font-size="9" font-weight="700" fill="#1a1612">Sparse (DSA):</text>
    <text x="120" y="70" font-size="9" fill="#1a1612">content-based block selection — GLM-5</text>
    <text x="15" y="84" font-size="9" font-weight="700" fill="#1a1612">CSA + HCA hybrid:</text>
    <text x="120" y="84" font-size="9" fill="#1a1612">DeepSeek V4 — 4× compress + 128× compress</text>
    <text x="15" y="98" font-size="9" font-weight="700" fill="#1a1612">Sliding window:</text>
    <text x="120" y="98" font-size="9" fill="#1a1612">local attention only — Mistral lineage</text>
    <text x="15" y="116" font-size="9" font-style="italic" fill="#c1502e">2026 trend: hybrid sparse for &gt;1M context</text>
  </g>

  <!-- Position encoding -->
  <g transform="translate(380, 50)">
    <rect x="0" y="0" width="340" height="125" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="170" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">② Position encoding</text>
    <text x="15" y="42" font-size="9" font-weight="700" fill="#1a1612">RoPE:</text>
    <text x="100" y="42" font-size="9" fill="#1a1612">rotary embeddings — most flagships' default</text>
    <text x="15" y="56" font-size="9" font-weight="700" fill="#1a1612">YaRN:</text>
    <text x="100" y="56" font-size="9" fill="#1a1612">RoPE scaling for long-context fine-tuning</text>
    <text x="15" y="70" font-size="9" font-weight="700" fill="#1a1612">iRoPE:</text>
    <text x="100" y="70" font-size="9" fill="#1a1612">RoPE + NoPE interleaved (Llama 4: 3:1 ratio)</text>
    <text x="15" y="84" font-size="9" font-weight="700" fill="#1a1612">NoPE:</text>
    <text x="100" y="84" font-size="9" fill="#1a1612">no positional encoding; needs causal mask</text>
    <text x="15" y="98" font-size="9" font-weight="700" fill="#1a1612">+ T-scaling:</text>
    <text x="100" y="98" font-size="9" fill="#1a1612">inference-time temperature for length gen.</text>
    <text x="15" y="116" font-size="9" font-style="italic" fill="#c1502e">2026 trend: hybrid (RoPE + NoPE) for ultra-long</text>
  </g>

  <!-- MoE patterns -->
  <g transform="translate(20, 185)">
    <rect x="0" y="0" width="340" height="125" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="170" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">③ MoE pattern</text>
    <text x="15" y="42" font-size="9" font-weight="700" fill="#1a1612">Pure MoE:</text>
    <text x="120" y="42" font-size="9" fill="#1a1612">every FFN replaced with experts (Scout, GLM-5)</text>
    <text x="15" y="56" font-size="9" font-weight="700" fill="#1a1612">Alternating:</text>
    <text x="120" y="56" font-size="9" fill="#1a1612">MoE + dense alternate (Maverick: 50/50)</text>
    <text x="15" y="70" font-size="9" font-weight="700" fill="#1a1612">Shared experts:</text>
    <text x="120" y="70" font-size="9" fill="#1a1612">always-active expert + routed (DeepSeek lineage)</text>
    <text x="15" y="84" font-size="9" font-weight="700" fill="#1a1612">Top-k routing:</text>
    <text x="120" y="84" font-size="9" fill="#1a1612">k=1 to k=8 typical; tradeoff load vs sparsity</text>
    <text x="15" y="98" font-size="9" font-weight="700" fill="#1a1612">Total/active:</text>
    <text x="120" y="98" font-size="9" fill="#1a1612">35:3 (Qwen) to 32:1 (DeepSeek) ratios</text>
    <text x="15" y="116" font-size="9" font-style="italic" fill="#c1502e">2026 trend: 30-40× sparsity ratio dominant</text>
  </g>

  <!-- Reasoning -->
  <g transform="translate(380, 185)">
    <rect x="0" y="0" width="340" height="125" fill="#fff5d8" stroke="#d4a017" stroke-width="1.5"/>
    <text x="170" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">④ Reasoning architecture</text>
    <text x="15" y="42" font-size="9" font-weight="700" fill="#1a1612">Separate reasoning model:</text>
    <text x="170" y="42" font-size="9" fill="#1a1612">DeepSeek R1, o1 (older)</text>
    <text x="15" y="56" font-size="9" font-weight="700" fill="#1a1612">Unified switchable:</text>
    <text x="170" y="56" font-size="9" fill="#1a1612">DeepSeek V4 (3 modes), Claude 4.7 ext.</text>
    <text x="15" y="70" font-size="9" font-weight="700" fill="#1a1612">Always-on thinking:</text>
    <text x="170" y="70" font-size="9" fill="#1a1612">Gemini 3 Deep Think variant</text>
    <text x="15" y="84" font-size="9" font-weight="700" fill="#1a1612">Multi-agent:</text>
    <text x="170" y="84" font-size="9" fill="#1a1612">Grok 4 (4 parallel agents)</text>
    <text x="15" y="98" font-size="9" font-weight="700" fill="#1a1612">RL-tuned reasoning:</text>
    <text x="170" y="98" font-size="9" fill="#1a1612">M39 territory; universal at frontier</text>
    <text x="15" y="116" font-size="9" font-style="italic" fill="#c1502e">2026 trend: unified switchable wins</text>
  </g>

  <!-- Optimizer / precision -->
  <g transform="translate(20, 320)">
    <rect x="0" y="0" width="700" height="125" fill="#ffd5dc" stroke="#b85a6c" stroke-width="2"/>
    <text x="350" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">⑤ Optimizer × Precision (training-time choices)</text>

    <text x="15" y="46" font-size="10" font-weight="700" fill="#1a1612">Optimizer:</text>
    <text x="130" y="46" font-size="10" fill="#1a1612">AdamW (almost all) → Muon (DeepSeek V4 first frontier adoption; AdamW retained for embeddings/head)</text>

    <text x="15" y="68" font-size="10" font-weight="700" fill="#1a1612">Precision:</text>
    <text x="130" y="68" font-size="10" fill="#1a1612">FP8 (M14): closed-source default; NVFP4 (M41): MLPerf v5.1, DeepSeek V4 expert weights, frontier 2026</text>

    <text x="15" y="90" font-size="10" font-weight="700" fill="#1a1612">Residual:</text>
    <text x="130" y="90" font-size="10" fill="#1a1612">Standard residual → mHC (Manifold-Constrained Hyper-Connections, DeepSeek V4) for trillion-scale stability</text>

    <text x="15" y="112" font-size="9" font-style="italic" fill="#1f5f5b">  2026 trend: AdamW + FP8/NVFP4 + standard residual is mainstream; novel choices in DeepSeek V4 are the experimental edge</text>
  </g>
</svg>
</div>

<p>The five primitives — attention mechanism, position encoding, MoE pattern, reasoning architecture, optimizer × precision — define a 2026 model. Reading any model card is identifying the choices on these axes. <em>Most flagship models share most choices</em>; the differences are concentrated in 1-2 axes per model.</p>

<h2>DeepSeek V4: the case study in current architectural innovation</h2>

<p>DeepSeek V4 (April 24, 2026) is the most architecturally novel frontier release of 2026 and worth detailed treatment. The release introduced three primitives that other labs will likely adopt:</p>

<h3>Hybrid Attention: CSA + HCA</h3>

<p>The headline innovation. Combines two attention mechanisms across layers:</p>

<ul>
  <li><strong>Compressed Sparse Attention (CSA)</strong>: compresses KV entries 4× along the sequence dimension; uses an FP4-based "Lightning Indexer" to select top-k compressed blocks per query. Local context with sparse selection.</li>
  <li><strong>Heavily Compressed Attention (HCA)</strong>: applies aggressive 128× compression to the sequence; the compressed sequence is so short that <em>dense attention becomes cheap</em>. Used for global view of full context.</li>
</ul>

<p>The two mechanisms are interleaved across layers. CSA layers handle local detail; HCA layers handle global retrieval. Combined result: <strong>27% of single-token inference FLOPs and 10% of KV cache vs DeepSeek V3.2 at 1M context length</strong>. The Lightning Indexer in FP4 is itself an interesting design — using NVFP4 (M41) for the routing decisions while keeping main attention at higher precision.</p>

<pre><code><span class="com"># Schematic of layer interleaving in DeepSeek V4</span>
<span class="kw">for</span> layer_idx <span class="kw">in</span> <span class="fn">range</span>(num_layers):
    <span class="kw">if</span> layer_idx % <span class="num">2</span> == <span class="num">0</span>:
        <span class="com"># CSA: 4× compression + FP4 lightning indexer + sparse selection</span>
        out = <span class="fn">compressed_sparse_attention</span>(x, compress=<span class="num">4</span>, top_k=<span class="num">64</span>)
    <span class="kw">else</span>:
        <span class="com"># HCA: 128× compression + dense attention on compressed sequence</span>
        out = <span class="fn">heavily_compressed_attention</span>(x, compress=<span class="num">128</span>)
    x = x + out  <span class="com"># residual (or mHC; see below)</span></code></pre>

<h3>Manifold-Constrained Hyper-Connections (mHC)</h3>

<p>Replaces standard residual connections. The problem: at trillion-parameter scale across hundreds of layers, residual connections suffer from signal amplification or collapse — small perturbations compound. mHC constrains the mixing matrices to lie in the <strong>Birkhoff Polytope</strong> (the space of doubly stochastic matrices) using the Sinkhorn-Knopp algorithm. Result: signal magnitude preserved through deep stacks; training remains stable at scale.</p>

<p>This is a clear example of <em>scale-driven architectural change</em>. Standard residual connections work fine at 70B; they don't at 1.6T without modification. Other trillion-parameter models will likely adopt similar mechanisms.</p>

<h3>Muon optimizer</h3>

<p>DeepSeek V4 switches from AdamW to <strong>Muon</strong> for most parameters. AdamW is retained for embeddings, prediction head, and RMSNorm weights (the parts that benefit from per-parameter adaptive rates). DeepSeek reports faster convergence and more stable training at trillion-parameter scale — peak learning rate 2.0e-4 with cosine decay.</p>

<p>Muon is the first non-AdamW optimizer to make a serious appearance at frontier scale; the choice is being closely watched. Whether it becomes the new default depends on independent reproductions over the next few months.</p>

<h3>Switchable thinking modes</h3>

<p>V4 folds the previously separate R reasoning model into the base model with three inference modes:</p>

<ul>
  <li><strong>Non-think (fast)</strong>: standard fast inference; no extended reasoning.</li>
  <li><strong>Think High (logical analysis)</strong>: moderate reasoning trace.</li>
  <li><strong>Think Max (full reasoning extent)</strong>: extended reasoning chain.</li>
</ul>

<p>Same model weights; mode determined at inference time. This is the 2026 industry direction — Claude Opus 4.7's extended thinking, Gemini 3 Deep Think, GPT-5.5's reasoning levels are all variants of this pattern. The era of separate "reasoning model" releases is ending.</p>

<h3>FP4 quantization-aware training</h3>

<p>DeepSeek V4 trained with <strong>FP4 quantization-aware training applied to MoE expert weights</strong>. This is the production deployment of M41's NVFP4 recipe — full integration into a frontier training run. V4-Pro at 33T training tokens with FP4 QAT validates the approach at trillion-parameter scale.</p>

<h2>Llama 4: the case study in long-context architecture</h2>

<p>Llama 4 (April 2025) is the other architectural innovation reference. The headline: <strong>10M token context window in Scout</strong>, achieved without any inference-time fine-tuning by extrapolating from 256K training context.</p>

<h3>iRoPE: interleaved RoPE + NoPE</h3>

<p>The mechanism. Layers alternate in a 3:1 pattern:</p>

<ul>
  <li><strong>RoPE layers (3 of every 4)</strong>: traditional rotary positional embeddings; preserve local token order. Apply chunked attention.</li>
  <li><strong>NoPE layers (1 of every 4)</strong>: no positional encoding at all. Apply full causal attention without positional constraints. The causal mask provides the only positional signal.</li>
</ul>

<p>The intuition: traditional RoPE degrades at extreme context lengths because rotational frequencies don't generalize beyond training distribution. NoPE layers don't have this problem — they only use the causal structure. Interleaving gives the model both local positional precision (RoPE) and unlimited length generalization (NoPE).</p>

<p>Combined with <strong>inference-time temperature scaling</strong>, Llama 4 trained at 256K context generalizes to 10M context with near-perfect retrieval (Scout on Needle-in-Haystack at 10M maintains &gt;99% accuracy). This is the published empirical claim; a substantial improvement over models that hit walls at 128K-1M.</p>

<pre><code><span class="kw">def</span> <span class="fn">irope_layer_pattern</span>(num_layers):
    <span class="dq">"</span><span class="dq">"</span><span class="dq">"</span><span class="com">Llama 4 iRoPE: 3 RoPE + 1 NoPE per group of 4 layers</span><span class="dq">"</span><span class="dq">"</span><span class="dq">"</span>
    pattern = []
    <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(num_layers):
        <span class="kw">if</span> (i + <span class="num">1</span>) % <span class="num">4</span> == <span class="num">0</span>:
            pattern.<span class="fn">append</span>(<span class="str">"NoPE"</span>)   <span class="com"># every 4th layer</span>
        <span class="kw">else</span>:
            pattern.<span class="fn">append</span>(<span class="str">"RoPE"</span>)   <span class="com"># the other 3</span>
    <span class="kw">return</span> pattern</code></pre>

<h3>Scout vs Maverick: pure MoE vs alternating MoE</h3>

<p>Llama 4 also illustrates the MoE pattern divergence:</p>

<ul>
  <li><strong>Scout (109B total / 17B active)</strong>: <em>pure MoE with 16 experts</em>; every FFN layer is a MoE layer. Designed for extreme efficiency at single-GPU inference (fits on one H100 with int4 quantization).</li>
  <li><strong>Maverick (400B total / 17B active)</strong>: <em>MoE alternating with dense layers</em>; 128 experts in MoE layers, but only half the layers are MoE — the other half are dense FFNs. The intuition: dense layers provide unsparsified capacity for tasks that don't fit expert specializations.</li>
</ul>

<p>Both models have 17B active parameters per token (similar inference speed); they differ in total capacity (109B vs 400B) and how they distribute it. <em>Same active-compute budget; different total-capacity strategies</em>. Production teams choose between them based on whether they need maximum capability (Maverick) or efficient deployment (Scout).</p>

<h3>Early fusion multimodality</h3>

<p>Llama 4 is natively multimodal — text and image inputs, text and code outputs. <strong>Early fusion</strong>: vision tokens and text tokens enter the unified token stream from the very first layer (rather than being projected later via a connector). This is the Era 3a VLM design (covered below); Llama 4 was an early adopter at frontier scale.</p>

<p>Co-distillation from <strong>Llama 4 Behemoth</strong> (~2T parameter teacher, still in preview at release) using a novel dynamic-weighting loss between student and teacher logits is the training methodology — <em>knowledge distillation at frontier scale</em>, with Behemoth providing the supervision signal for Scout/Maverick.</p>

<h2>The VLM taxonomy: three eras</h2>

<p>Vision-language models have a cleaner taxonomic structure than LLMs because the architectural evolution has clearer phases. The Vision-Language-Models-Overview survey (February 2026) names three distinct eras:</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 410" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Three eras of VLM design — and where 2026 models sit</text>

  <!-- Era 1 -->
  <g transform="translate(20, 50)">
    <rect x="0" y="0" width="700" height="100" fill="#fff8a8" stroke="#1a1612" stroke-width="1.5"/>
    <text x="20" y="22" font-size="13" font-weight="700" fill="#1a1612">Era 1 (2021-2023): Frozen encoder + frozen LLM + learnable connector</text>

    <!-- Mini diagram -->
    <rect x="20" y="35" width="80" height="30" fill="#fcecec" stroke="#c1502e" stroke-width="1"/>
    <text x="60" y="54" text-anchor="middle" font-size="9" font-weight="700" fill="#1a1612">Vision enc.</text>
    <text x="60" y="76" text-anchor="middle" font-size="8" fill="#6b5d4f">(frozen)</text>

    <rect x="120" y="35" width="60" height="30" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/>
    <text x="150" y="54" text-anchor="middle" font-size="9" font-weight="700" fill="#1a1612">Connect.</text>
    <text x="150" y="76" text-anchor="middle" font-size="8" fill="#6b5d4f">(trained)</text>

    <rect x="200" y="35" width="80" height="30" fill="#fff5d8" stroke="#d4a017" stroke-width="1"/>
    <text x="240" y="54" text-anchor="middle" font-size="9" font-weight="700" fill="#1a1612">LLM</text>
    <text x="240" y="76" text-anchor="middle" font-size="8" fill="#6b5d4f">(frozen)</text>

    <text x="300" y="48" font-size="10" font-weight="700" fill="#1a1612">Examples:</text>
    <text x="380" y="48" font-size="10" fill="#1a1612">CLIP, BLIP, BLIP-2, Flamingo, MiniGPT-4</text>
    <text x="300" y="64" font-size="10" font-weight="700" fill="#1a1612">Trade-off:</text>
    <text x="380" y="64" font-size="10" fill="#1a1612">cheap to train (only connector); limited capability</text>
    <text x="300" y="80" font-size="10" font-weight="700" fill="#1a1612">Status 2026:</text>
    <text x="380" y="80" font-size="10" fill="#1a1612">deprecated for new flagships; useful for research/edge</text>
  </g>

  <!-- Era 2 -->
  <g transform="translate(20, 165)">
    <rect x="0" y="0" width="700" height="100" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="20" y="22" font-size="13" font-weight="700" fill="#1a1612">Era 2 (2023-2025): LLM trunk with vision as bolt-on adapter</text>

    <!-- Mini diagram -->
    <rect x="20" y="35" width="80" height="30" fill="#fcecec" stroke="#c1502e" stroke-width="1"/>
    <text x="60" y="54" text-anchor="middle" font-size="9" font-weight="700" fill="#1a1612">Vision enc.</text>
    <text x="60" y="76" text-anchor="middle" font-size="8" fill="#6b5d4f">(fine-tuned)</text>

    <rect x="120" y="35" width="60" height="30" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/>
    <text x="150" y="54" text-anchor="middle" font-size="9" font-weight="700" fill="#1a1612">Connect.</text>
    <text x="150" y="76" text-anchor="middle" font-size="8" fill="#6b5d4f">(trained)</text>

    <rect x="200" y="35" width="80" height="30" fill="#fff5d8" stroke="#d4a017" stroke-width="2"/>
    <text x="240" y="54" text-anchor="middle" font-size="9" font-weight="700" fill="#1a1612">LLM trunk</text>
    <text x="240" y="76" text-anchor="middle" font-size="8" fill="#6b5d4f">(fine-tuned)</text>

    <text x="300" y="48" font-size="10" font-weight="700" fill="#1a1612">Examples:</text>
    <text x="380" y="48" font-size="10" fill="#1a1612">LLaVA / LLaVA-NeXT, Qwen2.5-VL, GPT-4V, InternVL</text>
    <text x="300" y="64" font-size="10" font-weight="700" fill="#1a1612">Trade-off:</text>
    <text x="380" y="64" font-size="10" fill="#1a1612">leverage existing LLM; vision is "bolted on"; mid-fusion</text>
    <text x="300" y="80" font-size="10" font-weight="700" fill="#1a1612">Status 2026:</text>
    <text x="380" y="80" font-size="10" fill="#1a1612">still common for open-weight VLMs; research workhorse</text>
  </g>

  <!-- Era 3a -->
  <g transform="translate(20, 280)">
    <rect x="0" y="0" width="340" height="115" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="170" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">Era 3a: Native MM input → text out</text>

    <text x="15" y="42" font-size="10" fill="#1a1612"><tspan font-weight="700">Image, video, (audio) → unified token</tspan></text>
    <text x="15" y="56" font-size="10" fill="#1a1612"><tspan font-weight="700">stream from layer 0</tspan>; text-only output</text>

    <text x="15" y="76" font-size="9" font-weight="700" fill="#1a1612">Examples:</text>
    <text x="15" y="89" font-size="9" fill="#1a1612">Qwen3.5/3.6, Gemma 4, Gemini 3,</text>
    <text x="15" y="102" font-size="9" fill="#1a1612">GPT-5.4/5.5, Claude Opus 4.6/4.7</text>
  </g>

  <!-- Era 3b -->
  <g transform="translate(380, 280)">
    <rect x="0" y="0" width="340" height="115" fill="#ffd5dc" stroke="#b85a6c" stroke-width="2"/>
    <text x="170" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">Era 3b: Native MM input → MM out</text>

    <text x="15" y="42" font-size="10" fill="#1a1612">Same trunk; <tspan font-weight="700">generates images/audio</tspan></text>
    <text x="15" y="56" font-size="10" fill="#1a1612">in addition to text; multi-head decoder</text>

    <text x="15" y="76" font-size="9" font-weight="700" fill="#1a1612">Examples:</text>
    <text x="15" y="89" font-size="9" fill="#1a1612">GPT-Image-2, Gemini Native Image,</text>
    <text x="15" y="102" font-size="9" fill="#1a1612">Chameleon, Llama 4 (image gen branch)</text>
  </g>
</svg>
</div>

<p>The trajectory is clear: <em>each era integrates more deeply</em>. Era 1 keeps everything separate and trained independently. Era 2 fine-tunes the LLM but treats vision as bolted on. Era 3 trains everything together from scratch with mixed modalities in the same token stream from the first layer.</p>

<p>The performance gap from Era 2 → Era 3a is substantial — early fusion captures cross-modal interactions that bolt-on architectures can't. The gap from Era 3a → Era 3b is currently smaller (Era 3b is harder to train and image generation has its own quality issues); the long-term direction favors 3b for general-purpose multimodal AI.</p>

<h3>Vision encoders specifically</h3>

<p>Within Era 3a (the dominant 2026 design), the choice of vision encoder is one of the few remaining axes:</p>

<div class="table-wrap">
<table>
<caption>Vision encoders in 2026 production VLMs</caption>
<thead><tr><th>Encoder</th><th>Origin</th><th>Used by</th><th>Notes</th></tr></thead>
<tbody>
<tr><td><strong>SigLIP 2</strong></td><td>Google (2025)</td><td>Qwen3-VL, Gemma 3</td><td>Successor to SigLIP; sigmoid contrastive, multilingual, dense features</td></tr>
<tr><td><strong>InternViT-6B</strong></td><td>Shanghai AI Lab</td><td>InternVL line</td><td>Scaled vision encoder (6B params); strong on document/chart tasks</td></tr>
<tr><td><strong>DINOv2 + SigLIP fused</strong></td><td>Meta + Google</td><td>OpenVLA (M43), Cambrian-1</td><td>Combines spatial detail with language alignment</td></tr>
<tr><td><strong>Cambrian-1 multi-encoder</strong></td><td>NYU/research</td><td>Cambrian-1</td><td>Combines several encoders; captures complementary features</td></tr>
<tr><td><strong>Proprietary</strong></td><td>OpenAI / Anthropic / Google</td><td>GPT-5, Claude Opus 4.7, Gemini 3</td><td>Undisclosed; likely SigLIP-derivative + custom fine-tuning</td></tr>
</tbody>
</table>
</div>

<p>SigLIP 2 has become the open-source default in 2025-2026, similar to how Llama became the LLM trunk default. Most open VLMs use SigLIP 2 for vision; the differentiation is in the LLM trunk and training recipe.</p>

<h2>The benchmark saturation phenomenon</h2>

<p>One of the defining 2026 features: <strong>benchmarks are saturating</strong>. The "best AI model" question has gotten harder to answer because all the frontier models score within noise of each other on standard benchmarks.</p>

<div class="table-wrap">
<table>
<caption>Benchmark scores for top frontier models, April 2026</caption>
<thead><tr><th>Benchmark</th><th>GPT-5.5</th><th>Claude Opus 4.7</th><th>Gemini 3.1 Pro</th><th>DeepSeek V4-Pro</th><th>Spread</th></tr></thead>
<tbody>
<tr><td>MMMU-Pro (multimodal)</td><td>~82.0</td><td>~81.0</td><td>~82.5</td><td>—</td><td>1.5 pts</td></tr>
<tr><td>SWE-bench Verified</td><td>~79.5</td><td>~80.8</td><td>80.6</td><td>80.6</td><td>1.3 pts</td></tr>
<tr><td>GPQA Diamond</td><td>92.8</td><td>91.3</td><td>94.3</td><td>~91</td><td>3 pts</td></tr>
<tr><td>HumanEval</td><td>~96</td><td>~96</td><td>~96</td><td>~95</td><td>1 pt</td></tr>
<tr><td>AIME 2025</td><td>~92</td><td>~90</td><td>~93</td><td>~88</td><td>5 pts</td></tr>
</tbody>
</table>
</div>

<p>The pattern: <em>top models cluster within 1-3 points on most established benchmarks</em>. The differentiation has moved to specialized axes (per the multimodal benchmarks article from Digital Applied):</p>

<ul>
  <li><strong>Video understanding</strong>: Gemini 3 leads decisively. Long-form video (movies, lectures) is one of the few benchmark axes still spreading models.</li>
  <li><strong>Audio comprehension + ASR-with-reasoning</strong>: Gemini 3 leads; Qwen 3.5 Omni close on real-time applications.</li>
  <li><strong>Long-document OCR</strong>: Claude Opus 4.7 holds the crown.</li>
  <li><strong>Chart reasoning + infographics</strong>: GPT-5.5 leads.</li>
  <li><strong>Code-with-vision</strong> (debugging UI from screenshots): GPT-5.5 leads with longer reasoning traces.</li>
  <li><strong>Real-time information</strong>: Grok 4 with live X/Twitter integration is a category of one.</li>
</ul>

<p>The implication for engineering teams: <strong>"which model is best?" has been replaced by "which model is best for our specific workload?"</strong> The architectural differentiation is real but specialized; the right answer depends on whether you primarily process video (Gemini), code with screenshots (GPT-5.5), long PDFs (Claude), or need 10M-token context (Llama 4 Scout).</p>

<h2>The 25× price gap</h2>

<p>The economic story is even more dramatic than the architectural one. The price spread between cheapest and most expensive frontier model in April 2026 is roughly 25×:</p>

<div class="table-wrap">
<table>
<caption>Frontier model API pricing, April 2026 (per 1M tokens)</caption>
<thead><tr><th>Model</th><th>Input</th><th>Output</th><th>License</th></tr></thead>
<tbody>
<tr><td>Claude Opus 4.7</td><td>$15</td><td>$75</td><td>Closed</td></tr>
<tr><td>Claude Opus 4.6</td><td>$5</td><td>$25</td><td>Closed</td></tr>
<tr><td>GPT-5.5</td><td>~$2-3</td><td>~$15-18</td><td>Closed</td></tr>
<tr><td>GPT-5.4</td><td>$2.50</td><td>$15</td><td>Closed</td></tr>
<tr><td>Claude Sonnet 4.6</td><td>$3</td><td>$15</td><td>Closed</td></tr>
<tr><td>Gemini 3.1 Pro</td><td>$2</td><td>$12</td><td>Closed</td></tr>
<tr><td>Grok 4</td><td>$2</td><td>$15</td><td>Closed</td></tr>
<tr><td>DeepSeek V4-Pro</td><td>$0.55-1.74</td><td>$3.48</td><td>MIT (open weights)</td></tr>
<tr><td>MiniMax M2.5</td><td>$0.30</td><td>$1.20</td><td>Open weight</td></tr>
<tr><td>DeepSeek V4-Flash</td><td>$0.14</td><td>$0.28</td><td>MIT (open weights)</td></tr>
<tr><td>Qwen 3.6 (35B-A3B)</td><td>$0.10</td><td>~$0.40</td><td>Apache 2.0</td></tr>
</tbody>
</table>
</div>

<p>The 25× output-price spread (Qwen 3.6 at ~$0.40/M to Claude Opus 4.7 at $75/M) at <em>roughly comparable benchmark scores</em> is the structural change. <strong>Tiered routing strategies are now table stakes for production deployments</strong>: ~70% traffic to the cheapest capable model, ~25% to mid-tier, ~5% to frontier-tier; overall performance indistinguishable from all-frontier routing at ~15% of the cost.</p>

<h2>Why convergence is happening</h2>

<p>Three forces compound to produce the 2026 convergence:</p>

<ol>
  <li><strong>Public papers + open weights cycle</strong>. DeepSeek R1 (Jan 2025) → GRPO universal by July 2025. Llama 4 (April 2025) → MoE universal by year-end. NVIDIA NVFP4 paper (Sep 2025) → DeepSeek V4 FP4 QAT (April 2026). The cycle is ~6 months.</li>
  <li><strong>Talent flow</strong>. Researchers move between labs; methods don't stay proprietary for long. The "secret sauce" of frontier labs is increasingly engineering execution and data, not architectural novelty.</li>
  <li><strong>Benchmark optimization pressure</strong>. Every lab targets the same benchmarks (MMMU, SWE-bench, GPQA, AIME). Models are trained to do well on these specific evals; convergence is partially a measurement artifact.</li>
</ol>

<p>The implication: <strong>architectural innovation is high-impact but short-lived</strong>. DeepSeek V4's hybrid attention is novel today; by Q3 2026, expect Llama 5 / Qwen 4 / GLM-6 / closed-source flagships to incorporate similar mechanisms. <em>The half-life of a unique architectural advantage is now ~6-9 months</em>.</p>

<h2>What's likely next: Q3 2026 and beyond</h2>

<p>From the Q2 2026 pipeline signals (per Build Fast With AI's leaderboard):</p>

<ul>
  <li><strong>GPT-5.5 (codename "Spud")</strong>: completed pretraining around March 24, 2026. OpenAI hasn't announced release date; likely Q2-Q3 2026.</li>
  <li><strong>Claude "Mythos"</strong> (Anthropic): leaked context from prediction markets in March 2026; unconfirmed by Anthropic.</li>
  <li><strong>Grok 5</strong>: Musk has discussed it publicly.</li>
  <li><strong>DeepSeek V5</strong>: V4 was preview; V5 likely in Q3 2026.</li>
  <li><strong>Qwen 4</strong>: Alibaba's flagship update.</li>
</ul>

<p>The likely architectural directions for Q3 2026 + based on current trajectories:</p>

<ul>
  <li><strong>Hybrid attention everywhere</strong>: DeepSeek V4-style CSA+HCA or similar in Llama 5, Qwen 4, GLM-6 — the FLOP and KV-cache savings are too large to ignore at 1M+ context.</li>
  <li><strong>Trillion-parameter open weights as standard</strong>: V4-Pro is the largest open model today; expect Llama 5, Qwen 4 Max, GLM-6 all in the 1T+ range.</li>
  <li><strong>Native NVFP4 training as default</strong>: M41's recipe is mature; new training runs default to it, not FP8.</li>
  <li><strong>Unified switchable reasoning</strong>: separate reasoning model releases end; every flagship has Think/Non-Think modes.</li>
  <li><strong>Multi-agent + tool use baked in</strong>: Grok 4's parallel-agent architecture, GPT-5.5's native computer use — agentic capability becomes architectural, not bolt-on.</li>
  <li><strong>10M+ context becomes common</strong>: Llama 4 Scout's iRoPE was novel in April 2025; by Q3 2026, several flagships will offer 10M+.</li>
  <li><strong>Era 3b multimodal output</strong>: more models generating images/audio natively; the gap from Era 3a closes.</li>
</ul>

<p>The likely <em>non</em>-changes (architectural choices that have settled):</p>

<ul>
  <li>Transformer base architecture (despite Mamba/SSM developments — see M33 — production sticks with transformers).</li>
  <li>Pre-normalization with RMSNorm.</li>
  <li>SwiGLU / GeGLU activation in FFN.</li>
  <li>Rotary positional embeddings (RoPE) as the foundation, with extensions (iRoPE, NoPE layers, YaRN).</li>
  <li>BPE / sentencepiece tokenization.</li>
  <li>AdamW optimizer (Muon's adoption rate uncertain; standard remains AdamW).</li>
</ul>

<div class="ndq">
<h4>About comparing 2026 LLM and VLM architectures</h4>

<p class="q">Why has every frontier model converged on MoE, even though dense models were dominant in 2023-2024?</p>
<p class="a">Three compounding reasons. (1) <strong>Scale economics</strong>: at trillion-parameter scale, dense models become prohibitively expensive at both training and inference. MoE decouples total parameters (capacity) from active parameters per token (compute). DeepSeek V4-Pro at 1.6T total / 49B active runs at the inference cost of a dense ~50B model with capability closer to a dense ~500B model. (2) <strong>Training maturity</strong>: MoE training was tricky in 2022-2023 (load balancing, expert collapse, communication overhead). DeepSeek V3 (Dec 2024) and Llama 4 (April 2025) demonstrated mature training recipes; the engineering problems are solved. (3) <strong>Specialization benefit</strong>: experts specialize during training; different experts handle math, code, multilingual content. The specialization itself improves quality at the same active-compute. <em>By April 2026, choosing dense at frontier scale is a deliberate eccentricity</em>; the question is which MoE pattern (pure, alternating, shared experts), not whether to use MoE at all.</p>

<p class="q">Will iRoPE-style hybrid position encoding generalize beyond Llama 4, or is it a one-off?</p>
<p class="a">Likely generalizes — and DeepSeek V4's Hybrid Attention is a related move, suggesting the pattern is broader than just position encoding. The general insight: <em>uniform layers don't optimize for both local precision AND long-range generalization simultaneously</em>; interleaving specialized layers does. iRoPE (RoPE for local + NoPE for global), Hybrid Attention (CSA for local + HCA for global), MoE alternating with dense (sparse for routing + dense for capacity) — all variations of the same pattern. <em>Expect more "interleaved-specialized-layer" architectures by Q3 2026</em>: e.g., interleaved precision (some layers FP4, others FP8), interleaved depths (some short, some long), interleaved attention types (local sliding-window with global-attention layers). The base insight — that homogeneous architectures suboptimize specific axes — is now established.</p>

<p class="q">Why are VLM benchmarks saturating, and what does this mean for evaluation?</p>
<p class="a">Saturation has two distinct causes. (1) <strong>Genuine capability ceiling on the benchmark</strong>: MMMU has finite ceiling (100% accuracy); models approaching it have less room to spread. (2) <strong>Training-set contamination + benchmark optimization</strong>: every lab trains against the same benchmarks; the resulting models look similar on those benchmarks but may differ substantially on out-of-distribution evaluation. The implication for engineering teams (per M37's eval rigor playbook): <strong>build your own evals on your specific data</strong>. Public benchmarks are no longer differentiating; held-out task-specific evals are. The "vibes" approach (subjective comparison on real tasks) and per-workload evaluation matrices have become more important than the public leaderboards. M37 territory matters more in 2026 than it did in 2024.</p>

<p class="q">What's the actual difference between DeepSeek V4 and Llama 4 Maverick in production?</p>
<p class="a">Concrete differences. <strong>Architecture</strong>: DeepSeek V4-Pro at 1.6T/49B active vs Maverick at 400B/17B active — DeepSeek has 4× the total capacity, ~3× the active compute per token. <strong>Attention</strong>: V4 uses Hybrid CSA+HCA; Maverick uses standard GQA + iRoPE. <strong>Context</strong>: both 1M (Scout extends to 10M). <strong>License</strong>: V4 is MIT; Llama 4 has the Llama Community License (more restrictive). <strong>Reasoning</strong>: V4 has unified switchable thinking modes; Maverick relies on prompt-engineered CoT or fine-tuning. <strong>Pricing</strong>: V4-Pro at $0.55/M input vs Maverick at proprietary-tier pricing on most providers. <strong>Use cases</strong>: V4-Pro for cost-sensitive workloads with frontier-quality reasoning; Maverick for permissive license + Meta ecosystem integration. Both are excellent; they target different deployment scenarios. <em>For most production workloads in April 2026, DeepSeek V4-Flash is the value leader; Claude Opus 4.7 / GPT-5.5 are the quality leaders; Llama 4 Scout is the long-context specialist; Qwen 3.6 is the budget leader.</em></p>

<p class="q">What's the relationship between this module and M14 (mixed precision) + M28 (MoE) + M30 (RoPE)?</p>
<p class="a">Direct extension. M14 covered FP16/BF16/FP8 mixed precision; M41 added NVFP4. This module's "Optimizer × Precision" axis is the production application — DeepSeek V4 trained with FP4 quantization-aware on MoE expert weights is M14 + M41 deployed at trillion-scale. M28 covered MoE mechanics (router, experts, load balancing); this module's "MoE pattern" axis is the variation surface — pure vs alternating vs shared experts is M28 plus production constraints. M30 covered RoPE; this module's iRoPE discussion is M30 plus length generalization extension. <em>Each architectural axis in this module is "M-X concept × production variant"</em>; you've already seen the underlying mechanics, this module shows how labs deploy them differently. The course taught the primitives; this module taxonomizes their compositions.</p>

<p class="q">If convergence is happening, why does architecture still matter for engineering teams?</p>
<p class="a">Two reasons. (1) <strong>Convergence is partial, not complete</strong>. The 2.4-point MMMU spread is small; the 25× price spread is large. Models converge on capability, diverge on cost/efficiency/license. Architecture choice (DeepSeek's hybrid attention saving 90% KV cache, Llama 4's 10M context) directly drives operational economics. (2) <strong>Architectural understanding enables better deployment</strong>. Knowing that DeepSeek V4 is hybrid CSA+HCA tells you why long-context is cheap on it; knowing Llama 4 is iRoPE tells you why 10M context retrieval works. M37's eval rigor + M42's compute economics + M44's architectural understanding compose: you pick the right model not by reading press releases but by matching architectural choices to your workload's requirements. <em>Architecture matters less for "which is best" and more for "which is right for this specific workload"</em>; that's a more sophisticated question that requires deeper architectural literacy, not less.</p>
</div>

<h2>Code Magnets: identify the architecture from a config</h2>

<p>You receive a model config dictionary. Identify what architecture it represents by combining the right magnets. Three magnets are misleading.</p>

<div class="magnets">
<p>Arrange the magnets to write a function that classifies a model config into one of: "DeepSeek V4-style", "Llama 4-style", "dense classic".</p>

<div class="magnet-pool">
  <span class="magnet">def identify_architecture(config):</span>
  <span class="magnet">    has_hybrid_attn = config.get("attention_type") in ("hybrid_csa_hca", "csa+hca")</span>
  <span class="magnet">    has_irope = config.get("position_encoding") == "iRoPE" and config.get("nope_interval") == 4</span>
  <span class="magnet">    has_irope = config.get("position_encoding") == "RoPE"</span>
  <span class="magnet">    is_moe = config.get("num_experts", 1) > 1</span>
  <span class="magnet">    is_moe = config.get("total_params") > 100e9</span>
  <span class="magnet">    if has_hybrid_attn and is_moe:</span>
  <span class="magnet">        return "DeepSeek V4-style"</span>
  <span class="magnet">    if has_irope and is_moe:</span>
  <span class="magnet">        return "Llama 4-style"</span>
  <span class="magnet">    if has_irope:</span>
  <span class="magnet">        return "Llama 4-style"</span>
  <span class="magnet">    if not is_moe:</span>
  <span class="magnet">        return "dense classic"</span>
  <span class="magnet">    return "MoE — uncategorized"</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">def</span> <span class="fn">identify_architecture</span>(config):
    has_hybrid_attn = config.<span class="fn">get</span>(<span class="str">"attention_type"</span>) <span class="kw">in</span> (<span class="str">"hybrid_csa_hca"</span>, <span class="str">"csa+hca"</span>)
    has_irope = config.<span class="fn">get</span>(<span class="str">"position_encoding"</span>) == <span class="str">"iRoPE"</span> <span class="kw">and</span> config.<span class="fn">get</span>(<span class="str">"nope_interval"</span>) == <span class="num">4</span>
    is_moe = config.<span class="fn">get</span>(<span class="str">"num_experts"</span>, <span class="num">1</span>) &gt; <span class="num">1</span>
    <span class="kw">if</span> has_hybrid_attn <span class="kw">and</span> is_moe:
        <span class="kw">return</span> <span class="str">"DeepSeek V4-style"</span>
    <span class="kw">if</span> has_irope <span class="kw">and</span> is_moe:
        <span class="kw">return</span> <span class="str">"Llama 4-style"</span>
    <span class="kw">if</span> <span class="kw">not</span> is_moe:
        <span class="kw">return</span> <span class="str">"dense classic"</span>
    <span class="kw">return</span> <span class="str">"MoE — uncategorized"</span></code></pre>
<p>The traps:</p>
<ul>
  <li><code>has_irope = config.get("position_encoding") == "RoPE"</code>: too permissive — checks only for RoPE, misses the iRoPE-specific signature (interleaved NoPE layers every 4th). Plain RoPE is used by most 2026 models (Qwen, GLM, DeepSeek pre-V4, etc.); it's not Llama 4 specific. The Llama 4 signature is <em>RoPE + NoPE in 3:1 interleaving</em>; you need to check for that interleaving pattern (<code>nope_interval == 4</code>) to identify it. Without that check, every RoPE model gets misclassified as Llama 4.</li>
  <li><code>is_moe = config.get("total_params") &gt; 100e9</code>: confuses model size with MoE structure. Many large models are dense (Nemotron Ultra 253B is dense; Llama 3.1 405B is dense). MoE is identified by having multiple experts in FFN layers — <code>num_experts &gt; 1</code> — not by parameter count. The wrong magnet would classify any large dense model as MoE, missing the distinction that the question hinges on.</li>
  <li><code>if has_irope: return "Llama 4-style"</code> (without checking <code>is_moe</code>): triggers Llama-4 classification on any model that happens to have iRoPE-like position encoding, even if it's dense. The complete Llama 4 signature requires both iRoPE AND MoE; either alone is insufficient to identify it specifically. The wrong magnet reduces the test to "if iRoPE → Llama 4," missing the architectural composition.</li>
</ul>
<p>The pattern: <strong>each architecture is identified by a composition of primitives, not a single primitive</strong>. DeepSeek V4 = MoE + hybrid CSA+HCA attention. Llama 4 = MoE + iRoPE with NoPE-every-4th interleaving. Dense classic = NOT MoE (regardless of position encoding). Reading 2026 model cards is exactly this composition-identification exercise — the primitives are well-known; the labs combine them differently. The most common mistake (Magnet 2) confuses <em>scale</em> with <em>structure</em>; the second most common (Magnet 3) confuses <em>part</em> with <em>whole</em>.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each architectural primitive or model to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>Hybrid Attention (CSA+HCA)</div>
  <div>A. DeepSeek V4: 4× compress + 128× compress interleaved; 27% FLOPs / 10% KV vs V3.2 at 1M.</div>

  <div>iRoPE</div>
  <div>B. Llama 4: 3 RoPE layers + 1 NoPE layer interleaved; enables 10M context generalization from 256K training.</div>

  <div>Era 3a VLM</div>
  <div>C. Native multimodal input → text-out; single transformer trained on mixed modalities (Qwen3.5+, Gemini 3, GPT-5+).</div>

  <div>Pure MoE</div>
  <div>D. Every FFN replaced with experts (Llama 4 Scout, GLM-5, DeepSeek V4); maximum sparsity for given total capacity.</div>

  <div>Manifold-Constrained Hyper-Connections (mHC)</div>
  <div>E. DeepSeek V4 residual replacement; constrains mixing matrices to Birkhoff Polytope for trillion-scale stability.</div>

  <div>Muon optimizer</div>
  <div>F. DeepSeek V4 first-frontier non-AdamW choice; faster convergence at trillion-scale; AdamW retained for embeddings/head.</div>

  <div>Switchable thinking modes</div>
  <div>G. Single base model with Non-think / Think High / Think Max modes (DeepSeek V4); 2026 industry direction unifying reasoning.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>Hybrid Attention</strong> → A<br>
<strong>iRoPE</strong> → B<br>
<strong>Era 3a VLM</strong> → C<br>
<strong>Pure MoE</strong> → D<br>
<strong>mHC</strong> → E<br>
<strong>Muon optimizer</strong> → F<br>
<strong>Switchable thinking</strong> → G
</p>
<p>The mental shortcut: <em>CSA+HCA compresses attention, iRoPE interleaves position, Era 3a fuses early, pure MoE replaces FFNs, mHC stabilizes residuals, Muon replaces AdamW, switchable thinking unifies reasoning</em>. Each is the 2026 production answer to a specific scaling problem.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> Your team needs to choose a model for a new product: a customer support bot processing internal tickets that include screenshots and PDFs. ~50M tokens/month projected. Walk through the model selection.</p>
<details class="answer"><summary>show answer</summary>
<p>Apply the M42 + M44 framework:</p>
<p>(1) <strong>Workload characterization</strong>. Customer support → moderate reasoning (categorize, draft response). Screenshots → vision needed. PDFs → long-document OCR / understanding. ~50M tokens/month → moderate volume; pricing matters but not extreme.</p>
<p>(2) <strong>Match to model strengths</strong>. Long-document OCR is Claude Opus 4.7's strength (per the differentiation matrix). Code-with-vision (parsing UI screenshots) is GPT-5.5's strength. Both are reasonable choices. Gemini 3.1 Pro is also strong on vision and offers the best $/quality at $2/$12.</p>
<p>(3) <strong>Cost analysis at 50M tokens/month</strong>. Assume 70% input / 30% output split. Monthly cost: Claude Opus 4.7 = 35M × $15 + 15M × $75 = $525 + $1,125 = ~$1,650/month for Opus-tier. GPT-5.5 ≈ ~$300/month. Gemini 3.1 Pro = 35M × $2 + 15M × $12 = $250/month. <em>Gemini 3.1 Pro wins on price-quality for a moderate-volume vision workload.</em></p>
<p>(4) <strong>Tiered routing strategy</strong>. Per the M42 + April 2026 framework: 70% of tickets are simple (categorization, quick response) → route to DeepSeek V4-Flash at $0.14/$0.28 = nearly free at 35M tokens/month. 25% are moderate (need vision + reasoning) → route to Gemini 3.1 Pro at $250/month effective. 5% are complex (escalation drafting, multi-step) → route to Claude Opus 4.7 at premium pricing. Total: ~$60/month for the 70% + ~$60/month for the 25% + ~$80/month for the 5% = <em>~$200/month total</em>, vs $1,650/month if routing everything to Opus.</p>
<p>(5) <strong>Validation</strong>. Build M37-grade evals on your specific tickets. Test each tier of the routing strategy. Verify that the cheap-tier handles 70% of tickets correctly; if not, shift the routing percentages.</p>
<p>(6) <strong>Implementation</strong>. Use a routing layer (LiteLLM, custom router); prompt-based difficulty classification (the easy tier sees a quick "is this a simple categorization?" check); fall through to higher tiers on failure.</p>
<p>The general lesson: <strong>in 2026, model selection is portfolio optimization, not single-choice</strong>. The right answer is usually a routing strategy across 2-3 models, not "pick the best model and use it for everything." This is the practical implementation of M42's three-pool decomposition + M44's architectural understanding.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Walk through why DeepSeek V4-Pro can reach 1M context with 27% of V3.2's FLOPs. What's the actual mechanism?</p>
<details class="answer"><summary>show answer</summary>
<p>The mechanism is a layered cost reduction:</p>
<p>(1) <strong>Standard attention cost at 1M context</strong>. Self-attention is O(N² × d) where N is sequence length and d is head dimension. At 1M tokens: ~10¹² operations per head. Multiply by ~64 heads × hundreds of layers → astronomical compute. KV cache: ~40-80 GB of GPU memory just for the cache at 1M tokens with normal attention. Both compute and memory are blockers.</p>
<p>(2) <strong>CSA layers (Compressed Sparse Attention)</strong>. Compress KV entries 4× along sequence dimension. Now N effectively reduced to N/4 = 250K equivalent. Then use FP4-based Lightning Indexer to select top-k blocks per query — only attend to the most relevant compressed blocks. Combined: ~16× FLOP reduction vs full attention at this layer (4× from compression, ~4× from sparsity).</p>
<p>(3) <strong>HCA layers (Heavily Compressed Attention)</strong>. Compress 128× — sequence becomes ~8K equivalent at 1M input. Sequence is so short that dense attention on the compressed version becomes cheap. ~16,000× FLOP reduction vs full attention at this layer (128² compression of the quadratic).</p>
<p>(4) <strong>Layer interleaving</strong>. CSA and HCA layers alternate. CSA captures local detail (with sparsity for efficiency); HCA captures global retrieval (with heavy compression). Each layer's role is specialized; together they cover the attention requirements with much less total compute than uniform full attention.</p>
<p>(5) <strong>KV cache savings</strong>. Compressed KV entries are smaller. CSA stores 4×-compressed KV; HCA stores 128×-compressed KV. Total KV cache: 10% of V3.2's at 1M context. Memory bandwidth and storage both scale down.</p>
<p>(6) <strong>The 27% / 10% numbers</strong>. Single-token inference FLOPs at 1M context: V4 uses 27% of V3.2 (3.7× speedup). KV cache: 10% of V3.2 (10× memory reduction). The combination makes 1M context economically practical for production deployment.</p>
<p>(7) <strong>The Lightning Indexer FP4 detail</strong>. The indexer (which decides which compressed blocks to attend to) runs in FP4. This is M41's NVFP4 applied to the routing decision specifically; the routing accuracy doesn't need full precision since it's selecting top-k anyway.</p>
<p>The general lesson: <strong>the FLOP and memory savings from architectural innovation can dwarf the savings from any other source</strong>. M14's FP8 → M41's NVFP4 saves ~2× compute. DeepSeek V4's hybrid attention saves ~3.7×. Stacking them gives ~7× total. This is why architectural choices matter more than precision choices alone — the FLOP reduction is multiplicative across all the levers.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> A researcher wants to extend a 70B dense model to 1M context efficiently. Sketch the practical paths in 2026.</p>
<details class="answer"><summary>show answer</summary>
<p>Three approaches with different cost/quality tradeoffs:</p>
<p>(1) <strong>YaRN-style RoPE scaling</strong>. Cheapest approach: fine-tune for 1-3 days on long-context data with YaRN frequency rescaling. Works to ~256K-512K with reasonable quality; degrades beyond. ~$10-30K compute. Quick to ship; doesn't really get you to a usable 1M.</p>
<p>(2) <strong>iRoPE-style retrofit</strong>. Replace some RoPE layers with NoPE layers (every 4th, like Llama 4). Requires fine-tuning at extended context (256K typical) and inference-time temperature scaling. Works to 1M-10M with good quality if the original model trained at 256K+. ~$50-200K compute. Substantial engineering investment but proven approach (Llama 4 published recipe).</p>
<p>(3) <strong>Hybrid attention retrofit</strong>. Add CSA + HCA layers in DeepSeek V4 style. Most expensive but gives the best 1M-context economics (27% FLOPs, 10% KV vs original at 1M). Requires substantial restructuring — this is essentially "rebuild the attention mechanism." ~$200K-1M compute and significant engineering. Not a fine-tuning operation; closer to architectural surgery. Best long-term answer if 1M context is critical.</p>
<p>(4) <strong>Hybrid approach</strong>. Use YaRN to get to 256K cheaply; deploy production at 256K; iterate to iRoPE retrofit for 1M+ when needed. Many teams sit at intermediate context lengths because the marginal benefit of going from 256K to 1M doesn't justify the engineering cost for their specific use case.</p>
<p>(5) <strong>Practical recommendation</strong>. For most teams: <em>don't extend your 70B dense model to 1M; switch to a model designed for it</em>. DeepSeek V4-Flash at $0.14/M input handles 1M natively. Llama 4 Scout handles 10M natively. The cost of switching to a purpose-built long-context model is typically lower than the cost of retrofitting a 70B dense. The engineering math favors using the right tool, not converting the wrong tool.</p>
<p>The general lesson: <strong>in 2026, architectural retrofitting is rarely the right answer for long-context</strong>. The labs that designed for 1M+ from scratch (DeepSeek V4, Llama 4) have order-of-magnitude advantages. Build on top of them; don't try to bolt 1M onto a 128K architecture.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> Forecast: which architectural choices will be in every flagship by Q3 2027? Sketch your reasoning.</p>
<details class="answer"><summary>show answer</summary>
<p>Reading the convergence trajectory and current frontier signals:</p>
<p><strong>Highly likely universal by Q3 2027</strong>:</p>
<p>(1) <strong>MoE at frontier scale</strong>. Already universal April 2026; no reason to expect divergence. Expect 30-40× sparsity ratios (total/active) as the standard.</p>
<p>(2) <strong>1M+ context windows</strong>. Already standard for new flagships; expect 5M+ as the new baseline by Q3 2027 with iRoPE-style innovations widespread.</p>
<p>(3) <strong>Hybrid attention (CSA+HCA-like)</strong>. DeepSeek V4 demonstrated; expect Llama 5, Qwen 4, GLM-6, GPT-6 (whatever name), Claude 5 to incorporate similar mechanisms. The FLOP and KV savings are too compelling to leave on the table.</p>
<p>(4) <strong>Unified switchable reasoning</strong>. Already converging in April 2026 (DeepSeek V4, Claude Opus 4.7 extended thinking, Gemini 3 Deep Think); separate reasoning models are unlikely to see new releases.</p>
<p>(5) <strong>NVFP4 training as default</strong>. M41's recipe is mature; new training runs default to it. Expect FP4 to be the standard for 2026-2027 training.</p>
<p>(6) <strong>Native multimodal (Era 3a)</strong>. Already universal for closed flagships; open-weight models converging rapidly (Qwen3-VL, Gemma 4, Llama 4 are early adopters).</p>
<p>(7) <strong>Tool use / agentic capability as architectural feature</strong>, not bolt-on. Grok 4's parallel agents, GPT-5.5's native computer use, Claude's MCP integration all point this direction.</p>
<p><strong>Possibly universal but uncertain</strong>:</p>
<p>(8) <strong>Muon optimizer</strong>. DeepSeek V4 was the first frontier adoption; whether AdamW gets displaced depends on independent reproductions and whether the convergence advantages prove robust. <em>Could go either way</em>.</p>
<p>(9) <strong>Manifold-Constrained Hyper-Connections (mHC)</strong>. Specific to DeepSeek V4; might be needed only at trillion-parameter scale. Unclear if smaller models adopt it.</p>
<p>(10) <strong>Era 3b multimodal output</strong>. Currently smaller-scale; depends on whether image/audio generation quality from unified models matches specialized models.</p>
<p><strong>Unlikely to converge by Q3 2027</strong>:</p>
<p>(11) <strong>Mamba/SSM replacing transformer</strong>. M33's SSM lineage continues but transformers remain dominant. Possibly hybrid architectures (transformer + SSM layers) emerge but full replacement unlikely.</p>
<p>(12) <strong>Sub-FP4 precision</strong>. FP4 is at the edge of viability; FP3 / FP2 training would require substantial new techniques. Possible but not by Q3 2027.</p>
<p>(13) <strong>Multi-agent as the dominant paradigm</strong>. Grok 4's approach is novel but unproven; possible direction but not guaranteed.</p>
<p>The general framework: <strong>architectural choices that compound (compose with multiple other choices) and have public implementations spread fastest</strong>. MoE + hybrid attention + native multimodal + 1M+ context all compound. Specific optimizer choices (Muon) or specific residual choices (mHC) need independent validation before widespread adoption. <em>Forecast bias: assume convergence on the compound primitives; assume divergence on the specialized ones.</em></p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>The April 2026 frontier LLM zoo has converged on three macro-patterns: <strong>MoE is universal at frontier scale</strong>, <strong>1M+ context is table stakes</strong>, <strong>reasoning consolidates into the base model</strong> with switchable thinking modes.</li>
  <li>Five architectural primitives differentiate 2026 models: <strong>attention mechanism</strong> (full / GQA / sparse / hybrid CSA+HCA / sliding-window), <strong>position encoding</strong> (RoPE / iRoPE / YaRN / NoPE), <strong>MoE pattern</strong> (pure / alternating / shared experts / routing variant), <strong>reasoning architecture</strong> (separate model / unified switchable / always-on / multi-agent), <strong>optimizer × precision</strong> (AdamW + FP8 dominant; Muon + NVFP4 frontier).</li>
  <li><strong>DeepSeek V4 (April 24, 2026)</strong>: MoE 1.6T/49B active + 284B/13B Flash; <strong>Hybrid Attention</strong> (CSA 4× compress + Lightning Indexer FP4 + HCA 128× compress dense); <strong>mHC residuals</strong> (Birkhoff Polytope, Sinkhorn-Knopp); <strong>Muon optimizer</strong>; switchable thinking (Non-think / Think High / Think Max); FP4 quantization-aware training; 27% FLOPs / 10% KV vs V3.2 at 1M context. <strong>MIT licensed</strong>.</li>
  <li><strong>Llama 4 Scout/Maverick (April 2025)</strong>: MoE 17B active; <strong>iRoPE</strong> (3 RoPE + 1 NoPE per 4 layers); <strong>10M context</strong> (Scout) / 1M (Maverick); early fusion multimodal; co-distillation from Behemoth teacher.</li>
  <li>Other production frontier MoE models: <strong>GLM-5.1</strong> (744B/40B, MIT, DSA, agentic-focused), <strong>Qwen 3.6</strong> (35B/3B, Apache 2.0, multilingual), <strong>Kimi K2.6</strong> (Moonshot, agentic), <strong>MiniMax M2.5</strong>, <strong>MiMo V2.5</strong> (Xiaomi, 1.02T/42B + 310B/15B multimodal).</li>
  <li><strong>Closed-source flagships</strong>: GPT-5.4/5.5 (256K/1M Codex), Claude Opus 4.6/4.7 (200K + 1M beta), Gemini 3.1 Pro (1M, multimodal leader), Grok 4 (multi-agent variant). All converge within 2.4 points on MMMU-Pro, 1.3 points on SWE-bench Verified.</li>
  <li><strong>VLM design taxonomy (3 eras)</strong>: <strong>Era 1</strong> frozen encoder + frozen LLM + connector (CLIP, BLIP, Flamingo); <strong>Era 2</strong> LLM trunk + bolt-on vision adapter (LLaVA, Qwen2.5-VL, GPT-4V); <strong>Era 3</strong> single transformer trained on mixed-modality data, forking into <strong>3a</strong> (early fusion text-out, dominant in 2026: Qwen3.5+, Gemini 3, GPT-5.x, Claude 4.x) and <strong>3b</strong> (multimodal output: GPT-Image-2, Chameleon).</li>
  <li><strong>Vision encoders</strong>: SigLIP 2 dominant (Qwen3-VL, Gemma 3); InternViT-6B for InternVL; DINOv2+SigLIP fused (M43's OpenVLA, Cambrian-1); proprietary (GPT-5, Claude Opus 4.7, Gemini 3 — undisclosed).</li>
  <li><strong>Benchmark saturation</strong>: top four flagships within 2.4 points on MMMU-Pro by April 2026; differentiation moved to <strong>video</strong> (Gemini 3), <strong>audio</strong> (Gemini 3), <strong>long-document OCR</strong> (Claude Opus 4.7), <strong>chart reasoning</strong> (GPT-5.5), <strong>code-with-vision</strong> (GPT-5.5), <strong>real-time</strong> (Grok 4).</li>
  <li><strong>The 25× price gap</strong> between cheapest and most expensive frontier model defines 2026 economics. Tiered routing (70% cheap / 25% mid / 5% premium) is the production default; performance indistinguishable from all-frontier at ~15% the cost.</li>
  <li><strong>Convergence cycle is ~6 months</strong>: novel architectural choice published → industry-wide adoption. DeepSeek R1 → GRPO universal in 6 months. Llama 4 MoE → universal MoE by year-end. NVFP4 paper → DeepSeek V4 FP4 training in 6 months.</li>
  <li>Q3 2026+ likely directions: <strong>hybrid attention everywhere</strong>, <strong>trillion-parameter open weights as standard</strong>, <strong>NVFP4 default</strong>, <strong>unified switchable reasoning universal</strong>, <strong>10M+ context common</strong>, <strong>multi-agent / tool use baked in</strong>, <strong>Era 3b multimodal output</strong> closes gap with 3a.</li>
  <li>The reflex for 2026 model selection: <strong>identify your workload's primary axis</strong> (video / coding / long-doc / multilingual / cost-sensitive); <strong>pick the model with architectural advantages on that axis</strong>; <strong>route 70% of traffic to a cheaper tier</strong>; <strong>build M37-grade evals on your specific data</strong>. Architectural literacy enables better deployment decisions; convergence makes the broad differences smaller and the specific ones more important.</li>
</ul>
</div>

<h2>Where this leaves us</h2>

<p>You started this course with <code>x.stride()</code>. You finished it identifying the architectural primitives that distinguish DeepSeek V4 from Llama 4 from GLM-5 from Claude Opus 4.7. The trajectory: tensor mechanics → autograd → modules → training → distributed → kernels → MoE → transformer → reasoning models → mech interp → multimodal → distributed RLHF → agentic environments → Blackwell/NVFP4 → scaling laws → embodied AI → and now, the 2026 model zoo.</p>

<p><em>The architectural literacy you've built is the foundation</em>. The specific frontier models change every 6 months — DeepSeek V5 by Q3 2026, Llama 5 likely by Q4, Claude 5 / GPT-6 sometime. But the primitives don't change as fast: attention variants, position encodings, MoE patterns, reasoning architectures, optimizers, precision. Reading any new model card is identifying the choices on these axes; you can do that now.</p>

<p>The 2027 frontier models will mostly use these same primitives, in different compositions. The primitives that emerge over the next year (sub-FP4? Mamba-transformer hybrids? something nobody's published yet?) will extend this taxonomy, not replace it. <em>You'll be able to read those papers</em>.</p>

<p>This is what the course was for. Now go build something that didn't exist before.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">44</span>
  <span>Comparing latest LLM &amp; VLM architectures · April 2026 fin.</span>
</div>
"""

emit("44_architecture_comparison", "Module 44 — Comparing latest LLM & VLM architectures", BODY)
