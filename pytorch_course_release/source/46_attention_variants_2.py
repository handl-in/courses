#!/usr/bin/env python3
"""Module 46: Attention Variants II — Head Topology & KV-Cache Engineering. April 2026."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Attention Trilogy · Module 46 · April 2026 currency</div>
  <h1 class="module-title"><em>Attention Variants II:</em> head topology &amp; KV-cache engineering</h1>
  <p class="module-sub">— Axis C of the four-axis taxonomy: MHA → MQA → GQA → MLA → IHA → Slim Attention; the decoupled RoPE math that makes MLA work; KV cache compression at 2-bit (TurboQuant); the production decision tree</p>
</div>

<p>M45 covered Axis A (semantic role) and Axis B (receptive field) of the attention-variants taxonomy. This module covers <strong>Axis C — head topology</strong> — how queries, keys, and values are structured across attention heads, and how that structure determines KV-cache cost.</p>

<p>Of all the axes, this is the most production-critical. Head topology is where 2026 frontier models actually differ in deployment economics. The progression — MHA → MQA → GQA → MLA → IHA → Slim Attention — is the story of <em>how to share KV between heads</em>, and each step trades a different combination of memory, compute, and quality. By the end you'll know why GQA became the open-source default in 2024-2025, why DeepSeek's MLA outperformed it at scale, why IHA (Feb 2026) breaks the head-isolation barrier with pseudo-heads, and why Slim Attention can losslessly cut KV cache in half by reconstructing values from keys.</p>

<p>The other half of this module is <strong>KV-cache engineering</strong> — the techniques that compress, evict, and offload the KV cache after the head topology is chosen. These compose with topology choices: TurboQuant 2-bit quantization (April 2026, merged into vLLM) reduces a GQA model's KV by 4×; expected attention KV eviction reduces it further; combined with an MLA architecture the savings compound. <em>Production 2026 inference combines a head-topology choice with one or more cache-compression strategies</em>; understanding both is the practical literacy.</p>

<div class="keyidea">
Axis C — head topology — is where queries, keys, and values are structured across heads. The production progression: <strong>MHA</strong> (every head has its own K, V — full quality, full KV cost) → <strong>MQA</strong> (all heads share one K, one V — minimal KV but quality drops) → <strong>GQA</strong> (groups of heads share K, V — the open-source default 2024-2026) → <strong>MLA</strong> (compress K, V into low-rank latent space, decompress on demand — DeepSeek V2/V3/V4) → <strong>IHA</strong> (pseudo-heads enable cross-head mixing — Feb 2026 frontier) → <strong>Slim Attention</strong> (K-cache only; reconstruct V from K — Mar 2025; lossless 2× compression).

<strong>The math that makes MLA work</strong>: the naive "compress KV via low-rank projection" idea conflicts with RoPE because rotation must be applied to original-dimension K, not the compressed latent. <strong>Decoupled RoPE</strong> (DeepSeek's solution): split into a position-aware component (RoPE applied) and a position-free component (compressed). Cache only the latent + a small RoPE-component. Cache size: r_kv + d_qk_rope per token (vs h × (d_K + d_V) for MHA) — ~6× compression, often more.

<strong>IHA's mechanism</strong>: for each of H attention heads, construct P pseudo-heads as learned linear combinations of all H original heads (typically P = H). Interleave pseudo-heads along the sequence dimension; apply standard attention on the expanded sequence; combine pseudo-heads back into a single head. Result: P² interaction patterns per head with O(H²P) parameter overhead — strictly more expressive than MHA. Improves Multi-Key retrieval on RULER by 10-20%; +5.8% GSM8K, +2.8% MATH-500 after fine-tuning.

<strong>Slim Attention's trick</strong>: when projection matrices are square (d_K = d_V = d_model/h), V can be computed from K via V = K · W_V · W_K⁻¹. Cache only K; compute V on demand. Mathematically identical to standard attention; halves KV cache; doubles inference speed at long context. T5-11B gets 32× compression because its projection dimension exceeds embedding dimension.

<strong>KV-cache engineering compounds with head topology</strong>. <strong>TurboQuant</strong> (ICLR 2026, merged into vLLM April 2026): online vector quantization via random rotations + Lloyd-Max scalar quantization; 2-bit values, 3-bit keys, 4× capacity. Available as <code>--kv-cache-dtype turboquant_3bit_nc</code> in vLLM. <strong>Expected Attention</strong>: estimates future query distribution, evicts unlikely keys. <strong>Quest</strong>, <strong>H2O</strong>: heuristic KV eviction. <strong>InfiniGen</strong>: KV cache offloading to CPU with prefetching. <strong>Production 2026</strong>: GQA + TurboQuant 3-bit, MLA + DeepGEMM, or any combination depending on the deployment.
</div>

<h2>Two new faces — head topology and the cache</h2>

<div class="character" style="--c: #c1502e;">
  <div class="avatar" style="background: #c1502e; color: #fff;">K</div>
  <div>
    <p class="who">Cache Custodian</p>
    <p class="name">"I store the K and V tensors that grow with every generated token. When sequence length hits 1M, I become the bottleneck."</p>
    <p class="says">During autoregressive generation, every new token requires the model to attend to all previous K and V tensors — so I store them. <em>I grow linearly with sequence length, multiplied by number of layers, multiplied by number of heads, multiplied by head dimension</em>. For a 70B model at 1M tokens with MHA, I'd need 320 GB just for me. <strong>I'm why head topology matters in production</strong>: every variant on Axis C is, fundamentally, a strategy for making me smaller. MQA shares heads to reduce me. GQA groups heads to reduce me partially. MLA compresses me into a latent space. Slim Attention stores only my K half. KV-cache engineering layers (TurboQuant, eviction, offloading) compress what's left. <em>Long-context inference is essentially the engineering of how to manage me</em>.</p>
  </div>
</div>

<div class="character" style="--c: #1f5f5b;">
  <div class="avatar" style="background: #1f5f5b; color: #fff;">L</div>
  <div>
    <p class="who">Latent</p>
    <p class="name">"I'm a low-rank compressed representation of K and V. Decompress on demand, cache only me — that's MLA's secret."</p>
    <p class="says">In MHA, the cache stores K and V at full dimension — for every head, every layer, every token. Most of that information is redundant; you don't need full-precision keys and values, you need their <em>essential information</em>. <strong>I'm what's essential</strong>. DeepSeek's MLA factorizes the K, V projections through a low-rank intermediate (me) — typically dimension ~512 instead of the full ~4096. Cache only me; up-project to full K, V on demand for attention computation. <em>The catch was RoPE</em> — standard rotary embeddings need to be applied at full K dimension before storage, which would defeat my compression. <strong>Decoupled RoPE</strong> is the solution: split into position-aware (RoPE-applied, small) and position-free (compressed via me, large). Cache me + the small position-aware part. Total cache: ~93.3% reduction vs MHA. <em>I'm what makes 671B-parameter inference economically viable</em>.</p>
  </div>
</div>

<h2>The mathematical baseline: MHA in detail</h2>

<p>To make the variants legible, the baseline first. M11/M22 already covered MHA at high level; here we focus on the specific dimensions that matter for KV-cache analysis.</p>

<pre><code><span class="kw">def</span> <span class="fn">multi_head_attention</span>(x, W_Q, W_K, W_V, W_O, n_heads):
    <span class="com"># x: [batch, seq, d_model]</span>
    <span class="com"># d_model = 4096 (typical 7B model), n_heads = 32, d_head = 128</span>

    <span class="com"># Project to Q, K, V — each [batch, seq, d_model]</span>
    Q = x @ W_Q   <span class="com"># W_Q: [d_model, d_model] = [4096, 4096]</span>
    K = x @ W_K   <span class="com"># W_K: [d_model, d_model] = [4096, 4096]</span>
    V = x @ W_V   <span class="com"># W_V: [d_model, d_model] = [4096, 4096]</span>

    <span class="com"># Reshape to [batch, n_heads, seq, d_head]</span>
    Q = Q.<span class="fn">view</span>(batch, seq, n_heads, d_head).<span class="fn">transpose</span>(<span class="num">1</span>, <span class="num">2</span>)
    K = K.<span class="fn">view</span>(batch, seq, n_heads, d_head).<span class="fn">transpose</span>(<span class="num">1</span>, <span class="num">2</span>)
    V = V.<span class="fn">view</span>(batch, seq, n_heads, d_head).<span class="fn">transpose</span>(<span class="num">1</span>, <span class="num">2</span>)

    <span class="com"># Apply RoPE to Q and K (M30 territory)</span>
    Q, K = <span class="fn">apply_rope</span>(Q, K)

    <span class="com"># Standard attention per head</span>
    scores = Q @ K.<span class="fn">transpose</span>(-<span class="num">2</span>, -<span class="num">1</span>) / math.<span class="fn">sqrt</span>(d_head)
    output = F.<span class="fn">softmax</span>(scores, dim=-<span class="num">1</span>) @ V

    <span class="com"># Combine heads and project</span>
    output = output.<span class="fn">transpose</span>(<span class="num">1</span>, <span class="num">2</span>).<span class="fn">view</span>(batch, seq, d_model)
    output = output @ W_O
    <span class="kw">return</span> output</code></pre>

<p>The KV-cache analysis. During autoregressive generation:</p>

<ul>
  <li>Each new token produces a Q vector (used once, not cached) and K, V vectors (cached for future tokens to attend to).</li>
  <li><strong>Per-token KV cache cost</strong>: <code>n_heads × (d_head + d_head) × n_layers × precision</code>.</li>
  <li>For a typical 7B model (32 heads × 128 d_head × 32 layers × 2 bytes for FP16): <strong>524 KB per token</strong>.</li>
  <li>At 100K tokens of context: 52 GB just for the cache.</li>
  <li>At 1M tokens: 524 GB. <em>This doesn't fit on any single GPU</em>.</li>
</ul>

<p>Frontier models in 2026 routinely target 1M+ contexts. The KV cache becomes the dominant memory cost; reducing it is the central problem of Axis C. Every variant below is a strategy for making this number smaller.</p>

<h2>MQA — Multi-Query Attention</h2>

<p>Shazeer et al. (2019). The simplest reduction: <strong>all attention heads share a single K and V projection</strong>.</p>

<pre><code><span class="kw">def</span> <span class="fn">multi_query_attention</span>(x, W_Q, W_K, W_V, W_O, n_heads):
    <span class="com"># Q still has full multi-head structure</span>
    Q = x @ W_Q   <span class="com"># W_Q: [d_model, d_model] — full</span>

    <span class="com"># K and V have only ONE head's worth of dimension</span>
    K = x @ W_K   <span class="com"># W_K: [d_model, d_head] — shrunk!</span>
    V = x @ W_V   <span class="com"># W_V: [d_model, d_head] — shrunk!</span>

    <span class="com"># All Q heads attend to the same K, V</span>
    Q = Q.<span class="fn">view</span>(batch, seq, n_heads, d_head).<span class="fn">transpose</span>(<span class="num">1</span>, <span class="num">2</span>)
    K = K.<span class="fn">view</span>(batch, seq, <span class="num">1</span>, d_head).<span class="fn">transpose</span>(<span class="num">1</span>, <span class="num">2</span>)   <span class="com"># 1 head only</span>
    V = V.<span class="fn">view</span>(batch, seq, <span class="num">1</span>, d_head).<span class="fn">transpose</span>(<span class="num">1</span>, <span class="num">2</span>)   <span class="com"># 1 head only</span>

    <span class="com"># Broadcast: K, V get implicitly expanded for the n_heads dimension</span>
    scores = Q @ K.<span class="fn">transpose</span>(-<span class="num">2</span>, -<span class="num">1</span>) / math.<span class="fn">sqrt</span>(d_head)
    output = F.<span class="fn">softmax</span>(scores, dim=-<span class="num">1</span>) @ V
    <span class="kw">return</span> output</code></pre>

<p><strong>KV cache cost reduction</strong>: from <code>n_heads × 2 × d_head</code> per token to <code>1 × 2 × d_head</code> per token. For 32 heads, that's <strong>32× compression</strong>. Massive.</p>

<p><strong>The catch</strong>: with all queries forced to attend to the same K, V, the model loses the per-head specialization that gives MHA its expressive power. Empirical result: notable quality degradation, especially on tasks requiring fine-grained pattern matching.</p>

<p><strong>Production use</strong>: Falcon, PaLM, ChatGLM2 — popular in 2023-2024 when KV cache was the dominant memory pressure. Mostly displaced by GQA in 2024-2026 because GQA achieves most of the cache savings without as much quality loss.</p>

<h2>GQA — Grouped-Query Attention</h2>

<p>Ainslie et al. (2023). The pragmatic compromise: <strong>group query heads; each group shares one K, V head</strong>.</p>

<pre><code><span class="kw">def</span> <span class="fn">grouped_query_attention</span>(x, W_Q, W_K, W_V, W_O, n_heads, n_kv_heads):
    <span class="com"># n_heads = 32, n_kv_heads = 8 (typical) → 4 query heads per group</span>
    Q = x @ W_Q   <span class="com"># W_Q: [d_model, n_heads * d_head]</span>
    K = x @ W_K   <span class="com"># W_K: [d_model, n_kv_heads * d_head]</span>
    V = x @ W_V   <span class="com"># W_V: [d_model, n_kv_heads * d_head]</span>

    Q = Q.<span class="fn">view</span>(batch, seq, n_heads, d_head)
    K = K.<span class="fn">view</span>(batch, seq, n_kv_heads, d_head)
    V = V.<span class="fn">view</span>(batch, seq, n_kv_heads, d_head)

    <span class="com"># Each group of (n_heads / n_kv_heads) query heads attends to one KV head</span>
    K = K.<span class="fn">repeat_interleave</span>(n_heads // n_kv_heads, dim=<span class="num">2</span>)
    V = V.<span class="fn">repeat_interleave</span>(n_heads // n_kv_heads, dim=<span class="num">2</span>)

    <span class="com"># Now standard attention with shared K, V across groups</span>
    Q = Q.<span class="fn">transpose</span>(<span class="num">1</span>, <span class="num">2</span>)
    K = K.<span class="fn">transpose</span>(<span class="num">1</span>, <span class="num">2</span>)
    V = V.<span class="fn">transpose</span>(<span class="num">1</span>, <span class="num">2</span>)
    scores = Q @ K.<span class="fn">transpose</span>(-<span class="num">2</span>, -<span class="num">1</span>) / math.<span class="fn">sqrt</span>(d_head)
    output = F.<span class="fn">softmax</span>(scores, dim=-<span class="num">1</span>) @ V
    <span class="kw">return</span> output</code></pre>

<p><strong>KV cache cost reduction</strong>: from 32 heads' worth of K, V to 8 heads' worth. <strong>4× compression</strong> — less than MQA's 32× but with much better quality preservation.</p>

<p>The intuition: <em>not all heads need to specialize on different keys</em>. Grouping query heads that compute similar attention patterns is a reasonable approximation; the quality loss is small.</p>

<p><strong>Production use</strong>: GQA became the open-source default 2024-2026:</p>
<ul>
  <li><strong>Llama 2 70B</strong>: 32 query heads, 8 KV heads (4-per-group)</li>
  <li><strong>Llama 3 70B / Llama 4</strong>: same 8-group pattern</li>
  <li><strong>Mistral 7B</strong>: 32 query heads, 8 KV heads</li>
  <li><strong>Mistral Large, Mixtral</strong>: GQA throughout</li>
  <li><strong>Qwen 3.5/3.6 attention layers</strong> (the non-GDN layers): GQA</li>
  <li><strong>MiniMax M2.5</strong>: deliberately classic GQA, called out in M44</li>
</ul>

<p><em>If you're not using MLA or hybrid attention, you're using GQA</em>. It's the safe, well-tooled, FlashAttention-compatible choice that maximizes quality given a KV-cache budget.</p>

<h2>MLA — Multi-Head Latent Attention</h2>

<p>DeepSeek-AI 2024 (introduced in V2). The fundamental rethink: <em>instead of reducing the number of KV heads, compress the full-dimensional K and V into a low-rank latent space, cache that, and decompress on demand</em>.</p>

<h3>The basic mechanism</h3>

<p>Rather than projecting input <code>x → K, V</code> directly, MLA factorizes the projections:</p>

<pre><code><span class="com"># MHA standard: x → K, V directly</span>
K = x @ W_K   <span class="com"># [d_model, d_kv = n_heads * d_head]</span>
V = x @ W_V   <span class="com"># [d_model, d_kv]</span>

<span class="com"># MLA: x → c_KV (latent) → K, V via up-projection</span>
c_KV = x @ W_DKV   <span class="com"># W_DKV: [d_model, r_kv] — DOWN projection</span>
K = c_KV @ W_UK    <span class="com"># W_UK: [r_kv, d_kv] — UP projection</span>
V = c_KV @ W_UV    <span class="com"># W_UV: [r_kv, d_kv] — UP projection</span>

<span class="com"># Cache c_KV (small) instead of K, V (large)</span>
<span class="com"># r_kv ≈ 4 * d_head; in DeepSeek V2/V3, r_kv = 512 (vs full d_kv = 16384 for 128 heads)</span></code></pre>

<p>The KV cache stores only <code>c_KV</code> — typically dimension r_kv = 512 — instead of full K and V at dimension d_kv = 4096-16384. <strong>~32× to 64× cache reduction</strong> compared to MHA, with quality matching MHA at scale.</p>

<p>But there's a problem.</p>

<h3>The RoPE paradox</h3>

<p>RoPE (M30) rotates Q and K by position-dependent angles before the dot-product:</p>

<p><code>attention_score(q_i, k_j) = (R(i) · q_i) · (R(j) · k_j)</code></p>

<p>This rotation must be applied to <em>full-dimension K</em> — there's no way to compose the rotation cleanly with a low-rank decomposition. If you cache c_KV and reconstruct K = c_KV @ W_UK, then apply RoPE, the RoPE rotation interferes with the up-projection in a way that prevents the algebraic absorption tricks that make MLA fast.</p>

<p>More fundamentally: <em>RoPE rotates by position, but c_KV is the same vector regardless of position</em>. Position information has to enter somewhere; if c_KV is purely position-free, where does it come in?</p>

<h3>The decoupled RoPE solution</h3>

<p>DeepSeek's elegant fix: <strong>split each query and key into two parts — one with RoPE applied, one without</strong>.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 420" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">MLA's decoupled RoPE: split position-aware vs position-free components</text>

  <!-- Input -->
  <g transform="translate(20, 50)">
    <rect x="0" y="0" width="100" height="40" fill="#fff8a8" stroke="#1a1612" stroke-width="1.5"/>
    <text x="50" y="18" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Input x_i</text>
    <text x="50" y="32" text-anchor="middle" font-size="9" fill="#1a1612">[d_model]</text>
  </g>

  <!-- Query path -->
  <g transform="translate(140, 45)">
    <text x="0" y="14" font-size="11" font-weight="700" fill="#1a1612">Query path</text>

    <rect x="0" y="20" width="280" height="50" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="10" y="38" font-size="9" font-weight="700" fill="#1a1612">q_nope (no RoPE):</text>
    <text x="10" y="52" font-size="9" fill="#1a1612">  q_nope = x_i @ W_dq @ W_uq[h]</text>
    <text x="10" y="64" font-size="9" fill="#1a1612">  Position-free; absorbed via low-rank</text>

    <rect x="0" y="80" width="280" height="50" fill="#fff5d8" stroke="#d4a017" stroke-width="1.5"/>
    <text x="10" y="98" font-size="9" font-weight="700" fill="#1a1612">q_rope (RoPE applied):</text>
    <text x="10" y="112" font-size="9" fill="#1a1612">  q_rope = RoPE(x_i @ W_qr[h])</text>
    <text x="10" y="124" font-size="9" fill="#1a1612">  Small dimension d_qk_rope ≈ d_head/2</text>
  </g>

  <!-- Key path -->
  <g transform="translate(440, 45)">
    <text x="0" y="14" font-size="11" font-weight="700" fill="#1a1612">Key path</text>

    <rect x="0" y="20" width="280" height="50" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="10" y="38" font-size="9" font-weight="700" fill="#1a1612">k_nope (no RoPE):</text>
    <text x="10" y="52" font-size="9" fill="#1a1612">  c_KV = x_i @ W_dkv  ← cache this</text>
    <text x="10" y="64" font-size="9" fill="#1a1612">  k_nope = c_KV @ W_uk[h] (per-head)</text>

    <rect x="0" y="80" width="280" height="50" fill="#fff5d8" stroke="#d4a017" stroke-width="1.5"/>
    <text x="10" y="98" font-size="9" font-weight="700" fill="#1a1612">k_rope (RoPE applied, SHARED):</text>
    <text x="10" y="112" font-size="9" fill="#1a1612">  k_rope = RoPE(x_i @ W_kr) ← cache this</text>
    <text x="10" y="124" font-size="9" fill="#1a1612">  Single shared head; small dimension</text>
  </g>

  <!-- Combination -->
  <g transform="translate(20, 195)">
    <rect x="0" y="0" width="700" height="60" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="350" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">Attention computation: concatenate, then dot-product</text>
    <text x="350" y="42" text-anchor="middle" font-size="11" font-family="monospace" fill="#1a1612">score = [q_nope; q_rope] · [k_nope; k_rope] = q_nope · k_nope + q_rope · k_rope</text>
  </g>

  <!-- Cache -->
  <g transform="translate(20, 275)">
    <rect x="0" y="0" width="340" height="130" fill="#fff8a8" stroke="#1a1612" stroke-width="2"/>
    <text x="170" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">What gets cached</text>
    <text x="20" y="46" font-size="10" font-weight="700" fill="#1a1612">  c_KV (per token):</text>
    <text x="20" y="60" font-size="10" fill="#1a1612">  shape [r_kv] — typically 512</text>
    <text x="20" y="80" font-size="10" font-weight="700" fill="#1a1612">  k_rope (per token, shared):</text>
    <text x="20" y="94" font-size="10" fill="#1a1612">  shape [d_qk_rope] — typically 64</text>
    <text x="20" y="116" font-size="10" font-weight="700" fill="#c1502e">  Total per token: 576 dims</text>
  </g>

  <g transform="translate(380, 275)">
    <rect x="0" y="0" width="340" height="130" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="170" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">Comparison: vs MHA at 128 heads, d_head=128</text>
    <text x="20" y="46" font-size="10" font-weight="700" fill="#1a1612">  MHA per token:</text>
    <text x="20" y="60" font-size="10" fill="#1a1612">  128 heads × 2 × 128 = 32,768 dims</text>
    <text x="20" y="80" font-size="10" font-weight="700" fill="#1a1612">  MLA per token:</text>
    <text x="20" y="94" font-size="10" fill="#1a1612">  576 dims</text>
    <text x="20" y="116" font-size="10" font-weight="700" fill="#1f5f5b">  Compression ratio: ~57× — &gt; 6× over GQA-8</text>
  </g>
</svg>
</div>

<p>The MLA forward pass with decoupled RoPE:</p>

<pre><code><span class="kw">def</span> <span class="fn">mla_forward</span>(x, W_dq, W_uq, W_dkv, W_uk, W_uv, W_qr, W_kr, n_heads):
    <span class="com"># DOWN-project x to compressed query and KV latents</span>
    c_q = x @ W_dq                 <span class="com"># [batch, seq, r_q]</span>
    c_KV = x @ W_dkv               <span class="com"># [batch, seq, r_kv]  ← CACHE THIS</span>

    <span class="com"># UP-project to per-head queries (position-free part)</span>
    q_nope = c_q @ W_uq            <span class="com"># [batch, seq, n_heads, d_qk_nope]</span>

    <span class="com"># Compute RoPE-aware query parts (per-head)</span>
    q_rope = x @ W_qr              <span class="com"># [batch, seq, n_heads, d_qk_rope]</span>
    q_rope = <span class="fn">apply_rope</span>(q_rope)

    <span class="com"># UP-project to per-head keys (position-free part)</span>
    k_nope = c_KV @ W_uk           <span class="com"># [batch, seq, n_heads, d_qk_nope]</span>

    <span class="com"># Compute RoPE-aware key part (SHARED across heads)</span>
    k_rope = x @ W_kr              <span class="com"># [batch, seq, d_qk_rope]  ← CACHE THIS (single head)</span>
    k_rope = <span class="fn">apply_rope</span>(k_rope)
    k_rope = k_rope.<span class="fn">unsqueeze</span>(-<span class="num">2</span>).<span class="fn">expand</span>(-<span class="num">1</span>, -<span class="num">1</span>, n_heads, -<span class="num">1</span>)   <span class="com"># broadcast</span>

    <span class="com"># UP-project values from latent</span>
    v = c_KV @ W_uv                <span class="com"># [batch, seq, n_heads, d_v]</span>

    <span class="com"># Concatenate q parts and k parts</span>
    Q = torch.<span class="fn">cat</span>([q_nope, q_rope], dim=-<span class="num">1</span>)   <span class="com"># [B, S, H, d_qk_nope + d_qk_rope]</span>
    K = torch.<span class="fn">cat</span>([k_nope, k_rope], dim=-<span class="num">1</span>)   <span class="com"># [B, S, H, d_qk_nope + d_qk_rope]</span>

    <span class="com"># Standard attention</span>
    scores = Q @ K.<span class="fn">transpose</span>(-<span class="num">2</span>, -<span class="num">1</span>) / math.<span class="fn">sqrt</span>(Q.<span class="fn">size</span>(-<span class="num">1</span>))
    output = F.<span class="fn">softmax</span>(scores, dim=-<span class="num">1</span>) @ v
    <span class="kw">return</span> output</code></pre>

<p>The cached items are <em>only</em> c_KV (compressed) and k_rope (small, shared across heads). At inference, both can be loaded and the up-projections re-run on the fly.</p>

<h3>Weight absorption: the inference trick</h3>

<p>MLA admits a further inference optimization: the up-projections W_UK and W_UV can be <em>algebraically absorbed</em> into adjacent matrices. The query projection and output projection get composed with W_UK and W_UV respectively; the actual inference path computes attention scores directly from c_KV without ever materializing K, V at full dimension.</p>

<pre><code><span class="com"># Without absorption: cache c_KV, decompress K, V, compute attention</span>
<span class="com"># With absorption: precompute W_q_absorbed = W_uq @ W_uk^T</span>
<span class="com"># Now the query "lives" in the latent c_KV space; attention computed directly there</span>

W_q_absorbed = W_uq @ W_uk.T   <span class="com"># Composed offline</span>
score_nope = (c_q @ W_q_absorbed) @ c_KV.T
<span class="com"># Now never materializes K_nope at full dimension</span></code></pre>

<p>The result: MLA at inference behaves <em>like MQA</em> in cache footprint (single small cache per token) but with <em>per-head specialization</em> preserved through the query and output paths. This is the key economic insight — MQA-like deployment cost, MHA-like quality.</p>

<h3>MLA quality results</h3>

<p>From DeepSeek V2 ablations:</p>

<ul>
  <li><strong>MLA matches or exceeds MHA</strong> on most benchmarks at 128B+ scale.</li>
  <li><strong>GQA underperforms MHA</strong> in the same ablations.</li>
  <li><strong>The compression-decompression step doesn't degrade information enough to matter</strong>; in some cases it acts as regularization.</li>
</ul>

<p>The caveat (from Sebastian Raschka and others): <strong>MLA only seems to win at 100B+ scale</strong>. At smaller scales (under 100B), GQA is easier to tune and at least as good. <em>MLA is for trillion-parameter territory; GQA covers everything below</em>.</p>

<p><strong>Production use</strong>: DeepSeek V2 (236B), DeepSeek V3 (671B), DeepSeek R1, DeepSeek V3.2, DeepSeek V4 (1.6T). Sarvam 105B uses MLA; Sarvam 30B uses GQA. Kimi Linear's full-attention layers use MLA (they replaced Qwen3-Next's gated attention with MLA specifically for the non-DGN layers). Ling 2.5 (1T MoE) uses MLA + Lightning Attention hybrid.</p>

<h2>IHA — Interleaved Head Attention (Feb 2026)</h2>

<p>Duvvuri et al. (Meta, UT Austin, Berkeley, Harvard, MIT). The newest entry on Axis C — and one that breaks a fundamental MHA limitation.</p>

<h3>The compositional bottleneck</h3>

<p>Standard MHA has H heads, producing H independent attention matrices. <em>No information flows between heads during attention computation</em>. After attention, the head outputs are combined via the output projection, but during attention itself, each head is isolated.</p>

<p>This becomes a problem for multi-step reasoning. Tasks requiring evidence aggregation across multiple parts of the context, with multiple intermediate transformations, need <em>composition</em> of attention patterns. MHA can only represent k distinct patterns within one layer using O(k) heads — linear scaling with task complexity.</p>

<h3>The IHA mechanism</h3>

<p>IHA introduces <strong>pseudo-heads</strong>: for each of H attention heads, construct P pseudo-heads (typically P = H) as learned linear combinations of <em>all H original heads' projections</em>.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 380" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">IHA — pseudo-head construction enables cross-head mixing</text>

  <!-- Original heads -->
  <g transform="translate(20, 50)">
    <text x="0" y="14" font-size="11" font-weight="700" fill="#1a1612">Step 1: Standard Q, K, V projections (H = 4 heads in this example)</text>

    <g transform="translate(0, 25)">
      <rect x="0" y="0" width="60" height="30" fill="#fcecec" stroke="#c1502e" stroke-width="1"/>
      <text x="30" y="20" text-anchor="middle" font-size="10" fill="#1a1612">Q_1</text>

      <rect x="65" y="0" width="60" height="30" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/>
      <text x="95" y="20" text-anchor="middle" font-size="10" fill="#1a1612">Q_2</text>

      <rect x="130" y="0" width="60" height="30" fill="#fff5d8" stroke="#d4a017" stroke-width="1"/>
      <text x="160" y="20" text-anchor="middle" font-size="10" fill="#1a1612">Q_3</text>

      <rect x="195" y="0" width="60" height="30" fill="#fff8a8" stroke="#1a1612" stroke-width="1"/>
      <text x="225" y="20" text-anchor="middle" font-size="10" fill="#1a1612">Q_4</text>

      <text x="270" y="20" font-size="10" fill="#1a1612">(same for K, V)</text>
    </g>
  </g>

  <!-- Pseudo-head construction -->
  <g transform="translate(20, 115)">
    <text x="0" y="14" font-size="11" font-weight="700" fill="#1a1612">Step 2: Each head is mixed into P=4 pseudo-heads (linear combination of all H originals)</text>

    <g transform="translate(0, 25)">
      <text x="0" y="14" font-size="9" fill="#1a1612">Head 1 → 4 pseudos:</text>
      <rect x="120" y="0" width="40" height="20" fill="#ffd5dc" stroke="#b85a6c" stroke-width="0.5"/>
      <text x="140" y="14" text-anchor="middle" font-size="9" fill="#1a1612">P1.1</text>
      <rect x="165" y="0" width="40" height="20" fill="#ffd5dc" stroke="#b85a6c" stroke-width="0.5"/>
      <text x="185" y="14" text-anchor="middle" font-size="9" fill="#1a1612">P1.2</text>
      <rect x="210" y="0" width="40" height="20" fill="#ffd5dc" stroke="#b85a6c" stroke-width="0.5"/>
      <text x="230" y="14" text-anchor="middle" font-size="9" fill="#1a1612">P1.3</text>
      <rect x="255" y="0" width="40" height="20" fill="#ffd5dc" stroke="#b85a6c" stroke-width="0.5"/>
      <text x="275" y="14" text-anchor="middle" font-size="9" fill="#1a1612">P1.4</text>
      <text x="305" y="14" font-size="9" fill="#1a1612">Each = α₁₁Q₁ + α₁₂Q₂ + α₁₃Q₃ + α₁₄Q₄</text>
    </g>

    <text x="0" y="44" font-size="9" fill="#1a1612">Head 2 → 4 pseudos: (similar mixing with α₂)</text>
    <text x="0" y="58" font-size="9" fill="#1a1612">Head 3 → 4 pseudos: (α₃)</text>
    <text x="0" y="72" font-size="9" fill="#1a1612">Head 4 → 4 pseudos: (α₄)</text>
  </g>

  <!-- Interleave -->
  <g transform="translate(20, 215)">
    <text x="0" y="14" font-size="11" font-weight="700" fill="#1a1612">Step 3: Interleave pseudo-heads along sequence dimension (sequence appears P× longer)</text>

    <g transform="translate(0, 25)">
      <rect x="0" y="0" width="700" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
      <text x="350" y="18" text-anchor="middle" font-size="10" fill="#1a1612">Original tokens × P pseudos = N × P tokens (sequence appears 4× longer)</text>
      <text x="350" y="34" text-anchor="middle" font-size="9" font-style="italic" fill="#1a1612">Causal mask must be defined over the expanded sequence length</text>
    </g>
  </g>

  <!-- Final attention -->
  <g transform="translate(20, 285)">
    <text x="0" y="14" font-size="11" font-weight="700" fill="#1a1612">Step 4: Apply standard attention; combine pseudo-heads back into single head per original</text>

    <g transform="translate(0, 25)">
      <rect x="0" y="0" width="700" height="60" fill="#fff5d8" stroke="#d4a017" stroke-width="2"/>
      <text x="350" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">Result: P² = 16 attention patterns per head with O(H²P) parameter overhead</text>
      <text x="350" y="44" text-anchor="middle" font-size="10" fill="#1a1612">FlashAttention-compatible (mixing happens BEFORE attention, attention itself is standard)</text>
    </g>
  </g>
</svg>
</div>

<p>The IHA forward pass:</p>

<pre><code><span class="kw">def</span> <span class="fn">iha_forward</span>(x, W_Q, W_K, W_V, W_O, alpha_q, alpha_k, alpha_v, n_heads, P):
    <span class="com"># Step 1: Standard Q, K, V — same as MHA</span>
    Q = x @ W_Q   <span class="com"># [B, N, H, d_head]</span>
    K = x @ W_K
    V = x @ W_V

    <span class="com"># Step 2: Construct P pseudo-heads per head via learned linear combinations</span>
    <span class="com"># alpha_q: [H, H, P] — for each head h, P pseudos as combos of all H originals</span>
    Q_pseudo = torch.<span class="fn">einsum</span>(<span class="str">"bnhd,hkp-&gt;bnhpd"</span>, Q, alpha_q)
    K_pseudo = torch.<span class="fn">einsum</span>(<span class="str">"bnhd,hkp-&gt;bnhpd"</span>, K, alpha_k)
    V_pseudo = torch.<span class="fn">einsum</span>(<span class="str">"bnhd,hkp-&gt;bnhpd"</span>, V, alpha_v)
    <span class="com"># Each is now [B, N, H, P, d_head]</span>

    <span class="com"># Step 3: Interleave pseudo-heads along sequence dim — sequence appears N*P long</span>
    Q_inter = Q_pseudo.<span class="fn">flatten</span>(start_dim=<span class="num">1</span>, end_dim=<span class="num">2</span>)   <span class="com"># Treat (N, P) as one extended seq</span>
    K_inter = K_pseudo.<span class="fn">flatten</span>(start_dim=<span class="num">1</span>, end_dim=<span class="num">2</span>)
    V_inter = V_pseudo.<span class="fn">flatten</span>(start_dim=<span class="num">1</span>, end_dim=<span class="num">2</span>)
    <span class="com"># Each is now [B, N*P, H, d_head]</span>

    <span class="com"># Step 4: Standard attention with expanded causal mask</span>
    <span class="com"># Causal mask is over N*P, not N — every Pth position is the original token</span>
    expanded_mask = <span class="fn">build_iha_causal_mask</span>(N, P)
    output_pseudo = <span class="fn">flash_attention</span>(Q_inter, K_inter, V_inter, mask=expanded_mask)

    <span class="com"># Step 5: Combine pseudo-heads back into a single head via output projection R</span>
    output = output_pseudo.<span class="fn">view</span>(B, N, P, n_heads, d_head)
    output = (output * R).<span class="fn">sum</span>(dim=<span class="num">2</span>)   <span class="com"># [B, N, H, d_head]</span>
    output = output.<span class="fn">flatten</span>(<span class="num">2</span>) @ W_O
    <span class="kw">return</span> output</code></pre>

<p><strong>Why it's strictly more expressive</strong>: with P pseudo-heads per head, each original head can express up to P² distinct attention patterns (P pseudo-queries × P pseudo-keys). MHA can express only H patterns total per layer. IHA at P=H gets H × H² = H³ effective patterns per layer with O(H²P) = O(H³) parameter overhead — substantial expressive gain.</p>

<p><strong>Theoretical results from the paper</strong>: on synthetic Polynomial Filter tasks, IHA needs Θ(√k × n²) parameters where MHA needs Θ(k × n²) — quadratic to linear improvement in k (chain length). On synthetic CPM-3 (order-sensitive composition), IHA uses ⌈√N_max⌉ heads vs MHA's N_max — square-root reduction.</p>

<p><strong>Empirical results</strong>:</p>
<ul>
  <li>RULER Multi-Key Retrieval: <strong>10-20% improvement</strong> over full attention at 4K-16K context.</li>
  <li>GSM8K (after OpenThoughts fine-tuning): <strong>+5.8% improvement</strong> Maj@16 over full attention (54.2% vs 48.4%).</li>
  <li>MATH-500: <strong>+2.8% improvement</strong> Maj@16 over full attention (18.4% vs 15.6%).</li>
  <li>MBPP coding: Talking-Heads slightly ahead, IHA second.</li>
</ul>

<p><strong>FlashAttention compatibility</strong>: this is the key practical win. Unlike Talking-Heads attention (Shazeer 2020) which mixes heads at the level of attention logits — incompatible with FlashAttention's softmax-internal computation — IHA mixes heads <em>before</em> the attention operator. The actual attention computation is standard; FlashAttention runs unmodified.</p>

<p><strong>Production status</strong>: as of April 2026, IHA is published research (Feb 2026, Meta + UT Austin + UC Berkeley + Harvard + MIT) without major production adoption yet. Expect adoption in 2026-2027 frontier models given the FlashAttention compatibility and substantial RULER/reasoning gains.</p>

<h2>Slim Attention — when MHA still appears</h2>

<p>Graef &amp; Wasielewski (March 2025). A clever optimization specifically for models that still use MHA (rather than GQA/MQA/MLA) — primarily encoder-decoder models like Whisper, T5, and some older LLMs.</p>

<h3>The mathematical trick</h3>

<p>Standard MHA caches both K and V — total cache width is <code>n_heads × (d_K + d_V)</code> per token. Slim Attention's observation: <strong>when projection matrices are square (d_model = d_K = d_V), V can be exactly computed from K via</strong>:</p>

<pre><code><span class="com"># Standard MHA: V = X @ W_V, K = X @ W_K</span>
<span class="com"># If W_K is invertible: X = K @ W_K_inverse</span>
<span class="com"># Therefore: V = K @ W_K_inverse @ W_V = K @ W_KV</span>

<span class="com"># where W_KV = W_K_inverse @ W_V is precomputed offline</span>

<span class="kw">def</span> <span class="fn">slim_attention</span>(x, W_Q, W_K, W_KV_combined, W_O, n_heads):
    Q = x @ W_Q
    K = x @ W_K   <span class="com"># Cache only K!</span>

    <span class="com"># Compute V from K on demand using precomputed W_KV</span>
    V = K @ W_KV_combined   <span class="com"># V never stored — recomputed per attention call</span>

    <span class="com"># Standard attention from here</span>
    <span class="kw">return</span> <span class="fn">attention</span>(Q, K, V)</code></pre>

<p><strong>Cache reduction</strong>: from K + V to K only. <strong>Lossless 2× compression</strong> — mathematically identical to standard attention, no quality loss.</p>

<h3>Practical considerations</h3>

<p>The trick has three subtleties:</p>

<ol>
  <li><strong>Requires square projection matrices</strong>. d_model must equal n_heads × d_K. This is true for vanilla MHA but not for GQA/MQA (where K projection is smaller) or MLA (where K projection is via low-rank latent).</li>
  <li><strong>RoPE incompatibility (sort of)</strong>. If RoPE is applied to K between projection and dot-product, then cached K is post-RoPE, and <code>V = K @ W_KV</code> no longer holds because W_KV was derived assuming pre-RoPE K. Workaround: cache pre-RoPE K, apply RoPE on demand. Or use position-encoding schemes that don't rotate features (Alibi, T5's relative PE, FIRE) which fully support Slim Attention.</li>
  <li><strong>Bias terms</strong>: most modern transformers (PaLM-style) drop biases from projections. For models with biases (Whisper, older models), the paper shows how to fold biases into adjacent layers in a mathematically equivalent way.</li>
</ol>

<h3>Where the savings get bigger</h3>

<p>For <strong>encoder-decoder transformers</strong> like Whisper and T5, the cache memory reduction is substantially larger than 2×:</p>

<ul>
  <li><strong>Whisper</strong>: 8× cache reduction (encoder-decoder cross-attention has different math).</li>
  <li><strong>T5-11B</strong>: <strong>32× cache reduction</strong> because its MHA projection dimension exceeds embedding dimension.</li>
  <li>Inference speedup: up to <strong>5× for token generation at batch size 64</strong> on Whisper.</li>
</ul>

<p>The compression factor is <code>c = d_cache / d_model</code> where <code>d_cache = h_KV × (d_K + d_V)</code>. Larger c → larger savings. Typical c is 2 (Slim → 2× savings); models with non-square projections can have c much larger.</p>

<p><strong>Production use</strong>: speech-to-text systems (Whisper deployment), translation models (T5 derivatives), time-series forecasting (Amazon Chronos models). For modern LLMs using GQA/MLA, Slim Attention doesn't directly apply — but <em>its mathematical idea</em> (reconstruct one cached tensor from another) shows up in MLA's weight absorption.</p>

<h2>The decision tree: which head topology when?</h2>

<p>Putting Axis C choices together for a 2026 production engineering decision:</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 480" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrM46" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Head topology decision tree, 2026 production</text>

  <!-- Root -->
  <g transform="translate(280, 50)">
    <rect x="0" y="0" width="180" height="40" fill="#fff5d8" stroke="#d4a017" stroke-width="2"/>
    <text x="90" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">Pretraining new model?</text>
    <text x="90" y="32" text-anchor="middle" font-size="9" fill="#1a1612">or fine-tuning existing?</text>
  </g>

  <!-- Pretraining branch -->
  <g transform="translate(40, 130)">
    <rect x="0" y="0" width="160" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="80" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">Pretraining: scale &gt;100B?</text>
  </g>

  <line x1="280" y1="90" x2="120" y2="130" stroke="#1f5f5b" stroke-width="1.5" marker-end="url(#arrM46)"/>
  <text x="180" y="115" font-size="9" fill="#1a1612">pretraining</text>

  <!-- &gt;100B → MLA -->
  <g transform="translate(20, 220)">
    <rect x="0" y="0" width="200" height="60" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="100" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">MLA</text>
    <text x="100" y="38" text-anchor="middle" font-size="9" fill="#1a1612">DeepSeek lineage</text>
    <text x="100" y="52" text-anchor="middle" font-size="9" fill="#1a1612">5-7× cache vs GQA</text>
  </g>

  <line x1="120" y1="170" x2="120" y2="220" stroke="#1f5f5b" stroke-width="1.5" marker-end="url(#arrM46)"/>
  <text x="125" y="200" font-size="9" fill="#1a1612">yes &gt;100B</text>

  <!-- &lt;100B → GQA -->
  <g transform="translate(240, 220)">
    <rect x="0" y="0" width="200" height="60" fill="#fff5d8" stroke="#d4a017" stroke-width="2"/>
    <text x="100" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">GQA (8 KV heads)</text>
    <text x="100" y="38" text-anchor="middle" font-size="9" fill="#1a1612">Llama, Mistral, Qwen</text>
    <text x="100" y="52" text-anchor="middle" font-size="9" fill="#1a1612">Default for &lt;100B</text>
  </g>

  <line x1="200" y1="170" x2="320" y2="220" stroke="#1f5f5b" stroke-width="1.5" marker-end="url(#arrM46)"/>
  <text x="240" y="200" font-size="9" fill="#1a1612">no &lt;100B</text>

  <!-- IHA branch -->
  <g transform="translate(460, 220)">
    <rect x="0" y="0" width="200" height="60" fill="#ffd5dc" stroke="#b85a6c" stroke-width="2"/>
    <text x="100" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">IHA (experimental)</text>
    <text x="100" y="38" text-anchor="middle" font-size="9" fill="#1a1612">If reasoning is target</text>
    <text x="100" y="52" text-anchor="middle" font-size="9" fill="#1a1612">+5.8% GSM8K</text>
  </g>

  <line x1="200" y1="170" x2="540" y2="220" stroke="#1f5f5b" stroke-width="1.5" marker-end="url(#arrM46)"/>
  <text x="380" y="195" font-size="9" fill="#1a1612">reasoning-focused</text>

  <!-- Fine-tuning branch -->
  <g transform="translate(540, 130)">
    <rect x="0" y="0" width="160" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="80" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">Use existing model</text>
    <text x="80" y="32" text-anchor="middle" font-size="9" fill="#1a1612">topology fixed</text>
  </g>

  <line x1="460" y1="90" x2="620" y2="130" stroke="#1f5f5b" stroke-width="1.5" marker-end="url(#arrM46)"/>
  <text x="540" y="115" font-size="9" fill="#1a1612">fine-tuning</text>

  <!-- Fine-tuning leaf: cache compression -->
  <g transform="translate(440, 320)">
    <rect x="0" y="0" width="280" height="80" fill="#fff8a8" stroke="#1a1612" stroke-width="2"/>
    <text x="140" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">Apply KV-cache engineering</text>
    <text x="20" y="42" font-size="10" fill="#1a1612">  • <tspan font-weight="700">TurboQuant</tspan> 3-bit: 4× capacity (vLLM)</text>
    <text x="20" y="56" font-size="10" fill="#1a1612">  • <tspan font-weight="700">Expected Attention</tspan>: KV eviction</text>
    <text x="20" y="70" font-size="10" fill="#1a1612">  • <tspan font-weight="700">InfiniGen / Quest</tspan>: KV offloading</text>
  </g>

  <line x1="620" y1="170" x2="600" y2="320" stroke="#1f5f5b" stroke-width="1.5" marker-end="url(#arrM46)"/>

  <!-- Slim attention edge -->
  <g transform="translate(20, 320)">
    <rect x="0" y="0" width="280" height="80" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="140" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">Edge case: encoder-decoder MHA</text>
    <text x="20" y="42" font-size="10" fill="#1a1612">  • Whisper, T5, Chronos</text>
    <text x="20" y="58" font-size="10" fill="#1a1612">  • Apply <tspan font-weight="700">Slim Attention</tspan> for 2-32× cache</text>
    <text x="20" y="72" font-size="10" fill="#1a1612">  • Lossless; no retraining needed</text>
  </g>

  <!-- Bottom unification -->
  <g transform="translate(20, 420)">
    <rect x="0" y="0" width="700" height="50" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="350" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">Production reality: 90% of teams use GQA + TurboQuant 3-bit; 10% use MLA + DeepGEMM</text>
    <text x="350" y="40" text-anchor="middle" font-size="10" font-style="italic" fill="#1a1612">Both work; choice depends on scale and where you sit on the open-weight ecosystem map</text>
  </g>
</svg>
</div>

<h2>KV-cache engineering: compounding with head topology</h2>

<p>Once head topology is chosen, KV-cache engineering compresses what's left. April 2026 production stack:</p>

<h3>TurboQuant — the new vLLM default for 2-bit/3-bit KV quantization</h3>

<p>ICLR 2026 (Rotem et al.). Merged into vLLM April 15, 2026 (PR #38479). The flagship 2026 KV-cache compression:</p>

<ul>
  <li><strong>Online vector quantization</strong> — no calibration, no fine-tuning, no model-specific tuning</li>
  <li><strong>PolarQuant transform</strong>: random rotations (Walsh-Hadamard transforms) to make distributions more amenable to scalar quantization, then Lloyd-Max scalar quantization</li>
  <li><strong>Asymmetric bit allocation</strong>: 3 bits for keys, 2 bits for values (values tolerate compression better)</li>
  <li><strong>4× capacity vs FP16 KV cache</strong> (3-bit configuration); higher with mixed-mode quantization</li>
  <li><strong>Triton kernels + CUDA fallback</strong> for on-the-fly dequantization during attention</li>
</ul>

<p>vLLM usage:</p>

<pre><code><span class="com"># Available in vLLM after April 15, 2026 (PR #38479 merged)</span>
vllm serve Qwen/Qwen3-4B --kv-cache-dtype turboquant_3bit_nc

<span class="com"># Available cache types:</span>
<span class="com">#   turboquant_3bit_nc  — 3-bit no calibration (default recommendation)</span>
<span class="com">#   k8v4                — 8-bit keys, 4-bit values (asymmetric)</span>
<span class="com">#   4bit_nc             — 4-bit no calibration</span>
<span class="com">#   k3v4_nc             — 3-bit keys, 4-bit values</span></code></pre>

<p>The trade: <strong>2.6-4.9× KV capacity at the cost of 35-43% of decode throughput</strong>. Worth it when KV is the memory bottleneck (long context, high concurrency); not worth it when throughput is the bottleneck. The honest reading: <em>it's a substantial win for long-context serving but isn't free</em>.</p>

<p>Composition with head topology: TurboQuant 3-bit on top of GQA-8 gives ~4× × 4× = 16× total cache reduction over MHA-FP16. On top of MLA, it gives ~4× × 6× = 24×.</p>

<h3>KV cache eviction</h3>

<p>Don't compress; <em>discard</em>. Estimate which tokens are unlikely to be attended to in the future, evict them.</p>

<ul>
  <li><strong>Quest (Tang et al. 2024)</strong>: query-aware on-demand KV fetching — keep all KV in CPU, prefetch to GPU based on query similarity. Lazy eviction.</li>
  <li><strong>H2O (Heavy-Hitter Oracle)</strong>: keep tokens with high accumulated attention scores; evict the rest. Heuristic but effective.</li>
  <li><strong>Expected Attention</strong>: estimate the future query distribution, evict keys with low expected future attention. More principled than H2O.</li>
  <li><strong>SnapKV</strong>: cluster KV by attention pattern; keep cluster representatives.</li>
</ul>

<p>Tradeoff: eviction is <em>lossy</em>. If you evict a key that turns out to be needed later, the model performance degrades. Quality-cost tradeoff that depends on the specific eviction policy and workload.</p>

<h3>KV cache offloading</h3>

<p>Move KV cache from GPU memory to CPU memory (or even disk), prefetch back when needed.</p>

<ul>
  <li><strong>InfiniGen (Lee et al. 2024)</strong>: lightweight rehearsal using partial model weights to predict which KV blocks are needed; prefetch to overlap with computation. Offline SVD for efficient prediction.</li>
  <li><strong>SlimInfer</strong>: layer-wise hidden-state pruning naturally enables predictor-free prefetching.</li>
  <li><strong>FlexGen</strong>: extreme offloading for long-context inference on resource-constrained hardware.</li>
</ul>

<p>Tradeoff: offloading is <em>lossless</em> (everything's preserved) but PCIe bandwidth becomes the bottleneck. Useful when GPU memory is the constraint and you have time/bandwidth to swap.</p>

<h3>The full 2026 inference stack</h3>

<p>A typical April 2026 long-context inference deployment combines:</p>

<ol>
  <li><strong>Head topology</strong> (chosen at training time): GQA-8 or MLA. Determines baseline cache size.</li>
  <li><strong>FlashAttention 3 / 4</strong> (M47 territory): never materialize the full attention matrix. Determines compute kernel.</li>
  <li><strong>PagedAttention</strong> (vLLM): block-based KV cache layout. Eliminates fragmentation.</li>
  <li><strong>TurboQuant 3-bit</strong>: quantize the KV cache values. 4× capacity.</li>
  <li><strong>Expected Attention or H2O</strong> (optional): evict low-attention tokens for very long contexts.</li>
  <li><strong>InfiniGen</strong> (optional): offload cold KV pages to CPU when GPU memory is exhausted.</li>
</ol>

<p>At 1M tokens with a Llama 3 70B (GQA-8), the resulting cache is ~6.7 GB instead of the naive 320 GB. Concrete production numbers from the TurboQuant deployment ecosystem.</p>

<h2>Production zoo: Axis C in April 2026 frontier models</h2>

<p>Cross-referencing M44's frontier zoo with the Axis C choices covered in this module:</p>

<div class="table-wrap">
<table>
<caption>Axis C choices across April 2026 frontier models</caption>
<thead><tr><th>Model</th><th>Head topology</th><th>KV cache mechanism</th></tr></thead>
<tbody>
<tr><td><strong>Llama 4 Scout / Maverick</strong></td><td>GQA</td><td>FA3/FA4 + PagedAttention</td></tr>
<tr><td><strong>DeepSeek V4-Pro / V4-Flash</strong></td><td>MLA + CSA+HCA hybrid (M44)</td><td>DeepGEMM with FP4 Lightning Indexer</td></tr>
<tr><td><strong>GLM-5.1</strong></td><td>MLA-style with DSA</td><td>FA4 + custom kernels</td></tr>
<tr><td><strong>Qwen 3.5 / 3.6</strong></td><td>GQA in attention layers; GDN in linear layers</td><td>FA3 + standard PagedAttention</td></tr>
<tr><td><strong>Kimi Linear / K2.6</strong></td><td>MLA in full-attention layers; KDA in linear layers</td><td>FA4 + custom hybrid kernel</td></tr>
<tr><td><strong>Ling 2.5</strong></td><td>MLA + Lightning Attention hybrid</td><td>FA4 + custom hybrid kernel</td></tr>
<tr><td><strong>Mistral Large</strong></td><td>GQA</td><td>FA3 + PagedAttention</td></tr>
<tr><td><strong>MiniMax M2.5</strong></td><td>GQA (deliberately classic)</td><td>FA3 + PagedAttention</td></tr>
<tr><td><strong>Sarvam 105B</strong></td><td><strong>MLA</strong></td><td>FA3</td></tr>
<tr><td><strong>Sarvam 30B</strong></td><td><strong>GQA</strong></td><td>FA3</td></tr>
<tr><td><strong>Whisper / T5</strong></td><td>MHA (encoder-decoder; supports Slim Attention retrofit)</td><td>Slim Attention available; 2-32× compression</td></tr>
<tr><td><strong>IHA-based models</strong></td><td>IHA (experimental, Feb 2026)</td><td>FlashAttention compatible</td></tr>
<tr><td><strong>Closed (GPT-5.5, Claude 4.7, Gemini 3.1)</strong></td><td>Undisclosed (likely GQA-derivatives)</td><td>Undisclosed</td></tr>
</tbody>
</table>
</div>

<p>Three observations:</p>

<ul>
  <li><strong>The 100B threshold is real</strong>. Sarvam ships 105B with MLA and 30B with GQA — same family, different head topologies for different scales. Validates the empirical finding.</li>
  <li><strong>Hybrid stacks combine head-topology choices per layer type</strong>. Qwen 3.5/3.6 uses GQA in its full-attention layers and GDN in its linear-attention layers (M45). Kimi Linear uses MLA in full-attention + KDA in linear. The Axis C choice is layer-specific, not global.</li>
  <li><strong>IHA is the entry to watch</strong>. Feb 2026 publication; FlashAttention compatible; substantial reasoning gains. By Q3-Q4 2026 expect frontier models incorporating it as a head-topology choice.</li>
</ul>

<div class="ndq">
<h4>About head topology and KV-cache engineering</h4>

<p class="q">Why does GQA underperform MHA in DeepSeek's ablations but everyone uses GQA anyway?</p>
<p class="a">Three reasons that compound. (1) <strong>The DeepSeek ablations were at MoE+frontier scale</strong>. At 100B+ parameters, MLA's compression-decompression overhead is a small fraction of total compute, and its slight regularization effect helps. At smaller scales, the ratio is different — GQA's simpler structure is easier to optimize for. (2) <strong>Engineering ecosystem matters</strong>. GQA was published in 2023; FlashAttention, vLLM, PyTorch's scaled_dot_product_attention all have first-class GQA support. MLA is more recent and less tooled — the inference path requires custom kernels for the absorption tricks, and frameworks are still catching up. (3) <strong>Risk tolerance</strong>. GQA is well-understood; MLA is newer. Most teams default to the safer choice. The empirical pattern (Sarvam shipping both, with GQA at 30B and MLA at 105B; everyone else using GQA at smaller scales) reflects this. <em>If you're under 100B and don't have DeepSeek-level kernel engineering, GQA is the practical answer; if you're at frontier MoE scale, MLA is the production choice</em>.</p>

<p class="q">Why does decoupled RoPE work? It feels like a hack — why split into two parts?</p>
<p class="a">Three nested reasons. (1) <strong>The mathematical issue is real</strong>: RoPE rotates K by position, which is incompatible with low-rank K decomposition. There's no clean way to compose rotation with up-projection that preserves both compression and position-awareness. (2) <strong>The split decouples concerns</strong>: position information lives in a small position-aware dimension (k_rope, ~64 dims, shared across heads). Content information lives in the large position-free dimension (k_nope, via c_KV). The two are concatenated for attention computation but cached separately. <em>This is a clean factorization of the original tensor</em>. (3) <strong>The position-aware dimension is shared</strong>. Unlike per-head k_nope, k_rope is single-head — the same RoPE-rotated vector is used by all heads. This is consistent with the empirical observation that position information is more "shared" across heads than content. <em>The hack works because position genuinely is more fungible across heads than content</em>. The decoupled RoPE construction is somewhat baroque mathematically but reflects real structural properties of attention. It's not a workaround — it's a discovery about what factorization the architecture wants.</p>

<p class="q">If IHA is so much better at reasoning, why isn't it in production yet?</p>
<p class="a">Three reasons related to research-to-production timing. (1) <strong>It's only 2 months old</strong> as of the publication of this module. Production training runs take months; teams that want IHA in their next model are starting now, with releases targeted for Q3-Q4 2026. (2) <strong>The interleaving requires expanded sequence lengths</strong>. With P=H, the sequence appears H× longer during attention. This means H× more activation memory during training — a real cost. The IHA paper validates at 1.5B parameters; scaling to frontier (300B+) is engineering work nobody has published yet. (3) <strong>Hybrid architecture compatibility</strong>. Modern frontier models combine attention with MoE, with sparse attention, with hybrid linear layers. IHA needs to compose with these; the interactions haven't been validated. <em>The window between research publication and production adoption is typically 6-12 months for fundamental architectural changes</em>. Expect IHA in production by late 2026 if the empirical results hold up; if they don't generalize beyond the paper's scale, it stays as a research curiosity. The history of attention variants is full of both outcomes.</p>

<p class="q">Why does Slim Attention only matter for older / encoder-decoder models?</p>
<p class="a">Slim Attention's central trick (compute V from K via V = K @ W_KV) requires <em>square projection matrices</em>. Specifically, it needs the projection from input to K to be invertible — which means d_model = n_heads × d_K with no sharing across heads. This is true for vanilla MHA. But in 2026 production:</p>
<p>(a) <strong>GQA breaks it</strong> — fewer KV heads than query heads means W_K is not square; the inversion doesn't work cleanly.</p>
<p>(b) <strong>MQA breaks it</strong> — only one KV head; same issue.</p>
<p>(c) <strong>MLA breaks it</strong> — K comes from low-rank projection through c_KV; the structure is fundamentally different.</p>
<p>So Slim Attention's natural habitat is full-MHA models, which by 2026 are mostly the older / specialized ones: Whisper (speech), T5 (encoder-decoder NLP), Chronos (time series). For these, Slim Attention provides a substantial drop-in win at zero quality cost. <em>For modern decoder-only LLMs (Llama, DeepSeek, Qwen, Mistral, Claude, GPT, Gemini), it doesn't apply directly</em>. The intellectual contribution lives on in MLA's weight absorption (a similar "reconstruct from compressed" idea) but the specific Slim Attention method is tied to MHA architectures.</p>

<p class="q">What's the practical limit on KV cache compression — can we keep going below 2 bits?</p>
<p class="a">Probably yes, but with diminishing returns and increasing risk. Three considerations. (1) <strong>The TurboQuant authors note 4-bit is the "sweet spot"</strong> — quality essentially indistinguishable from FP16 for 3B+ models. At 3-bit, quality starts degrading on smaller models. At 2-bit, more degradation. The empirical curve shows the marginal compression vs marginal quality curve flattening rapidly below 4 bits. (2) <strong>Information-theoretic floor</strong>: there's a minimum number of bits per KV element to preserve essential information. Random projections (used in TurboQuant's PolarQuant) help by spreading information uniformly across dimensions, but you can't go arbitrarily low. (3) <strong>Specific tasks matter</strong>: long-context retrieval (NIAH-style) is more sensitive to KV precision than general-purpose generation. Aggressive compression (1.5-bit, 1-bit) might work for chat but fail on retrieval. <em>The realistic floor in 2026-2027 is probably 1.5-2 bits</em>. Going below that requires fundamentally different techniques — perhaps cache eviction (drop tokens entirely) or learned compression (autoencoder-like KV compression trained alongside the model). Both are active research areas.</p>

<p class="q">How do head topology choices interact with M44's MoE patterns?</p>
<p class="a">Mostly orthogonally, but with some interesting interactions. <strong>MoE replaces the FFN</strong>, head topology shapes the attention block. They live in different parts of the transformer block. So you can mix any head topology with any MoE pattern: DeepSeek V4 = MLA + alternating MoE/dense. Llama 4 Maverick = GQA + alternating MoE. Qwen 3.6 = GQA in attention layers + MoE in FFN layers. <strong>The interesting interactions</strong>: (a) <em>MLA's weight absorption interacts with MoE expert routing</em> — the absorbed query projection depends on which up-projection is used, and per-expert routing introduces variability. DeepSeek's V3 paper handles this; the interaction is non-trivial. (b) <em>IHA's expanded sequence interacts with expert capacity</em> — the P× longer sequence during attention means more tokens routed to each expert. Capacity factors must be retuned. (c) <em>GQA's grouping doesn't interact with MoE routing</em> — they live in different blocks. <em>The general rule</em>: for production engineering, treat head topology and MoE pattern as independent choices. Compose them as needed. Tooling (Megatron-Core, vLLM) supports the combinations.</p>
</div>

<h2>Code Magnets: implement MLA forward with decoupled RoPE</h2>

<p>You're writing the MLA forward pass with decoupled RoPE. Three magnets are wrong choices.</p>

<div class="magnets">
<p>Arrange the magnets to compute MLA attention output for a single token.</p>

<div class="magnet-pool">
  <span class="magnet">def mla_forward(x, weights, n_heads):</span>
  <span class="magnet">    c_q = x @ weights.W_dq</span>
  <span class="magnet">    c_KV = x @ weights.W_dkv  # cache this</span>
  <span class="magnet">    q_nope = c_q @ weights.W_uq</span>
  <span class="magnet">    q_rope = apply_rope(x @ weights.W_qr)</span>
  <span class="magnet">    q_rope = x @ weights.W_qr  # forgot RoPE</span>
  <span class="magnet">    k_nope = c_KV @ weights.W_uk</span>
  <span class="magnet">    k_rope = apply_rope(x @ weights.W_kr)  # cache this — single shared head</span>
  <span class="magnet">    k_rope = apply_rope(c_KV @ weights.W_kr)  # rope on latent — wrong</span>
  <span class="magnet">    v = c_KV @ weights.W_uv</span>
  <span class="magnet">    k_rope_per_head = k_rope.unsqueeze(-2).expand(-1, -1, n_heads, -1)</span>
  <span class="magnet">    Q = torch.cat([q_nope, q_rope], dim=-1)</span>
  <span class="magnet">    K = torch.cat([k_nope, k_rope_per_head], dim=-1)</span>
  <span class="magnet">    K = k_nope + k_rope_per_head  # element-wise — wrong</span>
  <span class="magnet">    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))</span>
  <span class="magnet">    output = F.softmax(scores, dim=-1) @ v</span>
  <span class="magnet">    return output</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">def</span> <span class="fn">mla_forward</span>(x, weights, n_heads):
    c_q = x @ weights.W_dq
    c_KV = x @ weights.W_dkv  <span class="com"># cache this</span>
    q_nope = c_q @ weights.W_uq
    q_rope = <span class="fn">apply_rope</span>(x @ weights.W_qr)
    k_nope = c_KV @ weights.W_uk
    k_rope = <span class="fn">apply_rope</span>(x @ weights.W_kr)  <span class="com"># cache this — single shared head</span>
    v = c_KV @ weights.W_uv
    k_rope_per_head = k_rope.<span class="fn">unsqueeze</span>(-<span class="num">2</span>).<span class="fn">expand</span>(-<span class="num">1</span>, -<span class="num">1</span>, n_heads, -<span class="num">1</span>)
    Q = torch.<span class="fn">cat</span>([q_nope, q_rope], dim=-<span class="num">1</span>)
    K = torch.<span class="fn">cat</span>([k_nope, k_rope_per_head], dim=-<span class="num">1</span>)
    scores = Q @ K.<span class="fn">transpose</span>(-<span class="num">2</span>, -<span class="num">1</span>) / math.<span class="fn">sqrt</span>(Q.<span class="fn">size</span>(-<span class="num">1</span>))
    output = F.<span class="fn">softmax</span>(scores, dim=-<span class="num">1</span>) @ v
    <span class="kw">return</span> output</code></pre>
<p>The traps:</p>
<ul>
  <li><code>q_rope = x @ weights.W_qr  # forgot RoPE</code>: skips the RoPE rotation that gives this branch its purpose. The whole point of the decoupled RoPE design is that <em>q_rope</em> carries position information via RoPE, while <em>q_nope</em> is position-free. Without RoPE applied, q_rope is just another position-free projection — which means the model has no way to encode position at all. Attention scores would no longer depend on token position; long-context tasks would fail catastrophically. The RoPE application is mandatory for the position-aware branch.</li>
  <li><code>k_rope = apply_rope(c_KV @ weights.W_kr)  # rope on latent — wrong</code>: applies RoPE to the wrong projection. Decoupled RoPE specifically computes k_rope from the <em>raw input x</em>, not from the latent c_KV. The reason: if k_rope were computed via the latent, it would suffer from the same compression issue that motivated decoupling in the first place. The whole architectural insight is that position-information flows through a separate, uncompressed path. Computing k_rope through c_KV defeats the purpose; the decoupling becomes structural fiction.</li>
  <li><code>K = k_nope + k_rope_per_head  # element-wise — wrong</code>: adds the two key components instead of concatenating them. The two components have different semantic roles (k_nope is content-aware via the latent, k_rope is position-aware) AND potentially different dimensions (d_qk_nope vs d_qk_rope). Adding them mixes information that should remain separable; concatenation along the feature dimension preserves the structure: <code>score = Q · K = q_nope · k_nope + q_rope · k_rope</code>. Element-wise addition gives a different (and wrong) score formula. The concat is what makes the math work out to MHA-equivalent attention with split position handling.</li>
</ul>
<p>The pattern: <strong>down-project to latent, up-project for nope, project + RoPE for rope, concatenate (not add), standard attention</strong>. Each step has a precise role; the most common implementation bugs are forgetting RoPE on q_rope (silent quality loss for long-context), routing RoPE through the latent (defeats decoupling), and using add instead of concat (breaks the dot-product structure).</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each Axis C variant to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>MHA (Multi-Head Attention)</div>
  <div>A. Vanilla; every head has its own K, V; full quality, full KV cache cost.</div>

  <div>MQA (Multi-Query Attention)</div>
  <div>B. All heads share one K, one V; minimal cache but quality drops; Falcon, PaLM-era.</div>

  <div>GQA (Grouped-Query Attention)</div>
  <div>C. Groups of heads share K, V; the open-source default 2024-2026; Llama, Mistral, Qwen.</div>

  <div>MLA (Multi-Head Latent Attention)</div>
  <div>D. Compress K, V into low-rank latent; decompress on demand; decoupled RoPE; DeepSeek V2/V3/V4.</div>

  <div>Decoupled RoPE</div>
  <div>E. Split into position-aware (RoPE applied, small) + position-free (compressed via latent, large).</div>

  <div>IHA (Interleaved Head Attention)</div>
  <div>F. Pseudo-heads enable cross-head mixing; P² patterns per head; Feb 2026, +5.8% GSM8K.</div>

  <div>Slim Attention</div>
  <div>G. K-cache only; reconstruct V from K via precomputed W_KV; lossless 2× compression for MHA.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>MHA</strong> → A<br>
<strong>MQA</strong> → B<br>
<strong>GQA</strong> → C<br>
<strong>MLA</strong> → D<br>
<strong>Decoupled RoPE</strong> → E<br>
<strong>IHA</strong> → F<br>
<strong>Slim Attention</strong> → G
</p>
<p>The mental shortcut: <em>MHA every head solo, MQA shares all, GQA shares groups, MLA compresses to latent, decoupled RoPE splits position from content, IHA mixes heads via pseudos, Slim reconstructs V from K</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team is deploying a 405B-parameter model at 1M context. They have a fixed budget of GPU memory. Walk through their head topology + cache compression decision.</p>
<details class="answer"><summary>show answer</summary>
<p>The math is the deciding factor here. Step through:</p>
<p>(1) <strong>Compute baseline KV cache for MHA at 1M context</strong>. 405B model has approximately 80 layers × 128 heads × 128 d_head × 2 bytes = 2.6 MB per token in MHA. At 1M tokens: <strong>2.6 TB of KV cache</strong>. Doesn't fit on any single H100 (80GB) or B200 (192GB). Doesn't even fit on a GB200 NVL72 system (72 × 192GB = 13.8 TB total) once you account for batching multiple users.</p>
<p>(2) <strong>Try GQA-8</strong>. 4× compression. 650 GB. Still doesn't fit; needs distributed serving across many GPUs even for batch size 1.</p>
<p>(3) <strong>Try MLA</strong>. 5-7× compression over GQA-8. ~100-130 GB at 1M context. Fits on a single B200 or B300 with headroom for batching. <em>This is the production answer for 405B at 1M context.</em></p>
<p>(4) <strong>Add TurboQuant 3-bit on top</strong>. Another 4× compression. ~25-33 GB. Fits comfortably; allows substantial batching for multiple concurrent users. The decode throughput penalty (35-43% reduction) is acceptable for long-context use cases where the alternative is "doesn't run at all."</p>
<p>(5) <strong>Optional: KV eviction (Expected Attention or H2O)</strong>. For the most concurrent users, evict tokens with low expected attention. Adds another 2-4× capacity at some quality risk. Worth it if your traffic pattern is high-concurrency.</p>
<p>(6) <strong>Hardware deployment</strong>: with MLA + TurboQuant 3-bit, a single B300 NVL72 rack can serve hundreds of concurrent 1M-context sessions. Without these compressions, even a single session would require multiple racks.</p>
<p><strong>The recommendation</strong>: pretrain (or use a model that pretrained) with MLA. Deploy with TurboQuant 3-bit KV cache. Add Expected Attention eviction if concurrency is high. Skip KV offloading unless GPU memory is truly insufficient (PCIe bandwidth becomes the bottleneck).</p>
<p>The general lesson: <strong>at frontier scale × frontier context length, head topology compresses by 5-7× and cache engineering compresses by another 4-8×, totaling 25-50× over MHA-FP16</strong>. This is what makes 1M context economically viable. Without these compressions, 1M context exists only in research papers.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Walk through why MLA's weight absorption makes inference cheaper than the naive MLA forward pass would suggest.</p>
<details class="answer"><summary>show answer</summary>
<p>The naive MLA inference path (what the forward-pass code shows) is:</p>
<p><strong>Naive path</strong>: cache c_KV; at attention time, decompress to K_nope = c_KV @ W_uk and V = c_KV @ W_uv (full dimension); compute attention with K_nope, V at full dimension.</p>
<p>This works but materializes K_nope and V at full dimension during every attention call — a substantial compute cost.</p>
<p><strong>The absorbed path</strong>: precompute W_q_absorbed = W_uq @ W_uk^T offline (W_uq is the query up-projection, W_uk is the key up-projection). Now at inference time:</p>
<pre>score_nope = (c_q @ W_q_absorbed) @ c_KV.T</pre>
<p>Note what happened: we never materialized K_nope. The query was projected into the latent space (dimension r_kv ≈ 512) via W_q_absorbed, and the dot product was computed directly with c_KV. <strong>Attention computation now happens at the latent dimension, not the full K dimension</strong>.</p>
<p>For values, similar: W_o (output projection) can be composed with W_uv to give W_o_absorbed = W_uv @ W_o. The attention output is computed at the latent dimension and projected to output via W_o_absorbed in a single step.</p>
<p>The savings:</p>
<p>(1) <strong>Compute</strong>: attention happens at latent dimension (~512) instead of full K dimension (~16,384). 32× reduction in attention FLOPs at 128 heads × 128 d_head.</p>
<p>(2) <strong>Memory bandwidth</strong>: only c_KV (small) is loaded from cache, not K and V at full dimension. ~57× reduction in memory traffic.</p>
<p>(3) <strong>Effective speedup</strong>: in the DeepSeek V2 paper, MLA at inference is reported at 5.76× generation speed of MHA — exactly the kind of speedup you'd expect from this compute + bandwidth reduction.</p>
<p>The trick that makes weight absorption work: <em>matrix multiplication is associative</em>. (q @ W_uq) @ (c_KV @ W_uk).T can be reordered to q @ (W_uq @ W_uk.T) @ c_KV.T. The middle matrix W_q_absorbed is precomputed offline; the runtime path becomes much cheaper.</p>
<p>The general lesson: <strong>algebraic absorption is what makes MLA practical for inference</strong>. Without it, MLA would just be a "smaller cache" architecture; with it, MLA actually accelerates inference. <em>Production MLA implementations (DeepGEMM in DeepSeek's stack) bake the absorption in</em>; naive implementations would miss most of the benefit.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> A team is fine-tuning Qwen3-VL (which uses GQA) for their production application. They want long-context (256K) inference but are GPU-memory-constrained. Sketch the plan.</p>
<details class="answer"><summary>show answer</summary>
<p>The decision tree from the SVG points to "use existing model + KV cache engineering." Step through:</p>
<p>(1) <strong>Head topology is fixed</strong>: GQA-8 (whatever the original Qwen3-VL ratio is). Can't change without retraining from scratch, which the team isn't doing.</p>
<p>(2) <strong>Apply TurboQuant 3-bit KV quantization</strong>. Available in vLLM as of April 15, 2026:</p>
<pre>vllm serve qwen/qwen3-vl-X --kv-cache-dtype turboquant_3bit_nc</pre>
<p>4× capacity over FP16 GQA. Expected quality impact: minimal for 3B+ models on most workloads. Validate on the team's specific tasks (the Qwen3-VL multimodal evals) — if quality holds, ship it.</p>
<p>(3) <strong>Add Expected Attention or H2O eviction</strong> for very long contexts. At 256K, low-attention tokens can be evicted with minimal quality loss. Adds another 2-3× effective capacity. Risk: rare retrieval tasks might miss tokens that were evicted — test on the workload before committing.</p>
<p>(4) <strong>If still memory-bound, KV offloading</strong>. InfiniGen-style with prefetching. Lossless but PCIe bandwidth becomes the bottleneck — typically only worth it if the team has large CPU memory and tolerable latency budget.</p>
<p>(5) <strong>Estimate the resulting cache size</strong>. For a 7B Qwen3-VL with GQA-8 at 256K context:</p>
<ul>
<li>MHA-FP16 baseline: ~32 GB</li>
<li>GQA-8 (already used): ~8 GB</li>
<li>+ TurboQuant 3-bit: ~2 GB</li>
<li>+ 50% Expected Attention eviction: ~1 GB</li>
</ul>
<p>1 GB of KV cache fits on any modern GPU with comfortable headroom for batching.</p>
<p>(6) <strong>Validate quality on production tasks</strong>. M37's eval rigor applies. Specifically test:</p>
<ul>
<li>Long-context retrieval (does the model still find specific facts buried in 256K?)</li>
<li>Multi-modal reasoning (does the vision-language alignment survive KV compression?)</li>
<li>End-to-end task accuracy on the team's specific benchmarks</li>
</ul>
<p>(7) <strong>Compare alternatives</strong>: switching to a model with MLA (like a Kimi or DeepSeek variant) would give better cache compression natively, but switching models has a cost (engineering, prompt tuning, eval re-validation). For most teams, optimizing the existing Qwen3-VL with cache engineering is the right path.</p>
<p>The general lesson: <strong>head topology is fixed once you've chosen a model; KV cache engineering is the lever you have at deployment time</strong>. TurboQuant + eviction + offloading compose; pick the combination that meets your memory budget and quality bar.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> Forecast: what's the most likely successor to MLA at frontier scale by 2027?</p>
<details class="answer"><summary>show answer</summary>
<p>Three candidate directions, each with arguments for and against:</p>
<p><strong>Candidate 1: MLA-extended (deeper compression hierarchies)</strong>. Argument for: MLA proved low-rank compression works at scale; deeper hierarchies (compress c_KV further into c_meta) might extend the gains. Argument against: each level of compression adds compute overhead at decompression; the tradeoff curve flattens past some point. <em>Likely</em>: incremental MLA refinements continue, but no qualitative leap.</p>
<p><strong>Candidate 2: IHA-derivatives (cross-head mixing as standard)</strong>. Argument for: IHA's expressivity gains are substantial (10-20% on RULER, +5.8% GSM8K) and FlashAttention compatible. The paper's parameter efficiency theory (sqrt(k) heads vs k heads for compositional tasks) suggests this is fundamental, not incremental. Argument against: scaling IHA to frontier scale hasn't been validated; the expanded sequence length during attention is a real engineering cost. <em>Likely</em>: IHA-style cross-head mixing becomes a standard option, possibly composed with MLA's compression.</p>
<p><strong>Candidate 3: DeepSeek V4-style hybrid (CSA+HCA + MLA)</strong>. Argument for: DeepSeek V4 already shipped this combination — interleaved CSA (4× compress + sparse selection) and HCA (128× compress + dense) layers, with MLA-style head topology. Multiplies compression beyond what MLA alone gives. Argument against: substantial engineering complexity; harder to reproduce and tune. <em>Likely</em>: hybrid sparse-attention + compressed-head-topology becomes the frontier pattern, with DeepSeek V4 as the prototype and others (Llama 5, Qwen 4) following.</p>
<p><strong>The wildcard: linear attention takes more share</strong>. As Mamba-3, Gated DeltaNet, KDA, Lightning Attention mature (M45 territory), the head-topology question may become less central — if 75-90% of layers are linear attention, the remaining attention layers' KV cache is no longer the dominant memory cost. The Axis C choices matter less when there are fewer attention layers. <em>Less likely</em> than pure attention-architecture refinement, but possible if linear attention closes more quality gaps.</p>
<p><strong>Concrete forecast</strong>: by Q3-Q4 2027, expect the dominant frontier pattern to be:</p>
<ul>
<li><strong>50-70% linear attention layers</strong> (M45) — Mamba-3 derivatives or Kimi Delta Attention-style</li>
<li><strong>30-50% softmax attention layers with MLA + sparse pattern</strong> — DeepSeek V4-style hybrid</li>
<li><strong>IHA-style cross-head mixing</strong> within those softmax layers, if the scaling work pans out</li>
<li><strong>TurboQuant successor at 1.5-2 bits</strong> for the remaining KV cache</li>
</ul>
<p>The general lesson: <em>"replacement of MLA" is the wrong frame</em>. Frontier 2027 architectures will compose multiple Axis C ideas (MLA's compression, IHA's mixing, sparse patterns) rather than picking one. The interesting research questions are about which compositions work, not which single mechanism wins.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>Axis C of the four-axis taxonomy: head topology — how Q, K, V are structured across heads. The most production-critical axis because it determines KV-cache cost.</li>
  <li><strong>MHA</strong>: vanilla; every head has independent Q, K, V projections; full quality but full cache cost. Per-token KV: n_heads × 2 × d_head.</li>
  <li><strong>MQA</strong>: all heads share one K, one V. 32× cache compression at 32 heads, but quality drops. Falcon, PaLM, ChatGLM2 era.</li>
  <li><strong>GQA</strong>: groups of query heads share K, V. 4-8× cache compression with minimal quality loss. <strong>Open-source default 2024-2026</strong>: Llama 2/3/4, Mistral, Qwen 3.5/3.6, MiniMax M2.5.</li>
  <li><strong>MLA (DeepSeek V2-V4)</strong>: compress K, V into low-rank latent c_KV (dim ~512); decompress on demand. Cache c_KV instead of full K, V. ~57× cache compression. Quality matches or exceeds MHA at scale.</li>
  <li><strong>Decoupled RoPE</strong> makes MLA work: split each query and key into <strong>position-aware</strong> (RoPE applied, small dimension d_qk_rope ≈ 64) and <strong>position-free</strong> (compressed via latent, large dimension). Cache c_KV + shared k_rope. Concatenate at attention time.</li>
  <li><strong>Weight absorption</strong>: at inference, W_uq @ W_uk^T precomputed offline; attention happens at latent dimension instead of full K dimension. ~32× compute reduction, ~57× memory bandwidth reduction. <strong>5.76× generation speedup vs MHA</strong> in DeepSeek V2.</li>
  <li><strong>The 100B threshold</strong>: MLA wins at &gt;100B scale; GQA wins below. Sarvam ships 105B with MLA, 30B with GQA — same family, different topologies for different scales.</li>
  <li><strong>IHA — Interleaved Head Attention (Duvvuri et al., Feb 2026)</strong>: pseudo-heads enable cross-head mixing. P pseudo-heads per head as learned linear combinations of all H originals. Interleave along sequence; standard attention; combine back. <strong>P² patterns per head with O(H²P) parameter overhead</strong>. FlashAttention compatible. Empirical: +10-20% RULER Multi-Key, +5.8% GSM8K, +2.8% MATH-500.</li>
  <li><strong>Slim Attention (Graef &amp; Wasielewski, March 2025)</strong>: when projection matrices are square, V can be reconstructed from K via V = K · W_KV. Cache only K. Lossless 2× compression. 8× for Whisper, 32× for T5-11B (non-square projections). Not applicable to GQA/MQA/MLA.</li>
  <li><strong>KV-cache engineering compounds with head topology</strong>:
    <ul>
      <li><strong>TurboQuant (ICLR 2026, merged into vLLM April 15, 2026)</strong>: PolarQuant random rotations + Lloyd-Max quantization. 3-bit keys + 2-bit values. <strong>4× capacity</strong>. Available as <code>--kv-cache-dtype turboquant_3bit_nc</code>. Decode penalty: 35-43% throughput reduction.</li>
      <li><strong>Expected Attention</strong>: estimate future query distribution, evict low-expected-attention keys. Quality-cost tradeoff.</li>
      <li><strong>Quest, H2O, SnapKV</strong>: various heuristic KV eviction schemes.</li>
      <li><strong>InfiniGen</strong>: KV cache offloading to CPU with prefetching. Lossless but PCIe-bandwidth-bound.</li>
    </ul>
  </li>
  <li>Production stack: GQA + TurboQuant 3-bit (90% of teams) or MLA + DeepGEMM (10%, frontier scale). Both compose with FlashAttention 3/4 + PagedAttention.</li>
  <li>Production zoo: Llama 4 = GQA; DeepSeek V4 = MLA + CSA+HCA hybrid; GLM-5.1 = MLA-style + DSA; Qwen 3.5/3.6 = GQA in attention layers + GDN in linear layers (M45); Kimi Linear = MLA + KDA hybrid; Sarvam 105B = MLA, Sarvam 30B = GQA; Whisper/T5 = MHA + Slim Attention retrofit.</li>
  <li>Decision tree: pretraining + &gt;100B → MLA; pretraining + &lt;100B → GQA; reasoning-focused new model → consider IHA (experimental); fine-tuning existing model → KV cache engineering (TurboQuant + eviction); MHA encoder-decoder → Slim Attention.</li>
  <li>The forecast for Q3-Q4 2027: hybrid stacks dominate. 50-70% linear attention layers (M45), 30-50% softmax with MLA + sparse pattern (DeepSeek V4-style), IHA-style cross-head mixing within softmax layers, TurboQuant successor at 1.5-2 bits. <em>The interesting question is which compositions work, not which single mechanism wins</em>.</li>
</ul>
</div>

<p>Module 47 (next, completing the trilogy) tackles <strong>Axes D and E — kernel implementation and compositional architectures</strong>: FlashAttention 1-4 with the FA4 reverse-engineered Blackwell pipeline, PagedAttention's KV cache layout, Ring Attention and Striped Attention for sequence parallelism at &gt;1M context, MoDA's depth attention (March 2026), and the hybrid composition patterns across the production zoo (Qwen3-Next 3:1, Kimi Linear, Ling 2.5, Nemotron 3 Nano, DeepSeek V4 CSA+HCA).</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">46</span>
  <span>Attention Variants II — Head Topology &amp; KV-Cache Engineering</span>
</div>
"""

emit("46_attention_variants_2", "Module 46 — Attention Variants II: Head Topology & KV-Cache Engineering", BODY)
