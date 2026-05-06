#!/usr/bin/env python3
"""Module 29: Building a transformer from scratch — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part IX · Module 29 · synthesis</div>
  <h1 class="module-title">Building a <em>transformer</em> from scratch</h1>
  <p class="module-sub">— a complete GPT-style model in ~250 lines of PyTorch, every line explained, with KV cache wiring and a debugging checklist for first-run failures</p>
</div>

<p>For 28 modules we've covered the components — tensors, autograd, modules, optimizers, attention kernels, distributed training, serving. <em>This module assembles them into a working model</em>. By the end, you'll have a complete GPT-style transformer that trains, generates text, and exposes the same KV cache interface that vLLM (M27) consumes. ~250 lines of code, every line motivated by a specific module's content.</p>

<p>The point isn't novelty. There are plenty of "build GPT from scratch" tutorials. The point is that, having seen the components, you should now see <em>why every architectural decision is the way it is</em>. The head dimension being a multiple of 8 is M22 (tensor cores). Pre-norm vs post-norm is M14 (numerical stability). The init scheme is M8. Causal masking integrates with attention from M25. The KV cache layout matches M27's paged design. <strong>This is the module where the framework you've built becomes a competent engineer's reflex set.</strong></p>

<div class="keyidea">
A GPT-style transformer is: <strong>token embeddings → N transformer blocks (each: pre-norm + attention + residual; pre-norm + MLP + residual) → final norm → lm_head</strong>. The attention block uses RoPE for positional encoding (M30 covers this in depth), Grouped-Query Attention for KV cache efficiency (M27), and PyTorch's <code>F.scaled_dot_product_attention</code> for FlashAttention dispatch (M25). The MLP uses SwiGLU (the modern gated variant). Init follows M8's scheme; the optimizer is AdamW (M9); training mixed-precision in bf16 (M14). The whole thing fits in ~250 lines. <strong>Every line has a reason that traces back to a specific module.</strong>
</div>

<h2>Two final new faces</h2>

<div class="character" style="--c: #133e3b;">
  <div class="avatar" style="background: #133e3b; color: #fff;">G</div>
  <div>
    <p class="who">GPT</p>
    <p class="name">"I'm what you've been building toward — the assembled model."</p>
    <p class="says">I'm a stack of transformer blocks with a token embedding at the bottom and an LM head at the top. Each block has attention (Q, K, V projections, FlashAttention via SDPA, output projection) and an MLP (SwiGLU). I emit logits over the vocabulary; you sample from those to generate. I'm small (~85M params for a GPT-2-small clone, ~7B for a Llama-style 7B); the architecture is the same at every scale, just wider and deeper. <em>Every component you've met in this course is somewhere inside me.</em></p>
  </div>
</div>

<div class="character" style="--c: #6b5d4f;">
  <div class="avatar" style="background: #6b5d4f; color: #fff;">▲</div>
  <div>
    <p class="who">Causal Mask</p>
    <p class="name">"I'm the upper-triangular bouncer. Token i can't attend to token j when j > i."</p>
    <p class="says">In training, I make sure each position only sees the past. Without me, the model trivially cheats — it can read the next token's embedding directly. I'm logically a (T, T) boolean matrix where <code>mask[i, j] = (j ≤ i)</code>, but in modern kernels I'm not materialized: <code>F.scaled_dot_product_attention(..., is_causal=True)</code> tells FlashAttention (M25) to skip K/V tiles above the diagonal entirely. <em>Free at training time, free at prefill, automatic at decode</em>. I'm the simplest thing in the model, but forgetting me is the most common bug in first implementations.</p>
  </div>
</div>

<h2>The architecture, top-down</h2>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 420" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">GPT data flow: tokens → embedding → N blocks → norm → logits</text>

  <!-- Token IDs in -->
  <g transform="translate(20, 50)">
    <rect x="0" y="0" width="120" height="40" fill="#fff8a8" stroke="#1a1612" stroke-width="1.5"/>
    <text x="60" y="20" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">token_ids</text>
    <text x="60" y="34" text-anchor="middle" font-size="9" fill="#1a1612">[B, T] int64</text>
  </g>
  <path d="M 145 70 L 175 70" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrG)"/>

  <!-- Embedding -->
  <g transform="translate(180, 50)">
    <rect x="0" y="0" width="140" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="70" y="20" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">Embedding</text>
    <text x="70" y="34" text-anchor="middle" font-size="9" fill="#1a1612">[vocab, D] lookup</text>
  </g>
  <path d="M 325 70 L 355 70" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrG)"/>

  <text x="360" y="60" font-size="9" fill="#6b5d4f">x: [B, T, D]</text>

  <!-- N transformer blocks -->
  <g transform="translate(440, 35)">
    <rect x="0" y="0" width="280" height="180" fill="#fcecec" stroke="#c1502e" stroke-width="2" stroke-dasharray="4 3"/>
    <text x="140" y="20" text-anchor="middle" font-size="11" font-weight="700" fill="#c1502e">× N transformer blocks</text>

    <!-- Block contents: pre-norm + attn + residual; pre-norm + mlp + residual -->
    <rect x="20" y="35" width="240" height="60" fill="#fff8a8" stroke="#1a1612" stroke-width="1"/>
    <text x="140" y="52" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Sub-block 1: Attention</text>
    <text x="140" y="68" text-anchor="middle" font-size="9" fill="#1a1612">x = x + attn(norm₁(x))</text>
    <text x="140" y="84" text-anchor="middle" font-size="9" fill="#1a1612">(RoPE inside, FlashAttention via SDPA)</text>

    <rect x="20" y="105" width="240" height="60" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/>
    <text x="140" y="122" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Sub-block 2: MLP</text>
    <text x="140" y="138" text-anchor="middle" font-size="9" fill="#1a1612">x = x + mlp(norm₂(x))</text>
    <text x="140" y="154" text-anchor="middle" font-size="9" fill="#1a1612">(SwiGLU: down(silu(gate)·up))</text>
  </g>

  <!-- After blocks: final norm + lm_head -->
  <g transform="translate(180, 250)">
    <rect x="0" y="0" width="140" height="40" fill="#fff5d8" stroke="#d4a017" stroke-width="1.5"/>
    <text x="70" y="20" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">final norm</text>
    <text x="70" y="34" text-anchor="middle" font-size="9" fill="#1a1612">RMSNorm</text>
  </g>
  <path d="M 325 270 L 355 270" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrG)"/>

  <g transform="translate(360, 250)">
    <rect x="0" y="0" width="140" height="40" fill="#ffd5dc" stroke="#b85a6c" stroke-width="1.5"/>
    <text x="70" y="20" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">lm_head (Linear)</text>
    <text x="70" y="34" text-anchor="middle" font-size="9" fill="#1a1612">[D, vocab]</text>
  </g>
  <path d="M 505 270 L 535 270" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrG)"/>

  <g transform="translate(540, 250)">
    <rect x="0" y="0" width="180" height="40" fill="#ede2cc" stroke="#6b5d4f" stroke-width="1.5"/>
    <text x="90" y="20" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">logits</text>
    <text x="90" y="34" text-anchor="middle" font-size="9" fill="#1a1612">[B, T, vocab_size]</text>
  </g>

  <!-- Connecting line from blocks down to final norm -->
  <path d="M 580 220 L 580 240 L 250 240 L 250 250" stroke="#1f5f5b" stroke-width="1.5" fill="none" stroke-dasharray="3 2"/>

  <!-- Annotations -->
  <text x="370" y="335" font-family="'Caveat', cursive" font-size="20" fill="#1a1612" text-anchor="middle">"residual stream": x flows top-to-bottom, each block adds its contribution</text>
  <text x="370" y="360" font-family="'Caveat', cursive" font-size="20" fill="#c1502e" text-anchor="middle">pre-norm: norm BEFORE the sub-layer, residual ADDS the unnormed path</text>
  <text x="370" y="385" font-family="'Caveat', cursive" font-size="22" fill="#1f5f5b" text-anchor="middle">lm_head shares weights with embedding (tied embeddings) — saves params</text>
  <text x="370" y="410" font-family="'Caveat', cursive" font-size="18" fill="#6b5d4f" text-anchor="middle">N=12 for ~85M params (GPT-2-small); N=32 for ~7B (Llama-7B)</text>
</svg>
</div>

<p>Read the diagram top-to-bottom: token IDs come in, get embedded, flow through N transformer blocks (each adds its contribution to the residual stream), get normed one final time, and project to vocabulary logits. Three structural decisions worth flagging:</p>

<ul>
  <li><strong>Pre-norm</strong>: each sub-block is structured as <code>x + sublayer(norm(x))</code>, not <code>norm(x + sublayer(x))</code>. Pre-norm is more numerically stable at depth — gradients flow through the residual path unchanged, only the sub-layer's contribution gets normed. Post-norm is what the original Transformer paper used; modern transformers (GPT-2 onward) all use pre-norm. <em>The exception that proves the rule</em>: post-norm needs careful warmup; pre-norm trains stably from step 0.</li>
  <li><strong>RMSNorm, not LayerNorm</strong>: RMSNorm drops the mean-subtraction step (M8). Same quality, ~30% faster, used by Llama, Mistral, Gemma. We'll use it.</li>
  <li><strong>Tied embeddings</strong>: the <code>lm_head</code> Linear shares weights with the input embedding. Saves <code>vocab_size × D</code> parameters (a few percent of the model for typical vocab sizes). Most modern GPTs do this; Llama-3 stopped tying for the largest sizes.</li>
</ul>

<h2>Building it: the components</h2>

<p>Six components, one at a time. Each ~30-50 lines.</p>

<h3>1. Config</h3>

<p>Hold all the model dimensions in one dataclass. Makes scaling experiments trivial.</p>

<pre><code><span class="kw">from</span> dataclasses <span class="kw">import</span> dataclass

<span class="kw">@</span>dataclass
<span class="kw">class</span> <span class="ty">GPTConfig</span>:
    vocab_size:    <span class="fn">int</span> = <span class="num">50304</span>     <span class="com"># GPT-2 BPE; bumped to multiple-of-64 for tensor-core dim alignment (M22)</span>
    n_layer:       <span class="fn">int</span> = <span class="num">12</span>        <span class="com"># GPT-2-small depth</span>
    n_head:        <span class="fn">int</span> = <span class="num">12</span>        <span class="com"># number of query heads</span>
    n_kv_head:     <span class="fn">int</span> = <span class="num">4</span>         <span class="com"># number of key/value heads (GQA, M27)</span>
    d_model:       <span class="fn">int</span> = <span class="num">768</span>       <span class="com"># hidden dim; per-head dim = d_model / n_head = 64</span>
    d_mlp:         <span class="fn">int</span> = <span class="num">2048</span>      <span class="com"># MLP hidden; ~2.67× d_model for SwiGLU (vs 4× for GELU)</span>
    max_seq_len:   <span class="fn">int</span> = <span class="num">2048</span>      <span class="com"># context length</span>
    rope_base:     <span class="fn">float</span> = <span class="num">10000.0</span>   <span class="com"># RoPE θ-base; 500000 for long context (M30)</span>
    norm_eps:      <span class="fn">float</span> = <span class="num">1e-6</span>      <span class="com"># RMSNorm epsilon</span>
    tied_embeds:   <span class="fn">bool</span>  = <span class="kw">True</span>      <span class="com"># share input emb with output head</span>
    init_std:      <span class="fn">float</span> = <span class="num">0.02</span>      <span class="com"># init std for linear weights (M8)</span>

    @property
    <span class="kw">def</span> <span class="fn">head_dim</span>(self) -&gt; <span class="fn">int</span>:
        <span class="kw">return</span> self.d_model // self.n_head</code></pre>

<p>Several knobs encode lessons from earlier modules. <code>vocab_size = 50304</code> instead of GPT-2's actual 50257 — the next multiple of 64, so the LM head matmul is tensor-core-friendly (M22). <code>head_dim = 64</code> (multiple of 8). <code>n_kv_head = 4</code> means 3:1 query-to-KV ratio (GQA-3, similar to Llama-3-8B's GQA). <code>d_mlp = 2048</code> instead of the ~3072 a 4× expansion would give — SwiGLU has a gate path so the effective parameter count and compute are similar to a non-gated MLP at 4× expansion.</p>

<h3>2. RMSNorm</h3>

<p>Already covered in M8 and reimplemented as a custom kernel in M23. Here it is in plain PyTorch — what we'll actually use.</p>

<pre><code><span class="kw">class</span> <span class="ty">RMSNorm</span>(nn.Module):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, d, eps=<span class="num">1e-6</span>):
        <span class="fn">super</span>().<span class="fn">__init__</span>()
        self.weight = nn.<span class="fn">Parameter</span>(torch.<span class="fn">ones</span>(d))
        self.eps = eps

    <span class="kw">def</span> <span class="fn">forward</span>(self, x):
        <span class="com"># Cast to fp32 for the reduction (M14: numerical stability matters for norms)</span>
        x_fp32 = x.<span class="fn">float</span>()
        rms = x_fp32.<span class="fn">pow</span>(<span class="num">2</span>).<span class="fn">mean</span>(-<span class="num">1</span>, keepdim=<span class="kw">True</span>).<span class="fn">add</span>(self.eps).<span class="fn">rsqrt</span>()
        <span class="com"># Multiply (still fp32), then cast back to input dtype</span>
        <span class="kw">return</span> (x_fp32 * rms).<span class="fn">to</span>(x.dtype) * self.weight</code></pre>

<p>Three things from M14: cast to fp32 for the reduction (preserves precision when the input is bf16), the rescale happens in fp32, cast back to bf16 only on the output. The <code>self.weight</code> stays in bf16 the whole time — it's a multiplicative scaling that doesn't need fp32 precision. Same recipe as the production RMSNorm in <code>torch.nn.functional</code>.</p>

<h3>3. RoPE</h3>

<p>Rotary positional encoding. M30 covers RoPE in depth — here we just need the apply function. The high-level idea: rotate every pair of dimensions in Q and K by an angle proportional to the position. The result: <code>(Qᵢ · Kⱼ)</code> depends only on <code>i − j</code>.</p>

<pre><code><span class="kw">def</span> <span class="fn">build_rope_cache</span>(seq_len, head_dim, base=<span class="num">10000.0</span>, device=<span class="str">"cuda"</span>):
    <span class="com"># Frequencies for each dim pair</span>
    inv_freq = <span class="num">1.0</span> / (base ** (torch.<span class="fn">arange</span>(<span class="num">0</span>, head_dim, <span class="num">2</span>, device=device).<span class="fn">float</span>() / head_dim))
    <span class="com"># Outer product of positions and frequencies</span>
    t = torch.<span class="fn">arange</span>(seq_len, device=device).<span class="fn">float</span>()
    freqs = torch.<span class="fn">outer</span>(t, inv_freq)              <span class="com"># [T, head_dim/2]</span>
    cos = freqs.<span class="fn">cos</span>().<span class="fn">repeat_interleave</span>(<span class="num">2</span>, dim=-<span class="num">1</span>)  <span class="com"># [T, head_dim]</span>
    sin = freqs.<span class="fn">sin</span>().<span class="fn">repeat_interleave</span>(<span class="num">2</span>, dim=-<span class="num">1</span>)
    <span class="kw">return</span> cos, sin

<span class="kw">def</span> <span class="fn">apply_rope</span>(x, cos, sin):
    <span class="com"># x: [B, n_heads, T, head_dim]. Apply rotation in fp32 for stability.</span>
    x_fp32 = x.<span class="fn">float</span>()
    <span class="com"># Pair adjacent dims into (even, odd); rotate each pair.</span>
    x1, x2 = x_fp32[..., ::<span class="num">2</span>], x_fp32[..., <span class="num">1</span>::<span class="num">2</span>]
    rotated = torch.<span class="fn">stack</span>([-x2, x1], dim=-<span class="num">1</span>).<span class="fn">flatten</span>(-<span class="num">2</span>)
    out = (x_fp32 * cos + rotated * sin).<span class="fn">to</span>(x.dtype)
    <span class="kw">return</span> out</code></pre>

<p>The rotation trick: pair dim 0 with dim 1, dim 2 with dim 3, etc. For each pair, treat them as (real, imaginary), and rotate by the position-dependent angle. The <code>repeat_interleave(2)</code> broadcasts the per-pair frequency to both members of the pair. The <code>stack + flatten</code> implements the imaginary-part swap (real → imaginary; imaginary → -real).</p>

<p>Why fp32 inside the rotation? <code>cos</code> and <code>sin</code> at large positions can produce phase-cancellation issues if computed in bf16. Standard recipe: do the math in fp32, cast back at the end.</p>

<h3>4. Attention</h3>

<p>The biggest single component. Q/K/V projections, head reshape, RoPE, GQA expansion, FlashAttention, output projection.</p>

<pre><code><span class="kw">class</span> <span class="ty">Attention</span>(nn.Module):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, cfg):
        <span class="fn">super</span>().<span class="fn">__init__</span>()
        self.cfg = cfg
        self.q_proj  = nn.<span class="fn">Linear</span>(cfg.d_model, cfg.n_head    * cfg.head_dim, bias=<span class="kw">False</span>)
        self.k_proj  = nn.<span class="fn">Linear</span>(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=<span class="kw">False</span>)
        self.v_proj  = nn.<span class="fn">Linear</span>(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=<span class="kw">False</span>)
        self.o_proj  = nn.<span class="fn">Linear</span>(cfg.n_head    * cfg.head_dim, cfg.d_model, bias=<span class="kw">False</span>)

    <span class="kw">def</span> <span class="fn">forward</span>(self, x, cos, sin, kv_cache=<span class="kw">None</span>):
        B, T, _ = x.shape
        H, KH, D = self.cfg.n_head, self.cfg.n_kv_head, self.cfg.head_dim

        <span class="com"># 1. Project to Q, K, V — three matmuls.</span>
        q = self.<span class="fn">q_proj</span>(x).<span class="fn">view</span>(B, T, H,  D).<span class="fn">transpose</span>(<span class="num">1</span>, <span class="num">2</span>)   <span class="com"># [B, H,  T, D]</span>
        k = self.<span class="fn">k_proj</span>(x).<span class="fn">view</span>(B, T, KH, D).<span class="fn">transpose</span>(<span class="num">1</span>, <span class="num">2</span>)   <span class="com"># [B, KH, T, D]</span>
        v = self.<span class="fn">v_proj</span>(x).<span class="fn">view</span>(B, T, KH, D).<span class="fn">transpose</span>(<span class="num">1</span>, <span class="num">2</span>)   <span class="com"># [B, KH, T, D]</span>

        <span class="com"># 2. Apply RoPE to Q and K (V is positionally invariant).</span>
        q = <span class="fn">apply_rope</span>(q, cos, sin)
        k = <span class="fn">apply_rope</span>(k, cos, sin)

        <span class="com"># 3. KV cache: append new K/V to cached, retrieve full.</span>
        <span class="kw">if</span> kv_cache <span class="kw">is</span> <span class="kw">not</span> <span class="kw">None</span>:
            k = torch.<span class="fn">cat</span>([kv_cache[<span class="num">0</span>], k], dim=<span class="num">2</span>)        <span class="com"># along T axis</span>
            v = torch.<span class="fn">cat</span>([kv_cache[<span class="num">1</span>], v], dim=<span class="num">2</span>)
            kv_cache_out = (k, v)
        <span class="kw">else</span>:
            kv_cache_out = (k, v)

        <span class="com"># 4. GQA: expand K and V to match Q's head count via repeat.</span>
        <span class="com">#    enable_gqa=True in SDPA does this implicitly; we keep it explicit for clarity.</span>
        k = k.<span class="fn">repeat_interleave</span>(H // KH, dim=<span class="num">1</span>)   <span class="com"># [B, H, T_full, D]</span>
        v = v.<span class="fn">repeat_interleave</span>(H // KH, dim=<span class="num">1</span>)

        <span class="com"># 5. The attention itself — FlashAttention via SDPA (M25).</span>
        <span class="com">#    is_causal=True for training/prefill; False for single-token decode (causality automatic with KV cache).</span>
        is_causal = (kv_cache <span class="kw">is</span> <span class="kw">None</span>)
        out = F.<span class="fn">scaled_dot_product_attention</span>(q, k, v, is_causal=is_causal)

        <span class="com"># 6. Reshape back, output projection.</span>
        out = out.<span class="fn">transpose</span>(<span class="num">1</span>, <span class="num">2</span>).<span class="fn">contiguous</span>().<span class="fn">view</span>(B, T, H * D)
        <span class="kw">return</span> self.<span class="fn">o_proj</span>(out), kv_cache_out</code></pre>

<p>Five things to call out, all tracing back to specific modules:</p>

<ul>
  <li><strong>No bias in Linear</strong> (<code>bias=False</code>): biases hurt slightly for transformers and add params for no win. Standard since GPT-J. Same for the output projection.</li>
  <li><strong>K and V have fewer heads than Q</strong> (GQA from M27). The K and V projections are physically smaller. KV cache shrinks proportionally — for GQA-3 here (12 query heads, 4 KV heads), cache is 3× smaller than full multi-head.</li>
  <li><strong>RoPE applied to Q and K, not V</strong>. V is positionally invariant — its values don't depend on where the token is. Common bug: applying RoPE to V too. Don't.</li>
  <li><strong>The KV cache in step 3</strong> is the inference-time path. <code>kv_cache</code> is <code>(K_cached, V_cached)</code> from previous decode steps; we concat new K/V to the cached ones. <em>This is exactly the buffer that vLLM (M27) pages</em>.</li>
  <li><strong><code>F.scaled_dot_product_attention</code> with <code>is_causal=True</code></strong>. PyTorch's high-level entry point picks FlashAttention as the backend (M25). The <code>is_causal=True</code> tells the kernel to skip K/V tiles above the diagonal — ~2× speedup vs explicit mask. <em>For decode (when kv_cache is provided), causality is automatic — Q is shape [B, H, 1, D], no mask needed.</em></li>
</ul>

<h3>5. SwiGLU MLP</h3>

<p>The modern gated MLP. <code>down(silu(gate(x)) * up(x))</code>. Three Linears instead of GELU's two, but the gate × up multiplicative interaction reliably outperforms simple GELU.</p>

<pre><code><span class="kw">class</span> <span class="ty">SwiGLU</span>(nn.Module):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, cfg):
        <span class="fn">super</span>().<span class="fn">__init__</span>()
        self.gate_proj = nn.<span class="fn">Linear</span>(cfg.d_model, cfg.d_mlp, bias=<span class="kw">False</span>)
        self.up_proj   = nn.<span class="fn">Linear</span>(cfg.d_model, cfg.d_mlp, bias=<span class="kw">False</span>)
        self.down_proj = nn.<span class="fn">Linear</span>(cfg.d_mlp, cfg.d_model, bias=<span class="kw">False</span>)

    <span class="kw">def</span> <span class="fn">forward</span>(self, x):
        <span class="com"># Gated path: silu(gate) * up</span>
        <span class="kw">return</span> self.<span class="fn">down_proj</span>(F.<span class="fn">silu</span>(self.<span class="fn">gate_proj</span>(x)) * self.<span class="fn">up_proj</span>(x))</code></pre>

<p>Why SwiGLU over plain GELU MLP? Empirically: ~0.5-1% perplexity improvement at the same parameter count. The gate × up product introduces a multiplicative non-linearity that GELU can't replicate. Cost: one extra matmul (gate_proj). Worth it.</p>

<p>Why <code>silu</code> specifically? <code>silu(x) = x * sigmoid(x)</code> — also called <code>swish</code>. Smooth, non-monotonic, similar to GELU's shape. The "Swi" in SwiGLU is from "Swish."</p>

<h3>6. Transformer block</h3>

<p>Pre-norm + attn + residual; pre-norm + mlp + residual.</p>

<pre><code><span class="kw">class</span> <span class="ty">Block</span>(nn.Module):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, cfg):
        <span class="fn">super</span>().<span class="fn">__init__</span>()
        self.norm1 = <span class="fn">RMSNorm</span>(cfg.d_model, cfg.norm_eps)
        self.attn  = <span class="fn">Attention</span>(cfg)
        self.norm2 = <span class="fn">RMSNorm</span>(cfg.d_model, cfg.norm_eps)
        self.mlp   = <span class="fn">SwiGLU</span>(cfg)

    <span class="kw">def</span> <span class="fn">forward</span>(self, x, cos, sin, kv_cache=<span class="kw">None</span>):
        <span class="com"># Pre-norm: norm BEFORE the sub-layer. Residual ADDS the unnormed x.</span>
        attn_out, kv_cache_out = self.<span class="fn">attn</span>(self.<span class="fn">norm1</span>(x), cos, sin, kv_cache)
        x = x + attn_out
        x = x + self.<span class="fn">mlp</span>(self.<span class="fn">norm2</span>(x))
        <span class="kw">return</span> x, kv_cache_out</code></pre>

<p>The pre-norm structure is the simplest part of the model and also the most-debated detail historically. The original Transformer was post-norm (<code>norm(x + sublayer(x))</code>) and required learning-rate warmup to train stably. Pre-norm trains stably from step 0 because gradients flow unobstructed through the residual path. <em>If you remember one fact about transformer architecture, remember pre-norm.</em></p>

<h3>7. The full model</h3>

<p>Embedding + N blocks + final norm + lm_head.</p>

<pre><code><span class="kw">class</span> <span class="ty">GPT</span>(nn.Module):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, cfg):
        <span class="fn">super</span>().<span class="fn">__init__</span>()
        self.cfg = cfg
        self.embed = nn.<span class="fn">Embedding</span>(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.<span class="fn">ModuleList</span>([<span class="fn">Block</span>(cfg) <span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(cfg.n_layer)])
        self.final_norm = <span class="fn">RMSNorm</span>(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.<span class="fn">Linear</span>(cfg.d_model, cfg.vocab_size, bias=<span class="kw">False</span>)

        <span class="com"># Tied embeddings: lm_head shares weight tensor with input embedding.</span>
        <span class="kw">if</span> cfg.tied_embeds:
            self.lm_head.weight = self.embed.weight

        <span class="com"># Precompute RoPE cache once (M30).</span>
        cos, sin = <span class="fn">build_rope_cache</span>(cfg.max_seq_len, cfg.head_dim, cfg.rope_base)
        self.<span class="fn">register_buffer</span>(<span class="str">"cos"</span>, cos, persistent=<span class="kw">False</span>)
        self.<span class="fn">register_buffer</span>(<span class="str">"sin"</span>, sin, persistent=<span class="kw">False</span>)

        <span class="com"># Init scheme (M8). std=0.02 standard for GPT-style models.</span>
        self.<span class="fn">apply</span>(self._init_weights)

    <span class="kw">def</span> <span class="fn">_init_weights</span>(self, m):
        <span class="kw">if</span> <span class="fn">isinstance</span>(m, nn.Linear):
            torch.nn.init.<span class="fn">normal_</span>(m.weight, mean=<span class="num">0.0</span>, std=self.cfg.init_std)
        <span class="kw">elif</span> <span class="fn">isinstance</span>(m, nn.Embedding):
            torch.nn.init.<span class="fn">normal_</span>(m.weight, mean=<span class="num">0.0</span>, std=self.cfg.init_std)

    <span class="kw">def</span> <span class="fn">forward</span>(self, token_ids, kv_caches=<span class="kw">None</span>, start_pos=<span class="num">0</span>):
        B, T = token_ids.shape
        x = self.<span class="fn">embed</span>(token_ids)                <span class="com"># [B, T, D]</span>

        <span class="com"># Slice RoPE cache for this segment.</span>
        cos = self.cos[start_pos : start_pos + T]
        sin = self.sin[start_pos : start_pos + T]

        <span class="com"># Run through each block, threading the KV cache.</span>
        new_kv_caches = []
        <span class="kw">for</span> i, block <span class="kw">in</span> <span class="fn">enumerate</span>(self.blocks):
            kv_in = kv_caches[i] <span class="kw">if</span> kv_caches <span class="kw">is</span> <span class="kw">not</span> <span class="kw">None</span> <span class="kw">else</span> <span class="kw">None</span>
            x, kv_out = <span class="fn">block</span>(x, cos, sin, kv_in)
            new_kv_caches.<span class="fn">append</span>(kv_out)

        x = self.<span class="fn">final_norm</span>(x)
        logits = self.<span class="fn">lm_head</span>(x)                  <span class="com"># [B, T, vocab_size]
</span>
        <span class="kw">return</span> logits, new_kv_caches</code></pre>

<p>That's the whole model. ~80 lines for the full GPT class plus its components, ~200 lines counting Config and the helpers. It works.</p>

<h3>The init scheme matters more than you'd think</h3>

<p>One detail on init: <code>std=0.02</code> across all linears is the GPT-style default. It's a reasonable starting point, but at depth, you typically want to <em>scale down</em> the init for the residual-output projections (<code>o_proj</code> in attention, <code>down_proj</code> in MLP) by <code>1/sqrt(2 * n_layer)</code>. This keeps the residual stream's variance from blowing up as depth accumulates contributions from each block.</p>

<pre><code><span class="kw">def</span> <span class="fn">_init_weights</span>(self, m):
    <span class="kw">if</span> <span class="fn">isinstance</span>(m, nn.Linear):
        torch.nn.init.<span class="fn">normal_</span>(m.weight, std=self.cfg.init_std)
    <span class="kw">elif</span> <span class="fn">isinstance</span>(m, nn.Embedding):
        torch.nn.init.<span class="fn">normal_</span>(m.weight, std=self.cfg.init_std)

<span class="com"># After the apply, scale residual-output projections (M8 redux).</span>
<span class="kw">for</span> name, p <span class="kw">in</span> self.<span class="fn">named_parameters</span>():
    <span class="kw">if</span> name.<span class="fn">endswith</span>(<span class="str">"o_proj.weight"</span>) <span class="kw">or</span> name.<span class="fn">endswith</span>(<span class="str">"down_proj.weight"</span>):
        <span class="kw">with</span> torch.<span class="fn">no_grad</span>():
            p.<span class="fn">mul_</span>(<span class="num">1.0</span> / <span class="fn">math.sqrt</span>(<span class="num">2</span> * self.cfg.n_layer))</code></pre>

<p>This is the <em>actually-correct</em> GPT-2 init. Skipping it gives slightly worse loss curves at depth ≥ 24. Most modern codebases include it.</p>

<h2>The training loop</h2>

<p>From M11, with bf16 mixed precision (M14):</p>

<pre><code>cfg = <span class="fn">GPTConfig</span>()
model = <span class="fn">GPT</span>(cfg).<span class="fn">cuda</span>().<span class="fn">to</span>(torch.bfloat16)
optimizer = torch.optim.<span class="fn">AdamW</span>(model.<span class="fn">parameters</span>(), lr=<span class="num">3e-4</span>, betas=(<span class="num">0.9</span>, <span class="num">0.95</span>),
                              weight_decay=<span class="num">0.1</span>, fused=<span class="kw">True</span>)
scheduler = torch.optim.lr_scheduler.<span class="fn">CosineAnnealingLR</span>(optimizer, T_max=<span class="num">10000</span>)

<span class="kw">for</span> step, (x, y) <span class="kw">in</span> <span class="fn">enumerate</span>(train_loader):
    x, y = x.<span class="fn">cuda</span>(), y.<span class="fn">cuda</span>()
    logits, _ = <span class="fn">model</span>(x)                     <span class="com"># [B, T, vocab]</span>

    <span class="com"># Cross-entropy in fp32 for stability (M14).</span>
    loss = F.<span class="fn">cross_entropy</span>(logits.<span class="fn">flatten</span>(<span class="num">0</span>, <span class="num">1</span>).<span class="fn">float</span>(), y.<span class="fn">flatten</span>())

    optimizer.<span class="fn">zero_grad</span>()
    loss.<span class="fn">backward</span>()
    torch.nn.utils.<span class="fn">clip_grad_norm_</span>(model.<span class="fn">parameters</span>(), <span class="num">1.0</span>)
    optimizer.<span class="fn">step</span>()
    scheduler.<span class="fn">step</span>()

    <span class="kw">if</span> step % <span class="num">100</span> == <span class="num">0</span>:
        <span class="com"># tokens/sec — the canonical training throughput metric.</span>
        toks_per_sec = (x.<span class="fn">numel</span>() * <span class="num">100</span>) / (time.<span class="fn">time</span>() - last_t)
        <span class="fn">print</span>(<span class="fn">f</span><span class="str">"step {step} loss {loss:.4f} tok/s {toks_per_sec:.0f}"</span>)
        last_t = time.<span class="fn">time</span>()</code></pre>

<p>Standard pieces: AdamW with weight decay 0.1 and β₂=0.95 (the GPT-3 / Llama default — slightly more conservative than the AdamW default β₂=0.999), cosine schedule (M9), gradient clipping at 1.0 (M9), bf16 cast on the model. The loss is computed in fp32 (<code>logits.float()</code>) — same recipe as M14.</p>

<p>The <code>tokens/sec</code> log is the throughput metric you'll see most. For a GPT-2-small on an H100 in bf16, expect ~500K tokens/sec; for a 7B Llama on an H100, ~10K tokens/sec; with FSDP across 8 H100s on a 70B Llama, ~3K tokens/sec.</p>

<h2>KV cache wiring for inference</h2>

<p>Generation uses a different path. The first call processes the prompt (prefill); subsequent calls pass the previous KV cache and one new token at a time (decode). Same model, two different access patterns — exactly what M27 described.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 320" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrK" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#c1502e"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">KV cache shape evolution: prefill writes T positions, decode appends 1 per step</text>

  <!-- Prefill state -->
  <text x="20" y="55" font-size="12" font-weight="700" fill="#1f5f5b">① Prefill: process the prompt — one big forward over all T_prompt tokens</text>
  <g transform="translate(20, 65)">
    <rect x="0" y="0" width="300" height="50" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="150" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">model(prompt_ids, kv_caches=None)</text>
    <text x="150" y="38" text-anchor="middle" font-size="9" fill="#1a1612">writes K, V for positions 0..T_prompt−1</text>
    <text x="350" y="20" font-size="11" fill="#1a1612">→ KV cache: <tspan font-family="'IBM Plex Mono', monospace" font-weight="700">[L, 2, B, KH, T_prompt, D]</tspan></text>
    <text x="350" y="38" font-size="9" fill="#6b5d4f">L = num_layers, 2 = (K, V), KH = n_kv_heads</text>
  </g>

  <!-- Decode steps -->
  <text x="20" y="145" font-size="12" font-weight="700" fill="#c1502e">② Decode: generate one token at a time, passing previous cache + 1 new token</text>
  <g transform="translate(20, 155)">
    <rect x="0"   y="0" width="115" height="40" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="57"  y="18" text-anchor="middle" font-size="9" fill="#1a1612">step 1: 1 new tok</text>
    <text x="57"  y="32" text-anchor="middle" font-size="9" fill="#1a1612">cache: T_p+1</text>

    <path d="M 120 20 L 145 20" stroke="#c1502e" stroke-width="1.5" fill="none" marker-end="url(#arrK)"/>

    <rect x="150" y="0" width="115" height="40" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="207" y="18" text-anchor="middle" font-size="9" fill="#1a1612">step 2: 1 new tok</text>
    <text x="207" y="32" text-anchor="middle" font-size="9" fill="#1a1612">cache: T_p+2</text>

    <path d="M 270 20 L 295 20" stroke="#c1502e" stroke-width="1.5" fill="none" marker-end="url(#arrK)"/>

    <rect x="300" y="0" width="115" height="40" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="357" y="18" text-anchor="middle" font-size="9" fill="#1a1612">step 3: 1 new tok</text>
    <text x="357" y="32" text-anchor="middle" font-size="9" fill="#1a1612">cache: T_p+3</text>

    <text x="430" y="22" font-size="11" fill="#6b5d4f">…</text>

    <rect x="500" y="0" width="115" height="40" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="557" y="18" text-anchor="middle" font-size="9" fill="#1a1612">step k: 1 new tok</text>
    <text x="557" y="32" text-anchor="middle" font-size="9" fill="#1a1612">cache: T_p+k</text>
  </g>

  <!-- Memory growth -->
  <text x="20" y="240" font-size="12" font-weight="700" fill="#1a1612">Memory cost per layer per token (Llama-style 7B with GQA):</text>
  <text x="40" y="258" font-size="10" fill="#1a1612">  2 (K,V) × 4 (KH) × 128 (D) × 2 bytes (bf16) = 2 KB per layer per token</text>
  <text x="40" y="272" font-size="10" fill="#1a1612">  × 32 layers = 64 KB/token</text>
  <text x="40" y="286" font-size="10" fill="#c1502e">  → for 32K context: ~2 GB just for KV cache (one request)</text>

  <text x="370" y="312" font-family="'Caveat', cursive" font-size="20" fill="#c1502e" text-anchor="middle">this is exactly the buffer vLLM (M27) splits into 16-token pages</text>
</svg>
</div>

<p>The key shape detail: the KV cache for one request is <code>[L, 2, B, KH, T, D]</code> (per-layer × K-or-V × batch × kv-heads × time × head-dim). T grows as decode progresses. The <em>total</em> cache size scales linearly with conversation length — exactly the inference pain point M27 addressed via paging.</p>

<p>The generation loop:</p>

<pre><code><span class="kw">@</span>torch.<span class="fn">no_grad</span>()
<span class="kw">def</span> <span class="fn">generate</span>(model, prompt_ids, max_new_tokens=<span class="num">100</span>,
             temperature=<span class="num">1.0</span>, top_k=<span class="num">50</span>):
    cfg = model.cfg
    B, T = prompt_ids.shape
    device = prompt_ids.device

    <span class="com"># 1. Prefill — process the prompt all at once.</span>
    logits, kv_caches = <span class="fn">model</span>(prompt_ids)
    next_logits = logits[:, -<span class="num">1</span>, :]      <span class="com"># [B, vocab] — only the last position matters</span>

    generated = [prompt_ids]
    <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(max_new_tokens):
        <span class="com"># 2. Sample next token.</span>
        next_id = <span class="fn">sample</span>(next_logits, temperature, top_k)
        generated.<span class="fn">append</span>(next_id)

        <span class="com"># 3. Decode step: pass last token + previous cache.</span>
        start_pos = T + i
        logits, kv_caches = <span class="fn">model</span>(next_id, kv_caches=kv_caches, start_pos=start_pos)
        next_logits = logits[:, -<span class="num">1</span>, :]

    <span class="kw">return</span> torch.<span class="fn">cat</span>(generated, dim=<span class="num">1</span>)


<span class="kw">def</span> <span class="fn">sample</span>(logits, temperature, top_k):
    <span class="com"># Apply temperature.</span>
    logits = logits / <span class="fn">max</span>(temperature, <span class="num">1e-5</span>)
    <span class="com"># Top-k filter.</span>
    <span class="kw">if</span> top_k <span class="kw">is</span> <span class="kw">not</span> <span class="kw">None</span>:
        v, _ = torch.<span class="fn">topk</span>(logits, top_k)
        logits[logits &lt; v[:, -<span class="num">1</span>:]] = -<span class="fn">float</span>(<span class="str">"inf"</span>)
    <span class="com"># Sample from the resulting categorical.</span>
    probs = F.<span class="fn">softmax</span>(logits, dim=-<span class="num">1</span>)
    <span class="kw">return</span> torch.<span class="fn">multinomial</span>(probs, num_samples=<span class="num">1</span>)</code></pre>

<p>Three sampling knobs to expose:</p>

<ul>
  <li><strong>Temperature</strong>: divides logits before softmax. <code>T=1.0</code> is the natural distribution; <code>T&lt;1</code> sharpens (greedy at <code>T→0</code>); <code>T&gt;1</code> flattens (more creative / chaotic).</li>
  <li><strong>Top-k</strong>: keep only the k highest-logit tokens, set the rest to -∞. Prevents the model from sampling absurd low-probability tokens.</li>
  <li><strong>Top-p (nucleus)</strong> (not shown here): keep the smallest set of tokens whose cumulative probability exceeds p. More flexible than top-k for varying-entropy distributions.</li>
</ul>

<p>Production setups expose all three plus repetition penalties; vLLM's <code>SamplingParams</code> (M27) covers them. Same logic, packaged for serving.</p>

<h2>Debugging your first run: failures in order of likelihood</h2>

<p>You will run this and something will go wrong. Here's the diagnostic checklist, in the order issues appear:</p>

<div class="table-wrap">
<table>
<caption>First-run failures and how to spot them</caption>
<thead><tr><th>Symptom</th><th>Likely cause</th><th>Fix (reference)</th></tr></thead>
<tbody>
<tr><td>NaN in loss at step 1</td><td>fp16 instead of bf16, or accumulator dtype wrong</td><td>Use bf16 not fp16; cast loss computation to fp32 (M14)</td></tr>
<tr><td>NaN appears after a few hundred steps</td><td>LR too high; init too aggressive</td><td>Lower LR to 1e-4; verify init scheme; clip grads (M8, M9)</td></tr>
<tr><td>Loss decreases for the first ~100 steps then plateaus near random</td><td>Causal mask bug (model can see the future)</td><td>Verify <code>is_causal=True</code> in SDPA call; check generation produces sensible token at position 0 vs T (M25)</td></tr>
<tr><td>Loss is exactly <code>log(vocab_size)</code> and never moves</td><td>Forward returning zeros; LM head weight tied wrong</td><td>Check <code>lm_head.weight is embed.weight</code> if tied; verify init isn't zero</td></tr>
<tr><td>Generation produces gibberish but loss looks fine</td><td>Tokenizer mismatch (training vs inference); positional encoding bug</td><td>Verify same tokenizer round-trips; check RoPE applied (M30)</td></tr>
<tr><td>OOM during prefill but training works</td><td>FlashAttention not picked up; falling back to math backend</td><td>Check head_dim ≤ 256 and is multiple of 8; bf16 not fp32; <code>sdpa_kernel</code> verify (M25)</td></tr>
<tr><td>Slower than expected</td><td>Wrong dtype, no torch.compile, dataloader bottleneck</td><td>bf16 weights; <code>torch.compile(model)</code>; check num_workers and pin_memory (M11, M21)</td></tr>
<tr><td>Generation is slower than training, per-token</td><td>Decode is memory-bound; expected!</td><td>Quantize weights (M26); paged KV (M27); add speculative decoding (M27)</td></tr>
</tbody>
</table>
</div>

<p>A practical reflex: <em>always sanity-check on a tiny model first</em>. Before running real training, build a 2-layer 64-dim model on 100 batches of random data. If loss decreases visibly within 100 steps, the architecture is wired correctly. If it doesn't, the bug is structural (wrong mask, wrong residual, wrong init), not a hyperparameter.</p>

<h2>Connecting back to the rest of the course</h2>

<p>Read the model code one more time and notice: every meaningful decision is a callback to a specific module.</p>

<div class="table-wrap">
<table>
<caption>Where each architectural decision came from in the course</caption>
<thead><tr><th>Decision</th><th>Module</th></tr></thead>
<tbody>
<tr><td>Vocab size 50304 (multiple of 64) for tensor cores</td><td>M22 — GPU programming model</td></tr>
<tr><td>Head dim 64 (multiple of 8)</td><td>M22 — tensor core dim alignment</td></tr>
<tr><td>RMSNorm with fp32 reduction</td><td>M14, M23 — mixed precision &amp; the kernel</td></tr>
<tr><td>RoPE applied in fp32</td><td>M14 — numerical stability</td></tr>
<tr><td>Pre-norm, not post-norm</td><td>M8 — init &amp; norms; depth stability</td></tr>
<tr><td>Bias=False on Linears</td><td>Standard since GPT-J; saves params for no quality cost</td></tr>
<tr><td>GQA (n_kv_head &lt; n_head)</td><td>M27 — KV cache reduction</td></tr>
<tr><td>SwiGLU MLP</td><td>Modern preference; gated path beats GELU</td></tr>
<tr><td>Tied embeddings</td><td>Param savings; M11 mentions it</td></tr>
<tr><td>Init std=0.02 with residual scaling 1/sqrt(2·n_layer)</td><td>M8 — init schemes</td></tr>
<tr><td>F.scaled_dot_product_attention with is_causal=True</td><td>M25 — FlashAttention via the front door</td></tr>
<tr><td>KV cache as <code>(K, V)</code> tuples per layer</td><td>M27 — the buffer vLLM pages</td></tr>
<tr><td>AdamW with β₂=0.95, weight decay 0.1, fused=True</td><td>M9 — optimizers</td></tr>
<tr><td>Cosine schedule + linear warmup</td><td>M9 — schedulers</td></tr>
<tr><td>bf16 weights, fp32 loss</td><td>M14 — mixed precision</td></tr>
<tr><td>Gradient clipping at 1.0</td><td>M9 — optimizer hygiene</td></tr>
<tr><td>tokens/sec as throughput metric</td><td>M11 — training-loop telemetry</td></tr>
</tbody>
</table>
</div>

<p>Every line traces back. <em>This is what mastery looks like</em>: not memorizing the architecture, but being able to derive each decision from a constraint or empirical finding you've internalized.</p>

<h2>Scaling up: what changes for a real model</h2>

<p>The code above trains a GPT-2-small. To go to a real production model (Llama-3-8B, say), the architecture barely changes. What changes is the engineering around it:</p>

<ul>
  <li><strong>Wider and deeper</strong>: <code>n_layer=32</code>, <code>d_model=4096</code>, <code>n_head=32</code>, <code>n_kv_head=8</code>, <code>d_mlp=14336</code>. Same code, bigger numbers.</li>
  <li><strong>Distributed training</strong>: FSDP (M17) for &gt;1 GPU, eventually TP (M18) for &gt;8 GPUs, eventually PP (M18) for multi-node. Code changes are at the wrapping layer, not in the model itself.</li>
  <li><strong>Activation checkpointing</strong> (M12): wrap each <code>Block</code> in <code>checkpoint</code>. Recomputes during backward to save memory.</li>
  <li><strong>Compile</strong>: <code>model = torch.compile(model)</code> (M21). For a 7B model, ~1.3-1.5× speedup.</li>
  <li><strong>Long context</strong>: increase <code>rope_base</code> from 10000 to 500000 for context extension to 32K+ (M30 covers this in detail). Possibly use sliding window or interleaved attention patterns for &gt;128K.</li>
  <li><strong>Real data pipeline</strong>: M10's DistributedSampler, real tokenization (tiktoken or sentencepiece), shuffled multi-file iteration with proper sharding.</li>
</ul>

<p>The model itself is unchanged. <em>Architecture is not where production complexity lives — it's in the engineering scaffolding.</em></p>

<div class="ndq">
<h4>About the build</h4>

<p class="q">Why is RoPE applied to Q and K but not V?</p>
<p class="a">Attention scores depend on Q · K^T. RoPE makes that dot product depend on (i − j), the relative position. V doesn't enter the score; it's just the value being weighted. Applying RoPE to V would rotate the values without any compensating un-rotation later — you'd just be scrambling the values. Common bug; the fix is to never apply RoPE to V.</p>

<p class="q">Why is the lm_head a linear with no bias, and why tied to the embedding?</p>
<p class="a">No bias because biases on output projections don't help language models — empirically and theoretically (a uniform bias just adds the same scalar to all vocab logits, which softmax cancels out). Tied because the embedding (vocab → D) and lm_head (D → vocab) are both shape-(vocab, D) matrices doing inverse-ish operations. Sharing the weight saves <code>vocab × D</code> parameters — for a 7B model with vocab=128K and D=4096, that's 524M params (~7% of the model). Not all models tie (Llama-3 stopped for the largest sizes); for small models, always tie.</p>

<p class="q">Why <code>is_causal=True</code> for prefill but <code>False</code> for decode?</p>
<p class="a">In prefill, Q has shape <code>[B, H, T_prompt, D]</code> — multiple positions attending to each other. The causal mask prevents position i from attending to position j&gt;i. In decode, Q has shape <code>[B, H, 1, D]</code> — just one new token. There's only one query position; it attends to all cached K/V positions, all of which are at positions ≤ current. Causality is automatic; passing <code>is_causal=True</code> would (depending on the kernel version) try to apply a causal mask of shape (1, T_full) which is degenerate. <em>For the case <code>kv_cache is not None</code>, set is_causal=False.</em></p>

<p class="q">Why is <code>n_kv_head</code> 4 and not 1 (full MQA)?</p>
<p class="a">MQA (single KV head shared across all query heads) saves the most KV cache memory but sacrifices some quality. GQA with a small number of KV heads (4-8) is the sweet spot — most of the memory benefit, almost none of the quality cost. Llama-3-8B uses 8 KV heads with 32 query heads; ours is 4 with 12. The ratio matters more than the absolute number.</p>

<p class="q">My loss looks fine but generation is gibberish. What's likely wrong?</p>
<p class="a">Three most common: (1) Tokenizer mismatch — training tokenized with one BPE, generation with a different one. Verify <code>tokenizer.decode(tokenizer.encode(text)) == text</code> for a sample. (2) RoPE positions not advancing during generation — the <code>start_pos</code> parameter must increment with each decode step. If you re-use position 0 for every token, attention is scrambled. (3) Causal mask issue at training, fine at generation — the model trained with future leakage, so it expects to see tokens after the current one. At generation, those tokens don't exist; outputs are garbage. Verify generation matches training time loss-on-full-prompt for the prefill portion.</p>

<p class="q">Why bf16 weights instead of fp32?</p>
<p class="a">bf16 has fp32-comparable range (M14) and 2× less memory. For inference, the speed and memory wins are huge. For training, you store an fp32 master copy in the optimizer (M12) for accurate updates while the forward/backward use bf16. <em>If you're storing a checkpoint, use bf16</em>; the fp32 master is a training-time-only artifact.</p>

<p class="q">When should I switch from this code to a library like <code>transformers</code>?</p>
<p class="a">For research where you control the architecture: the code above is great. For production where you want robust support for hundreds of model variants, tokenizers, generation strategies, and pre-trained checkpoints: use <code>transformers</code>. The code in M29 is exactly what's inside <code>transformers/models/llama/modeling_llama.py</code>, with more bells and whistles. <em>Knowing how it works is what lets you debug when the library has a bug or doesn't support what you need.</em></p>
</div>

<h2>Code Magnets: assemble the transformer block forward</h2>

<p>You're writing the forward pass of a transformer block (pre-norm + attention + residual; pre-norm + MLP + residual). Three magnets are wrong choices.</p>

<div class="magnets">
<p>Arrange the magnets into the correct forward.</p>

<div class="magnet-pool">
  <span class="magnet">def forward(self, x, cos, sin, kv_cache=None):</span>
  <span class="magnet">    attn_out, kv_cache_out = self.attn(self.norm1(x), cos, sin, kv_cache)</span>
  <span class="magnet">    attn_out, kv_cache_out = self.attn(x, cos, sin, kv_cache)</span>
  <span class="magnet">    x = x + attn_out</span>
  <span class="magnet">    x = self.norm1(x + attn_out)</span>
  <span class="magnet">    x = x + self.mlp(self.norm2(x))</span>
  <span class="magnet">    x = self.norm2(x + self.mlp(x))</span>
  <span class="magnet">    return x, kv_cache_out</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">def</span> <span class="fn">forward</span>(self, x, cos, sin, kv_cache=<span class="kw">None</span>):
    attn_out, kv_cache_out = self.<span class="fn">attn</span>(self.<span class="fn">norm1</span>(x), cos, sin, kv_cache)
    x = x + attn_out
    x = x + self.<span class="fn">mlp</span>(self.<span class="fn">norm2</span>(x))
    <span class="kw">return</span> x, kv_cache_out</code></pre>
<p>The traps:</p>
<ul>
  <li><code>attn_out, kv_cache_out = self.attn(x, cos, sin, kv_cache)</code>: passes the unnormed x to attention. <em>Pre-norm</em> means norm BEFORE the sub-layer; this version is post-norm-ish, breaks training stability at depth.</li>
  <li><code>x = self.norm1(x + attn_out)</code>: this is the original Transformer's post-norm. Trains poorly without warmup; not what modern transformers use.</li>
  <li><code>x = self.norm2(x + self.mlp(x))</code>: same post-norm bug for the MLP sub-block. Note also that the MLP input here would be unnormed — double bug.</li>
</ul>
<p>The pattern: <strong>norm BEFORE the sub-layer's input; residual ADDS the sub-layer's output to the unnormed x</strong>. Two sub-blocks (attn, mlp), two norms (norm1, norm2), two residuals.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each transformer component to its real role.</p>

<div class="match-grid">
  <div class="header">Component</div>
  <div class="header">Real role</div>

  <div>nn.Embedding</div>
  <div>A. Lookup table mapping token IDs to D-dim vectors; (vocab_size, D) shape.</div>

  <div>RMSNorm</div>
  <div>B. Per-token rescaling by RMS magnitude; faster than LayerNorm at same quality.</div>

  <div>RoPE</div>
  <div>C. Position-dependent rotation of Q and K; makes attention depend on (i − j).</div>

  <div>Causal mask (is_causal=True)</div>
  <div>D. Tells SDPA to skip K/V tiles above the diagonal; ~2× attention speedup.</div>

  <div>SwiGLU MLP</div>
  <div>E. Gated MLP: down(silu(gate(x)) · up(x)); ~0.5-1% perplexity win over GELU.</div>

  <div>KV cache</div>
  <div>F. Cached K/V per layer per request; grows with conversation length, paged in serving.</div>

  <div>Tied lm_head + embedding</div>
  <div>G. Shared weight for input lookup and output projection; saves vocab×D parameters.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>nn.Embedding</strong> → A<br>
<strong>RMSNorm</strong> → B<br>
<strong>RoPE</strong> → C<br>
<strong>Causal mask</strong> → D<br>
<strong>SwiGLU MLP</strong> → E<br>
<strong>KV cache</strong> → F<br>
<strong>Tied lm_head + embedding</strong> → G
</p>
<p>The mental shortcut: <em>embedding looks up, RMSNorm rescales, RoPE rotates, causal mask gates the future, SwiGLU gates the MLP, KV cache grows over time, tied embeddings save parameters</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team writes the transformer block as <code>x = self.norm(x + self.attn(x))</code>. Training is unstable past 12 layers; loss spikes that don't recover. What's wrong, and what's the fix?</p>
<details class="answer"><summary>show answer</summary>
<p>That's post-norm: applying the norm to the residual sum. The original Transformer used this; modern transformers (GPT-2 onward) use pre-norm. The instability is the well-known post-norm-at-depth issue: gradients flow through the norm operator before reaching the sub-layer's input, accumulating norm-induced rescaling that interferes with the residual stream.</p>
<p>Fix: switch to pre-norm — <code>x = x + self.attn(self.norm(x))</code>. Now the norm only sees the path going into the sub-layer; the residual path is untouched. Trains stably from step 0. <strong>Pre-norm is one of the most reliable architectural choices in modern deep learning.</strong> If a paper proposes post-norm, it almost always also proposes the warmup or careful-init regime needed to make it work — the variants are not interchangeable.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Why does the GPT class register <code>cos</code> and <code>sin</code> as buffers (not parameters), and why <code>persistent=False</code>?</p>
<details class="answer"><summary>show answer</summary>
<p>They're <em>not parameters</em>: they're not learned. They're computed once at init from <code>build_rope_cache</code>. Registering them as buffers (M7) means they're moved with the model (<code>.cuda()</code>, <code>.to(dtype)</code>) but not seen by the optimizer. Calling them parameters would either crash (the optimizer would try to update them and break the deterministic computation) or, with <code>requires_grad=False</code>, work but be confusing.</p>
<p><code>persistent=False</code> means they're not included in <code>state_dict()</code> — when you save a checkpoint, cos and sin aren't saved. They're recomputed at model init from the config, so saving them is wasteful (they're tens of MB for long context) and brittle (re-using the checkpoint with a different max_seq_len or rope_base would silently use the old, wrong values). <strong>Computed-from-config tensors should be persistent=False buffers.</strong></p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> A team's GPT trains and the loss curve looks healthy, but generated text is repetitive — the model loops on phrases. What's likely going on, and what's the fix?</p>
<details class="answer"><summary>show answer</summary>
<p>This is usually a generation-time issue, not a training issue. The most common cause: <strong>greedy or low-temperature sampling on a model trained on diverse data</strong>. The model's logit distribution has a clear winner at each step that ends up cycling — "the cat sat on the mat. The cat sat on the mat. ..." The model isn't broken; the sampling is too deterministic.</p>
<p>Fixes, in order of typical efficacy: (1) Increase temperature to 0.7-0.9 (creative tasks) or use top-p sampling with p=0.9. (2) Add a repetition penalty: scale down logits for tokens that appeared recently (typical penalty 1.1-1.3). (3) If repetition still appears, the issue might be undertraining or low-quality training data — but verify the sampling first.</p>
<p>Independent check: at training-time-equivalent temperature (whatever the loss was computed at), generation should be reasonable. If repetition appears at temperature=0.7+ and increases sharply at lower temperatures, sampling is the issue.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> Sketch the difference in computation between prefill (one big forward over T prompt tokens) and decode (one forward per generated token). Why is decode slower per token?</p>
<details class="answer"><summary>show answer</summary>
<p>Prefill: one forward processes [B, T_prompt, D] activations. Each matmul is shape <code>[B*T_prompt, K] × [K, M]</code> — large rectangular matmuls that saturate tensor cores. The whole prompt costs the price of one forward.</p>
<p>Decode: one forward processes [B, 1, D] for the new token, but reads <em>all</em> of the cumulative KV cache during attention. Per-step matmuls (Q/K/V projections, MLP) are shape <code>[B*1, K] × [K, M]</code> — extremely tall-and-skinny. Attention is <code>[B, H, 1, D] × [B, H, T_full, D]</code> for the QK matmul.</p>
<p>Why decode is slower per token: the matmul <em>compute</em> is tiny but the <em>weight loads</em> are the same as in prefill (you still have to read every weight matrix from HBM). For decode, this means weight bandwidth dominates and tensor cores are idle — exactly the memory-bound regime from M22's roofline. The fix is everything in M27: quantization (smaller weights), continuous batching (more useful work per weight load), speculative decoding (multiple tokens per forward).</p>
<p>Concrete numbers for a 7B model on H100: prefill processes ~50K tokens/sec; decode produces ~70 tokens/sec at batch 1. Per-token, prefill is ~700× faster. <em>This 700× gap is the entire reason inference systems exist as their own engineering discipline</em>.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>A GPT-style transformer is <strong>embedding → N pre-norm transformer blocks → final norm → lm_head</strong>. Each block has attention + residual + MLP + residual.</li>
  <li><strong>Pre-norm</strong> (<code>x + sublayer(norm(x))</code>) is the modern default. Post-norm is the original; trains worse at depth.</li>
  <li><strong>RMSNorm</strong> instead of LayerNorm — drops mean subtraction, ~30% faster, same quality. <strong>SwiGLU</strong> instead of GELU MLP — gated path, ~0.5-1% perplexity win.</li>
  <li><strong>GQA</strong> (n_kv_head &lt; n_head) shrinks the KV cache without quality loss. <strong>Tied embeddings</strong> save vocab×D params.</li>
  <li><strong>RoPE</strong> for positional encoding (M30 covers in depth). Applied to Q and K only; in fp32 for stability.</li>
  <li>Attention uses <strong><code>F.scaled_dot_product_attention(..., is_causal=True)</code></strong> — picks FlashAttention (M25) when applicable. Causal masking is free at the kernel level.</li>
  <li>Many small choices trace back to specific modules: vocab=50304 for tensor-core alignment (M22), bias=False for parameter efficiency, RMSNorm cast to fp32 for numerical stability (M14), residual init scaling 1/sqrt(2·n_layer) for depth stability (M8).</li>
  <li>Training: <strong>AdamW (β₂=0.95, weight_decay=0.1, fused), cosine schedule, gradient clipping at 1.0, bf16 weights with fp32 loss</strong>.</li>
  <li>The KV cache is <code>[L, 2, B, KH, T, D]</code> — per-layer (K, V) tuples. <strong>Prefill writes T positions in one shot; decode appends 1 per step</strong>. This is exactly what vLLM (M27) pages.</li>
  <li>Generation: prefill processes the prompt; decode generates tokens one at a time, threading the KV cache. <strong>Decode is ~700× slower per token than prefill</strong> — entirely because of the memory-bound regime (M22, M27).</li>
  <li>Sampling: temperature, top-k, top-p, repetition penalty. Deterministic at temperature → 0; more creative as temperature rises.</li>
  <li><strong>Most first-run failures fall into a small set</strong>: NaN (mixed-precision setup), causal mask bug (model cheats), tokenizer mismatch (gibberish despite good loss), wrong RoPE position threading at generation. Walk the table top-to-bottom.</li>
  <li><strong>Architecture is not where production complexity lives.</strong> The model itself is ~250 lines. The complexity is in distributed training (M16-M18), memory management (M12), serving (M27), kernels (M22-M25). The architecture is the easy part once you've internalized the pieces.</li>
  <li>The reflex from this module: when you read a paper proposing an architecture change, ask "which module's content does this conflict with?" Most proposed changes either revisit a settled question (post-norm is back!) or introduce a real win in a specific regime. Knowing the components lets you tell which is which.</li>
</ul>
</div>

<p>Module 30 takes RoPE — which we used here without much explanation — and unpacks it properly. The math behind why pair-wise rotation gives relative-position attention; the choices of θ-base; the recipes for context extension (NTK, YARN, Dynamic NTK); and how RoPE interacts with the attention kernel. After M30, M31 covers post-training (SFT, RLHF, DPO), M32 consolidates debugging skills, and M33 explores Mamba as a transformer alternative.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">29</span>
  <span>Building a transformer from scratch</span>
</div>
"""

emit("29_build_transformer", "Module 29 — Building a transformer from scratch", BODY)
