#!/usr/bin/env python3
"""Module 30: RoPE and positional encodings — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part IX · Module 30</div>
  <h1 class="module-title">RoPE and <em>positional encodings</em></h1>
  <p class="module-sub">— why attention needs position information, the four families that solve it, and how RoPE became the universal default through one elegant property: <code>(Qᵢ · Kⱼ)</code> depends only on <code>i − j</code></p>
</div>

<p>In M29 we built a transformer and used Rotary Position Embedding without much explanation. This module unpacks it. Why does attention need position information at all? Why has rotation specifically — not addition, not learned vectors, not biases — become the universal default in modern LLMs? And how do you extend a model trained at 4K context to 32K, 128K, or 1M tokens at inference time?</p>

<p>The answers form a small but tight body of knowledge. By the end you'll understand why RoPE works, the design space it sits in, and the practical recipes (NTK-aware interpolation, YARN, Dynamic NTK) that let production models stretch their context to where the training data didn't reach.</p>

<div class="keyidea">
Attention without positional encoding is <em>set-equivariant</em>: shuffle the input tokens and the output shuffles correspondingly. To make sequence position matter, you have to inject it somewhere. <strong>RoPE</strong> (Rotary Position Embedding) does this by rotating Q and K vectors by an angle that depends on token position. The clever bit: when you take the dot product Q · K, the rotation angles subtract — <strong>the score depends only on the <em>relative</em> position <code>i − j</code></strong>, not on absolute positions. This is the property modern transformers want, and rotation is the unique way to get it without modifying the dot-product structure. The frequency schedule (<code>θ_base = 10000</code>) determines the range over which the model can distinguish positions; <strong>raising θ_base extends the model's effective context</strong>, which is what NTK-aware scaling does for long-context fine-tuning.
</div>

<h2>One new face</h2>

<div class="character" style="--c: #1f5f5b;">
  <div class="avatar" style="background: #1f5f5b; color: #fff;">⊙</div>
  <div>
    <p class="who">RoPE</p>
    <p class="name">"I rotate. I'm not added; I'm not concatenated; I rotate Q and K by position-dependent angles."</p>
    <p class="says">For every pair of dimensions in your Q vector, I rotate them in the 2D plane by an angle proportional to the token's position. Same for K. Different pairs of dimensions rotate at different frequencies — high frequency for early dims, low frequency for late dims. <em>The dot product Qᵢ · Kⱼ then depends only on i − j</em>, because the rotations partially cancel. I'm parameter-free, I extrapolate gracefully, and I leave the attention kernel completely unchanged. <strong>I'm what every modern LLM uses.</strong> The catch: my frequency schedule decides the position range I can distinguish, so extending context length needs me re-tuned (NTK, YARN, Dynamic NTK) — the practical recipes you'll meet later in this module.</p>
  </div>
</div>

<h2>Why position matters</h2>

<p>Think about what attention does without position information. For tokens with embeddings <code>x₁, x₂, ..., xₜ</code>, the attention output for position i is:</p>

<pre><code>Output_i = Σⱼ softmax((Qᵢ · Kⱼ) / √d) · Vⱼ</code></pre>

<p>Now permute the input: tokens arrive as <code>x₃, x₁, x₂</code> instead of <code>x₁, x₂, x₃</code>. The Q, K, V values get permuted correspondingly. The dot products are the same (just paired differently), the softmax is over the same set of values, and the output for what was originally position 1 is identical regardless of where in the sequence it now sits. <strong>Attention treats its input as a set, not a sequence.</strong></p>

<p>This is fatal for language. "Dog bites man" and "Man bites dog" have the same token set but radically different meanings. To make attention sequence-aware, position information must enter somewhere — either added to the embeddings, baked into the attention weights, or applied to Q and K directly.</p>

<p>The four families of solutions:</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 400" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Four families of positional encoding (shown for one head)</text>

  <!-- Family 1: Absolute learned -->
  <g transform="translate(20, 50)">
    <rect x="0" y="0" width="340" height="160" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="170" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">① Absolute learned (BERT, GPT-2)</text>
    <text x="10" y="44" font-size="10" fill="#1a1612">x_i = token_emb[t_i] + pos_emb[i]</text>
    <text x="10" y="60" font-size="10" fill="#1a1612">  (separate (max_len, D) learned table)</text>

    <text x="10" y="85" font-size="10" font-weight="700" fill="#1a1612">Pros:</text>
    <text x="10" y="100" font-size="9" fill="#1a1612">• Simple, learned end-to-end</text>
    <text x="10" y="115" font-size="9" fill="#1a1612">• No architectural changes to attention</text>

    <text x="10" y="135" font-size="10" font-weight="700" fill="#c1502e">Cons:</text>
    <text x="10" y="150" font-size="9" fill="#1a1612">• Cannot extrapolate past max trained position</text>
    <text x="200" y="150" font-size="9" fill="#1a1612">• Adds (max_len × D) params</text>
  </g>

  <!-- Family 2: Sinusoidal -->
  <g transform="translate(380, 50)">
    <rect x="0" y="0" width="340" height="160" fill="#fff8a8" stroke="#1a1612" stroke-width="2"/>
    <text x="170" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">② Sinusoidal (original Transformer)</text>
    <text x="10" y="44" font-size="10" fill="#1a1612">x_i = token_emb[t_i] + sin/cos(i / 10000^(2k/D))</text>
    <text x="10" y="60" font-size="10" fill="#1a1612">  (deterministic, multi-frequency)</text>

    <text x="10" y="85" font-size="10" font-weight="700" fill="#1a1612">Pros:</text>
    <text x="10" y="100" font-size="9" fill="#1a1612">• Parameter-free</text>
    <text x="10" y="115" font-size="9" fill="#1a1612">• Some extrapolation in theory</text>

    <text x="10" y="135" font-size="10" font-weight="700" fill="#c1502e">Cons:</text>
    <text x="10" y="150" font-size="9" fill="#1a1612">• Extrapolation poor in practice</text>
    <text x="200" y="150" font-size="9" fill="#1a1612">• Position is absolute, not relative</text>
  </g>

  <!-- Family 3: Relative bias / ALiBi -->
  <g transform="translate(20, 230)">
    <rect x="0" y="0" width="340" height="160" fill="#fff5d8" stroke="#d4a017" stroke-width="2"/>
    <text x="170" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">③ Relative position bias (T5, ALiBi)</text>
    <text x="10" y="44" font-size="10" fill="#1a1612">attn[i, j] += b(i − j)  /* learned or fixed */</text>
    <text x="10" y="60" font-size="10" fill="#1a1612">  (modify attention scores directly)</text>

    <text x="10" y="85" font-size="10" font-weight="700" fill="#1a1612">Pros:</text>
    <text x="10" y="100" font-size="9" fill="#1a1612">• Position is naturally relative</text>
    <text x="10" y="115" font-size="9" fill="#1a1612">• ALiBi extrapolates well (linear bias)</text>

    <text x="10" y="135" font-size="10" font-weight="700" fill="#c1502e">Cons:</text>
    <text x="10" y="150" font-size="9" fill="#1a1612">• T5 form: large learned bias table</text>
    <text x="200" y="150" font-size="9" fill="#1a1612">• Mediocre quality at long context</text>
  </g>

  <!-- Family 4: Rotary (RoPE) -->
  <g transform="translate(380, 230)">
    <rect x="0" y="0" width="340" height="160" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="3"/>
    <text x="170" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1f5f5b">④ Rotary (RoPE) — the modern default ★</text>
    <text x="10" y="44" font-size="10" fill="#1a1612">Rotate Q_i, K_j by angles θ_i, θ_j;</text>
    <text x="10" y="60" font-size="10" fill="#1a1612">  result: Q · K depends only on (i − j)</text>

    <text x="10" y="85" font-size="10" font-weight="700" fill="#1a1612">Pros:</text>
    <text x="10" y="100" font-size="9" fill="#1a1612">• Parameter-free, kernel-friendly</text>
    <text x="10" y="115" font-size="9" fill="#1a1612">• Relative-position via construction</text>
    <text x="10" y="130" font-size="9" fill="#1a1612">• Extends to long context with NTK/YARN</text>

    <text x="10" y="148" font-size="10" font-weight="700" fill="#1f5f5b">Used by: Llama, Mistral, Qwen, Gemma, DeepSeek</text>
  </g>
</svg>
</div>

<h2>The mathematical setup</h2>

<p>What property do we want? In an attention layer, the score for token i attending to token j is <code>Qᵢ · Kⱼ</code>. We want this to depend on <em>where</em> tokens are relative to each other — specifically, we want:</p>

<pre><code>Q_i · K_j = f(x_i, x_j, i − j)</code></pre>

<p>The score should depend on the content of the tokens (<code>x_i</code>, <code>x_j</code>) and on their <em>relative</em> position <code>i − j</code>, but <em>not</em> on the absolute positions <code>i</code> and <code>j</code> separately. Why relative? Two reasons. (1) Empirically: language structure is mostly local — what matters is "the verb is two tokens before this noun," not "this verb is at position 1572." (2) Generalization: a model that learns about relative positions extrapolates to longer sequences than one that learned about specific absolute positions.</p>

<p>RoPE's insight: <strong>rotation is the unique transformation that achieves this property without changing the dot-product structure.</strong> Here's why.</p>

<h2>Deriving RoPE</h2>

<p>Start with one pair of dimensions in Q (call them <code>q_a, q_b</code>) and one pair in K (<code>k_a, k_b</code>). Suppose we transform them by some position-dependent function:</p>

<pre><code>Q_i_pair = R(i) · [q_a, q_b]
K_j_pair = R(j) · [k_a, k_b]</code></pre>

<p>where <code>R(p)</code> is some 2×2 matrix depending on position <code>p</code>. We want:</p>

<pre><code>Q_i_pair · K_j_pair  =  R(i) [q_a, q_b]  ·  R(j) [k_a, k_b]
                     =  [q_a, q_b]^T · R(i)^T · R(j) · [k_a, k_b]</code></pre>

<p>For this to depend only on <code>i − j</code>, the product <code>R(i)^T · R(j)</code> must be a function of <code>i − j</code> alone. The 2D rotation matrix has exactly this property:</p>

<pre><code>R(p) = [[cos(pθ), -sin(pθ)],
        [sin(pθ),  cos(pθ)]]

R(i)^T · R(j) = R(j − i)              <span class="com"># standard rotation identity</span></code></pre>

<p>So if <code>R(p)</code> is a rotation by angle <code>pθ</code>, the transformed dot product is:</p>

<pre><code>Q_i_pair · K_j_pair = [q_a, q_b]^T · R(j − i) · [k_a, k_b]</code></pre>

<p>This depends on <code>j − i</code>, the relative position. <em>Not on absolute positions.</em> Done.</p>

<p>To extend this to a full Q vector of dimension D, we apply <em>different</em> rotations to different pairs of dimensions: pair (0, 1) rotates with one frequency, pair (2, 3) with another, and so on. Different frequencies let the model encode position information at multiple scales — high-frequency rotations distinguish nearby tokens; low-frequency rotations distinguish far-apart tokens.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 360" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">RoPE: each dimension pair rotates by position × its own frequency</text>

  <!-- Q vector at position i, broken into dimension pairs -->
  <text x="20" y="55" font-size="12" font-weight="700" fill="#1a1612">Q_i (head_dim = 8, shown as 4 pairs):</text>
  <g transform="translate(20, 65)">
    <rect x="0"   y="0" width="60" height="40" fill="#fff8a8" stroke="#1a1612"/>
    <text x="30" y="18" text-anchor="middle" font-size="9" fill="#1a1612">(q_0, q_1)</text>
    <text x="30" y="32" text-anchor="middle" font-size="9" fill="#c1502e">freq θ₀ (high)</text>

    <rect x="65"  y="0" width="60" height="40" fill="#fff8a8" stroke="#1a1612"/>
    <text x="95" y="18" text-anchor="middle" font-size="9" fill="#1a1612">(q_2, q_3)</text>
    <text x="95" y="32" text-anchor="middle" font-size="9" fill="#c1502e">freq θ₁</text>

    <rect x="130" y="0" width="60" height="40" fill="#fff8a8" stroke="#1a1612"/>
    <text x="160" y="18" text-anchor="middle" font-size="9" fill="#1a1612">(q_4, q_5)</text>
    <text x="160" y="32" text-anchor="middle" font-size="9" fill="#c1502e">freq θ₂</text>

    <rect x="195" y="0" width="60" height="40" fill="#fff8a8" stroke="#1a1612"/>
    <text x="225" y="18" text-anchor="middle" font-size="9" fill="#1a1612">(q_6, q_7)</text>
    <text x="225" y="32" text-anchor="middle" font-size="9" fill="#c1502e">freq θ₃ (low)</text>
  </g>

  <!-- Arrow down to "rotate each pair by i × theta_k" -->
  <path d="M 130 115 L 130 145" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrR)"/>
  <text x="220" y="135" font-size="11" fill="#1f5f5b">rotate pair k by angle (i × θ_k)</text>

  <!-- After RoPE -->
  <text x="20" y="170" font-size="12" font-weight="700" fill="#1f5f5b">Q_i after RoPE — each pair rotated by its own angle:</text>
  <g transform="translate(20, 180)">
    <circle cx="30" cy="20" r="20" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <line x1="30" y1="20" x2="48" y2="14" stroke="#c1502e" stroke-width="2"/>
    <text x="30" y="56" text-anchor="middle" font-size="9" fill="#1a1612">rot by i·θ₀</text>

    <circle cx="95" cy="20" r="20" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <line x1="95" y1="20" x2="108" y2="6" stroke="#c1502e" stroke-width="2"/>
    <text x="95" y="56" text-anchor="middle" font-size="9" fill="#1a1612">rot by i·θ₁</text>

    <circle cx="160" cy="20" r="20" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <line x1="160" y1="20" x2="158" y2="0" stroke="#c1502e" stroke-width="2"/>
    <text x="160" y="56" text-anchor="middle" font-size="9" fill="#1a1612">rot by i·θ₂</text>

    <circle cx="225" cy="20" r="20" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <line x1="225" y1="20" x2="220" y2="2" stroke="#c1502e" stroke-width="2"/>
    <text x="225" y="56" text-anchor="middle" font-size="9" fill="#1a1612">rot by i·θ₃</text>
  </g>

  <!-- Frequency formula -->
  <text x="20" y="280" font-size="12" font-weight="700" fill="#1a1612">The frequencies (θ_base = 10000 standard):</text>
  <text x="40" y="298" font-size="11" font-family="'IBM Plex Mono', monospace" fill="#1a1612">  θ_k = 1 / (θ_base ^ (2k/D))</text>
  <text x="40" y="315" font-size="10" fill="#6b5d4f">  → θ_0 = 1 (period 2π), θ_1 ≈ 0.31, ..., θ_{D/2−1} ≈ 1/θ_base (very long period)</text>

  <text x="370" y="345" font-family="'Caveat', cursive" font-size="20" fill="#c1502e" text-anchor="middle">high freqs = local position cues; low freqs = global position cues</text>
</svg>
</div>

<h2>The implementation</h2>

<p>The pair-wise rotation can be implemented as elementwise products with cached cos and sin tables. From M29:</p>

<pre><code><span class="kw">def</span> <span class="fn">build_rope_cache</span>(seq_len, head_dim, base=<span class="num">10000.0</span>, device=<span class="str">"cuda"</span>):
    <span class="com"># Frequencies: θ_k = 1 / base^(2k/head_dim) for k = 0, 1, ..., head_dim/2 − 1</span>
    inv_freq = <span class="num">1.0</span> / (base ** (torch.<span class="fn">arange</span>(<span class="num">0</span>, head_dim, <span class="num">2</span>, device=device).<span class="fn">float</span>() / head_dim))
    <span class="com"># Outer product of positions and frequencies → angles</span>
    t = torch.<span class="fn">arange</span>(seq_len, device=device).<span class="fn">float</span>()
    freqs = torch.<span class="fn">outer</span>(t, inv_freq)              <span class="com"># [T, head_dim/2]</span>
    <span class="com"># cos and sin tables, replicated to head_dim (each pair gets the same cos/sin)</span>
    cos = freqs.<span class="fn">cos</span>().<span class="fn">repeat_interleave</span>(<span class="num">2</span>, dim=-<span class="num">1</span>)  <span class="com"># [T, head_dim]</span>
    sin = freqs.<span class="fn">sin</span>().<span class="fn">repeat_interleave</span>(<span class="num">2</span>, dim=-<span class="num">1</span>)
    <span class="kw">return</span> cos, sin

<span class="kw">def</span> <span class="fn">apply_rope</span>(x, cos, sin):
    <span class="com"># x: [..., T, head_dim]. Apply 2D rotation in fp32.</span>
    x_fp32 = x.<span class="fn">float</span>()
    <span class="com"># Pair adjacent dims: (x[0], x[1]), (x[2], x[3]), ...</span>
    x1, x2 = x_fp32[..., ::<span class="num">2</span>], x_fp32[..., <span class="num">1</span>::<span class="num">2</span>]
    <span class="com"># Rotation: [x1, x2] → [x1·cos − x2·sin, x1·sin + x2·cos]</span>
    rotated = torch.<span class="fn">stack</span>([-x2, x1], dim=-<span class="num">1</span>).<span class="fn">flatten</span>(-<span class="num">2</span>)
    <span class="kw">return</span> (x_fp32 * cos + rotated * sin).<span class="fn">to</span>(x.dtype)</code></pre>

<p>Three details worth dwelling on:</p>

<ul>
  <li><strong>fp32 throughout the rotation</strong>. Cosines and sines at large positions can lose precision in bf16; the standard recipe is to cast to fp32 for the rotation, then back to the input dtype. Same pattern as M14's mixed-precision recipe.</li>
  <li><strong>The trick with <code>stack([-x2, x1])</code></strong>: this implements the imaginary-part swap of the rotation. <code>(x1, x2)</code> rotated by angle <code>θ</code> becomes <code>(x1·cos − x2·sin, x1·sin + x2·cos)</code>. The first component uses <code>x1</code> with cos and <code>−x2</code> with sin; the second uses <code>x2</code> with cos and <code>x1</code> with sin. The <code>stack([-x2, x1])</code> reorders into the layout that, when multiplied elementwise by <code>sin</code>, gives exactly the <code>x1·sin</code> and <code>−x2·sin</code> terms.</li>
  <li><strong><code>repeat_interleave(2)</code></strong>: each <em>pair</em> of dimensions shares the same cos/sin value (since they rotate by the same angle). The interleave broadcasts the <code>[T, head_dim/2]</code> table to <code>[T, head_dim]</code>.</li>
</ul>

<h2>The complex-number view</h2>

<p>There's an equivalent, mathematically cleaner formulation of RoPE that uses complex numbers. Treat each pair of dimensions as a complex number: <code>z = x1 + i·x2</code>. Rotation by angle <code>θ</code> is then just complex multiplication:</p>

<pre><code>z_rotated = z · exp(i·θ) = (x1 + i·x2) · (cos θ + i·sin θ)
                         = (x1·cos θ − x2·sin θ) + i·(x1·sin θ + x2·cos θ)</code></pre>

<p>Same result as before. The dot product of two rotated complex numbers becomes:</p>

<pre><code>z_a_rotated · conj(z_b_rotated) = (z_a · exp(i·θ_a)) · conj(z_b · exp(i·θ_b))
                                = z_a · conj(z_b) · exp(i·(θ_a − θ_b))</code></pre>

<p>The exponential factor <code>exp(i·(θ_a − θ_b))</code> only depends on the angle difference — i.e., on the relative position. This is mathematically the same property as before; the complex notation just makes it transparent.</p>

<p>Some implementations (and PyTorch's reference RoPE in <code>torch.nn.functional</code>) use complex tensors directly: pack pairs into <code>complex64</code>, multiply by <code>exp(i·θ)</code>, unpack. Others use the real-valued split-and-combine trick from M29. Performance is identical; the complex form is shorter to write but requires complex tensor support in your kernel.</p>

<h2>The frequency choice: why θ_base = 10000?</h2>

<p>The original RoPE paper used <code>θ_base = 10000</code>, matching the sinusoidal positional encoding from the original Transformer. The choice has a concrete consequence: it determines the range of positions over which the rotation pattern is "non-degenerate."</p>

<p>The lowest frequency in the schedule is <code>θ_{D/2−1} = 10000^(−1) = 0.0001</code>. The corresponding period is <code>2π / 0.0001 ≈ 62832</code>. So even the slowest-rotating dim pair completes one rotation every ~63K positions. For a model trained at 4K context, all the rotations are well-distinguished — every position gets a unique rotation pattern across all pairs.</p>

<p><strong>What changes at <code>θ_base = 500000</code>?</strong> The lowest frequency becomes <code>500000^(−1) = 2e−6</code>, period <code>~3.1M</code>. The rotation pattern remains non-degenerate over much longer ranges. This is what Llama-3 did to extend context from 8K (training) to 128K (inference): they set <code>θ_base = 500000</code> during fine-tuning, retraining the model briefly to use the new schedule. The model now distinguishes positions out to ~500K-1M before the rotations alias.</p>

<p>The general rule: <strong>θ_base must be set so the slowest-rotating dim pair completes &lt; 1 full rotation over the maximum sequence length</strong>. If it completes more than one rotation, positions alias (position 0 and position N rotate to the same angle), and the model can't distinguish them.</p>

<h2>Context extension: NTK, YARN, Dynamic NTK</h2>

<p>You have a model trained at context length 4K with <code>θ_base = 10000</code>. You want to use it at 32K context without retraining from scratch. What do you do?</p>

<p>The naïve approach — just extend the cos/sin tables to position 32K — fails. Positions beyond 4K were never seen at training, and the model doesn't know how to interpret rotations beyond that range. Loss spikes; outputs become nonsensical.</p>

<p>Three recipes that actually work:</p>

<h3>1. Position interpolation</h3>

<p>The simplest extension. To run at 32K context with a model trained at 4K, divide all positions by <code>32K / 4K = 8</code>. The model sees positions <code>0, 0.125, 0.25, 0.375, ..., 4000</code> (32K positions, but rescaled into the trained range). All position indices fit into the original training range; the model sees them as it always did.</p>

<pre><code>scale = original_context / new_context     <span class="com"># 4K / 32K = 0.125</span>
positions_scaled = positions * scale
<span class="com"># Now use these in the standard RoPE formula</span></code></pre>

<p>This works! But it loses high-frequency information — the model now distinguishes positions only at 1/8 the original resolution. Quality degrades on tasks needing fine-grained positional reasoning.</p>

<h3>2. NTK-aware interpolation</h3>

<p>The clever fix. The insight: high-frequency dim pairs (small <code>k</code>) are responsible for fine-grained position distinctions; low-frequency pairs handle long-range. Position interpolation hurts the high frequencies more than the low ones (rescaling by 8 changes a fine-grained "2 vs 3" distinction into a "0.25 vs 0.375" one — much harder to detect).</p>

<p>NTK-aware scaling instead changes <code>θ_base</code> rather than the positions. Specifically:</p>

<pre><code>scale_factor = new_context / original_context        <span class="com"># e.g., 8</span>
new_theta_base = theta_base * (scale_factor)^(D / (D − 2))</code></pre>

<p>The exponent <code>D / (D − 2)</code> is chosen so that high-frequency pairs are barely affected (their rotation rate stays close to original) while low-frequency pairs are rescaled to span the new context length. <strong>Fine-grained position cues are preserved; coarse-grained ones are stretched.</strong></p>

<p>This is what Llama-3 did to get from 8K to 128K context. Apply NTK-aware scaling, fine-tune for 1-2K steps, done. Quality at 128K matches or exceeds the 8K baseline on most tasks.</p>

<h3>3. YARN</h3>

<p>YARN (Yet Another RoPE extensioN) refines NTK further. The core idea: NTK-aware scaling treats all dim pairs as "equally important" to extend, but in practice some pairs benefit from interpolation while others benefit from extrapolation. YARN partitions the dim pairs into three regions:</p>

<ul>
  <li><strong>High-frequency region</strong>: extrapolate (don't rescale; use original θ_k).</li>
  <li><strong>Low-frequency region</strong>: interpolate (rescale by the full factor).</li>
  <li><strong>Middle region</strong>: a smooth blend between the two.</li>
</ul>

<p>YARN also adds a "temperature" tweak — slightly raising the softmax temperature in attention to compensate for the increased entropy of attention scores at long distances. Quality at extreme context (1M+) is meaningfully better with YARN than with vanilla NTK scaling.</p>

<h3>4. Dynamic NTK at inference time</h3>

<p>The above recipes assume you fine-tune at the new context length. Dynamic NTK is a runtime-only trick: <em>scale θ_base based on the current sequence length, not a fixed factor</em>. As the conversation grows past the trained context, the rotation frequencies stretch dynamically.</p>

<pre><code><span class="kw">def</span> <span class="fn">dynamic_ntk_theta_base</span>(theta_base, current_seq_len, original_max_seq_len, head_dim):
    <span class="kw">if</span> current_seq_len &lt;= original_max_seq_len:
        <span class="kw">return</span> theta_base                  <span class="com"># short — use original</span>
    scale = current_seq_len / original_max_seq_len
    <span class="kw">return</span> theta_base * scale ** (head_dim / (head_dim - <span class="num">2</span>))</code></pre>

<p>No fine-tuning required. Quality is worse than YARN with fine-tuning, but acceptable for shorter excursions (e.g., a 4K-trained model handling 8-16K conversations). Used in many production servers as a "best-effort" fallback.</p>

<h2>RoPE in the kernel</h2>

<p>One detail that's load-bearing for performance: <strong>RoPE applies to Q and K <em>before</em> the attention matmul, not inside the attention kernel</strong>. In M25's FlashAttention, the inputs are already-rotated Q and K. The RoPE rotation is its own kernel (or a small set of element-wise ops), and FlashAttention is unchanged.</p>

<p>This separation matters for two reasons:</p>

<ol>
  <li><strong>FlashAttention stays a black box</strong>. The same FlashAttention kernel handles RoPE'd attention, ALiBi'd attention, vanilla attention — all by varying what gets fed in. PyTorch's <code>F.scaled_dot_product_attention</code> doesn't know or care about RoPE; it just does attention on whatever Q and K it receives.</li>
  <li><strong>The KV cache stores post-RoPE keys</strong>. When you append a new token's K to the cache during decode, you've already applied RoPE to it. The cache is thus "natively" positioned. <em>Reading from the cache during attention requires no extra rotation</em> — the rotations are baked in at write time.</li>
</ol>

<p>The KV cache implication has a sharp consequence: <strong>changing the positional encoding scheme requires rebuilding the cache</strong>. If you train a model with <code>θ_base = 10000</code> and want to switch to <code>θ_base = 500000</code> at inference, every cached key has been rotated with the wrong frequency. Discard the cache and re-prefill.</p>

<p>This is also why some advanced attention variants (notably FlashAttention v3 with built-in RoPE) absorb RoPE into the kernel. The performance gain is small for typical workloads but adds up at scale.</p>

<h2>ALiBi: the alternative that didn't quite win</h2>

<p>ALiBi (Attention with Linear Biases) takes the opposite approach. Instead of rotating Q and K, ALiBi adds a position-dependent bias directly to the attention scores:</p>

<pre><code>attn_scores[i, j] = Q_i · K_j  +  m_h · (j − i)</code></pre>

<p>where <code>m_h</code> is a per-head slope (negative; tokens further away get a more negative bias, attended to less). The slopes <code>m_h</code> are fixed (geometric sequence, no learned parameters).</p>

<p>ALiBi has a remarkable property: it extrapolates to longer contexts <em>without any rescaling or fine-tuning</em>. A model trained at 1K context handles 16K context out of the box, with quality decay but no catastrophic failure. The mechanism: long-distance attention scores are biased very negative, so attention naturally focuses on local context — and "local" means the same thing at 1K and 16K positions.</p>

<p>So why didn't ALiBi win?</p>

<ul>
  <li><strong>Quality at the trained context length is slightly lower</strong> than RoPE. Across many benchmarks, RoPE-trained models outperform ALiBi-trained models when both are evaluated at training context.</li>
  <li><strong>Less flexibility for context extension</strong>. NTK / YARN give RoPE a continuous knob to tune for new context lengths; ALiBi's extrapolation, while graceful, has no equivalent fine-tuning recipe.</li>
  <li><strong>Network effects</strong>. RoPE was adopted by Llama-1, then Mistral, then Gemma, then Qwen, then DeepSeek. Once an ecosystem coalesces, the marginal benefit of switching is rarely worth it. ALiBi survives in a few research models (e.g., Bloom, Falcon).</li>
</ul>

<p>If you're building a new model in 2026, use RoPE. If you're working with an ALiBi model, the ideas in this module still apply, just to the bias function instead of the rotation angles.</p>

<h2>The quick-reference table</h2>

<div class="table-wrap">
<table>
<caption>Positional encoding schemes — production status as of 2026</caption>
<thead><tr><th>Scheme</th><th>Used by</th><th>Extrapolates?</th><th>Recommended?</th></tr></thead>
<tbody>
<tr><td>Absolute learned</td><td>BERT, GPT-2 (legacy)</td><td>No (hard wall at trained max)</td><td>No — superseded</td></tr>
<tr><td>Sinusoidal</td><td>Original Transformer</td><td>Marginally</td><td>No — superseded</td></tr>
<tr><td>T5 relative bias</td><td>T5</td><td>Some</td><td>Niche — works for T5-style architectures</td></tr>
<tr><td>ALiBi</td><td>BLOOM, Falcon</td><td>Yes (gracefully)</td><td>Niche — alternative to RoPE</td></tr>
<tr><td>RoPE (θ_base = 10000)</td><td>Llama-1, Llama-2, Mistral, Gemma-1, Qwen, DeepSeek</td><td>With NTK/YARN scaling</td><td><strong>Yes — modern default</strong></td></tr>
<tr><td>RoPE (θ_base = 500000)</td><td>Llama-3, Llama-3.1, Mistral Large, Gemma-2</td><td>To 128K+ with retraining</td><td><strong>Yes — for long context</strong></td></tr>
<tr><td>YARN</td><td>Yi-VL-200K, Qwen2.5-1M</td><td>To 1M+</td><td>Yes — for very long context</td></tr>
</tbody>
</table>
</div>

<div class="ndq">
<h4>About RoPE and positions</h4>

<p class="q">Why pair-wise rotation (2D rotations of consecutive dim pairs) instead of, say, full D-dimensional rotation by some angle?</p>
<p class="a">The pair-wise structure has two advantages. (1) It's <em>cheap</em>: D/2 independent 2D rotations cost ~2D scalar multiplies per token, vs O(D²) for a full D-dim rotation. (2) It admits multiple frequencies: each pair rotates at its own rate, encoding position information at multiple scales simultaneously. A single D-dim rotation has one angle, hence one "frequency" — much less expressive. The pair-wise trick is the standard way to get multi-scale relative-position encoding without the cost of generalized rotations.</p>

<p class="q">Does the order of dimension pairs matter? Could I pair (0, 4), (1, 5), instead of (0, 1), (2, 3)?</p>
<p class="a">Mathematically, no — the pairing is arbitrary as long as it's consistent between Q and K. In practice, two pairing conventions exist: <strong>interleaved</strong> (the split-and-stack code in M29: pair (0, 1), (2, 3), ...) and <strong>halved</strong> (pair (0, D/2), (1, D/2+1), ..., used by Hugging Face's Llama implementation). They're mathematically equivalent but produce different bit patterns. <em>This causes occasional bugs when porting weights between frameworks</em> — make sure your RoPE implementation matches the one used during training.</p>

<p class="q">Why must RoPE be applied to Q and K but not V?</p>
<p class="a">The "depend only on i − j" property only matters for the dot product Q · K. V doesn't enter the attention score; it's the value being weighted by the attention pattern. Applying a position-dependent rotation to V would scramble the values without any compensating un-rotation downstream — V_j would emerge from the attention weighted sum rotated, which is not what you want. <em>RoPE rotates the keys to "look up by position"; V is the data being looked up, untouched.</em></p>

<p class="q">What's the right θ_base for very long context (1M+)?</p>
<p class="a">Empirically, around <code>θ_base ≈ context_length × 100</code> seems to work well, paired with YARN-style frequency partitioning. For 1M context, that's <code>θ_base ≈ 1e8</code>. But this is approximate — at extreme context lengths, the choice of θ_base matters less than the YARN region cutoffs and the temperature tweak. <em>Read the YARN paper for the precise recipe; trust empirical results over theoretical predictions at this scale</em>.</p>

<p class="q">If RoPE is so universal, why do we still teach the older positional encoding schemes?</p>
<p class="a">Three reasons. (1) Legacy: many production checkpoints (BERT, GPT-2, T5) use older schemes; you'll encounter them when working with older models. (2) Conceptual: understanding "why RoPE works" requires understanding what it solved — namely the absolute-vs-relative tension that the older schemes didn't handle well. (3) Future: the next big-paradigm-shift in positional encoding (e.g., learned implicit positional functions, or RoPE alternatives for very long context) will likely be motivated by RoPE's remaining limitations. <em>Knowing the design space is what lets you evaluate the next thing</em>.</p>

<p class="q">Does RoPE interact with quantization (M26)?</p>
<p class="a">Yes, in two places. (1) The cos/sin tables themselves should stay in fp32 — they're small (a few MB), and quantizing them loses precision in the rotation. Most production quantization recipes leave RoPE buffers in fp32. (2) The post-RoPE Q and K can be quantized for the attention matmul. Some FlashAttention variants (especially the int8 / fp8 paths) apply quantization to the rotated Q and K, not the original. <em>The cache stores post-RoPE post-quantization values</em>.</p>
</div>

<h2>Code Magnets: a complete RoPE apply</h2>

<p>You're writing the apply step of RoPE. Three magnets are wrong choices.</p>

<div class="magnets">
<p>Arrange the magnets into a correct apply_rope.</p>

<div class="magnet-pool">
  <span class="magnet">def apply_rope(x, cos, sin):</span>
  <span class="magnet">    x_fp32 = x.float()</span>
  <span class="magnet">    x_fp32 = x</span>
  <span class="magnet">    x1, x2 = x_fp32[..., ::2], x_fp32[..., 1::2]</span>
  <span class="magnet">    x1, x2 = x_fp32[..., :head_dim//2], x_fp32[..., head_dim//2:]</span>
  <span class="magnet">    rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)</span>
  <span class="magnet">    rotated = torch.cat([-x2, x1], dim=-1)</span>
  <span class="magnet">    return (x_fp32 * cos + rotated * sin).to(x.dtype)</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">def</span> <span class="fn">apply_rope</span>(x, cos, sin):
    x_fp32 = x.<span class="fn">float</span>()
    x1, x2 = x_fp32[..., ::<span class="num">2</span>], x_fp32[..., <span class="num">1</span>::<span class="num">2</span>]
    rotated = torch.<span class="fn">stack</span>([-x2, x1], dim=-<span class="num">1</span>).<span class="fn">flatten</span>(-<span class="num">2</span>)
    <span class="kw">return</span> (x_fp32 * cos + rotated * sin).<span class="fn">to</span>(x.dtype)</code></pre>
<p>The traps:</p>
<ul>
  <li><code>x_fp32 = x</code> (no <code>.float()</code>): skips fp32 promotion. Rotation in bf16 loses precision at large positions; cos and sin near phase boundaries cancel poorly. Standard recipe is to do the rotation in fp32 and cast back at the end.</li>
  <li><code>x1, x2 = x_fp32[..., :head_dim//2], x_fp32[..., head_dim//2:]</code>: this is the <em>halved</em> pairing convention (Hugging Face's Llama). The cos/sin tables built with <code>repeat_interleave</code> match the <em>interleaved</em> convention. Mixing conventions silently produces wrong outputs (mathematically valid rotations, but rotating different pairs than the cos/sin tables expect).</li>
  <li><code>rotated = torch.cat([-x2, x1], dim=-1)</code>: concatenation puts all the −x2 values first, then all the x1 values. The interleaved cos/sin tables expect alternating layout (−x2[0], x1[0], −x2[1], x1[1], ...) which <code>stack + flatten</code> produces correctly. <code>cat</code> mismatches.</li>
</ul>
<p>The pattern: <strong>cast to fp32, split into adjacent pairs (interleaved), construct the rotated layout via stack-and-flatten, multiply elementwise with cos and sin, cast back</strong>. Every step has a specific role; mixing conventions or skipping fp32 silently breaks the model.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each positional-encoding concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>Set-equivariance of attention</div>
  <div>A. Attention treats input as a set; positions must be injected somewhere.</div>

  <div>Relative-position attention</div>
  <div>B. Q · K depends only on i − j, not on absolute positions; what RoPE achieves.</div>

  <div>θ_base = 10000</div>
  <div>C. Standard RoPE frequency base; the slowest pair has period ~63K positions.</div>

  <div>θ_base = 500000</div>
  <div>D. Used in Llama-3 to extend context to 128K; slowest pair has period ~3M.</div>

  <div>NTK-aware scaling</div>
  <div>E. Rescales θ_base by scale^(D/(D−2)); preserves high-frequency cues, stretches low-frequency.</div>

  <div>YARN</div>
  <div>F. Partitions dim pairs into extrapolate / interpolate / blend regions; best for very long context.</div>

  <div>ALiBi</div>
  <div>G. Linear bias on attention scores; extrapolates without retraining but quality slightly below RoPE.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>Set-equivariance of attention</strong> → A<br>
<strong>Relative-position attention</strong> → B<br>
<strong>θ_base = 10000</strong> → C<br>
<strong>θ_base = 500000</strong> → D<br>
<strong>NTK-aware scaling</strong> → E<br>
<strong>YARN</strong> → F<br>
<strong>ALiBi</strong> → G
</p>
<p>The mental shortcut: <em>attention is order-blind, RoPE makes it relative, 10K is the standard base, 500K is for long context, NTK rescales the base, YARN partitions the schedule, ALiBi is the linear-bias alternative</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team trains a model with RoPE θ_base=10000 at 4K context, then deploys it at 16K context without retraining. Loss spikes immediately at sequences past 4K. Why, and what's the simplest fix that doesn't require retraining?</p>
<details class="answer"><summary>show answer</summary>
<p>The model has only ever seen rotation angles up to those produced by position 4095. At positions beyond 4K, the rotations enter a regime the model never trained on — the rotation patterns alias or wrap around in unfamiliar ways, and attention scores become noise. The model can't extrapolate cleanly because high-frequency rotation pairs cycle multiple times within the new range.</p>
<p>Simplest no-retrain fix: <strong>Dynamic NTK</strong>. At runtime, when the sequence length exceeds the trained max, scale θ_base by <code>(seq_len / trained_len)^(D/(D−2))</code>. The high-frequency pairs barely change (stay near their trained behavior); low-frequency pairs stretch to span the new context. Quality is degraded vs proper YARN-fine-tuning but usable for moderate excursions (4K → 8K-16K). For longer extensions, retrain with NTK or YARN.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Why is the cos/sin cache stored as a buffer with <code>persistent=False</code> rather than computed on the fly each forward?</p>
<details class="answer"><summary>show answer</summary>
<p>Two performance reasons. (1) The cache is computed <em>once</em> at model init from the config (max_seq_len, head_dim, θ_base). Recomputing it every forward would mean redoing the trig functions for every position every step — wasted work. (2) The cache is small (a few MB for typical configs) and lives entirely on the GPU after init; sliced reads from it for the current segment cost almost nothing.</p>
<p><code>persistent=False</code> means it's not saved in the checkpoint. Why exclude? The cache is fully determined by the config — saving it just bloats checkpoints. And critically, if you load a checkpoint and want to run at a longer context (or different θ_base), the saved cache would be wrong. <strong>persistent=False forces the cache to be regenerated from the current config every load</strong>, which is what you want.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> When a team applies NTK scaling and fine-tunes for 2K steps, validation perplexity drops smoothly. They then notice that some retrieval tasks (find a fact 100K tokens back) work great, but tasks that need precise local position (token-level reasoning, e.g. "swap word 5 with word 8 in the prompt") get worse. Explain.</p>
<details class="answer"><summary>show answer</summary>
<p>NTK scaling preserves high-frequency rotation pairs but stretches low-frequency ones. The high-frequency pairs are still encoding fine-grained position cues, so local positional reasoning should be preserved — but only if the fine-tuning preserved the model's reliance on them. In practice, fine-tuning at long context often shifts the model's "attention budget" toward exploiting the newly-extended low-frequency pairs (because that's where the gradient signal is during long-context tasks). The high-frequency pairs effectively atrophy.</p>
<p>Diagnostic: probe attention patterns on local-reasoning tasks pre- and post-extension. The pre-extension model attends to specific nearby tokens; the post-extension model attends more diffusely. Mitigation: include short-context examples in the fine-tuning mix (Llama-3 did this — the long-context fine-tuning data was 50% short-context, 50% long-context to prevent regression). YARN's blended region partition is also designed to address this.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> Why doesn't FlashAttention need to be modified to support RoPE, when adding ALiBi requires a small modification to the kernel?</p>
<details class="answer"><summary>show answer</summary>
<p>RoPE is applied to Q and K <em>before</em> they enter the attention kernel. From FlashAttention's perspective, it just receives "some Q and K" and computes attention; whether they were rotated by RoPE, normalized, or projected from a different space is irrelevant. The kernel is RoPE-agnostic.</p>
<p>ALiBi adds a position-dependent bias to the attention scores <em>inside</em> the kernel: <code>scores[i, j] += m_h × (j − i)</code>. This addition has to happen between the Q · K matmul and the softmax. Since the scores tile is materialized in registers/shared memory inside FlashAttention, adding the ALiBi bias is a small modification to the kernel's inner loop. Some FlashAttention versions (especially v2+) ship explicit ALiBi support; others require a small fork.</p>
<p>This is part of why RoPE won the architecture war: it slots in front of any attention kernel without modification. <em>The right design moves the complexity to where it can be expressed cleanly, leaving the high-throughput kernel untouched</em>.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>Without positional encoding, attention is <strong>set-equivariant</strong> — order-blind. Position must be injected somewhere.</li>
  <li>Four families of solutions: <strong>absolute learned</strong> (legacy), <strong>sinusoidal</strong> (legacy), <strong>relative bias / ALiBi</strong> (niche), <strong>rotary / RoPE</strong> (modern default).</li>
  <li>RoPE rotates each pair of dimensions in Q and K by an angle proportional to the token's position. Different pairs use different frequencies — high frequency for early dim pairs (local cues), low frequency for late ones (global cues).</li>
  <li>The mathematical magic: when you take Q · K, the rotation angles subtract — <strong>Q_i · K_j depends only on (i − j)</strong>. Relative-position attention by construction.</li>
  <li>The <strong>complex-number view</strong>: pair adjacent dims as a complex number, multiply by <code>exp(i·θ)</code>. Mathematically equivalent to the real-valued formulation, sometimes shorter to implement.</li>
  <li>The <strong>frequency schedule</strong> (θ_k = 1 / θ_base^(2k/D)) determines the position range over which rotations are non-degenerate. <strong>θ_base = 10000</strong> standard; <strong>500000</strong> for long-context fine-tuning (Llama-3 used this for 8K → 128K).</li>
  <li>Implementation: precompute cos/sin tables once, apply via elementwise products in fp32 (cast back at the end). Stored as a buffer with <code>persistent=False</code>.</li>
  <li>RoPE applies to Q and K only, not V. Common bug: applying to V scrambles the values.</li>
  <li><strong>Context extension recipes</strong>: position interpolation (simple, loses fine resolution); <strong>NTK-aware scaling</strong> (rescale θ_base, preserves high frequencies); <strong>YARN</strong> (partition the dim pairs, best for very long context); <strong>Dynamic NTK</strong> (runtime-only, no fine-tuning needed).</li>
  <li>RoPE applies <em>before</em> the attention kernel. FlashAttention is RoPE-agnostic — receives already-rotated Q and K. ALiBi requires a small kernel modification to add bias to scores.</li>
  <li>The KV cache stores <strong>post-RoPE keys</strong>. Changing θ_base requires rebuilding the cache.</li>
  <li>Two pairing conventions exist: <strong>interleaved</strong> (pair (0,1), (2,3)) and <strong>halved</strong> (pair (0, D/2), (1, D/2+1)). Mathematically equivalent but mixing them silently breaks weights ports between frameworks.</li>
  <li>RoPE quantization (M26): cos/sin tables stay in fp32; post-RoPE Q and K can be quantized for the matmul.</li>
  <li>The reflex: when designing or evaluating an architecture change, ask "does this preserve the relative-position property and stay outside the attention kernel?" If yes, it composes cleanly. If no, expect kernel-level engineering.</li>
</ul>
</div>

<p>Module 31 turns to <strong>post-training</strong> — the phase that takes a base model and shapes it into something useful via SFT, reward modeling, RLHF (PPO), and the modern simpler alternative DPO. The engineering of having three or four models in memory at once (policy, reference, reward, value); why DPO became the default for most teams; and the distributed-rollout architecture that production RLHF setups use. After that, M32 consolidates debugging skills, and M33 explores Mamba as the transformer alternative.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">30</span>
  <span>RoPE and positional encodings</span>
</div>
"""

emit("30_rope", "Module 30 — RoPE and positional encodings", BODY)
