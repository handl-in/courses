#!/usr/bin/env python3
"""Module 33: Mamba and state-space models — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part IX · Module 33 · the parallel architecture</div>
  <h1 class="module-title">Mamba and <em>state-space models</em></h1>
  <p class="module-sub">— a non-transformer architecture with O(T) time and fixed-size state, the parallel-scan kernel that makes it fast on GPUs, and why "transformers vs SSMs" turned out to be a false dichotomy</p>
</div>

<p>Every module so far has been transformer-centric. M33 steps outside that frame and looks at the most credible alternative architecture of the past few years: <strong>state-space models (SSMs)</strong>, particularly the Mamba family. SSMs swap attention's quadratic cost and ever-growing KV cache for a fixed-size hidden state and linear-time updates — at least in principle.</p>

<p>The pedagogical reason this module is worth your time isn't to convince you to abandon transformers — most production models in 2026 are still transformers. It's that <strong>the kernel patterns and engineering tradeoffs in SSMs map directly onto everything you've learned</strong>. The selective-scan kernel that powers Mamba uses the same memory-hierarchy mindset as FlashAttention (M25). The hybrid architectures that combine transformer and SSM blocks share the dispatcher (M19), the parallelism axes (M16-M18), and the inference systems (M27). M33 is a case study in "how a frontier architecture you might not have built before slots into the framework you've now built."</p>

<div class="keyidea">
A <strong>state-space model</strong> evolves a fixed-size hidden state through a sequence: <code>hₜ = A·hₜ₋₁ + B·xₜ; yₜ = C·hₜ</code>. The state size doesn't grow with sequence length — solves the O(T²) attention problem and the unbounded-KV-cache problem in one move. <strong>Mamba's contribution</strong> was making A, B, C <em>input-dependent</em> ("selective"), which restored the model quality SSMs needed to be competitive with transformers. The cost: input-dependent recurrence breaks the convolutional shortcut classical SSMs used; you need a real sequential scan, which is hard on GPUs. <strong>The selective-scan kernel</strong> uses the parallel-scan algorithm (Blelloch tree, log T depth) to run a recurrence on a GPU in parallel — same memory-hierarchy patterns as FlashAttention. <strong>Mamba-2</strong> later showed SSMs and a special form of attention are mathematically equivalent, letting Mamba-2 use matmul-heavy kernels and converging the two architecture families. The 2026 reality: hybrid architectures (transformer + SSM blocks) win on many tasks; pure transformers still dominate; pure SSMs occupy specific niches.
</div>

<h2>Two new faces — the last new characters</h2>

<div class="character" style="--c: #6b5d4f;">
  <div class="avatar" style="background: #6b5d4f; color: #fff;">S</div>
  <div>
    <p class="who">State</p>
    <p class="name">"I'm a fixed-size hidden vector that gets updated as the sequence flows past."</p>
    <p class="says">In a transformer, every previous token gets cached in the KV cache (M27) and lives forever in memory — that's why long contexts get expensive. <em>I'm different.</em> I'm a single vector of dimension N (typically 64-256), and I don't grow. After processing token 1, I've absorbed token 1's information; after token 1000, I've absorbed all of them — but I'm still N-dimensional. The catch: I have to <strong>compress</strong> all that history into N dimensions. If I'm too small, I lose information; if I'm well-designed, I capture what matters. <em>I trade memory cost for compression difficulty</em>. Mamba's job is to compress well.</p>
  </div>
</div>

<div class="character" style="--c: #c1502e;">
  <div class="avatar" style="background: #c1502e; color: #fff;">↯</div>
  <div>
    <p class="who">Scan</p>
    <p class="name">"I run a recurrence in parallel on the GPU. Logarithmic depth, not linear."</p>
    <p class="says">A naive recurrence is <code>for t in range(T): h_t = f(h_{t-1}, x_t)</code> — sequential, no parallelism, terrible for GPUs. But if <code>f</code> is associative, the recurrence is a <em>scan</em> (the prefix-sum generalization), and the <strong>Blelloch parallel scan algorithm</strong> runs it in <code>log₂(T)</code> depth on the GPU. Like a tree reduction, but producing all prefix-sums, not just the total. Mamba's selective scan is structured so this works. The kernel pattern: tile-based, same memory-hierarchy mindset as FlashAttention (M25) — keep state tiles in shared memory, stream input through, write output once. <em>Sequential-looking algorithms can run in parallel if they have associativity.</em></p>
  </div>
</div>

<h2>The motivation: attention's two costs</h2>

<p>From M22 and M27, you know two facts about transformer attention:</p>

<ol>
  <li><strong>Quadratic compute</strong>: attention is O(T²) in sequence length. FlashAttention (M25) reduces the memory traffic but doesn't change the FLOPs.</li>
  <li><strong>Unbounded KV cache</strong>: every token's K and V stay in memory for all subsequent decodes. Cost grows linearly with conversation length.</li>
</ol>

<p>Both problems disappear if you use a recurrent architecture. RNNs in the 2010s had a fixed hidden state and O(T) time — so why didn't they win? Two reasons. (1) <strong>Quality</strong>: vanilla RNNs and LSTMs underperformed transformers at scale on language tasks. The compression-into-fixed-state cost too much information. (2) <strong>Throughput</strong>: RNN training requires sequential processing of the sequence, killing GPU utilization.</p>

<p>State-space models are the modern attempt to fix both. The S4 paper (2021) showed a careful state-space parameterization could match transformer quality on long-context tasks. Mamba (2023) added input-dependence ("selectivity") and a fast parallel-scan kernel. Mamba-2 (2024) connected SSMs to attention mathematically and showed how to run them with matmul-heavy kernels. The story is iterative architectural improvement combined with careful kernel engineering.</p>

<h2>The structured state space model</h2>

<p>Start with the simplest version. A continuous-time linear dynamical system:</p>

<pre><code>ẋ(t) = A·x(t) + B·u(t)        <span class="com"># state evolves under input u</span>
y(t) = C·x(t)                  <span class="com"># output is a projection of state</span></code></pre>

<p>Discretize this for token-by-token processing — replace the continuous derivative with a finite difference. You get the discrete-time recurrence:</p>

<pre><code>hₜ = Ā·hₜ₋₁ + B̄·xₜ              <span class="com"># state update</span>
yₜ = C·hₜ                      <span class="com"># output</span></code></pre>

<p>Where:</p>
<ul>
  <li><code>xₜ ∈ ℝ</code> is the input scalar at position t (per-channel; for D channels you have D parallel SSMs).</li>
  <li><code>hₜ ∈ ℝᴺ</code> is the hidden state of dimension N (typically 16, 64, or 256).</li>
  <li><code>Ā ∈ ℝᴺˣᴺ</code> is the discrete-time state matrix.</li>
  <li><code>B̄ ∈ ℝᴺ</code> is the input projection.</li>
  <li><code>C ∈ ℝᴺ</code> is the output projection.</li>
  <li><code>yₜ ∈ ℝ</code> is the output scalar at position t.</li>
</ul>

<p>If <code>Ā, B̄, C</code> are constant (don't depend on input), you can unroll the recurrence in closed form:</p>

<pre><code>yₜ = C · (Āᵗ⁻¹·B̄·x₀ + Āᵗ⁻²·B̄·x₁ + ... + Ā⁰·B̄·xₜ)
   = sum_{k=0}^{t} (C·Āᵗ⁻ᵏ·B̄) · xₖ</code></pre>

<p>This is a <strong>convolution</strong> of the input with a fixed kernel <code>K_k = C·Āᵏ·B̄</code>. So a linear time-invariant SSM can be computed as either (a) a sequential recurrence, or (b) a parallel convolution. The convolution form runs efficiently on a GPU using FFT, with O(T log T) complexity. <em>This is what S4 did.</em></p>

<h2>The selective scan: Mamba's contribution</h2>

<p>The catch with linear time-invariant (LTI) SSMs: the same A, B, C apply to every token, regardless of content. If you're processing "the cat that I saw yesterday was sleeping," the state-update mechanism doesn't change based on whether the current token is "cat" (a content word) or "the" (a function word). For language, this matters — content-dependent attention is a big part of what makes transformers good.</p>

<p>Mamba's fix: make B, C, and the time-step ∆ <em>input-dependent</em>. Specifically:</p>

<pre><code>B̄ₜ = <span class="fn">linear_B</span>(xₜ)           <span class="com"># now B depends on input</span>
C̄ₜ = <span class="fn">linear_C</span>(xₜ)           <span class="com"># C also input-dependent</span>
∆ₜ = <span class="fn">softplus</span>(<span class="fn">linear_∆</span>(xₜ))  <span class="com"># time-step also input-dependent</span>
Āₜ = <span class="fn">discretize</span>(A, ∆ₜ)       <span class="com"># A is structured (often diagonal); discretized via ∆ₜ</span></code></pre>

<p>Now the recurrence is "selective": each step's state-update depends on the current token, allowing the model to choose what to remember and what to forget. This restores the content-aware processing that vanilla SSMs lacked.</p>

<p>The cost: the convolutional shortcut is gone. With input-dependent A, you can't pre-compute a fixed kernel. <strong>You have to run the recurrence as a true scan</strong>. On GPUs that's a problem — sequential scans don't parallelize naturally.</p>

<p>The fix is the parallel scan algorithm.</p>

<h2>The parallel scan</h2>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 380" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrSc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Blelloch parallel scan: a recurrent computation in log₂(T) depth on the GPU</text>

  <!-- Naive sequential -->
  <text x="20" y="55" font-size="12" font-weight="700" fill="#c1502e">Naive sequential — T steps, no parallelism:</text>
  <g transform="translate(20, 65)">
    <rect x="0"   y="0" width="55" height="30" fill="#fcecec" stroke="#c1502e" stroke-width="1"/>
    <text x="27" y="20" text-anchor="middle" font-size="9" fill="#1a1612">h₀</text>
    <path d="M 60 15 L 70 15" stroke="#c1502e" stroke-width="1" fill="none" marker-end="url(#arrSc)"/>
    <rect x="75"  y="0" width="55" height="30" fill="#fcecec" stroke="#c1502e" stroke-width="1"/>
    <text x="102" y="20" text-anchor="middle" font-size="9" fill="#1a1612">h₁</text>
    <path d="M 135 15 L 145 15" stroke="#c1502e" stroke-width="1" fill="none" marker-end="url(#arrSc)"/>
    <rect x="150" y="0" width="55" height="30" fill="#fcecec" stroke="#c1502e" stroke-width="1"/>
    <text x="177" y="20" text-anchor="middle" font-size="9" fill="#1a1612">h₂</text>
    <path d="M 210 15 L 220 15" stroke="#c1502e" stroke-width="1" fill="none" marker-end="url(#arrSc)"/>
    <rect x="225" y="0" width="55" height="30" fill="#fcecec" stroke="#c1502e" stroke-width="1"/>
    <text x="252" y="20" text-anchor="middle" font-size="9" fill="#1a1612">h₃</text>
    <path d="M 285 15 L 295 15" stroke="#c1502e" stroke-width="1" fill="none" marker-end="url(#arrSc)"/>
    <text x="310" y="20" font-size="11" fill="#c1502e">… T steps total, latency = T</text>
  </g>

  <!-- Parallel scan tree -->
  <text x="20" y="135" font-size="12" font-weight="700" fill="#1f5f5b">Parallel scan (T=8): tree-shaped, log₂(8) = 3 depth, all positions computed concurrently:</text>
  
  <!-- Level 0: inputs -->
  <text x="20" y="170" font-size="10" font-weight="700" fill="#1a1612">level 0:</text>
  <g transform="translate(80, 155)">
    <rect x="0"   y="0" width="60" height="22" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/>
    <text x="30" y="15" text-anchor="middle" font-size="9" fill="#1a1612">A₀,B₀x₀</text>
    <rect x="65"  y="0" width="60" height="22" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/>
    <text x="95" y="15" text-anchor="middle" font-size="9" fill="#1a1612">A₁,B₁x₁</text>
    <rect x="130" y="0" width="60" height="22" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/>
    <text x="160" y="15" text-anchor="middle" font-size="9" fill="#1a1612">A₂,B₂x₂</text>
    <rect x="195" y="0" width="60" height="22" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/>
    <text x="225" y="15" text-anchor="middle" font-size="9" fill="#1a1612">A₃,B₃x₃</text>
    <rect x="260" y="0" width="60" height="22" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/>
    <text x="290" y="15" text-anchor="middle" font-size="9" fill="#1a1612">A₄,B₄x₄</text>
    <rect x="325" y="0" width="60" height="22" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/>
    <text x="355" y="15" text-anchor="middle" font-size="9" fill="#1a1612">A₅,B₅x₅</text>
    <rect x="390" y="0" width="60" height="22" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/>
    <text x="420" y="15" text-anchor="middle" font-size="9" fill="#1a1612">A₆,B₆x₆</text>
    <rect x="455" y="0" width="60" height="22" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/>
    <text x="485" y="15" text-anchor="middle" font-size="9" fill="#1a1612">A₇,B₇x₇</text>
  </g>

  <!-- Level 1: pairs combine -->
  <text x="20" y="215" font-size="10" font-weight="700" fill="#1a1612">level 1:</text>
  <g transform="translate(80, 200)">
    <rect x="0"   y="0" width="125" height="22" fill="#fff8a8" stroke="#1a1612" stroke-width="1"/>
    <text x="62" y="15" text-anchor="middle" font-size="9" fill="#1a1612">combine(0:1)</text>
    <rect x="130" y="0" width="125" height="22" fill="#fff8a8" stroke="#1a1612" stroke-width="1"/>
    <text x="192" y="15" text-anchor="middle" font-size="9" fill="#1a1612">combine(2:3)</text>
    <rect x="260" y="0" width="125" height="22" fill="#fff8a8" stroke="#1a1612" stroke-width="1"/>
    <text x="322" y="15" text-anchor="middle" font-size="9" fill="#1a1612">combine(4:5)</text>
    <rect x="390" y="0" width="125" height="22" fill="#fff8a8" stroke="#1a1612" stroke-width="1"/>
    <text x="452" y="15" text-anchor="middle" font-size="9" fill="#1a1612">combine(6:7)</text>
  </g>

  <!-- Level 2: pairs of pairs -->
  <text x="20" y="260" font-size="10" font-weight="700" fill="#1a1612">level 2:</text>
  <g transform="translate(80, 245)">
    <rect x="0"   y="0" width="255" height="22" fill="#fff5d8" stroke="#d4a017" stroke-width="1"/>
    <text x="127" y="15" text-anchor="middle" font-size="9" fill="#1a1612">combine(0:3)</text>
    <rect x="260" y="0" width="255" height="22" fill="#fff5d8" stroke="#d4a017" stroke-width="1"/>
    <text x="387" y="15" text-anchor="middle" font-size="9" fill="#1a1612">combine(4:7)</text>
  </g>

  <!-- Level 3: full -->
  <text x="20" y="305" font-size="10" font-weight="700" fill="#1a1612">level 3:</text>
  <g transform="translate(80, 290)">
    <rect x="0" y="0" width="515" height="22" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="257" y="15" text-anchor="middle" font-size="9" fill="#1a1612">combine(0:7) — final state contains everything</text>
  </g>

  <!-- Connecting tree lines -->
  <line x1="110" y1="177" x2="110" y2="200" stroke="#6b5d4f" stroke-width="0.5" />
  <line x1="175" y1="177" x2="175" y2="200" stroke="#6b5d4f" stroke-width="0.5" />
  <line x1="240" y1="177" x2="240" y2="200" stroke="#6b5d4f" stroke-width="0.5" />
  <line x1="305" y1="177" x2="305" y2="200" stroke="#6b5d4f" stroke-width="0.5" />

  <text x="370" y="350" font-family="'Caveat', cursive" font-size="20" fill="#1f5f5b" text-anchor="middle">key requirement: the combine operator must be associative</text>
  <text x="370" y="372" font-family="'Caveat', cursive" font-size="20" fill="#c1502e" text-anchor="middle">selective scan IS associative if structured properly → log T parallelism</text>
</svg>
</div>

<p>Read the diagram bottom-up. The naïve sequential scan does T steps, each waiting for the previous. The parallel scan organizes the same computation as a tree: in log₂(T) levels, pairs combine into pairs of pairs, into pairs of pairs of pairs, and so on. Each level can run in parallel across all its pairs. At T=8, the depth is 3 instead of 8.</p>

<p>The requirement is <strong>associativity</strong> of the combine operator. For ordinary scans (prefix sum), the operator is +, which is associative. For Mamba's selective scan, the combine operator pairs up consecutive transitions:</p>

<pre><code><span class="com"># Combine two consecutive SSM steps into one effective step:</span>
<span class="com"># Step a: h_a = A_a · h_prev + B_a · x_a</span>
<span class="com"># Step b: h_b = A_b · h_a    + B_b · x_b</span>
<span class="com"># Combined: h_b = (A_b · A_a) · h_prev + (A_b · B_a · x_a + B_b · x_b)</span>
<span class="com"># So the combined "effective" matrices are:</span>
<span class="com">#   A_combined = A_b · A_a</span>
<span class="com">#   B_combined·x_combined = A_b · B_a · x_a + B_b · x_b</span></code></pre>

<p>Combining two combined-pairs follows the same rule. The operation is associative: combine((a,b), (c,d)) = combine(a, combine(b,c,d)). <em>The selective scan, despite looking sequential, is mathematically a parallel-scannable operation</em>.</p>

<p>The Mamba kernel implements this Blelloch scan over the input sequence. Block-tile structure, with each block handling a chunk of sequence positions, scanning within the chunk in registers/shared memory and exchanging chunk-boundary states across blocks. <strong>Same memory-hierarchy mindset as FlashAttention (M25)</strong>: the working data stays in shared memory; HBM is touched only at the block boundaries.</p>

<h2>The Mamba block</h2>

<p>The selective scan is the core, but a real Mamba block has additional components for stability and quality. The full block:</p>

<pre><code><span class="kw">class</span> <span class="ty">MambaBlock</span>(nn.Module):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, d_model, d_state=<span class="num">16</span>, d_conv=<span class="num">4</span>, expand=<span class="num">2</span>):
        <span class="fn">super</span>().<span class="fn">__init__</span>()
        d_inner = expand * d_model

        <span class="com"># Input projection: x → (xz, gated path)</span>
        self.in_proj = nn.<span class="fn">Linear</span>(d_model, d_inner * <span class="num">2</span>, bias=<span class="kw">False</span>)

        <span class="com"># 1D causal conv (short-range mixing — captures local patterns the SSM may miss)</span>
        self.conv1d = nn.<span class="fn">Conv1d</span>(d_inner, d_inner, kernel_size=d_conv,
                                groups=d_inner, padding=d_conv - <span class="num">1</span>)

        <span class="com"># SSM parameters (selective: B, C, ∆ are input-dependent)</span>
        self.x_proj = nn.<span class="fn">Linear</span>(d_inner, d_state * <span class="num">2</span> + d_inner, bias=<span class="kw">False</span>)
        self.dt_proj = nn.<span class="fn">Linear</span>(d_inner, d_inner)        <span class="com"># for ∆</span>

        <span class="com"># A is structured: (d_inner, d_state) parameters, learned in log space</span>
        A = torch.<span class="fn">arange</span>(<span class="num">1</span>, d_state + <span class="num">1</span>).<span class="fn">repeat</span>(d_inner, <span class="num">1</span>).<span class="fn">float</span>()
        self.A_log = nn.<span class="fn">Parameter</span>(torch.<span class="fn">log</span>(A))

        <span class="com"># Output projection</span>
        self.out_proj = nn.<span class="fn">Linear</span>(d_inner, d_model, bias=<span class="kw">False</span>)

    <span class="kw">def</span> <span class="fn">forward</span>(self, x):                              <span class="com"># x: [B, T, d_model]</span>
        <span class="com"># 1. Input projection splits into x and z (gating)</span>
        xz = self.<span class="fn">in_proj</span>(x)                            <span class="com"># [B, T, 2*d_inner]</span>
        x, z = xz.<span class="fn">chunk</span>(<span class="num">2</span>, dim=-<span class="num">1</span>)                     <span class="com"># [B, T, d_inner] each</span>

        <span class="com"># 2. Causal 1D conv (short-range token mixing)</span>
        x = x.<span class="fn">transpose</span>(<span class="num">1</span>, <span class="num">2</span>)                          <span class="com"># [B, d_inner, T] for conv</span>
        x = self.<span class="fn">conv1d</span>(x)[:, :, :T]                  <span class="com"># trim to T</span>
        x = F.<span class="fn">silu</span>(x).<span class="fn">transpose</span>(<span class="num">1</span>, <span class="num">2</span>)                   <span class="com"># [B, T, d_inner]</span>

        <span class="com"># 3. Compute input-dependent SSM parameters</span>
        x_proj = self.<span class="fn">x_proj</span>(x)                          <span class="com"># [B, T, d_state*2 + d_inner]</span>
        B, C, dt_inner = torch.<span class="fn">split</span>(x_proj,
                                          [d_state, d_state, d_inner], dim=-<span class="num">1</span>)
        delta = F.<span class="fn">softplus</span>(self.<span class="fn">dt_proj</span>(dt_inner))     <span class="com"># [B, T, d_inner]</span>
        A = -torch.<span class="fn">exp</span>(self.A_log)                      <span class="com"># [d_inner, d_state]</span>

        <span class="com"># 4. The selective scan! This is the kernel.</span>
        y = <span class="fn">selective_scan</span>(x, delta, A, B, C)         <span class="com"># [B, T, d_inner]</span>

        <span class="com"># 5. Gate output by z</span>
        y = y * F.<span class="fn">silu</span>(z)

        <span class="com"># 6. Output projection</span>
        <span class="kw">return</span> self.<span class="fn">out_proj</span>(y)</code></pre>

<p>Read the structure top-to-bottom:</p>

<ul>
  <li><strong>Input projection splits into <code>x</code> and <code>z</code></strong>: <code>x</code> is what goes through the scan; <code>z</code> is a gate that modulates the output (similar to SwiGLU in transformers).</li>
  <li><strong>1D causal conv with kernel size 4</strong>: captures very short-range token interactions before the SSM. SSMs alone can miss exact-position-N-back patterns; the conv fills that gap.</li>
  <li><strong>Input-dependent B, C, ∆</strong>: this is the "selective" part. Three small linear projections from <code>x</code> produce per-token B, C, and time-step.</li>
  <li><strong>The selective_scan call</strong>: the kernel where almost all the work happens. Internally: parallel scan over T positions, with input-dependent A·exp(∆), B·exp(∆), and C-projection.</li>
  <li><strong>Output gating</strong>: <code>y * silu(z)</code> — same multiplicative gate as SwiGLU.</li>
  <li><strong>Output projection</strong>: project back from <code>d_inner</code> (= 2·d_model) to <code>d_model</code>.</li>
</ul>

<p>Total: roughly the same parameter count as a transformer block at the same d_model. The compute is also similar — the scan dominates at long sequences, the projections at short. Where Mamba wins is the fixed-state inference path: at decode time, the state is just the d_inner × d_state hidden state matrix, ~64 KB for typical configs. <em>The state replaces the KV cache from M27.</em></p>

<h2>Mamba-2: the convergence with attention</h2>

<p>Mamba-2 (Dao et al, 2024) is the second-generation architecture. Two contributions:</p>

<ol>
  <li><strong>A simpler parameterization</strong>: instead of the full structured A matrix, use a scalar A per channel (input-dependent). Reduces compute and simplifies the scan.</li>
  <li><strong>The mathematical equivalence</strong>: with this parameterization, the SSM is mathematically equivalent to a particular form of "structured masked attention." This means Mamba-2 can be implemented with the same matmul-heavy kernels that power transformers — including FlashAttention-like algorithms.</li>
</ol>

<p>The equivalence is striking. Roughly: a Mamba-2 layer can be expressed as <code>Y = (LowerTriangularMask ⊙ (Q · K^T)) · V</code> for specific Q, K, V, and mask structures. The structure constrains it (it's not <em>full</em> attention; it's restricted to a particular causal-masked form), but the kernel can use the same hardware-friendly tile-based matmul patterns as FlashAttention.</p>

<p><strong>The practical consequence</strong>: Mamba-2 trains and runs efficiently on standard tensor-core hardware, without requiring specialized scan kernels. This made it much easier to scale and deploy. Production Mamba-2 models in 2026 use kernels that look more like FlashAttention than like the original Mamba's selective scan.</p>

<p>The deeper lesson: <em>"transformers vs SSMs" was a false dichotomy</em>. The two families are points in a design space of "how do you compute weighted contributions of past tokens efficiently on GPUs," and the practical engineering choices converge.</p>

<h2>Hybrid architectures: the production reality</h2>

<p>By 2026, the most successful applications of SSMs are not pure Mamba models but <strong>hybrid architectures</strong>: alternating transformer and SSM blocks within the same model. Examples:</p>

<ul>
  <li><strong>Jamba</strong> (AI21, 2024): alternates Mamba blocks with transformer blocks, ~7:1 ratio. Extends to 256K context.</li>
  <li><strong>Mamba-Codestral</strong>: Mistral's code-focused Mamba-2 model. Pure SSM at ~7B scale.</li>
  <li><strong>Hymba</strong>, <strong>Zamba</strong>, <strong>Samba</strong>: research models exploring various transformer/SSM hybrid recipes.</li>
</ul>

<p>The motivation for hybrids: SSMs and transformers have <em>complementary failure modes</em>. SSMs are great at long-context throughput but can struggle with exact retrieval ("what was that token at position 47?"). Transformers excel at retrieval but struggle with long-context inference cost. Alternating block types lets each handle what it's good at.</p>

<p>The architectural pattern:</p>

<pre><code><span class="kw">class</span> <span class="ty">HybridModel</span>(nn.Module):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, cfg):
        <span class="fn">super</span>().<span class="fn">__init__</span>()
        self.embed = nn.<span class="fn">Embedding</span>(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.<span class="fn">ModuleList</span>()
        <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(cfg.n_layer):
            <span class="kw">if</span> i % cfg.attn_period == <span class="num">0</span>:
                <span class="com"># every Nth block is a transformer block</span>
                self.blocks.<span class="fn">append</span>(<span class="fn">TransformerBlock</span>(cfg))
            <span class="kw">else</span>:
                self.blocks.<span class="fn">append</span>(<span class="fn">MambaBlock</span>(cfg.d_model))
        self.norm = <span class="fn">RMSNorm</span>(cfg.d_model)
        self.lm_head = nn.<span class="fn">Linear</span>(cfg.d_model, cfg.vocab_size, bias=<span class="kw">False</span>)</code></pre>

<p>For Jamba's recipe: every 8th block is attention; the rest are Mamba. The KV cache (M27) only exists for the attention blocks — much smaller than a pure transformer. The state buffers exist for the Mamba blocks. <em>Inference systems that handle Jamba (vLLM and similar) need to manage both</em>.</p>

<h2>The empirical state in 2026</h2>

<p>What's actually winning, two years after Mamba?</p>

<div class="table-wrap">
<table>
<caption>Architecture choice by application domain (2026)</caption>
<thead><tr><th>Application</th><th>Dominant architecture</th><th>Notes</th></tr></thead>
<tbody>
<tr><td>Frontier chat models (Llama, Claude, GPT)</td><td>Pure transformer</td><td>Attention's flexibility wins; quality is paramount; long-context tricks (M30) keep transformers competitive</td></tr>
<tr><td>Long-context reasoning</td><td>Transformer + careful inference</td><td>Paged KV (M27) + speculative decode handle the cost</td></tr>
<tr><td>Code generation</td><td>Transformer or hybrid</td><td>Mamba-Codestral shows pure-SSM viability; transformers still dominate</td></tr>
<tr><td>Edge / mobile inference</td><td>Hybrid (Jamba-style)</td><td>Smaller KV cache makes constrained-memory deployment easier</td></tr>
<tr><td>Time-series and signal processing</td><td>Pure SSM (S4, Mamba)</td><td>SSMs excel where signals have known structure; the original SSM win-zone</td></tr>
<tr><td>Genomics / long sequences</td><td>Pure SSM or hybrid</td><td>Sequences in hundreds of thousands of tokens; transformer attention is cost-prohibitive</td></tr>
<tr><td>Vision (image and video)</td><td>Transformer (ViT, DiT) or hybrid</td><td>Pure transformers dominate; SSM-ViT variants exist but haven't taken over</td></tr>
</tbody>
</table>
</div>

<p>The honest summary: <strong>transformers won the language race</strong>. SSMs are valuable in specific domains (long sequences, signals, edge deployment) and as components in hybrid architectures. The "Mamba kills the transformer" headlines from 2023 didn't pan out, but the engineering ideas — input-dependent recurrence, parallel-scan kernels, hybrid architectures — are now part of the standard toolkit.</p>

<h2>The selective-scan kernel walkthrough</h2>

<p>For completeness, the high-level structure of the selective scan kernel in Triton. This is what powers Mamba-1; Mamba-2 uses matmul-heavy kernels instead.</p>

<pre><code><span class="kw">@</span>triton.<span class="fn">jit</span>
<span class="kw">def</span> <span class="fn">selective_scan_kernel</span>(
    x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, y_ptr,
    B_size, T, D, N,                           <span class="com"># batch, time, channels, state_dim</span>
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    <span class="com"># Each program handles one (batch, channel-tile) pair.</span>
    pid_b = tl.<span class="fn">program_id</span>(<span class="num">0</span>)
    pid_d = tl.<span class="fn">program_id</span>(<span class="num">1</span>)

    <span class="com"># 1. Load A for this channel-tile (small, fits in registers).</span>
    A = tl.<span class="fn">load</span>(A_ptr + ...)                 <span class="com"># [BLOCK_D, N]</span>

    <span class="com"># 2. Initialize state to zero.</span>
    state = tl.<span class="fn">zeros</span>([BLOCK_D, N], dtype=tl.float32)

    <span class="com"># 3. Stream through time tiles.</span>
    <span class="kw">for</span> t_start <span class="kw">in</span> <span class="fn">range</span>(<span class="num">0</span>, T, BLOCK_T):
        <span class="com"># Load this tile's inputs and per-token parameters.</span>
        x      = tl.<span class="fn">load</span>(x_ptr  + ...)          <span class="com"># [BLOCK_T, BLOCK_D]</span>
        dt     = tl.<span class="fn">load</span>(dt_ptr + ...)          <span class="com"># [BLOCK_T, BLOCK_D]</span>
        B_t    = tl.<span class="fn">load</span>(B_ptr  + ...)          <span class="com"># [BLOCK_T, N]</span>
        C_t    = tl.<span class="fn">load</span>(C_ptr  + ...)          <span class="com"># [BLOCK_T, N]</span>

        <span class="com"># Discretize: A_eff = exp(dt · A), B_eff = dt · B</span>
        <span class="com"># (in fp32; cast back at the end)</span>
        A_eff = tl.<span class="fn">exp</span>(dt[:, :, <span class="kw">None</span>] * A[<span class="kw">None</span>, :, :])  <span class="com"># [BLOCK_T, BLOCK_D, N]</span>
        B_eff = dt[:, :, <span class="kw">None</span>] * B_t[:, <span class="kw">None</span>, :]      <span class="com"># [BLOCK_T, BLOCK_D, N]</span>

        <span class="com"># Within-tile parallel scan (Blelloch tree over BLOCK_T).</span>
        state_tile = <span class="fn">parallel_scan_within_tile</span>(state, A_eff, B_eff, x)

        <span class="com"># Compute outputs y = C · state for each time step in tile.</span>
        y = tl.<span class="fn">sum</span>(state_tile * C_t[:, <span class="kw">None</span>, :], axis=-<span class="num">1</span>)   <span class="com"># [BLOCK_T, BLOCK_D]</span>
        tl.<span class="fn">store</span>(y_ptr + ..., y)

        <span class="com"># Update state to last position's state — carries to next tile.</span>
        state = state_tile[-<span class="num">1</span>]                  <span class="com"># [BLOCK_D, N]</span></code></pre>

<p>The patterns will look familiar from M24 and M25:</p>

<ul>
  <li><strong>Tile-based</strong>: process BLOCK_T positions at a time. State stays in registers across tiles within one program.</li>
  <li><strong>fp32 accumulators</strong>: the recurrence uses fp32 internally even when inputs are bf16. Same recipe as M14, M23, M25.</li>
  <li><strong>Block-internal parallel scan</strong>: within each tile, run the Blelloch scan in shared memory. Across tiles, just carry the state forward (sequentially across tiles, parallel within).</li>
  <li><strong>Memory-bound at most sizes</strong>: the per-token compute is small; the kernel is dominated by reading A, B, C, x from HBM. Tile-based access keeps things efficient.</li>
</ul>

<p>The reference Mamba kernel is ~250 lines of Triton. Same kernel-engineering toolkit, applied to a different mathematical structure. <em>If you can read M25's FlashAttention kernel, you can read this one</em>.</p>

<div class="ndq">
<h4>About SSMs and Mamba</h4>

<p class="q">Why is the state size N typically small (16-256)?</p>
<p class="a">Two reasons. (1) <strong>Memory cost</strong>: at inference, the state buffer is <code>d_inner × N</code> per layer. For d_inner=2048 and N=16, that's 32K fp32 numbers = 128 KB per layer per token. Doubling N doubles the inference memory. Keeping N small keeps the SSM's "fixed state" win meaningful. (2) <strong>Quality plateau</strong>: empirically, increasing N past 64-128 gives diminishing returns. The information bottleneck of the recurrence helps more than it hurts up to a point; past that, you're just adding parameters without adding capacity. <em>The "right" N depends on the task</em>; long-context retrieval-heavy tasks benefit from larger N; pure language modeling is fine with N=16.</p>

<p class="q">Doesn't a fixed-size state lose information about long sequences?</p>
<p class="a">Yes — and this is the central tradeoff. The state can only carry forward what it has compressed; tokens far back in the sequence get squeezed through repeated applications of A. For tasks needing exact retrieval ("what was the second word of the prompt?"), this is bad. For tasks where local context dominates ("predict the next word in this sentence"), it's fine. Mamba's selectivity helps — input-dependent A means the state can choose what to remember and what to forget — but it doesn't fully solve the retrieval problem. <strong>This is why hybrid architectures with some attention layers exist</strong>: those layers handle the retrieval queries.</p>

<p class="q">If Mamba-2 is mathematically equivalent to a form of attention, what's the actual difference?</p>
<p class="a">The difference is the <em>structure</em> of the equivalent attention. Standard transformer attention has free-form Q, K, V (any matrix can attend to any matrix). Mamba-2's "structured masked attention" form constrains Q, K, V to a specific input-dependent decomposition. This constraint reduces compute and parameter count, but also restricts the model's flexibility. <em>Mamba-2 is best understood as "constrained attention with parallel-scan kernel options"</em> — not a fundamentally different architecture, but a constrained subset of attention with engineering advantages in some regimes.</p>

<p class="q">Why does the 1D causal conv exist in the Mamba block? Doesn't the SSM handle sequence mixing?</p>
<p class="a">The SSM mixes via the recurrence — long-range — but it's not great at exact-N-back patterns ("what was the token 3 positions ago?"). The 1D conv with kernel size 4 explicitly captures positions 0, -1, -2, -3 of the input. Adding it before the SSM gives the model a clean way to handle local patterns. The conv is parameter-cheap (kernel size 4, depthwise) and its omission empirically hurts quality. <em>Architectural lesson: small architecture details can matter, especially when the main mechanism has known weaknesses</em>.</p>

<p class="q">Can I use Mamba in a torch.compile graph?</p>
<p class="a">Yes, but: the standard Mamba implementations use custom CUDA kernels (or Triton kernels with the parallel scan). torch.compile (M21) treats these as opaque ops via the dispatcher (M19), so the SSM block traces fine, but compile won't <em>fuse</em> through the scan. The fusion you get is around the scan: input projections + scan + output projection becomes (projection-fused + opaque scan + projection-fused). For pure Mamba models, this is acceptable; for hybrids, the transformer blocks compile normally. <em>Inductor (M21) doesn't generate selective-scan kernels itself; you bring your own</em>.</p>

<p class="q">What about training stability — Mamba had a reputation for being finicky?</p>
<p class="a">Mamba-1 indeed had stability issues (the recurrence can produce exploding or vanishing states without careful initialization). The standard fixes: initialize A in log space (so A is always negative, ensuring stability), use softplus on ∆ (positive time-steps), apply RMSNorm after the SSM. With these in place, Mamba trains stably. Mamba-2's simpler parameterization is more robust still. <em>For a from-scratch implementation, study a reference codebase carefully — the small init details matter</em>.</p>

<p class="q">Is Mamba a good choice for a new project I'm starting?</p>
<p class="a">Probably not as the only architecture. Pure-Mamba pretraining is research territory; the recipes are less mature than transformer training. Hybrid architectures are credible but require maintaining two architectural paths (KV cache for attention blocks, state buffers for SSM blocks). For most projects, <em>start with a transformer</em>. The exception: if you have a clear reason — extreme long sequences (genomics, audio at high sample rate, time series), severe inference memory constraints (edge devices), or you're explicitly researching architecture — then SSMs become competitive.</p>
</div>

<h2>Code Magnets: implement the Mamba block forward (high-level)</h2>

<p>You're writing the Mamba block forward (not the inner scan kernel — the surrounding logic). Three magnets are wrong choices.</p>

<div class="magnets">
<p>Arrange the magnets into a working forward pass.</p>

<div class="magnet-pool">
  <span class="magnet">def forward(self, x):</span>
  <span class="magnet">    xz = self.in_proj(x)</span>
  <span class="magnet">    x, z = xz.chunk(2, dim=-1)</span>
  <span class="magnet">    x = self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2)</span>
  <span class="magnet">    x = F.silu(x)</span>
  <span class="magnet">    B, C, dt_in = self.x_proj(x).split([self.d_state, self.d_state, self.d_inner], dim=-1)</span>
  <span class="magnet">    delta = F.softplus(self.dt_proj(dt_in))</span>
  <span class="magnet">    delta = self.dt_proj(dt_in)</span>
  <span class="magnet">    A = -torch.exp(self.A_log)</span>
  <span class="magnet">    A = self.A_log</span>
  <span class="magnet">    y = selective_scan(x, delta, A, B, C)</span>
  <span class="magnet">    y = y * F.silu(z)</span>
  <span class="magnet">    y = y + z</span>
  <span class="magnet">    return self.out_proj(y)</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">def</span> <span class="fn">forward</span>(self, x):
    xz = self.<span class="fn">in_proj</span>(x)
    x, z = xz.<span class="fn">chunk</span>(<span class="num">2</span>, dim=-<span class="num">1</span>)
    x = self.<span class="fn">conv1d</span>(x.<span class="fn">transpose</span>(<span class="num">1</span>, <span class="num">2</span>))[:, :, :T].<span class="fn">transpose</span>(<span class="num">1</span>, <span class="num">2</span>)
    x = F.<span class="fn">silu</span>(x)
    B, C, dt_in = self.<span class="fn">x_proj</span>(x).<span class="fn">split</span>([self.d_state, self.d_state, self.d_inner], dim=-<span class="num">1</span>)
    delta = F.<span class="fn">softplus</span>(self.<span class="fn">dt_proj</span>(dt_in))
    A = -torch.<span class="fn">exp</span>(self.A_log)
    y = <span class="fn">selective_scan</span>(x, delta, A, B, C)
    y = y * F.<span class="fn">silu</span>(z)
    <span class="kw">return</span> self.<span class="fn">out_proj</span>(y)</code></pre>
<p>The traps:</p>
<ul>
  <li><code>delta = self.dt_proj(dt_in)</code> (no softplus): the time-step ∆ must be positive (it's a discretization step size). Without softplus, ∆ can be negative, producing unstable or exploding state updates. Softplus enforces ∆ &gt; 0 and provides a smooth gradient.</li>
  <li><code>A = self.A_log</code> (no negation, no exp): A is parameterized in log space and constrained to be negative for stability. The actual matrix A is <code>-exp(A_log)</code> — exponentiation produces positive numbers, then negation makes them negative. Skipping this step (using <code>A_log</code> directly as A) breaks the stability invariant; the SSM's eigenvalues can grow, producing explosions.</li>
  <li><code>y = y + z</code>: additive instead of multiplicative gating. SwiGLU-style gating uses multiplication (<code>y * silu(z)</code>) — the gate modulates which features pass through. Addition is just a residual connection on the gate path, which doesn't gate anything.</li>
</ul>
<p>The pattern: <strong>project, gate-split, conv, silu, project to per-token SSM params, softplus on ∆, negate-and-exp on A, scan, multiplicative gate with z, output projection</strong>. Skipping any of softplus, the A parameterization, or the multiplicative gate breaks the model. The reference Mamba implementations include all three; learning Mamba is largely learning these small-but-essential details.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each SSM concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>State (h_t)</div>
  <div>A. Fixed-size hidden vector (typically dim 16-256) carrying compressed history.</div>

  <div>Selective parameters (B_t, C_t, ∆_t)</div>
  <div>B. Input-dependent SSM coefficients; what makes Mamba selective vs vanilla SSMs.</div>

  <div>Parallel scan (Blelloch)</div>
  <div>C. Tree-structured algorithm running an associative recurrence in log T depth.</div>

  <div>Selective scan kernel</div>
  <div>D. Triton kernel implementing the parallel scan with tile-based memory hierarchy.</div>

  <div>Mamba-2 / structured attention</div>
  <div>E. Mamba-2 reformulation as a form of structured attention; matmul-friendly kernels.</div>

  <div>1D causal conv (in Mamba block)</div>
  <div>F. Short-range token mixing; covers positions the SSM may struggle with.</div>

  <div>Hybrid architecture</div>
  <div>G. Alternates transformer and SSM blocks (Jamba-style, ~7:1 ratio).</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>State (h_t)</strong> → A<br>
<strong>Selective parameters</strong> → B<br>
<strong>Parallel scan</strong> → C<br>
<strong>Selective scan kernel</strong> → D<br>
<strong>Mamba-2 / structured attention</strong> → E<br>
<strong>1D causal conv</strong> → F<br>
<strong>Hybrid architecture</strong> → G
</p>
<p>The mental shortcut: <em>state is fixed-size compressed history, selective parameters make the recurrence input-dependent, parallel scan parallelizes the recurrence on GPUs, the kernel implements it tile-based, Mamba-2 reformulates as attention, the conv handles local patterns, hybrids combine both block types</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team trains a pure Mamba model on a long-context QA task ("answer a question about a 32K-token document"). It does well on summarization but fails dramatically on questions like "what's the third word of paragraph 5?" Why?</p>
<details class="answer"><summary>show answer</summary>
<p>The fixed-size state is the issue. Mamba compresses 32K tokens into a state of dimension <code>d_inner × N</code> (~64-256K parameters). For aggregate questions like summarization, this compression is fine — the answer depends on the gist, not specific tokens. For exact-position retrieval ("third word of paragraph 5"), the model needs to recall a specific token's content, which competes for state capacity with all the other tokens.</p>
<p>Two practical fixes: (1) <strong>Switch to a hybrid architecture</strong> like Jamba — interspersed attention blocks have unbounded retrieval capability via the KV cache; the SSM blocks handle long-range trends. (2) <strong>Increase state size</strong> N — gives more capacity, at cost of memory. Doubling N costs ~2× the inference memory but improves retrieval. For pure-Mamba retrieval, you'd typically want N=128-256 and accept the memory cost. <em>For tasks heavily dependent on exact retrieval, transformers (or hybrids) are still the right choice in 2026</em>.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Why does the parallel scan algorithm work for Mamba's selective scan but not for, say, a vanilla LSTM?</p>
<details class="answer"><summary>show answer</summary>
<p>The parallel scan requires the combine operator to be <strong>associative</strong>. For Mamba's selective scan, two consecutive transitions can be combined as:</p>
<pre>(A_a, B_a·x_a) ∘ (A_b, B_b·x_b) = (A_b·A_a, A_b·B_a·x_a + B_b·x_b)</pre>
<p>This combine operation is associative because matrix multiplication is associative — combining (a,b) with c gives the same result as combining a with (b,c). Three or more transitions chain by the same rule.</p>
<p>For a vanilla LSTM, the recurrence involves nonlinearities (sigmoids on gates, tanh on the cell state):</p>
<pre>h_t = o_t · tanh(c_t),  c_t = f_t · c_{t-1} + i_t · tanh(...)</pre>
<p>The nonlinearities mean the combine operator isn't associative — applying tanh after combining two steps gives a different result than combining "tanh(step a)" and "step b" (you can't pull tanh through linear combinations). This is why LSTMs are intrinsically sequential at training time: they can't be expressed as an associative scan. Mamba's structural choice — keep the recurrence linear, push nonlinearities outside the scan — is what makes parallel scan possible. <em>The architecture's parallelism story is baked into its mathematical structure</em>.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> Why is the Mamba kernel typically memory-bound rather than compute-bound, and what does this imply for hardware optimization?</p>
<details class="answer"><summary>show answer</summary>
<p>Each step of the selective scan does very little arithmetic per token: a few multiplies and adds for the state update, an output projection. The total FLOPs per token are O(N) where N is the state dim — typically 16-64 ops. But the state, A, B, C, x, and y all need to be loaded from HBM (or carried across tiles) for each token. <strong>Arithmetic intensity (FLOPs/byte) is low</strong> — well below the H100 ridge point of ~333 ops/byte (M22). The kernel is memory-bound.</p>
<p>Implications for hardware optimization:</p>
<ul>
  <li>The kernel benefits from <strong>shared memory tiling</strong> — keep the state in SRAM across tile boundaries, reducing HBM round-trips for state. Same FlashAttention pattern.</li>
  <li><strong>Quantization helps</strong> — int8 or fp8 weights for B, C, and the scan parameters cut bandwidth ~2× without changing FLOPs. Less impactful than for transformer matmul (which is compute-bound at large sizes), but still worthwhile for inference.</li>
  <li><strong>Tensor cores don't help much</strong> — the kernel doesn't have a big matmul to feed them. This is one reason Mamba-2's matmul-heavy reformulation is faster on modern GPUs.</li>
  <li><strong>Long sequences amortize launch overhead</strong> — for short sequences, kernel launch dominates. Mamba shines at very long contexts where the per-token cost is what matters.</li>
</ul>
<p>The takeaway: Mamba kernels follow the same engineering recipe as FlashAttention — shared-memory tiling, low-precision storage, fused operations — but the underlying compute pattern is different (scan, not matmul).</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> A team is debating between (a) a pure Mamba 7B model, (b) a hybrid 7B model with 7:1 SSM:attention ratio, (c) a pure transformer 7B model with sliding-window attention. They want to serve 128K-context conversations. What are the tradeoffs?</p>
<details class="answer"><summary>show answer</summary>
<p>Each option has a different inference cost profile and quality character:</p>
<ul>
  <li><strong>(a) Pure Mamba</strong>: KV cache replaced by state buffers (~64 KB/layer, fixed regardless of context). Decode latency dominated by state-update kernel. Quality on retrieval-heavy tasks may suffer. Inference memory: very low. Easiest to scale to 1M+ context.</li>
  <li><strong>(b) Hybrid (7:1)</strong>: 1/8 of layers are attention with full KV cache. KV cache size: 1/8 of pure transformer at same context. Quality: usually best of the three (retrieval handled by attention layers; long-range trends by Mamba). Inference memory: medium.</li>
  <li><strong>(c) Transformer + sliding window</strong>: KV cache bounded by window size (e.g., 4K). Tokens outside the window are forgotten. Quality on long-range dependencies suffers; quality on local tasks is best. Inference memory: bounded but small.</li>
</ul>
<p>For 128K-context chat (typical of a long-document Q&amp;A application): the hybrid (b) usually wins. Quality benefits from both architectures' strengths; memory is manageable. For pure throughput optimization where quality is fine: (a). For applications where you only need recent context: (c). <em>The decision is task-dependent; benchmarking on representative data is essential</em>. The right answer 5 years ago would have been (c); the right answer in 2026 is increasingly (b).</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>State-space models (SSMs) evolve a <strong>fixed-size hidden state</strong> through a sequence: <code>h_t = A·h_{t-1} + B·x_t</code>. The state size doesn't grow — solves both attention's O(T²) cost and the unbounded KV cache problem.</li>
  <li><strong>S4 (2021)</strong>: linear time-invariant SSMs that can be computed as a parallel convolution.</li>
  <li><strong>Mamba (2023)</strong>: makes A, B, C, ∆ <em>input-dependent</em> ("selective") — restores quality, but breaks the convolutional shortcut. Now you need a true sequential scan.</li>
  <li>The <strong>parallel scan algorithm</strong> (Blelloch tree) runs an associative recurrence in <code>log₂(T)</code> depth on the GPU. Mamba's selective scan IS associative when structured properly.</li>
  <li>The <strong>selective-scan kernel</strong> is tile-based, shared-memory-resident state, fp32 accumulators — same kernel-engineering recipe as FlashAttention (M25).</li>
  <li><strong>The Mamba block</strong>: input projection (split into x and z gate) → 1D causal conv → input-dependent SSM parameters → selective scan → multiplicative gating with z → output projection.</li>
  <li>Stability tricks: <code>A = -exp(A_log)</code> ensures eigenvalues are negative; <code>∆ = softplus(...)</code> ensures positive time-steps; RMSNorm after SSM.</li>
  <li><strong>Mamba-2 (2024)</strong>: simpler parameterization; mathematically equivalent to a "structured masked attention." Can use matmul-heavy kernels — converges with transformer engineering.</li>
  <li><strong>The 2026 reality</strong>: pure transformers won the language race. SSMs are valuable for long-sequence domains (genomics, signals, time series), edge inference, and as components in <strong>hybrid architectures</strong> (Jamba alternates 7 Mamba blocks per attention block).</li>
  <li>Trade-offs: SSMs lose information about specific past tokens (compressed state); transformers retain all of it (KV cache). Hybrid architectures get the best of both at moderate memory cost.</li>
  <li>The kernel is memory-bound (low arithmetic intensity); benefits from shared-memory tiling and quantization but doesn't use tensor cores heavily.</li>
  <li>The deeper lesson: <em>"transformers vs SSMs" was a false dichotomy</em>. They're points in a design space of efficient sequence-mixing on GPUs, and the engineering choices converge.</li>
  <li>The reflex: when evaluating an architecture, ask "what's its access pattern?" — quadratic scan over all pairs (attention), fixed-state recurrence (SSM), or a constrained subset (Mamba-2). Each maps to a kernel pattern from M22-M25.</li>
</ul>
</div>

<h2>Closing Part IX</h2>

<p>Five extension modules — M29 built a transformer end-to-end, M30 unpacked RoPE, M31 covered post-training (SFT, RM, PPO, DPO), M32 consolidated debugging skills, and M33 explored Mamba as the parallel architecture.</p>

<p>What you have now: a course that takes you from <code>x.stride()</code> to <code>fully_shard</code> to <code>tl.dot</code> to MoE expert parallelism to RoPE rotation derivations to DPO loss to Blelloch parallel scans. Thirty-three modules, ~45 characters. The original capstone in M28 is still the synthesis exercise; M29's transformer is the synthesis you build with your own hands.</p>

<p>What's still missing from the course as a whole: everything mentioned in the "what's not covered" review — RLHF in finer detail, multimodal architectures, vision encoders, smaller-scale and edge deployment, AMD/TPU specifics, NCCL internals. Those are their own course-extensions if you want them.</p>

<p>The point isn't to know everything — nobody does — but to <em>have the framework to learn anything new in the field</em>. When the next architectural breakthrough lands, you'll have the dispatcher, the roofline, the parallelism axes, the tile-based kernel mindset, the post-training pipeline, the RoPE derivation, and the SSM design space to slot it in.</p>

<p>Thanks for reading this far. Now go make something.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">33</span>
  <span>Mamba and state-space models — Part IX fin.</span>
</div>
"""

emit("33_mamba", "Module 33 — Mamba and state-space models", BODY)
