#!/usr/bin/env python3
"""Module 25: FlashAttention case study — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part VIII · Module 25</div>
  <h1 class="module-title">FlashAttention: <em>a case study</em></h1>
  <p class="module-sub">— how the online softmax trick turned attention from memory-bound to compute-bound, and why the kernel is one of the most consequential pieces of code of the past five years</p>
</div>

<p>This is the module where everything you've built converges. We're going to walk through FlashAttention — the kernel that, more than any other single piece of work, made long-context transformers practical. Not "make them faster" — <em>make them possible</em>. Without it, training a 32K-context model would require terabytes of activation memory; serving long-prompt inference would be unaffordable. With it, both fit on commodity hardware.</p>

<p>Here's what makes FlashAttention worth a whole module: its trick isn't algorithmic novelty (the math is just a careful refactoring of softmax). The trick is <em>understanding the hardware well enough</em> (M22) <em>to restructure the algorithm so the right things stay in the right level of memory</em>. Every concept we've built leads here. The roofline (M22) tells us why naive attention is bad; the GPU memory hierarchy (M22) tells us where to keep what; tile-based programming (M24) is how we write it; the recompute trick (M12) is how the backward works; <code>torch.library</code> (M19, M23) is how we register it; mixed precision (M14) is how we keep it stable.</p>

<div class="keyidea">
Naive attention is memory-bound: it computes the full <code>(T, T)</code> attention matrix in HBM, reads it back to softmax, then reads it again to multiply by V. <strong>FlashAttention restructures the computation so the attention matrix never lives in HBM</strong> — it's recomputed in tiles inside shared memory using the <em>online softmax</em> trick. Streaming K and V through one Q tile at a time, the kernel produces the same numerical result with O(T) memory and ~5× less HBM traffic. The cost: the kernel is harder to write than naive attention. The win: long-context training and inference become tractable. <strong>This pattern — restructure to keep working data in shared memory, stream the rest — is the template for modern attention variants (paged attention, MQA/GQA, sliding window) covered in M27.</strong>
</div>

<h2>The naive attention recipe (and why it's bad)</h2>

<p>Standard scaled dot-product attention, for one head:</p>

<pre><code>S = Q @ K.<span class="fn">transpose</span>(-<span class="num">2</span>, -<span class="num">1</span>) / <span class="fn">sqrt</span>(d)        <span class="com"># (T, T) attention scores</span>
P = torch.<span class="fn">softmax</span>(S, dim=-<span class="num">1</span>)                  <span class="com"># (T, T) probabilities</span>
O = P @ V                                       <span class="com"># (T, d) output</span></code></pre>

<p>Three operations, each writing its result to HBM. Per head: read <code>Q</code>, <code>K</code>, <code>V</code> (each <code>T × d</code>); write <code>S</code> (<code>T × T</code>); read <code>S</code> back; write <code>P</code>; read <code>P</code> back; write <code>O</code>. The (T, T) intermediates are the killers — for T=2048 and fp16, that's 8 MB <em>per head, per layer</em>. For a 32-layer 32-head transformer, peak activation memory from attention alone is ~16 GB at T=2048 — and it's <em>quadratic in T</em>.</p>

<p>Roofline analysis (M22) on naive attention at typical sizes:</p>

<div class="table-wrap">
<table>
<caption>Naive attention vs FlashAttention: arithmetic intensity at T=2048, d=64</caption>
<thead><tr><th></th><th>FLOPs</th><th>HBM bytes</th><th>Intensity</th><th>Regime (H100)</th></tr></thead>
<tbody>
<tr><td>Naive (writes (T,T) to HBM)</td><td>~33M</td><td>~50 MB</td><td>~0.7 ops/byte</td><td>Memory-bound (way below ridge)</td></tr>
<tr><td>FlashAttention (no (T,T) in HBM)</td><td>~33M</td><td>~5 MB</td><td>~7 ops/byte</td><td>Still memory-bound but ~10× higher</td></tr>
<tr><td>FlashAttention with larger heads (d=128)</td><td>~66M</td><td>~5 MB</td><td>~13 ops/byte</td><td>Approaching compute-bound</td></tr>
</tbody>
</table>
</div>

<p>Naive attention is dominated by the (T, T) writes and reads. FlashAttention removes them. The kernel does the same FLOPs but ~10× less HBM traffic. <em>Same compute, far less memory — the textbook fusion win, applied to attention.</em></p>

<h2>The online softmax trick</h2>

<p>The mathematical core. Standard softmax over a vector x:</p>

<pre><code>softmax(x_i) = exp(x_i) / sum_j(exp(x_j))</code></pre>

<p>To do this safely in floating point, you subtract the max before exponentiating (otherwise <code>exp</code> overflows for large values):</p>

<pre><code>m = max(x)
softmax(x_i) = exp(x_i - m) / sum_j(exp(x_j - m))</code></pre>

<p>This requires <em>two passes</em> over x: one to compute m (the max), one to compute the sum and divide. If x is a row of the attention matrix S, that means materializing S — exactly what we want to avoid.</p>

<p>The trick: <em>can we compute softmax incrementally, looking at one block of x at a time, without seeing the rest?</em> Yes — by maintaining a running max and a running sum, and "patching" them when a new larger max appears.</p>

<p>Suppose we've seen blocks <code>x⁽¹⁾</code> and <code>x⁽²⁾</code>, and we have:</p>

<pre><code>m₁ = max(x⁽¹⁾)                                    <span class="com"># current running max</span>
ℓ₁ = sum_j(exp(x⁽¹⁾_j - m₁))                       <span class="com"># sum of exps, normalized to current max</span></code></pre>

<p>Now block <code>x⁽²⁾</code> arrives, with its own local max <code>m₂_local</code>. The new global max is:</p>

<pre><code>m₂ = max(m₁, m₂_local)</code></pre>

<p>To get the new running sum (referenced to <code>m₂</code> instead of <code>m₁</code>), we need to <em>rescale</em> the old sum and add the new contribution:</p>

<pre><code>ℓ₂ = ℓ₁ * exp(m₁ - m₂) + sum_j(exp(x⁽²⁾_j - m₂))</code></pre>

<p>The key term is <code>exp(m₁ - m₂)</code>. If the new max is bigger (<code>m₂ &gt; m₁</code>), this is &lt; 1 and shrinks the old contributions to match the new normalization. If the new max is the same (<code>m₂ = m₁</code>), it's 1 and nothing changes. <strong>The same patching trick works for the partial output P @ V</strong> — when the running max updates, you rescale the partial output by <code>exp(m_old - m_new)</code>.</p>

<p>Worked example with three values, processed one at a time:</p>

<div class="ascii">x = [2, 5, 1]

After x[0]=2:  m=2,    ℓ=exp(0)=1
After x[1]=5:  m=5,    ℓ_old_rescaled = 1*exp(2-5) = 0.0498
                       ℓ_new = 0.0498 + exp(0) = 1.0498
After x[2]=1:  m=5 (unchanged), 
                       ℓ_new = 1.0498 + exp(1-5) = 1.0498 + 0.0183 = 1.0681

Standard softmax for comparison:
  m = 5
  ℓ = exp(2-5) + exp(5-5) + exp(1-5) = 0.0498 + 1 + 0.0183 = 1.0681  ✓</div>

<p>Same answer. We never had to see all three values at once — we processed them one at a time, maintaining (m, ℓ) and rescaling when needed.</p>

<p>This is the entire conceptual breakthrough. <strong>You can compute exact softmax by streaming.</strong> Apply this to the rows of S — process K (and V) in blocks, maintain running (max, sum, partial output) per row of Q, patch as needed. The full S never exists.</p>

<h2>The FlashAttention algorithm (forward)</h2>

<p>Here's the algorithm in pseudocode. Each program in the kernel handles one tile of Q (BLOCK_M rows). It iterates over K and V in BLOCK_N column tiles, accumulating into the output and the running stats.</p>

<pre><code><span class="com"># For one Q tile: rows [r..r+BLOCK_M] of Q</span>
load Q_tile  ([BLOCK_M, d])  into shared/registers, KEEP IT THERE
initialize:
    m_running = -inf  ([BLOCK_M])               <span class="com"># running max per row</span>
    ℓ_running = 0     ([BLOCK_M])               <span class="com"># running sum-of-exps per row</span>
    O_running = 0     ([BLOCK_M, d])            <span class="com"># running partial output per row</span>

<span class="kw">for</span> each K, V column tile (cols [c..c+BLOCK_N]):
    load K_tile, V_tile  ([BLOCK_N, d])
    
    <span class="com"># 1. Compute attention scores for this tile</span>
    S_tile = Q_tile @ K_tile.T / sqrt(d)        <span class="com"># [BLOCK_M, BLOCK_N]</span>
    apply causal mask if needed (set future positions to -inf)
    
    <span class="com"># 2. Online softmax update</span>
    m_tile = max(S_tile, axis=1)                <span class="com"># [BLOCK_M]</span>
    m_new = max(m_running, m_tile)              <span class="com"># [BLOCK_M]</span>
    
    <span class="com"># Rescale running stats to new max</span>
    α = exp(m_running - m_new)                  <span class="com"># [BLOCK_M] — typically &lt; 1</span>
    ℓ_running = α * ℓ_running
    O_running = α[:, None] * O_running
    
    <span class="com"># Add new contribution</span>
    P_tile = exp(S_tile - m_new[:, None])       <span class="com"># [BLOCK_M, BLOCK_N]</span>
    ℓ_running += sum(P_tile, axis=1)
    O_running += P_tile @ V_tile                <span class="com"># [BLOCK_M, d]</span>
    
    m_running = m_new

<span class="com"># Final normalization</span>
O = O_running / ℓ_running[:, None]              <span class="com"># [BLOCK_M, d] — actual output</span>
write O to HBM at row positions [r..r+BLOCK_M]

<span class="com"># For backward: save logsumexp (= m + log(ℓ)), not the full attention matrix</span>
write LSE = m_running + log(ℓ_running) to HBM   <span class="com"># [BLOCK_M] per Q tile</span></code></pre>

<p>Read the algorithm carefully. Three things make it work:</p>

<ol>
  <li><strong>Q tile is "pinned" in shared memory for the whole iteration</strong>. K and V tiles stream through. This is the asymmetry — Q is loaded once per output tile; K and V are loaded once per K/V tile. Total HBM reads: O(T·d) instead of O(T²).</li>
  <li><strong>The (T, T) attention matrix never exists</strong>. <code>S_tile</code> is a (BLOCK_M, BLOCK_N) tile that lives in registers/shared memory for the duration of one inner-loop iteration. It's discarded as soon as we update the running stats and output.</li>
  <li><strong>The output is built incrementally</strong>. After processing all K/V tiles, <code>O_running / ℓ_running</code> gives the same result as <code>softmax(S) @ V</code> would have — exactly, not approximately. This is provable from the math above.</li>
</ol>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 380" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrFA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">FlashAttention: pin one Q tile, stream K and V tiles through</text>

  <!-- Q tile pinned in shared memory -->
  <g transform="translate(40, 50)">
    <text x="60" y="-5" text-anchor="middle" font-size="11" font-weight="700" fill="#1f5f5b">Q tile (pinned)</text>
    <rect x="0" y="0" width="120" height="100" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2.5"/>
    <text x="60" y="40" text-anchor="middle" font-size="11" fill="#1a1612">[BLOCK_M, d]</text>
    <text x="60" y="55" text-anchor="middle" font-size="11" fill="#1a1612">in shared mem</text>
    <text x="60" y="75" text-anchor="middle" font-size="9" fill="#1f5f5b" font-style="italic">stays for whole loop</text>
    <text x="60" y="92" text-anchor="middle" font-size="9" fill="#1f5f5b" font-style="italic">never re-read from HBM</text>
  </g>

  <!-- Streaming K, V tiles -->
  <g transform="translate(220, 50)">
    <text x="60" y="-5" text-anchor="middle" font-size="11" font-weight="700" fill="#c1502e">K, V tiles (stream)</text>
    <rect x="0"  y="0" width="38" height="100" fill="#fcecec" stroke="#c1502e" stroke-width="1"/>
    <text x="19" y="55" text-anchor="middle" font-size="9" fill="#1a1612">K1</text>
    <rect x="42" y="0" width="38" height="100" fill="#fcecec" stroke="#c1502e" stroke-width="1"/>
    <text x="61" y="55" text-anchor="middle" font-size="9" fill="#1a1612">K2</text>
    <rect x="84" y="0" width="38" height="100" fill="#fcecec" stroke="#c1502e" stroke-width="1"/>
    <text x="103" y="55" text-anchor="middle" font-size="9" fill="#1a1612">K3</text>
    <text x="60" y="118" text-anchor="middle" font-size="9" fill="#c1502e">[BLOCK_N, d] each</text>
  </g>

  <!-- Inner-loop compute box -->
  <g transform="translate(380, 50)">
    <rect x="0" y="0" width="180" height="100" fill="#fff5d8" stroke="#d4a017" stroke-width="2"/>
    <text x="90" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">For each K, V tile:</text>
    <text x="10" y="40" font-size="10" fill="#1a1612">  S = Q @ K.T / √d</text>
    <text x="10" y="54" font-size="10" fill="#1a1612">  P = exp(S - m_new)</text>
    <text x="10" y="68" font-size="10" fill="#1a1612">  ℓ += sum(P)</text>
    <text x="10" y="82" font-size="10" fill="#1a1612">  O += P @ V</text>
    <text x="10" y="96" font-size="9" fill="#c1502e">  → patch with α = exp(m_old−m_new)</text>
  </g>

  <!-- Output -->
  <g transform="translate(580, 50)">
    <text x="60" y="-5" text-anchor="middle" font-size="11" font-weight="700" fill="#b85a6c">O tile (built up)</text>
    <rect x="0" y="0" width="120" height="100" fill="#ffd5dc" stroke="#b85a6c" stroke-width="2"/>
    <text x="60" y="40" text-anchor="middle" font-size="11" fill="#1a1612">[BLOCK_M, d]</text>
    <text x="60" y="55" text-anchor="middle" font-size="11" fill="#1a1612">in registers</text>
    <text x="60" y="75" text-anchor="middle" font-size="9" fill="#b85a6c" font-style="italic">written to HBM</text>
    <text x="60" y="90" text-anchor="middle" font-size="9" fill="#b85a6c" font-style="italic">at the end</text>
  </g>

  <!-- Arrows -->
  <path d="M 165 100 L 215 100" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrFA)"/>
  <path d="M 345 100 L 375 100" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrFA)"/>
  <path d="M 565 100 L 575 100" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrFA)"/>

  <!-- Memory traffic comparison -->
  <text x="20" y="200" font-size="13" font-weight="700" fill="#1a1612">HBM traffic: naive vs FlashAttention (per Q tile)</text>

  <g transform="translate(20, 215)">
    <text x="0" y="14" font-size="11" font-weight="700" fill="#c1502e">Naive:</text>
    <rect x="80" y="3" width="600" height="14" fill="#fcecec" stroke="#c1502e" stroke-width="1"/>
    <text x="380" y="13" text-anchor="middle" font-size="10" fill="#1a1612">Q + K + V + S(write) + S(read) + P(write) + P(read) + O</text>
    <text x="690" y="13" font-size="10" fill="#c1502e">≈ 5T·d + 2T²</text>
  </g>

  <g transform="translate(20, 240)">
    <text x="0" y="14" font-size="11" font-weight="700" fill="#1f5f5b">Flash:</text>
    <rect x="80" y="3" width="180" height="14" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/>
    <text x="170" y="13" text-anchor="middle" font-size="10" fill="#1a1612">Q + K + V + O</text>
    <text x="270" y="13" font-size="10" fill="#1f5f5b">≈ 4T·d  (no T² term!)</text>
  </g>

  <!-- Annotations -->
  <text x="370" y="295" font-family="'Caveat', cursive" font-size="20" fill="#1a1612" text-anchor="middle">naive: T² term dominates at long context (T=8K → ~64 MB intermediate)</text>
  <text x="370" y="320" font-family="'Caveat', cursive" font-size="20" fill="#1f5f5b" text-anchor="middle">flash: linear in T → same kernel scales to 32K, 128K, beyond</text>
  <text x="370" y="350" font-family="'Caveat', cursive" font-size="22" fill="#c1502e" text-anchor="middle">this is what makes long-context training and inference tractable</text>
</svg>
</div>

<p>Stare at this picture. The Q tile sits in shared memory for the whole iteration. K and V tiles stream through, one inner-loop iteration at a time. The (BLOCK_M, BLOCK_N) attention scores tile lives only in registers during that one iteration, then is discarded. The output tile is incrementally built up in registers, written to HBM only at the very end.</p>

<p><strong>The (T, T) intermediate never exists in HBM.</strong> That's the entire trick.</p>

<h2>The Triton implementation (simplified)</h2>

<p>Real production FlashAttention has many bells and whistles (varying head dims, dropout, alibi, paged KV, etc.). Here's the conceptual core stripped down. Read it next to the algorithm above.</p>

<pre><code><span class="kw">import</span> triton
<span class="kw">import</span> triton.language <span class="kw">as</span> tl

<span class="kw">@</span>triton.<span class="fn">jit</span>
<span class="kw">def</span> <span class="fn">flash_attn_fwd_kernel</span>(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    <span class="com"># ... K, V, O strides similar ...</span>
    H, M, N, D,
    softmax_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    <span class="com"># Each program handles one (Q tile) for one (batch, head).</span>
    pid_m = tl.<span class="fn">program_id</span>(<span class="num">0</span>)         <span class="com"># Q tile index</span>
    pid_bh = tl.<span class="fn">program_id</span>(<span class="num">1</span>)        <span class="com"># batch * head index</span>

    <span class="com"># Compute pointers to this Q tile's data.</span>
    offs_m = pid_m * BLOCK_M + tl.<span class="fn">arange</span>(<span class="num">0</span>, BLOCK_M)
    offs_d = tl.<span class="fn">arange</span>(<span class="num">0</span>, BLOCK_D)

    Q_block_ptr = Q_ptr + pid_bh * stride_qh \
                  + offs_m[:, <span class="kw">None</span>] * stride_qm + offs_d[<span class="kw">None</span>, :] * stride_qd

    <span class="com"># Load Q tile ONCE — stays in registers/shared for whole loop.</span>
    q = tl.<span class="fn">load</span>(Q_block_ptr, mask=offs_m[:, <span class="kw">None</span>] &lt; M, other=<span class="num">0.</span>)

    <span class="com"># Initialize running stats.</span>
    m_i = tl.<span class="fn">full</span>([BLOCK_M], -<span class="fn">float</span>(<span class="str">"inf"</span>), dtype=tl.float32)
    l_i = tl.<span class="fn">zeros</span>([BLOCK_M], dtype=tl.float32)
    acc = tl.<span class="fn">zeros</span>([BLOCK_M, BLOCK_D], dtype=tl.float32)

    <span class="com"># Determine the range of K/V tiles to iterate over.</span>
    <span class="kw">if</span> IS_CAUSAL:
        <span class="com"># With causal masking, only iterate over K/V tiles up to this Q tile.</span>
        n_end = (pid_m + <span class="num">1</span>) * BLOCK_M
    <span class="kw">else</span>:
        n_end = N

    <span class="com"># Inner loop: stream K and V tiles.</span>
    <span class="kw">for</span> start_n <span class="kw">in</span> <span class="fn">range</span>(<span class="num">0</span>, n_end, BLOCK_N):
        offs_n = start_n + tl.<span class="fn">arange</span>(<span class="num">0</span>, BLOCK_N)

        <span class="com"># Load K and V tiles.</span>
        k = tl.<span class="fn">load</span>(...)            <span class="com"># [BLOCK_N, BLOCK_D]</span>
        v = tl.<span class="fn">load</span>(...)            <span class="com"># [BLOCK_N, BLOCK_D]</span>

        <span class="com"># 1. Attention scores for this tile.</span>
        s = tl.<span class="fn">dot</span>(q, k.<span class="fn">trans</span>()) * softmax_scale    <span class="com"># [BLOCK_M, BLOCK_N], in fp32</span>

        <span class="com"># 2. Apply causal mask if needed.</span>
        <span class="kw">if</span> IS_CAUSAL:
            mask = offs_m[:, <span class="kw">None</span>] &gt;= offs_n[<span class="kw">None</span>, :]
            s = tl.<span class="fn">where</span>(mask, s, -<span class="fn">float</span>(<span class="str">"inf"</span>))

        <span class="com"># 3. Online softmax update.</span>
        m_ij = tl.<span class="fn">max</span>(s, axis=<span class="num">1</span>)        <span class="com"># [BLOCK_M] — local max for this tile</span>
        m_new = tl.<span class="fn">maximum</span>(m_i, m_ij) <span class="com"># [BLOCK_M] — new global max</span>

        <span class="com"># Rescale running stats to new max.</span>
        alpha = tl.<span class="fn">exp</span>(m_i - m_new)
        l_i = l_i * alpha
        acc = acc * alpha[:, <span class="kw">None</span>]

        <span class="com"># Add this tile's contribution.</span>
        p = tl.<span class="fn">exp</span>(s - m_new[:, <span class="kw">None</span>])  <span class="com"># [BLOCK_M, BLOCK_N]</span>
        l_i = l_i + tl.<span class="fn">sum</span>(p, axis=<span class="num">1</span>)
        acc = acc + tl.<span class="fn">dot</span>(p.<span class="fn">to</span>(v.dtype), v)  <span class="com"># tensor cores fire here</span>

        m_i = m_new

    <span class="com"># Final normalization.</span>
    acc = acc / l_i[:, <span class="kw">None</span>]
    lse = m_i + tl.<span class="fn">log</span>(l_i)        <span class="com"># save for backward</span>

    <span class="com"># Write output and LSE.</span>
    tl.<span class="fn">store</span>(O_ptr + ..., acc.<span class="fn">to</span>(Q_ptr.dtype.element_ty), mask=...)
    tl.<span class="fn">store</span>(LSE_ptr + ..., lse, mask=offs_m &lt; M)</code></pre>

<p>This is a real Triton kernel — the production FlashAttention-2 kernel is essentially this with more autotune configs, more head dimension variants, and dropout/alibi/etc. options. The conceptual core fits in 50 lines.</p>

<p>Notable points to call out:</p>

<ul>
  <li><strong>Two <code>tl.dot</code> calls per inner iteration</strong>: <code>q @ k.T</code> for scores, <code>p @ v</code> for the output update. Both fire tensor cores. The kernel is dominated by these two matmuls per K/V tile.</li>
  <li><strong>fp32 throughout the algorithm body</strong>: <code>m_i, l_i, acc</code> are all fp32. Inputs are loaded in bf16/fp16 but immediately participate in fp32 ops. Same recipe as M14 / M23.</li>
  <li><strong>Causal masking is just <code>tl.where</code></strong>. Set future positions to <code>-inf</code> before the exp, and they contribute zero to the softmax. <em>Smart implementations skip K/V tiles entirely above the diagonal — no compute wasted on positions that will be fully masked.</em></li>
  <li><strong>LSE is saved for backward</strong>. We don't need to save the (T, T) attention matrix — just <code>logsumexp = m + log(ℓ)</code> per row. That's <code>O(T)</code> auxiliary storage, vs <code>O(T²)</code> for naive backward.</li>
</ul>

<h2>The backward: rematerialize, don't store</h2>

<p>Forward saved O and LSE. The backward pass reconstructs P from these — it doesn't read a saved attention matrix.</p>

<p>Recall <code>P = softmax(S)</code>. The forward saved <code>LSE = m + log(ℓ)</code>, the per-row log-sum-exp. We can recover P given S and LSE via:</p>

<pre><code>P_ij = exp(S_ij - LSE_i)        <span class="com"># numerically stable, exact</span></code></pre>

<p>So the backward is structured similarly to forward:</p>

<ol>
  <li>Iterate Q, K, V, O, LSE, dO tiles (we do an outer loop over K, V tiles for backward; the iteration pattern is different from forward for memory reasons).</li>
  <li>Recompute S = Q @ K.T / sqrt(d) for the current tile.</li>
  <li>Reconstruct P = exp(S - LSE) using the saved LSE — exact, not approximated.</li>
  <li>Compute the gradients: dV = P.T @ dO, dP = dO @ V.T, dS via softmax derivative, dQ and dK via standard matmul backprop.</li>
</ol>

<p>The recompute happens entirely inside the backward kernel, in shared memory and registers. The (T, T) attention matrix is reconstructed tile-by-tile, used, and discarded — same pattern as forward. <strong>This is exactly the activation-checkpointing trick from M12, but baked into the kernel:</strong> trade some recomputation for huge memory savings, with the recomputation cheap because it stays on-chip.</p>

<p>Result: backward memory is also O(T) instead of O(T²). Both forward and backward avoid the quadratic blowup.</p>

<h2>FlashAttention versions: a quick history</h2>

<div class="table-wrap">
<table>
<caption>The FlashAttention version line</caption>
<thead><tr><th>Version</th><th>Year</th><th>Key change</th><th>Speed</th></tr></thead>
<tbody>
<tr><td>FlashAttention v1</td><td>2022</td><td>Online softmax + tile-based fused kernel; the breakthrough.</td><td>~2-4× over naive</td></tr>
<tr><td>FlashAttention v2</td><td>2023</td><td>Better parallelism over batch+heads; reduced non-matmul flops; better warp partitioning. Reference Triton implementation.</td><td>~2× faster than v1</td></tr>
<tr><td>FlashAttention v3</td><td>2024</td><td>Hopper-specific (uses TMA for async loads, FP8 path, warp specialization). CUDA implementation.</td><td>~1.5-2× faster than v2 on H100</td></tr>
</tbody>
</table>
</div>

<p>For practical use:</p>

<ul>
  <li><strong>Don't write FlashAttention from scratch unless you're learning</strong>. The reference implementations (Tri Dao's <code>flash-attn</code> package, PyTorch's <code>scaled_dot_product_attention</code> backend) are heavily optimized.</li>
  <li><strong>PyTorch ships FlashAttention</strong>. <code>F.scaled_dot_product_attention(q, k, v, is_causal=True)</code> picks FlashAttention as the backend automatically when available. This is the right way to use it from training code.</li>
  <li><strong>If you want to study the kernel</strong>: read the v2 Triton implementation in the <a href="https://github.com/triton-lang/triton">Triton tutorials repo</a>. ~250 lines, well-commented, runs at near-cuBLAS speed.</li>
  <li><strong>If you want to extend it</strong> (custom mask patterns, sliding window, RoPE inside the kernel, etc.), forking the v2 Triton implementation is the standard starting point.</li>
</ul>

<h2>Why FlashAttention's pattern generalizes</h2>

<p>The "stream the big thing through, keep the small thing pinned in fast memory, never materialize the intermediate" pattern shows up everywhere now. M27 covers two prominent cases that descended directly from FlashAttention:</p>

<ul>
  <li><strong>Paged Attention</strong>: for inference KV cache. Same kernel idea but K and V live in non-contiguous "pages" of GPU memory, allowing efficient memory management for variable-length conversations.</li>
  <li><strong>Sliding-window attention</strong> (Mistral, Gemma): the same kernel with a window mask — only K/V tiles within the window are visited, the rest are skipped entirely. Linear in T instead of quadratic, no algorithmic change beyond the iteration bounds.</li>
</ul>

<p>And in less-attention contexts: similar techniques are used in <strong>fused softmax + cross-entropy + KL kernels</strong>, <strong>fused layer-norm + dropout + residual</strong>, <strong>fused MoE routing</strong>. Whenever you have an expensive intermediate that doesn't need to live in HBM, the FlashAttention pattern (online normalization + tile streaming + recompute backward) tells you how to eliminate it.</p>

<h2>Using FlashAttention from your training code</h2>

<p>You almost never write a FlashAttention kernel directly. PyTorch's <code>scaled_dot_product_attention</code> is the front door:</p>

<pre><code><span class="kw">import</span> torch.nn.functional <span class="kw">as</span> F

<span class="com"># PyTorch picks the best backend automatically: FlashAttention if available,</span>
<span class="com"># memory-efficient attention as fallback, or the math (naive) backend.</span>
out = F.<span class="fn">scaled_dot_product_attention</span>(
    q, k, v,
    attn_mask=<span class="kw">None</span>,
    is_causal=<span class="kw">True</span>,           <span class="com"># lets the kernel skip masked tiles efficiently</span>
    dropout_p=<span class="num">0.0</span>,
    scale=<span class="kw">None</span>,              <span class="com"># default = 1/sqrt(d)</span>
)</code></pre>

<p>You can inspect or restrict which backend is used:</p>

<pre><code><span class="kw">from</span> torch.nn.attention <span class="kw">import</span> sdpa_kernel, SDPBackend

<span class="kw">with</span> <span class="fn">sdpa_kernel</span>([SDPBackend.FLASH_ATTENTION]):
    out = F.<span class="fn">scaled_dot_product_attention</span>(q, k, v, is_causal=<span class="kw">True</span>)
    <span class="com"># Errors if FlashAttention isn't usable (e.g., wrong head dim, unsupported mask)</span></code></pre>

<p>FlashAttention is finicky about supported configurations:</p>

<ul>
  <li><strong>Head dim</strong> typically must be a multiple of 8 and ≤ 256 (varies by version).</li>
  <li><strong>dtype</strong> must be fp16 or bf16 (no fp32 path; fp8 in v3 only).</li>
  <li><strong>Mask</strong>: dense additive masks fall back to slower paths; <code>is_causal=True</code> uses the fast-causal optimization.</li>
  <li><strong>Dropout</strong> at non-zero rates is supported but adds overhead.</li>
</ul>

<p>If your config doesn't qualify, PyTorch falls back to the next-best backend (memory-efficient attention, then the math/naive backend) — no error, just less performance.</p>

<div class="ndq">
<h4>About FlashAttention</h4>

<p class="q">If FlashAttention is exact, why isn't it the only attention kernel?</p>
<p class="a">FlashAttention has tight constraints: specific dtypes (fp16/bf16/fp8 only), bounded head dimensions, particular mask patterns. For unusual setups (very large head dim, custom mask shapes, fp32, exotic biases like ALiBi or RoPE-in-kernel), other backends or custom kernels are needed. PyTorch's <code>scaled_dot_product_attention</code> picks among Flash, memory-efficient, and math backends based on what works for your inputs.</p>

<p class="q">Does FlashAttention save activations for backward?</p>
<p class="a">Yes — it saves Q, K, V, O, and LSE. That's O(T·d) per head, vs O(T²) for naive saving the attention matrix. The backward recomputes the attention scores from Q and K and reconstructs P from S and LSE — exact, not approximate.</p>

<p class="q">What's the speedup on a typical training step?</p>
<p class="a">Depends on T, d, and the rest of the model. For a transformer at T=2048, attention is maybe 25-40% of step time naive; FlashAttention cuts it by ~3-4×, giving a net 1.3-1.5× step speedup. At T=8K, attention dominates more (because it's quadratic in T), so the speedup is bigger — sometimes 2-3× total. At T=32K, naive doesn't even fit; FlashAttention is the only option.</p>

<p class="q">Why is FlashAttention important for inference, not just training?</p>
<p class="a">Inference with long prompts hits the same quadratic-memory wall. The KV cache for a 32K-token conversation × 32 layers × 32 heads × 128 head dim × 2 bytes = ~16 GB just for K and V. FlashAttention's tile streaming lets the prefill (reading the prompt) and the decode (generating one token at a time) both work efficiently with this much KV state. Combined with paged attention (M27), you can serve multiple long-context conversations on one GPU.</p>

<p class="q">What's "TMA" in FlashAttention v3?</p>
<p class="a">Tensor Memory Accelerator — a Hopper-specific hardware unit that does asynchronous bulk loads from HBM into shared memory. It frees up the warp scheduler to do compute while loads are in flight, increasing tensor core utilization. v3 also uses warp specialization (some warps do loads, others do compute, others do softmax) for better latency hiding. Heavily Hopper-specific; A100 doesn't have TMA.</p>

<p class="q">When does FlashAttention NOT help?</p>
<p class="a">Three cases. (1) Very short sequences (T &lt; 256) — fixed kernel overheads dominate, and a simple matmul-attention-matmul flow can be competitive. (2) Very small batches or head counts where the kernel can't fill the GPU. (3) Heads with extremely large d (&gt; 256) — falls back to other backends. For 95% of modern transformer training and inference, it helps; for those edge cases, the dispatcher (M19) routes elsewhere.</p>
</div>

<h2>The takeaway pattern</h2>

<p>FlashAttention is one specific algorithm, but the lessons generalize. The recipe:</p>

<ol>
  <li><strong>Identify the memory-bound bottleneck.</strong> Compute arithmetic intensity (M22). If it's well below the ridge point, the algorithm has a fusion opportunity.</li>
  <li><strong>Find the intermediate that lives in HBM but doesn't need to.</strong> For naive attention, it's the (T, T) attention matrix. For other algorithms it might be the softmax output, an intermediate broadcast, etc.</li>
  <li><strong>Restructure the computation to keep that intermediate in shared memory.</strong> Stream the big thing through; pin the small thing in fast memory; tile carefully so each tile fits in shared memory.</li>
  <li><strong>For incremental algorithms (softmax, normalization), find the running-state form.</strong> Online softmax for attention; running mean+variance for normalization; running argmax for top-k.</li>
  <li><strong>For backward, save just enough to recompute</strong> — typically a few O(T)-sized statistics rather than the full O(T²) intermediate.</li>
  <li><strong>Implement in Triton for portability</strong>; drop to CUDA only for hardware-specific tricks (TMA, warp specialization).</li>
</ol>

<p>This recipe will produce many of the kernels you'll write or want to understand. FlashAttention is the canonical example because attention is everywhere and the speedup is large — but the technique is universal.</p>

<h2>Code Magnets: implement an online softmax pass</h2>

<p>You're implementing the running-stats update for one inner-loop iteration of FlashAttention. Three magnets are wrong choices.</p>

<div class="magnets">
<p>Arrange the magnets into the correct online softmax + output update.</p>

<div class="magnet-pool">
  <span class="magnet">m_ij = tl.max(s, axis=1)</span>
  <span class="magnet">m_new = tl.maximum(m_i, m_ij)</span>
  <span class="magnet">m_new = m_ij</span>
  <span class="magnet">alpha = tl.exp(m_i - m_new)</span>
  <span class="magnet">alpha = tl.exp(m_new - m_i)</span>
  <span class="magnet">l_i = l_i * alpha</span>
  <span class="magnet">acc = acc * alpha[:, None]</span>
  <span class="magnet">p = tl.exp(s - m_new[:, None])</span>
  <span class="magnet">p = tl.softmax(s, axis=1)</span>
  <span class="magnet">l_i = l_i + tl.sum(p, axis=1)</span>
  <span class="magnet">acc = acc + tl.dot(p.to(v.dtype), v)</span>
  <span class="magnet">m_i = m_new</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code>m_ij = tl.<span class="fn">max</span>(s, axis=<span class="num">1</span>)
m_new = tl.<span class="fn">maximum</span>(m_i, m_ij)
alpha = tl.<span class="fn">exp</span>(m_i - m_new)
l_i = l_i * alpha
acc = acc * alpha[:, <span class="kw">None</span>]
p = tl.<span class="fn">exp</span>(s - m_new[:, <span class="kw">None</span>])
l_i = l_i + tl.<span class="fn">sum</span>(p, axis=<span class="num">1</span>)
acc = acc + tl.<span class="fn">dot</span>(p.<span class="fn">to</span>(v.dtype), v)
m_i = m_new</code></pre>
<p>The traps:</p>
<ul>
  <li><code>m_new = m_ij</code>: forgets the running max from previous iterations. The correct update is <code>m_new = max(m_i, m_ij)</code> — keep the larger of running and local.</li>
  <li><code>alpha = tl.exp(m_new - m_i)</code>: sign flipped. The rescale is <code>exp(m_old − m_new)</code> — typically &lt; 1 because m_new ≥ m_old. The flipped form would blow up.</li>
  <li><code>p = tl.softmax(s, axis=1)</code>: softmax over the tile gives <em>local</em> probabilities, normalized to this tile's sum. We want <code>exp(s − m_new)</code> — unnormalized but referenced to the global running max. Normalization happens at the very end via <code>acc / l_i</code>.</li>
</ul>
<p>The correct sequence: <strong>find local max → update global max → rescale running stats with α → compute exp(s − global_max) → accumulate into ℓ and acc → record the new global max</strong>. Each step depends on the one before; reordering breaks the math.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each FlashAttention concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>Online softmax</div>
  <div>A. Compute exact softmax incrementally with a running (max, sum) — no need to see the whole row at once.</div>

  <div>Q tile pinned, K/V streamed</div>
  <div>B. Outer loop over Q tiles; inner loop over K/V tiles. Asymmetry that minimizes HBM reads.</div>

  <div>(T, T) attention matrix never in HBM</div>
  <div>C. Per-tile S lives only in registers/shared during one iteration; discarded after.</div>

  <div>LSE saved for backward</div>
  <div>D. log-sum-exp per Q row — O(T) instead of O(T²) auxiliary storage.</div>

  <div>Recompute attention in backward</div>
  <div>E. Reconstruct P = exp(S − LSE) on the fly inside the backward kernel.</div>

  <div>F.scaled_dot_product_attention</div>
  <div>F. PyTorch's high-level entry point that auto-picks FlashAttention when applicable.</div>

  <div>FlashAttention v3 + TMA</div>
  <div>G. Hopper-specific: tensor memory accelerator + warp specialization for max H100 throughput.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>Online softmax</strong> → A<br>
<strong>Q tile pinned, K/V streamed</strong> → B<br>
<strong>(T, T) attention matrix never in HBM</strong> → C<br>
<strong>LSE saved for backward</strong> → D<br>
<strong>Recompute attention in backward</strong> → E<br>
<strong>F.scaled_dot_product_attention</strong> → F<br>
<strong>FlashAttention v3 + TMA</strong> → G
</p>
<p>The mental shortcut: <em>online softmax = streaming exact softmax, Q pinned + K/V streamed = the asymmetry, no (T,T) in HBM = the win, LSE = the only state we save, recompute in backward = M12 trick inside the kernel, F.scaled_dot_product_attention = the front door, v3 + TMA = Hopper-specific extras</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> Show that the online softmax algorithm produces exactly the same result as the standard two-pass version, for a row processed in two blocks of equal size.</p>
<details class="answer"><summary>show answer</summary>
<p>Let x = [a₁, a₂; b₁, b₂] (block 1: a's; block 2: b's). Standard softmax:</p>
<pre>m = max(a₁, a₂, b₁, b₂)
ℓ = exp(a₁−m) + exp(a₂−m) + exp(b₁−m) + exp(b₂−m)
softmax_i = exp(x_i − m) / ℓ</pre>
<p>Online algorithm:</p>
<p>After block 1: <code>m₁ = max(a₁, a₂)</code>, <code>ℓ₁ = exp(a₁−m₁) + exp(a₂−m₁)</code>.</p>
<p>After block 2: <code>m₂_local = max(b₁, b₂)</code>, <code>m_new = max(m₁, m₂_local) = m</code> (the global max).</p>
<p><code>α = exp(m₁ − m_new)</code>. New <code>ℓ = α·ℓ₁ + exp(b₁−m_new) + exp(b₂−m_new)</code>.</p>
<p>Substituting: <code>α·ℓ₁ = exp(m₁−m_new) · [exp(a₁−m₁) + exp(a₂−m₁)] = exp(a₁−m_new) + exp(a₂−m_new)</code>. So:</p>
<pre>new ℓ = exp(a₁−m_new) + exp(a₂−m_new) + exp(b₁−m_new) + exp(b₂−m_new)
      = ℓ_standard ✓</pre>
<p>Identical. The α rescaling reconstructs the contributions of block 1 referenced to the new global max — exactly what we'd have computed in a single pass. The whole point: <em>online softmax isn't an approximation; it's an algebraic refactoring</em>.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Why does causal masking let FlashAttention skip K/V tiles entirely (not just mask their entries to −inf)?</p>
<details class="answer"><summary>show answer</summary>
<p>Causal masking sets attention[i, j] = −inf for j &gt; i (token i can't attend to future tokens). For an entire K/V tile that lies <em>entirely</em> in the upper triangle (all positions j &gt; all positions i in the current Q tile), every entry would be masked to −inf, contribute zero to the softmax, and zero to the output. <em>So skip that tile entirely</em> — don't load K/V from HBM, don't run the matmul. The inner loop just iterates from <code>start_n=0</code> to <code>n_end = (pid_m + 1) * BLOCK_M</code> instead of all the way to N. <strong>This gives causal FlashAttention ~2× speedup over non-causal at the same T</strong> — half the K/V tiles are skipped. The optimization is in the loop bounds, not in the masking inside the kernel.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> Why does FlashAttention save Q, K, V, O, and LSE for backward instead of just LSE?</p>
<details class="answer"><summary>show answer</summary>
<p>The backward needs to compute gradients dQ, dK, dV. The chain rule formulas involve P (attention matrix), S (scores), Q, K, V, O, and dO. We've already chosen not to save P or S (that's the whole point of FlashAttention). So we save the O(T·d) tensors that are inputs/outputs we can't reconstruct — Q, K, V, O — plus LSE which lets us recompute P = exp(S − LSE) given S, and S can be recomputed from Q and K. So:</p>
<ul>
  <li>Q, K, V: needed to recompute S = Q @ K.T / sqrt(d).</li>
  <li>LSE: needed to reconstruct P from S without re-doing the softmax (which would need another running-max pass).</li>
  <li>O and dO: needed for the dV = P.T @ dO and dP = dO @ V.T computations.</li>
</ul>
<p>Total: ~5T·d bytes per head per layer. Same order as just storing the activations naively (which would also be ~T·d for Q, K, V), no quadratic blowup.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> A team adopts FlashAttention via <code>F.scaled_dot_product_attention</code> and sees big speedups for their T=4K transformer. They then increase head dim from 64 to 256, expecting another speedup. Instead they get a slowdown. What's likely happening?</p>
<details class="answer"><summary>show answer</summary>
<p>FlashAttention's per-tile shared memory budget scales with head dim. At d=64, a (BLOCK_M=128, d=64) Q tile needs 128 × 64 × 2 = 16 KB, easy to fit alongside the K/V tile and stats. At d=256, the Q tile is 128 × 256 × 2 = 64 KB — plus the K/V tile of similar size, plus stats. The kernel may have to shrink BLOCK_M (fewer Q rows per program) to fit shared memory budget, reducing tensor-core utilization. Or the FlashAttention backend may not support d=256 at all (some versions cap at 128 or 192) and PyTorch falls back to the math backend. Diagnosis: profile with <code>nvidia-smi</code> + the PyTorch profiler — check which SDP backend is active via <code>sdpa_kernel</code> or the dispatcher trace. <strong>Head dimensions matter for kernel performance, not just model quality</strong> — most production transformers stick to d=64 or 128 partly because of kernel friendliness.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li><strong>Naive attention is memory-bound</strong>: it materializes a (T, T) intermediate in HBM, doing O(T²) memory traffic. Roofline analysis shows it's far below the ridge.</li>
  <li><strong>The online softmax trick</strong>: compute softmax exactly while streaming the input in blocks, by maintaining a running (max, sum) and rescaling old contributions when a new block's max exceeds the running max. <code>α = exp(m_old − m_new)</code> is the key term.</li>
  <li><strong>FlashAttention applies online softmax to attention</strong>: pin a Q tile in shared memory, iterate K and V in tiles, accumulate (running max, running sum-of-exps, partial output) for each Q row. The (T, T) attention matrix never exists in HBM.</li>
  <li><strong>Memory traffic drops from O(T·d + T²) to O(T·d)</strong>. Memory-bound attention becomes much closer to compute-bound.</li>
  <li><strong>The kernel uses two <code>tl.dot</code> calls per inner iteration</strong>: Q @ K.T for scores, P @ V for output update. Both fire tensor cores. fp32 accumulators throughout.</li>
  <li><strong>Causal masking is a loop-bound optimization</strong> — skip K/V tiles entirely above the diagonal, ~2× speedup over non-causal at same T.</li>
  <li><strong>Backward saves Q, K, V, O, LSE</strong> — O(T) auxiliary state instead of O(T²). Reconstruct P from S and LSE inside the backward kernel; same recompute trick as M12 activation checkpointing, baked into the kernel.</li>
  <li><strong>FlashAttention versions</strong>: v1 (2022) introduced the trick; v2 (2023) is the standard production reference, written in Triton; v3 (2024) is Hopper-specific with TMA and warp specialization.</li>
  <li><strong>Use it through <code>F.scaled_dot_product_attention</code></strong>. PyTorch picks FlashAttention as the backend when configurations allow (fp16/bf16, head dim ≤ 256, supported masks).</li>
  <li><strong>The pattern generalizes</strong>: paged attention (M27), sliding-window attention, fused softmax+CE losses, fused norm+dropout — all use the same "stream the big thing, pin the small thing, never materialize the intermediate" recipe.</li>
  <li>The reflex when designing a kernel: identify the memory-bound bottleneck, find the intermediate that doesn't need HBM, find the running-state form of any normalization, and use the recompute trick for backward.</li>
</ul>
</div>

<p>Module 26 takes everything from Part VIII so far and applies it to <strong>quantization</strong>. We'll cover post-training quantization, GPTQ, AWQ, and the int4/int8/fp8 inference recipes — all hand-written kernel territory because efficient int4 matmul doesn't exist in cuBLAS. Then M27 covers inference systems (KV cache, paged attention, speculative decoding, vLLM-style serving). M28 closes the course with MoE and a frontier capstone.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">25</span>
  <span>FlashAttention: a case study</span>
</div>
"""

emit("25_flashattention", "Module 25 — FlashAttention: a case study", BODY)
