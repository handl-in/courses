#!/usr/bin/env python3
"""Module 24: Triton — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part VIII · Module 24</div>
  <h1 class="module-title">Triton: <em>Python kernels at CUDA speed</em></h1>
  <p class="module-sub">— the language Inductor writes, the language FlashAttention is written in, and the right tool for nearly every custom kernel you'll write</p>
</div>

<p>M23 walked through writing a CUDA extension end-to-end: a 60-line kernel, a 30-line C++ binding, a 20-line Python wrapper, and a 30-line registration block. Roughly 140 lines of code to express "do RMSNorm fast." This module rewrites the same thing in Triton in about 30 lines of pure Python — and adds autotuning while we're at it.</p>

<p>Triton is a domain-specific language for GPU kernels, embedded in Python. The key idea: instead of writing per-<em>thread</em> code (M23's CUDA pattern), you write per-<em>block</em> code that operates on small tensor tiles. The Triton compiler maps your tile operations onto warps, threads, registers, and shared memory automatically. You keep the parts you need to control (tile sizes, memory access patterns, parallelism strategy) and lose the parts you don't (warp scheduling, thread indexing, register allocation).</p>

<div class="keyidea">
A Triton kernel is a Python function decorated with <code>@triton.jit</code>, written in terms of <em>blocks</em> (program instances) and <em>tiles</em> (tensors of compile-time-known shape). Inside the kernel, you use <code>tl.load</code> / <code>tl.store</code> for HBM I/O with explicit masks, <code>tl.dot</code> for matmul on tensor cores, and standard arithmetic (<code>tl.sum</code>, <code>tl.exp</code>, etc.) on tiles. The Triton compiler handles the per-thread mapping, register allocation, and shared memory placement. <strong><code>@triton.autotune</code></strong> searches over block sizes and warp counts to pick the fastest configuration for your shapes — the autotuning is what makes Triton kernels frequently match cuBLAS and FlashAttention performance with a fraction of the code.
</div>

<h2>One new face</h2>

<div class="character" style="--c: #c1502e;">
  <div class="avatar" style="background: #c1502e; color: #fff;">⛧</div>
  <div>
    <p class="who">Triton</p>
    <p class="name">"I write blocks, not threads. The compiler handles warps for you."</p>
    <p class="says">CUDA makes you think one thread at a time. I make you think one <em>tile</em> at a time. Each instance of my kernel handles a tile of, say, <code>(BLOCK_M, BLOCK_N) = (128, 64)</code>. You describe what to do with that whole tile — load it, multiply it, sum it, store it — and I figure out how to map it onto warps and threads underneath. <em>You still need to understand the hardware</em> (M22): the tile sizes you pick determine occupancy, shared memory usage, and tensor core utilization. But you don't write the per-thread bookkeeping. The result: kernels that fit on one screen and run at 80-95% of cuBLAS speed, with autotuning baked in.</p>
  </div>
</div>

<h2>RMSNorm in Triton</h2>

<p>Side-by-side: the same algorithm in both languages.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 320" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">RMSNorm: CUDA (M23) vs Triton (M24) — same kernel, same speed</text>

  <!-- CUDA box -->
  <g transform="translate(20, 50)">
    <rect x="0" y="0" width="350" height="240" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="175" y="20" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">CUDA (rmsnorm.cu)</text>
    <text x="10" y="40" font-size="10" fill="#1a1612">~60 lines kernel + 30 lines C++ binding</text>
    <text x="10" y="55" font-size="10" fill="#1a1612">+ AT_DISPATCH macros for dtypes</text>
    <text x="10" y="70" font-size="10" fill="#1a1612">+ shared memory tree reduction by hand</text>
    <text x="10" y="85" font-size="10" fill="#1a1612">+ thread/block indexing arithmetic</text>
    <text x="10" y="100" font-size="10" fill="#1a1612">+ __syncthreads barriers</text>
    <text x="10" y="115" font-size="10" fill="#1a1612">+ PYBIND11 registration</text>
    <text x="10" y="130" font-size="10" fill="#1a1612">+ JIT or setuptools build</text>
    <text x="10" y="155" font-size="11" font-weight="700" fill="#c1502e">~140 LOC total</text>

    <text x="10" y="185" font-size="10" fill="#1a1612">build time: 30-60s on first call</text>
    <text x="10" y="200" font-size="10" fill="#1a1612">block size: hand-picked (BLOCK=256)</text>
    <text x="10" y="215" font-size="10" fill="#1a1612">autotuning: write your own</text>
    <text x="10" y="230" font-size="10" fill="#1a1612">debugging: CUDA_LAUNCH_BLOCKING=1</text>
  </g>

  <!-- Triton box -->
  <g transform="translate(390, 50)">
    <rect x="0" y="0" width="330" height="240" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="165" y="20" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">Triton (rmsnorm.py)</text>
    <text x="10" y="40" font-size="10" fill="#1a1612">~25 lines @triton.jit kernel</text>
    <text x="10" y="55" font-size="10" fill="#1a1612">+ ~5 lines launcher with cdiv</text>
    <text x="10" y="70" font-size="10" fill="#1a1612">→ tl.load / tl.sum / tl.store</text>
    <text x="10" y="85" font-size="10" fill="#1a1612">→ no thread/block indexing</text>
    <text x="10" y="100" font-size="10" fill="#1a1612">→ no __syncthreads</text>
    <text x="10" y="115" font-size="10" fill="#1a1612">→ no PYBIND11</text>
    <text x="10" y="130" font-size="10" fill="#1a1612">→ no separate build step</text>
    <text x="10" y="155" font-size="11" font-weight="700" fill="#1f5f5b">~30 LOC total</text>

    <text x="10" y="185" font-size="10" fill="#1a1612">build time: ~1s (first call only)</text>
    <text x="10" y="200" font-size="10" fill="#1a1612">block size: @triton.autotune searches</text>
    <text x="10" y="215" font-size="10" fill="#1a1612">autotuning: free, decorator</text>
    <text x="10" y="230" font-size="10" fill="#1a1612">debugging: TRITON_INTERPRET=1</text>
  </g>

  <text x="370" y="310" font-family="'Caveat', cursive" font-size="20" fill="#c1502e" text-anchor="middle">same speed, ~5× less code, ~30× faster compile, autotuning included</text>
</svg>
</div>

<p>Here's the actual Triton code. Read top-to-bottom; this is a complete, working kernel:</p>

<pre><code><span class="kw">import</span> torch
<span class="kw">import</span> triton
<span class="kw">import</span> triton.language <span class="kw">as</span> tl

<span class="kw">@</span>triton.<span class="fn">jit</span>
<span class="kw">def</span> <span class="fn">rmsnorm_fwd_kernel</span>(
    x_ptr, w_ptr, y_ptr, rstd_ptr,
    M, D, eps,
    BLOCK_SIZE: tl.constexpr,        <span class="com"># compile-time constant</span>
):
    <span class="com"># Each program instance handles ONE row.</span>
    row = tl.<span class="fn">program_id</span>(<span class="num">0</span>)

    <span class="com"># Compute pointers + a mask for boundary handling.</span>
    cols = tl.<span class="fn">arange</span>(<span class="num">0</span>, BLOCK_SIZE)
    mask = cols &lt; D
    x_ptrs = x_ptr + row * D + cols
    y_ptrs = y_ptr + row * D + cols

    <span class="com"># Load the full row (BLOCK_SIZE elements; mask handles tail).</span>
    x = tl.<span class="fn">load</span>(x_ptrs, mask=mask, other=<span class="num">0.</span>).<span class="fn">to</span>(tl.float32)

    <span class="com"># Compute mean of squares, then 1/RMS, in fp32.</span>
    mean_sq = tl.<span class="fn">sum</span>(x * x, axis=<span class="num">0</span>) / D
    inv_rms = <span class="num">1.0</span> / tl.<span class="fn">sqrt</span>(mean_sq + eps)
    tl.<span class="fn">store</span>(rstd_ptr + row, inv_rms)

    <span class="com"># Apply the rescale + weight; cast back to input dtype on store.</span>
    w = tl.<span class="fn">load</span>(w_ptr + cols, mask=mask, other=<span class="num">0.</span>).<span class="fn">to</span>(tl.float32)
    y = (x * inv_rms) * w
    tl.<span class="fn">store</span>(y_ptrs, y, mask=mask)


<span class="kw">def</span> <span class="fn">rmsnorm</span>(x, weight, eps=<span class="num">1e-6</span>):
    <span class="kw">assert</span> x.is_contiguous() <span class="kw">and</span> weight.is_contiguous()
    M, D = x.shape
    y = torch.<span class="fn">empty_like</span>(x)
    rstd = torch.<span class="fn">empty</span>(M, device=x.device, dtype=torch.float32)

    <span class="com"># Pick BLOCK_SIZE as the next power-of-2 ≥ D, so D fits in one block load.</span>
    BLOCK_SIZE = triton.<span class="fn">next_power_of_2</span>(D)

    <span class="com"># Grid: one program per row.</span>
    rmsnorm_fwd_kernel[(M,)](
        x, weight, y, rstd,
        M, D, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    <span class="kw">return</span> y</code></pre>

<p>That's the complete kernel — 25 lines including launcher. Compare to M23's ~140 lines for the same algorithm.</p>

<p>What replaced what:</p>

<ul>
  <li><strong><code>tl.program_id(0)</code></strong> instead of <code>blockIdx.x</code>. The "program" is Triton's name for a kernel instance — equivalent to a CUDA block.</li>
  <li><strong><code>tl.arange(0, BLOCK_SIZE)</code></strong> instead of <code>threadIdx.x</code>. You get a tile of indices, not one per-thread index. The compiler decides how to distribute that tile across warps and threads.</li>
  <li><strong><code>tl.load(ptrs, mask=mask)</code></strong> instead of explicit thread-strided loads. Triton handles coalescing automatically; the mask handles boundary cases (when D isn't a multiple of BLOCK_SIZE).</li>
  <li><strong><code>tl.sum(x * x, axis=0)</code></strong> instead of a hand-written shared memory tree reduction. The compiler picks the right reduction strategy.</li>
  <li><strong>Type promotion</strong> via <code>.to(tl.float32)</code> — same fp32-accumulation pattern as CUDA, but in one expression.</li>
  <li><strong>Implicit synchronization</strong>: there's no <code>__syncthreads()</code> here because Triton's reductions and stores are synchronous within a program by construction.</li>
</ul>

<h3>Calling it from Python</h3>

<p>The launcher is also pure Python:</p>

<pre><code><span class="com"># Standard usage</span>
x = torch.<span class="fn">randn</span>(<span class="num">128</span>, <span class="num">2048</span>, device=<span class="str">'cuda'</span>, dtype=torch.bfloat16)
w = torch.<span class="fn">randn</span>(<span class="num">2048</span>, device=<span class="str">'cuda'</span>, dtype=torch.bfloat16)
y = <span class="fn">rmsnorm</span>(x, w)         <span class="com"># first call: ~1s compile; subsequent: instant</span></code></pre>

<p>No build step, no setup.py, no CUDA Toolkit version pinning. Triton compiles to PTX directly via LLVM, caches the result like Inductor caches its kernels, and reruns from cache on subsequent calls. The first call to a kernel triggers compilation (typically 1-3 seconds for a simple kernel; longer for autotuned ones); subsequent calls hit the cache.</p>

<h2>Autotuning: the killer feature</h2>

<p>Triton kernels often have one or two performance-critical knobs: <code>BLOCK_SIZE</code> in our example, plus <code>num_warps</code> and <code>num_stages</code> (we'll see those in matmul). Different shapes want different choices. Hand-picking is tedious; autotuning finds the best for each shape automatically.</p>

<pre><code><span class="kw">@</span>triton.<span class="fn">autotune</span>(
    configs=[
        triton.<span class="fn">Config</span>({<span class="str">'BLOCK_SIZE'</span>: <span class="num">512</span>},  num_warps=<span class="num">4</span>),
        triton.<span class="fn">Config</span>({<span class="str">'BLOCK_SIZE'</span>: <span class="num">1024</span>}, num_warps=<span class="num">4</span>),
        triton.<span class="fn">Config</span>({<span class="str">'BLOCK_SIZE'</span>: <span class="num">2048</span>}, num_warps=<span class="num">8</span>),
        triton.<span class="fn">Config</span>({<span class="str">'BLOCK_SIZE'</span>: <span class="num">4096</span>}, num_warps=<span class="num">8</span>),
    ],
    key=[<span class="str">'D'</span>],                       <span class="com"># re-tune when D changes</span>
)
<span class="kw">@</span>triton.<span class="fn">jit</span>
<span class="kw">def</span> <span class="fn">rmsnorm_fwd_kernel</span>(
    x_ptr, w_ptr, y_ptr, rstd_ptr,
    M, D, eps,
    BLOCK_SIZE: tl.constexpr,
):
    <span class="com"># ... same kernel body as before ...</span>
    <span class="kw">pass</span></code></pre>

<p>The first call for each unique value of <code>D</code> times all 4 configs and caches the winner. Total upfront cost: roughly 4× the single-config compile time (one per config). Total benefit: the kernel runs at the optimal config for <em>your</em> shape without you knowing what that config is.</p>

<p>The <code>key</code> parameter says "re-tune whenever this argument changes." For RMSNorm, the only relevant shape parameter is <code>D</code> (the row width); <code>M</code> (the row count) just controls how many programs we launch and doesn't affect the per-row tuning. Get this wrong (e.g., add <code>M</code> to <code>key</code>) and you'll re-tune every time the batch size changes — wasted compile time.</p>

<h2>The Triton execution model</h2>

<p>Triton's mental model is the same as CUDA's, but expressed differently. Each piece of CUDA terminology has a Triton equivalent:</p>

<div class="table-wrap">
<table>
<caption>CUDA → Triton terminology</caption>
<thead><tr><th>CUDA</th><th>Triton</th><th>Notes</th></tr></thead>
<tbody>
<tr><td>Kernel launch <code>&lt;&lt;&lt;grid, block&gt;&gt;&gt;</code></td><td><code>kernel[grid_tuple](args, ...)</code></td><td>Grid is just a tuple; block is implicit (configured by num_warps).</td></tr>
<tr><td>blockIdx.x, blockIdx.y</td><td><code>tl.program_id(0)</code>, <code>tl.program_id(1)</code></td><td>The program is the unit; threads are below the abstraction.</td></tr>
<tr><td>threadIdx.x</td><td>(absent — handled by compiler)</td><td>You think tiles, the compiler thinks threads.</td></tr>
<tr><td>__syncthreads()</td><td>(absent in most kernels)</td><td>Reductions are atomic; control flow handles synchronization.</td></tr>
<tr><td>__shared__ memory</td><td><code>tl.dot</code> auto-uses; otherwise compiler decides</td><td>Tiles in registers; large tiles automatically promoted to shared memory.</td></tr>
<tr><td>WMMA / mma.sync (tensor cores)</td><td><code>tl.dot(a, b)</code></td><td>One line. Triton picks the right tensor core instruction.</td></tr>
<tr><td>num_warps (you control)</td><td>num_warps in <code>@triton.jit</code> or <code>Config</code></td><td>Same concept; just a Python kwarg.</td></tr>
</tbody>
</table>
</div>

<p>The big move: <strong>tiles instead of threads</strong>. In CUDA, you wrote a per-thread program; in Triton, you write a per-block program that operates on tiles. The compiler maps your tile operations onto threads. This is why Triton code is so much shorter — most of CUDA's verbosity is per-thread bookkeeping.</p>

<h2>A second example: fused matmul + bias + GELU</h2>

<p>Let's do something more interesting. A common transformer pattern: matmul, add a bias, apply GELU. In eager PyTorch, that's three kernels (matmul, add, gelu) with intermediate writes to HBM between. In Triton, one fused kernel:</p>

<pre><code><span class="kw">@</span>triton.<span class="fn">autotune</span>(
    configs=[
        triton.<span class="fn">Config</span>({<span class="str">'BLOCK_M'</span>: <span class="num">128</span>, <span class="str">'BLOCK_N'</span>: <span class="num">128</span>, <span class="str">'BLOCK_K'</span>: <span class="num">32</span>},
                      num_warps=<span class="num">4</span>, num_stages=<span class="num">3</span>),
        triton.<span class="fn">Config</span>({<span class="str">'BLOCK_M'</span>: <span class="num">128</span>, <span class="str">'BLOCK_N'</span>: <span class="num">256</span>, <span class="str">'BLOCK_K'</span>: <span class="num">32</span>},
                      num_warps=<span class="num">8</span>, num_stages=<span class="num">3</span>),
        triton.<span class="fn">Config</span>({<span class="str">'BLOCK_M'</span>: <span class="num">256</span>, <span class="str">'BLOCK_N'</span>: <span class="num">128</span>, <span class="str">'BLOCK_K'</span>: <span class="num">32</span>},
                      num_warps=<span class="num">8</span>, num_stages=<span class="num">3</span>),
    ],
    key=[<span class="str">'M'</span>, <span class="str">'N'</span>, <span class="str">'K'</span>],
)
<span class="kw">@</span>triton.<span class="fn">jit</span>
<span class="kw">def</span> <span class="fn">matmul_bias_gelu_kernel</span>(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    <span class="com"># Each program handles a (BLOCK_M, BLOCK_N) tile of the output.</span>
    pid_m = tl.<span class="fn">program_id</span>(<span class="num">0</span>)
    pid_n = tl.<span class="fn">program_id</span>(<span class="num">1</span>)

    offs_m = pid_m * BLOCK_M + tl.<span class="fn">arange</span>(<span class="num">0</span>, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.<span class="fn">arange</span>(<span class="num">0</span>, BLOCK_N)
    offs_k = tl.<span class="fn">arange</span>(<span class="num">0</span>, BLOCK_K)

    <span class="com"># Pointer arithmetic for the A and B tiles.</span>
    a_ptrs = a_ptr + (offs_m[:, <span class="kw">None</span>] * stride_am + offs_k[<span class="kw">None</span>, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, <span class="kw">None</span>] * stride_bk + offs_n[<span class="kw">None</span>, :] * stride_bn)

    <span class="com"># Accumulate in fp32.</span>
    acc = tl.<span class="fn">zeros</span>((BLOCK_M, BLOCK_N), dtype=tl.float32)

    <span class="com"># Loop over K in BLOCK_K chunks.</span>
    <span class="kw">for</span> k <span class="kw">in</span> <span class="fn">range</span>(<span class="num">0</span>, K, BLOCK_K):
        a = tl.<span class="fn">load</span>(a_ptrs, mask=offs_k[<span class="kw">None</span>, :] &lt; K - k, other=<span class="num">0.</span>)
        b = tl.<span class="fn">load</span>(b_ptrs, mask=offs_k[:, <span class="kw">None</span>] &lt; K - k, other=<span class="num">0.</span>)
        acc += tl.<span class="fn">dot</span>(a, b)        <span class="com"># tensor cores!</span>
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    <span class="com"># Add bias (broadcasts across rows).</span>
    bias = tl.<span class="fn">load</span>(bias_ptr + offs_n, mask=offs_n &lt; N, other=<span class="num">0.</span>)
    acc += bias[<span class="kw">None</span>, :]

    <span class="com"># Apply GELU (using the tanh approximation: faster than the erf form).</span>
    sqrt_2_pi = <span class="num">0.7978845608</span>
    acc = <span class="num">0.5</span> * acc * (<span class="num">1.0</span> + tl.<span class="fn">tanh</span>(sqrt_2_pi * (acc + <span class="num">0.044715</span> * acc * acc * acc)))

    <span class="com"># Store back, casting to output dtype.</span>
    c_ptrs = c_ptr + (offs_m[:, <span class="kw">None</span>] * stride_cm + offs_n[<span class="kw">None</span>, :] * stride_cn)
    mask = (offs_m[:, <span class="kw">None</span>] &lt; M) &amp; (offs_n[<span class="kw">None</span>, :] &lt; N)
    tl.<span class="fn">store</span>(c_ptrs, acc.<span class="fn">to</span>(tl.bfloat16), mask=mask)


<span class="kw">def</span> <span class="fn">matmul_bias_gelu</span>(a, b, bias):
    M, K = a.shape
    K2, N = b.shape
    <span class="kw">assert</span> K == K2
    c = torch.<span class="fn">empty</span>((M, N), device=a.device, dtype=torch.bfloat16)
    grid = <span class="kw">lambda</span> meta: (triton.<span class="fn">cdiv</span>(M, meta[<span class="str">'BLOCK_M'</span>]),
                         triton.<span class="fn">cdiv</span>(N, meta[<span class="str">'BLOCK_N'</span>]))
    matmul_bias_gelu_kernel[grid](
        a, b, bias, c,
        M, N, K,
        a.<span class="fn">stride</span>(<span class="num">0</span>), a.<span class="fn">stride</span>(<span class="num">1</span>), b.<span class="fn">stride</span>(<span class="num">0</span>), b.<span class="fn">stride</span>(<span class="num">1</span>),
        c.<span class="fn">stride</span>(<span class="num">0</span>), c.<span class="fn">stride</span>(<span class="num">1</span>),
    )
    <span class="kw">return</span> c</code></pre>

<p>40 lines for matmul + bias + GELU fused. Notable points:</p>

<ul>
  <li><strong><code>tl.dot(a, b)</code></strong> dispatches to tensor cores when shapes and dtypes are right. This single call replaces the entire WMMA / mma.sync dance you'd write in CUDA.</li>
  <li><strong>2D program grid</strong> — one program per output tile. <code>tl.program_id(0)</code> for the row tile, <code>tl.program_id(1)</code> for the column tile.</li>
  <li><strong>Strides as kernel arguments</strong> — instead of assuming contiguous layout, we pass strides explicitly so the kernel works with arbitrary (e.g., transposed) inputs.</li>
  <li><strong>Numerical accumulator in fp32</strong> — same recipe as M14 and M23. Inputs in bf16; accumulate in fp32; cast back on store.</li>
  <li><strong>The K-loop</strong> is the matmul accumulation loop. Tiles of K=32 elements are loaded at a time; <code>tl.dot</code> on (BLOCK_M, 32) × (32, BLOCK_N) accumulates into the (BLOCK_M, BLOCK_N) tile.</li>
  <li><strong>num_stages=3</strong> in the autotune configs enables software pipelining: while one K-tile is being computed by tensor cores, the next is being loaded. This is the same prefetching idea as M16's DDP gradient overlap, but for memory loads inside a kernel.</li>
</ul>

<p>This kernel will run within 80-95% of cuBLAS's matmul speed on the matmul portion alone, and <em>save additional time</em> by avoiding the HBM round-trips for bias and GELU. Pure win for the fused pattern.</p>

<h2>The matmul tile pattern, visualized</h2>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 380" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Matmul tiling: each program computes one (BLOCK_M, BLOCK_N) output tile</text>

  <!-- A matrix on the left -->
  <g transform="translate(60, 60)">
    <text x="80" y="-5" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">A: (M, K)</text>
    <rect x="0" y="0" width="160" height="200" fill="#fff8a8" stroke="#1a1612"/>
    <!-- Highlight one row tile -->
    <rect x="0" y="40" width="160" height="40" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="80" y="65" text-anchor="middle" font-size="10" fill="#1a1612">row tile (BLOCK_M, K)</text>
    <text x="80" y="220" text-anchor="middle" font-size="9" fill="#6b5d4f">processed in BLOCK_K chunks →</text>
  </g>

  <!-- B matrix on top -->
  <g transform="translate(310, 60)">
    <text x="120" y="-5" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">B: (K, N)</text>
    <rect x="0" y="-40" width="240" height="40" fill="#fff8a8" stroke="#1a1612"/>
    <!-- Just a label position; rotate B visually below -->
    <rect x="0" y="0" width="240" height="200" fill="#fff8a8" stroke="#1a1612"/>
    <!-- Highlight one column tile -->
    <rect x="80" y="0" width="60" height="200" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="110" y="105" text-anchor="middle" font-size="10" fill="#1a1612" transform="rotate(-90 110 105)">col tile (K, BLOCK_N)</text>
  </g>

  <!-- Result tile -->
  <g transform="translate(310, 290)">
    <text x="120" y="-5" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">C: (M, N) — output tile being computed</text>
    <rect x="80" y="0" width="60" height="40" fill="#ffd5dc" stroke="#b85a6c" stroke-width="3"/>
    <text x="110" y="25" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">acc</text>
    <text x="180" y="25" font-size="10" fill="#1a1612">= row tile @ col tile, accumulated in fp32</text>
  </g>

  <!-- Connector arrows -->
  <text x="240" y="170" font-family="'Caveat', cursive" font-size="20" fill="#1a1612">×</text>
  <text x="370" y="290" font-family="'Caveat', cursive" font-size="18" fill="#c1502e">accumulate over K-tiles → tensor core (tl.dot)</text>
</svg>
</div>

<p>One program per output tile: the program reads its (BLOCK_M, K) row-strip of A and its (K, BLOCK_N) column-strip of B, accumulates the matmul into an fp32 (BLOCK_M, BLOCK_N) tile, then writes the result. The K dimension is iterated in BLOCK_K-sized chunks; each iteration is one <code>tl.dot</code> call that fires the tensor cores. <strong>This pattern — partition outputs into tiles, each program computes one tile — is universal for matmul kernels.</strong> FlashAttention (M25) is a creative variant of this same pattern.</p>

<h2>Practical knobs and gotchas</h2>

<h3>The <code>num_warps</code> dial</h3>

<p>Triton lets you set <code>num_warps</code> (in the autotune Config or as a kwarg to <code>@triton.jit</code>). Each program runs with that many warps available. Common values: 4 (small kernels with little parallelism per program), 8 (medium, like our matmul example), 16 (large tiles, want more parallelism within the program). Larger num_warps means more threads per program but fewer programs that fit per SM (occupancy tradeoff from M22). Autotune with several values.</p>

<h3>Numerical precision: cast at the right boundaries</h3>

<p>The cast pattern from M14 and M23 carries over directly. <code>tl.load(...).to(tl.float32)</code> reads bf16/fp16 and promotes to fp32 for the kernel body. <code>acc.to(tl.bfloat16)</code> at the store. Don't accumulate in low precision — same numerical hazards as before.</p>

<p>One Triton-specific subtlety: <code>tl.dot</code> can take an <code>out_dtype</code> parameter that controls the accumulator dtype. <code>tl.dot(a, b, out_dtype=tl.float32)</code> uses fp32 tensor cores; the default is whatever Triton thinks is appropriate. Specify it explicitly when you care.</p>

<h3>Debugging: <code>TRITON_INTERPRET=1</code></h3>

<p>The killer feature for development. Set this env var and Triton runs your kernel in pure Python (using PyTorch ops) instead of compiling to GPU. You can then use <code>print()</code>, <code>pdb</code>, anything — your kernel runs as ordinary Python code. <em>Slow as molasses</em> for real workloads, but invaluable when debugging.</p>

<pre><code>TRITON_INTERPRET=<span class="num">1</span> python my_kernel_test.py
<span class="com"># prints, breakpoints, asserts all work; runs at Python speed</span></code></pre>

<p>Once the kernel produces correct outputs in interpret mode, switch back to JIT for performance.</p>

<h3>Compile cache and disk persistence</h3>

<p>Triton caches compiled kernels in <code>~/.triton/cache</code> by default. Each (kernel, signature, config) tuple gets a cache entry. Subsequent runs with the same combination skip compilation. To force a rebuild (e.g., after upgrading Triton), <code>rm -rf ~/.triton/cache</code>.</p>

<h3>Combining with <code>torch.library</code></h3>

<p>Same as M23 — wrap the Triton launcher in <code>@torch.library.custom_op</code> for autograd, autocast, and compile integration:</p>

<pre><code><span class="kw">@</span>torch.library.<span class="fn">custom_op</span>(<span class="str">"my_lib::rmsnorm"</span>, mutates_args=())
<span class="kw">def</span> <span class="fn">rmsnorm_op</span>(x: torch.Tensor, w: torch.Tensor, eps: <span class="fn">float</span>) -&gt; torch.Tensor:
    <span class="kw">return</span> <span class="fn">rmsnorm_torch_fallback</span>(x, w, eps)

<span class="kw">@</span>rmsnorm_op.<span class="fn">register_kernel</span>(<span class="str">"cuda"</span>)
<span class="kw">def</span> <span class="fn">_cuda</span>(x, w, eps):
    <span class="kw">return</span> <span class="fn">rmsnorm</span>(x.<span class="fn">contiguous</span>(), w.<span class="fn">contiguous</span>(), eps)   <span class="com"># calls our Triton kernel</span>

<span class="kw">@</span>rmsnorm_op.<span class="fn">register_fake</span>()
<span class="kw">def</span> <span class="fn">_fake</span>(x, w, eps):
    <span class="kw">return</span> torch.<span class="fn">empty_like</span>(x)

<span class="com"># Plus register_autograd as in M23.</span></code></pre>

<p>Now <code>torch.ops.my_lib.rmsnorm(x, w, eps)</code> goes through the dispatcher tower, calls our Triton kernel on CUDA, falls back to PyTorch ops on CPU, traces under <code>torch.compile</code>, and gets gradients via the registered backward.</p>

<h2>Triton vs Inductor: who writes which?</h2>

<p>Recall from M21: <code>torch.compile</code>'s Inductor backend code-generates Triton kernels for the fused regions of your model. So when you write <code>torch.compile(model)</code>, Triton is doing the work under the hood — Inductor is essentially an automatic Triton-kernel writer.</p>

<p>So when do you write Triton yourself vs let Inductor do it?</p>

<div class="table-wrap">
<table>
<caption>When to hand-write Triton vs let Inductor generate it</caption>
<thead><tr><th>Situation</th><th>Best path</th></tr></thead>
<tbody>
<tr><td>Standard transformer layers, eager-mode-equivalent semantics</td><td><code>torch.compile</code>; let Inductor generate.</td></tr>
<tr><td>Custom op with no PyTorch equivalent (custom attention, exotic activation)</td><td>Hand-write Triton; register via <code>torch.library</code>.</td></tr>
<tr><td>Op where Inductor's generated kernel is suboptimal</td><td>Hand-write Triton, register, let compile use yours.</td></tr>
<tr><td>FlashAttention, paged attention, similar memory-trick algorithms</td><td>Hand-written Triton (or the prebuilt libraries that include them).</td></tr>
<tr><td>Need specific tensor core instructions Inductor doesn't emit</td><td>Hand-written Triton with explicit <code>tl.dot</code> shapes.</td></tr>
</tbody>
</table>
</div>

<p>The 90% case is "use compile, don't write kernels." The 10% case is "write a Triton kernel, register it, let compile use it for the parts you didn't write." Inductor and hand-written Triton are complementary, not competing.</p>

<div class="ndq">
<h4>About Triton</h4>

<p class="q">Why does Triton compile so much faster than CUDA?</p>
<p class="a">Triton's compile pipeline is Python → MLIR → LLVM → PTX. CUDA's is C++ → C++ frontend → LLVM → PTX with a lot of templating overhead. The Python+MLIR path is much cleaner — no template instantiation, no header parsing, no PYBIND11. For simple kernels, Triton compile is a few hundred ms; for autotuned matmul kernels, a few seconds. CUDA can take 30+ seconds for the same work.</p>

<p class="q">Does Triton work on AMD GPUs?</p>
<p class="a">Yes — Triton has had ROCm/HIP backend support for a while; the same kernel code typically runs on both NVIDIA and AMD with the right Triton install. Performance varies (some patterns optimized harder for NVIDIA), but the language is portable. Same kernel code → different backends compile differently.</p>

<p class="q">Why use <code>tl.constexpr</code> for BLOCK_SIZE?</p>
<p class="a">Triton specializes the kernel on each unique value of compile-time-constant arguments. <code>BLOCK_SIZE: tl.constexpr</code> tells Triton: "this is a compile-time constant; generate a different kernel for each value." This lets the compiler unroll loops, allocate fixed-size tiles in registers, and pick optimal codegen. Pass it as a regular int and it'd be a runtime variable — slower and less optimized. <strong>All shape parameters that affect tile sizes should be <code>tl.constexpr</code>.</strong></p>

<p class="q">My Triton kernel works in interpret mode but produces wrong results when compiled. What gives?</p>
<p class="a">Most often: a subtle issue with masking, pointer arithmetic, or boundary handling that's masked (no pun) in interpret mode. Interpret mode handles edge cases more permissively than compiled code. Common culprits: forgetting <code>mask=</code> on a load that runs off the end of an array; integer overflow in pointer arithmetic for large tensors (use <code>tl.int64</code> for offsets when tensors are big); using a non-power-of-2 BLOCK_SIZE for a reduction (Triton requires power-of-2 for many ops). Add explicit asserts in interpret mode to catch the boundary issues, then re-test compiled.</p>

<p class="q">Can I call PyTorch functions from inside a Triton kernel?</p>
<p class="a">No. The Triton kernel body is a tightly restricted DSL — it has tensor-tile primitives but no general Python. You can call other <code>@triton.jit</code> functions (Triton supports kernel-to-kernel function calls), but not arbitrary PyTorch ops. PyTorch ops only exist on the launcher side.</p>

<p class="q">When do I need to worry about num_stages?</p>
<p class="a">In matmul-style kernels with a K-loop (like our matmul example). <code>num_stages</code> enables software pipelining: while iteration i is running on tensor cores, iteration i+1's data is being prefetched from HBM. With <code>num_stages=3</code>, you have 3 K-tiles in flight at once. Higher num_stages = better overlap but more shared memory needed. For purely pointwise kernels (like RMSNorm), <code>num_stages</code> doesn't matter and the default is fine.</p>
</div>

<h2>The minimal mental model</h2>

<p>Three things to remember from M24:</p>

<ol>
  <li><strong>Triton is per-block, not per-thread.</strong> Each program instance handles one tile of work. The compiler maps tiles to warps, threads, and registers automatically. You think tiles; the compiler thinks threads.</li>
  <li><strong>Autotuning is free.</strong> Add a decorator with a few configs; Triton finds the best. This is the practical reason Triton kernels often match cuBLAS — the search space is explored, not guessed.</li>
  <li><strong>Triton + <code>torch.library</code> is the standard recipe</strong> for adding a new op to PyTorch. Write the kernel in Triton, wrap with <code>custom_op</code>, register fake and autograd. The op slots into autograd, autocast, compile, distributed — all the way down M19's dispatcher tower.</li>
</ol>

<h2>Code Magnets: build a complete vector-add Triton kernel</h2>

<p>You're writing a fused vector add + scalar multiply: <code>z = (x + y) * alpha</code>. Three magnets are wrong choices.</p>

<div class="magnets">
<p>Arrange the magnets into a working kernel + launcher.</p>

<div class="magnet-pool">
  <span class="magnet">@triton.jit</span>
  <span class="magnet">def add_mul_kernel(x_ptr, y_ptr, z_ptr, alpha, n, BLOCK: tl.constexpr):</span>
  <span class="magnet">    pid = tl.program_id(0)</span>
  <span class="magnet">    pid = threadIdx.x</span>
  <span class="magnet">    offs = pid * BLOCK + tl.arange(0, BLOCK)</span>
  <span class="magnet">    mask = offs < n</span>
  <span class="magnet">    x = tl.load(x_ptr + offs, mask=mask)</span>
  <span class="magnet">    y = tl.load(y_ptr + offs, mask=mask)</span>
  <span class="magnet">    tl.store(z_ptr + offs, (x + y) * alpha, mask=mask)</span>
  <span class="magnet">def add_mul(x, y, alpha):</span>
  <span class="magnet">    z = torch.empty_like(x)</span>
  <span class="magnet">    grid = (triton.cdiv(x.numel(), 1024),)</span>
  <span class="magnet">    add_mul_kernel[grid](x, y, z, alpha, x.numel(), BLOCK=1024)</span>
  <span class="magnet">    add_mul_kernel(x, y, z, alpha, x.numel(), BLOCK=1024)</span>
  <span class="magnet">    return z</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">@</span>triton.<span class="fn">jit</span>
<span class="kw">def</span> <span class="fn">add_mul_kernel</span>(x_ptr, y_ptr, z_ptr, alpha, n, BLOCK: tl.constexpr):
    pid = tl.<span class="fn">program_id</span>(<span class="num">0</span>)
    offs = pid * BLOCK + tl.<span class="fn">arange</span>(<span class="num">0</span>, BLOCK)
    mask = offs &lt; n
    x = tl.<span class="fn">load</span>(x_ptr + offs, mask=mask)
    y = tl.<span class="fn">load</span>(y_ptr + offs, mask=mask)
    tl.<span class="fn">store</span>(z_ptr + offs, (x + y) * alpha, mask=mask)

<span class="kw">def</span> <span class="fn">add_mul</span>(x, y, alpha):
    z = torch.<span class="fn">empty_like</span>(x)
    grid = (triton.<span class="fn">cdiv</span>(x.<span class="fn">numel</span>(), <span class="num">1024</span>),)
    add_mul_kernel[grid](x, y, z, alpha, x.<span class="fn">numel</span>(), BLOCK=<span class="num">1024</span>)
    <span class="kw">return</span> z</code></pre>
<p>The traps:</p>
<ul>
  <li><code>pid = threadIdx.x</code>: that's CUDA. Triton uses <code>tl.program_id(0)</code> — there's no thread-level identifier in Triton's user-facing API.</li>
  <li><code>add_mul_kernel(x, y, z, alpha, x.numel(), BLOCK=1024)</code>: missing the <code>[grid]</code> launch syntax. Triton requires the grid to be specified via <code>kernel[grid](args)</code>; calling it like a regular function fails to launch.</li>
</ul>
<p>Three things to internalize from this puzzle: (1) <code>tl.program_id(N)</code> not threadIdx, (2) the <code>kernel[grid](args)</code> launch syntax is mandatory, (3) the mask handles boundary when <code>n</code> isn't a multiple of BLOCK.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each Triton concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>@triton.jit</div>
  <div>A. Marks a function as a Triton kernel; compiles to PTX on first call.</div>

  <div>tl.program_id(axis)</div>
  <div>B. Returns the index of this kernel instance along the launch grid axis.</div>

  <div>tl.constexpr</div>
  <div>C. Marks a parameter as compile-time constant — the kernel specializes per value.</div>

  <div>tl.load with mask</div>
  <div>D. Reads from HBM with boundary handling — masked-off lanes get the <code>other</code> value.</div>

  <div>tl.dot</div>
  <div>E. Tile matmul that maps to tensor cores — replaces WMMA / mma.sync.</div>

  <div>@triton.autotune</div>
  <div>F. Searches over configs (BLOCK_SIZE, num_warps, num_stages) for the fastest.</div>

  <div>TRITON_INTERPRET=1</div>
  <div>G. Runs the kernel as ordinary Python — slow but enables print/pdb debugging.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>@triton.jit</strong> → A<br>
<strong>tl.program_id(axis)</strong> → B<br>
<strong>tl.constexpr</strong> → C<br>
<strong>tl.load with mask</strong> → D<br>
<strong>tl.dot</strong> → E<br>
<strong>@triton.autotune</strong> → F<br>
<strong>TRITON_INTERPRET=1</strong> → G
</p>
<p>The mental shortcut: <em>jit compiles, program_id indexes the grid, constexpr triggers specialization, masked load handles boundaries, tl.dot fires tensor cores, autotune searches configs, INTERPRET=1 enables Python debugging</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team's Triton matmul kernel is 30% slower than cuBLAS for square matrices but matches cuBLAS for very tall-and-skinny ones. What's likely going on?</p>
<details class="answer"><summary>show answer</summary>
<p>The autotune configs they're searching probably weren't designed for the square shapes. cuBLAS uses a sophisticated heuristic for picking tile sizes and split-K strategies that Triton's autotune doesn't replicate by default. Two diagnoses: (a) check whether their autotune config list includes the larger tile sizes (BLOCK_M=128, BLOCK_N=256, BLOCK_K=64+) that cuBLAS uses for square shapes — if not, add them. (b) For very large square matmuls, cuBLAS uses split-K (multiple programs cooperate on the same output tile, summing their partial K-results), which Triton can do but most simple matmul examples don't. Adding split-K configs to autotune is what closes the last gap. <em>Tall-skinny shapes don't benefit from split-K (small K already), so the basic tile pattern wins there — explaining the asymmetry.</em></p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Why is <code>BLOCK_SIZE: tl.constexpr</code> required for performance in our RMSNorm kernel?</p>
<details class="answer"><summary>show answer</summary>
<p>Without <code>tl.constexpr</code>, BLOCK_SIZE would be a runtime variable and the compiler couldn't (a) statically size the tile in registers, (b) unroll the implicit loops over BLOCK_SIZE elements, or (c) pick the right reduction strategy. With <code>constexpr</code>, the compiler generates one specialized kernel per BLOCK_SIZE value and inlines all the size-dependent decisions. The cost: more compile time (one kernel per unique value); the benefit: each kernel runs at peak efficiency. <strong>For shape parameters: always constexpr. For data values (eps, alpha): regular runtime args.</strong></p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> A team has a working CUDA extension (M23) and wants to migrate to Triton. Sketch the migration plan.</p>
<details class="answer"><summary>show answer</summary>
<ol>
  <li><strong>Identify the kernel boundary.</strong> What was the <code>__global__</code> function in CUDA becomes the <code>@triton.jit</code> function in Triton.</li>
  <li><strong>Translate the indexing.</strong> blockIdx.x → tl.program_id(0); replace per-thread loops with tile-shaped tl.arange + tl.load.</li>
  <li><strong>Translate the reductions.</strong> Hand-coded shared-memory tree reductions become tl.sum / tl.max etc. Same with tl.softmax for attention-style code.</li>
  <li><strong>Translate matmul.</strong> WMMA / mma.sync becomes tl.dot. Often dramatically simplifies code.</li>
  <li><strong>Add autotune configs.</strong> Wrap with @triton.autotune over the dimensional knobs (BLOCK_SIZE, num_warps, num_stages).</li>
  <li><strong>Re-register through torch.library.</strong> The custom_op + register_kernel + register_fake + register_autograd dance from M19 / M23 is unchanged; only the kernel implementation changes.</li>
  <li><strong>Validate with gradcheck (fp64) and a numerical comparison (fp16/bf16) against the original CUDA kernel.</strong> Triton kernels can have slightly different reduction orders, so expect ~1e-3 relative differences in low precision; that's fine.</li>
</ol>
<p>Total migration effort: typically a few hours for a non-trivial kernel, ending with shorter, autotuned, easier-to-debug code.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> Why does Triton's tile-based programming model fit GPUs naturally, despite GPUs being thread-based at the hardware level?</p>
<details class="answer"><summary>show answer</summary>
<p>GPUs are thread-based at the <em>execution</em> level (warps of 32 threads in lockstep) but tile-based at the <em>useful work</em> level — most GPU algorithms operate on contiguous chunks of memory (vector slices, matrix tiles, attention tiles). The CUDA model exposes the per-thread reality and forces you to manually decompose your tile operations into thread-level work. Triton's insight: <strong>this decomposition is mechanical and the compiler can do it.</strong> A tile load decomposes into per-warp vectorized loads with coalescing. A tile reduction decomposes into a tree of shfl operations within a warp + shared memory across warps. The compiler picks the right decomposition based on the tile shape and target hardware. You think at the level your algorithm naturally lives at; the compiler bridges to the level the hardware needs. Same idea as <code>torch.compile</code> for whole models, just at the kernel level.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li><strong>Triton</strong> is a Python DSL for GPU kernels. You write per-tile code, the compiler maps it to per-thread CUDA. Result: ~3-5× shorter code at near-cuBLAS performance.</li>
  <li><strong>The basic kernel pattern</strong>: <code>@triton.jit</code> on a function, <code>tl.program_id</code> for grid indices, <code>tl.arange + tl.load(mask=...)</code> for tile loads, <code>tl.sum/max/dot</code> for tile ops, <code>tl.store(mask=...)</code> for tile stores.</li>
  <li><strong>Tile sizes are <code>tl.constexpr</code></strong> — compile-time constants the compiler specializes on. Always mark BLOCK_SIZE, num_warps configs, etc. as constexpr.</li>
  <li><strong>Boundary handling</strong> via <code>mask=</code> argument to load/store. The mask handles the case when the dimension isn't a multiple of BLOCK_SIZE.</li>
  <li><strong>fp32 accumulation pattern</strong> carries over from M14/M23: <code>tl.load(...).to(tl.float32)</code> on read, <code>acc.to(input_dtype)</code> on write. Don't accumulate in low precision.</li>
  <li><strong><code>tl.dot</code></strong> dispatches to tensor cores. One line replaces the WMMA/mma.sync dance.</li>
  <li><strong><code>@triton.autotune</code></strong> with a list of Configs searches for the fastest BLOCK_SIZE / num_warps / num_stages for each shape. Free performance.</li>
  <li><strong>num_stages</strong> enables software pipelining in K-loops (matmul). Higher = better overlap, more shared memory.</li>
  <li><strong>Launch syntax</strong>: <code>kernel[grid](args)</code>. Grid is a tuple of program counts per axis. <code>triton.cdiv(N, BLOCK)</code> for ceil-div sizing.</li>
  <li><strong>Debugging</strong>: <code>TRITON_INTERPRET=1</code> runs the kernel as Python — slow but supports print/pdb.</li>
  <li><strong>Integration</strong>: same <code>torch.library.custom_op</code> + <code>register_kernel("cuda")</code> + <code>register_fake</code> + <code>register_autograd</code> pattern as M23 — the kernel just changes from CUDA to Triton.</li>
  <li><strong>Triton vs Inductor</strong>: Inductor generates Triton automatically for fused regions. Hand-write Triton when you need an op Inductor doesn't produce well or that doesn't exist in PyTorch (FlashAttention, paged attention, custom layers).</li>
  <li>The reflex: when reaching for a custom kernel, <em>Triton first</em>. Drop to raw CUDA only for things Triton can't express (CUTLASS-level template gymnastics, async copies, warp specialization).</li>
</ul>
</div>

<p>Module 25 takes everything from Part VIII so far — the GPU model (M22), the kernel-writing toolkit (M23 CUDA, M24 Triton), the dispatcher integration (M19) — and walks through one of the most consequential kernels of the past five years: <strong>FlashAttention</strong>. We'll see how the algorithm transforms attention from memory-bound to compute-bound by keeping the attention matrix in shared memory and never materializing it in HBM, doing the recompute trick from M12 inside a single kernel. The kernel is in Triton; the techniques are universal.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">24</span>
  <span>Triton: Python kernels at CUDA speed</span>
</div>
"""

emit("24_triton", "Module 24 — Triton: Python kernels at CUDA speed", BODY)
