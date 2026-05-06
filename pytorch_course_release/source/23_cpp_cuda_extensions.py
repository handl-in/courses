#!/usr/bin/env python3
"""Module 23: C++/CUDA extensions — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part VIII · Module 23</div>
  <h1 class="module-title">C++/CUDA <em>extensions</em></h1>
  <p class="module-sub">— writing your first real CUDA kernel, building it via <code>cpp_extension</code>, and wiring it into PyTorch through <code>torch.library</code> so autograd, autocast, and compile see it as a first-class op</p>
</div>

<p>M22 gave you the hardware model. Now we use it. This module walks through writing a real CUDA kernel — RMSNorm, the normalization layer used in Llama, Mistral, Gemma, and most modern transformers. It's small enough to display in full, useful enough to be motivating, and a good warm-up for the FlashAttention case study in M25.</p>

<p>Three things to internalize before the code starts. (1) A PyTorch CUDA extension is just <em>a CUDA kernel + a C++ wrapper + Python glue</em>. There's no magic; the patterns repeat. (2) The hard part isn't the CUDA — it's the integration with autograd (so backward works), autocast (so mixed precision works), and torch.compile (so it gets fused). The <code>torch.library.custom_op</code> API from M19 handles all three. (3) For most kernels you'll write, <em>Triton</em> (M24) is a better tool than raw CUDA. We'll do this in CUDA first because it teaches you what's underneath; then M24 shows the same kind of work in 1/3 the lines.</p>

<div class="keyidea">
A CUDA extension has three files: a <code>.cu</code> file with the kernel, a <code>.cpp</code> file with C++/Python bindings, and a Python <code>.py</code> file that loads and wraps the extension. PyTorch's <code>torch.utils.cpp_extension.load</code> JIT-compiles them on first use; <code>setuptools</code> with <code>CUDAExtension</code> does ahead-of-time builds for distribution. Once compiled, register the resulting C function via <code>torch.library.custom_op</code> for full autograd/autocast/compile integration. The mental model: <strong>raw CUDA gives you total control; the Python and library layers exist to make your kernel a first-class PyTorch op.</strong>
</div>

<h2>The example: RMSNorm</h2>

<p>RMSNorm is a simpler cousin of LayerNorm. For an input vector <code>x ∈ ℝᴰ</code>:</p>

<pre><code>RMSNorm(x) = (x / RMS(x)) * weight    <span class="com"># where RMS(x) = sqrt(mean(x²) + ε)</span></code></pre>

<p>One reduction (the mean of squares), one rescale, one elementwise multiply by a learnable weight. Memory-bound (M22) — almost no arithmetic per byte. The job of a custom kernel is to do the reduction and the rescale in one fused pass over HBM, instead of multiple passes.</p>

<p>Per-row, the algorithm is:</p>

<ol>
  <li>Load all D elements of the row into registers.</li>
  <li>Square them, sum across threads in the block, reduce to a single scalar.</li>
  <li>Compute <code>1/sqrt(mean + ε)</code>.</li>
  <li>Multiply each element by that scalar and the weight, write out.</li>
</ol>

<p>The key engineering decision: <strong>one block per row</strong>. A row fits in shared memory + registers; each block handles one row's worth of work; no inter-block communication. Standard pattern for normalizations.</p>

<h2>The file layout</h2>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 300" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrEX" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Building a CUDA extension: three files, two compilers</text>

  <!-- Source files row -->
  <g transform="translate(20, 50)">
    <rect x="0" y="0" width="160" height="60" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="80" y="20" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">rmsnorm.cu</text>
    <text x="80" y="36" text-anchor="middle" font-size="9" fill="#1a1612">CUDA kernel</text>
    <text x="80" y="48" text-anchor="middle" font-size="9" fill="#1a1612">__global__ void ...</text>
  </g>

  <g transform="translate(200, 50)">
    <rect x="0" y="0" width="160" height="60" fill="#fff8a8" stroke="#1a1612" stroke-width="2"/>
    <text x="80" y="20" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">rmsnorm.cpp</text>
    <text x="80" y="36" text-anchor="middle" font-size="9" fill="#1a1612">C++ wrapper</text>
    <text x="80" y="48" text-anchor="middle" font-size="9" fill="#1a1612">PYBIND11_MODULE(...)</text>
  </g>

  <g transform="translate(380, 50)">
    <rect x="0" y="0" width="160" height="60" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="80" y="20" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">rmsnorm.py</text>
    <text x="80" y="36" text-anchor="middle" font-size="9" fill="#1a1612">Python wrapper</text>
    <text x="80" y="48" text-anchor="middle" font-size="9" fill="#1a1612">cpp_extension.load(...)</text>
  </g>

  <!-- Down arrow to compilers -->
  <path d="M 100 115 L 100 145" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrEX)"/>
  <path d="M 280 115 L 280 145" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrEX)"/>

  <!-- Compilers -->
  <g transform="translate(40, 150)">
    <rect x="0" y="0" width="120" height="40" fill="#ede2cc" stroke="#6b5d4f" stroke-width="2"/>
    <text x="60" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">nvcc</text>
    <text x="60" y="32" text-anchor="middle" font-size="9" fill="#1a1612">.cu → .o (PTX/SASS)</text>
  </g>

  <g transform="translate(220, 150)">
    <rect x="0" y="0" width="120" height="40" fill="#ede2cc" stroke="#6b5d4f" stroke-width="2"/>
    <text x="60" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">g++</text>
    <text x="60" y="32" text-anchor="middle" font-size="9" fill="#1a1612">.cpp → .o</text>
  </g>

  <!-- Down arrow to linker -->
  <path d="M 200 195 L 280 220" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrEX)"/>
  <path d="M 100 195 L 200 220" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrEX)"/>

  <!-- Linker → shared library -->
  <g transform="translate(170, 225)">
    <rect x="0" y="0" width="220" height="40" fill="#fff5d8" stroke="#d4a017" stroke-width="2"/>
    <text x="110" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">linker</text>
    <text x="110" y="32" text-anchor="middle" font-size="9" fill="#1a1612">→ rmsnorm.so (loadable .so)</text>
  </g>

  <!-- Right side: high-level path -->
  <g transform="translate(450, 150)">
    <text x="0" y="0" font-size="11" font-weight="700" fill="#1a1612">Python loads → uses kernel</text>
    <rect x="0" y="10" width="270" height="100" fill="#d3e9f5" stroke="#133e3b" stroke-width="2"/>
    <text x="10" y="30" font-size="10" fill="#1a1612">  ext = cpp_extension.load(</text>
    <text x="10" y="44" font-size="10" fill="#1a1612">    name="rmsnorm",</text>
    <text x="10" y="58" font-size="10" fill="#1a1612">    sources=["rmsnorm.cu",</text>
    <text x="10" y="72" font-size="10" fill="#1a1612">             "rmsnorm.cpp"]</text>
    <text x="10" y="86" font-size="10" fill="#1a1612">  )</text>
    <text x="10" y="100" font-size="10" fill="#1a1612">  y = ext.rmsnorm_fwd(x, w)</text>
  </g>

  <text x="370" y="290" font-family="'Caveat', cursive" font-size="20" fill="#c1502e" text-anchor="middle">first call: ~30s compile. subsequent calls: cached, instant.</text>
</svg>
</div>

<p>Three files, two compilers. <code>nvcc</code> compiles the <code>.cu</code> file into a CUDA-aware object containing PTX/SASS for the GPU side and host code for the launch wrappers. <code>g++</code> compiles the <code>.cpp</code> binding (using PYBIND11). The linker glues both with the PyTorch C++ libraries (<code>libtorch</code>) into a Python-loadable <code>.so</code>.</p>

<p>You don't run nvcc and g++ by hand. <code>cpp_extension.load(...)</code> handles it — JIT-compiles on first call, caches the result, and gives you a Python module containing your kernel as a function.</p>

<h2>The CUDA kernel</h2>

<p>Here's the real kernel. Read it slowly; the comments map each piece to M22's concepts.</p>

<pre><code><span class="com">// rmsnorm.cu</span>
<span class="kw">#include</span> &lt;cuda_runtime.h&gt;
<span class="kw">#include</span> &lt;torch/extension.h&gt;

<span class="com">// One block per row of the input.</span>
<span class="com">// Within a block, we use blockDim.x threads to cooperatively reduce.</span>

<span class="kw">template</span>&lt;<span class="kw">typename</span> scalar_t, <span class="kw">int</span> BLOCK_SIZE&gt;
__global__ <span class="kw">void</span> <span class="fn">rmsnorm_fwd_kernel</span>(
    <span class="kw">const</span> scalar_t* __restrict__ x,        <span class="com">// (M, D) input</span>
    <span class="kw">const</span> scalar_t* __restrict__ weight,   <span class="com">// (D,)  scaling weights</span>
    scalar_t* __restrict__ y,              <span class="com">// (M, D) output</span>
    <span class="kw">float</span>* __restrict__ rstd,              <span class="com">// (M,)   1/RMS, saved for backward</span>
    <span class="kw">int</span> D,
    <span class="kw">float</span> eps
) {
    <span class="com">// One block handles one row.</span>
    <span class="kw">int</span> row = blockIdx.x;
    <span class="kw">int</span> tid = threadIdx.x;

    <span class="com">// Pointers to this row.</span>
    <span class="kw">const</span> scalar_t* row_x = x + row * D;
    scalar_t*       row_y = y + row * D;

    <span class="com">// 1. Each thread sums the squares of its strided slice.</span>
    <span class="kw">float</span> local_sum = 0.0f;
    <span class="kw">for</span> (<span class="kw">int</span> i = tid; i &lt; D; i += BLOCK_SIZE) {
        <span class="kw">float</span> v = (<span class="kw">float</span>)row_x[i];
        local_sum += v * v;
    }

    <span class="com">// 2. Block-wide reduction via shared memory.</span>
    <span class="kw">__shared__</span> <span class="kw">float</span> shared_sum[BLOCK_SIZE];
    shared_sum[tid] = local_sum;
    __syncthreads();

    <span class="com">// Tree reduction: O(log BLOCK_SIZE) steps.</span>
    <span class="kw">for</span> (<span class="kw">int</span> stride = BLOCK_SIZE / 2; stride &gt; 0; stride /= 2) {
        <span class="kw">if</span> (tid &lt; stride) shared_sum[tid] += shared_sum[tid + stride];
        __syncthreads();
    }

    <span class="com">// 3. Compute the rescale factor on thread 0; broadcast via shared memory.</span>
    <span class="kw">__shared__</span> <span class="kw">float</span> inv_rms;
    <span class="kw">if</span> (tid == 0) {
        <span class="kw">float</span> mean_sq = shared_sum[0] / (<span class="kw">float</span>)D;
        inv_rms = <span class="fn">rsqrtf</span>(mean_sq + eps);
        rstd[row] = inv_rms;            <span class="com">// save for backward</span>
    }
    __syncthreads();

    <span class="com">// 4. Apply rescale + weight; write out. Coalesced.</span>
    <span class="kw">for</span> (<span class="kw">int</span> i = tid; i &lt; D; i += BLOCK_SIZE) {
        <span class="kw">float</span> v = (<span class="kw">float</span>)row_x[i] * inv_rms * (<span class="kw">float</span>)weight[i];
        row_y[i] = (scalar_t)v;
    }
}

<span class="com">// Host-side launcher</span>
torch::Tensor <span class="fn">rmsnorm_fwd_cuda</span>(
    torch::Tensor x, torch::Tensor weight, <span class="kw">float</span> eps,
    torch::Tensor rstd
) {
    <span class="kw">auto</span> y = torch::<span class="fn">empty_like</span>(x);
    <span class="kw">int</span> M = x.<span class="fn">size</span>(0), D = x.<span class="fn">size</span>(1);
    <span class="kw">constexpr</span> <span class="kw">int</span> BLOCK = <span class="num">256</span>;

    <span class="com">// Dispatch on dtype using AT_DISPATCH_FLOATING_TYPES_AND2.</span>
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.<span class="fn">scalar_type</span>(), <span class="str">"rmsnorm_fwd"</span>, [&amp;]() {
            rmsnorm_fwd_kernel&lt;scalar_t, BLOCK&gt;&lt;&lt;&lt;M, BLOCK&gt;&gt;&gt;(
                x.<span class="fn">data_ptr</span>&lt;scalar_t&gt;(),
                weight.<span class="fn">data_ptr</span>&lt;scalar_t&gt;(),
                y.<span class="fn">data_ptr</span>&lt;scalar_t&gt;(),
                rstd.<span class="fn">data_ptr</span>&lt;<span class="kw">float</span>&gt;(),
                D, eps
            );
        });

    <span class="kw">return</span> y;
}</code></pre>

<p>Three things to call out from the kernel:</p>

<ol>
  <li><strong>One block per row, BLOCK_SIZE threads per block</strong>. With 256 threads = 8 warps per block, each block lives entirely on one SM and does its row's work end-to-end. No inter-block communication. <em>This pattern (one block per "outer" dimension) is ubiquitous.</em></li>
  <li><strong>Shared memory for the reduction</strong>. The squares are summed locally, then a tree reduction in shared memory aggregates across threads. After the final <code>__syncthreads()</code>, every thread can read the same result.</li>
  <li><strong>fp32 accumulation, scalar_t I/O</strong>. We accept bf16 or fp16 inputs (<code>scalar_t</code>) but accumulate sums in fp32 (<code>float local_sum</code>). This is the standard recipe for stable reductions in low precision (M14). The same pattern shows up in cuBLAS and FlashAttention.</li>
</ol>

<h3>Thread/block layout, visualized</h3>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 280" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">RMSNorm kernel: one block per row, threads stride across columns</text>

  <!-- Input matrix layout -->
  <text x="20" y="55" font-size="11" font-weight="700" fill="#1a1612">Input x: shape (M=4, D=2048) — 4 rows, 2048 cols each</text>
  <g transform="translate(20, 65)">
    <!-- Row 0 -->
    <rect x="0" y="0" width="700" height="35" fill="#fff8a8" stroke="#1a1612" stroke-width="1.5"/>
    <text x="-5" y="22" text-anchor="end" font-size="10" fill="#6b5d4f">row 0:</text>
    <text x="350" y="22" text-anchor="middle" font-size="11" fill="#1a1612">→ Block 0 handles this entire row (256 threads cooperate)</text>
    <!-- Row 1 -->
    <rect x="0" y="40" width="700" height="35" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="-5" y="62" text-anchor="end" font-size="10" fill="#6b5d4f">row 1:</text>
    <text x="350" y="62" text-anchor="middle" font-size="11" fill="#1a1612">→ Block 1 handles this entire row</text>
    <!-- Row 2 -->
    <rect x="0" y="80" width="700" height="35" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="-5" y="102" text-anchor="end" font-size="10" fill="#6b5d4f">row 2:</text>
    <text x="350" y="102" text-anchor="middle" font-size="11" fill="#1a1612">→ Block 2</text>
    <!-- Row 3 -->
    <rect x="0" y="120" width="700" height="35" fill="#ffd5dc" stroke="#b85a6c" stroke-width="1.5"/>
    <text x="-5" y="142" text-anchor="end" font-size="10" fill="#6b5d4f">row 3:</text>
    <text x="350" y="142" text-anchor="middle" font-size="11" fill="#1a1612">→ Block 3</text>
  </g>

  <!-- Zoom into row 0 thread layout -->
  <text x="20" y="195" font-size="11" font-weight="700" fill="#c1502e">Inside Block 0 (256 threads on D=2048): each thread strides by 256</text>
  <g transform="translate(20, 205)">
    <rect x="0" y="0" width="700" height="35" fill="#fff8a8" stroke="#1a1612" stroke-width="1"/>
    <!-- Show positions -->
    <line x1="0"   y1="0" x2="0"   y2="35" stroke="#c1502e" stroke-width="1"/>
    <line x1="87"  y1="0" x2="87"  y2="35" stroke="#c1502e" stroke-width="1"/>
    <line x1="175" y1="0" x2="175" y2="35" stroke="#c1502e" stroke-width="1"/>
    <line x1="263" y1="0" x2="263" y2="35" stroke="#c1502e" stroke-width="1"/>
    <line x1="350" y1="0" x2="350" y2="35" stroke="#c1502e" stroke-width="1"/>
    <line x1="438" y1="0" x2="438" y2="35" stroke="#c1502e" stroke-width="1"/>
    <line x1="525" y1="0" x2="525" y2="35" stroke="#c1502e" stroke-width="1"/>
    <line x1="613" y1="0" x2="613" y2="35" stroke="#c1502e" stroke-width="1"/>
    <line x1="700" y1="0" x2="700" y2="35" stroke="#c1502e" stroke-width="1"/>
    <text x="44"  y="20" text-anchor="middle" font-size="9" fill="#1a1612">[0..255]</text>
    <text x="131" y="20" text-anchor="middle" font-size="9" fill="#1a1612">[256..511]</text>
    <text x="219" y="20" text-anchor="middle" font-size="9" fill="#1a1612">[512..767]</text>
    <text x="306" y="20" text-anchor="middle" font-size="9" fill="#1a1612">[768..1023]</text>
    <text x="394" y="20" text-anchor="middle" font-size="9" fill="#1a1612">[1024..1279]</text>
    <text x="482" y="20" text-anchor="middle" font-size="9" fill="#1a1612">[1280..1535]</text>
    <text x="569" y="20" text-anchor="middle" font-size="9" fill="#1a1612">[1536..1791]</text>
    <text x="657" y="20" text-anchor="middle" font-size="9" fill="#1a1612">[1792..2047]</text>
  </g>
  <text x="370" y="265" font-family="'Caveat', cursive" font-size="18" fill="#c1502e" text-anchor="middle">thread 0 reads cols 0, 256, 512, 768, …  thread 1 reads 1, 257, 513, …  → coalesced loads</text>
</svg>
</div>

<p>The thread-layout pattern is worth memorizing: <strong>thread <code>tid</code> processes element <code>tid + k * BLOCK_SIZE</code> for k = 0, 1, 2, ...</strong>. Why? <em>Coalesced memory access</em>. The 32 threads of a warp end up reading 32 consecutive columns at the same time, which the memory controller can serve as a single transaction. The naive alternative — thread 0 reads cols 0..7, thread 1 reads 8..15 — would be uncoalesced and ~10× slower.</p>

<h2>The C++ binding</h2>

<p>The <code>.cpp</code> file is a thin wrapper. It declares the host-side launcher and exposes it to Python via PYBIND11.</p>

<pre><code><span class="com">// rmsnorm.cpp</span>
<span class="kw">#include</span> &lt;torch/extension.h&gt;

<span class="com">// Forward declaration of the function defined in rmsnorm.cu</span>
torch::Tensor <span class="fn">rmsnorm_fwd_cuda</span>(
    torch::Tensor x, torch::Tensor weight, <span class="kw">float</span> eps,
    torch::Tensor rstd);

<span class="com">// User-facing forward: validates inputs, allocates rstd buffer, dispatches.</span>
std::tuple&lt;torch::Tensor, torch::Tensor&gt; <span class="fn">rmsnorm_fwd</span>(
    torch::Tensor x, torch::Tensor weight, <span class="kw">float</span> eps
) {
    TORCH_CHECK(x.<span class="fn">is_cuda</span>(), <span class="str">"x must be CUDA"</span>);
    TORCH_CHECK(weight.<span class="fn">is_cuda</span>(), <span class="str">"weight must be CUDA"</span>);
    TORCH_CHECK(x.<span class="fn">dim</span>() == <span class="num">2</span>, <span class="str">"x must be 2D"</span>);
    TORCH_CHECK(x.<span class="fn">is_contiguous</span>(), <span class="str">"x must be contiguous"</span>);

    <span class="kw">auto</span> rstd = torch::<span class="fn">empty</span>({x.<span class="fn">size</span>(<span class="num">0</span>)},
                              x.<span class="fn">options</span>().<span class="fn">dtype</span>(torch::kFloat32));
    <span class="kw">auto</span> y = <span class="fn">rmsnorm_fwd_cuda</span>(x, weight, eps, rstd);
    <span class="kw">return</span> {y, rstd};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.<span class="fn">def</span>(<span class="str">"rmsnorm_fwd"</span>, &amp;rmsnorm_fwd, <span class="str">"RMSNorm forward (CUDA)"</span>);
}</code></pre>

<p>Three patterns to internalize:</p>

<ul>
  <li><code>TORCH_CHECK</code> for input validation. Failed checks raise Python <code>RuntimeError</code> with the message you provided. Always check device, contiguity, dtype, dimensions — kernels assume these and silent assumption violations are a debugging nightmare.</li>
  <li><code>x.options()</code> propagates device/dtype properties. <code>x.options().dtype(torch::kFloat32)</code> says "same device as x but fp32" — useful when allocating auxiliary buffers like <code>rstd</code>.</li>
  <li><code>PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)</code> is the standard incantation. <code>TORCH_EXTENSION_NAME</code> is a macro that gets the module name from the build system. <code>m.def("rmsnorm_fwd", &rmsnorm_fwd, ...)</code> exposes the C++ function as a Python attribute.</li>
</ul>

<h2>Loading and using from Python</h2>

<p>Now the Python side. <code>cpp_extension.load</code> JIT-builds and loads the extension:</p>

<pre><code><span class="com"># rmsnorm.py</span>
<span class="kw">from</span> torch.utils.cpp_extension <span class="kw">import</span> load

_ext = <span class="fn">load</span>(
    name=<span class="str">"rmsnorm_ext"</span>,
    sources=[<span class="str">"rmsnorm.cu"</span>, <span class="str">"rmsnorm.cpp"</span>],
    extra_cuda_cflags=[<span class="str">"-O3"</span>],
    verbose=<span class="kw">True</span>,         <span class="com"># first-time build prints nvcc/g++ output</span>
)

<span class="kw">def</span> <span class="fn">rmsnorm</span>(x, weight, eps=<span class="num">1e-6</span>):
    <span class="kw">assert</span> x.is_contiguous(), <span class="str">"x must be contiguous"</span>
    y, rstd = _ext.<span class="fn">rmsnorm_fwd</span>(x, weight, eps)
    <span class="kw">return</span> y</code></pre>

<p>First call to <code>load</code> compiles (~30 seconds for a small kernel; longer for bigger ones). Result is cached in <code>~/.cache/torch_extensions</code>; subsequent imports are instant. If you change the source, the cache is invalidated automatically by file hash.</p>

<p>For production / packaging, you'd use setuptools instead:</p>

<pre><code><span class="com"># setup.py</span>
<span class="kw">from</span> setuptools <span class="kw">import</span> setup
<span class="kw">from</span> torch.utils.cpp_extension <span class="kw">import</span> CUDAExtension, BuildExtension

<span class="fn">setup</span>(
    name=<span class="str">"rmsnorm_ext"</span>,
    ext_modules=[<span class="fn">CUDAExtension</span>(
        <span class="str">"rmsnorm_ext"</span>,
        sources=[<span class="str">"rmsnorm.cu"</span>, <span class="str">"rmsnorm.cpp"</span>],
        extra_compile_args={<span class="str">"cxx"</span>: [<span class="str">"-O3"</span>], <span class="str">"nvcc"</span>: [<span class="str">"-O3"</span>]},
    )],
    cmdclass={<span class="str">"build_ext"</span>: BuildExtension},
)</code></pre>

<p>Run <code>pip install .</code> to build and install. Now <code>import rmsnorm_ext</code> works without JIT cost. JIT for development; setuptools for shipping.</p>

<h2>Wiring through <code>torch.library</code></h2>

<p>The function above runs, but it's not a first-class PyTorch op. Autograd doesn't know about it; autocast won't auto-cast inputs; <code>torch.compile</code> can't trace through it cleanly. Time to register it properly. From M19:</p>

<pre><code><span class="kw">import</span> torch
<span class="kw">from</span> rmsnorm <span class="kw">import</span> _ext

<span class="kw">@</span>torch.library.<span class="fn">custom_op</span>(<span class="str">"my_lib::rmsnorm"</span>, mutates_args=())
<span class="kw">def</span> <span class="fn">rmsnorm</span>(x: torch.Tensor, weight: torch.Tensor, eps: <span class="fn">float</span>) -&gt; torch.Tensor:
    <span class="com"># Pure-Python fallback for non-CUDA tensors</span>
    rms = (x * x).<span class="fn">mean</span>(-<span class="num">1</span>, keepdim=<span class="kw">True</span>).<span class="fn">add</span>(eps).<span class="fn">rsqrt</span>()
    <span class="kw">return</span> x * rms * weight

<span class="kw">@</span>rmsnorm.<span class="fn">register_kernel</span>(<span class="str">"cuda"</span>)
<span class="kw">def</span> <span class="fn">_rmsnorm_cuda</span>(x, weight, eps):
    y, _rstd = _ext.<span class="fn">rmsnorm_fwd</span>(x.<span class="fn">contiguous</span>(), weight, eps)
    <span class="kw">return</span> y

<span class="kw">@</span>rmsnorm.<span class="fn">register_fake</span>()
<span class="kw">def</span> <span class="fn">_rmsnorm_fake</span>(x, weight, eps):
    <span class="kw">return</span> torch.<span class="fn">empty_like</span>(x)</code></pre>

<p>Now <code>torch.ops.my_lib.rmsnorm(x, w, eps)</code> goes through the dispatcher tower (M19). On CUDA it hits the kernel; on CPU it falls back to the Python implementation. <code>torch.compile</code> can trace via the fake function. <strong>Same dispatch machinery as built-in ops.</strong></p>

<h2>The backward — and the clever shortcut</h2>

<p>Forward alone isn't enough for training. We need to register an autograd backward. The math: given <code>y = (x / rms) * weight</code>, with <code>rms = sqrt(mean(x²) + ε)</code>:</p>

<pre><code>∂L/∂weight = sum_over_M(∂L/∂y * (x / rms))
∂L/∂x = (1/rms) * (∂L/∂y * weight - x * (1/D) * sum(∂L/∂y * weight * x * (1/rms²)))</code></pre>

<p>Yes, that's the actual derivative. RMSNorm has a non-trivial backward because the rescale factor depends on every element of the input.</p>

<p>The clever shortcut: we already saved <code>rstd = 1/rms</code> per row in the forward. The backward kernel reads <code>rstd</code> instead of recomputing it — saves one full pass over the input. This is why production normalization kernels return <em>both</em> the output and any auxiliary tensors needed by backward.</p>

<p>Skipping the full backward kernel for space (it's another <code>rmsnorm_bwd_kernel</code> with a similar shape — one block per row, fp32 accumulation, etc.). The autograd registration:</p>

<pre><code><span class="kw">def</span> <span class="fn">_rmsnorm_setup_context</span>(ctx, inputs, output):
    x, weight, eps = inputs
    <span class="com"># We need to recompute rstd here for the public API; in real code,</span>
    <span class="com"># return rstd from the kernel and pass it through.</span>
    rstd = (x * x).<span class="fn">mean</span>(-<span class="num">1</span>, keepdim=<span class="kw">True</span>).<span class="fn">add</span>(eps).<span class="fn">rsqrt</span>()
    ctx.<span class="fn">save_for_backward</span>(x, weight, rstd)

<span class="kw">def</span> <span class="fn">_rmsnorm_backward</span>(ctx, grad_y):
    x, weight, rstd = ctx.saved_tensors
    grad_x, grad_weight = _ext.<span class="fn">rmsnorm_bwd</span>(grad_y, x, weight, rstd)
    <span class="kw">return</span> grad_x, grad_weight, <span class="kw">None</span>     <span class="com"># None for eps (non-tensor)</span>

torch.library.<span class="fn">register_autograd</span>(
    <span class="str">"my_lib::rmsnorm"</span>, _rmsnorm_backward,
    setup_context=_rmsnorm_setup_context,
)</code></pre>

<p>The op now has full autograd support. <code>torch.ops.my_lib.rmsnorm(x, w, eps).sum().backward()</code> works.</p>

<h2>Numerical testing: <code>gradcheck</code> is mandatory</h2>

<p>Hand-derived backwards are notoriously bug-prone. Sign errors, missing terms, wrong reduction axes — any of these gives subtly wrong gradients that train the model in the wrong direction. PyTorch ships a tool to catch this: <code>torch.autograd.gradcheck</code>.</p>

<pre><code><span class="kw">import</span> torch
<span class="kw">from</span> torch.autograd <span class="kw">import</span> gradcheck

<span class="com"># Use double precision — finite-diff gradcheck is too noisy in float</span>
x = torch.<span class="fn">randn</span>(<span class="num">4</span>, <span class="num">128</span>, dtype=torch.float64, requires_grad=<span class="kw">True</span>, device=<span class="str">'cuda'</span>)
w = torch.<span class="fn">randn</span>(<span class="num">128</span>, dtype=torch.float64, requires_grad=<span class="kw">True</span>, device=<span class="str">'cuda'</span>)

<span class="fn">assert</span> <span class="fn">gradcheck</span>(<span class="kw">lambda</span> x, w: torch.ops.my_lib.<span class="fn">rmsnorm</span>(x, w, <span class="num">1e-6</span>), (x, w))
<span class="fn">print</span>(<span class="str">"gradcheck PASSED"</span>)</code></pre>

<p><code>gradcheck</code> compares your analytical backward to numerical finite differences. For each input, it perturbs by ±ε, measures (f(x+ε) - f(x-ε))/(2ε), and checks against your registered backward. Tolerances are tight — sub-1e-6 relative error in fp64. Pass it before trusting your kernel.</p>

<p>From M6: this is exactly the same tool used to validate <code>torch.autograd.Function</code> custom autograd. The pattern is identical for <code>custom_op</code> with autograd.</p>

<h2>Common pitfalls</h2>

<div class="table-wrap">
<table>
<caption>Common CUDA extension bugs and what they look like</caption>
<thead><tr><th>Bug</th><th>Symptom</th><th>Fix</th></tr></thead>
<tbody>
<tr><td>Non-contiguous input</td><td>Wrong outputs, often subtle (right values in wrong places)</td><td>Call <code>x.contiguous()</code> in the wrapper, or assert contiguous</td></tr>
<tr><td>Missing <code>__syncthreads()</code></td><td>Race condition; output non-deterministic</td><td>Sync between phases of shared-memory access</td></tr>
<tr><td>Wrong block/grid sizing</td><td>Crashes, illegal memory access</td><td>Use <code>cudaGetLastError()</code> to catch launch failures; print kernel error</td></tr>
<tr><td>fp16/bf16 accumulation</td><td>Numerical drift, divergent training</td><td>Accumulate in fp32 (the kernel cast pattern); same recipe as M14</td></tr>
<tr><td>Backward doesn't match forward</td><td>gradcheck fails; loss curves diverge from reference</td><td>gradcheck with <code>fast_mode=False</code> first; debug each input's gradient</td></tr>
<tr><td>Forgot to pass autocast policy</td><td>Different precision in eager vs autocasted</td><td>For bf16/fp16 input the kernel handles it (via dispatch); ensure you accept those dtypes in <code>AT_DISPATCH</code></td></tr>
<tr><td>Memory layout mismatches</td><td>Crashes or garbage outputs</td><td>Match what your kernel expects; add <code>TORCH_CHECK</code> for shape/stride</td></tr>
</tbody>
</table>
</div>

<p>The single most useful debugging discipline: <strong>run with <code>CUDA_LAUNCH_BLOCKING=1</code></strong>. Without it, kernel errors are async and reported many calls later, with the wrong stack trace. With it set, errors surface at the kernel that caused them.</p>

<pre><code>CUDA_LAUNCH_BLOCKING=<span class="num">1</span> python my_test.py</code></pre>

<p>Slows things down substantially (kills async kernel queueing — M20). But for debugging, indispensable.</p>

<h2>When to write CUDA vs Triton</h2>

<p>You've now seen what writing a CUDA kernel looks like end-to-end. <em>Most kernels you'll write should be in Triton (M24) instead.</em> Reasons:</p>

<ul>
  <li><strong>Triton is ~3× shorter.</strong> The same RMSNorm in Triton fits in 30 lines, all Python. No dispatch templates, no <code>AT_DISPATCH</code> macros, no PYBIND11.</li>
  <li><strong>Triton is autotuned.</strong> The block sizes, num_warps, num_stages are searched automatically. The CUDA kernel above uses BLOCK=256 because that's a reasonable default — a Triton version would find the optimal value for your shape.</li>
  <li><strong>Triton compiles to PTX too.</strong> No performance penalty for the abstraction; Triton kernels run at near-cuBLAS speed for many ops. FlashAttention-2's reference implementation is in Triton.</li>
  <li><strong>Inductor uses Triton.</strong> When <code>torch.compile</code> generates fused kernels (M21), they're Triton. By using Triton yourself, you stay in the same ecosystem.</li>
</ul>

<p>Reasons to use raw CUDA anyway: (1) extreme low-level control (e.g., async copies via <code>cp.async</code>, tensor core MMA intrinsics, warp specialization on Hopper). (2) Integrating with existing CUDA libraries (CUTLASS templates, cuBLAS, cuDNN). (3) Educational. For typical custom ops in research code, Triton is the right tool — and you've now seen the lower-level mechanics it sits on top of.</p>

<div class="ndq">
<h4>About CUDA extensions</h4>

<p class="q">Why is <code>cpp_extension.load</code> so slow on first call?</p>
<p class="a">It runs nvcc + g++. CUDA compilation is genuinely slow — for a kernel with templates and aggressive optimization, 30-60 seconds is normal. PyTorch caches the compiled <code>.so</code> in <code>~/.cache/torch_extensions</code> keyed by source content hash; subsequent imports are essentially free. If you're iterating fast on a kernel, set <code>verbose=True</code> to see exactly what nvcc is doing.</p>

<p class="q">My kernel works in eager mode but breaks under <code>torch.compile</code>. Why?</p>
<p class="a">Most common: you registered with <code>custom_op</code> but didn't register a fake function. Without the fake, <code>torch.compile</code>'s tracer can't figure out output shapes during meta-tensor execution. Add <code>@op.register_fake()</code> with a function that returns an empty tensor of the correct shape and dtype.</p>

<p class="q">When does writing my own CUDA kernel actually beat ATen / cuBLAS?</p>
<p class="a">Three scenarios. (1) Operations with no library equivalent (custom activations, exotic attention variants). (2) Fused operations where avoiding HBM round-trips is the win — e.g., layer_norm + linear + activation in one kernel. (3) Operations with shape patterns the library doesn't handle well (very small or very irregular). Don't reimplement matmul; cuBLAS will beat you. Don't reimplement softmax; ATen's is excellent. <em>Find the gap.</em></p>

<p class="q">How do I profile my custom kernel?</p>
<p class="a">Same tools as ATen kernels (M13). The PyTorch profiler captures it as a regular CUDA op. For deeper analysis, <code>nsys</code> and <code>ncu</code>: <code>nsys profile python script.py</code> gives a system-level timeline; <code>ncu --kernel-name your_kernel python script.py</code> gives per-kernel hardware counters (occupancy, memory bandwidth utilization, register usage). When tuning, <code>ncu</code> is the tool that tells you whether you're hitting the roofline.</p>

<p class="q">What's <code>__restrict__</code>?</p>
<p class="a">A hint to the compiler: "no other pointer in this function aliases this one." In our kernel, <code>x</code>, <code>weight</code>, and <code>y</code> are guaranteed disjoint, so the compiler can keep more values in registers and avoid redundant loads. Without it, the compiler has to assume a write to <code>y[i]</code> might invalidate <code>x[j]</code> reads. Standard practice for performance-critical kernels.</p>

<p class="q">Can my CUDA extension call cuBLAS or other libraries?</p>
<p class="a">Yes — link against them in the build args. <code>extra_ldflags=["-lcublas"]</code> in <code>cpp_extension.load</code>; in setuptools, add <code>libraries=["cublas"]</code> to the extension. Useful when you want a fused op that calls cuBLAS for the matmul portion and your kernel for surrounding ops. Wrap cuBLAS handles in a long-lived object — creating handles is expensive.</p>
</div>

<h2>The complete picture (zoomed out)</h2>

<p>You've built one custom kernel end-to-end. The pieces:</p>

<ol>
  <li><strong>The CUDA kernel</strong> in <code>.cu</code>: <code>__global__</code> function with thread/block indexing, shared memory reductions, fp32 accumulation. One block per row; threads stride for coalesced loads.</li>
  <li><strong>The C++ wrapper</strong> in <code>.cpp</code>: input validation, allocates auxiliary buffers, dispatches to the CUDA launcher, exposes via PYBIND11.</li>
  <li><strong>The build</strong>: JIT via <code>cpp_extension.load</code> for development, setuptools/CUDAExtension for shipping.</li>
  <li><strong>The Python wrapper</strong>: <code>cpp_extension.load(...)</code> on import, function that calls into the extension.</li>
  <li><strong>The op registration</strong>: <code>@torch.library.custom_op</code> with a Python fallback, <code>@register_kernel("cuda")</code> for the CUDA path, <code>@register_fake()</code> for compile/tracing, <code>register_autograd</code> for the backward.</li>
  <li><strong>The validation</strong>: <code>gradcheck</code> in fp64 to verify backward correctness; numerical comparison against a reference implementation in fp16/bf16 to catch low-precision drift.</li>
</ol>

<p>Six pieces, each individually simple. Together they make a kernel that's a first-class PyTorch op — autograd recognizes it, autocast handles it, <code>torch.compile</code> traces it, distributed training composes with it. <strong>Everything you've learned about PyTorch internals from M19-M21 is what makes this possible.</strong></p>

<h2>Code Magnets: register a CUDA extension as a first-class op</h2>

<p>You've compiled <code>_ext</code> via <code>cpp_extension.load</code> and it has a <code>my_op</code> function. Now register it through <code>torch.library</code> so autograd, compile, and dispatch all work. Three magnets are wrong choices.</p>

<div class="magnets">
<p>Arrange the magnets into a complete registration.</p>

<div class="magnet-pool">
  <span class="magnet">@torch.library.custom_op("my_lib::my_op", mutates_args=())</span>
  <span class="magnet">def my_op(x: torch.Tensor) -> torch.Tensor:</span>
  <span class="magnet">    return x * x          # CPU fallback</span>
  <span class="magnet">@my_op.register_kernel("cuda")</span>
  <span class="magnet">def _cuda_impl(x): return _ext.my_op(x.contiguous())</span>
  <span class="magnet">@my_op.register_fake()</span>
  <span class="magnet">def _fake(x): return torch.empty_like(x)</span>
  <span class="magnet">def _bwd(ctx, grad): return 2 * ctx.saved_tensors[0] * grad</span>
  <span class="magnet">def _setup_ctx(ctx, inputs, output): ctx.save_for_backward(inputs[0])</span>
  <span class="magnet">torch.library.register_autograd("my_lib::my_op", _bwd, setup_context=_setup_ctx)</span>
  <span class="magnet">torch.ops.my_lib.my_op = my_op</span>
  <span class="magnet">my_op = torch.compile(my_op)</span>
  <span class="magnet">def _cuda_impl(x): return _ext.my_op(x)        # no .contiguous()</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">@</span>torch.library.<span class="fn">custom_op</span>(<span class="str">"my_lib::my_op"</span>, mutates_args=())
<span class="kw">def</span> <span class="fn">my_op</span>(x: torch.Tensor) -&gt; torch.Tensor:
    <span class="kw">return</span> x * x          <span class="com"># CPU fallback</span>

<span class="kw">@</span>my_op.<span class="fn">register_kernel</span>(<span class="str">"cuda"</span>)
<span class="kw">def</span> <span class="fn">_cuda_impl</span>(x): <span class="kw">return</span> _ext.<span class="fn">my_op</span>(x.<span class="fn">contiguous</span>())

<span class="kw">@</span>my_op.<span class="fn">register_fake</span>()
<span class="kw">def</span> <span class="fn">_fake</span>(x): <span class="kw">return</span> torch.<span class="fn">empty_like</span>(x)

<span class="kw">def</span> <span class="fn">_setup_ctx</span>(ctx, inputs, output): ctx.<span class="fn">save_for_backward</span>(inputs[<span class="num">0</span>])
<span class="kw">def</span> <span class="fn">_bwd</span>(ctx, grad): <span class="kw">return</span> <span class="num">2</span> * ctx.saved_tensors[<span class="num">0</span>] * grad
torch.library.<span class="fn">register_autograd</span>(<span class="str">"my_lib::my_op"</span>, _bwd, setup_context=_setup_ctx)</code></pre>
<p>The traps:</p>
<ul>
  <li><code>torch.ops.my_lib.my_op = my_op</code>: the registration framework manages this internally. Manual assignment is unnecessary and can break ABI invariants.</li>
  <li><code>my_op = torch.compile(my_op)</code>: wrapping the registered op in compile breaks the dispatch — compile is applied at the call site (e.g., wrapping the model), not at the op definition.</li>
  <li><code>def _cuda_impl(x): return _ext.my_op(x)</code> without <code>.contiguous()</code>: the kernel assumes contiguous input. If a non-contiguous tensor is passed (a transpose or slice), the kernel reads garbage. Always defensively call <code>.contiguous()</code> in the wrapper.</li>
</ul>
<p>The full pattern: <strong>custom_op decorator → register_kernel for each backend → register_fake for shape inference → register_autograd for backward.</strong></p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each extension concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>__global__ function</div>
  <div>A. CUDA kernel entry point — runs on the device, called from host with <code>&lt;&lt;&lt;grid, block&gt;&gt;&gt;</code>.</div>

  <div>__syncthreads()</div>
  <div>B. Block-wide barrier — all threads in the block wait for each other.</div>

  <div>AT_DISPATCH_FLOATING_TYPES_AND2</div>
  <div>C. Generates type-specialized kernel calls for fp16/bf16/fp32 from one template.</div>

  <div>cpp_extension.load</div>
  <div>D. JIT-compiles the .cu/.cpp sources, caches the .so, returns a Python module.</div>

  <div>torch.library.custom_op</div>
  <div>E. Registers the function as a first-class op participating in the dispatcher.</div>

  <div>register_fake</div>
  <div>F. Provides a meta-tensor implementation for compile/tracing — shape only.</div>

  <div>gradcheck</div>
  <div>G. Validates analytical backward against finite differences. Run in fp64.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>__global__ function</strong> → A<br>
<strong>__syncthreads()</strong> → B<br>
<strong>AT_DISPATCH_FLOATING_TYPES_AND2</strong> → C<br>
<strong>cpp_extension.load</strong> → D<br>
<strong>torch.library.custom_op</strong> → E<br>
<strong>register_fake</strong> → F<br>
<strong>gradcheck</strong> → G
</p>
<p>The mental shortcut: <em>__global__ is the kernel, __syncthreads is the block barrier, AT_DISPATCH_FLOATING_TYPES_AND2 = type-dispatch macro, cpp_extension.load = JIT build, custom_op = register, register_fake = compile compatibility, gradcheck = backward validation in fp64</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team's hand-written CUDA RMSNorm kernel works for fp32 inputs but produces garbage for bf16. The Python tests for fp32 pass; bf16 fails silently with reasonable-but-wrong values. What's the most likely cause?</p>
<details class="answer"><summary>show answer</summary>
<p>Accumulating in bf16 instead of fp32. Summing thousands of squared values with bf16's 7-bit mantissa is numerically unstable — small contributions get rounded away when added to a larger running sum. The standard recipe (which our kernel uses): inputs in <code>scalar_t</code> (bf16/fp16/fp32), but the reduction accumulator is always <code>float</code>. The cast is explicit: <code>float v = (float)row_x[i]</code>, <code>local_sum += v * v</code>. This is a direct application of M14: low precision for storage and matmul, fp32 for reductions and norms. Skipping the fp32 accumulation is the most common bug in custom normalization kernels.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Why does the kernel use stride loops (<code>for (int i = tid; i &lt; D; i += BLOCK_SIZE)</code>) instead of a fixed range per thread (<code>for (int i = tid * (D / BLOCK_SIZE); i &lt; (tid + 1) * (D / BLOCK_SIZE); i++)</code>)?</p>
<details class="answer"><summary>show answer</summary>
<p>Memory coalescing. The stride pattern means thread 0 reads index 0, thread 1 reads index 1, ..., thread 31 reads index 31 — 32 consecutive elements at the same time. The memory subsystem can serve those as a single transaction (one coalesced read of 32 × dtype bytes). With the contiguous-range pattern, thread 0 reads indices 0..7, thread 1 reads 8..15, etc. — at any given moment, the 32 threads of a warp are reading indices spread by 8 each, which is uncoalesced and ~10× slower. <strong>Stride-by-block is the standard pattern for a reason.</strong> It looks weird until you remember the 32-threads-of-a-warp execute the loop iterations in lockstep.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> A team's custom kernel "works" but their loss curve diverges slightly from the PyTorch reference implementation. <code>gradcheck</code> passes in fp64. What might still be wrong?</p>
<details class="answer"><summary>show answer</summary>
<p><code>gradcheck</code> runs in fp64 — your analytical backward is correct in exact arithmetic. But in fp16/bf16 the reduction order matters: a different summation tree gives slightly different results from the reference, accumulating over many steps into a visible curve divergence. Diagnosis: compare outputs op-by-op in fp16 against the reference. If individual outputs differ by ~1e-3 relative, that's normal numerical noise; if they differ by ~1e-1, something is wrong (e.g., accumulator dtype, or a computation simplification that's mathematically equivalent but numerically different). The standard fix: ensure your reduction order matches the reference, or accept the difference if it's within "stochastic noise" of training.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> Sketch what the <code>register_fake</code> function should do for a custom op that performs <code>(B, M, K) × (B, K, N) -&gt; (B, M, N)</code> batched matmul.</p>
<details class="answer"><summary>show answer</summary>
<pre><code><span class="kw">@</span>my_op.<span class="fn">register_fake</span>()
<span class="kw">def</span> <span class="fn">_my_op_fake</span>(a, b):
    <span class="com"># a: (B, M, K), b: (B, K, N)</span>
    <span class="fn">assert</span> a.<span class="fn">size</span>(<span class="num">0</span>) == b.<span class="fn">size</span>(<span class="num">0</span>), <span class="str">"batch mismatch"</span>
    <span class="fn">assert</span> a.<span class="fn">size</span>(<span class="num">2</span>) == b.<span class="fn">size</span>(<span class="num">1</span>), <span class="str">"K mismatch"</span>
    B, M, K = a.<span class="fn">shape</span>
    N = b.<span class="fn">size</span>(<span class="num">2</span>)
    <span class="kw">return</span> a.<span class="fn">new_empty</span>((B, M, N))</code></pre>
<p>Three things: (1) verify shapes (so compile-time tracing fails fast on bugs), (2) compute the output shape from inputs, (3) return an empty tensor of the right shape, dtype, and device using <code>a.new_empty(shape)</code> — which inherits a's dtype and device. The fake function never runs the actual kernel; it just produces meta-info for the FX graph.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>A CUDA extension is <strong>three files</strong>: <code>.cu</code> (kernel), <code>.cpp</code> (binding), <code>.py</code> (loader). <code>cpp_extension.load</code> JIT-compiles them; <code>setuptools + CUDAExtension</code> ahead-of-time builds.</li>
  <li>The kernel pattern: <code>__global__</code> entry, thread/block indexing, shared memory for cooperation, <code>__syncthreads()</code> for barriers, fp32 accumulation for reductions.</li>
  <li><strong>One block per "outer" dimension</strong>, threads stride for coalesced loads. Standard layout for normalizations and per-row ops.</li>
  <li><strong>Coalesced memory access</strong>: thread <code>tid</code> reads element <code>tid + k*BLOCK_SIZE</code>. The 32 threads of a warp read 32 consecutive elements simultaneously — single memory transaction.</li>
  <li><code>AT_DISPATCH_FLOATING_TYPES_AND2</code> generates type-specialized kernel calls for fp16/bf16/fp32 from one template — same kernel handles multiple dtypes.</li>
  <li>The C++ binding validates inputs (<code>TORCH_CHECK</code>), allocates auxiliary buffers (<code>x.options()</code>), and exposes via <code>PYBIND11_MODULE</code>.</li>
  <li><strong>Register through <code>torch.library</code></strong>: <code>@custom_op</code> for the Python fallback, <code>@register_kernel("cuda")</code> for the CUDA path, <code>@register_fake()</code> for compile, <code>register_autograd</code> for backward. Now the op is first-class.</li>
  <li><strong><code>gradcheck</code> in fp64</strong> is mandatory for hand-written backwards. Run it once, every time you change the backward.</li>
  <li>Common pitfalls: non-contiguous inputs, missing syncs, fp16/bf16 accumulation, missing autograd registration, async kernel errors. <code>CUDA_LAUNCH_BLOCKING=1</code> for debugging.</li>
  <li><strong>For most ops, use Triton (M24) instead</strong>. Triton is shorter, autotuned, and integrates with Inductor. Raw CUDA when you need extreme control or library integration.</li>
  <li>The reflex: a custom kernel is justified when <em>fusion</em> would save HBM round-trips (memory-bound ops) or when you need an op that isn't in ATen. Don't reimplement matmul.</li>
</ul>
</div>

<p>Module 24 takes the same RMSNorm and writes it in Triton. You'll see the same hardware concepts (blocks, shared memory, fp32 accumulation) but expressed in Python with autotuning baked in. After Triton, M25 walks through FlashAttention as the case study where all of this becomes load-bearing — tiles, shared memory, fused softmax, and the recompute trick that turns attention from memory-bound to compute-bound.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">23</span>
  <span>C++/CUDA extensions</span>
</div>
"""

emit("23_cpp_cuda_extensions", "Module 23 — C++/CUDA extensions", BODY)
