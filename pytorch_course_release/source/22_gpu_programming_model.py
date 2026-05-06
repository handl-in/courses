#!/usr/bin/env python3
"""Module 22: GPU programming model — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part VIII · Module 22</div>
  <h1 class="module-title">GPU programming <em>model</em></h1>
  <p class="module-sub">— SMs, warps, threads, the memory hierarchy, and the roofline that decides whether your kernel is bound by compute or by memory bandwidth</p>
</div>

<p>For 21 modules we treated the GPU as a fast box that runs whatever ATen kernel the dispatcher hands it. Part VIII opens the box. The next four modules — GPU model (this one), CUDA extensions (M23), Triton (M24), and FlashAttention as a case study (M25) — are about <em>writing</em> the kernels yourself, not just calling them.</p>

<p>Before we write any code, we need to know what we're targeting. A modern GPU is not a CPU with more cores; it's a fundamentally different machine. CPUs minimize per-thread latency; GPUs maximize collective throughput. CPUs have huge caches and complex out-of-order execution; GPUs have tiny caches per thread and rely on having tens of thousands of threads in flight to hide memory latency. Until you have the hardware model in your head, kernel performance feels arbitrary.</p>

<div class="keyidea">
A GPU is built around three nested concepts. The hardware unit is the <strong>SM</strong> (streaming multiprocessor) — an A100 has 108, an H100 has 132. SMs execute work in <strong>warps</strong> of 32 threads in lockstep. Threads are organized into <strong>blocks</strong>, and blocks form a <strong>grid</strong> — a kernel launches a grid of blocks across all the SMs. The other axis is the <strong>memory hierarchy</strong>: registers (per thread, fastest) → shared memory (per block, ~10× slower) → L2 (chip-wide, ~50× slower) → HBM (off-chip, ~500× slower than registers). Every kernel-writing decision is shaped by these two structures.
</div>

<h2>One new face</h2>

<div class="character" style="--c: #1f5f5b;">
  <div class="avatar" style="background: #1f5f5b; color: #fff;">W</div>
  <div>
    <p class="who">Warp</p>
    <p class="name">"I'm 32 threads that execute in lockstep. We share a single program counter."</p>
    <p class="says">When you launch a CUDA kernel with a thread block of 256 threads, what really runs is 8 of me — 8 warps. Within me, all 32 threads execute the <em>same instruction</em> at the same time, just on different data. If your code branches and half my threads take one path while half take another, I have to run <em>both</em> paths sequentially with the inactive lane masked off. That's <strong>warp divergence</strong>: the cost of writing if/else over per-thread state. I'm efficient when my 32 threads agree.</p>
  </div>
</div>

<h2>The execution hierarchy</h2>

<p>Three levels nested inside each other:</p>

<ul>
  <li><strong>Thread</strong>: the smallest unit. Has its own registers and program counter (logically). Runs your kernel code.</li>
  <li><strong>Warp</strong>: 32 threads that <em>actually</em> execute together in lockstep on the hardware. The warp is the real execution unit — a thread is a software abstraction inside a warp.</li>
  <li><strong>Block (Cooperative Thread Array)</strong>: a group of threads (1-1024) that runs entirely on one SM. Threads in a block can share fast on-chip memory and synchronize.</li>
  <li><strong>Grid</strong>: the collection of all blocks for one kernel launch. Blocks are independent and may run on any SM in any order.</li>
</ul>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 420" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">CUDA execution hierarchy: Grid → Blocks → Warps → Threads</text>

  <!-- The Grid -->
  <text x="20" y="55" font-size="12" font-weight="700" fill="#1a1612">Grid (one kernel launch — many blocks)</text>
  <g transform="translate(20, 65)">
    <rect x="0" y="0" width="700" height="60" fill="none" stroke="#6b5d4f" stroke-width="1.5" stroke-dasharray="4 3"/>
    <!-- Blocks distributed -->
    <rect x="10"  y="10" width="80" height="40" fill="#fff8a8" stroke="#1a1612"/><text x="50" y="34" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Block 0</text>
    <rect x="100" y="10" width="80" height="40" fill="#fff8a8" stroke="#1a1612"/><text x="140" y="34" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Block 1</text>
    <rect x="190" y="10" width="80" height="40" fill="#fff8a8" stroke="#1a1612"/><text x="230" y="34" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Block 2</text>
    <rect x="280" y="10" width="80" height="40" fill="#fff8a8" stroke="#1a1612"/><text x="320" y="34" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Block 3</text>
    <text x="420" y="34" font-size="11" font-style="italic" fill="#6b5d4f">… many more, distributed across SMs</text>
  </g>

  <!-- One block expanded into warps -->
  <text x="20" y="155" font-size="12" font-weight="700" fill="#c1502e">Zoom into Block 0: 256 threads = 8 warps</text>
  <g transform="translate(20, 165)">
    <rect x="0" y="0" width="700" height="80" fill="#fff8a8" stroke="#1a1612" stroke-width="1.5"/>
    <!-- 8 warps -->
    <rect x="10"  y="10" width="80" height="60" fill="#fcecec" stroke="#c1502e"/><text x="50" y="44" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Warp 0</text><text x="50" y="58" text-anchor="middle" font-size="9" fill="#1a1612">32 threads</text>
    <rect x="95"  y="10" width="80" height="60" fill="#fcecec" stroke="#c1502e"/><text x="135" y="44" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Warp 1</text><text x="135" y="58" text-anchor="middle" font-size="9" fill="#1a1612">32 threads</text>
    <rect x="180" y="10" width="80" height="60" fill="#fcecec" stroke="#c1502e"/><text x="220" y="44" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Warp 2</text><text x="220" y="58" text-anchor="middle" font-size="9" fill="#1a1612">32 threads</text>
    <rect x="265" y="10" width="80" height="60" fill="#fcecec" stroke="#c1502e"/><text x="305" y="44" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Warp 3</text><text x="305" y="58" text-anchor="middle" font-size="9" fill="#1a1612">32 threads</text>
    <rect x="350" y="10" width="80" height="60" fill="#fcecec" stroke="#c1502e"/><text x="390" y="44" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Warp 4</text>
    <rect x="435" y="10" width="80" height="60" fill="#fcecec" stroke="#c1502e"/><text x="475" y="44" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Warp 5</text>
    <rect x="520" y="10" width="80" height="60" fill="#fcecec" stroke="#c1502e"/><text x="560" y="44" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Warp 6</text>
    <rect x="605" y="10" width="80" height="60" fill="#fcecec" stroke="#c1502e"/><text x="645" y="44" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Warp 7</text>
  </g>

  <!-- Zoom into a warp: 32 threads in lockstep -->
  <text x="20" y="275" font-size="12" font-weight="700" fill="#1f5f5b">Zoom into Warp 0: 32 threads run the SAME instruction in lockstep</text>
  <g transform="translate(20, 285)">
    <rect x="0" y="0" width="700" height="50" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <!-- 32 threads -->
    <g>
    <rect x="5"   y="10" width="20" height="30" fill="#fff" stroke="#1f5f5b" stroke-width="0.5"/><text x="15" y="27" text-anchor="middle" font-size="8" fill="#1a1612">t0</text>
    <rect x="27"  y="10" width="20" height="30" fill="#fff" stroke="#1f5f5b" stroke-width="0.5"/><text x="37" y="27" text-anchor="middle" font-size="8" fill="#1a1612">t1</text>
    <rect x="49"  y="10" width="20" height="30" fill="#fff" stroke="#1f5f5b" stroke-width="0.5"/><text x="59" y="27" text-anchor="middle" font-size="8" fill="#1a1612">t2</text>
    <rect x="71"  y="10" width="20" height="30" fill="#fff" stroke="#1f5f5b" stroke-width="0.5"/><text x="81" y="27" text-anchor="middle" font-size="8" fill="#1a1612">t3</text>
    <rect x="93"  y="10" width="20" height="30" fill="#fff" stroke="#1f5f5b" stroke-width="0.5"/><text x="103" y="27" text-anchor="middle" font-size="8" fill="#1a1612">t4</text>
    <rect x="115" y="10" width="20" height="30" fill="#fff" stroke="#1f5f5b" stroke-width="0.5"/><text x="125" y="27" text-anchor="middle" font-size="8" fill="#1a1612">t5</text>
    <rect x="137" y="10" width="20" height="30" fill="#fff" stroke="#1f5f5b" stroke-width="0.5"/><text x="147" y="27" text-anchor="middle" font-size="8" fill="#1a1612">t6</text>
    <rect x="159" y="10" width="20" height="30" fill="#fff" stroke="#1f5f5b" stroke-width="0.5"/><text x="169" y="27" text-anchor="middle" font-size="8" fill="#1a1612">t7</text>
    </g>
    <text x="220" y="28" font-size="11" fill="#6b5d4f" font-style="italic">…</text>
    <g>
    <rect x="500" y="10" width="20" height="30" fill="#fff" stroke="#1f5f5b" stroke-width="0.5"/><text x="510" y="27" text-anchor="middle" font-size="8" fill="#1a1612">t28</text>
    <rect x="522" y="10" width="20" height="30" fill="#fff" stroke="#1f5f5b" stroke-width="0.5"/><text x="532" y="27" text-anchor="middle" font-size="8" fill="#1a1612">t29</text>
    <rect x="544" y="10" width="20" height="30" fill="#fff" stroke="#1f5f5b" stroke-width="0.5"/><text x="554" y="27" text-anchor="middle" font-size="8" fill="#1a1612">t30</text>
    <rect x="566" y="10" width="20" height="30" fill="#fff" stroke="#1f5f5b" stroke-width="0.5"/><text x="576" y="27" text-anchor="middle" font-size="8" fill="#1a1612">t31</text>
    </g>
    <text x="650" y="28" font-size="10" fill="#1f5f5b">→ same op, 32 lanes</text>
  </g>

  <!-- Annotations -->
  <text x="370" y="370" font-family="'Caveat', cursive" font-size="20" fill="#1a1612" text-anchor="middle">grid is the launch shape; SMs pull blocks off the queue</text>
  <text x="370" y="395" font-family="'Caveat', cursive" font-size="20" fill="#c1502e" text-anchor="middle">a block lives entirely on ONE SM — never moves once started</text>
  <text x="370" y="415" font-family="'Caveat', cursive" font-size="20" fill="#1f5f5b" text-anchor="middle">warps are the real execution unit — 32 threads, single PC, lockstep</text>
</svg>
</div>

<p>The whole point of this hierarchy is to give you a way to map a problem onto thousands of threads and have them coordinate at the right granularity. Threads in the same warp can share registers via shuffle ops. Threads in the same block can share <em>shared memory</em> — fast on-chip RAM — and synchronize via <code>__syncthreads()</code>. Threads in different blocks generally don't communicate; the only way to globally synchronize is to end the kernel and start another.</p>

<h3>Warp divergence: the lockstep tax</h3>

<p>Warps execute in lockstep. If you write:</p>

<pre><code><span class="kw">if</span> (threadIdx.x &lt; <span class="num">16</span>) {
    <span class="com">// half the warp does this</span>
    a = <span class="fn">expensive_op</span>(x);
} <span class="kw">else</span> {
    <span class="com">// other half does this</span>
    a = <span class="fn">other_op</span>(x);
}</code></pre>

<p>Within one warp, half the threads (lanes 0-15) take the if branch; the other half (lanes 16-31) take the else. The hardware <em>can't</em> run both branches simultaneously — there's only one program counter per warp. So it runs the if branch with lanes 16-31 masked off, then runs the else branch with lanes 0-15 masked off. Total time is the sum of both branches.</p>

<p>This is <strong>warp divergence</strong>. It's not always a disaster — branches that nearly always go one way are cheap (the unused branch is skipped most of the time). The bad case is when divergence happens on a hot path with substantial work in each branch.</p>

<p>Better: when you can, structure code so the branch condition aligns with warp boundaries. <code>if (threadIdx.x &lt; 32 * some_warp_id)</code> is fine — entire warps go one way or the other, no divergence. Or use predication: compute both, select the right one with a masked write.</p>

<h2>The streaming multiprocessor (SM)</h2>

<p>The SM is the hardware unit that <em>executes</em> warps. An A100 has 108 SMs. An H100 has 132. Each SM has:</p>

<ul>
  <li><strong>CUDA cores</strong>: arithmetic units. An H100 SM has 128 fp32 CUDA cores.</li>
  <li><strong>Tensor Cores</strong>: matmul-specialized units. An H100 SM has 4. They do 4×4 fp16/bf16 matmuls per cycle (much more for tf32/fp8).</li>
  <li><strong>Register file</strong>: ~64K 32-bit registers per SM. Distributed across active threads.</li>
  <li><strong>Shared memory + L1 cache</strong>: ~228 KB on H100, configurable split between shared memory (programmer-managed) and L1 cache (hardware-managed).</li>
  <li><strong>Warp schedulers</strong>: 4 per SM. Each cycle, each scheduler picks one ready warp and issues an instruction from it.</li>
</ul>

<p>The SM is a heavily over-subscribed machine. It has compute resources for executing some warps, but the warp scheduler keeps <em>more</em> warps in flight than it can run simultaneously. When one warp stalls (waiting for memory), another runs. <strong>This is how GPUs hide memory latency: by always having other work to switch to.</strong> An SM can hold up to 64 warps "resident" simultaneously on H100 (= 2048 threads); the scheduler keeps cycling through them.</p>

<h2>The memory hierarchy</h2>

<p>The other axis. GPUs have a steep memory hierarchy — orders of magnitude difference between levels in both bandwidth and latency. Knowing the hierarchy is the difference between writing a fast kernel and a slow one.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 380" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">GPU memory hierarchy (H100 numbers; A100 similar within ~30%)</text>

  <!-- Pyramid: register at top, HBM at bottom -->
  <!-- Registers -->
  <g transform="translate(0, 50)">
    <rect x="220" y="0" width="300" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="370" y="18" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">Registers (per thread)</text>
    <text x="370" y="32" text-anchor="middle" font-size="9" fill="#1a1612">~256 per thread · ~zero latency · accessed every instruction</text>
    <text x="540" y="25" font-size="10" fill="#6b5d4f">~10 PB/s aggregate</text>
  </g>

  <!-- Shared memory -->
  <g transform="translate(0, 100)">
    <rect x="170" y="0" width="400" height="40" fill="#fff8a8" stroke="#1a1612" stroke-width="2"/>
    <text x="370" y="18" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">Shared memory + L1 (per SM)</text>
    <text x="370" y="32" text-anchor="middle" font-size="9" fill="#1a1612">~228 KB · ~30 cycle latency · programmer-managed scratchpad</text>
    <text x="585" y="25" font-size="10" fill="#6b5d4f">~20 TB/s per SM</text>
  </g>

  <!-- L2 cache -->
  <g transform="translate(0, 150)">
    <rect x="120" y="0" width="500" height="40" fill="#fff5d8" stroke="#d4a017" stroke-width="2"/>
    <text x="370" y="18" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">L2 cache (chip-wide, shared by all SMs)</text>
    <text x="370" y="32" text-anchor="middle" font-size="9" fill="#1a1612">~50 MB · ~250 cycle latency · hardware-managed cache</text>
    <text x="635" y="25" font-size="10" fill="#6b5d4f">~5 TB/s</text>
  </g>

  <!-- HBM -->
  <g transform="translate(0, 200)">
    <rect x="70" y="0" width="600" height="50" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="370" y="20" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">HBM (global memory) — what nvidia-smi calls "GPU memory"</text>
    <text x="370" y="36" text-anchor="middle" font-size="10" fill="#1a1612">80 GB · ~500 cycle latency · 3 TB/s on H100, 2 TB/s on A100</text>
    <text x="685" y="25" font-size="10" fill="#6b5d4f">3 TB/s</text>
  </g>

  <!-- HBM is "off-chip" callout -->
  <text x="20" y="275" font-size="11" font-weight="700" fill="#c1502e">↑ Off-chip — bandwidth is the dominant cost</text>

  <!-- Speed comparison -->
  <text x="20" y="305" font-size="12" font-weight="700" fill="#1a1612">Bandwidth ratio (one byte fetched from each level):</text>
  <g transform="translate(50, 320)">
    <rect x="0" y="0" width="600" height="14" fill="#d4ecc8" stroke="#1f5f5b"/>
    <text x="100" y="10" text-anchor="middle" font-size="9" fill="#1a1612">register: 1×</text>
    <rect x="0" y="20" width="300" height="14" fill="#fff8a8" stroke="#1a1612"/>
    <text x="100" y="30" text-anchor="middle" font-size="9" fill="#1a1612">shared: ~1/2× (SM aggregate)</text>
    <rect x="0" y="40" width="60"  height="14" fill="#fff5d8" stroke="#d4a017"/>
    <text x="100" y="50" text-anchor="middle" font-size="9" fill="#1a1612">L2: ~1/8×</text>
    <rect x="0" y="60" width="20"  height="14" fill="#fcecec" stroke="#c1502e"/>
    <text x="100" y="70" text-anchor="middle" font-size="9" fill="#1a1612">HBM: ~1/30× — the wall</text>
  </g>
</svg>
</div>

<p>Read the diagram literally. Registers are essentially free — every arithmetic instruction reads and writes them with no extra cost. Shared memory is fast but limited (~228 KB per SM, shared by all blocks running on it). L2 is bigger but ~10× slower than shared. HBM is enormous (80 GB) but ~30× slower than registers in aggregate bandwidth — and far longer in latency.</p>

<p>The implication: <strong>kernels that read each byte of HBM only once and reuse it heavily in registers/shared memory are fast. Kernels that go back to HBM repeatedly are slow.</strong> A matmul of two 4K×4K fp32 matrices is 32 MB + 32 MB = 64 MB of input data. Read each byte once: 64 MB ÷ 3 TB/s = 21 µs. The actual compute is ~131 GFLOPs at fp32 ÷ 60 TFLOPs/s = ~2 µs. <em>If your matmul takes 21 µs you're memory-bound; if it takes 2 µs you're compute-bound</em>. Modern matmul kernels reuse data heavily and hit close to the 2 µs number.</p>

<h2>Roofline analysis: the rule that decides everything</h2>

<p>Every kernel falls into one of two regimes: <strong>memory-bound</strong> (limited by bandwidth) or <strong>compute-bound</strong> (limited by FLOP throughput). The roofline model captures this with a single number per kernel: <em>arithmetic intensity</em>.</p>

<pre><code>arithmetic_intensity = FLOPs_done / bytes_transferred</code></pre>

<p>Then you compare to the hardware's "ridge point":</p>

<pre><code>ridge_point = peak_FLOPs / peak_bandwidth   <span class="com"># operations per byte at the boundary</span></code></pre>

<p>For H100 fp16 with tensor cores: peak ~1000 TFLOPs/s, HBM bandwidth ~3 TB/s. Ridge point ≈ 1000 / 3 = 333 ops/byte. So:</p>

<ul>
  <li><strong>Arithmetic intensity &lt; 333 ops/byte</strong>: memory-bound. The kernel can't keep tensor cores fed. Bottleneck is HBM bandwidth.</li>
  <li><strong>Arithmetic intensity &gt; 333 ops/byte</strong>: compute-bound. Tensor cores are saturated; bandwidth is fine.</li>
</ul>

<p>A few real examples:</p>

<div class="table-wrap">
<table>
<caption>Arithmetic intensity of common ops on H100 fp16</caption>
<thead><tr><th>Op</th><th>FLOPs</th><th>Bytes</th><th>Intensity</th><th>Regime</th></tr></thead>
<tbody>
<tr><td>relu (1M elements)</td><td>1M</td><td>4 MB (read+write)</td><td>0.25</td><td>Memory-bound</td></tr>
<tr><td>matmul 256×256 × 256×256</td><td>34M</td><td>0.4 MB</td><td>~80</td><td>Memory-bound</td></tr>
<tr><td>matmul 4096×4096 × 4096×4096</td><td>137G</td><td>96 MB</td><td>~1430</td><td>Compute-bound</td></tr>
<tr><td>attention (one head, T=2048)</td><td>~33M</td><td>~50 MB (naive)</td><td>0.7</td><td>Memory-bound (until FlashAttention)</td></tr>
</tbody>
</table>
</div>

<p>Two takeaways. (1) Pointwise ops (relu, add, layer norm) are <em>always</em> memory-bound — there's barely any arithmetic per byte. Their fastest possible time is just the time to read inputs from HBM and write outputs back. <strong>This is exactly why fusion (M21) wins so dramatically</strong>: fusing 5 pointwise ops doesn't change FLOPs (still memory-bound) but cuts HBM traffic by 5×.</p>

<p>(2) Attention is memory-bound at typical sizes. The standard implementation reads and writes the full attention matrix (T×T per head) to HBM, which is huge memory traffic. <strong>FlashAttention</strong> (M25) restructures the computation to keep the attention matrix in shared memory and never materialize it in HBM — turning a memory-bound op into a compute-bound one.</p>

<h2>Tensor cores: the matmul accelerators</h2>

<p>One more piece. Standard CUDA cores do scalar operations: one multiply-add per cycle per core. Tensor cores do <em>small matrix multiply-adds</em>: one 4×4×4 matmul per cycle on Volta, much bigger blocks on Hopper.</p>

<p>For matmul-heavy work, tensor cores are 5-30× faster than CUDA cores at the same precision. The catch: they only accelerate specific shapes and dtypes. The shapes need to be multiples of 8 or 16 (depending on dtype) for tensor cores to even fire. The dtypes need to be fp16, bf16, tf32, int8, fp8, or fp4 (newer hardware adds more); plain fp32 mostly doesn't get tensor-core acceleration (TF32 does — that's why TF32 was created, M14).</p>

<p>Implications:</p>

<ul>
  <li><strong>Pad your matmul shapes to multiples of 8</strong> for fp16/bf16, multiples of 16 for fp8. A 4097×4097 matmul falls off tensor cores; pad to 4104×4104 and you get full speed.</li>
  <li><strong>Use bf16 (or fp16, or fp8 on H100+)</strong>. fp32 matmul without TF32 is dramatically slower than bf16 with tensor cores.</li>
  <li><strong>Embedding lookups, layer norm, softmax don't benefit from tensor cores</strong> — they're not matmul-shaped. They run on CUDA cores and are usually memory-bound regardless.</li>
</ul>

<h2>Occupancy: how many warps fit per SM</h2>

<p>An SM has fixed resources: register file, shared memory, warp scheduler slots. The number of warps that can be "resident" simultaneously on one SM is limited by whichever resource runs out first.</p>

<p><strong>Occupancy</strong> = (active warps per SM) / (maximum warps per SM). Higher occupancy means more warps available to switch to when one stalls — better latency hiding.</p>

<p>Three things drain occupancy:</p>

<ol>
  <li><strong>Register pressure</strong>: each thread's registers come from the SM's register file. If your kernel uses 256 registers per thread, you can only have a few warps resident (the file runs out). Compilers report this with <code>--ptxas-options=-v</code>.</li>
  <li><strong>Shared memory</strong>: blocks need shared memory to run; if each block needs 100 KB, only 2 blocks fit per SM.</li>
  <li><strong>Block size</strong>: smaller blocks → fewer threads → fewer warps. But blocks too small (&lt;128 threads) waste warp scheduler slots.</li>
</ol>

<p>Higher occupancy is usually better — but not always. Some kernels (especially compute-bound ones with heavy register reuse) want <em>low</em> occupancy: more registers per thread to keep more data in registers, fewer warps to context-switch between. <strong>The matmul kernels in cuBLAS run at very low occupancy (~25%) but saturate tensor cores by keeping huge tiles in registers and shared memory.</strong> Higher occupancy would actually slow them down.</p>

<p>The rule of thumb: if your kernel is memory-bound, want high occupancy (more warps to hide latency). If it's compute-bound on tensor cores, low-to-medium occupancy with good register reuse is fine.</p>

<h2>Concrete numbers: A100 vs H100</h2>

<p>Numbers ground intuition. Memorize roughly:</p>

<div class="table-wrap">
<table>
<caption>The two GPUs you'll most often target (rough numbers)</caption>
<thead><tr><th>Spec</th><th>A100 80GB</th><th>H100 80GB SXM</th></tr></thead>
<tbody>
<tr><td>SMs</td><td>108</td><td>132</td></tr>
<tr><td>Tensor cores per SM</td><td>4 (3rd gen)</td><td>4 (4th gen, much bigger)</td></tr>
<tr><td>Peak fp16/bf16 TFLOPs (tensor cores)</td><td>312</td><td>~1000 (with sparsity 2×)</td></tr>
<tr><td>Peak fp8 TFLOPs (tensor cores)</td><td>—</td><td>~2000</td></tr>
<tr><td>HBM bandwidth</td><td>2 TB/s</td><td>3.35 TB/s</td></tr>
<tr><td>L2 cache</td><td>40 MB</td><td>50 MB</td></tr>
<tr><td>Shared memory + L1 per SM</td><td>192 KB</td><td>228 KB</td></tr>
<tr><td>Memory</td><td>80 GB HBM2e</td><td>80 GB HBM3</td></tr>
<tr><td>NVLink bandwidth</td><td>600 GB/s</td><td>900 GB/s</td></tr>
</tbody>
</table>
</div>

<p>Roughly: H100 is 3× faster in compute (especially fp8), 50% faster in memory bandwidth, with somewhat more shared memory and L2. The ridge point gets <em>higher</em> on H100 — meaning more ops have to be done per byte to be compute-bound. Memory bandwidth is increasingly the binding constraint.</p>

<h2>The kernel writer's mental model</h2>

<p>Putting it all together. When you sit down to write a kernel:</p>

<ol>
  <li><strong>Compute arithmetic intensity</strong>. FLOPs / bytes. Compare to the ridge point. Are you memory-bound or compute-bound?</li>
  <li><strong>If memory-bound</strong>: minimize HBM traffic. Read inputs from HBM <em>once</em>. Reuse data across many ops via shared memory. Fuse multiple ops into one kernel.</li>
  <li><strong>If compute-bound</strong>: keep tensor cores fed. Use the right dtypes (bf16/fp16/fp8). Pad shapes to tensor-core-friendly multiples. Maximize register reuse — blocked algorithms with tiles that fit in registers and shared memory.</li>
  <li><strong>Manage occupancy intentionally</strong>. High occupancy for memory-bound (latency hiding); low-medium with good register reuse for compute-bound.</li>
  <li><strong>Avoid warp divergence</strong> on hot paths. Align branches to warp boundaries when possible.</li>
  <li><strong>Plan synchronization</strong>. <code>__syncthreads()</code> is cheap within a block; cross-block sync requires ending the kernel. Most kernels are designed to do all the work for one tile within a single block.</li>
</ol>

<div class="ndq">
<h4>About the GPU model</h4>

<p class="q">If a thread is just a software abstraction inside a warp, why does CUDA pretend each thread is independent?</p>
<p class="a">Programming model abstraction. Writing "for each thread, do X" is much easier than "for each warp, do X for 32 lanes simultaneously." The hardware enforces the lockstep underneath; the language lets you ignore it most of the time. You only think about warps explicitly when (a) using warp-level primitives like shuffles, or (b) avoiding divergence.</p>

<p class="q">How do I know how many SMs a GPU has, dynamically?</p>
<p class="a">In Python: <code>torch.cuda.get_device_properties(0).multi_processor_count</code>. Useful when sizing grids: a common heuristic is to launch enough blocks to keep all SMs busy. <code>nvidia-smi -q</code> also shows it.</p>

<p class="q">Why is shared memory only 228 KB per SM? Couldn't they make it bigger?</p>

<p class="a">Shared memory is built from SRAM cells right inside each SM, in the shadow of the compute units. SRAM is fast but expensive in silicon area; making it 10× bigger would mean a much bigger chip or fewer SMs. The size is a deliberate engineering tradeoff. The way around it: tile your computation so each tile fits in 228 KB. That's exactly what FlashAttention does (M25).</p>

<p class="q">What does <code>__syncthreads()</code> actually do?</p>
<p class="a">It's a barrier inside a block. All warps in the block wait until every warp has reached the barrier; then they continue together. Cheap — typically a few cycles. Used between phases of a tile-based algorithm: load to shared, sync, compute, sync, write out. Only synchronizes <em>within a block</em>; cross-block sync requires ending the kernel.</p>

<p class="q">When does it make sense to write a kernel by hand vs use a library like cuBLAS or Triton?</p>
<p class="a">Three cases for hand-writing: (1) operations that don't have a library (exotic activation functions, custom attention variants). (2) Performance-critical fused operations where avoiding intermediate writes to HBM is the win — Triton is usually the right tool here, not raw CUDA. (3) Educational. For matmul: <em>just use cuBLAS</em>. The cuBLAS kernels are hand-tuned by NVIDIA experts and you won't beat them for the standard cases. Even hand-writing in CUDA, you'd typically link against cuBLAS for the matmul cores.</p>
</div>

<h2>Looking ahead</h2>

<p>You now know what we're targeting:</p>

<ul>
  <li>A grid of blocks, where each block lives entirely on one SM.</li>
  <li>Within a block, threads grouped into warps of 32 that execute in lockstep.</li>
  <li>A steep memory hierarchy: registers (free) → shared (228 KB, fast) → L2 (50 MB, slow) → HBM (80 GB, very slow).</li>
  <li>Tensor cores for matmul — fast but shape-restrictive.</li>
  <li>Roofline: arithmetic intensity vs ridge point decides memory-bound vs compute-bound.</li>
</ul>

<p>M23 picks up immediately: how to write CUDA C++ kernels and link them into PyTorch via the cpp_extension and torch.library APIs. M24 introduces Triton, the higher-level kernel language that maps cleanly to this model and is what Inductor (M21) uses internally. M25 takes everything we've built — memory hierarchy, tiling, shared memory, fusion — and walks through FlashAttention as the canonical case study.</p>

<h2>Code Magnets: identify the regime</h2>

<p>You're given four ops. For each, determine whether it's memory-bound or compute-bound on H100 (ridge point ~333 ops/byte at bf16). Three of the magnets pair an op with the wrong regime; arrange the correct ones.</p>

<div class="magnets">
<p>Pick the four correct (op, regime) pairings.</p>

<div class="magnet-pool">
  <span class="magnet">layer_norm on (4096, 1024) bf16 → memory-bound</span>
  <span class="magnet">layer_norm on (4096, 1024) bf16 → compute-bound</span>
  <span class="magnet">matmul (4096×4096) × (4096×4096) bf16 → compute-bound</span>
  <span class="magnet">matmul (4096×4096) × (4096×4096) bf16 → memory-bound</span>
  <span class="magnet">elementwise relu on 1B elements bf16 → memory-bound</span>
  <span class="magnet">attention (B=8, H=32, T=2048, naive impl) bf16 → memory-bound (HBM-resident attention matrix)</span>
  <span class="magnet">attention (B=8, H=32, T=2048, naive impl) bf16 → compute-bound</span>
  <span class="magnet">small matmul (16×16) × (16×16) bf16 → compute-bound</span>
  <span class="magnet">small matmul (16×16) × (16×16) bf16 → memory-bound (overhead dominates)</span>
</div>

<details class="answer"><summary>show solution</summary>
<p>The four correct pairings:</p>
<ol>
  <li>layer_norm on (4096, 1024) bf16 → <strong>memory-bound</strong>. Reads/writes ~16 MB; does ~16M flops. Intensity ~1, far below ridge.</li>
  <li>matmul (4096×4096) × (4096×4096) bf16 → <strong>compute-bound</strong>. ~137G flops, ~96 MB read. Intensity ~1430, well above ridge of 333. With tensor cores hitting full throughput, this is one of the few ops that's compute-bound on H100.</li>
  <li>elementwise relu on 1B elements bf16 → <strong>memory-bound</strong>. 1G flops vs 4 GB transfer. Intensity 0.25.</li>
  <li>attention (B=8, H=32, T=2048, naive) bf16 → <strong>memory-bound</strong>. The naive implementation materializes the (T×T) attention scores in HBM. Bytes transferred dominates flops. Hence FlashAttention (M25) — restructure to keep attention matrix in shared memory.</li>
</ol>
<p>The "small matmul" (16×16) is also memory-bound (the matmul itself has good intensity, but the launch overhead and HBM read-write of small inputs dominates), but for a different reason than the others — at this scale the kernel can barely warm up tensor cores. Both pairings ("memory-bound") in the pool are correct, but it's a stretch — this is mostly an "overhead-bound" regime.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each GPU concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>SM (streaming multiprocessor)</div>
  <div>A. Hardware unit that executes warps; an A100 has 108, an H100 has 132.</div>

  <div>Warp</div>
  <div>B. 32 threads in lockstep — the real execution unit.</div>

  <div>Block</div>
  <div>C. Group of threads on one SM; can share fast memory and synchronize.</div>

  <div>Shared memory</div>
  <div>D. ~228 KB per SM; programmer-managed scratchpad, ~10× faster than L2.</div>

  <div>HBM</div>
  <div>E. Off-chip "GPU memory" — 80 GB, slowest level, the bandwidth wall.</div>

  <div>Tensor core</div>
  <div>F. Matmul-specialized unit; 5-30× faster than CUDA cores for fp16/bf16 matmul.</div>

  <div>Roofline</div>
  <div>G. Compares arithmetic intensity to ridge point — decides memory- vs compute-bound.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>SM</strong> → A<br>
<strong>Warp</strong> → B<br>
<strong>Block</strong> → C<br>
<strong>Shared memory</strong> → D<br>
<strong>HBM</strong> → E<br>
<strong>Tensor core</strong> → F<br>
<strong>Roofline</strong> → G
</p>
<p>The mental shortcut: <em>SMs are the hardware units, warps are lockstep groups of 32, blocks live entirely on one SM, shared memory is the on-chip scratchpad, HBM is the off-chip wall, tensor cores accelerate matmul, roofline is the framework for deciding what bounds you</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> Why does adding a tiny <code>F.relu</code> (memory-bound) right after a giant matmul (compute-bound) almost halve the matmul's effective speed when not fused?</p>
<details class="answer"><summary>show answer</summary>
<p>The matmul's output sits in HBM after the matmul kernel finishes (write phase). Then the relu kernel reads it back from HBM, applies relu, writes the result to HBM. Net: the matmul output makes <em>two</em> round-trips through HBM (matmul writes it, relu reads + writes). Each pass is bandwidth-limited at 96 MB ÷ 3 TB/s = ~32 µs. With fusion (the matmul kernel can append relu inline before writing), you'd skip the extra read + the second write = ~64 µs saved. For a matmul that itself takes ~130 µs in the compute, that's about half its time gone to a pointwise op that "should" be free. This is exactly why <code>torch.compile</code> (M21) and Inductor's fusion logic exist — and why FlashAttention (M25) is so impactful.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> A kernel writer says "I doubled my register usage and got 30% slower despite using fewer threads — but the algorithm should be identical." What's likely happening?</p>
<details class="answer"><summary>show answer</summary>
<p>Occupancy collapse. By doubling register usage per thread, the SM's register file (which is fixed) can hold half as many threads. If they were already at marginal occupancy, the new occupancy might be too low to hide memory latency — warps stall on memory and there aren't enough other warps to switch to. Two fixes: (a) reduce register usage (smaller tiles, simpler loops), (b) accept the lower occupancy if the kernel is compute-bound and register-rich (cuBLAS does this on purpose). The diagnostic: <code>nsys</code> or <code>ncu</code> shows occupancy; if it dropped substantially with the change, that's likely your story.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> Sketch why FlashAttention (M25) takes attention from memory-bound to compute-bound, in roofline terms.</p>
<details class="answer"><summary>show answer</summary>
<p>Naive attention computes the full T×T attention scores matrix and writes it to HBM, then reads it back to multiply with V. Memory traffic scales with T². For T=2048, that's a (2048×2048) matrix = 8 MB per head per layer, all going through HBM repeatedly. Arithmetic intensity is low — most of the time is HBM round-trips.</p>
<p>FlashAttention restructures: tile Q, K, V into blocks. For each tile, compute partial softmax and partial output, accumulating into V's output buffer. The attention matrix never materializes in HBM — it lives in shared memory inside each block. Memory traffic is reduced to just reading Q, K, V once and writing the output once. Arithmetic per byte goes up dramatically. The kernel becomes compute-bound, hitting tensor-core peak throughput. We'll do this concretely in M25.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> Why are PyTorch's elementwise ops (add, relu, sigmoid) usually memory-bound regardless of how big the tensors are?</p>
<details class="answer"><summary>show answer</summary>
<p>They have constant arithmetic per element (one add, one max, one sigmoid). Per element: ~1 FLOP, but you read 4 bytes (fp32) or 2 bytes (bf16) and write 4 or 2 back. Intensity is on the order of 0.1 ops/byte regardless of tensor size. Always far below the ridge point of ~333 — always memory-bound. <em>The only knob you have for elementwise ops is to fuse them with neighboring ops to amortize the HBM read</em>. This is precisely what <code>torch.compile</code> does. There's no algorithmic trick that makes a single elementwise op compute-bound — you'd have to do hundreds of FLOPs per element, and the op wouldn't be elementwise anymore.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>A GPU is built around <strong>SMs</strong> (108 on A100, 132 on H100). SMs execute work in <strong>warps</strong> of 32 threads in lockstep. Threads form <strong>blocks</strong>; blocks form a <strong>grid</strong>.</li>
  <li><strong>Warp divergence</strong>: branches inside a warp serialize. Align branches to warp boundaries on hot paths.</li>
  <li>The <strong>memory hierarchy</strong>: registers (per thread, free) → shared memory + L1 (~228 KB per SM, fast scratchpad) → L2 (~50 MB chip-wide) → HBM (80 GB, off-chip, the wall).</li>
  <li>HBM bandwidth is the dominant cost for most kernels. Reading each byte once and reusing it heavily is the path to fast.</li>
  <li><strong>Roofline analysis</strong>: <code>arithmetic intensity = FLOPs/bytes</code>. Compare to ridge point (~333 ops/byte on H100 fp16). Below ridge: memory-bound. Above: compute-bound.</li>
  <li>Pointwise ops (relu, layer_norm) are <em>always</em> memory-bound. Fusion is the only lever — that's why <code>torch.compile</code> wins.</li>
  <li>Big matmuls are compute-bound on tensor cores. Use bf16/fp16, pad shapes to multiples of 8 (or 16 for fp8).</li>
  <li><strong>Tensor cores</strong>: matmul-specialized units, 5-30× faster than CUDA cores. Specific shapes and dtypes only.</li>
  <li><strong>Occupancy</strong>: active warps per SM ÷ max warps per SM. Higher hides memory latency. But not always better — compute-bound kernels with rich register reuse run best at low occupancy.</li>
  <li><code>__syncthreads()</code> is the within-block barrier — cheap. Cross-block sync requires ending the kernel.</li>
  <li>The kernel-writer's reflex: compute arithmetic intensity. Compare to ridge. Pick the strategy (minimize HBM traffic vs feed tensor cores) accordingly.</li>
  <li>Concrete numbers worth remembering: <strong>H100: 132 SMs, ~1000 fp16 TFLOPs, 3 TB/s HBM, 228 KB shared per SM</strong>. That fits on the back of a napkin and supports 90% of kernel reasoning.</li>
</ul>
</div>

<p>Module 23 takes this hardware model and writes our first kernel. C++/CUDA extensions through PyTorch's <code>cpp_extension</code> module: how to write a CUDA kernel, build it via JIT or setuptools, and register it through <code>torch.library.custom_op</code> from M19 so it integrates with autograd, autocast, and compile.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">22</span>
  <span>GPU programming model</span>
</div>
"""

emit("22_gpu_programming_model", "Module 22 — GPU programming model", BODY)
