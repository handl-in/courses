#!/usr/bin/env python3
"""Module 13: Profiling — where does the time go? — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part V · Module 13</div>
  <h1 class="module-title">Profiling: <em>where does the time go?</em></h1>
  <p class="module-sub">— why <code>time.time()</code> lies on GPU, the four canonical bottleneck modes, and reading a trace until you know exactly which lever to pull</p>
</div>

<p>M12 was about predicting memory before you OOM. This module is about predicting <em>time</em> before you spend three days waiting for a slow training run that didn't need to be slow.</p>

<p>The core problem: GPU work is asynchronous. When you write <code>y = model(x)</code> in Python, you didn't compute anything — you queued kernels for the GPU to run later. <code>time.time()</code> measures Python-side wall-clock and lies about kernel cost. Worse, in any nontrivial training loop your time is split across forward, backward, optimizer, data loading, and communication, and the bottleneck moves around as you scale. You can't fix what you can't see. This module teaches you to see.</p>

<div class="keyidea">
There are <strong>four canonical bottleneck modes</strong> in PyTorch training: <em>GPU-bound</em> (kernels are slow), <em>CPU-bound</em> (Python/dispatcher overhead is slow), <em>data-bound</em> (the loader can't keep up), and <em>communication-bound</em> (collective ops eat the schedule). Each shows a distinctive shape in a profiler trace. Reading traces is a small skill that pays back quickly: 10 minutes of profiling can save 10 hours of optimization in the wrong place.
</div>

<h2>The new face</h2>

<div class="character" style="--c: #1f5f5b;">
  <div class="avatar" style="background: #1f5f5b; color: #fff;">P</div>
  <div>
    <p class="who">Profiler</p>
    <p class="name">"I record what your code <em>actually did</em> on the GPU, with kernel-level precision."</p>
    <p class="says">Unlike <code>time.time()</code>, I don't lie about async kernels. I instrument both Python and CUDA, recording every op's launch time and execution time. I produce traces you can open in Chrome or Perfetto. I can tell you, for each kernel: how long it took, what shape its inputs were, who called it from Python. Use me before you optimize. The temptation to "guess and tune" is strong — resist it. Three minutes with me beats three hours of guessing.</p>
  </div>
</div>

<h2>Why <code>time.time()</code> lies (one more time)</h2>

<p>You met this in M3, but it's worth seeing the full failure mode:</p>

<pre><code><span class="kw">import</span> time, torch
x = torch.<span class="fn">randn</span>(<span class="num">8192</span>, <span class="num">8192</span>, device=<span class="str">'cuda'</span>)
w = torch.<span class="fn">randn</span>(<span class="num">8192</span>, <span class="num">8192</span>, device=<span class="str">'cuda'</span>)

t0 = time.<span class="fn">time</span>()
y = x @ w                          <span class="com"># queues a matmul kernel</span>
t1 = time.<span class="fn">time</span>()
<span class="fn">print</span>(<span class="fn">f"matmul: {(t1-t0)*1000:.2f} ms"</span>)     <span class="com"># prints something like 0.05 ms</span></code></pre>

<p>0.05 ms? An 8192² matmul takes more like 5 ms on an A100. The kernel hadn't actually run yet — Python returned as soon as the launch was queued. To measure correctly, force a sync:</p>

<pre><code>torch.cuda.<span class="fn">synchronize</span>()                 <span class="com"># wait for any prior queued work</span>
t0 = time.<span class="fn">time</span>()
y = x @ w
torch.cuda.<span class="fn">synchronize</span>()                 <span class="com"># wait for our kernel to actually finish</span>
t1 = time.<span class="fn">time</span>()
<span class="fn">print</span>(<span class="fn">f"matmul: {(t1-t0)*1000:.2f} ms"</span>)     <span class="com"># now ~5 ms — actual</span></code></pre>

<p>Better, but still not great — <code>time.time()</code> on Linux has microsecond-ish resolution and the syncs add overhead. For real measurements, use <code>torch.cuda.Event</code>:</p>

<pre><code>start = torch.cuda.<span class="fn">Event</span>(enable_timing=<span class="kw">True</span>)
end   = torch.cuda.<span class="fn">Event</span>(enable_timing=<span class="kw">True</span>)

start.<span class="fn">record</span>()
y = x @ w
end.<span class="fn">record</span>()
torch.cuda.<span class="fn">synchronize</span>()                 <span class="com"># needed before reading elapsed_time</span>
<span class="fn">print</span>(<span class="fn">f"matmul: {start.elapsed_time(end):.2f} ms"</span>)</code></pre>

<p>CUDA events record markers <em>on the GPU's command stream</em>, so the elapsed time between them is the actual kernel duration, not the Python wall clock. Resolution is sub-microsecond. This is what real benchmarking code uses.</p>

<div class="warn">
<strong>Always warm up before benchmarking.</strong> The first call to a kernel often triggers JIT compilation, autotuning (cuDNN), or memory allocation. Discard the first few iterations. Standard pattern: 5-10 warmup iterations, then 50-100 timed iterations, take the median. The first iteration can easily be 10× slower than steady state.
</div>

<h2><code>torch.profiler</code>: the proper tool</h2>

<p>Manually instrumenting with CUDA events is fine for one-off timing. For real profiling — finding the slowest 5% of operations across an entire training step — use <code>torch.profiler</code>.</p>

<pre><code><span class="kw">from</span> torch.profiler <span class="kw">import</span> profile, ProfilerActivity, schedule

<span class="kw">with</span> <span class="fn">profile</span>(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=<span class="fn">schedule</span>(wait=<span class="num">1</span>, warmup=<span class="num">2</span>, active=<span class="num">3</span>, repeat=<span class="num">1</span>),
    on_trace_ready=torch.profiler.<span class="fn">tensorboard_trace_handler</span>(<span class="str">'./trace'</span>),
    record_shapes=<span class="kw">True</span>,
    profile_memory=<span class="kw">True</span>,
    with_stack=<span class="kw">True</span>,
) <span class="kw">as</span> prof:
    <span class="kw">for</span> step, batch <span class="kw">in</span> <span class="fn">enumerate</span>(loader):
        <span class="fn">train_step</span>(model, opt, batch)
        prof.<span class="fn">step</span>()                       <span class="com"># tells the profiler we finished a step</span>
        <span class="kw">if</span> step &gt;= <span class="num">10</span>:
            <span class="kw">break</span></code></pre>

<p>The schedule is the part everyone gets wrong. Read it left to right:</p>

<ul>
  <li><code>wait=1</code>: skip the first step (often slow due to lazy init).</li>
  <li><code>warmup=2</code>: profile the next 2 steps but discard data — gives autotuners time to settle.</li>
  <li><code>active=3</code>: actually record the next 3 steps. This is the data you'll analyze.</li>
  <li><code>repeat=1</code>: do this whole cycle once (so 6 steps total, last 3 captured).</li>
</ul>

<p>Three things you almost always want:</p>

<ul>
  <li><code>record_shapes=True</code> — annotates each kernel with its input shapes. Essential for spotting "wait, why is this matmul tiny?"</li>
  <li><code>profile_memory=True</code> — overlays memory allocations on the timeline. Lets you see where the activation peak comes from.</li>
  <li><code>with_stack=True</code> — captures the Python call stack for each op. So when you see a slow kernel, you can find which line of code launched it.</li>
</ul>

<p>The <code>tensorboard_trace_handler</code> writes a JSON trace file you can open in Chrome's tracing tool, Perfetto (<code>ui.perfetto.dev</code>), or TensorBoard. Most modern PyTorch users prefer Perfetto — it's the same tool Android engineers use, and the UI is much better than Chrome's old one.</p>

<h2>The four bottleneck modes</h2>

<p>Open a profiler trace and you'll see two horizontal lanes for each step: a <strong>CPU lane</strong> (Python, kernel launches, dispatcher work) and a <strong>GPU lane</strong> (the actual kernels). The shape tells you everything about what's slow.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 480" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">The four canonical bottleneck shapes</text>

  <!-- ====== Panel 1: GPU-bound (the goal!) ====== -->
  <text x="20" y="55" font-size="12" font-weight="700" fill="#1f5f5b">① GPU-bound (you want this — GPU is the slow link)</text>
  <g transform="translate(20, 65)">
    <text x="-5" y="14" font-size="9" fill="#6b5d4f" text-anchor="end">CPU</text>
    <text x="-5" y="34" font-size="9" fill="#6b5d4f" text-anchor="end">GPU</text>
    <!-- CPU has small bursts (kernel launches) -->
    <rect x="0"   y="3" width="20" height="14" fill="#fff8a8" stroke="#1a1612" stroke-width="0.5"/>
    <rect x="120" y="3" width="20" height="14" fill="#fff8a8" stroke="#1a1612" stroke-width="0.5"/>
    <rect x="240" y="3" width="20" height="14" fill="#fff8a8" stroke="#1a1612" stroke-width="0.5"/>
    <rect x="360" y="3" width="20" height="14" fill="#fff8a8" stroke="#1a1612" stroke-width="0.5"/>
    <rect x="480" y="3" width="20" height="14" fill="#fff8a8" stroke="#1a1612" stroke-width="0.5"/>
    <rect x="600" y="3" width="20" height="14" fill="#fff8a8" stroke="#1a1612" stroke-width="0.5"/>
    <!-- GPU continuous, packed -->
    <rect x="20"  y="23" width="120" height="14" fill="#1f5f5b"/>
    <rect x="140" y="23" width="120" height="14" fill="#1f5f5b"/>
    <rect x="260" y="23" width="120" height="14" fill="#1f5f5b"/>
    <rect x="380" y="23" width="120" height="14" fill="#1f5f5b"/>
    <rect x="500" y="23" width="120" height="14" fill="#1f5f5b"/>
    <rect x="620" y="23" width="80"  height="14" fill="#1f5f5b"/>
  </g>
  <text x="370" y="115" font-size="10" fill="#1f5f5b" text-anchor="middle" font-style="italic">GPU is solid — packed end-to-end. You're using the hardware. Tune kernels next (Part VIII).</text>

  <!-- ====== Panel 2: CPU-bound ====== -->
  <text x="20" y="155" font-size="12" font-weight="700" fill="#c1502e">② CPU-bound (kernel launches and dispatcher overhead dominate)</text>
  <g transform="translate(20, 165)">
    <text x="-5" y="14" font-size="9" fill="#6b5d4f" text-anchor="end">CPU</text>
    <text x="-5" y="34" font-size="9" fill="#6b5d4f" text-anchor="end">GPU</text>
    <!-- CPU continuous (lots of launching) -->
    <rect x="0"   y="3" width="700" height="14" fill="#fcecec" stroke="#c1502e" stroke-width="0.5"/>
    <text x="350" y="13" font-size="10" fill="#c1502e" text-anchor="middle">Python + dispatcher: tons of small ops</text>
    <!-- GPU bursts with idle gaps -->
    <rect x="0"   y="23" width="40"  height="14" fill="#1f5f5b"/>
    <rect x="80"  y="23" width="35"  height="14" fill="#1f5f5b"/>
    <rect x="160" y="23" width="40"  height="14" fill="#1f5f5b"/>
    <rect x="240" y="23" width="35"  height="14" fill="#1f5f5b"/>
    <rect x="320" y="23" width="40"  height="14" fill="#1f5f5b"/>
    <rect x="400" y="23" width="35"  height="14" fill="#1f5f5b"/>
    <rect x="480" y="23" width="40"  height="14" fill="#1f5f5b"/>
    <rect x="560" y="23" width="35"  height="14" fill="#1f5f5b"/>
    <rect x="640" y="23" width="40"  height="14" fill="#1f5f5b"/>
  </g>
  <text x="370" y="215" font-size="10" fill="#c1502e" text-anchor="middle" font-style="italic">GPU has gaps between kernels — small ops or eager-mode overhead. Fix: torch.compile, fuse ops, larger batches.</text>

  <!-- ====== Panel 3: Data-bound ====== -->
  <text x="20" y="255" font-size="12" font-weight="700" fill="#d4a017">③ Data-bound (GPU starves waiting for the loader)</text>
  <g transform="translate(20, 265)">
    <text x="-5" y="14" font-size="9" fill="#6b5d4f" text-anchor="end">CPU</text>
    <text x="-5" y="34" font-size="9" fill="#6b5d4f" text-anchor="end">GPU</text>
    <!-- CPU has long idle blocks (waiting for next batch from worker) -->
    <rect x="0"   y="3" width="20"  height="14" fill="#fff8a8" stroke="#1a1612" stroke-width="0.5"/>
    <rect x="20"  y="3" width="160" height="14" fill="#fff5d8" stroke="#d4a017" stroke-width="0.5" stroke-dasharray="3 2"/>
    <text x="100" y="13" font-size="9" fill="#8c6512" text-anchor="middle" font-style="italic">waiting for batch...</text>
    <rect x="180" y="3" width="20"  height="14" fill="#fff8a8" stroke="#1a1612" stroke-width="0.5"/>
    <rect x="200" y="3" width="160" height="14" fill="#fff5d8" stroke="#d4a017" stroke-width="0.5" stroke-dasharray="3 2"/>
    <text x="280" y="13" font-size="9" fill="#8c6512" text-anchor="middle" font-style="italic">waiting...</text>
    <rect x="360" y="3" width="20"  height="14" fill="#fff8a8" stroke="#1a1612" stroke-width="0.5"/>
    <rect x="380" y="3" width="160" height="14" fill="#fff5d8" stroke="#d4a017" stroke-width="0.5" stroke-dasharray="3 2"/>
    <text x="460" y="13" font-size="9" fill="#8c6512" text-anchor="middle" font-style="italic">waiting...</text>
    <rect x="540" y="3" width="20"  height="14" fill="#fff8a8" stroke="#1a1612" stroke-width="0.5"/>
    <rect x="560" y="3" width="140" height="14" fill="#fff5d8" stroke="#d4a017" stroke-width="0.5" stroke-dasharray="3 2"/>
    <!-- GPU short bursts then long idle -->
    <rect x="20"  y="23" width="60"  height="14" fill="#1f5f5b"/>
    <rect x="200" y="23" width="60"  height="14" fill="#1f5f5b"/>
    <rect x="380" y="23" width="60"  height="14" fill="#1f5f5b"/>
    <rect x="560" y="23" width="60"  height="14" fill="#1f5f5b"/>
  </g>
  <text x="370" y="315" font-size="10" fill="#8c6512" text-anchor="middle" font-style="italic">GPU only runs in short bursts. Workers can't keep up. Fix: more num_workers, faster __getitem__, prefetch_factor.</text>

  <!-- ====== Panel 4: Communication-bound ====== -->
  <text x="20" y="355" font-size="12" font-weight="700" fill="#b85a6c">④ Communication-bound (multi-GPU collectives eat the schedule)</text>
  <g transform="translate(20, 365)">
    <text x="-5" y="14" font-size="9" fill="#6b5d4f" text-anchor="end">CPU</text>
    <text x="-5" y="34" font-size="9" fill="#6b5d4f" text-anchor="end">GPU</text>
    <text x="-5" y="54" font-size="9" fill="#6b5d4f" text-anchor="end">NCCL</text>
    <!-- CPU continuous launching -->
    <rect x="0"   y="3" width="700" height="14" fill="#fcecec" stroke="#c1502e" stroke-width="0.5"/>
    <!-- GPU bursts -->
    <rect x="0"   y="23" width="120" height="14" fill="#1f5f5b"/>
    <rect x="270" y="23" width="120" height="14" fill="#1f5f5b"/>
    <rect x="540" y="23" width="120" height="14" fill="#1f5f5b"/>
    <!-- NCCL collectives (alternating with GPU work) -->
    <rect x="120" y="43" width="150" height="14" fill="#ffd5dc" stroke="#b85a6c" stroke-width="0.5"/>
    <text x="195" y="53" font-size="9" fill="#8c3a55" text-anchor="middle" font-style="italic">all_reduce</text>
    <rect x="390" y="43" width="150" height="14" fill="#ffd5dc" stroke="#b85a6c" stroke-width="0.5"/>
    <text x="465" y="53" font-size="9" fill="#8c3a55" text-anchor="middle" font-style="italic">all_reduce</text>
  </g>
  <text x="370" y="440" font-size="10" fill="#8c3a55" text-anchor="middle" font-style="italic">Collectives serialize compute. Fix: overlap compute+comm (DDP gradient bucketing, FSDP, M16-M17).</text>

  <text x="370" y="465" font-family="'Caveat', cursive" font-size="20" fill="#c1502e" text-anchor="middle">your trace will look like exactly one of these. find the shape, then the fix.</text>
</svg>
</div>

<p>Read each panel:</p>

<ol>
  <li><strong>GPU-bound</strong>: GPU lane is solid, CPU lane shows brief launch bursts. <em>This is the goal.</em> The GPU is the slow link, which means you're using the hardware. The optimization next step is kernel-level: better matmul choice, fused kernels, FlashAttention.</li>
  <li><strong>CPU-bound</strong>: CPU lane is busy, GPU lane has gaps between kernels. Common with very small models, very small batches, or eager mode with lots of small ops. The CPU spends so much time launching kernels that the GPU drains the queue and waits. <em>Fix: <code>torch.compile</code> (M21), op fusion, larger batches</em>.</li>
  <li><strong>Data-bound</strong>: GPU lane has long idle stretches. Workers can't deliver batches fast enough. <em>Fix: more <code>num_workers</code>, faster <code>__getitem__</code>, higher <code>prefetch_factor</code> (M10)</em>. Also check pinned memory + non_blocking transfers (M3).</li>
  <li><strong>Communication-bound</strong>: only relevant in distributed training. NCCL all-reduce ops alternate with compute and dominate the timeline. <em>Fix: overlap compute and communication (DDP gradient bucketing, FSDP — M16-M17)</em>.</li>
</ol>

<p>Most production training spends time in some mix of these. Your job is to look at the trace and identify which mode <em>dominates</em>. That's where the optimization lever lives.</p>

<h2>Reading a profiler summary table</h2>

<p>Before opening the visual trace, the table summary often tells you most of what you need:</p>

<pre><code><span class="fn">print</span>(prof.<span class="fn">key_averages</span>().<span class="fn">table</span>(sort_by=<span class="str">"cuda_time_total"</span>, row_limit=<span class="num">15</span>))</code></pre>

<p>Output looks roughly like:</p>

<div class="ascii">--------------------------------  ------------  ------------  ------------
                            Name      CUDA total      CUDA avg      # calls
--------------------------------  ------------  ------------  ------------
                          aten::mm      342.1 ms      11.4 ms           30
              aten::scaled_dot_pro      298.4 ms       9.9 ms           30
                      aten::linear      121.7 ms       4.0 ms           30
            aten::_log_softmax_back       45.2 ms       1.5 ms           30
                     aten::layer_n       31.8 ms       1.1 ms           29
                     aten::dropout       12.4 ms       0.4 ms           29
--------------------------------  ------------  ------------  ------------
Self CUDA time total: 856.4 ms</div>

<p>Read top-down. The first two ops (matmul and scaled_dot_product_attention) consume 75% of GPU time. <em>That's where to focus</em>. There's no point optimizing the dropout — it's 1.4% of total time, you'll save 12 ms out of 856.</p>

<p>The 80/20 rule applies aggressively in profiling: the slowest 2-3 kernels usually account for 60-80% of GPU time. Optimize those. Ignore the rest until you have to.</p>

<h2>Spotting common pitfalls in a trace</h2>

<h3>The <code>.item()</code> sync</h3>

<p>You met this in M3 — calling <code>.item()</code> on a CUDA tensor forces a CPU-GPU sync, which blocks Python until the GPU's queue drains. In a trace, you'll see a stretch of GPU activity, then a <em>tall vertical line</em> on the CPU lane labelled "cudaStreamSynchronize", then the GPU goes idle while CPU waits.</p>

<p>If you see this on every step, you have a logging bug. Common offenders:</p>

<pre><code><span class="kw">if</span> step % <span class="num">10</span> == <span class="num">0</span>:
    <span class="fn">print</span>(<span class="fn">f"loss: {loss.item():.4f}"</span>)             <span class="com"># sync. fine if rare.</span>

losses.<span class="fn">append</span>(loss.<span class="fn">item</span>())                        <span class="com"># sync EVERY step. bad.</span>

<span class="kw">if</span> torch.<span class="fn">isnan</span>(loss).<span class="fn">item</span>():                       <span class="com"># sync on a Boolean every step. bad.</span>
    <span class="kw">break</span></code></pre>

<p>Fix: log every-N-steps (M11), accumulate scalars on GPU and sync once per epoch, use <code>torch.isnan(loss).any()</code> only at checkpoint boundaries.</p>

<h3>The H2D / D2H copy storm</h3>

<p>If your trace shows lots of small <code>cudaMemcpyAsync</code> ops, you're moving small tensors across the PCIe bus repeatedly. Common causes:</p>

<ul>
  <li>Calling <code>.cuda()</code> inside a hot loop instead of moving once before the loop.</li>
  <li>Computing something on CPU and moving it to GPU per-step (e.g., creating a mask in NumPy and casting to CUDA every batch).</li>
  <li>Logging or tensorboard ops that move metrics back to CPU per step.</li>
</ul>

<p>Each H2D/D2H is small but synchronous-ish (the queue still needs to flush). Death by a thousand cuts.</p>

<h3>Tiny kernels everywhere</h3>

<p>If your trace is full of kernels that are 5-50 microseconds each (rather than milliseconds), you're paying kernel launch overhead. Each kernel launch costs a few microseconds in CPU dispatcher work. With thousands of tiny ops per step, that adds up.</p>

<p>Fixes:</p>

<ol>
  <li><code>torch.compile</code> (M21) fuses many small ops into bigger fused kernels.</li>
  <li>Manual fusion: e.g., <code>x.add_(y).mul_(z)</code> in-place chained, or rewrite as a single tensor expression that avoids intermediates.</li>
  <li>CUDA graphs (M20) record a sequence of kernel launches once and replay the recording — eliminates the per-launch overhead for static graphs.</li>
</ol>

<h2>The diagnosis tree (what to optimize first)</h2>

<p>Once you've identified the bottleneck mode, the optimization lever is usually clear. Here's the decision tree:</p>

<div class="table-wrap">
<table>
<caption>Bottleneck → fix, ranked by impact</caption>
<thead><tr><th>Symptom</th><th>Most likely fix</th><th>Module</th></tr></thead>
<tbody>
<tr><td>GPU lane is solid, GPU-bound on matmul</td><td>Mixed precision (bf16 matmul); flash attention if applicable</td><td>M14, M25</td></tr>
<tr><td>GPU-bound but on small ops</td><td><code>torch.compile</code> for fusion</td><td>M21</td></tr>
<tr><td>CPU lane busy, GPU has gaps</td><td><code>torch.compile</code>; bigger batch; reduce eager-mode overhead</td><td>M21</td></tr>
<tr><td>GPU long idle gaps; loader-side waits</td><td>Tune <code>num_workers</code>, <code>prefetch_factor</code>; profile <code>__getitem__</code></td><td>M10</td></tr>
<tr><td>NCCL collectives serialize compute</td><td>DDP gradient bucketing; FSDP; reduce param count being all-reduced</td><td>M16, M17</td></tr>
<tr><td>OOM (no time issue but won't fit)</td><td>Activation checkpointing; FSDP; reduce batch/seq</td><td>M12, M17</td></tr>
<tr><td>Suspicious tall vertical lines (cudaSync)</td><td>Find and remove the <code>.item()</code> in your hot path</td><td>M3, this module</td></tr>
<tr><td>Small kernels dominate</td><td><code>torch.compile</code> or CUDA graphs</td><td>M20, M21</td></tr>
</tbody>
</table>
</div>

<p>The progression of optimizations to try, in order:</p>

<ol>
  <li><strong>Mixed precision</strong> (M14) — usually 1.5-2× speedup, single-line change.</li>
  <li><strong>Tune the data pipeline</strong> (M10) — make sure GPU isn't starving.</li>
  <li><strong><code>torch.compile</code></strong> (M21) — fuses kernels; another 1.3-2× often.</li>
  <li><strong>FlashAttention</strong> (M25) — for long-sequence transformers, large win.</li>
  <li><strong>Distribute</strong> (M16-M17) — when single-GPU isn't enough.</li>
  <li><strong>Custom kernels</strong> (M22-M24) — last resort, when none of the above gets you there.</li>
</ol>

<p>Most teams stop after step 4 and call it a day. Step 6 is for when you're in the kernel-writing seat.</p>

<h2>Memory profiling (briefly)</h2>

<p>Module 12 covered the memory accounting. The profiler also captures memory events when you pass <code>profile_memory=True</code>:</p>

<pre><code><span class="fn">print</span>(prof.<span class="fn">key_averages</span>().<span class="fn">table</span>(sort_by=<span class="str">"self_cuda_memory_usage"</span>, row_limit=<span class="num">10</span>))</code></pre>

<p>This sorts by per-op memory delta, helping identify the operations that allocate the most. Useful for tracking down activation hotspots before reaching for activation checkpointing.</p>

<p>For deeper memory analysis (allocation timeline, fragmentation visualization), use the dedicated memory snapshot API:</p>

<pre><code>torch.cuda.memory.<span class="fn">_record_memory_history</span>()      <span class="com"># start recording</span>
<span class="com"># ... run training ...</span>
torch.cuda.memory.<span class="fn">_dump_snapshot</span>(<span class="str">'mem.pickle'</span>)
torch.cuda.memory.<span class="fn">_record_memory_history</span>(enabled=<span class="kw">None</span>)  <span class="com"># stop</span></code></pre>

<p>Then visualize with PyTorch's memory snapshot viewer — <code>pytorch.org/memory_viz</code>. Drag the pickle in. You see every allocation with its call stack. Beats guessing.</p>

<h2>NVIDIA Nsight (one paragraph)</h2>

<p>For the deepest kernel-level profiling — looking at SM utilization, memory bandwidth, occupancy, instruction mix — there's <code>nsys</code> (Nsight Systems) and <code>ncu</code> (Nsight Compute). They're NVIDIA's first-party tools. <code>nsys</code> profile gives you a system-level view (your traces + CUDA + NCCL + cuDNN, all timed); <code>ncu</code> profile gives you per-kernel hardware counters (warp efficiency, memory throughput, etc.). Most PyTorch users never need them. If you're <em>writing</em> CUDA kernels (Part VIII), they become essential. <code>torch.profiler</code> + Perfetto is enough for 95% of people.</p>

<h2>Code Magnets: build a clean profiler invocation</h2>

<p>You're profiling a training step to find the bottleneck. Use a schedule that warms up properly, captures shapes and memory, and writes the trace for Perfetto.</p>

<div class="magnets">
<p>Arrange the magnets into a working profiler block. Two are red herrings.</p>

<div class="magnet-pool">
  <span class="magnet">from torch.profiler import profile, ProfilerActivity, schedule</span>
  <span class="magnet">with profile(</span>
  <span class="magnet">    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],</span>
  <span class="magnet">    activities=[ProfilerActivity.CPU],</span>
  <span class="magnet">    schedule=schedule(wait=1, warmup=2, active=3),</span>
  <span class="magnet">    on_trace_ready=torch.profiler.tensorboard_trace_handler('./trace'),</span>
  <span class="magnet">    record_shapes=True, profile_memory=True,</span>
  <span class="magnet">) as prof:</span>
  <span class="magnet">    for step, batch in enumerate(loader):</span>
  <span class="magnet">        train_step(model, opt, batch)</span>
  <span class="magnet">        prof.step()</span>
  <span class="magnet">        if step >= 6: break</span>
  <span class="magnet">    torch.cuda.synchronize()</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">from</span> torch.profiler <span class="kw">import</span> profile, ProfilerActivity, schedule

<span class="kw">with</span> <span class="fn">profile</span>(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=<span class="fn">schedule</span>(wait=<span class="num">1</span>, warmup=<span class="num">2</span>, active=<span class="num">3</span>),
    on_trace_ready=torch.profiler.<span class="fn">tensorboard_trace_handler</span>(<span class="str">'./trace'</span>),
    record_shapes=<span class="kw">True</span>, profile_memory=<span class="kw">True</span>,
) <span class="kw">as</span> prof:
    <span class="kw">for</span> step, batch <span class="kw">in</span> <span class="fn">enumerate</span>(loader):
        <span class="fn">train_step</span>(model, opt, batch)
        prof.<span class="fn">step</span>()
        <span class="kw">if</span> step &gt;= <span class="num">6</span>: <span class="kw">break</span></code></pre>
<p>The traps:</p>
<ul>
  <li><code>activities=[ProfilerActivity.CPU]</code> only — would miss CUDA timing entirely. You almost always want both.</li>
  <li><code>torch.cuda.synchronize()</code> outside the profiler block — unnecessary; the profiler handles its own sync at trace boundaries.</li>
</ul>
<p>Note the magic 6 — with <code>wait=1, warmup=2, active=3</code>, you need at least 6 steps for the profiler to capture all 3 active steps. If you stop earlier, the trace is incomplete.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each profiling concept to its purpose.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Purpose</div>

  <div>torch.cuda.Event</div>
  <div>A. Stops Python until all queued GPU work finishes.</div>

  <div>torch.cuda.synchronize()</div>
  <div>B. Records markers on the GPU command stream for sub-microsecond timing.</div>

  <div>torch.profiler with schedule</div>
  <div>C. Captures CPU + CUDA activity over a windowed range of training steps.</div>

  <div>record_shapes=True</div>
  <div>D. Forces an implicit GPU sync because the result is needed on CPU.</div>

  <div>.item() in hot loop</div>
  <div>E. Annotates each captured op with its input tensor shapes — diagnoses size-related slowness.</div>

  <div>GPU lane has gaps</div>
  <div>F. The data loader can't keep up — workers/prefetch are the lever.</div>

  <div>NCCL bars alternate with GPU work</div>
  <div>G. Communication-bound: collectives serialize compute. DDP/FSDP overlap is the fix.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>torch.cuda.Event</strong> → B<br>
<strong>torch.cuda.synchronize()</strong> → A<br>
<strong>torch.profiler with schedule</strong> → C<br>
<strong>record_shapes=True</strong> → E<br>
<strong>.item() in hot loop</strong> → D<br>
<strong>GPU lane has gaps</strong> → F<br>
<strong>NCCL bars alternate with GPU work</strong> → G
</p>
<p>The mental shortcut: <em>Event for accurate kernel timing, synchronize for "wait until done", profiler+schedule for full-step breakdown, record_shapes for diagnosis, .item() syncs are the silent killer, gaps mean data starvation, NCCL bars mean comm bottleneck</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> Time the matmul <code>x @ w</code> for <code>x</code>, <code>w</code> both <code>(4096, 4096)</code> on GPU using <code>torch.cuda.Event</code>, with proper warmup. Report the median over 50 timed iterations.</p>
<details class="answer"><summary>show answer</summary>
<pre><code>x = torch.<span class="fn">randn</span>(<span class="num">4096</span>, <span class="num">4096</span>, device=<span class="str">'cuda'</span>)
w = torch.<span class="fn">randn</span>(<span class="num">4096</span>, <span class="num">4096</span>, device=<span class="str">'cuda'</span>)

<span class="com"># warmup</span>
<span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(<span class="num">10</span>):
    _ = x @ w
torch.cuda.<span class="fn">synchronize</span>()

times = []
<span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(<span class="num">50</span>):
    s, e = torch.cuda.<span class="fn">Event</span>(enable_timing=<span class="kw">True</span>), torch.cuda.<span class="fn">Event</span>(enable_timing=<span class="kw">True</span>)
    s.<span class="fn">record</span>()
    _ = x @ w
    e.<span class="fn">record</span>()
    torch.cuda.<span class="fn">synchronize</span>()
    times.<span class="fn">append</span>(s.<span class="fn">elapsed_time</span>(e))
times.<span class="fn">sort</span>()
<span class="fn">print</span>(<span class="fn">f"median: {times[25]:.3f} ms"</span>)</code></pre>
<p>On an A100 you'll get something like 0.7-1.0 ms; on an H100 maybe 0.4 ms. Note the warmup loop — the first iteration is often 5-10× slower because cuBLAS picks an algorithm and caches it.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Your training trace shows the GPU lane has roughly equal-sized gaps after every kernel, and the CPU lane is busy throughout. What bottleneck mode are you in, and what's the first thing to try?</p>
<details class="answer"><summary>show answer</summary>
<p>CPU-bound. The CPU is busy launching kernels but each launch's overhead is comparable to or larger than the kernel itself, so the GPU drains the queue and waits. First fix: <code>torch.compile(model)</code>. It fuses multiple eager-mode ops into a single kernel, dramatically reducing launch overhead. Often a 1.3-2× speedup with one line. Second fix: increase batch size if memory allows — bigger kernels make the launch overhead a smaller fraction of work.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> A team's trace shows long idle stretches on both CPU and GPU lanes, with no workers visible. They run with <code>num_workers=0</code>. What's happening?</p>
<details class="answer"><summary>show answer</summary>
<p>With <code>num_workers=0</code>, the DataLoader runs <code>__getitem__</code> in the main process, synchronously, between training steps. So the timeline is: GPU finishes step → main process loads next batch (CPU work, no kernel launches → both lanes idle) → main process kicks off the next forward → GPU starts. The "idle on both lanes" is <em>data loading time on the same thread as the trainer</em>. Fix: <code>num_workers=4</code> minimum. Workers run in parallel with training, and the next batch is ready when the previous step finishes.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> You profile a single training step and the table summary shows <code>aten::mm</code> takes 60% of the time. Is that good or bad?</p>
<details class="answer"><summary>show answer</summary>
<p>It's the closest thing to "good" you can be in this analysis. Matmul is the right thing to spend time on — it's the bulk of useful work in any deep learning model, and matmul kernels (cuBLAS) are some of the most optimized code on the planet. If you've reached the point where matmul is the bottleneck, you've successfully eliminated all the <em>other</em> bottlenecks. Next-level optimizations: mixed precision to use tensor cores (M14), <code>torch.compile</code> to fuse surrounding ops, FlashAttention if you have attention. But "matmul-bound" is largely the goal state.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li><strong><code>time.time()</code> lies on GPU</strong> because kernels are async. For accurate timing, use <code>torch.cuda.Event</code> with <code>elapsed_time</code>, and warm up before measuring.</li>
  <li><strong><code>torch.profiler</code></strong> captures both CPU and CUDA activity. Pass <code>activities=[CPU, CUDA]</code>, set up a <code>schedule(wait, warmup, active, repeat)</code>, and route the output to <code>tensorboard_trace_handler</code> for Perfetto/TensorBoard.</li>
  <li><strong>Useful flags</strong>: <code>record_shapes=True</code>, <code>profile_memory=True</code>, <code>with_stack=True</code>.</li>
  <li><strong>Four bottleneck modes</strong> visible in a trace:
    <ul>
      <li><strong>GPU-bound</strong>: GPU lane solid (the goal — kernels are now the limit).</li>
      <li><strong>CPU-bound</strong>: GPU lane has small gaps; CPU lane busy. Fix: <code>torch.compile</code>, larger batches.</li>
      <li><strong>Data-bound</strong>: GPU lane has long gaps. Fix: more workers, faster <code>__getitem__</code>, prefetch.</li>
      <li><strong>Communication-bound</strong>: NCCL bars alternate with compute. Fix: DDP bucketing, FSDP overlap.</li>
    </ul>
  </li>
  <li><strong>The 80/20 rule</strong>: usually 2-3 kernels account for 60-80% of GPU time. Optimize those; ignore the rest.</li>
  <li><strong>Common pitfalls</strong>: <code>.item()</code> in a hot loop (sync per step), small H2D/D2H copies (death by 1000 cuts), tiny kernels (launch overhead dominates).</li>
  <li><strong>Optimization order</strong>: mixed precision → data pipeline → <code>torch.compile</code> → FlashAttention → distribute → custom kernels. Most teams stop after step 4.</li>
  <li><strong>Memory profiling</strong>: use <code>profile_memory=True</code> for per-op summaries; <code>torch.cuda.memory._record_memory_history</code> + memory_viz for deeper analysis.</li>
  <li><strong>Nsight</strong> (<code>nsys</code>, <code>ncu</code>) for kernel-level hardware counter analysis. Mostly relevant when writing kernels (Part VIII).</li>
  <li>The reflex: when training is slow, <em>profile first, optimize second</em>. Two minutes with the profiler beats two hours of guessing where the cycles went.</li>
</ul>
</div>

<p>That closes the predictive half of Part V. M12 taught you to predict and audit memory; M13 taught you to predict and audit time. Next up: M14 cashes in the mixed-precision recipe that's appeared in every memory and timing discussion. Autocast, GradScaler, the bf16 vs fp16 decision, and the numerics tricks that keep training stable in low precision.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">13</span>
  <span>Profiling: where does the time go?</span>
</div>
"""

emit("13_profiling", "Module 13 — Profiling: where does the time go?", BODY)
