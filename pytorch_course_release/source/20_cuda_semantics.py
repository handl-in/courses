#!/usr/bin/env python3
"""Module 20: CUDA semantics & the caching allocator — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part VII · Module 20</div>
  <h1 class="module-title">CUDA semantics &amp; <em>the caching allocator</em></h1>
  <p class="module-sub">— the async kernel queue under every torch op, why <code>cudaMalloc</code> is too slow to call directly, and how CUDA Graphs let you replay a whole training step in microseconds</p>
</div>

<p>M19 walked you down the dispatcher tower until you hit a CUDA backend kernel. This module zooms into what <em>CUDA itself</em> does once that kernel gets called. Three things that show up everywhere from now on: streams (the queue model), the caching allocator (why <code>x = torch.empty(1024)</code> doesn't actually call <code>cudaMalloc</code>), and CUDA Graphs (the trick for cutting per-launch overhead to nearly zero).</p>

<p>You've felt all three already without naming them. The "<code>time.time()</code> lies because GPU is async" warning from M3 was about streams. The "PyTorch holds onto memory after you free a tensor" oddity from M12 was the caching allocator. The "compile is 2× faster" promise from earlier was, in part, CUDA Graphs in disguise. Now we'll name and understand each.</p>

<div class="keyidea">
PyTorch's CUDA backend is built on three systems. <strong>Streams</strong> are FIFO queues of GPU work — the host queues kernels into a stream and returns immediately; the device drains the stream asynchronously. <strong>The caching allocator</strong> sits between PyTorch and <code>cudaMalloc</code>, recycling freed blocks instead of returning them to the driver — making allocation effectively free. <strong>CUDA Graphs</strong> record a sequence of kernel launches once and replay the whole sequence with a single API call, slashing per-launch overhead from microseconds to nanoseconds for static graphs.
</div>

<h2>One new face</h2>

<div class="character" style="--c: #1f5f5b;">
  <div class="avatar" style="background: #1f5f5b; color: #fff;">⏵⏵</div>
  <div>
    <p class="who">Stream</p>
    <p class="name">"I'm a FIFO queue. The host puts kernels in; the device runs them in order, asynchronously."</p>
    <p class="says">When Python calls <code>z = x @ w</code>, the host doesn't run a matmul. It enqueues a matmul kernel into me — the default stream — and returns. Python is already on the next line while the GPU is still on this one. <em>Within a single stream, work runs in order</em> — a kernel that writes <code>z</code> won't start until prior kernels writing to <code>x</code> or <code>w</code> have finished. <em>Between streams, work can run concurrently</em> — that's how you overlap data copies with compute. <code>torch.cuda.synchronize()</code> drains me, blocking the host until all my queued work completes.</p>
  </div>
</div>

<h2>The async kernel queue</h2>

<p>The single most important fact about CUDA programming: <strong>kernel launches are non-blocking</strong>. When you write <code>y = relu(x)</code>, three things happen.</p>

<ol>
  <li>The host (CPU/Python) queues a relu kernel into the current stream.</li>
  <li>The host immediately returns to Python — the next line of your code runs.</li>
  <li>Sometime later, the GPU pulls the kernel off the stream and executes it. The result lands in <code>y</code>'s storage.</li>
</ol>

<p>This is why <code>time.time()</code> lies (M3): when you measure the wall clock around <code>y = relu(x)</code>, you're measuring how long it took to <em>queue</em> the kernel, not run it. The kernel is still running on the GPU after Python has moved on.</p>

<p>This is also why <code>.item()</code> on a CUDA tensor is so expensive. To return the scalar value to Python, the host must <em>wait</em> for the GPU to finish all queued work that produces this tensor. That's a cudaStreamSynchronize — a hard synchronization point that drains the queue.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 360" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrCS" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">CUDA streams: the host queues, the device drains</text>

  <!-- ====== Single stream ====== -->
  <text x="20" y="55" font-size="12" font-weight="700" fill="#1a1612">Single stream — work runs in order</text>
  <g transform="translate(20, 65)">
    <text x="-5" y="14" font-size="9" fill="#6b5d4f" text-anchor="end">Host</text>
    <text x="-5" y="34" font-size="9" fill="#6b5d4f" text-anchor="end">Stream</text>
    <text x="-5" y="54" font-size="9" fill="#6b5d4f" text-anchor="end">Device</text>
    <!-- Host queues kernels rapidly -->
    <rect x="0"   y="3" width="14" height="14" fill="#fff8a8" stroke="#1a1612" stroke-width="0.5"/>
    <rect x="16"  y="3" width="14" height="14" fill="#fff8a8" stroke="#1a1612" stroke-width="0.5"/>
    <rect x="32"  y="3" width="14" height="14" fill="#fff8a8" stroke="#1a1612" stroke-width="0.5"/>
    <rect x="48"  y="3" width="14" height="14" fill="#fff8a8" stroke="#1a1612" stroke-width="0.5"/>
    <text x="80" y="14" font-size="9" fill="#6b5d4f" font-style="italic">host queues 4 kernels then returns…</text>
    <!-- Stream is the queue itself -->
    <rect x="0"   y="23" width="200" height="14" fill="#ede2cc" stroke="#6b5d4f" stroke-width="1"/>
    <text x="100" y="33" text-anchor="middle" font-size="9" fill="#6b5d4f">[ K1 K2 K3 K4 ]  → device pulls one at a time</text>
    <!-- Device runs them sequentially with delay -->
    <rect x="20"  y="43" width="60" height="14" fill="#1f5f5b"/><text x="50" y="53" text-anchor="middle" font-size="9" fill="#fff">K1</text>
    <rect x="80"  y="43" width="50" height="14" fill="#1f5f5b"/><text x="105" y="53" text-anchor="middle" font-size="9" fill="#fff">K2</text>
    <rect x="130" y="43" width="80" height="14" fill="#1f5f5b"/><text x="170" y="53" text-anchor="middle" font-size="9" fill="#fff">K3</text>
    <rect x="210" y="43" width="60" height="14" fill="#1f5f5b"/><text x="240" y="53" text-anchor="middle" font-size="9" fill="#fff">K4</text>
    <text x="280" y="53" font-size="9" fill="#6b5d4f" font-style="italic">device executes in order</text>
    <text x="285" y="68" font-size="9" fill="#6b5d4f" font-style="italic">host's already done queueing</text>
  </g>

  <!-- ====== Two streams ====== -->
  <text x="20" y="170" font-size="12" font-weight="700" fill="#c1502e">Two streams — work overlaps</text>
  <g transform="translate(20, 180)">
    <text x="-5" y="14" font-size="9" fill="#6b5d4f" text-anchor="end">Compute</text>
    <text x="-5" y="34" font-size="9" fill="#6b5d4f" text-anchor="end">Copy</text>

    <!-- Compute stream: matmul -->
    <rect x="0"   y="3" width="120" height="14" fill="#1f5f5b"/><text x="60" y="13" text-anchor="middle" font-size="9" fill="#fff">matmul (compute)</text>
    <rect x="125" y="3" width="100" height="14" fill="#1f5f5b"/><text x="175" y="13" text-anchor="middle" font-size="9" fill="#fff">matmul</text>
    <rect x="230" y="3" width="120" height="14" fill="#1f5f5b"/><text x="290" y="13" text-anchor="middle" font-size="9" fill="#fff">matmul</text>

    <!-- Copy stream: H2D in parallel -->
    <rect x="0"   y="23" width="80" height="14" fill="#fff8a8" stroke="#1a1612" stroke-width="0.5"/><text x="40" y="33" text-anchor="middle" font-size="9" fill="#1a1612">H2D batch i+1</text>
    <rect x="125" y="23" width="80" height="14" fill="#fff8a8" stroke="#1a1612" stroke-width="0.5"/><text x="165" y="33" text-anchor="middle" font-size="9" fill="#1a1612">H2D batch i+2</text>
    <rect x="250" y="23" width="80" height="14" fill="#fff8a8" stroke="#1a1612" stroke-width="0.5"/><text x="290" y="33" text-anchor="middle" font-size="9" fill="#1a1612">H2D batch i+3</text>

    <text x="450" y="20" font-size="11" fill="#c1502e" font-weight="700">two independent streams</text>
    <text x="450" y="34" font-size="11" fill="#c1502e">→ data copy hides behind compute</text>
  </g>

  <!-- ====== Sync points ====== -->
  <text x="20" y="270" font-size="12" font-weight="700" fill="#1a1612">Sync primitives</text>
  <g transform="translate(20, 280)">
    <rect x="0" y="0" width="220" height="50" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="110" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">torch.cuda.synchronize()</text>
    <text x="110" y="36" text-anchor="middle" font-size="9" fill="#1a1612">host waits for stream to drain</text>

    <rect x="240" y="0" width="220" height="50" fill="#fff5d8" stroke="#d4a017" stroke-width="1.5"/>
    <text x="350" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">stream.wait_event(e)</text>
    <text x="350" y="36" text-anchor="middle" font-size="9" fill="#1a1612">stream waits for event on another stream</text>

    <rect x="480" y="0" width="220" height="50" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="590" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">.item() / .to('cpu')</text>
    <text x="590" y="36" text-anchor="middle" font-size="9" fill="#1a1612">implicit sync — host needs the value</text>
  </g>

  <text x="370" y="350" font-family="'Caveat', cursive" font-size="20" fill="#c1502e" text-anchor="middle">most training uses ONE stream — the default. multi-stream is for explicit overlap.</text>
</svg>
</div>

<p>Read it carefully. The host queues four kernels into the stream nearly instantly — that's why kernel-launch overhead in eager mode is "only" microseconds, not milliseconds. The device then drains the stream at its own pace. By the time the host gets to the fourth Python line, the device might still be on K2.</p>

<p>The lower panel is what makes streams interesting beyond async dispatch: <em>two streams run concurrently on the GPU</em>. Putting H2D copies on a copy stream and matmuls on a compute stream means the next batch's data uploads while the current batch computes. PyTorch's <code>DataLoader(pin_memory=True)</code> + <code>x.to(device, non_blocking=True)</code> exploits this transparently (M3, M10).</p>

<h3>Most training uses one stream</h3>

<p>You'd be forgiven for thinking multi-stream is the norm. It's not. The default stream is what nearly all PyTorch code runs on, and that's fine — within a single stream, work is fully ordered, so you don't need to think about race conditions. Multi-stream is for <em>explicit overlap</em>: copy + compute, computation + collectives in some setups (FSDP uses extra streams internally for prefetching, M17), kernel pipelines in advanced custom code.</p>

<p>If you do need to coordinate across streams, the primitive is the <strong>event</strong>:</p>

<pre><code>copy_stream = torch.cuda.<span class="fn">Stream</span>()
compute_stream = torch.cuda.<span class="fn">current_stream</span>()

<span class="kw">with</span> torch.cuda.<span class="fn">stream</span>(copy_stream):
    x_next = batch.<span class="fn">to</span>(device, non_blocking=<span class="kw">True</span>)
    copy_done = torch.cuda.<span class="fn">Event</span>()
    copy_done.<span class="fn">record</span>()

<span class="com"># Make compute stream wait for the copy</span>
compute_stream.<span class="fn">wait_event</span>(copy_done)
out = <span class="fn">model</span>(x_next)         <span class="com"># runs on compute stream after copy completes</span></code></pre>

<p>The event marks "this point on the copy stream" and the compute stream's <code>wait_event</code> creates a dependency: the next kernel on compute won't start until the copy finished. You're building an explicit DAG of GPU work across streams.</p>

<h2>The caching allocator</h2>

<p>Every time you create a tensor — <code>torch.zeros(1024)</code>, <code>x + y</code>, <code>F.relu(z)</code> — PyTorch needs GPU memory for the result. The naive approach: call <code>cudaMalloc</code> for the bytes, return a pointer, eventually call <code>cudaFree</code> when the tensor's storage is destroyed.</p>

<p>Naive doesn't work. <code>cudaMalloc</code> takes <em>tens of microseconds</em> per call. <code>cudaFree</code> is similar. A single training step might allocate hundreds of intermediate tensors. That's milliseconds of pure allocator overhead per step — bigger than the kernel work in many cases.</p>

<p>Solution: a caching allocator sits between PyTorch and CUDA. <code>cudaMalloc</code> is called rarely, in large chunks. Tensor creates and destroys go through a fast in-process free-list manager.</p>

<h3>Blocks, segments, and the free list</h3>

<p>The allocator keeps memory organized as <strong>segments</strong> — large regions allocated from <code>cudaMalloc</code> (often 2 MB or more). Each segment is split into <strong>blocks</strong>, the units actually handed out as tensors. Small allocations come from "small block" pools (size ≤ 1 MB), large allocations from a separate "large block" pool. Blocks have a free list per size class.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 320" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Caching allocator: blocks within segments</text>

  <!-- A segment, divided into blocks of various sizes -->
  <text x="20" y="55" font-size="12" font-weight="700" fill="#1a1612">Segment 0 (one cudaMalloc call):</text>
  <g transform="translate(20, 65)">
    <rect x="0" y="0" width="700" height="40" fill="none" stroke="#6b5d4f" stroke-width="1.5"/>
    <!-- Used blocks (filled) -->
    <rect x="0"   y="0" width="60"  height="40" fill="#1f5f5b"/>
    <text x="30" y="24" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">used</text>
    <rect x="60"  y="0" width="40"  height="40" fill="#1f5f5b"/>
    <text x="80" y="24" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">used</text>
    <!-- Free block -->
    <rect x="100" y="0" width="120" height="40" fill="#fff5d8" stroke="#d4a017" stroke-width="0.5"/>
    <text x="160" y="24" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">FREE</text>
    <!-- Used -->
    <rect x="220" y="0" width="80"  height="40" fill="#1f5f5b"/>
    <text x="260" y="24" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">used</text>
    <!-- Free -->
    <rect x="300" y="0" width="60"  height="40" fill="#fff5d8" stroke="#d4a017" stroke-width="0.5"/>
    <text x="330" y="24" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">FREE</text>
    <!-- Used -->
    <rect x="360" y="0" width="100" height="40" fill="#1f5f5b"/>
    <text x="410" y="24" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">used</text>
    <rect x="460" y="0" width="60" height="40" fill="#1f5f5b"/>
    <text x="490" y="24" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">used</text>
    <!-- Free at end -->
    <rect x="520" y="0" width="180" height="40" fill="#fff5d8" stroke="#d4a017" stroke-width="0.5"/>
    <text x="610" y="24" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">FREE (large)</text>
  </g>

  <!-- After requesting a 100-byte block -->
  <text x="20" y="145" font-size="12" font-weight="700" fill="#c1502e">A new request for ~100B comes in:</text>
  <g transform="translate(20, 155)">
    <rect x="0" y="0" width="700" height="40" fill="none" stroke="#6b5d4f" stroke-width="1.5"/>
    <rect x="0"   y="0" width="60"  height="40" fill="#1f5f5b"/>
    <rect x="60"  y="0" width="40"  height="40" fill="#1f5f5b"/>
    <!-- This was the 120-byte FREE; now split: 100 used + 20 free -->
    <rect x="100" y="0" width="100" height="40" fill="#c1502e"/>
    <text x="150" y="24" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">NEW (100B)</text>
    <rect x="200" y="0" width="20"  height="40" fill="#fff5d8" stroke="#d4a017" stroke-width="0.5"/>
    <rect x="220" y="0" width="80"  height="40" fill="#1f5f5b"/>
    <rect x="300" y="0" width="60"  height="40" fill="#fff5d8" stroke="#d4a017" stroke-width="0.5"/>
    <rect x="360" y="0" width="100" height="40" fill="#1f5f5b"/>
    <rect x="460" y="0" width="60" height="40" fill="#1f5f5b"/>
    <rect x="520" y="0" width="180" height="40" fill="#fff5d8" stroke="#d4a017" stroke-width="0.5"/>
  </g>
  <text x="280" y="225" font-family="'Caveat', cursive" font-size="18" fill="#c1502e" text-anchor="middle">allocator splits a free block — fast, no cudaMalloc</text>

  <!-- Fragmentation -->
  <text x="20" y="270" font-size="12" font-weight="700" fill="#1a1612">Fragmentation: enough free TOTAL but no contiguous block big enough</text>
  <g transform="translate(20, 280)">
    <rect x="0" y="0" width="700" height="20" fill="none" stroke="#6b5d4f" stroke-width="1"/>
    <rect x="0"   y="0" width="60"  height="20" fill="#1f5f5b"/>
    <rect x="60"  y="0" width="20"  height="20" fill="#fff5d8" stroke="#d4a017" stroke-width="0.5"/>
    <rect x="80"  y="0" width="80"  height="20" fill="#1f5f5b"/>
    <rect x="160" y="0" width="20"  height="20" fill="#fff5d8" stroke="#d4a017" stroke-width="0.5"/>
    <rect x="180" y="0" width="120" height="20" fill="#1f5f5b"/>
    <rect x="300" y="0" width="20"  height="20" fill="#fff5d8" stroke="#d4a017" stroke-width="0.5"/>
    <rect x="320" y="0" width="100" height="20" fill="#1f5f5b"/>
    <rect x="420" y="0" width="20"  height="20" fill="#fff5d8" stroke="#d4a017" stroke-width="0.5"/>
    <rect x="440" y="0" width="180" height="20" fill="#1f5f5b"/>
    <rect x="620" y="0" width="20"  height="20" fill="#fff5d8" stroke="#d4a017" stroke-width="0.5"/>
    <rect x="640" y="0" width="60" height="20" fill="#1f5f5b"/>
  </g>
  <text x="280" y="315" font-family="'Caveat', cursive" font-size="18" fill="#c1502e" text-anchor="middle">100B total free, scattered as 5×20B — request for 100B contiguous: OOM despite "free memory"</text>
</svg>
</div>

<p>Three things this picture explains:</p>

<ol>
  <li><strong>Why allocation is fast</strong>: a tensor create scans the free list for a block of suitable size; if found, splits it. No <code>cudaMalloc</code>. Microseconds-fast.</li>
  <li><strong>Why <code>nvidia-smi</code> shows more memory used than the sum of your tensors</strong> (M12): the allocator holds onto segments even after individual tensor frees. The freed bytes go on the free list; the segment stays.</li>
  <li><strong>Why fragmentation causes "phantom" OOMs</strong>: you might have 5 GB free total, but if it's scattered as a hundred small holes, a request for a contiguous 1 GB block fails. The allocator can't combine non-adjacent blocks.</li>
</ol>

<p>When fragmentation strikes, the allocator can fall back to <code>cudaMalloc</code> for fresh segments — but only if there's free GPU memory available beyond what's in segments. If the allocator already owns most of GPU memory, fragmentation = OOM.</p>

<h3>Tools and knobs</h3>

<pre><code><span class="com"># Inspect</span>
torch.cuda.<span class="fn">memory_allocated</span>()      <span class="com"># bytes used by live tensors</span>
torch.cuda.<span class="fn">memory_reserved</span>()       <span class="com"># bytes the allocator has from cudaMalloc</span>
torch.cuda.<span class="fn">memory_summary</span>()        <span class="com"># detailed table — allocations by size class, fragmentation, etc.</span>

<span class="com"># Force the allocator to release segments back to the driver</span>
torch.cuda.<span class="fn">empty_cache</span>()           <span class="com"># slow! only do at boundaries (e.g., before validation)</span>

<span class="com"># Cap how much GPU memory PyTorch can use (useful for shared GPUs)</span>
torch.cuda.<span class="fn">set_per_process_memory_fraction</span>(<span class="num">0.5</span>)    <span class="com"># 50% of total</span></code></pre>

<p>The most useful trick when fragmentation is hurting you: <code>expandable_segments</code>. Set the env var <code>PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True</code> before launching, and the allocator uses <code>cudaMallocAsync</code> with extending segments — much less fragmentation in practice. It's not yet the default because it has limitations on older GPUs and shared-GPU setups, but for modern training (Ampere+ on dedicated GPUs) it's nearly always a win.</p>

<pre><code>export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun ... train.py</code></pre>

<p>Other useful keys for that env var: <code>max_split_size_mb:N</code> (cap how big a free block the allocator will split — sometimes helps with very large activations), <code>garbage_collection_threshold:0.8</code> (try to free memory back to the OS at 80% utilization).</p>

<h2>Memory snapshots: the visual debugger</h2>

<p>From M13: the memory snapshot tool. Now you understand what it's showing.</p>

<pre><code>torch.cuda.memory.<span class="fn">_record_memory_history</span>()
<span class="com"># ... run training step ...</span>
torch.cuda.memory.<span class="fn">_dump_snapshot</span>(<span class="str">'mem.pickle'</span>)
torch.cuda.memory.<span class="fn">_record_memory_history</span>(enabled=<span class="kw">None</span>)</code></pre>

<p>Drag the pickle into <a href="https://pytorch.org/memory_viz">pytorch.org/memory_viz</a>. You see segments on the y-axis, time on the x-axis, every allocation as a colored block tied to a Python call stack. Fragmentation jumps out — the segment fills with little holes that linger across many steps. The allocations that <em>cause</em> peak memory are visible (the tall stack at peak time tells you what's live).</p>

<p>Use it when memory is mysterious: the size, lifetime, and origin of each allocation are visible. No more "but my tensors only add up to 30 GB and I'm getting OOM at 60."</p>

<h2>CUDA Graphs: the dispatcher killer</h2>

<p>Recall from M19: per-op dispatcher overhead in eager mode is roughly 5-15 microseconds. For a transformer with thousands of small ops per step, that's tens of milliseconds of pure CPU overhead per step. The "CPU-bound" trace shape from M13.</p>

<p>One fix is <code>torch.compile</code> (M21) — compile the trace into fused kernels, eliminating the per-op tower for the compiled region. But there's another, more surgical fix: <strong>CUDA Graphs</strong>.</p>

<p>The idea: a CUDA Graph is a recorded sequence of kernel launches with their dependencies. Once recorded, the entire graph can be replayed with a single API call (<code>cudaGraphLaunch</code>), bypassing per-launch overhead entirely. The kernels themselves still run; the host-side launch-and-dispatch machinery doesn't.</p>

<pre><code><span class="com"># Warm up — first few iterations populate caches and finalize shapes</span>
<span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(<span class="num">3</span>):
    <span class="fn">train_step</span>(model, opt, batch)
torch.cuda.<span class="fn">synchronize</span>()

<span class="com"># Capture</span>
g = torch.cuda.<span class="fn">CUDAGraph</span>()
opt.<span class="fn">zero_grad</span>(set_to_none=<span class="kw">True</span>)
<span class="kw">with</span> torch.cuda.<span class="fn">graph</span>(g):
    static_loss = <span class="fn">criterion</span>(model(static_x), static_y)
    static_loss.<span class="fn">backward</span>()
    opt.<span class="fn">step</span>()

<span class="com"># Replay — for every subsequent step</span>
<span class="kw">for</span> step <span class="kw">in</span> <span class="fn">range</span>(num_steps):
    static_x.<span class="fn">copy_</span>(real_batch_x)             <span class="com"># reuse the captured input tensor's storage</span>
    static_y.<span class="fn">copy_</span>(real_batch_y)
    g.<span class="fn">replay</span>()                              <span class="com"># one API call, runs the entire graph</span>
    <span class="com"># static_loss now contains this step's loss</span></code></pre>

<p>The catch: <strong>the captured kernels operate on fixed memory addresses</strong>. You can't pass new tensors each step — you have to reuse the same input/output buffers, copying new data into them. Variable shapes break capture: if your batch size changes, the capture is invalidated. So CUDA Graphs work beautifully for static training (LLM pretraining, where every step has identical shapes) and poorly for dynamic workloads (variable-length input, conditional branches).</p>

<p>How fast? On a typical transformer step, replaying a graph can be 2-5× faster than eager-mode launches — the entire CPU side of the timeline collapses to almost nothing. Per-launch overhead drops from microseconds to nanoseconds.</p>

<p>You don't have to write the capture-and-replay machinery yourself. <code>torch.compile(model, mode="reduce-overhead")</code> uses CUDA Graphs internally for static regions of your model. M21 covers the integration.</p>

<div class="warn">
<strong>Capturing a graph requires a clean stream and stable shapes.</strong> Capture happens on a non-default stream; any work outstanding on the default stream at capture time will not be captured (and might race). Common pitfalls: <code>.item()</code> calls inside the captured region (these need a host-device sync, which graphs don't allow), Python control flow that varies between captures, or in-place ops on tensors that have other live references. Errors from CUDA Graphs are notoriously cryptic — when they fail, simplify the captured region until it works, then add things back.
</div>

<h2>Topology and NUMA (briefly)</h2>

<p>One last system concern. On multi-GPU machines, GPUs are not all equally close to each other. An 8-GPU node typically has GPUs grouped in pairs sharing a PCIe switch and NVSwitch fabric across the pairs. Cross-pair bandwidth is faster than cross-node, but slower than within-pair.</p>

<p>NCCL (M15) is topology-aware — it queries the system at <code>init_process_group</code> time and chooses ring/tree algorithms that minimize cross-link traffic. You usually don't think about this. The two cases where it matters:</p>

<ul>
  <li><strong>Pinned memory and NUMA</strong>: on multi-socket nodes, pinned host memory should be allocated near the GPU that will use it. PyTorch handles this automatically via <code>torch.cuda.set_device</code> + standard pinning. If you're using custom pinning paths, you might need to bind the host thread to a NUMA node first (<code>numactl --cpunodebind=0</code>).</li>
  <li><strong>Process placement</strong>: <code>torchrun</code> places one process per GPU, in order. If GPU 0 and GPU 1 share a PCIe switch, processes 0 and 1 are also "close" — fine. On unusual topologies, look at <code>nvidia-smi topo -m</code> to confirm.</li>
</ul>

<p>For most training, you don't tune topology. NCCL handles it. The reflex is: if collective performance is mysteriously slow despite a fast-looking network on paper, run <code>NCCL_DEBUG=INFO</code> and check what topology NCCL is using.</p>

<div class="ndq">
<h4>About CUDA semantics</h4>

<p class="q">Why is the default stream "the default" — what's special about it?</p>
<p class="a">Historically, CUDA had a "legacy default stream" that was implicitly synchronizing — every other stream had to wait at sync points with it. Modern PyTorch uses <em>per-thread default streams</em> that are non-synchronizing (each thread has its own default that runs concurrently with other threads' work). For most users it's just "the stream your work goes on if you don't explicitly create one."</p>

<p class="q">When should I call <code>torch.cuda.empty_cache()</code>?</p>
<p class="a">Almost never in normal training. The caching allocator's whole point is to <em>not</em> return memory to the driver — calling <code>empty_cache</code> defeats it, and any subsequent allocation pays the <code>cudaMalloc</code> cost. Two exceptions: (1) before switching to evaluation/validation, where memory layout differs significantly, you may benefit from giving the allocator a clean slate. (2) Before allocating one giant tensor that requires a contiguous block, after a fragmentation-heavy phase. Otherwise, leave it alone.</p>

<p class="q">My step has a sync point I can't find — backward is mysteriously slow. Where do I look?</p>
<p class="a">Common culprits: a <code>.item()</code> on a CUDA tensor (logs, conditionals), <code>print()</code>ing a CUDA tensor (forces a D2H copy and sync), an old GAN-style <code>if loss &lt; threshold: break</code> pattern, calling <code>.cpu()</code> on intermediate values for logging. Profile (M13) and look for <code>cudaStreamSynchronize</code> events on the CPU lane during backward — the ones <em>not</em> at step boundaries are bugs.</p>

<p class="q">CUDA Graphs sound great. Why aren't they always on?</p>
<p class="a">Two reasons. (1) They demand stable shapes — variable-length input, dynamic batching, or conditional model paths break capture. (2) They demand stable storage — the captured kernels write to specific addresses, so you can't pass new tensors per step. For LLM pretraining (fixed seqlen, fixed batch), Graphs are nearly free. For inference with variable-length user prompts, they need careful work (batched prefill with bucket sizes, etc.). <code>torch.compile(mode="reduce-overhead")</code> applies Graphs where it's safe and falls back where it isn't.</p>

<p class="q">What's the deal with <code>cudaMallocAsync</code> and "expandable segments"?</p>
<p class="a"><code>cudaMallocAsync</code> is a newer CUDA allocator API that supports stream-aware allocation and can grow segments dynamically (instead of pre-fixed segment sizes). PyTorch's "expandable_segments:True" mode uses it. Less fragmentation in practice — segments grow into available memory rather than being chosen up front. Recommended on Ampere+ for dedicated GPUs. Doesn't yet work in every shared-GPU configuration, which is why it's not default.</p>
</div>

<h2>Code Magnets: capture and replay a CUDA Graph</h2>

<p>You're capturing a static training step with a CUDA Graph. Three magnets are wrong.</p>

<div class="magnets">
<p>Arrange the magnets into a working capture-and-replay.</p>

<div class="magnet-pool">
  <span class="magnet">for _ in range(3): train_step(model, opt, batch)</span>
  <span class="magnet">torch.cuda.synchronize()</span>
  <span class="magnet">g = torch.cuda.CUDAGraph()</span>
  <span class="magnet">opt.zero_grad(set_to_none=True)</span>
  <span class="magnet">with torch.cuda.graph(g):</span>
  <span class="magnet">    static_loss = criterion(model(static_x), static_y)</span>
  <span class="magnet">    static_loss.backward()</span>
  <span class="magnet">    opt.step()</span>
  <span class="magnet">    print(f"loss: {static_loss.item()}")</span>
  <span class="magnet">for step in range(num_steps):</span>
  <span class="magnet">    static_x.copy_(real_x); static_y.copy_(real_y)</span>
  <span class="magnet">    g.replay()</span>
  <span class="magnet">    static_x = real_x; static_y = real_y</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(<span class="num">3</span>): <span class="fn">train_step</span>(model, opt, batch)
torch.cuda.<span class="fn">synchronize</span>()
g = torch.cuda.<span class="fn">CUDAGraph</span>()
opt.<span class="fn">zero_grad</span>(set_to_none=<span class="kw">True</span>)
<span class="kw">with</span> torch.cuda.<span class="fn">graph</span>(g):
    static_loss = <span class="fn">criterion</span>(<span class="fn">model</span>(static_x), static_y)
    static_loss.<span class="fn">backward</span>()
    opt.<span class="fn">step</span>()
<span class="kw">for</span> step <span class="kw">in</span> <span class="fn">range</span>(num_steps):
    static_x.<span class="fn">copy_</span>(real_x); static_y.<span class="fn">copy_</span>(real_y)
    g.<span class="fn">replay</span>()</code></pre>
<p>The traps:</p>
<ul>
  <li><code>print(f"loss: {static_loss.item()}")</code> <em>inside</em> the capture: <code>.item()</code> requires a host-device sync, which CUDA Graphs forbid. Capture would error or produce a graph that fails to replay.</li>
  <li><code>static_x = real_x; static_y = real_y</code>: rebinding the names breaks the capture. The graph holds <em>storage pointers</em>, not Python references — you must copy data <em>into</em> the captured tensors via <code>copy_</code>, not reassign.</li>
</ul>
<p>The shape: <strong>warm up → sync → capture (no syncing ops inside) → replay with copy_-into-static-buffers</strong>. After <code>g.replay()</code>, <code>static_loss</code> holds the new step's loss; you can read it later (after a sync) without affecting capture.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each CUDA-semantics concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>CUDA stream</div>
  <div>A. FIFO queue of GPU work; ops within one stream run in order.</div>

  <div>torch.cuda.synchronize()</div>
  <div>B. Block the host until all queued GPU work in the current stream completes.</div>

  <div>Caching allocator</div>
  <div>C. Layer between PyTorch and cudaMalloc; recycles freed blocks for fast tensor allocation.</div>

  <div>Segment</div>
  <div>D. A large region from one cudaMalloc, split into many tensor-sized blocks.</div>

  <div>memory_reserved &gt; memory_allocated</div>
  <div>E. Allocator holds segments beyond live tensors — normal, not a leak.</div>

  <div>Expandable segments</div>
  <div>F. Newer mode using cudaMallocAsync; segments grow dynamically, less fragmentation.</div>

  <div>CUDA Graph</div>
  <div>G. Recorded kernel sequence replayed with a single API call — kills launch overhead.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>CUDA stream</strong> → A<br>
<strong>torch.cuda.synchronize()</strong> → B<br>
<strong>Caching allocator</strong> → C<br>
<strong>Segment</strong> → D<br>
<strong>memory_reserved &gt; memory_allocated</strong> → E<br>
<strong>Expandable segments</strong> → F<br>
<strong>CUDA Graph</strong> → G
</p>
<p>The mental shortcut: <em>stream is the queue, sync drains it, allocator recycles, segment = one cudaMalloc, reserved = held bytes incl free blocks, expandable_segments shrinks fragmentation, Graphs replay launches</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team's training shows <code>memory_allocated() = 28 GB</code> but <code>memory_reserved() = 78 GB</code> on an 80 GB H100 — barely room to grow. They expect to be at 28 GB. Is this a leak?</p>
<details class="answer"><summary>show answer</summary>
<p>Almost certainly not. <code>memory_reserved</code> is what the allocator holds via <code>cudaMalloc</code>; <code>memory_allocated</code> is what's bound to live tensors. The 50 GB gap is the allocator's free list — segments that previously held tensors but were freed; the allocator hangs onto them for fast reuse. This is normal and usually fine. If they're worried about headroom, two moves: (1) try <code>PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True</code> — much less fragmentation in practice. (2) Before any phase requiring a big contiguous allocation (loading a checkpoint, switching to validation), call <code>empty_cache()</code> to release segments back. But "reserved &gt; allocated" by itself is not a leak — it's the design.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Why does <code>x.to('cpu')</code> implicitly call <code>cudaStreamSynchronize</code>?</p>
<details class="answer"><summary>show answer</summary>
<p>Because the host needs the actual <em>bytes</em>. Any kernel queued on the stream that writes <code>x</code> hasn't necessarily finished by the time you call <code>.to('cpu')</code>. To return correct data, the framework must wait for those kernels to complete before issuing the D2H copy. That's a hard sync point. Same logic for <code>.item()</code>, <code>.tolist()</code>, <code>.numpy()</code>, and any path that returns the values to Python. <em>This is why these calls are sneaky: they don't look expensive, but they drain your queue.</em></p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> A team captures a CUDA Graph for their training step. It works for the first step but errors on the second. Why might that be?</p>
<details class="answer"><summary>show answer</summary>
<p>Most likely: shape variance. CUDA Graphs are sensitive to anything that changes between captures. Check (a) batch size — is the dataloader producing variable-size last batches? (b) sequence length — is padding consistent or per-batch? (c) any conditional that runs differently (e.g., logging or saving every N steps causing different ops to be queued). Quick fix: drop the last partial batch (<code>drop_last=True</code> in DataLoader, M10) and ensure consistent padding. If the second-step error is more cryptic ("graph cannot be modified"), suspect a tensor that's growing — typically an accumulator or a per-step list. CUDA Graphs require strictly identical kernel sequences and tensor identities each replay.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> When would you intentionally use multiple CUDA streams in PyTorch (beyond the default)?</p>
<details class="answer"><summary>show answer</summary>
<p>Three legitimate cases:</p>
<ol>
  <li><strong>Compute/copy overlap</strong>: put data H2D copies on a copy stream so the next batch uploads while the current batch computes. PyTorch's <code>pin_memory=True</code> + <code>non_blocking=True</code> handles this automatically; you don't write the streams yourself.</li>
  <li><strong>Compute/communication overlap in distributed training</strong>: FSDP (M17) uses extra streams internally to prefetch parameter all-gathers concurrently with the previous layer's compute. DDP (M16) bucketing implicitly relies on this for backward+all-reduce overlap.</li>
  <li><strong>Custom pipelines</strong>: when you have multiple independent ops that don't share data and you want them concurrent (rare in user-level code; common in custom kernel libraries).</li>
</ol>
<p>For ordinary training-loop code: don't manually create streams. The default works, and PyTorch's higher-level features manage extra streams for you when it actually helps.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li><strong>CUDA streams</strong> are FIFO queues. Host queues kernels (fast); device drains and runs them (async). Within a stream, ops are ordered. Across streams, they can be concurrent.</li>
  <li><strong>Most training uses one stream</strong>, the default — fine. Multi-stream is for explicit overlap (copies/compute, communication/compute).</li>
  <li><strong>Sync primitives</strong>: <code>torch.cuda.synchronize()</code> drains the stream from the host. <code>.item()</code>, <code>.cpu()</code>, <code>.numpy()</code> implicitly sync because they need the values on the host. Avoid these in hot loops.</li>
  <li><strong>The caching allocator</strong> sits between PyTorch and <code>cudaMalloc</code>. Tensors are carved from pre-allocated <strong>segments</strong> as <strong>blocks</strong>. Allocation is microseconds-fast.</li>
  <li><code>memory_allocated()</code> = bytes in live tensors. <code>memory_reserved()</code> = bytes the allocator holds (live + free). Reserved &gt; allocated is normal, not a leak.</li>
  <li><strong>Fragmentation</strong> happens when free memory is scattered across non-contiguous blocks. Causes "phantom OOMs" — enough free total, no contiguous block of the right size.</li>
  <li><strong>Fixes</strong>: <code>PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True</code> (Ampere+, dedicated GPUs); <code>empty_cache()</code> at major phase transitions (rarely useful otherwise); <code>set_per_process_memory_fraction</code> for shared GPUs.</li>
  <li><strong>Memory snapshot tool</strong> visualizes every allocation with its call stack. Use when memory is mysterious.</li>
  <li><strong>CUDA Graphs</strong>: record a kernel sequence once, replay with one API call. Bypasses per-launch overhead. Best for static-shape workloads (LLM pretraining). <code>torch.compile(mode="reduce-overhead")</code> uses them automatically where safe.</li>
  <li><strong>Graph rules</strong>: stable shapes, stable tensor identities, no host-device syncs (no <code>.item()</code>) inside capture. Copy data into pre-captured buffers; don't rebind names.</li>
  <li><strong>Topology / NUMA</strong>: NCCL handles topology-aware routing automatically. <code>nvidia-smi topo -m</code> shows physical layout if needed.</li>
  <li>The reflex: when something seems mysteriously slow on GPU, ask <em>which stream is busy, who is waiting, and is anyone forcing a sync that shouldn't be there?</em> Profile (M13) and look at the CPU lane for <code>cudaStreamSynchronize</code> events.</li>
</ul>
</div>

<p>Module 21 takes the dispatcher tower (M19) and the CUDA semantics (this module) and shows how <code>torch.compile</code> collapses both. Dynamo traces Python into FX graphs; AOTAutograd handles forward + backward together; Inductor code-generates fused C++/Triton kernels; <code>mode="reduce-overhead"</code> wires up CUDA Graphs. The full system that makes "slow eager → fast compiled" possible.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">20</span>
  <span>CUDA semantics &amp; the caching allocator</span>
</div>
"""

emit("20_cuda_semantics", "Module 20 — CUDA semantics & the caching allocator", BODY)
