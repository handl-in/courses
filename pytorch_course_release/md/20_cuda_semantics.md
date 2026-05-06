# Module 20 — CUDA semantics & the caching allocator

# CUDA semantics & _the caching allocator_

_Part VII · Module 20_

— the async kernel queue under every torch op, why `cudaMalloc` is too slow to call directly, and how CUDA Graphs let you replay a whole training step in microseconds

\--- 

M19 walked you down the dispatcher tower until you hit a CUDA backend kernel. This module zooms into what _CUDA itself_ does once that kernel gets called. Three things that show up everywhere from now on: streams (the queue model), the caching allocator (why `x = torch.empty(1024)` doesn't actually call `cudaMalloc`), and CUDA Graphs (the trick for cutting per-launch overhead to nearly zero).

You've felt all three already without naming them. The "`time.time()` lies because GPU is async" warning from M3 was about streams. The "PyTorch holds onto memory after you free a tensor" oddity from M12 was the caching allocator. The "compile is 2× faster" promise from earlier was, in part, CUDA Graphs in disguise. Now we'll name and understand each.

> **★ KEY IDEA**  
>  PyTorch's CUDA backend is built on three systems. **Streams** are FIFO queues of GPU work — the host queues kernels into a stream and returns immediately; the device drains the stream asynchronously. **The caching allocator** sits between PyTorch and `cudaMalloc`, recycling freed blocks instead of returning them to the driver — making allocation effectively free. **CUDA Graphs** record a sequence of kernel launches once and replay the whole sequence with a single API call, slashing per-launch overhead from microseconds to nanoseconds for static graphs. 

## One new face

⏵⏵

Stream

"I'm a FIFO queue. The host puts kernels in; the device runs them in order, asynchronously."

When Python calls `z = x @ w`, the host doesn't run a matmul. It enqueues a matmul kernel into me — the default stream — and returns. Python is already on the next line while the GPU is still on this one. _Within a single stream, work runs in order_ — a kernel that writes `z` won't start until prior kernels writing to `x` or `w` have finished. _Between streams, work can run concurrently_ — that's how you overlap data copies with compute. `torch.cuda.synchronize()` drains me, blocking the host until all my queued work completes.

## The async kernel queue

The single most important fact about CUDA programming: **kernel launches are non-blocking**. When you write `y = relu(x)`, three things happen.

  1. The host (CPU/Python) queues a relu kernel into the current stream.
  2. The host immediately returns to Python — the next line of your code runs.
  3. Sometime later, the GPU pulls the kernel off the stream and executes it. The result lands in `y`'s storage.

This is why `time.time()` lies (M3): when you measure the wall clock around `y = relu(x)`, you're measuring how long it took to _queue_ the kernel, not run it. The kernel is still running on the GPU after Python has moved on.

This is also why `.item()` on a CUDA tensor is so expensive. To return the scalar value to Python, the host must _wait_ for the GPU to finish all queued work that produces this tensor. That's a cudaStreamSynchronize — a hard synchronization point that drains the queue.

CUDA streams: the host queues, the device drains Single stream — work runs in order Host Stream Device host queues 4 kernels then returns… [ K1 K2 K3 K4 ] → device pulls one at a time K1 K2 K3 K4 device executes in order host's already done queueing Two streams — work overlaps Compute Copy matmul (compute) matmul matmul H2D batch i+1 H2D batch i+2 H2D batch i+3 two independent streams → data copy hides behind compute Sync primitives torch.cuda.synchronize() host waits for stream to drain stream.wait_event(e) stream waits for event on another stream .item() / .to('cpu') implicit sync — host needs the value most training uses ONE stream — the default. multi-stream is for explicit overlap.

Read it carefully. The host queues four kernels into the stream nearly instantly — that's why kernel-launch overhead in eager mode is "only" microseconds, not milliseconds. The device then drains the stream at its own pace. By the time the host gets to the fourth Python line, the device might still be on K2.

The lower panel is what makes streams interesting beyond async dispatch: _two streams run concurrently on the GPU_. Putting H2D copies on a copy stream and matmuls on a compute stream means the next batch's data uploads while the current batch computes. PyTorch's `DataLoader(pin_memory=True)` \+ `x.to(device, non_blocking=True)` exploits this transparently (M3, M10).

### Most training uses one stream

You'd be forgiven for thinking multi-stream is the norm. It's not. The default stream is what nearly all PyTorch code runs on, and that's fine — within a single stream, work is fully ordered, so you don't need to think about race conditions. Multi-stream is for _explicit overlap_ : copy + compute, computation + collectives in some setups (FSDP uses extra streams internally for prefetching, M17), kernel pipelines in advanced custom code.

If you do need to coordinate across streams, the primitive is the **event** :
    
    
    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.current_stream()
    
    with torch.cuda.stream(copy_stream):
        x_next = batch.to(device, non_blocking=True)
        copy_done = torch.cuda.Event()
        copy_done.record()
    
    # Make compute stream wait for the copy
    compute_stream.wait_event(copy_done)
    out = model(x_next)         # runs on compute stream after copy completes

The event marks "this point on the copy stream" and the compute stream's `wait_event` creates a dependency: the next kernel on compute won't start until the copy finished. You're building an explicit DAG of GPU work across streams.

## The caching allocator

Every time you create a tensor — `torch.zeros(1024)`, `x + y`, `F.relu(z)` — PyTorch needs GPU memory for the result. The naive approach: call `cudaMalloc` for the bytes, return a pointer, eventually call `cudaFree` when the tensor's storage is destroyed.

Naive doesn't work. `cudaMalloc` takes _tens of microseconds_ per call. `cudaFree` is similar. A single training step might allocate hundreds of intermediate tensors. That's milliseconds of pure allocator overhead per step — bigger than the kernel work in many cases.

Solution: a caching allocator sits between PyTorch and CUDA. `cudaMalloc` is called rarely, in large chunks. Tensor creates and destroys go through a fast in-process free-list manager.

### Blocks, segments, and the free list

The allocator keeps memory organized as **segments** — large regions allocated from `cudaMalloc` (often 2 MB or more). Each segment is split into **blocks** , the units actually handed out as tensors. Small allocations come from "small block" pools (size ≤ 1 MB), large allocations from a separate "large block" pool. Blocks have a free list per size class.

Caching allocator: blocks within segments Segment 0 (one cudaMalloc call): used used FREE used FREE used used FREE (large) A new request for ~100B comes in: NEW (100B) allocator splits a free block — fast, no cudaMalloc Fragmentation: enough free TOTAL but no contiguous block big enough 100B total free, scattered as 5×20B — request for 100B contiguous: OOM despite "free memory"

Three things this picture explains:

  1. **Why allocation is fast** : a tensor create scans the free list for a block of suitable size; if found, splits it. No `cudaMalloc`. Microseconds-fast.
  2. **Why`nvidia-smi` shows more memory used than the sum of your tensors** (M12): the allocator holds onto segments even after individual tensor frees. The freed bytes go on the free list; the segment stays.
  3. **Why fragmentation causes "phantom" OOMs** : you might have 5 GB free total, but if it's scattered as a hundred small holes, a request for a contiguous 1 GB block fails. The allocator can't combine non-adjacent blocks.

When fragmentation strikes, the allocator can fall back to `cudaMalloc` for fresh segments — but only if there's free GPU memory available beyond what's in segments. If the allocator already owns most of GPU memory, fragmentation = OOM.

### Tools and knobs
    
    
    # Inspect
    torch.cuda.memory_allocated()      # bytes used by live tensors
    torch.cuda.memory_reserved()       # bytes the allocator has from cudaMalloc
    torch.cuda.memory_summary()        # detailed table — allocations by size class, fragmentation, etc.
    
    # Force the allocator to release segments back to the driver
    torch.cuda.empty_cache()           # slow! only do at boundaries (e.g., before validation)
    
    # Cap how much GPU memory PyTorch can use (useful for shared GPUs)
    torch.cuda.set_per_process_memory_fraction(0.5)    # 50% of total

The most useful trick when fragmentation is hurting you: `expandable_segments`. Set the env var `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before launching, and the allocator uses `cudaMallocAsync` with extending segments — much less fragmentation in practice. It's not yet the default because it has limitations on older GPUs and shared-GPU setups, but for modern training (Ampere+ on dedicated GPUs) it's nearly always a win.
    
    
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    torchrun ... train.py

Other useful keys for that env var: `max_split_size_mb:N` (cap how big a free block the allocator will split — sometimes helps with very large activations), `garbage_collection_threshold:0.8` (try to free memory back to the OS at 80% utilization).

## Memory snapshots: the visual debugger

From M13: the memory snapshot tool. Now you understand what it's showing.
    
    
    torch.cuda.memory._record_memory_history()
    # ... run training step ...
    torch.cuda.memory._dump_snapshot('mem.pickle')
    torch.cuda.memory._record_memory_history(enabled=None)

Drag the pickle into [pytorch.org/memory_viz](https://pytorch.org/memory_viz). You see segments on the y-axis, time on the x-axis, every allocation as a colored block tied to a Python call stack. Fragmentation jumps out — the segment fills with little holes that linger across many steps. The allocations that _cause_ peak memory are visible (the tall stack at peak time tells you what's live).

Use it when memory is mysterious: the size, lifetime, and origin of each allocation are visible. No more "but my tensors only add up to 30 GB and I'm getting OOM at 60."

## CUDA Graphs: the dispatcher killer

Recall from M19: per-op dispatcher overhead in eager mode is roughly 5-15 microseconds. For a transformer with thousands of small ops per step, that's tens of milliseconds of pure CPU overhead per step. The "CPU-bound" trace shape from M13.

One fix is `torch.compile` (M21) — compile the trace into fused kernels, eliminating the per-op tower for the compiled region. But there's another, more surgical fix: **CUDA Graphs**.

The idea: a CUDA Graph is a recorded sequence of kernel launches with their dependencies. Once recorded, the entire graph can be replayed with a single API call (`cudaGraphLaunch`), bypassing per-launch overhead entirely. The kernels themselves still run; the host-side launch-and-dispatch machinery doesn't.
    
    
    # Warm up — first few iterations populate caches and finalize shapes
    for _ in range(3):
        train_step(model, opt, batch)
    torch.cuda.synchronize()
    
    # Capture
    g = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        static_loss = criterion(model(static_x), static_y)
        static_loss.backward()
        opt.step()
    
    # Replay — for every subsequent step
    for step in range(num_steps):
        static_x.copy_(real_batch_x)             # reuse the captured input tensor's storage
        static_y.copy_(real_batch_y)
        g.replay()                              # one API call, runs the entire graph
        # static_loss now contains this step's loss

The catch: **the captured kernels operate on fixed memory addresses**. You can't pass new tensors each step — you have to reuse the same input/output buffers, copying new data into them. Variable shapes break capture: if your batch size changes, the capture is invalidated. So CUDA Graphs work beautifully for static training (LLM pretraining, where every step has identical shapes) and poorly for dynamic workloads (variable-length input, conditional branches).

How fast? On a typical transformer step, replaying a graph can be 2-5× faster than eager-mode launches — the entire CPU side of the timeline collapses to almost nothing. Per-launch overhead drops from microseconds to nanoseconds.

You don't have to write the capture-and-replay machinery yourself. `torch.compile(model, mode="reduce-overhead")` uses CUDA Graphs internally for static regions of your model. M21 covers the integration.

> **⚠ WARNING**  
>  **Capturing a graph requires a clean stream and stable shapes.** Capture happens on a non-default stream; any work outstanding on the default stream at capture time will not be captured (and might race). Common pitfalls: `.item()` calls inside the captured region (these need a host-device sync, which graphs don't allow), Python control flow that varies between captures, or in-place ops on tensors that have other live references. Errors from CUDA Graphs are notoriously cryptic — when they fail, simplify the captured region until it works, then add things back. 

## Topology and NUMA (briefly)

One last system concern. On multi-GPU machines, GPUs are not all equally close to each other. An 8-GPU node typically has GPUs grouped in pairs sharing a PCIe switch and NVSwitch fabric across the pairs. Cross-pair bandwidth is faster than cross-node, but slower than within-pair.

NCCL (M15) is topology-aware — it queries the system at `init_process_group` time and chooses ring/tree algorithms that minimize cross-link traffic. You usually don't think about this. The two cases where it matters:

  * **Pinned memory and NUMA** : on multi-socket nodes, pinned host memory should be allocated near the GPU that will use it. PyTorch handles this automatically via `torch.cuda.set_device` \+ standard pinning. If you're using custom pinning paths, you might need to bind the host thread to a NUMA node first (`numactl --cpunodebind=0`).
  * **Process placement** : `torchrun` places one process per GPU, in order. If GPU 0 and GPU 1 share a PCIe switch, processes 0 and 1 are also "close" — fine. On unusual topologies, look at `nvidia-smi topo -m` to confirm.

For most training, you don't tune topology. NCCL handles it. The reflex is: if collective performance is mysteriously slow despite a fast-looking network on paper, run `NCCL_DEBUG=INFO` and check what topology NCCL is using.

#### Q&A; — About CUDA semantics **Q:** Why is the default stream "the default" — what's special about it? **A:** Historically, CUDA had a "legacy default stream" that was implicitly synchronizing — every other stream had to wait at sync points with it. Modern PyTorch uses _per-thread default streams_ that are non-synchronizing (each thread has its own default that runs concurrently with other threads' work). For most users it's just "the stream your work goes on if you don't explicitly create one." **Q:** When should I call `torch.cuda.empty_cache()`? **A:** Almost never in normal training. The caching allocator's whole point is to _not_ return memory to the driver — calling `empty_cache` defeats it, and any subsequent allocation pays the `cudaMalloc` cost. Two exceptions: (1) before switching to evaluation/validation, where memory layout differs significantly, you may benefit from giving the allocator a clean slate. (2) Before allocating one giant tensor that requires a contiguous block, after a fragmentation-heavy phase. Otherwise, leave it alone. **Q:** My step has a sync point I can't find — backward is mysteriously slow. Where do I look? **A:** Common culprits: a `.item()` on a CUDA tensor (logs, conditionals), `print()`ing a CUDA tensor (forces a D2H copy and sync), an old GAN-style `if loss < threshold: break` pattern, calling `.cpu()` on intermediate values for logging. Profile (M13) and look for `cudaStreamSynchronize` events on the CPU lane during backward — the ones _not_ at step boundaries are bugs. **Q:** CUDA Graphs sound great. Why aren't they always on? **A:** Two reasons. (1) They demand stable shapes — variable-length input, dynamic batching, or conditional model paths break capture. (2) They demand stable storage — the captured kernels write to specific addresses, so you can't pass new tensors per step. For LLM pretraining (fixed seqlen, fixed batch), Graphs are nearly free. For inference with variable-length user prompts, they need careful work (batched prefill with bucket sizes, etc.). `torch.compile(mode="reduce-overhead")` applies Graphs where it's safe and falls back where it isn't. **Q:** What's the deal with `cudaMallocAsync` and "expandable segments"? **A:** `cudaMallocAsync` is a newer CUDA allocator API that supports stream-aware allocation and can grow segments dynamically (instead of pre-fixed segment sizes). PyTorch's "expandable_segments:True" mode uses it. Less fragmentation in practice — segments grow into available memory rather than being chosen up front. Recommended on Ampere+ for dedicated GPUs. Doesn't yet work in every shared-GPU configuration, which is why it's not default. 

## Code Magnets: capture and replay a CUDA Graph

You're capturing a static training step with a CUDA Graph. Three magnets are wrong.

Arrange the magnets into a working capture-and-replay.

for _ in range(3): train_step(model, opt, batch) torch.cuda.synchronize() g = torch.cuda.CUDAGraph() opt.zero_grad(set_to_none=True) with torch.cuda.graph(g): static_loss = criterion(model(static_x), static_y) static_loss.backward() opt.step() print(f"loss: {static_loss.item()}") for step in range(num_steps): static_x.copy_(real_x); static_y.copy_(real_y) g.replay() static_x = real_x; static_y = real_y

show solution
    
    
    for _ in range(3): train_step(model, opt, batch)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        static_loss = criterion(model(static_x), static_y)
        static_loss.backward()
        opt.step()
    for step in range(num_steps):
        static_x.copy_(real_x); static_y.copy_(real_y)
        g.replay()

The traps:

  * `print(f"loss: {static_loss.item()}")` _inside_ the capture: `.item()` requires a host-device sync, which CUDA Graphs forbid. Capture would error or produce a graph that fails to replay.
  * `static_x = real_x; static_y = real_y`: rebinding the names breaks the capture. The graph holds _storage pointers_ , not Python references — you must copy data _into_ the captured tensors via `copy_`, not reassign.

The shape: **warm up → sync → capture (no syncing ops inside) → replay with copy_-into-static-buffers**. After `g.replay()`, `static_loss` holds the new step's loss; you can read it later (after a sync) without affecting capture.

## Who does what?

Match each CUDA-semantics concept to its real role.

Concept

Real role

CUDA stream

A. FIFO queue of GPU work; ops within one stream run in order.

torch.cuda.synchronize()

B. Block the host until all queued GPU work in the current stream completes.

Caching allocator

C. Layer between PyTorch and cudaMalloc; recycles freed blocks for fast tensor allocation.

Segment

D. A large region from one cudaMalloc, split into many tensor-sized blocks.

memory_reserved > memory_allocated

E. Allocator holds segments beyond live tensors — normal, not a leak.

Expandable segments

F. Newer mode using cudaMallocAsync; segments grow dynamically, less fragmentation.

CUDA Graph

G. Recorded kernel sequence replayed with a single API call — kills launch overhead.

show solution

**CUDA stream** → A  
**torch.cuda.synchronize()** → B  
**Caching allocator** → C  
**Segment** → D  
**memory_reserved > memory_allocated** → E  
**Expandable segments** → F  
**CUDA Graph** → G 

The mental shortcut: _stream is the queue, sync drains it, allocator recycles, segment = one cudaMalloc, reserved = held bytes incl free blocks, expandable_segments shrinks fragmentation, Graphs replay launches_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team's training shows `memory_allocated() = 28 GB` but `memory_reserved() = 78 GB` on an 80 GB H100 — barely room to grow. They expect to be at 28 GB. Is this a leak?

show answer

Almost certainly not. `memory_reserved` is what the allocator holds via `cudaMalloc`; `memory_allocated` is what's bound to live tensors. The 50 GB gap is the allocator's free list — segments that previously held tensors but were freed; the allocator hangs onto them for fast reuse. This is normal and usually fine. If they're worried about headroom, two moves: (1) try `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — much less fragmentation in practice. (2) Before any phase requiring a big contiguous allocation (loading a checkpoint, switching to validation), call `empty_cache()` to release segments back. But "reserved > allocated" by itself is not a leak — it's the design.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why does `x.to('cpu')` implicitly call `cudaStreamSynchronize`?

show answer

Because the host needs the actual _bytes_. Any kernel queued on the stream that writes `x` hasn't necessarily finished by the time you call `.to('cpu')`. To return correct data, the framework must wait for those kernels to complete before issuing the D2H copy. That's a hard sync point. Same logic for `.item()`, `.tolist()`, `.numpy()`, and any path that returns the values to Python. _This is why these calls are sneaky: they don't look expensive, but they drain your queue._

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** A team captures a CUDA Graph for their training step. It works for the first step but errors on the second. Why might that be?

show answer

Most likely: shape variance. CUDA Graphs are sensitive to anything that changes between captures. Check (a) batch size — is the dataloader producing variable-size last batches? (b) sequence length — is padding consistent or per-batch? (c) any conditional that runs differently (e.g., logging or saving every N steps causing different ops to be queued). Quick fix: drop the last partial batch (`drop_last=True` in DataLoader, M10) and ensure consistent padding. If the second-step error is more cryptic ("graph cannot be modified"), suspect a tensor that's growing — typically an accumulator or a per-step list. CUDA Graphs require strictly identical kernel sequences and tensor identities each replay.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** When would you intentionally use multiple CUDA streams in PyTorch (beyond the default)?

show answer

Three legitimate cases:

  1. **Compute/copy overlap** : put data H2D copies on a copy stream so the next batch uploads while the current batch computes. PyTorch's `pin_memory=True` \+ `non_blocking=True` handles this automatically; you don't write the streams yourself.
  2. **Compute/communication overlap in distributed training** : FSDP (M17) uses extra streams internally to prefetch parameter all-gathers concurrently with the previous layer's compute. DDP (M16) bucketing implicitly relies on this for backward+all-reduce overlap.
  3. **Custom pipelines** : when you have multiple independent ops that don't share data and you want them concurrent (rare in user-level code; common in custom kernel libraries).

For ordinary training-loop code: don't manually create streams. The default works, and PyTorch's higher-level features manage extra streams for you when it actually helps.

### What just happened?

  * **CUDA streams** are FIFO queues. Host queues kernels (fast); device drains and runs them (async). Within a stream, ops are ordered. Across streams, they can be concurrent.
  * **Most training uses one stream** , the default — fine. Multi-stream is for explicit overlap (copies/compute, communication/compute).
  * **Sync primitives** : `torch.cuda.synchronize()` drains the stream from the host. `.item()`, `.cpu()`, `.numpy()` implicitly sync because they need the values on the host. Avoid these in hot loops.
  * **The caching allocator** sits between PyTorch and `cudaMalloc`. Tensors are carved from pre-allocated **segments** as **blocks**. Allocation is microseconds-fast.
  * `memory_allocated()` = bytes in live tensors. `memory_reserved()` = bytes the allocator holds (live + free). Reserved > allocated is normal, not a leak.
  * **Fragmentation** happens when free memory is scattered across non-contiguous blocks. Causes "phantom OOMs" — enough free total, no contiguous block of the right size.
  * **Fixes** : `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (Ampere+, dedicated GPUs); `empty_cache()` at major phase transitions (rarely useful otherwise); `set_per_process_memory_fraction` for shared GPUs.
  * **Memory snapshot tool** visualizes every allocation with its call stack. Use when memory is mysterious.
  * **CUDA Graphs** : record a kernel sequence once, replay with one API call. Bypasses per-launch overhead. Best for static-shape workloads (LLM pretraining). `torch.compile(mode="reduce-overhead")` uses them automatically where safe.
  * **Graph rules** : stable shapes, stable tensor identities, no host-device syncs (no `.item()`) inside capture. Copy data into pre-captured buffers; don't rebind names.
  * **Topology / NUMA** : NCCL handles topology-aware routing automatically. `nvidia-smi topo -m` shows physical layout if needed.
  * The reflex: when something seems mysteriously slow on GPU, ask _which stream is busy, who is waiting, and is anyone forcing a sync that shouldn't be there?_ Profile (M13) and look at the CPU lane for `cudaStreamSynchronize` events.

Module 21 takes the dispatcher tower (M19) and the CUDA semantics (this module) and shows how `torch.compile` collapses both. Dynamo traces Python into FX graphs; AOTAutograd handles forward + backward together; Inductor code-generates fused C++/Triton kernels; `mode="reduce-overhead"` wires up CUDA Graphs. The full system that makes "slow eager → fast compiled" possible.
