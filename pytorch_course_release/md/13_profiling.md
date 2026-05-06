# Module 13 — Profiling: where does the time go?

# Profiling: _where does the time go?_

_Part V · Module 13_

— why `time.time()` lies on GPU, the four canonical bottleneck modes, and reading a trace until you know exactly which lever to pull

\--- 

M12 was about predicting memory before you OOM. This module is about predicting _time_ before you spend three days waiting for a slow training run that didn't need to be slow.

The core problem: GPU work is asynchronous. When you write `y = model(x)` in Python, you didn't compute anything — you queued kernels for the GPU to run later. `time.time()` measures Python-side wall-clock and lies about kernel cost. Worse, in any nontrivial training loop your time is split across forward, backward, optimizer, data loading, and communication, and the bottleneck moves around as you scale. You can't fix what you can't see. This module teaches you to see.

> **★ KEY IDEA**  
>  There are **four canonical bottleneck modes** in PyTorch training: _GPU-bound_ (kernels are slow), _CPU-bound_ (Python/dispatcher overhead is slow), _data-bound_ (the loader can't keep up), and _communication-bound_ (collective ops eat the schedule). Each shows a distinctive shape in a profiler trace. Reading traces is a small skill that pays back quickly: 10 minutes of profiling can save 10 hours of optimization in the wrong place. 

## The new face

P

Profiler

"I record what your code _actually did_ on the GPU, with kernel-level precision."

Unlike `time.time()`, I don't lie about async kernels. I instrument both Python and CUDA, recording every op's launch time and execution time. I produce traces you can open in Chrome or Perfetto. I can tell you, for each kernel: how long it took, what shape its inputs were, who called it from Python. Use me before you optimize. The temptation to "guess and tune" is strong — resist it. Three minutes with me beats three hours of guessing.

## Why `time.time()` lies (one more time)

You met this in M3, but it's worth seeing the full failure mode:
    
    
    import time, torch
    x = torch.randn(8192, 8192, device='cuda')
    w = torch.randn(8192, 8192, device='cuda')
    
    t0 = time.time()
    y = x @ w                          # queues a matmul kernel
    t1 = time.time()
    print(f"matmul: {(t1-t0)*1000:.2f} ms")     # prints something like 0.05 ms

0.05 ms? An 8192² matmul takes more like 5 ms on an A100. The kernel hadn't actually run yet — Python returned as soon as the launch was queued. To measure correctly, force a sync:
    
    
    torch.cuda.synchronize()                 # wait for any prior queued work
    t0 = time.time()
    y = x @ w
    torch.cuda.synchronize()                 # wait for our kernel to actually finish
    t1 = time.time()
    print(f"matmul: {(t1-t0)*1000:.2f} ms")     # now ~5 ms — actual

Better, but still not great — `time.time()` on Linux has microsecond-ish resolution and the syncs add overhead. For real measurements, use `torch.cuda.Event`:
    
    
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    
    start.record()
    y = x @ w
    end.record()
    torch.cuda.synchronize()                 # needed before reading elapsed_time
    print(f"matmul: {start.elapsed_time(end):.2f} ms")

CUDA events record markers _on the GPU's command stream_ , so the elapsed time between them is the actual kernel duration, not the Python wall clock. Resolution is sub-microsecond. This is what real benchmarking code uses.

> **⚠ WARNING**  
>  **Always warm up before benchmarking.** The first call to a kernel often triggers JIT compilation, autotuning (cuDNN), or memory allocation. Discard the first few iterations. Standard pattern: 5-10 warmup iterations, then 50-100 timed iterations, take the median. The first iteration can easily be 10× slower than steady state. 

## `torch.profiler`: the proper tool

Manually instrumenting with CUDA events is fine for one-off timing. For real profiling — finding the slowest 5% of operations across an entire training step — use `torch.profiler`.
    
    
    from torch.profiler import profile, ProfilerActivity, schedule
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./trace'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step, batch in enumerate(loader):
            train_step(model, opt, batch)
            prof.step()                       # tells the profiler we finished a step
            if step >= 10:
                break

The schedule is the part everyone gets wrong. Read it left to right:

  * `wait=1`: skip the first step (often slow due to lazy init).
  * `warmup=2`: profile the next 2 steps but discard data — gives autotuners time to settle.
  * `active=3`: actually record the next 3 steps. This is the data you'll analyze.
  * `repeat=1`: do this whole cycle once (so 6 steps total, last 3 captured).

Three things you almost always want:

  * `record_shapes=True` — annotates each kernel with its input shapes. Essential for spotting "wait, why is this matmul tiny?"
  * `profile_memory=True` — overlays memory allocations on the timeline. Lets you see where the activation peak comes from.
  * `with_stack=True` — captures the Python call stack for each op. So when you see a slow kernel, you can find which line of code launched it.

The `tensorboard_trace_handler` writes a JSON trace file you can open in Chrome's tracing tool, Perfetto (`ui.perfetto.dev`), or TensorBoard. Most modern PyTorch users prefer Perfetto — it's the same tool Android engineers use, and the UI is much better than Chrome's old one.

## The four bottleneck modes

Open a profiler trace and you'll see two horizontal lanes for each step: a **CPU lane** (Python, kernel launches, dispatcher work) and a **GPU lane** (the actual kernels). The shape tells you everything about what's slow.

The four canonical bottleneck shapes ① GPU-bound (you want this — GPU is the slow link) CPU GPU GPU is solid — packed end-to-end. You're using the hardware. Tune kernels next (Part VIII). ② CPU-bound (kernel launches and dispatcher overhead dominate) CPU GPU Python + dispatcher: tons of small ops GPU has gaps between kernels — small ops or eager-mode overhead. Fix: torch.compile, fuse ops, larger batches. ③ Data-bound (GPU starves waiting for the loader) CPU GPU waiting for batch... waiting... waiting... GPU only runs in short bursts. Workers can't keep up. Fix: more num_workers, faster __getitem__, prefetch_factor. ④ Communication-bound (multi-GPU collectives eat the schedule) CPU GPU NCCL all_reduce all_reduce Collectives serialize compute. Fix: overlap compute+comm (DDP gradient bucketing, FSDP, M16-M17). your trace will look like exactly one of these. find the shape, then the fix.

Read each panel:

  1. **GPU-bound** : GPU lane is solid, CPU lane shows brief launch bursts. _This is the goal._ The GPU is the slow link, which means you're using the hardware. The optimization next step is kernel-level: better matmul choice, fused kernels, FlashAttention.
  2. **CPU-bound** : CPU lane is busy, GPU lane has gaps between kernels. Common with very small models, very small batches, or eager mode with lots of small ops. The CPU spends so much time launching kernels that the GPU drains the queue and waits. _Fix:`torch.compile` (M21), op fusion, larger batches_.
  3. **Data-bound** : GPU lane has long idle stretches. Workers can't deliver batches fast enough. _Fix: more`num_workers`, faster `__getitem__`, higher `prefetch_factor` (M10)_. Also check pinned memory + non_blocking transfers (M3).
  4. **Communication-bound** : only relevant in distributed training. NCCL all-reduce ops alternate with compute and dominate the timeline. _Fix: overlap compute and communication (DDP gradient bucketing, FSDP — M16-M17)_.

Most production training spends time in some mix of these. Your job is to look at the trace and identify which mode _dominates_. That's where the optimization lever lives.

## Reading a profiler summary table

Before opening the visual trace, the table summary often tells you most of what you need:
    
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

Output looks roughly like:
    
    
    --------------------------------  ------------  ------------  ------------
                                Name      CUDA total      CUDA avg      # calls
    --------------------------------  ------------  ------------  ------------
                              aten::mm      342.1 ms      11.4 ms           30
                  aten::scaled_dot_pro      298.4 ms       9.9 ms           30
                          aten::linear      121.7 ms       4.0 ms           30
                aten::_log_softmax_back       45.2 ms       1.5 ms           30
                         aten::layer_n       31.8 ms       1.1 ms           29
                         aten::dropout       12.4 ms       0.4 ms           29
    --------------------------------  ------------  ------------  ------------
    Self CUDA time total: 856.4 ms

Read top-down. The first two ops (matmul and scaled_dot_product_attention) consume 75% of GPU time. _That's where to focus_. There's no point optimizing the dropout — it's 1.4% of total time, you'll save 12 ms out of 856.

The 80/20 rule applies aggressively in profiling: the slowest 2-3 kernels usually account for 60-80% of GPU time. Optimize those. Ignore the rest until you have to.

## Spotting common pitfalls in a trace

### The `.item()` sync

You met this in M3 — calling `.item()` on a CUDA tensor forces a CPU-GPU sync, which blocks Python until the GPU's queue drains. In a trace, you'll see a stretch of GPU activity, then a _tall vertical line_ on the CPU lane labelled "cudaStreamSynchronize", then the GPU goes idle while CPU waits.

If you see this on every step, you have a logging bug. Common offenders:
    
    
    if step % 10 == 0:
        print(f"loss: {loss.item():.4f}")             # sync. fine if rare.
    
    losses.append(loss.item())                        # sync EVERY step. bad.
    
    if torch.isnan(loss).item():                       # sync on a Boolean every step. bad.
        break

Fix: log every-N-steps (M11), accumulate scalars on GPU and sync once per epoch, use `torch.isnan(loss).any()` only at checkpoint boundaries.

### The H2D / D2H copy storm

If your trace shows lots of small `cudaMemcpyAsync` ops, you're moving small tensors across the PCIe bus repeatedly. Common causes:

  * Calling `.cuda()` inside a hot loop instead of moving once before the loop.
  * Computing something on CPU and moving it to GPU per-step (e.g., creating a mask in NumPy and casting to CUDA every batch).
  * Logging or tensorboard ops that move metrics back to CPU per step.

Each H2D/D2H is small but synchronous-ish (the queue still needs to flush). Death by a thousand cuts.

### Tiny kernels everywhere

If your trace is full of kernels that are 5-50 microseconds each (rather than milliseconds), you're paying kernel launch overhead. Each kernel launch costs a few microseconds in CPU dispatcher work. With thousands of tiny ops per step, that adds up.

Fixes:

  1. `torch.compile` (M21) fuses many small ops into bigger fused kernels.
  2. Manual fusion: e.g., `x.add_(y).mul_(z)` in-place chained, or rewrite as a single tensor expression that avoids intermediates.
  3. CUDA graphs (M20) record a sequence of kernel launches once and replay the recording — eliminates the per-launch overhead for static graphs.

## The diagnosis tree (what to optimize first)

Once you've identified the bottleneck mode, the optimization lever is usually clear. Here's the decision tree:

Bottleneck → fix, ranked by impact Symptom| Most likely fix| Module  
---|---|---  
GPU lane is solid, GPU-bound on matmul| Mixed precision (bf16 matmul); flash attention if applicable| M14, M25  
GPU-bound but on small ops| `torch.compile` for fusion| M21  
CPU lane busy, GPU has gaps| `torch.compile`; bigger batch; reduce eager-mode overhead| M21  
GPU long idle gaps; loader-side waits| Tune `num_workers`, `prefetch_factor`; profile `__getitem__`| M10  
NCCL collectives serialize compute| DDP gradient bucketing; FSDP; reduce param count being all-reduced| M16, M17  
OOM (no time issue but won't fit)| Activation checkpointing; FSDP; reduce batch/seq| M12, M17  
Suspicious tall vertical lines (cudaSync)| Find and remove the `.item()` in your hot path| M3, this module  
Small kernels dominate| `torch.compile` or CUDA graphs| M20, M21  
  
The progression of optimizations to try, in order:

  1. **Mixed precision** (M14) — usually 1.5-2× speedup, single-line change.
  2. **Tune the data pipeline** (M10) — make sure GPU isn't starving.
  3. **`torch.compile`** (M21) — fuses kernels; another 1.3-2× often.
  4. **FlashAttention** (M25) — for long-sequence transformers, large win.
  5. **Distribute** (M16-M17) — when single-GPU isn't enough.
  6. **Custom kernels** (M22-M24) — last resort, when none of the above gets you there.

Most teams stop after step 4 and call it a day. Step 6 is for when you're in the kernel-writing seat.

## Memory profiling (briefly)

Module 12 covered the memory accounting. The profiler also captures memory events when you pass `profile_memory=True`:
    
    
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

This sorts by per-op memory delta, helping identify the operations that allocate the most. Useful for tracking down activation hotspots before reaching for activation checkpointing.

For deeper memory analysis (allocation timeline, fragmentation visualization), use the dedicated memory snapshot API:
    
    
    torch.cuda.memory._record_memory_history()      # start recording
    # ... run training ...
    torch.cuda.memory._dump_snapshot('mem.pickle')
    torch.cuda.memory._record_memory_history(enabled=None)  # stop

Then visualize with PyTorch's memory snapshot viewer — `pytorch.org/memory_viz`. Drag the pickle in. You see every allocation with its call stack. Beats guessing.

## NVIDIA Nsight (one paragraph)

For the deepest kernel-level profiling — looking at SM utilization, memory bandwidth, occupancy, instruction mix — there's `nsys` (Nsight Systems) and `ncu` (Nsight Compute). They're NVIDIA's first-party tools. `nsys` profile gives you a system-level view (your traces + CUDA + NCCL + cuDNN, all timed); `ncu` profile gives you per-kernel hardware counters (warp efficiency, memory throughput, etc.). Most PyTorch users never need them. If you're _writing_ CUDA kernels (Part VIII), they become essential. `torch.profiler` \+ Perfetto is enough for 95% of people.

## Code Magnets: build a clean profiler invocation

You're profiling a training step to find the bottleneck. Use a schedule that warms up properly, captures shapes and memory, and writes the trace for Perfetto.

Arrange the magnets into a working profiler block. Two are red herrings.

from torch.profiler import profile, ProfilerActivity, schedule with profile( activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], activities=[ProfilerActivity.CPU], schedule=schedule(wait=1, warmup=2, active=3), on_trace_ready=torch.profiler.tensorboard_trace_handler('./trace'), record_shapes=True, profile_memory=True, ) as prof: for step, batch in enumerate(loader): train_step(model, opt, batch) prof.step() if step >= 6: break torch.cuda.synchronize()

show solution
    
    
    from torch.profiler import profile, ProfilerActivity, schedule
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=2, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./trace'),
        record_shapes=True, profile_memory=True,
    ) as prof:
        for step, batch in enumerate(loader):
            train_step(model, opt, batch)
            prof.step()
            if step >= 6: break

The traps:

  * `activities=[ProfilerActivity.CPU]` only — would miss CUDA timing entirely. You almost always want both.
  * `torch.cuda.synchronize()` outside the profiler block — unnecessary; the profiler handles its own sync at trace boundaries.

Note the magic 6 — with `wait=1, warmup=2, active=3`, you need at least 6 steps for the profiler to capture all 3 active steps. If you stop earlier, the trace is incomplete.

## Who does what?

Match each profiling concept to its purpose.

Concept

Purpose

torch.cuda.Event

A. Stops Python until all queued GPU work finishes.

torch.cuda.synchronize()

B. Records markers on the GPU command stream for sub-microsecond timing.

torch.profiler with schedule

C. Captures CPU + CUDA activity over a windowed range of training steps.

record_shapes=True

D. Forces an implicit GPU sync because the result is needed on CPU.

.item() in hot loop

E. Annotates each captured op with its input tensor shapes — diagnoses size-related slowness.

GPU lane has gaps

F. The data loader can't keep up — workers/prefetch are the lever.

NCCL bars alternate with GPU work

G. Communication-bound: collectives serialize compute. DDP/FSDP overlap is the fix.

show solution

**torch.cuda.Event** → B  
**torch.cuda.synchronize()** → A  
**torch.profiler with schedule** → C  
**record_shapes=True** → E  
**.item() in hot loop** → D  
**GPU lane has gaps** → F  
**NCCL bars alternate with GPU work** → G 

The mental shortcut: _Event for accurate kernel timing, synchronize for "wait until done", profiler+schedule for full-step breakdown, record_shapes for diagnosis, .item() syncs are the silent killer, gaps mean data starvation, NCCL bars mean comm bottleneck_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** Time the matmul `x @ w` for `x`, `w` both `(4096, 4096)` on GPU using `torch.cuda.Event`, with proper warmup. Report the median over 50 timed iterations.

show answer
    
    
    x = torch.randn(4096, 4096, device='cuda')
    w = torch.randn(4096, 4096, device='cuda')
    
    # warmup
    for _ in range(10):
        _ = x @ w
    torch.cuda.synchronize()
    
    times = []
    for _ in range(50):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        _ = x @ w
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    print(f"median: {times[25]:.3f} ms")

On an A100 you'll get something like 0.7-1.0 ms; on an H100 maybe 0.4 ms. Note the warmup loop — the first iteration is often 5-10× slower because cuBLAS picks an algorithm and caches it.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Your training trace shows the GPU lane has roughly equal-sized gaps after every kernel, and the CPU lane is busy throughout. What bottleneck mode are you in, and what's the first thing to try?

show answer

CPU-bound. The CPU is busy launching kernels but each launch's overhead is comparable to or larger than the kernel itself, so the GPU drains the queue and waits. First fix: `torch.compile(model)`. It fuses multiple eager-mode ops into a single kernel, dramatically reducing launch overhead. Often a 1.3-2× speedup with one line. Second fix: increase batch size if memory allows — bigger kernels make the launch overhead a smaller fraction of work.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** A team's trace shows long idle stretches on both CPU and GPU lanes, with no workers visible. They run with `num_workers=0`. What's happening?

show answer

With `num_workers=0`, the DataLoader runs `__getitem__` in the main process, synchronously, between training steps. So the timeline is: GPU finishes step → main process loads next batch (CPU work, no kernel launches → both lanes idle) → main process kicks off the next forward → GPU starts. The "idle on both lanes" is _data loading time on the same thread as the trainer_. Fix: `num_workers=4` minimum. Workers run in parallel with training, and the next batch is ready when the previous step finishes.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** You profile a single training step and the table summary shows `aten::mm` takes 60% of the time. Is that good or bad?

show answer

It's the closest thing to "good" you can be in this analysis. Matmul is the right thing to spend time on — it's the bulk of useful work in any deep learning model, and matmul kernels (cuBLAS) are some of the most optimized code on the planet. If you've reached the point where matmul is the bottleneck, you've successfully eliminated all the _other_ bottlenecks. Next-level optimizations: mixed precision to use tensor cores (M14), `torch.compile` to fuse surrounding ops, FlashAttention if you have attention. But "matmul-bound" is largely the goal state.

### What just happened?

  * **`time.time()` lies on GPU** because kernels are async. For accurate timing, use `torch.cuda.Event` with `elapsed_time`, and warm up before measuring.
  * **`torch.profiler`** captures both CPU and CUDA activity. Pass `activities=[CPU, CUDA]`, set up a `schedule(wait, warmup, active, repeat)`, and route the output to `tensorboard_trace_handler` for Perfetto/TensorBoard.
  * **Useful flags** : `record_shapes=True`, `profile_memory=True`, `with_stack=True`.
  * **Four bottleneck modes** visible in a trace: 
    * **GPU-bound** : GPU lane solid (the goal — kernels are now the limit).
    * **CPU-bound** : GPU lane has small gaps; CPU lane busy. Fix: `torch.compile`, larger batches.
    * **Data-bound** : GPU lane has long gaps. Fix: more workers, faster `__getitem__`, prefetch.
    * **Communication-bound** : NCCL bars alternate with compute. Fix: DDP bucketing, FSDP overlap.
  * **The 80/20 rule** : usually 2-3 kernels account for 60-80% of GPU time. Optimize those; ignore the rest.
  * **Common pitfalls** : `.item()` in a hot loop (sync per step), small H2D/D2H copies (death by 1000 cuts), tiny kernels (launch overhead dominates).
  * **Optimization order** : mixed precision → data pipeline → `torch.compile` → FlashAttention → distribute → custom kernels. Most teams stop after step 4.
  * **Memory profiling** : use `profile_memory=True` for per-op summaries; `torch.cuda.memory._record_memory_history` \+ memory_viz for deeper analysis.
  * **Nsight** (`nsys`, `ncu`) for kernel-level hardware counter analysis. Mostly relevant when writing kernels (Part VIII).
  * The reflex: when training is slow, _profile first, optimize second_. Two minutes with the profiler beats two hours of guessing where the cycles went.

That closes the predictive half of Part V. M12 taught you to predict and audit memory; M13 taught you to predict and audit time. Next up: M14 cashes in the mixed-precision recipe that's appeared in every memory and timing discussion. Autocast, GradScaler, the bf16 vs fp16 decision, and the numerics tricks that keep training stable in low precision.
