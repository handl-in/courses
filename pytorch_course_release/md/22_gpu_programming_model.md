# Module 22 — GPU programming model

# GPU programming _model_

_Part VIII · Module 22_

— SMs, warps, threads, the memory hierarchy, and the roofline that decides whether your kernel is bound by compute or by memory bandwidth

\--- 

For 21 modules we treated the GPU as a fast box that runs whatever ATen kernel the dispatcher hands it. Part VIII opens the box. The next four modules — GPU model (this one), CUDA extensions (M23), Triton (M24), and FlashAttention as a case study (M25) — are about _writing_ the kernels yourself, not just calling them.

Before we write any code, we need to know what we're targeting. A modern GPU is not a CPU with more cores; it's a fundamentally different machine. CPUs minimize per-thread latency; GPUs maximize collective throughput. CPUs have huge caches and complex out-of-order execution; GPUs have tiny caches per thread and rely on having tens of thousands of threads in flight to hide memory latency. Until you have the hardware model in your head, kernel performance feels arbitrary.

> **★ KEY IDEA**  
>  A GPU is built around three nested concepts. The hardware unit is the **SM** (streaming multiprocessor) — an A100 has 108, an H100 has 132. SMs execute work in **warps** of 32 threads in lockstep. Threads are organized into **blocks** , and blocks form a **grid** — a kernel launches a grid of blocks across all the SMs. The other axis is the **memory hierarchy** : registers (per thread, fastest) → shared memory (per block, ~10× slower) → L2 (chip-wide, ~50× slower) → HBM (off-chip, ~500× slower than registers). Every kernel-writing decision is shaped by these two structures. 

## One new face

W

Warp

"I'm 32 threads that execute in lockstep. We share a single program counter."

When you launch a CUDA kernel with a thread block of 256 threads, what really runs is 8 of me — 8 warps. Within me, all 32 threads execute the _same instruction_ at the same time, just on different data. If your code branches and half my threads take one path while half take another, I have to run _both_ paths sequentially with the inactive lane masked off. That's **warp divergence** : the cost of writing if/else over per-thread state. I'm efficient when my 32 threads agree.

## The execution hierarchy

Three levels nested inside each other:

  * **Thread** : the smallest unit. Has its own registers and program counter (logically). Runs your kernel code.
  * **Warp** : 32 threads that _actually_ execute together in lockstep on the hardware. The warp is the real execution unit — a thread is a software abstraction inside a warp.
  * **Block (Cooperative Thread Array)** : a group of threads (1-1024) that runs entirely on one SM. Threads in a block can share fast on-chip memory and synchronize.
  * **Grid** : the collection of all blocks for one kernel launch. Blocks are independent and may run on any SM in any order.

CUDA execution hierarchy: Grid → Blocks → Warps → Threads Grid (one kernel launch — many blocks) Block 0 Block 1 Block 2 Block 3 … many more, distributed across SMs Zoom into Block 0: 256 threads = 8 warps Warp 032 threads Warp 132 threads Warp 232 threads Warp 332 threads Warp 4 Warp 5 Warp 6 Warp 7 Zoom into Warp 0: 32 threads run the SAME instruction in lockstep t0 t1 t2 t3 t4 t5 t6 t7 … t28 t29 t30 t31 → same op, 32 lanes grid is the launch shape; SMs pull blocks off the queue a block lives entirely on ONE SM — never moves once started warps are the real execution unit — 32 threads, single PC, lockstep

The whole point of this hierarchy is to give you a way to map a problem onto thousands of threads and have them coordinate at the right granularity. Threads in the same warp can share registers via shuffle ops. Threads in the same block can share _shared memory_ — fast on-chip RAM — and synchronize via `__syncthreads()`. Threads in different blocks generally don't communicate; the only way to globally synchronize is to end the kernel and start another.

### Warp divergence: the lockstep tax

Warps execute in lockstep. If you write:
    
    
    if (threadIdx.x < 16) {
        // half the warp does this
        a = expensive_op(x);
    } else {
        // other half does this
        a = other_op(x);
    }

Within one warp, half the threads (lanes 0-15) take the if branch; the other half (lanes 16-31) take the else. The hardware _can't_ run both branches simultaneously — there's only one program counter per warp. So it runs the if branch with lanes 16-31 masked off, then runs the else branch with lanes 0-15 masked off. Total time is the sum of both branches.

This is **warp divergence**. It's not always a disaster — branches that nearly always go one way are cheap (the unused branch is skipped most of the time). The bad case is when divergence happens on a hot path with substantial work in each branch.

Better: when you can, structure code so the branch condition aligns with warp boundaries. `if (threadIdx.x < 32 * some_warp_id)` is fine — entire warps go one way or the other, no divergence. Or use predication: compute both, select the right one with a masked write.

## The streaming multiprocessor (SM)

The SM is the hardware unit that _executes_ warps. An A100 has 108 SMs. An H100 has 132. Each SM has:

  * **CUDA cores** : arithmetic units. An H100 SM has 128 fp32 CUDA cores.
  * **Tensor Cores** : matmul-specialized units. An H100 SM has 4. They do 4×4 fp16/bf16 matmuls per cycle (much more for tf32/fp8).
  * **Register file** : ~64K 32-bit registers per SM. Distributed across active threads.
  * **Shared memory + L1 cache** : ~228 KB on H100, configurable split between shared memory (programmer-managed) and L1 cache (hardware-managed).
  * **Warp schedulers** : 4 per SM. Each cycle, each scheduler picks one ready warp and issues an instruction from it.

The SM is a heavily over-subscribed machine. It has compute resources for executing some warps, but the warp scheduler keeps _more_ warps in flight than it can run simultaneously. When one warp stalls (waiting for memory), another runs. **This is how GPUs hide memory latency: by always having other work to switch to.** An SM can hold up to 64 warps "resident" simultaneously on H100 (= 2048 threads); the scheduler keeps cycling through them.

## The memory hierarchy

The other axis. GPUs have a steep memory hierarchy — orders of magnitude difference between levels in both bandwidth and latency. Knowing the hierarchy is the difference between writing a fast kernel and a slow one.

GPU memory hierarchy (H100 numbers; A100 similar within ~30%) Registers (per thread) ~256 per thread · ~zero latency · accessed every instruction ~10 PB/s aggregate Shared memory + L1 (per SM) ~228 KB · ~30 cycle latency · programmer-managed scratchpad ~20 TB/s per SM L2 cache (chip-wide, shared by all SMs) ~50 MB · ~250 cycle latency · hardware-managed cache ~5 TB/s HBM (global memory) — what nvidia-smi calls "GPU memory" 80 GB · ~500 cycle latency · 3 TB/s on H100, 2 TB/s on A100 3 TB/s ↑ Off-chip — bandwidth is the dominant cost Bandwidth ratio (one byte fetched from each level): register: 1× shared: ~1/2× (SM aggregate) L2: ~1/8× HBM: ~1/30× — the wall

Read the diagram literally. Registers are essentially free — every arithmetic instruction reads and writes them with no extra cost. Shared memory is fast but limited (~228 KB per SM, shared by all blocks running on it). L2 is bigger but ~10× slower than shared. HBM is enormous (80 GB) but ~30× slower than registers in aggregate bandwidth — and far longer in latency.

The implication: **kernels that read each byte of HBM only once and reuse it heavily in registers/shared memory are fast. Kernels that go back to HBM repeatedly are slow.** A matmul of two 4K×4K fp32 matrices is 32 MB + 32 MB = 64 MB of input data. Read each byte once: 64 MB ÷ 3 TB/s = 21 µs. The actual compute is ~131 GFLOPs at fp32 ÷ 60 TFLOPs/s = ~2 µs. _If your matmul takes 21 µs you're memory-bound; if it takes 2 µs you're compute-bound_. Modern matmul kernels reuse data heavily and hit close to the 2 µs number.

## Roofline analysis: the rule that decides everything

Every kernel falls into one of two regimes: **memory-bound** (limited by bandwidth) or **compute-bound** (limited by FLOP throughput). The roofline model captures this with a single number per kernel: _arithmetic intensity_.
    
    
    arithmetic_intensity = FLOPs_done / bytes_transferred

Then you compare to the hardware's "ridge point":
    
    
    ridge_point = peak_FLOPs / peak_bandwidth   # operations per byte at the boundary

For H100 fp16 with tensor cores: peak ~1000 TFLOPs/s, HBM bandwidth ~3 TB/s. Ridge point ≈ 1000 / 3 = 333 ops/byte. So:

  * **Arithmetic intensity < 333 ops/byte**: memory-bound. The kernel can't keep tensor cores fed. Bottleneck is HBM bandwidth.
  * **Arithmetic intensity > 333 ops/byte**: compute-bound. Tensor cores are saturated; bandwidth is fine.

A few real examples:

Arithmetic intensity of common ops on H100 fp16 Op| FLOPs| Bytes| Intensity| Regime  
---|---|---|---|---  
relu (1M elements)| 1M| 4 MB (read+write)| 0.25| Memory-bound  
matmul 256×256 × 256×256| 34M| 0.4 MB| ~80| Memory-bound  
matmul 4096×4096 × 4096×4096| 137G| 96 MB| ~1430| Compute-bound  
attention (one head, T=2048)| ~33M| ~50 MB (naive)| 0.7| Memory-bound (until FlashAttention)  
  
Two takeaways. (1) Pointwise ops (relu, add, layer norm) are _always_ memory-bound — there's barely any arithmetic per byte. Their fastest possible time is just the time to read inputs from HBM and write outputs back. **This is exactly why fusion (M21) wins so dramatically** : fusing 5 pointwise ops doesn't change FLOPs (still memory-bound) but cuts HBM traffic by 5×.

(2) Attention is memory-bound at typical sizes. The standard implementation reads and writes the full attention matrix (T×T per head) to HBM, which is huge memory traffic. **FlashAttention** (M25) restructures the computation to keep the attention matrix in shared memory and never materialize it in HBM — turning a memory-bound op into a compute-bound one.

## Tensor cores: the matmul accelerators

One more piece. Standard CUDA cores do scalar operations: one multiply-add per cycle per core. Tensor cores do _small matrix multiply-adds_ : one 4×4×4 matmul per cycle on Volta, much bigger blocks on Hopper.

For matmul-heavy work, tensor cores are 5-30× faster than CUDA cores at the same precision. The catch: they only accelerate specific shapes and dtypes. The shapes need to be multiples of 8 or 16 (depending on dtype) for tensor cores to even fire. The dtypes need to be fp16, bf16, tf32, int8, fp8, or fp4 (newer hardware adds more); plain fp32 mostly doesn't get tensor-core acceleration (TF32 does — that's why TF32 was created, M14).

Implications:

  * **Pad your matmul shapes to multiples of 8** for fp16/bf16, multiples of 16 for fp8. A 4097×4097 matmul falls off tensor cores; pad to 4104×4104 and you get full speed.
  * **Use bf16 (or fp16, or fp8 on H100+)**. fp32 matmul without TF32 is dramatically slower than bf16 with tensor cores.
  * **Embedding lookups, layer norm, softmax don't benefit from tensor cores** — they're not matmul-shaped. They run on CUDA cores and are usually memory-bound regardless.

## Occupancy: how many warps fit per SM

An SM has fixed resources: register file, shared memory, warp scheduler slots. The number of warps that can be "resident" simultaneously on one SM is limited by whichever resource runs out first.

**Occupancy** = (active warps per SM) / (maximum warps per SM). Higher occupancy means more warps available to switch to when one stalls — better latency hiding.

Three things drain occupancy:

  1. **Register pressure** : each thread's registers come from the SM's register file. If your kernel uses 256 registers per thread, you can only have a few warps resident (the file runs out). Compilers report this with `--ptxas-options=-v`.
  2. **Shared memory** : blocks need shared memory to run; if each block needs 100 KB, only 2 blocks fit per SM.
  3. **Block size** : smaller blocks → fewer threads → fewer warps. But blocks too small (<128 threads) waste warp scheduler slots.

Higher occupancy is usually better — but not always. Some kernels (especially compute-bound ones with heavy register reuse) want _low_ occupancy: more registers per thread to keep more data in registers, fewer warps to context-switch between. **The matmul kernels in cuBLAS run at very low occupancy (~25%) but saturate tensor cores by keeping huge tiles in registers and shared memory.** Higher occupancy would actually slow them down.

The rule of thumb: if your kernel is memory-bound, want high occupancy (more warps to hide latency). If it's compute-bound on tensor cores, low-to-medium occupancy with good register reuse is fine.

## Concrete numbers: A100 vs H100

Numbers ground intuition. Memorize roughly:

The two GPUs you'll most often target (rough numbers) Spec| A100 80GB| H100 80GB SXM  
---|---|---  
SMs| 108| 132  
Tensor cores per SM| 4 (3rd gen)| 4 (4th gen, much bigger)  
Peak fp16/bf16 TFLOPs (tensor cores)| 312| ~1000 (with sparsity 2×)  
Peak fp8 TFLOPs (tensor cores)| —| ~2000  
HBM bandwidth| 2 TB/s| 3.35 TB/s  
L2 cache| 40 MB| 50 MB  
Shared memory + L1 per SM| 192 KB| 228 KB  
Memory| 80 GB HBM2e| 80 GB HBM3  
NVLink bandwidth| 600 GB/s| 900 GB/s  
  
Roughly: H100 is 3× faster in compute (especially fp8), 50% faster in memory bandwidth, with somewhat more shared memory and L2. The ridge point gets _higher_ on H100 — meaning more ops have to be done per byte to be compute-bound. Memory bandwidth is increasingly the binding constraint.

## The kernel writer's mental model

Putting it all together. When you sit down to write a kernel:

  1. **Compute arithmetic intensity**. FLOPs / bytes. Compare to the ridge point. Are you memory-bound or compute-bound?
  2. **If memory-bound** : minimize HBM traffic. Read inputs from HBM _once_. Reuse data across many ops via shared memory. Fuse multiple ops into one kernel.
  3. **If compute-bound** : keep tensor cores fed. Use the right dtypes (bf16/fp16/fp8). Pad shapes to tensor-core-friendly multiples. Maximize register reuse — blocked algorithms with tiles that fit in registers and shared memory.
  4. **Manage occupancy intentionally**. High occupancy for memory-bound (latency hiding); low-medium with good register reuse for compute-bound.
  5. **Avoid warp divergence** on hot paths. Align branches to warp boundaries when possible.
  6. **Plan synchronization**. `__syncthreads()` is cheap within a block; cross-block sync requires ending the kernel. Most kernels are designed to do all the work for one tile within a single block.

#### Q&A; — About the GPU model **Q:** If a thread is just a software abstraction inside a warp, why does CUDA pretend each thread is independent? **A:** Programming model abstraction. Writing "for each thread, do X" is much easier than "for each warp, do X for 32 lanes simultaneously." The hardware enforces the lockstep underneath; the language lets you ignore it most of the time. You only think about warps explicitly when (a) using warp-level primitives like shuffles, or (b) avoiding divergence. **Q:** How do I know how many SMs a GPU has, dynamically? **A:** In Python: `torch.cuda.get_device_properties(0).multi_processor_count`. Useful when sizing grids: a common heuristic is to launch enough blocks to keep all SMs busy. `nvidia-smi -q` also shows it. **Q:** Why is shared memory only 228 KB per SM? Couldn't they make it bigger? **A:** Shared memory is built from SRAM cells right inside each SM, in the shadow of the compute units. SRAM is fast but expensive in silicon area; making it 10× bigger would mean a much bigger chip or fewer SMs. The size is a deliberate engineering tradeoff. The way around it: tile your computation so each tile fits in 228 KB. That's exactly what FlashAttention does (M25). **Q:** What does `__syncthreads()` actually do? **A:** It's a barrier inside a block. All warps in the block wait until every warp has reached the barrier; then they continue together. Cheap — typically a few cycles. Used between phases of a tile-based algorithm: load to shared, sync, compute, sync, write out. Only synchronizes _within a block_ ; cross-block sync requires ending the kernel. **Q:** When does it make sense to write a kernel by hand vs use a library like cuBLAS or Triton? **A:** Three cases for hand-writing: (1) operations that don't have a library (exotic activation functions, custom attention variants). (2) Performance-critical fused operations where avoiding intermediate writes to HBM is the win — Triton is usually the right tool here, not raw CUDA. (3) Educational. For matmul: _just use cuBLAS_. The cuBLAS kernels are hand-tuned by NVIDIA experts and you won't beat them for the standard cases. Even hand-writing in CUDA, you'd typically link against cuBLAS for the matmul cores. 

## Looking ahead

You now know what we're targeting:

  * A grid of blocks, where each block lives entirely on one SM.
  * Within a block, threads grouped into warps of 32 that execute in lockstep.
  * A steep memory hierarchy: registers (free) → shared (228 KB, fast) → L2 (50 MB, slow) → HBM (80 GB, very slow).
  * Tensor cores for matmul — fast but shape-restrictive.
  * Roofline: arithmetic intensity vs ridge point decides memory-bound vs compute-bound.

M23 picks up immediately: how to write CUDA C++ kernels and link them into PyTorch via the cpp_extension and torch.library APIs. M24 introduces Triton, the higher-level kernel language that maps cleanly to this model and is what Inductor (M21) uses internally. M25 takes everything we've built — memory hierarchy, tiling, shared memory, fusion — and walks through FlashAttention as the canonical case study.

## Code Magnets: identify the regime

You're given four ops. For each, determine whether it's memory-bound or compute-bound on H100 (ridge point ~333 ops/byte at bf16). Three of the magnets pair an op with the wrong regime; arrange the correct ones.

Pick the four correct (op, regime) pairings.

layer_norm on (4096, 1024) bf16 → memory-bound layer_norm on (4096, 1024) bf16 → compute-bound matmul (4096×4096) × (4096×4096) bf16 → compute-bound matmul (4096×4096) × (4096×4096) bf16 → memory-bound elementwise relu on 1B elements bf16 → memory-bound attention (B=8, H=32, T=2048, naive impl) bf16 → memory-bound (HBM-resident attention matrix) attention (B=8, H=32, T=2048, naive impl) bf16 → compute-bound small matmul (16×16) × (16×16) bf16 → compute-bound small matmul (16×16) × (16×16) bf16 → memory-bound (overhead dominates)

show solution

The four correct pairings:

  1. layer_norm on (4096, 1024) bf16 → **memory-bound**. Reads/writes ~16 MB; does ~16M flops. Intensity ~1, far below ridge.
  2. matmul (4096×4096) × (4096×4096) bf16 → **compute-bound**. ~137G flops, ~96 MB read. Intensity ~1430, well above ridge of 333. With tensor cores hitting full throughput, this is one of the few ops that's compute-bound on H100.
  3. elementwise relu on 1B elements bf16 → **memory-bound**. 1G flops vs 4 GB transfer. Intensity 0.25.
  4. attention (B=8, H=32, T=2048, naive) bf16 → **memory-bound**. The naive implementation materializes the (T×T) attention scores in HBM. Bytes transferred dominates flops. Hence FlashAttention (M25) — restructure to keep attention matrix in shared memory.

The "small matmul" (16×16) is also memory-bound (the matmul itself has good intensity, but the launch overhead and HBM read-write of small inputs dominates), but for a different reason than the others — at this scale the kernel can barely warm up tensor cores. Both pairings ("memory-bound") in the pool are correct, but it's a stretch — this is mostly an "overhead-bound" regime.

## Who does what?

Match each GPU concept to its real role.

Concept

Real role

SM (streaming multiprocessor)

A. Hardware unit that executes warps; an A100 has 108, an H100 has 132.

Warp

B. 32 threads in lockstep — the real execution unit.

Block

C. Group of threads on one SM; can share fast memory and synchronize.

Shared memory

D. ~228 KB per SM; programmer-managed scratchpad, ~10× faster than L2.

HBM

E. Off-chip "GPU memory" — 80 GB, slowest level, the bandwidth wall.

Tensor core

F. Matmul-specialized unit; 5-30× faster than CUDA cores for fp16/bf16 matmul.

Roofline

G. Compares arithmetic intensity to ridge point — decides memory- vs compute-bound.

show solution

**SM** → A  
**Warp** → B  
**Block** → C  
**Shared memory** → D  
**HBM** → E  
**Tensor core** → F  
**Roofline** → G 

The mental shortcut: _SMs are the hardware units, warps are lockstep groups of 32, blocks live entirely on one SM, shared memory is the on-chip scratchpad, HBM is the off-chip wall, tensor cores accelerate matmul, roofline is the framework for deciding what bounds you_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** Why does adding a tiny `F.relu` (memory-bound) right after a giant matmul (compute-bound) almost halve the matmul's effective speed when not fused?

show answer

The matmul's output sits in HBM after the matmul kernel finishes (write phase). Then the relu kernel reads it back from HBM, applies relu, writes the result to HBM. Net: the matmul output makes _two_ round-trips through HBM (matmul writes it, relu reads + writes). Each pass is bandwidth-limited at 96 MB ÷ 3 TB/s = ~32 µs. With fusion (the matmul kernel can append relu inline before writing), you'd skip the extra read + the second write = ~64 µs saved. For a matmul that itself takes ~130 µs in the compute, that's about half its time gone to a pointwise op that "should" be free. This is exactly why `torch.compile` (M21) and Inductor's fusion logic exist — and why FlashAttention (M25) is so impactful.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** A kernel writer says "I doubled my register usage and got 30% slower despite using fewer threads — but the algorithm should be identical." What's likely happening?

show answer

Occupancy collapse. By doubling register usage per thread, the SM's register file (which is fixed) can hold half as many threads. If they were already at marginal occupancy, the new occupancy might be too low to hide memory latency — warps stall on memory and there aren't enough other warps to switch to. Two fixes: (a) reduce register usage (smaller tiles, simpler loops), (b) accept the lower occupancy if the kernel is compute-bound and register-rich (cuBLAS does this on purpose). The diagnostic: `nsys` or `ncu` shows occupancy; if it dropped substantially with the change, that's likely your story.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** Sketch why FlashAttention (M25) takes attention from memory-bound to compute-bound, in roofline terms.

show answer

Naive attention computes the full T×T attention scores matrix and writes it to HBM, then reads it back to multiply with V. Memory traffic scales with T². For T=2048, that's a (2048×2048) matrix = 8 MB per head per layer, all going through HBM repeatedly. Arithmetic intensity is low — most of the time is HBM round-trips.

FlashAttention restructures: tile Q, K, V into blocks. For each tile, compute partial softmax and partial output, accumulating into V's output buffer. The attention matrix never materializes in HBM — it lives in shared memory inside each block. Memory traffic is reduced to just reading Q, K, V once and writing the output once. Arithmetic per byte goes up dramatically. The kernel becomes compute-bound, hitting tensor-core peak throughput. We'll do this concretely in M25.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Why are PyTorch's elementwise ops (add, relu, sigmoid) usually memory-bound regardless of how big the tensors are?

show answer

They have constant arithmetic per element (one add, one max, one sigmoid). Per element: ~1 FLOP, but you read 4 bytes (fp32) or 2 bytes (bf16) and write 4 or 2 back. Intensity is on the order of 0.1 ops/byte regardless of tensor size. Always far below the ridge point of ~333 — always memory-bound. _The only knob you have for elementwise ops is to fuse them with neighboring ops to amortize the HBM read_. This is precisely what `torch.compile` does. There's no algorithmic trick that makes a single elementwise op compute-bound — you'd have to do hundreds of FLOPs per element, and the op wouldn't be elementwise anymore.

### What just happened?

  * A GPU is built around **SMs** (108 on A100, 132 on H100). SMs execute work in **warps** of 32 threads in lockstep. Threads form **blocks** ; blocks form a **grid**.
  * **Warp divergence** : branches inside a warp serialize. Align branches to warp boundaries on hot paths.
  * The **memory hierarchy** : registers (per thread, free) → shared memory + L1 (~228 KB per SM, fast scratchpad) → L2 (~50 MB chip-wide) → HBM (80 GB, off-chip, the wall).
  * HBM bandwidth is the dominant cost for most kernels. Reading each byte once and reusing it heavily is the path to fast.
  * **Roofline analysis** : `arithmetic intensity = FLOPs/bytes`. Compare to ridge point (~333 ops/byte on H100 fp16). Below ridge: memory-bound. Above: compute-bound.
  * Pointwise ops (relu, layer_norm) are _always_ memory-bound. Fusion is the only lever — that's why `torch.compile` wins.
  * Big matmuls are compute-bound on tensor cores. Use bf16/fp16, pad shapes to multiples of 8 (or 16 for fp8).
  * **Tensor cores** : matmul-specialized units, 5-30× faster than CUDA cores. Specific shapes and dtypes only.
  * **Occupancy** : active warps per SM ÷ max warps per SM. Higher hides memory latency. But not always better — compute-bound kernels with rich register reuse run best at low occupancy.
  * `__syncthreads()` is the within-block barrier — cheap. Cross-block sync requires ending the kernel.
  * The kernel-writer's reflex: compute arithmetic intensity. Compare to ridge. Pick the strategy (minimize HBM traffic vs feed tensor cores) accordingly.
  * Concrete numbers worth remembering: **H100: 132 SMs, ~1000 fp16 TFLOPs, 3 TB/s HBM, 228 KB shared per SM**. That fits on the back of a napkin and supports 90% of kernel reasoning.

Module 23 takes this hardware model and writes our first kernel. C++/CUDA extensions through PyTorch's `cpp_extension` module: how to write a CUDA kernel, build it via JIT or setuptools, and register it through `torch.library.custom_op` from M19 so it integrates with autograd, autocast, and compile.
