# Module 24 — Triton: Python kernels at CUDA speed

# Triton: _Python kernels at CUDA speed_

_Part VIII · Module 24_

— the language Inductor writes, the language FlashAttention is written in, and the right tool for nearly every custom kernel you'll write

\--- 

M23 walked through writing a CUDA extension end-to-end: a 60-line kernel, a 30-line C++ binding, a 20-line Python wrapper, and a 30-line registration block. Roughly 140 lines of code to express "do RMSNorm fast." This module rewrites the same thing in Triton in about 30 lines of pure Python — and adds autotuning while we're at it.

Triton is a domain-specific language for GPU kernels, embedded in Python. The key idea: instead of writing per-_thread_ code (M23's CUDA pattern), you write per-_block_ code that operates on small tensor tiles. The Triton compiler maps your tile operations onto warps, threads, registers, and shared memory automatically. You keep the parts you need to control (tile sizes, memory access patterns, parallelism strategy) and lose the parts you don't (warp scheduling, thread indexing, register allocation).

> **★ KEY IDEA**  
>  A Triton kernel is a Python function decorated with `@triton.jit`, written in terms of _blocks_ (program instances) and _tiles_ (tensors of compile-time-known shape). Inside the kernel, you use `tl.load` / `tl.store` for HBM I/O with explicit masks, `tl.dot` for matmul on tensor cores, and standard arithmetic (`tl.sum`, `tl.exp`, etc.) on tiles. The Triton compiler handles the per-thread mapping, register allocation, and shared memory placement. **`@triton.autotune`** searches over block sizes and warp counts to pick the fastest configuration for your shapes — the autotuning is what makes Triton kernels frequently match cuBLAS and FlashAttention performance with a fraction of the code. 

## One new face

⛧

Triton

"I write blocks, not threads. The compiler handles warps for you."

CUDA makes you think one thread at a time. I make you think one _tile_ at a time. Each instance of my kernel handles a tile of, say, `(BLOCK_M, BLOCK_N) = (128, 64)`. You describe what to do with that whole tile — load it, multiply it, sum it, store it — and I figure out how to map it onto warps and threads underneath. _You still need to understand the hardware_ (M22): the tile sizes you pick determine occupancy, shared memory usage, and tensor core utilization. But you don't write the per-thread bookkeeping. The result: kernels that fit on one screen and run at 80-95% of cuBLAS speed, with autotuning baked in.

## RMSNorm in Triton

Side-by-side: the same algorithm in both languages.

RMSNorm: CUDA (M23) vs Triton (M24) — same kernel, same speed CUDA (rmsnorm.cu) ~60 lines kernel + 30 lines C++ binding \+ AT_DISPATCH macros for dtypes \+ shared memory tree reduction by hand \+ thread/block indexing arithmetic \+ __syncthreads barriers \+ PYBIND11 registration \+ JIT or setuptools build ~140 LOC total build time: 30-60s on first call block size: hand-picked (BLOCK=256) autotuning: write your own debugging: CUDA_LAUNCH_BLOCKING=1 Triton (rmsnorm.py) ~25 lines @triton.jit kernel \+ ~5 lines launcher with cdiv → tl.load / tl.sum / tl.store → no thread/block indexing → no __syncthreads → no PYBIND11 → no separate build step ~30 LOC total build time: ~1s (first call only) block size: @triton.autotune searches autotuning: free, decorator debugging: TRITON_INTERPRET=1 same speed, ~5× less code, ~30× faster compile, autotuning included

Here's the actual Triton code. Read top-to-bottom; this is a complete, working kernel:
    
    
    import torch
    import triton
    import triton.language as tl
    
    @triton.jit
    def rmsnorm_fwd_kernel(
        x_ptr, w_ptr, y_ptr, rstd_ptr,
        M, D, eps,
        BLOCK_SIZE: tl.constexpr,        # compile-time constant
    ):
        # Each program instance handles ONE row.
        row = tl.program_id(0)
    
        # Compute pointers + a mask for boundary handling.
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x_ptrs = x_ptr + row * D + cols
        y_ptrs = y_ptr + row * D + cols
    
        # Load the full row (BLOCK_SIZE elements; mask handles tail).
        x = tl.load(x_ptrs, mask=mask, other=0.).to(tl.float32)
    
        # Compute mean of squares, then 1/RMS, in fp32.
        mean_sq = tl.sum(x * x, axis=0) / D
        inv_rms = 1.0 / tl.sqrt(mean_sq + eps)
        tl.store(rstd_ptr + row, inv_rms)
    
        # Apply the rescale + weight; cast back to input dtype on store.
        w = tl.load(w_ptr + cols, mask=mask, other=0.).to(tl.float32)
        y = (x * inv_rms) * w
        tl.store(y_ptrs, y, mask=mask)
    
    
    def rmsnorm(x, weight, eps=1e-6):
        assert x.is_contiguous() and weight.is_contiguous()
        M, D = x.shape
        y = torch.empty_like(x)
        rstd = torch.empty(M, device=x.device, dtype=torch.float32)
    
        # Pick BLOCK_SIZE as the next power-of-2 ≥ D, so D fits in one block load.
        BLOCK_SIZE = triton.next_power_of_2(D)
    
        # Grid: one program per row.
        rmsnorm_fwd_kernel[(M,)](
            x, weight, y, rstd,
            M, D, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return y

That's the complete kernel — 25 lines including launcher. Compare to M23's ~140 lines for the same algorithm.

What replaced what:

  * **`tl.program_id(0)`** instead of `blockIdx.x`. The "program" is Triton's name for a kernel instance — equivalent to a CUDA block.
  * **`tl.arange(0, BLOCK_SIZE)`** instead of `threadIdx.x`. You get a tile of indices, not one per-thread index. The compiler decides how to distribute that tile across warps and threads.
  * **`tl.load(ptrs, mask=mask)`** instead of explicit thread-strided loads. Triton handles coalescing automatically; the mask handles boundary cases (when D isn't a multiple of BLOCK_SIZE).
  * **`tl.sum(x * x, axis=0)`** instead of a hand-written shared memory tree reduction. The compiler picks the right reduction strategy.
  * **Type promotion** via `.to(tl.float32)` — same fp32-accumulation pattern as CUDA, but in one expression.
  * **Implicit synchronization** : there's no `__syncthreads()` here because Triton's reductions and stores are synchronous within a program by construction.

### Calling it from Python

The launcher is also pure Python:
    
    
    # Standard usage
    x = torch.randn(128, 2048, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(2048, device='cuda', dtype=torch.bfloat16)
    y = rmsnorm(x, w)         # first call: ~1s compile; subsequent: instant

No build step, no setup.py, no CUDA Toolkit version pinning. Triton compiles to PTX directly via LLVM, caches the result like Inductor caches its kernels, and reruns from cache on subsequent calls. The first call to a kernel triggers compilation (typically 1-3 seconds for a simple kernel; longer for autotuned ones); subsequent calls hit the cache.

## Autotuning: the killer feature

Triton kernels often have one or two performance-critical knobs: `BLOCK_SIZE` in our example, plus `num_warps` and `num_stages` (we'll see those in matmul). Different shapes want different choices. Hand-picking is tedious; autotuning finds the best for each shape automatically.
    
    
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
            triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        ],
        key=['D'],                       # re-tune when D changes
    )
    @triton.jit
    def rmsnorm_fwd_kernel(
        x_ptr, w_ptr, y_ptr, rstd_ptr,
        M, D, eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        # ... same kernel body as before ...
        pass

The first call for each unique value of `D` times all 4 configs and caches the winner. Total upfront cost: roughly 4× the single-config compile time (one per config). Total benefit: the kernel runs at the optimal config for _your_ shape without you knowing what that config is.

The `key` parameter says "re-tune whenever this argument changes." For RMSNorm, the only relevant shape parameter is `D` (the row width); `M` (the row count) just controls how many programs we launch and doesn't affect the per-row tuning. Get this wrong (e.g., add `M` to `key`) and you'll re-tune every time the batch size changes — wasted compile time.

## The Triton execution model

Triton's mental model is the same as CUDA's, but expressed differently. Each piece of CUDA terminology has a Triton equivalent:

CUDA → Triton terminology CUDA| Triton| Notes  
---|---|---  
Kernel launch `<<<grid, block>>>`| `kernel[grid_tuple](args, ...)`| Grid is just a tuple; block is implicit (configured by num_warps).  
blockIdx.x, blockIdx.y| `tl.program_id(0)`, `tl.program_id(1)`| The program is the unit; threads are below the abstraction.  
threadIdx.x| (absent — handled by compiler)| You think tiles, the compiler thinks threads.  
__syncthreads()| (absent in most kernels)| Reductions are atomic; control flow handles synchronization.  
__shared__ memory| `tl.dot` auto-uses; otherwise compiler decides| Tiles in registers; large tiles automatically promoted to shared memory.  
WMMA / mma.sync (tensor cores)| `tl.dot(a, b)`| One line. Triton picks the right tensor core instruction.  
num_warps (you control)| num_warps in `@triton.jit` or `Config`| Same concept; just a Python kwarg.  
  
The big move: **tiles instead of threads**. In CUDA, you wrote a per-thread program; in Triton, you write a per-block program that operates on tiles. The compiler maps your tile operations onto threads. This is why Triton code is so much shorter — most of CUDA's verbosity is per-thread bookkeeping.

## A second example: fused matmul + bias + GELU

Let's do something more interesting. A common transformer pattern: matmul, add a bias, apply GELU. In eager PyTorch, that's three kernels (matmul, add, gelu) with intermediate writes to HBM between. In Triton, one fused kernel:
    
    
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
                          num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32},
                          num_warps=8, num_stages=3),
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32},
                          num_warps=8, num_stages=3),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def matmul_bias_gelu_kernel(
        a_ptr, b_ptr, bias_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        # Each program handles a (BLOCK_M, BLOCK_N) tile of the output.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
    
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
    
        # Pointer arithmetic for the A and B tiles.
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
        # Accumulate in fp32.
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
        # Loop over K in BLOCK_K chunks.
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.)
            acc += tl.dot(a, b)        # tensor cores!
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    
        # Add bias (broadcasts across rows).
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.)
        acc += bias[None, :]
    
        # Apply GELU (using the tanh approximation: faster than the erf form).
        sqrt_2_pi = 0.7978845608
        acc = 0.5 * acc * (1.0 + tl.tanh(sqrt_2_pi * (acc + 0.044715 * acc * acc * acc)))
    
        # Store back, casting to output dtype.
        c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask)
    
    
    def matmul_bias_gelu(a, b, bias):
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                             triton.cdiv(N, meta['BLOCK_N']))
        matmul_bias_gelu_kernel[grid](
            a, b, bias, c,
            M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )
        return c

40 lines for matmul + bias + GELU fused. Notable points:

  * **`tl.dot(a, b)`** dispatches to tensor cores when shapes and dtypes are right. This single call replaces the entire WMMA / mma.sync dance you'd write in CUDA.
  * **2D program grid** — one program per output tile. `tl.program_id(0)` for the row tile, `tl.program_id(1)` for the column tile.
  * **Strides as kernel arguments** — instead of assuming contiguous layout, we pass strides explicitly so the kernel works with arbitrary (e.g., transposed) inputs.
  * **Numerical accumulator in fp32** — same recipe as M14 and M23. Inputs in bf16; accumulate in fp32; cast back on store.
  * **The K-loop** is the matmul accumulation loop. Tiles of K=32 elements are loaded at a time; `tl.dot` on (BLOCK_M, 32) × (32, BLOCK_N) accumulates into the (BLOCK_M, BLOCK_N) tile.
  * **num_stages=3** in the autotune configs enables software pipelining: while one K-tile is being computed by tensor cores, the next is being loaded. This is the same prefetching idea as M16's DDP gradient overlap, but for memory loads inside a kernel.

This kernel will run within 80-95% of cuBLAS's matmul speed on the matmul portion alone, and _save additional time_ by avoiding the HBM round-trips for bias and GELU. Pure win for the fused pattern.

## The matmul tile pattern, visualized

Matmul tiling: each program computes one (BLOCK_M, BLOCK_N) output tile A: (M, K) row tile (BLOCK_M, K) processed in BLOCK_K chunks → B: (K, N) col tile (K, BLOCK_N) C: (M, N) — output tile being computed acc = row tile @ col tile, accumulated in fp32 × accumulate over K-tiles → tensor core (tl.dot)

One program per output tile: the program reads its (BLOCK_M, K) row-strip of A and its (K, BLOCK_N) column-strip of B, accumulates the matmul into an fp32 (BLOCK_M, BLOCK_N) tile, then writes the result. The K dimension is iterated in BLOCK_K-sized chunks; each iteration is one `tl.dot` call that fires the tensor cores. **This pattern — partition outputs into tiles, each program computes one tile — is universal for matmul kernels.** FlashAttention (M25) is a creative variant of this same pattern.

## Practical knobs and gotchas

### The `num_warps` dial

Triton lets you set `num_warps` (in the autotune Config or as a kwarg to `@triton.jit`). Each program runs with that many warps available. Common values: 4 (small kernels with little parallelism per program), 8 (medium, like our matmul example), 16 (large tiles, want more parallelism within the program). Larger num_warps means more threads per program but fewer programs that fit per SM (occupancy tradeoff from M22). Autotune with several values.

### Numerical precision: cast at the right boundaries

The cast pattern from M14 and M23 carries over directly. `tl.load(...).to(tl.float32)` reads bf16/fp16 and promotes to fp32 for the kernel body. `acc.to(tl.bfloat16)` at the store. Don't accumulate in low precision — same numerical hazards as before.

One Triton-specific subtlety: `tl.dot` can take an `out_dtype` parameter that controls the accumulator dtype. `tl.dot(a, b, out_dtype=tl.float32)` uses fp32 tensor cores; the default is whatever Triton thinks is appropriate. Specify it explicitly when you care.

### Debugging: `TRITON_INTERPRET=1`

The killer feature for development. Set this env var and Triton runs your kernel in pure Python (using PyTorch ops) instead of compiling to GPU. You can then use `print()`, `pdb`, anything — your kernel runs as ordinary Python code. _Slow as molasses_ for real workloads, but invaluable when debugging.
    
    
    TRITON_INTERPRET=1 python my_kernel_test.py
    # prints, breakpoints, asserts all work; runs at Python speed

Once the kernel produces correct outputs in interpret mode, switch back to JIT for performance.

### Compile cache and disk persistence

Triton caches compiled kernels in `~/.triton/cache` by default. Each (kernel, signature, config) tuple gets a cache entry. Subsequent runs with the same combination skip compilation. To force a rebuild (e.g., after upgrading Triton), `rm -rf ~/.triton/cache`.

### Combining with `torch.library`

Same as M23 — wrap the Triton launcher in `@torch.library.custom_op` for autograd, autocast, and compile integration:
    
    
    @torch.library.custom_op("my_lib::rmsnorm", mutates_args=())
    def rmsnorm_op(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        return rmsnorm_torch_fallback(x, w, eps)
    
    @rmsnorm_op.register_kernel("cuda")
    def _cuda(x, w, eps):
        return rmsnorm(x.contiguous(), w.contiguous(), eps)   # calls our Triton kernel
    
    @rmsnorm_op.register_fake()
    def _fake(x, w, eps):
        return torch.empty_like(x)
    
    # Plus register_autograd as in M23.

Now `torch.ops.my_lib.rmsnorm(x, w, eps)` goes through the dispatcher tower, calls our Triton kernel on CUDA, falls back to PyTorch ops on CPU, traces under `torch.compile`, and gets gradients via the registered backward.

## Triton vs Inductor: who writes which?

Recall from M21: `torch.compile`'s Inductor backend code-generates Triton kernels for the fused regions of your model. So when you write `torch.compile(model)`, Triton is doing the work under the hood — Inductor is essentially an automatic Triton-kernel writer.

So when do you write Triton yourself vs let Inductor do it?

When to hand-write Triton vs let Inductor generate it Situation| Best path  
---|---  
Standard transformer layers, eager-mode-equivalent semantics| `torch.compile`; let Inductor generate.  
Custom op with no PyTorch equivalent (custom attention, exotic activation)| Hand-write Triton; register via `torch.library`.  
Op where Inductor's generated kernel is suboptimal| Hand-write Triton, register, let compile use yours.  
FlashAttention, paged attention, similar memory-trick algorithms| Hand-written Triton (or the prebuilt libraries that include them).  
Need specific tensor core instructions Inductor doesn't emit| Hand-written Triton with explicit `tl.dot` shapes.  
  
The 90% case is "use compile, don't write kernels." The 10% case is "write a Triton kernel, register it, let compile use it for the parts you didn't write." Inductor and hand-written Triton are complementary, not competing.

#### Q&A; — About Triton **Q:** Why does Triton compile so much faster than CUDA? **A:** Triton's compile pipeline is Python → MLIR → LLVM → PTX. CUDA's is C++ → C++ frontend → LLVM → PTX with a lot of templating overhead. The Python+MLIR path is much cleaner — no template instantiation, no header parsing, no PYBIND11. For simple kernels, Triton compile is a few hundred ms; for autotuned matmul kernels, a few seconds. CUDA can take 30+ seconds for the same work. **Q:** Does Triton work on AMD GPUs? **A:** Yes — Triton has had ROCm/HIP backend support for a while; the same kernel code typically runs on both NVIDIA and AMD with the right Triton install. Performance varies (some patterns optimized harder for NVIDIA), but the language is portable. Same kernel code → different backends compile differently. **Q:** Why use `tl.constexpr` for BLOCK_SIZE? **A:** Triton specializes the kernel on each unique value of compile-time-constant arguments. `BLOCK_SIZE: tl.constexpr` tells Triton: "this is a compile-time constant; generate a different kernel for each value." This lets the compiler unroll loops, allocate fixed-size tiles in registers, and pick optimal codegen. Pass it as a regular int and it'd be a runtime variable — slower and less optimized. **All shape parameters that affect tile sizes should be`tl.constexpr`.** **Q:** My Triton kernel works in interpret mode but produces wrong results when compiled. What gives? **A:** Most often: a subtle issue with masking, pointer arithmetic, or boundary handling that's masked (no pun) in interpret mode. Interpret mode handles edge cases more permissively than compiled code. Common culprits: forgetting `mask=` on a load that runs off the end of an array; integer overflow in pointer arithmetic for large tensors (use `tl.int64` for offsets when tensors are big); using a non-power-of-2 BLOCK_SIZE for a reduction (Triton requires power-of-2 for many ops). Add explicit asserts in interpret mode to catch the boundary issues, then re-test compiled. **Q:** Can I call PyTorch functions from inside a Triton kernel? **A:** No. The Triton kernel body is a tightly restricted DSL — it has tensor-tile primitives but no general Python. You can call other `@triton.jit` functions (Triton supports kernel-to-kernel function calls), but not arbitrary PyTorch ops. PyTorch ops only exist on the launcher side. **Q:** When do I need to worry about num_stages? **A:** In matmul-style kernels with a K-loop (like our matmul example). `num_stages` enables software pipelining: while iteration i is running on tensor cores, iteration i+1's data is being prefetched from HBM. With `num_stages=3`, you have 3 K-tiles in flight at once. Higher num_stages = better overlap but more shared memory needed. For purely pointwise kernels (like RMSNorm), `num_stages` doesn't matter and the default is fine. 

## The minimal mental model

Three things to remember from M24:

  1. **Triton is per-block, not per-thread.** Each program instance handles one tile of work. The compiler maps tiles to warps, threads, and registers automatically. You think tiles; the compiler thinks threads.
  2. **Autotuning is free.** Add a decorator with a few configs; Triton finds the best. This is the practical reason Triton kernels often match cuBLAS — the search space is explored, not guessed.
  3. **Triton +`torch.library` is the standard recipe** for adding a new op to PyTorch. Write the kernel in Triton, wrap with `custom_op`, register fake and autograd. The op slots into autograd, autocast, compile, distributed — all the way down M19's dispatcher tower.

## Code Magnets: build a complete vector-add Triton kernel

You're writing a fused vector add + scalar multiply: `z = (x + y) * alpha`. Three magnets are wrong choices.

Arrange the magnets into a working kernel + launcher.

@triton.jit def add_mul_kernel(x_ptr, y_ptr, z_ptr, alpha, n, BLOCK: tl.constexpr): pid = tl.program_id(0) pid = threadIdx.x offs = pid * BLOCK + tl.arange(0, BLOCK) mask = offs < n x = tl.load(x_ptr + offs, mask=mask) y = tl.load(y_ptr + offs, mask=mask) tl.store(z_ptr + offs, (x + y) * alpha, mask=mask) def add_mul(x, y, alpha): z = torch.empty_like(x) grid = (triton.cdiv(x.numel(), 1024),) add_mul_kernel[grid](x, y, z, alpha, x.numel(), BLOCK=1024) add_mul_kernel(x, y, z, alpha, x.numel(), BLOCK=1024) return z

show solution
    
    
    @triton.jit
    def add_mul_kernel(x_ptr, y_ptr, z_ptr, alpha, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(z_ptr + offs, (x + y) * alpha, mask=mask)
    
    def add_mul(x, y, alpha):
        z = torch.empty_like(x)
        grid = (triton.cdiv(x.numel(), 1024),)
        add_mul_kernel[grid](x, y, z, alpha, x.numel(), BLOCK=1024)
        return z

The traps:

  * `pid = threadIdx.x`: that's CUDA. Triton uses `tl.program_id(0)` — there's no thread-level identifier in Triton's user-facing API.
  * `add_mul_kernel(x, y, z, alpha, x.numel(), BLOCK=1024)`: missing the `[grid]` launch syntax. Triton requires the grid to be specified via `kernel[grid](args)`; calling it like a regular function fails to launch.

Three things to internalize from this puzzle: (1) `tl.program_id(N)` not threadIdx, (2) the `kernel[grid](args)` launch syntax is mandatory, (3) the mask handles boundary when `n` isn't a multiple of BLOCK.

## Who does what?

Match each Triton concept to its real role.

Concept

Real role

@triton.jit

A. Marks a function as a Triton kernel; compiles to PTX on first call.

tl.program_id(axis)

B. Returns the index of this kernel instance along the launch grid axis.

tl.constexpr

C. Marks a parameter as compile-time constant — the kernel specializes per value.

tl.load with mask

D. Reads from HBM with boundary handling — masked-off lanes get the `other` value.

tl.dot

E. Tile matmul that maps to tensor cores — replaces WMMA / mma.sync.

@triton.autotune

F. Searches over configs (BLOCK_SIZE, num_warps, num_stages) for the fastest.

TRITON_INTERPRET=1

G. Runs the kernel as ordinary Python — slow but enables print/pdb debugging.

show solution

**@triton.jit** → A  
**tl.program_id(axis)** → B  
**tl.constexpr** → C  
**tl.load with mask** → D  
**tl.dot** → E  
**@triton.autotune** → F  
**TRITON_INTERPRET=1** → G 

The mental shortcut: _jit compiles, program_id indexes the grid, constexpr triggers specialization, masked load handles boundaries, tl.dot fires tensor cores, autotune searches configs, INTERPRET=1 enables Python debugging_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team's Triton matmul kernel is 30% slower than cuBLAS for square matrices but matches cuBLAS for very tall-and-skinny ones. What's likely going on?

show answer

The autotune configs they're searching probably weren't designed for the square shapes. cuBLAS uses a sophisticated heuristic for picking tile sizes and split-K strategies that Triton's autotune doesn't replicate by default. Two diagnoses: (a) check whether their autotune config list includes the larger tile sizes (BLOCK_M=128, BLOCK_N=256, BLOCK_K=64+) that cuBLAS uses for square shapes — if not, add them. (b) For very large square matmuls, cuBLAS uses split-K (multiple programs cooperate on the same output tile, summing their partial K-results), which Triton can do but most simple matmul examples don't. Adding split-K configs to autotune is what closes the last gap. _Tall-skinny shapes don't benefit from split-K (small K already), so the basic tile pattern wins there — explaining the asymmetry._

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why is `BLOCK_SIZE: tl.constexpr` required for performance in our RMSNorm kernel?

show answer

Without `tl.constexpr`, BLOCK_SIZE would be a runtime variable and the compiler couldn't (a) statically size the tile in registers, (b) unroll the implicit loops over BLOCK_SIZE elements, or (c) pick the right reduction strategy. With `constexpr`, the compiler generates one specialized kernel per BLOCK_SIZE value and inlines all the size-dependent decisions. The cost: more compile time (one kernel per unique value); the benefit: each kernel runs at peak efficiency. **For shape parameters: always constexpr. For data values (eps, alpha): regular runtime args.**

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** A team has a working CUDA extension (M23) and wants to migrate to Triton. Sketch the migration plan.

show answer

  1. **Identify the kernel boundary.** What was the `__global__` function in CUDA becomes the `@triton.jit` function in Triton.
  2. **Translate the indexing.** blockIdx.x → tl.program_id(0); replace per-thread loops with tile-shaped tl.arange + tl.load.
  3. **Translate the reductions.** Hand-coded shared-memory tree reductions become tl.sum / tl.max etc. Same with tl.softmax for attention-style code.
  4. **Translate matmul.** WMMA / mma.sync becomes tl.dot. Often dramatically simplifies code.
  5. **Add autotune configs.** Wrap with @triton.autotune over the dimensional knobs (BLOCK_SIZE, num_warps, num_stages).
  6. **Re-register through torch.library.** The custom_op + register_kernel + register_fake + register_autograd dance from M19 / M23 is unchanged; only the kernel implementation changes.
  7. **Validate with gradcheck (fp64) and a numerical comparison (fp16/bf16) against the original CUDA kernel.** Triton kernels can have slightly different reduction orders, so expect ~1e-3 relative differences in low precision; that's fine.

Total migration effort: typically a few hours for a non-trivial kernel, ending with shorter, autotuned, easier-to-debug code.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Why does Triton's tile-based programming model fit GPUs naturally, despite GPUs being thread-based at the hardware level?

show answer

GPUs are thread-based at the _execution_ level (warps of 32 threads in lockstep) but tile-based at the _useful work_ level — most GPU algorithms operate on contiguous chunks of memory (vector slices, matrix tiles, attention tiles). The CUDA model exposes the per-thread reality and forces you to manually decompose your tile operations into thread-level work. Triton's insight: **this decomposition is mechanical and the compiler can do it.** A tile load decomposes into per-warp vectorized loads with coalescing. A tile reduction decomposes into a tree of shfl operations within a warp + shared memory across warps. The compiler picks the right decomposition based on the tile shape and target hardware. You think at the level your algorithm naturally lives at; the compiler bridges to the level the hardware needs. Same idea as `torch.compile` for whole models, just at the kernel level.

### What just happened?

  * **Triton** is a Python DSL for GPU kernels. You write per-tile code, the compiler maps it to per-thread CUDA. Result: ~3-5× shorter code at near-cuBLAS performance.
  * **The basic kernel pattern** : `@triton.jit` on a function, `tl.program_id` for grid indices, `tl.arange + tl.load(mask=...)` for tile loads, `tl.sum/max/dot` for tile ops, `tl.store(mask=...)` for tile stores.
  * **Tile sizes are`tl.constexpr`** — compile-time constants the compiler specializes on. Always mark BLOCK_SIZE, num_warps configs, etc. as constexpr.
  * **Boundary handling** via `mask=` argument to load/store. The mask handles the case when the dimension isn't a multiple of BLOCK_SIZE.
  * **fp32 accumulation pattern** carries over from M14/M23: `tl.load(...).to(tl.float32)` on read, `acc.to(input_dtype)` on write. Don't accumulate in low precision.
  * **`tl.dot`** dispatches to tensor cores. One line replaces the WMMA/mma.sync dance.
  * **`@triton.autotune`** with a list of Configs searches for the fastest BLOCK_SIZE / num_warps / num_stages for each shape. Free performance.
  * **num_stages** enables software pipelining in K-loops (matmul). Higher = better overlap, more shared memory.
  * **Launch syntax** : `kernel[grid](args)`. Grid is a tuple of program counts per axis. `triton.cdiv(N, BLOCK)` for ceil-div sizing.
  * **Debugging** : `TRITON_INTERPRET=1` runs the kernel as Python — slow but supports print/pdb.
  * **Integration** : same `torch.library.custom_op` \+ `register_kernel("cuda")` \+ `register_fake` \+ `register_autograd` pattern as M23 — the kernel just changes from CUDA to Triton.
  * **Triton vs Inductor** : Inductor generates Triton automatically for fused regions. Hand-write Triton when you need an op Inductor doesn't produce well or that doesn't exist in PyTorch (FlashAttention, paged attention, custom layers).
  * The reflex: when reaching for a custom kernel, _Triton first_. Drop to raw CUDA only for things Triton can't express (CUTLASS-level template gymnastics, async copies, warp specialization).

Module 25 takes everything from Part VIII so far — the GPU model (M22), the kernel-writing toolkit (M23 CUDA, M24 Triton), the dispatcher integration (M19) — and walks through one of the most consequential kernels of the past five years: **FlashAttention**. We'll see how the algorithm transforms attention from memory-bound to compute-bound by keeping the attention matrix in shared memory and never materializing it in HBM, doing the recompute trick from M12 inside a single kernel. The kernel is in Triton; the techniques are universal.
