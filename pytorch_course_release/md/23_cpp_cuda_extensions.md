# Module 23 — C++/CUDA extensions

# C++/CUDA _extensions_

_Part VIII · Module 23_

— writing your first real CUDA kernel, building it via `cpp_extension`, and wiring it into PyTorch through `torch.library` so autograd, autocast, and compile see it as a first-class op

\--- 

M22 gave you the hardware model. Now we use it. This module walks through writing a real CUDA kernel — RMSNorm, the normalization layer used in Llama, Mistral, Gemma, and most modern transformers. It's small enough to display in full, useful enough to be motivating, and a good warm-up for the FlashAttention case study in M25.

Three things to internalize before the code starts. (1) A PyTorch CUDA extension is just _a CUDA kernel + a C++ wrapper + Python glue_. There's no magic; the patterns repeat. (2) The hard part isn't the CUDA — it's the integration with autograd (so backward works), autocast (so mixed precision works), and torch.compile (so it gets fused). The `torch.library.custom_op` API from M19 handles all three. (3) For most kernels you'll write, _Triton_ (M24) is a better tool than raw CUDA. We'll do this in CUDA first because it teaches you what's underneath; then M24 shows the same kind of work in 1/3 the lines.

> **★ KEY IDEA**  
>  A CUDA extension has three files: a `.cu` file with the kernel, a `.cpp` file with C++/Python bindings, and a Python `.py` file that loads and wraps the extension. PyTorch's `torch.utils.cpp_extension.load` JIT-compiles them on first use; `setuptools` with `CUDAExtension` does ahead-of-time builds for distribution. Once compiled, register the resulting C function via `torch.library.custom_op` for full autograd/autocast/compile integration. The mental model: **raw CUDA gives you total control; the Python and library layers exist to make your kernel a first-class PyTorch op.**

## The example: RMSNorm

RMSNorm is a simpler cousin of LayerNorm. For an input vector `x ∈ ℝᴰ`:
    
    
    RMSNorm(x) = (x / RMS(x)) * weight    # where RMS(x) = sqrt(mean(x²) + ε)

One reduction (the mean of squares), one rescale, one elementwise multiply by a learnable weight. Memory-bound (M22) — almost no arithmetic per byte. The job of a custom kernel is to do the reduction and the rescale in one fused pass over HBM, instead of multiple passes.

Per-row, the algorithm is:

  1. Load all D elements of the row into registers.
  2. Square them, sum across threads in the block, reduce to a single scalar.
  3. Compute `1/sqrt(mean + ε)`.
  4. Multiply each element by that scalar and the weight, write out.

The key engineering decision: **one block per row**. A row fits in shared memory + registers; each block handles one row's worth of work; no inter-block communication. Standard pattern for normalizations.

## The file layout

Building a CUDA extension: three files, two compilers rmsnorm.cu CUDA kernel __global__ void ... rmsnorm.cpp C++ wrapper PYBIND11_MODULE(...) rmsnorm.py Python wrapper cpp_extension.load(...) nvcc .cu → .o (PTX/SASS) g++ .cpp → .o linker → rmsnorm.so (loadable .so) Python loads → uses kernel ext = cpp_extension.load( name="rmsnorm", sources=["rmsnorm.cu", "rmsnorm.cpp"] ) y = ext.rmsnorm_fwd(x, w) first call: ~30s compile. subsequent calls: cached, instant.

Three files, two compilers. `nvcc` compiles the `.cu` file into a CUDA-aware object containing PTX/SASS for the GPU side and host code for the launch wrappers. `g++` compiles the `.cpp` binding (using PYBIND11). The linker glues both with the PyTorch C++ libraries (`libtorch`) into a Python-loadable `.so`.

You don't run nvcc and g++ by hand. `cpp_extension.load(...)` handles it — JIT-compiles on first call, caches the result, and gives you a Python module containing your kernel as a function.

## The CUDA kernel

Here's the real kernel. Read it slowly; the comments map each piece to M22's concepts.
    
    
    // rmsnorm.cu
    #include <cuda_runtime.h>
    #include <torch/extension.h>
    
    // One block per row of the input.
    // Within a block, we use blockDim.x threads to cooperatively reduce.
    
    template<typename scalar_t, int BLOCK_SIZE>
    __global__ void rmsnorm_fwd_kernel(
        const scalar_t* __restrict__ x,        // (M, D) input
        const scalar_t* __restrict__ weight,   // (D,)  scaling weights
        scalar_t* __restrict__ y,              // (M, D) output
        float* __restrict__ rstd,              // (M,)   1/RMS, saved for backward
        int D,
        float eps
    ) {
        // One block handles one row.
        int row = blockIdx.x;
        int tid = threadIdx.x;
    
        // Pointers to this row.
        const scalar_t* row_x = x + row * D;
        scalar_t*       row_y = y + row * D;
    
        // 1. Each thread sums the squares of its strided slice.
        float local_sum = 0.0f;
        for (int i = tid; i < D; i += BLOCK_SIZE) {
            float v = (float)row_x[i];
            local_sum += v * v;
        }
    
        // 2. Block-wide reduction via shared memory.
        __shared__ float shared_sum[BLOCK_SIZE];
        shared_sum[tid] = local_sum;
        __syncthreads();
    
        // Tree reduction: O(log BLOCK_SIZE) steps.
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
            if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
            __syncthreads();
        }
    
        // 3. Compute the rescale factor on thread 0; broadcast via shared memory.
        __shared__ float inv_rms;
        if (tid == 0) {
            float mean_sq = shared_sum[0] / (float)D;
            inv_rms = rsqrtf(mean_sq + eps);
            rstd[row] = inv_rms;            // save for backward
        }
        __syncthreads();
    
        // 4. Apply rescale + weight; write out. Coalesced.
        for (int i = tid; i < D; i += BLOCK_SIZE) {
            float v = (float)row_x[i] * inv_rms * (float)weight[i];
            row_y[i] = (scalar_t)v;
        }
    }
    
    // Host-side launcher
    torch::Tensor rmsnorm_fwd_cuda(
        torch::Tensor x, torch::Tensor weight, float eps,
        torch::Tensor rstd
    ) {
        auto y = torch::empty_like(x);
        int M = x.size(0), D = x.size(1);
        constexpr int BLOCK = 256;
    
        // Dispatch on dtype using AT_DISPATCH_FLOATING_TYPES_AND2.
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            x.scalar_type(), "rmsnorm_fwd", [&]() {
                rmsnorm_fwd_kernel<scalar_t, BLOCK><<<M, BLOCK>>>(
                    x.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    y.data_ptr<scalar_t>(),
                    rstd.data_ptr<float>(),
                    D, eps
                );
            });
    
        return y;
    }

Three things to call out from the kernel:

  1. **One block per row, BLOCK_SIZE threads per block**. With 256 threads = 8 warps per block, each block lives entirely on one SM and does its row's work end-to-end. No inter-block communication. _This pattern (one block per "outer" dimension) is ubiquitous._
  2. **Shared memory for the reduction**. The squares are summed locally, then a tree reduction in shared memory aggregates across threads. After the final `__syncthreads()`, every thread can read the same result.
  3. **fp32 accumulation, scalar_t I/O**. We accept bf16 or fp16 inputs (`scalar_t`) but accumulate sums in fp32 (`float local_sum`). This is the standard recipe for stable reductions in low precision (M14). The same pattern shows up in cuBLAS and FlashAttention.

### Thread/block layout, visualized

RMSNorm kernel: one block per row, threads stride across columns Input x: shape (M=4, D=2048) — 4 rows, 2048 cols each row 0: → Block 0 handles this entire row (256 threads cooperate) row 1: → Block 1 handles this entire row row 2: → Block 2 row 3: → Block 3 Inside Block 0 (256 threads on D=2048): each thread strides by 256 [0..255] [256..511] [512..767] [768..1023] [1024..1279] [1280..1535] [1536..1791] [1792..2047] thread 0 reads cols 0, 256, 512, 768, … thread 1 reads 1, 257, 513, … → coalesced loads

The thread-layout pattern is worth memorizing: **thread`tid` processes element `tid + k * BLOCK_SIZE` for k = 0, 1, 2, ...**. Why? _Coalesced memory access_. The 32 threads of a warp end up reading 32 consecutive columns at the same time, which the memory controller can serve as a single transaction. The naive alternative — thread 0 reads cols 0..7, thread 1 reads 8..15 — would be uncoalesced and ~10× slower.

## The C++ binding

The `.cpp` file is a thin wrapper. It declares the host-side launcher and exposes it to Python via PYBIND11.
    
    
    // rmsnorm.cpp
    #include <torch/extension.h>
    
    // Forward declaration of the function defined in rmsnorm.cu
    torch::Tensor rmsnorm_fwd_cuda(
        torch::Tensor x, torch::Tensor weight, float eps,
        torch::Tensor rstd);
    
    // User-facing forward: validates inputs, allocates rstd buffer, dispatches.
    std::tuple<torch::Tensor, torch::Tensor> rmsnorm_fwd(
        torch::Tensor x, torch::Tensor weight, float eps
    ) {
        TORCH_CHECK(x.is_cuda(), "x must be CUDA");
        TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
        TORCH_CHECK(x.dim() == 2, "x must be 2D");
        TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    
        auto rstd = torch::empty({x.size(0)},
                                  x.options().dtype(torch::kFloat32));
        auto y = rmsnorm_fwd_cuda(x, weight, eps, rstd);
        return {y, rstd};
    }
    
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("rmsnorm_fwd", &rmsnorm_fwd, "RMSNorm forward (CUDA)");
    }

Three patterns to internalize:

  * `TORCH_CHECK` for input validation. Failed checks raise Python `RuntimeError` with the message you provided. Always check device, contiguity, dtype, dimensions — kernels assume these and silent assumption violations are a debugging nightmare.
  * `x.options()` propagates device/dtype properties. `x.options().dtype(torch::kFloat32)` says "same device as x but fp32" — useful when allocating auxiliary buffers like `rstd`.
  * `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` is the standard incantation. `TORCH_EXTENSION_NAME` is a macro that gets the module name from the build system. `m.def("rmsnorm_fwd", &rmsnorm;_fwd, ...)` exposes the C++ function as a Python attribute.

## Loading and using from Python

Now the Python side. `cpp_extension.load` JIT-builds and loads the extension:
    
    
    # rmsnorm.py
    from torch.utils.cpp_extension import load
    
    _ext = load(
        name="rmsnorm_ext",
        sources=["rmsnorm.cu", "rmsnorm.cpp"],
        extra_cuda_cflags=["-O3"],
        verbose=True,         # first-time build prints nvcc/g++ output
    )
    
    def rmsnorm(x, weight, eps=1e-6):
        assert x.is_contiguous(), "x must be contiguous"
        y, rstd = _ext.rmsnorm_fwd(x, weight, eps)
        return y

First call to `load` compiles (~30 seconds for a small kernel; longer for bigger ones). Result is cached in `~/.cache/torch_extensions`; subsequent imports are instant. If you change the source, the cache is invalidated automatically by file hash.

For production / packaging, you'd use setuptools instead:
    
    
    # setup.py
    from setuptools import setup
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
    
    setup(
        name="rmsnorm_ext",
        ext_modules=[CUDAExtension(
            "rmsnorm_ext",
            sources=["rmsnorm.cu", "rmsnorm.cpp"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )],
        cmdclass={"build_ext": BuildExtension},
    )

Run `pip install .` to build and install. Now `import rmsnorm_ext` works without JIT cost. JIT for development; setuptools for shipping.

## Wiring through `torch.library`

The function above runs, but it's not a first-class PyTorch op. Autograd doesn't know about it; autocast won't auto-cast inputs; `torch.compile` can't trace through it cleanly. Time to register it properly. From M19:
    
    
    import torch
    from rmsnorm import _ext
    
    @torch.library.custom_op("my_lib::rmsnorm", mutates_args=())
    def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        # Pure-Python fallback for non-CUDA tensors
        rms = (x * x).mean(-1, keepdim=True).add(eps).rsqrt()
        return x * rms * weight
    
    @rmsnorm.register_kernel("cuda")
    def _rmsnorm_cuda(x, weight, eps):
        y, _rstd = _ext.rmsnorm_fwd(x.contiguous(), weight, eps)
        return y
    
    @rmsnorm.register_fake()
    def _rmsnorm_fake(x, weight, eps):
        return torch.empty_like(x)

Now `torch.ops.my_lib.rmsnorm(x, w, eps)` goes through the dispatcher tower (M19). On CUDA it hits the kernel; on CPU it falls back to the Python implementation. `torch.compile` can trace via the fake function. **Same dispatch machinery as built-in ops.**

## The backward — and the clever shortcut

Forward alone isn't enough for training. We need to register an autograd backward. The math: given `y = (x / rms) * weight`, with `rms = sqrt(mean(x²) + ε)`:
    
    
    ∂L/∂weight = sum_over_M(∂L/∂y * (x / rms))
    ∂L/∂x = (1/rms) * (∂L/∂y * weight - x * (1/D) * sum(∂L/∂y * weight * x * (1/rms²)))

Yes, that's the actual derivative. RMSNorm has a non-trivial backward because the rescale factor depends on every element of the input.

The clever shortcut: we already saved `rstd = 1/rms` per row in the forward. The backward kernel reads `rstd` instead of recomputing it — saves one full pass over the input. This is why production normalization kernels return _both_ the output and any auxiliary tensors needed by backward.

Skipping the full backward kernel for space (it's another `rmsnorm_bwd_kernel` with a similar shape — one block per row, fp32 accumulation, etc.). The autograd registration:
    
    
    def _rmsnorm_setup_context(ctx, inputs, output):
        x, weight, eps = inputs
        # We need to recompute rstd here for the public API; in real code,
        # return rstd from the kernel and pass it through.
        rstd = (x * x).mean(-1, keepdim=True).add(eps).rsqrt()
        ctx.save_for_backward(x, weight, rstd)
    
    def _rmsnorm_backward(ctx, grad_y):
        x, weight, rstd = ctx.saved_tensors
        grad_x, grad_weight = _ext.rmsnorm_bwd(grad_y, x, weight, rstd)
        return grad_x, grad_weight, None     # None for eps (non-tensor)
    
    torch.library.register_autograd(
        "my_lib::rmsnorm", _rmsnorm_backward,
        setup_context=_rmsnorm_setup_context,
    )

The op now has full autograd support. `torch.ops.my_lib.rmsnorm(x, w, eps).sum().backward()` works.

## Numerical testing: `gradcheck` is mandatory

Hand-derived backwards are notoriously bug-prone. Sign errors, missing terms, wrong reduction axes — any of these gives subtly wrong gradients that train the model in the wrong direction. PyTorch ships a tool to catch this: `torch.autograd.gradcheck`.
    
    
    import torch
    from torch.autograd import gradcheck
    
    # Use double precision — finite-diff gradcheck is too noisy in float
    x = torch.randn(4, 128, dtype=torch.float64, requires_grad=True, device='cuda')
    w = torch.randn(128, dtype=torch.float64, requires_grad=True, device='cuda')
    
    assert gradcheck(lambda x, w: torch.ops.my_lib.rmsnorm(x, w, 1e-6), (x, w))
    print("gradcheck PASSED")

`gradcheck` compares your analytical backward to numerical finite differences. For each input, it perturbs by ±ε, measures (f(x+ε) - f(x-ε))/(2ε), and checks against your registered backward. Tolerances are tight — sub-1e-6 relative error in fp64. Pass it before trusting your kernel.

From M6: this is exactly the same tool used to validate `torch.autograd.Function` custom autograd. The pattern is identical for `custom_op` with autograd.

## Common pitfalls

Common CUDA extension bugs and what they look like Bug| Symptom| Fix  
---|---|---  
Non-contiguous input| Wrong outputs, often subtle (right values in wrong places)| Call `x.contiguous()` in the wrapper, or assert contiguous  
Missing `__syncthreads()`| Race condition; output non-deterministic| Sync between phases of shared-memory access  
Wrong block/grid sizing| Crashes, illegal memory access| Use `cudaGetLastError()` to catch launch failures; print kernel error  
fp16/bf16 accumulation| Numerical drift, divergent training| Accumulate in fp32 (the kernel cast pattern); same recipe as M14  
Backward doesn't match forward| gradcheck fails; loss curves diverge from reference| gradcheck with `fast_mode=False` first; debug each input's gradient  
Forgot to pass autocast policy| Different precision in eager vs autocasted| For bf16/fp16 input the kernel handles it (via dispatch); ensure you accept those dtypes in `AT_DISPATCH`  
Memory layout mismatches| Crashes or garbage outputs| Match what your kernel expects; add `TORCH_CHECK` for shape/stride  
  
The single most useful debugging discipline: **run with`CUDA_LAUNCH_BLOCKING=1`**. Without it, kernel errors are async and reported many calls later, with the wrong stack trace. With it set, errors surface at the kernel that caused them.
    
    
    CUDA_LAUNCH_BLOCKING=1 python my_test.py

Slows things down substantially (kills async kernel queueing — M20). But for debugging, indispensable.

## When to write CUDA vs Triton

You've now seen what writing a CUDA kernel looks like end-to-end. _Most kernels you'll write should be in Triton (M24) instead._ Reasons:

  * **Triton is ~3× shorter.** The same RMSNorm in Triton fits in 30 lines, all Python. No dispatch templates, no `AT_DISPATCH` macros, no PYBIND11.
  * **Triton is autotuned.** The block sizes, num_warps, num_stages are searched automatically. The CUDA kernel above uses BLOCK=256 because that's a reasonable default — a Triton version would find the optimal value for your shape.
  * **Triton compiles to PTX too.** No performance penalty for the abstraction; Triton kernels run at near-cuBLAS speed for many ops. FlashAttention-2's reference implementation is in Triton.
  * **Inductor uses Triton.** When `torch.compile` generates fused kernels (M21), they're Triton. By using Triton yourself, you stay in the same ecosystem.

Reasons to use raw CUDA anyway: (1) extreme low-level control (e.g., async copies via `cp.async`, tensor core MMA intrinsics, warp specialization on Hopper). (2) Integrating with existing CUDA libraries (CUTLASS templates, cuBLAS, cuDNN). (3) Educational. For typical custom ops in research code, Triton is the right tool — and you've now seen the lower-level mechanics it sits on top of.

#### Q&A; — About CUDA extensions **Q:** Why is `cpp_extension.load` so slow on first call? **A:** It runs nvcc + g++. CUDA compilation is genuinely slow — for a kernel with templates and aggressive optimization, 30-60 seconds is normal. PyTorch caches the compiled `.so` in `~/.cache/torch_extensions` keyed by source content hash; subsequent imports are essentially free. If you're iterating fast on a kernel, set `verbose=True` to see exactly what nvcc is doing. **Q:** My kernel works in eager mode but breaks under `torch.compile`. Why? **A:** Most common: you registered with `custom_op` but didn't register a fake function. Without the fake, `torch.compile`'s tracer can't figure out output shapes during meta-tensor execution. Add `@op.register_fake()` with a function that returns an empty tensor of the correct shape and dtype. **Q:** When does writing my own CUDA kernel actually beat ATen / cuBLAS? **A:** Three scenarios. (1) Operations with no library equivalent (custom activations, exotic attention variants). (2) Fused operations where avoiding HBM round-trips is the win — e.g., layer_norm + linear + activation in one kernel. (3) Operations with shape patterns the library doesn't handle well (very small or very irregular). Don't reimplement matmul; cuBLAS will beat you. Don't reimplement softmax; ATen's is excellent. _Find the gap._ **Q:** How do I profile my custom kernel? **A:** Same tools as ATen kernels (M13). The PyTorch profiler captures it as a regular CUDA op. For deeper analysis, `nsys` and `ncu`: `nsys profile python script.py` gives a system-level timeline; `ncu --kernel-name your_kernel python script.py` gives per-kernel hardware counters (occupancy, memory bandwidth utilization, register usage). When tuning, `ncu` is the tool that tells you whether you're hitting the roofline. **Q:** What's `__restrict__`? **A:** A hint to the compiler: "no other pointer in this function aliases this one." In our kernel, `x`, `weight`, and `y` are guaranteed disjoint, so the compiler can keep more values in registers and avoid redundant loads. Without it, the compiler has to assume a write to `y[i]` might invalidate `x[j]` reads. Standard practice for performance-critical kernels. **Q:** Can my CUDA extension call cuBLAS or other libraries? **A:** Yes — link against them in the build args. `extra_ldflags=["-lcublas"]` in `cpp_extension.load`; in setuptools, add `libraries=["cublas"]` to the extension. Useful when you want a fused op that calls cuBLAS for the matmul portion and your kernel for surrounding ops. Wrap cuBLAS handles in a long-lived object — creating handles is expensive. 

## The complete picture (zoomed out)

You've built one custom kernel end-to-end. The pieces:

  1. **The CUDA kernel** in `.cu`: `__global__` function with thread/block indexing, shared memory reductions, fp32 accumulation. One block per row; threads stride for coalesced loads.
  2. **The C++ wrapper** in `.cpp`: input validation, allocates auxiliary buffers, dispatches to the CUDA launcher, exposes via PYBIND11.
  3. **The build** : JIT via `cpp_extension.load` for development, setuptools/CUDAExtension for shipping.
  4. **The Python wrapper** : `cpp_extension.load(...)` on import, function that calls into the extension.
  5. **The op registration** : `@torch.library.custom_op` with a Python fallback, `@register_kernel("cuda")` for the CUDA path, `@register_fake()` for compile/tracing, `register_autograd` for the backward.
  6. **The validation** : `gradcheck` in fp64 to verify backward correctness; numerical comparison against a reference implementation in fp16/bf16 to catch low-precision drift.

Six pieces, each individually simple. Together they make a kernel that's a first-class PyTorch op — autograd recognizes it, autocast handles it, `torch.compile` traces it, distributed training composes with it. **Everything you've learned about PyTorch internals from M19-M21 is what makes this possible.**

## Code Magnets: register a CUDA extension as a first-class op

You've compiled `_ext` via `cpp_extension.load` and it has a `my_op` function. Now register it through `torch.library` so autograd, compile, and dispatch all work. Three magnets are wrong choices.

Arrange the magnets into a complete registration.

@torch.library.custom_op("my_lib::my_op", mutates_args=()) def my_op(x: torch.Tensor) -> torch.Tensor: return x * x # CPU fallback @my_op.register_kernel("cuda") def _cuda_impl(x): return _ext.my_op(x.contiguous()) @my_op.register_fake() def _fake(x): return torch.empty_like(x) def _bwd(ctx, grad): return 2 * ctx.saved_tensors[0] * grad def _setup_ctx(ctx, inputs, output): ctx.save_for_backward(inputs[0]) torch.library.register_autograd("my_lib::my_op", _bwd, setup_context=_setup_ctx) torch.ops.my_lib.my_op = my_op my_op = torch.compile(my_op) def _cuda_impl(x): return _ext.my_op(x) # no .contiguous()

show solution
    
    
    @torch.library.custom_op("my_lib::my_op", mutates_args=())
    def my_op(x: torch.Tensor) -> torch.Tensor:
        return x * x          # CPU fallback
    
    @my_op.register_kernel("cuda")
    def _cuda_impl(x): return _ext.my_op(x.contiguous())
    
    @my_op.register_fake()
    def _fake(x): return torch.empty_like(x)
    
    def _setup_ctx(ctx, inputs, output): ctx.save_for_backward(inputs[0])
    def _bwd(ctx, grad): return 2 * ctx.saved_tensors[0] * grad
    torch.library.register_autograd("my_lib::my_op", _bwd, setup_context=_setup_ctx)

The traps:

  * `torch.ops.my_lib.my_op = my_op`: the registration framework manages this internally. Manual assignment is unnecessary and can break ABI invariants.
  * `my_op = torch.compile(my_op)`: wrapping the registered op in compile breaks the dispatch — compile is applied at the call site (e.g., wrapping the model), not at the op definition.
  * `def _cuda_impl(x): return _ext.my_op(x)` without `.contiguous()`: the kernel assumes contiguous input. If a non-contiguous tensor is passed (a transpose or slice), the kernel reads garbage. Always defensively call `.contiguous()` in the wrapper.

The full pattern: **custom_op decorator → register_kernel for each backend → register_fake for shape inference → register_autograd for backward.**

## Who does what?

Match each extension concept to its real role.

Concept

Real role

__global__ function

A. CUDA kernel entry point — runs on the device, called from host with `<<<grid, block>>>`.

__syncthreads()

B. Block-wide barrier — all threads in the block wait for each other.

AT_DISPATCH_FLOATING_TYPES_AND2

C. Generates type-specialized kernel calls for fp16/bf16/fp32 from one template.

cpp_extension.load

D. JIT-compiles the .cu/.cpp sources, caches the .so, returns a Python module.

torch.library.custom_op

E. Registers the function as a first-class op participating in the dispatcher.

register_fake

F. Provides a meta-tensor implementation for compile/tracing — shape only.

gradcheck

G. Validates analytical backward against finite differences. Run in fp64.

show solution

**__global__ function** → A  
**__syncthreads()** → B  
**AT_DISPATCH_FLOATING_TYPES_AND2** → C  
**cpp_extension.load** → D  
**torch.library.custom_op** → E  
**register_fake** → F  
**gradcheck** → G 

The mental shortcut: ___global__ is the kernel, __syncthreads is the block barrier, AT_DISPATCH_FLOATING_TYPES_AND2 = type-dispatch macro, cpp_extension.load = JIT build, custom_op = register, register_fake = compile compatibility, gradcheck = backward validation in fp64_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team's hand-written CUDA RMSNorm kernel works for fp32 inputs but produces garbage for bf16. The Python tests for fp32 pass; bf16 fails silently with reasonable-but-wrong values. What's the most likely cause?

show answer

Accumulating in bf16 instead of fp32. Summing thousands of squared values with bf16's 7-bit mantissa is numerically unstable — small contributions get rounded away when added to a larger running sum. The standard recipe (which our kernel uses): inputs in `scalar_t` (bf16/fp16/fp32), but the reduction accumulator is always `float`. The cast is explicit: `float v = (float)row_x[i]`, `local_sum += v * v`. This is a direct application of M14: low precision for storage and matmul, fp32 for reductions and norms. Skipping the fp32 accumulation is the most common bug in custom normalization kernels.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why does the kernel use stride loops (`for (int i = tid; i < D; i += BLOCK_SIZE)`) instead of a fixed range per thread (`for (int i = tid * (D / BLOCK_SIZE); i < (tid + 1) * (D / BLOCK_SIZE); i++)`)?

show answer

Memory coalescing. The stride pattern means thread 0 reads index 0, thread 1 reads index 1, ..., thread 31 reads index 31 — 32 consecutive elements at the same time. The memory subsystem can serve those as a single transaction (one coalesced read of 32 × dtype bytes). With the contiguous-range pattern, thread 0 reads indices 0..7, thread 1 reads 8..15, etc. — at any given moment, the 32 threads of a warp are reading indices spread by 8 each, which is uncoalesced and ~10× slower. **Stride-by-block is the standard pattern for a reason.** It looks weird until you remember the 32-threads-of-a-warp execute the loop iterations in lockstep.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** A team's custom kernel "works" but their loss curve diverges slightly from the PyTorch reference implementation. `gradcheck` passes in fp64. What might still be wrong?

show answer

`gradcheck` runs in fp64 — your analytical backward is correct in exact arithmetic. But in fp16/bf16 the reduction order matters: a different summation tree gives slightly different results from the reference, accumulating over many steps into a visible curve divergence. Diagnosis: compare outputs op-by-op in fp16 against the reference. If individual outputs differ by ~1e-3 relative, that's normal numerical noise; if they differ by ~1e-1, something is wrong (e.g., accumulator dtype, or a computation simplification that's mathematically equivalent but numerically different). The standard fix: ensure your reduction order matches the reference, or accept the difference if it's within "stochastic noise" of training.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Sketch what the `register_fake` function should do for a custom op that performs `(B, M, K) × (B, K, N) -> (B, M, N)` batched matmul.

show answer
    
    
    @my_op.register_fake()
    def _my_op_fake(a, b):
        # a: (B, M, K), b: (B, K, N)
        assert a.size(0) == b.size(0), "batch mismatch"
        assert a.size(2) == b.size(1), "K mismatch"
        B, M, K = a.shape
        N = b.size(2)
        return a.new_empty((B, M, N))

Three things: (1) verify shapes (so compile-time tracing fails fast on bugs), (2) compute the output shape from inputs, (3) return an empty tensor of the right shape, dtype, and device using `a.new_empty(shape)` — which inherits a's dtype and device. The fake function never runs the actual kernel; it just produces meta-info for the FX graph.

### What just happened?

  * A CUDA extension is **three files** : `.cu` (kernel), `.cpp` (binding), `.py` (loader). `cpp_extension.load` JIT-compiles them; `setuptools + CUDAExtension` ahead-of-time builds.
  * The kernel pattern: `__global__` entry, thread/block indexing, shared memory for cooperation, `__syncthreads()` for barriers, fp32 accumulation for reductions.
  * **One block per "outer" dimension** , threads stride for coalesced loads. Standard layout for normalizations and per-row ops.
  * **Coalesced memory access** : thread `tid` reads element `tid + k*BLOCK_SIZE`. The 32 threads of a warp read 32 consecutive elements simultaneously — single memory transaction.
  * `AT_DISPATCH_FLOATING_TYPES_AND2` generates type-specialized kernel calls for fp16/bf16/fp32 from one template — same kernel handles multiple dtypes.
  * The C++ binding validates inputs (`TORCH_CHECK`), allocates auxiliary buffers (`x.options()`), and exposes via `PYBIND11_MODULE`.
  * **Register through`torch.library`**: `@custom_op` for the Python fallback, `@register_kernel("cuda")` for the CUDA path, `@register_fake()` for compile, `register_autograd` for backward. Now the op is first-class.
  * **`gradcheck` in fp64** is mandatory for hand-written backwards. Run it once, every time you change the backward.
  * Common pitfalls: non-contiguous inputs, missing syncs, fp16/bf16 accumulation, missing autograd registration, async kernel errors. `CUDA_LAUNCH_BLOCKING=1` for debugging.
  * **For most ops, use Triton (M24) instead**. Triton is shorter, autotuned, and integrates with Inductor. Raw CUDA when you need extreme control or library integration.
  * The reflex: a custom kernel is justified when _fusion_ would save HBM round-trips (memory-bound ops) or when you need an op that isn't in ATen. Don't reimplement matmul.

Module 24 takes the same RMSNorm and writes it in Triton. You'll see the same hardware concepts (blocks, shared memory, fp32 accumulation) but expressed in Python with autotuning baked in. After Triton, M25 walks through FlashAttention as the case study where all of this becomes load-bearing — tiles, shared memory, fused softmax, and the recompute trick that turns attention from memory-bound to compute-bound.
