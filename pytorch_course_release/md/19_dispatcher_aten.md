# Module 19 — Dispatcher, ATen & storage internals

# Dispatcher, ATen & _storage internals_

_Part VII · Module 19_

— what really happens between `torch.add(x, y)` in Python and the CUDA kernel that runs, walked through the dispatcher's layered tower

\--- 

You've used `torch.add`, `x.relu_()`, `F.cross_entropy`, and a hundred other ops for eighteen modules without ever asking what happens after Python returns control to PyTorch. This module lifts the hood. Not because you'll edit C++ tomorrow, but because every kernel we'll write in Parts VIII has to plug into this machinery — and the conceptual model is much simpler than its reputation.

The whole story has three layers. Python calls into a thin C++ wrapper. The _dispatcher_ looks at the tensor's properties (its device, whether it requires gradients, whether it's traced/compiled, etc.) and picks the right concrete implementation. The _ATen_ library is where those concrete implementations live — kernels for CPU, CUDA, MPS, and so on. Once we understand this routing, registering a custom kernel (Module 23) is just "add another entry to ATen's table."

> **★ KEY IDEA**  
>  The PyTorch dispatcher is a **layered tower** of dispatch keys. When you call an op, the dispatcher walks the tower from the top, peeling off one key per layer. The _Autograd_ layer wraps the call so backward gets recorded. The _AMP_ layer casts dtypes if you're inside autocast. The _Python_ layer routes to a Python-defined override if there is one. Eventually you hit a _backend_ key (CPU, CUDA, MPS) where a real kernel runs. The whole detour is a few microseconds, dispatched through a small lookup. **Custom kernels register at the backend layer; tools like autocast and torch.compile register at higher layers.**

## One new face

▶

Dispatcher

"I'm the air-traffic controller. Every tensor op asks me 'where do I go?' and I route based on a stack of keys."

When Python calls `x + y`, it lands in C++ and asks me which kernel to run. I look at `x`'s and `y`'s metadata: their dispatch keys. _Autograd? AMP? CUDA?_ I find the highest-priority key with an implementation registered for this op, and I run that. That implementation usually does its work and then re-dispatches at a lower key — peeling off layers. Eventually we hit a backend (CPU, CUDA) and a real kernel runs. Then the chain unwinds. Total overhead: a few microseconds. Total magic: zero.

## The journey of `torch.add(x, y)`

Let's trace one call all the way down. `x` and `y` are both float32 CUDA tensors with `requires_grad=True`, and we're inside an `autocast(dtype=bfloat16)` block.
    
    
    with torch.autocast('cuda', dtype=torch.bfloat16):
        z = torch.add(x, y)         # what actually happens?

This Python line triggers a chain. Each layer in the dispatcher tower does its own thing, optionally re-dispatching to the next layer, and the final result bubbles back up.

torch.add(x, y) descends the dispatcher tower Python: torch.add(x, y) enters C++ via pybind ① Autograd dispatch key wraps the op in a grad_fn (M4); records inputs; re-dispatches → lower key ② AMP / Autocast key checks autocast policy (M14); add is "promote" — keeps fp32; re-dispatches ③ Other keys (Functorch, Python, ADInplaceOrView…) most pass through unless something hooks them; re-dispatches → backend ④ CUDA backend key at::native::add_kernel_cuda — the real CUDA elementwise add launches a CUDA kernel; returns z return path unwinds layers total overhead: a handful of microseconds (≈ 5-15 µs eager) torch.compile collapses this whole tower into one fused C++ call (M21) custom kernels register at layer ④ — same path, just different ATen entry

Each layer's job in plain English:

  1. **Autograd** sees the inputs have `requires_grad=True` and wraps the op so backward will be recorded (M4). It re-dispatches with the autograd key removed — the lower layers don't need to think about gradients.
  2. **AMP** checks the autocast policy for `add`. `add` is in the "promote" list (M14) — it doesn't down-cast its inputs to bf16; it leaves them in their original precision. Other ops like `matmul` would down-cast here.
  3. **Other keys** — Functorch, Python, ADInplaceOrView, Tracer, etc. — usually pass through. They each exist to hook a specific feature (vmap, custom autograd, fx tracing). For a vanilla `add` none of them do anything.
  4. **CUDA backend** finally calls `at::native::add_kernel_cuda`, which launches an elementwise CUDA kernel. The kernel writes `z = x + y` into the output tensor. Returns up the chain.

Back up: layer 1 captures the result and registers the backward function. Python gets a tensor `z` with `grad_fn=AddBackward0`. Total wall-clock cost in eager mode: a few microseconds. **Most of that is dispatcher overhead, not kernel time** for tiny ops, which is why `torch.compile` (M21) flattens this whole tower into one fused C++ call.

## Dispatch keys, more concretely

A "key" is just an identifier. Each tensor carries a _dispatch key set_ : the union of keys that apply to it. The dispatcher uses these to decide what to run.

Some keys you'll hear about:

The dispatch keys that come up most Key| Set when…| What it does  
---|---|---  
`Autograd`| tensor has `requires_grad=True`| Records the op for backward; runs the original op via re-dispatch  
`AutocastCUDA`| inside `autocast('cuda', ...)`| Casts inputs per the autocast policy; re-dispatches  
`ADInplaceOrView`| tensor is being viewed or modified in-place| Manages version counters; integrity for autograd  
`Functorch`| inside `vmap`, `grad`, `jacrev`| Implements the functional transforms (M5)  
`Python`| tensor is a subclass with overrides| Calls the user's Python implementation (e.g., `__torch_function__`)  
`CUDA` / `CPU` / `MPS`| tensor lives on that device| Calls the actual backend kernel — bottom of the tower  
`Meta`| tensor is a "fake" tensor with no storage| Computes shape/dtype only; no kernel runs (used by `torch.compile`)  
  
Keys are _ordered_ by priority. Autograd is high (so it can wrap before backend kernels run). The backend keys are low (they're terminal — they run actual computation). The "interesting" middle keys (AMP, Functorch, Python) sit in between.

### The Meta key: an aside worth knowing

The `Meta` backend is special. A meta tensor has shape and dtype but _no actual data_. When the dispatcher hits the Meta backend, it runs a "shape function" that just figures out what the output tensor's shape and dtype would be — without doing any computation.

This is invaluable for two things. (1) `torch.compile` traces a model on meta tensors first to learn shapes without running anything. (2) Building large models without allocating memory: `with torch.device('meta'): model = build_model()`. The model exists, has the right structure and parameter shapes, but no weights are allocated. Useful when the full model wouldn't fit during construction (you'd later `.to_empty(device='cuda')` to materialize on GPU and then load weights).

## ATen: where the kernels actually live

"ATen" is the C++ tensor library inside PyTorch. Every `torch.add`, `torch.mm`, `F.relu` ultimately dispatches to a function in ATen. The op signatures are declared in `native_functions.yaml` (yes, a YAML file with thousands of entries). Each entry says "this op has these argument types, dispatches via these keys, and is implemented by these C++ functions."

A simplified entry looks like:
    
    
    # native_functions.yaml (simplified)
    - func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
      dispatch:
        CPU, CUDA: add_Tensor
        SparseCPU, SparseCUDA: add_sparse
        MkldnnCPU: mkldnn_add

This declares: `add` takes two tensors and a scalar alpha, returns a tensor. For dense CPU/CUDA tensors, it dispatches to `add_Tensor`; sparse tensors get a different implementation; MKLDNN-flavored tensors get yet another.

The build system code-generates the C++ glue from this YAML. The actual kernel implementations live in C++ files alongside it (`aten/src/ATen/native/`). When you "register a custom op" (M23), you're effectively adding rows to this dispatch table — though through a Python-friendly API.

#### Q&A; — About the dispatcher and ATen **Q:** Why is the dispatcher overhead such a big deal? **A:** For one big op (a 2K×2K matmul), the dispatcher overhead is invisible — the kernel takes milliseconds, the dispatch takes microseconds. For thousands of tiny ops (a transformer with hundreds of element-wise ops per layer), it adds up. `torch.compile` (M21) skips the per-op dispatch by tracing once and generating fused kernels — eliminating the tower for the trace. This is why "CPU-bound" traces (M13) often improve dramatically under compile. **Q:** Can I look at `native_functions.yaml` myself? **A:** Yes — it's in the PyTorch source tree at `aten/src/ATen/native/native_functions.yaml`. It's huge (~10k lines, many ops with many variants). When you wonder "what does this op actually call," reading the dispatch entry tells you. Look up `matmul`, `scaled_dot_product_attention`, `sum` to see the full pattern. **Q:** What's "structured kernels"? **A:** A pattern in ATen for ops that have a clean two-phase shape: a _meta function_ computes the output's shape and prepares the output tensor; an _impl function_ writes data into it. Splitting them lets the framework pre-allocate the output once, run the meta on the meta backend (for compile/trace), and run the impl on real backends. Most modern ops in ATen are structured. You don't need to write structured kernels until you're contributing to PyTorch core, but knowing the pattern helps when you read the source. **Q:** How does `torch.compile` interact with the dispatcher? **A:** Compile traces your model — usually on the Meta backend — building an FX graph (M21). It then converts that graph to optimized C++/Triton kernels via the Inductor backend. Each compiled fused region is registered as a single op that bypasses most of the dispatcher tower. The Autograd key still wraps it (so backward works), but AMP and other middle keys are absorbed into the trace. Net effect: the per-op dispatcher overhead vanishes for the compiled region. **Q:** What's the difference between `torch.add` and `torch.ops.aten.add`? **A:** The first is the user-facing API; the second is the direct ATen operator handle, which gives you slightly less overhead and is what most internal code uses. They eventually call the same kernel. `torch.ops` is what you'll see in FX graphs and compile internals. 

## Storage internals revisited

From M1: a tensor is a header (shape, stride, dtype, offset) pointing to a Storage of bytes. The dispatcher tower we just walked operates on tensor metadata. Once we hit the backend kernel, the kernel reads _storage bytes_ directly. Let's double-click on what that means.

### Empty tensors and the meta backend

A _meta tensor_ is a tensor with valid metadata but no Storage:
    
    
    x = torch.ones(3, 4, device='meta')
    print(x.shape)               # torch.Size([3, 4])
    print(x.dtype)               # torch.float32
    print(x.device)              # meta
    print(x.untyped_storage())   # empty / no actual bytes
    
    y = x + x                      # works! shape inference runs.
    print(y.shape)               # torch.Size([3, 4])
    # But y has no actual data — it's the shape, not the values.

This is what `torch.compile` uses internally during tracing. It also lets you build a 70B-param model on a single GPU without OOM:
    
    
    with torch.device('meta'):
        model = build_huge_model()      # no allocation; 0 GB used
    print(sum(p.numel() for p in model.parameters()))   # 70_000_000_000
    
    # Now materialize on real device
    model = model.to_empty(device='cuda')    # allocates uninitialized GPU memory
    load_pretrained_weights_into(model)   # load checkpoint shards

This pattern is the foundation of how huge models get loaded without "OOM during model init." FSDP and friends use it. You can use it directly when building anything where construction-time memory matters.

### The "_out" variants and out-of-place vs in-place

Many ops have three variants in ATen:

  * **Functional** : `torch.add(x, y)` → returns a new tensor.
  * **In-place** : `x.add_(y)` → modifies x, returns x.
  * **"_out" variant** : `torch.add(x, y, out=z)` → writes the result into a pre-allocated `z`.

The `_out` variant is what kernel writers care about. It separates "compute the result" from "allocate memory for it" — the framework can pre-allocate, the kernel just writes. This is the "structured kernels" pattern in action: shape inference + allocation happen in one place; the kernel implementation is decoupled.
    
    
    x = torch.randn(3, 4, device='cuda')
    y = torch.randn(3, 4, device='cuda')
    z = torch.empty_like(x)             # pre-allocate
    torch.add(x, y, out=z)              # write into z; no extra allocation

For very tight loops, the `_out` variants save the cost of one tensor allocation per call. Used heavily in optimizer code and custom kernels. For most user code, the regular form is clearer.

## Looking up an op in the wild

Suppose you want to know what `F.scaled_dot_product_attention` actually does. The path:

  1. Find the Python function: `torch.nn.functional.scaled_dot_product_attention` in `torch/nn/functional.py`. It's mostly a thin wrapper that calls into `torch._C._scaled_dot_product_attention(...)`.
  2. Look up the op in `native_functions.yaml`: search for `scaled_dot_product_attention`. You'll find an entry with `dispatch:` entries listing several backends (efficient_attention, flash_attention, math, etc.).
  3. Each backend points to a C++ function. Find that function in `aten/src/ATen/native/transformers/`. The flash and efficient attention paths call into bundled CUDA kernels.

That's the entire stack. From `F.scaled_dot_product_attention` in your training loop to a CUDA kernel: three jumps. **Every PyTorch op follows this pattern.**

## Custom op registration (preview of M23)

If you have a custom CUDA kernel and want to expose it as a first-class PyTorch op, you register it through ATen. The new (post-2.4) recommended API:
    
    
    import torch
    
    @torch.library.custom_op("my_lib::fast_relu", mutates_args=())
    def fast_relu(x: torch.Tensor) -> torch.Tensor:
        # Pure Python fallback — runs on CPU/all backends
        return torch.where(x > 0, x, torch.zeros_like(x))
    
    @fast_relu.register_kernel("cuda")
    def fast_relu_cuda(x: torch.Tensor) -> torch.Tensor:
        # Call your custom CUDA kernel here (via cpp_extension or torchscript)
        return my_extension.fast_relu_cuda(x)
    
    @fast_relu.register_fake()
    def fast_relu_fake(x):
        # Shape function — runs on Meta backend for compile/tracing
        return torch.empty_like(x)
    
    # Register a backward — so it works with autograd
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])
    
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * (x > 0).to(grad.dtype)
    
    torch.library.register_autograd(
        "my_lib::fast_relu", backward, setup_context=setup_context,
    )

Once registered: `torch.ops.my_lib.fast_relu(x)` goes through the dispatcher tower exactly like `torch.add`. Autograd records it, autocast handles dtype if needed, the dispatcher routes to your CUDA kernel on CUDA tensors and to your Python fallback on CPU. `torch.compile` can trace it via the fake function. **Custom ops aren't second-class — they slot into the same machinery as built-ins.**

We'll write a real CUDA kernel and wire it up in M23. For now, internalize that "registering a kernel" is "adding an entry to ATen's table that the dispatcher will find."

## One more dispatcher trick: `__torch_function__`

If you want to intercept _any_ tensor op for a particular tensor subclass, you can implement `__torch_function__`:
    
    
    class LoggingTensor(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if kwargs is None: kwargs = {}
            print(f"call: {func.__name__}")
            return super().__torch_function__(func, types, args, kwargs)
    
    x = LoggingTensor(torch.randn(3, 4))
    y = x + x        # prints: call: add
    z = y.sum()      # prints: call: sum

This hooks the dispatcher at the Python key. You see every op called on instances of your subclass before it descends to backend kernels. Useful for logging, validation, custom semantics, or building tensor-like abstractions (e.g., `DTensor` from M17 uses similar machinery internally). Most users never need this; it's worth knowing exists.

## The mental model, summarized

When you call any tensor op:

  1. Python lands in C++ via pybind11 wrappers.
  2. The dispatcher computes the dispatch key set from the inputs.
  3. It walks down the keys (highest priority first), running each one's implementation. Each implementation typically does its own thing (autograd records, autocast casts, etc.) and re-dispatches at a lower key.
  4. Eventually a backend key (CPU/CUDA/MPS/etc.) runs an actual kernel from ATen.
  5. The result bubbles back up; the higher-key implementations finalize their bookkeeping.

That's the entire framework's runtime. Once you've got it, the rest of Part VII (the CUDA stream model, the caching allocator, `torch.compile`) and Part VIII (CUDA, Triton, kernel-level work) all hook into specific points of this tower.

## Code Magnets: register a custom op end-to-end

You're registering a custom op `my_lib::scaled_relu(x, alpha)` = `alpha * relu(x)`. Need: Python fallback, CUDA kernel, fake function for compile, and autograd backward. Three magnets are red herrings.

Arrange the magnets into a complete registration.

@torch.library.custom_op("my_lib::scaled_relu", mutates_args=()) def scaled_relu(x: torch.Tensor, alpha: float) -> torch.Tensor: return alpha * torch.relu(x) @scaled_relu.register_kernel("cuda") def scaled_relu_cuda(x, alpha): return my_ext.scaled_relu_cuda(x, alpha) @scaled_relu.register_fake() def scaled_relu_fake(x, alpha): return torch.empty_like(x) def scaled_relu_kernel(x, alpha): return my_ext.fast(x, alpha) scaled_relu = torch.compile(scaled_relu) torch.ops.my_lib.scaled_relu = scaled_relu

show solution
    
    
    @torch.library.custom_op("my_lib::scaled_relu", mutates_args=())
    def scaled_relu(x: torch.Tensor, alpha: float) -> torch.Tensor:
        return alpha * torch.relu(x)
    
    @scaled_relu.register_kernel("cuda")
    def scaled_relu_cuda(x, alpha):
        return my_ext.scaled_relu_cuda(x, alpha)
    
    @scaled_relu.register_fake()
    def scaled_relu_fake(x, alpha):
        return torch.empty_like(x)

The traps:

  * `def scaled_relu_kernel(x, alpha): ...` alone — without the `@scaled_relu.register_kernel("cuda")` decorator, this is just a free function the framework never finds.
  * `scaled_relu = torch.compile(scaled_relu)` — wrapping the registered op in compile changes the object — the registration framework expects to track the original; compile would obscure that.
  * `torch.ops.my_lib.scaled_relu = scaled_relu` — manual assignment to the ops namespace. The decorator-based registration handles this for you; manually overwriting can break invariants.

For full autograd support you'd also call `torch.library.register_autograd(...)` separately, with a `setup_context` and `backward`. Skipping that means the op runs in forward but errors in backward — fine for inference-only ops; mandatory for training.

## Who does what?

Match each dispatcher / ATen concept to its real role.

Concept

Real role

Dispatcher

A. Routes a tensor op to the right implementation based on a stack of keys.

Autograd dispatch key

B. Wraps the op with backward-recording then re-dispatches without itself.

Meta backend

C. Computes shape/dtype only; no real allocation or computation.

ATen

D. The C++ tensor library; where every op's concrete kernels live.

native_functions.yaml

E. Declarative registry of every op's signature and per-backend implementations.

Out variants (op_out)

F. Pre-allocate the output; kernel writes into it without allocating.

__torch_function__

G. Subclass hook for intercepting any tensor op before it descends the dispatcher.

show solution

**Dispatcher** → A  
**Autograd dispatch key** → B  
**Meta backend** → C  
**ATen** → D  
**native_functions.yaml** → E  
**Out variants (op_out)** → F  
**__torch_function__** → G 

The mental shortcut: _dispatcher routes by keys, autograd key wraps for backward, meta computes shapes only, ATen has the kernels, YAML declares signatures, _out separates compute from allocation, __torch_function__ is the subclass hook_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team is profiling a tiny model with thousands of small element-wise ops per step. They notice their step time is dominated by Python — the GPU is idle most of the time. Explain in dispatcher terms what's happening, and propose two fixes.

show answer

For each tiny op, Python crosses into C++, descends the dispatcher tower (Autograd, AMP, Functorch, ADInplaceOrView, ..., CUDA), launches a kernel that takes microseconds, returns up the tower. The dispatcher overhead per op is ~5-15 µs. With thousands of ops per step, that's tens of milliseconds of pure overhead, with the GPU idle waiting for the next launch. This is the "CPU-bound" trace shape from M13.

Two fixes: (1) **`torch.compile`** traces the model once, fuses ops into larger kernels, eliminates per-op dispatch — typically 1.5-3× speedup on this pattern. (2) **CUDA Graphs** record a sequence of kernel launches once and replay the recording — eliminates per-launch overhead for static graphs. M20 covers CUDA graphs; M21 covers compile.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Build a 7B-param transformer on the meta device, then materialize it on CUDA without ever holding the full model on CPU. Sketch the code.

show answer
    
    
    with torch.device('meta'):
        model = build_transformer(n_params=7e9)        # 0 GB used
    
    # Materialize on CUDA with empty allocations
    model = model.to_empty(device='cuda')            # allocates GPU memory, weights are uninitialized
    
    # Now load pretrained weights, e.g., shard by shard
    sd = torch.load('shard_00.pt', weights_only=True)
    model.load_state_dict(sd, strict=False, assign=True)
    # OR initialize fresh:
    for p in model.parameters():
        nn.init.normal_(p, mean=0, std=0.02)

The `assign=True` flag in `load_state_dict` reuses the storage of the loaded tensors instead of copying them — important for huge models where you don't want to double-allocate. This pattern is the foundation of how FSDP and friends initialize at scale.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** Why does `torch.compile` care about the meta backend specifically?

show answer

Compile needs to _trace_ the model — build an FX graph of every op called — without running real computation. Real computation would force kernel launches, allocate memory, sync GPU, etc. — all expensive. The meta backend lets compile run the model symbolically: every op's shape function runs (so the FX graph has shape-correct nodes), but no actual kernel is launched and no memory is allocated. Compile then takes that FX graph and code-generates fused kernels via Inductor (M21). Without the meta backend, tracing huge models would either OOM or take forever.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** A teammate writes a custom autograd Function (M6) instead of using `torch.library.custom_op`. When does that matter?

show answer

Both produce ops the framework can backprop through, but they integrate differently:

  * `autograd.Function` (M6) creates a new _Python_ autograd node. `torch.compile` can usually trace through it but treats it as opaque — it can't fuse the body of your forward with surrounding ops. Good for one-off custom backwards (FlashAttention-style memory tricks).
  * `torch.library.custom_op` registers the op in ATen's dispatch table. The op gets a fake implementation (for tracing), an autograd registration (for backward), and proper compile/distributed integration. Better for ops you want to expose as first-class — including custom CUDA kernels you want compile to fuse around.

Rough rule: use `autograd.Function` for "I want a custom backward for these few lines"; use `torch.library.custom_op` for "I'm exposing a new operator with its own kernel."

### What just happened?

  * Every PyTorch op enters C++ via pybind, then descends the **dispatcher tower** : a stack of dispatch keys, each with its own implementation that re-dispatches at a lower key.
  * The keys, top to bottom: **Autograd** (records backward), **AMP/Autocast** (handles dtype policy), **Functorch / Python / ADInplaceOrView** (transform/subclass hooks), and **backend** keys (CPU, CUDA, MPS, Meta).
  * Backend keys are _terminal_ : they run a real kernel from **ATen** , the C++ tensor library.
  * `native_functions.yaml` declares every op's signature and per-backend dispatch. The build system code-gens C++ glue from it.
  * The **Meta backend** runs shape functions only — no allocation, no computation. Used by `torch.compile` for tracing and by users for "build huge model without OOM" (`torch.device('meta')` \+ `to_empty`).
  * `op_out` variants separate "compute" from "allocate" — kernel writes into pre-allocated output. Foundation of "structured kernels."
  * **Dispatcher overhead** is ~5-15 µs per op in eager mode. Negligible for big ops; dominates for many tiny ops. `torch.compile` (M21) and CUDA Graphs (M20) are the answers.
  * Custom ops register through `torch.library.custom_op`: define Python fallback, register backend kernel(s), register fake (shape function), register autograd backward. The op then participates fully in the dispatcher.
  * `__torch_function__` is the Python-level intercept for tensor subclasses — runs at the Python dispatch key, lets you observe or override every op.
  * **Custom ops aren't second-class** — they go through the same machinery as built-ins. The dispatcher doesn't care if a kernel was written by Anthropic or by you.
  * The reflex: when you wonder "what does this op actually do," the path is Python → dispatcher → ATen → backend kernel. Each step is documented; `native_functions.yaml` is the source of truth.

Module 20 zooms in on the CUDA backend specifically. The CUDA caching allocator (we touched on it in M12, now we look at how it works), CUDA streams and the async kernel queue, CUDA Graphs as a way to bypass dispatcher overhead, and the topology-aware allocations that show up in distributed training. The mental model from this module is the foundation; M20 fills in the GPU half.
