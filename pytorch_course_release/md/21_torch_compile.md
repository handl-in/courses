# Module 21 — torch.compile: tracing, fusing, generating

# `torch.compile`: _tracing, fusing, generating_

_Part VII · Module 21_

— how Dynamo turns your Python into FX graphs, AOTAutograd glues forward and backward together, and Inductor emits Triton kernels that collapse the dispatcher tower

\--- 

You've now seen everything compile competes with: the per-op dispatcher tower from M19, the CUDA stream queue and graph machinery from M20. `torch.compile` is the system that _collapses_ the per-op overhead into traced, fused, generated kernels — typically 1.3-2× speedup with one line, sometimes 3-5× when the eager version was severely CPU-bound.

The mental model is simple even if the internals are intricate. Compile takes your Python function, _traces_ it once on representative inputs to learn what ops it runs (Dynamo). It composes those ops with their backwards into a single graph (AOTAutograd). Then it generates a fast C++/Triton kernel that runs the whole graph in one shot, skipping the dispatcher (Inductor). For static-shape regions, it can additionally wrap the result in a CUDA Graph (M20) for nearly-zero launch overhead.

> **★ KEY IDEA**  
>  `torch.compile(model)` is a four-stage pipeline. **Dynamo** intercepts Python execution and traces ops into an FX graph, hitting "graph breaks" at constructs it can't capture. **AOTAutograd** fuses the forward graph with its backward into a single joint graph. **Inductor** code-generates fused C++/Triton kernels from that graph. Optionally, **CUDA Graphs** (mode="reduce-overhead") wrap the result for static-shape replay. The user-facing rule: write normal Python; if it's clean and shape-stable, compile makes it fast. 

## One new face

D

Dynamo

"I watch your Python execute, op by op, and capture an FX graph of what you actually did."

Unlike older tracers, I don't need you to "make your code traceable." I hook CPython at the bytecode level — when your function runs, I see every op as it's dispatched and record it. If you write a `print()`, a Python conditional on a tensor's value, or call into a library I can't see through, I hit a **graph break** : I split your function into compiled chunks separated by the unsupported bit. Within each chunk, the rest of the pipeline (AOTAutograd, Inductor) takes over. _The fewer graph breaks you have, the more compile can fuse._

## The minimal API
    
    
    model = build_model()
    model = torch.compile(model)              # that's it
    
    for step, batch in enumerate(loader):
        out = model(batch.x)               # first call: ~5-30 sec compile time
        loss = criterion(out, batch.y)
        loss.backward()
        opt.step(); opt.zero_grad()

One line. The first forward pass triggers compilation (which takes seconds — sometimes tens of seconds for big models), and subsequent calls are fast. The backward is also compiled, even though you didn't wrap it explicitly: AOTAutograd captures the backward at the same time as the forward.

You can compile a function instead of a model:
    
    
    @torch.compile
    def fused_attention(q, k, v):
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        weights = torch.softmax(scores, dim=-1)
        return weights @ v

Compile traces the function on first call, then caches the compiled version. On subsequent calls with the same input shapes/dtypes, it skips straight to the compiled kernel.

## The four-stage pipeline

torch.compile pipeline: Python → fast kernel your Python def f(x): return x.relu().sum() trace ① Dynamo Python frame eval → FX graph \+ backward ② AOTAutograd forward + backward joint graph codegen ③ Inductor → Triton (GPU) → C++ (CPU) if mode= "reduce- overhead" ④ CUDA Graph wrap replay = ~0 launch cost FX graph (what Dynamo sees): x = placeholder() a = call_function(aten.relu, (x,)) b = call_function(aten.sum, (a,)) ... a clean DAG of ATen ops, no Python overhead Inductor's generated Triton kernel (excerpt): @triton.jit def fused_relu_sum(in_ptr, out_ptr, n): pid = tl.program_id(0) x = tl.load(in_ptr + pid*BLOCK + tl.arange(0, BLOCK)) tl.atomic_add(out_ptr, tl.sum(tl.maximum(x, 0))) # relu+sum FUSED

Three things to internalize:

  1. **Dynamo** hooks Python's frame evaluation. As your function runs, Dynamo records every tensor op into an FX graph — a small DAG of ATen ops. It does this transparently; your code looks the same.
  2. **AOTAutograd** takes the forward FX graph and runs autograd _once_ , ahead of time, to produce the backward graph. Forward and backward become a single joint graph the optimizer can analyze together — enabling fusions across the forward/backward boundary that eager mode can't see.
  3. **Inductor** compiles the joint graph to actual code. For CUDA, it generates Triton kernels (M24) that fuse multiple ATen ops into one kernel — what was relu-then-sum (two kernels in eager) becomes a single fused-relu-sum kernel that does both in one pass over memory.

The optional fourth stage: if you pass `mode="reduce-overhead"`, Inductor's output is wrapped in a CUDA Graph (M20). Per-launch overhead disappears for the captured region.

## What "fusion" means in practice

Eager mode runs each op as its own kernel:
    
    
    y = torch.relu(x)             # kernel 1: read x, write y
    z = y.sum()                  # kernel 2: read y, write z (scalar)

Two trips to memory. `x` is loaded, relu'd into `y`; then `y` is loaded again, summed into `z`. Memory bandwidth is the bottleneck for most pointwise ops, so a second memory pass means a second wall-clock cost.

Compile fuses them into one kernel that does both in a single pass:
    
    
    # Inductor generates:
    def fused_kernel(x_ptr, z_ptr):
        x = load(x_ptr)
        y = max(x, 0)             # relu
        z = reduce_sum(y)         # sum, all in registers, no memory write of y
        store(z_ptr, z)

One memory pass, no intermediate `y` in DRAM. That's a 2× speedup right there, and the savings compound when many pointwise ops chain together (which is the common case in transformers — residual + layer norm + linear + activation + dropout, etc.).

This is also why CPU-bound traces benefit so much from compile: not only does the per-op dispatcher overhead disappear, but the kernels themselves are fewer and bigger. _Both ends of M13's "GPU has gaps" problem get solved at once._

## Graph breaks: when Dynamo gives up locally

The thing every compile user has to learn: **graph breaks**. Dynamo can trace most PyTorch operations, but there are constructs it can't (or won't) capture:

  * **Data-dependent Python control flow** : `if x.sum() > 0: ...` requires syncing the GPU to evaluate the condition, then branches based on the value. Dynamo doesn't know which branch will run at compile time.
  * **Calls into untraceable Python** : arbitrary Python libraries, third-party C extensions Dynamo doesn't have integration for, `print()`, certain `numpy` operations.
  * **Mutating Python data structures** : appending to a list inside the function, modifying a dict that's defined outside.
  * **Calling`.item()`**, returning a Python scalar from a tensor.

When Dynamo encounters one, it splits the function: compile the part up to the break, run the unsupported bit in eager mode, compile the part after. Each compiled chunk is fast; each transition has overhead and prevents fusion across the break.

To diagnose graph breaks, set the explain flag:
    
    
    import torch._dynamo
    
    torch._dynamo.explain(my_func)(*args)
    # Prints: number of graphs, number of breaks, reason for each break, line numbers

Or use `fullgraph=True` to _require_ a single graph and error on any break:
    
    
    model = torch.compile(model, fullgraph=True)
    # Now any unsupported construct raises an error instead of silently splitting

For library code (research projects, training loops), `fullgraph=True` is a valuable discipline — it forces you to write code that compiles cleanly. For exploratory work, the default (allow breaks) is more forgiving.

### Common graph-break fixes

Graph-break causes and fixes Cause| Fix  
---|---  
`print(x)` for debugging| Remove or move outside compiled region. Use `torch._dynamo.config.verbose = True` for compile-time logging instead.  
`if x.item() > 0`: data-dependent branch| Use `torch.where`, masked ops, or accept the break (sometimes inevitable).  
`x.tolist()`, `x.numpy()`: leaves PyTorch land| Avoid in hot path. If unavoidable, accept the break.  
Custom `nn.Module` with Python-side state mutation| Move state to tensors; mutate via tensor ops.  
Calls into NumPy / SciPy| Most common cases (basic NumPy) compile fine in 2.4+; some don't. If it breaks, replace with PyTorch equivalents.  
Variable shapes| Set `dynamic=True` (see below) instead of accepting recompilation breaks.  
  
## Modes and what they do

`torch.compile` takes a `mode` argument:

The four compile modes Mode| What it does| When to use  
---|---|---  
`"default"`| Standard fusion, no CUDA Graphs, balanced compile time| Most workloads. The right starting point.  
`"reduce-overhead"`| Standard fusion + CUDA Graphs wrap| Static-shape inference, small models with launch overhead  
`"max-autotune"`| Aggressive Triton autotuning of every kernel + CUDA Graphs| Worth the long compile time when you'll run many steps  
`"max-autotune-no-cudagraphs"`| Same autotuning, no CUDA Graphs| Variable shapes that can't use CUDA Graphs but you still want autotuning  
  
The trade-off is compile time vs runtime speed. `"default"` compiles in seconds and gets ~80% of the available speedup. `"max-autotune"` can take minutes (and on huge models, tens of minutes) but extracts the last 20%. For long training runs (millions of steps), the autotuning amortizes; for short evaluations, it doesn't.

## Dynamic shapes

By default, compile generates a kernel specialized for the input shapes it sees. If you call the compiled function with different shapes — say batch size changes — it _recompiles_. After several recompiles, Dynamo gives up on specialization and starts compiling a more general (slightly slower) kernel that handles a range of shapes.

You can opt into dynamic-shape compilation upfront:
    
    
    model = torch.compile(model, dynamic=True)

This compiles a kernel that accepts a range of shapes from the start, eliminating recompilation. There's a small per-step overhead (the kernel doesn't get to specialize on, e.g., a fixed sequence length), but you avoid the recompile churn.

The right default for production: **start with dynamic=False (or unset), profile, and only switch to dynamic=True if you see frequent recompilation**. Recompilation events show up in `torch._dynamo.config.verbose = True` output as "recompiling" messages.

For inference with variable-length input (LLM serving), dynamic shapes are nearly always the right choice. For pretraining with fixed batch and seqlen, static is better.

## Compile + autocast + DDP/FSDP: the integrations

The standard production pattern: compile the model after wrapping it in DDP or FSDP, with autocast inside the model's forward.
    
    
    # DDP + compile
    model = build_model().to(device)
    model = DDP(model, device_ids=[local_rank])
    model = torch.compile(model)
    
    # FSDP2 + compile (modern)
    model = build_model().to(device)
    for block in model.transformer.layers:
        fully_shard(block)
    fully_shard(model)
    model = torch.compile(model)

For DDP: compile is applied _after_ the DDP wrap (the compile target is the wrapped module). DDP's hooks for gradient bucketing still fire correctly inside compiled code. For FSDP1, the integration is rougher — compile inside an FSDP1 model often graph-breaks at the FSDP unit boundaries. For FSDP2, the integration is clean and intended; `fully_shard` \+ `torch.compile` is the recommended path for new code.

Autocast also composes correctly:
    
    
    model = torch.compile(model)
    
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(x)
        loss = criterion(out, y)
    loss.backward()

The autocast context manager is captured by Dynamo and Inductor generates kernels at the right precision (bf16 for matmul, fp32 for reductions and softmax — same as eager).

## Common compile failures and how to debug them

### Failure 1: "Internal compile error"

Sometimes Inductor's codegen fails — usually for a combination of ops it doesn't yet handle well, or for an op with unusual shapes. The fallback:
    
    
    # Disable Inductor for this region; fall back to standard ATen kernels
    @torch.compile(backend="aot_eager")
    def f(x): ...

`backend="aot_eager"` uses AOTAutograd's joint graph but skips Inductor's codegen — runs each op via the standard dispatcher, just with the graph traced. Slower than full compile but useful when full compile breaks.

### Failure 2: "Recompilation triggered"

If you see frequent recompiles, your inputs have changing characteristics that compile is specializing on. The verbose output:
    
    
    torch._dynamo.config.verbose = True
    # Now compile prints reasons: "recompiling because input 0 has shape (32, 512) but cached shape was (32, 256)"

Fixes: use `dynamic=True`, or pad inputs to a small set of canonical shapes (bucketing).

### Failure 3: silent graph breaks

Subtle: compile silently breaks at unsupported constructs and you get less-than-expected speedup without realizing. Diagnose with:
    
    
    torch._dynamo.explain(model)(*example_inputs)
    # Reports: "1 graphs, 3 breaks, line 47: graph break in foo() ..."

Or run with `fullgraph=True` for a hard error.

### Failure 4: "compile is slower than eager"

Possible if your model is dominated by a single big kernel (e.g., one giant matmul) where eager already had near-zero dispatcher overhead. Compile's overhead in some cases (the wrapper, the bookkeeping) exceeds the negligible savings. Profile both versions; if eager is genuinely faster, leave compile off for that region.

More commonly: many graph breaks. Check with `explain`; fix the breaks.

## Eager vs compile: the timeline view

What does the speedup look like on a profiler trace? Roughly this:

Same training step: eager vs torch.compile Eager mode (~1.0× baseline) CPU GPU dispatcher tower per op (5-15 µs each, hundreds of ops) 12 small kernels with launch gaps; CPU dominates torch.compile (default mode, ~1.7× faster) CPU GPU few large fused-kernel launches fused kernel 1 fused kernel 2 fused kernel 3 3 fused kernels, packed end-to-end; CPU lane mostly idle same work, fewer kernels, less dispatcher tax → real speedup

Two changes visible in the lower panel: (1) the GPU kernels are bigger because Inductor fused multiple ATen ops into one. (2) The CPU lane goes from "constantly busy" (dispatcher) to "occasional brief launches" — most of the per-op overhead is gone. Both effects compound into the speedup.

#### About `torch.compile`

**Q:** When does compile help most? **A:** Three cases. (1) **CPU-bound traces** (M13): compile collapses the dispatcher tower, biggest wins. (2) **Models with many pointwise/fusable ops** : norm + linear + activation + dropout chains in transformers. (3) **Static shapes** with mode="reduce-overhead": CUDA Graphs add another layer of speedup. **Q:** When does compile NOT help? **A:** When you're _matmul-bound_ already (compile can't speed up cuBLAS), when you have lots of graph breaks, or when your kernels are already fused (FlashAttention, etc.). Profile before and after. **Q:** My first compile call takes 30 seconds. Is that normal? **A:** Yes. Compilation = tracing + autograd lowering + graph simplification + Triton kernel generation + autotuning. For a transformer model, 5-30 seconds is normal in `"default"` mode; `"max-autotune"` can push it to minutes. Subsequent calls hit a cache. PyTorch caches compiled artifacts in `~/.cache/torch/inductor` across runs (off by default in most setups; enable with `TORCHINDUCTOR_FX_GRAPH_CACHE=1` or via `torch._inductor.config.fx_graph_cache = True`). **Q:** Compile broke training — losses are different from eager. What now? **A:** Rare but real. Most often: a numerical edge case where Inductor's fused kernel computes things in a different order than eager (e.g., a softmax fused with surrounding ops uses different reduction patterns). Try (a) `backend="aot_eager"` to disable Inductor while keeping the joint graph — narrows whether Inductor or AOTAutograd is the cause. (b) Compare outputs op-by-op with hooks (M5). (c) If Inductor is the issue, file a GitHub issue with a minimal repro — these are bugs and the team fixes them. **Q:** What's the "FX graph"? **A:** FX is PyTorch's intermediate representation: a small DAG with placeholder nodes (inputs), `call_function` nodes (each ATen op), and output nodes. You can print one with `torch.fx.symbolic_trace`. Dynamo produces FX graphs as its output; Inductor consumes them. They're human-readable and useful for debugging — running `torch._dynamo.explain` dumps the FX graph for inspection. **Q:** Is compile compatible with custom autograd Functions (M6)? **A:** Generally yes — Dynamo treats them as opaque ops and includes them in the graph. They won't get fused with surrounding ops, but they don't break compilation. If you want full integration (your custom op being part of fusions), use `torch.library.custom_op` from M19 instead, which registers as a proper ATen op. 

## The minimal mental model

If you only remember three things from this module:

  1. **`torch.compile(model)` traces, fuses, and code-generates**. The result is one or a few large kernels instead of many small ones. Typical speedup: 1.3-2× with one line.
  2. **Graph breaks are the enemy of fusion**. `fullgraph=True` or `torch._dynamo.explain` finds them. Fix or accept.
  3. **Pick the mode based on your workload** : `"default"` for most things, `"reduce-overhead"` for static-shape inference, `"max-autotune"` for long training runs where compile time is amortized.

## Code Magnets: a clean compile setup

You're configuring compile for an FSDP-wrapped transformer with autocast, with a discipline of erroring on graph breaks. Three magnets are wrong choices.

Arrange the magnets into a working setup.

model = build_model().to(device) for block in model.transformer.layers: fully_shard(block) fully_shard(model) model = torch.compile(model, fullgraph=True) model = torch.compile(model, mode="reduce-overhead") model = torch.compile(fully_shard(model)) with torch.autocast('cuda', dtype=torch.bfloat16): out = model(x) loss = criterion(out, y) loss.backward() opt.step(); opt.zero_grad() model = torch.compile(model.cpu())

show solution
    
    
    model = build_model().to(device)
    for block in model.transformer.layers: fully_shard(block)
    fully_shard(model)
    model = torch.compile(model, fullgraph=True)
    
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(x)
        loss = criterion(out, y)
    loss.backward()
    opt.step(); opt.zero_grad()

The traps:

  * `torch.compile(model, mode="reduce-overhead")`: would also work, but with FSDP and dynamic-shape activations from variable-length inputs, CUDA Graphs often break. Default mode is safer for distributed training; switch to `reduce-overhead` only if you've confirmed shapes are static and FSDP integration is clean.
  * `torch.compile(fully_shard(model))`: compile _after_ all fully_shard calls. Wrapping the result of fully_shard inline like this works syntactically but obscures the order — easier to read and reason about as separate steps.
  * `torch.compile(model.cpu())`: compiling on CPU then trying to use on GPU — would re-compile or fail. Compile is device-aware; do it after `.to(device)`.

The order: **build → to(device) → fully_shard → compile → train with autocast**. `fullgraph=True` is a reasonable discipline once you've fixed the graph breaks; start without it for new models.

## Who does what?

Match each compile concept to its real role.

Concept

Real role

Dynamo

A. Hooks Python frame eval; captures tensor ops as an FX graph.

AOTAutograd

B. Composes the forward FX graph with backward into a single joint graph.

Inductor

C. Code-generates Triton kernels (CUDA) or fused C++ (CPU) from the joint graph.

Graph break

D. Dynamo splits the function at constructs it can't trace; transitions cost.

fullgraph=True

E. Errors on any graph break instead of silently splitting — discipline mode.

mode="reduce-overhead"

F. Wraps the compiled output in a CUDA Graph for static-shape replay.

dynamic=True

G. Compile a single shape-flexible kernel up front; avoids recompile churn.

show solution

**Dynamo** → A  
**AOTAutograd** → B  
**Inductor** → C  
**Graph break** → D  
**fullgraph=True** → E  
**mode="reduce-overhead"** → F  
**dynamic=True** → G 

The mental shortcut: _Dynamo traces, AOTAutograd joints, Inductor codegens, breaks split graphs, fullgraph=True forbids them, reduce-overhead = + CUDA Graphs, dynamic=True = shape-agnostic from start_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team's small transformer trains in 50 ms/step in eager. They wrap with `torch.compile` and now it's 20 ms/step. Profile both. What's the most likely shape of the speedup, and what would Exercise 1 from M13 (the CPU-bound case) say about why?

show answer

2.5× speedup on a small model is the textbook CPU-bound case from M13 — the eager trace had GPU gaps between many small kernels (dispatcher overhead per op was significant relative to kernel time). After compile: (a) fewer kernels because Inductor fused pointwise sequences, and (b) the dispatcher tower is gone for the compiled region. The GPU lane goes from "many small kernels with gaps" to "few large fused kernels packed end-to-end." The CPU lane goes from "constantly busy" to "occasional launches." This is the canonical compile success case — the smaller the model relative to dispatcher overhead, the bigger the win.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why does `fullgraph=True` sometimes catch bugs that compile silently swallows?

show answer

Without `fullgraph=True`, Dynamo silently breaks at any unsupported construct and continues. You get less-than-expected speedup, but no error. With `fullgraph=True`, it errors immediately, telling you exactly which construct broke. Common findings: (a) a debug `print` you forgot to remove, (b) a `tensor.tolist()` in the hot path, (c) a third-party library call inside forward, (d) Python control flow that secretly depends on tensor values. The discipline is: develop with `fullgraph=True` until your model compiles cleanly, then optionally relax for production. This is much like running with strict warnings enabled — catches issues that would otherwise compound silently.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** A team's training has slightly different loss numbers (~0.3% relative diff) when compiled vs eager. Should they worry?

show answer

Probably not, but verify. Sub-1% loss differences from compile are usually due to (a) different reduction order in fused kernels (e.g., a sum computed in a different order is bit-different but mathematically equivalent), (b) slightly different mixed-precision casting points (compile may cast at slightly different ATen-op boundaries than eager). Both are within the noise floor of stochastic optimization. To verify it's not a real bug: train two models from identical seeds, one compiled and one eager, for many steps. They should track each other within ordinary stochastic variance and converge to the same final quality. If the gap grows over time or final metrics differ meaningfully, that's a real divergence — try `backend="aot_eager"` to isolate Inductor as the cause.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** When would `torch.compile` hurt more than help?

show answer

Three legitimate cases:

  1. **Models dominated by one giant op** : a single 16K×16K matmul takes 50ms, dispatcher overhead is 10µs. Compile saves 10µs out of 50ms — invisible. The compile-time cost (seconds) isn't worth it for short runs.
  2. **Heavy graph-break code** : if your forward has many breaks (Python control flow, library calls, `.item()`), compile produces many small compiled chunks separated by eager-mode transitions. Sometimes net-slower than pure eager.
  3. **Variable-shape workloads with rapid recompilation** : if shapes change every step and you didn't set `dynamic=True`, you'll spend most of your time recompiling. The fix is `dynamic=True`; if even that has overhead exceeding eager, leave compile off.

The reflex: profile both. `torch.compile` is a tool, not a magic incantation. Most of the time it helps; sometimes it doesn't. M13's profiling skills are how you tell.

### What just happened?

  * `torch.compile(model)` is a four-stage pipeline: **Dynamo** traces Python into FX graphs, **AOTAutograd** joints forward+backward, **Inductor** code-generates fused C++/Triton kernels, optionally **CUDA Graphs** wrap the result.
  * **Fusion** means many ATen ops become one kernel — fewer memory passes, fewer dispatcher entries, fewer launches. Compounds into 1.3-2× speedup on typical models, more on CPU-bound traces.
  * **Graph breaks** are the main thing to manage. Dynamo splits the function at unsupported constructs (Python control flow on tensor values, `print`, `.item()`, third-party libs). Diagnose with `torch._dynamo.explain`; require single-graph with `fullgraph=True`.
  * Modes: **default** (start here), **reduce-overhead** (+ CUDA Graphs, static shapes), **max-autotune** (long compile, fastest runtime), **max-autotune-no-cudagraphs** (autotune without graph wrap).
  * **Dynamic shapes** : `dynamic=True` compiles shape-agnostic kernels up front. Good for variable-length workloads (LLM serving). Without it, varying shapes cause recompilation.
  * **Combines with autocast and DDP/FSDP2**. Order: build → to(device) → DDP/fully_shard → compile.
  * **FSDP1 + compile** has rough edges; FSDP2 + compile is the recommended modern path.
  * **First call is slow** — seconds to tens of seconds for trace + lower + codegen. Subsequent calls hit cache. Persistent disk cache via `TORCHINDUCTOR_FX_GRAPH_CACHE=1`.
  * **Failure modes** : internal codegen errors (fall back to `backend="aot_eager"`), recompilation churn (use `dynamic=True`), silent breaks (use `fullgraph=True`), slight numerical differences (usually fine, occasionally bugs).
  * **Cases compile won't help** : matmul-bound models, graph-break-heavy code, variable-shape without `dynamic=True`.
  * The reflex: try `torch.compile(model)` early. If speedup is good, keep it. If not, profile, find the breaks or the bottleneck, and decide.

Part VII is now complete. You have the dispatcher (M19), the CUDA stream and allocator machinery (M20), and the compile pipeline (this module) — three layers that together explain how every PyTorch op gets from Python to GPU and how to make that fast.

Part VIII is the kernel work. Module 22 starts with the GPU programming model itself: warps, SMs, the memory hierarchy, occupancy. We need this to know what we're _writing_ when we get to CUDA (M23) and Triton (M24). Then M25 — FlashAttention as a case study — pulls it all together. After that, M26 (quantization), M27 (inference/serving), M28 (MoE & frontier capstone).
