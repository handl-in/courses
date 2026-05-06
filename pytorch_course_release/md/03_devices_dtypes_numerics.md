# Module 03 — Devices, dtypes & numerics

# Devices, dtypes & _numerics_

_Part I · Module 03_

— what the bits actually mean, why your loss went NaN at step 4000, and the difference between "moved to GPU" and "actually moved to GPU"

\--- 

You've got the storage/stride model from Module 01 and the shape rules from Module 02. Now we get to the part that decides whether your code _just works_ or trains for nine hours and then dies: the numbers themselves.

Floating-point numbers are not real numbers. They're a finite, lumpy approximation that gets weirder near zero and weirder near infinity. And tensors don't just float around — they live on devices, get moved across PCIe, and pretend to be deterministic when they aren't. This module is about the parts of PyTorch where physics meets your code.

> **★ KEY IDEA**  
>  A dtype is a **contract about precision and range**. A device is a **physical location with a finite pipe to other devices**. The bugs in this module aren't logic bugs — they're _physics_ bugs. Loss goes NaN because numbers overflowed. Training is slow because you're shuffling bytes across PCIe one tensor at a time. Results aren't reproducible because the GPU runs reductions in a non-deterministic order. None of these are PyTorch's fault. They're how computers work. 

## Meet the float family

Three new characters today, plus a returning face. They will fight in subtle ways for the rest of the course.

F32

Float32

"I'm 4 bytes, 8 decimal digits of precision, and 38 orders of magnitude of range. I'm the default for a reason."

23 bits of mantissa, 8 bits of exponent, 1 sign bit. If you don't know what dtype you want, you want me. Training works in me. Gradients are stable in me. The only reason to leave is speed or memory — and both of those reasons are real, but you should leave deliberately, not by accident.

F16

Float16

"I'm 2 bytes, fast, and I overflow at 65,504."

10 bits of mantissa, 5 bits of exponent. Half the memory of Float32, twice the throughput on tensor cores. But my range is tiny — anything above 65,504 becomes `inf`, anything below 6e-5 underflows to zero. Gradient norms during training routinely exceed me. You need a _loss scaler_ to use me safely. Otherwise: NaN. We'll talk in Module 14.

BF16

BFloat16

"I'm Float16's smarter cousin. Same memory, fp32's range."

7 bits of mantissa (yes, fewer than Float16!), but 8 bits of exponent — same as Float32. So I overflow at the same gigantic value F32 does, around 3.4e38. I'm less _precise_ than F16, but that almost never matters in deep learning, where you've got noise everywhere anyway. I'm the modern training default on hardware that supports me (Ampere+, TPUs).

P

Pinned Memory

"I'm CPU memory that's promised the GPU I won't move."

Normally the OS can swap CPU memory pages around freely. That's bad for the GPU, which wants to DMA bytes directly without worrying they'll vanish mid-transfer. So you ask for _pinned_ (page-locked) memory, and the OS guarantees I stay put. The benefit: GPU transfers can be _async_ and overlap with compute. Without me, every transfer is a hidden synchronous copy.

## What the bits actually mean

Time to look inside a float. Skip this if you remember it from your numerical-methods class; otherwise read carefully because every "weird precision" issue you'll ever debug starts here.

A floating-point number is three fields: a sign bit, an exponent, and a mantissa (also called significand). The value is roughly:
    
    
    value = (-1)sign × 1.mantissa × 2exponent − bias

The number of mantissa bits sets your _precision_ (how finely you can resolve nearby numbers). The number of exponent bits sets your _range_ (how far you can go from zero in either direction). Trading bits between mantissa and exponent is the whole story of why we have so many small float types.

Same 16 bits, different deals Float32 — 32 bits total S exponent (8) mantissa (23) range ≈ ±3.4e38 · ε ≈ 1.2e-7 Float16 — 16 bits total S exp (5) mantissa (10) range ≈ ±65,504 · ε ≈ 9.8e-4 BFloat16 — 16 bits total S exponent (8) — same as fp32! mantissa (7) range ≈ ±3.4e38 · ε ≈ 7.8e-3 F16 trades range for precision — BF16 keeps range, drops precision

Look at that diagram. F16 and BF16 are the _same number of bits_ — but BF16 spends more of them on the exponent. That's the entire difference, and it's why BF16 dominates training while F16 is mostly relegated to inference and gradient tensors with explicit scaling.

The dtypes, with the only stats that matter Dtype| Bytes| Max value| Smallest normal| Decimal digits  
---|---|---|---|---  
`float64`| 8| ~1.8e308| ~2.2e-308| ~16  
`float32` (default)| 4| ~3.4e38| ~1.2e-38| ~7  
`bfloat16`| 2| ~3.4e38| ~1.2e-38| ~3  
`float16`| 2| **65,504**|  ~6.0e-5| ~3  
`float8 (e4m3)`| 1| 448| ~1.5e-3| ~1  
`float8 (e5m2)`| 1| 57,344| ~6.1e-5| ~1  
  
That tiny "65,504" for float16 is the entire reason mixed-precision training is hard. Imagine a forward pass that produces an intermediate value of 70,000 — you've already overflowed before any gradient computation begins. And once a single value in your tensor becomes `inf`, everything downstream becomes `NaN`, and training is over.

> **⚠ WARNING**  
>  **fp16 vs bf16 in one rule.** If your hardware supports bf16 (Ampere/A100/H100/TPU/MI200+), use bf16 for training. If you're stuck with fp16 (older GPUs), you need a GradScaler (Module 14) and you'll spend more time hunting NaNs. The mantissa precision difference (10 bits vs 7) almost never matters in deep learning. The exponent range difference (5 bits vs 8) almost always does. 

## Casting and the dtype hierarchy

You can move tensors between dtypes with `.to()`, `.float()`, `.half()`, `.bfloat16()`, and friends. The interesting question is what happens when you mix dtypes in a single op.
    
    
    a = torch.tensor([1.0], dtype=torch.float32)
    b = torch.tensor([1.0], dtype=torch.float16)
    
    c = a + b
    print(c.dtype)        # torch.float32 — promotes to the wider type
    
    i = torch.tensor([1], dtype=torch.int32)
    f = torch.tensor([1.0], dtype=torch.float32)
    print((i + f).dtype)  # torch.float32 — int promotes to float

PyTorch follows a type promotion table almost identical to NumPy's: when two dtypes meet, the result is the "wider" one in a careful sense (more range, then more precision). This is usually what you want, but it can sneak up on you in mixed-precision code: an fp16 activation added to an fp32 buffer becomes fp32, possibly costing you the speed win you were after.

Casting itself is cheap — it's a kernel that walks the tensor and converts each element. It's not free though, especially on big tensors. Aim to minimize unnecessary casts in hot loops.

#### Q&A; — About dtypes **Q:** If bf16 has the same range as fp32, why is anyone still using fp16? **A:** Three reasons. (1) Older GPUs (Pascal, Volta) only have fp16 tensor cores, no bf16. (2) Some research code is still written assuming fp16 + GradScaler, and porting is non-trivial. (3) For inference, fp16 has more precision than bf16 in the same byte count, which can matter for quality-sensitive workloads. **Q:** What is fp8 actually for? **A:** Inference acceleration on H100+ and MI300+ hardware that has dedicated fp8 tensor cores. It comes in two flavors — e4m3 (more precision, less range) for forward activations, and e5m2 (more range) for backward gradients. You don't typically work in fp8 by hand; libraries like TransformerEngine or torchao handle the casting and scaling. **Q:** My loss tensor is in fp32 but the model weights are fp16. Is that bad? **A:** It's actually good. The standard mixed-precision recipe is: _weights and activations in fp16/bf16, but loss and reductions in fp32_. Reductions like `sum` and `norm` are particularly bite-y in low precision because they accumulate many small errors. Module 14 has the full story. **Q:** When should I use float64? **A:** Almost never in deep learning. Two real use cases: (1) `torch.autograd.gradcheck` for verifying custom autograd functions (Module 6) — it requires fp64 to get reliable finite-difference comparisons. (2) Scientific simulations where you actually need the precision. Otherwise fp64 will halve your throughput for nothing. 

## Numerical stability primitives

Now we get to the operations that look correct but explode. The classic example is computing softmax naively:
    
    
    def naive_softmax(x):
        e = x.exp()
        return e / e.sum(dim=-1, keepdim=True)
    
    x = torch.tensor([1000.0, 2.0, 3.0])
    print(naive_softmax(x))   # tensor([nan, nan, nan])

What happened? `1000.exp()` overflows even fp32 (`e^1000` ≈ 10^434, way past 3.4e38), giving `inf`. Then `inf / inf` = `NaN`. The math is right, the implementation is broken.

The fix is to subtract the max _before_ exponentiating. Mathematically the result is identical (the constant cancels in numerator and denominator). Numerically, every exponent now lies in `(-∞, 0]`, so every `exp` lies in `(0, 1]`. No overflow.
    
    
    def stable_softmax(x):
        x = x - x.max(dim=-1, keepdim=True).values
        e = x.exp()
        return e / e.sum(dim=-1, keepdim=True)
    
    print(stable_softmax(torch.tensor([1000.0, 2.0, 3.0])))
    # tensor([1.0000e+00, 0.0000e+00, 0.0000e+00])

This trick — _subtract the max before exponentiating_ — appears all over deep learning. It's the reason real softmax implementations (yours, PyTorch's, FlashAttention's) are slightly different from the textbook formula.

### logsumexp: the numerical workhorse

You'll see this function constantly: `torch.logsumexp(x, dim)`. It computes `log(sum(exp(x)))`, but does it stably using the same max-subtraction trick. Why care? Because cross-entropy loss is essentially a `logsumexp`:
    
    
    # Cross-entropy of logits against targets, by hand:
    def my_cross_entropy(logits, targets):
        # logits: (B, V), targets: (B,)
        log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        return -log_probs.gather(1, targets[:, None]).squeeze(1).mean()

This is exactly what `F.cross_entropy` does internally. `logsumexp` is the numerically-stable building block.

> **📝 NOTE**  
>  **Whiteboard tattoo:** the pair "`x.exp()` \+ `sum` \+ `log`" is always wrong. Use `logsumexp`. Always. Even if it looks the same on small inputs, the exp-overflow lurks the moment your input has a big positive value. 

### Kahan summation and reduction order

Floating-point addition is not associative. `(a + b) + c` is not, in general, `a + (b + c)`. This bites you when summing many small values into a large running total — each addition loses some precision, and the errors compound.

The classic fix is _Kahan summation_ , which keeps a separate "compensation" term to track the lost bits. PyTorch doesn't expose Kahan-style sums by default (the cost is real), but the existence of the problem explains why:

  * Sums of large arrays in fp16 lose precision badly. `torch.sum(x, dim=..., dtype=torch.float32)` lets you accumulate in fp32 even if `x` is fp16.
  * Different reduction orders (e.g., parallel vs sequential) produce slightly different results. Two GPUs computing the "same" sum can disagree in the last few bits.
  * That non-determinism is also why `torch.use_deterministic_algorithms(True)` matters — and why it sometimes errors, because some kernels have no deterministic implementation.

    
    
    # Reduce in higher precision when accumulating fp16 tensors
    x = torch.randn(1_000_000, dtype=torch.float16)
    sum_low  = x.sum()                              # in fp16 — lossy
    sum_safe = x.sum(dtype=torch.float32)           # cast on the fly

## Devices: where the bytes actually live

A device is a physical location with its own memory and compute. The big three:

The devices you'll see Device| What it is| Notable quirks  
---|---|---  
`cpu`| Your computer's RAM and CPU cores| Always available. Ops use OpenMP or MKL. Slow for large tensors but fine for small ones.  
`cuda` / `cuda:0` / `cuda:1`| NVIDIA GPU memory + CUDA cores| Asynchronous launches. Caching allocator. 99% of PyTorch GPU code targets this.  
`mps`| Apple Silicon GPU (Metal Performance Shaders)| Available on M1/M2/M3 Macs. Coverage is improving but not 100% of ops are supported.  
`xpu`| Intel GPUs| Real but rare in practice today.  
`meta`| A "fake" device — shape only, no storage| Used for model construction without allocating memory. Module 7 again.  
  
Three things to internalize about devices:

  1. A tensor lives on exactly one device. Ops between tensors require both on the same device, or PyTorch raises a clear error.
  2. Moving tensors between devices costs bytes-per-second — the PCIe (or NVLink) bandwidth between CPU and GPU. For a typical PCIe 4.0 x16 link, you get about 25 GB/s in each direction. Move a billion-byte tensor and you've spent 40ms of pure transfer.
  3. GPU operations are _asynchronous_ by default. `x = x.cuda(); y = x + 1` queues two operations and returns immediately. The result isn't ready until you observe it (e.g., `print(y)`, `y.cpu()`, or `torch.cuda.synchronize()`).

∂

Autograd (with a warning)

"Async kernels mean profiling lies."

Because GPU ops queue up and return immediately, you can't time them with `time.time()`. The Python clock will say "0.001 seconds" while the GPU is still working. To measure correctly, call `torch.cuda.synchronize()` right before reading the timer. We'll cover the proper profiler in Module 13. For now: _don't trust naive timings on GPU._

## Pinned memory and async transfers

Now the punch line. The standard `x.to('cuda')` from a regular CPU tensor is _synchronous_ — Python waits while the bytes copy. That's a problem if your training loop is loading data on the CPU while the GPU is doing forward+backward. The CPU finishes loading, kicks off a transfer, the GPU sits idle waiting for the bytes. Then the GPU computes. Then it sits idle while the CPU loads the next batch. Pipeline stall.

The fix is _pinned memory_ \+ _non-blocking transfer_. Pinned memory is CPU memory that's locked in place, allowing the GPU's DMA engine to read from it without OS interference. With pinned memory, the transfer can proceed asynchronously while the GPU does other work.
    
    
    # Without pinning — transfer is synchronous, blocks Python
    x = torch.randn(1024, 1024)             # pageable CPU memory
    x_gpu = x.to('cuda')                  # Python waits here
    
    # With pinning + non-blocking — transfer overlaps with compute
    x = torch.randn(1024, 1024, pin_memory=True)
    x_gpu = x.to('cuda', non_blocking=True)   # returns immediately
    # ... GPU can now overlap this transfer with prior queued ops

The DataLoader handles this for you (Module 10) when you pass `pin_memory=True`. For one-off transfers, you do it yourself.

> **⚠ WARNING**  
>  **Pinned memory isn't free.** It comes from a smaller pool than regular pageable RAM, and excessive pinning can starve the OS. Don't pin everything — pin the things that get transferred to GPU repeatedly (data batches), not your one-off scratch tensors. 

### The async kernel model in one diagram
    
    
    CPU side:
      Python:  load_batch()  -->  x.to('cuda', non_blocking=True)  -->  model(x)  -->  ...
                                    |                                        |
                                    v                                        v
      GPU command queue:    [transfer_kernel]    [forward_kernel_1]   [forward_kernel_2] ...
                                    ↑                  ↑                     ↑
                              DMA from pinned     starts as soon as          continues
                              mem; overlaps       transfer is done           the queue
                              with prior work     ↑
                                                  this is where async wins:
                                                  GPU never stalls waiting
                                                  for the bytes

The whole point of pinned memory + non-blocking is to keep the GPU command queue full. As long as there's a kernel queued, the GPU isn't idle. Empty queue = GPU sleeping = wasted hardware.

## Code Magnets: build a fast move-to-GPU helper

You're writing a utility that takes a CPU tensor and gets it to GPU as efficiently as possible. The user might pass a regular tensor or a pre-pinned one; you want to do the right thing in both cases.

Arrange these magnets into a function body. There are two red herrings.

if not x.is_pinned(): x = x.pin_memory() x = x.contiguous() return x.to('cuda', non_blocking=True) return x.to('cuda') torch.cuda.synchronize()

show solution
    
    
    def to_gpu_fast(x):
        if not x.is_pinned():
            x = x.pin_memory()
        return x.to('cuda', non_blocking=True)

The traps:

  * `x.contiguous()` looks reasonable but is unrelated. `pin_memory()` already requires contiguous storage and will handle it.
  * `torch.cuda.synchronize()` would _defeat_ the whole purpose — it forces a wait, killing the async benefit. The caller may want to sync later, but this helper shouldn't.
  * `return x.to('cuda')` (no `non_blocking`) gives up the async win.

The check `if not x.is_pinned()` matters because pinning is expensive — calling it on already-pinned memory wastes a copy.

## Determinism: the train of compromises

You ran your model twice and got slightly different loss values. Are you losing your mind? No — you're meeting one of PyTorch's most important caveats.

True determinism on a GPU costs throughput. A non-deterministic `scatter_add` can run as a single fused kernel with atomic operations; a deterministic version has to serialize the writes. The default is non-deterministic. You opt in to the slower-but-reproducible path explicitly.

The full incantation:
    
    
    import torch
    import random
    import numpy as np
    
    def set_deterministic(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
        # cuDNN convolutions and similar — pick deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
        # PyTorch operator-level — error if a non-deterministic op is used
        torch.use_deterministic_algorithms(True)
    
        # Required env var for some CUDA kernels (set BEFORE importing torch in real code)
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

Three things to know about this:

  1. **Some ops have no deterministic implementation.** `torch.use_deterministic_algorithms(True)` will _raise an error_ if you call one. That's a feature — it tells you exactly where determinism breaks. Common offenders: `scatter_add` on CUDA, certain `index_put` patterns, some atomics.
  2. **Performance hits are real.** Convolutions can be 2× slower in deterministic mode. For research/debugging, that's fine. For production training, you usually accept non-determinism.
  3. **Cross-device determinism is harder still.** Two GPUs of the same model may produce slightly different results because of float-summation order in collective ops (Module 15). Even with all the flags, you can't generally guarantee bit-equality across hardware.

**JUNIOR:** I set `torch.manual_seed(42)` and my loss is _still_ different across runs. Why? **SENIOR:** You probably have a CUDA op that's non-deterministic. Add `torch.use_deterministic_algorithms(True)`; if anything errors, that's the culprit. **JUNIOR:** And if it does error? **SENIOR:** You either rewrite the offending op (sometimes a deterministic alternative exists), or accept that your training run isn't going to be bit-reproducible. You can usually still get _statistically_ reproducible results — same final accuracy, same general loss trajectory — just not bitwise. 

## Who does what?

Match each operation or flag to what it actually does.

Operation / flag

What it does

x.pin_memory()

A. Numerically stable computation of log(sum(exp(x))) without overflow.

torch.logsumexp(x, dim)

B. Locks CPU memory in place so the GPU can DMA from it asynchronously.

torch.use_deterministic_algorithms(True)

C. Tells cuDNN to pick deterministic algorithms for convolutions.

torch.cuda.synchronize()

D. Waits for all queued GPU work to finish before returning.

torch.backends.cudnn.deterministic = True

E. Errors when any non-deterministic PyTorch op is called.

x.to('cuda', non_blocking=True)

F. Queues an async transfer; only beneficial if x is in pinned memory.

show solution

**x.pin_memory()** → B  
**torch.logsumexp(x, dim)** → A  
**torch.use_deterministic_algorithms(True)** → E  
**torch.cuda.synchronize()** → D  
**torch.backends.cudnn.deterministic = True** → C  
**x.to('cuda', non_blocking=True)** → F 

The two determinism flags do different things: the cudnn one is about _which convolution algorithm_ (auto-tuned vs deterministic), while the use_deterministic_algorithms one is about _any op_ in the broader codebase. You usually want both for true reproducibility.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** Compute the smallest positive value representable in float16 vs bfloat16. Why is the bf16 version smaller?

show answer
    
    
    import torch
    print(torch.finfo(torch.float16).tiny)    # 6.103515625e-05
    print(torch.finfo(torch.bfloat16).tiny)   # 1.1754943508222875e-38

BF16 has 8 exponent bits to F16's 5, giving it a hugely larger range — both upward and downward. The _precision_ near zero is much worse in bf16 (because of the 7-bit mantissa), but you can _represent_ much smaller numbers without underflow.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Naive softmax overflows on large inputs. Will it also have problems with very _negative_ inputs (e.g., `[-1000, -1000, -1000]`)?

show answer

Yes, but a different problem: _underflow_. `(-1000).exp()` is approximately zero in any float type. Then `0 / 0` = NaN. The max-subtraction trick still saves you: subtracting `-1000` gives `[0, 0, 0]`, then `exp` gives `[1, 1, 1]`, then divide by 3 gives `[1/3, 1/3, 1/3]` — the correct uniform distribution.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** You're training in bf16 and your loss is computed by hand as `(logits.exp() / logits.exp().sum()).log() ...`. Even with the max trick, the loss is noisy. What's the fix?

show answer

Compute the loss in fp32 even though the model runs in bf16. Either use `F.cross_entropy(logits, targets)` (which handles dtype promotion internally) or upcast manually: `logits.float()` before the loss. The 7-bit mantissa of bf16 isn't precise enough for the small differences that matter at the loss tail.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** You time a GPU operation with `t0 = time.time(); model(x); t1 = time.time()` and get suspiciously fast results. What's the proper way?

show answer
    
    
    torch.cuda.synchronize()
    t0 = time.time()
    out = model(x)
    torch.cuda.synchronize()    # wait for queued work to finish
    t1 = time.time()

Without the syncs, you're timing the Python overhead of _queuing_ the kernels, not their actual execution. The first sync ensures any prior work is done so it doesn't bleed into your measurement; the second waits for your code's work to actually complete. For more rigorous benchmarking, use `torch.cuda.Event` with `event.elapsed_time` — Module 13.

### What just happened?

  * A **dtype** is a precision/range tradeoff. **fp32** is the safe default. **bf16** = same range as fp32, less precision — usually best for training. **fp16** = better precision than bf16 but tiny range (overflows at 65,504), needs a GradScaler. **fp8** is for inference on H100+ via specialized libraries.
  * Type promotion follows NumPy-like rules: mixing dtypes in an op promotes to the wider type. Watch for accidental upcasts in mixed-precision code.
  * **Naive softmax overflows.** Always subtract the max before exp. **Use`logsumexp`** instead of `log(sum(exp(x)))`. These tricks aren't optional — they're the difference between training and NaN.
  * Floating-point sums are **not associative**. Different reduction orders give different results in the last few bits. This is why GPU code is non-deterministic by default.
  * Use `x.sum(dtype=torch.float32)` to accumulate fp16/bf16 tensors in higher precision without an explicit upcast.
  * A tensor lives on **exactly one device**. Use `.to(device)`; `cpu`, `cuda`, `mps`, `meta` are the ones you'll see.
  * **GPU ops are asynchronous.** Python returns immediately. `torch.cuda.synchronize()` forces a wait. Naive `time.time()` measurements lie.
  * **Pinned memory + non-blocking transfer** = async CPU→GPU copy that overlaps with compute. Standard for DataLoader. Don't over-pin — it's a finite resource.
  * Reproducibility requires four things: seeds (Python/NumPy/torch/cuda), `cudnn.deterministic = True`, `torch.use_deterministic_algorithms(True)`, and the `CUBLAS_WORKSPACE_CONFIG` env var. It costs throughput. Use only when needed.
  * Bit-exact reproducibility across GPUs is essentially impossible. Statistical reproducibility (same final result, modulo last-decimal noise) is achievable.

That closes Part I. You now have the storage/stride model, the shape rules, and the dtype/device model. In Part II we go meet the framework that makes all of this _differentiable_ : autograd. Module 04 walks the dynamic graph by hand on a small example until `.backward()` stops being magic.
