# PyTorch: From Tensor to Kernel

# PyTorch, _From Tensor to Kernel_

_A CRASH COURSE • 28 MODULES_

— a Head-First-style climb from "what's a tensor?" to "I just wrote a fused attention kernel"

\--- 

This course is not a documentation reskin. PyTorch's official docs are excellent reference material and bad pedagogy. Reference material assumes you already know what you're looking for. Pedagogy walks you up the hill in the right order, points out the cliffs, and tells you why anyone bothered to climb in the first place.

The hill we're climbing is steep. The bottom is _"a tensor is a multi-dimensional array."_ The top is _"here is a fused FlashAttention-style kernel I wrote in Triton, and here's why my hand-tuned version beats the cuBLAS path on this shape."_ Most people who reach the top did it through a mixture of pain, blog posts, reading PyTorch source code, and one very patient senior engineer. This course is a synthesis of all of that.

> **★ KEY IDEA**  
>  This is a **climbing course** , not a reference manual. Each module assumes the previous one. If you skip Part II (autograd) and jump to Part VIII (kernels), you will be very confused about why your custom backward pass produces NaNs. Trust the order. 

## How each module is built

Every module follows the same Head First rhythm. You will see, in some order:

  * **Conversational explanation** — concepts introduced through dialogue, analogy, and worked examples. We talk to you, not at you.
  * **Code that runs** — every snippet is real PyTorch you can paste into a notebook. No pseudocode pretending to be code.
  * **Tradeoff tables** — when there are two ways to do something (and there always are), we put them side by side with the verdict.
  * **"No dumb questions"** — the questions you'd be embarrassed to ask in a meeting, answered without judgment.
  * **Brain Power boxes** — pause and think before reading the answer. The exercises matter.
  * **Sticky notes** — the kind of thing a senior engineer would scribble on a whiteboard and tell you to remember forever.
  * **Failure modes** — the exact ways things break in production, and how to spot them.

> **📝 NOTE**  
>  **Read this part:** the modules are dense. Resist the urge to skim. If a section has a Brain Power box, actually pause. The whole point of Head First style is that your brain remembers what it _worked for_ , not what it skimmed. 

## Course map

### Eight parts. Twenty-eight modules. One mountain.

PART I — Tensor Foundations

  1. **Tensors from scratch** — storage, stride, shape, dtype, device. Why `.contiguous()` exists.
  2. **Indexing, broadcasting & shape gymnastics** — view vs reshape vs permute, advanced indexing, einsum, gather/scatter.
  3. **Devices, dtypes & numerics** — fp32/fp16/bf16/fp8, pinned memory, async transfer, numerical-stability primitives.

PART II — Autograd, Deeply

  4. **What`.backward()` actually does** — the dynamic graph, leaf vs non-leaf, reverse-mode walked by hand.
  5. **Autograd in practice** — `requires_grad`, `detach`, `no_grad`, hooks, `torch.func`.
  6. **Custom autograd:`torch.autograd.Function`** — writing your own forward/backward, gradcheck, when it earns its keep.

PART III — Building Models

  7. **nn.Module deeply** — parameters vs buffers, hooks, state_dict, weight tying, the registration gotcha, parametrize.
  8. **Initialization, layers & norms** — Xavier vs Kaiming vs GPT-style init; Dropout, BatchNorm, LayerNorm, RMSNorm, GroupNorm.
  9. **Losses, optimizers & schedulers** — CE/BCE/focal, SGD/Adam/AdamW/Lion/Muon math, cosine + warmup, gradient clipping.

PART IV — Data & Training Loops

  10. **Datasets & DataLoaders done right** — map vs iterable, samplers, num_workers, prefetch, collate, pinned memory.
  11. **Training loops you can trust** — overfit-a-batch, checkpoint/resume, accumulation, EMA, deterministic vs fast.

PART V — Performance & Debugging

  12. **Memory: where does it all go?** — params + grads + optimizer + activations, the transformer memory formula, checkpointing.
  13. **Profiling & finding bottlenecks** — torch.profiler, Nsight, the async kernel model, the "GPU is idle" diagnosis tree.
  14. **Mixed precision & numerics** — autocast, GradScaler, fp16 vs bf16, NaN-hunting.

PART VI — Distributed Training

  15. **Communication primitives** — all-reduce, all-gather, reduce-scatter, ring vs tree, NCCL basics.
  16. **DDP from first principles** — gradient buckets, overlap with backward, find_unused_parameters, scaling rules.
  17. **Sharding: ZeRO and FSDP** — ZeRO-1/2/3, wrapping policies, mixed precision, CPU offload, activation memory.
  18. **Tensor & pipeline parallelism** — Megatron-style TP, 1F1B and interleaved PP, sequence parallelism, 3D parallelism.

PART VII — Compilation & Internals

  19. **PyTorch internals: dispatcher, ATen, storage** — how `a + b` actually executes, dispatch keys, views and aliasing.
  20. **CUDA semantics & the caching allocator** — streams, events, async kernels, the memory pool, fragmentation, CUDA graphs.
  21. **torch.compile deep dive** — TorchDynamo, AOTAutograd, Inductor, graph breaks, debugging compile errors.

PART VIII — Kernel Writing & Frontiers

  22. **The GPU programming model** — SMs, warps, threads, shared memory, coalescing, occupancy, why this matters.
  23. **Custom C++ / CUDA extensions** — writing a fused-op kernel, pybind11, integrating with autograd.
  24. **Triton: kernels in Python** — block-based programming, softmax / matmul / layernorm kernels, autotuning.
  25. **FlashAttention as a case study** — IO-aware design, online softmax, tiling for SRAM, recomputation, building one in Triton.
  26. **Quantization: PTQ, GPTQ, AWQ** — int8/int4 math, calibration, QAT, the dequant fusion trick, KV-cache quantization.
  27. **Inference & serving** — KV cache layout, paged attention, continuous batching, speculative decoding, CUDA graphs.
  28. **MoE & frontier patterns** — top-k routing, expert parallelism, dropless MoE, capstone: a tiny MoE with FSDP + custom kernel.

## Who this is for

#### This works for you if…

  * You can read Python and have written some PyTorch (even if you copy-pasted half of it).
  * You're comfortable with linear algebra (matrices, vectors, dot products) and basic calculus (chain rule).
  * You want to _understand_ PyTorch, not just use it.
  * You have access to a GPU (Colab works for the early modules; you'll want a real one by Part V).

#### This is the wrong book if…

  * You've never written Python.
  * You want a "build a chatbot in 3 lines" tutorial. You'll be miserable here.
  * You only need to _use_ a model, not understand the framework. Hugging Face Transformers' docs are better for that.
  * You hate code. We are about to read a _lot_ of code.

## How to use the course

Three modes work, depending on where you're starting:

#### Q&A; — Reading paths **Q:** I'm new to PyTorch. Where do I start? **A:** Module 1, in order, no skipping. The first six modules build the foundation everything else stands on. Many "advanced" bugs are actually misunderstandings of Module 4 (autograd) or Module 12 (memory). Build the base. **Q:** I've trained models for years. Can I jump ahead? **A:** Skim Parts I–IV to fill gaps (most engineers have at least one — usually around custom autograd or DataLoader internals). Read Parts V–VI properly. Parts VII–VIII are where most experienced engineers find new things. **Q:** I just want to write kernels. **A:** You can't, yet. Or rather, you can write a kernel — but you can't write one that's _correctly integrated with PyTorch's autograd, doesn't break under torch.compile, and beats the existing kernel on memory-bound shapes_. That's what Part VIII is for. You'll need at least Modules 4, 6, 12, 19, and 20 first. Sorry. The shortcut isn't a shortcut. **Q:** How long should this take? **A:** If you read one module a day and do the exercises, about a month. If you skim, a week — but you won't remember it. The whole structure of Head First is built around your brain's tendency to drop anything it didn't _work_ for. 

## A note on PyTorch versions

PyTorch is a fast-moving target. This course targets PyTorch 2.x (2.0+, ideally 2.3+). If you're on 1.x, much of Parts I–IV still applies, but anything mentioning `torch.compile`, FSDP-2, or modern Triton integration will be different or absent. Upgrade.

API names occasionally shift. We'll flag the ones likely to change. The _concepts_ — autograd, the dispatcher, the GPU programming model, IO-aware kernel design — do not change.

> **📝 NOTE**  
>  **One last thing.** Reading this course will not make you good at PyTorch. _Doing the exercises and shipping projects_ will. Build something at the end of every part. Train something. Profile something. Break something. The book is scaffolding; you do the climbing.
