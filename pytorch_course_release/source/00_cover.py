#!/usr/bin/env python3
"""Cover page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">A CRASH COURSE • 28 MODULES</div>
  <h1 class="module-title">PyTorch, <em>From Tensor to Kernel</em></h1>
  <p class="module-sub">— a Head-First-style climb from "what's a tensor?" to "I just wrote a fused attention kernel"</p>
</div>

<p>This course is not a documentation reskin. PyTorch's official docs are excellent reference material and bad pedagogy. Reference material assumes you already know what you're looking for. Pedagogy walks you up the hill in the right order, points out the cliffs, and tells you why anyone bothered to climb in the first place.</p>

<p>The hill we're climbing is steep. The bottom is <em>"a tensor is a multi-dimensional array."</em> The top is <em>"here is a fused FlashAttention-style kernel I wrote in Triton, and here's why my hand-tuned version beats the cuBLAS path on this shape."</em> Most people who reach the top did it through a mixture of pain, blog posts, reading PyTorch source code, and one very patient senior engineer. This course is a synthesis of all of that.</p>

<div class="keyidea">
This is a <strong>climbing course</strong>, not a reference manual. Each module assumes the previous one. If you skip Part II (autograd) and jump to Part VIII (kernels), you will be very confused about why your custom backward pass produces NaNs. Trust the order.
</div>

<h2>How each module is built</h2>

<p>Every module follows the same Head First rhythm. You will see, in some order:</p>

<ul>
  <li><strong>Conversational explanation</strong> — concepts introduced through dialogue, analogy, and worked examples. We talk to you, not at you.</li>
  <li><strong>Code that runs</strong> — every snippet is real PyTorch you can paste into a notebook. No pseudocode pretending to be code.</li>
  <li><strong>Tradeoff tables</strong> — when there are two ways to do something (and there always are), we put them side by side with the verdict.</li>
  <li><strong>"No dumb questions"</strong> — the questions you'd be embarrassed to ask in a meeting, answered without judgment.</li>
  <li><strong>Brain Power boxes</strong> — pause and think before reading the answer. The exercises matter.</li>
  <li><strong>Sticky notes</strong> — the kind of thing a senior engineer would scribble on a whiteboard and tell you to remember forever.</li>
  <li><strong>Failure modes</strong> — the exact ways things break in production, and how to spot them.</li>
</ul>

<div class="sticky">
<strong>Read this part:</strong> the modules are dense. Resist the urge to skim. If a section has a <span style="background: #ffd5dc; padding: 2px 6px;">Brain Power</span> box, actually pause. The whole point of Head First style is that your brain remembers what it <em>worked for</em>, not what it skimmed.
</div>

<h2>Course map</h2>

<div class="toc">
  <h3>Eight parts. Twenty-eight modules. One mountain.</h3>

  <div class="part">PART I — Tensor Foundations</div>
  <ol start="1">
    <li><strong>Tensors from scratch</strong> — storage, stride, shape, dtype, device. Why <code>.contiguous()</code> exists.</li>
    <li><strong>Indexing, broadcasting & shape gymnastics</strong> — view vs reshape vs permute, advanced indexing, einsum, gather/scatter.</li>
    <li><strong>Devices, dtypes & numerics</strong> — fp32/fp16/bf16/fp8, pinned memory, async transfer, numerical-stability primitives.</li>
  </ol>

  <div class="part">PART II — Autograd, Deeply</div>
  <ol start="4">
    <li><strong>What <code>.backward()</code> actually does</strong> — the dynamic graph, leaf vs non-leaf, reverse-mode walked by hand.</li>
    <li><strong>Autograd in practice</strong> — <code>requires_grad</code>, <code>detach</code>, <code>no_grad</code>, hooks, <code>torch.func</code>.</li>
    <li><strong>Custom autograd: <code>torch.autograd.Function</code></strong> — writing your own forward/backward, gradcheck, when it earns its keep.</li>
  </ol>

  <div class="part">PART III — Building Models</div>
  <ol start="7">
    <li><strong>nn.Module deeply</strong> — parameters vs buffers, hooks, state_dict, weight tying, the registration gotcha, parametrize.</li>
    <li><strong>Initialization, layers & norms</strong> — Xavier vs Kaiming vs GPT-style init; Dropout, BatchNorm, LayerNorm, RMSNorm, GroupNorm.</li>
    <li><strong>Losses, optimizers & schedulers</strong> — CE/BCE/focal, SGD/Adam/AdamW/Lion/Muon math, cosine + warmup, gradient clipping.</li>
  </ol>

  <div class="part">PART IV — Data & Training Loops</div>
  <ol start="10">
    <li><strong>Datasets & DataLoaders done right</strong> — map vs iterable, samplers, num_workers, prefetch, collate, pinned memory.</li>
    <li><strong>Training loops you can trust</strong> — overfit-a-batch, checkpoint/resume, accumulation, EMA, deterministic vs fast.</li>
  </ol>

  <div class="part">PART V — Performance & Debugging</div>
  <ol start="12">
    <li><strong>Memory: where does it all go?</strong> — params + grads + optimizer + activations, the transformer memory formula, checkpointing.</li>
    <li><strong>Profiling & finding bottlenecks</strong> — torch.profiler, Nsight, the async kernel model, the "GPU is idle" diagnosis tree.</li>
    <li><strong>Mixed precision & numerics</strong> — autocast, GradScaler, fp16 vs bf16, NaN-hunting.</li>
  </ol>

  <div class="part">PART VI — Distributed Training</div>
  <ol start="15">
    <li><strong>Communication primitives</strong> — all-reduce, all-gather, reduce-scatter, ring vs tree, NCCL basics.</li>
    <li><strong>DDP from first principles</strong> — gradient buckets, overlap with backward, find_unused_parameters, scaling rules.</li>
    <li><strong>Sharding: ZeRO and FSDP</strong> — ZeRO-1/2/3, wrapping policies, mixed precision, CPU offload, activation memory.</li>
    <li><strong>Tensor & pipeline parallelism</strong> — Megatron-style TP, 1F1B and interleaved PP, sequence parallelism, 3D parallelism.</li>
  </ol>

  <div class="part">PART VII — Compilation & Internals</div>
  <ol start="19">
    <li><strong>PyTorch internals: dispatcher, ATen, storage</strong> — how <code>a + b</code> actually executes, dispatch keys, views and aliasing.</li>
    <li><strong>CUDA semantics & the caching allocator</strong> — streams, events, async kernels, the memory pool, fragmentation, CUDA graphs.</li>
    <li><strong>torch.compile deep dive</strong> — TorchDynamo, AOTAutograd, Inductor, graph breaks, debugging compile errors.</li>
  </ol>

  <div class="part">PART VIII — Kernel Writing & Frontiers</div>
  <ol start="22">
    <li><strong>The GPU programming model</strong> — SMs, warps, threads, shared memory, coalescing, occupancy, why this matters.</li>
    <li><strong>Custom C++ / CUDA extensions</strong> — writing a fused-op kernel, pybind11, integrating with autograd.</li>
    <li><strong>Triton: kernels in Python</strong> — block-based programming, softmax / matmul / layernorm kernels, autotuning.</li>
    <li><strong>FlashAttention as a case study</strong> — IO-aware design, online softmax, tiling for SRAM, recomputation, building one in Triton.</li>
    <li><strong>Quantization: PTQ, GPTQ, AWQ</strong> — int8/int4 math, calibration, QAT, the dequant fusion trick, KV-cache quantization.</li>
    <li><strong>Inference & serving</strong> — KV cache layout, paged attention, continuous batching, speculative decoding, CUDA graphs.</li>
    <li><strong>MoE & frontier patterns</strong> — top-k routing, expert parallelism, dropless MoE, capstone: a tiny MoE with FSDP + custom kernel.</li>
  </ol>
</div>

<h2>Who this is for</h2>

<div class="twocol">
  <div class="good">
    <h4>This works for you if…</h4>
    <ul>
      <li>You can read Python and have written some PyTorch (even if you copy-pasted half of it).</li>
      <li>You're comfortable with linear algebra (matrices, vectors, dot products) and basic calculus (chain rule).</li>
      <li>You want to <em>understand</em> PyTorch, not just use it.</li>
      <li>You have access to a GPU (Colab works for the early modules; you'll want a real one by Part V).</li>
    </ul>
  </div>
  <div class="bad">
    <h4>This is the wrong book if…</h4>
    <ul>
      <li>You've never written Python.</li>
      <li>You want a "build a chatbot in 3 lines" tutorial. You'll be miserable here.</li>
      <li>You only need to <em>use</em> a model, not understand the framework. Hugging Face Transformers' docs are better for that.</li>
      <li>You hate code. We are about to read a <em>lot</em> of code.</li>
    </ul>
  </div>
</div>

<h2>How to use the course</h2>

<p>Three modes work, depending on where you're starting:</p>

<div class="ndq">
<h4>Reading paths</h4>

<p class="q">I'm new to PyTorch. Where do I start?</p>
<p class="a">Module 1, in order, no skipping. The first six modules build the foundation everything else stands on. Many "advanced" bugs are actually misunderstandings of Module 4 (autograd) or Module 12 (memory). Build the base.</p>

<p class="q">I've trained models for years. Can I jump ahead?</p>
<p class="a">Skim Parts I–IV to fill gaps (most engineers have at least one — usually around custom autograd or DataLoader internals). Read Parts V–VI properly. Parts VII–VIII are where most experienced engineers find new things.</p>

<p class="q">I just want to write kernels.</p>
<p class="a">You can't, yet. Or rather, you can write a kernel — but you can't write one that's <em>correctly integrated with PyTorch's autograd, doesn't break under torch.compile, and beats the existing kernel on memory-bound shapes</em>. That's what Part VIII is for. You'll need at least Modules 4, 6, 12, 19, and 20 first. Sorry. The shortcut isn't a shortcut.</p>

<p class="q">How long should this take?</p>
<p class="a">If you read one module a day and do the exercises, about a month. If you skim, a week — but you won't remember it. The whole structure of Head First is built around your brain's tendency to drop anything it didn't <em>work</em> for.</p>
</div>

<h2>A note on PyTorch versions</h2>

<p>PyTorch is a fast-moving target. This course targets PyTorch 2.x (2.0+, ideally 2.3+). If you're on 1.x, much of Parts I–IV still applies, but anything mentioning <code>torch.compile</code>, FSDP-2, or modern Triton integration will be different or absent. Upgrade.</p>

<p>API names occasionally shift. We'll flag the ones likely to change. The <em>concepts</em> — autograd, the dispatcher, the GPU programming model, IO-aware kernel design — do not change.</p>

<div class="sticky pink">
<strong>One last thing.</strong> Reading this course will not make you good at PyTorch. <em>Doing the exercises and shipping projects</em> will. Build something at the end of every part. Train something. Profile something. Break something. The book is scaffolding; you do the climbing.
</div>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">★</span>
  <span>Cover</span>
</div>
"""

emit("00_cover", "PyTorch: From Tensor to Kernel", BODY)
