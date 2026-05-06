# Module 12 — Memory: where does it all go?

# Memory: _where does it all go?_

_Part V · Module 12_

— the five buckets of GPU memory, the transformer memory formula that lets you predict OOM before you hit it, and the recompute trick that buys you depth at the cost of compute

\--- 

You've trained models. You've also gotten `CUDA out of memory` at step 1247 of a long run. Welcome to Part V, where we stop pretending memory is infinite.

The good news: GPU memory consumption during training is almost completely predictable. Five buckets account for ≥99% of it: **parameters, gradients, optimizer state, activations, and workspace**. If you can estimate each bucket from your config, you can predict OOM in advance, choose batch size and depth deliberately, and know exactly which lever to pull when something doesn't fit. The bad news: PyTorch's caching allocator hides some of this from `nvidia-smi`, which is why "but I have free memory!" doesn't always mean what you think.

> **★ KEY IDEA**  
>  Training memory = **params + grads + optimizer state + activations + workspace**. The first three scale with _model size_ (and stay constant during training). The fourth scales with _batch size × sequence length × depth_ and dominates at scale. Activation checkpointing trades compute for memory by recomputing activations in the backward pass instead of storing them. Once you internalize the five buckets, you can predict OOM from a config sheet without running a single step. 

## One new face

A

Activation

"I'm what every layer leaves behind for backward to find."

When you do `y = layer(x)` with autograd on, I'm what gets saved so backward can compute the gradient w.r.t. `x`. There's one of me per layer per forward pass, and my size is **proportional to the batch size, sequence length, and hidden dim**. For deep models I dominate memory. The recompute trick (M6) and gradient checkpointing both target me — they delete me after forward and recompute me during backward. The framework treats me as ephemeral, but at training time I'm often the biggest bucket.

## The five buckets

Let's enumerate everything that lives in GPU memory during a training step.

Where every GPU byte goes during training Bucket| Lifetime| Scales with| Typical size for a 1B-param transformer in mixed precision  
---|---|---|---  
**Parameters**|  Whole training run| Param count × 2 (bf16)| ~2 GB  
**Master params** (mixed precision)| Whole training run| Param count × 4 (fp32)| ~4 GB  
**Gradients**|  One step (until next zero_grad)| Param count × 4 (fp32 typically)| ~4 GB  
**Optimizer state**|  Whole training run| Adam: param count × 8 (m + v in fp32)| ~8 GB  
**Activations**|  Forward → released during backward| Batch × seq × hidden × layers| varies massively — see below  
**Workspace**|  Per-op transient| cuDNN/cuBLAS scratch buffers| ~few hundred MB  
  
Look at that 1B-param column. **~18 GB before activations**. That's just to _hold the model and optimizer_. Add 1B parameters' worth of activations on top — easily 20+ GB more for a typical transformer at 4K sequence length and modest batch size — and you're at 40 GB. That's a single A100 used up at 1B scale. This is why people obsess over the activation bucket and why FSDP/ZeRO exist (Module 17).

A 1B-param transformer in mixed-precision Adam, batch=16 × seq=4096 bf16 params (~2 GB) fp32 master copy (~4 GB) gradients (~4 GB) Adam optimizer state — m + v in fp32 (~8 GB) activations (~30 GB at this batch×seq×depth) — THE bucket you actually have control over "why am I OOM" answers from this picture 90% of the time

## Bucket 1-3: parameters, gradients, optimizer state

These three are easy. They scale only with parameter count, not with batch size or sequence length. They're constant for the whole training run.

### Parameters

For a model with `N` parameters in fp32, that's `4N` bytes. In bf16/fp16, `2N` bytes. With mixed precision (M14), you typically have _both_ : a bf16 working copy used for forward and backward, plus an fp32 "master copy" used by the optimizer for accurate updates. So 6 bytes per param.

### Gradients

One gradient per parameter. By default in fp32 (even when params are in bf16), so `4N` bytes. They live from `backward()` until `zero_grad()` — i.e., for one optimizer step at a time. With gradient accumulation (M11), they live across multiple micro-batches.

### Optimizer state

Depends on the optimizer (M9):

  * **SGD (no momentum)** : 0 buffers per param.
  * **SGD + momentum** : 1 buffer (velocity), `4N` bytes in fp32.
  * **Adam / AdamW** : 2 buffers (first and second moment), `8N` bytes in fp32.
  * **Lion** : 1 buffer, `4N` bytes.

For Adam in fp32, this is **2× the parameter count** just for state. Combined with master params and gradients, the "fixed cost" of training a 1B model with mixed-precision AdamW is `2N + 4N + 4N + 8N = 18N` bytes ≈ 18 GB.

#### Q&A; — About fixed-cost memory **Q:** Why are gradients in fp32 even when params are bf16? **A:** Two reasons. (1) Numerical precision — gradient values can be very small, and bf16's 7-bit mantissa loses precision in the tail of the distribution. (2) Optimizer compatibility — Adam's running second moment is sensitive to gradient precision; bf16 gradients passed through the EMA accumulate noticeable error. So the standard mixed-precision recipe is: weights and activations in bf16, gradients and optimizer state in fp32. We'll re-derive this in M14. **Q:** Can I reduce optimizer state with bitsandbytes-style 8-bit Adam? **A:** Yes — 8-bit Adam (and similar quantized optimizers) keeps the m and v buffers in 8-bit form with per-block scales. Roughly 4× smaller than fp32 Adam, with negligible quality impact for most tasks. Useful for fitting bigger models on smaller GPUs. You wire it in by replacing your `torch.optim.AdamW` import with a quantized one — same API. **Q:** Why not bf16 master copy? **A:** Because the optimizer's update `p ← p − lr · g` can have `lr · g` tiny enough to vanish into bf16's 7-bit mantissa when added to `p`. The fp32 master copy preserves these tiny accumulated changes. After the optimizer step, you cast back to bf16 for forward/backward. Without the master copy, training stops making progress after the LR decays. 

## Bucket 4: activations (the variable one)

Now the bucket that matters. For each layer in your forward pass, the autograd engine saves intermediate values needed to compute gradients in the backward pass. _How much_ depends on the layer.

For a transformer layer (attention + MLP), the activations saved in forward include:

  * The input `x` to the layer (for residual)
  * Q, K, V tensors after projections — three of them, each shape `(B, T, D)`
  * The attention matrix or softmax output — shape `(B, H, T, T)`
  * The attention output before projection — shape `(B, T, D)`
  * The MLP intermediate — shape `(B, T, 4D)` typically
  * Various small tensors from LayerNorm (mean, rstd) — shape `(B, T)`

The headline term is **the attention matrix at`(B, H, T, T)`**. For B=4, H=32, T=8192, that single tensor in bf16 is:
    
    
    4 × 32 × 8192 × 8192 × 2 = 17.2 GB     # PER LAYER

For a 32-layer transformer at this config, attention matrices alone are **550 GB**. This is the entire reason FlashAttention exists (M25) — it computes attention without ever materializing the (T, T) matrix.

### The transformer memory formula

For a vanilla transformer (no FlashAttention) in mixed precision:
    
    
    activation_bytes_per_layer ≈ B × T × 2 × (11D + 5HT)

Where:

  * `B` = batch size, `T` = sequence length, `D` = hidden dim, `H` = number of heads
  * The `11D` term covers Q, K, V, attention output, MLP intermediate (4D), residuals, and LayerNorm intermediates
  * The `5HT` term is the attention matrix and its softmax — the quadratic-in-T part
  * The factor of 2 is for bf16

This formula has appeared in multiple reduced forms in the literature; consider it a back-of-envelope estimate. The point isn't precision — it's that **the quadratic-in-T term blows up first**. At T=2048, attention matrices are roughly equal to the linear terms. At T=8192, they're 4× larger. At T=32768, they're enormous.

### Comparison with FlashAttention

FlashAttention (M25) replaces the `5HT` term with O(BT × D), eliminating the quadratic-in-T memory. The new formula:
    
    
    activation_bytes_per_layer ≈ B × T × 2 × (11D + small)

For most modern training, this is the difference between "fits on an A100" and "doesn't." We'll see how it works at the kernel level in Part VIII.

## Bucket 5: workspace

Each cuDNN convolution, cuBLAS matmul, etc. allocates a scratch buffer for its operation. These are usually small (a few MB to a few hundred MB) and ephemeral, but they show up under heavy use.

The workspace is set by the cuDNN benchmark mode (M3): with `torch.backends.cudnn.benchmark=True`, cuDNN tries multiple algorithms for each convolution shape on first encounter and caches the fastest. Some algorithms use more workspace than others. Speed wins; memory hits a fixed peak.

You rarely need to think about workspace explicitly. If you're seeing surprising memory usage that isn't accounted for by the other four buckets, this is a candidate.

## Activation checkpointing: trade compute for memory

You met the idea in M6 (custom Functions). Activation checkpointing — also called gradient checkpointing — applies the recompute trick at the level of _blocks of layers_. The framework handles the bookkeeping for you.

The idea: instead of saving every intermediate activation between layer 1 and layer N, save only checkpoints at strategic points (e.g., every K-th layer). During backward, when you reach a checkpointed region, _re-run the forward pass for that region_ to recompute the missing activations on the fly.

Memory profile: with vs without activation checkpointing time → forward backward memory used peak (start of backward) no checkpointing — peak grows with depth with checkpointing — sawtooth, lower peak

The sawtooth pattern in the green curve is the recompute happening: in backward, when we reach a checkpointed boundary, memory briefly spikes as we recompute the activations for that block, then drops as we use them and discard them. Peak memory is roughly `√N` times smaller than without checkpointing (where N is the number of layers and you checkpoint every `√N`-th).

The cost: **~33% more compute**. Forward is run twice (once originally, once during backward) for the checkpointed regions. For a memory-bound training run, that's a great trade.

### Using checkpoint() in PyTorch
    
    
    from torch.utils.checkpoint import checkpoint, checkpoint_sequential
    
    class CheckpointedTransformer(nn.Module):
        def __init__(self, n_layers, dim, n_heads):
            super().__init__()
            self.layers = nn.ModuleList([Block(dim, n_heads) for _ in range(n_layers)])
    
        def forward(self, x):
            for layer in self.layers:
                x = checkpoint(layer, x, use_reentrant=False)
            return x

That's it. Each `checkpoint(layer, x)` call says "don't save activations from inside this layer; recompute them in backward." The `use_reentrant=False` selects the modern implementation (the old reentrant one had subtle issues with autograd).

Tradeoffs to know:

  * **Granularity matters** : checkpointing every layer = max memory savings, max compute overhead (~33%). Checkpointing every K layers = less savings, less overhead. The Megatron / DeepSpeed default is "every layer" for the largest models.
  * **Don't checkpoint cheap things** : checkpointing a single LayerNorm is silly — recomputing it costs more than storing its (small) activation. Checkpoint _blocks_ (e.g., a whole transformer layer).
  * **RNG state** : if your block uses Dropout or any other random op, the checkpoint API needs to _save and restore the RNG state_ so the recomputed forward gives the same answer. PyTorch handles this for you, but it's why you can't trivially checkpoint things that have non-deterministic side effects.

> **⚠ WARNING**  
>  **Activation checkpointing + FlashAttention together is the modern combo.** FlashAttention removes the quadratic-in-T attention term; activation checkpointing reduces the linear-in-depth term by ~√N. Together they make 100B+ models trainable on A100/H100 GPUs. Don't pick one or the other — modern training uses both. 

## The CUDA caching allocator (briefly)

When you call `torch.zeros(...)` or any tensor-creating op, PyTorch doesn't always go to `cudaMalloc`. It maintains a _caching allocator_ : when a tensor's storage is freed, the memory is returned to a pool, not to the OS. Next allocation of similar size reuses the cached block.

Why? Because `cudaMalloc` and `cudaFree` are slow (synchronous calls into the driver). Constant alloc/free cycles in the training loop would dominate runtime. The caching allocator amortizes this.

The implication for memory accounting:

  * `nvidia-smi` shows you what the OS thinks PyTorch holds — i.e., the size of the caching allocator pool. Not what's actually _in use_.
  * `torch.cuda.memory_allocated()` shows what's currently allocated to live tensors.
  * `torch.cuda.memory_reserved()` shows what the caching allocator has reserved from the OS (= what nvidia-smi sees, modulo other GPU users).

The discrepancy between `memory_allocated` and `memory_reserved` can be huge — sometimes 10+ GB of "free" memory in the caching pool that isn't returned to the OS. That's normal. `torch.cuda.empty_cache()` forces it back, but you almost never want to call this in production (it's slow and the next allocation will go back to `cudaMalloc`).

### Memory diagnosis tools
    
    
    # 1. Quick snapshot
    print(torch.cuda.memory_summary())
    
    # 2. Track over time — useful for spotting memory leaks
    def log_mem(tag):
        a = torch.cuda.memory_allocated() / 1e9
        r = torch.cuda.memory_reserved() / 1e9
        print(f"{tag}: allocated={a:.2f}GB reserved={r:.2f}GB")
    
    log_mem("after model load")
    # ... a few steps of training ...
    log_mem("after step 100")
    log_mem("after step 1000")         # should be roughly the same as step 100!

If `memory_allocated` grows over training steps, you have a leak. The most common causes:

  1. **Holding tensor references** : storing per-step losses as a list of CUDA tensors instead of `.item()` floats. Each step adds a tensor to the list and never releases.
  2. **Autograd graphs being held** : assigning a tensor with `requires_grad=True` to a class attribute prevents the graph from being freed. `.detach()` before storing.
  3. **Unclosed hooks** : as we discussed in M5, hooks can hold closures that hold model references.

## Memory fragmentation

The caching allocator's pool can fragment. You allocate a 4 GB tensor, free it, allocate two 2 GB tensors. The 4 GB block is now split. If you later need a contiguous 4 GB allocation, you might get OOM _even though you have 4+ GB total free_ because no single contiguous block is that big.

Fragmentation symptoms:

  * OOM with significant free memory in `memory_reserved - memory_allocated`
  * Variable batch sizes or sequence lengths in training (each new shape opens a new block)
  * Long-running training that gradually approaches OOM near the memory limit

Mitigations:

  * **Use fixed shapes** when possible — bucket sequence lengths (M10), use the same batch size every step.
  * `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — newer allocator mode that handles fragmentation better. Set as an env var before importing torch.
  * `torch.cuda.empty_cache()` as a last resort — releases the pool back to the OS. Slow but defragments.

## The mental model: predict before you run

Pull all of this together into a memory budget you can compute by hand. For a typical mixed-precision AdamW training of a transformer:
    
    
    fixed_per_param = 2(bf16) + 4(fp32 master) + 4(grad fp32) + 8(adam fp32) = 18 bytes
    fixed_total = 18 × N
    
    per_layer_activations = B × T × 2 × (11D + 5HT)        # bf16
    total_activations = layers × per_layer_activations
    
    total_memory = fixed_total + total_activations + workspace

Worked example: GPT-2 medium-ish, 350M params, 24 layers, dim=1024, 16 heads, T=1024, B=8:
    
    
    fixed       = 18 × 350M = 6.3 GB
    per_layer   = 8 × 1024 × 2 × (11×1024 + 5×16×1024) = 8 × 1024 × 2 × 93184 ≈ 1.5 GB
    total_acts  = 24 × 1.5 GB = 36 GB
    total       ≈ 42 GB           # before workspace, kernel scratch, etc.

So this config eats an A100 (40 GB) and OOMs. Options:

  1. Halve the batch size: activations halve → `~24 GB`, fits comfortably
  2. Activation checkpointing: activations × ~0.2 → `~7 GB`, easy fit
  3. FlashAttention: kills the `5HT` term, big savings on long sequences
  4. Switch to 8-bit Adam: optimizer state goes from 8N to 2N, saves ~3 GB

The lesson isn't the specific arithmetic — it's that you can _predict_ memory consumption from your config. When training OOMs, you can compute the bottleneck instead of guessing.

## Code Magnets: build a memory-aware training step

You're modifying a training step to use activation checkpointing on the model's transformer layers and to log memory usage every 100 steps. Build the inner step.

Arrange the magnets into a working step. Two are red herrings.

def train_step(model, opt, scheduler, batch, step): x, y = [t.to(device, non_blocking=True) for t in batch] out = model(x) out = checkpoint(model, x, use_reentrant=False) loss = F.cross_entropy(out, y) loss.backward() opt.step(); opt.zero_grad(); scheduler.step() if step % 100 == 0: log_mem(f"step {step}") torch.cuda.empty_cache() return loss.item()

show solution
    
    
    def train_step(model, opt, scheduler, batch, step):
        x, y = [t.to(device, non_blocking=True) for t in batch]
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        opt.step(); opt.zero_grad(); scheduler.step()
        if step % 100 == 0:
            log_mem(f"step {step}")
        return loss.item()

The traps:

  * `out = checkpoint(model, x, ...)` — wrapping the _whole_ model in a checkpoint defeats the purpose. Checkpointing belongs _inside_ the model's `forward`, around individual blocks. Doing it at the top means recomputing the entire forward pass during backward — same cost as just not checkpointing at all (worse, actually, because of the bookkeeping).
  * `torch.cuda.empty_cache()` after every log is wrong — it's slow, and the cache is supposed to grow to a steady state. Calling it in the hot path defeats the caching allocator. Reserve it for emergencies.

The right pattern: model's `forward` wraps each block in `checkpoint(block, x, use_reentrant=False)` internally. Training step looks like normal training, calls `log_mem` for diagnosis, and trusts the allocator.

## Who does what?

Match each memory concept to its real role.

Concept

Real role

fp32 master copy

A. Per-tensor scratch buffer used by cuDNN/cuBLAS for one op.

Activations

B. Trade ~33% more compute for √N less peak activation memory.

Activation checkpointing

C. The biggest memory bucket at scale; scales with batch × seq × depth.

CUDA caching allocator

D. Preserves precision when applying tiny optimizer updates that bf16 would round away.

memory_reserved vs memory_allocated

E. Holds freed tensor memory in a pool to avoid slow cudaMalloc/cudaFree.

cuDNN workspace

F. The gap is normal — cached blocks the allocator hasn't returned to the OS.

Fragmentation

G. OOM despite having free memory total — no single block is large enough.

show solution

**fp32 master copy** → D  
**Activations** → C  
**Activation checkpointing** → B  
**CUDA caching allocator** → E  
**memory_reserved vs memory_allocated** → F  
**cuDNN workspace** → A  
**Fragmentation** → G 

The mental shortcut: _master copy preserves tiny updates, activations dominate at scale, checkpointing trades compute for memory, allocator caches frees, reserved-allocated gap is the cache, workspace is per-op scratch, fragmentation is contiguous-block starvation_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A 7B-parameter transformer trained with mixed-precision AdamW. Compute the fixed memory cost (params + master + grads + Adam state) in GB.

show answer

Per parameter: 2 (bf16) + 4 (fp32 master) + 4 (grad fp32) + 8 (Adam m + v in fp32) = **18 bytes**. For 7B params: `18 × 7e9 = 126 GB`. That's already more than a single 80 GB H100 — which is why training a 7B model on a single GPU isn't possible without sharding (FSDP, M17). At minimum 2 H100s; in practice 4-8 with activation memory.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Your training is OOMing at step 1247. `nvidia-smi` shows you have 5 GB free. `torch.cuda.memory_allocated()` says you're using 38 GB. `torch.cuda.memory_reserved()` says 43 GB. What's the most likely cause and what should you try?

show answer

Fragmentation. You have 5 GB free in the pool (43 - 38), but no contiguous block large enough for the new allocation. Two things to try: (1) Check if you have variable-shape batches — sort/bucket by length (M10) so shapes repeat. (2) Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as an env var; this allocator mode handles fragmentation much better. As a last resort, `torch.cuda.empty_cache()` before the suspect allocation, but this slows training so use only as diagnosis.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** A vanilla transformer training run (no FlashAttention) at T=4096 OOMs. Switching to T=2048 fits comfortably. By roughly what factor did peak memory drop, and which term dominated?

show answer

The attention matrix term is `5HT × T = 5HT²` per layer. Halving T quarters this term. The linear-in-T terms (`11D × T`) only halve. So the dominant change is in the quadratic term — peak memory dropped roughly 4× for that bucket. This is exactly why FlashAttention exists: by replacing the `O(T²)` attention term with `O(T)`, you can run T=8192 or T=32768 with the memory you used to spend on T=2048.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** A team reports their training memory grows by ~50 MB per 1000 steps and eventually OOMs after a million steps. Sketch the diagnosis path.

show answer

Linear growth = leak. Step 1: log `memory_allocated` at regular intervals to confirm the rate. Step 2: look for tensor references being held — most common: appending CUDA tensors to a list (e.g., `losses.append(loss)` instead of `losses.append(loss.item())`); class attributes accumulating per-batch outputs; hooks holding closures over tensors; metrics objects storing tensors instead of floats. Step 3: if not obvious, use `torch.cuda.memory_snapshot()` for the per-allocation history, or `torch.cuda.memory._record_memory_history()` followed by snapshot dumps. The leak is almost always at the Python level, not in the framework.

### What just happened?

  * Training memory has **five buckets** : parameters, gradients, optimizer state, activations, workspace. The first three are constant; activations dominate at scale.
  * Mixed-precision AdamW costs **18 bytes per parameter** in fixed memory (2 bf16 + 4 fp32 master + 4 grad + 8 Adam). For 1B params, that's 18 GB _before_ activations.
  * Why fp32 master copy: bf16's 7-bit mantissa would round away tiny optimizer updates. Master copy preserves precision; cast to bf16 for forward/backward.
  * Why fp32 grads: bf16 grad accumulation (Adam's `v`) accumulates noticeable error. Standard recipe: weights/activations bf16, grads/optimizer fp32.
  * Activations have a **linear-in-depth term** (~11D per token per layer) and a **quadratic-in-T term** (~5HT per layer for the attention matrix). Long sequences are dominated by the quadratic.
  * **FlashAttention** kills the `O(T²)` term — it's the single biggest activation-memory win for long-sequence training.
  * **Activation checkpointing** trades ~33% more compute for ~√N less peak memory. Checkpoint _blocks_ , not individual cheap ops.
  * The modern combo: FlashAttention + activation checkpointing. Together they make 100B+ models trainable on A100/H100.
  * **The CUDA caching allocator** keeps freed memory in a pool. `memory_reserved - memory_allocated` is the gap; that's normal.
  * **Fragmentation** : OOM with free memory total but no contiguous block. Mitigate with consistent shapes, `expandable_segments:True`, or `empty_cache()` as last resort.
  * **Memory leaks** at the Python level: storing CUDA tensors instead of `.item()` floats, retained autograd graphs, unclosed hooks. Diagnose with `memory_allocated` over time.
  * The reflex: when OOM hits, _predict_ first (compute the budget), then _profile_ if predict didn't match (memory_summary, snapshot). Don't guess — the budget is calculable.

Module 13 takes the same predictive mindset to time. `torch.profiler`, traces in Chrome/Perfetto, the GPU/CPU/data-pipeline-bound diagnosis. By the end you'll know whether your training is bottlenecked by the optimizer step, the dataloader, or a single bad kernel — and you'll have the tools to find each one.
