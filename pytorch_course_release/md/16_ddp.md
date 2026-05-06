# Module 16 — Distributed Data Parallel (DDP)

# Distributed Data Parallel _(DDP)_

_Part VI · Module 16_

— the same model on every GPU, different data on each, and the gradient-bucketing trick that overlaps communication with backward to make multi-GPU training nearly free

\--- 

You have collectives. You have `torchrun`. The simplest non-trivial use of both is also by far the most common: _data parallelism_. Same model on every GPU. Different mini-batch on each. All-reduce gradients after backward. Step. Repeat.

The math is trivial — you're computing a mean of gradients, no different than gradient accumulation (M11) but spread across hardware. The interesting parts are (a) doing it efficiently with bucketing and overlap, (b) avoiding the half-dozen subtle bugs that bite first-time DDP users, and (c) knowing when DDP runs out of room and you need FSDP (M17) instead.

> **★ KEY IDEA**  
>  DDP wraps a model so every `backward()` fires **all-reduce** s on gradients in the background, _overlapping with the rest of the backward pass_. The optimizer step, run on each rank with the now-averaged gradients, takes the same step everywhere — keeping models in lockstep. The whole pattern is ~5 lines of code. The reason it works at scale is the gradient-bucketing trick that makes the all-reduce nearly free in wall-clock time. 

## One new face

DDP

DDP

"I'm the wrapper that watches your backward pass and fires off all-reduces in the background."

When you do `model = DDP(model)`, I attach hooks to every parameter's `grad_fn`. As backward computes each parameter's gradient, my hook fires and queues that gradient for all-reduce. I group small gradients into ~25 MB _buckets_ to amortize collective latency, and I dispatch the all-reduce _while backward is still running_ on other layers. By the time backward finishes, most of the network traffic is already done. After backward, the optimizer step uses the averaged gradients, keeping every rank's model identical. **I assume every rank ran the same code path on the same parameters.** If your forward had a conditional branch that ran differently on different ranks, I'll hang.

## The semantics in one paragraph

Each of N ranks gets its own data shard via DistributedSampler (M10). Each forward and backward runs independently — no communication during compute. After backward, each rank has its own gradient tensors, computed from its local mini-batch. DDP all-reduces those gradients with reduction op SUM, then divides by N (or with op AVG directly). Now every rank has the _average_ of all per-rank gradients. The optimizer step runs on each rank with those identical gradients, producing identical parameter updates. The models stay in sync.

The model on rank 0 and the model on rank 7 are _bit-identical_ at every step boundary. They diverge during forward and backward (different inputs!), but the all-reduce + identical optimizer brings them back to identical state by step end. This is what makes DDP "data-parallel" in the strict sense: only the data differs.

## The minimal DDP setup
    
    
    import os, torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Build model — every rank constructs its own copy of the same architecture
    model = build_model().to(device)
    model = DDP(model, device_ids=[local_rank])
    
    # Build dataloader — DistributedSampler shards the data across ranks
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_per_rank, sampler=sampler,
                        num_workers=4, pin_memory=True, persistent_workers=True)
    
    opt = build_optimizer(model)
    
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)              # critical — different shuffle per epoch
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = criterion(model(x), y)
            opt.zero_grad()
            loss.backward()                    # DDP fires all-reduces during this
            opt.step()

That's it. Five practical changes from a single-GPU training loop:

  1. `init_process_group` \+ device selection — from M15.
  2. Build the model and move it to `cuda:local_rank` _before_ wrapping with DDP.
  3. Wrap with `DDP(model, device_ids=[local_rank])`.
  4. Use `DistributedSampler` in your DataLoader, with `sampler.set_epoch(epoch)` per epoch.
  5. Run the script with `torchrun --nproc_per_node=N train.py`.

Backward and step are unchanged. The all-reduce happens automatically inside `backward()` via DDP's hooks.

## The bucketing-and-overlap trick

The naive way to do data-parallel: wait for backward to finish, all-reduce every gradient one at a time, then step. Each all-reduce pays ~10-50 μs of latency, plus bandwidth proportional to the gradient size. For a 1B-param model with thousands of parameters, you'd issue thousands of all-reduces — most of them small. Latency dominates.

DDP does two things to fix this:

  1. **Bucketing** : group adjacent parameters' gradients into ~25 MB chunks. Issue one all-reduce per bucket. For a 1B-param model in fp32 (4 GB), that's about 160 all-reduces instead of thousands — latency-amortized.
  2. **Overlap** : don't wait for backward to finish. Each bucket all-reduces _as soon as all its gradients are ready_ , while backward continues computing earlier-layer gradients. The communication overlaps with the rest of the compute.

Why DDP is fast: bucketing + overlap with backward ① Naïve: backward, then all-reduce GPU NCCL backward (compute) 5 buckets all-reduced sequentially total step ≈ backward + all-reduce ② DDP: bucketed all-reduces overlap with backward GPU NCCL backward (compute) B5 B4 B3 B2 B1 total step ≈ backward + tail of last bucket backward emits gradients in REVERSE order (last layer first) → DDP fires bucket B5 while still computing earlier layers' gradients communication is hidden behind compute — nearly free in wall-clock

Look at the difference. The naïve schedule has total time = backward + all-reduce. DDP's schedule has total time ≈ backward + tail (just the last bucket's all-reduce, which can't be hidden). On a fast network, the all-reduce is essentially invisible. On a slow network, the tail can still be a significant fraction — but it's far better than the naïve serialization.

The bucket size (25 MB by default) is the result of an empirical tradeoff. Smaller buckets → more chunks → more concurrency with backward but more latency overhead. Larger buckets → less concurrency but better bandwidth utilization per collective. 25 MB is the rough sweet spot for typical models on typical networks. You can tune it with the `bucket_cap_mb` arg of DDP if you want to experiment.

## The four DDP foot-guns

### 1\. `find_unused_parameters`

If your model has parameters that don't contribute to the loss for a given forward pass, DDP's bucket-tracking machinery doesn't know to expect their gradients — so it waits forever for them. The classic case: conditional branches in `forward`, or multi-task models where some heads are inactive on some samples.

The flag:
    
    
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

This tells DDP to scan after each forward and identify parameters that didn't get used, marking them as "no gradient expected" so the all-reduce doesn't wait. _It's slower_ (the scan adds overhead per step), so don't enable it unless you actually have unused parameters.

Better fix where possible: ensure _all parameters always contribute_ to the loss, even tiny multiplicative factors. Some teams add a fake `+ 0 * sum(p.sum() for p in ...)` to the loss to keep everything live. Hacky but explicit.

### 2\. The forward must be identical-shape across ranks

If rank 0's forward sees a batch of length 100 and rank 1's sees length 200, DDP doesn't directly care — it averages whatever gradients come out. But there are two indirect failure modes. (a) _BatchNorm running statistics_ : with vanilla BN, each rank updates its own running stats from its own batch — so they diverge unless you use `SyncBatchNorm`. (b) _Variable-shape attention or convolution_ : padding and masking still need to be consistent so gradient values agree.

For BN specifically:
    
    
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank])

SyncBN does a small all-reduce inside its forward to compute statistics across all ranks. Slower per step but correct. If your model uses BatchNorm at all in distributed training, you almost always want SyncBN.

### 3\. Gradient accumulation needs `no_sync()`

If you do gradient accumulation across K micro-batches before stepping (M11), you don't want DDP to all-reduce on every micro-batch — only on the last one. Otherwise you're paying network cost K-1 times for nothing.
    
    
    for step, batch in enumerate(loader):
        is_accum_step = (step + 1) % accum_steps != 0
        with model.no_sync() if is_accum_step else contextlib.nullcontext():
            loss = criterion(model(batch.x), batch.y) / accum_steps
            loss.backward()                  # gradients accumulate locally; no all-reduce
    
        if not is_accum_step:
            # this final backward fires the all-reduce on accumulated grads
            opt.step()
            opt.zero_grad()

The pattern: wrap micro-batch backwards (except the last in a group) in `model.no_sync()`. The grads accumulate locally without a network call. The final unwrapped backward fires the all-reduce on the now-accumulated gradient. Network traffic is reduced K× to once per _real_ step.

### 4\. Save the unwrapped model

From M7: a DDP-wrapped model's `state_dict()` has every key prefixed with `module.` — because `DDP` stores the original model as `self.module`. If you save and load on different rank counts (or unwrap for inference), the prefix mismatches.
    
    
    # Save the underlying model, not the DDP wrapper:
    torch.save(model.module.state_dict(), 'checkpoint.pt')
    
    # Or, equivalently, strip the prefix on load:
    sd = torch.load('checkpoint.pt')
    sd = {k.removeprefix('module.'): v for k, v in sd.items()}
    plain_model.load_state_dict(sd)

Also: **only rank 0 should save the checkpoint**. All ranks have identical model state (DDP guarantees that), so there's no point saving N copies. Standard pattern:
    
    
    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), f"ckpt_step{step}.pt")
    dist.barrier()         # every rank waits for rank 0 to finish writing

The `dist.barrier()` ensures the other ranks don't blast ahead while rank 0 is still writing — useful especially before re-loading.

## The "find_unused_parameters" trace signature

If you forget `find_unused_parameters=True` on a model that has unused parameters, DDP doesn't fail immediately. Instead, you get a hang at the _second_ training step. The first step works (DDP hasn't yet pinned which buckets to expect). The second step waits for an all-reduce that will never arrive, because the corresponding gradient was never produced. _Symptom: training logs step 0, then nothing, no NCCL timeout for 30 minutes_.

The fix is to enable the flag, but the diagnosis is the lesson: **look at the rank that's stuck and ask "which parameter's all-reduce is it waiting for?"** The answer is whatever your model didn't run on this batch.

## Effective batch size and learning rate

With DDP, your _effective batch size_ is `per_rank_batch × N`. If you trained on 1 GPU at batch=64 and now run on 8 GPUs at the same per-rank batch=64, your effective batch is 512. The standard rules of thumb (M9):

  * **Linear scaling rule** (square-root for adaptive optimizers): scale LR up roughly with batch size to maintain training dynamics. For Adam, often LR scales with √(batch_size_ratio); for SGD, LR scales linearly.
  * **Warmup gets longer** : bigger effective batches need longer warmup. A rule of thumb: warmup duration scales with batch size.

If you don't adjust LR, you might see worse training despite more compute — because effectively you're under-stepping. The "trains fine on 1 GPU, diverges on 8 GPUs" bug from M9's exercises is precisely this.

#### Q&A; — About DDP **Q:** When does DDP run out of room and I need FSDP? **A:** When your model + gradients + optimizer state don't fit on a single GPU. DDP requires the full model on each rank — for a 7B parameter model in mixed-precision AdamW (~120 GB of state, M12), you simply can't fit on a single 80 GB H100. FSDP shards the parameters, gradients, and optimizer state across ranks, trading more communication for the ability to train models that don't fit. We'll do that in M17. **Q:** My DDP run is slower per step than my single-GPU baseline. What's wrong? **A:** First, sanity check: _per-rank_ batch should be the same as single-GPU; total batch is N×. If it's slower per-step at the same per-rank size, suspect (a) network is slower than NVLink (intra-node check via topology tools), (b) `find_unused_parameters=True` when it doesn't need to be (the scan adds overhead), or (c) bucket size doesn't match your model — try `bucket_cap_mb=50` or `10` and profile. **Q:** What's `broadcast_buffers`? **A:** DDP option (default `True`) that broadcasts buffers (e.g., BN running stats) from rank 0 to all ranks at the start of each forward, keeping them in sync. Set `False` if your buffers shouldn't be synced (rare). For BN specifically, prefer SyncBatchNorm over relying on broadcast_buffers — broadcast just copies rank 0's stats, which weren't computed across ranks anyway. **Q:** Can I have different model architectures on different ranks? **A:** No — DDP's invariant is "same model everywhere." If you need heterogeneous models (e.g., pipeline parallel where each rank holds a different stage), DDP isn't the right tool. M18 covers pipeline parallelism, where you do exactly that with different mechanisms (point-to-point sends/recvs). **Q:** Does DDP work with `torch.compile`? **A:** Yes, but the order matters: compile the model first, then wrap with DDP. `model = torch.compile(model); model = DDP(model)`. Otherwise the compile boundary doesn't match DDP's hooks. M21 covers compile in detail. 

## The complete DDP training script (skeleton)
    
    
    import os, contextlib, torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import DataLoader
    
    def train(config):
        dist.init_process_group(backend='nccl')
        rank        = dist.get_rank()
        world_size  = dist.get_world_size()
        local_rank  = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    
        model = build_model(config).to(device)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)        # if you have BN
        model = DDP(model, device_ids=[local_rank])
        opt = build_optimizer(model)
    
        sampler = DistributedSampler(dataset, shuffle=True)
        loader = DataLoader(dataset, batch_size=config.batch_per_rank,
                            sampler=sampler, num_workers=4,
                            pin_memory=True, persistent_workers=True)
    
        for epoch in range(config.num_epochs):
            sampler.set_epoch(epoch)
            model.train()
            for step, (x, y) in enumerate(loader):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                is_accum = (step + 1) % config.accum_steps != 0
                with model.no_sync() if is_accum else contextlib.nullcontext():
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = criterion(model(x), y) / config.accum_steps
                    loss.backward()
                if not is_accum:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); opt.zero_grad()
    
            if rank == 0 and epoch % config.ckpt_every == 0:
                torch.save(model.module.state_dict(), f"ckpt_e{epoch}.pt")
            dist.barrier()
    
        dist.destroy_process_group()

Run with `torchrun --nproc_per_node=8 train.py`. That's a complete, production-flavored DDP training script. Note all the patterns from M10 (DistributedSampler + set_epoch), M11 (gradient accumulation, checkpoint discipline), M14 (autocast), and M15 (init_process_group + LOCAL_RANK device selection) all wired together.

## Code Magnets: assemble a DDP-correct backward step

Build the inner DDP step that handles gradient accumulation correctly. Three magnets are wrong choices.

Arrange the magnets into a working accumulation step.

is_accum = (step + 1) % accum_steps != 0 with model.no_sync() if is_accum else contextlib.nullcontext(): with model.no_sync(): loss = criterion(model(x), y) / accum_steps loss.backward() if not is_accum: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) opt.step(); opt.zero_grad() opt.step(); opt.zero_grad() dist.all_reduce(loss)

show solution
    
    
    is_accum = (step + 1) % accum_steps != 0
    with model.no_sync() if is_accum else contextlib.nullcontext():
        loss = criterion(model(x), y) / accum_steps
        loss.backward()
    if not is_accum:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()

The traps:

  * `with model.no_sync():` always, no condition — that suppresses the all-reduce _even on the final accum step_. Models drift apart, training is broken.
  * `opt.step(); opt.zero_grad()` outside the `if not is_accum` guard — the optimizer would step every micro-batch, defeating accumulation.
  * `dist.all_reduce(loss)` — DDP all-reduces the _gradients_ , not the loss. The loss is local; for logging, you can optionally all-reduce it, but it's not part of the training-step correctness.

The structure: **no_sync only on micro-batches that are not the final one in a group** ; the final backward (without no_sync) fires the all-reduce on accumulated gradients; then the optimizer steps.

## Who does what?

Match each DDP concept to its real role.

Concept

Real role

DistributedDataParallel

A. Wrapper that hooks into backward and fires bucket all-reduces in the background.

Gradient bucketing

B. Synchronizes BatchNorm statistics across ranks via mini-collectives in forward.

find_unused_parameters=True

C. Group ~25 MB of gradients per all-reduce to amortize collective latency.

SyncBatchNorm

D. Suppresses DDP's gradient sync for one backward pass — used in accumulation.

model.no_sync()

E. Tells DDP to scan for parameters that didn't get gradients, marking them as expected-missing.

sampler.set_epoch(epoch)

F. Without this, DistributedSampler shuffles the same way every epoch.

save model.module.state_dict()

G. Strips the DDP wrapper's `module.` prefix before saving — clean checkpoints.

show solution

**DistributedDataParallel** → A  
**Gradient bucketing** → C  
**find_unused_parameters=True** → E  
**SyncBatchNorm** → B  
**model.no_sync()** → D  
**sampler.set_epoch(epoch)** → F  
**save model.module.state_dict()** → G 

The mental shortcut: _DDP wraps and hooks, bucketing amortizes collectives, find_unused handles divergent forwards, SyncBN cross-rank stats, no_sync suppresses one all-reduce, set_epoch reshuffles per epoch, .module unwraps the prefix_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** Your DDP run hangs after the first training step. `nvidia-smi` shows GPUs at 0%, and you don't see any error. What's the first thing to check?

show answer

`find_unused_parameters`. The hang-after-first-step pattern is DDP's signature for "the bucket-tracking machinery is waiting for an all-reduce on a parameter that didn't receive a gradient." Either (a) wrap with `find_unused_parameters=True` (slow but correct), or (b) restructure the forward so all parameters are always touched by the loss. If the model has conditional branches, you usually want (a). If it's a multi-task model where some heads are skipped, (a) again. Note that this hang doesn't trigger NCCL's timeout for 30 minutes by default — you have to recognize the pattern from the trace yourself.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** A team trained their CNN on 1 GPU at batch=64 and got 76% val accuracy. They scale to 8 GPUs at per-rank batch=64 (effective batch=512), run the same recipe, and get 71%. What's likely happening?

show answer

Two candidates. (1) **BatchNorm contamination** : each rank's BN computes statistics from 64 samples instead of the full 512. The statistics are noisier, especially at lower batch sizes. Fix: `SyncBatchNorm.convert_sync_batchnorm(model)`. (2) **Effective batch size shift** : at 8× the batch, you typically need to scale LR up (linearly for SGD, by √8 for Adam) and extend warmup. Without that, you're effectively under-stepping. Try both fixes — usually you need both.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** You time your DDP backward and see it takes the same wall time as your single-GPU backward. Is that suspicious?

show answer

No — that's actually _good news_. Backward in DDP includes the all-reduces, which are running concurrently with backward computation. If they're fully hidden by the overlap, your DDP backward time ≈ your single-GPU backward time. That's the magic of bucketing + overlap. If the times differ noticeably, then either (a) network is slow enough that the tail of the last bucket extends past backward, or (b) something is forcing a sync (printing intermediate values, calling .item() in the hot path). Profile to confirm.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** You want to all-reduce a per-batch metric (e.g., total loss across ranks for logging). Write the line.

show answer
    
    
    loss_for_logging = loss.detach()
    dist.all_reduce(loss_for_logging, op=dist.ReduceOp.AVG)
    if rank == 0:
        log(f"step {step}: loss {loss_for_logging.item():.4f}")

Two important things. (1) `.detach()` first — you don't want autograd to think this all-reduce is part of the graph. (2) `op=dist.ReduceOp.AVG` is supported in modern PyTorch and avoids the need to manually divide by world_size. Older versions used SUM and divided manually. (3) Only log on rank 0; otherwise N copies of the same line spam stdout.

### What just happened?

  * **DDP's contract** : same model on every rank, different data, all-reduce gradients after backward. Models stay bit-identical at step boundaries.
  * **The wrapper** : `DDP(model, device_ids=[local_rank])` attaches autograd hooks that fire all-reduces during backward.
  * **Bucketing** : gradients grouped into ~25 MB buckets to amortize collective latency. ~hundreds of all-reduces per step instead of thousands.
  * **Overlap** : each bucket all-reduces as soon as its gradients are ready, while backward continues on earlier layers. Communication is largely hidden behind compute.
  * **DistributedSampler** shards data across ranks. Always call `sampler.set_epoch(epoch)` per epoch — otherwise shuffle is identical every epoch.
  * **find_unused_parameters=True** : needed when forward conditionally skips parameters. Symptom of forgetting it: hang on the second training step, no error.
  * **SyncBatchNorm** : replace BN layers with this when training distributed. `nn.SyncBatchNorm.convert_sync_batchnorm(model)` before DDP wrap.
  * **`model.no_sync()`** : context manager that suppresses DDP's all-reduce for one backward. Use it on micro-batches in gradient accumulation; let the final backward fire the all-reduce on accumulated grads.
  * **Save`model.module.state_dict()`** — unwraps the DDP prefix. Save only on rank 0; `dist.barrier()` after.
  * **Effective batch = per_rank × N**. Adjust LR and warmup accordingly. Linear scaling for SGD, √-scaling for Adam, longer warmup with bigger batches.
  * **DDP runs out when the model+grads+state don't fit on one GPU.** Then you need FSDP — Module 17.
  * The reflex: when DDP misbehaves, ask which collective is hanging or missing. Print rank IDs at suspect lines. Look at trace for the four bottleneck shapes from M13.

Module 17 takes the model that doesn't fit on a single GPU and shards it. ZeRO-1, ZeRO-2, ZeRO-3. FSDP. The all-gather + reduce-scatter dance that lets you train 70B models on commodity hardware. The same identity from M15 (`all_reduce ≡ reduce_scatter + all_gather`) is the entire conceptual basis.
