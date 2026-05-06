# Module 15 — Communication primitives

# Communication _primitives_

_Part VI · Module 15_

— the four collective ops that every distributed training scheme is built on, why ring all-reduce is bandwidth-optimal, and how to read a "this is communication-bound" trace

\--- 

Welcome to Part VI. We're done squeezing single-GPU performance — the next four modules are about what happens when you have _many_ GPUs and they need to agree on something.

The whole story of distributed deep learning is built on five or six communication primitives. DDP is "all-reduce after backward" (M16). ZeRO/FSDP is "all-gather params, reduce-scatter grads" (M17). Tensor parallelism is "all-reduce activations inside layers" (M18). Pipeline parallelism is "send/recv between stages" (M18 also). Once you understand the primitives, every distributed scheme is just a different schedule of them.

> **★ KEY IDEA**  
>  The four collective ops you'll meet over and over: **all-reduce** (sum each tensor across ranks; everyone gets the sum), **all-gather** (concatenate each rank's tensor; everyone gets the concatenation), **reduce-scatter** (sum across ranks, but each rank only keeps its slice), and **broadcast** (one rank sends to everyone). Every distributed training scheme is a clever scheduling of these. The cost of each is dominated by _bytes transferred per rank_ , and the optimal algorithm — ring all-reduce — achieves `2(N-1)/N · message_size` per rank, asymptotically optimal. 

## Two new players

R

Rank

"I'm a process with a number. I see the world as my slice of the work."

In a distributed run with N GPUs, there are N of me, numbered 0 through N-1. We each have our own copy of the model (or a slice — depends on the scheme). We each load our own data shard. We process our own forward pass. The only time I talk to my siblings is during a _collective_ — and it costs me real seconds of bandwidth, so I try to do it rarely and in big chunks.

⊕

Collective

"I'm an op every Rank participates in. We all enter, none of us leaves until the result is everywhere it needs to be."

I'm `all_reduce`, `all_gather`, `reduce_scatter`, `broadcast`, `scatter`, `gather`, `all_to_all`. I'm a synchronization point — every Rank must call me, in the same order, with matching shapes. Skip me on one rank and the others wait forever. Call me with the wrong shape and you get a deadlock or a memory corruption. I'm strict because I have to be. Get me right, and I scale to thousands of GPUs.

## The four primitives, in pictures

Each primitive is defined by what each rank starts with and what each rank ends with. Read the diagrams literally — every column is one rank, every row is the data on that rank before/after.

Four collectives — what each rank has before and after ① Broadcast (1 → many) before A · · · after A A A A rank 0rank 1rank 2rank 3 "sender's data goes to every rank" ② All-Reduce (sum, all rks) before A B C D after A+B+C+D A+B+C+D A+B+C+D A+B+C+D rank 0rank 1rank 2rank 3 ③ All-Gather (concat, all) before A B C D after A|B|C|D A|B|C|D A|B|C|D A|B|C|D rank 0rank 1rank 2rank 3 "each rank's piece becomes everyone's whole" ④ Reduce-Scatter (sum then split) before [a₀ a₁ a₂ a₃] [b₀ b₁ b₂ b₃] [c₀ c₁ c₂ c₃] [d₀ d₁ d₂ d₃] after a₀+b₀+c₀+d₀ a₁+b₁+c₁+d₁ a₂+b₂+c₂+d₂ a₃+b₃+c₃+d₃ rank 0rank 1rank 2rank 3 Two key identities to memorize: all_reduce ≡ reduce_scatter + all_gather all_gather ≡ N broadcasts (one per rank) — but cheaper as one collective FSDP exploits identity #1: shard params, all-gather to compute, reduce-scatter the grads

Read each panel. The "before" row shows what each rank starts with; the "after" row shows what each rank ends with. The colors are the same within a panel where the data has the same meaning.

Spend ten seconds on each:

  * **Broadcast** : one rank has data, the rest don't, after — everyone has it. The simplest collective. _Used at the start of training to sync the initial weights from rank 0 to all others._
  * **All-reduce** : every rank has different data. After — everyone has the sum (or other reduction op: max, min, mean). _The DDP gradient sync._
  * **All-gather** : every rank has its own slice. After — everyone has the concatenation of all slices. _FSDP uses this to materialize a sharded parameter for one layer's forward._
  * **Reduce-scatter** : every rank has the same shape, broken into N chunks. After — each rank has only _its_ chunk, summed across all ranks. _FSDP uses this for the gradient reduction._

The two identities at the bottom of the diagram are worth memorizing. **All-reduce equals reduce-scatter followed by all-gather** — and this is exactly how NCCL implements ring all-reduce internally. **FSDP/ZeRO-3 exploits this identity** : instead of doing the full all-reduce, you split the work and only do the reduce-scatter for gradients (since you only need your slice anyway), saving bandwidth.

## Ring all-reduce: why it's bandwidth-optimal

The naive all-reduce is "everyone sends to rank 0; rank 0 sums; rank 0 broadcasts back." That works but the bandwidth at rank 0 is O(N × M) — every rank's full message goes through it. For 1024 GPUs, this is unworkable.

The trick: arrange ranks in a logical ring. Pass the data around the ring in N-1 steps, summing as it goes. Then pass the summed result around for another N-1 steps so everyone has the final answer. _Each rank only ever sends and receives`M/N` bytes per step_, where M is the message size. Total per-rank traffic: `2(N-1)/N × M`, which approaches `2M` as N grows. Independent of N. That's bandwidth-optimal.

Ring all-reduce: each rank sends only M/N per step step k of N-1: reduce-scatter phase R0 R1 R2 R3 a₀+r3a₀ b₁+r0b₁ c₂+r1c₂ d₃+r2d₃ After N-1 = 3 reduce-scatter steps: rank 0 has the FULL SUM of chunk 0 (= a₀+b₀+c₀+d₀) rank 1 has the FULL SUM of chunk 1 rank 2 has the FULL SUM of chunk 2 rank 3 has the FULL SUM of chunk 3 Then N-1 all-gather steps to broadcast: every rank now has the FULL SUM of every chunk → done! total bytes per rank ≈ 2(N-1)/N × M ≈ 2M for large N — independent of N! vs naïve gather-and-broadcast: O(N × M) at the root this is why scaling DDP from 8 GPUs to 1024 GPUs only adds ~2× to the all-reduce

The result: doubling N barely changes the per-rank cost. This is the bandwidth-optimality that makes large-scale training possible. NCCL implements ring all-reduce by default for medium messages; for very small or very large messages, it switches to tree-based or double-binary-tree algorithms that have different latency/bandwidth tradeoffs.

## The PyTorch API

The collective ops live in `torch.distributed`. Initialization first:
    
    
    import torch.distributed as dist
    import os
    
    def setup():
        dist.init_process_group(
            backend='nccl',                      # GPU collectives — gloo for CPU
            init_method='env://',                # reads MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank, world_size
    
    def cleanup():
        dist.destroy_process_group()

Each process is a _rank_ in the world. The launcher (`torchrun`, see below) sets the env vars; `init_process_group` reads them. NCCL is the GPU backend; it manages the actual NIC traffic and uses NVLink between same-host GPUs when available.

### The collectives themselves
    
    
    # Broadcast: rank 0 sends, others receive
    x = torch.randn(1024, device='cuda') if rank == 0 else torch.zeros(1024, device='cuda')
    dist.broadcast(x, src=0)             # in-place: every rank now has rank 0's data
    
    # All-reduce: in-place, sum across all ranks
    g = torch.randn(1024, device='cuda')
    dist.all_reduce(g, op=dist.ReduceOp.SUM)   # every rank now has the sum
    
    # All-gather: each rank's tensor → list of tensors on every rank
    local = torch.randn(256, device='cuda')        # my slice
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)            # gathered[i] is rank i's local tensor
    
    # Reduce-scatter: list of tensors per rank → each rank gets one summed slice
    inputs = [torch.randn(256, device='cuda') for _ in range(world_size)]
    out = torch.empty(256, device='cuda')
    dist.reduce_scatter(out, inputs)            # out = sum_i inputs_per_rank[i][my_rank]

Three things to note about the API. (1) Collectives are **in-place by default** for `all_reduce` and `broadcast` — they modify their argument tensor. (2) The dtype must match across ranks — passing fp32 on rank 0 and fp16 on rank 1 will deadlock or error. (3) The shapes must match. Both are easy to violate when conditional code paths differ across ranks.

> **⚠ WARNING**  
>  **Asymmetric calls = silent hang.** If rank 0 calls `all_reduce` and rank 1 forgets to (e.g., skips because of a divergent branch), every rank waits forever. NCCL has a timeout (default 30 minutes — yes, really) but the "hang" looks like training just stopped logging. Symptom: trace looks fine until step N, then nothing. Diagnosis: print rank ID at every collective call site; the rank that gets stuck is the one that didn't call. **Always make sure every collective is called by every rank with matching args.**

## The launcher: `torchrun`

You don't manually start N processes. Use the launcher:
    
    
    # Single node, 8 GPUs:
    torchrun --nproc_per_node=8 train.py
    
    # Two nodes, 8 GPUs each (run on each node):
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
             --rdzv_backend=c10d --rdzv_endpoint="node0:29500" train.py
    
    # Same on node 1, but with --node_rank=1

What torchrun does: spawns N processes per node, sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` env vars in each, then runs your script. Inside your script, you read these via `init_process_group(init_method='env://')`.

The two ranks worth knowing:

  * **Global rank** (`dist.get_rank()`): your unique ID across the whole job (0 to world_size-1).
  * **Local rank** (`os.environ['LOCAL_RANK']`): your ID on this node (0 to gpus_per_node-1). Used for `cuda:LOCAL_RANK` device selection.

The pattern `torch.cuda.set_device(int(os.environ['LOCAL_RANK']))` is what binds each process to its dedicated GPU. Get this wrong and multiple processes contend for the same GPU.

## Bandwidth math: predict before you scale

Like memory and time before it, distributed cost is calculable. For an all-reduce of message size M bytes across N ranks on a network with bandwidth B bytes/sec:
    
    
    time_per_allreduce ≈ 2 × M / B               # for large enough N

Worked example. DDP all-reduces all gradients after every backward. For a 1B-param model in fp32, gradient size is 4 GB. On NVLink 4 (~600 GB/s within a node), that's `2 × 4 GB / 600 GB/s ≈ 13 ms` per step — negligible if your step takes 200 ms. On 100 Gb/s ethernet between nodes, it's `2 × 4 GB / 12.5 GB/s ≈ 640 ms` — that's eating your training time.

This is why intra-node bandwidth (NVLink) is so much higher than inter-node (PCIe + ethernet/InfiniBand): you want all the cheap collectives to happen within a node, and only the unavoidable ones to cross the network. Modern training schemes (FSDP, hybrid sharded data parallel) are explicitly designed around this hierarchy.

#### Q&A; — About communication primitives **Q:** Why is there a separate `reduce` when we have `all_reduce`? **A:** `reduce` sums to one rank; `all_reduce` sums and gives the result to every rank. `reduce` is half the bandwidth of `all_reduce` (skips the gather phase). Useful when only one rank actually needs the result — e.g., logging from rank 0, or saving a checkpoint. **Q:** What about `scatter` and `gather`? **A:** `scatter` is the inverse of `gather`: one rank has a list, scatters one element to each other rank. `gather`: every rank has data, one rank ends up with the list. They're less common in modern training (mostly subsumed by all_gather and all_to_all), but show up in some sharding schemes and in debugging. **Q:** When is `all_to_all` used? **A:** In Mixture-of-Experts (M28). Each token gets routed to a specific expert on a specific rank; the routing is essentially a transpose of the data layout, which is exactly what `all_to_all` does. Also used in some tensor-parallel attention patterns. **Q:** My collective hangs intermittently. What should I check? **A:** Order of the diagnosis tree: (1) Are all ranks calling the same collectives in the same order? Print rank ID at every collective. (2) Are dtypes and shapes identical across ranks? `print(g.dtype, g.shape)` right before. (3) Are you mixing CPU and CUDA tensors? NCCL only does GPU. (4) Is one rank exiting early (e.g., crashed, ran out of data)? The other ranks block forever. NCCL's `NCCL_TIMEOUT` env var lets you fail fast (set to e.g. 60 seconds for development). **Q:** Should I use `backend='gloo'` instead of NCCL for some workloads? **A:** Use NCCL for GPU. Use Gloo for CPU-only training (rare in modern DL). Gloo on GPU works but is much slower. NCCL also has better topology awareness — it knows about NVLink, NVSwitch, InfiniBand and routes accordingly. 

## Cost model: latency vs bandwidth

One more nuance. A collective has two costs:

  * **Latency** : a startup cost per call, typically 5-50 microseconds. Doesn't depend on message size.
  * **Bandwidth** : bytes-per-second cost, scales with message size.

For tiny messages, latency dominates and ring all-reduce isn't optimal — you want a tree algorithm with O(log N) depth. For huge messages, bandwidth dominates and ring is optimal. NCCL automatically picks the right algorithm based on message size.

The implication for your code: **fewer larger collectives are better than many small ones**. This is exactly what DDP's "gradient bucketing" exploits (M16) — instead of all-reducing each parameter's gradient as it becomes available (many small collectives, dominated by latency), DDP groups gradients into ~25 MB buckets and all-reduces each bucket (few medium collectives, near-optimal bandwidth utilization).

## Code Magnets: build a minimal distributed setup

Build a small distributed program that initializes process groups, all-reduces a tensor, and shuts down cleanly. Two magnets are red herrings.

Arrange the magnets into a working setup. Two are red herrings.

import os, torch, torch.distributed as dist dist.init_process_group(backend='nccl') dist.init_process_group(backend='gloo') rank = dist.get_rank(); world_size = dist.get_world_size() torch.cuda.set_device(int(os.environ['LOCAL_RANK'])) torch.cuda.set_device(rank) x = torch.tensor([float(rank)], device='cuda') dist.all_reduce(x, op=dist.ReduceOp.SUM) print(f"rank {rank}: x = {x.item()}") dist.destroy_process_group()

show solution
    
    
    import os, torch, torch.distributed as dist
    
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank(); world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    
    x = torch.tensor([float(rank)], device='cuda')
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    print(f"rank {rank}: x = {x.item()}")
    
    dist.destroy_process_group()

Run with: `torchrun --nproc_per_node=4 program.py`. Each rank starts with `x = float(rank)`; after all-reduce, every rank has `0+1+2+3 = 6.0`.

The traps:

  * `dist.init_process_group(backend='gloo')`: Gloo is for CPU. For GPU collectives, use NCCL.
  * `torch.cuda.set_device(rank)`: works for single-node, but breaks for multi-node where global rank can exceed the per-node GPU count. Always use `LOCAL_RANK`.

## Who does what?

Match each collective concept to its real role.

Concept

Real role

broadcast

A. Sum across ranks; result goes to every rank.

all_reduce

B. One rank's data → every rank.

all_gather

C. Each rank has a slice; after, every rank has the concatenation of all slices.

reduce_scatter

D. Sum across ranks, but each rank only keeps its slice — used in FSDP for gradients.

Ring all-reduce

E. Bandwidth-optimal algorithm: per-rank traffic ≈ 2M, independent of N.

LOCAL_RANK

F. Per-node rank index — what you pass to `torch.cuda.set_device`.

Gradient bucketing

G. Group small gradients into larger buckets to amortize collective latency.

show solution

**broadcast** → B  
**all_reduce** → A  
**all_gather** → C  
**reduce_scatter** → D  
**Ring all-reduce** → E  
**LOCAL_RANK** → F  
**Gradient bucketing** → G 

The mental shortcut: _broadcast = 1→all, all_reduce = sum-everywhere, all_gather = concat-everywhere, reduce_scatter = sum-then-shard, ring is optimal, LOCAL_RANK selects GPU, bucketing amortizes latency_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team's distributed run hangs at step 47 of training. `nvidia-smi` shows all GPUs at 0% utilization. What's the most likely cause and the diagnostic?

show answer

An asymmetric collective. One rank skipped a collective (often because of a conditional that's true on some ranks but not others — e.g., "only do extra work on rank 0 every 10 steps") and now the others wait forever. Diagnosis: print rank id and a tag at every collective call site. The rank with the missing tag is the culprit. Quick fixes: replicate the work on all ranks, or use a barrier (`dist.barrier()`) before the divergent code so you can spot the imbalance fast. Setting `NCCL_TIMEOUT` low (60s) during development makes hangs fail fast instead of silently.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** You're all-reducing a 100 MB tensor across 16 GPUs on NVLink (600 GB/s effective). What's the rough wall-clock time?

show answer

Ring all-reduce: `2 × M / B = 2 × 0.1 GB / 600 GB/s ≈ 0.33 ms`. Roughly negligible compared to a typical training step. Now do it on a 100 Gb/s ethernet network (12.5 GB/s effective): `2 × 0.1 / 12.5 = 16 ms` — getting noticeable. This is exactly why intra-node communication is faster — and why FSDP's ZeRO-3 (M17) tries to keep the heaviest collectives (parameter all-gathers) within a node when possible.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** Implement a manual gradient sync for a model trained on N ranks (assume the model and grads exist on each rank already). What single collective do you need?

show answer
    
    
    def sync_gradients(model, world_size):
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world_size       # average rather than sum

This is what DDP does internally, except DDP _buckets_ gradients (groups them into ~25 MB chunks) so the collective is amortized over many parameters. Calling `all_reduce` per-parameter (as above) hits the latency cost N times. Module 16 covers DDP's bucketing.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Why does `all_reduce ≡ reduce_scatter + all_gather**? Sketch why this matters for FSDP.

show answer

The identity: an all-reduce produces the full sum on every rank. You can split that into two phases: (1) reduce-scatter — each rank computes the sum of _its own slice_ ; (2) all-gather — each rank's slice gets distributed to everyone. Total bandwidth is unchanged, just split.

Why FSDP cares: in ZeRO-3 / FSDP, each rank only needs to _store_ its own slice of each parameter and gradient (1/N memory). For the optimizer step, each rank only needs the sum of _its_ gradient slice — so reduce-scatter is enough. The all-gather only happens at parameter-fetch time during the forward pass. By splitting the all-reduce, you save not bandwidth but _memory_ : each rank holds 1/N of each tensor instead of all of it. This is the core trick behind sharding (M17).

### What just happened?

  * **Five primitives** are the foundation of all distributed deep learning: broadcast, all-reduce, all-gather, reduce-scatter, and (for MoE) all-to-all.
  * **Two key identities** : `all_reduce = reduce_scatter + all_gather` (the FSDP/ZeRO insight) and `all_gather = N broadcasts but cheaper`.
  * **Ring all-reduce** is bandwidth-optimal: per-rank cost ≈ `2(N-1)/N × M ≈ 2M`, independent of N.
  * The naive "gather to root, broadcast back" is O(N × M) at the root — unworkable at scale. NCCL never does this.
  * **NCCL** is the GPU collectives backend; **Gloo** is for CPU. Always use NCCL for GPU training.
  * **Process groups** : every rank calls `init_process_group` before doing any collective. The launcher (`torchrun`) sets the env vars; `init_method='env://'` reads them.
  * **Two ranks** : global rank (across the whole world), local rank (within this node — used to pick the GPU).
  * `torchrun --nproc_per_node=8` for single-node; add `\--nnodes`, `\--node_rank`, `\--rdzv_endpoint` for multi-node.
  * **Asymmetric calls cause silent hangs.** Every rank must call every collective in matching order with matching dtypes and shapes. Print rank IDs at call sites for diagnosis.
  * **Bandwidth math** : `time ≈ 2M / B` per all-reduce. For a 1B-param fp32 gradient on NVLink: ~13 ms. On 100 Gb ethernet: ~640 ms. Network choice matters a lot.
  * **Latency vs bandwidth** : small messages dominated by latency, large by bandwidth. Bucket small ops into larger collectives — this is exactly what DDP does (M16).
  * The reflex: when distributed training is slow or hangs, ask _which collective is firing how often, with what message size, on what network_. The bandwidth math tells you whether the cost is reasonable; if it isn't, the schedule (DDP / FSDP / TP) is the lever.

Module 16 takes these primitives and assembles the most common distributed pattern: Distributed Data Parallel. Same model on every GPU, different data shards, all-reduce after backward. The bucketing trick, the find-unused-parameters foot-gun, and the gradient-overlap optimization that makes DDP scale.
