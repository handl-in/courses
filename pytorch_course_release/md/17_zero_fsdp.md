# Module 17 — ZeRO & FSDP

# ZeRO & _FSDP_

_Part VI · Module 17_

— sharding parameters, gradients, and optimizer state across ranks so a 70B model can train on hardware that couldn't hold its optimizer state, let alone its activations

\--- 

DDP has one limit: _the model has to fit on a single GPU_. Add the master fp32 copy, gradients in fp32, and Adam's two moment buffers, and you're at 18 bytes per parameter (M12). For 7B parameters, that's 126 GB — more than an 80 GB H100. DDP is done.

The fix is conceptually simple: **each rank stores only 1/N of every parameter and 1/N of every optimizer state**. When forward needs a layer's full weight, all ranks gather their slices to materialize it; do the matmul; free the gathered copy. Same for backward. Gradients get reduce-scattered (not all-reduced) so each rank ends up with only its slice's worth of gradient to feed to its slice's worth of optimizer state. The math is the M15 identity: `all_reduce ≡ reduce_scatter + all_gather`, but now we exploit it for memory rather than just bandwidth.

> **★ KEY IDEA**  
>  ZeRO has **three stages** of progressively aggressive sharding. _ZeRO-1_ : shard optimizer state across ranks (Adam moments, master copy). _ZeRO-2_ : also shard gradients. _ZeRO-3_ (= FSDP): also shard parameters. Each stage saves more memory at the cost of more communication. ZeRO-3/FSDP is the modern default for large models. The implementation is one wrapper line; the conceptual cost is understanding that **parameters are no longer always on the GPU when you need them — they get materialized per layer**. 

## The new face

⊟

Shard

"I'm a slice of a parameter. My rank owns me; the others own theirs."

Before, you had one parameter per Module. Now there are N of me — one per rank — each holding 1/N of the original tensor's bytes. We sit quiet most of the time. When forward arrives at our layer, all ranks all-gather us to reconstitute the full parameter, just in time for the matmul. Then we're freed again. Backward does the same — gather, compute, free, then reduce-scatter the gradient so each rank gets only the slice it owns. _I am ephemeral as a whole; only my slices persist._

## The memory math, made painful

Let's revisit M12's accounting in the context of multi-GPU training. Per-rank memory under each scheme, for a model with N parameters in mixed-precision AdamW, on world_size W GPUs:

Per-rank memory in different schemes (mixed-precision AdamW) | bf16 params| fp32 master| grads (fp32)| Adam (m,v fp32)| Total/rank  
---|---|---|---|---|---  
**DDP**|  2N| 4N| 4N| 8N| 18N  
**ZeRO-1** (shard optimizer)| 2N| 4N/W| 4N| 8N/W| 6N + 12N/W  
**ZeRO-2** (+ shard grads)| 2N| 4N/W| 4N/W| 8N/W| 2N + 16N/W  
**ZeRO-3 / FSDP**|  2N/W| 4N/W| 4N/W| 8N/W| 18N/W  
  
Read the rightmost column. DDP is `18N` per rank — the same as single-GPU. ZeRO-1 saves the largest single bucket (Adam). ZeRO-2 saves another big chunk (gradients). ZeRO-3 shards _everything_ , giving you full `1/W` scaling on fixed-cost memory.

For a 7B model on 8 GPUs:

  * DDP: 126 GB/rank → won't fit on 80 GB H100
  * ZeRO-1: `6×7e9 + 12×7e9/8 ≈ 52.5 GB/rank` → fits
  * ZeRO-2: `2×7e9 + 16×7e9/8 ≈ 28 GB/rank` → comfortable
  * ZeRO-3: `18×7e9/8 ≈ 15.75 GB/rank` → lots of room for activations

For a 70B model on 64 GPUs, the only option is ZeRO-3 — the others still wouldn't fit. **Modern training uses ZeRO-3 (FSDP) by default at scale** because it's the only thing that scales to arbitrarily large models.

Per-rank fixed-cost memory: 7B model, world_size=8 GB / rank 130 90 45 15 bf16 params (14) fp32 master (28) grads fp32 (28) Adam (56) DDP 126 GB ✗ won't fit bf16 (14) grads fp32 (28) ↑shrunk 8x ZeRO-1 52.5 GB ✓ bf16 (14) all sharded except bf16 ZeRO-2 28 GB ✓✓ everything 1/W ZeRO-3 15.75 GB ✓✓✓ 80 GB H100

The 80 GB line shows the H100 limit. DDP overshoots; the rest fit progressively more comfortably, leaving room for activations. **Activations still scale with batch size and sequence length** — they're not affected by sharding (each rank still computes its full activations for its data). So the savings shown above only solve the _fixed-cost_ memory problem; you still need activation checkpointing (M12) and FlashAttention (M25) for the variable-cost part.

## How ZeRO-3 / FSDP works (the dance)

The key insight: _you don't need the parameter materialized everywhere all the time. You only need it during the layer that uses it._

So FSDP wraps the model into a tree of FSDP units (typically each transformer layer is one unit). At init, each unit's parameters are sharded — split across W ranks, each rank holds 1/W. The flow during forward pass through a unit:

  1. **All-gather** the unit's parameter shards into the full parameter on every rank.
  2. **Forward** through the unit — it now has the full parameter and runs normally.
  3. **Free** the gathered copy on every rank. The shards persist; the materialized full weight is gone.

The flow during backward is the symmetric mirror — re-gather, compute the gradient, reduce-scatter the gradient (so each rank gets only its slice's worth), free.

FSDP forward/backward through one unit (e.g., one transformer layer) Forward through layer L: shards each rank: 1/W all_gather full param briefly on every rank forward() activations to next layer free shards full param freed Backward through layer L: shards re-gather needed all_gather full param \+ ∂L/∂out backward() full grad on every rank reduce_scatter grad shard 1/W on each rank forward all-gather → compute → free → next layer backward re-gather → compute grad → reduce-scatter → free double the all-gather cost, but each rank stores only 1/W of everything

Two things to internalize:

  1. **Each layer's full parameters exist on every rank only briefly** — for the duration of that layer's forward, then they're freed. Same in backward.
  2. **The all-gather happens twice per layer per step** (once in forward, once in backward). FSDP's communication cost is roughly _1.5×_ DDP's (DDP does a single all-reduce at backward end, FSDP does two all-gathers + a reduce-scatter per layer).

The 1.5× communication overhead is the price of memory savings. On well-connected hardware (NVLink within node, fast InfiniBand between), this overhead is mostly hidden behind compute via the same prefetching trick DDP uses. On bandwidth-limited setups, you'll see real overhead.

## The PyTorch FSDP API

PyTorch has two FSDP APIs:

  * **FSDP1** : `torch.distributed.fsdp.FullyShardedDataParallel` — the original, mature, what you'll see in most existing code.
  * **FSDP2** : `torch.distributed.fsdp.fully_shard` — the new per-parameter API, more flexible, integrates better with `torch.compile`. Modern recommended path.

Both share the same conceptual model. We'll show FSDP1 for clarity (most production code today), with notes on FSDP2.

### FSDP1 minimal setup
    
    
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    import functools
    
    # Decide which submodules to shard. For transformers: each block is its own FSDP unit.
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},     # your block class
    )
    
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,                   # param all-gathers in bf16
        reduce_dtype=torch.float32,                   # grad reduce-scatter in fp32
        buffer_dtype=torch.bfloat16,
    )
    
    model = build_model().to(device)
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,   # ZeRO-3
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        device_id=local_rank,
    )

Three knobs that matter:

  1. **`sharding_strategy`** : `FULL_SHARD` (ZeRO-3, default), `SHARD_GRAD_OP` (ZeRO-2), `NO_SHARD` (DDP-equivalent), `HYBRID_SHARD` (shard within node, replicate across — see below).
  2. **`auto_wrap_policy`** : which submodules to treat as sharding units. For transformers, each block is one unit. Smaller units = more communication; larger units = more transient memory.
  3. **`mixed_precision`** : per-stage dtype control. The standard recipe is bf16 for params and forward activations, fp32 for the gradient reduce-scatter (precision matters for Adam's `v`).

### FSDP2 (the future)
    
    
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    
    # FSDP2 wraps per-module, applied recursively
    for block in model.transformer.layers:
        fully_shard(block, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)

FSDP2 uses `DTensor` under the hood for cleaner integration with tensor parallelism (M18) and `torch.compile` (M21). The model on disk is a normal `state_dict` — no per-rank sharded checkpoints to wrangle. For new code, FSDP2 is the recommended path; existing FSDP1 code will keep working.

## Combining with activation checkpointing

FSDP solves the _fixed-cost_ memory problem. Activations are still per-rank (each rank computes its own data shard's activations), so they still scale with batch × seq × depth. For really big models you combine FSDP with activation checkpointing (M12):
    
    
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper, CheckpointImpl,
    )
    
    def apply_activation_checkpointing(model, target_cls):
        def check_fn(submodule):
            return isinstance(submodule, target_cls)
        torch.distributed.algorithms._checkpoint.checkpoint_wrapper.apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn,
        )
    
    # Apply to each transformer block
    apply_activation_checkpointing(model, target_cls=TransformerBlock)

The order matters: **apply activation checkpointing _before_ wrapping with FSDP**. Checkpoint inside the FSDP unit means the recompute happens after the all-gather, so the param stays gathered for the recompute pass — no extra communication.

This combination — FSDP + activation checkpointing + FlashAttention (M25) — is what makes 70B-scale model training possible on commodity (8× H100) clusters.

## Hybrid Sharded Data Parallel (HSDP)

The bandwidth math: all-gathers and reduce-scatters across many ranks are expensive. If you have 64 GPUs spread across 8 nodes, all-gathering a parameter across all 64 means crossing 7 inter-node hops. Painful.

HSDP exploits the topology: _shard within node, replicate across nodes_. Within a node (8 GPUs on NVLink), you do FSDP — fast intra-node all-gathers. Across nodes (slow ethernet/InfiniBand), you do DDP — one all-reduce per step.
    
    
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        ...
    )

The tradeoff: each node holds a full copy of the model (8× more fixed-cost memory than full FSDP), but inter-node communication is just one DDP all-reduce instead of many FSDP gathers. For models that fit on a single node, HSDP is often the fastest option. For models that don't fit, you're back to FULL_SHARD.

## CPU offload (for the truly desperate)

If even ZeRO-3 doesn't fit, you can offload optimizer state to CPU memory:
    
    
    from torch.distributed.fsdp import CPUOffload
    
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=True),
        ...
    )

The optimizer state and parameter shards live in CPU RAM; they get DMAed to GPU during the all-gather. This drastically slows training (PCIe bandwidth, not NVLink) but lets you train models that wouldn't otherwise fit anywhere. Use as a last resort.

## Checkpointing FSDP models

FSDP1's checkpoint story has historically been the worst part of the library. Each rank has only its slice of each parameter, so the obvious approach (each rank saves its own state_dict) gives you W files that are useless without all of them. Loading on a different world size is broken.

The modern fix uses `distributed_checkpoint` (DCP) which produces sharded files that can be reloaded on any world size:
    
    
    import torch.distributed.checkpoint as dcp
    
    # Save
    state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    dcp.save(state_dict, checkpoint_id=f"ckpt_step{step}")
    
    # Load — works on any world size
    state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    dcp.load(state_dict, checkpoint_id=f"ckpt_step{step}")
    # model and optimizer are now restored, with each rank having its own slice

For _final_ checkpoints (e.g., for inference or deployment), you usually want a full unsharded state_dict — DCP can dump that too via `torch.distributed.checkpoint.format_utils.dcp_to_torch_save`, or for FSDP1 you can use the older `FullStateDictConfig` \+ `StateDictType.FULL_STATE_DICT` patterns (look it up when needed).

#### Q&A; — About FSDP and ZeRO **Q:** When should I use ZeRO-1 instead of ZeRO-3? **A:** When (a) you have memory headroom and (b) you want to minimize communication. ZeRO-1 only does the optimizer-state shard; the rest of training proceeds like DDP. Less communication overhead than ZeRO-3, but only saves ~67% of fixed-cost memory (the Adam buckets) instead of close to 100%. If your model fits with ZeRO-1, you might not need ZeRO-3. **Q:** Why is FSDP slower per step than DDP at the same world size? **A:** FSDP does ~1.5× the communication of DDP (two all-gathers + a reduce-scatter per layer, vs DDP's single all-reduce). On a slow network, that overhead can be 10-30% of step time. The savings are in _memory_ , not throughput. The argument for FSDP is "this model didn't fit at all under DDP." If your model fits comfortably under DDP, use DDP. **Q:** Does FSDP work with torch.compile? **A:** FSDP1 has rough edges with compile. FSDP2 is designed for it — `fully_shard` \+ `torch.compile` work well together. For new projects requiring compile + sharding, use FSDP2. **Q:** My loss is different in FSDP vs DDP for the same model and data. Is that a bug? **A:** Probably not — small differences (≤1% in loss) are expected from numerical differences in the reduce-scatter vs all-reduce paths and from different mixed-precision casting points. The training trajectories should converge to similar quality. If the difference is large or the loss diverges, suspect a bug in your wrapping (e.g., layers that should be FSDP-wrapped but aren't, or weight tying not handled correctly across shards). **Q:** What about weight tying with FSDP? **A:** Tied weights (e.g., embedding == output projection from M7) are tricky under FSDP because FSDP shards each parameter independently — the "tying" can break. Modern FSDP detects tied weights and handles them, but you must do the tying _before_ wrapping. Always tie, then wrap. Test that `id(model.embed.weight) == id(model.head.weight)` after FSDP wrapping; if not, retie or refactor. 

## Code Magnets: build a FSDP setup for a transformer

You're setting up FSDP for a transformer with mixed precision, auto-wrapping each block, with activation checkpointing. Three magnets are wrong choices.

Arrange the magnets into a working setup.

model = build_model().to(device) apply_activation_checkpointing(model, target_cls=TransformerBlock) model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, auto_wrap_policy=transformer_wrap_policy, mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32), device_id=local_rank) model = DDP(model, device_ids=[local_rank]) opt = torch.optim.AdamW(model.parameters(), lr=3e-4) model = FSDP(model); apply_activation_checkpointing(model, target_cls=TransformerBlock)

show solution
    
    
    model = build_model().to(device)
    apply_activation_checkpointing(model, target_cls=TransformerBlock)
    model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=transformer_wrap_policy,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                        reduce_dtype=torch.float32),
        device_id=local_rank)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

The traps:

  * `model = DDP(model, device_ids=[local_rank])`: that's the DDP wrapper, not FSDP. Stacking on top of FSDP would error.
  * `model = FSDP(model); apply_activation_checkpointing(model, ...)`: **order matters**. Activation checkpointing must be applied _before_ FSDP wrapping. Otherwise the checkpoint boundary is around the FSDP unit, not the inner transformer block, and the recompute happens after each layer's gather+free — defeating the purpose.

The right order: **build → activation checkpointing → FSDP wrap → optimizer**. Optimizer must be created _after_ FSDP because FSDP modifies the parameter list (each rank only sees its shards).

## Who does what?

Match each FSDP/ZeRO concept to its real role.

Concept

Real role

ZeRO-1

A. Same as DDP for params/grads, but optimizer state sharded across ranks.

ZeRO-3 / FSDP

B. Everything sharded — params, grads, optimizer state — each rank holds 1/W.

FSDP unit

C. A submodule that is the granularity of all-gather/free; typically one transformer block.

HYBRID_SHARD

D. Shard within node, replicate across — exploits NVLink vs network bandwidth gap.

reduce-scatter (in FSDP)

E. The collective that produces only-my-slice gradients, replacing DDP's all-reduce.

Activation checkpointing before FSDP wrap

F. Ensures recompute happens after the all-gather, so it doesn't trigger another one.

Distributed Checkpoint (DCP)

G. Saves and loads sharded state_dicts that can be re-sharded on any world size.

show solution

**ZeRO-1** → A  
**ZeRO-3 / FSDP** → B  
**FSDP unit** → C  
**HYBRID_SHARD** → D  
**reduce-scatter (in FSDP)** → E  
**Activation checkpointing before FSDP wrap** → F  
**Distributed Checkpoint (DCP)** → G 

The mental shortcut: _ZeRO-1 shards optimizer, ZeRO-3 shards everything, FSDP unit is the gather granularity, HYBRID_SHARD is intra-node FSDP + inter-node DDP, reduce-scatter replaces all-reduce in FSDP, AC before wrap = recompute inside the gather, DCP for resharding_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team has a 13B transformer, 16 GPUs (2× 8-GPU nodes), and a fast NVLink network within each node but slow inter-node ethernet. Which sharding strategy should they pick?

show answer

HYBRID_SHARD. The model fits within a node under FSDP (per-rank fixed cost = 18×13B/8 ≈ 30 GB, comfortable on 80 GB H100 with room for activations). HYBRID_SHARD will give them fast intra-node all-gathers (NVLink) for the heavy FSDP communication, and a single inter-node DDP all-reduce per step (one slow collective rather than many). FULL_SHARD across all 16 GPUs would force every all-gather across the slow inter-node link — much worse. The choice is a topology decision: pick the sharding boundary at the bandwidth boundary.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why is FSDP forward typically faster than FSDP backward for the same layer?

show answer

Forward needs one all-gather per layer (gather params, compute, free). Backward needs an all-gather _and_ a reduce-scatter — first to gather params for the backward pass through the layer, then to reduce-scatter the gradient down to per-rank shards. So backward has roughly 1.5-2× the communication of forward per layer. (This is also why activation checkpointing has interesting implications for FSDP: the recompute during backward triggers another forward all-gather. It's why "apply AC before FSDP wrap" matters — done right, the recompute uses the same gathered params as the original backward, costing nothing extra.)

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** Compute the per-rank fixed-cost memory for a 70B-parameter model in mixed-precision AdamW under DDP, ZeRO-1, ZeRO-2, and ZeRO-3, on 64 GPUs.

show answer

From the formula: `18N` (DDP), `6N + 12N/W` (ZeRO-1), `2N + 16N/W` (ZeRO-2), `18N/W` (ZeRO-3).

For N=70B, W=64:

  * DDP: 1260 GB ✗ (massively over)
  * ZeRO-1: `6×70 + 12×70/64 = 420 + 13.1 = 433 GB` ✗
  * ZeRO-2: `2×70 + 16×70/64 = 140 + 17.5 = 157.5 GB` ✗
  * ZeRO-3: `18×70/64 = 19.7 GB` ✓

For 70B at this world size, ZeRO-3 is the only option that fits. Plus activations on top, so realistically you also want activation checkpointing and FlashAttention. This is why the modern recipe at scale is "FSDP + activation checkpointing + FlashAttention + bf16."

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** A team's FSDP run trains correctly but their checkpoints can't be loaded on a different world size — they trained on 8 GPUs and want to resume on 16. What's the issue and the modern fix?

show answer

FSDP1's default state_dict is rank-local (each rank has its slice). Saving each rank's state_dict gives you W files that can only be reloaded on the same W. Modern fix: use `torch.distributed.checkpoint` (DCP). DCP saves a directory of sharded files with metadata; on load, it figures out the new world size's sharding and remaps. This makes resharding trivial — train on 8, resume on 16, fine-tune on 4, all without manual conversion. For older FSDP1 code, you can also force `FullStateDictConfig` at save time to gather everything onto rank 0, but it's slow at scale and has memory implications. DCP is the right answer for new code.

### What just happened?

  * **DDP runs out** when the 18N bytes of fixed-cost training state don't fit per rank. ZeRO/FSDP shards that state across ranks.
  * **ZeRO-1** : shard optimizer state (Adam moments + master copy). Saves the biggest single bucket. ~67% of fixed-cost savings.
  * **ZeRO-2** : also shard gradients. Now only the bf16 working params are replicated.
  * **ZeRO-3 / FSDP** : shard everything. Each rank holds 1/W of all training state. Modern default for large models.
  * **The dance** : forward all-gathers each layer's params, computes, frees. Backward re-gathers, computes grads, reduce-scatters to per-rank slices. ~1.5× DDP's communication, much less memory.
  * **FSDP unit** is the granularity of gather/free. For transformers, one block per unit. Smaller units = more communication; larger = more transient memory at peak.
  * **Mixed precision policy** : param all-gathers in bf16 (smaller messages), grad reduce-scatter in fp32 (precision matters for Adam).
  * **Combine with activation checkpointing** for the variable-cost (activation) memory. Apply AC _before_ FSDP wrapping so the recompute happens inside the gathered window.
  * **HYBRID_SHARD** : shard within node (fast NVLink), DDP across nodes (slow ethernet). Best when the model fits on one node and inter-node bandwidth is the bottleneck.
  * **CPU offload** : last-resort move of optimizer state to CPU. Lets you train models that wouldn't fit anywhere else, at significant throughput cost.
  * **Distributed Checkpoint (DCP)** is the modern way to save/load FSDP state. Sharded files, world-size agnostic, the right answer for new code.
  * **FSDP1 vs FSDP2** : FSDP1 mature, what most code uses. FSDP2 (`fully_shard`) cleaner, designed for `torch.compile`, recommended for new projects.
  * The reflex: pick the smallest sharding that fits. DDP if it fits. ZeRO-1 if you have headroom. FSDP if you need to scale up. HYBRID_SHARD if you're spread across nodes with a fast intra-node fabric.

Module 18 closes Part VI with the kinds of parallelism that aren't replication of work — _tensor parallelism_ (split a layer's matmul across ranks) and _pipeline parallelism_ (split the layers themselves into stages, each on different ranks, with micro-batches flowing through). For the largest models, you stack 2D, 3D, even 4D parallelism — data + tensor + pipeline + sequence. We'll see why and how.
