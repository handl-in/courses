# Module 18 — Tensor & pipeline parallelism

# Tensor & _pipeline parallelism_

_Part VI · Module 18_

— splitting a single matmul across ranks (tensor parallel), splitting the model into stages with micro-batches (pipeline parallel), and stacking these axes for the largest models

\--- 

DDP scaled data. FSDP scaled fixed-cost memory. There are two more axes of parallelism that show up at the very largest scale, and they're qualitatively different from sharding-then-reassembling: they actually _split work_. Tensor parallel splits a matmul across ranks. Pipeline parallel splits the model into stages, each on a different rank, with micro-batches flowing through.

You probably won't write tensor or pipeline parallel code from scratch — modern training stacks (Megatron-LM, DeepSpeed, NVIDIA's NeMo, PyTorch's `torch.distributed.tensor.parallel` and `pipeline_parallel`) handle the wiring. But you absolutely need to understand _what these schemes do_ , because hybrid setups (4D parallelism: data + tensor + pipeline + sequence) are the standard for frontier-scale models and the choice of which axis goes where determines whether your training runs in 30 days or 3 months.

> **★ KEY IDEA**  
>  **Tensor parallel (TP)** splits one operation across ranks — column-parallel splits a linear layer's output dim, row-parallel splits its input dim. An all-reduce stitches results back together at the end of each layer. Best _within a node_ (NVLink), where the all-reduce is cheap. **Pipeline parallel (PP)** splits the layer stack into stages, one stage per rank, with activations flowing rank-to-rank. Micro-batches keep all stages busy concurrently, minimizing the "bubble" of idle time. Best _across nodes_ , where bandwidth is scarce. **Sequence parallel (SP)** splits along the sequence dim — handles long contexts. Modern frontier training stacks all four: data + tensor + pipeline + sequence. 

## One new face

St

Stage

"I'm a slice of layers. Activations come in from the previous stage; outputs go to the next."

Pipeline parallelism turns the model into a relay race: stage 0 holds layers 0-7, stage 1 holds 8-15, and so on. I receive activations from the rank ahead of me, run my layers' forward, and send activations to the rank behind. In backward, the gradient flows the other way: I receive grad-of-output from behind, run backward through my layers, send grad-of-input ahead. **While my forward is running on micro-batch i, the rank ahead is on micro-batch i+1.** That's how the whole pipeline stays busy.

## Tensor parallelism: splitting a matmul

Inside a transformer block, the dominant compute is matrix multiplication. Two matmuls in attention (QKV projection, output projection) and two in the MLP (up, down). Each is a `(B, T, D) × (D, D')` matmul. Tensor parallel splits these across W ranks.

The trick is choosing _which dimension_ of which weight to split. For an MLP `x → W₁(linear, D→4D) → activation → W₂(linear, 4D→D) → y`, the canonical Megatron pattern:

  * Split `W₁` along its _output_ dim — each rank computes 1/W of the columns. _Column-parallel._
  * Apply the activation function (per-element, no communication needed).
  * Split `W₂` along its _input_ dim — each rank holds one slice of rows. _Row-parallel._
  * All-reduce the partial outputs at the end. Done.

Tensor parallel MLP: column-parallel W₁ + row-parallel W₂ x (B, T, D) replicated W₁⁰ W₁¹ W₁² W₁³ W₁ split by columns (D→4D, each rank gets 4D/4) column-parallel — each rank: x @ W₁ⁱ → h_i (B, T, 4D/4) σ(h_i) (B, T, 4D/4) activation per-element, no communication W₂⁰ W₂¹ W₂² W₂³ W₂ split by rows row-parallel all_reduce y (B, T, D) replicated column-parallel: split OUTPUT dim → each rank gets 1/W slice of output row-parallel: split INPUT dim → partial sums on each rank "col-then-row" pattern needs ONE all-reduce per MLP — bandwidth-cheap on NVLink in autograd terms: f = identity-fwd / all-reduce-bwd g = all-reduce-fwd / identity-bwd

The clever part is the dimension choice. Each rank computes `x @ W₁ⁱ` independently (no communication — `x` is replicated, `W₁ⁱ` is local). The activation runs element-wise, no communication. Then `h_i @ W₂ⁱ` on each rank produces a _partial_ output (the full sum requires all ranks' contributions). One all-reduce stitches them together.

One all-reduce per MLP block — and another for the attention block (the QKV column-split + output-row-split has the same shape). For a transformer with N layers, that's 2N all-reduces per forward, plus the same for backward. _This is why TP is best within a node_ : NVLink can absorb that many all-reduces; cross-node ethernet can't.

### The f/g operator notation

In Megatron's literature you'll see operators called `f` and `g`:

  * **f** : identity in forward, all-reduce in backward. Goes at the input boundary of a TP region.
  * **g** : all-reduce in forward, identity in backward. Goes at the output boundary.

The MLP becomes `y = g(σ(f(x) @ W₁) @ W₂)`. The `f` at the input is a no-op in forward (we already have `x` on every rank); but in backward, the gradient through it is the sum of all ranks' contributions, requiring an all-reduce. The `g` at the output is the all-reduce we drew above; in backward, it's a no-op (the gradient flows independently into each rank's row-parallel slice).

You don't write `f` and `g` by hand — modern PyTorch's `torch.distributed.tensor.parallel` module wires this for you via `parallelize_module` and column/row plans:
    
    
    from torch.distributed.tensor.parallel import (
        parallelize_module, ColwiseParallel, RowwiseParallel,
    )
    
    mp_plan = {
        "mlp.W1": ColwiseParallel(),
        "mlp.W2": RowwiseParallel(),
    }
    parallelize_module(model, tp_mesh, mp_plan)

The framework figures out the f and g insertions, hooks them into autograd, and you get a correctly TP'd model. Same idea for attention: column-parallel on QKV proj, row-parallel on the output proj, no other plumbing required.

## Pipeline parallelism: layers as stages

The other axis. Tensor parallel splits a single op across ranks; pipeline parallel splits the layer stack _sequentially_ : ranks 0-3 hold layers 0-7, ranks 4-7 hold layers 8-15, etc. Each rank only stores its own stage's parameters and only computes its own stage.

The naïve schedule is bad. If you just run forward through stage 0, then stage 1, then stage 2, etc., only one stage is busy at a time — you've gone from "1 GPU does the whole forward" to "8 GPUs do the whole forward, but serially." Same total time, more hardware. Useless.

The fix is **micro-batches**. Split the batch into M micro-batches and pipeline them through:

Pipeline parallelism: micro-batches keep all stages busy ① Naïve: one batch, sequential stages St 0 St 1 St 2 St 3 F F F F B B ← 3 stages idle → total time ≈ N_stages × forward + N_stages × backward — most stages idle most of the time ② 1F1B: micro-batches keep all stages busy St 0 St 1 St 2 St 3 F1 F2 F3 F4 B1 F5 B2 F6 B3 B4 B5 B6 F1 F2 F3 F4 B1 F5 B2 F6 B3 B4 B5 F1 F2 F3 F4 B1 F5 B2 F6 B3 B4 F1 F2 F3 F4 B1 F5 B2 F6 B3 ↖ initial bubble: stages fill up steady-state: every stage busy final bubble: stages drain ↘ bubble fraction = (P − 1) / (M + P − 1) — more micro-batches → smaller bubble

Read the bottom panel. With M=6 micro-batches and P=4 stages, after the initial fill-up, every stage is processing _some_ micro-batch (forward or backward) every step. The 1F1B schedule (alternate one forward, one backward) keeps memory low: a stage is processing at most P micro-batches' activations at once.

### The bubble math

The fraction of time stages are idle (the bubble):
    
    
    bubble_fraction = (P − 1) / (M + P − 1)

Where P = number of pipeline stages, M = number of micro-batches per training step. Concrete: P=8 stages, M=4 micro-batches → bubble = 7/11 ≈ 64% (terrible). M=64 → bubble = 7/71 ≈ 10% (acceptable). M=256 → bubble = 7/263 ≈ 2.7% (great).

This is why pipeline parallelism wants **large effective batch sizes** — you need many micro-batches to amortize the bubble. For LLM training where effective batch sizes are huge (~4M tokens), this is fine. For smaller-scale training, the bubble can dominate.

### Pipeline schedules: GPipe vs 1F1B vs interleaved

Three pipeline schedules Schedule| Memory cost| Tradeoff  
---|---|---  
**GPipe**|  All M micro-batches' activations on every stage at peak| Simple to reason about, but huge activation memory  
**1F1B** (one forward, one backward)| Up to P micro-batches' activations at peak| The standard. Balances memory and bubble.  
**Interleaved 1F1B**|  Smaller bubble, more communication| Each stage gets multiple "virtual" stages, smaller bubble at the cost of more sends/recvs. Used in Megatron's largest configurations.  
  
1F1B is the modern default for transformer pretraining. Interleaved is a Megatron-LM specialty for the very largest setups (1T-parameter scale).

## Sequence parallelism (briefly)

One more axis. _Sequence parallel_ splits along the sequence dimension T instead of the hidden dimension D or the layer index. It's a refinement of TP: in regions where the work is purely per-token (LayerNorm, activation, dropout), there's no need to replicate `(B, T, D)` tensors across TP ranks — you can split T across them and each rank processes its own slice.

The benefit: activation memory drops by a factor of TP_size in those regions. For long-context training (T=32768+), this matters because activation memory dominates. The cost: extra all-gather/reduce-scatter at the boundaries between sequence-parallel and tensor-parallel regions. Megatron-style codebases combine TP and SP together; there's roughly no extra communication compared to TP-only because they amortize.

You'll see SP mentioned in any modern large-model paper. Conceptually: it's TP that exploits per-token operations.

## 4D parallelism: the modern stack

Frontier training uses all four:

  * **Data parallel (DP)** : shards the batch across nodes. Cheapest communication (one all-reduce after backward, M16). Always present.
  * **Tensor parallel (TP)** : splits a matmul within a layer. Stay within a node — needs NVLink for the per-layer all-reduces.
  * **Pipeline parallel (PP)** : splits the model into stages across nodes. Tolerates lower bandwidth (one send/recv between stages per micro-batch).
  * **Sequence parallel (SP)** : tags onto TP. Cuts activation memory along the sequence dim.

The standard pattern at scale: **TP within a node (8 ranks), PP across nodes (e.g., 8 stages), DP/FSDP across pipeline copies, SP integrated with TP**. Total ranks = TP × PP × DP. For a 70B model on a 256-GPU cluster: TP=8, PP=4, DP=8 (8×4×8=256). The choice of axis sizes is a topology optimization — match the heavier collectives to faster fabrics.

#### Q&A; — About TP, PP, and friends **Q:** When do I need TP at all? Can't I just use FSDP? **A:** For models that fit a single-layer's parameters comfortably on one GPU, FSDP is enough. TP becomes essential when _even a single layer is too big_ — for example, a 70B+ model where one layer's weights exceed a GPU's memory after FSDP gathers them. TP splits the layer itself, so even the materialized weight is sharded. At extreme scale (100B+), TP is mandatory. **Q:** Can I combine FSDP with TP? **A:** Yes — that's the standard pattern. FSDP shards across the data-parallel dimension; TP shards within each layer. PyTorch's `DTensor` abstraction handles the multi-axis sharding cleanly. The combination is sometimes called "2D parallel." **Q:** Why doesn't TP scale beyond a node? **A:** Bandwidth. TP does one all-reduce per layer (or two with attention + MLP). For 32 layers × 2 collectives = 64 all-reduces per forward + same for backward = 128 collectives per step. Each is on the order of a megabyte. On NVLink (600 GB/s), each is a few microseconds. On 400 Gbps inter-node (~50 GB/s), 10× slower, plus higher latency per call. TP outside a node tanks throughput. **Q:** My PP run has correct loss but is incredibly slow. What gives? **A:** Probably bubble dominance. Check M (micro-batch count) vs P (stage count). If M is comparable to P, your bubble fraction is near 50%. Increase micro-batches by either (a) more accumulation steps per step (more micro-batches, same effective batch) or (b) smaller per-microbatch batch size (more micro-batches at the same total batch). Aim for M ≥ 4P. **Q:** Is TP/PP coming or going? **A:** Coming. Models keep growing. FSDP-only training plateaus around ~100B parameters depending on hardware. Above that, TP and PP are mandatory. PyTorch's `torch.distributed.tensor.parallel` module is actively developed; the new DTensor + `parallelize_module` API is much cleaner than older Megatron-style hand-coded TP. Expect this area to be a big theme over the next few years. 

## Where each parallelism lives

A simple mental model that captures the design pressures:

Choosing the right axis for the right hardware Parallelism| What it scales| Bandwidth need| Where it goes  
---|---|---|---  
Data parallel| Effective batch size| Low (one all-reduce/step)| Outermost — across nodes / racks  
FSDP / ZeRO-3| Fixed-cost memory| Medium-high (per-layer all-gather + reduce-scatter)| Middle — usually intra-node, sometimes hybrid  
Tensor parallel| Single-layer compute & param size| High (per-layer all-reduce)| Innermost — within a node, on NVLink  
Sequence parallel| Activation memory at long T| Bundled with TP| Within TP groups  
Pipeline parallel| Total parameter count| Low-medium (send/recv between adjacent stages)| Across nodes  
  
The principle: **match the heavier collective to the faster fabric**. NVLink (600 GB/s within node) is for TP. InfiniBand or fast ethernet (~50-200 GB/s between nodes) is for FSDP or PP. The slowest fabric (or the cheapest collective) is for DP.

## Code Magnets: assemble a 2D parallel setup

You're configuring 2D parallelism: TP=4 within each node, FSDP across the rest. World size 16 means 4 TP groups of 4 GPUs each. Three magnets are wrong choices.

Arrange the magnets into a working setup.

from torch.distributed.device_mesh import init_device_mesh mesh = init_device_mesh("cuda", (4, 4), mesh_dim_names=("dp", "tp")) mesh = init_device_mesh("cuda", (16,), mesh_dim_names=("dp",)) tp_plan = {"mlp.W1": ColwiseParallel(), "mlp.W2": RowwiseParallel()} parallelize_module(model, mesh["tp"], tp_plan) model = FSDP(model, device_mesh=mesh["dp"], sharding_strategy=ShardingStrategy.FULL_SHARD) model = FSDP(model, device_mesh=mesh, sharding_strategy=ShardingStrategy.FULL_SHARD) model = DDP(model)

show solution
    
    
    from torch.distributed.device_mesh import init_device_mesh
    
    mesh = init_device_mesh("cuda", (4, 4), mesh_dim_names=("dp", "tp"))
    tp_plan = {"mlp.W1": ColwiseParallel(), "mlp.W2": RowwiseParallel()}
    parallelize_module(model, mesh["tp"], tp_plan)
    model = FSDP(model, device_mesh=mesh["dp"], sharding_strategy=ShardingStrategy.FULL_SHARD)

The traps:

  * `(16,)` mesh with one dim called "dp": no TP at all, just plain FSDP.
  * `FSDP(model, device_mesh=mesh, ...)` without selecting a sub-mesh: tries to FSDP-shard across all 16 ranks including the TP dim, which double-shards the already-TP'd parameters.
  * `DDP(model)` after TP: DDP doesn't know about TP-sharded parameters — would all-reduce TP shards as if they were complete tensors.

The right pattern: **build a 2D mesh, parallelize_module on the TP submesh, FSDP on the DP submesh**. Each parallelism gets its own slice of the mesh.

## Who does what?

Match each parallelism concept to its real role.

Concept

Real role

Column-parallel

A. Splits a linear layer's output dim across ranks; each rank owns a column slice.

Row-parallel

B. Splits a linear layer's input dim; outputs are partial sums needing all-reduce.

Pipeline stage

C. A contiguous set of layers held by one rank; passes activations to next stage.

1F1B schedule

D. Alternates forward/backward per stage; balances bubble vs activation memory.

Bubble fraction (P-1)/(M+P-1)

E. Shows why you need M ≥ 4P micro-batches for pipeline efficiency.

Sequence parallel

F. Splits along T in per-token regions; reduces activation memory in TP groups.

4D parallelism

G. Stack DP + FSDP + TP + PP (+ SP); match heaviest collective to fastest fabric.

show solution

**Column-parallel** → A  
**Row-parallel** → B  
**Pipeline stage** → C  
**1F1B schedule** → D  
**Bubble fraction (P-1)/(M+P-1)** → E  
**Sequence parallel** → F  
**4D parallelism** → G 

The mental shortcut: _col-par splits outputs, row-par splits inputs (with all-reduce), stages are layer slices, 1F1B alternates F/B, bubble math sets M, SP cuts activation along T, 4D matches collectives to fabric speed_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team's PP=8 run has bubble fraction 25%. Their effective batch is 256, micro-batch=8 (so M=32). Compute the bubble; what M would they need for 5%?

show answer

Bubble = (P-1)/(M+P-1) = 7/(32+7) = 7/39 ≈ 18%. Their 25% number is probably from including communication overhead beyond just the bubble. For 5% bubble: 7/(M+7) = 0.05 → M+7 = 140 → M = 133. With per-microbatch batch=8, effective batch becomes 8×133 = 1064. Doable for LLM pretraining (effective batches in the millions are normal); pricey if your effective batch is otherwise constrained. The takeaway: **bubble is why pipeline parallelism is most useful at very large effective batch sizes**.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why does column-parallel-then-row-parallel need only one all-reduce, while column-parallel-then-column-parallel would need two?

show answer

Column-parallel splits the _output_ dim, so each rank's result has shape `(B, T, D'/W)` — a _partial_ output along the feature dim, no summing needed yet. Then row-parallel splits the _input_ dim, so each rank computes a partial sum of the _final_ output and one all-reduce stitches them together. Each rank's chunk lines up with the next layer's input partition: _no all-reduce in between_.

If you did column-parallel-then-column-parallel: rank 0 has columns 0..D'/W of the first output, but the second column-parallel layer needs the _full_ activation as input. So you'd have to all-gather the first output (or all-reduce some equivalent), then proceed. Two collectives instead of one. Megatron's "col-then-row" pattern is the cheapest schedule.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** A team is training a 175B model on 1024 GPUs across 128 nodes. They configure 4D parallelism. Suggest reasonable axis sizes and explain.

show answer

One common configuration: **TP=8, PP=8, DP=16** (8 × 8 × 16 = 1024). TP=8 fits within a single 8-GPU node (NVLink). PP=8 stages spread across 8 nodes — each pipeline copy spans 8 nodes via inter-node communication. DP=16 means 16 copies of the entire pipeline — one all-reduce per training step across DP, on the slowest fabric. Effective batch is 16× the micro-batch × pipeline-microbatches.

The principle: TP gets the fastest fabric (NVLink within a node), PP the next-fastest (inter-node, but only point-to-point sends per micro-batch), DP the slowest (one all-reduce per step). Different teams pick slightly different splits — Llama 3 used TP=8, PP=16, DP=N for the 405B run, for instance.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** What's the difference, conceptually, between FSDP all-gather and TP's column-parallel? Both gather across ranks before a matmul...

show answer

FSDP _materializes a complete parameter_ on each rank temporarily, then runs the layer normally, then frees. The matmul each rank does is the _full_ matmul on its full data — same compute as single-GPU.

TP column-parallel _doesn't materialize_ the complete weight. Each rank only ever has its slice of the weight. The matmul each rank does is a smaller matmul (1/W of the columns). The "gather" in TP is conceptually about gathering _partial outputs_ , not weights.

Memory: FSDP saves per-rank weight storage (1/W) but recomputes the full weight at use time. TP also stores 1/W but never needs the full thing. _So TP is strictly less memory than FSDP for that one layer._ Compute: FSDP does full-size matmuls; TP does smaller matmuls in parallel — the per-rank compute is less, but you need W ranks running concurrently. They're solving different problems: FSDP shards _all training state_ , TP shards _compute_.

### What just happened?

  * **Tensor parallel (TP)** splits a single matmul across W ranks: column-parallel splits output dim, row-parallel splits input dim, "col-then-row" needs one all-reduce per region.
  * **The f/g operators** : f = identity-fwd / all-reduce-bwd at TP entry; g = all-reduce-fwd / identity-bwd at TP exit. Modern PyTorch's `parallelize_module` wires these for you.
  * **TP is high-bandwidth** — many small all-reduces per step. **Stay within a node** (NVLink). Across nodes, TP tanks throughput.
  * **Pipeline parallel (PP)** splits the layer stack into P stages, one per rank. Each stage receives activations from the previous rank, processes its layers, sends to the next.
  * **Naïve PP is bad** — only one stage runs at a time. **Micro-batches** fix it: split a batch into M chunks; the pipeline stays full once it's primed.
  * **1F1B** is the standard schedule: alternate one forward, one backward. Bounds activation memory at ~P micro-batches per stage.
  * **Bubble fraction** = (P-1)/(M+P-1). Need M ≥ 4P or so for acceptable efficiency. PP wants _large effective batches_ — fits naturally for LLM pretraining.
  * **Sequence parallel (SP)** rides on top of TP: splits the per-token computations along T. Shrinks activation memory at long context. Bundled with TP in modern frameworks.
  * **4D parallelism** (DP × FSDP × TP × PP × SP) is the modern stack. Match the heavier collective to the faster fabric: TP intra-node, PP/FSDP inter-node, DP outermost.
  * **FSDP vs TP** : FSDP materializes the full weight per layer briefly; TP never materializes it. Different memory/compute tradeoffs; combine for very large models.
  * The reflex: when picking a parallelism plan, draw your hardware topology, mark the bandwidth between every pair, and place the most-communication-heavy axis on the fastest connection.

That closes Part VI. You now have the full distributed toolkit: communication primitives (M15), data parallelism (M16), state sharding (M17), and operation+layer sharding (M18). Combined with the memory + profiling work from Part V, you can predict where any training run will be bound and pick the right tool.

Part VII pivots inward. Modules 19-21 open up the framework itself — the dispatcher and ATen layer, the CUDA caching allocator and stream model, and finally `torch.compile` as the system that fuses everything. That's the foundation for Part VIII, where we start writing the kernels.
