# Module 28 — MoE & frontier capstone

Part VIII · Module 28 · _finale_

# Mixture of Experts & _frontier capstone_

— how MoE breaks the parameter-count vs compute tradeoff, the all-to-all collective finally deployed for real, and one capstone problem that touches every module of this course

Welcome to the last module. Twenty-seven modules ago we were dissecting strides on a single tensor. We've come a long way. This module covers two things. First: **Mixture of Experts (MoE)** — the architectural pattern that powers Mixtral, DeepSeek-V3, GPT-4-class models, and most frontier-scale systems being built today. Second: a **capstone problem** that asks you to design the training and serving stack for a frontier-scale MoE model from scratch, drawing on every part of the course.

And then a closing note — what we did, what's left to learn, and where to point your effort next.

> **★ KEY IDEA**  
>  A standard transformer layer has one MLP. **An MoE layer has N parallel MLPs ("experts")** and a small **router** that picks the top-K (typically K=2) for each token. The forward pass: token goes to router, router picks 2 of 8 experts, those 2 experts compute, weighted sum is the output. _Compute scales with K, parameters with N_. A model with 8× the parameters does the same compute as a dense model — sparse activation in action. The price: experts must be sharded across ranks (the **all-to-all** collective from M15 is finally load-bearing), the router needs an auxiliary loss to prevent expert collapse, and the entire model has to be resident in GPU memory at inference time even though only K/N is active per token. **Frontier-scale models are MoE because the parameter-count vs compute tradeoff is the binding constraint at scale.**

## Two new faces (the last new ones)

E

Expert

"I'm one of N parallel MLPs. Most tokens never visit me; the few that do get my full attention."

In a standard transformer, every token goes through the same MLP. In MoE, there are 8 of me (or 32, or 128 in DeepSeek's case). The router picks K=2 of us per token. For most tokens I'm idle — and that's the point. The model has 8× the parameters of a dense version, but each token only activates 2/8 = 25% of them. _Same compute, much more capacity._ My expertise specializes during training: some experts handle code, others natural language, others math. Nobody hand-engineers this — it emerges from the routing.

→

Router

"I'm a tiny linear layer that decides which K experts each token visits."

For each token (a vector of dimension D), I produce a logit per expert (a vector of dimension N). I take the top-K and softmax over them — that gives me K experts and K weights summing to 1. The expert outputs are weighted by these and summed. I'm just a single linear layer plus a top-K — typically <0.1% of the model's parameters. The catch: _without a load-balancing loss, I collapse to always picking the same expert_ (it's a winner-take-all dynamic). The auxiliary loss penalizes uneven assignment, keeping me democratic. When I work, I learn meaningful specialization — that's the magic.

## The MoE layer, mechanically

Replace one MLP with this:
    
    
    class MoELayer(nn.Module):
        def __init__(self, d, n_experts=8, k=2):
            super().__init__()
            self.experts = nn.ModuleList([MLP(d) for _ in range(n_experts)])
            self.router = nn.Linear(d, n_experts)
            self.k = k
    
        def forward(self, x):                              # x: [B, T, D]
            logits = self.router(x)                       # [B, T, N_experts]
            weights, indices = logits.topk(self.k, dim=-1)  # [B, T, K], [B, T, K]
            weights = weights.softmax(dim=-1)              # normalize chosen-K weights
    
            # For each token, gather outputs from its chosen K experts
            out = torch.zeros_like(x)
            for e_idx in range(len(self.experts)):
                mask = (indices == e_idx).any(dim=-1)        # tokens that chose this expert
                if mask.any():
                    expert_out = self.experts[e_idx](x[mask])
                    # Multiply by the routing weight that selected this expert and add
                    # (simplified — real implementations are vectorized)
                    out[mask] += apply_weight(expert_out, weights, indices, e_idx)
            return out

Three things to highlight:

  1. **The router is a linear layer** : input dim D, output dim N_experts. Cheap. About 0.05% of the layer's parameters in typical configs.
  2. **Top-K + softmax** : pick the K best experts per token, then softmax over just those K to get weights that sum to 1. Tokens going to expert i contribute weight `weights[t, i]` to the final sum.
  3. **Sparse computation** : each expert only processes the tokens routed to it. If routing is balanced, each expert sees ~K/N of all tokens. Total compute per token: K experts × MLP cost = same as a dense model with 1 MLP at K=1, or 2× at K=2. _But the parameters are N× bigger_.

MoE block: router selects top-K experts, weighted sum is the output x [B, T, D] Router Linear(D, N) → top-K + softmax ~0.05% of params → selection e.g. token A → {1, 5} token B → {3, 5} \+ weights w₁, w₂ Experts (N=8 parallel MLPs): Expert 0 idle for token A Expert 1 A → here Expert 2 idle Expert 3 B → here Expert 4 idle Expert 5 A,B both → here Expert 6 Expert 7 output_t = w₁ · expert_a(x_t) + w₂ · expert_b(x_t) 8× parameters, 2× compute → "sparse activation" pattern

## Why MoE: the compute-vs-parameters tradeoff

Recall from M22: GPU performance has two regimes — compute-bound (limited by FLOPs) and memory-bound (limited by HBM bandwidth). Decoding LLMs is memory-bound on weight loads (M27). For a fixed compute budget, a model that has more parameters but the same per-token compute is "free" until you hit a memory-bandwidth wall.

Concretely: a dense 70B model spends 70B × 2 bytes = 140 GB of weight loads per token (in bf16). A MoE 8x7B model has 56B parameters total but only activates 2 of 8 experts per token: ~14B params worth of MLP loads + the (shared) attention layers. _Less weight bandwidth per token, larger model capacity_. Quality scales with parameter count; speed scales with active-per-token compute.

MoE vs dense: the parameter/compute decoupling Model| Total params| Active per token| Inference cost ≈  
---|---|---|---  
Llama 3 70B (dense)| 70B| 70B| Big and slow  
Mixtral 8x7B (8 experts, K=2)| ~47B| ~13B| Quality of 70B at speed of 13B  
DeepSeek-V3 (256 experts + 1 shared, K=8 + 1)| 671B| ~37B| Quality of much larger dense at the speed of a small dense  
"GPT-4-class" (rumored MoE)| ~1.8T| ~280B (rumored)| Frontier capability at amortized inference cost  
  
The pattern: **frontier models scale parameter count via MoE while keeping per-token active compute manageable**. The challenge is engineering — distributed training, expert parallelism, all-to-all collectives. Quality wins are real; the engineering cost is the price.

## The distributed problem: tokens go to experts on different ranks

For real frontier MoE, you can't fit all N experts on one GPU. Experts get sharded across ranks — typically one or a few experts per GPU. _This is a new axis of parallelism, distinct from data, tensor, and pipeline parallel_.

The forward path through one MoE layer:

  1. Each rank has tokens (its data shard) and a few experts.
  2. The router runs locally on each rank — produces routing decisions for its tokens.
  3. Tokens get sent to the rank holding their assigned experts. **This is an all-to-all collective** (M15).
  4. Each rank's experts process the tokens they received.
  5. Outputs get sent back to the originating ranks (another all-to-all).
  6. Each rank assembles the final output from the K experts' weighted contributions.

Expert parallelism: tokens go to experts on remote ranks via all-to-all Initial: each rank has its tokens and its expert(s) Rank 0 — Expert 0 tokens [a, b, c, d] Rank 1 — Expert 1 tokens [e, f, g, h] Rank 2 — Expert 2 tokens [i, j, k, l] Rank 3 — Expert 3 tokens [m, n, o, p] all-to-all #1: shuffle tokens to expert ranks After all-to-all #1: each rank has the tokens routed to its expert Rank 0 — Expert 0 [a, e, j, n] (chose Expert 0) Rank 1 — Expert 1 [b, f, k, m] Rank 2 — Expert 2 [c, g, i, o] Rank 3 — Expert 3 [d, h, l, p] Each expert processes its tokens locally → all-to-all #2 sends results back to originating ranks → weighted sum all-to-all is the dominant communication cost in MoE two collectives per MoE layer × N layers per step → comm-heavy

The all-to-all collective from M15 finally gets its real-world workout. **Each token's data has to travel from its data-rank to its expert-rank and back, every MoE layer, every step.** For a model with 8 MoE layers across 16 ranks, that's 32 all-to-all collectives per training step. Inter-node bandwidth becomes a real constraint — which is why MoE training stacks pin expert parallelism inside a node when possible.

## Load balancing: the auxiliary loss

The naïve router has a winner-take-all dynamic: if one expert is slightly better at the start, it gets more tokens, which trains it more, which makes it even better. Eventually the router collapses to picking the same 1-2 experts for every token. _You've trained an MoE model that's effectively a dense model with one expert worth of capacity._

The fix is an **auxiliary load-balancing loss**. After the router runs, compute:
    
    
    # For one MoE layer, given routing decisions on all tokens in the batch:
    # f_i = fraction of tokens routed to expert i
    # P_i = average router probability assigned to expert i (over all tokens)
    #
    # aux_loss = N * sum_i(f_i * P_i)
    #
    # Both f_i and P_i are 1/N if perfectly balanced; loss is minimized at balance.
    
    aux_loss = N_experts * (token_fractions * router_probs).sum()
    total_loss = main_loss + 0.01 * aux_loss   # typical coefficient ~0.01-0.1

The combination of `f_i` (counted; differentiable through token assignment is tricky) and `P_i` (the smooth probabilities) gives a smooth gradient that pushes toward balance. **This is mandatory for MoE training** — without it, the model collapses within hundreds of steps.

Other balancing techniques — different schemes are stacked together in production:

  * **Capacity factor** : cap how many tokens each expert can take per batch (typically `1.25 × tokens / N_experts`). Overflow tokens get dropped or routed to a fallback. Hard limit on imbalance.
  * **Expert-choice routing** : invert the role — instead of each token picking K experts, each expert picks the top-T tokens. Guarantees perfect balance but breaks autoregressive generation (an expert can "see" tokens before they're emitted).
  * **Free-form routing with router-Z loss** : penalize the router's logits' L2 magnitude to keep routing decisions soft.

## 5D parallelism: the modern stack

Module 18 introduced 4D parallelism: data + tensor + pipeline + sequence. MoE adds a fifth axis — **expert parallelism (EP)**. The full hierarchy of axes you'll see in a frontier-scale training run:

The full 5D parallelism stack for frontier MoE Axis| What it splits| Bandwidth need| Where  
---|---|---|---  
Data parallel (DP)| Batch across copies| Low (one all-reduce/step)| Outermost — across nodes/clusters  
Pipeline parallel (PP)| Layers into stages| Low-medium (point-to-point)| Across nodes  
Expert parallel (EP)| Experts across ranks| Medium (all-to-all per layer)| Within node when possible  
Tensor parallel (TP)| Matmul within layer| High (all-reduce per matmul)| Within node — NVLink  
Sequence parallel (SP)| Per-token regions of activations| Bundled with TP| Within TP groups  
  
For a 1T-param MoE training run on 1024 GPUs, a typical configuration: TP=8, EP=8, PP=4, DP=4. Total: 8×8×4×4 = 1024. The choice of axis sizes is a topology decision — match each parallelism's communication pattern to the network fabric that can serve it. NVLink for TP and EP (high-bandwidth all-reduces and all-to-alls); InfiniBand for PP and DP (lower-frequency, larger messages).

## Inference of MoE

MoE inference has a different shape from dense inference. The key facts:

  * **Compute scales with K active experts**. For Mixtral 8x7B with K=2: ~13B active params per token. Throughput is similar to a dense 13B model.
  * **Memory scales with N total experts**. The whole model has to be resident — all 47B params for Mixtral 8x7B. You can't unload inactive experts because routing is dynamic.
  * **The all-to-all in serving** : even at inference, MoE adds collective overhead. For single-GPU serving (small MoE), no problem. For multi-GPU serving, the EP all-to-all becomes a real cost.
  * **Routing imbalance at inference** : at training the auxiliary loss kept things balanced; at inference there's no loss. Real input distributions can cause some experts to be hot. Production servers (vLLM, TensorRT-LLM) handle this with capacity overflow buffers and expert replication.

The practical recipe for MoE serving: **quantize all experts (M26), one expert per rank in EP=N configuration, paged KV (M27), continuous batching (M27)**. You can run Mixtral 8x7B comfortably on a single H100 with int4 expert weights; DeepSeek-V3 needs multi-node.

## The capstone problem

Time to put it together. Here's the scenario.

> **★ KEY IDEA**  
>  **Capstone:** You're tasked with designing the training and serving recipe for a new frontier-scale model: 
> 
>   * Architecture: **MoE transformer, 64 experts × 4B params/expert + 12B shared (attention + embedding) = ~270B total params**
>   * Active per token: K=4 experts → ~28B active params
>   * Context length: 128K tokens
>   * Training cluster: 256 H100 GPUs across 32 nodes, NVLink within nodes, 400 Gbps InfiniBand between 
>   * Inference target: serve at ≥40 tok/sec per request, >100 concurrent users on 8 H100s
> 
You need to specify: parallelism configuration, mixed precision setup, attention implementation, optimizer setup, KV cache strategy, quantization recipe, and serving stack. Make every decision and justify it against a specific module's content. 

Work through it. The proposed solution below references every part of the course. Read it after you've drafted your own.

show proposed solution

#### Training stack

  * **5D parallelism** : TP=8 (within node, NVLink for matmul all-reduces — M18), EP=8 (within node, NVLink for expert all-to-alls — M28), PP=4 (across 4 nodes per pipeline copy, point-to-point sends — M18), DP=1 (after TP × EP × PP = 256, no DP axis remaining for 256 GPUs). Total: 8 × 8 × 4 × 1 = 256.
  * **Why those sizes** : TP and EP both want NVLink, fit within an 8-GPU node together. PP across nodes, tolerates higher latency point-to-point. The 4-node pipeline depth gives 4 pipeline stages.
  * **FSDP within the DP axis would help** if we had more GPUs (M17), but with TP=8 and EP=8 each rank already holds only 1/64 of parameters — comparable savings to ZeRO-3 already.
  * **Mixed precision** : bf16 for forward and backward (M14), with fp32 master copy (M12). fp8 (M14, M26) is an option for matmuls on H100, would give ~1.5× more throughput, but adds engineering complexity — defer to a v2.
  * **Attention** : FlashAttention (M25) is mandatory at 128K context. Without it, the (T,T) intermediate is (128K)² × 2 bytes / head × heads = many GB per layer just in activations. Use `F.scaled_dot_product_attention` with FlashAttention backend; pads to TC-friendly head dim 128.
  * **Sequence parallelism** bundled with TP (M18) — splits activation memory along T in non-matmul regions. At 128K seq length this saves a lot of activation memory; nearly required.
  * **Activation checkpointing** (M12): every transformer block. Doubles forward time but cuts activation memory by ~10×. Combined with FSDP + FlashAttention + SP, makes 128K training fit.
  * **Optimizer** : AdamW (M9) with linear warmup → cosine schedule. Maybe Muon (M9) for select layers if recent research holds — would reduce optimizer state and improve loss curves.
  * **Auxiliary load-balancing loss** : critical for MoE (M28). Coefficient ~0.01. Plus capacity factor 1.25 to cap per-expert imbalance.
  * **DataLoader** : DistributedSampler (M10) with persistent_workers, pin_memory, num_workers=8 per rank. For 128K context, pre-pack documents into fixed-size sequences server-side.
  * **Compile** : `torch.compile` (M21) with FSDP2-style integration. MoE's dynamic routing has historically been hard for Dynamo; may need to compile per-expert and the router separately. fullgraph=True initially to find break points.
  * **Profiling** : PyTorch profiler (M13) every few thousand steps. Watch for the four bottleneck shapes — likely GPU-bound on matmul early, then comm-bound (all-to-all latency) as scale grows.

#### Serving stack

  * **Quantize** : AWQ-int4 (M26) on all expert weights. Total weight memory: 270B × 0.5 bytes = 135 GB. Distributed: ~17 GB per H100 with EP=8 — fits. Keep attention weights and embeddings in bf16 (M26 — small relative to expert MLPs, sensitive to precision).
  * **Tensor + Expert parallel** : TP=4, EP=8 (or TP=8, EP=8 for batched inference of larger sizes). 32 GPUs serving the model, but a 8-GPU configuration is fine for most loads.
  * **Paged FlashAttention** (M27): handles 128K-token requests with paged KV cache, no fragmentation. Block size 16 tokens.
  * **Continuous batching** (M27): vLLM or TensorRT-LLM with MoE support. Dynamic batch composition.
  * **Speculative decoding** (M27): the MoE router has variable per-token compute, which complicates draft acceptance. Use a small (1-2B) dense drafter; ~50-65% acceptance rate typical for MoE targets.
  * **Prefix caching** (M27): essential for shared system prompts. ~30-50% prefill savings in chat workloads.
  * **Per-expert capacity buffer** : at inference, hot experts can overflow capacity. Buffer 1.5× expected per-expert tokens, with overflow handled via sequential fallback.

#### Failure modes to anticipate

  * **Training divergence from imbalanced routing** : monitor router entropy + expert utilization variance. Tweak aux loss coefficient or capacity factor.
  * **OOM during 128K context training** : most likely from forgetting sequence parallelism or activation checkpointing. Use memory snapshot tool (M20) to find the peak.
  * **Slow all-to-all** : network topology mismatch between EP and physical layout. Run NCCL_DEBUG=INFO to verify topology routing.
  * **Inference TPS lower than 40** : most likely decode bottleneck. Quantize to int4 (M26) first, then add speculative decoding (M27).
  * **NaN in loss** : walk M14's diagnosis tree. Forward hooks (M5) to find first NaN; reduce LR or switch optimizer; check init scheme (M8).

**What this exercise reveals** : every decision references a specific module's content. Modern frontier ML is not one technique but a coordinated set of optimizations — memory, compute, communication, numerics, kernels, serving — each individually understandable, jointly difficult. _You've now seen the components. The art is composing them._

## Closing notes

You started Module 1 by inspecting strides on a tensor. You're ending here, designing the training and inference stack for a frontier-scale model. The journey covered:

  * **Part I (Tensor Foundations)** : tensors, indexing, dtypes — the data structures everything else builds on.
  * **Part II (Autograd)** : the computational graph that makes gradient descent work.
  * **Part III (Building Models)** : the modules, init schemes, optimizers — the toolkit for actual training.
  * **Part IV (Data & Training)**: the loops, samplers, checkpoints — the production training engineering.
  * **Part V (Performance)** : memory, profiling, mixed precision — how to predict and optimize what your code will do.
  * **Part VI (Distributed)** : collectives, DDP, FSDP, tensor & pipeline parallelism — scaling across hardware.
  * **Part VII (Compilation/Internals)** : dispatcher, CUDA semantics, compile — what's actually happening under PyTorch.
  * **Part VIII (Kernels & Frontiers)**: GPU model, CUDA, Triton, FlashAttention, quantization, serving, MoE — writing and serving the modern frontier.

What you're missing:

  * **Reinforcement learning from human feedback** — the post-training phase that turns base models into chat models. Out of scope here; covered well by other resources.
  * **Specific architectures** — diffusion models, multimodal models, retrieval-augmented systems. Each has its own engineering quirks.
  * **The empirical knowledge** — what hyperparameters work, which init schemes are robust, how to debug specific training failures. This comes from running experiments, not from reading.
  * **The frontier itself** — fp4, mixture-of-depths, world models, longer-than-1M context. The list keeps growing. You now have the framework to read papers and understand them; that's what we built.

What to do next:

  1. **Build something**. Pick a small, real task — a tiny LLM trained from scratch, a custom kernel for an op you care about, a serving system for a model you've fine-tuned. Run into the problems firsthand.
  2. **Read papers actively**. With this background, papers like FlashAttention, Megatron, ZeRO, DeepSpeed, Mixtral, Llama 3, vLLM are now legible — go read them and trace the engineering decisions to their motivations.
  3. **Profile**. Every claim you read in a paper is checkable with the profiler. Build the habit of "is that real?" — reproducing claims grounds your understanding.
  4. **Contribute**. PyTorch, vLLM, Triton, torchao, FlashAttention — all open-source, all welcoming contributors. Your kernel knowledge from M22-M25 is exactly what's wanted.

The point of all of this is not to know everything — nobody does — but to _have the framework to learn anything new in the field_. When the next architectural breakthrough lands, when fp4 ships, when whatever comes after MoE is announced — you'll have the dispatcher, the roofline, the parallelism axes, the tile-based kernel mindset to slot it in.

Thanks for reading. Now go make something.

## Code Magnets: design an MoE forward pass

You're writing the simplified forward pass of an MoE layer. Three magnets are wrong choices.

Arrange the magnets into a working forward.

def forward(self, x): logits = self.router(x) weights, indices = logits.topk(self.k, dim=-1) weights = weights.softmax(dim=-1) weights = weights.sigmoid() out = torch.zeros_like(x) for e_idx in range(len(self.experts)): mask = (indices == e_idx).any(dim=-1) if mask.any(): expert_out = self.experts[e_idx](x[mask]) out[mask] += apply_weight(expert_out, weights, indices, e_idx) aux_loss = self.compute_load_balance_loss(logits, indices) return out, aux_loss return out expert_out = self.experts[indices.argmax()](x)

show solution
    
    
    def forward(self, x):
        logits = self.router(x)
        weights, indices = logits.topk(self.k, dim=-1)
        weights = weights.softmax(dim=-1)
        out = torch.zeros_like(x)
        for e_idx in range(len(self.experts)):
            mask = (indices == e_idx).any(dim=-1)
            if mask.any():
                expert_out = self.experts[e_idx](x[mask])
                out[mask] += apply_weight(expert_out, weights, indices, e_idx)
        aux_loss = self.compute_load_balance_loss(logits, indices)
        return out, aux_loss

The traps:

  * `weights = weights.sigmoid()`: sigmoid doesn't normalize across the K experts. Need softmax over the top-K so weights sum to 1 (preserves output magnitude).
  * `return out`: drops the auxiliary load-balancing loss. Without returning it, the auxiliary loss never gets added to the main training loss → routing collapses.
  * `expert_out = self.experts[indices.argmax()](x)`: picks the single best expert globally and runs the full batch through it — that's "single expert routing," not top-K MoE. Defeats the purpose.

The full pattern: **router → top-K + softmax → per-expert masked compute → weighted sum → return both output AND aux loss**. The aux loss return is what most first-time MoE implementations forget; downstream training breaks because there's no force keeping the router balanced.

## Who does what?

Match each MoE concept to its real role.

Concept

Real role

Router

A. Tiny linear layer + top-K + softmax — picks K experts per token.

Expert

B. One of N parallel MLPs; specializes during training; idle for most tokens.

Top-K (typically K=2)

C. Each token activates only K of N experts — sparse activation pattern.

Auxiliary load-balancing loss

D. Penalizes uneven token-to-expert assignment; mandatory to prevent router collapse.

Expert parallelism (EP)

E. Fifth parallelism axis; experts split across ranks; uses all-to-all collectives.

All-to-all collective

F. The communication primitive that shuffles tokens to expert ranks and back each layer.

Capacity factor

G. Hard cap on tokens per expert per batch — prevents extreme imbalance with overflow handling.

show solution

**Router** → A  
**Expert** → B  
**Top-K** → C  
**Auxiliary load-balancing loss** → D  
**Expert parallelism (EP)** → E  
**All-to-all collective** → F  
**Capacity factor** → G 

The mental shortcut: _router picks, experts compute, top-K is sparse, aux loss balances, EP is the new axis, all-to-all shuffles tokens, capacity factor caps per-expert_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team trains a MoE model and observes that two of the eight experts get ~80% of all tokens after 1000 steps. What's likely going on, and what's the fix?

show answer

The auxiliary load-balancing loss is missing or has too small a coefficient. The winner-take-all dynamic of routing — small initial preferences amplify into total dominance — has taken over. Fix: ensure the aux loss is added to the main loss with a meaningful coefficient (try 0.01-0.1; tune from there). Also verify _both_ the f_i (assignment fractions) and P_i (router probabilities) terms are present — some implementations only do the P term, which is too soft. Verify with a metric: log per-expert load (number of tokens routed to each) at validation; histogram should be roughly uniform after a few hundred steps once aux loss is correctly applied.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why is the all-to-all collective a "natural fit" for MoE — what makes it the right primitive vs all-reduce or all-gather?

show answer

All-reduce is "everyone has data; sum and replicate to everyone." All-gather is "everyone has a slice; everyone gets the concatenation." Neither matches MoE's pattern.

MoE needs: "rank 0 has tokens routed to expert 1 → send them to rank 1; rank 0 also has tokens for expert 5 → send those to rank 5; and rank 0 receives from rank 2's tokens routed to rank 0's expert." Every rank sends a different subset of its data to every other rank, simultaneously. _That's literally the all-to-all primitive's definition_ — each rank has a list of buffers (one per destination rank) and exchanges them with all destinations in one collective.

An all-reduce or all-gather would either send too much (every rank's data to everyone) or not match the destination structure. MoE was waiting for all-to-all all along; M15 was the foundation, M28 is the deployment.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** A team's MoE model trains well but inference is unexpectedly slow — about the same speed as a dense model with the same total parameters, not the same as a dense model with the active-per-token parameters. What's likely happening?

show answer

The all-to-all overhead is dominating. In a small-batch inference setting, the all-to-all collective per MoE layer × number of layers can add substantial latency, especially across nodes. With low batch sizes, the matmul work per expert is small (each expert sees only a few tokens), so the constant communication overhead becomes a large fraction of the total time.

Diagnostics: profile (M13). Check if the trace is "comm-bound" — NCCL bars alternating with compute (M13's communication-bound shape). Fixes: (a) batch more requests together via continuous batching (M27) — more tokens per batch means more useful work per all-to-all. (b) Replicate experts across ranks instead of strict EP — eliminate the all-to-all but use more memory. (c) Consider whether your hardware topology suits your EP factor — cross-node EP is much slower than within-node.

This is why production MoE inference (vLLM, TensorRT-LLM) often uses small EP within a node + DP across nodes — keeps the all-to-all on the fastest fabric.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** If MoE gives "model quality of N×K active params at compute of K active params," why isn't every model MoE?

show answer

Several real costs that MoE adds:

  1. **Memory** : the entire model has to fit on your hardware even though only K/N is active per token. Mixtral 8x7B needs 47B params worth of memory at inference time, not 13B. For memory-constrained settings (mobile, single-GPU consumer), this is prohibitive.
  2. **Engineering complexity** : expert parallelism adds a new parallelism axis; routing requires careful load-balancing; all-to-all collectives add comm overhead; quantizing routes to balance experts is tricky. Frontier teams handle this; smaller teams find it daunting.
  3. **Specialized inference systems** : not all serving frameworks support MoE well. Some do; some don't. Throws constraints on deployment.
  4. **Quality trade-offs** : MoE gains per-active-FLOP are real, but per-total-param quality is sometimes worse than dense (you're effectively training fewer interactions per parameter). Distillation from MoE to dense for deployment is common.

For frontier-scale capability targets where compute, not memory, is the binding constraint, MoE wins. For memory-constrained or simple-deployment settings, dense models still dominate. _The optimal choice depends on what's binding_.

### What just happened?

  * **Mixture of Experts** : replace one MLP per layer with N parallel MLPs ("experts") and a small router that picks top-K (typically K=2) per token. Compute scales with K; parameters with N.
  * **Sparse activation pattern** : each token only activates K/N of the experts. 8× the parameters at the same per-token compute as a dense model.
  * **The router** : a tiny linear layer (D → N) followed by top-K + softmax. ~0.05% of the layer's parameters. Outputs are routing decisions and weights.
  * **Load-balancing auxiliary loss** is mandatory: `aux_loss = N × sum(f_i × P_i)`. Without it, the router collapses to picking 1-2 experts winner-take-all.
  * **Capacity factor** : hard cap (~1.25 × tokens / N) on per-expert load per batch. Overflow handled with fallback or drop.
  * **Expert parallelism (EP)** : a fifth parallelism axis. Experts split across ranks; tokens get routed to their expert rank via **all-to-all** (M15 finally deployed for real).
  * **5D parallelism** : DP × PP × EP × TP × SP. Match each axis to its right network fabric (TP/EP within node on NVLink; PP/DP across nodes on InfiniBand).
  * **Frontier MoE** : Mixtral 8x7B (47B total / 13B active), DeepSeek-V3 (671B total / 37B active), GPT-4-class (rumored ~1.8T total / ~280B active).
  * **Inference of MoE** : compute scales with active K, but memory scales with total N — entire model must be resident. all-to-all overhead can dominate at low batches; EP within node + DP across nodes is the standard config.
  * Modern frontier-scale training is the orchestrated combination of _everything_ — mixed precision, FSDP, FlashAttention, kernels, quantization, paged serving, MoE — each layer adding 1.3-3× efficiency. The capstone showed this composition.

> **★ KEY IDEA**  
>  **End of the core course.** Twenty-eight modules. From `x.stride()` to `fully_shard` to `tl.dot` to MoE expert parallelism. You now have the full mental model — the framework to read any modern paper, debug any training failure, and design any inference system. **The art is in the composition. The science is in the components. You've seen both.** A short Part IX follows for synthesis modules: building a transformer end-to-end (M29), RoPE (M30), post-training (M31), debugging (M32), and Mamba (M33).
