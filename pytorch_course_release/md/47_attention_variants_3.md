# Module 47 — Attention Variants III: Kernels, Implementation & Compositional Architectures

# _Attention Variants III:_ kernels, implementation & compositional architectures

_Attention Trilogy · Module 47 · Final module · April 2026 currency_

— Axes D and E of the four-axis taxonomy: FlashAttention 1-4 with the FA4 Blackwell pipeline, PagedAttention & Ring Attention & Star Attention for sequence parallelism, MoDA's depth attention (March 2026), and the hybrid composition patterns that close the course

\--- 

This is the final module of the trilogy and of the course. M45 covered Axis A (semantic role) and Axis B (receptive field). M46 covered Axis C (head topology and KV-cache engineering). This module closes the taxonomy with **Axis D — kernel implementation** and **Axis E — compositional architectures**.

Axis D is where attention _actually runs_ on hardware. M22 introduced FlashAttention as a tiled IO-aware reformulation; the lineage from FA1 (2022) through FA4 (March 2026) tracks GPU architecture evolution — Ampere → Hopper → Blackwell — and how the kernel structure had to change at each step. PagedAttention, Ring Attention, Striped Attention, Star Attention, TokenRing, DeepSpeed Ulysses, USP — all are kernel-level innovations that compose with the head-topology choices from M46.

Axis E is where attention _composes with the rest of the architecture_. The 2026 frontier reality (M44, M45, M46): no production model uses one attention mechanism uniformly. **Hybrid stacks dominate** — different layer types interleaved according to specific patterns. Qwen3-Next 3:1, Kimi Linear, Ling 2.5, Nemotron 3 Nano, DeepSeek V4 CSA+HCA, Llama 4 iRoPE, MoDA's depth attention (March 2026). Each represents a different composition strategy.

By the end you'll know why FA4's warp specialization on Blackwell is fundamentally different from FA3 on Hopper, how PagedAttention eliminated the 60-80% KV cache fragmentation that plagued early vLLM, how Ring Attention enables multi-million-token contexts via sequence parallelism, why MoDA found that "depth attention" was the missing primitive for deep-stack scaling, and how to read any 2026 frontier architecture as a composition of choices on the five axes you've now learned.

> **★ KEY IDEA**  
>  **Axis D — kernel implementation** : how attention runs on GPUs. **FlashAttention lineage** : **FA1 (2022)** : tiled exact attention; never materializes the N×N matrix; works on Ampere. **FA2 (2023)** : better work splitting across thread blocks; reduces non-matmul FLOPs. **FA3 (2024)** : warp specialization (producer/consumer warps) for Hopper async; FP8 support. **FA4 (March 2026)** : full Blackwell pipeline co-design via CuTeDSL; warp-specialized async MMA + softmax overlap; software-emulated exponentials via FMA polynomial approximation; conditional softmax rescaling (10× fewer rescaling ops); 2-CTA MMA up to 256×256×16; ~20% over cuDNN FA3. **Sequence parallelism family** : **PagedAttention** (vLLM, 2023): block-based KV cache layout eliminates 60-80% fragmentation, 24× throughput vs HuggingFace; **Ring Attention** (Liu et al., 2023): split sequence across devices, attention via ring communication, enables multi-million-token contexts; **Striped Attention** (MIT, 2023): causal-balanced ring for load balancing in causal attention; **Star Attention** (NVIDIA, 2024): block-sparse + global anchor, 11× speedup; **TokenRing** (SJTU, 2025): bidirectional ring with point-to-point comm; **DeepSpeed Ulysses** : extreme long-sequence; **USP — Unified Sequence Parallel** (Tencent): hybrid Ring + Ulysses. **Axis E — compositional architectures** : how attention layers compose. **Hybrid layer-type stacks** : **Qwen3-Next/Qwen3.5** : 3:1 Gated DeltaNet (linear) : Gated Attention (softmax). **Kimi Linear** : Kimi Delta Attention (linear) + MLA (softmax). **Ling 2.5** (1T MoE): Lightning Attention + MLA. **Nemotron 3 Nano** : Mamba-2 + sparse attention hybrid. **DeepSeek V4** : CSA (4× compress + Lightning Indexer) + HCA (128× compress dense) interleaved. **Llama 4** : iRoPE (3 RoPE layers + 1 NoPE layer interleaved, enables 10M context). **Architectural primitives that compose with attention** : **MoDA — Mixture-of-Depths Attention** (Zhu et al., March 2026, ByteDance Seed + HUST): each head attends to current-layer sequence KV AND depth KV from all preceding layers; mitigates information dilution in deep stacks; 97.3% of FA2 efficiency at 64K via chunk-visible kernel; +0.2 perplexity, +2.11% downstream, only 3.7% FLOPs overhead. **KArAt — Kolmogorov-Arnold Attention** : replaces softmax with learnable basis functions; vision-focused. **Multi-Token Attention (MTA)** : group normalization with depth-aware scaling. **Differential Transformer** : difference of two softmaxes for sharper attention patterns. **Talking-Heads Attention** : head mixing at the logit/weight level (incompatible with FlashAttention; superseded by IHA). **The decision framework** : given a workload + hardware + context length, identify the right composition: (1) primary attention mechanism (full softmax / hybrid linear-softmax / sparse); (2) head topology (GQA / MLA / IHA — M46); (3) position encoding (RoPE / iRoPE / YaRN); (4) kernel implementation (FA3 / FA4 / Ring / PagedAttention); (5) cache compression (TurboQuant / eviction / offloading — M46). All five compose. Production 2026 frontier models pick a point in the resulting 5D space; reading any model is identifying its coordinates. 

## Two new faces — closing the trilogy

K

Kernel

"I'm where attention actually runs. Algorithms become hardware. The math from M22 onward is theoretical until it lands on tensor cores."

Attention is just `softmax(QK^T)V` mathematically — it's the implementation that determines whether your model trains in 12 hours or 12 days. **I'm the difference between cuDNN and Triton, FA3 and FA4, naive kernel and ~600 TFLOPs/s on B200**. Each GPU generation shifted my priorities: Ampere needed tiled IO-aware (FA1 era); Hopper added async TMA + warp-group MMA (FA3); Blackwell broke the pipeline open with warp specialization, TMEM, and asymmetric scaling — Tensor Cores got 2.25× faster but SFU exponentials didn't, so FA4 had to redesign the pipeline so softmax overlaps MMA. _Reading FA4 source is reading what GPU kernel engineering looks like in 2026_ : CuTeDSL Python that compiles to PTX, software-emulated exponentials via FMA polynomial approximation, 2-CTA MMA spanning paired thread blocks. I'm the practical bridge between abstract attention and the silicon that runs it.

∁

Composer

"I arrange attention layers into stacks. Three linear, one softmax. Compress + sparse + sliding. Sequence + depth. The mix is the model."

No 2026 frontier model uses uniform attention. **Every flagship is a composition** — Qwen3-Next is 3:1 GDN to softmax; Kimi Linear is KDA with MLA; Ling 2.5 is Lightning + MLA; Nemotron 3 Nano is Mamba + sparse; DeepSeek V4 alternates CSA and HCA; Llama 4 alternates RoPE and NoPE; MoDA adds depth attention to standard sequence attention. _I'm what determines model character at the architectural level_. The choice of "75% linear + 25% softmax" vs "alternating sparse + dense" vs "depth-augmented sequence" produces qualitatively different models even with identical training data and parameter count. Reading a 2026 model card means identifying what I do — which layer types, which ratios, which interleaving patterns. The five-axis taxonomy you've learned across M45-M47 lets you parse any composition I produce.

## Axis D part one: FlashAttention lineage

M22 introduced FlashAttention's foundational insight: standard attention is memory-bound, not compute-bound, on modern GPUs. The N×N attention matrix is read and written multiple times to HBM (slow); the actual matmul work is fast on tensor cores. FA's solution: tile the computation so each tile fits in SRAM (fast), use online softmax to avoid materializing the full attention matrix, and never write the N×N matrix to HBM.

That algorithmic insight has remained constant; what changed across FA1 → FA4 is **how the algorithm is mapped onto each new GPU architecture**.

### FA1 (2022) — Ampere/Hopper basics

The original. Tiled forward and backward passes; online softmax with running max and running sum; recomputation in the backward to avoid storing intermediate attention matrices. Targeted Ampere (A100) primarily. Achieved ~150 TFLOPs/s on A100 at sequence length 4K — substantial speedup over PyTorch's stock attention but well below the theoretical peak.

### FA2 (2023) — better parallelism

The bottleneck FA1 missed: thread block parallelism. FA1 parallelized over batch and heads; FA2 added parallelism over sequence length blocks. Plus reduced non-matmul FLOPs (the rescaling ops in online softmax) and improved register usage. Result: ~230 TFLOPs/s on A100; closer to peak. FA2 became the production default through 2023-2024.

### FA3 (2024) — Hopper async

The first FA generation specifically co-designed for a GPU architecture (Hopper / H100). Hopper introduced **TMA (Tensor Memory Accelerator)** for async global memory copies and **WGMMA (Warp-Group MMA)** for async tensor-core ops. FA3 introduced **warp specialization** :

  * **Producer warps** : exclusively issue TMA loads from HBM into a circular shared-memory buffer. Asynchronous; one warp keeps the memory pipeline saturated.
  * **Consumer warps** : exclusively execute WGMMA matmuls and softmax computation. They never issue memory loads.

This is hardware-level parallelism — while consumers compute on stage _i_ , producers load stage _i+1_. Plus FA3 added FP8 support and dynamic register reallocation (the `setmaxnreg` instruction). Achieved ~750 TFLOPs/s on H100 in FP8 — major leap.

### FA4 (March 2026) — Blackwell co-design

Published March 5, 2026. The most significant architectural change since FA1 itself.

The motivation: **Blackwell's asymmetric hardware scaling**. From H100 to B200, BF16 Tensor Core throughput jumped 2.25× (1 → 2.25 PFLOPs); but SFU throughput (used for exponentials in softmax) and shared memory bandwidth stayed approximately constant. _The bottleneck shifted_ : Tensor Cores became so fast that softmax computation became the limit. FA3's pipeline (async loads + sync softmax) was no longer enough.

FA4's four core innovations:

FA4 — Blackwell warp-specialized pipeline (March 2026) ① Redesigned async pipeline (warp specialization) Tensor cores got 2.25× faster on Blackwell, but SFUs/SMEM bandwidth did not. Bottleneck shifts to softmax/transcendentals. Solution: Orchestration warps manage async TMA loads + tcgen05.mma scheduling; Compute warps handle softmax in parallel. Result: tile _i_ 's MMA overlaps with tile _i-1_ 's softmax — both tensor cores AND SFUs continuously occupied. ② Software-emulated exponentials The exp() in softmax normally routes through MUFU.EX2 (special function unit; limited throughput). FA4: implement exp() via polynomial approximation on FMA units (general-purpose float compute). Result: exp moves to abundant FMA hardware instead of bottlenecked SFUs. Removes the softmax throughput cliff. ③ Conditional softmax rescaling Online softmax normally rescales intermediate results EVERY time the running max changes. FA4: skip rescaling unless the shift is large enough to threaten numerical stability. ~10× fewer rescaling ops (Tri Dao, Hot Chips 2025). ④ 2-CTA MMA — paired CTAs share one large MMA Blackwell SM100: paired CTAs in same cluster jointly execute one UMMA spanning both TMEMs. Tile size up to 256×256×16; reduces redundant data transfer per CTA. Implemented via tcgen05.mma. Hardware constraint: requires SM100 (B200/GB200/B300). _Does NOT run on RTX 5090 (SM120)_ — no UTC*/UTMA* opcodes.

Implementation: FA4 is fully written in **CuTeDSL** , a Python-based kernel DSL provided by NVIDIA's CUTLASS team. The Python source compiles directly to PTX/SASS through CuTeDSL's compiler. This is a substantial change — earlier FA versions used C++ + CUDA + CUTLASS templates. CuTeDSL is more concise and easier to modify; the tradeoff is requiring CUTLASS DSL infrastructure.

**Performance results** :

  * ~600+ TFLOPs/s on B200 in FP8 — roughly **20% over cuDNN FA3** on Hopper-equivalent precision.
  * 1.2-3.2× over Triton-based FlexAttention on compute-bound workloads (PyTorch announcement, March 2026).
  * Default for MLA prefill in vLLM (late 2025-2026 deployment).

**Hardware availability** : FA4 requires SM100 (Blackwell B200, GB200, GB300) or specifically-supported SM120 paths. _RTX 5090 (consumer Blackwell desktop, SM120) cannot run FA4_ — the GB202 die lacks the UTMA*/UTC* opcodes that FA4's pipeline depends on. This is a meaningful production constraint: FA4 is for datacenter Blackwell, not consumer.

FA4 forward-pass schematic in pseudocode:
    
    
    def flash_attention_4_forward(Q, K, V, block_size_M=128, block_size_N=128):
        """FA4 forward — Blackwell-specific warp-specialized pipeline (schematic)."""
        # Allocate output and softmax statistics
        O = torch.zeros_like(Q)
        L = torch.zeros(Q.shape[:-1], dtype=torch.float32)   # log-sum-exp
    
        for i in range(0, N, block_size_M):
            # ORCHESTRATION WARPS: async load Q_i, K_j, V_j blocks via TMA into TMEM
            Q_i = tma_load_async(Q[i:i+block_size_M])
    
            m_i = -float("inf")   # running max
            l_i = 0.0             # running sum (log-sum-exp partial)
            O_i = torch.zeros(block_size_M, d_v)
    
            for j in range(0, N, block_size_N):
                K_j = tma_load_async(K[j:j+block_size_N])
                V_j = tma_load_async(V[j:j+block_size_N])
    
                # 2-CTA MMA: paired CTAs jointly compute Q_i @ K_j^T (256x256 tile)
                S_ij = tcgen05_mma_2cta(Q_i, K_j.T)   # [block_M, block_N]
    
                # COMPUTE WARPS execute softmax IN PARALLEL with next MMA's load
                m_new = max(m_i, S_ij.max())
    
                # CONDITIONAL RESCALING: skip if max didn't shift much
                if abs(m_new - m_i) > numerical_threshold():
                    # FMA-polynomial exp instead of MUFU.EX2 SFU instruction
                    rescale = poly_exp(m_i - m_new)
                    O_i = O_i * rescale
                    l_i = l_i * rescale
    
                # Local exp via FMA polynomial approximation
                P_ij = poly_exp(S_ij - m_new)
                l_i += P_ij.sum(dim=-1)
    
                # Update output: another 2-CTA MMA — overlaps with next iteration's load
                O_i += tcgen05_mma_2cta(P_ij, V_j)
    
                m_i = m_new
    
            O[i:i+block_size_M] = O_i / l_i.unsqueeze(-1)
            L[i:i+block_size_M] = m_i + torch.log(l_i)
    
        return O, L

The lesson from the FA1 → FA4 lineage: **each generation's bottleneck moved as the GPU evolved**. FA1 was memory-bound (HBM bandwidth); FA2 was parallelism-limited; FA3 needed async to hide latency; FA4 had to engineer around Blackwell's asymmetric tensor-core-vs-SFU scaling. _Future FA5 will respond to whatever the next bottleneck is_ — likely involving TMEM bandwidth or interconnect (NVLink) at multi-GPU scales.

## Axis D part two: Sequence parallelism family

FlashAttention handles attention _within_ a single GPU. At multi-million-token contexts, the sequence itself is too long for any single GPU's memory. Sequence parallelism splits the sequence across devices.

### PagedAttention — eliminating KV cache fragmentation

vLLM (Kwon et al. 2023). Not strictly "sequence parallelism" but the foundation that enables all the rest.

The problem PagedAttention solved: traditional KV cache implementations allocated contiguous memory per request, sized for the maximum possible context length. Most requests were shorter; the unused memory was wasted. **60-80% fragmentation** was typical pre-PagedAttention.

The solution: **block-based allocation, OS-style virtual memory**. KV cache split into fixed-size blocks (typically 16 or 32 tokens). Each request gets allocated blocks dynamically as it grows. A block table maps logical positions to physical blocks. Result: fragmentation drops from 60-80% to under 4%.

Production impact: **~24× throughput improvement vs HuggingFace stock generation**. Eliminated the largest single bottleneck in long-context serving.

### Ring Attention — sequence-parallel attention across devices

Liu et al. (2023). When the sequence is too long for one device, split it into chunks across devices and have them rotate K, V around a ring during attention computation.
    
    
    def ring_attention_forward(Q_local, K_local, V_local, num_devices):
        """Ring Attention — sequence-parallel attention across devices."""
        # Each device holds 1/num_devices of the sequence (Q, K, V)
        O_local = torch.zeros_like(Q_local)
        K_recv, V_recv = K_local, V_local
    
        for step in range(num_devices):
            # Compute partial attention with current K, V chunk
            partial_O = flash_attention(Q_local, K_recv, V_recv)
            O_local = accumulate(O_local, partial_O)   # online softmax merge
    
            # Send K, V to next device in ring; receive from previous
            K_recv, V_recv = ring_send_recv(K_recv, V_recv)
    
        return O_local

Each device computes partial attention with each rotating K, V chunk; the partial outputs are merged via online softmax. Total compute is the same as full attention (each device sees all K, V eventually); the win is that _no single device needs to store the full sequence_.

Communication cost: O(N × d) per ring step × num_devices steps = O(N × d × num_devices) total — but overlapped with computation. With careful tiling, communication is fully hidden.

Production use: enables 1M-10M token contexts across multi-GPU systems. The basis for most long-context training in 2024-2026.

### Striped Attention — causal-balanced ring

MIT (2023). Ring Attention has a load imbalance problem with **causal attention** : in a causal mask, later sequence positions need to attend to MORE previous positions than earlier ones. Naive ring assigns later positions to specific devices, which means those devices do more work.

Striped Attention's fix: **interleave sequence positions across devices in a striped pattern** rather than chunked. Now each device holds a roughly balanced number of "early" and "late" positions; load is balanced across the ring.

Result: ~2× speedup over naive Ring Attention for causal training at long context. Used in some long-context training pipelines.

### Star Attention — block-sparse with global anchor

NVIDIA (2024). Treats inference differently from training. For inference: split context into blocks; each block attends locally + to a small global "anchor" set. Anchor tokens are typically the first few tokens of the sequence (system prompt, query) plus selected attention sinks.

Result: **~11× speedup for long-context inference** at quality matching full attention. Especially effective for retrieval-style tasks where you have a long document and a short query.

Production use: inference frameworks for very-long-context retrieval and Q&A.;

### TokenRing — bidirectional point-to-point

SJTU (2025). Refinement of Ring Attention. Standard Ring Attention uses unidirectional ring communication (each device sends to next, receives from previous). TokenRing uses bidirectional point-to-point — each device sends partial results to other devices directly based on need.

Result: better utilization of NVLink bandwidth on systems with rich interconnect (B200 NVL72). Up to 1.5× speedup over unidirectional Ring Attention at 1M+ contexts.

### DeepSpeed Ulysses — extreme long-sequence parallelism

Microsoft DeepSpeed. Different parallelism strategy than Ring: instead of rotating K, V around devices, **partition heads across devices** during attention. The all-to-all communication pattern means each device computes a subset of heads on the full sequence.

Trade-off vs Ring: Ring scales better with sequence length (linear comm); Ulysses scales better with number of devices. The two are complementary.

### USP — Unified Sequence Parallel

Tencent (2024). Combines Ring Attention and Ulysses in a hybrid pattern: Ulysses-parallel within node (using fast NVLink), Ring-parallel across nodes (using slower InfiniBand). Captures benefits of both.

Production use: large-scale training infrastructure where sequence length AND number of devices both matter.

## Axis E part one: Hybrid layer-type compositions

The Axis E reality from M44, M45, M46: **no 2026 frontier model uses uniform attention**. Each is a specific composition of layer types.

Hybrid composition patterns in 2026 frontier models Qwen3-Next / Qwen3.5 — 3:1 GDN to softmax Layers 0-2: Gated DeltaNet (linear); Layer 3: Gated Attention (softmax); Layers 4-6: GDN; Layer 7: Gated Attention; ... GDN GDN GDN Attn GDN ... 75% linear (cheap), 25% softmax (exact recall) Kimi Linear — KDA (channel-wise) + MLA Linear-attention layers use Kimi Delta Attention; Full-attention layers use MLA (M46) for KV compression. KDA KDA KDA MLA KDA ... Compounds linear efficiency with MLA's compressed KV cache Ling 2.5 (1T MoE) — Lightning + MLA Linear: Lightning Attention; Full: MLA. Same hybrid pattern as Kimi Linear with different linear variant. Trillion-parameter open-weight; demonstrates hybrid scales to 1T+ Nemotron 3 Nano — Mamba-2 + sparse softmax Most layers: Mamba-2 SSM blocks (M33, M45); Periodic layers: sparse self-attention. NVIDIA's bet: SSM as primary, attention as exception DeepSeek V4 — CSA + HCA alternating CSA: 4× compress + Lightning Indexer FP4 + sparse; HCA: 128× compress + dense; alternating. 27% FLOPs / 10% KV vs V3.2 at 1M context (M44) Llama 4 — iRoPE (3 RoPE + 1 NoPE per 4) RoPE layers: chunked attention with rotary; NoPE layers: full causal attention, no positional encoding. Enables 10M context generalization from 256K training (M44) The general pattern: each model picks 2-3 layer types and interleaves them in a specific ratio "Composition is the architecture" — uniform stacks are an artifact of 2017-2024 transformers

Three observations from the hybrid landscape:

  * **The 75:25 ratio has emerged as approximately optimal**. Qwen3-Next (3:1 GDN to attention), Llama 4 (3 RoPE to 1 NoPE), Mamba-Transformer hybrids — all gravitate toward roughly this ratio. Why: linear/cheap layers carry most context aggregation; expensive layers provide exact recall + precision when needed.
  * **The "expensive" layer type increasingly is itself compressed**. Kimi Linear and Ling 2.5 use MLA in their full-attention layers, not vanilla MHA. The full-attention layer is "expensive" only relative to linear attention; it's still cache-compressed via MLA.
  * **The compositional approach is the dominant 2026 pattern**. Pure-architecture stacks (Llama 3 = all GQA-MHA; pure Mamba) remain in production but represent simplified or specialized cases. The frontier is hybrid.

## Axis E part two: New compositional primitives

### MoDA — Mixture-of-Depths Attention (March 2026)

Zhu et al., ByteDance Seed + Huazhong University. The most interesting Axis E entry of 2026.

The problem MoDA addresses: **signal degradation in deep stacks**. As LLMs scale to 80+ layers, informative features formed in shallow layers get diluted by repeated residual updates — the "information dilution" problem. By the time information reaches deeper layers, important signals are obscured.

MoDA's mechanism: **each attention head attends to BOTH (a) sequence KV pairs at the current layer AND (b) "depth KV pairs" from preceding layers at the same token position**. The query jointly attends to sequence and depth information under a single softmax normalization.
    
    
    def moda_forward(x, layer_idx, depth_KV_history, n_heads):
        """MoDA forward — query attends to current sequence KV + depth KV from preceding layers.
        depth_KV_history: list of (K_l, V_l) from layers 0 .. layer_idx-1 at SAME token positions."""
        Q = x @ W_Q
        K_seq = x @ W_K   # Standard sequence keys at current layer
        V_seq = x @ W_V
    
        # Concatenate sequence KV with depth KV from all preceding layers
        K_all = [K_seq]
        V_all = [V_seq]
        for K_l, V_l in depth_KV_history:
            K_all.append(K_l)   # Same token positions, different layers
            V_all.append(V_l)
    
        # Stack: K_combined has shape [N, (1 + L_prev) * d]
        K_combined = torch.cat(K_all, dim=-2)
        V_combined = torch.cat(V_all, dim=-2)
    
        # Single unified attention with single softmax over sequence+depth
        scores = Q @ K_combined.transpose(-2, -1) / math.sqrt(d_head)
        output = F.softmax(scores, dim=-1) @ V_combined
        return output

Architecturally, MoDA sits between two extremes:

  * **Depth residual** (standard transformers): each layer adds to a single residual stream; depth information is mixed but undifferentiated.
  * **Depth dense** : every layer attends to every preceding layer densely; full cross-layer attention but expensive.
  * **MoDA (depth attention)** : data-dependent depth retrieval via attention scores, so the model picks which historical layers to attend to. Efficient AND expressive.

**The hardware-efficient algorithm** : naive MoDA has non-contiguous memory access patterns (depth KV from different layers at same token positions vs sequence KV at current layer). The MoDA paper introduces a **chunk-aware layout + group-aware indexing** kernel that resolves this. Result: **97.3% of FlashAttention-2's efficiency at 64K sequence length**.

**Empirical results** (1.5B parameters):

  * +0.2 average perplexity improvement across 10 validation benchmarks vs OLMo2 baseline.
  * +2.11% average performance on 10 downstream tasks.
  * 3.7% additional FLOPs only — _highly favorable cost/benefit_.
  * Works best with post-norm rather than pre-norm layer normalization.
  * Reduces "attention sink" behavior (the empirical phenomenon where models put excess attention on early tokens).

**Production status** : April 2026, MoDA paper is published with code on GitHub (hustvl/MoDA). Triton kernel and chunk-visible variants released. No frontier production model has adopted it yet (paper is one month old) but the favorable cost/benefit ratio + FlashAttention-compatible kernel suggest it could be incorporated quickly.

### KArAt — Kolmogorov-Arnold Attention

Replaces softmax with learnable basis-function compositions inspired by the Kolmogorov-Arnold representation theorem. Vision-focused. Each attention "score function" is a learnable composition of univariate functions.

Production status: research-stage. Vision benchmarks show modest improvements; language model adoption hasn't happened yet. Promising direction if learnable score functions prove generalizable.

### Multi-Token Attention (MTA)

Allows queries to attend to multiple tokens jointly via group normalization with depth-aware scaling. Still being explored; partial overlap with MoDA's depth attention.

### Differential Transformer

Computes attention as the **difference of two softmaxes**. The intuition: standard softmax produces overly diffuse attention; subtracting two softmaxes with different temperatures sharpens the focus. Empirical: better noise rejection on long contexts.

Production status: published research, not yet in frontier production. The mechanism could combine with MLA / GQA but kernel support lags.

### Talking-Heads Attention

Shazeer (2020). Mixes attention information at the level of attention logits/weights — between query-key dot products and the softmax. Mathematically more expressive than MHA but **incompatible with FlashAttention** because the mixing happens inside the attention operator. Largely **superseded by IHA (M46)** which achieves similar expressivity gains with FlashAttention compatibility.

## The five-axis decision framework

Putting it all together. Given a workload, hardware, and context-length target, a 2026 production architecture is identified by choices on five axes:

The five-axis decision framework — pick a coordinate on each Axis| Question| Choices| Module  
---|---|---|---  
**A — Semantic role / normalization**|  Self vs cross? Softmax or linear?| self / cross; softmax / linear (Mamba-3, GDN, KDA, Lightning)| M45  
**B — Receptive field**|  Where does each query attend?| full / sliding window / sparse trainable (NSA, MoBA, DSA, XAttention)| M45  
**C — Head topology**|  How are Q, K, V structured across heads?| MHA / MQA / GQA / MLA / IHA / Slim Attention| M46  
**D — Kernel implementation**|  How does it run on hardware?| FA1-4 / PagedAttention / Ring / Striped / Star / Ulysses / USP| M47 (this)  
**E — Compositional architecture**|  How do attention layers compose?| Uniform / hybrid (3:1, alternating) / depth-augmented (MoDA) / interleaved (iRoPE, CSA+HCA)| M47 (this)  
  
Worked examples:

Reading 2026 frontier models as five-axis coordinates Model| A| B| C| D| E  
---|---|---|---|---|---  
**Llama 4 Scout**|  Self causal softmax| Full + chunked iRoPE| GQA| FA3/FA4 + PagedAttention| iRoPE 3:1 RoPE/NoPE  
**DeepSeek V4-Pro**|  Self causal softmax| Hybrid CSA+HCA| MLA| FA4 + DeepGEMM| CSA+HCA alternating  
**GLM-5.1**|  Self causal softmax| DSA sparse| MLA-style| FA4 + custom| DSA throughout  
**Qwen3-Next**|  Self causal hybrid linear/softmax| Full in attn layers| GQA + GDN linear layers| FA3 + custom GDN kernel| 3:1 GDN to softmax  
**Kimi Linear**|  Self causal hybrid| Full| MLA + KDA linear layers| FA4 + custom| KDA + MLA hybrid  
**Ling 2.5**|  Self causal hybrid| Full| MLA + Lightning linear| FA4 + custom| Lightning + MLA  
**Nemotron 3 Nano**|  Hybrid SSM/softmax| Sparse| Mixed (Mamba blocks + attention)| Custom| Mamba-Transformer hybrid  
**Mistral Large**|  Self causal softmax| Sliding window| GQA| FA3 + PagedAttention| Uniform sliding  
**MiniMax M2.5**|  Self causal softmax| Full| GQA| FA3 + PagedAttention| Uniform classic  
**(Future) MoDA-based**|  Self causal softmax| Full| GQA / MLA| MoDA Triton kernel| Sequence + depth attention  
  
Reading these coordinates is the architectural-literacy skill the trilogy was built to develop. Any 2026 model card can be parsed this way; new entries (Llama 5, DeepSeek V5, GPT-6) will fit on these axes, possibly extending one or more.

## Production deployment recipes

Concrete recipes for common 2026 deployment scenarios:

### Scenario 1 — Frontier MoE training, 1T+ parameters, 1M context

  1. **Axis A** : hybrid linear-softmax (3:1 ratio); Lightning Attention or KDA for linear layers.
  2. **Axis B** : full attention in softmax layers; CSA+HCA hybrid sparse for ultra-long-context efficiency.
  3. **Axis C** : MLA in softmax layers; >100B threshold met (M46).
  4. **Axis D** : FA4 (Blackwell B200/GB200/B300); Ring Attention + DeepSpeed Ulysses for sequence parallelism.
  5. **Axis E** : alternating sparse + dense pattern; possible MoDA depth attention if validation shows benefit.

Reference: DeepSeek V4-Pro, Ling 2.5.

### Scenario 2 — Open-weight 30-100B, 256K context

  1. **Axis A** : hybrid linear-softmax (3:1); GDN linear layers.
  2. **Axis B** : full attention in softmax layers (no sparse needed at this scale).
  3. **Axis C** : GQA (under 100B threshold).
  4. **Axis D** : FA3 (Hopper) or FA4 (Blackwell); PagedAttention; TurboQuant 3-bit KV cache for serving.
  5. **Axis E** : 3:1 hybrid; uniform within each layer type.

Reference: Qwen3-Next/3.5/3.6.

### Scenario 3 — 7B for edge/single-GPU deployment

  1. **Axis A** : pure softmax (simpler engineering; quality lost is small at this scale).
  2. **Axis B** : full attention or sliding window if context > 32K.
  3. **Axis C** : GQA-8.
  4. **Axis D** : FA3 + PagedAttention; TurboQuant 3-bit if memory-constrained.
  5. **Axis E** : uniform.

Reference: Llama 3 8B-equivalent deployments.

### Scenario 4 — Long-context inference on existing pretrained model

  1. Head topology fixed by the pretrained model.
  2. **Apply XAttention** (M45) for plug-and-play sparse acceleration.
  3. **Apply TurboQuant 3-bit** KV quantization (M46).
  4. **Apply Expected Attention or H2O** KV eviction if context > 128K.
  5. **InfiniGen offloading** if GPU memory still insufficient.

The pattern: _can't change architecture → stack inference-time compressions_.

#### Q&A; — About Axes D and E **Q:** Why does FA4 use software-emulated exponentials? Isn't FMA polynomial approximation slower than the hardware exp instruction? **A:** Three nested reasons. (1) **Throughput, not latency**. FMA polynomial exp has higher latency per operation than MUFU.EX2 (the hardware exp instruction), but FMA units have far higher aggregate throughput across the SM than the SFU. When attention is bottlenecked on softmax exp, the SFU becomes the limiting resource — even with high latency per op, FMA wins on aggregate throughput. (2) **Asymmetric scaling on Blackwell**. From H100 to B200, Tensor Core throughput jumped 2.25× but SFU throughput stayed flat. This means the relative cost of routing through SFU went up; FMA-based exp became the better choice. (3) **Pipeline integration**. SFU instructions create a different latency profile than FMA, complicating warp-specialized pipelines. FMA-based exp integrates cleanly with the rest of the pipeline. _FA3 used SFU; FA4's redesign treats SFU as a bottleneck to engineer around_. The lesson: hardware evolution drives algorithmic redesign even when the underlying math is identical. **Q:** If hybrid stacks (3:1 linear:softmax) are the winning pattern, why does Llama 4 still use uniform softmax? **A:** Three reasons that compound. (1) **Engineering risk**. Hybrid stacks introduce complexity — different layer types, different kernels, different optimizer dynamics. Llama 4 chose to deliver via well-understood architecture (full softmax + iRoPE position encoding) rather than introduce hybrid attention. The bet: simpler architecture is easier to scale and ship. (2) **iRoPE solves a different problem**. Llama 4's position-encoding innovation (3 RoPE + 1 NoPE) addresses length generalization (10M context from 256K training). Hybrid linear-softmax doesn't directly help with that. The two innovations are in different design dimensions. (3) **Context limits matter for the choice**. At 1M-10M tokens, the savings from linear attention layers are huge (linear vs quadratic). At 32K-128K, the savings are smaller; the engineering cost of hybrids may not pay off. Llama 4's targets emphasize length, not just efficiency. _The "winning pattern" isn't universal — it's specific to certain workloads_. Llama 4 deliberately chose differently. By Q3-Q4 2026, expect Llama 5 to adopt some hybrid pattern; the convergence will continue but specific labs will deviate based on their priorities. **Q:** Is MoDA more important architecturally than IHA, or vice versa? They both came out within weeks of each other. **A:** They solve different problems and could compose. **IHA (M46, Feb 2026)** : addresses head isolation in attention; pseudo-heads enable cross-head mixing within a layer. Improves multi-step reasoning specifically. **MoDA (March 2026)** : addresses signal degradation across depth; queries attend to KV from preceding layers. Improves deep-stack scaling specifically. _They're orthogonal contributions_ — IHA is about cross-head expressivity within layers; MoDA is about cross-layer attention. A model could (and likely will) use both: IHA for head expressivity inside each attention layer, MoDA for depth-attention augmentation across layers. Forecast: by Q4 2026 / Q1 2027, expect frontier models to incorporate one or both. The "more important" framing is wrong — they're complementary mechanisms targeting different inefficiencies. The deeper observation: _2026 is producing architectural primitives at a faster rate than they can be incorporated into production_. The research-to-production lag of 6-9 months means the frontier of April 2026 reflects mid-2025 research; Feb-March 2026 contributions show up in late 2026 / early 2027 frontier models. **Q:** When should I use Ring Attention vs DeepSpeed Ulysses for sequence parallelism? **A:** Different scaling regimes: 

**Use Ring Attention when** : (a) sequence length is the primary scaling dimension (1M-100M tokens); (b) number of devices is moderate (8-64); (c) your interconnect is slower (cross-node InfiniBand). Ring's communication scales O(N × d) per ring step; better for long sequences with limited devices.

**Use DeepSpeed Ulysses when** : (a) device count is the primary scaling dimension (256-1024+ GPUs); (b) sequence length is moderate (32K-256K); (c) your interconnect is fast (NVLink). Ulysses uses all-to-all; better for many devices with fast interconnect.

**Use USP (Unified Sequence Parallel) when** : you need both — long sequences (1M+) AND many devices (256+). USP applies Ulysses within node + Ring across nodes; captures the strengths of both. Tencent's production workloads use this pattern.

**Use Star Attention when** : the workload is inference-heavy, retrieval-style (long document + short query). Star's block-sparse + global anchor pattern is optimized for this access pattern; ~11× speedup vs full attention.

The practical reality: most teams don't make this choice from scratch. Frameworks (Megatron-LM, DeepSpeed, ColossalAI) bundle specific sequence parallelism strategies; choosing the framework chooses the strategy. _Custom sequence parallelism is for frontier labs; everyone else uses what's bundled with their training framework_.

**Q:** Where do attention variants go in 2027? What's the next architectural primitive? **A:** Three plausible directions, ranked by likelihood: 

(1) **Most likely — composition of existing primitives**. The 2026 contributions (IHA, MoDA, NSA, MLA, hybrid linear-softmax) haven't been jointly composed yet. The frontier 2027 model probably combines: hybrid linear/softmax stack (Qwen3.5-style) + MLA + IHA + MoDA + NSA. Each primitive is well-understood individually; the engineering challenge is making them all work together. _The expressivity gains compound; the implementation complexity adds up_.

(2) **Possible — sub-FP4 attention**. KV cache compression hit 1.5-2 bit floor (M46). Attention computation at FP4 is partial in 2026 (DeepSeek V4 indexer); fully FP4 attention is the next step. Requires kernel-level work (FA5?) and quality validation, but the precision-quality curve suggests it's feasible.

(3) **Less likely but high-impact — fundamentally new mechanism**. Mamba was the last "fundamentally new" sequence model (2023). Since then, all advances have been refinements (Mamba-2, Mamba-3, GDN, KDA) or compositions. A genuinely new primitive would change the field. Wild cards: associative scan over attention; hierarchical attention over arbitrary token clusters; learned sparsity that adapts at inference time. None of these have a clear winner yet.

The deeper observation: _attention is becoming a design space rather than a single mechanism_. The five axes you've learned in M45-M47 will likely persist; new entries on each axis will emerge. The framework outlasts the specific contributions.

**Q:** How does the five-axis taxonomy connect to the rest of the course? **A:** Direct threads to most prior modules: 

**M11/M14 (transformer building, mixed precision)** : M45-M47 describe how attention layers vary; M11 was the baseline transformer block. M14's FP8/BF16 gives the precision context that interacts with FA3/FA4.

**M22 (FlashAttention/Triton kernels)** : M22 introduced FA1/FA2 + Triton; M47's FA4 is the Blackwell-era continuation. The "tiled IO-aware" insight in M22 is what powers FA1-4.

**M26 (KV cache, serving)** : M26 introduced KV cache as a concept; M46 deepened with head topology compression; M47's PagedAttention is the layout that enables the compressions to be deployed.

**M28 (MoE)** : M28's expert routing connects to NSA's gate routing (M45) and MoBA's block routing — same pattern, different objects. Routing is a recurring primitive.

**M30 (RoPE)** : M30 introduced rotary embeddings; M45 mentioned Mamba-3's complex SSM as data-dependent RoPE; M46's MLA decoupled RoPE keeps position info on a separate path; M47's iRoPE alternates RoPE with NoPE layers. _RoPE shows up everywhere in the attention space_.

**M33 (Mamba/SSM)** : M33 introduced selective state spaces; M45 connected this to the linear-attention lineage and Mamba-3 (ICLR 2026); M47's Nemotron 3 Nano is a Mamba-Transformer hybrid in production.

**M41 (NVFP4/Blackwell)** : M41 introduced FP4 quantization; M46's TurboQuant uses 3-bit/2-bit KV; M47's FA4 runs on Blackwell. _The hardware evolution and attention evolution are co-determining_.

**M42-M44 (scaling laws, embodied AI, model zoo)** : M44 inventoried frontier models; M45-M47 explained why their attention choices differ. _The attention trilogy makes M44 readable at the architectural level_.

The deeper point: this trilogy is the synthesis of much of the course. The attention taxonomy is what 47 modules of context lets you see — primitives composing into architectures composing into frontier systems.

## Code Magnets: implement MoDA forward with depth+sequence KV

You're writing the MoDA forward pass that combines current-layer sequence KV with depth KV from preceding layers. Three magnets are wrong choices.

Arrange the magnets to compute MoDA attention output for a single layer.

def moda_forward(x, layer_idx, depth_KV_history, weights): Q = x @ weights.W_Q K_seq = x @ weights.W_K # sequence keys at current layer V_seq = x @ weights.W_V K_all = [K_seq] V_all = [V_seq] K_all = [K_seq] + [K_l for K_l, V_l in depth_KV_history] V_all = [V_seq] + [V_l for K_l, V_l in depth_KV_history] K_combined = torch.cat(K_all, dim=-2) V_combined = torch.cat(V_all, dim=-2) K_combined = torch.stack(K_all, dim=0).mean(dim=0) # average scores = Q @ K_combined.transpose(-2, -1) / math.sqrt(d_head) output = F.softmax(scores, dim=-1) @ V_combined # single softmax output_seq = F.softmax(Q @ K_seq.T) @ V_seq output_depth = F.softmax(Q @ K_depth.T) @ V_depth output = output_seq + output_depth # separate softmaxes return output

show solution
    
    
    def moda_forward(x, layer_idx, depth_KV_history, weights):
        Q = x @ weights.W_Q
        K_seq = x @ weights.W_K  # sequence keys at current layer
        V_seq = x @ weights.W_V
        K_all = [K_seq] + [K_l for K_l, V_l in depth_KV_history]
        V_all = [V_seq] + [V_l for K_l, V_l in depth_KV_history]
        K_combined = torch.cat(K_all, dim=-2)
        V_combined = torch.cat(V_all, dim=-2)
        scores = Q @ K_combined.transpose(-2, -1) / math.sqrt(d_head)
        output = F.softmax(scores, dim=-1) @ V_combined  # single softmax
        return output

The traps:

  * `K_combined = torch.stack(K_all, dim=0).mean(dim=0)`: averages the depth KVs together instead of concatenating. The whole point of MoDA is _data-dependent depth retrieval via attention scores_ — the attention mechanism picks which historical layers to attend to for which queries. Averaging defeats this; the model can't selectively access different layers' information. Concatenation along the sequence dimension keeps each layer's KV available as separate items for attention to choose between. Averaging is the depth dense baseline that MoDA explicitly outperforms because it's not data-dependent.
  * `output_seq = F.softmax(Q @ K_seq.T) @ V_seq` \+ `output_depth = ...` \+ `output = output_seq + output_depth`: applies separate softmaxes to sequence and depth attention, then adds them. MoDA's specific architectural insight is that **queries jointly attend to sequence and depth under a SINGLE softmax normalization**. Separate softmaxes mean the sequence-attention probabilities sum to 1 AND the depth-attention probabilities sum to 1, so adding them gives a total that sums to 2 — semantically inconsistent. Worse, the model can't trade off between sequence and depth — each dimension gets full attention budget. The single softmax over concatenated [K_seq; K_depth] lets the model dynamically allocate attention between current-layer sequence patterns and historical-depth patterns based on what each query needs.
  * The two-line magnets `K_all = [K_seq]` and `V_all = [V_seq]` alone (without the comprehension): incomplete — they create lists with only sequence KV, missing the depth KVs entirely. Without the depth history concatenated in, this is just standard sequence attention; no MoDA mechanism. The full magnets that combine sequence with depth via list comprehension are required.

The pattern: **concatenate sequence KV with depth KV from all preceding layers along the sequence dimension; single unified softmax; standard attention**. Each step has a precise role; the most common implementation bugs are using averaging instead of concatenation (defeats data-dependent retrieval), using separate softmaxes (breaks the unified attention budget), and forgetting to include the depth history entirely.

## Who does what?

Match each Axis D / Axis E variant to its real role.

Concept

Real role

FlashAttention 4

A. March 2026 Blackwell co-design via CuTeDSL; warp specialization, async MMA + softmax overlap, FMA-poly exp, conditional rescaling, 2-CTA MMA; ~20% over cuDNN FA3.

PagedAttention

B. vLLM block-based KV cache layout (16-32 token blocks); eliminates 60-80% fragmentation; 24× throughput vs HuggingFace.

Ring Attention

C. Sequence parallelism via ring K/V rotation across devices; enables 1M-100M token contexts; communication overlapped with compute.

Star Attention

D. NVIDIA inference-time block-sparse + global anchor; 11× speedup for long-context retrieval workloads.

MoDA

E. March 2026 ByteDance + HUST; queries attend to current-layer sequence KV + depth KV from preceding layers under single softmax; +2.11% downstream at 3.7% FLOPs cost.

iRoPE

F. Llama 4: 3 RoPE + 1 NoPE per 4 layers interleaved; enables 10M context generalization from 256K training.

Differential Transformer

G. Computes attention as difference of two softmaxes for sharper patterns; better noise rejection on long contexts.

show solution

**FlashAttention 4** → A  
**PagedAttention** → B  
**Ring Attention** → C  
**Star Attention** → D  
**MoDA** → E  
**iRoPE** → F  
**Differential Transformer** → G 

The mental shortcut: _FA4 redesigns Blackwell's pipeline, PagedAttention pages KV like virtual memory, Ring rotates KV around devices, Star anchors plus locals, MoDA adds depth attention, iRoPE alternates with NoPE, Differential subtracts softmaxes_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team is training a new 70B model targeting 1M-token context on a Blackwell B200 cluster. Walk through their full attention-stack decision, applying the five-axis framework.

show answer

Apply the five-axis framework:

(1) **Axis A — semantic role / normalization** : 70B is below the 100B threshold where MLA wins decisively; the team should consider hybrid linear-softmax for efficiency at 1M context. Recommendation: 3:1 hybrid with Gated DeltaNet (linear) and softmax attention. The Qwen3-Next pattern is well-validated. _Decision: 75% GDN layers, 25% softmax layers._

(2) **Axis B — receptive field** : in the softmax attention layers, full attention is fine at 1M context if combined with proper position encoding and kernel choice. If quality is critical and compute is the bottleneck, consider NSA-style trainable sparse for the softmax layers. _Decision: full attention with FA4; revisit NSA if pretraining shows compute pressure._

(3) **Axis C — head topology** : GQA-8 is the right choice at 70B. MLA's benefits don't fully manifest below 100B; GQA's tooling is mature. _Decision: GQA with 8 KV heads._

(4) **Axis D — kernel implementation** : B200 supports FA4. Use it. PagedAttention for KV layout. For 1M context training, sequence parallelism: USP (Unified Sequence Parallel) — Ulysses within node, Ring across nodes. _Decision: FA4 + PagedAttention + USP._

(5) **Axis E — composition** : 3:1 hybrid with iRoPE-style position handling for length generalization. Optionally add MoDA depth attention if the empirical results scale to 70B (paper validates at 1.5B). _Decision: 3:1 GDN/softmax hybrid + iRoPE position encoding; MoDA as an optional addition pending validation._

**The full architecture** :

  * 80 transformer layers total: 60 GDN linear-attention layers + 20 softmax attention layers (3:1 ratio).
  * Softmax layers use GQA-8, full attention with chunked iRoPE (3 RoPE + 1 NoPE per 4 layers among the softmax layers).
  * FA4 kernels for softmax attention; custom GDN kernel for linear-attention layers.
  * USP sequence parallelism for 1M-token training: Ulysses within 8-GPU node, Ring across 8 nodes.
  * TurboQuant 3-bit KV quantization for inference deployment.

**For deployment** : vLLM with FA4 backend, --kv-cache-dtype turboquant_3bit_nc, PagedAttention block size 32, Star Attention for retrieval-heavy workloads optionally.

The general lesson: **each axis decision flows from workload (1M context), hardware (Blackwell B200), and scale (70B)**. The five axes are not independent — Axis C decision (GQA vs MLA) is constrained by scale; Axis D decision is constrained by hardware; Axis E decision is constrained by context length. Reading 2026 production architectures means parsing these interdependencies.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Explain why FA4 cannot run on the RTX 5090 (consumer Blackwell) despite both being Blackwell generation. What does this tell us about the relationship between algorithm and hardware?

show answer

The hardware-algorithm coupling:

(1) **Two Blackwell variants exist with different ISAs**. SM100 is datacenter Blackwell (B200, GB200, GB300 in NVL72 systems). SM120 is consumer/desktop Blackwell (RTX 5090). Both are "Blackwell" architecturally but have different instruction sets at the SASS level.

(2) **FA4 depends on SM100-specific opcodes**. The pipeline uses:

  * **UTMA*/UTC*** opcodes — Tensor Memory Accelerator instructions for moving data between global memory, shared memory, and TMEM (Tensor Memory).
  * **tcgen05.mma** — the asynchronous matrix-multiply-accumulate that reads operand A directly from TMEM.
  * **2-CTA MMA** — paired CTAs jointly executing one UMMA across both TMEMs.

(3) **SM120 lacks these opcodes entirely**. The GB202 die used in RTX 5090 does not have the tensor memory subsystem; its tensor cores use HMMA (the standard register-to-register MMA approach from Volta SM70 onward). Zero UTC* or UTMA* opcodes in any SM120a binary.

(4) **Result** : FA4 cannot compile to SM120. The CuTeDSL paths emit instructions that don't exist on consumer Blackwell. Multiple Python-level checks block SM120 from reaching the FA4 compilation path; even if those were bypassed, the resulting kernel wouldn't run.

(5) **The deeper lesson — algorithms become hardware-specific** :

  * FA1 (2022) was hardware-portable across Ampere, Volta, Turing, Pascal — same algorithm with different tile sizes.
  * FA3 (2024) required Hopper-specific TMA + WGMMA — couldn't run on Ampere.
  * FA4 (2026) requires SM100-specific TMEM + tcgen05.mma — couldn't run on Hopper, couldn't run on consumer Blackwell.

_Each FA generation targets narrower hardware_. The algorithm-hardware coupling is tightening because GPU architectures are evolving faster — the algorithmic gains require taking advantage of new hardware features that earlier generations don't have.

(6) **Implication for the field** : open-source attention kernels are increasingly hardware-specific. The community fork ecosystem has to maintain multiple kernel variants (FA2 for Ampere, FA3 for Hopper, FA4 for SM100 datacenter Blackwell, separate paths for SM120 consumer Blackwell, AMD ROCm equivalents, etc.). Production frameworks (vLLM, PyTorch SDPA, FlexAttention) handle the dispatch automatically; users specify their hardware and the framework picks the kernel.

The general lesson: **"FlashAttention" is no longer a single algorithm — it's a family of hardware-specific kernels sharing a common algorithmic foundation**. Reading attention kernel work in 2026 means tracking which hardware target you're on; algorithmic claims don't transfer between SM generations.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** A research team is exploring whether to add MoDA to their existing 7B model. Sketch the experimental plan to validate whether MoDA is worth adopting.

show answer

The experimental plan (M37 territory):

(1) **Define the hypothesis precisely**. The MoDA paper validates at 1.5B parameters; will the gains transfer to 7B? The paper claims +2.11% downstream and +0.2 perplexity at 3.7% FLOPs overhead. The team's hypothesis: similar gains at 7B with no quality regression.

(2) **Reproduce baseline first**. Train a 7B baseline with the team's standard architecture (likely GQA + RoPE + FA3/FA4) on a fixed pretraining corpus. Evaluate on the same 10 validation perplexity benchmarks and 10 downstream tasks the MoDA paper used. This establishes the baseline number.

(3) **Train MoDA-augmented variant**. Same pretraining corpus, same hyperparameters, same training duration. Replace standard attention with MoDA in either all layers or every Nth layer (the paper validates several ratios). Compare to baseline.

(4) **Watch for the specific empirical signals** :

  * Perplexity improvement (paper says +0.2 average) — does this hold at 7B?
  * Downstream task improvement (paper says +2.11%) — does this hold?
  * FLOPs overhead (paper says 3.7%) — measure actual training and inference cost.
  * Reduced attention sink behavior — visualize attention patterns; check whether early-token attention concentration is reduced.
  * Post-norm vs pre-norm — paper says MoDA works better with post-norm. Test both.

(5) **Test the kernel performance specifically**. The MoDA paper claims 97.3% of FA2 efficiency at 64K. Measure on the team's hardware (likely H100 or B200) at relevant context lengths. If the actual efficiency is lower (say 80% on the team's setup), the FLOPs overhead claim doesn't hold and the cost-benefit shifts.

(6) **Test sensitivity to layer ratio**. The paper proposes specific patterns (every layer with MoDA, or specific ratios). Test 100% MoDA, 50%, 25%, 10% — find the right ratio for 7B specifically. The paper's optimal might not be the team's optimal.

(7) **Ablation: depth-attention only vs depth-attention + post-norm**. Determine which contribution is doing the work. If most gains come from post-norm itself (independent of depth attention), the team should adopt post-norm without MoDA's complexity.

(8) **Long-context evaluation**. Paper validates at 64K context. The team's deployment is at 256K — does MoDA hold at longer context? The depth-KV memory cost scales with sequence length AND number of preceding layers, so longer contexts amplify the cost.

(9) **Production decision criteria** :

  * If gains hold (+1-2% downstream, FLOPs overhead under 5%): adopt for next major version.
  * If gains are smaller (+0.5-1%): keep as research direction; not worth production engineering complexity.
  * If gains don't transfer (close to noise): publish negative result; don't adopt.

(10) **Engineering cost to factor in** : MoDA's depth-KV access is a new memory pattern; integrating with existing serving infrastructure (vLLM, PagedAttention) requires kernel work. The published Triton kernel is a starting point but production-grade integration is more.

The general lesson: **research papers published at one scale don't automatically translate to production at a different scale**. M37's eval rigor + this kind of structured ablation is what separates "we read the paper and it sounds promising" from "we validated the claim on our actual deployment." Most architectural innovations require this 2-4 week validation cycle before adoption.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Forecast: by Q4 2027, what does the typical frontier-scale attention stack look like? Reason from the convergence patterns in M44-M47.

show answer

Reading the convergence trajectory across the trilogy:

**The likely Q4 2027 frontier stack** :

(1) **Axis A — hybrid linear-softmax dominant**. The 75:25 ratio (Qwen3-Next pattern) becomes universal at frontier. Linear-attention variant: probably Mamba-3 or its successor (data-dependent RoPE-equivalent updates). Softmax variant: standard attention with MLA head topology. _Pure softmax stacks (Llama 4 today) become a minority pattern_.

(2) **Axis B — sparse attention in softmax layers**. The 25% of softmax layers themselves use sparse trainable attention (NSA-style or DeepSeek's CSA+HCA hybrid). Full attention becomes the layer of last resort, used only when exact recall matters. _Pure full attention disappears from frontier production_.

(3) **Axis C — MLA + IHA composition**. MLA for KV compression; IHA for cross-head expressivity. The composition was implicit in M46 (MLA's per-head specialization preserved through query/output paths) and M46's IHA discussion (cross-head mixing); their joint adoption is the Q4 2027 prediction. _GQA persists below 100B; MLA + IHA is the >100B answer_.

(4) **Axis D — FA5 (or successor) on Blackwell-Ultra / Rubin**. NVIDIA's roadmap has Rubin GPUs in 2026-2027. FA5 will likely co-design for those. Sequence parallelism via USP or its successor; PagedAttention extends to multi-GPU coordination. _Single-GPU attention is mature; multi-GPU coordination is the new frontier_.

(5) **Axis E — depth attention as standard option**. MoDA-style depth attention or a successor becomes a standard architectural option. Frontier models include depth-KV alongside sequence-KV; the 3.7% FLOPs cost is small enough that adoption is broad. _"Depth as a first-class dimension" alongside sequence and head dimensions_.

**The KV-cache stack at Q4 2027** :

  * MLA-compressed KV cache (M46) → ~6× reduction over MHA.
  * TurboQuant successor at 1.5-2 bit precision → another ~3-4× reduction.
  * Expected Attention or learned eviction → another 2-3× effective capacity.
  * InfiniGen-style offloading for cold pages.
  * Total: ~50-100× compression vs MHA-FP16, allowing 10M-100M context routinely on B200/Rubin systems.

**Likely non-changes (architectural choices that have settled)** :

  * Transformer base architecture (despite M33's SSM developments — production sticks with transformers as the framing).
  * Pre-normalization with RMSNorm (post-norm rebound from MoDA aside).
  * SwiGLU / GeGLU FFN activation.
  * RoPE foundation with extensions (iRoPE, NoPE-layer interleaving).
  * FA-style tiled IO-aware attention (the algorithmic insight from M22, refined over generations).

**Wild cards that could disrupt the forecast** :

  * Pure linear-attention models (Mamba-only) become competitive — would shift the 75:25 ratio toward 100:0. Currently looks unlikely but possible.
  * A fundamentally new mechanism (not on the five axes) emerges — Mamba in 2023 was the last such; unclear if another is brewing.
  * Sub-FP4 attention computation (FA5 in FP3 or FP2) — would change the precision-quality calculus.

**The deeper observation** : _the five-axis taxonomy you've learned is likely to persist_. New entries on each axis will emerge; the framework outlasts the specific contributions. By Q4 2027, expect to use M45-M47 as the structure for understanding new architectures, with new specific instances filled in. The entries change; the taxonomy doesn't.

The general lesson: **architectural literacy is durable; specific architecture choices are not**. The trilogy was built so that you'd be able to read 2027 model cards even though they don't yet exist.

### What just happened?

  * Closed the four-axis taxonomy with Axis D (kernel implementation) and Axis E (compositional architecture). The trilogy provides the framework for parsing any 2026 attention layer.
  * **FlashAttention lineage** : **FA1 (2022)** tiled exact attention; **FA2 (2023)** better thread-block parallelism; **FA3 (2024)** Hopper warp specialization with TMA + WGMMA; **FA4 (March 2026)** Blackwell co-design via CuTeDSL.
  * **FA4's four innovations** : (1) redesigned async pipeline with orchestration warps + compute warps; (2) software-emulated exponentials via FMA polynomial approximation (avoids SFU bottleneck); (3) conditional softmax rescaling (10× fewer rescaling ops); (4) 2-CTA MMA up to 256×256×16 tile size. ~600+ TFLOPs/s on B200 in FP8; ~20% over cuDNN FA3; 1.2-3.2× over Triton FlexAttention.
  * **FA4 hardware constraint** : requires SM100 (datacenter Blackwell B200/GB200/GB300). Does NOT run on RTX 5090 (SM120 consumer Blackwell) — the GB202 die lacks UTMA*/UTC* opcodes and has no TMEM subsystem.
  * **Sequence parallelism family** : **PagedAttention** (vLLM 2023) — block-based KV layout, fragmentation 60-80% → under 4%, 24× throughput. **Ring Attention** (Liu et al.) — sequence-parallel attention via rotating K/V around devices, enables 1M-100M token contexts. **Striped Attention** (MIT) — causal-balanced ring for load balancing. **Star Attention** (NVIDIA) — block-sparse + global anchor, 11× speedup for retrieval. **TokenRing** (SJTU) — bidirectional point-to-point. **DeepSpeed Ulysses** — extreme long-sequence via head partitioning. **USP** (Tencent) — Ulysses + Ring hybrid.
  * **Hybrid layer-type compositions in 2026 frontier** : 
    * **Qwen3-Next/Qwen3.5** : 3:1 Gated DeltaNet to softmax attention.
    * **Kimi Linear** : KDA linear + MLA softmax.
    * **Ling 2.5** (1T MoE): Lightning Attention + MLA.
    * **Nemotron 3 Nano** : Mamba-2 SSM + sparse softmax hybrid.
    * **DeepSeek V4** : CSA (4× compress + Lightning Indexer FP4) + HCA (128× compress dense) alternating.
    * **Llama 4** : iRoPE — 3 RoPE layers + 1 NoPE layer per 4, enables 10M context.
  * The 75:25 ratio of cheap-to-expensive layers has emerged as approximately optimal across multiple labs; uniform stacks are an artifact of pre-2024 transformers.
  * **MoDA — Mixture-of-Depths Attention (Zhu et al., March 2026, ByteDance Seed + HUST)** : each head attends to current-layer sequence KV AND depth KV from preceding layers under a single softmax. Hardware-efficient via chunk-aware layout: **97.3% of FA2 efficiency at 64K**. Empirical: +0.2 perplexity, +2.11% downstream, only 3.7% FLOPs overhead. Reduces attention-sink behavior. Works best with post-norm.
  * **Other compositional primitives** : KArAt (vision, learnable basis functions); Multi-Token Attention; Differential Transformer (subtracts two softmaxes); Talking-Heads (superseded by IHA — M46).
  * **The five-axis decision framework** : every 2026 attention layer is a coordinate on (A) semantic role/normalization, (B) receptive field, (C) head topology, (D) kernel implementation, (E) compositional architecture. The trilogy gives the framework; specific frontier models are points in this 5D space.
  * Production deployment recipes: 
    * **1T+ frontier MoE at 1M context** : hybrid linear/softmax + MLA + sparse + FA4 + USP.
    * **30-100B at 256K** : hybrid GDN/softmax + GQA + FA3/4 + TurboQuant.
    * **7B edge** : pure softmax + GQA-8 + FA3 + PagedAttention.
    * **Long-context inference on existing model** : XAttention + TurboQuant + Expected Attention + InfiniGen.
  * The reflex for 2026: read any model card by identifying its five-axis coordinate. Each axis decision flows from workload + hardware + scale; the axes are not independent.
  * Forecast for Q4 2027: hybrid linear/softmax universal at frontier; MLA + IHA composition for head topology >100B; FA5 on Rubin/Blackwell-Ultra; depth attention (MoDA-style) as a standard option; ~50-100× KV cache compression via MLA + sub-FP4 quant + eviction + offloading.

## Closing the trilogy — and the course

You started the course with `x.stride()` on a tensor. You finished it parsing the attention design space across five axes, recognizing the hardware-algorithm coupling that makes FA4 a Blackwell-specific kernel, and identifying the compositional patterns that distinguish DeepSeek V4 from Qwen3-Next from Llama 4 from Kimi Linear.

The trajectory through 47 modules: _tensors → autograd → modules → training → distributed → kernels → MoE → transformer → reasoning models → mech interp → multimodal → distributed RLHF → agentic environments → Blackwell/NVFP4 → scaling laws → embodied AI → comparing the model zoo → and now, the attention compendium that synthesizes much of the rest_.

_The framework is what's durable_. Specific frontier architectures change every six months — DeepSeek V5 by Q3 2026, Llama 5 by Q4, FA5 by 2027, MoDA-derivatives by year-end. But the five axes will likely persist, with new entries on each. Reading any 2026-2027 paper means identifying its place in the taxonomy: which axis does it extend? What's the new entry? How does it compose with existing primitives?

The deeper craft you've built across the course: _architectural literacy isn't memorizing specific models_. It's the ability to parse new architectures as compositions of primitives you understand, to evaluate empirical claims with M37's eval rigor, to estimate compute economics with M42's framework, and to integrate hardware constraints (M22/M41) with algorithmic choices (M30/M33/M44/M45/M46/M47). New papers will keep being published; new models will keep shipping; the framework you've built lets you read them.

The 2027 frontier models will mostly use these same primitives, in different compositions. Possibly new primitives (one or two on each axis); possibly fundamentally new mechanisms not yet imagined. The taxonomy has room for both. _You'll be able to read those papers_.

This is what the course was for. Now go build something that didn't exist before.
