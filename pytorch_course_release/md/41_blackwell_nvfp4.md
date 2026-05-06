# Module 41 — Blackwell & NVFP4 training

# _Blackwell & NVFP4 training:_ when 4 bits is enough

_Part XI · Module 41 · April 2026 currency_

— from FP8 in M14 to NVFP4 with two-level scaling, why MLPerf Training v5.1 swept on FP4, the recipe ingredients (Hadamard transforms, stochastic rounding, healing), and what Tensor Memory + CTA pairs change in CUTLASS

\--- 

M14 covered mixed precision through FP8. The hardware moved. Two generations later — Blackwell (B200, B300, GB200/GB300 NVL72) and Blackwell Ultra (B300 with the second-gen Transformer Engine) — bring native FP4 acceleration. The early skeptical question ("can you actually train at 4 bits without quality loss?") got an answer in late 2025: yes, with the right recipe. NVIDIA's MLPerf Training v5.1 submission (November 2025) swept all seven benchmarks using NVFP4; Llama 3.1 405B trained in 10 minutes on 5,120 Blackwell GPUs. The recipe is now public and reproducible.

This module is the hardware-and-numerics update for 2026. It assumes M14's mixed-precision foundations and pushes them down: **what changes when you go from FP8 to FP4, what NVFP4 specifically does that distinguishes it from MXFP4, what the production training recipe looks like (Random Hadamard transforms, stochastic rounding, two-dimensional quantization, healing in FP8), and what's new in Blackwell that CUTLASS exposes** (Tensor Memory, CTA pairs, 5th-gen Tensor Cores, TMA multicast). By the end you'll know whether to use FP4 for your next training run, what infrastructure changes that implies, and what the Rubin generation arriving in 2026-2027 will likely change again.

> **★ KEY IDEA**  
>  The hardware has two generational updates since M14. **Blackwell (B200, GB200 NVL72)** : 5th-gen Tensor Cores with native FP4, 192GB HBM3e at 8 TB/s, NVLink 5 at 1.8 TB/s, 192GB Tensor Memory (TMEM), CTA-pair MMA. **Blackwell Ultra (B300, GB300 NVL72)** : refresh with 270GB memory per GPU and 1.5× NVFP4 throughput vs B200 (15 PFLOPS dense NVFP4 per GPU). **NVFP4 format** : 16-value micro-blocks, each with FP8 (E4M3) scale, plus per-tensor FP32 scale. Two-level scaling beats MXFP4's E8M0 single-level by enabling fine-grained dynamic range adaptation; near-FP8 accuracy with 2× memory reduction. **The training recipe** (NVIDIA NVFP4 pretraining paper, Sep 2025; FP4 All the Way, May 2025; Quartet II, 2026): Random Hadamard transforms (RHT) to disperse outliers, stochastic rounding (SR) for unbiased gradient estimation, split rounding (SR backward / RtN forward), two-dimensional quantization for forward-backward consistency, selective high-precision layers (embedding, output head stay FP8/BF16), **healing** (last few iterations in FP8 to recover final accuracy). Validated at 12B/10T tokens matching FP8 baseline; 7B/200T tokens (FP4 All the Way) matching BF16 baseline. **MLPerf Training v5.1** : NVIDIA swept all 7 benchmarks; Llama 3.1 405B in 10 minutes on 5,120 Blackwell GPUs; 3.2× over Hopper FP8 at same count; 1.4× over FP8 on same Blackwell rack; GB300 NVL72 1.9× over GB200 NVL72. **CUTLASS 3.8+** : Tensor Memory (TMEM) replaces registers for MMA accumulators; CTA pairs (two SMs share operands, 2-SM UMMA); TMA multicast; single-thread MMA launch; `tcgen05.mma` instruction. **CuTile (late 2025)** : Python tile-centric DSL abstracting warps/registers/SMEM. **Rubin (2026-2027 lookahead)** : 35 PFLOPS NVFP4 training, 50 PFLOPS NVFP4 inference — 3.5×/5× over Blackwell. 

## Two new faces — Part XI continued

N

NVFP4

"I'm 4 bits per value with two-level scaling. Near-FP8 accuracy at half the memory and double the speed."

Each of my values is E2M1 — 1 sign bit, 2 exponent bits, 1 mantissa bit. That's not enough by itself; FP4 alone is too coarse for serious training. **The trick is two-level scaling**. I group values into _micro-blocks of 16_ ; each block shares an FP8 (E4M3) scaling factor; on top of that, the entire tensor has a single FP32 scalar. The two-level approach beats MXFP4's E8M0 single-level scaling because my 16-value blocks adapt to local dynamic range better, and my E4M3 block scales are fine-grained where MXFP4's power-of-two E8M0 is coarse. **I'm what enabled MLPerf Training v5.1's clean sweep**. With the right recipe — Hadamard transforms, stochastic rounding, healing — I match FP8 quality on 12B/10T-token training. I'm what M14's FP8 graduates to.

T

TMEM

"I'm Tensor Memory — Blackwell's new on-chip tier between SMEM and registers, designed for MMA accumulators."

In Hopper, MMA accumulators sat in registers. That meant register pressure scaled with tile size; large tiles starved threads. In Blackwell, I'm a new on-chip memory tier (192KB per SM) specifically for MMA accumulators and operands. _UMMA — Universal MMA, the 5th-gen Tensor Core instruction`tcgen05.mma` — uses me directly_; no registers needed for matrix data, no register pressure tied to tile size. Combined with CTA pairs (two SMs sharing operands), I let single-thread MMA launches run with massive tile sizes the previous generation couldn't fit. The CUTLASS 3.8 Blackwell GEMM examples are organized around me. **I'm why Blackwell GEMM throughput is 7× over Hopper** ; the silicon has more FP4 FLOPs, but I'm what lets kernels actually use them.

## The hardware progression: H100 → B200 → B300

Walking through the relevant specs to anchor the rest of the module. The Hopper generation (M14's home turf) and the two Blackwell variants:

Datacenter GPU progression for AI training, April 2026 Spec| H100 (Hopper, 2022)| H200 (Hopper, 2024)| B200 (Blackwell, 2024)| B300 (Blackwell Ultra, 2025)  
---|---|---|---|---  
Memory| 80GB HBM3| 141GB HBM3e| 192GB HBM3e| 270GB HBM3e  
Mem bandwidth| 3.4 TB/s| 4.8 TB/s| 8 TB/s| ~8 TB/s  
NVLink| NVLink 4 (900 GB/s)| NVLink 4 (900 GB/s)| NVLink 5 (1.8 TB/s)| NVLink 5 (1.8 TB/s)  
FP8 dense| ~2 PFLOPS| ~2 PFLOPS| 4.5 PFLOPS| ~5 PFLOPS  
FP4 dense| —| —| 9-10 PFLOPS| 15 PFLOPS (Ultra)  
Tensor Memory| —| —| 192KB / SM| 192KB / SM  
5th-gen Tensor Core| —| —| `tcgen05.mma`| `tcgen05.mma`  
Rack-scale unit| HGX H100 (8 GPU)| HGX H200 (8 GPU)| **GB200 NVL72** (72 GPU + 36 Grace)| **GB300 NVL72** (72 GPU + 36 Grace)  
Per-rack FP4 compute| —| —| ~720 PFLOPS| ~1.1 EFLOPS  
  
Three things matter most for training:

  1. **FP4 native compute**. Blackwell Tensor Cores execute NVFP4 GEMMs at 2× the FP8 rate; Blackwell Ultra at 3×. This is the headline performance gain — if your training pipeline is GEMM-bound and you can train in FP4, you halve (or third) compute time.
  2. **Memory and bandwidth**. 192GB → 270GB and 4.8 → 8 TB/s remove memory pressure on large models. A 70B model fits in a single B300 with comfortable headroom for KV cache; a 405B model fits across 8 B300s. M16-M17's distributed parallelism still applies but at a different scale.
  3. **Rack-scale interconnect**. GB200/GB300 NVL72: 72 GPUs in one NVLink 5 domain. Communication patterns that were "across-node" on HGX become "within-rack" on NVL72. Tensor parallelism, expert parallelism for MoE, sequence parallelism — all benefit substantially.

The economic implication is concrete. NVIDIA's published numbers (December 2025 blog): **GB200 NVL72 delivered 3.2× faster Llama 3.1 405B training than Hopper FP8 at same GPU count, and ~2× the performance per dollar at GPU rental prices**. Blackwell Ultra adds another ~12% on average over Blackwell. Inference deltas are larger (15× per-system in some cases) but training is what matters here.

## NVFP4: the format itself

The headline question of any narrow-precision format: _where do you put the bits?_ NVFP4 is 4 bits per value (1 sign, 2 exponent, 1 mantissa = E2M1) plus a multi-level scaling scheme that recovers dynamic range without spending bits per value.

NVFP4 two-level scaling: 16-value micro-blocks with E4M3 scales + tensor-level FP32 Tensor (e.g., a weight matrix tile) Block 0: 16 values … each = E2M1 (4 bits) Block 1: 16 values … Block 2: 16 values … … many blocks … Each value is 4 bits — total tensor uses ½ bytes per element of stored data Level 1: Per-block scales (one per micro-block of 16) FP8 E4M3 scale₀ FP8 E4M3 scale₁ FP8 E4M3 scale₂ … 8 bits per block, fine-grained, adapts to local dynamic range; vs MXFP4's 8-bit E8M0 (power-of-two only) Level 2: Per-tensor scale FP32 scalar tensor-wide → Provides global dynamic range; computed once per tensor Effective value = E2M1_value × FP8_block_scale × FP32_tensor_scale

The two-level scaling matters because of the dynamic range problem. A weight tensor in a transformer has values ranging from ~10⁻⁵ to ~10¹; FP4 (E2M1) has ~4 unique non-zero magnitudes per sign. Without scaling, almost everything underflows or saturates. With _per-tensor_ scaling alone (single scalar), you can shift the whole range but can't adapt to local variation — some blocks will have extreme values that swamp the rest, losing precision elsewhere. With _per-block_ scaling alone, you handle local variation but can't represent the tensor's overall scale efficiently.

Two-level scaling does both: per-tensor FP32 captures the global magnitude; per-block FP8 (E4M3) captures local variation across 16-value micro-blocks. The block size of 16 is empirically validated (FP4 All the Way, May 2025): below 16 elements, accuracy gains are diminishing; above 32 (MXFP4's choice), important local information is lost.

### NVFP4 vs MXFP4

The Open Compute Project's MXFP4 format predated NVFP4 and uses a different design:

NVFP4 vs MXFP4 design choices Choice| NVFP4 (NVIDIA)| MXFP4 (OCP)  
---|---|---  
Block size| 16 values| 32 values  
Block scale format| FP8 (E4M3)| E8M0 (power-of-two)  
Tensor-level scale| FP32 scalar| None  
Element format| FP4 (E2M1)| FP4 (E2M1)  
Native HW support| Blackwell Tensor Cores| Blackwell Tensor Cores (also)  
  
The empirical comparison (NVIDIA NVFP4 paper, FP4 All the Way): NVFP4 produces lower quantization error and better training stability than MXFP4 at the same compute cost. The reasons: smaller blocks (16 vs 32) capture local variation better; E4M3 scales are fine-grained where E8M0 is power-of-two only; the per-tensor FP32 captures global range MXFP4 lacks. _Blackwell hardware accelerates both formats; NVFP4 is the production default for new training in 2026._

## The training recipe: what makes NVFP4 actually work

FP4 training fails naively. Just casting weights and activations to NVFP4 produces unstable training, with loss curves diverging in the first few thousand steps. The recipe that makes it work is a stack of techniques developed through 2024-2026:

### 1\. Random Hadamard transforms (RHT)

The problem: weight and activation tensors have outliers — a small number of extreme values that dominate block-level scaling. With outliers present, the FP8 block scale gets pulled toward the outlier, and the other 15 values in the block lose precision.

The fix: **apply a random Hadamard transform before quantization**. The Hadamard transform is a structured orthogonal transform (constructable from ±1 matrices); it's its own inverse. Multiplying a vector by a random Hadamard matrix produces a transformed vector whose values have similar magnitude — outliers get distributed across all positions. After GEMM, you transform back.
    
    
    def nvfp4_quantize_with_rht(tensor, hadamard_seed):
        # Step 1: apply random Hadamard transform to disperse outliers
        H = random_hadamard_matrix(tensor.shape[-1], seed=hadamard_seed)
        transformed = tensor @ H
    
        # Step 2: quantize to NVFP4 (now with bounded block-level outliers)
        nvfp4_values, block_scales, tensor_scale = quantize_nvfp4(transformed)
    
        return nvfp4_values, block_scales, tensor_scale, H
    
    # During GEMM:
    #   y = (W @ H) @ (H^T @ x) = W @ x   (since H @ H^T = I)
    # Both operands quantized in their RHT-transformed space
    # The two H's cancel mathematically; in practice, you absorb them into adjacent layers

The Hadamard transform is cheap (O(n log n) and structured), reversible (orthogonal), and provably bounds the variance of stochastic rounding. _RHT is the single most important technique for FP4 training stability_ ; the published recipes (NVIDIA's, FP4 All the Way's, Quartet's) all use some variant.

### 2\. Stochastic rounding (SR) for unbiased gradients

Round-to-nearest (RtN) is the default rounding mode and the obvious choice. For training, it has a problem: it's _biased_. When you average many quantized gradients, the rounding errors accumulate non-zero — small biases compound across millions of steps.

**Stochastic rounding** replaces deterministic rounding with probabilistic rounding: a value lying 0.3 between two representable quantization levels rounds up with probability 0.3, down with probability 0.7. The expected value equals the original; quantization is unbiased.

The empirical finding (FP4 All the Way, May 2025; Chmiel et al.): **split rounding strategy — SR backward / RtN forward — beats both pure-SR and pure-RtN**. Forward pass uses RtN for stability (deterministic forward outputs, no extra noise during inference); backward pass uses SR for unbiased gradient estimates. Combining them gives the best of both.
    
    
    def stochastic_round_to_nvfp4(value, block_scale, tensor_scale):
        # Effective scale = block_scale * tensor_scale
        # Quantize value into FP4 levels [-6, -4, -3, -2, -1, -0.5, -0, 0, 0.5, 1, 2, 3, 4, 6, 8, 12]
        # (the 16 representable magnitudes of E2M1 with sign)
        scaled = value / (block_scale * tensor_scale)
        floor_level = find_lower_level(scaled)
        ceil_level = find_upper_level(scaled)
        p = (scaled - floor_level) / (ceil_level - floor_level)
    
        # Probabilistic rounding
        if torch.rand(1).item() < p:
            return ceil_level   # round up with probability p
        else:
            return floor_level # round down with probability 1-p

Blackwell Tensor Cores support stochastic rounding natively for FP4 conversion instructions; the recipe doesn't pay extra latency for SR. _Without SR, FP4 training diverges; with SR, it converges to FP8-equivalent quality._

### 3\. Two-dimensional quantization

For a GEMM `C = A @ B`, both A and B need to be quantized. The naive approach quantizes each independently: A's blocks are along its rows, B's blocks are along its columns. The problem: forward (`C = A @ B`) and backward (`dA = dC @ B^T`, `dB = A^T @ dC`) passes use the operands in different orientations, and per-row quantization for forward doesn't match per-column quantization for backward — the same tensor has two different quantization representations.

**Two-dimensional quantization** : quantize A along both dimensions (block-scaled rows AND block-scaled columns), keep both representations. Forward uses one; backward uses the other. The cost is 2× the scale storage; the benefit is consistency between forward and backward, which substantially improves convergence.

This is implementation-heavy — most production NVFP4 training stacks ship with 2D quantization built into the linear layer. Megatron-Core, NVIDIA Transformer Engine, and the open-source FP4 training repositories all support it as of April 2026.

### 4\. Selective high-precision layers

Not every layer benefits from FP4. The published recipes keep specific layers in higher precision:

  * **Embedding layer** : token embeddings often have high dynamic range and are looked up sparsely; FP4 quantization hurts. Keep in FP8 or BF16.
  * **Output (LM head)** : the final linear projection to vocabulary logits is sensitive to small errors in unlikely tokens. Keep in FP8.
  * **LayerNorm / RMSNorm** : normalization statistics need higher precision; the multiply by gain is fine in FP4 but the actual normalization should be in FP32.
  * **Optimizer state** : Adam moments and master weights stay in FP32 always.
  * **Loss computation** : cross-entropy and softmax in FP32 to preserve numerical stability.

The bulk of the GEMMs (attention QKV projections, attention output, MLP up/down) all run in NVFP4. That's where the FLOP savings are.

### 5\. Healing

NVIDIA's MLPerf v5.1 submission introduced **healing** : _keep the last few iterations of training in FP8 precision rather than NVFP4_. The intuition: by the end of training, gradients are small and quantization noise (relative to gradient magnitude) becomes more harmful. Switching to higher precision for the final iterations recovers the last fraction of accuracy that pure FP4 would leave behind.

Concretely: train ~95% of total iterations in NVFP4; switch to FP8 for the final ~5%. The cost is small (5% of compute is roughly 1.7× more expensive in FP8 vs FP4); the accuracy gain is meaningful — typically 0.2-0.5% on downstream evals, sometimes more for sensitive tasks.

Healing was a key ingredient in NVIDIA's MLPerf v5.1 sweep. Production training runs increasingly include a healing phase as standard.

### The combined recipe at a glance

The NVFP4 training recipe — five techniques compose to FP8-equivalent quality 1\. RHT disperse outliers before quantize 2\. Split rounding SR backward, RtN forward 3\. 2D quant consistent fwd/bwd representations 4\. Selective embed/head/norm stay FP8/BF16 5\. Healing last 5% in FP8 final accuracy Validation: published results matching higher-precision baselines • NVIDIA NVFP4 paper (Sep 2025, refined Mar 2026): 12B model on 10T tokens, NVFP4 matches FP8 baseline on loss + downstream evals • FP4 All the Way (May 2025): 7B model on 200T tokens, fully quantized FP4 weights+activations+gradients matches BF16 • MLPerf Training v5.1 (Nov 2025): NVIDIA swept all 7 benchmarks with NVFP4 (LLM tests included) • Quartet II / MS-EDEN (2026): stochasticity moved to microscale factors, further improving convergence → As of April 2026, NVFP4 is production-grade for pretraining with the right recipe.

## MLPerf Training v5.1: the proof point

NVIDIA's MLPerf Training v5.1 submission (November 2025) was the public validation that NVFP4 training works at scale:

  * **Sweep of all 7 benchmarks**. NVIDIA was the only platform to submit on every test; Blackwell-based systems delivered fastest time-to-train across LLM, image generation, recommender systems, computer vision, and graph neural networks.
  * **Llama 3.1 405B in 10 minutes** on 5,120 Blackwell GPUs (640 GB200 NVL72 systems). This is the headline number — pretraining a 405B model in single-digit minutes was unthinkable a year earlier.
  * **3.2× faster than Hopper FP8** at the same GPU count on Llama 3.1 405B. Pure hardware-software combination of NVFP4 + Blackwell Tensor Cores + improved cuBLAS / Transformer Engine / Megatron-Core.
  * **1.4× higher than FP8 on the same Blackwell rack**. Apples-to-apples improvement from FP8 → NVFP4 on identical hardware.
  * **Llama 3.1 405B at 5,120 Blackwell GPUs is 1.9× faster than at 512 across multiple GB200 NVL72 systems using FP8 in the prior round**.
  * **GB300 NVL72 vs GB200 NVL72** : ~12% reduction in training time on Llama-2-70B LoRA and Llama-3.1-8B (Nebius's submission), representing the Blackwell Ultra refresh.

NVIDIA's submission used: **NVFP4 for the bulk of training** ; **FP8 for attention BMM inputs** ; **fused RoPE kernels** ; **healing in FP8 for the final iterations** ; **device-to-device memory copy avoidance** ; and the underlying CUTLASS, Transformer Engine, and Megatron-Core infrastructure. _The recipe is now public; the same components are available open-source for production training._

## What's new in Blackwell silicon

Beyond raw FLOPs, Blackwell introduces architectural changes that CUTLASS and other kernel libraries must use to extract performance. Three matter most:

### Tensor Memory (TMEM)

A new on-chip memory tier between SMEM and registers, dedicated to MMA accumulators and operands. 192KB per SM. Replaces the old Hopper pattern where MMA accumulators sat in registers.

The implication: **MMA operations don't need register space for matrix data**. UMMA (Universal MMA, the 5th-gen Tensor Core instruction `tcgen05.mma`) takes both inputs from SMEM (or one from SMEM, one from TMEM); the accumulator lives in TMEM. With no register pressure tied to tile size, you can run massive tiles that wouldn't fit on Hopper. Blackwell GEMM throughput is ~7× over Hopper for the same algorithm partly because of TMEM-enabled larger tiles.

### CTA pairs and 2-SM UMMA

Two CTAs (Cooperative Thread Arrays — Blackwell's term for thread blocks) in a thread block cluster form a "CTA pair" if their ranks differ in only the last bit (0/1, 4/5, etc.). A CTA pair maps to a Texture Processing Cluster (TPC) — two physically adjacent SMs.

The 5th-gen Tensor Core's **2-SM UMMA** instruction operates at CTA-pair granularity: the two CTAs share input operands. One CTA loads operand A, the other loads operand B; both compute partial outputs; the result is combined. Memory bandwidth for shared operands is halved (each operand loaded once across the pair instead of twice).

The CUTLASS 3.8 Blackwell tutorials walk through implementing 2-SM UMMA: TMA multicast loads operand into both CTAs; tile sizes increase to 256×256 or larger; the synchronization primitives are different from Hopper's WGMMA. _Production GEMM kernels for Blackwell increasingly use 2-SM UMMA_ ; the CUTLASS examples are organized around it as the default Blackwell pattern.

### Single-thread MMA launch

On Hopper, MMA instructions required all threads in a warp to participate; on Blackwell, **UMMA is launched by a single thread**. The MMA executes asynchronously; other threads in the CTA can do other work simultaneously.

The implication: **warp specialization becomes more flexible**. The traditional pattern (one warp does TMA loads, one warp does MMA, one warp does epilogue) extends to fine-grained per-thread specialization. Combined with the no-register MMA from TMEM, the CTA's main execution can decouple substantially from MMA work — MMA "fires and forgets" while other threads handle pre/post-processing.

### CUTLASS 3.8 and CuTile

The software side of Blackwell's hardware features. **CUTLASS 3.8** (released through 2025-2026) added Blackwell support — TMEM, CTA pairs, 2-SM UMMA, TMA multicast, native NVFP4 and MXFP4 GEMMs with hardware block-scaling. The CUTLASS team's Blackwell tutorials (Colfax Research, late 2025 / early 2026) walk through writing GEMM kernels from scratch using these features.

**CuTile (NVIDIA, late 2025)** : a Python-based tile-centric DSL that abstracts warps, registers, and SMEM behind high-level primitives (`ct.load`, `ct.mma`, `ct.store`). Targets the same hardware features (Tensor Cores, TMA) but reduces the engineering burden of writing CUTLASS kernels. Closer to Triton in usability; closer to CUTLASS in performance ceiling.

For most production training in 2026: **use Megatron-Core or Transformer Engine** (which use CUTLASS underneath); the kernel-level details are abstracted. CuTile and CUTLASS are for kernel authors and performance specialists; the typical training engineer writes high-level PyTorch.

## The economics: why this matters

The hardware/format generation jumps compound on training cost. Concrete published numbers as of December 2025 / early 2026:

  * **~2× performance-per-dollar of H100** : GB200 NVL72 vs HGX H100 on Llama 3.1 405B at GPU rental prices.
  * **~3.2× per-GPU training speedup** : GB200 NVL72 with NVFP4 vs HGX H100 with FP8 at same GPU count.
  * **Inference: $0.02 per million tokens** on GPT-OSS-120B with Blackwell + FP4 + TensorRT-LLM (SemiAnalysis InferenceX, Q1 2026); ~4.5× cheaper than Hopper-vLLM.
  * **~50% memory reduction** per parameter going from FP8 to NVFP4. A 70B model needs ~140GB in FP8; ~70GB in NVFP4. Single-GPU training of 70B-class models becomes practical.

The implication for project planning: **same training budget produces 2-3× the model quality, or the same model 2-3× cheaper**. Training that cost $10M on Hopper costs ~$3-5M on Blackwell. Training that cost $1M scales up to $1M of more capable training. The economic shift is substantial enough to reshape who can train frontier models.

## Lookahead: Rubin (2026-2027)

NVIDIA has publicly committed to Rubin as Blackwell's successor. The published projections (Feb 2026 NVIDIA developer blog):

  * **35 PFLOPS NVFP4 training compute per GPU** — 3.5× over Blackwell.
  * **50 PFLOPS NVFP4 Transformer Engine inference** — 5× over Blackwell.
  * Larger HBM (likely HBM4) at higher bandwidth.
  * NVLink 6 with substantial inter-GPU bandwidth gains.
  * Likely additional precision modes — narrower than FP4 is theoretically possible (FP3, custom logarithmic formats); industry watching for what the Rubin Tensor Cores will support.

The expected production timeline: Rubin sampling 2026, broad availability 2027. _The training recipe established for NVFP4 on Blackwell will likely transfer to Rubin's supported formats with minor adjustments_ — the techniques (Hadamard transforms, stochastic rounding, two-dimensional quantization, healing) generalize. Whether sub-FP4 training (3-bit, 2-bit) becomes practical is the open question; published research suggests difficult below 4 bits without substantial accuracy loss.

## Failure modes specific to NVFP4 training

FP4 training has its own characteristic failures, distinct from FP8 training:

  * **Loss divergence at small gradients**. The FP4 All the Way paper identified a theoretical threshold: when gradient norm falls below ~√3 × quantization noise, FP4 training stops being effective. Symptom: loss plateaus or oscillates in late training. Mitigation: healing (switch to FP8) before crossing this threshold; monitor gradient-noise ratio.
  * **Outlier sensitivity**. Some layers (particularly attention) develop large activation outliers that overwhelm block-level scaling. Symptom: spiky loss curves with occasional explosions. Mitigation: ensure RHT is applied; consider keeping attention outputs in FP8.
  * **Forward-backward mismatch**. Without two-dimensional quantization, the same tensor has different representations in forward and backward, causing gradient inconsistency. Symptom: slow convergence, accuracy plateaus below FP8 baseline. Mitigation: ensure 2D quantization is enabled in your training stack.
  * **Optimizer state corruption**. If Adam moments are quantized to FP4 (mistakenly or aggressively), they accumulate noise across millions of steps. Symptom: training instability that emerges after many steps, not during early training. Mitigation: keep optimizer states in FP32 always; only the GEMM operands are FP4.
  * **Embedding collapse**. If embedding tables are FP4, rare tokens have too few representable values. Symptom: rare-token perplexity poor; tail of vocabulary degrades. Mitigation: embeddings in FP8/BF16 (selective high precision).
  * **Hardware-software mismatch**. Not every framework version supports NVFP4 well. Symptom: training works on B200 with one PyTorch version, fails or performs poorly with another. Mitigation: use Transformer Engine ≥ 1.x with documented Blackwell support; cross-check Megatron-Core / DeepSpeed versions.

The general defense: **monitor more carefully than for FP8 training**. Per-layer activation statistics, gradient-noise ratios, and per-iteration loss should all be tracked. FP4 training has less margin than FP8; problems compound faster.

## The April 2026 production stack

Putting M14 + M41 together — what a frontier-class training pipeline looks like as of April 2026:

  1. **Hardware** : GB200 NVL72 or GB300 NVL72 racks (72 GPUs in a single NVLink 5 domain). For smaller training, HGX B200 / HGX B300 8-GPU systems.
  2. **Numerics** : NVFP4 for all linear layer GEMMs (QKV projections, attention output, MLP up/down). FP8 for attention BMM inputs (the matmul inside attention itself). FP8 or BF16 for embedding, output head, normalization, optimizer state, loss computation.
  3. **Recipe** : Random Hadamard transform before quantization; split rounding (SR backward, RtN forward); two-dimensional quantization for forward-backward consistency; selective high-precision layers; healing in FP8 for final ~5% of iterations.
  4. **Software stack** : NVIDIA Transformer Engine for the precision-aware operators; Megatron-Core for distributed training (pipeline + tensor + sequence parallelism); CUTLASS 3.8+ underlying the GEMMs; PyTorch 2.x as the framework.
  5. **Distributed training** : tensor parallelism within rack (NVLink 5 domain — high bandwidth); pipeline parallelism across racks; sequence parallelism for long contexts; FSDP/ZeRO-3 if needed.
  6. **Monitoring** : MLPerf-style metrics — time-to-train to target loss, FLOPs achieved vs theoretical peak, communication efficiency. Plus precision-specific: gradient-noise ratio, outlier statistics per layer, FP4 vs FP8 healing-phase overlap.

This stack is concretely available — Megatron-Core + Transformer Engine + Blackwell hardware ships as a coherent package. The published recipes are reproducible; the MLPerf v5.1 results are independently verified. _FP4 training is no longer experimental in 2026; it's the production default for new pretraining runs on Blackwell hardware._

#### Q&A; — About Blackwell and NVFP4 training **Q:** When should I NOT use FP4 training? **A:** Three cases. (1) **Models below ~1B parameters** : at small scales, the engineering overhead of FP4 (recipe complexity, monitoring) outweighs the speedup. FP8 is mature enough and the differences are small. (2) **Highly noise-sensitive tasks** : some specialized training (e.g., scientific computing models, certain RL settings with very small effective gradients) hits the √3 threshold from FP4 All the Way faster than general LLM training. Use FP8 if you see plateauing that doesn't respond to healing. (3) **You don't have Blackwell hardware** : FP4 acceleration is Blackwell-native; on Hopper it's emulated and slower than FP8. The hardware-software pairing matters. _For new LLM pretraining in 2026 on Blackwell, FP4 is the default; for fine-tuning, FP8 is often sufficient and simpler._ **Q:** How is NVFP4 different from quantization-aware training (QAT) for inference? **A:** Different problem entirely. QAT trains a model with simulated quantization (forward pass uses quantized weights, backward uses higher precision) so the final weights are robust to quantization at inference. The training itself runs in BF16 or FP8 — the quantization is "for the model's benefit at deployment time." NVFP4 training is the opposite: _the actual training compute happens in FP4_. Forward and backward GEMMs both use FP4; the goal is faster training, not better post-training quantization. The two are complementary: you might train in NVFP4 for speed, then do QAT in NVFP4 for further inference robustness, then deploy in NVFP4 for inference. As of April 2026, the lines are blurring — natively-FP4-trained models are already FP4-friendly at inference, reducing the need for separate QAT. **Q:** Why does the block size of 16 specifically matter? **A:** Empirical and analytical reasons converge. The FP4 All the Way ablation (May 2025) tested block sizes from 4 to 64; below 16 elements, accuracy gains from smaller blocks plateau (the per-block scale captures most local variation already at 16); above 32 (MXFP4's choice), important local information is lost — block-level outliers pull the scale toward themselves and hurt the other 31 values. 16 is the sweet spot. The hardware design (Blackwell Tensor Cores) is built around 16-value blocks for NVFP4 specifically; supporting smaller blocks would multiply scaling-factor storage (8 bits per 16 values is reasonable; 8 bits per 4 values would dominate). _The number 16 is a hardware-software co-design optimum, not arbitrary._ **Q:** What's the relationship between Hadamard transforms and the rotation-based jailbreak / interpretability literature? **A:** Same mathematical machinery, very different applications. Hadamard transforms are structured orthogonal transforms with the property that they spread information across all dimensions. In NVFP4 training: spread outliers across positions to avoid block-scale dominance. In interpretability (M36): rotate activation space to align with sparse features (random rotations are a baseline for SAE training). In some jailbreak research: random rotations of the input embeddings to find adversarial directions. The connection is that orthogonal transforms preserve dot products and norms — they don't change the "meaning" of data, just its representation. _Hadamard transforms specifically are popular because they're cheap (O(n log n) and structured), invertible, and have provable variance-bounding properties._ **Q:** If FP4 is so much faster, why isn't every training run using it? **A:** Three frictions. (1) **Engineering investment** : the recipe (RHT, SR, 2D quant, selective precision, healing) requires careful integration. Most existing training stacks default to FP8 or BF16 because that's what's been mature longer. The migration cost is real. (2) **Hardware availability** : Blackwell GPUs are new and expensive; the installed base of Hopper is much larger. Most teams still train on H100/H200 where FP4 doesn't help. (3) **Risk tolerance** : pretraining a frontier model is a $10M-100M investment; teams are conservative about adopting numerics changes that could destabilize the run. NVIDIA's MLPerf v5.1 sweep (Nov 2025) and the open recipes (Sep 2025-Mar 2026) substantially derisked this; production adoption is accelerating through 2026. _By Q4 2026, expect FP4 to be the default for new Blackwell-based training runs; legacy Hopper continues with FP8._ **Q:** What's the most practical path to learn NVFP4 training hands-on without GB200 access? **A:** Three options. (1) **Cloud GPU rental** : B200 instances available on Lambda, RunPod, CoreWeave, Modal, Lambda Labs, and others. ~$3-6/hour as of April 2026; affordable for learning experiments. (2) **Open-source NVFP4 training repos** : NVIDIA's Megatron-Core has Blackwell-targeted examples; Hugging Face's TRL has support; community projects like the FP4-All-the-Way repo and Quartet are reproducible. Run on rented B200s for a few hours. (3) **Read the recipes** : NVIDIA's NVFP4 pretraining paper (Sep 2025, refined Mar 2026), FP4 All the Way (May 2025), Quartet II (2026), the CUTLASS Blackwell tutorials (Colfax Research) — together cover the full landscape. Reading + small B200-rental experiments gets you working knowledge in 1-2 weeks; production deployment is a longer commitment but builds on the same foundation. _The barrier to entry is lower than for previous numerics shifts (FP16, FP8); the recipes are publicly documented and the hardware is rentable._

## Code Magnets: implement the NVFP4 quantization step

You're writing the per-tensor NVFP4 quantization step with RHT and split rounding. Three magnets are wrong choices.

Arrange the magnets to compute NVFP4 quantization for a GEMM operand.

def nvfp4_quantize(tensor, hadamard_seed, is_backward): H = random_hadamard_matrix(tensor.shape[-1], seed=hadamard_seed) transformed = tensor @ H transformed = tensor tensor_scale = transformed.abs().amax() / FP4_MAX_VALUE blocks = transformed.reshape(-1, 16) block_scales = blocks.abs().amax(dim=-1, keepdim=True) / FP4_MAX_VALUE rounding = stochastic_round if is_backward else round_to_nearest rounding = round_to_nearest fp4_values = rounding(blocks / (block_scales * tensor_scale)) fp4_values = rounding(blocks / block_scales) return fp4_values, block_scales, tensor_scale, H

show solution
    
    
    def nvfp4_quantize(tensor, hadamard_seed, is_backward):
        H = random_hadamard_matrix(tensor.shape[-1], seed=hadamard_seed)
        transformed = tensor @ H
        tensor_scale = transformed.abs().amax() / FP4_MAX_VALUE
        blocks = transformed.reshape(-1, 16)
        block_scales = blocks.abs().amax(dim=-1, keepdim=True) / FP4_MAX_VALUE
        rounding = stochastic_round if is_backward else round_to_nearest
        fp4_values = rounding(blocks / (block_scales * tensor_scale))
        return fp4_values, block_scales, tensor_scale, H

The traps:

  * `transformed = tensor`: skips the Hadamard transform entirely. Without RHT, block-level outliers dominate the per-block scale; the other 15 values in each block lose precision; training quality degrades substantially. RHT is the most important single technique in the FP4 recipe — every published successful training run includes it. Skipping it means accepting MXFP4-level quality at NVFP4 cost; you get the worst of both formats.
  * `rounding = round_to_nearest` (always RtN, no split rounding): uses RtN for both forward and backward passes. Forward in RtN is correct; backward needs SR for unbiased gradient estimation. Pure-RtN backward produces biased gradient estimates; the bias compounds across millions of training steps and pulls the model toward systematic errors. The published findings (FP4 All the Way and others) are clear: split rounding (SR backward, RtN forward) substantially beats pure-RtN. The `is_backward` parameter exists precisely to enable this; ignoring it defeats the purpose.
  * `fp4_values = rounding(blocks / block_scales)`: divides only by the block scale, not by both block scale AND tensor scale. The two-level scaling means each value's effective magnitude is divided by both: block-level for local variation, tensor-level for global range. Dividing by only the block scale leaves the tensor-level dynamic range unaccounted for; values either underflow or saturate FP4's narrow range. Both scales must be applied in quantization (and reconstructed when computing the GEMM result).

The pattern: **random Hadamard transform → compute tensor scale → reshape into 16-value blocks → compute per-block scales → choose rounding mode based on forward/backward → quantize using BOTH scales**. Each step has a specific role; the most common production bugs are skipping RHT (silent quality loss) and using a single scale level (silent precision loss).

## Who does what?

Match each Blackwell/NVFP4 concept to its real role.

Concept

Real role

NVFP4

A. 4-bit format with 16-value blocks, FP8 (E4M3) block scales, FP32 tensor scale; near-FP8 accuracy.

Random Hadamard transform

B. Disperses block-level outliers before quantization; bounds variance of stochastic rounding.

Split rounding

C. Stochastic rounding in backward pass for unbiased gradients; RtN in forward for stability.

2D quantization

D. Quantize tensor along both row and column dimensions for forward-backward consistency.

Healing

E. Last 5% of training in FP8 to recover accuracy that pure FP4 would leave behind.

Tensor Memory (TMEM)

F. Blackwell on-chip memory tier between SMEM and registers; eliminates register pressure for MMA.

CTA pair / 2-SM UMMA

G. Two SMs share input operands via the 5th-gen Tensor Core; halves operand memory bandwidth.

show solution

**NVFP4** → A  
**Random Hadamard transform** → B  
**Split rounding** → C  
**2D quantization** → D  
**Healing** → E  
**Tensor Memory (TMEM)** → F  
**CTA pair / 2-SM UMMA** → G 

The mental shortcut: _NVFP4 packs values, Hadamard disperses outliers, split rounding handles fwd/bwd asymmetry, 2D quant handles tensor orientation, healing recovers final accuracy, TMEM holds accumulators, CTA pairs share operands_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** Your team is training a 30B model on 32× B200 GPUs. Initial NVFP4 training works for 2K steps, then loss starts spiking, then diverges by step 4K. Walk through the diagnosis.

show answer

FP4 training divergence has a few characteristic causes. Walk through them in order:

(1) **Verify Hadamard transform is enabled**. The most common cause of FP4 instability is missing or misapplied RHT. Check the linear layer config; ensure RHT is on for both forward and backward passes. Without RHT, block-level outliers cause occasional gradient spikes that diverge the policy. _Easiest first thing to check._

(2) **Check stochastic rounding configuration**. Verify backward passes use SR, not RtN. Pure-RtN backward produces biased gradients that compound; biased compounding doesn't show up immediately, but emerges as instability in late training (consistent with your 2K-step onset). _The split rounding strategy is mandatory._

(3) **Check 2D quantization**. If the same tensor has different representations in forward and backward (because only one dimension is quantized), gradient estimates are inconsistent. Symptom: slow convergence followed by instability. Verify the linear layer uses 2D quantization (most production stacks do by default in 2026; older versions might not).

(4) **Examine which layers are in NVFP4**. If embedding or LM head is in NVFP4, that's a problem — they need higher precision (FP8 or BF16). Check the precision config; ensure selective high-precision layers are configured.

(5) **Plot per-layer gradient norms**. If specific layers (commonly attention output projection or specific MLP layers) have spiky gradient norms, those layers may need to stay in FP8 even if RHT is on. NVIDIA's MLPerf submissions kept "attention BMM inputs" specifically in FP8 — there's a reason.

(6) **Check Transformer Engine / Megatron-Core version**. Older versions had bugs in NVFP4 paths. Update to current versions; check the release notes for FP4 training stability fixes.

(7) **If all else fails, switch the problematic layers to FP8**. The heterogeneous-precision approach (most layers NVFP4, some FP8) is sometimes necessary. Costs some throughput but recovers stability. Often a small number of layers are the source of all instability.

The general lesson: _FP4 training has less margin than FP8_. Each technique in the recipe matters; missing one often leads to divergence. The published recipes are recipes for a reason — they encode hard-won lessons.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why does the per-tensor FP32 scale matter when you already have per-block FP8 (E4M3) scales? Couldn't you just use larger block scales?

show answer

Two reasons that compound:

(1) **FP8 (E4M3) has limited dynamic range**. E4M3 has 4 exponent bits — its range is roughly 2⁻⁶ to 2⁸ ≈ 256 in magnitude. A weight tensor in a transformer can have values ranging from 10⁻⁵ (small initialized weights) to 10¹ (after some training). That's 6 orders of magnitude. If the per-block scale itself is in FP8, the block scale can't cover both extremes — some blocks would have scales that overflow E4M3.

(2) **Globally normalizing first**. The per-tensor FP32 scale captures the tensor's global magnitude (its max absolute value, scaled to fit in the format). Once you've divided by the per-tensor scale, all values are roughly within 0-1 (or so). The per-block FP8 scale then captures the local variation within each 16-value block — and FP8 has enough range for that.

The two-level decomposition matches the actual structure of weight tensors: _large global magnitude (handled by FP32 scalar) plus moderate local variation within blocks (handled by FP8 per-block scales)_. Without the FP32 layer, you'd need much wider per-block scales (e.g., FP16 instead of FP8), doubling the scale storage and reducing the format's compression benefit.

The MXFP4 design uses E8M0 single-level scaling — power-of-two-only block scales without a tensor-level FP32. This works but loses precision because E8M0 is coarse (only powers of 2 are representable as scale factors). NVFP4's two-level approach with E4M3 (fine-grained, fractional) plus FP32 (global) outperforms it empirically by ~5-15% on training quality at the same compute cost.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** Walk through what changes when porting a Hopper FP8 GEMM kernel to Blackwell NVFP4 in CUTLASS. What new concepts does the kernel author face?

show answer

Six substantive changes to the kernel structure:

(1) **MMA instruction**. Hopper used WGMMA (Warp-Group MMA, `wgmma.mma_async`) where a warp group of 4 warps cooperatively executed an MMA. Blackwell uses UMMA (Universal MMA, `tcgen05.mma`) launched from a single thread. The kernel structure changes from "issue WGMMA from warp group" to "elect one thread, issue UMMA from it." Synchronization is different.

(2) **Accumulator location**. Hopper accumulators sat in registers; tile sizes were limited by register pressure. Blackwell accumulators sit in TMEM; tile sizes can be much larger (256×256 instead of 128×128 typically). The kernel allocates TMEM for accumulators using `tmem_allocator_sm100.hpp`; no register allocation needed for accumulators.

(3) **2-SM UMMA / CTA pairs**. Blackwell offers 2-SM UMMA where two CTAs share input operands. The kernel can use 1-SM (simpler, like Hopper) or 2-SM (faster, requires CTA-pair coordination). Most production GEMMs use 2-SM. The TMA loads need to use multicast for the shared operand.

(4) **Block-scaled FP4 GEMMs**. NVFP4 has hardware-supported block scaling, but the SMEM data layout must match the format's expected shape. The TMA tensor map type changes (`CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B` for sub-byte types). Scale factors are loaded separately via additional TMA atoms (e.g., `sfa_op` and `sfb_op` in the CUTLASS examples).

(5) **Tile size constraints**. For block-scaled FP4 GEMMs, the M dimension must be 128 for 1-CTA MMA or 128/256 for 2-CTA MMA. The K dimension has its own constraints based on block size (16 for NVFP4). Tile shape selection is more constrained than on Hopper.

(6) **Synchronization**. Hopper used `mbarrier` for fine-grained sync between TMA loads and WGMMA computes; Blackwell extends this with new primitives (different multicast TMA atoms, TMEM allocator synchronization). The CUTLASS Blackwell examples use these directly.

The general lesson: **Hopper-to-Blackwell isn't a drop-in port**. Kernel authors face a substantial rewrite. The CUTLASS Blackwell tutorials (Colfax Research) walk through the changes; CUDA Tile / CuTile (NVIDIA, late 2025) abstracts much of this for kernel authors who don't want to deal with the low-level details. Most production training engineers don't write CUTLASS kernels directly — they use Megatron-Core / Transformer Engine, which abstract the kernels away.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** A team is training a 70B reasoning model with M39's distributed RLHF stack (REINFORCE++ baseline, Hybrid Engine, OpenRLHF) on 32× B200. Should they use NVFP4? Walk through the analysis.

show answer

RLHF on Blackwell with NVFP4 is interesting because of the asymmetric workload. Walk through the considerations:

(1) **Rollout phase (~80% of compute, M39's framing)** : vLLM-served inference. NVFP4 inference is well-supported on Blackwell — the published numbers ($0.02 per million tokens for GPT-OSS-120B) demonstrate substantial inference speedups in FP4. **Yes, use NVFP4 for rollout**. Memory savings (1.8× over FP8) also matter for KV cache, enabling larger batch sizes.

(2) **Training step (~20% of compute)** : Actor backward pass. This is where FP4 training recipe applies. **Probably yes, but check stability carefully**. RL training has different gradient characteristics than pretraining — sparse rewards, off-policy correction, KL terms. The RLHF gradient signal can be smaller than pretraining gradients, potentially hitting the √3 threshold from FP4 All the Way faster. Test on a small run first; monitor gradient-noise ratio.

(3) **Reference model forward** : pure inference, used to compute KL term. Same as rollout; **NVFP4 inference is the right choice**.

(4) **Reward model** : pure inference, used to score rollouts. Same as rollout; **NVFP4 inference**.

(5) **Critical paths in RLHF** : KL penalty computation, advantage computation, log-prob computation. These are mostly elementwise operations done in FP32 anyway; precision changes don't help.

(6) **Hybrid Engine implications** : vLLM sleep/wake transitions handle weight conversion. The trainer's NVFP4-quantized weights need to be converted to the inference engine's expected format on each wake. This conversion is fast on Blackwell (native hardware support) but is an extra step the framework must handle. Verify your OpenRLHF / veRL version handles NVFP4 cleanly in Hybrid Engine mode.

(7) **Practical recommendation** : **start with NVFP4 inference (rollout, reference, reward) + FP8 training**. This captures most of the speedup (rollout dominates) with minimal stability risk. Once that's stable, optionally migrate training to NVFP4 with healing. Total expected speedup: 2-3× over the same RLHF pipeline on Hopper FP8.

The general lesson: **NVFP4 wins decisively for inference workloads** (which RLHF rollout largely is) and is reasonable but slightly riskier for training. RLHF specifically benefits asymmetrically because rollout is so dominant.

### What just happened?

  * The hardware moved twice since M14: **Blackwell (B200, GB200 NVL72)** introduced 5th-gen Tensor Cores with native FP4, 192GB HBM3e, NVLink 5, Tensor Memory (TMEM), CTA-pair MMA. **Blackwell Ultra (B300, GB300 NVL72)** refreshed with 270GB memory, 1.5× NVFP4 throughput.
  * **NVFP4 format** : 4-bit values (E2M1) in 16-value micro-blocks, each with FP8 (E4M3) block scale, plus per-tensor FP32 scale. Two-level scaling beats MXFP4's single-level by enabling fine-grained dynamic range adaptation.
  * **The training recipe** (NVIDIA NVFP4 paper Sep 2025-Mar 2026; FP4 All the Way May 2025; Quartet II 2026): five techniques compose to FP8-equivalent quality. **Random Hadamard transforms (RHT)** disperse block-level outliers; **stochastic rounding (SR)** for unbiased gradient estimation; **split rounding** (SR backward, RtN forward); **two-dimensional quantization** for forward-backward consistency; **selective high-precision layers** (embedding, output head, normalization, optimizer state stay FP8/BF16); **healing** (last 5% in FP8).
  * **MLPerf Training v5.1 (Nov 2025)** : NVIDIA swept all 7 benchmarks with NVFP4. **Llama 3.1 405B in 10 minutes on 5,120 Blackwell GPUs**. 3.2× over Hopper FP8 at same count; 1.4× over FP8 on same Blackwell rack.
  * Validated at scale: NVIDIA 12B/10T tokens matching FP8; FP4 All the Way 7B/200T tokens matching BF16; Quartet II / MS-EDEN improving stochasticity placement.
  * **Block size of 16** is empirically optimal — below 16 diminishing returns; above 32 (MXFP4) loses local information.
  * **Theoretical threshold** (FP4 All the Way): training becomes ineffective when gradient norm falls below ~√3 × quantization noise. Healing addresses this for late training.
  * **Tensor Memory (TMEM)** : new on-chip memory tier (192KB/SM) for MMA accumulators. Replaces register-based accumulation; enables larger tile sizes; 7× GEMM throughput gain over Hopper.
  * **CTA pairs / 2-SM UMMA** : two adjacent SMs share input operands via the 5th-gen Tensor Core; halves operand memory bandwidth; `tcgen05.mma` instruction. Single-thread MMA launch enables fine-grained warp specialization.
  * **CUTLASS 3.8+** : Blackwell support — TMEM, CTA pairs, 2-SM UMMA, TMA multicast, native NVFP4/MXFP4 GEMMs with hardware block-scaling. **CuTile (late 2025)** : Python tile-centric DSL abstracting warps/registers/SMEM.
  * **Economics** : GB200 NVL72 ~2× perf/dollar of H100 on Llama 3.1 405B; ~3.2× per-GPU speedup vs Hopper FP8; inference at $0.02 per million tokens (4.5× cheaper than Hopper). Memory: 50% reduction per parameter (FP8 → NVFP4).
  * **Failure modes specific to NVFP4** : outlier sensitivity (without RHT), forward-backward mismatch (without 2D quant), √3 threshold late-training (mitigated by healing), embedding collapse (mitigated by selective precision), optimizer state corruption (Adam stays FP32 always).
  * **Rubin (2026-2027 lookahead)** : 35 PFLOPS NVFP4 training, 50 PFLOPS NVFP4 inference — 3.5×/5× over Blackwell. Recipe likely transfers with minor adjustments.
  * The reflex: when planning a new pretraining run on Blackwell hardware in 2026, **default to NVFP4 with the standard recipe** ; for fine-tuning or smaller models, FP8 is often simpler; for Hopper, stay with FP8.

Module 42 (next, if Part XI continues) would tackle **scaling laws & compute economics** — Chinchilla and successors, the train-time vs test-time compute tradeoff (M34 territory generalized), FLOP-to-quality curves for different architectures, the economics of choosing where to spend FLOPs. Foundational; threads through every previous module.
