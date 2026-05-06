# Module 26 — Quantization: int8, int4, GPTQ, AWQ & fp8

# _Quantization:_ int8, int4, GPTQ, AWQ & fp8

_Part VIII · Module 26_

— how to run a 70B model in 35 GB instead of 140 GB, why int4 matmul kernels are hand-written, and the calibration recipes that make low-bit weights actually work

\--- 

Quantization is two things at once. _Mathematically_ , it's the most boring optimization in deep learning: round each fp16 weight to one of 16 values (int4), pay a tiny accuracy cost, ship the model. _Practically_ , it's the optimization that decides whether a 70B model runs on consumer hardware or doesn't. A 70B-param model in bf16 is 140 GB of weights — won't fit on any single GPU. Quantized to int4: 35 GB. Fits on a single H100, runs at high throughput. Without quantization, the entire LLM-on-laptop / cheap-inference / open-weight-models story doesn't happen.

This module sits in Part VIII because _efficient int4/int8 matmul is hand-written kernel territory_. cuBLAS doesn't ship an int4-weight × bf16-activation matmul (because nothing standard does). The kernels live in `torchao`, `bitsandbytes`, `marlin`, `vllm`, and they're written in CUDA or Triton using everything from M22-M25.

> **★ KEY IDEA**  
>  Quantization stores weights in fewer bits than they were trained in: **int8** (8 bits, 2× compression), **int4** (4 bits, 4× compression), with finer granularities for some schemes. _Memory-bound_ inference (which is most LLM serving) gets near-linear speedup from this — fewer bytes per param means faster HBM reads. The challenge: low-bit representations lose precision, and naively rounding kills quality. **GPTQ** and **AWQ** are calibration algorithms that minimize the quality cost. **fp8** is a different game — it preserves dynamic range but reduces precision; used for training on H100+. Pick the scheme based on whether you're doing training (fp8), inference (int8/int4), or both (mixed). 

## One new face

Qb

Quantizer

"I shrink your weights from 16 bits to 4 by rounding to one of 16 values per group, with a scale that re-expands them back."

Each group of 64 or 128 weights gets a single fp16 _scale_. To dequantize: `w_real = scale * (q − zero_point)`. To quantize: `q = round(w_real / scale + zero_point)`. The art is picking the scale to minimize information loss — and that's what GPTQ and AWQ are about. _I'm cheap to apply but expensive to apply well._

## The math: scale, zero-point, granularity

The basic affine quantization formula:
    
    
    q = round(w / s + z)               # quantize: real → integer
    w_hat = s * (q − z)                # dequantize: integer → real (with rounding error)

Where `s` is the _scale_ (a positive fp16 number) and `z` is the _zero-point_ (an integer in the quantized range). For symmetric quantization (no zero-point shift), `z = 0`; for asymmetric, `z` is chosen to map the real-valued zero exactly to an integer.

The bit-width sets the integer range:

Common quantization bit widths and ranges Bits| Symmetric range| Asymmetric range| Compression vs fp16  
---|---|---|---  
int8| [−127, 127]| [0, 255]| 2×  
int4| [−7, 7]| [0, 15]| 4×  
int3| [−3, 3]| [0, 7]| ~5.3×  
int2| [−1, 1]| [0, 3]| 8×  
  
Sub-int4 (int3, int2, int1.58 / ternary) is mostly research territory — quality drops fast below 4 bits. **int4 weight-only is the sweet spot for inference** in 2024-2026: 4× memory savings, near-lossless quality with good calibration.

### Granularity: where the scales live

The single biggest design choice in quantization. How much of the weight tensor shares one scale?

Quantization granularity: how much of the tensor shares a scale? ① Per-tensor — 1 scale per weight matrix (cheapest, lowest quality) one scale s for all 4096 × 4096 weights s ← one fp16 number Storage overhead: ~0%. Quality: poor for outlier-heavy distributions (most LLMs). ② Per-channel — 1 scale per output channel (per row) 4096 channels × one scale each ← 4096 fp16 scales Storage overhead: ~0.05%. Quality: much better — scales adapt to per-channel range. ③ Per-group (group_size=128) — 1 scale per 128 weights along input dim 8 scales for 1 row of 1024 weights (in our int4 quant scheme) Storage overhead: ~0.4% (1 fp16 per 128 int4 = ~10% of int4 storage). Quality: best.

Three things to internalize:

  1. **Per-tensor** : one scale for the whole matrix. Cheapest in storage, easiest to implement, but quality is poor when weights have outliers (which they do in modern LLMs — see AWQ below).
  2. **Per-channel** : one scale per row (output channel). Much better quality at trivial storage cost. _The default for int8 weight-only._
  3. **Per-group** with group size 64 or 128: one scale per group of 128 weights along the input dim. Best quality, used in GPTQ/AWQ int4. The overhead is 1 fp16 per 128 int4 weights — about 6% of total storage but huge quality gain.

The scheme you'll see most for int4 weight-only inference: **group-size 128, asymmetric, per-group scale + zero-point**. Storage per param: 4 bits + (16 + 4) / 128 ≈ 4.16 bits effective.

## The PTQ algorithms: how to pick the scales well

The math above tells you _how to apply_ a scale. Picking _which_ scale gives the best quality is the algorithm question. Three approaches, in order of sophistication:

### 1\. MinMax / max-abs (the naive baseline)

For each group, compute `s = max(|w|) / 7` (for int4 symmetric). Done. Round all weights to the nearest int4 value. Easy to implement, 30 seconds to apply to a 7B model.

Why it's bad: a single outlier weight blows up the scale, and now the other 127 weights in the group lose precision. LLMs have long-tailed weight distributions — outliers are common.

Quality: usable for int8 (the bit budget is forgiving). Awful for int4 — typical perplexity rise of 50%+ vs fp16 baseline.

### 2\. GPTQ (Hessian-aware reconstruction)

The insight: when you quantize a weight matrix W to W_q, the matrix-multiply output changes by ΔY = X(W − W_q). What you actually care about isn't `||W − W_q||` but `||X(W − W_q)||` — the layer's output error. GPTQ minimizes the latter.

Concretely: for each layer, GPTQ computes H = X.T @ X (the input Gram matrix, also called the Hessian of the squared-error objective). Then it quantizes weights one column at a time, in an order chosen to minimize the cumulative output error, propagating the rounding error from each quantized column to the still-unquantized columns via a closed-form update derived from H.
    
    
    # Simplified GPTQ pseudocode for one layer (W: weight matrix, X: calibration activations)
    H = X.T @ X                        # input Gram matrix — captures which weights matter how much
    H = H + dampening * I              # numerical stability (Hessian damping)
    H_inv = cholesky_inverse(H)        # invert once
    
    for col in chosen_order(H_inv):    # typically diagonal-ascending
        w_col = W[:, col]
        w_q = quantize_to_int4(w_col)  # the rounding step
        error = w_col − w_q
        # Propagate the error to remaining columns via H_inv
        W[:, remaining] −= error * H_inv[col, remaining] / H_inv[col, col]
        W[:, col] = w_q

The key term is the error-propagation update. Quantizing column k introduces error; GPTQ adjusts the remaining columns to compensate, minimizing the cumulative impact on layer output. This is essentially OBQ (optimal brain quantization) applied per layer with calibration data.

Cost: ~100-300 calibration sequences (a few thousand tokens), one Cholesky per layer. Total: minutes to an hour for a 7B model. Quality: typically 1-3% perplexity rise vs fp16 at int4. **The standard for int4 quantization since 2022.**

### 3\. AWQ (Activation-aware weight quantization)

The complementary insight: not all weights matter equally. Some output columns of a weight matrix carry "salient" features — large activations flow through them. Quantizing those columns more carefully (or not at all) preserves quality.

AWQ's algorithm:

  1. Run calibration data through the model. For each layer, observe the per-channel activation magnitudes `a = max|X[:, i]|`.
  2. For each output channel, compute a per-channel scaling factor `α` proportional to `a^β` for some β (typically ≈ 0.5).
  3. Apply `W = W / α` and `X = X * α` — the multiplication output is unchanged, but the weights to be quantized are now smaller in their salient dimensions, so quantization rounds them more precisely.
  4. Quantize the rescaled W with standard group-wise int4.

The trick: _at inference time_ , the α scaling is folded back so the kernel doesn't see it. The runtime kernel is plain int4-weight × bf16-activation matmul — no overhead from AWQ's calibration step.

Cost: similar to GPTQ — calibration data, an optimization step (search for best β). Quality: typically matches or slightly beats GPTQ at int4. **The other major int4 PTQ algorithm — try both for your model and pick the winner.**

### Comparison

The three PTQ algorithms — when to use which Algorithm| Calibration data needed| Calibration time| Quality (int4)| Notes  
---|---|---|---|---  
MinMax / max-abs| None| Seconds| Poor| Use for int8 only; for int4 the quality is unacceptable  
GPTQ| ~128 sequences| Minutes-hours| Good| Hessian-aware; the long-standing standard  
AWQ| ~128 sequences| Minutes-hours| Good (often best)| Activation-aware; preserves salient channels  
SmoothQuant (W8A8)| ~512 sequences| Minutes| —| Different goal: int8 weights + int8 activations  
QAT (Quantization-Aware Training)| Full training set| Days-weeks| Best possible| Train with simulated quantization; expensive but optimal  
  
Practical recipe for inference-time quantization: **start with GPTQ-int4 or AWQ-int4. Try both, evaluate on a representative eval set, pick the winner.** Both are well-supported in PyTorch (`torchao`) and external libraries (`autoawq`, `auto-gptq`).

## The kernel: int4 weight-only matmul

Now the kernel. This is why M26 is in Part VIII. Standard cuBLAS doesn't have an int4-weight × bf16-activation matmul kernel — it's a niche pattern that only matters for inference with quantized models. The kernel has to be written by hand.

The pattern: **weights are int4 in HBM, activations are bf16, output is bf16. Inside the kernel, weights get dequantized to bf16 in registers right before the matmul.**
    
    
    @triton.jit
    def int4_weight_only_matmul_kernel(
        a_ptr,            # bf16 activations [M, K]
        b_q_ptr,          # int4 weights packed [K // 8, N]  (8 int4s per int32)
        scales_ptr,       # fp16 scales       [K // group_size, N]
        zeros_ptr,        # int4 zero-points  [K // group_size, N // 8]
        c_ptr,            # bf16 output       [M, N]
        M, N, K,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
    
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
    
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
        for k in range(0, K, BLOCK_K):
            # 1. Load activations (bf16, normal pattern)
            a = tl.load(a_ptr + offs_m[:, None] * K + (offs_k[None, :] + k))
    
            # 2. Load packed int4 weights — each int32 holds 8 int4 values
            #    Decode: extract 4-bit nibbles, mask, sign-extend or zero-extend.
            b_packed = tl.load(b_q_ptr + ((offs_k[None, :] + k) // 8) * N + offs_n[None, :])
            #  Shift by (offs_k % 8) * 4 bits, mask 4 bits, subtract zero-point
            shift = ((offs_k[None, :] + k) % 8) * 4
            b_int4 = (b_packed >> shift) & 0xF          # [BLOCK_K, BLOCK_N]
    
            # 3. Load the per-group scale and zero-point for this K block
            group_idx = (offs_k[None, :] + k) // GROUP_SIZE
            scale = tl.load(scales_ptr + group_idx * N + offs_n[None, :])
            zero  = tl.load(zeros_ptr  + group_idx * N + offs_n[None, :])
    
            # 4. Dequantize: w = scale * (q - zero), entirely in registers
            b = (b_int4.to(tl.float32) - zero.to(tl.float32)) * scale.to(tl.float32)
    
            # 5. The actual matmul — tensor cores, bf16 inputs (cast b on the fly)
            acc += tl.dot(a, b.to(tl.bfloat16))
    
        # Store output in bf16
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16))

The kernel is structurally identical to a regular matmul (the M24 pattern) _plus a dequant step_. Steps 2-4 are the new bit:

  1. **Pack/unpack int4 from int32**. Eight int4 values fit in one int32. The kernel does bit-shifts and masking to extract the 4-bit nibble for each weight.
  2. **Load the per-group scale and zero-point**. These are small — one fp16 + one int4 per 128 weights — and the per-group lookup is fast.
  3. **Dequantize in registers**. `w = scale * (q - zero)` is an elementwise op that runs entirely in fast on-chip memory. The dequantized bf16 weights live in registers just long enough to feed `tl.dot`.
  4. **Standard tensor-core matmul**. Once the weights are in bf16 in registers, this is the same `tl.dot` from M24's matmul.

int4-weight × bf16-activation matmul: dequant in registers, never in HBM HBM (slow, 3 TB/s): int4 weights 4 bits / param 35 GB for 70B scales fp16 1 per 128 params activations bf16 output bf16 On-chip (registers + shared, ~10 PB/s): Triton kernel inner loop (per K tile) 1\. extract 4-bit nibbles from packed int32 → b_int4 [BLOCK_K, BLOCK_N] 2\. dequantize: b = scale * (b_int4 - zero) → bf16 in registers 3\. tl.dot(a, b) → acc += matmul on tensor cores key win: 4× less HBM bandwidth for weights, same compute memory-bound inference → 2-3× wall-clock speedup over fp16

Why this is fast: **most LLM inference is memory-bound** , especially the autoregressive decode phase where you're processing one token at a time and the GPU is waiting for the next layer's weights to load from HBM. Cutting the weight bytes by 4× cuts the wait time by ~4× for the memory-bound regions. The matmul compute itself doesn't change — it's still bf16 × bf16 in tensor cores — but you spend much less time waiting for weights to arrive.

For the prefill phase (processing a long prompt) which is more compute-bound, the speedup is smaller — but still positive because of better cache pressure with smaller weight footprint.

### Why are these kernels so finicky?

A few sources of complexity that make int4 kernels hand-written-only:

  * **Packing layouts**. Eight int4s in one int32 saves space, but the pack order matters — kernels assume specific layouts (Marlin uses one layout, GPTQ another, AWQ a third). Mixing them gives garbage.
  * **Group-size variations**. group=128 is most common, but some libraries use 64 or 32. The kernel needs to know.
  * **Symmetric vs asymmetric**. Different kernels assume different schemes; the dequantization step is slightly different.
  * **Hardware-specific MMA layouts**. The bit-extraction and dequant work has to align with the tensor-core MMA tile layout for maximum throughput. The Marlin kernel does extensive bit-twiddling to get this right on Ampere/Hopper.
  * **Per-architecture optimizations**. Hopper has cleaner async-load primitives (TMA), so the int4 kernel layout differs from Ampere's.

Practical advice: _don't write your own int4 matmul_. Use `torchao`, `marlin`, `vllm`'s GPTQ kernels, or `bitsandbytes`. They've all been tuned heavily. Read them to learn the patterns; use them in production.

## fp8: a different game

fp8 is conceptually a different beast. We met it briefly in M14: e4m3 (max ~448) and e5m2 (max ~57344). Both are 8-bit floats with hardware tensor-core support on H100+.

fp8 is for _training_ as well as inference. The pattern: forward activations and weights in e4m3 (more precision, less range — fine for forward); backward gradients in e5m2 (more range, less precision — needed because gradients can have large magnitudes during training). Per-tensor scaling (similar to GradScaler from M14, but per-tensor and managed separately for each direction) keeps values in the representable range.

Library support: `TransformerEngine` from NVIDIA, `torchao`, native PyTorch in 2.4+. The recipe:
    
    
    # With torchao (modern approach)
    from torchao.float8 import convert_to_float8_training
    
    model = build_model()
    convert_to_float8_training(model)            # in-place; replaces matmuls with fp8 versions
    # Now train normally — fp8 is handled inside the matmul kernels

Speedup vs bf16: roughly 1.3-1.7× on H100 (tensor cores are 2× faster at fp8, but everything-else stays bf16). Quality: usually within noise of bf16 with proper scaling. Used in Llama 3 405B training and most frontier-scale fp8 deployments.

fp8 inference is a separate thing — for serving, the standard recipe is e4m3 weights + e4m3 activations, with per-tensor static scales. Frameworks like vLLM and TensorRT-LLM ship fp8 inference paths.

#### Q&A; — About quantization **Q:** Why does int4 weight-only quantization help so much for inference but not much for training? **A:** Inference is dominated by weight loads — for autoregressive decoding, you're processing one token through the whole model, repeatedly fetching every weight matrix from HBM. 4× smaller weights = ~4× faster weight loads. Training is different: you have the same weights but now ALSO 4-8× of training state (gradients, Adam moments, master copy from M12). Quantizing weights doesn't help that. _For training, fp8 helps more — it speeds up the matmul itself, which dominates training throughput._ **Q:** Are quantized models worse than fp16 models in any measurable way? **A:** Slightly. Typical results: int8 weight-only is essentially indistinguishable from bf16 (perplexity within 0.1%). int4 weight-only with GPTQ/AWQ is usually within 1-3% perplexity. Below int4 (int3, int2, ternary), quality drops noticeably — used only when memory pressure dominates. The trade is highly favorable: 4× memory savings for <3% quality cost is a no-brainer for serving. **Q:** What's "outliers" mean in the quantization context? **A:** In modern LLMs, a small fraction (often < 1%) of weights or activations have magnitudes ~10× larger than the rest. When you compute a per-tensor scale via max-abs, that single outlier sets the scale and the other 99% lose precision. Outliers are the reason naive quantization fails for LLMs and the reason GPTQ/AWQ exist — both algorithms handle outliers more gracefully (GPTQ via Hessian-aware reconstruction; AWQ by scaling salient channels separately). **Q:** When should I use QAT (quantization-aware training)? **A:** Three cases. (1) When you need to push past int4 (int3, int2) and PTQ quality isn't enough. (2) When you need very specific deployment constraints (e.g., int8 weights AND int8 activations end-to-end, where activation quantization needs the model to "know" about it during training). (3) When compute is cheap and quality must be perfect. For most LLM inference: _PTQ (GPTQ or AWQ) is sufficient and 100× cheaper than QAT_. **Q:** Can I quantize just some layers and not others? **A:** Yes — and you should for sensitive layers. The lm_head (final projection to vocab) and embeddings often stay in fp16/bf16 even when the rest is int4 — they're small relative to the transformer body, and their precision matters disproportionately. Some recipes also keep the first and last transformer layers in higher precision. Tools like `torchao` let you specify per-module quantization configs. **Q:** What's "double quantization" / "QLoRA"-style? **A:** A space-saving trick: the per-group scales (which are fp16, ~0.4% overhead) can themselves be quantized to int8 with their own meta-scales. Saves a few percent of memory at trivial quality cost. Used in QLoRA fine-tuning to fit 65B models on a single 24GB GPU. The frozen-base-model is double-quantized; only the LoRA adapters train in fp16/bf16. 

## Practical recipe: what to use when

Quantization decision tree Goal| Recipe| Library  
---|---|---  
Inference: fastest, max compression| int4 weight-only (GPTQ or AWQ), group-size 128| `torchao`, `auto-gptq`, `autoawq`  
Inference: simplest, best quality| int8 weight-only, per-channel| `torchao`, `bitsandbytes`  
Inference: H100, max throughput| fp8 weights + fp8 activations| `vLLM`, `TensorRT-LLM`  
Training: H100, want bf16-equivalent quality at higher speed| fp8 (e4m3 fwd + e5m2 bwd)| `torchao.float8`, `TransformerEngine`  
Fine-tuning a quantized base model with LoRA| int4 base + bf16 adapters| `QLoRA` / `peft`  
Sub-int4 (research)| int3 / int2 / ternary; QAT recommended| research codebases  
  
The 90% case for production LLM inference: **int4 weight-only with GPTQ or AWQ, served via vLLM or similar**. The 10% case is fp8 on H100+ for very high throughput.

## Code Magnets: identify the dequant kernel structure

You're writing the inner loop of an int4-weight × bf16-activation matmul kernel. Three magnets are wrong choices.

Arrange the magnets into a correct K-tile inner loop.

a = tl.load(a_ptr + offs_m[:, None] * K + (offs_k[None, :] + k)) b_packed = tl.load(b_q_ptr + ((offs_k[None, :] + k) // 8) * N + offs_n[None, :]) b_packed = tl.load(b_q_ptr + (offs_k[None, :] + k) * N + offs_n[None, :]) shift = ((offs_k[None, :] + k) % 8) * 4 b_int4 = (b_packed >> shift) & 0xF b_int4 = b_packed & 0xF scale = tl.load(scales_ptr + ((offs_k[None, :] + k) // GROUP_SIZE) * N + offs_n[None, :]) b = (b_int4.to(tl.float32) - zero.to(tl.float32)) * scale.to(tl.float32) acc += tl.dot(a, b.to(tl.bfloat16)) acc += tl.dot(a, b_int4.to(tl.bfloat16))

show solution
    
    
    a = tl.load(a_ptr + offs_m[:, None] * K + (offs_k[None, :] + k))
    b_packed = tl.load(b_q_ptr + ((offs_k[None, :] + k) // 8) * N + offs_n[None, :])
    shift = ((offs_k[None, :] + k) % 8) * 4
    b_int4 = (b_packed >> shift) & 0xF
    scale = tl.load(scales_ptr + ((offs_k[None, :] + k) // GROUP_SIZE) * N + offs_n[None, :])
    b = (b_int4.to(tl.float32) - zero.to(tl.float32)) * scale.to(tl.float32)
    acc += tl.dot(a, b.to(tl.bfloat16))

The traps:

  * `b_packed = tl.load(b_q_ptr + (offs_k[None, :] + k) * N + offs_n[None, :])`: missing the `// 8` for the int4 packing. Each int32 holds 8 int4 values, so the K index in the packed array is K // 8.
  * `b_int4 = b_packed & 0xF`: missing the bit shift. Without shifting, you only get the lowest int4 of the int32 — same value for all 8 K positions. Bug produces garbage.
  * `acc += tl.dot(a, b_int4.to(tl.bfloat16))`: skips the dequantization (scale * (q - zero)). The int4 values themselves aren't the real weights — they need to be rescaled.

The full pattern: **load packed int32 → bit-shift to extract this position's int4 → load per-group scale + zero → dequantize via scale*(q-zero) → cast to bf16 → matmul on tensor cores**. Every step matters; skipping any of them produces silently wrong outputs.

## Who does what?

Match each quantization concept to its real role.

Concept

Real role

Per-group quantization

A. One scale + one zero-point per 128 weights along input dim — best quality at small overhead.

GPTQ

B. Hessian-aware quantization that minimizes per-layer output reconstruction error.

AWQ

C. Activation-aware: scale up salient channels before quantizing so they round more precisely.

int4 weight-only matmul

D. Inference kernel: int4 weights from HBM, dequantize in registers, matmul on tensor cores in bf16.

Outlier weights / activations

E. The reason naive quantization fails for LLMs — a few large values ruin per-tensor scales.

fp8 (e4m3 + e5m2)

F. Training-friendly low-bit floats: e4m3 forward, e5m2 backward, with per-tensor scales.

QAT (quantization-aware training)

G. Train with simulated quantization in the loop. Best quality, most expensive.

show solution

**Per-group quantization** → A  
**GPTQ** → B  
**AWQ** → C  
**int4 weight-only matmul** → D  
**Outlier weights / activations** → E  
**fp8 (e4m3 + e5m2)** → F  
**QAT** → G 

The mental shortcut: _per-group = best PTQ granularity, GPTQ = Hessian-aware, AWQ = activation-aware, int4 kernel = dequant-in-registers, outliers = the reason for fancy algorithms, fp8 = training-friendly, QAT = train with quant simulated_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team quantizes their 7B model to int4 with simple max-abs (no GPTQ/AWQ). Eval perplexity rises by 60%. What likely happened, and what's the fix?

show answer

Outliers. The max-abs scale is dominated by a small number of large weights, which forces the other 99% of weights into a coarse rounding grid. The 60% perplexity rise is symptomatic — modern LLM weight distributions are heavy-tailed enough that naive int4 PTQ is borderline unusable. Fix: switch to GPTQ or AWQ, both of which handle outliers explicitly. GPTQ propagates rounding error via Hessian to compensate; AWQ rescales salient channels so they don't dominate. Either should drop the perplexity rise to 1-3%. The key insight: **for int4, naive max-abs isn't a valid baseline — use a proper PTQ algorithm**.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why does int4 weight-only quantization give a much bigger speedup for autoregressive decode than for prefill?

show answer

Decode processes one token at a time through every layer of the model. The matmul shapes are `(1, K) × (K, N)` — extremely tall-and-skinny, with very few FLOPs per byte of weight loaded. _Decode is heavily memory-bound_ : most time is spent waiting for weights to come from HBM. Cutting weight bytes by 4× cuts wait time by ~4×, giving a near-linear speedup.

Prefill processes the whole prompt at once (e.g., 2048 tokens). Matmuls have shape `(2048, K) × (K, N)` — much more compute per byte of weight, closer to compute-bound. Weight bandwidth still helps but isn't the bottleneck. Speedup is smaller, often 1.3-1.5×. **This is why "int4 makes inference fast" is mostly about decode latency — the prefill TFLOPs/sec doesn't change much.**

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** A team's 7B model with GPTQ-int4 has good perplexity but produces garbage on certain inputs. Investigation shows it's specifically inputs with rare tokens. What's likely going on?

show answer

The embedding and lm_head matrices are likely quantized along with the rest of the model, and the rows for rare tokens — used very rarely during calibration — were quantized poorly. Calibration data didn't activate those rows enough for GPTQ to optimize them. Fix: **keep embeddings and lm_head in fp16/bf16 even when the rest is int4**. They're small (typically <5% of model size) and their precision matters disproportionately for token quality. Most production quantization recipes do this by default — explicitly excluding the embedding and head from quantization. _Lesson: not all layers should be quantized at the same precision. Sensitive parts stay in higher precision._

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Why is fp8 useful for training but int4 isn't? They're both ~2× compression vs bf16 weights.

show answer

Two reasons.

(1) **Range vs precision tradeoff**. fp8 (especially e5m2) has fp32-comparable range — it can represent gradients across many orders of magnitude. int4 has 16 evenly-spaced values within [-7, 7]; gradients (which span many orders of magnitude during training, especially in early steps and for sparse updates) fall off the end. You'd need per-tensor scaling that adapts every step (essentially what fp16 + GradScaler does).

(2) **Hardware tensor cores**. fp8 has direct tensor-core support on H100 — the matmul itself is twice as fast as bf16. int4 has no native tensor-core matmul (CUTLASS has int4 MMA but it's specialized and rarely useful for training); the dequant-then-matmul pattern is for inference where weights are static. _For training, you need the matmul itself to be faster, which fp8 does and int4 doesn't._

This is why the modern recipe is: int4 (or int8) for inference (decode is bandwidth-bound; weights are static; per-group calibration is fine); fp8 for training (matmul is the bottleneck; dynamic range matters; tensor cores accelerate both fwd and bwd).

### What just happened?

  * Quantization stores weights in fewer bits: **int8** (2× compression), **int4** (4×), **fp8** (2×, but trains).
  * The math: `q = round(w/s + z)`, `w_hat = s*(q − z)`. Scales and zero-points are stored alongside the quantized weights.
  * **Granularity** : per-tensor (worst quality), per-channel (good for int8), per-group with group-size 128 (best, used for int4).
  * **PTQ algorithms** : MinMax (naive, ok for int8 only), GPTQ (Hessian-aware reconstruction), AWQ (activation-aware salient-channel preservation). Both GPTQ and AWQ work for int4; try both.
  * **Outliers** (a few weights/activations ~10× larger than the rest) are the reason naive quantization fails for LLMs — they dominate per-tensor scales.
  * **The int4 weight-only matmul kernel** : load packed int4 weights from HBM, extract nibbles in registers, dequantize via `w = scale*(q − zero)`, matmul in bf16 on tensor cores. Hand-written kernel territory; not in cuBLAS.
  * **Why int4 is fast for inference** : most LLM inference is memory-bound on weight loads; 4× smaller weights ≈ near-linear speedup, especially for decode.
  * **fp8** is for training and inference both: e4m3 forward, e5m2 backward, with per-tensor scaling (similar to GradScaler). Matmul itself is faster on H100 tensor cores.
  * Practical recipe for inference: **int4 with GPTQ or AWQ, group-size 128, with embedding and lm_head kept in bf16**. For training on H100: **fp8 via torchao.float8 or TransformerEngine**.
  * Don't write int4 kernels yourself — use `torchao`, `marlin`, `vllm`, `bitsandbytes`. Read them to learn the patterns.
  * **Sensitive layers stay in higher precision** : embeddings and lm_head usually in bf16 even when the body is int4.
  * The reflex: when serving an LLM, ask "is this memory-bound or compute-bound?" Quantize for the memory-bound parts; fp8 for the compute-bound matmuls; leave outlier-sensitive layers alone.

Module 27 takes quantization plus FlashAttention plus the dispatcher and assembles them into **inference systems**. The KV cache and how it's managed; paged attention as the descendant of FlashAttention for variable-length serving; speculative decoding; the throughput vs latency tradeoff; what vLLM and TensorRT-LLM are doing under the hood. Then M28 closes the course with mixture-of-experts and a frontier capstone.
