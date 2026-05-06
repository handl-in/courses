# Module 30 — RoPE and positional encodings

# RoPE and _positional encodings_

_Part IX · Module 30_

— why attention needs position information, the four families that solve it, and how RoPE became the universal default through one elegant property: `(Qᵢ · Kⱼ)` depends only on `i − j`

\--- 

In M29 we built a transformer and used Rotary Position Embedding without much explanation. This module unpacks it. Why does attention need position information at all? Why has rotation specifically — not addition, not learned vectors, not biases — become the universal default in modern LLMs? And how do you extend a model trained at 4K context to 32K, 128K, or 1M tokens at inference time?

The answers form a small but tight body of knowledge. By the end you'll understand why RoPE works, the design space it sits in, and the practical recipes (NTK-aware interpolation, YARN, Dynamic NTK) that let production models stretch their context to where the training data didn't reach.

> **★ KEY IDEA**  
>  Attention without positional encoding is _set-equivariant_ : shuffle the input tokens and the output shuffles correspondingly. To make sequence position matter, you have to inject it somewhere. **RoPE** (Rotary Position Embedding) does this by rotating Q and K vectors by an angle that depends on token position. The clever bit: when you take the dot product Q · K, the rotation angles subtract — **the score depends only on the _relative_ position `i − j`**, not on absolute positions. This is the property modern transformers want, and rotation is the unique way to get it without modifying the dot-product structure. The frequency schedule (`θ_base = 10000`) determines the range over which the model can distinguish positions; **raising θ_base extends the model's effective context** , which is what NTK-aware scaling does for long-context fine-tuning. 

## One new face

⊙

RoPE

"I rotate. I'm not added; I'm not concatenated; I rotate Q and K by position-dependent angles."

For every pair of dimensions in your Q vector, I rotate them in the 2D plane by an angle proportional to the token's position. Same for K. Different pairs of dimensions rotate at different frequencies — high frequency for early dims, low frequency for late dims. _The dot product Qᵢ · Kⱼ then depends only on i − j_ , because the rotations partially cancel. I'm parameter-free, I extrapolate gracefully, and I leave the attention kernel completely unchanged. **I'm what every modern LLM uses.** The catch: my frequency schedule decides the position range I can distinguish, so extending context length needs me re-tuned (NTK, YARN, Dynamic NTK) — the practical recipes you'll meet later in this module.

## Why position matters

Think about what attention does without position information. For tokens with embeddings `x₁, x₂, ..., xₜ`, the attention output for position i is:
    
    
    Output_i = Σⱼ softmax((Qᵢ · Kⱼ) / √d) · Vⱼ

Now permute the input: tokens arrive as `x₃, x₁, x₂` instead of `x₁, x₂, x₃`. The Q, K, V values get permuted correspondingly. The dot products are the same (just paired differently), the softmax is over the same set of values, and the output for what was originally position 1 is identical regardless of where in the sequence it now sits. **Attention treats its input as a set, not a sequence.**

This is fatal for language. "Dog bites man" and "Man bites dog" have the same token set but radically different meanings. To make attention sequence-aware, position information must enter somewhere — either added to the embeddings, baked into the attention weights, or applied to Q and K directly.

The four families of solutions:

Four families of positional encoding (shown for one head) ① Absolute learned (BERT, GPT-2) x_i = token_emb[t_i] + pos_emb[i] (separate (max_len, D) learned table) Pros: • Simple, learned end-to-end • No architectural changes to attention Cons: • Cannot extrapolate past max trained position • Adds (max_len × D) params ② Sinusoidal (original Transformer) x_i = token_emb[t_i] + sin/cos(i / 10000^(2k/D)) (deterministic, multi-frequency) Pros: • Parameter-free • Some extrapolation in theory Cons: • Extrapolation poor in practice • Position is absolute, not relative ③ Relative position bias (T5, ALiBi) attn[i, j] += b(i − j) /* learned or fixed */ (modify attention scores directly) Pros: • Position is naturally relative • ALiBi extrapolates well (linear bias) Cons: • T5 form: large learned bias table • Mediocre quality at long context ④ Rotary (RoPE) — the modern default ★ Rotate Q_i, K_j by angles θ_i, θ_j; result: Q · K depends only on (i − j) Pros: • Parameter-free, kernel-friendly • Relative-position via construction • Extends to long context with NTK/YARN Used by: Llama, Mistral, Qwen, Gemma, DeepSeek

## The mathematical setup

What property do we want? In an attention layer, the score for token i attending to token j is `Qᵢ · Kⱼ`. We want this to depend on _where_ tokens are relative to each other — specifically, we want:
    
    
    Q_i · K_j = f(x_i, x_j, i − j)

The score should depend on the content of the tokens (`x_i`, `x_j`) and on their _relative_ position `i − j`, but _not_ on the absolute positions `i` and `j` separately. Why relative? Two reasons. (1) Empirically: language structure is mostly local — what matters is "the verb is two tokens before this noun," not "this verb is at position 1572." (2) Generalization: a model that learns about relative positions extrapolates to longer sequences than one that learned about specific absolute positions.

RoPE's insight: **rotation is the unique transformation that achieves this property without changing the dot-product structure.** Here's why.

## Deriving RoPE

Start with one pair of dimensions in Q (call them `q_a, q_b`) and one pair in K (`k_a, k_b`). Suppose we transform them by some position-dependent function:
    
    
    Q_i_pair = R(i) · [q_a, q_b]
    K_j_pair = R(j) · [k_a, k_b]

where `R(p)` is some 2×2 matrix depending on position `p`. We want:
    
    
    Q_i_pair · K_j_pair  =  R(i) [q_a, q_b]  ·  R(j) [k_a, k_b]
                         =  [q_a, q_b]^T · R(i)^T · R(j) · [k_a, k_b]

For this to depend only on `i − j`, the product `R(i)^T · R(j)` must be a function of `i − j` alone. The 2D rotation matrix has exactly this property:
    
    
    R(p) = [[cos(pθ), -sin(pθ)],
            [sin(pθ),  cos(pθ)]]
    
    R(i)^T · R(j) = R(j − i)              # standard rotation identity

So if `R(p)` is a rotation by angle `pθ`, the transformed dot product is:
    
    
    Q_i_pair · K_j_pair = [q_a, q_b]^T · R(j − i) · [k_a, k_b]

This depends on `j − i`, the relative position. _Not on absolute positions._ Done.

To extend this to a full Q vector of dimension D, we apply _different_ rotations to different pairs of dimensions: pair (0, 1) rotates with one frequency, pair (2, 3) with another, and so on. Different frequencies let the model encode position information at multiple scales — high-frequency rotations distinguish nearby tokens; low-frequency rotations distinguish far-apart tokens.

RoPE: each dimension pair rotates by position × its own frequency Q_i (head_dim = 8, shown as 4 pairs): (q_0, q_1) freq θ₀ (high) (q_2, q_3) freq θ₁ (q_4, q_5) freq θ₂ (q_6, q_7) freq θ₃ (low) rotate pair k by angle (i × θ_k) Q_i after RoPE — each pair rotated by its own angle: rot by i·θ₀ rot by i·θ₁ rot by i·θ₂ rot by i·θ₃ The frequencies (θ_base = 10000 standard): θ_k = 1 / (θ_base ^ (2k/D)) → θ_0 = 1 (period 2π), θ_1 ≈ 0.31, ..., θ_{D/2−1} ≈ 1/θ_base (very long period) high freqs = local position cues; low freqs = global position cues

## The implementation

The pair-wise rotation can be implemented as elementwise products with cached cos and sin tables. From M29:
    
    
    def build_rope_cache(seq_len, head_dim, base=10000.0, device="cuda"):
        # Frequencies: θ_k = 1 / base^(2k/head_dim) for k = 0, 1, ..., head_dim/2 − 1
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        # Outer product of positions and frequencies → angles
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, inv_freq)              # [T, head_dim/2]
        # cos and sin tables, replicated to head_dim (each pair gets the same cos/sin)
        cos = freqs.cos().repeat_interleave(2, dim=-1)  # [T, head_dim]
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        return cos, sin
    
    def apply_rope(x, cos, sin):
        # x: [..., T, head_dim]. Apply 2D rotation in fp32.
        x_fp32 = x.float()
        # Pair adjacent dims: (x[0], x[1]), (x[2], x[3]), ...
        x1, x2 = x_fp32[..., ::2], x_fp32[..., 1::2]
        # Rotation: [x1, x2] → [x1·cos − x2·sin, x1·sin + x2·cos]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return (x_fp32 * cos + rotated * sin).to(x.dtype)

Three details worth dwelling on:

  * **fp32 throughout the rotation**. Cosines and sines at large positions can lose precision in bf16; the standard recipe is to cast to fp32 for the rotation, then back to the input dtype. Same pattern as M14's mixed-precision recipe.
  * **The trick with`stack([-x2, x1])`**: this implements the imaginary-part swap of the rotation. `(x1, x2)` rotated by angle `θ` becomes `(x1·cos − x2·sin, x1·sin + x2·cos)`. The first component uses `x1` with cos and `−x2` with sin; the second uses `x2` with cos and `x1` with sin. The `stack([-x2, x1])` reorders into the layout that, when multiplied elementwise by `sin`, gives exactly the `x1·sin` and `−x2·sin` terms.
  * **`repeat_interleave(2)`** : each _pair_ of dimensions shares the same cos/sin value (since they rotate by the same angle). The interleave broadcasts the `[T, head_dim/2]` table to `[T, head_dim]`.

## The complex-number view

There's an equivalent, mathematically cleaner formulation of RoPE that uses complex numbers. Treat each pair of dimensions as a complex number: `z = x1 + i·x2`. Rotation by angle `θ` is then just complex multiplication:
    
    
    z_rotated = z · exp(i·θ) = (x1 + i·x2) · (cos θ + i·sin θ)
                             = (x1·cos θ − x2·sin θ) + i·(x1·sin θ + x2·cos θ)

Same result as before. The dot product of two rotated complex numbers becomes:
    
    
    z_a_rotated · conj(z_b_rotated) = (z_a · exp(i·θ_a)) · conj(z_b · exp(i·θ_b))
                                    = z_a · conj(z_b) · exp(i·(θ_a − θ_b))

The exponential factor `exp(i·(θ_a − θ_b))` only depends on the angle difference — i.e., on the relative position. This is mathematically the same property as before; the complex notation just makes it transparent.

Some implementations (and PyTorch's reference RoPE in `torch.nn.functional`) use complex tensors directly: pack pairs into `complex64`, multiply by `exp(i·θ)`, unpack. Others use the real-valued split-and-combine trick from M29. Performance is identical; the complex form is shorter to write but requires complex tensor support in your kernel.

## The frequency choice: why θ_base = 10000?

The original RoPE paper used `θ_base = 10000`, matching the sinusoidal positional encoding from the original Transformer. The choice has a concrete consequence: it determines the range of positions over which the rotation pattern is "non-degenerate."

The lowest frequency in the schedule is `θ_{D/2−1} = 10000^(−1) = 0.0001`. The corresponding period is `2π / 0.0001 ≈ 62832`. So even the slowest-rotating dim pair completes one rotation every ~63K positions. For a model trained at 4K context, all the rotations are well-distinguished — every position gets a unique rotation pattern across all pairs.

**What changes at`θ_base = 500000`?** The lowest frequency becomes `500000^(−1) = 2e−6`, period `~3.1M`. The rotation pattern remains non-degenerate over much longer ranges. This is what Llama-3 did to extend context from 8K (training) to 128K (inference): they set `θ_base = 500000` during fine-tuning, retraining the model briefly to use the new schedule. The model now distinguishes positions out to ~500K-1M before the rotations alias.

The general rule: **θ_base must be set so the slowest-rotating dim pair completes < 1 full rotation over the maximum sequence length**. If it completes more than one rotation, positions alias (position 0 and position N rotate to the same angle), and the model can't distinguish them.

## Context extension: NTK, YARN, Dynamic NTK

You have a model trained at context length 4K with `θ_base = 10000`. You want to use it at 32K context without retraining from scratch. What do you do?

The naïve approach — just extend the cos/sin tables to position 32K — fails. Positions beyond 4K were never seen at training, and the model doesn't know how to interpret rotations beyond that range. Loss spikes; outputs become nonsensical.

Three recipes that actually work:

### 1\. Position interpolation

The simplest extension. To run at 32K context with a model trained at 4K, divide all positions by `32K / 4K = 8`. The model sees positions `0, 0.125, 0.25, 0.375, ..., 4000` (32K positions, but rescaled into the trained range). All position indices fit into the original training range; the model sees them as it always did.
    
    
    scale = original_context / new_context     # 4K / 32K = 0.125
    positions_scaled = positions * scale
    # Now use these in the standard RoPE formula

This works! But it loses high-frequency information — the model now distinguishes positions only at 1/8 the original resolution. Quality degrades on tasks needing fine-grained positional reasoning.

### 2\. NTK-aware interpolation

The clever fix. The insight: high-frequency dim pairs (small `k`) are responsible for fine-grained position distinctions; low-frequency pairs handle long-range. Position interpolation hurts the high frequencies more than the low ones (rescaling by 8 changes a fine-grained "2 vs 3" distinction into a "0.25 vs 0.375" one — much harder to detect).

NTK-aware scaling instead changes `θ_base` rather than the positions. Specifically:
    
    
    scale_factor = new_context / original_context        # e.g., 8
    new_theta_base = theta_base * (scale_factor)^(D / (D − 2))

The exponent `D / (D − 2)` is chosen so that high-frequency pairs are barely affected (their rotation rate stays close to original) while low-frequency pairs are rescaled to span the new context length. **Fine-grained position cues are preserved; coarse-grained ones are stretched.**

This is what Llama-3 did to get from 8K to 128K context. Apply NTK-aware scaling, fine-tune for 1-2K steps, done. Quality at 128K matches or exceeds the 8K baseline on most tasks.

### 3\. YARN

YARN (Yet Another RoPE extensioN) refines NTK further. The core idea: NTK-aware scaling treats all dim pairs as "equally important" to extend, but in practice some pairs benefit from interpolation while others benefit from extrapolation. YARN partitions the dim pairs into three regions:

  * **High-frequency region** : extrapolate (don't rescale; use original θ_k).
  * **Low-frequency region** : interpolate (rescale by the full factor).
  * **Middle region** : a smooth blend between the two.

YARN also adds a "temperature" tweak — slightly raising the softmax temperature in attention to compensate for the increased entropy of attention scores at long distances. Quality at extreme context (1M+) is meaningfully better with YARN than with vanilla NTK scaling.

### 4\. Dynamic NTK at inference time

The above recipes assume you fine-tune at the new context length. Dynamic NTK is a runtime-only trick: _scale θ_base based on the current sequence length, not a fixed factor_. As the conversation grows past the trained context, the rotation frequencies stretch dynamically.
    
    
    def dynamic_ntk_theta_base(theta_base, current_seq_len, original_max_seq_len, head_dim):
        if current_seq_len <= original_max_seq_len:
            return theta_base                  # short — use original
        scale = current_seq_len / original_max_seq_len
        return theta_base * scale ** (head_dim / (head_dim - 2))

No fine-tuning required. Quality is worse than YARN with fine-tuning, but acceptable for shorter excursions (e.g., a 4K-trained model handling 8-16K conversations). Used in many production servers as a "best-effort" fallback.

## RoPE in the kernel

One detail that's load-bearing for performance: **RoPE applies to Q and K _before_ the attention matmul, not inside the attention kernel**. In M25's FlashAttention, the inputs are already-rotated Q and K. The RoPE rotation is its own kernel (or a small set of element-wise ops), and FlashAttention is unchanged.

This separation matters for two reasons:

  1. **FlashAttention stays a black box**. The same FlashAttention kernel handles RoPE'd attention, ALiBi'd attention, vanilla attention — all by varying what gets fed in. PyTorch's `F.scaled_dot_product_attention` doesn't know or care about RoPE; it just does attention on whatever Q and K it receives.
  2. **The KV cache stores post-RoPE keys**. When you append a new token's K to the cache during decode, you've already applied RoPE to it. The cache is thus "natively" positioned. _Reading from the cache during attention requires no extra rotation_ — the rotations are baked in at write time.

The KV cache implication has a sharp consequence: **changing the positional encoding scheme requires rebuilding the cache**. If you train a model with `θ_base = 10000` and want to switch to `θ_base = 500000` at inference, every cached key has been rotated with the wrong frequency. Discard the cache and re-prefill.

This is also why some advanced attention variants (notably FlashAttention v3 with built-in RoPE) absorb RoPE into the kernel. The performance gain is small for typical workloads but adds up at scale.

## ALiBi: the alternative that didn't quite win

ALiBi (Attention with Linear Biases) takes the opposite approach. Instead of rotating Q and K, ALiBi adds a position-dependent bias directly to the attention scores:
    
    
    attn_scores[i, j] = Q_i · K_j  +  m_h · (j − i)

where `m_h` is a per-head slope (negative; tokens further away get a more negative bias, attended to less). The slopes `m_h` are fixed (geometric sequence, no learned parameters).

ALiBi has a remarkable property: it extrapolates to longer contexts _without any rescaling or fine-tuning_. A model trained at 1K context handles 16K context out of the box, with quality decay but no catastrophic failure. The mechanism: long-distance attention scores are biased very negative, so attention naturally focuses on local context — and "local" means the same thing at 1K and 16K positions.

So why didn't ALiBi win?

  * **Quality at the trained context length is slightly lower** than RoPE. Across many benchmarks, RoPE-trained models outperform ALiBi-trained models when both are evaluated at training context.
  * **Less flexibility for context extension**. NTK / YARN give RoPE a continuous knob to tune for new context lengths; ALiBi's extrapolation, while graceful, has no equivalent fine-tuning recipe.
  * **Network effects**. RoPE was adopted by Llama-1, then Mistral, then Gemma, then Qwen, then DeepSeek. Once an ecosystem coalesces, the marginal benefit of switching is rarely worth it. ALiBi survives in a few research models (e.g., Bloom, Falcon).

If you're building a new model in 2026, use RoPE. If you're working with an ALiBi model, the ideas in this module still apply, just to the bias function instead of the rotation angles.

## The quick-reference table

Positional encoding schemes — production status as of 2026 Scheme| Used by| Extrapolates?| Recommended?  
---|---|---|---  
Absolute learned| BERT, GPT-2 (legacy)| No (hard wall at trained max)| No — superseded  
Sinusoidal| Original Transformer| Marginally| No — superseded  
T5 relative bias| T5| Some| Niche — works for T5-style architectures  
ALiBi| BLOOM, Falcon| Yes (gracefully)| Niche — alternative to RoPE  
RoPE (θ_base = 10000)| Llama-1, Llama-2, Mistral, Gemma-1, Qwen, DeepSeek| With NTK/YARN scaling| **Yes — modern default**  
RoPE (θ_base = 500000)| Llama-3, Llama-3.1, Mistral Large, Gemma-2| To 128K+ with retraining| **Yes — for long context**  
YARN| Yi-VL-200K, Qwen2.5-1M| To 1M+| Yes — for very long context  
  
#### Q&A; — About RoPE and positions **Q:** Why pair-wise rotation (2D rotations of consecutive dim pairs) instead of, say, full D-dimensional rotation by some angle? **A:** The pair-wise structure has two advantages. (1) It's _cheap_ : D/2 independent 2D rotations cost ~2D scalar multiplies per token, vs O(D²) for a full D-dim rotation. (2) It admits multiple frequencies: each pair rotates at its own rate, encoding position information at multiple scales simultaneously. A single D-dim rotation has one angle, hence one "frequency" — much less expressive. The pair-wise trick is the standard way to get multi-scale relative-position encoding without the cost of generalized rotations. **Q:** Does the order of dimension pairs matter? Could I pair (0, 4), (1, 5), instead of (0, 1), (2, 3)? **A:** Mathematically, no — the pairing is arbitrary as long as it's consistent between Q and K. In practice, two pairing conventions exist: **interleaved** (the split-and-stack code in M29: pair (0, 1), (2, 3), ...) and **halved** (pair (0, D/2), (1, D/2+1), ..., used by Hugging Face's Llama implementation). They're mathematically equivalent but produce different bit patterns. _This causes occasional bugs when porting weights between frameworks_ — make sure your RoPE implementation matches the one used during training. **Q:** Why must RoPE be applied to Q and K but not V? **A:** The "depend only on i − j" property only matters for the dot product Q · K. V doesn't enter the attention score; it's the value being weighted by the attention pattern. Applying a position-dependent rotation to V would scramble the values without any compensating un-rotation downstream — V_j would emerge from the attention weighted sum rotated, which is not what you want. _RoPE rotates the keys to "look up by position"; V is the data being looked up, untouched._ **Q:** What's the right θ_base for very long context (1M+)? **A:** Empirically, around `θ_base ≈ context_length × 100` seems to work well, paired with YARN-style frequency partitioning. For 1M context, that's `θ_base ≈ 1e8`. But this is approximate — at extreme context lengths, the choice of θ_base matters less than the YARN region cutoffs and the temperature tweak. _Read the YARN paper for the precise recipe; trust empirical results over theoretical predictions at this scale_. **Q:** If RoPE is so universal, why do we still teach the older positional encoding schemes? **A:** Three reasons. (1) Legacy: many production checkpoints (BERT, GPT-2, T5) use older schemes; you'll encounter them when working with older models. (2) Conceptual: understanding "why RoPE works" requires understanding what it solved — namely the absolute-vs-relative tension that the older schemes didn't handle well. (3) Future: the next big-paradigm-shift in positional encoding (e.g., learned implicit positional functions, or RoPE alternatives for very long context) will likely be motivated by RoPE's remaining limitations. _Knowing the design space is what lets you evaluate the next thing_. **Q:** Does RoPE interact with quantization (M26)? **A:** Yes, in two places. (1) The cos/sin tables themselves should stay in fp32 — they're small (a few MB), and quantizing them loses precision in the rotation. Most production quantization recipes leave RoPE buffers in fp32. (2) The post-RoPE Q and K can be quantized for the attention matmul. Some FlashAttention variants (especially the int8 / fp8 paths) apply quantization to the rotated Q and K, not the original. _The cache stores post-RoPE post-quantization values_. 

## Code Magnets: a complete RoPE apply

You're writing the apply step of RoPE. Three magnets are wrong choices.

Arrange the magnets into a correct apply_rope.

def apply_rope(x, cos, sin): x_fp32 = x.float() x_fp32 = x x1, x2 = x_fp32[..., ::2], x_fp32[..., 1::2] x1, x2 = x_fp32[..., :head_dim//2], x_fp32[..., head_dim//2:] rotated = torch.stack([-x2, x1], dim=-1).flatten(-2) rotated = torch.cat([-x2, x1], dim=-1) return (x_fp32 * cos + rotated * sin).to(x.dtype)

show solution
    
    
    def apply_rope(x, cos, sin):
        x_fp32 = x.float()
        x1, x2 = x_fp32[..., ::2], x_fp32[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return (x_fp32 * cos + rotated * sin).to(x.dtype)

The traps:

  * `x_fp32 = x` (no `.float()`): skips fp32 promotion. Rotation in bf16 loses precision at large positions; cos and sin near phase boundaries cancel poorly. Standard recipe is to do the rotation in fp32 and cast back at the end.
  * `x1, x2 = x_fp32[..., :head_dim//2], x_fp32[..., head_dim//2:]`: this is the _halved_ pairing convention (Hugging Face's Llama). The cos/sin tables built with `repeat_interleave` match the _interleaved_ convention. Mixing conventions silently produces wrong outputs (mathematically valid rotations, but rotating different pairs than the cos/sin tables expect).
  * `rotated = torch.cat([-x2, x1], dim=-1)`: concatenation puts all the −x2 values first, then all the x1 values. The interleaved cos/sin tables expect alternating layout (−x2[0], x1[0], −x2[1], x1[1], ...) which `stack + flatten` produces correctly. `cat` mismatches.

The pattern: **cast to fp32, split into adjacent pairs (interleaved), construct the rotated layout via stack-and-flatten, multiply elementwise with cos and sin, cast back**. Every step has a specific role; mixing conventions or skipping fp32 silently breaks the model.

## Who does what?

Match each positional-encoding concept to its real role.

Concept

Real role

Set-equivariance of attention

A. Attention treats input as a set; positions must be injected somewhere.

Relative-position attention

B. Q · K depends only on i − j, not on absolute positions; what RoPE achieves.

θ_base = 10000

C. Standard RoPE frequency base; the slowest pair has period ~63K positions.

θ_base = 500000

D. Used in Llama-3 to extend context to 128K; slowest pair has period ~3M.

NTK-aware scaling

E. Rescales θ_base by scale^(D/(D−2)); preserves high-frequency cues, stretches low-frequency.

YARN

F. Partitions dim pairs into extrapolate / interpolate / blend regions; best for very long context.

ALiBi

G. Linear bias on attention scores; extrapolates without retraining but quality slightly below RoPE.

show solution

**Set-equivariance of attention** → A  
**Relative-position attention** → B  
**θ_base = 10000** → C  
**θ_base = 500000** → D  
**NTK-aware scaling** → E  
**YARN** → F  
**ALiBi** → G 

The mental shortcut: _attention is order-blind, RoPE makes it relative, 10K is the standard base, 500K is for long context, NTK rescales the base, YARN partitions the schedule, ALiBi is the linear-bias alternative_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team trains a model with RoPE θ_base=10000 at 4K context, then deploys it at 16K context without retraining. Loss spikes immediately at sequences past 4K. Why, and what's the simplest fix that doesn't require retraining?

show answer

The model has only ever seen rotation angles up to those produced by position 4095. At positions beyond 4K, the rotations enter a regime the model never trained on — the rotation patterns alias or wrap around in unfamiliar ways, and attention scores become noise. The model can't extrapolate cleanly because high-frequency rotation pairs cycle multiple times within the new range.

Simplest no-retrain fix: **Dynamic NTK**. At runtime, when the sequence length exceeds the trained max, scale θ_base by `(seq_len / trained_len)^(D/(D−2))`. The high-frequency pairs barely change (stay near their trained behavior); low-frequency pairs stretch to span the new context. Quality is degraded vs proper YARN-fine-tuning but usable for moderate excursions (4K → 8K-16K). For longer extensions, retrain with NTK or YARN.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why is the cos/sin cache stored as a buffer with `persistent=False` rather than computed on the fly each forward?

show answer

Two performance reasons. (1) The cache is computed _once_ at model init from the config (max_seq_len, head_dim, θ_base). Recomputing it every forward would mean redoing the trig functions for every position every step — wasted work. (2) The cache is small (a few MB for typical configs) and lives entirely on the GPU after init; sliced reads from it for the current segment cost almost nothing.

`persistent=False` means it's not saved in the checkpoint. Why exclude? The cache is fully determined by the config — saving it just bloats checkpoints. And critically, if you load a checkpoint and want to run at a longer context (or different θ_base), the saved cache would be wrong. **persistent=False forces the cache to be regenerated from the current config every load** , which is what you want.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** When a team applies NTK scaling and fine-tunes for 2K steps, validation perplexity drops smoothly. They then notice that some retrieval tasks (find a fact 100K tokens back) work great, but tasks that need precise local position (token-level reasoning, e.g. "swap word 5 with word 8 in the prompt") get worse. Explain.

show answer

NTK scaling preserves high-frequency rotation pairs but stretches low-frequency ones. The high-frequency pairs are still encoding fine-grained position cues, so local positional reasoning should be preserved — but only if the fine-tuning preserved the model's reliance on them. In practice, fine-tuning at long context often shifts the model's "attention budget" toward exploiting the newly-extended low-frequency pairs (because that's where the gradient signal is during long-context tasks). The high-frequency pairs effectively atrophy.

Diagnostic: probe attention patterns on local-reasoning tasks pre- and post-extension. The pre-extension model attends to specific nearby tokens; the post-extension model attends more diffusely. Mitigation: include short-context examples in the fine-tuning mix (Llama-3 did this — the long-context fine-tuning data was 50% short-context, 50% long-context to prevent regression). YARN's blended region partition is also designed to address this.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Why doesn't FlashAttention need to be modified to support RoPE, when adding ALiBi requires a small modification to the kernel?

show answer

RoPE is applied to Q and K _before_ they enter the attention kernel. From FlashAttention's perspective, it just receives "some Q and K" and computes attention; whether they were rotated by RoPE, normalized, or projected from a different space is irrelevant. The kernel is RoPE-agnostic.

ALiBi adds a position-dependent bias to the attention scores _inside_ the kernel: `scores[i, j] += m_h × (j − i)`. This addition has to happen between the Q · K matmul and the softmax. Since the scores tile is materialized in registers/shared memory inside FlashAttention, adding the ALiBi bias is a small modification to the kernel's inner loop. Some FlashAttention versions (especially v2+) ship explicit ALiBi support; others require a small fork.

This is part of why RoPE won the architecture war: it slots in front of any attention kernel without modification. _The right design moves the complexity to where it can be expressed cleanly, leaving the high-throughput kernel untouched_.

### What just happened?

  * Without positional encoding, attention is **set-equivariant** — order-blind. Position must be injected somewhere.
  * Four families of solutions: **absolute learned** (legacy), **sinusoidal** (legacy), **relative bias / ALiBi** (niche), **rotary / RoPE** (modern default).
  * RoPE rotates each pair of dimensions in Q and K by an angle proportional to the token's position. Different pairs use different frequencies — high frequency for early dim pairs (local cues), low frequency for late ones (global cues).
  * The mathematical magic: when you take Q · K, the rotation angles subtract — **Q_i · K_j depends only on (i − j)**. Relative-position attention by construction.
  * The **complex-number view** : pair adjacent dims as a complex number, multiply by `exp(i·θ)`. Mathematically equivalent to the real-valued formulation, sometimes shorter to implement.
  * The **frequency schedule** (θ_k = 1 / θ_base^(2k/D)) determines the position range over which rotations are non-degenerate. **θ_base = 10000** standard; **500000** for long-context fine-tuning (Llama-3 used this for 8K → 128K).
  * Implementation: precompute cos/sin tables once, apply via elementwise products in fp32 (cast back at the end). Stored as a buffer with `persistent=False`.
  * RoPE applies to Q and K only, not V. Common bug: applying to V scrambles the values.
  * **Context extension recipes** : position interpolation (simple, loses fine resolution); **NTK-aware scaling** (rescale θ_base, preserves high frequencies); **YARN** (partition the dim pairs, best for very long context); **Dynamic NTK** (runtime-only, no fine-tuning needed).
  * RoPE applies _before_ the attention kernel. FlashAttention is RoPE-agnostic — receives already-rotated Q and K. ALiBi requires a small kernel modification to add bias to scores.
  * The KV cache stores **post-RoPE keys**. Changing θ_base requires rebuilding the cache.
  * Two pairing conventions exist: **interleaved** (pair (0,1), (2,3)) and **halved** (pair (0, D/2), (1, D/2+1)). Mathematically equivalent but mixing them silently breaks weights ports between frameworks.
  * RoPE quantization (M26): cos/sin tables stay in fp32; post-RoPE Q and K can be quantized for the matmul.
  * The reflex: when designing or evaluating an architecture change, ask "does this preserve the relative-position property and stay outside the attention kernel?" If yes, it composes cleanly. If no, expect kernel-level engineering.

Module 31 turns to **post-training** — the phase that takes a base model and shapes it into something useful via SFT, reward modeling, RLHF (PPO), and the modern simpler alternative DPO. The engineering of having three or four models in memory at once (policy, reference, reward, value); why DPO became the default for most teams; and the distributed-rollout architecture that production RLHF setups use. After that, M32 consolidates debugging skills, and M33 explores Mamba as the transformer alternative.
