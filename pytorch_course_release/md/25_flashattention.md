# Module 25 — FlashAttention: a case study

# FlashAttention: _a case study_

_Part VIII · Module 25_

— how the online softmax trick turned attention from memory-bound to compute-bound, and why the kernel is one of the most consequential pieces of code of the past five years

\--- 

This is the module where everything you've built converges. We're going to walk through FlashAttention — the kernel that, more than any other single piece of work, made long-context transformers practical. Not "make them faster" — _make them possible_. Without it, training a 32K-context model would require terabytes of activation memory; serving long-prompt inference would be unaffordable. With it, both fit on commodity hardware.

Here's what makes FlashAttention worth a whole module: its trick isn't algorithmic novelty (the math is just a careful refactoring of softmax). The trick is _understanding the hardware well enough_ (M22) _to restructure the algorithm so the right things stay in the right level of memory_. Every concept we've built leads here. The roofline (M22) tells us why naive attention is bad; the GPU memory hierarchy (M22) tells us where to keep what; tile-based programming (M24) is how we write it; the recompute trick (M12) is how the backward works; `torch.library` (M19, M23) is how we register it; mixed precision (M14) is how we keep it stable.

> **★ KEY IDEA**  
>  Naive attention is memory-bound: it computes the full `(T, T)` attention matrix in HBM, reads it back to softmax, then reads it again to multiply by V. **FlashAttention restructures the computation so the attention matrix never lives in HBM** — it's recomputed in tiles inside shared memory using the _online softmax_ trick. Streaming K and V through one Q tile at a time, the kernel produces the same numerical result with O(T) memory and ~5× less HBM traffic. The cost: the kernel is harder to write than naive attention. The win: long-context training and inference become tractable. **This pattern — restructure to keep working data in shared memory, stream the rest — is the template for modern attention variants (paged attention, MQA/GQA, sliding window) covered in M27.**

## The naive attention recipe (and why it's bad)

Standard scaled dot-product attention, for one head:
    
    
    S = Q @ K.transpose(-2, -1) / sqrt(d)        # (T, T) attention scores
    P = torch.softmax(S, dim=-1)                  # (T, T) probabilities
    O = P @ V                                       # (T, d) output

Three operations, each writing its result to HBM. Per head: read `Q`, `K`, `V` (each `T × d`); write `S` (`T × T`); read `S` back; write `P`; read `P` back; write `O`. The (T, T) intermediates are the killers — for T=2048 and fp16, that's 8 MB _per head, per layer_. For a 32-layer 32-head transformer, peak activation memory from attention alone is ~16 GB at T=2048 — and it's _quadratic in T_.

Roofline analysis (M22) on naive attention at typical sizes:

Naive attention vs FlashAttention: arithmetic intensity at T=2048, d=64 | FLOPs| HBM bytes| Intensity| Regime (H100)  
---|---|---|---|---  
Naive (writes (T,T) to HBM)| ~33M| ~50 MB| ~0.7 ops/byte| Memory-bound (way below ridge)  
FlashAttention (no (T,T) in HBM)| ~33M| ~5 MB| ~7 ops/byte| Still memory-bound but ~10× higher  
FlashAttention with larger heads (d=128)| ~66M| ~5 MB| ~13 ops/byte| Approaching compute-bound  
  
Naive attention is dominated by the (T, T) writes and reads. FlashAttention removes them. The kernel does the same FLOPs but ~10× less HBM traffic. _Same compute, far less memory — the textbook fusion win, applied to attention._

## The online softmax trick

The mathematical core. Standard softmax over a vector x:
    
    
    softmax(x_i) = exp(x_i) / sum_j(exp(x_j))

To do this safely in floating point, you subtract the max before exponentiating (otherwise `exp` overflows for large values):
    
    
    m = max(x)
    softmax(x_i) = exp(x_i - m) / sum_j(exp(x_j - m))

This requires _two passes_ over x: one to compute m (the max), one to compute the sum and divide. If x is a row of the attention matrix S, that means materializing S — exactly what we want to avoid.

The trick: _can we compute softmax incrementally, looking at one block of x at a time, without seeing the rest?_ Yes — by maintaining a running max and a running sum, and "patching" them when a new larger max appears.

Suppose we've seen blocks `x⁽¹⁾` and `x⁽²⁾`, and we have:
    
    
    m₁ = max(x⁽¹⁾)                                    # current running max
    ℓ₁ = sum_j(exp(x⁽¹⁾_j - m₁))                       # sum of exps, normalized to current max

Now block `x⁽²⁾` arrives, with its own local max `m₂_local`. The new global max is:
    
    
    m₂ = max(m₁, m₂_local)

To get the new running sum (referenced to `m₂` instead of `m₁`), we need to _rescale_ the old sum and add the new contribution:
    
    
    ℓ₂ = ℓ₁ * exp(m₁ - m₂) + sum_j(exp(x⁽²⁾_j - m₂))

The key term is `exp(m₁ - m₂)`. If the new max is bigger (`m₂ > m₁`), this is < 1 and shrinks the old contributions to match the new normalization. If the new max is the same (`m₂ = m₁`), it's 1 and nothing changes. **The same patching trick works for the partial output P @ V** — when the running max updates, you rescale the partial output by `exp(m_old - m_new)`.

Worked example with three values, processed one at a time:
    
    
    x = [2, 5, 1]
    
    After x[0]=2:  m=2,    ℓ=exp(0)=1
    After x[1]=5:  m=5,    ℓ_old_rescaled = 1*exp(2-5) = 0.0498
                           ℓ_new = 0.0498 + exp(0) = 1.0498
    After x[2]=1:  m=5 (unchanged), 
                           ℓ_new = 1.0498 + exp(1-5) = 1.0498 + 0.0183 = 1.0681
    
    Standard softmax for comparison:
      m = 5
      ℓ = exp(2-5) + exp(5-5) + exp(1-5) = 0.0498 + 1 + 0.0183 = 1.0681  ✓

Same answer. We never had to see all three values at once — we processed them one at a time, maintaining (m, ℓ) and rescaling when needed.

This is the entire conceptual breakthrough. **You can compute exact softmax by streaming.** Apply this to the rows of S — process K (and V) in blocks, maintain running (max, sum, partial output) per row of Q, patch as needed. The full S never exists.

## The FlashAttention algorithm (forward)

Here's the algorithm in pseudocode. Each program in the kernel handles one tile of Q (BLOCK_M rows). It iterates over K and V in BLOCK_N column tiles, accumulating into the output and the running stats.
    
    
    # For one Q tile: rows [r..r+BLOCK_M] of Q
    load Q_tile  ([BLOCK_M, d])  into shared/registers, KEEP IT THERE
    initialize:
        m_running = -inf  ([BLOCK_M])               # running max per row
        ℓ_running = 0     ([BLOCK_M])               # running sum-of-exps per row
        O_running = 0     ([BLOCK_M, d])            # running partial output per row
    
    for each K, V column tile (cols [c..c+BLOCK_N]):
        load K_tile, V_tile  ([BLOCK_N, d])
        
        # 1. Compute attention scores for this tile
        S_tile = Q_tile @ K_tile.T / sqrt(d)        # [BLOCK_M, BLOCK_N]
        apply causal mask if needed (set future positions to -inf)
        
        # 2. Online softmax update
        m_tile = max(S_tile, axis=1)                # [BLOCK_M]
        m_new = max(m_running, m_tile)              # [BLOCK_M]
        
        # Rescale running stats to new max
        α = exp(m_running - m_new)                  # [BLOCK_M] — typically < 1
        ℓ_running = α * ℓ_running
        O_running = α[:, None] * O_running
        
        # Add new contribution
        P_tile = exp(S_tile - m_new[:, None])       # [BLOCK_M, BLOCK_N]
        ℓ_running += sum(P_tile, axis=1)
        O_running += P_tile @ V_tile                # [BLOCK_M, d]
        
        m_running = m_new
    
    # Final normalization
    O = O_running / ℓ_running[:, None]              # [BLOCK_M, d] — actual output
    write O to HBM at row positions [r..r+BLOCK_M]
    
    # For backward: save logsumexp (= m + log(ℓ)), not the full attention matrix
    write LSE = m_running + log(ℓ_running) to HBM   # [BLOCK_M] per Q tile

Read the algorithm carefully. Three things make it work:

  1. **Q tile is "pinned" in shared memory for the whole iteration**. K and V tiles stream through. This is the asymmetry — Q is loaded once per output tile; K and V are loaded once per K/V tile. Total HBM reads: O(T·d) instead of O(T²).
  2. **The (T, T) attention matrix never exists**. `S_tile` is a (BLOCK_M, BLOCK_N) tile that lives in registers/shared memory for the duration of one inner-loop iteration. It's discarded as soon as we update the running stats and output.
  3. **The output is built incrementally**. After processing all K/V tiles, `O_running / ℓ_running` gives the same result as `softmax(S) @ V` would have — exactly, not approximately. This is provable from the math above.

FlashAttention: pin one Q tile, stream K and V tiles through Q tile (pinned) [BLOCK_M, d] in shared mem stays for whole loop never re-read from HBM K, V tiles (stream) K1 K2 K3 [BLOCK_N, d] each For each K, V tile: S = Q @ K.T / √d P = exp(S - m_new) ℓ += sum(P) O += P @ V → patch with α = exp(m_old−m_new) O tile (built up) [BLOCK_M, d] in registers written to HBM at the end HBM traffic: naive vs FlashAttention (per Q tile) Naive: Q + K + V + S(write) + S(read) + P(write) + P(read) + O ≈ 5T·d + 2T² Flash: Q + K + V + O ≈ 4T·d (no T² term!) naive: T² term dominates at long context (T=8K → ~64 MB intermediate) flash: linear in T → same kernel scales to 32K, 128K, beyond this is what makes long-context training and inference tractable

Stare at this picture. The Q tile sits in shared memory for the whole iteration. K and V tiles stream through, one inner-loop iteration at a time. The (BLOCK_M, BLOCK_N) attention scores tile lives only in registers during that one iteration, then is discarded. The output tile is incrementally built up in registers, written to HBM only at the very end.

**The (T, T) intermediate never exists in HBM.** That's the entire trick.

## The Triton implementation (simplified)

Real production FlashAttention has many bells and whistles (varying head dims, dropout, alibi, paged KV, etc.). Here's the conceptual core stripped down. Read it next to the algorithm above.
    
    
    import triton
    import triton.language as tl
    
    @triton.jit
    def flash_attn_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
        stride_qb, stride_qh, stride_qm, stride_qd,
        # ... K, V, O strides similar ...
        H, M, N, D,
        softmax_scale,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        # Each program handles one (Q tile) for one (batch, head).
        pid_m = tl.program_id(0)         # Q tile index
        pid_bh = tl.program_id(1)        # batch * head index
    
        # Compute pointers to this Q tile's data.
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
    
        Q_block_ptr = Q_ptr + pid_bh * stride_qh \
                      + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    
        # Load Q tile ONCE — stays in registers/shared for whole loop.
        q = tl.load(Q_block_ptr, mask=offs_m[:, None] < M, other=0.)
    
        # Initialize running stats.
        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
        # Determine the range of K/V tiles to iterate over.
        if IS_CAUSAL:
            # With causal masking, only iterate over K/V tiles up to this Q tile.
            n_end = (pid_m + 1) * BLOCK_M
        else:
            n_end = N
    
        # Inner loop: stream K and V tiles.
        for start_n in range(0, n_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
    
            # Load K and V tiles.
            k = tl.load(...)            # [BLOCK_N, BLOCK_D]
            v = tl.load(...)            # [BLOCK_N, BLOCK_D]
    
            # 1. Attention scores for this tile.
            s = tl.dot(q, k.trans()) * softmax_scale    # [BLOCK_M, BLOCK_N], in fp32
    
            # 2. Apply causal mask if needed.
            if IS_CAUSAL:
                mask = offs_m[:, None] >= offs_n[None, :]
                s = tl.where(mask, s, -float("inf"))
    
            # 3. Online softmax update.
            m_ij = tl.max(s, axis=1)        # [BLOCK_M] — local max for this tile
            m_new = tl.maximum(m_i, m_ij) # [BLOCK_M] — new global max
    
            # Rescale running stats to new max.
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha
            acc = acc * alpha[:, None]
    
            # Add this tile's contribution.
            p = tl.exp(s - m_new[:, None])  # [BLOCK_M, BLOCK_N]
            l_i = l_i + tl.sum(p, axis=1)
            acc = acc + tl.dot(p.to(v.dtype), v)  # tensor cores fire here
    
            m_i = m_new
    
        # Final normalization.
        acc = acc / l_i[:, None]
        lse = m_i + tl.log(l_i)        # save for backward
    
        # Write output and LSE.
        tl.store(O_ptr + ..., acc.to(Q_ptr.dtype.element_ty), mask=...)
        tl.store(LSE_ptr + ..., lse, mask=offs_m < M)

This is a real Triton kernel — the production FlashAttention-2 kernel is essentially this with more autotune configs, more head dimension variants, and dropout/alibi/etc. options. The conceptual core fits in 50 lines.

Notable points to call out:

  * **Two`tl.dot` calls per inner iteration**: `q @ k.T` for scores, `p @ v` for the output update. Both fire tensor cores. The kernel is dominated by these two matmuls per K/V tile.
  * **fp32 throughout the algorithm body** : `m_i, l_i, acc` are all fp32. Inputs are loaded in bf16/fp16 but immediately participate in fp32 ops. Same recipe as M14 / M23.
  * **Causal masking is just`tl.where`**. Set future positions to `-inf` before the exp, and they contribute zero to the softmax. _Smart implementations skip K/V tiles entirely above the diagonal — no compute wasted on positions that will be fully masked._
  * **LSE is saved for backward**. We don't need to save the (T, T) attention matrix — just `logsumexp = m + log(ℓ)` per row. That's `O(T)` auxiliary storage, vs `O(T²)` for naive backward.

## The backward: rematerialize, don't store

Forward saved O and LSE. The backward pass reconstructs P from these — it doesn't read a saved attention matrix.

Recall `P = softmax(S)`. The forward saved `LSE = m + log(ℓ)`, the per-row log-sum-exp. We can recover P given S and LSE via:
    
    
    P_ij = exp(S_ij - LSE_i)        # numerically stable, exact

So the backward is structured similarly to forward:

  1. Iterate Q, K, V, O, LSE, dO tiles (we do an outer loop over K, V tiles for backward; the iteration pattern is different from forward for memory reasons).
  2. Recompute S = Q @ K.T / sqrt(d) for the current tile.
  3. Reconstruct P = exp(S - LSE) using the saved LSE — exact, not approximated.
  4. Compute the gradients: dV = P.T @ dO, dP = dO @ V.T, dS via softmax derivative, dQ and dK via standard matmul backprop.

The recompute happens entirely inside the backward kernel, in shared memory and registers. The (T, T) attention matrix is reconstructed tile-by-tile, used, and discarded — same pattern as forward. **This is exactly the activation-checkpointing trick from M12, but baked into the kernel:** trade some recomputation for huge memory savings, with the recomputation cheap because it stays on-chip.

Result: backward memory is also O(T) instead of O(T²). Both forward and backward avoid the quadratic blowup.

## FlashAttention versions: a quick history

The FlashAttention version line Version| Year| Key change| Speed  
---|---|---|---  
FlashAttention v1| 2022| Online softmax + tile-based fused kernel; the breakthrough.| ~2-4× over naive  
FlashAttention v2| 2023| Better parallelism over batch+heads; reduced non-matmul flops; better warp partitioning. Reference Triton implementation.| ~2× faster than v1  
FlashAttention v3| 2024| Hopper-specific (uses TMA for async loads, FP8 path, warp specialization). CUDA implementation.| ~1.5-2× faster than v2 on H100  
  
For practical use:

  * **Don't write FlashAttention from scratch unless you're learning**. The reference implementations (Tri Dao's `flash-attn` package, PyTorch's `scaled_dot_product_attention` backend) are heavily optimized.
  * **PyTorch ships FlashAttention**. `F.scaled_dot_product_attention(q, k, v, is_causal=True)` picks FlashAttention as the backend automatically when available. This is the right way to use it from training code.
  * **If you want to study the kernel** : read the v2 Triton implementation in the [Triton tutorials repo](https://github.com/triton-lang/triton). ~250 lines, well-commented, runs at near-cuBLAS speed.
  * **If you want to extend it** (custom mask patterns, sliding window, RoPE inside the kernel, etc.), forking the v2 Triton implementation is the standard starting point.

## Why FlashAttention's pattern generalizes

The "stream the big thing through, keep the small thing pinned in fast memory, never materialize the intermediate" pattern shows up everywhere now. M27 covers two prominent cases that descended directly from FlashAttention:

  * **Paged Attention** : for inference KV cache. Same kernel idea but K and V live in non-contiguous "pages" of GPU memory, allowing efficient memory management for variable-length conversations.
  * **Sliding-window attention** (Mistral, Gemma): the same kernel with a window mask — only K/V tiles within the window are visited, the rest are skipped entirely. Linear in T instead of quadratic, no algorithmic change beyond the iteration bounds.

And in less-attention contexts: similar techniques are used in **fused softmax + cross-entropy + KL kernels** , **fused layer-norm + dropout + residual** , **fused MoE routing**. Whenever you have an expensive intermediate that doesn't need to live in HBM, the FlashAttention pattern (online normalization + tile streaming + recompute backward) tells you how to eliminate it.

## Using FlashAttention from your training code

You almost never write a FlashAttention kernel directly. PyTorch's `scaled_dot_product_attention` is the front door:
    
    
    import torch.nn.functional as F
    
    # PyTorch picks the best backend automatically: FlashAttention if available,
    # memory-efficient attention as fallback, or the math (naive) backend.
    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        is_causal=True,           # lets the kernel skip masked tiles efficiently
        dropout_p=0.0,
        scale=None,              # default = 1/sqrt(d)
    )

You can inspect or restrict which backend is used:
    
    
    from torch.nn.attention import sdpa_kernel, SDPBackend
    
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Errors if FlashAttention isn't usable (e.g., wrong head dim, unsupported mask)

FlashAttention is finicky about supported configurations:

  * **Head dim** typically must be a multiple of 8 and ≤ 256 (varies by version).
  * **dtype** must be fp16 or bf16 (no fp32 path; fp8 in v3 only).
  * **Mask** : dense additive masks fall back to slower paths; `is_causal=True` uses the fast-causal optimization.
  * **Dropout** at non-zero rates is supported but adds overhead.

If your config doesn't qualify, PyTorch falls back to the next-best backend (memory-efficient attention, then the math/naive backend) — no error, just less performance.

#### Q&A; — About FlashAttention **Q:** If FlashAttention is exact, why isn't it the only attention kernel? **A:** FlashAttention has tight constraints: specific dtypes (fp16/bf16/fp8 only), bounded head dimensions, particular mask patterns. For unusual setups (very large head dim, custom mask shapes, fp32, exotic biases like ALiBi or RoPE-in-kernel), other backends or custom kernels are needed. PyTorch's `scaled_dot_product_attention` picks among Flash, memory-efficient, and math backends based on what works for your inputs. **Q:** Does FlashAttention save activations for backward? **A:** Yes — it saves Q, K, V, O, and LSE. That's O(T·d) per head, vs O(T²) for naive saving the attention matrix. The backward recomputes the attention scores from Q and K and reconstructs P from S and LSE — exact, not approximate. **Q:** What's the speedup on a typical training step? **A:** Depends on T, d, and the rest of the model. For a transformer at T=2048, attention is maybe 25-40% of step time naive; FlashAttention cuts it by ~3-4×, giving a net 1.3-1.5× step speedup. At T=8K, attention dominates more (because it's quadratic in T), so the speedup is bigger — sometimes 2-3× total. At T=32K, naive doesn't even fit; FlashAttention is the only option. **Q:** Why is FlashAttention important for inference, not just training? **A:** Inference with long prompts hits the same quadratic-memory wall. The KV cache for a 32K-token conversation × 32 layers × 32 heads × 128 head dim × 2 bytes = ~16 GB just for K and V. FlashAttention's tile streaming lets the prefill (reading the prompt) and the decode (generating one token at a time) both work efficiently with this much KV state. Combined with paged attention (M27), you can serve multiple long-context conversations on one GPU. **Q:** What's "TMA" in FlashAttention v3? **A:** Tensor Memory Accelerator — a Hopper-specific hardware unit that does asynchronous bulk loads from HBM into shared memory. It frees up the warp scheduler to do compute while loads are in flight, increasing tensor core utilization. v3 also uses warp specialization (some warps do loads, others do compute, others do softmax) for better latency hiding. Heavily Hopper-specific; A100 doesn't have TMA. **Q:** When does FlashAttention NOT help? **A:** Three cases. (1) Very short sequences (T < 256) — fixed kernel overheads dominate, and a simple matmul-attention-matmul flow can be competitive. (2) Very small batches or head counts where the kernel can't fill the GPU. (3) Heads with extremely large d (> 256) — falls back to other backends. For 95% of modern transformer training and inference, it helps; for those edge cases, the dispatcher (M19) routes elsewhere. 

## The takeaway pattern

FlashAttention is one specific algorithm, but the lessons generalize. The recipe:

  1. **Identify the memory-bound bottleneck.** Compute arithmetic intensity (M22). If it's well below the ridge point, the algorithm has a fusion opportunity.
  2. **Find the intermediate that lives in HBM but doesn't need to.** For naive attention, it's the (T, T) attention matrix. For other algorithms it might be the softmax output, an intermediate broadcast, etc.
  3. **Restructure the computation to keep that intermediate in shared memory.** Stream the big thing through; pin the small thing in fast memory; tile carefully so each tile fits in shared memory.
  4. **For incremental algorithms (softmax, normalization), find the running-state form.** Online softmax for attention; running mean+variance for normalization; running argmax for top-k.
  5. **For backward, save just enough to recompute** — typically a few O(T)-sized statistics rather than the full O(T²) intermediate.
  6. **Implement in Triton for portability** ; drop to CUDA only for hardware-specific tricks (TMA, warp specialization).

This recipe will produce many of the kernels you'll write or want to understand. FlashAttention is the canonical example because attention is everywhere and the speedup is large — but the technique is universal.

## Code Magnets: implement an online softmax pass

You're implementing the running-stats update for one inner-loop iteration of FlashAttention. Three magnets are wrong choices.

Arrange the magnets into the correct online softmax + output update.

m_ij = tl.max(s, axis=1) m_new = tl.maximum(m_i, m_ij) m_new = m_ij alpha = tl.exp(m_i - m_new) alpha = tl.exp(m_new - m_i) l_i = l_i * alpha acc = acc * alpha[:, None] p = tl.exp(s - m_new[:, None]) p = tl.softmax(s, axis=1) l_i = l_i + tl.sum(p, axis=1) acc = acc + tl.dot(p.to(v.dtype), v) m_i = m_new

show solution
    
    
    m_ij = tl.max(s, axis=1)
    m_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_new)
    l_i = l_i * alpha
    acc = acc * alpha[:, None]
    p = tl.exp(s - m_new[:, None])
    l_i = l_i + tl.sum(p, axis=1)
    acc = acc + tl.dot(p.to(v.dtype), v)
    m_i = m_new

The traps:

  * `m_new = m_ij`: forgets the running max from previous iterations. The correct update is `m_new = max(m_i, m_ij)` — keep the larger of running and local.
  * `alpha = tl.exp(m_new - m_i)`: sign flipped. The rescale is `exp(m_old − m_new)` — typically < 1 because m_new ≥ m_old. The flipped form would blow up.
  * `p = tl.softmax(s, axis=1)`: softmax over the tile gives _local_ probabilities, normalized to this tile's sum. We want `exp(s − m_new)` — unnormalized but referenced to the global running max. Normalization happens at the very end via `acc / l_i`.

The correct sequence: **find local max → update global max → rescale running stats with α → compute exp(s − global_max) → accumulate into ℓ and acc → record the new global max**. Each step depends on the one before; reordering breaks the math.

## Who does what?

Match each FlashAttention concept to its real role.

Concept

Real role

Online softmax

A. Compute exact softmax incrementally with a running (max, sum) — no need to see the whole row at once.

Q tile pinned, K/V streamed

B. Outer loop over Q tiles; inner loop over K/V tiles. Asymmetry that minimizes HBM reads.

(T, T) attention matrix never in HBM

C. Per-tile S lives only in registers/shared during one iteration; discarded after.

LSE saved for backward

D. log-sum-exp per Q row — O(T) instead of O(T²) auxiliary storage.

Recompute attention in backward

E. Reconstruct P = exp(S − LSE) on the fly inside the backward kernel.

F.scaled_dot_product_attention

F. PyTorch's high-level entry point that auto-picks FlashAttention when applicable.

FlashAttention v3 + TMA

G. Hopper-specific: tensor memory accelerator + warp specialization for max H100 throughput.

show solution

**Online softmax** → A  
**Q tile pinned, K/V streamed** → B  
**(T, T) attention matrix never in HBM** → C  
**LSE saved for backward** → D  
**Recompute attention in backward** → E  
**F.scaled_dot_product_attention** → F  
**FlashAttention v3 + TMA** → G 

The mental shortcut: _online softmax = streaming exact softmax, Q pinned + K/V streamed = the asymmetry, no (T,T) in HBM = the win, LSE = the only state we save, recompute in backward = M12 trick inside the kernel, F.scaled_dot_product_attention = the front door, v3 + TMA = Hopper-specific extras_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** Show that the online softmax algorithm produces exactly the same result as the standard two-pass version, for a row processed in two blocks of equal size.

show answer

Let x = [a₁, a₂; b₁, b₂] (block 1: a's; block 2: b's). Standard softmax:
    
    
    m = max(a₁, a₂, b₁, b₂)
    ℓ = exp(a₁−m) + exp(a₂−m) + exp(b₁−m) + exp(b₂−m)
    softmax_i = exp(x_i − m) / ℓ

Online algorithm:

After block 1: `m₁ = max(a₁, a₂)`, `ℓ₁ = exp(a₁−m₁) + exp(a₂−m₁)`.

After block 2: `m₂_local = max(b₁, b₂)`, `m_new = max(m₁, m₂_local) = m` (the global max).

`α = exp(m₁ − m_new)`. New `ℓ = α·ℓ₁ + exp(b₁−m_new) + exp(b₂−m_new)`.

Substituting: `α·ℓ₁ = exp(m₁−m_new) · [exp(a₁−m₁) + exp(a₂−m₁)] = exp(a₁−m_new) + exp(a₂−m_new)`. So:
    
    
    new ℓ = exp(a₁−m_new) + exp(a₂−m_new) + exp(b₁−m_new) + exp(b₂−m_new)
          = ℓ_standard ✓

Identical. The α rescaling reconstructs the contributions of block 1 referenced to the new global max — exactly what we'd have computed in a single pass. The whole point: _online softmax isn't an approximation; it's an algebraic refactoring_.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why does causal masking let FlashAttention skip K/V tiles entirely (not just mask their entries to −inf)?

show answer

Causal masking sets attention[i, j] = −inf for j > i (token i can't attend to future tokens). For an entire K/V tile that lies _entirely_ in the upper triangle (all positions j > all positions i in the current Q tile), every entry would be masked to −inf, contribute zero to the softmax, and zero to the output. _So skip that tile entirely_ — don't load K/V from HBM, don't run the matmul. The inner loop just iterates from `start_n=0` to `n_end = (pid_m + 1) * BLOCK_M` instead of all the way to N. **This gives causal FlashAttention ~2× speedup over non-causal at the same T** — half the K/V tiles are skipped. The optimization is in the loop bounds, not in the masking inside the kernel.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** Why does FlashAttention save Q, K, V, O, and LSE for backward instead of just LSE?

show answer

The backward needs to compute gradients dQ, dK, dV. The chain rule formulas involve P (attention matrix), S (scores), Q, K, V, O, and dO. We've already chosen not to save P or S (that's the whole point of FlashAttention). So we save the O(T·d) tensors that are inputs/outputs we can't reconstruct — Q, K, V, O — plus LSE which lets us recompute P = exp(S − LSE) given S, and S can be recomputed from Q and K. So:

  * Q, K, V: needed to recompute S = Q @ K.T / sqrt(d).
  * LSE: needed to reconstruct P from S without re-doing the softmax (which would need another running-max pass).
  * O and dO: needed for the dV = P.T @ dO and dP = dO @ V.T computations.

Total: ~5T·d bytes per head per layer. Same order as just storing the activations naively (which would also be ~T·d for Q, K, V), no quadratic blowup.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** A team adopts FlashAttention via `F.scaled_dot_product_attention` and sees big speedups for their T=4K transformer. They then increase head dim from 64 to 256, expecting another speedup. Instead they get a slowdown. What's likely happening?

show answer

FlashAttention's per-tile shared memory budget scales with head dim. At d=64, a (BLOCK_M=128, d=64) Q tile needs 128 × 64 × 2 = 16 KB, easy to fit alongside the K/V tile and stats. At d=256, the Q tile is 128 × 256 × 2 = 64 KB — plus the K/V tile of similar size, plus stats. The kernel may have to shrink BLOCK_M (fewer Q rows per program) to fit shared memory budget, reducing tensor-core utilization. Or the FlashAttention backend may not support d=256 at all (some versions cap at 128 or 192) and PyTorch falls back to the math backend. Diagnosis: profile with `nvidia-smi` \+ the PyTorch profiler — check which SDP backend is active via `sdpa_kernel` or the dispatcher trace. **Head dimensions matter for kernel performance, not just model quality** — most production transformers stick to d=64 or 128 partly because of kernel friendliness.

### What just happened?

  * **Naive attention is memory-bound** : it materializes a (T, T) intermediate in HBM, doing O(T²) memory traffic. Roofline analysis shows it's far below the ridge.
  * **The online softmax trick** : compute softmax exactly while streaming the input in blocks, by maintaining a running (max, sum) and rescaling old contributions when a new block's max exceeds the running max. `α = exp(m_old − m_new)` is the key term.
  * **FlashAttention applies online softmax to attention** : pin a Q tile in shared memory, iterate K and V in tiles, accumulate (running max, running sum-of-exps, partial output) for each Q row. The (T, T) attention matrix never exists in HBM.
  * **Memory traffic drops from O(T·d + T²) to O(T·d)**. Memory-bound attention becomes much closer to compute-bound.
  * **The kernel uses two`tl.dot` calls per inner iteration**: Q @ K.T for scores, P @ V for output update. Both fire tensor cores. fp32 accumulators throughout.
  * **Causal masking is a loop-bound optimization** — skip K/V tiles entirely above the diagonal, ~2× speedup over non-causal at same T.
  * **Backward saves Q, K, V, O, LSE** — O(T) auxiliary state instead of O(T²). Reconstruct P from S and LSE inside the backward kernel; same recompute trick as M12 activation checkpointing, baked into the kernel.
  * **FlashAttention versions** : v1 (2022) introduced the trick; v2 (2023) is the standard production reference, written in Triton; v3 (2024) is Hopper-specific with TMA and warp specialization.
  * **Use it through`F.scaled_dot_product_attention`**. PyTorch picks FlashAttention as the backend when configurations allow (fp16/bf16, head dim ≤ 256, supported masks).
  * **The pattern generalizes** : paged attention (M27), sliding-window attention, fused softmax+CE losses, fused norm+dropout — all use the same "stream the big thing, pin the small thing, never materialize the intermediate" recipe.
  * The reflex when designing a kernel: identify the memory-bound bottleneck, find the intermediate that doesn't need HBM, find the running-state form of any normalization, and use the recompute trick for backward.

Module 26 takes everything from Part VIII so far and applies it to **quantization**. We'll cover post-training quantization, GPTQ, AWQ, and the int4/int8/fp8 inference recipes — all hand-written kernel territory because efficient int4 matmul doesn't exist in cuBLAS. Then M27 covers inference systems (KV cache, paged attention, speculative decoding, vLLM-style serving). M28 closes the course with MoE and a frontier capstone.
