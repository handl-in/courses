# Module 46 — Attention Variants II: Head Topology & KV-Cache Engineering

# _Attention Variants II:_ head topology & KV-cache engineering

_Attention Trilogy · Module 46 · April 2026 currency_

— Axis C of the four-axis taxonomy: MHA → MQA → GQA → MLA → IHA → Slim Attention; the decoupled RoPE math that makes MLA work; KV cache compression at 2-bit (TurboQuant); the production decision tree

\--- 

M45 covered Axis A (semantic role) and Axis B (receptive field) of the attention-variants taxonomy. This module covers **Axis C — head topology** — how queries, keys, and values are structured across attention heads, and how that structure determines KV-cache cost.

Of all the axes, this is the most production-critical. Head topology is where 2026 frontier models actually differ in deployment economics. The progression — MHA → MQA → GQA → MLA → IHA → Slim Attention — is the story of _how to share KV between heads_ , and each step trades a different combination of memory, compute, and quality. By the end you'll know why GQA became the open-source default in 2024-2025, why DeepSeek's MLA outperformed it at scale, why IHA (Feb 2026) breaks the head-isolation barrier with pseudo-heads, and why Slim Attention can losslessly cut KV cache in half by reconstructing values from keys.

The other half of this module is **KV-cache engineering** — the techniques that compress, evict, and offload the KV cache after the head topology is chosen. These compose with topology choices: TurboQuant 2-bit quantization (April 2026, merged into vLLM) reduces a GQA model's KV by 4×; expected attention KV eviction reduces it further; combined with an MLA architecture the savings compound. _Production 2026 inference combines a head-topology choice with one or more cache-compression strategies_ ; understanding both is the practical literacy.

> **★ KEY IDEA**  
>  Axis C — head topology — is where queries, keys, and values are structured across heads. The production progression: **MHA** (every head has its own K, V — full quality, full KV cost) → **MQA** (all heads share one K, one V — minimal KV but quality drops) → **GQA** (groups of heads share K, V — the open-source default 2024-2026) → **MLA** (compress K, V into low-rank latent space, decompress on demand — DeepSeek V2/V3/V4) → **IHA** (pseudo-heads enable cross-head mixing — Feb 2026 frontier) → **Slim Attention** (K-cache only; reconstruct V from K — Mar 2025; lossless 2× compression). **The math that makes MLA work** : the naive "compress KV via low-rank projection" idea conflicts with RoPE because rotation must be applied to original-dimension K, not the compressed latent. **Decoupled RoPE** (DeepSeek's solution): split into a position-aware component (RoPE applied) and a position-free component (compressed). Cache only the latent + a small RoPE-component. Cache size: r_kv + d_qk_rope per token (vs h × (d_K + d_V) for MHA) — ~6× compression, often more. **IHA's mechanism** : for each of H attention heads, construct P pseudo-heads as learned linear combinations of all H original heads (typically P = H). Interleave pseudo-heads along the sequence dimension; apply standard attention on the expanded sequence; combine pseudo-heads back into a single head. Result: P² interaction patterns per head with O(H²P) parameter overhead — strictly more expressive than MHA. Improves Multi-Key retrieval on RULER by 10-20%; +5.8% GSM8K, +2.8% MATH-500 after fine-tuning. **Slim Attention's trick** : when projection matrices are square (d_K = d_V = d_model/h), V can be computed from K via V = K · W_V · W_K⁻¹. Cache only K; compute V on demand. Mathematically identical to standard attention; halves KV cache; doubles inference speed at long context. T5-11B gets 32× compression because its projection dimension exceeds embedding dimension. **KV-cache engineering compounds with head topology**. **TurboQuant** (ICLR 2026, merged into vLLM April 2026): online vector quantization via random rotations + Lloyd-Max scalar quantization; 2-bit values, 3-bit keys, 4× capacity. Available as `--kv-cache-dtype turboquant_3bit_nc` in vLLM. **Expected Attention** : estimates future query distribution, evicts unlikely keys. **Quest** , **H2O** : heuristic KV eviction. **InfiniGen** : KV cache offloading to CPU with prefetching. **Production 2026** : GQA + TurboQuant 3-bit, MLA + DeepGEMM, or any combination depending on the deployment. 

## Two new faces — head topology and the cache

K

Cache Custodian

"I store the K and V tensors that grow with every generated token. When sequence length hits 1M, I become the bottleneck."

During autoregressive generation, every new token requires the model to attend to all previous K and V tensors — so I store them. _I grow linearly with sequence length, multiplied by number of layers, multiplied by number of heads, multiplied by head dimension_. For a 70B model at 1M tokens with MHA, I'd need 320 GB just for me. **I'm why head topology matters in production** : every variant on Axis C is, fundamentally, a strategy for making me smaller. MQA shares heads to reduce me. GQA groups heads to reduce me partially. MLA compresses me into a latent space. Slim Attention stores only my K half. KV-cache engineering layers (TurboQuant, eviction, offloading) compress what's left. _Long-context inference is essentially the engineering of how to manage me_.

L

Latent

"I'm a low-rank compressed representation of K and V. Decompress on demand, cache only me — that's MLA's secret."

In MHA, the cache stores K and V at full dimension — for every head, every layer, every token. Most of that information is redundant; you don't need full-precision keys and values, you need their _essential information_. **I'm what's essential**. DeepSeek's MLA factorizes the K, V projections through a low-rank intermediate (me) — typically dimension ~512 instead of the full ~4096. Cache only me; up-project to full K, V on demand for attention computation. _The catch was RoPE_ — standard rotary embeddings need to be applied at full K dimension before storage, which would defeat my compression. **Decoupled RoPE** is the solution: split into position-aware (RoPE-applied, small) and position-free (compressed via me, large). Cache me + the small position-aware part. Total cache: ~93.3% reduction vs MHA. _I'm what makes 671B-parameter inference economically viable_.

## The mathematical baseline: MHA in detail

To make the variants legible, the baseline first. M11/M22 already covered MHA at high level; here we focus on the specific dimensions that matter for KV-cache analysis.
    
    
    def multi_head_attention(x, W_Q, W_K, W_V, W_O, n_heads):
        # x: [batch, seq, d_model]
        # d_model = 4096 (typical 7B model), n_heads = 32, d_head = 128
    
        # Project to Q, K, V — each [batch, seq, d_model]
        Q = x @ W_Q   # W_Q: [d_model, d_model] = [4096, 4096]
        K = x @ W_K   # W_K: [d_model, d_model] = [4096, 4096]
        V = x @ W_V   # W_V: [d_model, d_model] = [4096, 4096]
    
        # Reshape to [batch, n_heads, seq, d_head]
        Q = Q.view(batch, seq, n_heads, d_head).transpose(1, 2)
        K = K.view(batch, seq, n_heads, d_head).transpose(1, 2)
        V = V.view(batch, seq, n_heads, d_head).transpose(1, 2)
    
        # Apply RoPE to Q and K (M30 territory)
        Q, K = apply_rope(Q, K)
    
        # Standard attention per head
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_head)
        output = F.softmax(scores, dim=-1) @ V
    
        # Combine heads and project
        output = output.transpose(1, 2).view(batch, seq, d_model)
        output = output @ W_O
        return output

The KV-cache analysis. During autoregressive generation:

  * Each new token produces a Q vector (used once, not cached) and K, V vectors (cached for future tokens to attend to).
  * **Per-token KV cache cost** : `n_heads × (d_head + d_head) × n_layers × precision`.
  * For a typical 7B model (32 heads × 128 d_head × 32 layers × 2 bytes for FP16): **524 KB per token**.
  * At 100K tokens of context: 52 GB just for the cache.
  * At 1M tokens: 524 GB. _This doesn't fit on any single GPU_.

Frontier models in 2026 routinely target 1M+ contexts. The KV cache becomes the dominant memory cost; reducing it is the central problem of Axis C. Every variant below is a strategy for making this number smaller.

## MQA — Multi-Query Attention

Shazeer et al. (2019). The simplest reduction: **all attention heads share a single K and V projection**.
    
    
    def multi_query_attention(x, W_Q, W_K, W_V, W_O, n_heads):
        # Q still has full multi-head structure
        Q = x @ W_Q   # W_Q: [d_model, d_model] — full
    
        # K and V have only ONE head's worth of dimension
        K = x @ W_K   # W_K: [d_model, d_head] — shrunk!
        V = x @ W_V   # W_V: [d_model, d_head] — shrunk!
    
        # All Q heads attend to the same K, V
        Q = Q.view(batch, seq, n_heads, d_head).transpose(1, 2)
        K = K.view(batch, seq, 1, d_head).transpose(1, 2)   # 1 head only
        V = V.view(batch, seq, 1, d_head).transpose(1, 2)   # 1 head only
    
        # Broadcast: K, V get implicitly expanded for the n_heads dimension
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_head)
        output = F.softmax(scores, dim=-1) @ V
        return output

**KV cache cost reduction** : from `n_heads × 2 × d_head` per token to `1 × 2 × d_head` per token. For 32 heads, that's **32× compression**. Massive.

**The catch** : with all queries forced to attend to the same K, V, the model loses the per-head specialization that gives MHA its expressive power. Empirical result: notable quality degradation, especially on tasks requiring fine-grained pattern matching.

**Production use** : Falcon, PaLM, ChatGLM2 — popular in 2023-2024 when KV cache was the dominant memory pressure. Mostly displaced by GQA in 2024-2026 because GQA achieves most of the cache savings without as much quality loss.

## GQA — Grouped-Query Attention

Ainslie et al. (2023). The pragmatic compromise: **group query heads; each group shares one K, V head**.
    
    
    def grouped_query_attention(x, W_Q, W_K, W_V, W_O, n_heads, n_kv_heads):
        # n_heads = 32, n_kv_heads = 8 (typical) → 4 query heads per group
        Q = x @ W_Q   # W_Q: [d_model, n_heads * d_head]
        K = x @ W_K   # W_K: [d_model, n_kv_heads * d_head]
        V = x @ W_V   # W_V: [d_model, n_kv_heads * d_head]
    
        Q = Q.view(batch, seq, n_heads, d_head)
        K = K.view(batch, seq, n_kv_heads, d_head)
        V = V.view(batch, seq, n_kv_heads, d_head)
    
        # Each group of (n_heads / n_kv_heads) query heads attends to one KV head
        K = K.repeat_interleave(n_heads // n_kv_heads, dim=2)
        V = V.repeat_interleave(n_heads // n_kv_heads, dim=2)
    
        # Now standard attention with shared K, V across groups
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_head)
        output = F.softmax(scores, dim=-1) @ V
        return output

**KV cache cost reduction** : from 32 heads' worth of K, V to 8 heads' worth. **4× compression** — less than MQA's 32× but with much better quality preservation.

The intuition: _not all heads need to specialize on different keys_. Grouping query heads that compute similar attention patterns is a reasonable approximation; the quality loss is small.

**Production use** : GQA became the open-source default 2024-2026:

  * **Llama 2 70B** : 32 query heads, 8 KV heads (4-per-group)
  * **Llama 3 70B / Llama 4** : same 8-group pattern
  * **Mistral 7B** : 32 query heads, 8 KV heads
  * **Mistral Large, Mixtral** : GQA throughout
  * **Qwen 3.5/3.6 attention layers** (the non-GDN layers): GQA
  * **MiniMax M2.5** : deliberately classic GQA, called out in M44

_If you're not using MLA or hybrid attention, you're using GQA_. It's the safe, well-tooled, FlashAttention-compatible choice that maximizes quality given a KV-cache budget.

## MLA — Multi-Head Latent Attention

DeepSeek-AI 2024 (introduced in V2). The fundamental rethink: _instead of reducing the number of KV heads, compress the full-dimensional K and V into a low-rank latent space, cache that, and decompress on demand_.

### The basic mechanism

Rather than projecting input `x → K, V` directly, MLA factorizes the projections:
    
    
    # MHA standard: x → K, V directly
    K = x @ W_K   # [d_model, d_kv = n_heads * d_head]
    V = x @ W_V   # [d_model, d_kv]
    
    # MLA: x → c_KV (latent) → K, V via up-projection
    c_KV = x @ W_DKV   # W_DKV: [d_model, r_kv] — DOWN projection
    K = c_KV @ W_UK    # W_UK: [r_kv, d_kv] — UP projection
    V = c_KV @ W_UV    # W_UV: [r_kv, d_kv] — UP projection
    
    # Cache c_KV (small) instead of K, V (large)
    # r_kv ≈ 4 * d_head; in DeepSeek V2/V3, r_kv = 512 (vs full d_kv = 16384 for 128 heads)

The KV cache stores only `c_KV` — typically dimension r_kv = 512 — instead of full K and V at dimension d_kv = 4096-16384. **~32× to 64× cache reduction** compared to MHA, with quality matching MHA at scale.

But there's a problem.

### The RoPE paradox

RoPE (M30) rotates Q and K by position-dependent angles before the dot-product:

`attention_score(q_i, k_j) = (R(i) · q_i) · (R(j) · k_j)`

This rotation must be applied to _full-dimension K_ — there's no way to compose the rotation cleanly with a low-rank decomposition. If you cache c_KV and reconstruct K = c_KV @ W_UK, then apply RoPE, the RoPE rotation interferes with the up-projection in a way that prevents the algebraic absorption tricks that make MLA fast.

More fundamentally: _RoPE rotates by position, but c_KV is the same vector regardless of position_. Position information has to enter somewhere; if c_KV is purely position-free, where does it come in?

### The decoupled RoPE solution

DeepSeek's elegant fix: **split each query and key into two parts — one with RoPE applied, one without**.

MLA's decoupled RoPE: split position-aware vs position-free components Input x_i [d_model] Query path q_nope (no RoPE): q_nope = x_i @ W_dq @ W_uq[h] Position-free; absorbed via low-rank q_rope (RoPE applied): q_rope = RoPE(x_i @ W_qr[h]) Small dimension d_qk_rope ≈ d_head/2 Key path k_nope (no RoPE): c_KV = x_i @ W_dkv ← cache this k_nope = c_KV @ W_uk[h] (per-head) k_rope (RoPE applied, SHARED): k_rope = RoPE(x_i @ W_kr) ← cache this Single shared head; small dimension Attention computation: concatenate, then dot-product score = [q_nope; q_rope] · [k_nope; k_rope] = q_nope · k_nope + q_rope · k_rope What gets cached c_KV (per token): shape [r_kv] — typically 512 k_rope (per token, shared): shape [d_qk_rope] — typically 64 Total per token: 576 dims Comparison: vs MHA at 128 heads, d_head=128 MHA per token: 128 heads × 2 × 128 = 32,768 dims MLA per token: 576 dims Compression ratio: ~57× — > 6× over GQA-8

The MLA forward pass with decoupled RoPE:
    
    
    def mla_forward(x, W_dq, W_uq, W_dkv, W_uk, W_uv, W_qr, W_kr, n_heads):
        # DOWN-project x to compressed query and KV latents
        c_q = x @ W_dq                 # [batch, seq, r_q]
        c_KV = x @ W_dkv               # [batch, seq, r_kv]  ← CACHE THIS
    
        # UP-project to per-head queries (position-free part)
        q_nope = c_q @ W_uq            # [batch, seq, n_heads, d_qk_nope]
    
        # Compute RoPE-aware query parts (per-head)
        q_rope = x @ W_qr              # [batch, seq, n_heads, d_qk_rope]
        q_rope = apply_rope(q_rope)
    
        # UP-project to per-head keys (position-free part)
        k_nope = c_KV @ W_uk           # [batch, seq, n_heads, d_qk_nope]
    
        # Compute RoPE-aware key part (SHARED across heads)
        k_rope = x @ W_kr              # [batch, seq, d_qk_rope]  ← CACHE THIS (single head)
        k_rope = apply_rope(k_rope)
        k_rope = k_rope.unsqueeze(-2).expand(-1, -1, n_heads, -1)   # broadcast
    
        # UP-project values from latent
        v = c_KV @ W_uv                # [batch, seq, n_heads, d_v]
    
        # Concatenate q parts and k parts
        Q = torch.cat([q_nope, q_rope], dim=-1)   # [B, S, H, d_qk_nope + d_qk_rope]
        K = torch.cat([k_nope, k_rope], dim=-1)   # [B, S, H, d_qk_nope + d_qk_rope]
    
        # Standard attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        output = F.softmax(scores, dim=-1) @ v
        return output

The cached items are _only_ c_KV (compressed) and k_rope (small, shared across heads). At inference, both can be loaded and the up-projections re-run on the fly.

### Weight absorption: the inference trick

MLA admits a further inference optimization: the up-projections W_UK and W_UV can be _algebraically absorbed_ into adjacent matrices. The query projection and output projection get composed with W_UK and W_UV respectively; the actual inference path computes attention scores directly from c_KV without ever materializing K, V at full dimension.
    
    
    # Without absorption: cache c_KV, decompress K, V, compute attention
    # With absorption: precompute W_q_absorbed = W_uq @ W_uk^T
    # Now the query "lives" in the latent c_KV space; attention computed directly there
    
    W_q_absorbed = W_uq @ W_uk.T   # Composed offline
    score_nope = (c_q @ W_q_absorbed) @ c_KV.T
    # Now never materializes K_nope at full dimension

The result: MLA at inference behaves _like MQA_ in cache footprint (single small cache per token) but with _per-head specialization_ preserved through the query and output paths. This is the key economic insight — MQA-like deployment cost, MHA-like quality.

### MLA quality results

From DeepSeek V2 ablations:

  * **MLA matches or exceeds MHA** on most benchmarks at 128B+ scale.
  * **GQA underperforms MHA** in the same ablations.
  * **The compression-decompression step doesn't degrade information enough to matter** ; in some cases it acts as regularization.

The caveat (from Sebastian Raschka and others): **MLA only seems to win at 100B+ scale**. At smaller scales (under 100B), GQA is easier to tune and at least as good. _MLA is for trillion-parameter territory; GQA covers everything below_.

**Production use** : DeepSeek V2 (236B), DeepSeek V3 (671B), DeepSeek R1, DeepSeek V3.2, DeepSeek V4 (1.6T). Sarvam 105B uses MLA; Sarvam 30B uses GQA. Kimi Linear's full-attention layers use MLA (they replaced Qwen3-Next's gated attention with MLA specifically for the non-DGN layers). Ling 2.5 (1T MoE) uses MLA + Lightning Attention hybrid.

## IHA — Interleaved Head Attention (Feb 2026)

Duvvuri et al. (Meta, UT Austin, Berkeley, Harvard, MIT). The newest entry on Axis C — and one that breaks a fundamental MHA limitation.

### The compositional bottleneck

Standard MHA has H heads, producing H independent attention matrices. _No information flows between heads during attention computation_. After attention, the head outputs are combined via the output projection, but during attention itself, each head is isolated.

This becomes a problem for multi-step reasoning. Tasks requiring evidence aggregation across multiple parts of the context, with multiple intermediate transformations, need _composition_ of attention patterns. MHA can only represent k distinct patterns within one layer using O(k) heads — linear scaling with task complexity.

### The IHA mechanism

IHA introduces **pseudo-heads** : for each of H attention heads, construct P pseudo-heads (typically P = H) as learned linear combinations of _all H original heads' projections_.

IHA — pseudo-head construction enables cross-head mixing Step 1: Standard Q, K, V projections (H = 4 heads in this example) Q_1 Q_2 Q_3 Q_4 (same for K, V) Step 2: Each head is mixed into P=4 pseudo-heads (linear combination of all H originals) Head 1 → 4 pseudos: P1.1 P1.2 P1.3 P1.4 Each = α₁₁Q₁ + α₁₂Q₂ + α₁₃Q₃ + α₁₄Q₄ Head 2 → 4 pseudos: (similar mixing with α₂) Head 3 → 4 pseudos: (α₃) Head 4 → 4 pseudos: (α₄) Step 3: Interleave pseudo-heads along sequence dimension (sequence appears P× longer) Original tokens × P pseudos = N × P tokens (sequence appears 4× longer) Causal mask must be defined over the expanded sequence length Step 4: Apply standard attention; combine pseudo-heads back into single head per original Result: P² = 16 attention patterns per head with O(H²P) parameter overhead FlashAttention-compatible (mixing happens BEFORE attention, attention itself is standard)

The IHA forward pass:
    
    
    def iha_forward(x, W_Q, W_K, W_V, W_O, alpha_q, alpha_k, alpha_v, n_heads, P):
        # Step 1: Standard Q, K, V — same as MHA
        Q = x @ W_Q   # [B, N, H, d_head]
        K = x @ W_K
        V = x @ W_V
    
        # Step 2: Construct P pseudo-heads per head via learned linear combinations
        # alpha_q: [H, H, P] — for each head h, P pseudos as combos of all H originals
        Q_pseudo = torch.einsum("bnhd,hkp->bnhpd", Q, alpha_q)
        K_pseudo = torch.einsum("bnhd,hkp->bnhpd", K, alpha_k)
        V_pseudo = torch.einsum("bnhd,hkp->bnhpd", V, alpha_v)
        # Each is now [B, N, H, P, d_head]
    
        # Step 3: Interleave pseudo-heads along sequence dim — sequence appears N*P long
        Q_inter = Q_pseudo.flatten(start_dim=1, end_dim=2)   # Treat (N, P) as one extended seq
        K_inter = K_pseudo.flatten(start_dim=1, end_dim=2)
        V_inter = V_pseudo.flatten(start_dim=1, end_dim=2)
        # Each is now [B, N*P, H, d_head]
    
        # Step 4: Standard attention with expanded causal mask
        # Causal mask is over N*P, not N — every Pth position is the original token
        expanded_mask = build_iha_causal_mask(N, P)
        output_pseudo = flash_attention(Q_inter, K_inter, V_inter, mask=expanded_mask)
    
        # Step 5: Combine pseudo-heads back into a single head via output projection R
        output = output_pseudo.view(B, N, P, n_heads, d_head)
        output = (output * R).sum(dim=2)   # [B, N, H, d_head]
        output = output.flatten(2) @ W_O
        return output

**Why it's strictly more expressive** : with P pseudo-heads per head, each original head can express up to P² distinct attention patterns (P pseudo-queries × P pseudo-keys). MHA can express only H patterns total per layer. IHA at P=H gets H × H² = H³ effective patterns per layer with O(H²P) = O(H³) parameter overhead — substantial expressive gain.

**Theoretical results from the paper** : on synthetic Polynomial Filter tasks, IHA needs Θ(√k × n²) parameters where MHA needs Θ(k × n²) — quadratic to linear improvement in k (chain length). On synthetic CPM-3 (order-sensitive composition), IHA uses ⌈√N_max⌉ heads vs MHA's N_max — square-root reduction.

**Empirical results** :

  * RULER Multi-Key Retrieval: **10-20% improvement** over full attention at 4K-16K context.
  * GSM8K (after OpenThoughts fine-tuning): **+5.8% improvement** Maj@16 over full attention (54.2% vs 48.4%).
  * MATH-500: **+2.8% improvement** Maj@16 over full attention (18.4% vs 15.6%).
  * MBPP coding: Talking-Heads slightly ahead, IHA second.

**FlashAttention compatibility** : this is the key practical win. Unlike Talking-Heads attention (Shazeer 2020) which mixes heads at the level of attention logits — incompatible with FlashAttention's softmax-internal computation — IHA mixes heads _before_ the attention operator. The actual attention computation is standard; FlashAttention runs unmodified.

**Production status** : as of April 2026, IHA is published research (Feb 2026, Meta + UT Austin + UC Berkeley + Harvard + MIT) without major production adoption yet. Expect adoption in 2026-2027 frontier models given the FlashAttention compatibility and substantial RULER/reasoning gains.

## Slim Attention — when MHA still appears

Graef & Wasielewski (March 2025). A clever optimization specifically for models that still use MHA (rather than GQA/MQA/MLA) — primarily encoder-decoder models like Whisper, T5, and some older LLMs.

### The mathematical trick

Standard MHA caches both K and V — total cache width is `n_heads × (d_K + d_V)` per token. Slim Attention's observation: **when projection matrices are square (d_model = d_K = d_V), V can be exactly computed from K via** :
    
    
    # Standard MHA: V = X @ W_V, K = X @ W_K
    # If W_K is invertible: X = K @ W_K_inverse
    # Therefore: V = K @ W_K_inverse @ W_V = K @ W_KV
    
    # where W_KV = W_K_inverse @ W_V is precomputed offline
    
    def slim_attention(x, W_Q, W_K, W_KV_combined, W_O, n_heads):
        Q = x @ W_Q
        K = x @ W_K   # Cache only K!
    
        # Compute V from K on demand using precomputed W_KV
        V = K @ W_KV_combined   # V never stored — recomputed per attention call
    
        # Standard attention from here
        return attention(Q, K, V)

**Cache reduction** : from K + V to K only. **Lossless 2× compression** — mathematically identical to standard attention, no quality loss.

### Practical considerations

The trick has three subtleties:

  1. **Requires square projection matrices**. d_model must equal n_heads × d_K. This is true for vanilla MHA but not for GQA/MQA (where K projection is smaller) or MLA (where K projection is via low-rank latent).
  2. **RoPE incompatibility (sort of)**. If RoPE is applied to K between projection and dot-product, then cached K is post-RoPE, and `V = K @ W_KV` no longer holds because W_KV was derived assuming pre-RoPE K. Workaround: cache pre-RoPE K, apply RoPE on demand. Or use position-encoding schemes that don't rotate features (Alibi, T5's relative PE, FIRE) which fully support Slim Attention.
  3. **Bias terms** : most modern transformers (PaLM-style) drop biases from projections. For models with biases (Whisper, older models), the paper shows how to fold biases into adjacent layers in a mathematically equivalent way.

### Where the savings get bigger

For **encoder-decoder transformers** like Whisper and T5, the cache memory reduction is substantially larger than 2×:

  * **Whisper** : 8× cache reduction (encoder-decoder cross-attention has different math).
  * **T5-11B** : **32× cache reduction** because its MHA projection dimension exceeds embedding dimension.
  * Inference speedup: up to **5× for token generation at batch size 64** on Whisper.

The compression factor is `c = d_cache / d_model` where `d_cache = h_KV × (d_K + d_V)`. Larger c → larger savings. Typical c is 2 (Slim → 2× savings); models with non-square projections can have c much larger.

**Production use** : speech-to-text systems (Whisper deployment), translation models (T5 derivatives), time-series forecasting (Amazon Chronos models). For modern LLMs using GQA/MLA, Slim Attention doesn't directly apply — but _its mathematical idea_ (reconstruct one cached tensor from another) shows up in MLA's weight absorption.

## The decision tree: which head topology when?

Putting Axis C choices together for a 2026 production engineering decision:

Head topology decision tree, 2026 production Pretraining new model? or fine-tuning existing? Pretraining: scale >100B? pretraining MLA DeepSeek lineage 5-7× cache vs GQA yes >100B GQA (8 KV heads) Llama, Mistral, Qwen Default for <100B no <100B IHA (experimental) If reasoning is target +5.8% GSM8K reasoning-focused Use existing model topology fixed fine-tuning Apply KV-cache engineering • TurboQuant 3-bit: 4× capacity (vLLM) • Expected Attention: KV eviction • InfiniGen / Quest: KV offloading Edge case: encoder-decoder MHA • Whisper, T5, Chronos • Apply Slim Attention for 2-32× cache • Lossless; no retraining needed Production reality: 90% of teams use GQA + TurboQuant 3-bit; 10% use MLA + DeepGEMM Both work; choice depends on scale and where you sit on the open-weight ecosystem map

## KV-cache engineering: compounding with head topology

Once head topology is chosen, KV-cache engineering compresses what's left. April 2026 production stack:

### TurboQuant — the new vLLM default for 2-bit/3-bit KV quantization

ICLR 2026 (Rotem et al.). Merged into vLLM April 15, 2026 (PR #38479). The flagship 2026 KV-cache compression:

  * **Online vector quantization** — no calibration, no fine-tuning, no model-specific tuning
  * **PolarQuant transform** : random rotations (Walsh-Hadamard transforms) to make distributions more amenable to scalar quantization, then Lloyd-Max scalar quantization
  * **Asymmetric bit allocation** : 3 bits for keys, 2 bits for values (values tolerate compression better)
  * **4× capacity vs FP16 KV cache** (3-bit configuration); higher with mixed-mode quantization
  * **Triton kernels + CUDA fallback** for on-the-fly dequantization during attention

vLLM usage:
    
    
    # Available in vLLM after April 15, 2026 (PR #38479 merged)
    vllm serve Qwen/Qwen3-4B --kv-cache-dtype turboquant_3bit_nc
    
    # Available cache types:
    #   turboquant_3bit_nc  — 3-bit no calibration (default recommendation)
    #   k8v4                — 8-bit keys, 4-bit values (asymmetric)
    #   4bit_nc             — 4-bit no calibration
    #   k3v4_nc             — 3-bit keys, 4-bit values

The trade: **2.6-4.9× KV capacity at the cost of 35-43% of decode throughput**. Worth it when KV is the memory bottleneck (long context, high concurrency); not worth it when throughput is the bottleneck. The honest reading: _it's a substantial win for long-context serving but isn't free_.

Composition with head topology: TurboQuant 3-bit on top of GQA-8 gives ~4× × 4× = 16× total cache reduction over MHA-FP16. On top of MLA, it gives ~4× × 6× = 24×.

### KV cache eviction

Don't compress; _discard_. Estimate which tokens are unlikely to be attended to in the future, evict them.

  * **Quest (Tang et al. 2024)** : query-aware on-demand KV fetching — keep all KV in CPU, prefetch to GPU based on query similarity. Lazy eviction.
  * **H2O (Heavy-Hitter Oracle)** : keep tokens with high accumulated attention scores; evict the rest. Heuristic but effective.
  * **Expected Attention** : estimate the future query distribution, evict keys with low expected future attention. More principled than H2O.
  * **SnapKV** : cluster KV by attention pattern; keep cluster representatives.

Tradeoff: eviction is _lossy_. If you evict a key that turns out to be needed later, the model performance degrades. Quality-cost tradeoff that depends on the specific eviction policy and workload.

### KV cache offloading

Move KV cache from GPU memory to CPU memory (or even disk), prefetch back when needed.

  * **InfiniGen (Lee et al. 2024)** : lightweight rehearsal using partial model weights to predict which KV blocks are needed; prefetch to overlap with computation. Offline SVD for efficient prediction.
  * **SlimInfer** : layer-wise hidden-state pruning naturally enables predictor-free prefetching.
  * **FlexGen** : extreme offloading for long-context inference on resource-constrained hardware.

Tradeoff: offloading is _lossless_ (everything's preserved) but PCIe bandwidth becomes the bottleneck. Useful when GPU memory is the constraint and you have time/bandwidth to swap.

### The full 2026 inference stack

A typical April 2026 long-context inference deployment combines:

  1. **Head topology** (chosen at training time): GQA-8 or MLA. Determines baseline cache size.
  2. **FlashAttention 3 / 4** (M47 territory): never materialize the full attention matrix. Determines compute kernel.
  3. **PagedAttention** (vLLM): block-based KV cache layout. Eliminates fragmentation.
  4. **TurboQuant 3-bit** : quantize the KV cache values. 4× capacity.
  5. **Expected Attention or H2O** (optional): evict low-attention tokens for very long contexts.
  6. **InfiniGen** (optional): offload cold KV pages to CPU when GPU memory is exhausted.

At 1M tokens with a Llama 3 70B (GQA-8), the resulting cache is ~6.7 GB instead of the naive 320 GB. Concrete production numbers from the TurboQuant deployment ecosystem.

## Production zoo: Axis C in April 2026 frontier models

Cross-referencing M44's frontier zoo with the Axis C choices covered in this module:

Axis C choices across April 2026 frontier models Model| Head topology| KV cache mechanism  
---|---|---  
**Llama 4 Scout / Maverick**|  GQA| FA3/FA4 + PagedAttention  
**DeepSeek V4-Pro / V4-Flash**|  MLA + CSA+HCA hybrid (M44)| DeepGEMM with FP4 Lightning Indexer  
**GLM-5.1**|  MLA-style with DSA| FA4 + custom kernels  
**Qwen 3.5 / 3.6**|  GQA in attention layers; GDN in linear layers| FA3 + standard PagedAttention  
**Kimi Linear / K2.6**|  MLA in full-attention layers; KDA in linear layers| FA4 + custom hybrid kernel  
**Ling 2.5**|  MLA + Lightning Attention hybrid| FA4 + custom hybrid kernel  
**Mistral Large**|  GQA| FA3 + PagedAttention  
**MiniMax M2.5**|  GQA (deliberately classic)| FA3 + PagedAttention  
**Sarvam 105B**| **MLA**|  FA3  
**Sarvam 30B**| **GQA**|  FA3  
**Whisper / T5**|  MHA (encoder-decoder; supports Slim Attention retrofit)| Slim Attention available; 2-32× compression  
**IHA-based models**|  IHA (experimental, Feb 2026)| FlashAttention compatible  
**Closed (GPT-5.5, Claude 4.7, Gemini 3.1)**|  Undisclosed (likely GQA-derivatives)| Undisclosed  
  
Three observations:

  * **The 100B threshold is real**. Sarvam ships 105B with MLA and 30B with GQA — same family, different head topologies for different scales. Validates the empirical finding.
  * **Hybrid stacks combine head-topology choices per layer type**. Qwen 3.5/3.6 uses GQA in its full-attention layers and GDN in its linear-attention layers (M45). Kimi Linear uses MLA in full-attention + KDA in linear. The Axis C choice is layer-specific, not global.
  * **IHA is the entry to watch**. Feb 2026 publication; FlashAttention compatible; substantial reasoning gains. By Q3-Q4 2026 expect frontier models incorporating it as a head-topology choice.

#### Q&A; — About head topology and KV-cache engineering **Q:** Why does GQA underperform MHA in DeepSeek's ablations but everyone uses GQA anyway? **A:** Three reasons that compound. (1) **The DeepSeek ablations were at MoE+frontier scale**. At 100B+ parameters, MLA's compression-decompression overhead is a small fraction of total compute, and its slight regularization effect helps. At smaller scales, the ratio is different — GQA's simpler structure is easier to optimize for. (2) **Engineering ecosystem matters**. GQA was published in 2023; FlashAttention, vLLM, PyTorch's scaled_dot_product_attention all have first-class GQA support. MLA is more recent and less tooled — the inference path requires custom kernels for the absorption tricks, and frameworks are still catching up. (3) **Risk tolerance**. GQA is well-understood; MLA is newer. Most teams default to the safer choice. The empirical pattern (Sarvam shipping both, with GQA at 30B and MLA at 105B; everyone else using GQA at smaller scales) reflects this. _If you're under 100B and don't have DeepSeek-level kernel engineering, GQA is the practical answer; if you're at frontier MoE scale, MLA is the production choice_. **Q:** Why does decoupled RoPE work? It feels like a hack — why split into two parts? **A:** Three nested reasons. (1) **The mathematical issue is real** : RoPE rotates K by position, which is incompatible with low-rank K decomposition. There's no clean way to compose rotation with up-projection that preserves both compression and position-awareness. (2) **The split decouples concerns** : position information lives in a small position-aware dimension (k_rope, ~64 dims, shared across heads). Content information lives in the large position-free dimension (k_nope, via c_KV). The two are concatenated for attention computation but cached separately. _This is a clean factorization of the original tensor_. (3) **The position-aware dimension is shared**. Unlike per-head k_nope, k_rope is single-head — the same RoPE-rotated vector is used by all heads. This is consistent with the empirical observation that position information is more "shared" across heads than content. _The hack works because position genuinely is more fungible across heads than content_. The decoupled RoPE construction is somewhat baroque mathematically but reflects real structural properties of attention. It's not a workaround — it's a discovery about what factorization the architecture wants. **Q:** If IHA is so much better at reasoning, why isn't it in production yet? **A:** Three reasons related to research-to-production timing. (1) **It's only 2 months old** as of the publication of this module. Production training runs take months; teams that want IHA in their next model are starting now, with releases targeted for Q3-Q4 2026. (2) **The interleaving requires expanded sequence lengths**. With P=H, the sequence appears H× longer during attention. This means H× more activation memory during training — a real cost. The IHA paper validates at 1.5B parameters; scaling to frontier (300B+) is engineering work nobody has published yet. (3) **Hybrid architecture compatibility**. Modern frontier models combine attention with MoE, with sparse attention, with hybrid linear layers. IHA needs to compose with these; the interactions haven't been validated. _The window between research publication and production adoption is typically 6-12 months for fundamental architectural changes_. Expect IHA in production by late 2026 if the empirical results hold up; if they don't generalize beyond the paper's scale, it stays as a research curiosity. The history of attention variants is full of both outcomes. **Q:** Why does Slim Attention only matter for older / encoder-decoder models? **A:** Slim Attention's central trick (compute V from K via V = K @ W_KV) requires _square projection matrices_. Specifically, it needs the projection from input to K to be invertible — which means d_model = n_heads × d_K with no sharing across heads. This is true for vanilla MHA. But in 2026 production: 

(a) **GQA breaks it** — fewer KV heads than query heads means W_K is not square; the inversion doesn't work cleanly.

(b) **MQA breaks it** — only one KV head; same issue.

(c) **MLA breaks it** — K comes from low-rank projection through c_KV; the structure is fundamentally different.

So Slim Attention's natural habitat is full-MHA models, which by 2026 are mostly the older / specialized ones: Whisper (speech), T5 (encoder-decoder NLP), Chronos (time series). For these, Slim Attention provides a substantial drop-in win at zero quality cost. _For modern decoder-only LLMs (Llama, DeepSeek, Qwen, Mistral, Claude, GPT, Gemini), it doesn't apply directly_. The intellectual contribution lives on in MLA's weight absorption (a similar "reconstruct from compressed" idea) but the specific Slim Attention method is tied to MHA architectures.

**Q:** What's the practical limit on KV cache compression — can we keep going below 2 bits? **A:** Probably yes, but with diminishing returns and increasing risk. Three considerations. (1) **The TurboQuant authors note 4-bit is the "sweet spot"** — quality essentially indistinguishable from FP16 for 3B+ models. At 3-bit, quality starts degrading on smaller models. At 2-bit, more degradation. The empirical curve shows the marginal compression vs marginal quality curve flattening rapidly below 4 bits. (2) **Information-theoretic floor** : there's a minimum number of bits per KV element to preserve essential information. Random projections (used in TurboQuant's PolarQuant) help by spreading information uniformly across dimensions, but you can't go arbitrarily low. (3) **Specific tasks matter** : long-context retrieval (NIAH-style) is more sensitive to KV precision than general-purpose generation. Aggressive compression (1.5-bit, 1-bit) might work for chat but fail on retrieval. _The realistic floor in 2026-2027 is probably 1.5-2 bits_. Going below that requires fundamentally different techniques — perhaps cache eviction (drop tokens entirely) or learned compression (autoencoder-like KV compression trained alongside the model). Both are active research areas. **Q:** How do head topology choices interact with M44's MoE patterns? **A:** Mostly orthogonally, but with some interesting interactions. **MoE replaces the FFN** , head topology shapes the attention block. They live in different parts of the transformer block. So you can mix any head topology with any MoE pattern: DeepSeek V4 = MLA + alternating MoE/dense. Llama 4 Maverick = GQA + alternating MoE. Qwen 3.6 = GQA in attention layers + MoE in FFN layers. **The interesting interactions** : (a) _MLA's weight absorption interacts with MoE expert routing_ — the absorbed query projection depends on which up-projection is used, and per-expert routing introduces variability. DeepSeek's V3 paper handles this; the interaction is non-trivial. (b) _IHA's expanded sequence interacts with expert capacity_ — the P× longer sequence during attention means more tokens routed to each expert. Capacity factors must be retuned. (c) _GQA's grouping doesn't interact with MoE routing_ — they live in different blocks. _The general rule_ : for production engineering, treat head topology and MoE pattern as independent choices. Compose them as needed. Tooling (Megatron-Core, vLLM) supports the combinations. 

## Code Magnets: implement MLA forward with decoupled RoPE

You're writing the MLA forward pass with decoupled RoPE. Three magnets are wrong choices.

Arrange the magnets to compute MLA attention output for a single token.

def mla_forward(x, weights, n_heads): c_q = x @ weights.W_dq c_KV = x @ weights.W_dkv # cache this q_nope = c_q @ weights.W_uq q_rope = apply_rope(x @ weights.W_qr) q_rope = x @ weights.W_qr # forgot RoPE k_nope = c_KV @ weights.W_uk k_rope = apply_rope(x @ weights.W_kr) # cache this — single shared head k_rope = apply_rope(c_KV @ weights.W_kr) # rope on latent — wrong v = c_KV @ weights.W_uv k_rope_per_head = k_rope.unsqueeze(-2).expand(-1, -1, n_heads, -1) Q = torch.cat([q_nope, q_rope], dim=-1) K = torch.cat([k_nope, k_rope_per_head], dim=-1) K = k_nope + k_rope_per_head # element-wise — wrong scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1)) output = F.softmax(scores, dim=-1) @ v return output

show solution
    
    
    def mla_forward(x, weights, n_heads):
        c_q = x @ weights.W_dq
        c_KV = x @ weights.W_dkv  # cache this
        q_nope = c_q @ weights.W_uq
        q_rope = apply_rope(x @ weights.W_qr)
        k_nope = c_KV @ weights.W_uk
        k_rope = apply_rope(x @ weights.W_kr)  # cache this — single shared head
        v = c_KV @ weights.W_uv
        k_rope_per_head = k_rope.unsqueeze(-2).expand(-1, -1, n_heads, -1)
        Q = torch.cat([q_nope, q_rope], dim=-1)
        K = torch.cat([k_nope, k_rope_per_head], dim=-1)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        output = F.softmax(scores, dim=-1) @ v
        return output

The traps:

  * `q_rope = x @ weights.W_qr # forgot RoPE`: skips the RoPE rotation that gives this branch its purpose. The whole point of the decoupled RoPE design is that _q_rope_ carries position information via RoPE, while _q_nope_ is position-free. Without RoPE applied, q_rope is just another position-free projection — which means the model has no way to encode position at all. Attention scores would no longer depend on token position; long-context tasks would fail catastrophically. The RoPE application is mandatory for the position-aware branch.
  * `k_rope = apply_rope(c_KV @ weights.W_kr) # rope on latent — wrong`: applies RoPE to the wrong projection. Decoupled RoPE specifically computes k_rope from the _raw input x_ , not from the latent c_KV. The reason: if k_rope were computed via the latent, it would suffer from the same compression issue that motivated decoupling in the first place. The whole architectural insight is that position-information flows through a separate, uncompressed path. Computing k_rope through c_KV defeats the purpose; the decoupling becomes structural fiction.
  * `K = k_nope + k_rope_per_head # element-wise — wrong`: adds the two key components instead of concatenating them. The two components have different semantic roles (k_nope is content-aware via the latent, k_rope is position-aware) AND potentially different dimensions (d_qk_nope vs d_qk_rope). Adding them mixes information that should remain separable; concatenation along the feature dimension preserves the structure: `score = Q · K = q_nope · k_nope + q_rope · k_rope`. Element-wise addition gives a different (and wrong) score formula. The concat is what makes the math work out to MHA-equivalent attention with split position handling.

The pattern: **down-project to latent, up-project for nope, project + RoPE for rope, concatenate (not add), standard attention**. Each step has a precise role; the most common implementation bugs are forgetting RoPE on q_rope (silent quality loss for long-context), routing RoPE through the latent (defeats decoupling), and using add instead of concat (breaks the dot-product structure).

## Who does what?

Match each Axis C variant to its real role.

Concept

Real role

MHA (Multi-Head Attention)

A. Vanilla; every head has its own K, V; full quality, full KV cache cost.

MQA (Multi-Query Attention)

B. All heads share one K, one V; minimal cache but quality drops; Falcon, PaLM-era.

GQA (Grouped-Query Attention)

C. Groups of heads share K, V; the open-source default 2024-2026; Llama, Mistral, Qwen.

MLA (Multi-Head Latent Attention)

D. Compress K, V into low-rank latent; decompress on demand; decoupled RoPE; DeepSeek V2/V3/V4.

Decoupled RoPE

E. Split into position-aware (RoPE applied, small) + position-free (compressed via latent, large).

IHA (Interleaved Head Attention)

F. Pseudo-heads enable cross-head mixing; P² patterns per head; Feb 2026, +5.8% GSM8K.

Slim Attention

G. K-cache only; reconstruct V from K via precomputed W_KV; lossless 2× compression for MHA.

show solution

**MHA** → A  
**MQA** → B  
**GQA** → C  
**MLA** → D  
**Decoupled RoPE** → E  
**IHA** → F  
**Slim Attention** → G 

The mental shortcut: _MHA every head solo, MQA shares all, GQA shares groups, MLA compresses to latent, decoupled RoPE splits position from content, IHA mixes heads via pseudos, Slim reconstructs V from K_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team is deploying a 405B-parameter model at 1M context. They have a fixed budget of GPU memory. Walk through their head topology + cache compression decision.

show answer

The math is the deciding factor here. Step through:

(1) **Compute baseline KV cache for MHA at 1M context**. 405B model has approximately 80 layers × 128 heads × 128 d_head × 2 bytes = 2.6 MB per token in MHA. At 1M tokens: **2.6 TB of KV cache**. Doesn't fit on any single H100 (80GB) or B200 (192GB). Doesn't even fit on a GB200 NVL72 system (72 × 192GB = 13.8 TB total) once you account for batching multiple users.

(2) **Try GQA-8**. 4× compression. 650 GB. Still doesn't fit; needs distributed serving across many GPUs even for batch size 1.

(3) **Try MLA**. 5-7× compression over GQA-8. ~100-130 GB at 1M context. Fits on a single B200 or B300 with headroom for batching. _This is the production answer for 405B at 1M context._

(4) **Add TurboQuant 3-bit on top**. Another 4× compression. ~25-33 GB. Fits comfortably; allows substantial batching for multiple concurrent users. The decode throughput penalty (35-43% reduction) is acceptable for long-context use cases where the alternative is "doesn't run at all."

(5) **Optional: KV eviction (Expected Attention or H2O)**. For the most concurrent users, evict tokens with low expected attention. Adds another 2-4× capacity at some quality risk. Worth it if your traffic pattern is high-concurrency.

(6) **Hardware deployment** : with MLA + TurboQuant 3-bit, a single B300 NVL72 rack can serve hundreds of concurrent 1M-context sessions. Without these compressions, even a single session would require multiple racks.

**The recommendation** : pretrain (or use a model that pretrained) with MLA. Deploy with TurboQuant 3-bit KV cache. Add Expected Attention eviction if concurrency is high. Skip KV offloading unless GPU memory is truly insufficient (PCIe bandwidth becomes the bottleneck).

The general lesson: **at frontier scale × frontier context length, head topology compresses by 5-7× and cache engineering compresses by another 4-8×, totaling 25-50× over MHA-FP16**. This is what makes 1M context economically viable. Without these compressions, 1M context exists only in research papers.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Walk through why MLA's weight absorption makes inference cheaper than the naive MLA forward pass would suggest.

show answer

The naive MLA inference path (what the forward-pass code shows) is:

**Naive path** : cache c_KV; at attention time, decompress to K_nope = c_KV @ W_uk and V = c_KV @ W_uv (full dimension); compute attention with K_nope, V at full dimension.

This works but materializes K_nope and V at full dimension during every attention call — a substantial compute cost.

**The absorbed path** : precompute W_q_absorbed = W_uq @ W_uk^T offline (W_uq is the query up-projection, W_uk is the key up-projection). Now at inference time:
    
    
    score_nope = (c_q @ W_q_absorbed) @ c_KV.T

Note what happened: we never materialized K_nope. The query was projected into the latent space (dimension r_kv ≈ 512) via W_q_absorbed, and the dot product was computed directly with c_KV. **Attention computation now happens at the latent dimension, not the full K dimension**.

For values, similar: W_o (output projection) can be composed with W_uv to give W_o_absorbed = W_uv @ W_o. The attention output is computed at the latent dimension and projected to output via W_o_absorbed in a single step.

The savings:

(1) **Compute** : attention happens at latent dimension (~512) instead of full K dimension (~16,384). 32× reduction in attention FLOPs at 128 heads × 128 d_head.

(2) **Memory bandwidth** : only c_KV (small) is loaded from cache, not K and V at full dimension. ~57× reduction in memory traffic.

(3) **Effective speedup** : in the DeepSeek V2 paper, MLA at inference is reported at 5.76× generation speed of MHA — exactly the kind of speedup you'd expect from this compute + bandwidth reduction.

The trick that makes weight absorption work: _matrix multiplication is associative_. (q @ W_uq) @ (c_KV @ W_uk).T can be reordered to q @ (W_uq @ W_uk.T) @ c_KV.T. The middle matrix W_q_absorbed is precomputed offline; the runtime path becomes much cheaper.

The general lesson: **algebraic absorption is what makes MLA practical for inference**. Without it, MLA would just be a "smaller cache" architecture; with it, MLA actually accelerates inference. _Production MLA implementations (DeepGEMM in DeepSeek's stack) bake the absorption in_ ; naive implementations would miss most of the benefit.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** A team is fine-tuning Qwen3-VL (which uses GQA) for their production application. They want long-context (256K) inference but are GPU-memory-constrained. Sketch the plan.

show answer

The decision tree from the SVG points to "use existing model + KV cache engineering." Step through:

(1) **Head topology is fixed** : GQA-8 (whatever the original Qwen3-VL ratio is). Can't change without retraining from scratch, which the team isn't doing.

(2) **Apply TurboQuant 3-bit KV quantization**. Available in vLLM as of April 15, 2026:
    
    
    vllm serve qwen/qwen3-vl-X --kv-cache-dtype turboquant_3bit_nc

4× capacity over FP16 GQA. Expected quality impact: minimal for 3B+ models on most workloads. Validate on the team's specific tasks (the Qwen3-VL multimodal evals) — if quality holds, ship it.

(3) **Add Expected Attention or H2O eviction** for very long contexts. At 256K, low-attention tokens can be evicted with minimal quality loss. Adds another 2-3× effective capacity. Risk: rare retrieval tasks might miss tokens that were evicted — test on the workload before committing.

(4) **If still memory-bound, KV offloading**. InfiniGen-style with prefetching. Lossless but PCIe bandwidth becomes the bottleneck — typically only worth it if the team has large CPU memory and tolerable latency budget.

(5) **Estimate the resulting cache size**. For a 7B Qwen3-VL with GQA-8 at 256K context:

  * MHA-FP16 baseline: ~32 GB
  * GQA-8 (already used): ~8 GB
  * \+ TurboQuant 3-bit: ~2 GB
  * \+ 50% Expected Attention eviction: ~1 GB

1 GB of KV cache fits on any modern GPU with comfortable headroom for batching.

(6) **Validate quality on production tasks**. M37's eval rigor applies. Specifically test:

  * Long-context retrieval (does the model still find specific facts buried in 256K?)
  * Multi-modal reasoning (does the vision-language alignment survive KV compression?)
  * End-to-end task accuracy on the team's specific benchmarks

(7) **Compare alternatives** : switching to a model with MLA (like a Kimi or DeepSeek variant) would give better cache compression natively, but switching models has a cost (engineering, prompt tuning, eval re-validation). For most teams, optimizing the existing Qwen3-VL with cache engineering is the right path.

The general lesson: **head topology is fixed once you've chosen a model; KV cache engineering is the lever you have at deployment time**. TurboQuant + eviction + offloading compose; pick the combination that meets your memory budget and quality bar.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Forecast: what's the most likely successor to MLA at frontier scale by 2027?

show answer

Three candidate directions, each with arguments for and against:

**Candidate 1: MLA-extended (deeper compression hierarchies)**. Argument for: MLA proved low-rank compression works at scale; deeper hierarchies (compress c_KV further into c_meta) might extend the gains. Argument against: each level of compression adds compute overhead at decompression; the tradeoff curve flattens past some point. _Likely_ : incremental MLA refinements continue, but no qualitative leap.

**Candidate 2: IHA-derivatives (cross-head mixing as standard)**. Argument for: IHA's expressivity gains are substantial (10-20% on RULER, +5.8% GSM8K) and FlashAttention compatible. The paper's parameter efficiency theory (sqrt(k) heads vs k heads for compositional tasks) suggests this is fundamental, not incremental. Argument against: scaling IHA to frontier scale hasn't been validated; the expanded sequence length during attention is a real engineering cost. _Likely_ : IHA-style cross-head mixing becomes a standard option, possibly composed with MLA's compression.

**Candidate 3: DeepSeek V4-style hybrid (CSA+HCA + MLA)**. Argument for: DeepSeek V4 already shipped this combination — interleaved CSA (4× compress + sparse selection) and HCA (128× compress + dense) layers, with MLA-style head topology. Multiplies compression beyond what MLA alone gives. Argument against: substantial engineering complexity; harder to reproduce and tune. _Likely_ : hybrid sparse-attention + compressed-head-topology becomes the frontier pattern, with DeepSeek V4 as the prototype and others (Llama 5, Qwen 4) following.

**The wildcard: linear attention takes more share**. As Mamba-3, Gated DeltaNet, KDA, Lightning Attention mature (M45 territory), the head-topology question may become less central — if 75-90% of layers are linear attention, the remaining attention layers' KV cache is no longer the dominant memory cost. The Axis C choices matter less when there are fewer attention layers. _Less likely_ than pure attention-architecture refinement, but possible if linear attention closes more quality gaps.

**Concrete forecast** : by Q3-Q4 2027, expect the dominant frontier pattern to be:

  * **50-70% linear attention layers** (M45) — Mamba-3 derivatives or Kimi Delta Attention-style
  * **30-50% softmax attention layers with MLA + sparse pattern** — DeepSeek V4-style hybrid
  * **IHA-style cross-head mixing** within those softmax layers, if the scaling work pans out
  * **TurboQuant successor at 1.5-2 bits** for the remaining KV cache

The general lesson: _"replacement of MLA" is the wrong frame_. Frontier 2027 architectures will compose multiple Axis C ideas (MLA's compression, IHA's mixing, sparse patterns) rather than picking one. The interesting research questions are about which compositions work, not which single mechanism wins.

### What just happened?

  * Axis C of the four-axis taxonomy: head topology — how Q, K, V are structured across heads. The most production-critical axis because it determines KV-cache cost.
  * **MHA** : vanilla; every head has independent Q, K, V projections; full quality but full cache cost. Per-token KV: n_heads × 2 × d_head.
  * **MQA** : all heads share one K, one V. 32× cache compression at 32 heads, but quality drops. Falcon, PaLM, ChatGLM2 era.
  * **GQA** : groups of query heads share K, V. 4-8× cache compression with minimal quality loss. **Open-source default 2024-2026** : Llama 2/3/4, Mistral, Qwen 3.5/3.6, MiniMax M2.5.
  * **MLA (DeepSeek V2-V4)** : compress K, V into low-rank latent c_KV (dim ~512); decompress on demand. Cache c_KV instead of full K, V. ~57× cache compression. Quality matches or exceeds MHA at scale.
  * **Decoupled RoPE** makes MLA work: split each query and key into **position-aware** (RoPE applied, small dimension d_qk_rope ≈ 64) and **position-free** (compressed via latent, large dimension). Cache c_KV + shared k_rope. Concatenate at attention time.
  * **Weight absorption** : at inference, W_uq @ W_uk^T precomputed offline; attention happens at latent dimension instead of full K dimension. ~32× compute reduction, ~57× memory bandwidth reduction. **5.76× generation speedup vs MHA** in DeepSeek V2.
  * **The 100B threshold** : MLA wins at >100B scale; GQA wins below. Sarvam ships 105B with MLA, 30B with GQA — same family, different topologies for different scales.
  * **IHA — Interleaved Head Attention (Duvvuri et al., Feb 2026)** : pseudo-heads enable cross-head mixing. P pseudo-heads per head as learned linear combinations of all H originals. Interleave along sequence; standard attention; combine back. **P² patterns per head with O(H²P) parameter overhead**. FlashAttention compatible. Empirical: +10-20% RULER Multi-Key, +5.8% GSM8K, +2.8% MATH-500.
  * **Slim Attention (Graef & Wasielewski, March 2025)**: when projection matrices are square, V can be reconstructed from K via V = K · W_KV. Cache only K. Lossless 2× compression. 8× for Whisper, 32× for T5-11B (non-square projections). Not applicable to GQA/MQA/MLA.
  * **KV-cache engineering compounds with head topology** : 
    * **TurboQuant (ICLR 2026, merged into vLLM April 15, 2026)** : PolarQuant random rotations + Lloyd-Max quantization. 3-bit keys + 2-bit values. **4× capacity**. Available as `--kv-cache-dtype turboquant_3bit_nc`. Decode penalty: 35-43% throughput reduction.
    * **Expected Attention** : estimate future query distribution, evict low-expected-attention keys. Quality-cost tradeoff.
    * **Quest, H2O, SnapKV** : various heuristic KV eviction schemes.
    * **InfiniGen** : KV cache offloading to CPU with prefetching. Lossless but PCIe-bandwidth-bound.
  * Production stack: GQA + TurboQuant 3-bit (90% of teams) or MLA + DeepGEMM (10%, frontier scale). Both compose with FlashAttention 3/4 + PagedAttention.
  * Production zoo: Llama 4 = GQA; DeepSeek V4 = MLA + CSA+HCA hybrid; GLM-5.1 = MLA-style + DSA; Qwen 3.5/3.6 = GQA in attention layers + GDN in linear layers (M45); Kimi Linear = MLA + KDA hybrid; Sarvam 105B = MLA, Sarvam 30B = GQA; Whisper/T5 = MHA + Slim Attention retrofit.
  * Decision tree: pretraining + >100B → MLA; pretraining + <100B → GQA; reasoning-focused new model → consider IHA (experimental); fine-tuning existing model → KV cache engineering (TurboQuant + eviction); MHA encoder-decoder → Slim Attention.
  * The forecast for Q3-Q4 2027: hybrid stacks dominate. 50-70% linear attention layers (M45), 30-50% softmax with MLA + sparse pattern (DeepSeek V4-style), IHA-style cross-head mixing within softmax layers, TurboQuant successor at 1.5-2 bits. _The interesting question is which compositions work, not which single mechanism wins_.

Module 47 (next, completing the trilogy) tackles **Axes D and E — kernel implementation and compositional architectures** : FlashAttention 1-4 with the FA4 reverse-engineered Blackwell pipeline, PagedAttention's KV cache layout, Ring Attention and Striped Attention for sequence parallelism at >1M context, MoDA's depth attention (March 2026), and the hybrid composition patterns across the production zoo (Qwen3-Next 3:1, Kimi Linear, Ling 2.5, Nemotron 3 Nano, DeepSeek V4 CSA+HCA).
