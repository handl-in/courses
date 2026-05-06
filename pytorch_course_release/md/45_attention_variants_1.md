# Module 45 — Attention Variants I: Roles, Receptive Fields & Sparsity

# _Attention Variants I:_ roles, receptive fields & sparsity

_Attention Trilogy · Module 45 · April 2026 currency_

— first installment of a three-module attention compendium; the four-axis taxonomy, the linear-attention lineage from Mamba to Kimi Delta Attention, and the trainable sparse attention family (NSA, MoBA, DSA, XAttention)

\--- 

M44 took inventory of 2026 LLMs and named "attention variant" as one of the five primitives where they differ. This module starts a three-part deep-dive into that primitive specifically. _The reason it deserves three modules is that "attention" in 2026 means dozens of distinct mechanisms_ — each solving a different problem, each with subtle interactions with the rest of the architecture, each with its own kernel-implementation considerations. Treating them all in a single module would compress them into a list; the list is the wrong abstraction.

This module covers **Axis A (semantic role)** and **Axis B (receptive field)**. M46 covers Axis C (head topology and KV-cache engineering — MQA / GQA / MLA / IHA / Slim Attention). M47 covers Axis D (kernel implementations — FlashAttention 1-4, PagedAttention, Ring Attention) and Axis E (compositional architectures — MoDA, hybrid patterns). The three modules compose: any production attention layer in 2026 makes a choice on each axis simultaneously.

The Axis A material is the foundation: _what is attention, mathematically_ , and what happens when you replace its softmax with kernel-trick linearizations. The Axis B material is the production-relevant frontier: _where each query attends_ , with sliding window, global, and the new-generation trainable sparse attention mechanisms (NSA, MoBA, DSA, XAttention). By the end you'll know why linear attention has had three failed comebacks and what makes the 2025-2026 generation (Mamba-3, Gated DeltaNet, Kimi Delta Attention) genuinely different, and how the trainable sparse attention family closes the long-context gap that older fixed-pattern sparse attention couldn't.

> **★ KEY IDEA**  
>  The four-axis taxonomy of attention. **Axis A — semantic role** : what's attending to what (self, cross, causal) and how the score is normalized (softmax, linear). **Axis B — receptive field** : where each query attends (full, sliding window, global, sparse). **Axis C — head topology** : how Q/K/V are shared across heads (MHA, MQA, GQA, MLA, IHA, Slim — M46). **Axis D — kernel implementation** : how it runs on hardware (FlashAttention 1-4, PagedAttention, Ring — M47). **Axis E — composition** : how attention layers compose with depth, MoE, and other primitives (MoDA, hybrid stacks — M47). Every production attention layer is a point in this 5D space. This module's frontier material. **Linear attention is back** : not as a softmax replacement (the 2020-2022 attempts failed) but as a _complementary mechanism_ in hybrid stacks. **Mamba-3 (ICLR 2026)** : complex-valued state update equivalent to data-dependent RoPE; trapezoidal discretization replacing Euler; MIMO formulation for arithmetic intensity; +1.8 points downstream over Gated DeltaNet at 1.5B; matches Mamba-2 perplexity at half the state size. **Gated DeltaNet (Yang et al.)** : Mamba-2 + delta rule + scalar gate; the linear-attention layer in Qwen3-Next and Qwen3.5. **Kimi Delta Attention (KDA)** : Gated DeltaNet refined with channel-wise gating. **Lightning Attention** : Ling 2.5's linear-attention choice. **Sparse attention is solved at the trainable level** : **NSA (Yuan et al. 2025, ACL 2025 award)** : three parallel paths — compressed coarse + selected fine + sliding window — with learned gates; surpasses full attention on most tasks despite sparsity. **MoBA (Lu et al. 2025)** : Mixture of Block Attention, trainable block routing. **DSA (DeepSeek V3.2 → V4)** : element-wise trainable sparsity with FP4 Lightning Indexer. **XAttention (MIT-Han Lab, ICML 2025)** : plug-and-play antidiagonal scoring, 13.5× speedup, no retraining. **SeerAttention, InfLLM-V2, FSA, DMA** : the broader trainable sparse family. _The 2026 frontier in attention isn't replacing softmax — it's adding sparsity-aware paths or hybridizing with linear-attention layers_. 

## Two new faces — taking attention seriously

R

Receptive Field

"For each query token, I decide which key tokens it can see. Full, local, or sparse — that's me."

In vanilla attention, every query attends to every key — the receptive field is global, the cost is O(N²). I'm the constraint that says _your query at position 500 doesn't need to attend to the token at position 5_ in many cases. I come in three flavors. **Sliding window** : query at position i attends to keys in [i−w, i] only — fixed local. **Global** : every query attends to every key — full O(N²). **Sparse** : query attends to a content-dependent subset, learned during training. The 2025-2026 generation of trainable sparse attention (NSA, MoBA, DSA) is what made sparse attention competitive with full at scale — older fixed-pattern sparse (BigBird, Longformer) couldn't match full attention quality. _I'm the difference between O(N²) and O(N log N) or O(N) with no quality loss_ when designed correctly.

Λ

Linearity

"I replace softmax with a kernel trick. Constant memory, linear compute — the dream that almost works."

Standard attention is `softmax(QK^T)V`; the softmax forces O(N²) computation because it normalizes across all keys per query. _I rewrite this to avoid materializing QK^T at all_ — by replacing softmax with a kernel function φ such that `φ(Q)φ(K)^T V = φ(Q)(φ(K)^T V)`, where the right-hand parenthesization is O(N×d²) instead of O(N²×d). Constant-size state, linear in N. **I died three times before working**. Performer (2020) used random features, lost too much accuracy. RWKV/RetNet (2023) added decay, marginal results. The 2024-2026 generation — **Mamba-2, Gated DeltaNet, Kimi Delta Attention, Lightning Attention** — finally got the recipe right by combining selective decay (data-dependent gates), discretization tricks (trapezoidal in Mamba-3), and complex-valued state updates (Mamba-3's RoPE bridge). _I'm not a softmax replacement; I'm a complement_. Production 2026 stacks use me as 3 of every 4 layers with full attention as the 4th.

## The four-axis taxonomy, drawn

Every attention variant in production is a point in five-dimensional space. This module's two axes:

The five axes of attention design — this module covers A and B Axis A — Semantic role & normalization Self vs cross (what attends to what) · Causal vs bidirectional (mask shape) Softmax (standard, quadratic) vs Linear (kernel-trick, recurrent state) Examples: vanilla self-attention · cross-attention · causal LM · Mamba-3 · Gated DeltaNet · Kimi Delta Attention · Lightning Attention Axis B — Receptive field (where each query attends) Full (every query → every key, O(N²)) · Sliding window (local only) · Global (special tokens) Sparse trainable (content-dependent subset learned during training) Examples: vanilla full · Mistral sliding window · NSA (3-path) · MoBA · DSA · XAttention · SeerAttention · InfLLM-V2 · FSA · DMA Axis C — Head topology & KV-cache engineering · M46 MHA → MQA → GQA → MLA → IHA → Slim Attention · DeepSeek V4's CSA+HCA Axis D — Kernel implementation · M47 FlashAttention 1/2/3/4 (Blackwell) · PagedAttention · Ring · Striped · Star · TokenRing · DeepSpeed Ulysses · USP Axis E — Compositional architecture · M47 Hybrid stacks (Qwen3-Next 3:1, Kimi Linear, Ling 2.5, Nemotron 3 Nano) · MoDA depth attention · KArAt · MTA · Differential Transformer

Reading any 2026 model card is identifying the choices on these five axes simultaneously. This module's job is to make Axis A and Axis B legible.

## The mathematical baseline

Before variants, the canonical thing they all modify. Standard scaled dot-product attention from Vaswani et al. (2017):
    
    
    def vanilla_attention(Q, K, V, mask=None):
        # Q, K, V: [batch, seq, n_heads, d_head]
        # Compute attention scores
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_head)
        # Optional mask (causal, padding, custom)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float("inf"))
        # Softmax normalization (the expensive bit at long N)
        weights = F.softmax(scores, dim=-1)
        # Weighted sum of values
        output = weights @ V
        return output  # [batch, seq, n_heads, d_head]

The key facts that motivate all variants:

  * **Computational cost** : O(N²d) for sequence length N, head dimension d. Quadratic in N is the central problem at long context.
  * **Memory cost** : the attention matrix `scores` is N×N. At N=1M with batch=1 and 64 heads, that's 64TB of FP16 memory. M22's FlashAttention solved this by never materializing the full matrix; M47 covers it.
  * **Why softmax** : makes scores into a probability distribution; ensures gradient flow; produces interpretable attention weights. Replacing it loses these properties.
  * **The mask** : a [seq, seq] tensor that determines what each query can see. Different masks → different attention semantics (causal, bidirectional, local).

## Axis A: Semantic roles

### Self vs cross attention

The earliest distinction. **Self-attention** : Q, K, V all come from the same input sequence — the sequence attends to itself. This is the dominant pattern in modern decoder-only LLMs. **Cross-attention** : Q comes from one sequence (typically the decoder), K and V come from another (typically the encoder).
    
    
    def self_attention(x, W_Q, W_K, W_V):
        Q = x @ W_Q
        K = x @ W_K   # same input
        V = x @ W_V
        return vanilla_attention(Q, K, V)
    
    def cross_attention(x_decoder, x_encoder, W_Q, W_K, W_V):
        Q = x_decoder @ W_Q
        K = x_encoder @ W_K   # different input
        V = x_encoder @ W_V
        return vanilla_attention(Q, K, V)

Where you find each in 2026:

  * **Self-attention** : every transformer LLM uses this in its main blocks. Llama 4, GPT-5.5, Claude Opus 4.7, DeepSeek V4 — all primary attention is self.
  * **Cross-attention** : VLA models (M43) — robot demonstrations cross-attend image tokens with text tokens; multimodal models (M38) where vision and language enter as separate streams; encoder-decoder translation (T5 lineage); diffusion models cross-attend conditioning text into the image-noise stream.

Architectural note: in modern decoder-only multimodal models (Llama 4, Era 3a from M44), vision and text tokens are concatenated into a single sequence before the first attention layer — this turns what would be cross-attention into early-fused self-attention. _Cross-attention as a distinct architectural element is mostly retreating_ ; early fusion subsumes it.

### Causal masking

A constraint, not a separate mechanism. The mask shape:
    
    
    # Causal mask (lower-triangular): query at position i can only attend to keys at j ≤ i
    mask = torch.tril(torch.ones(N, N))
    
    # Bidirectional: every query attends to every key (used in BERT, encoder-only)
    mask = torch.ones(N, N)
    
    # Causal + sliding window (Mistral): can attend to last w tokens only
    mask = torch.tril(torch.ones(N, N)) * (~distance_from_diagonal(N) > w)

Causal attention is a property of how decoders work — autoregressive next-token prediction forbids leaking future information into past predictions. _Every modern decoder-only LLM uses causal attention_ ; non-causal is reserved for encoder-only models (BERT-derivatives) and certain task-specific architectures.

### Softmax vs linear: the central Axis-A choice

The deeper Axis A question: _does the model use softmax normalization or some kernelized alternative?_ The answer determines the asymptotic compute cost and the architectural class of the model.

## Linear attention: the recurring comeback

The kernel-trick reformulation. Start with softmax attention written explicitly:

`output[i] = Σ_j (exp(Q[i] · K[j]) / Z) × V[j]` where `Z = Σ_j exp(Q[i] · K[j])`

This is O(N²) because Z requires summing across all j per i. The kernel-trick rewrite: replace `exp(Q[i] · K[j])` with `φ(Q[i]) · φ(K[j])` where φ is some feature map. Now:

`output[i] = (Σ_j φ(Q[i]) · φ(K[j]) × V[j]) / Z = φ(Q[i]) · (Σ_j φ(K[j]) ⊗ V[j]) / Z`

The crucial move: the inner sum `S = Σ_j φ(K[j]) ⊗ V[j]` is a fixed-size matrix that can be computed once and reused across all queries — or, in autoregressive decoding, updated incrementally per token:
    
    
    def linear_attention_step(q_t, k_t, v_t, S, z):
        # State S has shape [d_feature, d_value] — fixed-size regardless of sequence length
        # z has shape [d_feature] — normalizer state
        phi_q = feature_map(q_t)   # [d_feature]
        phi_k = feature_map(k_t)   # [d_feature]
    
        # Update state (this is the recurrence)
        S = S + torch.outer(phi_k, v_t)   # [d_feature, d_value]
        z = z + phi_k                       # [d_feature]
    
        # Output = read from state
        output = (phi_q @ S) / (phi_q @ z + 1e-6)   # [d_value]
        return output, S, z

The properties:

  * **Constant memory** : S is [d_feature × d_value], doesn't grow with N.
  * **Linear compute** : each step is O(d²); total is O(Nd²). At long N this dominates over O(N²d).
  * **Recurrent during decoding** : like an RNN — state updates token by token. M33's territory.
  * **Parallel during training** : with associative scan tricks, you can recompute the prefix in chunks (chunkwise recurrent training).

_This sounds like a free win_. The catch: **fixed-size state means fixed-capacity memory**. Distant tokens get blurred into the state; long-range exact recall degrades. The 2020-2022 attempts (Performer, Linformer, RetNet) demonstrated this empirically — performance lagged softmax attention by enough that nobody adopted them at scale.

### The 2024-2026 linear attention renaissance

Three innovations changed the picture:

#### Innovation 1: Selective decay (data-dependent gates)

The fundamental fix to "state gets stale": **decay old state with a learned, data-dependent gate** :
    
    
    # Gated linear attention update:
    S = γ_t * S + torch.outer(phi_k, v_t)
    # γ_t is data-dependent — computed from current input
    γ_t = sigmoid(W_gate @ x_t)

Now the state has a forgetting mechanism. Old information decays unless reinforced. This breaks pure associativity (γ_t depends on past), but you can recover parallel training via **chunkwise recurrent** algorithms (process chunks in parallel, recurrence between chunks). The same pattern as Mamba's selective scan.

#### Innovation 2: Delta rule + state correction

DeltaNet (Schlag et al., 2021; revived by Yang et al., 2025) adds a correction step. Instead of just additive update, it computes an _error_ between the new key-value association and what's currently stored, then corrects:
    
    
    # Delta rule: compute current value for k_t given current state, correct it
    v_predicted = (phi_k @ S) / (phi_k @ z)
    delta = β * (v_t - v_predicted)
    S = S + torch.outer(phi_k, delta)
    # β is a learnable rate

This is the "delta" — it learns to fix wrong associations rather than blindly accumulate. **Gated DeltaNet** (Yang et al., 2025): combines Mamba-2's gating with the delta rule. Used as the linear-attention layer in Qwen3-Next and Qwen3.5.

#### Innovation 3: Complex-valued state (Mamba-3)

The April 2026 frontier. **Mamba-3 (ICLR 2026, Lahoti et al.)** introduced three improvements over Mamba-2:

  * **Trapezoidal discretization** : replaces Mamba-2's Euler discretization with the trapezoidal rule — more accurate when discretizing the continuous-time state-space equations into discrete updates. The math: instead of `x_{t+1} = x_t + dt · f(x_t)` (Euler), use `x_{t+1} = x_t + dt/2 · (f(x_t) + f(x_{t+1}))` (trapezoidal). Higher-order accuracy.
  * **Complex-valued state update** : equivalent to data-dependent rotary embeddings. The complex multiplication `S_t = γ_t · S_{t-1}` with γ as a complex number with magnitude in [0,1] and phase that rotates with input naturally encodes _both_ decay AND positional rotation. The connection: Mamba-3's complex SSM is mathematically equivalent to RoPE on the state. _This solves the state-tracking weakness of prior linear models_ — synthetic tasks like parity that Mamba-2 fails, Mamba-3 solves.
  * **MIMO formulation** : Multi-Input Multi-Output state updates. Instead of one input → one output per channel, allows multiple inputs and outputs per state, increasing arithmetic intensity (the FLOP/byte ratio). This makes the kernel actually fast on modern GPUs — older Mamba was theoretically linear but memory-bound; Mamba-3 fixes this.

Mamba-3 results: at 1.5B scale, +0.6 points downstream over Gated DeltaNet; MIMO variant adds another +1.2; matches Mamba-2 perplexity at half the state size. _The first time a linear-attention architecture has clearly advanced the Pareto frontier without trading off too many quality points_.

#### Production linear-attention layers in April 2026

Linear-attention variants in production / leading research, April 2026 Variant| Origin| Mechanism| Used by  
---|---|---|---  
**Mamba**|  Gu & Dao 2023| Selective SSM with input-dependent A, B, C| Original SSM-based model line  
**Mamba-2**|  Dao & Gu 2024| Structured State Space Duality (SSD); transformer-equivalent perspective| Nemotron-H, Bamba, hybrid stacks  
**Mamba-3**|  Lahoti et al. ICLR 2026| Trapezoidal disc. + complex SSM (data-dep. RoPE) + MIMO| Latest research; pre-production  
**RetNet**|  Sun et al. 2023| Retention with exponential decay; parallel/recurrent dual form| Microsoft research lineage  
**GLA — Gated Linear Attention**|  Yang et al. 2024| Linear attention + scalar gating per head| Several open-weight architectures  
**DeltaNet**|  Schlag et al. 2021 / 2025| Delta rule for state correction| Foundational; revived recently  
**Gated DeltaNet (GDN)**|  Yang et al. 2025| DeltaNet + Mamba-2 gating; chunkwise recurrent training| Qwen3-Next, Qwen3.5 (3:1 ratio with attention)  
**Kimi Delta Attention (KDA)**|  Moonshot 2026| GDN refined: channel-wise gating instead of scalar| Kimi Linear architecture  
**Lightning Attention**|  Various 2024-2025| Simpler linear-attention variant (decay + projection)| Ling 2.5 (1T MoE)  
  
The 2026 production reality: **none of these models use linear attention exclusively**. They use it as 3 of every 4 layers (Qwen3-Next, Qwen3.5) or in alternation with full attention layers (Ling 2.5, Nemotron 3 Nano). The full-attention layer provides exact recall when needed; the linear layers provide cheap context aggregation. _The lesson: linear attention is a complement, not a replacement_.

## Axis B: Receptive field

From M44's M22 reference, the canonical attention computation is global: every query attends to every key in the sequence (within the causal mask for decoders). This is the O(N²) cost. The receptive-field axis is about _what subset of keys each query attends to_.

### Sliding window (local attention)

The simplest restriction: each query attends to a fixed-size local window. Mistral 7B's signature design choice; older going back to Longformer (2020).
    
    
    def sliding_window_mask(N, w):
        """Causal sliding window: query i attends to keys [max(0, i-w+1), i]"""
        mask = torch.zeros(N, N)
        for i in range(N):
            mask[i, max(0, i - w + 1):i + 1] = 1
        return mask

Cost: O(N×w) instead of O(N²) — when w is fixed and N is large, this is asymptotically much cheaper. For w=4096 and N=1M, that's a 244× reduction in attention cost.

The downside: each query has access only to local context. Information from far back in the sequence has to _propagate_ through layer-by-layer aggregation — a 32-layer model with w=4096 has theoretical receptive field ~131K tokens, but exponential decay in effective dependency strength. Long-range exact recall is poor.

**Production use** : Mistral lineage. Mostly displaced in 2026 frontier models because long-range recall matters. Often used as _part of a hybrid_ — sliding-window local layers alternating with full-attention global layers.

### Global attention (special tokens)

Older idea, mostly historical now. A small number of "global" tokens (often 1-32) attend to the entire sequence; everyone else uses sliding window. Used in Longformer (2020) and BigBird (2020). Production frontier doesn't use this pattern in 2026 — sparse trainable attention does the job better.

### Sparse attention: the 2025-2026 trainable revolution

The recent class of methods that made sparse attention competitive with full attention. Older sparse attention used _fixed patterns_ (BigBird's combination of local + random + global) — these worked in narrow regimes but couldn't match full attention quality at frontier scale because the sparsity pattern was hand-designed and didn't adapt to content.

The 2025-2026 generation: **trainable sparse attention**. Sparsity patterns are learned during training based on content. Models pretrained with sparse attention from the start match or exceed full-attention models on benchmarks while being asymptotically cheaper.

#### NSA — Native Sparse Attention (Yuan et al., DeepSeek-AI 2025)

Won ACL 2025 award. The reference design for trainable sparse attention.

NSA's three-path hierarchical structure:

NSA — three parallel attention paths combined via learned gates Query Q_t at position t ① Compressed (coarse) Block KVs into chunks of 32, stride 16. Mean+max pool to one vector per block. Attend to all blocks. ② Selected (fine) Reuse compression scores to rank blocks; keep top-16 blocks of size 64 in original (uncompressed) form. ③ Sliding window Most recent 512 tokens; standard local attention. Captures immediate context patterns. O_cmp global summary O_slc precise blocks O_win local context Learned gates g_cmp(q), g_slc(q), g_win(q) from small MLP over query Output = g_cmp · O_cmp + g_slc · O_slc \+ g_win · O_win Why NSA works at training and inference Native trainability: hierarchical structure has differentiable gates, end-to-end gradients flow through all three paths Hardware alignment: queries grouped by GQA, sparse blocks accessed contiguously, balanced arithmetic intensity No extra forward pass for selection: reuses compression attention scores to rank blocks (the key efficiency trick) Empirical result: surpasses full attention on most tasks at 27B/260B-tokens; perfect 64K NIAH retrieval; substantial speedups at 64K decode/forward/backward Hyperparameters from paper: window w=512, compression block l=32, stride d=16, selection block l'=64, top-n=16

NSA's algorithm walked through:
    
    
    def nsa_attention(Q, K, V, query_idx):
        # PATH 1: Compressed attention
        # Block K, V into chunks of 32 tokens, stride 16
        K_blocks = block_pool(K, block_size=32, stride=16, op="mean+max")
        V_blocks = block_pool(V, block_size=32, stride=16, op="mean+max")
        cmp_scores = Q[query_idx] @ K_blocks.T   # [num_blocks]
        O_cmp = F.softmax(cmp_scores) @ V_blocks
        # Crucial: cmp_scores will be reused below — no extra forward pass
    
        # PATH 2: Selected attention (uses scores from PATH 1)
        top_block_idxs = cmp_scores.topk(k=16).indices   # top-16 blocks
        K_selected = gather_blocks_uncompressed(K, top_block_idxs, block_size=64)
        V_selected = gather_blocks_uncompressed(V, top_block_idxs, block_size=64)
        slc_scores = Q[query_idx] @ K_selected.T   # [16 * 64]
        O_slc = F.softmax(slc_scores) @ V_selected
    
        # PATH 3: Sliding window (last 512 tokens)
        K_window = K[query_idx - 512:query_idx + 1]
        V_window = V[query_idx - 512:query_idx + 1]
        win_scores = Q[query_idx] @ K_window.T
        O_win = F.softmax(win_scores) @ V_window
    
        # GATING — learned weights from query
        gates = F.softmax(gate_mlp(Q[query_idx]))   # [3]
        output = gates[0] * O_cmp + gates[1] * O_slc + gates[2] * O_win
        return output

The empirical claim that justifies the complexity: **NSA pretrained on 27B parameters, 260B tokens — surpasses full attention on most benchmarks despite using sparse computation**. This includes reasoning-heavy tasks like DROP and GSM8K. The paper's interpretation: forced sparsity acts as regularization, focusing the model on important information and filtering noise.

#### MoBA — Mixture of Block Attention (Lu et al., Moonshot 2025)

Same family as NSA but with different routing. MoBA divides the sequence into blocks; for each query, a router selects which blocks to attend to, similar to MoE expert routing but at the attention level. The selection is data-dependent and trainable.

Production use: Kimi family models. Compared to NSA, MoBA is conceptually simpler (one path with routing) but doesn't have the explicit hierarchical compression-then-selection structure.

#### DSA — DeepSeek Sparse Attention (V3.2 → V4 foundation)

The trainable sparse attention used in DeepSeek V3.2, evolved into V4's CSA+HCA Hybrid Attention (covered in M44). DSA introduces **element-wise trainable sparsity** suitable for ultra-large LLMs — instead of block-level selection, decisions are per-element.

The mechanism: for each query, a learned indexer (in V4, an FP4-based "Lightning Indexer") scores all keys and selects the top-k. Unlike NSA's compression step, DSA computes selection directly. The Lightning Indexer being in FP4 (M41 territory) is the cost-saving move — selection doesn't need full precision.

Production use: DeepSeek V3.2 directly; V4 extends it with the CSA+HCA hybrid; **GLM-5/5.1 also uses DSA**.

#### XAttention — block-sparse with antidiagonal scoring (MIT-Han Lab, ICML 2025)

The plug-and-play option. Unlike NSA/MoBA/DSA which require pretraining with the sparse mechanism in place, XAttention is a _post-training_ sparsification scheme.

The key insight: the sum of antidiagonal values in the attention matrix (lower-left to upper-right) is a powerful proxy for block importance. Compute these sums cheaply; rank blocks by importance; prune below a threshold.
    
    
    def xattention_block_score(Q_block, K_block):
        """Antidiagonal score for block importance.
        Insight: antidiagonal sum captures block's diagonal energy concentration"""
        scores = Q_block @ K_block.T   # [block_size, block_size]
        # Sum along the antidiagonal (lower-left to upper-right)
        antidiag_sum = torch.sum(torch.flip(scores, dims=[-1]).diagonal())
        return antidiag_sum
    
    # Use antidiagonal scores to keep only the most important blocks per query
    # Up to 13.5× speedup vs full attention with comparable accuracy

Production properties: drop-in for existing models, no retraining. Validated on RULER, LongBench, VideoMME, VBench. Up to 13.5× attention speedup with accuracy comparable to full attention. _The right choice when you can't retrain_ — applies to existing closed-weight models served via inference frameworks.

#### Other 2025-2026 trainable sparse variants

Trainable / training-aware sparse attention family, 2025-2026 Variant| Origin| Mechanism| Notes  
---|---|---|---  
**NSA**|  Yuan et al. 2025 (DeepSeek-AI)| Three-path: compressed + selected + sliding| ACL 2025 award; reference design  
**MoBA**|  Lu et al. 2025 (Moonshot)| Block-level routing similar to MoE| Used in Kimi family  
**SeerAttention**|  Gao et al. 2024| Predicts attention sparsity from context| Self-distillation training pipeline  
**InfLLM-V2**|  Zhao et al. 2025| Unified short and long context sparsity| Single kernel handles both regimes  
**FSA — Flash Sparse Attention**|  Yan et al. 2025| Improves kernel efficiency for small query-head counts| Hardware-aware optimization  
**DSA**|  DeepSeek-AI 2025-2026| Element-wise trainable sparsity with FP4 indexer| DeepSeek V3.2, V4 foundation; GLM-5  
**DMA**|  Shi et al. 2025| Eviction-based sparse selection| Like KV cache eviction made trainable  
**XAttention**|  Xu et al. ICML 2025| Antidiagonal scoring for block importance| Plug-and-play; no retraining required  
  
The trajectory: **fixed-pattern sparse (BigBird, Longformer 2020) → trainable sparse with hierarchies (NSA 2025) → element-wise trainable (DSA 2026) → hybrid attention (DeepSeek V4 CSA+HCA 2026)**. Each generation closes the quality gap with full attention while extending context-length efficiency.

## Production zoo: Axis A and B in April 2026 frontier models

Cross-referencing the variants discussed above with the M44 model zoo:

Axis A and B choices across April 2026 frontier models Model| Self/Cross| Causal| Softmax/Linear| Receptive field  
---|---|---|---|---  
**Llama 4 Scout**|  Self (early-fused MM)| Causal| Softmax| Full + iRoPE chunked attention in RoPE layers  
**Llama 4 Maverick**|  Self (early-fused MM)| Causal| Softmax| Full + iRoPE  
**DeepSeek V4-Pro**|  Self| Causal| Softmax| **CSA + HCA hybrid sparse** (4× compress + 128× compress, FP4 indexer)  
**GLM-5.1**|  Self| Causal| Softmax| **DSA** (DeepSeek Sparse Attention)  
**Qwen3-Next / Qwen3.5**|  Self| Causal| **3:1 hybrid** — Linear (Gated DeltaNet) : Softmax| Full softmax in attention layers  
**Kimi Linear / K2.6**|  Self| Causal| **Hybrid** — Linear (KDA) : Softmax (MLA)| Hybrid pattern  
**Ling 2.5**|  Self| Causal| **Hybrid** — Linear (Lightning Attention) : Softmax (MLA)| Hybrid pattern  
**Nemotron 3 Nano**|  Self| Causal| **Hybrid** — Mamba-2 SSM blocks + sparse self-attention| Mamba-Transformer hybrid  
**Mistral Large**|  Self| Causal| Softmax| **Sliding window** (Mistral signature)  
**MiniMax M2.5**|  Self| Causal| Softmax| Full (deliberately classic)  
**Closed (GPT-5.5, Claude 4.7, Gemini 3.1)**|  Self (multimodal)| Causal| Softmax (presumed)| Undisclosed  
  
Three observations:

  * **Hybrid is the new mainstream for non-Llama open-weight models**. Qwen3-Next, Kimi Linear, Ling 2.5, Nemotron 3 Nano all combine linear-attention layers with softmax-attention layers. The specific linear variant differs (GDN, KDA, Lightning Attention, Mamba-2) but the hybrid pattern is consistent.
  * **DeepSeek lineage drives sparse attention adoption**. DSA → V4's CSA+HCA → GLM-5.1 picking up DSA. The Chinese open-weight models are converging on sparse attention for efficiency.
  * **Llama 4 deliberately stays softmax + full attention**. The bet: combine standard attention with iRoPE position encoding for length generalization, rather than introducing sparsity. Maverick at 1M context and Scout at 10M validate this approach.

#### Q&A; — About Axis A and B variants **Q:** Why did linear attention fail in 2020-2022 but succeed in 2024-2026? **A:** Three compounding reasons. (1) **State decay was wrong**. Early linear attention (Performer 2020) used random feature maps without state decay; old information accumulated indefinitely and corrupted the state. The 2024-2026 generation (GDN, KDA, Mamba) all use data-dependent decay gates that selectively forget. (2) **Pure linear was the wrong target**. Early work tried to fully replace softmax attention. The empirical result: linear models lagged transformers by enough to be unviable. The 2024-2026 reframing: _linear attention as 75% of layers, softmax as 25%_. The full-attention layer provides exact recall when needed; the linear layers provide cheap context aggregation. _Hybrid stacks dominate; pure linear stacks remain niche_. (3) **Hardware utilization was bad**. Earlier linear-attention kernels were memory-bound, running far below theoretical peak. Mamba-2 (SSD) and FlashAttention-style techniques fixed this; Mamba-3's MIMO formulation pushed arithmetic intensity higher still. _The renaissance is partly algorithmic, partly hardware-engineering_. **Q:** What's the actual difference between NSA's three paths and a fixed BigBird-style sparse pattern? **A:** Three differences that compound. (1) **Trainable, not fixed**. NSA's gates and selection are learned; BigBird's local + random + global pattern is hand-designed. The model can adapt its sparsity to content, BigBird can't. (2) **Hierarchical compression**. NSA's first path compresses blocks of 32 tokens into single representations — this gives a cheap global view that BigBird's "random tokens" approximation can't match. The compression captures actual content (mean+max pooling), not just samples. (3) **Selection reuses compression**. NSA's selection path reranks blocks based on compression attention scores — no separate computation needed. BigBird had no equivalent; selection in BigBird is fixed at training time. _Empirically, NSA matches or exceeds full attention; BigBird-style sparse always lagged_. The fixed-pattern era of sparse attention is over for frontier models. **Q:** When should I use sliding window attention specifically? **A:** Three cases where it's still the right answer. (1) **Truly local-dependency tasks**. Some tasks (POS tagging, syntax parsing, simple summarization of bounded paragraphs) genuinely don't need long-range dependencies. Sliding window is sufficient and cheaper than even sparse attention. (2) **Hardware-constrained inference**. On edge devices or older hardware where you can't afford the kernel complexity of NSA/DSA, sliding window is well-supported and fast. Mistral 7B targets this. (3) **Hybrid layer in larger architectures**. Several 2026 architectures alternate sliding-window layers with full-attention layers (similar to Qwen3-Next's GDN/Attn alternation but with sliding instead of GDN). For most other use cases — particularly if long-context retrieval matters — sparse trainable attention dominates. _Sliding window is the cheap, reliable, mostly-displaced incumbent_. **Q:** Why does Mamba-3's complex-valued state matter, mathematically? **A:** Three intertwined reasons. (1) **Complex multiplication encodes both magnitude and rotation**. A real-valued decay γ ∈ [0,1] just shrinks the state. A complex-valued γ = r·e^(iθ) with r ∈ [0,1] shrinks AND rotates. The rotation is what enables state-tracking — synthetic tasks like parity require the model to remember the parity of how many 1s have been seen, which is a binary state that needs rotation (specifically, multiplication by e^(iπ) = -1) to track. Real-valued decay can't represent this; complex can. (2) **Equivalent to data-dependent RoPE**. The complex update `S_t = γ_t · S_{t-1}` with input-dependent γ_t can be reformulated as RoPE on the state, where the rotation angle is determined by the input. M30's RoPE on positions; Mamba-3's RoPE on state. The mathematical bridge unifies SSMs and rotary embeddings. (3) **Computationally efficient**. Complex multiplication is just two real multiplications and an addition. The hardware cost is small; the expressivity gain (state-tracking) is substantial. _Mamba-3 finally solved the state-tracking weakness that plagued every prior linear-attention architecture_. **Q:** Should I expect linear attention to displace softmax attention in the next 2-3 years? **A:** No, but it will become a substantial fraction of layers in production stacks. Three reasons. (1) **Pure linear models still lose on certain tasks**. Information extraction from semi-structured documents, exact retrieval from long contexts — softmax attention remains better at these. The hybrid pattern (3:1 ratio of linear to softmax) reflects this. (2) **Softmax has decades of engineering**. FlashAttention 1-4, PagedAttention, the entire serving infrastructure (M47) is built around softmax. Linear attention's kernels are catching up but lag in production maturity. (3) **The pattern that actually wins is hybrid**. Production 2026 (Qwen3.5, Kimi Linear, Ling 2.5, Nemotron 3 Nano) uses linear AS PART OF a stack with softmax. Pure linear architectures (Mamba-only) work but rarely win at frontier scale. _Forecast: by 2027-2028, expect 50-75% of layers in frontier models to be linear-attention variants, with the remaining softmax layers providing exact-recall capabilities. Pure softmax stacks (current Llama 4) become a minority pattern_. **Q:** How do trainable sparse attention and linear attention compare as long-context strategies? **A:** Different tradeoffs, different regimes. **Trainable sparse attention (NSA, DSA, MoBA)** : maintains softmax semantics; trained from scratch with sparsity; matches full attention quality with linear-or-better cost. Best for models where exact retrieval matters and you can pretrain with the mechanism. **Linear attention (GDN, KDA, Mamba-3)** : replaces softmax with kernel trick; constant memory regardless of N; slight quality cost vs full attention but with major efficiency wins. Best as a layer type within hybrid stacks where 75% of layers can be cheap and 25% provide exact recall. **The trend** : most 2026 frontier models use one or the other, sometimes both — DeepSeek V4 uses CSA+HCA (sparse softmax); Qwen3.5 uses Gated DeltaNet (linear) + softmax in 3:1 ratio. _The decision factor is whether you're willing to pretrain from scratch with the new mechanism (linear or sparse) or need to retrofit existing models (XAttention is the only post-training-only option here)_. 

## Code Magnets: implement the NSA forward pass

You're writing the forward pass for NSA's three-path attention. Three magnets are wrong choices.

Arrange the magnets to compute the NSA output for a single query.

def nsa_forward(Q, K, V, query_idx, gate_mlp): K_blocks = block_pool(K, block_size=32, stride=16, op="mean+max") V_blocks = block_pool(V, block_size=32, stride=16, op="mean+max") K_blocks = K[::32] # naive subsample cmp_scores = Q[query_idx] @ K_blocks.T O_cmp = F.softmax(cmp_scores, dim=-1) @ V_blocks top_block_idxs = cmp_scores.topk(k=16).indices K_selected = gather_blocks_uncompressed(K, top_block_idxs, block_size=64) V_selected = gather_blocks_uncompressed(V, top_block_idxs, block_size=64) slc_scores = Q[query_idx] @ K_selected.T slc_scores = K_selected @ Q[query_idx] # no scaling O_slc = F.softmax(slc_scores, dim=-1) @ V_selected K_window = K[max(0, query_idx - 511):query_idx + 1] V_window = V[max(0, query_idx - 511):query_idx + 1] win_scores = Q[query_idx] @ K_window.T O_win = F.softmax(win_scores, dim=-1) @ V_window gates = F.softmax(gate_mlp(Q[query_idx]), dim=-1) # 3-way gates = gate_mlp(Q[query_idx]) # raw — sum could be anything output = gates[0] * O_cmp + gates[1] * O_slc + gates[2] * O_win return output

show solution
    
    
    def nsa_forward(Q, K, V, query_idx, gate_mlp):
        K_blocks = block_pool(K, block_size=32, stride=16, op="mean+max")
        V_blocks = block_pool(V, block_size=32, stride=16, op="mean+max")
        cmp_scores = Q[query_idx] @ K_blocks.T
        O_cmp = F.softmax(cmp_scores, dim=-1) @ V_blocks
        top_block_idxs = cmp_scores.topk(k=16).indices
        K_selected = gather_blocks_uncompressed(K, top_block_idxs, block_size=64)
        V_selected = gather_blocks_uncompressed(V, top_block_idxs, block_size=64)
        slc_scores = Q[query_idx] @ K_selected.T
        O_slc = F.softmax(slc_scores, dim=-1) @ V_selected
        K_window = K[max(0, query_idx - 511):query_idx + 1]
        V_window = V[max(0, query_idx - 511):query_idx + 1]
        win_scores = Q[query_idx] @ K_window.T
        O_win = F.softmax(win_scores, dim=-1) @ V_window
        gates = F.softmax(gate_mlp(Q[query_idx]), dim=-1)
        output = gates[0] * O_cmp + gates[1] * O_slc + gates[2] * O_win
        return output

The traps:

  * `K_blocks = K[::32] # naive subsample`: replaces NSA's mean+max pooling with simple stride-32 subsampling. The compression path needs each block to _summarize_ its 32-token contents — mean+max pooling captures the average and the dominant features. Naive subsampling just keeps every 32nd token; it discards 31/32 of the information per block. The compression path then has no meaningful global view; it's just a downsampled view. Quality collapses. The pooling is essential to NSA's design.
  * `slc_scores = K_selected @ Q[query_idx] # no scaling`: missing the canonical attention pattern. The query-key dot product should be `Q @ K.T` (query first, then transpose key). The wrong order produces a different shape and different semantics — it's computing how much each _key_ attends to the query, not the other way around. Most production attention also includes scaling by sqrt(d_head) inside the dot product, but the bigger error here is the operand order. The result has the wrong shape for the subsequent softmax and weighted-sum.
  * `gates = gate_mlp(Q[query_idx]) # raw`: forgets the softmax over gate logits. The three gates need to be a probability distribution (g_cmp + g_slc + g_win = 1) so the combined output is a proper convex combination. Raw MLP outputs can be anything — negative, very large, sum to anything. Without softmax, the output magnitude varies arbitrarily, and the relative weighting of paths can flip during training in unintended ways. Standard practice for routed-gate mechanisms (NSA, MoE routing in M28) is to softmax over gate logits.

The pattern: **compress with pooling → use compression scores to select top blocks → uncompressed attention on selected blocks → sliding window for local context → combine via softmax-normalized gates**. Each step has a specific role; the most common implementation bugs are skipping the actual pooling (Magnet 3), reversing operand order in attention (Magnet 10), and forgetting to normalize gates (Magnet 17).

## Who does what?

Match each variant to its real role.

Concept

Real role

Self-attention

A. Q, K, V all from same input sequence; the dominant pattern in modern decoder-only LLMs.

Linear attention (kernel trick)

B. Replaces softmax with φ(Q)φ(K)^T V to get O(N) compute and constant memory; needs decay gates to work.

Mamba-3

C. Trapezoidal discretization + complex-valued state (data-dep. RoPE) + MIMO; ICLR 2026; +1.8 over GDN at 1.5B.

Gated DeltaNet

D. Mamba-2 gating + delta rule for state correction; the linear-attention layer in Qwen3-Next/Qwen3.5.

Sliding window attention

E. Each query attends to last w tokens only; Mistral signature; cheap but no long-range exact recall.

NSA (Native Sparse Attention)

F. Three paths — compressed coarse + selected fine + sliding window — combined via learned gates; ACL 2025.

XAttention

G. Plug-and-play sparse: antidiagonal scoring identifies important blocks; 13.5× speedup, no retraining.

show solution

**Self-attention** → A  
**Linear attention** → B  
**Mamba-3** → C  
**Gated DeltaNet** → D  
**Sliding window** → E  
**NSA** → F  
**XAttention** → G 

The mental shortcut: _self attends to itself, linear kernel-tricks, Mamba-3 goes complex, GDN gates the delta rule, sliding window stays local, NSA splits three ways, XAttention scores antidiagonals_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team is training a 7B model with 256K context target. They're choosing between (a) full attention with FlashAttention-3, (b) Qwen3-Next-style hybrid with Gated DeltaNet, (c) NSA-style trainable sparse, (d) XAttention applied at inference time. Walk through the decision.

show answer

The decision factors:

(1) **Full attention with FA3** at 256K: cost is O(N²) for attention, which becomes ~2/3 of forward pass time at this context length. Manageable for 7B at 256K but becomes painful approaching 1M. Quality is the gold standard. _Right answer if the team prioritizes quality and can afford the compute_.

(2) **Qwen3-Next-style hybrid (3:1 GDN to attention)** : 75% of layers are linear-attention (constant memory, linear compute), 25% are full attention. At 256K, the attention layers still dominate cost but you've cut total attention work by 4×. Quality is slightly below pure full attention on retrieval tasks but matches on most others. _Right answer if cost matters and the model will be deployed at long context regularly_.

(3) **NSA trainable sparse** : requires training the model from scratch with the NSA mechanism. Compute saving is substantial (NSA paper reports speedups across decoding/forward/backward). Quality matches or exceeds full attention. _Right answer if the team can train from scratch and wants the best efficiency/quality tradeoff_. The catch: more engineering complexity (three paths, learned gates, hardware-aligned kernel).

(4) **XAttention applied at inference** : only viable if you've already trained with full attention (or are using a pretrained model). At inference, prune attention blocks via antidiagonal scoring; up to 13.5× speedup with comparable accuracy. _Right answer if the team is fine-tuning an existing pretrained model and can't retrain from scratch_. Also right for serving optimization on already-deployed models.

**The recommendation** for a typical 2026 team training 7B-256K from scratch: option (b) hybrid with Gated DeltaNet. Reasons: NSA is more complex to implement; XAttention is post-training-only; full attention is more expensive than necessary at long context. The hybrid pattern is well-validated (Qwen3-Next, Qwen3.5) and tooling is mature. Plan: 75% GDN layers + 25% full softmax attention with FA3, training from scratch.

The general lesson: _the right answer depends on training-from-scratch ability and whether you can afford the engineering to implement complex mechanisms like NSA_. For most teams without DeepSeek-AI-scale resources, the hybrid Qwen3-Next pattern is the practical sweet spot.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Walk through why NSA's compression path uses mean+max pooling rather than just mean, and why it uses 32-token blocks rather than 16 or 64.

show answer

(1) **Mean+max pooling vs mean only** :

Mean pooling captures the average token in a block — useful for understanding the general "topic" of a block. Max pooling captures the most extreme features — useful for understanding the dominant signals (the most semantically loaded tokens). Combining both gives a richer block summary.

Why it matters for NSA specifically: the compression path's output is used _twice_ — once for its own attention output (O_cmp), and once for selecting top-k blocks for the selection path. Better block representations → better selection of which blocks contain the actually-relevant information. If compression were just mean pooling, blocks with rare-but-important tokens (named entities, key facts) would be averaged out; max pooling preserves these. The combination empirically outperforms mean alone.

(2) **Block size of 32, not 16 or 64** :

The tradeoff: smaller blocks → more granular but more compressed-block tokens (more compute in the compression path). Larger blocks → fewer compressed-block tokens but each one summarizes too much (less information per representation).

NSA's choice of 32 (with stride 16, so blocks overlap) is empirically validated:

  * **32 tokens ≈ one sentence's worth of content** for typical English. This matches a meaningful semantic unit — short enough that the block represents a coherent thought, long enough that you don't have too many blocks to attend to.
  * **Stride 16 (50% overlap)** : the half-overlap gives smoother coverage — token boundaries don't perfectly align with sentence boundaries, so overlapping blocks ensure no critical information falls between blocks.
  * **Computational sweet spot** : at 64K context, block size 32 with stride 16 gives ~4K compressed tokens. This is small enough that attending over them is cheap (4K² is tractable) but large enough to preserve global structure.

The general lesson: **NSA's hyperparameters are not arbitrary**. They reflect specific tradeoffs between granularity and compression cost, validated empirically. Different tasks might benefit from different choices — long-form documents might use larger blocks (64-128); code might use smaller (16-32) since code structure is denser.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** Why does Mamba-3's complex-valued state update connect to RoPE (M30), and what does this tell us about the relationship between linear attention and positional encoding?

show answer

The mathematical bridge:

(1) **RoPE (M30) is rotation in 2D feature pairs**. Take two adjacent feature dimensions; treat them as the real and imaginary parts of a complex number. RoPE rotates this complex number by a position-dependent angle θ(pos). The rotation preserves magnitude but encodes position via phase.

(2) **Mamba-3's complex state update is rotation in 2D state pairs**. The state S_t = γ_t × S_{t-1} where γ_t is a complex number with magnitude in [0,1] and phase determined by input. Rewriting in real/imaginary pairs: this is a rotation by angle phase(γ_t) combined with a magnitude decay |γ_t|. The rotation is data-dependent (depends on the current input), unlike RoPE which is position-dependent.

(3) **The bridge**. RoPE: position-dependent rotation of features for self-attention. Mamba-3: input-dependent rotation of state for SSM updates. Both are rotations in complex 2D feature spaces — the difference is what the rotation depends on (position vs input), and what it acts on (features per token vs cumulative state).

(4) **Implication** : Mamba-3 is mathematically connecting state-space models to rotary embeddings, which suggests deeper structural similarity between transformers and SSMs than was previously understood. The Mamba-3 paper makes this explicit. _Both architectures benefit from rotation-as-encoding; both can be viewed through the same complex-multiplication lens_.

(5) **Why complex matters for state-tracking** : synthetic tasks like parity require the model to maintain a state that flips when seeing a 1 (multiplication by e^(iπ) = -1). Real-valued decay can only shrink the state; it can't flip sign. Complex multiplication can. _Mamba-3 solves the state-tracking weakness by giving its state the algebraic structure (rotation in C) needed to represent these tasks_.

(6) **Broader lesson** : **positional information and state information are dual perspectives on the same problem**. Transformers handle position via RoPE on per-token features. SSMs handle "where am I in the sequence" via cumulative state. Mamba-3 unifies them — the SSM's state update IS a position-dependent transformation, made explicit via complex multiplication. This is conceptual progress, not just engineering.

The general implication: _future architectures will likely continue to merge these views_. Hybrid stacks (Qwen3.5 with GDN + softmax) are the engineering manifestation; Mamba-3's unification is the theoretical manifestation.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Sketch the case for and against pure linear-attention models (Mamba-3-only, no softmax) replacing transformers in production by 2028.

show answer

**Case for replacement** :

(1) **Inference economics** : pure linear models have constant memory and linear compute regardless of context length. At 1M-token contexts, they're vastly cheaper than even the best sparse attention. As context windows continue growing (10M+ in Llama 4 Scout, more expected), the economics tilt further toward linear.

(2) **Mamba-3 closed major gaps** : state-tracking (via complex updates), arithmetic intensity (via MIMO), retrieval (via state-size scaling). The previously-fatal weaknesses of pure linear-attention architectures are addressed.

(3) **Architectural simplicity** : a pure Mamba-3 model has uniform layers, simpler than hybrid stacks that need to alternate two layer types. Less engineering complexity in serving.

(4) **Hardware trends favor linear** : Blackwell and beyond are bandwidth-bound for attention; linear attention is compute-bound, which is the side hardware keeps improving.

**Case against (replacement unlikely by 2028)** :

(1) **Information extraction still favors softmax** : tasks requiring exact retrieval from semi-structured or unstructured documents — table parsing, named-entity extraction from forms, citation matching — softmax attention's content-based addressing remains better. Mamba-3 closed gaps but didn't fully close this one.

(2) **Hybrid stacks already capture most benefits** : 75% linear + 25% softmax (Qwen3-Next/3.5) gets you most of the efficiency gain while keeping exact-recall capability when needed. The marginal benefit of going pure linear isn't large; the engineering cost of mishandling extraction tasks is real.

(3) **Production momentum favors hybrid** : every leading 2026 hybrid model (Qwen3.5, Kimi Linear, Ling 2.5, Nemotron 3 Nano) chose hybrid. The convergence pattern from M44 suggests this is the stable equilibrium, not pure-linear.

(4) **Closed-source frontier still uses softmax** : GPT-5.5, Claude Opus 4.7, Gemini 3.1 Pro all use full softmax (with various sparsity tricks at inference time). The leaders pulling the field don't show signs of switching to pure linear. Open-weight catch-up follows the leaders' patterns.

(5) **Long-tail tasks haven't been validated** : Mamba-3 results are from 1.5B-scale benchmarks. At frontier scale (300B+), pure linear hasn't been demonstrated to match transformers across the full task distribution. Risk of distribution-specific quality cliffs.

**The forecast** : by 2028, expect 50-75% of layers in production frontier models to be linear-attention variants (up from ~75% of layers in current hybrid models like Qwen3.5). Pure linear-attention frontier models remain niche; hybrid is the dominant pattern. Pure softmax frontier models also remain niche (Llama lineage, MiniMax M2.5); hybrid wins in the middle.

The deeper lesson: _"replacement" is the wrong frame_. The interesting question isn't whether linear replaces softmax; it's what mix of mechanisms wins. The current answer (hybrid) is likely stable.

### What just happened?

  * The four-axis taxonomy of attention. **Axis A** : semantic role and normalization (self/cross/causal; softmax/linear). **Axis B** : receptive field (full/sliding/global/sparse). Axes C, D, E covered in M46-M47.
  * **Self vs cross attention** : cross-attention mostly retreating in 2026 frontier models; early-fused multimodal turns most former cross-attention into self-attention.
  * **Causal masking** : a property of decoder-only architectures, not a separate mechanism. Lower-triangular mask forbids future-token leakage.
  * **Linear attention** : replaces softmax with kernel-trick φ(Q)φ(K)^T V; constant memory, linear compute. Three failed attempts (Performer, Linformer, RetNet) → 2024-2026 renaissance via selective decay, delta rule corrections, and (Mamba-3) complex-valued state.
  * **Mamba-3 (ICLR 2026)** : trapezoidal discretization, complex-valued state (equivalent to data-dependent RoPE), MIMO formulation. +1.8 points over Gated DeltaNet at 1.5B; matches Mamba-2 perplexity at half the state size; solves prior linear-attention state-tracking weakness.
  * **Gated DeltaNet (Yang et al. 2025)** : Mamba-2 gating + delta rule; the linear-attention layer in Qwen3-Next and Qwen3.5 (3:1 ratio with softmax attention).
  * **Kimi Delta Attention (KDA)** : GDN refined with channel-wise gating instead of scalar gating; used in Kimi Linear architecture.
  * **Lightning Attention** : simpler linear-attention variant; used in Ling 2.5 (1T MoE, hybrid with MLA).
  * The 2026 production reality: **no frontier model uses linear attention exclusively** ; hybrid stacks (3:1 linear:softmax) dominate.
  * **Sliding window attention** : O(N×w) cost; Mistral signature; mostly displaced in 2026 frontier; hybrid layer in some architectures.
  * **Trainable sparse attention family (2025-2026)** : **NSA** (Yuan et al., ACL 2025): three paths (compressed + selected + sliding) with learned gates; surpasses full attention. **MoBA** (Lu et al.): trainable block routing, used in Kimi family. **DSA** (DeepSeek V3.2 → V4): element-wise trainable sparsity with FP4 indexer, foundation of CSA+HCA. **XAttention** (MIT-Han Lab, ICML 2025): plug-and-play antidiagonal scoring, 13.5× speedup, no retraining. **SeerAttention, InfLLM-V2, FSA, DMA** : broader trainable sparse family.
  * NSA hyperparameters: window w=512, compression block size=32, stride=16, selection block size=64, top-n=16. Empirically validated at 27B/260B-tokens scale.
  * The trajectory: **fixed-pattern sparse (BigBird, Longformer 2020) → trainable hierarchical (NSA 2025) → element-wise trainable (DSA 2026) → hybrid attention (DeepSeek V4 CSA+HCA 2026)**.
  * Production cross-reference: Qwen3-Next/3.5 use GDN+softmax hybrid; Kimi Linear uses KDA+MLA hybrid; Ling 2.5 uses Lightning+MLA; Nemotron 3 Nano uses Mamba-Transformer hybrid; DeepSeek V4 uses CSA+HCA sparse softmax; GLM-5.1 uses DSA; Llama 4 uses standard full softmax with iRoPE; Mistral lineage uses sliding window; closed-source flagships undisclosed.
  * The reflex for 2026: **identify the workload first** — context length, retrieval requirements, training-from-scratch ability — then pick the Axis A and Axis B choices. M46 covers Axis C; M47 covers Axes D and E.

Module 46 (next) tackles **Axis C — head topology and KV-cache engineering** : the deep dive on MHA → MQA → GQA → MLA → IHA → Slim Attention, the decoupled RoPE math that makes MLA work, the IHA pseudo-head construction, and the V-from-K reconstruction in Slim Attention.
