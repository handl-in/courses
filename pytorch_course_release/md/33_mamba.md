# Module 33 — Mamba and state-space models

# Mamba and _state-space models_

_Part IX · Module 33 · the parallel architecture_

— a non-transformer architecture with O(T) time and fixed-size state, the parallel-scan kernel that makes it fast on GPUs, and why "transformers vs SSMs" turned out to be a false dichotomy

\--- 

Every module so far has been transformer-centric. M33 steps outside that frame and looks at the most credible alternative architecture of the past few years: **state-space models (SSMs)** , particularly the Mamba family. SSMs swap attention's quadratic cost and ever-growing KV cache for a fixed-size hidden state and linear-time updates — at least in principle.

The pedagogical reason this module is worth your time isn't to convince you to abandon transformers — most production models in 2026 are still transformers. It's that **the kernel patterns and engineering tradeoffs in SSMs map directly onto everything you've learned**. The selective-scan kernel that powers Mamba uses the same memory-hierarchy mindset as FlashAttention (M25). The hybrid architectures that combine transformer and SSM blocks share the dispatcher (M19), the parallelism axes (M16-M18), and the inference systems (M27). M33 is a case study in "how a frontier architecture you might not have built before slots into the framework you've now built."

> **★ KEY IDEA**  
>  A **state-space model** evolves a fixed-size hidden state through a sequence: `hₜ = A·hₜ₋₁ + B·xₜ; yₜ = C·hₜ`. The state size doesn't grow with sequence length — solves the O(T²) attention problem and the unbounded-KV-cache problem in one move. **Mamba's contribution** was making A, B, C _input-dependent_ ("selective"), which restored the model quality SSMs needed to be competitive with transformers. The cost: input-dependent recurrence breaks the convolutional shortcut classical SSMs used; you need a real sequential scan, which is hard on GPUs. **The selective-scan kernel** uses the parallel-scan algorithm (Blelloch tree, log T depth) to run a recurrence on a GPU in parallel — same memory-hierarchy patterns as FlashAttention. **Mamba-2** later showed SSMs and a special form of attention are mathematically equivalent, letting Mamba-2 use matmul-heavy kernels and converging the two architecture families. The 2026 reality: hybrid architectures (transformer + SSM blocks) win on many tasks; pure transformers still dominate; pure SSMs occupy specific niches. 

## Two new faces — the last new characters

S

State

"I'm a fixed-size hidden vector that gets updated as the sequence flows past."

In a transformer, every previous token gets cached in the KV cache (M27) and lives forever in memory — that's why long contexts get expensive. _I'm different._ I'm a single vector of dimension N (typically 64-256), and I don't grow. After processing token 1, I've absorbed token 1's information; after token 1000, I've absorbed all of them — but I'm still N-dimensional. The catch: I have to **compress** all that history into N dimensions. If I'm too small, I lose information; if I'm well-designed, I capture what matters. _I trade memory cost for compression difficulty_. Mamba's job is to compress well.

↯

Scan

"I run a recurrence in parallel on the GPU. Logarithmic depth, not linear."

A naive recurrence is `for t in range(T): h_t = f(h_{t-1}, x_t)` — sequential, no parallelism, terrible for GPUs. But if `f` is associative, the recurrence is a _scan_ (the prefix-sum generalization), and the **Blelloch parallel scan algorithm** runs it in `log₂(T)` depth on the GPU. Like a tree reduction, but producing all prefix-sums, not just the total. Mamba's selective scan is structured so this works. The kernel pattern: tile-based, same memory-hierarchy mindset as FlashAttention (M25) — keep state tiles in shared memory, stream input through, write output once. _Sequential-looking algorithms can run in parallel if they have associativity._

## The motivation: attention's two costs

From M22 and M27, you know two facts about transformer attention:

  1. **Quadratic compute** : attention is O(T²) in sequence length. FlashAttention (M25) reduces the memory traffic but doesn't change the FLOPs.
  2. **Unbounded KV cache** : every token's K and V stay in memory for all subsequent decodes. Cost grows linearly with conversation length.

Both problems disappear if you use a recurrent architecture. RNNs in the 2010s had a fixed hidden state and O(T) time — so why didn't they win? Two reasons. (1) **Quality** : vanilla RNNs and LSTMs underperformed transformers at scale on language tasks. The compression-into-fixed-state cost too much information. (2) **Throughput** : RNN training requires sequential processing of the sequence, killing GPU utilization.

State-space models are the modern attempt to fix both. The S4 paper (2021) showed a careful state-space parameterization could match transformer quality on long-context tasks. Mamba (2023) added input-dependence ("selectivity") and a fast parallel-scan kernel. Mamba-2 (2024) connected SSMs to attention mathematically and showed how to run them with matmul-heavy kernels. The story is iterative architectural improvement combined with careful kernel engineering.

## The structured state space model

Start with the simplest version. A continuous-time linear dynamical system:
    
    
    ẋ(t) = A·x(t) + B·u(t)        # state evolves under input u
    y(t) = C·x(t)                  # output is a projection of state

Discretize this for token-by-token processing — replace the continuous derivative with a finite difference. You get the discrete-time recurrence:
    
    
    hₜ = Ā·hₜ₋₁ + B̄·xₜ              # state update
    yₜ = C·hₜ                      # output

Where:

  * `xₜ ∈ ℝ` is the input scalar at position t (per-channel; for D channels you have D parallel SSMs).
  * `hₜ ∈ ℝᴺ` is the hidden state of dimension N (typically 16, 64, or 256).
  * `Ā ∈ ℝᴺˣᴺ` is the discrete-time state matrix.
  * `B̄ ∈ ℝᴺ` is the input projection.
  * `C ∈ ℝᴺ` is the output projection.
  * `yₜ ∈ ℝ` is the output scalar at position t.

If `Ā, B̄, C` are constant (don't depend on input), you can unroll the recurrence in closed form:
    
    
    yₜ = C · (Āᵗ⁻¹·B̄·x₀ + Āᵗ⁻²·B̄·x₁ + ... + Ā⁰·B̄·xₜ)
       = sum_{k=0}^{t} (C·Āᵗ⁻ᵏ·B̄) · xₖ

This is a **convolution** of the input with a fixed kernel `K_k = C·Āᵏ·B̄`. So a linear time-invariant SSM can be computed as either (a) a sequential recurrence, or (b) a parallel convolution. The convolution form runs efficiently on a GPU using FFT, with O(T log T) complexity. _This is what S4 did._

## The selective scan: Mamba's contribution

The catch with linear time-invariant (LTI) SSMs: the same A, B, C apply to every token, regardless of content. If you're processing "the cat that I saw yesterday was sleeping," the state-update mechanism doesn't change based on whether the current token is "cat" (a content word) or "the" (a function word). For language, this matters — content-dependent attention is a big part of what makes transformers good.

Mamba's fix: make B, C, and the time-step ∆ _input-dependent_. Specifically:
    
    
    B̄ₜ = linear_B(xₜ)           # now B depends on input
    C̄ₜ = linear_C(xₜ)           # C also input-dependent
    ∆ₜ = softplus(linear_∆(xₜ))  # time-step also input-dependent
    Āₜ = discretize(A, ∆ₜ)       # A is structured (often diagonal); discretized via ∆ₜ

Now the recurrence is "selective": each step's state-update depends on the current token, allowing the model to choose what to remember and what to forget. This restores the content-aware processing that vanilla SSMs lacked.

The cost: the convolutional shortcut is gone. With input-dependent A, you can't pre-compute a fixed kernel. **You have to run the recurrence as a true scan**. On GPUs that's a problem — sequential scans don't parallelize naturally.

The fix is the parallel scan algorithm.

## The parallel scan

Blelloch parallel scan: a recurrent computation in log₂(T) depth on the GPU Naive sequential — T steps, no parallelism: h₀ h₁ h₂ h₃ … T steps total, latency = T Parallel scan (T=8): tree-shaped, log₂(8) = 3 depth, all positions computed concurrently: level 0: A₀,B₀x₀ A₁,B₁x₁ A₂,B₂x₂ A₃,B₃x₃ A₄,B₄x₄ A₅,B₅x₅ A₆,B₆x₆ A₇,B₇x₇ level 1: combine(0:1) combine(2:3) combine(4:5) combine(6:7) level 2: combine(0:3) combine(4:7) level 3: combine(0:7) — final state contains everything key requirement: the combine operator must be associative selective scan IS associative if structured properly → log T parallelism

Read the diagram bottom-up. The naïve sequential scan does T steps, each waiting for the previous. The parallel scan organizes the same computation as a tree: in log₂(T) levels, pairs combine into pairs of pairs, into pairs of pairs of pairs, and so on. Each level can run in parallel across all its pairs. At T=8, the depth is 3 instead of 8.

The requirement is **associativity** of the combine operator. For ordinary scans (prefix sum), the operator is +, which is associative. For Mamba's selective scan, the combine operator pairs up consecutive transitions:
    
    
    # Combine two consecutive SSM steps into one effective step:
    # Step a: h_a = A_a · h_prev + B_a · x_a
    # Step b: h_b = A_b · h_a    + B_b · x_b
    # Combined: h_b = (A_b · A_a) · h_prev + (A_b · B_a · x_a + B_b · x_b)
    # So the combined "effective" matrices are:
    #   A_combined = A_b · A_a
    #   B_combined·x_combined = A_b · B_a · x_a + B_b · x_b

Combining two combined-pairs follows the same rule. The operation is associative: combine((a,b), (c,d)) = combine(a, combine(b,c,d)). _The selective scan, despite looking sequential, is mathematically a parallel-scannable operation_.

The Mamba kernel implements this Blelloch scan over the input sequence. Block-tile structure, with each block handling a chunk of sequence positions, scanning within the chunk in registers/shared memory and exchanging chunk-boundary states across blocks. **Same memory-hierarchy mindset as FlashAttention (M25)** : the working data stays in shared memory; HBM is touched only at the block boundaries.

## The Mamba block

The selective scan is the core, but a real Mamba block has additional components for stability and quality. The full block:
    
    
    class MambaBlock(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            d_inner = expand * d_model
    
            # Input projection: x → (xz, gated path)
            self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
    
            # 1D causal conv (short-range mixing — captures local patterns the SSM may miss)
            self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                    groups=d_inner, padding=d_conv - 1)
    
            # SSM parameters (selective: B, C, ∆ are input-dependent)
            self.x_proj = nn.Linear(d_inner, d_state * 2 + d_inner, bias=False)
            self.dt_proj = nn.Linear(d_inner, d_inner)        # for ∆
    
            # A is structured: (d_inner, d_state) parameters, learned in log space
            A = torch.arange(1, d_state + 1).repeat(d_inner, 1).float()
            self.A_log = nn.Parameter(torch.log(A))
    
            # Output projection
            self.out_proj = nn.Linear(d_inner, d_model, bias=False)
    
        def forward(self, x):                              # x: [B, T, d_model]
            # 1. Input projection splits into x and z (gating)
            xz = self.in_proj(x)                            # [B, T, 2*d_inner]
            x, z = xz.chunk(2, dim=-1)                     # [B, T, d_inner] each
    
            # 2. Causal 1D conv (short-range token mixing)
            x = x.transpose(1, 2)                          # [B, d_inner, T] for conv
            x = self.conv1d(x)[:, :, :T]                  # trim to T
            x = F.silu(x).transpose(1, 2)                   # [B, T, d_inner]
    
            # 3. Compute input-dependent SSM parameters
            x_proj = self.x_proj(x)                          # [B, T, d_state*2 + d_inner]
            B, C, dt_inner = torch.split(x_proj,
                                              [d_state, d_state, d_inner], dim=-1)
            delta = F.softplus(self.dt_proj(dt_inner))     # [B, T, d_inner]
            A = -torch.exp(self.A_log)                      # [d_inner, d_state]
    
            # 4. The selective scan! This is the kernel.
            y = selective_scan(x, delta, A, B, C)         # [B, T, d_inner]
    
            # 5. Gate output by z
            y = y * F.silu(z)
    
            # 6. Output projection
            return self.out_proj(y)

Read the structure top-to-bottom:

  * **Input projection splits into`x` and `z`**: `x` is what goes through the scan; `z` is a gate that modulates the output (similar to SwiGLU in transformers).
  * **1D causal conv with kernel size 4** : captures very short-range token interactions before the SSM. SSMs alone can miss exact-position-N-back patterns; the conv fills that gap.
  * **Input-dependent B, C, ∆** : this is the "selective" part. Three small linear projections from `x` produce per-token B, C, and time-step.
  * **The selective_scan call** : the kernel where almost all the work happens. Internally: parallel scan over T positions, with input-dependent A·exp(∆), B·exp(∆), and C-projection.
  * **Output gating** : `y * silu(z)` — same multiplicative gate as SwiGLU.
  * **Output projection** : project back from `d_inner` (= 2·d_model) to `d_model`.

Total: roughly the same parameter count as a transformer block at the same d_model. The compute is also similar — the scan dominates at long sequences, the projections at short. Where Mamba wins is the fixed-state inference path: at decode time, the state is just the d_inner × d_state hidden state matrix, ~64 KB for typical configs. _The state replaces the KV cache from M27._

## Mamba-2: the convergence with attention

Mamba-2 (Dao et al, 2024) is the second-generation architecture. Two contributions:

  1. **A simpler parameterization** : instead of the full structured A matrix, use a scalar A per channel (input-dependent). Reduces compute and simplifies the scan.
  2. **The mathematical equivalence** : with this parameterization, the SSM is mathematically equivalent to a particular form of "structured masked attention." This means Mamba-2 can be implemented with the same matmul-heavy kernels that power transformers — including FlashAttention-like algorithms.

The equivalence is striking. Roughly: a Mamba-2 layer can be expressed as `Y = (LowerTriangularMask ⊙ (Q · K^T)) · V` for specific Q, K, V, and mask structures. The structure constrains it (it's not _full_ attention; it's restricted to a particular causal-masked form), but the kernel can use the same hardware-friendly tile-based matmul patterns as FlashAttention.

**The practical consequence** : Mamba-2 trains and runs efficiently on standard tensor-core hardware, without requiring specialized scan kernels. This made it much easier to scale and deploy. Production Mamba-2 models in 2026 use kernels that look more like FlashAttention than like the original Mamba's selective scan.

The deeper lesson: _"transformers vs SSMs" was a false dichotomy_. The two families are points in a design space of "how do you compute weighted contributions of past tokens efficiently on GPUs," and the practical engineering choices converge.

## Hybrid architectures: the production reality

By 2026, the most successful applications of SSMs are not pure Mamba models but **hybrid architectures** : alternating transformer and SSM blocks within the same model. Examples:

  * **Jamba** (AI21, 2024): alternates Mamba blocks with transformer blocks, ~7:1 ratio. Extends to 256K context.
  * **Mamba-Codestral** : Mistral's code-focused Mamba-2 model. Pure SSM at ~7B scale.
  * **Hymba** , **Zamba** , **Samba** : research models exploring various transformer/SSM hybrid recipes.

The motivation for hybrids: SSMs and transformers have _complementary failure modes_. SSMs are great at long-context throughput but can struggle with exact retrieval ("what was that token at position 47?"). Transformers excel at retrieval but struggle with long-context inference cost. Alternating block types lets each handle what it's good at.

The architectural pattern:
    
    
    class HybridModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.blocks = nn.ModuleList()
            for i in range(cfg.n_layer):
                if i % cfg.attn_period == 0:
                    # every Nth block is a transformer block
                    self.blocks.append(TransformerBlock(cfg))
                else:
                    self.blocks.append(MambaBlock(cfg.d_model))
            self.norm = RMSNorm(cfg.d_model)
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

For Jamba's recipe: every 8th block is attention; the rest are Mamba. The KV cache (M27) only exists for the attention blocks — much smaller than a pure transformer. The state buffers exist for the Mamba blocks. _Inference systems that handle Jamba (vLLM and similar) need to manage both_.

## The empirical state in 2026

What's actually winning, two years after Mamba?

Architecture choice by application domain (2026) Application| Dominant architecture| Notes  
---|---|---  
Frontier chat models (Llama, Claude, GPT)| Pure transformer| Attention's flexibility wins; quality is paramount; long-context tricks (M30) keep transformers competitive  
Long-context reasoning| Transformer + careful inference| Paged KV (M27) + speculative decode handle the cost  
Code generation| Transformer or hybrid| Mamba-Codestral shows pure-SSM viability; transformers still dominate  
Edge / mobile inference| Hybrid (Jamba-style)| Smaller KV cache makes constrained-memory deployment easier  
Time-series and signal processing| Pure SSM (S4, Mamba)| SSMs excel where signals have known structure; the original SSM win-zone  
Genomics / long sequences| Pure SSM or hybrid| Sequences in hundreds of thousands of tokens; transformer attention is cost-prohibitive  
Vision (image and video)| Transformer (ViT, DiT) or hybrid| Pure transformers dominate; SSM-ViT variants exist but haven't taken over  
  
The honest summary: **transformers won the language race**. SSMs are valuable in specific domains (long sequences, signals, edge deployment) and as components in hybrid architectures. The "Mamba kills the transformer" headlines from 2023 didn't pan out, but the engineering ideas — input-dependent recurrence, parallel-scan kernels, hybrid architectures — are now part of the standard toolkit.

## The selective-scan kernel walkthrough

For completeness, the high-level structure of the selective scan kernel in Triton. This is what powers Mamba-1; Mamba-2 uses matmul-heavy kernels instead.
    
    
    @triton.jit
    def selective_scan_kernel(
        x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, y_ptr,
        B_size, T, D, N,                           # batch, time, channels, state_dim
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        # Each program handles one (batch, channel-tile) pair.
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
    
        # 1. Load A for this channel-tile (small, fits in registers).
        A = tl.load(A_ptr + ...)                 # [BLOCK_D, N]
    
        # 2. Initialize state to zero.
        state = tl.zeros([BLOCK_D, N], dtype=tl.float32)
    
        # 3. Stream through time tiles.
        for t_start in range(0, T, BLOCK_T):
            # Load this tile's inputs and per-token parameters.
            x      = tl.load(x_ptr  + ...)          # [BLOCK_T, BLOCK_D]
            dt     = tl.load(dt_ptr + ...)          # [BLOCK_T, BLOCK_D]
            B_t    = tl.load(B_ptr  + ...)          # [BLOCK_T, N]
            C_t    = tl.load(C_ptr  + ...)          # [BLOCK_T, N]
    
            # Discretize: A_eff = exp(dt · A), B_eff = dt · B
            # (in fp32; cast back at the end)
            A_eff = tl.exp(dt[:, :, None] * A[None, :, :])  # [BLOCK_T, BLOCK_D, N]
            B_eff = dt[:, :, None] * B_t[:, None, :]      # [BLOCK_T, BLOCK_D, N]
    
            # Within-tile parallel scan (Blelloch tree over BLOCK_T).
            state_tile = parallel_scan_within_tile(state, A_eff, B_eff, x)
    
            # Compute outputs y = C · state for each time step in tile.
            y = tl.sum(state_tile * C_t[:, None, :], axis=-1)   # [BLOCK_T, BLOCK_D]
            tl.store(y_ptr + ..., y)
    
            # Update state to last position's state — carries to next tile.
            state = state_tile[-1]                  # [BLOCK_D, N]

The patterns will look familiar from M24 and M25:

  * **Tile-based** : process BLOCK_T positions at a time. State stays in registers across tiles within one program.
  * **fp32 accumulators** : the recurrence uses fp32 internally even when inputs are bf16. Same recipe as M14, M23, M25.
  * **Block-internal parallel scan** : within each tile, run the Blelloch scan in shared memory. Across tiles, just carry the state forward (sequentially across tiles, parallel within).
  * **Memory-bound at most sizes** : the per-token compute is small; the kernel is dominated by reading A, B, C, x from HBM. Tile-based access keeps things efficient.

The reference Mamba kernel is ~250 lines of Triton. Same kernel-engineering toolkit, applied to a different mathematical structure. _If you can read M25's FlashAttention kernel, you can read this one_.

#### Q&A; — About SSMs and Mamba **Q:** Why is the state size N typically small (16-256)? **A:** Two reasons. (1) **Memory cost** : at inference, the state buffer is `d_inner × N` per layer. For d_inner=2048 and N=16, that's 32K fp32 numbers = 128 KB per layer per token. Doubling N doubles the inference memory. Keeping N small keeps the SSM's "fixed state" win meaningful. (2) **Quality plateau** : empirically, increasing N past 64-128 gives diminishing returns. The information bottleneck of the recurrence helps more than it hurts up to a point; past that, you're just adding parameters without adding capacity. _The "right" N depends on the task_ ; long-context retrieval-heavy tasks benefit from larger N; pure language modeling is fine with N=16. **Q:** Doesn't a fixed-size state lose information about long sequences? **A:** Yes — and this is the central tradeoff. The state can only carry forward what it has compressed; tokens far back in the sequence get squeezed through repeated applications of A. For tasks needing exact retrieval ("what was the second word of the prompt?"), this is bad. For tasks where local context dominates ("predict the next word in this sentence"), it's fine. Mamba's selectivity helps — input-dependent A means the state can choose what to remember and what to forget — but it doesn't fully solve the retrieval problem. **This is why hybrid architectures with some attention layers exist** : those layers handle the retrieval queries. **Q:** If Mamba-2 is mathematically equivalent to a form of attention, what's the actual difference? **A:** The difference is the _structure_ of the equivalent attention. Standard transformer attention has free-form Q, K, V (any matrix can attend to any matrix). Mamba-2's "structured masked attention" form constrains Q, K, V to a specific input-dependent decomposition. This constraint reduces compute and parameter count, but also restricts the model's flexibility. _Mamba-2 is best understood as "constrained attention with parallel-scan kernel options"_ — not a fundamentally different architecture, but a constrained subset of attention with engineering advantages in some regimes. **Q:** Why does the 1D causal conv exist in the Mamba block? Doesn't the SSM handle sequence mixing? **A:** The SSM mixes via the recurrence — long-range — but it's not great at exact-N-back patterns ("what was the token 3 positions ago?"). The 1D conv with kernel size 4 explicitly captures positions 0, -1, -2, -3 of the input. Adding it before the SSM gives the model a clean way to handle local patterns. The conv is parameter-cheap (kernel size 4, depthwise) and its omission empirically hurts quality. _Architectural lesson: small architecture details can matter, especially when the main mechanism has known weaknesses_. **Q:** Can I use Mamba in a torch.compile graph? **A:** Yes, but: the standard Mamba implementations use custom CUDA kernels (or Triton kernels with the parallel scan). torch.compile (M21) treats these as opaque ops via the dispatcher (M19), so the SSM block traces fine, but compile won't _fuse_ through the scan. The fusion you get is around the scan: input projections + scan + output projection becomes (projection-fused + opaque scan + projection-fused). For pure Mamba models, this is acceptable; for hybrids, the transformer blocks compile normally. _Inductor (M21) doesn't generate selective-scan kernels itself; you bring your own_. **Q:** What about training stability — Mamba had a reputation for being finicky? **A:** Mamba-1 indeed had stability issues (the recurrence can produce exploding or vanishing states without careful initialization). The standard fixes: initialize A in log space (so A is always negative, ensuring stability), use softplus on ∆ (positive time-steps), apply RMSNorm after the SSM. With these in place, Mamba trains stably. Mamba-2's simpler parameterization is more robust still. _For a from-scratch implementation, study a reference codebase carefully — the small init details matter_. **Q:** Is Mamba a good choice for a new project I'm starting? **A:** Probably not as the only architecture. Pure-Mamba pretraining is research territory; the recipes are less mature than transformer training. Hybrid architectures are credible but require maintaining two architectural paths (KV cache for attention blocks, state buffers for SSM blocks). For most projects, _start with a transformer_. The exception: if you have a clear reason — extreme long sequences (genomics, audio at high sample rate, time series), severe inference memory constraints (edge devices), or you're explicitly researching architecture — then SSMs become competitive. 

## Code Magnets: implement the Mamba block forward (high-level)

You're writing the Mamba block forward (not the inner scan kernel — the surrounding logic). Three magnets are wrong choices.

Arrange the magnets into a working forward pass.

def forward(self, x): xz = self.in_proj(x) x, z = xz.chunk(2, dim=-1) x = self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2) x = F.silu(x) B, C, dt_in = self.x_proj(x).split([self.d_state, self.d_state, self.d_inner], dim=-1) delta = F.softplus(self.dt_proj(dt_in)) delta = self.dt_proj(dt_in) A = -torch.exp(self.A_log) A = self.A_log y = selective_scan(x, delta, A, B, C) y = y * F.silu(z) y = y + z return self.out_proj(y)

show solution
    
    
    def forward(self, x):
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x = F.silu(x)
        B, C, dt_in = self.x_proj(x).split([self.d_state, self.d_state, self.d_inner], dim=-1)
        delta = F.softplus(self.dt_proj(dt_in))
        A = -torch.exp(self.A_log)
        y = selective_scan(x, delta, A, B, C)
        y = y * F.silu(z)
        return self.out_proj(y)

The traps:

  * `delta = self.dt_proj(dt_in)` (no softplus): the time-step ∆ must be positive (it's a discretization step size). Without softplus, ∆ can be negative, producing unstable or exploding state updates. Softplus enforces ∆ > 0 and provides a smooth gradient.
  * `A = self.A_log` (no negation, no exp): A is parameterized in log space and constrained to be negative for stability. The actual matrix A is `-exp(A_log)` — exponentiation produces positive numbers, then negation makes them negative. Skipping this step (using `A_log` directly as A) breaks the stability invariant; the SSM's eigenvalues can grow, producing explosions.
  * `y = y + z`: additive instead of multiplicative gating. SwiGLU-style gating uses multiplication (`y * silu(z)`) — the gate modulates which features pass through. Addition is just a residual connection on the gate path, which doesn't gate anything.

The pattern: **project, gate-split, conv, silu, project to per-token SSM params, softplus on ∆, negate-and-exp on A, scan, multiplicative gate with z, output projection**. Skipping any of softplus, the A parameterization, or the multiplicative gate breaks the model. The reference Mamba implementations include all three; learning Mamba is largely learning these small-but-essential details.

## Who does what?

Match each SSM concept to its real role.

Concept

Real role

State (h_t)

A. Fixed-size hidden vector (typically dim 16-256) carrying compressed history.

Selective parameters (B_t, C_t, ∆_t)

B. Input-dependent SSM coefficients; what makes Mamba selective vs vanilla SSMs.

Parallel scan (Blelloch)

C. Tree-structured algorithm running an associative recurrence in log T depth.

Selective scan kernel

D. Triton kernel implementing the parallel scan with tile-based memory hierarchy.

Mamba-2 / structured attention

E. Mamba-2 reformulation as a form of structured attention; matmul-friendly kernels.

1D causal conv (in Mamba block)

F. Short-range token mixing; covers positions the SSM may struggle with.

Hybrid architecture

G. Alternates transformer and SSM blocks (Jamba-style, ~7:1 ratio).

show solution

**State (h_t)** → A  
**Selective parameters** → B  
**Parallel scan** → C  
**Selective scan kernel** → D  
**Mamba-2 / structured attention** → E  
**1D causal conv** → F  
**Hybrid architecture** → G 

The mental shortcut: _state is fixed-size compressed history, selective parameters make the recurrence input-dependent, parallel scan parallelizes the recurrence on GPUs, the kernel implements it tile-based, Mamba-2 reformulates as attention, the conv handles local patterns, hybrids combine both block types_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team trains a pure Mamba model on a long-context QA task ("answer a question about a 32K-token document"). It does well on summarization but fails dramatically on questions like "what's the third word of paragraph 5?" Why?

show answer

The fixed-size state is the issue. Mamba compresses 32K tokens into a state of dimension `d_inner × N` (~64-256K parameters). For aggregate questions like summarization, this compression is fine — the answer depends on the gist, not specific tokens. For exact-position retrieval ("third word of paragraph 5"), the model needs to recall a specific token's content, which competes for state capacity with all the other tokens.

Two practical fixes: (1) **Switch to a hybrid architecture** like Jamba — interspersed attention blocks have unbounded retrieval capability via the KV cache; the SSM blocks handle long-range trends. (2) **Increase state size** N — gives more capacity, at cost of memory. Doubling N costs ~2× the inference memory but improves retrieval. For pure-Mamba retrieval, you'd typically want N=128-256 and accept the memory cost. _For tasks heavily dependent on exact retrieval, transformers (or hybrids) are still the right choice in 2026_.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why does the parallel scan algorithm work for Mamba's selective scan but not for, say, a vanilla LSTM?

show answer

The parallel scan requires the combine operator to be **associative**. For Mamba's selective scan, two consecutive transitions can be combined as:
    
    
    (A_a, B_a·x_a) ∘ (A_b, B_b·x_b) = (A_b·A_a, A_b·B_a·x_a + B_b·x_b)

This combine operation is associative because matrix multiplication is associative — combining (a,b) with c gives the same result as combining a with (b,c). Three or more transitions chain by the same rule.

For a vanilla LSTM, the recurrence involves nonlinearities (sigmoids on gates, tanh on the cell state):
    
    
    h_t = o_t · tanh(c_t),  c_t = f_t · c_{t-1} + i_t · tanh(...)

The nonlinearities mean the combine operator isn't associative — applying tanh after combining two steps gives a different result than combining "tanh(step a)" and "step b" (you can't pull tanh through linear combinations). This is why LSTMs are intrinsically sequential at training time: they can't be expressed as an associative scan. Mamba's structural choice — keep the recurrence linear, push nonlinearities outside the scan — is what makes parallel scan possible. _The architecture's parallelism story is baked into its mathematical structure_.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** Why is the Mamba kernel typically memory-bound rather than compute-bound, and what does this imply for hardware optimization?

show answer

Each step of the selective scan does very little arithmetic per token: a few multiplies and adds for the state update, an output projection. The total FLOPs per token are O(N) where N is the state dim — typically 16-64 ops. But the state, A, B, C, x, and y all need to be loaded from HBM (or carried across tiles) for each token. **Arithmetic intensity (FLOPs/byte) is low** — well below the H100 ridge point of ~333 ops/byte (M22). The kernel is memory-bound.

Implications for hardware optimization:

  * The kernel benefits from **shared memory tiling** — keep the state in SRAM across tile boundaries, reducing HBM round-trips for state. Same FlashAttention pattern.
  * **Quantization helps** — int8 or fp8 weights for B, C, and the scan parameters cut bandwidth ~2× without changing FLOPs. Less impactful than for transformer matmul (which is compute-bound at large sizes), but still worthwhile for inference.
  * **Tensor cores don't help much** — the kernel doesn't have a big matmul to feed them. This is one reason Mamba-2's matmul-heavy reformulation is faster on modern GPUs.
  * **Long sequences amortize launch overhead** — for short sequences, kernel launch dominates. Mamba shines at very long contexts where the per-token cost is what matters.

The takeaway: Mamba kernels follow the same engineering recipe as FlashAttention — shared-memory tiling, low-precision storage, fused operations — but the underlying compute pattern is different (scan, not matmul).

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** A team is debating between (a) a pure Mamba 7B model, (b) a hybrid 7B model with 7:1 SSM:attention ratio, (c) a pure transformer 7B model with sliding-window attention. They want to serve 128K-context conversations. What are the tradeoffs?

show answer

Each option has a different inference cost profile and quality character:

  * **(a) Pure Mamba** : KV cache replaced by state buffers (~64 KB/layer, fixed regardless of context). Decode latency dominated by state-update kernel. Quality on retrieval-heavy tasks may suffer. Inference memory: very low. Easiest to scale to 1M+ context.
  * **(b) Hybrid (7:1)** : 1/8 of layers are attention with full KV cache. KV cache size: 1/8 of pure transformer at same context. Quality: usually best of the three (retrieval handled by attention layers; long-range trends by Mamba). Inference memory: medium.
  * **(c) Transformer + sliding window** : KV cache bounded by window size (e.g., 4K). Tokens outside the window are forgotten. Quality on long-range dependencies suffers; quality on local tasks is best. Inference memory: bounded but small.

For 128K-context chat (typical of a long-document Q&A application): the hybrid (b) usually wins. Quality benefits from both architectures' strengths; memory is manageable. For pure throughput optimization where quality is fine: (a). For applications where you only need recent context: (c). _The decision is task-dependent; benchmarking on representative data is essential_. The right answer 5 years ago would have been (c); the right answer in 2026 is increasingly (b).

### What just happened?

  * State-space models (SSMs) evolve a **fixed-size hidden state** through a sequence: `h_t = A·h_{t-1} + B·x_t`. The state size doesn't grow — solves both attention's O(T²) cost and the unbounded KV cache problem.
  * **S4 (2021)** : linear time-invariant SSMs that can be computed as a parallel convolution.
  * **Mamba (2023)** : makes A, B, C, ∆ _input-dependent_ ("selective") — restores quality, but breaks the convolutional shortcut. Now you need a true sequential scan.
  * The **parallel scan algorithm** (Blelloch tree) runs an associative recurrence in `log₂(T)` depth on the GPU. Mamba's selective scan IS associative when structured properly.
  * The **selective-scan kernel** is tile-based, shared-memory-resident state, fp32 accumulators — same kernel-engineering recipe as FlashAttention (M25).
  * **The Mamba block** : input projection (split into x and z gate) → 1D causal conv → input-dependent SSM parameters → selective scan → multiplicative gating with z → output projection.
  * Stability tricks: `A = -exp(A_log)` ensures eigenvalues are negative; `∆ = softplus(...)` ensures positive time-steps; RMSNorm after SSM.
  * **Mamba-2 (2024)** : simpler parameterization; mathematically equivalent to a "structured masked attention." Can use matmul-heavy kernels — converges with transformer engineering.
  * **The 2026 reality** : pure transformers won the language race. SSMs are valuable for long-sequence domains (genomics, signals, time series), edge inference, and as components in **hybrid architectures** (Jamba alternates 7 Mamba blocks per attention block).
  * Trade-offs: SSMs lose information about specific past tokens (compressed state); transformers retain all of it (KV cache). Hybrid architectures get the best of both at moderate memory cost.
  * The kernel is memory-bound (low arithmetic intensity); benefits from shared-memory tiling and quantization but doesn't use tensor cores heavily.
  * The deeper lesson: _"transformers vs SSMs" was a false dichotomy_. They're points in a design space of efficient sequence-mixing on GPUs, and the engineering choices converge.
  * The reflex: when evaluating an architecture, ask "what's its access pattern?" — quadratic scan over all pairs (attention), fixed-state recurrence (SSM), or a constrained subset (Mamba-2). Each maps to a kernel pattern from M22-M25.

## Closing Part IX

Five extension modules — M29 built a transformer end-to-end, M30 unpacked RoPE, M31 covered post-training (SFT, RM, PPO, DPO), M32 consolidated debugging skills, and M33 explored Mamba as the parallel architecture.

What you have now: a course that takes you from `x.stride()` to `fully_shard` to `tl.dot` to MoE expert parallelism to RoPE rotation derivations to DPO loss to Blelloch parallel scans. Thirty-three modules, ~45 characters. The original capstone in M28 is still the synthesis exercise; M29's transformer is the synthesis you build with your own hands.

What's still missing from the course as a whole: everything mentioned in the "what's not covered" review — RLHF in finer detail, multimodal architectures, vision encoders, smaller-scale and edge deployment, AMD/TPU specifics, NCCL internals. Those are their own course-extensions if you want them.

The point isn't to know everything — nobody does — but to _have the framework to learn anything new in the field_. When the next architectural breakthrough lands, you'll have the dispatcher, the roofline, the parallelism axes, the tile-based kernel mindset, the post-training pipeline, the RoPE derivation, and the SSM design space to slot it in.

Thanks for reading this far. Now go make something.
