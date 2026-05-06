# Module 08 — Initialization, layers & norms

# Initialization, _layers & norms_

_Part III · Module 08_

— why your network dies at layer 20 if you use the wrong init, and the single axis decision that distinguishes BatchNorm from LayerNorm from RMSNorm

\--- 

You have `nn.Module`. Now you fill it. This module is about the small but vicious choices inside every layer: _how do I initialize weights so the network doesn't immediately collapse?_ and _what kind of normalization stops activations from drifting into NaN over deep stacks?_

These are not aesthetic choices. A deep network with default-PyTorch-Linear init and no normalization will train for a few steps and die. A correctly-initialized network with the right normalization can train for a million steps and stay stable. The difference is roughly six lines of code and zero added compute. Worth understanding.

> **★ KEY IDEA**  
>  Initialization and normalization solve the same underlying problem: **keeping the magnitude of activations and gradients roughly constant across layers**. Init does it at step 0 by choosing weight scales correctly. Normalization does it at every step by rescaling activations. They're complementary, not redundant — modern transformers use both. 

## Two new faces

σ

Init Scheme

"I'm the reason your network doesn't die at layer 20."

When you stack 20 randomly-initialized linear layers, the activations either explode (variance grows by a factor each layer) or collapse to zero (variance shrinks by a factor each layer). Either way, gradients vanish or explode and training fails. My job is to choose weight scales so that _the variance of activations stays roughly constant from input to output_. Xavier, Kaiming, GPT-style — they're all answers to the same question, with slightly different assumptions about what comes next (which activation, which depth, which residual structure).

N

Norm

"I subtract the mean and divide by the std. The only question is _over what_."

Whether I'm BatchNorm, LayerNorm, GroupNorm, or RMSNorm — the formula is essentially the same: `(x - mean) / sqrt(var + ε) * weight + bias`. The whole drama is which dimensions I average over. BatchNorm averages over the batch and spatial dims (per-channel statistics). LayerNorm averages over the feature dim (per-token statistics). GroupNorm splits the channels into groups and averages within each group. RMSNorm skips the mean entirely and just divides by the RMS. Pick wisely — the choice has consequences.

## Why initialization matters: the variance-preservation argument

Here's the simplest possible setup. A 20-layer linear network with no nonlinearities — just `x → W₁x → W₂(W₁x) → ...`. If `W` has entries drawn from `N(0, σ²)`, what variance does the output have, given input variance 1?

For a single layer: `y = Wx`, where `W` is shape `(D, D)`. Each output `y[i] = sum(W[i, :] * x)`. The variance of that sum is `D · σ²` (assuming x is unit-variance and zero-mean). So one layer multiplies the variance by `D · σ²`.

After 20 layers, the variance is multiplied by `(D · σ²)²⁰`. For `D = 512` and `σ = 0.1` (a "reasonable" guess), that's `(5.12)²⁰ ≈ 10¹⁴`. Activations explode. Gradients explode. Loss is NaN by step 1.

For variance to stay constant, we need `D · σ² = 1`, i.e. `σ = 1/√D`. That's the entire idea. Xavier and Kaiming are refinements that account for the activation function and the forward/backward asymmetry.

Activation variance through a 20-layer linear network σ = 0.1, D = 512 — wrong scale var=1layer 0 var≈5.12layer 1 var≈26layer 2 var≈690layer 4 var≈5e5layer 8 var≈10¹⁴layer 20 NaNsoon σ = 1/√D ≈ 0.044 — correct scale var=1layer 0 var≈1layer 1 var≈1layer 2 var≈1layer 4 var≈1layer 8 var≈1layer 20 trainsfine init is the cheapest training stability hack ever invented

## The init schemes you'll actually use

Three schemes cover ~95% of real models. Their formulas differ in the choice of `fan_in` (number of input units) vs `fan_out` (output units), and in the constant that accounts for the activation function.

The init schemes that earn their keep Scheme| Std formula| Designed for| Use it for…  
---|---|---|---  
**Xavier (Glorot)**| `σ = √(2 / (fan_in + fan_out))`| Symmetric activations (tanh, sigmoid)| Older architectures; most LayerNorm-stabilized models do fine with it  
**Kaiming (He)**| `σ = √(2 / fan_in)`| ReLU-family activations (the 2× compensates for ReLU killing half the variance)| CNNs, MLPs with ReLU/GELU before the next layer  
**GPT-style**| `σ = 0.02` for most, `σ = 0.02/√(2·n_layers)` for residual projections| Deep transformer stacks with residual connections| LLaMA, GPT, modern transformer training  
**Orthogonal**|  Random orthogonal matrix scaled by gain| RNNs, situations where you want strict isometry at init| Sequence models with long unrolling; less common today  
  
Two things worth saying about the "GPT-style" entry, because it's the modern default. (1) Most weights are just `N(0, 0.02²)` — surprisingly small, surprisingly fixed regardless of dim. (2) For weights at the end of a residual block (the projection that adds back to the residual stream), the std gets divided by `√(2·n_layers)`. This compensates for the fact that residuals add up over depth — without the correction, deeper transformers blow up their residual stream. Karpathy's nanoGPT made this widely known.

### Doing init in PyTorch
    
    
    import torch.nn as nn
    
    # PyTorch's default Linear init is roughly Kaiming uniform — usually fine
    # but you can override:
    
    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    model.apply(init_weights)        # walks every module in the tree

That's the canonical pattern: define a function that handles each module type, call `model.apply(fn)` to walk the tree. Note the trailing underscore — these are _in-place_ initializers (M1's underscore convention).

> **⚠ WARNING**  
>  **Bias initialization matters.** Always initialize biases to zero (or near-zero) unless you have a specific reason not to. Non-zero bias init breaks most variance-preservation arguments. The one common exception: the bias of the last layer in a binary classification model is sometimes init'd to `log(p/(1-p))` where p is the class prior — speeds up training when classes are imbalanced. 

### The GPT-style init in code
    
    
    def init_gpt_style(model, n_layers):
        std = 0.02
        proj_std = 0.02 / (2 * n_layers) ** 0.5
    
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                # Most weights: N(0, 0.02²)
                nn.init.normal_(p, mean=0.0, std=std)
                # Residual-block projections: scale down by √(2*n_layers)
                if name.endswith('attn.out_proj.weight') or name.endswith('mlp.down_proj.weight'):
                    nn.init.normal_(p, mean=0.0, std=proj_std)
            elif 'bias' in name:
                nn.init.zeros_(p)
            elif p.dim() == 1:
                # LayerNorm weights and similar 1-D things — init to 1
                nn.init.ones_(p)

The naming convention on the projections is brittle (depends on your layer naming), so most real codebases identify them via `isinstance` checks or special module subclasses. The principle is what matters.

## Normalization layers: the axis decision

This is where most engineers get hand-wavy. Let me be precise.

Every normalization layer does the same arithmetic: subtract the mean, divide by the standard deviation, scale by a learned weight, shift by a learned bias. _The only thing that changes across BN / LN / GN / IN / RMSNorm is which axes the mean and std are computed over._

Consider a tensor of shape `(N, C, H, W)` — batch, channels, height, width — typical for CNNs. Or `(B, T, D)` — batch, sequence, hidden dim — typical for transformers. The choice of axes determines the layer's character.

Same tensor (N, C, H, W). Different axes get normalized. BatchNorm over (N, H, W) per channel N1 N1 N1 N1 N2 N2 N2 N2 N3 N3 N3 N3 C1 C2 C3 C4 stats per channel, across batch + spatial LayerNorm over (C, H, W) per sample N1 N1 N1 N1 N2 N2 N2 N2 N3 N3 N3 N3 stats per sample, across all features GroupNorm over (group of C, H, W) per sample N1 N1 N1 N1 N2 N2 N2 N2 N3 N3 N3 N3 stats per group of channels, per sample (here: groups of 2) InstanceNorm over (H, W) per (sample, channel) N1 N1 N1 N1 N2 N2 N2 N2 N3 N3 N3 N3 stats per (sample, channel) — smallest scope colored cells = "averaged together for one mean and one std" BN: same channel across batch · LN: same sample across features GN: middle ground · IN: per (sample, channel) only

Read the diagram carefully — the colored cells are the ones that get averaged together to produce _one_ mean and _one_ standard deviation. Different sets of cells produce different means, hence different normalization characters.

### BatchNorm: per-channel statistics across the batch

For input `(N, C, H, W)`: compute mean and var _per channel_ , averaged over the batch and the spatial dims. So you get one (mean, var) pair per channel — `C` means and `C` vars.
    
    
    import torch.nn as nn
    bn = nn.BatchNorm2d(num_features=64)   # expects (N, 64, H, W)

Two key behaviors:

  * **Train mode** : compute mean/var from the current batch. Update running mean/var (the buffers we registered in M7) as exponential moving averages.
  * **Eval mode** : use the running mean/var, ignore the current batch. This is what gives you stable inference even on batch size 1.

BN's strength: enables training of very deep CNNs that wouldn't otherwise converge. BN's weakness: **requires batch size >= 8 or so to estimate stable per-channel statistics**. Small-batch training (memory-constrained) breaks it. Distributed training requires sync (Module 16). Sequence models hate it because the batch axis is heterogeneous.

### LayerNorm: per-sample statistics across the features

For input `(B, T, D)` (transformer-style): compute mean and var _per (batch, position)_ , across the feature dim. So you get `B × T` (mean, var) pairs.
    
    
    ln = nn.LayerNorm(normalized_shape=512)   # normalize over the last dim of size 512
    y = ln(x)                              # x: (B, T, 512)

LN doesn't use the batch dim at all, so batch size is irrelevant. There's no train/eval split for LN — it does the same thing in both modes. This makes it the natural fit for transformers: works at batch size 1, doesn't need running stats, doesn't break under distributed training.

The `normalized_shape` argument is subtle: it's _the shape of the trailing dims to normalize over_. For a transformer with shape `(B, T, D)`, you pass `D` (or `(D,)`). For multi-axis normalization (rare), you can pass `(C, H, W)` to normalize across all three.

### RMSNorm: drop the mean, keep the rescaling

The simplest and newest of the bunch. Identical to LayerNorm except: _don't subtract the mean, just divide by the RMS (root-mean-square)_. There's also no learned bias.
    
    
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps
    
        def forward(self, x):
            rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
            return x / rms * self.weight

Why does this work? Empirically, the mean-subtraction in LayerNorm doesn't add much when activations are already roughly zero-mean (which they are in deep networks with skip connections). Skipping the mean saves memory and one reduction. LLaMA, Mistral, and most modern LLMs use RMSNorm. PyTorch added `nn.RMSNorm` in 2.4.

### GroupNorm: the middle ground

For input `(N, C, H, W)`: split `C` into `G` groups, compute mean/var per `(sample, group)`. So you get `N × G` (mean, var) pairs.
    
    
    gn = nn.GroupNorm(num_groups=32, num_channels=512)
    # Splits 512 channels into 32 groups of 16

GroupNorm is what you reach for when you want the batch-independence of LN but with the channel-grouping flavor of CNNs. It works well in image generation (Stable Diffusion uses it) and in cases where batch size is small or variable.

#### Q&A; — About normalization choices **Q:** My CNN trains fine with BN. Should I switch to GN? **A:** Only if your batch size is small (<8) or you need batch-independent inference (e.g., real-time inference at batch size 1 where the running stats don't fit your data distribution). For standard training with batch sizes of 32+, BN is faster and usually slightly better. **Q:** Why does every modern transformer use LN or RMSNorm and not BN? **A:** Three reasons. (1) Sequence models often have variable-length sequences in a batch — BN's per-channel statistics get contaminated by padding. (2) Inference at batch size 1 is common — BN needs running stats; LN doesn't. (3) Transformers train at huge effective batch sizes via accumulation, where each micro-batch's statistics aren't representative; LN sidesteps the issue entirely. **Q:** Pre-norm vs post-norm — what's that about? **A:** Where you place the LN relative to the residual addition. Pre-norm: `x + Sublayer(LN(x))`. Post-norm: `LN(x + Sublayer(x))`. Pre-norm is the modern default; it's much more stable for deep stacks because the residual stream stays unnormalized and gradients flow cleanly through it. Post-norm was the original Transformer paper's choice; it tends to need careful learning-rate warmup. _Always pre-norm unless you have a specific reason._ **Q:** Should I use the affine parameters (weight and bias)? **A:** Almost always yes for LN/BN/GN. The affine restoration gives the network the ability to undo the normalization if it needs to, while still gaining the optimization benefits of normalized gradients. The default in PyTorch is to include them. RMSNorm only includes weight, not bias. 

## Activations, briefly

Most of the activation choice has been settled. The shortlist:

Activation functions you'll actually meet Activation| Formula| Use it for…  
---|---|---  
`ReLU`| `max(0, x)`| Old reliable. CNNs, MLPs, anything where you want speed.  
`GELU`| `x · Φ(x)` (Φ is normal CDF)| Default for transformers. Smooth, slight negative leakage.  
`SiLU` / `Swish`| `x · sigmoid(x)`| Modern transformers (LLaMA-style FFNs). Similar to GELU.  
`SwiGLU` (gated)| `SiLU(W₁x) · (W₂x)`| State-of-the-art FFN block. 2-layer MLP becomes 3 weights with gating.  
`tanh`| self-explanatory| Old RNNs, the value head of some RL nets.  
  
For deep networks today, GELU or SiLU are the safe choices. SwiGLU squeezes a bit more performance out of FFN blocks but uses more parameters (3 matrices instead of 2 for the same hidden dim). Most LLaMA-family models use SwiGLU.

## Embeddings: a Linear with integer indexing

`nn.Embedding` is conceptually a `(vocab_size, embed_dim)` weight matrix indexed by integer tokens. Forward pass is just `weight[indices]` — fancy indexing, returns shape `(*indices.shape, embed_dim)`.
    
    
    emb = nn.Embedding(num_embeddings=50257, embedding_dim=768)   # GPT-2 vocab
    tokens = torch.tensor([[15496, 995, 11, 995]])                # shape (1, 4)
    out = emb(tokens)                                          # shape (1, 4, 768)

Init for embeddings is conventionally `N(0, 0.02²)`, same as GPT-style. Not Kaiming, because the "fan_in" concept doesn't apply — the input is an integer index, not a vector of activations.

For very large vocabularies (LLM-scale), the embedding table dominates parameter count. This is why **weight tying** (M7) — sharing the embedding with the output projection — is standard.

## Dropout, briefly

Drops each element of the input independently with probability `p` during training; scales the kept ones by `1/(1-p)` to preserve the expected value. In eval mode: identity.
    
    
    drop = nn.Dropout(p=0.1)
    y = drop(x)             # scaled-dropped in train mode, identity in eval mode

Modern transformers use very low dropout (0 or 0.1) — large training corpora make heavy regularization unnecessary. CNNs and smaller-scale models can use higher (0.2-0.5).

Dropout is _the_ archetype of "train/eval matters" — it's why `model.eval()` exists. If you forget `eval()` at inference, dropout is still firing, and your outputs are stochastic.

## Code Magnets: build the canonical transformer block

You're building a single transformer block: pre-norm, multi-head attention, residual, pre-norm, MLP, residual. Use LayerNorm, GELU activation, and apply Dropout. Build the `forward`.

Arrange the magnets into a working `forward`. Three are red herrings.

def forward(self, x, attn_mask=None): x = x + self.dropout(self.attn(self.ln1(x), attn_mask=attn_mask)) x = self.ln1(x + self.dropout(self.attn(x, attn_mask=attn_mask))) x = x + self.dropout(self.mlp(self.ln2(x))) x = self.ln2(x + self.dropout(self.mlp(x))) return x return self.ln_final(x)

show solution
    
    
    def forward(self, x, attn_mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), attn_mask=attn_mask))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x

The traps:

  * `x = self.ln1(x + ...)` is post-norm. Works, but harder to train deep stacks. Pre-norm puts the LN _inside_ the residual: `x + Sublayer(LN(x))`.
  * `x = self.ln2(x + ...)`: same post-norm mistake for the MLP block.
  * `return self.ln_final(x)`: a final LN can exist (GPT-2 has one at the end of the stack), but it's not part of the per-block return — it'd cause a double LN with the next block's pre-norm.

The structural rule: each sublayer is `x = x + Sublayer(LN(x))`. Two sublayers per block (attention, MLP). Each contributes to the residual stream without normalizing it.

## Who does what?

Match each layer/init concept to its purpose.

Concept

Purpose

Kaiming init

A. Normalize per-channel using running stats during eval; needs batch size to be reasonable.

GPT-style residual init

B. Per-sample, across-features normalization; works at batch size 1.

BatchNorm

C. Drop the mean from LayerNorm; just divide by RMS.

LayerNorm

D. Variance-preserving init for ReLU-family activations; uses fan_in.

RMSNorm

E. Scales residual-projection weights down by √(2·n_layers) to prevent residual blow-up.

SwiGLU

F. Gated MLP: SiLU(W₁x) · (W₂x). Better than vanilla MLP at the same parameter budget.

show solution

**Kaiming init** → D  
**GPT-style residual init** → E  
**BatchNorm** → A  
**LayerNorm** → B  
**RMSNorm** → C  
**SwiGLU** → F 

The mental shortcut: _Kaiming for ReLU, GPT-init scales residuals, BN normalizes per-channel-cross-batch, LN per-sample-cross-features, RMSNorm skips mean, SwiGLU is the modern gated MLP_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** Implement `LayerNorm` from scratch as an `nn.Module`. Verify it matches `nn.LayerNorm` on a random input.

show answer
    
    
    class MyLN(nn.Module):
        def __init__(self, dim, eps=1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias   = nn.Parameter(torch.zeros(dim))
            self.eps = eps
    
        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            var  = x.var(dim=-1, keepdim=True, unbiased=False)
            x_hat = (x - mean) / (var + self.eps).sqrt()
            return x_hat * self.weight + self.bias
    
    x = torch.randn(2, 3, 8)
    assert torch.allclose(MyLN(8)(x), nn.LayerNorm(8)(x), atol=1e-6)

Two things to note: (1) `unbiased=False` — LN uses the population variance, divide by N not N−1. (2) The default `weight` is ones and `bias` is zeros, so right after init the layer is the identity (apart from numerical normalization).

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** What's the variance of the activations after a single Linear layer initialized with `std = 1/√(fan_in)`, given input variance 1? Do the math.

show answer

For `y = Wx` where `W` is shape `(D_out, D_in)` with entries ~ N(0, σ²) and `x` ~ N(0, 1) elementwise:

`Var(y[i]) = sum over j of Var(W[i,j]·x[j]) = D_in · σ² · 1 = D_in · σ²`.

With `σ² = 1/D_in`, we get `Var(y[i]) = 1`. ✓ Variance is preserved. This is exactly Xavier/Kaiming for the linear case (without an activation function). The activation function adds a multiplicative correction — for ReLU it's a factor of 2 (since ReLU zeros half the variance), which is why Kaiming uses `2/fan_in`.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** A model uses BatchNorm and trains fine, but inference produces wildly different outputs depending on what other samples are in the batch. What's wrong?

show answer

The model is in train mode at inference time. `model.train()` is on, so BatchNorm is computing per-batch statistics from whatever samples are currently in the batch. Each different batch composition gives different normalization. Fix: `model.eval()` before inference, which switches BN to using running stats (computed during training and stored as buffers). This is the canonical "I trained a CNN and inference is broken" bug.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Write the GPT-style init for a transformer where you can identify residual-projection weights by their suffix (e.g., names ending in `'.out_proj.weight'`).

show answer
    
    
    def init_gpt_style(model, n_layers):
        std = 0.02
        proj_std = std / (2 * n_layers) ** 0.5
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                if name.endswith('.out_proj.weight'):
                    nn.init.normal_(p, mean=0.0, std=proj_std)
                else:
                    nn.init.normal_(p, mean=0.0, std=std)
            elif 'bias' in name:
                nn.init.zeros_(p)
            elif p.dim() == 1:
                nn.init.ones_(p)

The check `p.dim() >= 2` picks up weight matrices; `p.dim() == 1` catches LayerNorm weights and similar 1-D tensors that should init to 1, not 0. Biases (also 1-D, but identifiable by name) get zeros.

### What just happened?

  * **Initialization preserves activation variance across layers.** Wrong scale → exponential blowup or collapse → NaN or dead network. Right scale → variance ≈ 1 throughout.
  * **Kaiming init** (`σ = √(2/fan_in)`) is the right default for ReLU/GELU networks. **Xavier** is for symmetric activations (older). **GPT-style** uses fixed σ=0.02 plus residual-projection scaling by `√(2·n_layers)` — this is the modern transformer default.
  * Initialize biases to zero. Initialize LayerNorm weights to 1. Initialize embeddings with the same N(0, 0.02²) as GPT-style.
  * Use `model.apply(init_fn)` to walk the module tree and apply per-type init.
  * **Every normalization layer does the same arithmetic.** The character is determined entirely by _which axes the mean and std are computed over_.
  * **BatchNorm** : per-channel stats across batch + spatial. Needs reasonable batch size; uses running stats at eval. Default for CNNs.
  * **LayerNorm** : per-sample stats across features. Batch-size-independent. No train/eval split. Default for transformers.
  * **RMSNorm** : drop the mean from LN; just divide by RMS. Skips a reduction. Used in LLaMA, Mistral, modern LLMs. `nn.RMSNorm` in PyTorch 2.4+.
  * **GroupNorm** : middle ground — per-(sample, channel-group). Works well when batch is small and channels can be meaningfully grouped (Stable Diffusion uses it).
  * **Pre-norm** (`x + Sublayer(LN(x))`) beats post-norm for deep stacks. Modern default.
  * **Activations** : GELU or SiLU for transformers; SwiGLU for FFN blocks if you can afford the extra matrix; ReLU for everything else where speed matters.
  * **Embeddings** are conceptually a Linear indexed by integer tokens. Init with N(0, 0.02²). Often weight-tied with the output projection.
  * **Dropout** is the archetype of "train/eval matters." Forget `eval()` at inference and outputs become stochastic.
  * The reflex: when activations explode or collapse, suspect init first. When training is stable but you're hitting batch-size limits, check normalization choice. Both are five-line fixes that change everything.

Module 09 closes Part III with the optimizer side of the equation: loss functions you'll actually use (CE, BCE, focal, MSE, Huber), optimizer algorithms with their math (SGD, Adam, AdamW, Lion, Muon), learning-rate schedules (cosine, warmup, OneCycle), and gradient clipping.
