# Module 29 — Building a transformer from scratch

# Building a _transformer_ from scratch

_Part IX · Module 29 · synthesis_

— a complete GPT-style model in ~250 lines of PyTorch, every line explained, with KV cache wiring and a debugging checklist for first-run failures

\--- 

For 28 modules we've covered the components — tensors, autograd, modules, optimizers, attention kernels, distributed training, serving. _This module assembles them into a working model_. By the end, you'll have a complete GPT-style transformer that trains, generates text, and exposes the same KV cache interface that vLLM (M27) consumes. ~250 lines of code, every line motivated by a specific module's content.

The point isn't novelty. There are plenty of "build GPT from scratch" tutorials. The point is that, having seen the components, you should now see _why every architectural decision is the way it is_. The head dimension being a multiple of 8 is M22 (tensor cores). Pre-norm vs post-norm is M14 (numerical stability). The init scheme is M8. Causal masking integrates with attention from M25. The KV cache layout matches M27's paged design. **This is the module where the framework you've built becomes a competent engineer's reflex set.**

> **★ KEY IDEA**  
>  A GPT-style transformer is: **token embeddings → N transformer blocks (each: pre-norm + attention + residual; pre-norm + MLP + residual) → final norm → lm_head**. The attention block uses RoPE for positional encoding (M30 covers this in depth), Grouped-Query Attention for KV cache efficiency (M27), and PyTorch's `F.scaled_dot_product_attention` for FlashAttention dispatch (M25). The MLP uses SwiGLU (the modern gated variant). Init follows M8's scheme; the optimizer is AdamW (M9); training mixed-precision in bf16 (M14). The whole thing fits in ~250 lines. **Every line has a reason that traces back to a specific module.**

## Two final new faces

G

GPT

"I'm what you've been building toward — the assembled model."

I'm a stack of transformer blocks with a token embedding at the bottom and an LM head at the top. Each block has attention (Q, K, V projections, FlashAttention via SDPA, output projection) and an MLP (SwiGLU). I emit logits over the vocabulary; you sample from those to generate. I'm small (~85M params for a GPT-2-small clone, ~7B for a Llama-style 7B); the architecture is the same at every scale, just wider and deeper. _Every component you've met in this course is somewhere inside me._

▲

Causal Mask

"I'm the upper-triangular bouncer. Token i can't attend to token j when j > i."

In training, I make sure each position only sees the past. Without me, the model trivially cheats — it can read the next token's embedding directly. I'm logically a (T, T) boolean matrix where `mask[i, j] = (j ≤ i)`, but in modern kernels I'm not materialized: `F.scaled_dot_product_attention(..., is_causal=True)` tells FlashAttention (M25) to skip K/V tiles above the diagonal entirely. _Free at training time, free at prefill, automatic at decode_. I'm the simplest thing in the model, but forgetting me is the most common bug in first implementations.

## The architecture, top-down

GPT data flow: tokens → embedding → N blocks → norm → logits token_ids [B, T] int64 Embedding [vocab, D] lookup x: [B, T, D] × N transformer blocks Sub-block 1: Attention x = x + attn(norm₁(x)) (RoPE inside, FlashAttention via SDPA) Sub-block 2: MLP x = x + mlp(norm₂(x)) (SwiGLU: down(silu(gate)·up)) final norm RMSNorm lm_head (Linear) [D, vocab] logits [B, T, vocab_size] "residual stream": x flows top-to-bottom, each block adds its contribution pre-norm: norm BEFORE the sub-layer, residual ADDS the unnormed path lm_head shares weights with embedding (tied embeddings) — saves params N=12 for ~85M params (GPT-2-small); N=32 for ~7B (Llama-7B)

Read the diagram top-to-bottom: token IDs come in, get embedded, flow through N transformer blocks (each adds its contribution to the residual stream), get normed one final time, and project to vocabulary logits. Three structural decisions worth flagging:

  * **Pre-norm** : each sub-block is structured as `x + sublayer(norm(x))`, not `norm(x + sublayer(x))`. Pre-norm is more numerically stable at depth — gradients flow through the residual path unchanged, only the sub-layer's contribution gets normed. Post-norm is what the original Transformer paper used; modern transformers (GPT-2 onward) all use pre-norm. _The exception that proves the rule_ : post-norm needs careful warmup; pre-norm trains stably from step 0.
  * **RMSNorm, not LayerNorm** : RMSNorm drops the mean-subtraction step (M8). Same quality, ~30% faster, used by Llama, Mistral, Gemma. We'll use it.
  * **Tied embeddings** : the `lm_head` Linear shares weights with the input embedding. Saves `vocab_size × D` parameters (a few percent of the model for typical vocab sizes). Most modern GPTs do this; Llama-3 stopped tying for the largest sizes.

## Building it: the components

Six components, one at a time. Each ~30-50 lines.

### 1\. Config

Hold all the model dimensions in one dataclass. Makes scaling experiments trivial.
    
    
    from dataclasses import dataclass
    
    @dataclass
    class GPTConfig:
        vocab_size:    int = 50304     # GPT-2 BPE; bumped to multiple-of-64 for tensor-core dim alignment (M22)
        n_layer:       int = 12        # GPT-2-small depth
        n_head:        int = 12        # number of query heads
        n_kv_head:     int = 4         # number of key/value heads (GQA, M27)
        d_model:       int = 768       # hidden dim; per-head dim = d_model / n_head = 64
        d_mlp:         int = 2048      # MLP hidden; ~2.67× d_model for SwiGLU (vs 4× for GELU)
        max_seq_len:   int = 2048      # context length
        rope_base:     float = 10000.0   # RoPE θ-base; 500000 for long context (M30)
        norm_eps:      float = 1e-6      # RMSNorm epsilon
        tied_embeds:   bool  = True      # share input emb with output head
        init_std:      float = 0.02      # init std for linear weights (M8)
    
        @property
        def head_dim(self) -> int:
            return self.d_model // self.n_head

Several knobs encode lessons from earlier modules. `vocab_size = 50304` instead of GPT-2's actual 50257 — the next multiple of 64, so the LM head matmul is tensor-core-friendly (M22). `head_dim = 64` (multiple of 8). `n_kv_head = 4` means 3:1 query-to-KV ratio (GQA-3, similar to Llama-3-8B's GQA). `d_mlp = 2048` instead of the ~3072 a 4× expansion would give — SwiGLU has a gate path so the effective parameter count and compute are similar to a non-gated MLP at 4× expansion.

### 2\. RMSNorm

Already covered in M8 and reimplemented as a custom kernel in M23. Here it is in plain PyTorch — what we'll actually use.
    
    
    class RMSNorm(nn.Module):
        def __init__(self, d, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))
            self.eps = eps
    
        def forward(self, x):
            # Cast to fp32 for the reduction (M14: numerical stability matters for norms)
            x_fp32 = x.float()
            rms = x_fp32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
            # Multiply (still fp32), then cast back to input dtype
            return (x_fp32 * rms).to(x.dtype) * self.weight

Three things from M14: cast to fp32 for the reduction (preserves precision when the input is bf16), the rescale happens in fp32, cast back to bf16 only on the output. The `self.weight` stays in bf16 the whole time — it's a multiplicative scaling that doesn't need fp32 precision. Same recipe as the production RMSNorm in `torch.nn.functional`.

### 3\. RoPE

Rotary positional encoding. M30 covers RoPE in depth — here we just need the apply function. The high-level idea: rotate every pair of dimensions in Q and K by an angle proportional to the position. The result: `(Qᵢ · Kⱼ)` depends only on `i − j`.
    
    
    def build_rope_cache(seq_len, head_dim, base=10000.0, device="cuda"):
        # Frequencies for each dim pair
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        # Outer product of positions and frequencies
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, inv_freq)              # [T, head_dim/2]
        cos = freqs.cos().repeat_interleave(2, dim=-1)  # [T, head_dim]
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        return cos, sin
    
    def apply_rope(x, cos, sin):
        # x: [B, n_heads, T, head_dim]. Apply rotation in fp32 for stability.
        x_fp32 = x.float()
        # Pair adjacent dims into (even, odd); rotate each pair.
        x1, x2 = x_fp32[..., ::2], x_fp32[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        out = (x_fp32 * cos + rotated * sin).to(x.dtype)
        return out

The rotation trick: pair dim 0 with dim 1, dim 2 with dim 3, etc. For each pair, treat them as (real, imaginary), and rotate by the position-dependent angle. The `repeat_interleave(2)` broadcasts the per-pair frequency to both members of the pair. The `stack + flatten` implements the imaginary-part swap (real → imaginary; imaginary → -real).

Why fp32 inside the rotation? `cos` and `sin` at large positions can produce phase-cancellation issues if computed in bf16. Standard recipe: do the math in fp32, cast back at the end.

### 4\. Attention

The biggest single component. Q/K/V projections, head reshape, RoPE, GQA expansion, FlashAttention, output projection.
    
    
    class Attention(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.q_proj  = nn.Linear(cfg.d_model, cfg.n_head    * cfg.head_dim, bias=False)
            self.k_proj  = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
            self.v_proj  = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
            self.o_proj  = nn.Linear(cfg.n_head    * cfg.head_dim, cfg.d_model, bias=False)
    
        def forward(self, x, cos, sin, kv_cache=None):
            B, T, _ = x.shape
            H, KH, D = self.cfg.n_head, self.cfg.n_kv_head, self.cfg.head_dim
    
            # 1. Project to Q, K, V — three matmuls.
            q = self.q_proj(x).view(B, T, H,  D).transpose(1, 2)   # [B, H,  T, D]
            k = self.k_proj(x).view(B, T, KH, D).transpose(1, 2)   # [B, KH, T, D]
            v = self.v_proj(x).view(B, T, KH, D).transpose(1, 2)   # [B, KH, T, D]
    
            # 2. Apply RoPE to Q and K (V is positionally invariant).
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
    
            # 3. KV cache: append new K/V to cached, retrieve full.
            if kv_cache is not None:
                k = torch.cat([kv_cache[0], k], dim=2)        # along T axis
                v = torch.cat([kv_cache[1], v], dim=2)
                kv_cache_out = (k, v)
            else:
                kv_cache_out = (k, v)
    
            # 4. GQA: expand K and V to match Q's head count via repeat.
            #    enable_gqa=True in SDPA does this implicitly; we keep it explicit for clarity.
            k = k.repeat_interleave(H // KH, dim=1)   # [B, H, T_full, D]
            v = v.repeat_interleave(H // KH, dim=1)
    
            # 5. The attention itself — FlashAttention via SDPA (M25).
            #    is_causal=True for training/prefill; False for single-token decode (causality automatic with KV cache).
            is_causal = (kv_cache is None)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    
            # 6. Reshape back, output projection.
            out = out.transpose(1, 2).contiguous().view(B, T, H * D)
            return self.o_proj(out), kv_cache_out

Five things to call out, all tracing back to specific modules:

  * **No bias in Linear** (`bias=False`): biases hurt slightly for transformers and add params for no win. Standard since GPT-J. Same for the output projection.
  * **K and V have fewer heads than Q** (GQA from M27). The K and V projections are physically smaller. KV cache shrinks proportionally — for GQA-3 here (12 query heads, 4 KV heads), cache is 3× smaller than full multi-head.
  * **RoPE applied to Q and K, not V**. V is positionally invariant — its values don't depend on where the token is. Common bug: applying RoPE to V too. Don't.
  * **The KV cache in step 3** is the inference-time path. `kv_cache` is `(K_cached, V_cached)` from previous decode steps; we concat new K/V to the cached ones. _This is exactly the buffer that vLLM (M27) pages_.
  * **`F.scaled_dot_product_attention` with `is_causal=True`**. PyTorch's high-level entry point picks FlashAttention as the backend (M25). The `is_causal=True` tells the kernel to skip K/V tiles above the diagonal — ~2× speedup vs explicit mask. _For decode (when kv_cache is provided), causality is automatic — Q is shape [B, H, 1, D], no mask needed._

### 5\. SwiGLU MLP

The modern gated MLP. `down(silu(gate(x)) * up(x))`. Three Linears instead of GELU's two, but the gate × up multiplicative interaction reliably outperforms simple GELU.
    
    
    class SwiGLU(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.gate_proj = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
            self.up_proj   = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
            self.down_proj = nn.Linear(cfg.d_mlp, cfg.d_model, bias=False)
    
        def forward(self, x):
            # Gated path: silu(gate) * up
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

Why SwiGLU over plain GELU MLP? Empirically: ~0.5-1% perplexity improvement at the same parameter count. The gate × up product introduces a multiplicative non-linearity that GELU can't replicate. Cost: one extra matmul (gate_proj). Worth it.

Why `silu` specifically? `silu(x) = x * sigmoid(x)` — also called `swish`. Smooth, non-monotonic, similar to GELU's shape. The "Swi" in SwiGLU is from "Swish."

### 6\. Transformer block

Pre-norm + attn + residual; pre-norm + mlp + residual.
    
    
    class Block(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.attn  = Attention(cfg)
            self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.mlp   = SwiGLU(cfg)
    
        def forward(self, x, cos, sin, kv_cache=None):
            # Pre-norm: norm BEFORE the sub-layer. Residual ADDS the unnormed x.
            attn_out, kv_cache_out = self.attn(self.norm1(x), cos, sin, kv_cache)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, kv_cache_out

The pre-norm structure is the simplest part of the model and also the most-debated detail historically. The original Transformer was post-norm (`norm(x + sublayer(x))`) and required learning-rate warmup to train stably. Pre-norm trains stably from step 0 because gradients flow unobstructed through the residual path. _If you remember one fact about transformer architecture, remember pre-norm._

### 7\. The full model

Embedding + N blocks + final norm + lm_head.
    
    
    class GPT(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
            self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
    
            # Tied embeddings: lm_head shares weight tensor with input embedding.
            if cfg.tied_embeds:
                self.lm_head.weight = self.embed.weight
    
            # Precompute RoPE cache once (M30).
            cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_base)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
    
            # Init scheme (M8). std=0.02 standard for GPT-style models.
            self.apply(self._init_weights)
    
        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
    
        def forward(self, token_ids, kv_caches=None, start_pos=0):
            B, T = token_ids.shape
            x = self.embed(token_ids)                # [B, T, D]
    
            # Slice RoPE cache for this segment.
            cos = self.cos[start_pos : start_pos + T]
            sin = self.sin[start_pos : start_pos + T]
    
            # Run through each block, threading the KV cache.
            new_kv_caches = []
            for i, block in enumerate(self.blocks):
                kv_in = kv_caches[i] if kv_caches is not None else None
                x, kv_out = block(x, cos, sin, kv_in)
                new_kv_caches.append(kv_out)
    
            x = self.final_norm(x)
            logits = self.lm_head(x)                  # [B, T, vocab_size]
    
            return logits, new_kv_caches

That's the whole model. ~80 lines for the full GPT class plus its components, ~200 lines counting Config and the helpers. It works.

### The init scheme matters more than you'd think

One detail on init: `std=0.02` across all linears is the GPT-style default. It's a reasonable starting point, but at depth, you typically want to _scale down_ the init for the residual-output projections (`o_proj` in attention, `down_proj` in MLP) by `1/sqrt(2 * n_layer)`. This keeps the residual stream's variance from blowing up as depth accumulates contributions from each block.
    
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=self.cfg.init_std)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, std=self.cfg.init_std)
    
    # After the apply, scale residual-output projections (M8 redux).
    for name, p in self.named_parameters():
        if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
            with torch.no_grad():
                p.mul_(1.0 / math.sqrt(2 * self.cfg.n_layer))

This is the _actually-correct_ GPT-2 init. Skipping it gives slightly worse loss curves at depth ≥ 24. Most modern codebases include it.

## The training loop

From M11, with bf16 mixed precision (M14):
    
    
    cfg = GPTConfig()
    model = GPT(cfg).cuda().to(torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                                  weight_decay=0.1, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)
    
    for step, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()
        logits, _ = model(x)                     # [B, T, vocab]
    
        # Cross-entropy in fp32 for stability (M14).
        loss = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten())
    
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
        if step % 100 == 0:
            # tokens/sec — the canonical training throughput metric.
            toks_per_sec = (x.numel() * 100) / (time.time() - last_t)
            print(f"step {step} loss {loss:.4f} tok/s {toks_per_sec:.0f}")
            last_t = time.time()

Standard pieces: AdamW with weight decay 0.1 and β₂=0.95 (the GPT-3 / Llama default — slightly more conservative than the AdamW default β₂=0.999), cosine schedule (M9), gradient clipping at 1.0 (M9), bf16 cast on the model. The loss is computed in fp32 (`logits.float()`) — same recipe as M14.

The `tokens/sec` log is the throughput metric you'll see most. For a GPT-2-small on an H100 in bf16, expect ~500K tokens/sec; for a 7B Llama on an H100, ~10K tokens/sec; with FSDP across 8 H100s on a 70B Llama, ~3K tokens/sec.

## KV cache wiring for inference

Generation uses a different path. The first call processes the prompt (prefill); subsequent calls pass the previous KV cache and one new token at a time (decode). Same model, two different access patterns — exactly what M27 described.

KV cache shape evolution: prefill writes T positions, decode appends 1 per step ① Prefill: process the prompt — one big forward over all T_prompt tokens model(prompt_ids, kv_caches=None) writes K, V for positions 0..T_prompt−1 → KV cache: [L, 2, B, KH, T_prompt, D] L = num_layers, 2 = (K, V), KH = n_kv_heads ② Decode: generate one token at a time, passing previous cache + 1 new token step 1: 1 new tok cache: T_p+1 step 2: 1 new tok cache: T_p+2 step 3: 1 new tok cache: T_p+3 … step k: 1 new tok cache: T_p+k Memory cost per layer per token (Llama-style 7B with GQA): 2 (K,V) × 4 (KH) × 128 (D) × 2 bytes (bf16) = 2 KB per layer per token × 32 layers = 64 KB/token → for 32K context: ~2 GB just for KV cache (one request) this is exactly the buffer vLLM (M27) splits into 16-token pages

The key shape detail: the KV cache for one request is `[L, 2, B, KH, T, D]` (per-layer × K-or-V × batch × kv-heads × time × head-dim). T grows as decode progresses. The _total_ cache size scales linearly with conversation length — exactly the inference pain point M27 addressed via paging.

The generation loop:
    
    
    @torch.no_grad()
    def generate(model, prompt_ids, max_new_tokens=100,
                 temperature=1.0, top_k=50):
        cfg = model.cfg
        B, T = prompt_ids.shape
        device = prompt_ids.device
    
        # 1. Prefill — process the prompt all at once.
        logits, kv_caches = model(prompt_ids)
        next_logits = logits[:, -1, :]      # [B, vocab] — only the last position matters
    
        generated = [prompt_ids]
        for i in range(max_new_tokens):
            # 2. Sample next token.
            next_id = sample(next_logits, temperature, top_k)
            generated.append(next_id)
    
            # 3. Decode step: pass last token + previous cache.
            start_pos = T + i
            logits, kv_caches = model(next_id, kv_caches=kv_caches, start_pos=start_pos)
            next_logits = logits[:, -1, :]
    
        return torch.cat(generated, dim=1)
    
    
    def sample(logits, temperature, top_k):
        # Apply temperature.
        logits = logits / max(temperature, 1e-5)
        # Top-k filter.
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, -1:]] = -float("inf")
        # Sample from the resulting categorical.
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

Three sampling knobs to expose:

  * **Temperature** : divides logits before softmax. `T=1.0` is the natural distribution; `T<1` sharpens (greedy at `T→0`); `T>1` flattens (more creative / chaotic).
  * **Top-k** : keep only the k highest-logit tokens, set the rest to -∞. Prevents the model from sampling absurd low-probability tokens.
  * **Top-p (nucleus)** (not shown here): keep the smallest set of tokens whose cumulative probability exceeds p. More flexible than top-k for varying-entropy distributions.

Production setups expose all three plus repetition penalties; vLLM's `SamplingParams` (M27) covers them. Same logic, packaged for serving.

## Debugging your first run: failures in order of likelihood

You will run this and something will go wrong. Here's the diagnostic checklist, in the order issues appear:

First-run failures and how to spot them Symptom| Likely cause| Fix (reference)  
---|---|---  
NaN in loss at step 1| fp16 instead of bf16, or accumulator dtype wrong| Use bf16 not fp16; cast loss computation to fp32 (M14)  
NaN appears after a few hundred steps| LR too high; init too aggressive| Lower LR to 1e-4; verify init scheme; clip grads (M8, M9)  
Loss decreases for the first ~100 steps then plateaus near random| Causal mask bug (model can see the future)| Verify `is_causal=True` in SDPA call; check generation produces sensible token at position 0 vs T (M25)  
Loss is exactly `log(vocab_size)` and never moves| Forward returning zeros; LM head weight tied wrong| Check `lm_head.weight is embed.weight` if tied; verify init isn't zero  
Generation produces gibberish but loss looks fine| Tokenizer mismatch (training vs inference); positional encoding bug| Verify same tokenizer round-trips; check RoPE applied (M30)  
OOM during prefill but training works| FlashAttention not picked up; falling back to math backend| Check head_dim ≤ 256 and is multiple of 8; bf16 not fp32; `sdpa_kernel` verify (M25)  
Slower than expected| Wrong dtype, no torch.compile, dataloader bottleneck| bf16 weights; `torch.compile(model)`; check num_workers and pin_memory (M11, M21)  
Generation is slower than training, per-token| Decode is memory-bound; expected!| Quantize weights (M26); paged KV (M27); add speculative decoding (M27)  
  
A practical reflex: _always sanity-check on a tiny model first_. Before running real training, build a 2-layer 64-dim model on 100 batches of random data. If loss decreases visibly within 100 steps, the architecture is wired correctly. If it doesn't, the bug is structural (wrong mask, wrong residual, wrong init), not a hyperparameter.

## Connecting back to the rest of the course

Read the model code one more time and notice: every meaningful decision is a callback to a specific module.

Where each architectural decision came from in the course Decision| Module  
---|---  
Vocab size 50304 (multiple of 64) for tensor cores| M22 — GPU programming model  
Head dim 64 (multiple of 8)| M22 — tensor core dim alignment  
RMSNorm with fp32 reduction| M14, M23 — mixed precision & the kernel  
RoPE applied in fp32| M14 — numerical stability  
Pre-norm, not post-norm| M8 — init & norms; depth stability  
Bias=False on Linears| Standard since GPT-J; saves params for no quality cost  
GQA (n_kv_head < n_head)| M27 — KV cache reduction  
SwiGLU MLP| Modern preference; gated path beats GELU  
Tied embeddings| Param savings; M11 mentions it  
Init std=0.02 with residual scaling 1/sqrt(2·n_layer)| M8 — init schemes  
F.scaled_dot_product_attention with is_causal=True| M25 — FlashAttention via the front door  
KV cache as `(K, V)` tuples per layer| M27 — the buffer vLLM pages  
AdamW with β₂=0.95, weight decay 0.1, fused=True| M9 — optimizers  
Cosine schedule + linear warmup| M9 — schedulers  
bf16 weights, fp32 loss| M14 — mixed precision  
Gradient clipping at 1.0| M9 — optimizer hygiene  
tokens/sec as throughput metric| M11 — training-loop telemetry  
  
Every line traces back. _This is what mastery looks like_ : not memorizing the architecture, but being able to derive each decision from a constraint or empirical finding you've internalized.

## Scaling up: what changes for a real model

The code above trains a GPT-2-small. To go to a real production model (Llama-3-8B, say), the architecture barely changes. What changes is the engineering around it:

  * **Wider and deeper** : `n_layer=32`, `d_model=4096`, `n_head=32`, `n_kv_head=8`, `d_mlp=14336`. Same code, bigger numbers.
  * **Distributed training** : FSDP (M17) for >1 GPU, eventually TP (M18) for >8 GPUs, eventually PP (M18) for multi-node. Code changes are at the wrapping layer, not in the model itself.
  * **Activation checkpointing** (M12): wrap each `Block` in `checkpoint`. Recomputes during backward to save memory.
  * **Compile** : `model = torch.compile(model)` (M21). For a 7B model, ~1.3-1.5× speedup.
  * **Long context** : increase `rope_base` from 10000 to 500000 for context extension to 32K+ (M30 covers this in detail). Possibly use sliding window or interleaved attention patterns for >128K.
  * **Real data pipeline** : M10's DistributedSampler, real tokenization (tiktoken or sentencepiece), shuffled multi-file iteration with proper sharding.

The model itself is unchanged. _Architecture is not where production complexity lives — it's in the engineering scaffolding._

#### Q&A; — About the build **Q:** Why is RoPE applied to Q and K but not V? **A:** Attention scores depend on Q · K^T. RoPE makes that dot product depend on (i − j), the relative position. V doesn't enter the score; it's just the value being weighted. Applying RoPE to V would rotate the values without any compensating un-rotation later — you'd just be scrambling the values. Common bug; the fix is to never apply RoPE to V. **Q:** Why is the lm_head a linear with no bias, and why tied to the embedding? **A:** No bias because biases on output projections don't help language models — empirically and theoretically (a uniform bias just adds the same scalar to all vocab logits, which softmax cancels out). Tied because the embedding (vocab → D) and lm_head (D → vocab) are both shape-(vocab, D) matrices doing inverse-ish operations. Sharing the weight saves `vocab × D` parameters — for a 7B model with vocab=128K and D=4096, that's 524M params (~7% of the model). Not all models tie (Llama-3 stopped for the largest sizes); for small models, always tie. **Q:** Why `is_causal=True` for prefill but `False` for decode? **A:** In prefill, Q has shape `[B, H, T_prompt, D]` — multiple positions attending to each other. The causal mask prevents position i from attending to position j>i. In decode, Q has shape `[B, H, 1, D]` — just one new token. There's only one query position; it attends to all cached K/V positions, all of which are at positions ≤ current. Causality is automatic; passing `is_causal=True` would (depending on the kernel version) try to apply a causal mask of shape (1, T_full) which is degenerate. _For the case`kv_cache is not None`, set is_causal=False._ **Q:** Why is `n_kv_head` 4 and not 1 (full MQA)? **A:** MQA (single KV head shared across all query heads) saves the most KV cache memory but sacrifices some quality. GQA with a small number of KV heads (4-8) is the sweet spot — most of the memory benefit, almost none of the quality cost. Llama-3-8B uses 8 KV heads with 32 query heads; ours is 4 with 12. The ratio matters more than the absolute number. **Q:** My loss looks fine but generation is gibberish. What's likely wrong? **A:** Three most common: (1) Tokenizer mismatch — training tokenized with one BPE, generation with a different one. Verify `tokenizer.decode(tokenizer.encode(text)) == text` for a sample. (2) RoPE positions not advancing during generation — the `start_pos` parameter must increment with each decode step. If you re-use position 0 for every token, attention is scrambled. (3) Causal mask issue at training, fine at generation — the model trained with future leakage, so it expects to see tokens after the current one. At generation, those tokens don't exist; outputs are garbage. Verify generation matches training time loss-on-full-prompt for the prefill portion. **Q:** Why bf16 weights instead of fp32? **A:** bf16 has fp32-comparable range (M14) and 2× less memory. For inference, the speed and memory wins are huge. For training, you store an fp32 master copy in the optimizer (M12) for accurate updates while the forward/backward use bf16. _If you're storing a checkpoint, use bf16_ ; the fp32 master is a training-time-only artifact. **Q:** When should I switch from this code to a library like `transformers`? **A:** For research where you control the architecture: the code above is great. For production where you want robust support for hundreds of model variants, tokenizers, generation strategies, and pre-trained checkpoints: use `transformers`. The code in M29 is exactly what's inside `transformers/models/llama/modeling_llama.py`, with more bells and whistles. _Knowing how it works is what lets you debug when the library has a bug or doesn't support what you need._

## Code Magnets: assemble the transformer block forward

You're writing the forward pass of a transformer block (pre-norm + attention + residual; pre-norm + MLP + residual). Three magnets are wrong choices.

Arrange the magnets into the correct forward.

def forward(self, x, cos, sin, kv_cache=None): attn_out, kv_cache_out = self.attn(self.norm1(x), cos, sin, kv_cache) attn_out, kv_cache_out = self.attn(x, cos, sin, kv_cache) x = x + attn_out x = self.norm1(x + attn_out) x = x + self.mlp(self.norm2(x)) x = self.norm2(x + self.mlp(x)) return x, kv_cache_out

show solution
    
    
    def forward(self, x, cos, sin, kv_cache=None):
        attn_out, kv_cache_out = self.attn(self.norm1(x), cos, sin, kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, kv_cache_out

The traps:

  * `attn_out, kv_cache_out = self.attn(x, cos, sin, kv_cache)`: passes the unnormed x to attention. _Pre-norm_ means norm BEFORE the sub-layer; this version is post-norm-ish, breaks training stability at depth.
  * `x = self.norm1(x + attn_out)`: this is the original Transformer's post-norm. Trains poorly without warmup; not what modern transformers use.
  * `x = self.norm2(x + self.mlp(x))`: same post-norm bug for the MLP sub-block. Note also that the MLP input here would be unnormed — double bug.

The pattern: **norm BEFORE the sub-layer's input; residual ADDS the sub-layer's output to the unnormed x**. Two sub-blocks (attn, mlp), two norms (norm1, norm2), two residuals.

## Who does what?

Match each transformer component to its real role.

Component

Real role

nn.Embedding

A. Lookup table mapping token IDs to D-dim vectors; (vocab_size, D) shape.

RMSNorm

B. Per-token rescaling by RMS magnitude; faster than LayerNorm at same quality.

RoPE

C. Position-dependent rotation of Q and K; makes attention depend on (i − j).

Causal mask (is_causal=True)

D. Tells SDPA to skip K/V tiles above the diagonal; ~2× attention speedup.

SwiGLU MLP

E. Gated MLP: down(silu(gate(x)) · up(x)); ~0.5-1% perplexity win over GELU.

KV cache

F. Cached K/V per layer per request; grows with conversation length, paged in serving.

Tied lm_head + embedding

G. Shared weight for input lookup and output projection; saves vocab×D parameters.

show solution

**nn.Embedding** → A  
**RMSNorm** → B  
**RoPE** → C  
**Causal mask** → D  
**SwiGLU MLP** → E  
**KV cache** → F  
**Tied lm_head + embedding** → G 

The mental shortcut: _embedding looks up, RMSNorm rescales, RoPE rotates, causal mask gates the future, SwiGLU gates the MLP, KV cache grows over time, tied embeddings save parameters_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team writes the transformer block as `x = self.norm(x + self.attn(x))`. Training is unstable past 12 layers; loss spikes that don't recover. What's wrong, and what's the fix?

show answer

That's post-norm: applying the norm to the residual sum. The original Transformer used this; modern transformers (GPT-2 onward) use pre-norm. The instability is the well-known post-norm-at-depth issue: gradients flow through the norm operator before reaching the sub-layer's input, accumulating norm-induced rescaling that interferes with the residual stream.

Fix: switch to pre-norm — `x = x + self.attn(self.norm(x))`. Now the norm only sees the path going into the sub-layer; the residual path is untouched. Trains stably from step 0. **Pre-norm is one of the most reliable architectural choices in modern deep learning.** If a paper proposes post-norm, it almost always also proposes the warmup or careful-init regime needed to make it work — the variants are not interchangeable.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why does the GPT class register `cos` and `sin` as buffers (not parameters), and why `persistent=False`?

show answer

They're _not parameters_ : they're not learned. They're computed once at init from `build_rope_cache`. Registering them as buffers (M7) means they're moved with the model (`.cuda()`, `.to(dtype)`) but not seen by the optimizer. Calling them parameters would either crash (the optimizer would try to update them and break the deterministic computation) or, with `requires_grad=False`, work but be confusing.

`persistent=False` means they're not included in `state_dict()` — when you save a checkpoint, cos and sin aren't saved. They're recomputed at model init from the config, so saving them is wasteful (they're tens of MB for long context) and brittle (re-using the checkpoint with a different max_seq_len or rope_base would silently use the old, wrong values). **Computed-from-config tensors should be persistent=False buffers.**

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** A team's GPT trains and the loss curve looks healthy, but generated text is repetitive — the model loops on phrases. What's likely going on, and what's the fix?

show answer

This is usually a generation-time issue, not a training issue. The most common cause: **greedy or low-temperature sampling on a model trained on diverse data**. The model's logit distribution has a clear winner at each step that ends up cycling — "the cat sat on the mat. The cat sat on the mat. ..." The model isn't broken; the sampling is too deterministic.

Fixes, in order of typical efficacy: (1) Increase temperature to 0.7-0.9 (creative tasks) or use top-p sampling with p=0.9. (2) Add a repetition penalty: scale down logits for tokens that appeared recently (typical penalty 1.1-1.3). (3) If repetition still appears, the issue might be undertraining or low-quality training data — but verify the sampling first.

Independent check: at training-time-equivalent temperature (whatever the loss was computed at), generation should be reasonable. If repetition appears at temperature=0.7+ and increases sharply at lower temperatures, sampling is the issue.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Sketch the difference in computation between prefill (one big forward over T prompt tokens) and decode (one forward per generated token). Why is decode slower per token?

show answer

Prefill: one forward processes [B, T_prompt, D] activations. Each matmul is shape `[B*T_prompt, K] × [K, M]` — large rectangular matmuls that saturate tensor cores. The whole prompt costs the price of one forward.

Decode: one forward processes [B, 1, D] for the new token, but reads _all_ of the cumulative KV cache during attention. Per-step matmuls (Q/K/V projections, MLP) are shape `[B*1, K] × [K, M]` — extremely tall-and-skinny. Attention is `[B, H, 1, D] × [B, H, T_full, D]` for the QK matmul.

Why decode is slower per token: the matmul _compute_ is tiny but the _weight loads_ are the same as in prefill (you still have to read every weight matrix from HBM). For decode, this means weight bandwidth dominates and tensor cores are idle — exactly the memory-bound regime from M22's roofline. The fix is everything in M27: quantization (smaller weights), continuous batching (more useful work per weight load), speculative decoding (multiple tokens per forward).

Concrete numbers for a 7B model on H100: prefill processes ~50K tokens/sec; decode produces ~70 tokens/sec at batch 1. Per-token, prefill is ~700× faster. _This 700× gap is the entire reason inference systems exist as their own engineering discipline_.

### What just happened?

  * A GPT-style transformer is **embedding → N pre-norm transformer blocks → final norm → lm_head**. Each block has attention + residual + MLP + residual.
  * **Pre-norm** (`x + sublayer(norm(x))`) is the modern default. Post-norm is the original; trains worse at depth.
  * **RMSNorm** instead of LayerNorm — drops mean subtraction, ~30% faster, same quality. **SwiGLU** instead of GELU MLP — gated path, ~0.5-1% perplexity win.
  * **GQA** (n_kv_head < n_head) shrinks the KV cache without quality loss. **Tied embeddings** save vocab×D params.
  * **RoPE** for positional encoding (M30 covers in depth). Applied to Q and K only; in fp32 for stability.
  * Attention uses **`F.scaled_dot_product_attention(..., is_causal=True)`** — picks FlashAttention (M25) when applicable. Causal masking is free at the kernel level.
  * Many small choices trace back to specific modules: vocab=50304 for tensor-core alignment (M22), bias=False for parameter efficiency, RMSNorm cast to fp32 for numerical stability (M14), residual init scaling 1/sqrt(2·n_layer) for depth stability (M8).
  * Training: **AdamW (β₂=0.95, weight_decay=0.1, fused), cosine schedule, gradient clipping at 1.0, bf16 weights with fp32 loss**.
  * The KV cache is `[L, 2, B, KH, T, D]` — per-layer (K, V) tuples. **Prefill writes T positions in one shot; decode appends 1 per step**. This is exactly what vLLM (M27) pages.
  * Generation: prefill processes the prompt; decode generates tokens one at a time, threading the KV cache. **Decode is ~700× slower per token than prefill** — entirely because of the memory-bound regime (M22, M27).
  * Sampling: temperature, top-k, top-p, repetition penalty. Deterministic at temperature → 0; more creative as temperature rises.
  * **Most first-run failures fall into a small set** : NaN (mixed-precision setup), causal mask bug (model cheats), tokenizer mismatch (gibberish despite good loss), wrong RoPE position threading at generation. Walk the table top-to-bottom.
  * **Architecture is not where production complexity lives.** The model itself is ~250 lines. The complexity is in distributed training (M16-M18), memory management (M12), serving (M27), kernels (M22-M25). The architecture is the easy part once you've internalized the pieces.
  * The reflex from this module: when you read a paper proposing an architecture change, ask "which module's content does this conflict with?" Most proposed changes either revisit a settled question (post-norm is back!) or introduce a real win in a specific regime. Knowing the components lets you tell which is which.

Module 30 takes RoPE — which we used here without much explanation — and unpacks it properly. The math behind why pair-wise rotation gives relative-position attention; the choices of θ-base; the recipes for context extension (NTK, YARN, Dynamic NTK); and how RoPE interacts with the attention kernel. After M30, M31 covers post-training (SFT, RLHF, DPO), M32 consolidates debugging skills, and M33 explores Mamba as a transformer alternative.
