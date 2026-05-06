# Module 27 — Inference systems: KV cache, paged attention, speculative decoding

# _Inference systems:_ KV cache, paged attention, speculative decoding

_Part VIII · Module 27_

— how the prefill/decode split, paged KV memory, continuous batching, and speculative drafting turn a slow autoregressive model into a high-throughput serving system

\--- 

For 26 modules we've focused on training and on writing fast kernels. This module is about _serving_ a trained model — taking an LLM and running it efficiently for users hitting an API. It turns out serving is fundamentally different from training: the access patterns, the bottlenecks, and the optimizations are all distinct. A model that trains beautifully on bf16 transformers may need to be quantized, KV-cached, paged, batched, and speculatively decoded to serve at acceptable cost.

The good news: every concept in M27 builds on tools you already have. KV cache management is just memory layout (M12). Paged attention is FlashAttention (M25) with a different memory layout. Speculative decoding is a clever use of compute parallelism (M22). Continuous batching uses the same async stream model from M20. Quantization (M26) makes it all fit. **You're reading a synthesis module, not a new layer.**

> **★ KEY IDEA**  
>  LLM inference has two distinct phases. **Prefill** processes the entire prompt at once — compute-bound, like training. **Decode** generates one token at a time, autoregressively — memory-bound, completely different kernel shapes. The **KV cache** stores per-layer keys and values for every previous token to avoid recomputing them. **Paged attention** manages KV cache as fixed-size pages so memory isn't wasted on max-length pre-allocation. **Continuous batching** processes many requests at varying stages together, joining new ones mid-flight. **Speculative decoding** uses a small drafter model to propose K tokens that the target model verifies in one forward — turning K decode steps into one. Combine these, add quantization, and a 70B model serves at 100+ tokens/sec on a single H100. 

## Two new faces

KV

KV Cache

"I'm the per-token state that grows with the conversation. I'm why long contexts are expensive."

Every token your model generates produces a key and a value vector for each attention layer. To compute the next token's attention, the model needs the keys and values from _all previous tokens_. So I store them. For a 70B model with 80 layers, 64 heads (head_dim 128), at bf16: I'm _400 KB per token_. For a 32K-context conversation: 12.8 GB just for me. I'm the dominant memory cost in inference, larger than the weights for long enough conversations. **Managing me efficiently is what separates a working system from one that runs out of memory at request 3.**

D

Drafter

"I'm a small fast model. I propose K candidate tokens; the big model verifies all K in one shot."

Decode is sequential — one token at a time, K forwards for K tokens. But the big model is memory-bound, so a forward on K tokens at once is barely slower than a forward on 1 token. I exploit that. I'm a 1B-parameter draft model that runs in < 1ms per token. I propose 5 tokens; the 70B target processes all 5 in one forward (~30ms instead of 5×30ms). For each draft token I got right, we got it for free. For each I got wrong, we discard everything after and continue from there. **Average acceptance rate ~70-80% → 3-5× throughput speedup.**

## The two phases of LLM inference

Standard pattern when an LLM serves a request:

  1. **Prefill** : encode the user's prompt. Process all N input tokens at once through the model. Produces the KV cache for those N tokens, plus the logits for the last token (used to sample the first output).
  2. **Decode** : generate output tokens one at a time. Each step processes 1 new token through all layers, producing a new K and V for each layer (added to the cache), and one new logit (used to sample the next token).

These two phases have _completely different_ performance characteristics:

Prefill vs decode: same model, completely different kernel shapes PREFILL — compute-bound "Process the prompt" Per layer matmul: (N=2048, K) × (K, N_out) → big matmul, tensor cores saturated Roofline (M22): high arithmetic intensity (2N FLOPs/byte) → approaches peak FLOPs ⚡ ~50-200ms for 2K tokens on a 7B model DECODE — memory-bound "One token at a time" Per layer matmul: (1, K) × (K, N_out) → tall-and-skinny, weights dominate I/O Roofline (M22): low arithmetic intensity (2 FLOPs/byte) → bandwidth-limited; tensor cores idle ⚡ ~10-30ms PER TOKEN on a 70B model prefill: weights loaded once, reused 2K times → use compute decode: weights loaded once, reused ONCE → bandwidth wall → different kernels, different optimizations, different products most user-facing latency comes from decode (200 tokens × 20ms = 4 sec)

This split is the most important fact about LLM inference. **Prefill cost is roughly proportional to prompt length** (compute scales with N, but compute is fast); **decode cost is roughly proportional to output length** (one autoregressive step per token, each at memory-bandwidth speed).

For a typical chat: 200-token prompt, 200-token response. Prefill: ~50ms. Decode: 200 × 20ms = 4 seconds. **Decode dominates user latency by 50×** in this case. Most inference optimizations target decode for this reason — every millisecond per token saves seconds per response.

## The KV cache

Why does decode work the way it does? Because attention re-uses every previous token's K and V. To compute attention at step t, you need K and V from steps 0 through t-1. Recomputing them every step would be quadratic in t. So you cache.

The cache size per layer per token: `2 (K + V) × num_kv_heads × head_dim × bytes_per_element`. For a Llama-3-70B-style model:
    
    
    num_layers       = 80
    num_kv_heads     = 8        # GQA — much fewer than num_heads (which is 64)
    head_dim         = 128
    bytes_per_elem   = 2        # bf16
    
    bytes_per_token  = 80 * 2 * 8 * 128 * 2 = 327,680 bytes ≈ 320 KB
    bytes_for_32K    = 32768 × 320 KB     ≈ 10.2 GB

That's per _request_. Serving 16 concurrent 32K conversations: 163 GB of KV cache. Larger than the model itself.

**Three levers to shrink the KV cache:**

  1. **Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)**. Standard multi-head attention has one K and V per attention head — 64 each for our model. MQA shares one K/V across all heads (1 each). GQA is in between (e.g., 8 K/V groups for 64 heads). Llama 3 uses GQA-8: 8× smaller cache than full multi-head, with minimal quality cost.
  2. **Lower-precision cache**. Store K and V in fp8 or int4 instead of bf16. Halves or quarters the memory. Quality cost is small for inference (the attention computation is robust to this).
  3. **Sliding-window attention** (Mistral, Gemma). Only attend to the last W tokens; older keys can be evicted. KV cache size is bounded at W·320 KB regardless of conversation length.

GQA is the dominant choice in 2024-2026 frontier models. It's why Llama 3's KV cache is "only" 320 KB/token instead of 2.5 MB.

## The naïve serving problem: pre-allocation waste

Suppose you want to serve 10 concurrent users, each with a max output length of 4096 tokens. Naïve approach: allocate 10 × 4096 × 320 KB = 12.5 GB of KV cache space upfront, in 10 contiguous regions.

The problem: _users finish at different lengths_. User A finishes at 200 tokens; their pre-allocated 4096-slot region is now ~95% empty until A's request is gone. User B is at 3500 tokens; their region is 85% full. _The empty parts of A's region can't help B_ because the regions are pre-assigned and contiguous. Memory utilization is terrible.

Worse: if some user actually generates 4097 tokens, you crash. So you over-provision and waste even more.

This is the same fragmentation problem as the caching allocator (M20), one level up. The fix is the same: **break the cache into fixed-size pages and allocate them on demand from a shared pool.**

## Paged attention

The vLLM team's contribution: borrow virtual memory's idea. The KV cache is divided into fixed-size **pages** — typically 16 tokens per page per layer. Each request has a _block table_ that maps its logical token positions (0..N-1) to the physical pages where its KV vectors actually live. Pages don't need to be contiguous; the block table handles indirection.

Paged attention: block tables map logical tokens to physical pages Logical sequences (what the model sees): Req A (40 toks): tok 0-15 tok 16-31 tok 32-39 Req B (24 toks): tok 0-15 tok 16-23 Block tables (per request): Req A: [ page_3, page_7, page_2 ] Req B: [ page_5, page_9 ] → pointers to physical pages Physical page pool (shared GPU memory): page 0 (free) page 1 (free) page 2 A[32-39] page 3 A[0-15] page 4 (free) page 5 B[0-15] page 6 (free) page 7 A[16-31] page 8 (free) page 9 B[16-23] when req A finishes: pages 2, 3, 7 freed → next request grabs them no fragmentation — pages are uniform, allocate one when token 16 of a new req fills its current page ~3-5× higher concurrent users on the same GPU vs naïve pre-allocation

The attention kernel becomes _paged FlashAttention_ : instead of contiguous K and V tensors, it takes the block table as input and indirects through it. Each Q tile in the kernel iterates over the K/V pages listed in the block table, in order. The kernel is structurally identical to FlashAttention (M25); only the memory access pattern changes.

The win:

  * **No pre-allocation waste** : requests grow page-by-page (one new page every 16 tokens). Short requests use few pages.
  * **No fragmentation** : pages are uniform-sized, slot in anywhere.
  * **Prefix sharing** : two requests with the same prompt prefix can share pages for the prefix portion (copy-on-write when they diverge). Saves memory for batched evaluations of the same prompt with different sampling parameters.

This is what **vLLM** and similar inference servers do. The page size is 16 in vLLM's default; some implementations use other sizes. Production systems that don't use paging waste 50-90% of KV memory.

## Continuous batching

The other major serving optimization. Without batching, each request goes through the model alone — bad GPU utilization since the model is memory-bound on weights, and one request barely uses any compute.

With _static batching_ , you wait for B requests to arrive, prefill all B together, then decode all B together. Problem: requests finish at different lengths. Once the first request is done, you have B-1 requests still active in a "batch" that no longer fully reflects the GPU shape. Batch size silently shrinks; GPU utilization drops.

With **continuous batching** (vLLM, TGI, etc.), the batch is dynamic. Each forward pass through the model contains _whichever requests are currently active_. When a request finishes, it's dropped. When a new request arrives, it's added (its prefill happens as a single forward step or chunked over a few steps; then it joins the decode batch).

The implementation requires the inference engine to handle:

  1. **Mixed prefill+decode batches** : in one forward, some sequences are doing prefill (long N) while others are doing decode (N=1). The kernel must handle both — typically by treating each sequence's tokens as variable-length, with paged KV.
  2. **Per-sequence state** : each sequence has its own KV cache pages, sampling parameters, stop conditions, generation counter.
  3. **Scheduler** : decides each step which requests are in this batch, when to add new ones, how to handle preemption when memory is full.

Result: the GPU stays at high utilization throughout, processing as many concurrent requests as KV cache memory allows. **Throughput is 5-10× higher than static batching** for typical chat workloads.

## Speculative decoding

The decode phase is one token per forward. Each forward is memory-bound — most time is spent loading weights from HBM, not computing. _What if a forward could produce multiple tokens?_

Speculative decoding's insight: the cost of a forward on K tokens is barely more than on 1 token (memory loads are the bottleneck and scale ~linearly only with K when K is small). So if you could _guess_ the next K tokens cheaply and then verify all of them with the big model in one forward, you'd be doing K decode steps in the time of one — when the guess is right.

The recipe:

  1. A small **drafter** model (typically 1-7B params if the target is 70B) generates K candidate tokens autoregressively. Cost: ~K × 1ms at the drafter's speed.
  2. The big **target** model processes all K tokens in one forward, producing K logits. Cost: ~30ms (about the same as one decode step since memory still dominates).
  3. For each draft token, compare the target's logit distribution with the drafter's. Use rejection sampling: accept the draft token with some probability based on the ratio; on rejection, sample from a corrected distribution. After the first reject, throw away all subsequent drafts for this round.

Speculative decoding: drafter proposes K, target verifies all K in one forward ① Drafter (1B model) generates K=5 candidate tokens autoregressively: "The" "cat" "sat" "on" "the" ~5 × 1ms = 5ms ② Target (70B model) verifies all 5 in ONE forward — same cost as 1 decode: target forward over [draft_1..draft_5] → 5 logits ~30ms (vs 5×30=150ms) ③ Acceptance: target's distribution accepts/rejects each draft via rejection sampling ✓ "The" ✓ "cat" ✓ "sat" ✗ "ON" discarded 3 accepted + 1 corrected = 4 tokens 4 tokens for the price of one decode → ~3-5× speedup typical key requirement: rejection sampling preserves the target's exact output distribution

Critical detail: **the rejection sampling is mathematically exact**. The output distribution is identical to what the target model alone would produce. You're not approximating — you're computing the same answer faster. This is why "speculative decoding" doesn't degrade quality.

The acceptance rate depends on how well-aligned the drafter is with the target. For aligned models (e.g., a 7B Llama drafting for 70B Llama), acceptance is typically 60-80% per token. With K=5 and 75% acceptance: average ~3.75 accepted per round, 3-5× speedup. _Higher_ acceptance with techniques like Medusa (multiple draft heads on the target) or EAGLE (a learned drafter trained jointly).

Variants and improvements:

  * **Medusa** : instead of a separate drafter, train K small heads on the target model that each predict K tokens ahead. The drafts come from the same model (free), and the verification step is inherent. Used in some commercial APIs.
  * **Lookahead decoding** : use n-gram statistics from earlier in the conversation as drafts. No external model needed.
  * **EAGLE** : train a small lightweight drafter that uses the target's hidden states. Higher acceptance than a generic small model.

## Putting it together: a real serving system

A modern LLM serving system (vLLM, TensorRT-LLM, SGLang, lmdeploy) combines:

  1. **Quantized weights** (M26): int4 or fp8, 2-4× smaller and faster on memory-bound decode.
  2. **Paged KV cache** : 16-token pages, block tables, no fragmentation.
  3. **Paged FlashAttention** : the kernel handles paged K and V via block tables.
  4. **Continuous batching** : dynamic batch composition, mixed prefill+decode forwards.
  5. **Speculative decoding** (optional): drafter + target with rejection sampling.
  6. **Tensor parallel** (M18) for big models: split across 4-8 GPUs within a node.
  7. **Prefix caching** : reuse pages for repeated prefixes (e.g., system prompts shared across requests).

End-to-end on a 70B model with 8×H100s, with all of these on: ~50-100 tokens/sec per request at batch sizes of 100+ concurrent users. Without these: maybe 5-10 tokens/sec at batch 8. **The gap between "naive serving" and "production serving" is roughly 10×.** Every layer adds another 1.3-3×.

## Latency vs throughput: the tradeoff

Two SLOs that pull in opposite directions:

  * **Time to first token (TTFT)** : the prefill latency. Lower batch size + tensor parallel helps. Important for interactive UX.
  * **Tokens per second (TPS) per request** : the decode rate. Lower batch size helps too — your request gets a bigger share of GPU time.
  * **Total throughput** : tokens/sec across all requests. _Higher_ batch size helps. Important for cost.

Increasing the batch size reduces per-request throughput (each request gets less compute per step) but increases overall throughput (more requests served per unit time). Different products pick different points: ChatGPT-style interactive chat optimizes per-request TTFT and TPS; batch-API endpoints optimize total throughput. vLLM exposes `max_num_batched_tokens` and similar knobs to tune the tradeoff.

## Using a modern serving system

You typically don't write inference servers from scratch. The standard tools:
    
    
    # vLLM — most popular open-source server
    from vllm import LLM, SamplingParams
    
    llm = LLM(
        model="meta-llama/Llama-3.1-70B-Instruct",
        quantization="awq",                  # auto-detect AWQ checkpoint
        tensor_parallel_size=4,             # 4 GPUs
        max_model_len=32768,
        enable_prefix_caching=True,
        speculative_config={
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "num_speculative_tokens": 5,
        },
    )
    
    prompts = ["Hello, how are you?", "What's the capital of France?"]
    sampling = SamplingParams(temperature=0.7, max_tokens=512)
    outputs = llm.generate(prompts, sampling)

That single setup turns on: int4 quantization (AWQ), tensor parallelism across 4 GPUs, paged KV cache, continuous batching, prefix caching, and speculative decoding with an 8B drafter for the 70B target. Everything we discussed in M27, configured in 8 lines.

For HTTP serving, vLLM offers an OpenAI-compatible API server: `vllm serve meta-llama/Llama-3.1-70B-Instruct ...`. TGI (HuggingFace's Text Generation Inference), TensorRT-LLM (NVIDIA's), and SGLang are similar.

#### Q&A; — About inference serving **Q:** Why is decode memory-bound but prefill compute-bound, when they're the same model? **A:** Matmul shape. Prefill: `(N, K) × (K, M)` where N is the prompt length (e.g., 2048). FLOPs = 2NKM, bytes ≈ KM (weights, loaded once for all N tokens). Arithmetic intensity ≈ 2N — high, compute-bound. Decode: `(1, K) × (K, M)`. FLOPs = 2KM, bytes ≈ KM. Intensity ≈ 2 — low, memory-bound. The N in the numerator of intensity is what makes prefill compute-bound. Same model, same weights, totally different bottlenecks based on how many tokens you're processing per forward. **Q:** Can speculative decoding hurt quality? **A:** Mathematically no — the rejection sampling preserves the exact target distribution. Empirically, you might see slightly different outputs for the same prompt+seed compared to standard decoding, because rejection sampling consumes more random numbers. But in expectation, the two distributions are identical. _This is why speculative decoding is a "free lunch" optimization_ — quality is preserved exactly, only the wall clock time changes. **Q:** Why use a 1B drafter instead of just running the 70B model normally? **A:** Because most decode time on the 70B is spent waiting for weights to load from HBM (memory-bound). The drafter's KV cache + weights fit easily in cache; the drafter runs ~30× faster than the target. Generating 5 tokens with the drafter takes ~5ms; verifying all 5 with the target takes ~30ms (one forward, weights loaded once). Total: 35ms for ~4 accepted tokens vs 5×30 = 150ms for 5 tokens normally. Speedup ≈ 4×. _The mismatch between drafter speed and target speed is what creates the opportunity_. **Q:** What's "prefix caching"? **A:** When many requests share a prefix — e.g., the same system prompt across all chat sessions, or in batched evaluations of one prompt with different sampling — paged KV cache lets you store the KV vectors for that prefix once and reference them from many requests' block tables. Saves prefill cost and KV memory. vLLM enables this with `enable_prefix_caching=True`. For high-throughput chat with shared system prompts, can save 30-70% of prefill compute. **Q:** Why doesn't sliding-window attention help inference more than it does? **A:** It does help — Mistral's 4K window cap means the KV cache for any conversation tops out at 4K tokens regardless of length. The KV cache is bounded. But: long-context tasks (RAG, long documents) need to attend beyond the window. So sliding-window is great for chat-like applications and a constraint for document-processing applications. Modern models often combine: full attention in some layers, sliding window in others ("hybrid attention"). Gemma 2 and others use this. **Q:** How does speculative decoding interact with continuous batching? **A:** Carefully. Each request can have its own number of draft tokens accepted per step, so different requests advance by different amounts. The scheduler tracks per-request token counts; the batched forward processes K candidates per request, but each request's KV cache and stop-condition logic is per-request. Adds engineering complexity, which is part of why production servers (vLLM, TensorRT-LLM) handle it for you. 

## Code Magnets: configure a high-throughput vLLM server

You're configuring a vLLM server for a 70B chat model with low TTFT, high concurrent throughput, and a smaller drafter for speculative decoding. Three magnets are wrong choices.

Arrange the magnets into a working server config.

from vllm import LLM, SamplingParams llm = LLM( model="meta-llama/Llama-3.1-70B-Instruct", quantization="awq", quantization="fp16", tensor_parallel_size=4, tensor_parallel_size=1, max_model_len=32768, enable_prefix_caching=True, speculative_config={ "model": "meta-llama/Llama-3.1-8B-Instruct", "num_speculative_tokens": 5, }, enable_chunked_prefill=False, )

show solution
    
    
    from vllm import LLM, SamplingParams
    llm = LLM(
        model="meta-llama/Llama-3.1-70B-Instruct",
        quantization="awq",
        tensor_parallel_size=4,
        max_model_len=32768,
        enable_prefix_caching=True,
        speculative_config={
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "num_speculative_tokens": 5,
        },
    )

The traps:

  * `quantization="fp16"`: not actually quantization — just the dtype. For a 70B model, you want `"awq"` or `"gptq"` or `"fp8"` to get the memory and speed benefits of low-bit weights.
  * `tensor_parallel_size=1`: a 70B model in int4 needs ~35 GB; in bf16, ~140 GB. With AWQ-int4 it might fit on a single H100, but you'd want TP=4 anyway for throughput. TP=1 leaves performance on the table.
  * `enable_chunked_prefill=False`: chunked prefill is what splits long-prompt prefills across multiple steps so they can interleave with active decodes (helps continuous batching). Disabling it hurts throughput when there's a mix of short and long prompts in flight.

The full pattern: **quantize the weights (AWQ-int4), tensor-parallelize across 4 GPUs, enable prefix caching, configure speculative decoding with an aligned 8B drafter**. This is roughly the standard production config for serving a 70B chat model on an 8-GPU node.

## Who does what?

Match each inference concept to its real role.

Concept

Real role

Prefill

A. Compute-bound phase that processes the prompt in one big matmul per layer.

Decode

B. Memory-bound phase that generates one token per autoregressive step.

KV cache

C. Per-token K and V vectors saved across decode steps so attention isn't recomputed.

Paged attention

D. KV cache split into uniform pages; block table maps logical tokens → physical pages.

Continuous batching

E. Dynamic batch where requests join and leave each step; mixed prefill+decode forwards.

Speculative decoding

F. Small drafter proposes K tokens, large target verifies all K in one forward.

GQA / MQA

G. Share K and V across multiple attention heads → smaller KV cache, minimal quality cost.

show solution

**Prefill** → A  
**Decode** → B  
**KV cache** → C  
**Paged attention** → D  
**Continuous batching** → E  
**Speculative decoding** → F  
**GQA / MQA** → G 

The mental shortcut: _prefill = compute-bound, decode = memory-bound, KV cache = the per-token state, paged = no fragmentation via block tables, continuous batch = dynamic batch composition, spec decoding = drafter+target with rejection sampling, GQA = shared KV heads_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team's chat app has p95 TTFT of 200ms but average per-token decode time of 100ms. Their users complain the responses feel slow. Which lever should they pull first, and why?

show answer

The decode rate. With 100ms per token, generating a 200-token response takes 20 seconds. TTFT of 200ms is fine — users tolerate up to ~500ms of "thinking" before the first token streams. But streaming at 10 tokens/sec feels glacial; users want at least 30 tokens/sec to feel responsive (faster than reading).

Levers in order of impact: (1) **Quantize weights to int4 with AWQ** — typically 2-3× decode speedup since decode is memory-bound on weight loads. (2) **Add speculative decoding** with a small drafter — another 2-4× on top. (3) **Add tensor parallelism** if not present — splits weight bandwidth across GPUs. After all three: 100ms → ~10ms/token = 100 tok/s, comfortably above the responsiveness threshold. _Don't waste effort on TTFT; the bottleneck is decode rate_.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why does paged attention work without performance penalty despite the indirect block-table lookup on every memory access?

show answer

Three reasons. (1) **The block table is tiny** : ~one int per logical page (16 tokens). For a 32K-token request that's 2K integers = 8 KB. Easily fits in L2 cache, often in L1. The indirect lookup is an L2-cache hit. (2) **Pages are contiguous internally** : once you've followed the indirection, you're reading 16 tokens × 320 KB worth of contiguous K and V — coalesced (M22) and exactly what the kernel wants. (3) **The kernel structure absorbs the indirection** : paged FlashAttention does one block-table lookup per K/V tile (every BLOCK_N tokens), not per element. The bookkeeping cost is in the noise. _The mechanical "indirection is slow" intuition from CPU programming doesn't apply when the indirection table fits in fast cache_.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** A team adopts speculative decoding with a 1B drafter and a 70B target. They measure 60% acceptance rate, but only see a 1.5× speedup instead of the expected 3-4×. What's likely going on?

show answer

Two common culprits. (1) **The drafter isn't aligned with the target**. Acceptance rate of 60% is low — typically 70-80% for well-aligned drafters (same model family, same finetuning recipe, similar training data). Try a drafter that's trained from the target as a teacher (distillation) or a Medusa-style drafter that uses the target's hidden states. (2) **The drafter is too slow per token**. If drafting K tokens takes K × 5ms instead of K × 1ms, the overhead eats into the win. Drafter should run at ~30× the target's per-token speed for good amortization. Profile drafter latency — if it's not in the <1ms range, switch to a smaller drafter or one with better quantization. _Speculative decoding's speedup is sensitive to both metrics_.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Why does increasing batch size improve total throughput but hurt per-request throughput?

show answer

Each forward through the model has roughly fixed cost (memory bandwidth limit on weight loads). At batch 1: one forward processes 1 token. At batch 64: one forward processes 64 tokens (one per active request, in continuous batching). Batch 64 takes maybe 1.5× as long as batch 1 (extra activation work, slightly more memory pressure), but processes 64× the tokens. _Per-step throughput_ goes up ~40×.

But each _individual_ request waited for the same forward to complete — it got 1/64 of the batch's tokens, not all of them. Per-request throughput goes _down_ : each request advances 1 token per step, where each step now takes 1.5× as long. So **per-request tokens/sec ≈ 1/(1.5 × step_time_at_batch_1) ≈ 67% of batch-1 speed**.

The fundamental tradeoff: you can serve more users at lower per-user speed, or fewer users at higher per-user speed. Different products pick different points; vLLM exposes the knobs to tune.

### What just happened?

  * **LLM inference has two phases** with completely different performance characteristics. **Prefill** is compute-bound (long matmuls, high arithmetic intensity). **Decode** is memory-bound (one-token-at-a-time, weight bandwidth dominates).
  * **Decode dominates user latency** : generating a 200-token response is 200 × per-token decode latency, often 50× the prefill cost.
  * **The KV cache** stores per-token K and V vectors to avoid recomputing attention. Size scales with conversation length × num_layers × num_kv_heads × head_dim. For long-context conversations it can exceed the model's parameter size.
  * **GQA / MQA** share K and V across attention heads — much smaller cache, minimal quality cost. Llama 3 uses GQA-8 (8× cache reduction).
  * **Naive serving wastes memory** by pre-allocating max-length KV per request. Fragmentation and over-provisioning eat 50-90% of available memory.
  * **Paged attention** (vLLM): KV cache split into 16-token pages; per-request block tables map logical tokens to physical pages. Pages allocated on demand from a shared pool. No fragmentation. ~3-5× more concurrent users on the same GPU.
  * **Continuous batching** : dynamic batch where requests join and leave each step. Mixed prefill+decode batches keep GPU utilization high. ~5-10× higher throughput than static batching.
  * **Speculative decoding** : small drafter proposes K tokens; target verifies all K in one forward via rejection sampling. Mathematically exact (preserves target distribution). ~3-5× decode speedup with 70-80% acceptance rate.
  * **Variants** : Medusa (heads on the target), EAGLE (learned drafter using target hidden states), lookahead (n-gram drafting). All preserve quality exactly.
  * **The full production stack** : quantized weights (M26) + paged KV + paged FlashAttention (M25) + continuous batching + speculative decoding + tensor parallelism (M18) + prefix caching. Combined: ~10× over naive serving.
  * **Latency vs throughput tradeoff** : smaller batch → faster per-request, lower total throughput. Larger batch → slower per-request, higher total throughput. Different products tune differently.
  * **Use existing systems** : vLLM, TensorRT-LLM, TGI, SGLang. Don't build inference servers from scratch — these handle the engineering.
  * The reflex: when serving an LLM, ask "is decode the bottleneck?" (almost always yes). Then quantize + spec decode + continuous batch in that order of priority.

Module 28 is the last module — **Mixture of Experts and a frontier capstone**. We'll cover MoE architectures (top-K routing, the all-to-all collective from M15 in action, expert sharding), why frontier models like Mixtral and DeepSeek use them, and the system-level engineering that goes into training and serving them. Then a capstone exercise that ties together everything from M1 to M28 — a single training/inference scenario where you need to reason about all of it.
