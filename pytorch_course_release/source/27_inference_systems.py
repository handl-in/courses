#!/usr/bin/env python3
"""Module 27: Inference systems — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part VIII · Module 27</div>
  <h1 class="module-title"><em>Inference systems:</em> KV cache, paged attention, speculative decoding</h1>
  <p class="module-sub">— how the prefill/decode split, paged KV memory, continuous batching, and speculative drafting turn a slow autoregressive model into a high-throughput serving system</p>
</div>

<p>For 26 modules we've focused on training and on writing fast kernels. This module is about <em>serving</em> a trained model — taking an LLM and running it efficiently for users hitting an API. It turns out serving is fundamentally different from training: the access patterns, the bottlenecks, and the optimizations are all distinct. A model that trains beautifully on bf16 transformers may need to be quantized, KV-cached, paged, batched, and speculatively decoded to serve at acceptable cost.</p>

<p>The good news: every concept in M27 builds on tools you already have. KV cache management is just memory layout (M12). Paged attention is FlashAttention (M25) with a different memory layout. Speculative decoding is a clever use of compute parallelism (M22). Continuous batching uses the same async stream model from M20. Quantization (M26) makes it all fit. <strong>You're reading a synthesis module, not a new layer.</strong></p>

<div class="keyidea">
LLM inference has two distinct phases. <strong>Prefill</strong> processes the entire prompt at once — compute-bound, like training. <strong>Decode</strong> generates one token at a time, autoregressively — memory-bound, completely different kernel shapes. The <strong>KV cache</strong> stores per-layer keys and values for every previous token to avoid recomputing them. <strong>Paged attention</strong> manages KV cache as fixed-size pages so memory isn't wasted on max-length pre-allocation. <strong>Continuous batching</strong> processes many requests at varying stages together, joining new ones mid-flight. <strong>Speculative decoding</strong> uses a small drafter model to propose K tokens that the target model verifies in one forward — turning K decode steps into one. Combine these, add quantization, and a 70B model serves at 100+ tokens/sec on a single H100.
</div>

<h2>Two new faces</h2>

<div class="character" style="--c: #b85a6c;">
  <div class="avatar" style="background: #b85a6c; color: #fff;">KV</div>
  <div>
    <p class="who">KV Cache</p>
    <p class="name">"I'm the per-token state that grows with the conversation. I'm why long contexts are expensive."</p>
    <p class="says">Every token your model generates produces a key and a value vector for each attention layer. To compute the next token's attention, the model needs the keys and values from <em>all previous tokens</em>. So I store them. For a 70B model with 80 layers, 64 heads (head_dim 128), at bf16: I'm <em>400 KB per token</em>. For a 32K-context conversation: 12.8 GB just for me. I'm the dominant memory cost in inference, larger than the weights for long enough conversations. <strong>Managing me efficiently is what separates a working system from one that runs out of memory at request 3.</strong></p>
  </div>
</div>

<div class="character" style="--c: #d4a017;">
  <div class="avatar" style="background: #d4a017; color: #fff;">D</div>
  <div>
    <p class="who">Drafter</p>
    <p class="name">"I'm a small fast model. I propose K candidate tokens; the big model verifies all K in one shot."</p>
    <p class="says">Decode is sequential — one token at a time, K forwards for K tokens. But the big model is memory-bound, so a forward on K tokens at once is barely slower than a forward on 1 token. I exploit that. I'm a 1B-parameter draft model that runs in &lt; 1ms per token. I propose 5 tokens; the 70B target processes all 5 in one forward (~30ms instead of 5×30ms). For each draft token I got right, we got it for free. For each I got wrong, we discard everything after and continue from there. <strong>Average acceptance rate ~70-80% → 3-5× throughput speedup.</strong></p>
  </div>
</div>

<h2>The two phases of LLM inference</h2>

<p>Standard pattern when an LLM serves a request:</p>

<ol>
  <li><strong>Prefill</strong>: encode the user's prompt. Process all N input tokens at once through the model. Produces the KV cache for those N tokens, plus the logits for the last token (used to sample the first output).</li>
  <li><strong>Decode</strong>: generate output tokens one at a time. Each step processes 1 new token through all layers, producing a new K and V for each layer (added to the cache), and one new logit (used to sample the next token).</li>
</ol>

<p>These two phases have <em>completely different</em> performance characteristics:</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 350" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Prefill vs decode: same model, completely different kernel shapes</text>

  <!-- Prefill panel -->
  <g transform="translate(20, 50)">
    <rect x="0" y="0" width="340" height="170" fill="#fff8a8" stroke="#1a1612" stroke-width="2"/>
    <text x="170" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">PREFILL — compute-bound</text>
    <text x="170" y="40" text-anchor="middle" font-size="10" fill="#1a1612">"Process the prompt"</text>

    <text x="20" y="65" font-size="10" font-weight="700" fill="#1a1612">Per layer matmul:</text>
    <text x="20" y="80" font-size="10" fill="#1a1612">  (N=2048, K) × (K, N_out)</text>
    <text x="20" y="95" font-size="10" fill="#1a1612">  → big matmul, tensor cores saturated</text>

    <text x="20" y="115" font-size="10" font-weight="700" fill="#1a1612">Roofline (M22):</text>
    <text x="20" y="130" font-size="10" fill="#1a1612">  high arithmetic intensity (2N FLOPs/byte)</text>
    <text x="20" y="143" font-size="10" fill="#1a1612">  → approaches peak FLOPs</text>

    <text x="20" y="160" font-size="10" fill="#c1502e">  ⚡ ~50-200ms for 2K tokens on a 7B model</text>
  </g>

  <!-- Decode panel -->
  <g transform="translate(380, 50)">
    <rect x="0" y="0" width="340" height="170" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="170" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">DECODE — memory-bound</text>
    <text x="170" y="40" text-anchor="middle" font-size="10" fill="#1a1612">"One token at a time"</text>

    <text x="20" y="65" font-size="10" font-weight="700" fill="#1a1612">Per layer matmul:</text>
    <text x="20" y="80" font-size="10" fill="#1a1612">  (1, K) × (K, N_out)</text>
    <text x="20" y="95" font-size="10" fill="#1a1612">  → tall-and-skinny, weights dominate I/O</text>

    <text x="20" y="115" font-size="10" font-weight="700" fill="#1a1612">Roofline (M22):</text>
    <text x="20" y="130" font-size="10" fill="#1a1612">  low arithmetic intensity (2 FLOPs/byte)</text>
    <text x="20" y="143" font-size="10" fill="#1a1612">  → bandwidth-limited; tensor cores idle</text>

    <text x="20" y="160" font-size="10" fill="#c1502e">  ⚡ ~10-30ms PER TOKEN on a 70B model</text>
  </g>

  <!-- Bottom annotations -->
  <text x="370" y="245" font-family="'Caveat', cursive" font-size="20" fill="#1f5f5b" text-anchor="middle">prefill: weights loaded once, reused 2K times → use compute</text>
  <text x="370" y="270" font-family="'Caveat', cursive" font-size="20" fill="#c1502e" text-anchor="middle">decode: weights loaded once, reused ONCE → bandwidth wall</text>
  <text x="370" y="298" font-family="'Caveat', cursive" font-size="20" fill="#1a1612" text-anchor="middle">→ different kernels, different optimizations, different products</text>
  <text x="370" y="328" font-family="'Caveat', cursive" font-size="22" fill="#c1502e" text-anchor="middle">most user-facing latency comes from decode (200 tokens × 20ms = 4 sec)</text>
</svg>
</div>

<p>This split is the most important fact about LLM inference. <strong>Prefill cost is roughly proportional to prompt length</strong> (compute scales with N, but compute is fast); <strong>decode cost is roughly proportional to output length</strong> (one autoregressive step per token, each at memory-bandwidth speed).</p>

<p>For a typical chat: 200-token prompt, 200-token response. Prefill: ~50ms. Decode: 200 × 20ms = 4 seconds. <strong>Decode dominates user latency by 50×</strong> in this case. Most inference optimizations target decode for this reason — every millisecond per token saves seconds per response.</p>

<h2>The KV cache</h2>

<p>Why does decode work the way it does? Because attention re-uses every previous token's K and V. To compute attention at step t, you need K and V from steps 0 through t-1. Recomputing them every step would be quadratic in t. So you cache.</p>

<p>The cache size per layer per token: <code>2 (K + V) × num_kv_heads × head_dim × bytes_per_element</code>. For a Llama-3-70B-style model:</p>

<pre><code>num_layers       = 80
num_kv_heads     = 8        <span class="com"># GQA — much fewer than num_heads (which is 64)</span>
head_dim         = 128
bytes_per_elem   = 2        <span class="com"># bf16</span>

bytes_per_token  = 80 * 2 * 8 * 128 * 2 = 327,680 bytes ≈ 320 KB
bytes_for_32K    = 32768 × 320 KB     ≈ 10.2 GB</code></pre>

<p>That's per <em>request</em>. Serving 16 concurrent 32K conversations: 163 GB of KV cache. Larger than the model itself.</p>

<p><strong>Three levers to shrink the KV cache:</strong></p>

<ol>
  <li><strong>Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)</strong>. Standard multi-head attention has one K and V per attention head — 64 each for our model. MQA shares one K/V across all heads (1 each). GQA is in between (e.g., 8 K/V groups for 64 heads). Llama 3 uses GQA-8: 8× smaller cache than full multi-head, with minimal quality cost.</li>
  <li><strong>Lower-precision cache</strong>. Store K and V in fp8 or int4 instead of bf16. Halves or quarters the memory. Quality cost is small for inference (the attention computation is robust to this).</li>
  <li><strong>Sliding-window attention</strong> (Mistral, Gemma). Only attend to the last W tokens; older keys can be evicted. KV cache size is bounded at W·320 KB regardless of conversation length.</li>
</ol>

<p>GQA is the dominant choice in 2024-2026 frontier models. It's why Llama 3's KV cache is "only" 320 KB/token instead of 2.5 MB.</p>

<h2>The naïve serving problem: pre-allocation waste</h2>

<p>Suppose you want to serve 10 concurrent users, each with a max output length of 4096 tokens. Naïve approach: allocate 10 × 4096 × 320 KB = 12.5 GB of KV cache space upfront, in 10 contiguous regions.</p>

<p>The problem: <em>users finish at different lengths</em>. User A finishes at 200 tokens; their pre-allocated 4096-slot region is now ~95% empty until A's request is gone. User B is at 3500 tokens; their region is 85% full. <em>The empty parts of A's region can't help B</em> because the regions are pre-assigned and contiguous. Memory utilization is terrible.</p>

<p>Worse: if some user actually generates 4097 tokens, you crash. So you over-provision and waste even more.</p>

<p>This is the same fragmentation problem as the caching allocator (M20), one level up. The fix is the same: <strong>break the cache into fixed-size pages and allocate them on demand from a shared pool.</strong></p>

<h2>Paged attention</h2>

<p>The vLLM team's contribution: borrow virtual memory's idea. The KV cache is divided into fixed-size <strong>pages</strong> — typically 16 tokens per page per layer. Each request has a <em>block table</em> that maps its logical token positions (0..N-1) to the physical pages where its KV vectors actually live. Pages don't need to be contiguous; the block table handles indirection.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 360" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Paged attention: block tables map logical tokens to physical pages</text>

  <!-- Two requests, each with their own logical sequence -->
  <text x="20" y="55" font-size="12" font-weight="700" fill="#1a1612">Logical sequences (what the model sees):</text>
  <g transform="translate(20, 65)">
    <text x="0" y="14" font-size="11" font-weight="700" fill="#1f5f5b">Req A (40 toks):</text>
    <rect x="120" y="0" width="50" height="20" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/>
    <text x="145" y="14" text-anchor="middle" font-size="9" fill="#1a1612">tok 0-15</text>
    <rect x="172" y="0" width="50" height="20" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/>
    <text x="197" y="14" text-anchor="middle" font-size="9" fill="#1a1612">tok 16-31</text>
    <rect x="224" y="0" width="50" height="20" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/>
    <text x="249" y="14" text-anchor="middle" font-size="9" fill="#1a1612">tok 32-39</text>

    <text x="0" y="44" font-size="11" font-weight="700" fill="#b85a6c">Req B (24 toks):</text>
    <rect x="120" y="30" width="50" height="20" fill="#ffd5dc" stroke="#b85a6c" stroke-width="1"/>
    <text x="145" y="44" text-anchor="middle" font-size="9" fill="#1a1612">tok 0-15</text>
    <rect x="172" y="30" width="50" height="20" fill="#ffd5dc" stroke="#b85a6c" stroke-width="1"/>
    <text x="197" y="44" text-anchor="middle" font-size="9" fill="#1a1612">tok 16-23</text>
  </g>

  <!-- Block tables -->
  <text x="20" y="145" font-size="12" font-weight="700" fill="#1a1612">Block tables (per request):</text>
  <g transform="translate(20, 155)">
    <text x="0" y="14" font-size="10" font-weight="700" fill="#1f5f5b">Req A:</text>
    <text x="55" y="14" font-size="10" fill="#1a1612">[ page_3, page_7, page_2 ]</text>

    <text x="0" y="32" font-size="10" font-weight="700" fill="#b85a6c">Req B:</text>
    <text x="55" y="32" font-size="10" fill="#1a1612">[ page_5, page_9 ]</text>
    <text x="350" y="22" font-size="10" font-style="italic" fill="#6b5d4f">→ pointers to physical pages</text>
  </g>

  <!-- Physical page pool -->
  <text x="20" y="220" font-size="12" font-weight="700" fill="#1a1612">Physical page pool (shared GPU memory):</text>
  <g transform="translate(20, 230)">
    <rect x="0"   y="0" width="60" height="40" fill="#ede2cc" stroke="#6b5d4f" stroke-width="1"/>
    <text x="30" y="22" text-anchor="middle" font-size="9" fill="#1a1612">page 0</text>
    <text x="30" y="34" text-anchor="middle" font-size="8" fill="#6b5d4f">(free)</text>

    <rect x="65"  y="0" width="60" height="40" fill="#ede2cc" stroke="#6b5d4f" stroke-width="1"/>
    <text x="95" y="22" text-anchor="middle" font-size="9" fill="#1a1612">page 1</text>
    <text x="95" y="34" text-anchor="middle" font-size="8" fill="#6b5d4f">(free)</text>

    <rect x="130" y="0" width="60" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="160" y="22" text-anchor="middle" font-size="9" fill="#1a1612">page 2</text>
    <text x="160" y="34" text-anchor="middle" font-size="8" fill="#1f5f5b">A[32-39]</text>

    <rect x="195" y="0" width="60" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="225" y="22" text-anchor="middle" font-size="9" fill="#1a1612">page 3</text>
    <text x="225" y="34" text-anchor="middle" font-size="8" fill="#1f5f5b">A[0-15]</text>

    <rect x="260" y="0" width="60" height="40" fill="#ede2cc" stroke="#6b5d4f" stroke-width="1"/>
    <text x="290" y="22" text-anchor="middle" font-size="9" fill="#1a1612">page 4</text>
    <text x="290" y="34" text-anchor="middle" font-size="8" fill="#6b5d4f">(free)</text>

    <rect x="325" y="0" width="60" height="40" fill="#ffd5dc" stroke="#b85a6c" stroke-width="2"/>
    <text x="355" y="22" text-anchor="middle" font-size="9" fill="#1a1612">page 5</text>
    <text x="355" y="34" text-anchor="middle" font-size="8" fill="#b85a6c">B[0-15]</text>

    <rect x="390" y="0" width="60" height="40" fill="#ede2cc" stroke="#6b5d4f" stroke-width="1"/>
    <text x="420" y="22" text-anchor="middle" font-size="9" fill="#1a1612">page 6</text>
    <text x="420" y="34" text-anchor="middle" font-size="8" fill="#6b5d4f">(free)</text>

    <rect x="455" y="0" width="60" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="485" y="22" text-anchor="middle" font-size="9" fill="#1a1612">page 7</text>
    <text x="485" y="34" text-anchor="middle" font-size="8" fill="#1f5f5b">A[16-31]</text>

    <rect x="520" y="0" width="60" height="40" fill="#ede2cc" stroke="#6b5d4f" stroke-width="1"/>
    <text x="550" y="22" text-anchor="middle" font-size="9" fill="#1a1612">page 8</text>
    <text x="550" y="34" text-anchor="middle" font-size="8" fill="#6b5d4f">(free)</text>

    <rect x="585" y="0" width="60" height="40" fill="#ffd5dc" stroke="#b85a6c" stroke-width="2"/>
    <text x="615" y="22" text-anchor="middle" font-size="9" fill="#1a1612">page 9</text>
    <text x="615" y="34" text-anchor="middle" font-size="8" fill="#b85a6c">B[16-23]</text>
  </g>

  <text x="370" y="305" font-family="'Caveat', cursive" font-size="18" fill="#1a1612" text-anchor="middle">when req A finishes: pages 2, 3, 7 freed → next request grabs them</text>
  <text x="370" y="328" font-family="'Caveat', cursive" font-size="18" fill="#c1502e" text-anchor="middle">no fragmentation — pages are uniform, allocate one when token 16 of a new req fills its current page</text>
  <text x="370" y="350" font-family="'Caveat', cursive" font-size="20" fill="#1f5f5b" text-anchor="middle">~3-5× higher concurrent users on the same GPU vs naïve pre-allocation</text>
</svg>
</div>

<p>The attention kernel becomes <em>paged FlashAttention</em>: instead of contiguous K and V tensors, it takes the block table as input and indirects through it. Each Q tile in the kernel iterates over the K/V pages listed in the block table, in order. The kernel is structurally identical to FlashAttention (M25); only the memory access pattern changes.</p>

<p>The win:</p>

<ul>
  <li><strong>No pre-allocation waste</strong>: requests grow page-by-page (one new page every 16 tokens). Short requests use few pages.</li>
  <li><strong>No fragmentation</strong>: pages are uniform-sized, slot in anywhere.</li>
  <li><strong>Prefix sharing</strong>: two requests with the same prompt prefix can share pages for the prefix portion (copy-on-write when they diverge). Saves memory for batched evaluations of the same prompt with different sampling parameters.</li>
</ul>

<p>This is what <strong>vLLM</strong> and similar inference servers do. The page size is 16 in vLLM's default; some implementations use other sizes. Production systems that don't use paging waste 50-90% of KV memory.</p>

<h2>Continuous batching</h2>

<p>The other major serving optimization. Without batching, each request goes through the model alone — bad GPU utilization since the model is memory-bound on weights, and one request barely uses any compute.</p>

<p>With <em>static batching</em>, you wait for B requests to arrive, prefill all B together, then decode all B together. Problem: requests finish at different lengths. Once the first request is done, you have B-1 requests still active in a "batch" that no longer fully reflects the GPU shape. Batch size silently shrinks; GPU utilization drops.</p>

<p>With <strong>continuous batching</strong> (vLLM, TGI, etc.), the batch is dynamic. Each forward pass through the model contains <em>whichever requests are currently active</em>. When a request finishes, it's dropped. When a new request arrives, it's added (its prefill happens as a single forward step or chunked over a few steps; then it joins the decode batch).</p>

<p>The implementation requires the inference engine to handle:</p>

<ol>
  <li><strong>Mixed prefill+decode batches</strong>: in one forward, some sequences are doing prefill (long N) while others are doing decode (N=1). The kernel must handle both — typically by treating each sequence's tokens as variable-length, with paged KV.</li>
  <li><strong>Per-sequence state</strong>: each sequence has its own KV cache pages, sampling parameters, stop conditions, generation counter.</li>
  <li><strong>Scheduler</strong>: decides each step which requests are in this batch, when to add new ones, how to handle preemption when memory is full.</li>
</ol>

<p>Result: the GPU stays at high utilization throughout, processing as many concurrent requests as KV cache memory allows. <strong>Throughput is 5-10× higher than static batching</strong> for typical chat workloads.</p>

<h2>Speculative decoding</h2>

<p>The decode phase is one token per forward. Each forward is memory-bound — most time is spent loading weights from HBM, not computing. <em>What if a forward could produce multiple tokens?</em></p>

<p>Speculative decoding's insight: the cost of a forward on K tokens is barely more than on 1 token (memory loads are the bottleneck and scale ~linearly only with K when K is small). So if you could <em>guess</em> the next K tokens cheaply and then verify all of them with the big model in one forward, you'd be doing K decode steps in the time of one — when the guess is right.</p>

<p>The recipe:</p>

<ol>
  <li>A small <strong>drafter</strong> model (typically 1-7B params if the target is 70B) generates K candidate tokens autoregressively. Cost: ~K × 1ms at the drafter's speed.</li>
  <li>The big <strong>target</strong> model processes all K tokens in one forward, producing K logits. Cost: ~30ms (about the same as one decode step since memory still dominates).</li>
  <li>For each draft token, compare the target's logit distribution with the drafter's. Use rejection sampling: accept the draft token with some probability based on the ratio; on rejection, sample from a corrected distribution. After the first reject, throw away all subsequent drafts for this round.</li>
</ol>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 320" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrSD" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Speculative decoding: drafter proposes K, target verifies all K in one forward</text>

  <!-- Drafter generation row -->
  <text x="20" y="55" font-size="12" font-weight="700" fill="#d4a017">① Drafter (1B model) generates K=5 candidate tokens autoregressively:</text>
  <g transform="translate(20, 65)">
    <rect x="0"   y="0" width="60" height="30" fill="#fff5d8" stroke="#d4a017"/>
    <text x="30" y="20" text-anchor="middle" font-size="10" fill="#1a1612">"The"</text>
    <rect x="65"  y="0" width="60" height="30" fill="#fff5d8" stroke="#d4a017"/>
    <text x="95" y="20" text-anchor="middle" font-size="10" fill="#1a1612">"cat"</text>
    <rect x="130" y="0" width="60" height="30" fill="#fff5d8" stroke="#d4a017"/>
    <text x="160" y="20" text-anchor="middle" font-size="10" fill="#1a1612">"sat"</text>
    <rect x="195" y="0" width="60" height="30" fill="#fff5d8" stroke="#d4a017"/>
    <text x="225" y="20" text-anchor="middle" font-size="10" fill="#1a1612">"on"</text>
    <rect x="260" y="0" width="60" height="30" fill="#fff5d8" stroke="#d4a017"/>
    <text x="290" y="20" text-anchor="middle" font-size="10" fill="#1a1612">"the"</text>
    <text x="350" y="20" font-size="10" fill="#6b5d4f">~5 × 1ms = 5ms</text>
  </g>

  <!-- Target verification row -->
  <text x="20" y="125" font-size="12" font-weight="700" fill="#1f5f5b">② Target (70B model) verifies all 5 in ONE forward — same cost as 1 decode:</text>
  <g transform="translate(20, 135)">
    <rect x="0" y="0" width="320" height="30" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="160" y="20" text-anchor="middle" font-size="11" fill="#1a1612">target forward over [draft_1..draft_5] → 5 logits</text>
    <text x="350" y="20" font-size="10" fill="#6b5d4f">~30ms (vs 5×30=150ms)</text>
  </g>

  <!-- Acceptance row -->
  <text x="20" y="195" font-size="12" font-weight="700" fill="#c1502e">③ Acceptance: target's distribution accepts/rejects each draft via rejection sampling</text>
  <g transform="translate(20, 205)">
    <rect x="0"   y="0" width="60" height="30" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="30" y="20" text-anchor="middle" font-size="10" fill="#1a1612">✓ "The"</text>
    <rect x="65"  y="0" width="60" height="30" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="95" y="20" text-anchor="middle" font-size="10" fill="#1a1612">✓ "cat"</text>
    <rect x="130" y="0" width="60" height="30" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="160" y="20" text-anchor="middle" font-size="10" fill="#1a1612">✓ "sat"</text>
    <rect x="195" y="0" width="60" height="30" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="225" y="20" text-anchor="middle" font-size="10" fill="#1a1612">✗ "ON"</text>
    <rect x="260" y="0" width="60" height="30" fill="#ede2cc" stroke="#6b5d4f" stroke-width="1" stroke-dasharray="3 3"/>
    <text x="290" y="20" text-anchor="middle" font-size="9" fill="#6b5d4f">discarded</text>
    <text x="350" y="20" font-size="10" fill="#6b5d4f">3 accepted + 1 corrected = 4 tokens</text>
  </g>

  <!-- Result -->
  <text x="370" y="280" font-family="'Caveat', cursive" font-size="22" fill="#c1502e" text-anchor="middle">4 tokens for the price of one decode → ~3-5× speedup typical</text>
  <text x="370" y="305" font-family="'Caveat', cursive" font-size="18" fill="#1f5f5b" text-anchor="middle">key requirement: rejection sampling preserves the target's exact output distribution</text>
</svg>
</div>

<p>Critical detail: <strong>the rejection sampling is mathematically exact</strong>. The output distribution is identical to what the target model alone would produce. You're not approximating — you're computing the same answer faster. This is why "speculative decoding" doesn't degrade quality.</p>

<p>The acceptance rate depends on how well-aligned the drafter is with the target. For aligned models (e.g., a 7B Llama drafting for 70B Llama), acceptance is typically 60-80% per token. With K=5 and 75% acceptance: average ~3.75 accepted per round, 3-5× speedup. <em>Higher</em> acceptance with techniques like Medusa (multiple draft heads on the target) or EAGLE (a learned drafter trained jointly).</p>

<p>Variants and improvements:</p>

<ul>
  <li><strong>Medusa</strong>: instead of a separate drafter, train K small heads on the target model that each predict K tokens ahead. The drafts come from the same model (free), and the verification step is inherent. Used in some commercial APIs.</li>
  <li><strong>Lookahead decoding</strong>: use n-gram statistics from earlier in the conversation as drafts. No external model needed.</li>
  <li><strong>EAGLE</strong>: train a small lightweight drafter that uses the target's hidden states. Higher acceptance than a generic small model.</li>
</ul>

<h2>Putting it together: a real serving system</h2>

<p>A modern LLM serving system (vLLM, TensorRT-LLM, SGLang, lmdeploy) combines:</p>

<ol>
  <li><strong>Quantized weights</strong> (M26): int4 or fp8, 2-4× smaller and faster on memory-bound decode.</li>
  <li><strong>Paged KV cache</strong>: 16-token pages, block tables, no fragmentation.</li>
  <li><strong>Paged FlashAttention</strong>: the kernel handles paged K and V via block tables.</li>
  <li><strong>Continuous batching</strong>: dynamic batch composition, mixed prefill+decode forwards.</li>
  <li><strong>Speculative decoding</strong> (optional): drafter + target with rejection sampling.</li>
  <li><strong>Tensor parallel</strong> (M18) for big models: split across 4-8 GPUs within a node.</li>
  <li><strong>Prefix caching</strong>: reuse pages for repeated prefixes (e.g., system prompts shared across requests).</li>
</ol>

<p>End-to-end on a 70B model with 8×H100s, with all of these on: ~50-100 tokens/sec per request at batch sizes of 100+ concurrent users. Without these: maybe 5-10 tokens/sec at batch 8. <strong>The gap between "naive serving" and "production serving" is roughly 10×.</strong> Every layer adds another 1.3-3×.</p>

<h2>Latency vs throughput: the tradeoff</h2>

<p>Two SLOs that pull in opposite directions:</p>

<ul>
  <li><strong>Time to first token (TTFT)</strong>: the prefill latency. Lower batch size + tensor parallel helps. Important for interactive UX.</li>
  <li><strong>Tokens per second (TPS) per request</strong>: the decode rate. Lower batch size helps too — your request gets a bigger share of GPU time.</li>
  <li><strong>Total throughput</strong>: tokens/sec across all requests. <em>Higher</em> batch size helps. Important for cost.</li>
</ul>

<p>Increasing the batch size reduces per-request throughput (each request gets less compute per step) but increases overall throughput (more requests served per unit time). Different products pick different points: ChatGPT-style interactive chat optimizes per-request TTFT and TPS; batch-API endpoints optimize total throughput. vLLM exposes <code>max_num_batched_tokens</code> and similar knobs to tune the tradeoff.</p>

<h2>Using a modern serving system</h2>

<p>You typically don't write inference servers from scratch. The standard tools:</p>

<pre><code><span class="com"># vLLM — most popular open-source server</span>
<span class="kw">from</span> vllm <span class="kw">import</span> LLM, SamplingParams

llm = <span class="fn">LLM</span>(
    model=<span class="str">"meta-llama/Llama-3.1-70B-Instruct"</span>,
    quantization=<span class="str">"awq"</span>,                  <span class="com"># auto-detect AWQ checkpoint</span>
    tensor_parallel_size=<span class="num">4</span>,             <span class="com"># 4 GPUs</span>
    max_model_len=<span class="num">32768</span>,
    enable_prefix_caching=<span class="kw">True</span>,
    speculative_config={
        <span class="str">"model"</span>: <span class="str">"meta-llama/Llama-3.1-8B-Instruct"</span>,
        <span class="str">"num_speculative_tokens"</span>: <span class="num">5</span>,
    },
)

prompts = [<span class="str">"Hello, how are you?"</span>, <span class="str">"What's the capital of France?"</span>]
sampling = <span class="fn">SamplingParams</span>(temperature=<span class="num">0.7</span>, max_tokens=<span class="num">512</span>)
outputs = llm.<span class="fn">generate</span>(prompts, sampling)</code></pre>

<p>That single setup turns on: int4 quantization (AWQ), tensor parallelism across 4 GPUs, paged KV cache, continuous batching, prefix caching, and speculative decoding with an 8B drafter for the 70B target. Everything we discussed in M27, configured in 8 lines.</p>

<p>For HTTP serving, vLLM offers an OpenAI-compatible API server: <code>vllm serve meta-llama/Llama-3.1-70B-Instruct ...</code>. TGI (HuggingFace's Text Generation Inference), TensorRT-LLM (NVIDIA's), and SGLang are similar.</p>

<div class="ndq">
<h4>About inference serving</h4>

<p class="q">Why is decode memory-bound but prefill compute-bound, when they're the same model?</p>
<p class="a">Matmul shape. Prefill: <code>(N, K) × (K, M)</code> where N is the prompt length (e.g., 2048). FLOPs = 2NKM, bytes ≈ KM (weights, loaded once for all N tokens). Arithmetic intensity ≈ 2N — high, compute-bound. Decode: <code>(1, K) × (K, M)</code>. FLOPs = 2KM, bytes ≈ KM. Intensity ≈ 2 — low, memory-bound. The N in the numerator of intensity is what makes prefill compute-bound. Same model, same weights, totally different bottlenecks based on how many tokens you're processing per forward.</p>

<p class="q">Can speculative decoding hurt quality?</p>
<p class="a">Mathematically no — the rejection sampling preserves the exact target distribution. Empirically, you might see slightly different outputs for the same prompt+seed compared to standard decoding, because rejection sampling consumes more random numbers. But in expectation, the two distributions are identical. <em>This is why speculative decoding is a "free lunch" optimization</em> — quality is preserved exactly, only the wall clock time changes.</p>

<p class="q">Why use a 1B drafter instead of just running the 70B model normally?</p>
<p class="a">Because most decode time on the 70B is spent waiting for weights to load from HBM (memory-bound). The drafter's KV cache + weights fit easily in cache; the drafter runs ~30× faster than the target. Generating 5 tokens with the drafter takes ~5ms; verifying all 5 with the target takes ~30ms (one forward, weights loaded once). Total: 35ms for ~4 accepted tokens vs 5×30 = 150ms for 5 tokens normally. Speedup ≈ 4×. <em>The mismatch between drafter speed and target speed is what creates the opportunity</em>.</p>

<p class="q">What's "prefix caching"?</p>
<p class="a">When many requests share a prefix — e.g., the same system prompt across all chat sessions, or in batched evaluations of one prompt with different sampling — paged KV cache lets you store the KV vectors for that prefix once and reference them from many requests' block tables. Saves prefill cost and KV memory. vLLM enables this with <code>enable_prefix_caching=True</code>. For high-throughput chat with shared system prompts, can save 30-70% of prefill compute.</p>

<p class="q">Why doesn't sliding-window attention help inference more than it does?</p>
<p class="a">It does help — Mistral's 4K window cap means the KV cache for any conversation tops out at 4K tokens regardless of length. The KV cache is bounded. But: long-context tasks (RAG, long documents) need to attend beyond the window. So sliding-window is great for chat-like applications and a constraint for document-processing applications. Modern models often combine: full attention in some layers, sliding window in others ("hybrid attention"). Gemma 2 and others use this.</p>

<p class="q">How does speculative decoding interact with continuous batching?</p>
<p class="a">Carefully. Each request can have its own number of draft tokens accepted per step, so different requests advance by different amounts. The scheduler tracks per-request token counts; the batched forward processes K candidates per request, but each request's KV cache and stop-condition logic is per-request. Adds engineering complexity, which is part of why production servers (vLLM, TensorRT-LLM) handle it for you.</p>
</div>

<h2>Code Magnets: configure a high-throughput vLLM server</h2>

<p>You're configuring a vLLM server for a 70B chat model with low TTFT, high concurrent throughput, and a smaller drafter for speculative decoding. Three magnets are wrong choices.</p>

<div class="magnets">
<p>Arrange the magnets into a working server config.</p>

<div class="magnet-pool">
  <span class="magnet">from vllm import LLM, SamplingParams</span>
  <span class="magnet">llm = LLM(</span>
  <span class="magnet">    model="meta-llama/Llama-3.1-70B-Instruct",</span>
  <span class="magnet">    quantization="awq",</span>
  <span class="magnet">    quantization="fp16",</span>
  <span class="magnet">    tensor_parallel_size=4,</span>
  <span class="magnet">    tensor_parallel_size=1,</span>
  <span class="magnet">    max_model_len=32768,</span>
  <span class="magnet">    enable_prefix_caching=True,</span>
  <span class="magnet">    speculative_config={</span>
  <span class="magnet">        "model": "meta-llama/Llama-3.1-8B-Instruct",</span>
  <span class="magnet">        "num_speculative_tokens": 5,</span>
  <span class="magnet">    },</span>
  <span class="magnet">    enable_chunked_prefill=False,</span>
  <span class="magnet">)</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">from</span> vllm <span class="kw">import</span> LLM, SamplingParams
llm = <span class="fn">LLM</span>(
    model=<span class="str">"meta-llama/Llama-3.1-70B-Instruct"</span>,
    quantization=<span class="str">"awq"</span>,
    tensor_parallel_size=<span class="num">4</span>,
    max_model_len=<span class="num">32768</span>,
    enable_prefix_caching=<span class="kw">True</span>,
    speculative_config={
        <span class="str">"model"</span>: <span class="str">"meta-llama/Llama-3.1-8B-Instruct"</span>,
        <span class="str">"num_speculative_tokens"</span>: <span class="num">5</span>,
    },
)</code></pre>
<p>The traps:</p>
<ul>
  <li><code>quantization="fp16"</code>: not actually quantization — just the dtype. For a 70B model, you want <code>"awq"</code> or <code>"gptq"</code> or <code>"fp8"</code> to get the memory and speed benefits of low-bit weights.</li>
  <li><code>tensor_parallel_size=1</code>: a 70B model in int4 needs ~35 GB; in bf16, ~140 GB. With AWQ-int4 it might fit on a single H100, but you'd want TP=4 anyway for throughput. TP=1 leaves performance on the table.</li>
  <li><code>enable_chunked_prefill=False</code>: chunked prefill is what splits long-prompt prefills across multiple steps so they can interleave with active decodes (helps continuous batching). Disabling it hurts throughput when there's a mix of short and long prompts in flight.</li>
</ul>
<p>The full pattern: <strong>quantize the weights (AWQ-int4), tensor-parallelize across 4 GPUs, enable prefix caching, configure speculative decoding with an aligned 8B drafter</strong>. This is roughly the standard production config for serving a 70B chat model on an 8-GPU node.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each inference concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>Prefill</div>
  <div>A. Compute-bound phase that processes the prompt in one big matmul per layer.</div>

  <div>Decode</div>
  <div>B. Memory-bound phase that generates one token per autoregressive step.</div>

  <div>KV cache</div>
  <div>C. Per-token K and V vectors saved across decode steps so attention isn't recomputed.</div>

  <div>Paged attention</div>
  <div>D. KV cache split into uniform pages; block table maps logical tokens → physical pages.</div>

  <div>Continuous batching</div>
  <div>E. Dynamic batch where requests join and leave each step; mixed prefill+decode forwards.</div>

  <div>Speculative decoding</div>
  <div>F. Small drafter proposes K tokens, large target verifies all K in one forward.</div>

  <div>GQA / MQA</div>
  <div>G. Share K and V across multiple attention heads → smaller KV cache, minimal quality cost.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>Prefill</strong> → A<br>
<strong>Decode</strong> → B<br>
<strong>KV cache</strong> → C<br>
<strong>Paged attention</strong> → D<br>
<strong>Continuous batching</strong> → E<br>
<strong>Speculative decoding</strong> → F<br>
<strong>GQA / MQA</strong> → G
</p>
<p>The mental shortcut: <em>prefill = compute-bound, decode = memory-bound, KV cache = the per-token state, paged = no fragmentation via block tables, continuous batch = dynamic batch composition, spec decoding = drafter+target with rejection sampling, GQA = shared KV heads</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team's chat app has p95 TTFT of 200ms but average per-token decode time of 100ms. Their users complain the responses feel slow. Which lever should they pull first, and why?</p>
<details class="answer"><summary>show answer</summary>
<p>The decode rate. With 100ms per token, generating a 200-token response takes 20 seconds. TTFT of 200ms is fine — users tolerate up to ~500ms of "thinking" before the first token streams. But streaming at 10 tokens/sec feels glacial; users want at least 30 tokens/sec to feel responsive (faster than reading).</p>
<p>Levers in order of impact: (1) <strong>Quantize weights to int4 with AWQ</strong> — typically 2-3× decode speedup since decode is memory-bound on weight loads. (2) <strong>Add speculative decoding</strong> with a small drafter — another 2-4× on top. (3) <strong>Add tensor parallelism</strong> if not present — splits weight bandwidth across GPUs. After all three: 100ms → ~10ms/token = 100 tok/s, comfortably above the responsiveness threshold. <em>Don't waste effort on TTFT; the bottleneck is decode rate</em>.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Why does paged attention work without performance penalty despite the indirect block-table lookup on every memory access?</p>
<details class="answer"><summary>show answer</summary>
<p>Three reasons. (1) <strong>The block table is tiny</strong>: ~one int per logical page (16 tokens). For a 32K-token request that's 2K integers = 8 KB. Easily fits in L2 cache, often in L1. The indirect lookup is an L2-cache hit. (2) <strong>Pages are contiguous internally</strong>: once you've followed the indirection, you're reading 16 tokens × 320 KB worth of contiguous K and V — coalesced (M22) and exactly what the kernel wants. (3) <strong>The kernel structure absorbs the indirection</strong>: paged FlashAttention does one block-table lookup per K/V tile (every BLOCK_N tokens), not per element. The bookkeeping cost is in the noise. <em>The mechanical "indirection is slow" intuition from CPU programming doesn't apply when the indirection table fits in fast cache</em>.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> A team adopts speculative decoding with a 1B drafter and a 70B target. They measure 60% acceptance rate, but only see a 1.5× speedup instead of the expected 3-4×. What's likely going on?</p>
<details class="answer"><summary>show answer</summary>
<p>Two common culprits. (1) <strong>The drafter isn't aligned with the target</strong>. Acceptance rate of 60% is low — typically 70-80% for well-aligned drafters (same model family, same finetuning recipe, similar training data). Try a drafter that's trained from the target as a teacher (distillation) or a Medusa-style drafter that uses the target's hidden states. (2) <strong>The drafter is too slow per token</strong>. If drafting K tokens takes K × 5ms instead of K × 1ms, the overhead eats into the win. Drafter should run at ~30× the target's per-token speed for good amortization. Profile drafter latency — if it's not in the &lt;1ms range, switch to a smaller drafter or one with better quantization. <em>Speculative decoding's speedup is sensitive to both metrics</em>.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> Why does increasing batch size improve total throughput but hurt per-request throughput?</p>
<details class="answer"><summary>show answer</summary>
<p>Each forward through the model has roughly fixed cost (memory bandwidth limit on weight loads). At batch 1: one forward processes 1 token. At batch 64: one forward processes 64 tokens (one per active request, in continuous batching). Batch 64 takes maybe 1.5× as long as batch 1 (extra activation work, slightly more memory pressure), but processes 64× the tokens. <em>Per-step throughput</em> goes up ~40×.</p>
<p>But each <em>individual</em> request waited for the same forward to complete — it got 1/64 of the batch's tokens, not all of them. Per-request throughput goes <em>down</em>: each request advances 1 token per step, where each step now takes 1.5× as long. So <strong>per-request tokens/sec ≈ 1/(1.5 × step_time_at_batch_1) ≈ 67% of batch-1 speed</strong>.</p>
<p>The fundamental tradeoff: you can serve more users at lower per-user speed, or fewer users at higher per-user speed. Different products pick different points; vLLM exposes the knobs to tune.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li><strong>LLM inference has two phases</strong> with completely different performance characteristics. <strong>Prefill</strong> is compute-bound (long matmuls, high arithmetic intensity). <strong>Decode</strong> is memory-bound (one-token-at-a-time, weight bandwidth dominates).</li>
  <li><strong>Decode dominates user latency</strong>: generating a 200-token response is 200 × per-token decode latency, often 50× the prefill cost.</li>
  <li><strong>The KV cache</strong> stores per-token K and V vectors to avoid recomputing attention. Size scales with conversation length × num_layers × num_kv_heads × head_dim. For long-context conversations it can exceed the model's parameter size.</li>
  <li><strong>GQA / MQA</strong> share K and V across attention heads — much smaller cache, minimal quality cost. Llama 3 uses GQA-8 (8× cache reduction).</li>
  <li><strong>Naive serving wastes memory</strong> by pre-allocating max-length KV per request. Fragmentation and over-provisioning eat 50-90% of available memory.</li>
  <li><strong>Paged attention</strong> (vLLM): KV cache split into 16-token pages; per-request block tables map logical tokens to physical pages. Pages allocated on demand from a shared pool. No fragmentation. ~3-5× more concurrent users on the same GPU.</li>
  <li><strong>Continuous batching</strong>: dynamic batch where requests join and leave each step. Mixed prefill+decode batches keep GPU utilization high. ~5-10× higher throughput than static batching.</li>
  <li><strong>Speculative decoding</strong>: small drafter proposes K tokens; target verifies all K in one forward via rejection sampling. Mathematically exact (preserves target distribution). ~3-5× decode speedup with 70-80% acceptance rate.</li>
  <li><strong>Variants</strong>: Medusa (heads on the target), EAGLE (learned drafter using target hidden states), lookahead (n-gram drafting). All preserve quality exactly.</li>
  <li><strong>The full production stack</strong>: quantized weights (M26) + paged KV + paged FlashAttention (M25) + continuous batching + speculative decoding + tensor parallelism (M18) + prefix caching. Combined: ~10× over naive serving.</li>
  <li><strong>Latency vs throughput tradeoff</strong>: smaller batch → faster per-request, lower total throughput. Larger batch → slower per-request, higher total throughput. Different products tune differently.</li>
  <li><strong>Use existing systems</strong>: vLLM, TensorRT-LLM, TGI, SGLang. Don't build inference servers from scratch — these handle the engineering.</li>
  <li>The reflex: when serving an LLM, ask "is decode the bottleneck?" (almost always yes). Then quantize + spec decode + continuous batch in that order of priority.</li>
</ul>
</div>

<p>Module 28 is the last module — <strong>Mixture of Experts and a frontier capstone</strong>. We'll cover MoE architectures (top-K routing, the all-to-all collective from M15 in action, expert sharding), why frontier models like Mixtral and DeepSeek use them, and the system-level engineering that goes into training and serving them. Then a capstone exercise that ties together everything from M1 to M28 — a single training/inference scenario where you need to reason about all of it.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">27</span>
  <span>Inference systems: KV cache, paged attention, speculative decoding</span>
</div>
"""

emit("27_inference_systems", "Module 27 — Inference systems: KV cache, paged attention, speculative decoding", BODY)
