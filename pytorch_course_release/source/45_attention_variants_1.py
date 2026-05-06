#!/usr/bin/env python3
"""Module 45: Attention Variants I — Roles, Receptive Fields & Sparsity. April 2026."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Attention Trilogy · Module 45 · April 2026 currency</div>
  <h1 class="module-title"><em>Attention Variants I:</em> roles, receptive fields &amp; sparsity</h1>
  <p class="module-sub">— first installment of a three-module attention compendium; the four-axis taxonomy, the linear-attention lineage from Mamba to Kimi Delta Attention, and the trainable sparse attention family (NSA, MoBA, DSA, XAttention)</p>
</div>

<p>M44 took inventory of 2026 LLMs and named "attention variant" as one of the five primitives where they differ. This module starts a three-part deep-dive into that primitive specifically. <em>The reason it deserves three modules is that "attention" in 2026 means dozens of distinct mechanisms</em> — each solving a different problem, each with subtle interactions with the rest of the architecture, each with its own kernel-implementation considerations. Treating them all in a single module would compress them into a list; the list is the wrong abstraction.</p>

<p>This module covers <strong>Axis A (semantic role)</strong> and <strong>Axis B (receptive field)</strong>. M46 covers Axis C (head topology and KV-cache engineering — MQA / GQA / MLA / IHA / Slim Attention). M47 covers Axis D (kernel implementations — FlashAttention 1-4, PagedAttention, Ring Attention) and Axis E (compositional architectures — MoDA, hybrid patterns). The three modules compose: any production attention layer in 2026 makes a choice on each axis simultaneously.</p>

<p>The Axis A material is the foundation: <em>what is attention, mathematically</em>, and what happens when you replace its softmax with kernel-trick linearizations. The Axis B material is the production-relevant frontier: <em>where each query attends</em>, with sliding window, global, and the new-generation trainable sparse attention mechanisms (NSA, MoBA, DSA, XAttention). By the end you'll know why linear attention has had three failed comebacks and what makes the 2025-2026 generation (Mamba-3, Gated DeltaNet, Kimi Delta Attention) genuinely different, and how the trainable sparse attention family closes the long-context gap that older fixed-pattern sparse attention couldn't.</p>

<div class="keyidea">
The four-axis taxonomy of attention. <strong>Axis A — semantic role</strong>: what's attending to what (self, cross, causal) and how the score is normalized (softmax, linear). <strong>Axis B — receptive field</strong>: where each query attends (full, sliding window, global, sparse). <strong>Axis C — head topology</strong>: how Q/K/V are shared across heads (MHA, MQA, GQA, MLA, IHA, Slim — M46). <strong>Axis D — kernel implementation</strong>: how it runs on hardware (FlashAttention 1-4, PagedAttention, Ring — M47). <strong>Axis E — composition</strong>: how attention layers compose with depth, MoE, and other primitives (MoDA, hybrid stacks — M47). Every production attention layer is a point in this 5D space.

This module's frontier material. <strong>Linear attention is back</strong>: not as a softmax replacement (the 2020-2022 attempts failed) but as a <em>complementary mechanism</em> in hybrid stacks. <strong>Mamba-3 (ICLR 2026)</strong>: complex-valued state update equivalent to data-dependent RoPE; trapezoidal discretization replacing Euler; MIMO formulation for arithmetic intensity; +1.8 points downstream over Gated DeltaNet at 1.5B; matches Mamba-2 perplexity at half the state size. <strong>Gated DeltaNet (Yang et al.)</strong>: Mamba-2 + delta rule + scalar gate; the linear-attention layer in Qwen3-Next and Qwen3.5. <strong>Kimi Delta Attention (KDA)</strong>: Gated DeltaNet refined with channel-wise gating. <strong>Lightning Attention</strong>: Ling 2.5's linear-attention choice. <strong>Sparse attention is solved at the trainable level</strong>: <strong>NSA (Yuan et al. 2025, ACL 2025 award)</strong>: three parallel paths — compressed coarse + selected fine + sliding window — with learned gates; surpasses full attention on most tasks despite sparsity. <strong>MoBA (Lu et al. 2025)</strong>: Mixture of Block Attention, trainable block routing. <strong>DSA (DeepSeek V3.2 → V4)</strong>: element-wise trainable sparsity with FP4 Lightning Indexer. <strong>XAttention (MIT-Han Lab, ICML 2025)</strong>: plug-and-play antidiagonal scoring, 13.5× speedup, no retraining. <strong>SeerAttention, InfLLM-V2, FSA, DMA</strong>: the broader trainable sparse family. <em>The 2026 frontier in attention isn't replacing softmax — it's adding sparsity-aware paths or hybridizing with linear-attention layers</em>.
</div>

<h2>Two new faces — taking attention seriously</h2>

<div class="character" style="--c: #1f5f5b;">
  <div class="avatar" style="background: #1f5f5b; color: #fff;">R</div>
  <div>
    <p class="who">Receptive Field</p>
    <p class="name">"For each query token, I decide which key tokens it can see. Full, local, or sparse — that's me."</p>
    <p class="says">In vanilla attention, every query attends to every key — the receptive field is global, the cost is O(N²). I'm the constraint that says <em>your query at position 500 doesn't need to attend to the token at position 5</em> in many cases. I come in three flavors. <strong>Sliding window</strong>: query at position i attends to keys in [i−w, i] only — fixed local. <strong>Global</strong>: every query attends to every key — full O(N²). <strong>Sparse</strong>: query attends to a content-dependent subset, learned during training. The 2025-2026 generation of trainable sparse attention (NSA, MoBA, DSA) is what made sparse attention competitive with full at scale — older fixed-pattern sparse (BigBird, Longformer) couldn't match full attention quality. <em>I'm the difference between O(N²) and O(N log N) or O(N) with no quality loss</em> when designed correctly.</p>
  </div>
</div>

<div class="character" style="--c: #d4a017;">
  <div class="avatar" style="background: #d4a017; color: #fff;">Λ</div>
  <div>
    <p class="who">Linearity</p>
    <p class="name">"I replace softmax with a kernel trick. Constant memory, linear compute — the dream that almost works."</p>
    <p class="says">Standard attention is <code>softmax(QK^T)V</code>; the softmax forces O(N²) computation because it normalizes across all keys per query. <em>I rewrite this to avoid materializing QK^T at all</em> — by replacing softmax with a kernel function φ such that <code>φ(Q)φ(K)^T V = φ(Q)(φ(K)^T V)</code>, where the right-hand parenthesization is O(N×d²) instead of O(N²×d). Constant-size state, linear in N. <strong>I died three times before working</strong>. Performer (2020) used random features, lost too much accuracy. RWKV/RetNet (2023) added decay, marginal results. The 2024-2026 generation — <strong>Mamba-2, Gated DeltaNet, Kimi Delta Attention, Lightning Attention</strong> — finally got the recipe right by combining selective decay (data-dependent gates), discretization tricks (trapezoidal in Mamba-3), and complex-valued state updates (Mamba-3's RoPE bridge). <em>I'm not a softmax replacement; I'm a complement</em>. Production 2026 stacks use me as 3 of every 4 layers with full attention as the 4th.</p>
  </div>
</div>

<h2>The four-axis taxonomy, drawn</h2>

<p>Every attention variant in production is a point in five-dimensional space. This module's two axes:</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 470" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">The five axes of attention design — this module covers A and B</text>

  <!-- Axis A -->
  <g transform="translate(20, 50)">
    <rect x="0" y="0" width="700" height="80" fill="#fff5d8" stroke="#d4a017" stroke-width="2"/>
    <text x="20" y="22" font-size="12" font-weight="700" fill="#1a1612">Axis A — Semantic role &amp; normalization</text>
    <text x="20" y="42" font-size="10" fill="#1a1612">  Self vs cross (what attends to what)  ·  Causal vs bidirectional (mask shape)</text>
    <text x="20" y="58" font-size="10" fill="#1a1612">  <tspan font-weight="700">Softmax</tspan> (standard, quadratic) vs <tspan font-weight="700">Linear</tspan> (kernel-trick, recurrent state)</text>
    <text x="20" y="74" font-size="9" font-style="italic" fill="#c1502e">  Examples: vanilla self-attention · cross-attention · causal LM · Mamba-3 · Gated DeltaNet · Kimi Delta Attention · Lightning Attention</text>
  </g>

  <!-- Axis B -->
  <g transform="translate(20, 145)">
    <rect x="0" y="0" width="700" height="80" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="20" y="22" font-size="12" font-weight="700" fill="#1a1612">Axis B — Receptive field (where each query attends)</text>
    <text x="20" y="42" font-size="10" fill="#1a1612">  <tspan font-weight="700">Full</tspan> (every query → every key, O(N²))  ·  <tspan font-weight="700">Sliding window</tspan> (local only)  ·  <tspan font-weight="700">Global</tspan> (special tokens)</text>
    <text x="20" y="58" font-size="10" fill="#1a1612">  <tspan font-weight="700">Sparse trainable</tspan> (content-dependent subset learned during training)</text>
    <text x="20" y="74" font-size="9" font-style="italic" fill="#1f5f5b">  Examples: vanilla full · Mistral sliding window · NSA (3-path) · MoBA · DSA · XAttention · SeerAttention · InfLLM-V2 · FSA · DMA</text>
  </g>

  <!-- Axis C -->
  <g transform="translate(20, 240)">
    <rect x="0" y="0" width="700" height="60" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="20" y="22" font-size="12" font-weight="700" fill="#1a1612">Axis C — Head topology &amp; KV-cache engineering · M46</text>
    <text x="20" y="42" font-size="10" fill="#1a1612">  MHA → MQA → GQA → MLA → IHA → Slim Attention · DeepSeek V4's CSA+HCA</text>
  </g>

  <!-- Axis D -->
  <g transform="translate(20, 315)">
    <rect x="0" y="0" width="700" height="60" fill="#ffd5dc" stroke="#b85a6c" stroke-width="1.5"/>
    <text x="20" y="22" font-size="12" font-weight="700" fill="#1a1612">Axis D — Kernel implementation · M47</text>
    <text x="20" y="42" font-size="10" fill="#1a1612">  FlashAttention 1/2/3/4 (Blackwell) · PagedAttention · Ring · Striped · Star · TokenRing · DeepSpeed Ulysses · USP</text>
  </g>

  <!-- Axis E -->
  <g transform="translate(20, 390)">
    <rect x="0" y="0" width="700" height="60" fill="#fff8a8" stroke="#1a1612" stroke-width="1.5"/>
    <text x="20" y="22" font-size="12" font-weight="700" fill="#1a1612">Axis E — Compositional architecture · M47</text>
    <text x="20" y="42" font-size="10" fill="#1a1612">  Hybrid stacks (Qwen3-Next 3:1, Kimi Linear, Ling 2.5, Nemotron 3 Nano) · MoDA depth attention · KArAt · MTA · Differential Transformer</text>
  </g>
</svg>
</div>

<p>Reading any 2026 model card is identifying the choices on these five axes simultaneously. This module's job is to make Axis A and Axis B legible.</p>

<h2>The mathematical baseline</h2>

<p>Before variants, the canonical thing they all modify. Standard scaled dot-product attention from Vaswani et al. (2017):</p>

<pre><code><span class="kw">def</span> <span class="fn">vanilla_attention</span>(Q, K, V, mask=<span class="kw">None</span>):
    <span class="com"># Q, K, V: [batch, seq, n_heads, d_head]</span>
    <span class="com"># Compute attention scores</span>
    scores = Q @ K.<span class="fn">transpose</span>(-<span class="num">2</span>, -<span class="num">1</span>) / math.<span class="fn">sqrt</span>(d_head)
    <span class="com"># Optional mask (causal, padding, custom)</span>
    <span class="kw">if</span> mask <span class="kw">is</span> <span class="kw">not</span> <span class="kw">None</span>:
        scores = scores.<span class="fn">masked_fill</span>(mask == <span class="num">0</span>, -<span class="fn">float</span>(<span class="str">"inf"</span>))
    <span class="com"># Softmax normalization (the expensive bit at long N)</span>
    weights = F.<span class="fn">softmax</span>(scores, dim=-<span class="num">1</span>)
    <span class="com"># Weighted sum of values</span>
    output = weights @ V
    <span class="kw">return</span> output  <span class="com"># [batch, seq, n_heads, d_head]</span></code></pre>

<p>The key facts that motivate all variants:</p>

<ul>
  <li><strong>Computational cost</strong>: O(N²d) for sequence length N, head dimension d. Quadratic in N is the central problem at long context.</li>
  <li><strong>Memory cost</strong>: the attention matrix <code>scores</code> is N×N. At N=1M with batch=1 and 64 heads, that's 64TB of FP16 memory. M22's FlashAttention solved this by never materializing the full matrix; M47 covers it.</li>
  <li><strong>Why softmax</strong>: makes scores into a probability distribution; ensures gradient flow; produces interpretable attention weights. Replacing it loses these properties.</li>
  <li><strong>The mask</strong>: a [seq, seq] tensor that determines what each query can see. Different masks → different attention semantics (causal, bidirectional, local).</li>
</ul>

<h2>Axis A: Semantic roles</h2>

<h3>Self vs cross attention</h3>

<p>The earliest distinction. <strong>Self-attention</strong>: Q, K, V all come from the same input sequence — the sequence attends to itself. This is the dominant pattern in modern decoder-only LLMs. <strong>Cross-attention</strong>: Q comes from one sequence (typically the decoder), K and V come from another (typically the encoder).</p>

<pre><code><span class="kw">def</span> <span class="fn">self_attention</span>(x, W_Q, W_K, W_V):
    Q = x @ W_Q
    K = x @ W_K   <span class="com"># same input</span>
    V = x @ W_V
    <span class="kw">return</span> <span class="fn">vanilla_attention</span>(Q, K, V)

<span class="kw">def</span> <span class="fn">cross_attention</span>(x_decoder, x_encoder, W_Q, W_K, W_V):
    Q = x_decoder @ W_Q
    K = x_encoder @ W_K   <span class="com"># different input</span>
    V = x_encoder @ W_V
    <span class="kw">return</span> <span class="fn">vanilla_attention</span>(Q, K, V)</code></pre>

<p>Where you find each in 2026:</p>

<ul>
  <li><strong>Self-attention</strong>: every transformer LLM uses this in its main blocks. Llama 4, GPT-5.5, Claude Opus 4.7, DeepSeek V4 — all primary attention is self.</li>
  <li><strong>Cross-attention</strong>: VLA models (M43) — robot demonstrations cross-attend image tokens with text tokens; multimodal models (M38) where vision and language enter as separate streams; encoder-decoder translation (T5 lineage); diffusion models cross-attend conditioning text into the image-noise stream.</li>
</ul>

<p>Architectural note: in modern decoder-only multimodal models (Llama 4, Era 3a from M44), vision and text tokens are concatenated into a single sequence before the first attention layer — this turns what would be cross-attention into early-fused self-attention. <em>Cross-attention as a distinct architectural element is mostly retreating</em>; early fusion subsumes it.</p>

<h3>Causal masking</h3>

<p>A constraint, not a separate mechanism. The mask shape:</p>

<pre><code><span class="com"># Causal mask (lower-triangular): query at position i can only attend to keys at j ≤ i</span>
mask = torch.<span class="fn">tril</span>(torch.<span class="fn">ones</span>(N, N))

<span class="com"># Bidirectional: every query attends to every key (used in BERT, encoder-only)</span>
mask = torch.<span class="fn">ones</span>(N, N)

<span class="com"># Causal + sliding window (Mistral): can attend to last w tokens only</span>
mask = torch.<span class="fn">tril</span>(torch.<span class="fn">ones</span>(N, N)) * (~<span class="fn">distance_from_diagonal</span>(N) &gt; w)</code></pre>

<p>Causal attention is a property of how decoders work — autoregressive next-token prediction forbids leaking future information into past predictions. <em>Every modern decoder-only LLM uses causal attention</em>; non-causal is reserved for encoder-only models (BERT-derivatives) and certain task-specific architectures.</p>

<h3>Softmax vs linear: the central Axis-A choice</h3>

<p>The deeper Axis A question: <em>does the model use softmax normalization or some kernelized alternative?</em> The answer determines the asymptotic compute cost and the architectural class of the model.</p>

<h2>Linear attention: the recurring comeback</h2>

<p>The kernel-trick reformulation. Start with softmax attention written explicitly:</p>

<p><code>output[i] = Σ_j (exp(Q[i] · K[j]) / Z) × V[j]</code> where <code>Z = Σ_j exp(Q[i] · K[j])</code></p>

<p>This is O(N²) because Z requires summing across all j per i. The kernel-trick rewrite: replace <code>exp(Q[i] · K[j])</code> with <code>φ(Q[i]) · φ(K[j])</code> where φ is some feature map. Now:</p>

<p><code>output[i] = (Σ_j φ(Q[i]) · φ(K[j]) × V[j]) / Z = φ(Q[i]) · (Σ_j φ(K[j]) ⊗ V[j]) / Z</code></p>

<p>The crucial move: the inner sum <code>S = Σ_j φ(K[j]) ⊗ V[j]</code> is a fixed-size matrix that can be computed once and reused across all queries — or, in autoregressive decoding, updated incrementally per token:</p>

<pre><code><span class="kw">def</span> <span class="fn">linear_attention_step</span>(q_t, k_t, v_t, S, z):
    <span class="com"># State S has shape [d_feature, d_value] — fixed-size regardless of sequence length</span>
    <span class="com"># z has shape [d_feature] — normalizer state</span>
    phi_q = <span class="fn">feature_map</span>(q_t)   <span class="com"># [d_feature]</span>
    phi_k = <span class="fn">feature_map</span>(k_t)   <span class="com"># [d_feature]</span>

    <span class="com"># Update state (this is the recurrence)</span>
    S = S + torch.<span class="fn">outer</span>(phi_k, v_t)   <span class="com"># [d_feature, d_value]</span>
    z = z + phi_k                       <span class="com"># [d_feature]</span>

    <span class="com"># Output = read from state</span>
    output = (phi_q @ S) / (phi_q @ z + <span class="num">1e-6</span>)   <span class="com"># [d_value]</span>
    <span class="kw">return</span> output, S, z</code></pre>

<p>The properties:</p>

<ul>
  <li><strong>Constant memory</strong>: S is [d_feature × d_value], doesn't grow with N.</li>
  <li><strong>Linear compute</strong>: each step is O(d²); total is O(Nd²). At long N this dominates over O(N²d).</li>
  <li><strong>Recurrent during decoding</strong>: like an RNN — state updates token by token. M33's territory.</li>
  <li><strong>Parallel during training</strong>: with associative scan tricks, you can recompute the prefix in chunks (chunkwise recurrent training).</li>
</ul>

<p><em>This sounds like a free win</em>. The catch: <strong>fixed-size state means fixed-capacity memory</strong>. Distant tokens get blurred into the state; long-range exact recall degrades. The 2020-2022 attempts (Performer, Linformer, RetNet) demonstrated this empirically — performance lagged softmax attention by enough that nobody adopted them at scale.</p>

<h3>The 2024-2026 linear attention renaissance</h3>

<p>Three innovations changed the picture:</p>

<h4>Innovation 1: Selective decay (data-dependent gates)</h4>

<p>The fundamental fix to "state gets stale": <strong>decay old state with a learned, data-dependent gate</strong>:</p>

<pre><code><span class="com"># Gated linear attention update:</span>
S = γ_t * S + torch.<span class="fn">outer</span>(phi_k, v_t)
<span class="com"># γ_t is data-dependent — computed from current input</span>
γ_t = sigmoid(W_gate @ x_t)</code></pre>

<p>Now the state has a forgetting mechanism. Old information decays unless reinforced. This breaks pure associativity (γ_t depends on past), but you can recover parallel training via <strong>chunkwise recurrent</strong> algorithms (process chunks in parallel, recurrence between chunks). The same pattern as Mamba's selective scan.</p>

<h4>Innovation 2: Delta rule + state correction</h4>

<p>DeltaNet (Schlag et al., 2021; revived by Yang et al., 2025) adds a correction step. Instead of just additive update, it computes an <em>error</em> between the new key-value association and what's currently stored, then corrects:</p>

<pre><code><span class="com"># Delta rule: compute current value for k_t given current state, correct it</span>
v_predicted = (phi_k @ S) / (phi_k @ z)
delta = β * (v_t - v_predicted)
S = S + torch.<span class="fn">outer</span>(phi_k, delta)
<span class="com"># β is a learnable rate</span></code></pre>

<p>This is the "delta" — it learns to fix wrong associations rather than blindly accumulate. <strong>Gated DeltaNet</strong> (Yang et al., 2025): combines Mamba-2's gating with the delta rule. Used as the linear-attention layer in Qwen3-Next and Qwen3.5.</p>

<h4>Innovation 3: Complex-valued state (Mamba-3)</h4>

<p>The April 2026 frontier. <strong>Mamba-3 (ICLR 2026, Lahoti et al.)</strong> introduced three improvements over Mamba-2:</p>

<ul>
  <li><strong>Trapezoidal discretization</strong>: replaces Mamba-2's Euler discretization with the trapezoidal rule — more accurate when discretizing the continuous-time state-space equations into discrete updates. The math: instead of <code>x_{t+1} = x_t + dt · f(x_t)</code> (Euler), use <code>x_{t+1} = x_t + dt/2 · (f(x_t) + f(x_{t+1}))</code> (trapezoidal). Higher-order accuracy.</li>
  <li><strong>Complex-valued state update</strong>: equivalent to data-dependent rotary embeddings. The complex multiplication <code>S_t = γ_t · S_{t-1}</code> with γ as a complex number with magnitude in [0,1] and phase that rotates with input naturally encodes <em>both</em> decay AND positional rotation. The connection: Mamba-3's complex SSM is mathematically equivalent to RoPE on the state. <em>This solves the state-tracking weakness of prior linear models</em> — synthetic tasks like parity that Mamba-2 fails, Mamba-3 solves.</li>
  <li><strong>MIMO formulation</strong>: Multi-Input Multi-Output state updates. Instead of one input → one output per channel, allows multiple inputs and outputs per state, increasing arithmetic intensity (the FLOP/byte ratio). This makes the kernel actually fast on modern GPUs — older Mamba was theoretically linear but memory-bound; Mamba-3 fixes this.</li>
</ul>

<p>Mamba-3 results: at 1.5B scale, +0.6 points downstream over Gated DeltaNet; MIMO variant adds another +1.2; matches Mamba-2 perplexity at half the state size. <em>The first time a linear-attention architecture has clearly advanced the Pareto frontier without trading off too many quality points</em>.</p>

<h4>Production linear-attention layers in April 2026</h4>

<div class="table-wrap">
<table>
<caption>Linear-attention variants in production / leading research, April 2026</caption>
<thead><tr><th>Variant</th><th>Origin</th><th>Mechanism</th><th>Used by</th></tr></thead>
<tbody>
<tr><td><strong>Mamba</strong></td><td>Gu &amp; Dao 2023</td><td>Selective SSM with input-dependent A, B, C</td><td>Original SSM-based model line</td></tr>
<tr><td><strong>Mamba-2</strong></td><td>Dao &amp; Gu 2024</td><td>Structured State Space Duality (SSD); transformer-equivalent perspective</td><td>Nemotron-H, Bamba, hybrid stacks</td></tr>
<tr><td><strong>Mamba-3</strong></td><td>Lahoti et al. ICLR 2026</td><td>Trapezoidal disc. + complex SSM (data-dep. RoPE) + MIMO</td><td>Latest research; pre-production</td></tr>
<tr><td><strong>RetNet</strong></td><td>Sun et al. 2023</td><td>Retention with exponential decay; parallel/recurrent dual form</td><td>Microsoft research lineage</td></tr>
<tr><td><strong>GLA — Gated Linear Attention</strong></td><td>Yang et al. 2024</td><td>Linear attention + scalar gating per head</td><td>Several open-weight architectures</td></tr>
<tr><td><strong>DeltaNet</strong></td><td>Schlag et al. 2021 / 2025</td><td>Delta rule for state correction</td><td>Foundational; revived recently</td></tr>
<tr><td><strong>Gated DeltaNet (GDN)</strong></td><td>Yang et al. 2025</td><td>DeltaNet + Mamba-2 gating; chunkwise recurrent training</td><td>Qwen3-Next, Qwen3.5 (3:1 ratio with attention)</td></tr>
<tr><td><strong>Kimi Delta Attention (KDA)</strong></td><td>Moonshot 2026</td><td>GDN refined: channel-wise gating instead of scalar</td><td>Kimi Linear architecture</td></tr>
<tr><td><strong>Lightning Attention</strong></td><td>Various 2024-2025</td><td>Simpler linear-attention variant (decay + projection)</td><td>Ling 2.5 (1T MoE)</td></tr>
</tbody>
</table>
</div>

<p>The 2026 production reality: <strong>none of these models use linear attention exclusively</strong>. They use it as 3 of every 4 layers (Qwen3-Next, Qwen3.5) or in alternation with full attention layers (Ling 2.5, Nemotron 3 Nano). The full-attention layer provides exact recall when needed; the linear layers provide cheap context aggregation. <em>The lesson: linear attention is a complement, not a replacement</em>.</p>

<h2>Axis B: Receptive field</h2>

<p>From M44's M22 reference, the canonical attention computation is global: every query attends to every key in the sequence (within the causal mask for decoders). This is the O(N²) cost. The receptive-field axis is about <em>what subset of keys each query attends to</em>.</p>

<h3>Sliding window (local attention)</h3>

<p>The simplest restriction: each query attends to a fixed-size local window. Mistral 7B's signature design choice; older going back to Longformer (2020).</p>

<pre><code><span class="kw">def</span> <span class="fn">sliding_window_mask</span>(N, w):
    <span class="dq">"</span><span class="dq">"</span><span class="dq">"</span><span class="com">Causal sliding window: query i attends to keys [max(0, i-w+1), i]</span><span class="dq">"</span><span class="dq">"</span><span class="dq">"</span>
    mask = torch.<span class="fn">zeros</span>(N, N)
    <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(N):
        mask[i, <span class="fn">max</span>(<span class="num">0</span>, i - w + <span class="num">1</span>):i + <span class="num">1</span>] = <span class="num">1</span>
    <span class="kw">return</span> mask</code></pre>

<p>Cost: O(N×w) instead of O(N²) — when w is fixed and N is large, this is asymptotically much cheaper. For w=4096 and N=1M, that's a 244× reduction in attention cost.</p>

<p>The downside: each query has access only to local context. Information from far back in the sequence has to <em>propagate</em> through layer-by-layer aggregation — a 32-layer model with w=4096 has theoretical receptive field ~131K tokens, but exponential decay in effective dependency strength. Long-range exact recall is poor.</p>

<p><strong>Production use</strong>: Mistral lineage. Mostly displaced in 2026 frontier models because long-range recall matters. Often used as <em>part of a hybrid</em> — sliding-window local layers alternating with full-attention global layers.</p>

<h3>Global attention (special tokens)</h3>

<p>Older idea, mostly historical now. A small number of "global" tokens (often 1-32) attend to the entire sequence; everyone else uses sliding window. Used in Longformer (2020) and BigBird (2020). Production frontier doesn't use this pattern in 2026 — sparse trainable attention does the job better.</p>

<h3>Sparse attention: the 2025-2026 trainable revolution</h3>

<p>The recent class of methods that made sparse attention competitive with full attention. Older sparse attention used <em>fixed patterns</em> (BigBird's combination of local + random + global) — these worked in narrow regimes but couldn't match full attention quality at frontier scale because the sparsity pattern was hand-designed and didn't adapt to content.</p>

<p>The 2025-2026 generation: <strong>trainable sparse attention</strong>. Sparsity patterns are learned during training based on content. Models pretrained with sparse attention from the start match or exceed full-attention models on benchmarks while being asymptotically cheaper.</p>

<h4>NSA — Native Sparse Attention (Yuan et al., DeepSeek-AI 2025)</h4>

<p>Won ACL 2025 award. The reference design for trainable sparse attention.</p>

<p>NSA's three-path hierarchical structure:</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 400" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">NSA — three parallel attention paths combined via learned gates</text>

  <!-- Input -->
  <g transform="translate(20, 50)">
    <rect x="0" y="0" width="120" height="40" fill="#fff8a8" stroke="#1a1612" stroke-width="1.5"/>
    <text x="60" y="18" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Query Q_t</text>
    <text x="60" y="32" text-anchor="middle" font-size="9" fill="#1a1612">at position t</text>
  </g>

  <!-- Three paths -->
  <g transform="translate(180, 40)">
    <!-- Path 1: Compressed -->
    <rect x="0" y="0" width="170" height="65" fill="#fff5d8" stroke="#d4a017" stroke-width="2"/>
    <text x="85" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">① Compressed (coarse)</text>
    <text x="10" y="34" font-size="9" fill="#1a1612">Block KVs into chunks of 32,</text>
    <text x="10" y="48" font-size="9" fill="#1a1612">stride 16. Mean+max pool to one</text>
    <text x="10" y="60" font-size="9" fill="#1a1612">vector per block. Attend to all blocks.</text>

    <!-- Path 2: Selected -->
    <rect x="0" y="75" width="170" height="65" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="85" y="93" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">② Selected (fine)</text>
    <text x="10" y="109" font-size="9" fill="#1a1612">Reuse compression scores to rank</text>
    <text x="10" y="121" font-size="9" fill="#1a1612">blocks; keep top-16 blocks of</text>
    <text x="10" y="133" font-size="9" fill="#1a1612">size 64 in original (uncompressed) form.</text>

    <!-- Path 3: Sliding -->
    <rect x="0" y="150" width="170" height="65" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="85" y="168" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">③ Sliding window</text>
    <text x="10" y="184" font-size="9" fill="#1a1612">Most recent 512 tokens; standard</text>
    <text x="10" y="196" font-size="9" fill="#1a1612">local attention. Captures immediate</text>
    <text x="10" y="208" font-size="9" fill="#1a1612">context patterns.</text>
  </g>

  <!-- Outputs -->
  <g transform="translate(390, 40)">
    <rect x="0" y="0" width="100" height="65" fill="#fff5d8" stroke="#d4a017" stroke-width="1.5"/>
    <text x="50" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">O_cmp</text>
    <text x="50" y="38" text-anchor="middle" font-size="8" fill="#1a1612">global summary</text>

    <rect x="0" y="75" width="100" height="65" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="50" y="97" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">O_slc</text>
    <text x="50" y="113" font-size="8" fill="#1a1612" text-anchor="middle">precise blocks</text>

    <rect x="0" y="150" width="100" height="65" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="50" y="172" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">O_win</text>
    <text x="50" y="188" font-size="8" fill="#1a1612" text-anchor="middle">local context</text>
  </g>

  <!-- Gating -->
  <g transform="translate(530, 40)">
    <rect x="0" y="50" width="170" height="100" fill="#ffd5dc" stroke="#b85a6c" stroke-width="2"/>
    <text x="85" y="72" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">Learned gates</text>
    <text x="10" y="92" font-size="9" fill="#1a1612">g_cmp(q), g_slc(q), g_win(q)</text>
    <text x="10" y="106" font-size="9" fill="#1a1612">from small MLP over query</text>
    <text x="10" y="124" font-size="10" font-weight="700" fill="#1a1612">Output =</text>
    <text x="10" y="138" font-size="10" fill="#1a1612">  g_cmp · O_cmp + g_slc · O_slc</text>
    <text x="10" y="148" font-size="10" fill="#1a1612">    + g_win · O_win</text>
  </g>

  <!-- Bottom note -->
  <g transform="translate(20, 270)">
    <rect x="0" y="0" width="700" height="120" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="350" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">Why NSA works at training and inference</text>

    <text x="20" y="46" font-size="10" fill="#1a1612">  <tspan font-weight="700">Native trainability</tspan>: hierarchical structure has differentiable gates, end-to-end gradients flow through all three paths</text>
    <text x="20" y="62" font-size="10" fill="#1a1612">  <tspan font-weight="700">Hardware alignment</tspan>: queries grouped by GQA, sparse blocks accessed contiguously, balanced arithmetic intensity</text>
    <text x="20" y="78" font-size="10" fill="#1a1612">  <tspan font-weight="700">No extra forward pass for selection</tspan>: reuses compression attention scores to rank blocks (the key efficiency trick)</text>
    <text x="20" y="94" font-size="10" fill="#1a1612">  <tspan font-weight="700">Empirical result</tspan>: surpasses full attention on most tasks at 27B/260B-tokens; perfect 64K NIAH retrieval; substantial speedups at 64K decode/forward/backward</text>
    <text x="20" y="112" font-size="10" font-style="italic" fill="#1f5f5b">  Hyperparameters from paper: window w=512, compression block l=32, stride d=16, selection block l'=64, top-n=16</text>
  </g>
</svg>
</div>

<p>NSA's algorithm walked through:</p>

<pre><code><span class="kw">def</span> <span class="fn">nsa_attention</span>(Q, K, V, query_idx):
    <span class="com"># PATH 1: Compressed attention</span>
    <span class="com"># Block K, V into chunks of 32 tokens, stride 16</span>
    K_blocks = <span class="fn">block_pool</span>(K, block_size=<span class="num">32</span>, stride=<span class="num">16</span>, op=<span class="str">"mean+max"</span>)
    V_blocks = <span class="fn">block_pool</span>(V, block_size=<span class="num">32</span>, stride=<span class="num">16</span>, op=<span class="str">"mean+max"</span>)
    cmp_scores = Q[query_idx] @ K_blocks.T   <span class="com"># [num_blocks]</span>
    O_cmp = F.<span class="fn">softmax</span>(cmp_scores) @ V_blocks
    <span class="com"># Crucial: cmp_scores will be reused below — no extra forward pass</span>

    <span class="com"># PATH 2: Selected attention (uses scores from PATH 1)</span>
    top_block_idxs = cmp_scores.<span class="fn">topk</span>(k=<span class="num">16</span>).indices   <span class="com"># top-16 blocks</span>
    K_selected = <span class="fn">gather_blocks_uncompressed</span>(K, top_block_idxs, block_size=<span class="num">64</span>)
    V_selected = <span class="fn">gather_blocks_uncompressed</span>(V, top_block_idxs, block_size=<span class="num">64</span>)
    slc_scores = Q[query_idx] @ K_selected.T   <span class="com"># [16 * 64]</span>
    O_slc = F.<span class="fn">softmax</span>(slc_scores) @ V_selected

    <span class="com"># PATH 3: Sliding window (last 512 tokens)</span>
    K_window = K[query_idx - <span class="num">512</span>:query_idx + <span class="num">1</span>]
    V_window = V[query_idx - <span class="num">512</span>:query_idx + <span class="num">1</span>]
    win_scores = Q[query_idx] @ K_window.T
    O_win = F.<span class="fn">softmax</span>(win_scores) @ V_window

    <span class="com"># GATING — learned weights from query</span>
    gates = F.<span class="fn">softmax</span>(<span class="fn">gate_mlp</span>(Q[query_idx]))   <span class="com"># [3]</span>
    output = gates[<span class="num">0</span>] * O_cmp + gates[<span class="num">1</span>] * O_slc + gates[<span class="num">2</span>] * O_win
    <span class="kw">return</span> output</code></pre>

<p>The empirical claim that justifies the complexity: <strong>NSA pretrained on 27B parameters, 260B tokens — surpasses full attention on most benchmarks despite using sparse computation</strong>. This includes reasoning-heavy tasks like DROP and GSM8K. The paper's interpretation: forced sparsity acts as regularization, focusing the model on important information and filtering noise.</p>

<h4>MoBA — Mixture of Block Attention (Lu et al., Moonshot 2025)</h4>

<p>Same family as NSA but with different routing. MoBA divides the sequence into blocks; for each query, a router selects which blocks to attend to, similar to MoE expert routing but at the attention level. The selection is data-dependent and trainable.</p>

<p>Production use: Kimi family models. Compared to NSA, MoBA is conceptually simpler (one path with routing) but doesn't have the explicit hierarchical compression-then-selection structure.</p>

<h4>DSA — DeepSeek Sparse Attention (V3.2 → V4 foundation)</h4>

<p>The trainable sparse attention used in DeepSeek V3.2, evolved into V4's CSA+HCA Hybrid Attention (covered in M44). DSA introduces <strong>element-wise trainable sparsity</strong> suitable for ultra-large LLMs — instead of block-level selection, decisions are per-element.</p>

<p>The mechanism: for each query, a learned indexer (in V4, an FP4-based "Lightning Indexer") scores all keys and selects the top-k. Unlike NSA's compression step, DSA computes selection directly. The Lightning Indexer being in FP4 (M41 territory) is the cost-saving move — selection doesn't need full precision.</p>

<p>Production use: DeepSeek V3.2 directly; V4 extends it with the CSA+HCA hybrid; <strong>GLM-5/5.1 also uses DSA</strong>.</p>

<h4>XAttention — block-sparse with antidiagonal scoring (MIT-Han Lab, ICML 2025)</h4>

<p>The plug-and-play option. Unlike NSA/MoBA/DSA which require pretraining with the sparse mechanism in place, XAttention is a <em>post-training</em> sparsification scheme.</p>

<p>The key insight: the sum of antidiagonal values in the attention matrix (lower-left to upper-right) is a powerful proxy for block importance. Compute these sums cheaply; rank blocks by importance; prune below a threshold.</p>

<pre><code><span class="kw">def</span> <span class="fn">xattention_block_score</span>(Q_block, K_block):
    <span class="dq">"</span><span class="dq">"</span><span class="dq">"</span><span class="com">Antidiagonal score for block importance.</span>
<span class="com">    Insight: antidiagonal sum captures block's diagonal energy concentration</span><span class="dq">"</span><span class="dq">"</span><span class="dq">"</span>
    scores = Q_block @ K_block.T   <span class="com"># [block_size, block_size]</span>
    <span class="com"># Sum along the antidiagonal (lower-left to upper-right)</span>
    antidiag_sum = torch.<span class="fn">sum</span>(torch.<span class="fn">flip</span>(scores, dims=[-<span class="num">1</span>]).<span class="fn">diagonal</span>())
    <span class="kw">return</span> antidiag_sum

<span class="com"># Use antidiagonal scores to keep only the most important blocks per query</span>
<span class="com"># Up to 13.5× speedup vs full attention with comparable accuracy</span></code></pre>

<p>Production properties: drop-in for existing models, no retraining. Validated on RULER, LongBench, VideoMME, VBench. Up to 13.5× attention speedup with accuracy comparable to full attention. <em>The right choice when you can't retrain</em> — applies to existing closed-weight models served via inference frameworks.</p>

<h4>Other 2025-2026 trainable sparse variants</h4>

<div class="table-wrap">
<table>
<caption>Trainable / training-aware sparse attention family, 2025-2026</caption>
<thead><tr><th>Variant</th><th>Origin</th><th>Mechanism</th><th>Notes</th></tr></thead>
<tbody>
<tr><td><strong>NSA</strong></td><td>Yuan et al. 2025 (DeepSeek-AI)</td><td>Three-path: compressed + selected + sliding</td><td>ACL 2025 award; reference design</td></tr>
<tr><td><strong>MoBA</strong></td><td>Lu et al. 2025 (Moonshot)</td><td>Block-level routing similar to MoE</td><td>Used in Kimi family</td></tr>
<tr><td><strong>SeerAttention</strong></td><td>Gao et al. 2024</td><td>Predicts attention sparsity from context</td><td>Self-distillation training pipeline</td></tr>
<tr><td><strong>InfLLM-V2</strong></td><td>Zhao et al. 2025</td><td>Unified short and long context sparsity</td><td>Single kernel handles both regimes</td></tr>
<tr><td><strong>FSA — Flash Sparse Attention</strong></td><td>Yan et al. 2025</td><td>Improves kernel efficiency for small query-head counts</td><td>Hardware-aware optimization</td></tr>
<tr><td><strong>DSA</strong></td><td>DeepSeek-AI 2025-2026</td><td>Element-wise trainable sparsity with FP4 indexer</td><td>DeepSeek V3.2, V4 foundation; GLM-5</td></tr>
<tr><td><strong>DMA</strong></td><td>Shi et al. 2025</td><td>Eviction-based sparse selection</td><td>Like KV cache eviction made trainable</td></tr>
<tr><td><strong>XAttention</strong></td><td>Xu et al. ICML 2025</td><td>Antidiagonal scoring for block importance</td><td>Plug-and-play; no retraining required</td></tr>
</tbody>
</table>
</div>

<p>The trajectory: <strong>fixed-pattern sparse (BigBird, Longformer 2020) → trainable sparse with hierarchies (NSA 2025) → element-wise trainable (DSA 2026) → hybrid attention (DeepSeek V4 CSA+HCA 2026)</strong>. Each generation closes the quality gap with full attention while extending context-length efficiency.</p>

<h2>Production zoo: Axis A and B in April 2026 frontier models</h2>

<p>Cross-referencing the variants discussed above with the M44 model zoo:</p>

<div class="table-wrap">
<table>
<caption>Axis A and B choices across April 2026 frontier models</caption>
<thead><tr><th>Model</th><th>Self/Cross</th><th>Causal</th><th>Softmax/Linear</th><th>Receptive field</th></tr></thead>
<tbody>
<tr><td><strong>Llama 4 Scout</strong></td><td>Self (early-fused MM)</td><td>Causal</td><td>Softmax</td><td>Full + iRoPE chunked attention in RoPE layers</td></tr>
<tr><td><strong>Llama 4 Maverick</strong></td><td>Self (early-fused MM)</td><td>Causal</td><td>Softmax</td><td>Full + iRoPE</td></tr>
<tr><td><strong>DeepSeek V4-Pro</strong></td><td>Self</td><td>Causal</td><td>Softmax</td><td><strong>CSA + HCA hybrid sparse</strong> (4× compress + 128× compress, FP4 indexer)</td></tr>
<tr><td><strong>GLM-5.1</strong></td><td>Self</td><td>Causal</td><td>Softmax</td><td><strong>DSA</strong> (DeepSeek Sparse Attention)</td></tr>
<tr><td><strong>Qwen3-Next / Qwen3.5</strong></td><td>Self</td><td>Causal</td><td><strong>3:1 hybrid</strong> — Linear (Gated DeltaNet) : Softmax</td><td>Full softmax in attention layers</td></tr>
<tr><td><strong>Kimi Linear / K2.6</strong></td><td>Self</td><td>Causal</td><td><strong>Hybrid</strong> — Linear (KDA) : Softmax (MLA)</td><td>Hybrid pattern</td></tr>
<tr><td><strong>Ling 2.5</strong></td><td>Self</td><td>Causal</td><td><strong>Hybrid</strong> — Linear (Lightning Attention) : Softmax (MLA)</td><td>Hybrid pattern</td></tr>
<tr><td><strong>Nemotron 3 Nano</strong></td><td>Self</td><td>Causal</td><td><strong>Hybrid</strong> — Mamba-2 SSM blocks + sparse self-attention</td><td>Mamba-Transformer hybrid</td></tr>
<tr><td><strong>Mistral Large</strong></td><td>Self</td><td>Causal</td><td>Softmax</td><td><strong>Sliding window</strong> (Mistral signature)</td></tr>
<tr><td><strong>MiniMax M2.5</strong></td><td>Self</td><td>Causal</td><td>Softmax</td><td>Full (deliberately classic)</td></tr>
<tr><td><strong>Closed (GPT-5.5, Claude 4.7, Gemini 3.1)</strong></td><td>Self (multimodal)</td><td>Causal</td><td>Softmax (presumed)</td><td>Undisclosed</td></tr>
</tbody>
</table>
</div>

<p>Three observations:</p>

<ul>
  <li><strong>Hybrid is the new mainstream for non-Llama open-weight models</strong>. Qwen3-Next, Kimi Linear, Ling 2.5, Nemotron 3 Nano all combine linear-attention layers with softmax-attention layers. The specific linear variant differs (GDN, KDA, Lightning Attention, Mamba-2) but the hybrid pattern is consistent.</li>
  <li><strong>DeepSeek lineage drives sparse attention adoption</strong>. DSA → V4's CSA+HCA → GLM-5.1 picking up DSA. The Chinese open-weight models are converging on sparse attention for efficiency.</li>
  <li><strong>Llama 4 deliberately stays softmax + full attention</strong>. The bet: combine standard attention with iRoPE position encoding for length generalization, rather than introducing sparsity. Maverick at 1M context and Scout at 10M validate this approach.</li>
</ul>

<div class="ndq">
<h4>About Axis A and B variants</h4>

<p class="q">Why did linear attention fail in 2020-2022 but succeed in 2024-2026?</p>
<p class="a">Three compounding reasons. (1) <strong>State decay was wrong</strong>. Early linear attention (Performer 2020) used random feature maps without state decay; old information accumulated indefinitely and corrupted the state. The 2024-2026 generation (GDN, KDA, Mamba) all use data-dependent decay gates that selectively forget. (2) <strong>Pure linear was the wrong target</strong>. Early work tried to fully replace softmax attention. The empirical result: linear models lagged transformers by enough to be unviable. The 2024-2026 reframing: <em>linear attention as 75% of layers, softmax as 25%</em>. The full-attention layer provides exact recall when needed; the linear layers provide cheap context aggregation. <em>Hybrid stacks dominate; pure linear stacks remain niche</em>. (3) <strong>Hardware utilization was bad</strong>. Earlier linear-attention kernels were memory-bound, running far below theoretical peak. Mamba-2 (SSD) and FlashAttention-style techniques fixed this; Mamba-3's MIMO formulation pushed arithmetic intensity higher still. <em>The renaissance is partly algorithmic, partly hardware-engineering</em>.</p>

<p class="q">What's the actual difference between NSA's three paths and a fixed BigBird-style sparse pattern?</p>
<p class="a">Three differences that compound. (1) <strong>Trainable, not fixed</strong>. NSA's gates and selection are learned; BigBird's local + random + global pattern is hand-designed. The model can adapt its sparsity to content, BigBird can't. (2) <strong>Hierarchical compression</strong>. NSA's first path compresses blocks of 32 tokens into single representations — this gives a cheap global view that BigBird's "random tokens" approximation can't match. The compression captures actual content (mean+max pooling), not just samples. (3) <strong>Selection reuses compression</strong>. NSA's selection path reranks blocks based on compression attention scores — no separate computation needed. BigBird had no equivalent; selection in BigBird is fixed at training time. <em>Empirically, NSA matches or exceeds full attention; BigBird-style sparse always lagged</em>. The fixed-pattern era of sparse attention is over for frontier models.</p>

<p class="q">When should I use sliding window attention specifically?</p>
<p class="a">Three cases where it's still the right answer. (1) <strong>Truly local-dependency tasks</strong>. Some tasks (POS tagging, syntax parsing, simple summarization of bounded paragraphs) genuinely don't need long-range dependencies. Sliding window is sufficient and cheaper than even sparse attention. (2) <strong>Hardware-constrained inference</strong>. On edge devices or older hardware where you can't afford the kernel complexity of NSA/DSA, sliding window is well-supported and fast. Mistral 7B targets this. (3) <strong>Hybrid layer in larger architectures</strong>. Several 2026 architectures alternate sliding-window layers with full-attention layers (similar to Qwen3-Next's GDN/Attn alternation but with sliding instead of GDN). For most other use cases — particularly if long-context retrieval matters — sparse trainable attention dominates. <em>Sliding window is the cheap, reliable, mostly-displaced incumbent</em>.</p>

<p class="q">Why does Mamba-3's complex-valued state matter, mathematically?</p>
<p class="a">Three intertwined reasons. (1) <strong>Complex multiplication encodes both magnitude and rotation</strong>. A real-valued decay γ ∈ [0,1] just shrinks the state. A complex-valued γ = r·e^(iθ) with r ∈ [0,1] shrinks AND rotates. The rotation is what enables state-tracking — synthetic tasks like parity require the model to remember the parity of how many 1s have been seen, which is a binary state that needs rotation (specifically, multiplication by e^(iπ) = -1) to track. Real-valued decay can't represent this; complex can. (2) <strong>Equivalent to data-dependent RoPE</strong>. The complex update <code>S_t = γ_t · S_{t-1}</code> with input-dependent γ_t can be reformulated as RoPE on the state, where the rotation angle is determined by the input. M30's RoPE on positions; Mamba-3's RoPE on state. The mathematical bridge unifies SSMs and rotary embeddings. (3) <strong>Computationally efficient</strong>. Complex multiplication is just two real multiplications and an addition. The hardware cost is small; the expressivity gain (state-tracking) is substantial. <em>Mamba-3 finally solved the state-tracking weakness that plagued every prior linear-attention architecture</em>.</p>

<p class="q">Should I expect linear attention to displace softmax attention in the next 2-3 years?</p>
<p class="a">No, but it will become a substantial fraction of layers in production stacks. Three reasons. (1) <strong>Pure linear models still lose on certain tasks</strong>. Information extraction from semi-structured documents, exact retrieval from long contexts — softmax attention remains better at these. The hybrid pattern (3:1 ratio of linear to softmax) reflects this. (2) <strong>Softmax has decades of engineering</strong>. FlashAttention 1-4, PagedAttention, the entire serving infrastructure (M47) is built around softmax. Linear attention's kernels are catching up but lag in production maturity. (3) <strong>The pattern that actually wins is hybrid</strong>. Production 2026 (Qwen3.5, Kimi Linear, Ling 2.5, Nemotron 3 Nano) uses linear AS PART OF a stack with softmax. Pure linear architectures (Mamba-only) work but rarely win at frontier scale. <em>Forecast: by 2027-2028, expect 50-75% of layers in frontier models to be linear-attention variants, with the remaining softmax layers providing exact-recall capabilities. Pure softmax stacks (current Llama 4) become a minority pattern</em>.</p>

<p class="q">How do trainable sparse attention and linear attention compare as long-context strategies?</p>
<p class="a">Different tradeoffs, different regimes. <strong>Trainable sparse attention (NSA, DSA, MoBA)</strong>: maintains softmax semantics; trained from scratch with sparsity; matches full attention quality with linear-or-better cost. Best for models where exact retrieval matters and you can pretrain with the mechanism. <strong>Linear attention (GDN, KDA, Mamba-3)</strong>: replaces softmax with kernel trick; constant memory regardless of N; slight quality cost vs full attention but with major efficiency wins. Best as a layer type within hybrid stacks where 75% of layers can be cheap and 25% provide exact recall. <strong>The trend</strong>: most 2026 frontier models use one or the other, sometimes both — DeepSeek V4 uses CSA+HCA (sparse softmax); Qwen3.5 uses Gated DeltaNet (linear) + softmax in 3:1 ratio. <em>The decision factor is whether you're willing to pretrain from scratch with the new mechanism (linear or sparse) or need to retrofit existing models (XAttention is the only post-training-only option here)</em>.</p>
</div>

<h2>Code Magnets: implement the NSA forward pass</h2>

<p>You're writing the forward pass for NSA's three-path attention. Three magnets are wrong choices.</p>

<div class="magnets">
<p>Arrange the magnets to compute the NSA output for a single query.</p>

<div class="magnet-pool">
  <span class="magnet">def nsa_forward(Q, K, V, query_idx, gate_mlp):</span>
  <span class="magnet">    K_blocks = block_pool(K, block_size=32, stride=16, op="mean+max")</span>
  <span class="magnet">    V_blocks = block_pool(V, block_size=32, stride=16, op="mean+max")</span>
  <span class="magnet">    K_blocks = K[::32]  # naive subsample</span>
  <span class="magnet">    cmp_scores = Q[query_idx] @ K_blocks.T</span>
  <span class="magnet">    O_cmp = F.softmax(cmp_scores, dim=-1) @ V_blocks</span>
  <span class="magnet">    top_block_idxs = cmp_scores.topk(k=16).indices</span>
  <span class="magnet">    K_selected = gather_blocks_uncompressed(K, top_block_idxs, block_size=64)</span>
  <span class="magnet">    V_selected = gather_blocks_uncompressed(V, top_block_idxs, block_size=64)</span>
  <span class="magnet">    slc_scores = Q[query_idx] @ K_selected.T</span>
  <span class="magnet">    slc_scores = K_selected @ Q[query_idx]  # no scaling</span>
  <span class="magnet">    O_slc = F.softmax(slc_scores, dim=-1) @ V_selected</span>
  <span class="magnet">    K_window = K[max(0, query_idx - 511):query_idx + 1]</span>
  <span class="magnet">    V_window = V[max(0, query_idx - 511):query_idx + 1]</span>
  <span class="magnet">    win_scores = Q[query_idx] @ K_window.T</span>
  <span class="magnet">    O_win = F.softmax(win_scores, dim=-1) @ V_window</span>
  <span class="magnet">    gates = F.softmax(gate_mlp(Q[query_idx]), dim=-1)  # 3-way</span>
  <span class="magnet">    gates = gate_mlp(Q[query_idx])  # raw — sum could be anything</span>
  <span class="magnet">    output = gates[0] * O_cmp + gates[1] * O_slc + gates[2] * O_win</span>
  <span class="magnet">    return output</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">def</span> <span class="fn">nsa_forward</span>(Q, K, V, query_idx, gate_mlp):
    K_blocks = <span class="fn">block_pool</span>(K, block_size=<span class="num">32</span>, stride=<span class="num">16</span>, op=<span class="str">"mean+max"</span>)
    V_blocks = <span class="fn">block_pool</span>(V, block_size=<span class="num">32</span>, stride=<span class="num">16</span>, op=<span class="str">"mean+max"</span>)
    cmp_scores = Q[query_idx] @ K_blocks.T
    O_cmp = F.<span class="fn">softmax</span>(cmp_scores, dim=-<span class="num">1</span>) @ V_blocks
    top_block_idxs = cmp_scores.<span class="fn">topk</span>(k=<span class="num">16</span>).indices
    K_selected = <span class="fn">gather_blocks_uncompressed</span>(K, top_block_idxs, block_size=<span class="num">64</span>)
    V_selected = <span class="fn">gather_blocks_uncompressed</span>(V, top_block_idxs, block_size=<span class="num">64</span>)
    slc_scores = Q[query_idx] @ K_selected.T
    O_slc = F.<span class="fn">softmax</span>(slc_scores, dim=-<span class="num">1</span>) @ V_selected
    K_window = K[<span class="fn">max</span>(<span class="num">0</span>, query_idx - <span class="num">511</span>):query_idx + <span class="num">1</span>]
    V_window = V[<span class="fn">max</span>(<span class="num">0</span>, query_idx - <span class="num">511</span>):query_idx + <span class="num">1</span>]
    win_scores = Q[query_idx] @ K_window.T
    O_win = F.<span class="fn">softmax</span>(win_scores, dim=-<span class="num">1</span>) @ V_window
    gates = F.<span class="fn">softmax</span>(<span class="fn">gate_mlp</span>(Q[query_idx]), dim=-<span class="num">1</span>)
    output = gates[<span class="num">0</span>] * O_cmp + gates[<span class="num">1</span>] * O_slc + gates[<span class="num">2</span>] * O_win
    <span class="kw">return</span> output</code></pre>
<p>The traps:</p>
<ul>
  <li><code>K_blocks = K[::32]  # naive subsample</code>: replaces NSA's mean+max pooling with simple stride-32 subsampling. The compression path needs each block to <em>summarize</em> its 32-token contents — mean+max pooling captures the average and the dominant features. Naive subsampling just keeps every 32nd token; it discards 31/32 of the information per block. The compression path then has no meaningful global view; it's just a downsampled view. Quality collapses. The pooling is essential to NSA's design.</li>
  <li><code>slc_scores = K_selected @ Q[query_idx]  # no scaling</code>: missing the canonical attention pattern. The query-key dot product should be <code>Q @ K.T</code> (query first, then transpose key). The wrong order produces a different shape and different semantics — it's computing how much each <em>key</em> attends to the query, not the other way around. Most production attention also includes scaling by sqrt(d_head) inside the dot product, but the bigger error here is the operand order. The result has the wrong shape for the subsequent softmax and weighted-sum.</li>
  <li><code>gates = gate_mlp(Q[query_idx])  # raw</code>: forgets the softmax over gate logits. The three gates need to be a probability distribution (g_cmp + g_slc + g_win = 1) so the combined output is a proper convex combination. Raw MLP outputs can be anything — negative, very large, sum to anything. Without softmax, the output magnitude varies arbitrarily, and the relative weighting of paths can flip during training in unintended ways. Standard practice for routed-gate mechanisms (NSA, MoE routing in M28) is to softmax over gate logits.</li>
</ul>
<p>The pattern: <strong>compress with pooling → use compression scores to select top blocks → uncompressed attention on selected blocks → sliding window for local context → combine via softmax-normalized gates</strong>. Each step has a specific role; the most common implementation bugs are skipping the actual pooling (Magnet 3), reversing operand order in attention (Magnet 10), and forgetting to normalize gates (Magnet 17).</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each variant to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>Self-attention</div>
  <div>A. Q, K, V all from same input sequence; the dominant pattern in modern decoder-only LLMs.</div>

  <div>Linear attention (kernel trick)</div>
  <div>B. Replaces softmax with φ(Q)φ(K)^T V to get O(N) compute and constant memory; needs decay gates to work.</div>

  <div>Mamba-3</div>
  <div>C. Trapezoidal discretization + complex-valued state (data-dep. RoPE) + MIMO; ICLR 2026; +1.8 over GDN at 1.5B.</div>

  <div>Gated DeltaNet</div>
  <div>D. Mamba-2 gating + delta rule for state correction; the linear-attention layer in Qwen3-Next/Qwen3.5.</div>

  <div>Sliding window attention</div>
  <div>E. Each query attends to last w tokens only; Mistral signature; cheap but no long-range exact recall.</div>

  <div>NSA (Native Sparse Attention)</div>
  <div>F. Three paths — compressed coarse + selected fine + sliding window — combined via learned gates; ACL 2025.</div>

  <div>XAttention</div>
  <div>G. Plug-and-play sparse: antidiagonal scoring identifies important blocks; 13.5× speedup, no retraining.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>Self-attention</strong> → A<br>
<strong>Linear attention</strong> → B<br>
<strong>Mamba-3</strong> → C<br>
<strong>Gated DeltaNet</strong> → D<br>
<strong>Sliding window</strong> → E<br>
<strong>NSA</strong> → F<br>
<strong>XAttention</strong> → G
</p>
<p>The mental shortcut: <em>self attends to itself, linear kernel-tricks, Mamba-3 goes complex, GDN gates the delta rule, sliding window stays local, NSA splits three ways, XAttention scores antidiagonals</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team is training a 7B model with 256K context target. They're choosing between (a) full attention with FlashAttention-3, (b) Qwen3-Next-style hybrid with Gated DeltaNet, (c) NSA-style trainable sparse, (d) XAttention applied at inference time. Walk through the decision.</p>
<details class="answer"><summary>show answer</summary>
<p>The decision factors:</p>
<p>(1) <strong>Full attention with FA3</strong> at 256K: cost is O(N²) for attention, which becomes ~2/3 of forward pass time at this context length. Manageable for 7B at 256K but becomes painful approaching 1M. Quality is the gold standard. <em>Right answer if the team prioritizes quality and can afford the compute</em>.</p>
<p>(2) <strong>Qwen3-Next-style hybrid (3:1 GDN to attention)</strong>: 75% of layers are linear-attention (constant memory, linear compute), 25% are full attention. At 256K, the attention layers still dominate cost but you've cut total attention work by 4×. Quality is slightly below pure full attention on retrieval tasks but matches on most others. <em>Right answer if cost matters and the model will be deployed at long context regularly</em>.</p>
<p>(3) <strong>NSA trainable sparse</strong>: requires training the model from scratch with the NSA mechanism. Compute saving is substantial (NSA paper reports speedups across decoding/forward/backward). Quality matches or exceeds full attention. <em>Right answer if the team can train from scratch and wants the best efficiency/quality tradeoff</em>. The catch: more engineering complexity (three paths, learned gates, hardware-aligned kernel).</p>
<p>(4) <strong>XAttention applied at inference</strong>: only viable if you've already trained with full attention (or are using a pretrained model). At inference, prune attention blocks via antidiagonal scoring; up to 13.5× speedup with comparable accuracy. <em>Right answer if the team is fine-tuning an existing pretrained model and can't retrain from scratch</em>. Also right for serving optimization on already-deployed models.</p>
<p><strong>The recommendation</strong> for a typical 2026 team training 7B-256K from scratch: option (b) hybrid with Gated DeltaNet. Reasons: NSA is more complex to implement; XAttention is post-training-only; full attention is more expensive than necessary at long context. The hybrid pattern is well-validated (Qwen3-Next, Qwen3.5) and tooling is mature. Plan: 75% GDN layers + 25% full softmax attention with FA3, training from scratch.</p>
<p>The general lesson: <em>the right answer depends on training-from-scratch ability and whether you can afford the engineering to implement complex mechanisms like NSA</em>. For most teams without DeepSeek-AI-scale resources, the hybrid Qwen3-Next pattern is the practical sweet spot.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Walk through why NSA's compression path uses mean+max pooling rather than just mean, and why it uses 32-token blocks rather than 16 or 64.</p>
<details class="answer"><summary>show answer</summary>
<p>(1) <strong>Mean+max pooling vs mean only</strong>:</p>
<p>Mean pooling captures the average token in a block — useful for understanding the general "topic" of a block. Max pooling captures the most extreme features — useful for understanding the dominant signals (the most semantically loaded tokens). Combining both gives a richer block summary.</p>
<p>Why it matters for NSA specifically: the compression path's output is used <em>twice</em> — once for its own attention output (O_cmp), and once for selecting top-k blocks for the selection path. Better block representations → better selection of which blocks contain the actually-relevant information. If compression were just mean pooling, blocks with rare-but-important tokens (named entities, key facts) would be averaged out; max pooling preserves these. The combination empirically outperforms mean alone.</p>
<p>(2) <strong>Block size of 32, not 16 or 64</strong>:</p>
<p>The tradeoff: smaller blocks → more granular but more compressed-block tokens (more compute in the compression path). Larger blocks → fewer compressed-block tokens but each one summarizes too much (less information per representation).</p>
<p>NSA's choice of 32 (with stride 16, so blocks overlap) is empirically validated:</p>
<ul>
<li><strong>32 tokens ≈ one sentence's worth of content</strong> for typical English. This matches a meaningful semantic unit — short enough that the block represents a coherent thought, long enough that you don't have too many blocks to attend to.</li>
<li><strong>Stride 16 (50% overlap)</strong>: the half-overlap gives smoother coverage — token boundaries don't perfectly align with sentence boundaries, so overlapping blocks ensure no critical information falls between blocks.</li>
<li><strong>Computational sweet spot</strong>: at 64K context, block size 32 with stride 16 gives ~4K compressed tokens. This is small enough that attending over them is cheap (4K² is tractable) but large enough to preserve global structure.</li>
</ul>
<p>The general lesson: <strong>NSA's hyperparameters are not arbitrary</strong>. They reflect specific tradeoffs between granularity and compression cost, validated empirically. Different tasks might benefit from different choices — long-form documents might use larger blocks (64-128); code might use smaller (16-32) since code structure is denser.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> Why does Mamba-3's complex-valued state update connect to RoPE (M30), and what does this tell us about the relationship between linear attention and positional encoding?</p>
<details class="answer"><summary>show answer</summary>
<p>The mathematical bridge:</p>
<p>(1) <strong>RoPE (M30) is rotation in 2D feature pairs</strong>. Take two adjacent feature dimensions; treat them as the real and imaginary parts of a complex number. RoPE rotates this complex number by a position-dependent angle θ(pos). The rotation preserves magnitude but encodes position via phase.</p>
<p>(2) <strong>Mamba-3's complex state update is rotation in 2D state pairs</strong>. The state S_t = γ_t × S_{t-1} where γ_t is a complex number with magnitude in [0,1] and phase determined by input. Rewriting in real/imaginary pairs: this is a rotation by angle phase(γ_t) combined with a magnitude decay |γ_t|. The rotation is data-dependent (depends on the current input), unlike RoPE which is position-dependent.</p>
<p>(3) <strong>The bridge</strong>. RoPE: position-dependent rotation of features for self-attention. Mamba-3: input-dependent rotation of state for SSM updates. Both are rotations in complex 2D feature spaces — the difference is what the rotation depends on (position vs input), and what it acts on (features per token vs cumulative state).</p>
<p>(4) <strong>Implication</strong>: Mamba-3 is mathematically connecting state-space models to rotary embeddings, which suggests deeper structural similarity between transformers and SSMs than was previously understood. The Mamba-3 paper makes this explicit. <em>Both architectures benefit from rotation-as-encoding; both can be viewed through the same complex-multiplication lens</em>.</p>
<p>(5) <strong>Why complex matters for state-tracking</strong>: synthetic tasks like parity require the model to maintain a state that flips when seeing a 1 (multiplication by e^(iπ) = -1). Real-valued decay can only shrink the state; it can't flip sign. Complex multiplication can. <em>Mamba-3 solves the state-tracking weakness by giving its state the algebraic structure (rotation in C) needed to represent these tasks</em>.</p>
<p>(6) <strong>Broader lesson</strong>: <strong>positional information and state information are dual perspectives on the same problem</strong>. Transformers handle position via RoPE on per-token features. SSMs handle "where am I in the sequence" via cumulative state. Mamba-3 unifies them — the SSM's state update IS a position-dependent transformation, made explicit via complex multiplication. This is conceptual progress, not just engineering.</p>
<p>The general implication: <em>future architectures will likely continue to merge these views</em>. Hybrid stacks (Qwen3.5 with GDN + softmax) are the engineering manifestation; Mamba-3's unification is the theoretical manifestation.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> Sketch the case for and against pure linear-attention models (Mamba-3-only, no softmax) replacing transformers in production by 2028.</p>
<details class="answer"><summary>show answer</summary>
<p><strong>Case for replacement</strong>:</p>
<p>(1) <strong>Inference economics</strong>: pure linear models have constant memory and linear compute regardless of context length. At 1M-token contexts, they're vastly cheaper than even the best sparse attention. As context windows continue growing (10M+ in Llama 4 Scout, more expected), the economics tilt further toward linear.</p>
<p>(2) <strong>Mamba-3 closed major gaps</strong>: state-tracking (via complex updates), arithmetic intensity (via MIMO), retrieval (via state-size scaling). The previously-fatal weaknesses of pure linear-attention architectures are addressed.</p>
<p>(3) <strong>Architectural simplicity</strong>: a pure Mamba-3 model has uniform layers, simpler than hybrid stacks that need to alternate two layer types. Less engineering complexity in serving.</p>
<p>(4) <strong>Hardware trends favor linear</strong>: Blackwell and beyond are bandwidth-bound for attention; linear attention is compute-bound, which is the side hardware keeps improving.</p>
<p><strong>Case against (replacement unlikely by 2028)</strong>:</p>
<p>(1) <strong>Information extraction still favors softmax</strong>: tasks requiring exact retrieval from semi-structured or unstructured documents — table parsing, named-entity extraction from forms, citation matching — softmax attention's content-based addressing remains better. Mamba-3 closed gaps but didn't fully close this one.</p>
<p>(2) <strong>Hybrid stacks already capture most benefits</strong>: 75% linear + 25% softmax (Qwen3-Next/3.5) gets you most of the efficiency gain while keeping exact-recall capability when needed. The marginal benefit of going pure linear isn't large; the engineering cost of mishandling extraction tasks is real.</p>
<p>(3) <strong>Production momentum favors hybrid</strong>: every leading 2026 hybrid model (Qwen3.5, Kimi Linear, Ling 2.5, Nemotron 3 Nano) chose hybrid. The convergence pattern from M44 suggests this is the stable equilibrium, not pure-linear.</p>
<p>(4) <strong>Closed-source frontier still uses softmax</strong>: GPT-5.5, Claude Opus 4.7, Gemini 3.1 Pro all use full softmax (with various sparsity tricks at inference time). The leaders pulling the field don't show signs of switching to pure linear. Open-weight catch-up follows the leaders' patterns.</p>
<p>(5) <strong>Long-tail tasks haven't been validated</strong>: Mamba-3 results are from 1.5B-scale benchmarks. At frontier scale (300B+), pure linear hasn't been demonstrated to match transformers across the full task distribution. Risk of distribution-specific quality cliffs.</p>
<p><strong>The forecast</strong>: by 2028, expect 50-75% of layers in production frontier models to be linear-attention variants (up from ~75% of layers in current hybrid models like Qwen3.5). Pure linear-attention frontier models remain niche; hybrid is the dominant pattern. Pure softmax frontier models also remain niche (Llama lineage, MiniMax M2.5); hybrid wins in the middle.</p>
<p>The deeper lesson: <em>"replacement" is the wrong frame</em>. The interesting question isn't whether linear replaces softmax; it's what mix of mechanisms wins. The current answer (hybrid) is likely stable.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>The four-axis taxonomy of attention. <strong>Axis A</strong>: semantic role and normalization (self/cross/causal; softmax/linear). <strong>Axis B</strong>: receptive field (full/sliding/global/sparse). Axes C, D, E covered in M46-M47.</li>
  <li><strong>Self vs cross attention</strong>: cross-attention mostly retreating in 2026 frontier models; early-fused multimodal turns most former cross-attention into self-attention.</li>
  <li><strong>Causal masking</strong>: a property of decoder-only architectures, not a separate mechanism. Lower-triangular mask forbids future-token leakage.</li>
  <li><strong>Linear attention</strong>: replaces softmax with kernel-trick φ(Q)φ(K)^T V; constant memory, linear compute. Three failed attempts (Performer, Linformer, RetNet) → 2024-2026 renaissance via selective decay, delta rule corrections, and (Mamba-3) complex-valued state.</li>
  <li><strong>Mamba-3 (ICLR 2026)</strong>: trapezoidal discretization, complex-valued state (equivalent to data-dependent RoPE), MIMO formulation. +1.8 points over Gated DeltaNet at 1.5B; matches Mamba-2 perplexity at half the state size; solves prior linear-attention state-tracking weakness.</li>
  <li><strong>Gated DeltaNet (Yang et al. 2025)</strong>: Mamba-2 gating + delta rule; the linear-attention layer in Qwen3-Next and Qwen3.5 (3:1 ratio with softmax attention).</li>
  <li><strong>Kimi Delta Attention (KDA)</strong>: GDN refined with channel-wise gating instead of scalar gating; used in Kimi Linear architecture.</li>
  <li><strong>Lightning Attention</strong>: simpler linear-attention variant; used in Ling 2.5 (1T MoE, hybrid with MLA).</li>
  <li>The 2026 production reality: <strong>no frontier model uses linear attention exclusively</strong>; hybrid stacks (3:1 linear:softmax) dominate.</li>
  <li><strong>Sliding window attention</strong>: O(N×w) cost; Mistral signature; mostly displaced in 2026 frontier; hybrid layer in some architectures.</li>
  <li><strong>Trainable sparse attention family (2025-2026)</strong>: <strong>NSA</strong> (Yuan et al., ACL 2025): three paths (compressed + selected + sliding) with learned gates; surpasses full attention. <strong>MoBA</strong> (Lu et al.): trainable block routing, used in Kimi family. <strong>DSA</strong> (DeepSeek V3.2 → V4): element-wise trainable sparsity with FP4 indexer, foundation of CSA+HCA. <strong>XAttention</strong> (MIT-Han Lab, ICML 2025): plug-and-play antidiagonal scoring, 13.5× speedup, no retraining. <strong>SeerAttention, InfLLM-V2, FSA, DMA</strong>: broader trainable sparse family.</li>
  <li>NSA hyperparameters: window w=512, compression block size=32, stride=16, selection block size=64, top-n=16. Empirically validated at 27B/260B-tokens scale.</li>
  <li>The trajectory: <strong>fixed-pattern sparse (BigBird, Longformer 2020) → trainable hierarchical (NSA 2025) → element-wise trainable (DSA 2026) → hybrid attention (DeepSeek V4 CSA+HCA 2026)</strong>.</li>
  <li>Production cross-reference: Qwen3-Next/3.5 use GDN+softmax hybrid; Kimi Linear uses KDA+MLA hybrid; Ling 2.5 uses Lightning+MLA; Nemotron 3 Nano uses Mamba-Transformer hybrid; DeepSeek V4 uses CSA+HCA sparse softmax; GLM-5.1 uses DSA; Llama 4 uses standard full softmax with iRoPE; Mistral lineage uses sliding window; closed-source flagships undisclosed.</li>
  <li>The reflex for 2026: <strong>identify the workload first</strong> — context length, retrieval requirements, training-from-scratch ability — then pick the Axis A and Axis B choices. M46 covers Axis C; M47 covers Axes D and E.</li>
</ul>
</div>

<p>Module 46 (next) tackles <strong>Axis C — head topology and KV-cache engineering</strong>: the deep dive on MHA → MQA → GQA → MLA → IHA → Slim Attention, the decoupled RoPE math that makes MLA work, the IHA pseudo-head construction, and the V-from-K reconstruction in Slim Attention.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">45</span>
  <span>Attention Variants I — Roles, Receptive Fields &amp; Sparsity</span>
</div>
"""

emit("45_attention_variants_1", "Module 45 — Attention Variants I: Roles, Receptive Fields & Sparsity", BODY)
