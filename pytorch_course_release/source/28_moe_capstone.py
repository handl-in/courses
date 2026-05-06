#!/usr/bin/env python3
"""Module 28: Mixture of Experts & frontier capstone — the final module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part VIII · Module 28 · <em>finale</em></div>
  <h1 class="module-title">Mixture of Experts &amp; <em>frontier capstone</em></h1>
  <p class="module-sub">— how MoE breaks the parameter-count vs compute tradeoff, the all-to-all collective finally deployed for real, and one capstone problem that touches every module of this course</p>
</div>

<p>Welcome to the last module. Twenty-seven modules ago we were dissecting strides on a single tensor. We've come a long way. This module covers two things. First: <strong>Mixture of Experts (MoE)</strong> — the architectural pattern that powers Mixtral, DeepSeek-V3, GPT-4-class models, and most frontier-scale systems being built today. Second: a <strong>capstone problem</strong> that asks you to design the training and serving stack for a frontier-scale MoE model from scratch, drawing on every part of the course.</p>

<p>And then a closing note — what we did, what's left to learn, and where to point your effort next.</p>

<div class="keyidea">
A standard transformer layer has one MLP. <strong>An MoE layer has N parallel MLPs ("experts")</strong> and a small <strong>router</strong> that picks the top-K (typically K=2) for each token. The forward pass: token goes to router, router picks 2 of 8 experts, those 2 experts compute, weighted sum is the output. <em>Compute scales with K, parameters with N</em>. A model with 8× the parameters does the same compute as a dense model — sparse activation in action. The price: experts must be sharded across ranks (the <strong>all-to-all</strong> collective from M15 is finally load-bearing), the router needs an auxiliary loss to prevent expert collapse, and the entire model has to be resident in GPU memory at inference time even though only K/N is active per token. <strong>Frontier-scale models are MoE because the parameter-count vs compute tradeoff is the binding constraint at scale.</strong>
</div>

<h2>Two new faces (the last new ones)</h2>

<div class="character" style="--c: #1f5f5b;">
  <div class="avatar" style="background: #1f5f5b; color: #fff;">E</div>
  <div>
    <p class="who">Expert</p>
    <p class="name">"I'm one of N parallel MLPs. Most tokens never visit me; the few that do get my full attention."</p>
    <p class="says">In a standard transformer, every token goes through the same MLP. In MoE, there are 8 of me (or 32, or 128 in DeepSeek's case). The router picks K=2 of us per token. For most tokens I'm idle — and that's the point. The model has 8× the parameters of a dense version, but each token only activates 2/8 = 25% of them. <em>Same compute, much more capacity.</em> My expertise specializes during training: some experts handle code, others natural language, others math. Nobody hand-engineers this — it emerges from the routing.</p>
  </div>
</div>

<div class="character" style="--c: #c1502e;">
  <div class="avatar" style="background: #c1502e; color: #fff;">→</div>
  <div>
    <p class="who">Router</p>
    <p class="name">"I'm a tiny linear layer that decides which K experts each token visits."</p>
    <p class="says">For each token (a vector of dimension D), I produce a logit per expert (a vector of dimension N). I take the top-K and softmax over them — that gives me K experts and K weights summing to 1. The expert outputs are weighted by these and summed. I'm just a single linear layer plus a top-K — typically &lt;0.1% of the model's parameters. The catch: <em>without a load-balancing loss, I collapse to always picking the same expert</em> (it's a winner-take-all dynamic). The auxiliary loss penalizes uneven assignment, keeping me democratic. When I work, I learn meaningful specialization — that's the magic.</p>
  </div>
</div>

<h2>The MoE layer, mechanically</h2>

<p>Replace one MLP with this:</p>

<pre><code><span class="kw">class</span> <span class="ty">MoELayer</span>(nn.Module):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, d, n_experts=<span class="num">8</span>, k=<span class="num">2</span>):
        <span class="fn">super</span>().<span class="fn">__init__</span>()
        self.experts = nn.<span class="fn">ModuleList</span>([<span class="fn">MLP</span>(d) <span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(n_experts)])
        self.router = nn.<span class="fn">Linear</span>(d, n_experts)
        self.k = k

    <span class="kw">def</span> <span class="fn">forward</span>(self, x):                              <span class="com"># x: [B, T, D]</span>
        logits = self.<span class="fn">router</span>(x)                       <span class="com"># [B, T, N_experts]</span>
        weights, indices = logits.<span class="fn">topk</span>(self.k, dim=-<span class="num">1</span>)  <span class="com"># [B, T, K], [B, T, K]</span>
        weights = weights.<span class="fn">softmax</span>(dim=-<span class="num">1</span>)              <span class="com"># normalize chosen-K weights</span>

        <span class="com"># For each token, gather outputs from its chosen K experts</span>
        out = torch.<span class="fn">zeros_like</span>(x)
        <span class="kw">for</span> e_idx <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(self.experts)):
            mask = (indices == e_idx).<span class="fn">any</span>(dim=-<span class="num">1</span>)        <span class="com"># tokens that chose this expert</span>
            <span class="kw">if</span> mask.<span class="fn">any</span>():
                expert_out = self.experts[e_idx](x[mask])
                <span class="com"># Multiply by the routing weight that selected this expert and add</span>
                <span class="com"># (simplified — real implementations are vectorized)</span>
                out[mask] += <span class="fn">apply_weight</span>(expert_out, weights, indices, e_idx)
        <span class="kw">return</span> out</code></pre>

<p>Three things to highlight:</p>

<ol>
  <li><strong>The router is a linear layer</strong>: input dim D, output dim N_experts. Cheap. About 0.05% of the layer's parameters in typical configs.</li>
  <li><strong>Top-K + softmax</strong>: pick the K best experts per token, then softmax over just those K to get weights that sum to 1. Tokens going to expert i contribute weight <code>weights[t, i]</code> to the final sum.</li>
  <li><strong>Sparse computation</strong>: each expert only processes the tokens routed to it. If routing is balanced, each expert sees ~K/N of all tokens. Total compute per token: K experts × MLP cost = same as a dense model with 1 MLP at K=1, or 2× at K=2. <em>But the parameters are N× bigger</em>.</li>
</ol>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 320" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrM" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">MoE block: router selects top-K experts, weighted sum is the output</text>

  <!-- Token in -->
  <g transform="translate(20, 60)">
    <rect x="0" y="0" width="80" height="50" fill="#fff8a8" stroke="#1a1612" stroke-width="1.5"/>
    <text x="40" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">x</text>
    <text x="40" y="38" text-anchor="middle" font-size="9" fill="#1a1612">[B, T, D]</text>
  </g>

  <!-- Router branch -->
  <path d="M 105 75 L 130 75" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrM)"/>
  <g transform="translate(135, 50)">
    <rect x="0" y="0" width="100" height="70" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="50" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">Router</text>
    <text x="50" y="38" text-anchor="middle" font-size="9" fill="#1a1612">Linear(D, N)</text>
    <text x="50" y="52" text-anchor="middle" font-size="9" fill="#1a1612">→ top-K + softmax</text>
    <text x="50" y="64" text-anchor="middle" font-size="9" fill="#c1502e">~0.05% of params</text>
  </g>

  <!-- Per-token routing decisions (e.g., token gets expert 1 + 5) -->
  <text x="245" y="75" font-size="11" font-weight="700" fill="#c1502e">→</text>
  <g transform="translate(260, 50)">
    <rect x="0" y="0" width="100" height="70" fill="#fff5d8" stroke="#d4a017" stroke-width="1"/>
    <text x="50" y="20" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">selection</text>
    <text x="50" y="36" text-anchor="middle" font-size="9" fill="#1a1612">e.g. token A → {1, 5}</text>
    <text x="50" y="48" text-anchor="middle" font-size="9" fill="#1a1612">token B → {3, 5}</text>
    <text x="50" y="60" text-anchor="middle" font-size="9" fill="#1a1612">+ weights w₁, w₂</text>
  </g>

  <!-- Experts row -->
  <text x="20" y="160" font-size="12" font-weight="700" fill="#1f5f5b">Experts (N=8 parallel MLPs):</text>
  <g transform="translate(20, 170)">
    <rect x="0"   y="0" width="80" height="50" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="40" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Expert 0</text>
    <text x="40" y="38" text-anchor="middle" font-size="8" fill="#6b5d4f">idle for token A</text>

    <rect x="85"  y="0" width="80" height="50" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2.5"/>
    <text x="125" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Expert 1</text>
    <text x="125" y="38" text-anchor="middle" font-size="8" fill="#1f5f5b" font-weight="700">A → here</text>

    <rect x="170" y="0" width="80" height="50" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="210" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Expert 2</text>
    <text x="210" y="38" text-anchor="middle" font-size="8" fill="#6b5d4f">idle</text>

    <rect x="255" y="0" width="80" height="50" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2.5"/>
    <text x="295" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Expert 3</text>
    <text x="295" y="38" text-anchor="middle" font-size="8" fill="#1f5f5b" font-weight="700">B → here</text>

    <rect x="340" y="0" width="80" height="50" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="380" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Expert 4</text>
    <text x="380" y="38" text-anchor="middle" font-size="8" fill="#6b5d4f">idle</text>

    <rect x="425" y="0" width="80" height="50" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2.5"/>
    <text x="465" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Expert 5</text>
    <text x="465" y="38" text-anchor="middle" font-size="8" fill="#1f5f5b" font-weight="700">A,B both → here</text>

    <rect x="510" y="0" width="80" height="50" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="550" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Expert 6</text>

    <rect x="595" y="0" width="80" height="50" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="635" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Expert 7</text>
  </g>

  <text x="370" y="265" font-family="'Caveat', cursive" font-size="20" fill="#1a1612" text-anchor="middle">output_t = w₁ · expert_a(x_t) + w₂ · expert_b(x_t)</text>
  <text x="370" y="290" font-family="'Caveat', cursive" font-size="22" fill="#c1502e" text-anchor="middle">8× parameters, 2× compute → "sparse activation" pattern</text>
</svg>
</div>

<h2>Why MoE: the compute-vs-parameters tradeoff</h2>

<p>Recall from M22: GPU performance has two regimes — compute-bound (limited by FLOPs) and memory-bound (limited by HBM bandwidth). Decoding LLMs is memory-bound on weight loads (M27). For a fixed compute budget, a model that has more parameters but the same per-token compute is "free" until you hit a memory-bandwidth wall.</p>

<p>Concretely: a dense 70B model spends 70B × 2 bytes = 140 GB of weight loads per token (in bf16). A MoE 8x7B model has 56B parameters total but only activates 2 of 8 experts per token: ~14B params worth of MLP loads + the (shared) attention layers. <em>Less weight bandwidth per token, larger model capacity</em>. Quality scales with parameter count; speed scales with active-per-token compute.</p>

<div class="table-wrap">
<table>
<caption>MoE vs dense: the parameter/compute decoupling</caption>
<thead><tr><th>Model</th><th>Total params</th><th>Active per token</th><th>Inference cost ≈</th></tr></thead>
<tbody>
<tr><td>Llama 3 70B (dense)</td><td>70B</td><td>70B</td><td>Big and slow</td></tr>
<tr><td>Mixtral 8x7B (8 experts, K=2)</td><td>~47B</td><td>~13B</td><td>Quality of 70B at speed of 13B</td></tr>
<tr><td>DeepSeek-V3 (256 experts + 1 shared, K=8 + 1)</td><td>671B</td><td>~37B</td><td>Quality of much larger dense at the speed of a small dense</td></tr>
<tr><td>"GPT-4-class" (rumored MoE)</td><td>~1.8T</td><td>~280B (rumored)</td><td>Frontier capability at amortized inference cost</td></tr>
</tbody>
</table>
</div>

<p>The pattern: <strong>frontier models scale parameter count via MoE while keeping per-token active compute manageable</strong>. The challenge is engineering — distributed training, expert parallelism, all-to-all collectives. Quality wins are real; the engineering cost is the price.</p>

<h2>The distributed problem: tokens go to experts on different ranks</h2>

<p>For real frontier MoE, you can't fit all N experts on one GPU. Experts get sharded across ranks — typically one or a few experts per GPU. <em>This is a new axis of parallelism, distinct from data, tensor, and pipeline parallel</em>.</p>

<p>The forward path through one MoE layer:</p>

<ol>
  <li>Each rank has tokens (its data shard) and a few experts.</li>
  <li>The router runs locally on each rank — produces routing decisions for its tokens.</li>
  <li>Tokens get sent to the rank holding their assigned experts. <strong>This is an all-to-all collective</strong> (M15).</li>
  <li>Each rank's experts process the tokens they received.</li>
  <li>Outputs get sent back to the originating ranks (another all-to-all).</li>
  <li>Each rank assembles the final output from the K experts' weighted contributions.</li>
</ol>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 360" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrEP" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#c1502e"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Expert parallelism: tokens go to experts on remote ranks via all-to-all</text>

  <!-- Initial state: 4 ranks, each with own tokens + own expert -->
  <text x="20" y="55" font-size="12" font-weight="700" fill="#1a1612">Initial: each rank has its tokens and its expert(s)</text>
  <g transform="translate(20, 65)">
    <rect x="0"   y="0" width="160" height="50" fill="#fff8a8" stroke="#1a1612" stroke-width="1"/>
    <text x="80" y="20" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Rank 0 — Expert 0</text>
    <text x="80" y="36" text-anchor="middle" font-size="9" fill="#1a1612">tokens [a, b, c, d]</text>

    <rect x="180" y="0" width="160" height="50" fill="#fff8a8" stroke="#1a1612" stroke-width="1"/>
    <text x="260" y="20" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Rank 1 — Expert 1</text>
    <text x="260" y="36" text-anchor="middle" font-size="9" fill="#1a1612">tokens [e, f, g, h]</text>

    <rect x="360" y="0" width="160" height="50" fill="#fff8a8" stroke="#1a1612" stroke-width="1"/>
    <text x="440" y="20" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Rank 2 — Expert 2</text>
    <text x="440" y="36" text-anchor="middle" font-size="9" fill="#1a1612">tokens [i, j, k, l]</text>

    <rect x="540" y="0" width="160" height="50" fill="#fff8a8" stroke="#1a1612" stroke-width="1"/>
    <text x="620" y="20" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Rank 3 — Expert 3</text>
    <text x="620" y="36" text-anchor="middle" font-size="9" fill="#1a1612">tokens [m, n, o, p]</text>
  </g>

  <!-- Arrow with all-to-all -->
  <path d="M 350 130 L 350 160" stroke="#c1502e" stroke-width="2" fill="none" marker-end="url(#arrEP)"/>
  <text x="365" y="148" font-size="11" fill="#c1502e" font-weight="700">all-to-all #1: shuffle tokens to expert ranks</text>

  <!-- After dispatch: each rank now has tokens going to its expert -->
  <text x="20" y="190" font-size="12" font-weight="700" fill="#c1502e">After all-to-all #1: each rank has the tokens routed to its expert</text>
  <g transform="translate(20, 200)">
    <rect x="0"   y="0" width="160" height="50" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/>
    <text x="80" y="20" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Rank 0 — Expert 0</text>
    <text x="80" y="36" text-anchor="middle" font-size="9" fill="#1a1612">[a, e, j, n] (chose Expert 0)</text>

    <rect x="180" y="0" width="160" height="50" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/>
    <text x="260" y="20" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Rank 1 — Expert 1</text>
    <text x="260" y="36" text-anchor="middle" font-size="9" fill="#1a1612">[b, f, k, m]</text>

    <rect x="360" y="0" width="160" height="50" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/>
    <text x="440" y="20" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Rank 2 — Expert 2</text>
    <text x="440" y="36" text-anchor="middle" font-size="9" fill="#1a1612">[c, g, i, o]</text>

    <rect x="540" y="0" width="160" height="50" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/>
    <text x="620" y="20" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Rank 3 — Expert 3</text>
    <text x="620" y="36" text-anchor="middle" font-size="9" fill="#1a1612">[d, h, l, p]</text>
  </g>

  <!-- compute then all-to-all back -->
  <text x="20" y="280" font-size="11" fill="#1a1612" font-style="italic">Each expert processes its tokens locally → all-to-all #2 sends results back to originating ranks → weighted sum</text>

  <text x="370" y="320" font-family="'Caveat', cursive" font-size="20" fill="#c1502e" text-anchor="middle">all-to-all is the dominant communication cost in MoE</text>
  <text x="370" y="345" font-family="'Caveat', cursive" font-size="20" fill="#1f5f5b" text-anchor="middle">two collectives per MoE layer × N layers per step → comm-heavy</text>
</svg>
</div>

<p>The all-to-all collective from M15 finally gets its real-world workout. <strong>Each token's data has to travel from its data-rank to its expert-rank and back, every MoE layer, every step.</strong> For a model with 8 MoE layers across 16 ranks, that's 32 all-to-all collectives per training step. Inter-node bandwidth becomes a real constraint — which is why MoE training stacks pin expert parallelism inside a node when possible.</p>

<h2>Load balancing: the auxiliary loss</h2>

<p>The naïve router has a winner-take-all dynamic: if one expert is slightly better at the start, it gets more tokens, which trains it more, which makes it even better. Eventually the router collapses to picking the same 1-2 experts for every token. <em>You've trained an MoE model that's effectively a dense model with one expert worth of capacity.</em></p>

<p>The fix is an <strong>auxiliary load-balancing loss</strong>. After the router runs, compute:</p>

<pre><code><span class="com"># For one MoE layer, given routing decisions on all tokens in the batch:</span>
<span class="com"># f_i = fraction of tokens routed to expert i</span>
<span class="com"># P_i = average router probability assigned to expert i (over all tokens)</span>
<span class="com">#</span>
<span class="com"># aux_loss = N * sum_i(f_i * P_i)</span>
<span class="com">#</span>
<span class="com"># Both f_i and P_i are 1/N if perfectly balanced; loss is minimized at balance.</span>

aux_loss = N_experts * (token_fractions * router_probs).<span class="fn">sum</span>()
total_loss = main_loss + <span class="num">0.01</span> * aux_loss   <span class="com"># typical coefficient ~0.01-0.1</span></code></pre>

<p>The combination of <code>f_i</code> (counted; differentiable through token assignment is tricky) and <code>P_i</code> (the smooth probabilities) gives a smooth gradient that pushes toward balance. <strong>This is mandatory for MoE training</strong> — without it, the model collapses within hundreds of steps.</p>

<p>Other balancing techniques — different schemes are stacked together in production:</p>

<ul>
  <li><strong>Capacity factor</strong>: cap how many tokens each expert can take per batch (typically <code>1.25 × tokens / N_experts</code>). Overflow tokens get dropped or routed to a fallback. Hard limit on imbalance.</li>
  <li><strong>Expert-choice routing</strong>: invert the role — instead of each token picking K experts, each expert picks the top-T tokens. Guarantees perfect balance but breaks autoregressive generation (an expert can "see" tokens before they're emitted).</li>
  <li><strong>Free-form routing with router-Z loss</strong>: penalize the router's logits' L2 magnitude to keep routing decisions soft.</li>
</ul>

<h2>5D parallelism: the modern stack</h2>

<p>Module 18 introduced 4D parallelism: data + tensor + pipeline + sequence. MoE adds a fifth axis — <strong>expert parallelism (EP)</strong>. The full hierarchy of axes you'll see in a frontier-scale training run:</p>

<div class="table-wrap">
<table>
<caption>The full 5D parallelism stack for frontier MoE</caption>
<thead><tr><th>Axis</th><th>What it splits</th><th>Bandwidth need</th><th>Where</th></tr></thead>
<tbody>
<tr><td>Data parallel (DP)</td><td>Batch across copies</td><td>Low (one all-reduce/step)</td><td>Outermost — across nodes/clusters</td></tr>
<tr><td>Pipeline parallel (PP)</td><td>Layers into stages</td><td>Low-medium (point-to-point)</td><td>Across nodes</td></tr>
<tr><td>Expert parallel (EP)</td><td>Experts across ranks</td><td>Medium (all-to-all per layer)</td><td>Within node when possible</td></tr>
<tr><td>Tensor parallel (TP)</td><td>Matmul within layer</td><td>High (all-reduce per matmul)</td><td>Within node — NVLink</td></tr>
<tr><td>Sequence parallel (SP)</td><td>Per-token regions of activations</td><td>Bundled with TP</td><td>Within TP groups</td></tr>
</tbody>
</table>
</div>

<p>For a 1T-param MoE training run on 1024 GPUs, a typical configuration: TP=8, EP=8, PP=4, DP=4. Total: 8×8×4×4 = 1024. The choice of axis sizes is a topology decision — match each parallelism's communication pattern to the network fabric that can serve it. NVLink for TP and EP (high-bandwidth all-reduces and all-to-alls); InfiniBand for PP and DP (lower-frequency, larger messages).</p>

<h2>Inference of MoE</h2>

<p>MoE inference has a different shape from dense inference. The key facts:</p>

<ul>
  <li><strong>Compute scales with K active experts</strong>. For Mixtral 8x7B with K=2: ~13B active params per token. Throughput is similar to a dense 13B model.</li>
  <li><strong>Memory scales with N total experts</strong>. The whole model has to be resident — all 47B params for Mixtral 8x7B. You can't unload inactive experts because routing is dynamic.</li>
  <li><strong>The all-to-all in serving</strong>: even at inference, MoE adds collective overhead. For single-GPU serving (small MoE), no problem. For multi-GPU serving, the EP all-to-all becomes a real cost.</li>
  <li><strong>Routing imbalance at inference</strong>: at training the auxiliary loss kept things balanced; at inference there's no loss. Real input distributions can cause some experts to be hot. Production servers (vLLM, TensorRT-LLM) handle this with capacity overflow buffers and expert replication.</li>
</ul>

<p>The practical recipe for MoE serving: <strong>quantize all experts (M26), one expert per rank in EP=N configuration, paged KV (M27), continuous batching (M27)</strong>. You can run Mixtral 8x7B comfortably on a single H100 with int4 expert weights; DeepSeek-V3 needs multi-node.</p>

<h2>The capstone problem</h2>

<p>Time to put it together. Here's the scenario.</p>

<div class="keyidea">
<strong>Capstone:</strong> You're tasked with designing the training and serving recipe for a new frontier-scale model:
<ul>
  <li>Architecture: <strong>MoE transformer, 64 experts × 4B params/expert + 12B shared (attention + embedding) = ~270B total params</strong></li>
  <li>Active per token: K=4 experts → ~28B active params</li>
  <li>Context length: 128K tokens</li>
  <li>Training cluster: 256 H100 GPUs across 32 nodes, NVLink within nodes, 400 Gbps InfiniBand between
  <li>Inference target: serve at ≥40 tok/sec per request, &gt;100 concurrent users on 8 H100s</li>
</ul>
You need to specify: parallelism configuration, mixed precision setup, attention implementation, optimizer setup, KV cache strategy, quantization recipe, and serving stack. Make every decision and justify it against a specific module's content.
</div>

<p>Work through it. The proposed solution below references every part of the course. Read it after you've drafted your own.</p>

<details class="answer"><summary>show proposed solution</summary>

<h4>Training stack</h4>

<ul>
  <li><strong>5D parallelism</strong>: TP=8 (within node, NVLink for matmul all-reduces — M18), EP=8 (within node, NVLink for expert all-to-alls — M28), PP=4 (across 4 nodes per pipeline copy, point-to-point sends — M18), DP=1 (after TP × EP × PP = 256, no DP axis remaining for 256 GPUs). Total: 8 × 8 × 4 × 1 = 256.</li>
  <li><strong>Why those sizes</strong>: TP and EP both want NVLink, fit within an 8-GPU node together. PP across nodes, tolerates higher latency point-to-point. The 4-node pipeline depth gives 4 pipeline stages.</li>
  <li><strong>FSDP within the DP axis would help</strong> if we had more GPUs (M17), but with TP=8 and EP=8 each rank already holds only 1/64 of parameters — comparable savings to ZeRO-3 already.</li>
  <li><strong>Mixed precision</strong>: bf16 for forward and backward (M14), with fp32 master copy (M12). fp8 (M14, M26) is an option for matmuls on H100, would give ~1.5× more throughput, but adds engineering complexity — defer to a v2.</li>
  <li><strong>Attention</strong>: FlashAttention (M25) is mandatory at 128K context. Without it, the (T,T) intermediate is (128K)² × 2 bytes / head × heads = many GB per layer just in activations. Use <code>F.scaled_dot_product_attention</code> with FlashAttention backend; pads to TC-friendly head dim 128.</li>
  <li><strong>Sequence parallelism</strong> bundled with TP (M18) — splits activation memory along T in non-matmul regions. At 128K seq length this saves a lot of activation memory; nearly required.</li>
  <li><strong>Activation checkpointing</strong> (M12): every transformer block. Doubles forward time but cuts activation memory by ~10×. Combined with FSDP + FlashAttention + SP, makes 128K training fit.</li>
  <li><strong>Optimizer</strong>: AdamW (M9) with linear warmup → cosine schedule. Maybe Muon (M9) for select layers if recent research holds — would reduce optimizer state and improve loss curves.</li>
  <li><strong>Auxiliary load-balancing loss</strong>: critical for MoE (M28). Coefficient ~0.01. Plus capacity factor 1.25 to cap per-expert imbalance.</li>
  <li><strong>DataLoader</strong>: DistributedSampler (M10) with persistent_workers, pin_memory, num_workers=8 per rank. For 128K context, pre-pack documents into fixed-size sequences server-side.</li>
  <li><strong>Compile</strong>: <code>torch.compile</code> (M21) with FSDP2-style integration. MoE's dynamic routing has historically been hard for Dynamo; may need to compile per-expert and the router separately. fullgraph=True initially to find break points.</li>
  <li><strong>Profiling</strong>: PyTorch profiler (M13) every few thousand steps. Watch for the four bottleneck shapes — likely GPU-bound on matmul early, then comm-bound (all-to-all latency) as scale grows.</li>
</ul>

<h4>Serving stack</h4>

<ul>
  <li><strong>Quantize</strong>: AWQ-int4 (M26) on all expert weights. Total weight memory: 270B × 0.5 bytes = 135 GB. Distributed: ~17 GB per H100 with EP=8 — fits. Keep attention weights and embeddings in bf16 (M26 — small relative to expert MLPs, sensitive to precision).</li>
  <li><strong>Tensor + Expert parallel</strong>: TP=4, EP=8 (or TP=8, EP=8 for batched inference of larger sizes). 32 GPUs serving the model, but a 8-GPU configuration is fine for most loads.</li>
  <li><strong>Paged FlashAttention</strong> (M27): handles 128K-token requests with paged KV cache, no fragmentation. Block size 16 tokens.</li>
  <li><strong>Continuous batching</strong> (M27): vLLM or TensorRT-LLM with MoE support. Dynamic batch composition.</li>
  <li><strong>Speculative decoding</strong> (M27): the MoE router has variable per-token compute, which complicates draft acceptance. Use a small (1-2B) dense drafter; ~50-65% acceptance rate typical for MoE targets.</li>
  <li><strong>Prefix caching</strong> (M27): essential for shared system prompts. ~30-50% prefill savings in chat workloads.</li>
  <li><strong>Per-expert capacity buffer</strong>: at inference, hot experts can overflow capacity. Buffer 1.5× expected per-expert tokens, with overflow handled via sequential fallback.</li>
</ul>

<h4>Failure modes to anticipate</h4>

<ul>
  <li><strong>Training divergence from imbalanced routing</strong>: monitor router entropy + expert utilization variance. Tweak aux loss coefficient or capacity factor.</li>
  <li><strong>OOM during 128K context training</strong>: most likely from forgetting sequence parallelism or activation checkpointing. Use memory snapshot tool (M20) to find the peak.</li>
  <li><strong>Slow all-to-all</strong>: network topology mismatch between EP and physical layout. Run NCCL_DEBUG=INFO to verify topology routing.</li>
  <li><strong>Inference TPS lower than 40</strong>: most likely decode bottleneck. Quantize to int4 (M26) first, then add speculative decoding (M27).</li>
  <li><strong>NaN in loss</strong>: walk M14's diagnosis tree. Forward hooks (M5) to find first NaN; reduce LR or switch optimizer; check init scheme (M8).</li>
</ul>

<p><strong>What this exercise reveals</strong>: every decision references a specific module's content. Modern frontier ML is not one technique but a coordinated set of optimizations — memory, compute, communication, numerics, kernels, serving — each individually understandable, jointly difficult. <em>You've now seen the components. The art is composing them.</em></p>

</details>

<h2>Closing notes</h2>

<p>You started Module 1 by inspecting strides on a tensor. You're ending here, designing the training and inference stack for a frontier-scale model. The journey covered:</p>

<ul>
  <li><strong>Part I (Tensor Foundations)</strong>: tensors, indexing, dtypes — the data structures everything else builds on.</li>
  <li><strong>Part II (Autograd)</strong>: the computational graph that makes gradient descent work.</li>
  <li><strong>Part III (Building Models)</strong>: the modules, init schemes, optimizers — the toolkit for actual training.</li>
  <li><strong>Part IV (Data &amp; Training)</strong>: the loops, samplers, checkpoints — the production training engineering.</li>
  <li><strong>Part V (Performance)</strong>: memory, profiling, mixed precision — how to predict and optimize what your code will do.</li>
  <li><strong>Part VI (Distributed)</strong>: collectives, DDP, FSDP, tensor &amp; pipeline parallelism — scaling across hardware.</li>
  <li><strong>Part VII (Compilation/Internals)</strong>: dispatcher, CUDA semantics, compile — what's actually happening under PyTorch.</li>
  <li><strong>Part VIII (Kernels &amp; Frontiers)</strong>: GPU model, CUDA, Triton, FlashAttention, quantization, serving, MoE — writing and serving the modern frontier.</li>
</ul>

<p>What you're missing:</p>

<ul>
  <li><strong>Reinforcement learning from human feedback</strong> — the post-training phase that turns base models into chat models. Out of scope here; covered well by other resources.</li>
  <li><strong>Specific architectures</strong> — diffusion models, multimodal models, retrieval-augmented systems. Each has its own engineering quirks.</li>
  <li><strong>The empirical knowledge</strong> — what hyperparameters work, which init schemes are robust, how to debug specific training failures. This comes from running experiments, not from reading.</li>
  <li><strong>The frontier itself</strong> — fp4, mixture-of-depths, world models, longer-than-1M context. The list keeps growing. You now have the framework to read papers and understand them; that's what we built.</li>
</ul>

<p>What to do next:</p>

<ol>
  <li><strong>Build something</strong>. Pick a small, real task — a tiny LLM trained from scratch, a custom kernel for an op you care about, a serving system for a model you've fine-tuned. Run into the problems firsthand.</li>
  <li><strong>Read papers actively</strong>. With this background, papers like FlashAttention, Megatron, ZeRO, DeepSpeed, Mixtral, Llama 3, vLLM are now legible — go read them and trace the engineering decisions to their motivations.</li>
  <li><strong>Profile</strong>. Every claim you read in a paper is checkable with the profiler. Build the habit of "is that real?" — reproducing claims grounds your understanding.</li>
  <li><strong>Contribute</strong>. PyTorch, vLLM, Triton, torchao, FlashAttention — all open-source, all welcoming contributors. Your kernel knowledge from M22-M25 is exactly what's wanted.</li>
</ol>

<p>The point of all of this is not to know everything — nobody does — but to <em>have the framework to learn anything new in the field</em>. When the next architectural breakthrough lands, when fp4 ships, when whatever comes after MoE is announced — you'll have the dispatcher, the roofline, the parallelism axes, the tile-based kernel mindset to slot it in.</p>

<p>Thanks for reading. Now go make something.</p>

<h2>Code Magnets: design an MoE forward pass</h2>

<p>You're writing the simplified forward pass of an MoE layer. Three magnets are wrong choices.</p>

<div class="magnets">
<p>Arrange the magnets into a working forward.</p>

<div class="magnet-pool">
  <span class="magnet">def forward(self, x):</span>
  <span class="magnet">    logits = self.router(x)</span>
  <span class="magnet">    weights, indices = logits.topk(self.k, dim=-1)</span>
  <span class="magnet">    weights = weights.softmax(dim=-1)</span>
  <span class="magnet">    weights = weights.sigmoid()</span>
  <span class="magnet">    out = torch.zeros_like(x)</span>
  <span class="magnet">    for e_idx in range(len(self.experts)):</span>
  <span class="magnet">        mask = (indices == e_idx).any(dim=-1)</span>
  <span class="magnet">        if mask.any():</span>
  <span class="magnet">            expert_out = self.experts[e_idx](x[mask])</span>
  <span class="magnet">            out[mask] += apply_weight(expert_out, weights, indices, e_idx)</span>
  <span class="magnet">    aux_loss = self.compute_load_balance_loss(logits, indices)</span>
  <span class="magnet">    return out, aux_loss</span>
  <span class="magnet">    return out</span>
  <span class="magnet">    expert_out = self.experts[indices.argmax()](x)</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">def</span> <span class="fn">forward</span>(self, x):
    logits = self.<span class="fn">router</span>(x)
    weights, indices = logits.<span class="fn">topk</span>(self.k, dim=-<span class="num">1</span>)
    weights = weights.<span class="fn">softmax</span>(dim=-<span class="num">1</span>)
    out = torch.<span class="fn">zeros_like</span>(x)
    <span class="kw">for</span> e_idx <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(self.experts)):
        mask = (indices == e_idx).<span class="fn">any</span>(dim=-<span class="num">1</span>)
        <span class="kw">if</span> mask.<span class="fn">any</span>():
            expert_out = self.experts[e_idx](x[mask])
            out[mask] += <span class="fn">apply_weight</span>(expert_out, weights, indices, e_idx)
    aux_loss = self.<span class="fn">compute_load_balance_loss</span>(logits, indices)
    <span class="kw">return</span> out, aux_loss</code></pre>
<p>The traps:</p>
<ul>
  <li><code>weights = weights.sigmoid()</code>: sigmoid doesn't normalize across the K experts. Need softmax over the top-K so weights sum to 1 (preserves output magnitude).</li>
  <li><code>return out</code>: drops the auxiliary load-balancing loss. Without returning it, the auxiliary loss never gets added to the main training loss → routing collapses.</li>
  <li><code>expert_out = self.experts[indices.argmax()](x)</code>: picks the single best expert globally and runs the full batch through it — that's "single expert routing," not top-K MoE. Defeats the purpose.</li>
</ul>
<p>The full pattern: <strong>router → top-K + softmax → per-expert masked compute → weighted sum → return both output AND aux loss</strong>. The aux loss return is what most first-time MoE implementations forget; downstream training breaks because there's no force keeping the router balanced.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each MoE concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>Router</div>
  <div>A. Tiny linear layer + top-K + softmax — picks K experts per token.</div>

  <div>Expert</div>
  <div>B. One of N parallel MLPs; specializes during training; idle for most tokens.</div>

  <div>Top-K (typically K=2)</div>
  <div>C. Each token activates only K of N experts — sparse activation pattern.</div>

  <div>Auxiliary load-balancing loss</div>
  <div>D. Penalizes uneven token-to-expert assignment; mandatory to prevent router collapse.</div>

  <div>Expert parallelism (EP)</div>
  <div>E. Fifth parallelism axis; experts split across ranks; uses all-to-all collectives.</div>

  <div>All-to-all collective</div>
  <div>F. The communication primitive that shuffles tokens to expert ranks and back each layer.</div>

  <div>Capacity factor</div>
  <div>G. Hard cap on tokens per expert per batch — prevents extreme imbalance with overflow handling.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>Router</strong> → A<br>
<strong>Expert</strong> → B<br>
<strong>Top-K</strong> → C<br>
<strong>Auxiliary load-balancing loss</strong> → D<br>
<strong>Expert parallelism (EP)</strong> → E<br>
<strong>All-to-all collective</strong> → F<br>
<strong>Capacity factor</strong> → G
</p>
<p>The mental shortcut: <em>router picks, experts compute, top-K is sparse, aux loss balances, EP is the new axis, all-to-all shuffles tokens, capacity factor caps per-expert</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team trains a MoE model and observes that two of the eight experts get ~80% of all tokens after 1000 steps. What's likely going on, and what's the fix?</p>
<details class="answer"><summary>show answer</summary>
<p>The auxiliary load-balancing loss is missing or has too small a coefficient. The winner-take-all dynamic of routing — small initial preferences amplify into total dominance — has taken over. Fix: ensure the aux loss is added to the main loss with a meaningful coefficient (try 0.01-0.1; tune from there). Also verify <em>both</em> the f_i (assignment fractions) and P_i (router probabilities) terms are present — some implementations only do the P term, which is too soft. Verify with a metric: log per-expert load (number of tokens routed to each) at validation; histogram should be roughly uniform after a few hundred steps once aux loss is correctly applied.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Why is the all-to-all collective a "natural fit" for MoE — what makes it the right primitive vs all-reduce or all-gather?</p>
<details class="answer"><summary>show answer</summary>
<p>All-reduce is "everyone has data; sum and replicate to everyone." All-gather is "everyone has a slice; everyone gets the concatenation." Neither matches MoE's pattern.</p>
<p>MoE needs: "rank 0 has tokens routed to expert 1 → send them to rank 1; rank 0 also has tokens for expert 5 → send those to rank 5; and rank 0 receives from rank 2's tokens routed to rank 0's expert." Every rank sends a different subset of its data to every other rank, simultaneously. <em>That's literally the all-to-all primitive's definition</em> — each rank has a list of buffers (one per destination rank) and exchanges them with all destinations in one collective.</p>
<p>An all-reduce or all-gather would either send too much (every rank's data to everyone) or not match the destination structure. MoE was waiting for all-to-all all along; M15 was the foundation, M28 is the deployment.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> A team's MoE model trains well but inference is unexpectedly slow — about the same speed as a dense model with the same total parameters, not the same as a dense model with the active-per-token parameters. What's likely happening?</p>
<details class="answer"><summary>show answer</summary>
<p>The all-to-all overhead is dominating. In a small-batch inference setting, the all-to-all collective per MoE layer × number of layers can add substantial latency, especially across nodes. With low batch sizes, the matmul work per expert is small (each expert sees only a few tokens), so the constant communication overhead becomes a large fraction of the total time.</p>
<p>Diagnostics: profile (M13). Check if the trace is "comm-bound" — NCCL bars alternating with compute (M13's communication-bound shape). Fixes: (a) batch more requests together via continuous batching (M27) — more tokens per batch means more useful work per all-to-all. (b) Replicate experts across ranks instead of strict EP — eliminate the all-to-all but use more memory. (c) Consider whether your hardware topology suits your EP factor — cross-node EP is much slower than within-node.</p>
<p>This is why production MoE inference (vLLM, TensorRT-LLM) often uses small EP within a node + DP across nodes — keeps the all-to-all on the fastest fabric.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> If MoE gives "model quality of N×K active params at compute of K active params," why isn't every model MoE?</p>
<details class="answer"><summary>show answer</summary>
<p>Several real costs that MoE adds:</p>
<ol>
  <li><strong>Memory</strong>: the entire model has to fit on your hardware even though only K/N is active per token. Mixtral 8x7B needs 47B params worth of memory at inference time, not 13B. For memory-constrained settings (mobile, single-GPU consumer), this is prohibitive.</li>
  <li><strong>Engineering complexity</strong>: expert parallelism adds a new parallelism axis; routing requires careful load-balancing; all-to-all collectives add comm overhead; quantizing routes to balance experts is tricky. Frontier teams handle this; smaller teams find it daunting.</li>
  <li><strong>Specialized inference systems</strong>: not all serving frameworks support MoE well. Some do; some don't. Throws constraints on deployment.</li>
  <li><strong>Quality trade-offs</strong>: MoE gains per-active-FLOP are real, but per-total-param quality is sometimes worse than dense (you're effectively training fewer interactions per parameter). Distillation from MoE to dense for deployment is common.</li>
</ol>
<p>For frontier-scale capability targets where compute, not memory, is the binding constraint, MoE wins. For memory-constrained or simple-deployment settings, dense models still dominate. <em>The optimal choice depends on what's binding</em>.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li><strong>Mixture of Experts</strong>: replace one MLP per layer with N parallel MLPs ("experts") and a small router that picks top-K (typically K=2) per token. Compute scales with K; parameters with N.</li>
  <li><strong>Sparse activation pattern</strong>: each token only activates K/N of the experts. 8× the parameters at the same per-token compute as a dense model.</li>
  <li><strong>The router</strong>: a tiny linear layer (D → N) followed by top-K + softmax. ~0.05% of the layer's parameters. Outputs are routing decisions and weights.</li>
  <li><strong>Load-balancing auxiliary loss</strong> is mandatory: <code>aux_loss = N × sum(f_i × P_i)</code>. Without it, the router collapses to picking 1-2 experts winner-take-all.</li>
  <li><strong>Capacity factor</strong>: hard cap (~1.25 × tokens / N) on per-expert load per batch. Overflow handled with fallback or drop.</li>
  <li><strong>Expert parallelism (EP)</strong>: a fifth parallelism axis. Experts split across ranks; tokens get routed to their expert rank via <strong>all-to-all</strong> (M15 finally deployed for real).</li>
  <li><strong>5D parallelism</strong>: DP × PP × EP × TP × SP. Match each axis to its right network fabric (TP/EP within node on NVLink; PP/DP across nodes on InfiniBand).</li>
  <li><strong>Frontier MoE</strong>: Mixtral 8x7B (47B total / 13B active), DeepSeek-V3 (671B total / 37B active), GPT-4-class (rumored ~1.8T total / ~280B active).</li>
  <li><strong>Inference of MoE</strong>: compute scales with active K, but memory scales with total N — entire model must be resident. all-to-all overhead can dominate at low batches; EP within node + DP across nodes is the standard config.</li>
  <li>Modern frontier-scale training is the orchestrated combination of <em>everything</em> — mixed precision, FSDP, FlashAttention, kernels, quantization, paged serving, MoE — each layer adding 1.3-3× efficiency. The capstone showed this composition.</li>
</ul>
</div>

<div class="keyidea">
<strong>End of the core course.</strong> Twenty-eight modules. From <code>x.stride()</code> to <code>fully_shard</code> to <code>tl.dot</code> to MoE expert parallelism. You now have the full mental model — the framework to read any modern paper, debug any training failure, and design any inference system. <strong>The art is in the composition. The science is in the components. You've seen both.</strong> A short Part IX follows for synthesis modules: building a transformer end-to-end (M29), RoPE (M30), post-training (M31), debugging (M32), and Mamba (M33).
</div>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">28</span>
  <span>MoE &amp; frontier capstone — fin.</span>
</div>
"""

emit("28_moe_capstone", "Module 28 — MoE & frontier capstone", BODY)
