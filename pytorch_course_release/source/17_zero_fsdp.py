#!/usr/bin/env python3
"""Module 17: ZeRO / FSDP — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part VI · Module 17</div>
  <h1 class="module-title">ZeRO &amp; <em>FSDP</em></h1>
  <p class="module-sub">— sharding parameters, gradients, and optimizer state across ranks so a 70B model can train on hardware that couldn't hold its optimizer state, let alone its activations</p>
</div>

<p>DDP has one limit: <em>the model has to fit on a single GPU</em>. Add the master fp32 copy, gradients in fp32, and Adam's two moment buffers, and you're at 18 bytes per parameter (M12). For 7B parameters, that's 126 GB — more than an 80 GB H100. DDP is done.</p>

<p>The fix is conceptually simple: <strong>each rank stores only 1/N of every parameter and 1/N of every optimizer state</strong>. When forward needs a layer's full weight, all ranks gather their slices to materialize it; do the matmul; free the gathered copy. Same for backward. Gradients get reduce-scattered (not all-reduced) so each rank ends up with only its slice's worth of gradient to feed to its slice's worth of optimizer state. The math is the M15 identity: <code>all_reduce ≡ reduce_scatter + all_gather</code>, but now we exploit it for memory rather than just bandwidth.</p>

<div class="keyidea">
ZeRO has <strong>three stages</strong> of progressively aggressive sharding. <em>ZeRO-1</em>: shard optimizer state across ranks (Adam moments, master copy). <em>ZeRO-2</em>: also shard gradients. <em>ZeRO-3</em> (= FSDP): also shard parameters. Each stage saves more memory at the cost of more communication. ZeRO-3/FSDP is the modern default for large models. The implementation is one wrapper line; the conceptual cost is understanding that <strong>parameters are no longer always on the GPU when you need them — they get materialized per layer</strong>.
</div>

<h2>The new face</h2>

<div class="character" style="--c: #c1502e;">
  <div class="avatar" style="background: #c1502e; color: #fff;">⊟</div>
  <div>
    <p class="who">Shard</p>
    <p class="name">"I'm a slice of a parameter. My rank owns me; the others own theirs."</p>
    <p class="says">Before, you had one parameter per Module. Now there are N of me — one per rank — each holding 1/N of the original tensor's bytes. We sit quiet most of the time. When forward arrives at our layer, all ranks all-gather us to reconstitute the full parameter, just in time for the matmul. Then we're freed again. Backward does the same — gather, compute, free, then reduce-scatter the gradient so each rank gets only the slice it owns. <em>I am ephemeral as a whole; only my slices persist.</em></p>
  </div>
</div>

<h2>The memory math, made painful</h2>

<p>Let's revisit M12's accounting in the context of multi-GPU training. Per-rank memory under each scheme, for a model with N parameters in mixed-precision AdamW, on world_size W GPUs:</p>

<div class="table-wrap">
<table>
<caption>Per-rank memory in different schemes (mixed-precision AdamW)</caption>
<thead><tr><th></th><th>bf16 params</th><th>fp32 master</th><th>grads (fp32)</th><th>Adam (m,v fp32)</th><th>Total/rank</th></tr></thead>
<tbody>
<tr><td><strong>DDP</strong></td><td>2N</td><td>4N</td><td>4N</td><td>8N</td><td>18N</td></tr>
<tr><td><strong>ZeRO-1</strong> (shard optimizer)</td><td>2N</td><td>4N/W</td><td>4N</td><td>8N/W</td><td>6N + 12N/W</td></tr>
<tr><td><strong>ZeRO-2</strong> (+ shard grads)</td><td>2N</td><td>4N/W</td><td>4N/W</td><td>8N/W</td><td>2N + 16N/W</td></tr>
<tr><td><strong>ZeRO-3 / FSDP</strong></td><td>2N/W</td><td>4N/W</td><td>4N/W</td><td>8N/W</td><td>18N/W</td></tr>
</tbody>
</table>
</div>

<p>Read the rightmost column. DDP is <code>18N</code> per rank — the same as single-GPU. ZeRO-1 saves the largest single bucket (Adam). ZeRO-2 saves another big chunk (gradients). ZeRO-3 shards <em>everything</em>, giving you full <code>1/W</code> scaling on fixed-cost memory.</p>

<p>For a 7B model on 8 GPUs:</p>

<ul>
  <li>DDP: 126 GB/rank → won't fit on 80 GB H100</li>
  <li>ZeRO-1: <code>6×7e9 + 12×7e9/8 ≈ 52.5 GB/rank</code> → fits</li>
  <li>ZeRO-2: <code>2×7e9 + 16×7e9/8 ≈ 28 GB/rank</code> → comfortable</li>
  <li>ZeRO-3: <code>18×7e9/8 ≈ 15.75 GB/rank</code> → lots of room for activations</li>
</ul>

<p>For a 70B model on 64 GPUs, the only option is ZeRO-3 — the others still wouldn't fit. <strong>Modern training uses ZeRO-3 (FSDP) by default at scale</strong> because it's the only thing that scales to arbitrarily large models.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 320" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Per-rank fixed-cost memory: 7B model, world_size=8</text>

  <!-- Y-axis labeling (memory) -->
  <line x1="80"  y1="60"  x2="80"  y2="280" stroke="#6b5d4f" stroke-width="1"/>
  <line x1="80"  y1="280" x2="700" y2="280" stroke="#6b5d4f" stroke-width="1"/>
  <text x="40" y="170" font-size="11" fill="#6b5d4f" text-anchor="middle" transform="rotate(-90 40 170)">GB / rank</text>
  <line x1="78"  y1="80"  x2="82" y2="80"  stroke="#6b5d4f" stroke-width="1"/><text x="74" y="84" font-size="9" fill="#6b5d4f" text-anchor="end">130</text>
  <line x1="78"  y1="140" x2="82" y2="140" stroke="#6b5d4f" stroke-width="1"/><text x="74" y="144" font-size="9" fill="#6b5d4f" text-anchor="end">90</text>
  <line x1="78"  y1="200" x2="82" y2="200" stroke="#6b5d4f" stroke-width="1"/><text x="74" y="204" font-size="9" fill="#6b5d4f" text-anchor="end">45</text>
  <line x1="78"  y1="260" x2="82" y2="260" stroke="#6b5d4f" stroke-width="1"/><text x="74" y="264" font-size="9" fill="#6b5d4f" text-anchor="end">15</text>

  <!-- Bar 1: DDP (126 GB) -->
  <g transform="translate(110, 0)">
    <rect x="0" y="84" width="100" height="14" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/><text x="50" y="94" text-anchor="middle" font-size="9" fill="#1a1612">bf16 params (14)</text>
    <rect x="0" y="98" width="100" height="28" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/><text x="50" y="115" text-anchor="middle" font-size="9" fill="#1a1612">fp32 master (28)</text>
    <rect x="0" y="126" width="100" height="28" fill="#fff8a8" stroke="#c1502e" stroke-width="1"/><text x="50" y="143" text-anchor="middle" font-size="9" fill="#1a1612">grads fp32 (28)</text>
    <rect x="0" y="154" width="100" height="56" fill="#fcecec" stroke="#c1502e" stroke-width="1"/><text x="50" y="186" text-anchor="middle" font-size="10" fill="#1a1612">Adam (56)</text>
    <text x="50" y="295" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">DDP</text>
    <text x="50" y="308" text-anchor="middle" font-size="10" fill="#c1502e">126 GB ✗ won't fit</text>
  </g>

  <!-- Bar 2: ZeRO-1 (52.5 GB) -->
  <g transform="translate(260, 0)">
    <rect x="0" y="200" width="100" height="14" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/><text x="50" y="210" text-anchor="middle" font-size="9" fill="#1a1612">bf16 (14)</text>
    <rect x="0" y="214" width="100" height="3" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="0.5"/>
    <rect x="0" y="217" width="100" height="28" fill="#fff8a8" stroke="#c1502e" stroke-width="1"/><text x="50" y="234" text-anchor="middle" font-size="9" fill="#1a1612">grads fp32 (28)</text>
    <rect x="0" y="245" width="100" height="7" fill="#fcecec" stroke="#c1502e" stroke-width="0.5"/>
    <text x="50" y="265" text-anchor="middle" font-size="9" fill="#c1502e">↑shrunk 8x</text>
    <text x="50" y="295" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">ZeRO-1</text>
    <text x="50" y="308" text-anchor="middle" font-size="10" fill="#1f5f5b">52.5 GB ✓</text>
  </g>

  <!-- Bar 3: ZeRO-2 (28 GB) -->
  <g transform="translate(410, 0)">
    <rect x="0" y="240" width="100" height="14" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1"/><text x="50" y="250" text-anchor="middle" font-size="9" fill="#1a1612">bf16 (14)</text>
    <rect x="0" y="254" width="100" height="3" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="0.5"/>
    <rect x="0" y="257" width="100" height="3" fill="#fff8a8" stroke="#c1502e" stroke-width="0.5"/>
    <rect x="0" y="260" width="100" height="7" fill="#fcecec" stroke="#c1502e" stroke-width="0.5"/>
    <text x="50" y="278" text-anchor="middle" font-size="9" fill="#c1502e">all sharded except bf16</text>
    <text x="50" y="295" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">ZeRO-2</text>
    <text x="50" y="308" text-anchor="middle" font-size="10" fill="#1f5f5b">28 GB ✓✓</text>
  </g>

  <!-- Bar 4: ZeRO-3 / FSDP (15.75 GB) -->
  <g transform="translate(560, 0)">
    <rect x="0" y="260" width="100" height="2" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="0.5"/>
    <rect x="0" y="262" width="100" height="3" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="0.5"/>
    <rect x="0" y="265" width="100" height="3" fill="#fff8a8" stroke="#c1502e" stroke-width="0.5"/>
    <rect x="0" y="268" width="100" height="7" fill="#fcecec" stroke="#c1502e" stroke-width="0.5"/>
    <text x="50" y="290" text-anchor="middle" font-size="9" fill="#1f5f5b">everything 1/W</text>
    <text x="50" y="295" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">ZeRO-3</text>
    <text x="50" y="308" text-anchor="middle" font-size="10" fill="#1f5f5b">15.75 GB ✓✓✓</text>
  </g>

  <!-- 80GB line -->
  <line x1="80" y1="115" x2="700" y2="115" stroke="#1a1612" stroke-width="1" stroke-dasharray="4 3"/>
  <text x="700" y="111" font-size="10" fill="#1a1612" text-anchor="end">80 GB H100</text>
</svg>
</div>

<p>The 80 GB line shows the H100 limit. DDP overshoots; the rest fit progressively more comfortably, leaving room for activations. <strong>Activations still scale with batch size and sequence length</strong> — they're not affected by sharding (each rank still computes its full activations for its data). So the savings shown above only solve the <em>fixed-cost</em> memory problem; you still need activation checkpointing (M12) and FlashAttention (M25) for the variable-cost part.</p>

<h2>How ZeRO-3 / FSDP works (the dance)</h2>

<p>The key insight: <em>you don't need the parameter materialized everywhere all the time. You only need it during the layer that uses it.</em></p>

<p>So FSDP wraps the model into a tree of FSDP units (typically each transformer layer is one unit). At init, each unit's parameters are sharded — split across W ranks, each rank holds 1/W. The flow during forward pass through a unit:</p>

<ol>
  <li><strong>All-gather</strong> the unit's parameter shards into the full parameter on every rank.</li>
  <li><strong>Forward</strong> through the unit — it now has the full parameter and runs normally.</li>
  <li><strong>Free</strong> the gathered copy on every rank. The shards persist; the materialized full weight is gone.</li>
</ol>

<p>The flow during backward is the symmetric mirror — re-gather, compute the gradient, reduce-scatter the gradient (so each rank gets only its slice's worth), free.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 340" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrFS" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">FSDP forward/backward through one unit (e.g., one transformer layer)</text>

  <!-- Forward pass row -->
  <text x="20" y="55" font-size="12" font-weight="700" fill="#1f5f5b">Forward through layer L:</text>
  <g transform="translate(20, 65)">
    <!-- Step 1: shards -->
    <rect x="0"   y="0" width="80" height="40" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="40" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">shards</text>
    <text x="40" y="34" text-anchor="middle" font-size="9" fill="#1a1612">each rank: 1/W</text>

    <!-- Arrow + all-gather -->
    <path d="M 85 20 L 105 20" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrFS)"/>
    <text x="135" y="14" font-size="9" fill="#1a1612" text-anchor="middle">all_gather</text>

    <!-- Step 2: full param -->
    <rect x="170" y="0" width="120" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="230" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">full param</text>
    <text x="230" y="34" text-anchor="middle" font-size="9" fill="#1a1612">briefly on every rank</text>

    <!-- Arrow + forward -->
    <path d="M 295 20 L 315 20" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrFS)"/>
    <text x="345" y="14" font-size="9" fill="#1a1612" text-anchor="middle">forward()</text>

    <!-- Step 3: output -->
    <rect x="380" y="0" width="100" height="40" fill="#fff8a8" stroke="#c1502e" stroke-width="2"/>
    <text x="430" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">activations</text>
    <text x="430" y="34" text-anchor="middle" font-size="9" fill="#1a1612">to next layer</text>

    <!-- Arrow + free -->
    <path d="M 485 20 L 505 20" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrFS)"/>
    <text x="535" y="14" font-size="9" fill="#1a1612" text-anchor="middle">free</text>

    <!-- Step 4: shards again -->
    <rect x="570" y="0" width="80" height="40" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="610" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">shards</text>
    <text x="610" y="34" text-anchor="middle" font-size="9" fill="#1a1612">full param freed</text>
  </g>

  <!-- Backward pass row -->
  <text x="20" y="155" font-size="12" font-weight="700" fill="#c1502e">Backward through layer L:</text>
  <g transform="translate(20, 165)">
    <!-- Step 1: shards -->
    <rect x="0"   y="0" width="80" height="40" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="40" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">shards</text>
    <text x="40" y="34" text-anchor="middle" font-size="9" fill="#1a1612">re-gather needed</text>

    <!-- Arrow + all-gather -->
    <path d="M 85 20 L 105 20" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrFS)"/>
    <text x="135" y="14" font-size="9" fill="#1a1612" text-anchor="middle">all_gather</text>

    <!-- Step 2: full param + grad output -->
    <rect x="170" y="0" width="120" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="230" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">full param</text>
    <text x="230" y="34" text-anchor="middle" font-size="9" fill="#1a1612">+ ∂L/∂out</text>

    <!-- Arrow + backward -->
    <path d="M 295 20 L 315 20" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrFS)"/>
    <text x="345" y="14" font-size="9" fill="#1a1612" text-anchor="middle">backward()</text>

    <!-- Step 3: full grad -->
    <rect x="380" y="0" width="100" height="40" fill="#ffd5dc" stroke="#b85a6c" stroke-width="2"/>
    <text x="430" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">full grad</text>
    <text x="430" y="34" text-anchor="middle" font-size="9" fill="#1a1612">on every rank</text>

    <!-- Arrow + reduce_scatter -->
    <path d="M 485 20 L 505 20" stroke="#b85a6c" stroke-width="1.5" fill="none" marker-end="url(#arrFS)"/>
    <text x="535" y="14" font-size="9" fill="#b85a6c" text-anchor="middle">reduce_scatter</text>

    <!-- Step 4: grad shard -->
    <rect x="570" y="0" width="80" height="40" fill="#ffd5dc" stroke="#b85a6c" stroke-width="2"/>
    <text x="610" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">grad shard</text>
    <text x="610" y="34" text-anchor="middle" font-size="9" fill="#1a1612">1/W on each rank</text>
  </g>

  <!-- Annotations -->
  <text x="370" y="250" font-family="'Caveat', cursive" font-size="20" fill="#1a1612" text-anchor="middle">forward all-gather → compute → free → next layer</text>
  <text x="370" y="275" font-family="'Caveat', cursive" font-size="20" fill="#1a1612" text-anchor="middle">backward re-gather → compute grad → reduce-scatter → free</text>
  <text x="370" y="305" font-family="'Caveat', cursive" font-size="22" fill="#c1502e" text-anchor="middle">double the all-gather cost, but each rank stores only 1/W of everything</text>
</svg>
</div>

<p>Two things to internalize:</p>

<ol>
  <li><strong>Each layer's full parameters exist on every rank only briefly</strong> — for the duration of that layer's forward, then they're freed. Same in backward.</li>
  <li><strong>The all-gather happens twice per layer per step</strong> (once in forward, once in backward). FSDP's communication cost is roughly <em>1.5×</em> DDP's (DDP does a single all-reduce at backward end, FSDP does two all-gathers + a reduce-scatter per layer).</li>
</ol>

<p>The 1.5× communication overhead is the price of memory savings. On well-connected hardware (NVLink within node, fast InfiniBand between), this overhead is mostly hidden behind compute via the same prefetching trick DDP uses. On bandwidth-limited setups, you'll see real overhead.</p>

<h2>The PyTorch FSDP API</h2>

<p>PyTorch has two FSDP APIs:</p>

<ul>
  <li><strong>FSDP1</strong>: <code>torch.distributed.fsdp.FullyShardedDataParallel</code> — the original, mature, what you'll see in most existing code.</li>
  <li><strong>FSDP2</strong>: <code>torch.distributed.fsdp.fully_shard</code> — the new per-parameter API, more flexible, integrates better with <code>torch.compile</code>. Modern recommended path.</li>
</ul>

<p>Both share the same conceptual model. We'll show FSDP1 for clarity (most production code today), with notes on FSDP2.</p>

<h3>FSDP1 minimal setup</h3>

<pre><code><span class="kw">from</span> torch.distributed.fsdp <span class="kw">import</span> FullyShardedDataParallel <span class="kw">as</span> FSDP
<span class="kw">from</span> torch.distributed.fsdp <span class="kw">import</span> ShardingStrategy, MixedPrecision
<span class="kw">from</span> torch.distributed.fsdp.wrap <span class="kw">import</span> transformer_auto_wrap_policy
<span class="kw">import</span> functools

<span class="com"># Decide which submodules to shard. For transformers: each block is its own FSDP unit.</span>
auto_wrap_policy = functools.<span class="fn">partial</span>(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},     <span class="com"># your block class</span>
)

mixed_precision = <span class="fn">MixedPrecision</span>(
    param_dtype=torch.bfloat16,                   <span class="com"># param all-gathers in bf16</span>
    reduce_dtype=torch.float32,                   <span class="com"># grad reduce-scatter in fp32</span>
    buffer_dtype=torch.bfloat16,
)

model = <span class="fn">build_model</span>().<span class="fn">to</span>(device)
model = <span class="fn">FSDP</span>(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,   <span class="com"># ZeRO-3</span>
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mixed_precision,
    device_id=local_rank,
)</code></pre>

<p>Three knobs that matter:</p>

<ol>
  <li><strong><code>sharding_strategy</code></strong>: <code>FULL_SHARD</code> (ZeRO-3, default), <code>SHARD_GRAD_OP</code> (ZeRO-2), <code>NO_SHARD</code> (DDP-equivalent), <code>HYBRID_SHARD</code> (shard within node, replicate across — see below).</li>
  <li><strong><code>auto_wrap_policy</code></strong>: which submodules to treat as sharding units. For transformers, each block is one unit. Smaller units = more communication; larger units = more transient memory.</li>
  <li><strong><code>mixed_precision</code></strong>: per-stage dtype control. The standard recipe is bf16 for params and forward activations, fp32 for the gradient reduce-scatter (precision matters for Adam's <code>v</code>).</li>
</ol>

<h3>FSDP2 (the future)</h3>

<pre><code><span class="kw">from</span> torch.distributed.fsdp <span class="kw">import</span> fully_shard, MixedPrecisionPolicy

mp_policy = <span class="fn">MixedPrecisionPolicy</span>(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

<span class="com"># FSDP2 wraps per-module, applied recursively</span>
<span class="kw">for</span> block <span class="kw">in</span> model.transformer.layers:
    <span class="fn">fully_shard</span>(block, mp_policy=mp_policy)
<span class="fn">fully_shard</span>(model, mp_policy=mp_policy)</code></pre>

<p>FSDP2 uses <code>DTensor</code> under the hood for cleaner integration with tensor parallelism (M18) and <code>torch.compile</code> (M21). The model on disk is a normal <code>state_dict</code> — no per-rank sharded checkpoints to wrangle. For new code, FSDP2 is the recommended path; existing FSDP1 code will keep working.</p>

<h2>Combining with activation checkpointing</h2>

<p>FSDP solves the <em>fixed-cost</em> memory problem. Activations are still per-rank (each rank computes its own data shard's activations), so they still scale with batch × seq × depth. For really big models you combine FSDP with activation checkpointing (M12):</p>

<pre><code><span class="kw">from</span> torch.distributed.algorithms._checkpoint.checkpoint_wrapper <span class="kw">import</span> (
    checkpoint_wrapper, CheckpointImpl,
)

<span class="kw">def</span> <span class="fn">apply_activation_checkpointing</span>(model, target_cls):
    <span class="kw">def</span> <span class="fn">check_fn</span>(submodule):
        <span class="kw">return</span> <span class="fn">isinstance</span>(submodule, target_cls)
    torch.distributed.algorithms._checkpoint.checkpoint_wrapper.<span class="fn">apply_activation_checkpointing</span>(
        model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn,
    )

<span class="com"># Apply to each transformer block</span>
<span class="fn">apply_activation_checkpointing</span>(model, target_cls=TransformerBlock)</code></pre>

<p>The order matters: <strong>apply activation checkpointing <em>before</em> wrapping with FSDP</strong>. Checkpoint inside the FSDP unit means the recompute happens after the all-gather, so the param stays gathered for the recompute pass — no extra communication.</p>

<p>This combination — FSDP + activation checkpointing + FlashAttention (M25) — is what makes 70B-scale model training possible on commodity (8× H100) clusters.</p>

<h2>Hybrid Sharded Data Parallel (HSDP)</h2>

<p>The bandwidth math: all-gathers and reduce-scatters across many ranks are expensive. If you have 64 GPUs spread across 8 nodes, all-gathering a parameter across all 64 means crossing 7 inter-node hops. Painful.</p>

<p>HSDP exploits the topology: <em>shard within node, replicate across nodes</em>. Within a node (8 GPUs on NVLink), you do FSDP — fast intra-node all-gathers. Across nodes (slow ethernet/InfiniBand), you do DDP — one all-reduce per step.</p>

<pre><code>model = <span class="fn">FSDP</span>(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    ...
)</code></pre>

<p>The tradeoff: each node holds a full copy of the model (8× more fixed-cost memory than full FSDP), but inter-node communication is just one DDP all-reduce instead of many FSDP gathers. For models that fit on a single node, HSDP is often the fastest option. For models that don't fit, you're back to FULL_SHARD.</p>

<h2>CPU offload (for the truly desperate)</h2>

<p>If even ZeRO-3 doesn't fit, you can offload optimizer state to CPU memory:</p>

<pre><code><span class="kw">from</span> torch.distributed.fsdp <span class="kw">import</span> CPUOffload

model = <span class="fn">FSDP</span>(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=<span class="fn">CPUOffload</span>(offload_params=<span class="kw">True</span>),
    ...
)</code></pre>

<p>The optimizer state and parameter shards live in CPU RAM; they get DMAed to GPU during the all-gather. This drastically slows training (PCIe bandwidth, not NVLink) but lets you train models that wouldn't otherwise fit anywhere. Use as a last resort.</p>

<h2>Checkpointing FSDP models</h2>

<p>FSDP1's checkpoint story has historically been the worst part of the library. Each rank has only its slice of each parameter, so the obvious approach (each rank saves its own state_dict) gives you W files that are useless without all of them. Loading on a different world size is broken.</p>

<p>The modern fix uses <code>distributed_checkpoint</code> (DCP) which produces sharded files that can be reloaded on any world size:</p>

<pre><code><span class="kw">import</span> torch.distributed.checkpoint <span class="kw">as</span> dcp

<span class="com"># Save</span>
state_dict = {<span class="str">"model"</span>: model.<span class="fn">state_dict</span>(), <span class="str">"optimizer"</span>: optimizer.<span class="fn">state_dict</span>()}
dcp.<span class="fn">save</span>(state_dict, checkpoint_id=<span class="fn">f"ckpt_step{step}"</span>)

<span class="com"># Load — works on any world size</span>
state_dict = {<span class="str">"model"</span>: model.<span class="fn">state_dict</span>(), <span class="str">"optimizer"</span>: optimizer.<span class="fn">state_dict</span>()}
dcp.<span class="fn">load</span>(state_dict, checkpoint_id=<span class="fn">f"ckpt_step{step}"</span>)
<span class="com"># model and optimizer are now restored, with each rank having its own slice</span></code></pre>

<p>For <em>final</em> checkpoints (e.g., for inference or deployment), you usually want a full unsharded state_dict — DCP can dump that too via <code>torch.distributed.checkpoint.format_utils.dcp_to_torch_save</code>, or for FSDP1 you can use the older <code>FullStateDictConfig</code> + <code>StateDictType.FULL_STATE_DICT</code> patterns (look it up when needed).</p>

<div class="ndq">
<h4>About FSDP and ZeRO</h4>

<p class="q">When should I use ZeRO-1 instead of ZeRO-3?</p>
<p class="a">When (a) you have memory headroom and (b) you want to minimize communication. ZeRO-1 only does the optimizer-state shard; the rest of training proceeds like DDP. Less communication overhead than ZeRO-3, but only saves ~67% of fixed-cost memory (the Adam buckets) instead of close to 100%. If your model fits with ZeRO-1, you might not need ZeRO-3.</p>

<p class="q">Why is FSDP slower per step than DDP at the same world size?</p>
<p class="a">FSDP does ~1.5× the communication of DDP (two all-gathers + a reduce-scatter per layer, vs DDP's single all-reduce). On a slow network, that overhead can be 10-30% of step time. The savings are in <em>memory</em>, not throughput. The argument for FSDP is "this model didn't fit at all under DDP." If your model fits comfortably under DDP, use DDP.</p>

<p class="q">Does FSDP work with torch.compile?</p>
<p class="a">FSDP1 has rough edges with compile. FSDP2 is designed for it — <code>fully_shard</code> + <code>torch.compile</code> work well together. For new projects requiring compile + sharding, use FSDP2.</p>

<p class="q">My loss is different in FSDP vs DDP for the same model and data. Is that a bug?</p>
<p class="a">Probably not — small differences (≤1% in loss) are expected from numerical differences in the reduce-scatter vs all-reduce paths and from different mixed-precision casting points. The training trajectories should converge to similar quality. If the difference is large or the loss diverges, suspect a bug in your wrapping (e.g., layers that should be FSDP-wrapped but aren't, or weight tying not handled correctly across shards).</p>

<p class="q">What about weight tying with FSDP?</p>
<p class="a">Tied weights (e.g., embedding == output projection from M7) are tricky under FSDP because FSDP shards each parameter independently — the "tying" can break. Modern FSDP detects tied weights and handles them, but you must do the tying <em>before</em> wrapping. Always tie, then wrap. Test that <code>id(model.embed.weight) == id(model.head.weight)</code> after FSDP wrapping; if not, retie or refactor.</p>
</div>

<h2>Code Magnets: build a FSDP setup for a transformer</h2>

<p>You're setting up FSDP for a transformer with mixed precision, auto-wrapping each block, with activation checkpointing. Three magnets are wrong choices.</p>

<div class="magnets">
<p>Arrange the magnets into a working setup.</p>

<div class="magnet-pool">
  <span class="magnet">model = build_model().to(device)</span>
  <span class="magnet">apply_activation_checkpointing(model, target_cls=TransformerBlock)</span>
  <span class="magnet">model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,</span>
  <span class="magnet">    auto_wrap_policy=transformer_wrap_policy,</span>
  <span class="magnet">    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,</span>
  <span class="magnet">                                    reduce_dtype=torch.float32),</span>
  <span class="magnet">    device_id=local_rank)</span>
  <span class="magnet">model = DDP(model, device_ids=[local_rank])</span>
  <span class="magnet">opt = torch.optim.AdamW(model.parameters(), lr=3e-4)</span>
  <span class="magnet">model = FSDP(model); apply_activation_checkpointing(model, target_cls=TransformerBlock)</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code>model = <span class="fn">build_model</span>().<span class="fn">to</span>(device)
<span class="fn">apply_activation_checkpointing</span>(model, target_cls=TransformerBlock)
model = <span class="fn">FSDP</span>(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=transformer_wrap_policy,
    mixed_precision=<span class="fn">MixedPrecision</span>(param_dtype=torch.bfloat16,
                                    reduce_dtype=torch.float32),
    device_id=local_rank)
opt = torch.optim.<span class="fn">AdamW</span>(model.<span class="fn">parameters</span>(), lr=<span class="num">3e-4</span>)</code></pre>
<p>The traps:</p>
<ul>
  <li><code>model = DDP(model, device_ids=[local_rank])</code>: that's the DDP wrapper, not FSDP. Stacking on top of FSDP would error.</li>
  <li><code>model = FSDP(model); apply_activation_checkpointing(model, ...)</code>: <strong>order matters</strong>. Activation checkpointing must be applied <em>before</em> FSDP wrapping. Otherwise the checkpoint boundary is around the FSDP unit, not the inner transformer block, and the recompute happens after each layer's gather+free — defeating the purpose.</li>
</ul>
<p>The right order: <strong>build → activation checkpointing → FSDP wrap → optimizer</strong>. Optimizer must be created <em>after</em> FSDP because FSDP modifies the parameter list (each rank only sees its shards).</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each FSDP/ZeRO concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>ZeRO-1</div>
  <div>A. Same as DDP for params/grads, but optimizer state sharded across ranks.</div>

  <div>ZeRO-3 / FSDP</div>
  <div>B. Everything sharded — params, grads, optimizer state — each rank holds 1/W.</div>

  <div>FSDP unit</div>
  <div>C. A submodule that is the granularity of all-gather/free; typically one transformer block.</div>

  <div>HYBRID_SHARD</div>
  <div>D. Shard within node, replicate across — exploits NVLink vs network bandwidth gap.</div>

  <div>reduce-scatter (in FSDP)</div>
  <div>E. The collective that produces only-my-slice gradients, replacing DDP's all-reduce.</div>

  <div>Activation checkpointing before FSDP wrap</div>
  <div>F. Ensures recompute happens after the all-gather, so it doesn't trigger another one.</div>

  <div>Distributed Checkpoint (DCP)</div>
  <div>G. Saves and loads sharded state_dicts that can be re-sharded on any world size.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>ZeRO-1</strong> → A<br>
<strong>ZeRO-3 / FSDP</strong> → B<br>
<strong>FSDP unit</strong> → C<br>
<strong>HYBRID_SHARD</strong> → D<br>
<strong>reduce-scatter (in FSDP)</strong> → E<br>
<strong>Activation checkpointing before FSDP wrap</strong> → F<br>
<strong>Distributed Checkpoint (DCP)</strong> → G
</p>
<p>The mental shortcut: <em>ZeRO-1 shards optimizer, ZeRO-3 shards everything, FSDP unit is the gather granularity, HYBRID_SHARD is intra-node FSDP + inter-node DDP, reduce-scatter replaces all-reduce in FSDP, AC before wrap = recompute inside the gather, DCP for resharding</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team has a 13B transformer, 16 GPUs (2× 8-GPU nodes), and a fast NVLink network within each node but slow inter-node ethernet. Which sharding strategy should they pick?</p>
<details class="answer"><summary>show answer</summary>
<p>HYBRID_SHARD. The model fits within a node under FSDP (per-rank fixed cost = 18×13B/8 ≈ 30 GB, comfortable on 80 GB H100 with room for activations). HYBRID_SHARD will give them fast intra-node all-gathers (NVLink) for the heavy FSDP communication, and a single inter-node DDP all-reduce per step (one slow collective rather than many). FULL_SHARD across all 16 GPUs would force every all-gather across the slow inter-node link — much worse. The choice is a topology decision: pick the sharding boundary at the bandwidth boundary.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Why is FSDP forward typically faster than FSDP backward for the same layer?</p>
<details class="answer"><summary>show answer</summary>
<p>Forward needs one all-gather per layer (gather params, compute, free). Backward needs an all-gather <em>and</em> a reduce-scatter — first to gather params for the backward pass through the layer, then to reduce-scatter the gradient down to per-rank shards. So backward has roughly 1.5-2× the communication of forward per layer. (This is also why activation checkpointing has interesting implications for FSDP: the recompute during backward triggers another forward all-gather. It's why "apply AC before FSDP wrap" matters — done right, the recompute uses the same gathered params as the original backward, costing nothing extra.)</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> Compute the per-rank fixed-cost memory for a 70B-parameter model in mixed-precision AdamW under DDP, ZeRO-1, ZeRO-2, and ZeRO-3, on 64 GPUs.</p>
<details class="answer"><summary>show answer</summary>
<p>From the formula: <code>18N</code> (DDP), <code>6N + 12N/W</code> (ZeRO-1), <code>2N + 16N/W</code> (ZeRO-2), <code>18N/W</code> (ZeRO-3).</p>
<p>For N=70B, W=64:</p>
<ul>
  <li>DDP: 1260 GB ✗ (massively over)</li>
  <li>ZeRO-1: <code>6×70 + 12×70/64 = 420 + 13.1 = 433 GB</code> ✗</li>
  <li>ZeRO-2: <code>2×70 + 16×70/64 = 140 + 17.5 = 157.5 GB</code> ✗</li>
  <li>ZeRO-3: <code>18×70/64 = 19.7 GB</code> ✓</li>
</ul>
<p>For 70B at this world size, ZeRO-3 is the only option that fits. Plus activations on top, so realistically you also want activation checkpointing and FlashAttention. This is why the modern recipe at scale is "FSDP + activation checkpointing + FlashAttention + bf16."</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> A team's FSDP run trains correctly but their checkpoints can't be loaded on a different world size — they trained on 8 GPUs and want to resume on 16. What's the issue and the modern fix?</p>
<details class="answer"><summary>show answer</summary>
<p>FSDP1's default state_dict is rank-local (each rank has its slice). Saving each rank's state_dict gives you W files that can only be reloaded on the same W. Modern fix: use <code>torch.distributed.checkpoint</code> (DCP). DCP saves a directory of sharded files with metadata; on load, it figures out the new world size's sharding and remaps. This makes resharding trivial — train on 8, resume on 16, fine-tune on 4, all without manual conversion. For older FSDP1 code, you can also force <code>FullStateDictConfig</code> at save time to gather everything onto rank 0, but it's slow at scale and has memory implications. DCP is the right answer for new code.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li><strong>DDP runs out</strong> when the 18N bytes of fixed-cost training state don't fit per rank. ZeRO/FSDP shards that state across ranks.</li>
  <li><strong>ZeRO-1</strong>: shard optimizer state (Adam moments + master copy). Saves the biggest single bucket. ~67% of fixed-cost savings.</li>
  <li><strong>ZeRO-2</strong>: also shard gradients. Now only the bf16 working params are replicated.</li>
  <li><strong>ZeRO-3 / FSDP</strong>: shard everything. Each rank holds 1/W of all training state. Modern default for large models.</li>
  <li><strong>The dance</strong>: forward all-gathers each layer's params, computes, frees. Backward re-gathers, computes grads, reduce-scatters to per-rank slices. ~1.5× DDP's communication, much less memory.</li>
  <li><strong>FSDP unit</strong> is the granularity of gather/free. For transformers, one block per unit. Smaller units = more communication; larger = more transient memory at peak.</li>
  <li><strong>Mixed precision policy</strong>: param all-gathers in bf16 (smaller messages), grad reduce-scatter in fp32 (precision matters for Adam).</li>
  <li><strong>Combine with activation checkpointing</strong> for the variable-cost (activation) memory. Apply AC <em>before</em> FSDP wrapping so the recompute happens inside the gathered window.</li>
  <li><strong>HYBRID_SHARD</strong>: shard within node (fast NVLink), DDP across nodes (slow ethernet). Best when the model fits on one node and inter-node bandwidth is the bottleneck.</li>
  <li><strong>CPU offload</strong>: last-resort move of optimizer state to CPU. Lets you train models that wouldn't fit anywhere else, at significant throughput cost.</li>
  <li><strong>Distributed Checkpoint (DCP)</strong> is the modern way to save/load FSDP state. Sharded files, world-size agnostic, the right answer for new code.</li>
  <li><strong>FSDP1 vs FSDP2</strong>: FSDP1 mature, what most code uses. FSDP2 (<code>fully_shard</code>) cleaner, designed for <code>torch.compile</code>, recommended for new projects.</li>
  <li>The reflex: pick the smallest sharding that fits. DDP if it fits. ZeRO-1 if you have headroom. FSDP if you need to scale up. HYBRID_SHARD if you're spread across nodes with a fast intra-node fabric.</li>
</ul>
</div>

<p>Module 18 closes Part VI with the kinds of parallelism that aren't replication of work — <em>tensor parallelism</em> (split a layer's matmul across ranks) and <em>pipeline parallelism</em> (split the layers themselves into stages, each on different ranks, with micro-batches flowing through). For the largest models, you stack 2D, 3D, even 4D parallelism — data + tensor + pipeline + sequence. We'll see why and how.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">17</span>
  <span>ZeRO &amp; FSDP</span>
</div>
"""

emit("17_zero_fsdp", "Module 17 — ZeRO & FSDP", BODY)
