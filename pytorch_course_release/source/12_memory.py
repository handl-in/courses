#!/usr/bin/env python3
"""Module 12: Memory: where does it all go? — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part V · Module 12</div>
  <h1 class="module-title">Memory: <em>where does it all go?</em></h1>
  <p class="module-sub">— the five buckets of GPU memory, the transformer memory formula that lets you predict OOM before you hit it, and the recompute trick that buys you depth at the cost of compute</p>
</div>

<p>You've trained models. You've also gotten <code>CUDA out of memory</code> at step 1247 of a long run. Welcome to Part V, where we stop pretending memory is infinite.</p>

<p>The good news: GPU memory consumption during training is almost completely predictable. Five buckets account for ≥99% of it: <strong>parameters, gradients, optimizer state, activations, and workspace</strong>. If you can estimate each bucket from your config, you can predict OOM in advance, choose batch size and depth deliberately, and know exactly which lever to pull when something doesn't fit. The bad news: PyTorch's caching allocator hides some of this from <code>nvidia-smi</code>, which is why "but I have free memory!" doesn't always mean what you think.</p>

<div class="keyidea">
Training memory = <strong>params + grads + optimizer state + activations + workspace</strong>. The first three scale with <em>model size</em> (and stay constant during training). The fourth scales with <em>batch size × sequence length × depth</em> and dominates at scale. Activation checkpointing trades compute for memory by recomputing activations in the backward pass instead of storing them. Once you internalize the five buckets, you can predict OOM from a config sheet without running a single step.
</div>

<h2>One new face</h2>

<div class="character" style="--c: #d4a017;">
  <div class="avatar" style="background: #d4a017;">A</div>
  <div>
    <p class="who">Activation</p>
    <p class="name">"I'm what every layer leaves behind for backward to find."</p>
    <p class="says">When you do <code>y = layer(x)</code> with autograd on, I'm what gets saved so backward can compute the gradient w.r.t. <code>x</code>. There's one of me per layer per forward pass, and my size is <strong>proportional to the batch size, sequence length, and hidden dim</strong>. For deep models I dominate memory. The recompute trick (M6) and gradient checkpointing both target me — they delete me after forward and recompute me during backward. The framework treats me as ephemeral, but at training time I'm often the biggest bucket.</p>
  </div>
</div>

<h2>The five buckets</h2>

<p>Let's enumerate everything that lives in GPU memory during a training step.</p>

<div class="table-wrap">
<table>
<caption>Where every GPU byte goes during training</caption>
<thead><tr><th>Bucket</th><th>Lifetime</th><th>Scales with</th><th>Typical size for a 1B-param transformer in mixed precision</th></tr></thead>
<tbody>
<tr><td><strong>Parameters</strong></td><td>Whole training run</td><td>Param count × 2 (bf16)</td><td>~2 GB</td></tr>
<tr><td><strong>Master params</strong> (mixed precision)</td><td>Whole training run</td><td>Param count × 4 (fp32)</td><td>~4 GB</td></tr>
<tr><td><strong>Gradients</strong></td><td>One step (until next zero_grad)</td><td>Param count × 4 (fp32 typically)</td><td>~4 GB</td></tr>
<tr><td><strong>Optimizer state</strong></td><td>Whole training run</td><td>Adam: param count × 8 (m + v in fp32)</td><td>~8 GB</td></tr>
<tr><td><strong>Activations</strong></td><td>Forward → released during backward</td><td>Batch × seq × hidden × layers</td><td>varies massively — see below</td></tr>
<tr><td><strong>Workspace</strong></td><td>Per-op transient</td><td>cuDNN/cuBLAS scratch buffers</td><td>~few hundred MB</td></tr>
</tbody>
</table>
</div>

<p>Look at that 1B-param column. <strong>~18 GB before activations</strong>. That's just to <em>hold the model and optimizer</em>. Add 1B parameters' worth of activations on top — easily 20+ GB more for a typical transformer at 4K sequence length and modest batch size — and you're at 40 GB. That's a single A100 used up at 1B scale. This is why people obsess over the activation bucket and why FSDP/ZeRO exist (Module 17).</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 280" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">A 1B-param transformer in mixed-precision Adam, batch=16 × seq=4096</text>

  <!-- Stacked bar showing 5 buckets to scale -->
  <g transform="translate(60, 60)">
    <!-- Total bar height represents ~50 GB -->
    <!-- Bucket sizes (in GB approximately): bf16 params 2, fp32 master 4, grads 4, adam 8, activations 30 -->

    <rect x="0" y="0" width="600" height="20" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="300" y="14" text-anchor="middle" font-size="11" fill="#1a1612" font-weight="700">bf16 params (~2 GB)</text>

    <rect x="0" y="20" width="600" height="35" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="300" y="42" text-anchor="middle" font-size="11" fill="#1a1612" font-weight="700">fp32 master copy (~4 GB)</text>

    <rect x="0" y="55" width="600" height="35" fill="#fff8a8" stroke="#c1502e" stroke-width="1.5"/>
    <text x="300" y="77" text-anchor="middle" font-size="11" fill="#1a1612" font-weight="700">gradients (~4 GB)</text>

    <rect x="0" y="90" width="600" height="60" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="300" y="124" text-anchor="middle" font-size="12" fill="#1a1612" font-weight="700">Adam optimizer state — m + v in fp32 (~8 GB)</text>

    <rect x="0" y="150" width="600" height="120" fill="#ffd5dc" stroke="#b85a6c" stroke-width="2"/>
    <text x="300" y="200" text-anchor="middle" font-size="14" fill="#1a1612" font-weight="700">activations (~30 GB at this batch×seq×depth)</text>
    <text x="300" y="230" text-anchor="middle" font-size="11" fill="#1a1612" font-style="italic">— THE bucket you actually have control over</text>
  </g>

  <text x="370" y="252" font-family="'Caveat', cursive" font-size="20" fill="#c1502e" text-anchor="middle">"why am I OOM" answers from this picture 90% of the time</text>
</svg>
</div>

<h2>Bucket 1-3: parameters, gradients, optimizer state</h2>

<p>These three are easy. They scale only with parameter count, not with batch size or sequence length. They're constant for the whole training run.</p>

<h3>Parameters</h3>

<p>For a model with <code>N</code> parameters in fp32, that's <code>4N</code> bytes. In bf16/fp16, <code>2N</code> bytes. With mixed precision (M14), you typically have <em>both</em>: a bf16 working copy used for forward and backward, plus an fp32 "master copy" used by the optimizer for accurate updates. So 6 bytes per param.</p>

<h3>Gradients</h3>

<p>One gradient per parameter. By default in fp32 (even when params are in bf16), so <code>4N</code> bytes. They live from <code>backward()</code> until <code>zero_grad()</code> — i.e., for one optimizer step at a time. With gradient accumulation (M11), they live across multiple micro-batches.</p>

<h3>Optimizer state</h3>

<p>Depends on the optimizer (M9):</p>

<ul>
  <li><strong>SGD (no momentum)</strong>: 0 buffers per param.</li>
  <li><strong>SGD + momentum</strong>: 1 buffer (velocity), <code>4N</code> bytes in fp32.</li>
  <li><strong>Adam / AdamW</strong>: 2 buffers (first and second moment), <code>8N</code> bytes in fp32.</li>
  <li><strong>Lion</strong>: 1 buffer, <code>4N</code> bytes.</li>
</ul>

<p>For Adam in fp32, this is <strong>2× the parameter count</strong> just for state. Combined with master params and gradients, the "fixed cost" of training a 1B model with mixed-precision AdamW is <code>2N + 4N + 4N + 8N = 18N</code> bytes ≈ 18 GB.</p>

<div class="ndq">
<h4>About fixed-cost memory</h4>

<p class="q">Why are gradients in fp32 even when params are bf16?</p>
<p class="a">Two reasons. (1) Numerical precision — gradient values can be very small, and bf16's 7-bit mantissa loses precision in the tail of the distribution. (2) Optimizer compatibility — Adam's running second moment is sensitive to gradient precision; bf16 gradients passed through the EMA accumulate noticeable error. So the standard mixed-precision recipe is: weights and activations in bf16, gradients and optimizer state in fp32. We'll re-derive this in M14.</p>

<p class="q">Can I reduce optimizer state with bitsandbytes-style 8-bit Adam?</p>
<p class="a">Yes — 8-bit Adam (and similar quantized optimizers) keeps the m and v buffers in 8-bit form with per-block scales. Roughly 4× smaller than fp32 Adam, with negligible quality impact for most tasks. Useful for fitting bigger models on smaller GPUs. You wire it in by replacing your <code>torch.optim.AdamW</code> import with a quantized one — same API.</p>

<p class="q">Why not bf16 master copy?</p>
<p class="a">Because the optimizer's update <code>p ← p − lr · g</code> can have <code>lr · g</code> tiny enough to vanish into bf16's 7-bit mantissa when added to <code>p</code>. The fp32 master copy preserves these tiny accumulated changes. After the optimizer step, you cast back to bf16 for forward/backward. Without the master copy, training stops making progress after the LR decays.</p>
</div>

<h2>Bucket 4: activations (the variable one)</h2>

<p>Now the bucket that matters. For each layer in your forward pass, the autograd engine saves intermediate values needed to compute gradients in the backward pass. <em>How much</em> depends on the layer.</p>

<p>For a transformer layer (attention + MLP), the activations saved in forward include:</p>

<ul>
  <li>The input <code>x</code> to the layer (for residual)</li>
  <li>Q, K, V tensors after projections — three of them, each shape <code>(B, T, D)</code></li>
  <li>The attention matrix or softmax output — shape <code>(B, H, T, T)</code></li>
  <li>The attention output before projection — shape <code>(B, T, D)</code></li>
  <li>The MLP intermediate — shape <code>(B, T, 4D)</code> typically</li>
  <li>Various small tensors from LayerNorm (mean, rstd) — shape <code>(B, T)</code></li>
</ul>

<p>The headline term is <strong>the attention matrix at <code>(B, H, T, T)</code></strong>. For B=4, H=32, T=8192, that single tensor in bf16 is:</p>

<pre><code>4 × <span class="num">32</span> × <span class="num">8192</span> × <span class="num">8192</span> × <span class="num">2</span> = <span class="num">17.2 GB</span>     <span class="com"># PER LAYER</span></code></pre>

<p>For a 32-layer transformer at this config, attention matrices alone are <strong>550 GB</strong>. This is the entire reason FlashAttention exists (M25) — it computes attention without ever materializing the (T, T) matrix.</p>

<h3>The transformer memory formula</h3>

<p>For a vanilla transformer (no FlashAttention) in mixed precision:</p>

<pre><code>activation_bytes_per_layer ≈ B × T × <span class="num">2</span> × (<span class="num">11</span>D + <span class="num">5</span>HT)</code></pre>

<p>Where:</p>

<ul>
  <li><code>B</code> = batch size, <code>T</code> = sequence length, <code>D</code> = hidden dim, <code>H</code> = number of heads</li>
  <li>The <code>11D</code> term covers Q, K, V, attention output, MLP intermediate (4D), residuals, and LayerNorm intermediates</li>
  <li>The <code>5HT</code> term is the attention matrix and its softmax — the quadratic-in-T part</li>
  <li>The factor of 2 is for bf16</li>
</ul>

<p>This formula has appeared in multiple reduced forms in the literature; consider it a back-of-envelope estimate. The point isn't precision — it's that <strong>the quadratic-in-T term blows up first</strong>. At T=2048, attention matrices are roughly equal to the linear terms. At T=8192, they're 4× larger. At T=32768, they're enormous.</p>

<h3>Comparison with FlashAttention</h3>

<p>FlashAttention (M25) replaces the <code>5HT</code> term with O(BT × D), eliminating the quadratic-in-T memory. The new formula:</p>

<pre><code>activation_bytes_per_layer ≈ B × T × <span class="num">2</span> × (<span class="num">11</span>D + small)</code></pre>

<p>For most modern training, this is the difference between "fits on an A100" and "doesn't." We'll see how it works at the kernel level in Part VIII.</p>

<h2>Bucket 5: workspace</h2>

<p>Each cuDNN convolution, cuBLAS matmul, etc. allocates a scratch buffer for its operation. These are usually small (a few MB to a few hundred MB) and ephemeral, but they show up under heavy use.</p>

<p>The workspace is set by the cuDNN benchmark mode (M3): with <code>torch.backends.cudnn.benchmark=True</code>, cuDNN tries multiple algorithms for each convolution shape on first encounter and caches the fastest. Some algorithms use more workspace than others. Speed wins; memory hits a fixed peak.</p>

<p>You rarely need to think about workspace explicitly. If you're seeing surprising memory usage that isn't accounted for by the other four buckets, this is a candidate.</p>

<h2>Activation checkpointing: trade compute for memory</h2>

<p>You met the idea in M6 (custom Functions). Activation checkpointing — also called gradient checkpointing — applies the recompute trick at the level of <em>blocks of layers</em>. The framework handles the bookkeeping for you.</p>

<p>The idea: instead of saving every intermediate activation between layer 1 and layer N, save only checkpoints at strategic points (e.g., every K-th layer). During backward, when you reach a checkpointed region, <em>re-run the forward pass for that region</em> to recompute the missing activations on the fly.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 280" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Memory profile: with vs without activation checkpointing</text>

  <!-- axes -->
  <line x1="60" y1="200" x2="700" y2="200" stroke="#6b5d4f" stroke-width="1"/>
  <line x1="60" y1="50"  x2="60"  y2="200" stroke="#6b5d4f" stroke-width="1"/>
  <text x="370" y="225" font-size="11" fill="#6b5d4f" text-anchor="middle">time →   forward    backward</text>
  <text x="40" y="125" font-size="11" fill="#6b5d4f" text-anchor="middle" transform="rotate(-90 40 125)">memory used</text>

  <!-- midline (forward → backward boundary) -->
  <line x1="380" y1="50" x2="380" y2="200" stroke="#c8b89a" stroke-width="0.5" stroke-dasharray="3 3"/>
  <text x="380" y="46" font-size="10" fill="#6b5d4f" text-anchor="middle">peak (start of backward)</text>

  <!-- without checkpointing — peak grows linearly during forward, drops during backward -->
  <path d="M 60 195 L 200 130 L 380 60 L 540 130 L 700 195" fill="none" stroke="#c1502e" stroke-width="3"/>
  <text x="180" y="100" font-size="12" font-weight="700" fill="#c1502e">no checkpointing — peak grows with depth</text>

  <!-- with checkpointing — sawtooth pattern -->
  <path d="M 60 195 L 100 175 L 100 195 L 140 170 L 140 195 L 180 165 L 180 195 L 220 160 L 220 195 L 260 155 L 260 195 L 300 150 L 300 195 L 340 145 L 340 195 L 380 140 L 420 100 L 460 110 L 500 120 L 540 135 L 580 150 L 620 165 L 660 180 L 700 195" fill="none" stroke="#1f5f5b" stroke-width="2.5"/>
  <text x="180" y="183" font-size="12" font-weight="700" fill="#1f5f5b">with checkpointing — sawtooth, lower peak</text>
</svg>
</div>

<p>The sawtooth pattern in the green curve is the recompute happening: in backward, when we reach a checkpointed boundary, memory briefly spikes as we recompute the activations for that block, then drops as we use them and discard them. Peak memory is roughly <code>√N</code> times smaller than without checkpointing (where N is the number of layers and you checkpoint every <code>√N</code>-th).</p>

<p>The cost: <strong>~33% more compute</strong>. Forward is run twice (once originally, once during backward) for the checkpointed regions. For a memory-bound training run, that's a great trade.</p>

<h3>Using checkpoint() in PyTorch</h3>

<pre><code><span class="kw">from</span> torch.utils.checkpoint <span class="kw">import</span> checkpoint, checkpoint_sequential

<span class="kw">class</span> <span class="ty">CheckpointedTransformer</span>(nn.Module):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, n_layers, dim, n_heads):
        <span class="fn">super</span>().<span class="fn">__init__</span>()
        self.layers = nn.<span class="fn">ModuleList</span>([<span class="fn">Block</span>(dim, n_heads) <span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(n_layers)])

    <span class="kw">def</span> <span class="fn">forward</span>(self, x):
        <span class="kw">for</span> layer <span class="kw">in</span> self.layers:
            x = <span class="fn">checkpoint</span>(layer, x, use_reentrant=<span class="kw">False</span>)
        <span class="kw">return</span> x</code></pre>

<p>That's it. Each <code>checkpoint(layer, x)</code> call says "don't save activations from inside this layer; recompute them in backward." The <code>use_reentrant=False</code> selects the modern implementation (the old reentrant one had subtle issues with autograd).</p>

<p>Tradeoffs to know:</p>

<ul>
  <li><strong>Granularity matters</strong>: checkpointing every layer = max memory savings, max compute overhead (~33%). Checkpointing every K layers = less savings, less overhead. The Megatron / DeepSpeed default is "every layer" for the largest models.</li>
  <li><strong>Don't checkpoint cheap things</strong>: checkpointing a single LayerNorm is silly — recomputing it costs more than storing its (small) activation. Checkpoint <em>blocks</em> (e.g., a whole transformer layer).</li>
  <li><strong>RNG state</strong>: if your block uses Dropout or any other random op, the checkpoint API needs to <em>save and restore the RNG state</em> so the recomputed forward gives the same answer. PyTorch handles this for you, but it's why you can't trivially checkpoint things that have non-deterministic side effects.</li>
</ul>

<div class="warn">
<strong>Activation checkpointing + FlashAttention together is the modern combo.</strong> FlashAttention removes the quadratic-in-T attention term; activation checkpointing reduces the linear-in-depth term by ~√N. Together they make 100B+ models trainable on A100/H100 GPUs. Don't pick one or the other — modern training uses both.
</div>

<h2>The CUDA caching allocator (briefly)</h2>

<p>When you call <code>torch.zeros(...)</code> or any tensor-creating op, PyTorch doesn't always go to <code>cudaMalloc</code>. It maintains a <em>caching allocator</em>: when a tensor's storage is freed, the memory is returned to a pool, not to the OS. Next allocation of similar size reuses the cached block.</p>

<p>Why? Because <code>cudaMalloc</code> and <code>cudaFree</code> are slow (synchronous calls into the driver). Constant alloc/free cycles in the training loop would dominate runtime. The caching allocator amortizes this.</p>

<p>The implication for memory accounting:</p>

<ul>
  <li><code>nvidia-smi</code> shows you what the OS thinks PyTorch holds — i.e., the size of the caching allocator pool. Not what's actually <em>in use</em>.</li>
  <li><code>torch.cuda.memory_allocated()</code> shows what's currently allocated to live tensors.</li>
  <li><code>torch.cuda.memory_reserved()</code> shows what the caching allocator has reserved from the OS (= what nvidia-smi sees, modulo other GPU users).</li>
</ul>

<p>The discrepancy between <code>memory_allocated</code> and <code>memory_reserved</code> can be huge — sometimes 10+ GB of "free" memory in the caching pool that isn't returned to the OS. That's normal. <code>torch.cuda.empty_cache()</code> forces it back, but you almost never want to call this in production (it's slow and the next allocation will go back to <code>cudaMalloc</code>).</p>

<h3>Memory diagnosis tools</h3>

<pre><code><span class="com"># 1. Quick snapshot</span>
<span class="fn">print</span>(torch.cuda.<span class="fn">memory_summary</span>())

<span class="com"># 2. Track over time — useful for spotting memory leaks</span>
<span class="kw">def</span> <span class="fn">log_mem</span>(tag):
    a = torch.cuda.<span class="fn">memory_allocated</span>() / <span class="num">1e9</span>
    r = torch.cuda.<span class="fn">memory_reserved</span>() / <span class="num">1e9</span>
    <span class="fn">print</span>(<span class="str">f"{tag}: allocated={a:.2f}GB reserved={r:.2f}GB"</span>)

<span class="fn">log_mem</span>(<span class="str">"after model load"</span>)
<span class="com"># ... a few steps of training ...</span>
<span class="fn">log_mem</span>(<span class="str">"after step 100"</span>)
<span class="fn">log_mem</span>(<span class="str">"after step 1000"</span>)         <span class="com"># should be roughly the same as step 100!</span></code></pre>

<p>If <code>memory_allocated</code> grows over training steps, you have a leak. The most common causes:</p>

<ol>
  <li><strong>Holding tensor references</strong>: storing per-step losses as a list of CUDA tensors instead of <code>.item()</code> floats. Each step adds a tensor to the list and never releases.</li>
  <li><strong>Autograd graphs being held</strong>: assigning a tensor with <code>requires_grad=True</code> to a class attribute prevents the graph from being freed. <code>.detach()</code> before storing.</li>
  <li><strong>Unclosed hooks</strong>: as we discussed in M5, hooks can hold closures that hold model references.</li>
</ol>

<h2>Memory fragmentation</h2>

<p>The caching allocator's pool can fragment. You allocate a 4 GB tensor, free it, allocate two 2 GB tensors. The 4 GB block is now split. If you later need a contiguous 4 GB allocation, you might get OOM <em>even though you have 4+ GB total free</em> because no single contiguous block is that big.</p>

<p>Fragmentation symptoms:</p>

<ul>
  <li>OOM with significant free memory in <code>memory_reserved - memory_allocated</code></li>
  <li>Variable batch sizes or sequence lengths in training (each new shape opens a new block)</li>
  <li>Long-running training that gradually approaches OOM near the memory limit</li>
</ul>

<p>Mitigations:</p>

<ul>
  <li><strong>Use fixed shapes</strong> when possible — bucket sequence lengths (M10), use the same batch size every step.</li>
  <li><code>PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True</code> — newer allocator mode that handles fragmentation better. Set as an env var before importing torch.</li>
  <li><code>torch.cuda.empty_cache()</code> as a last resort — releases the pool back to the OS. Slow but defragments.</li>
</ul>

<h2>The mental model: predict before you run</h2>

<p>Pull all of this together into a memory budget you can compute by hand. For a typical mixed-precision AdamW training of a transformer:</p>

<pre><code>fixed_per_param = <span class="num">2</span>(bf16) + <span class="num">4</span>(fp32 master) + <span class="num">4</span>(grad fp32) + <span class="num">8</span>(adam fp32) = <span class="num">18</span> bytes
fixed_total = <span class="num">18</span> × N

per_layer_activations = B × T × <span class="num">2</span> × (<span class="num">11</span>D + <span class="num">5</span>HT)        <span class="com"># bf16</span>
total_activations = layers × per_layer_activations

total_memory = fixed_total + total_activations + workspace</code></pre>

<p>Worked example: GPT-2 medium-ish, 350M params, 24 layers, dim=1024, 16 heads, T=1024, B=8:</p>

<pre><code>fixed       = <span class="num">18</span> × <span class="num">350</span>M = <span class="num">6.3</span> GB
per_layer   = <span class="num">8</span> × <span class="num">1024</span> × <span class="num">2</span> × (<span class="num">11</span>×<span class="num">1024</span> + <span class="num">5</span>×<span class="num">16</span>×<span class="num">1024</span>) = <span class="num">8</span> × <span class="num">1024</span> × <span class="num">2</span> × <span class="num">93184</span> ≈ <span class="num">1.5</span> GB
total_acts  = <span class="num">24</span> × <span class="num">1.5</span> GB = <span class="num">36</span> GB
total       ≈ <span class="num">42</span> GB           <span class="com"># before workspace, kernel scratch, etc.</span></code></pre>

<p>So this config eats an A100 (40 GB) and OOMs. Options:</p>

<ol>
  <li>Halve the batch size: activations halve → <code>~24 GB</code>, fits comfortably</li>
  <li>Activation checkpointing: activations × ~0.2 → <code>~7 GB</code>, easy fit</li>
  <li>FlashAttention: kills the <code>5HT</code> term, big savings on long sequences</li>
  <li>Switch to 8-bit Adam: optimizer state goes from 8N to 2N, saves ~3 GB</li>
</ol>

<p>The lesson isn't the specific arithmetic — it's that you can <em>predict</em> memory consumption from your config. When training OOMs, you can compute the bottleneck instead of guessing.</p>

<h2>Code Magnets: build a memory-aware training step</h2>

<p>You're modifying a training step to use activation checkpointing on the model's transformer layers and to log memory usage every 100 steps. Build the inner step.</p>

<div class="magnets">
<p>Arrange the magnets into a working step. Two are red herrings.</p>

<div class="magnet-pool">
  <span class="magnet">def train_step(model, opt, scheduler, batch, step):</span>
  <span class="magnet">    x, y = [t.to(device, non_blocking=True) for t in batch]</span>
  <span class="magnet">    out = model(x)</span>
  <span class="magnet">    out = checkpoint(model, x, use_reentrant=False)</span>
  <span class="magnet">    loss = F.cross_entropy(out, y)</span>
  <span class="magnet">    loss.backward()</span>
  <span class="magnet">    opt.step(); opt.zero_grad(); scheduler.step()</span>
  <span class="magnet">    if step % 100 == 0:</span>
  <span class="magnet">        log_mem(f"step {step}")</span>
  <span class="magnet">        torch.cuda.empty_cache()</span>
  <span class="magnet">    return loss.item()</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">def</span> <span class="fn">train_step</span>(model, opt, scheduler, batch, step):
    x, y = [t.<span class="fn">to</span>(device, non_blocking=<span class="kw">True</span>) <span class="kw">for</span> t <span class="kw">in</span> batch]
    out = <span class="fn">model</span>(x)
    loss = F.<span class="fn">cross_entropy</span>(out, y)
    loss.<span class="fn">backward</span>()
    opt.<span class="fn">step</span>(); opt.<span class="fn">zero_grad</span>(); scheduler.<span class="fn">step</span>()
    <span class="kw">if</span> step % <span class="num">100</span> == <span class="num">0</span>:
        <span class="fn">log_mem</span>(<span class="fn">f"step {step}"</span>)
    <span class="kw">return</span> loss.<span class="fn">item</span>()</code></pre>
<p>The traps:</p>
<ul>
  <li><code>out = checkpoint(model, x, ...)</code> — wrapping the <em>whole</em> model in a checkpoint defeats the purpose. Checkpointing belongs <em>inside</em> the model's <code>forward</code>, around individual blocks. Doing it at the top means recomputing the entire forward pass during backward — same cost as just not checkpointing at all (worse, actually, because of the bookkeeping).</li>
  <li><code>torch.cuda.empty_cache()</code> after every log is wrong — it's slow, and the cache is supposed to grow to a steady state. Calling it in the hot path defeats the caching allocator. Reserve it for emergencies.</li>
</ul>
<p>The right pattern: model's <code>forward</code> wraps each block in <code>checkpoint(block, x, use_reentrant=False)</code> internally. Training step looks like normal training, calls <code>log_mem</code> for diagnosis, and trusts the allocator.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each memory concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>fp32 master copy</div>
  <div>A. Per-tensor scratch buffer used by cuDNN/cuBLAS for one op.</div>

  <div>Activations</div>
  <div>B. Trade ~33% more compute for √N less peak activation memory.</div>

  <div>Activation checkpointing</div>
  <div>C. The biggest memory bucket at scale; scales with batch × seq × depth.</div>

  <div>CUDA caching allocator</div>
  <div>D. Preserves precision when applying tiny optimizer updates that bf16 would round away.</div>

  <div>memory_reserved vs memory_allocated</div>
  <div>E. Holds freed tensor memory in a pool to avoid slow cudaMalloc/cudaFree.</div>

  <div>cuDNN workspace</div>
  <div>F. The gap is normal — cached blocks the allocator hasn't returned to the OS.</div>

  <div>Fragmentation</div>
  <div>G. OOM despite having free memory total — no single block is large enough.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>fp32 master copy</strong> → D<br>
<strong>Activations</strong> → C<br>
<strong>Activation checkpointing</strong> → B<br>
<strong>CUDA caching allocator</strong> → E<br>
<strong>memory_reserved vs memory_allocated</strong> → F<br>
<strong>cuDNN workspace</strong> → A<br>
<strong>Fragmentation</strong> → G
</p>
<p>The mental shortcut: <em>master copy preserves tiny updates, activations dominate at scale, checkpointing trades compute for memory, allocator caches frees, reserved-allocated gap is the cache, workspace is per-op scratch, fragmentation is contiguous-block starvation</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A 7B-parameter transformer trained with mixed-precision AdamW. Compute the fixed memory cost (params + master + grads + Adam state) in GB.</p>
<details class="answer"><summary>show answer</summary>
<p>Per parameter: 2 (bf16) + 4 (fp32 master) + 4 (grad fp32) + 8 (Adam m + v in fp32) = <strong>18 bytes</strong>. For 7B params: <code>18 × 7e9 = 126 GB</code>. That's already more than a single 80 GB H100 — which is why training a 7B model on a single GPU isn't possible without sharding (FSDP, M17). At minimum 2 H100s; in practice 4-8 with activation memory.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Your training is OOMing at step 1247. <code>nvidia-smi</code> shows you have 5 GB free. <code>torch.cuda.memory_allocated()</code> says you're using 38 GB. <code>torch.cuda.memory_reserved()</code> says 43 GB. What's the most likely cause and what should you try?</p>
<details class="answer"><summary>show answer</summary>
<p>Fragmentation. You have 5 GB free in the pool (43 - 38), but no contiguous block large enough for the new allocation. Two things to try: (1) Check if you have variable-shape batches — sort/bucket by length (M10) so shapes repeat. (2) Set <code>PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True</code> as an env var; this allocator mode handles fragmentation much better. As a last resort, <code>torch.cuda.empty_cache()</code> before the suspect allocation, but this slows training so use only as diagnosis.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> A vanilla transformer training run (no FlashAttention) at T=4096 OOMs. Switching to T=2048 fits comfortably. By roughly what factor did peak memory drop, and which term dominated?</p>
<details class="answer"><summary>show answer</summary>
<p>The attention matrix term is <code>5HT × T = 5HT²</code> per layer. Halving T quarters this term. The linear-in-T terms (<code>11D × T</code>) only halve. So the dominant change is in the quadratic term — peak memory dropped roughly 4× for that bucket. This is exactly why FlashAttention exists: by replacing the <code>O(T²)</code> attention term with <code>O(T)</code>, you can run T=8192 or T=32768 with the memory you used to spend on T=2048.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> A team reports their training memory grows by ~50 MB per 1000 steps and eventually OOMs after a million steps. Sketch the diagnosis path.</p>
<details class="answer"><summary>show answer</summary>
<p>Linear growth = leak. Step 1: log <code>memory_allocated</code> at regular intervals to confirm the rate. Step 2: look for tensor references being held — most common: appending CUDA tensors to a list (e.g., <code>losses.append(loss)</code> instead of <code>losses.append(loss.item())</code>); class attributes accumulating per-batch outputs; hooks holding closures over tensors; metrics objects storing tensors instead of floats. Step 3: if not obvious, use <code>torch.cuda.memory_snapshot()</code> for the per-allocation history, or <code>torch.cuda.memory._record_memory_history()</code> followed by snapshot dumps. The leak is almost always at the Python level, not in the framework.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>Training memory has <strong>five buckets</strong>: parameters, gradients, optimizer state, activations, workspace. The first three are constant; activations dominate at scale.</li>
  <li>Mixed-precision AdamW costs <strong>18 bytes per parameter</strong> in fixed memory (2 bf16 + 4 fp32 master + 4 grad + 8 Adam). For 1B params, that's 18 GB <em>before</em> activations.</li>
  <li>Why fp32 master copy: bf16's 7-bit mantissa would round away tiny optimizer updates. Master copy preserves precision; cast to bf16 for forward/backward.</li>
  <li>Why fp32 grads: bf16 grad accumulation (Adam's <code>v</code>) accumulates noticeable error. Standard recipe: weights/activations bf16, grads/optimizer fp32.</li>
  <li>Activations have a <strong>linear-in-depth term</strong> (~11D per token per layer) and a <strong>quadratic-in-T term</strong> (~5HT per layer for the attention matrix). Long sequences are dominated by the quadratic.</li>
  <li><strong>FlashAttention</strong> kills the <code>O(T²)</code> term — it's the single biggest activation-memory win for long-sequence training.</li>
  <li><strong>Activation checkpointing</strong> trades ~33% more compute for ~√N less peak memory. Checkpoint <em>blocks</em>, not individual cheap ops.</li>
  <li>The modern combo: FlashAttention + activation checkpointing. Together they make 100B+ models trainable on A100/H100.</li>
  <li><strong>The CUDA caching allocator</strong> keeps freed memory in a pool. <code>memory_reserved - memory_allocated</code> is the gap; that's normal.</li>
  <li><strong>Fragmentation</strong>: OOM with free memory total but no contiguous block. Mitigate with consistent shapes, <code>expandable_segments:True</code>, or <code>empty_cache()</code> as last resort.</li>
  <li><strong>Memory leaks</strong> at the Python level: storing CUDA tensors instead of <code>.item()</code> floats, retained autograd graphs, unclosed hooks. Diagnose with <code>memory_allocated</code> over time.</li>
  <li>The reflex: when OOM hits, <em>predict</em> first (compute the budget), then <em>profile</em> if predict didn't match (memory_summary, snapshot). Don't guess — the budget is calculable.</li>
</ul>
</div>

<p>Module 13 takes the same predictive mindset to time. <code>torch.profiler</code>, traces in Chrome/Perfetto, the GPU/CPU/data-pipeline-bound diagnosis. By the end you'll know whether your training is bottlenecked by the optimizer step, the dataloader, or a single bad kernel — and you'll have the tools to find each one.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">12</span>
  <span>Memory: where does it all go?</span>
</div>
"""

emit("12_memory", "Module 12 — Memory: where does it all go?", BODY)
