#!/usr/bin/env python3
"""Module 14: Mixed precision: autocast & GradScaler — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part V · Module 14</div>
  <h1 class="module-title">Mixed precision: <em>autocast &amp; GradScaler</em></h1>
  <p class="module-sub">— what dtype lives where in a training step, why bf16 is "free" but fp16 needs a loss scaler, and the modern recipe that's the silent default of every transformer codebase</p>
</div>

<p>Five modules ago we met Float32, Float16, and BFloat16 as personalities. Module 9 reminded us that the optimizer keeps Adam state in fp32. Module 12 said "you keep both a bf16 working copy and an fp32 master copy of every parameter." Module 13 said "mixed precision is the first optimization to try." Now we wire it all together.</p>

<p>The headline: a modern training step doesn't run in <em>one</em> dtype — it runs in a carefully chosen mix. The forward pass uses bf16 (or fp16) for matmul speed and memory. The loss is computed in fp32 for stability. Gradients are stored in fp32. The optimizer updates an fp32 master copy. Then the result is cast back to bf16 for the next forward. Each piece is in the precision where it works best, and PyTorch wires it up for you with two APIs: <code>torch.autocast</code> and (for fp16) <code>torch.amp.GradScaler</code>.</p>

<div class="keyidea">
Mixed precision is a <strong>per-region dtype contract</strong>: <em>matmul-heavy ops in bf16/fp16</em> (fast tensor cores, half the memory), <em>numerically-sensitive ops in fp32</em> (loss reductions, softmax, layer norm), <em>all gradients in fp32</em> (precision matters for Adam), and <em>parameters kept in both</em> bf16 (working copy) and fp32 (master copy for the optimizer). <code>autocast</code> handles the per-op routing; <code>GradScaler</code> handles the fp16-specific underflow problem. bf16 doesn't need the scaler. That's the whole story.
</div>

<h2>One new face</h2>

<div class="character" style="--c: #c1502e;">
  <div class="avatar" style="background: #c1502e; color: #fff;">×k</div>
  <div>
    <p class="who">GradScaler</p>
    <p class="name">"I solve fp16's tiny-gradient problem by multiplying everything by a big number."</p>
    <p class="says">When you train in fp16, lots of gradient values are smaller than fp16's smallest representable number (~6e-5) and underflow to zero. That's training death by silent loss of signal. My job: scale the loss by a big constant K (typically 65536) before backward. Gradients come out K times larger, safely inside fp16's representable range. Then I unscale them before the optimizer step. I also detect overflow (any inf/nan) and skip the step + halve K when it happens. <strong>bf16 has the same range as fp32, so it doesn't underflow — and doesn't need me.</strong></p>
  </div>
</div>

<h2>What dtype lives where</h2>

<p>The single most important diagram in this module:</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 340" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrMP" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">A mixed-precision training step: the dtype at each point</text>

  <!-- Storage panel (left) -->
  <text x="20" y="55" font-size="12" font-weight="700" fill="#6b5d4f">stored across whole run:</text>
  <g transform="translate(20, 65)">
    <rect x="0" y="0" width="120" height="36" fill="#fff8a8" stroke="#c1502e" stroke-width="2"/>
    <text x="60" y="16" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">params (bf16)</text>
    <text x="60" y="28" text-anchor="middle" font-size="10" fill="#1a1612">used in forward</text>

    <rect x="0" y="40" width="120" height="36" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="60" y="56" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">master (fp32)</text>
    <text x="60" y="68" text-anchor="middle" font-size="10" fill="#1a1612">read by optimizer</text>

    <rect x="0" y="80" width="120" height="36" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="60" y="96" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">grads (fp32)</text>
    <text x="60" y="108" text-anchor="middle" font-size="10" fill="#1a1612">cleared each step</text>

    <rect x="0" y="120" width="120" height="36" fill="#d3e9f5" stroke="#133e3b" stroke-width="2"/>
    <text x="60" y="136" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">Adam m,v (fp32)</text>
    <text x="60" y="148" text-anchor="middle" font-size="10" fill="#1a1612">precision matters</text>
  </g>

  <!-- Step pipeline (right of storage) -->
  <text x="200" y="55" font-size="12" font-weight="700" fill="#6b5d4f">one training step:</text>

  <!-- Forward in bf16 -->
  <g transform="translate(200, 65)">
    <rect x="0" y="0" width="100" height="36" fill="#fff8a8" stroke="#c1502e" stroke-width="2"/>
    <text x="50" y="16" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">forward</text>
    <text x="50" y="28" text-anchor="middle" font-size="10" fill="#1a1612">bf16 matmuls</text>
  </g>

  <!-- LayerNorm/softmax exception in fp32 -->
  <g transform="translate(310, 65)">
    <rect x="0" y="0" width="100" height="36" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2" stroke-dasharray="4 3"/>
    <text x="50" y="16" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">norms/softmax</text>
    <text x="50" y="28" text-anchor="middle" font-size="9" fill="#1a1612">cast up to fp32</text>
  </g>

  <!-- Loss in fp32 -->
  <g transform="translate(420, 65)">
    <rect x="0" y="0" width="100" height="36" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="50" y="16" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">loss (fp32)</text>
    <text x="50" y="28" text-anchor="middle" font-size="10" fill="#1a1612">CE/logsumexp</text>
  </g>

  <!-- Backward -->
  <g transform="translate(530, 65)">
    <rect x="0" y="0" width="100" height="36" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="50" y="16" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">backward</text>
    <text x="50" y="28" text-anchor="middle" font-size="10" fill="#1a1612">grads → fp32</text>
  </g>

  <!-- Optimizer -->
  <g transform="translate(420, 130)">
    <rect x="0" y="0" width="100" height="36" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="50" y="16" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">optimizer.step</text>
    <text x="50" y="28" text-anchor="middle" font-size="10" fill="#1a1612">all in fp32</text>
  </g>

  <!-- Cast back -->
  <g transform="translate(310, 130)">
    <rect x="0" y="0" width="100" height="36" fill="#fff8a8" stroke="#c1502e" stroke-width="2"/>
    <text x="50" y="16" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">cast → bf16</text>
    <text x="50" y="28" text-anchor="middle" font-size="10" fill="#1a1612">for next forward</text>
  </g>

  <!-- arrows along top row -->
  <path d="M 300 83 L 308 83" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrMP)"/>
  <path d="M 410 83 L 418 83" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrMP)"/>
  <path d="M 520 83 L 528 83" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrMP)"/>
  <!-- backward to optimizer -->
  <path d="M 580 102 Q 580 120 522 145" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrMP)"/>
  <!-- optimizer to cast -->
  <path d="M 420 148 L 412 148" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrMP)"/>

  <!-- annotations -->
  <text x="370" y="220" font-family="'Caveat', cursive" font-size="20" fill="#c1502e" text-anchor="middle">"autocast" handles forward routing automatically — you don't write the casts</text>
  <text x="370" y="245" font-family="'Caveat', cursive" font-size="20" fill="#1a1612" text-anchor="middle">grads come back in fp32 because the framework upcast them on output</text>
  <text x="370" y="275" font-family="'Caveat', cursive" font-size="20" fill="#1f5f5b" text-anchor="middle">optimizer reads master fp32 + grad fp32 → updates master → casts to bf16</text>
  <text x="370" y="305" font-family="'Caveat', cursive" font-size="22" fill="#c1502e" text-anchor="middle">total cost: 18 bytes/param &middot; total speed: ~2× free</text>
</svg>
</div>

<p>Three things to internalize. (1) <strong>The forward pass is not all bf16</strong> — autocast routes individual ops to whichever precision is appropriate. Matmuls go to bf16 (fast tensor cores). Reductions, normalizations, and softmax stay in fp32 (numerically sensitive). (2) <strong>Gradients always come back as fp32</strong>, regardless of forward precision. The framework upcasts at the boundary so your optimizer sees consistent fp32 grads. (3) <strong>The master fp32 copy</strong> is what the optimizer updates. Without it, the bf16 weights would round away tiny optimizer updates and training would stall.</p>

<h2><code>torch.autocast</code>: the per-op router</h2>

<p>The minimal API:</p>

<pre><code><span class="kw">with</span> torch.<span class="fn">autocast</span>(<span class="str">'cuda'</span>, dtype=torch.bfloat16):
    output = <span class="fn">model</span>(x)
    loss = F.<span class="fn">cross_entropy</span>(output, y)</code></pre>

<p>Inside the <code>with</code> block, autocast looks at every op being dispatched and decides:</p>

<ul>
  <li><strong>Cast inputs to bf16</strong> for ops that benefit and are stable: matmul, conv, linear, attention.</li>
  <li><strong>Keep in fp32</strong> for ops where bf16 hurts: <code>logsumexp</code>, <code>softmax</code>, <code>log</code>, <code>exp</code>, layer norm reductions, loss functions.</li>
  <li><strong>Pass through unchanged</strong> for ops that are neutral.</li>
</ul>

<p>The result: matmuls run on tensor cores at bf16 throughput; the few ops where bf16 would lose precision automatically run in fp32. You don't think about it.</p>

<h3>The autocast op list (you don't have to memorize this)</h3>

<p>PyTorch maintains a list of which ops cast which way. Three categories:</p>

<div class="table-wrap">
<table>
<caption>How autocast routes ops (abbreviated)</caption>
<thead><tr><th>Category</th><th>Treatment</th><th>Examples</th></tr></thead>
<tbody>
<tr><td><strong>Lower precision</strong></td><td>cast inputs to bf16/fp16</td><td><code>matmul</code>, <code>linear</code>, <code>conv2d</code>, <code>scaled_dot_product_attention</code></td></tr>
<tr><td><strong>fp32-preserving</strong></td><td>cast inputs to fp32 if they're lower</td><td><code>softmax</code>, <code>logsumexp</code>, <code>layer_norm</code>'s reduction parts, <code>cross_entropy</code></td></tr>
<tr><td><strong>Promotes inputs</strong></td><td>finds the highest-precision input and uses that</td><td>elementwise add/sub when mixing dtypes, comparisons</td></tr>
</tbody>
</table>
</div>

<p>The first category is where the speedup comes from — tensor cores in bf16/fp16 are 2-4× faster than fp32 on Ampere/Hopper hardware. The second category is the safety net — the ops that would otherwise lose meaningful precision in low precision. The third is consistency.</p>

<div class="warn">
<strong>Don't manually <code>.to(torch.bfloat16)</code> inside an autocast block.</strong> You'll fight the autocast logic. Either trust autocast for everything (the standard recipe) or turn it off and manually manage dtypes (rarely worth it). Mixing the two leads to hard-to-debug type promotion bugs.
</div>

<h3>Where to put the autocast block</h3>

<pre><code><span class="kw">for</span> step, batch <span class="kw">in</span> <span class="fn">enumerate</span>(loader):
    x, y = batch
    opt.<span class="fn">zero_grad</span>()

    <span class="kw">with</span> torch.<span class="fn">autocast</span>(<span class="str">'cuda'</span>, dtype=torch.bfloat16):
        out = <span class="fn">model</span>(x)
        loss = F.<span class="fn">cross_entropy</span>(out, y)

    loss.<span class="fn">backward</span>()                                     <span class="com"># OUTSIDE the autocast — autograd handles dtype</span>
    torch.nn.utils.<span class="fn">clip_grad_norm_</span>(model.<span class="fn">parameters</span>(), <span class="num">1.0</span>)
    opt.<span class="fn">step</span>()                                         <span class="com"># OUTSIDE — runs in fp32 master</span>
    scheduler.<span class="fn">step</span>()</code></pre>

<p>The pattern: <strong>wrap the forward and loss in <code>autocast</code>; leave backward and step outside</strong>. Autograd already records the dtype at forward time and handles the casting in backward correctly. The optimizer step uses the master fp32 copy, which is fp32 by construction.</p>

<h2>bf16 vs fp16: the decision</h2>

<p>You met the dtype layouts in M3. Here's the decision in one sentence: <strong>use bf16 if your hardware supports it (Ampere / A100 / H100 / TPUs / MI200+), fp16 only if you're stuck on older hardware (Pascal/Volta/T4)</strong>.</p>

<div class="table-wrap">
<table>
<caption>bf16 vs fp16 for training</caption>
<thead><tr><th></th><th>bf16</th><th>fp16</th></tr></thead>
<tbody>
<tr><td>Range (max value)</td><td>~3.4e38 (same as fp32)</td><td><strong>65,504</strong> (overflows easily)</td></tr>
<tr><td>Underflow (min positive)</td><td>~1.2e-38</td><td>~6e-5 (gradients underflow)</td></tr>
<tr><td>Mantissa precision</td><td>7 bits</td><td>10 bits</td></tr>
<tr><td>Need GradScaler?</td><td><strong>No</strong></td><td><strong>Yes</strong> (mandatory)</td></tr>
<tr><td>Speed on tensor cores</td><td>Same as fp16</td><td>Same as bf16</td></tr>
<tr><td>Hardware support</td><td>Ampere+ (A100/H100), TPUs</td><td>Anything from Pascal+</td></tr>
<tr><td>Recommended for training</td><td><strong>Yes — modern default</strong></td><td>Only on older hardware</td></tr>
</tbody>
</table>
</div>

<p>Two things bear repeating:</p>

<ol>
  <li><strong>Tensor core speed is the same for bf16 and fp16</strong> on supported hardware. There's no speed reason to prefer one over the other.</li>
  <li><strong>The numerical reason favors bf16 strongly.</strong> Same exponent range as fp32 (no overflow, no underflow at gradient scale). The 3-bit mantissa difference vs fp16 doesn't matter in deep learning where activations have noise from regularization and stochastic optimization.</li>
</ol>

<p>If you're on H100 or newer, you can also use fp8 for some operations (more on this at the end of the module). But for the next few years, bf16 is the workhorse.</p>

<h2><code>GradScaler</code>: the fp16 safety net</h2>

<p>Skip this section if you only train in bf16 — you don't need GradScaler. Read it if (a) you train on Volta/Pascal/T4 or (b) you want to understand why bf16 won.</p>

<p>The problem: fp16's smallest positive value is ~6e-5. Many gradients in deep learning are smaller than that — especially in deep networks where chain-rule shrinks gradients with depth. They <em>underflow to zero</em>. The model stops learning in those parameters.</p>

<p>The trick: scale the loss before backward. Since gradients are linear in the loss (chain rule), multiplying loss by K multiplies every gradient by K. Pick K large enough that all the meaningful gradients are above 6e-5. Then unscale before the optimizer step.</p>

<pre><code>scaler = torch.amp.<span class="fn">GradScaler</span>(<span class="str">'cuda'</span>)              <span class="com"># the modern API</span>

<span class="kw">for</span> step, batch <span class="kw">in</span> <span class="fn">enumerate</span>(loader):
    x, y = batch
    opt.<span class="fn">zero_grad</span>()

    <span class="kw">with</span> torch.<span class="fn">autocast</span>(<span class="str">'cuda'</span>, dtype=torch.float16):
        out = <span class="fn">model</span>(x)
        loss = F.<span class="fn">cross_entropy</span>(out, y)

    scaler.<span class="fn">scale</span>(loss).<span class="fn">backward</span>()                  <span class="com"># scales loss by K, backward uses scaled grads</span>
    scaler.<span class="fn">unscale_</span>(opt)                            <span class="com"># divides grads by K so clipping sees real magnitudes</span>
    torch.nn.utils.<span class="fn">clip_grad_norm_</span>(model.<span class="fn">parameters</span>(), <span class="num">1.0</span>)
    scaler.<span class="fn">step</span>(opt)                                <span class="com"># applies the step (unscale already done)</span>
    scaler.<span class="fn">update</span>()                                  <span class="com"># adjusts K for next iteration</span></code></pre>

<p>What's happening:</p>

<ol>
  <li><code>scaler.scale(loss)</code> multiplies the loss by K (initially 65536).</li>
  <li><code>backward()</code> on the scaled loss produces gradients that are K times larger — safely inside fp16's range.</li>
  <li><code>scaler.unscale_(opt)</code> divides the gradients by K so the optimizer sees the true magnitudes (and so gradient clipping works on real values).</li>
  <li><code>scaler.step(opt)</code> calls the optimizer's step. <strong>If any gradient is inf/nan</strong> (overflow), it skips the step entirely.</li>
  <li><code>scaler.update()</code> adjusts K. If overflow happened, halve K. If many steps without overflow, double K. K converges to the largest value that doesn't overflow.</li>
</ol>

<p>The dynamic-scaling part is what makes GradScaler magical — you don't manually tune K. It auto-tunes to the largest value the current model can handle, which changes as training progresses.</p>

<div class="warn">
<strong>The unscale-then-clip ordering matters.</strong> If you clip gradients <em>before</em> unscaling, you'd be clipping inflated values — the clip threshold wouldn't be meaningful. Always: <code>unscale_</code> first, then <code>clip_grad_norm_</code>, then <code>scaler.step</code>. Get the order wrong and clipping becomes a no-op.
</div>

<h3>Why bf16 doesn't need GradScaler</h3>

<p>Two reasons. (1) bf16's smallest positive value is ~1.2e-38 — gradients don't underflow. (2) bf16's max is ~3.4e38 — gradients don't overflow either. The whole problem GradScaler exists to solve doesn't apply to bf16.</p>

<p>Practically: with bf16, you write the simpler training loop with no <code>scaler</code>, and it just works. This is a substantial simplicity win and one of several reasons bf16 became the default.</p>

<h2>TF32: the surprise extra format</h2>

<p>One more dtype to know about. On Ampere and newer GPUs, when you do an fp32 matmul, the hardware doesn't actually use full fp32 internally — it uses <strong>TF32</strong>: 8-bit exponent (same as fp32), 10-bit mantissa (same as fp16), 19 bits total. Roughly 2× the throughput of true fp32.</p>

<p>This is on by default in PyTorch. You probably benefit from it without realizing. The relevant flags:</p>

<pre><code>torch.backends.cuda.matmul.allow_tf32 = <span class="kw">True</span>     <span class="com"># default True on Ampere+</span>
torch.backends.cudnn.allow_tf32 = <span class="kw">True</span>            <span class="com"># default True on Ampere+</span>

<span class="com"># Disable for bit-exact fp32 (rare):</span>
torch.backends.cuda.matmul.allow_tf32 = <span class="kw">False</span></code></pre>

<p>Why does this matter? Because if you write code that's "definitely fp32 throughout" (no autocast), you might still be silently using TF32. For most workloads this is fine — TF32's precision is similar to bf16 for the matmul itself, while keeping fp32 dynamic range. For some scientific workloads, you want true fp32 — turn TF32 off explicitly.</p>

<h2>Common NaN sources in mixed precision</h2>

<p>bf16 is mostly NaN-proof. fp16 is not. If your loss goes NaN, walk through this list:</p>

<div class="table-wrap">
<table>
<caption>NaN diagnosis tree</caption>
<thead><tr><th>Symptom</th><th>Likely cause</th><th>Fix</th></tr></thead>
<tbody>
<tr><td>Loss NaN at step 1</td><td>Bad init or huge LR</td><td>Check init scheme (M8); halve LR; warmup harder</td></tr>
<tr><td>Loss NaN after some steps in fp16</td><td>GradScaler not used</td><td>Wrap in scaler.scale → backward → step pattern</td></tr>
<tr><td>Loss NaN intermittently in fp16</td><td>Activations spiking past 65504</td><td>Lower LR; check for un-clipped activations; switch to bf16</td></tr>
<tr><td>Loss NaN in bf16</td><td>Numerical instability — softmax overflow, log of 0, etc.</td><td>Use stable variants (logsumexp, F.cross_entropy from logits); add small ε</td></tr>
<tr><td>Loss NaN from a custom op</td><td>Custom backward in fp16/bf16 hits numerical edge</td><td>Force the op to fp32 inside its forward; gradcheck (M6)</td></tr>
<tr><td>NaN appears in one parameter, then spreads</td><td>That parameter blew up, then propagated through residual stream</td><td>Find first NaN with hook (M5); investigate that layer specifically</td></tr>
</tbody>
</table>
</div>

<p>Two diagnostic patterns worth knowing:</p>

<h3>Find the first NaN with hooks</h3>

<pre><code><span class="kw">def</span> <span class="fn">nan_check_hook</span>(name):
    <span class="kw">def</span> <span class="fn">hook</span>(module, inputs, output):
        <span class="kw">if</span> torch.<span class="fn">isnan</span>(output).<span class="fn">any</span>() <span class="kw">or</span> torch.<span class="fn">isinf</span>(output).<span class="fn">any</span>():
            <span class="fn">print</span>(<span class="fn">f"!!! NaN/Inf in {name} output"</span>)
    <span class="kw">return</span> hook

<span class="kw">for</span> name, m <span class="kw">in</span> model.<span class="fn">named_modules</span>():
    m.<span class="fn">register_forward_hook</span>(<span class="fn">nan_check_hook</span>(name))</code></pre>

<p>Run a few steps. The first hook to fire is where the NaN starts. Investigate <em>that</em> layer's inputs and weights. (Don't leave these hooks installed in production — they sync the GPU on every layer.)</p>

<h3>Use <code>anomaly_mode</code> for backward NaNs</h3>

<pre><code><span class="kw">with</span> torch.autograd.<span class="fn">set_detect_anomaly</span>(<span class="kw">True</span>):
    out = <span class="fn">model</span>(x)
    loss = F.<span class="fn">cross_entropy</span>(out, y)
    loss.<span class="fn">backward</span>()</code></pre>

<p>This tells autograd to track which forward op produced any inf/NaN that appears in backward, with a stack trace. <em>Slow as molasses</em>, so use it for debugging only — a 5-10× slowdown is typical. Find the bug, then turn it off.</p>

<h2>fp8: the newest player (briefly)</h2>

<p>On H100 and newer, hardware fp8 matmuls become available. Two formats:</p>

<ul>
  <li><strong>e4m3</strong>: 4 exponent bits, 3 mantissa bits. Max value ~448. More precision, less range.</li>
  <li><strong>e5m2</strong>: 5 exponent bits, 2 mantissa bits. Max value ~57344. More range, less precision.</li>
</ul>

<p>The standard recipe: <code>e4m3</code> for forward activations and weights, <code>e5m2</code> for backward gradients. Tensor cores accept fp8 inputs and produce fp32 accumulators. With careful per-tensor scaling (similar in spirit to GradScaler but per-tensor and per-direction), training can run at near-bf16 quality at half the memory.</p>

<p>You don't typically write fp8 code by hand — libraries like NVIDIA's <em>TransformerEngine</em>, <em>torchao</em>, and increasingly PyTorch core handle the casting and scaling. The recipe is still being refined; it works well for inference and is becoming usable for training. We'll see fp8 quantization for inference deployment in M26.</p>

<div class="ndq">
<h4>About mixed precision</h4>

<p class="q">My validation loss is slightly worse than my training loss. Is mixed precision the cause?</p>
<p class="a">Almost certainly not. Mixed precision (especially bf16) tends to give nearly identical results to fp32 for transformers — usually within 0.1% on val metrics. The train-vs-val gap is normal generalization gap. If you're seeing a large drop (>1% absolute), suspect the loss / data / regularization choices, not the dtype.</p>

<p class="q">Should I disable autocast for evaluation?</p>
<p class="a">It depends. For research / measurement, yes — run eval in fp32 to remove any precision-related variance. For production deployment, no — eval in the same precision as training, since that's what your real users will see. Most people leave eval in autocast and don't worry about it.</p>

<p class="q">Why is my model fast in fp32 with TF32 but slow when I add bf16 autocast?</p>
<p class="a">Two possibilities. (1) Your model is dominated by ops that <em>don't</em> use tensor cores (small ops, custom kernels, embedding lookups), so autocast adds cast overhead without speeding anything up. (2) You have shape mismatches that cause bf16 ops to fall off the fast tensor-core path. Profile (M13) and check kernel times before and after; if matmul kernels aren't faster in bf16, your shapes might not be tensor-core-aligned (multiples of 8 for bf16).</p>

<p class="q">Does mixed precision break torch.compile?</p>
<p class="a">No, they work together. <code>torch.compile</code> will trace through the autocast region and compile fused kernels at the right precision. We'll see the integration in M21.</p>

<p class="q">What about distributed training — any mixed-precision gotchas?</p>
<p class="a">DDP all-reduce of gradients: if your gradients are bf16, the all-reduce happens in bf16 and you can lose precision in the reduction. Most modern setups force gradients to fp32 before the reduce — it's the default behavior in PyTorch's DDP. FSDP has more knobs (M17) for trading off reduction precision and bandwidth.</p>
</div>

<h2>The complete modern recipe</h2>

<p>Here's the full mixed-precision training step you'll write a thousand times. bf16 version, the modern default:</p>

<pre><code><span class="kw">def</span> <span class="fn">train_step</span>(model, opt, scheduler, batch):
    x, y = [t.<span class="fn">to</span>(device, non_blocking=<span class="kw">True</span>) <span class="kw">for</span> t <span class="kw">in</span> batch]
    opt.<span class="fn">zero_grad</span>()

    <span class="kw">with</span> torch.<span class="fn">autocast</span>(<span class="str">'cuda'</span>, dtype=torch.bfloat16):
        out = <span class="fn">model</span>(x)
        loss = F.<span class="fn">cross_entropy</span>(out, y)

    loss.<span class="fn">backward</span>()
    torch.nn.utils.<span class="fn">clip_grad_norm_</span>(model.<span class="fn">parameters</span>(), <span class="num">1.0</span>)
    opt.<span class="fn">step</span>()
    scheduler.<span class="fn">step</span>()
    <span class="kw">return</span> loss.<span class="fn">item</span>()</code></pre>

<p>And the fp16 version for older hardware:</p>

<pre><code>scaler = torch.amp.<span class="fn">GradScaler</span>(<span class="str">'cuda'</span>)

<span class="kw">def</span> <span class="fn">train_step_fp16</span>(model, opt, scheduler, batch):
    x, y = [t.<span class="fn">to</span>(device, non_blocking=<span class="kw">True</span>) <span class="kw">for</span> t <span class="kw">in</span> batch]
    opt.<span class="fn">zero_grad</span>()

    <span class="kw">with</span> torch.<span class="fn">autocast</span>(<span class="str">'cuda'</span>, dtype=torch.float16):
        out = <span class="fn">model</span>(x)
        loss = F.<span class="fn">cross_entropy</span>(out, y)

    scaler.<span class="fn">scale</span>(loss).<span class="fn">backward</span>()
    scaler.<span class="fn">unscale_</span>(opt)
    torch.nn.utils.<span class="fn">clip_grad_norm_</span>(model.<span class="fn">parameters</span>(), <span class="num">1.0</span>)
    scaler.<span class="fn">step</span>(opt)
    scaler.<span class="fn">update</span>()
    scheduler.<span class="fn">step</span>()
    <span class="kw">return</span> loss.<span class="fn">item</span>()</code></pre>

<p>Notice how much simpler the bf16 version is. <em>Two fewer lines, no scaler object, no order-sensitive scale/unscale dance.</em> This simplicity is a real practical win on top of the numerical robustness.</p>

<h2>Code Magnets: build the bf16 training step</h2>

<p>Build the canonical bf16 mixed-precision training step. Three magnets are wrong choices.</p>

<div class="magnets">
<p>Arrange the magnets into a working step.</p>

<div class="magnet-pool">
  <span class="magnet">x, y = [t.to(device, non_blocking=True) for t in batch]</span>
  <span class="magnet">opt.zero_grad()</span>
  <span class="magnet">with torch.autocast('cuda', dtype=torch.bfloat16):</span>
  <span class="magnet">    out = model(x)</span>
  <span class="magnet">    loss = F.cross_entropy(out, y)</span>
  <span class="magnet">    loss.backward()</span>
  <span class="magnet">loss.backward()</span>
  <span class="magnet">scaler.scale(loss).backward()</span>
  <span class="magnet">torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)</span>
  <span class="magnet">opt.step(); scheduler.step()</span>
  <span class="magnet">x = x.to(torch.bfloat16)</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code>x, y = [t.<span class="fn">to</span>(device, non_blocking=<span class="kw">True</span>) <span class="kw">for</span> t <span class="kw">in</span> batch]
opt.<span class="fn">zero_grad</span>()
<span class="kw">with</span> torch.<span class="fn">autocast</span>(<span class="str">'cuda'</span>, dtype=torch.bfloat16):
    out = <span class="fn">model</span>(x)
    loss = F.<span class="fn">cross_entropy</span>(out, y)
loss.<span class="fn">backward</span>()
torch.nn.utils.<span class="fn">clip_grad_norm_</span>(model.<span class="fn">parameters</span>(), <span class="num">1.0</span>)
opt.<span class="fn">step</span>(); scheduler.<span class="fn">step</span>()</code></pre>
<p>The traps:</p>
<ul>
  <li><code>loss.backward()</code> <em>inside</em> the autocast block: backward should be outside. Autograd handles dtypes correctly when called from the outer context.</li>
  <li><code>scaler.scale(loss).backward()</code>: that's the fp16 pattern. bf16 doesn't need a GradScaler.</li>
  <li><code>x = x.to(torch.bfloat16)</code>: don't manually cast inputs. Autocast handles per-op casting; manually pre-casting fights the framework.</li>
</ul>
<p>The right structure: <strong>autocast wraps forward + loss; backward, clip, step, scheduler outside</strong>. No GradScaler needed for bf16.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each mixed-precision concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>autocast block</div>
  <div>A. fp32 working copy of weights, used by the optimizer for accurate updates.</div>

  <div>Master fp32 copy</div>
  <div>B. Per-op router that casts inputs based on op type — matmul to bf16, softmax to fp32.</div>

  <div>GradScaler</div>
  <div>C. Hardware fp32 matmul that internally uses 10-bit mantissa for ~2× throughput.</div>

  <div>TF32</div>
  <div>D. Multiplies loss by K to keep fp16 gradients above the underflow threshold.</div>

  <div>bf16</div>
  <div>E. Same range as fp32, fewer mantissa bits, no scaler needed — modern training default.</div>

  <div>fp16</div>
  <div>F. Tiny range (max 65504), needs GradScaler, used only on older hardware.</div>

  <div>fp8 (e4m3)</div>
  <div>G. 1-byte format for forward activations on H100+, used via TransformerEngine etc.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>autocast block</strong> → B<br>
<strong>Master fp32 copy</strong> → A<br>
<strong>GradScaler</strong> → D<br>
<strong>TF32</strong> → C<br>
<strong>bf16</strong> → E<br>
<strong>fp16</strong> → F<br>
<strong>fp8 (e4m3)</strong> → G
</p>
<p>The mental shortcut: <em>autocast routes per-op, master copy preserves precision, GradScaler is the fp16 underflow fix, TF32 is silent fp32 acceleration, bf16 = modern default, fp16 = older HW with scaler, fp8 = newest H100+ feature</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team's training in fp16 mode trains for ~50 steps then loss goes to NaN. They use autocast + GradScaler. They check the GradScaler's <code>get_scale()</code> and see it's 65536. What's likely wrong?</p>
<details class="answer"><summary>show answer</summary>
<p>The scaler hasn't yet halved its scale, so it didn't catch an overflow that propagated. Either (a) the model's activations are spiking high before any gradient is even computed (so loss becomes Inf inside forward, not in grads), or (b) there's a numerical bug in a custom op that doesn't go through autocast. Quick test: lower the initial scaler value (<code>GradScaler(init_scale=2**14)</code>), and add NaN-check forward hooks (Module text) to find which layer's output goes Inf first. If even fp32 training is fine but fp16 fails, the answer is "switch to bf16" — fp16's tiny range is the actual issue.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Why is the loss computed inside the autocast block but the backward called outside?</p>
<details class="answer"><summary>show answer</summary>
<p>Two reasons. (1) <code>F.cross_entropy</code> is on the autocast "fp32-preserving" list — it'll cast its inputs to fp32 internally for stability anyway, so you might as well let it see the bf16 logits and handle the cast at the boundary. (2) Backward doesn't need to be inside autocast because autograd <em>already recorded the dtype at forward time</em> in each grad_fn. When backward runs, each op's backward uses whatever dtype was used in the corresponding forward, regardless of the current autocast state. Putting backward inside autocast would have no effect — it's structured logic, not a global mode flag.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> You're training in bf16 and want to verify nothing meaningful is being lost vs fp32. Outline a quick experiment.</p>
<details class="answer"><summary>show answer</summary>
<p>Run two short experiments with identical seeds: one with autocast disabled (full fp32 / TF32), one with autocast bf16. Compare loss curves and final validation metric over say 1000 steps. Both should match within ~0.1%. If bf16's curve diverges noticeably, suspect: a custom op that's numerically unstable in low precision; a manual fp32 → bf16 cast somewhere fighting autocast; or genuinely numerically-sensitive task (very deep network, very long sequences). For 99% of transformer training, bf16 is indistinguishable from fp32 in the noise.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> You see significant matmul speedup from fp32-with-TF32 to bf16 autocast on an A100, but virtually none on a T4. Why?</p>
<details class="answer"><summary>show answer</summary>
<p>The T4 (Volta-era) doesn't have bf16 tensor cores. It has fp16 tensor cores and full fp32 paths but no bf16 acceleration — bf16 on T4 falls back to fp32 emulation, with overhead from the casting. So on T4, bf16 is slower than fp32, not faster. On T4 you'd want fp16 + GradScaler, or fp32 alone. This is the kind of thing that profiling (M13) catches immediately — kernel times don't change as expected when you flip the dtype, hardware support is the answer.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li><strong>Mixed precision is a per-region dtype contract</strong>. Forward in bf16, normalization/softmax/loss in fp32, backward grads in fp32, optimizer step on fp32 master copy.</li>
  <li><code>torch.autocast('cuda', dtype=torch.bfloat16)</code> wraps the forward + loss. <strong>Backward and step go outside</strong> — autograd handles dtypes from recorded forward state.</li>
  <li><strong>Autocast's three op categories</strong>: lower-precision (matmul, conv, attention) → cast to bf16/fp16; fp32-preserving (softmax, logsumexp, cross_entropy) → cast to fp32; promoting (mixed-input arithmetic) → use highest input precision.</li>
  <li><strong>The master fp32 copy</strong> exists because bf16's 7-bit mantissa would round away tiny optimizer updates. The master is what the optimizer reads/writes; cast back to bf16 for the next forward.</li>
  <li><strong>bf16 vs fp16</strong>: same speed on tensor cores. bf16 has fp32-equivalent range, no underflow/overflow at gradient scale, no GradScaler needed. fp16 has tiny range (max 65504), needs GradScaler.</li>
  <li><strong>Use bf16 if your hardware supports it</strong> (Ampere/A100/H100/TPUs/MI200+). Only fall back to fp16 on older hardware.</li>
  <li><code>torch.amp.GradScaler('cuda')</code> is the fp16 safety net: scales loss by K (initially 65536) before backward, unscales before step, halves K on overflow, doubles after many steps without overflow.</li>
  <li><strong>Order matters in fp16</strong>: <code>scaler.scale(loss).backward()</code> → <code>scaler.unscale_(opt)</code> → <code>clip_grad_norm_</code> → <code>scaler.step(opt)</code> → <code>scaler.update()</code>.</li>
  <li><strong>TF32 is silent fp32 acceleration</strong> on Ampere+. Default on. Internally truncates fp32 mantissa to 10 bits for matmul speedup.</li>
  <li><strong>NaN diagnosis</strong>: bf16 is mostly NaN-proof. fp16 NaN comes from underflow (no scaler) or overflow (LR too high, activations spiking). Use forward hooks to find first NaN; <code>set_detect_anomaly(True)</code> for backward NaN with stack trace (slow — debug only).</li>
  <li><strong>fp8</strong> on H100+: e4m3 for forward, e5m2 for backward. Use via TransformerEngine / torchao; you don't typically wire it manually.</li>
  <li>The reflex: when training is slow on modern hardware, the first thing to try is <code>autocast('cuda', dtype=torch.bfloat16)</code>. One line, ~2× speedup, no quality loss for transformers.</li>
</ul>
</div>

<p>That closes Part V. You now have the predictive tools (memory budget, time budget) and the headline acceleration (mixed precision). The next part takes the same training step and runs it across multiple GPUs — same model, same step, same loss, but with the hardware multiplied. Module 15 opens Part VI by introducing the communication primitives that distributed training is built on: all-reduce, all-gather, scatter, broadcast.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">14</span>
  <span>Mixed precision: autocast & GradScaler</span>
</div>
"""

emit("14_mixed_precision", "Module 14 — Mixed precision: autocast & GradScaler", BODY)
