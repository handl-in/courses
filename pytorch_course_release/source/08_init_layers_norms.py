#!/usr/bin/env python3
"""Module 08: Initialization, layers & norms — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part III · Module 08</div>
  <h1 class="module-title">Initialization, <em>layers &amp; norms</em></h1>
  <p class="module-sub">— why your network dies at layer 20 if you use the wrong init, and the single axis decision that distinguishes BatchNorm from LayerNorm from RMSNorm</p>
</div>

<p>You have <code>nn.Module</code>. Now you fill it. This module is about the small but vicious choices inside every layer: <em>how do I initialize weights so the network doesn't immediately collapse?</em> and <em>what kind of normalization stops activations from drifting into NaN over deep stacks?</em></p>

<p>These are not aesthetic choices. A deep network with default-PyTorch-Linear init and no normalization will train for a few steps and die. A correctly-initialized network with the right normalization can train for a million steps and stay stable. The difference is roughly six lines of code and zero added compute. Worth understanding.</p>

<div class="keyidea">
Initialization and normalization solve the same underlying problem: <strong>keeping the magnitude of activations and gradients roughly constant across layers</strong>. Init does it at step 0 by choosing weight scales correctly. Normalization does it at every step by rescaling activations. They're complementary, not redundant — modern transformers use both.
</div>

<h2>Two new faces</h2>

<div class="character" style="--c: #1f5f5b;">
  <div class="avatar" style="background: #1f5f5b; color: #fff;">σ</div>
  <div>
    <p class="who">Init Scheme</p>
    <p class="name">"I'm the reason your network doesn't die at layer 20."</p>
    <p class="says">When you stack 20 randomly-initialized linear layers, the activations either explode (variance grows by a factor each layer) or collapse to zero (variance shrinks by a factor each layer). Either way, gradients vanish or explode and training fails. My job is to choose weight scales so that <em>the variance of activations stays roughly constant from input to output</em>. Xavier, Kaiming, GPT-style — they're all answers to the same question, with slightly different assumptions about what comes next (which activation, which depth, which residual structure).</p>
  </div>
</div>

<div class="character" style="--c: #b85a6c;">
  <div class="avatar" style="background: #b85a6c; color: #fff;">N</div>
  <div>
    <p class="who">Norm</p>
    <p class="name">"I subtract the mean and divide by the std. The only question is <em>over what</em>."</p>
    <p class="says">Whether I'm BatchNorm, LayerNorm, GroupNorm, or RMSNorm — the formula is essentially the same: <code>(x - mean) / sqrt(var + ε) * weight + bias</code>. The whole drama is which dimensions I average over. BatchNorm averages over the batch and spatial dims (per-channel statistics). LayerNorm averages over the feature dim (per-token statistics). GroupNorm splits the channels into groups and averages within each group. RMSNorm skips the mean entirely and just divides by the RMS. Pick wisely — the choice has consequences.</p>
  </div>
</div>

<h2>Why initialization matters: the variance-preservation argument</h2>

<p>Here's the simplest possible setup. A 20-layer linear network with no nonlinearities — just <code>x → W₁x → W₂(W₁x) → ...</code>. If <code>W</code> has entries drawn from <code>N(0, σ²)</code>, what variance does the output have, given input variance 1?</p>

<p>For a single layer: <code>y = Wx</code>, where <code>W</code> is shape <code>(D, D)</code>. Each output <code>y[i] = sum(W[i, :] * x)</code>. The variance of that sum is <code>D · σ²</code> (assuming x is unit-variance and zero-mean). So one layer multiplies the variance by <code>D · σ²</code>.</p>

<p>After 20 layers, the variance is multiplied by <code>(D · σ²)²⁰</code>. For <code>D = 512</code> and <code>σ = 0.1</code> (a "reasonable" guess), that's <code>(5.12)²⁰ ≈ 10¹⁴</code>. Activations explode. Gradients explode. Loss is NaN by step 1.</p>

<p>For variance to stay constant, we need <code>D · σ² = 1</code>, i.e. <code>σ = 1/√D</code>. That's the entire idea. Xavier and Kaiming are refinements that account for the activation function and the forward/backward asymmetry.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 290" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrV" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
  </defs>
  <text x="370" y="24" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Activation variance through a 20-layer linear network</text>

  <!-- top: bad init -->
  <text x="20" y="60" font-size="12" font-weight="700" fill="#c1502e">σ = 0.1, D = 512 — wrong scale</text>
  <g transform="translate(20, 75)">
    <g font-family="'IBM Plex Mono', monospace" font-size="12" text-anchor="middle">
      <rect x="0"   y="0" width="80" height="40" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/><text x="40"  y="18" fill="#1a1612">var=1</text><text x="40" y="32" font-size="10" fill="#1a1612">layer 0</text>
      <rect x="100" y="0" width="80" height="40" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/><text x="140" y="18" fill="#1a1612">var≈5.12</text><text x="140" y="32" font-size="10" fill="#1a1612">layer 1</text>
      <rect x="200" y="0" width="80" height="40" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/><text x="240" y="18" fill="#1a1612">var≈26</text><text x="240" y="32" font-size="10" fill="#1a1612">layer 2</text>
      <rect x="300" y="0" width="80" height="40" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/><text x="340" y="18" fill="#1a1612">var≈690</text><text x="340" y="32" font-size="10" fill="#1a1612">layer 4</text>
      <rect x="400" y="0" width="100" height="40" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/><text x="450" y="18" fill="#1a1612">var≈5e5</text><text x="450" y="32" font-size="10" fill="#1a1612">layer 8</text>
      <rect x="520" y="0" width="100" height="40" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/><text x="570" y="18" fill="#1a1612">var≈10¹⁴</text><text x="570" y="32" font-size="10" fill="#1a1612">layer 20</text>
      <rect x="640" y="0" width="80" height="40" fill="#1a1612" stroke="#c1502e" stroke-width="1.5"/><text x="680" y="20" fill="#fff">NaN</text><text x="680" y="34" font-size="10" fill="#fff">soon</text>
    </g>
  </g>

  <!-- bottom: correct init -->
  <text x="20" y="160" font-size="12" font-weight="700" fill="#1f5f5b">σ = 1/√D ≈ 0.044 — correct scale</text>
  <g transform="translate(20, 175)">
    <g font-family="'IBM Plex Mono', monospace" font-size="12" text-anchor="middle">
      <rect x="0"   y="0" width="80" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/><text x="40"  y="18" fill="#1a1612">var=1</text><text x="40" y="32" font-size="10" fill="#1a1612">layer 0</text>
      <rect x="100" y="0" width="80" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/><text x="140" y="18" fill="#1a1612">var≈1</text><text x="140" y="32" font-size="10" fill="#1a1612">layer 1</text>
      <rect x="200" y="0" width="80" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/><text x="240" y="18" fill="#1a1612">var≈1</text><text x="240" y="32" font-size="10" fill="#1a1612">layer 2</text>
      <rect x="300" y="0" width="80" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/><text x="340" y="18" fill="#1a1612">var≈1</text><text x="340" y="32" font-size="10" fill="#1a1612">layer 4</text>
      <rect x="400" y="0" width="100" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/><text x="450" y="18" fill="#1a1612">var≈1</text><text x="450" y="32" font-size="10" fill="#1a1612">layer 8</text>
      <rect x="520" y="0" width="100" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/><text x="570" y="18" fill="#1a1612">var≈1</text><text x="570" y="32" font-size="10" fill="#1a1612">layer 20</text>
      <rect x="640" y="0" width="80" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/><text x="680" y="20" fill="#1a1612">trains</text><text x="680" y="34" font-size="10" fill="#1a1612">fine</text>
    </g>
  </g>

  <text x="370" y="265" font-family="'Caveat', cursive" font-size="22" fill="#c1502e" text-anchor="middle">init is the cheapest training stability hack ever invented</text>
</svg>
</div>

<h2>The init schemes you'll actually use</h2>

<p>Three schemes cover ~95% of real models. Their formulas differ in the choice of <code>fan_in</code> (number of input units) vs <code>fan_out</code> (output units), and in the constant that accounts for the activation function.</p>

<div class="table-wrap">
<table>
<caption>The init schemes that earn their keep</caption>
<thead><tr><th>Scheme</th><th>Std formula</th><th>Designed for</th><th>Use it for…</th></tr></thead>
<tbody>
<tr><td><strong>Xavier (Glorot)</strong></td><td><code>σ = √(2 / (fan_in + fan_out))</code></td><td>Symmetric activations (tanh, sigmoid)</td><td>Older architectures; most LayerNorm-stabilized models do fine with it</td></tr>
<tr><td><strong>Kaiming (He)</strong></td><td><code>σ = √(2 / fan_in)</code></td><td>ReLU-family activations (the 2× compensates for ReLU killing half the variance)</td><td>CNNs, MLPs with ReLU/GELU before the next layer</td></tr>
<tr><td><strong>GPT-style</strong></td><td><code>σ = 0.02</code> for most, <code>σ = 0.02/√(2·n_layers)</code> for residual projections</td><td>Deep transformer stacks with residual connections</td><td>LLaMA, GPT, modern transformer training</td></tr>
<tr><td><strong>Orthogonal</strong></td><td>Random orthogonal matrix scaled by gain</td><td>RNNs, situations where you want strict isometry at init</td><td>Sequence models with long unrolling; less common today</td></tr>
</tbody>
</table>
</div>

<p>Two things worth saying about the "GPT-style" entry, because it's the modern default. (1) Most weights are just <code>N(0, 0.02²)</code> — surprisingly small, surprisingly fixed regardless of dim. (2) For weights at the end of a residual block (the projection that adds back to the residual stream), the std gets divided by <code>√(2·n_layers)</code>. This compensates for the fact that residuals add up over depth — without the correction, deeper transformers blow up their residual stream. Karpathy's nanoGPT made this widely known.</p>

<h3>Doing init in PyTorch</h3>

<pre><code><span class="kw">import</span> torch.nn <span class="kw">as</span> nn

<span class="com"># PyTorch's default Linear init is roughly Kaiming uniform — usually fine</span>
<span class="com"># but you can override:</span>

<span class="kw">def</span> <span class="fn">init_weights</span>(module):
    <span class="kw">if</span> <span class="fn">isinstance</span>(module, nn.Linear):
        nn.init.<span class="fn">kaiming_normal_</span>(module.weight, nonlinearity=<span class="str">'relu'</span>)
        <span class="kw">if</span> module.bias <span class="kw">is not</span> <span class="kw">None</span>:
            nn.init.<span class="fn">zeros_</span>(module.bias)
    <span class="kw">elif</span> <span class="fn">isinstance</span>(module, nn.Embedding):
        nn.init.<span class="fn">normal_</span>(module.weight, mean=<span class="num">0</span>, std=<span class="num">0.02</span>)
    <span class="kw">elif</span> <span class="fn">isinstance</span>(module, nn.LayerNorm):
        nn.init.<span class="fn">ones_</span>(module.weight)
        nn.init.<span class="fn">zeros_</span>(module.bias)

model.<span class="fn">apply</span>(init_weights)        <span class="com"># walks every module in the tree</span></code></pre>

<p>That's the canonical pattern: define a function that handles each module type, call <code>model.apply(fn)</code> to walk the tree. Note the trailing underscore — these are <em>in-place</em> initializers (M1's underscore convention).</p>

<div class="warn">
<strong>Bias initialization matters.</strong> Always initialize biases to zero (or near-zero) unless you have a specific reason not to. Non-zero bias init breaks most variance-preservation arguments. The one common exception: the bias of the last layer in a binary classification model is sometimes init'd to <code>log(p/(1-p))</code> where p is the class prior — speeds up training when classes are imbalanced.
</div>

<h3>The GPT-style init in code</h3>

<pre><code><span class="kw">def</span> <span class="fn">init_gpt_style</span>(model, n_layers):
    std = <span class="num">0.02</span>
    proj_std = <span class="num">0.02</span> / (<span class="num">2</span> * n_layers) ** <span class="num">0.5</span>

    <span class="kw">for</span> name, p <span class="kw">in</span> model.<span class="fn">named_parameters</span>():
        <span class="kw">if</span> p.dim() &gt;= <span class="num">2</span>:
            <span class="com"># Most weights: N(0, 0.02²)</span>
            nn.init.<span class="fn">normal_</span>(p, mean=<span class="num">0.0</span>, std=std)
            <span class="com"># Residual-block projections: scale down by √(2*n_layers)</span>
            <span class="kw">if</span> name.<span class="fn">endswith</span>(<span class="str">'attn.out_proj.weight'</span>) <span class="kw">or</span> name.<span class="fn">endswith</span>(<span class="str">'mlp.down_proj.weight'</span>):
                nn.init.<span class="fn">normal_</span>(p, mean=<span class="num">0.0</span>, std=proj_std)
        <span class="kw">elif</span> <span class="str">'bias'</span> <span class="kw">in</span> name:
            nn.init.<span class="fn">zeros_</span>(p)
        <span class="kw">elif</span> p.dim() == <span class="num">1</span>:
            <span class="com"># LayerNorm weights and similar 1-D things — init to 1</span>
            nn.init.<span class="fn">ones_</span>(p)</code></pre>

<p>The naming convention on the projections is brittle (depends on your layer naming), so most real codebases identify them via <code>isinstance</code> checks or special module subclasses. The principle is what matters.</p>

<h2>Normalization layers: the axis decision</h2>

<p>This is where most engineers get hand-wavy. Let me be precise.</p>

<p>Every normalization layer does the same arithmetic: subtract the mean, divide by the standard deviation, scale by a learned weight, shift by a learned bias. <em>The only thing that changes across BN / LN / GN / IN / RMSNorm is which axes the mean and std are computed over.</em></p>

<p>Consider a tensor of shape <code>(N, C, H, W)</code> — batch, channels, height, width — typical for CNNs. Or <code>(B, T, D)</code> — batch, sequence, hidden dim — typical for transformers. The choice of axes determines the layer's character.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 360" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="24" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Same tensor (N, C, H, W). Different axes get normalized.</text>

  <!-- BatchNorm -->
  <g transform="translate(20, 50)">
    <text x="80" y="14" font-size="12" font-weight="700" fill="#c1502e" text-anchor="middle">BatchNorm</text>
    <text x="80" y="30" font-size="10" fill="#1a1612" text-anchor="middle">over (N, H, W) per channel</text>
    <!-- 4 channel slabs, with N spread across, H,W per slab. We'll show 4 columns (channels), each with 3 rows (samples), and shade dim. -->
    <g font-family="'IBM Plex Mono', monospace" font-size="9" text-anchor="middle">
      <!-- batch dim: 3 rows -->
      <!-- channel dim: 4 cols -->
      <rect x="0"  y="40" width="32" height="22" fill="#fcecec" stroke="#c1502e"/><text x="16" y="55" fill="#1a1612">N1</text>
      <rect x="33" y="40" width="32" height="22" fill="#fff" stroke="#c1502e"/><text x="49" y="55" fill="#1a1612">N1</text>
      <rect x="66" y="40" width="32" height="22" fill="#fff" stroke="#c1502e"/><text x="82" y="55" fill="#1a1612">N1</text>
      <rect x="99" y="40" width="32" height="22" fill="#fff" stroke="#c1502e"/><text x="115" y="55" fill="#1a1612">N1</text>

      <rect x="0"  y="63" width="32" height="22" fill="#fcecec" stroke="#c1502e"/><text x="16" y="78" fill="#1a1612">N2</text>
      <rect x="33" y="63" width="32" height="22" fill="#fff" stroke="#c1502e"/><text x="49" y="78" fill="#1a1612">N2</text>
      <rect x="66" y="63" width="32" height="22" fill="#fff" stroke="#c1502e"/><text x="82" y="78" fill="#1a1612">N2</text>
      <rect x="99" y="63" width="32" height="22" fill="#fff" stroke="#c1502e"/><text x="115" y="78" fill="#1a1612">N2</text>

      <rect x="0"  y="86" width="32" height="22" fill="#fcecec" stroke="#c1502e"/><text x="16" y="101" fill="#1a1612">N3</text>
      <rect x="33" y="86" width="32" height="22" fill="#fff" stroke="#c1502e"/><text x="49" y="101" fill="#1a1612">N3</text>
      <rect x="66" y="86" width="32" height="22" fill="#fff" stroke="#c1502e"/><text x="82" y="101" fill="#1a1612">N3</text>
      <rect x="99" y="86" width="32" height="22" fill="#fff" stroke="#c1502e"/><text x="115" y="101" fill="#1a1612">N3</text>
    </g>
    <text x="16" y="125" font-size="9" fill="#c1502e" text-anchor="middle">C1</text>
    <text x="49" y="125" font-size="9" fill="#1a1612" text-anchor="middle">C2</text>
    <text x="82" y="125" font-size="9" fill="#1a1612" text-anchor="middle">C3</text>
    <text x="115" y="125" font-size="9" fill="#1a1612" text-anchor="middle">C4</text>
    <text x="80" y="148" font-size="10" fill="#1a1612" text-anchor="middle" font-style="italic">stats per channel,</text>
    <text x="80" y="160" font-size="10" fill="#1a1612" text-anchor="middle" font-style="italic">across batch + spatial</text>
  </g>

  <!-- LayerNorm -->
  <g transform="translate(195, 50)">
    <text x="80" y="14" font-size="12" font-weight="700" fill="#1f5f5b" text-anchor="middle">LayerNorm</text>
    <text x="80" y="30" font-size="10" fill="#1a1612" text-anchor="middle">over (C, H, W) per sample</text>
    <g font-family="'IBM Plex Mono', monospace" font-size="9" text-anchor="middle">
      <rect x="0"  y="40" width="32" height="22" fill="#d4ecc8" stroke="#1f5f5b"/><text x="16" y="55" fill="#1a1612">N1</text>
      <rect x="33" y="40" width="32" height="22" fill="#d4ecc8" stroke="#1f5f5b"/><text x="49" y="55" fill="#1a1612">N1</text>
      <rect x="66" y="40" width="32" height="22" fill="#d4ecc8" stroke="#1f5f5b"/><text x="82" y="55" fill="#1a1612">N1</text>
      <rect x="99" y="40" width="32" height="22" fill="#d4ecc8" stroke="#1f5f5b"/><text x="115" y="55" fill="#1a1612">N1</text>

      <rect x="0"  y="63" width="32" height="22" fill="#fff" stroke="#1f5f5b"/><text x="16" y="78" fill="#1a1612">N2</text>
      <rect x="33" y="63" width="32" height="22" fill="#fff" stroke="#1f5f5b"/><text x="49" y="78" fill="#1a1612">N2</text>
      <rect x="66" y="63" width="32" height="22" fill="#fff" stroke="#1f5f5b"/><text x="82" y="78" fill="#1a1612">N2</text>
      <rect x="99" y="63" width="32" height="22" fill="#fff" stroke="#1f5f5b"/><text x="115" y="78" fill="#1a1612">N2</text>

      <rect x="0"  y="86" width="32" height="22" fill="#fff" stroke="#1f5f5b"/><text x="16" y="101" fill="#1a1612">N3</text>
      <rect x="33" y="86" width="32" height="22" fill="#fff" stroke="#1f5f5b"/><text x="49" y="101" fill="#1a1612">N3</text>
      <rect x="66" y="86" width="32" height="22" fill="#fff" stroke="#1f5f5b"/><text x="82" y="101" fill="#1a1612">N3</text>
      <rect x="99" y="86" width="32" height="22" fill="#fff" stroke="#1f5f5b"/><text x="115" y="101" fill="#1a1612">N3</text>
    </g>
    <text x="80" y="148" font-size="10" fill="#1a1612" text-anchor="middle" font-style="italic">stats per sample,</text>
    <text x="80" y="160" font-size="10" fill="#1a1612" text-anchor="middle" font-style="italic">across all features</text>
  </g>

  <!-- GroupNorm -->
  <g transform="translate(370, 50)">
    <text x="80" y="14" font-size="12" font-weight="700" fill="#d4a017" text-anchor="middle">GroupNorm</text>
    <text x="80" y="30" font-size="10" fill="#1a1612" text-anchor="middle">over (group of C, H, W) per sample</text>
    <g font-family="'IBM Plex Mono', monospace" font-size="9" text-anchor="middle">
      <rect x="0"  y="40" width="32" height="22" fill="#fff5d8" stroke="#d4a017"/><text x="16" y="55" fill="#1a1612">N1</text>
      <rect x="33" y="40" width="32" height="22" fill="#fff5d8" stroke="#d4a017"/><text x="49" y="55" fill="#1a1612">N1</text>
      <rect x="66" y="40" width="32" height="22" fill="#fff" stroke="#d4a017"/><text x="82" y="55" fill="#1a1612">N1</text>
      <rect x="99" y="40" width="32" height="22" fill="#fff" stroke="#d4a017"/><text x="115" y="55" fill="#1a1612">N1</text>

      <rect x="0"  y="63" width="32" height="22" fill="#fff" stroke="#d4a017"/><text x="16" y="78" fill="#1a1612">N2</text>
      <rect x="33" y="63" width="32" height="22" fill="#fff" stroke="#d4a017"/><text x="49" y="78" fill="#1a1612">N2</text>
      <rect x="66" y="63" width="32" height="22" fill="#fff" stroke="#d4a017"/><text x="82" y="78" fill="#1a1612">N2</text>
      <rect x="99" y="63" width="32" height="22" fill="#fff" stroke="#d4a017"/><text x="115" y="78" fill="#1a1612">N2</text>

      <rect x="0"  y="86" width="32" height="22" fill="#fff" stroke="#d4a017"/><text x="16" y="101" fill="#1a1612">N3</text>
      <rect x="33" y="86" width="32" height="22" fill="#fff" stroke="#d4a017"/><text x="49" y="101" fill="#1a1612">N3</text>
      <rect x="66" y="86" width="32" height="22" fill="#fff" stroke="#d4a017"/><text x="82" y="101" fill="#1a1612">N3</text>
      <rect x="99" y="86" width="32" height="22" fill="#fff" stroke="#d4a017"/><text x="115" y="101" fill="#1a1612">N3</text>
    </g>
    <text x="80" y="148" font-size="10" fill="#1a1612" text-anchor="middle" font-style="italic">stats per group of channels,</text>
    <text x="80" y="160" font-size="10" fill="#1a1612" text-anchor="middle" font-style="italic">per sample (here: groups of 2)</text>
  </g>

  <!-- InstanceNorm -->
  <g transform="translate(545, 50)">
    <text x="80" y="14" font-size="12" font-weight="700" fill="#b85a6c" text-anchor="middle">InstanceNorm</text>
    <text x="80" y="30" font-size="10" fill="#1a1612" text-anchor="middle">over (H, W) per (sample, channel)</text>
    <g font-family="'IBM Plex Mono', monospace" font-size="9" text-anchor="middle">
      <rect x="0"  y="40" width="32" height="22" fill="#ffd5dc" stroke="#b85a6c"/><text x="16" y="55" fill="#1a1612">N1</text>
      <rect x="33" y="40" width="32" height="22" fill="#fff" stroke="#b85a6c"/><text x="49" y="55" fill="#1a1612">N1</text>
      <rect x="66" y="40" width="32" height="22" fill="#fff" stroke="#b85a6c"/><text x="82" y="55" fill="#1a1612">N1</text>
      <rect x="99" y="40" width="32" height="22" fill="#fff" stroke="#b85a6c"/><text x="115" y="55" fill="#1a1612">N1</text>

      <rect x="0"  y="63" width="32" height="22" fill="#fff" stroke="#b85a6c"/><text x="16" y="78" fill="#1a1612">N2</text>
      <rect x="33" y="63" width="32" height="22" fill="#fff" stroke="#b85a6c"/><text x="49" y="78" fill="#1a1612">N2</text>
      <rect x="66" y="63" width="32" height="22" fill="#fff" stroke="#b85a6c"/><text x="82" y="78" fill="#1a1612">N2</text>
      <rect x="99" y="63" width="32" height="22" fill="#fff" stroke="#b85a6c"/><text x="115" y="78" fill="#1a1612">N2</text>

      <rect x="0"  y="86" width="32" height="22" fill="#fff" stroke="#b85a6c"/><text x="16" y="101" fill="#1a1612">N3</text>
      <rect x="33" y="86" width="32" height="22" fill="#fff" stroke="#b85a6c"/><text x="49" y="101" fill="#1a1612">N3</text>
      <rect x="66" y="86" width="32" height="22" fill="#fff" stroke="#b85a6c"/><text x="82" y="101" fill="#1a1612">N3</text>
      <rect x="99" y="86" width="32" height="22" fill="#fff" stroke="#b85a6c"/><text x="115" y="101" fill="#1a1612">N3</text>
    </g>
    <text x="80" y="148" font-size="10" fill="#1a1612" text-anchor="middle" font-style="italic">stats per (sample, channel)</text>
    <text x="80" y="160" font-size="10" fill="#1a1612" text-anchor="middle" font-style="italic">— smallest scope</text>
  </g>

  <text x="370" y="220" font-family="'Caveat', cursive" font-size="22" fill="#c1502e" text-anchor="middle">colored cells = "averaged together for one mean and one std"</text>
  <text x="370" y="245" font-family="'Caveat', cursive" font-size="20" fill="#1a1612" text-anchor="middle">BN: same channel across batch &middot; LN: same sample across features</text>
  <text x="370" y="265" font-family="'Caveat', cursive" font-size="20" fill="#1a1612" text-anchor="middle">GN: middle ground &middot; IN: per (sample, channel) only</text>
</svg>
</div>

<p>Read the diagram carefully — the colored cells are the ones that get averaged together to produce <em>one</em> mean and <em>one</em> standard deviation. Different sets of cells produce different means, hence different normalization characters.</p>

<h3>BatchNorm: per-channel statistics across the batch</h3>

<p>For input <code>(N, C, H, W)</code>: compute mean and var <em>per channel</em>, averaged over the batch and the spatial dims. So you get one (mean, var) pair per channel — <code>C</code> means and <code>C</code> vars.</p>

<pre><code><span class="kw">import</span> torch.nn <span class="kw">as</span> nn
bn = nn.<span class="fn">BatchNorm2d</span>(num_features=<span class="num">64</span>)   <span class="com"># expects (N, 64, H, W)</span></code></pre>

<p>Two key behaviors:</p>

<ul>
  <li><strong>Train mode</strong>: compute mean/var from the current batch. Update running mean/var (the buffers we registered in M7) as exponential moving averages.</li>
  <li><strong>Eval mode</strong>: use the running mean/var, ignore the current batch. This is what gives you stable inference even on batch size 1.</li>
</ul>

<p>BN's strength: enables training of very deep CNNs that wouldn't otherwise converge. BN's weakness: <strong>requires batch size &gt;= 8 or so to estimate stable per-channel statistics</strong>. Small-batch training (memory-constrained) breaks it. Distributed training requires sync (Module 16). Sequence models hate it because the batch axis is heterogeneous.</p>

<h3>LayerNorm: per-sample statistics across the features</h3>

<p>For input <code>(B, T, D)</code> (transformer-style): compute mean and var <em>per (batch, position)</em>, across the feature dim. So you get <code>B × T</code> (mean, var) pairs.</p>

<pre><code>ln = nn.<span class="fn">LayerNorm</span>(<span class="fn">normalized_shape</span>=<span class="num">512</span>)   <span class="com"># normalize over the last dim of size 512</span>
y = <span class="fn">ln</span>(x)                              <span class="com"># x: (B, T, 512)</span></code></pre>

<p>LN doesn't use the batch dim at all, so batch size is irrelevant. There's no train/eval split for LN — it does the same thing in both modes. This makes it the natural fit for transformers: works at batch size 1, doesn't need running stats, doesn't break under distributed training.</p>

<p>The <code>normalized_shape</code> argument is subtle: it's <em>the shape of the trailing dims to normalize over</em>. For a transformer with shape <code>(B, T, D)</code>, you pass <code>D</code> (or <code>(D,)</code>). For multi-axis normalization (rare), you can pass <code>(C, H, W)</code> to normalize across all three.</p>

<h3>RMSNorm: drop the mean, keep the rescaling</h3>

<p>The simplest and newest of the bunch. Identical to LayerNorm except: <em>don't subtract the mean, just divide by the RMS (root-mean-square)</em>. There's also no learned bias.</p>

<pre><code><span class="kw">class</span> <span class="ty">RMSNorm</span>(nn.Module):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, dim, eps=<span class="num">1e-6</span>):
        <span class="fn">super</span>().<span class="fn">__init__</span>()
        self.weight = nn.<span class="fn">Parameter</span>(torch.<span class="fn">ones</span>(dim))
        self.eps = eps

    <span class="kw">def</span> <span class="fn">forward</span>(self, x):
        rms = x.<span class="fn">pow</span>(<span class="num">2</span>).<span class="fn">mean</span>(dim=-<span class="num">1</span>, keepdim=<span class="kw">True</span>).<span class="fn">add</span>(self.eps).<span class="fn">sqrt</span>()
        <span class="kw">return</span> x / rms * self.weight</code></pre>

<p>Why does this work? Empirically, the mean-subtraction in LayerNorm doesn't add much when activations are already roughly zero-mean (which they are in deep networks with skip connections). Skipping the mean saves memory and one reduction. LLaMA, Mistral, and most modern LLMs use RMSNorm. PyTorch added <code>nn.RMSNorm</code> in 2.4.</p>

<h3>GroupNorm: the middle ground</h3>

<p>For input <code>(N, C, H, W)</code>: split <code>C</code> into <code>G</code> groups, compute mean/var per <code>(sample, group)</code>. So you get <code>N × G</code> (mean, var) pairs.</p>

<pre><code>gn = nn.<span class="fn">GroupNorm</span>(num_groups=<span class="num">32</span>, num_channels=<span class="num">512</span>)
<span class="com"># Splits 512 channels into 32 groups of 16</span></code></pre>

<p>GroupNorm is what you reach for when you want the batch-independence of LN but with the channel-grouping flavor of CNNs. It works well in image generation (Stable Diffusion uses it) and in cases where batch size is small or variable.</p>

<div class="ndq">
<h4>About normalization choices</h4>

<p class="q">My CNN trains fine with BN. Should I switch to GN?</p>
<p class="a">Only if your batch size is small (&lt;8) or you need batch-independent inference (e.g., real-time inference at batch size 1 where the running stats don't fit your data distribution). For standard training with batch sizes of 32+, BN is faster and usually slightly better.</p>

<p class="q">Why does every modern transformer use LN or RMSNorm and not BN?</p>
<p class="a">Three reasons. (1) Sequence models often have variable-length sequences in a batch — BN's per-channel statistics get contaminated by padding. (2) Inference at batch size 1 is common — BN needs running stats; LN doesn't. (3) Transformers train at huge effective batch sizes via accumulation, where each micro-batch's statistics aren't representative; LN sidesteps the issue entirely.</p>

<p class="q">Pre-norm vs post-norm — what's that about?</p>
<p class="a">Where you place the LN relative to the residual addition. Pre-norm: <code>x + Sublayer(LN(x))</code>. Post-norm: <code>LN(x + Sublayer(x))</code>. Pre-norm is the modern default; it's much more stable for deep stacks because the residual stream stays unnormalized and gradients flow cleanly through it. Post-norm was the original Transformer paper's choice; it tends to need careful learning-rate warmup. <em>Always pre-norm unless you have a specific reason.</em></p>

<p class="q">Should I use the affine parameters (weight and bias)?</p>
<p class="a">Almost always yes for LN/BN/GN. The affine restoration gives the network the ability to undo the normalization if it needs to, while still gaining the optimization benefits of normalized gradients. The default in PyTorch is to include them. RMSNorm only includes weight, not bias.</p>
</div>

<h2>Activations, briefly</h2>

<p>Most of the activation choice has been settled. The shortlist:</p>

<div class="table-wrap">
<table>
<caption>Activation functions you'll actually meet</caption>
<thead><tr><th>Activation</th><th>Formula</th><th>Use it for…</th></tr></thead>
<tbody>
<tr><td><code>ReLU</code></td><td><code>max(0, x)</code></td><td>Old reliable. CNNs, MLPs, anything where you want speed.</td></tr>
<tr><td><code>GELU</code></td><td><code>x · Φ(x)</code> (Φ is normal CDF)</td><td>Default for transformers. Smooth, slight negative leakage.</td></tr>
<tr><td><code>SiLU</code> / <code>Swish</code></td><td><code>x · sigmoid(x)</code></td><td>Modern transformers (LLaMA-style FFNs). Similar to GELU.</td></tr>
<tr><td><code>SwiGLU</code> (gated)</td><td><code>SiLU(W₁x) · (W₂x)</code></td><td>State-of-the-art FFN block. 2-layer MLP becomes 3 weights with gating.</td></tr>
<tr><td><code>tanh</code></td><td>self-explanatory</td><td>Old RNNs, the value head of some RL nets.</td></tr>
</tbody>
</table>
</div>

<p>For deep networks today, GELU or SiLU are the safe choices. SwiGLU squeezes a bit more performance out of FFN blocks but uses more parameters (3 matrices instead of 2 for the same hidden dim). Most LLaMA-family models use SwiGLU.</p>

<h2>Embeddings: a Linear with integer indexing</h2>

<p><code>nn.Embedding</code> is conceptually a <code>(vocab_size, embed_dim)</code> weight matrix indexed by integer tokens. Forward pass is just <code>weight[indices]</code> — fancy indexing, returns shape <code>(*indices.shape, embed_dim)</code>.</p>

<pre><code>emb = nn.<span class="fn">Embedding</span>(num_embeddings=<span class="num">50257</span>, embedding_dim=<span class="num">768</span>)   <span class="com"># GPT-2 vocab</span>
tokens = torch.<span class="fn">tensor</span>([[<span class="num">15496</span>, <span class="num">995</span>, <span class="num">11</span>, <span class="num">995</span>]])                <span class="com"># shape (1, 4)</span>
out = <span class="fn">emb</span>(tokens)                                          <span class="com"># shape (1, 4, 768)</span></code></pre>

<p>Init for embeddings is conventionally <code>N(0, 0.02²)</code>, same as GPT-style. Not Kaiming, because the "fan_in" concept doesn't apply — the input is an integer index, not a vector of activations.</p>

<p>For very large vocabularies (LLM-scale), the embedding table dominates parameter count. This is why <strong>weight tying</strong> (M7) — sharing the embedding with the output projection — is standard.</p>

<h2>Dropout, briefly</h2>

<p>Drops each element of the input independently with probability <code>p</code> during training; scales the kept ones by <code>1/(1-p)</code> to preserve the expected value. In eval mode: identity.</p>

<pre><code>drop = nn.<span class="fn">Dropout</span>(p=<span class="num">0.1</span>)
y = <span class="fn">drop</span>(x)             <span class="com"># scaled-dropped in train mode, identity in eval mode</span></code></pre>

<p>Modern transformers use very low dropout (0 or 0.1) — large training corpora make heavy regularization unnecessary. CNNs and smaller-scale models can use higher (0.2-0.5).</p>

<p>Dropout is <em>the</em> archetype of "train/eval matters" — it's why <code>model.eval()</code> exists. If you forget <code>eval()</code> at inference, dropout is still firing, and your outputs are stochastic.</p>

<h2>Code Magnets: build the canonical transformer block</h2>

<p>You're building a single transformer block: pre-norm, multi-head attention, residual, pre-norm, MLP, residual. Use LayerNorm, GELU activation, and apply Dropout. Build the <code>forward</code>.</p>

<div class="magnets">
<p>Arrange the magnets into a working <code>forward</code>. Three are red herrings.</p>

<div class="magnet-pool">
  <span class="magnet">def forward(self, x, attn_mask=None):</span>
  <span class="magnet">    x = x + self.dropout(self.attn(self.ln1(x), attn_mask=attn_mask))</span>
  <span class="magnet">    x = self.ln1(x + self.dropout(self.attn(x, attn_mask=attn_mask)))</span>
  <span class="magnet">    x = x + self.dropout(self.mlp(self.ln2(x)))</span>
  <span class="magnet">    x = self.ln2(x + self.dropout(self.mlp(x)))</span>
  <span class="magnet">    return x</span>
  <span class="magnet">    return self.ln_final(x)</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">def</span> <span class="fn">forward</span>(self, x, attn_mask=<span class="kw">None</span>):
    x = x + self.<span class="fn">dropout</span>(self.<span class="fn">attn</span>(self.<span class="fn">ln1</span>(x), attn_mask=attn_mask))
    x = x + self.<span class="fn">dropout</span>(self.<span class="fn">mlp</span>(self.<span class="fn">ln2</span>(x)))
    <span class="kw">return</span> x</code></pre>
<p>The traps:</p>
<ul>
  <li><code>x = self.ln1(x + ...)</code> is post-norm. Works, but harder to train deep stacks. Pre-norm puts the LN <em>inside</em> the residual: <code>x + Sublayer(LN(x))</code>.</li>
  <li><code>x = self.ln2(x + ...)</code>: same post-norm mistake for the MLP block.</li>
  <li><code>return self.ln_final(x)</code>: a final LN can exist (GPT-2 has one at the end of the stack), but it's not part of the per-block return — it'd cause a double LN with the next block's pre-norm.</li>
</ul>
<p>The structural rule: each sublayer is <code>x = x + Sublayer(LN(x))</code>. Two sublayers per block (attention, MLP). Each contributes to the residual stream without normalizing it.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each layer/init concept to its purpose.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Purpose</div>

  <div>Kaiming init</div>
  <div>A. Normalize per-channel using running stats during eval; needs batch size to be reasonable.</div>

  <div>GPT-style residual init</div>
  <div>B. Per-sample, across-features normalization; works at batch size 1.</div>

  <div>BatchNorm</div>
  <div>C. Drop the mean from LayerNorm; just divide by RMS.</div>

  <div>LayerNorm</div>
  <div>D. Variance-preserving init for ReLU-family activations; uses fan_in.</div>

  <div>RMSNorm</div>
  <div>E. Scales residual-projection weights down by √(2·n_layers) to prevent residual blow-up.</div>

  <div>SwiGLU</div>
  <div>F. Gated MLP: SiLU(W₁x) · (W₂x). Better than vanilla MLP at the same parameter budget.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>Kaiming init</strong> → D<br>
<strong>GPT-style residual init</strong> → E<br>
<strong>BatchNorm</strong> → A<br>
<strong>LayerNorm</strong> → B<br>
<strong>RMSNorm</strong> → C<br>
<strong>SwiGLU</strong> → F
</p>
<p>The mental shortcut: <em>Kaiming for ReLU, GPT-init scales residuals, BN normalizes per-channel-cross-batch, LN per-sample-cross-features, RMSNorm skips mean, SwiGLU is the modern gated MLP</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> Implement <code>LayerNorm</code> from scratch as an <code>nn.Module</code>. Verify it matches <code>nn.LayerNorm</code> on a random input.</p>
<details class="answer"><summary>show answer</summary>
<pre><code><span class="kw">class</span> <span class="ty">MyLN</span>(nn.Module):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, dim, eps=<span class="num">1e-5</span>):
        <span class="fn">super</span>().<span class="fn">__init__</span>()
        self.weight = nn.<span class="fn">Parameter</span>(torch.<span class="fn">ones</span>(dim))
        self.bias   = nn.<span class="fn">Parameter</span>(torch.<span class="fn">zeros</span>(dim))
        self.eps = eps

    <span class="kw">def</span> <span class="fn">forward</span>(self, x):
        mean = x.<span class="fn">mean</span>(dim=-<span class="num">1</span>, keepdim=<span class="kw">True</span>)
        var  = x.<span class="fn">var</span>(dim=-<span class="num">1</span>, keepdim=<span class="kw">True</span>, unbiased=<span class="kw">False</span>)
        x_hat = (x - mean) / (var + self.eps).<span class="fn">sqrt</span>()
        <span class="kw">return</span> x_hat * self.weight + self.bias

x = torch.<span class="fn">randn</span>(<span class="num">2</span>, <span class="num">3</span>, <span class="num">8</span>)
<span class="fn">assert</span> torch.<span class="fn">allclose</span>(<span class="fn">MyLN</span>(<span class="num">8</span>)(x), nn.<span class="fn">LayerNorm</span>(<span class="num">8</span>)(x), atol=<span class="num">1e-6</span>)</code></pre>
<p>Two things to note: (1) <code>unbiased=False</code> — LN uses the population variance, divide by N not N−1. (2) The default <code>weight</code> is ones and <code>bias</code> is zeros, so right after init the layer is the identity (apart from numerical normalization).</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> What's the variance of the activations after a single Linear layer initialized with <code>std = 1/√(fan_in)</code>, given input variance 1? Do the math.</p>
<details class="answer"><summary>show answer</summary>
<p>For <code>y = Wx</code> where <code>W</code> is shape <code>(D_out, D_in)</code> with entries ~ N(0, σ²) and <code>x</code> ~ N(0, 1) elementwise:</p>
<p><code>Var(y[i]) = sum over j of Var(W[i,j]·x[j]) = D_in · σ² · 1 = D_in · σ²</code>.</p>
<p>With <code>σ² = 1/D_in</code>, we get <code>Var(y[i]) = 1</code>. ✓ Variance is preserved. This is exactly Xavier/Kaiming for the linear case (without an activation function). The activation function adds a multiplicative correction — for ReLU it's a factor of 2 (since ReLU zeros half the variance), which is why Kaiming uses <code>2/fan_in</code>.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> A model uses BatchNorm and trains fine, but inference produces wildly different outputs depending on what other samples are in the batch. What's wrong?</p>
<details class="answer"><summary>show answer</summary>
<p>The model is in train mode at inference time. <code>model.train()</code> is on, so BatchNorm is computing per-batch statistics from whatever samples are currently in the batch. Each different batch composition gives different normalization. Fix: <code>model.eval()</code> before inference, which switches BN to using running stats (computed during training and stored as buffers). This is the canonical "I trained a CNN and inference is broken" bug.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> Write the GPT-style init for a transformer where you can identify residual-projection weights by their suffix (e.g., names ending in <code>'.out_proj.weight'</code>).</p>
<details class="answer"><summary>show answer</summary>
<pre><code><span class="kw">def</span> <span class="fn">init_gpt_style</span>(model, n_layers):
    std = <span class="num">0.02</span>
    proj_std = std / (<span class="num">2</span> * n_layers) ** <span class="num">0.5</span>
    <span class="kw">for</span> name, p <span class="kw">in</span> model.<span class="fn">named_parameters</span>():
        <span class="kw">if</span> p.dim() &gt;= <span class="num">2</span>:
            <span class="kw">if</span> name.<span class="fn">endswith</span>(<span class="str">'.out_proj.weight'</span>):
                nn.init.<span class="fn">normal_</span>(p, mean=<span class="num">0.0</span>, std=proj_std)
            <span class="kw">else</span>:
                nn.init.<span class="fn">normal_</span>(p, mean=<span class="num">0.0</span>, std=std)
        <span class="kw">elif</span> <span class="str">'bias'</span> <span class="kw">in</span> name:
            nn.init.<span class="fn">zeros_</span>(p)
        <span class="kw">elif</span> p.dim() == <span class="num">1</span>:
            nn.init.<span class="fn">ones_</span>(p)</code></pre>
<p>The check <code>p.dim() &gt;= 2</code> picks up weight matrices; <code>p.dim() == 1</code> catches LayerNorm weights and similar 1-D tensors that should init to 1, not 0. Biases (also 1-D, but identifiable by name) get zeros.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li><strong>Initialization preserves activation variance across layers.</strong> Wrong scale → exponential blowup or collapse → NaN or dead network. Right scale → variance ≈ 1 throughout.</li>
  <li><strong>Kaiming init</strong> (<code>σ = √(2/fan_in)</code>) is the right default for ReLU/GELU networks. <strong>Xavier</strong> is for symmetric activations (older). <strong>GPT-style</strong> uses fixed σ=0.02 plus residual-projection scaling by <code>√(2·n_layers)</code> — this is the modern transformer default.</li>
  <li>Initialize biases to zero. Initialize LayerNorm weights to 1. Initialize embeddings with the same N(0, 0.02²) as GPT-style.</li>
  <li>Use <code>model.apply(init_fn)</code> to walk the module tree and apply per-type init.</li>
  <li><strong>Every normalization layer does the same arithmetic.</strong> The character is determined entirely by <em>which axes the mean and std are computed over</em>.</li>
  <li><strong>BatchNorm</strong>: per-channel stats across batch + spatial. Needs reasonable batch size; uses running stats at eval. Default for CNNs.</li>
  <li><strong>LayerNorm</strong>: per-sample stats across features. Batch-size-independent. No train/eval split. Default for transformers.</li>
  <li><strong>RMSNorm</strong>: drop the mean from LN; just divide by RMS. Skips a reduction. Used in LLaMA, Mistral, modern LLMs. <code>nn.RMSNorm</code> in PyTorch 2.4+.</li>
  <li><strong>GroupNorm</strong>: middle ground — per-(sample, channel-group). Works well when batch is small and channels can be meaningfully grouped (Stable Diffusion uses it).</li>
  <li><strong>Pre-norm</strong> (<code>x + Sublayer(LN(x))</code>) beats post-norm for deep stacks. Modern default.</li>
  <li><strong>Activations</strong>: GELU or SiLU for transformers; SwiGLU for FFN blocks if you can afford the extra matrix; ReLU for everything else where speed matters.</li>
  <li><strong>Embeddings</strong> are conceptually a Linear indexed by integer tokens. Init with N(0, 0.02²). Often weight-tied with the output projection.</li>
  <li><strong>Dropout</strong> is the archetype of "train/eval matters." Forget <code>eval()</code> at inference and outputs become stochastic.</li>
  <li>The reflex: when activations explode or collapse, suspect init first. When training is stable but you're hitting batch-size limits, check normalization choice. Both are five-line fixes that change everything.</li>
</ul>
</div>

<p>Module 09 closes Part III with the optimizer side of the equation: loss functions you'll actually use (CE, BCE, focal, MSE, Huber), optimizer algorithms with their math (SGD, Adam, AdamW, Lion, Muon), learning-rate schedules (cosine, warmup, OneCycle), and gradient clipping.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">08</span>
  <span>Initialization, layers & norms</span>
</div>
"""

emit("08_init_layers_norms", "Module 08 — Initialization, layers & norms", BODY)
