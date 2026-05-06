#!/usr/bin/env python3
"""Module 09: Losses, optimizers & schedulers — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part III · Module 09</div>
  <h1 class="module-title">Losses, <em>optimizers &amp; schedulers</em></h1>
  <p class="module-sub">— what <code>CrossEntropyLoss</code> is hiding, why AdamW exists, and the warmup-then-cosine schedule that powers every modern LLM</p>
</div>

<p>You have layers and you have init. You have autograd. The last piece of the training-loop puzzle is the part that <em>uses</em> the gradients: the loss that produces them, the optimizer that applies them, and the schedule that controls how aggressively. None of these are deep, but each one has a few sharp edges that take down beginners.</p>

<p>This module covers the cross-entropy stability tricks (the ones <code>F.cross_entropy</code> is doing for you), the actual update rules of the optimizers you'll use, the AdamW vs Adam-with-weight-decay difference (which is bigger than you think), and the cosine-with-warmup schedule that's the default for everything.</p>

<div class="keyidea">
A loss function is a <strong>differentiable scalar</strong>. An optimizer is a <strong>state machine</strong> that maintains per-parameter buffers (momentum, running squared gradients, etc.) and applies them on <code>.step()</code>. A scheduler is a <strong>boring function</strong> that decides what the learning rate is this step. Each piece earns its own seat in the loop, but together they're just three lines: <code>loss.backward()</code>, <code>optimizer.step()</code>, <code>scheduler.step()</code>.
</div>

<h2>Two new players</h2>

<div class="character" style="--c: #c1502e;">
  <div class="avatar" style="background: #c1502e; color: #fff;">⚙</div>
  <div>
    <p class="who">Optimizer</p>
    <p class="name">"I hold state. I apply updates. I'm where memory disappears in big training runs."</p>
    <p class="says">When you create me with <code>Adam(model.parameters())</code>, I don't just memorize the parameters — I allocate <em>buffers</em> for each one. SGD with momentum needs one buffer per parameter (the velocity). Adam needs <em>two</em> (first and second moments). For a 7B parameter model in fp32, that's <strong>56 GB of optimizer state</strong> just for Adam. This is why ZeRO and FSDP exist (Module 17). Don't underestimate me.</p>
  </div>
</div>

<div class="character" style="--c: #d4a017;">
  <div class="avatar" style="background: #d4a017;">⏱</div>
  <div>
    <p class="who">Scheduler</p>
    <p class="name">"I'm the boring one. I just tell Optimizer what learning rate to use."</p>
    <p class="says">I read the step counter and produce a learning rate. Linear warmup at the start (because adaptive optimizers need a few steps to estimate variance). Cosine decay over training (because empirically it works). That's most of what I do. The exotic ones (OneCycle, ReduceLROnPlateau) are situational. Most of the time, warmup-then-cosine is the right answer and you should stop second-guessing it.</p>
  </div>
</div>

<h2>Loss functions: the ones you'll actually use</h2>

<p>Five losses cover ~95% of training. Each has a numerical-stability story worth knowing.</p>

<div class="table-wrap">
<table>
<caption>The losses that earn their keep</caption>
<thead><tr><th>Loss</th><th>For</th><th>What's hidden inside</th></tr></thead>
<tbody>
<tr><td><code>F.cross_entropy</code></td><td>Multi-class classification with logits</td><td>Fused log-softmax + NLL; uses logsumexp for stability</td></tr>
<tr><td><code>F.binary_cross_entropy_with_logits</code></td><td>Binary / multi-label classification with logits</td><td>Fused sigmoid + BCE; uses log(1+exp) trick for stability</td></tr>
<tr><td><code>F.mse_loss</code></td><td>Regression</td><td>Just <code>(x - y)²</code>, no surprises</td></tr>
<tr><td><code>F.smooth_l1_loss</code> / Huber</td><td>Regression with outliers</td><td>Quadratic near zero, linear far from zero</td></tr>
<tr><td><code>focal_loss</code> (custom)</td><td>Imbalanced classification</td><td>Down-weights easy examples by <code>(1-p)^γ</code></td></tr>
</tbody>
</table>
</div>

<h3>Cross-entropy: what it's actually doing</h3>

<p>You feed it raw logits. It returns a scalar. In between, it's doing something subtle:</p>

<pre><code><span class="com"># What you write:</span>
loss = F.<span class="fn">cross_entropy</span>(logits, targets)

<span class="com"># What it's doing internally (roughly):</span>
log_probs = logits - torch.<span class="fn">logsumexp</span>(logits, dim=-<span class="num">1</span>, keepdim=<span class="kw">True</span>)
loss = -log_probs.<span class="fn">gather</span>(<span class="num">1</span>, targets.<span class="fn">unsqueeze</span>(<span class="num">1</span>)).<span class="fn">squeeze</span>(<span class="num">1</span>).<span class="fn">mean</span>()</code></pre>

<p>Three things to notice. (1) The log-softmax and the negative-log-likelihood are <em>fused</em> — never compute <code>softmax(logits).log()</code> separately, the intermediate <code>softmax</code> has bad numerics. (2) <code>logsumexp</code> from M3 does the heavy lifting on stability. (3) The <code>gather</code> picks just the logit corresponding to each example's true class.</p>

<div class="warn">
<strong>Don't pre-softmax your logits before <code>cross_entropy</code>.</strong> The function expects raw logits. People sometimes write <code>F.cross_entropy(logits.softmax(-1), targets)</code> as a "safety measure" — that's mathematically wrong AND numerically worse than just passing the logits. The function name says <code>cross_entropy</code> but its API contract is "give me the logits and I'll handle the rest."
</div>

<h3>Cross-entropy options worth knowing</h3>

<pre><code><span class="com"># label_smoothing: replace one-hot target with (1-α) * one-hot + α/V</span>
loss = F.<span class="fn">cross_entropy</span>(logits, targets, label_smoothing=<span class="num">0.1</span>)

<span class="com"># ignore_index: skip examples with this target value (e.g., padding tokens)</span>
loss = F.<span class="fn">cross_entropy</span>(logits, targets, ignore_index=-<span class="num">100</span>)

<span class="com"># reduction: 'mean' (default), 'sum', or 'none' (return per-example loss)</span>
losses_per_example = F.<span class="fn">cross_entropy</span>(logits, targets, reduction=<span class="str">'none'</span>)</code></pre>

<p><strong>Label smoothing</strong> at 0.1 is a standard regularizer for classification — softens overconfidence, often improves calibration and slightly improves accuracy. <strong><code>ignore_index=-100</code></strong> is the standard convention for "skip this token's loss" — you set targets at padding positions to <code>-100</code> and CE just doesn't include them in the mean. Both ship for free.</p>

<h3>Binary cross-entropy: log-sum-exp trick #2</h3>

<p>Same idea as CE but for binary outputs. Use <code>binary_cross_entropy_with_logits</code> (BCE-WL), <em>not</em> the version that takes probabilities.</p>

<pre><code><span class="com"># Right (numerically stable):</span>
loss = F.<span class="fn">binary_cross_entropy_with_logits</span>(logits, targets.<span class="fn">float</span>())

<span class="com"># Wrong (sigmoid before passing):</span>
probs = logits.<span class="fn">sigmoid</span>()
loss = F.<span class="fn">binary_cross_entropy</span>(probs, targets.<span class="fn">float</span>())   <span class="com"># overflows on large logits</span></code></pre>

<p>The internal trick: <code>log(1 + exp(x))</code> overflows for large positive <code>x</code>. The fused form computes it as <code>max(x, 0) + log(1 + exp(-|x|))</code>, which is stable in both directions. Passing pre-sigmoided probabilities goes through <code>log(p)</code> which loses precision near 0 and 1.</p>

<h3>Focal loss for imbalance</h3>

<p>For severely imbalanced datasets — say, object detection where most boxes are background. Cross-entropy is dominated by the easy negatives. Focal loss multiplies CE by <code>(1 - p_correct)^γ</code>, which down-weights examples the model is already confident about.</p>

<pre><code><span class="kw">def</span> <span class="fn">focal_loss</span>(logits, targets, alpha=<span class="num">0.25</span>, gamma=<span class="num">2.0</span>):
    ce = F.<span class="fn">cross_entropy</span>(logits, targets, reduction=<span class="str">'none'</span>)
    pt = (-ce).<span class="fn">exp</span>()                  <span class="com"># prob assigned to the correct class</span>
    <span class="kw">return</span> (alpha * (<span class="num">1</span> - pt) ** gamma * ce).<span class="fn">mean</span>()</code></pre>

<p>γ=2 is the standard. The clever part is computing <code>pt</code> from the CE itself — you don't need a separate softmax pass.</p>

<h3>Regression: MSE vs Huber</h3>

<p>MSE (L2): squared error. Penalizes outliers heavily. Huber (smooth L1): quadratic near zero, linear far from zero. Less sensitive to outliers.</p>

<pre><code>loss = F.<span class="fn">mse_loss</span>(pred, target)
loss = F.<span class="fn">smooth_l1_loss</span>(pred, target)              <span class="com"># Huber with default beta=1.0</span></code></pre>

<p>For most regression tasks, just use MSE. Reach for Huber when you have known outlier contamination (e.g., RL value targets, where one bad bootstrap can wreck training).</p>

<div class="ndq">
<h4>About losses</h4>

<p class="q">Should I use the functional form (<code>F.cross_entropy</code>) or the module form (<code>nn.CrossEntropyLoss</code>)?</p>
<p class="a">Either. The functional version is what most modern code uses because it's a one-liner. The module form exists for the rare case where you want to attach the loss as part of your <code>nn.Module</code> tree (e.g., to use hooks on it). They do the same thing.</p>

<p class="q">Why is my loss negative?</p>
<p class="a">If you're using cross-entropy or BCE-WL, it shouldn't be — those are always non-negative. If you see a negative loss, you probably (a) used the wrong target dtype (CE expects long for class indices, not float), (b) accidentally used <code>cross_entropy</code> on already-softmaxed inputs, or (c) wrote a custom loss with a sign error.</p>

<p class="q">What's <code>reduction='none'</code> for?</p>
<p class="a">When you want per-example losses to combine in a custom way — e.g., per-token cross-entropy weighted by some attention mask, or per-sample loss for hard-example mining. You get back a tensor of losses, and you reduce it yourself.</p>

<p class="q">Does cross-entropy work for sequence models?</p>
<p class="a">Yes — flatten the batch and sequence dims first. <code>F.cross_entropy(logits.view(-1, V), targets.view(-1))</code> where <code>logits</code> was <code>(B, T, V)</code> and <code>targets</code> was <code>(B, T)</code>. Or use the modern signature that accepts <code>(B, V, T)</code> directly.</p>
</div>

<h2>Optimizers: the math, the memory, the choice</h2>

<p>Every optimizer has the same job: given gradients, update parameters. They differ in <em>what state they keep</em> to make the update smarter than naive gradient descent. Each optimizer's character is its state.</p>

<div class="table-wrap">
<table>
<caption>The optimizers you'll meet, with their state cost</caption>
<thead><tr><th>Optimizer</th><th>Per-param state</th><th>Update rule (sketch)</th></tr></thead>
<tbody>
<tr><td><code>SGD</code> (no momentum)</td><td>None — 0× params</td><td><code>p ← p − lr · g</code></td></tr>
<tr><td><code>SGD + momentum</code></td><td>1 buffer (velocity) — 1× params</td><td><code>v ← μv + g; p ← p − lr · v</code></td></tr>
<tr><td><code>Adam</code> / <code>AdamW</code></td><td>2 buffers (m, v) — 2× params</td><td>EMA of g, EMA of g²; rescale; bias-correct</td></tr>
<tr><td><code>Lion</code></td><td>1 buffer (momentum) — 1× params</td><td>Sign-based; <code>p ← p − lr · sign(β₁m + (1-β₁)g)</code></td></tr>
<tr><td><code>Muon</code> (newer)</td><td>1 buffer + orthogonalization compute</td><td>Newton-Schulz orthogonalization on the momentum matrix</td></tr>
</tbody>
</table>
</div>

<p>Three things to internalize. (1) Adam takes <strong>2× the parameter memory just for state</strong> — and the parameters themselves are usually in fp32 even when the model is in bf16 (so it's another 2×). For a 7B model that's <em>56 GB of optimizer state</em>. (2) Lion is half the memory of Adam and a strong baseline; sign-based, less precise but works. (3) Muon is the new cool kid in 2024-25 — orthogonalizes the momentum matrix per step, hot in research circles for slightly better convergence.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 280" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="24" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Memory cost of optimizer state, per parameter (fp32)</text>

  <!-- Each bar: 4 bytes for params (always shown), then state on top -->
  <g transform="translate(40, 60)">
    <!-- y axis label -->
    <text x="-30" y="100" font-size="11" fill="#1a1612" text-anchor="middle" transform="rotate(-90 -30 100)">bytes per parameter</text>

    <!-- SGD -->
    <g transform="translate(0, 0)">
      <rect x="0" y="120" width="60" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
      <text x="30" y="145" font-size="11" text-anchor="middle" fill="#1a1612">params</text>
      <text x="30" y="180" font-size="11" font-weight="700" text-anchor="middle" fill="#1a1612">SGD</text>
      <text x="30" y="195" font-size="10" text-anchor="middle" fill="#1f5f5b">4 bytes</text>
    </g>

    <!-- SGD+momentum -->
    <g transform="translate(110, 0)">
      <rect x="0" y="80" width="60" height="40" fill="#fff8a8" stroke="#c1502e" stroke-width="1.5"/>
      <text x="30" y="105" font-size="11" text-anchor="middle" fill="#1a1612">velocity</text>
      <rect x="0" y="120" width="60" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
      <text x="30" y="145" font-size="11" text-anchor="middle" fill="#1a1612">params</text>
      <text x="30" y="180" font-size="11" font-weight="700" text-anchor="middle" fill="#1a1612">SGD+mom</text>
      <text x="30" y="195" font-size="10" text-anchor="middle" fill="#1f5f5b">8 bytes</text>
    </g>

    <!-- Lion -->
    <g transform="translate(220, 0)">
      <rect x="0" y="80" width="60" height="40" fill="#fff8a8" stroke="#c1502e" stroke-width="1.5"/>
      <text x="30" y="105" font-size="11" text-anchor="middle" fill="#1a1612">momentum</text>
      <rect x="0" y="120" width="60" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
      <text x="30" y="145" font-size="11" text-anchor="middle" fill="#1a1612">params</text>
      <text x="30" y="180" font-size="11" font-weight="700" text-anchor="middle" fill="#1a1612">Lion</text>
      <text x="30" y="195" font-size="10" text-anchor="middle" fill="#1f5f5b">8 bytes</text>
    </g>

    <!-- Adam / AdamW -->
    <g transform="translate(330, 0)">
      <rect x="0" y="40" width="60" height="40" fill="#ffd5dc" stroke="#b85a6c" stroke-width="1.5"/>
      <text x="30" y="65" font-size="11" text-anchor="middle" fill="#1a1612">v (g²)</text>
      <rect x="0" y="80" width="60" height="40" fill="#fff8a8" stroke="#c1502e" stroke-width="1.5"/>
      <text x="30" y="105" font-size="11" text-anchor="middle" fill="#1a1612">m (g)</text>
      <rect x="0" y="120" width="60" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
      <text x="30" y="145" font-size="11" text-anchor="middle" fill="#1a1612">params</text>
      <text x="30" y="180" font-size="11" font-weight="700" text-anchor="middle" fill="#1a1612">Adam</text>
      <text x="30" y="195" font-size="10" text-anchor="middle" fill="#c1502e">12 bytes</text>
    </g>

    <!-- Adam mixed-precision (params copy in fp32) -->
    <g transform="translate(440, 0)">
      <rect x="0" y="20" width="60" height="20" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
      <text x="30" y="35" font-size="9" text-anchor="middle" fill="#1a1612">master copy</text>
      <rect x="0" y="40" width="60" height="40" fill="#ffd5dc" stroke="#b85a6c" stroke-width="1.5"/>
      <text x="30" y="65" font-size="11" text-anchor="middle" fill="#1a1612">v (g²)</text>
      <rect x="0" y="80" width="60" height="40" fill="#fff8a8" stroke="#c1502e" stroke-width="1.5"/>
      <text x="30" y="105" font-size="11" text-anchor="middle" fill="#1a1612">m (g)</text>
      <rect x="0" y="140" width="60" height="20" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
      <text x="30" y="155" font-size="9" text-anchor="middle" fill="#1a1612">bf16 params</text>
      <text x="30" y="180" font-size="10" font-weight="700" text-anchor="middle" fill="#1a1612">Adam mixed</text>
      <text x="30" y="195" font-size="10" text-anchor="middle" fill="#c1502e">14 bytes</text>
    </g>

    <!-- 7B model annotation -->
    <text x="610" y="100" font-family="'Caveat', cursive" font-size="22" fill="#c1502e">7B params × 14B</text>
    <text x="610" y="125" font-family="'Caveat', cursive" font-size="22" fill="#c1502e">= ~98 GB just for</text>
    <text x="610" y="150" font-family="'Caveat', cursive" font-size="22" fill="#c1502e">optimizer + params</text>
  </g>
</svg>
</div>

<p>Look at that. <em>Optimizer state is the biggest single line item in big-model training.</em> The "8× model size" rule of thumb for total training memory (Module 12) is dominated by optimizer state. ZeRO-1 (Module 17) shards exactly this state across GPUs.</p>

<h3>SGD with momentum: the simplest serious optimizer</h3>

<pre><code>opt = torch.optim.<span class="fn">SGD</span>(model.<span class="fn">parameters</span>(), lr=<span class="num">0.1</span>, momentum=<span class="num">0.9</span>, weight_decay=<span class="num">5e-4</span>)

<span class="com"># Internally, per parameter p with gradient g:</span>
<span class="com">#   v ← momentum * v + g           (the velocity buffer)</span>
<span class="com">#   p ← p − lr * v                  (the update)</span>
<span class="com"># weight_decay is added to g before the velocity update (L2 regularization)</span></code></pre>

<p>SGD+momentum is the right baseline for image classification (ResNets, EfficientNets). It's fast, has 1 buffer per param, and with the right LR and schedule beats Adam on most CV tasks.</p>

<h3>Adam: the adaptive workhorse</h3>

<p>Adam keeps two EMAs per parameter: one of the gradient (first moment) and one of the squared gradient (second moment). It uses the second moment to <em>rescale</em> the update — parameters with consistently large gradients get smaller steps; parameters with small gradients get larger steps. The "adaptive" in Adam.</p>

<pre><code>opt = torch.optim.<span class="fn">Adam</span>(model.<span class="fn">parameters</span>(), lr=<span class="num">1e-3</span>, betas=(<span class="num">0.9</span>, <span class="num">0.999</span>), eps=<span class="num">1e-8</span>)

<span class="com"># Per parameter p with gradient g:</span>
<span class="com">#   m ← β₁ * m + (1 - β₁) * g          (first moment)</span>
<span class="com">#   v ← β₂ * v + (1 - β₂) * g²         (second moment)</span>
<span class="com">#   m̂ = m / (1 - β₁^t)                 (bias correction)</span>
<span class="com">#   v̂ = v / (1 - β₂^t)                 (bias correction)</span>
<span class="com">#   p ← p − lr * m̂ / (√v̂ + ε)</span></code></pre>

<p>The <strong>bias correction</strong> matters at the start of training. <code>m</code> and <code>v</code> are initialized to zero, so for the first few steps they're biased toward zero. The <code>m̂ = m / (1 - β₁^t)</code> correction undoes this. After ~100 steps the correction is negligible. But without it, your first few updates are tiny — exactly when you need them most.</p>

<h3>AdamW vs Adam-with-weight-decay: not the same thing</h3>

<p>This trips up a lot of people. There are two ways to "weight decay" in Adam:</p>

<pre><code><span class="com"># Way 1: Adam with weight_decay arg — adds w·p to the gradient BEFORE the EMA</span>
opt = torch.optim.<span class="fn">Adam</span>(model.<span class="fn">parameters</span>(), lr=<span class="num">1e-3</span>, weight_decay=<span class="num">0.01</span>)
<span class="com"># Effective: m ← β₁m + (1-β₁) * (g + wd*p), then adaptive rescale</span>
<span class="com"># Problem: the rescaling by √v̂ also scales the wd term, so decay is</span>
<span class="com"># larger for params with small gradients — usually NOT what you want</span>

<span class="com"># Way 2: AdamW — applies decoupled weight decay AFTER the adaptive update</span>
opt = torch.optim.<span class="fn">AdamW</span>(model.<span class="fn">parameters</span>(), lr=<span class="num">1e-3</span>, weight_decay=<span class="num">0.01</span>)
<span class="com"># Effective: p ← p − lr * (m̂/(√v̂+ε) + wd*p)</span>
<span class="com"># Decay is uniform across params regardless of their gradient magnitude</span></code></pre>

<p>The fix in AdamW is just decoupling: <em>apply weight decay independently of the adaptive scaling</em>. The empirical effect on transformer training is significant — better generalization, especially for the "decay-the-weights, but-not-the-biases-and-LayerNorms" pattern.</p>

<div class="warn">
<strong>Always use AdamW for transformers.</strong> The original Transformer paper used SGD+momentum; modern transformers use AdamW. The "Adam(weight_decay=0.01)" path that some old tutorials show is the wrong version. <code>torch.optim.AdamW</code> is what you want.
</div>

<h3>The "no decay on biases and LayerNorms" trick</h3>

<p>The standard parameter-grouping pattern:</p>

<pre><code><span class="kw">def</span> <span class="fn">make_param_groups</span>(model, weight_decay):
    decay, no_decay = [], []
    <span class="kw">for</span> name, p <span class="kw">in</span> model.<span class="fn">named_parameters</span>():
        <span class="kw">if</span> <span class="kw">not</span> p.requires_grad:
            <span class="kw">continue</span>
        <span class="com"># Don't decay biases or 1-D parameters (LayerNorm/RMSNorm scales)</span>
        <span class="kw">if</span> p.dim() &lt;= <span class="num">1</span> <span class="kw">or</span> name.<span class="fn">endswith</span>(<span class="str">'.bias'</span>):
            no_decay.<span class="fn">append</span>(p)
        <span class="kw">else</span>:
            decay.<span class="fn">append</span>(p)
    <span class="kw">return</span> [
        {<span class="str">'params'</span>: decay, <span class="str">'weight_decay'</span>: weight_decay},
        {<span class="str">'params'</span>: no_decay, <span class="str">'weight_decay'</span>: <span class="num">0.0</span>},
    ]

opt = torch.optim.<span class="fn">AdamW</span>(<span class="fn">make_param_groups</span>(model, <span class="num">0.1</span>), lr=<span class="num">3e-4</span>)</code></pre>

<p>Why? Decaying a LayerNorm scale toward zero would weaken normalization, which fights the network's whole purpose. Decaying a bias toward zero is also slightly bad — biases shift the distribution and 0 isn't always the right shift. The 2-D (and higher) weights are what we want regularized. This is the standard Hugging Face / nanoGPT pattern.</p>

<h3>Lion: half the memory, comparable performance</h3>

<pre><code>opt = torch.optim.<span class="fn">Lion</span>(model.<span class="fn">parameters</span>(), lr=<span class="num">1e-4</span>, weight_decay=<span class="num">0.01</span>)   <span class="com"># PyTorch 2.4+</span>

<span class="com"># Per parameter p with gradient g:</span>
<span class="com">#   update = sign(β₁ * m + (1 - β₁) * g)        # sign-based!</span>
<span class="com">#   p ← p − lr * (update + wd * p)</span>
<span class="com">#   m ← β₂ * m + (1 - β₂) * g                   # update m AFTER using it</span></code></pre>

<p>Lion (Lin et al., 2023) keeps just one buffer (the momentum), not two. The trick: take the <em>sign</em> of the momentum-blended gradient, not its value. Surprisingly this works — pretrained transformer performance is comparable to AdamW at a smaller memory footprint. Tune the LR ~3-10× lower than you'd use with AdamW.</p>

<h2>Learning rate schedules</h2>

<p>The learning rate is by far the most important hyperparameter. The schedule controls how it changes over training. Four common shapes:</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 240" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">The four schedules you'll meet</text>

  <!-- 4 panels -->
  <g transform="translate(20, 50)">
    <text x="80" y="14" font-size="12" font-weight="700" fill="#1f5f5b" text-anchor="middle">Constant</text>
    <line x1="0" y1="120" x2="160" y2="120" stroke="#6b5d4f" stroke-width="0.5"/>
    <line x1="0" y1="40"  x2="160" y2="40"  stroke="#6b5d4f" stroke-width="0.5" stroke-dasharray="2 2"/>
    <line x1="0" y1="40"  x2="160" y2="40"  stroke="#1f5f5b" stroke-width="2.5"/>
    <text x="80" y="160" font-size="10" text-anchor="middle" fill="#1a1612">no decay; baseline only</text>
  </g>

  <g transform="translate(200, 50)">
    <text x="80" y="14" font-size="12" font-weight="700" fill="#c1502e" text-anchor="middle">Warmup + cosine</text>
    <line x1="0" y1="120" x2="160" y2="120" stroke="#6b5d4f" stroke-width="0.5"/>
    <line x1="0" y1="40"  x2="160" y2="40"  stroke="#6b5d4f" stroke-width="0.5" stroke-dasharray="2 2"/>
    <!-- linear warmup 0-30, then cosine 30-160 -->
    <path d="M 0 120 L 30 40 Q 80 40 160 120" fill="none" stroke="#c1502e" stroke-width="2.5"/>
    <text x="80" y="160" font-size="10" text-anchor="middle" fill="#1a1612">modern LLM default</text>
  </g>

  <g transform="translate(380, 50)">
    <text x="80" y="14" font-size="12" font-weight="700" fill="#d4a017" text-anchor="middle">OneCycle</text>
    <line x1="0" y1="120" x2="160" y2="120" stroke="#6b5d4f" stroke-width="0.5"/>
    <line x1="0" y1="40"  x2="160" y2="40"  stroke="#6b5d4f" stroke-width="0.5" stroke-dasharray="2 2"/>
    <!-- ramp up 0-60 to peak, then ramp down 60-160 -->
    <path d="M 0 100 Q 30 100 60 40 Q 100 40 160 110 L 160 120" fill="none" stroke="#d4a017" stroke-width="2.5"/>
    <text x="80" y="160" font-size="10" text-anchor="middle" fill="#1a1612">vision tasks; super-convergence</text>
  </g>

  <g transform="translate(560, 50)">
    <text x="80" y="14" font-size="12" font-weight="700" fill="#b85a6c" text-anchor="middle">Step decay</text>
    <line x1="0" y1="120" x2="160" y2="120" stroke="#6b5d4f" stroke-width="0.5"/>
    <line x1="0" y1="40"  x2="160" y2="40"  stroke="#6b5d4f" stroke-width="0.5" stroke-dasharray="2 2"/>
    <path d="M 0 40 L 50 40 L 50 70 L 100 70 L 100 95 L 150 95 L 150 110" fill="none" stroke="#b85a6c" stroke-width="2.5"/>
    <text x="80" y="160" font-size="10" text-anchor="middle" fill="#1a1612">classic CNN training</text>
  </g>

  <text x="370" y="220" font-family="'Caveat', cursive" font-size="20" fill="#c1502e" text-anchor="middle">when in doubt: warmup + cosine</text>
</svg>
</div>

<h3>Warmup-then-cosine: the modern default</h3>

<pre><code><span class="kw">from</span> torch.optim.lr_scheduler <span class="kw">import</span> LambdaLR
<span class="kw">import</span> math

<span class="kw">def</span> <span class="fn">warmup_cosine</span>(step, warmup_steps, total_steps, min_lr_ratio=<span class="num">0.1</span>):
    <span class="kw">if</span> step &lt; warmup_steps:
        <span class="kw">return</span> step / warmup_steps                          <span class="com"># linear warmup 0 → 1</span>
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cos = <span class="num">0.5</span> * (<span class="num">1</span> + math.<span class="fn">cos</span>(math.pi * progress))         <span class="com"># 1 → 0</span>
    <span class="kw">return</span> min_lr_ratio + (<span class="num">1</span> - min_lr_ratio) * cos          <span class="com"># 1 → min_lr_ratio</span>

scheduler = <span class="fn">LambdaLR</span>(opt, <span class="kw">lambda</span> step: <span class="fn">warmup_cosine</span>(step, <span class="num">2000</span>, <span class="num">100_000</span>))</code></pre>

<p>Why warmup? Adam-family optimizers' second moment <code>v</code> is unreliable for the first few hundred steps (it's biased and noisy). Taking large updates in this regime can destabilize training. Warming up the learning rate from 0 to peak over 1-5% of total training gives the optimizer time to estimate <code>v</code> properly. Without warmup, big transformers diverge.</p>

<p>Why cosine? Empirically, smoothly decaying the LR works better than constant or step decay. The cosine shape spends most of its time near the peak (high learning) and ramps down smoothly at the end (high precision). It's not magic — linear decay also works fine — but cosine is the convention.</p>

<h3>Other useful schedulers</h3>

<pre><code><span class="kw">from</span> torch.optim.lr_scheduler <span class="kw">import</span> *

<span class="com"># Linear: warmup then linear decay (common in HF transformers)</span>
LinearLR(opt, start_factor=<span class="num">0.1</span>, end_factor=<span class="num">1.0</span>, total_iters=<span class="num">2000</span>)

<span class="com"># Cosine annealing (without warmup)</span>
CosineAnnealingLR(opt, T_max=<span class="num">100_000</span>, eta_min=<span class="num">0</span>)

<span class="com"># Reduce on plateau (validation-triggered)</span>
ReduceLROnPlateau(opt, mode=<span class="str">'min'</span>, factor=<span class="num">0.5</span>, patience=<span class="num">3</span>)

<span class="com"># Compose schedulers — warmup then cosine, the official way</span>
warmup = LinearLR(opt, start_factor=<span class="num">0.001</span>, total_iters=<span class="num">2000</span>)
cosine = CosineAnnealingLR(opt, T_max=<span class="num">98_000</span>)
scheduler = SequentialLR(opt, [warmup, cosine], milestones=[<span class="num">2000</span>])</code></pre>

<p>The <code>LambdaLR</code> approach is more flexible; the <code>SequentialLR</code> approach is more declarative. Both work.</p>

<div class="warn">
<strong>Step the scheduler at the right granularity.</strong> Most schedulers expect <code>scheduler.step()</code> per <em>training step</em> (i.e., per optimizer.step()). Some old code calls it per epoch. <code>ReduceLROnPlateau</code> is the exception — it expects per-epoch with a metric: <code>scheduler.step(val_loss)</code>. Match your scheduler's expected granularity.
</div>

<h2>Gradient clipping</h2>

<p>Last piece. The optimizer applies updates proportional to the gradient. If the gradient occasionally spikes — common in training with bf16 or long sequences — that one big update can wreck a parameter and destabilize training. Gradient clipping caps the update magnitude.</p>

<pre><code>loss.<span class="fn">backward</span>()
torch.nn.utils.<span class="fn">clip_grad_norm_</span>(model.<span class="fn">parameters</span>(), max_norm=<span class="num">1.0</span>)
optimizer.<span class="fn">step</span>()
optimizer.<span class="fn">zero_grad</span>()</code></pre>

<p>Two flavors:</p>

<ul>
  <li><strong>By norm</strong> (<code>clip_grad_norm_</code>): if the total L2 norm of all gradients exceeds <code>max_norm</code>, scale them all down proportionally. <em>Preserves direction.</em> Recommended.</li>
  <li><strong>By value</strong> (<code>clip_grad_value_</code>): clamp each individual gradient element to <code>[-c, c]</code>. <em>Distorts direction.</em> Rarely used.</li>
</ul>

<p>For transformers, <code>max_norm = 1.0</code> is the standard. For RL, often higher (5-10) because reward signals are noisier. <strong>Place clipping between <code>backward()</code> and <code>step()</code></strong>; clipping after step is too late.</p>

<h2>Putting it all together: the canonical training step</h2>

<p>Every modern training loop ends up looking like this:</p>

<pre><code>opt = torch.optim.<span class="fn">AdamW</span>(<span class="fn">make_param_groups</span>(model, <span class="num">0.1</span>), lr=<span class="num">3e-4</span>, betas=(<span class="num">0.9</span>, <span class="num">0.95</span>))
scheduler = <span class="fn">LambdaLR</span>(opt, <span class="kw">lambda</span> s: <span class="fn">warmup_cosine</span>(s, <span class="num">2000</span>, <span class="num">100_000</span>))

<span class="kw">for</span> step <span class="kw">in</span> <span class="fn">range</span>(<span class="num">100_000</span>):
    x, y = <span class="fn">next</span>(loader)
    logits = <span class="fn">model</span>(x)
    loss = F.<span class="fn">cross_entropy</span>(logits.<span class="fn">view</span>(-<span class="num">1</span>, V), y.<span class="fn">view</span>(-<span class="num">1</span>), ignore_index=-<span class="num">100</span>)

    opt.<span class="fn">zero_grad</span>()
    loss.<span class="fn">backward</span>()
    torch.nn.utils.<span class="fn">clip_grad_norm_</span>(model.<span class="fn">parameters</span>(), max_norm=<span class="num">1.0</span>)
    opt.<span class="fn">step</span>()
    scheduler.<span class="fn">step</span>()</code></pre>

<p>That's the modern LLM training inner loop. Memorize this template — you'll use it forever. Module 11 will harden it with checkpointing, accumulation, and resumption.</p>

<h2>Code Magnets: build the AdamW + cosine + clip step</h2>

<p>You're writing the training step for a transformer with the modern recipe. Order matters.</p>

<div class="magnets">
<p>Arrange the magnets into the correct training step. Two are red herrings.</p>

<div class="magnet-pool">
  <span class="magnet">opt.zero_grad()</span>
  <span class="magnet">logits = model(x)</span>
  <span class="magnet">loss = F.cross_entropy(logits.view(-1, V), y.view(-1))</span>
  <span class="magnet">loss = F.cross_entropy(logits.softmax(-1).view(-1, V), y.view(-1))</span>
  <span class="magnet">loss.backward()</span>
  <span class="magnet">torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)</span>
  <span class="magnet">opt.step()</span>
  <span class="magnet">scheduler.step()</span>
  <span class="magnet">opt.zero_grad(); opt.step(); loss.backward()</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code>opt.<span class="fn">zero_grad</span>()
logits = <span class="fn">model</span>(x)
loss = F.<span class="fn">cross_entropy</span>(logits.<span class="fn">view</span>(-<span class="num">1</span>, V), y.<span class="fn">view</span>(-<span class="num">1</span>))
loss.<span class="fn">backward</span>()
torch.nn.utils.<span class="fn">clip_grad_norm_</span>(model.<span class="fn">parameters</span>(), max_norm=<span class="num">1.0</span>)
opt.<span class="fn">step</span>()
scheduler.<span class="fn">step</span>()</code></pre>
<p>The traps:</p>
<ul>
  <li>Pre-softmaxing logits before <code>cross_entropy</code> is wrong — CE expects raw logits and does the log-softmax internally with stable arithmetic.</li>
  <li>The merged "<code>opt.zero_grad(); opt.step(); loss.backward()</code>" reverses the order. <code>step()</code> needs gradients in <code>.grad</code>, which only exist after <code>backward()</code>.</li>
</ul>
<p>The order matters: <strong>zero → forward → backward → clip → step → schedule</strong>. Clip <em>between</em> backward and step (clip the gradients that step will use). Schedule <em>after</em> step (most schedulers want to be called once per training step).</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>F.cross_entropy</div>
  <div>A. Decoupled weight decay; the right Adam variant for transformers.</div>

  <div>AdamW</div>
  <div>B. Fused log-softmax + NLL with logsumexp; expects raw logits.</div>

  <div>Bias correction in Adam</div>
  <div>C. Caps total gradient norm to a max value; preserves direction.</div>

  <div>Warmup phase</div>
  <div>D. Skip decay on biases and 1-D LayerNorm scales.</div>

  <div>Parameter grouping (no_decay)</div>
  <div>E. Lets the second-moment estimate <code>v</code> stabilize before taking large steps.</div>

  <div>clip_grad_norm_</div>
  <div>F. Undoes the zero-init bias on <code>m</code> and <code>v</code> in early steps.</div>

  <div>label_smoothing=0.1</div>
  <div>G. Replaces one-hot target with a softer (1-α)·one-hot + α/V; regularizer.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>F.cross_entropy</strong> → B<br>
<strong>AdamW</strong> → A<br>
<strong>Bias correction in Adam</strong> → F<br>
<strong>Warmup phase</strong> → E<br>
<strong>Parameter grouping (no_decay)</strong> → D<br>
<strong>clip_grad_norm_</strong> → C<br>
<strong>label_smoothing=0.1</strong> → G
</p>
<p>The mental shortcut: <em>CE fuses log-softmax+NLL, AdamW decouples decay, bias correction fixes early steps, warmup stabilizes the second moment, no-decay protects LayerNorms and biases, clip-grad caps norms, label smoothing softens targets</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> Compute the optimizer state size for a 1B parameter model trained with AdamW in fp32 (params and state both in fp32). Express in GB.</p>
<details class="answer"><summary>show answer</summary>
<p>Per parameter: 4 bytes for params + 4 for first moment + 4 for second moment = 12 bytes. For 1B params: 12 GB just for params + Adam state. (In mixed precision with a master copy of params, you'd add another 2 bytes per param for the bf16 working copy = 14 GB.) This is why FSDP exists.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> You see <code>F.cross_entropy(logits, targets)</code> producing a loss of <code>nan</code> at the very first step. The logits are large (~30 in magnitude) but finite. What's the most likely cause?</p>
<details class="answer"><summary>show answer</summary>
<p>Targets out of range. <code>cross_entropy</code> expects target indices in <code>[0, V)</code> (where V is the number of classes). If a target is <code>V</code> or larger, or negative (and not <code>ignore_index</code>), CE produces nonsense. The logsumexp on large logits is fine (the trick from M3 keeps it stable). Check <code>targets.max()</code> and <code>targets.min()</code>.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> A colleague's transformer trains fine on 1 GPU but diverges on 8 GPUs. They use AdamW with the same per-GPU batch size and the same LR. What's likely wrong?</p>
<details class="answer"><summary>show answer</summary>
<p>The effective batch size is 8× larger across 8 GPUs, but the LR didn't change. With Adam-family optimizers, you typically scale LR sublinearly with batch size (<code>lr ∝ √(batch_size)</code> is the rule of thumb), or you keep it the same and accept that you've effectively widened the LR-stable region. Either keep the original LR but extend warmup proportionally, or scale LR up by a small factor. Just running on 8 GPUs without thinking about effective batch size is one of the top distributed training bugs.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> Implement a "linear warmup, then constant" scheduler — useful when you want the optimizer to ramp in but don't have a fixed total step count.</p>
<details class="answer"><summary>show answer</summary>
<pre><code><span class="kw">from</span> torch.optim.lr_scheduler <span class="kw">import</span> LambdaLR

<span class="kw">def</span> <span class="fn">warmup_constant</span>(step, warmup_steps):
    <span class="kw">if</span> step &lt; warmup_steps:
        <span class="kw">return</span> step / warmup_steps
    <span class="kw">return</span> <span class="num">1.0</span>

scheduler = <span class="fn">LambdaLR</span>(opt, <span class="kw">lambda</span> s: <span class="fn">warmup_constant</span>(s, warmup_steps=<span class="num">2000</span>))</code></pre>
<p>Useful when you're doing exploratory training and don't have a fixed budget. Easy to extend to "warmup, constant, decay at the end" if you later want a wind-down phase.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li><strong><code>F.cross_entropy</code> takes raw logits.</strong> It fuses log-softmax + NLL with logsumexp inside. Don't pre-softmax.</li>
  <li><code>label_smoothing=0.1</code>, <code>ignore_index=-100</code>, and <code>reduction='none'</code> are the three CE options you'll actually use.</li>
  <li><code>F.binary_cross_entropy_with_logits</code> for binary/multi-label; <strong>never</strong> the version that takes probs.</li>
  <li><strong>Optimizer memory</strong> = (number of state buffers) × (param size). SGD: 0×, SGD+mom: 1×, Adam: 2×, Lion: 1×. For 7B in fp32 + Adam in fp32, that's ~98 GB.</li>
  <li><strong>Adam keeps two EMAs</strong>: first moment of <code>g</code>, second moment of <code>g²</code>. Bias correction undoes the zero-init of those buffers in early steps.</li>
  <li><strong>AdamW decouples weight decay</strong> from the adaptive scaling. Always use AdamW for transformers, not Adam-with-weight-decay.</li>
  <li>The standard trick: <strong>no decay on biases and 1-D parameters</strong> (LayerNorm/RMSNorm scales). Group parameters and apply <code>weight_decay=0</code> to the no-decay group.</li>
  <li><strong>Lion</strong> halves the optimizer memory at the cost of slightly worse convergence. Tune LR ~3-10× lower than AdamW.</li>
  <li><strong>Warmup-then-cosine</strong> is the modern LR schedule. Warmup gives Adam's variance estimate time to stabilize. Cosine decays smoothly to ~0.1× peak LR.</li>
  <li><strong>Gradient clipping by norm</strong>, <code>max_norm=1.0</code>, between <code>backward()</code> and <code>step()</code>. Standard for transformer training. Preserves gradient direction; just caps magnitude.</li>
  <li><strong>The canonical step</strong>: <code>zero_grad → forward → backward → clip → step → scheduler.step</code>. Memorize the order.</li>
</ul>
</div>

<p>That closes Part III. You've gone from <code>nn.Module</code> as a container, through init and normalization choices that make a network trainable, to the optimizer + loss + schedule trio that <em>actually trains it</em>. With one more part — the data pipeline and training-loop hardening — you'll have everything needed to ship a real training run.</p>

<p>Module 10 opens Part IV with the half of the system that nobody photographs: <code>Dataset</code>, <code>DataLoader</code>, samplers, <code>collate_fn</code>, the <code>num_workers</code> tuning game. The unsexy part where most performance bugs actually live.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">09</span>
  <span>Losses, optimizers & schedulers</span>
</div>
"""

emit("09_losses_optimizers_schedulers", "Module 09 — Losses, optimizers & schedulers", BODY)
