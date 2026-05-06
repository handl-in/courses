#!/usr/bin/env python3
"""Module 11: Training loops you can trust — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part IV · Module 11</div>
  <h1 class="module-title">Training loops <em>you can trust</em></h1>
  <p class="module-sub">— overfit-a-batch as the cheapest sanity check ever, the five things to save in a checkpoint, and gradient accumulation when you actually understand it</p>
</div>

<p>You have all the pieces from Parts I-III: tensors, autograd, modules, init, optimizer, scheduler. You have data pipelines from M10. Now we wire them together into something that actually trains a model — and keeps training, and survives a process restart, and tells you when it's broken before you've burned three days of GPU time.</p>

<p>This module is about <em>discipline</em>. The training loop itself is ten lines; the discipline around it — overfit-a-batch first, log the right things, checkpoint everything, validate without leaks — is what separates "I trained a model once" from "I run training pipelines every week."</p>

<div class="keyidea">
A trustworthy training loop has <strong>five components</strong> beyond the obvious forward/backward/step: an <em>overfit-a-batch sanity check</em> before you spend any real GPU time, a <em>logging cadence</em> that lets you spot trouble at glance, a <em>checkpoint format</em> that captures everything needed for bit-equal resumption, a <em>validation loop</em> that doesn't leak training state, and an <em>accumulation pattern</em> that handles tiny GPUs and large effective batches. Each one is a few lines of code and saves you days of debugging.
</div>

<h2>Two new players</h2>

<div class="character" style="--c: #1f5f5b;">
  <div class="avatar" style="background: #1f5f5b; color: #fff;">T</div>
  <div>
    <p class="who">Trainer</p>
    <p class="name">"I'm the script that owns the loop. I'm an <em>architecture decision</em>, not a class."</p>
    <p class="says">In small projects I'm a function. In larger ones I'm a class with hooks and callbacks. Either way, I'm where the data, model, optimizer, scheduler, and logger meet. I'm the place where you have to be most careful — every bug in me is a bug in <em>every</em> training run. Three rules: I keep the inner loop simple, I log enough to debug remotely, and I save enough state to resume bit-perfectly. Frameworks like Lightning and HuggingFace's <code>Trainer</code> are just opinionated implementations of me.</p>
  </div>
</div>

<div class="character" style="--c: #c1502e;">
  <div class="avatar" style="background: #c1502e; color: #fff;">⟁</div>
  <div>
    <p class="who">Checkpoint</p>
    <p class="name">"I'm a dict that gets pickled to disk. Save more than you think you need."</p>
    <p class="says">People think I'm just <code>model.state_dict()</code>. I'm not. To resume training bit-perfectly, you need: (1) the model state, (2) the optimizer state, (3) the scheduler state, (4) all RNG states (Python, NumPy, torch CPU, torch CUDA), and (5) the step counter. Skip any one and your "resumed" run silently diverges from what would have happened with no restart. The disk cost of saving all five is rounding error. The cost of <em>not</em> saving them is "why did training go off the rails after the restart?"</p>
  </div>
</div>

<h2>The training loop, anatomy</h2>

<p>Let's stare at the 10 lines you'll write a thousand times. Each one earns its place.</p>

<pre><code>opt = torch.optim.<span class="fn">AdamW</span>(<span class="fn">make_param_groups</span>(model, <span class="num">0.1</span>), lr=<span class="num">3e-4</span>)
scheduler = <span class="fn">LambdaLR</span>(opt, <span class="kw">lambda</span> s: <span class="fn">warmup_cosine</span>(s, <span class="num">2000</span>, <span class="num">100_000</span>))
scaler = torch.amp.<span class="fn">GradScaler</span>(<span class="str">'cuda'</span>)         <span class="com"># for fp16; bf16 doesn't need it</span>

<span class="kw">for</span> step, batch <span class="kw">in</span> <span class="fn">enumerate</span>(loader):
    x, y = [t.<span class="fn">to</span>(device, non_blocking=<span class="kw">True</span>) <span class="kw">for</span> t <span class="kw">in</span> batch]

    <span class="kw">with</span> torch.<span class="fn">autocast</span>(<span class="str">'cuda'</span>, dtype=torch.bfloat16):
        out = <span class="fn">model</span>(x)
        loss = F.<span class="fn">cross_entropy</span>(out, y)

    opt.<span class="fn">zero_grad</span>()
    loss.<span class="fn">backward</span>()
    torch.nn.utils.<span class="fn">clip_grad_norm_</span>(model.<span class="fn">parameters</span>(), <span class="num">1.0</span>)
    opt.<span class="fn">step</span>()
    scheduler.<span class="fn">step</span>()</code></pre>

<p>Read each line as a question:</p>

<ol>
  <li><strong>Optimizer</strong>: which parameter groups, what LR, which decay strategy? (M9)</li>
  <li><strong>Scheduler</strong>: what warmup, what schedule? (M9)</li>
  <li><strong>Scaler</strong>: are we using fp16 (yes → scaler) or bf16 (no scaler needed)? (M14)</li>
  <li><strong>Loader iteration</strong>: pinned, async, persistent workers? (M10)</li>
  <li><strong>Move to device</strong>: <code>non_blocking</code> only if pinned. (M3, M10)</li>
  <li><strong>Autocast block</strong>: which dtype, which ops cast? (M14)</li>
  <li><strong>Forward</strong>: model in train or eval mode? (M7, M8)</li>
  <li><strong>Loss</strong>: which loss function, raw logits, ignored padding? (M9)</li>
  <li><strong>Zero grad</strong>: between every step, or accumulating? (M4)</li>
  <li><strong>Backward / clip / step / scheduler.step</strong>: in the right order. (M4, M9)</li>
</ol>

<p>Every module so far feeds into one of these lines. That's why we did them in that order.</p>

<h2>Reflex 1: overfit a single batch</h2>

<p>Before you spend a dollar of GPU time on real training, run this:</p>

<pre><code>x, y = <span class="fn">next</span>(<span class="fn">iter</span>(loader))
x, y = x.<span class="fn">to</span>(device), y.<span class="fn">to</span>(device)

<span class="kw">for</span> step <span class="kw">in</span> <span class="fn">range</span>(<span class="num">200</span>):
    out = <span class="fn">model</span>(x)
    loss = <span class="fn">criterion</span>(out, y)
    opt.<span class="fn">zero_grad</span>(); loss.<span class="fn">backward</span>(); opt.<span class="fn">step</span>()
    <span class="kw">if</span> step % <span class="num">10</span> == <span class="num">0</span>:
        <span class="fn">print</span>(<span class="str">f"step {step}: loss {loss.item():.4f}"</span>)</code></pre>

<p>This loop trains on the <em>same single batch</em> 200 times. The model has more than enough capacity to memorize 32 examples. Therefore: <strong>the loss must go to ~0</strong>. If it doesn't, you have a bug.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 240" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Overfit-a-batch: a 60-second sanity check</text>

  <!-- axes -->
  <line x1="80"  y1="180" x2="660" y2="180" stroke="#6b5d4f" stroke-width="1"/>
  <line x1="80"  y1="50"  x2="80"  y2="180" stroke="#6b5d4f" stroke-width="1"/>
  <text x="370" y="208" font-size="11" fill="#6b5d4f" text-anchor="middle">step</text>
  <text x="60" y="120" font-size="11" fill="#6b5d4f" text-anchor="middle" transform="rotate(-90 60 120)">loss</text>

  <!-- gridlines -->
  <line x1="80" y1="80"  x2="660" y2="80"  stroke="#c8b89a" stroke-width="0.5" stroke-dasharray="2 2"/>
  <line x1="80" y1="120" x2="660" y2="120" stroke="#c8b89a" stroke-width="0.5" stroke-dasharray="2 2"/>
  <line x1="80" y1="160" x2="660" y2="160" stroke="#c8b89a" stroke-width="0.5" stroke-dasharray="2 2"/>

  <!-- good curve: drops to ~0 -->
  <path d="M 80 60 Q 200 70 300 110 Q 400 150 500 170 Q 580 178 660 178" fill="none" stroke="#1f5f5b" stroke-width="3"/>
  <text x="500" y="155" font-size="12" font-weight="700" fill="#1f5f5b">good — model memorizes ✓</text>

  <!-- bad curve 1: plateau -->
  <path d="M 80 70 Q 200 90 300 100 L 660 105" fill="none" stroke="#c1502e" stroke-width="3" stroke-dasharray="4 3"/>
  <text x="380" y="98" font-size="12" font-weight="700" fill="#c1502e">bad — plateau (architecture/init bug)</text>

  <!-- bad curve 2: noise -->
  <path d="M 80 80 L 130 100 L 180 70 L 230 95 L 280 75 L 330 105 L 380 80 L 430 100 L 480 70 L 530 95 L 580 80 L 630 105" fill="none" stroke="#b85a6c" stroke-width="2" stroke-dasharray="2 2"/>
  <text x="350" y="60" font-size="12" font-weight="700" fill="#b85a6c">bad — noisy/unstable (LR too high)</text>
</svg>
</div>

<p>The three curves above are real outcomes. Solid green is what you want — loss approaches zero in a few hundred steps. The other two are the bugs you catch with this check:</p>

<ul>
  <li><strong>Plateau</strong> at some non-zero loss → architecture has a bottleneck (a frozen layer that shouldn't be, a dimension mismatch that's silently broadcasting wrong, an init scheme that's wedged the network into a flat region).</li>
  <li><strong>Noisy/oscillating</strong> → learning rate too high. Halve it and try again.</li>
  <li><strong>Goes to NaN</strong> → numerical instability, probably mixed precision related (Module 14).</li>
  <li><strong>Slowly decreases but doesn't reach zero</strong> → not enough capacity for the batch (try a bigger model — sometimes the issue is genuinely small batch with redundant samples).</li>
</ul>

<p>Sixty seconds of compute saves you from spending a weekend wondering why a million-step run isn't learning. Run it before every new architecture, before every config change, before every restart of training on a new dataset. <strong>It is the cheapest debugging tool in the entire course.</strong></p>

<div class="warn">
<strong>"It overfits, ship it" is not the rule.</strong> Overfit-a-batch tells you the gradients flow correctly, the loss is differentiable, and the model has capacity. It does <em>not</em> tell you the architecture is good, the data is right, or the task is well-posed. It's a <em>necessary</em> sanity check, not a sufficient one.
</div>

<h2>Reflex 2: log the right things at the right cadence</h2>

<p>"Log everything, then look at it later" generates 50 GB of irrelevant numbers and one 30-minute scroll session. "Log the four things that actually matter, every step" gives you debuggable training.</p>

<div class="table-wrap">
<table>
<caption>What to log, and how often</caption>
<thead><tr><th>Quantity</th><th>Cadence</th><th>What it tells you</th></tr></thead>
<tbody>
<tr><td><strong>Loss</strong></td><td>Every step</td><td>Is it learning? Is it spiking? Is it NaN?</td></tr>
<tr><td><strong>Learning rate</strong></td><td>Every step (or every N)</td><td>Is the schedule doing what you think?</td></tr>
<tr><td><strong>Gradient norm</strong> (pre-clip)</td><td>Every step (or every N)</td><td>Is the model exploding? Did clipping save you?</td></tr>
<tr><td><strong>Throughput</strong> (steps/sec, tokens/sec)</td><td>Every N steps</td><td>Is the GPU starving? Did something get slower?</td></tr>
<tr><td><strong>Weight / gradient histograms</strong></td><td>Every 100-1000 steps</td><td>Are layers dying? Are gradients vanishing?</td></tr>
<tr><td><strong>Validation loss / metric</strong></td><td>Every N epochs (or N steps)</td><td>Is overfitting starting? Is real performance improving?</td></tr>
<tr><td><strong>Sample outputs</strong> (for generative)</td><td>Every checkpoint</td><td>Is the model producing reasonable outputs at all?</td></tr>
</tbody>
</table>
</div>

<p>The first three matter most. <code>loss</code> shows you if anything is happening. <code>lr</code> confirms the schedule. <code>grad_norm</code> catches instability before it becomes NaN.</p>

<pre><code><span class="kw">def</span> <span class="fn">log_step</span>(step, loss, opt, model, log_every=<span class="num">10</span>):
    <span class="kw">if</span> step % log_every == <span class="num">0</span>:
        lr = opt.param_groups[<span class="num">0</span>][<span class="str">'lr'</span>]
        grad_norm = torch.nn.utils.<span class="fn">clip_grad_norm_</span>(model.<span class="fn">parameters</span>(), <span class="fn">float</span>(<span class="str">'inf'</span>))
        <span class="com"># clip_grad_norm_ with inf max_norm just measures, doesn't clip</span>
        <span class="fn">print</span>(<span class="str">f"step {step}: loss {loss:.4f}  lr {lr:.2e}  |g| {grad_norm:.3f}"</span>)</code></pre>

<p>Note the trick: <code>clip_grad_norm_</code> with <code>max_norm=inf</code> returns the total grad norm without actually clipping. Cheap way to monitor. In real code you'd then call the actual clip with finite max_norm.</p>

<div class="warn">
<strong>Don't log every step in production.</strong> Logging via <code>print</code> or even W&B at every step is fine for debugging — for a 1M-step run it's ~30 MB of log data and a few percent overhead. But if you log <code>.item()</code> on a CUDA tensor every step you're forcing a CPU-GPU sync (M3) — and that <em>can</em> visibly slow training. Use <code>log_every=10</code> or higher in production. Reserve every-step logging for debugging.
</div>

<h2>Reflex 3: validation loop discipline</h2>

<p>Three rules:</p>

<ol>
  <li><code>model.eval()</code> at the start, <code>model.train()</code> at the end. Otherwise BN running stats keep updating, dropout keeps firing.</li>
  <li><code>with torch.no_grad():</code> around the whole thing. Otherwise you build a graph for activations you'll throw away.</li>
  <li>Don't update training state during validation. No optimizer step. No scheduler step. Definitely no zero_grad on training-state grads.</li>
</ol>

<pre><code><span class="kw">@</span>torch.no_grad()
<span class="kw">def</span> <span class="fn">validate</span>(model, val_loader, device):
    model.<span class="fn">eval</span>()
    total_loss, total_correct, total_count = <span class="num">0.0</span>, <span class="num">0</span>, <span class="num">0</span>
    <span class="kw">for</span> x, y <span class="kw">in</span> val_loader:
        x, y = x.<span class="fn">to</span>(device), y.<span class="fn">to</span>(device)
        out = <span class="fn">model</span>(x)
        loss = F.<span class="fn">cross_entropy</span>(out, y, reduction=<span class="str">'sum'</span>)
        total_loss += loss.<span class="fn">item</span>()
        total_correct += (out.<span class="fn">argmax</span>(-<span class="num">1</span>) == y).<span class="fn">sum</span>().<span class="fn">item</span>()
        total_count += y.<span class="fn">size</span>(<span class="num">0</span>)
    model.<span class="fn">train</span>()                    <span class="com"># RESET to train mode!</span>
    <span class="kw">return</span> total_loss / total_count, total_correct / total_count</code></pre>

<p>The <code>@torch.no_grad()</code> decorator is cleaner than wrapping the body in a context manager. Note the <code>reduction='sum'</code> on the loss — we accumulate per-sample sums and divide by total count at the end, which is correct even when the last batch is short. Using <code>reduction='mean'</code> and averaging the means would give wrong weights for the last batch.</p>

<p>The <code>model.train()</code> at the end is the most common forgotten step. Forgetting it means training continues in eval mode for the rest of the epoch — Dropout off, BN frozen — which slowly degrades training without any obvious symptom. Use a context-manager helper if you want bulletproof:</p>

<pre><code><span class="kw">from</span> contextlib <span class="kw">import</span> contextmanager

<span class="kw">@</span>contextmanager
<span class="kw">def</span> <span class="fn">eval_mode</span>(model):
    was_training = model.training
    model.<span class="fn">eval</span>()
    <span class="kw">try</span>:
        <span class="kw">yield</span>
    <span class="kw">finally</span>:
        model.<span class="fn">train</span>(was_training)        <span class="com"># restore the previous mode</span>

<span class="kw">with</span> <span class="fn">eval_mode</span>(model), torch.no_grad():
    <span class="com"># validation logic here</span>
    ...</code></pre>

<p>This pattern survives exceptions and nested calls. Worth using in any non-toy training loop.</p>

<h2>Reflex 4: gradient accumulation</h2>

<p>Sometimes the batch size you want exceeds GPU memory. The fix is to split the batch, accumulate gradients across micro-batches, and step once. <em>This is exactly equivalent</em> to a single big-batch step (modulo the per-batch-norm statistics in BatchNorm — for LayerNorm/RMSNorm models, it's mathematically identical).</p>

<pre><code>accum_steps = <span class="num">4</span>          <span class="com"># effective batch = micro_batch * 4</span>

<span class="kw">for</span> step, batch <span class="kw">in</span> <span class="fn">enumerate</span>(loader):
    x, y = batch
    <span class="kw">with</span> torch.<span class="fn">autocast</span>(<span class="str">'cuda'</span>, dtype=torch.bfloat16):
        out = <span class="fn">model</span>(x)
        loss = F.<span class="fn">cross_entropy</span>(out, y) / accum_steps    <span class="com"># normalize!</span>
    loss.<span class="fn">backward</span>()                                     <span class="com"># accumulates into .grad</span>

    <span class="kw">if</span> (step + <span class="num">1</span>) % accum_steps == <span class="num">0</span>:
        torch.nn.utils.<span class="fn">clip_grad_norm_</span>(model.<span class="fn">parameters</span>(), <span class="num">1.0</span>)
        opt.<span class="fn">step</span>()
        opt.<span class="fn">zero_grad</span>()
        scheduler.<span class="fn">step</span>()</code></pre>

<p>Two non-obvious things. (1) <strong>Divide the loss by <code>accum_steps</code></strong> — otherwise the accumulated gradient is <code>accum_steps</code> times too big. The math is: gradient of the mean is the mean of the gradients, so to get the same effective gradient as a single big batch, you need to mean across micro-batches, which means scaling each contribution by <code>1/accum_steps</code> before backward. (2) <strong><code>opt.zero_grad()</code> only every <code>accum_steps</code> steps</strong>, not every step. The whole point is to let gradients accumulate.</p>

<p>This works even with mixed precision and DDP, with one caveat for DDP: you should wrap micro-batch backwards (except the last in a group) with <code>model.no_sync()</code> to avoid all-reducing partial gradients on every micro-batch. We'll see this in M16.</p>

<h2>Reflex 5: checkpoint and resume</h2>

<p>The five things you must save to resume bit-perfectly:</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 250" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">What goes in a checkpoint</text>

  <g transform="translate(50, 50)">
    <!-- 5 boxes side by side -->
    <rect x="0" y="0" width="120" height="80" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="60" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">model</text>
    <text x="60" y="42" text-anchor="middle" font-size="11" fill="#1a1612">.state_dict()</text>
    <text x="60" y="62" text-anchor="middle" font-size="10" fill="#1f5f5b">params + buffers</text>

    <rect x="130" y="0" width="120" height="80" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="190" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">optimizer</text>
    <text x="190" y="42" text-anchor="middle" font-size="11" fill="#1a1612">.state_dict()</text>
    <text x="190" y="62" text-anchor="middle" font-size="10" fill="#c1502e">moments, step</text>

    <rect x="260" y="0" width="120" height="80" fill="#fff8a8" stroke="#8c6512" stroke-width="2"/>
    <text x="320" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">scheduler</text>
    <text x="320" y="42" text-anchor="middle" font-size="11" fill="#1a1612">.state_dict()</text>
    <text x="320" y="62" text-anchor="middle" font-size="10" fill="#8c6512">last_epoch / step</text>

    <rect x="390" y="0" width="120" height="80" fill="#ffd5dc" stroke="#b85a6c" stroke-width="2"/>
    <text x="450" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">RNG state</text>
    <text x="450" y="42" text-anchor="middle" font-size="11" fill="#1a1612">all generators</text>
    <text x="450" y="62" text-anchor="middle" font-size="10" fill="#b85a6c">torch + py + np + cuda</text>

    <rect x="520" y="0" width="120" height="80" fill="#d3e9f5" stroke="#133e3b" stroke-width="2"/>
    <text x="580" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">step counter</text>
    <text x="580" y="42" text-anchor="middle" font-size="11" fill="#1a1612">int</text>
    <text x="580" y="62" text-anchor="middle" font-size="10" fill="#133e3b">+ best_metric, etc.</text>
  </g>

  <text x="370" y="180" font-family="'Caveat', cursive" font-size="22" fill="#c1502e" text-anchor="middle">skip ANY one of these → silent divergence after restart</text>
  <text x="370" y="210" font-family="'Caveat', cursive" font-size="20" fill="#1a1612" text-anchor="middle">disk cost: rounding error. correctness benefit: enormous.</text>
</svg>
</div>

<p>The complete checkpoint:</p>

<pre><code><span class="kw">def</span> <span class="fn">save_checkpoint</span>(path, model, optimizer, scheduler, step, **extra):
    state = {
        <span class="str">'model'</span>: model.<span class="fn">state_dict</span>(),
        <span class="str">'optimizer'</span>: optimizer.<span class="fn">state_dict</span>(),
        <span class="str">'scheduler'</span>: scheduler.<span class="fn">state_dict</span>(),
        <span class="str">'step'</span>: step,
        <span class="str">'rng'</span>: {
            <span class="str">'torch'</span>:    torch.<span class="fn">get_rng_state</span>(),
            <span class="str">'cuda'</span>:     torch.cuda.<span class="fn">get_rng_state_all</span>() <span class="kw">if</span> torch.cuda.<span class="fn">is_available</span>() <span class="kw">else</span> <span class="kw">None</span>,
            <span class="str">'numpy'</span>:    np.random.<span class="fn">get_state</span>(),
            <span class="str">'python'</span>:   random.<span class="fn">getstate</span>(),
        },
        **extra,             <span class="com"># best_val_loss, config dict, etc.</span>
    }
    torch.<span class="fn">save</span>(state, path)

<span class="kw">def</span> <span class="fn">load_checkpoint</span>(path, model, optimizer, scheduler):
    state = torch.<span class="fn">load</span>(path, weights_only=<span class="kw">False</span>)   <span class="com"># can't be True; rng state is non-tensor</span>
    model.<span class="fn">load_state_dict</span>(state[<span class="str">'model'</span>])
    optimizer.<span class="fn">load_state_dict</span>(state[<span class="str">'optimizer'</span>])
    scheduler.<span class="fn">load_state_dict</span>(state[<span class="str">'scheduler'</span>])
    torch.<span class="fn">set_rng_state</span>(state[<span class="str">'rng'</span>][<span class="str">'torch'</span>])
    <span class="kw">if</span> state[<span class="str">'rng'</span>][<span class="str">'cuda'</span>] <span class="kw">is not</span> <span class="kw">None</span>:
        torch.cuda.<span class="fn">set_rng_state_all</span>(state[<span class="str">'rng'</span>][<span class="str">'cuda'</span>])
    np.random.<span class="fn">set_state</span>(state[<span class="str">'rng'</span>][<span class="str">'numpy'</span>])
    random.<span class="fn">setstate</span>(state[<span class="str">'rng'</span>][<span class="str">'python'</span>])
    <span class="kw">return</span> state[<span class="str">'step'</span>], {k: v <span class="kw">for</span> k, v <span class="kw">in</span> state.<span class="fn">items</span>()
                              <span class="kw">if</span> k <span class="kw">not in</span> {<span class="str">'model'</span>, <span class="str">'optimizer'</span>, <span class="str">'scheduler'</span>, <span class="str">'rng'</span>}}</code></pre>

<p>Two notes. (1) <code>weights_only=False</code> — the RNG states aren't tensors, so we can't use the safe-mode loader. Make sure you trust the source of the file. (2) <code>set_rng_state_all</code> for CUDA covers multi-GPU; <code>set_rng_state</code> would only set device 0.</p>

<h3>Resuming the loader</h3>

<p>Even with all five buckets restored, there's one last hiccup: the <code>DataLoader</code>'s internal state. If you saved at step 50,000 and resume, you want the loader to start from sample 50,000 × batch_size, not from the top.</p>

<p>For map-style datasets, this is straightforward: skip-ahead by re-running the sampler N times, or use a <code>StatefulDataLoader</code> (the <code>torchdata</code> library has this; PyTorch core is adding it gradually). Pragmatic shortcut for most cases: just save the epoch number, restart from the beginning of that epoch. The first few thousand samples seen will differ from the original run, but the data was already shuffled, so this is usually fine.</p>

<p>For iterable datasets (streaming), it's harder — you need the dataset to know its position and resume from there. This is one of several reasons big LLM pretraining uses careful sharding with deterministic order so you can resume by skipping the right number of batches.</p>

<div class="ndq">
<h4>About checkpointing</h4>

<p class="q">How often should I checkpoint?</p>
<p class="a">Two answers. <em>Save-every-N-steps</em> for resumability — a 1-2 hour rate is reasonable, depending on how expensive a re-run would be. <em>Save-best-validation</em> as a separate file — overwrite this whenever val improves, so you have your "best so far" model decoupled from the most recent. Most teams maintain both: <code>latest.pt</code> for resumption, <code>best.pt</code> for evaluation.</p>

<p class="q">Should I save the loss history / config / hyperparameters?</p>
<p class="a">Yes — they cost nothing and answer the "what was I doing" question 6 months later. Throw them in <code>extra</code>: <code>save_checkpoint(..., config=cfg, loss_history=losses)</code>.</p>

<p class="q">My checkpoint is 50 GB. Anything I can drop?</p>
<p class="a">If it's an LLM-scale model, the optimizer state (Adam moments) is ~2× the model size. You can save it less frequently than the model itself — e.g., every 10× longer interval. The model alone (state_dict) is enough for inference; the optimizer state is only needed for training resumption. For deployment-only checkpoints, drop the optimizer.</p>

<p class="q">What about FSDP / DDP wrapped models?</p>
<p class="a">DDP: get the underlying state_dict via <code>model.module.state_dict()</code> (or strip the <code>module.</code> prefix on load — M7). FSDP: use the FSDP-specific state_dict APIs (<code>FullyShardedDataParallel.set_state_dict_type</code>) — Module 17. Both add complexity. Plan for it before you scale up.</p>
</div>

<h2>EMA: a thing outside the optimizer</h2>

<p>Exponential Moving Average of model weights. Maintain a separate copy of the model that's a slow-moving smoothed version of the actual training weights. Use it for evaluation and final deployment.</p>

<pre><code><span class="kw">class</span> <span class="ty">EMA</span>:
    <span class="kw">def</span> <span class="fn">__init__</span>(self, model, decay=<span class="num">0.999</span>):
        self.decay = decay
        self.shadow = {n: p.<span class="fn">clone</span>().<span class="fn">detach</span>() <span class="kw">for</span> n, p <span class="kw">in</span> model.<span class="fn">named_parameters</span>() <span class="kw">if</span> p.requires_grad}

    <span class="kw">@</span>torch.no_grad()
    <span class="kw">def</span> <span class="fn">update</span>(self, model):
        <span class="kw">for</span> n, p <span class="kw">in</span> model.<span class="fn">named_parameters</span>():
            <span class="kw">if</span> p.requires_grad:
                self.shadow[n].<span class="fn">mul_</span>(self.decay).<span class="fn">add_</span>(p.<span class="fn">data</span>, alpha=<span class="num">1</span> - self.decay)

    <span class="kw">def</span> <span class="fn">apply_to</span>(self, model):
        <span class="com"># temporarily swap in the EMA weights for evaluation</span>
        ...

ema = <span class="fn">EMA</span>(model, decay=<span class="num">0.999</span>)
<span class="kw">for</span> step, batch <span class="kw">in</span> <span class="fn">enumerate</span>(loader):
    <span class="com"># normal forward/backward/step ...</span>
    opt.<span class="fn">step</span>()
    ema.<span class="fn">update</span>(model)         <span class="com"># after each optimizer step</span></code></pre>

<p>Why? The optimizer's noisy updates jitter the model around the actual minimum. The EMA smooths over the trajectory, often giving better validation performance than the "live" weights. Decay around 0.999-0.9999 is typical (so the shadow weights are an EMA over the last ~1000-10000 steps). Standard in image generation (Stable Diffusion's UNet uses it), some RL setups, and increasingly in LLM training.</p>

<p>EMA weights need their own checkpoint entry — they're a separate state, not part of model.state_dict() unless you put them there as buffers (M7). Most teams keep them as a side dict.</p>

<h2>Deterministic vs fast: pick once, commit</h2>

<p>From M3: deterministic GPU operations cost throughput. The choice you make for a training run depends on what you're optimizing.</p>

<div class="table-wrap">
<table>
<caption>Deterministic vs fast — pick by use case</caption>
<thead><tr><th>Scenario</th><th>Setting</th><th>Why</th></tr></thead>
<tbody>
<tr><td>Debugging</td><td>Deterministic</td><td>Reproduce the bug to fix it</td></tr>
<tr><td>Research / paper experiments</td><td>Deterministic if affordable</td><td>Bit-equality across runs is gold for science</td></tr>
<tr><td>Production training</td><td>Fast</td><td>Throughput dominates; statistical reproducibility is enough</td></tr>
<tr><td>Comparing two configs A/B</td><td>Fast, run multiple seeds</td><td>Variance from non-determinism &lt; variance from initial random seed</td></tr>
</tbody>
</table>
</div>

<p>The pragmatic rule: <em>start non-deterministic, fast</em>. If you suspect a bug or need to reproduce a specific run, flip the determinism flags from M3. Don't make production training slow for the comfort of theoretical reproducibility.</p>

<h2>Putting it all together: the trustworthy training script</h2>

<pre><code><span class="kw">def</span> <span class="fn">train</span>(config, resume_from=<span class="kw">None</span>):
    <span class="com"># 1. Setup</span>
    <span class="fn">set_seed</span>(config.seed)
    model = <span class="fn">build_model</span>(config).<span class="fn">to</span>(device)
    opt = <span class="fn">build_optimizer</span>(model, config)
    scheduler = <span class="fn">build_scheduler</span>(opt, config)
    loader = <span class="fn">build_loader</span>(config)
    val_loader = <span class="fn">build_val_loader</span>(config)
    ema = <span class="fn">EMA</span>(model, decay=config.ema_decay) <span class="kw">if</span> config.use_ema <span class="kw">else</span> <span class="kw">None</span>

    <span class="com"># 2. Resume if requested</span>
    start_step = <span class="num">0</span>
    <span class="kw">if</span> resume_from:
        start_step, _ = <span class="fn">load_checkpoint</span>(resume_from, model, opt, scheduler)

    <span class="com"># 3. Sanity check (skip if resuming from far in)</span>
    <span class="kw">if</span> start_step == <span class="num">0</span>:
        <span class="fn">overfit_one_batch</span>(model, opt, loader)
        <span class="fn">reinit_optimizer</span>(opt, model)         <span class="com"># reset moments after sanity check</span>

    <span class="com"># 4. Main loop</span>
    model.<span class="fn">train</span>()
    <span class="kw">for</span> step, batch <span class="kw">in</span> <span class="fn">enumerate</span>(loader, start=start_step):
        loss = <span class="fn">train_step</span>(model, opt, scheduler, batch, config)
        <span class="kw">if</span> ema:
            ema.<span class="fn">update</span>(model)

        <span class="kw">if</span> step % config.log_every == <span class="num">0</span>:
            <span class="fn">log_step</span>(step, loss, opt, model)

        <span class="kw">if</span> step % config.val_every == <span class="num">0</span> <span class="kw">and</span> step &gt; <span class="num">0</span>:
            val_loss, val_acc = <span class="fn">validate</span>(model, val_loader, device)
            <span class="fn">log_val</span>(step, val_loss, val_acc)

        <span class="kw">if</span> step % config.ckpt_every == <span class="num">0</span> <span class="kw">and</span> step &gt; <span class="num">0</span>:
            <span class="fn">save_checkpoint</span>(<span class="str">f"latest.pt"</span>, model, opt, scheduler, step,
                            config=config, ema=ema.shadow <span class="kw">if</span> ema <span class="kw">else</span> <span class="kw">None</span>)

        <span class="kw">if</span> step &gt;= config.max_steps:
            <span class="kw">break</span></code></pre>

<p>That's the skeleton. Real codebases dress it up — config systems, distributed wrappers, callbacks for plugins — but every modification is to a piece of <em>this</em>. Internalize this loop and you can read any training framework.</p>

<h2>Code Magnets: assemble a robust train_step</h2>

<p>Build the inner training step for a transformer with mixed precision, gradient accumulation (4 micro-batches), and clipping.</p>

<div class="magnets">
<p>Arrange the magnets into the function body. Two are red herrings.</p>

<div class="magnet-pool">
  <span class="magnet">def train_step(model, opt, scheduler, batch, accum_steps, step):</span>
  <span class="magnet">    x, y = [t.to(device, non_blocking=True) for t in batch]</span>
  <span class="magnet">    with torch.autocast('cuda', dtype=torch.bfloat16):</span>
  <span class="magnet">        loss = F.cross_entropy(model(x), y) / accum_steps</span>
  <span class="magnet">        loss = F.cross_entropy(model(x), y)</span>
  <span class="magnet">    loss.backward()</span>
  <span class="magnet">    if (step + 1) % accum_steps == 0:</span>
  <span class="magnet">        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)</span>
  <span class="magnet">        opt.step(); opt.zero_grad(); scheduler.step()</span>
  <span class="magnet">        opt.zero_grad(); opt.step(); scheduler.step()</span>
  <span class="magnet">    return loss.item() * accum_steps</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">def</span> <span class="fn">train_step</span>(model, opt, scheduler, batch, accum_steps, step):
    x, y = [t.<span class="fn">to</span>(device, non_blocking=<span class="kw">True</span>) <span class="kw">for</span> t <span class="kw">in</span> batch]
    <span class="kw">with</span> torch.<span class="fn">autocast</span>(<span class="str">'cuda'</span>, dtype=torch.bfloat16):
        loss = F.<span class="fn">cross_entropy</span>(<span class="fn">model</span>(x), y) / accum_steps
    loss.<span class="fn">backward</span>()
    <span class="kw">if</span> (step + <span class="num">1</span>) % accum_steps == <span class="num">0</span>:
        torch.nn.utils.<span class="fn">clip_grad_norm_</span>(model.<span class="fn">parameters</span>(), <span class="num">1.0</span>)
        opt.<span class="fn">step</span>(); opt.<span class="fn">zero_grad</span>(); scheduler.<span class="fn">step</span>()
    <span class="kw">return</span> loss.<span class="fn">item</span>() * accum_steps</code></pre>
<p>The traps:</p>
<ul>
  <li><code>loss = F.cross_entropy(model(x), y)</code> without dividing by accum_steps gives a 4× too-large gradient. The accumulated gradient must be the <em>mean</em> across micro-batches.</li>
  <li><code>opt.zero_grad(); opt.step(); scheduler.step()</code> reverses the order — <code>step()</code> needs gradients in <code>.grad</code>, but <code>zero_grad()</code> would have just wiped them.</li>
</ul>
<p>Notice the <code>loss.item() * accum_steps</code> at the end — we scaled the loss for backward, but for logging we want the unscaled value. Multiply back to undo. Forgetting this gives logs that look like the loss is mysteriously tiny.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each training-loop concept to its real purpose.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real purpose</div>

  <div>Overfit a single batch</div>
  <div>A. Smoothed copy of model weights, often used for final evaluation.</div>

  <div>Gradient accumulation</div>
  <div>B. Cheapest sanity check: confirms gradients flow correctly through the model.</div>

  <div>RNG state in checkpoint</div>
  <div>C. Simulate a larger effective batch size when memory is the bottleneck.</div>

  <div>EMA</div>
  <div>D. Required for bit-perfect resumption — without it, restarted training silently diverges.</div>

  <div>model.eval() in validation</div>
  <div>E. Disables Dropout and switches BN to running stats; pair with no_grad().</div>

  <div>set_epoch on DistributedSampler</div>
  <div>F. Ensures each epoch shuffles differently across distributed ranks.</div>

  <div>loss / accum_steps</div>
  <div>G. Normalizes the per-micro-batch contribution so accumulated gradient = mean.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>Overfit a single batch</strong> → B<br>
<strong>Gradient accumulation</strong> → C<br>
<strong>RNG state in checkpoint</strong> → D<br>
<strong>EMA</strong> → A<br>
<strong>model.eval() in validation</strong> → E<br>
<strong>set_epoch on DistributedSampler</strong> → F<br>
<strong>loss / accum_steps</strong> → G
</p>
<p>The mental shortcut: <em>overfit-batch is the smoke test, accumulation is virtual batch size, RNG is for resumption, EMA is smoothed weights, eval mode is layer behavior, set_epoch fixes shuffle determinism, divide-by-accum normalizes the accumulated gradient</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team's training run resumes from checkpoint at step 50,000 and the loss spikes from 2.1 to 4.5 in the first 100 steps before recovering. What did they probably forget to save?</p>
<details class="answer"><summary>show answer</summary>
<p>Optimizer state. Without restoring the Adam <code>m</code> and <code>v</code> buffers, the optimizer effectively starts fresh — its second moment is 0, bias correction is back to early-step territory, and the first few steps after restart are huge. The loss spike then a recovery is the textbook signature of "model weights restored, optimizer state lost." Save <code>opt.state_dict()</code> too. Same problem can happen for the scheduler if you don't save its <code>last_epoch</code>.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Why does the validation loss curve in your training look noisier than the training loss curve? They should be smoother because you're averaging over the validation set...</p>
<details class="answer"><summary>show answer</summary>
<p>Most likely you're using <code>reduction='mean'</code> per batch and then averaging the per-batch means — but the last batch is shorter than the rest. The result is that the last batch is over-weighted in the average. Use <code>reduction='sum'</code> per batch, accumulate, divide by total sample count at the end. The validation loss will still have noise (a finite val set), but at least it'll be the right number. Alternatively: drop the last batch (<code>drop_last=True</code> on the val loader) — fine if your val set is large.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> You're using gradient accumulation with <code>accum_steps=8</code>. You log the loss as <code>loss.item()</code> every step. Why is the curve 8× smaller than expected?</p>
<details class="answer"><summary>show answer</summary>
<p>Because you divided the loss by <code>accum_steps</code> before backward (correct) but then logged the divided value (wrong for display). For logging, multiply back by <code>accum_steps</code>: <code>logged_loss = loss.item() * accum_steps</code>. Or compute and log the un-normalized loss separately. This is one of those bugs that makes training look weird but doesn't affect correctness — and people stare at the loss curve for an hour before realizing.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> Implement a context manager <code>training_state(model, training)</code> that switches the model into the given mode (True = train, False = eval) and restores the previous mode on exit, even on exception.</p>
<details class="answer"><summary>show answer</summary>
<pre><code><span class="kw">from</span> contextlib <span class="kw">import</span> contextmanager

<span class="kw">@</span>contextmanager
<span class="kw">def</span> <span class="fn">training_state</span>(model, training):
    was_training = model.training
    model.<span class="fn">train</span>(training)
    <span class="kw">try</span>:
        <span class="kw">yield</span>
    <span class="kw">finally</span>:
        model.<span class="fn">train</span>(was_training)

<span class="com"># Use:</span>
<span class="kw">with</span> <span class="fn">training_state</span>(model, <span class="kw">False</span>), torch.no_grad():
    val_loss = <span class="fn">validate</span>(model, val_loader)
<span class="com"># model is back to its previous training state, regardless of exceptions</span></code></pre>
<p>This pattern — capture-state, set-state, restore-state in <code>finally</code> — is the right way to write any "temporarily change something" helper. Useful for layers in inference mode, frozen-then-thawed parameter groups, autocast contexts, etc.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>The training loop is <strong>ten lines</strong>; the discipline around it is what matters: sanity checks, logging, validation hygiene, accumulation, checkpointing.</li>
  <li><strong>Overfit a single batch.</strong> Sixty seconds of compute, catches gradient/architecture/init bugs before you spend a weekend on them. Loss must approach zero. Plateau / noise / NaN each tells you something specific.</li>
  <li><strong>Log loss, LR, gradient norm</strong> at every step (or every N for production). Throughput, weight histograms, val metrics at coarser cadences.</li>
  <li>Avoid <code>.item()</code> on a CUDA tensor every step — it forces sync. Log every 10 steps in production.</li>
  <li><strong>Validation discipline:</strong> <code>model.eval()</code> at start, <code>train()</code> at end (or use a context manager), <code>@torch.no_grad()</code> for the body. Use <code>reduction='sum'</code> + divide by total count for correct averaging.</li>
  <li><strong>Gradient accumulation</strong>: divide loss by <code>accum_steps</code>, backward each micro-batch, <code>step()</code>+<code>zero_grad()</code> only every <code>accum_steps</code>. Mathematically equivalent to a big batch (modulo BN).</li>
  <li><strong>The 5-bucket checkpoint</strong>: model, optimizer, scheduler, RNG (all four — torch + cuda + numpy + python), step counter. Skip any one → silent divergence on resume.</li>
  <li>Two checkpoint files: <code>latest.pt</code> for resumption (overwrite often), <code>best.pt</code> for evaluation (overwrite when val improves).</li>
  <li><code>DataLoader</code> resumption is the hardest part — for map-style, restart from epoch top is usually fine. For streaming, plan for it from day one.</li>
  <li><strong>EMA</strong> = exponential moving average of weights, decay 0.999-0.9999. Standard in image gen, increasingly in LLM training. Save separately from model state_dict.</li>
  <li><strong>Determinism</strong> is a tradeoff: pick fast for production, deterministic for debugging or paper-quality reproduction. Don't pay the throughput tax in production.</li>
  <li><strong>The reflex</strong>: when something looks wrong in training, run overfit-a-batch first. If that works, the loop is fine — the bug is probably in data or hyperparameters. If it fails, the bug is in model, init, or autograd.</li>
</ul>
</div>

<p>That closes Part IV. You can now ship a trustworthy training run: pipeline, model, loss, optimizer, schedule, accumulation, checkpoint, validation, EMA. Everything from M1 onwards has earned its place in this loop.</p>

<p>Part V opens with Module 12: where does all that GPU memory actually go? Params, gradients, optimizer states, activations, workspace — the formula that lets you predict OOM before you hit it, and the activation-checkpointing trick that buys you depth at the cost of compute.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">11</span>
  <span>Training loops you can trust</span>
</div>
"""

emit("11_training_loops", "Module 11 — Training loops you can trust", BODY)
