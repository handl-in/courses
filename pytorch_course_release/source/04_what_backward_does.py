#!/usr/bin/env python3
"""Module 04: What .backward() actually does — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part II · Module 04</div>
  <h1 class="module-title">What <code>.backward()</code> <em>actually does</em></h1>
  <p class="module-sub">— the dynamic graph, leaf vs non-leaf, and reverse-mode autodiff walked by hand on a tiny example until it stops being magic</p>
</div>

<p>Here is the exact moment most PyTorch users decide to stop asking questions: someone says "autograd computes gradients automatically using reverse-mode automatic differentiation," they nod, and they go back to copying training loops from blog posts. They never <em>understand</em> what <code>loss.backward()</code> did. So when their gradient is <code>None</code>, or their custom layer's backward is wrong, or their memory blows up because of <code>retain_graph=True</code>, they have nowhere to stand.</p>

<p>That ends today. By the end of this module you will be able to take a small expression, write down its computational graph by hand, walk the backward pass by hand, and tell PyTorch exactly which gradient buffer should hold what value. Once that's concrete, every later autograd topic — custom Functions, double backward, hooks, gradient checkpointing — is just a refinement.</p>

<div class="keyidea">
<code>.backward()</code> is just <strong>the chain rule, applied to a graph that PyTorch built for you during the forward pass</strong>. Every operation produces a tensor and silently records "how to undo me" in a <code>grad_fn</code>. <code>.backward()</code> walks that graph in reverse from the loss, multiplying gradients along the way, and dumps the final values into the <code>.grad</code> attribute of every <em>leaf</em> tensor that asked for it. There is no other magic.
</div>

<h2>Meet the autograd cast</h2>

<p>Five characters in this module. Three are returning faces; two are new.</p>

<div class="character autograd">
  <div class="avatar">∂</div>
  <div>
    <p class="who">Autograd</p>
    <p class="name">"I'm finally on stage. Let me explain my job."</p>
    <p class="says">When you build a tensor expression, I follow along quietly, recording every operation and its inputs. When you call <code>.backward()</code> on something, I walk that recording <em>in reverse</em>, applying the chain rule at each step. I produce a number for every leaf tensor: "if you nudge this value by ε, the loss will change by approximately (gradient × ε)." That's it. I'm a tape recorder with calculus.</p>
  </div>
</div>

<div class="character" style="--c: #133e3b;">
  <div class="avatar" style="background: #133e3b; color: #fff;">G</div>
  <div>
    <p class="who">Graph</p>
    <p class="name">"I'm rebuilt from scratch on every forward pass."</p>
    <p class="says">This is the famous <em>dynamic</em> part of PyTorch. Other frameworks built me once and reused me. PyTorch builds me fresh every time — every forward pass creates a new me, with whatever Python control flow you wrote. <code>if</code>, <code>for</code>, recursion — they all work. The price is that I'm thrown away after each <code>backward()</code>. The benefit is that you can debug me with <code>print</code> statements like normal Python.</p>
  </div>
</div>

<div class="character" style="--c: #c1502e;">
  <div class="avatar" style="background: #c1502e; color: #fff;">⌬</div>
  <div>
    <p class="who">grad_fn</p>
    <p class="name">"I'm the 'how to undo me' note attached to every non-leaf tensor."</p>
    <p class="says">When you do <code>z = x + y</code>, I get attached to <code>z</code> as <code>z.grad_fn = &lt;AddBackward0&gt;</code>. I remember that I came from an addition, and I know that the gradient of an addition just passes through to both inputs unchanged. Every operation in PyTorch has its own grad_fn class — <code>MulBackward0</code>, <code>SumBackward0</code>, <code>MmBackward0</code>, hundreds of them. I'm how the graph stores its structure.</p>
  </div>
</div>

<div class="character" style="--c: #1f5f5b;">
  <div class="avatar" style="background: #1f5f5b; color: #fff;">L</div>
  <div>
    <p class="who">Leaf</p>
    <p class="name">"I'm a tensor at the root of the graph. I own a <code>.grad</code>."</p>
    <p class="says">If you created me directly with <code>torch.tensor(...)</code> or <code>torch.randn(...)</code> or <code>nn.Parameter(...)</code>, and I have <code>requires_grad=True</code>, then I'm a leaf. After <code>.backward()</code>, my <code>.grad</code> attribute holds the gradient of the loss with respect to me. The non-leaves <em>don't</em> get a <code>.grad</code> by default — they're just intermediates the graph used.</p>
  </div>
</div>

<div class="character" style="--c: #d4a017;">
  <div class="avatar" style="background: #d4a017;">∇</div>
  <div>
    <p class="who">Grad</p>
    <p class="name">"I'm a tensor too. I have the same shape as my parent."</p>
    <p class="says">I live on a Leaf's <code>.grad</code> attribute. I'm initially <code>None</code>. After your first <code>.backward()</code>, I exist. After the second one, I <em>accumulate</em> — yes, I add to whatever was there. That's the famous "you forgot to call <code>optimizer.zero_grad()</code>" bug. It's a feature, not a glitch — but you need to know about it.</p>
  </div>
</div>

<h2>The forward pass builds the graph</h2>

<p>Let's start small. Two scalars, one combination, one observation.</p>

<pre><code><span class="kw">import</span> torch

x = torch.tensor(<span class="num">2.0</span>, requires_grad=<span class="kw">True</span>)
y = torch.tensor(<span class="num">3.0</span>, requires_grad=<span class="kw">True</span>)

a = x + y          <span class="com"># a = 5.0</span>
z = a * y          <span class="com"># z = 15.0</span>

<span class="fn">print</span>(x.is_leaf, x.grad_fn)   <span class="com"># True None</span>
<span class="fn">print</span>(y.is_leaf, y.grad_fn)   <span class="com"># True None</span>
<span class="fn">print</span>(a.is_leaf, a.grad_fn)   <span class="com"># False &lt;AddBackward0 object at ...&gt;</span>
<span class="fn">print</span>(z.is_leaf, z.grad_fn)   <span class="com"># False &lt;MulBackward0 object at ...&gt;</span></code></pre>

<p>Every line of code above did <em>two</em> things: it computed a value, and it recorded a graph node. The recording is what makes the next steps work.</p>

<p>Here is the graph PyTorch built, in your head:</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 700 360" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrFwd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
  </defs>

  <text x="350" y="28" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">The graph after  z = (x + y) * y</text>

  <!-- Leaves -->
  <g transform="translate(120, 70)">
    <circle cx="40" cy="40" r="38" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2.5"/>
    <text x="40" y="38" text-anchor="middle" font-size="20" font-weight="700" fill="#1a1612">x</text>
    <text x="40" y="56" text-anchor="middle" font-size="13" fill="#1a1612">= 2.0</text>
    <text x="40" y="100" text-anchor="middle" font-size="11" fill="#6b5d4f">leaf, requires_grad</text>
  </g>

  <g transform="translate(420, 70)">
    <circle cx="40" cy="40" r="38" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2.5"/>
    <text x="40" y="38" text-anchor="middle" font-size="20" font-weight="700" fill="#1a1612">y</text>
    <text x="40" y="56" text-anchor="middle" font-size="13" fill="#1a1612">= 3.0</text>
    <text x="40" y="100" text-anchor="middle" font-size="11" fill="#6b5d4f">leaf, requires_grad</text>
  </g>

  <!-- a = x + y -->
  <g transform="translate(220, 180)">
    <rect x="0" y="0" width="120" height="60" fill="#fff8a8" stroke="#c1502e" stroke-width="2"/>
    <text x="60" y="26" text-anchor="middle" font-size="16" font-weight="700" fill="#1a1612">a = x + y</text>
    <text x="60" y="46" text-anchor="middle" font-size="13" fill="#8c3a20">grad_fn: AddBackward0</text>
  </g>

  <!-- z = a * y -->
  <g transform="translate(330, 280)">
    <rect x="0" y="0" width="120" height="60" fill="#fcecec" stroke="#c1502e" stroke-width="2.5"/>
    <text x="60" y="26" text-anchor="middle" font-size="16" font-weight="700" fill="#1a1612">z = a * y</text>
    <text x="60" y="46" text-anchor="middle" font-size="13" fill="#8c3a20">grad_fn: MulBackward0</text>
  </g>

  <!-- Forward arrows -->
  <path d="M 165 145 Q 200 165 240 180" fill="none" stroke="#1f5f5b" stroke-width="2" marker-end="url(#arrFwd)"/>
  <path d="M 460 145 Q 410 165 360 180" fill="none" stroke="#1f5f5b" stroke-width="2" marker-end="url(#arrFwd)"/>
  <!-- a → z -->
  <path d="M 320 240 Q 350 260 380 280" fill="none" stroke="#1f5f5b" stroke-width="2" marker-end="url(#arrFwd)"/>
  <!-- y → z (curved) -->
  <path d="M 460 145 Q 530 220 450 282" fill="none" stroke="#1f5f5b" stroke-width="2" stroke-dasharray="5 3" marker-end="url(#arrFwd)"/>

  <text x="600" y="220" font-family="'Caveat', cursive" font-size="20" fill="#1f5f5b">y is used twice!</text>
  <text x="600" y="242" font-family="'Caveat', cursive" font-size="20" fill="#1f5f5b">(once in a, once in z)</text>

  <text x="50" y="350" font-family="'Caveat', cursive" font-size="20" fill="#c1502e">forward direction →</text>
</svg>
</div>

<p>Two things to notice in that diagram. <strong>First</strong>: <code>x</code> and <code>y</code> are leaves (the green circles), <code>a</code> and <code>z</code> are non-leaves (the orange rectangles). Leaves are the ones you created directly; non-leaves are intermediate computations. <strong>Second</strong>: <code>y</code> is used twice — once to make <code>a</code>, once to make <code>z</code>. The graph remembers both edges. This will matter in 30 seconds.</p>

<div class="character" style="--c: #133e3b;">
  <div class="avatar" style="background: #133e3b; color: #fff;">G</div>
  <div>
    <p class="who">Graph</p>
    <p class="name">"By the way — I only contain <em>tensors that need gradients</em>."</p>
    <p class="says">If neither <code>x</code> nor <code>y</code> had <code>requires_grad=True</code>, I wouldn't have been built at all. PyTorch tracks me lazily. The moment <em>one</em> input has <code>requires_grad</code>, the output gets a <code>grad_fn</code> and joins me. Otherwise the operation runs as plain math, no recording. That's why <code>torch.no_grad()</code> is a speed win for inference — no graph means no overhead.</p>
  </div>
</div>

<h2>The backward pass walked by hand</h2>

<p>Now the punch line. We want <code>dz/dx</code> and <code>dz/dy</code>. Let's compute them by hand <em>before</em> letting PyTorch do it, so we can check.</p>

<p>The function is <code>z = (x + y) * y</code>. Expanding: <code>z = xy + y²</code>. Calculus class:</p>

<ul>
  <li><code>∂z/∂x = y</code> — at <code>x=2, y=3</code>, this is <strong>3</strong>.</li>
  <li><code>∂z/∂y = x + 2y</code> — at <code>x=2, y=3</code>, this is <strong>2 + 6 = 8</strong>.</li>
</ul>

<p>Hold those numbers in your head: gradient on x should be 3, on y should be 8. Now let's see how the <em>chain rule</em> walks the graph backwards to produce the same answers.</p>

<h3>The chain rule, step by step</h3>

<p>Backward starts at the output, with seed gradient <code>1</code> (because <code>dz/dz = 1</code>). Then we walk each grad_fn in reverse, multiplying gradients as we go.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 720 380" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrBack" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#c1502e"/>
    </marker>
  </defs>

  <text x="360" y="26" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Backward pass: walk in reverse, multiply gradients</text>

  <!-- z node at top -->
  <g transform="translate(310, 50)">
    <rect x="0" y="0" width="120" height="50" fill="#fcecec" stroke="#c1502e" stroke-width="2.5"/>
    <text x="60" y="22" text-anchor="middle" font-size="14" font-weight="700" fill="#1a1612">z (output)</text>
    <text x="60" y="40" text-anchor="middle" font-size="12" fill="#8c3a20">seed grad: 1</text>
  </g>

  <!-- Step 1 annotation -->
  <text x="20" y="100" font-size="12" fill="#c1502e" font-weight="700">① At MulBackward (z = a * y):</text>
  <text x="20" y="118" font-size="12" fill="#1a1612">  ∂z/∂a = y = 3      |      ∂z/∂y = a = 5</text>
  <text x="20" y="135" font-size="12" fill="#1a1612">  upstream grad × local grad = 1·3 = 3 (to a),  1·5 = 5 (to y)</text>

  <!-- a node and y first contribution -->
  <g transform="translate(160, 170)">
    <rect x="0" y="0" width="120" height="50" fill="#fff8a8" stroke="#c1502e" stroke-width="2"/>
    <text x="60" y="22" text-anchor="middle" font-size="14" font-weight="700" fill="#1a1612">a (intermediate)</text>
    <text x="60" y="40" text-anchor="middle" font-size="12" fill="#8c3a20">grad: 3</text>
  </g>

  <g transform="translate(460, 170)">
    <circle cx="60" cy="25" r="32" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="60" y="22" text-anchor="middle" font-size="14" font-weight="700" fill="#1a1612">y</text>
    <text x="60" y="38" text-anchor="middle" font-size="11" fill="#1a1612">first: +5</text>
  </g>

  <!-- Arrows from z down -->
  <path d="M 350 100 Q 300 130 240 170" fill="none" stroke="#c1502e" stroke-width="2" marker-end="url(#arrBack)"/>
  <text x="265" y="135" font-size="11" fill="#c1502e" font-weight="700">3</text>
  <path d="M 410 100 Q 460 130 510 170" fill="none" stroke="#c1502e" stroke-width="2" marker-end="url(#arrBack)"/>
  <text x="455" y="135" font-size="11" fill="#c1502e" font-weight="700">5</text>

  <!-- Step 2 annotation -->
  <text x="20" y="240" font-size="12" fill="#c1502e" font-weight="700">② At AddBackward (a = x + y):</text>
  <text x="20" y="258" font-size="12" fill="#1a1612">  ∂a/∂x = 1   |   ∂a/∂y = 1   (addition just passes the gradient through)</text>
  <text x="20" y="275" font-size="12" fill="#1a1612">  upstream grad (3) × local grad (1) = 3 (to x),  3·1 = 3 (to y, 2nd contribution)</text>

  <!-- x and y second contribution -->
  <g transform="translate(70, 305)">
    <circle cx="40" cy="25" r="32" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="40" y="22" text-anchor="middle" font-size="14" font-weight="700" fill="#1a1612">x</text>
    <text x="40" y="38" text-anchor="middle" font-size="11" fill="#8c3a20">grad: 3</text>
  </g>

  <g transform="translate(280, 305)">
    <circle cx="40" cy="25" r="32" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="40" y="22" text-anchor="middle" font-size="14" font-weight="700" fill="#1a1612">y</text>
    <text x="40" y="38" text-anchor="middle" font-size="11" fill="#1a1612">2nd: +3</text>
  </g>

  <!-- arrows from a -->
  <path d="M 200 220 Q 160 270 120 305" fill="none" stroke="#c1502e" stroke-width="2" marker-end="url(#arrBack)"/>
  <text x="135" y="280" font-size="11" fill="#c1502e" font-weight="700">3</text>
  <path d="M 240 220 Q 280 270 310 305" fill="none" stroke="#c1502e" stroke-width="2" marker-end="url(#arrBack)"/>
  <text x="278" y="280" font-size="11" fill="#c1502e" font-weight="700">3</text>

  <!-- y total -->
  <text x="540" y="320" font-family="'Caveat', cursive" font-size="22" fill="#c1502e">y gets BOTH:</text>
  <text x="540" y="345" font-family="'Caveat', cursive" font-size="22" fill="#c1502e">5 + 3 = 8 ✓</text>
</svg>
</div>

<p>Walk through it slowly:</p>

<ol>
  <li><strong>Start at z</strong>, with seed gradient 1 (we always start from <code>dz/dz = 1</code>).</li>
  <li><strong>At MulBackward</strong> (the multiplication that made <code>z = a * y</code>): the local derivatives are <code>∂z/∂a = y = 3</code> and <code>∂z/∂y = a = 5</code>. We multiply each by the upstream gradient (1) and pass: <code>3</code> to <code>a</code>, <code>5</code> to <code>y</code>.</li>
  <li><strong>At AddBackward</strong> (the addition that made <code>a = x + y</code>): both local derivatives are 1 (addition passes the gradient through unchanged). We multiply by the upstream gradient (3): <code>3</code> to <code>x</code>, <code>3</code> to <code>y</code>.</li>
  <li><strong>x</strong> is a leaf — its <code>.grad</code> becomes <code>3</code>. ✓</li>
  <li><strong>y</strong> is a leaf — its <code>.grad</code> accumulates both contributions: <code>5 + 3 = 8</code>. ✓ (This is why "y used twice" mattered.)</li>
</ol>

<p>Now the punch line: do it in PyTorch and confirm.</p>

<pre><code>x = torch.tensor(<span class="num">2.0</span>, requires_grad=<span class="kw">True</span>)
y = torch.tensor(<span class="num">3.0</span>, requires_grad=<span class="kw">True</span>)
z = (x + y) * y

z.backward()

<span class="fn">print</span>(x.grad)    <span class="com"># tensor(3.) ✓</span>
<span class="fn">print</span>(y.grad)    <span class="com"># tensor(8.) ✓</span></code></pre>

<p>Same numbers. <em>That's all backward did</em>. It walked the graph, multiplied at each grad_fn, and added contributions when a leaf had multiple paths.</p>

<div class="brainpower">
<p><strong>Pause and predict.</strong> Suppose you call <code>z.backward()</code> a second time on the same graph. What happens?</p>
<p>It errors. By default, PyTorch frees the graph after backward to save memory. To run backward again on the same graph, you'd need <code>z.backward(retain_graph=True)</code>.</p>
<p>And if you ran <em>forward</em> again to rebuild the graph, then called backward? <code>x.grad</code> would become <code>6</code> (3 + 3) and <code>y.grad</code> would become <code>16</code> (8 + 8). Gradients <strong>accumulate</strong>. This is the "you forgot to call <code>optimizer.zero_grad()</code>" bug everyone meets once.</p>
</div>

<h2>Why it accumulates (and why it's a feature)</h2>

<p>The accumulation behavior surprises beginners. Why doesn't PyTorch <em>overwrite</em> <code>.grad</code> on each backward? Because in real training, you sometimes <em>want</em> to accumulate. Two examples you'll meet later:</p>

<ol>
  <li><strong>Gradient accumulation across micro-batches.</strong> If your batch is too big to fit in GPU memory, you split it into N micro-batches, run forward+backward on each, and let <code>.grad</code> accumulate the sum. After N iterations, you call <code>optimizer.step()</code>. This is how you simulate large batch training on small GPUs (Module 11).</li>
  <li><strong>Multiple loss terms.</strong> If your loss is <code>l1 + l2</code> and you computed them separately, you can call <code>l1.backward()</code> then <code>l2.backward()</code>. The two contributions add into <code>.grad</code> automatically.</li>
</ol>

<p>So accumulation is the default, and you opt out by zeroing the grads explicitly:</p>

<pre><code>optimizer.zero_grad()       <span class="com"># sets all param.grad to None (or 0, older API)</span>
loss.backward()
optimizer.step()</code></pre>

<p>That three-line dance is the heartbeat of every training loop in PyTorch. Now you know what each line is doing to the graph.</p>

<div class="warn">
<strong>Common foot-gun.</strong> If you forget <code>zero_grad()</code>, your model still trains — but with effectively-larger gradients. The optimizer takes bigger steps, training becomes unstable, and you spend an afternoon hunting a bug that's <em>visible in the loss curve but hard to identify</em>. The fix is one line. Add it to your training loop template once and never think about it again.
</div>

<h2>Leaf vs non-leaf, in detail</h2>

<p>One of the most confusing parts of autograd is which tensors get <code>.grad</code> and which don't. The rule is simple but easy to violate.</p>

<div class="table-wrap">
<table>
<caption>Who's a leaf?</caption>
<thead><tr><th>Tensor</th><th>Leaf?</th><th>Gets <code>.grad</code>?</th></tr></thead>
<tbody>
<tr><td><code>x = torch.tensor(2.0, requires_grad=True)</code></td><td>Yes</td><td>Yes (after backward)</td></tr>
<tr><td><code>x = torch.randn(3, requires_grad=True)</code></td><td>Yes</td><td>Yes</td></tr>
<tr><td><code>x = nn.Parameter(torch.randn(3))</code></td><td>Yes</td><td>Yes</td></tr>
<tr><td><code>x = some_tensor.detach()</code></td><td>Yes</td><td>Only if <code>requires_grad=True</code> set after</td></tr>
<tr><td><code>z = x + y</code> (where x, y require grad)</td><td><strong>No</strong></td><td>No (intermediate)</td></tr>
<tr><td><code>z = x.cuda()</code> (where x requires grad)</td><td><strong>No</strong></td><td>No — <code>cuda()</code> is an op</td></tr>
<tr><td><code>z = x.float()</code> (where x requires grad)</td><td><strong>No</strong></td><td>No — same reason</td></tr>
</tbody>
</table>
</div>

<p>That last batch of rows surprises everyone. If you do:</p>

<pre><code>x = torch.randn(<span class="num">3</span>, <span class="num">3</span>, requires_grad=<span class="kw">True</span>)
x = x.cuda()                       <span class="com"># reassign — original is now a non-leaf!</span>
loss = x.sum()
loss.backward()
<span class="fn">print</span>(x.grad)                       <span class="com"># None! 😱</span></code></pre>

<p>The <code>.cuda()</code> call <em>created a new tensor</em> with a grad_fn (<code>CopyBackwards</code>). The new tensor is non-leaf. The original leaf is now unreferenced and never receives a grad. The fix is to construct the tensor on the right device from the start, or use <code>nn.Parameter</code>:</p>

<pre><code><span class="com"># Right way</span>
x = torch.randn(<span class="num">3</span>, <span class="num">3</span>, device=<span class="str">'cuda'</span>, requires_grad=<span class="kw">True</span>)

<span class="com"># Or, for module parameters</span>
self.weight = nn.Parameter(torch.randn(<span class="num">3</span>, <span class="num">3</span>, device=<span class="str">'cuda'</span>))</code></pre>

<div class="ndq">
<h4>About leaves and grads</h4>

<p class="q">Can I get the gradient of an intermediate (non-leaf) tensor?</p>
<p class="a">Yes — call <code>.retain_grad()</code> on it before backward. Without that, PyTorch discards intermediate gradients to save memory. <code>z.retain_grad(); loss.backward(); print(z.grad)</code>. You'll see this trick in debugging and in implementations of things like Grad-CAM.</p>

<p class="q">Why does <code>requires_grad</code> propagate forward through ops?</p>
<p class="a">Because if <em>any</em> input to an op requires gradients, the output must also be tracked — otherwise you couldn't backprop through it. PyTorch sets <code>requires_grad=True</code> on the result automatically. The reverse is also true: if no inputs require grads, the result doesn't either, and no graph is built.</p>

<p class="q">What's the difference between <code>requires_grad=False</code> and <code>detach()</code>?</p>
<p class="a">Subtle but important. <code>requires_grad=False</code> on a leaf means "I'm not a parameter; don't track me as one." <code>detach()</code> creates a <em>new</em> tensor that shares storage but is cut from the graph — its <code>grad_fn</code> is None, so backward stops at it. You use <code>detach()</code> to deliberately break gradient flow, e.g., to stop gradients from flowing into a frozen part of a model, or to use a quantity as a "constant" in a loss.</p>

<p class="q">Can I have a tensor with <code>requires_grad=True</code> but no <code>.grad</code> after backward?</p>
<p class="a">Yes — if it's a leaf but the graph never reached it. For example, if your loss doesn't depend on a particular parameter (think: a layer that's effectively dead because of zero weights), backward never sends a gradient there. <code>.grad</code> stays <code>None</code>. This is also why <code>find_unused_parameters=True</code> matters in DDP (Module 16).</p>
</div>

<h2>The graph for vector ops, briefly</h2>

<p>Everything we just walked through generalizes from scalars to tensors. The chain rule still applies, but at each grad_fn the "local derivative" is now a Jacobian (a matrix or higher-rank tensor of partial derivatives). PyTorch never materializes the full Jacobian — that would be insane for a 1024×1024 weight matrix. Instead, each grad_fn implements a <em>vector-Jacobian product</em>: given the upstream gradient, it computes "what gradient would have produced this through me?" without ever building the full Jacobian.</p>

<p>For <code>z = a * y</code> with vector tensors:</p>

<ul>
  <li>Local "derivative" of <code>z</code> w.r.t. <code>a</code> is element-wise <code>y</code>.</li>
  <li>Backward at this node: <code>grad_a = upstream_grad * y</code> (element-wise multiply, same shape).</li>
  <li>Same idea for <code>grad_y = upstream_grad * a</code>.</li>
</ul>

<p>For matmul <code>z = A @ B</code>:</p>

<ul>
  <li><code>grad_A = upstream_grad @ B.T</code></li>
  <li><code>grad_B = A.T @ upstream_grad</code></li>
</ul>

<p>You don't have to memorize these. PyTorch knows them. But internalizing that <em>each grad_fn is just a function from upstream grad to downstream grad</em> is what lets you write your own (Module 6) and reason about memory cost (Module 12).</p>

<h2>Code Magnets: build the simplest possible loss step</h2>

<p>You're writing the most basic gradient descent step imaginable — one parameter, one loss, one update. Arrange the magnets into a working snippet that does <em>three</em> steps of gradient descent on <code>w</code>, with the goal of minimizing <code>(w - 5)²</code>.</p>

<div class="magnets">
<p>Use exactly the magnets needed. There are two red herrings.</p>

<div class="magnet-pool">
  <span class="magnet">w = torch.tensor(0.0, requires_grad=True)</span>
  <span class="magnet">w = torch.tensor(0.0)</span>
  <span class="magnet">for _ in range(3):</span>
  <span class="magnet">    loss = (w - 5) ** 2</span>
  <span class="magnet">    loss.backward()</span>
  <span class="magnet">    w.data -= 0.1 * w.grad</span>
  <span class="magnet">    w.grad = None</span>
  <span class="magnet">    w -= 0.1 * w.grad</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code>w = torch.tensor(<span class="num">0.0</span>, requires_grad=<span class="kw">True</span>)
<span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(<span class="num">3</span>):
    loss = (w - <span class="num">5</span>) ** <span class="num">2</span>
    loss.backward()
    w.data -= <span class="num">0.1</span> * w.grad
    w.grad = <span class="kw">None</span></code></pre>
<p>The two traps:</p>
<ul>
  <li><code>w = torch.tensor(0.0)</code> — without <code>requires_grad=True</code>, no graph is built and <code>w.grad</code> is always None.</li>
  <li><code>w -= 0.1 * w.grad</code> — this looks identical to <code>w.data -= ...</code> but it tries to do the update <em>through autograd</em>, which complains about an in-place op on a leaf that requires grad. The <code>.data</code> bypass is the lazy fix; the proper modern fix is <code>with torch.no_grad(): w -= ...</code>. We'll meet both in Module 5.</li>
</ul>
<p>The <code>w.grad = None</code> at the end of each iteration is what <code>optimizer.zero_grad(set_to_none=True)</code> does internally — it's slightly more efficient than zeroing because the next backward can just allocate fresh.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each autograd concept to what it actually is.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">What it is</div>

  <div>requires_grad</div>
  <div>A. The function attached to a non-leaf tensor that knows how to compute its backward.</div>

  <div>grad_fn</div>
  <div>B. A flag that tells PyTorch to track operations on this tensor for backward.</div>

  <div>is_leaf</div>
  <div>C. The accumulated gradient of the loss with respect to a leaf tensor.</div>

  <div>.grad</div>
  <div>D. True if the tensor was created directly (not produced by an op) — only leaves get .grad by default.</div>

  <div>.backward()</div>
  <div>E. Returns a new tensor that shares storage but is cut from the graph (grad_fn = None).</div>

  <div>.detach()</div>
  <div>F. Walks the graph in reverse from this tensor, populating .grad on every reachable leaf.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>requires_grad</strong> → B<br>
<strong>grad_fn</strong> → A<br>
<strong>is_leaf</strong> → D<br>
<strong>.grad</strong> → C<br>
<strong>.backward()</strong> → F<br>
<strong>.detach()</strong> → E
</p>
<p>The mental shortcut: <em>requires_grad</em> is the input switch, <em>grad_fn</em> is the recorded operation, <em>is_leaf</em> tells you if a tensor will receive grad, <em>.grad</em> is the result, <em>.backward()</em> is the action, <em>.detach()</em> is the cut.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> Write a function <code>grad(f, x)</code> that takes a Python function <code>f</code> from a scalar to a scalar and a Python number <code>x</code>, and returns the derivative at <code>x</code> using PyTorch autograd.</p>
<details class="answer"><summary>show answer</summary>
<pre><code><span class="kw">def</span> <span class="fn">grad</span>(f, x):
    t = torch.tensor(<span class="fn">float</span>(x), requires_grad=<span class="kw">True</span>)
    f(t).backward()
    <span class="kw">return</span> t.grad.item()

<span class="com"># Use:</span>
<span class="fn">grad</span>(<span class="kw">lambda</span> x: x ** <span class="num">3</span>, <span class="num">2.0</span>)   <span class="com"># 12.0  (3x² at x=2)</span></code></pre>
<p>This is essentially what <code>torch.func.grad</code> does in modern PyTorch (we'll meet that API in Module 5). Writing it yourself once cements the model.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Compute by hand: for <code>w = torch.tensor([2.0, 3.0], requires_grad=True)</code> and <code>loss = (w * w).sum()</code>, what should <code>w.grad</code> be?</p>
<details class="answer"><summary>show answer</summary>
<p><code>loss = w[0]² + w[1]² = 4 + 9 = 13</code>. Derivatives: <code>∂loss/∂w[0] = 2w[0] = 4</code>, <code>∂loss/∂w[1] = 2w[1] = 6</code>. So <code>w.grad</code> should be <code>tensor([4., 6.])</code>. Verify in PyTorch.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> What's wrong with this code? Why does <code>x.grad</code> end up as <code>None</code>?
<pre><code>x = torch.tensor([<span class="num">1.0</span>, <span class="num">2.0</span>], requires_grad=<span class="kw">True</span>)
x = x.to(<span class="str">'cuda'</span>)
loss = (x ** <span class="num">2</span>).sum()
loss.backward()
<span class="fn">print</span>(x.grad)        <span class="com"># None</span></code></pre></p>
<details class="answer"><summary>show answer</summary>
<p>The <code>x = x.to('cuda')</code> reassigned <code>x</code> to a non-leaf tensor (the result of a copy op). The <em>original</em> leaf is now garbage-collected without ever having received a grad, and the new <code>x</code> is a non-leaf that doesn't get <code>.grad</code> populated by default. The fix: <code>x = torch.tensor([1.0, 2.0], device='cuda', requires_grad=True)</code>.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> If you call <code>loss.backward()</code> twice in a row without re-running the forward pass, what error do you get and why?</p>
<details class="answer"><summary>show answer</summary>
<p>You get: <em>"Trying to backward through the graph a second time, but the buffers have already been freed."</em> By default, PyTorch frees the saved tensors needed for backward as soon as one backward completes. The graph remains in memory (the grad_fn nodes), but the saved data inside them is gone. To do backward twice, pass <code>retain_graph=True</code> on the first call. This is rare in practice — usually you re-run forward.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li><strong>The forward pass builds a graph.</strong> Every operation produces a tensor and attaches a <code>grad_fn</code> recording how to undo itself.</li>
  <li><strong>Leaves vs non-leaves.</strong> Tensors you create directly (with <code>requires_grad=True</code>) are leaves and own a <code>.grad</code>. Tensors produced by ops are non-leaves and don't get <code>.grad</code> by default — call <code>.retain_grad()</code> if you need it.</li>
  <li><strong>The graph is dynamic</strong> — built fresh on every forward pass. Python control flow works. The graph is freed after backward unless you ask it to stay.</li>
  <li><strong>Backward is the chain rule on the graph.</strong> Start at the output with seed gradient 1, walk in reverse, multiply at each grad_fn, accumulate at every leaf with multiple paths. That's the entire algorithm.</li>
  <li><strong>Each grad_fn implements a vector-Jacobian product</strong>, not a full Jacobian. This is why backward is fast — no giant matrix is ever materialized.</li>
  <li><strong>Gradients accumulate in <code>.grad</code></strong>. This is a feature (gradient accumulation, multiple loss terms) but the source of the "forgot zero_grad()" bug. <code>optimizer.zero_grad()</code> at the start of each step.</li>
  <li><strong>The reassignment trap</strong>: <code>x = x.cuda()</code> or <code>x = x.float()</code> creates a non-leaf and orphans the original leaf. Construct tensors on the right device from the start.</li>
  <li><code>requires_grad</code> propagates: if any input to an op requires grad, the output does too. <code>torch.no_grad()</code> disables this for inference.</li>
  <li><code>.detach()</code> cuts a tensor from the graph (sets its grad_fn to None). Use it to stop gradient flow deliberately.</li>
  <li>The reflex: when something in autograd surprises you, draw the graph by hand. Find the leaves. Trace where the gradient flows. The answer is in the picture.</li>
</ul>
</div>

<p>Module 05 takes the autograd model you just built and turns it into practical reflexes: when to use <code>requires_grad</code> vs <code>detach</code> vs <code>no_grad</code> vs <code>inference_mode</code>, how to debug "gradient is None," what hooks let you peek inside the graph, and why <code>torch.func</code> exists as a parallel API.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">04</span>
  <span>What .backward() actually does</span>
</div>
"""

emit("04_what_backward_does", "Module 04 — What .backward() actually does", BODY)
