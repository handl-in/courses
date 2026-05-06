#!/usr/bin/env python3
"""Module 05: Autograd in practice — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part II · Module 05</div>
  <h1 class="module-title">Autograd <em>in practice</em></h1>
  <p class="module-sub">— stopping gradients on purpose, peeking at them with hooks, and the <code>torch.func</code> API that finally lets you treat gradients as values</p>
</div>

<p>Module 04 gave you the model. Now you need the tool belt. Real autograd code is a constant negotiation: <em>track this</em>, <em>stop tracking that</em>, <em>peek at the gradient before the optimizer eats it</em>, <em>compute a Hessian without writing one by hand</em>. Each of those needs a different switch, and each of those switches has subtle failure modes.</p>

<p>This module is organized around two questions that come up every day: <strong>"how do I stop a gradient from flowing where I don't want it?"</strong> and <strong>"how do I see what's actually happening inside the graph?"</strong> Plus the obligatory section on <em>"why is my gradient None?"</em>, which is the question every PyTorch user has stared at on Stack Overflow at least once.</p>

<div class="keyidea">
There are <strong>four ways to stop a gradient</strong>, and they're not interchangeable: <code>requires_grad=False</code> is a property on a leaf tensor, <code>detach()</code> creates a new tensor cut from the graph, <code>no_grad()</code> is a context manager that disables tracking entirely, and <code>inference_mode()</code> is the faster cousin of <code>no_grad</code> for production. Picking the wrong one usually still gives you a working program — but it costs memory, speed, or both.
</div>

<h2>Welcome back to the cast (with one new face)</h2>

<p>You already know the autograd characters from M4. One new one for today.</p>

<div class="character" style="--c: #b85a6c;">
  <div class="avatar" style="background: #b85a6c; color: #fff;">👁</div>
  <div>
    <p class="who">Hook</p>
    <p class="name">"I'm the spy you place on a tensor or module to peek at what flows through."</p>
    <p class="says">Register me, and I get called every time a gradient passes through that tensor — or every time a module's forward runs. I can <em>look</em> at the gradient. I can <em>modify</em> the gradient if I'm feeling cheeky. I'm how you debug "is the gradient even reaching layer 7?" without printing in 200 places. Just remember to remove me when you're done — I leak otherwise.</p>
  </div>
</div>

<h2>Four ways to stop a gradient</h2>

<p>Let's compare them side by side first, then take each apart.</p>

<div class="table-wrap">
<table>
<caption>The four "stop the gradient" mechanisms</caption>
<thead><tr><th>Mechanism</th><th>What it does</th><th>When to use</th></tr></thead>
<tbody>
<tr><td><code>x.requires_grad = False</code></td><td>Marks a <em>leaf</em> as not needing gradients. Permanent (until you flip it back).</td><td>Freezing model parameters: <code>for p in encoder.parameters(): p.requires_grad = False</code>.</td></tr>
<tr><td><code>x.detach()</code></td><td>Returns a <em>new</em> tensor sharing storage with x but with no <code>grad_fn</code>. Backward stops at it.</td><td>Cutting one tensor out of the graph mid-computation while letting the rest of the graph keep flowing. Stop-gradient in some loss formulations.</td></tr>
<tr><td><code>with torch.no_grad():</code></td><td>Context manager. Inside it, no operations record to the graph. Outputs are non-leaf tensors with <code>requires_grad=False</code>.</td><td>Inference. Computing metrics. Updating parameters in place outside autograd.</td></tr>
<tr><td><code>with torch.inference_mode():</code></td><td>Stricter version of <code>no_grad</code>. Outputs cannot ever be tracked again. Skips even more bookkeeping. Faster.</td><td>Pure inference loops in production where you'll never need backward on the result.</td></tr>
</tbody>
</table>
</div>

<h3><code>requires_grad = False</code> — freezing parameters</h3>

<pre><code><span class="com"># Freeze a pre-trained encoder, train only the classifier head</span>
<span class="kw">for</span> p <span class="kw">in</span> encoder.parameters():
    p.requires_grad = <span class="kw">False</span>

<span class="com"># Now the optimizer should ONLY get the trainable params</span>
optimizer = torch.optim.AdamW(
    [p <span class="kw">for</span> p <span class="kw">in</span> model.parameters() <span class="kw">if</span> p.requires_grad],
    lr=<span class="num">1e-4</span>,
)</code></pre>

<p>This is the most common use. Note the optimizer line — if you give it parameters that don't require grad, you'll either waste memory on dead optimizer state or get an error from the optimizer's sanity check. Filter with the comprehension.</p>

<div class="warn">
<strong>BatchNorm and Dropout don't care about <code>requires_grad</code>.</strong> Freezing parameters does NOT put a layer in eval mode. BN still updates its running statistics during forward, Dropout still drops. To actually freeze a layer's <em>behavior</em>, you also need <code>module.eval()</code>. We'll dig into this in Module 7.
</div>

<h3><code>detach()</code> — cutting one tensor out</h3>

<p>Use this when you have an expression where you want gradients to flow through <em>some</em> inputs but not others. Classic example: target networks in Q-learning, where the target's parameters shouldn't receive gradients from the loss.</p>

<pre><code><span class="com"># Forward pass through both networks</span>
q_values = q_net(state)               <span class="com"># trainable</span>
next_q = target_net(next_state)       <span class="com"># also trainable, but we don't want grads</span>

<span class="com"># Bellman target: r + γ * max_a' Q_target(s', a')</span>
<span class="com"># We want gradients to flow into q_net via q_values, but NOT via next_q</span>
target = reward + gamma * next_q.max(dim=-<span class="num">1</span>).values.detach()
loss = (q_values.gather(<span class="num">1</span>, action[:, <span class="kw">None</span>]) - target[:, <span class="kw">None</span>]).<span class="fn">pow</span>(<span class="num">2</span>).mean()
loss.backward()                       <span class="com"># gradients flow ONLY into q_net</span></code></pre>

<p>Without the <code>.detach()</code>, gradients would leak into <code>target_net</code>, which we explicitly don't want — the whole point of having a target net is to keep it stable.</p>

<div class="character" style="--c: #c1502e;">
  <div class="avatar" style="background: #c1502e; color: #fff;">⌬</div>
  <div>
    <p class="who">grad_fn</p>
    <p class="name">"<code>detach()</code> just sets me to None."</p>
    <p class="says">A detached tensor still shares storage with the original — same bytes, same data. The only difference is that I'm gone. So during backward, when the walk reaches the detached tensor, it stops. The original tensor (still in the graph elsewhere) is unaffected.</p>
  </div>
</div>

<h3><code>no_grad()</code> — disabling tracking entirely</h3>

<p>The most heavyweight option, and the right one for inference and metric computation. Inside the <code>with</code> block, <em>no</em> operation builds graph nodes:</p>

<pre><code>model.<span class="fn">eval</span>()                     <span class="com"># also turn off Dropout, BN train mode</span>
<span class="kw">with</span> torch.no_grad():
    <span class="kw">for</span> batch <span class="kw">in</span> val_loader:
        out = model(batch)
        loss = criterion(out, target)
        <span class="com"># No graph built. out and loss have requires_grad=False.</span>
        <span class="com"># Memory used = forward activations only, no saved-for-backward buffers.</span></code></pre>

<p>Why does this matter? Because building the graph isn't free. Every op allocates a grad_fn object and saves the inputs needed for backward. On a transformer forward pass, those saved buffers can be 5-10× the size of the output tensors. <code>no_grad</code> skips all of that.</p>

<h3><code>inference_mode()</code> — the faster <code>no_grad</code></h3>

<p>Introduced in PyTorch 1.9 as a stricter, faster version. The difference is that tensors created inside <code>inference_mode</code> can <em>never</em> participate in autograd later, even if you wrap them in <code>requires_grad=True</code> or pass them through a graph-building op. PyTorch can therefore skip even more bookkeeping (specifically, the version counter we met in Module 1).</p>

<pre><code><span class="kw">with</span> torch.inference_mode():
    out = model(batch)
    <span class="com"># out.is_leaf is True, requires_grad is False, and out is "stuck"</span>
    <span class="com"># in inference mode — you can't use it as input to a backward graph.</span>

<span class="com"># Outside the block:</span>
y = out.requires_grad_(<span class="kw">True</span>)        <span class="com"># RuntimeError! inference tensors can't be tracked.</span></code></pre>

<p>For pure inference servers, <code>inference_mode</code> can give a few percent speedup over <code>no_grad</code>. For research code where you might want to do something with the output later, stick with <code>no_grad</code>.</p>

<div class="ndq">
<h4>Picking the right "stop"</h4>

<p class="q">When should I prefer <code>detach()</code> over <code>no_grad()</code>?</p>
<p class="a">When you only want to stop the gradient at <em>one specific tensor</em> while letting other parts of the same expression still build the graph. <code>no_grad</code> disables tracking for everything inside the block; <code>detach</code> is surgical.</p>

<p class="q">Is there a performance difference between <code>requires_grad=False</code> and detaching during forward?</p>
<p class="a">Yes. Setting <code>requires_grad=False</code> on a parameter means PyTorch never tracks operations involving it (assuming no other input requires grad). Detaching mid-computation means the graph was built but stops at the detach point. The first is faster if you're freezing for a long time; the second is fine for one-off cuts.</p>

<p class="q">What's the difference between <code>model.eval()</code> and <code>torch.no_grad()</code>?</p>
<p class="a">They solve completely different problems. <code>model.eval()</code> changes module behavior — Dropout off, BN uses running stats. <code>no_grad()</code> changes autograd behavior — no graph built. For inference you almost always want both.</p>

<p class="q">Can I nest these?</p>
<p class="a">Yes. <code>no_grad</code> nested inside <code>no_grad</code> is harmless. <code>enable_grad</code> exists as the inverse — useful inside a function that's called from many places, some of which are in <code>no_grad</code> mode.</p>
</div>

<h2>Why is my gradient None?</h2>

<p>This is the single most-Googled PyTorch question. Let's build a diagnostic flowchart you can run through in your head.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 460" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrFlow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
  </defs>

  <text x="370" y="24" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Why is my gradient None?</text>

  <!-- Q1 -->
  <g transform="translate(260, 50)">
    <rect x="0" y="0" width="220" height="56" fill="#fff8a8" stroke="#1a1612" stroke-width="2"/>
    <text x="110" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">Does the tensor have</text>
    <text x="110" y="42" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">requires_grad = True?</text>
  </g>

  <!-- No → leaf -->
  <text x="160" y="135" font-size="12" font-weight="700" fill="#c1502e">NO</text>
  <path d="M 270 95 L 165 130" stroke="#c1502e" stroke-width="2" fill="none" marker-end="url(#arrFlow)"/>
  <g transform="translate(40, 145)">
    <rect x="0" y="0" width="240" height="50" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="120" y="22" text-anchor="middle" font-size="12" fill="#1a1612">Set it. Or check that you didn't</text>
    <text x="120" y="40" text-anchor="middle" font-size="12" fill="#1a1612">reassign x = x.cuda() and orphan the leaf.</text>
  </g>

  <!-- Yes → Q2 -->
  <text x="540" y="135" font-size="12" font-weight="700" fill="#1f5f5b">YES</text>
  <path d="M 470 95 L 575 130" stroke="#1f5f5b" stroke-width="2" fill="none" marker-end="url(#arrFlow)"/>
  <g transform="translate(440, 145)">
    <rect x="0" y="0" width="260" height="56" fill="#fff8a8" stroke="#1a1612" stroke-width="2"/>
    <text x="130" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">Is it a leaf? Check x.is_leaf.</text>
    <text x="130" y="42" text-anchor="middle" font-size="12" fill="#1a1612">Non-leaves don't get .grad by default.</text>
  </g>

  <!-- Yes → Q3 -->
  <path d="M 570 205 L 570 235" stroke="#1f5f5b" stroke-width="2" fill="none" marker-end="url(#arrFlow)"/>
  <text x="585" y="225" font-size="12" font-weight="700" fill="#1f5f5b">YES (it's a leaf)</text>

  <!-- Q3 -->
  <g transform="translate(440, 245)">
    <rect x="0" y="0" width="260" height="56" fill="#fff8a8" stroke="#1a1612" stroke-width="2"/>
    <text x="130" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">Did you actually call</text>
    <text x="130" y="42" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">.backward() since the last zero?</text>
  </g>

  <!-- Q4 -->
  <path d="M 570 305 L 570 335" stroke="#1f5f5b" stroke-width="2" fill="none" marker-end="url(#arrFlow)"/>
  <text x="585" y="325" font-size="12" font-weight="700" fill="#1f5f5b">YES</text>

  <g transform="translate(440, 345)">
    <rect x="0" y="0" width="260" height="56" fill="#fff8a8" stroke="#1a1612" stroke-width="2"/>
    <text x="130" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">Did the loss actually depend</text>
    <text x="130" y="42" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">on this tensor at all?</text>
  </g>

  <!-- Final answer -->
  <path d="M 440 372 L 360 405" stroke="#c1502e" stroke-width="2" fill="none" marker-end="url(#arrFlow)"/>
  <text x="370" y="390" font-size="12" font-weight="700" fill="#c1502e">NO</text>
  <g transform="translate(40, 405)">
    <rect x="0" y="0" width="320" height="48" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="160" y="20" text-anchor="middle" font-size="12" fill="#1a1612">Backward never reached this leaf →</text>
    <text x="160" y="38" text-anchor="middle" font-size="12" fill="#1a1612">.grad stays None. (Common with frozen branches.)</text>
  </g>

  <!-- non-leaf branch -->
  <path d="M 440 175 L 320 220" stroke="#c1502e" stroke-width="2" fill="none" marker-end="url(#arrFlow)"/>
  <text x="370" y="200" font-size="12" font-weight="700" fill="#c1502e">NO (non-leaf)</text>
  <g transform="translate(40, 220)">
    <rect x="0" y="0" width="280" height="50" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="140" y="22" text-anchor="middle" font-size="12" fill="#1a1612">Add x.retain_grad() before backward,</text>
    <text x="140" y="40" text-anchor="middle" font-size="12" fill="#1a1612">or reach for hooks (next section).</text>
  </g>
</svg>
</div>

<p>Walk through it on every "grad is None" case you ever see. The four checks cover ~95% of incidents:</p>

<ol>
  <li>Does it have <code>requires_grad=True</code>? If not, fix that — and check you didn't reassign through an op like <code>.cuda()</code> or <code>.float()</code> (the M4 trap).</li>
  <li>Is it a leaf? If not, you wanted <code>retain_grad()</code> or hooks.</li>
  <li>Did you call backward at all? Did <code>zero_grad</code> wipe it out before you could read it?</li>
  <li>Did the loss actually depend on this tensor? If you have a layer with effectively-zero contribution to the loss, backward never sends a gradient there. <code>.grad</code> stays <code>None</code>. (DDP catches this with <code>find_unused_parameters=True</code> — Module 16.)</li>
</ol>

<h2>Hooks: peeking inside the graph</h2>

<p>A hook is a function that runs at a specific point in the forward or backward pass. They're how you peek at intermediate gradients, modify them, or just print them for debugging.</p>

<p>Three flavors you'll meet:</p>

<div class="table-wrap">
<table>
<caption>The three hook types</caption>
<thead><tr><th>Hook</th><th>Attach to</th><th>When it fires</th><th>Can it modify?</th></tr></thead>
<tbody>
<tr><td><code>tensor.register_hook(fn)</code></td><td>A tensor</td><td>When the gradient flows back through it</td><td>Yes — return a new tensor to replace the gradient</td></tr>
<tr><td><code>module.register_forward_hook(fn)</code></td><td>An <code>nn.Module</code></td><td>After the module's forward returns</td><td>Yes — return a new output</td></tr>
<tr><td><code>module.register_full_backward_hook(fn)</code></td><td>An <code>nn.Module</code></td><td>When the backward through the module finishes</td><td>Yes — return new gradients w.r.t. inputs</td></tr>
</tbody>
</table>
</div>

<h3>A tensor hook for debugging</h3>

<pre><code>x = torch.tensor([<span class="num">2.0</span>, <span class="num">3.0</span>], requires_grad=<span class="kw">True</span>)
y = (x ** <span class="num">2</span>).sum()

<span class="com"># Print the gradient flowing back into x</span>
handle = x.register_hook(<span class="kw">lambda</span> grad: <span class="fn">print</span>(<span class="str">"grad on x:"</span>, grad))

y.backward()
<span class="com"># prints: grad on x: tensor([4., 6.])</span>

handle.remove()    <span class="com"># clean up — hooks leak otherwise!</span></code></pre>

<p>Crucially, the hook fires <em>during</em> backward, not after. So if you're trying to debug why gradients are zero somewhere, register a hook before calling backward, and you'll see the gradient at the moment it passes through.</p>

<h3>A module forward hook for activation inspection</h3>

<pre><code>activations = {}

<span class="kw">def</span> <span class="fn">save_activation</span>(name):
    <span class="kw">def</span> <span class="fn">hook</span>(module, inputs, output):
        activations[name] = output.detach()
    <span class="kw">return</span> hook

handle = model.layer3.register_forward_hook(save_activation(<span class="str">'layer3'</span>))

out = model(batch)
<span class="fn">print</span>(activations[<span class="str">'layer3'</span>].shape)

handle.remove()</code></pre>

<p>This pattern is everywhere in interpretability research — extracting activations from a frozen model without modifying its code.</p>

<h3>Modifying gradients with a hook</h3>

<p>You can return a new tensor from a hook to replace the gradient. This is how gradient clipping per layer used to be implemented before the formal API:</p>

<pre><code><span class="com"># Clip the gradient flowing into x to be in [-1, 1]</span>
x.register_hook(<span class="kw">lambda</span> grad: grad.<span class="fn">clamp</span>(-<span class="num">1</span>, <span class="num">1</span>))</code></pre>

<p>Or to <em>reverse</em> a gradient (the gradient reversal layer from domain-adaptation papers):</p>

<pre><code><span class="com"># Reverse the gradient — useful in adversarial training</span>
x.register_hook(<span class="kw">lambda</span> grad: -grad * lambda_)</code></pre>

<p>Hooks are powerful and a common source of subtle bugs (because they run silently and order matters). Use them deliberately, and remove them with the returned handle when done.</p>

<div class="warn">
<strong>Hooks leak if you forget to remove them.</strong> Every hook you register holds a reference to the closure, which can hold references to model parameters or worse. In a long-running process (training notebook, server) this adds up. The pattern is: <code>handle = x.register_hook(...)</code>, do work, <code>handle.remove()</code>. Or use the context-manager pattern with a small helper class.
</div>

<h2><code>retain_graph</code> and double backward</h2>

<p>You'll occasionally need to call backward more than once on the same forward graph, or compute gradients of gradients. Two switches make this work.</p>

<p><strong><code>retain_graph=True</code></strong> tells PyTorch not to free the saved-for-backward buffers after this backward call, so you can call backward again on the same graph. Use case: you have multiple losses you compute backward for separately.</p>

<pre><code>loss1.backward(retain_graph=<span class="kw">True</span>)    <span class="com"># keeps the graph around</span>
loss2.backward()                       <span class="com"># now this works; graph is freed after</span></code></pre>

<p>Note: <em>just calling backward twice in a row on the same loss</em> is almost always a bug. The case for <code>retain_graph</code> is when you have separate losses that share a forward computation.</p>

<p><strong><code>create_graph=True</code></strong> tells PyTorch to build a graph <em>over the backward pass itself</em>, so the resulting gradients are differentiable and you can backward through them. Use case: anywhere you need second-order gradients (Hessian-vector products, MAML, gradient penalties in WGAN-GP).</p>

<pre><code><span class="com"># Compute a Hessian-vector product</span>
x = torch.tensor(<span class="num">2.0</span>, requires_grad=<span class="kw">True</span>)
y = x ** <span class="num">3</span>                            <span class="com"># y = 8</span>

grad_x = torch.autograd.grad(y, x, create_graph=<span class="kw">True</span>)[<span class="num">0</span>]
<span class="com"># grad_x = 3x² = 12, AND grad_x has requires_grad=True (because of create_graph)</span>

grad2_x = torch.autograd.grad(grad_x, x)[<span class="num">0</span>]
<span class="com"># grad2_x = 6x = 12 — this is d²y/dx²</span></code></pre>

<p>Double backward is memory-intensive. The graph for the gradient computation is roughly the same size as the original forward graph. For most uses, <code>torch.func.hessian</code> or <code>torch.func.jacrev(jacrev(f))</code> is cleaner — see below.</p>

<h2><code>torch.func</code>: gradients as values</h2>

<p>The traditional autograd API is <em>imperative</em>: you build a graph, you call <code>.backward()</code>, you read <code>.grad</code>. It works, but it's awkward when you want to <em>compose</em> gradients — take the gradient of a gradient, vmap a gradient computation across a batch of inputs, compute a Jacobian without writing a loop.</p>

<p><code>torch.func</code> (formerly <code>functorch</code>) is the functional API. Gradients are <em>values returned by functions</em>, not side effects on tensors. This composes beautifully.</p>

<div class="character autograd">
  <div class="avatar">∂</div>
  <div>
    <p class="who">Autograd</p>
    <p class="name">"<code>torch.func</code> is JAX-style for PyTorch."</p>
    <p class="says">If you've used JAX, this will feel familiar. <code>grad(f)</code> takes a function and returns a function that returns the gradient. <code>vmap(f)</code> takes a function and returns a batched version. <code>jacrev(f)</code> returns the Jacobian. They compose: <code>grad(grad(f))</code> is the second derivative; <code>vmap(grad(f))</code> is per-sample gradients across a batch. The same operations you can do with the imperative API, but composable.</p>
  </div>
</div>

<pre><code><span class="kw">import</span> torch
<span class="kw">from</span> torch.func <span class="kw">import</span> grad, vmap, jacrev

<span class="kw">def</span> <span class="fn">f</span>(x):
    <span class="kw">return</span> (x ** <span class="num">3</span>).sum()

<span class="com"># grad: returns a function that returns the gradient</span>
df = grad(f)
<span class="fn">print</span>(df(torch.tensor([<span class="num">2.0</span>, <span class="num">3.0</span>])))   <span class="com"># tensor([12., 27.]) — that's 3x²</span>

<span class="com"># Second derivative — just compose grad twice</span>
d2f = grad(grad(<span class="kw">lambda</span> x: f(x)))         <span class="com"># f as a scalar fn for grad-grad</span>

<span class="com"># vmap: batched version of any function</span>
batch = torch.randn(<span class="num">8</span>, <span class="num">3</span>)               <span class="com"># 8 examples of shape (3,)</span>
per_sample_grads = vmap(df)(batch)        <span class="com"># shape (8, 3) — gradient for each example</span>

<span class="com"># jacrev: full Jacobian by reverse-mode</span>
<span class="kw">def</span> <span class="fn">g</span>(x):
    <span class="kw">return</span> torch.<span class="fn">stack</span>([(x ** <span class="num">2</span>).sum(), (x ** <span class="num">3</span>).sum()])
J = jacrev(g)(torch.tensor([<span class="num">1.0</span>, <span class="num">2.0</span>, <span class="num">3.0</span>]))
<span class="com"># J shape: (2, 3) — Jacobian of g at x</span></code></pre>

<p>Three things to internalize about <code>torch.func</code>:</p>

<ol>
  <li><strong>It's pure-functional.</strong> No <code>.backward()</code>, no <code>.grad</code>. Gradients come back as return values. This makes them composable.</li>
  <li><strong><code>vmap</code> is genuinely useful, even without grad.</strong> Want per-sample anything? <code>vmap</code> the per-sample function. No more "I have to re-implement this without batching for the inner step."</li>
  <li><strong>It interoperates with the rest of PyTorch.</strong> The functions you pass can use <code>nn.Module</code>s; you call <code>functional_call</code> to bind parameters as arguments. Module 5 doesn't go deep on this — it's a topic worth its own walkthrough — but be aware it's the modern way to compute things like per-sample gradients (used in differentially-private training).</li>
</ol>

<h3>Per-sample gradients in 4 lines</h3>

<p>One of the most useful applications. In standard training, the optimizer sees the <em>average</em> gradient over a batch. Sometimes you want the per-sample gradient — for differential privacy, influence-function analysis, or just curiosity.</p>

<pre><code><span class="kw">from</span> torch.func <span class="kw">import</span> functional_call, grad, vmap

<span class="kw">def</span> <span class="fn">loss_fn</span>(params, sample, target):
    pred = functional_call(model, params, sample.<span class="fn">unsqueeze</span>(<span class="num">0</span>))
    <span class="kw">return</span> F.<span class="fn">cross_entropy</span>(pred, target.<span class="fn">unsqueeze</span>(<span class="num">0</span>))

per_sample_grad_fn = vmap(grad(loss_fn), in_dims=(<span class="kw">None</span>, <span class="num">0</span>, <span class="num">0</span>))
per_sample_grads = per_sample_grad_fn(<span class="fn">dict</span>(model.<span class="fn">named_parameters</span>()), batch_x, batch_y)
<span class="com"># per_sample_grads is a dict where each value has shape (batch_size, *param_shape)</span></code></pre>

<p>Before <code>torch.func</code>, this required either a <code>for</code> loop over the batch (slow) or a clever rewrite of every layer (painful). Now it's four lines.</p>

<h2>Code Magnets: build a gradient-saving training step</h2>

<p>You're writing a training step that, in addition to the normal forward / backward / step / zero, <em>saves a copy of the gradient on the first layer's weight</em> for later inspection. Use a hook.</p>

<div class="magnets">
<p>Arrange the magnets into a working step. There are two red herrings.</p>

<div class="magnet-pool">
  <span class="magnet">saved_grads = []</span>
  <span class="magnet">handle = model.layer1.weight.register_hook(lambda g: saved_grads.append(g.clone()))</span>
  <span class="magnet">handle = model.layer1.register_forward_hook(lambda m, i, o: saved_grads.append(o))</span>
  <span class="magnet">optimizer.zero_grad()</span>
  <span class="magnet">loss = criterion(model(x), y)</span>
  <span class="magnet">loss.backward()</span>
  <span class="magnet">optimizer.step()</span>
  <span class="magnet">handle.remove()</span>
  <span class="magnet">model.layer1.weight.requires_grad = False</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code>saved_grads = []
handle = model.layer1.weight.register_hook(<span class="kw">lambda</span> g: saved_grads.<span class="fn">append</span>(g.<span class="fn">clone</span>()))

optimizer.<span class="fn">zero_grad</span>()
loss = <span class="fn">criterion</span>(<span class="fn">model</span>(x), y)
loss.<span class="fn">backward</span>()
optimizer.<span class="fn">step</span>()

handle.<span class="fn">remove</span>()</code></pre>
<p>The two traps:</p>
<ul>
  <li><code>register_forward_hook</code> records the layer's <em>output</em>, not the gradient. Wrong tool.</li>
  <li><code>requires_grad = False</code> would stop the gradient from being computed at all — defeating the purpose.</li>
</ul>
<p>The <code>g.clone()</code> inside the hook matters: <code>g</code> is the gradient tensor flowing through; if you don't clone it, you're saving a reference that could be mutated or freed by the time you look at it. Always clone gradients you want to keep.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each tool to the problem it solves.</p>

<div class="match-grid">
  <div class="header">Tool</div>
  <div class="header">Problem</div>

  <div>x.detach()</div>
  <div>A. Need to backward more than once on the same forward graph (separate losses).</div>

  <div>with torch.no_grad():</div>
  <div>B. Want second-order gradients (gradient of a gradient).</div>

  <div>with torch.inference_mode():</div>
  <div>C. Want to peek at the gradient flowing into a tensor without modifying it.</div>

  <div>retain_graph=True</div>
  <div>D. Surgical: cut just one tensor out of the graph, let others keep flowing.</div>

  <div>create_graph=True</div>
  <div>E. Production inference loop, never need to backprop on the result.</div>

  <div>tensor.register_hook(fn)</div>
  <div>F. Disable graph building entirely for a block of code (e.g., validation).</div>

  <div>vmap(grad(f))</div>
  <div>G. Per-sample gradients across a batch in one call.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>x.detach()</strong> → D<br>
<strong>with torch.no_grad():</strong> → F<br>
<strong>with torch.inference_mode():</strong> → E<br>
<strong>retain_graph=True</strong> → A<br>
<strong>create_graph=True</strong> → B<br>
<strong>tensor.register_hook(fn)</strong> → C<br>
<strong>vmap(grad(f))</strong> → G
</p>
<p>The mental shortcut: <em>detach is surgical, no_grad is a block, inference_mode is the strict no_grad, retain_graph keeps memory, create_graph builds graph over backward, hooks observe, vmap+grad batches per-sample</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> You have a model with two heads — <code>head_a</code> and <code>head_b</code> — sharing an encoder. You want to train both heads, but you only want the encoder to be updated by <code>head_a</code>'s loss, not <code>head_b</code>'s. Write the loss computation.</p>
<details class="answer"><summary>show answer</summary>
<pre><code>features = encoder(x)

<span class="com"># head_a sees features through the graph as normal</span>
loss_a = <span class="fn">criterion</span>(head_a(features), y_a)

<span class="com"># head_b sees a detached version, so its loss can't backprop into encoder</span>
loss_b = <span class="fn">criterion</span>(head_b(features.<span class="fn">detach</span>()), y_b)

(loss_a + loss_b).<span class="fn">backward</span>()</code></pre>
<p>The encoder's gradient comes only from <code>loss_a</code> because <code>loss_b</code>'s path through <code>features.detach()</code> stops at the detach. Both heads still receive their own gradients normally because <code>head_a</code> and <code>head_b</code>'s parameters are leaves.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Why might <code>torch.no_grad()</code> not be enough for a production inference server, and when does <code>inference_mode()</code> actually win?</p>
<details class="answer"><summary>show answer</summary>
<p><code>no_grad</code> still maintains version counters on tensors (the in-place-detection mechanism we met in Module 1) and other small bits of bookkeeping. <code>inference_mode</code> skips that too. The win is usually a few percent in throughput, but it can matter at scale. The catch: tensors created in <code>inference_mode</code> can never be tracked by autograd later. If your server occasionally needs to compute a gradient (e.g., for adversarial robustness checks), <code>inference_mode</code> will break that path.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> Compute the second derivative of <code>sin(x)</code> at <code>x = π/4</code> using <code>torch.func</code>.</p>
<details class="answer"><summary>show answer</summary>
<pre><code><span class="kw">from</span> torch.func <span class="kw">import</span> grad
<span class="kw">import</span> math

f = <span class="kw">lambda</span> x: torch.<span class="fn">sin</span>(x)
d2f = grad(grad(f))

x = torch.<span class="fn">tensor</span>(math.pi / <span class="num">4</span>)
<span class="fn">print</span>(<span class="fn">d2f</span>(x))    <span class="com"># tensor(-0.7071) — that's -sin(π/4) ✓</span></code></pre>
<p>Compare to the imperative API: you'd need <code>create_graph=True</code> on the first <code>autograd.grad</code> call, then a second <code>autograd.grad</code> call with the right inputs. Three lines vs five, but the <code>torch.func</code> version reads as math.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> Write a hook that prints whenever a NaN appears in the gradient of any parameter. Mock-test it on a tiny model.</p>
<details class="answer"><summary>show answer</summary>
<pre><code><span class="kw">def</span> <span class="fn">nan_warning_hook</span>(name):
    <span class="kw">def</span> <span class="fn">hook</span>(grad):
        <span class="kw">if</span> torch.<span class="fn">isnan</span>(grad).<span class="fn">any</span>():
            <span class="fn">print</span>(<span class="str">f"NaN in grad of {name}!"</span>)
    <span class="kw">return</span> hook

handles = []
<span class="kw">for</span> name, p <span class="kw">in</span> model.<span class="fn">named_parameters</span>():
    <span class="kw">if</span> p.requires_grad:
        handles.<span class="fn">append</span>(p.<span class="fn">register_hook</span>(<span class="fn">nan_warning_hook</span>(name)))

<span class="com"># ...train as normal; if any param's gradient gets NaN, you'll know which one ...</span>

<span class="kw">for</span> h <span class="kw">in</span> handles:
    h.<span class="fn">remove</span>()</code></pre>
<p>This is a real diagnostic pattern, especially in mixed-precision training. The hook fires as the gradient is computed, so you spot the NaN at its source instead of much later when training has gone off the rails.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li><strong>Four ways to stop a gradient</strong>, none interchangeable: <code>requires_grad=False</code> (per-tensor flag, freezing), <code>detach()</code> (surgical cut, returns a new tensor), <code>no_grad()</code> (block-level, no graph built), <code>inference_mode()</code> (stricter and faster, for production).</li>
  <li><code>model.eval()</code> and <code>no_grad()</code> solve <strong>different</strong> problems. Eval changes layer behavior (Dropout off, BN uses running stats); no_grad disables autograd. Inference wants both.</li>
  <li>The <strong>"why is my grad None?"</strong> diagnostic: check requires_grad, check is_leaf, check that backward ran, check that the loss actually depends on the tensor.</li>
  <li><strong>Hooks</strong> are spies. <code>tensor.register_hook</code> for gradients, <code>module.register_forward_hook</code> for activations, <code>module.register_full_backward_hook</code> for module-level grads. <strong>Always remove them</strong> with the returned handle.</li>
  <li>Hooks can <em>modify</em> what flows through (clip, reverse, scale). The gradient reversal layer is one line.</li>
  <li><code>retain_graph=True</code> = "don't free the saved buffers, I'll backward again." Use only when you have separate losses.</li>
  <li><code>create_graph=True</code> = "build a graph over the backward pass." Enables double backward (Hessian-vector products, MAML, gradient penalties).</li>
  <li><strong><code>torch.func</code></strong> is the functional API. <code>grad(f)</code>, <code>vmap(f)</code>, <code>jacrev(f)</code>, <code>jacfwd(f)</code>, <code>hessian(f)</code> compose freely. Per-sample gradients in 4 lines.</li>
  <li>Always <code>g.clone()</code> a gradient you want to keep — references are fragile.</li>
  <li>The reflex: when in doubt, draw the graph from M4. Then ask which switch turns off which edge.</li>
</ul>
</div>

<p>Module 06 takes the autograd model and lets you write your own. We'll build a custom <code>torch.autograd.Function</code> with explicit forward and backward, verify it with <code>gradcheck</code>, and discuss when it earns its keep — fused ops, memory-saving tricks, straight-through estimators, and the cases where the framework's automatic backward isn't what you want.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">05</span>
  <span>Autograd in practice</span>
</div>
"""

emit("05_autograd_in_practice", "Module 05 — Autograd in practice", BODY)
