#!/usr/bin/env python3
"""Module 07: nn.Module deeply — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part III · Module 07</div>
  <h1 class="module-title"><code>nn.Module</code>, <em>deeply</em></h1>
  <p class="module-sub">— parameters vs buffers, the registration gotcha that silently orphans your weights, and what really happens when you call <code>model.cuda()</code></p>
</div>

<p>You've used <code>nn.Module</code> a hundred times. You subclass it, define <code>__init__</code> and <code>forward</code>, call <code>model(x)</code>, and gradients magically appear in <code>model.parameters()</code>. This module is about why "magically" is doing too much work in that sentence.</p>

<p>The actual story is: <code>nn.Module</code> is a <em>container</em> — a tree of named tensors and other Modules. It maintains three internal dictionaries: one for parameters, one for buffers, one for child modules. When you assign a tensor or Module to <code>self.something</code>, PyTorch decides which dictionary it belongs in based on its type. Most "weird module behavior" — vanishing weights, missing keys in state_dict, frozen layers that aren't really frozen — is one of those decisions going wrong.</p>

<div class="keyidea">
An <code>nn.Module</code> is <strong>three dictionaries plus a forward method</strong>: <code>_parameters</code>, <code>_buffers</code>, and <code>_modules</code>. Anything you want PyTorch to find, save, move, or update must live in one of those. Plain Python attributes (lists, dicts, raw tensors) are <em>invisible</em> to <code>.parameters()</code>, <code>.state_dict()</code>, <code>.cuda()</code>, and <code>optimizer</code>. The "registration gotcha" is when you stash something in a place PyTorch can't see.
</div>

<h2>Meet the Module family</h2>

<div class="character" style="--c: #133e3b;">
  <div class="avatar" style="background: #133e3b; color: #fff;">M</div>
  <div>
    <p class="who">Module</p>
    <p class="name">"I'm a tree. Everything you do to me, I do to my children."</p>
    <p class="says">When you call <code>model.cuda()</code>, I recursively visit every parameter, buffer, and child module, moving each to the GPU. Same for <code>.train()</code>, <code>.eval()</code>, <code>.float()</code>, <code>.state_dict()</code>. The <em>tree</em> is the abstraction. But I can only walk children I know about — and I only know about what was registered. If you stuck a Module in a plain Python list, I won't find it.</p>
  </div>
</div>

<div class="character" style="--c: #c1502e;">
  <div class="avatar" style="background: #c1502e; color: #fff;">P</div>
  <div>
    <p class="who">Parameter</p>
    <p class="name">"I'm a tensor that gets gradients and shows up in <code>.parameters()</code>."</p>
    <p class="says">Wrap a tensor in <code>nn.Parameter(...)</code> and assign me to <code>self.weight</code>. Now I'm registered. I appear in <code>model.parameters()</code> (so the optimizer finds me), I appear in <code>state_dict()</code> (so checkpointing finds me), I get moved by <code>.cuda()</code>, and I have <code>requires_grad=True</code> by default. I'm the thing the optimizer updates.</p>
  </div>
</div>

<div class="character" style="--c: #1f5f5b;">
  <div class="avatar" style="background: #1f5f5b; color: #fff;">B</div>
  <div>
    <p class="who">Buffer</p>
    <p class="name">"I'm a tensor that's part of the model state but never gets a gradient."</p>
    <p class="says">Register me with <code>self.register_buffer('running_mean', torch.zeros(D))</code>. I show up in <code>state_dict()</code>, I get moved by <code>.cuda()</code>, but I'm <em>not</em> in <code>.parameters()</code> and the optimizer ignores me. BatchNorm's running statistics are buffers. Causal masks are buffers. RoPE's precomputed cos/sin tables are buffers. Anything the model needs to remember but shouldn't be trained.</p>
  </div>
</div>

<h2>The three dictionaries</h2>

<p>Let's open up <code>nn.Module</code> and stare at the three dictionaries that run the show.</p>

<pre><code><span class="kw">import</span> torch
<span class="kw">import</span> torch.nn <span class="kw">as</span> nn

<span class="kw">class</span> <span class="ty">Tiny</span>(nn.Module):
    <span class="kw">def</span> <span class="fn">__init__</span>(self):
        <span class="fn">super</span>().<span class="fn">__init__</span>()
        self.weight = nn.<span class="fn">Parameter</span>(torch.<span class="fn">randn</span>(<span class="num">3</span>, <span class="num">4</span>))    <span class="com"># goes to _parameters</span>
        self.<span class="fn">register_buffer</span>(<span class="str">'mask'</span>, torch.<span class="fn">ones</span>(<span class="num">3</span>))     <span class="com"># goes to _buffers</span>
        self.head = nn.<span class="fn">Linear</span>(<span class="num">4</span>, <span class="num">10</span>)                  <span class="com"># goes to _modules</span>
        self.scratch = torch.<span class="fn">randn</span>(<span class="num">5</span>)                  <span class="com"># goes to... NOWHERE useful</span>

m = <span class="fn">Tiny</span>()
<span class="fn">print</span>(m._parameters.keys())     <span class="com"># odict_keys(['weight'])</span>
<span class="fn">print</span>(m._buffers.keys())        <span class="com"># odict_keys(['mask'])</span>
<span class="fn">print</span>(m._modules.keys())        <span class="com"># odict_keys(['head'])</span>
<span class="com"># scratch is just an instance attribute, not in any of these</span></code></pre>

<p>Look at that last line. <code>self.scratch</code> is a regular Python attribute. It's a tensor, sure, but PyTorch has no idea it's there. Specifically:</p>

<ul>
  <li><code>m.parameters()</code> won't yield it.</li>
  <li><code>m.state_dict()</code> won't include it.</li>
  <li><code>m.cuda()</code> won't move it (it'll stay on CPU).</li>
  <li><code>m.eval()</code> won't affect it.</li>
</ul>

<p>That's the registration gotcha in miniature. PyTorch's <code>__setattr__</code> hook checks whether you're assigning a <code>Parameter</code>, a <code>Module</code>, or something else, and routes it to the right dictionary. Plain tensors fall through to the default Python attribute set, which the framework never visits.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 360" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrM" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
  </defs>

  <text x="370" y="24" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">What __setattr__ does when you assign self.something = X</text>

  <!-- Top: assignment expression -->
  <g transform="translate(260, 50)">
    <rect x="0" y="0" width="220" height="40" fill="#fff8a8" stroke="#1a1612" stroke-width="2"/>
    <text x="110" y="25" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">self.x = something</text>
  </g>

  <!-- four buckets -->
  <g transform="translate(20, 140)">
    <rect x="0" y="0" width="160" height="60" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="80" y="20" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">_parameters</text>
    <text x="80" y="40" text-anchor="middle" font-size="11" fill="#1a1612">if it's nn.Parameter</text>
    <text x="80" y="80" text-anchor="middle" font-size="11" fill="#c1502e">.parameters() ✓</text>
    <text x="80" y="95" text-anchor="middle" font-size="11" fill="#c1502e">state_dict ✓ &middot; cuda ✓</text>
    <text x="80" y="110" text-anchor="middle" font-size="11" fill="#c1502e">trained by optimizer ✓</text>
  </g>

  <g transform="translate(195, 140)">
    <rect x="0" y="0" width="160" height="60" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="80" y="20" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">_buffers</text>
    <text x="80" y="40" text-anchor="middle" font-size="11" fill="#1a1612">if you call register_buffer</text>
    <text x="80" y="80" text-anchor="middle" font-size="11" fill="#1f5f5b">.parameters() ✗</text>
    <text x="80" y="95" text-anchor="middle" font-size="11" fill="#1f5f5b">state_dict ✓ &middot; cuda ✓</text>
    <text x="80" y="110" text-anchor="middle" font-size="11" fill="#1f5f5b">no gradient</text>
  </g>

  <g transform="translate(370, 140)">
    <rect x="0" y="0" width="160" height="60" fill="#d3e9f5" stroke="#133e3b" stroke-width="2"/>
    <text x="80" y="20" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">_modules</text>
    <text x="80" y="40" text-anchor="middle" font-size="11" fill="#1a1612">if it's nn.Module</text>
    <text x="80" y="80" text-anchor="middle" font-size="11" fill="#133e3b">recursive — its params</text>
    <text x="80" y="95" text-anchor="middle" font-size="11" fill="#133e3b">become my params</text>
    <text x="80" y="110" text-anchor="middle" font-size="11" fill="#133e3b">cuda/state_dict cascade</text>
  </g>

  <g transform="translate(545, 140)">
    <rect x="0" y="0" width="180" height="60" fill="#ede2cc" stroke="#6b5d4f" stroke-width="2" stroke-dasharray="4 3"/>
    <text x="90" y="20" text-anchor="middle" font-size="13" font-weight="700" fill="#6b5d4f">(plain attribute)</text>
    <text x="90" y="40" text-anchor="middle" font-size="11" fill="#1a1612">anything else</text>
    <text x="90" y="80" text-anchor="middle" font-size="11" fill="#6b5d4f">INVISIBLE to PyTorch</text>
    <text x="90" y="95" text-anchor="middle" font-size="11" fill="#6b5d4f">no .parameters(), no</text>
    <text x="90" y="110" text-anchor="middle" font-size="11" fill="#6b5d4f">state_dict, no .cuda()</text>
  </g>

  <!-- arrows -->
  <path d="M 320 92 L 100 138" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrM)"/>
  <path d="M 350 92 L 275 138" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrM)"/>
  <path d="M 390 92 L 450 138" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrM)"/>
  <path d="M 420 92 L 635 138" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrM)"/>

  <!-- annotation -->
  <text x="370" y="320" font-family="'Caveat', cursive" font-size="22" fill="#c1502e" text-anchor="middle">the right-most bucket is where bugs live</text>
</svg>
</div>

<h2>The registration gotcha (the one that bites everyone)</h2>

<p>Here's the bug in its purest form:</p>

<pre><code><span class="kw">class</span> <span class="ty">BrokenStack</span>(nn.Module):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, n_layers, dim):
        <span class="fn">super</span>().<span class="fn">__init__</span>()
        self.layers = [nn.<span class="fn">Linear</span>(dim, dim) <span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(n_layers)]    <span class="com"># Python list!</span>

    <span class="kw">def</span> <span class="fn">forward</span>(self, x):
        <span class="kw">for</span> layer <span class="kw">in</span> self.layers:
            x = <span class="fn">layer</span>(x)
        <span class="kw">return</span> x

m = <span class="fn">BrokenStack</span>(<span class="num">3</span>, <span class="num">4</span>)
<span class="fn">print</span>(<span class="fn">list</span>(m.<span class="fn">parameters</span>()))     <span class="com"># []  ← EMPTY. 0 trainable parameters.</span>
m.<span class="fn">cuda</span>()                       <span class="com"># NO-OP for the layers — they stay on CPU</span></code></pre>

<p>The model <em>runs</em>. Forward works. But the optimizer has nothing to train, the layers never move to GPU, and <code>state_dict()</code> is empty. You can train for hours and the layers don't update because the optimizer literally doesn't know they exist.</p>

<p>The fix: use <code>nn.ModuleList</code> instead of a plain list. Same iteration interface, but PyTorch sees the children.</p>

<pre><code><span class="kw">class</span> <span class="ty">FixedStack</span>(nn.Module):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, n_layers, dim):
        <span class="fn">super</span>().<span class="fn">__init__</span>()
        self.layers = nn.<span class="fn">ModuleList</span>([nn.<span class="fn">Linear</span>(dim, dim) <span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(n_layers)])

    <span class="kw">def</span> <span class="fn">forward</span>(self, x):
        <span class="kw">for</span> layer <span class="kw">in</span> self.layers:
            x = <span class="fn">layer</span>(x)
        <span class="kw">return</span> x

m = <span class="fn">FixedStack</span>(<span class="num">3</span>, <span class="num">4</span>)
<span class="fn">print</span>(<span class="fn">sum</span>(p.<span class="fn">numel</span>() <span class="kw">for</span> p <span class="kw">in</span> m.<span class="fn">parameters</span>()))   <span class="com"># 60 (= 3 * (4*4 + 4))</span></code></pre>

<p>Same gotcha for dictionaries — use <code>nn.ModuleDict</code>. Same for sequential composition — <code>nn.Sequential</code> is also Module-aware.</p>

<div class="warn">
<strong>The silent version.</strong> The bug above is <em>visible</em> because <code>parameters()</code> returns nothing. The really nasty version is when you have <em>some</em> registered modules and a Python list with one extra layer:
<pre><code>self.encoder = nn.<span class="fn">Linear</span>(dim, dim)                  <span class="com"># registered ✓</span>
self.heads = [nn.<span class="fn">Linear</span>(dim, <span class="num">10</span>) <span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(<span class="num">3</span>)]   <span class="com"># NOT registered ✗</span></code></pre>
The encoder trains fine. The heads don't. The model produces output (the heads are still callable from <code>forward</code>), but their weights are random forever. You'll see val accuracy plateau and assume your architecture is bad. It's actually that 3/4 of your model isn't training.
</div>

<h2>Parameters vs Buffers — when to use which</h2>

<p>Both live in the model's state. Both move with <code>.cuda()</code>. Both appear in <code>state_dict()</code>. The difference comes down to two questions: <em>does the optimizer touch it?</em> and <em>does autograd track it?</em></p>

<div class="table-wrap">
<table>
<caption>Parameters vs Buffers, side by side</caption>
<thead><tr><th>Behavior</th><th>Parameter</th><th>Buffer</th></tr></thead>
<tbody>
<tr><td>In <code>.parameters()</code>?</td><td>Yes</td><td>No</td></tr>
<tr><td>In <code>.state_dict()</code>?</td><td>Yes</td><td>Yes (by default)</td></tr>
<tr><td>Moved by <code>.cuda()</code> / <code>.to()</code>?</td><td>Yes</td><td>Yes</td></tr>
<tr><td><code>requires_grad</code> by default?</td><td>True</td><td>False</td></tr>
<tr><td>Updated by optimizer?</td><td>Yes</td><td>No</td></tr>
<tr><td>Updated during forward (e.g., running stats)?</td><td>No</td><td>Often yes</td></tr>
<tr><td>How to register</td><td><code>self.x = nn.Parameter(t)</code></td><td><code>self.register_buffer('x', t)</code></td></tr>
</tbody>
</table>
</div>

<p>Real examples of buffers in the wild:</p>

<ul>
  <li><strong>BatchNorm's running mean and variance.</strong> Updated during training-mode forward, used during eval-mode forward. Never trained.</li>
  <li><strong>Causal masks in attention.</strong> Precomputed once, never updated. Tied to the model so they move with it.</li>
  <li><strong>RoPE / position-encoding tables.</strong> Computed at init from the dim and max sequence length. Constant during training.</li>
  <li><strong>EMA model weights.</strong> Updated outside the optimizer (manually), saved with the checkpoint. Buffers fit perfectly.</li>
</ul>

<div class="dialogue">
<span class="speaker">JUNIOR</span>I added a causal mask as a regular tensor on <code>self</code>. Training works, but when I move the model to GPU the mask stays on CPU and I get a device mismatch. Why?
<span class="speaker b">SENIOR</span>Because plain tensor attributes aren't registered. <code>self.mask = torch.tril(...)</code> is invisible to <code>.cuda()</code>. Use <code>self.register_buffer('mask', torch.tril(...))</code>. Same data, but now PyTorch knows about it.
<span class="speaker">JUNIOR</span>And it'll be in my checkpoints too?
<span class="speaker b">SENIOR</span>Yes — buffers go in <code>state_dict()</code> by default. If you don't want it persisted (because it's recomputed at init), pass <code>persistent=False</code>: <code>self.register_buffer('mask', tensor, persistent=False)</code>. Then it's still tracked for device moves but not saved.
</div>

<h2>The state_dict lifecycle</h2>

<p><code>state_dict()</code> returns an <em>OrderedDict</em> mapping fully-qualified parameter and buffer names to tensors. It's how you save and load models.</p>

<pre><code>model = <span class="fn">FixedStack</span>(<span class="num">2</span>, <span class="num">4</span>)
sd = model.<span class="fn">state_dict</span>()
<span class="fn">print</span>(<span class="fn">list</span>(sd.<span class="fn">keys</span>()))
<span class="com"># ['layers.0.weight', 'layers.0.bias', 'layers.1.weight', 'layers.1.bias']</span>
<span class="fn">print</span>(sd[<span class="str">'layers.0.weight'</span>].shape)        <span class="com"># torch.Size([4, 4])</span></code></pre>

<p>The key naming follows the module tree: <code>layers</code> is the ModuleList, <code>0</code> is the index, <code>weight</code> is the parameter name. Nested modules produce dotted paths like <code>encoder.attn.k_proj.weight</code>.</p>

<p>Saving and loading:</p>

<pre><code><span class="com"># Save</span>
torch.save(model.<span class="fn">state_dict</span>(), <span class="str">'model.pt'</span>)

<span class="com"># Load (the standard pattern)</span>
new_model = <span class="fn">FixedStack</span>(<span class="num">2</span>, <span class="num">4</span>)              <span class="com"># build the same architecture first</span>
sd = torch.<span class="fn">load</span>(<span class="str">'model.pt'</span>, weights_only=<span class="kw">True</span>)
new_model.<span class="fn">load_state_dict</span>(sd)              <span class="com"># in-place restore</span></code></pre>

<p>Why save the <code>state_dict</code> and not the model object directly? Two reasons. (1) The state_dict is just tensors — portable, version-stable, doesn't depend on your code being importable at load time. (2) Pickled model objects break the moment you rename a class or move a file. State dicts survive refactors as long as the parameter names match.</p>

<h3><code>load_state_dict</code> options worth knowing</h3>

<pre><code><span class="com"># Strict mode (default) — every key must match exactly</span>
model.<span class="fn">load_state_dict</span>(sd, strict=<span class="kw">True</span>)

<span class="com"># Non-strict — partial loads, useful for fine-tuning checkpoints</span>
missing, unexpected = model.<span class="fn">load_state_dict</span>(sd, strict=<span class="kw">False</span>)
<span class="fn">print</span>(<span class="str">"missing keys (in model but not sd):"</span>, missing)
<span class="fn">print</span>(<span class="str">"unexpected keys (in sd but not model):"</span>, unexpected)

<span class="com"># Useful when you've added a new head to a pretrained model:</span>
<span class="com">#   missing = ['new_head.weight', 'new_head.bias']  ← OK, will train these from scratch</span>
<span class="com">#   unexpected = ['old_head.weight', ...]           ← OK, you removed the old head</span></code></pre>

<p>The <code>strict=False</code> path with explicit handling of missing/unexpected keys is the right way to do most fine-tuning. Print the lists, eyeball them, make sure the diffs match what you expected.</p>

<div class="ndq">
<h4>About state_dict</h4>

<p class="q">Why does PyTorch warn me about <code>weights_only</code> when I load a checkpoint?</p>
<p class="a">Pickled checkpoints can execute arbitrary code on load (it's a known Python pickle issue). Modern PyTorch defaults to <code>weights_only=True</code> for safety — it'll load tensors but reject anything else. If you trust the source, set <code>weights_only=True</code> explicitly to silence the warning. If you don't trust the source, this is your friend.</p>

<p class="q">Can I rename keys at load time?</p>
<p class="a">Yes — load the dict as a regular Python dict, manipulate the keys, then call <code>load_state_dict</code>. <pre><code>sd = torch.<span class="fn">load</span>(<span class="str">'old.pt'</span>)
sd = {k.<span class="fn">replace</span>(<span class="str">'old_name'</span>, <span class="str">'new_name'</span>): v <span class="kw">for</span> k, v <span class="kw">in</span> sd.<span class="fn">items</span>()}
model.<span class="fn">load_state_dict</span>(sd)</code></pre>
This is how you handle renamed layers between checkpoint versions.</p>

<p class="q">What about the optimizer state?</p>
<p class="a">Save it separately. <code>optimizer.state_dict()</code> contains momentum buffers, Adam moments, step counts. To resume training mid-run, you save and load both: <code>{'model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': ...}</code>. We'll do this properly in Module 11.</p>

<p class="q">My state_dict has weird keys like <code>module.encoder.weight</code> (with a leading "module."). What's that?</p>
<p class="a">You saved a checkpoint while wrapped in <code>DataParallel</code> or <code>DistributedDataParallel</code> (Module 16). The wrapper adds the prefix. To load into a non-wrapped model, strip the prefix: <code>{k.removeprefix('module.'): v ...}</code>. Or wrap before loading.</p>
</div>

<h2><code>train()</code> vs <code>eval()</code> — they don't do what you think</h2>

<p>The world's most misunderstood API. Let's be precise.</p>

<p><code>model.train()</code> sets the <code>training</code> flag to <code>True</code> on every Module in the tree. <code>model.eval()</code> sets it to <code>False</code>. <strong>That's it.</strong> They don't disable autograd, don't freeze parameters, don't move anything. They flip a single Boolean.</p>

<p>Why does it matter, then? Because some layers <em>check</em> that flag and behave differently:</p>

<ul>
  <li><strong>Dropout</strong>: drops in train mode, identity in eval mode.</li>
  <li><strong>BatchNorm</strong>: uses batch statistics + updates running stats in train mode; uses running stats only in eval mode.</li>
  <li><strong>Custom modules</strong> you write can check <code>self.training</code> and behave differently if you want.</li>
</ul>

<p>Layers that don't care about training mode (Linear, Conv, LayerNorm, almost everything else) behave identically in either mode.</p>

<div class="warn">
<strong>Three "eval" things people confuse.</strong>
<ul>
  <li><code>model.eval()</code> — flips the training flag. Affects Dropout/BN behavior. Does NOT disable autograd.</li>
  <li><code>torch.no_grad()</code> — disables autograd. Does NOT touch the training flag. Dropout still drops, BN still updates stats.</li>
  <li><code>for p in model.parameters(): p.requires_grad = False</code> — freezes parameters from the optimizer's view. Does NOT touch the training flag or autograd-as-a-whole.</li>
</ul>
For inference, you want <strong>both</strong> <code>model.eval()</code> AND <code>torch.no_grad()</code>. They solve different problems.
</div>

<h2>Module hooks (briefly — full coverage was M5)</h2>

<p>Module-level hooks fire around the module's <code>forward</code>. They're useful for things you can't easily do inside <code>forward</code>: capturing intermediate activations from a frozen model, modifying outputs of a third-party layer, instrumenting for debugging.</p>

<pre><code><span class="kw">def</span> <span class="fn">capture_output</span>(module, inputs, output):
    captures.<span class="fn">append</span>(output.<span class="fn">detach</span>())

captures = []
handle = model.encoder.layer[<span class="num">3</span>].<span class="fn">register_forward_hook</span>(capture_output)
out = <span class="fn">model</span>(x)
<span class="fn">print</span>(<span class="fn">len</span>(captures))         <span class="com"># 1</span>
<span class="fn">print</span>(captures[<span class="num">0</span>].shape)
handle.<span class="fn">remove</span>()              <span class="com"># always remove</span></code></pre>

<p>The signature: <code>hook(module, inputs, output)</code>. <code>inputs</code> is a tuple even for single-input modules. You can return a new value from a forward hook to <em>replace</em> the output (rare but occasionally needed).</p>

<h2>Weight tying</h2>

<p>Sometimes you want two parameters to <em>be the same tensor</em>. Classic case: in a transformer language model, the input embedding and the output projection often share weights — same matrix, two roles. The savings are huge (the embedding can be 30% of the model).</p>

<p>The naive (and wrong) approach:</p>

<pre><code>self.embed = nn.<span class="fn">Embedding</span>(vocab, dim)
self.out_proj = nn.<span class="fn">Linear</span>(dim, vocab, bias=<span class="kw">False</span>)

<span class="com"># Wrong: copies the data, doesn't tie</span>
self.out_proj.weight.<span class="fn">copy_</span>(self.embed.weight)</code></pre>

<p>The right approach is to make the two attributes refer to the <em>same Parameter</em>:</p>

<pre><code>self.embed = nn.<span class="fn">Embedding</span>(vocab, dim)
self.out_proj = nn.<span class="fn">Linear</span>(dim, vocab, bias=<span class="kw">False</span>)
self.out_proj.weight = self.embed.weight    <span class="com"># now both names point to one tensor</span></code></pre>

<p>Now both names refer to one underlying tensor. Updates from one path affect the other. Memory is halved. The optimizer sees one parameter, not two (PyTorch dedupes by tensor identity in <code>parameters()</code>).</p>

<p>Two things to know:</p>

<ol>
  <li>Do this <em>after</em> both Modules exist. The assignment replaces <code>out_proj</code>'s parameter dict entry.</li>
  <li><code>state_dict()</code> will only have one entry (whichever was first). On load, the same tying must be re-established before <code>load_state_dict</code> — or the loaded weights won't be propagated to both names.</li>
</ol>

<h2>Lazy modules and <code>parametrize</code>: the convenient extras</h2>

<p>Two newer APIs worth knowing about, briefly.</p>

<h3>Lazy modules</h3>

<p><code>nn.LazyLinear</code>, <code>nn.LazyConv2d</code>, etc. defer parameter shape determination until the first forward pass:</p>

<pre><code>layer = nn.<span class="fn">LazyLinear</span>(<span class="num">10</span>)            <span class="com"># in_features unknown</span>
<span class="fn">print</span>(layer.weight)                <span class="com"># UninitializedParameter</span>

x = torch.<span class="fn">randn</span>(<span class="num">4</span>, <span class="num">128</span>)
y = <span class="fn">layer</span>(x)                       <span class="com"># first call infers in_features=128, materializes weights</span>
<span class="fn">print</span>(layer.weight.shape)          <span class="com"># torch.Size([10, 128])</span></code></pre>

<p>Useful when you don't want to plumb shape information through your <code>__init__</code>. The catch: until materialized, the optimizer can't see the parameters, and <code>state_dict</code> entries don't exist. So you typically run a dummy forward pass before doing anything else.</p>

<h3><code>nn.utils.parametrize</code></h3>

<p>Reparameterize a parameter as a function of an underlying tensor — useful for constraints (orthogonal matrices, weight normalization, low-rank updates):</p>

<pre><code><span class="kw">import</span> torch.nn.utils.parametrize <span class="kw">as</span> P

<span class="kw">class</span> <span class="ty">Symmetric</span>(nn.Module):
    <span class="kw">def</span> <span class="fn">forward</span>(self, X):
        <span class="kw">return</span> X.<span class="fn">triu</span>() + X.<span class="fn">triu</span>(<span class="num">1</span>).<span class="fn">transpose</span>(-<span class="num">1</span>, -<span class="num">2</span>)

linear = nn.<span class="fn">Linear</span>(<span class="num">5</span>, <span class="num">5</span>)
P.<span class="fn">register_parametrization</span>(linear, <span class="str">"weight"</span>, <span class="fn">Symmetric</span>())

<span class="com"># Now linear.weight is always symmetric, computed from a smaller underlying tensor</span></code></pre>

<p>Less common in everyday code, but the foundation of LoRA-style adapters and weight-norm reparameterization.</p>

<h2>Code Magnets: build a small Transformer block — correctly</h2>

<p>You're putting together a tiny transformer block: a multi-head attention, a layer norm, an MLP. Layer counts and module choices matter. Build the <code>__init__</code> that registers everything correctly.</p>

<div class="magnets">
<p>Arrange the magnets into a working <code>__init__</code>. Three are red herrings.</p>

<div class="magnet-pool">
  <span class="magnet">class Block(nn.Module):</span>
  <span class="magnet">def __init__(self, dim, n_heads, n_mlp_layers):</span>
  <span class="magnet">    super().__init__()</span>
  <span class="magnet">    self.norm1 = nn.LayerNorm(dim)</span>
  <span class="magnet">    self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)</span>
  <span class="magnet">    self.norm2 = nn.LayerNorm(dim)</span>
  <span class="magnet">    self.mlp_layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_mlp_layers)])</span>
  <span class="magnet">    self.mlp_layers = [nn.Linear(dim, dim) for _ in range(n_mlp_layers)]</span>
  <span class="magnet">    self.causal_mask = torch.tril(torch.ones(1024, 1024)).bool()</span>
  <span class="magnet">    self.register_buffer('causal_mask', torch.tril(torch.ones(1024, 1024)).bool())</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">class</span> <span class="ty">Block</span>(nn.Module):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, dim, n_heads, n_mlp_layers):
        <span class="fn">super</span>().<span class="fn">__init__</span>()
        self.norm1 = nn.<span class="fn">LayerNorm</span>(dim)
        self.attn = nn.<span class="fn">MultiheadAttention</span>(dim, n_heads, batch_first=<span class="kw">True</span>)
        self.norm2 = nn.<span class="fn">LayerNorm</span>(dim)
        self.mlp_layers = nn.<span class="fn">ModuleList</span>([nn.<span class="fn">Linear</span>(dim, dim) <span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(n_mlp_layers)])
        self.<span class="fn">register_buffer</span>(<span class="str">'causal_mask'</span>, torch.<span class="fn">tril</span>(torch.<span class="fn">ones</span>(<span class="num">1024</span>, <span class="num">1024</span>)).<span class="fn">bool</span>())</code></pre>
<p>The traps:</p>
<ul>
  <li>The plain <code>self.mlp_layers = [...]</code> is the canonical bug — looks fine, runs fine, but the layers aren't registered. Use <code>ModuleList</code>.</li>
  <li>The plain <code>self.causal_mask = torch.tril(...)</code> doesn't get moved to GPU. Use <code>register_buffer</code> so it travels with the model.</li>
</ul>
<p>Notice that <code>nn.LayerNorm</code>, <code>nn.MultiheadAttention</code>, and <code>nn.Linear</code> are all registered automatically just by being assigned to <code>self.x</code> — because they're <code>nn.Module</code> subclasses, <code>__setattr__</code> routes them to <code>_modules</code>. The traps are about types that <em>aren't</em> Modules: lists and raw tensors.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each Module concept to its purpose.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Purpose</div>

  <div>nn.Parameter</div>
  <div>A. Recursively flips a Boolean flag that some layers (Dropout, BN) check.</div>

  <div>register_buffer</div>
  <div>B. A Module-aware list. Children are visible to .parameters() and .cuda().</div>

  <div>nn.ModuleList</div>
  <div>C. Marks a tensor as trainable; appears in .parameters() and gets gradients.</div>

  <div>state_dict()</div>
  <div>D. An OrderedDict of fully-qualified names to parameter and buffer tensors.</div>

  <div>model.train() / .eval()</div>
  <div>E. Registers a non-trainable tensor as part of model state — moves with .cuda(), saved by default.</div>

  <div>weight tying</div>
  <div>F. Two attributes pointing to the same underlying Parameter. Halves memory; updates from either path affect both.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>nn.Parameter</strong> → C<br>
<strong>register_buffer</strong> → E<br>
<strong>nn.ModuleList</strong> → B<br>
<strong>state_dict()</strong> → D<br>
<strong>model.train() / .eval()</strong> → A<br>
<strong>weight tying</strong> → F
</p>
<p>The mental shortcut: <em>Parameter is for trainable, buffer for non-trainable-but-stateful, ModuleList for collections of modules, state_dict is the persistence shape, train/eval is just a flag, weight tying is shared identity</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> Write a small Module that wraps a list of Linear layers. Test that all parameters are visible to <code>.parameters()</code>, that <code>.cuda()</code> moves all of them, and that <code>state_dict()</code> contains entries for each.</p>
<details class="answer"><summary>show answer</summary>
<pre><code><span class="kw">class</span> <span class="ty">MLP</span>(nn.Module):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, dims):
        <span class="fn">super</span>().<span class="fn">__init__</span>()
        self.layers = nn.<span class="fn">ModuleList</span>(
            [nn.<span class="fn">Linear</span>(a, b) <span class="kw">for</span> a, b <span class="kw">in</span> <span class="fn">zip</span>(dims[:-<span class="num">1</span>], dims[<span class="num">1</span>:])]
        )

    <span class="kw">def</span> <span class="fn">forward</span>(self, x):
        <span class="kw">for</span> layer <span class="kw">in</span> self.layers[:-<span class="num">1</span>]:
            x = <span class="fn">layer</span>(x).<span class="fn">relu</span>()
        <span class="kw">return</span> self.layers[-<span class="num">1</span>](x)

m = <span class="fn">MLP</span>([<span class="num">10</span>, <span class="num">20</span>, <span class="num">20</span>, <span class="num">5</span>])
<span class="fn">print</span>(<span class="fn">sum</span>(p.<span class="fn">numel</span>() <span class="kw">for</span> p <span class="kw">in</span> m.<span class="fn">parameters</span>()))   <span class="com"># 645</span>
<span class="fn">print</span>(<span class="fn">list</span>(m.<span class="fn">state_dict</span>().<span class="fn">keys</span>()))
<span class="com"># ['layers.0.weight', 'layers.0.bias', 'layers.1.weight', 'layers.1.bias',</span>
<span class="com">#  'layers.2.weight', 'layers.2.bias']</span></code></pre>
<p>Try replacing <code>nn.ModuleList(...)</code> with a plain list <code>[...]</code> and re-run the prints. Both will collapse to <code>0</code> and <code>[]</code>.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> A colleague's model has BatchNorm layers and they want to "freeze" the encoder. They write <code>for p in encoder.parameters(): p.requires_grad = False</code>. Why is the model still slowly drifting during training, and what's the fix?</p>
<details class="answer"><summary>show answer</summary>
<p>BatchNorm's running statistics are <em>buffers</em>, not parameters. <code>requires_grad = False</code> only affects parameters — the BN running mean and variance still update during forward as long as the layer is in training mode. The fix: also call <code>encoder.eval()</code> to flip the training flag. Now BN uses (and stops updating) the running stats. For full freeze, you typically want <em>both</em> the requires_grad flip and the eval mode.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> You have a checkpoint trained with <code>DistributedDataParallel</code>. Every key in the state_dict starts with <code>module.</code>. Load it into a non-DDP-wrapped model.</p>
<details class="answer"><summary>show answer</summary>
<pre><code>sd = torch.<span class="fn">load</span>(<span class="str">'ddp_checkpoint.pt'</span>, weights_only=<span class="kw">True</span>)
sd = {k.<span class="fn">removeprefix</span>(<span class="str">'module.'</span>): v <span class="kw">for</span> k, v <span class="kw">in</span> sd.<span class="fn">items</span>()}
model.<span class="fn">load_state_dict</span>(sd)</code></pre>
<p>Or alternatively, wrap the model in DDP first and then load — that adds the prefix on the model side. The string-manipulation approach is more flexible and the standard pattern. Note <code>.removeprefix</code> is Python 3.9+; pre-3.9 you'd write <code>k[7:] if k.startswith('module.') else k</code>.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> Implement weight tying between an embedding and an output projection. Verify that updating one name's weight tensor in place changes the other.</p>
<details class="answer"><summary>show answer</summary>
<pre><code><span class="kw">class</span> <span class="ty">TiedLM</span>(nn.Module):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, vocab, dim):
        <span class="fn">super</span>().<span class="fn">__init__</span>()
        self.embed = nn.<span class="fn">Embedding</span>(vocab, dim)
        self.head = nn.<span class="fn">Linear</span>(dim, vocab, bias=<span class="kw">False</span>)
        self.head.weight = self.embed.weight    <span class="com"># tie</span>

m = <span class="fn">TiedLM</span>(<span class="num">100</span>, <span class="num">8</span>)
<span class="fn">print</span>(m.embed.weight.<span class="fn">data_ptr</span>() == m.head.weight.<span class="fn">data_ptr</span>())   <span class="com"># True</span>

m.embed.weight.<span class="fn">data</span>[<span class="num">0</span>, <span class="num">0</span>] = <span class="num">42.0</span>
<span class="fn">print</span>(m.head.weight[<span class="num">0</span>, <span class="num">0</span>].<span class="fn">item</span>())                       <span class="com"># 42.0</span>

<span class="fn">print</span>(<span class="fn">sum</span>(p.<span class="fn">numel</span>() <span class="kw">for</span> p <span class="kw">in</span> m.<span class="fn">parameters</span>()))             <span class="com"># 800 (not 1600)</span></code></pre>
<p>The dedup happens because <code>parameters()</code> tracks tensors by identity (Python <code>id()</code>), so the same Parameter referenced under two names is yielded only once. The optimizer therefore allocates one set of momentum buffers, not two. Free memory.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>An <code>nn.Module</code> is <strong>three dictionaries plus a forward method</strong>: <code>_parameters</code>, <code>_buffers</code>, <code>_modules</code>. Anything you want PyTorch to find must be in one of those.</li>
  <li><code>nn.Parameter</code> = trainable tensor (in <code>parameters()</code>, gets gradients, optimizer updates it).</li>
  <li><code>register_buffer</code> = stateful tensor that's not trained (BN running stats, masks, RoPE tables). In state_dict by default; set <code>persistent=False</code> if you don't want it saved.</li>
  <li>The <strong>registration gotcha</strong>: <code>self.layers = [...]</code> is invisible. Use <code>nn.ModuleList</code>, <code>nn.ModuleDict</code>, or <code>nn.Sequential</code>.</li>
  <li><code>state_dict()</code> uses <strong>dotted hierarchical keys</strong> reflecting the module tree. Save the dict, not the model object — survives refactors.</li>
  <li><code>load_state_dict(sd, strict=False)</code> returns <code>(missing, unexpected)</code> for partial loads. Use it during fine-tuning.</li>
  <li><code>model.train()</code> / <code>.eval()</code> just <strong>flips a Boolean</strong>. It only matters because Dropout and BN check it.</li>
  <li><strong>Three things commonly confused for "eval mode"</strong>: training flag (Dropout/BN), <code>no_grad</code> (autograd off), <code>requires_grad=False</code> (parameter freeze). Inference wants the first two; freezing for fine-tuning wants the last one (often plus eval).</li>
  <li><strong>Weight tying</strong>: <code>self.head.weight = self.embed.weight</code>. Same tensor, two names. Halves memory and parameter count.</li>
  <li><strong>DDP prefix</strong>: state_dict keys from a DDP model have <code>module.</code> prepended. Strip with <code>removeprefix</code> when loading into an unwrapped model.</li>
  <li>The reflex: when something registered "isn't there," ask <em>which dictionary should it have gone into, and did <code>__setattr__</code> route it correctly?</em> The answer is almost always "you assigned a plain Python container or raw tensor."</li>
</ul>
</div>

<p>Module 08 takes the container model you just built and fills it with the layers and initialization schemes that make a model trainable. We'll cover Xavier vs Kaiming vs GPT-style init, Dropout / BatchNorm / LayerNorm / RMSNorm / GroupNorm, and the activation/normalization choices that matter most.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">07</span>
  <span>nn.Module deeply</span>
</div>
"""

emit("07_nn_module_deeply", "Module 07 — nn.Module deeply", BODY)
