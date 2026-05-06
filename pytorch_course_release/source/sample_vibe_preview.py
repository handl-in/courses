#!/usr/bin/env python3
"""Vibe preview — same M1 content, full Head First treatment.

Demonstrates: Code Magnets, anthropomorphized characters, Bullet Points
recap section, "Who Does What?" matching, and a couple of SVG visual moments.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Vibe Preview · Module 01 (excerpt)</div>
  <h1 class="module-title">Tensors, <em>from scratch</em></h1>
  <p class="module-sub">— full Head First treatment: characters talking, magnets to arrange, matching games, and the bullet-points recap</p>
</div>

<p>This is a preview of what Module 01 looks like with the full vibe upgrade. Same teaching content, more Head First personality. Compare to the original M1 to see the difference.</p>

<h2>Meet the cast</h2>

<p>Before we dissect a tensor, let's hear from the players themselves. Each one is going to play a role in this story, so it helps to put a face to the name.</p>

<div class="character storage">
  <div class="avatar">S</div>
  <div>
    <p class="who">Storage</p>
    <p class="name">"Hi, I'm a flat buffer of bytes."</p>
    <p class="says">I'm just a chunk of memory. I don't know whether I'm a 3×4 matrix or a 12-element vector. That's not my job. My job is to hold bytes. Talk to Shape if you want opinions.</p>
  </div>
</div>

<div class="character tensor">
  <div class="avatar">T</div>
  <div>
    <p class="who">Tensor</p>
    <p class="name">"I'm the face of the operation."</p>
    <p class="says">When you call <code>x.shape</code> or <code>x[2, 3]</code>, you're talking to me. But I don't store any of the actual numbers. I just hold a reference to Storage and a recipe for how to read it. Without my friends Shape, Stride, and Offset, I'd be useless.</p>
  </div>
</div>

<div class="character stride">
  <div class="avatar">→</div>
  <div>
    <p class="who">Stride</p>
    <p class="name">"I'm the recipe."</p>
    <p class="says">If you want element <code>[2, 3]</code>, ask me where to look in Storage. I'll say: "go to byte <code>(2 × my_first_value + 3 × my_second_value) × dtype_size</code>." That's it. I'm the secret to why <code>transpose</code> is free. Change me and the same Storage looks completely different.</p>
  </div>
</div>

<div class="character dtype">
  <div class="avatar">#</div>
  <div>
    <p class="who">Dtype</p>
    <p class="name">"I tell you how big each number is."</p>
    <p class="says">Storage just sees bytes. I'm the one who says "every 4 of these bytes is one float32" or "every 8 is one int64." Change me and you reinterpret the same memory as different numbers. This is more dangerous than it sounds.</p>
  </div>
</div>

<p>That's the cast. Storage holds the bytes. Tensor is the public face. Stride is the recipe for navigating Storage. Dtype tells you the unit. Now let's watch them work together.</p>

<h2>The five-thing model</h2>

<p>Open a Python REPL — actually do this — and let's poke a tensor with a stick.</p>

<pre><code><span class="kw">import</span> torch

x = torch.arange(<span class="num">12</span>).reshape(<span class="num">3</span>, <span class="num">4</span>)
<span class="fn">print</span>(x)
<span class="com"># tensor([[ 0,  1,  2,  3],</span>
<span class="com">#         [ 4,  5,  6,  7],</span>
<span class="com">#         [ 8,  9, 10, 11]])</span>

<span class="fn">print</span>(x.shape)              <span class="com"># torch.Size([3, 4])</span>
<span class="fn">print</span>(x.stride())           <span class="com"># (4, 1)</span>
<span class="fn">print</span>(x.storage_offset())   <span class="com"># 0</span>
<span class="fn">print</span>(x.dtype)              <span class="com"># torch.int64</span>
<span class="fn">print</span>(x.device)             <span class="com"># cpu</span></code></pre>

<p>Five attributes. Five things you need to keep in your head:</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 700 320" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <!-- Storage row -->
  <text x="20" y="40" font-size="14" font-weight="700" fill="#1a1612">storage (the bytes)</text>
  <g transform="translate(20, 50)">
    <!-- 12 cells -->
    <g font-size="13" text-anchor="middle">
      <rect x="0"   y="0" width="50" height="40" fill="#fff" stroke="#1a1612" stroke-width="1.5"/><text x="25"  y="26" fill="#1a1612">0</text>
      <rect x="50"  y="0" width="50" height="40" fill="#fff" stroke="#1a1612" stroke-width="1.5"/><text x="75"  y="26" fill="#1a1612">1</text>
      <rect x="100" y="0" width="50" height="40" fill="#fff" stroke="#1a1612" stroke-width="1.5"/><text x="125" y="26" fill="#1a1612">2</text>
      <rect x="150" y="0" width="50" height="40" fill="#fff" stroke="#1a1612" stroke-width="1.5"/><text x="175" y="26" fill="#1a1612">3</text>
      <rect x="200" y="0" width="50" height="40" fill="#fff" stroke="#1a1612" stroke-width="1.5"/><text x="225" y="26" fill="#1a1612">4</text>
      <rect x="250" y="0" width="50" height="40" fill="#fff" stroke="#1a1612" stroke-width="1.5"/><text x="275" y="26" fill="#1a1612">5</text>
      <rect x="300" y="0" width="50" height="40" fill="#fff" stroke="#1a1612" stroke-width="1.5"/><text x="325" y="26" fill="#1a1612">6</text>
      <rect x="350" y="0" width="50" height="40" fill="#fff" stroke="#1a1612" stroke-width="1.5"/><text x="375" y="26" fill="#1a1612">7</text>
      <rect x="400" y="0" width="50" height="40" fill="#fff" stroke="#1a1612" stroke-width="1.5"/><text x="425" y="26" fill="#1a1612">8</text>
      <rect x="450" y="0" width="50" height="40" fill="#fff" stroke="#1a1612" stroke-width="1.5"/><text x="475" y="26" fill="#1a1612">9</text>
      <rect x="500" y="0" width="50" height="40" fill="#fff" stroke="#1a1612" stroke-width="1.5"/><text x="525" y="26" fill="#1a1612">10</text>
      <rect x="550" y="0" width="50" height="40" fill="#fff" stroke="#1a1612" stroke-width="1.5"/><text x="575" y="26" fill="#1a1612">11</text>
    </g>
    <!-- index labels below -->
    <g font-size="10" text-anchor="middle" fill="#6b5d4f">
      <text x="25" y="58">[0]</text>
      <text x="225" y="58">[4]</text>
      <text x="425" y="58">[8]</text>
    </g>
  </g>

  <!-- Brackets for rows -->
  <g transform="translate(20, 130)">
    <!-- Row 0 bracket -->
    <path d="M 0 0 L 0 -10 L 200 -10 L 200 0" fill="none" stroke="#c1502e" stroke-width="2"/>
    <text x="100" y="22" font-size="13" font-weight="700" fill="#c1502e" text-anchor="middle">row 0 → stride[0]=4</text>
    <!-- Row 1 bracket -->
    <path d="M 200 0 L 200 -10 L 400 -10 L 400 0" fill="none" stroke="#1f5f5b" stroke-width="2"/>
    <text x="300" y="22" font-size="13" font-weight="700" fill="#1f5f5b" text-anchor="middle">row 1 → +4 elements</text>
    <!-- Row 2 bracket -->
    <path d="M 400 0 L 400 -10 L 600 -10 L 600 0" fill="none" stroke="#d4a017" stroke-width="2"/>
    <text x="500" y="22" font-size="13" font-weight="700" fill="#8c6512" text-anchor="middle">row 2 → +4 elements</text>
  </g>

  <!-- 3x4 matrix -->
  <text x="350" y="195" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">how YOU see it: shape (3, 4) with stride (4, 1)</text>
  <g transform="translate(220, 210)" font-size="14" text-anchor="middle">
    <rect x="0"   y="0"  width="60" height="32" fill="#f6efe1" stroke="#c1502e" stroke-width="1.5"/><text x="30"  y="22" fill="#1a1612">0</text>
    <rect x="60"  y="0"  width="60" height="32" fill="#f6efe1" stroke="#c1502e" stroke-width="1.5"/><text x="90"  y="22" fill="#1a1612">1</text>
    <rect x="120" y="0"  width="60" height="32" fill="#f6efe1" stroke="#c1502e" stroke-width="1.5"/><text x="150" y="22" fill="#1a1612">2</text>
    <rect x="180" y="0"  width="60" height="32" fill="#f6efe1" stroke="#c1502e" stroke-width="1.5"/><text x="210" y="22" fill="#1a1612">3</text>
    <rect x="0"   y="32" width="60" height="32" fill="#f6efe1" stroke="#1f5f5b" stroke-width="1.5"/><text x="30"  y="54" fill="#1a1612">4</text>
    <rect x="60"  y="32" width="60" height="32" fill="#f6efe1" stroke="#1f5f5b" stroke-width="1.5"/><text x="90"  y="54" fill="#1a1612">5</text>
    <rect x="120" y="32" width="60" height="32" fill="#f6efe1" stroke="#1f5f5b" stroke-width="1.5"/><text x="150" y="54" fill="#1a1612">6</text>
    <rect x="180" y="32" width="60" height="32" fill="#f6efe1" stroke="#1f5f5b" stroke-width="1.5"/><text x="210" y="54" fill="#1a1612">7</text>
    <rect x="0"   y="64" width="60" height="32" fill="#f6efe1" stroke="#d4a017" stroke-width="1.5"/><text x="30"  y="86" fill="#1a1612">8</text>
    <rect x="60"  y="64" width="60" height="32" fill="#f6efe1" stroke="#d4a017" stroke-width="1.5"/><text x="90"  y="86" fill="#1a1612">9</text>
    <rect x="120" y="64" width="60" height="32" fill="#f6efe1" stroke="#d4a017" stroke-width="1.5"/><text x="150" y="86" fill="#1a1612">10</text>
    <rect x="180" y="64" width="60" height="32" fill="#f6efe1" stroke="#d4a017" stroke-width="1.5"/><text x="210" y="86" fill="#1a1612">11</text>
  </g>
  <!-- Caveat-style annotation -->
  <text x="500" y="260" font-family="'Caveat', cursive" font-size="20" fill="#c1502e">same bytes,</text>
  <text x="500" y="282" font-family="'Caveat', cursive" font-size="20" fill="#c1502e">different recipe!</text>
  <path d="M 490 260 Q 460 250 440 250" stroke="#c1502e" stroke-width="2" fill="none" marker-end="url(#arr)"/>
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#c1502e"/>
    </marker>
  </defs>
</svg>
</div>

<p>Look at the diagram. Storage on top is just bytes. The brackets show how Stride <code>(4, 1)</code> chops it into rows: jump 4 to start a new row, jump 1 to advance a column. The <em>matrix you see</em> is just the bytes plus the recipe.</p>

<div class="character stride">
  <div class="avatar">→</div>
  <div>
    <p class="who">Stride, again</p>
    <p class="name">"Want to transpose? Just swap me."</p>
    <p class="says">Look — if you call <code>x.transpose(0, 1)</code>, none of the bytes move. PyTorch just hands you a new Tensor that points at the same Storage but with my values flipped: <code>(1, 4)</code> instead of <code>(4, 1)</code>. Now to advance one row, you jump 1; to advance one column, jump 4. Same data, different journey. <em>That's why transpose is free.</em></p>
  </div>
</div>

<h2>Code Magnets</h2>

<p>Time to test your understanding. Below is a tray of code magnets. Your job: arrange them in your head (or on paper) to produce a tensor whose storage is the numbers <code>0..11</code> but whose <em>shape</em> is <code>(4, 3)</code> in transposed-looking order — that is, when you print it, you see:</p>

<pre><code>tensor([[ 0,  4,  8],
        [ 1,  5,  9],
        [ 2,  6, 10],
        [ 3,  7, 11]])</code></pre>

<div class="magnets">
<p>Drag these magnets (mentally) into a single Python statement. Use exactly four of them; one is a red herring.</p>

<div class="magnet-pool">
  <span class="magnet">torch.arange(12)</span>
  <span class="magnet">.reshape(3, 4)</span>
  <span class="magnet">.reshape(4, 3)</span>
  <span class="magnet">.transpose(0, 1)</span>
  <span class="magnet">.contiguous()</span>
</div>

<details class="answer"><summary>show solution</summary>
<p>The arrangement that works:</p>
<pre><code>torch.arange(<span class="num">12</span>).reshape(<span class="num">3</span>, <span class="num">4</span>).transpose(<span class="num">0</span>, <span class="num">1</span>).contiguous()</code></pre>
<p>The red herring is <code>.reshape(4, 3)</code> — it would give you the wrong layout (numbers <code>0..11</code> in row-major, not the transposed pattern). The trick is that <code>reshape(3, 4)</code> first lays out as <code>[[0,1,2,3], [4,5,6,7], [8,9,10,11]]</code>, then <code>transpose(0, 1)</code> swaps stride to <code>(1, 4)</code> giving the desired output. Calling <code>.contiguous()</code> at the end is optional unless you need the bytes physically reordered for a downstream kernel.</p>
</details>
</div>

<h2>Who does what?</h2>

<p>Now a matching exercise. Each operation on the left does <em>one</em> of the things on the right. Match them up.</p>

<div class="matching">
<p class="intro">For each operation in the left column, identify what it does to memory.</p>

<div class="match-grid">
  <div class="header">Operation</div>
  <div class="header">What does it do?</div>

  <div>x.view(-1)</div>
  <div>A. Always allocates new storage and copies.</div>

  <div>x.transpose(0, 1)</div>
  <div>B. Rewrites stride, no memory touched. Errors if not contiguous.</div>

  <div>x.clone()</div>
  <div>C. Returns a view with stride 0 in the broadcast dimensions.</div>

  <div>x.reshape(-1)</div>
  <div>D. Swaps two stride values. No bytes move. Result is non-contiguous.</div>

  <div>x.expand(3, 4)</div>
  <div>E. Tries to make a view; copies if it has to.</div>

  <div>x[2:5]</div>
  <div>F. Shifts the storage offset and creates a view.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>x.view(-1)</strong> → B (rewrites stride, errors if non-contiguous)<br>
<strong>x.transpose(0, 1)</strong> → D (swaps stride values, no bytes move)<br>
<strong>x.clone()</strong> → A (always allocates and copies)<br>
<strong>x.reshape(-1)</strong> → E (view if possible, copy if necessary)<br>
<strong>x.expand(3, 4)</strong> → C (stride 0 in broadcast dims)<br>
<strong>x[2:5]</strong> → F (shifts offset, view)
</p>
<p>If you got <em>view</em> and <em>reshape</em> mixed up: the difference is that <code>view</code> is strict — it refuses to copy. <code>reshape</code> is permissive — it'll silently copy when needed. Same outcome on contiguous tensors; very different on transposed ones.</p>
</details>
</div>

<h2>One more conversation</h2>

<div class="character autograd">
  <div class="avatar">∂</div>
  <div>
    <p class="who">Autograd (eavesdropping)</p>
    <p class="name">"Just a heads-up about views."</p>
    <p class="says">When Tensor and Storage chat about <em>views</em>, I'm the one who has to deal with the consequences. If you mutate a view in-place — like <code>x[0].zero_()</code> — that's a write to the parent's Storage, and I might need its <em>old</em> value for the backward pass. So I track version counters. If a Storage has been bumped, I refuse to compute the gradient for the stale value. We'll talk more in Module 4. For now, just know: <em>in-place writes through views are how I learn to mistrust you</em>.</p>
  </div>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>A <strong>Tensor</strong> is a small object that holds a reference to <strong>Storage</strong> (the actual bytes) plus a <em>recipe</em> for reading it: <strong>shape, stride, offset, dtype</strong>.</li>
  <li><strong>Storage doesn't know it's a matrix.</strong> It's a flat byte buffer. Multiple Tensors can share the same Storage with different recipes — that's why <code>transpose</code> is free.</li>
  <li>The lookup formula is always: <code>storage[offset + sum(index_i × stride_i)]</code>.</li>
  <li><strong>Stride 0</strong> is the broadcasting trick — the dimension lies, you read the same value many times.</li>
  <li><code>view</code> is strict (no copy ever, errors if it can't); <code>reshape</code> is permissive (copies when forced); <code>permute</code>/<code>transpose</code> just swap stride values.</li>
  <li>An operation that mutates Storage (in-place ops, the underscore family) affects every Tensor that views it. <strong>Autograd watches for this</strong> via version counters.</li>
  <li>The reflex: when something tensor-related surprises you, ask "what's the storage, what's the stride, what's the offset?" Nine times out of ten the answer is right there.</li>
</ul>
</div>

<p style="font-style: italic; color: var(--ink-muted); text-align: center; margin-top: 36px;">— end of vibe preview —</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">★</span>
  <span>Vibe Preview</span>
</div>
"""

emit("vibe_preview", "Vibe Preview — full Head First treatment", BODY)
