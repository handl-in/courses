#!/usr/bin/env python3
"""Module 06: Custom autograd: torch.autograd.Function — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part II · Module 06</div>
  <h1 class="module-title">Custom autograd: <em><code>torch.autograd.Function</code></em></h1>
  <p class="module-sub">— writing your own forward and backward, the four reasons it earns its keep, and the trick that makes FlashAttention possible</p>
</div>

<p>So far autograd has been a black box that does the right thing. You write the forward, PyTorch writes the backward, life is good. This module is about the moment you decide to <em>open the box</em>.</p>

<p>You will not do this often. PyTorch's automatic backward handles 99% of cases correctly and efficiently. But when you need the 1% — a fused kernel that runs in 1/4 the memory, a non-differentiable op that needs a "fake" gradient, a custom CUDA kernel you'll write in Module 23 — you need to know how. And the act of writing one custom backward by hand teaches you more about the framework than ten more modules of API tour.</p>

<div class="keyidea">
A custom <code>autograd.Function</code> is <strong>a class with two static methods</strong>: <code>forward</code> computes the output, <code>backward</code> takes the upstream gradient and returns the downstream gradients. PyTorch wraps your class in a grad_fn and treats it just like any built-in op. There are <strong>four real reasons to write one</strong>: memory savings (recompute instead of save), fusing many ops into one (custom kernel integration), non-differentiable ops with custom gradients (like rounding), and replacing PyTorch's backward with a smarter one (like FlashAttention).
</div>

<h2>Meet the new face</h2>

<div class="character" style="--c: #133e3b;">
  <div class="avatar" style="background: #133e3b; color: #fff;">𝓕</div>
  <div>
    <p class="who">Function</p>
    <p class="name">"I'm what <code>grad_fn</code> is when <em>you</em> make one."</p>
    <p class="says">When you subclass <code>torch.autograd.Function</code>, you're defining a new operation as far as Autograd is concerned. The <code>forward</code> method is the math; the <code>backward</code> method is the chain rule. PyTorch creates an instance of me each time you call your op — I become the <code>grad_fn</code> of the output tensor. From the outside, no one can tell I'm not a built-in op. From the inside, you have <em>complete control</em> over what gets saved and what gets recomputed.</p>
  </div>
</div>

<h2>Anatomy of a custom Function</h2>

<p>The minimal skeleton:</p>

<pre><code><span class="kw">class</span> <span class="ty">MyOp</span>(torch.autograd.Function):
    <span class="kw">@staticmethod</span>
    <span class="kw">def</span> <span class="fn">forward</span>(ctx, *inputs):
        <span class="com"># Compute the output.</span>
        <span class="com"># Save anything needed for backward via ctx.save_for_backward(...).</span>
        <span class="kw">return</span> output

    <span class="kw">@staticmethod</span>
    <span class="kw">def</span> <span class="fn">backward</span>(ctx, *grad_outputs):
        <span class="com"># grad_outputs[i] is the upstream gradient for forward's i-th output.</span>
        <span class="com"># Return one gradient per input to forward (in the same order).</span>
        <span class="com"># Return None for inputs that don't need a gradient.</span>
        <span class="kw">return</span> grad_input1, grad_input2, ...

<span class="com"># To use it:</span>
y = MyOp.<span class="fn">apply</span>(x1, x2)         <span class="com"># note: .apply, not direct construction</span></code></pre>

<p>Three things to internalize:</p>

<ol>
  <li><strong>Both methods are <code>@staticmethod</code>.</strong> PyTorch passes a fresh <code>ctx</code> object to each — that's where you stash anything backward needs to remember from forward.</li>
  <li><strong>You call <code>MyOp.apply(...)</code></strong>, not <code>MyOp(...)</code>. The <code>apply</code> method is what hooks into autograd, creating the grad_fn and tying it to the output.</li>
  <li><strong>Backward returns one gradient per <em>forward input</em></strong>, in the same order. If forward took <code>(x, y, alpha)</code>, backward returns <code>(grad_x, grad_y, grad_alpha)</code>. Inputs you don't differentiate (like a config flag) get <code>None</code>.</li>
</ol>

<h2>The simplest possible example</h2>

<p>Let's implement <code>square</code>. It's trivially correct and lets us focus on the API.</p>

<pre><code><span class="kw">class</span> <span class="ty">Square</span>(torch.autograd.Function):
    <span class="kw">@staticmethod</span>
    <span class="kw">def</span> <span class="fn">forward</span>(ctx, x):
        ctx.<span class="fn">save_for_backward</span>(x)        <span class="com"># save x because we need it in backward</span>
        <span class="kw">return</span> x * x

    <span class="kw">@staticmethod</span>
    <span class="kw">def</span> <span class="fn">backward</span>(ctx, grad_out):
        x, = ctx.saved_tensors             <span class="com"># note the comma — it's a tuple</span>
        <span class="com"># d(x²)/dx = 2x. Chain rule: dL/dx = dL/dy · dy/dx = grad_out · 2x</span>
        <span class="kw">return</span> grad_out * <span class="num">2</span> * x

<span class="com"># Try it:</span>
x = torch.tensor([<span class="num">3.0</span>, <span class="num">4.0</span>], requires_grad=<span class="kw">True</span>)
y = Square.<span class="fn">apply</span>(x).<span class="fn">sum</span>()
y.<span class="fn">backward</span>()
<span class="fn">print</span>(x.grad)                            <span class="com"># tensor([6., 8.]) — that's 2x ✓</span></code></pre>

<p>That's the entire pattern. <code>save_for_backward</code> stashes the tensor in <code>ctx</code>; <code>backward</code> retrieves it from <code>ctx.saved_tensors</code> and applies the chain rule by hand.</p>

<div class="character autograd">
  <div class="avatar">∂</div>
  <div>
    <p class="who">Autograd</p>
    <p class="name">"<code>save_for_backward</code> isn't just stashing — it's a contract."</p>
    <p class="says">When you save a tensor, I track it specially. If anyone modifies it in place between forward and backward, I'll detect that with version counters and yell at you (the "version mismatch" error). I also know not to free its memory until backward is done. <em>Always</em> save through <code>ctx.save_for_backward()</code>; never just stash on <code>ctx</code> as a plain attribute. The latter works for non-tensors but skips the safety net.</p>
  </div>
</div>

<div class="warn">
<strong>Save tensors via <code>save_for_backward</code>; save plain Python data as <code>ctx.attribute</code>.</strong> Mixing these up is a common bug. <code>ctx.x = some_tensor</code> works — barely — but skips version tracking and confuses anyone reading your code. Use <code>ctx.save_for_backward(some_tensor)</code> for tensors, <code>ctx.alpha = 0.5</code> for hyperparameters.
</div>

<h2>A real example: LayerNorm by hand</h2>

<p>Now let's implement something real. LayerNorm is a great target because (a) it's used everywhere, (b) its backward involves a non-trivial Jacobian, and (c) the official PyTorch version is more memory-efficient than the naive automatic backward — which is exactly the kind of win custom Functions are for.</p>

<p>The math, one more time. For input <code>x</code> of shape <code>(..., D)</code> and learnable <code>weight</code>, <code>bias</code> of shape <code>(D,)</code>:</p>

<pre><code>μ = mean(x, dim=-1)           <span class="com"># per-row mean</span>
σ² = var(x, dim=-1)           <span class="com"># per-row variance (biased)</span>
x̂ = (x - μ) / sqrt(σ² + ε)
y = x̂ * weight + bias</code></pre>

<p>The backward is uglier than this looks. Each output element <code>y[i, j]</code> depends on every <code>x[i, k]</code> through both the mean and the variance. The full derivation is two pages of tensor calculus; the result is short:</p>

<pre><code><span class="kw">class</span> <span class="ty">MyLayerNorm</span>(torch.autograd.Function):
    <span class="kw">@staticmethod</span>
    <span class="kw">def</span> <span class="fn">forward</span>(ctx, x, weight, bias, eps=<span class="num">1e-5</span>):
        <span class="com"># Normalize over the last dimension</span>
        mean = x.<span class="fn">mean</span>(dim=-<span class="num">1</span>, keepdim=<span class="kw">True</span>)              <span class="com"># (..., 1)</span>
        var = x.<span class="fn">var</span>(dim=-<span class="num">1</span>, keepdim=<span class="kw">True</span>, unbiased=<span class="kw">False</span>)
        rstd = <span class="num">1.0</span> / (var + eps).<span class="fn">sqrt</span>()                <span class="com"># (..., 1)</span>
        x_hat = (x - mean) * rstd                          <span class="com"># (..., D)</span>
        y = x_hat * weight + bias                          <span class="com"># (..., D)</span>

        <span class="com"># Save what we need for backward.</span>
        <span class="com"># Note: we save x_hat and rstd, NOT x and var. Why? Smaller scratch space.</span>
        <span class="com"># x_hat already encodes (x - mean) * rstd, which we'd recompute anyway.</span>
        ctx.<span class="fn">save_for_backward</span>(x_hat, weight, rstd)
        ctx.eps = eps
        <span class="kw">return</span> y

    <span class="kw">@staticmethod</span>
    <span class="kw">def</span> <span class="fn">backward</span>(ctx, grad_y):
        x_hat, weight, rstd = ctx.saved_tensors
        D = x_hat.<span class="fn">shape</span>[-<span class="num">1</span>]

        <span class="com"># Gradients to the affine params: easy</span>
        <span class="com"># Sum over all leading dims (e.g., batch and sequence) so shapes match.</span>
        leading_dims = <span class="fn">tuple</span>(<span class="fn">range</span>(grad_y.dim() - <span class="num">1</span>))
        grad_bias   = grad_y.<span class="fn">sum</span>(dim=leading_dims)                  <span class="com"># (D,)</span>
        grad_weight = (grad_y * x_hat).<span class="fn">sum</span>(dim=leading_dims)        <span class="com"># (D,)</span>

        <span class="com"># Gradient to x: this is the painful part.</span>
        <span class="com"># dL/dx = (1/D) * rstd * (D · g - sum(g) - x_hat · sum(g · x_hat))</span>
        <span class="com"># where g = grad_y * weight</span>
        g = grad_y * weight                                            <span class="com"># (..., D)</span>
        sum_g       = g.<span class="fn">sum</span>(dim=-<span class="num">1</span>, keepdim=<span class="kw">True</span>)                 <span class="com"># (..., 1)</span>
        sum_g_xhat  = (g * x_hat).<span class="fn">sum</span>(dim=-<span class="num">1</span>, keepdim=<span class="kw">True</span>)        <span class="com"># (..., 1)</span>
        grad_x = (rstd / D) * (D * g - sum_g - x_hat * sum_g_xhat)

        <span class="com"># Order matches forward: (x, weight, bias, eps) → (grad_x, grad_w, grad_b, None)</span>
        <span class="kw">return</span> grad_x, grad_weight, grad_bias, <span class="kw">None</span></code></pre>

<p>A few things worth pointing out about that backward. First, <code>grad_bias</code> is just the upstream gradient summed across the batch — bias is added uniformly, so the gradient is the sum. Same for <code>grad_weight</code>, but multiplied by the normalized <code>x_hat</code>. Both follow naturally from the chain rule.</p>

<p>Second, the <code>grad_x</code> formula looks scary but reflects something simple: changing one element <code>x[i, k]</code> changes the mean and variance of row <code>i</code>, which in turn changes <em>every</em> output in row <code>i</code>. The three terms — the local effect, the mean's contribution, the variance's contribution — are exactly that.</p>

<p>Third — and this is the point — we saved <code>x_hat</code> and <code>rstd</code>, not <code>x</code> and <code>var</code>. Same memory cost, but backward needs less computation because <code>x_hat</code> is what we'd need to recompute anyway. <em>This kind of micro-optimization is why hand-written custom backwards beat the autodiff version.</em></p>

<div class="brainpower">
<p><strong>Pause.</strong> Why do we use <code>unbiased=False</code> when computing the variance?</p>
<p>Two reasons. (1) The traditional LayerNorm definition uses the population variance (divide by D, not D−1), so the formula matches papers and other implementations. (2) The backward formula derived above assumes <code>1/D</code>, not <code>1/(D-1)</code>. If you switched to unbiased, you'd need to redo the math. Always check that your definition and your derivative agree.</p>
</div>

<h2>Always run <code>gradcheck</code></h2>

<p>You wrote a backward by hand. Are you sure it's right? <code>torch.autograd.gradcheck</code> compares your analytical backward against numerical finite differences. It catches sign errors, missing terms, and the wrong reduction order in seconds.</p>

<pre><code><span class="kw">from</span> torch.autograd <span class="kw">import</span> gradcheck

<span class="com"># gradcheck expects fp64 inputs for stable finite differences</span>
x = torch.<span class="fn">randn</span>(<span class="num">2</span>, <span class="num">5</span>, dtype=torch.float64, requires_grad=<span class="kw">True</span>)
w = torch.<span class="fn">randn</span>(<span class="num">5</span>,    dtype=torch.float64, requires_grad=<span class="kw">True</span>)
b = torch.<span class="fn">zeros</span>(<span class="num">5</span>,   dtype=torch.float64, requires_grad=<span class="kw">True</span>)

<span class="fn">assert</span> <span class="fn">gradcheck</span>(MyLayerNorm.apply, (x, w, b), eps=<span class="num">1e-6</span>)
<span class="com"># If your backward is wrong, this raises with a clear message about which input.</span></code></pre>

<p>Three rules of thumb for <code>gradcheck</code>:</p>

<ol>
  <li><strong>Use fp64.</strong> Finite differences need precision; fp32 is too noisy for reliable comparison.</li>
  <li><strong>Use small tensors.</strong> gradcheck's cost is O(n_params × n_outputs). On a (32, 1024) tensor it'll take forever. Test on (2, 5) and trust the result.</li>
  <li><strong>Try a few random seeds.</strong> A single seed can hit lucky cancellations. Run a few; if all pass, you're confident.</li>
</ol>

<p>For double backward (Module 5's <code>create_graph=True</code> friend), there's a sibling: <code>gradgradcheck</code>. Run both if your op might appear in a Hessian computation.</p>

<h2>The four reasons to write one</h2>

<p>You now know <em>how</em> to write a Function. The harder question is <em>when</em>. Four real motivations.</p>

<h3>1. Memory savings: recompute instead of save</h3>

<p>The autograd-generated backward for an op saves <em>everything it might need</em>. If your forward computes <code>y = sigmoid(x)</code>, autograd saves <code>y</code> (because the derivative of sigmoid is <code>y * (1 - y)</code>, expressible in terms of the output). If your forward computes a long chain like <code>y = c(b(a(x)))</code>, autograd saves the intermediate <code>a(x)</code>, <code>b(a(x))</code>, etc., for backward.</p>

<p>For deep networks, those saved activations dominate memory. Activation checkpointing (Module 12) is the systematic answer, but custom Functions are the targeted one. You write a Function whose forward computes the chain, saves <em>only the input</em>, and whose backward <em>recomputes</em> the intermediates locally. Trades compute for memory.</p>

<pre><code><span class="kw">class</span> <span class="ty">FusedChain</span>(torch.autograd.Function):
    <span class="kw">@staticmethod</span>
    <span class="kw">def</span> <span class="fn">forward</span>(ctx, x):
        ctx.<span class="fn">save_for_backward</span>(x)         <span class="com"># save ONLY x, not intermediates</span>
        a = torch.<span class="fn">sin</span>(x)
        b = a * a
        c = b.<span class="fn">exp</span>()
        <span class="kw">return</span> c

    <span class="kw">@staticmethod</span>
    <span class="kw">def</span> <span class="fn">backward</span>(ctx, grad_c):
        x, = ctx.saved_tensors
        <span class="com"># Recompute the intermediates (cheap, on-device)</span>
        a = torch.<span class="fn">sin</span>(x)
        b = a * a
        c = b.<span class="fn">exp</span>()
        <span class="com"># Now apply the chain rule</span>
        grad_b = grad_c * c                  <span class="com"># dc/db = exp(b) = c</span>
        grad_a = grad_b * <span class="num">2</span> * a              <span class="com"># db/da = 2a</span>
        grad_x = grad_a * torch.<span class="fn">cos</span>(x)        <span class="com"># da/dx = cos(x)</span>
        <span class="kw">return</span> grad_x</code></pre>

<p>Memory: 1 tensor (<code>x</code>) instead of 4 (<code>x, a, b, c</code>). Compute: roughly 2× because we recompute the chain. For long-running training where memory is the binding constraint, this trade is hugely worth it. <strong>This is exactly what FlashAttention does at scale</strong>, and we'll see it again in Module 25.</p>

<h3>2. Fusing many ops into one (custom kernel integration)</h3>

<p>When you write a custom CUDA or Triton kernel (Modules 23-24), you need a way to plug it into PyTorch's autograd. <code>autograd.Function</code> is exactly that bridge:</p>

<pre><code><span class="kw">class</span> <span class="ty">MyFusedKernel</span>(torch.autograd.Function):
    <span class="kw">@staticmethod</span>
    <span class="kw">def</span> <span class="fn">forward</span>(ctx, x, w):
        <span class="com"># Call your hand-written CUDA kernel here. Could be Triton, raw CUDA,</span>
        <span class="com"># or a C++ extension. The Python side just orchestrates.</span>
        out = my_cuda_kernel.<span class="fn">forward</span>(x, w)
        ctx.<span class="fn">save_for_backward</span>(x, w)
        <span class="kw">return</span> out

    <span class="kw">@staticmethod</span>
    <span class="kw">def</span> <span class="fn">backward</span>(ctx, grad_out):
        x, w = ctx.saved_tensors
        grad_x, grad_w = my_cuda_kernel.<span class="fn">backward</span>(grad_out, x, w)
        <span class="kw">return</span> grad_x, grad_w</code></pre>

<p>Without this wrapper, your kernel can't be used in training — autograd doesn't know how to compute its gradient. With it, your kernel is a first-class operation, indistinguishable from a built-in.</p>

<h3>3. Non-differentiable ops with custom gradients (straight-through estimators)</h3>

<p>Some operations have no gradient. <code>torch.round</code>, <code>torch.argmax</code>, <code>x.long()</code>, sampling from a discrete distribution — they all have gradients that are zero almost everywhere and undefined at the boundaries. Useless for training.</p>

<p>The <em>straight-through estimator</em> (STE) is a classic hack: in the forward pass, do the non-differentiable thing; in the backward pass, pretend it was the identity function and pass the gradient through unchanged. The math is wrong; the empirical results are good enough. STE is the foundation of QAT (Module 26), VQ-VAEs, Gumbel-softmax workarounds, and more.</p>

<pre><code><span class="kw">class</span> <span class="ty">RoundSTE</span>(torch.autograd.Function):
    <span class="kw">@staticmethod</span>
    <span class="kw">def</span> <span class="fn">forward</span>(ctx, x):
        <span class="kw">return</span> x.<span class="fn">round</span>()                <span class="com"># non-differentiable</span>

    <span class="kw">@staticmethod</span>
    <span class="kw">def</span> <span class="fn">backward</span>(ctx, grad_out):
        <span class="kw">return</span> grad_out               <span class="com"># pretend round was the identity. Lie. Works.</span>

<span class="com"># Use:</span>
x = torch.<span class="fn">randn</span>(<span class="num">5</span>, requires_grad=<span class="kw">True</span>) * <span class="num">3</span>
q = RoundSTE.<span class="fn">apply</span>(x)             <span class="com"># integer-valued, but with a fake gradient path</span>
loss = (q ** <span class="num">2</span>).sum()
loss.<span class="fn">backward</span>()                     <span class="com"># x.grad is 2*q, propagated through the lie</span></code></pre>

<p>The same pattern handles binarization, discretization, top-k selection in differentiable models — anywhere you'd otherwise be stuck.</p>

<h3>4. Replacing PyTorch's backward with a smarter one</h3>

<p>Sometimes PyTorch's automatic backward is correct but algorithmically inferior. The headline example is attention: the <em>standard</em> autograd backward of <code>softmax(QKᵀ/√d)V</code> needs to materialize the full <code>(B, H, T, T)</code> attention matrix. For T=8192 and B=4 batch and H=32 heads, that's hundreds of GB.</p>

<p>FlashAttention sidesteps this by computing the same gradient through a <em>different algorithm</em> that doesn't ever materialize the full matrix. The forward and backward both tile the computation and recompute as needed. The output is mathematically identical; the memory profile is dramatically better. None of that fits inside autograd-generated code — it requires writing forward and backward by hand. We'll do this in Module 25.</p>

<div class="character" style="--c: #133e3b;">
  <div class="avatar" style="background: #133e3b; color: #fff;">𝓕</div>
  <div>
    <p class="who">Function</p>
    <p class="name">"My existence is what makes those stories possible."</p>
    <p class="says">Without me, every op's backward must be derivable from the op's forward. With me, you can write any forward you want and any backward you want, as long as they're <em>consistent</em> (gradcheck enforces that). FlashAttention, gradient checkpointing, custom CUDA kernels, straight-through estimators — all of them are instances of "we want a backward that the framework wouldn't generate for us." That's my whole purpose.</p>
  </div>
</div>

<h2>Common pitfalls</h2>

<div class="ndq">
<h4>About custom Functions</h4>

<p class="q">My backward returns the wrong shape. What did I mess up?</p>
<p class="a">Almost certainly broadcasting. If your forward broadcast a <code>(D,)</code> tensor over a <code>(B, T, D)</code> tensor, the gradient w.r.t. the <code>(D,)</code> tensor needs to <em>sum back over the broadcast dims</em> to recover shape <code>(D,)</code>. The autograd engine does this for you on built-in ops; you have to do it explicitly in custom backwards. See the LayerNorm example — note the <code>.sum(dim=leading_dims)</code> on grad_weight and grad_bias.</p>

<p class="q">I get "one of the variables needed for gradient computation has been modified by an inplace operation." What's that about?</p>
<p class="a">You saved a tensor in forward, then someone (maybe in a different module) wrote to it in place between forward and backward. The version counter detected this and refuses to use the now-corrupted saved value. Either don't mutate in place, or save a <code>.clone()</code>. Performance cost: an extra copy.</p>

<p class="q">Can I do non-tensor returns?</p>
<p class="a">Yes — forward can return tuples of tensors, plus you can stash non-tensor values on <code>ctx</code>. But the position-matching rule still holds: backward gets one upstream gradient per <em>tensor</em> output of forward, in order. If forward returned <code>(out1, out2, "some_string")</code>, backward gets <code>(grad_out1, grad_out2)</code>.</p>

<p class="q">Does my Function work with <code>torch.compile</code>?</p>
<p class="a">Mostly yes, but with caveats. <code>torch.compile</code> can call your Function as an opaque op (it won't try to trace into it). You may need to mark it with <code>@torch._dynamo.allow_in_graph</code> for some configurations. We'll see this in Module 21. For most kernel-style Functions (those wrapping a CUDA call), it works out of the box.</p>

<p class="q">Should I use <code>autograd.Function</code> for everything custom?</p>
<p class="a">No. If you can write your op as a composition of existing PyTorch ops and the autodiff backward is fine, do that. Custom Functions are for the four reasons above — memory savings, kernel integration, non-differentiable ops, smarter algorithms. Otherwise you're adding complexity for no benefit.</p>
</div>

<h2>The "saved for backward" memory diagram</h2>

<p>The whole point of custom Functions is often controlling what gets saved. Here's the picture for a 5-step chain.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 320" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrSv" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
  </defs>

  <text x="370" y="24" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">What gets saved? Two strategies, same forward.</text>

  <!-- Top row: Autograd default -->
  <text x="20" y="60" font-size="12" font-weight="700" fill="#c1502e">Default autograd: save EVERY intermediate</text>

  <g transform="translate(20, 75)">
    <g font-family="'IBM Plex Mono', monospace" font-size="13" text-anchor="middle">
      <rect x="0"   y="0" width="100" height="48" fill="#fcecec" stroke="#c1502e" stroke-width="2"/><text x="50"  y="20" fill="#1a1612">x</text><text x="50" y="38" font-size="11" fill="#c1502e">SAVED</text>
      <rect x="120" y="0" width="100" height="48" fill="#fcecec" stroke="#c1502e" stroke-width="2"/><text x="170" y="20" fill="#1a1612">a = sin(x)</text><text x="170" y="38" font-size="11" fill="#c1502e">SAVED</text>
      <rect x="240" y="0" width="100" height="48" fill="#fcecec" stroke="#c1502e" stroke-width="2"/><text x="290" y="20" fill="#1a1612">b = a*a</text><text x="290" y="38" font-size="11" fill="#c1502e">SAVED</text>
      <rect x="360" y="0" width="100" height="48" fill="#fcecec" stroke="#c1502e" stroke-width="2"/><text x="410" y="20" fill="#1a1612">c = exp(b)</text><text x="410" y="38" font-size="11" fill="#c1502e">SAVED</text>
      <rect x="480" y="0" width="100" height="48" fill="#fcecec" stroke="#c1502e" stroke-width="2"/><text x="530" y="20" fill="#1a1612">y = c*2</text><text x="530" y="38" font-size="11" fill="#c1502e">SAVED</text>
    </g>
    <text x="600" y="32" font-size="13" fill="#c1502e" font-weight="700">Memory: 5N</text>
  </g>

  <!-- Bottom row: custom Function with recompute -->
  <text x="20" y="170" font-size="12" font-weight="700" fill="#1f5f5b">Custom Function: save just x, recompute in backward</text>

  <g transform="translate(20, 185)">
    <g font-family="'IBM Plex Mono', monospace" font-size="13" text-anchor="middle">
      <rect x="0"   y="0" width="100" height="48" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/><text x="50"  y="20" fill="#1a1612">x</text><text x="50" y="38" font-size="11" fill="#1f5f5b">SAVED</text>
      <rect x="120" y="0" width="100" height="48" fill="#fff8a8" stroke="#8c6512" stroke-width="2" stroke-dasharray="4 3"/><text x="170" y="20" fill="#1a1612">a (transient)</text><text x="170" y="38" font-size="11" fill="#8c6512">freed</text>
      <rect x="240" y="0" width="100" height="48" fill="#fff8a8" stroke="#8c6512" stroke-width="2" stroke-dasharray="4 3"/><text x="290" y="20" fill="#1a1612">b (transient)</text><text x="290" y="38" font-size="11" fill="#8c6512">freed</text>
      <rect x="360" y="0" width="100" height="48" fill="#fff8a8" stroke="#8c6512" stroke-width="2" stroke-dasharray="4 3"/><text x="410" y="20" fill="#1a1612">c (transient)</text><text x="410" y="38" font-size="11" fill="#8c6512">freed</text>
      <rect x="480" y="0" width="100" height="48" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/><text x="530" y="20" fill="#1a1612">y (output)</text><text x="530" y="38" font-size="11" fill="#1f5f5b">kept</text>
    </g>
    <text x="600" y="32" font-size="13" fill="#1f5f5b" font-weight="700">Memory: 2N</text>
  </g>

  <text x="50" y="290" font-family="'Caveat', cursive" font-size="20" fill="#c1502e">backward recomputes a, b, c from x — costs ~2× compute, saves ~3× memory</text>
</svg>
</div>

<p>This is the trade you're constantly making in custom autograd: <strong>save more = use more memory but skip recomputation</strong>; <strong>save less = use less memory but recompute in backward</strong>. The right answer depends on whether you're memory-bound or compute-bound. For large models, you're almost always memory-bound; saving less wins.</p>

<h2>Code Magnets: build a custom GELU with cheap backward</h2>

<p>You're implementing GELU (a common activation) and want to be smart about saved memory. The forward is <code>y = x * sigmoid(1.702 * x)</code> (the "approximate" form). The backward needs <code>x</code> and the sigmoid value (call it <code>s</code>). Build a Function that's correct and memory-frugal.</p>

<div class="magnets">
<p>Arrange the magnets into a complete Function. Three are red herrings.</p>

<div class="magnet-pool">
  <span class="magnet">class FastGELU(torch.autograd.Function):</span>
  <span class="magnet">@staticmethod</span>
  <span class="magnet">def forward(ctx, x):</span>
  <span class="magnet">    s = torch.sigmoid(1.702 * x)</span>
  <span class="magnet">    ctx.save_for_backward(x, s)</span>
  <span class="magnet">    ctx.x = x</span>
  <span class="magnet">    return x * s</span>
  <span class="magnet">def backward(ctx, grad_y):</span>
  <span class="magnet">    x, s = ctx.saved_tensors</span>
  <span class="magnet">    grad_x = grad_y * (s + x * s * (1 - s) * 1.702)</span>
  <span class="magnet">    return grad_x</span>
  <span class="magnet">    return grad_y</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">class</span> <span class="ty">FastGELU</span>(torch.autograd.Function):
    <span class="kw">@staticmethod</span>
    <span class="kw">def</span> <span class="fn">forward</span>(ctx, x):
        s = torch.<span class="fn">sigmoid</span>(<span class="num">1.702</span> * x)
        ctx.<span class="fn">save_for_backward</span>(x, s)
        <span class="kw">return</span> x * s

    <span class="kw">@staticmethod</span>
    <span class="kw">def</span> <span class="fn">backward</span>(ctx, grad_y):
        x, s = ctx.saved_tensors
        grad_x = grad_y * (s + x * s * (<span class="num">1</span> - s) * <span class="num">1.702</span>)
        <span class="kw">return</span> grad_x</code></pre>
<p>The traps:</p>
<ul>
  <li><code>ctx.x = x</code> bypasses <code>save_for_backward</code> — works in this case but loses the version-counter safety.</li>
  <li><code>return grad_y</code> alone would just pass the gradient through (a wrong identity backward, like the STE pattern but applied to the wrong op).</li>
  <li>The second <code>@staticmethod</code> for backward is needed — both methods must be static.</li>
</ul>
<p>The math: derivative of <code>x * sigmoid(c*x)</code> is <code>sigmoid(c*x) + x * c * sigmoid(c*x) * (1 - sigmoid(c*x))</code> by product rule plus the sigmoid derivative. We multiply that local derivative by <code>grad_y</code>.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each tool/concept to its purpose.</p>

<div class="match-grid">
  <div class="header">Tool / concept</div>
  <div class="header">Purpose</div>

  <div>ctx.save_for_backward(t)</div>
  <div>A. Verifies your custom backward against finite differences.</div>

  <div>ctx.alpha = 0.5</div>
  <div>B. Saves a tensor with version-counter tracking for backward.</div>

  <div>MyOp.apply(x)</div>
  <div>C. Stashes non-tensor data on the context for backward to read.</div>

  <div>gradcheck(fn, inputs)</div>
  <div>D. The correct way to invoke a custom Function — hooks into autograd.</div>

  <div>Straight-through estimator</div>
  <div>E. Returning grad_y unchanged from a backward, regardless of forward.</div>

  <div>Recompute in backward</div>
  <div>F. The trick that powers gradient checkpointing and FlashAttention.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>ctx.save_for_backward(t)</strong> → B<br>
<strong>ctx.alpha = 0.5</strong> → C<br>
<strong>MyOp.apply(x)</strong> → D<br>
<strong>gradcheck(fn, inputs)</strong> → A<br>
<strong>Straight-through estimator</strong> → E<br>
<strong>Recompute in backward</strong> → F
</p>
<p>The mental shortcut: <em>save_for_backward is for tensors with version safety, ctx.attr is for plain Python data, .apply is the entry point, gradcheck is the verifier, STE lies through backward, recomputation trades compute for memory</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> Implement <code>Cube</code> as a custom Function: <code>y = x³</code>. Don't save anything via <code>save_for_backward</code> — instead, use the trick that <code>3x²</code> can be expressed in terms of the output as <code>3 * y^(2/3) * sign(y)</code>... wait, that's awful. Just save <code>x</code> and verify with gradcheck.</p>
<details class="answer"><summary>show answer</summary>
<pre><code><span class="kw">class</span> <span class="ty">Cube</span>(torch.autograd.Function):
    <span class="kw">@staticmethod</span>
    <span class="kw">def</span> <span class="fn">forward</span>(ctx, x):
        ctx.<span class="fn">save_for_backward</span>(x)
        <span class="kw">return</span> x ** <span class="num">3</span>

    <span class="kw">@staticmethod</span>
    <span class="kw">def</span> <span class="fn">backward</span>(ctx, grad_y):
        x, = ctx.saved_tensors
        <span class="kw">return</span> grad_y * <span class="num">3</span> * x ** <span class="num">2</span>

x = torch.<span class="fn">randn</span>(<span class="num">3</span>, dtype=torch.float64, requires_grad=<span class="kw">True</span>)
<span class="fn">assert</span> torch.autograd.<span class="fn">gradcheck</span>(Cube.apply, (x,))</code></pre>
<p>The exercise framing was a bit of a trick — yes, you could derive <code>x</code> from <code>y</code> (via cube root, careful with sign), but doing so is more expensive than just saving <code>x</code>. The "save vs recompute" tradeoff isn't always recompute-wins. Save <code>x</code>; move on.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Write an STE-style Function called <code>BinarizeSTE</code> that returns <code>sign(x)</code> in forward and the identity in backward. Verify gradcheck <em>doesn't</em> catch the lie (it's checking your declared backward against your declared forward via finite differences — and your declared backward says "identity," which gradcheck cannot tell is a lie).</p>
<details class="answer"><summary>show answer</summary>
<pre><code><span class="kw">class</span> <span class="ty">BinarizeSTE</span>(torch.autograd.Function):
    <span class="kw">@staticmethod</span>
    <span class="kw">def</span> <span class="fn">forward</span>(ctx, x):
        <span class="kw">return</span> torch.<span class="fn">sign</span>(x)

    <span class="kw">@staticmethod</span>
    <span class="kw">def</span> <span class="fn">backward</span>(ctx, grad_y):
        <span class="kw">return</span> grad_y       <span class="com"># the lie</span>

x = torch.<span class="fn">randn</span>(<span class="num">3</span>, dtype=torch.float64, requires_grad=<span class="kw">True</span>)
<span class="kw">try</span>:
    torch.autograd.<span class="fn">gradcheck</span>(BinarizeSTE.apply, (x,))
<span class="kw">except</span> Exception <span class="kw">as</span> e:
    <span class="fn">print</span>(<span class="str">"gradcheck failed (correctly)"</span>)</code></pre>
<p>gradcheck <em>does</em> fail here, because finite-difference of sign(x) is approximately zero, but you declared the gradient to be 1. STE backward is intentionally inconsistent with its forward — gradcheck is doing its job. In real STE code, you simply don't run gradcheck on it; you trust the empirical training results.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> Why does the LayerNorm backward we wrote sum the affine gradients (<code>grad_weight</code>, <code>grad_bias</code>) over <em>all leading dims</em>? Why not just dim=0?</p>
<details class="answer"><summary>show answer</summary>
<p>Because LayerNorm's input typically has shape <code>(B, T, D)</code> (batch, sequence, hidden), and <code>weight</code>/<code>bias</code> have shape <code>(D,)</code>. The forward broadcasts <code>weight</code> across both <code>B</code> and <code>T</code>. The chain rule then says: the gradient w.r.t. <code>weight</code> is the sum of contributions over <em>every</em> position it was broadcast to — both the batch dim and the sequence dim. <code>sum(dim=(0, 1))</code>, which is <code>sum(dim=tuple(range(grad_y.dim() - 1)))</code> in shape-agnostic form. If you only summed over dim=0, you'd get a <code>(T, D)</code> gradient that doesn't match <code>weight</code>'s shape and PyTorch would error.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> Suppose your custom Function's forward calls <code>x.contiguous()</code> internally. Should backward return <code>grad_x</code> with the original layout, or contiguous?</p>
<details class="answer"><summary>show answer</summary>
<p>Either works numerically, but PyTorch expects <code>grad_x</code> to have the same shape (and ideally strides) as the original input <code>x</code>. If you return a contiguous gradient when the input was non-contiguous, autograd will accept it, but downstream ops that expected the original layout may incur a hidden copy. The polite thing is to <em>match the input's memory layout</em>. In practice, most kernels return contiguous gradients and let downstream code handle it. The framework doesn't enforce strictness here.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>A <strong>custom <code>autograd.Function</code></strong> is a class with two static methods. <code>forward(ctx, *inputs)</code> computes outputs and saves what backward needs. <code>backward(ctx, *grad_outputs)</code> returns one gradient per forward input.</li>
  <li>Use <code>MyOp.apply(...)</code>, never <code>MyOp(...)</code> — <code>.apply</code> is what hooks into autograd.</li>
  <li><strong>Save tensors</strong> via <code>ctx.save_for_backward(...)</code>; retrieve via <code>ctx.saved_tensors</code>. <strong>Save non-tensor data</strong> as plain attributes on <code>ctx</code> (e.g., <code>ctx.eps = 1e-5</code>).</li>
  <li><code>gradcheck</code> verifies your analytical backward against finite differences. Use <strong>fp64</strong>, <strong>small tensors</strong>, <strong>multiple seeds</strong>. <code>gradgradcheck</code> for double-backward consistency.</li>
  <li>The <strong>four reasons to write a custom Function</strong>: (1) memory savings via recompute, (2) integrating a custom CUDA/Triton kernel, (3) non-differentiable ops with a fake gradient (STE), (4) a smarter algorithm than autograd would generate (FlashAttention).</li>
  <li><strong>Save vs recompute</strong> is the central tradeoff. Save more = use more memory, skip recomputation. Save less = recompute in backward. Memory-bound workloads almost always want recompute.</li>
  <li>The <strong>straight-through estimator</strong> is the canonical "lie in backward to enable training through a non-differentiable op" pattern. Foundation of QAT, VQ-VAEs, discrete bottlenecks.</li>
  <li>Custom backward must handle <strong>broadcasting in reverse</strong>: if forward broadcast a small tensor over leading dims, the gradient must be summed back over those dims to recover the small shape.</li>
  <li><strong>Watch for in-place mutations</strong> of saved tensors. The version counter will catch you. Save a <code>.clone()</code> if you must.</li>
  <li>The reflex: when you reach for a custom Function, ask <em>which of the four reasons</em> applies. If none, you probably don't need one — pure-PyTorch will be simpler and just as fast.</li>
</ul>
</div>

<p>That closes Part II. You now have the storage/stride model, the shape rules, the dtype/device model, the autograd graph model, the practical autograd toolbox, and the ability to write your own backward. <strong>Every kernel-writing module from M22 onwards stands on this foundation</strong> — when we get to FlashAttention in M25, the "custom backward that recomputes the forward" pattern will not be a surprise.</p>

<p>Part III opens with Module 07: <code>nn.Module</code> as a container system. Parameters vs buffers, hooks at the module level, the registration gotcha that orphans your weights, and the <code>state_dict</code> lifecycle that powers checkpointing.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">06</span>
  <span>Custom autograd: torch.autograd.Function</span>
</div>
"""

emit("06_custom_autograd_function", "Module 06 — Custom autograd: torch.autograd.Function", BODY)
