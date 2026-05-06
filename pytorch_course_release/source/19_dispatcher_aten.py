#!/usr/bin/env python3
"""Module 19: Dispatcher, ATen, and storage internals — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part VII · Module 19</div>
  <h1 class="module-title">Dispatcher, ATen &amp; <em>storage internals</em></h1>
  <p class="module-sub">— what really happens between <code>torch.add(x, y)</code> in Python and the CUDA kernel that runs, walked through the dispatcher's layered tower</p>
</div>

<p>You've used <code>torch.add</code>, <code>x.relu_()</code>, <code>F.cross_entropy</code>, and a hundred other ops for eighteen modules without ever asking what happens after Python returns control to PyTorch. This module lifts the hood. Not because you'll edit C++ tomorrow, but because every kernel we'll write in Parts VIII has to plug into this machinery — and the conceptual model is much simpler than its reputation.</p>

<p>The whole story has three layers. Python calls into a thin C++ wrapper. The <em>dispatcher</em> looks at the tensor's properties (its device, whether it requires gradients, whether it's traced/compiled, etc.) and picks the right concrete implementation. The <em>ATen</em> library is where those concrete implementations live — kernels for CPU, CUDA, MPS, and so on. Once we understand this routing, registering a custom kernel (Module 23) is just "add another entry to ATen's table."</p>

<div class="keyidea">
The PyTorch dispatcher is a <strong>layered tower</strong> of dispatch keys. When you call an op, the dispatcher walks the tower from the top, peeling off one key per layer. The <em>Autograd</em> layer wraps the call so backward gets recorded. The <em>AMP</em> layer casts dtypes if you're inside autocast. The <em>Python</em> layer routes to a Python-defined override if there is one. Eventually you hit a <em>backend</em> key (CPU, CUDA, MPS) where a real kernel runs. The whole detour is a few microseconds, dispatched through a small lookup. <strong>Custom kernels register at the backend layer; tools like autocast and torch.compile register at higher layers.</strong>
</div>

<h2>One new face</h2>

<div class="character" style="--c: #133e3b;">
  <div class="avatar" style="background: #133e3b; color: #fff;">▶</div>
  <div>
    <p class="who">Dispatcher</p>
    <p class="name">"I'm the air-traffic controller. Every tensor op asks me 'where do I go?' and I route based on a stack of keys."</p>
    <p class="says">When Python calls <code>x + y</code>, it lands in C++ and asks me which kernel to run. I look at <code>x</code>'s and <code>y</code>'s metadata: their dispatch keys. <em>Autograd? AMP? CUDA?</em> I find the highest-priority key with an implementation registered for this op, and I run that. That implementation usually does its work and then re-dispatches at a lower key — peeling off layers. Eventually we hit a backend (CPU, CUDA) and a real kernel runs. Then the chain unwinds. Total overhead: a few microseconds. Total magic: zero.</p>
  </div>
</div>

<h2>The journey of <code>torch.add(x, y)</code></h2>

<p>Let's trace one call all the way down. <code>x</code> and <code>y</code> are both float32 CUDA tensors with <code>requires_grad=True</code>, and we're inside an <code>autocast(dtype=bfloat16)</code> block.</p>

<pre><code><span class="kw">with</span> torch.<span class="fn">autocast</span>(<span class="str">'cuda'</span>, dtype=torch.bfloat16):
    z = torch.<span class="fn">add</span>(x, y)         <span class="com"># what actually happens?</span></code></pre>

<p>This Python line triggers a chain. Each layer in the dispatcher tower does its own thing, optionally re-dispatching to the next layer, and the final result bubbles back up.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 520" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrD" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
    <marker id="arrU" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 10 0 L 0 5 L 10 10 z" fill="#c1502e"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">torch.add(x, y) descends the dispatcher tower</text>

  <!-- Python entry -->
  <g transform="translate(280, 50)">
    <rect x="0" y="0" width="180" height="40" fill="#fff8a8" stroke="#1a1612" stroke-width="2"/>
    <text x="90" y="18" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">Python: torch.add(x, y)</text>
    <text x="90" y="33" text-anchor="middle" font-size="10" fill="#1a1612">enters C++ via pybind</text>
  </g>

  <!-- Down arrow + Layer 1: AutogradCUDA -->
  <path d="M 370 95 L 370 115" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrD)"/>
  <g transform="translate(140, 120)">
    <rect x="0" y="0" width="460" height="50" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="230" y="20" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">① Autograd dispatch key</text>
    <text x="230" y="38" text-anchor="middle" font-size="10" fill="#1a1612">wraps the op in a grad_fn (M4); records inputs; re-dispatches → lower key</text>
  </g>

  <!-- Layer 2: AMP (autocast) -->
  <path d="M 370 175 L 370 195" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrD)"/>
  <g transform="translate(140, 200)">
    <rect x="0" y="0" width="460" height="50" fill="#fff5d8" stroke="#d4a017" stroke-width="2"/>
    <text x="230" y="20" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">② AMP / Autocast key</text>
    <text x="230" y="38" text-anchor="middle" font-size="10" fill="#1a1612">checks autocast policy (M14); add is "promote" — keeps fp32; re-dispatches</text>
  </g>

  <!-- Layer 3: ADInplaceOrView, Python, Functorch... abbreviated -->
  <path d="M 370 255 L 370 275" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrD)"/>
  <g transform="translate(140, 280)">
    <rect x="0" y="0" width="460" height="50" fill="#ede2cc" stroke="#6b5d4f" stroke-width="2"/>
    <text x="230" y="20" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">③ Other keys (Functorch, Python, ADInplaceOrView…)</text>
    <text x="230" y="38" text-anchor="middle" font-size="10" fill="#1a1612">most pass through unless something hooks them; re-dispatches → backend</text>
  </g>

  <!-- Layer 4: CUDA backend — the actual kernel -->
  <path d="M 370 335 L 370 355" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrD)"/>
  <g transform="translate(140, 360)">
    <rect x="0" y="0" width="460" height="60" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2.5"/>
    <text x="230" y="22" text-anchor="middle" font-size="14" font-weight="700" fill="#1a1612">④ CUDA backend key</text>
    <text x="230" y="40" text-anchor="middle" font-size="11" fill="#1a1612">at::native::add_kernel_cuda — the real CUDA elementwise add</text>
    <text x="230" y="55" text-anchor="middle" font-size="11" fill="#1a1612" font-style="italic">launches a CUDA kernel; returns z</text>
  </g>

  <!-- Return arrows -->
  <path d="M 200 360 L 200 130" stroke="#c1502e" stroke-width="1.5" fill="none" marker-end="url(#arrU)" stroke-dasharray="3 3"/>
  <text x="180" y="245" text-anchor="end" font-size="10" fill="#c1502e" font-style="italic">return path</text>
  <text x="180" y="258" text-anchor="end" font-size="10" fill="#c1502e" font-style="italic">unwinds layers</text>

  <!-- Bottom annotation -->
  <text x="370" y="455" font-family="'Caveat', cursive" font-size="20" fill="#1a1612" text-anchor="middle">total overhead: a handful of microseconds (≈ 5-15 µs eager)</text>
  <text x="370" y="478" font-family="'Caveat', cursive" font-size="20" fill="#c1502e" text-anchor="middle">torch.compile collapses this whole tower into one fused C++ call (M21)</text>
  <text x="370" y="500" font-family="'Caveat', cursive" font-size="20" fill="#1f5f5b" text-anchor="middle">custom kernels register at layer ④ — same path, just different ATen entry</text>
</svg>
</div>

<p>Each layer's job in plain English:</p>

<ol>
  <li><strong>Autograd</strong> sees the inputs have <code>requires_grad=True</code> and wraps the op so backward will be recorded (M4). It re-dispatches with the autograd key removed — the lower layers don't need to think about gradients.</li>
  <li><strong>AMP</strong> checks the autocast policy for <code>add</code>. <code>add</code> is in the "promote" list (M14) — it doesn't down-cast its inputs to bf16; it leaves them in their original precision. Other ops like <code>matmul</code> would down-cast here.</li>
  <li><strong>Other keys</strong> — Functorch, Python, ADInplaceOrView, Tracer, etc. — usually pass through. They each exist to hook a specific feature (vmap, custom autograd, fx tracing). For a vanilla <code>add</code> none of them do anything.</li>
  <li><strong>CUDA backend</strong> finally calls <code>at::native::add_kernel_cuda</code>, which launches an elementwise CUDA kernel. The kernel writes <code>z = x + y</code> into the output tensor. Returns up the chain.</li>
</ol>

<p>Back up: layer 1 captures the result and registers the backward function. Python gets a tensor <code>z</code> with <code>grad_fn=AddBackward0</code>. Total wall-clock cost in eager mode: a few microseconds. <strong>Most of that is dispatcher overhead, not kernel time</strong> for tiny ops, which is why <code>torch.compile</code> (M21) flattens this whole tower into one fused C++ call.</p>

<h2>Dispatch keys, more concretely</h2>

<p>A "key" is just an identifier. Each tensor carries a <em>dispatch key set</em>: the union of keys that apply to it. The dispatcher uses these to decide what to run.</p>

<p>Some keys you'll hear about:</p>

<div class="table-wrap">
<table>
<caption>The dispatch keys that come up most</caption>
<thead><tr><th>Key</th><th>Set when…</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>Autograd</code></td><td>tensor has <code>requires_grad=True</code></td><td>Records the op for backward; runs the original op via re-dispatch</td></tr>
<tr><td><code>AutocastCUDA</code></td><td>inside <code>autocast('cuda', ...)</code></td><td>Casts inputs per the autocast policy; re-dispatches</td></tr>
<tr><td><code>ADInplaceOrView</code></td><td>tensor is being viewed or modified in-place</td><td>Manages version counters; integrity for autograd</td></tr>
<tr><td><code>Functorch</code></td><td>inside <code>vmap</code>, <code>grad</code>, <code>jacrev</code></td><td>Implements the functional transforms (M5)</td></tr>
<tr><td><code>Python</code></td><td>tensor is a subclass with overrides</td><td>Calls the user's Python implementation (e.g., <code>__torch_function__</code>)</td></tr>
<tr><td><code>CUDA</code> / <code>CPU</code> / <code>MPS</code></td><td>tensor lives on that device</td><td>Calls the actual backend kernel — bottom of the tower</td></tr>
<tr><td><code>Meta</code></td><td>tensor is a "fake" tensor with no storage</td><td>Computes shape/dtype only; no kernel runs (used by <code>torch.compile</code>)</td></tr>
</tbody>
</table>
</div>

<p>Keys are <em>ordered</em> by priority. Autograd is high (so it can wrap before backend kernels run). The backend keys are low (they're terminal — they run actual computation). The "interesting" middle keys (AMP, Functorch, Python) sit in between.</p>

<h3>The Meta key: an aside worth knowing</h3>

<p>The <code>Meta</code> backend is special. A meta tensor has shape and dtype but <em>no actual data</em>. When the dispatcher hits the Meta backend, it runs a "shape function" that just figures out what the output tensor's shape and dtype would be — without doing any computation.</p>

<p>This is invaluable for two things. (1) <code>torch.compile</code> traces a model on meta tensors first to learn shapes without running anything. (2) Building large models without allocating memory: <code>with torch.device('meta'): model = build_model()</code>. The model exists, has the right structure and parameter shapes, but no weights are allocated. Useful when the full model wouldn't fit during construction (you'd later <code>.to_empty(device='cuda')</code> to materialize on GPU and then load weights).</p>

<h2>ATen: where the kernels actually live</h2>

<p>"ATen" is the C++ tensor library inside PyTorch. Every <code>torch.add</code>, <code>torch.mm</code>, <code>F.relu</code> ultimately dispatches to a function in ATen. The op signatures are declared in <code>native_functions.yaml</code> (yes, a YAML file with thousands of entries). Each entry says "this op has these argument types, dispatches via these keys, and is implemented by these C++ functions."</p>

<p>A simplified entry looks like:</p>

<pre><code><span class="com"># native_functions.yaml (simplified)</span>
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -&gt; Tensor
  dispatch:
    CPU, CUDA: add_Tensor
    SparseCPU, SparseCUDA: add_sparse
    MkldnnCPU: mkldnn_add</code></pre>

<p>This declares: <code>add</code> takes two tensors and a scalar alpha, returns a tensor. For dense CPU/CUDA tensors, it dispatches to <code>add_Tensor</code>; sparse tensors get a different implementation; MKLDNN-flavored tensors get yet another.</p>

<p>The build system code-generates the C++ glue from this YAML. The actual kernel implementations live in C++ files alongside it (<code>aten/src/ATen/native/</code>). When you "register a custom op" (M23), you're effectively adding rows to this dispatch table — though through a Python-friendly API.</p>

<div class="ndq">
<h4>About the dispatcher and ATen</h4>

<p class="q">Why is the dispatcher overhead such a big deal?</p>
<p class="a">For one big op (a 2K×2K matmul), the dispatcher overhead is invisible — the kernel takes milliseconds, the dispatch takes microseconds. For thousands of tiny ops (a transformer with hundreds of element-wise ops per layer), it adds up. <code>torch.compile</code> (M21) skips the per-op dispatch by tracing once and generating fused kernels — eliminating the tower for the trace. This is why "CPU-bound" traces (M13) often improve dramatically under compile.</p>

<p class="q">Can I look at <code>native_functions.yaml</code> myself?</p>
<p class="a">Yes — it's in the PyTorch source tree at <code>aten/src/ATen/native/native_functions.yaml</code>. It's huge (~10k lines, many ops with many variants). When you wonder "what does this op actually call," reading the dispatch entry tells you. Look up <code>matmul</code>, <code>scaled_dot_product_attention</code>, <code>sum</code> to see the full pattern.</p>

<p class="q">What's "structured kernels"?</p>
<p class="a">A pattern in ATen for ops that have a clean two-phase shape: a <em>meta function</em> computes the output's shape and prepares the output tensor; an <em>impl function</em> writes data into it. Splitting them lets the framework pre-allocate the output once, run the meta on the meta backend (for compile/trace), and run the impl on real backends. Most modern ops in ATen are structured. You don't need to write structured kernels until you're contributing to PyTorch core, but knowing the pattern helps when you read the source.</p>

<p class="q">How does <code>torch.compile</code> interact with the dispatcher?</p>
<p class="a">Compile traces your model — usually on the Meta backend — building an FX graph (M21). It then converts that graph to optimized C++/Triton kernels via the Inductor backend. Each compiled fused region is registered as a single op that bypasses most of the dispatcher tower. The Autograd key still wraps it (so backward works), but AMP and other middle keys are absorbed into the trace. Net effect: the per-op dispatcher overhead vanishes for the compiled region.</p>

<p class="q">What's the difference between <code>torch.add</code> and <code>torch.ops.aten.add</code>?</p>
<p class="a">The first is the user-facing API; the second is the direct ATen operator handle, which gives you slightly less overhead and is what most internal code uses. They eventually call the same kernel. <code>torch.ops</code> is what you'll see in FX graphs and compile internals.</p>
</div>

<h2>Storage internals revisited</h2>

<p>From M1: a tensor is a header (shape, stride, dtype, offset) pointing to a Storage of bytes. The dispatcher tower we just walked operates on tensor metadata. Once we hit the backend kernel, the kernel reads <em>storage bytes</em> directly. Let's double-click on what that means.</p>

<h3>Empty tensors and the meta backend</h3>

<p>A <em>meta tensor</em> is a tensor with valid metadata but no Storage:</p>

<pre><code>x = torch.<span class="fn">ones</span>(<span class="num">3</span>, <span class="num">4</span>, device=<span class="str">'meta'</span>)
<span class="fn">print</span>(x.shape)               <span class="com"># torch.Size([3, 4])</span>
<span class="fn">print</span>(x.dtype)               <span class="com"># torch.float32</span>
<span class="fn">print</span>(x.device)              <span class="com"># meta</span>
<span class="fn">print</span>(x.<span class="fn">untyped_storage</span>())   <span class="com"># empty / no actual bytes</span>

y = x + x                      <span class="com"># works! shape inference runs.</span>
<span class="fn">print</span>(y.shape)               <span class="com"># torch.Size([3, 4])</span>
<span class="com"># But y has no actual data — it's the shape, not the values.</span></code></pre>

<p>This is what <code>torch.compile</code> uses internally during tracing. It also lets you build a 70B-param model on a single GPU without OOM:</p>

<pre><code><span class="kw">with</span> torch.<span class="fn">device</span>(<span class="str">'meta'</span>):
    model = <span class="fn">build_huge_model</span>()      <span class="com"># no allocation; 0 GB used</span>
<span class="fn">print</span>(<span class="fn">sum</span>(p.<span class="fn">numel</span>() <span class="kw">for</span> p <span class="kw">in</span> model.<span class="fn">parameters</span>()))   <span class="com"># 70_000_000_000</span>

<span class="com"># Now materialize on real device</span>
model = model.<span class="fn">to_empty</span>(device=<span class="str">'cuda'</span>)    <span class="com"># allocates uninitialized GPU memory</span>
<span class="fn">load_pretrained_weights_into</span>(model)   <span class="com"># load checkpoint shards</span></code></pre>

<p>This pattern is the foundation of how huge models get loaded without "OOM during model init." FSDP and friends use it. You can use it directly when building anything where construction-time memory matters.</p>

<h3>The "_out" variants and out-of-place vs in-place</h3>

<p>Many ops have three variants in ATen:</p>

<ul>
  <li><strong>Functional</strong>: <code>torch.add(x, y)</code> → returns a new tensor.</li>
  <li><strong>In-place</strong>: <code>x.add_(y)</code> → modifies x, returns x.</li>
  <li><strong>"_out" variant</strong>: <code>torch.add(x, y, out=z)</code> → writes the result into a pre-allocated <code>z</code>.</li>
</ul>

<p>The <code>_out</code> variant is what kernel writers care about. It separates "compute the result" from "allocate memory for it" — the framework can pre-allocate, the kernel just writes. This is the "structured kernels" pattern in action: shape inference + allocation happen in one place; the kernel implementation is decoupled.</p>

<pre><code>x = torch.<span class="fn">randn</span>(<span class="num">3</span>, <span class="num">4</span>, device=<span class="str">'cuda'</span>)
y = torch.<span class="fn">randn</span>(<span class="num">3</span>, <span class="num">4</span>, device=<span class="str">'cuda'</span>)
z = torch.<span class="fn">empty_like</span>(x)             <span class="com"># pre-allocate</span>
torch.<span class="fn">add</span>(x, y, out=z)              <span class="com"># write into z; no extra allocation</span></code></pre>

<p>For very tight loops, the <code>_out</code> variants save the cost of one tensor allocation per call. Used heavily in optimizer code and custom kernels. For most user code, the regular form is clearer.</p>

<h2>Looking up an op in the wild</h2>

<p>Suppose you want to know what <code>F.scaled_dot_product_attention</code> actually does. The path:</p>

<ol>
  <li>Find the Python function: <code>torch.nn.functional.scaled_dot_product_attention</code> in <code>torch/nn/functional.py</code>. It's mostly a thin wrapper that calls into <code>torch._C._scaled_dot_product_attention(...)</code>.</li>
  <li>Look up the op in <code>native_functions.yaml</code>: search for <code>scaled_dot_product_attention</code>. You'll find an entry with <code>dispatch:</code> entries listing several backends (efficient_attention, flash_attention, math, etc.).</li>
  <li>Each backend points to a C++ function. Find that function in <code>aten/src/ATen/native/transformers/</code>. The flash and efficient attention paths call into bundled CUDA kernels.</li>
</ol>

<p>That's the entire stack. From <code>F.scaled_dot_product_attention</code> in your training loop to a CUDA kernel: three jumps. <strong>Every PyTorch op follows this pattern.</strong></p>

<h2>Custom op registration (preview of M23)</h2>

<p>If you have a custom CUDA kernel and want to expose it as a first-class PyTorch op, you register it through ATen. The new (post-2.4) recommended API:</p>

<pre><code><span class="kw">import</span> torch

<span class="kw">@</span>torch.library.<span class="fn">custom_op</span>(<span class="str">"my_lib::fast_relu"</span>, mutates_args=())
<span class="kw">def</span> <span class="fn">fast_relu</span>(x: torch.Tensor) -&gt; torch.Tensor:
    <span class="com"># Pure Python fallback — runs on CPU/all backends</span>
    <span class="kw">return</span> torch.<span class="fn">where</span>(x &gt; <span class="num">0</span>, x, torch.<span class="fn">zeros_like</span>(x))

<span class="kw">@</span>fast_relu.<span class="fn">register_kernel</span>(<span class="str">"cuda"</span>)
<span class="kw">def</span> <span class="fn">fast_relu_cuda</span>(x: torch.Tensor) -&gt; torch.Tensor:
    <span class="com"># Call your custom CUDA kernel here (via cpp_extension or torchscript)</span>
    <span class="kw">return</span> my_extension.<span class="fn">fast_relu_cuda</span>(x)

<span class="kw">@</span>fast_relu.<span class="fn">register_fake</span>()
<span class="kw">def</span> <span class="fn">fast_relu_fake</span>(x):
    <span class="com"># Shape function — runs on Meta backend for compile/tracing</span>
    <span class="kw">return</span> torch.<span class="fn">empty_like</span>(x)

<span class="com"># Register a backward — so it works with autograd</span>
<span class="kw">def</span> <span class="fn">setup_context</span>(ctx, inputs, output):
    ctx.<span class="fn">save_for_backward</span>(inputs[<span class="num">0</span>])

<span class="kw">def</span> <span class="fn">backward</span>(ctx, grad):
    x, = ctx.saved_tensors
    <span class="kw">return</span> grad * (x &gt; <span class="num">0</span>).<span class="fn">to</span>(grad.dtype)

torch.library.<span class="fn">register_autograd</span>(
    <span class="str">"my_lib::fast_relu"</span>, backward, setup_context=setup_context,
)</code></pre>

<p>Once registered: <code>torch.ops.my_lib.fast_relu(x)</code> goes through the dispatcher tower exactly like <code>torch.add</code>. Autograd records it, autocast handles dtype if needed, the dispatcher routes to your CUDA kernel on CUDA tensors and to your Python fallback on CPU. <code>torch.compile</code> can trace it via the fake function. <strong>Custom ops aren't second-class — they slot into the same machinery as built-ins.</strong></p>

<p>We'll write a real CUDA kernel and wire it up in M23. For now, internalize that "registering a kernel" is "adding an entry to ATen's table that the dispatcher will find."</p>

<h2>One more dispatcher trick: <code>__torch_function__</code></h2>

<p>If you want to intercept <em>any</em> tensor op for a particular tensor subclass, you can implement <code>__torch_function__</code>:</p>

<pre><code><span class="kw">class</span> <span class="ty">LoggingTensor</span>(torch.Tensor):
    <span class="kw">@</span><span class="fn">classmethod</span>
    <span class="kw">def</span> <span class="fn">__torch_function__</span>(cls, func, types, args=(), kwargs=<span class="kw">None</span>):
        <span class="kw">if</span> kwargs <span class="kw">is</span> <span class="kw">None</span>: kwargs = {}
        <span class="fn">print</span>(<span class="fn">f"call: {func.__name__}"</span>)
        <span class="kw">return</span> <span class="fn">super</span>().<span class="fn">__torch_function__</span>(func, types, args, kwargs)

x = <span class="fn">LoggingTensor</span>(torch.<span class="fn">randn</span>(<span class="num">3</span>, <span class="num">4</span>))
y = x + x        <span class="com"># prints: call: add</span>
z = y.<span class="fn">sum</span>()      <span class="com"># prints: call: sum</span></code></pre>

<p>This hooks the dispatcher at the Python key. You see every op called on instances of your subclass before it descends to backend kernels. Useful for logging, validation, custom semantics, or building tensor-like abstractions (e.g., <code>DTensor</code> from M17 uses similar machinery internally). Most users never need this; it's worth knowing exists.</p>

<h2>The mental model, summarized</h2>

<p>When you call any tensor op:</p>

<ol>
  <li>Python lands in C++ via pybind11 wrappers.</li>
  <li>The dispatcher computes the dispatch key set from the inputs.</li>
  <li>It walks down the keys (highest priority first), running each one's implementation. Each implementation typically does its own thing (autograd records, autocast casts, etc.) and re-dispatches at a lower key.</li>
  <li>Eventually a backend key (CPU/CUDA/MPS/etc.) runs an actual kernel from ATen.</li>
  <li>The result bubbles back up; the higher-key implementations finalize their bookkeeping.</li>
</ol>

<p>That's the entire framework's runtime. Once you've got it, the rest of Part VII (the CUDA stream model, the caching allocator, <code>torch.compile</code>) and Part VIII (CUDA, Triton, kernel-level work) all hook into specific points of this tower.</p>

<h2>Code Magnets: register a custom op end-to-end</h2>

<p>You're registering a custom op <code>my_lib::scaled_relu(x, alpha)</code> = <code>alpha * relu(x)</code>. Need: Python fallback, CUDA kernel, fake function for compile, and autograd backward. Three magnets are red herrings.</p>

<div class="magnets">
<p>Arrange the magnets into a complete registration.</p>

<div class="magnet-pool">
  <span class="magnet">@torch.library.custom_op("my_lib::scaled_relu", mutates_args=())</span>
  <span class="magnet">def scaled_relu(x: torch.Tensor, alpha: float) -> torch.Tensor:</span>
  <span class="magnet">    return alpha * torch.relu(x)</span>
  <span class="magnet">@scaled_relu.register_kernel("cuda")</span>
  <span class="magnet">def scaled_relu_cuda(x, alpha):</span>
  <span class="magnet">    return my_ext.scaled_relu_cuda(x, alpha)</span>
  <span class="magnet">@scaled_relu.register_fake()</span>
  <span class="magnet">def scaled_relu_fake(x, alpha):</span>
  <span class="magnet">    return torch.empty_like(x)</span>
  <span class="magnet">def scaled_relu_kernel(x, alpha): return my_ext.fast(x, alpha)</span>
  <span class="magnet">scaled_relu = torch.compile(scaled_relu)</span>
  <span class="magnet">torch.ops.my_lib.scaled_relu = scaled_relu</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">@</span>torch.library.<span class="fn">custom_op</span>(<span class="str">"my_lib::scaled_relu"</span>, mutates_args=())
<span class="kw">def</span> <span class="fn">scaled_relu</span>(x: torch.Tensor, alpha: <span class="fn">float</span>) -&gt; torch.Tensor:
    <span class="kw">return</span> alpha * torch.<span class="fn">relu</span>(x)

<span class="kw">@</span>scaled_relu.<span class="fn">register_kernel</span>(<span class="str">"cuda"</span>)
<span class="kw">def</span> <span class="fn">scaled_relu_cuda</span>(x, alpha):
    <span class="kw">return</span> my_ext.<span class="fn">scaled_relu_cuda</span>(x, alpha)

<span class="kw">@</span>scaled_relu.<span class="fn">register_fake</span>()
<span class="kw">def</span> <span class="fn">scaled_relu_fake</span>(x, alpha):
    <span class="kw">return</span> torch.<span class="fn">empty_like</span>(x)</code></pre>
<p>The traps:</p>
<ul>
  <li><code>def scaled_relu_kernel(x, alpha): ...</code> alone — without the <code>@scaled_relu.register_kernel("cuda")</code> decorator, this is just a free function the framework never finds.</li>
  <li><code>scaled_relu = torch.compile(scaled_relu)</code> — wrapping the registered op in compile changes the object — the registration framework expects to track the original; compile would obscure that.</li>
  <li><code>torch.ops.my_lib.scaled_relu = scaled_relu</code> — manual assignment to the ops namespace. The decorator-based registration handles this for you; manually overwriting can break invariants.</li>
</ul>
<p>For full autograd support you'd also call <code>torch.library.register_autograd(...)</code> separately, with a <code>setup_context</code> and <code>backward</code>. Skipping that means the op runs in forward but errors in backward — fine for inference-only ops; mandatory for training.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each dispatcher / ATen concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>Dispatcher</div>
  <div>A. Routes a tensor op to the right implementation based on a stack of keys.</div>

  <div>Autograd dispatch key</div>
  <div>B. Wraps the op with backward-recording then re-dispatches without itself.</div>

  <div>Meta backend</div>
  <div>C. Computes shape/dtype only; no real allocation or computation.</div>

  <div>ATen</div>
  <div>D. The C++ tensor library; where every op's concrete kernels live.</div>

  <div>native_functions.yaml</div>
  <div>E. Declarative registry of every op's signature and per-backend implementations.</div>

  <div>Out variants (op_out)</div>
  <div>F. Pre-allocate the output; kernel writes into it without allocating.</div>

  <div>__torch_function__</div>
  <div>G. Subclass hook for intercepting any tensor op before it descends the dispatcher.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>Dispatcher</strong> → A<br>
<strong>Autograd dispatch key</strong> → B<br>
<strong>Meta backend</strong> → C<br>
<strong>ATen</strong> → D<br>
<strong>native_functions.yaml</strong> → E<br>
<strong>Out variants (op_out)</strong> → F<br>
<strong>__torch_function__</strong> → G
</p>
<p>The mental shortcut: <em>dispatcher routes by keys, autograd key wraps for backward, meta computes shapes only, ATen has the kernels, YAML declares signatures, _out separates compute from allocation, __torch_function__ is the subclass hook</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team is profiling a tiny model with thousands of small element-wise ops per step. They notice their step time is dominated by Python — the GPU is idle most of the time. Explain in dispatcher terms what's happening, and propose two fixes.</p>
<details class="answer"><summary>show answer</summary>
<p>For each tiny op, Python crosses into C++, descends the dispatcher tower (Autograd, AMP, Functorch, ADInplaceOrView, ..., CUDA), launches a kernel that takes microseconds, returns up the tower. The dispatcher overhead per op is ~5-15 µs. With thousands of ops per step, that's tens of milliseconds of pure overhead, with the GPU idle waiting for the next launch. This is the "CPU-bound" trace shape from M13.</p>
<p>Two fixes: (1) <strong><code>torch.compile</code></strong> traces the model once, fuses ops into larger kernels, eliminates per-op dispatch — typically 1.5-3× speedup on this pattern. (2) <strong>CUDA Graphs</strong> record a sequence of kernel launches once and replay the recording — eliminates per-launch overhead for static graphs. M20 covers CUDA graphs; M21 covers compile.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Build a 7B-param transformer on the meta device, then materialize it on CUDA without ever holding the full model on CPU. Sketch the code.</p>
<details class="answer"><summary>show answer</summary>
<pre><code><span class="kw">with</span> torch.<span class="fn">device</span>(<span class="str">'meta'</span>):
    model = <span class="fn">build_transformer</span>(n_params=<span class="num">7e9</span>)        <span class="com"># 0 GB used</span>

<span class="com"># Materialize on CUDA with empty allocations</span>
model = model.<span class="fn">to_empty</span>(device=<span class="str">'cuda'</span>)            <span class="com"># allocates GPU memory, weights are uninitialized</span>

<span class="com"># Now load pretrained weights, e.g., shard by shard</span>
sd = torch.<span class="fn">load</span>(<span class="str">'shard_00.pt'</span>, weights_only=<span class="kw">True</span>)
model.<span class="fn">load_state_dict</span>(sd, strict=<span class="kw">False</span>, assign=<span class="kw">True</span>)
<span class="com"># OR initialize fresh:</span>
<span class="kw">for</span> p <span class="kw">in</span> model.<span class="fn">parameters</span>():
    nn.init.<span class="fn">normal_</span>(p, mean=<span class="num">0</span>, std=<span class="num">0.02</span>)</code></pre>
<p>The <code>assign=True</code> flag in <code>load_state_dict</code> reuses the storage of the loaded tensors instead of copying them — important for huge models where you don't want to double-allocate. This pattern is the foundation of how FSDP and friends initialize at scale.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> Why does <code>torch.compile</code> care about the meta backend specifically?</p>
<details class="answer"><summary>show answer</summary>
<p>Compile needs to <em>trace</em> the model — build an FX graph of every op called — without running real computation. Real computation would force kernel launches, allocate memory, sync GPU, etc. — all expensive. The meta backend lets compile run the model symbolically: every op's shape function runs (so the FX graph has shape-correct nodes), but no actual kernel is launched and no memory is allocated. Compile then takes that FX graph and code-generates fused kernels via Inductor (M21). Without the meta backend, tracing huge models would either OOM or take forever.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> A teammate writes a custom autograd Function (M6) instead of using <code>torch.library.custom_op</code>. When does that matter?</p>
<details class="answer"><summary>show answer</summary>
<p>Both produce ops the framework can backprop through, but they integrate differently:</p>
<ul>
  <li><code>autograd.Function</code> (M6) creates a new <em>Python</em> autograd node. <code>torch.compile</code> can usually trace through it but treats it as opaque — it can't fuse the body of your forward with surrounding ops. Good for one-off custom backwards (FlashAttention-style memory tricks).</li>
  <li><code>torch.library.custom_op</code> registers the op in ATen's dispatch table. The op gets a fake implementation (for tracing), an autograd registration (for backward), and proper compile/distributed integration. Better for ops you want to expose as first-class — including custom CUDA kernels you want compile to fuse around.</li>
</ul>
<p>Rough rule: use <code>autograd.Function</code> for "I want a custom backward for these few lines"; use <code>torch.library.custom_op</code> for "I'm exposing a new operator with its own kernel."</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>Every PyTorch op enters C++ via pybind, then descends the <strong>dispatcher tower</strong>: a stack of dispatch keys, each with its own implementation that re-dispatches at a lower key.</li>
  <li>The keys, top to bottom: <strong>Autograd</strong> (records backward), <strong>AMP/Autocast</strong> (handles dtype policy), <strong>Functorch / Python / ADInplaceOrView</strong> (transform/subclass hooks), and <strong>backend</strong> keys (CPU, CUDA, MPS, Meta).</li>
  <li>Backend keys are <em>terminal</em>: they run a real kernel from <strong>ATen</strong>, the C++ tensor library.</li>
  <li><code>native_functions.yaml</code> declares every op's signature and per-backend dispatch. The build system code-gens C++ glue from it.</li>
  <li>The <strong>Meta backend</strong> runs shape functions only — no allocation, no computation. Used by <code>torch.compile</code> for tracing and by users for "build huge model without OOM" (<code>torch.device('meta')</code> + <code>to_empty</code>).</li>
  <li><code>op_out</code> variants separate "compute" from "allocate" — kernel writes into pre-allocated output. Foundation of "structured kernels."</li>
  <li><strong>Dispatcher overhead</strong> is ~5-15 µs per op in eager mode. Negligible for big ops; dominates for many tiny ops. <code>torch.compile</code> (M21) and CUDA Graphs (M20) are the answers.</li>
  <li>Custom ops register through <code>torch.library.custom_op</code>: define Python fallback, register backend kernel(s), register fake (shape function), register autograd backward. The op then participates fully in the dispatcher.</li>
  <li><code>__torch_function__</code> is the Python-level intercept for tensor subclasses — runs at the Python dispatch key, lets you observe or override every op.</li>
  <li><strong>Custom ops aren't second-class</strong> — they go through the same machinery as built-ins. The dispatcher doesn't care if a kernel was written by Anthropic or by you.</li>
  <li>The reflex: when you wonder "what does this op actually do," the path is Python → dispatcher → ATen → backend kernel. Each step is documented; <code>native_functions.yaml</code> is the source of truth.</li>
</ul>
</div>

<p>Module 20 zooms in on the CUDA backend specifically. The CUDA caching allocator (we touched on it in M12, now we look at how it works), CUDA streams and the async kernel queue, CUDA Graphs as a way to bypass dispatcher overhead, and the topology-aware allocations that show up in distributed training. The mental model from this module is the foundation; M20 fills in the GPU half.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">19</span>
  <span>Dispatcher, ATen &amp; storage internals</span>
</div>
"""

emit("19_dispatcher_aten", "Module 19 — Dispatcher, ATen & storage internals", BODY)
