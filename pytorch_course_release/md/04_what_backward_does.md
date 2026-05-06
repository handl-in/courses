# Module 04 — What .backward() actually does

# What `.backward()` _actually does_

_Part II · Module 04_

— the dynamic graph, leaf vs non-leaf, and reverse-mode autodiff walked by hand on a tiny example until it stops being magic

\--- 

Here is the exact moment most PyTorch users decide to stop asking questions: someone says "autograd computes gradients automatically using reverse-mode automatic differentiation," they nod, and they go back to copying training loops from blog posts. They never _understand_ what `loss.backward()` did. So when their gradient is `None`, or their custom layer's backward is wrong, or their memory blows up because of `retain_graph=True`, they have nowhere to stand.

That ends today. By the end of this module you will be able to take a small expression, write down its computational graph by hand, walk the backward pass by hand, and tell PyTorch exactly which gradient buffer should hold what value. Once that's concrete, every later autograd topic — custom Functions, double backward, hooks, gradient checkpointing — is just a refinement.

> **★ KEY IDEA**  
>  `.backward()` is just **the chain rule, applied to a graph that PyTorch built for you during the forward pass**. Every operation produces a tensor and silently records "how to undo me" in a `grad_fn`. `.backward()` walks that graph in reverse from the loss, multiplying gradients along the way, and dumps the final values into the `.grad` attribute of every _leaf_ tensor that asked for it. There is no other magic. 

## Meet the autograd cast

Five characters in this module. Three are returning faces; two are new.

∂

Autograd

"I'm finally on stage. Let me explain my job."

When you build a tensor expression, I follow along quietly, recording every operation and its inputs. When you call `.backward()` on something, I walk that recording _in reverse_ , applying the chain rule at each step. I produce a number for every leaf tensor: "if you nudge this value by ε, the loss will change by approximately (gradient × ε)." That's it. I'm a tape recorder with calculus.

G

Graph

"I'm rebuilt from scratch on every forward pass."

This is the famous _dynamic_ part of PyTorch. Other frameworks built me once and reused me. PyTorch builds me fresh every time — every forward pass creates a new me, with whatever Python control flow you wrote. `if`, `for`, recursion — they all work. The price is that I'm thrown away after each `backward()`. The benefit is that you can debug me with `print` statements like normal Python.

⌬

grad_fn

"I'm the 'how to undo me' note attached to every non-leaf tensor."

When you do `z = x + y`, I get attached to `z` as `z.grad_fn = <AddBackward0>`. I remember that I came from an addition, and I know that the gradient of an addition just passes through to both inputs unchanged. Every operation in PyTorch has its own grad_fn class — `MulBackward0`, `SumBackward0`, `MmBackward0`, hundreds of them. I'm how the graph stores its structure.

L

Leaf

"I'm a tensor at the root of the graph. I own a `.grad`."

If you created me directly with `torch.tensor(...)` or `torch.randn(...)` or `nn.Parameter(...)`, and I have `requires_grad=True`, then I'm a leaf. After `.backward()`, my `.grad` attribute holds the gradient of the loss with respect to me. The non-leaves _don't_ get a `.grad` by default — they're just intermediates the graph used.

∇

Grad

"I'm a tensor too. I have the same shape as my parent."

I live on a Leaf's `.grad` attribute. I'm initially `None`. After your first `.backward()`, I exist. After the second one, I _accumulate_ — yes, I add to whatever was there. That's the famous "you forgot to call `optimizer.zero_grad()`" bug. It's a feature, not a glitch — but you need to know about it.

## The forward pass builds the graph

Let's start small. Two scalars, one combination, one observation.
    
    
    import torch
    
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    
    a = x + y          # a = 5.0
    z = a * y          # z = 15.0
    
    print(x.is_leaf, x.grad_fn)   # True None
    print(y.is_leaf, y.grad_fn)   # True None
    print(a.is_leaf, a.grad_fn)   # False <AddBackward0 object at ...>
    print(z.is_leaf, z.grad_fn)   # False <MulBackward0 object at ...>

Every line of code above did _two_ things: it computed a value, and it recorded a graph node. The recording is what makes the next steps work.

Here is the graph PyTorch built, in your head:

The graph after z = (x + y) * y x = 2.0 leaf, requires_grad y = 3.0 leaf, requires_grad a = x + y grad_fn: AddBackward0 z = a * y grad_fn: MulBackward0 y is used twice! (once in a, once in z) forward direction →

Two things to notice in that diagram. **First** : `x` and `y` are leaves (the green circles), `a` and `z` are non-leaves (the orange rectangles). Leaves are the ones you created directly; non-leaves are intermediate computations. **Second** : `y` is used twice — once to make `a`, once to make `z`. The graph remembers both edges. This will matter in 30 seconds.

G

Graph

"By the way — I only contain _tensors that need gradients_."

If neither `x` nor `y` had `requires_grad=True`, I wouldn't have been built at all. PyTorch tracks me lazily. The moment _one_ input has `requires_grad`, the output gets a `grad_fn` and joins me. Otherwise the operation runs as plain math, no recording. That's why `torch.no_grad()` is a speed win for inference — no graph means no overhead.

## The backward pass walked by hand

Now the punch line. We want `dz/dx` and `dz/dy`. Let's compute them by hand _before_ letting PyTorch do it, so we can check.

The function is `z = (x + y) * y`. Expanding: `z = xy + y²`. Calculus class:

  * `∂z/∂x = y` — at `x=2, y=3`, this is **3**.
  * `∂z/∂y = x + 2y` — at `x=2, y=3`, this is **2 + 6 = 8**.

Hold those numbers in your head: gradient on x should be 3, on y should be 8. Now let's see how the _chain rule_ walks the graph backwards to produce the same answers.

### The chain rule, step by step

Backward starts at the output, with seed gradient `1` (because `dz/dz = 1`). Then we walk each grad_fn in reverse, multiplying gradients as we go.

Backward pass: walk in reverse, multiply gradients z (output) seed grad: 1 ① At MulBackward (z = a * y): ∂z/∂a = y = 3 | ∂z/∂y = a = 5 upstream grad × local grad = 1·3 = 3 (to a), 1·5 = 5 (to y) a (intermediate) grad: 3 y first: +5 3 5 ② At AddBackward (a = x + y): ∂a/∂x = 1 | ∂a/∂y = 1 (addition just passes the gradient through) upstream grad (3) × local grad (1) = 3 (to x), 3·1 = 3 (to y, 2nd contribution) x grad: 3 y 2nd: +3 3 3 y gets BOTH: 5 + 3 = 8 ✓

Walk through it slowly:

  1. **Start at z** , with seed gradient 1 (we always start from `dz/dz = 1`).
  2. **At MulBackward** (the multiplication that made `z = a * y`): the local derivatives are `∂z/∂a = y = 3` and `∂z/∂y = a = 5`. We multiply each by the upstream gradient (1) and pass: `3` to `a`, `5` to `y`.
  3. **At AddBackward** (the addition that made `a = x + y`): both local derivatives are 1 (addition passes the gradient through unchanged). We multiply by the upstream gradient (3): `3` to `x`, `3` to `y`.
  4. **x** is a leaf — its `.grad` becomes `3`. ✓
  5. **y** is a leaf — its `.grad` accumulates both contributions: `5 + 3 = 8`. ✓ (This is why "y used twice" mattered.)

Now the punch line: do it in PyTorch and confirm.
    
    
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    z = (x + y) * y
    
    z.backward()
    
    print(x.grad)    # tensor(3.) ✓
    print(y.grad)    # tensor(8.) ✓

Same numbers. _That's all backward did_. It walked the graph, multiplied at each grad_fn, and added contributions when a leaf had multiple paths.

> **🧠 BRAIN POWER**  
> 
> 
> **Pause and predict.** Suppose you call `z.backward()` a second time on the same graph. What happens?
> 
> It errors. By default, PyTorch frees the graph after backward to save memory. To run backward again on the same graph, you'd need `z.backward(retain_graph=True)`.
> 
> And if you ran _forward_ again to rebuild the graph, then called backward? `x.grad` would become `6` (3 + 3) and `y.grad` would become `16` (8 + 8). Gradients **accumulate**. This is the "you forgot to call `optimizer.zero_grad()`" bug everyone meets once.

## Why it accumulates (and why it's a feature)

The accumulation behavior surprises beginners. Why doesn't PyTorch _overwrite_ `.grad` on each backward? Because in real training, you sometimes _want_ to accumulate. Two examples you'll meet later:

  1. **Gradient accumulation across micro-batches.** If your batch is too big to fit in GPU memory, you split it into N micro-batches, run forward+backward on each, and let `.grad` accumulate the sum. After N iterations, you call `optimizer.step()`. This is how you simulate large batch training on small GPUs (Module 11).
  2. **Multiple loss terms.** If your loss is `l1 + l2` and you computed them separately, you can call `l1.backward()` then `l2.backward()`. The two contributions add into `.grad` automatically.

So accumulation is the default, and you opt out by zeroing the grads explicitly:
    
    
    optimizer.zero_grad()       # sets all param.grad to None (or 0, older API)
    loss.backward()
    optimizer.step()

That three-line dance is the heartbeat of every training loop in PyTorch. Now you know what each line is doing to the graph.

> **⚠ WARNING**  
>  **Common foot-gun.** If you forget `zero_grad()`, your model still trains — but with effectively-larger gradients. The optimizer takes bigger steps, training becomes unstable, and you spend an afternoon hunting a bug that's _visible in the loss curve but hard to identify_. The fix is one line. Add it to your training loop template once and never think about it again. 

## Leaf vs non-leaf, in detail

One of the most confusing parts of autograd is which tensors get `.grad` and which don't. The rule is simple but easy to violate.

Who's a leaf? Tensor| Leaf?| Gets `.grad`?  
---|---|---  
`x = torch.tensor(2.0, requires_grad=True)`| Yes| Yes (after backward)  
`x = torch.randn(3, requires_grad=True)`| Yes| Yes  
`x = nn.Parameter(torch.randn(3))`| Yes| Yes  
`x = some_tensor.detach()`| Yes| Only if `requires_grad=True` set after  
`z = x + y` (where x, y require grad)| **No**|  No (intermediate)  
`z = x.cuda()` (where x requires grad)| **No**|  No — `cuda()` is an op  
`z = x.float()` (where x requires grad)| **No**|  No — same reason  
  
That last batch of rows surprises everyone. If you do:
    
    
    x = torch.randn(3, 3, requires_grad=True)
    x = x.cuda()                       # reassign — original is now a non-leaf!
    loss = x.sum()
    loss.backward()
    print(x.grad)                       # None! 😱

The `.cuda()` call _created a new tensor_ with a grad_fn (`CopyBackwards`). The new tensor is non-leaf. The original leaf is now unreferenced and never receives a grad. The fix is to construct the tensor on the right device from the start, or use `nn.Parameter`:
    
    
    # Right way
    x = torch.randn(3, 3, device='cuda', requires_grad=True)
    
    # Or, for module parameters
    self.weight = nn.Parameter(torch.randn(3, 3, device='cuda'))

#### Q&A; — About leaves and grads **Q:** Can I get the gradient of an intermediate (non-leaf) tensor? **A:** Yes — call `.retain_grad()` on it before backward. Without that, PyTorch discards intermediate gradients to save memory. `z.retain_grad(); loss.backward(); print(z.grad)`. You'll see this trick in debugging and in implementations of things like Grad-CAM. **Q:** Why does `requires_grad` propagate forward through ops? **A:** Because if _any_ input to an op requires gradients, the output must also be tracked — otherwise you couldn't backprop through it. PyTorch sets `requires_grad=True` on the result automatically. The reverse is also true: if no inputs require grads, the result doesn't either, and no graph is built. **Q:** What's the difference between `requires_grad=False` and `detach()`? **A:** Subtle but important. `requires_grad=False` on a leaf means "I'm not a parameter; don't track me as one." `detach()` creates a _new_ tensor that shares storage but is cut from the graph — its `grad_fn` is None, so backward stops at it. You use `detach()` to deliberately break gradient flow, e.g., to stop gradients from flowing into a frozen part of a model, or to use a quantity as a "constant" in a loss. **Q:** Can I have a tensor with `requires_grad=True` but no `.grad` after backward? **A:** Yes — if it's a leaf but the graph never reached it. For example, if your loss doesn't depend on a particular parameter (think: a layer that's effectively dead because of zero weights), backward never sends a gradient there. `.grad` stays `None`. This is also why `find_unused_parameters=True` matters in DDP (Module 16). 

## The graph for vector ops, briefly

Everything we just walked through generalizes from scalars to tensors. The chain rule still applies, but at each grad_fn the "local derivative" is now a Jacobian (a matrix or higher-rank tensor of partial derivatives). PyTorch never materializes the full Jacobian — that would be insane for a 1024×1024 weight matrix. Instead, each grad_fn implements a _vector-Jacobian product_ : given the upstream gradient, it computes "what gradient would have produced this through me?" without ever building the full Jacobian.

For `z = a * y` with vector tensors:

  * Local "derivative" of `z` w.r.t. `a` is element-wise `y`.
  * Backward at this node: `grad_a = upstream_grad * y` (element-wise multiply, same shape).
  * Same idea for `grad_y = upstream_grad * a`.

For matmul `z = A @ B`:

  * `grad_A = upstream_grad @ B.T`
  * `grad_B = A.T @ upstream_grad`

You don't have to memorize these. PyTorch knows them. But internalizing that _each grad_fn is just a function from upstream grad to downstream grad_ is what lets you write your own (Module 6) and reason about memory cost (Module 12).

## Code Magnets: build the simplest possible loss step

You're writing the most basic gradient descent step imaginable — one parameter, one loss, one update. Arrange the magnets into a working snippet that does _three_ steps of gradient descent on `w`, with the goal of minimizing `(w - 5)²`.

Use exactly the magnets needed. There are two red herrings.

w = torch.tensor(0.0, requires_grad=True) w = torch.tensor(0.0) for _ in range(3): loss = (w - 5) ** 2 loss.backward() w.data -= 0.1 * w.grad w.grad = None w -= 0.1 * w.grad

show solution
    
    
    w = torch.tensor(0.0, requires_grad=True)
    for _ in range(3):
        loss = (w - 5) ** 2
        loss.backward()
        w.data -= 0.1 * w.grad
        w.grad = None

The two traps:

  * `w = torch.tensor(0.0)` — without `requires_grad=True`, no graph is built and `w.grad` is always None.
  * `w -= 0.1 * w.grad` — this looks identical to `w.data -= ...` but it tries to do the update _through autograd_ , which complains about an in-place op on a leaf that requires grad. The `.data` bypass is the lazy fix; the proper modern fix is `with torch.no_grad(): w -= ...`. We'll meet both in Module 5.

The `w.grad = None` at the end of each iteration is what `optimizer.zero_grad(set_to_none=True)` does internally — it's slightly more efficient than zeroing because the next backward can just allocate fresh.

## Who does what?

Match each autograd concept to what it actually is.

Concept

What it is

requires_grad

A. The function attached to a non-leaf tensor that knows how to compute its backward.

grad_fn

B. A flag that tells PyTorch to track operations on this tensor for backward.

is_leaf

C. The accumulated gradient of the loss with respect to a leaf tensor.

.grad

D. True if the tensor was created directly (not produced by an op) — only leaves get .grad by default.

.backward()

E. Returns a new tensor that shares storage but is cut from the graph (grad_fn = None).

.detach()

F. Walks the graph in reverse from this tensor, populating .grad on every reachable leaf.

show solution

**requires_grad** → B  
**grad_fn** → A  
**is_leaf** → D  
**.grad** → C  
**.backward()** → F  
**.detach()** → E 

The mental shortcut: _requires_grad_ is the input switch, _grad_fn_ is the recorded operation, _is_leaf_ tells you if a tensor will receive grad, _.grad_ is the result, _.backward()_ is the action, _.detach()_ is the cut.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** Write a function `grad(f, x)` that takes a Python function `f` from a scalar to a scalar and a Python number `x`, and returns the derivative at `x` using PyTorch autograd.

show answer
    
    
    def grad(f, x):
        t = torch.tensor(float(x), requires_grad=True)
        f(t).backward()
        return t.grad.item()
    
    # Use:
    grad(lambda x: x ** 3, 2.0)   # 12.0  (3x² at x=2)

This is essentially what `torch.func.grad` does in modern PyTorch (we'll meet that API in Module 5). Writing it yourself once cements the model.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Compute by hand: for `w = torch.tensor([2.0, 3.0], requires_grad=True)` and `loss = (w * w).sum()`, what should `w.grad` be?

show answer

`loss = w[0]² + w[1]² = 4 + 9 = 13`. Derivatives: `∂loss/∂w[0] = 2w[0] = 4`, `∂loss/∂w[1] = 2w[1] = 6`. So `w.grad` should be `tensor([4., 6.])`. Verify in PyTorch.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** What's wrong with this code? Why does `x.grad` end up as `None`? 
    
    
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    x = x.to('cuda')
    loss = (x ** 2).sum()
    loss.backward()
    print(x.grad)        # None

show answer

The `x = x.to('cuda')` reassigned `x` to a non-leaf tensor (the result of a copy op). The _original_ leaf is now garbage-collected without ever having received a grad, and the new `x` is a non-leaf that doesn't get `.grad` populated by default. The fix: `x = torch.tensor([1.0, 2.0], device='cuda', requires_grad=True)`.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** If you call `loss.backward()` twice in a row without re-running the forward pass, what error do you get and why?

show answer

You get: _"Trying to backward through the graph a second time, but the buffers have already been freed."_ By default, PyTorch frees the saved tensors needed for backward as soon as one backward completes. The graph remains in memory (the grad_fn nodes), but the saved data inside them is gone. To do backward twice, pass `retain_graph=True` on the first call. This is rare in practice — usually you re-run forward.

### What just happened?

  * **The forward pass builds a graph.** Every operation produces a tensor and attaches a `grad_fn` recording how to undo itself.
  * **Leaves vs non-leaves.** Tensors you create directly (with `requires_grad=True`) are leaves and own a `.grad`. Tensors produced by ops are non-leaves and don't get `.grad` by default — call `.retain_grad()` if you need it.
  * **The graph is dynamic** — built fresh on every forward pass. Python control flow works. The graph is freed after backward unless you ask it to stay.
  * **Backward is the chain rule on the graph.** Start at the output with seed gradient 1, walk in reverse, multiply at each grad_fn, accumulate at every leaf with multiple paths. That's the entire algorithm.
  * **Each grad_fn implements a vector-Jacobian product** , not a full Jacobian. This is why backward is fast — no giant matrix is ever materialized.
  * **Gradients accumulate in`.grad`**. This is a feature (gradient accumulation, multiple loss terms) but the source of the "forgot zero_grad()" bug. `optimizer.zero_grad()` at the start of each step.
  * **The reassignment trap** : `x = x.cuda()` or `x = x.float()` creates a non-leaf and orphans the original leaf. Construct tensors on the right device from the start.
  * `requires_grad` propagates: if any input to an op requires grad, the output does too. `torch.no_grad()` disables this for inference.
  * `.detach()` cuts a tensor from the graph (sets its grad_fn to None). Use it to stop gradient flow deliberately.
  * The reflex: when something in autograd surprises you, draw the graph by hand. Find the leaves. Trace where the gradient flows. The answer is in the picture.

Module 05 takes the autograd model you just built and turns it into practical reflexes: when to use `requires_grad` vs `detach` vs `no_grad` vs `inference_mode`, how to debug "gradient is None," what hooks let you peek inside the graph, and why `torch.func` exists as a parallel API.
