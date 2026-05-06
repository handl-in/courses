# Module 06 — Custom autograd: torch.autograd.Function

# Custom autograd: _`torch.autograd.Function`_

_Part II · Module 06_

— writing your own forward and backward, the four reasons it earns its keep, and the trick that makes FlashAttention possible

\--- 

So far autograd has been a black box that does the right thing. You write the forward, PyTorch writes the backward, life is good. This module is about the moment you decide to _open the box_.

You will not do this often. PyTorch's automatic backward handles 99% of cases correctly and efficiently. But when you need the 1% — a fused kernel that runs in 1/4 the memory, a non-differentiable op that needs a "fake" gradient, a custom CUDA kernel you'll write in Module 23 — you need to know how. And the act of writing one custom backward by hand teaches you more about the framework than ten more modules of API tour.

> **★ KEY IDEA**  
>  A custom `autograd.Function` is **a class with two static methods** : `forward` computes the output, `backward` takes the upstream gradient and returns the downstream gradients. PyTorch wraps your class in a grad_fn and treats it just like any built-in op. There are **four real reasons to write one** : memory savings (recompute instead of save), fusing many ops into one (custom kernel integration), non-differentiable ops with custom gradients (like rounding), and replacing PyTorch's backward with a smarter one (like FlashAttention). 

## Meet the new face

𝓕

Function

"I'm what `grad_fn` is when _you_ make one."

When you subclass `torch.autograd.Function`, you're defining a new operation as far as Autograd is concerned. The `forward` method is the math; the `backward` method is the chain rule. PyTorch creates an instance of me each time you call your op — I become the `grad_fn` of the output tensor. From the outside, no one can tell I'm not a built-in op. From the inside, you have _complete control_ over what gets saved and what gets recomputed.

## Anatomy of a custom Function

The minimal skeleton:
    
    
    class MyOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *inputs):
            # Compute the output.
            # Save anything needed for backward via ctx.save_for_backward(...).
            return output
    
        @staticmethod
        def backward(ctx, *grad_outputs):
            # grad_outputs[i] is the upstream gradient for forward's i-th output.
            # Return one gradient per input to forward (in the same order).
            # Return None for inputs that don't need a gradient.
            return grad_input1, grad_input2, ...
    
    # To use it:
    y = MyOp.apply(x1, x2)         # note: .apply, not direct construction

Three things to internalize:

  1. **Both methods are`@staticmethod`.** PyTorch passes a fresh `ctx` object to each — that's where you stash anything backward needs to remember from forward.
  2. **You call`MyOp.apply(...)`**, not `MyOp(...)`. The `apply` method is what hooks into autograd, creating the grad_fn and tying it to the output.
  3. **Backward returns one gradient per _forward input_** , in the same order. If forward took `(x, y, alpha)`, backward returns `(grad_x, grad_y, grad_alpha)`. Inputs you don't differentiate (like a config flag) get `None`.

## The simplest possible example

Let's implement `square`. It's trivially correct and lets us focus on the API.
    
    
    class Square(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)        # save x because we need it in backward
            return x * x
    
        @staticmethod
        def backward(ctx, grad_out):
            x, = ctx.saved_tensors             # note the comma — it's a tuple
            # d(x²)/dx = 2x. Chain rule: dL/dx = dL/dy · dy/dx = grad_out · 2x
            return grad_out * 2 * x
    
    # Try it:
    x = torch.tensor([3.0, 4.0], requires_grad=True)
    y = Square.apply(x).sum()
    y.backward()
    print(x.grad)                            # tensor([6., 8.]) — that's 2x ✓

That's the entire pattern. `save_for_backward` stashes the tensor in `ctx`; `backward` retrieves it from `ctx.saved_tensors` and applies the chain rule by hand.

∂

Autograd

"`save_for_backward` isn't just stashing — it's a contract."

When you save a tensor, I track it specially. If anyone modifies it in place between forward and backward, I'll detect that with version counters and yell at you (the "version mismatch" error). I also know not to free its memory until backward is done. _Always_ save through `ctx.save_for_backward()`; never just stash on `ctx` as a plain attribute. The latter works for non-tensors but skips the safety net.

> **⚠ WARNING**  
>  **Save tensors via`save_for_backward`; save plain Python data as `ctx.attribute`.** Mixing these up is a common bug. `ctx.x = some_tensor` works — barely — but skips version tracking and confuses anyone reading your code. Use `ctx.save_for_backward(some_tensor)` for tensors, `ctx.alpha = 0.5` for hyperparameters. 

## A real example: LayerNorm by hand

Now let's implement something real. LayerNorm is a great target because (a) it's used everywhere, (b) its backward involves a non-trivial Jacobian, and (c) the official PyTorch version is more memory-efficient than the naive automatic backward — which is exactly the kind of win custom Functions are for.

The math, one more time. For input `x` of shape `(..., D)` and learnable `weight`, `bias` of shape `(D,)`:
    
    
    μ = mean(x, dim=-1)           # per-row mean
    σ² = var(x, dim=-1)           # per-row variance (biased)
    x̂ = (x - μ) / sqrt(σ² + ε)
    y = x̂ * weight + bias

The backward is uglier than this looks. Each output element `y[i, j]` depends on every `x[i, k]` through both the mean and the variance. The full derivation is two pages of tensor calculus; the result is short:
    
    
    class MyLayerNorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, eps=1e-5):
            # Normalize over the last dimension
            mean = x.mean(dim=-1, keepdim=True)              # (..., 1)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            rstd = 1.0 / (var + eps).sqrt()                # (..., 1)
            x_hat = (x - mean) * rstd                          # (..., D)
            y = x_hat * weight + bias                          # (..., D)
    
            # Save what we need for backward.
            # Note: we save x_hat and rstd, NOT x and var. Why? Smaller scratch space.
            # x_hat already encodes (x - mean) * rstd, which we'd recompute anyway.
            ctx.save_for_backward(x_hat, weight, rstd)
            ctx.eps = eps
            return y
    
        @staticmethod
        def backward(ctx, grad_y):
            x_hat, weight, rstd = ctx.saved_tensors
            D = x_hat.shape[-1]
    
            # Gradients to the affine params: easy
            # Sum over all leading dims (e.g., batch and sequence) so shapes match.
            leading_dims = tuple(range(grad_y.dim() - 1))
            grad_bias   = grad_y.sum(dim=leading_dims)                  # (D,)
            grad_weight = (grad_y * x_hat).sum(dim=leading_dims)        # (D,)
    
            # Gradient to x: this is the painful part.
            # dL/dx = (1/D) * rstd * (D · g - sum(g) - x_hat · sum(g · x_hat))
            # where g = grad_y * weight
            g = grad_y * weight                                            # (..., D)
            sum_g       = g.sum(dim=-1, keepdim=True)                 # (..., 1)
            sum_g_xhat  = (g * x_hat).sum(dim=-1, keepdim=True)        # (..., 1)
            grad_x = (rstd / D) * (D * g - sum_g - x_hat * sum_g_xhat)
    
            # Order matches forward: (x, weight, bias, eps) → (grad_x, grad_w, grad_b, None)
            return grad_x, grad_weight, grad_bias, None

A few things worth pointing out about that backward. First, `grad_bias` is just the upstream gradient summed across the batch — bias is added uniformly, so the gradient is the sum. Same for `grad_weight`, but multiplied by the normalized `x_hat`. Both follow naturally from the chain rule.

Second, the `grad_x` formula looks scary but reflects something simple: changing one element `x[i, k]` changes the mean and variance of row `i`, which in turn changes _every_ output in row `i`. The three terms — the local effect, the mean's contribution, the variance's contribution — are exactly that.

Third — and this is the point — we saved `x_hat` and `rstd`, not `x` and `var`. Same memory cost, but backward needs less computation because `x_hat` is what we'd need to recompute anyway. _This kind of micro-optimization is why hand-written custom backwards beat the autodiff version._

> **🧠 BRAIN POWER**  
> 
> 
> **Pause.** Why do we use `unbiased=False` when computing the variance?
> 
> Two reasons. (1) The traditional LayerNorm definition uses the population variance (divide by D, not D−1), so the formula matches papers and other implementations. (2) The backward formula derived above assumes `1/D`, not `1/(D-1)`. If you switched to unbiased, you'd need to redo the math. Always check that your definition and your derivative agree.

## Always run `gradcheck`

You wrote a backward by hand. Are you sure it's right? `torch.autograd.gradcheck` compares your analytical backward against numerical finite differences. It catches sign errors, missing terms, and the wrong reduction order in seconds.
    
    
    from torch.autograd import gradcheck
    
    # gradcheck expects fp64 inputs for stable finite differences
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    w = torch.randn(5,    dtype=torch.float64, requires_grad=True)
    b = torch.zeros(5,   dtype=torch.float64, requires_grad=True)
    
    assert gradcheck(MyLayerNorm.apply, (x, w, b), eps=1e-6)
    # If your backward is wrong, this raises with a clear message about which input.

Three rules of thumb for `gradcheck`:

  1. **Use fp64.** Finite differences need precision; fp32 is too noisy for reliable comparison.
  2. **Use small tensors.** gradcheck's cost is O(n_params × n_outputs). On a (32, 1024) tensor it'll take forever. Test on (2, 5) and trust the result.
  3. **Try a few random seeds.** A single seed can hit lucky cancellations. Run a few; if all pass, you're confident.

For double backward (Module 5's `create_graph=True` friend), there's a sibling: `gradgradcheck`. Run both if your op might appear in a Hessian computation.

## The four reasons to write one

You now know _how_ to write a Function. The harder question is _when_. Four real motivations.

### 1\. Memory savings: recompute instead of save

The autograd-generated backward for an op saves _everything it might need_. If your forward computes `y = sigmoid(x)`, autograd saves `y` (because the derivative of sigmoid is `y * (1 - y)`, expressible in terms of the output). If your forward computes a long chain like `y = c(b(a(x)))`, autograd saves the intermediate `a(x)`, `b(a(x))`, etc., for backward.

For deep networks, those saved activations dominate memory. Activation checkpointing (Module 12) is the systematic answer, but custom Functions are the targeted one. You write a Function whose forward computes the chain, saves _only the input_ , and whose backward _recomputes_ the intermediates locally. Trades compute for memory.
    
    
    class FusedChain(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)         # save ONLY x, not intermediates
            a = torch.sin(x)
            b = a * a
            c = b.exp()
            return c
    
        @staticmethod
        def backward(ctx, grad_c):
            x, = ctx.saved_tensors
            # Recompute the intermediates (cheap, on-device)
            a = torch.sin(x)
            b = a * a
            c = b.exp()
            # Now apply the chain rule
            grad_b = grad_c * c                  # dc/db = exp(b) = c
            grad_a = grad_b * 2 * a              # db/da = 2a
            grad_x = grad_a * torch.cos(x)        # da/dx = cos(x)
            return grad_x

Memory: 1 tensor (`x`) instead of 4 (`x, a, b, c`). Compute: roughly 2× because we recompute the chain. For long-running training where memory is the binding constraint, this trade is hugely worth it. **This is exactly what FlashAttention does at scale** , and we'll see it again in Module 25.

### 2\. Fusing many ops into one (custom kernel integration)

When you write a custom CUDA or Triton kernel (Modules 23-24), you need a way to plug it into PyTorch's autograd. `autograd.Function` is exactly that bridge:
    
    
    class MyFusedKernel(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w):
            # Call your hand-written CUDA kernel here. Could be Triton, raw CUDA,
            # or a C++ extension. The Python side just orchestrates.
            out = my_cuda_kernel.forward(x, w)
            ctx.save_for_backward(x, w)
            return out
    
        @staticmethod
        def backward(ctx, grad_out):
            x, w = ctx.saved_tensors
            grad_x, grad_w = my_cuda_kernel.backward(grad_out, x, w)
            return grad_x, grad_w

Without this wrapper, your kernel can't be used in training — autograd doesn't know how to compute its gradient. With it, your kernel is a first-class operation, indistinguishable from a built-in.

### 3\. Non-differentiable ops with custom gradients (straight-through estimators)

Some operations have no gradient. `torch.round`, `torch.argmax`, `x.long()`, sampling from a discrete distribution — they all have gradients that are zero almost everywhere and undefined at the boundaries. Useless for training.

The _straight-through estimator_ (STE) is a classic hack: in the forward pass, do the non-differentiable thing; in the backward pass, pretend it was the identity function and pass the gradient through unchanged. The math is wrong; the empirical results are good enough. STE is the foundation of QAT (Module 26), VQ-VAEs, Gumbel-softmax workarounds, and more.
    
    
    class RoundSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.round()                # non-differentiable
    
        @staticmethod
        def backward(ctx, grad_out):
            return grad_out               # pretend round was the identity. Lie. Works.
    
    # Use:
    x = torch.randn(5, requires_grad=True) * 3
    q = RoundSTE.apply(x)             # integer-valued, but with a fake gradient path
    loss = (q ** 2).sum()
    loss.backward()                     # x.grad is 2*q, propagated through the lie

The same pattern handles binarization, discretization, top-k selection in differentiable models — anywhere you'd otherwise be stuck.

### 4\. Replacing PyTorch's backward with a smarter one

Sometimes PyTorch's automatic backward is correct but algorithmically inferior. The headline example is attention: the _standard_ autograd backward of `softmax(QKᵀ/√d)V` needs to materialize the full `(B, H, T, T)` attention matrix. For T=8192 and B=4 batch and H=32 heads, that's hundreds of GB.

FlashAttention sidesteps this by computing the same gradient through a _different algorithm_ that doesn't ever materialize the full matrix. The forward and backward both tile the computation and recompute as needed. The output is mathematically identical; the memory profile is dramatically better. None of that fits inside autograd-generated code — it requires writing forward and backward by hand. We'll do this in Module 25.

𝓕

Function

"My existence is what makes those stories possible."

Without me, every op's backward must be derivable from the op's forward. With me, you can write any forward you want and any backward you want, as long as they're _consistent_ (gradcheck enforces that). FlashAttention, gradient checkpointing, custom CUDA kernels, straight-through estimators — all of them are instances of "we want a backward that the framework wouldn't generate for us." That's my whole purpose.

## Common pitfalls

#### Q&A; — About custom Functions **Q:** My backward returns the wrong shape. What did I mess up? **A:** Almost certainly broadcasting. If your forward broadcast a `(D,)` tensor over a `(B, T, D)` tensor, the gradient w.r.t. the `(D,)` tensor needs to _sum back over the broadcast dims_ to recover shape `(D,)`. The autograd engine does this for you on built-in ops; you have to do it explicitly in custom backwards. See the LayerNorm example — note the `.sum(dim=leading_dims)` on grad_weight and grad_bias. **Q:** I get "one of the variables needed for gradient computation has been modified by an inplace operation." What's that about? **A:** You saved a tensor in forward, then someone (maybe in a different module) wrote to it in place between forward and backward. The version counter detected this and refuses to use the now-corrupted saved value. Either don't mutate in place, or save a `.clone()`. Performance cost: an extra copy. **Q:** Can I do non-tensor returns? **A:** Yes — forward can return tuples of tensors, plus you can stash non-tensor values on `ctx`. But the position-matching rule still holds: backward gets one upstream gradient per _tensor_ output of forward, in order. If forward returned `(out1, out2, "some_string")`, backward gets `(grad_out1, grad_out2)`. **Q:** Does my Function work with `torch.compile`? **A:** Mostly yes, but with caveats. `torch.compile` can call your Function as an opaque op (it won't try to trace into it). You may need to mark it with `@torch._dynamo.allow_in_graph` for some configurations. We'll see this in Module 21. For most kernel-style Functions (those wrapping a CUDA call), it works out of the box. **Q:** Should I use `autograd.Function` for everything custom? **A:** No. If you can write your op as a composition of existing PyTorch ops and the autodiff backward is fine, do that. Custom Functions are for the four reasons above — memory savings, kernel integration, non-differentiable ops, smarter algorithms. Otherwise you're adding complexity for no benefit. 

## The "saved for backward" memory diagram

The whole point of custom Functions is often controlling what gets saved. Here's the picture for a 5-step chain.

What gets saved? Two strategies, same forward. Default autograd: save EVERY intermediate xSAVED a = sin(x)SAVED b = a*aSAVED c = exp(b)SAVED y = c*2SAVED Memory: 5N Custom Function: save just x, recompute in backward xSAVED a (transient)freed b (transient)freed c (transient)freed y (output)kept Memory: 2N backward recomputes a, b, c from x — costs ~2× compute, saves ~3× memory

This is the trade you're constantly making in custom autograd: **save more = use more memory but skip recomputation** ; **save less = use less memory but recompute in backward**. The right answer depends on whether you're memory-bound or compute-bound. For large models, you're almost always memory-bound; saving less wins.

## Code Magnets: build a custom GELU with cheap backward

You're implementing GELU (a common activation) and want to be smart about saved memory. The forward is `y = x * sigmoid(1.702 * x)` (the "approximate" form). The backward needs `x` and the sigmoid value (call it `s`). Build a Function that's correct and memory-frugal.

Arrange the magnets into a complete Function. Three are red herrings.

class FastGELU(torch.autograd.Function): @staticmethod def forward(ctx, x): s = torch.sigmoid(1.702 * x) ctx.save_for_backward(x, s) ctx.x = x return x * s def backward(ctx, grad_y): x, s = ctx.saved_tensors grad_x = grad_y * (s + x * s * (1 - s) * 1.702) return grad_x return grad_y

show solution
    
    
    class FastGELU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            s = torch.sigmoid(1.702 * x)
            ctx.save_for_backward(x, s)
            return x * s
    
        @staticmethod
        def backward(ctx, grad_y):
            x, s = ctx.saved_tensors
            grad_x = grad_y * (s + x * s * (1 - s) * 1.702)
            return grad_x

The traps:

  * `ctx.x = x` bypasses `save_for_backward` — works in this case but loses the version-counter safety.
  * `return grad_y` alone would just pass the gradient through (a wrong identity backward, like the STE pattern but applied to the wrong op).
  * The second `@staticmethod` for backward is needed — both methods must be static.

The math: derivative of `x * sigmoid(c*x)` is `sigmoid(c*x) + x * c * sigmoid(c*x) * (1 - sigmoid(c*x))` by product rule plus the sigmoid derivative. We multiply that local derivative by `grad_y`.

## Who does what?

Match each tool/concept to its purpose.

Tool / concept

Purpose

ctx.save_for_backward(t)

A. Verifies your custom backward against finite differences.

ctx.alpha = 0.5

B. Saves a tensor with version-counter tracking for backward.

MyOp.apply(x)

C. Stashes non-tensor data on the context for backward to read.

gradcheck(fn, inputs)

D. The correct way to invoke a custom Function — hooks into autograd.

Straight-through estimator

E. Returning grad_y unchanged from a backward, regardless of forward.

Recompute in backward

F. The trick that powers gradient checkpointing and FlashAttention.

show solution

**ctx.save_for_backward(t)** → B  
**ctx.alpha = 0.5** → C  
**MyOp.apply(x)** → D  
**gradcheck(fn, inputs)** → A  
**Straight-through estimator** → E  
**Recompute in backward** → F 

The mental shortcut: _save_for_backward is for tensors with version safety, ctx.attr is for plain Python data, .apply is the entry point, gradcheck is the verifier, STE lies through backward, recomputation trades compute for memory_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** Implement `Cube` as a custom Function: `y = x³`. Don't save anything via `save_for_backward` — instead, use the trick that `3x²` can be expressed in terms of the output as `3 * y^(2/3) * sign(y)`... wait, that's awful. Just save `x` and verify with gradcheck.

show answer
    
    
    class Cube(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x ** 3
    
        @staticmethod
        def backward(ctx, grad_y):
            x, = ctx.saved_tensors
            return grad_y * 3 * x ** 2
    
    x = torch.randn(3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(Cube.apply, (x,))

The exercise framing was a bit of a trick — yes, you could derive `x` from `y` (via cube root, careful with sign), but doing so is more expensive than just saving `x`. The "save vs recompute" tradeoff isn't always recompute-wins. Save `x`; move on.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Write an STE-style Function called `BinarizeSTE` that returns `sign(x)` in forward and the identity in backward. Verify gradcheck _doesn't_ catch the lie (it's checking your declared backward against your declared forward via finite differences — and your declared backward says "identity," which gradcheck cannot tell is a lie).

show answer
    
    
    class BinarizeSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return torch.sign(x)
    
        @staticmethod
        def backward(ctx, grad_y):
            return grad_y       # the lie
    
    x = torch.randn(3, dtype=torch.float64, requires_grad=True)
    try:
        torch.autograd.gradcheck(BinarizeSTE.apply, (x,))
    except Exception as e:
        print("gradcheck failed (correctly)")

gradcheck _does_ fail here, because finite-difference of sign(x) is approximately zero, but you declared the gradient to be 1. STE backward is intentionally inconsistent with its forward — gradcheck is doing its job. In real STE code, you simply don't run gradcheck on it; you trust the empirical training results.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** Why does the LayerNorm backward we wrote sum the affine gradients (`grad_weight`, `grad_bias`) over _all leading dims_? Why not just dim=0?

show answer

Because LayerNorm's input typically has shape `(B, T, D)` (batch, sequence, hidden), and `weight`/`bias` have shape `(D,)`. The forward broadcasts `weight` across both `B` and `T`. The chain rule then says: the gradient w.r.t. `weight` is the sum of contributions over _every_ position it was broadcast to — both the batch dim and the sequence dim. `sum(dim=(0, 1))`, which is `sum(dim=tuple(range(grad_y.dim() - 1)))` in shape-agnostic form. If you only summed over dim=0, you'd get a `(T, D)` gradient that doesn't match `weight`'s shape and PyTorch would error.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Suppose your custom Function's forward calls `x.contiguous()` internally. Should backward return `grad_x` with the original layout, or contiguous?

show answer

Either works numerically, but PyTorch expects `grad_x` to have the same shape (and ideally strides) as the original input `x`. If you return a contiguous gradient when the input was non-contiguous, autograd will accept it, but downstream ops that expected the original layout may incur a hidden copy. The polite thing is to _match the input's memory layout_. In practice, most kernels return contiguous gradients and let downstream code handle it. The framework doesn't enforce strictness here.

### What just happened?

  * A **custom`autograd.Function`** is a class with two static methods. `forward(ctx, *inputs)` computes outputs and saves what backward needs. `backward(ctx, *grad_outputs)` returns one gradient per forward input.
  * Use `MyOp.apply(...)`, never `MyOp(...)` — `.apply` is what hooks into autograd.
  * **Save tensors** via `ctx.save_for_backward(...)`; retrieve via `ctx.saved_tensors`. **Save non-tensor data** as plain attributes on `ctx` (e.g., `ctx.eps = 1e-5`).
  * `gradcheck` verifies your analytical backward against finite differences. Use **fp64** , **small tensors** , **multiple seeds**. `gradgradcheck` for double-backward consistency.
  * The **four reasons to write a custom Function** : (1) memory savings via recompute, (2) integrating a custom CUDA/Triton kernel, (3) non-differentiable ops with a fake gradient (STE), (4) a smarter algorithm than autograd would generate (FlashAttention).
  * **Save vs recompute** is the central tradeoff. Save more = use more memory, skip recomputation. Save less = recompute in backward. Memory-bound workloads almost always want recompute.
  * The **straight-through estimator** is the canonical "lie in backward to enable training through a non-differentiable op" pattern. Foundation of QAT, VQ-VAEs, discrete bottlenecks.
  * Custom backward must handle **broadcasting in reverse** : if forward broadcast a small tensor over leading dims, the gradient must be summed back over those dims to recover the small shape.
  * **Watch for in-place mutations** of saved tensors. The version counter will catch you. Save a `.clone()` if you must.
  * The reflex: when you reach for a custom Function, ask _which of the four reasons_ applies. If none, you probably don't need one — pure-PyTorch will be simpler and just as fast.

That closes Part II. You now have the storage/stride model, the shape rules, the dtype/device model, the autograd graph model, the practical autograd toolbox, and the ability to write your own backward. **Every kernel-writing module from M22 onwards stands on this foundation** — when we get to FlashAttention in M25, the "custom backward that recomputes the forward" pattern will not be a surprise.

Part III opens with Module 07: `nn.Module` as a container system. Parameters vs buffers, hooks at the module level, the registration gotcha that orphans your weights, and the `state_dict` lifecycle that powers checkpointing.
