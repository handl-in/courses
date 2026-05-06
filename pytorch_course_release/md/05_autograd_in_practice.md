# Module 05 — Autograd in practice

# Autograd _in practice_

_Part II · Module 05_

— stopping gradients on purpose, peeking at them with hooks, and the `torch.func` API that finally lets you treat gradients as values

\--- 

Module 04 gave you the model. Now you need the tool belt. Real autograd code is a constant negotiation: _track this_ , _stop tracking that_ , _peek at the gradient before the optimizer eats it_ , _compute a Hessian without writing one by hand_. Each of those needs a different switch, and each of those switches has subtle failure modes.

This module is organized around two questions that come up every day: **"how do I stop a gradient from flowing where I don't want it?"** and **"how do I see what's actually happening inside the graph?"** Plus the obligatory section on _"why is my gradient None?"_ , which is the question every PyTorch user has stared at on Stack Overflow at least once.

> **★ KEY IDEA**  
>  There are **four ways to stop a gradient** , and they're not interchangeable: `requires_grad=False` is a property on a leaf tensor, `detach()` creates a new tensor cut from the graph, `no_grad()` is a context manager that disables tracking entirely, and `inference_mode()` is the faster cousin of `no_grad` for production. Picking the wrong one usually still gives you a working program — but it costs memory, speed, or both. 

## Welcome back to the cast (with one new face)

You already know the autograd characters from M4. One new one for today.

👁

Hook

"I'm the spy you place on a tensor or module to peek at what flows through."

Register me, and I get called every time a gradient passes through that tensor — or every time a module's forward runs. I can _look_ at the gradient. I can _modify_ the gradient if I'm feeling cheeky. I'm how you debug "is the gradient even reaching layer 7?" without printing in 200 places. Just remember to remove me when you're done — I leak otherwise.

## Four ways to stop a gradient

Let's compare them side by side first, then take each apart.

The four "stop the gradient" mechanisms Mechanism| What it does| When to use  
---|---|---  
`x.requires_grad = False`| Marks a _leaf_ as not needing gradients. Permanent (until you flip it back).| Freezing model parameters: `for p in encoder.parameters(): p.requires_grad = False`.  
`x.detach()`| Returns a _new_ tensor sharing storage with x but with no `grad_fn`. Backward stops at it.| Cutting one tensor out of the graph mid-computation while letting the rest of the graph keep flowing. Stop-gradient in some loss formulations.  
`with torch.no_grad():`| Context manager. Inside it, no operations record to the graph. Outputs are non-leaf tensors with `requires_grad=False`.| Inference. Computing metrics. Updating parameters in place outside autograd.  
`with torch.inference_mode():`| Stricter version of `no_grad`. Outputs cannot ever be tracked again. Skips even more bookkeeping. Faster.| Pure inference loops in production where you'll never need backward on the result.  
  
### `requires_grad = False` — freezing parameters
    
    
    # Freeze a pre-trained encoder, train only the classifier head
    for p in encoder.parameters():
        p.requires_grad = False
    
    # Now the optimizer should ONLY get the trainable params
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
    )

This is the most common use. Note the optimizer line — if you give it parameters that don't require grad, you'll either waste memory on dead optimizer state or get an error from the optimizer's sanity check. Filter with the comprehension.

> **⚠ WARNING**  
>  **BatchNorm and Dropout don't care about`requires_grad`.** Freezing parameters does NOT put a layer in eval mode. BN still updates its running statistics during forward, Dropout still drops. To actually freeze a layer's _behavior_ , you also need `module.eval()`. We'll dig into this in Module 7. 

### `detach()` — cutting one tensor out

Use this when you have an expression where you want gradients to flow through _some_ inputs but not others. Classic example: target networks in Q-learning, where the target's parameters shouldn't receive gradients from the loss.
    
    
    # Forward pass through both networks
    q_values = q_net(state)               # trainable
    next_q = target_net(next_state)       # also trainable, but we don't want grads
    
    # Bellman target: r + γ * max_a' Q_target(s', a')
    # We want gradients to flow into q_net via q_values, but NOT via next_q
    target = reward + gamma * next_q.max(dim=-1).values.detach()
    loss = (q_values.gather(1, action[:, None]) - target[:, None]).pow(2).mean()
    loss.backward()                       # gradients flow ONLY into q_net

Without the `.detach()`, gradients would leak into `target_net`, which we explicitly don't want — the whole point of having a target net is to keep it stable.

⌬

grad_fn

"`detach()` just sets me to None."

A detached tensor still shares storage with the original — same bytes, same data. The only difference is that I'm gone. So during backward, when the walk reaches the detached tensor, it stops. The original tensor (still in the graph elsewhere) is unaffected.

### `no_grad()` — disabling tracking entirely

The most heavyweight option, and the right one for inference and metric computation. Inside the `with` block, _no_ operation builds graph nodes:
    
    
    model.eval()                     # also turn off Dropout, BN train mode
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch)
            loss = criterion(out, target)
            # No graph built. out and loss have requires_grad=False.
            # Memory used = forward activations only, no saved-for-backward buffers.

Why does this matter? Because building the graph isn't free. Every op allocates a grad_fn object and saves the inputs needed for backward. On a transformer forward pass, those saved buffers can be 5-10× the size of the output tensors. `no_grad` skips all of that.

### `inference_mode()` — the faster `no_grad`

Introduced in PyTorch 1.9 as a stricter, faster version. The difference is that tensors created inside `inference_mode` can _never_ participate in autograd later, even if you wrap them in `requires_grad=True` or pass them through a graph-building op. PyTorch can therefore skip even more bookkeeping (specifically, the version counter we met in Module 1).
    
    
    with torch.inference_mode():
        out = model(batch)
        # out.is_leaf is True, requires_grad is False, and out is "stuck"
        # in inference mode — you can't use it as input to a backward graph.
    
    # Outside the block:
    y = out.requires_grad_(True)        # RuntimeError! inference tensors can't be tracked.

For pure inference servers, `inference_mode` can give a few percent speedup over `no_grad`. For research code where you might want to do something with the output later, stick with `no_grad`.

#### Q&A; — Picking the right "stop" **Q:** When should I prefer `detach()` over `no_grad()`? **A:** When you only want to stop the gradient at _one specific tensor_ while letting other parts of the same expression still build the graph. `no_grad` disables tracking for everything inside the block; `detach` is surgical. **Q:** Is there a performance difference between `requires_grad=False` and detaching during forward? **A:** Yes. Setting `requires_grad=False` on a parameter means PyTorch never tracks operations involving it (assuming no other input requires grad). Detaching mid-computation means the graph was built but stops at the detach point. The first is faster if you're freezing for a long time; the second is fine for one-off cuts. **Q:** What's the difference between `model.eval()` and `torch.no_grad()`? **A:** They solve completely different problems. `model.eval()` changes module behavior — Dropout off, BN uses running stats. `no_grad()` changes autograd behavior — no graph built. For inference you almost always want both. **Q:** Can I nest these? **A:** Yes. `no_grad` nested inside `no_grad` is harmless. `enable_grad` exists as the inverse — useful inside a function that's called from many places, some of which are in `no_grad` mode. 

## Why is my gradient None?

This is the single most-Googled PyTorch question. Let's build a diagnostic flowchart you can run through in your head.

Why is my gradient None? Does the tensor have requires_grad = True? NO Set it. Or check that you didn't reassign x = x.cuda() and orphan the leaf. YES Is it a leaf? Check x.is_leaf. Non-leaves don't get .grad by default. YES (it's a leaf) Did you actually call .backward() since the last zero? YES Did the loss actually depend on this tensor at all? NO Backward never reached this leaf → .grad stays None. (Common with frozen branches.) NO (non-leaf) Add x.retain_grad() before backward, or reach for hooks (next section).

Walk through it on every "grad is None" case you ever see. The four checks cover ~95% of incidents:

  1. Does it have `requires_grad=True`? If not, fix that — and check you didn't reassign through an op like `.cuda()` or `.float()` (the M4 trap).
  2. Is it a leaf? If not, you wanted `retain_grad()` or hooks.
  3. Did you call backward at all? Did `zero_grad` wipe it out before you could read it?
  4. Did the loss actually depend on this tensor? If you have a layer with effectively-zero contribution to the loss, backward never sends a gradient there. `.grad` stays `None`. (DDP catches this with `find_unused_parameters=True` — Module 16.)

## Hooks: peeking inside the graph

A hook is a function that runs at a specific point in the forward or backward pass. They're how you peek at intermediate gradients, modify them, or just print them for debugging.

Three flavors you'll meet:

The three hook types Hook| Attach to| When it fires| Can it modify?  
---|---|---|---  
`tensor.register_hook(fn)`| A tensor| When the gradient flows back through it| Yes — return a new tensor to replace the gradient  
`module.register_forward_hook(fn)`| An `nn.Module`| After the module's forward returns| Yes — return a new output  
`module.register_full_backward_hook(fn)`| An `nn.Module`| When the backward through the module finishes| Yes — return new gradients w.r.t. inputs  
  
### A tensor hook for debugging
    
    
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = (x ** 2).sum()
    
    # Print the gradient flowing back into x
    handle = x.register_hook(lambda grad: print("grad on x:", grad))
    
    y.backward()
    # prints: grad on x: tensor([4., 6.])
    
    handle.remove()    # clean up — hooks leak otherwise!

Crucially, the hook fires _during_ backward, not after. So if you're trying to debug why gradients are zero somewhere, register a hook before calling backward, and you'll see the gradient at the moment it passes through.

### A module forward hook for activation inspection
    
    
    activations = {}
    
    def save_activation(name):
        def hook(module, inputs, output):
            activations[name] = output.detach()
        return hook
    
    handle = model.layer3.register_forward_hook(save_activation('layer3'))
    
    out = model(batch)
    print(activations['layer3'].shape)
    
    handle.remove()

This pattern is everywhere in interpretability research — extracting activations from a frozen model without modifying its code.

### Modifying gradients with a hook

You can return a new tensor from a hook to replace the gradient. This is how gradient clipping per layer used to be implemented before the formal API:
    
    
    # Clip the gradient flowing into x to be in [-1, 1]
    x.register_hook(lambda grad: grad.clamp(-1, 1))

Or to _reverse_ a gradient (the gradient reversal layer from domain-adaptation papers):
    
    
    # Reverse the gradient — useful in adversarial training
    x.register_hook(lambda grad: -grad * lambda_)

Hooks are powerful and a common source of subtle bugs (because they run silently and order matters). Use them deliberately, and remove them with the returned handle when done.

> **⚠ WARNING**  
>  **Hooks leak if you forget to remove them.** Every hook you register holds a reference to the closure, which can hold references to model parameters or worse. In a long-running process (training notebook, server) this adds up. The pattern is: `handle = x.register_hook(...)`, do work, `handle.remove()`. Or use the context-manager pattern with a small helper class. 

## `retain_graph` and double backward

You'll occasionally need to call backward more than once on the same forward graph, or compute gradients of gradients. Two switches make this work.

**`retain_graph=True`** tells PyTorch not to free the saved-for-backward buffers after this backward call, so you can call backward again on the same graph. Use case: you have multiple losses you compute backward for separately.
    
    
    loss1.backward(retain_graph=True)    # keeps the graph around
    loss2.backward()                       # now this works; graph is freed after

Note: _just calling backward twice in a row on the same loss_ is almost always a bug. The case for `retain_graph` is when you have separate losses that share a forward computation.

**`create_graph=True`** tells PyTorch to build a graph _over the backward pass itself_ , so the resulting gradients are differentiable and you can backward through them. Use case: anywhere you need second-order gradients (Hessian-vector products, MAML, gradient penalties in WGAN-GP).
    
    
    # Compute a Hessian-vector product
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 3                            # y = 8
    
    grad_x = torch.autograd.grad(y, x, create_graph=True)[0]
    # grad_x = 3x² = 12, AND grad_x has requires_grad=True (because of create_graph)
    
    grad2_x = torch.autograd.grad(grad_x, x)[0]
    # grad2_x = 6x = 12 — this is d²y/dx²

Double backward is memory-intensive. The graph for the gradient computation is roughly the same size as the original forward graph. For most uses, `torch.func.hessian` or `torch.func.jacrev(jacrev(f))` is cleaner — see below.

## `torch.func`: gradients as values

The traditional autograd API is _imperative_ : you build a graph, you call `.backward()`, you read `.grad`. It works, but it's awkward when you want to _compose_ gradients — take the gradient of a gradient, vmap a gradient computation across a batch of inputs, compute a Jacobian without writing a loop.

`torch.func` (formerly `functorch`) is the functional API. Gradients are _values returned by functions_ , not side effects on tensors. This composes beautifully.

∂

Autograd

"`torch.func` is JAX-style for PyTorch."

If you've used JAX, this will feel familiar. `grad(f)` takes a function and returns a function that returns the gradient. `vmap(f)` takes a function and returns a batched version. `jacrev(f)` returns the Jacobian. They compose: `grad(grad(f))` is the second derivative; `vmap(grad(f))` is per-sample gradients across a batch. The same operations you can do with the imperative API, but composable.
    
    
    import torch
    from torch.func import grad, vmap, jacrev
    
    def f(x):
        return (x ** 3).sum()
    
    # grad: returns a function that returns the gradient
    df = grad(f)
    print(df(torch.tensor([2.0, 3.0])))   # tensor([12., 27.]) — that's 3x²
    
    # Second derivative — just compose grad twice
    d2f = grad(grad(lambda x: f(x)))         # f as a scalar fn for grad-grad
    
    # vmap: batched version of any function
    batch = torch.randn(8, 3)               # 8 examples of shape (3,)
    per_sample_grads = vmap(df)(batch)        # shape (8, 3) — gradient for each example
    
    # jacrev: full Jacobian by reverse-mode
    def g(x):
        return torch.stack([(x ** 2).sum(), (x ** 3).sum()])
    J = jacrev(g)(torch.tensor([1.0, 2.0, 3.0]))
    # J shape: (2, 3) — Jacobian of g at x

Three things to internalize about `torch.func`:

  1. **It's pure-functional.** No `.backward()`, no `.grad`. Gradients come back as return values. This makes them composable.
  2. **`vmap` is genuinely useful, even without grad.** Want per-sample anything? `vmap` the per-sample function. No more "I have to re-implement this without batching for the inner step."
  3. **It interoperates with the rest of PyTorch.** The functions you pass can use `nn.Module`s; you call `functional_call` to bind parameters as arguments. Module 5 doesn't go deep on this — it's a topic worth its own walkthrough — but be aware it's the modern way to compute things like per-sample gradients (used in differentially-private training).

### Per-sample gradients in 4 lines

One of the most useful applications. In standard training, the optimizer sees the _average_ gradient over a batch. Sometimes you want the per-sample gradient — for differential privacy, influence-function analysis, or just curiosity.
    
    
    from torch.func import functional_call, grad, vmap
    
    def loss_fn(params, sample, target):
        pred = functional_call(model, params, sample.unsqueeze(0))
        return F.cross_entropy(pred, target.unsqueeze(0))
    
    per_sample_grad_fn = vmap(grad(loss_fn), in_dims=(None, 0, 0))
    per_sample_grads = per_sample_grad_fn(dict(model.named_parameters()), batch_x, batch_y)
    # per_sample_grads is a dict where each value has shape (batch_size, *param_shape)

Before `torch.func`, this required either a `for` loop over the batch (slow) or a clever rewrite of every layer (painful). Now it's four lines.

## Code Magnets: build a gradient-saving training step

You're writing a training step that, in addition to the normal forward / backward / step / zero, _saves a copy of the gradient on the first layer's weight_ for later inspection. Use a hook.

Arrange the magnets into a working step. There are two red herrings.

saved_grads = [] handle = model.layer1.weight.register_hook(lambda g: saved_grads.append(g.clone())) handle = model.layer1.register_forward_hook(lambda m, i, o: saved_grads.append(o)) optimizer.zero_grad() loss = criterion(model(x), y) loss.backward() optimizer.step() handle.remove() model.layer1.weight.requires_grad = False

show solution
    
    
    saved_grads = []
    handle = model.layer1.weight.register_hook(lambda g: saved_grads.append(g.clone()))
    
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    
    handle.remove()

The two traps:

  * `register_forward_hook` records the layer's _output_ , not the gradient. Wrong tool.
  * `requires_grad = False` would stop the gradient from being computed at all — defeating the purpose.

The `g.clone()` inside the hook matters: `g` is the gradient tensor flowing through; if you don't clone it, you're saving a reference that could be mutated or freed by the time you look at it. Always clone gradients you want to keep.

## Who does what?

Match each tool to the problem it solves.

Tool

Problem

x.detach()

A. Need to backward more than once on the same forward graph (separate losses).

with torch.no_grad():

B. Want second-order gradients (gradient of a gradient).

with torch.inference_mode():

C. Want to peek at the gradient flowing into a tensor without modifying it.

retain_graph=True

D. Surgical: cut just one tensor out of the graph, let others keep flowing.

create_graph=True

E. Production inference loop, never need to backprop on the result.

tensor.register_hook(fn)

F. Disable graph building entirely for a block of code (e.g., validation).

vmap(grad(f))

G. Per-sample gradients across a batch in one call.

show solution

**x.detach()** → D  
**with torch.no_grad():** → F  
**with torch.inference_mode():** → E  
**retain_graph=True** → A  
**create_graph=True** → B  
**tensor.register_hook(fn)** → C  
**vmap(grad(f))** → G 

The mental shortcut: _detach is surgical, no_grad is a block, inference_mode is the strict no_grad, retain_graph keeps memory, create_graph builds graph over backward, hooks observe, vmap+grad batches per-sample_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** You have a model with two heads — `head_a` and `head_b` — sharing an encoder. You want to train both heads, but you only want the encoder to be updated by `head_a`'s loss, not `head_b`'s. Write the loss computation.

show answer
    
    
    features = encoder(x)
    
    # head_a sees features through the graph as normal
    loss_a = criterion(head_a(features), y_a)
    
    # head_b sees a detached version, so its loss can't backprop into encoder
    loss_b = criterion(head_b(features.detach()), y_b)
    
    (loss_a + loss_b).backward()

The encoder's gradient comes only from `loss_a` because `loss_b`'s path through `features.detach()` stops at the detach. Both heads still receive their own gradients normally because `head_a` and `head_b`'s parameters are leaves.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why might `torch.no_grad()` not be enough for a production inference server, and when does `inference_mode()` actually win?

show answer

`no_grad` still maintains version counters on tensors (the in-place-detection mechanism we met in Module 1) and other small bits of bookkeeping. `inference_mode` skips that too. The win is usually a few percent in throughput, but it can matter at scale. The catch: tensors created in `inference_mode` can never be tracked by autograd later. If your server occasionally needs to compute a gradient (e.g., for adversarial robustness checks), `inference_mode` will break that path.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** Compute the second derivative of `sin(x)` at `x = π/4` using `torch.func`.

show answer
    
    
    from torch.func import grad
    import math
    
    f = lambda x: torch.sin(x)
    d2f = grad(grad(f))
    
    x = torch.tensor(math.pi / 4)
    print(d2f(x))    # tensor(-0.7071) — that's -sin(π/4) ✓

Compare to the imperative API: you'd need `create_graph=True` on the first `autograd.grad` call, then a second `autograd.grad` call with the right inputs. Three lines vs five, but the `torch.func` version reads as math.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Write a hook that prints whenever a NaN appears in the gradient of any parameter. Mock-test it on a tiny model.

show answer
    
    
    def nan_warning_hook(name):
        def hook(grad):
            if torch.isnan(grad).any():
                print(f"NaN in grad of {name}!")
        return hook
    
    handles = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            handles.append(p.register_hook(nan_warning_hook(name)))
    
    # ...train as normal; if any param's gradient gets NaN, you'll know which one ...
    
    for h in handles:
        h.remove()

This is a real diagnostic pattern, especially in mixed-precision training. The hook fires as the gradient is computed, so you spot the NaN at its source instead of much later when training has gone off the rails.

### What just happened?

  * **Four ways to stop a gradient** , none interchangeable: `requires_grad=False` (per-tensor flag, freezing), `detach()` (surgical cut, returns a new tensor), `no_grad()` (block-level, no graph built), `inference_mode()` (stricter and faster, for production).
  * `model.eval()` and `no_grad()` solve **different** problems. Eval changes layer behavior (Dropout off, BN uses running stats); no_grad disables autograd. Inference wants both.
  * The **"why is my grad None?"** diagnostic: check requires_grad, check is_leaf, check that backward ran, check that the loss actually depends on the tensor.
  * **Hooks** are spies. `tensor.register_hook` for gradients, `module.register_forward_hook` for activations, `module.register_full_backward_hook` for module-level grads. **Always remove them** with the returned handle.
  * Hooks can _modify_ what flows through (clip, reverse, scale). The gradient reversal layer is one line.
  * `retain_graph=True` = "don't free the saved buffers, I'll backward again." Use only when you have separate losses.
  * `create_graph=True` = "build a graph over the backward pass." Enables double backward (Hessian-vector products, MAML, gradient penalties).
  * **`torch.func`** is the functional API. `grad(f)`, `vmap(f)`, `jacrev(f)`, `jacfwd(f)`, `hessian(f)` compose freely. Per-sample gradients in 4 lines.
  * Always `g.clone()` a gradient you want to keep — references are fragile.
  * The reflex: when in doubt, draw the graph from M4. Then ask which switch turns off which edge.

Module 06 takes the autograd model and lets you write your own. We'll build a custom `torch.autograd.Function` with explicit forward and backward, verify it with `gradcheck`, and discuss when it earns its keep — fused ops, memory-saving tricks, straight-through estimators, and the cases where the framework's automatic backward isn't what you want.
