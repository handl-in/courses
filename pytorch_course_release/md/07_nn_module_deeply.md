# Module 07 — nn.Module deeply

# `nn.Module`, _deeply_

_Part III · Module 07_

— parameters vs buffers, the registration gotcha that silently orphans your weights, and what really happens when you call `model.cuda()`

\--- 

You've used `nn.Module` a hundred times. You subclass it, define `__init__` and `forward`, call `model(x)`, and gradients magically appear in `model.parameters()`. This module is about why "magically" is doing too much work in that sentence.

The actual story is: `nn.Module` is a _container_ — a tree of named tensors and other Modules. It maintains three internal dictionaries: one for parameters, one for buffers, one for child modules. When you assign a tensor or Module to `self.something`, PyTorch decides which dictionary it belongs in based on its type. Most "weird module behavior" — vanishing weights, missing keys in state_dict, frozen layers that aren't really frozen — is one of those decisions going wrong.

> **★ KEY IDEA**  
>  An `nn.Module` is **three dictionaries plus a forward method** : `_parameters`, `_buffers`, and `_modules`. Anything you want PyTorch to find, save, move, or update must live in one of those. Plain Python attributes (lists, dicts, raw tensors) are _invisible_ to `.parameters()`, `.state_dict()`, `.cuda()`, and `optimizer`. The "registration gotcha" is when you stash something in a place PyTorch can't see. 

## Meet the Module family

M

Module

"I'm a tree. Everything you do to me, I do to my children."

When you call `model.cuda()`, I recursively visit every parameter, buffer, and child module, moving each to the GPU. Same for `.train()`, `.eval()`, `.float()`, `.state_dict()`. The _tree_ is the abstraction. But I can only walk children I know about — and I only know about what was registered. If you stuck a Module in a plain Python list, I won't find it.

P

Parameter

"I'm a tensor that gets gradients and shows up in `.parameters()`."

Wrap a tensor in `nn.Parameter(...)` and assign me to `self.weight`. Now I'm registered. I appear in `model.parameters()` (so the optimizer finds me), I appear in `state_dict()` (so checkpointing finds me), I get moved by `.cuda()`, and I have `requires_grad=True` by default. I'm the thing the optimizer updates.

B

Buffer

"I'm a tensor that's part of the model state but never gets a gradient."

Register me with `self.register_buffer('running_mean', torch.zeros(D))`. I show up in `state_dict()`, I get moved by `.cuda()`, but I'm _not_ in `.parameters()` and the optimizer ignores me. BatchNorm's running statistics are buffers. Causal masks are buffers. RoPE's precomputed cos/sin tables are buffers. Anything the model needs to remember but shouldn't be trained.

## The three dictionaries

Let's open up `nn.Module` and stare at the three dictionaries that run the show.
    
    
    import torch
    import torch.nn as nn
    
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(3, 4))    # goes to _parameters
            self.register_buffer('mask', torch.ones(3))     # goes to _buffers
            self.head = nn.Linear(4, 10)                  # goes to _modules
            self.scratch = torch.randn(5)                  # goes to... NOWHERE useful
    
    m = Tiny()
    print(m._parameters.keys())     # odict_keys(['weight'])
    print(m._buffers.keys())        # odict_keys(['mask'])
    print(m._modules.keys())        # odict_keys(['head'])
    # scratch is just an instance attribute, not in any of these

Look at that last line. `self.scratch` is a regular Python attribute. It's a tensor, sure, but PyTorch has no idea it's there. Specifically:

  * `m.parameters()` won't yield it.
  * `m.state_dict()` won't include it.
  * `m.cuda()` won't move it (it'll stay on CPU).
  * `m.eval()` won't affect it.

That's the registration gotcha in miniature. PyTorch's `__setattr__` hook checks whether you're assigning a `Parameter`, a `Module`, or something else, and routes it to the right dictionary. Plain tensors fall through to the default Python attribute set, which the framework never visits.

What __setattr__ does when you assign self.something = X self.x = something _parameters if it's nn.Parameter .parameters() ✓ state_dict ✓ · cuda ✓ trained by optimizer ✓ _buffers if you call register_buffer .parameters() ✗ state_dict ✓ · cuda ✓ no gradient _modules if it's nn.Module recursive — its params become my params cuda/state_dict cascade (plain attribute) anything else INVISIBLE to PyTorch no .parameters(), no state_dict, no .cuda() the right-most bucket is where bugs live

## The registration gotcha (the one that bites everyone)

Here's the bug in its purest form:
    
    
    class BrokenStack(nn.Module):
        def __init__(self, n_layers, dim):
            super().__init__()
            self.layers = [nn.Linear(dim, dim) for _ in range(n_layers)]    # Python list!
    
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    m = BrokenStack(3, 4)
    print(list(m.parameters()))     # []  ← EMPTY. 0 trainable parameters.
    m.cuda()                       # NO-OP for the layers — they stay on CPU

The model _runs_. Forward works. But the optimizer has nothing to train, the layers never move to GPU, and `state_dict()` is empty. You can train for hours and the layers don't update because the optimizer literally doesn't know they exist.

The fix: use `nn.ModuleList` instead of a plain list. Same iteration interface, but PyTorch sees the children.
    
    
    class FixedStack(nn.Module):
        def __init__(self, n_layers, dim):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
    
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    m = FixedStack(3, 4)
    print(sum(p.numel() for p in m.parameters()))   # 60 (= 3 * (4*4 + 4))

Same gotcha for dictionaries — use `nn.ModuleDict`. Same for sequential composition — `nn.Sequential` is also Module-aware.

> **⚠ WARNING**  
>  **The silent version.** The bug above is _visible_ because `parameters()` returns nothing. The really nasty version is when you have _some_ registered modules and a Python list with one extra layer: 
>     
>     
>     self.encoder = nn.Linear(dim, dim)                  # registered ✓
>     self.heads = [nn.Linear(dim, 10) for _ in range(3)]   # NOT registered ✗
> 
> The encoder trains fine. The heads don't. The model produces output (the heads are still callable from `forward`), but their weights are random forever. You'll see val accuracy plateau and assume your architecture is bad. It's actually that 3/4 of your model isn't training. 

## Parameters vs Buffers — when to use which

Both live in the model's state. Both move with `.cuda()`. Both appear in `state_dict()`. The difference comes down to two questions: _does the optimizer touch it?_ and _does autograd track it?_

Parameters vs Buffers, side by side Behavior| Parameter| Buffer  
---|---|---  
In `.parameters()`?| Yes| No  
In `.state_dict()`?| Yes| Yes (by default)  
Moved by `.cuda()` / `.to()`?| Yes| Yes  
`requires_grad` by default?| True| False  
Updated by optimizer?| Yes| No  
Updated during forward (e.g., running stats)?| No| Often yes  
How to register| `self.x = nn.Parameter(t)`| `self.register_buffer('x', t)`  
  
Real examples of buffers in the wild:

  * **BatchNorm's running mean and variance.** Updated during training-mode forward, used during eval-mode forward. Never trained.
  * **Causal masks in attention.** Precomputed once, never updated. Tied to the model so they move with it.
  * **RoPE / position-encoding tables.** Computed at init from the dim and max sequence length. Constant during training.
  * **EMA model weights.** Updated outside the optimizer (manually), saved with the checkpoint. Buffers fit perfectly.

**JUNIOR:** I added a causal mask as a regular tensor on `self`. Training works, but when I move the model to GPU the mask stays on CPU and I get a device mismatch. Why? **SENIOR:** Because plain tensor attributes aren't registered. `self.mask = torch.tril(...)` is invisible to `.cuda()`. Use `self.register_buffer('mask', torch.tril(...))`. Same data, but now PyTorch knows about it. **JUNIOR:** And it'll be in my checkpoints too? **SENIOR:** Yes — buffers go in `state_dict()` by default. If you don't want it persisted (because it's recomputed at init), pass `persistent=False`: `self.register_buffer('mask', tensor, persistent=False)`. Then it's still tracked for device moves but not saved. 

## The state_dict lifecycle

`state_dict()` returns an _OrderedDict_ mapping fully-qualified parameter and buffer names to tensors. It's how you save and load models.
    
    
    model = FixedStack(2, 4)
    sd = model.state_dict()
    print(list(sd.keys()))
    # ['layers.0.weight', 'layers.0.bias', 'layers.1.weight', 'layers.1.bias']
    print(sd['layers.0.weight'].shape)        # torch.Size([4, 4])

The key naming follows the module tree: `layers` is the ModuleList, `0` is the index, `weight` is the parameter name. Nested modules produce dotted paths like `encoder.attn.k_proj.weight`.

Saving and loading:
    
    
    # Save
    torch.save(model.state_dict(), 'model.pt')
    
    # Load (the standard pattern)
    new_model = FixedStack(2, 4)              # build the same architecture first
    sd = torch.load('model.pt', weights_only=True)
    new_model.load_state_dict(sd)              # in-place restore

Why save the `state_dict` and not the model object directly? Two reasons. (1) The state_dict is just tensors — portable, version-stable, doesn't depend on your code being importable at load time. (2) Pickled model objects break the moment you rename a class or move a file. State dicts survive refactors as long as the parameter names match.

### `load_state_dict` options worth knowing
    
    
    # Strict mode (default) — every key must match exactly
    model.load_state_dict(sd, strict=True)
    
    # Non-strict — partial loads, useful for fine-tuning checkpoints
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing keys (in model but not sd):", missing)
    print("unexpected keys (in sd but not model):", unexpected)
    
    # Useful when you've added a new head to a pretrained model:
    #   missing = ['new_head.weight', 'new_head.bias']  ← OK, will train these from scratch
    #   unexpected = ['old_head.weight', ...]           ← OK, you removed the old head

The `strict=False` path with explicit handling of missing/unexpected keys is the right way to do most fine-tuning. Print the lists, eyeball them, make sure the diffs match what you expected.

#### Q&A; — About state_dict **Q:** Why does PyTorch warn me about `weights_only` when I load a checkpoint? **A:** Pickled checkpoints can execute arbitrary code on load (it's a known Python pickle issue). Modern PyTorch defaults to `weights_only=True` for safety — it'll load tensors but reject anything else. If you trust the source, set `weights_only=True` explicitly to silence the warning. If you don't trust the source, this is your friend. **Q:** Can I rename keys at load time? **A:** Yes — load the dict as a regular Python dict, manipulate the keys, then call `load_state_dict`. 
    
    
    sd = torch.load('old.pt')
    sd = {k.replace('old_name', 'new_name'): v for k, v in sd.items()}
    model.load_state_dict(sd)

This is how you handle renamed layers between checkpoint versions. **Q:** What about the optimizer state? **A:** Save it separately. `optimizer.state_dict()` contains momentum buffers, Adam moments, step counts. To resume training mid-run, you save and load both: `{'model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': ...}`. We'll do this properly in Module 11. **Q:** My state_dict has weird keys like `module.encoder.weight` (with a leading "module."). What's that? **A:** You saved a checkpoint while wrapped in `DataParallel` or `DistributedDataParallel` (Module 16). The wrapper adds the prefix. To load into a non-wrapped model, strip the prefix: `{k.removeprefix('module.'): v ...}`. Or wrap before loading. 

## `train()` vs `eval()` — they don't do what you think

The world's most misunderstood API. Let's be precise.

`model.train()` sets the `training` flag to `True` on every Module in the tree. `model.eval()` sets it to `False`. **That's it.** They don't disable autograd, don't freeze parameters, don't move anything. They flip a single Boolean.

Why does it matter, then? Because some layers _check_ that flag and behave differently:

  * **Dropout** : drops in train mode, identity in eval mode.
  * **BatchNorm** : uses batch statistics + updates running stats in train mode; uses running stats only in eval mode.
  * **Custom modules** you write can check `self.training` and behave differently if you want.

Layers that don't care about training mode (Linear, Conv, LayerNorm, almost everything else) behave identically in either mode.

> **⚠ WARNING**  
>  **Three "eval" things people confuse.**
> 
>   * `model.eval()` — flips the training flag. Affects Dropout/BN behavior. Does NOT disable autograd.
>   * `torch.no_grad()` — disables autograd. Does NOT touch the training flag. Dropout still drops, BN still updates stats.
>   * `for p in model.parameters(): p.requires_grad = False` — freezes parameters from the optimizer's view. Does NOT touch the training flag or autograd-as-a-whole.
> 
For inference, you want **both** `model.eval()` AND `torch.no_grad()`. They solve different problems. 

## Module hooks (briefly — full coverage was M5)

Module-level hooks fire around the module's `forward`. They're useful for things you can't easily do inside `forward`: capturing intermediate activations from a frozen model, modifying outputs of a third-party layer, instrumenting for debugging.
    
    
    def capture_output(module, inputs, output):
        captures.append(output.detach())
    
    captures = []
    handle = model.encoder.layer[3].register_forward_hook(capture_output)
    out = model(x)
    print(len(captures))         # 1
    print(captures[0].shape)
    handle.remove()              # always remove

The signature: `hook(module, inputs, output)`. `inputs` is a tuple even for single-input modules. You can return a new value from a forward hook to _replace_ the output (rare but occasionally needed).

## Weight tying

Sometimes you want two parameters to _be the same tensor_. Classic case: in a transformer language model, the input embedding and the output projection often share weights — same matrix, two roles. The savings are huge (the embedding can be 30% of the model).

The naive (and wrong) approach:
    
    
    self.embed = nn.Embedding(vocab, dim)
    self.out_proj = nn.Linear(dim, vocab, bias=False)
    
    # Wrong: copies the data, doesn't tie
    self.out_proj.weight.copy_(self.embed.weight)

The right approach is to make the two attributes refer to the _same Parameter_ :
    
    
    self.embed = nn.Embedding(vocab, dim)
    self.out_proj = nn.Linear(dim, vocab, bias=False)
    self.out_proj.weight = self.embed.weight    # now both names point to one tensor

Now both names refer to one underlying tensor. Updates from one path affect the other. Memory is halved. The optimizer sees one parameter, not two (PyTorch dedupes by tensor identity in `parameters()`).

Two things to know:

  1. Do this _after_ both Modules exist. The assignment replaces `out_proj`'s parameter dict entry.
  2. `state_dict()` will only have one entry (whichever was first). On load, the same tying must be re-established before `load_state_dict` — or the loaded weights won't be propagated to both names.

## Lazy modules and `parametrize`: the convenient extras

Two newer APIs worth knowing about, briefly.

### Lazy modules

`nn.LazyLinear`, `nn.LazyConv2d`, etc. defer parameter shape determination until the first forward pass:
    
    
    layer = nn.LazyLinear(10)            # in_features unknown
    print(layer.weight)                # UninitializedParameter
    
    x = torch.randn(4, 128)
    y = layer(x)                       # first call infers in_features=128, materializes weights
    print(layer.weight.shape)          # torch.Size([10, 128])

Useful when you don't want to plumb shape information through your `__init__`. The catch: until materialized, the optimizer can't see the parameters, and `state_dict` entries don't exist. So you typically run a dummy forward pass before doing anything else.

### `nn.utils.parametrize`

Reparameterize a parameter as a function of an underlying tensor — useful for constraints (orthogonal matrices, weight normalization, low-rank updates):
    
    
    import torch.nn.utils.parametrize as P
    
    class Symmetric(nn.Module):
        def forward(self, X):
            return X.triu() + X.triu(1).transpose(-1, -2)
    
    linear = nn.Linear(5, 5)
    P.register_parametrization(linear, "weight", Symmetric())
    
    # Now linear.weight is always symmetric, computed from a smaller underlying tensor

Less common in everyday code, but the foundation of LoRA-style adapters and weight-norm reparameterization.

## Code Magnets: build a small Transformer block — correctly

You're putting together a tiny transformer block: a multi-head attention, a layer norm, an MLP. Layer counts and module choices matter. Build the `__init__` that registers everything correctly.

Arrange the magnets into a working `__init__`. Three are red herrings.

class Block(nn.Module): def __init__(self, dim, n_heads, n_mlp_layers): super().__init__() self.norm1 = nn.LayerNorm(dim) self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True) self.norm2 = nn.LayerNorm(dim) self.mlp_layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_mlp_layers)]) self.mlp_layers = [nn.Linear(dim, dim) for _ in range(n_mlp_layers)] self.causal_mask = torch.tril(torch.ones(1024, 1024)).bool() self.register_buffer('causal_mask', torch.tril(torch.ones(1024, 1024)).bool())

show solution
    
    
    class Block(nn.Module):
        def __init__(self, dim, n_heads, n_mlp_layers):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp_layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_mlp_layers)])
            self.register_buffer('causal_mask', torch.tril(torch.ones(1024, 1024)).bool())

The traps:

  * The plain `self.mlp_layers = [...]` is the canonical bug — looks fine, runs fine, but the layers aren't registered. Use `ModuleList`.
  * The plain `self.causal_mask = torch.tril(...)` doesn't get moved to GPU. Use `register_buffer` so it travels with the model.

Notice that `nn.LayerNorm`, `nn.MultiheadAttention`, and `nn.Linear` are all registered automatically just by being assigned to `self.x` — because they're `nn.Module` subclasses, `__setattr__` routes them to `_modules`. The traps are about types that _aren't_ Modules: lists and raw tensors.

## Who does what?

Match each Module concept to its purpose.

Concept

Purpose

nn.Parameter

A. Recursively flips a Boolean flag that some layers (Dropout, BN) check.

register_buffer

B. A Module-aware list. Children are visible to .parameters() and .cuda().

nn.ModuleList

C. Marks a tensor as trainable; appears in .parameters() and gets gradients.

state_dict()

D. An OrderedDict of fully-qualified names to parameter and buffer tensors.

model.train() / .eval()

E. Registers a non-trainable tensor as part of model state — moves with .cuda(), saved by default.

weight tying

F. Two attributes pointing to the same underlying Parameter. Halves memory; updates from either path affect both.

show solution

**nn.Parameter** → C  
**register_buffer** → E  
**nn.ModuleList** → B  
**state_dict()** → D  
**model.train() / .eval()** → A  
**weight tying** → F 

The mental shortcut: _Parameter is for trainable, buffer for non-trainable-but-stateful, ModuleList for collections of modules, state_dict is the persistence shape, train/eval is just a flag, weight tying is shared identity_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** Write a small Module that wraps a list of Linear layers. Test that all parameters are visible to `.parameters()`, that `.cuda()` moves all of them, and that `state_dict()` contains entries for each.

show answer
    
    
    class MLP(nn.Module):
        def __init__(self, dims):
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]
            )
    
        def forward(self, x):
            for layer in self.layers[:-1]:
                x = layer(x).relu()
            return self.layers[-1](x)
    
    m = MLP([10, 20, 20, 5])
    print(sum(p.numel() for p in m.parameters()))   # 645
    print(list(m.state_dict().keys()))
    # ['layers.0.weight', 'layers.0.bias', 'layers.1.weight', 'layers.1.bias',
    #  'layers.2.weight', 'layers.2.bias']

Try replacing `nn.ModuleList(...)` with a plain list `[...]` and re-run the prints. Both will collapse to `0` and `[]`.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** A colleague's model has BatchNorm layers and they want to "freeze" the encoder. They write `for p in encoder.parameters(): p.requires_grad = False`. Why is the model still slowly drifting during training, and what's the fix?

show answer

BatchNorm's running statistics are _buffers_ , not parameters. `requires_grad = False` only affects parameters — the BN running mean and variance still update during forward as long as the layer is in training mode. The fix: also call `encoder.eval()` to flip the training flag. Now BN uses (and stops updating) the running stats. For full freeze, you typically want _both_ the requires_grad flip and the eval mode.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** You have a checkpoint trained with `DistributedDataParallel`. Every key in the state_dict starts with `module.`. Load it into a non-DDP-wrapped model.

show answer
    
    
    sd = torch.load('ddp_checkpoint.pt', weights_only=True)
    sd = {k.removeprefix('module.'): v for k, v in sd.items()}
    model.load_state_dict(sd)

Or alternatively, wrap the model in DDP first and then load — that adds the prefix on the model side. The string-manipulation approach is more flexible and the standard pattern. Note `.removeprefix` is Python 3.9+; pre-3.9 you'd write `k[7:] if k.startswith('module.') else k`.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Implement weight tying between an embedding and an output projection. Verify that updating one name's weight tensor in place changes the other.

show answer
    
    
    class TiedLM(nn.Module):
        def __init__(self, vocab, dim):
            super().__init__()
            self.embed = nn.Embedding(vocab, dim)
            self.head = nn.Linear(dim, vocab, bias=False)
            self.head.weight = self.embed.weight    # tie
    
    m = TiedLM(100, 8)
    print(m.embed.weight.data_ptr() == m.head.weight.data_ptr())   # True
    
    m.embed.weight.data[0, 0] = 42.0
    print(m.head.weight[0, 0].item())                       # 42.0
    
    print(sum(p.numel() for p in m.parameters()))             # 800 (not 1600)

The dedup happens because `parameters()` tracks tensors by identity (Python `id()`), so the same Parameter referenced under two names is yielded only once. The optimizer therefore allocates one set of momentum buffers, not two. Free memory.

### What just happened?

  * An `nn.Module` is **three dictionaries plus a forward method** : `_parameters`, `_buffers`, `_modules`. Anything you want PyTorch to find must be in one of those.
  * `nn.Parameter` = trainable tensor (in `parameters()`, gets gradients, optimizer updates it).
  * `register_buffer` = stateful tensor that's not trained (BN running stats, masks, RoPE tables). In state_dict by default; set `persistent=False` if you don't want it saved.
  * The **registration gotcha** : `self.layers = [...]` is invisible. Use `nn.ModuleList`, `nn.ModuleDict`, or `nn.Sequential`.
  * `state_dict()` uses **dotted hierarchical keys** reflecting the module tree. Save the dict, not the model object — survives refactors.
  * `load_state_dict(sd, strict=False)` returns `(missing, unexpected)` for partial loads. Use it during fine-tuning.
  * `model.train()` / `.eval()` just **flips a Boolean**. It only matters because Dropout and BN check it.
  * **Three things commonly confused for "eval mode"** : training flag (Dropout/BN), `no_grad` (autograd off), `requires_grad=False` (parameter freeze). Inference wants the first two; freezing for fine-tuning wants the last one (often plus eval).
  * **Weight tying** : `self.head.weight = self.embed.weight`. Same tensor, two names. Halves memory and parameter count.
  * **DDP prefix** : state_dict keys from a DDP model have `module.` prepended. Strip with `removeprefix` when loading into an unwrapped model.
  * The reflex: when something registered "isn't there," ask _which dictionary should it have gone into, and did`__setattr__` route it correctly?_ The answer is almost always "you assigned a plain Python container or raw tensor."

Module 08 takes the container model you just built and fills it with the layers and initialization schemes that make a model trainable. We'll cover Xavier vs Kaiming vs GPT-style init, Dropout / BatchNorm / LayerNorm / RMSNorm / GroupNorm, and the activation/normalization choices that matter most.
