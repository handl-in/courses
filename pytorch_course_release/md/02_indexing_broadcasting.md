# Module 02 — Indexing, broadcasting & shape gymnastics

# Indexing, broadcasting & _shape gymnastics_

_Part I · Module 02_

— the rules behind every reshape, every weird shape error, and the `einsum` notation you've been avoiding

\--- 

If Module 01 was about what a tensor _is_ , this module is about how to _move it around_ : take slices, fold dimensions, broadcast, transpose, reshape. The bad news is that PyTorch has approximately fifteen ways to do each of these, with subtle differences. The good news is that most of them are dialects of three or four ideas, and once you can read the dialects you stop being afraid of any of them.

> **★ KEY IDEA**  
>  There are exactly **three rules** for how shapes interact: _broadcasting_ (one-sided dimension extension), _contraction_ (matrix multiply / einsum), and _reshape-or-permute_ (rearranging without changing total elements). Almost every shape error you'll ever see is one of these three rules being violated. 

## Meet the new players

This module brings two new characters onto the stage. They're going to do most of the work.

B

Broadcaster

"I right-align everything."

When you ask me to add two tensors of different shapes, I line them up on the _right_ — yes, the right, not the left. Then I check each pair of dimensions: if they match, fine. If one is 1, I stretch it (Stride 0!). If neither is 1 and they don't match, I yell. That's literally my whole job.

Σ

Einsum

"Just give me letters and I'll do the rest."

Name every axis with a letter. Tell me which letters you want in the output. Any letter that appears in the inputs but _not_ the output, I sum over. Any letter that appears in multiple inputs, I contract on. That's it. `'bhtd,bhsd->bhts'` isn't hieroglyphs — it's me saying "for each batch, head, query position, key position, sum over d." Easy.

## Indexing: the four kinds

You think you know indexing. PyTorch has four flavors and they each behave differently with respect to _views_ , _shape_ , and _autograd_. Naming them up front saves you a lot of debugging.

The four indexing flavors Flavor| Example| Returns| View?  
---|---|---|---  
**Basic (slicing)**| `x[2:5, :, ::2]`| Slice with regular stride| Yes  
**Integer single**| `x[3]` or `x[3, 7]`| Drops a dim| Yes  
**Boolean mask**| `x[x > 0]`| 1-D flattened| No (copy)  
**Fancy / advanced**| `x[[0,2,5]]` or `x[idx_tensor]`| Gathered values| No (copy)  
  
The key distinction: **basic indexing is a view, fancy indexing is a copy**. Why? Because basic indexing maps to a regular pattern (start + step × i), expressible as offset + stride. Fancy indexing means "go fetch elements in this arbitrary order" — there's no stride trick that does that, so PyTorch allocates and copies.
    
    
    x = torch.arange(20).reshape(4, 5)
    
    # Basic — view
    a = x[1:3]                         # shape (2,5), shares storage
    print(a.untyped_storage().data_ptr() == x.untyped_storage().data_ptr())  # True
    
    # Integer — view (drops a dim)
    b = x[2]                            # shape (5,), shares storage
    
    # Boolean mask — copy, flattens
    c = x[x > 10]                       # shape (9,) [numbers 11..19]
    
    # Fancy — copy, keeps the indexed dim's shape
    d = x[[0, 2, 3]]                    # shape (3, 5)
    e = x[[0, 2, 3], [1, 2, 4]]          # shape (3,) — pairs of indices

That last one is worth staring at. When you supply two index _tensors_ of the same shape, you're asking for _paired_ indexing: `(0,1), (2,2), (3,4)`. NumPy and PyTorch share this convention. It's not "rows 0,2,3 then columns 1,2,4" — it's "the elements at those (row,col) pairs."

### Newaxis and `None`: the dimension-adder

You'll see this constantly:
    
    
    x = torch.tensor([1, 2, 3])           # shape (3,)
    y = x[:, None]                       # shape (3, 1) — added axis
    z = x[None, :]                       # shape (1, 3) — added axis at the front
    w = x[None, :, None]                 # shape (1, 3, 1)

`None` is just `numpy.newaxis`. It inserts a size-1 dimension at that slot. Equivalent to `x.unsqueeze(0)` or `x.unsqueeze(-1)`. People prefer `None` in indexing expressions because it composes — `x[None, :, None]` is shorter than `x.unsqueeze(0).unsqueeze(-1)`.

> **🧠 BRAIN POWER**  
> 
> 
> **Quick test:** if `a` has shape `(5,)` and `b` has shape `(7,)`, what shape does `a[:, None] * b[None, :]` have?
> 
> (walk the broadcasting): `(5, 1) * (1, 7)` → broadcast → `(5, 7)`. This is the standard outer-product idiom in PyTorch. You'll write it dozens of times. Memorize it.

## The broadcasting rules, finally explained

Broadcasting is the source of more "but it should work!" frustration than any other PyTorch feature. The rules feel arbitrary until you internalize _why_ they were chosen, at which point they become obvious.

B

Broadcaster

"Two rules. That's all."

One: I right-align the shapes — pad the shorter one on the LEFT with implicit 1s. Two: for each pair of aligned dims, they must be equal OR one of them must be 1. If one is 1, I stretch it for free using Stride 0. Anything else, I throw a "size mismatch" error. That's literally everything I do.

Broadcasting: align RIGHT, then check each dim Example 1: (4, 1, 6) and (5, 6) a: 4 1 6 b: 1 5 6 → 4 5 6 b padded with 1 on left ✓ result: (4, 5, 6) Example 2: (3, 5) and (4, 5) — FAILS a: 3 5 b: 4 5 3 ≠ 4, neither is 1 → ERROR "The size of tensor a (3) must match the size of tensor b (4) at non-singleton dim 0"

The two rules:

  1. **Right-align the shapes.** Pad shorter shapes with 1s on the _left_.
  2. **For each aligned dimension, compatibility means equal OR one of them is 1.** If one is 1, it broadcasts (stride-zero trick from Module 01) up to the other.

Right-alignment is the rule everyone forgets. You think it'll match dimensions left-to-right because you read left-to-right. PyTorch (and NumPy) align right because the rightmost dimensions are typically the "innermost" — channels, embedding dims, the small numbers. Padding 1s on the left of the shorter shape lets a small tensor "broadcast over" the outer batch/spatial dimensions of a larger one.

**JUNIOR:** I have a batch of shape `(B, T, D)` and a positional embedding of shape `(T, D)`. `x + pos` works? **SENIOR:** Yes. `(B, T, D)` right-aligned with `(T, D)` pads to `(1, T, D)`. Then dim-0 is `(B, 1)` → broadcasts to `B`. Same positions used for every batch item. Standard transformer move. **JUNIOR:** And `(B, T, D) + (B, D)`? **SENIOR:** That breaks. Right-align: `(B, T, D)` vs `(_, B, D)`. Now dim-1 is `T` vs `B` — almost certainly not equal, neither is 1. PyTorch yells. You probably wanted `(B, T, D) + (B, 1, D)`, with an explicit `unsqueeze(1)` on the second tensor. 

### The "one in the middle" trick

Sometimes you want to broadcast over a dimension that isn't on the edge. The fix is always the same: `unsqueeze` a 1 in the right slot.
    
    
    # Tensor of shape (B, T, D), per-position scaling factor of shape (T,)
    x = torch.randn(2, 5, 8)
    scale = torch.linspace(0.1, 1.0, 5)        # shape (5,)
    
    # Wrong: (5,) right-aligns to (1, 1, 5) — that's per-D, not per-T
    y_wrong = x * scale          # wrong dimension!
    
    # Right: explicitly add a trailing 1 so (5,) becomes (5, 1) → (1, 5, 1)
    y = x * scale[:, None]               # scale shape (5, 1) → (1, 5, 1) → broadcasts

The general procedure: when you broadcast and the result is wrong, ask "what shape does my smaller tensor need to _look like_ after right-aligning?" Then explicitly `unsqueeze` or use `None` to make it that shape. PyTorch will broadcast correctly from there.

## view vs reshape vs permute vs transpose

Time to settle these once and for all.

Match each shape-changing operation to what it actually does.

Operation

What it does

x.view(new_shape)

A. Swaps two dimensions. Special case of permute. Result is non-contiguous.

x.reshape(new_shape)

B. Reorders all dimensions according to a permutation. Result is non-contiguous.

x.permute(*dims)

C. Reinterprets storage as new shape. ERRORS if not contiguous.

x.transpose(d1, d2)

D. Adds or removes a size-1 dimension. View only.

x.flatten(s, e)

E. Same as view, but COPIES if it has to. Always works (if total elements match).

x.squeeze() / unsqueeze()

F. Merges a range of consecutive dimensions into one. Calls reshape underneath.

show solution

**x.view(new_shape)** → C  
**x.reshape(new_shape)** → E  
**x.permute(*dims)** → B  
**x.transpose(d1, d2)** → A  
**x.flatten(s, e)** → F  
**x.squeeze() / unsqueeze()** → D 

The mental shortcut: _view is strict, reshape is permissive, permute reorders, transpose swaps two, flatten/unflatten are sugar over reshape, squeeze/unsqueeze fiddle with size-1 dims_.

The pattern most people get wrong: `permute` changes _which dimensions are which_ ; `view` changes _how to count elements_. They are not interchangeable. `x.view(C, B, T)` does _not_ swap dims; it reinterprets the flat buffer as if it were already in `(C, B, T)` order. If your tensor was actually `(B, T, C)`, you've now got garbage.
    
    
    x = torch.arange(24).reshape(2, 3, 4)        # (B=2, T=3, C=4)
    
    # I want to transpose to (C, T, B). Do NOT do:
    wrong = x.view(4, 3, 2)         # this is garbage — same buffer, wrong meaning
    
    # Do this:
    right = x.permute(2, 1, 0).contiguous()   # now bytes are in (C, T, B) order

> **⚠ WARNING**  
>  **The classic shape bug.** Reshape works on element _count_ , not on element _meaning_. Two tensors of shape `(2, 3, 4)` and `(4, 3, 2)` hold the same 24 numbers, but the second one is the first viewed in a totally different way. `view` and `reshape` will happily turn one into the other. The numbers will be wrong, training will diverge, and you'll spend a day finding out it was a missing `permute`. 

### The flatten/unflatten idiom

You'll see this all over transformer code:
    
    
    # In a transformer, attention scores have shape (B, H, T, T)
    # where H is heads. Sometimes you want to fold heads into the batch:
    
    scores = torch.randn(2, 4, 10, 10)
    flat = scores.flatten(0, 1)            # shape (8, 10, 10)
    
    # And to unfold:
    back = flat.unflatten(0, (2, 4))         # shape (2, 4, 10, 10)

This is just `reshape` under the hood, but the names are clearer. Use `flatten`/`unflatten` for "merge these adjacent dims" / "split this dim into these factors." Use `view`/`reshape` when the operation is more general.

## einsum: the universal solvent

If you've avoided `einsum` because it looks like Egyptian hieroglyphs, today is the day. It's the single most readable way to express tensor operations once you spend twenty minutes with it.

Σ

Einsum

"My algorithm in three lines."

(1) Name each axis of each input with a letter. (2) Name the output axes with letters too. (3) Any letter that appears in the inputs but _not_ the output, I sum over. That's the entire rule. `'mk,kn->mn'` is matmul. `'i,i->'` is dot product. `'bhtd,bhsd->bhts'` is attention scores. Same rule, every time.
    
    
    # Matrix multiply: (M, K) × (K, N) → (M, N)
    torch.einsum('mk,kn->mn', A, B)
    
    # Batched matmul: (B, M, K) × (B, K, N) → (B, M, N)
    torch.einsum('bmk,bkn->bmn', A, B)
    
    # Outer product: (M,) × (N,) → (M, N)
    torch.einsum('i,j->ij', a, b)
    
    # Element-wise multiply + sum (dot product): (N,) × (N,) → scalar
    torch.einsum('i,i->', a, b)
    
    # Trace of a matrix: sum of diagonal
    torch.einsum('ii->', M)
    
    # Transpose: (M, N) → (N, M)
    torch.einsum('ij->ji', M)
    
    # Multi-head attention scores: q (B,H,T,D) × k (B,H,T,D) → (B,H,T,T)
    scores = torch.einsum('bhtd,bhsd->bhts', q, k)

Read each one as: "letters appearing in any input but not the output are summed." That's the entire rule. It generalizes matmul, batched matmul, dot product, outer product, trace, transpose, and most of attention.

> **📝 NOTE**  
>  **einsum's secret superpower:** when you write `'bhtd,bhsd->bhts'`, the shape contract is _self-documenting_. Anyone reading the line knows the input shapes and the output shape. Compare to `q @ k.transpose(-1,-2)`, which leaves the reader to figure out the dims. Use einsum when the operation has more than two axes; use matmul/`@` when it's a plain matrix multiply. 

### einsum performance

One myth: einsum is slow. It used to be — early implementations didn't fuse well. Modern PyTorch's einsum dispatches to optimized matmul/bmm calls when possible, and `torch.compile` can fuse it further. For matmul-shaped einsums, you get the cuBLAS path. For exotic contractions, it falls back to a generic kernel that's slower but still vectorized.

The rule of thumb: write the operation in einsum first; if it's in a hot loop, profile it; if it's slow, rewrite as a matmul or use `torch.compile`. _Don't_ preemptively avoid einsum.

## Code Magnets: build attention from einsum pieces

Here's a real-world puzzle. You have `q`, `k`, `v`, all of shape `(B, H, T, D)`. Build the attention output `(B, H, T, D)` using einsum and a softmax. The full sequence is: scores = q·kᵀ, then softmax over the key dimension, then output = scores·v.

Arrange these magnets into a 3-line attention computation. Two are red herrings.

scores = torch.einsum('bhtd,bhsd->bhts', q, k) scores = torch.einsum('bhtd,bhds->bhts', q, k) attn = scores.softmax(dim=-1) attn = scores.softmax(dim=-2) out = torch.einsum('bhts,bhsd->bhtd', attn, v) out = torch.einsum('bhts,bhtd->bhsd', attn, v) scores = scores / (D ** 0.5)

show solution
    
    
    scores = torch.einsum('bhtd,bhsd->bhts', q, k)
    scores = scores / (D ** 0.5)
    attn = scores.softmax(dim=-1)
    out = torch.einsum('bhts,bhsd->bhtd', attn, v)

The traps:

  * `'bhtd,bhds->bhts'` would assume `k` is already transposed in its last two dims — but it's `(B,H,T,D)`, not `(B,H,D,T)`. The 's' has to align with k's T-axis, both labeled at the last D position.
  * `softmax(dim=-2)` normalizes over the query axis, not the key axis. Wrong distribution.
  * The output einsum needs to "consume" the s axis (key positions) and produce d (the value dimension). `'bhts,bhtd->bhsd'` doesn't even share an axis between the two operands — that's an outer product, not a contraction.

The scale factor `/ (D ** 0.5)` is included; it's the canonical scaled-dot-product attention.

## Reductions: the axis argument is everything

Every reduction in PyTorch — `sum`, `mean`, `max`, `argmax`, `std`, `any`, `norm` — takes a `dim` argument and a `keepdim` flag. Master both and you're 80% of the way through "PyTorch shape gymnastics."
    
    
    x = torch.randn(2, 3, 4)
    
    x.sum()                      # scalar — sums everything
    x.sum(dim=0)                 # shape (3, 4) — collapse dim 0
    x.sum(dim=1)                 # shape (2, 4) — collapse dim 1
    x.sum(dim=-1)                # shape (2, 3) — last dim
    x.sum(dim=(0, 2))            # shape (3,) — collapse multiple dims
    
    x.sum(dim=1, keepdim=True)    # shape (2, 1, 4) — keep dim with size 1

`keepdim=True` is the move that lets you broadcast the reduction back against the original tensor without fiddling with `unsqueeze`:
    
    
    # Mean-center each row of a (B, D) batch
    x = torch.randn(32, 128)
    mean = x.mean(dim=-1, keepdim=True)         # shape (32, 1)
    x_centered = x - mean                          # broadcasts cleanly
    
    # Without keepdim, you'd need:
    mean = x.mean(dim=-1)                       # shape (32,)
    x_centered = x - mean[:, None]                 # manual unsqueeze

Both work; the first is idiomatic. Norm computations, layer normalization, attention softmax — they all use `keepdim=True` for exactly this reason.

#### Q&A; — About reductions **Q:** What's the difference between `x.sum(0)` and `x.sum(dim=0)`? **A:** Nothing. The first arg of `sum` is `dim`. Same for `mean`, `max`, etc. People often skip the keyword name. Just be aware: in some functions, the first positional is something else (like `x.scatter(0, ...)` where `0` is the dim — but the rest of the args are different). **Q:** Why do `max` and `argmax` behave differently? `x.max(dim=0)` returns a tuple and `x.argmax(dim=0)` returns one tensor. **A:** Historical accident. `max` with `dim` returns _both_ values and indices, so you can do `vals, idx = x.max(dim=0)` in one call. `argmax` is the values-discarded shortcut. If you want only the values, you can use `x.max(dim=0).values` or `x.amax(dim=0)`. Modern PyTorch leans toward `amax`/`amin` for the values-only case. **Q:** When does `norm` blow up? **A:** When you compute `x.norm()` on a tensor with very large values, fp16/bf16 overflows during the squaring. Use `torch.linalg.vector_norm` with explicit dtype upcasting in mixed-precision code. We'll come back to this in Module 14. 

## Concatenation, stacking, splitting

Three operations, often confused.

Joining and splitting tensors Operation| What it does| Shape change  
---|---|---  
`torch.cat([a, b], dim=k)`| Concatenate along an existing dim| That dim grows; others identical.  
`torch.stack([a, b], dim=k)`| Add a NEW dim, stack tensors along it| Adds a dim of size N (number of inputs).  
`x.split(size, dim=k)`| Split into chunks of given size along dim| Returns a list/tuple.  
`x.chunk(n, dim=k)`| Split into n equal-ish chunks along dim| Returns a list.  
`x.unbind(dim=k)`| Remove a dim by splitting it into a tuple| List of N tensors with that dim gone.  
      
    
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    
    torch.cat([a, b], dim=0).shape    # (6, 4) — extends dim 0
    torch.stack([a, b], dim=0).shape  # (2, 3, 4) — adds new dim 0
    
    x = torch.randn(10, 5)
    x.split(3, dim=0)               # [(3,5), (3,5), (3,5), (1,5)] — last is short
    x.chunk(3, dim=0)               # [(4,5), (4,5), (2,5)] — n chunks, equal-ish

The mnemonic: **cat** is "concatenate" — like extending a list. **stack** is "make a stack" — like piling things on top of each other, which inherently adds a dimension. If you find yourself doing `torch.cat([a[None], b[None]], dim=0)`, that's `torch.stack([a, b], dim=0)`.

## gather and scatter: the index ops

Two operations that look weird at first but are workhorses for things like top-k selection, embedding lookup, and computing per-sample loss.

**gather** : read _specific indices along a dimension_. Where regular indexing reads with a single index per dim, gather reads with a tensor of indices the same shape as the output.
    
    
    # I have a batch of logits and per-batch target indices
    logits = torch.randn(4, 10)              # 4 examples, 10 classes
    targets = torch.tensor([3, 7, 1, 5])      # target index for each
    
    # I want logits[i, targets[i]] for each i — the logit for the correct class
    correct = logits.gather(1, targets[:, None]).squeeze(1)  # shape (4,)

Read this: "along dim 1, pick the indices given by `targets`." The index tensor must have the same number of dims as `logits`; we wrap `targets` in `[:, None]` to make it `(4, 1)`, telling gather "for batch `i`, pick element `targets[i]`."

**scatter** : the inverse — write specific indices along a dim.
    
    
    # Build a one-hot encoding from class indices
    classes = torch.tensor([2, 0, 3])            # shape (3,)
    onehot = torch.zeros(3, 5)
    onehot.scatter_(1, classes[:, None], 1.0)
    # onehot is now
    # [[0, 0, 1, 0, 0],
    #  [1, 0, 0, 0, 0],
    #  [0, 0, 0, 1, 0]]

Modern PyTorch has `F.one_hot` for this exact use case, but understanding scatter is worth it because dozens of advanced ops (e.g., MoE routing — Module 28) are scatter operations underneath.

> **📝 NOTE**  
>  **Whiteboard tattoo:** if you're writing a Python `for` loop to read elements from a tensor based on a list of indices, stop. The op you want is either `gather`, `index_select`, or fancy indexing. Almost any per-batch indexing has a vectorized form. 

### `index_select` and `masked_select`

Two more in the family worth knowing:
    
    
    # index_select — pick specific indices along ONE dim, keeps the dim's shape
    x = torch.arange(20).reshape(4, 5)
    y = x.index_select(0, torch.tensor([0, 2]))      # shape (2, 5)
    
    # Roughly equivalent to fancy indexing on a single dim:
    y = x[[0, 2]]                                # same thing
    
    # masked_select — pull out the elements where mask is True, ALWAYS flat
    m = x > 10
    y = x.masked_select(m)                          # shape (k,) where k = number of Trues
    
    # masked_fill — write a scalar where mask is True
    y = x.masked_fill(m, -1)                       # shape preserved, masked entries become -1

## Common shape-bug patterns and how to spot them

A short rogues' gallery.

### 1\. Forgot to squeeze after gather/index
    
    
    logits = torch.randn(4, 10)
    targets = torch.tensor([3, 7, 1, 5])
    correct = logits.gather(1, targets[:, None])     # shape (4, 1) — !
    
    # Now I take the mean and try to log it
    loss = -correct.mean()                          # works, but...
    
    # Later I want to compare to another (4,) tensor
    mismatch = correct - other                      # shape (4, 1) - (4,) → (4, 4)!

Solution: `.squeeze(1)` after gather, or `.squeeze(-1)`. Get into the habit.

### 2\. Off-by-one in the dim argument
    
    
    x = torch.randn(8, 3, 224, 224)             # NCHW
    mean_per_channel = x.mean(dim=0)              # wrong! averages over batch only
    
    # Want mean over batch AND spatial
    mean_per_channel = x.mean(dim=(0, 2, 3))         # shape (3,) — correct

When working with 4-D tensors, name your dims in comments: `# (B, C, H, W)`. Future-you will thank present-you.

### 3\. Broadcasting the wrong direction
    
    
    # I have logits (B, T, V) and lengths (B,) — want to mask out padding
    logits = torch.randn(4, 10, 100)
    lengths = torch.tensor([7, 3, 10, 5])
    mask = torch.arange(10) < lengths             # wrong! shape mismatch
    
    # torch.arange(10) is (10,); lengths is (4,). Right-align: (10,) vs (4,) → 10≠4, error.
    
    # Correct: make lengths broadcast against the position arange
    positions = torch.arange(10)                  # (10,)
    mask = positions[None, :] < lengths[:, None]  # (1, 10) < (4, 1) → (4, 10) ✓
    masked = logits.masked_fill(~mask[:, :, None], float('-inf'))

The `positions[None, :] < lengths[:, None]` idiom for building masks is so common in NLP that it's worth tattooing into memory.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** Given `x` of shape `(B, T, D)` and `w` of shape `(D, E)`, write the matmul `x @ w` three ways: with `@`, with `torch.matmul`, with `einsum`.

show answer
    
    
    y = x @ w                                  # shape (B, T, E)
    y = torch.matmul(x, w)                     # same
    y = torch.einsum('btd,de->bte', x, w)     # same

All three produce identical output. The einsum version is the most verbose but also the most readable when D appears in many places.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Given a batch `x` of shape `(B, T, D)`, compute the mean and variance per-position-per-feature across the batch (so output shapes are `(T, D)` each).

show answer
    
    
    mean = x.mean(dim=0)                     # (T, D)
    var = x.var(dim=0, unbiased=False)        # (T, D)

The `unbiased=False` uses divide by N (not N−1); typical for ML normalization where you want the population variance, not the sample variance.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** Given `x` of shape `(B, T, D)` and a _per-batch_ length tensor `lengths` of shape `(B,)`, build a mask that is True for positions `< lengths[i]` and False otherwise. The result should have shape `(B, T)`.

show answer
    
    
    positions = torch.arange(T, device=lengths.device)
    mask = positions[None, :] < lengths[:, None]   # (B, T)

This appears in every transformer's attention mask building. Memorize it.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Given a _(B, H, T, T)_ attention score matrix and a (T, T) causal mask of booleans (True where allowed), apply the mask using `masked_fill` with `-inf` where False.

show answer
    
    
    scores = scores.masked_fill(~causal_mask[None, None, :, :], float('-inf'))

The `~` flips the mask (False where allowed, True where to fill). The double `None` broadcasts the (T, T) mask over the (B, H) dims. Because we used `masked_fill` (out-of-place) instead of `masked_fill_`, autograd is happy.

> **✎ SHARPEN YOUR PENCIL** > > 

**5.** Implement multi-head attention's "split the embedding into heads" reshape. Given `x` of shape `(B, T, D)` with `D = H * d_head`, get to shape `(B, H, T, d_head)`.

show answer
    
    
    B, T, D = x.shape
    x = x.view(B, T, H, d_head)                  # (B, T, H, d_head)
    x = x.transpose(1, 2)                       # (B, H, T, d_head)

Note the `view`+`transpose` combo — first split the D dimension into H groups of d_head, then transpose to put H next to B. This is the canonical reshape in every attention implementation. The result is _non-contiguous_ ; a subsequent matmul handles it fine but a `.view()` would error.

### What just happened?

  * **Three rules govern shape interactions:** right-aligned broadcasting, contraction (matmul/einsum), and reshape-or-permute (preserving total elements). Every shape error is one of these being violated.
  * **Indexing has four flavors:** basic slicing (view), integer (view, drops a dim), boolean mask (copy, flat), fancy/advanced (copy, shape-preserving).
  * **Broadcasting right-aligns** and treats size-1 dims as flexible. Pad on the left with implicit 1s. The Broadcaster character has just two rules: align right, then check each dim is equal or 1.
  * `None` in a slice = `unsqueeze`. `x[:, None] * y[None, :]` is the outer-product idiom. Use it.
  * `view` rewrites stride strictly; `permute`/`transpose` swap stride values; `reshape` = view-or-copy. They're not interchangeable — `view(C, B, T)` does NOT swap dims.
  * **einsum is the readable contraction syntax.** Letters not in the output are summed. Letters in multiple inputs are contracted. The whole rule fits in one sentence.
  * `keepdim=True` on reductions makes broadcasting back trivial. Use it for normalizations.
  * **cat extends** an existing dim; **stack adds** a new dim. Don't write `torch.cat([a[None], b[None]], dim=0)` when you mean `torch.stack([a, b], dim=0)`.
  * **gather/scatter** read/write at index tensors. Use them whenever you'd write a Python loop over a batch. Same for `index_select`, `masked_select`, `masked_fill`.
  * The mask-building idiom `positions[None, :] < lengths[:, None]` appears in every NLP codebase. Tattoo it.

Module 03 will turn from _shapes_ to _numbers_ : the dtypes, devices, transfer mechanics, and numerical-stability primitives that make the difference between training that works and training that mysteriously NaNs at step 4000.
