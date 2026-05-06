# Module 01 — Tensors, from scratch

# Tensors, _from scratch_

_Part I · Module 01_

— what they actually are, why `.contiguous()` exists, and the "view vs copy" mental model that will save you a hundred bugs

\--- 

Most PyTorch tutorials begin with: _"a tensor is like a NumPy array but it can run on a GPU."_ That sentence is true. It is also useless. It tells you nothing about why your code crashed at 3am with a `view size is not compatible` error, or why `x.transpose(0,1).view(-1)` screams at you while `x.transpose(0,1).reshape(-1)` is fine.

So we're going to start one level deeper. We'll build a model in your head of what a tensor _actually is_ , in memory, and then everything else — broadcasting, views, contiguity, strides, advanced indexing — will fall out of that model with very little memorization.

> **★ KEY IDEA**  
>  A tensor is **five things** : a chunk of contiguous memory (the _storage_), plus a recipe for interpreting that chunk as a multi-dimensional array (_shape_ , _stride_ , _offset_ , _dtype_), plus the _device_ the chunk lives on. Most "weird tensor behavior" is unsurprising once you keep these five things in your head. 

## Meet the cast

Before we dissect a tensor, let's hear from the players themselves. Each one is going to play a role in this story, so it helps to put a face to the name.

S

Storage

"Hi, I'm a flat buffer of bytes."

I'm just a chunk of memory. I don't know whether I'm a 3×4 matrix or a 12-element vector. That's not my job. My job is to hold bytes. Talk to Shape if you want opinions.

T

Tensor

"I'm the face of the operation."

When you call `x.shape` or `x[2, 3]`, you're talking to me. But I don't store any of the actual numbers. I just hold a reference to Storage and a recipe for how to read it. Without my friends Shape, Stride, and Offset, I'd be useless.

→

Stride

"I'm the recipe."

If you want element `[2, 3]`, ask me where to look in Storage. I'll say: "go to byte `(2 × my_first_value + 3 × my_second_value) × dtype_size`." That's it. I'm the secret to why `transpose` is free. Change me and the same Storage looks completely different.

#

Dtype

"I tell you how big each number is."

Storage just sees bytes. I'm the one who says "every 4 of these bytes is one float32" or "every 8 is one int64." Change me and you reinterpret the same memory as different numbers. This is more dangerous than it sounds.

That's the cast. Storage holds the bytes. Tensor is the public face. Stride is the recipe for navigating Storage. Dtype tells you the unit. Now let's watch them work together.

## The five-thing model

Open a Python REPL — actually do this — and let's poke a tensor with a stick.
    
    
    import torch
    
    x = torch.arange(12).reshape(3, 4)
    print(x)
    # tensor([[ 0,  1,  2,  3],
    #         [ 4,  5,  6,  7],
    #         [ 8,  9, 10, 11]])
    
    print(x.shape)              # torch.Size([3, 4])
    print(x.stride())           # (4, 1)
    print(x.storage_offset())   # 0
    print(x.dtype)              # torch.int64
    print(x.device)             # cpu
    print(x.untyped_storage())  # 96 bytes (12 * 8)

Five attributes. Five things you need to keep in your head:

storage (the bytes) 0 1 2 3 4 5 6 7 8 9 10 11 [0] [4] [8] row 0 → stride[0]=4 row 1 → +4 elements row 2 → +4 elements how YOU see it: shape (3, 4) with stride (4, 1) 0 1 2 3 4 5 6 7 8 9 10 11 same bytes, different recipe!

Look at the diagram. Storage on top is just bytes. The brackets show how Stride `(4, 1)` chops it into rows: jump 4 to start a new row, jump 1 to advance a column. The _matrix you see_ is just the bytes plus the recipe.

  1. **Storage** — a flat buffer of 12 int64s sitting in CPU memory. 96 bytes, period. This buffer doesn't know it's a 3×4 matrix.
  2. **Shape** — `(3, 4)`. The dimensions you logically want.
  3. **Stride** — `(4, 1)`. To advance one row, jump 4 elements in storage. To advance one column, jump 1.
  4. **Offset** — 0. Start at element 0 of the storage.
  5. **Dtype** — `int64`. Each element is 8 bytes.

That's it. Everything PyTorch does to a tensor is some combination of (a) reading/writing the storage, (b) changing the shape/stride/offset to reinterpret the storage, or (c) allocating a new storage. The whole framework is built on this distinction.

### Why strides are the secret to everything

Here is the trick that makes PyTorch fast. Most reshape-y operations don't touch memory at all — they just rewrite the stride. Watch:
    
    
    y = x.transpose(0, 1)            # swap rows and columns
    print(y)
    # tensor([[ 0,  4,  8],
    #         [ 1,  5,  9],
    #         [ 2,  6, 10],
    #         [ 3,  7, 11]])
    
    print(y.shape)               # torch.Size([4, 3])
    print(y.stride())            # (1, 4)   ← swapped!
    print(y.untyped_storage().data_ptr() == x.untyped_storage().data_ptr())  # True

→

Stride, again

"Want to transpose? Just swap me."

Look — none of the bytes moved. PyTorch just handed you a new Tensor pointing at the same Storage, but with my values flipped: `(1, 4)` instead of `(4, 1)`. Now to advance one row you jump 1; to advance one column, jump 4. Same data, different journey. _That's why transpose is free._

> **🧠 BRAIN POWER**  
> 
> 
> **Pause.** Without running anything: if I do `y[0, 1]`, what storage offset does PyTorch read?
> 
> (stride is `(1, 4)`, offset is 0, indexing is row 0, col 1) → _0·1 + 1·4 = 4_. So it reads `storage[4]`, which is the value 4. That matches `y[0,1] == 4`. ✓
> 
> That tiny computation — `offset + sum(index_i × stride_i)` — is the heart of every tensor read in PyTorch.

## View vs copy: the bug factory

Operations that _just rewrite stride/shape/offset_ are called **views**. They share storage with the parent. Operations that _allocate new storage and copy_ are called **copies**. Knowing which is which prevents two classes of bugs: surprise mutations (you wrote to a view and changed someone else's data) and surprise crashes (you tried to take a view that's mathematically impossible).

Match each operation to what it does to memory.

Operation

What does it do?

x.view(-1)

A. Always allocates new storage and copies.

x.transpose(0, 1)

B. Rewrites stride only — no memory touched. Errors if not contiguous.

x.clone()

C. Returns a view with stride 0 in the broadcast dimensions.

x.reshape(-1)

D. Swaps two stride values. No bytes move. Result is non-contiguous.

x.expand(3, 4)

E. Tries to make a view; copies if it has to.

x[2:5]

F. Shifts the storage offset and creates a view.

show solution

**x.view(-1)** → B (rewrites stride, errors if non-contiguous)  
**x.transpose(0, 1)** → D (swaps stride values, no bytes move)  
**x.clone()** → A (always allocates and copies)  
**x.reshape(-1)** → E (view if possible, copy if necessary)  
**x.expand(3, 4)** → C (stride 0 in broadcast dims)  
**x[2:5]** → F (shifts offset, view) 

If you got _view_ and _reshape_ mixed up: the difference is that `view` is strict — it refuses to copy. `reshape` is permissive — it'll silently copy when needed. Same outcome on contiguous tensors; very different on transposed ones.

Here's the surprise-mutation trap, lived through by every PyTorch user at some point:
    
    
    weights = torch.randn(4, 4)
    first_row = weights[0]            # view!
    first_row.zero_()                # in-place
    print(weights[0])             # tensor([0., 0., 0., 0.]) ← weights got nuked

If you wanted an independent copy, you needed `weights[0].clone()`. The leading rule of thumb: _any tensor you got from slicing or reshaping is probably a view; if you mean to mutate it, mutate it on purpose._

> **⚠ WARNING**  
>  **Common foot-gun:** the in-place ops (`add_`, `mul_`, `zero_`, `copy_`, etc., the trailing-underscore family) write through views. So `x.view(-1).zero_()` zeros _all of x_. This is a feature, not a bug — but it's a feature that has chewed off many fingers. 

## Contiguity: the secret rule of `view`

Why does `x.transpose(0,1).view(-1)` raise _"view size is not compatible with input tensor's size and stride"_? Because `view` requires a single contiguous chunk of storage that matches the requested shape. After a transpose, the elements you'd need to walk through in order to read row-by-row are _not_ in linear order in storage:
    
    
    x  shape (3,4) stride (4,1):
       reading row-major: storage[0], [1], [2], [3], [4], [5]...      ✓ linear
    
    x.transpose(0,1)  shape (4,3) stride (1,4):
       reading row-major: storage[0], [4], [8], [1], [5], [9]...      ✗ jumpy

A tensor is _contiguous_ when reading it in row-major order matches the linear order in storage. Formally: `stride[i] == prod(shape[i+1:])` for the row-major (C-order) case. Transposes break that. You have two options:
    
    
    # Option A: view requires contiguity, so make it so
    y = x.transpose(0, 1).contiguous().view(-1)
    
    # Option B: reshape will quietly copy if needed
    y = x.transpose(0, 1).reshape(-1)

People often ask "why doesn't `view` just call `contiguous` for me?" Because hidden copies kill performance. `view` is the strict version: it promises zero memory traffic. If you want the friendly version, ask for `reshape`. The framework is being honest with you.

## Code Magnets

Time to test your understanding. Below is a tray of code magnets. Your job: arrange them in your head (or on paper) to produce a tensor whose storage is the numbers `0..11` but whose _shape_ is `(4, 3)` in transposed-looking order — that is, when you print it, you see:
    
    
    tensor([[ 0,  4,  8],
            [ 1,  5,  9],
            [ 2,  6, 10],
            [ 3,  7, 11]])

Drag these magnets (mentally) into a single Python statement. Use exactly four of them; one is a red herring.

torch.arange(12) .reshape(3, 4) .reshape(4, 3) .transpose(0, 1) .contiguous()

show solution

The arrangement that works:
    
    
    torch.arange(12).reshape(3, 4).transpose(0, 1).contiguous()

The red herring is `.reshape(4, 3)` — it would give you the wrong layout (numbers `0..11` in row-major, not the transposed pattern). The trick is that `reshape(3, 4)` first lays out as `[[0,1,2,3], [4,5,6,7], [8,9,10,11]]`, then `transpose(0, 1)` swaps stride to `(1, 4)` giving the desired output. `.contiguous()` at the end is optional unless you need the bytes physically reordered for a downstream kernel.

#### Q&A; — About contiguity **Q:** Should I just always call `.contiguous()` to be safe? **A:** No. `.contiguous()` is a copy if the tensor isn't already contiguous, and the cost scales with tensor size. In a tight inner loop on a 1-billion-element tensor, that's serious. Use `reshape` when you'd otherwise have written `contiguous().view(...)`; let PyTorch decide. **Q:** Why does the error message say "size and stride"? My size _fits_. **A:** Because `view` doesn't just check element count. It checks whether you can _walk the new shape_ without leaving storage in row-major order. After a transpose, you can't. The error is technically correct but the wording trips everyone up. **Q:** Is there a way to ask "is this a view of that?" **A:** There's no public API for "view of," but you can check storage identity: `a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()`. If those match, they share storage. They might still be looking at non-overlapping regions, but you've at least narrowed it down. 

## Stride 0: the broadcasting trick

Now for a beautiful piece of engineering. What does `x.expand(3, 4)` do when `x` is a `(1, 4)` tensor?
    
    
    x = torch.tensor([[10, 20, 30, 40]])      # shape (1, 4)
    y = x.expand(3, 4)
    print(y)
    # tensor([[10, 20, 30, 40],
    #         [10, 20, 30, 40],
    #         [10, 20, 30, 40]])
    
    print(y.shape)        # torch.Size([3, 4])
    print(y.stride())     # (0, 1)   ← zero!

Stride zero. To advance one row, jump _zero_ elements in storage. The matrix lies. There aren't 12 elements; there are still only 4. The view just keeps re-reading them. This is how broadcasting works under the hood — duplication is virtual, free, and lazy.

This is also why you sometimes see "broadcasting was much faster than I expected" results. There's no copy; you're just reading the same cache lines several times. Hardware loves that.

> **📝 NOTE**  
>  **Whiteboard tattoo:** stride 0 = "this dimension lies; it's a single value pretending to be many." If you ever see `(0, 1, 0)` in a stride, broadcasting happened. It's not a bug. 

## Creating tensors: the menu

You will create tensors a thousand different ways across your career. Here is the menu in roughly the order you'll need it.
    
    
    # From Python data
    torch.tensor([1, 2, 3])              # infers dtype (int64 here)
    torch.tensor([1.0, 2.0, 3.0])          # float32
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    
    # Shape-based factories
    torch.zeros(3, 4)
    torch.ones(3, 4)
    torch.empty(3, 4)                    # UNINITIALIZED — garbage values, fast
    torch.full((3, 4), 3.14)
    torch.eye(5)                          # identity matrix
    
    # Ranges
    torch.arange(10)                       # 0..9
    torch.arange(2, 10, 2)                  # 2,4,6,8
    torch.linspace(0, 1, 100)               # 100 evenly spaced values
    
    # Random
    torch.rand(3, 4)                      # uniform [0, 1)
    torch.randn(3, 4)                     # standard normal
    torch.randint(0, 10, (3, 4))            # integers in [0, 10)
    
    # Match an existing tensor's shape/dtype/device
    torch.zeros_like(x)
    torch.randn_like(x)
    torch.full_like(x, 7)

The `_like` family is underused. When you're allocating intermediate buffers in a model, `torch.zeros_like(x)` is almost always more correct than `torch.zeros(x.shape)` — the first respects dtype _and_ device automatically. The second silently allocates on CPU even if `x` is on GPU.

> **⚠ WARNING**  
>  **`torch.empty` gives you garbage.** It's faster than `zeros` because it skips the memset, but the contents are whatever was previously in that memory. It's correct only when you're about to overwrite every element. Beginners sometimes train models for hours wondering why loss is NaN; they used `empty` as if it were `zeros` and the optimizer choked on uninitialized values. 

## Dtypes you actually use

PyTorch has more dtypes than you'll ever need. The ones that matter:

The dtypes that earn their keep Dtype| Bytes| Use it for…  
---|---|---  
`torch.float32` (default)| 4| The safe choice for training. Models train, gradients are stable.  
`torch.float16` (half)| 2| Speed/memory wins on GPU, but narrow range — overflows easily.  
`torch.bfloat16`| 2| Same range as fp32, less precision. Usually beats fp16 for training.  
`torch.float64`| 8| Scientific computing, gradcheck. Almost never for training.  
`torch.int64` (long, default int)| 8| Indices, labels.  
`torch.int32` / `int8`| 4 / 1| Quantization, packed embeddings.  
`torch.bool`| 1| Masks. _Not_ packed bits — each is a byte.  
  
Module 03 will dig into fp16 vs bf16 properly — for now, just internalize that **fp32 is the default and you should not change it without thinking**.

## Devices: cpu, cuda, mps

Every tensor lives on exactly one device. Operations between tensors require them on the same device — or you get the world's most explicit error.
    
    
    x = torch.randn(3, 4)
    y = x.to('cuda')            # or x.cuda(), or x.to(0)
    z = x + y                  # RuntimeError: Expected all tensors to be on same device
    
    # Best practice: pick a device once, propagate it
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(3, 4, device=device)
    y = torch.zeros_like(x)            # inherits device, dtype

The `.to()` method is the universal mover — it handles device, dtype, or both: `x.to(device='cuda', dtype=torch.float16)`. It's smart enough to be a no-op if the tensor is already where you asked.

One subtlety: `.to()` on a CPU→GPU move is asynchronous if non-blocking is enabled and the source is in pinned memory. We'll get to that in Module 03 and Module 13. For now, just know `x.cuda()` and `x.to('cuda')` are the same and you can use either.

## Mutating tensors: the in-place rules

PyTorch has two flavors of most operations: out-of-place (returns a new tensor) and in-place (modifies the tensor, marked with a trailing underscore). They look almost identical:
    
    
    x = torch.tensor([1.0, 2.0, 3.0])
    
    y = x.add(1.0)         # out-of-place: y is new, x unchanged
    x.add_(1.0)            # in-place: x mutated, returns x
    
    x[0] = 100             # also in-place
    x.copy_(y)             # also in-place — copies y's data into x's storage

In-place ops save memory (no new allocation), but they have two costs:

  1. **Autograd hates them.** If a tensor is needed for the backward pass and you mutate it in place, PyTorch may either error or compute wrong gradients. The framework tries to detect this with version counters, but the rule is: don't mutate intermediate tensors during forward.
  2. **They write through views.** If you got the tensor from slicing, mutating it changes the parent. Sometimes this is what you want; often it isn't.

∂

Autograd (eavesdropping)

"Just a heads-up about views."

When Tensor and Storage chat about _views_ , I'm the one who has to deal with the consequences. If you mutate a view in-place — like `x[0].zero_()` — that's a write to the parent's Storage, and I might need its _old_ value for the backward pass. So I track version counters. If a Storage has been bumped, I refuse to compute the gradient for the stale value. We'll talk more in Module 4. For now, just know: _in-place writes through views are how I learn to mistrust you_.

**JUNIOR:** I want to set all the negatives in `x` to zero. `x[x < 0] = 0`? **SENIOR:** That works. It's an in-place write through a boolean mask. Just be aware: if `x` needs gradients, that's an in-place op and you might get yelled at by autograd. **JUNIOR:** Alternative? **SENIOR:** `x = x.clamp(min=0)`. Out-of-place, autograd-friendly, single line. `F.relu(x)` or `x.relu()` also. 

## Putting the model together: a worked debug

Here's the kind of bug you'll spot in 30 seconds once you have the five-thing model.
    
    
    x = torch.randn(8, 3, 224, 224)              # NCHW image batch
    x_nhwc = x.permute(0, 2, 3, 1)               # to channels-last
    
    # Pass to a layer that expects channels-last bytes contiguous
    out = some_kernel(x_nhwc)
    # Throws: "expected contiguous input"

The _shape_ of `x_nhwc` is right — `(8, 224, 224, 3)`, channels-last. But the _stride_ is still the old (NCHW) stride, just permuted. The bytes in memory are not in NHWC order. You either need `x_nhwc.contiguous()`, or use the proper memory format API: `x.contiguous(memory_format=torch.channels_last)`, which lays out the bytes for real.

Notice how the bug isn't about _logic_. The math is correct. The bug is that you assumed permute physically reorders memory, and you're now feeding a kernel a buffer it doesn't recognize. The five-thing model spots this immediately.

## Tiny exercises that bake the model in

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** What's the stride of `torch.zeros(2, 3, 5)` in row-major order? Compute it before checking.

show answer

For row-major (C-order), `stride[i] = prod(shape[i+1:])`. So shape `(2, 3, 5)` gives stride `(15, 5, 1)`: advance one in dim-0 means skipping a whole 3×5=15 plane; one in dim-1 skips 5; one in dim-2 skips 1.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** I do `x = torch.arange(24).reshape(2, 3, 4); y = x.permute(2, 0, 1)`. What are `y.shape` and `y.stride()`?

show answer

Shape `(4, 2, 3)`. Original stride was `(12, 4, 1)`. Permuting `(2, 0, 1)` reorders the strides the same way: `(1, 12, 4)`. Verify mentally: to advance one in _new dim 0_ (which was old dim 2), jump 1; to advance one in new dim 1 (old dim 0), jump 12; new dim 2 (old dim 1) jumps 4.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** I have `x = torch.randn(3, 4)`. I want to add a fifth column of zeros to get a (3, 5) tensor. View, reshape, or new allocation?

show answer

**New allocation.** No combination of stride/shape/offset can describe a tensor whose last column is zero while the rest are `x`'s data — the zeros aren't in `x`'s storage. You need `torch.cat([x, torch.zeros(3, 1)], dim=1)` or to allocate a `(3,5)` and copy. Spotting "is the data already there?" is the right reflex.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Write a one-liner to test whether two tensors share storage.

show answer
    
    
    def share_storage(a, b):
        return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()

Useful for debugging "did this op make a copy?" suspicions. Note: same storage doesn't mean fully overlapping — two slices of the same buffer share storage but might not overlap.

### What just happened?

  * A **Tensor** is a small object that holds a reference to **Storage** (the actual bytes) plus a recipe for reading it: **shape, stride, offset, dtype**. Plus the device the bytes live on.
  * **Storage doesn't know it's a matrix.** It's a flat byte buffer. Multiple Tensors can share the same Storage with different recipes — that's why `transpose` is free.
  * The lookup formula is always: `storage[offset + sum(index_i × stride_i)]`. Memorize this; it's the heart of every read.
  * **Stride 0** is the broadcasting trick — the dimension lies, you read the same value many times. `x.expand(...)` uses it.
  * `view` is strict (no copy ever, errors if it can't); `reshape` is permissive (copies when forced); `permute`/`transpose` just swap stride values.
  * A tensor is **contiguous** when row-major reading order matches storage order. `.contiguous()` physically re-lays out memory if needed.
  * An operation that mutates Storage (in-place ops, the underscore family) affects every Tensor that views it. **Autograd watches for this** via version counters.
  * `torch.empty` returns garbage memory — fast but dangerous. Use `zeros` / `ones` when content matters.
  * Use `_like` factories (`torch.zeros_like(x)`) to inherit dtype _and_ device automatically.
  * The reflex: when something tensor-related surprises you, ask "what's the storage, what's the stride, what's the offset?" Nine times out of ten the answer is right there.

In Module 02 we'll get serious about indexing, broadcasting, and shape gymnastics — once you have the storage/stride model, those topics stop being magic and become rules.
