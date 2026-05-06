# Vibe Preview — full Head First treatment

# Tensors, _from scratch_

_Vibe Preview · Module 01 (excerpt)_

— full Head First treatment: characters talking, magnets to arrange, matching games, and the bullet-points recap

\--- 

This is a preview of what Module 01 looks like with the full vibe upgrade. Same teaching content, more Head First personality. Compare to the original M1 to see the difference.

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

Five attributes. Five things you need to keep in your head:

storage (the bytes) 0 1 2 3 4 5 6 7 8 9 10 11 [0] [4] [8] row 0 → stride[0]=4 row 1 → +4 elements row 2 → +4 elements how YOU see it: shape (3, 4) with stride (4, 1) 0 1 2 3 4 5 6 7 8 9 10 11 same bytes, different recipe!

Look at the diagram. Storage on top is just bytes. The brackets show how Stride `(4, 1)` chops it into rows: jump 4 to start a new row, jump 1 to advance a column. The _matrix you see_ is just the bytes plus the recipe.

→

Stride, again

"Want to transpose? Just swap me."

Look — if you call `x.transpose(0, 1)`, none of the bytes move. PyTorch just hands you a new Tensor that points at the same Storage but with my values flipped: `(1, 4)` instead of `(4, 1)`. Now to advance one row, you jump 1; to advance one column, jump 4. Same data, different journey. _That's why transpose is free._

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

The red herring is `.reshape(4, 3)` — it would give you the wrong layout (numbers `0..11` in row-major, not the transposed pattern). The trick is that `reshape(3, 4)` first lays out as `[[0,1,2,3], [4,5,6,7], [8,9,10,11]]`, then `transpose(0, 1)` swaps stride to `(1, 4)` giving the desired output. Calling `.contiguous()` at the end is optional unless you need the bytes physically reordered for a downstream kernel.

## Who does what?

Now a matching exercise. Each operation on the left does _one_ of the things on the right. Match them up.

For each operation in the left column, identify what it does to memory.

Operation

What does it do?

x.view(-1)

A. Always allocates new storage and copies.

x.transpose(0, 1)

B. Rewrites stride, no memory touched. Errors if not contiguous.

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

## One more conversation

∂

Autograd (eavesdropping)

"Just a heads-up about views."

When Tensor and Storage chat about _views_ , I'm the one who has to deal with the consequences. If you mutate a view in-place — like `x[0].zero_()` — that's a write to the parent's Storage, and I might need its _old_ value for the backward pass. So I track version counters. If a Storage has been bumped, I refuse to compute the gradient for the stale value. We'll talk more in Module 4. For now, just know: _in-place writes through views are how I learn to mistrust you_.

### What just happened?

  * A **Tensor** is a small object that holds a reference to **Storage** (the actual bytes) plus a _recipe_ for reading it: **shape, stride, offset, dtype**.
  * **Storage doesn't know it's a matrix.** It's a flat byte buffer. Multiple Tensors can share the same Storage with different recipes — that's why `transpose` is free.
  * The lookup formula is always: `storage[offset + sum(index_i × stride_i)]`.
  * **Stride 0** is the broadcasting trick — the dimension lies, you read the same value many times.
  * `view` is strict (no copy ever, errors if it can't); `reshape` is permissive (copies when forced); `permute`/`transpose` just swap stride values.
  * An operation that mutates Storage (in-place ops, the underscore family) affects every Tensor that views it. **Autograd watches for this** via version counters.
  * The reflex: when something tensor-related surprises you, ask "what's the storage, what's the stride, what's the offset?" Nine times out of ten the answer is right there.

— end of vibe preview —
