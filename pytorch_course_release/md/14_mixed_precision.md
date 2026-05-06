# Module 14 — Mixed precision: autocast & GradScaler

# Mixed precision: _autocast & GradScaler_

_Part V · Module 14_

— what dtype lives where in a training step, why bf16 is "free" but fp16 needs a loss scaler, and the modern recipe that's the silent default of every transformer codebase

\--- 

Five modules ago we met Float32, Float16, and BFloat16 as personalities. Module 9 reminded us that the optimizer keeps Adam state in fp32. Module 12 said "you keep both a bf16 working copy and an fp32 master copy of every parameter." Module 13 said "mixed precision is the first optimization to try." Now we wire it all together.

The headline: a modern training step doesn't run in _one_ dtype — it runs in a carefully chosen mix. The forward pass uses bf16 (or fp16) for matmul speed and memory. The loss is computed in fp32 for stability. Gradients are stored in fp32. The optimizer updates an fp32 master copy. Then the result is cast back to bf16 for the next forward. Each piece is in the precision where it works best, and PyTorch wires it up for you with two APIs: `torch.autocast` and (for fp16) `torch.amp.GradScaler`.

> **★ KEY IDEA**  
>  Mixed precision is a **per-region dtype contract** : _matmul-heavy ops in bf16/fp16_ (fast tensor cores, half the memory), _numerically-sensitive ops in fp32_ (loss reductions, softmax, layer norm), _all gradients in fp32_ (precision matters for Adam), and _parameters kept in both_ bf16 (working copy) and fp32 (master copy for the optimizer). `autocast` handles the per-op routing; `GradScaler` handles the fp16-specific underflow problem. bf16 doesn't need the scaler. That's the whole story. 

## One new face

×k

GradScaler

"I solve fp16's tiny-gradient problem by multiplying everything by a big number."

When you train in fp16, lots of gradient values are smaller than fp16's smallest representable number (~6e-5) and underflow to zero. That's training death by silent loss of signal. My job: scale the loss by a big constant K (typically 65536) before backward. Gradients come out K times larger, safely inside fp16's representable range. Then I unscale them before the optimizer step. I also detect overflow (any inf/nan) and skip the step + halve K when it happens. **bf16 has the same range as fp32, so it doesn't underflow — and doesn't need me.**

## What dtype lives where

The single most important diagram in this module:

A mixed-precision training step: the dtype at each point stored across whole run: params (bf16) used in forward master (fp32) read by optimizer grads (fp32) cleared each step Adam m,v (fp32) precision matters one training step: forward bf16 matmuls norms/softmax cast up to fp32 loss (fp32) CE/logsumexp backward grads → fp32 optimizer.step all in fp32 cast → bf16 for next forward "autocast" handles forward routing automatically — you don't write the casts grads come back in fp32 because the framework upcast them on output optimizer reads master fp32 + grad fp32 → updates master → casts to bf16 total cost: 18 bytes/param · total speed: ~2× free

Three things to internalize. (1) **The forward pass is not all bf16** — autocast routes individual ops to whichever precision is appropriate. Matmuls go to bf16 (fast tensor cores). Reductions, normalizations, and softmax stay in fp32 (numerically sensitive). (2) **Gradients always come back as fp32** , regardless of forward precision. The framework upcasts at the boundary so your optimizer sees consistent fp32 grads. (3) **The master fp32 copy** is what the optimizer updates. Without it, the bf16 weights would round away tiny optimizer updates and training would stall.

## `torch.autocast`: the per-op router

The minimal API:
    
    
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(x)
        loss = F.cross_entropy(output, y)

Inside the `with` block, autocast looks at every op being dispatched and decides:

  * **Cast inputs to bf16** for ops that benefit and are stable: matmul, conv, linear, attention.
  * **Keep in fp32** for ops where bf16 hurts: `logsumexp`, `softmax`, `log`, `exp`, layer norm reductions, loss functions.
  * **Pass through unchanged** for ops that are neutral.

The result: matmuls run on tensor cores at bf16 throughput; the few ops where bf16 would lose precision automatically run in fp32. You don't think about it.

### The autocast op list (you don't have to memorize this)

PyTorch maintains a list of which ops cast which way. Three categories:

How autocast routes ops (abbreviated) Category| Treatment| Examples  
---|---|---  
**Lower precision**|  cast inputs to bf16/fp16| `matmul`, `linear`, `conv2d`, `scaled_dot_product_attention`  
**fp32-preserving**|  cast inputs to fp32 if they're lower| `softmax`, `logsumexp`, `layer_norm`'s reduction parts, `cross_entropy`  
**Promotes inputs**|  finds the highest-precision input and uses that| elementwise add/sub when mixing dtypes, comparisons  
  
The first category is where the speedup comes from — tensor cores in bf16/fp16 are 2-4× faster than fp32 on Ampere/Hopper hardware. The second category is the safety net — the ops that would otherwise lose meaningful precision in low precision. The third is consistency.

> **⚠ WARNING**  
>  **Don't manually`.to(torch.bfloat16)` inside an autocast block.** You'll fight the autocast logic. Either trust autocast for everything (the standard recipe) or turn it off and manually manage dtypes (rarely worth it). Mixing the two leads to hard-to-debug type promotion bugs. 

### Where to put the autocast block
    
    
    for step, batch in enumerate(loader):
        x, y = batch
        opt.zero_grad()
    
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
            loss = F.cross_entropy(out, y)
    
        loss.backward()                                     # OUTSIDE the autocast — autograd handles dtype
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()                                         # OUTSIDE — runs in fp32 master
        scheduler.step()

The pattern: **wrap the forward and loss in`autocast`; leave backward and step outside**. Autograd already records the dtype at forward time and handles the casting in backward correctly. The optimizer step uses the master fp32 copy, which is fp32 by construction.

## bf16 vs fp16: the decision

You met the dtype layouts in M3. Here's the decision in one sentence: **use bf16 if your hardware supports it (Ampere / A100 / H100 / TPUs / MI200+), fp16 only if you're stuck on older hardware (Pascal/Volta/T4)**.

bf16 vs fp16 for training | bf16| fp16  
---|---|---  
Range (max value)| ~3.4e38 (same as fp32)| **65,504** (overflows easily)  
Underflow (min positive)| ~1.2e-38| ~6e-5 (gradients underflow)  
Mantissa precision| 7 bits| 10 bits  
Need GradScaler?| **No**| **Yes** (mandatory)  
Speed on tensor cores| Same as fp16| Same as bf16  
Hardware support| Ampere+ (A100/H100), TPUs| Anything from Pascal+  
Recommended for training| **Yes — modern default**|  Only on older hardware  
  
Two things bear repeating:

  1. **Tensor core speed is the same for bf16 and fp16** on supported hardware. There's no speed reason to prefer one over the other.
  2. **The numerical reason favors bf16 strongly.** Same exponent range as fp32 (no overflow, no underflow at gradient scale). The 3-bit mantissa difference vs fp16 doesn't matter in deep learning where activations have noise from regularization and stochastic optimization.

If you're on H100 or newer, you can also use fp8 for some operations (more on this at the end of the module). But for the next few years, bf16 is the workhorse.

## `GradScaler`: the fp16 safety net

Skip this section if you only train in bf16 — you don't need GradScaler. Read it if (a) you train on Volta/Pascal/T4 or (b) you want to understand why bf16 won.

The problem: fp16's smallest positive value is ~6e-5. Many gradients in deep learning are smaller than that — especially in deep networks where chain-rule shrinks gradients with depth. They _underflow to zero_. The model stops learning in those parameters.

The trick: scale the loss before backward. Since gradients are linear in the loss (chain rule), multiplying loss by K multiplies every gradient by K. Pick K large enough that all the meaningful gradients are above 6e-5. Then unscale before the optimizer step.
    
    
    scaler = torch.amp.GradScaler('cuda')              # the modern API
    
    for step, batch in enumerate(loader):
        x, y = batch
        opt.zero_grad()
    
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(x)
            loss = F.cross_entropy(out, y)
    
        scaler.scale(loss).backward()                  # scales loss by K, backward uses scaled grads
        scaler.unscale_(opt)                            # divides grads by K so clipping sees real magnitudes
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)                                # applies the step (unscale already done)
        scaler.update()                                  # adjusts K for next iteration

What's happening:

  1. `scaler.scale(loss)` multiplies the loss by K (initially 65536).
  2. `backward()` on the scaled loss produces gradients that are K times larger — safely inside fp16's range.
  3. `scaler.unscale_(opt)` divides the gradients by K so the optimizer sees the true magnitudes (and so gradient clipping works on real values).
  4. `scaler.step(opt)` calls the optimizer's step. **If any gradient is inf/nan** (overflow), it skips the step entirely.
  5. `scaler.update()` adjusts K. If overflow happened, halve K. If many steps without overflow, double K. K converges to the largest value that doesn't overflow.

The dynamic-scaling part is what makes GradScaler magical — you don't manually tune K. It auto-tunes to the largest value the current model can handle, which changes as training progresses.

> **⚠ WARNING**  
>  **The unscale-then-clip ordering matters.** If you clip gradients _before_ unscaling, you'd be clipping inflated values — the clip threshold wouldn't be meaningful. Always: `unscale_` first, then `clip_grad_norm_`, then `scaler.step`. Get the order wrong and clipping becomes a no-op. 

### Why bf16 doesn't need GradScaler

Two reasons. (1) bf16's smallest positive value is ~1.2e-38 — gradients don't underflow. (2) bf16's max is ~3.4e38 — gradients don't overflow either. The whole problem GradScaler exists to solve doesn't apply to bf16.

Practically: with bf16, you write the simpler training loop with no `scaler`, and it just works. This is a substantial simplicity win and one of several reasons bf16 became the default.

## TF32: the surprise extra format

One more dtype to know about. On Ampere and newer GPUs, when you do an fp32 matmul, the hardware doesn't actually use full fp32 internally — it uses **TF32** : 8-bit exponent (same as fp32), 10-bit mantissa (same as fp16), 19 bits total. Roughly 2× the throughput of true fp32.

This is on by default in PyTorch. You probably benefit from it without realizing. The relevant flags:
    
    
    torch.backends.cuda.matmul.allow_tf32 = True     # default True on Ampere+
    torch.backends.cudnn.allow_tf32 = True            # default True on Ampere+
    
    # Disable for bit-exact fp32 (rare):
    torch.backends.cuda.matmul.allow_tf32 = False

Why does this matter? Because if you write code that's "definitely fp32 throughout" (no autocast), you might still be silently using TF32. For most workloads this is fine — TF32's precision is similar to bf16 for the matmul itself, while keeping fp32 dynamic range. For some scientific workloads, you want true fp32 — turn TF32 off explicitly.

## Common NaN sources in mixed precision

bf16 is mostly NaN-proof. fp16 is not. If your loss goes NaN, walk through this list:

NaN diagnosis tree Symptom| Likely cause| Fix  
---|---|---  
Loss NaN at step 1| Bad init or huge LR| Check init scheme (M8); halve LR; warmup harder  
Loss NaN after some steps in fp16| GradScaler not used| Wrap in scaler.scale → backward → step pattern  
Loss NaN intermittently in fp16| Activations spiking past 65504| Lower LR; check for un-clipped activations; switch to bf16  
Loss NaN in bf16| Numerical instability — softmax overflow, log of 0, etc.| Use stable variants (logsumexp, F.cross_entropy from logits); add small ε  
Loss NaN from a custom op| Custom backward in fp16/bf16 hits numerical edge| Force the op to fp32 inside its forward; gradcheck (M6)  
NaN appears in one parameter, then spreads| That parameter blew up, then propagated through residual stream| Find first NaN with hook (M5); investigate that layer specifically  
  
Two diagnostic patterns worth knowing:

### Find the first NaN with hooks
    
    
    def nan_check_hook(name):
        def hook(module, inputs, output):
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"!!! NaN/Inf in {name} output")
        return hook
    
    for name, m in model.named_modules():
        m.register_forward_hook(nan_check_hook(name))

Run a few steps. The first hook to fire is where the NaN starts. Investigate _that_ layer's inputs and weights. (Don't leave these hooks installed in production — they sync the GPU on every layer.)

### Use `anomaly_mode` for backward NaNs
    
    
    with torch.autograd.set_detect_anomaly(True):
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()

This tells autograd to track which forward op produced any inf/NaN that appears in backward, with a stack trace. _Slow as molasses_ , so use it for debugging only — a 5-10× slowdown is typical. Find the bug, then turn it off.

## fp8: the newest player (briefly)

On H100 and newer, hardware fp8 matmuls become available. Two formats:

  * **e4m3** : 4 exponent bits, 3 mantissa bits. Max value ~448. More precision, less range.
  * **e5m2** : 5 exponent bits, 2 mantissa bits. Max value ~57344. More range, less precision.

The standard recipe: `e4m3` for forward activations and weights, `e5m2` for backward gradients. Tensor cores accept fp8 inputs and produce fp32 accumulators. With careful per-tensor scaling (similar in spirit to GradScaler but per-tensor and per-direction), training can run at near-bf16 quality at half the memory.

You don't typically write fp8 code by hand — libraries like NVIDIA's _TransformerEngine_ , _torchao_ , and increasingly PyTorch core handle the casting and scaling. The recipe is still being refined; it works well for inference and is becoming usable for training. We'll see fp8 quantization for inference deployment in M26.

#### Q&A; — About mixed precision **Q:** My validation loss is slightly worse than my training loss. Is mixed precision the cause? **A:** Almost certainly not. Mixed precision (especially bf16) tends to give nearly identical results to fp32 for transformers — usually within 0.1% on val metrics. The train-vs-val gap is normal generalization gap. If you're seeing a large drop (>1% absolute), suspect the loss / data / regularization choices, not the dtype. **Q:** Should I disable autocast for evaluation? **A:** It depends. For research / measurement, yes — run eval in fp32 to remove any precision-related variance. For production deployment, no — eval in the same precision as training, since that's what your real users will see. Most people leave eval in autocast and don't worry about it. **Q:** Why is my model fast in fp32 with TF32 but slow when I add bf16 autocast? **A:** Two possibilities. (1) Your model is dominated by ops that _don't_ use tensor cores (small ops, custom kernels, embedding lookups), so autocast adds cast overhead without speeding anything up. (2) You have shape mismatches that cause bf16 ops to fall off the fast tensor-core path. Profile (M13) and check kernel times before and after; if matmul kernels aren't faster in bf16, your shapes might not be tensor-core-aligned (multiples of 8 for bf16). **Q:** Does mixed precision break torch.compile? **A:** No, they work together. `torch.compile` will trace through the autocast region and compile fused kernels at the right precision. We'll see the integration in M21. **Q:** What about distributed training — any mixed-precision gotchas? **A:** DDP all-reduce of gradients: if your gradients are bf16, the all-reduce happens in bf16 and you can lose precision in the reduction. Most modern setups force gradients to fp32 before the reduce — it's the default behavior in PyTorch's DDP. FSDP has more knobs (M17) for trading off reduction precision and bandwidth. 

## The complete modern recipe

Here's the full mixed-precision training step you'll write a thousand times. bf16 version, the modern default:
    
    
    def train_step(model, opt, scheduler, batch):
        x, y = [t.to(device, non_blocking=True) for t in batch]
        opt.zero_grad()
    
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
            loss = F.cross_entropy(out, y)
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        return loss.item()

And the fp16 version for older hardware:
    
    
    scaler = torch.amp.GradScaler('cuda')
    
    def train_step_fp16(model, opt, scheduler, batch):
        x, y = [t.to(device, non_blocking=True) for t in batch]
        opt.zero_grad()
    
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(x)
            loss = F.cross_entropy(out, y)
    
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        return loss.item()

Notice how much simpler the bf16 version is. _Two fewer lines, no scaler object, no order-sensitive scale/unscale dance._ This simplicity is a real practical win on top of the numerical robustness.

## Code Magnets: build the bf16 training step

Build the canonical bf16 mixed-precision training step. Three magnets are wrong choices.

Arrange the magnets into a working step.

x, y = [t.to(device, non_blocking=True) for t in batch] opt.zero_grad() with torch.autocast('cuda', dtype=torch.bfloat16): out = model(x) loss = F.cross_entropy(out, y) loss.backward() loss.backward() scaler.scale(loss).backward() torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) opt.step(); scheduler.step() x = x.to(torch.bfloat16)

show solution
    
    
    x, y = [t.to(device, non_blocking=True) for t in batch]
    opt.zero_grad()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(x)
        loss = F.cross_entropy(out, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); scheduler.step()

The traps:

  * `loss.backward()` _inside_ the autocast block: backward should be outside. Autograd handles dtypes correctly when called from the outer context.
  * `scaler.scale(loss).backward()`: that's the fp16 pattern. bf16 doesn't need a GradScaler.
  * `x = x.to(torch.bfloat16)`: don't manually cast inputs. Autocast handles per-op casting; manually pre-casting fights the framework.

The right structure: **autocast wraps forward + loss; backward, clip, step, scheduler outside**. No GradScaler needed for bf16.

## Who does what?

Match each mixed-precision concept to its real role.

Concept

Real role

autocast block

A. fp32 working copy of weights, used by the optimizer for accurate updates.

Master fp32 copy

B. Per-op router that casts inputs based on op type — matmul to bf16, softmax to fp32.

GradScaler

C. Hardware fp32 matmul that internally uses 10-bit mantissa for ~2× throughput.

TF32

D. Multiplies loss by K to keep fp16 gradients above the underflow threshold.

bf16

E. Same range as fp32, fewer mantissa bits, no scaler needed — modern training default.

fp16

F. Tiny range (max 65504), needs GradScaler, used only on older hardware.

fp8 (e4m3)

G. 1-byte format for forward activations on H100+, used via TransformerEngine etc.

show solution

**autocast block** → B  
**Master fp32 copy** → A  
**GradScaler** → D  
**TF32** → C  
**bf16** → E  
**fp16** → F  
**fp8 (e4m3)** → G 

The mental shortcut: _autocast routes per-op, master copy preserves precision, GradScaler is the fp16 underflow fix, TF32 is silent fp32 acceleration, bf16 = modern default, fp16 = older HW with scaler, fp8 = newest H100+ feature_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team's training in fp16 mode trains for ~50 steps then loss goes to NaN. They use autocast + GradScaler. They check the GradScaler's `get_scale()` and see it's 65536. What's likely wrong?

show answer

The scaler hasn't yet halved its scale, so it didn't catch an overflow that propagated. Either (a) the model's activations are spiking high before any gradient is even computed (so loss becomes Inf inside forward, not in grads), or (b) there's a numerical bug in a custom op that doesn't go through autocast. Quick test: lower the initial scaler value (`GradScaler(init_scale=2**14)`), and add NaN-check forward hooks (Module text) to find which layer's output goes Inf first. If even fp32 training is fine but fp16 fails, the answer is "switch to bf16" — fp16's tiny range is the actual issue.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why is the loss computed inside the autocast block but the backward called outside?

show answer

Two reasons. (1) `F.cross_entropy` is on the autocast "fp32-preserving" list — it'll cast its inputs to fp32 internally for stability anyway, so you might as well let it see the bf16 logits and handle the cast at the boundary. (2) Backward doesn't need to be inside autocast because autograd _already recorded the dtype at forward time_ in each grad_fn. When backward runs, each op's backward uses whatever dtype was used in the corresponding forward, regardless of the current autocast state. Putting backward inside autocast would have no effect — it's structured logic, not a global mode flag.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** You're training in bf16 and want to verify nothing meaningful is being lost vs fp32. Outline a quick experiment.

show answer

Run two short experiments with identical seeds: one with autocast disabled (full fp32 / TF32), one with autocast bf16. Compare loss curves and final validation metric over say 1000 steps. Both should match within ~0.1%. If bf16's curve diverges noticeably, suspect: a custom op that's numerically unstable in low precision; a manual fp32 → bf16 cast somewhere fighting autocast; or genuinely numerically-sensitive task (very deep network, very long sequences). For 99% of transformer training, bf16 is indistinguishable from fp32 in the noise.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** You see significant matmul speedup from fp32-with-TF32 to bf16 autocast on an A100, but virtually none on a T4. Why?

show answer

The T4 (Volta-era) doesn't have bf16 tensor cores. It has fp16 tensor cores and full fp32 paths but no bf16 acceleration — bf16 on T4 falls back to fp32 emulation, with overhead from the casting. So on T4, bf16 is slower than fp32, not faster. On T4 you'd want fp16 + GradScaler, or fp32 alone. This is the kind of thing that profiling (M13) catches immediately — kernel times don't change as expected when you flip the dtype, hardware support is the answer.

### What just happened?

  * **Mixed precision is a per-region dtype contract**. Forward in bf16, normalization/softmax/loss in fp32, backward grads in fp32, optimizer step on fp32 master copy.
  * `torch.autocast('cuda', dtype=torch.bfloat16)` wraps the forward + loss. **Backward and step go outside** — autograd handles dtypes from recorded forward state.
  * **Autocast's three op categories** : lower-precision (matmul, conv, attention) → cast to bf16/fp16; fp32-preserving (softmax, logsumexp, cross_entropy) → cast to fp32; promoting (mixed-input arithmetic) → use highest input precision.
  * **The master fp32 copy** exists because bf16's 7-bit mantissa would round away tiny optimizer updates. The master is what the optimizer reads/writes; cast back to bf16 for the next forward.
  * **bf16 vs fp16** : same speed on tensor cores. bf16 has fp32-equivalent range, no underflow/overflow at gradient scale, no GradScaler needed. fp16 has tiny range (max 65504), needs GradScaler.
  * **Use bf16 if your hardware supports it** (Ampere/A100/H100/TPUs/MI200+). Only fall back to fp16 on older hardware.
  * `torch.amp.GradScaler('cuda')` is the fp16 safety net: scales loss by K (initially 65536) before backward, unscales before step, halves K on overflow, doubles after many steps without overflow.
  * **Order matters in fp16** : `scaler.scale(loss).backward()` → `scaler.unscale_(opt)` → `clip_grad_norm_` → `scaler.step(opt)` → `scaler.update()`.
  * **TF32 is silent fp32 acceleration** on Ampere+. Default on. Internally truncates fp32 mantissa to 10 bits for matmul speedup.
  * **NaN diagnosis** : bf16 is mostly NaN-proof. fp16 NaN comes from underflow (no scaler) or overflow (LR too high, activations spiking). Use forward hooks to find first NaN; `set_detect_anomaly(True)` for backward NaN with stack trace (slow — debug only).
  * **fp8** on H100+: e4m3 for forward, e5m2 for backward. Use via TransformerEngine / torchao; you don't typically wire it manually.
  * The reflex: when training is slow on modern hardware, the first thing to try is `autocast('cuda', dtype=torch.bfloat16)`. One line, ~2× speedup, no quality loss for transformers.

That closes Part V. You now have the predictive tools (memory budget, time budget) and the headline acceleration (mixed precision). The next part takes the same training step and runs it across multiple GPUs — same model, same step, same loss, but with the hardware multiplied. Module 15 opens Part VI by introducing the communication primitives that distributed training is built on: all-reduce, all-gather, scatter, broadcast.
