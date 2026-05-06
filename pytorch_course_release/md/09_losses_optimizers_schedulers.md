# Module 09 — Losses, optimizers & schedulers

# Losses, _optimizers & schedulers_

_Part III · Module 09_

— what `CrossEntropyLoss` is hiding, why AdamW exists, and the warmup-then-cosine schedule that powers every modern LLM

\--- 

You have layers and you have init. You have autograd. The last piece of the training-loop puzzle is the part that _uses_ the gradients: the loss that produces them, the optimizer that applies them, and the schedule that controls how aggressively. None of these are deep, but each one has a few sharp edges that take down beginners.

This module covers the cross-entropy stability tricks (the ones `F.cross_entropy` is doing for you), the actual update rules of the optimizers you'll use, the AdamW vs Adam-with-weight-decay difference (which is bigger than you think), and the cosine-with-warmup schedule that's the default for everything.

> **★ KEY IDEA**  
>  A loss function is a **differentiable scalar**. An optimizer is a **state machine** that maintains per-parameter buffers (momentum, running squared gradients, etc.) and applies them on `.step()`. A scheduler is a **boring function** that decides what the learning rate is this step. Each piece earns its own seat in the loop, but together they're just three lines: `loss.backward()`, `optimizer.step()`, `scheduler.step()`. 

## Two new players

⚙

Optimizer

"I hold state. I apply updates. I'm where memory disappears in big training runs."

When you create me with `Adam(model.parameters())`, I don't just memorize the parameters — I allocate _buffers_ for each one. SGD with momentum needs one buffer per parameter (the velocity). Adam needs _two_ (first and second moments). For a 7B parameter model in fp32, that's **56 GB of optimizer state** just for Adam. This is why ZeRO and FSDP exist (Module 17). Don't underestimate me.

⏱

Scheduler

"I'm the boring one. I just tell Optimizer what learning rate to use."

I read the step counter and produce a learning rate. Linear warmup at the start (because adaptive optimizers need a few steps to estimate variance). Cosine decay over training (because empirically it works). That's most of what I do. The exotic ones (OneCycle, ReduceLROnPlateau) are situational. Most of the time, warmup-then-cosine is the right answer and you should stop second-guessing it.

## Loss functions: the ones you'll actually use

Five losses cover ~95% of training. Each has a numerical-stability story worth knowing.

The losses that earn their keep Loss| For| What's hidden inside  
---|---|---  
`F.cross_entropy`| Multi-class classification with logits| Fused log-softmax + NLL; uses logsumexp for stability  
`F.binary_cross_entropy_with_logits`| Binary / multi-label classification with logits| Fused sigmoid + BCE; uses log(1+exp) trick for stability  
`F.mse_loss`| Regression| Just `(x - y)²`, no surprises  
`F.smooth_l1_loss` / Huber| Regression with outliers| Quadratic near zero, linear far from zero  
`focal_loss` (custom)| Imbalanced classification| Down-weights easy examples by `(1-p)^γ`  
  
### Cross-entropy: what it's actually doing

You feed it raw logits. It returns a scalar. In between, it's doing something subtle:
    
    
    # What you write:
    loss = F.cross_entropy(logits, targets)
    
    # What it's doing internally (roughly):
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    loss = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).mean()

Three things to notice. (1) The log-softmax and the negative-log-likelihood are _fused_ — never compute `softmax(logits).log()` separately, the intermediate `softmax` has bad numerics. (2) `logsumexp` from M3 does the heavy lifting on stability. (3) The `gather` picks just the logit corresponding to each example's true class.

> **⚠ WARNING**  
>  **Don't pre-softmax your logits before`cross_entropy`.** The function expects raw logits. People sometimes write `F.cross_entropy(logits.softmax(-1), targets)` as a "safety measure" — that's mathematically wrong AND numerically worse than just passing the logits. The function name says `cross_entropy` but its API contract is "give me the logits and I'll handle the rest." 

### Cross-entropy options worth knowing
    
    
    # label_smoothing: replace one-hot target with (1-α) * one-hot + α/V
    loss = F.cross_entropy(logits, targets, label_smoothing=0.1)
    
    # ignore_index: skip examples with this target value (e.g., padding tokens)
    loss = F.cross_entropy(logits, targets, ignore_index=-100)
    
    # reduction: 'mean' (default), 'sum', or 'none' (return per-example loss)
    losses_per_example = F.cross_entropy(logits, targets, reduction='none')

**Label smoothing** at 0.1 is a standard regularizer for classification — softens overconfidence, often improves calibration and slightly improves accuracy. **`ignore_index=-100`** is the standard convention for "skip this token's loss" — you set targets at padding positions to `-100` and CE just doesn't include them in the mean. Both ship for free.

### Binary cross-entropy: log-sum-exp trick #2

Same idea as CE but for binary outputs. Use `binary_cross_entropy_with_logits` (BCE-WL), _not_ the version that takes probabilities.
    
    
    # Right (numerically stable):
    loss = F.binary_cross_entropy_with_logits(logits, targets.float())
    
    # Wrong (sigmoid before passing):
    probs = logits.sigmoid()
    loss = F.binary_cross_entropy(probs, targets.float())   # overflows on large logits

The internal trick: `log(1 + exp(x))` overflows for large positive `x`. The fused form computes it as `max(x, 0) + log(1 + exp(-|x|))`, which is stable in both directions. Passing pre-sigmoided probabilities goes through `log(p)` which loses precision near 0 and 1.

### Focal loss for imbalance

For severely imbalanced datasets — say, object detection where most boxes are background. Cross-entropy is dominated by the easy negatives. Focal loss multiplies CE by `(1 - p_correct)^γ`, which down-weights examples the model is already confident about.
    
    
    def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = (-ce).exp()                  # prob assigned to the correct class
        return (alpha * (1 - pt) ** gamma * ce).mean()

γ=2 is the standard. The clever part is computing `pt` from the CE itself — you don't need a separate softmax pass.

### Regression: MSE vs Huber

MSE (L2): squared error. Penalizes outliers heavily. Huber (smooth L1): quadratic near zero, linear far from zero. Less sensitive to outliers.
    
    
    loss = F.mse_loss(pred, target)
    loss = F.smooth_l1_loss(pred, target)              # Huber with default beta=1.0

For most regression tasks, just use MSE. Reach for Huber when you have known outlier contamination (e.g., RL value targets, where one bad bootstrap can wreck training).

#### Q&A; — About losses **Q:** Should I use the functional form (`F.cross_entropy`) or the module form (`nn.CrossEntropyLoss`)? **A:** Either. The functional version is what most modern code uses because it's a one-liner. The module form exists for the rare case where you want to attach the loss as part of your `nn.Module` tree (e.g., to use hooks on it). They do the same thing. **Q:** Why is my loss negative? **A:** If you're using cross-entropy or BCE-WL, it shouldn't be — those are always non-negative. If you see a negative loss, you probably (a) used the wrong target dtype (CE expects long for class indices, not float), (b) accidentally used `cross_entropy` on already-softmaxed inputs, or (c) wrote a custom loss with a sign error. **Q:** What's `reduction='none'` for? **A:** When you want per-example losses to combine in a custom way — e.g., per-token cross-entropy weighted by some attention mask, or per-sample loss for hard-example mining. You get back a tensor of losses, and you reduce it yourself. **Q:** Does cross-entropy work for sequence models? **A:** Yes — flatten the batch and sequence dims first. `F.cross_entropy(logits.view(-1, V), targets.view(-1))` where `logits` was `(B, T, V)` and `targets` was `(B, T)`. Or use the modern signature that accepts `(B, V, T)` directly. 

## Optimizers: the math, the memory, the choice

Every optimizer has the same job: given gradients, update parameters. They differ in _what state they keep_ to make the update smarter than naive gradient descent. Each optimizer's character is its state.

The optimizers you'll meet, with their state cost Optimizer| Per-param state| Update rule (sketch)  
---|---|---  
`SGD` (no momentum)| None — 0× params| `p ← p − lr · g`  
`SGD + momentum`| 1 buffer (velocity) — 1× params| `v ← μv + g; p ← p − lr · v`  
`Adam` / `AdamW`| 2 buffers (m, v) — 2× params| EMA of g, EMA of g²; rescale; bias-correct  
`Lion`| 1 buffer (momentum) — 1× params| Sign-based; `p ← p − lr · sign(β₁m + (1-β₁)g)`  
`Muon` (newer)| 1 buffer + orthogonalization compute| Newton-Schulz orthogonalization on the momentum matrix  
  
Three things to internalize. (1) Adam takes **2× the parameter memory just for state** — and the parameters themselves are usually in fp32 even when the model is in bf16 (so it's another 2×). For a 7B model that's _56 GB of optimizer state_. (2) Lion is half the memory of Adam and a strong baseline; sign-based, less precise but works. (3) Muon is the new cool kid in 2024-25 — orthogonalizes the momentum matrix per step, hot in research circles for slightly better convergence.

Memory cost of optimizer state, per parameter (fp32) bytes per parameter params SGD 4 bytes velocity params SGD+mom 8 bytes momentum params Lion 8 bytes v (g²) m (g) params Adam 12 bytes master copy v (g²) m (g) bf16 params Adam mixed 14 bytes 7B params × 14B = ~98 GB just for optimizer + params

Look at that. _Optimizer state is the biggest single line item in big-model training._ The "8× model size" rule of thumb for total training memory (Module 12) is dominated by optimizer state. ZeRO-1 (Module 17) shards exactly this state across GPUs.

### SGD with momentum: the simplest serious optimizer
    
    
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    # Internally, per parameter p with gradient g:
    #   v ← momentum * v + g           (the velocity buffer)
    #   p ← p − lr * v                  (the update)
    # weight_decay is added to g before the velocity update (L2 regularization)

SGD+momentum is the right baseline for image classification (ResNets, EfficientNets). It's fast, has 1 buffer per param, and with the right LR and schedule beats Adam on most CV tasks.

### Adam: the adaptive workhorse

Adam keeps two EMAs per parameter: one of the gradient (first moment) and one of the squared gradient (second moment). It uses the second moment to _rescale_ the update — parameters with consistently large gradients get smaller steps; parameters with small gradients get larger steps. The "adaptive" in Adam.
    
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    
    # Per parameter p with gradient g:
    #   m ← β₁ * m + (1 - β₁) * g          (first moment)
    #   v ← β₂ * v + (1 - β₂) * g²         (second moment)
    #   m̂ = m / (1 - β₁^t)                 (bias correction)
    #   v̂ = v / (1 - β₂^t)                 (bias correction)
    #   p ← p − lr * m̂ / (√v̂ + ε)

The **bias correction** matters at the start of training. `m` and `v` are initialized to zero, so for the first few steps they're biased toward zero. The `m̂ = m / (1 - β₁^t)` correction undoes this. After ~100 steps the correction is negligible. But without it, your first few updates are tiny — exactly when you need them most.

### AdamW vs Adam-with-weight-decay: not the same thing

This trips up a lot of people. There are two ways to "weight decay" in Adam:
    
    
    # Way 1: Adam with weight_decay arg — adds w·p to the gradient BEFORE the EMA
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
    # Effective: m ← β₁m + (1-β₁) * (g + wd*p), then adaptive rescale
    # Problem: the rescaling by √v̂ also scales the wd term, so decay is
    # larger for params with small gradients — usually NOT what you want
    
    # Way 2: AdamW — applies decoupled weight decay AFTER the adaptive update
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    # Effective: p ← p − lr * (m̂/(√v̂+ε) + wd*p)
    # Decay is uniform across params regardless of their gradient magnitude

The fix in AdamW is just decoupling: _apply weight decay independently of the adaptive scaling_. The empirical effect on transformer training is significant — better generalization, especially for the "decay-the-weights, but-not-the-biases-and-LayerNorms" pattern.

> **⚠ WARNING**  
>  **Always use AdamW for transformers.** The original Transformer paper used SGD+momentum; modern transformers use AdamW. The "Adam(weight_decay=0.01)" path that some old tutorials show is the wrong version. `torch.optim.AdamW` is what you want. 

### The "no decay on biases and LayerNorms" trick

The standard parameter-grouping pattern:
    
    
    def make_param_groups(model, weight_decay):
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # Don't decay biases or 1-D parameters (LayerNorm/RMSNorm scales)
            if p.dim() <= 1 or name.endswith('.bias'):
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ]
    
    opt = torch.optim.AdamW(make_param_groups(model, 0.1), lr=3e-4)

Why? Decaying a LayerNorm scale toward zero would weaken normalization, which fights the network's whole purpose. Decaying a bias toward zero is also slightly bad — biases shift the distribution and 0 isn't always the right shift. The 2-D (and higher) weights are what we want regularized. This is the standard Hugging Face / nanoGPT pattern.

### Lion: half the memory, comparable performance
    
    
    opt = torch.optim.Lion(model.parameters(), lr=1e-4, weight_decay=0.01)   # PyTorch 2.4+
    
    # Per parameter p with gradient g:
    #   update = sign(β₁ * m + (1 - β₁) * g)        # sign-based!
    #   p ← p − lr * (update + wd * p)
    #   m ← β₂ * m + (1 - β₂) * g                   # update m AFTER using it

Lion (Lin et al., 2023) keeps just one buffer (the momentum), not two. The trick: take the _sign_ of the momentum-blended gradient, not its value. Surprisingly this works — pretrained transformer performance is comparable to AdamW at a smaller memory footprint. Tune the LR ~3-10× lower than you'd use with AdamW.

## Learning rate schedules

The learning rate is by far the most important hyperparameter. The schedule controls how it changes over training. Four common shapes:

The four schedules you'll meet Constant no decay; baseline only Warmup + cosine modern LLM default OneCycle vision tasks; super-convergence Step decay classic CNN training when in doubt: warmup + cosine

### Warmup-then-cosine: the modern default
    
    
    from torch.optim.lr_scheduler import LambdaLR
    import math
    
    def warmup_cosine(step, warmup_steps, total_steps, min_lr_ratio=0.1):
        if step < warmup_steps:
            return step / warmup_steps                          # linear warmup 0 → 1
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cos = 0.5 * (1 + math.cos(math.pi * progress))         # 1 → 0
        return min_lr_ratio + (1 - min_lr_ratio) * cos          # 1 → min_lr_ratio
    
    scheduler = LambdaLR(opt, lambda step: warmup_cosine(step, 2000, 100_000))

Why warmup? Adam-family optimizers' second moment `v` is unreliable for the first few hundred steps (it's biased and noisy). Taking large updates in this regime can destabilize training. Warming up the learning rate from 0 to peak over 1-5% of total training gives the optimizer time to estimate `v` properly. Without warmup, big transformers diverge.

Why cosine? Empirically, smoothly decaying the LR works better than constant or step decay. The cosine shape spends most of its time near the peak (high learning) and ramps down smoothly at the end (high precision). It's not magic — linear decay also works fine — but cosine is the convention.

### Other useful schedulers
    
    
    from torch.optim.lr_scheduler import *
    
    # Linear: warmup then linear decay (common in HF transformers)
    LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=2000)
    
    # Cosine annealing (without warmup)
    CosineAnnealingLR(opt, T_max=100_000, eta_min=0)
    
    # Reduce on plateau (validation-triggered)
    ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    
    # Compose schedulers — warmup then cosine, the official way
    warmup = LinearLR(opt, start_factor=0.001, total_iters=2000)
    cosine = CosineAnnealingLR(opt, T_max=98_000)
    scheduler = SequentialLR(opt, [warmup, cosine], milestones=[2000])

The `LambdaLR` approach is more flexible; the `SequentialLR` approach is more declarative. Both work.

> **⚠ WARNING**  
>  **Step the scheduler at the right granularity.** Most schedulers expect `scheduler.step()` per _training step_ (i.e., per optimizer.step()). Some old code calls it per epoch. `ReduceLROnPlateau` is the exception — it expects per-epoch with a metric: `scheduler.step(val_loss)`. Match your scheduler's expected granularity. 

## Gradient clipping

Last piece. The optimizer applies updates proportional to the gradient. If the gradient occasionally spikes — common in training with bf16 or long sequences — that one big update can wreck a parameter and destabilize training. Gradient clipping caps the update magnitude.
    
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

Two flavors:

  * **By norm** (`clip_grad_norm_`): if the total L2 norm of all gradients exceeds `max_norm`, scale them all down proportionally. _Preserves direction._ Recommended.
  * **By value** (`clip_grad_value_`): clamp each individual gradient element to `[-c, c]`. _Distorts direction._ Rarely used.

For transformers, `max_norm = 1.0` is the standard. For RL, often higher (5-10) because reward signals are noisier. **Place clipping between`backward()` and `step()`**; clipping after step is too late.

## Putting it all together: the canonical training step

Every modern training loop ends up looking like this:
    
    
    opt = torch.optim.AdamW(make_param_groups(model, 0.1), lr=3e-4, betas=(0.9, 0.95))
    scheduler = LambdaLR(opt, lambda s: warmup_cosine(s, 2000, 100_000))
    
    for step in range(100_000):
        x, y = next(loader)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1), ignore_index=-100)
    
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()

That's the modern LLM training inner loop. Memorize this template — you'll use it forever. Module 11 will harden it with checkpointing, accumulation, and resumption.

## Code Magnets: build the AdamW + cosine + clip step

You're writing the training step for a transformer with the modern recipe. Order matters.

Arrange the magnets into the correct training step. Two are red herrings.

opt.zero_grad() logits = model(x) loss = F.cross_entropy(logits.view(-1, V), y.view(-1)) loss = F.cross_entropy(logits.softmax(-1).view(-1, V), y.view(-1)) loss.backward() torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) opt.step() scheduler.step() opt.zero_grad(); opt.step(); loss.backward()

show solution
    
    
    opt.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    scheduler.step()

The traps:

  * Pre-softmaxing logits before `cross_entropy` is wrong — CE expects raw logits and does the log-softmax internally with stable arithmetic.
  * The merged "`opt.zero_grad(); opt.step(); loss.backward()`" reverses the order. `step()` needs gradients in `.grad`, which only exist after `backward()`.

The order matters: **zero → forward → backward → clip → step → schedule**. Clip _between_ backward and step (clip the gradients that step will use). Schedule _after_ step (most schedulers want to be called once per training step).

## Who does what?

Match each concept to its real role.

Concept

Real role

F.cross_entropy

A. Decoupled weight decay; the right Adam variant for transformers.

AdamW

B. Fused log-softmax + NLL with logsumexp; expects raw logits.

Bias correction in Adam

C. Caps total gradient norm to a max value; preserves direction.

Warmup phase

D. Skip decay on biases and 1-D LayerNorm scales.

Parameter grouping (no_decay)

E. Lets the second-moment estimate `v` stabilize before taking large steps.

clip_grad_norm_

F. Undoes the zero-init bias on `m` and `v` in early steps.

label_smoothing=0.1

G. Replaces one-hot target with a softer (1-α)·one-hot + α/V; regularizer.

show solution

**F.cross_entropy** → B  
**AdamW** → A  
**Bias correction in Adam** → F  
**Warmup phase** → E  
**Parameter grouping (no_decay)** → D  
**clip_grad_norm_** → C  
**label_smoothing=0.1** → G 

The mental shortcut: _CE fuses log-softmax+NLL, AdamW decouples decay, bias correction fixes early steps, warmup stabilizes the second moment, no-decay protects LayerNorms and biases, clip-grad caps norms, label smoothing softens targets_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** Compute the optimizer state size for a 1B parameter model trained with AdamW in fp32 (params and state both in fp32). Express in GB.

show answer

Per parameter: 4 bytes for params + 4 for first moment + 4 for second moment = 12 bytes. For 1B params: 12 GB just for params + Adam state. (In mixed precision with a master copy of params, you'd add another 2 bytes per param for the bf16 working copy = 14 GB.) This is why FSDP exists.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** You see `F.cross_entropy(logits, targets)` producing a loss of `nan` at the very first step. The logits are large (~30 in magnitude) but finite. What's the most likely cause?

show answer

Targets out of range. `cross_entropy` expects target indices in `[0, V)` (where V is the number of classes). If a target is `V` or larger, or negative (and not `ignore_index`), CE produces nonsense. The logsumexp on large logits is fine (the trick from M3 keeps it stable). Check `targets.max()` and `targets.min()`.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** A colleague's transformer trains fine on 1 GPU but diverges on 8 GPUs. They use AdamW with the same per-GPU batch size and the same LR. What's likely wrong?

show answer

The effective batch size is 8× larger across 8 GPUs, but the LR didn't change. With Adam-family optimizers, you typically scale LR sublinearly with batch size (`lr ∝ √(batch_size)` is the rule of thumb), or you keep it the same and accept that you've effectively widened the LR-stable region. Either keep the original LR but extend warmup proportionally, or scale LR up by a small factor. Just running on 8 GPUs without thinking about effective batch size is one of the top distributed training bugs.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Implement a "linear warmup, then constant" scheduler — useful when you want the optimizer to ramp in but don't have a fixed total step count.

show answer
    
    
    from torch.optim.lr_scheduler import LambdaLR
    
    def warmup_constant(step, warmup_steps):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    
    scheduler = LambdaLR(opt, lambda s: warmup_constant(s, warmup_steps=2000))

Useful when you're doing exploratory training and don't have a fixed budget. Easy to extend to "warmup, constant, decay at the end" if you later want a wind-down phase.

### What just happened?

  * **`F.cross_entropy` takes raw logits.** It fuses log-softmax + NLL with logsumexp inside. Don't pre-softmax.
  * `label_smoothing=0.1`, `ignore_index=-100`, and `reduction='none'` are the three CE options you'll actually use.
  * `F.binary_cross_entropy_with_logits` for binary/multi-label; **never** the version that takes probs.
  * **Optimizer memory** = (number of state buffers) × (param size). SGD: 0×, SGD+mom: 1×, Adam: 2×, Lion: 1×. For 7B in fp32 + Adam in fp32, that's ~98 GB.
  * **Adam keeps two EMAs** : first moment of `g`, second moment of `g²`. Bias correction undoes the zero-init of those buffers in early steps.
  * **AdamW decouples weight decay** from the adaptive scaling. Always use AdamW for transformers, not Adam-with-weight-decay.
  * The standard trick: **no decay on biases and 1-D parameters** (LayerNorm/RMSNorm scales). Group parameters and apply `weight_decay=0` to the no-decay group.
  * **Lion** halves the optimizer memory at the cost of slightly worse convergence. Tune LR ~3-10× lower than AdamW.
  * **Warmup-then-cosine** is the modern LR schedule. Warmup gives Adam's variance estimate time to stabilize. Cosine decays smoothly to ~0.1× peak LR.
  * **Gradient clipping by norm** , `max_norm=1.0`, between `backward()` and `step()`. Standard for transformer training. Preserves gradient direction; just caps magnitude.
  * **The canonical step** : `zero_grad → forward → backward → clip → step → scheduler.step`. Memorize the order.

That closes Part III. You've gone from `nn.Module` as a container, through init and normalization choices that make a network trainable, to the optimizer + loss + schedule trio that _actually trains it_. With one more part — the data pipeline and training-loop hardening — you'll have everything needed to ship a real training run.

Module 10 opens Part IV with the half of the system that nobody photographs: `Dataset`, `DataLoader`, samplers, `collate_fn`, the `num_workers` tuning game. The unsexy part where most performance bugs actually live.
