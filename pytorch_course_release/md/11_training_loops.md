# Module 11 — Training loops you can trust

# Training loops _you can trust_

_Part IV · Module 11_

— overfit-a-batch as the cheapest sanity check ever, the five things to save in a checkpoint, and gradient accumulation when you actually understand it

\--- 

You have all the pieces from Parts I-III: tensors, autograd, modules, init, optimizer, scheduler. You have data pipelines from M10. Now we wire them together into something that actually trains a model — and keeps training, and survives a process restart, and tells you when it's broken before you've burned three days of GPU time.

This module is about _discipline_. The training loop itself is ten lines; the discipline around it — overfit-a-batch first, log the right things, checkpoint everything, validate without leaks — is what separates "I trained a model once" from "I run training pipelines every week."

> **★ KEY IDEA**  
>  A trustworthy training loop has **five components** beyond the obvious forward/backward/step: an _overfit-a-batch sanity check_ before you spend any real GPU time, a _logging cadence_ that lets you spot trouble at glance, a _checkpoint format_ that captures everything needed for bit-equal resumption, a _validation loop_ that doesn't leak training state, and an _accumulation pattern_ that handles tiny GPUs and large effective batches. Each one is a few lines of code and saves you days of debugging. 

## Two new players

T

Trainer

"I'm the script that owns the loop. I'm an _architecture decision_ , not a class."

In small projects I'm a function. In larger ones I'm a class with hooks and callbacks. Either way, I'm where the data, model, optimizer, scheduler, and logger meet. I'm the place where you have to be most careful — every bug in me is a bug in _every_ training run. Three rules: I keep the inner loop simple, I log enough to debug remotely, and I save enough state to resume bit-perfectly. Frameworks like Lightning and HuggingFace's `Trainer` are just opinionated implementations of me.

⟁

Checkpoint

"I'm a dict that gets pickled to disk. Save more than you think you need."

People think I'm just `model.state_dict()`. I'm not. To resume training bit-perfectly, you need: (1) the model state, (2) the optimizer state, (3) the scheduler state, (4) all RNG states (Python, NumPy, torch CPU, torch CUDA), and (5) the step counter. Skip any one and your "resumed" run silently diverges from what would have happened with no restart. The disk cost of saving all five is rounding error. The cost of _not_ saving them is "why did training go off the rails after the restart?"

## The training loop, anatomy

Let's stare at the 10 lines you'll write a thousand times. Each one earns its place.
    
    
    opt = torch.optim.AdamW(make_param_groups(model, 0.1), lr=3e-4)
    scheduler = LambdaLR(opt, lambda s: warmup_cosine(s, 2000, 100_000))
    scaler = torch.amp.GradScaler('cuda')         # for fp16; bf16 doesn't need it
    
    for step, batch in enumerate(loader):
        x, y = [t.to(device, non_blocking=True) for t in batch]
    
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
            loss = F.cross_entropy(out, y)
    
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

Read each line as a question:

  1. **Optimizer** : which parameter groups, what LR, which decay strategy? (M9)
  2. **Scheduler** : what warmup, what schedule? (M9)
  3. **Scaler** : are we using fp16 (yes → scaler) or bf16 (no scaler needed)? (M14)
  4. **Loader iteration** : pinned, async, persistent workers? (M10)
  5. **Move to device** : `non_blocking` only if pinned. (M3, M10)
  6. **Autocast block** : which dtype, which ops cast? (M14)
  7. **Forward** : model in train or eval mode? (M7, M8)
  8. **Loss** : which loss function, raw logits, ignored padding? (M9)
  9. **Zero grad** : between every step, or accumulating? (M4)
  10. **Backward / clip / step / scheduler.step** : in the right order. (M4, M9)

Every module so far feeds into one of these lines. That's why we did them in that order.

## Reflex 1: overfit a single batch

Before you spend a dollar of GPU time on real training, run this:
    
    
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    
    for step in range(200):
        out = model(x)
        loss = criterion(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 10 == 0:
            print(f"step {step}: loss {loss.item():.4f}")

This loop trains on the _same single batch_ 200 times. The model has more than enough capacity to memorize 32 examples. Therefore: **the loss must go to ~0**. If it doesn't, you have a bug.

Overfit-a-batch: a 60-second sanity check step loss good — model memorizes ✓ bad — plateau (architecture/init bug) bad — noisy/unstable (LR too high)

The three curves above are real outcomes. Solid green is what you want — loss approaches zero in a few hundred steps. The other two are the bugs you catch with this check:

  * **Plateau** at some non-zero loss → architecture has a bottleneck (a frozen layer that shouldn't be, a dimension mismatch that's silently broadcasting wrong, an init scheme that's wedged the network into a flat region).
  * **Noisy/oscillating** → learning rate too high. Halve it and try again.
  * **Goes to NaN** → numerical instability, probably mixed precision related (Module 14).
  * **Slowly decreases but doesn't reach zero** → not enough capacity for the batch (try a bigger model — sometimes the issue is genuinely small batch with redundant samples).

Sixty seconds of compute saves you from spending a weekend wondering why a million-step run isn't learning. Run it before every new architecture, before every config change, before every restart of training on a new dataset. **It is the cheapest debugging tool in the entire course.**

> **⚠ WARNING**  
>  **"It overfits, ship it" is not the rule.** Overfit-a-batch tells you the gradients flow correctly, the loss is differentiable, and the model has capacity. It does _not_ tell you the architecture is good, the data is right, or the task is well-posed. It's a _necessary_ sanity check, not a sufficient one. 

## Reflex 2: log the right things at the right cadence

"Log everything, then look at it later" generates 50 GB of irrelevant numbers and one 30-minute scroll session. "Log the four things that actually matter, every step" gives you debuggable training.

What to log, and how often Quantity| Cadence| What it tells you  
---|---|---  
**Loss**|  Every step| Is it learning? Is it spiking? Is it NaN?  
**Learning rate**|  Every step (or every N)| Is the schedule doing what you think?  
**Gradient norm** (pre-clip)| Every step (or every N)| Is the model exploding? Did clipping save you?  
**Throughput** (steps/sec, tokens/sec)| Every N steps| Is the GPU starving? Did something get slower?  
**Weight / gradient histograms**|  Every 100-1000 steps| Are layers dying? Are gradients vanishing?  
**Validation loss / metric**|  Every N epochs (or N steps)| Is overfitting starting? Is real performance improving?  
**Sample outputs** (for generative)| Every checkpoint| Is the model producing reasonable outputs at all?  
  
The first three matter most. `loss` shows you if anything is happening. `lr` confirms the schedule. `grad_norm` catches instability before it becomes NaN.
    
    
    def log_step(step, loss, opt, model, log_every=10):
        if step % log_every == 0:
            lr = opt.param_groups[0]['lr']
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            # clip_grad_norm_ with inf max_norm just measures, doesn't clip
            print(f"step {step}: loss {loss:.4f}  lr {lr:.2e}  |g| {grad_norm:.3f}")

Note the trick: `clip_grad_norm_` with `max_norm=inf` returns the total grad norm without actually clipping. Cheap way to monitor. In real code you'd then call the actual clip with finite max_norm.

> **⚠ WARNING**  
>  **Don't log every step in production.** Logging via `print` or even W&B; at every step is fine for debugging — for a 1M-step run it's ~30 MB of log data and a few percent overhead. But if you log `.item()` on a CUDA tensor every step you're forcing a CPU-GPU sync (M3) — and that _can_ visibly slow training. Use `log_every=10` or higher in production. Reserve every-step logging for debugging. 

## Reflex 3: validation loop discipline

Three rules:

  1. `model.eval()` at the start, `model.train()` at the end. Otherwise BN running stats keep updating, dropout keeps firing.
  2. `with torch.no_grad():` around the whole thing. Otherwise you build a graph for activations you'll throw away.
  3. Don't update training state during validation. No optimizer step. No scheduler step. Definitely no zero_grad on training-state grads.

    
    
    @torch.no_grad()
    def validate(model, val_loader, device):
        model.eval()
        total_loss, total_correct, total_count = 0.0, 0, 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y, reduction='sum')
            total_loss += loss.item()
            total_correct += (out.argmax(-1) == y).sum().item()
            total_count += y.size(0)
        model.train()                    # RESET to train mode!
        return total_loss / total_count, total_correct / total_count

The `@torch.no_grad()` decorator is cleaner than wrapping the body in a context manager. Note the `reduction='sum'` on the loss — we accumulate per-sample sums and divide by total count at the end, which is correct even when the last batch is short. Using `reduction='mean'` and averaging the means would give wrong weights for the last batch.

The `model.train()` at the end is the most common forgotten step. Forgetting it means training continues in eval mode for the rest of the epoch — Dropout off, BN frozen — which slowly degrades training without any obvious symptom. Use a context-manager helper if you want bulletproof:
    
    
    from contextlib import contextmanager
    
    @contextmanager
    def eval_mode(model):
        was_training = model.training
        model.eval()
        try:
            yield
        finally:
            model.train(was_training)        # restore the previous mode
    
    with eval_mode(model), torch.no_grad():
        # validation logic here
        ...

This pattern survives exceptions and nested calls. Worth using in any non-toy training loop.

## Reflex 4: gradient accumulation

Sometimes the batch size you want exceeds GPU memory. The fix is to split the batch, accumulate gradients across micro-batches, and step once. _This is exactly equivalent_ to a single big-batch step (modulo the per-batch-norm statistics in BatchNorm — for LayerNorm/RMSNorm models, it's mathematically identical).
    
    
    accum_steps = 4          # effective batch = micro_batch * 4
    
    for step, batch in enumerate(loader):
        x, y = batch
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
            loss = F.cross_entropy(out, y) / accum_steps    # normalize!
        loss.backward()                                     # accumulates into .grad
    
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            scheduler.step()

Two non-obvious things. (1) **Divide the loss by`accum_steps`** — otherwise the accumulated gradient is `accum_steps` times too big. The math is: gradient of the mean is the mean of the gradients, so to get the same effective gradient as a single big batch, you need to mean across micro-batches, which means scaling each contribution by `1/accum_steps` before backward. (2) **`opt.zero_grad()` only every `accum_steps` steps**, not every step. The whole point is to let gradients accumulate.

This works even with mixed precision and DDP, with one caveat for DDP: you should wrap micro-batch backwards (except the last in a group) with `model.no_sync()` to avoid all-reducing partial gradients on every micro-batch. We'll see this in M16.

## Reflex 5: checkpoint and resume

The five things you must save to resume bit-perfectly:

What goes in a checkpoint model .state_dict() params + buffers optimizer .state_dict() moments, step scheduler .state_dict() last_epoch / step RNG state all generators torch + py + np + cuda step counter int \+ best_metric, etc. skip ANY one of these → silent divergence after restart disk cost: rounding error. correctness benefit: enormous.

The complete checkpoint:
    
    
    def save_checkpoint(path, model, optimizer, scheduler, step, **extra):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'step': step,
            'rng': {
                'torch':    torch.get_rng_state(),
                'cuda':     torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'numpy':    np.random.get_state(),
                'python':   random.getstate(),
            },
            **extra,             # best_val_loss, config dict, etc.
        }
        torch.save(state, path)
    
    def load_checkpoint(path, model, optimizer, scheduler):
        state = torch.load(path, weights_only=False)   # can't be True; rng state is non-tensor
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['rng']['torch'])
        if state['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(state['rng']['cuda'])
        np.random.set_state(state['rng']['numpy'])
        random.setstate(state['rng']['python'])
        return state['step'], {k: v for k, v in state.items()
                                  if k not in {'model', 'optimizer', 'scheduler', 'rng'}}

Two notes. (1) `weights_only=False` — the RNG states aren't tensors, so we can't use the safe-mode loader. Make sure you trust the source of the file. (2) `set_rng_state_all` for CUDA covers multi-GPU; `set_rng_state` would only set device 0.

### Resuming the loader

Even with all five buckets restored, there's one last hiccup: the `DataLoader`'s internal state. If you saved at step 50,000 and resume, you want the loader to start from sample 50,000 × batch_size, not from the top.

For map-style datasets, this is straightforward: skip-ahead by re-running the sampler N times, or use a `StatefulDataLoader` (the `torchdata` library has this; PyTorch core is adding it gradually). Pragmatic shortcut for most cases: just save the epoch number, restart from the beginning of that epoch. The first few thousand samples seen will differ from the original run, but the data was already shuffled, so this is usually fine.

For iterable datasets (streaming), it's harder — you need the dataset to know its position and resume from there. This is one of several reasons big LLM pretraining uses careful sharding with deterministic order so you can resume by skipping the right number of batches.

#### Q&A; — About checkpointing **Q:** How often should I checkpoint? **A:** Two answers. _Save-every-N-steps_ for resumability — a 1-2 hour rate is reasonable, depending on how expensive a re-run would be. _Save-best-validation_ as a separate file — overwrite this whenever val improves, so you have your "best so far" model decoupled from the most recent. Most teams maintain both: `latest.pt` for resumption, `best.pt` for evaluation. **Q:** Should I save the loss history / config / hyperparameters? **A:** Yes — they cost nothing and answer the "what was I doing" question 6 months later. Throw them in `extra`: `save_checkpoint(..., config=cfg, loss_history=losses)`. **Q:** My checkpoint is 50 GB. Anything I can drop? **A:** If it's an LLM-scale model, the optimizer state (Adam moments) is ~2× the model size. You can save it less frequently than the model itself — e.g., every 10× longer interval. The model alone (state_dict) is enough for inference; the optimizer state is only needed for training resumption. For deployment-only checkpoints, drop the optimizer. **Q:** What about FSDP / DDP wrapped models? **A:** DDP: get the underlying state_dict via `model.module.state_dict()` (or strip the `module.` prefix on load — M7). FSDP: use the FSDP-specific state_dict APIs (`FullyShardedDataParallel.set_state_dict_type`) — Module 17. Both add complexity. Plan for it before you scale up. 

## EMA: a thing outside the optimizer

Exponential Moving Average of model weights. Maintain a separate copy of the model that's a slow-moving smoothed version of the actual training weights. Use it for evaluation and final deployment.
    
    
    class EMA:
        def __init__(self, model, decay=0.999):
            self.decay = decay
            self.shadow = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
    
        @torch.no_grad()
        def update(self, model):
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)
    
        def apply_to(self, model):
            # temporarily swap in the EMA weights for evaluation
            ...
    
    ema = EMA(model, decay=0.999)
    for step, batch in enumerate(loader):
        # normal forward/backward/step ...
        opt.step()
        ema.update(model)         # after each optimizer step

Why? The optimizer's noisy updates jitter the model around the actual minimum. The EMA smooths over the trajectory, often giving better validation performance than the "live" weights. Decay around 0.999-0.9999 is typical (so the shadow weights are an EMA over the last ~1000-10000 steps). Standard in image generation (Stable Diffusion's UNet uses it), some RL setups, and increasingly in LLM training.

EMA weights need their own checkpoint entry — they're a separate state, not part of model.state_dict() unless you put them there as buffers (M7). Most teams keep them as a side dict.

## Deterministic vs fast: pick once, commit

From M3: deterministic GPU operations cost throughput. The choice you make for a training run depends on what you're optimizing.

Deterministic vs fast — pick by use case Scenario| Setting| Why  
---|---|---  
Debugging| Deterministic| Reproduce the bug to fix it  
Research / paper experiments| Deterministic if affordable| Bit-equality across runs is gold for science  
Production training| Fast| Throughput dominates; statistical reproducibility is enough  
Comparing two configs A/B| Fast, run multiple seeds| Variance from non-determinism < variance from initial random seed  
  
The pragmatic rule: _start non-deterministic, fast_. If you suspect a bug or need to reproduce a specific run, flip the determinism flags from M3. Don't make production training slow for the comfort of theoretical reproducibility.

## Putting it all together: the trustworthy training script
    
    
    def train(config, resume_from=None):
        # 1. Setup
        set_seed(config.seed)
        model = build_model(config).to(device)
        opt = build_optimizer(model, config)
        scheduler = build_scheduler(opt, config)
        loader = build_loader(config)
        val_loader = build_val_loader(config)
        ema = EMA(model, decay=config.ema_decay) if config.use_ema else None
    
        # 2. Resume if requested
        start_step = 0
        if resume_from:
            start_step, _ = load_checkpoint(resume_from, model, opt, scheduler)
    
        # 3. Sanity check (skip if resuming from far in)
        if start_step == 0:
            overfit_one_batch(model, opt, loader)
            reinit_optimizer(opt, model)         # reset moments after sanity check
    
        # 4. Main loop
        model.train()
        for step, batch in enumerate(loader, start=start_step):
            loss = train_step(model, opt, scheduler, batch, config)
            if ema:
                ema.update(model)
    
            if step % config.log_every == 0:
                log_step(step, loss, opt, model)
    
            if step % config.val_every == 0 and step > 0:
                val_loss, val_acc = validate(model, val_loader, device)
                log_val(step, val_loss, val_acc)
    
            if step % config.ckpt_every == 0 and step > 0:
                save_checkpoint(f"latest.pt", model, opt, scheduler, step,
                                config=config, ema=ema.shadow if ema else None)
    
            if step >= config.max_steps:
                break

That's the skeleton. Real codebases dress it up — config systems, distributed wrappers, callbacks for plugins — but every modification is to a piece of _this_. Internalize this loop and you can read any training framework.

## Code Magnets: assemble a robust train_step

Build the inner training step for a transformer with mixed precision, gradient accumulation (4 micro-batches), and clipping.

Arrange the magnets into the function body. Two are red herrings.

def train_step(model, opt, scheduler, batch, accum_steps, step): x, y = [t.to(device, non_blocking=True) for t in batch] with torch.autocast('cuda', dtype=torch.bfloat16): loss = F.cross_entropy(model(x), y) / accum_steps loss = F.cross_entropy(model(x), y) loss.backward() if (step + 1) % accum_steps == 0: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) opt.step(); opt.zero_grad(); scheduler.step() opt.zero_grad(); opt.step(); scheduler.step() return loss.item() * accum_steps

show solution
    
    
    def train_step(model, opt, scheduler, batch, accum_steps, step):
        x, y = [t.to(device, non_blocking=True) for t in batch]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x), y) / accum_steps
        loss.backward()
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(); scheduler.step()
        return loss.item() * accum_steps

The traps:

  * `loss = F.cross_entropy(model(x), y)` without dividing by accum_steps gives a 4× too-large gradient. The accumulated gradient must be the _mean_ across micro-batches.
  * `opt.zero_grad(); opt.step(); scheduler.step()` reverses the order — `step()` needs gradients in `.grad`, but `zero_grad()` would have just wiped them.

Notice the `loss.item() * accum_steps` at the end — we scaled the loss for backward, but for logging we want the unscaled value. Multiply back to undo. Forgetting this gives logs that look like the loss is mysteriously tiny.

## Who does what?

Match each training-loop concept to its real purpose.

Concept

Real purpose

Overfit a single batch

A. Smoothed copy of model weights, often used for final evaluation.

Gradient accumulation

B. Cheapest sanity check: confirms gradients flow correctly through the model.

RNG state in checkpoint

C. Simulate a larger effective batch size when memory is the bottleneck.

EMA

D. Required for bit-perfect resumption — without it, restarted training silently diverges.

model.eval() in validation

E. Disables Dropout and switches BN to running stats; pair with no_grad().

set_epoch on DistributedSampler

F. Ensures each epoch shuffles differently across distributed ranks.

loss / accum_steps

G. Normalizes the per-micro-batch contribution so accumulated gradient = mean.

show solution

**Overfit a single batch** → B  
**Gradient accumulation** → C  
**RNG state in checkpoint** → D  
**EMA** → A  
**model.eval() in validation** → E  
**set_epoch on DistributedSampler** → F  
**loss / accum_steps** → G 

The mental shortcut: _overfit-batch is the smoke test, accumulation is virtual batch size, RNG is for resumption, EMA is smoothed weights, eval mode is layer behavior, set_epoch fixes shuffle determinism, divide-by-accum normalizes the accumulated gradient_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team's training run resumes from checkpoint at step 50,000 and the loss spikes from 2.1 to 4.5 in the first 100 steps before recovering. What did they probably forget to save?

show answer

Optimizer state. Without restoring the Adam `m` and `v` buffers, the optimizer effectively starts fresh — its second moment is 0, bias correction is back to early-step territory, and the first few steps after restart are huge. The loss spike then a recovery is the textbook signature of "model weights restored, optimizer state lost." Save `opt.state_dict()` too. Same problem can happen for the scheduler if you don't save its `last_epoch`.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why does the validation loss curve in your training look noisier than the training loss curve? They should be smoother because you're averaging over the validation set...

show answer

Most likely you're using `reduction='mean'` per batch and then averaging the per-batch means — but the last batch is shorter than the rest. The result is that the last batch is over-weighted in the average. Use `reduction='sum'` per batch, accumulate, divide by total sample count at the end. The validation loss will still have noise (a finite val set), but at least it'll be the right number. Alternatively: drop the last batch (`drop_last=True` on the val loader) — fine if your val set is large.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** You're using gradient accumulation with `accum_steps=8`. You log the loss as `loss.item()` every step. Why is the curve 8× smaller than expected?

show answer

Because you divided the loss by `accum_steps` before backward (correct) but then logged the divided value (wrong for display). For logging, multiply back by `accum_steps`: `logged_loss = loss.item() * accum_steps`. Or compute and log the un-normalized loss separately. This is one of those bugs that makes training look weird but doesn't affect correctness — and people stare at the loss curve for an hour before realizing.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Implement a context manager `training_state(model, training)` that switches the model into the given mode (True = train, False = eval) and restores the previous mode on exit, even on exception.

show answer
    
    
    from contextlib import contextmanager
    
    @contextmanager
    def training_state(model, training):
        was_training = model.training
        model.train(training)
        try:
            yield
        finally:
            model.train(was_training)
    
    # Use:
    with training_state(model, False), torch.no_grad():
        val_loss = validate(model, val_loader)
    # model is back to its previous training state, regardless of exceptions

This pattern — capture-state, set-state, restore-state in `finally` — is the right way to write any "temporarily change something" helper. Useful for layers in inference mode, frozen-then-thawed parameter groups, autocast contexts, etc.

### What just happened?

  * The training loop is **ten lines** ; the discipline around it is what matters: sanity checks, logging, validation hygiene, accumulation, checkpointing.
  * **Overfit a single batch.** Sixty seconds of compute, catches gradient/architecture/init bugs before you spend a weekend on them. Loss must approach zero. Plateau / noise / NaN each tells you something specific.
  * **Log loss, LR, gradient norm** at every step (or every N for production). Throughput, weight histograms, val metrics at coarser cadences.
  * Avoid `.item()` on a CUDA tensor every step — it forces sync. Log every 10 steps in production.
  * **Validation discipline:** `model.eval()` at start, `train()` at end (or use a context manager), `@torch.no_grad()` for the body. Use `reduction='sum'` \+ divide by total count for correct averaging.
  * **Gradient accumulation** : divide loss by `accum_steps`, backward each micro-batch, `step()`+`zero_grad()` only every `accum_steps`. Mathematically equivalent to a big batch (modulo BN).
  * **The 5-bucket checkpoint** : model, optimizer, scheduler, RNG (all four — torch + cuda + numpy + python), step counter. Skip any one → silent divergence on resume.
  * Two checkpoint files: `latest.pt` for resumption (overwrite often), `best.pt` for evaluation (overwrite when val improves).
  * `DataLoader` resumption is the hardest part — for map-style, restart from epoch top is usually fine. For streaming, plan for it from day one.
  * **EMA** = exponential moving average of weights, decay 0.999-0.9999. Standard in image gen, increasingly in LLM training. Save separately from model state_dict.
  * **Determinism** is a tradeoff: pick fast for production, deterministic for debugging or paper-quality reproduction. Don't pay the throughput tax in production.
  * **The reflex** : when something looks wrong in training, run overfit-a-batch first. If that works, the loop is fine — the bug is probably in data or hyperparameters. If it fails, the bug is in model, init, or autograd.

That closes Part IV. You can now ship a trustworthy training run: pipeline, model, loss, optimizer, schedule, accumulation, checkpoint, validation, EMA. Everything from M1 onwards has earned its place in this loop.

Part V opens with Module 12: where does all that GPU memory actually go? Params, gradients, optimizer states, activations, workspace — the formula that lets you predict OOM before you hit it, and the activation-checkpointing trick that buys you depth at the cost of compute.
