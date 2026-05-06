# Module 32 — Debugging training runs

# _Debugging_ training runs

_Part IX · Module 32_

— the four-layer failure model, the NaN walking tree, loss curve forensics, gradient-norm patterns, and the bisection mindset that turns "the model isn't training" into a tractable diagnosis

\--- 

You will run a training job and something will go wrong. This is not pessimism; it's an empirical observation about real ML engineering. Loss will spike, NaNs will appear, the model will train but produce gibberish, OOM will hit at step 4000, the eval will silently diverge from train. Each of these has its own diagnostic approach and a small set of likely causes. **This module is the consolidated playbook.**

The skills covered here are scattered through the rest of the course — hooks (M5), gradcheck (M6), init schemes (M8), mixed precision (M14), memory profiling (M12, M20), the four bottleneck shapes (M13). M32's job is to organize them into a systematic diagnostic workflow you can apply when you don't know what's wrong. _Most debugging time isn't spent fixing bugs; it's spent localizing them._ Master the localization and the fixes usually become obvious.

> **★ KEY IDEA**  
>  Training failures fall into four layers: **numerical** (NaN, inf, divergence), **algorithmic** (wrong loss, wrong masks, wrong gradient flow), **systems** (hangs, OOM, slow), and **scientific** (training runs but the model is bad). Each layer has different diagnostic patterns. **Numerical bugs** are caught with hooks and gradcheck — they're loud. **Algorithmic bugs** often pass numerical tests but produce wrong loss curves — caught by sanity checks (overfit a tiny batch, compare to a known-good reference). **Systems bugs** are caught with the profiler (M13) and memory snapshot (M20). **Scientific bugs** are the worst — loss looks fine, model trains, but eval diverges from training because of distribution shift, data leakage, or evaluation methodology issues. The reflex: _don't guess. Bisect._ Halve the search space until the failure is local enough to see. 

## Two new faces — the diagnostic pair

D

Debugger

"I bisect. When you don't know where the bug is, halve the search space."

My core technique applies to every layer of failure. Loss diverging? Halve the LR — does it still diverge? Halve again. Gibberish output? Run with a tiny model on a tiny dataset — does it overfit? If yes, scale up; if no, the architecture is wrong. NaN at step 1000? Run with a fixed seed; the NaN appears at the same step. Now save the activations at step 999 and 1000 — diff them. _I never guess at fixes_. I localize, then I look. Most bugs that seem mysterious are obvious once they're surrounded by enough context. The patient bisection is what turns "it's broken" into "it's the layer-norm epsilon."

!

Sentinel

"I catch the first NaN. I'm the hook on every module that fires when something goes off."

By the time loss is NaN, the actual cause may be 50 layers ago. I'm a forward hook on every module that asserts the output is finite and within a sane range. The first time something explodes, I fire — with the module name, the activation statistics, and the input that caused it. _Without me, you're back-tracking from a symptom to a cause through ten thousand floating-point operations._ I'm cheap to install (one decorator) and I save you hours per failure. Use me whenever a training run is acting weird; remove me once you know it's clean. Cost: ~2-5% per-step throughput. Value: orders of magnitude in debugging time.

## The four-layer failure model

Before any specific tool, internalize the taxonomy. Symptoms come from one of four layers:

The four-layer failure model: start from symptom, identify layer, apply tool Observed symptom "my training is broken" ① Numerical NaN, inf, exploding grads, divergence Tools: hooks, gradcheck, fp32 ② Algorithmic loss not decreasing, wrong masks/gradients Tools: overfit-tiny, ref impl ③ Systems OOM, hang, slow, device errors Tools: profiler, mem snapshot ④ Scientific loss fine, model bad data leakage, eval bug Tools: eval bisection, audits Routing symptoms to the right layer: • "loss is NaN" → ① numerical (NaN walking tree) • "loss not decreasing" → ② algorithmic (overfit tiny batch first) • "OOM at step 4000" → ③ systems (memory snapshot at peak) • "train loss good, eval bad" → ④ scientific (eval bisection) most "mystery" failures are scientific — and the hardest to localize

Why categorize? Because _each layer has a different diagnostic toolkit_. Reaching for a profiler when your loss is NaN won't help; reaching for a hook when the issue is data leakage won't help. The first move is layer identification.

The cost ordering matters too. Numerical bugs tend to be loud (NaN crashes, gradient explosions); systems bugs are loud in a different way (OOM, hangs); algorithmic bugs are quieter (silently wrong loss); scientific bugs are silent until you evaluate. _Loud bugs are easier; silent bugs require discipline to catch._

## Layer 1: Numerical bugs and the NaN walking tree

"Loss is NaN" is the most common explicit failure. The cause is almost always one of a small set; walk them in order:

The NaN walking tree — diagnose by checking each cause in order Order| Cause| Signal| Fix  
---|---|---|---  
1| fp16 overflow (not bf16)| NaN appears in fp16 forward, esp. softmax/exp| Switch to bf16 (M14); if fp16 required, use GradScaler  
2| Missing fp32 in softmax/norm| NaN in attention scores or layer norm| Cast inputs to fp32 inside softmax/norm; cast back at output (M14, M23)  
3| Bad init| NaN at step 1; activations grow exponentially through depth| Reduce init std; add residual scaling 1/sqrt(2·n_layer) (M8)  
4| LR too high| NaN after 100-1000 steps; gradient norm spikes immediately before| Reduce LR by 5-10×; add warmup; clip grads at 1.0 (M9)  
5| Unstable loss formulation| NaN at specific data examples; `log(0)`, `div by 0`| Use log-sum-exp trick; add ε to denominators; clamp  
6| Mixed-precision accumulator| Gradients vanish then NaN; reductions in low precision| fp32 accumulators in custom kernels (M14, M23, M25)  
7| Bad data| NaN at one specific batch and only that batch| Inspect the batch; check for inf/nan in inputs; data sanitization  
  
The walking strategy: **start with #1, fix it (or rule it out), then #2, etc**. About 80% of NaN bugs are in the top three. If you reach #6 or #7, you're in unusual territory.

### Forward-hook bisection: catch the first NaN

When NaN appears partway through the forward pass, the actual _cause_ may be many layers earlier — by the time you observe NaN in the output, the propagation has obscured the source. **Forward hooks let you catch the first module whose output goes bad.**
    
    
    def install_nan_sentinel(model, abort_on_nan=True):
        def make_hook(name):
            def hook(module, inputs, output):
                if isinstance(output, torch.Tensor):
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        abs_max = output.abs().max().item()
                        n_nan = torch.isnan(output).sum().item()
                        n_inf = torch.isinf(output).sum().item()
                        print(f"BAD OUTPUT in {name}: max|out|={abs_max:.2e} "
                              f"nan={n_nan} inf={n_inf}")
                        if abort_on_nan:
                            raise RuntimeError(f"first NaN/inf in {name}")
            return hook
    
        for name, module in model.named_modules():
            module.register_forward_hook(make_hook(name))

Install at the start of training; remove once stable. The first time NaN appears, you get the exact module name. From there:

  1. **Localize the call site** : name the module printed by Sentinel (e.g., `blocks.7.attn.q_proj`). That's where the bad output originates.
  2. **Inspect inputs** : at the next iteration, when Sentinel fires, examine the inputs to that module. Are they already bad? Then the cause is upstream.
  3. **Walk upstream** : if inputs are good but outputs are bad, the bug is _in_ that module. If inputs are already bad, the prior module is the new suspect.

**Backward hooks** work analogously for "gradient is NaN":
    
    
    module.register_full_backward_hook(
        lambda mod, grad_in, grad_out: check_nan("backward", mod, grad_out)
    )

Use these when forward is fine but backward NaN-explodes. The first NaN in backward is usually the module right before a numerically unstable operation in forward (e.g., a softmax that produced 0.0, then 1/0 in backward).

### The gradcheck ritual

If you've written a custom op (M6, M23), **run`torch.autograd.gradcheck` in fp64 every time you change the backward**. Hand-derived backwards have a strikingly high bug rate.
    
    
    from torch.autograd import gradcheck
    
    # Use fp64 — finite-diff gradcheck is too noisy in lower precision
    inputs = (
        torch.randn(4, 128, dtype=torch.float64, requires_grad=True, device="cuda"),
        # ... other inputs
    )
    assert gradcheck(my_op, inputs, eps=1e-6, atol=1e-5)

If gradcheck fails: the analytical backward is wrong. Common cause: missing a term in the chain rule. Diagnose by checking each input's gradient separately — fp64 finite-diff vs analytical, with very tight tolerance. Whichever input fails first is the gradient with the bug.

## Layer 2: Algorithmic bugs

"Training runs but the loss curve looks wrong." Numerical setup is fine; the model is computing something, just not the right something. These are subtler than NaNs because everything appears to work.

### The overfit-tiny-batch test

The single most useful sanity check in deep learning: **can your model overfit a tiny batch?** If the architecture, loss, and optimizer are wired correctly, training on 16-64 examples for 1000 steps should drive the loss to near-zero. If it can't, the model has a structural bug.
    
    
    # Sanity check before any real training
    tiny_batch = next(iter(real_loader))
    for step in range(1000):
        optimizer.zero_grad()
        loss = model(tiny_batch).loss
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f}")
    # Healthy: loss decreases monotonically toward 0.001 or below
    # Pathological: loss plateaus near random initialization (≈ log(vocab_size))

If overfit fails, the bug is structural. Common causes:

  * **Missing mask** : causal mask not applied; model can see the future and trivially achieves training loss = 0... wait, that's a sign it _worked too well_. Actually missing causal mask makes overfit succeed instantly with eval garbage; check for that signal too.
  * **Wrong loss target** : predicting the wrong thing (off-by-one in next-token prediction; using prompt loss instead of response loss in SFT).
  * **Disconnected gradient** : a tensor was created with `torch.no_grad()`, `.detach()`, or in a `@torch.no_grad()` region, so gradients don't flow.
  * **Frozen parameters** : `requires_grad=False` on parameters that should train. Check `sum(p.numel() for p in model.parameters() if p.requires_grad)`.
  * **Init collapses to zero** : all-zero or near-zero init for a Linear layer outputs zeros; gradient through zero is zero; nothing learns.
  * **Optimizer not seeing parameters** : `torch.optim.AdamW(model.params)` typo for `model.parameters()`; or you constructed the optimizer before adding modules.

The fix once you know overfit-tiny works: _scale up one axis at a time_. Larger batch → still overfits? Larger model? Real data? Each step might surface a new bug, but you'll know which addition broke it.

### Comparing to a reference implementation

For canonical components (RMSNorm, attention, RoPE, AdamW), _diff against a reference_. Take the same inputs through your implementation and Hugging Face's (or PyTorch's), check the outputs match to numerical precision.
    
    
    import torch
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    
    x = torch.randn(2, 10, 128, dtype=torch.float32)
    # Build reference
    ref = LlamaRMSNorm(128, eps=1e-6)
    # Build mine — copy ref's weight
    mine = MyRMSNorm(128, eps=1e-6)
    mine.weight.data.copy_(ref.weight.data)
    
    out_ref = ref(x)
    out_mine = mine(x)
    print("max diff:", (out_ref - out_mine).abs().max().item())
    # Healthy: <1e-6 in fp32, <1e-3 in bf16 (numerical noise)
    # Bug: anything noticeably larger

If you see a meaningful difference: it's an algorithmic bug. Common ones for transformer components: wrong RMSNorm formula (forgetting to multiply by weight), wrong RoPE pairing (interleaved vs halved from M30), wrong masked positions (off-by-one in causal mask), wrong sinusoid frequencies.

## Loss curve forensics

The shape of your loss curve tells you what's wrong. Six archetypes worth recognizing:

Six loss curve archetypes — recognize the shape, know the cause ① Healthy smooth descent, occasional dips ② LR too high brief dip then explosion → NaN imminent ③ Plateau near random flat at log(vocab_size) → structural bug (disconnected grads, frozen params) ④ Spikes that recover grad clip is working → acceptable; reduce LR if too frequent ⑤ Spikes that don't recover numerical instability → post-norm instability, bad data, fp16 issues ⑥ Train ↓, eval flat/up overfit OR data leakage train eval → scientific failure layer "loss is bad" is too vague — name the SHAPE first

Reading the panels in order:

  1. **Healthy** : smooth monotonic descent, occasional dips. Initial sharp drop (model learns the prior token distribution), then slower descent (learns structure). What every healthy run looks like.
  2. **LR too high** : a brief dip then climbing back up, often spiking. Reduce LR by 5-10×, add warmup.
  3. **Plateau near random** : loss is flat at `log(vocab_size)` — random-initialization perplexity. The model isn't learning anything. _Structural bug_ : disconnected gradients, frozen parameters, wrong loss. Run the overfit-tiny test.
  4. **Spikes that recover** : occasional sharp rises followed by return to trend. Gradient clipping is working — clips are firing on the spikes. Tolerable if rare; if >5% of steps trigger clipping, reduce LR.
  5. **Spikes that don't recover** : a spike that the run never climbs back from. Indicates numerical instability the optimizer can't compensate for — post-norm at depth, fp16 overflow, bad data. Check the batch that triggered the spike.
  6. **Train decreasing, eval flat or increasing** : classic overfit shape, BUT also the signature of _data leakage_ , _distribution shift_ , or _evaluation methodology bugs_. The scientific layer.

Pattern matching is a real skill. After a few hundred runs you recognize the shapes instantly; until then, log loss frequently and look at the curves regularly.

## Gradient norm tracking

Per-layer gradient norms reveal the health of gradient flow. Log them at every step (or sample, every 100 steps) and visualize:
    
    
    def log_grad_norms(model, step):
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                wandb.log({f"grad_norm/{name}": grad_norm}, step=step)
        # Total grad norm too
        total = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        wandb.log({"grad_norm/total": total.item()}, step=step)

The patterns to recognize:

Gradient norm patterns and what they mean Pattern| Likely cause| Fix  
---|---|---  
All layers near 1.0, stable| Healthy| —  
Early layers near 0, late layers normal| Vanishing gradients| Init scheme issue (M8); check residual connections; check norms  
Late layers exploding (10×+ early layers)| Init too aggressive at depth| Apply 1/sqrt(2·n_layer) scaling to residual outputs (M8)  
Spike on every step| LR too high or unstable loss| Reduce LR; check loss formulation  
Steady drift up over 1000s of steps| Optimizer divergence (esp. without weight decay)| Add weight decay; check Adam β₂ (often 0.95 better than 0.999 for LLMs)  
Total norm = 0 for some parameters| Frozen or disconnected| Check requires_grad; trace gradient flow with hooks  
  
Per-layer is more diagnostic than total. Total grad norm of 5.0 might mean "every layer at 5.0 (mild explosion)" or "one layer at 50.0, rest near 0 (specific pathology)" — wildly different fixes. _Always log per-layer when investigating_.

## Layer 3: Systems bugs

Failures in this category: OOM, hangs, stragglers, dataloader bottlenecks, NCCL errors. The diagnostic toolkit is the profiler (M13) and the memory snapshot (M20).

### OOM diagnostics — the cost ladder

When you hit OOM, the question isn't "how do I get more memory" but "where is the memory going and what can I trade off?" The fixes form a ladder of increasing cost (engineering effort + throughput hit):

OOM fix ladder — apply in order, escalate as needed Fix| Memory savings| Throughput cost| Engineering effort  
---|---|---|---  
Reduce microbatch size| Linear in batch| ~0% (with grad accum)| ~zero  
Mixed precision (bf16) if not already| ~2× model + activations| +10-30% throughput| One-line  
Activation checkpointing (M12)| ~5-10× activations| −20% throughput| Wrap blocks  
FSDP / ZeRO-3 (M17)| ~N× across N GPUs| −10-15% throughput| One-line wrap  
Offload optimizer state to CPU| ~4× (Adam state)| −40-60% throughput| Config change  
Quantization (M26)| 2-4× weights| ±0% inference, training varies| Library integration  
Tensor parallel (M18)| ~N× across N GPUs| −5-10% throughput| Significant  
Pipeline parallel (M18)| Per-stage memory| Bubble cost| Significant  
  
The discipline: _start at the top of the ladder; only descend when needed_. Reducing microbatch is free and almost always works. Activation checkpointing is the next step. Most OOM problems for sub-70B models are solved by step 3 of the ladder.

The diagnostic flow:

  1. **Take a memory snapshot** at the OOM step: `torch.cuda.memory._record_memory_history()` \+ `torch.cuda.memory._dump_snapshot()` (M20). Visualize via `https://pytorch.org/memory_viz`.
  2. **Identify the dominant consumers** : weights, activations, optimizer state, KV cache?
  3. **Apply the right fix** from the ladder. If activations dominate: checkpointing. If optimizer state dominates: FSDP or offload. If weights dominate: quantization or model parallel.

### Hangs and stragglers in distributed training

"Training stops at step N forever, no error." The dreaded silent hang. Common causes in priority order:

  1. **One rank crashed silently** : check all rank logs, not just rank 0. NCCL collectives wait forever for the missing rank.
  2. **Ranks taking different code paths** : an `if` branching on rank-local data leads to one rank doing one collective and another rank doing a different one. Both wait. Set `NCCL_DEBUG=INFO` to see what each rank is doing.
  3. **Variable-length data without proper handling** : one rank's batch has 100 samples, another has 99, and the ranks call different numbers of all-reduces. Pad to fixed shapes or use NCCL's variable-size operations.
  4. **Network issue** : rare but real. Check `nccl-tests` for cluster health.
  5. **Deadlock on locks** : typically in custom dataloader workers using shared resources. `py-spy dump --pid <trainer>` shows where each rank is stuck.

The single most useful debugging incantation:
    
    
    NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL python train.py

Floods the logs but tells you exactly which collective is hanging on which rank.

## Layer 4: Scientific bugs

The hardest layer. Loss looks fine, training runs cleanly, but the model isn't learning what you think it's learning.

### Train-eval divergence

You expected eval loss to track training loss; instead, eval loss is flat or rising while train loss decreases. The pattern from archetype #6.

Three common causes, each with its own diagnosis:

  * **Overfitting** : classical, expected at high parameter-to-data ratios. Fix: more data, regularization (weight decay, dropout), early stopping. The "easy" case if the gap appears gradually after thousands of steps.
  * **Data leakage** : training data overlaps with eval data, sometimes in subtle ways (paraphrases, near-duplicates, same source documents split differently). Fix: aggressive deduplication; check eval data origin. _This makes loss look better than reality_.
  * **Distribution shift** : training data is from one distribution, eval from another, even though they "should be" the same. Examples: training on a snapshot of web data; eval on a recent snapshot — different distributional balance. Fix: align distributions; report eval on multiple sets.

Diagnostic: **shrink the eval set to exact-overlap test cases (memorization probes)**. If your training data contains `"the capital of France is Paris"` verbatim and eval contains the same string, the model should perfect-predict it. If it does: model is fine, the gap is elsewhere. If it doesn't: the model isn't learning even in-distribution.

### Tokenizer mismatches

An entire category of bugs: _training and inference use different tokenizations_. The model trained on tokens [a, b, c] for "hello"; at inference, the tokenizer produces [a, d] for "hello" — different sequence, gibberish output.

The check: `tokenizer.decode(tokenizer.encode(text)) == text` for several sample texts. Should be true. If false, you have a tokenizer round-trip issue.

For chat models, additional checks: **verify chat template**. The training-time template (e.g., Llama-3's `<|begin_of_text|>...<|end_of_text|>`) must match the inference-time template exactly. _One missing special token can break generation entirely_.

### Eval methodology bugs

The evaluation itself can have bugs that make your model look better or worse than it really is. Common issues:

  * **Wrong loss masking on eval** : SFT eval should compute loss on response tokens only (matching training); computing on full sequence underestimates eval loss vs train.
  * **Inconsistent generation parameters** : comparing models with temperature 0.7 vs 1.0 — one will look better not because it is.
  * **Stale benchmarks** : the eval set leaked into the model's training data via the web. Check known contamination indicators.
  * **Numerical precision differences** : eval in fp32 vs training in bf16 produces small differences; for sensitive metrics (top-k accuracy on close logits), this can matter.

The reflex: **before believing an eval delta, verify the eval is reproducible**. Run it twice; small variance is normal, large variance means the eval has a bug.

## The reproducibility checklist

"Works on my machine but not on the cluster" is debugging's nightmare. The following checklist catches most reproducibility issues:

Reproducibility checklist for cross-machine training Item| Check  
---|---  
Random seed| Set torch, numpy, random, AND CUDA seeds; `torch.manual_seed(42); torch.cuda.manual_seed_all(42)`  
DataLoader determinism| Set `generator=torch.Generator().manual_seed(s)`; use `worker_init_fn` to seed workers (M10)  
CUDA non-determinism| `torch.use_deterministic_algorithms(True)` \+ `CUBLAS_WORKSPACE_CONFIG=:4096:8`  
Library versions| Pin PyTorch, transformers, triton, CUDA toolkit; export with `pip freeze > requirements.txt`  
GPU model differences| A100 vs H100 produce slightly different results in some kernels (esp. attention); document the GPU type  
Dataloader shuffle| Same shuffle seed across runs; `DistributedSampler(set_epoch=epoch)` for distributed  
Master weights vs model weights| Save master weights (fp32) for reproducibility; bf16-only checkpoints aren't bit-exact reloadable  
Mixed precision behavior| bf16 vs fp16 produce different round-off behaviors; pin the dtype  
  
Note that **perfect determinism is expensive**. `torch.use_deterministic_algorithms(True)` can slow training 20-50% and forbids some kernels (cuBLAS chooses different algorithms per call by default). For most production runs, _statistical reproducibility_ (similar loss curves) is sufficient; _bit-exact reproducibility_ is for debugging only.

## The bisection mindset

The single most underrated debugging skill: **when you don't know where the bug is, halve the search space**.

Examples in practice:

  * **Loss diverging at step 1000** : was it diverging at 500? Run with checkpoint loaded at 500, retrain. If it diverges at 1500 (500 more steps in): the bug accumulates. If it diverges immediately: the checkpoint at 500 is already bad, bug was earlier.
  * **OOM at some point in training** : at step 0, memory is X; at OOM step, memory is Y. Bisect: at step OOM/2, what's memory? Linear growth means a leak; sudden growth means a specific event.
  * **Model produces gibberish** : replace half the architecture with a known-good reference (e.g., Hugging Face's). Does it work? If yes, the bug is in the half you replaced; if no, in the half you kept. Halve again.
  * **Distributed hang** : half the ranks log "passed step 100"; other half don't. The passing half is fine; bug is in the other half. Reduce world size to just one rank from the failing group; debug locally.
  * **Eval regression after a code change** : bisect the commit history. `git bisect` \+ an automated eval script localizes the offending commit in log₂(N) tries.

The mindset: _don't try to understand the whole system at once_. Understand "is the bug before this point or after this point" — a binary question — and apply it recursively. Each halving reduces the search space by 50%; eight halvings narrows 1000 candidates to about 4.

## The full diagnostic playbook

Putting it together. When you observe a training failure:

  1. **Identify the layer** : numerical / algorithmic / systems / scientific. Use the symptom router from earlier.
  2. **For numerical** : install Sentinel hooks; walk the NaN tree; check fp32 reductions in softmax/norm/cross-entropy.
  3. **For algorithmic** : run overfit-tiny; if fails, structural bug. Compare against reference implementation; diff outputs.
  4. **For systems** : profile (M13) for slow / hangs; memory snapshot (M20) for OOM. Apply the OOM ladder or the four-bottleneck shapes.
  5. **For scientific** : verify reproducibility; check tokenizer round-trip; eval on memorization probes; check for data leakage.
  6. **If still stuck** : bisect. Find a known-good state (earlier checkpoint, smaller model, simpler eval) and a known-bad state. Halve.
  7. **Document what you tried** : a debugging journal. The same bug will reappear; future-you will thank past-you.

The discipline is in NOT trying random fixes. _Each diagnostic step localizes the bug; each random fix obscures it._

#### Q&A; — About debugging **Q:** When should I install Sentinel hooks vs run gradcheck? **A:** Different scopes. Sentinel hooks (forward/backward NaN checks) catch numerical issues during a real training run — the kind that depend on actual data and accumulated state. Gradcheck verifies the analytical backward of a custom op against finite differences in fp64 — it catches math bugs, not numerical-precision bugs. _Use gradcheck once when writing the op (and after every change); use Sentinel hooks during development training when something feels off_. They're complementary. **Q:** My loss is healthy at step 100, then NaN at step 500. Hooks installed but never fire on forward — only on backward. What's happening? **A:** A common pattern: the _forward_ produces a finite-but-degenerate value (e.g., a softmax entry equal to exactly 0.0), and the _backward_ divides by it, producing NaN. Forward looks healthy; backward explodes. The fix: find the operation in the backward graph that divides by a saved forward output, and add an epsilon. Common culprits: softmax outputs (use log-softmax + log-sum-exp trick), denominators in normalization, attention weights when masked positions get exactly -inf. **Q:** How aggressive should I be with overfit-tiny tests? **A:** Use them _before any real training_ , every time you change the architecture, loss, or data pipeline. They take minutes and catch hours of debugging. The check: 16-64 examples, 500-1000 steps, loss should drive to near-zero (perplexity < 1.05 or so). If not, the architecture has a bug. _Don't proceed to real training until overfit-tiny passes_. The 5 minutes you save by skipping it tend to cost 5 hours later. **Q:** My distributed training hangs at exactly the same step every time. NCCL_DEBUG shows ranks 0-3 waiting on an all-reduce; ranks 4-7 waiting on a different all-reduce. What's going on? **A:** Classic divergent code path. Some ranks called all-reduce A, others called all-reduce B, and now both groups are waiting for the other. Causes are usually rank-dependent control flow: an `if` on rank-local data, a length-dependent number of operations, conditional skipping of layers. Diagnose with `NCCL_DEBUG=INFO` \+ `TORCH_DISTRIBUTED_DEBUG=DETAIL`; one will be at "step 100, op X", the other at "step 100, op Y". Examine your code at op X vs Y to find the divergence. Fix by ensuring all ranks always call the same collective sequence. **Q:** When does a bug really require bisection vs just careful reasoning? **A:** If you can predict the cause from the symptom (e.g., "loss spike + bf16 + softmax = needs fp32 in softmax"), reason. If multiple causes seem equally plausible (e.g., "loss isn't decreasing — could be init, LR, gradient flow, mask, optimizer setup"), bisect. _Reasoning works when the symptom maps cleanly to a known cause; bisection is for when the symptom underconstrains the cause_. The skill is recognizing which situation you're in. Junior engineers tend to reason when they should bisect; senior engineers reach for bisection earlier. **Q:** How do I know if a slow-training problem is dataloader-bound vs GPU-bound? **A:** Profile with the four-bottleneck-shapes mental model from M13. Run with the profiler for 50 steps; look at the GPU timeline. **Dataloader-bound** : the GPU sits idle (no kernels running) at the start of every step, waiting for data. Fix: more workers, pin_memory, prefetch_factor higher, persistent_workers=True. **GPU-bound** : the GPU is busy throughout each step, no idle gaps. You're already running as fast as the model lets you; further gains require kernel-level work or larger batches. _The shape of idle gaps in the timeline is diagnostic_ ; you don't need separate tools. 

## Code Magnets: install a NaN sentinel

You're installing a forward-hook sentinel that fires on the first NaN/inf and prints the offending module. Three magnets are wrong choices.

Arrange the magnets into a working sentinel installer.

def install_nan_sentinel(model): def make_hook(name): def hook(module, inputs, output): def hook(module, inputs): if isinstance(output, torch.Tensor): if torch.isnan(output).any() or torch.isinf(output).any(): if output.isnan() or output.isinf(): print(f"BAD OUTPUT in {name}") raise RuntimeError(f"first NaN/inf in {name}") return hook for name, module in model.named_modules(): module.register_forward_hook(make_hook(name)) module.register_forward_pre_hook(make_hook(name))

show solution
    
    
    def install_nan_sentinel(model):
        def make_hook(name):
            def hook(module, inputs, output):
                if isinstance(output, torch.Tensor):
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        print(f"BAD OUTPUT in {name}")
                        raise RuntimeError(f"first NaN/inf in {name}")
            return hook
        for name, module in model.named_modules():
            module.register_forward_hook(make_hook(name))

The traps:

  * `def hook(module, inputs):` with two args: that's a _forward pre-hook_ signature — fires before the module runs and only sees inputs, not output. Useless for catching bad outputs. Forward hooks have signature `(module, inputs, output)` — three args.
  * `if output.isnan() or output.isinf():`: tensor methods `isnan()` and `isinf()` return _tensors_ , not booleans. `if tensor` is ambiguous — if the tensor has multiple elements, raises an error; if it has one element, evaluates that element's truthiness. The correct pattern is `.any()` after the check, which reduces to a single bool.
  * `module.register_forward_pre_hook(make_hook(name))`: pre-hooks fire before the forward and don't see the output. We want post-forward hooks (the regular `register_forward_hook`) so we can inspect the output.

The pattern: **forward hook (3-arg signature) → check output is a Tensor → use .any() to reduce isnan/isinf to a bool → print module name → raise to abort training**. Aborting means the next training run will fail immediately on the bug; without aborting, you might miss subsequent occurrences and lose the diagnostic value.

## Who does what?

Match each debugging concept to its real role.

Concept

Real role

The four-layer failure model

A. Numerical / algorithmic / systems / scientific — categorize the symptom first.

NaN walking tree

B. Diagnostic order for "loss is NaN" — fp16, missing fp32, bad init, LR, etc.

Forward-hook sentinel

C. Catches the first NaN/inf in any module's output, with the module name.

Overfit-tiny test

D. Sanity check before real training: can the model overfit 16-64 examples?

Loss curve archetypes

E. Six canonical shapes (healthy, LR-too-high, plateau, spikes-recover, spikes-don't, train-eval-divergence) — recognize the shape, know the cause.

Per-layer gradient norm tracking

F. Reveals vanishing/exploding gradient patterns invisible in total grad norm.

Bisection mindset

G. When you don't know where the bug is, halve the search space recursively.

show solution

**Four-layer failure model** → A  
**NaN walking tree** → B  
**Forward-hook sentinel** → C  
**Overfit-tiny test** → D  
**Loss curve archetypes** → E  
**Per-layer gradient norm** → F  
**Bisection mindset** → G 

The mental shortcut: _categorize the symptom (4 layers), walk the NaN tree, install hooks to catch the first bad value, sanity-check with overfit-tiny, recognize loss-curve shapes, log per-layer grad norms, bisect when stuck_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team's GPT-style model trains for 800 steps with healthy loss curves, then NaN appears. The Sentinel hook fires on `blocks.11.attn.softmax`. They're using bf16 throughout. What's the most likely cause and fix?

show answer

The softmax in attention is being computed in bf16. bf16 has wide range but only 7-bit mantissa — exponentials of values near the maximum can produce results that are far enough apart in magnitude that rounding produces zeros where there shouldn't be any. The softmax then divides by a sum that's effectively the largest entry, and a near-zero entry gets rounded to exactly zero. Subsequent computations may divide by it (especially in backward) and produce NaN.

The standard fix: **compute the softmax in fp32**. Cast the attention scores to fp32 before softmax, do the softmax in fp32, cast the result back to bf16 for the matmul with V. PyTorch's `F.scaled_dot_product_attention` handles this internally; the bug appears in hand-rolled implementations that skip the fp32 promotion. _This is the M14 mixed-precision recipe applied to attention_ — and it's the second-most-common NaN cause in transformer training (after fp16 overflow).

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** A team's loss decreases healthily but their model produces gibberish at generation time. Greedy decoding gives `"the the the the..."`. They've checked the architecture against a reference and it matches; gradcheck passes. What's likely wrong?

show answer

This is a generation-time bug, not a training-time bug. Three most-likely candidates, in priority order:

(1) **Tokenizer mismatch** : training and generation use different tokenizers (or chat templates). Verify with `tokenizer.decode(tokenizer.encode("hello world")) == "hello world"` — should be true. For chat models, verify the chat template tokens match what was used during SFT.

(2) **RoPE position bug** : at generation, the position index for each new token must increment correctly. If the same position (0) is used for every decode step, the rotations don't advance and attention scores become degenerate. The KV cache typically stores post-RoPE keys; `start_pos` must increment each decode step (M29).

(3) **Causal mask absent during prefill** : if the model was trained with causal masking but generation doesn't apply it during prefill, attention sees future tokens — which don't exist in the cache, leading to garbage. `F.scaled_dot_product_attention(..., is_causal=True)` on prefill, `False` on single-token decode.

The diagnostic test: run the model on the same prompt at training-time (loss computation) and generation-time (forward + sampling). The output logits should match. If they differ, you have a setup difference between paths.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** A team observes that their training run reproduces perfectly on a single H100 but produces meaningfully different loss curves on different machines (also H100s). They've set `torch.manual_seed` and `torch.cuda.manual_seed_all`. What's likely going on?

show answer

Several possibilities, in order of likelihood:

(1) **cuBLAS non-determinism** : cuBLAS chooses among multiple matmul algorithms based on shape, and the selection can vary with workspace memory. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True)`. Slows training by 20-50% but eliminates this source.

(2) **DataLoader worker non-determinism** : the worker processes spawn at slightly different times across machines, and unless you've used `worker_init_fn` with the worker_id-based seed, they shuffle differently. Use the recipe from M10 to seed each worker deterministically.

(3) **Different driver/library versions** : same GPU model can run different cuDNN versions on different machines, producing slightly different kernels. Pin everything: PyTorch version, CUDA version, cuDNN version (via NVIDIA driver pinning).

(4) **Multi-GPU non-determinism** : NCCL all-reduces have non-deterministic order at the bit level (the reduction tree is built dynamically). For exact reproducibility, set `NCCL_DETERMINISTIC=1`.

For most production runs, _statistical reproducibility_ (loss curves match within noise) is enough; bit-exact reproducibility is for debugging specific issues. If you really need bit-exact, the cost is real.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Your model's training loss decreases beautifully, but on the held-out eval set, accuracy is barely above chance. You've ruled out overfitting (regularization is appropriate, gap is too large for normal overfit). Walk through your diagnostic plan.

show answer

Train-eval divergence with low eval accuracy is the scientific failure layer. The diagnostic walk:

(1) **Memorization probe** : extract a few exact training-data examples and run them through the model at eval time. Does it predict them well? If yes, the model is fine; the gap is in eval. If no, training itself isn't producing a usable model — probably a tokenizer or template bug.

(2) **Tokenizer round-trip** : verify `tokenizer.decode(tokenizer.encode(text)) == text` on training and eval samples. Check chat templates if applicable. A subtle template difference is a common cause of train-good-eval-bad.

(3) **Eval methodology** : is loss computed the same way on train and eval? In SFT, are response tokens the only ones with loss? Are eval samples being preprocessed differently?

(4) **Distribution shift** : are training and eval really from the same distribution? Look at length statistics, vocabulary distribution, topic distribution. Common case: training data is filtered for length 200-2048; eval has uniform 50-4096. The model never learned to handle short or long inputs.

(5) **Data leakage in reverse** : is the eval set somehow easier than training (filtered to "good" examples)? Check eval set construction.

(6) **Bisect the training run** : load the checkpoint at step N and step N/2; eval both. Where did the divergence start? If it started immediately, the issue is in the data pipeline or template; if it grew gradually, overfitting or distribution shift.

The pattern across these: _each step rules out a class of cause_. Most "scientific" failures end up being tokenizer mismatches or eval methodology bugs, not deep modeling issues.

### What just happened?

  * Training failures fall into **four layers** : numerical (NaN, inf), algorithmic (wrong loss/gradient), systems (OOM, hang, slow), scientific (loss fine, model bad).
  * Each layer has its own diagnostic toolkit. **Identifying the layer is the first move** — wrong tool for the layer wastes time.
  * **The NaN walking tree** : 7 causes in priority order — fp16 overflow, missing fp32 in softmax/norm, bad init, LR too high, unstable loss formulation, mixed-precision accumulator, bad data. Walk it in order.
  * **Forward-hook sentinel** : install a hook on every module that fires on the first NaN/inf output. Catches the cause near the source instead of the symptom 50 layers later. ~2-5% throughput cost.
  * **Backward hooks** work analogously for backward NaNs (typical cause: division by saved-forward zero in backward).
  * **gradcheck in fp64** is mandatory for hand-written backwards. Tight tolerances catch math bugs the regular training won't.
  * **Overfit-tiny test** : can the model drive loss to near-zero on 16-64 examples? If not, structural bug. The single most useful sanity check in deep learning.
  * **Reference comparison** : diff your implementation against a reference (Hugging Face, PyTorch). Same inputs through both, outputs should match to numerical precision.
  * **Six loss curve archetypes** : healthy (smooth descent), LR-too-high (dip then explode), plateau-near-random (structural bug), spikes-that-recover (grad clip working), spikes-that-don't (numerical instability), train-eval-divergence (overfit OR data leakage OR distribution shift).
  * **Per-layer gradient norm tracking** reveals patterns invisible in total grad norm. Vanishing in early layers → init issue. Exploding in late → init at depth. Steady drift up → optimizer divergence.
  * **OOM ladder** : reduce microbatch → bf16 → activation checkpointing → FSDP → offload optimizer → quantization → tensor parallel → pipeline parallel. Each step has higher engineering cost; start at the top.
  * **Distributed hangs** : usually divergent code paths between ranks. Set `NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL` to see what each rank is doing.
  * **Scientific bugs** are silent failures: train-eval divergence, tokenizer mismatches, eval methodology bugs. Memorization probes, tokenizer round-trip checks, and distribution-shift audits are the diagnostic tools.
  * **Reproducibility checklist** : seed torch+numpy+CUDA, seed dataloader workers, set deterministic algorithms, pin library versions. Bit-exact reproducibility costs ~20-50% throughput.
  * **The bisection mindset** : when you don't know where the bug is, halve the search space recursively. Each halving cuts candidates by 50%; eight halvings localize among 1000 candidates to ~4.
  * The discipline: **localize first, fix second**. Random fixes obscure bugs; patient diagnosis reveals them. Junior engineers reason when they should bisect; senior engineers bisect earlier.
  * The reflex: when training breaks, ask "which of the four layers?" Then apply that layer's tools. Don't reach for a profiler when loss is NaN; don't reach for hooks when eval is bad.

This concludes Part IX. M29 built a transformer end-to-end, M30 unpacked RoPE, M31 covered post-training, M32 (this module) consolidated debugging, and M33 explored Mamba and SSMs. Combined with the original 28 modules, you now have a complete framework: from `x.stride()` to `fully_shard` to `tl.dot` to RoPE rotations to DPO to parallel scans — and the debugging skills to make them all work in practice.
