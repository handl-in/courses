# Module 34 — Reasoning models: long-thinking, GRPO & test-time compute

# _Reasoning models:_ long-thinking, GRPO & test-time compute

_Part X · Module 34 · the reasoning frontier_

— how o1- and R1-style models train themselves to reason via verifiable rewards, why GRPO replaced PPO for this purpose, and the test-time-compute architecture that turns "more inference compute" into "better answers"

\--- 

Through M33 we've covered standard transformer training and serving. M34 turns to what's arguably the most consequential post-training development of 2024-2025: **reasoning models**. These are models that, when given a hard problem, "think" by generating thousands or tens of thousands of intermediate tokens before producing a short final answer. OpenAI's o1, DeepSeek's R1, and their successors are this shape. The capability gain on math, code, and formal reasoning has been substantial.

The architectural change is roughly zero. A reasoning model's transformer is the same transformer you've built (M29). What's different is two things: **how they're trained** (RL with verifiable rewards, not preference data) and **how they're served** (long generations, test-time compute search). M34 covers both, with a focus on the algorithm — **GRPO** — that made R1's training feasible without needing a reward model. By the end you'll know what's actually happening when "the model is reasoning," what makes it work, and the serving infrastructure that handles 100K-token outputs.

> **★ KEY IDEA**  
>  A reasoning model is a transformer trained to _think before answering_ — produce a long chain of intermediate tokens, then a short final answer. The training algorithm of choice is **GRPO** (Group Relative Policy Optimization): for each prompt, sample K responses (e.g., 16-64), score them with a **programmatic verifier** (math equality, code unit tests, formal proof check), compute advantages relative to the group mean, update via a clipped policy gradient. **No reward model, no value model** — the verifier replaces both. R1 demonstrated that with this setup and enough compute, the model _spontaneously learns_ to produce long chains of thought, including backtracking and error correction. At inference, reasoning models are served with long decode budgets (10K-100K tokens), and additional quality comes from **test-time compute** : best-of-N, majority voting, tree search, and verifier-guided search. The Pareto frontier between inference compute and answer quality is the new dimension along which reasoning systems are tuned. 

## Two new faces — the reasoning pair

R

Reasoner

"I think for 10,000 tokens before I answer. My final answer is short, but the path is what matters."

When you ask me a hard math problem, I don't just produce the answer. I work through it: try an approach, check if it fits, abandon it if it doesn't, try another. _Most of my output is intermediate reasoning_ ; the answer is the last few tokens. I emerged from RL training with a programmatic verifier — when the verifier rewarded correct final answers, I gradually learned that long, careful, self-correcting reasoning produced more correct answers than short guesses. **I wasn't taught to reason; I learned that reasoning works.** My architecture is identical to a normal transformer (M29). The difference is what trained me.

V

Verifier

"I don't have opinions. I check answers — and I'm always right about what I check."

For math: I evaluate whether two expressions are equivalent (sympy, numerical comparison). For code: I run unit tests. For formal proofs: I check the proof terms. _I produce a single bit per response: correct or incorrect_ — sometimes a small set of partial-credit signals, but mostly binary. I can't grade essays, evaluate creativity, or judge ethics. But on the tasks I cover, I'm **perfectly aligned** by construction — I don't suffer from the reward-model problems M31 covered (out-of-distribution, hacking, drift). I'm what makes RL-from-scratch on reasoning tasks work without humans in the loop. My existence is also the limit on which tasks reasoning models can train on directly.

## What changed: from preference learning to verifiable rewards

M31 covered post-training under the assumption that quality is judged by humans (or AI judges trained on human judgment). For most chat tasks — was this response helpful? polite? safe? — that's the only available signal. There's no ground truth for "is this a good response to _tell me a joke_?"

For some tasks, ground truth exists. **Math problems have correct answers** — sympy can check equivalence. **Code has tests** — run the unit tests. **Formal proofs check syntactically** — Lean or Coq either accept the proof term or don't. For these tasks, the path that classic RLHF takes (collect preferences, train RM, run PPO) is wasteful. _Why approximate human judgment of correctness when correctness is directly checkable?_

This is the insight that made reasoning models possible. With a verifier, the reward signal is exact, free, and not subject to gaming. You can sample millions of responses and grade each in milliseconds. The training-data bottleneck of RLHF (expensive human labels) goes away. The reward-model failure modes (drift, hacking, out-of-distribution) go away. **The remaining question is just: what RL algorithm works well in this regime?**

## GRPO: the algorithm that worked

GRPO — Group Relative Policy Optimization — is the algorithm DeepSeek used for R1. It's a streamlined PPO variant designed specifically for verifiable-reward settings. The key simplifications:

  * **No reward model** : the verifier produces rewards directly.
  * **No value model** : the baseline for variance reduction comes from the _group mean_ — sample K responses to the same prompt, use the mean as the baseline.
  * **Just policy + reference** : like DPO (M31), but with sampled rollouts and an explicit RL objective.

The result: a clean two-model setup (policy + frozen reference) with the simplicity of DPO and the exploration capability of PPO.

GRPO step: sample K, verify each, compute group-relative advantages, update policy prompt "Solve: 2x+5=11" K rollouts r₁: "x=3 ✓ ..." (correct) r₂: "x=4 ✗ ..." (wrong) r₃: "x=3 ✓ ..." (correct) ... (K=16 typical) Verifier scores R₁ = 1.0 R₂ = 0.0 R₃ = 1.0 R₄ = 0.0 group mean = 0.5 Advantages A_i = R_i − mean A₁ = +0.5 A₂ = −0.5 Policy update — PPO-style clipped objective with KL to reference for each rollout i: ratio_i = π(r_i | x) / π_old(r_i | x) # importance ratio surrogate_i = min(ratio_i · A_i, clip(ratio_i, 1−ε, 1+ε) · A_i) L = -mean(surrogate) + β · KL(π ‖ π_ref) no reward model, no value model — verifier replaces both advantage from group mean = baseline for free

### The GRPO loss in code

Compare this directly with M31's DPO loss to see the structural similarity (and differences):
    
    
    def grpo_loss(policy, ref, prompts, K=16, eps=0.2, beta=0.04):
        # 1. ROLLOUT: sample K responses per prompt from the current policy.
        #    In production, this is done by vLLM workers (M27); here we sketch the logic.
        with torch.no_grad():
            responses = []                          # list of K · len(prompts) responses
            old_logprobs = []                       # cached for the importance ratio
            for _ in range(K):
                r = policy.generate(prompts, do_sample=True, temperature=1.0)
                responses.append(r)
                old_logprobs.append(compute_logprobs(policy, prompts, r))
    
        # 2. VERIFY: programmatic check on each response.
        rewards = torch.tensor([
            verify(prompts[i % len(prompts)], responses[i])    # 0.0 or 1.0 typically
            for i in range(len(responses))
        ])
    
        # 3. ADVANTAGE via group mean — for each prompt's group of K responses,
        #    advantage = reward - group_mean. Optionally normalize by group std.
        rewards = rewards.view(len(prompts), K)
        group_mean = rewards.mean(dim=1, keepdim=True)
        group_std = rewards.std(dim=1, keepdim=True) + 1e-8
        advantages = ((rewards - group_mean) / group_std).flatten()
    
        # 4. POLICY UPDATE: PPO-style clipped surrogate, with KL penalty against ref.
        new_logprobs = compute_logprobs(policy, prompts, responses)
        with torch.no_grad():
            ref_logprobs = compute_logprobs(ref, prompts, responses)
    
        # Importance ratio (per-token, summed over response tokens)
        ratio = (new_logprobs - torch.cat(old_logprobs)).exp()
    
        # Clipped objective — same as PPO from M31
        surrogate1 = ratio * advantages
        surrogate2 = ratio.clamp(1 - eps, 1 + eps) * advantages
        pg_loss = -torch.min(surrogate1, surrogate2).mean()
    
        # KL penalty to reference: stops policy from drifting too far
        kl = (new_logprobs - ref_logprobs).mean()
    
        return pg_loss + beta * kl

Three things to call out, all setting GRPO apart from the M31 algorithms:

  * **The K-sample rollout** : each prompt produces K responses, all from the current policy. This is where the exploration happens — sampling at temperature 1.0 produces diverse responses; the verifier picks the wheat from the chaff. K=16 is a common starting point; production reasoning runs use K up to 64 or 128.
  * **The group-mean baseline** : subtracting the group mean from each reward gives a per-rollout advantage. _Equivalent to a value-model baseline but free._ Normalizing by group std (the "Z-score" form above) further reduces variance and improves training stability — this is one of the small details in production GRPO that makes a real difference.
  * **The PPO-clipped objective** : GRPO uses PPO's clipped surrogate to bound policy updates. The clip prevents any single sample from causing a large policy change, which would destabilize training.

The whole loss is ~25 lines. **The complexity is not in the algorithm; it's in the rollout infrastructure.** Generating K responses per prompt for thousands of prompts per training step is expensive — typically the rollout dominates total training compute, more so than for PPO. Production GRPO setups separate rollout (vLLM workers, M27) from training (the gradient updates) and treat them as a producer-consumer pipeline.

## The chain-of-thought emergence story

The most striking finding of the R1 paper: **with the GRPO setup above and a base model that's been pretrained on reasoning content, the model spontaneously learns to produce long chains of thought**. Not because anyone trained it to. Because long, careful reasoning produces correct answers more often than short guesses, and the verifier rewards correct answers.

The trajectory R1 reported during training:

  * **Early steps** : model produces short answers, often wrong. Verifier rewards few. Gradients push the policy toward whatever responses got the rewards.
  * **Mid-training** : model starts producing intermediate reasoning steps. Average response length grows from a few hundred tokens to a few thousand. Accuracy on the math benchmarks climbs.
  * **Later** : _the model learns backtracking_. Spontaneous "wait, let me reconsider" patterns emerge. Self-correction, alternative-approach exploration, error checking. Average response length climbs to 5K-10K tokens.
  * **The "aha moment"** : at some training step, the model starts reliably producing extended reasoning chains for hard problems. This is an emergent transition, not a gradual one.

None of this was supervised. The reasoning patterns weren't in the SFT data; they emerged because the verifier kept rewarding correct final answers, and long reasoning was the path that produced more correct final answers. **This is the most consequential finding in post-training since RLHF itself** — that with the right reward signal, complex behaviors can emerge from RL alone.

That said, R1's full recipe wasn't pure RL-from-scratch. The published recipe involved:

  1. **Cold-start SFT** : a small amount of SFT on hand-written reasoning traces, to give the base model a starting distribution of "how reasoning is structured." Without this, RL training was unstable.
  2. **RL phase 1 (reasoning)** : GRPO on math/code tasks with verifiers. Long-CoT capability emerges here.
  3. **SFT phase 2** : collect the model's own outputs from the RL phase, filter for correct ones, SFT a fresh checkpoint on this distilled data. Mixed with general-purpose SFT data.
  4. **RL phase 2 (alignment)** : standard preference learning (DPO-style) for safety, helpfulness, honesty.

Step 2 is where the reasoning capability comes from; step 1 is the prerequisite that makes step 2 stable; steps 3-4 turn the reasoning model into a deployable product (the bare R1-zero from step 2 has reasoning capability but is hard to use).

## Reasoning model serving: the test-time-compute architecture

Once you have a reasoning model, serving it is qualitatively different from serving a standard chat model. The change comes down to one fact: **generations are long**. Where a chat response is 100-500 tokens typical, a reasoning response is 5K-50K tokens, sometimes more.

This stresses the M27 inference stack in specific ways:

  * **KV cache pressure goes up dramatically**. A 32K-token reasoning response × 32 layers × 8 KV heads × 128 head dim × 2 bytes = ~2 GB just for KV cache for one request. Paged KV (M27) and aggressive page reuse become essential.
  * **Per-request latency is fundamentally tens of seconds**. At 100 tokens/sec generation, 10K reasoning tokens = 100 seconds. There's no fixing this with infrastructure — the model needs to think. _UX shifts to "submit and wait" or "stream the reasoning visibly so users see progress."_
  * **Throughput vs latency tradeoff intensifies**. Reasoning serving is the regime where larger batches matter most — the per-request latency is so high that batching up many concurrent requests is the only way to keep GPUs busy. Continuous batching (M27) and speculative decoding (M27) do more work per H100 here than on standard chat.
  * **Prefill-vs-decode ratio changes**. Standard chat: short prompt, short response — roughly balanced. Reasoning: short prompt, very long response — decode dominates compute by 10-100×. This shifts what optimizations matter.

The architectural pattern that emerges: **reasoning models are typically served on dedicated infrastructure** , separate from regular chat models. Different batch sizes, different memory budgets, different SLOs. vLLM and TensorRT-LLM both have reasoning-mode configurations.

## Test-time compute: spending more compute for better answers

The serving infrastructure gives you the per-request mechanics. But reasoning models open up a new dimension that standard chat doesn't have: **spend more inference compute, get better answers**. This is the test-time-compute story.

Test-time compute Pareto: spend more inference, get better quality inference compute (log scale: tokens generated × verifier evals) accuracy on hard task 1× 10× 100× 1000× 10000× 0% 25% 50% ~80% greedy: flat (no extra compute) best-of-N (with verifier) majority voting (no verifier) verifier-guided search single long-CoT key insight: this dimension didn't exist for standard chat models

The simplest test-time compute method is **just letting the reasoning model think for longer** — give it a higher max-tokens budget. Quality climbs because the model has room to verify its work, try alternatives, catch errors. This is the curve labeled "single long-CoT" in the chart.

But you can go further by _combining multiple inferences_ :

  * **Best-of-N with verifier** : sample N reasoning responses, run the verifier on each, return the one that verified. Strictly better than greedy when the verifier is reliable. Cost: N× more inference compute. Quality: dramatically better — N=64 is often enough to push accuracy from 50% to 80%+ on hard math.
  * **Majority voting** : when no verifier is available (or verification is partial), sample N responses, take the most common final answer. Works because correct answers tend to converge while wrong ones diverge. Less powerful than best-of-N but applicable when verification is impossible.
  * **Verifier-guided beam / tree search** : at each step in the reasoning, branch into multiple continuations, evaluate each branch with a process reward model (PRM), prune low-scoring branches. The most computationally intensive option but also the most quality-efficient on hard problems. Frontier reasoning systems (rumored) use variants of this.

### PRM vs ORM: outcome vs process rewards

Two reward-model styles, useful in different parts of the test-time-compute story:

  * **Outcome Reward Model (ORM)** : scores the final answer. Cheap to use (one call per response), trained on (response, correct/incorrect) pairs. Good for best-of-N where you only care about the final answer. _Limitation_ : can't tell you which step in the reasoning went wrong.
  * **Process Reward Model (PRM)** : scores each step in the reasoning chain. Trained on (step, was-this-step-correct) pairs — typically labeled by humans or by an ORM applied to truncated chains. Used during search to prune bad branches early. _Cost_ : needs more annotations to train; one call per reasoning step at inference time.

For pure verifier-friendly tasks (math, code), the verifier itself replaces both. For tasks where verification is partial or expensive, ORMs and PRMs let you scale test-time search at lower cost.

## Distillation from reasoners

One of R1's most practically impactful findings: **reasoning capabilities transfer to smaller models via SFT on the bigger model's outputs**. Take a 70B reasoning model, generate reasoning traces on math/code problems (filtered for correctness via verifier), and SFT a 7B base model on those traces. The 7B model inherits substantial reasoning capability without needing its own RL training.

The distillation recipe roughly:

  1. Take a strong reasoning model (the teacher).
  2. Generate reasoning traces on a large pool of math/code problems. Use temperature 0.7-1.0 to get diversity.
  3. Filter: keep only traces where the final answer was correct (via verifier).
  4. SFT a smaller base model (the student) on the filtered traces. Standard SFT — instruction format, prompt-masked loss.

The result: the student is much weaker than the teacher (smaller models are weaker), but its reasoning capability is much stronger than direct SFT on equivalent-size data. R1-distill 7B and 14B models are notably competitive with much larger non-reasoning models on math.

_Why does this work?_ Probably because reasoning is partly a "behavioral" capability — knowing the structure of a good reasoning trace, when to backtrack, what to verify — that can be imitated from examples. Some of the underlying reasoning capacity may be missed (a 7B model has less raw capacity than a 70B), but enough transfers to make this a practical recipe.

Distillation is also _much cheaper_ than RL training. SFT compute for a 7B model on 100K reasoning traces is maybe a day on 8 H100s; RL training of a reasoning model is weeks on hundreds. **Most production reasoning models people deploy are distilled, not trained from scratch with RL.**

## What works, what's still hard

The reasoning-model recipe works well for tasks with verifiable rewards. The main domains as of 2026:

Reasoning model task domains and verifier choices Domain| Verifier| Maturity  
---|---|---  
Math (AIME, MATH, AMC)| sympy equivalence; numerical match| Standard; widely deployed  
Competitive programming| unit tests / hidden tests| Standard; competitive with humans on Codeforces-style tasks  
Software engineering| repo unit tests + functional tests| Active; SWE-bench-style benchmarks  
Formal proofs (Lean, Coq)| proof checker| Active research; promising  
Research math| partial (steps verifiable, end-to-end not)| Frontier; PRM-based search becomes essential  
Scientific reasoning| partial (calculations checkable, hypotheses not)| Active; mostly empirical  
Long-form writing, creativity| None (no programmatic check)| Doesn't directly apply  
Persuasion, ethics, judgment| None| Doesn't apply; back to RLHF (M31)  
  
The pattern: **where a verifier exists, RL with verifiable rewards works extraordinarily well**. Where it doesn't, classic RLHF (M31) is still needed. Most production reasoning systems are hybrids: RL-from-verifier on the math/code subset, RLHF on the chat/safety subset.

Open questions still very much being worked on:

  * **Generalization beyond the trained domain** : a model trained on competition math — does it improve at reasoning generally? Empirical evidence is mixed; transfer to non-math reasoning tasks is real but smaller than within-domain gains.
  * **Process rewards from human preferences** : getting humans to label "was this reasoning step correct" reliably is hard. Most PRMs use AI feedback; human-labeled PRMs are more accurate but expensive.
  * **The hallucination-reasoning tension** : long reasoning chains have more opportunities to hallucinate facts. Reasoning models often confidently compute wrong arithmetic in their traces. Active mitigation work.
  * **Compute scaling laws for test-time vs train-time** : how do you optimally allocate a fixed compute budget between training a bigger model, training longer, and spending more at inference time? Empirically this is being actively studied; theoretical understanding is partial.

#### Q&A; — About reasoning models **Q:** Why does GRPO work without a value model when classic PPO needs one? **A:** PPO's value model exists for variance reduction — to subtract a baseline from the reward so the gradient signal isn't dominated by reward magnitude noise. GRPO gets a baseline for free: the mean reward across the K samples for the same prompt is itself a baseline (and, after subtracting, advantages have zero mean by construction). Normalizing by the group std makes this even more PPO-equivalent. _The tradeoff_ : GRPO requires K samples per prompt, where PPO uses 1 — but verifier evaluation is cheap, and in the LLM regime K=16-64 samples cost less than maintaining a separate value-model train loop. _For verifiable-reward settings, GRPO dominates PPO on engineering simplicity at no quality cost._ **Q:** If reasoning models emerge from RL, why is the cold-start SFT step needed? **A:** Stability. R1's authors reported that pure RL from the base model was unstable — the base model's response distribution was too far from "reasoning trace" structure for the RL signal to find good gradients quickly. The cold-start SFT (a few thousand examples of human-written reasoning traces) gives the model a starting distribution where reasoning-shaped responses are already plausible. Then RL refines and extends. _Without cold-start, training works but takes much longer and may get stuck in degenerate local optima._ The published "R1-Zero" variant skipped cold-start and reached good but lower quality than R1 proper. **Q:** Why does distillation work? It seems like it shouldn't transfer the underlying reasoning capability. **A:** It transfers the _behavioral pattern_ of reasoning more than the underlying capability. The student learns to produce reasoning-shaped outputs (intermediate steps, verification, backtracking patterns) from the teacher's traces. Some of the underlying problem-solving ability is captured (the smaller model can follow the same algorithmic patterns); some isn't (smaller models hit harder ceilings on really difficult problems). But for tasks within the smaller model's capacity, the reasoning structure is a major boost. Empirically, R1-distill 14B reasoning beats GPT-4o non-reasoning on AIME-level math by a wide margin. _Reasoning is partly a "skill" that can be imitated, partly a "capacity" that can't._ **Q:** Can I use GRPO for non-verifiable tasks? **A:** Sort of. You can replace the verifier with a reward model and run GRPO with that as the score function. This works and is sometimes called "GRPO-RM." But you've now added back the reward model, with all its drift and hacking risks (M31). At that point, the engineering simplification GRPO offered is partly lost. _For non-verifiable tasks, DPO is usually a better choice_ — its closed-form bypasses the rollout cost. GRPO's main wins (group baseline, no value model, exploration) are most valuable when the reward signal is exact, which is exactly the verifier setting. **Q:** How do reasoning models interact with MoE architectures (M28)? **A:** DeepSeek-V3 and DeepSeek-R1 are both MoE — V3 is the base, R1 is V3 post-trained for reasoning. The MoE structure doesn't fundamentally change the reasoning training: GRPO works the same way on an MoE policy. What changes is the engineering: rollouts are slower (MoE inference has overhead from routing and all-to-all collectives), and serving a reasoning MoE has both the long-decode characteristics of reasoning _and_ the memory characteristics of MoE (whole model resident in memory even though only some experts active per token). _It's the union of both M28's and M27's complexities; not multiplicative, but additive._ **Q:** When does reasoning compute become more cost-effective than training a bigger model? **A:** An active research question, but the empirical pattern: at the high end of difficulty (frontier math, hard programming), test-time compute (best-of-N, verifier-guided search) gives steeper quality-vs-compute curves than train-time scaling. At the low end (easy questions), the model usually gets it on the first try and extra inference compute is wasted. _Cost-effectiveness inverts at the difficulty boundary_ : for easy chat, train a bigger model; for hard reasoning, lean on test-time. This is also why "reasoning models" are usually served separately — different cost-effective operating points. 

## Code Magnets: implement the GRPO advantage

You're computing the per-rollout advantages for one batch of GRPO. Three magnets are wrong choices.

Arrange the magnets to compute the advantages.

rewards = torch.tensor([verify(p, r) for p, r in zip(prompts_repeated, responses)]) rewards = rewards.view(num_prompts, K) rewards = rewards.view(K, num_prompts) group_mean = rewards.mean(dim=1, keepdim=True) group_mean = rewards.mean() group_std = rewards.std(dim=1, keepdim=True) + 1e-8 advantages = ((rewards - group_mean) / group_std).flatten() advantages = (rewards - group_mean).flatten()

show solution
    
    
    rewards = torch.tensor([verify(p, r) for p, r in zip(prompts_repeated, responses)])
    rewards = rewards.view(num_prompts, K)
    group_mean = rewards.mean(dim=1, keepdim=True)
    group_std = rewards.std(dim=1, keepdim=True) + 1e-8
    advantages = ((rewards - group_mean) / group_std).flatten()

The traps:

  * `rewards.view(K, num_prompts)`: wrong reshape order. The rollout loop iterates K times for each prompt, then prompts; flattening that gives shape (num_prompts × K) — reshaping to (num_prompts, K) groups responses for the same prompt together. Reversing it groups across prompts, which makes the group mean meaningless.
  * `group_mean = rewards.mean()`: takes the global mean across all prompts and all rollouts. This loses the per-prompt baseline — easy prompts and hard prompts get the same baseline, and easy-prompt advantages are crushed while hard-prompt advantages are inflated. The whole point of "group relative" is per-prompt baseline; `mean(dim=1, keepdim=True)` is what gives that.
  * `advantages = (rewards - group_mean).flatten()`: skips the std normalization. The unnormalized form works (and is what the original GRPO paper used) but is more sensitive to reward scale and can produce very different gradient magnitudes when verifiers return different reward ranges. Dividing by group std (the Z-score form) makes the algorithm scale-invariant and more stable in practice. _Production GRPO implementations use the normalized form._

The pattern: **reshape to (prompts, K) → per-row mean for baseline → per-row std for normalization → subtract mean, divide by std → flatten back to (prompts × K) for the policy update**. The reshape and the per-prompt aggregation are what make GRPO "group relative."

## Who does what?

Match each reasoning-model concept to its real role.

Concept

Real role

Verifier

A. Programmatic check on a response (sympy, unit tests, proof checker); produces a score with no learned bias.

GRPO

B. RL algorithm that uses K rollouts and group-mean baseline instead of a value model.

Group-mean baseline

C. Per-prompt advantage = (reward − group_mean) / group_std; replaces the value model.

Cold-start SFT

D. Pre-RL stage on hand-written reasoning traces; gives the policy a stable starting distribution.

Best-of-N with verifier

E. Sample N responses, return the one that verified; major test-time-compute lever.

Process Reward Model (PRM)

F. Scores each step in a reasoning chain; used to prune branches in tree search.

R1-distill

G. SFT a smaller base model on filtered (verified-correct) reasoning traces from a larger reasoner.

show solution

**Verifier** → A  
**GRPO** → B  
**Group-mean baseline** → C  
**Cold-start SFT** → D  
**Best-of-N with verifier** → E  
**Process Reward Model** → F  
**R1-distill** → G 

The mental shortcut: _verifier scores, GRPO trains, group mean baselines, cold-start initializes, best-of-N searches, PRM scores steps, distillation transfers_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team trains a reasoning model with GRPO. After 5,000 steps, the model produces correct answers on training problems with high accuracy, but average response length has dropped to about 50 tokens — the model just states the answer. What's likely going on, and what would you change?

show answer

The verifier is rewarding correct final answers regardless of how the model got there. If the base model can pattern-match many of the training problems directly (perhaps they're well-represented in pretraining data), it learns that _shorter responses_ with the right answer get the same reward as long reasoning chains. Why think when you can just write "x = 3"? Group-mean advantages don't penalize short responses; they reward whatever produces the highest reward in the group.

The fix is reward shaping. Two common approaches: (1) **Length-aware rewards** : small bonus for responses that include intermediate reasoning steps, or penalty for very short responses on hard problems. (2) **Difficulty-stratified training** : train on problems where the base model fails frequently — those force the model to actually reason rather than pattern-match. (3) **Process rewards (PRM)** : explicitly reward the existence of structured reasoning steps, not just final correctness. The R1 paper noted this exact issue and addressed it by carefully selecting training problems that the base model couldn't solve directly. _Reward shaping is half the engineering art of reasoning-model training_ ; the GRPO algorithm itself is the easier part.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Walk through why GRPO uses K samples per prompt instead of just one (like vanilla policy gradient). What's lost if K=1?

show answer

Two things are lost with K=1.

First, **the baseline disappears**. The whole point of "group relative" is that `advantage_i = reward_i − mean(rewards in group)` — the group mean is the baseline. With K=1, the "group" is one sample, the mean is the single reward, advantage is exactly zero, gradient is zero. _You'd need a value model to recover any baseline_ , defeating the no-value-model property.

Second, **exploration is lost**. With K=1 you're doing on-policy training: the policy generates one response, the verifier scores it, gradient nudges the policy. With K=16, you're sampling 16 diverse responses for each prompt, including some that pure greedy would never produce. The verifier identifies which of those 16 was correct (or most correct), and the gradient pushes the policy toward producing those. _Exploration via temperature-1 sampling is what lets the policy learn behaviors not in its current distribution_ — like long reasoning chains when it currently produces short ones.

Practically, most GRPO implementations use K=8 to 64. Higher K means more exploration but more rollout compute. K=16 is a good starting point.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** A team's reasoning model serves with 8K-token max generations. Latency is acceptable but their math benchmark accuracy is 55%. They have spare GPU capacity and want to push accuracy higher without retraining. What test-time compute approach would you suggest, and what's the likely trade?

show answer

The most reliable lever is **best-of-N with the verifier**. For each problem, generate N=16 (or 32, 64) responses, run the verifier on each, return the first that verified. On hard math benchmarks, best-of-16 with a strong verifier typically pushes accuracy from 55% to 75-85% — the model often produces a correct answer in _some_ of the 16 samples even when greedy is wrong. The verifier picks it out. **Cost** : 16× more inference compute per problem. **Quality** : a substantial step up.

If they don't have a usable verifier at inference time (e.g., serving end-users with no formal answer to check against), the alternative is **majority voting** : sample N responses, take the most common final answer. Less powerful than verifier best-of-N but applicable without ground truth. Typical gain: 55% → 65-70% with N=16. Still substantial.

Higher-end option: **verifier-guided tree search** with a PRM. At each reasoning step, branch and evaluate; prune low-PRM branches. Most expensive (often 100×+ vs greedy) but pushes the highest-difficulty problems. Probably overkill if best-of-N gets them to 80% — diminishing returns past that.

The trade space is well-defined: pick where on the Pareto frontier you want to operate. Verifier best-of-N at N=16 is usually the right entry point.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Why are reasoning models a poor fit for tasks like creative writing, even though they handle complex math beautifully?

show answer

Three reasons that compound.

(1) **No verifier exists for creative writing**. The whole reasoning-model recipe relies on a programmatic check on the final output. There's no equivalent for "is this poem good" or "is this story creative." The training signal that produces reasoning behavior simply doesn't apply.

(2) **The reasoning behavior itself is misaligned with creative tasks**. A reasoning model trained on math will, when asked to write a poem, often produce intermediate reasoning ("Let me think about what makes a good poem...") followed by a poem. The reasoning prefix is verbose and rarely improves the creative output — sometimes it actively hurts by anchoring the model on its first thoughts about structure rather than letting it generate fluidly.

(3) **Long-thinking and creativity have different optima**. Reasoning emphasizes correctness, verification, backtracking — useful for math, harmful for novel idea generation where the first draft is often the most creative. Production reasoning models often have a "do not engage reasoning mode" flag for tasks where it's counterproductive.

The practical pattern: **route by task type**. Hard math/code goes to the reasoning model; chat/creative goes to a standard chat model. Some products do this transparently; some let the user choose. A unified model that's optimal at both is still future work — if it's even achievable.

### What just happened?

  * Reasoning models — o1, R1, and successors — are transformers that learn to **think before answering** : produce thousands of intermediate reasoning tokens, then a short final answer.
  * The architectural change is roughly zero. The training and serving change radically.
  * **Verifiable rewards** replace human preferences for tasks where ground truth exists: math (sympy), code (unit tests), formal proofs (proof checker). The verifier is exact, free, and immune to the reward-model failure modes from M31.
  * **GRPO** is the RL algorithm of choice for verifiable-reward settings. Sample K rollouts per prompt, score each, advantage = (reward − group_mean) / group_std. PPO-clipped objective, KL penalty against frozen reference.
  * **Two models in memory** (policy + reference) — same as DPO. **No reward model, no value model** — the verifier replaces the RM, the group mean replaces the value baseline.
  * **Long-CoT emergence** : with the right reward shaping, the model spontaneously learns reasoning patterns including backtracking and self-correction — none of which were in the SFT data. R1's "aha moment."
  * **R1's full recipe** : cold-start SFT on hand-written reasoning traces → GRPO on math/code → SFT on filtered self-generated traces → final RLHF for safety/alignment.
  * Serving reasoning models stresses M27's stack: KV cache pressure (long generations), per-request latency in tens of seconds, throughput-vs-latency tradeoff intensified, decode-dominant compute mix.
  * **Test-time compute** : a new dimension. Beyond "make the model thinker longer," combine multiple inferences via **best-of-N with verifier** , **majority voting** , **verifier-guided tree search**.
  * ORM (outcome reward model) scores final answers — cheap, used in best-of-N. PRM (process reward model) scores each reasoning step — more expensive, used in tree search.
  * **Distillation from reasoners** : SFT a smaller base model on the bigger model's verifier-filtered traces. Inherits much of the reasoning capability at a fraction of the training cost. R1-distill family.
  * The recipe works where verifiers exist (math, code, proofs). For creative writing, persuasion, ethics, judgment — back to RLHF (M31). Production systems route by task type.
  * The reflex: when the task has an exact correctness check, reach for verifiable-reward RL. When it doesn't, reach for preference-based RL. _Match the algorithm to the available signal._

This module ties together M27 (serving long generations), M28 (MoE — many reasoning models are MoE), and M31 (the post-training pipeline that GRPO sits in). The reasoning-model recipe represents the most consequential post-training development since RLHF itself, and the engineering it requires — verifier infrastructure, rollout pipelines, test-time-compute search systems — is rapidly maturing into standard practice.

If a Part X continues, the natural next module would be **pretraining data engineering at scale** — quality classifiers, domain mixing, deduplication pipelines, contamination filtering. Quietly the biggest source of modern quality gains and almost no public material at the engineering level.
