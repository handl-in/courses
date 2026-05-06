# Module 39 — Distributed RLHF & GRPO infrastructure

# _Distributed RLHF & GRPO infrastructure:_ Ray, Hybrid Engine, and the four-models-in-memory problem

_Part XI · Module 39 · April 2026 currency_

— why RLHF is 80% rollout, how OpenRLHF and veRL diverge architecturally, what Hybrid Engine actually does, and why REINFORCE++ became the frontier algorithm in 2025-2026

\--- 

M31 covered the post-training algorithms — PPO, DPO, reward model training. M34 covered the GRPO algorithm and the reasoning-model recipe. Both treated the algorithms in isolation. _This module is about the infrastructure_ : what it actually takes to run RLHF or GRPO at scale, what the dominant frameworks (OpenRLHF, veRL) do differently, and why "Hybrid Engine" became the production default. The systems engineering of distributed RL is now its own discipline; nothing in M1-M38 covered it directly.

The reality: **RLHF spends ~80% of compute on rollout, not on the gradient update.** The training step is the small part; generating trajectories from the policy under training (with the rollout engine) and scoring them (with reward and reference models) is what dominates. A 70B Actor + 70B Reference + Reward + Critic configuration consumes 8-16 H100s just for model weights before optimizer states. Coordinating these four models — some training, some frozen, all communicating — is harder than coordinating any single-model training run you've seen. By April 2026, the public frameworks have settled into recognizable patterns; the recipes are teachable. This module is what you need before training your own reasoning model.

> **★ KEY IDEA**  
>  RLHF/GRPO is a **systems problem disguised as an ML problem**. PPO requires four models running concurrently: **Actor** (policy being trained), **Reference** (frozen baseline for KL), **Reward** (frozen scorer), **Critic** (value head, PPO only). GRPO eliminates the Critic by group-normalizing rewards within the same prompt; REINFORCE++ baseline (the 2025-2026 frontier choice) eliminates both Critic and group-rollout overhead. **~80% of compute goes to rollout** (sample generation), so the framework's job is keeping rollout fed and minimizing GPU idle time. Two architectural patterns dominate. **OpenRLHF** : dedicated Ray worker group per model, colocated by rank, vLLM-accelerated rollout, ZeRO-3 training. **veRL** : single unified WorkerDict housing all models, naturally shares resources. Both support **Hybrid Engine** — colocate models on the same GPUs and use vLLM sleep/wake to swap between training and inference modes — which became the production default. The disaggregated alternative (StreamRL, AReal) splits rollout to inference-optimized hardware and training to training-optimized hardware, async pipeline; better for very large rollout ratios. **Distributed SFT** is much simpler — single model, standard data parallel — but has its own concerns (packing, sequence parallelism for long contexts). **Algorithm choice drives infrastructure** : PPO needs four models, GRPO needs three (no Critic), REINFORCE++ needs two (no Critic, no group rollout). Frontier 2026 production: Magistral, ProRL V2, ScaleRL all use REINFORCE++ baseline. 

## Two new faces opening Part XI

R

Rollout

"I'm where 80% of the compute goes. My throughput determines how fast you train."

Every RLHF step starts with me. The Actor needs trajectories — sequences of generated tokens with their log-probs — to compute policy gradients. I produce them. Under the hood I'm vLLM (or SGLang, or TRT-LLM), running with continuous batching and PagedAttention. The Actor's weights synchronize to me at every step (or every K steps with off-policy variants); I generate a batch of rollouts; the trainer scores them and updates the Actor. **If I'm slow, everything is slow.** The frameworks compete on how fast they can keep me fed and how efficiently they can swap my GPU memory back to the trainer when needed. _I'm the bottleneck you didn't expect_.

⊕

Orchestrator

"I schedule four models across N GPUs without anyone stepping on each other."

In 2026, I'm Ray. I'm an actor framework — every model is a Ray actor (a process with its own state and methods); I route messages between them. The Actor calls me to fetch a fresh batch of prompts; I dispatch to Rollout to generate; I send the trajectories to Reward and Reference for scoring; I gather everything and ship it to the trainer. I handle failures (a worker dies, I restart it; gradient sync hangs, I detect it and recover). I handle resource scheduling (which actor gets which GPU, when does Hybrid Engine swap rollout for training). **The reason RLHF can scale to 70B+ models is that I exist** ; without distributed orchestration, you're stuck at single-node and can't fit the four-models reality.

## Why RLHF is a systems problem

To see why distributed RLHF needs its own treatment, walk through what's running concurrently during a single PPO step:

  1. **Actor** generates rollouts. Forward pass on the policy model, sampling tokens autoregressively, recording per-token log-probabilities. Typically uses vLLM for fast generation. Memory: model weights (140GB for 70B in bf16) + KV cache.
  2. **Reference** scores the rollouts for KL divergence. Forward pass on a frozen copy of the policy at SFT initialization. Same architecture as Actor; weights different (not updated). Memory: model weights again (another 140GB for 70B).
  3. **Reward** scores the rollouts for reward signal. Forward pass on a separately trained reward model. Architecture often similar to Actor (regression head replacing LM head). Memory: model weights (another 140GB for 70B-class reward model, less if smaller).
  4. **Critic** estimates value (PPO only). Forward + backward pass; trained alongside Actor. Memory: weights + optimizer states (heavier than Actor because optimizer states for both Actor and Critic).
  5. **Actor backward pass** with PPO loss. Compute advantages, importance ratios, clip, KL penalty; backprop through Actor. Memory: weights + grads + optimizer states (Adam: 2× weights for moments).

The math gets aggressive quickly. **For 70B model class** : 140GB Actor + 140GB Reference + 140GB Reward + 140GB Critic = 560GB just for weights. With Adam optimizer states for Actor + Critic (~280GB), gradients (~280GB), and rollout KV cache (~50-100GB), you're at ~1.2-1.3TB total. That's 16+ H100s (80GB each) just for state — before doing any actual computation.

This is why naive "load four models, run them" doesn't scale. The four frameworks that emerged (OpenRLHF, veRL, RLHFuse, StreamRL/AReal) are different answers to the question: _given this memory pressure and this compute distribution, how do you minimize idle GPU time?_

## The four-model anatomy

PPO four-model topology — what's running, what's training, what's frozen Prompts batch from dataset Actor (policy) vLLM rollout + ZeRO training TRAINED Rollouts tokens + log-probs ~80% of compute Reference SFT-init copy FROZEN Reward trained separately FROZEN Critic (value) value head TRAINED (PPO only) Combined: KL(Actor‖Ref) + reward + advantage PPO loss with clipping; backprop into Actor (and Critic) PPO update → Algorithm choice = which models you need PPO: Actor + Reference + Reward + Critic (4 models) GRPO: Actor + Reference + Reward (3 models — group-normalize rewards instead of value) REINFORCE++ baseline: Actor + Reference + Reward (3 models, simpler than GRPO; ProRL V2 / ScaleRL / Magistral 2025-2026)

### What each model does

**Actor** is the policy under training. It generates rollouts and is updated via gradient descent. In Hybrid Engine setups, Actor wears two hats: the rollout engine (vLLM-served, optimized for inference) and the training process (ZeRO-3-distributed, optimized for backprop). The two configurations share GPU memory but operate at different times.

**Reference** is the SFT-initialized model frozen at the start of RL training. Its purpose: provide a baseline so the Actor's KL divergence can be computed (KL(Actor ‖ Reference)). The KL term in the PPO loss prevents the policy from drifting too far from the well-behaved SFT baseline. Always frozen, always in inference mode.

**Reward** is a separately trained model that scores rollouts. Trained on preference data (chosen vs rejected pairs) using ranking loss; produces a scalar reward per sequence. Always frozen during RL. In 2026, the reward model is increasingly itself an LLM (LLM-as-judge with rubric, see M37) rather than a small classification head, which complicates infrastructure but improves reward quality.

**Critic** is the value head — a model (often a copy of the Actor with a regression head) that estimates expected return from a given state. Trained alongside Actor in PPO. _GRPO and REINFORCE++ eliminate the Critic_ , which is a substantial infrastructure simplification (one fewer trainable model, no Critic optimizer state, no value-function estimation noise).

### Why GRPO and REINFORCE++ won on infrastructure

The original PPO recipe used a Critic to reduce variance in advantage estimates. Two newer approaches eliminate this:

  * **GRPO (DeepSeek, 2024)** : generate K rollouts per prompt; normalize rewards within the group (subtract group mean, divide by group std). The "advantage" is just the normalized reward. No Critic needed. Cost: K× the rollout per training example.
  * **REINFORCE++ baseline** : classical REINFORCE with a moving-average baseline subtracted to reduce variance. No Critic, no group-rollout overhead. The OpenRLHF team analysis (Dec 2024) and follow-on work showed it's _more stable than GRPO and faster than PPO_. Logic-RL and PRIME (Feb 2025) demonstrated this; Magistral (June 2025) used it; ProRL V2 (Feb 2026) trained a SOTA 1.5B reasoning model with it; ScaleRL (Oct 2026) validated at large scale.

The infrastructure implication: _frontier 2026 production is increasingly REINFORCE++ baseline, not PPO or GRPO_. Three models instead of four; simpler scheduling; less memory; same or better quality.

## Ray as the orchestration layer

Why Ray and not a custom solution? Three reasons that compound:

  * **Actor model fits the problem**. Each of (Actor, Reference, Reward, Critic) is naturally a stateful actor — it owns model weights, processes requests (forward pass, training step), reports results. Ray's actor abstraction is exactly this.
  * **Distributed scheduling without writing it**. Ray handles GPU allocation, message passing between actors across nodes, fault recovery (worker dies, Ray restarts it). Building this from scratch is months of work that's already been done.
  * **Heterogeneous resources**. The Rollout engine wants inference-optimized hardware (high memory bandwidth, FP8/FP4 acceleration); the trainer wants training-optimized (high FLOPs, NVLink-rich topology). Ray naturally handles "this actor goes on these GPUs, that actor goes on those GPUs."

The basic Ray actor pattern for a model in RLHF:
    
    
    import ray
    
    @ray.remote(num_gpus=8)
    class RolloutWorker:
        def __init__(self, model_path):
            # Ray manages this actor on dedicated GPUs
            from vllm import LLM
            self.engine = LLM(
                model=model_path,
                tensor_parallel_size=8,
                gpu_memory_utilization=0.5,    # leave room for trainer in Hybrid Engine
                enable_sleep_mode=True,           # can release memory back to trainer
            )
    
        def generate(self, prompts, sampling_params):
            return self.engine.generate(prompts, sampling_params)
    
        def sleep(self):
            # Release GPU memory; trainer can use it
            self.engine.sleep()
    
        def wake_up(self, new_weights):
            # Reload weights from latest training step, resume rollout
            self.engine.wake_up()
            self.engine.load_weights(new_weights)
    
    @ray.remote(num_gpus=8)
    class TrainerWorker:
        def __init__(self, model_path):
            import deepspeed
            # ZeRO-3 sharded training
            self.engine = init_zero3_engine(model_path)
    
        def train_step(self, rollouts, advantages, kl_terms):
            loss = ppo_loss(self.engine, rollouts, advantages, kl_terms)
            self.engine.backward(loss)
            self.engine.step()
            return self.engine.get_weights()      # pass back to Rollout
    
    # Top-level coordinator
    def rlhf_step(rollout_actor, trainer_actor, ref_actor, reward_actor, prompts):
        # 1. Generate rollouts
        rollouts = ray.get(rollout_actor.generate.remote(prompts, sampling_params))
    
        # 2. Score with reward and reference (in parallel)
        reward_future = reward_actor.score.remote(rollouts)
        ref_future    = ref_actor.log_probs.remote(rollouts)
        rewards, ref_lps = ray.get([reward_future, ref_future])
    
        # 3. Compute advantages, KL, etc.
        advantages = compute_advantages(rewards, rollouts.log_probs)
        kl_terms = rollouts.log_probs - ref_lps
    
        # 4. Hybrid Engine: rollout sleeps, trainer takes over the GPUs
        ray.get(rollout_actor.sleep.remote())
    
        # 5. Train Actor
        new_weights = ray.get(trainer_actor.train_step.remote(rollouts, advantages, kl_terms))
    
        # 6. Reload Actor weights into rollout, resume
        ray.get(rollout_actor.wake_up.remote(new_weights))

This is the skeleton. Real OpenRLHF/veRL has more sophistication — async pipelining, fault tolerance, gradient accumulation, multi-step rollout, etc. — but the structure is recognizable. _The Hybrid Engine pattern (steps 4-6) is the key innovation_ : the same GPUs serve both rollout and training, swapping memory between modes via vLLM's sleep/wake.

## OpenRLHF vs veRL: two architectural philosophies

The two dominant frameworks in 2026 differ in a specific architectural decision: _how to organize the four model workers across resources_.

### OpenRLHF: dedicated workers per model

Each model (Actor, Reference, Reward, Critic) has its own Ray worker group, each colocated by rank. If you allocate 32 GPUs total, OpenRLHF might give you 8 GPUs each for Actor, Reference, Reward, Critic — separately scheduled, separately tensor-parallelized.

Pros: clean separation; failure of one model's worker doesn't crash others; per-model resource tuning (Reference can use less compute than Actor).

Cons: GPUs allocated to (e.g.) Reference are idle when Reference isn't being called. The Hybrid Engine variant (--colocate_all_models) addresses this by sharing GPUs across all model workers and using vLLM sleep/wake to swap.

OpenRLHF's main features as of v0.10 (April 2026):

  * Distributed Ray architecture, ZeRO-3 + vLLM combination
  * Hybrid Engine support via `--colocate_all_models` flag
  * Algorithms: PPO, GRPO, REINFORCE++, REINFORCE++ baseline, RLOO, DPO, online RLHF
  * VLM RLHF (April 2026 release): train Qwen3.5-VL with image inputs end-to-end
  * Multi-turn VLM RL: agent loops with screenshots in environment feedback
  * QLoRA/LoRA, RingAttention, FlashAttention, packing samples, MoE support

### veRL: unified WorkerDict

veRL takes the opposite approach: _a single WorkerDict that houses all models_ , sharing resources naturally. Instead of "Actor's 8 GPUs, Reference's 8 GPUs," veRL has "32 GPUs, every WorkerDict instance has Actor + Reference + Reward weights co-resident."

Pros: maximizes GPU utilization (no idle GPUs holding only Reference weights); cleaner model swapping; simpler scheduling.

Cons: more memory pressure per GPU (must fit weights for all models); failure modes are coupled.

veRL's distinctive features:

  * Single-controller pattern inspired by Google's Pathways
  * Megatron-LM integration for very large model training (671B-class)
  * HybridFlow design: separate computation graphs for each algorithm phase, dispatch based on phase
  * Strong support for MoE training in RLHF settings

### Performance: published comparisons

OpenRLHF's published comparison (v0.8.5 vs veRL v0.4.0, mid-2025) on 1.5B/7B/14B models with 1K/2K/4K/8K generation lengths showed OpenRLHF faster on average across the matrix, particularly at longer generation lengths. veRL's strength is at very large scale (70B+ MoE) where its unified scheduling pays off more. _For most production workloads (1.5B-70B dense), OpenRLHF tends to be faster; veRL's advantages grow with model size and architectural complexity._

The choice in 2026 isn't categorical — both frameworks are actively developed and converge feature-wise. Default to OpenRLHF for ease-of-use and broad algorithm support; choose veRL when you need Megatron integration or are training at the 100B+ scale.

## Hybrid Engine: the production default

The single most important infrastructure idea in distributed RLHF as of 2026: **Hybrid Engine**. Both OpenRLHF and veRL support it; it's the production default for memory efficiency.

The problem it solves: in a four-model RLHF setup, each model needs GPUs. Naive allocation gives each model dedicated GPUs that sit idle when that model isn't being called. With four 70B models and dedicated allocation, you might use 32 GPUs, of which only 8 are doing work at any moment.

Hybrid Engine: _colocate all four models on the same GPUs_. Use vLLM's sleep/wake mechanism to swap GPU memory between modes. When the Actor is doing rollout, Actor's vLLM engine is awake; Reference, Reward, and Critic are sleeping (weights paged to CPU or compressed). When the trainer is doing the gradient step, vLLM sleeps; the trainer's ZeRO engine wakes.

Hybrid Engine vs Disaggregated: two ways to schedule the four-model dance Hybrid Engine (production default) Same GPUs serve all roles, swap via sleep/wake Shared GPU pool (e.g., 32× H100) Phase A: rollout vLLM (Actor) awake trainer asleep Phase B: training trainer awake vLLM asleep Sleep/wake roundtrip: ~1-3 seconds Weight transfer: NCCL or CUDA IPC Pros: max GPU utilization, simple cluster Cons: serial; no rollout-train overlap Disaggregated (StreamRL, AReal) Separate clusters; async pipeline Rollout cluster B200/H200, FP4/FP8 Training cluster H100, NVLink-rich Async pipeline step k rollout step k train step k+1 rollout step k+1 train Rollout(k+1) overlaps with train(k) Off-policy correction needed Pros: rollout-train overlap; specialized HW Cons: complex; off-policy bias

The mechanism: **vLLM's sleep mode**. Released in 2024, refined through 2025-2026, this is what makes Hybrid Engine work. When called, vLLM unloads model weights from GPU memory (paging them to CPU, or releasing entirely if checkpointed elsewhere) and frees its KV cache. The GPU memory becomes available to whatever else needs it (the trainer). When wake_up is called, vLLM reloads weights and rebuilds KV cache. Roundtrip cost: 1-3 seconds for a 70B model on 8× H100.

The trainer-to-rollout weight transfer also matters. After a training step, the new Actor weights need to reach the rollout engine. Three patterns:

  * **NCCL broadcast** : trainer ranks broadcast updated weights to rollout ranks. Fast but requires both to be alive and in the same NCCL communicator.
  * **CUDA IPC** : shared CUDA memory between processes. Zero-copy if same node; fastest option.
  * **Disk + reload** : trainer saves checkpoint, rollout reloads. Slowest but simplest; used in some early frameworks.

OpenRLHF defaults to NCCL broadcast; veRL uses CUDA IPC where possible. The transfer takes 5-30 seconds for 70B-class models depending on pattern.

## Disaggregated: when rollout dominates

Hybrid Engine's weakness: it's serial. Rollout phase runs, then training phase runs; they can't overlap because they share GPU memory. If rollout takes 80% of step time and training takes 20%, the trainer GPUs sit idle for 80% of the cycle.

**Disaggregated architectures** (StreamRL from Alibaba, AReal from Anthropic-influenced teams) split rollout and training onto separate clusters with async pipelining. While step k+1 rollouts are generating, step k gradients are being computed. Both clusters are busy.

The catch: _the rollouts used for step k's gradient update are slightly off-policy_ (they were generated with the Actor's weights from before step k's update). Algorithmic corrections (importance sampling, KL clipping) handle the mismatch but introduce some bias. For large rollout-to-training ratios this is acceptable; for small ratios the bias can dominate.

The disaggregated pattern wins specifically when:

  * **Rollout dominates training compute** (long generation, large group sizes for GRPO)
  * **Heterogeneous hardware available** — B200/B300 for rollout (FP4 inference acceleration), H100 for training (NVLink-rich, FP8 training)
  * **Off-policy bias is tolerable** for the algorithm

For most production workloads, Hybrid Engine on a homogeneous cluster is simpler and sufficient. Disaggregation is the choice for very-large-scale or heterogeneous-hardware setups.

## Algorithm choice drives infrastructure

RLHF algorithms and their infrastructure footprint, April 2026 Algorithm| Models needed| Rollout cost| Notes  
---|---|---|---  
PPO (classical)| 4 (Actor, Ref, Reward, Critic)| 1× per prompt| Largest infra; original RLHF recipe  
GRPO (DeepSeek)| 3 (Actor, Ref, Reward)| K× per prompt (group)| No Critic; high rollout cost; reasoning models  
RLOO| 3 (Actor, Ref, Reward)| K× per prompt| Leave-one-out variant of GRPO  
**REINFORCE++ baseline**|  3 (Actor, Ref, Reward)| 1× per prompt| **Frontier 2026** ; Magistral, ProRL V2, ScaleRL  
DPO / IPO| 2 (Actor, Ref)| 0× (offline)| No RL loop; pure preference optimization  
Iterative DPO| 2 + judge| partial (sampling for new prefs)| Online preference loop  
Online RLHF| varies| varies| Continuous preference collection  
  
The trajectory through 2024-2026 is unmistakable: _algorithms are getting simpler infrastructure-wise_. PPO had four models; GRPO has three with K× rollout; REINFORCE++ has three with 1× rollout. Each step removes complexity. The quality results have followed — REINFORCE++ at ProRL V2 scale (Feb 2026) and ScaleRL scale (Oct 2026) match or exceed PPO/GRPO with substantially less infrastructure overhead. The **"PPO is the gold standard" era ended** ; the field is consolidating around REINFORCE++ baseline for new training runs.

## Distributed SFT: the prerequisite

Before any RLHF, there's SFT (supervised fine-tuning). M11 covered training loop fundamentals; SFT adds specific concerns at the distributed scale:

  * **Single-model simplicity**. SFT is just standard training on (prompt, response) pairs. One model, one optimizer, no four-model coordination. Vastly simpler than RL.
  * **Sample packing** : pack short sequences to fill the model's context window. With 4K-context model and average 500-token sequences, 8 sequences pack into one example. Reduces wasted compute on padding by ~85%. Essentially required for cost-effective SFT in 2026.
  * **Sequence parallelism** : for long-context SFT (32K+ context), the activation memory of a single sequence exceeds single-GPU memory. Sequence parallelism (RingAttention, M16-adjacent) splits a single long sequence across GPUs. Standard for reasoning-model SFT where chain-of-thought outputs are long.
  * **FSDP or ZeRO-3** : weight sharding so the model fits across the cluster. Same machinery as M17.
  * **Cold-start SFT for RL** : the standard recipe is "SFT first to give the model basic capabilities, then RL to optimize." SFT phase typically uses 10K-1M high-quality examples; RL phase uses preference data + rollouts.

SFT data quality matters more than quantity for the cold-start phase. M35's data engineering applies: dedupe, filter for quality, ensure no contamination with downstream evals. The SFT phase's output is the Actor and Reference initialization for the subsequent RL phase.

## What goes wrong: failure modes

Distributed RLHF has characteristic failure modes that infrastructure choices interact with:

  * **Entropy collapse** : the policy becomes deterministic, exploring only a narrow distribution of outputs. Symptoms: rewards plateau, KL goes to zero, generation diversity drops. _Mitigation_ : Entropulse (alternating SFT and RL phases, used in ComputerRL); entropy regularization in the loss; early stopping based on entropy floor.
  * **Reward hacking** : the policy finds rollouts that score high on the Reward model but aren't actually good. Symptoms: reward goes up while human judgment of quality goes down. _Mitigation_ : better reward models (often LLM-as-judge with rubric, M37); shorter training; KL penalty to keep policy near Reference.
  * **KL blowup** : KL(Actor ‖ Reference) grows without bound; the policy drifts into incoherent territory. _Mitigation_ : tune KL coefficient; clip KL term; use adaptive KL scheduling.
  * **Length bias** : policy learns to generate longer outputs because the reward model favors them. Symptoms: average generation length grows without quality improving. _Mitigation_ : length penalty in reward; train reward model with length-controlled preferences.
  * **Rollout-train mismatch** : in disaggregated setups, the rollouts are too off-policy and PPO's importance ratio explodes. _Mitigation_ : clip importance ratios more aggressively; reduce rollout-train staleness; switch to Hybrid Engine.
  * **Hybrid Engine memory thrash** : sleep/wake transitions take longer than expected because of fragmentation. _Mitigation_ : pre-warm vLLM caches; use CUDA IPC for weight transfer; reduce vLLM gpu_memory_utilization to leave headroom.

Most of these have _characteristic signatures in metrics_ — you can detect them by watching the right curves. Production RL infrastructure includes dashboards for: per-step rollout time, training step time, KL divergence, reward distribution (mean and tail), entropy, average generation length, GPU utilization per actor. Any of these going off the expected curve is a debugging signal.

## Production reality April 2026

Putting the pieces together — what's actually being run in production at the frontier as of April 2026:

  * **Magistral (June 2025, Mistral)** : REINFORCE++ baseline for reasoning model training. Confirmed the algorithm at frontier scale.
  * **DeepSeek R1 series** : GRPO with rule-based rewards (math/code verification); large rollout groups; no learned reward model for the base R1-Zero variant. The reasoning-model baseline.
  * **ProRL V2 (Feb 2026)** : REINFORCE++ baseline trained a state-of-the-art 1.5B reasoning model with prolonged RL training (700+ stable RL steps). Demonstrated REINFORCE++ scales to long horizons.
  * **ScaleRL (Oct 2026, projected)** : REINFORCE++ baseline at large-scale validation. The trajectory from "interesting algorithm" to "frontier production default."
  * **OpenRLHF v0.10 (April 2026)** : VLM RLHF support added. Multi-turn VLM RL with image-in-prompt and image-in-environment-feedback (e.g., screenshots in computer-use agent training).
  * **OpenRLHF-M** : dedicated multimodal fork; trains Qwen3.5-VL-class models with image inputs end-to-end.
  * **Hybrid Engine on B200/B300** : emerging pattern. Rollout uses NVFP4 (4× memory of FP8, near-FP8 quality); training uses FP8 or BF16. Single-cluster Hybrid Engine becomes more attractive as B200's per-GPU memory (192GB) accommodates both modes more comfortably.
  * **RL Environments as a service** : Scale RL Environments, Tinker, Fireworks AI all sell environments. Anthropic estimated to spend tens of millions/year. Coding and computer-use are the proving grounds.

The frontier algorithm-infrastructure stack as of April 2026: **Cold-start SFT (with packing, sequence parallelism, FSDP) → Reward Model training (or rule-based / LLM-judge rewards) → REINFORCE++ baseline RL with Hybrid Engine, Ray orchestration, vLLM rollout, ZeRO-3 training**. OpenRLHF or veRL as the framework layer. This pattern composes everything from M11 (training loop) through M17 (FSDP) through M27 (vLLM) through M34 (GRPO algorithm). M39 is what ties them into a working RLHF system.

#### Q&A; — About distributed RLHF/GRPO infrastructure **Q:** Why does rollout take 80% of compute? Isn't generation cheap relative to training? **A:** Per-token, generation is cheaper than training. But RLHF requires generating _thousands of tokens per training example_ — the rollout has to complete a full response (or reasoning trace, often 2K-10K tokens) before the gradient step happens. Training, by contrast, processes that completed sequence in a single forward+backward pass. So even though training is FLOP-heavier per token, rollout dominates because it's many tokens. The ratio gets worse for reasoning models (longer generations, more tokens per example) and for GRPO (K× rollouts per prompt). For a 4K-token rollout with K=8 group size, you generate 32K tokens per training example; the train step processes ~4K tokens (one trajectory) plus advantages. Rollout-to-train ratio is 8:1 in token count, but generation tokens are cheaper per-FLOP, so the wall-clock ratio lands around 4:1 or 80/20. **Q:** Why eliminate the Critic? Wasn't variance reduction its whole point? **A:** Two practical reasons. (1) **Critic adds infrastructure overhead** — another model in memory, another optimizer state, another set of gradients to coordinate. For a 70B Actor with full Adam optimizer, each saved model is ~280GB of state. (2) **Critic adds noise of its own** — the value function estimator is itself learned, with its own training dynamics, and bad value estimates can hurt more than they help. GRPO's group-normalization and REINFORCE++'s moving-average baseline are both simpler variance-reduction schemes that work empirically as well or better. The Critic's theoretical role (per-state baseline that minimizes variance) was always a relatively weak argument; in practice, _simpler baselines are competitive and have lower infrastructure overhead_. The 2025-2026 results consistently show this. **Q:** When should I pick disaggregated (StreamRL) over Hybrid Engine? **A:** When two conditions hold: (a) **rollout strongly dominates training time** (>80% of step time) — this happens with very long generation, large GRPO group size, or computationally cheap training (LoRA, small model); (b) **you have heterogeneous hardware available** — B200/B300 cluster for rollout (FP4 acceleration), H100 cluster for training. If both hold, disaggregated lets you fully utilize both clusters by running rollout(k+1) in parallel with train(k). If either doesn't hold, Hybrid Engine is simpler and roughly as fast. _For most production teams in 2026 on homogeneous H100/H200 clusters, Hybrid Engine is the default._ Disaggregated is for frontier labs with mixed Blackwell/Hopper fleets and very large rollout ratios. **Q:** Is the move from PPO to GRPO to REINFORCE++ a one-way trip, or might PPO come back? **A:** Probably not coming back, but the field isn't fully settled. PPO's value is variance reduction via the Critic; if someone shows that very high-quality value estimation matters at extremely large scale (trillion-parameter models with long-horizon rollouts), the Critic's overhead might re-justify itself. As of April 2026, no public results suggest this — REINFORCE++ baseline with moving-average variance reduction matches or beats PPO at every published scale. _The most likely future: REINFORCE++ baseline becomes the standard for general RLHF; specialized variants (e.g., DPO for offline preference learning, GRPO for reasoning models with verifier rewards) keep their niches_. PPO becomes a "historical baseline" — taught for completeness, not used for new training. **Q:** How does this all change for MoE models? **A:** MoE adds two complications. (1) **Routing instability during RL** : the policy update changes which experts are active for which inputs; if routing changes too quickly, the experts don't get consistent gradient signals and quality degrades. Mitigation: lower learning rate; add routing-stability loss term; freeze routing for the first N RL steps. (2) **All-to-all communication in rollout** : MoE inference requires expert all-to-all every layer, which interacts with vLLM's batching. The vLLM MoE path is mature in 2026 but not universal across frameworks. veRL has stronger MoE support (Megatron-derived all-to-all); OpenRLHF added MoE support via aux_loss_coef and expert-parallelism flags. _For MoE RLHF in 2026, default to veRL unless you have specific OpenRLHF features you need_ ; expect to spend extra engineering on routing stability. **Q:** For a small team training a 7B reasoning model with limited GPUs, what's the actual recipe? **A:** Concrete recipe for April 2026: (1) Use **OpenRLHF** as the framework. (2) **Cold-start SFT** on 10K-100K reasoning traces (M35-quality data). 8× H100, FSDP/ZeRO-3, packing samples, 1-3 epochs. (3) Skip the learned reward model if you can — use **rule-based rewards** (math/code verification) for tasks with verifiable answers; this avoids reward hacking and saves a model. (4) **REINFORCE++ baseline** as the algorithm. (5) **Hybrid Engine** : `--colocate_all_models --vllm_enable_sleep --vllm_gpu_memory_utilization 0.5`. 8× H100 fits 7B Actor + Reference + (small or rule-based) Reward easily. (6) Run for 500-2000 RL steps. (7) Monitor entropy, reward, KL; intervene if entropy collapses. _Total compute: roughly 1-2 weeks on 8× H100_ ; cost: $5-15K. This is genuinely tractable for a small team in 2026 — the infrastructure has matured enough that the bottleneck is usually data and reward design, not compute or framework engineering. **Q:** How does this compose with multimodal (M38)? **A:** It composes via OpenRLHF-M (or OpenRLHF v0.10's VLM support, April 2026). The pattern: **Actor and Reference are VLMs** (vision encoder + LLM trunk + connector); rollout includes image-conditioned generation; reward model can score image-text pairs (or be rule-based for verifiable tasks). The infrastructure changes are specifically: (1) vLLM's VLM path is required for fast multimodal rollout; (2) memory pressure increases (vision tokens are added to context); (3) for multi-turn VLM RL (computer-use agents seeing screenshots), the environment provides images in feedback. _Computer-use agent training is the dominant production VLM RL workload in 2026_ ; ComputerRL achieved 48.9% on OSWorld using this stack. Coming attraction for M40. 

## Code Magnets: implement a Hybrid Engine RLHF step

You're writing the per-step coordinator for a Hybrid Engine RLHF pipeline. Three magnets are wrong choices.

Arrange the magnets to compute one RLHF step.

def rlhf_step(rollout, trainer, ref, reward, prompts): rollouts = ray.get(rollout.generate.remote(prompts)) rollouts = rollout.generate(prompts) rewards, ref_lps = ray.get([reward.score.remote(rollouts), ref.log_probs.remote(rollouts)]) rewards = ray.get(reward.score.remote(rollouts)); ref_lps = ray.get(ref.log_probs.remote(rollouts)) advantages = compute_advantages(rewards, rollouts.log_probs) ray.get(rollout.sleep.remote()) new_weights = ray.get(trainer.train_step.remote(rollouts, advantages)) ray.get(rollout.wake_up.remote(new_weights)) new_weights = trainer.train_step(rollouts, advantages)

show solution
    
    
    def rlhf_step(rollout, trainer, ref, reward, prompts):
        rollouts = ray.get(rollout.generate.remote(prompts))
        rewards, ref_lps = ray.get([reward.score.remote(rollouts), ref.log_probs.remote(rollouts)])
        advantages = compute_advantages(rewards, rollouts.log_probs)
        ray.get(rollout.sleep.remote())
        new_weights = ray.get(trainer.train_step.remote(rollouts, advantages))
        ray.get(rollout.wake_up.remote(new_weights))

The traps:

  * `rollouts = rollout.generate(prompts)`: calls the actor's method directly instead of going through Ray's `.remote()` dispatch. This is wrong on multiple levels: (a) Ray actors don't have synchronous methods accessible from outside the actor — calling `rollout.generate(prompts)` would either fail or call a method on the local proxy without actually running it on the worker; (b) Ray's whole point is process-isolated, GPU-pinned execution — direct calls bypass the orchestration layer; (c) the worker's GPU resources don't even exist in the calling process, so the call would fail with missing CUDA context. Always go through `actor.method.remote(args)` \+ `ray.get(future)` for Ray actors.
  * `rewards = ray.get(reward.score.remote(rollouts)); ref_lps = ray.get(ref.log_probs.remote(rollouts))`: serializes two calls that should be parallel. `ray.get` blocks until the future resolves; calling it twice in sequence means waiting for reward scoring before starting reference log-prob computation, even though both are independent and can run on separate GPUs simultaneously. The correct pattern: launch both `.remote` calls (which return immediately), then `ray.get` the list of futures (which waits for all in parallel). For 70B Reference + Reward, this can save several seconds per step — adds up over thousands of steps.
  * `new_weights = trainer.train_step(rollouts, advantages)` (without `ray.get` \+ `.remote`): same issue as the first trap — direct call to a Ray actor's method. The trainer is on different GPUs in a different process; you have to dispatch via Ray. Additionally, this version skips the `rollout.sleep()` \+ `rollout.wake_up()` Hybrid Engine dance: without sleep/wake, the trainer can't allocate GPU memory because vLLM is still holding it. Both bugs compound to a non-functional pipeline.

The pattern: **generate via Ray.remote → score with reward and reference in parallel via Ray.remote → compute advantages locally → sleep rollout → train via Ray.remote → wake rollout with new weights**. The `.remote()`/`ray.get()` dispatch is mandatory for cross-actor communication; Hybrid Engine's sleep/wake is mandatory for memory sharing; parallelizing reward and reference scoring is the optimization that compounds across thousands of steps.

## Who does what?

Match each distributed RLHF concept to its real role.

Concept

Real role

Actor

A. The policy under training; generates rollouts and receives gradient updates.

Reference

B. Frozen SFT-init copy used to compute KL divergence; prevents policy drift.

Critic

C. Trained value head used by PPO; eliminated by GRPO and REINFORCE++.

Hybrid Engine

D. Colocate all models on shared GPUs; vLLM sleep/wake to swap rollout/training modes.

Disaggregated (StreamRL)

E. Separate rollout and training clusters; async pipeline; off-policy correction needed.

REINFORCE++ baseline

F. Frontier 2026 algorithm; 3 models, 1× rollout, moving-average variance reduction.

Ray

G. Actor-framework orchestration layer; manages model workers, scheduling, fault tolerance.

show solution

**Actor** → A  
**Reference** → B  
**Critic** → C  
**Hybrid Engine** → D  
**Disaggregated** → E  
**REINFORCE++ baseline** → F  
**Ray** → G 

The mental shortcut: _Actor trains, Reference baselines KL, Critic estimated value (now retired), Hybrid Engine swaps modes on shared GPUs, Disaggregated splits clusters async, REINFORCE++ is the new frontier default, Ray orchestrates everything_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team is training a 70B reasoning model with GRPO on 32× H100 (80GB each). They report rollouts take 12 seconds, training takes 3 seconds, but each "step" takes 25 seconds total. What's eating the missing 10 seconds, and how would you fix it?

show answer

The missing 10 seconds is almost certainly **Hybrid Engine swap overhead and weight transfer**. The 12 + 3 = 15 seconds is just the active compute; the other 10 seconds is everything that happens between phases:

(1) **Rollout sleep** (~1-3s): vLLM unloads weights, frees KV cache, releases GPU memory. (2) **Trainer wake** (~1-2s): ZeRO engine acquires GPU memory, materializes gradients/optimizer states. (3) **Trainer sleep after step** (~1-2s): release GPU. (4) **Rollout wake** (~2-3s): vLLM reloads weights, rebuilds KV cache. (5) **Weight transfer** (~3-5s for 70B): the new Actor weights from trainer to rollout via NCCL or CUDA IPC.

Mitigations to try, in priority order:

(a) **CUDA IPC for weight transfer** instead of NCCL broadcast — typically 2-3× faster within a node. veRL defaults to this; OpenRLHF needs configuration.

(b) **Reduce vllm_gpu_memory_utilization** to leave more headroom — paradoxically, less aggressive memory packing can make sleep/wake faster because there's less fragmentation to recover from.

(c) **Pre-compile vLLM CUDA graphs** at the first wake; subsequent wakes reuse the compilation.

(d) **Switch to disaggregated** (StreamRL/AReal pattern) if the rollout/train ratio is favorable. With 12s rollout and 3s training, ratio is 4:1 — borderline. Disaggregated would let train(k) overlap with rollout(k+1), potentially halving step time at the cost of off-policy bias.

(e) **Verify the algorithm choice** : GRPO with K=8 group size means each "training example" required 8 rollouts. If switching to REINFORCE++ baseline (1× rollout per example), rollout time drops from 12s to ~1.5s, and the swap overhead becomes a much smaller fraction of step time. _This is often the biggest single win — algorithm choice dominates infrastructure choice for total throughput_.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why does REINFORCE++ baseline scale better than GRPO in production, even though GRPO's variance reduction (group-normalization) seems "more principled"?

show answer

Three compounding reasons:

(1) **Rollout cost is K× lower**. GRPO requires K rollouts per prompt to compute the group statistics (typical K=4-16). REINFORCE++ baseline uses 1 rollout per prompt with a moving-average baseline computed across the batch. For the same training-example count, GRPO does 4-16× more generation. Since rollout is 80% of compute, REINFORCE++ is 3-15× faster wall-clock per training example.

(2) **Variance reduction quality is comparable empirically**. The "principled" argument for group-normalization (reduces variance per group of K rollouts) is real in expectation but the actual variance reduction depends on K and on how similar the K rollouts are. With small K (K=4), the group statistics are themselves noisy; with large K, you pay K× rollout cost. Moving-average baseline (REINFORCE++) achieves similar variance reduction by averaging across thousands of past rollouts — strictly more samples than any group can provide. Empirically, this is comparable or better at scale.

(3) **Stability properties are better**. Logic-RL and PRIME (Feb 2025) showed REINFORCE++ is more stable than GRPO under prolonged training. ProRL V2 (Feb 2026) trained 700+ stable steps on a 1.5B reasoning model with REINFORCE++ baseline, while GRPO often shows entropy collapse or KL blowup at similar horizons. _Stability matters more than peak quality at scale_ — a stable algorithm that runs for 1000 steps beats a higher-peak algorithm that diverges at step 300.

The combined argument: lower compute, comparable variance reduction, better stability. The "more principled" framing mistook a theoretical property for a practical advantage. _2026's lesson: empirical scaling beats theoretical elegance for RLHF algorithm choice_.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** Walk through the failure mode "entropy collapse" — what causes it, what it looks like in metrics, and what infrastructure choices interact with it.

show answer

**What causes it** : the policy becomes too confident, sampling only a narrow distribution of outputs. This happens because (a) the reward signal is consistently rewarding a particular output style, (b) the KL penalty isn't strong enough to keep the policy near Reference, (c) training has run too long without re-injecting diversity. The policy effectively memorizes "high-reward outputs" and stops exploring.

**What it looks like in metrics** : _entropy of policy outputs drops sharply_ (the per-token entropy averaged across rollouts goes from ~3-5 nats to ~0.5-1 nats). _KL(Actor ‖ Reference) plateaus or even decreases_ (because the policy becomes deterministic, the KL term saturates). _Reward plateaus or decreases_ (the policy is exploiting a local optimum that doesn't generalize). _Generation diversity drops_ (samples become repetitive, formulaic).

**Infrastructure interactions** :

(a) **Hybrid Engine vs Disaggregated** : Hybrid Engine has stricter on-policy semantics (rollouts and training are synchronous), making collapse more visible step-to-step. Disaggregated has rollout-train staleness which can _mask_ early collapse signs but doesn't prevent collapse.

(b) **Algorithm choice** : GRPO with rule-based rewards and small K can collapse faster than REINFORCE++ because group-normalization removes information about absolute reward levels — the policy can game the relative ranking even when absolute reward stagnates. REINFORCE++ baseline preserves more reward signal.

(c) **Rollout sampling parameters** : temperature=0.7 vs 1.0 makes a big difference. Lower temperature produces more deterministic outputs, accelerating collapse. Production runs use temperature 0.7-1.0 for rollout, sometimes higher early in training.

(d) **Mitigations as infrastructure features** : **Entropulse** (alternating SFT and RL, used in ComputerRL) is a training-loop pattern that requires infrastructure support — periodically pause RL, run a few SFT steps on diverse data, resume RL. Frameworks need to support this. Entropy bonus in the loss is simpler — add a small coefficient × negative entropy to the objective. Most frameworks support this; tune the coefficient.

The general lesson: **monitoring entropy is mandatory**. If your dashboard doesn't have a per-step entropy plot, you'll discover collapse only after the model is already broken. Production RL infrastructure includes entropy alerts.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** A team wants to do "Distributed SFT then GRPO" for a 30B reasoning model. Sketch the infrastructure stack they need at each phase.

show answer

Two phases with substantially different infrastructure profiles.

**Phase 1: SFT**. Single 30B model, ~60GB in bf16, plus optimizer states (Adam: ~120GB) plus gradients (~60GB) ≈ 240GB total state. Fits in 8× H100 with ZeRO-2 or smaller cluster with ZeRO-3. Stack:

  * **Framework** : Hugging Face `transformers` \+ `accelerate` with DeepSpeed ZeRO-3, or `torchtitan`, or `lit-gpt`. Standard options.
  * **Parallelism** : ZeRO-3 (or FSDP) for memory; data parallelism across nodes; sequence parallelism (RingAttention) if context exceeds 16K.
  * **Data** : 10K-100K reasoning traces; sample-packed to fill context efficiently. Quality > quantity per M35.
  * **Compute** : 8-32 H100s, 1-3 days for 30B with packed 4K context.
  * **Output** : SFT checkpoint that becomes both the Actor initialization AND the Reference for the RL phase.

**Phase 2: GRPO**. Three models needed (Actor, Reference, Reward). For 30B at 60GB each: 180GB just for weights. Add Actor optimizer states (120GB) and gradients (60GB), plus rollout KV cache. Stack:

  * **Framework** : OpenRLHF or veRL with Hybrid Engine. OpenRLHF for ease of use; veRL for stronger MoE support if applicable.
  * **Configuration** : `--colocate_all_models --vllm_enable_sleep --vllm_gpu_memory_utilization 0.5` on 16-32 H100s.
  * **Algorithm** : GRPO with K=8 group size, OR (recommended in 2026) REINFORCE++ baseline for substantially lower rollout cost.
  * **Reward** : rule-based if tasks have verifiable answers (math, code), or trained 7B reward model, or LLM-as-judge.
  * **Data** : 10K-100K prompts (no need for canonical answers — just the prompt; the policy generates responses).
  * **Compute** : 16-32 H100s for 1-2 weeks for 500-2000 RL steps.
  * **Monitoring** : per-step entropy, reward distribution (mean + tail), KL divergence, average generation length, GPU utilization. Alerts on entropy floor.

**Practical advice** : don't skip the cold-start SFT — RL from scratch (R1-Zero style) works for some tasks but is much harder. SFT gives the model a good starting policy that RL can refine. Total budget: ~3-4 weeks of engineering + ~$50-100K of compute on 16-32 H100s for 30B. _Tractable for a well-resourced team in 2026; unthinkable two years ago._

### What just happened?

  * RLHF/GRPO is fundamentally a **systems problem disguised as an ML problem**. ~80% of compute goes to rollout (sample generation); coordinating four models concurrently is harder than coordinating any single training run.
  * **Four-model anatomy of PPO** : Actor (trained, generates rollouts), Reference (frozen, KL baseline), Reward (frozen, scores rollouts), Critic (trained, value head). For 70B-class: ~560GB just for weights, ~1.2-1.3TB total state with optimizer + gradients + KV cache.
  * **Algorithm choice drives infrastructure**. PPO needs 4 models. GRPO eliminates the Critic via group-normalization — 3 models, but K× rollout cost. **REINFORCE++ baseline** (the 2025-2026 frontier) eliminates both Critic and group-rollout overhead — 3 models, 1× rollout, moving-average variance reduction.
  * Frontier 2026 production using REINFORCE++ baseline: **Magistral** (Mistral, June 2025), **ProRL V2** (Feb 2026, SOTA 1.5B reasoning), **ScaleRL** (Oct 2026, large-scale validation).
  * **Ray** is the orchestration layer for all major frameworks. Each model is a Ray actor; Ray handles scheduling, message passing, fault tolerance.
  * **OpenRLHF vs veRL** : dedicated worker groups per model (OpenRLHF) vs unified WorkerDict housing all models (veRL). OpenRLHF is faster on average for 1.5B-70B dense; veRL has stronger MoE and very-large-scale support. Both support Hybrid Engine.
  * **Hybrid Engine** : production default. Colocate models on shared GPUs; use vLLM sleep/wake to swap between rollout (vLLM awake) and training (ZeRO awake) modes. Roundtrip: 1-3 seconds for 70B.
  * **Weight transfer trainer→rollout** : NCCL broadcast (OpenRLHF default), CUDA IPC (veRL default, faster), or disk reload (slow, simple).
  * **Disaggregated (StreamRL, AReal)** : separate rollout and training clusters; async pipeline; rollout(k+1) overlaps with train(k). Better when rollout strongly dominates and heterogeneous hardware (Blackwell rollout + Hopper training) is available. Off-policy bias requires algorithmic correction.
  * **Distributed SFT** is the prerequisite. Single model, standard training loop. Specific concerns: sample packing (8× efficiency for short sequences), sequence parallelism for long contexts, FSDP/ZeRO-3 for memory. Cold-start SFT before RL is the standard recipe.
  * **Failure modes** : entropy collapse (Entropulse alternating), reward hacking (better reward / shorter training), KL blowup (clip / adaptive scheduling), length bias (length-controlled rewards), Hybrid Engine memory thrash. Monitoring entropy, reward distribution, KL, and average length is mandatory.
  * **OpenRLHF v0.10 (April 2026)** : VLM RLHF support; multi-turn VLM RL with screenshots in environment feedback. Composes with M38's multimodal architectures.
  * The reflex: when designing an RLHF/GRPO training run, ask "which algorithm? what's the rollout-to-train ratio? Hybrid Engine or disaggregated? Rule-based reward or learned?" These four choices determine 80% of the infrastructure complexity. Get them right and the rest follows.

Module 40 (next) will cover **agentic RL and RL Environments** — Environment-as-a-Service, ComputerRL/OSWorld at 48.9%, RLinf for embodied AI, the verifier engineering problem, online vs local environments. The dominant production RL paradigm in 2026 builds directly on M39's distributed infrastructure.
