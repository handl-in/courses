#!/usr/bin/env python3
"""Module 39: Distributed RLHF & GRPO infrastructure — full HF vibe, April 2026 current."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part XI · Module 39 · April 2026 currency</div>
  <h1 class="module-title"><em>Distributed RLHF &amp; GRPO infrastructure:</em> Ray, Hybrid Engine, and the four-models-in-memory problem</h1>
  <p class="module-sub">— why RLHF is 80% rollout, how OpenRLHF and veRL diverge architecturally, what Hybrid Engine actually does, and why REINFORCE++ became the frontier algorithm in 2025-2026</p>
</div>

<p>M31 covered the post-training algorithms — PPO, DPO, reward model training. M34 covered the GRPO algorithm and the reasoning-model recipe. Both treated the algorithms in isolation. <em>This module is about the infrastructure</em>: what it actually takes to run RLHF or GRPO at scale, what the dominant frameworks (OpenRLHF, veRL) do differently, and why "Hybrid Engine" became the production default. The systems engineering of distributed RL is now its own discipline; nothing in M1-M38 covered it directly.</p>

<p>The reality: <strong>RLHF spends ~80% of compute on rollout, not on the gradient update.</strong> The training step is the small part; generating trajectories from the policy under training (with the rollout engine) and scoring them (with reward and reference models) is what dominates. A 70B Actor + 70B Reference + Reward + Critic configuration consumes 8-16 H100s just for model weights before optimizer states. Coordinating these four models — some training, some frozen, all communicating — is harder than coordinating any single-model training run you've seen. By April 2026, the public frameworks have settled into recognizable patterns; the recipes are teachable. This module is what you need before training your own reasoning model.</p>

<div class="keyidea">
RLHF/GRPO is a <strong>systems problem disguised as an ML problem</strong>. PPO requires four models running concurrently: <strong>Actor</strong> (policy being trained), <strong>Reference</strong> (frozen baseline for KL), <strong>Reward</strong> (frozen scorer), <strong>Critic</strong> (value head, PPO only). GRPO eliminates the Critic by group-normalizing rewards within the same prompt; REINFORCE++ baseline (the 2025-2026 frontier choice) eliminates both Critic and group-rollout overhead. <strong>~80% of compute goes to rollout</strong> (sample generation), so the framework's job is keeping rollout fed and minimizing GPU idle time. Two architectural patterns dominate. <strong>OpenRLHF</strong>: dedicated Ray worker group per model, colocated by rank, vLLM-accelerated rollout, ZeRO-3 training. <strong>veRL</strong>: single unified WorkerDict housing all models, naturally shares resources. Both support <strong>Hybrid Engine</strong> — colocate models on the same GPUs and use vLLM sleep/wake to swap between training and inference modes — which became the production default. The disaggregated alternative (StreamRL, AReal) splits rollout to inference-optimized hardware and training to training-optimized hardware, async pipeline; better for very large rollout ratios. <strong>Distributed SFT</strong> is much simpler — single model, standard data parallel — but has its own concerns (packing, sequence parallelism for long contexts). <strong>Algorithm choice drives infrastructure</strong>: PPO needs four models, GRPO needs three (no Critic), REINFORCE++ needs two (no Critic, no group rollout). Frontier 2026 production: Magistral, ProRL V2, ScaleRL all use REINFORCE++ baseline.
</div>

<h2>Two new faces opening Part XI</h2>

<div class="character" style="--c: #c1502e;">
  <div class="avatar" style="background: #c1502e; color: #fff;">R</div>
  <div>
    <p class="who">Rollout</p>
    <p class="name">"I'm where 80% of the compute goes. My throughput determines how fast you train."</p>
    <p class="says">Every RLHF step starts with me. The Actor needs trajectories — sequences of generated tokens with their log-probs — to compute policy gradients. I produce them. Under the hood I'm vLLM (or SGLang, or TRT-LLM), running with continuous batching and PagedAttention. The Actor's weights synchronize to me at every step (or every K steps with off-policy variants); I generate a batch of rollouts; the trainer scores them and updates the Actor. <strong>If I'm slow, everything is slow.</strong> The frameworks compete on how fast they can keep me fed and how efficiently they can swap my GPU memory back to the trainer when needed. <em>I'm the bottleneck you didn't expect</em>.</p>
  </div>
</div>

<div class="character" style="--c: #1f5f5b;">
  <div class="avatar" style="background: #1f5f5b; color: #fff;">⊕</div>
  <div>
    <p class="who">Orchestrator</p>
    <p class="name">"I schedule four models across N GPUs without anyone stepping on each other."</p>
    <p class="says">In 2026, I'm Ray. I'm an actor framework — every model is a Ray actor (a process with its own state and methods); I route messages between them. The Actor calls me to fetch a fresh batch of prompts; I dispatch to Rollout to generate; I send the trajectories to Reward and Reference for scoring; I gather everything and ship it to the trainer. I handle failures (a worker dies, I restart it; gradient sync hangs, I detect it and recover). I handle resource scheduling (which actor gets which GPU, when does Hybrid Engine swap rollout for training). <strong>The reason RLHF can scale to 70B+ models is that I exist</strong>; without distributed orchestration, you're stuck at single-node and can't fit the four-models reality.</p>
  </div>
</div>

<h2>Why RLHF is a systems problem</h2>

<p>To see why distributed RLHF needs its own treatment, walk through what's running concurrently during a single PPO step:</p>

<ol>
  <li><strong>Actor</strong> generates rollouts. Forward pass on the policy model, sampling tokens autoregressively, recording per-token log-probabilities. Typically uses vLLM for fast generation. Memory: model weights (140GB for 70B in bf16) + KV cache.</li>
  <li><strong>Reference</strong> scores the rollouts for KL divergence. Forward pass on a frozen copy of the policy at SFT initialization. Same architecture as Actor; weights different (not updated). Memory: model weights again (another 140GB for 70B).</li>
  <li><strong>Reward</strong> scores the rollouts for reward signal. Forward pass on a separately trained reward model. Architecture often similar to Actor (regression head replacing LM head). Memory: model weights (another 140GB for 70B-class reward model, less if smaller).</li>
  <li><strong>Critic</strong> estimates value (PPO only). Forward + backward pass; trained alongside Actor. Memory: weights + optimizer states (heavier than Actor because optimizer states for both Actor and Critic).</li>
  <li><strong>Actor backward pass</strong> with PPO loss. Compute advantages, importance ratios, clip, KL penalty; backprop through Actor. Memory: weights + grads + optimizer states (Adam: 2× weights for moments).</li>
</ol>

<p>The math gets aggressive quickly. <strong>For 70B model class</strong>: 140GB Actor + 140GB Reference + 140GB Reward + 140GB Critic = 560GB just for weights. With Adam optimizer states for Actor + Critic (~280GB), gradients (~280GB), and rollout KV cache (~50-100GB), you're at ~1.2-1.3TB total. That's 16+ H100s (80GB each) just for state — before doing any actual computation.</p>

<p>This is why naive "load four models, run them" doesn't scale. The four frameworks that emerged (OpenRLHF, veRL, RLHFuse, StreamRL/AReal) are different answers to the question: <em>given this memory pressure and this compute distribution, how do you minimize idle GPU time?</em></p>

<h2>The four-model anatomy</h2>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 420" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
    <marker id="arrR2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#c1502e"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">PPO four-model topology — what's running, what's training, what's frozen</text>

  <!-- Prompts source -->
  <g transform="translate(20, 60)">
    <rect x="0" y="0" width="100" height="40" fill="#fff8a8" stroke="#1a1612" stroke-width="1.5"/>
    <text x="50" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">Prompts</text>
    <text x="50" y="32" text-anchor="middle" font-size="9" fill="#1a1612">batch from dataset</text>
  </g>

  <!-- Actor (training) -->
  <g transform="translate(180, 50)">
    <rect x="0" y="0" width="160" height="60" fill="#fff5d8" stroke="#d4a017" stroke-width="2"/>
    <text x="80" y="20" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">Actor (policy)</text>
    <text x="80" y="35" text-anchor="middle" font-size="9" fill="#1a1612">vLLM rollout + ZeRO training</text>
    <text x="80" y="50" text-anchor="middle" font-size="9" font-weight="700" fill="#1f5f5b">TRAINED</text>
  </g>

  <!-- Rollouts -->
  <g transform="translate(420, 50)">
    <rect x="0" y="0" width="140" height="60" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="70" y="20" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">Rollouts</text>
    <text x="70" y="35" text-anchor="middle" font-size="9" fill="#1a1612">tokens + log-probs</text>
    <text x="70" y="50" text-anchor="middle" font-size="9" fill="#6b5d4f">~80% of compute</text>
  </g>

  <line x1="120" y1="80" x2="178" y2="80" stroke="#1f5f5b" stroke-width="1.5" marker-end="url(#arrR)"/>
  <line x1="340" y1="80" x2="418" y2="80" stroke="#c1502e" stroke-width="1.5" marker-end="url(#arrR2)"/>

  <!-- Three frozen scorers -->
  <g transform="translate(180, 150)">
    <rect x="0" y="0" width="120" height="60" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1" stroke-dasharray="3 2"/>
    <text x="60" y="20" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">Reference</text>
    <text x="60" y="35" text-anchor="middle" font-size="8" fill="#1a1612">SFT-init copy</text>
    <text x="60" y="50" text-anchor="middle" font-size="9" font-weight="700" fill="#c1502e">FROZEN</text>
  </g>

  <g transform="translate(320, 150)">
    <rect x="0" y="0" width="120" height="60" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1" stroke-dasharray="3 2"/>
    <text x="60" y="20" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">Reward</text>
    <text x="60" y="35" text-anchor="middle" font-size="8" fill="#1a1612">trained separately</text>
    <text x="60" y="50" text-anchor="middle" font-size="9" font-weight="700" fill="#c1502e">FROZEN</text>
  </g>

  <g transform="translate(460, 150)">
    <rect x="0" y="0" width="120" height="60" fill="#fff5d8" stroke="#d4a017" stroke-width="2"/>
    <text x="60" y="20" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">Critic (value)</text>
    <text x="60" y="35" text-anchor="middle" font-size="8" fill="#1a1612">value head</text>
    <text x="60" y="50" text-anchor="middle" font-size="9" font-weight="700" fill="#1f5f5b">TRAINED (PPO only)</text>
  </g>

  <line x1="490" y1="105" x2="240" y2="148" stroke="#c1502e" stroke-width="1" marker-end="url(#arrR2)"/>
  <line x1="490" y1="105" x2="380" y2="148" stroke="#c1502e" stroke-width="1" marker-end="url(#arrR2)"/>
  <line x1="490" y1="105" x2="520" y2="148" stroke="#c1502e" stroke-width="1" marker-end="url(#arrR2)"/>

  <!-- Combined signals -->
  <g transform="translate(220, 240)">
    <rect x="0" y="0" width="320" height="50" fill="#ffd5dc" stroke="#b85a6c" stroke-width="2"/>
    <text x="160" y="20" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">Combined: KL(Actor‖Ref) + reward + advantage</text>
    <text x="160" y="40" text-anchor="middle" font-size="9" fill="#1a1612">PPO loss with clipping; backprop into Actor (and Critic)</text>
  </g>

  <line x1="240" y1="212" x2="320" y2="238" stroke="#1f5f5b" stroke-width="1" marker-end="url(#arrR)"/>
  <line x1="380" y1="212" x2="380" y2="238" stroke="#1f5f5b" stroke-width="1" marker-end="url(#arrR)"/>
  <line x1="520" y1="212" x2="440" y2="238" stroke="#1f5f5b" stroke-width="1" marker-end="url(#arrR)"/>

  <!-- Update arrow back to Actor -->
  <path d="M 380 290 Q 380 350 200 350 Q 100 350 100 110 Q 100 80 178 80" stroke="#1f5f5b" stroke-width="2" fill="none" marker-end="url(#arrR)"/>
  <text x="65" y="230" font-size="10" font-weight="700" fill="#1f5f5b" transform="rotate(-90 65 230)">PPO update →</text>

  <!-- Algorithm comparison box -->
  <g transform="translate(20, 320)">
    <rect x="0" y="0" width="700" height="80" fill="#fff8a8" stroke="#1a1612" stroke-width="1.5"/>
    <text x="350" y="20" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">Algorithm choice = which models you need</text>
    <text x="20" y="40" font-size="10" fill="#1a1612"><tspan font-weight="700">PPO</tspan>: Actor + Reference + Reward + Critic (4 models)</text>
    <text x="20" y="55" font-size="10" fill="#1a1612"><tspan font-weight="700">GRPO</tspan>: Actor + Reference + Reward (3 models — group-normalize rewards instead of value)</text>
    <text x="20" y="70" font-size="10" fill="#1a1612"><tspan font-weight="700">REINFORCE++ baseline</tspan>: Actor + Reference + Reward (3 models, simpler than GRPO; ProRL V2 / ScaleRL / Magistral 2025-2026)</text>
  </g>
</svg>
</div>

<h3>What each model does</h3>

<p><strong>Actor</strong> is the policy under training. It generates rollouts and is updated via gradient descent. In Hybrid Engine setups, Actor wears two hats: the rollout engine (vLLM-served, optimized for inference) and the training process (ZeRO-3-distributed, optimized for backprop). The two configurations share GPU memory but operate at different times.</p>

<p><strong>Reference</strong> is the SFT-initialized model frozen at the start of RL training. Its purpose: provide a baseline so the Actor's KL divergence can be computed (KL(Actor ‖ Reference)). The KL term in the PPO loss prevents the policy from drifting too far from the well-behaved SFT baseline. Always frozen, always in inference mode.</p>

<p><strong>Reward</strong> is a separately trained model that scores rollouts. Trained on preference data (chosen vs rejected pairs) using ranking loss; produces a scalar reward per sequence. Always frozen during RL. In 2026, the reward model is increasingly itself an LLM (LLM-as-judge with rubric, see M37) rather than a small classification head, which complicates infrastructure but improves reward quality.</p>

<p><strong>Critic</strong> is the value head — a model (often a copy of the Actor with a regression head) that estimates expected return from a given state. Trained alongside Actor in PPO. <em>GRPO and REINFORCE++ eliminate the Critic</em>, which is a substantial infrastructure simplification (one fewer trainable model, no Critic optimizer state, no value-function estimation noise).</p>

<h3>Why GRPO and REINFORCE++ won on infrastructure</h3>

<p>The original PPO recipe used a Critic to reduce variance in advantage estimates. Two newer approaches eliminate this:</p>

<ul>
  <li><strong>GRPO (DeepSeek, 2024)</strong>: generate K rollouts per prompt; normalize rewards within the group (subtract group mean, divide by group std). The "advantage" is just the normalized reward. No Critic needed. Cost: K× the rollout per training example.</li>
  <li><strong>REINFORCE++ baseline</strong>: classical REINFORCE with a moving-average baseline subtracted to reduce variance. No Critic, no group-rollout overhead. The OpenRLHF team analysis (Dec 2024) and follow-on work showed it's <em>more stable than GRPO and faster than PPO</em>. Logic-RL and PRIME (Feb 2025) demonstrated this; Magistral (June 2025) used it; ProRL V2 (Feb 2026) trained a SOTA 1.5B reasoning model with it; ScaleRL (Oct 2026) validated at large scale.</li>
</ul>

<p>The infrastructure implication: <em>frontier 2026 production is increasingly REINFORCE++ baseline, not PPO or GRPO</em>. Three models instead of four; simpler scheduling; less memory; same or better quality.</p>

<h2>Ray as the orchestration layer</h2>

<p>Why Ray and not a custom solution? Three reasons that compound:</p>

<ul>
  <li><strong>Actor model fits the problem</strong>. Each of (Actor, Reference, Reward, Critic) is naturally a stateful actor — it owns model weights, processes requests (forward pass, training step), reports results. Ray's actor abstraction is exactly this.</li>
  <li><strong>Distributed scheduling without writing it</strong>. Ray handles GPU allocation, message passing between actors across nodes, fault recovery (worker dies, Ray restarts it). Building this from scratch is months of work that's already been done.</li>
  <li><strong>Heterogeneous resources</strong>. The Rollout engine wants inference-optimized hardware (high memory bandwidth, FP8/FP4 acceleration); the trainer wants training-optimized (high FLOPs, NVLink-rich topology). Ray naturally handles "this actor goes on these GPUs, that actor goes on those GPUs."</li>
</ul>

<p>The basic Ray actor pattern for a model in RLHF:</p>

<pre><code><span class="kw">import</span> ray

<span class="dec">@ray.remote</span>(num_gpus=<span class="num">8</span>)
<span class="kw">class</span> <span class="ty">RolloutWorker</span>:
    <span class="kw">def</span> <span class="fn">__init__</span>(self, model_path):
        <span class="com"># Ray manages this actor on dedicated GPUs</span>
        <span class="kw">from</span> vllm <span class="kw">import</span> LLM
        self.engine = <span class="fn">LLM</span>(
            model=model_path,
            tensor_parallel_size=<span class="num">8</span>,
            gpu_memory_utilization=<span class="num">0.5</span>,    <span class="com"># leave room for trainer in Hybrid Engine</span>
            enable_sleep_mode=<span class="kw">True</span>,           <span class="com"># can release memory back to trainer</span>
        )

    <span class="kw">def</span> <span class="fn">generate</span>(self, prompts, sampling_params):
        <span class="kw">return</span> self.engine.<span class="fn">generate</span>(prompts, sampling_params)

    <span class="kw">def</span> <span class="fn">sleep</span>(self):
        <span class="com"># Release GPU memory; trainer can use it</span>
        self.engine.<span class="fn">sleep</span>()

    <span class="kw">def</span> <span class="fn">wake_up</span>(self, new_weights):
        <span class="com"># Reload weights from latest training step, resume rollout</span>
        self.engine.<span class="fn">wake_up</span>()
        self.engine.<span class="fn">load_weights</span>(new_weights)

<span class="dec">@ray.remote</span>(num_gpus=<span class="num">8</span>)
<span class="kw">class</span> <span class="ty">TrainerWorker</span>:
    <span class="kw">def</span> <span class="fn">__init__</span>(self, model_path):
        <span class="kw">import</span> deepspeed
        <span class="com"># ZeRO-3 sharded training</span>
        self.engine = <span class="fn">init_zero3_engine</span>(model_path)

    <span class="kw">def</span> <span class="fn">train_step</span>(self, rollouts, advantages, kl_terms):
        loss = <span class="fn">ppo_loss</span>(self.engine, rollouts, advantages, kl_terms)
        self.engine.<span class="fn">backward</span>(loss)
        self.engine.<span class="fn">step</span>()
        <span class="kw">return</span> self.engine.<span class="fn">get_weights</span>()      <span class="com"># pass back to Rollout</span>

<span class="com"># Top-level coordinator</span>
<span class="kw">def</span> <span class="fn">rlhf_step</span>(rollout_actor, trainer_actor, ref_actor, reward_actor, prompts):
    <span class="com"># 1. Generate rollouts</span>
    rollouts = ray.<span class="fn">get</span>(rollout_actor.<span class="fn">generate</span>.<span class="fn">remote</span>(prompts, sampling_params))

    <span class="com"># 2. Score with reward and reference (in parallel)</span>
    reward_future = reward_actor.<span class="fn">score</span>.<span class="fn">remote</span>(rollouts)
    ref_future    = ref_actor.<span class="fn">log_probs</span>.<span class="fn">remote</span>(rollouts)
    rewards, ref_lps = ray.<span class="fn">get</span>([reward_future, ref_future])

    <span class="com"># 3. Compute advantages, KL, etc.</span>
    advantages = <span class="fn">compute_advantages</span>(rewards, rollouts.log_probs)
    kl_terms = rollouts.log_probs - ref_lps

    <span class="com"># 4. Hybrid Engine: rollout sleeps, trainer takes over the GPUs</span>
    ray.<span class="fn">get</span>(rollout_actor.<span class="fn">sleep</span>.<span class="fn">remote</span>())

    <span class="com"># 5. Train Actor</span>
    new_weights = ray.<span class="fn">get</span>(trainer_actor.<span class="fn">train_step</span>.<span class="fn">remote</span>(rollouts, advantages, kl_terms))

    <span class="com"># 6. Reload Actor weights into rollout, resume</span>
    ray.<span class="fn">get</span>(rollout_actor.<span class="fn">wake_up</span>.<span class="fn">remote</span>(new_weights))</code></pre>

<p>This is the skeleton. Real OpenRLHF/veRL has more sophistication — async pipelining, fault tolerance, gradient accumulation, multi-step rollout, etc. — but the structure is recognizable. <em>The Hybrid Engine pattern (steps 4-6) is the key innovation</em>: the same GPUs serve both rollout and training, swapping memory between modes via vLLM's sleep/wake.</p>

<h2>OpenRLHF vs veRL: two architectural philosophies</h2>

<p>The two dominant frameworks in 2026 differ in a specific architectural decision: <em>how to organize the four model workers across resources</em>.</p>

<h3>OpenRLHF: dedicated workers per model</h3>

<p>Each model (Actor, Reference, Reward, Critic) has its own Ray worker group, each colocated by rank. If you allocate 32 GPUs total, OpenRLHF might give you 8 GPUs each for Actor, Reference, Reward, Critic — separately scheduled, separately tensor-parallelized.</p>

<p>Pros: clean separation; failure of one model's worker doesn't crash others; per-model resource tuning (Reference can use less compute than Actor).</p>

<p>Cons: GPUs allocated to (e.g.) Reference are idle when Reference isn't being called. The Hybrid Engine variant (--colocate_all_models) addresses this by sharing GPUs across all model workers and using vLLM sleep/wake to swap.</p>

<p>OpenRLHF's main features as of v0.10 (April 2026):</p>

<ul>
  <li>Distributed Ray architecture, ZeRO-3 + vLLM combination</li>
  <li>Hybrid Engine support via <code>--colocate_all_models</code> flag</li>
  <li>Algorithms: PPO, GRPO, REINFORCE++, REINFORCE++ baseline, RLOO, DPO, online RLHF</li>
  <li>VLM RLHF (April 2026 release): train Qwen3.5-VL with image inputs end-to-end</li>
  <li>Multi-turn VLM RL: agent loops with screenshots in environment feedback</li>
  <li>QLoRA/LoRA, RingAttention, FlashAttention, packing samples, MoE support</li>
</ul>

<h3>veRL: unified WorkerDict</h3>

<p>veRL takes the opposite approach: <em>a single WorkerDict that houses all models</em>, sharing resources naturally. Instead of "Actor's 8 GPUs, Reference's 8 GPUs," veRL has "32 GPUs, every WorkerDict instance has Actor + Reference + Reward weights co-resident."</p>

<p>Pros: maximizes GPU utilization (no idle GPUs holding only Reference weights); cleaner model swapping; simpler scheduling.</p>

<p>Cons: more memory pressure per GPU (must fit weights for all models); failure modes are coupled.</p>

<p>veRL's distinctive features:</p>

<ul>
  <li>Single-controller pattern inspired by Google's Pathways</li>
  <li>Megatron-LM integration for very large model training (671B-class)</li>
  <li>HybridFlow design: separate computation graphs for each algorithm phase, dispatch based on phase</li>
  <li>Strong support for MoE training in RLHF settings</li>
</ul>

<h3>Performance: published comparisons</h3>

<p>OpenRLHF's published comparison (v0.8.5 vs veRL v0.4.0, mid-2025) on 1.5B/7B/14B models with 1K/2K/4K/8K generation lengths showed OpenRLHF faster on average across the matrix, particularly at longer generation lengths. veRL's strength is at very large scale (70B+ MoE) where its unified scheduling pays off more. <em>For most production workloads (1.5B-70B dense), OpenRLHF tends to be faster; veRL's advantages grow with model size and architectural complexity.</em></p>

<p>The choice in 2026 isn't categorical — both frameworks are actively developed and converge feature-wise. Default to OpenRLHF for ease-of-use and broad algorithm support; choose veRL when you need Megatron integration or are training at the 100B+ scale.</p>

<h2>Hybrid Engine: the production default</h2>

<p>The single most important infrastructure idea in distributed RLHF as of 2026: <strong>Hybrid Engine</strong>. Both OpenRLHF and veRL support it; it's the production default for memory efficiency.</p>

<p>The problem it solves: in a four-model RLHF setup, each model needs GPUs. Naive allocation gives each model dedicated GPUs that sit idle when that model isn't being called. With four 70B models and dedicated allocation, you might use 32 GPUs, of which only 8 are doing work at any moment.</p>

<p>Hybrid Engine: <em>colocate all four models on the same GPUs</em>. Use vLLM's sleep/wake mechanism to swap GPU memory between modes. When the Actor is doing rollout, Actor's vLLM engine is awake; Reference, Reward, and Critic are sleeping (weights paged to CPU or compressed). When the trainer is doing the gradient step, vLLM sleeps; the trainer's ZeRO engine wakes.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 320" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Hybrid Engine vs Disaggregated: two ways to schedule the four-model dance</text>

  <!-- Hybrid Engine -->
  <g transform="translate(20, 50)">
    <rect x="0" y="0" width="340" height="240" fill="#fff5d8" stroke="#d4a017" stroke-width="2"/>
    <text x="170" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">Hybrid Engine (production default)</text>
    <text x="170" y="38" text-anchor="middle" font-size="9" fill="#6b5d4f">Same GPUs serve all roles, swap via sleep/wake</text>

    <!-- GPU pool -->
    <rect x="20" y="55" width="300" height="40" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="170" y="78" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">Shared GPU pool (e.g., 32× H100)</text>

    <!-- Phase A: rollout active -->
    <rect x="20" y="110" width="140" height="50" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="90" y="125" text-anchor="middle" font-size="9" font-weight="700" fill="#1a1612">Phase A: rollout</text>
    <text x="90" y="140" text-anchor="middle" font-size="8" fill="#1a1612">vLLM (Actor) awake</text>
    <text x="90" y="152" text-anchor="middle" font-size="8" fill="#1a1612">trainer asleep</text>

    <!-- Phase B: training active -->
    <rect x="180" y="110" width="140" height="50" fill="#fff5d8" stroke="#d4a017" stroke-width="1.5"/>
    <text x="250" y="125" text-anchor="middle" font-size="9" font-weight="700" fill="#1a1612">Phase B: training</text>
    <text x="250" y="140" text-anchor="middle" font-size="8" fill="#1a1612">trainer awake</text>
    <text x="250" y="152" text-anchor="middle" font-size="8" fill="#1a1612">vLLM asleep</text>

    <text x="170" y="180" text-anchor="middle" font-size="9" fill="#1a1612">Sleep/wake roundtrip: ~1-3 seconds</text>
    <text x="170" y="195" text-anchor="middle" font-size="9" fill="#1a1612">Weight transfer: NCCL or CUDA IPC</text>

    <text x="20" y="220" font-size="9" font-weight="700" fill="#1f5f5b">Pros: max GPU utilization, simple cluster</text>
    <text x="20" y="232" font-size="9" font-weight="700" fill="#c1502e">Cons: serial; no rollout-train overlap</text>
  </g>

  <!-- Disaggregated -->
  <g transform="translate(380, 50)">
    <rect x="0" y="0" width="340" height="240" fill="#ffd5dc" stroke="#b85a6c" stroke-width="2"/>
    <text x="170" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">Disaggregated (StreamRL, AReal)</text>
    <text x="170" y="38" text-anchor="middle" font-size="9" fill="#6b5d4f">Separate clusters; async pipeline</text>

    <!-- Two pools -->
    <rect x="20" y="55" width="140" height="40" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="90" y="73" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Rollout cluster</text>
    <text x="90" y="86" text-anchor="middle" font-size="8" fill="#1a1612">B200/H200, FP4/FP8</text>

    <rect x="180" y="55" width="140" height="40" fill="#fff5d8" stroke="#d4a017" stroke-width="1.5"/>
    <text x="250" y="73" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Training cluster</text>
    <text x="250" y="86" text-anchor="middle" font-size="8" fill="#1a1612">H100, NVLink-rich</text>

    <!-- Pipeline -->
    <text x="170" y="115" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Async pipeline</text>
    <rect x="20" y="125" width="60" height="35" fill="#d4ecc8" stroke="#1f5f5b"/>
    <text x="50" y="140" text-anchor="middle" font-size="8" fill="#1a1612">step k</text>
    <text x="50" y="152" text-anchor="middle" font-size="8" fill="#1a1612">rollout</text>
    <rect x="90" y="125" width="60" height="35" fill="#fff5d8" stroke="#d4a017"/>
    <text x="120" y="140" text-anchor="middle" font-size="8" fill="#1a1612">step k</text>
    <text x="120" y="152" text-anchor="middle" font-size="8" fill="#1a1612">train</text>
    <rect x="160" y="125" width="60" height="35" fill="#d4ecc8" stroke="#1f5f5b"/>
    <text x="190" y="140" text-anchor="middle" font-size="8" fill="#1a1612">step k+1</text>
    <text x="190" y="152" text-anchor="middle" font-size="8" fill="#1a1612">rollout</text>
    <rect x="230" y="125" width="60" height="35" fill="#fff5d8" stroke="#d4a017"/>
    <text x="260" y="140" text-anchor="middle" font-size="8" fill="#1a1612">step k+1</text>
    <text x="260" y="152" text-anchor="middle" font-size="8" fill="#1a1612">train</text>

    <text x="170" y="180" text-anchor="middle" font-size="9" fill="#1a1612">Rollout(k+1) overlaps with train(k)</text>
    <text x="170" y="195" text-anchor="middle" font-size="9" fill="#1a1612">Off-policy correction needed</text>

    <text x="20" y="220" font-size="9" font-weight="700" fill="#b85a6c">Pros: rollout-train overlap; specialized HW</text>
    <text x="20" y="232" font-size="9" font-weight="700" fill="#c1502e">Cons: complex; off-policy bias</text>
  </g>
</svg>
</div>

<p>The mechanism: <strong>vLLM's sleep mode</strong>. Released in 2024, refined through 2025-2026, this is what makes Hybrid Engine work. When called, vLLM unloads model weights from GPU memory (paging them to CPU, or releasing entirely if checkpointed elsewhere) and frees its KV cache. The GPU memory becomes available to whatever else needs it (the trainer). When wake_up is called, vLLM reloads weights and rebuilds KV cache. Roundtrip cost: 1-3 seconds for a 70B model on 8× H100.</p>

<p>The trainer-to-rollout weight transfer also matters. After a training step, the new Actor weights need to reach the rollout engine. Three patterns:</p>

<ul>
  <li><strong>NCCL broadcast</strong>: trainer ranks broadcast updated weights to rollout ranks. Fast but requires both to be alive and in the same NCCL communicator.</li>
  <li><strong>CUDA IPC</strong>: shared CUDA memory between processes. Zero-copy if same node; fastest option.</li>
  <li><strong>Disk + reload</strong>: trainer saves checkpoint, rollout reloads. Slowest but simplest; used in some early frameworks.</li>
</ul>

<p>OpenRLHF defaults to NCCL broadcast; veRL uses CUDA IPC where possible. The transfer takes 5-30 seconds for 70B-class models depending on pattern.</p>

<h2>Disaggregated: when rollout dominates</h2>

<p>Hybrid Engine's weakness: it's serial. Rollout phase runs, then training phase runs; they can't overlap because they share GPU memory. If rollout takes 80% of step time and training takes 20%, the trainer GPUs sit idle for 80% of the cycle.</p>

<p><strong>Disaggregated architectures</strong> (StreamRL from Alibaba, AReal from Anthropic-influenced teams) split rollout and training onto separate clusters with async pipelining. While step k+1 rollouts are generating, step k gradients are being computed. Both clusters are busy.</p>

<p>The catch: <em>the rollouts used for step k's gradient update are slightly off-policy</em> (they were generated with the Actor's weights from before step k's update). Algorithmic corrections (importance sampling, KL clipping) handle the mismatch but introduce some bias. For large rollout-to-training ratios this is acceptable; for small ratios the bias can dominate.</p>

<p>The disaggregated pattern wins specifically when:</p>

<ul>
  <li><strong>Rollout dominates training compute</strong> (long generation, large group sizes for GRPO)</li>
  <li><strong>Heterogeneous hardware available</strong> — B200/B300 for rollout (FP4 inference acceleration), H100 for training (NVLink-rich, FP8 training)</li>
  <li><strong>Off-policy bias is tolerable</strong> for the algorithm</li>
</ul>

<p>For most production workloads, Hybrid Engine on a homogeneous cluster is simpler and sufficient. Disaggregation is the choice for very-large-scale or heterogeneous-hardware setups.</p>

<h2>Algorithm choice drives infrastructure</h2>

<div class="table-wrap">
<table>
<caption>RLHF algorithms and their infrastructure footprint, April 2026</caption>
<thead><tr><th>Algorithm</th><th>Models needed</th><th>Rollout cost</th><th>Notes</th></tr></thead>
<tbody>
<tr><td>PPO (classical)</td><td>4 (Actor, Ref, Reward, Critic)</td><td>1× per prompt</td><td>Largest infra; original RLHF recipe</td></tr>
<tr><td>GRPO (DeepSeek)</td><td>3 (Actor, Ref, Reward)</td><td>K× per prompt (group)</td><td>No Critic; high rollout cost; reasoning models</td></tr>
<tr><td>RLOO</td><td>3 (Actor, Ref, Reward)</td><td>K× per prompt</td><td>Leave-one-out variant of GRPO</td></tr>
<tr><td><strong>REINFORCE++ baseline</strong></td><td>3 (Actor, Ref, Reward)</td><td>1× per prompt</td><td><strong>Frontier 2026</strong>; Magistral, ProRL V2, ScaleRL</td></tr>
<tr><td>DPO / IPO</td><td>2 (Actor, Ref)</td><td>0× (offline)</td><td>No RL loop; pure preference optimization</td></tr>
<tr><td>Iterative DPO</td><td>2 + judge</td><td>partial (sampling for new prefs)</td><td>Online preference loop</td></tr>
<tr><td>Online RLHF</td><td>varies</td><td>varies</td><td>Continuous preference collection</td></tr>
</tbody>
</table>
</div>

<p>The trajectory through 2024-2026 is unmistakable: <em>algorithms are getting simpler infrastructure-wise</em>. PPO had four models; GRPO has three with K× rollout; REINFORCE++ has three with 1× rollout. Each step removes complexity. The quality results have followed — REINFORCE++ at ProRL V2 scale (Feb 2026) and ScaleRL scale (Oct 2026) match or exceed PPO/GRPO with substantially less infrastructure overhead. The <strong>"PPO is the gold standard" era ended</strong>; the field is consolidating around REINFORCE++ baseline for new training runs.</p>

<h2>Distributed SFT: the prerequisite</h2>

<p>Before any RLHF, there's SFT (supervised fine-tuning). M11 covered training loop fundamentals; SFT adds specific concerns at the distributed scale:</p>

<ul>
  <li><strong>Single-model simplicity</strong>. SFT is just standard training on (prompt, response) pairs. One model, one optimizer, no four-model coordination. Vastly simpler than RL.</li>
  <li><strong>Sample packing</strong>: pack short sequences to fill the model's context window. With 4K-context model and average 500-token sequences, 8 sequences pack into one example. Reduces wasted compute on padding by ~85%. Essentially required for cost-effective SFT in 2026.</li>
  <li><strong>Sequence parallelism</strong>: for long-context SFT (32K+ context), the activation memory of a single sequence exceeds single-GPU memory. Sequence parallelism (RingAttention, M16-adjacent) splits a single long sequence across GPUs. Standard for reasoning-model SFT where chain-of-thought outputs are long.</li>
  <li><strong>FSDP or ZeRO-3</strong>: weight sharding so the model fits across the cluster. Same machinery as M17.</li>
  <li><strong>Cold-start SFT for RL</strong>: the standard recipe is "SFT first to give the model basic capabilities, then RL to optimize." SFT phase typically uses 10K-1M high-quality examples; RL phase uses preference data + rollouts.</li>
</ul>

<p>SFT data quality matters more than quantity for the cold-start phase. M35's data engineering applies: dedupe, filter for quality, ensure no contamination with downstream evals. The SFT phase's output is the Actor and Reference initialization for the subsequent RL phase.</p>

<h2>What goes wrong: failure modes</h2>

<p>Distributed RLHF has characteristic failure modes that infrastructure choices interact with:</p>

<ul>
  <li><strong>Entropy collapse</strong>: the policy becomes deterministic, exploring only a narrow distribution of outputs. Symptoms: rewards plateau, KL goes to zero, generation diversity drops. <em>Mitigation</em>: Entropulse (alternating SFT and RL phases, used in ComputerRL); entropy regularization in the loss; early stopping based on entropy floor.</li>
  <li><strong>Reward hacking</strong>: the policy finds rollouts that score high on the Reward model but aren't actually good. Symptoms: reward goes up while human judgment of quality goes down. <em>Mitigation</em>: better reward models (often LLM-as-judge with rubric, M37); shorter training; KL penalty to keep policy near Reference.</li>
  <li><strong>KL blowup</strong>: KL(Actor ‖ Reference) grows without bound; the policy drifts into incoherent territory. <em>Mitigation</em>: tune KL coefficient; clip KL term; use adaptive KL scheduling.</li>
  <li><strong>Length bias</strong>: policy learns to generate longer outputs because the reward model favors them. Symptoms: average generation length grows without quality improving. <em>Mitigation</em>: length penalty in reward; train reward model with length-controlled preferences.</li>
  <li><strong>Rollout-train mismatch</strong>: in disaggregated setups, the rollouts are too off-policy and PPO's importance ratio explodes. <em>Mitigation</em>: clip importance ratios more aggressively; reduce rollout-train staleness; switch to Hybrid Engine.</li>
  <li><strong>Hybrid Engine memory thrash</strong>: sleep/wake transitions take longer than expected because of fragmentation. <em>Mitigation</em>: pre-warm vLLM caches; use CUDA IPC for weight transfer; reduce vLLM gpu_memory_utilization to leave headroom.</li>
</ul>

<p>Most of these have <em>characteristic signatures in metrics</em> — you can detect them by watching the right curves. Production RL infrastructure includes dashboards for: per-step rollout time, training step time, KL divergence, reward distribution (mean and tail), entropy, average generation length, GPU utilization per actor. Any of these going off the expected curve is a debugging signal.</p>

<h2>Production reality April 2026</h2>

<p>Putting the pieces together — what's actually being run in production at the frontier as of April 2026:</p>

<ul>
  <li><strong>Magistral (June 2025, Mistral)</strong>: REINFORCE++ baseline for reasoning model training. Confirmed the algorithm at frontier scale.</li>
  <li><strong>DeepSeek R1 series</strong>: GRPO with rule-based rewards (math/code verification); large rollout groups; no learned reward model for the base R1-Zero variant. The reasoning-model baseline.</li>
  <li><strong>ProRL V2 (Feb 2026)</strong>: REINFORCE++ baseline trained a state-of-the-art 1.5B reasoning model with prolonged RL training (700+ stable RL steps). Demonstrated REINFORCE++ scales to long horizons.</li>
  <li><strong>ScaleRL (Oct 2026, projected)</strong>: REINFORCE++ baseline at large-scale validation. The trajectory from "interesting algorithm" to "frontier production default."</li>
  <li><strong>OpenRLHF v0.10 (April 2026)</strong>: VLM RLHF support added. Multi-turn VLM RL with image-in-prompt and image-in-environment-feedback (e.g., screenshots in computer-use agent training).</li>
  <li><strong>OpenRLHF-M</strong>: dedicated multimodal fork; trains Qwen3.5-VL-class models with image inputs end-to-end.</li>
  <li><strong>Hybrid Engine on B200/B300</strong>: emerging pattern. Rollout uses NVFP4 (4× memory of FP8, near-FP8 quality); training uses FP8 or BF16. Single-cluster Hybrid Engine becomes more attractive as B200's per-GPU memory (192GB) accommodates both modes more comfortably.</li>
  <li><strong>RL Environments as a service</strong>: Scale RL Environments, Tinker, Fireworks AI all sell environments. Anthropic estimated to spend tens of millions/year. Coding and computer-use are the proving grounds.</li>
</ul>

<p>The frontier algorithm-infrastructure stack as of April 2026: <strong>Cold-start SFT (with packing, sequence parallelism, FSDP) → Reward Model training (or rule-based / LLM-judge rewards) → REINFORCE++ baseline RL with Hybrid Engine, Ray orchestration, vLLM rollout, ZeRO-3 training</strong>. OpenRLHF or veRL as the framework layer. This pattern composes everything from M11 (training loop) through M17 (FSDP) through M27 (vLLM) through M34 (GRPO algorithm). M39 is what ties them into a working RLHF system.</p>

<div class="ndq">
<h4>About distributed RLHF/GRPO infrastructure</h4>

<p class="q">Why does rollout take 80% of compute? Isn't generation cheap relative to training?</p>
<p class="a">Per-token, generation is cheaper than training. But RLHF requires generating <em>thousands of tokens per training example</em> — the rollout has to complete a full response (or reasoning trace, often 2K-10K tokens) before the gradient step happens. Training, by contrast, processes that completed sequence in a single forward+backward pass. So even though training is FLOP-heavier per token, rollout dominates because it's many tokens. The ratio gets worse for reasoning models (longer generations, more tokens per example) and for GRPO (K× rollouts per prompt). For a 4K-token rollout with K=8 group size, you generate 32K tokens per training example; the train step processes ~4K tokens (one trajectory) plus advantages. Rollout-to-train ratio is 8:1 in token count, but generation tokens are cheaper per-FLOP, so the wall-clock ratio lands around 4:1 or 80/20.</p>

<p class="q">Why eliminate the Critic? Wasn't variance reduction its whole point?</p>
<p class="a">Two practical reasons. (1) <strong>Critic adds infrastructure overhead</strong> — another model in memory, another optimizer state, another set of gradients to coordinate. For a 70B Actor with full Adam optimizer, each saved model is ~280GB of state. (2) <strong>Critic adds noise of its own</strong> — the value function estimator is itself learned, with its own training dynamics, and bad value estimates can hurt more than they help. GRPO's group-normalization and REINFORCE++'s moving-average baseline are both simpler variance-reduction schemes that work empirically as well or better. The Critic's theoretical role (per-state baseline that minimizes variance) was always a relatively weak argument; in practice, <em>simpler baselines are competitive and have lower infrastructure overhead</em>. The 2025-2026 results consistently show this.</p>

<p class="q">When should I pick disaggregated (StreamRL) over Hybrid Engine?</p>
<p class="a">When two conditions hold: (a) <strong>rollout strongly dominates training time</strong> (>80% of step time) — this happens with very long generation, large GRPO group size, or computationally cheap training (LoRA, small model); (b) <strong>you have heterogeneous hardware available</strong> — B200/B300 cluster for rollout (FP4 acceleration), H100 cluster for training. If both hold, disaggregated lets you fully utilize both clusters by running rollout(k+1) in parallel with train(k). If either doesn't hold, Hybrid Engine is simpler and roughly as fast. <em>For most production teams in 2026 on homogeneous H100/H200 clusters, Hybrid Engine is the default.</em> Disaggregated is for frontier labs with mixed Blackwell/Hopper fleets and very large rollout ratios.</p>

<p class="q">Is the move from PPO to GRPO to REINFORCE++ a one-way trip, or might PPO come back?</p>
<p class="a">Probably not coming back, but the field isn't fully settled. PPO's value is variance reduction via the Critic; if someone shows that very high-quality value estimation matters at extremely large scale (trillion-parameter models with long-horizon rollouts), the Critic's overhead might re-justify itself. As of April 2026, no public results suggest this — REINFORCE++ baseline with moving-average variance reduction matches or beats PPO at every published scale. <em>The most likely future: REINFORCE++ baseline becomes the standard for general RLHF; specialized variants (e.g., DPO for offline preference learning, GRPO for reasoning models with verifier rewards) keep their niches</em>. PPO becomes a "historical baseline" — taught for completeness, not used for new training.</p>

<p class="q">How does this all change for MoE models?</p>

<p class="a">MoE adds two complications. (1) <strong>Routing instability during RL</strong>: the policy update changes which experts are active for which inputs; if routing changes too quickly, the experts don't get consistent gradient signals and quality degrades. Mitigation: lower learning rate; add routing-stability loss term; freeze routing for the first N RL steps. (2) <strong>All-to-all communication in rollout</strong>: MoE inference requires expert all-to-all every layer, which interacts with vLLM's batching. The vLLM MoE path is mature in 2026 but not universal across frameworks. veRL has stronger MoE support (Megatron-derived all-to-all); OpenRLHF added MoE support via aux_loss_coef and expert-parallelism flags. <em>For MoE RLHF in 2026, default to veRL unless you have specific OpenRLHF features you need</em>; expect to spend extra engineering on routing stability.</p>

<p class="q">For a small team training a 7B reasoning model with limited GPUs, what's the actual recipe?</p>
<p class="a">Concrete recipe for April 2026: (1) Use <strong>OpenRLHF</strong> as the framework. (2) <strong>Cold-start SFT</strong> on 10K-100K reasoning traces (M35-quality data). 8× H100, FSDP/ZeRO-3, packing samples, 1-3 epochs. (3) Skip the learned reward model if you can — use <strong>rule-based rewards</strong> (math/code verification) for tasks with verifiable answers; this avoids reward hacking and saves a model. (4) <strong>REINFORCE++ baseline</strong> as the algorithm. (5) <strong>Hybrid Engine</strong>: <code>--colocate_all_models --vllm_enable_sleep --vllm_gpu_memory_utilization 0.5</code>. 8× H100 fits 7B Actor + Reference + (small or rule-based) Reward easily. (6) Run for 500-2000 RL steps. (7) Monitor entropy, reward, KL; intervene if entropy collapses. <em>Total compute: roughly 1-2 weeks on 8× H100</em>; cost: $5-15K. This is genuinely tractable for a small team in 2026 — the infrastructure has matured enough that the bottleneck is usually data and reward design, not compute or framework engineering.</p>

<p class="q">How does this compose with multimodal (M38)?</p>
<p class="a">It composes via OpenRLHF-M (or OpenRLHF v0.10's VLM support, April 2026). The pattern: <strong>Actor and Reference are VLMs</strong> (vision encoder + LLM trunk + connector); rollout includes image-conditioned generation; reward model can score image-text pairs (or be rule-based for verifiable tasks). The infrastructure changes are specifically: (1) vLLM's VLM path is required for fast multimodal rollout; (2) memory pressure increases (vision tokens are added to context); (3) for multi-turn VLM RL (computer-use agents seeing screenshots), the environment provides images in feedback. <em>Computer-use agent training is the dominant production VLM RL workload in 2026</em>; ComputerRL achieved 48.9% on OSWorld using this stack. Coming attraction for M40.</p>
</div>

<h2>Code Magnets: implement a Hybrid Engine RLHF step</h2>

<p>You're writing the per-step coordinator for a Hybrid Engine RLHF pipeline. Three magnets are wrong choices.</p>

<div class="magnets">
<p>Arrange the magnets to compute one RLHF step.</p>

<div class="magnet-pool">
  <span class="magnet">def rlhf_step(rollout, trainer, ref, reward, prompts):</span>
  <span class="magnet">    rollouts = ray.get(rollout.generate.remote(prompts))</span>
  <span class="magnet">    rollouts = rollout.generate(prompts)</span>
  <span class="magnet">    rewards, ref_lps = ray.get([reward.score.remote(rollouts), ref.log_probs.remote(rollouts)])</span>
  <span class="magnet">    rewards = ray.get(reward.score.remote(rollouts)); ref_lps = ray.get(ref.log_probs.remote(rollouts))</span>
  <span class="magnet">    advantages = compute_advantages(rewards, rollouts.log_probs)</span>
  <span class="magnet">    ray.get(rollout.sleep.remote())</span>
  <span class="magnet">    new_weights = ray.get(trainer.train_step.remote(rollouts, advantages))</span>
  <span class="magnet">    ray.get(rollout.wake_up.remote(new_weights))</span>
  <span class="magnet">    new_weights = trainer.train_step(rollouts, advantages)</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">def</span> <span class="fn">rlhf_step</span>(rollout, trainer, ref, reward, prompts):
    rollouts = ray.<span class="fn">get</span>(rollout.generate.<span class="fn">remote</span>(prompts))
    rewards, ref_lps = ray.<span class="fn">get</span>([reward.score.<span class="fn">remote</span>(rollouts), ref.log_probs.<span class="fn">remote</span>(rollouts)])
    advantages = <span class="fn">compute_advantages</span>(rewards, rollouts.log_probs)
    ray.<span class="fn">get</span>(rollout.sleep.<span class="fn">remote</span>())
    new_weights = ray.<span class="fn">get</span>(trainer.train_step.<span class="fn">remote</span>(rollouts, advantages))
    ray.<span class="fn">get</span>(rollout.wake_up.<span class="fn">remote</span>(new_weights))</code></pre>
<p>The traps:</p>
<ul>
  <li><code>rollouts = rollout.generate(prompts)</code>: calls the actor's method directly instead of going through Ray's <code>.remote()</code> dispatch. This is wrong on multiple levels: (a) Ray actors don't have synchronous methods accessible from outside the actor — calling <code>rollout.generate(prompts)</code> would either fail or call a method on the local proxy without actually running it on the worker; (b) Ray's whole point is process-isolated, GPU-pinned execution — direct calls bypass the orchestration layer; (c) the worker's GPU resources don't even exist in the calling process, so the call would fail with missing CUDA context. Always go through <code>actor.method.remote(args)</code> + <code>ray.get(future)</code> for Ray actors.</li>
  <li><code>rewards = ray.get(reward.score.remote(rollouts)); ref_lps = ray.get(ref.log_probs.remote(rollouts))</code>: serializes two calls that should be parallel. <code>ray.get</code> blocks until the future resolves; calling it twice in sequence means waiting for reward scoring before starting reference log-prob computation, even though both are independent and can run on separate GPUs simultaneously. The correct pattern: launch both <code>.remote</code> calls (which return immediately), then <code>ray.get</code> the list of futures (which waits for all in parallel). For 70B Reference + Reward, this can save several seconds per step — adds up over thousands of steps.</li>
  <li><code>new_weights = trainer.train_step(rollouts, advantages)</code> (without <code>ray.get</code> + <code>.remote</code>): same issue as the first trap — direct call to a Ray actor's method. The trainer is on different GPUs in a different process; you have to dispatch via Ray. Additionally, this version skips the <code>rollout.sleep()</code> + <code>rollout.wake_up()</code> Hybrid Engine dance: without sleep/wake, the trainer can't allocate GPU memory because vLLM is still holding it. Both bugs compound to a non-functional pipeline.</li>
</ul>
<p>The pattern: <strong>generate via Ray.remote → score with reward and reference in parallel via Ray.remote → compute advantages locally → sleep rollout → train via Ray.remote → wake rollout with new weights</strong>. The <code>.remote()</code>/<code>ray.get()</code> dispatch is mandatory for cross-actor communication; Hybrid Engine's sleep/wake is mandatory for memory sharing; parallelizing reward and reference scoring is the optimization that compounds across thousands of steps.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each distributed RLHF concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>Actor</div>
  <div>A. The policy under training; generates rollouts and receives gradient updates.</div>

  <div>Reference</div>
  <div>B. Frozen SFT-init copy used to compute KL divergence; prevents policy drift.</div>

  <div>Critic</div>
  <div>C. Trained value head used by PPO; eliminated by GRPO and REINFORCE++.</div>

  <div>Hybrid Engine</div>
  <div>D. Colocate all models on shared GPUs; vLLM sleep/wake to swap rollout/training modes.</div>

  <div>Disaggregated (StreamRL)</div>
  <div>E. Separate rollout and training clusters; async pipeline; off-policy correction needed.</div>

  <div>REINFORCE++ baseline</div>
  <div>F. Frontier 2026 algorithm; 3 models, 1× rollout, moving-average variance reduction.</div>

  <div>Ray</div>
  <div>G. Actor-framework orchestration layer; manages model workers, scheduling, fault tolerance.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>Actor</strong> → A<br>
<strong>Reference</strong> → B<br>
<strong>Critic</strong> → C<br>
<strong>Hybrid Engine</strong> → D<br>
<strong>Disaggregated</strong> → E<br>
<strong>REINFORCE++ baseline</strong> → F<br>
<strong>Ray</strong> → G
</p>
<p>The mental shortcut: <em>Actor trains, Reference baselines KL, Critic estimated value (now retired), Hybrid Engine swaps modes on shared GPUs, Disaggregated splits clusters async, REINFORCE++ is the new frontier default, Ray orchestrates everything</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team is training a 70B reasoning model with GRPO on 32× H100 (80GB each). They report rollouts take 12 seconds, training takes 3 seconds, but each "step" takes 25 seconds total. What's eating the missing 10 seconds, and how would you fix it?</p>
<details class="answer"><summary>show answer</summary>
<p>The missing 10 seconds is almost certainly <strong>Hybrid Engine swap overhead and weight transfer</strong>. The 12 + 3 = 15 seconds is just the active compute; the other 10 seconds is everything that happens between phases:</p>
<p>(1) <strong>Rollout sleep</strong> (~1-3s): vLLM unloads weights, frees KV cache, releases GPU memory. (2) <strong>Trainer wake</strong> (~1-2s): ZeRO engine acquires GPU memory, materializes gradients/optimizer states. (3) <strong>Trainer sleep after step</strong> (~1-2s): release GPU. (4) <strong>Rollout wake</strong> (~2-3s): vLLM reloads weights, rebuilds KV cache. (5) <strong>Weight transfer</strong> (~3-5s for 70B): the new Actor weights from trainer to rollout via NCCL or CUDA IPC.</p>
<p>Mitigations to try, in priority order:</p>
<p>(a) <strong>CUDA IPC for weight transfer</strong> instead of NCCL broadcast — typically 2-3× faster within a node. veRL defaults to this; OpenRLHF needs configuration.</p>
<p>(b) <strong>Reduce vllm_gpu_memory_utilization</strong> to leave more headroom — paradoxically, less aggressive memory packing can make sleep/wake faster because there's less fragmentation to recover from.</p>
<p>(c) <strong>Pre-compile vLLM CUDA graphs</strong> at the first wake; subsequent wakes reuse the compilation.</p>
<p>(d) <strong>Switch to disaggregated</strong> (StreamRL/AReal pattern) if the rollout/train ratio is favorable. With 12s rollout and 3s training, ratio is 4:1 — borderline. Disaggregated would let train(k) overlap with rollout(k+1), potentially halving step time at the cost of off-policy bias.</p>
<p>(e) <strong>Verify the algorithm choice</strong>: GRPO with K=8 group size means each "training example" required 8 rollouts. If switching to REINFORCE++ baseline (1× rollout per example), rollout time drops from 12s to ~1.5s, and the swap overhead becomes a much smaller fraction of step time. <em>This is often the biggest single win — algorithm choice dominates infrastructure choice for total throughput</em>.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Why does REINFORCE++ baseline scale better than GRPO in production, even though GRPO's variance reduction (group-normalization) seems "more principled"?</p>
<details class="answer"><summary>show answer</summary>
<p>Three compounding reasons:</p>
<p>(1) <strong>Rollout cost is K× lower</strong>. GRPO requires K rollouts per prompt to compute the group statistics (typical K=4-16). REINFORCE++ baseline uses 1 rollout per prompt with a moving-average baseline computed across the batch. For the same training-example count, GRPO does 4-16× more generation. Since rollout is 80% of compute, REINFORCE++ is 3-15× faster wall-clock per training example.</p>
<p>(2) <strong>Variance reduction quality is comparable empirically</strong>. The "principled" argument for group-normalization (reduces variance per group of K rollouts) is real in expectation but the actual variance reduction depends on K and on how similar the K rollouts are. With small K (K=4), the group statistics are themselves noisy; with large K, you pay K× rollout cost. Moving-average baseline (REINFORCE++) achieves similar variance reduction by averaging across thousands of past rollouts — strictly more samples than any group can provide. Empirically, this is comparable or better at scale.</p>
<p>(3) <strong>Stability properties are better</strong>. Logic-RL and PRIME (Feb 2025) showed REINFORCE++ is more stable than GRPO under prolonged training. ProRL V2 (Feb 2026) trained 700+ stable steps on a 1.5B reasoning model with REINFORCE++ baseline, while GRPO often shows entropy collapse or KL blowup at similar horizons. <em>Stability matters more than peak quality at scale</em> — a stable algorithm that runs for 1000 steps beats a higher-peak algorithm that diverges at step 300.</p>
<p>The combined argument: lower compute, comparable variance reduction, better stability. The "more principled" framing mistook a theoretical property for a practical advantage. <em>2026's lesson: empirical scaling beats theoretical elegance for RLHF algorithm choice</em>.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> Walk through the failure mode "entropy collapse" — what causes it, what it looks like in metrics, and what infrastructure choices interact with it.</p>
<details class="answer"><summary>show answer</summary>
<p><strong>What causes it</strong>: the policy becomes too confident, sampling only a narrow distribution of outputs. This happens because (a) the reward signal is consistently rewarding a particular output style, (b) the KL penalty isn't strong enough to keep the policy near Reference, (c) training has run too long without re-injecting diversity. The policy effectively memorizes "high-reward outputs" and stops exploring.</p>
<p><strong>What it looks like in metrics</strong>: <em>entropy of policy outputs drops sharply</em> (the per-token entropy averaged across rollouts goes from ~3-5 nats to ~0.5-1 nats). <em>KL(Actor ‖ Reference) plateaus or even decreases</em> (because the policy becomes deterministic, the KL term saturates). <em>Reward plateaus or decreases</em> (the policy is exploiting a local optimum that doesn't generalize). <em>Generation diversity drops</em> (samples become repetitive, formulaic).</p>
<p><strong>Infrastructure interactions</strong>:</p>
<p>(a) <strong>Hybrid Engine vs Disaggregated</strong>: Hybrid Engine has stricter on-policy semantics (rollouts and training are synchronous), making collapse more visible step-to-step. Disaggregated has rollout-train staleness which can <em>mask</em> early collapse signs but doesn't prevent collapse.</p>
<p>(b) <strong>Algorithm choice</strong>: GRPO with rule-based rewards and small K can collapse faster than REINFORCE++ because group-normalization removes information about absolute reward levels — the policy can game the relative ranking even when absolute reward stagnates. REINFORCE++ baseline preserves more reward signal.</p>
<p>(c) <strong>Rollout sampling parameters</strong>: temperature=0.7 vs 1.0 makes a big difference. Lower temperature produces more deterministic outputs, accelerating collapse. Production runs use temperature 0.7-1.0 for rollout, sometimes higher early in training.</p>
<p>(d) <strong>Mitigations as infrastructure features</strong>: <strong>Entropulse</strong> (alternating SFT and RL, used in ComputerRL) is a training-loop pattern that requires infrastructure support — periodically pause RL, run a few SFT steps on diverse data, resume RL. Frameworks need to support this. Entropy bonus in the loss is simpler — add a small coefficient × negative entropy to the objective. Most frameworks support this; tune the coefficient.</p>
<p>The general lesson: <strong>monitoring entropy is mandatory</strong>. If your dashboard doesn't have a per-step entropy plot, you'll discover collapse only after the model is already broken. Production RL infrastructure includes entropy alerts.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> A team wants to do "Distributed SFT then GRPO" for a 30B reasoning model. Sketch the infrastructure stack they need at each phase.</p>
<details class="answer"><summary>show answer</summary>
<p>Two phases with substantially different infrastructure profiles.</p>
<p><strong>Phase 1: SFT</strong>. Single 30B model, ~60GB in bf16, plus optimizer states (Adam: ~120GB) plus gradients (~60GB) ≈ 240GB total state. Fits in 8× H100 with ZeRO-2 or smaller cluster with ZeRO-3. Stack:</p>
<ul>
  <li><strong>Framework</strong>: Hugging Face <code>transformers</code> + <code>accelerate</code> with DeepSpeed ZeRO-3, or <code>torchtitan</code>, or <code>lit-gpt</code>. Standard options.</li>
  <li><strong>Parallelism</strong>: ZeRO-3 (or FSDP) for memory; data parallelism across nodes; sequence parallelism (RingAttention) if context exceeds 16K.</li>
  <li><strong>Data</strong>: 10K-100K reasoning traces; sample-packed to fill context efficiently. Quality &gt; quantity per M35.</li>
  <li><strong>Compute</strong>: 8-32 H100s, 1-3 days for 30B with packed 4K context.</li>
  <li><strong>Output</strong>: SFT checkpoint that becomes both the Actor initialization AND the Reference for the RL phase.</li>
</ul>
<p><strong>Phase 2: GRPO</strong>. Three models needed (Actor, Reference, Reward). For 30B at 60GB each: 180GB just for weights. Add Actor optimizer states (120GB) and gradients (60GB), plus rollout KV cache. Stack:</p>
<ul>
  <li><strong>Framework</strong>: OpenRLHF or veRL with Hybrid Engine. OpenRLHF for ease of use; veRL for stronger MoE support if applicable.</li>
  <li><strong>Configuration</strong>: <code>--colocate_all_models --vllm_enable_sleep --vllm_gpu_memory_utilization 0.5</code> on 16-32 H100s.</li>
  <li><strong>Algorithm</strong>: GRPO with K=8 group size, OR (recommended in 2026) REINFORCE++ baseline for substantially lower rollout cost.</li>
  <li><strong>Reward</strong>: rule-based if tasks have verifiable answers (math, code), or trained 7B reward model, or LLM-as-judge.</li>
  <li><strong>Data</strong>: 10K-100K prompts (no need for canonical answers — just the prompt; the policy generates responses).</li>
  <li><strong>Compute</strong>: 16-32 H100s for 1-2 weeks for 500-2000 RL steps.</li>
  <li><strong>Monitoring</strong>: per-step entropy, reward distribution (mean + tail), KL divergence, average generation length, GPU utilization. Alerts on entropy floor.</li>
</ul>
<p><strong>Practical advice</strong>: don't skip the cold-start SFT — RL from scratch (R1-Zero style) works for some tasks but is much harder. SFT gives the model a good starting policy that RL can refine. Total budget: ~3-4 weeks of engineering + ~$50-100K of compute on 16-32 H100s for 30B. <em>Tractable for a well-resourced team in 2026; unthinkable two years ago.</em></p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>RLHF/GRPO is fundamentally a <strong>systems problem disguised as an ML problem</strong>. ~80% of compute goes to rollout (sample generation); coordinating four models concurrently is harder than coordinating any single training run.</li>
  <li><strong>Four-model anatomy of PPO</strong>: Actor (trained, generates rollouts), Reference (frozen, KL baseline), Reward (frozen, scores rollouts), Critic (trained, value head). For 70B-class: ~560GB just for weights, ~1.2-1.3TB total state with optimizer + gradients + KV cache.</li>
  <li><strong>Algorithm choice drives infrastructure</strong>. PPO needs 4 models. GRPO eliminates the Critic via group-normalization — 3 models, but K× rollout cost. <strong>REINFORCE++ baseline</strong> (the 2025-2026 frontier) eliminates both Critic and group-rollout overhead — 3 models, 1× rollout, moving-average variance reduction.</li>
  <li>Frontier 2026 production using REINFORCE++ baseline: <strong>Magistral</strong> (Mistral, June 2025), <strong>ProRL V2</strong> (Feb 2026, SOTA 1.5B reasoning), <strong>ScaleRL</strong> (Oct 2026, large-scale validation).</li>
  <li><strong>Ray</strong> is the orchestration layer for all major frameworks. Each model is a Ray actor; Ray handles scheduling, message passing, fault tolerance.</li>
  <li><strong>OpenRLHF vs veRL</strong>: dedicated worker groups per model (OpenRLHF) vs unified WorkerDict housing all models (veRL). OpenRLHF is faster on average for 1.5B-70B dense; veRL has stronger MoE and very-large-scale support. Both support Hybrid Engine.</li>
  <li><strong>Hybrid Engine</strong>: production default. Colocate models on shared GPUs; use vLLM sleep/wake to swap between rollout (vLLM awake) and training (ZeRO awake) modes. Roundtrip: 1-3 seconds for 70B.</li>
  <li><strong>Weight transfer trainer→rollout</strong>: NCCL broadcast (OpenRLHF default), CUDA IPC (veRL default, faster), or disk reload (slow, simple).</li>
  <li><strong>Disaggregated (StreamRL, AReal)</strong>: separate rollout and training clusters; async pipeline; rollout(k+1) overlaps with train(k). Better when rollout strongly dominates and heterogeneous hardware (Blackwell rollout + Hopper training) is available. Off-policy bias requires algorithmic correction.</li>
  <li><strong>Distributed SFT</strong> is the prerequisite. Single model, standard training loop. Specific concerns: sample packing (8× efficiency for short sequences), sequence parallelism for long contexts, FSDP/ZeRO-3 for memory. Cold-start SFT before RL is the standard recipe.</li>
  <li><strong>Failure modes</strong>: entropy collapse (Entropulse alternating), reward hacking (better reward / shorter training), KL blowup (clip / adaptive scheduling), length bias (length-controlled rewards), Hybrid Engine memory thrash. Monitoring entropy, reward distribution, KL, and average length is mandatory.</li>
  <li><strong>OpenRLHF v0.10 (April 2026)</strong>: VLM RLHF support; multi-turn VLM RL with screenshots in environment feedback. Composes with M38's multimodal architectures.</li>
  <li>The reflex: when designing an RLHF/GRPO training run, ask "which algorithm? what's the rollout-to-train ratio? Hybrid Engine or disaggregated? Rule-based reward or learned?" These four choices determine 80% of the infrastructure complexity. Get them right and the rest follows.</li>
</ul>
</div>

<p>Module 40 (next) will cover <strong>agentic RL and RL Environments</strong> — Environment-as-a-Service, ComputerRL/OSWorld at 48.9%, RLinf for embodied AI, the verifier engineering problem, online vs local environments. The dominant production RL paradigm in 2026 builds directly on M39's distributed infrastructure.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">39</span>
  <span>Distributed RLHF &amp; GRPO infrastructure</span>
</div>
"""

emit("39_distributed_rlhf", "Module 39 — Distributed RLHF & GRPO infrastructure", BODY)
