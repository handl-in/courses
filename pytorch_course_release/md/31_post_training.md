# Module 31 — Post-training: SFT, RM, PPO & DPO

# _Post-training:_ SFT, reward models, PPO & DPO

_Part IX · Module 31_

— how a base model becomes a useful chat model, why PPO needs four models in memory, why DPO replaced it for most teams, and the variants (KTO, ORPO, GRPO) that solve the data and infrastructure problems

\--- 

For 30 modules we've covered training a base model — the pretraining run. But a base model trained to predict the next token isn't directly usable as a chat assistant. Ask it "What's the capital of France?" and it might continue with another question instead of answering. It hasn't learned to follow instructions, refuse harmful requests, or distinguish good answers from bad ones. **Post-training** is the phase that closes this gap.

This module covers the four-stage RLHF pipeline (Supervised Fine-Tuning → Reward Modeling → RLHF), the modern simplification (Direct Preference Optimization replacing the last two stages), the variants competing for the post-training spotlight, and the distributed-rollout engineering that production setups require. By the end you'll know which method to pick for your situation, what each costs in compute and engineering effort, and why the field went from "PPO is the answer" in 2022 to "DPO and friends" in 2024-2026.

> **★ KEY IDEA**  
>  The classic RLHF pipeline has four stages: **(1) pretrain** the base model on next-token prediction; **(2) SFT** on instruction-response pairs (loss-masked on the prompt); **(3) train a reward model** on preference pairs via Bradley-Terry; **(4) RL fine-tune** the policy with PPO against the reward, with a KL penalty against the SFT model. **DPO** short-circuits this: under a Bradley-Terry assumption, the optimal RLHF policy has a closed form involving only the preference data, eliminating stages 3 and 4. PPO needs _four models in memory_ (policy, ref, reward, value) and elaborate trainer-rollout infrastructure; DPO needs _two models_ (policy + frozen reference) and a single trainer. **DPO became the default for most teams in 2024-2025** because the engineering simplification is enormous and the quality is comparable on most benchmarks. 

## Two new faces

RM

Reward Model

"I'm a scalar judge. Given a prompt and a response, I produce one number — how much someone preferred this response."

I'm initialized from the SFT model, then trained on pairs of (chosen, rejected) responses to predict which one humans (or AI labelers) preferred. My loss is `-log σ(r(chosen) - r(rejected))` — Bradley-Terry. I'm the same shape as the policy I'll judge, with one new linear head producing a scalar instead of vocab logits. **I'm the bottleneck of classic RLHF** : my quality caps the policy's quality, my biases become the policy's biases, and reward-hacking against me is what makes PPO unstable. The shift to DPO eliminated me entirely — and many teams haven't missed me.

DPO

DPO

"I'm the closed-form RLHF. Skip the reward model entirely; train directly on preference pairs."

My loss is a single line: `-log σ(β · (log π(chosen) - log π_ref(chosen)) - β · (log π(rejected) - log π_ref(rejected)))`. Just policy log-probs, reference log-probs, and the chosen/rejected pair. _That's the entire algorithm._ No reward model to train, no rollouts to run, no PPO instabilities. Two models in memory (policy + frozen reference). I'm derived from the same Bradley-Terry assumption as the reward model — my math says: _under that assumption, the optimal RLHF policy can be expressed in closed form, and that closed form gives this loss_. The engineering simplification is enormous and the quality is competitive. **Most teams in 2026 use me or one of my variants.**

## The base-model-to-chat-model gap

A pretrained model is good at one thing: predicting the next token in text that looks like its training distribution. Chat assistants need three properties the base model lacks:

  1. **Instruction following** : the model should respond to "What is X?" with an explanation, not another question or a continuation in the same style.
  2. **Helpful and stylistic alignment** : responses should be useful, well-structured, in the right register. The base model has no reason to be helpful; it just continues plausible text.
  3. **Refusal of harmful requests** : the model should decline to produce dangerous content. The base model will happily continue any prompt it sees.

Closing each gap requires its own training signal. Instruction following needs examples of instructions and good responses (SFT). Helpfulness needs preference data — humans (or AIs) judging which responses are better. Refusal needs deliberate red-teaming and refusal training.

## The pipeline, with the modern shortcut

Post-training: classic RLHF (top) vs DPO shortcut (bottom) ① Pretrain next-token prediction on web-scale text ② SFT instruction-response pairs, prompt-masked ③ Reward Model Bradley-Terry on preference pairs ④ PPO RL fine-tune policy + ref + RM + value — FOUR models in memory — DPO short-circuit ↗ ③+④: DPO single loss on policy + frozen reference, — TWO models in memory — Engineering cost (rough order-of-magnitude): Classic PPO: weeks to months. 4 models. Distributed rollouts. KL coefficient tuning. Reward-hacking detection. DPO: days. 2 models. Standard distributed training infra. Lower variance. DPO won't always match PPO peak quality, but the engineering gap is huge in 2026: most teams DPO; frontier labs still PPO for the last 1-2% quality

The diagram shows both paths. Three checkpoints flow: from pretrain to SFT (where instruction-following is taught), then either down to reward model + PPO (classic, more powerful, more expensive) or directly to DPO (modern shortcut, simpler, fast). The choice is mostly about engineering capacity and the value of the last few percent of quality.

## Stage 1: Pretrain (briefly)

Already covered through M29. The base model is a transformer trained on next-token prediction on internet-scale text. By the time post-training begins, the base model has the world knowledge — it just can't access it usefully.

## Stage 2: Supervised Fine-Tuning (SFT)

SFT is exactly the same loss as pretraining (cross-entropy on next-token prediction) with two changes:

  1. **Different data** : instruction-response pairs formatted with a chat template, instead of raw web text. Examples: `{"messages": [{"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris."}]}`.
  2. **Loss masking on the prompt** : only compute loss on the _response_ tokens, not the prompt tokens. We don't want the model learning to "predict" instructions; we want it learning to follow them and produce good responses.

    
    
    def sft_loss(model, batch):
        # batch contains: input_ids, labels (where labels=-100 means "ignore" — applied to prompt tokens)
        logits = model(batch["input_ids"])              # [B, T, vocab]
    
        # Shift: predict token t from positions 0..t-1
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
    
        # Cross-entropy ignores positions where label = -100 (prompt tokens)
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            ignore_index=-100,
        )

That's it — the rest is data engineering. The model architecture is unchanged; the optimizer is AdamW (M9) with a much lower learning rate than pretraining (typically 1e-5 to 5e-6); the schedule is cosine with brief warmup. SFT runs for 1-3 epochs on the curated dataset; longer than 3 epochs typically overfits.

**Dataset quality dominates everything in SFT**. A small, carefully curated dataset (1K-10K examples) can outperform a large, noisy one (100K+ examples). Critical observations from the LIMA paper, Alpaca, and downstream work:

  * **Diverse instructions matter more than volume**. 1K examples covering many task types beats 100K examples that are all variants of the same task.
  * **Response quality is non-negotiable**. If the SFT data has bad responses (incorrect, poorly formatted, in the wrong tone), the model learns those patterns.
  * **Filter aggressively for length, refusals, and tone consistency**. SFT is when stylistic patterns get baked in.

The chat template is also fixed at SFT time. Llama-3's chat template uses special tokens like `<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`. Once the model learns this format during SFT, deviating at inference time will break it.

## Stage 3: Reward Modeling

The reward model takes a prompt and a response and outputs a scalar — how good the response is for that prompt. The training signal comes from **preference pairs** : pairs of responses to the same prompt where one was judged better than the other.

The standard loss is **Bradley-Terry** — a classical model from psychometrics for ranking from pairwise comparisons:
    
    
    P(chosen > rejected | prompt) = σ(r(prompt, chosen) - r(prompt, rejected))
    
    L_RM = -E[log σ(r(prompt, chosen) - r(prompt, rejected))]

Train the reward model to maximize this. The reward model architecture: take the SFT model, replace the LM head (vocab projection) with a single linear layer producing a scalar. Initialize this new head, freeze nothing. Train end-to-end on preference pairs.
    
    
    class RewardModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            # Same transformer as base model, but with scalar head instead of LM head
            self.transformer = base_model.transformer
            self.value_head  = nn.Linear(base_model.cfg.d_model, 1, bias=False)
    
        def forward(self, input_ids):
            # Run the transformer; take the hidden state at the LAST token
            hidden = self.transformer(input_ids)         # [B, T, D]
            last_hidden = hidden[:, -1, :]                 # [B, D]
            # Project to a scalar reward
            reward = self.value_head(last_hidden).squeeze(-1)  # [B]
            return reward
    
    def rm_loss(rm, chosen_ids, rejected_ids):
        r_chosen   = rm(chosen_ids)                       # [B]
        r_rejected = rm(rejected_ids)                     # [B]
        return -F.logsigmoid(r_chosen - r_rejected).mean()

The reward at the last token is the reward for the entire sequence. Training takes the SFT model's compute budget (similar dataset size, similar number of steps). Output: a model the same size as the policy that scores responses.

Two practical issues with reward models:

  * **Reward hacking** : the policy will exploit any flaw in the reward model. If the RM rewards length, the policy generates verbose responses. If the RM is over-confident on a domain, the policy gravitates there. Whatever pattern the RM rewards, the policy amplifies. **This is the central failure mode of PPO**.
  * **Distribution shift** : the RM was trained on responses from the SFT model. As the policy improves during PPO, its outputs drift away from the RM's training distribution. The RM becomes less reliable on the new outputs — and the policy is now optimizing against a less reliable signal.

## Stage 4: PPO — the engineering nightmare

Now the policy is trained to maximize reward via PPO (Proximal Policy Optimization). This is where post-training stops being deep learning and starts being reinforcement learning — with all the instabilities that entails.

The objective:
    
    
    L_PPO = E[r(prompt, response)]                       # maximize reward
            - β · KL(policy || ref_policy)              # stay close to SFT model
    
      where:
        response   ~ policy(· | prompt)                  # sampled from current policy
        r(...)     = reward_model(prompt, response)
        KL(...)    = sum_t log(policy(token_t | ...) / ref_policy(token_t | ...))

The KL penalty is critical. Without it, the policy drifts arbitrarily far from the SFT model — exploiting the reward model and producing degenerate outputs ("repeat 'helpful' 1000 times"). The KL keeps the policy close to the SFT distribution; the reward signal moves it gently in the helpful direction.

**Four models live in memory simultaneously during PPO:**

PPO has four models in memory; DPO has two — the engineering gap is enormous PPO — four models ~5× the base model's memory at 70B ① Policy trainable, full bf16 \+ optimizer state (4×) ② Reference frozen, bf16 SFT checkpoint ③ Reward Model frozen, bf16 scores responses ④ Value trainable, bf16 predicts future reward For 70B base in bf16: policy: 140 GB + 280 GB optimizer = 420 GB reference: 140 GB RM: 140 GB · value: 140 GB + 280 GB optim = 420 GB ≈ 1.1 TB across all GPUs DPO — two models ~3× the base model's memory at 70B ① Policy trainable, full bf16 \+ optimizer state (4×) ② Reference frozen, bf16 SFT checkpoint For 70B base in bf16: policy: 140 GB + 280 GB optim = 420 GB reference: 140 GB ≈ 560 GB total No rollouts. No reward model. Just standard distributed training. PPO needs ~2× the GPUs of DPO for the same model — and the rollout infrastructure

Four models, plus their optimizer states for the trainable ones. For a 70B base model:

  * **Policy** : 70B params × 2 bytes (bf16) = 140 GB. Plus AdamW optimizer state (M12): 4× model size in fp32 master weights + first/second moments = 280 GB. Subtotal: 420 GB.
  * **Reference** (frozen SFT checkpoint): 140 GB.
  * **Reward Model** (frozen): 140 GB.
  * **Value Model** (trainable, used to predict future reward for variance reduction): 420 GB with optimizer.

Total: ~1.1 TB across the cluster. With FSDP (M17) and quantization (M26), this can be reduced to ~700 GB, but you still need the GPUs to host four model copies.

**The trainer-rollout architecture** is what makes PPO actually work at scale. The naive approach — generate responses on the trainer, then update the policy — wastes the GPU. Generation is decode-bound (M27); training is compute-bound. Mixing them gives terrible utilization.

Modern PPO setups separate the two:

  1. **Rollout workers** (typically vLLM instances): hold a copy of the current policy in inference mode (paged KV, continuous batching). Generate batches of responses to prompts.
  2. **Trainer** : takes the (prompt, response, reward) tuples and updates the policy via PPO. After every K updates, broadcasts the new weights to the rollout workers.

The two run on different GPUs (or at least different processes), with the rollout queue providing a buffer between them. **This architecture is what TRL (Hugging Face's RL library), OpenRLHF, NeMo-RL, and most production setups implement.** Without it, PPO at scale is effectively impossible — you'd spend 80% of your compute on inference inside the trainer process.

The PPO step itself, simplified:
    
    
    def ppo_step(policy, ref, value, rm, prompts):
        # 1. Rollout: generate responses with the current policy
        responses = policy.generate(prompts)            # done by rollout workers in production
    
        # 2. Compute rewards
        rewards = rm(prompts, responses)                # scalar per response
    
        # 3. Compute log-probs from policy and reference
        with torch.no_grad():
            ref_logprobs = ref(prompts, responses).log_softmax(-1).gather(...)
        pol_logprobs = policy(prompts, responses).log_softmax(-1).gather(...)
    
        # 4. Compute the per-token KL penalty
        kl_penalty = pol_logprobs - ref_logprobs
    
        # 5. Adjust rewards: reward at last token + per-token KL penalty
        shaped_rewards = rewards.unsqueeze(-1) - beta * kl_penalty
    
        # 6. Compute advantages via GAE (generalized advantage estimation), using the value model
        values = value(prompts, responses)
        advantages = compute_gae(shaped_rewards, values, gamma=0.99, lam=0.95)
    
        # 7. PPO clipped objective
        ratio = (pol_logprobs - pol_logprobs_old).exp()
        pg_loss = -torch.min(ratio * advantages, ratio.clamp(1-eps, 1+eps) * advantages).mean()
    
        # 8. Value loss: predict observed returns
        v_loss = (values - returns).pow(2).mean()
    
        return pg_loss + v_coef * v_loss

Even simplified, this is a lot. Real PPO setups have additional components: gradient accumulation across rollout batches, careful KL coefficient adaptation (the KL coefficient should auto-adjust to keep KL ≈ target), reward whitening, advantage normalization, response-length penalties to prevent length-hacking, reward-clipping. _Each of these has its own production failure mode_.

The list of things that can go wrong:

  * **Reward hacking** : policy finds a degenerate way to score high on the reward model that humans wouldn't endorse. Repetitive outputs, sycophancy, length-padding.
  * **KL collapse** : KL penalty too low, policy drifts too far from reference, outputs become incoherent.
  * **KL explosion** : KL penalty too high, policy doesn't learn anything new.
  * **Mode collapse** : policy converges to a single response style.
  * **Reward shift** : reward model becomes unreliable on new policy outputs (out-of-distribution).
  * **Distributed training instabilities** : numerical issues from the multi-model setup compound the usual training instabilities.

This complexity is why DPO became the modern default.

## DPO: the closed-form alternative

The Direct Preference Optimization paper (Rafailov et al, 2023) noticed something elegant: **under the Bradley-Terry assumption that the reward model uses, you can solve for the optimal RLHF policy in closed form**. The optimal policy turns out to be:
    
    
    π*(y | x) ∝ π_ref(y | x) · exp(r(x, y) / β)

So the reward function can be expressed in terms of the policy:
    
    
    r(x, y) = β · log(π*(y | x) / π_ref(y | x)) + Z(x)

where `Z(x)` is a partition function that depends on the prompt but not the response.

Now plug this into the Bradley-Terry preference loss. The partition function cancels (it's the same for both responses to the same prompt):
    
    
    L_DPO = -log σ(β · [log π(y_chosen | x)/π_ref(y_chosen | x) 
                      - log π(y_rejected | x)/π_ref(y_rejected | x)])

That's the entire algorithm. _No reward model. No rollouts. Just preference pairs and a clean loss._

The implementation is short:
    
    
    def dpo_loss(policy, ref, batch, beta=0.1):
        # batch contains: prompt_ids, chosen_ids, rejected_ids (all already tokenized + chat-templated)
    
        # 1. Get log-probs from policy and reference for chosen and rejected.
        #    For each, sum log-probs over the response tokens (NOT the prompt).
        def get_response_logprobs(model, full_ids, response_mask):
            logits = model(full_ids).logits[:, :-1]            # [B, T-1, vocab]
            targets = full_ids[:, 1:]                              # [B, T-1]
            log_probs = logits.log_softmax(-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
            # Mask: only sum over response tokens (response_mask is 1 on response, 0 on prompt)
            return (log_probs * response_mask[:, 1:]).sum(-1)        # [B]
    
        pol_chosen   = get_response_logprobs(policy, batch["chosen_ids"],   batch["chosen_mask"])
        pol_rejected = get_response_logprobs(policy, batch["rejected_ids"], batch["rejected_mask"])
        with torch.no_grad():
            ref_chosen   = get_response_logprobs(ref, batch["chosen_ids"],   batch["chosen_mask"])
            ref_rejected = get_response_logprobs(ref, batch["rejected_ids"], batch["rejected_mask"])
    
        # 2. The DPO loss.
        pi_logratios = pol_chosen   - pol_rejected
        ref_logratios = ref_chosen  - ref_rejected
        logits = beta * (pi_logratios - ref_logratios)
        loss = -F.logsigmoid(logits).mean()
        return loss

Two models, one loss. The optimizer is plain AdamW. The training infrastructure is whatever you already use for SFT — no special rollout system, no four-model coordination, no KL coefficient tuning. **Engineering effort is roughly the same as SFT.**

Quality: DPO matches PPO on most benchmarks. PPO sometimes still wins on hard tasks (complex reasoning, long-form generation, tasks requiring exploration) — frontier labs like Anthropic and OpenAI still use PPO-family methods for the last few percent. But for the vast majority of teams: DPO is the right answer.

## The variants: KTO, IPO, ORPO, GRPO

DPO triggered a Cambrian explosion of preference-learning algorithms. A quick guided tour:

Post-training algorithm landscape, 2026 Algorithm| What's different| When to use  
---|---|---  
SFT| Cross-entropy on responses| Mandatory step 1 of post-training  
PPO + RM| Classic RLHF, four models| Frontier labs; tasks needing exploration; last 1-2% quality  
DPO| Closed-form over preference pairs| **Modern default for most teams**  
IPO| DPO with squared loss instead of log-sigmoid; more robust to reward overfitting| When DPO overfits or your preference data has noise  
KTO| Works on individual ratings (good/bad), not pairs| When you have thumbs-up/down data, not pairwise comparisons  
ORPO| Combines SFT and preference learning in one stage; no separate reference| When you have preference data and want fast iteration  
GRPO| Group-relative scoring — no reward model, no value model. Used for DeepSeek-R1.| Reasoning tasks where you can verify answers programmatically  
RLAIF| Reward signal comes from another LLM, not humans| When human labels are expensive; standard for scaling preference data  
  
Some quick notes:

  * **IPO** (Identity Preference Optimization) is DPO with a squared loss instead of log-sigmoid. The DPO loss can saturate when one preference is overwhelming, leading the model to push margins arbitrarily wide. IPO's squared loss is bounded, more robust. _If DPO overfits, try IPO before tuning hyperparameters_.
  * **KTO** (Kahneman-Tversky Optimization) is the algorithm to use when you have _unary_ feedback — thumbs up / thumbs down on individual responses, not pairs. Real production systems often have unary data (user clicks the thumbs-down button) but no pairs. KTO uses prospect theory's loss function on these.
  * **ORPO** (Odds Ratio Preference Optimization) folds SFT and preference learning into one loss, eliminating the separate SFT stage. Simpler pipeline; quality typically a small step below DPO.
  * **GRPO** (Group Relative Policy Optimization) is what DeepSeek used for R1's reasoning training. _No reward model_ : the reward is computed directly from program-verified correctness (math, code, formal logic). The "group" is multiple sampled responses to the same prompt; advantages are computed relative to the group mean. **This is the new frontier for verifiable-reward tasks**.
  * **RLAIF** (Reinforcement Learning from AI Feedback) replaces human labelers with another LLM. Standard practice for scaling preference data — the model judging is often a frontier model (GPT-4, Claude, Gemini) and the model being trained is smaller. Quality of preference data depends on the judge's quality.

## The data side: where preferences come from

Post-training is downstream of data, and post-training data is _expensive_. Three main sources:

  1. **Human annotators** : humans rank responses or pick the better one. Expensive (~$10-50/hour-equivalent), slow, but high quality if the annotators are well-calibrated. The original RLHF datasets (Anthropic's HH-RLHF, OpenAI's WebGPT) used this. _Annotator instruction quality dominates label quality_ — vague instructions give noisy labels.
  2. **AI feedback (RLAIF)** : another LLM provides preference labels. Cheap (~1000× cheaper than humans), fast, scales to millions of labels. Quality depends on the judge model's quality on the task. Standard for the bulk of preference data in modern post-training; humans used for calibration and on hard cases.
  3. **Constitutional AI / synthetic preferences** : rather than collecting preferences directly, generate them programmatically — train the model to follow a "constitution" of principles, with the AI generating critiques and revisions. Used heavily by Anthropic. Combines well with RLAIF.

For a typical post-training run: SFT on ~100K-1M examples; DPO/PPO on ~10K-100K preference pairs. Each preference pair is "this prompt, response A, response B, label." The cost gap between humans and AI feedback is what made DPO/PPO at scale practical — without RLAIF, the data costs would dominate.

## The current production reality (2026)

What's actually happening at production teams in 2026:

  * **Most teams** : SFT → DPO. Engineering effort similar to two SFT runs. Quality competitive on benchmarks. Standard libraries (TRL, Axolotl, OpenRLHF) make this almost as straightforward as fine-tuning.
  * **Frontier labs** : SFT → RM → PPO with elaborate scaling, plus iterated rounds (re-collect preferences from the new model, re-train RM, re-run PPO). Some have moved to Iterated DPO or constitutional approaches.
  * **Reasoning models** (DeepSeek-R1, OpenAI o3, etc): SFT → GRPO with programmatic rewards. The reward signal is "did the answer match the ground truth?" — no preference data needed for the math/code domains. Mixed with traditional RLHF for non-verifiable tasks.
  * **Open-source community** : predominantly DPO with publicly available preference datasets (Anthropic HH-RLHF, UltraFeedback, etc). The recipe is well-understood; the bottleneck is dataset quality.

The trend: **simpler algorithms winning, with clever loss functions and data engineering doing the work that PPO's complexity used to do**. The history of post-training algorithms looks a lot like the history of optimization algorithms — initial complexity (PPO), realization that the complexity was solving the wrong problem (the RM was the bottleneck), simpler closed-form alternatives (DPO), then specialization (KTO, GRPO).

#### Q&A; — About post-training **Q:** Why does SFT need loss-masking on the prompt? **A:** Two reasons. (1) We want the model to learn "given this instruction, produce a good response," not "given this prefix of an instruction, predict the next token of the instruction." Loss on prompt tokens trains the model to predict prompts, which it doesn't need to do (the user provides them). (2) Without masking, prompt tokens dominate the loss because they're typically longer than responses. The gradient signal gets drowned out by the prompt's perplexity, slowing response-quality improvement. **Q:** Why is the reward model initialized from the SFT model rather than the base model or a smaller model? **A:** Three reasons. (1) The SFT model already understands the chat format, instruction-following, and the general response distribution — the RM needs that context to score responses meaningfully. (2) Same architecture means the RM and policy speak the same "embedding language," reducing distribution shift during PPO. (3) Practical: same tokenizer, same special tokens, same training infrastructure — no porting overhead. _Smaller RMs do exist_ (a 7B RM judging a 70B policy) for compute reasons, but quality typically suffers. The trade is engineering simplicity vs RM accuracy. **Q:** Why does DPO need a frozen reference model? Couldn't you just train the policy alone? **A:** The reference is the regularization. Without it, the DPO loss would push the policy to assign 100% probability to chosen responses and 0% to rejected — a degenerate solution that loses the SFT model's useful priors. The reference model anchors the policy: "stay close to what the SFT model already knows; only adjust where the preferences indicate." The β parameter trades off how much the policy can deviate. _β=0 collapses to the degenerate solution; β→∞ reduces to no learning at all_. Typical values are 0.1-0.5. **Q:** When does PPO actually beat DPO? **A:** Three regimes where PPO can win meaningfully: (1) **Tasks needing exploration** — math reasoning, code, complex multi-step planning. PPO can sample many responses and learn from the best ones; DPO only learns from the chosen/rejected pairs you provide. (2) **When you have a strong reward model** — if the RM captures human preferences well, PPO can climb that signal more aggressively than DPO can extract from preferences. (3) **Long-form generation** — RLHF's per-token KL gives more granular control than DPO's sequence-level loss. _For most chat fine-tuning, the gap is <2% on benchmarks_ — within the noise of post-training variance. **Q:** Why do reasoning models (R1, o3) use GRPO instead of DPO? **A:** GRPO doesn't need preference data. For verifiable tasks (math, code, formal logic), the "reward" is just "did this answer match?" — a programmatic check. GRPO can sample 16-64 responses per prompt, score them all programmatically, and update the policy toward the higher-scoring ones. _No human labels, no AI judges, no reward model_. Quality is bounded by the verifier's accuracy. For tasks without programmatic verification (creative writing, ethics, persona), GRPO doesn't apply — back to DPO/PPO. **Q:** What's "iterated" DPO/PPO? **A:** Run the post-training pipeline once to get version 1. Use version 1 to generate new responses to prompts; collect new preferences (often via RLAIF) on those responses; retrain. Iterate. Each round, the policy improves and the preference data freshens — keeping the signal aligned with the policy's current outputs. Llama-3, Llama-3.1, and most modern open-source post-training uses 3-6 iterations of DPO. _Iteration is where most of the modern quality gains come from_. **Q:** How do safety / refusal training fit in? **A:** Two main approaches. (1) **Safety SFT** : include explicit refusal examples in the SFT data (e.g., "I can't help with that") for prohibited categories. Trains the chat template's refusal pattern. (2) **Safety preferences** : include preference pairs where harmful responses are rejected and helpful refusals are chosen. Both DPO and PPO can absorb this signal. Real safety training is more elaborate (red-teaming, adversarial preference data, multi-turn jailbreak resistance) and is typically a separate post-training stage rather than a fold-in. 

## Picking your method: a practical decision tree

Which method to use, by situation Your situation| Recommended method  
---|---  
First post-training run; medium-quality model| SFT → DPO  
You have human-labeled preference pairs (~10K+)| SFT → DPO  
You have thumbs-up/down data, not pairs| SFT → KTO  
You want simpler pipeline at small quality cost| ORPO (combines SFT + preferences)  
Reasoning task with programmatic verification| SFT → GRPO  
Want last 1-2% quality, have engineering capacity| SFT → RM → PPO with iteration  
Limited preference data; want to scale| RLAIF (AI judges) → DPO  
Preference data is noisy| SFT → IPO (more robust than DPO)  
  
Default for most situations: **SFT followed by DPO with a few iterations**. Starts close to PPO's ceiling at a fraction of the engineering cost. Scale to PPO only when the engineering investment pays for itself in measurable quality.

## Code Magnets: implement the DPO loss

You're writing the DPO loss. Three magnets are wrong choices.

Arrange the magnets into a working DPO loss.

def dpo_loss(policy, ref, batch, beta=0.1): pol_chosen = get_response_logprobs(policy, batch["chosen_ids"]) pol_rejected = get_response_logprobs(policy, batch["rejected_ids"]) with torch.no_grad(): ref_chosen = get_response_logprobs(ref, batch["chosen_ids"]) ref_rejected = get_response_logprobs(ref, batch["rejected_ids"]) ref_chosen = get_response_logprobs(ref, batch["chosen_ids"]) pi_logratios = pol_chosen - pol_rejected ref_logratios = ref_chosen - ref_rejected logits = beta * (pi_logratios - ref_logratios) logits = beta * pi_logratios return -F.logsigmoid(logits).mean() return F.cross_entropy(logits).mean()

show solution
    
    
    def dpo_loss(policy, ref, batch, beta=0.1):
        pol_chosen   = get_response_logprobs(policy, batch["chosen_ids"])
        pol_rejected = get_response_logprobs(policy, batch["rejected_ids"])
        with torch.no_grad():
            ref_chosen   = get_response_logprobs(ref, batch["chosen_ids"])
            ref_rejected = get_response_logprobs(ref, batch["rejected_ids"])
        pi_logratios  = pol_chosen - pol_rejected
        ref_logratios = ref_chosen - ref_rejected
        logits = beta * (pi_logratios - ref_logratios)
        return -F.logsigmoid(logits).mean()

The traps:

  * `ref_chosen = get_response_logprobs(ref, batch["chosen_ids"])` outside the `torch.no_grad()` block: the reference model should never receive gradients (it's frozen). Computing reference log-probs without no_grad allocates gradient memory and adds the reference model to the autograd graph — wastes memory and produces misleading gradient norms.
  * `logits = beta * pi_logratios` (missing the ref subtraction): without subtracting `ref_logratios`, the loss is just maximum-likelihood on chosen responses with no regularization toward the reference. The policy will collapse to assigning probability 1 to chosen responses and ignore the SFT model's distribution. _The reference subtraction is what makes DPO an RLHF method, not just a preferred-response classifier_.
  * `return F.cross_entropy(logits).mean()`: cross-entropy needs class indices and isn't applicable here. The DPO loss is binary (preferred vs rejected) and uses log-sigmoid: `-log(σ(logits))`. Cross-entropy on a single scalar wouldn't typecheck and conceptually is the wrong loss family.

The pattern: **compute log-probs for chosen and rejected from both policy and reference, subtract reference from policy log-ratios, scale by β, take negative log-sigmoid**. The reference subtraction is the critical step that makes DPO faithful to the RLHF objective.

## Who does what?

Match each post-training concept to its real role.

Concept

Real role

SFT

A. Cross-entropy on instruction-response pairs with prompt-tokens loss-masked.

Reward Model (Bradley-Terry)

B. Scalar judge trained on preference pairs to score responses; `-log σ(r_chosen − r_rejected)`.

PPO with KL penalty

C. RL fine-tunes the policy to maximize reward while staying close to the SFT reference.

DPO

D. Closed-form RLHF; trains directly on preference pairs with policy + frozen reference.

Reference model

E. Frozen SFT checkpoint; provides the regularization anchor for both PPO and DPO.

Trainer-rollout architecture

F. Separates inference (vLLM workers) from training (gradient updates) for PPO at scale.

GRPO

G. RL with programmatic rewards; group-relative advantages; used by DeepSeek-R1.

show solution

**SFT** → A  
**Reward Model (Bradley-Terry)** → B  
**PPO with KL penalty** → C  
**DPO** → D  
**Reference model** → E  
**Trainer-rollout architecture** → F  
**GRPO** → G 

The mental shortcut: _SFT teaches instructions, RM scores them, PPO climbs the reward, DPO short-circuits the climb, the reference anchors the policy, trainer-rollout is the production architecture, GRPO uses verified rewards_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team has 50K preference pairs from human annotators and runs DPO. The model overfits — eval reward goes up, eval generation quality goes down. What's likely going on, and what would you try?

show answer

The DPO loss is asymmetric in a way that can drive overfitting. When the policy strongly prefers chosen over rejected, the loss saturates (log-sigmoid approaches 0), and the gradient still pushes the margin wider — even when the policy is already correct. On finite preference data, this leads the policy to memorize specific (chosen, rejected) preferences without generalizing.

Fixes in order of typical efficacy: (1) **Switch to IPO** — squared loss is bounded, doesn't saturate, more robust to overfitting. (2) **Increase β** — keeps the policy closer to the reference, reduces over-optimization. Try β=0.3-0.5 instead of the default 0.1. (3) **Add SFT loss back in** — many implementations add a small SFT term on the chosen response (~0.1× weight) to keep the policy near the SFT model's response distribution. (4) **Lower learning rate** — DPO is sensitive; LRs are typically 5e-7 to 5e-6, much lower than SFT.

_Diagnostic_ : monitor the per-batch margin `(pol_chosen - pol_rejected) - (ref_chosen - ref_rejected)`. If margins are growing rapidly past ~5-10, overfitting is likely.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why is the reward model initialized from the SFT model (not the base model), but in DPO the reference model is also the SFT model? Aren't these doing the same thing?

show answer

They serve completely different functions despite both being SFT-derived.

**Reward model** : replaces the LM head with a scalar value head and is trained on preference pairs to produce a number per response. Its job is to _judge_ responses. It's modified during reward modeling (the value head learns; sometimes the transformer body fine-tunes too).

**Reference model in DPO** : same architecture as the policy, frozen, used to compute reference log-probs for the policy's outputs. Its job is to _regularize_ the policy — the loss pushes the policy to deviate from the reference only where preferences indicate. Never modified.

So: SFT model is the seed for both, but they take very different paths after. The reward model becomes a scoring function; the reference becomes an immutable anchor. Both initialized from SFT for the same reason — same response distribution — but their roles are orthogonal.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** Why does PPO need a separate value model in addition to the reward model? What does the value model predict?

show answer

The value model predicts _expected future reward_ from a partial response — given a prompt and the first `k` tokens of a response, what total reward will the full response get? This is used for **variance reduction** in the policy gradient.

The naive policy gradient uses raw rewards as the signal: "if response got reward 7.2, push log-prob of these tokens up by 7.2." This is high-variance — the same response can get very different rewards on similar prompts, and the gradient noise drowns out the signal. The value model produces a baseline: "this prompt typically yields reward 6.5; this response was 0.7 above baseline." Use the _advantage_ (reward − value) as the signal instead of the raw reward. Variance drops; learning is much more stable.

This is why PPO has 4 models: policy (the thing being trained), reference (KL anchor), reward model (provides the reward), value model (provides the baseline for variance reduction). Without the value model, PPO's gradient noise overwhelms the signal at the scales we care about. _The value model is the reason PPO has 4× the memory cost of DPO_.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** A team's reasoning model fine-tuned with DPO does well on chat benchmarks but plateaus on math problems. They're considering switching to GRPO. What's the math-specific advantage GRPO would give them, and what's the catch?

show answer

**Advantage** : For math problems, you can verify answers programmatically — sympy can check algebraic equivalence; numeric answers can be matched to ground truth. GRPO uses this directly: sample 16-64 responses per problem, score each by program verification (correct/incorrect), advantage = score - mean(group). The policy learns to produce more responses in the high-reward (correct) group. _No preference data needed for the math domain_ — the verifier replaces both the reward model and human labels. This is exactly the setup where DeepSeek-R1 achieved its frontier reasoning quality.

For math, GRPO has two specific advantages over DPO: (1) **Exploration** : sampling many responses per prompt and learning from the rare correct ones lets the model discover reasoning chains that aren't in any preference data. DPO can only learn from pairs you provide. (2) **Verifier-tight reward** : programmatic verification is sharper than learned preferences — no reward hacking, no annotator noise.

**Catches** : (1) GRPO only applies to verifiable tasks. Switching to GRPO for math doesn't help with chat quality; you'd run GRPO on a math/code mixture and DPO on chat preferences in parallel or sequentially. (2) You need a reliable verifier — for non-trivial math problems, building the verifier is its own engineering project. (3) The "exploration via group sampling" only helps if the base model has some chance of solving problems correctly; below that threshold, the gradient signal is too sparse. Practical recipe used by DeepSeek-R1: SFT on reasoning traces first to get the base capability, then GRPO to push further.

### What just happened?

  * Post-training takes a base model (good at next-token prediction) and turns it into something useful (instruction-following chat assistant). Three goals: instruction following, helpfulness, refusal.
  * The classic four-stage pipeline: **pretrain → SFT → reward model → PPO**. The modern shortcut: **pretrain → SFT → DPO**.
  * **SFT** : cross-entropy on instruction-response pairs with prompt-tokens loss-masked. Same architecture, lower LR (1e-5 to 5e-6), 1-3 epochs. Dataset quality dominates everything.
  * **Reward modeling** : replace LM head with scalar value head, train on preference pairs via Bradley-Terry: `-log σ(r_chosen - r_rejected)`. Initialize from SFT.
  * **PPO** : maximize reward minus β × KL(policy ‖ ref). Four models in memory (policy, reference, reward, value). ~5× the base model memory at 70B. Engineering nightmare: rollout-trainer architecture, KL coefficient tuning, reward hacking, KL collapse/explosion, mode collapse.
  * **DPO** : closed-form alternative. Under Bradley-Terry, the optimal RLHF policy can be expressed in closed form, giving a direct loss on preference pairs: `-log σ(β · [(log π/π_ref)_chosen − (log π/π_ref)_rejected])`. Two models in memory; standard distributed training infrastructure.
  * **The trainer-rollout architecture** : vLLM-style inference workers generate responses, separate trainer process updates the policy. Required for PPO at scale; not needed for DPO.
  * **Variants** : IPO (squared loss, more robust), KTO (works on unary good/bad signals), ORPO (combines SFT + preferences), GRPO (programmatic rewards, used by DeepSeek-R1), RLAIF (AI judges replace humans for preference data).
  * **Reward hacking** is the central failure mode of PPO — the policy exploits any flaw in the RM. Length-padding, sycophancy, repetition. The reason for the KL penalty.
  * **The reference model** is frozen SFT in both PPO and DPO. It's the regularization anchor — without it, the policy collapses to degenerate solutions.
  * **Iterated** DPO/PPO: re-collect preferences from the new model, retrain. 3-6 iterations is standard. Most modern quality gains come from iteration.
  * **Data sources** : humans (~$10-50/hour, high quality), AI feedback (RLAIF, ~1000× cheaper, scales), constitutional AI (synthetic from principles).
  * Practical recipe: **most teams use SFT → DPO with a few iterations**. Frontier labs still use PPO for the last 1-2%. Reasoning models use GRPO. The trend is simpler algorithms with better data engineering.
  * The reflex: when picking a post-training method, ask "what kind of feedback signal do I have?" — pairs (DPO/PPO), unary (KTO), programmatic (GRPO), or principles (CAI). Match the algorithm to the data shape.

Module 32 takes the diagnostic skill that's been scattered across many modules and consolidates it into a single playbook: how to debug a training run that's silently going wrong. NaN walking, loss curve forensics, gradient norm tracking, the four-layer failure taxonomy. After that, M33 explores Mamba and state-space models as the parallel-architecture-world that uses the same kernel patterns as transformers.
