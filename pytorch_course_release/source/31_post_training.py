#!/usr/bin/env python3
"""Module 31: Post-training (SFT, RM, PPO, DPO) — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part IX · Module 31</div>
  <h1 class="module-title"><em>Post-training:</em> SFT, reward models, PPO &amp; DPO</h1>
  <p class="module-sub">— how a base model becomes a useful chat model, why PPO needs four models in memory, why DPO replaced it for most teams, and the variants (KTO, ORPO, GRPO) that solve the data and infrastructure problems</p>
</div>

<p>For 30 modules we've covered training a base model — the pretraining run. But a base model trained to predict the next token isn't directly usable as a chat assistant. Ask it "What's the capital of France?" and it might continue with another question instead of answering. It hasn't learned to follow instructions, refuse harmful requests, or distinguish good answers from bad ones. <strong>Post-training</strong> is the phase that closes this gap.</p>

<p>This module covers the four-stage RLHF pipeline (Supervised Fine-Tuning → Reward Modeling → RLHF), the modern simplification (Direct Preference Optimization replacing the last two stages), the variants competing for the post-training spotlight, and the distributed-rollout engineering that production setups require. By the end you'll know which method to pick for your situation, what each costs in compute and engineering effort, and why the field went from "PPO is the answer" in 2022 to "DPO and friends" in 2024-2026.</p>

<div class="keyidea">
The classic RLHF pipeline has four stages: <strong>(1) pretrain</strong> the base model on next-token prediction; <strong>(2) SFT</strong> on instruction-response pairs (loss-masked on the prompt); <strong>(3) train a reward model</strong> on preference pairs via Bradley-Terry; <strong>(4) RL fine-tune</strong> the policy with PPO against the reward, with a KL penalty against the SFT model. <strong>DPO</strong> short-circuits this: under a Bradley-Terry assumption, the optimal RLHF policy has a closed form involving only the preference data, eliminating stages 3 and 4. PPO needs <em>four models in memory</em> (policy, ref, reward, value) and elaborate trainer-rollout infrastructure; DPO needs <em>two models</em> (policy + frozen reference) and a single trainer. <strong>DPO became the default for most teams in 2024-2025</strong> because the engineering simplification is enormous and the quality is comparable on most benchmarks.
</div>

<h2>Two new faces</h2>

<div class="character" style="--c: #b85a6c;">
  <div class="avatar" style="background: #b85a6c; color: #fff;">RM</div>
  <div>
    <p class="who">Reward Model</p>
    <p class="name">"I'm a scalar judge. Given a prompt and a response, I produce one number — how much someone preferred this response."</p>
    <p class="says">I'm initialized from the SFT model, then trained on pairs of (chosen, rejected) responses to predict which one humans (or AI labelers) preferred. My loss is <code>-log σ(r(chosen) - r(rejected))</code> — Bradley-Terry. I'm the same shape as the policy I'll judge, with one new linear head producing a scalar instead of vocab logits. <strong>I'm the bottleneck of classic RLHF</strong>: my quality caps the policy's quality, my biases become the policy's biases, and reward-hacking against me is what makes PPO unstable. The shift to DPO eliminated me entirely — and many teams haven't missed me.</p>
  </div>
</div>

<div class="character" style="--c: #1f5f5b;">
  <div class="avatar" style="background: #1f5f5b; color: #fff;">DPO</div>
  <div>
    <p class="who">DPO</p>
    <p class="name">"I'm the closed-form RLHF. Skip the reward model entirely; train directly on preference pairs."</p>
    <p class="says">My loss is a single line: <code>-log σ(β · (log π(chosen) - log π_ref(chosen)) - β · (log π(rejected) - log π_ref(rejected)))</code>. Just policy log-probs, reference log-probs, and the chosen/rejected pair. <em>That's the entire algorithm.</em> No reward model to train, no rollouts to run, no PPO instabilities. Two models in memory (policy + frozen reference). I'm derived from the same Bradley-Terry assumption as the reward model — my math says: <em>under that assumption, the optimal RLHF policy can be expressed in closed form, and that closed form gives this loss</em>. The engineering simplification is enormous and the quality is competitive. <strong>Most teams in 2026 use me or one of my variants.</strong></p>
  </div>
</div>

<h2>The base-model-to-chat-model gap</h2>

<p>A pretrained model is good at one thing: predicting the next token in text that looks like its training distribution. Chat assistants need three properties the base model lacks:</p>

<ol>
  <li><strong>Instruction following</strong>: the model should respond to "What is X?" with an explanation, not another question or a continuation in the same style.</li>
  <li><strong>Helpful and stylistic alignment</strong>: responses should be useful, well-structured, in the right register. The base model has no reason to be helpful; it just continues plausible text.</li>
  <li><strong>Refusal of harmful requests</strong>: the model should decline to produce dangerous content. The base model will happily continue any prompt it sees.</li>
</ol>

<p>Closing each gap requires its own training signal. Instruction following needs examples of instructions and good responses (SFT). Helpfulness needs preference data — humans (or AIs) judging which responses are better. Refusal needs deliberate red-teaming and refusal training.</p>

<h2>The pipeline, with the modern shortcut</h2>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 380" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrP" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
    <marker id="arrPdpo" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#c1502e"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Post-training: classic RLHF (top) vs DPO shortcut (bottom)</text>

  <!-- Stage 1: Pretrain -->
  <g transform="translate(20, 60)">
    <rect x="0" y="0" width="120" height="60" fill="#fff8a8" stroke="#1a1612" stroke-width="2"/>
    <text x="60" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">① Pretrain</text>
    <text x="60" y="38" text-anchor="middle" font-size="9" fill="#1a1612">next-token prediction</text>
    <text x="60" y="50" text-anchor="middle" font-size="9" fill="#1a1612">on web-scale text</text>
  </g>

  <path d="M 145 90 L 175 90" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrP)"/>

  <!-- Stage 2: SFT -->
  <g transform="translate(180, 60)">
    <rect x="0" y="0" width="120" height="60" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="60" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">② SFT</text>
    <text x="60" y="38" text-anchor="middle" font-size="9" fill="#1a1612">instruction-response</text>
    <text x="60" y="50" text-anchor="middle" font-size="9" fill="#1a1612">pairs, prompt-masked</text>
  </g>

  <!-- Branch: classic vs DPO -->
  <path d="M 305 90 L 335 90" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrP)"/>

  <!-- Stage 3: Reward Model (classic only) -->
  <g transform="translate(340, 30)">
    <rect x="0" y="0" width="120" height="60" fill="#ffd5dc" stroke="#b85a6c" stroke-width="2"/>
    <text x="60" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">③ Reward Model</text>
    <text x="60" y="38" text-anchor="middle" font-size="9" fill="#1a1612">Bradley-Terry on</text>
    <text x="60" y="50" text-anchor="middle" font-size="9" fill="#1a1612">preference pairs</text>
  </g>

  <path d="M 465 60 L 495 60" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrP)"/>

  <!-- Stage 4: PPO (classic) -->
  <g transform="translate(500, 30)">
    <rect x="0" y="0" width="220" height="60" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="110" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">④ PPO RL fine-tune</text>
    <text x="110" y="38" text-anchor="middle" font-size="9" fill="#1a1612">policy + ref + RM + value</text>
    <text x="110" y="50" text-anchor="middle" font-size="9" fill="#c1502e">— FOUR models in memory —</text>
  </g>

  <!-- DPO shortcut path -->
  <path d="M 305 100 L 305 200 L 500 200 L 500 175" stroke="#c1502e" stroke-width="2" fill="none" stroke-dasharray="6 4" marker-end="url(#arrPdpo)"/>
  <text x="375" y="195" font-family="'Caveat', cursive" font-size="20" fill="#c1502e">DPO short-circuit ↗</text>

  <!-- DPO box -->
  <g transform="translate(440, 175)">
    <rect x="0" y="0" width="280" height="60" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="3"/>
    <text x="140" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1f5f5b">③+④: DPO</text>
    <text x="140" y="38" text-anchor="middle" font-size="9" fill="#1a1612">single loss on policy + frozen reference,</text>
    <text x="140" y="50" text-anchor="middle" font-size="9" fill="#1f5f5b">— TWO models in memory —</text>
  </g>

  <!-- Bottom annotations: cost / time -->
  <text x="20" y="280" font-size="12" font-weight="700" fill="#1a1612">Engineering cost (rough order-of-magnitude):</text>
  <text x="40" y="298" font-size="10" fill="#1a1612">  Classic PPO: weeks to months. 4 models. Distributed rollouts. KL coefficient tuning. Reward-hacking detection.</text>
  <text x="40" y="313" font-size="10" fill="#1f5f5b">  DPO:         days. 2 models. Standard distributed training infra. Lower variance.</text>

  <text x="370" y="345" font-family="'Caveat', cursive" font-size="22" fill="#c1502e" text-anchor="middle">DPO won't always match PPO peak quality, but the engineering gap is huge</text>
  <text x="370" y="370" font-family="'Caveat', cursive" font-size="20" fill="#1f5f5b" text-anchor="middle">in 2026: most teams DPO; frontier labs still PPO for the last 1-2% quality</text>
</svg>
</div>

<p>The diagram shows both paths. Three checkpoints flow: from pretrain to SFT (where instruction-following is taught), then either down to reward model + PPO (classic, more powerful, more expensive) or directly to DPO (modern shortcut, simpler, fast). The choice is mostly about engineering capacity and the value of the last few percent of quality.</p>

<h2>Stage 1: Pretrain (briefly)</h2>

<p>Already covered through M29. The base model is a transformer trained on next-token prediction on internet-scale text. By the time post-training begins, the base model has the world knowledge — it just can't access it usefully.</p>

<h2>Stage 2: Supervised Fine-Tuning (SFT)</h2>

<p>SFT is exactly the same loss as pretraining (cross-entropy on next-token prediction) with two changes:</p>

<ol>
  <li><strong>Different data</strong>: instruction-response pairs formatted with a chat template, instead of raw web text. Examples: <code>{"messages": [{"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris."}]}</code>.</li>
  <li><strong>Loss masking on the prompt</strong>: only compute loss on the <em>response</em> tokens, not the prompt tokens. We don't want the model learning to "predict" instructions; we want it learning to follow them and produce good responses.</li>
</ol>

<pre><code><span class="kw">def</span> <span class="fn">sft_loss</span>(model, batch):
    <span class="com"># batch contains: input_ids, labels (where labels=-100 means "ignore" — applied to prompt tokens)</span>
    logits = <span class="fn">model</span>(batch[<span class="str">"input_ids"</span>])              <span class="com"># [B, T, vocab]</span>

    <span class="com"># Shift: predict token t from positions 0..t-1</span>
    shift_logits = logits[..., :-<span class="num">1</span>, :].<span class="fn">contiguous</span>()
    shift_labels = batch[<span class="str">"labels"</span>][..., <span class="num">1</span>:].<span class="fn">contiguous</span>()

    <span class="com"># Cross-entropy ignores positions where label = -100 (prompt tokens)</span>
    <span class="kw">return</span> F.<span class="fn">cross_entropy</span>(
        shift_logits.<span class="fn">view</span>(-<span class="num">1</span>, shift_logits.<span class="fn">size</span>(-<span class="num">1</span>)).<span class="fn">float</span>(),
        shift_labels.<span class="fn">view</span>(-<span class="num">1</span>),
        ignore_index=-<span class="num">100</span>,
    )</code></pre>

<p>That's it — the rest is data engineering. The model architecture is unchanged; the optimizer is AdamW (M9) with a much lower learning rate than pretraining (typically 1e-5 to 5e-6); the schedule is cosine with brief warmup. SFT runs for 1-3 epochs on the curated dataset; longer than 3 epochs typically overfits.</p>

<p><strong>Dataset quality dominates everything in SFT</strong>. A small, carefully curated dataset (1K-10K examples) can outperform a large, noisy one (100K+ examples). Critical observations from the LIMA paper, Alpaca, and downstream work:</p>

<ul>
  <li><strong>Diverse instructions matter more than volume</strong>. 1K examples covering many task types beats 100K examples that are all variants of the same task.</li>
  <li><strong>Response quality is non-negotiable</strong>. If the SFT data has bad responses (incorrect, poorly formatted, in the wrong tone), the model learns those patterns.</li>
  <li><strong>Filter aggressively for length, refusals, and tone consistency</strong>. SFT is when stylistic patterns get baked in.</li>
</ul>

<p>The chat template is also fixed at SFT time. Llama-3's chat template uses special tokens like <code>&lt;|begin_of_text|&gt;</code>, <code>&lt;|start_header_id|&gt;</code>, <code>&lt;|end_header_id|&gt;</code>, <code>&lt;|eot_id|&gt;</code>. Once the model learns this format during SFT, deviating at inference time will break it.</p>

<h2>Stage 3: Reward Modeling</h2>

<p>The reward model takes a prompt and a response and outputs a scalar — how good the response is for that prompt. The training signal comes from <strong>preference pairs</strong>: pairs of responses to the same prompt where one was judged better than the other.</p>

<p>The standard loss is <strong>Bradley-Terry</strong> — a classical model from psychometrics for ranking from pairwise comparisons:</p>

<pre><code>P(chosen &gt; rejected | prompt) = σ(r(prompt, chosen) - r(prompt, rejected))

L_RM = -E[log σ(r(prompt, chosen) - r(prompt, rejected))]</code></pre>

<p>Train the reward model to maximize this. The reward model architecture: take the SFT model, replace the LM head (vocab projection) with a single linear layer producing a scalar. Initialize this new head, freeze nothing. Train end-to-end on preference pairs.</p>

<pre><code><span class="kw">class</span> <span class="ty">RewardModel</span>(nn.Module):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, base_model):
        <span class="fn">super</span>().<span class="fn">__init__</span>()
        <span class="com"># Same transformer as base model, but with scalar head instead of LM head</span>
        self.transformer = base_model.transformer
        self.value_head  = nn.<span class="fn">Linear</span>(base_model.cfg.d_model, <span class="num">1</span>, bias=<span class="kw">False</span>)

    <span class="kw">def</span> <span class="fn">forward</span>(self, input_ids):
        <span class="com"># Run the transformer; take the hidden state at the LAST token</span>
        hidden = self.<span class="fn">transformer</span>(input_ids)         <span class="com"># [B, T, D]</span>
        last_hidden = hidden[:, -<span class="num">1</span>, :]                 <span class="com"># [B, D]</span>
        <span class="com"># Project to a scalar reward</span>
        reward = self.<span class="fn">value_head</span>(last_hidden).<span class="fn">squeeze</span>(-<span class="num">1</span>)  <span class="com"># [B]</span>
        <span class="kw">return</span> reward

<span class="kw">def</span> <span class="fn">rm_loss</span>(rm, chosen_ids, rejected_ids):
    r_chosen   = <span class="fn">rm</span>(chosen_ids)                       <span class="com"># [B]</span>
    r_rejected = <span class="fn">rm</span>(rejected_ids)                     <span class="com"># [B]</span>
    <span class="kw">return</span> -F.<span class="fn">logsigmoid</span>(r_chosen - r_rejected).<span class="fn">mean</span>()</code></pre>

<p>The reward at the last token is the reward for the entire sequence. Training takes the SFT model's compute budget (similar dataset size, similar number of steps). Output: a model the same size as the policy that scores responses.</p>

<p>Two practical issues with reward models:</p>

<ul>
  <li><strong>Reward hacking</strong>: the policy will exploit any flaw in the reward model. If the RM rewards length, the policy generates verbose responses. If the RM is over-confident on a domain, the policy gravitates there. Whatever pattern the RM rewards, the policy amplifies. <strong>This is the central failure mode of PPO</strong>.</li>
  <li><strong>Distribution shift</strong>: the RM was trained on responses from the SFT model. As the policy improves during PPO, its outputs drift away from the RM's training distribution. The RM becomes less reliable on the new outputs — and the policy is now optimizing against a less reliable signal.</li>
</ul>

<h2>Stage 4: PPO — the engineering nightmare</h2>

<p>Now the policy is trained to maximize reward via PPO (Proximal Policy Optimization). This is where post-training stops being deep learning and starts being reinforcement learning — with all the instabilities that entails.</p>

<p>The objective:</p>

<pre><code>L_PPO = E[r(prompt, response)]                       <span class="com"># maximize reward</span>
        - β · KL(policy || ref_policy)              <span class="com"># stay close to SFT model</span>

  where:
    response   ~ policy(· | prompt)                  <span class="com"># sampled from current policy</span>
    r(...)     = reward_model(prompt, response)
    KL(...)    = sum_t log(policy(token_t | ...) / ref_policy(token_t | ...))</code></pre>

<p>The KL penalty is critical. Without it, the policy drifts arbitrarily far from the SFT model — exploiting the reward model and producing degenerate outputs ("repeat 'helpful' 1000 times"). The KL keeps the policy close to the SFT distribution; the reward signal moves it gently in the helpful direction.</p>

<p><strong>Four models live in memory simultaneously during PPO:</strong></p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 360" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">PPO has four models in memory; DPO has two — the engineering gap is enormous</text>

  <!-- PPO panel -->
  <g transform="translate(20, 50)">
    <rect x="0" y="0" width="340" height="270" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="170" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">PPO — four models</text>
    <text x="170" y="38" text-anchor="middle" font-size="10" fill="#c1502e">~5× the base model's memory at 70B</text>

    <rect x="20" y="55" width="140" height="55" fill="#fff" stroke="#c1502e" stroke-width="1.5"/>
    <text x="90" y="76" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">① Policy</text>
    <text x="90" y="92" text-anchor="middle" font-size="9" fill="#1a1612">trainable, full bf16</text>
    <text x="90" y="104" text-anchor="middle" font-size="9" fill="#1a1612">+ optimizer state (4×)</text>

    <rect x="180" y="55" width="140" height="55" fill="#fff" stroke="#c1502e" stroke-width="1.5"/>
    <text x="250" y="76" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">② Reference</text>
    <text x="250" y="92" text-anchor="middle" font-size="9" fill="#1a1612">frozen, bf16</text>
    <text x="250" y="104" text-anchor="middle" font-size="9" fill="#1a1612">SFT checkpoint</text>

    <rect x="20" y="125" width="140" height="55" fill="#fff" stroke="#c1502e" stroke-width="1.5"/>
    <text x="90" y="146" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">③ Reward Model</text>
    <text x="90" y="162" text-anchor="middle" font-size="9" fill="#1a1612">frozen, bf16</text>
    <text x="90" y="174" text-anchor="middle" font-size="9" fill="#1a1612">scores responses</text>

    <rect x="180" y="125" width="140" height="55" fill="#fff" stroke="#c1502e" stroke-width="1.5"/>
    <text x="250" y="146" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">④ Value</text>
    <text x="250" y="162" text-anchor="middle" font-size="9" fill="#1a1612">trainable, bf16</text>
    <text x="250" y="174" text-anchor="middle" font-size="9" fill="#1a1612">predicts future reward</text>

    <text x="170" y="210" text-anchor="middle" font-size="11" font-weight="700" fill="#c1502e">For 70B base in bf16:</text>
    <text x="170" y="226" text-anchor="middle" font-size="10" fill="#1a1612">policy: 140 GB + 280 GB optimizer = 420 GB</text>
    <text x="170" y="240" text-anchor="middle" font-size="10" fill="#1a1612">reference: 140 GB</text>
    <text x="170" y="254" text-anchor="middle" font-size="10" fill="#1a1612">RM: 140 GB · value: 140 GB + 280 GB optim = 420 GB</text>
    <text x="170" y="266" text-anchor="middle" font-size="10" font-weight="700" fill="#c1502e">≈ 1.1 TB across all GPUs</text>
  </g>

  <!-- DPO panel -->
  <g transform="translate(380, 50)">
    <rect x="0" y="0" width="340" height="270" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="170" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">DPO — two models</text>
    <text x="170" y="38" text-anchor="middle" font-size="10" fill="#1f5f5b">~3× the base model's memory at 70B</text>

    <rect x="20" y="55" width="140" height="55" fill="#fff" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="90" y="76" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">① Policy</text>
    <text x="90" y="92" text-anchor="middle" font-size="9" fill="#1a1612">trainable, full bf16</text>
    <text x="90" y="104" text-anchor="middle" font-size="9" fill="#1a1612">+ optimizer state (4×)</text>

    <rect x="180" y="55" width="140" height="55" fill="#fff" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="250" y="76" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">② Reference</text>
    <text x="250" y="92" text-anchor="middle" font-size="9" fill="#1a1612">frozen, bf16</text>
    <text x="250" y="104" text-anchor="middle" font-size="9" fill="#1a1612">SFT checkpoint</text>

    <text x="170" y="160" text-anchor="middle" font-size="11" font-weight="700" fill="#1f5f5b">For 70B base in bf16:</text>
    <text x="170" y="180" text-anchor="middle" font-size="10" fill="#1a1612">policy: 140 GB + 280 GB optim = 420 GB</text>
    <text x="170" y="194" text-anchor="middle" font-size="10" fill="#1a1612">reference: 140 GB</text>
    <text x="170" y="220" text-anchor="middle" font-size="10" font-weight="700" fill="#1f5f5b">≈ 560 GB total</text>

    <text x="170" y="248" text-anchor="middle" font-size="10" fill="#1a1612">No rollouts. No reward model.</text>
    <text x="170" y="262" text-anchor="middle" font-size="10" fill="#1a1612">Just standard distributed training.</text>
  </g>

  <text x="370" y="345" font-family="'Caveat', cursive" font-size="22" fill="#c1502e" text-anchor="middle">PPO needs ~2× the GPUs of DPO for the same model — and the rollout infrastructure</text>
</svg>
</div>

<p>Four models, plus their optimizer states for the trainable ones. For a 70B base model:</p>

<ul>
  <li><strong>Policy</strong>: 70B params × 2 bytes (bf16) = 140 GB. Plus AdamW optimizer state (M12): 4× model size in fp32 master weights + first/second moments = 280 GB. Subtotal: 420 GB.</li>
  <li><strong>Reference</strong> (frozen SFT checkpoint): 140 GB.</li>
  <li><strong>Reward Model</strong> (frozen): 140 GB.</li>
  <li><strong>Value Model</strong> (trainable, used to predict future reward for variance reduction): 420 GB with optimizer.</li>
</ul>

<p>Total: ~1.1 TB across the cluster. With FSDP (M17) and quantization (M26), this can be reduced to ~700 GB, but you still need the GPUs to host four model copies.</p>

<p><strong>The trainer-rollout architecture</strong> is what makes PPO actually work at scale. The naive approach — generate responses on the trainer, then update the policy — wastes the GPU. Generation is decode-bound (M27); training is compute-bound. Mixing them gives terrible utilization.</p>

<p>Modern PPO setups separate the two:</p>

<ol>
  <li><strong>Rollout workers</strong> (typically vLLM instances): hold a copy of the current policy in inference mode (paged KV, continuous batching). Generate batches of responses to prompts.</li>
  <li><strong>Trainer</strong>: takes the (prompt, response, reward) tuples and updates the policy via PPO. After every K updates, broadcasts the new weights to the rollout workers.</li>
</ol>

<p>The two run on different GPUs (or at least different processes), with the rollout queue providing a buffer between them. <strong>This architecture is what TRL (Hugging Face's RL library), OpenRLHF, NeMo-RL, and most production setups implement.</strong> Without it, PPO at scale is effectively impossible — you'd spend 80% of your compute on inference inside the trainer process.</p>

<p>The PPO step itself, simplified:</p>

<pre><code><span class="kw">def</span> <span class="fn">ppo_step</span>(policy, ref, value, rm, prompts):
    <span class="com"># 1. Rollout: generate responses with the current policy</span>
    responses = policy.<span class="fn">generate</span>(prompts)            <span class="com"># done by rollout workers in production</span>

    <span class="com"># 2. Compute rewards</span>
    rewards = <span class="fn">rm</span>(prompts, responses)                <span class="com"># scalar per response</span>

    <span class="com"># 3. Compute log-probs from policy and reference</span>
    <span class="kw">with</span> torch.<span class="fn">no_grad</span>():
        ref_logprobs = <span class="fn">ref</span>(prompts, responses).<span class="fn">log_softmax</span>(-<span class="num">1</span>).<span class="fn">gather</span>(...)
    pol_logprobs = <span class="fn">policy</span>(prompts, responses).<span class="fn">log_softmax</span>(-<span class="num">1</span>).<span class="fn">gather</span>(...)

    <span class="com"># 4. Compute the per-token KL penalty</span>
    kl_penalty = pol_logprobs - ref_logprobs

    <span class="com"># 5. Adjust rewards: reward at last token + per-token KL penalty</span>
    shaped_rewards = rewards.<span class="fn">unsqueeze</span>(-<span class="num">1</span>) - beta * kl_penalty

    <span class="com"># 6. Compute advantages via GAE (generalized advantage estimation), using the value model</span>
    values = <span class="fn">value</span>(prompts, responses)
    advantages = <span class="fn">compute_gae</span>(shaped_rewards, values, gamma=<span class="num">0.99</span>, lam=<span class="num">0.95</span>)

    <span class="com"># 7. PPO clipped objective</span>
    ratio = (pol_logprobs - pol_logprobs_old).<span class="fn">exp</span>()
    pg_loss = -torch.<span class="fn">min</span>(ratio * advantages, ratio.<span class="fn">clamp</span>(<span class="num">1</span>-eps, <span class="num">1</span>+eps) * advantages).<span class="fn">mean</span>()

    <span class="com"># 8. Value loss: predict observed returns</span>
    v_loss = (values - returns).<span class="fn">pow</span>(<span class="num">2</span>).<span class="fn">mean</span>()

    <span class="kw">return</span> pg_loss + v_coef * v_loss</code></pre>

<p>Even simplified, this is a lot. Real PPO setups have additional components: gradient accumulation across rollout batches, careful KL coefficient adaptation (the KL coefficient should auto-adjust to keep KL ≈ target), reward whitening, advantage normalization, response-length penalties to prevent length-hacking, reward-clipping. <em>Each of these has its own production failure mode</em>.</p>

<p>The list of things that can go wrong:</p>

<ul>
  <li><strong>Reward hacking</strong>: policy finds a degenerate way to score high on the reward model that humans wouldn't endorse. Repetitive outputs, sycophancy, length-padding.</li>
  <li><strong>KL collapse</strong>: KL penalty too low, policy drifts too far from reference, outputs become incoherent.</li>
  <li><strong>KL explosion</strong>: KL penalty too high, policy doesn't learn anything new.</li>
  <li><strong>Mode collapse</strong>: policy converges to a single response style.</li>
  <li><strong>Reward shift</strong>: reward model becomes unreliable on new policy outputs (out-of-distribution).</li>
  <li><strong>Distributed training instabilities</strong>: numerical issues from the multi-model setup compound the usual training instabilities.</li>
</ul>

<p>This complexity is why DPO became the modern default.</p>

<h2>DPO: the closed-form alternative</h2>

<p>The Direct Preference Optimization paper (Rafailov et al, 2023) noticed something elegant: <strong>under the Bradley-Terry assumption that the reward model uses, you can solve for the optimal RLHF policy in closed form</strong>. The optimal policy turns out to be:</p>

<pre><code>π*(y | x) ∝ π_ref(y | x) · exp(r(x, y) / β)</code></pre>

<p>So the reward function can be expressed in terms of the policy:</p>

<pre><code>r(x, y) = β · log(π*(y | x) / π_ref(y | x)) + Z(x)</code></pre>

<p>where <code>Z(x)</code> is a partition function that depends on the prompt but not the response.</p>

<p>Now plug this into the Bradley-Terry preference loss. The partition function cancels (it's the same for both responses to the same prompt):</p>

<pre><code>L_DPO = -log σ(β · [log π(y_chosen | x)/π_ref(y_chosen | x) 
                  - log π(y_rejected | x)/π_ref(y_rejected | x)])</code></pre>

<p>That's the entire algorithm. <em>No reward model. No rollouts. Just preference pairs and a clean loss.</em></p>

<p>The implementation is short:</p>

<pre><code><span class="kw">def</span> <span class="fn">dpo_loss</span>(policy, ref, batch, beta=<span class="num">0.1</span>):
    <span class="com"># batch contains: prompt_ids, chosen_ids, rejected_ids (all already tokenized + chat-templated)</span>

    <span class="com"># 1. Get log-probs from policy and reference for chosen and rejected.</span>
    <span class="com">#    For each, sum log-probs over the response tokens (NOT the prompt).</span>
    <span class="kw">def</span> <span class="fn">get_response_logprobs</span>(model, full_ids, response_mask):
        logits = <span class="fn">model</span>(full_ids).logits[:, :-<span class="num">1</span>]            <span class="com"># [B, T-1, vocab]</span>
        targets = full_ids[:, <span class="num">1</span>:]                              <span class="com"># [B, T-1]</span>
        log_probs = logits.<span class="fn">log_softmax</span>(-<span class="num">1</span>).<span class="fn">gather</span>(<span class="num">2</span>, targets.<span class="fn">unsqueeze</span>(-<span class="num">1</span>)).<span class="fn">squeeze</span>(-<span class="num">1</span>)
        <span class="com"># Mask: only sum over response tokens (response_mask is 1 on response, 0 on prompt)</span>
        <span class="kw">return</span> (log_probs * response_mask[:, <span class="num">1</span>:]).<span class="fn">sum</span>(-<span class="num">1</span>)        <span class="com"># [B]</span>

    pol_chosen   = <span class="fn">get_response_logprobs</span>(policy, batch[<span class="str">"chosen_ids"</span>],   batch[<span class="str">"chosen_mask"</span>])
    pol_rejected = <span class="fn">get_response_logprobs</span>(policy, batch[<span class="str">"rejected_ids"</span>], batch[<span class="str">"rejected_mask"</span>])
    <span class="kw">with</span> torch.<span class="fn">no_grad</span>():
        ref_chosen   = <span class="fn">get_response_logprobs</span>(ref, batch[<span class="str">"chosen_ids"</span>],   batch[<span class="str">"chosen_mask"</span>])
        ref_rejected = <span class="fn">get_response_logprobs</span>(ref, batch[<span class="str">"rejected_ids"</span>], batch[<span class="str">"rejected_mask"</span>])

    <span class="com"># 2. The DPO loss.</span>
    pi_logratios = pol_chosen   - pol_rejected
    ref_logratios = ref_chosen  - ref_rejected
    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.<span class="fn">logsigmoid</span>(logits).<span class="fn">mean</span>()
    <span class="kw">return</span> loss</code></pre>

<p>Two models, one loss. The optimizer is plain AdamW. The training infrastructure is whatever you already use for SFT — no special rollout system, no four-model coordination, no KL coefficient tuning. <strong>Engineering effort is roughly the same as SFT.</strong></p>

<p>Quality: DPO matches PPO on most benchmarks. PPO sometimes still wins on hard tasks (complex reasoning, long-form generation, tasks requiring exploration) — frontier labs like Anthropic and OpenAI still use PPO-family methods for the last few percent. But for the vast majority of teams: DPO is the right answer.</p>

<h2>The variants: KTO, IPO, ORPO, GRPO</h2>

<p>DPO triggered a Cambrian explosion of preference-learning algorithms. A quick guided tour:</p>

<div class="table-wrap">
<table>
<caption>Post-training algorithm landscape, 2026</caption>
<thead><tr><th>Algorithm</th><th>What's different</th><th>When to use</th></tr></thead>
<tbody>
<tr><td>SFT</td><td>Cross-entropy on responses</td><td>Mandatory step 1 of post-training</td></tr>
<tr><td>PPO + RM</td><td>Classic RLHF, four models</td><td>Frontier labs; tasks needing exploration; last 1-2% quality</td></tr>
<tr><td>DPO</td><td>Closed-form over preference pairs</td><td><strong>Modern default for most teams</strong></td></tr>
<tr><td>IPO</td><td>DPO with squared loss instead of log-sigmoid; more robust to reward overfitting</td><td>When DPO overfits or your preference data has noise</td></tr>
<tr><td>KTO</td><td>Works on individual ratings (good/bad), not pairs</td><td>When you have thumbs-up/down data, not pairwise comparisons</td></tr>
<tr><td>ORPO</td><td>Combines SFT and preference learning in one stage; no separate reference</td><td>When you have preference data and want fast iteration</td></tr>
<tr><td>GRPO</td><td>Group-relative scoring — no reward model, no value model. Used for DeepSeek-R1.</td><td>Reasoning tasks where you can verify answers programmatically</td></tr>
<tr><td>RLAIF</td><td>Reward signal comes from another LLM, not humans</td><td>When human labels are expensive; standard for scaling preference data</td></tr>
</tbody>
</table>
</div>

<p>Some quick notes:</p>

<ul>
  <li><strong>IPO</strong> (Identity Preference Optimization) is DPO with a squared loss instead of log-sigmoid. The DPO loss can saturate when one preference is overwhelming, leading the model to push margins arbitrarily wide. IPO's squared loss is bounded, more robust. <em>If DPO overfits, try IPO before tuning hyperparameters</em>.</li>
  <li><strong>KTO</strong> (Kahneman-Tversky Optimization) is the algorithm to use when you have <em>unary</em> feedback — thumbs up / thumbs down on individual responses, not pairs. Real production systems often have unary data (user clicks the thumbs-down button) but no pairs. KTO uses prospect theory's loss function on these.</li>
  <li><strong>ORPO</strong> (Odds Ratio Preference Optimization) folds SFT and preference learning into one loss, eliminating the separate SFT stage. Simpler pipeline; quality typically a small step below DPO.</li>
  <li><strong>GRPO</strong> (Group Relative Policy Optimization) is what DeepSeek used for R1's reasoning training. <em>No reward model</em>: the reward is computed directly from program-verified correctness (math, code, formal logic). The "group" is multiple sampled responses to the same prompt; advantages are computed relative to the group mean. <strong>This is the new frontier for verifiable-reward tasks</strong>.</li>
  <li><strong>RLAIF</strong> (Reinforcement Learning from AI Feedback) replaces human labelers with another LLM. Standard practice for scaling preference data — the model judging is often a frontier model (GPT-4, Claude, Gemini) and the model being trained is smaller. Quality of preference data depends on the judge's quality.</li>
</ul>

<h2>The data side: where preferences come from</h2>

<p>Post-training is downstream of data, and post-training data is <em>expensive</em>. Three main sources:</p>

<ol>
  <li><strong>Human annotators</strong>: humans rank responses or pick the better one. Expensive (~$10-50/hour-equivalent), slow, but high quality if the annotators are well-calibrated. The original RLHF datasets (Anthropic's HH-RLHF, OpenAI's WebGPT) used this. <em>Annotator instruction quality dominates label quality</em> — vague instructions give noisy labels.</li>
  <li><strong>AI feedback (RLAIF)</strong>: another LLM provides preference labels. Cheap (~1000× cheaper than humans), fast, scales to millions of labels. Quality depends on the judge model's quality on the task. Standard for the bulk of preference data in modern post-training; humans used for calibration and on hard cases.</li>
  <li><strong>Constitutional AI / synthetic preferences</strong>: rather than collecting preferences directly, generate them programmatically — train the model to follow a "constitution" of principles, with the AI generating critiques and revisions. Used heavily by Anthropic. Combines well with RLAIF.</li>
</ol>

<p>For a typical post-training run: SFT on ~100K-1M examples; DPO/PPO on ~10K-100K preference pairs. Each preference pair is "this prompt, response A, response B, label." The cost gap between humans and AI feedback is what made DPO/PPO at scale practical — without RLAIF, the data costs would dominate.</p>

<h2>The current production reality (2026)</h2>

<p>What's actually happening at production teams in 2026:</p>

<ul>
  <li><strong>Most teams</strong>: SFT → DPO. Engineering effort similar to two SFT runs. Quality competitive on benchmarks. Standard libraries (TRL, Axolotl, OpenRLHF) make this almost as straightforward as fine-tuning.</li>
  <li><strong>Frontier labs</strong>: SFT → RM → PPO with elaborate scaling, plus iterated rounds (re-collect preferences from the new model, re-train RM, re-run PPO). Some have moved to Iterated DPO or constitutional approaches.</li>
  <li><strong>Reasoning models</strong> (DeepSeek-R1, OpenAI o3, etc): SFT → GRPO with programmatic rewards. The reward signal is "did the answer match the ground truth?" — no preference data needed for the math/code domains. Mixed with traditional RLHF for non-verifiable tasks.</li>
  <li><strong>Open-source community</strong>: predominantly DPO with publicly available preference datasets (Anthropic HH-RLHF, UltraFeedback, etc). The recipe is well-understood; the bottleneck is dataset quality.</li>
</ul>

<p>The trend: <strong>simpler algorithms winning, with clever loss functions and data engineering doing the work that PPO's complexity used to do</strong>. The history of post-training algorithms looks a lot like the history of optimization algorithms — initial complexity (PPO), realization that the complexity was solving the wrong problem (the RM was the bottleneck), simpler closed-form alternatives (DPO), then specialization (KTO, GRPO).</p>

<div class="ndq">
<h4>About post-training</h4>

<p class="q">Why does SFT need loss-masking on the prompt?</p>
<p class="a">Two reasons. (1) We want the model to learn "given this instruction, produce a good response," not "given this prefix of an instruction, predict the next token of the instruction." Loss on prompt tokens trains the model to predict prompts, which it doesn't need to do (the user provides them). (2) Without masking, prompt tokens dominate the loss because they're typically longer than responses. The gradient signal gets drowned out by the prompt's perplexity, slowing response-quality improvement.</p>

<p class="q">Why is the reward model initialized from the SFT model rather than the base model or a smaller model?</p>
<p class="a">Three reasons. (1) The SFT model already understands the chat format, instruction-following, and the general response distribution — the RM needs that context to score responses meaningfully. (2) Same architecture means the RM and policy speak the same "embedding language," reducing distribution shift during PPO. (3) Practical: same tokenizer, same special tokens, same training infrastructure — no porting overhead. <em>Smaller RMs do exist</em> (a 7B RM judging a 70B policy) for compute reasons, but quality typically suffers. The trade is engineering simplicity vs RM accuracy.</p>

<p class="q">Why does DPO need a frozen reference model? Couldn't you just train the policy alone?</p>
<p class="a">The reference is the regularization. Without it, the DPO loss would push the policy to assign 100% probability to chosen responses and 0% to rejected — a degenerate solution that loses the SFT model's useful priors. The reference model anchors the policy: "stay close to what the SFT model already knows; only adjust where the preferences indicate." The β parameter trades off how much the policy can deviate. <em>β=0 collapses to the degenerate solution; β→∞ reduces to no learning at all</em>. Typical values are 0.1-0.5.</p>

<p class="q">When does PPO actually beat DPO?</p>
<p class="a">Three regimes where PPO can win meaningfully: (1) <strong>Tasks needing exploration</strong> — math reasoning, code, complex multi-step planning. PPO can sample many responses and learn from the best ones; DPO only learns from the chosen/rejected pairs you provide. (2) <strong>When you have a strong reward model</strong> — if the RM captures human preferences well, PPO can climb that signal more aggressively than DPO can extract from preferences. (3) <strong>Long-form generation</strong> — RLHF's per-token KL gives more granular control than DPO's sequence-level loss. <em>For most chat fine-tuning, the gap is &lt;2% on benchmarks</em> — within the noise of post-training variance.</p>

<p class="q">Why do reasoning models (R1, o3) use GRPO instead of DPO?</p>
<p class="a">GRPO doesn't need preference data. For verifiable tasks (math, code, formal logic), the "reward" is just "did this answer match?" — a programmatic check. GRPO can sample 16-64 responses per prompt, score them all programmatically, and update the policy toward the higher-scoring ones. <em>No human labels, no AI judges, no reward model</em>. Quality is bounded by the verifier's accuracy. For tasks without programmatic verification (creative writing, ethics, persona), GRPO doesn't apply — back to DPO/PPO.</p>

<p class="q">What's "iterated" DPO/PPO?</p>
<p class="a">Run the post-training pipeline once to get version 1. Use version 1 to generate new responses to prompts; collect new preferences (often via RLAIF) on those responses; retrain. Iterate. Each round, the policy improves and the preference data freshens — keeping the signal aligned with the policy's current outputs. Llama-3, Llama-3.1, and most modern open-source post-training uses 3-6 iterations of DPO. <em>Iteration is where most of the modern quality gains come from</em>.</p>

<p class="q">How do safety / refusal training fit in?</p>
<p class="a">Two main approaches. (1) <strong>Safety SFT</strong>: include explicit refusal examples in the SFT data (e.g., "I can't help with that") for prohibited categories. Trains the chat template's refusal pattern. (2) <strong>Safety preferences</strong>: include preference pairs where harmful responses are rejected and helpful refusals are chosen. Both DPO and PPO can absorb this signal. Real safety training is more elaborate (red-teaming, adversarial preference data, multi-turn jailbreak resistance) and is typically a separate post-training stage rather than a fold-in.</p>
</div>

<h2>Picking your method: a practical decision tree</h2>

<div class="table-wrap">
<table>
<caption>Which method to use, by situation</caption>
<thead><tr><th>Your situation</th><th>Recommended method</th></tr></thead>
<tbody>
<tr><td>First post-training run; medium-quality model</td><td>SFT → DPO</td></tr>
<tr><td>You have human-labeled preference pairs (~10K+)</td><td>SFT → DPO</td></tr>
<tr><td>You have thumbs-up/down data, not pairs</td><td>SFT → KTO</td></tr>
<tr><td>You want simpler pipeline at small quality cost</td><td>ORPO (combines SFT + preferences)</td></tr>
<tr><td>Reasoning task with programmatic verification</td><td>SFT → GRPO</td></tr>
<tr><td>Want last 1-2% quality, have engineering capacity</td><td>SFT → RM → PPO with iteration</td></tr>
<tr><td>Limited preference data; want to scale</td><td>RLAIF (AI judges) → DPO</td></tr>
<tr><td>Preference data is noisy</td><td>SFT → IPO (more robust than DPO)</td></tr>
</tbody>
</table>
</div>

<p>Default for most situations: <strong>SFT followed by DPO with a few iterations</strong>. Starts close to PPO's ceiling at a fraction of the engineering cost. Scale to PPO only when the engineering investment pays for itself in measurable quality.</p>

<h2>Code Magnets: implement the DPO loss</h2>

<p>You're writing the DPO loss. Three magnets are wrong choices.</p>

<div class="magnets">
<p>Arrange the magnets into a working DPO loss.</p>

<div class="magnet-pool">
  <span class="magnet">def dpo_loss(policy, ref, batch, beta=0.1):</span>
  <span class="magnet">    pol_chosen   = get_response_logprobs(policy, batch["chosen_ids"])</span>
  <span class="magnet">    pol_rejected = get_response_logprobs(policy, batch["rejected_ids"])</span>
  <span class="magnet">    with torch.no_grad():</span>
  <span class="magnet">        ref_chosen   = get_response_logprobs(ref, batch["chosen_ids"])</span>
  <span class="magnet">        ref_rejected = get_response_logprobs(ref, batch["rejected_ids"])</span>
  <span class="magnet">    ref_chosen   = get_response_logprobs(ref, batch["chosen_ids"])</span>
  <span class="magnet">    pi_logratios  = pol_chosen - pol_rejected</span>
  <span class="magnet">    ref_logratios = ref_chosen - ref_rejected</span>
  <span class="magnet">    logits = beta * (pi_logratios - ref_logratios)</span>
  <span class="magnet">    logits = beta * pi_logratios</span>
  <span class="magnet">    return -F.logsigmoid(logits).mean()</span>
  <span class="magnet">    return F.cross_entropy(logits).mean()</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">def</span> <span class="fn">dpo_loss</span>(policy, ref, batch, beta=<span class="num">0.1</span>):
    pol_chosen   = <span class="fn">get_response_logprobs</span>(policy, batch[<span class="str">"chosen_ids"</span>])
    pol_rejected = <span class="fn">get_response_logprobs</span>(policy, batch[<span class="str">"rejected_ids"</span>])
    <span class="kw">with</span> torch.<span class="fn">no_grad</span>():
        ref_chosen   = <span class="fn">get_response_logprobs</span>(ref, batch[<span class="str">"chosen_ids"</span>])
        ref_rejected = <span class="fn">get_response_logprobs</span>(ref, batch[<span class="str">"rejected_ids"</span>])
    pi_logratios  = pol_chosen - pol_rejected
    ref_logratios = ref_chosen - ref_rejected
    logits = beta * (pi_logratios - ref_logratios)
    <span class="kw">return</span> -F.<span class="fn">logsigmoid</span>(logits).<span class="fn">mean</span>()</code></pre>
<p>The traps:</p>
<ul>
  <li><code>ref_chosen = get_response_logprobs(ref, batch["chosen_ids"])</code> outside the <code>torch.no_grad()</code> block: the reference model should never receive gradients (it's frozen). Computing reference log-probs without no_grad allocates gradient memory and adds the reference model to the autograd graph — wastes memory and produces misleading gradient norms.</li>
  <li><code>logits = beta * pi_logratios</code> (missing the ref subtraction): without subtracting <code>ref_logratios</code>, the loss is just maximum-likelihood on chosen responses with no regularization toward the reference. The policy will collapse to assigning probability 1 to chosen responses and ignore the SFT model's distribution. <em>The reference subtraction is what makes DPO an RLHF method, not just a preferred-response classifier</em>.</li>
  <li><code>return F.cross_entropy(logits).mean()</code>: cross-entropy needs class indices and isn't applicable here. The DPO loss is binary (preferred vs rejected) and uses log-sigmoid: <code>-log(σ(logits))</code>. Cross-entropy on a single scalar wouldn't typecheck and conceptually is the wrong loss family.</li>
</ul>
<p>The pattern: <strong>compute log-probs for chosen and rejected from both policy and reference, subtract reference from policy log-ratios, scale by β, take negative log-sigmoid</strong>. The reference subtraction is the critical step that makes DPO faithful to the RLHF objective.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each post-training concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>SFT</div>
  <div>A. Cross-entropy on instruction-response pairs with prompt-tokens loss-masked.</div>

  <div>Reward Model (Bradley-Terry)</div>
  <div>B. Scalar judge trained on preference pairs to score responses; <code>-log σ(r_chosen − r_rejected)</code>.</div>

  <div>PPO with KL penalty</div>
  <div>C. RL fine-tunes the policy to maximize reward while staying close to the SFT reference.</div>

  <div>DPO</div>
  <div>D. Closed-form RLHF; trains directly on preference pairs with policy + frozen reference.</div>

  <div>Reference model</div>
  <div>E. Frozen SFT checkpoint; provides the regularization anchor for both PPO and DPO.</div>

  <div>Trainer-rollout architecture</div>
  <div>F. Separates inference (vLLM workers) from training (gradient updates) for PPO at scale.</div>

  <div>GRPO</div>
  <div>G. RL with programmatic rewards; group-relative advantages; used by DeepSeek-R1.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>SFT</strong> → A<br>
<strong>Reward Model (Bradley-Terry)</strong> → B<br>
<strong>PPO with KL penalty</strong> → C<br>
<strong>DPO</strong> → D<br>
<strong>Reference model</strong> → E<br>
<strong>Trainer-rollout architecture</strong> → F<br>
<strong>GRPO</strong> → G
</p>
<p>The mental shortcut: <em>SFT teaches instructions, RM scores them, PPO climbs the reward, DPO short-circuits the climb, the reference anchors the policy, trainer-rollout is the production architecture, GRPO uses verified rewards</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team has 50K preference pairs from human annotators and runs DPO. The model overfits — eval reward goes up, eval generation quality goes down. What's likely going on, and what would you try?</p>
<details class="answer"><summary>show answer</summary>
<p>The DPO loss is asymmetric in a way that can drive overfitting. When the policy strongly prefers chosen over rejected, the loss saturates (log-sigmoid approaches 0), and the gradient still pushes the margin wider — even when the policy is already correct. On finite preference data, this leads the policy to memorize specific (chosen, rejected) preferences without generalizing.</p>
<p>Fixes in order of typical efficacy: (1) <strong>Switch to IPO</strong> — squared loss is bounded, doesn't saturate, more robust to overfitting. (2) <strong>Increase β</strong> — keeps the policy closer to the reference, reduces over-optimization. Try β=0.3-0.5 instead of the default 0.1. (3) <strong>Add SFT loss back in</strong> — many implementations add a small SFT term on the chosen response (~0.1× weight) to keep the policy near the SFT model's response distribution. (4) <strong>Lower learning rate</strong> — DPO is sensitive; LRs are typically 5e-7 to 5e-6, much lower than SFT.</p>
<p><em>Diagnostic</em>: monitor the per-batch margin <code>(pol_chosen - pol_rejected) - (ref_chosen - ref_rejected)</code>. If margins are growing rapidly past ~5-10, overfitting is likely.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Why is the reward model initialized from the SFT model (not the base model), but in DPO the reference model is also the SFT model? Aren't these doing the same thing?</p>
<details class="answer"><summary>show answer</summary>
<p>They serve completely different functions despite both being SFT-derived.</p>
<p><strong>Reward model</strong>: replaces the LM head with a scalar value head and is trained on preference pairs to produce a number per response. Its job is to <em>judge</em> responses. It's modified during reward modeling (the value head learns; sometimes the transformer body fine-tunes too).</p>
<p><strong>Reference model in DPO</strong>: same architecture as the policy, frozen, used to compute reference log-probs for the policy's outputs. Its job is to <em>regularize</em> the policy — the loss pushes the policy to deviate from the reference only where preferences indicate. Never modified.</p>
<p>So: SFT model is the seed for both, but they take very different paths after. The reward model becomes a scoring function; the reference becomes an immutable anchor. Both initialized from SFT for the same reason — same response distribution — but their roles are orthogonal.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> Why does PPO need a separate value model in addition to the reward model? What does the value model predict?</p>
<details class="answer"><summary>show answer</summary>
<p>The value model predicts <em>expected future reward</em> from a partial response — given a prompt and the first <code>k</code> tokens of a response, what total reward will the full response get? This is used for <strong>variance reduction</strong> in the policy gradient.</p>
<p>The naive policy gradient uses raw rewards as the signal: "if response got reward 7.2, push log-prob of these tokens up by 7.2." This is high-variance — the same response can get very different rewards on similar prompts, and the gradient noise drowns out the signal. The value model produces a baseline: "this prompt typically yields reward 6.5; this response was 0.7 above baseline." Use the <em>advantage</em> (reward − value) as the signal instead of the raw reward. Variance drops; learning is much more stable.</p>
<p>This is why PPO has 4 models: policy (the thing being trained), reference (KL anchor), reward model (provides the reward), value model (provides the baseline for variance reduction). Without the value model, PPO's gradient noise overwhelms the signal at the scales we care about. <em>The value model is the reason PPO has 4× the memory cost of DPO</em>.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> A team's reasoning model fine-tuned with DPO does well on chat benchmarks but plateaus on math problems. They're considering switching to GRPO. What's the math-specific advantage GRPO would give them, and what's the catch?</p>
<details class="answer"><summary>show answer</summary>
<p><strong>Advantage</strong>: For math problems, you can verify answers programmatically — sympy can check algebraic equivalence; numeric answers can be matched to ground truth. GRPO uses this directly: sample 16-64 responses per problem, score each by program verification (correct/incorrect), advantage = score - mean(group). The policy learns to produce more responses in the high-reward (correct) group. <em>No preference data needed for the math domain</em> — the verifier replaces both the reward model and human labels. This is exactly the setup where DeepSeek-R1 achieved its frontier reasoning quality.</p>
<p>For math, GRPO has two specific advantages over DPO: (1) <strong>Exploration</strong>: sampling many responses per prompt and learning from the rare correct ones lets the model discover reasoning chains that aren't in any preference data. DPO can only learn from pairs you provide. (2) <strong>Verifier-tight reward</strong>: programmatic verification is sharper than learned preferences — no reward hacking, no annotator noise.</p>
<p><strong>Catches</strong>: (1) GRPO only applies to verifiable tasks. Switching to GRPO for math doesn't help with chat quality; you'd run GRPO on a math/code mixture and DPO on chat preferences in parallel or sequentially. (2) You need a reliable verifier — for non-trivial math problems, building the verifier is its own engineering project. (3) The "exploration via group sampling" only helps if the base model has some chance of solving problems correctly; below that threshold, the gradient signal is too sparse. Practical recipe used by DeepSeek-R1: SFT on reasoning traces first to get the base capability, then GRPO to push further.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>Post-training takes a base model (good at next-token prediction) and turns it into something useful (instruction-following chat assistant). Three goals: instruction following, helpfulness, refusal.</li>
  <li>The classic four-stage pipeline: <strong>pretrain → SFT → reward model → PPO</strong>. The modern shortcut: <strong>pretrain → SFT → DPO</strong>.</li>
  <li><strong>SFT</strong>: cross-entropy on instruction-response pairs with prompt-tokens loss-masked. Same architecture, lower LR (1e-5 to 5e-6), 1-3 epochs. Dataset quality dominates everything.</li>
  <li><strong>Reward modeling</strong>: replace LM head with scalar value head, train on preference pairs via Bradley-Terry: <code>-log σ(r_chosen - r_rejected)</code>. Initialize from SFT.</li>
  <li><strong>PPO</strong>: maximize reward minus β × KL(policy ‖ ref). Four models in memory (policy, reference, reward, value). ~5× the base model memory at 70B. Engineering nightmare: rollout-trainer architecture, KL coefficient tuning, reward hacking, KL collapse/explosion, mode collapse.</li>
  <li><strong>DPO</strong>: closed-form alternative. Under Bradley-Terry, the optimal RLHF policy can be expressed in closed form, giving a direct loss on preference pairs: <code>-log σ(β · [(log π/π_ref)_chosen − (log π/π_ref)_rejected])</code>. Two models in memory; standard distributed training infrastructure.</li>
  <li><strong>The trainer-rollout architecture</strong>: vLLM-style inference workers generate responses, separate trainer process updates the policy. Required for PPO at scale; not needed for DPO.</li>
  <li><strong>Variants</strong>: IPO (squared loss, more robust), KTO (works on unary good/bad signals), ORPO (combines SFT + preferences), GRPO (programmatic rewards, used by DeepSeek-R1), RLAIF (AI judges replace humans for preference data).</li>
  <li><strong>Reward hacking</strong> is the central failure mode of PPO — the policy exploits any flaw in the RM. Length-padding, sycophancy, repetition. The reason for the KL penalty.</li>
  <li><strong>The reference model</strong> is frozen SFT in both PPO and DPO. It's the regularization anchor — without it, the policy collapses to degenerate solutions.</li>
  <li><strong>Iterated</strong> DPO/PPO: re-collect preferences from the new model, retrain. 3-6 iterations is standard. Most modern quality gains come from iteration.</li>
  <li><strong>Data sources</strong>: humans (~$10-50/hour, high quality), AI feedback (RLAIF, ~1000× cheaper, scales), constitutional AI (synthetic from principles).</li>
  <li>Practical recipe: <strong>most teams use SFT → DPO with a few iterations</strong>. Frontier labs still use PPO for the last 1-2%. Reasoning models use GRPO. The trend is simpler algorithms with better data engineering.</li>
  <li>The reflex: when picking a post-training method, ask "what kind of feedback signal do I have?" — pairs (DPO/PPO), unary (KTO), programmatic (GRPO), or principles (CAI). Match the algorithm to the data shape.</li>
</ul>
</div>

<p>Module 32 takes the diagnostic skill that's been scattered across many modules and consolidates it into a single playbook: how to debug a training run that's silently going wrong. NaN walking, loss curve forensics, gradient norm tracking, the four-layer failure taxonomy. After that, M33 explores Mamba and state-space models as the parallel-architecture-world that uses the same kernel patterns as transformers.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">31</span>
  <span>Post-training: SFT, RM, PPO &amp; DPO</span>
</div>
"""

emit("31_post_training", "Module 31 — Post-training: SFT, RM, PPO & DPO", BODY)
