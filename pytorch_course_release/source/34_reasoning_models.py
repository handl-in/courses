#!/usr/bin/env python3
"""Module 34: Reasoning models — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part X · Module 34 · the reasoning frontier</div>
  <h1 class="module-title"><em>Reasoning models:</em> long-thinking, GRPO &amp; test-time compute</h1>
  <p class="module-sub">— how o1- and R1-style models train themselves to reason via verifiable rewards, why GRPO replaced PPO for this purpose, and the test-time-compute architecture that turns "more inference compute" into "better answers"</p>
</div>

<p>Through M33 we've covered standard transformer training and serving. M34 turns to what's arguably the most consequential post-training development of 2024-2025: <strong>reasoning models</strong>. These are models that, when given a hard problem, "think" by generating thousands or tens of thousands of intermediate tokens before producing a short final answer. OpenAI's o1, DeepSeek's R1, and their successors are this shape. The capability gain on math, code, and formal reasoning has been substantial.</p>

<p>The architectural change is roughly zero. A reasoning model's transformer is the same transformer you've built (M29). What's different is two things: <strong>how they're trained</strong> (RL with verifiable rewards, not preference data) and <strong>how they're served</strong> (long generations, test-time compute search). M34 covers both, with a focus on the algorithm — <strong>GRPO</strong> — that made R1's training feasible without needing a reward model. By the end you'll know what's actually happening when "the model is reasoning," what makes it work, and the serving infrastructure that handles 100K-token outputs.</p>

<div class="keyidea">
A reasoning model is a transformer trained to <em>think before answering</em> — produce a long chain of intermediate tokens, then a short final answer. The training algorithm of choice is <strong>GRPO</strong> (Group Relative Policy Optimization): for each prompt, sample K responses (e.g., 16-64), score them with a <strong>programmatic verifier</strong> (math equality, code unit tests, formal proof check), compute advantages relative to the group mean, update via a clipped policy gradient. <strong>No reward model, no value model</strong> — the verifier replaces both. R1 demonstrated that with this setup and enough compute, the model <em>spontaneously learns</em> to produce long chains of thought, including backtracking and error correction. At inference, reasoning models are served with long decode budgets (10K-100K tokens), and additional quality comes from <strong>test-time compute</strong>: best-of-N, majority voting, tree search, and verifier-guided search. The Pareto frontier between inference compute and answer quality is the new dimension along which reasoning systems are tuned.
</div>

<h2>Two new faces — the reasoning pair</h2>

<div class="character" style="--c: #1f5f5b;">
  <div class="avatar" style="background: #1f5f5b; color: #fff;">R</div>
  <div>
    <p class="who">Reasoner</p>
    <p class="name">"I think for 10,000 tokens before I answer. My final answer is short, but the path is what matters."</p>
    <p class="says">When you ask me a hard math problem, I don't just produce the answer. I work through it: try an approach, check if it fits, abandon it if it doesn't, try another. <em>Most of my output is intermediate reasoning</em>; the answer is the last few tokens. I emerged from RL training with a programmatic verifier — when the verifier rewarded correct final answers, I gradually learned that long, careful, self-correcting reasoning produced more correct answers than short guesses. <strong>I wasn't taught to reason; I learned that reasoning works.</strong> My architecture is identical to a normal transformer (M29). The difference is what trained me.</p>
  </div>
</div>

<div class="character" style="--c: #d4a017;">
  <div class="avatar" style="background: #d4a017; color: #fff;">V</div>
  <div>
    <p class="who">Verifier</p>
    <p class="name">"I don't have opinions. I check answers — and I'm always right about what I check."</p>
    <p class="says">For math: I evaluate whether two expressions are equivalent (sympy, numerical comparison). For code: I run unit tests. For formal proofs: I check the proof terms. <em>I produce a single bit per response: correct or incorrect</em> — sometimes a small set of partial-credit signals, but mostly binary. I can't grade essays, evaluate creativity, or judge ethics. But on the tasks I cover, I'm <strong>perfectly aligned</strong> by construction — I don't suffer from the reward-model problems M31 covered (out-of-distribution, hacking, drift). I'm what makes RL-from-scratch on reasoning tasks work without humans in the loop. My existence is also the limit on which tasks reasoning models can train on directly.</p>
  </div>
</div>

<h2>What changed: from preference learning to verifiable rewards</h2>

<p>M31 covered post-training under the assumption that quality is judged by humans (or AI judges trained on human judgment). For most chat tasks — was this response helpful? polite? safe? — that's the only available signal. There's no ground truth for "is this a good response to <em>tell me a joke</em>?"</p>

<p>For some tasks, ground truth exists. <strong>Math problems have correct answers</strong> — sympy can check equivalence. <strong>Code has tests</strong> — run the unit tests. <strong>Formal proofs check syntactically</strong> — Lean or Coq either accept the proof term or don't. For these tasks, the path that classic RLHF takes (collect preferences, train RM, run PPO) is wasteful. <em>Why approximate human judgment of correctness when correctness is directly checkable?</em></p>

<p>This is the insight that made reasoning models possible. With a verifier, the reward signal is exact, free, and not subject to gaming. You can sample millions of responses and grade each in milliseconds. The training-data bottleneck of RLHF (expensive human labels) goes away. The reward-model failure modes (drift, hacking, out-of-distribution) go away. <strong>The remaining question is just: what RL algorithm works well in this regime?</strong></p>

<h2>GRPO: the algorithm that worked</h2>

<p>GRPO — Group Relative Policy Optimization — is the algorithm DeepSeek used for R1. It's a streamlined PPO variant designed specifically for verifiable-reward settings. The key simplifications:</p>

<ul>
  <li><strong>No reward model</strong>: the verifier produces rewards directly.</li>
  <li><strong>No value model</strong>: the baseline for variance reduction comes from the <em>group mean</em> — sample K responses to the same prompt, use the mean as the baseline.</li>
  <li><strong>Just policy + reference</strong>: like DPO (M31), but with sampled rollouts and an explicit RL objective.</li>
</ul>

<p>The result: a clean two-model setup (policy + frozen reference) with the simplicity of DPO and the exploration capability of PPO.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 380" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">GRPO step: sample K, verify each, compute group-relative advantages, update policy</text>

  <!-- Step 1: prompt -->
  <g transform="translate(20, 50)">
    <rect x="0" y="0" width="120" height="50" fill="#fff8a8" stroke="#1a1612" stroke-width="2"/>
    <text x="60" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">prompt</text>
    <text x="60" y="38" text-anchor="middle" font-size="9" fill="#1a1612">"Solve: 2x+5=11"</text>
  </g>
  <path d="M 145 75 L 175 75" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrG)"/>

  <!-- Step 2: K rollouts -->
  <g transform="translate(180, 35)">
    <rect x="0" y="0" width="170" height="80" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="85" y="20" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">K rollouts</text>
    <text x="10" y="38" font-size="9" fill="#1a1612">  r₁: "x=3 ✓ ..." (correct)</text>
    <text x="10" y="51" font-size="9" fill="#1a1612">  r₂: "x=4 ✗ ..." (wrong)</text>
    <text x="10" y="64" font-size="9" fill="#1a1612">  r₃: "x=3 ✓ ..." (correct)</text>
    <text x="10" y="76" font-size="9" fill="#1a1612">  ... (K=16 typical)</text>
  </g>
  <path d="M 355 75 L 385 75" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrG)"/>

  <!-- Step 3: verifier scores -->
  <g transform="translate(390, 35)">
    <rect x="0" y="0" width="160" height="80" fill="#fff5d8" stroke="#d4a017" stroke-width="2"/>
    <text x="80" y="20" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">Verifier scores</text>
    <text x="10" y="40" font-size="9" fill="#1a1612">  R₁ = 1.0  R₂ = 0.0</text>
    <text x="10" y="54" font-size="9" fill="#1a1612">  R₃ = 1.0  R₄ = 0.0</text>
    <text x="10" y="68" font-size="9" fill="#1a1612">  group mean = 0.5</text>
  </g>
  <path d="M 555 75 L 585 75" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrG)"/>

  <!-- Step 4: advantages -->
  <g transform="translate(590, 35)">
    <rect x="0" y="0" width="130" height="80" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="65" y="20" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">Advantages</text>
    <text x="10" y="40" font-size="9" fill="#1a1612">  A_i = R_i − mean</text>
    <text x="10" y="54" font-size="9" fill="#1a1612">  A₁ = +0.5</text>
    <text x="10" y="68" font-size="9" fill="#1a1612">  A₂ = −0.5</text>
  </g>

  <!-- Step 5: policy update -->
  <path d="M 655 120 L 655 150 Q 655 160 645 160 L 95 160 Q 85 160 85 170 L 85 195" 
        stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrG)"/>

  <g transform="translate(20, 200)">
    <rect x="0" y="0" width="700" height="100" fill="#ffd5dc" stroke="#b85a6c" stroke-width="2"/>
    <text x="350" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">Policy update — PPO-style clipped objective with KL to reference</text>
    <text x="20" y="44" font-size="10" font-family="'IBM Plex Mono', monospace" fill="#1a1612">  for each rollout i:</text>
    <text x="40" y="58" font-size="10" font-family="'IBM Plex Mono', monospace" fill="#1a1612">    ratio_i = π(r_i | x) / π_old(r_i | x)        # importance ratio</text>
    <text x="40" y="72" font-size="10" font-family="'IBM Plex Mono', monospace" fill="#1a1612">    surrogate_i = min(ratio_i · A_i, clip(ratio_i, 1−ε, 1+ε) · A_i)</text>
    <text x="40" y="86" font-size="10" font-family="'IBM Plex Mono', monospace" fill="#1a1612">  L = -mean(surrogate) + β · KL(π ‖ π_ref)</text>
  </g>

  <text x="370" y="335" font-family="'Caveat', cursive" font-size="20" fill="#c1502e" text-anchor="middle">no reward model, no value model — verifier replaces both</text>
  <text x="370" y="362" font-family="'Caveat', cursive" font-size="20" fill="#1f5f5b" text-anchor="middle">advantage from group mean = baseline for free</text>
</svg>
</div>

<h3>The GRPO loss in code</h3>

<p>Compare this directly with M31's DPO loss to see the structural similarity (and differences):</p>

<pre><code><span class="kw">def</span> <span class="fn">grpo_loss</span>(policy, ref, prompts, K=<span class="num">16</span>, eps=<span class="num">0.2</span>, beta=<span class="num">0.04</span>):
    <span class="com"># 1. ROLLOUT: sample K responses per prompt from the current policy.</span>
    <span class="com">#    In production, this is done by vLLM workers (M27); here we sketch the logic.</span>
    <span class="kw">with</span> torch.<span class="fn">no_grad</span>():
        responses = []                          <span class="com"># list of K · len(prompts) responses</span>
        old_logprobs = []                       <span class="com"># cached for the importance ratio</span>
        <span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(K):
            r = policy.<span class="fn">generate</span>(prompts, do_sample=<span class="kw">True</span>, temperature=<span class="num">1.0</span>)
            responses.<span class="fn">append</span>(r)
            old_logprobs.<span class="fn">append</span>(<span class="fn">compute_logprobs</span>(policy, prompts, r))

    <span class="com"># 2. VERIFY: programmatic check on each response.</span>
    rewards = torch.<span class="fn">tensor</span>([
        <span class="fn">verify</span>(prompts[i % <span class="fn">len</span>(prompts)], responses[i])    <span class="com"># 0.0 or 1.0 typically</span>
        <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(responses))
    ])

    <span class="com"># 3. ADVANTAGE via group mean — for each prompt's group of K responses,</span>
    <span class="com">#    advantage = reward - group_mean. Optionally normalize by group std.</span>
    rewards = rewards.<span class="fn">view</span>(<span class="fn">len</span>(prompts), K)
    group_mean = rewards.<span class="fn">mean</span>(dim=<span class="num">1</span>, keepdim=<span class="kw">True</span>)
    group_std = rewards.<span class="fn">std</span>(dim=<span class="num">1</span>, keepdim=<span class="kw">True</span>) + <span class="num">1e-8</span>
    advantages = ((rewards - group_mean) / group_std).<span class="fn">flatten</span>()

    <span class="com"># 4. POLICY UPDATE: PPO-style clipped surrogate, with KL penalty against ref.</span>
    new_logprobs = <span class="fn">compute_logprobs</span>(policy, prompts, responses)
    <span class="kw">with</span> torch.<span class="fn">no_grad</span>():
        ref_logprobs = <span class="fn">compute_logprobs</span>(ref, prompts, responses)

    <span class="com"># Importance ratio (per-token, summed over response tokens)</span>
    ratio = (new_logprobs - <span class="fn">torch.cat</span>(old_logprobs)).<span class="fn">exp</span>()

    <span class="com"># Clipped objective — same as PPO from M31</span>
    surrogate1 = ratio * advantages
    surrogate2 = ratio.<span class="fn">clamp</span>(<span class="num">1</span> - eps, <span class="num">1</span> + eps) * advantages
    pg_loss = -torch.<span class="fn">min</span>(surrogate1, surrogate2).<span class="fn">mean</span>()

    <span class="com"># KL penalty to reference: stops policy from drifting too far</span>
    kl = (new_logprobs - ref_logprobs).<span class="fn">mean</span>()

    <span class="kw">return</span> pg_loss + beta * kl</code></pre>

<p>Three things to call out, all setting GRPO apart from the M31 algorithms:</p>

<ul>
  <li><strong>The K-sample rollout</strong>: each prompt produces K responses, all from the current policy. This is where the exploration happens — sampling at temperature 1.0 produces diverse responses; the verifier picks the wheat from the chaff. K=16 is a common starting point; production reasoning runs use K up to 64 or 128.</li>
  <li><strong>The group-mean baseline</strong>: subtracting the group mean from each reward gives a per-rollout advantage. <em>Equivalent to a value-model baseline but free.</em> Normalizing by group std (the "Z-score" form above) further reduces variance and improves training stability — this is one of the small details in production GRPO that makes a real difference.</li>
  <li><strong>The PPO-clipped objective</strong>: GRPO uses PPO's clipped surrogate to bound policy updates. The clip prevents any single sample from causing a large policy change, which would destabilize training.</li>
</ul>

<p>The whole loss is ~25 lines. <strong>The complexity is not in the algorithm; it's in the rollout infrastructure.</strong> Generating K responses per prompt for thousands of prompts per training step is expensive — typically the rollout dominates total training compute, more so than for PPO. Production GRPO setups separate rollout (vLLM workers, M27) from training (the gradient updates) and treat them as a producer-consumer pipeline.</p>

<h2>The chain-of-thought emergence story</h2>

<p>The most striking finding of the R1 paper: <strong>with the GRPO setup above and a base model that's been pretrained on reasoning content, the model spontaneously learns to produce long chains of thought</strong>. Not because anyone trained it to. Because long, careful reasoning produces correct answers more often than short guesses, and the verifier rewards correct answers.</p>

<p>The trajectory R1 reported during training:</p>

<ul>
  <li><strong>Early steps</strong>: model produces short answers, often wrong. Verifier rewards few. Gradients push the policy toward whatever responses got the rewards.</li>
  <li><strong>Mid-training</strong>: model starts producing intermediate reasoning steps. Average response length grows from a few hundred tokens to a few thousand. Accuracy on the math benchmarks climbs.</li>
  <li><strong>Later</strong>: <em>the model learns backtracking</em>. Spontaneous "wait, let me reconsider" patterns emerge. Self-correction, alternative-approach exploration, error checking. Average response length climbs to 5K-10K tokens.</li>
  <li><strong>The "aha moment"</strong>: at some training step, the model starts reliably producing extended reasoning chains for hard problems. This is an emergent transition, not a gradual one.</li>
</ul>

<p>None of this was supervised. The reasoning patterns weren't in the SFT data; they emerged because the verifier kept rewarding correct final answers, and long reasoning was the path that produced more correct final answers. <strong>This is the most consequential finding in post-training since RLHF itself</strong> — that with the right reward signal, complex behaviors can emerge from RL alone.</p>

<p>That said, R1's full recipe wasn't pure RL-from-scratch. The published recipe involved:</p>

<ol>
  <li><strong>Cold-start SFT</strong>: a small amount of SFT on hand-written reasoning traces, to give the base model a starting distribution of "how reasoning is structured." Without this, RL training was unstable.</li>
  <li><strong>RL phase 1 (reasoning)</strong>: GRPO on math/code tasks with verifiers. Long-CoT capability emerges here.</li>
  <li><strong>SFT phase 2</strong>: collect the model's own outputs from the RL phase, filter for correct ones, SFT a fresh checkpoint on this distilled data. Mixed with general-purpose SFT data.</li>
  <li><strong>RL phase 2 (alignment)</strong>: standard preference learning (DPO-style) for safety, helpfulness, honesty.</li>
</ol>

<p>Step 2 is where the reasoning capability comes from; step 1 is the prerequisite that makes step 2 stable; steps 3-4 turn the reasoning model into a deployable product (the bare R1-zero from step 2 has reasoning capability but is hard to use).</p>

<h2>Reasoning model serving: the test-time-compute architecture</h2>

<p>Once you have a reasoning model, serving it is qualitatively different from serving a standard chat model. The change comes down to one fact: <strong>generations are long</strong>. Where a chat response is 100-500 tokens typical, a reasoning response is 5K-50K tokens, sometimes more.</p>

<p>This stresses the M27 inference stack in specific ways:</p>

<ul>
  <li><strong>KV cache pressure goes up dramatically</strong>. A 32K-token reasoning response × 32 layers × 8 KV heads × 128 head dim × 2 bytes = ~2 GB just for KV cache for one request. Paged KV (M27) and aggressive page reuse become essential.</li>
  <li><strong>Per-request latency is fundamentally tens of seconds</strong>. At 100 tokens/sec generation, 10K reasoning tokens = 100 seconds. There's no fixing this with infrastructure — the model needs to think. <em>UX shifts to "submit and wait" or "stream the reasoning visibly so users see progress."</em></li>
  <li><strong>Throughput vs latency tradeoff intensifies</strong>. Reasoning serving is the regime where larger batches matter most — the per-request latency is so high that batching up many concurrent requests is the only way to keep GPUs busy. Continuous batching (M27) and speculative decoding (M27) do more work per H100 here than on standard chat.</li>
  <li><strong>Prefill-vs-decode ratio changes</strong>. Standard chat: short prompt, short response — roughly balanced. Reasoning: short prompt, very long response — decode dominates compute by 10-100×. This shifts what optimizations matter.</li>
</ul>

<p>The architectural pattern that emerges: <strong>reasoning models are typically served on dedicated infrastructure</strong>, separate from regular chat models. Different batch sizes, different memory budgets, different SLOs. vLLM and TensorRT-LLM both have reasoning-mode configurations.</p>

<h2>Test-time compute: spending more compute for better answers</h2>

<p>The serving infrastructure gives you the per-request mechanics. But reasoning models open up a new dimension that standard chat doesn't have: <strong>spend more inference compute, get better answers</strong>. This is the test-time-compute story.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 380" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Test-time compute Pareto: spend more inference, get better quality</text>

  <!-- Axes -->
  <line x1="80" y1="320" x2="700" y2="320" stroke="#1a1612" stroke-width="1.5"/>
  <line x1="80" y1="320" x2="80" y2="60" stroke="#1a1612" stroke-width="1.5"/>
  <text x="390" y="358" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">inference compute (log scale: tokens generated × verifier evals)</text>
  <text x="50" y="190" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612" transform="rotate(-90 50 190)">accuracy on hard task</text>

  <!-- X-axis labels -->
  <text x="120" y="335" text-anchor="middle" font-size="9" fill="#6b5d4f">1×</text>
  <text x="240" y="335" text-anchor="middle" font-size="9" fill="#6b5d4f">10×</text>
  <text x="360" y="335" text-anchor="middle" font-size="9" fill="#6b5d4f">100×</text>
  <text x="480" y="335" text-anchor="middle" font-size="9" fill="#6b5d4f">1000×</text>
  <text x="600" y="335" text-anchor="middle" font-size="9" fill="#6b5d4f">10000×</text>

  <!-- Y-axis labels -->
  <text x="75" y="320" text-anchor="end" font-size="9" fill="#6b5d4f">0%</text>
  <text x="75" y="240" text-anchor="end" font-size="9" fill="#6b5d4f">25%</text>
  <text x="75" y="160" text-anchor="end" font-size="9" fill="#6b5d4f">50%</text>
  <text x="75" y="80" text-anchor="end" font-size="9" fill="#6b5d4f">~80%</text>

  <!-- Curves -->
  <!-- Greedy decode -->
  <path d="M 120 280 L 700 280" stroke="#6b5d4f" stroke-width="2" stroke-dasharray="4 3" fill="none"/>
  <text x="710" y="284" font-size="9" fill="#6b5d4f">greedy: flat (no extra compute)</text>

  <!-- Best-of-N -->
  <path d="M 120 280 Q 200 250 280 220 Q 360 195 440 180 Q 520 170 600 165 L 700 162" stroke="#1f5f5b" stroke-width="2" fill="none"/>
  <text x="710" y="166" font-size="9" fill="#1f5f5b">best-of-N (with verifier)</text>

  <!-- Majority voting -->
  <path d="M 120 280 Q 200 260 280 240 Q 360 220 440 210 Q 520 200 600 195 L 700 192" stroke="#d4a017" stroke-width="2" fill="none"/>
  <text x="710" y="196" font-size="9" fill="#d4a017">majority voting (no verifier)</text>

  <!-- Verifier-guided search -->
  <path d="M 120 280 Q 200 240 280 195 Q 360 165 440 140 Q 520 120 600 110 L 700 105" stroke="#c1502e" stroke-width="2.5" fill="none"/>
  <text x="710" y="109" font-size="9" font-weight="700" fill="#c1502e">verifier-guided search</text>

  <!-- Long-CoT (single response, many tokens) -->
  <path d="M 120 280 Q 200 260 280 230 Q 360 200 440 175 Q 520 155 600 140 L 700 130" stroke="#b85a6c" stroke-width="2" fill="none"/>
  <text x="710" y="134" font-size="9" fill="#b85a6c">single long-CoT</text>

  <text x="390" y="60" text-anchor="middle" font-family="'Caveat', cursive" font-size="20" fill="#1a1612">key insight: this dimension didn't exist for standard chat models</text>
</svg>
</div>

<p>The simplest test-time compute method is <strong>just letting the reasoning model think for longer</strong> — give it a higher max-tokens budget. Quality climbs because the model has room to verify its work, try alternatives, catch errors. This is the curve labeled "single long-CoT" in the chart.</p>

<p>But you can go further by <em>combining multiple inferences</em>:</p>

<ul>
  <li><strong>Best-of-N with verifier</strong>: sample N reasoning responses, run the verifier on each, return the one that verified. Strictly better than greedy when the verifier is reliable. Cost: N× more inference compute. Quality: dramatically better — N=64 is often enough to push accuracy from 50% to 80%+ on hard math.</li>
  <li><strong>Majority voting</strong>: when no verifier is available (or verification is partial), sample N responses, take the most common final answer. Works because correct answers tend to converge while wrong ones diverge. Less powerful than best-of-N but applicable when verification is impossible.</li>
  <li><strong>Verifier-guided beam / tree search</strong>: at each step in the reasoning, branch into multiple continuations, evaluate each branch with a process reward model (PRM), prune low-scoring branches. The most computationally intensive option but also the most quality-efficient on hard problems. Frontier reasoning systems (rumored) use variants of this.</li>
</ul>

<h3>PRM vs ORM: outcome vs process rewards</h3>

<p>Two reward-model styles, useful in different parts of the test-time-compute story:</p>

<ul>
  <li><strong>Outcome Reward Model (ORM)</strong>: scores the final answer. Cheap to use (one call per response), trained on (response, correct/incorrect) pairs. Good for best-of-N where you only care about the final answer. <em>Limitation</em>: can't tell you which step in the reasoning went wrong.</li>
  <li><strong>Process Reward Model (PRM)</strong>: scores each step in the reasoning chain. Trained on (step, was-this-step-correct) pairs — typically labeled by humans or by an ORM applied to truncated chains. Used during search to prune bad branches early. <em>Cost</em>: needs more annotations to train; one call per reasoning step at inference time.</li>
</ul>

<p>For pure verifier-friendly tasks (math, code), the verifier itself replaces both. For tasks where verification is partial or expensive, ORMs and PRMs let you scale test-time search at lower cost.</p>

<h2>Distillation from reasoners</h2>

<p>One of R1's most practically impactful findings: <strong>reasoning capabilities transfer to smaller models via SFT on the bigger model's outputs</strong>. Take a 70B reasoning model, generate reasoning traces on math/code problems (filtered for correctness via verifier), and SFT a 7B base model on those traces. The 7B model inherits substantial reasoning capability without needing its own RL training.</p>

<p>The distillation recipe roughly:</p>

<ol>
  <li>Take a strong reasoning model (the teacher).</li>
  <li>Generate reasoning traces on a large pool of math/code problems. Use temperature 0.7-1.0 to get diversity.</li>
  <li>Filter: keep only traces where the final answer was correct (via verifier).</li>
  <li>SFT a smaller base model (the student) on the filtered traces. Standard SFT — instruction format, prompt-masked loss.</li>
</ol>

<p>The result: the student is much weaker than the teacher (smaller models are weaker), but its reasoning capability is much stronger than direct SFT on equivalent-size data. R1-distill 7B and 14B models are notably competitive with much larger non-reasoning models on math.</p>

<p><em>Why does this work?</em> Probably because reasoning is partly a "behavioral" capability — knowing the structure of a good reasoning trace, when to backtrack, what to verify — that can be imitated from examples. Some of the underlying reasoning capacity may be missed (a 7B model has less raw capacity than a 70B), but enough transfers to make this a practical recipe.</p>

<p>Distillation is also <em>much cheaper</em> than RL training. SFT compute for a 7B model on 100K reasoning traces is maybe a day on 8 H100s; RL training of a reasoning model is weeks on hundreds. <strong>Most production reasoning models people deploy are distilled, not trained from scratch with RL.</strong></p>

<h2>What works, what's still hard</h2>

<p>The reasoning-model recipe works well for tasks with verifiable rewards. The main domains as of 2026:</p>

<div class="table-wrap">
<table>
<caption>Reasoning model task domains and verifier choices</caption>
<thead><tr><th>Domain</th><th>Verifier</th><th>Maturity</th></tr></thead>
<tbody>
<tr><td>Math (AIME, MATH, AMC)</td><td>sympy equivalence; numerical match</td><td>Standard; widely deployed</td></tr>
<tr><td>Competitive programming</td><td>unit tests / hidden tests</td><td>Standard; competitive with humans on Codeforces-style tasks</td></tr>
<tr><td>Software engineering</td><td>repo unit tests + functional tests</td><td>Active; SWE-bench-style benchmarks</td></tr>
<tr><td>Formal proofs (Lean, Coq)</td><td>proof checker</td><td>Active research; promising</td></tr>
<tr><td>Research math</td><td>partial (steps verifiable, end-to-end not)</td><td>Frontier; PRM-based search becomes essential</td></tr>
<tr><td>Scientific reasoning</td><td>partial (calculations checkable, hypotheses not)</td><td>Active; mostly empirical</td></tr>
<tr><td>Long-form writing, creativity</td><td>None (no programmatic check)</td><td>Doesn't directly apply</td></tr>
<tr><td>Persuasion, ethics, judgment</td><td>None</td><td>Doesn't apply; back to RLHF (M31)</td></tr>
</tbody>
</table>
</div>

<p>The pattern: <strong>where a verifier exists, RL with verifiable rewards works extraordinarily well</strong>. Where it doesn't, classic RLHF (M31) is still needed. Most production reasoning systems are hybrids: RL-from-verifier on the math/code subset, RLHF on the chat/safety subset.</p>

<p>Open questions still very much being worked on:</p>

<ul>
  <li><strong>Generalization beyond the trained domain</strong>: a model trained on competition math — does it improve at reasoning generally? Empirical evidence is mixed; transfer to non-math reasoning tasks is real but smaller than within-domain gains.</li>
  <li><strong>Process rewards from human preferences</strong>: getting humans to label "was this reasoning step correct" reliably is hard. Most PRMs use AI feedback; human-labeled PRMs are more accurate but expensive.</li>
  <li><strong>The hallucination-reasoning tension</strong>: long reasoning chains have more opportunities to hallucinate facts. Reasoning models often confidently compute wrong arithmetic in their traces. Active mitigation work.</li>
  <li><strong>Compute scaling laws for test-time vs train-time</strong>: how do you optimally allocate a fixed compute budget between training a bigger model, training longer, and spending more at inference time? Empirically this is being actively studied; theoretical understanding is partial.</li>
</ul>

<div class="ndq">
<h4>About reasoning models</h4>

<p class="q">Why does GRPO work without a value model when classic PPO needs one?</p>
<p class="a">PPO's value model exists for variance reduction — to subtract a baseline from the reward so the gradient signal isn't dominated by reward magnitude noise. GRPO gets a baseline for free: the mean reward across the K samples for the same prompt is itself a baseline (and, after subtracting, advantages have zero mean by construction). Normalizing by the group std makes this even more PPO-equivalent. <em>The tradeoff</em>: GRPO requires K samples per prompt, where PPO uses 1 — but verifier evaluation is cheap, and in the LLM regime K=16-64 samples cost less than maintaining a separate value-model train loop. <em>For verifiable-reward settings, GRPO dominates PPO on engineering simplicity at no quality cost.</em></p>

<p class="q">If reasoning models emerge from RL, why is the cold-start SFT step needed?</p>
<p class="a">Stability. R1's authors reported that pure RL from the base model was unstable — the base model's response distribution was too far from "reasoning trace" structure for the RL signal to find good gradients quickly. The cold-start SFT (a few thousand examples of human-written reasoning traces) gives the model a starting distribution where reasoning-shaped responses are already plausible. Then RL refines and extends. <em>Without cold-start, training works but takes much longer and may get stuck in degenerate local optima.</em> The published "R1-Zero" variant skipped cold-start and reached good but lower quality than R1 proper.</p>

<p class="q">Why does distillation work? It seems like it shouldn't transfer the underlying reasoning capability.</p>
<p class="a">It transfers the <em>behavioral pattern</em> of reasoning more than the underlying capability. The student learns to produce reasoning-shaped outputs (intermediate steps, verification, backtracking patterns) from the teacher's traces. Some of the underlying problem-solving ability is captured (the smaller model can follow the same algorithmic patterns); some isn't (smaller models hit harder ceilings on really difficult problems). But for tasks within the smaller model's capacity, the reasoning structure is a major boost. Empirically, R1-distill 14B reasoning beats GPT-4o non-reasoning on AIME-level math by a wide margin. <em>Reasoning is partly a "skill" that can be imitated, partly a "capacity" that can't.</em></p>

<p class="q">Can I use GRPO for non-verifiable tasks?</p>
<p class="a">Sort of. You can replace the verifier with a reward model and run GRPO with that as the score function. This works and is sometimes called "GRPO-RM." But you've now added back the reward model, with all its drift and hacking risks (M31). At that point, the engineering simplification GRPO offered is partly lost. <em>For non-verifiable tasks, DPO is usually a better choice</em> — its closed-form bypasses the rollout cost. GRPO's main wins (group baseline, no value model, exploration) are most valuable when the reward signal is exact, which is exactly the verifier setting.</p>

<p class="q">How do reasoning models interact with MoE architectures (M28)?</p>
<p class="a">DeepSeek-V3 and DeepSeek-R1 are both MoE — V3 is the base, R1 is V3 post-trained for reasoning. The MoE structure doesn't fundamentally change the reasoning training: GRPO works the same way on an MoE policy. What changes is the engineering: rollouts are slower (MoE inference has overhead from routing and all-to-all collectives), and serving a reasoning MoE has both the long-decode characteristics of reasoning <em>and</em> the memory characteristics of MoE (whole model resident in memory even though only some experts active per token). <em>It's the union of both M28's and M27's complexities; not multiplicative, but additive.</em></p>

<p class="q">When does reasoning compute become more cost-effective than training a bigger model?</p>
<p class="a">An active research question, but the empirical pattern: at the high end of difficulty (frontier math, hard programming), test-time compute (best-of-N, verifier-guided search) gives steeper quality-vs-compute curves than train-time scaling. At the low end (easy questions), the model usually gets it on the first try and extra inference compute is wasted. <em>Cost-effectiveness inverts at the difficulty boundary</em>: for easy chat, train a bigger model; for hard reasoning, lean on test-time. This is also why "reasoning models" are usually served separately — different cost-effective operating points.</p>
</div>

<h2>Code Magnets: implement the GRPO advantage</h2>

<p>You're computing the per-rollout advantages for one batch of GRPO. Three magnets are wrong choices.</p>

<div class="magnets">
<p>Arrange the magnets to compute the advantages.</p>

<div class="magnet-pool">
  <span class="magnet">rewards = torch.tensor([verify(p, r) for p, r in zip(prompts_repeated, responses)])</span>
  <span class="magnet">rewards = rewards.view(num_prompts, K)</span>
  <span class="magnet">rewards = rewards.view(K, num_prompts)</span>
  <span class="magnet">group_mean = rewards.mean(dim=1, keepdim=True)</span>
  <span class="magnet">group_mean = rewards.mean()</span>
  <span class="magnet">group_std = rewards.std(dim=1, keepdim=True) + 1e-8</span>
  <span class="magnet">advantages = ((rewards - group_mean) / group_std).flatten()</span>
  <span class="magnet">advantages = (rewards - group_mean).flatten()</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code>rewards = torch.<span class="fn">tensor</span>([<span class="fn">verify</span>(p, r) <span class="kw">for</span> p, r <span class="kw">in</span> <span class="fn">zip</span>(prompts_repeated, responses)])
rewards = rewards.<span class="fn">view</span>(num_prompts, K)
group_mean = rewards.<span class="fn">mean</span>(dim=<span class="num">1</span>, keepdim=<span class="kw">True</span>)
group_std = rewards.<span class="fn">std</span>(dim=<span class="num">1</span>, keepdim=<span class="kw">True</span>) + <span class="num">1e-8</span>
advantages = ((rewards - group_mean) / group_std).<span class="fn">flatten</span>()</code></pre>
<p>The traps:</p>
<ul>
  <li><code>rewards.view(K, num_prompts)</code>: wrong reshape order. The rollout loop iterates K times for each prompt, then prompts; flattening that gives shape (num_prompts × K) — reshaping to (num_prompts, K) groups responses for the same prompt together. Reversing it groups across prompts, which makes the group mean meaningless.</li>
  <li><code>group_mean = rewards.mean()</code>: takes the global mean across all prompts and all rollouts. This loses the per-prompt baseline — easy prompts and hard prompts get the same baseline, and easy-prompt advantages are crushed while hard-prompt advantages are inflated. The whole point of "group relative" is per-prompt baseline; <code>mean(dim=1, keepdim=True)</code> is what gives that.</li>
  <li><code>advantages = (rewards - group_mean).flatten()</code>: skips the std normalization. The unnormalized form works (and is what the original GRPO paper used) but is more sensitive to reward scale and can produce very different gradient magnitudes when verifiers return different reward ranges. Dividing by group std (the Z-score form) makes the algorithm scale-invariant and more stable in practice. <em>Production GRPO implementations use the normalized form.</em></li>
</ul>
<p>The pattern: <strong>reshape to (prompts, K) → per-row mean for baseline → per-row std for normalization → subtract mean, divide by std → flatten back to (prompts × K) for the policy update</strong>. The reshape and the per-prompt aggregation are what make GRPO "group relative."</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each reasoning-model concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>Verifier</div>
  <div>A. Programmatic check on a response (sympy, unit tests, proof checker); produces a score with no learned bias.</div>

  <div>GRPO</div>
  <div>B. RL algorithm that uses K rollouts and group-mean baseline instead of a value model.</div>

  <div>Group-mean baseline</div>
  <div>C. Per-prompt advantage = (reward − group_mean) / group_std; replaces the value model.</div>

  <div>Cold-start SFT</div>
  <div>D. Pre-RL stage on hand-written reasoning traces; gives the policy a stable starting distribution.</div>

  <div>Best-of-N with verifier</div>
  <div>E. Sample N responses, return the one that verified; major test-time-compute lever.</div>

  <div>Process Reward Model (PRM)</div>
  <div>F. Scores each step in a reasoning chain; used to prune branches in tree search.</div>

  <div>R1-distill</div>
  <div>G. SFT a smaller base model on filtered (verified-correct) reasoning traces from a larger reasoner.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>Verifier</strong> → A<br>
<strong>GRPO</strong> → B<br>
<strong>Group-mean baseline</strong> → C<br>
<strong>Cold-start SFT</strong> → D<br>
<strong>Best-of-N with verifier</strong> → E<br>
<strong>Process Reward Model</strong> → F<br>
<strong>R1-distill</strong> → G
</p>
<p>The mental shortcut: <em>verifier scores, GRPO trains, group mean baselines, cold-start initializes, best-of-N searches, PRM scores steps, distillation transfers</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team trains a reasoning model with GRPO. After 5,000 steps, the model produces correct answers on training problems with high accuracy, but average response length has dropped to about 50 tokens — the model just states the answer. What's likely going on, and what would you change?</p>
<details class="answer"><summary>show answer</summary>
<p>The verifier is rewarding correct final answers regardless of how the model got there. If the base model can pattern-match many of the training problems directly (perhaps they're well-represented in pretraining data), it learns that <em>shorter responses</em> with the right answer get the same reward as long reasoning chains. Why think when you can just write "x = 3"? Group-mean advantages don't penalize short responses; they reward whatever produces the highest reward in the group.</p>
<p>The fix is reward shaping. Two common approaches: (1) <strong>Length-aware rewards</strong>: small bonus for responses that include intermediate reasoning steps, or penalty for very short responses on hard problems. (2) <strong>Difficulty-stratified training</strong>: train on problems where the base model fails frequently — those force the model to actually reason rather than pattern-match. (3) <strong>Process rewards (PRM)</strong>: explicitly reward the existence of structured reasoning steps, not just final correctness. The R1 paper noted this exact issue and addressed it by carefully selecting training problems that the base model couldn't solve directly. <em>Reward shaping is half the engineering art of reasoning-model training</em>; the GRPO algorithm itself is the easier part.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Walk through why GRPO uses K samples per prompt instead of just one (like vanilla policy gradient). What's lost if K=1?</p>
<details class="answer"><summary>show answer</summary>
<p>Two things are lost with K=1.</p>
<p>First, <strong>the baseline disappears</strong>. The whole point of "group relative" is that <code>advantage_i = reward_i − mean(rewards in group)</code> — the group mean is the baseline. With K=1, the "group" is one sample, the mean is the single reward, advantage is exactly zero, gradient is zero. <em>You'd need a value model to recover any baseline</em>, defeating the no-value-model property.</p>
<p>Second, <strong>exploration is lost</strong>. With K=1 you're doing on-policy training: the policy generates one response, the verifier scores it, gradient nudges the policy. With K=16, you're sampling 16 diverse responses for each prompt, including some that pure greedy would never produce. The verifier identifies which of those 16 was correct (or most correct), and the gradient pushes the policy toward producing those. <em>Exploration via temperature-1 sampling is what lets the policy learn behaviors not in its current distribution</em> — like long reasoning chains when it currently produces short ones.</p>
<p>Practically, most GRPO implementations use K=8 to 64. Higher K means more exploration but more rollout compute. K=16 is a good starting point.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> A team's reasoning model serves with 8K-token max generations. Latency is acceptable but their math benchmark accuracy is 55%. They have spare GPU capacity and want to push accuracy higher without retraining. What test-time compute approach would you suggest, and what's the likely trade?</p>
<details class="answer"><summary>show answer</summary>
<p>The most reliable lever is <strong>best-of-N with the verifier</strong>. For each problem, generate N=16 (or 32, 64) responses, run the verifier on each, return the first that verified. On hard math benchmarks, best-of-16 with a strong verifier typically pushes accuracy from 55% to 75-85% — the model often produces a correct answer in <em>some</em> of the 16 samples even when greedy is wrong. The verifier picks it out. <strong>Cost</strong>: 16× more inference compute per problem. <strong>Quality</strong>: a substantial step up.</p>
<p>If they don't have a usable verifier at inference time (e.g., serving end-users with no formal answer to check against), the alternative is <strong>majority voting</strong>: sample N responses, take the most common final answer. Less powerful than verifier best-of-N but applicable without ground truth. Typical gain: 55% → 65-70% with N=16. Still substantial.</p>
<p>Higher-end option: <strong>verifier-guided tree search</strong> with a PRM. At each reasoning step, branch and evaluate; prune low-PRM branches. Most expensive (often 100×+ vs greedy) but pushes the highest-difficulty problems. Probably overkill if best-of-N gets them to 80% — diminishing returns past that.</p>
<p>The trade space is well-defined: pick where on the Pareto frontier you want to operate. Verifier best-of-N at N=16 is usually the right entry point.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> Why are reasoning models a poor fit for tasks like creative writing, even though they handle complex math beautifully?</p>
<details class="answer"><summary>show answer</summary>
<p>Three reasons that compound.</p>
<p>(1) <strong>No verifier exists for creative writing</strong>. The whole reasoning-model recipe relies on a programmatic check on the final output. There's no equivalent for "is this poem good" or "is this story creative." The training signal that produces reasoning behavior simply doesn't apply.</p>
<p>(2) <strong>The reasoning behavior itself is misaligned with creative tasks</strong>. A reasoning model trained on math will, when asked to write a poem, often produce intermediate reasoning ("Let me think about what makes a good poem...") followed by a poem. The reasoning prefix is verbose and rarely improves the creative output — sometimes it actively hurts by anchoring the model on its first thoughts about structure rather than letting it generate fluidly.</p>
<p>(3) <strong>Long-thinking and creativity have different optima</strong>. Reasoning emphasizes correctness, verification, backtracking — useful for math, harmful for novel idea generation where the first draft is often the most creative. Production reasoning models often have a "do not engage reasoning mode" flag for tasks where it's counterproductive.</p>
<p>The practical pattern: <strong>route by task type</strong>. Hard math/code goes to the reasoning model; chat/creative goes to a standard chat model. Some products do this transparently; some let the user choose. A unified model that's optimal at both is still future work — if it's even achievable.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>Reasoning models — o1, R1, and successors — are transformers that learn to <strong>think before answering</strong>: produce thousands of intermediate reasoning tokens, then a short final answer.</li>
  <li>The architectural change is roughly zero. The training and serving change radically.</li>
  <li><strong>Verifiable rewards</strong> replace human preferences for tasks where ground truth exists: math (sympy), code (unit tests), formal proofs (proof checker). The verifier is exact, free, and immune to the reward-model failure modes from M31.</li>
  <li><strong>GRPO</strong> is the RL algorithm of choice for verifiable-reward settings. Sample K rollouts per prompt, score each, advantage = (reward − group_mean) / group_std. PPO-clipped objective, KL penalty against frozen reference.</li>
  <li><strong>Two models in memory</strong> (policy + reference) — same as DPO. <strong>No reward model, no value model</strong> — the verifier replaces the RM, the group mean replaces the value baseline.</li>
  <li><strong>Long-CoT emergence</strong>: with the right reward shaping, the model spontaneously learns reasoning patterns including backtracking and self-correction — none of which were in the SFT data. R1's "aha moment."</li>
  <li><strong>R1's full recipe</strong>: cold-start SFT on hand-written reasoning traces → GRPO on math/code → SFT on filtered self-generated traces → final RLHF for safety/alignment.</li>
  <li>Serving reasoning models stresses M27's stack: KV cache pressure (long generations), per-request latency in tens of seconds, throughput-vs-latency tradeoff intensified, decode-dominant compute mix.</li>
  <li><strong>Test-time compute</strong>: a new dimension. Beyond "make the model thinker longer," combine multiple inferences via <strong>best-of-N with verifier</strong>, <strong>majority voting</strong>, <strong>verifier-guided tree search</strong>.</li>
  <li>ORM (outcome reward model) scores final answers — cheap, used in best-of-N. PRM (process reward model) scores each reasoning step — more expensive, used in tree search.</li>
  <li><strong>Distillation from reasoners</strong>: SFT a smaller base model on the bigger model's verifier-filtered traces. Inherits much of the reasoning capability at a fraction of the training cost. R1-distill family.</li>
  <li>The recipe works where verifiers exist (math, code, proofs). For creative writing, persuasion, ethics, judgment — back to RLHF (M31). Production systems route by task type.</li>
  <li>The reflex: when the task has an exact correctness check, reach for verifiable-reward RL. When it doesn't, reach for preference-based RL. <em>Match the algorithm to the available signal.</em></li>
</ul>
</div>

<p>This module ties together M27 (serving long generations), M28 (MoE — many reasoning models are MoE), and M31 (the post-training pipeline that GRPO sits in). The reasoning-model recipe represents the most consequential post-training development since RLHF itself, and the engineering it requires — verifier infrastructure, rollout pipelines, test-time-compute search systems — is rapidly maturing into standard practice.</p>

<p>If a Part X continues, the natural next module would be <strong>pretraining data engineering at scale</strong> — quality classifiers, domain mixing, deduplication pipelines, contamination filtering. Quietly the biggest source of modern quality gains and almost no public material at the engineering level.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">34</span>
  <span>Reasoning models: long-thinking, GRPO &amp; test-time compute</span>
</div>
"""

emit("34_reasoning_models", "Module 34 — Reasoning models: long-thinking, GRPO & test-time compute", BODY)
