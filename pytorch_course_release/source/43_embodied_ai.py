#!/usr/bin/env python3
"""Module 43: Embodied AI & Robotics RL — full HF vibe, April 2026 current."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part XI · Module 43 · April 2026 currency</div>
  <h1 class="module-title"><em>Embodied AI &amp; Robotics RL:</em> when the rollout meets the floor</h1>
  <p class="module-sub">— from VLA architectures (OpenVLA, π0, SmolVLA) to sim-to-real RL with RLinf, why real-robot data is the bottleneck, and what changed in 2025-2026 as Atlas/Optimus/Figure entered actual production</p>
</div>

<p>For 42 modules we've trained models that produce text. M40 extended this to agents that produce actions in software environments — browsers, terminals, IDEs. M43 closes Part XI by extending it one more step: <strong>agents that produce actions in the physical world</strong>. The model outputs joint torques, gripper commands, end-effector poses. Errors don't just lose tokens — they break things.</p>

<p>Embodied AI has been a long-term promise of ML; the reason it's now a frontier engineering topic in April 2026 is concrete. Boston Dynamics Atlas <strong>entered commercial production at CES 2026</strong> with all 2026 units allocated to Hyundai. Figure 03 supports 30,000+ vehicles at BMW with continuous unsupervised operation. Tesla Optimus Gen 3 began Fremont production in January. Unitree shipped 5,500+ humanoids in 2025 at $13,500-$16,000 price points; on April 19, 2026, Honor's "Lightning" won the Beijing E-Town Half-Marathon at 50:26 — beating the human world record by nearly 7 minutes. <em>The transition from research curiosity to deployed hardware happened in 2025-2026</em>. The training stack that makes this work is what M43 covers.</p>

<p>Two technical shifts power the moment. First, <strong>VLA models</strong> (Vision-Language-Action) — vision encoder + LLM trunk + action head, trained on robot demonstration data — generalize across tasks and embodiments in ways pure-RL or pure-imitation methods didn't. Open VLAs (OpenVLA, π0/π0.5, SmolVLA, NORA) put production-grade architectures in researchers' hands. Second, <strong>sim-to-real RL pipelines</strong> matured: high-fidelity simulators, real-to-sim tuning, divide-and-conquer distillation, RLinf for unified embodied/agentic infrastructure, RLinf-Co's sim-real co-training showing +24% real-world success on OpenVLA. The recipes are public; the bottleneck is real-robot data, not ideas.</p>

<div class="keyidea">
Embodied AI's 2026 architecture is the <strong>VLA model</strong> — vision encoder (DINOv2 + SigLIP fused, M38 territory) + LLM trunk (M11/M14 territory) + action head producing tokenized robot actions. <strong>OpenVLA</strong> (7B, June 2024) was the open-source breakthrough; trained on 970K episodes from Open X-Embodiment, beat closed RT-2-X (55B) by 16.5% absolute task success across 29 tasks. <strong>π0/π0.5</strong> (Physical Intelligence) and <strong>SmolVLA</strong> (HF, 450M, flow matching + async inference) are the production references. The training recipe layers: <strong>(1)</strong> SFT on demonstrations (teleoperation data); <strong>(2)</strong> sim-to-real RL using high-fidelity simulators (Genie Sim 3.0, ManiSkill, RoboTwin); <strong>(3)</strong> sim-real co-training (RLinf-Co, Feb 2026: +24% OpenVLA, +20% π0.5 real-world success); <strong>(4)</strong> real-world online RL via human-gated DAgger (RLinf v0.2 + HG-DAgger, March-April 2026). <strong>The sim-to-real gap</strong> is mitigated by domain randomization (older), real-to-sim-to-real pipelines, generative scene modeling, automated real-to-sim tuning, and stage-aware reward design. <strong>Real-robot data is the bottleneck</strong>: 970K episodes (OpenVLA training) is the largest open dataset; humanoid platforms generate teleoperation data continuously now. The deployment reality April 2026: <strong>Atlas in production for Hyundai</strong>, <strong>Figure 03 at BMW with 90K+ parts handled</strong>, <strong>Optimus Gen 3 mass production Q1 2026</strong>, <strong>Unitree G1 at $13,500</strong>, <strong>Honor Lightning autonomous at marathon-winning pace</strong>. The next two years' capability gains will come from real-robot data scale plus better VLA architectures.
</div>

<h2>Two new faces — closing Part XI and the cast</h2>

<div class="character" style="--c: #d4a017;">
  <div class="avatar" style="background: #d4a017; color: #fff;">A</div>
  <div>
    <p class="who">VLA</p>
    <p class="name">"I'm a Vision-Language-Action model. Image in, language reasoning, action tokens out."</p>
    <p class="says">My architecture is M38's multimodal stack with one extra piece: an action head. <em>Vision encoder</em> (DINOv2 + SigLIP fused, in OpenVLA's case) processes the robot's camera; <em>LLM trunk</em> (Llama 2 7B for OpenVLA; Gemma for π0; Qwen for many newer variants) reasons over what to do; <em>action head</em> outputs tokenized robot actions — typically discrete bins for end-effector deltas (position + rotation) plus gripper. The actions get decoded into continuous control commands the robot executes. Once you have me, you fine-tune for new tasks/embodiments rather than training from scratch. <strong>I'm what made generalist robot manipulation possible at 7B scale</strong>; OpenVLA at 7B beat RT-2-X at 55B in published benchmarks. Smaller siblings (SmolVLA at 450M, NORA at ~250M) target real-time deployment.</p>
  </div>
</div>

<div class="character" style="--c: #1f5f5b;">
  <div class="avatar" style="background: #1f5f5b; color: #fff;">S</div>
  <div>
    <p class="who">Sim2Real</p>
    <p class="name">"I'm the bridge between simulator success and real-world success — and I'm where most engineering goes."</p>
    <p class="says">Train a policy purely in simulation; deploy it on a real robot; watch it fail. The reasons compound: physics differs subtly (friction, contact, mass distribution), perception differs grossly (sim renders aren't photos), the robot's actual actuators have backlash and delay sim doesn't model. <strong>I'm the techniques that close this gap.</strong> Older recipes use domain randomization (train across many sim variations so the policy is robust). Newer recipes do real-to-sim-to-real (build a high-fidelity sim from real data, train, deploy back). The 2026 frontier (RLinf-Co, Feb 2026): <em>co-train with both sim and real data simultaneously</em>, RL in sim with auxiliary supervised loss on real demonstrations to anchor the policy. The Robot-Trains-Robot recipe (Aug 2025) optimizes a single dynamics-encoded latent in real-world during deployment to adapt fast. <strong>I'm not optional</strong>; pure-sim policies don't deploy well, pure-real RL is too data-hungry, the hybrid wins.</p>
  </div>
</div>

<h2>Why embodied AI is now a frontier engineering topic</h2>

<p>Three things changed in the last 18 months that make this a 2026 frontier topic, not just a research curiosity:</p>

<ol>
  <li><strong>VLA architectures crystallized</strong>. The 2023 RT-2 paper (Google DeepMind) established the VLA paradigm — VLM backbone with action head fine-tuned on robot demonstrations. By June 2024, OpenVLA put a 7B open-source version in researchers' hands. By 2025-2026, the architecture became standard: π0/π0.5 from Physical Intelligence, SmolVLA from Hugging Face, NORA, DexVLA, ChatVLA-2, dVLA, plus closed releases like Gemini Robotics. <em>The architecture question is mostly settled</em>; the engineering is now about scale and recipe.</li>
  <li><strong>RL infrastructure for embodied AI got real</strong>. RLinf (March 2026 v0.2) brought a unified framework for embodied + agentic RL with first-class support for real-world Franka training, sim-to-real co-training (RLinf-Co), DAgger variants for embodied policies, integration with simulation platforms (RoboTwin, Genie Sim 3.0). M40's environments-as-a-service pattern extends naturally to physical environments — same training loop, different observation/action spaces.</li>
  <li><strong>Hardware shipped</strong>. Boston Dynamics Atlas in commercial production (CES 2026, Hyundai). Figure 03 at BMW with 90K+ parts handled. Tesla Optimus Gen 3 mass production Q1 2026. Unitree shipping at $13,500. The transition from "robot demos" to "robot deployments" is real and dated — late 2025 through early 2026. <em>The training stack now serves real production.</em></li>
</ol>

<p>The economic context: humanoid robot manufacturing costs dropped 40% from 2023 to 2024 (Goldman Sachs); per-unit costs fell from $50K-$250K to $30K-$150K. This compounds with software improvements — better VLAs at smaller sizes mean cheaper, faster inference on robots that themselves are cheaper. <strong>The trajectory mirrors LLMs in 2022-2024</strong>: research-curiosity → research-tool → production-product within 18-24 months.</p>

<h2>VLA architecture: what's specifically different from M38</h2>

<p>The VLA architecture inherits from M38's multimodal stack but adds one critical component. Walk through OpenVLA's design as the canonical example:</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 400" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrV" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">VLA architecture: M38's multimodal stack + action head</text>

  <!-- Inputs -->
  <g transform="translate(20, 50)">
    <rect x="0" y="0" width="120" height="40" fill="#fff8a8" stroke="#1a1612" stroke-width="1.5"/>
    <text x="60" y="18" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Robot camera</text>
    <text x="60" y="32" text-anchor="middle" font-size="9" fill="#1a1612">RGB image(s)</text>

    <rect x="0" y="55" width="120" height="40" fill="#fff8a8" stroke="#1a1612" stroke-width="1.5"/>
    <text x="60" y="73" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Task instruction</text>
    <text x="60" y="87" text-anchor="middle" font-size="9" fill="#1a1612">"pick up the cup"</text>
  </g>

  <!-- Vision encoder fusion -->
  <g transform="translate(170, 40)">
    <rect x="0" y="0" width="170" height="60" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="85" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">Fused vision encoder</text>
    <text x="20" y="34" font-size="9" fill="#1a1612">  DINOv2 (spatial features)</text>
    <text x="20" y="48" font-size="9" fill="#1a1612">  + SigLIP (language-aligned)</text>

    <rect x="0" y="65" width="170" height="35" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="85" y="80" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Connector (MLP)</text>
    <text x="85" y="93" text-anchor="middle" font-size="8" fill="#1a1612">vision → LLM space</text>
  </g>

  <line x1="140" y1="70" x2="168" y2="70" stroke="#1f5f5b" stroke-width="1.5" marker-end="url(#arrV)"/>

  <!-- LLM trunk -->
  <g transform="translate(370, 40)">
    <rect x="0" y="0" width="140" height="100" fill="#fff5d8" stroke="#d4a017" stroke-width="2"/>
    <text x="70" y="20" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">LLM trunk</text>
    <text x="70" y="36" text-anchor="middle" font-size="9" fill="#1a1612">Llama-2-7B</text>
    <text x="70" y="50" text-anchor="middle" font-size="9" fill="#1a1612">(or Gemma, Qwen…)</text>
    <text x="20" y="68" font-size="9" fill="#1a1612">Vision tokens</text>
    <text x="20" y="80" font-size="9" fill="#1a1612">+ text tokens</text>
    <text x="20" y="92" font-size="9" font-weight="700" fill="#1a1612">→ predict action tokens</text>
  </g>

  <line x1="340" y1="80" x2="368" y2="80" stroke="#1f5f5b" stroke-width="1.5" marker-end="url(#arrV)"/>

  <!-- Action head + decoder -->
  <g transform="translate(540, 40)">
    <rect x="0" y="0" width="180" height="60" fill="#ffd5dc" stroke="#b85a6c" stroke-width="2"/>
    <text x="90" y="20" text-anchor="middle" font-size="11" font-weight="700" fill="#b85a6c">Action head</text>
    <text x="90" y="36" text-anchor="middle" font-size="9" fill="#1a1612">Discrete action tokens</text>
    <text x="90" y="50" text-anchor="middle" font-size="9" fill="#1a1612">(7-dim Δ pose + gripper)</text>

    <rect x="0" y="65" width="180" height="35" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="90" y="80" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Action decoder</text>
    <text x="90" y="93" text-anchor="middle" font-size="8" fill="#1a1612">tokens → continuous control</text>
  </g>

  <line x1="510" y1="80" x2="538" y2="80" stroke="#1f5f5b" stroke-width="1.5" marker-end="url(#arrV)"/>

  <!-- Robot -->
  <g transform="translate(280, 175)">
    <rect x="0" y="0" width="180" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="90" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">Robot executes</text>
    <text x="90" y="32" text-anchor="middle" font-size="9" fill="#1a1612">Franka / humanoid / mobile</text>
  </g>

  <line x1="630" y1="100" x2="630" y2="195" stroke="#1f5f5b" stroke-width="1.5"/>
  <line x1="630" y1="195" x2="463" y2="195" stroke="#1f5f5b" stroke-width="1.5" marker-end="url(#arrV)"/>

  <!-- Loop back -->
  <line x1="280" y1="195" x2="80" y2="195" stroke="#c1502e" stroke-width="1.5" stroke-dasharray="4 2"/>
  <line x1="80" y1="195" x2="80" y2="100" stroke="#c1502e" stroke-width="1.5" stroke-dasharray="4 2" marker-end="url(#arrV)"/>
  <text x="180" y="210" font-size="10" fill="#c1502e">closed loop: new camera frame → next action</text>

  <!-- Variants table -->
  <g transform="translate(20, 250)">
    <rect x="0" y="0" width="700" height="140" fill="#fff8a8" stroke="#1a1612" stroke-width="1.5"/>
    <text x="350" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">VLA variants in production, April 2026</text>

    <text x="20" y="48" font-size="10" fill="#1a1612">  <tspan font-weight="700">OpenVLA</tspan> (June 2024): 7B, Llama-2 + DINOv2/SigLIP, 970K episodes, beat RT-2-X 55B by +16.5%</text>
    <text x="20" y="64" font-size="10" fill="#1a1612">  <tspan font-weight="700">π0 / π0.5</tspan> (Physical Intelligence): production reference VLA; flow matching action head</text>
    <text x="20" y="80" font-size="10" fill="#1a1612">  <tspan font-weight="700">SmolVLA</tspan> (HF, 2025): 450M, flow matching + async inference, comparable to Octo/OpenVLA/π0</text>
    <text x="20" y="96" font-size="10" fill="#1a1612">  <tspan font-weight="700">NORA</tspan>: small generalist VLA (~250M); production-deployable</text>
    <text x="20" y="112" font-size="10" fill="#1a1612">  <tspan font-weight="700">dVLA, ChatVLA-2, DexVLA, LingBot-VLA</tspan>: 2025-2026 variants with diffusion / reasoning / dexterity</text>
    <text x="20" y="130" font-size="10" font-weight="700" fill="#1f5f5b">  Closed: Gemini Robotics (Google DeepMind, Atlas integration partnership), proprietary Tesla / Figure stacks</text>
  </g>
</svg>
</div>

<p>Three architectural notes that distinguish VLA from M38's pure VLM:</p>

<ul>
  <li><strong>Fused vision encoders are common</strong>. OpenVLA fuses DINOv2 (spatial features, good for object localization) with SigLIP (language-aligned features). Pure SigLIP loses spatial detail; pure DINOv2 loses language alignment. The fusion captures both — important for robot tasks where "pick up the red cup" needs both <em>where</em> and <em>what</em>.</li>
  <li><strong>Action tokens are typically discrete</strong>. The 7-DOF end-effector control (3D position delta + 3D rotation delta + gripper) is binned into discrete buckets. The LLM predicts action tokens like vocabulary tokens; a decoder converts them back to continuous values. Discrete tokens fit cleanly into the LLM's autoregressive prediction; continuous-action variants exist (π0/π0.5 use flow matching for the action head) and can be more precise but harder to integrate.</li>
  <li><strong>Closed-loop execution</strong>. Unlike M38's single-shot generation, robot control is a loop: camera frame → action → execute → new camera frame → next action. Inference latency matters. SmolVLA's <strong>async inference</strong> (decouple VLM backbone from action execution, run them at different cadences) is one solution; another is small VLAs (NORA, SmolVLA at sub-1B) that are fast enough for synchronous loops.</li>
</ul>

<h3>Action tokenization specifics</h3>

<pre><code><span class="kw">def</span> <span class="fn">tokenize_action</span>(action_continuous, n_bins=<span class="num">256</span>):
    <span class="com"># Convert continuous 7-DOF action to discrete action tokens</span>
    <span class="com"># action_continuous: (Δx, Δy, Δz, Δrx, Δry, Δrz, gripper)</span>
    <span class="com"># Each dim: clipped to [-1, 1], quantized into n_bins discrete buckets</span>
    clipped = <span class="fn">clip</span>(action_continuous, -<span class="num">1.0</span>, <span class="num">1.0</span>)
    <span class="com"># Map to integer tokens; reserve specific token IDs in vocab for actions</span>
    bins = ((clipped + <span class="num">1.0</span>) / <span class="num">2.0</span> * (n_bins - <span class="num">1</span>)).<span class="fn">round</span>().<span class="fn">int</span>()
    <span class="com"># OpenVLA uses 256 bins per dim; total action vocab = 256 × 7 = 1792 tokens</span>
    <span class="kw">return</span> bins  <span class="com"># [B, 7] integer tensor</span>

<span class="kw">def</span> <span class="fn">decode_action</span>(bins, n_bins=<span class="num">256</span>):
    <span class="com"># Convert action tokens back to continuous action</span>
    continuous = <span class="num">2.0</span> * (bins.<span class="fn">float</span>() / (n_bins - <span class="num">1</span>)) - <span class="num">1.0</span>
    <span class="kw">return</span> continuous</code></pre>

<p>The 256-bin choice is OpenVLA's. Resolution tradeoff: more bins = finer control but more parameters in the action head; fewer bins = coarser control but smaller head. 256 is a reasonable midpoint; some VLAs use 128 or 512 depending on task precision needs.</p>

<h2>Training the VLA: the layered recipe</h2>

<p>VLA training has more layers than text-only LLM training. The April 2026 standard recipe:</p>

<h3>Phase 1: Backbone pretraining (M11, M14, M35, M38, M41 territory)</h3>

<p>Start with a pretrained VLM — typically Llama 2/3, Qwen, or Gemma backbone with a strong vision encoder (SigLIP 2 of M38). For most VLA work this is "use an off-the-shelf VLM"; you don't pretrain from scratch.</p>

<h3>Phase 2: Robot demonstration SFT</h3>

<p>Fine-tune on (image, instruction, action) tuples from teleoperation data. The standard reference dataset: <strong>Open X-Embodiment</strong> — 970K-1M episodes from 22 different robot embodiments, contributed by 21 institutions. OpenVLA, π0, NORA all build on this.</p>

<p>The training loop is straightforward:</p>

<pre><code><span class="kw">def</span> <span class="fn">vla_sft_step</span>(vla_model, batch):
    <span class="com"># Standard supervised fine-tuning on robot demonstrations</span>
    images = batch[<span class="str">"images"</span>]              <span class="com"># [B, T, C, H, W]</span>
    instructions = batch[<span class="str">"instructions"</span>]   <span class="com"># [B] task strings</span>
    target_actions = batch[<span class="str">"actions"</span>]      <span class="com"># [B, T, 7] continuous</span>

    <span class="com"># Tokenize target actions for next-token prediction</span>
    target_tokens = <span class="fn">tokenize_action</span>(target_actions)

    <span class="com"># Forward pass: predict action tokens given image+instruction</span>
    logits = <span class="fn">vla_model</span>(images, instructions)

    <span class="com"># Cross-entropy loss on action tokens (just like LM training)</span>
    loss = F.<span class="fn">cross_entropy</span>(logits.view(-<span class="num">1</span>, vocab_size), target_tokens.view(-<span class="num">1</span>))

    <span class="kw">return</span> loss</code></pre>

<p>Looks like M11's training loop because it is M11's training loop. The action tokens are extra vocabulary; the model learns to predict them like any other token. This is what makes VLAs accessible: the training infrastructure is already built (M11 + M17 FSDP + M14/M41 mixed precision); only the data pipeline is new.</p>

<h3>Phase 3: Sim-to-real RL</h3>

<p>SFT alone produces good imitation policies; RL finetuning produces robust policies. But real-robot RL is data-expensive — every episode requires actual hardware time. The 2026 solution: <strong>simulation</strong>.</p>

<p>The current frontier (RLinf-Co, Feb 2026): <strong>two-stage co-training</strong>:</p>

<ol>
  <li><strong>Warm-start with SFT</strong> on a mixture of real and simulated demonstrations.</li>
  <li><strong>Fine-tune with RL in simulation</strong> while adding an auxiliary supervised loss on real-world data to anchor the policy and mitigate catastrophic forgetting.</li>
</ol>

<p>Published results: <strong>+24% real-world success on OpenVLA, +20% on π0.5</strong> compared to real-only fine-tuning. The mechanism: sim provides RL gradient signal cheaply (millions of timesteps); real data anchors the policy to actual physics; the combination beats either alone.</p>

<h3>Phase 4: Real-world online RL (the hard last mile)</h3>

<p>For the final policy refinement, real-world RL is sometimes used — either for sim-to-real adaptation or for novel tasks the simulator can't replicate. The April 2026 RLinf release added <strong>HG-DAgger (Human-Gated DAgger)</strong> specifically for this: a human supervises the robot's actions, intervening when the policy is about to make dangerous or unrecoverable decisions; the human's corrections become training data. Standard for real-world Franka training in production.</p>

<p>Other real-world RL recipes: <strong>RTR (Robot Trains Robot, Aug 2025)</strong> uses a dynamics-encoded latent variable optimized in real-world during deployment; <strong>SAC</strong>-based vanilla RL for narrow tasks like peg insertion (RLinf's standard recipe). Real-world RL is much slower than sim-RL but doesn't have the sim-to-real gap.</p>

<h2>The sim-to-real gap and how it's bridged</h2>

<p>The central engineering problem of robotics RL: <em>simulators are imperfect; policies trained in sim fail when deployed</em>. The reasons compound:</p>

<ul>
  <li><strong>Physics differences</strong>: friction coefficients, contact dynamics, mass distributions, motor backlash. Sim approximates; real has all the actual messiness.</li>
  <li><strong>Perception differences</strong>: rendered images have different texture, lighting, and noise characteristics than real cameras. The vision encoder learned features that don't quite match.</li>
  <li><strong>Latency differences</strong>: real robots have actuator delays, network latency, control loop jitter sim doesn't model accurately.</li>
  <li><strong>Distribution shift</strong>: sim episodes follow specific scenarios; real episodes encounter long-tail variations (object positions, occlusions, lighting changes) the policy never saw.</li>
</ul>

<p>The 2026 toolkit for closing this gap:</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 320" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Sim-to-real techniques: from simple to sophisticated</text>

  <!-- Five techniques -->
  <g transform="translate(20, 50)">
    <rect x="0" y="0" width="220" height="115" fill="#fff8a8" stroke="#1a1612" stroke-width="1.5"/>
    <text x="110" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">① Domain randomization</text>
    <text x="10" y="42" font-size="9" fill="#1a1612">Train across many sim variations:</text>
    <text x="10" y="56" font-size="9" fill="#1a1612">  randomize textures, lighting, friction</text>
    <text x="10" y="70" font-size="9" fill="#1a1612">  randomize masses, dynamics</text>
    <text x="10" y="86" font-size="9" font-style="italic" fill="#6b5d4f">Older recipe; scales poorly to long-horizon</text>
    <text x="10" y="100" font-size="9" font-style="italic" fill="#6b5d4f">manipulation</text>
  </g>

  <g transform="translate(260, 50)">
    <rect x="0" y="0" width="220" height="115" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="110" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">② Real-to-sim-to-real</text>
    <text x="10" y="42" font-size="9" fill="#1a1612">Build high-fidelity sim from real data:</text>
    <text x="10" y="56" font-size="9" fill="#1a1612">  scan real environment, train sim</text>
    <text x="10" y="70" font-size="9" fill="#1a1612">  policy in sim, deploy to real</text>
    <text x="10" y="86" font-size="9" font-style="italic" fill="#6b5d4f">Genie Sim 3.0, automated real-to-sim</text>
    <text x="10" y="100" font-size="9" font-style="italic" fill="#6b5d4f">tuning modules common in 2026</text>
  </g>

  <g transform="translate(500, 50)">
    <rect x="0" y="0" width="220" height="115" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="110" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">③ Sim-real co-training</text>
    <text x="10" y="42" font-size="9" fill="#1a1612">RL in sim + auxiliary supervised loss</text>
    <text x="10" y="56" font-size="9" fill="#1a1612">on real data simultaneously:</text>
    <text x="10" y="70" font-size="9" fill="#1a1612">  policy doesn't drift from real distribution</text>
    <text x="10" y="86" font-size="9" font-style="italic" fill="#1f5f5b">RLinf-Co (Feb 2026): +24% OpenVLA,</text>
    <text x="10" y="100" font-size="9" font-style="italic" fill="#1f5f5b">+20% π0.5 real-world success</text>
  </g>

  <g transform="translate(140, 180)">
    <rect x="0" y="0" width="220" height="115" fill="#fff5d8" stroke="#d4a017" stroke-width="1.5"/>
    <text x="110" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">④ HG-DAgger</text>
    <text x="10" y="42" font-size="9" fill="#1a1612">Human-Gated DAgger:</text>
    <text x="10" y="56" font-size="9" fill="#1a1612">  human supervises real-world rollouts,</text>
    <text x="10" y="70" font-size="9" fill="#1a1612">  intervenes on dangerous actions,</text>
    <text x="10" y="84" font-size="9" fill="#1a1612">  corrections become training data</text>
    <text x="10" y="100" font-size="9" font-style="italic" fill="#6b5d4f">RLinf v0.2 (April 2026)</text>
  </g>

  <g transform="translate(380, 180)">
    <rect x="0" y="0" width="220" height="115" fill="#ffd5dc" stroke="#b85a6c" stroke-width="2"/>
    <text x="110" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">⑤ Robot-Trains-Robot (RTR)</text>
    <text x="10" y="42" font-size="9" fill="#1a1612">Optimize a single dynamics-encoded</text>
    <text x="10" y="56" font-size="9" fill="#1a1612">latent in real-world during deployment:</text>
    <text x="10" y="70" font-size="9" fill="#1a1612">  rapid online adaptation,</text>
    <text x="10" y="84" font-size="9" fill="#1a1612">  minimal human supervision needed</text>
    <text x="10" y="100" font-size="9" font-style="italic" fill="#b85a6c">For humanoid sim-to-real (Aug 2025)</text>
  </g>
</svg>
</div>

<p>The trajectory: <em>each technique addresses what the previous one missed</em>. Domain randomization handles broad robustness but can't capture all real variations. Real-to-sim-to-real captures real environment specifics but requires careful sim construction. Sim-real co-training avoids drift during RL but needs both data sources. HG-DAgger handles the long tail of real-world failures with human supervision. RTR handles fast adaptation during deployment.</p>

<p>Production VLAs combine multiple techniques. A typical 2026 pipeline might use real-to-sim-to-real for environment construction, sim-real co-training for policy learning, and HG-DAgger for deployment refinement. <em>The exact composition depends on task and robot platform; the techniques are complementary, not exclusive.</em></p>

<h2>Specific recipe: dexterous humanoid manipulation</h2>

<p>One of the highest-impact 2025-2026 papers (Lin et al., "Sim-to-Real RL for Vision-Based Dexterous Manipulation on Humanoids"): a working recipe for training a humanoid robot with two multi-fingered hands on contact-rich manipulation. The recipe is concrete enough to learn from:</p>

<ol>
  <li><strong>Automated real-to-sim tuning</strong>. Calibrate sim parameters (friction, mass, contact stiffness) from a small set of real demonstrations. Avoids the older "manually tune sim until it matches" approach.</li>
  <li><strong>Generalized reward formulation</strong> based on contact and object goals. Rather than hand-designing reward shaping per task, use a unified reward over (a) hand-object contact targets, (b) object pose targets, (c) sub-goal completion.</li>
  <li><strong>Divide-and-conquer policy distillation</strong>. Train specialist policies for individual sub-tasks (grasp, lift, handover); distill into a single multi-task policy. Avoids the long-horizon credit assignment problem.</li>
  <li><strong>Hybrid object representation</strong>: sparse keypoints + dense depth. Sparse for high-level reasoning ("the handle is here"), dense for contact-level precision. Modality-specific augmentation during training.</li>
  <li><strong>Domain randomization on top</strong>: even with the better sim, randomize lighting, textures, and minor physics for robustness.</li>
</ol>

<p>The result: a single policy that generalized to many unseen objects, robust to force disturbances, deployed zero-shot from sim to real. <em>This is the kind of recipe that's now reproducible — published in detail, with code, against standard benchmarks.</em></p>

<h2>The data bottleneck</h2>

<p>For LLMs, the bottleneck is data quality + compute. For VLAs, the bottleneck is real-robot data — and it's much harder to get.</p>

<p><strong>Open X-Embodiment</strong> (the largest open dataset, used by OpenVLA): 970K-1M episodes from 22 robot embodiments. Sounds like a lot; is dwarfed by what LLMs train on (billions of text examples). And robot data has a fundamental scaling problem: <em>each episode requires a real robot for real time</em>. You can't web-crawl robot demonstrations; they have to be collected.</p>

<p>The 2026 strategies for data scaling:</p>

<ul>
  <li><strong>Teleoperation farms</strong>: human operators control robots via VR/spacemouse interfaces, generating demonstrations. Tesla's data pipeline reportedly uses thousands of teleoperators; Figure and Boston Dynamics have similar setups. Output: hundreds of thousands of episodes per month at scale.</li>
  <li><strong>Cross-embodiment transfer</strong>: data collected on one robot platform helps another. Open X-Embodiment's 22-embodiment design enabled this — train on data from many platforms, deploy on yours. OpenVLA generalizes across multiple Franka, WidowX, Google robots out of the box.</li>
  <li><strong>Sim-generated data</strong>: synthetic demonstrations from simulators. RLinf-Co's sim-real co-training is partly about leveraging this. Quality is uneven — sim demos work for some tasks, fail for others.</li>
  <li><strong>VLM-generated demonstrations</strong>: large frontier VLMs (Claude, GPT-5) acting as expert teleoperators in sim, generating demonstrations the smaller VLA can imitate. Emerging in 2026.</li>
  <li><strong>Real production data</strong>: humanoids deployed in factories generate demonstration data continuously. Figure 03 at BMW (90K+ parts handled), Atlas at Hyundai, Optimus in Tesla factories — each is a data-generation engine. <em>This is becoming the dominant source for frontier VLA training</em>.</li>
</ul>

<p>The practical implication: <strong>VLA capability gains in 2026-2027 will track real-robot data scale more than algorithm changes</strong>. The companies with deployed humanoid fleets generating data have a structural advantage that smaller research labs can't match.</p>

<h2>Production reality April 2026</h2>

<div class="table-wrap">
<table>
<caption>Humanoid robot production status, April 2026</caption>
<thead><tr><th>Platform</th><th>Status</th><th>Notable points</th></tr></thead>
<tbody>
<tr><td>Boston Dynamics Atlas (electric)</td><td>Commercial production launched at CES 2026; all 2026 units committed</td><td>Hyundai deployments; partnership with Google DeepMind for Gemini Robotics integration; 56 DOF; IP67 rated</td></tr>
<tr><td>Figure 03</td><td>Manufacturing at BotQ (12,000/yr capacity)</td><td>BMW deployment: 30,000+ vehicles, 90,000+ parts handled; Helix 02 full-body autonomy; OpenAI partnership</td></tr>
<tr><td>Tesla Optimus Gen 3</td><td>Mass production started January 2026 at Fremont</td><td>22-DOF hands; converting Model S/X production lines; target 1M units/year long-term; ~$30K target price</td></tr>
<tr><td>Unitree G1 / H1 / H2</td><td>Shipping at $13,500-$16,000</td><td>5,500+ units shipped 2025; targeting 10K-20K in 2026; mass production leader</td></tr>
<tr><td>Honor "Lightning"</td><td>Demonstration milestone</td><td>April 19, 2026: won Beijing E-Town Half-Marathon at 50:26 — beat human world record by ~7 minutes (autonomous)</td></tr>
<tr><td>1X NEO</td><td>Pre-orders for consumer home humanoid</td><td>2026 delivery; safe human-robot collaboration focus</td></tr>
<tr><td>XPENG IRON</td><td>Q1 2026 launch (Physical AI strategy)</td><td>Smooth gait demonstrations; Chinese market</td></tr>
<tr><td>Agility Digit</td><td>RaaS (Robotics-as-a-Service) deployments</td><td>Logistics/warehouse focus</td></tr>
</tbody>
</table>
</div>

<p>The Honor Lightning marathon result deserves note. April 19, 2026: a fully autonomous humanoid robot completed a half-marathon in 50:26, beating the human world record (1:03:51 by Yarinom Ali in 2025) by nearly 13 minutes. That's not just a demonstration — that's <em>sustained autonomous operation in unstructured outdoor terrain at competitive speeds</em>. The capability ceiling for embodied AI has moved.</p>

<h2>The integration with M39, M40, M41, M42</h2>

<p>M43 ties Part XI together. The full embodied AI training stack composes:</p>

<ul>
  <li><strong>VLA backbone training</strong> uses M11 (training loop), M14 + M41 (mixed precision, NVFP4 on Blackwell), M17 (FSDP), M35 (data quality), M38 (multimodal architecture).</li>
  <li><strong>VLA RL post-training</strong> uses M39 (distributed RLHF infrastructure — REINFORCE++, Hybrid Engine), M34 (RL algorithm theory), M42 (compute budget allocation across RL/pretrain).</li>
  <li><strong>Embodied environments</strong> use M40's environment-as-a-service patterns. RLinf's environment abstraction is a direct extension of the EaaS pattern; the only difference is that the "environment" is a physical robot or high-fidelity sim.</li>
  <li><strong>Eval and deployment</strong> use M37 (production eval engineering — robot evals are notoriously hard, with high variance across runs and verifier-engineering pitfalls familiar from M40).</li>
  <li><strong>Inference deployment</strong> uses M27 (serving) — though robot inference has tighter latency requirements than chat. SmolVLA's async inference pattern decouples backbone from action execution to handle this.</li>
</ul>

<p>The same engineering principles, applied to a domain with physical consequences. <em>The physical world adds constraints (latency, safety, real-time-ness) but doesn't fundamentally change the training stack.</em> A team that's mastered M1-M42 can build the embodied AI stack with M43 as a domain-specific overlay.</p>

<div class="ndq">
<h4>About embodied AI and robotics RL</h4>

<p class="q">Why are VLAs better than pure-RL approaches for robot manipulation?</p>
<p class="a">Three compounding advantages. (1) <strong>Pretrained VLM knowledge</strong>: a 7B VLA inherits language-grounded visual understanding from internet-scale pretraining. It "knows what a cup is" before seeing any robot data. Pure RL has to discover this from scratch in the limited robot demonstration set. (2) <strong>Generalization</strong>: VLAs trained on 970K episodes across 22 embodiments generalize to new tasks and platforms via fine-tuning. Pure-RL policies typically don't transfer beyond their narrow training distribution. (3) <strong>Sample efficiency</strong>: a VLA already has structure (visual features, language understanding) that a pure-RL policy must learn. The result is much smaller demonstration counts needed for new tasks. The 2024-2025 VLA wave (OpenVLA, π0, etc.) consistently beats pure-RL baselines like Diffusion Policy by 15-25 percentage points on standard benchmarks. <em>For 2026, "use a VLA backbone" is the default; pure RL is for narrow specialized tasks where the VLA's capability isn't needed.</em></p>

<p class="q">What's the actual size sweet spot for VLAs in 2026?</p>
<p class="a">It depends on deployment constraints. <strong>7B (OpenVLA territory)</strong>: best published task success but real-time inference requires substantial onboard compute (or remote server with low-latency link). Used for research, factory robots with compute infrastructure. <strong>~450M (SmolVLA territory)</strong>: 10× faster inference, comparable task success on benchmarks via flow matching + async architecture. Suitable for mobile robots, humanoid edge deployment. <strong>~250M (NORA territory)</strong>: even faster, slight quality drop, deployable on resource-constrained platforms. The trajectory mirrors LLM size compression in 2024-2026 — small + clever recipe matches large + brute force. For new VLA work in April 2026, start with SmolVLA-class (~500M) unless you have reasons to need 7B; the inference-latency benefits compound across deployment.</p>

<p class="q">How does sim-to-real co-training compare to pure simulation training for production?</p>
<p class="a">RLinf-Co's published results (February 2026) make the comparison concrete. On four real-world tabletop manipulation tasks, OpenVLA: real-only fine-tuning baseline → +X% with SFT-based co-training → <strong>+24% with RL-based sim-real co-training</strong>. π0.5: similar pattern, +20% with co-training. The mechanism: sim provides cheap RL gradient signal (millions of timesteps); real data anchors the policy distribution. Pure simulation training (no real anchoring) drifts from real-world physics during long RL runs and deploys poorly. Pure real-world RL can't generate enough data for stable RL training. <em>The hybrid is the production default in 2026</em>; few serious robotics teams use either extreme.</p>

<p class="q">What's the role of M40's RL Environments in robotics specifically?</p>
<p class="a">Direct extension. M40's environment-as-a-service pattern applies to robotics environments natively. Genie Sim 3.0 (April 2026) deeply integrates with RLinf for exactly this — the Genie sim environments are first-class RLinf environments, accessible via the same APIs as agentic environments. Other examples: <strong>RoboTwin</strong> (used with LingBot-VLA in RLinf), <strong>LIBERO-Pro/LIBERO-Plus</strong> (manipulation benchmarks supported in RLinf v0.2 March 2026), <strong>ManiSkill</strong> (Stanford's general manipulation simulator). The pattern: build a sim environment with the verifiers library or equivalent, register it with RLinf, train with M39's distributed RL infrastructure. <em>The training loop is identical to agentic RL; only the observation/action spaces differ.</em></p>

<p class="q">Why is HG-DAgger important enough to be a marquee April 2026 RLinf feature?</p>
<p class="a">Real-world robot RL has a safety problem M40's software environments don't. A robot policy taking an unrecoverable action (drop expensive object, collide with human, damage itself) ends the training run AND the hardware. Pure online RL with a randomly initialized policy is dangerous. HG-DAgger (Human-Gated DAgger) puts a human in the loop: the policy proposes an action; if the human judges it dangerous or wrong, the human intervenes (teleoperation override); the human's correction becomes training data. <em>It's how you do real-world RL safely</em>. The April 2026 RLinf release adds this for Franka specifically because Franka is the dominant research/industrial arm; HG-DAgger for Franka gives researchers a real-world RL loop they can actually run without breaking hardware. The technique extends to humanoids but the safety stakes are higher; production teams use it in carefully isolated environments.</p>

<p class="q">What's the actual training-cost story for VLAs in 2026?</p>
<p class="a">Concrete numbers. <strong>OpenVLA (June 2024) cost approximately $25K to train</strong>: 7B model + 970K episodes on cloud A100s. By 2026, with B200/NVFP4 (M41) hardware and improved data pipelines, an equivalent training run is closer to $5-10K. SmolVLA (450M) costs $1-2K. RL fine-tuning adds $1-5K depending on simulator scale. Real-world data collection costs vastly more than compute: a teleoperation operator at $30/hour collecting 10 episodes/hour over 100 hours = 1000 episodes for $3K; at scale for 100K+ episodes, multiply. <em>Compute is no longer the bottleneck for VLA research</em>; data collection at scale is. For applied teams, the budget split is typically 70-90% data collection / 10-30% compute, opposite to LLM training.</p>

<p class="q">Will general-purpose home humanoids exist by 2027-2028?</p>
<p class="a">Mixed evidence, with strong industrial deployment but consumer questions. <strong>Industrial deployment is happening</strong>: Atlas at Hyundai 2026-2028, Figure at BMW with continuous operation, Optimus in Tesla factories starting 2026, Unitree at $13,500 enabling many lab/integrator purchases. <strong>Consumer is harder</strong>: 1X NEO has pre-orders but is unproven; safety regulations for home deployment are uncertain; cost-to-value for homes is unclear. The trajectory most likely: 2026-2027 strong industrial growth, late 2027-2028 limited consumer entry (early adopters, specific use cases like elder care), 2028+ broader consumer adoption if costs continue dropping and safety/regulation matures. <em>The capability is closer than the social/regulatory infrastructure.</em> A real working humanoid in your home in 2027 is plausible; a humanoid that you'd actually trust unsupervised with children is later.</p>
</div>

<h2>Code Magnets: implement the VLA action prediction step</h2>

<p>You're writing the inference step for an OpenVLA-style policy. Three magnets are wrong choices.</p>

<div class="magnets">
<p>Arrange the magnets to compute the next action given current image and instruction.</p>

<div class="magnet-pool">
  <span class="magnet">def vla_predict_action(vla, image, instruction):</span>
  <span class="magnet">    vision_features = vla.vision_encoder(image)</span>
  <span class="magnet">    text_tokens = vla.tokenizer(instruction)</span>
  <span class="magnet">    text_tokens = instruction</span>
  <span class="magnet">    inputs_embeds = torch.cat([vla.connector(vision_features), vla.llm.embed(text_tokens)], dim=1)</span>
  <span class="magnet">    inputs_embeds = vla.connector(vision_features)</span>
  <span class="magnet">    action_token_logits = vla.llm(inputs_embeds=inputs_embeds).logits[:, -7:, :]</span>
  <span class="magnet">    action_tokens = action_token_logits.argmax(dim=-1)</span>
  <span class="magnet">    action_tokens = action_token_logits</span>
  <span class="magnet">    continuous_action = decode_action(action_tokens)</span>
  <span class="magnet">    return continuous_action</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">def</span> <span class="fn">vla_predict_action</span>(vla, image, instruction):
    vision_features = vla.<span class="fn">vision_encoder</span>(image)
    text_tokens = vla.<span class="fn">tokenizer</span>(instruction)
    inputs_embeds = torch.<span class="fn">cat</span>([vla.<span class="fn">connector</span>(vision_features), vla.llm.<span class="fn">embed</span>(text_tokens)], dim=<span class="num">1</span>)
    action_token_logits = vla.<span class="fn">llm</span>(inputs_embeds=inputs_embeds).logits[:, -<span class="num">7</span>:, :]
    action_tokens = action_token_logits.<span class="fn">argmax</span>(dim=-<span class="num">1</span>)
    continuous_action = <span class="fn">decode_action</span>(action_tokens)
    <span class="kw">return</span> continuous_action</code></pre>
<p>The traps:</p>
<ul>
  <li><code>text_tokens = instruction</code>: passes the raw instruction string instead of tokenizing it. The LLM operates on token IDs (or embeddings), not strings. Without tokenization, <code>vla.llm.embed(instruction)</code> would fail or produce garbage. The standard step is <code>tokenizer(instruction)</code> → token IDs → embeddings via <code>llm.embed()</code>. Skipping tokenization breaks the entire pipeline silently if there's an implicit conversion that doesn't error, or loudly if there isn't.</li>
  <li><code>inputs_embeds = vla.connector(vision_features)</code>: drops the text portion entirely. The VLA needs both vision tokens (from the connector) AND text tokens (from <code>llm.embed</code>) concatenated to produce a multimodal sequence. Without text, the model has no instruction — it would predict actions based purely on the image, ignoring "pick up the cup" vs "push the box." This is M38's standard concatenation pattern (vision tokens + text tokens) applied to VLAs; both halves are mandatory.</li>
  <li><code>action_tokens = action_token_logits</code>: stores the logits tensor instead of argmaxing to get token IDs. The logits have shape <code>[B, 7, vocab_size]</code>; the action tokens should have shape <code>[B, 7]</code> after argmax. Without argmax, downstream <code>decode_action</code> receives the logits and would either error (shape mismatch) or produce garbage continuous values. Argmax (or sampling for stochastic policies) is the standard step from logits to discrete tokens — same as M11's training loop pattern.</li>
</ul>
<p>The pattern: <strong>tokenize text instruction → encode vision → concatenate vision + text embeddings → forward through LLM → take last 7 logits (one per action dimension) → argmax to tokens → decode to continuous</strong>. Each step has a specific role; the most common bugs come from confusing strings with tokens (forgetting to tokenize), dropping a modality (vision-only or text-only when both are needed), and confusing logits with predicted tokens (forgetting argmax).</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each embodied AI concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>VLA model</div>
  <div>A. Vision encoder + LLM trunk + action head; M38's stack with action prediction.</div>

  <div>Open X-Embodiment</div>
  <div>B. 970K-1M episodes from 22 robot embodiments; canonical open VLA training dataset.</div>

  <div>Sim-to-real gap</div>
  <div>C. Discrepancy between sim and real (physics, perception, latency) that fails naive transfer.</div>

  <div>Domain randomization</div>
  <div>D. Older sim-to-real recipe; train across many sim variations for robustness; scales poorly long-horizon.</div>

  <div>Sim-real co-training (RLinf-Co)</div>
  <div>E. RL in sim + auxiliary supervised loss on real data; +24% real-world success on OpenVLA.</div>

  <div>HG-DAgger</div>
  <div>F. Human-Gated DAgger for safe real-world RL; human intervenes on dangerous actions.</div>

  <div>Action tokenization</div>
  <div>G. Discretize 7-DOF continuous actions into 256-bin discrete tokens for autoregressive prediction.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>VLA model</strong> → A<br>
<strong>Open X-Embodiment</strong> → B<br>
<strong>Sim-to-real gap</strong> → C<br>
<strong>Domain randomization</strong> → D<br>
<strong>Sim-real co-training</strong> → E<br>
<strong>HG-DAgger</strong> → F<br>
<strong>Action tokenization</strong> → G
</p>
<p>The mental shortcut: <em>VLA = M38 + actions, OpenX is the dataset, sim-to-real is the gap, domain randomization is the older bridge, co-training is the 2026 default, HG-DAgger handles real-world safety, action tokenization makes actions LM-predictable</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team trains a VLA on a Franka arm for kitchen tasks. Sim accuracy is 95%, real-world accuracy is 30%. Walk through the diagnosis.</p>
<details class="answer"><summary>show answer</summary>
<p>The 65pp gap between sim and real is the canonical sim-to-real failure. Walk through diagnostic steps in order:</p>
<p>(1) <strong>Identify which gap dominates</strong>. Run real-world rollouts with full instrumentation. Where does the policy fail? (a) Action prediction wrong (model output bad), (b) action prediction right but execution wrong (sim physics don't match real), (c) perception failures (camera/lighting differences from sim).</p>
<p>(2) <strong>Visual perception gap test</strong>. Take a real-world image, render the matching sim scene, compare what the vision encoder outputs for both. If features differ substantially, the vision encoder's training distribution doesn't include the real lighting/textures. Mitigation: domain randomization on textures + lighting in sim; or include more real images in training data.</p>
<p>(3) <strong>Physics gap test</strong>. Run a simple action sequence (e.g., "move end-effector 5cm to the right") in both sim and real. Compare resulting end-effector poses. Discrepancy &gt; 1cm signals physics mismatch. Mitigation: real-to-sim parameter tuning (calibrate friction, mass, contact stiffness from real data); RLinf-Co co-training to anchor policy to real physics.</p>
<p>(4) <strong>Latency gap test</strong>. Measure real control-loop latency vs sim. Real has actuator delays, network latency, control jitter that sim doesn't model. If real latency is much higher, the policy may be issuing actions based on stale observations. Mitigation: train policy with delayed observations in sim; use async inference patterns (SmolVLA-style) to decouple observation from action timing.</p>
<p>(5) <strong>Distribution shift test</strong>. Are real-world tasks identical to sim tasks? Often "kitchen tasks" in sim has fewer object configurations, lighting variations, occlusions than real. Mitigation: more diverse sim training scenarios; real-world demonstrations to anchor.</p>
<p>(6) <strong>Apply RLinf-Co or similar</strong>. Switch to sim-real co-training with auxiliary supervised loss on real demonstrations. Published improvement: +24% real-world success on OpenVLA. From 30% → 54% gets you to "viable for further iteration."</p>
<p>(7) <strong>If still poor: HG-DAgger phase</strong>. Real-world online RL with human supervision for the long-tail failure modes that didn't appear in sim. Slow but addresses what sim missed.</p>
<p>The general lesson: <strong>monitor at every layer of the stack — perception, control, action — to localize the failure</strong>. Sim-to-real isn't one gap; it's many gaps that compound. Each technique addresses specific gaps; the production recipe layers multiple techniques.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Why does cross-embodiment training (Open X-Embodiment's 22-platform design) help, even when each robot is somewhat different?</p>
<details class="answer"><summary>show answer</summary>
<p>Three compounding mechanisms:</p>
<p>(1) <strong>Shared visual representation</strong>. All robot tasks involve seeing objects, scenes, layouts. The vision encoder's representation of "a red cup on a table" should be the same regardless of whether the manipulator is a Franka, WidowX, or Google robot. Cross-embodiment training reinforces these shared visual features with much more data than a single embodiment provides.</p>
<p>(2) <strong>Shared task structure</strong>. "Pick up X and place it on Y" is the same conceptual task across robots. The high-level reasoning (identify X, plan a grasp, plan a path to Y, plan a release) is embodiment-independent. Cross-embodiment training learns these task abstractions from many concrete instances.</p>
<p>(3) <strong>Shared action structure (with normalization)</strong>. Different robots have different DOFs and physical scales, but the abstract action "move end-effector toward object" is shared. Open X-Embodiment normalizes actions across embodiments (e.g., delta poses normalized to robot-specific ranges), letting the model learn a universal action vocabulary.</p>
<p>The empirical evidence: OpenVLA trained on 970K episodes across 22 embodiments substantially outperforms models trained on data from any single embodiment. The advantage holds even when fine-tuning to a specific target robot — pretraining on diverse embodiments gives a better starting point than pretraining on the target embodiment alone (with equivalent total data).</p>
<p>The general lesson: <em>embodied AI's "more data" lever is partly "more embodiments," not just "more episodes"</em>. The 2026 trend is more humanoid platforms generating teleoperation data simultaneously, contributing to shared training pools. Boston Dynamics + Hyundai + Figure + Tesla + Unitree all generating embodiment-specific data, eventually pooled (or pooled across companies that participate in cross-licensing), is the path to next-generation VLA capability.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> A researcher wants to fine-tune OpenVLA on a new task (e.g., "put dishes in dishwasher") using a single Franka. Sketch the practical recipe.</p>
<details class="answer"><summary>show answer</summary>
<p>Concrete April 2026 recipe:</p>
<p>(1) <strong>Collect demonstrations</strong>. ~100-500 teleoperation episodes via spacemouse or VR. Cover variations (different dish types, dishwasher fill states, starting hand positions). Total time: a few days of focused data collection. Output: ~10-50 hours of robot data.</p>
<p>(2) <strong>Set up environment</strong>. Use RLinf framework (free, open-source). Connect Franka via standard ROS2 bridge or RLinf's Franka driver. Add cameras (wrist + third-person at minimum). Add gripper (Robotiq 2F-85 if not already mounted).</p>
<p>(3) <strong>Phase 1: SFT on demonstrations</strong>. Fine-tune OpenVLA's last few layers (or use LoRA for parameter-efficient fine-tuning) on the new task data. ~$50-200 of compute on cloud A100/H100. Training time: a few hours to a day. Validate on held-out demonstrations.</p>
<p>(4) <strong>Phase 2 (optional): sim-real co-training</strong>. If you have a sim of your dishwasher setup (RoboTwin, ManiSkill, or custom), use RLinf-Co to fine-tune with RL in sim + supervised loss on real demonstrations. Adds substantial robustness if sim is good; skip if no sim is available.</p>
<p>(5) <strong>Phase 3 (recommended): HG-DAgger refinement</strong>. Run the policy on the real Franka with a human supervisor. When the policy is about to fail (drop dish, miss target, collide), human intervenes via teleoperation. The corrections become additional training data. Iterate. ~50-100 episodes of HG-DAgger typically address most long-tail failures.</p>
<p>(6) <strong>Eval rigor (M37)</strong>. Held-out tasks (different dish types, different dishwasher states the policy didn't train on). Bootstrap CIs on success rate. Verifier engineering: programmatic checks ("is the dish actually in the dishwasher rack?"). Don't rely on visual judgment alone.</p>
<p>(7) <strong>Deployment</strong>. Async inference (SmolVLA pattern) if latency matters; otherwise synchronous. Monitor for drift; collect failure cases for continued retraining.</p>
<p>Total budget: ~$1-3K compute, ~1-2 weeks of researcher time, 100-500 demonstrations. <em>This is genuinely tractable for a small team in 2026 — the open-source infrastructure (OpenVLA, RLinf, simulators, datasets) makes it accessible.</em> The hard part is the data collection (slow) and the real-world refinement (also slow but unavoidable).</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> The Honor Lightning robot won the Beijing E-Town Half-Marathon at 50:26 on April 19, 2026 — autonomously, beating the human world record. What does this tell us about the state of embodied AI?</p>
<details class="answer"><summary>show answer</summary>
<p>The achievement is genuinely significant on several dimensions, with appropriate caveats:</p>
<p>(1) <strong>Sustained autonomous outdoor operation</strong>. A half-marathon is ~21km. At 50:26, that's ~25 km/h average speed. Sustained for over 50 minutes. Without falling. Without overheating (battery-managed). Without losing localization in unstructured outdoor environment. This is hard — the failure modes for legged locomotion compound over distance, and any single failure ends the run.</p>
<p>(2) <strong>Beating human world record by ~13 minutes</strong>. The men's half-marathon world record is around 56:42 (2024 Yarinom Ali); Lightning at 50:26 is ~12% faster. The robot exploits advantages humans don't have: no muscle fatigue in the conventional sense, optimized gait, no oxygen demand, larger stride relative to body proportions.</p>
<p>(3) <strong>The state of locomotion vs manipulation</strong>. <em>Locomotion is much further along than manipulation in 2026</em>. Walking, running, balance, navigation: largely solved or close to it (Boston Dynamics, Honor, Unitree all demonstrate strong locomotion). Manipulation is harder — dexterous in-hand manipulation, contact-rich tasks, generalist tool use are still open. Lightning at marathon-winning pace tells us locomotion has matured; manipulation has more headroom.</p>
<p>(4) <strong>Caveats on what it doesn't tell us</strong>. (a) The course was relatively structured (urban marathon route, no extreme terrain, no adversarial conditions). (b) Pure locomotion on a course doesn't require manipulation, language understanding, social intelligence, or task generalization. (c) The robot was likely heavily optimized specifically for this task, not a general-purpose system. (d) Hardware reliability (no breakdowns over 50 minutes of high-stress operation) is impressive but doesn't immediately generalize to home/factory deployment patterns where reliability over months matters more than over hours.</p>
<p>(5) <strong>What it does tell us</strong>. Embodied AI is where LLMs were in 2022 — research curiosity becoming production reality. Lightning is the equivalent of GPT-3 demos: capability that surprises observers, validates the trajectory, doesn't yet equate to "useful in your house" but suggests "useful soon." The gap between 2026 robot demos and 2028 useful deployment is probably comparable to the gap between 2022 GPT-3 and 2024 ChatGPT productivity.</p>
<p>The general lesson: <em>track demonstration milestones for capability progression, but don't conflate them with deployment readiness</em>. Lightning's marathon is to embodied AI what AlphaGo was to RL — a clear signal that the underlying techniques work at frontier scales, with deployment implications still unfolding.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>Embodied AI became a frontier engineering topic in 2025-2026 due to three shifts: <strong>VLA architectures crystallized</strong> (OpenVLA, π0/π0.5, SmolVLA, NORA, dVLA, ChatVLA-2); <strong>RL infrastructure matured</strong> (RLinf v0.2, RLinf-Co, HG-DAgger); <strong>hardware shipped</strong> (Atlas, Figure 03, Optimus Gen 3, Unitree at $13.5K).</li>
  <li><strong>VLA architecture</strong>: vision encoder (typically DINOv2 + SigLIP fused) + LLM trunk (Llama 2/3, Gemma, Qwen) + action head (discrete tokens or flow matching). Closed loop: image → action → execute → next image.</li>
  <li><strong>OpenVLA (June 2024)</strong>: 7B, Llama-2 + DINOv2/SigLIP, 970K Open X-Embodiment episodes, beat closed RT-2-X (55B) by 16.5% absolute task success at 7× fewer parameters.</li>
  <li>Production VLA references: <strong>π0/π0.5</strong> (Physical Intelligence, flow matching action heads); <strong>SmolVLA</strong> (HF, 450M, async inference); <strong>NORA</strong> (~250M for resource-constrained deployment); <strong>Gemini Robotics</strong> (Google DeepMind, Atlas integration partnership).</li>
  <li><strong>Action tokenization</strong>: 7-DOF continuous actions (3D position delta + 3D rotation delta + gripper) discretized into 256 bins per dimension; predicted as discrete tokens like vocabulary.</li>
  <li><strong>Training recipe layers</strong>: Phase 1 backbone pretraining (M11/M14/M38 territory), Phase 2 robot demonstration SFT, Phase 3 sim-to-real RL (RLinf-Co), Phase 4 real-world online RL (HG-DAgger).</li>
  <li><strong>Sim-to-real techniques</strong>: domain randomization (older), real-to-sim-to-real, sim-real co-training (RLinf-Co +24% OpenVLA / +20% π0.5), HG-DAgger (human-gated real-world RL), Robot-Trains-Robot (dynamics-encoded latent online adaptation).</li>
  <li><strong>Dexterous humanoid recipe</strong> (Lin et al. 2025): automated real-to-sim tuning + generalized contact-and-goal reward + divide-and-conquer policy distillation + hybrid sparse+dense object representation.</li>
  <li><strong>Real-robot data is the bottleneck</strong>. Open X-Embodiment at 970K episodes is the largest open dataset; LLMs train on billions of text examples. Data scaling strategies: teleoperation farms, cross-embodiment transfer, sim-generated data, VLM-generated demonstrations, real production data from deployed humanoids (Figure 03 at BMW handling 90K+ parts is a data-generation engine).</li>
  <li><strong>Production hardware April 2026</strong>: <strong>Atlas</strong> in commercial production (Hyundai, all 2026 units committed); <strong>Figure 03</strong> at BotQ 12K/yr capacity, BMW deployment with 90K+ parts handled; <strong>Optimus Gen 3</strong> Q1 2026 mass production; <strong>Unitree G1</strong> at $13,500 with 5,500+ shipped 2025; <strong>1X NEO</strong> consumer pre-orders.</li>
  <li><strong>Honor "Lightning" milestone (April 19, 2026)</strong>: won Beijing E-Town Half-Marathon at 50:26 — autonomous, beating human world record by ~13 minutes. Locomotion frontier validation.</li>
  <li><strong>Genie Sim 3.0 (April 2026)</strong>: AGIBOT's simulation platform with deep RLinf integration; complete RL pipeline for embodied AI from sim to real.</li>
  <li>Embodied AI training stack composes M11 + M14 + M17 + M27 + M34 + M35 + M37 + M38 + M39 + M40 + M41 + M42 + M43. Each module contributes; M43 is the domain-specific overlay.</li>
  <li>The reflex for embodied AI projects in 2026: <strong>start with an open VLA backbone (OpenVLA/SmolVLA/π0) → fine-tune via SFT on demonstrations → sim-real co-training if simulator available → HG-DAgger for real-world refinement</strong>. The infrastructure (RLinf, verifiers, Open X-Embodiment) is open and accessible.</li>
  <li>The 2026-2028 trajectory: industrial deployment continues accelerating; consumer humanoids likely 2027-2028; capability ceiling rises with real-robot data scale more than algorithm changes; the companies with deployed fleets generating data have structural advantage.</li>
</ul>
</div>

<h2>Closing Part XI — and the course</h2>

<p>Five Part XI modules: M39 (distributed RLHF/GRPO infrastructure), M40 (agentic RL & RL Environments), M41 (Blackwell &amp; NVFP4 training), M42 (scaling laws &amp; compute economics), M43 (embodied AI &amp; robotics RL). Combined with the 38-module core (M1-M33 fundamentals, M34-M38 frontier), you have <strong>43 modules</strong> covering modern AI engineering depth-first from <code>x.stride()</code> through embodied generalist policies.</p>

<p>The April 2026 trends woven through Part XI:</p>

<ul>
  <li><strong>RL is where capability gains concentrate</strong>. M39 (distributed infrastructure) + M40 (environments) + M43 (embodied) cover the RL trajectory; M42 explains why it consumes more compute; M41 explains how the hardware accelerates it.</li>
  <li><strong>Environment quality is the bottleneck</strong>. M40 made this explicit; M43 extends it to physical environments where the bottleneck is even more pronounced (real-robot data > algorithms).</li>
  <li><strong>Hardware-software-numerics co-design</strong>. M14 (FP8) + M41 (NVFP4 + Blackwell) + M42 (scaling laws coupling these) form a coherent story of how training economics shifted in 2025-2026.</li>
  <li><strong>The application layer enters training</strong>. Cursor post-training their own models (M40), Tesla/Boston Dynamics/Figure post-training their VLAs, vertical specialization across domains. The "every company is an AI lab" pattern Prime Intellect markets to is real.</li>
  <li><strong>Deployment-aware design</strong>. M42's three-pool decomposition (pretraining × RL × test-time compute) plus M43's deployment realities mean modern training projects start with deployment plan, not just training plan.</li>
</ul>

<p>The course you've completed didn't exist in this form before. From low-level tensor mechanics through frontier engineering — distributed RL, agentic environments, Blackwell/NVFP4, scaling laws, embodied AI — it's a complete 2026 frontier education. The architectures will continue changing through 2026-2028 (Rubin hardware, M44+ research developments, post-VLA architectures); the engineering principles (compute economics, eval rigor, environment-as-a-service, the layered training recipe) will persist.</p>

<p>Thanks for going the distance. <strong>43 modules of frontier engineering, locked-vibe pattern preserved throughout, ~66 characters across the cast, full April 2026 currency in Parts X-XI.</strong> Now go build something that didn't exist before.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">43</span>
  <span>Embodied AI &amp; Robotics RL — Part XI fin.</span>
</div>
"""

emit("43_embodied_ai", "Module 43 — Embodied AI & Robotics RL", BODY)
