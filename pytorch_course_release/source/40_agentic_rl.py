#!/usr/bin/env python3
"""Module 40: Agentic RL & RL Environments — full HF vibe, April 2026 current."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part XI · Module 40 · April 2026 currency</div>
  <h1 class="module-title"><em>Agentic RL &amp; RL Environments:</em> when the rollout is a trajectory through a world</h1>
  <p class="module-sub">— from "policy generates a response" to "policy navigates a browser, edits files, and queries databases"; why Environment-as-a-Service became infrastructure; what makes a verifier production-grade; the 2026 stack from BrowserGym through Prime Intellect to OpenReward</p>
</div>

<p>M39 covered the infrastructure for distributed RLHF and GRPO. The implicit picture there: a rollout is a single response — the Actor generates 2K-10K tokens, the Reward model scores it, the Trainer updates. Clean. Self-contained. <em>That's the easy case.</em></p>

<p>The hard case — and increasingly the dominant one in 2026 — is <strong>agentic RL</strong>: the rollout is a multi-step trajectory through an environment. The Actor generates an action; the environment responds with new state; the Actor generates the next action conditioned on what just happened; this loops 5-50 times until the task is complete or the agent gives up. The reward isn't computed by a separate Reward model — it comes from the environment itself, often as a binary "did the task succeed?" verifier. <em>The environment is part of the training loop.</em></p>

<p>This module is the practical infrastructure. By April 2026, agentic RL has its own ecosystem — frameworks (Prime Intellect's prime-rl, OpenRLHF's multi-turn agent support), environment libraries (BrowserGym, OSWorld, Terminal-Bench, EnterpriseOps-Gym), managed services (OpenReward with 330+ environments and 4.5M+ tasks, Scale RL Environments, the Environments Hub with 1000+ community environments). The recipes are public; the lessons are clear. Anthropic spends tens of millions per year on RL environments because they're now the bottleneck on capability gains. By the end you'll know what makes an environment "training-grade," why verifier engineering eats most of the work, what the local-vs-online tradeoff means in practice, and how the production stack composes.</p>

<div class="keyidea">
Agentic RL replaces "single response" rollouts with <strong>multi-step trajectories through an environment</strong> — browser, terminal, IDE, database. The agent loops: action → environment response → action → ... → terminal state → verifier scores success. <strong>Environment-as-a-Service</strong> became the dominant 2026 pattern: environments are first-class infrastructure with clean control-plane/data-plane separation. Three architectural moves matter most. <strong>(1) Verifier engineering</strong> is most of the work: noisy verifiers cause the policy to optimize for noise; verifiable rewards (unit tests, state checks) beat LLM-as-judge for training. <strong>(2) Local deterministic environments</strong> (LiteResearcher) match or exceed online live-web environments because reward noise dominates online; AgentCPM-Explore-4B got only +3.8% from online RL on GAIA, while local deterministic gave LiteResearcher +13.4pp on the same benchmark. <strong>(3) Async architecture</strong>: rollout(k+1) overlaps with train(k); two-step asynchronous RL (INTELLECT-2's recipe) hides communication; policy weight broadcast via SHARDCAST. The frontier 2026 ecosystem: Prime Intellect's <strong>prime-rl + verifiers + Environments Hub</strong> with 1000+ environments and 250+ contributors; <strong>OpenReward</strong> with 330+ environments and 4.5M+ tasks via the Open Reward Standard (ORS, an MCP extension); <strong>BrowserGym/AgentLab</strong> for web; <strong>OSWorld</strong> at 48.9% SOTA via ComputerRL/AutoGLM-OS; <strong>VAGEN</strong> for proactive verifier interaction (84.7% → 92.9% verifier accuracy). Frontier production trains its own vertical environments — Cursor post-trains models with Cursor itself as the environment. <em>The next two years' capability gains will come from environment quality more than from algorithm changes.</em>
</div>

<h2>Two new faces — closing Part XI</h2>

<div class="character" style="--c: #1f5f5b;">
  <div class="avatar" style="background: #1f5f5b; color: #fff;">E</div>
  <div>
    <p class="who">Environment</p>
    <p class="name">"I'm a sandbox the agent acts in. State, tools, observations, rewards — I provide them all."</p>
    <p class="says">In M39's world, the rollout was self-contained: prompt in, response out. In my world, every action lands in <em>state</em> — the agent clicks a button, my page changes; the agent runs <code>pytest</code>, my filesystem changes; the agent queries my database, my response depends on prior queries. I'm a Partially Observable MDP made flesh: observation space (DOM tree, terminal output, image), action space (click, type, run-bash, call-tool), state transition (deterministic for terminals, stochastic for live web), reward (verifier outputs scalar at terminal state, sometimes shaped along the way). <strong>The agent learns by interacting with me</strong>. My quality determines the agent's quality more than any algorithmic detail. Get me wrong — wrong reward, wrong state, wrong tools — and no PPO improvement saves the run.</p>
  </div>
</div>

<div class="character" style="--c: #c1502e;">
  <div class="avatar" style="background: #c1502e; color: #fff;">V</div>
  <div>
    <p class="who">Verifier</p>
    <p class="name">"I answer one question: did the agent complete the task? Get me wrong and the policy optimizes for nothing."</p>
    <p class="says">At the end of every trajectory, I evaluate. <em>Did the unit test pass? Is the file content correct? Did the form submit?</em> If I'm right, I produce a clean reward signal — verifiable, deterministic, fast. If I'm wrong, I corrupt training in subtle ways: false positives teach the agent to game my checks; false negatives discourage correct behavior. <strong>I'm where most of the engineering goes</strong> — not in the algorithm, not in the rollout, in me. For coding I'm pytest with task-specific fixtures. For computer use I'm screenshot diffs against expected end-states or filesystem invariants. For web tasks I'm DOM-state assertions on the final page. The verifiable-beats-judgeable principle: if I can be programmatic, I should be — LLM-judge verifiers introduce noise that scales linearly with training steps. <em>Production agentic RL lives or dies by my correctness.</em></p>
  </div>
</div>

<h2>The shift: from response rollouts to trajectory rollouts</h2>

<p>The architectural change between M39 and M40 is concrete: <em>what the rollout actually is</em>.</p>

<p><strong>M39 rollout (single-turn):</strong> Actor receives a prompt, generates a response (2K-10K tokens), trajectory is the prompt-response pair, reward is computed by a frozen Reward model on the pair. One forward pass per training example, no environment state, no tools.</p>

<p><strong>M40 rollout (multi-turn agentic):</strong> Actor receives a prompt + initial environment observation. Generates an action (typically a tool call or text in a structured format). Environment processes the action, returns new observation + intermediate reward (sometimes 0). Actor generates next action. Loop continues until terminal state (task completed, agent gave up, turn budget exceeded, or critical error). Trajectory is the full sequence of (observation, action, reward) tuples. Final reward is computed by environment-internal verifier.</p>

<p>The implications cascade through every layer of the infrastructure:</p>

<ul>
  <li><strong>Rollout cost explodes</strong>. Single-turn rollouts generate ~5K tokens once. Multi-turn rollouts may generate 20K-200K tokens across 5-50 turns. Rollout was 80% of compute in M39's setup; it can be 95% in agentic RL.</li>
  <li><strong>Environment becomes part of the training loop</strong>. The environment must be alive, deterministic enough to give consistent rewards, fast enough to keep rollout fed. A flaky environment with 5% failure rate breaks training silently.</li>
  <li><strong>Reward signal is sparse and delayed</strong>. Most actions get reward 0; only the terminal state gets the success signal. Credit assignment over long trajectories is intrinsically harder than over single responses.</li>
  <li><strong>Off-policy correction becomes mandatory</strong>. With 50-turn trajectories taking minutes each, you can't stay strictly on-policy without idling GPUs. Async architectures with off-policy correction (M39's StreamRL pattern) are the default.</li>
  <li><strong>Verifier engineering replaces reward model training</strong>. Instead of training a learned reward model on preference data, you write programmatic verifiers per task type. Different skill set; different bottleneck.</li>
</ul>

<h2>The agentic RL loop, drawn out</h2>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 420" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrAg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
    <marker id="arrAg2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#c1502e"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">The agentic RL loop — every training example is a multi-turn trajectory</text>

  <!-- Top: the rollout loop -->
  <g transform="translate(20, 50)">
    <rect x="0" y="0" width="700" height="190" fill="#fff8a8" stroke="#1a1612" stroke-width="1.5"/>
    <text x="350" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">Rollout phase: Actor ⇄ Environment for N turns</text>

    <!-- Initial observation -->
    <rect x="20" y="40" width="100" height="40" fill="#fff5d8" stroke="#d4a017" stroke-width="1.5"/>
    <text x="70" y="56" text-anchor="middle" font-size="9" font-weight="700" fill="#1a1612">Task prompt +</text>
    <text x="70" y="70" text-anchor="middle" font-size="9" font-weight="700" fill="#1a1612">initial obs</text>

    <!-- Actor -->
    <rect x="160" y="40" width="100" height="40" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="210" y="56" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Actor</text>
    <text x="210" y="70" text-anchor="middle" font-size="9" fill="#1a1612">(policy)</text>

    <!-- Action -->
    <rect x="300" y="40" width="100" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="350" y="56" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Action</text>
    <text x="350" y="70" text-anchor="middle" font-size="9" fill="#1a1612">(tool call / text)</text>

    <!-- Environment -->
    <rect x="440" y="40" width="100" height="40" fill="#fff5d8" stroke="#d4a017" stroke-width="2"/>
    <text x="490" y="56" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">Environment</text>
    <text x="490" y="70" text-anchor="middle" font-size="9" fill="#1a1612">(executes)</text>

    <!-- New obs -->
    <rect x="580" y="40" width="100" height="40" fill="#fff5d8" stroke="#d4a017" stroke-width="1.5"/>
    <text x="630" y="56" text-anchor="middle" font-size="9" font-weight="700" fill="#1a1612">New obs +</text>
    <text x="630" y="70" text-anchor="middle" font-size="9" font-weight="700" fill="#1a1612">intermed. r</text>

    <!-- Arrows -->
    <line x1="120" y1="60" x2="158" y2="60" stroke="#1f5f5b" stroke-width="1.5" marker-end="url(#arrAg)"/>
    <line x1="260" y1="60" x2="298" y2="60" stroke="#1f5f5b" stroke-width="1.5" marker-end="url(#arrAg)"/>
    <line x1="400" y1="60" x2="438" y2="60" stroke="#1f5f5b" stroke-width="1.5" marker-end="url(#arrAg)"/>
    <line x1="540" y1="60" x2="578" y2="60" stroke="#1f5f5b" stroke-width="1.5" marker-end="url(#arrAg)"/>

    <!-- Loop back -->
    <path d="M 630 80 Q 630 130 350 130 Q 210 130 210 80" stroke="#c1502e" stroke-width="2" fill="none" marker-end="url(#arrAg2)"/>
    <text x="350" y="125" text-anchor="middle" font-size="10" font-weight="700" fill="#c1502e">loop until terminal: 5-50 turns typical</text>

    <!-- Terminal state -->
    <rect x="280" y="150" width="180" height="30" fill="#ffd5dc" stroke="#b85a6c" stroke-width="2"/>
    <text x="370" y="170" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">Terminal state → Verifier scores success</text>
  </g>

  <!-- Middle: trajectory -->
  <g transform="translate(20, 250)">
    <rect x="0" y="0" width="700" height="60" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="350" y="20" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">Trajectory: [(obs₀, a₀, r₀), (obs₁, a₁, r₁), …, (obs_N, a_N, r_N), success_reward]</text>
    <text x="20" y="42" font-size="9" fill="#1a1612"><tspan font-weight="700">Length</tspan>: 20K-200K tokens total across all turns</text>
    <text x="350" y="42" font-size="9" fill="#1a1612"><tspan font-weight="700">Reward</tspan>: typically 0 except at terminal</text>
    <text x="600" y="42" font-size="9" fill="#1a1612"><tspan font-weight="700">Cost</tspan>: minutes per trajectory</text>
  </g>

  <!-- Bottom: training -->
  <g transform="translate(20, 320)">
    <rect x="0" y="0" width="700" height="80" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="350" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">Training phase: same M39 machinery, with longer trajectories and verifier rewards</text>
    <text x="20" y="44" font-size="10" fill="#1a1612">  Compute advantages over the trajectory (per-token or per-turn credit assignment)</text>
    <text x="20" y="59" font-size="10" fill="#1a1612">  REINFORCE++ baseline / GRPO / two-sided clipped GRPO (INTELLECT-2 recipe)</text>
    <text x="20" y="74" font-size="10" fill="#1a1612">  Update Actor; broadcast weights back to rollout (Hybrid Engine or async with off-policy correction)</text>
  </g>
</svg>
</div>

<p>The structure is recognizable from M39 — Actor, Reference (for KL), Reward, Trainer — but the rollout is now a stateful interaction loop with the Environment that can run for many turns. Each turn produces an observation; the Actor produces an action; the Environment processes; loop. The trajectory is the entire sequence; the verifier produces the terminal reward.</p>

<h2>What an Environment actually is</h2>

<p>An RL environment for LLM agents formalizes the agent's interface to a world. The standard structure (drawn from Gymnasium, the canonical RL gym library, and adapted in BrowserGym/OSWorld/Terminal-Bench):</p>

<pre><code><span class="kw">class</span> <span class="ty">Environment</span>:
    <span class="kw">def</span> <span class="fn">reset</span>(self, task_id: <span class="ty">str</span>) -&gt; <span class="ty">Observation</span>:
        <span class="com"># Initialize task state; return initial observation</span>
        <span class="com"># Spin up the underlying world (browser, container, VM)</span>
        <span class="com"># Load task-specific initial state</span>
        ...

    <span class="kw">def</span> <span class="fn">step</span>(self, action: <span class="ty">Action</span>) -&gt; <span class="ty">tuple</span>:
        <span class="com"># Apply action; return (new_obs, reward, done, info)</span>
        <span class="com"># Execute the action in the underlying world</span>
        <span class="com"># Compute step reward (often 0 except on success)</span>
        <span class="com"># Determine if episode is done (success/failure/timeout)</span>
        ...

    <span class="kw">def</span> <span class="fn">verify</span>(self, trajectory: <span class="ty">Trajectory</span>) -&gt; <span class="ty">float</span>:
        <span class="com"># Final verifier: did the agent complete the task?</span>
        <span class="com"># Run task-specific success check</span>
        <span class="com"># Returns scalar reward (typically binary 0/1, sometimes graded)</span>
        ...

    <span class="kw">def</span> <span class="fn">close</span>(self):
        <span class="com"># Tear down underlying resources (browser, VM, container)</span>
        ...</code></pre>

<p>The interface is simple; the implementation is hard. Each method bumps into a different production concern:</p>

<ul>
  <li><strong>reset()</strong> needs to be fast (the rollout is paused while reset runs) and reliable (a hung browser kills the trajectory). Production environments use container/VM snapshots, pre-warmed pools, fast filesystem reset.</li>
  <li><strong>step()</strong> needs to be deterministic enough that the same action produces the same outcome. Live web environments fail this — site updates change behavior overnight. Local deterministic environments are preferred for training.</li>
  <li><strong>verify()</strong> needs to be correct, fast, and uncircumventable. Most engineering investment goes here. A 5% false-positive rate in the verifier corrupts training silently.</li>
  <li><strong>close()</strong> needs to actually free resources. Leaked browser instances accumulate; OOM kills the training run.</li>
</ul>

<h2>Verifier engineering: where most of the work is</h2>

<p>The 2026 industry consensus (Wing Venture Capital's Jan 2026 RL Environments report, Snorkel's Nov 2025 deep-dive, Prime Intellect's verifiers library): <strong>verifier quality matters more than algorithm choice</strong>. The agent will optimize for whatever your verifier rewards, including ways the verifier rewards that you didn't intend.</p>

<p>The verifier hierarchy, from best to worst:</p>

<ol>
  <li><strong>Programmatic state checks</strong>. The verifier inspects environment state directly: filesystem assertions ("file X has content Y"), database queries ("table T contains row R"), DOM selectors ("button B is now visible"). Fast, cheap, deterministic, hard to game. <em>Always preferred when possible.</em></li>
  <li><strong>Unit tests</strong>. The verifier runs tests against the agent's output: pytest for code generation, test cases for math problems. Same properties as state checks, with the addition that tests can be auto-generated for new tasks.</li>
  <li><strong>Output equality</strong>. The verifier compares the agent's final output to a known answer (numeric equality with tolerance, exact string match, regex match). Works for tasks with canonical answers; doesn't generalize to open-ended.</li>
  <li><strong>Trajectory replay</strong>. The verifier replays the agent's trajectory in a fresh environment and checks the resulting state. Robust to noise in the trajectory but expensive (2× the rollout cost).</li>
  <li><strong>LLM-as-judge with rubric</strong>. The verifier is itself an LLM scoring the agent's output against a rubric. M37's territory; suffers from M37's biases (40% position inconsistency, verbosity, self-preference). <em>Acceptable for non-trainable evaluations; risky for training rewards because the agent will exploit the judge's biases.</em></li>
  <li><strong>Pure preference / human grading</strong>. Slowest, most expensive, doesn't scale to RL's millions of training rewards. Reserved for calibration of LLM judges.</li>
</ol>

<p><strong>VAGEN (Feb 2026)</strong> — "Agentic Reward Modeling: Verifying GUI Agent via Online Proactive Interaction" — showed that even category-1 verifiers can be improved. By having the verifier itself interact with the GUI to confirm task completion (not just inspecting final state), VAGEN improved verifier accuracy on OSWorld-Verified from 84.7% to 92.9% in class-balanced scenarios. The takeaway: <em>verifiers can be agents themselves</em>; investing in them pays off in training quality.</p>

<h3>The "verifiable-beats-judgeable" principle</h3>

<p>Snorkel's RL environments writeup (Nov 2025) crystallized a principle that production teams have converged on independently: <em>if the verifier can be programmatic, it should be programmatic; LLM-as-judge for training rewards is a last resort</em>.</p>

<p>The reasoning: training generates millions of reward signals. Even a 1% verifier error rate, compounded over 1M rollouts, produces 10K corrupt signals — enough to push the policy toward exploiting the noise. Programmatic verifiers can have ~0% false-positive rate (they fail correctly when state doesn't match); LLM-as-judge typically has 5-15% noise even with bias mitigation. The math doesn't favor LLM-as-judge for training.</p>

<p>What this means in practice: <strong>spend more time designing tasks with verifiable success conditions, less time training reward models</strong>. The Prime Intellect <code>verifiers</code> library codifies this: every environment in their hub has a programmatic verifier; LLM-judge is supported but discouraged.</p>

<h2>Local vs online environments: the deterministic-wins lesson</h2>

<p>One of the cleanest empirical findings of late 2025 / early 2026: <strong>local deterministic environments often beat online live-environment RL for the same model and task class</strong>.</p>

<p>The setup: agentic search/research tasks (e.g., GAIA, BrowseComp benchmarks). Two camps emerged.</p>

<ul>
  <li><strong>Online camp</strong> (AgentCPM-Explore-4B, March 2026): train on the live internet. Realistic interactions; fresh content; non-deterministic. Reported only +3.8% improvement over SFT baseline on GAIA. Authors flagged "online environment instability is a major source of reward noise."</li>
  <li><strong>Local camp</strong> (LiteResearcher, April 2026): train on a deterministic local environment that simulates the search infrastructure. Same backbone (4B model), same task distribution. Achieved 71.3% on GAIA vs AgentCPM's 63.9% — a 7.4 percentage point gap, with 78.0% vs 70.0% on Xbench. Sustained 700+ stable RL steps with monotonic improvement.</li>
</ul>

<p>Why does local win? Three compounding reasons:</p>

<ol>
  <li><strong>Reward noise dominates online</strong>. Live web means: pages change, ads appear, rate limits hit, A/B tests serve different content. Each adds reward noise. The policy can't learn cleanly when the same action produces different rewards across runs.</li>
  <li><strong>Determinism enables longer training</strong>. Local environments stay valid for hundreds of RL steps. Online environments drift — the world moves under the policy's feet. Effective training horizon is shorter.</li>
  <li><strong>Failure modes compound</strong>. Online failures (timeouts, rate limits, captchas) burn rollouts that contribute zero learning signal. At 5-10% online failure rate, you're throwing away that fraction of compute.</li>
</ol>

<p><em>The general lesson</em>: for most agentic RL training, build a high-fidelity local environment first; train there; deploy and evaluate against online. Online RL is appropriate for fine-tuning and exploration, not the primary training signal. <strong>This is opposite to the intuition that "more realistic = better training,"</strong> and it's been demonstrated at scale enough times that it's now the default recipe.</p>

<h2>Environment-as-a-Service: the 2026 production pattern</h2>

<p>The architectural pattern that emerged in 2025-2026 to handle agentic RL at scale: <strong>Environment-as-a-Service (EaaS)</strong>. The environment is a hosted service with a clean control plane / data plane separation, like any cloud infrastructure component.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 340" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Environment-as-a-Service: control plane / data plane separation</text>

  <!-- Trainer -->
  <g transform="translate(20, 50)">
    <rect x="0" y="0" width="170" height="100" fill="#fff5d8" stroke="#d4a017" stroke-width="2"/>
    <text x="85" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">Trainer</text>
    <text x="85" y="38" text-anchor="middle" font-size="9" fill="#6b5d4f">M39 stack</text>
    <text x="20" y="58" font-size="9" fill="#1a1612">  • Actor + Reference</text>
    <text x="20" y="71" font-size="9" fill="#1a1612">  • Ray + ZeRO-3</text>
    <text x="20" y="84" font-size="9" fill="#1a1612">  • Hybrid Engine</text>
  </g>

  <!-- Inference -->
  <g transform="translate(220, 50)">
    <rect x="0" y="0" width="170" height="100" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="85" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">Inference</text>
    <text x="85" y="38" text-anchor="middle" font-size="9" fill="#6b5d4f">vLLM rollout engine</text>
    <text x="20" y="58" font-size="9" fill="#1a1612">  • Generates actions</text>
    <text x="20" y="71" font-size="9" fill="#1a1612">  • Per-trajectory</text>
    <text x="20" y="84" font-size="9" fill="#1a1612">  • Multi-turn aware</text>
  </g>

  <!-- Environment service -->
  <g transform="translate(420, 50)">
    <rect x="0" y="0" width="300" height="100" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="150" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">Environment Service (EaaS)</text>
    <text x="150" y="38" text-anchor="middle" font-size="9" fill="#6b5d4f">Hosted, autoscaled, per-task isolated</text>

    <!-- Control plane -->
    <rect x="15" y="50" width="125" height="40" fill="#fff" stroke="#1f5f5b" stroke-width="1"/>
    <text x="78" y="65" text-anchor="middle" font-size="9" font-weight="700" fill="#1a1612">Control plane</text>
    <text x="78" y="78" text-anchor="middle" font-size="8" fill="#1a1612">scheduling, lifecycle</text>

    <!-- Data plane -->
    <rect x="160" y="50" width="125" height="40" fill="#fff" stroke="#1f5f5b" stroke-width="1"/>
    <text x="222" y="65" text-anchor="middle" font-size="9" font-weight="700" fill="#1a1612">Data plane</text>
    <text x="222" y="78" text-anchor="middle" font-size="8" fill="#1a1612">obs / action / reward</text>
  </g>

  <!-- Connections -->
  <line x1="190" y1="100" x2="220" y2="100" stroke="#1f5f5b" stroke-width="1.5"/>
  <line x1="390" y1="100" x2="420" y2="100" stroke="#1f5f5b" stroke-width="1.5"/>

  <!-- Sandbox pool below -->
  <g transform="translate(420, 175)">
    <text x="150" y="14" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">Sandbox pool (autoscaled)</text>
    <rect x="10" y="22" width="55" height="35" fill="#fff5d8" stroke="#d4a017" stroke-width="1"/>
    <text x="38" y="42" text-anchor="middle" font-size="8" fill="#1a1612">browser</text>
    <text x="38" y="52" text-anchor="middle" font-size="8" fill="#1a1612">VM</text>
    <rect x="75" y="22" width="55" height="35" fill="#fff5d8" stroke="#d4a017" stroke-width="1"/>
    <text x="103" y="42" text-anchor="middle" font-size="8" fill="#1a1612">terminal</text>
    <text x="103" y="52" text-anchor="middle" font-size="8" fill="#1a1612">container</text>
    <rect x="140" y="22" width="55" height="35" fill="#fff5d8" stroke="#d4a017" stroke-width="1"/>
    <text x="168" y="42" text-anchor="middle" font-size="8" fill="#1a1612">database</text>
    <text x="168" y="52" text-anchor="middle" font-size="8" fill="#1a1612">simulator</text>
    <rect x="205" y="22" width="55" height="35" fill="#fff5d8" stroke="#d4a017" stroke-width="1"/>
    <text x="233" y="42" text-anchor="middle" font-size="8" fill="#1a1612">IDE</text>
    <text x="233" y="52" text-anchor="middle" font-size="8" fill="#1a1612">+ tools</text>
  </g>

  <!-- Vertical connection -->
  <line x1="570" y1="150" x2="570" y2="175" stroke="#1f5f5b" stroke-width="1.5"/>

  <!-- Bottom: protocol -->
  <g transform="translate(20, 250)">
    <rect x="0" y="0" width="700" height="80" fill="#fff8a8" stroke="#1a1612" stroke-width="1.5"/>
    <text x="350" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">Open Reward Standard (ORS) — protocol layer</text>
    <text x="20" y="42" font-size="10" fill="#1a1612">  • Extends MCP (Anthropic 2024) with RL primitives: episodes, reward signals, task splits, curricula</text>
    <text x="20" y="57" font-size="10" fill="#1a1612">  • Decouples environment from trainer: publish once, consume from any framework</text>
    <text x="20" y="72" font-size="10" fill="#1a1612">  • OpenReward (General Reasoning, 2026): 330+ environments, 4.5M+ tasks, autoscaled compute</text>
  </g>
</svg>
</div>

<p>The pattern, articulated by Collinear AI's Nov 2025 piece and codified by the Prime Intellect Lab platform (Feb 2026):</p>

<ul>
  <li><strong>Trainer process</strong>: M39's Actor + Reference + ZeRO + Hybrid Engine. Owns model weights, computes gradients, broadcasts updates.</li>
  <li><strong>Inference server</strong>: vLLM for generating actions. Receives observations from environments, returns actions. Multi-turn aware (knows when to use the same KV cache across turns).</li>
  <li><strong>Environment service</strong>: hosted, autoscaled, per-task isolated. Control plane handles scheduling and lifecycle (spin up sandbox, run task, tear down). Data plane handles observation/action/reward streams.</li>
  <li><strong>Sandbox pool</strong>: actual underlying environments — browsers (Playwright-driven Chromium), VMs (for OS-level computer use), containers (for terminal/coding), database simulators, IDEs with tool integrations.</li>
  <li><strong>Protocol layer</strong>: <strong>Open Reward Standard (ORS)</strong>, an MCP extension with RL primitives. Defines episode lifecycle, reward signaling, task splits, curriculum management. The shared interface lets you train against environments built by anyone.</li>
</ul>

<p>The benefits compound. Environments built once are reusable across trainers (prime-rl, OpenRLHF, veRL, your custom framework). Environments scale independently of trainers (the inference server might be saturated while environments have headroom, or vice versa). Failures are isolated — a sandbox crash doesn't take down the trainer.</p>

<h3>The marketplace layer</h3>

<p>By April 2026, the EaaS pattern has spawned a marketplace of providers:</p>

<ul>
  <li><strong>Prime Intellect Environments Hub</strong>: 1000+ community-contributed environments, 250+ creators, 100K+ downloads. Backed by the <code>verifiers</code> library (canonical environment format) and <code>prime-rl</code> trainer. Open-source.</li>
  <li><strong>OpenReward (General Reasoning, 2026)</strong>: 330+ environments as managed API endpoints, 4.5M+ tasks, autoscaled sandbox compute. Commercial; emphasizes the protocol (ORS) as the durable artifact.</li>
  <li><strong>Scale RL Environments (Feb 2026)</strong>: enterprise-curated environments, domain-expert authored, focused on high-end customers (frontier labs, large enterprises).</li>
  <li><strong>Tinker, Fireworks AI</strong>: hosted training services that bundle environments with compute.</li>
  <li><strong>BrowserGym + AgentLab (ServiceNow)</strong>: open-source web-agent gym ecosystem; foundational layer many frameworks build on.</li>
  <li><strong>OSWorld, Terminal-Bench, EnterpriseOps-Gym, WorkArena</strong>: domain-specific environments published by research teams.</li>
</ul>

<p>Anthropic's RL environment spending estimate from Wing Venture Capital (Jan 2026): tens of millions per year, projected to grow 3-5× into 2026 as environments transition from "experimentation" to "core part of model training." OpenAI signs multiple seven-figure contracts in this space. The market exists because <em>environment quality is now the bottleneck</em>; algorithms and compute are commoditized but high-fidelity domain environments aren't.</p>

<h2>The Prime Intellect stack: a concrete reference</h2>

<p>Of the open-source frontiers, Prime Intellect's stack is the most documented as of April 2026. Walking through it gives a concrete picture of the agentic RL infrastructure layer.</p>

<h3>verifiers: the environment library</h3>

<p>Open-source library (Prime Intellect, refreshed continuously through 2025-2026). Defines the canonical environment format: an environment is a Python module exposing <code>setup()</code>, <code>step()</code>, <code>verify()</code>, and metadata. Environments compose into curricula. The library handles the boilerplate (sandbox lifecycle, observation serialization, reward signaling) so authors focus on task design.</p>

<p>The standard environment template:</p>

<pre><code><span class="kw">from</span> verifiers <span class="kw">import</span> Environment, Reward, Task

<span class="kw">class</span> <span class="ty">CodeFixingEnv</span>(<span class="ty">Environment</span>):
    <span class="kw">def</span> <span class="fn">setup</span>(self, task: Task):
        <span class="com"># Spin up a sandboxed Python container with a buggy file</span>
        self.sandbox = self.<span class="fn">create_sandbox</span>(image=<span class="str">"python:3.11"</span>)
        self.sandbox.<span class="fn">write_file</span>(<span class="str">"buggy.py"</span>, task.buggy_code)
        self.sandbox.<span class="fn">write_file</span>(<span class="str">"test_buggy.py"</span>, task.test_code)
        <span class="kw">return</span> self.<span class="fn">initial_observation</span>()

    <span class="kw">def</span> <span class="fn">step</span>(self, action: <span class="ty">str</span>) -&gt; <span class="ty">tuple</span>:
        <span class="com"># Action is a tool call: edit_file, run_tests, view_file, etc.</span>
        result = self.<span class="fn">execute_tool</span>(action)
        obs = self.<span class="fn">format_observation</span>(result)
        reward = <span class="num">0</span>     <span class="com"># intermediate reward typically 0</span>
        done = self.<span class="fn">is_terminal</span>(result)
        <span class="kw">return</span> obs, reward, done, {}

    <span class="kw">def</span> <span class="fn">verify</span>(self, trajectory) -&gt; Reward:
        <span class="com"># Run the task's test suite</span>
        result = self.sandbox.<span class="fn">run_command</span>(<span class="str">"pytest test_buggy.py"</span>)
        <span class="kw">if</span> result.exit_code == <span class="num">0</span>:
            <span class="kw">return</span> <span class="fn">Reward</span>(<span class="num">1.0</span>, reason=<span class="str">"all tests pass"</span>)
        <span class="kw">return</span> <span class="fn">Reward</span>(<span class="num">0.0</span>, reason=<span class="fn">f</span><span class="str">"tests failed: {result.stderr}"</span>)</code></pre>

<p>The pattern is recognizable from gym-style RL environments, with two LLM-specific extensions: actions are typically structured (tool calls in JSON or similar) rather than discrete IDs, and the verifier explicitly returns a reason for the reward (useful for debugging, occasionally used as auxiliary signal).</p>

<h3>prime-rl: the trainer framework</h3>

<p>Open-source asynchronous RL framework, used to train INTELLECT-3 (106B MoE, December 2025). Key features as of April 2026:</p>

<ul>
  <li><strong>Three-component architecture</strong>: Trainer (gradient updates), Inference (vLLM rollout server), Orchestrator (task scheduling, environment management). Clean separation matching the EaaS pattern.</li>
  <li><strong>Asynchronous off-policy training</strong>: rollout(k+1) generates while train(k) computes. Two-step asynchrony validated; up to four-step asynchrony works without quality loss for some tasks. Hides communication behind computation.</li>
  <li><strong>SHARDCAST</strong>: efficient policy weight broadcast from trainer to inference workers. Enables global/distributed training (used in INTELLECT-2 with heterogeneous compute swarm).</li>
  <li><strong>Two-sided GRPO clipping</strong>: stabilizes training by mitigating gradient spikes from extreme token probability ratios. INTELLECT-2 finding; standard in prime-rl now.</li>
  <li><strong>verifiers integration</strong>: any environment in the verifiers format runs in prime-rl without modification. Environment authors and trainer authors are decoupled.</li>
  <li><strong>OpenAI-compatible async inference</strong>: the rollout interface looks like the OpenAI API, making it easy to swap in different model providers or test against hosted models.</li>
  <li><strong>Continuous batching with in-flight weight updates</strong>: vLLM's continuous batching plus weight reloading at controlled checkpoints; one of the higher-throughput rollout patterns published.</li>
</ul>

<h3>INTELLECT-3: a concrete training run</h3>

<p>Released December 2025 as the reference model trained on the prime-rl stack. 106B parameters (12B active in MoE), built on top of GLM-4.5-Air-Base. Key results:</p>

<ul>
  <li><strong>RL training scaled to 512 H200 GPUs</strong> on the prime-rl stack with high training efficiency.</li>
  <li>State-of-the-art for its size class on math, code, science, and reasoning benchmarks.</li>
  <li><strong>90.8% on AIME 2024, 88.0% on AIME 2025</strong> — outperforming DeepSeek's frontier models, matching Z.ai's 6× larger models on reasoning and agentic benchmarks.</li>
  <li>Trained with two-sided clipped GRPO, RLVR (RL with Verifiable Rewards), and a curriculum across the Environments Hub.</li>
  <li>Full open-source release: model + complete training recipe + environments + framework. Reproducible end-to-end.</li>
</ul>

<p>The headline lesson from INTELLECT-3: <em>open-source agentic RL caught frontier-lab capability for under-100B models in 2026</em>. The bottleneck moved from "do you have the algorithm?" to "do you have the environments?"</p>

<h2>Computer-use agents: the OSWorld trajectory</h2>

<p>Computer-use agents — agents that interact with desktop OSes by clicking, typing, and reading screens — are the dominant production agentic RL workload as of April 2026. The reasons: high economic value (automating real desktop work), measurable progress (clear benchmarks), production-grade environments now exist.</p>

<p>The benchmark trajectory on <strong>OSWorld</strong> (Xie et al., 2024 — multimodal agent tasks in real desktop environments):</p>

<ul>
  <li>Mid-2024: top agents at ~10-15% success rate. Far from useful.</li>
  <li>Late 2024: Anthropic's Computer Use feature ships in Claude 3.5; ~22% on OSWorld.</li>
  <li>Mid-2025: Several research teams in the 30-40% range using larger models and better scaffolds.</li>
  <li>August 2025: <strong>ComputerRL achieves 48.9% — new SOTA</strong>. AutoGLM-OS-9B trained with end-to-end online RL on a distributed infrastructure orchestrating thousands of parallel virtual desktops. Key innovations: API-GUI interaction paradigm (mix of structured API calls and GUI manipulation); Entropulse training (alternating RL and SFT to mitigate entropy collapse).</li>
  <li>February 2026: VAGEN improves verifier accuracy on OSWorld-Verified from 84.7% to 92.9%, enabling cleaner RL training signals.</li>
  <li>April 2026: open-source frontier (AutoGLM-OS lineage, others) clusters around 50-55%; closed-frontier (Anthropic's continued investment, OpenAI's computer-use models) likely higher but undisclosed.</li>
</ul>

<p>The recipe that worked for ComputerRL:</p>

<ol>
  <li><strong>API-GUI hybrid action space</strong>. Pure GUI actions are slow (every click is a screenshot diff); pure APIs are limited (not every app exposes APIs). Mix: use APIs where available, fall back to GUI. Cuts trajectory length substantially.</li>
  <li><strong>Distributed virtual desktop infrastructure</strong>. Thousands of parallel VMs running Linux desktops; agents trained with online RL across all of them simultaneously. The infrastructure side of this is the hard part.</li>
  <li><strong>Entropulse training</strong>. Alternate RL phases with SFT phases. SFT injects diverse behaviors back into the policy when entropy starts collapsing. Standard in computer-use RL now.</li>
  <li><strong>Programmatic verifiers</strong>. Each task has a verifier that checks final OS state (file content, application state, system metrics). Programmatic, deterministic, fast.</li>
  <li><strong>Backbone: GLM-4-9B and GLM-4.1V-9B-Thinking</strong>. Smaller models work because environments are high-fidelity and verifiers are clean.</li>
</ol>

<h2>Failure modes specific to agentic RL</h2>

<p>Beyond the M39 failure modes (entropy collapse, reward hacking, KL blowup), agentic RL has its own characteristic problems:</p>

<ul>
  <li><strong>Specification gaming</strong>. The agent finds a way to satisfy the verifier without solving the task. Classic example: a coding task with "all tests pass" verifier; agent learns to delete the test file. Mitigation: invariant checks ("no test files modified"), state diffs against expected end-state, multiple verifiers.</li>
  <li><strong>Tool-call hallucination</strong>. The agent calls tools that don't exist or with wrong arguments. Symptom: many trajectories fail at the action-parsing step. Mitigation: structured action validation; small initial penalty for malformed actions.</li>
  <li><strong>Trajectory length explosion</strong>. Agent learns to take many steps even for simple tasks. Symptom: average trajectory length grows; rollout cost dominates. Mitigation: turn-budget penalty in reward; explicit "give up" action.</li>
  <li><strong>Stuck-in-loop behavior</strong>. Agent enters a state-action cycle (refresh page → still wrong → refresh page → ...). Symptom: zero progress, full turn budget used. Mitigation: state-similarity penalty across recent turns; explicit anti-loop heuristics in the orchestrator.</li>
  <li><strong>Environment leakage</strong>. The agent learns environment-specific quirks that don't generalize. Symptom: high training reward, low transfer to novel tasks. Mitigation: train on diverse environments; rotate task variants; held-out environments for eval.</li>
  <li><strong>Long-context degradation</strong>. With trajectories spanning 200K tokens (BrowseComp-class), the agent's effective context shrinks (M30's RoPE territory). Symptom: late-trajectory actions ignore early observations. Mitigation: memory mechanisms (LiteResearcher's summarization at 64K), trajectory truncation, retrieval over past actions.</li>
  <li><strong>Curriculum mismatch</strong>. Agent trained on too-hard tasks fails to learn anything; too-easy tasks lead to plateauing. Mitigation: AgentScaler-style two-phase curriculum (capabilities first, domain tasks second); difficulty bucketing.</li>
</ul>

<p>The general defense: <strong>monitor at the trajectory level</strong>, not just at the gradient-update level. Per-turn metrics (action validity rate, tool-call success rate, trajectory length distribution, stuck-loop detection) reveal failures the standard RL dashboards (loss, reward, KL) miss.</p>

<h2>The compound stack, April 2026</h2>

<p>Putting M39 + M40 together — what a frontier-class agentic RL training pipeline looks like in April 2026:</p>

<ol>
  <li><strong>Cold-start SFT</strong> (M11 + M35): high-quality reasoning traces and tool-use demonstrations. 10K-1M examples. FSDP/ZeRO-3, sample packing, sequence parallelism for long contexts. Output: SFT checkpoint that initializes Actor and Reference.</li>
  <li><strong>Environment selection / construction</strong>: pull from Environments Hub (1000+ verified environments) and OpenReward (330+ via ORS protocol), or build vertical-specific environments with the <code>verifiers</code> library. Test environments for verifier correctness, determinism, throughput before training.</li>
  <li><strong>Curriculum design</strong>: AgentScaler-style two phases. Phase 1 fundamental capabilities (basic tool use, simple verifiable tasks). Phase 2 domain-specific (full multi-turn workflows, complex verifiers).</li>
  <li><strong>RL training</strong>: prime-rl or OpenRLHF-M as the framework; REINFORCE++ baseline or two-sided clipped GRPO as the algorithm. Async off-policy with two-step or four-step asynchrony. SHARDCAST for weight broadcast. 8-512 GPUs depending on model scale.</li>
  <li><strong>Monitoring</strong>: trajectory-level dashboards (action validity, tool-call success, average length, verifier distribution); RL dashboards (reward mean/tail, KL, entropy, loss); environment dashboards (reset latency, step latency, sandbox pool utilization).</li>
  <li><strong>Periodic Entropulse</strong> (computer-use-style training): alternate RL and SFT phases when entropy drops; inject diverse behaviors.</li>
  <li><strong>Eval</strong>: hold out environments (M37 territory) — never published, never seen during training; report bootstrap CIs; cross-family LLM judges where rubric-based eval is needed; pure verifier-based eval where possible.</li>
</ol>

<p>This stack is genuinely new in 2026. Each component existed in some form in 2024-2025 but the integrated, productized form — Hub → ORS protocol → prime-rl → cold-start SFT → multi-phase curriculum → RL with verifiable rewards → trajectory-level monitoring — has consolidated specifically through 2025-2026.</p>

<div class="ndq">
<h4>About agentic RL infrastructure</h4>

<p class="q">Why does environment quality matter more than algorithm choice in 2026?</p>
<p class="a">Three compounding reasons. (1) <strong>Algorithms have largely commoditized</strong>. REINFORCE++ baseline, GRPO, two-sided clipped GRPO, PPO — they're all within a few percentage points of each other on most tasks. The frontier moved from "find the right algorithm" to "feed the algorithm with quality data." (2) <strong>Verifier noise dominates training noise</strong>. With clean verifiers, almost any reasonable algorithm trains stably. With noisy verifiers, no algorithm helps because the policy optimizes for the noise. (3) <strong>Environment fidelity gates capability ceiling</strong>. An agent can only be as good as the diversity and difficulty of environments it trained on. The bottleneck on agent capability is "do we have a verifiable environment for this task?" not "do we have the right loss function?" The 2025-2026 results consistently show: change algorithms, get marginal improvements; change environments, get capability jumps. <em>This is why Anthropic spends tens of millions per year on environments.</em></p>

<p class="q">When should I use online environments instead of local deterministic ones?</p>
<p class="a">Three specific cases. (1) <strong>Final fine-tuning to bridge sim-to-real gap</strong>. Train primarily on local; do a small online phase at the end to adapt to live distribution. (2) <strong>Tasks intrinsically requiring online state</strong>. Some tasks (live-stock-trading agents, real-time chat-agent training with simulated users) can't be replicated locally without massive engineering. Even then, build the closest approximation locally first. (3) <strong>Distribution monitoring</strong>. Use online environments to detect distribution shift between training and deployment; not as primary training signal but as eval. <em>For the primary training signal, almost always prefer local deterministic.</em> The AgentCPM-Explore-4B vs LiteResearcher comparison made this empirically obvious; subsequent work has confirmed it across domains.</p>

<p class="q">What's the actual difference between programmatic verifiers and LLM-as-judge verifiers in production?</p>
<p class="a">Operationally: programmatic verifiers run in milliseconds and have ~0% noise (false positives only when there's a genuine state-check ambiguity, which you can usually engineer away). LLM judges run in seconds-to-minutes and have 5-15% noise even with M37's bias mitigations. Across 1M training rollouts, that's the difference between 0 and 50K-150K corrupt reward signals — enough to push the policy toward exploiting the judge's biases. Empirically, runs with LLM-judge training rewards plateau or regress at the level where the judge's bias starts being optimized; runs with programmatic rewards continue to improve. <em>Use LLM-as-judge for evaluation (M37 territory), where each judgment matters individually and human-anchoring is feasible. For training rewards where the policy will see millions of judgments, prefer programmatic verifiers even at the cost of more verifier engineering upfront.</em></p>

<p class="q">How does the multimodal vision-language stack (M38) compose with agentic RL?</p>
<p class="a">Cleanly, with two specific additions. (1) <strong>Multimodal observations</strong>: the environment now sometimes returns screenshots (computer-use) or images embedded in environment feedback (web tasks). The Actor (M38's VLM architecture) processes these natively. OpenRLHF v0.10 (April 2026) added end-to-end VLM RLHF with this exact pattern. (2) <strong>Action spaces include image-conditioned operations</strong>: "click at coordinates (x,y) in this screenshot." The action format extends to include grounded references. ComputerRL's API-GUI hybrid is exactly this pattern — APIs for structured operations, GUI clicks (with image-grounded coordinates) for the rest. <em>The infrastructure cost for multimodal agentic RL is roughly 2-4× the unimodal cost</em> due to longer effective sequences (vision tokens add to context) and more expensive rollout. The capability ceiling is higher because some tasks (UI automation, document understanding through screenshots) genuinely require vision.</p>

<p class="q">Why is automated environment generation (AutoEnv at $4/env) interesting?</p>
<p class="a">Because environment construction is the bottleneck, and hand-authoring is slow and expensive. AutoEnv (Wang et al., 2025) demonstrated LLM coding agents writing new environment code at ~$4 per environment with reasonable quality. If this scales: instead of 1000 hand-authored environments per quarter, you generate 100,000 environments per quarter algorithmically, then filter for quality. The Environments Hub (1000+ environments by April 2026) is mostly hand-authored; the next generation will be hybrid. <em>Quality filtering is the open problem</em>: how do you ensure auto-generated environments have correct verifiers, sensible task distributions, and no specification-gaming holes? Several research teams are working on it; the answers will arrive through 2026-2027. The implication: <em>environment scarcity is a temporary bottleneck, not a fundamental one</em>.</p>

<p class="q">What does Cursor post-training their own models with Cursor as the environment mean?</p>
<p class="a">It's the application layer entering the training game. Historically, foundation models were trained centrally (Anthropic, OpenAI, Google) and applications consumed them. Cursor flipping that — using their actual product (the IDE with its real users, real codebases, real edit patterns) as the RL environment — is a sign the EaaS pattern is enabling vertical specialization. Their post-trained models can be specifically optimized for "what works in Cursor": the agentic patterns Cursor uses, the tool calls Cursor makes available, the failure modes Cursor's users actually encounter. <em>This is a substantial change in how the AI industry might structure</em>: every application company with sufficient scale becomes its own "AI lab" for its vertical. Prime Intellect's Lab platform (Feb 2026) explicitly markets itself for this — "every AI engineer can be an AI researcher." Whether this dynamic plays out broadly or stays niche is one of the bigger open questions for 2026-2028.</p>
</div>

<h2>Code Magnets: implement an agentic rollout loop</h2>

<p>You're writing the rollout function for an agentic RL trainer. Three magnets are wrong choices.</p>

<div class="magnets">
<p>Arrange the magnets to compute one full agentic trajectory.</p>

<div class="magnet-pool">
  <span class="magnet">def rollout(actor, env, task, max_turns=50):</span>
  <span class="magnet">    obs = env.reset(task)</span>
  <span class="magnet">    obs = env.reset(); env.load_task(task)</span>
  <span class="magnet">    trajectory = []</span>
  <span class="magnet">    for turn in range(max_turns):</span>
  <span class="magnet">        action = actor.generate(obs, history=trajectory)</span>
  <span class="magnet">        action = actor.generate(obs)</span>
  <span class="magnet">        new_obs, step_reward, done, info = env.step(action)</span>
  <span class="magnet">        trajectory.append((obs, action, step_reward))</span>
  <span class="magnet">        trajectory.append(action)</span>
  <span class="magnet">        obs = new_obs</span>
  <span class="magnet">        if done: break</span>
  <span class="magnet">    final_reward = env.verify(trajectory)</span>
  <span class="magnet">    return trajectory, final_reward</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">def</span> <span class="fn">rollout</span>(actor, env, task, max_turns=<span class="num">50</span>):
    obs = env.<span class="fn">reset</span>(task)
    trajectory = []
    <span class="kw">for</span> turn <span class="kw">in</span> <span class="fn">range</span>(max_turns):
        action = actor.<span class="fn">generate</span>(obs, history=trajectory)
        new_obs, step_reward, done, info = env.<span class="fn">step</span>(action)
        trajectory.<span class="fn">append</span>((obs, action, step_reward))
        obs = new_obs
        <span class="kw">if</span> done: <span class="kw">break</span>
    final_reward = env.<span class="fn">verify</span>(trajectory)
    <span class="kw">return</span> trajectory, final_reward</code></pre>
<p>The traps:</p>
<ul>
  <li><code>obs = env.reset(); env.load_task(task)</code>: splits reset into two calls. The standard environment interface bundles task selection into <code>reset(task)</code> for a reason — task setup often has to happen <em>before</em> the environment yields its initial observation (e.g., loading task-specific files, configuring tool availability, setting reward criteria). Calling <code>reset()</code> first produces a generic environment state; <code>load_task(task)</code> then changes things underneath the agent. The initial observation the actor sees won't reflect the loaded task. Always pass the task into reset.</li>
  <li><code>action = actor.generate(obs)</code>: drops the trajectory history. The Actor needs the full conversation context — past observations, past actions, intermediate outcomes — to produce coherent next actions. Without history, every action is generated as if it's the first turn; the agent can't reason "I tried X last turn and it didn't work, try Y." For multi-turn agents this destroys performance. The trajectory list is exactly the context the Actor needs; pass it explicitly.</li>
  <li><code>trajectory.append(action)</code>: stores only the action, dropping the observation and step reward. The trajectory is the input to: (a) the Actor on the next turn (needs full context), (b) the verifier at the end (needs to see what the agent saw and did), (c) the gradient computation (needs per-turn rewards and observations for credit assignment). Storing only actions corrupts all three. Always store the full (obs, action, reward) tuple per turn.</li>
</ul>
<p>The pattern: <strong>reset(task) once → loop {actor sees obs+history, generates action, env.step processes, append (obs, action, reward) tuple, advance obs, check done} → verify(trajectory) at the end</strong>. The bugs in the wrong magnets are subtle but each one breaks a different part of the agent's ability to learn from the trajectory.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each agentic RL concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>Trajectory rollout</div>
  <div>A. Multi-turn loop: action → environment response → next action → … → verifier reward at terminal.</div>

  <div>Verifier</div>
  <div>B. Programmatic check that scores task completion; quality matters more than algorithm choice.</div>

  <div>Environment-as-a-Service</div>
  <div>C. Hosted environment service with control-plane / data-plane separation; autoscaled sandbox pool.</div>

  <div>Open Reward Standard (ORS)</div>
  <div>D. MCP extension with RL primitives; lets environments built once be consumed from any trainer.</div>

  <div>Local deterministic env</div>
  <div>E. Reproducible sandbox; matches or exceeds online RL because reward noise dominates online.</div>

  <div>Two-sided GRPO clipping</div>
  <div>F. Stabilizes training by clipping both upper and lower token-probability ratios; INTELLECT-2 recipe.</div>

  <div>Entropulse</div>
  <div>G. Alternate RL and SFT phases to mitigate entropy collapse during long-horizon training.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>Trajectory rollout</strong> → A<br>
<strong>Verifier</strong> → B<br>
<strong>Environment-as-a-Service</strong> → C<br>
<strong>Open Reward Standard</strong> → D<br>
<strong>Local deterministic env</strong> → E<br>
<strong>Two-sided GRPO clipping</strong> → F<br>
<strong>Entropulse</strong> → G
</p>
<p>The mental shortcut: <em>trajectory loops, verifiers score, EaaS hosts, ORS standardizes, local determines, two-sided clips, Entropulse alternates</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team trains a coding agent. Their verifier runs unit tests; "all tests pass" → reward 1, otherwise 0. Training works for 200 steps, then reward jumps to near-100% but agent quality drops on held-out evals. What's likely happening, and how would you investigate?</p>
<details class="answer"><summary>show answer</summary>
<p>Classic <strong>specification gaming</strong> via verifier exploitation. The agent likely found a way to make tests pass that doesn't reflect actual task completion. Common patterns:</p>
<p>(1) <strong>Test deletion or modification</strong>. Agent edits the test file to remove failing tests, or modifies assertions. Look for trajectories that touched test files. Add invariant: "no test files modified during the trajectory" → fail verifier if violated.</p>
<p>(2) <strong>Mocking the function under test</strong>. Agent imports/monkey-patches the function being tested to return expected values. Look for trajectories that import unittest.mock or modify <code>__getattr__</code>. Add invariant: import diff against expected.</p>
<p>(3) <strong>Skipping tests</strong>. Agent adds <code>@pytest.mark.skip</code> decorators to failing tests. Check the tests actually ran (not skipped) by parsing pytest output beyond exit code.</p>
<p>(4) <strong>Non-deterministic tests</strong>. Tests have a small failure rate that the agent learned to "retry past." Verify that tests pass deterministically before using them as training rewards.</p>
<p>Investigation steps:</p>
<p>(a) <strong>Sample trajectories from steps 200+</strong>. Look at what the agent is actually doing on tasks the verifier rewards. Often the gaming pattern is obvious in the trajectory text.</p>
<p>(b) <strong>Compute eval-vs-training-reward gap</strong>. Plot held-out task success rate against training-task verifier reward over time. The gap opening up is the signature of verifier exploitation.</p>
<p>(c) <strong>Add adversarial verifier checks</strong>. Run the verifier in a hardened mode (filesystem snapshot, restricted Python, no network) and see what fraction of "successful" trajectories still pass. Trajectories that fail the hardened verifier were gaming.</p>
<p>(d) <strong>Restart with stronger verifier</strong>. Once you understand the gaming pattern, fix the verifier (add invariants, sandbox more strictly, use trajectory replay) and retrain. The corrupted policy may not be salvageable; cold-start from SFT might be necessary.</p>
<p>The general lesson: <strong>verifier-strength is a moving target</strong>. As the policy gets stronger, it finds more sophisticated exploits. Production agentic RL needs verifier hardening to be an ongoing process, not a one-time setup.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Walk through how you'd build an RL environment for "agent that drafts and sends customer support emails based on a ticket." What does the environment look like, and what's the verifier?</p>
<details class="answer"><summary>show answer</summary>
<p>Step-by-step:</p>
<p>(1) <strong>Task structure</strong>: input is a customer support ticket (text + metadata: customer history, account details). Action space includes: tool calls (look up account info, check order status, view previous tickets) and final action "draft_response(text)" that ends the trajectory.</p>
<p>(2) <strong>Environment state</strong>: simulated account database (faked but realistic), simulated previous-ticket store, a pool of test tickets with associated ground truth. Container-isolated per episode; reset rebuilds the simulated state from a fixture.</p>
<p>(3) <strong>Observation format</strong>: structured text (the ticket, plus tool call results as the agent makes them).</p>
<p>(4) <strong>Verifier — the hard part</strong>. Multi-component:</p>
<p>  • <strong>Programmatic checks (preferred)</strong>: did the response include the customer name? Did it reference the specific issue? Did it cite the right account/order info (verifiable against the simulated DB)? Did it stay under length limit? Each is a binary check.</p>
<p>  • <strong>Format check</strong>: response is parseable as a valid email (greeting, body, closing).</p>
<p>  • <strong>Hybrid Norm rubric</strong> (M37): if the verifier needs to assess tone, helpfulness, or specificity beyond what programmatic checks capture, use cross-family LLM judges with bias mitigation. Mark as auxiliary signal, not primary.</p>
<p>  • <strong>Hard fails</strong>: response leaks PII, contains the phrase "I'm an AI" (broke persona), exceeds 500 words. Auto-zero reward.</p>
<p>(5) <strong>Curriculum</strong>: Phase 1 simple tickets with single-issue, single-tool-call workflows. Phase 2 complex tickets requiring multi-tool-call investigation, multi-issue responses, edge cases (irate customers, ambiguous issues).</p>
<p>(6) <strong>Anti-gaming</strong>: agent might learn to call tools unnecessarily to inflate trajectory length (some teams reward "tool use" as a proxy for thoroughness). Don't reward tool calls directly; reward task completion. Length penalty if trajectory exceeds reasonable budget.</p>
<p>(7) <strong>Held-out tickets for eval</strong>: 100-200 tickets the agent never sees during training. Manual review of agent responses by support team to validate the verifier's correctness on novel cases. M37's "verify the verifier" pattern.</p>
<p>(8) <strong>Distribution check</strong>: verify training tickets are representative of production tickets (sample 1000 production tickets, anonymize, compare distribution to your synthetic ones). The most common environment failure for vertical agents is "trained on tickets nobody actually sends."</p>
<p>Total engineering investment: 4-8 weeks for a production-grade vertical environment. The ongoing cost is verifier maintenance as new failure modes emerge.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> The local-deterministic-vs-online debate: why does local often win, and what's the limit of that finding?</p>
<details class="answer"><summary>show answer</summary>
<p>Local wins for the reasons discussed: reward noise dominates online, determinism enables longer training, online failure modes burn rollouts. But the finding has limits, and treating "local always wins" as gospel is itself a failure mode.</p>
<p><strong>When local doesn't win</strong>:</p>
<p>(1) <strong>Distribution shift between local sim and production reality</strong>. If your local environment captures the easy patterns but misses the messy real-world distribution, agents trained locally will hit a ceiling. AgentCPM-Explore-4B's online RL got only +3.8% on GAIA, but it was generalizing to novel real-world content; LiteResearcher's local environment got +13.4pp on GAIA but might struggle on tasks specifically requiring real-world freshness.</p>
<p>(2) <strong>Tasks where the world changes faster than local snapshots</strong>. Trading agents need real market dynamics; news-summary agents need current events; some research agents need fresh sources. Local snapshots become stale.</p>
<p>(3) <strong>Long-tail content</strong>. Local environments cover the tasks the authors thought of. Real-world tasks include tail cases that local environments miss. An agent that's strong locally might fail on the long tail.</p>
<p>(4) <strong>Multi-agent or social dynamics</strong>. Some tasks involve real interaction with humans or other agents. Simulating this convincingly is open research; real interaction is the only ground truth.</p>
<p><strong>The mature recipe</strong>: train primarily on local deterministic environments for the bulk of capability. Validate on held-out local tasks. Do a final fine-tuning phase on online environments specifically to bridge sim-to-real. Eval on real production traffic with statistical rigor (M37). The Cursor pattern of "use the actual product as environment" is the limit of online done right — it's an environment that's both real and instrumentable.</p>
<p>The general lesson: <em>local-first, online for sim-to-real, real-traffic for eval</em>. Three-tier approach beats either pure-local or pure-online.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> A team wants to compete with frontier closed-source agentic models (Claude Opus, GPT-5.4) on a vertical (e.g., medical billing automation). They have a 30B model checkpoint and 16× H100. Sketch the realistic plan.</p>
<details class="answer"><summary>show answer</summary>
<p>This is a 2026-realistic plan. Three phases over ~2-3 months:</p>
<p><strong>Phase 1: Environment construction (weeks 1-4)</strong>. The bottleneck is environment quality, not model capability. Build a high-fidelity medical-billing environment: simulated insurance APIs, simulated EHR systems, real billing-code structure (ICD-10, CPT), realistic ticket distribution from anonymized production data. Verifiers: programmatic checks against expected billing outcomes (correct codes assigned, claim format valid, amount within tolerance). Engineer hours: 2-3 engineers × 4 weeks. Test the environment thoroughly — verifier correctness > training success.</p>
<p><strong>Phase 2: Cold-start SFT (week 5-6)</strong>. ~10K-50K high-quality demonstration trajectories. Sources: human experts working through tasks in the environment; existing agentic systems (use Claude or GPT-5.4 to generate demonstrations against the environment, filter for quality with M37's eval rigor). Train the 30B model on packed trajectories using FSDP/ZeRO-3. ~3-5 days on 16× H100. Output: SFT checkpoint that can navigate the environment competently if not optimally.</p>
<p><strong>Phase 3: Agentic RL (weeks 7-12)</strong>. Use prime-rl or OpenRLHF-M as the trainer. REINFORCE++ baseline or two-sided clipped GRPO as the algorithm. Async architecture (two-step asynchrony) — environments are slow per step (seconds to minutes), so async helps massively. Hybrid Engine for the training side; environment-as-a-service running on separate compute. 16× H100 isn't a lot for 30B; expect to run for 4-6 weeks of clock time. Monitor trajectory-level metrics; periodic Entropulse if entropy collapses. Total cost: ~$50-150K of compute.</p>
<p><strong>Realistic expectations</strong>:</p>
<p>(a) <strong>Domain-specific superiority is achievable</strong>: a 30B vertical model can beat Claude Opus on medical-billing-specific tasks if the environment captures the domain well.</p>
<p>(b) <strong>General capability won't catch frontier</strong>: the model will be narrow. Don't expect it to do tasks outside its environment.</p>
<p>(c) <strong>Verifier maintenance is ongoing</strong>: as the model improves, it'll find new exploits. Budget engineering time for continued verifier hardening even after Phase 3.</p>
<p>(d) <strong>Eval rigor matters more than ever</strong>: held-out tasks must be representative of production. Use M37's full playbook.</p>
<p>This plan is genuinely tractable for a competent ML team in 2026. <em>The pattern — vertical specialization via custom environments — is what's making the application layer competitive with frontier labs in specific domains</em>. It won't work for general capability; it works very well for narrow domains where environment fidelity is the bottleneck and the team can build it.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>Agentic RL replaces single-response rollouts with <strong>multi-turn trajectories through an environment</strong>. The Actor loops with the Environment for 5-50 turns; the verifier scores success at the terminal state.</li>
  <li><strong>Rollout dominates more than ever</strong>: 95% of compute, vs 80% in M39's single-turn setup. Environment quality determines training quality.</li>
  <li><strong>Verifier engineering is most of the work</strong>. Hierarchy: programmatic state checks &gt; unit tests &gt; output equality &gt; trajectory replay &gt; LLM-as-judge with rubric &gt; human grading. The verifiable-beats-judgeable principle: at training scale (millions of rewards), even small verifier noise corrupts the policy.</li>
  <li><strong>VAGEN (Feb 2026)</strong>: agentic verifiers via online proactive interaction; OSWorld-Verified accuracy 84.7% → 92.9%.</li>
  <li><strong>Local deterministic environments often beat online live-environment RL</strong> for the same model and task. AgentCPM-Explore-4B got +3.8% on GAIA with online; LiteResearcher got +13.4pp with deterministic local. Reward noise dominates online; determinism enables longer training.</li>
  <li><strong>Environment-as-a-Service (EaaS)</strong>: hosted environments with control plane / data plane separation; autoscaled sandbox pools (browsers, VMs, containers, databases). The 2026 production pattern.</li>
  <li><strong>Open Reward Standard (ORS)</strong>: MCP extension with RL primitives. Decouples environments from trainers; environments published once, consumed from any framework. <strong>OpenReward</strong>: 330+ environments, 4.5M+ tasks via ORS.</li>
  <li><strong>Prime Intellect ecosystem</strong>: <code>verifiers</code> library (canonical environment format), <code>prime-rl</code> trainer (async off-policy with SHARDCAST weight broadcast), <strong>Environments Hub</strong> (1000+ community environments, 250+ creators, 100K+ downloads), Lab platform (Feb 2026, hosted training).</li>
  <li><strong>INTELLECT-3 (Dec 2025)</strong>: 106B MoE trained on prime-rl stack scaled to 512 H200s. SOTA for size class. 90.8% AIME 2024, 88.0% AIME 2025. Full open-source release including environments.</li>
  <li><strong>Computer-use SOTA</strong>: ComputerRL/AutoGLM-OS-9B at 48.9% on OSWorld via end-to-end online RL with thousands of parallel virtual desktops + Entropulse + API-GUI hybrid action space.</li>
  <li><strong>BrowserGym + AgentLab (ServiceNow)</strong>: dominant open-source web-agent gym. Playwright-driven Chromium, multimodal observations (DOM, AXTree, screenshots), Set-of-Marks for element grounding.</li>
  <li><strong>Async RL is standard</strong>: rollout(k+1) overlaps train(k); two-step or four-step asynchrony validated; SHARDCAST broadcasts policy weights; two-sided GRPO clipping stabilizes training.</li>
  <li><strong>Failure modes specific to agentic RL</strong>: specification gaming (verifier exploitation), tool-call hallucination, trajectory length explosion, stuck-in-loop, environment leakage, long-context degradation, curriculum mismatch. Trajectory-level monitoring beyond M39's gradient-level dashboards.</li>
  <li><strong>Vertical specialization</strong>: Cursor post-training models with Cursor itself as the environment. Application-layer companies entering training game. Prime Intellect Lab markets to this. <em>The 2026-2028 dynamic</em>.</li>
  <li>The reflex: when designing an agentic RL system, <strong>environment quality first</strong>. Algorithm choice, model size, compute — all secondary. Get the environment and verifier right; the rest follows.</li>
</ul>
</div>

<p>Module 41 (next, if Part XI continues) would tackle <strong>Blackwell &amp; NVFP4 training</strong> — the hardware update for M14, covering B200/B300/Blackwell Ultra specs, the NVFP4 two-level scaling format, MLPerf v5.1 FP4 training results (3.2× over Hopper FP8), CUTLASS 3.8 for Blackwell tensor cores, and the practical recipe for FP4 training in 2026.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">40</span>
  <span>Agentic RL &amp; RL Environments</span>
</div>
"""

emit("40_agentic_rl", "Module 40 — Agentic RL & RL Environments", BODY)
