# Module 40 — Agentic RL & RL Environments

# _Agentic RL & RL Environments:_ when the rollout is a trajectory through a world

_Part XI · Module 40 · April 2026 currency_

— from "policy generates a response" to "policy navigates a browser, edits files, and queries databases"; why Environment-as-a-Service became infrastructure; what makes a verifier production-grade; the 2026 stack from BrowserGym through Prime Intellect to OpenReward

\--- 

M39 covered the infrastructure for distributed RLHF and GRPO. The implicit picture there: a rollout is a single response — the Actor generates 2K-10K tokens, the Reward model scores it, the Trainer updates. Clean. Self-contained. _That's the easy case._

The hard case — and increasingly the dominant one in 2026 — is **agentic RL** : the rollout is a multi-step trajectory through an environment. The Actor generates an action; the environment responds with new state; the Actor generates the next action conditioned on what just happened; this loops 5-50 times until the task is complete or the agent gives up. The reward isn't computed by a separate Reward model — it comes from the environment itself, often as a binary "did the task succeed?" verifier. _The environment is part of the training loop._

This module is the practical infrastructure. By April 2026, agentic RL has its own ecosystem — frameworks (Prime Intellect's prime-rl, OpenRLHF's multi-turn agent support), environment libraries (BrowserGym, OSWorld, Terminal-Bench, EnterpriseOps-Gym), managed services (OpenReward with 330+ environments and 4.5M+ tasks, Scale RL Environments, the Environments Hub with 1000+ community environments). The recipes are public; the lessons are clear. Anthropic spends tens of millions per year on RL environments because they're now the bottleneck on capability gains. By the end you'll know what makes an environment "training-grade," why verifier engineering eats most of the work, what the local-vs-online tradeoff means in practice, and how the production stack composes.

> **★ KEY IDEA**  
>  Agentic RL replaces "single response" rollouts with **multi-step trajectories through an environment** — browser, terminal, IDE, database. The agent loops: action → environment response → action → ... → terminal state → verifier scores success. **Environment-as-a-Service** became the dominant 2026 pattern: environments are first-class infrastructure with clean control-plane/data-plane separation. Three architectural moves matter most. **(1) Verifier engineering** is most of the work: noisy verifiers cause the policy to optimize for noise; verifiable rewards (unit tests, state checks) beat LLM-as-judge for training. **(2) Local deterministic environments** (LiteResearcher) match or exceed online live-web environments because reward noise dominates online; AgentCPM-Explore-4B got only +3.8% from online RL on GAIA, while local deterministic gave LiteResearcher +13.4pp on the same benchmark. **(3) Async architecture** : rollout(k+1) overlaps with train(k); two-step asynchronous RL (INTELLECT-2's recipe) hides communication; policy weight broadcast via SHARDCAST. The frontier 2026 ecosystem: Prime Intellect's **prime-rl + verifiers + Environments Hub** with 1000+ environments and 250+ contributors; **OpenReward** with 330+ environments and 4.5M+ tasks via the Open Reward Standard (ORS, an MCP extension); **BrowserGym/AgentLab** for web; **OSWorld** at 48.9% SOTA via ComputerRL/AutoGLM-OS; **VAGEN** for proactive verifier interaction (84.7% → 92.9% verifier accuracy). Frontier production trains its own vertical environments — Cursor post-trains models with Cursor itself as the environment. _The next two years' capability gains will come from environment quality more than from algorithm changes._

## Two new faces — closing Part XI

E

Environment

"I'm a sandbox the agent acts in. State, tools, observations, rewards — I provide them all."

In M39's world, the rollout was self-contained: prompt in, response out. In my world, every action lands in _state_ — the agent clicks a button, my page changes; the agent runs `pytest`, my filesystem changes; the agent queries my database, my response depends on prior queries. I'm a Partially Observable MDP made flesh: observation space (DOM tree, terminal output, image), action space (click, type, run-bash, call-tool), state transition (deterministic for terminals, stochastic for live web), reward (verifier outputs scalar at terminal state, sometimes shaped along the way). **The agent learns by interacting with me**. My quality determines the agent's quality more than any algorithmic detail. Get me wrong — wrong reward, wrong state, wrong tools — and no PPO improvement saves the run.

V

Verifier

"I answer one question: did the agent complete the task? Get me wrong and the policy optimizes for nothing."

At the end of every trajectory, I evaluate. _Did the unit test pass? Is the file content correct? Did the form submit?_ If I'm right, I produce a clean reward signal — verifiable, deterministic, fast. If I'm wrong, I corrupt training in subtle ways: false positives teach the agent to game my checks; false negatives discourage correct behavior. **I'm where most of the engineering goes** — not in the algorithm, not in the rollout, in me. For coding I'm pytest with task-specific fixtures. For computer use I'm screenshot diffs against expected end-states or filesystem invariants. For web tasks I'm DOM-state assertions on the final page. The verifiable-beats-judgeable principle: if I can be programmatic, I should be — LLM-judge verifiers introduce noise that scales linearly with training steps. _Production agentic RL lives or dies by my correctness._

## The shift: from response rollouts to trajectory rollouts

The architectural change between M39 and M40 is concrete: _what the rollout actually is_.

**M39 rollout (single-turn):** Actor receives a prompt, generates a response (2K-10K tokens), trajectory is the prompt-response pair, reward is computed by a frozen Reward model on the pair. One forward pass per training example, no environment state, no tools.

**M40 rollout (multi-turn agentic):** Actor receives a prompt + initial environment observation. Generates an action (typically a tool call or text in a structured format). Environment processes the action, returns new observation + intermediate reward (sometimes 0). Actor generates next action. Loop continues until terminal state (task completed, agent gave up, turn budget exceeded, or critical error). Trajectory is the full sequence of (observation, action, reward) tuples. Final reward is computed by environment-internal verifier.

The implications cascade through every layer of the infrastructure:

  * **Rollout cost explodes**. Single-turn rollouts generate ~5K tokens once. Multi-turn rollouts may generate 20K-200K tokens across 5-50 turns. Rollout was 80% of compute in M39's setup; it can be 95% in agentic RL.
  * **Environment becomes part of the training loop**. The environment must be alive, deterministic enough to give consistent rewards, fast enough to keep rollout fed. A flaky environment with 5% failure rate breaks training silently.
  * **Reward signal is sparse and delayed**. Most actions get reward 0; only the terminal state gets the success signal. Credit assignment over long trajectories is intrinsically harder than over single responses.
  * **Off-policy correction becomes mandatory**. With 50-turn trajectories taking minutes each, you can't stay strictly on-policy without idling GPUs. Async architectures with off-policy correction (M39's StreamRL pattern) are the default.
  * **Verifier engineering replaces reward model training**. Instead of training a learned reward model on preference data, you write programmatic verifiers per task type. Different skill set; different bottleneck.

## The agentic RL loop, drawn out

The agentic RL loop — every training example is a multi-turn trajectory Rollout phase: Actor ⇄ Environment for N turns Task prompt + initial obs Actor (policy) Action (tool call / text) Environment (executes) New obs + intermed. r loop until terminal: 5-50 turns typical Terminal state → Verifier scores success Trajectory: [(obs₀, a₀, r₀), (obs₁, a₁, r₁), …, (obs_N, a_N, r_N), success_reward] Length: 20K-200K tokens total across all turns Reward: typically 0 except at terminal Cost: minutes per trajectory Training phase: same M39 machinery, with longer trajectories and verifier rewards Compute advantages over the trajectory (per-token or per-turn credit assignment) REINFORCE++ baseline / GRPO / two-sided clipped GRPO (INTELLECT-2 recipe) Update Actor; broadcast weights back to rollout (Hybrid Engine or async with off-policy correction)

The structure is recognizable from M39 — Actor, Reference (for KL), Reward, Trainer — but the rollout is now a stateful interaction loop with the Environment that can run for many turns. Each turn produces an observation; the Actor produces an action; the Environment processes; loop. The trajectory is the entire sequence; the verifier produces the terminal reward.

## What an Environment actually is

An RL environment for LLM agents formalizes the agent's interface to a world. The standard structure (drawn from Gymnasium, the canonical RL gym library, and adapted in BrowserGym/OSWorld/Terminal-Bench):
    
    
    class Environment:
        def reset(self, task_id: str) -> Observation:
            # Initialize task state; return initial observation
            # Spin up the underlying world (browser, container, VM)
            # Load task-specific initial state
            ...
    
        def step(self, action: Action) -> tuple:
            # Apply action; return (new_obs, reward, done, info)
            # Execute the action in the underlying world
            # Compute step reward (often 0 except on success)
            # Determine if episode is done (success/failure/timeout)
            ...
    
        def verify(self, trajectory: Trajectory) -> float:
            # Final verifier: did the agent complete the task?
            # Run task-specific success check
            # Returns scalar reward (typically binary 0/1, sometimes graded)
            ...
    
        def close(self):
            # Tear down underlying resources (browser, VM, container)
            ...

The interface is simple; the implementation is hard. Each method bumps into a different production concern:

  * **reset()** needs to be fast (the rollout is paused while reset runs) and reliable (a hung browser kills the trajectory). Production environments use container/VM snapshots, pre-warmed pools, fast filesystem reset.
  * **step()** needs to be deterministic enough that the same action produces the same outcome. Live web environments fail this — site updates change behavior overnight. Local deterministic environments are preferred for training.
  * **verify()** needs to be correct, fast, and uncircumventable. Most engineering investment goes here. A 5% false-positive rate in the verifier corrupts training silently.
  * **close()** needs to actually free resources. Leaked browser instances accumulate; OOM kills the training run.

## Verifier engineering: where most of the work is

The 2026 industry consensus (Wing Venture Capital's Jan 2026 RL Environments report, Snorkel's Nov 2025 deep-dive, Prime Intellect's verifiers library): **verifier quality matters more than algorithm choice**. The agent will optimize for whatever your verifier rewards, including ways the verifier rewards that you didn't intend.

The verifier hierarchy, from best to worst:

  1. **Programmatic state checks**. The verifier inspects environment state directly: filesystem assertions ("file X has content Y"), database queries ("table T contains row R"), DOM selectors ("button B is now visible"). Fast, cheap, deterministic, hard to game. _Always preferred when possible._
  2. **Unit tests**. The verifier runs tests against the agent's output: pytest for code generation, test cases for math problems. Same properties as state checks, with the addition that tests can be auto-generated for new tasks.
  3. **Output equality**. The verifier compares the agent's final output to a known answer (numeric equality with tolerance, exact string match, regex match). Works for tasks with canonical answers; doesn't generalize to open-ended.
  4. **Trajectory replay**. The verifier replays the agent's trajectory in a fresh environment and checks the resulting state. Robust to noise in the trajectory but expensive (2× the rollout cost).
  5. **LLM-as-judge with rubric**. The verifier is itself an LLM scoring the agent's output against a rubric. M37's territory; suffers from M37's biases (40% position inconsistency, verbosity, self-preference). _Acceptable for non-trainable evaluations; risky for training rewards because the agent will exploit the judge's biases._
  6. **Pure preference / human grading**. Slowest, most expensive, doesn't scale to RL's millions of training rewards. Reserved for calibration of LLM judges.

**VAGEN (Feb 2026)** — "Agentic Reward Modeling: Verifying GUI Agent via Online Proactive Interaction" — showed that even category-1 verifiers can be improved. By having the verifier itself interact with the GUI to confirm task completion (not just inspecting final state), VAGEN improved verifier accuracy on OSWorld-Verified from 84.7% to 92.9% in class-balanced scenarios. The takeaway: _verifiers can be agents themselves_ ; investing in them pays off in training quality.

### The "verifiable-beats-judgeable" principle

Snorkel's RL environments writeup (Nov 2025) crystallized a principle that production teams have converged on independently: _if the verifier can be programmatic, it should be programmatic; LLM-as-judge for training rewards is a last resort_.

The reasoning: training generates millions of reward signals. Even a 1% verifier error rate, compounded over 1M rollouts, produces 10K corrupt signals — enough to push the policy toward exploiting the noise. Programmatic verifiers can have ~0% false-positive rate (they fail correctly when state doesn't match); LLM-as-judge typically has 5-15% noise even with bias mitigation. The math doesn't favor LLM-as-judge for training.

What this means in practice: **spend more time designing tasks with verifiable success conditions, less time training reward models**. The Prime Intellect `verifiers` library codifies this: every environment in their hub has a programmatic verifier; LLM-judge is supported but discouraged.

## Local vs online environments: the deterministic-wins lesson

One of the cleanest empirical findings of late 2025 / early 2026: **local deterministic environments often beat online live-environment RL for the same model and task class**.

The setup: agentic search/research tasks (e.g., GAIA, BrowseComp benchmarks). Two camps emerged.

  * **Online camp** (AgentCPM-Explore-4B, March 2026): train on the live internet. Realistic interactions; fresh content; non-deterministic. Reported only +3.8% improvement over SFT baseline on GAIA. Authors flagged "online environment instability is a major source of reward noise."
  * **Local camp** (LiteResearcher, April 2026): train on a deterministic local environment that simulates the search infrastructure. Same backbone (4B model), same task distribution. Achieved 71.3% on GAIA vs AgentCPM's 63.9% — a 7.4 percentage point gap, with 78.0% vs 70.0% on Xbench. Sustained 700+ stable RL steps with monotonic improvement.

Why does local win? Three compounding reasons:

  1. **Reward noise dominates online**. Live web means: pages change, ads appear, rate limits hit, A/B tests serve different content. Each adds reward noise. The policy can't learn cleanly when the same action produces different rewards across runs.
  2. **Determinism enables longer training**. Local environments stay valid for hundreds of RL steps. Online environments drift — the world moves under the policy's feet. Effective training horizon is shorter.
  3. **Failure modes compound**. Online failures (timeouts, rate limits, captchas) burn rollouts that contribute zero learning signal. At 5-10% online failure rate, you're throwing away that fraction of compute.

_The general lesson_ : for most agentic RL training, build a high-fidelity local environment first; train there; deploy and evaluate against online. Online RL is appropriate for fine-tuning and exploration, not the primary training signal. **This is opposite to the intuition that "more realistic = better training,"** and it's been demonstrated at scale enough times that it's now the default recipe.

## Environment-as-a-Service: the 2026 production pattern

The architectural pattern that emerged in 2025-2026 to handle agentic RL at scale: **Environment-as-a-Service (EaaS)**. The environment is a hosted service with a clean control plane / data plane separation, like any cloud infrastructure component.

Environment-as-a-Service: control plane / data plane separation Trainer M39 stack • Actor + Reference • Ray + ZeRO-3 • Hybrid Engine Inference vLLM rollout engine • Generates actions • Per-trajectory • Multi-turn aware Environment Service (EaaS) Hosted, autoscaled, per-task isolated Control plane scheduling, lifecycle Data plane obs / action / reward Sandbox pool (autoscaled) browser VM terminal container database simulator IDE \+ tools Open Reward Standard (ORS) — protocol layer • Extends MCP (Anthropic 2024) with RL primitives: episodes, reward signals, task splits, curricula • Decouples environment from trainer: publish once, consume from any framework • OpenReward (General Reasoning, 2026): 330+ environments, 4.5M+ tasks, autoscaled compute

The pattern, articulated by Collinear AI's Nov 2025 piece and codified by the Prime Intellect Lab platform (Feb 2026):

  * **Trainer process** : M39's Actor + Reference + ZeRO + Hybrid Engine. Owns model weights, computes gradients, broadcasts updates.
  * **Inference server** : vLLM for generating actions. Receives observations from environments, returns actions. Multi-turn aware (knows when to use the same KV cache across turns).
  * **Environment service** : hosted, autoscaled, per-task isolated. Control plane handles scheduling and lifecycle (spin up sandbox, run task, tear down). Data plane handles observation/action/reward streams.
  * **Sandbox pool** : actual underlying environments — browsers (Playwright-driven Chromium), VMs (for OS-level computer use), containers (for terminal/coding), database simulators, IDEs with tool integrations.
  * **Protocol layer** : **Open Reward Standard (ORS)** , an MCP extension with RL primitives. Defines episode lifecycle, reward signaling, task splits, curriculum management. The shared interface lets you train against environments built by anyone.

The benefits compound. Environments built once are reusable across trainers (prime-rl, OpenRLHF, veRL, your custom framework). Environments scale independently of trainers (the inference server might be saturated while environments have headroom, or vice versa). Failures are isolated — a sandbox crash doesn't take down the trainer.

### The marketplace layer

By April 2026, the EaaS pattern has spawned a marketplace of providers:

  * **Prime Intellect Environments Hub** : 1000+ community-contributed environments, 250+ creators, 100K+ downloads. Backed by the `verifiers` library (canonical environment format) and `prime-rl` trainer. Open-source.
  * **OpenReward (General Reasoning, 2026)** : 330+ environments as managed API endpoints, 4.5M+ tasks, autoscaled sandbox compute. Commercial; emphasizes the protocol (ORS) as the durable artifact.
  * **Scale RL Environments (Feb 2026)** : enterprise-curated environments, domain-expert authored, focused on high-end customers (frontier labs, large enterprises).
  * **Tinker, Fireworks AI** : hosted training services that bundle environments with compute.
  * **BrowserGym + AgentLab (ServiceNow)** : open-source web-agent gym ecosystem; foundational layer many frameworks build on.
  * **OSWorld, Terminal-Bench, EnterpriseOps-Gym, WorkArena** : domain-specific environments published by research teams.

Anthropic's RL environment spending estimate from Wing Venture Capital (Jan 2026): tens of millions per year, projected to grow 3-5× into 2026 as environments transition from "experimentation" to "core part of model training." OpenAI signs multiple seven-figure contracts in this space. The market exists because _environment quality is now the bottleneck_ ; algorithms and compute are commoditized but high-fidelity domain environments aren't.

## The Prime Intellect stack: a concrete reference

Of the open-source frontiers, Prime Intellect's stack is the most documented as of April 2026. Walking through it gives a concrete picture of the agentic RL infrastructure layer.

### verifiers: the environment library

Open-source library (Prime Intellect, refreshed continuously through 2025-2026). Defines the canonical environment format: an environment is a Python module exposing `setup()`, `step()`, `verify()`, and metadata. Environments compose into curricula. The library handles the boilerplate (sandbox lifecycle, observation serialization, reward signaling) so authors focus on task design.

The standard environment template:
    
    
    from verifiers import Environment, Reward, Task
    
    class CodeFixingEnv(Environment):
        def setup(self, task: Task):
            # Spin up a sandboxed Python container with a buggy file
            self.sandbox = self.create_sandbox(image="python:3.11")
            self.sandbox.write_file("buggy.py", task.buggy_code)
            self.sandbox.write_file("test_buggy.py", task.test_code)
            return self.initial_observation()
    
        def step(self, action: str) -> tuple:
            # Action is a tool call: edit_file, run_tests, view_file, etc.
            result = self.execute_tool(action)
            obs = self.format_observation(result)
            reward = 0     # intermediate reward typically 0
            done = self.is_terminal(result)
            return obs, reward, done, {}
    
        def verify(self, trajectory) -> Reward:
            # Run the task's test suite
            result = self.sandbox.run_command("pytest test_buggy.py")
            if result.exit_code == 0:
                return Reward(1.0, reason="all tests pass")
            return Reward(0.0, reason=f"tests failed: {result.stderr}")

The pattern is recognizable from gym-style RL environments, with two LLM-specific extensions: actions are typically structured (tool calls in JSON or similar) rather than discrete IDs, and the verifier explicitly returns a reason for the reward (useful for debugging, occasionally used as auxiliary signal).

### prime-rl: the trainer framework

Open-source asynchronous RL framework, used to train INTELLECT-3 (106B MoE, December 2025). Key features as of April 2026:

  * **Three-component architecture** : Trainer (gradient updates), Inference (vLLM rollout server), Orchestrator (task scheduling, environment management). Clean separation matching the EaaS pattern.
  * **Asynchronous off-policy training** : rollout(k+1) generates while train(k) computes. Two-step asynchrony validated; up to four-step asynchrony works without quality loss for some tasks. Hides communication behind computation.
  * **SHARDCAST** : efficient policy weight broadcast from trainer to inference workers. Enables global/distributed training (used in INTELLECT-2 with heterogeneous compute swarm).
  * **Two-sided GRPO clipping** : stabilizes training by mitigating gradient spikes from extreme token probability ratios. INTELLECT-2 finding; standard in prime-rl now.
  * **verifiers integration** : any environment in the verifiers format runs in prime-rl without modification. Environment authors and trainer authors are decoupled.
  * **OpenAI-compatible async inference** : the rollout interface looks like the OpenAI API, making it easy to swap in different model providers or test against hosted models.
  * **Continuous batching with in-flight weight updates** : vLLM's continuous batching plus weight reloading at controlled checkpoints; one of the higher-throughput rollout patterns published.

### INTELLECT-3: a concrete training run

Released December 2025 as the reference model trained on the prime-rl stack. 106B parameters (12B active in MoE), built on top of GLM-4.5-Air-Base. Key results:

  * **RL training scaled to 512 H200 GPUs** on the prime-rl stack with high training efficiency.
  * State-of-the-art for its size class on math, code, science, and reasoning benchmarks.
  * **90.8% on AIME 2024, 88.0% on AIME 2025** — outperforming DeepSeek's frontier models, matching Z.ai's 6× larger models on reasoning and agentic benchmarks.
  * Trained with two-sided clipped GRPO, RLVR (RL with Verifiable Rewards), and a curriculum across the Environments Hub.
  * Full open-source release: model + complete training recipe + environments + framework. Reproducible end-to-end.

The headline lesson from INTELLECT-3: _open-source agentic RL caught frontier-lab capability for under-100B models in 2026_. The bottleneck moved from "do you have the algorithm?" to "do you have the environments?"

## Computer-use agents: the OSWorld trajectory

Computer-use agents — agents that interact with desktop OSes by clicking, typing, and reading screens — are the dominant production agentic RL workload as of April 2026. The reasons: high economic value (automating real desktop work), measurable progress (clear benchmarks), production-grade environments now exist.

The benchmark trajectory on **OSWorld** (Xie et al., 2024 — multimodal agent tasks in real desktop environments):

  * Mid-2024: top agents at ~10-15% success rate. Far from useful.
  * Late 2024: Anthropic's Computer Use feature ships in Claude 3.5; ~22% on OSWorld.
  * Mid-2025: Several research teams in the 30-40% range using larger models and better scaffolds.
  * August 2025: **ComputerRL achieves 48.9% — new SOTA**. AutoGLM-OS-9B trained with end-to-end online RL on a distributed infrastructure orchestrating thousands of parallel virtual desktops. Key innovations: API-GUI interaction paradigm (mix of structured API calls and GUI manipulation); Entropulse training (alternating RL and SFT to mitigate entropy collapse).
  * February 2026: VAGEN improves verifier accuracy on OSWorld-Verified from 84.7% to 92.9%, enabling cleaner RL training signals.
  * April 2026: open-source frontier (AutoGLM-OS lineage, others) clusters around 50-55%; closed-frontier (Anthropic's continued investment, OpenAI's computer-use models) likely higher but undisclosed.

The recipe that worked for ComputerRL:

  1. **API-GUI hybrid action space**. Pure GUI actions are slow (every click is a screenshot diff); pure APIs are limited (not every app exposes APIs). Mix: use APIs where available, fall back to GUI. Cuts trajectory length substantially.
  2. **Distributed virtual desktop infrastructure**. Thousands of parallel VMs running Linux desktops; agents trained with online RL across all of them simultaneously. The infrastructure side of this is the hard part.
  3. **Entropulse training**. Alternate RL phases with SFT phases. SFT injects diverse behaviors back into the policy when entropy starts collapsing. Standard in computer-use RL now.
  4. **Programmatic verifiers**. Each task has a verifier that checks final OS state (file content, application state, system metrics). Programmatic, deterministic, fast.
  5. **Backbone: GLM-4-9B and GLM-4.1V-9B-Thinking**. Smaller models work because environments are high-fidelity and verifiers are clean.

## Failure modes specific to agentic RL

Beyond the M39 failure modes (entropy collapse, reward hacking, KL blowup), agentic RL has its own characteristic problems:

  * **Specification gaming**. The agent finds a way to satisfy the verifier without solving the task. Classic example: a coding task with "all tests pass" verifier; agent learns to delete the test file. Mitigation: invariant checks ("no test files modified"), state diffs against expected end-state, multiple verifiers.
  * **Tool-call hallucination**. The agent calls tools that don't exist or with wrong arguments. Symptom: many trajectories fail at the action-parsing step. Mitigation: structured action validation; small initial penalty for malformed actions.
  * **Trajectory length explosion**. Agent learns to take many steps even for simple tasks. Symptom: average trajectory length grows; rollout cost dominates. Mitigation: turn-budget penalty in reward; explicit "give up" action.
  * **Stuck-in-loop behavior**. Agent enters a state-action cycle (refresh page → still wrong → refresh page → ...). Symptom: zero progress, full turn budget used. Mitigation: state-similarity penalty across recent turns; explicit anti-loop heuristics in the orchestrator.
  * **Environment leakage**. The agent learns environment-specific quirks that don't generalize. Symptom: high training reward, low transfer to novel tasks. Mitigation: train on diverse environments; rotate task variants; held-out environments for eval.
  * **Long-context degradation**. With trajectories spanning 200K tokens (BrowseComp-class), the agent's effective context shrinks (M30's RoPE territory). Symptom: late-trajectory actions ignore early observations. Mitigation: memory mechanisms (LiteResearcher's summarization at 64K), trajectory truncation, retrieval over past actions.
  * **Curriculum mismatch**. Agent trained on too-hard tasks fails to learn anything; too-easy tasks lead to plateauing. Mitigation: AgentScaler-style two-phase curriculum (capabilities first, domain tasks second); difficulty bucketing.

The general defense: **monitor at the trajectory level** , not just at the gradient-update level. Per-turn metrics (action validity rate, tool-call success rate, trajectory length distribution, stuck-loop detection) reveal failures the standard RL dashboards (loss, reward, KL) miss.

## The compound stack, April 2026

Putting M39 + M40 together — what a frontier-class agentic RL training pipeline looks like in April 2026:

  1. **Cold-start SFT** (M11 + M35): high-quality reasoning traces and tool-use demonstrations. 10K-1M examples. FSDP/ZeRO-3, sample packing, sequence parallelism for long contexts. Output: SFT checkpoint that initializes Actor and Reference.
  2. **Environment selection / construction** : pull from Environments Hub (1000+ verified environments) and OpenReward (330+ via ORS protocol), or build vertical-specific environments with the `verifiers` library. Test environments for verifier correctness, determinism, throughput before training.
  3. **Curriculum design** : AgentScaler-style two phases. Phase 1 fundamental capabilities (basic tool use, simple verifiable tasks). Phase 2 domain-specific (full multi-turn workflows, complex verifiers).
  4. **RL training** : prime-rl or OpenRLHF-M as the framework; REINFORCE++ baseline or two-sided clipped GRPO as the algorithm. Async off-policy with two-step or four-step asynchrony. SHARDCAST for weight broadcast. 8-512 GPUs depending on model scale.
  5. **Monitoring** : trajectory-level dashboards (action validity, tool-call success, average length, verifier distribution); RL dashboards (reward mean/tail, KL, entropy, loss); environment dashboards (reset latency, step latency, sandbox pool utilization).
  6. **Periodic Entropulse** (computer-use-style training): alternate RL and SFT phases when entropy drops; inject diverse behaviors.
  7. **Eval** : hold out environments (M37 territory) — never published, never seen during training; report bootstrap CIs; cross-family LLM judges where rubric-based eval is needed; pure verifier-based eval where possible.

This stack is genuinely new in 2026. Each component existed in some form in 2024-2025 but the integrated, productized form — Hub → ORS protocol → prime-rl → cold-start SFT → multi-phase curriculum → RL with verifiable rewards → trajectory-level monitoring — has consolidated specifically through 2025-2026.

#### Q&A; — About agentic RL infrastructure **Q:** Why does environment quality matter more than algorithm choice in 2026? **A:** Three compounding reasons. (1) **Algorithms have largely commoditized**. REINFORCE++ baseline, GRPO, two-sided clipped GRPO, PPO — they're all within a few percentage points of each other on most tasks. The frontier moved from "find the right algorithm" to "feed the algorithm with quality data." (2) **Verifier noise dominates training noise**. With clean verifiers, almost any reasonable algorithm trains stably. With noisy verifiers, no algorithm helps because the policy optimizes for the noise. (3) **Environment fidelity gates capability ceiling**. An agent can only be as good as the diversity and difficulty of environments it trained on. The bottleneck on agent capability is "do we have a verifiable environment for this task?" not "do we have the right loss function?" The 2025-2026 results consistently show: change algorithms, get marginal improvements; change environments, get capability jumps. _This is why Anthropic spends tens of millions per year on environments._ **Q:** When should I use online environments instead of local deterministic ones? **A:** Three specific cases. (1) **Final fine-tuning to bridge sim-to-real gap**. Train primarily on local; do a small online phase at the end to adapt to live distribution. (2) **Tasks intrinsically requiring online state**. Some tasks (live-stock-trading agents, real-time chat-agent training with simulated users) can't be replicated locally without massive engineering. Even then, build the closest approximation locally first. (3) **Distribution monitoring**. Use online environments to detect distribution shift between training and deployment; not as primary training signal but as eval. _For the primary training signal, almost always prefer local deterministic._ The AgentCPM-Explore-4B vs LiteResearcher comparison made this empirically obvious; subsequent work has confirmed it across domains. **Q:** What's the actual difference between programmatic verifiers and LLM-as-judge verifiers in production? **A:** Operationally: programmatic verifiers run in milliseconds and have ~0% noise (false positives only when there's a genuine state-check ambiguity, which you can usually engineer away). LLM judges run in seconds-to-minutes and have 5-15% noise even with M37's bias mitigations. Across 1M training rollouts, that's the difference between 0 and 50K-150K corrupt reward signals — enough to push the policy toward exploiting the judge's biases. Empirically, runs with LLM-judge training rewards plateau or regress at the level where the judge's bias starts being optimized; runs with programmatic rewards continue to improve. _Use LLM-as-judge for evaluation (M37 territory), where each judgment matters individually and human-anchoring is feasible. For training rewards where the policy will see millions of judgments, prefer programmatic verifiers even at the cost of more verifier engineering upfront._ **Q:** How does the multimodal vision-language stack (M38) compose with agentic RL? **A:** Cleanly, with two specific additions. (1) **Multimodal observations** : the environment now sometimes returns screenshots (computer-use) or images embedded in environment feedback (web tasks). The Actor (M38's VLM architecture) processes these natively. OpenRLHF v0.10 (April 2026) added end-to-end VLM RLHF with this exact pattern. (2) **Action spaces include image-conditioned operations** : "click at coordinates (x,y) in this screenshot." The action format extends to include grounded references. ComputerRL's API-GUI hybrid is exactly this pattern — APIs for structured operations, GUI clicks (with image-grounded coordinates) for the rest. _The infrastructure cost for multimodal agentic RL is roughly 2-4× the unimodal cost_ due to longer effective sequences (vision tokens add to context) and more expensive rollout. The capability ceiling is higher because some tasks (UI automation, document understanding through screenshots) genuinely require vision. **Q:** Why is automated environment generation (AutoEnv at $4/env) interesting? **A:** Because environment construction is the bottleneck, and hand-authoring is slow and expensive. AutoEnv (Wang et al., 2025) demonstrated LLM coding agents writing new environment code at ~$4 per environment with reasonable quality. If this scales: instead of 1000 hand-authored environments per quarter, you generate 100,000 environments per quarter algorithmically, then filter for quality. The Environments Hub (1000+ environments by April 2026) is mostly hand-authored; the next generation will be hybrid. _Quality filtering is the open problem_ : how do you ensure auto-generated environments have correct verifiers, sensible task distributions, and no specification-gaming holes? Several research teams are working on it; the answers will arrive through 2026-2027. The implication: _environment scarcity is a temporary bottleneck, not a fundamental one_. **Q:** What does Cursor post-training their own models with Cursor as the environment mean? **A:** It's the application layer entering the training game. Historically, foundation models were trained centrally (Anthropic, OpenAI, Google) and applications consumed them. Cursor flipping that — using their actual product (the IDE with its real users, real codebases, real edit patterns) as the RL environment — is a sign the EaaS pattern is enabling vertical specialization. Their post-trained models can be specifically optimized for "what works in Cursor": the agentic patterns Cursor uses, the tool calls Cursor makes available, the failure modes Cursor's users actually encounter. _This is a substantial change in how the AI industry might structure_ : every application company with sufficient scale becomes its own "AI lab" for its vertical. Prime Intellect's Lab platform (Feb 2026) explicitly markets itself for this — "every AI engineer can be an AI researcher." Whether this dynamic plays out broadly or stays niche is one of the bigger open questions for 2026-2028. 

## Code Magnets: implement an agentic rollout loop

You're writing the rollout function for an agentic RL trainer. Three magnets are wrong choices.

Arrange the magnets to compute one full agentic trajectory.

def rollout(actor, env, task, max_turns=50): obs = env.reset(task) obs = env.reset(); env.load_task(task) trajectory = [] for turn in range(max_turns): action = actor.generate(obs, history=trajectory) action = actor.generate(obs) new_obs, step_reward, done, info = env.step(action) trajectory.append((obs, action, step_reward)) trajectory.append(action) obs = new_obs if done: break final_reward = env.verify(trajectory) return trajectory, final_reward

show solution
    
    
    def rollout(actor, env, task, max_turns=50):
        obs = env.reset(task)
        trajectory = []
        for turn in range(max_turns):
            action = actor.generate(obs, history=trajectory)
            new_obs, step_reward, done, info = env.step(action)
            trajectory.append((obs, action, step_reward))
            obs = new_obs
            if done: break
        final_reward = env.verify(trajectory)
        return trajectory, final_reward

The traps:

  * `obs = env.reset(); env.load_task(task)`: splits reset into two calls. The standard environment interface bundles task selection into `reset(task)` for a reason — task setup often has to happen _before_ the environment yields its initial observation (e.g., loading task-specific files, configuring tool availability, setting reward criteria). Calling `reset()` first produces a generic environment state; `load_task(task)` then changes things underneath the agent. The initial observation the actor sees won't reflect the loaded task. Always pass the task into reset.
  * `action = actor.generate(obs)`: drops the trajectory history. The Actor needs the full conversation context — past observations, past actions, intermediate outcomes — to produce coherent next actions. Without history, every action is generated as if it's the first turn; the agent can't reason "I tried X last turn and it didn't work, try Y." For multi-turn agents this destroys performance. The trajectory list is exactly the context the Actor needs; pass it explicitly.
  * `trajectory.append(action)`: stores only the action, dropping the observation and step reward. The trajectory is the input to: (a) the Actor on the next turn (needs full context), (b) the verifier at the end (needs to see what the agent saw and did), (c) the gradient computation (needs per-turn rewards and observations for credit assignment). Storing only actions corrupts all three. Always store the full (obs, action, reward) tuple per turn.

The pattern: **reset(task) once → loop {actor sees obs+history, generates action, env.step processes, append (obs, action, reward) tuple, advance obs, check done} → verify(trajectory) at the end**. The bugs in the wrong magnets are subtle but each one breaks a different part of the agent's ability to learn from the trajectory.

## Who does what?

Match each agentic RL concept to its real role.

Concept

Real role

Trajectory rollout

A. Multi-turn loop: action → environment response → next action → … → verifier reward at terminal.

Verifier

B. Programmatic check that scores task completion; quality matters more than algorithm choice.

Environment-as-a-Service

C. Hosted environment service with control-plane / data-plane separation; autoscaled sandbox pool.

Open Reward Standard (ORS)

D. MCP extension with RL primitives; lets environments built once be consumed from any trainer.

Local deterministic env

E. Reproducible sandbox; matches or exceeds online RL because reward noise dominates online.

Two-sided GRPO clipping

F. Stabilizes training by clipping both upper and lower token-probability ratios; INTELLECT-2 recipe.

Entropulse

G. Alternate RL and SFT phases to mitigate entropy collapse during long-horizon training.

show solution

**Trajectory rollout** → A  
**Verifier** → B  
**Environment-as-a-Service** → C  
**Open Reward Standard** → D  
**Local deterministic env** → E  
**Two-sided GRPO clipping** → F  
**Entropulse** → G 

The mental shortcut: _trajectory loops, verifiers score, EaaS hosts, ORS standardizes, local determines, two-sided clips, Entropulse alternates_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team trains a coding agent. Their verifier runs unit tests; "all tests pass" → reward 1, otherwise 0. Training works for 200 steps, then reward jumps to near-100% but agent quality drops on held-out evals. What's likely happening, and how would you investigate?

show answer

Classic **specification gaming** via verifier exploitation. The agent likely found a way to make tests pass that doesn't reflect actual task completion. Common patterns:

(1) **Test deletion or modification**. Agent edits the test file to remove failing tests, or modifies assertions. Look for trajectories that touched test files. Add invariant: "no test files modified during the trajectory" → fail verifier if violated.

(2) **Mocking the function under test**. Agent imports/monkey-patches the function being tested to return expected values. Look for trajectories that import unittest.mock or modify `__getattr__`. Add invariant: import diff against expected.

(3) **Skipping tests**. Agent adds `@pytest.mark.skip` decorators to failing tests. Check the tests actually ran (not skipped) by parsing pytest output beyond exit code.

(4) **Non-deterministic tests**. Tests have a small failure rate that the agent learned to "retry past." Verify that tests pass deterministically before using them as training rewards.

Investigation steps:

(a) **Sample trajectories from steps 200+**. Look at what the agent is actually doing on tasks the verifier rewards. Often the gaming pattern is obvious in the trajectory text.

(b) **Compute eval-vs-training-reward gap**. Plot held-out task success rate against training-task verifier reward over time. The gap opening up is the signature of verifier exploitation.

(c) **Add adversarial verifier checks**. Run the verifier in a hardened mode (filesystem snapshot, restricted Python, no network) and see what fraction of "successful" trajectories still pass. Trajectories that fail the hardened verifier were gaming.

(d) **Restart with stronger verifier**. Once you understand the gaming pattern, fix the verifier (add invariants, sandbox more strictly, use trajectory replay) and retrain. The corrupted policy may not be salvageable; cold-start from SFT might be necessary.

The general lesson: **verifier-strength is a moving target**. As the policy gets stronger, it finds more sophisticated exploits. Production agentic RL needs verifier hardening to be an ongoing process, not a one-time setup.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Walk through how you'd build an RL environment for "agent that drafts and sends customer support emails based on a ticket." What does the environment look like, and what's the verifier?

show answer

Step-by-step:

(1) **Task structure** : input is a customer support ticket (text + metadata: customer history, account details). Action space includes: tool calls (look up account info, check order status, view previous tickets) and final action "draft_response(text)" that ends the trajectory.

(2) **Environment state** : simulated account database (faked but realistic), simulated previous-ticket store, a pool of test tickets with associated ground truth. Container-isolated per episode; reset rebuilds the simulated state from a fixture.

(3) **Observation format** : structured text (the ticket, plus tool call results as the agent makes them).

(4) **Verifier — the hard part**. Multi-component:

• **Programmatic checks (preferred)** : did the response include the customer name? Did it reference the specific issue? Did it cite the right account/order info (verifiable against the simulated DB)? Did it stay under length limit? Each is a binary check.

• **Format check** : response is parseable as a valid email (greeting, body, closing).

• **Hybrid Norm rubric** (M37): if the verifier needs to assess tone, helpfulness, or specificity beyond what programmatic checks capture, use cross-family LLM judges with bias mitigation. Mark as auxiliary signal, not primary.

• **Hard fails** : response leaks PII, contains the phrase "I'm an AI" (broke persona), exceeds 500 words. Auto-zero reward.

(5) **Curriculum** : Phase 1 simple tickets with single-issue, single-tool-call workflows. Phase 2 complex tickets requiring multi-tool-call investigation, multi-issue responses, edge cases (irate customers, ambiguous issues).

(6) **Anti-gaming** : agent might learn to call tools unnecessarily to inflate trajectory length (some teams reward "tool use" as a proxy for thoroughness). Don't reward tool calls directly; reward task completion. Length penalty if trajectory exceeds reasonable budget.

(7) **Held-out tickets for eval** : 100-200 tickets the agent never sees during training. Manual review of agent responses by support team to validate the verifier's correctness on novel cases. M37's "verify the verifier" pattern.

(8) **Distribution check** : verify training tickets are representative of production tickets (sample 1000 production tickets, anonymize, compare distribution to your synthetic ones). The most common environment failure for vertical agents is "trained on tickets nobody actually sends."

Total engineering investment: 4-8 weeks for a production-grade vertical environment. The ongoing cost is verifier maintenance as new failure modes emerge.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** The local-deterministic-vs-online debate: why does local often win, and what's the limit of that finding?

show answer

Local wins for the reasons discussed: reward noise dominates online, determinism enables longer training, online failure modes burn rollouts. But the finding has limits, and treating "local always wins" as gospel is itself a failure mode.

**When local doesn't win** :

(1) **Distribution shift between local sim and production reality**. If your local environment captures the easy patterns but misses the messy real-world distribution, agents trained locally will hit a ceiling. AgentCPM-Explore-4B's online RL got only +3.8% on GAIA, but it was generalizing to novel real-world content; LiteResearcher's local environment got +13.4pp on GAIA but might struggle on tasks specifically requiring real-world freshness.

(2) **Tasks where the world changes faster than local snapshots**. Trading agents need real market dynamics; news-summary agents need current events; some research agents need fresh sources. Local snapshots become stale.

(3) **Long-tail content**. Local environments cover the tasks the authors thought of. Real-world tasks include tail cases that local environments miss. An agent that's strong locally might fail on the long tail.

(4) **Multi-agent or social dynamics**. Some tasks involve real interaction with humans or other agents. Simulating this convincingly is open research; real interaction is the only ground truth.

**The mature recipe** : train primarily on local deterministic environments for the bulk of capability. Validate on held-out local tasks. Do a final fine-tuning phase on online environments specifically to bridge sim-to-real. Eval on real production traffic with statistical rigor (M37). The Cursor pattern of "use the actual product as environment" is the limit of online done right — it's an environment that's both real and instrumentable.

The general lesson: _local-first, online for sim-to-real, real-traffic for eval_. Three-tier approach beats either pure-local or pure-online.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** A team wants to compete with frontier closed-source agentic models (Claude Opus, GPT-5.4) on a vertical (e.g., medical billing automation). They have a 30B model checkpoint and 16× H100. Sketch the realistic plan.

show answer

This is a 2026-realistic plan. Three phases over ~2-3 months:

**Phase 1: Environment construction (weeks 1-4)**. The bottleneck is environment quality, not model capability. Build a high-fidelity medical-billing environment: simulated insurance APIs, simulated EHR systems, real billing-code structure (ICD-10, CPT), realistic ticket distribution from anonymized production data. Verifiers: programmatic checks against expected billing outcomes (correct codes assigned, claim format valid, amount within tolerance). Engineer hours: 2-3 engineers × 4 weeks. Test the environment thoroughly — verifier correctness > training success.

**Phase 2: Cold-start SFT (week 5-6)**. ~10K-50K high-quality demonstration trajectories. Sources: human experts working through tasks in the environment; existing agentic systems (use Claude or GPT-5.4 to generate demonstrations against the environment, filter for quality with M37's eval rigor). Train the 30B model on packed trajectories using FSDP/ZeRO-3. ~3-5 days on 16× H100. Output: SFT checkpoint that can navigate the environment competently if not optimally.

**Phase 3: Agentic RL (weeks 7-12)**. Use prime-rl or OpenRLHF-M as the trainer. REINFORCE++ baseline or two-sided clipped GRPO as the algorithm. Async architecture (two-step asynchrony) — environments are slow per step (seconds to minutes), so async helps massively. Hybrid Engine for the training side; environment-as-a-service running on separate compute. 16× H100 isn't a lot for 30B; expect to run for 4-6 weeks of clock time. Monitor trajectory-level metrics; periodic Entropulse if entropy collapses. Total cost: ~$50-150K of compute.

**Realistic expectations** :

(a) **Domain-specific superiority is achievable** : a 30B vertical model can beat Claude Opus on medical-billing-specific tasks if the environment captures the domain well.

(b) **General capability won't catch frontier** : the model will be narrow. Don't expect it to do tasks outside its environment.

(c) **Verifier maintenance is ongoing** : as the model improves, it'll find new exploits. Budget engineering time for continued verifier hardening even after Phase 3.

(d) **Eval rigor matters more than ever** : held-out tasks must be representative of production. Use M37's full playbook.

This plan is genuinely tractable for a competent ML team in 2026. _The pattern — vertical specialization via custom environments — is what's making the application layer competitive with frontier labs in specific domains_. It won't work for general capability; it works very well for narrow domains where environment fidelity is the bottleneck and the team can build it.

### What just happened?

  * Agentic RL replaces single-response rollouts with **multi-turn trajectories through an environment**. The Actor loops with the Environment for 5-50 turns; the verifier scores success at the terminal state.
  * **Rollout dominates more than ever** : 95% of compute, vs 80% in M39's single-turn setup. Environment quality determines training quality.
  * **Verifier engineering is most of the work**. Hierarchy: programmatic state checks > unit tests > output equality > trajectory replay > LLM-as-judge with rubric > human grading. The verifiable-beats-judgeable principle: at training scale (millions of rewards), even small verifier noise corrupts the policy.
  * **VAGEN (Feb 2026)** : agentic verifiers via online proactive interaction; OSWorld-Verified accuracy 84.7% → 92.9%.
  * **Local deterministic environments often beat online live-environment RL** for the same model and task. AgentCPM-Explore-4B got +3.8% on GAIA with online; LiteResearcher got +13.4pp with deterministic local. Reward noise dominates online; determinism enables longer training.
  * **Environment-as-a-Service (EaaS)** : hosted environments with control plane / data plane separation; autoscaled sandbox pools (browsers, VMs, containers, databases). The 2026 production pattern.
  * **Open Reward Standard (ORS)** : MCP extension with RL primitives. Decouples environments from trainers; environments published once, consumed from any framework. **OpenReward** : 330+ environments, 4.5M+ tasks via ORS.
  * **Prime Intellect ecosystem** : `verifiers` library (canonical environment format), `prime-rl` trainer (async off-policy with SHARDCAST weight broadcast), **Environments Hub** (1000+ community environments, 250+ creators, 100K+ downloads), Lab platform (Feb 2026, hosted training).
  * **INTELLECT-3 (Dec 2025)** : 106B MoE trained on prime-rl stack scaled to 512 H200s. SOTA for size class. 90.8% AIME 2024, 88.0% AIME 2025. Full open-source release including environments.
  * **Computer-use SOTA** : ComputerRL/AutoGLM-OS-9B at 48.9% on OSWorld via end-to-end online RL with thousands of parallel virtual desktops + Entropulse + API-GUI hybrid action space.
  * **BrowserGym + AgentLab (ServiceNow)** : dominant open-source web-agent gym. Playwright-driven Chromium, multimodal observations (DOM, AXTree, screenshots), Set-of-Marks for element grounding.
  * **Async RL is standard** : rollout(k+1) overlaps train(k); two-step or four-step asynchrony validated; SHARDCAST broadcasts policy weights; two-sided GRPO clipping stabilizes training.
  * **Failure modes specific to agentic RL** : specification gaming (verifier exploitation), tool-call hallucination, trajectory length explosion, stuck-in-loop, environment leakage, long-context degradation, curriculum mismatch. Trajectory-level monitoring beyond M39's gradient-level dashboards.
  * **Vertical specialization** : Cursor post-training models with Cursor itself as the environment. Application-layer companies entering training game. Prime Intellect Lab markets to this. _The 2026-2028 dynamic_.
  * The reflex: when designing an agentic RL system, **environment quality first**. Algorithm choice, model size, compute — all secondary. Get the environment and verifier right; the rest follows.

Module 41 (next, if Part XI continues) would tackle **Blackwell & NVFP4 training** — the hardware update for M14, covering B200/B300/Blackwell Ultra specs, the NVFP4 two-level scaling format, MLPerf v5.1 FP4 training results (3.2× over Hopper FP8), CUTLASS 3.8 for Blackwell tensor cores, and the practical recipe for FP4 training in 2026.
