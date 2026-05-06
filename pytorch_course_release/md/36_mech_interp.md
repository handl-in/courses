# Module 36 — Mechanistic interpretability

# _Mechanistic interpretability:_ SAEs, features & finding what the model is doing

_Part X · Module 36 · interpretability_

— from the superposition hypothesis to sparse autoencoders to production-scale feature dictionaries; the engineering of looking inside the model and reading out what it's computing

\--- 

For 35 modules we've built models, optimized them, served them, and trained them to reason. We've treated the model itself as a black box that maps inputs to outputs — what's _inside_ the model has been outside the scope. This module turns inward. **Mechanistic interpretability** (mech interp) is the engineering discipline of looking at the actual computations a model performs and translating them into something humans can read.

This was research curiosity in 2022. By 2024-2026, it's a production tool. Anthropic uses sparse autoencoders to detect deceptive behaviors in safety evaluations. OpenAI publishes interpretability work alongside capability releases. DeepMind released **Gemma Scope** , a public set of trained SAEs covering every layer of Gemma-2. Mech interp engineers are a hiring category at every major lab. _Yet the engineering side has almost no systematic treatment_. This module fills that gap. By the end you'll know what SAEs are, how they're trained, what the modern variants offer over the original L1 formulation, and what the practical "I want to find what feature does X" workflow looks like.

> **★ KEY IDEA**  
>  Models trained on next-token prediction develop internal representations that don't align cleanly with neurons. **Superposition** : a single neuron typically encodes a mix of unrelated concepts; a single concept is typically distributed across many neurons. To recover the concepts ("features"), train a **sparse autoencoder (SAE)** : a wide encoder + sparse hidden layer + narrow decoder, fit to reconstruct the model's activations. The _dictionary atoms_ in the SAE are the features. **Modern variants** : Top-K SAEs (hard k-sparsity instead of L1, fewer dead features); Gated SAEs (separate gate from magnitude, less shrinkage); JumpReLU / BatchTopK (recent refinements). Feature interpretation is automated: find max-activating examples, summarize what they share. **Feature circuits** compose features across layers. Production deployments train SAEs on every layer of large models — compute cost approaches main-model training. Uses include bias detection, capability evaluation, jailbreak analysis, model editing, distinguishing "the model knows X" from "the model can express X." _Open problem_ : SAEs find useful features but probably not all features, especially in attention heads. Active research; engineering recipes maturing. 

## Two new faces

A

SAE

"I'm a wide, sparse autoencoder. I expand your activations into a basis where each axis means one thing."

Drop me in front of a layer's activations. I encode them into a much wider space — typically 16-128× the model's dimension. With sparsity constraints, only a handful of my dictionary atoms activate per token. _Each atom becomes a feature_ : a direction in activation space that consistently fires for some interpretable concept. I'm just an autoencoder — encoder, sparse hidden, decoder, MSE reconstruction loss plus a sparsity penalty. The trick is the sparsity: with L1 (the original recipe), I'm finicky and produce lots of dead features. With Top-K, I'm robust and most features stay alive. **I'm the dominant approach to disentangling superposition**. Training me on a frontier model is an engineering project all by itself.

F

Feature

"I'm a direction in activation space. I might mean 'mentions of bridges' — or 'beginnings of arithmetic.'"

When SAE trains, I emerge as one of its dictionary atoms. My identity is decided by what activates me. To know what I am, you collect text where I fire and look for patterns: maybe I light up on every mention of San Francisco landmarks; maybe I fire when arithmetic is about to happen; maybe I'm "professional vs casual register." _Some features are crisp and interpretable; others are murky._ Once identified, I can be used: **steered** (artificially activate me, see what the model does), **ablated** (zero me out, watch behavior change), **probed** (use me as a classifier signal). I'm the unit of mechanistic understanding. The Golden Gate Bridge feature is one of me; so is "deception in the model's reasoning." I'm what you discover when you look inside.

## The interpretability question

"What is this model doing?" admits two distinct interpretations. **Behavioral interpretability** : study the input-output mapping. Run prompts; see what the model does; characterize patterns. This is what most ML engineering already does — running evals, characterizing failure modes, understanding model capabilities by what they output.

**Mechanistic interpretability** : study the internal computation. Look at the activations the model produces; trace which inputs cause which neurons to fire; understand the algorithm the model has learned. The goal is to translate the model's billions of weights into something a human can read.

Why isn't behavioral enough? Three concrete reasons:

  1. **Some questions are mechanistic by nature**. "Does this model deceive when its handlers aren't watching?" can't be answered by running prompts (the model behaves differently when watched). It can be answered by examining whether the model has internal representations of "being watched" and "deceiving" that fire under specific conditions.
  2. **Robustness to distribution shift**. Behavioral testing is bounded by the prompts you tested; mechanistic understanding gives a generalization argument ("this circuit always implements X").
  3. **Capability discovery**. A model might possess a capability (math, hacking, social manipulation) that doesn't surface in normal usage but exists internally. Mechanistic analysis can detect dormant capabilities.

For all three, you need to look inside the model. The question becomes: what's there to look at, and what does it mean?

## The superposition hypothesis

The natural starting point: each neuron encodes one concept. Look at a neuron, see what activates it, name the concept. This was the early-2010s vision of interpretability for vision models, and it partially worked there ("this neuron detects edges"; "this neuron detects faces").

For language models, it doesn't work. Look at any individual neuron and you'll find it activates on a confusing mix of concepts: a single neuron in a Llama-2-7B layer might fire on "DNA sequences," "code function definitions," and "lists of presidents." This is **polysemanticity** — one neuron, many meanings — and it's the rule, not the exception.

Why? The **superposition hypothesis** (formalized in Anthropic's "Toy Models of Superposition" paper, 2022): models represent more features than they have dimensions, and they do so by overlapping features in non-orthogonal ways. With sparsity (most features absent in any given input), the model can disambiguate the overlapping representations from context. The result: the model's "features" are not aligned with neurons; they're directions in activation space that may not be axis-aligned at all.

Superposition: N features compressed into D < N dimensions via overlapping 2D activation space (D=2), 6 features (N=6) neuron 1 neuron 2 f1 f2 f3 f4 f5 f6 No feature aligns with neurons 1 or 2 Each neuron fires on multiple features = polysemantic After SAE: 6 features as 6 axis-aligned dictionary atoms f1 "bridges" f2 f3 "arithmetic" f4 f5 f6 "casual" Most features ≈ 0 for any given input — sparse Each axis = one interpretable concept SAE undoes the superposition: from non-orthogonal features in low-dim space to axis-aligned features in a wider space, with most ≈ 0 (sparse)

Read left-to-right. The left panel: a 2D activation space (just 2 neurons) with 6 feature directions packed in non-orthogonally. Each neuron's activation is some mixture of all 6 features — it's polysemantic. The right panel: an SAE expands this 2D space into 6 axis-aligned dimensions where each feature has its own dimension. Most features are zero for any given input (sparse activation), but the ones that fire are interpretable.

The compression argument scales: a real LLM layer has D ≈ 4000-8000 dimensions but probably encodes O(100K-1M) features in superposition. The SAE expands those D dimensions into N >> D dictionary atoms, each axis-aligned. Sparsity makes this possible: if any given input only activates a few of the millions of possible features, you can recover them by linear decoding even with massive overlap.

## The SAE: architecture and training

The basic SAE is dead simple. An encoder-decoder pair with a sparsity constraint on the hidden layer:
    
    
    class SAE(nn.Module):
        def __init__(self, d_model, d_dict):
            super().__init__()
            # Encoder: project from model activation space to dictionary space
            self.encoder = nn.Linear(d_model, d_dict, bias=True)
            # Decoder: project back. Tied or untied weights are both used.
            self.decoder = nn.Linear(d_dict, d_model, bias=True)
            # Decoder bias often initialized to median activation
            # Decoder weights normalized to unit norm — prevents trivial scaling solutions
    
        def forward(self, x):
            # 1. Subtract pre-encoding bias (centers activations)
            x_centered = x - self.decoder.bias
    
            # 2. Encode and apply ReLU for sparsity
            features = F.relu(self.encoder(x_centered))   # [B, T, d_dict]
    
            # 3. Decode
            reconstruction = self.decoder(features)
    
            return features, reconstruction

The sizes matter: `d_dict` is typically 8× to 128× `d_model`. Common ratios in published work:

  * **8×-16× expansion** : smaller, faster to train, fewer interpretable features. Used in early SAE work.
  * **32×-64× expansion** : standard for production SAEs. Gemma Scope mostly uses 16K and 65K dictionaries on Gemma-2-2B's 2304-dim residual stream.
  * **128×+** : experimental. The intuition: more features, more interpretability, but harder training.

### The L1 SAE (the original recipe)

The standard sparsity penalty is L1 on the feature activations:
    
    
    def l1_sae_loss(sae, x, lambda_l1=5e-3):
        features, reconstruction = sae(x)
    
        # Reconstruction loss: MSE between input and reconstruction
        recon_loss = (reconstruction - x).pow(2).mean()
    
        # Sparsity penalty: L1 on feature activations
        # Optionally weighted by decoder norms (controls feature scale)
        decoder_norms = sae.decoder.weight.norm(dim=0)
        sparsity_loss = (features * decoder_norms).abs().sum(dim=-1).mean()
    
        return recon_loss + lambda_l1 * sparsity_loss

The L1 penalty pushes feature activations toward zero. The optimizer balances reconstruction (which wants features active) against sparsity (which wants them zero). The result: only features that meaningfully reduce reconstruction loss stay active for any given input.

The `lambda_l1` coefficient is the central hyperparameter. Too low: SAE produces dense features (every feature active for every input — useless). Too high: SAE produces too few features, reconstruction is bad. Practical range: 1e-4 to 1e-2; tuning is empirical.

Three persistent problems with L1 SAEs:

  * **Dead features** : a feature that never activates after some point in training. Stays at zero forever, contributing nothing. Production runs find 30-50% of features go dead. Wasted capacity.
  * **Shrinkage** : L1 not only pushes inactive features to zero but also pulls active features below their "true" magnitude. The reconstruction is systematically attenuated.
  * **Hyperparameter sensitivity** : lambda_l1 needs to be tuned per layer, per model, per dictionary size. Small changes cause large quality differences.

These problems motivated the modern variants.

### Top-K SAEs (the modern default)

OpenAI's "Scaling and evaluating sparse autoencoders" paper (2024) replaced L1 with a hard constraint: **only the top-K largest activations are kept; the rest are zeroed**. No L1 penalty needed.
    
    
    class TopKSAE(nn.Module):
        def __init__(self, d_model, d_dict, k=32):
            super().__init__()
            self.encoder = nn.Linear(d_model, d_dict, bias=True)
            self.decoder = nn.Linear(d_dict, d_model, bias=True)
            self.k = k
    
        def forward(self, x):
            x_centered = x - self.decoder.bias
    
            # Encode (no ReLU yet — top-K does the gating)
            pre_activation = self.encoder(x_centered)
    
            # Top-K: zero out all but the K largest activations per token
            topk_vals, topk_idx = pre_activation.topk(self.k, dim=-1)
            features = torch.zeros_like(pre_activation)
            features.scatter_(-1, topk_idx, F.relu(topk_vals))
    
            reconstruction = self.decoder(features)
            return features, reconstruction
    
    def topk_sae_loss(sae, x):
        _, reconstruction = sae(x)
        # Just MSE — sparsity is enforced architecturally
        return (reconstruction - x).pow(2).mean()

The benefits compound:

  * **Sparsity is enforced exactly** : K active features per token, period. No tuning of lambda_l1.
  * **Dead-feature rate plummets** : typical Top-K SAE has 90%+ of features alive after training, vs 50-70% for L1.
  * **No shrinkage** : the active features keep their full magnitude.
  * **Hyperparameter is interpretable** : K is "how many features per token" — directly related to expected sparsity.

Typical K values: 32-128 for production SAEs. The paper showed Top-K SAEs strictly dominate L1 SAEs at every dictionary size and reconstruction quality. _If you're training an SAE in 2026, start with Top-K._

### Other modern variants

The post-Top-K landscape has several refinements:

  * **Gated SAEs** (DeepMind, 2024): separate gate (decides which features fire) from magnitude (decides how strongly). Reduces shrinkage further than Top-K. Used in some Gemma Scope SAEs.
  * **JumpReLU SAEs** : replace ReLU with a jump function that has a hard threshold. Combines L0-like sparsity with smooth gradients via straight-through estimators.
  * **BatchTopK** : instead of per-token top-K, take top-K across the whole batch. Gives some tokens more features, others fewer; better for tokens that are "naturally" feature-dense.
  * **Matryoshka SAEs** (2024-2025): nested dictionary structure where the first K_outer features must reconstruct alone, then the next layer adds detail. Trains multi-scale features in a single pass.

The field is active and converging. As of 2026: Top-K and Gated are mainstream; the others are research alternatives with specific use cases. The differences between modern variants matter at the margin; choosing modern over L1 matters substantially.

## Training SAEs at scale

SAE training has its own engineering rhythm distinct from main-model training. The basic flow:

  1. **Collect activations** : run the target model on a large pile of text (often the same pretraining corpus from M35). At each token, save the activation at the layer you want to analyze. Typical scale: 100M to 10B tokens of activations, stored on disk.
  2. **Train the SAE** : run optimization on the activation data. SAEs are usually trained for 1-3 epochs over the activation buffer; training time is a fraction of main-model training (because SAEs are smaller and the loss is per-token-cheap).
  3. **Validate** : hold out activation data; check reconstruction loss and sparsity match the loss-curve targets. Compute "L0" metric: average number of non-zero features per token (should be the K you specified for Top-K, or whatever lambda_l1 produced).

The compute cost is real but manageable. For Gemma-2-2B, training one Top-K SAE on residual-stream activations costs ~$1K of cloud compute. For all 26 layers and multiple positions per layer, multiply. For Sonnet-scale models, multiply again — Anthropic's published Sonnet SAEs likely cost $1M+ to train.

### Key training tricks

SAE training has several recipe details that matter substantially:

  * **Initialize decoder bias to median activation** : the activations being autoencoded usually have a non-zero mean. Initializing the decoder bias to the median (or mean) lets the SAE focus on modeling deviations rather than reconstructing the mean every time.
  * **Normalize decoder weights** : after each step, project each decoder column to unit norm. Without this, the SAE can scale features arbitrarily (down-scale features, up-scale decoder columns; reconstruction is the same but L1 is artificially low). Forcing unit decoder norm makes the L1 penalty meaningful.
  * **fp32 training** : SAEs have small effective gradients. Training in bf16 often produces noticeably worse SAEs. Production runs use fp32 throughout.
  * **Resampling dead features** : periodically check which features haven't fired recently; reinitialize them with a new direction. Cuts dead-feature rate substantially in L1 SAEs (less needed for Top-K).
  * **Auxiliary losses** : some recipes add an "auxiliary k-loss" that uses a higher K to maintain reconstruction stability while the primary loss pushes for the lower-K target.

## Feature interpretation

You've trained an SAE. You have 65,536 features. What do they mean?

The standard pipeline is **auto-interpretation** , formalized by Bills et al. (OpenAI, 2023):

  1. **Find max-activating examples** : for each feature, run the model on a corpus and record the top-K text segments where that feature fires most strongly. Typical: top 20 contexts of 50-100 tokens each.
  2. **Generate explanation** : send the max-activating examples to a strong LLM with a prompt like "What pattern do these texts share?" The LLM produces a candidate explanation: "Mentions of bridges, especially the Golden Gate."
  3. **Score the explanation** : take a held-out set of contexts. Use another LLM to score "given this explanation, would feature X fire here?" Compare predictions to actual feature activations. The score is the explanation quality.

This pipeline is fully automated. Anthropic's "Scaling Monosemanticity" paper used variants of this on millions of features at Sonnet scale. The output is a labeled dictionary: feature 14722 → "Golden Gate Bridge"; feature 5829 → "code function definitions"; feature 31204 → "expressing emotional vulnerability."

Quality varies. Some features are crisp and the explanation matches every max-activating example. Some features are murky — they fire on a mix of related concepts that don't have a clean unifying description. The published metrics: ~30-50% of features get high-quality auto-interpretations; the rest are partially interpreted or remain mysterious.

### What does a feature actually look like?

A concrete example, drawn from published Anthropic work. They identified a "Golden Gate Bridge" feature in their Sonnet SAE — feature 34M something. The signature:

  * **Max-activating contexts** : text mentioning the Golden Gate Bridge (English, French, Spanish, Chinese), images of the bridge, references to Marin County and San Francisco landmarks.
  * **Steering effect** : artificially boosting this feature's activation during generation makes the model bring up the Golden Gate Bridge in unrelated contexts. Anthropic's "Golden Gate Claude" demo had the model respond to "What's your favorite color?" with stories about driving across the bridge.
  * **Ablation effect** : zeroing this feature during the model's normal processing has subtle effects — the model still understands the concept but is less likely to mention it spontaneously.

The feature is multimodal (fires on text and images of the same concept), multilingual (fires regardless of language), and abstract enough to capture "the concept of the Golden Gate Bridge" rather than just the literal string. _This is what successful feature discovery looks like._

## Feature circuits

Individual features are useful. The bigger question: how do features compose? Why does feature B in layer 23 activate? Probably because some combination of earlier features in earlier layers caused it. **Feature circuits** are the connections.

The standard technique is **path patching** (or **attribution patching** as a faster approximation): for each earlier feature, ablate it and see how much the later feature's activation changes. The earlier features whose ablation has large effects are "in the circuit" feeding the later feature.
    
    
    def attribution_patching(model, sae_early, sae_late, prompt):
        # Run with hooks to save activations and gradients of late feature.
        early_features, late_features = collect_features(model, sae_early, sae_late, prompt)
    
        # For each late feature, compute gradient w.r.t. each early feature.
        # An early-feature contribution = early_activation × ∂(late_activation)/∂(early_activation)
        # Attribution patch: linearized estimate of effect of zeroing each early feature.
        contributions = early_features * grads_of_late_w.r.t.early
        # Top contributors = circuit feeding into the late feature
        return contributions.topk(10)

Attribution patching is an approximation — actual path patching ablates each feature and re-runs the model, which is more accurate but vastly more expensive. The approximation is usually sufficient; production circuit discovery uses attribution.

The output is a directed graph: features at layer N point to features at layer N+1 (and beyond) that they cause. A complete circuit might span 5-15 layers, with 20-100 features participating. The narrative: "input contains arithmetic → arithmetic-detection feature in layer 5 → digit-tracking features in layer 8 → addition-result features in layer 14 → output token logits."

Finding circuits at scale is still an active research area. The published work has discovered specific circuits for tasks like indirect object identification, modular arithmetic, factual recall. Discovering circuits automatically across a model is largely future work.

## Production scale: Gemma Scope and Sonnet features

Two public reference points for production-scale SAE work:

  * **Gemma Scope** (DeepMind, 2024): trained SAEs on every layer of Gemma-2-2B, Gemma-2-9B, and Gemma-2-27B. Multiple SAE configurations per layer (different K, different expansion factors). Released openly. This is the largest public SAE release; a research baseline.
  * **Sonnet SAEs** (Anthropic, 2024-2025): SAEs at Claude Sonnet scale. Not openly released, but published results showed millions of features discovered. Used internally for safety evaluation.

The compute scaling: SAE compute is roughly proportional to (model dimension × dictionary expansion × tokens trained). For a 27B model with 4608-dim residual stream, 64× expansion = 295K dictionary, training on 1B tokens of activations: ~50 H100-days per SAE. Across 60 layers and multiple sites per layer, that's 1000s of H100-days for a complete SAE coverage of one model.

Despite the cost, the trend is toward more SAE coverage at every model release. Frontier labs increasingly view interpretability infrastructure as a complement to capability training — you ship a model alongside its SAE-derived feature dictionary.

## What interpretability is actually used for

The cynical question: with all this engineering, what do the features actually do? The honest 2026 answer:

  * **Bias and safety detection** : find features that fire on demographic categories, hate speech, deceptive reasoning. Use these as classifiers for safety filtering. Anthropic uses interpretability-derived features in their safety stack.
  * **Capability evaluation** : detect whether a model has learned a specific capability (e.g., "knows how to synthesize biological weapons") even if it never expresses that capability in normal usage. Active research, used in dangerous-capability evaluations.
  * **Model editing** : artificially boost or suppress specific features at inference time to change behavior without retraining. Steering vectors are a related technique. Production uses include "make the model less likely to refuse benign requests" and "increase the model's tendency to cite sources."
  * **Jailbreak analysis** : when a model is jailbroken, what changes internally? Compare features that fire under normal prompts vs jailbroken prompts. Helps understand the attack surface and design defenses.
  * **Distinguishing knowledge from expression** : a model can "know" something internally without expressing it (e.g., the model has accurate internal representations of geography but produces fluent-but-wrong outputs). Features can detect this gap, useful for hallucination analysis.
  * **Debugging training failures** : when post-training degrades a capability, examine which features changed. Often reveals the training signal misalignment (e.g., RLHF caused a "factual confidence" feature to be suppressed).

The first use case (safety detection) is now production. The others are mature-research-with-some-production-use. The trajectory is clear: _interpretability is becoming a standard layer in the ML stack_ , alongside training, evaluation, and serving.

## Limitations and open problems

SAEs find _many useful features_ , but probably _not all features_. Specific limitations to track:

  * **Attention features are harder than residual-stream features**. SAEs work cleanly on the residual stream; attention head outputs and weights are less SAE-amenable. Active research.
  * **Some features are inherently distributed**. Concepts that don't have a clean local representation (long-range dependencies, complex compositions) may not appear as single SAE features even at large dictionary sizes.
  * **Feature universality is partial**. Two SAEs trained with different random seeds find some of the same features and some different ones. The "real" features the model uses may not be discoverable up to permutation.
  * **Reconstruction quality has a ceiling**. Even at very large dictionary sizes, SAEs reconstruct only ~85-95% of the original activation. The remainder is genuinely high-dimensional structure that resists sparse decomposition.
  * **The "model thought it" interpretation is approximate**. SAE features are correlations between the model's activations and human-interpretable concepts. The model's actual computation may use a slightly different decomposition that the SAE only approximates.

The honest summary: **SAEs are the best tool we have for mechanistic interpretability of LLMs in 2026, and they are sufficient to be useful in production for specific tasks**. They're not the final word, and the field is rapidly producing better techniques. Interpretability research and engineering will likely look quite different in 5 years.

#### Q&A; — About interpretability and SAEs **Q:** Why train SAEs on activations rather than weights? **A:** Activations are where the model's "thinking" happens — what it's currently representing for the input it's processing. Weights are static and represent the model's learned knowledge but not its current cognitive state. A feature like "the model is currently thinking about the Golden Gate Bridge" only makes sense as an activation pattern; it's not localized in any specific weight. There are weight-level interpretability techniques (analyzing attention head matrices, MLP weights), but they're typically less successful than activation-based methods for LLMs at scale. **Q:** Why doesn't every neuron just learn to represent one feature, eliminating polysemanticity? **A:** Two reasons. (1) **Capacity constraint** : with D neurons and far more than D features the model needs, it can't have one neuron per feature. Superposition is forced by capacity scarcity. (2) **Optimization pressure** : even if the model had enough capacity in principle, gradient descent doesn't have a strong incentive to allocate one feature per neuron. The losses don't penalize polysemanticity. The model finds _some_ functional decomposition that minimizes loss; that decomposition isn't pretty. SAEs are useful precisely because they impose the human-friendly decomposition the model didn't naturally produce. **Q:** If features can be steered, doesn't that mean the SAE is finding what the model uses, not just what's correlated? **A:** Steering effects provide stronger evidence than correlation alone, but it's not airtight. Possible interpretations of "boosting feature X causes the model to talk about Y": (a) the model uses feature X to represent Y, and steering boosts the representation; (b) the SAE's feature X is correlated with the model's true representation of Y, and steering happens to push activations in a direction that overlaps the true representation; (c) some other interaction. Interpreting steering effects requires care. _That said_ : when steering works reliably across many contexts, the simplest explanation is usually that the SAE found something close to what the model is actually using. **Q:** Can I use SAE features as inputs to a probe classifier instead of training a classifier from scratch? **A:** Yes, and this is one of the most practical uses of SAEs. Once you have a feature dictionary, you can build classifiers as simple linear combinations of features. Want a "deception detector"? Find features related to deception, weight them, threshold. Often performs comparably to training a probe from scratch on the original activations, but with the huge advantage of being interpretable: you can see _why_ the classifier fires (which features lit up). Production safety stacks at frontier labs use this pattern extensively. **Q:** Why is fp32 needed for SAE training when bf16 is fine for the underlying model? **A:** SAE training has unusual gradient characteristics: most features are inactive most of the time (sparsity), so most parameter gradients are zero or near-zero per step. The signal-to-noise ratio in any given gradient update is much lower than in standard training. bf16's reduced precision adds noise that can swamp the small effective signal, leading to noticeably worse SAEs. fp32 preserves the precision needed for the small gradients to accumulate correctly. The compute cost is acceptable because SAEs are small relative to the main model. _If your SAE training looks unstable or has high dead-feature rates, switching from bf16 to fp32 often fixes it._ **Q:** What's the relationship between SAEs and LoRA / model editing more broadly? **A:** They're complementary techniques operating at different scales. LoRA (M11-adjacent) modifies weight matrices to change model behavior across many inputs; SAE-based steering modifies activations to change behavior at specific moments. LoRA is permanent (until you discard the adapter); steering is per-inference. For some applications (safety, persona shifts, reasoning style), SAE-based steering is more flexible because it's controllable at runtime. For others (specialized capabilities, format following), LoRA is more efficient because the change is built in. _Production deployments increasingly combine both_ : LoRA for capability shaping, SAE-features for runtime control. 

## Code Magnets: implement a Top-K SAE forward pass

You're writing the forward pass for a Top-K SAE. Three magnets are wrong choices.

Arrange the magnets to compute features and reconstruction.

def forward(self, x): x_centered = x - self.decoder.bias x_centered = x pre_activation = self.encoder(x_centered) topk_vals, topk_idx = pre_activation.topk(self.k, dim=-1) topk_vals, topk_idx = pre_activation.abs().topk(self.k, dim=-1) features = torch.zeros_like(pre_activation) features.scatter_(-1, topk_idx, F.relu(topk_vals)) features = F.relu(pre_activation) reconstruction = self.decoder(features) return features, reconstruction

show solution
    
    
    def forward(self, x):
        x_centered = x - self.decoder.bias
        pre_activation = self.encoder(x_centered)
        topk_vals, topk_idx = pre_activation.topk(self.k, dim=-1)
        features = torch.zeros_like(pre_activation)
        features.scatter_(-1, topk_idx, F.relu(topk_vals))
        reconstruction = self.decoder(features)
        return features, reconstruction

The traps:

  * `x_centered = x` (no bias subtraction): skips the centering step. The decoder bias represents the mean of the activation distribution; without subtracting it before encoding, the SAE has to spend dictionary capacity reconstructing the mean for every input. Reconstruction quality degrades and many features end up encoding the offset rather than meaningful structure. Always subtract the decoder bias before encoding.
  * `topk_vals, topk_idx = pre_activation.abs().topk(self.k, dim=-1)`: takes top-K by absolute value. Wrong direction. We want the K most-positive activations (which will then pass through ReLU). Taking by absolute value would include large negative pre-activations, but those get zeroed by ReLU anyway — and worse, we'd be selecting them at the expense of smaller-but-positive activations that should be the actual features. Standard top-K on the raw pre-activation is correct.
  * `features = F.relu(pre_activation)`: replaces the top-K mechanism entirely with a plain ReLU. This is the L1 SAE form, not Top-K. Without the top-K masking, sparsity isn't enforced architecturally — every positive pre-activation passes through. You'd then need an L1 penalty (which is what the L1 SAE does), but without one, the SAE produces dense features. The whole point of Top-K is the hard sparsity constraint.

The pattern: **subtract decoder bias → encode → top-K by raw value (not absolute) → scatter into zero tensor with ReLU on values → decode**. Each step has a specific role: centering, projecting up, hard sparsity constraint, applying ReLU only to the top-K values, projecting back.

## Who does what?

Match each interpretability concept to its real role.

Concept

Real role

Polysemanticity

A. A single neuron encodes multiple unrelated concepts; the rule, not the exception.

Superposition hypothesis

B. Models represent more features than dimensions by overlapping in non-orthogonal directions.

Sparse autoencoder (SAE)

C. Wide encoder + sparse hidden + narrow decoder; dictionary atoms become interpretable features.

Top-K sparsity

D. Architectural sparsity: only the K largest activations per token survive; dramatically reduces dead features.

Auto-interpretation

E. LLM-driven pipeline: find max-activating examples, summarize the pattern, score the explanation.

Path patching / attribution

F. Discover feature circuits by ablating earlier features and measuring effect on later ones.

Steering

G. Artificially boost or suppress a specific feature at inference time to change model behavior.

show solution

**Polysemanticity** → A  
**Superposition** → B  
**SAE** → C  
**Top-K sparsity** → D  
**Auto-interpretation** → E  
**Path patching / attribution** → F  
**Steering** → G 

The mental shortcut: _neurons are polysemantic because of superposition, SAEs disentangle them, Top-K enforces sparsity, auto-interp labels them, attribution finds circuits, steering manipulates them_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team trains an L1 SAE on Llama-3-8B's layer-15 residual stream with 64× expansion (262K features). After training, they find that 60% of features are dead (never fire). Reconstruction quality is mediocre. What would you change first?

show answer

Switch to Top-K. The L1 SAE's combination of "L1 sparsity penalty" plus "easy-to-zero ReLU" produces high dead-feature rates as a structural property — features that get pushed below the activation threshold early in training tend to stay there. Top-K replaces the L1 penalty with a hard architectural constraint: exactly K features active per token. Dead features can't form because every token must have K active features.

Empirically, switching from L1 to Top-K typically reduces dead-feature rates from 30-60% to 5-10% with the same dictionary size, no L1 tuning required. Reconstruction quality usually improves because the surviving features have full magnitude (no L1 shrinkage).

Other things to try if Top-K alone doesn't fix it: (a) **resampling** dead features periodically (reinitialize them with new directions); (b) **auxiliary k-loss** with higher K to maintain stability; (c) **fp32 training** if currently in bf16; (d) **longer training** if the SAE was trained for too few steps. But Top-K is the highest-leverage single change.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why might steering a "Golden Gate Bridge" feature successfully cause the model to mention the bridge in unrelated contexts, while steering a more abstract feature (e.g., "deception") might fail or produce strange behavior?

show answer

Several reasons that compound:

(1) **Concrete concepts have crisper representations**. "Golden Gate Bridge" is a specific named entity with consistent properties across contexts. The model likely has a relatively localized representation of it. Boosting that representation reliably causes the concept to be expressed.

(2) **Abstract concepts are likely distributed**. "Deception" doesn't have a single object referent; it's a property that depends on the relationship between many concepts (intent, falsehood, audience awareness). The model's representation may be spread across many features, and an SAE-discovered "deception feature" may capture only one component.

(3) **Causal vs correlational features**. The SAE finds features correlated with the model's behavior, but the causal structure may be more complex. A feature that lights up during deceptive outputs may not be the cause of those outputs — it might be a downstream consequence. Steering it would then be like adjusting the thermometer rather than the thermostat.

(4) **Off-distribution effects**. Steering is artificial — boosting a feature beyond its natural range pushes the model into off-distribution territory. For "Golden Gate Bridge," the feature is grounded enough that off-distribution boosting still produces sensible outputs. For "deception," boosting can produce incoherent or self-defeating behavior because the model is being pushed into states it's never naturally been in.

The general lesson: _SAE features are most useful for concrete, localized concepts_. They become less reliable for abstract or distributed concepts. This is an active research limitation — finding faithful representations of abstract concepts is harder than finding object-level features.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** A safety team wants to use SAE features to detect when the model is "considering producing harmful content" before it actually outputs anything. Walk through how you'd build this.

show answer

Step-by-step approach:

(1) **Train (or obtain) an SAE** on a layer of the model where high-level reasoning is happening — typically a mid-to-late layer's residual stream. Public Gemma Scope SAEs work for Gemma; for proprietary models, train your own.

(2) **Find candidate features** : collect a curated set of (prompt, harmful-trajectory-response) and (prompt, safe-trajectory-response) pairs. For each prompt, run the model; record SAE feature activations at the relevant layer for both trajectories. Find features that consistently differ between the two: "fires more during harmful trajectories than safe ones."

(3) **Validate by auto-interpretation** : for each candidate feature, run the auto-interp pipeline (max-activating examples, LLM-generated explanation). Confirm features have plausible interpretations (e.g., "discussion of weapon manufacturing," "instructions for evading security").

(4) **Build a classifier** : linear combination of safety-relevant features, trained as a binary classifier on the curated pairs. Linearity matters for interpretability — you want to know which features triggered the alert.

(5) **Deploy in serving** : at inference time, after the model has processed the prompt but before/during generation, extract activations at the chosen layer, run the SAE encoder to get features, run the classifier. If it fires above threshold, route the response through additional checks (or refuse).

**Caveats** : this is a complement to behavioral safety, not a replacement. SAE-derived classifiers have false positives (the model considers but rejects harmful content; the classifier flags this) and false negatives (the model uses representations not captured by the SAE). Calibrate thresholds carefully; combine with output-level filtering. Anthropic and others use variants of this in production safety stacks.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Two SAEs are trained on the same layer of the same model with different random seeds. They find some of the same features and some different ones. Why isn't there a single canonical "true feature set"?

show answer

Several reasons that compound:

(1) **Optimization landscape has many local optima**. The SAE training problem (minimize reconstruction subject to sparsity) is non-convex. Different initializations converge to different solutions, all of which achieve similar loss but partition activation space differently.

(2) **Some features may be genuinely arbitrary linear combinations**. If the model's representation is "the sum of feature A and feature B always appears together," an SAE might decompose this into (A, B) or into (A+B, A-B) or any other rotation. All are valid solutions to the reconstruction problem; the model's "true" decomposition may be undefined.

(3) **Dictionary size is finite**. With limited capacity, the SAE has to make tradeoffs about which features to allocate capacity to. Different runs make different tradeoffs based on training dynamics.

(4) **Feature universality is partial empirically**. The "feature universality" hypothesis (different SAEs find the same features) is supported for many concrete features (entity-like) but weaker for abstract features. Some features genuinely vary across SAE runs.

(5) **The model's "true" features may not be a fixed set**. The model's computation may not have a unique canonical decomposition. SAEs find _useful_ approximations; the existence of a "ground truth" set of features is an assumption, not a proven property of trained models.

The practical implication: when claiming a feature exists, it's safer to verify across multiple SAE training runs (or compare published SAEs from different labs). High-confidence features replicate; speculation about features that only appear in one SAE run should be tagged as such.

### What just happened?

  * **Mechanistic interpretability** studies the internal computation of models, not just inputs and outputs. By 2026, it's a production tool for safety, evaluation, and debugging.
  * **Polysemanticity** : a single neuron encodes multiple unrelated concepts. The rule, not the exception, in language models.
  * **Superposition hypothesis** : models represent more features than they have dimensions by overlapping them in non-orthogonal directions. With sparsity (most features zero per input), this works.
  * **Sparse autoencoders (SAEs)** disentangle superposition. Encoder-decoder pair with sparsity constraint; the dictionary atoms become interpretable features.
  * **L1 SAE** (the original): MSE reconstruction + L1 penalty on feature activations. Has dead-feature, shrinkage, and hyperparameter problems.
  * **Top-K SAE** (modern default): hard sparsity — only K features active per token. Much lower dead-feature rate, no L1 tuning. _Use Top-K in 2026 unless you have a specific reason not to_.
  * **Other variants** : Gated SAEs (separate gate and magnitude, less shrinkage), JumpReLU, BatchTopK, Matryoshka SAEs.
  * **Training tricks** : initialize decoder bias to median activation; normalize decoder weights to unit norm; use fp32; resample dead features periodically.
  * **Auto-interpretation pipeline** : find max-activating examples for each feature; have an LLM summarize the pattern; have another LLM score the explanation against held-out activations. ~30-50% of features get high-quality interpretations.
  * **Feature circuits** : how features compose. Path patching (or attribution patching as approximation) finds which earlier features feed into a later one.
  * **Production scale** : Gemma Scope (DeepMind, public, all layers of Gemma-2 family); Anthropic Sonnet SAEs (internal). Compute cost approaches a fraction of main-model training.
  * **Real uses** : safety detection, capability evaluation, model editing via steering, jailbreak analysis, distinguishing knowledge from expression, debugging training failures.
  * **Limitations** : SAEs find many useful features but probably not all features. Attention features harder than residual-stream. Some concepts are inherently distributed. Universality is partial.
  * The reflex: when you need to understand "what is the model doing here?" — train an SAE, find the features, validate via steering, build classifiers from features. The interpretability stack is engineering now, not just research.

Module 37 (if Part X continues) would tackle **production eval engineering** — non-saturating benchmarks, contamination resistance, statistical significance for LLM evals, LLM-as-judge bias, calibration. Most teams' biggest weakness; under-rigorous in most setups.
