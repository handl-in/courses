#!/usr/bin/env python3
"""Module 36: Mechanistic interpretability — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part X · Module 36 · interpretability</div>
  <h1 class="module-title"><em>Mechanistic interpretability:</em> SAEs, features &amp; finding what the model is doing</h1>
  <p class="module-sub">— from the superposition hypothesis to sparse autoencoders to production-scale feature dictionaries; the engineering of looking inside the model and reading out what it's computing</p>
</div>

<p>For 35 modules we've built models, optimized them, served them, and trained them to reason. We've treated the model itself as a black box that maps inputs to outputs — what's <em>inside</em> the model has been outside the scope. This module turns inward. <strong>Mechanistic interpretability</strong> (mech interp) is the engineering discipline of looking at the actual computations a model performs and translating them into something humans can read.</p>

<p>This was research curiosity in 2022. By 2024-2026, it's a production tool. Anthropic uses sparse autoencoders to detect deceptive behaviors in safety evaluations. OpenAI publishes interpretability work alongside capability releases. DeepMind released <strong>Gemma Scope</strong>, a public set of trained SAEs covering every layer of Gemma-2. Mech interp engineers are a hiring category at every major lab. <em>Yet the engineering side has almost no systematic treatment</em>. This module fills that gap. By the end you'll know what SAEs are, how they're trained, what the modern variants offer over the original L1 formulation, and what the practical "I want to find what feature does X" workflow looks like.</p>

<div class="keyidea">
Models trained on next-token prediction develop internal representations that don't align cleanly with neurons. <strong>Superposition</strong>: a single neuron typically encodes a mix of unrelated concepts; a single concept is typically distributed across many neurons. To recover the concepts ("features"), train a <strong>sparse autoencoder (SAE)</strong>: a wide encoder + sparse hidden layer + narrow decoder, fit to reconstruct the model's activations. The <em>dictionary atoms</em> in the SAE are the features. <strong>Modern variants</strong>: Top-K SAEs (hard k-sparsity instead of L1, fewer dead features); Gated SAEs (separate gate from magnitude, less shrinkage); JumpReLU / BatchTopK (recent refinements). Feature interpretation is automated: find max-activating examples, summarize what they share. <strong>Feature circuits</strong> compose features across layers. Production deployments train SAEs on every layer of large models — compute cost approaches main-model training. Uses include bias detection, capability evaluation, jailbreak analysis, model editing, distinguishing "the model knows X" from "the model can express X." <em>Open problem</em>: SAEs find useful features but probably not all features, especially in attention heads. Active research; engineering recipes maturing.
</div>

<h2>Two new faces</h2>

<div class="character" style="--c: #1f5f5b;">
  <div class="avatar" style="background: #1f5f5b; color: #fff;">A</div>
  <div>
    <p class="who">SAE</p>
    <p class="name">"I'm a wide, sparse autoencoder. I expand your activations into a basis where each axis means one thing."</p>
    <p class="says">Drop me in front of a layer's activations. I encode them into a much wider space — typically 16-128× the model's dimension. With sparsity constraints, only a handful of my dictionary atoms activate per token. <em>Each atom becomes a feature</em>: a direction in activation space that consistently fires for some interpretable concept. I'm just an autoencoder — encoder, sparse hidden, decoder, MSE reconstruction loss plus a sparsity penalty. The trick is the sparsity: with L1 (the original recipe), I'm finicky and produce lots of dead features. With Top-K, I'm robust and most features stay alive. <strong>I'm the dominant approach to disentangling superposition</strong>. Training me on a frontier model is an engineering project all by itself.</p>
  </div>
</div>

<div class="character" style="--c: #d4a017;">
  <div class="avatar" style="background: #d4a017; color: #fff;">F</div>
  <div>
    <p class="who">Feature</p>
    <p class="name">"I'm a direction in activation space. I might mean 'mentions of bridges' — or 'beginnings of arithmetic.'"</p>
    <p class="says">When SAE trains, I emerge as one of its dictionary atoms. My identity is decided by what activates me. To know what I am, you collect text where I fire and look for patterns: maybe I light up on every mention of San Francisco landmarks; maybe I fire when arithmetic is about to happen; maybe I'm "professional vs casual register." <em>Some features are crisp and interpretable; others are murky.</em> Once identified, I can be used: <strong>steered</strong> (artificially activate me, see what the model does), <strong>ablated</strong> (zero me out, watch behavior change), <strong>probed</strong> (use me as a classifier signal). I'm the unit of mechanistic understanding. The Golden Gate Bridge feature is one of me; so is "deception in the model's reasoning." I'm what you discover when you look inside.</p>
  </div>
</div>

<h2>The interpretability question</h2>

<p>"What is this model doing?" admits two distinct interpretations. <strong>Behavioral interpretability</strong>: study the input-output mapping. Run prompts; see what the model does; characterize patterns. This is what most ML engineering already does — running evals, characterizing failure modes, understanding model capabilities by what they output.</p>

<p><strong>Mechanistic interpretability</strong>: study the internal computation. Look at the activations the model produces; trace which inputs cause which neurons to fire; understand the algorithm the model has learned. The goal is to translate the model's billions of weights into something a human can read.</p>

<p>Why isn't behavioral enough? Three concrete reasons:</p>

<ol>
  <li><strong>Some questions are mechanistic by nature</strong>. "Does this model deceive when its handlers aren't watching?" can't be answered by running prompts (the model behaves differently when watched). It can be answered by examining whether the model has internal representations of "being watched" and "deceiving" that fire under specific conditions.</li>
  <li><strong>Robustness to distribution shift</strong>. Behavioral testing is bounded by the prompts you tested; mechanistic understanding gives a generalization argument ("this circuit always implements X").</li>
  <li><strong>Capability discovery</strong>. A model might possess a capability (math, hacking, social manipulation) that doesn't surface in normal usage but exists internally. Mechanistic analysis can detect dormant capabilities.</li>
</ol>

<p>For all three, you need to look inside the model. The question becomes: what's there to look at, and what does it mean?</p>

<h2>The superposition hypothesis</h2>

<p>The natural starting point: each neuron encodes one concept. Look at a neuron, see what activates it, name the concept. This was the early-2010s vision of interpretability for vision models, and it partially worked there ("this neuron detects edges"; "this neuron detects faces").</p>

<p>For language models, it doesn't work. Look at any individual neuron and you'll find it activates on a confusing mix of concepts: a single neuron in a Llama-2-7B layer might fire on "DNA sequences," "code function definitions," and "lists of presidents." This is <strong>polysemanticity</strong> — one neuron, many meanings — and it's the rule, not the exception.</p>

<p>Why? The <strong>superposition hypothesis</strong> (formalized in Anthropic's "Toy Models of Superposition" paper, 2022): models represent more features than they have dimensions, and they do so by overlapping features in non-orthogonal ways. With sparsity (most features absent in any given input), the model can disambiguate the overlapping representations from context. The result: the model's "features" are not aligned with neurons; they're directions in activation space that may not be axis-aligned at all.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 360" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrSp" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Superposition: N features compressed into D &lt; N dimensions via overlapping</text>

  <!-- LEFT: low-dim activation space with many non-orthogonal feature directions -->
  <g transform="translate(50, 50)">
    <text x="140" y="16" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">2D activation space (D=2), 6 features (N=6)</text>

    <!-- coordinate axes -->
    <line x1="20" y1="200" x2="260" y2="200" stroke="#1a1612" stroke-width="1"/>
    <line x1="140" y1="60" x2="140" y2="240" stroke="#1a1612" stroke-width="1"/>
    <text x="265" y="205" font-size="9" fill="#6b5d4f">neuron 1</text>
    <text x="120" y="55" font-size="9" fill="#6b5d4f">neuron 2</text>

    <!-- 6 feature directions, NOT axis-aligned -->
    <line x1="140" y1="200" x2="240" y2="105" stroke="#c1502e" stroke-width="2" marker-end="url(#arrSp)"/>
    <text x="245" y="100" font-size="9" font-weight="700" fill="#c1502e">f1</text>
    <line x1="140" y1="200" x2="220" y2="155" stroke="#1f5f5b" stroke-width="2" marker-end="url(#arrSp)"/>
    <text x="225" y="155" font-size="9" font-weight="700" fill="#1f5f5b">f2</text>
    <line x1="140" y1="200" x2="180" y2="80" stroke="#d4a017" stroke-width="2" marker-end="url(#arrSp)"/>
    <text x="183" y="75" font-size="9" font-weight="700" fill="#d4a017">f3</text>
    <line x1="140" y1="200" x2="100" y2="80" stroke="#b85a6c" stroke-width="2" marker-end="url(#arrSp)"/>
    <text x="85" y="75" font-size="9" font-weight="700" fill="#b85a6c">f4</text>
    <line x1="140" y1="200" x2="55" y2="160" stroke="#6b5d4f" stroke-width="2" marker-end="url(#arrSp)"/>
    <text x="40" y="160" font-size="9" font-weight="700" fill="#6b5d4f">f5</text>
    <line x1="140" y1="200" x2="60" y2="105" stroke="#1a1612" stroke-width="2" marker-end="url(#arrSp)"/>
    <text x="50" y="100" font-size="9" font-weight="700" fill="#1a1612">f6</text>

    <text x="140" y="260" text-anchor="middle" font-size="9" fill="#1a1612">No feature aligns with neurons 1 or 2</text>
    <text x="140" y="274" text-anchor="middle" font-size="9" fill="#1a1612">Each neuron fires on multiple features = polysemantic</text>
  </g>

  <!-- RIGHT: SAE expansion to wider space where features are axis-aligned -->
  <g transform="translate(380, 50)">
    <text x="170" y="16" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">After SAE: 6 features as 6 axis-aligned dictionary atoms</text>

    <!-- 6 axis-aligned bars -->
    <g transform="translate(20, 80)">
      <rect x="0"   y="50" width="40" height="80" fill="#c1502e" opacity="0.3"/>
      <rect x="0"   y="50" width="40" height="80" fill="none" stroke="#c1502e" stroke-width="1.5"/>
      <text x="20" y="145" text-anchor="middle" font-size="10" font-weight="700" fill="#c1502e">f1</text>
      <text x="20" y="158" text-anchor="middle" font-size="8" fill="#1a1612">"bridges"</text>

      <rect x="50"  y="100" width="40" height="30" fill="#1f5f5b" opacity="0.3"/>
      <rect x="50"  y="100" width="40" height="30" fill="none" stroke="#1f5f5b" stroke-width="1.5"/>
      <text x="70" y="145" text-anchor="middle" font-size="10" font-weight="700" fill="#1f5f5b">f2</text>

      <rect x="100" y="40" width="40" height="90" fill="#d4a017" opacity="0.3"/>
      <rect x="100" y="40" width="40" height="90" fill="none" stroke="#d4a017" stroke-width="1.5"/>
      <text x="120" y="145" text-anchor="middle" font-size="10" font-weight="700" fill="#d4a017">f3</text>
      <text x="120" y="158" text-anchor="middle" font-size="8" fill="#1a1612">"arithmetic"</text>

      <rect x="150" y="120" width="40" height="10" fill="#b85a6c" opacity="0.3"/>
      <rect x="150" y="120" width="40" height="10" fill="none" stroke="#b85a6c" stroke-width="1.5"/>
      <text x="170" y="145" text-anchor="middle" font-size="10" font-weight="700" fill="#b85a6c">f4</text>

      <rect x="200" y="120" width="40" height="10" fill="#6b5d4f" opacity="0.3"/>
      <rect x="200" y="120" width="40" height="10" fill="none" stroke="#6b5d4f" stroke-width="1.5"/>
      <text x="220" y="145" text-anchor="middle" font-size="10" font-weight="700" fill="#6b5d4f">f5</text>

      <rect x="250" y="60" width="40" height="70" fill="#1a1612" opacity="0.3"/>
      <rect x="250" y="60" width="40" height="70" fill="none" stroke="#1a1612" stroke-width="1.5"/>
      <text x="270" y="145" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">f6</text>
      <text x="270" y="158" text-anchor="middle" font-size="8" fill="#1a1612">"casual"</text>

      <line x1="0" y1="130" x2="290" y2="130" stroke="#1a1612" stroke-width="0.5"/>
    </g>

    <text x="170" y="260" text-anchor="middle" font-size="9" fill="#1a1612">Most features ≈ 0 for any given input — sparse</text>
    <text x="170" y="274" text-anchor="middle" font-size="9" fill="#1a1612">Each axis = one interpretable concept</text>
  </g>

  <text x="370" y="338" font-family="'Caveat', cursive" font-size="20" fill="#c1502e" text-anchor="middle">SAE undoes the superposition: from non-orthogonal features in low-dim space</text>
  <text x="370" y="358" font-family="'Caveat', cursive" font-size="20" fill="#1f5f5b" text-anchor="middle">to axis-aligned features in a wider space, with most ≈ 0 (sparse)</text>
</svg>
</div>

<p>Read left-to-right. The left panel: a 2D activation space (just 2 neurons) with 6 feature directions packed in non-orthogonally. Each neuron's activation is some mixture of all 6 features — it's polysemantic. The right panel: an SAE expands this 2D space into 6 axis-aligned dimensions where each feature has its own dimension. Most features are zero for any given input (sparse activation), but the ones that fire are interpretable.</p>

<p>The compression argument scales: a real LLM layer has D ≈ 4000-8000 dimensions but probably encodes O(100K-1M) features in superposition. The SAE expands those D dimensions into N >> D dictionary atoms, each axis-aligned. Sparsity makes this possible: if any given input only activates a few of the millions of possible features, you can recover them by linear decoding even with massive overlap.</p>

<h2>The SAE: architecture and training</h2>

<p>The basic SAE is dead simple. An encoder-decoder pair with a sparsity constraint on the hidden layer:</p>

<pre><code><span class="kw">class</span> <span class="ty">SAE</span>(nn.Module):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, d_model, d_dict):
        <span class="fn">super</span>().<span class="fn">__init__</span>()
        <span class="com"># Encoder: project from model activation space to dictionary space</span>
        self.encoder = nn.<span class="fn">Linear</span>(d_model, d_dict, bias=<span class="kw">True</span>)
        <span class="com"># Decoder: project back. Tied or untied weights are both used.</span>
        self.decoder = nn.<span class="fn">Linear</span>(d_dict, d_model, bias=<span class="kw">True</span>)
        <span class="com"># Decoder bias often initialized to median activation</span>
        <span class="com"># Decoder weights normalized to unit norm — prevents trivial scaling solutions</span>

    <span class="kw">def</span> <span class="fn">forward</span>(self, x):
        <span class="com"># 1. Subtract pre-encoding bias (centers activations)</span>
        x_centered = x - self.decoder.bias

        <span class="com"># 2. Encode and apply ReLU for sparsity</span>
        features = F.<span class="fn">relu</span>(self.<span class="fn">encoder</span>(x_centered))   <span class="com"># [B, T, d_dict]</span>

        <span class="com"># 3. Decode</span>
        reconstruction = self.<span class="fn">decoder</span>(features)

        <span class="kw">return</span> features, reconstruction</code></pre>

<p>The sizes matter: <code>d_dict</code> is typically 8× to 128× <code>d_model</code>. Common ratios in published work:</p>

<ul>
  <li><strong>8×-16× expansion</strong>: smaller, faster to train, fewer interpretable features. Used in early SAE work.</li>
  <li><strong>32×-64× expansion</strong>: standard for production SAEs. Gemma Scope mostly uses 16K and 65K dictionaries on Gemma-2-2B's 2304-dim residual stream.</li>
  <li><strong>128×+</strong>: experimental. The intuition: more features, more interpretability, but harder training.</li>
</ul>

<h3>The L1 SAE (the original recipe)</h3>

<p>The standard sparsity penalty is L1 on the feature activations:</p>

<pre><code><span class="kw">def</span> <span class="fn">l1_sae_loss</span>(sae, x, lambda_l1=<span class="num">5e-3</span>):
    features, reconstruction = <span class="fn">sae</span>(x)

    <span class="com"># Reconstruction loss: MSE between input and reconstruction</span>
    recon_loss = (reconstruction - x).<span class="fn">pow</span>(<span class="num">2</span>).<span class="fn">mean</span>()

    <span class="com"># Sparsity penalty: L1 on feature activations</span>
    <span class="com"># Optionally weighted by decoder norms (controls feature scale)</span>
    decoder_norms = sae.decoder.weight.<span class="fn">norm</span>(dim=<span class="num">0</span>)
    sparsity_loss = (features * decoder_norms).<span class="fn">abs</span>().<span class="fn">sum</span>(dim=-<span class="num">1</span>).<span class="fn">mean</span>()

    <span class="kw">return</span> recon_loss + lambda_l1 * sparsity_loss</code></pre>

<p>The L1 penalty pushes feature activations toward zero. The optimizer balances reconstruction (which wants features active) against sparsity (which wants them zero). The result: only features that meaningfully reduce reconstruction loss stay active for any given input.</p>

<p>The <code>lambda_l1</code> coefficient is the central hyperparameter. Too low: SAE produces dense features (every feature active for every input — useless). Too high: SAE produces too few features, reconstruction is bad. Practical range: 1e-4 to 1e-2; tuning is empirical.</p>

<p>Three persistent problems with L1 SAEs:</p>

<ul>
  <li><strong>Dead features</strong>: a feature that never activates after some point in training. Stays at zero forever, contributing nothing. Production runs find 30-50% of features go dead. Wasted capacity.</li>
  <li><strong>Shrinkage</strong>: L1 not only pushes inactive features to zero but also pulls active features below their "true" magnitude. The reconstruction is systematically attenuated.</li>
  <li><strong>Hyperparameter sensitivity</strong>: lambda_l1 needs to be tuned per layer, per model, per dictionary size. Small changes cause large quality differences.</li>
</ul>

<p>These problems motivated the modern variants.</p>

<h3>Top-K SAEs (the modern default)</h3>

<p>OpenAI's "Scaling and evaluating sparse autoencoders" paper (2024) replaced L1 with a hard constraint: <strong>only the top-K largest activations are kept; the rest are zeroed</strong>. No L1 penalty needed.</p>

<pre><code><span class="kw">class</span> <span class="ty">TopKSAE</span>(nn.Module):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, d_model, d_dict, k=<span class="num">32</span>):
        <span class="fn">super</span>().<span class="fn">__init__</span>()
        self.encoder = nn.<span class="fn">Linear</span>(d_model, d_dict, bias=<span class="kw">True</span>)
        self.decoder = nn.<span class="fn">Linear</span>(d_dict, d_model, bias=<span class="kw">True</span>)
        self.k = k

    <span class="kw">def</span> <span class="fn">forward</span>(self, x):
        x_centered = x - self.decoder.bias

        <span class="com"># Encode (no ReLU yet — top-K does the gating)</span>
        pre_activation = self.<span class="fn">encoder</span>(x_centered)

        <span class="com"># Top-K: zero out all but the K largest activations per token</span>
        topk_vals, topk_idx = pre_activation.<span class="fn">topk</span>(self.k, dim=-<span class="num">1</span>)
        features = torch.<span class="fn">zeros_like</span>(pre_activation)
        features.<span class="fn">scatter_</span>(-<span class="num">1</span>, topk_idx, F.<span class="fn">relu</span>(topk_vals))

        reconstruction = self.<span class="fn">decoder</span>(features)
        <span class="kw">return</span> features, reconstruction

<span class="kw">def</span> <span class="fn">topk_sae_loss</span>(sae, x):
    _, reconstruction = <span class="fn">sae</span>(x)
    <span class="com"># Just MSE — sparsity is enforced architecturally</span>
    <span class="kw">return</span> (reconstruction - x).<span class="fn">pow</span>(<span class="num">2</span>).<span class="fn">mean</span>()</code></pre>

<p>The benefits compound:</p>

<ul>
  <li><strong>Sparsity is enforced exactly</strong>: K active features per token, period. No tuning of lambda_l1.</li>
  <li><strong>Dead-feature rate plummets</strong>: typical Top-K SAE has 90%+ of features alive after training, vs 50-70% for L1.</li>
  <li><strong>No shrinkage</strong>: the active features keep their full magnitude.</li>
  <li><strong>Hyperparameter is interpretable</strong>: K is "how many features per token" — directly related to expected sparsity.</li>
</ul>

<p>Typical K values: 32-128 for production SAEs. The paper showed Top-K SAEs strictly dominate L1 SAEs at every dictionary size and reconstruction quality. <em>If you're training an SAE in 2026, start with Top-K.</em></p>

<h3>Other modern variants</h3>

<p>The post-Top-K landscape has several refinements:</p>

<ul>
  <li><strong>Gated SAEs</strong> (DeepMind, 2024): separate gate (decides which features fire) from magnitude (decides how strongly). Reduces shrinkage further than Top-K. Used in some Gemma Scope SAEs.</li>
  <li><strong>JumpReLU SAEs</strong>: replace ReLU with a jump function that has a hard threshold. Combines L0-like sparsity with smooth gradients via straight-through estimators.</li>
  <li><strong>BatchTopK</strong>: instead of per-token top-K, take top-K across the whole batch. Gives some tokens more features, others fewer; better for tokens that are "naturally" feature-dense.</li>
  <li><strong>Matryoshka SAEs</strong> (2024-2025): nested dictionary structure where the first K_outer features must reconstruct alone, then the next layer adds detail. Trains multi-scale features in a single pass.</li>
</ul>

<p>The field is active and converging. As of 2026: Top-K and Gated are mainstream; the others are research alternatives with specific use cases. The differences between modern variants matter at the margin; choosing modern over L1 matters substantially.</p>

<h2>Training SAEs at scale</h2>

<p>SAE training has its own engineering rhythm distinct from main-model training. The basic flow:</p>

<ol>
  <li><strong>Collect activations</strong>: run the target model on a large pile of text (often the same pretraining corpus from M35). At each token, save the activation at the layer you want to analyze. Typical scale: 100M to 10B tokens of activations, stored on disk.</li>
  <li><strong>Train the SAE</strong>: run optimization on the activation data. SAEs are usually trained for 1-3 epochs over the activation buffer; training time is a fraction of main-model training (because SAEs are smaller and the loss is per-token-cheap).</li>
  <li><strong>Validate</strong>: hold out activation data; check reconstruction loss and sparsity match the loss-curve targets. Compute "L0" metric: average number of non-zero features per token (should be the K you specified for Top-K, or whatever lambda_l1 produced).</li>
</ol>

<p>The compute cost is real but manageable. For Gemma-2-2B, training one Top-K SAE on residual-stream activations costs ~$1K of cloud compute. For all 26 layers and multiple positions per layer, multiply. For Sonnet-scale models, multiply again — Anthropic's published Sonnet SAEs likely cost $1M+ to train.</p>

<h3>Key training tricks</h3>

<p>SAE training has several recipe details that matter substantially:</p>

<ul>
  <li><strong>Initialize decoder bias to median activation</strong>: the activations being autoencoded usually have a non-zero mean. Initializing the decoder bias to the median (or mean) lets the SAE focus on modeling deviations rather than reconstructing the mean every time.</li>
  <li><strong>Normalize decoder weights</strong>: after each step, project each decoder column to unit norm. Without this, the SAE can scale features arbitrarily (down-scale features, up-scale decoder columns; reconstruction is the same but L1 is artificially low). Forcing unit decoder norm makes the L1 penalty meaningful.</li>
  <li><strong>fp32 training</strong>: SAEs have small effective gradients. Training in bf16 often produces noticeably worse SAEs. Production runs use fp32 throughout.</li>
  <li><strong>Resampling dead features</strong>: periodically check which features haven't fired recently; reinitialize them with a new direction. Cuts dead-feature rate substantially in L1 SAEs (less needed for Top-K).</li>
  <li><strong>Auxiliary losses</strong>: some recipes add an "auxiliary k-loss" that uses a higher K to maintain reconstruction stability while the primary loss pushes for the lower-K target.</li>
</ul>

<h2>Feature interpretation</h2>

<p>You've trained an SAE. You have 65,536 features. What do they mean?</p>

<p>The standard pipeline is <strong>auto-interpretation</strong>, formalized by Bills et al. (OpenAI, 2023):</p>

<ol>
  <li><strong>Find max-activating examples</strong>: for each feature, run the model on a corpus and record the top-K text segments where that feature fires most strongly. Typical: top 20 contexts of 50-100 tokens each.</li>
  <li><strong>Generate explanation</strong>: send the max-activating examples to a strong LLM with a prompt like "What pattern do these texts share?" The LLM produces a candidate explanation: "Mentions of bridges, especially the Golden Gate."</li>
  <li><strong>Score the explanation</strong>: take a held-out set of contexts. Use another LLM to score "given this explanation, would feature X fire here?" Compare predictions to actual feature activations. The score is the explanation quality.</li>
</ol>

<p>This pipeline is fully automated. Anthropic's "Scaling Monosemanticity" paper used variants of this on millions of features at Sonnet scale. The output is a labeled dictionary: feature 14722 → "Golden Gate Bridge"; feature 5829 → "code function definitions"; feature 31204 → "expressing emotional vulnerability."</p>

<p>Quality varies. Some features are crisp and the explanation matches every max-activating example. Some features are murky — they fire on a mix of related concepts that don't have a clean unifying description. The published metrics: ~30-50% of features get high-quality auto-interpretations; the rest are partially interpreted or remain mysterious.</p>

<h3>What does a feature actually look like?</h3>

<p>A concrete example, drawn from published Anthropic work. They identified a "Golden Gate Bridge" feature in their Sonnet SAE — feature 34M something. The signature:</p>

<ul>
  <li><strong>Max-activating contexts</strong>: text mentioning the Golden Gate Bridge (English, French, Spanish, Chinese), images of the bridge, references to Marin County and San Francisco landmarks.</li>
  <li><strong>Steering effect</strong>: artificially boosting this feature's activation during generation makes the model bring up the Golden Gate Bridge in unrelated contexts. Anthropic's "Golden Gate Claude" demo had the model respond to "What's your favorite color?" with stories about driving across the bridge.</li>
  <li><strong>Ablation effect</strong>: zeroing this feature during the model's normal processing has subtle effects — the model still understands the concept but is less likely to mention it spontaneously.</li>
</ul>

<p>The feature is multimodal (fires on text and images of the same concept), multilingual (fires regardless of language), and abstract enough to capture "the concept of the Golden Gate Bridge" rather than just the literal string. <em>This is what successful feature discovery looks like.</em></p>

<h2>Feature circuits</h2>

<p>Individual features are useful. The bigger question: how do features compose? Why does feature B in layer 23 activate? Probably because some combination of earlier features in earlier layers caused it. <strong>Feature circuits</strong> are the connections.</p>

<p>The standard technique is <strong>path patching</strong> (or <strong>attribution patching</strong> as a faster approximation): for each earlier feature, ablate it and see how much the later feature's activation changes. The earlier features whose ablation has large effects are "in the circuit" feeding the later feature.</p>

<pre><code><span class="kw">def</span> <span class="fn">attribution_patching</span>(model, sae_early, sae_late, prompt):
    <span class="com"># Run with hooks to save activations and gradients of late feature.</span>
    early_features, late_features = <span class="fn">collect_features</span>(model, sae_early, sae_late, prompt)

    <span class="com"># For each late feature, compute gradient w.r.t. each early feature.</span>
    <span class="com"># An early-feature contribution = early_activation × ∂(late_activation)/∂(early_activation)</span>
    <span class="com"># Attribution patch: linearized estimate of effect of zeroing each early feature.</span>
    contributions = early_features * grads_of_late_w.r.t.early
    <span class="com"># Top contributors = circuit feeding into the late feature</span>
    <span class="kw">return</span> contributions.<span class="fn">topk</span>(<span class="num">10</span>)</code></pre>

<p>Attribution patching is an approximation — actual path patching ablates each feature and re-runs the model, which is more accurate but vastly more expensive. The approximation is usually sufficient; production circuit discovery uses attribution.</p>

<p>The output is a directed graph: features at layer N point to features at layer N+1 (and beyond) that they cause. A complete circuit might span 5-15 layers, with 20-100 features participating. The narrative: "input contains arithmetic → arithmetic-detection feature in layer 5 → digit-tracking features in layer 8 → addition-result features in layer 14 → output token logits."</p>

<p>Finding circuits at scale is still an active research area. The published work has discovered specific circuits for tasks like indirect object identification, modular arithmetic, factual recall. Discovering circuits automatically across a model is largely future work.</p>

<h2>Production scale: Gemma Scope and Sonnet features</h2>

<p>Two public reference points for production-scale SAE work:</p>

<ul>
  <li><strong>Gemma Scope</strong> (DeepMind, 2024): trained SAEs on every layer of Gemma-2-2B, Gemma-2-9B, and Gemma-2-27B. Multiple SAE configurations per layer (different K, different expansion factors). Released openly. This is the largest public SAE release; a research baseline.</li>
  <li><strong>Sonnet SAEs</strong> (Anthropic, 2024-2025): SAEs at Claude Sonnet scale. Not openly released, but published results showed millions of features discovered. Used internally for safety evaluation.</li>
</ul>

<p>The compute scaling: SAE compute is roughly proportional to (model dimension × dictionary expansion × tokens trained). For a 27B model with 4608-dim residual stream, 64× expansion = 295K dictionary, training on 1B tokens of activations: ~50 H100-days per SAE. Across 60 layers and multiple sites per layer, that's 1000s of H100-days for a complete SAE coverage of one model.</p>

<p>Despite the cost, the trend is toward more SAE coverage at every model release. Frontier labs increasingly view interpretability infrastructure as a complement to capability training — you ship a model alongside its SAE-derived feature dictionary.</p>

<h2>What interpretability is actually used for</h2>

<p>The cynical question: with all this engineering, what do the features actually do? The honest 2026 answer:</p>

<ul>
  <li><strong>Bias and safety detection</strong>: find features that fire on demographic categories, hate speech, deceptive reasoning. Use these as classifiers for safety filtering. Anthropic uses interpretability-derived features in their safety stack.</li>
  <li><strong>Capability evaluation</strong>: detect whether a model has learned a specific capability (e.g., "knows how to synthesize biological weapons") even if it never expresses that capability in normal usage. Active research, used in dangerous-capability evaluations.</li>
  <li><strong>Model editing</strong>: artificially boost or suppress specific features at inference time to change behavior without retraining. Steering vectors are a related technique. Production uses include "make the model less likely to refuse benign requests" and "increase the model's tendency to cite sources."</li>
  <li><strong>Jailbreak analysis</strong>: when a model is jailbroken, what changes internally? Compare features that fire under normal prompts vs jailbroken prompts. Helps understand the attack surface and design defenses.</li>
  <li><strong>Distinguishing knowledge from expression</strong>: a model can "know" something internally without expressing it (e.g., the model has accurate internal representations of geography but produces fluent-but-wrong outputs). Features can detect this gap, useful for hallucination analysis.</li>
  <li><strong>Debugging training failures</strong>: when post-training degrades a capability, examine which features changed. Often reveals the training signal misalignment (e.g., RLHF caused a "factual confidence" feature to be suppressed).</li>
</ul>

<p>The first use case (safety detection) is now production. The others are mature-research-with-some-production-use. The trajectory is clear: <em>interpretability is becoming a standard layer in the ML stack</em>, alongside training, evaluation, and serving.</p>

<h2>Limitations and open problems</h2>

<p>SAEs find <em>many useful features</em>, but probably <em>not all features</em>. Specific limitations to track:</p>

<ul>
  <li><strong>Attention features are harder than residual-stream features</strong>. SAEs work cleanly on the residual stream; attention head outputs and weights are less SAE-amenable. Active research.</li>
  <li><strong>Some features are inherently distributed</strong>. Concepts that don't have a clean local representation (long-range dependencies, complex compositions) may not appear as single SAE features even at large dictionary sizes.</li>
  <li><strong>Feature universality is partial</strong>. Two SAEs trained with different random seeds find some of the same features and some different ones. The "real" features the model uses may not be discoverable up to permutation.</li>
  <li><strong>Reconstruction quality has a ceiling</strong>. Even at very large dictionary sizes, SAEs reconstruct only ~85-95% of the original activation. The remainder is genuinely high-dimensional structure that resists sparse decomposition.</li>
  <li><strong>The "model thought it" interpretation is approximate</strong>. SAE features are correlations between the model's activations and human-interpretable concepts. The model's actual computation may use a slightly different decomposition that the SAE only approximates.</li>
</ul>

<p>The honest summary: <strong>SAEs are the best tool we have for mechanistic interpretability of LLMs in 2026, and they are sufficient to be useful in production for specific tasks</strong>. They're not the final word, and the field is rapidly producing better techniques. Interpretability research and engineering will likely look quite different in 5 years.</p>

<div class="ndq">
<h4>About interpretability and SAEs</h4>

<p class="q">Why train SAEs on activations rather than weights?</p>
<p class="a">Activations are where the model's "thinking" happens — what it's currently representing for the input it's processing. Weights are static and represent the model's learned knowledge but not its current cognitive state. A feature like "the model is currently thinking about the Golden Gate Bridge" only makes sense as an activation pattern; it's not localized in any specific weight. There are weight-level interpretability techniques (analyzing attention head matrices, MLP weights), but they're typically less successful than activation-based methods for LLMs at scale.</p>

<p class="q">Why doesn't every neuron just learn to represent one feature, eliminating polysemanticity?</p>
<p class="a">Two reasons. (1) <strong>Capacity constraint</strong>: with D neurons and far more than D features the model needs, it can't have one neuron per feature. Superposition is forced by capacity scarcity. (2) <strong>Optimization pressure</strong>: even if the model had enough capacity in principle, gradient descent doesn't have a strong incentive to allocate one feature per neuron. The losses don't penalize polysemanticity. The model finds <em>some</em> functional decomposition that minimizes loss; that decomposition isn't pretty. SAEs are useful precisely because they impose the human-friendly decomposition the model didn't naturally produce.</p>

<p class="q">If features can be steered, doesn't that mean the SAE is finding what the model uses, not just what's correlated?</p>
<p class="a">Steering effects provide stronger evidence than correlation alone, but it's not airtight. Possible interpretations of "boosting feature X causes the model to talk about Y": (a) the model uses feature X to represent Y, and steering boosts the representation; (b) the SAE's feature X is correlated with the model's true representation of Y, and steering happens to push activations in a direction that overlaps the true representation; (c) some other interaction. Interpreting steering effects requires care. <em>That said</em>: when steering works reliably across many contexts, the simplest explanation is usually that the SAE found something close to what the model is actually using.</p>

<p class="q">Can I use SAE features as inputs to a probe classifier instead of training a classifier from scratch?</p>
<p class="a">Yes, and this is one of the most practical uses of SAEs. Once you have a feature dictionary, you can build classifiers as simple linear combinations of features. Want a "deception detector"? Find features related to deception, weight them, threshold. Often performs comparably to training a probe from scratch on the original activations, but with the huge advantage of being interpretable: you can see <em>why</em> the classifier fires (which features lit up). Production safety stacks at frontier labs use this pattern extensively.</p>

<p class="q">Why is fp32 needed for SAE training when bf16 is fine for the underlying model?</p>
<p class="a">SAE training has unusual gradient characteristics: most features are inactive most of the time (sparsity), so most parameter gradients are zero or near-zero per step. The signal-to-noise ratio in any given gradient update is much lower than in standard training. bf16's reduced precision adds noise that can swamp the small effective signal, leading to noticeably worse SAEs. fp32 preserves the precision needed for the small gradients to accumulate correctly. The compute cost is acceptable because SAEs are small relative to the main model. <em>If your SAE training looks unstable or has high dead-feature rates, switching from bf16 to fp32 often fixes it.</em></p>

<p class="q">What's the relationship between SAEs and LoRA / model editing more broadly?</p>
<p class="a">They're complementary techniques operating at different scales. LoRA (M11-adjacent) modifies weight matrices to change model behavior across many inputs; SAE-based steering modifies activations to change behavior at specific moments. LoRA is permanent (until you discard the adapter); steering is per-inference. For some applications (safety, persona shifts, reasoning style), SAE-based steering is more flexible because it's controllable at runtime. For others (specialized capabilities, format following), LoRA is more efficient because the change is built in. <em>Production deployments increasingly combine both</em>: LoRA for capability shaping, SAE-features for runtime control.</p>
</div>

<h2>Code Magnets: implement a Top-K SAE forward pass</h2>

<p>You're writing the forward pass for a Top-K SAE. Three magnets are wrong choices.</p>

<div class="magnets">
<p>Arrange the magnets to compute features and reconstruction.</p>

<div class="magnet-pool">
  <span class="magnet">def forward(self, x):</span>
  <span class="magnet">    x_centered = x - self.decoder.bias</span>
  <span class="magnet">    x_centered = x</span>
  <span class="magnet">    pre_activation = self.encoder(x_centered)</span>
  <span class="magnet">    topk_vals, topk_idx = pre_activation.topk(self.k, dim=-1)</span>
  <span class="magnet">    topk_vals, topk_idx = pre_activation.abs().topk(self.k, dim=-1)</span>
  <span class="magnet">    features = torch.zeros_like(pre_activation)</span>
  <span class="magnet">    features.scatter_(-1, topk_idx, F.relu(topk_vals))</span>
  <span class="magnet">    features = F.relu(pre_activation)</span>
  <span class="magnet">    reconstruction = self.decoder(features)</span>
  <span class="magnet">    return features, reconstruction</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">def</span> <span class="fn">forward</span>(self, x):
    x_centered = x - self.decoder.bias
    pre_activation = self.<span class="fn">encoder</span>(x_centered)
    topk_vals, topk_idx = pre_activation.<span class="fn">topk</span>(self.k, dim=-<span class="num">1</span>)
    features = torch.<span class="fn">zeros_like</span>(pre_activation)
    features.<span class="fn">scatter_</span>(-<span class="num">1</span>, topk_idx, F.<span class="fn">relu</span>(topk_vals))
    reconstruction = self.<span class="fn">decoder</span>(features)
    <span class="kw">return</span> features, reconstruction</code></pre>
<p>The traps:</p>
<ul>
  <li><code>x_centered = x</code> (no bias subtraction): skips the centering step. The decoder bias represents the mean of the activation distribution; without subtracting it before encoding, the SAE has to spend dictionary capacity reconstructing the mean for every input. Reconstruction quality degrades and many features end up encoding the offset rather than meaningful structure. Always subtract the decoder bias before encoding.</li>
  <li><code>topk_vals, topk_idx = pre_activation.abs().topk(self.k, dim=-1)</code>: takes top-K by absolute value. Wrong direction. We want the K most-positive activations (which will then pass through ReLU). Taking by absolute value would include large negative pre-activations, but those get zeroed by ReLU anyway — and worse, we'd be selecting them at the expense of smaller-but-positive activations that should be the actual features. Standard top-K on the raw pre-activation is correct.</li>
  <li><code>features = F.relu(pre_activation)</code>: replaces the top-K mechanism entirely with a plain ReLU. This is the L1 SAE form, not Top-K. Without the top-K masking, sparsity isn't enforced architecturally — every positive pre-activation passes through. You'd then need an L1 penalty (which is what the L1 SAE does), but without one, the SAE produces dense features. The whole point of Top-K is the hard sparsity constraint.</li>
</ul>
<p>The pattern: <strong>subtract decoder bias → encode → top-K by raw value (not absolute) → scatter into zero tensor with ReLU on values → decode</strong>. Each step has a specific role: centering, projecting up, hard sparsity constraint, applying ReLU only to the top-K values, projecting back.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each interpretability concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>Polysemanticity</div>
  <div>A. A single neuron encodes multiple unrelated concepts; the rule, not the exception.</div>

  <div>Superposition hypothesis</div>
  <div>B. Models represent more features than dimensions by overlapping in non-orthogonal directions.</div>

  <div>Sparse autoencoder (SAE)</div>
  <div>C. Wide encoder + sparse hidden + narrow decoder; dictionary atoms become interpretable features.</div>

  <div>Top-K sparsity</div>
  <div>D. Architectural sparsity: only the K largest activations per token survive; dramatically reduces dead features.</div>

  <div>Auto-interpretation</div>
  <div>E. LLM-driven pipeline: find max-activating examples, summarize the pattern, score the explanation.</div>

  <div>Path patching / attribution</div>
  <div>F. Discover feature circuits by ablating earlier features and measuring effect on later ones.</div>

  <div>Steering</div>
  <div>G. Artificially boost or suppress a specific feature at inference time to change model behavior.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>Polysemanticity</strong> → A<br>
<strong>Superposition</strong> → B<br>
<strong>SAE</strong> → C<br>
<strong>Top-K sparsity</strong> → D<br>
<strong>Auto-interpretation</strong> → E<br>
<strong>Path patching / attribution</strong> → F<br>
<strong>Steering</strong> → G
</p>
<p>The mental shortcut: <em>neurons are polysemantic because of superposition, SAEs disentangle them, Top-K enforces sparsity, auto-interp labels them, attribution finds circuits, steering manipulates them</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team trains an L1 SAE on Llama-3-8B's layer-15 residual stream with 64× expansion (262K features). After training, they find that 60% of features are dead (never fire). Reconstruction quality is mediocre. What would you change first?</p>
<details class="answer"><summary>show answer</summary>
<p>Switch to Top-K. The L1 SAE's combination of "L1 sparsity penalty" plus "easy-to-zero ReLU" produces high dead-feature rates as a structural property — features that get pushed below the activation threshold early in training tend to stay there. Top-K replaces the L1 penalty with a hard architectural constraint: exactly K features active per token. Dead features can't form because every token must have K active features.</p>
<p>Empirically, switching from L1 to Top-K typically reduces dead-feature rates from 30-60% to 5-10% with the same dictionary size, no L1 tuning required. Reconstruction quality usually improves because the surviving features have full magnitude (no L1 shrinkage).</p>
<p>Other things to try if Top-K alone doesn't fix it: (a) <strong>resampling</strong> dead features periodically (reinitialize them with new directions); (b) <strong>auxiliary k-loss</strong> with higher K to maintain stability; (c) <strong>fp32 training</strong> if currently in bf16; (d) <strong>longer training</strong> if the SAE was trained for too few steps. But Top-K is the highest-leverage single change.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Why might steering a "Golden Gate Bridge" feature successfully cause the model to mention the bridge in unrelated contexts, while steering a more abstract feature (e.g., "deception") might fail or produce strange behavior?</p>
<details class="answer"><summary>show answer</summary>
<p>Several reasons that compound:</p>
<p>(1) <strong>Concrete concepts have crisper representations</strong>. "Golden Gate Bridge" is a specific named entity with consistent properties across contexts. The model likely has a relatively localized representation of it. Boosting that representation reliably causes the concept to be expressed.</p>
<p>(2) <strong>Abstract concepts are likely distributed</strong>. "Deception" doesn't have a single object referent; it's a property that depends on the relationship between many concepts (intent, falsehood, audience awareness). The model's representation may be spread across many features, and an SAE-discovered "deception feature" may capture only one component.</p>
<p>(3) <strong>Causal vs correlational features</strong>. The SAE finds features correlated with the model's behavior, but the causal structure may be more complex. A feature that lights up during deceptive outputs may not be the cause of those outputs — it might be a downstream consequence. Steering it would then be like adjusting the thermometer rather than the thermostat.</p>
<p>(4) <strong>Off-distribution effects</strong>. Steering is artificial — boosting a feature beyond its natural range pushes the model into off-distribution territory. For "Golden Gate Bridge," the feature is grounded enough that off-distribution boosting still produces sensible outputs. For "deception," boosting can produce incoherent or self-defeating behavior because the model is being pushed into states it's never naturally been in.</p>
<p>The general lesson: <em>SAE features are most useful for concrete, localized concepts</em>. They become less reliable for abstract or distributed concepts. This is an active research limitation — finding faithful representations of abstract concepts is harder than finding object-level features.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> A safety team wants to use SAE features to detect when the model is "considering producing harmful content" before it actually outputs anything. Walk through how you'd build this.</p>
<details class="answer"><summary>show answer</summary>
<p>Step-by-step approach:</p>
<p>(1) <strong>Train (or obtain) an SAE</strong> on a layer of the model where high-level reasoning is happening — typically a mid-to-late layer's residual stream. Public Gemma Scope SAEs work for Gemma; for proprietary models, train your own.</p>
<p>(2) <strong>Find candidate features</strong>: collect a curated set of (prompt, harmful-trajectory-response) and (prompt, safe-trajectory-response) pairs. For each prompt, run the model; record SAE feature activations at the relevant layer for both trajectories. Find features that consistently differ between the two: "fires more during harmful trajectories than safe ones."</p>
<p>(3) <strong>Validate by auto-interpretation</strong>: for each candidate feature, run the auto-interp pipeline (max-activating examples, LLM-generated explanation). Confirm features have plausible interpretations (e.g., "discussion of weapon manufacturing," "instructions for evading security").</p>
<p>(4) <strong>Build a classifier</strong>: linear combination of safety-relevant features, trained as a binary classifier on the curated pairs. Linearity matters for interpretability — you want to know which features triggered the alert.</p>
<p>(5) <strong>Deploy in serving</strong>: at inference time, after the model has processed the prompt but before/during generation, extract activations at the chosen layer, run the SAE encoder to get features, run the classifier. If it fires above threshold, route the response through additional checks (or refuse).</p>
<p><strong>Caveats</strong>: this is a complement to behavioral safety, not a replacement. SAE-derived classifiers have false positives (the model considers but rejects harmful content; the classifier flags this) and false negatives (the model uses representations not captured by the SAE). Calibrate thresholds carefully; combine with output-level filtering. Anthropic and others use variants of this in production safety stacks.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> Two SAEs are trained on the same layer of the same model with different random seeds. They find some of the same features and some different ones. Why isn't there a single canonical "true feature set"?</p>
<details class="answer"><summary>show answer</summary>
<p>Several reasons that compound:</p>
<p>(1) <strong>Optimization landscape has many local optima</strong>. The SAE training problem (minimize reconstruction subject to sparsity) is non-convex. Different initializations converge to different solutions, all of which achieve similar loss but partition activation space differently.</p>
<p>(2) <strong>Some features may be genuinely arbitrary linear combinations</strong>. If the model's representation is "the sum of feature A and feature B always appears together," an SAE might decompose this into (A, B) or into (A+B, A-B) or any other rotation. All are valid solutions to the reconstruction problem; the model's "true" decomposition may be undefined.</p>
<p>(3) <strong>Dictionary size is finite</strong>. With limited capacity, the SAE has to make tradeoffs about which features to allocate capacity to. Different runs make different tradeoffs based on training dynamics.</p>
<p>(4) <strong>Feature universality is partial empirically</strong>. The "feature universality" hypothesis (different SAEs find the same features) is supported for many concrete features (entity-like) but weaker for abstract features. Some features genuinely vary across SAE runs.</p>
<p>(5) <strong>The model's "true" features may not be a fixed set</strong>. The model's computation may not have a unique canonical decomposition. SAEs find <em>useful</em> approximations; the existence of a "ground truth" set of features is an assumption, not a proven property of trained models.</p>
<p>The practical implication: when claiming a feature exists, it's safer to verify across multiple SAE training runs (or compare published SAEs from different labs). High-confidence features replicate; speculation about features that only appear in one SAE run should be tagged as such.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li><strong>Mechanistic interpretability</strong> studies the internal computation of models, not just inputs and outputs. By 2026, it's a production tool for safety, evaluation, and debugging.</li>
  <li><strong>Polysemanticity</strong>: a single neuron encodes multiple unrelated concepts. The rule, not the exception, in language models.</li>
  <li><strong>Superposition hypothesis</strong>: models represent more features than they have dimensions by overlapping them in non-orthogonal directions. With sparsity (most features zero per input), this works.</li>
  <li><strong>Sparse autoencoders (SAEs)</strong> disentangle superposition. Encoder-decoder pair with sparsity constraint; the dictionary atoms become interpretable features.</li>
  <li><strong>L1 SAE</strong> (the original): MSE reconstruction + L1 penalty on feature activations. Has dead-feature, shrinkage, and hyperparameter problems.</li>
  <li><strong>Top-K SAE</strong> (modern default): hard sparsity — only K features active per token. Much lower dead-feature rate, no L1 tuning. <em>Use Top-K in 2026 unless you have a specific reason not to</em>.</li>
  <li><strong>Other variants</strong>: Gated SAEs (separate gate and magnitude, less shrinkage), JumpReLU, BatchTopK, Matryoshka SAEs.</li>
  <li><strong>Training tricks</strong>: initialize decoder bias to median activation; normalize decoder weights to unit norm; use fp32; resample dead features periodically.</li>
  <li><strong>Auto-interpretation pipeline</strong>: find max-activating examples for each feature; have an LLM summarize the pattern; have another LLM score the explanation against held-out activations. ~30-50% of features get high-quality interpretations.</li>
  <li><strong>Feature circuits</strong>: how features compose. Path patching (or attribution patching as approximation) finds which earlier features feed into a later one.</li>
  <li><strong>Production scale</strong>: Gemma Scope (DeepMind, public, all layers of Gemma-2 family); Anthropic Sonnet SAEs (internal). Compute cost approaches a fraction of main-model training.</li>
  <li><strong>Real uses</strong>: safety detection, capability evaluation, model editing via steering, jailbreak analysis, distinguishing knowledge from expression, debugging training failures.</li>
  <li><strong>Limitations</strong>: SAEs find many useful features but probably not all features. Attention features harder than residual-stream. Some concepts are inherently distributed. Universality is partial.</li>
  <li>The reflex: when you need to understand "what is the model doing here?" — train an SAE, find the features, validate via steering, build classifiers from features. The interpretability stack is engineering now, not just research.</li>
</ul>
</div>

<p>Module 37 (if Part X continues) would tackle <strong>production eval engineering</strong> — non-saturating benchmarks, contamination resistance, statistical significance for LLM evals, LLM-as-judge bias, calibration. Most teams' biggest weakness; under-rigorous in most setups.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">36</span>
  <span>Mechanistic interpretability</span>
</div>
"""

emit("36_mech_interp", "Module 36 — Mechanistic interpretability", BODY)
