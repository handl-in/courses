# Module 44 — Comparing latest LLM & VLM architectures

# _Comparing latest LLM & VLM architectures:_ the April 2026 zoo, taxonomized

_Bonus · Module 44 · April 2026 currency_

— from DeepSeek V4's hybrid attention to Llama 4's iRoPE, why every flagship is now MoE, the three-era VLM taxonomy, and the architectural primitives that distinguish 2026 models from each other

\--- 

You've spent 43 modules learning how to build models. This bonus module steps back to compare what's actually deployed in April 2026. Same pedagogy, different scope: instead of explaining one architecture, we taxonomize the family.

The 2026 model landscape has more entrants than any prior year. Closed-source: GPT-5.4 / 5.5, Claude Opus 4.6 / 4.7 / Sonnet 4.6, Gemini 3 / 3.1 Pro, Grok 4. Open-weight: Llama 4 Scout / Maverick, DeepSeek V4-Pro / V4-Flash, GLM-5 / 5.1, Qwen 3.5 / 3.6, Kimi K2.5 / K2.6, MiniMax M2.5, MiMo V2.5, Gemma 3, Nemotron Ultra. _Every frontier model is now MoE_. _Every flagship has 1M+ context window_. _The 25× price gap_ between cheapest and most expensive frontier model is the defining 2026 economics.

This module taxonomizes them. The architectural primitives that matter: attention variants (full, GQA, sparse, hybrid), position encoding (RoPE, iRoPE, YaRN), MoE patterns (pure, alternating, shared experts), reasoning modes (separate vs unified switchable), and the VLM design eras. By the end you'll know _what makes DeepSeek V4 different from Llama 4 different from GLM-5 different from Qwen 3.6_ , why all the closed-source flagships keep landing within 2.5 points of each other on benchmarks, and which architectural choices are converging vs diverging.

> **★ KEY IDEA**  
>  The April 2026 LLM zoo has converged on three macro-patterns. **(1) Mixture-of-Experts is universal** at frontier scale — DeepSeek V4-Pro (1.6T/49B active), Llama 4 Maverick (400B/17B), GLM-5 (744B/40B), Qwen 3.6 (35B/3B), Kimi K2.6, MiniMax M2.5, MiMo V2.5 (1.02T/42B). Every flagship is sparse. **(2) 1M+ token context is table stakes** — Llama 4 Scout extends to 10M via iRoPE; DeepSeek V4 hits 1M via Hybrid Attention (CSA + HCA, 27% FLOPs / 10% KV vs V3.2); GPT-5.5 has 256K, Claude Opus 4.7 has 200K with 1M beta, Gemini 3.1 Pro has 1M. **(3) Reasoning consolidates into the base model** — DeepSeek V4 folds the R reasoning line into a single model with switchable Thinking / Non-Thinking modes; closed-source flagships do the same. The architectural primitives that differentiate: **Attention variants** — full vs GQA vs sparse-content-based (DSA) vs hybrid CSA+HCA (DeepSeek V4) vs sliding-window. **Position encoding** — RoPE vs iRoPE (Llama 4: 3 RoPE layers + 1 NoPE layer interleaved, enables 10M context) vs YaRN scaling. **MoE patterns** — pure MoE (Scout, GLM-5, DeepSeek) vs MoE/dense alternating (Maverick) vs shared experts (DeepSeek lineage) vs different routing (top-k, expert choice). **Optimizer** — AdamW remains dominant; **Muon** (DeepSeek V4) is the first frontier replacement. **Precision** — NVFP4 training (M41) maturing; FP8 still dominant for closed-source. For VLMs, the survey-level taxonomy (Vision-Language-Models-Overview, Feb 2026): **three eras**. **Era 1** — frozen vision encoder + frozen LLM + learnable connector (CLIP, BLIP, Flamingo, 2021-2023). **Era 2** — LLM trunk with vision as bolt-on adapter (LLaVA, Qwen2.5-VL, GPT-4V, 2023-2025). **Era 3** — single transformer trained from scratch on mixed-modality data; forks into **3a (early fusion, text-out)** used by Qwen3.5/3.6, Gemma 4, Gemini 3, GPT-5.4, Claude Opus 4.6, and **3b (mixed-modality output)** for models generating images/audio. **Vision encoders** : SigLIP 2 dominant (Qwen3-VL, Gemma 3); InternViT-6B for InternVL line; DINOv2+SigLIP fused (M43's OpenVLA). **The benchmarks have saturated** — MMMU-Pro spread between top four flagships is 2.4 points by April 2026 (within run-to-run noise). Differentiation moved to video (Gemini 3 leads), audio (Gemini 3), long-document OCR (Claude Opus 4.7), chart reasoning (GPT-5.5), code-with-vision (GPT-5.5). 

## Two new faces — taking inventory

T

Taxonomist

"I categorize the zoo. Same family tree, divergent traits — what distinguishes one species from another?"

Every frontier 2026 model is a transformer with attention, position encoding, normalization, MoE, training recipe. _The differences between them are which variants of each primitive they use._ Llama 4 Scout differs from DeepSeek V4-Pro because of: pure-MoE 16-expert vs alternating MoE+dense 256-expert; iRoPE positional with NoPE layers vs standard RoPE with hybrid-attention compression; 10M context via length generalization vs 1M context via FLOP reduction. Both work; both are excellent; the choices reflect different deployment priorities. **Reading 2026 model cards is now an exercise in primitive identification** : spot the attention variant, the MoE pattern, the position encoding, the optimizer, the precision. The base transformer is invariant; the variants tell the story.

C

Convergence

"Frontier models converge faster than press releases suggest. The architectural moves are public, then everyone copies."

DeepSeek R1 (Jan 2025) showed RL with verifiable rewards works → every lab adopted GRPO-family methods within 6 months. Llama 4 (April 2025) showed MoE at frontier scale → DeepSeek, GLM, Qwen, Kimi, MiniMax all shipped MoE flagships by end of 2025. NVIDIA's NVFP4 paper (Sep 2025) → MLPerf v5.1 sweep (Nov 2025) → DeepSeek V4 trained with FP4 quantization-aware (April 2026). **I'm the gravity that pulls architectures toward each other**. Public papers + open weights + ~6 months = convergence. The gap between "leading-edge novel architecture" and "everyone has it" is one model generation. _The 2.4-point spread between top flagships on MMMU-Pro is structural, not coincidental_ — they're approximately the same architecture trained on approximately the same data.

## The frontier zoo, April 2026

Inventory first. The current generation of LLMs in production:

Frontier LLMs in production, April 2026 Model| Source| Architecture| Context| Released  
---|---|---|---|---  
**GPT-5.5** (Codex)| OpenAI| Closed; reasoning unified| 256K (1M in Codex)| March 2026  
**Claude Opus 4.7**|  Anthropic| Closed; Sonnet 4.6 sibling| 200K (1M beta)| April 2026  
**Gemini 3.1 Pro**|  Google DeepMind| Closed; multimodal-native| 1M| February 2026  
**Grok 4**|  xAI| Closed; multi-agent variant| ~256K| Late 2025  
**DeepSeek V4-Pro**|  DeepSeek| Open MoE: 1.6T / 49B active| 1M| April 24, 2026  
**DeepSeek V4-Flash**|  DeepSeek| Open MoE: 284B / 13B active| 1M| April 24, 2026  
**Llama 4 Scout**|  Meta| Open MoE: 109B / 17B active, 16 experts| 10M| April 2025  
**Llama 4 Maverick**|  Meta| Open MoE: 400B / 17B active, 128 experts (alternating)| 1M| April 2025  
**Llama 4 Behemoth** (preview)| Meta| Open MoE teacher; ~2T total| —| Preview 2025  
**GLM-5.1**|  Z.ai (Zhipu)| Open MoE: 744B / 40B active, MIT licensed, DSA| ~200K| Q1 2026  
**Qwen 3.6** (35B-A3B)| Alibaba| Open MoE: 35B / 3B active, Apache 2.0| ~128K| Q1 2026  
**Kimi K2.6**|  Moonshot AI| Open MoE; agentic-focused| ~128K| Q1 2026  
**MiniMax M2.5**|  MiniMax| Open-weight MoE| ~256K| Feb 2026  
**MiMo V2.5-Pro**|  Xiaomi| Open MoE: 1.02T / 42B active, FP8 native| 32K native| Q1 2026  
**Nemotron Ultra 253B**|  NVIDIA| Open dense reasoning model| ~128K| 2025  
**Gemma 3 (27B)**|  Google| Open dense; multilingual| ~128K| 2025  
  
Three observations from the inventory alone, before any architectural detail:

  * **Every frontier model is MoE except small "edge" models (Gemma 3, Nemotron 253B dense)**. The dense-vs-MoE debate from 2023-2024 is over; MoE won at scale.
  * **1M+ context is table stakes for new releases**. Models without it (Qwen 3.6 at 128K, Kimi K2.6 at 128K) are explicitly trading context for cost or speed; new flagship releases extend further.
  * **Open-weight has caught closed-source on coding benchmarks**. GLM-5.1 leads SWE-bench Pro at 58.4%; DeepSeek V4-Pro matches Claude Opus 4.6 at 80.6% on SWE-bench Verified. The "two years behind" gap of 2024 is gone for coding; partial gaps remain for multimodal maturity and safety.

## The architectural primitives that distinguish 2026 models

Every transformer LLM in 2026 has the same skeleton from M11/M14: token embedding → stacked transformer blocks (attention + FFN with normalization) → output projection. The blocks have specific variant choices:

Five primitives where 2026 LLMs differ — and what each lab chose ① Attention mechanism Full attention: vanilla; quadratic in seq length GQA / MQA: share K/V across query heads — Llama, GPT Sparse (DSA): content-based block selection — GLM-5 CSA + HCA hybrid: DeepSeek V4 — 4× compress + 128× compress Sliding window: local attention only — Mistral lineage 2026 trend: hybrid sparse for >1M context ② Position encoding RoPE: rotary embeddings — most flagships' default YaRN: RoPE scaling for long-context fine-tuning iRoPE: RoPE + NoPE interleaved (Llama 4: 3:1 ratio) NoPE: no positional encoding; needs causal mask \+ T-scaling: inference-time temperature for length gen. 2026 trend: hybrid (RoPE + NoPE) for ultra-long ③ MoE pattern Pure MoE: every FFN replaced with experts (Scout, GLM-5) Alternating: MoE + dense alternate (Maverick: 50/50) Shared experts: always-active expert + routed (DeepSeek lineage) Top-k routing: k=1 to k=8 typical; tradeoff load vs sparsity Total/active: 35:3 (Qwen) to 32:1 (DeepSeek) ratios 2026 trend: 30-40× sparsity ratio dominant ④ Reasoning architecture Separate reasoning model: DeepSeek R1, o1 (older) Unified switchable: DeepSeek V4 (3 modes), Claude 4.7 ext. Always-on thinking: Gemini 3 Deep Think variant Multi-agent: Grok 4 (4 parallel agents) RL-tuned reasoning: M39 territory; universal at frontier 2026 trend: unified switchable wins ⑤ Optimizer × Precision (training-time choices) Optimizer: AdamW (almost all) → Muon (DeepSeek V4 first frontier adoption; AdamW retained for embeddings/head) Precision: FP8 (M14): closed-source default; NVFP4 (M41): MLPerf v5.1, DeepSeek V4 expert weights, frontier 2026 Residual: Standard residual → mHC (Manifold-Constrained Hyper-Connections, DeepSeek V4) for trillion-scale stability 2026 trend: AdamW + FP8/NVFP4 + standard residual is mainstream; novel choices in DeepSeek V4 are the experimental edge

The five primitives — attention mechanism, position encoding, MoE pattern, reasoning architecture, optimizer × precision — define a 2026 model. Reading any model card is identifying the choices on these axes. _Most flagship models share most choices_ ; the differences are concentrated in 1-2 axes per model.

## DeepSeek V4: the case study in current architectural innovation

DeepSeek V4 (April 24, 2026) is the most architecturally novel frontier release of 2026 and worth detailed treatment. The release introduced three primitives that other labs will likely adopt:

### Hybrid Attention: CSA + HCA

The headline innovation. Combines two attention mechanisms across layers:

  * **Compressed Sparse Attention (CSA)** : compresses KV entries 4× along the sequence dimension; uses an FP4-based "Lightning Indexer" to select top-k compressed blocks per query. Local context with sparse selection.
  * **Heavily Compressed Attention (HCA)** : applies aggressive 128× compression to the sequence; the compressed sequence is so short that _dense attention becomes cheap_. Used for global view of full context.

The two mechanisms are interleaved across layers. CSA layers handle local detail; HCA layers handle global retrieval. Combined result: **27% of single-token inference FLOPs and 10% of KV cache vs DeepSeek V3.2 at 1M context length**. The Lightning Indexer in FP4 is itself an interesting design — using NVFP4 (M41) for the routing decisions while keeping main attention at higher precision.
    
    
    # Schematic of layer interleaving in DeepSeek V4
    for layer_idx in range(num_layers):
        if layer_idx % 2 == 0:
            # CSA: 4× compression + FP4 lightning indexer + sparse selection
            out = compressed_sparse_attention(x, compress=4, top_k=64)
        else:
            # HCA: 128× compression + dense attention on compressed sequence
            out = heavily_compressed_attention(x, compress=128)
        x = x + out  # residual (or mHC; see below)

### Manifold-Constrained Hyper-Connections (mHC)

Replaces standard residual connections. The problem: at trillion-parameter scale across hundreds of layers, residual connections suffer from signal amplification or collapse — small perturbations compound. mHC constrains the mixing matrices to lie in the **Birkhoff Polytope** (the space of doubly stochastic matrices) using the Sinkhorn-Knopp algorithm. Result: signal magnitude preserved through deep stacks; training remains stable at scale.

This is a clear example of _scale-driven architectural change_. Standard residual connections work fine at 70B; they don't at 1.6T without modification. Other trillion-parameter models will likely adopt similar mechanisms.

### Muon optimizer

DeepSeek V4 switches from AdamW to **Muon** for most parameters. AdamW is retained for embeddings, prediction head, and RMSNorm weights (the parts that benefit from per-parameter adaptive rates). DeepSeek reports faster convergence and more stable training at trillion-parameter scale — peak learning rate 2.0e-4 with cosine decay.

Muon is the first non-AdamW optimizer to make a serious appearance at frontier scale; the choice is being closely watched. Whether it becomes the new default depends on independent reproductions over the next few months.

### Switchable thinking modes

V4 folds the previously separate R reasoning model into the base model with three inference modes:

  * **Non-think (fast)** : standard fast inference; no extended reasoning.
  * **Think High (logical analysis)** : moderate reasoning trace.
  * **Think Max (full reasoning extent)** : extended reasoning chain.

Same model weights; mode determined at inference time. This is the 2026 industry direction — Claude Opus 4.7's extended thinking, Gemini 3 Deep Think, GPT-5.5's reasoning levels are all variants of this pattern. The era of separate "reasoning model" releases is ending.

### FP4 quantization-aware training

DeepSeek V4 trained with **FP4 quantization-aware training applied to MoE expert weights**. This is the production deployment of M41's NVFP4 recipe — full integration into a frontier training run. V4-Pro at 33T training tokens with FP4 QAT validates the approach at trillion-parameter scale.

## Llama 4: the case study in long-context architecture

Llama 4 (April 2025) is the other architectural innovation reference. The headline: **10M token context window in Scout** , achieved without any inference-time fine-tuning by extrapolating from 256K training context.

### iRoPE: interleaved RoPE + NoPE

The mechanism. Layers alternate in a 3:1 pattern:

  * **RoPE layers (3 of every 4)** : traditional rotary positional embeddings; preserve local token order. Apply chunked attention.
  * **NoPE layers (1 of every 4)** : no positional encoding at all. Apply full causal attention without positional constraints. The causal mask provides the only positional signal.

The intuition: traditional RoPE degrades at extreme context lengths because rotational frequencies don't generalize beyond training distribution. NoPE layers don't have this problem — they only use the causal structure. Interleaving gives the model both local positional precision (RoPE) and unlimited length generalization (NoPE).

Combined with **inference-time temperature scaling** , Llama 4 trained at 256K context generalizes to 10M context with near-perfect retrieval (Scout on Needle-in-Haystack at 10M maintains >99% accuracy). This is the published empirical claim; a substantial improvement over models that hit walls at 128K-1M.
    
    
    def irope_layer_pattern(num_layers):
        """Llama 4 iRoPE: 3 RoPE + 1 NoPE per group of 4 layers"""
        pattern = []
        for i in range(num_layers):
            if (i + 1) % 4 == 0:
                pattern.append("NoPE")   # every 4th layer
            else:
                pattern.append("RoPE")   # the other 3
        return pattern

### Scout vs Maverick: pure MoE vs alternating MoE

Llama 4 also illustrates the MoE pattern divergence:

  * **Scout (109B total / 17B active)** : _pure MoE with 16 experts_ ; every FFN layer is a MoE layer. Designed for extreme efficiency at single-GPU inference (fits on one H100 with int4 quantization).
  * **Maverick (400B total / 17B active)** : _MoE alternating with dense layers_ ; 128 experts in MoE layers, but only half the layers are MoE — the other half are dense FFNs. The intuition: dense layers provide unsparsified capacity for tasks that don't fit expert specializations.

Both models have 17B active parameters per token (similar inference speed); they differ in total capacity (109B vs 400B) and how they distribute it. _Same active-compute budget; different total-capacity strategies_. Production teams choose between them based on whether they need maximum capability (Maverick) or efficient deployment (Scout).

### Early fusion multimodality

Llama 4 is natively multimodal — text and image inputs, text and code outputs. **Early fusion** : vision tokens and text tokens enter the unified token stream from the very first layer (rather than being projected later via a connector). This is the Era 3a VLM design (covered below); Llama 4 was an early adopter at frontier scale.

Co-distillation from **Llama 4 Behemoth** (~2T parameter teacher, still in preview at release) using a novel dynamic-weighting loss between student and teacher logits is the training methodology — _knowledge distillation at frontier scale_ , with Behemoth providing the supervision signal for Scout/Maverick.

## The VLM taxonomy: three eras

Vision-language models have a cleaner taxonomic structure than LLMs because the architectural evolution has clearer phases. The Vision-Language-Models-Overview survey (February 2026) names three distinct eras:

Three eras of VLM design — and where 2026 models sit Era 1 (2021-2023): Frozen encoder + frozen LLM + learnable connector Vision enc. (frozen) Connect. (trained) LLM (frozen) Examples: CLIP, BLIP, BLIP-2, Flamingo, MiniGPT-4 Trade-off: cheap to train (only connector); limited capability Status 2026: deprecated for new flagships; useful for research/edge Era 2 (2023-2025): LLM trunk with vision as bolt-on adapter Vision enc. (fine-tuned) Connect. (trained) LLM trunk (fine-tuned) Examples: LLaVA / LLaVA-NeXT, Qwen2.5-VL, GPT-4V, InternVL Trade-off: leverage existing LLM; vision is "bolted on"; mid-fusion Status 2026: still common for open-weight VLMs; research workhorse Era 3a: Native MM input → text out Image, video, (audio) → unified token stream from layer 0; text-only output Examples: Qwen3.5/3.6, Gemma 4, Gemini 3, GPT-5.4/5.5, Claude Opus 4.6/4.7 Era 3b: Native MM input → MM out Same trunk; generates images/audio in addition to text; multi-head decoder Examples: GPT-Image-2, Gemini Native Image, Chameleon, Llama 4 (image gen branch)

The trajectory is clear: _each era integrates more deeply_. Era 1 keeps everything separate and trained independently. Era 2 fine-tunes the LLM but treats vision as bolted on. Era 3 trains everything together from scratch with mixed modalities in the same token stream from the first layer.

The performance gap from Era 2 → Era 3a is substantial — early fusion captures cross-modal interactions that bolt-on architectures can't. The gap from Era 3a → Era 3b is currently smaller (Era 3b is harder to train and image generation has its own quality issues); the long-term direction favors 3b for general-purpose multimodal AI.

### Vision encoders specifically

Within Era 3a (the dominant 2026 design), the choice of vision encoder is one of the few remaining axes:

Vision encoders in 2026 production VLMs Encoder| Origin| Used by| Notes  
---|---|---|---  
**SigLIP 2**|  Google (2025)| Qwen3-VL, Gemma 3| Successor to SigLIP; sigmoid contrastive, multilingual, dense features  
**InternViT-6B**|  Shanghai AI Lab| InternVL line| Scaled vision encoder (6B params); strong on document/chart tasks  
**DINOv2 + SigLIP fused**|  Meta + Google| OpenVLA (M43), Cambrian-1| Combines spatial detail with language alignment  
**Cambrian-1 multi-encoder**|  NYU/research| Cambrian-1| Combines several encoders; captures complementary features  
**Proprietary**|  OpenAI / Anthropic / Google| GPT-5, Claude Opus 4.7, Gemini 3| Undisclosed; likely SigLIP-derivative + custom fine-tuning  
  
SigLIP 2 has become the open-source default in 2025-2026, similar to how Llama became the LLM trunk default. Most open VLMs use SigLIP 2 for vision; the differentiation is in the LLM trunk and training recipe.

## The benchmark saturation phenomenon

One of the defining 2026 features: **benchmarks are saturating**. The "best AI model" question has gotten harder to answer because all the frontier models score within noise of each other on standard benchmarks.

Benchmark scores for top frontier models, April 2026 Benchmark| GPT-5.5| Claude Opus 4.7| Gemini 3.1 Pro| DeepSeek V4-Pro| Spread  
---|---|---|---|---|---  
MMMU-Pro (multimodal)| ~82.0| ~81.0| ~82.5| —| 1.5 pts  
SWE-bench Verified| ~79.5| ~80.8| 80.6| 80.6| 1.3 pts  
GPQA Diamond| 92.8| 91.3| 94.3| ~91| 3 pts  
HumanEval| ~96| ~96| ~96| ~95| 1 pt  
AIME 2025| ~92| ~90| ~93| ~88| 5 pts  
  
The pattern: _top models cluster within 1-3 points on most established benchmarks_. The differentiation has moved to specialized axes (per the multimodal benchmarks article from Digital Applied):

  * **Video understanding** : Gemini 3 leads decisively. Long-form video (movies, lectures) is one of the few benchmark axes still spreading models.
  * **Audio comprehension + ASR-with-reasoning** : Gemini 3 leads; Qwen 3.5 Omni close on real-time applications.
  * **Long-document OCR** : Claude Opus 4.7 holds the crown.
  * **Chart reasoning + infographics** : GPT-5.5 leads.
  * **Code-with-vision** (debugging UI from screenshots): GPT-5.5 leads with longer reasoning traces.
  * **Real-time information** : Grok 4 with live X/Twitter integration is a category of one.

The implication for engineering teams: **"which model is best?" has been replaced by "which model is best for our specific workload?"** The architectural differentiation is real but specialized; the right answer depends on whether you primarily process video (Gemini), code with screenshots (GPT-5.5), long PDFs (Claude), or need 10M-token context (Llama 4 Scout).

## The 25× price gap

The economic story is even more dramatic than the architectural one. The price spread between cheapest and most expensive frontier model in April 2026 is roughly 25×:

Frontier model API pricing, April 2026 (per 1M tokens) Model| Input| Output| License  
---|---|---|---  
Claude Opus 4.7| $15| $75| Closed  
Claude Opus 4.6| $5| $25| Closed  
GPT-5.5| ~$2-3| ~$15-18| Closed  
GPT-5.4| $2.50| $15| Closed  
Claude Sonnet 4.6| $3| $15| Closed  
Gemini 3.1 Pro| $2| $12| Closed  
Grok 4| $2| $15| Closed  
DeepSeek V4-Pro| $0.55-1.74| $3.48| MIT (open weights)  
MiniMax M2.5| $0.30| $1.20| Open weight  
DeepSeek V4-Flash| $0.14| $0.28| MIT (open weights)  
Qwen 3.6 (35B-A3B)| $0.10| ~$0.40| Apache 2.0  
  
The 25× output-price spread (Qwen 3.6 at ~$0.40/M to Claude Opus 4.7 at $75/M) at _roughly comparable benchmark scores_ is the structural change. **Tiered routing strategies are now table stakes for production deployments** : ~70% traffic to the cheapest capable model, ~25% to mid-tier, ~5% to frontier-tier; overall performance indistinguishable from all-frontier routing at ~15% of the cost.

## Why convergence is happening

Three forces compound to produce the 2026 convergence:

  1. **Public papers + open weights cycle**. DeepSeek R1 (Jan 2025) → GRPO universal by July 2025. Llama 4 (April 2025) → MoE universal by year-end. NVIDIA NVFP4 paper (Sep 2025) → DeepSeek V4 FP4 QAT (April 2026). The cycle is ~6 months.
  2. **Talent flow**. Researchers move between labs; methods don't stay proprietary for long. The "secret sauce" of frontier labs is increasingly engineering execution and data, not architectural novelty.
  3. **Benchmark optimization pressure**. Every lab targets the same benchmarks (MMMU, SWE-bench, GPQA, AIME). Models are trained to do well on these specific evals; convergence is partially a measurement artifact.

The implication: **architectural innovation is high-impact but short-lived**. DeepSeek V4's hybrid attention is novel today; by Q3 2026, expect Llama 5 / Qwen 4 / GLM-6 / closed-source flagships to incorporate similar mechanisms. _The half-life of a unique architectural advantage is now ~6-9 months_.

## What's likely next: Q3 2026 and beyond

From the Q2 2026 pipeline signals (per Build Fast With AI's leaderboard):

  * **GPT-5.5 (codename "Spud")** : completed pretraining around March 24, 2026. OpenAI hasn't announced release date; likely Q2-Q3 2026.
  * **Claude "Mythos"** (Anthropic): leaked context from prediction markets in March 2026; unconfirmed by Anthropic.
  * **Grok 5** : Musk has discussed it publicly.
  * **DeepSeek V5** : V4 was preview; V5 likely in Q3 2026.
  * **Qwen 4** : Alibaba's flagship update.

The likely architectural directions for Q3 2026 + based on current trajectories:

  * **Hybrid attention everywhere** : DeepSeek V4-style CSA+HCA or similar in Llama 5, Qwen 4, GLM-6 — the FLOP and KV-cache savings are too large to ignore at 1M+ context.
  * **Trillion-parameter open weights as standard** : V4-Pro is the largest open model today; expect Llama 5, Qwen 4 Max, GLM-6 all in the 1T+ range.
  * **Native NVFP4 training as default** : M41's recipe is mature; new training runs default to it, not FP8.
  * **Unified switchable reasoning** : separate reasoning model releases end; every flagship has Think/Non-Think modes.
  * **Multi-agent + tool use baked in** : Grok 4's parallel-agent architecture, GPT-5.5's native computer use — agentic capability becomes architectural, not bolt-on.
  * **10M+ context becomes common** : Llama 4 Scout's iRoPE was novel in April 2025; by Q3 2026, several flagships will offer 10M+.
  * **Era 3b multimodal output** : more models generating images/audio natively; the gap from Era 3a closes.

The likely _non_ -changes (architectural choices that have settled):

  * Transformer base architecture (despite Mamba/SSM developments — see M33 — production sticks with transformers).
  * Pre-normalization with RMSNorm.
  * SwiGLU / GeGLU activation in FFN.
  * Rotary positional embeddings (RoPE) as the foundation, with extensions (iRoPE, NoPE layers, YaRN).
  * BPE / sentencepiece tokenization.
  * AdamW optimizer (Muon's adoption rate uncertain; standard remains AdamW).

#### Q&A; — About comparing 2026 LLM and VLM architectures **Q:** Why has every frontier model converged on MoE, even though dense models were dominant in 2023-2024? **A:** Three compounding reasons. (1) **Scale economics** : at trillion-parameter scale, dense models become prohibitively expensive at both training and inference. MoE decouples total parameters (capacity) from active parameters per token (compute). DeepSeek V4-Pro at 1.6T total / 49B active runs at the inference cost of a dense ~50B model with capability closer to a dense ~500B model. (2) **Training maturity** : MoE training was tricky in 2022-2023 (load balancing, expert collapse, communication overhead). DeepSeek V3 (Dec 2024) and Llama 4 (April 2025) demonstrated mature training recipes; the engineering problems are solved. (3) **Specialization benefit** : experts specialize during training; different experts handle math, code, multilingual content. The specialization itself improves quality at the same active-compute. _By April 2026, choosing dense at frontier scale is a deliberate eccentricity_ ; the question is which MoE pattern (pure, alternating, shared experts), not whether to use MoE at all. **Q:** Will iRoPE-style hybrid position encoding generalize beyond Llama 4, or is it a one-off? **A:** Likely generalizes — and DeepSeek V4's Hybrid Attention is a related move, suggesting the pattern is broader than just position encoding. The general insight: _uniform layers don't optimize for both local precision AND long-range generalization simultaneously_ ; interleaving specialized layers does. iRoPE (RoPE for local + NoPE for global), Hybrid Attention (CSA for local + HCA for global), MoE alternating with dense (sparse for routing + dense for capacity) — all variations of the same pattern. _Expect more "interleaved-specialized-layer" architectures by Q3 2026_ : e.g., interleaved precision (some layers FP4, others FP8), interleaved depths (some short, some long), interleaved attention types (local sliding-window with global-attention layers). The base insight — that homogeneous architectures suboptimize specific axes — is now established. **Q:** Why are VLM benchmarks saturating, and what does this mean for evaluation? **A:** Saturation has two distinct causes. (1) **Genuine capability ceiling on the benchmark** : MMMU has finite ceiling (100% accuracy); models approaching it have less room to spread. (2) **Training-set contamination + benchmark optimization** : every lab trains against the same benchmarks; the resulting models look similar on those benchmarks but may differ substantially on out-of-distribution evaluation. The implication for engineering teams (per M37's eval rigor playbook): **build your own evals on your specific data**. Public benchmarks are no longer differentiating; held-out task-specific evals are. The "vibes" approach (subjective comparison on real tasks) and per-workload evaluation matrices have become more important than the public leaderboards. M37 territory matters more in 2026 than it did in 2024. **Q:** What's the actual difference between DeepSeek V4 and Llama 4 Maverick in production? **A:** Concrete differences. **Architecture** : DeepSeek V4-Pro at 1.6T/49B active vs Maverick at 400B/17B active — DeepSeek has 4× the total capacity, ~3× the active compute per token. **Attention** : V4 uses Hybrid CSA+HCA; Maverick uses standard GQA + iRoPE. **Context** : both 1M (Scout extends to 10M). **License** : V4 is MIT; Llama 4 has the Llama Community License (more restrictive). **Reasoning** : V4 has unified switchable thinking modes; Maverick relies on prompt-engineered CoT or fine-tuning. **Pricing** : V4-Pro at $0.55/M input vs Maverick at proprietary-tier pricing on most providers. **Use cases** : V4-Pro for cost-sensitive workloads with frontier-quality reasoning; Maverick for permissive license + Meta ecosystem integration. Both are excellent; they target different deployment scenarios. _For most production workloads in April 2026, DeepSeek V4-Flash is the value leader; Claude Opus 4.7 / GPT-5.5 are the quality leaders; Llama 4 Scout is the long-context specialist; Qwen 3.6 is the budget leader._ **Q:** What's the relationship between this module and M14 (mixed precision) + M28 (MoE) + M30 (RoPE)? **A:** Direct extension. M14 covered FP16/BF16/FP8 mixed precision; M41 added NVFP4. This module's "Optimizer × Precision" axis is the production application — DeepSeek V4 trained with FP4 quantization-aware on MoE expert weights is M14 + M41 deployed at trillion-scale. M28 covered MoE mechanics (router, experts, load balancing); this module's "MoE pattern" axis is the variation surface — pure vs alternating vs shared experts is M28 plus production constraints. M30 covered RoPE; this module's iRoPE discussion is M30 plus length generalization extension. _Each architectural axis in this module is "M-X concept × production variant"_ ; you've already seen the underlying mechanics, this module shows how labs deploy them differently. The course taught the primitives; this module taxonomizes their compositions. **Q:** If convergence is happening, why does architecture still matter for engineering teams? **A:** Two reasons. (1) **Convergence is partial, not complete**. The 2.4-point MMMU spread is small; the 25× price spread is large. Models converge on capability, diverge on cost/efficiency/license. Architecture choice (DeepSeek's hybrid attention saving 90% KV cache, Llama 4's 10M context) directly drives operational economics. (2) **Architectural understanding enables better deployment**. Knowing that DeepSeek V4 is hybrid CSA+HCA tells you why long-context is cheap on it; knowing Llama 4 is iRoPE tells you why 10M context retrieval works. M37's eval rigor + M42's compute economics + M44's architectural understanding compose: you pick the right model not by reading press releases but by matching architectural choices to your workload's requirements. _Architecture matters less for "which is best" and more for "which is right for this specific workload"_ ; that's a more sophisticated question that requires deeper architectural literacy, not less. 

## Code Magnets: identify the architecture from a config

You receive a model config dictionary. Identify what architecture it represents by combining the right magnets. Three magnets are misleading.

Arrange the magnets to write a function that classifies a model config into one of: "DeepSeek V4-style", "Llama 4-style", "dense classic".

def identify_architecture(config): has_hybrid_attn = config.get("attention_type") in ("hybrid_csa_hca", "csa+hca") has_irope = config.get("position_encoding") == "iRoPE" and config.get("nope_interval") == 4 has_irope = config.get("position_encoding") == "RoPE" is_moe = config.get("num_experts", 1) > 1 is_moe = config.get("total_params") > 100e9 if has_hybrid_attn and is_moe: return "DeepSeek V4-style" if has_irope and is_moe: return "Llama 4-style" if has_irope: return "Llama 4-style" if not is_moe: return "dense classic" return "MoE — uncategorized"

show solution
    
    
    def identify_architecture(config):
        has_hybrid_attn = config.get("attention_type") in ("hybrid_csa_hca", "csa+hca")
        has_irope = config.get("position_encoding") == "iRoPE" and config.get("nope_interval") == 4
        is_moe = config.get("num_experts", 1) > 1
        if has_hybrid_attn and is_moe:
            return "DeepSeek V4-style"
        if has_irope and is_moe:
            return "Llama 4-style"
        if not is_moe:
            return "dense classic"
        return "MoE — uncategorized"

The traps:

  * `has_irope = config.get("position_encoding") == "RoPE"`: too permissive — checks only for RoPE, misses the iRoPE-specific signature (interleaved NoPE layers every 4th). Plain RoPE is used by most 2026 models (Qwen, GLM, DeepSeek pre-V4, etc.); it's not Llama 4 specific. The Llama 4 signature is _RoPE + NoPE in 3:1 interleaving_ ; you need to check for that interleaving pattern (`nope_interval == 4`) to identify it. Without that check, every RoPE model gets misclassified as Llama 4.
  * `is_moe = config.get("total_params") > 100e9`: confuses model size with MoE structure. Many large models are dense (Nemotron Ultra 253B is dense; Llama 3.1 405B is dense). MoE is identified by having multiple experts in FFN layers — `num_experts > 1` — not by parameter count. The wrong magnet would classify any large dense model as MoE, missing the distinction that the question hinges on.
  * `if has_irope: return "Llama 4-style"` (without checking `is_moe`): triggers Llama-4 classification on any model that happens to have iRoPE-like position encoding, even if it's dense. The complete Llama 4 signature requires both iRoPE AND MoE; either alone is insufficient to identify it specifically. The wrong magnet reduces the test to "if iRoPE → Llama 4," missing the architectural composition.

The pattern: **each architecture is identified by a composition of primitives, not a single primitive**. DeepSeek V4 = MoE + hybrid CSA+HCA attention. Llama 4 = MoE + iRoPE with NoPE-every-4th interleaving. Dense classic = NOT MoE (regardless of position encoding). Reading 2026 model cards is exactly this composition-identification exercise — the primitives are well-known; the labs combine them differently. The most common mistake (Magnet 2) confuses _scale_ with _structure_ ; the second most common (Magnet 3) confuses _part_ with _whole_.

## Who does what?

Match each architectural primitive or model to its real role.

Concept

Real role

Hybrid Attention (CSA+HCA)

A. DeepSeek V4: 4× compress + 128× compress interleaved; 27% FLOPs / 10% KV vs V3.2 at 1M.

iRoPE

B. Llama 4: 3 RoPE layers + 1 NoPE layer interleaved; enables 10M context generalization from 256K training.

Era 3a VLM

C. Native multimodal input → text-out; single transformer trained on mixed modalities (Qwen3.5+, Gemini 3, GPT-5+).

Pure MoE

D. Every FFN replaced with experts (Llama 4 Scout, GLM-5, DeepSeek V4); maximum sparsity for given total capacity.

Manifold-Constrained Hyper-Connections (mHC)

E. DeepSeek V4 residual replacement; constrains mixing matrices to Birkhoff Polytope for trillion-scale stability.

Muon optimizer

F. DeepSeek V4 first-frontier non-AdamW choice; faster convergence at trillion-scale; AdamW retained for embeddings/head.

Switchable thinking modes

G. Single base model with Non-think / Think High / Think Max modes (DeepSeek V4); 2026 industry direction unifying reasoning.

show solution

**Hybrid Attention** → A  
**iRoPE** → B  
**Era 3a VLM** → C  
**Pure MoE** → D  
**mHC** → E  
**Muon optimizer** → F  
**Switchable thinking** → G 

The mental shortcut: _CSA+HCA compresses attention, iRoPE interleaves position, Era 3a fuses early, pure MoE replaces FFNs, mHC stabilizes residuals, Muon replaces AdamW, switchable thinking unifies reasoning_. Each is the 2026 production answer to a specific scaling problem.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** Your team needs to choose a model for a new product: a customer support bot processing internal tickets that include screenshots and PDFs. ~50M tokens/month projected. Walk through the model selection.

show answer

Apply the M42 + M44 framework:

(1) **Workload characterization**. Customer support → moderate reasoning (categorize, draft response). Screenshots → vision needed. PDFs → long-document OCR / understanding. ~50M tokens/month → moderate volume; pricing matters but not extreme.

(2) **Match to model strengths**. Long-document OCR is Claude Opus 4.7's strength (per the differentiation matrix). Code-with-vision (parsing UI screenshots) is GPT-5.5's strength. Both are reasonable choices. Gemini 3.1 Pro is also strong on vision and offers the best $/quality at $2/$12.

(3) **Cost analysis at 50M tokens/month**. Assume 70% input / 30% output split. Monthly cost: Claude Opus 4.7 = 35M × $15 + 15M × $75 = $525 + $1,125 = ~$1,650/month for Opus-tier. GPT-5.5 ≈ ~$300/month. Gemini 3.1 Pro = 35M × $2 + 15M × $12 = $250/month. _Gemini 3.1 Pro wins on price-quality for a moderate-volume vision workload._

(4) **Tiered routing strategy**. Per the M42 + April 2026 framework: 70% of tickets are simple (categorization, quick response) → route to DeepSeek V4-Flash at $0.14/$0.28 = nearly free at 35M tokens/month. 25% are moderate (need vision + reasoning) → route to Gemini 3.1 Pro at $250/month effective. 5% are complex (escalation drafting, multi-step) → route to Claude Opus 4.7 at premium pricing. Total: ~$60/month for the 70% + ~$60/month for the 25% + ~$80/month for the 5% = _~$200/month total_ , vs $1,650/month if routing everything to Opus.

(5) **Validation**. Build M37-grade evals on your specific tickets. Test each tier of the routing strategy. Verify that the cheap-tier handles 70% of tickets correctly; if not, shift the routing percentages.

(6) **Implementation**. Use a routing layer (LiteLLM, custom router); prompt-based difficulty classification (the easy tier sees a quick "is this a simple categorization?" check); fall through to higher tiers on failure.

The general lesson: **in 2026, model selection is portfolio optimization, not single-choice**. The right answer is usually a routing strategy across 2-3 models, not "pick the best model and use it for everything." This is the practical implementation of M42's three-pool decomposition + M44's architectural understanding.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Walk through why DeepSeek V4-Pro can reach 1M context with 27% of V3.2's FLOPs. What's the actual mechanism?

show answer

The mechanism is a layered cost reduction:

(1) **Standard attention cost at 1M context**. Self-attention is O(N² × d) where N is sequence length and d is head dimension. At 1M tokens: ~10¹² operations per head. Multiply by ~64 heads × hundreds of layers → astronomical compute. KV cache: ~40-80 GB of GPU memory just for the cache at 1M tokens with normal attention. Both compute and memory are blockers.

(2) **CSA layers (Compressed Sparse Attention)**. Compress KV entries 4× along sequence dimension. Now N effectively reduced to N/4 = 250K equivalent. Then use FP4-based Lightning Indexer to select top-k blocks per query — only attend to the most relevant compressed blocks. Combined: ~16× FLOP reduction vs full attention at this layer (4× from compression, ~4× from sparsity).

(3) **HCA layers (Heavily Compressed Attention)**. Compress 128× — sequence becomes ~8K equivalent at 1M input. Sequence is so short that dense attention on the compressed version becomes cheap. ~16,000× FLOP reduction vs full attention at this layer (128² compression of the quadratic).

(4) **Layer interleaving**. CSA and HCA layers alternate. CSA captures local detail (with sparsity for efficiency); HCA captures global retrieval (with heavy compression). Each layer's role is specialized; together they cover the attention requirements with much less total compute than uniform full attention.

(5) **KV cache savings**. Compressed KV entries are smaller. CSA stores 4×-compressed KV; HCA stores 128×-compressed KV. Total KV cache: 10% of V3.2's at 1M context. Memory bandwidth and storage both scale down.

(6) **The 27% / 10% numbers**. Single-token inference FLOPs at 1M context: V4 uses 27% of V3.2 (3.7× speedup). KV cache: 10% of V3.2 (10× memory reduction). The combination makes 1M context economically practical for production deployment.

(7) **The Lightning Indexer FP4 detail**. The indexer (which decides which compressed blocks to attend to) runs in FP4. This is M41's NVFP4 applied to the routing decision specifically; the routing accuracy doesn't need full precision since it's selecting top-k anyway.

The general lesson: **the FLOP and memory savings from architectural innovation can dwarf the savings from any other source**. M14's FP8 → M41's NVFP4 saves ~2× compute. DeepSeek V4's hybrid attention saves ~3.7×. Stacking them gives ~7× total. This is why architectural choices matter more than precision choices alone — the FLOP reduction is multiplicative across all the levers.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** A researcher wants to extend a 70B dense model to 1M context efficiently. Sketch the practical paths in 2026.

show answer

Three approaches with different cost/quality tradeoffs:

(1) **YaRN-style RoPE scaling**. Cheapest approach: fine-tune for 1-3 days on long-context data with YaRN frequency rescaling. Works to ~256K-512K with reasonable quality; degrades beyond. ~$10-30K compute. Quick to ship; doesn't really get you to a usable 1M.

(2) **iRoPE-style retrofit**. Replace some RoPE layers with NoPE layers (every 4th, like Llama 4). Requires fine-tuning at extended context (256K typical) and inference-time temperature scaling. Works to 1M-10M with good quality if the original model trained at 256K+. ~$50-200K compute. Substantial engineering investment but proven approach (Llama 4 published recipe).

(3) **Hybrid attention retrofit**. Add CSA + HCA layers in DeepSeek V4 style. Most expensive but gives the best 1M-context economics (27% FLOPs, 10% KV vs original at 1M). Requires substantial restructuring — this is essentially "rebuild the attention mechanism." ~$200K-1M compute and significant engineering. Not a fine-tuning operation; closer to architectural surgery. Best long-term answer if 1M context is critical.

(4) **Hybrid approach**. Use YaRN to get to 256K cheaply; deploy production at 256K; iterate to iRoPE retrofit for 1M+ when needed. Many teams sit at intermediate context lengths because the marginal benefit of going from 256K to 1M doesn't justify the engineering cost for their specific use case.

(5) **Practical recommendation**. For most teams: _don't extend your 70B dense model to 1M; switch to a model designed for it_. DeepSeek V4-Flash at $0.14/M input handles 1M natively. Llama 4 Scout handles 10M natively. The cost of switching to a purpose-built long-context model is typically lower than the cost of retrofitting a 70B dense. The engineering math favors using the right tool, not converting the wrong tool.

The general lesson: **in 2026, architectural retrofitting is rarely the right answer for long-context**. The labs that designed for 1M+ from scratch (DeepSeek V4, Llama 4) have order-of-magnitude advantages. Build on top of them; don't try to bolt 1M onto a 128K architecture.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Forecast: which architectural choices will be in every flagship by Q3 2027? Sketch your reasoning.

show answer

Reading the convergence trajectory and current frontier signals:

**Highly likely universal by Q3 2027** :

(1) **MoE at frontier scale**. Already universal April 2026; no reason to expect divergence. Expect 30-40× sparsity ratios (total/active) as the standard.

(2) **1M+ context windows**. Already standard for new flagships; expect 5M+ as the new baseline by Q3 2027 with iRoPE-style innovations widespread.

(3) **Hybrid attention (CSA+HCA-like)**. DeepSeek V4 demonstrated; expect Llama 5, Qwen 4, GLM-6, GPT-6 (whatever name), Claude 5 to incorporate similar mechanisms. The FLOP and KV savings are too compelling to leave on the table.

(4) **Unified switchable reasoning**. Already converging in April 2026 (DeepSeek V4, Claude Opus 4.7 extended thinking, Gemini 3 Deep Think); separate reasoning models are unlikely to see new releases.

(5) **NVFP4 training as default**. M41's recipe is mature; new training runs default to it. Expect FP4 to be the standard for 2026-2027 training.

(6) **Native multimodal (Era 3a)**. Already universal for closed flagships; open-weight models converging rapidly (Qwen3-VL, Gemma 4, Llama 4 are early adopters).

(7) **Tool use / agentic capability as architectural feature** , not bolt-on. Grok 4's parallel agents, GPT-5.5's native computer use, Claude's MCP integration all point this direction.

**Possibly universal but uncertain** :

(8) **Muon optimizer**. DeepSeek V4 was the first frontier adoption; whether AdamW gets displaced depends on independent reproductions and whether the convergence advantages prove robust. _Could go either way_.

(9) **Manifold-Constrained Hyper-Connections (mHC)**. Specific to DeepSeek V4; might be needed only at trillion-parameter scale. Unclear if smaller models adopt it.

(10) **Era 3b multimodal output**. Currently smaller-scale; depends on whether image/audio generation quality from unified models matches specialized models.

**Unlikely to converge by Q3 2027** :

(11) **Mamba/SSM replacing transformer**. M33's SSM lineage continues but transformers remain dominant. Possibly hybrid architectures (transformer + SSM layers) emerge but full replacement unlikely.

(12) **Sub-FP4 precision**. FP4 is at the edge of viability; FP3 / FP2 training would require substantial new techniques. Possible but not by Q3 2027.

(13) **Multi-agent as the dominant paradigm**. Grok 4's approach is novel but unproven; possible direction but not guaranteed.

The general framework: **architectural choices that compound (compose with multiple other choices) and have public implementations spread fastest**. MoE + hybrid attention + native multimodal + 1M+ context all compound. Specific optimizer choices (Muon) or specific residual choices (mHC) need independent validation before widespread adoption. _Forecast bias: assume convergence on the compound primitives; assume divergence on the specialized ones._

### What just happened?

  * The April 2026 frontier LLM zoo has converged on three macro-patterns: **MoE is universal at frontier scale** , **1M+ context is table stakes** , **reasoning consolidates into the base model** with switchable thinking modes.
  * Five architectural primitives differentiate 2026 models: **attention mechanism** (full / GQA / sparse / hybrid CSA+HCA / sliding-window), **position encoding** (RoPE / iRoPE / YaRN / NoPE), **MoE pattern** (pure / alternating / shared experts / routing variant), **reasoning architecture** (separate model / unified switchable / always-on / multi-agent), **optimizer × precision** (AdamW + FP8 dominant; Muon + NVFP4 frontier).
  * **DeepSeek V4 (April 24, 2026)** : MoE 1.6T/49B active + 284B/13B Flash; **Hybrid Attention** (CSA 4× compress + Lightning Indexer FP4 + HCA 128× compress dense); **mHC residuals** (Birkhoff Polytope, Sinkhorn-Knopp); **Muon optimizer** ; switchable thinking (Non-think / Think High / Think Max); FP4 quantization-aware training; 27% FLOPs / 10% KV vs V3.2 at 1M context. **MIT licensed**.
  * **Llama 4 Scout/Maverick (April 2025)** : MoE 17B active; **iRoPE** (3 RoPE + 1 NoPE per 4 layers); **10M context** (Scout) / 1M (Maverick); early fusion multimodal; co-distillation from Behemoth teacher.
  * Other production frontier MoE models: **GLM-5.1** (744B/40B, MIT, DSA, agentic-focused), **Qwen 3.6** (35B/3B, Apache 2.0, multilingual), **Kimi K2.6** (Moonshot, agentic), **MiniMax M2.5** , **MiMo V2.5** (Xiaomi, 1.02T/42B + 310B/15B multimodal).
  * **Closed-source flagships** : GPT-5.4/5.5 (256K/1M Codex), Claude Opus 4.6/4.7 (200K + 1M beta), Gemini 3.1 Pro (1M, multimodal leader), Grok 4 (multi-agent variant). All converge within 2.4 points on MMMU-Pro, 1.3 points on SWE-bench Verified.
  * **VLM design taxonomy (3 eras)** : **Era 1** frozen encoder + frozen LLM + connector (CLIP, BLIP, Flamingo); **Era 2** LLM trunk + bolt-on vision adapter (LLaVA, Qwen2.5-VL, GPT-4V); **Era 3** single transformer trained on mixed-modality data, forking into **3a** (early fusion text-out, dominant in 2026: Qwen3.5+, Gemini 3, GPT-5.x, Claude 4.x) and **3b** (multimodal output: GPT-Image-2, Chameleon).
  * **Vision encoders** : SigLIP 2 dominant (Qwen3-VL, Gemma 3); InternViT-6B for InternVL; DINOv2+SigLIP fused (M43's OpenVLA, Cambrian-1); proprietary (GPT-5, Claude Opus 4.7, Gemini 3 — undisclosed).
  * **Benchmark saturation** : top four flagships within 2.4 points on MMMU-Pro by April 2026; differentiation moved to **video** (Gemini 3), **audio** (Gemini 3), **long-document OCR** (Claude Opus 4.7), **chart reasoning** (GPT-5.5), **code-with-vision** (GPT-5.5), **real-time** (Grok 4).
  * **The 25× price gap** between cheapest and most expensive frontier model defines 2026 economics. Tiered routing (70% cheap / 25% mid / 5% premium) is the production default; performance indistinguishable from all-frontier at ~15% the cost.
  * **Convergence cycle is ~6 months** : novel architectural choice published → industry-wide adoption. DeepSeek R1 → GRPO universal in 6 months. Llama 4 MoE → universal MoE by year-end. NVFP4 paper → DeepSeek V4 FP4 training in 6 months.
  * Q3 2026+ likely directions: **hybrid attention everywhere** , **trillion-parameter open weights as standard** , **NVFP4 default** , **unified switchable reasoning universal** , **10M+ context common** , **multi-agent / tool use baked in** , **Era 3b multimodal output** closes gap with 3a.
  * The reflex for 2026 model selection: **identify your workload's primary axis** (video / coding / long-doc / multilingual / cost-sensitive); **pick the model with architectural advantages on that axis** ; **route 70% of traffic to a cheaper tier** ; **build M37-grade evals on your specific data**. Architectural literacy enables better deployment decisions; convergence makes the broad differences smaller and the specific ones more important.

## Where this leaves us

You started this course with `x.stride()`. You finished it identifying the architectural primitives that distinguish DeepSeek V4 from Llama 4 from GLM-5 from Claude Opus 4.7. The trajectory: tensor mechanics → autograd → modules → training → distributed → kernels → MoE → transformer → reasoning models → mech interp → multimodal → distributed RLHF → agentic environments → Blackwell/NVFP4 → scaling laws → embodied AI → and now, the 2026 model zoo.

_The architectural literacy you've built is the foundation_. The specific frontier models change every 6 months — DeepSeek V5 by Q3 2026, Llama 5 likely by Q4, Claude 5 / GPT-6 sometime. But the primitives don't change as fast: attention variants, position encodings, MoE patterns, reasoning architectures, optimizers, precision. Reading any new model card is identifying the choices on these axes; you can do that now.

The 2027 frontier models will mostly use these same primitives, in different compositions. The primitives that emerge over the next year (sub-FP4? Mamba-transformer hybrids? something nobody's published yet?) will extend this taxonomy, not replace it. _You'll be able to read those papers_.

This is what the course was for. Now go build something that didn't exist before.
