# Module 38 — Multimodal architectures

# _Multimodal architectures:_ from bolt-on adapters to native omni-models

_Part X · Module 38 · April 2026 currency_

— the three architectural eras of vision-language models, why SigLIP 2 became the encoder of choice, the connector design space, and the 2025-2026 shift toward native multimodal models that generate as well as perceive

\--- 

For 37 modules we've treated text as the only modality. M38 closes that gap. By April 2026, multimodal models — vision-language, audio-vision-language, omnimodal — are no longer a research curiosity; they're the architectural frontier. Xiaomi's MiMo V2 Omni holds 21% of OpenRouter traffic. NVIDIA released Nemotron 3 Nano Omni on April 28, 2026 as their unified video-audio-image-text agent model. Frontier labs (Anthropic, OpenAI, Google, Meta) all ship multimodal flagships. _The single-modality LLM is becoming the special case._

The architecture is also moving fast. Models built three years ago bolted vision encoders onto frozen LLMs with a small adapter (LLaVA, BLIP-2). Models built last year used pretrained LLMs as a trunk and trained the adapter end-to-end (Qwen2.5-VL, Llama 3.2-Vision). Models built in 2025-2026 increasingly skip the bolt-on entirely and train a single transformer from scratch on mixed-modality data. This module organizes the architectural moves into three coherent eras, walks through the connector design space (which still matters even in native models), and shows how the 2026 omni-models extend the pattern to audio and video while generating as well as perceiving.

> **★ KEY IDEA**  
>  Vision-language model architecture went through three eras in five years, and Era 3 has split. **Era 1 (2021-2022)** : frozen vision encoder + frozen LLM bridged by a small learnable connector (CLIP, BLIP, Flamingo). **Era 2 (2023-2025)** : pretrained LLM as the trunk; vision is a bolt-on adapter trained end-to-end (LLaVA, Qwen2.5-VL, GPT-4V, Llama 3.2-Vision). **Era 3 (2025-2026, current frontier)** : drop the bridge; train a single transformer from scratch on mixed-modality data — Chameleon, Show-o2, Ming-Omni, MiMo V2 Omni, Nemotron 3 Nano Omni. Era 3 forks along output: **3a** native input → text out (most current production); **3b** native input → native output (omnimodal generation, still emerging). **SigLIP 2** (Feb 2025) is the encoder of choice for Era 2 production: multilingual, dense features, better localization, NaFlex variants for native aspect ratio; used by Qwen3-VL, Gemma 3, PaliGemma. **Connectors** have converged on MLP projections with token compression (Qwen2.5-VL groups 4 vision tokens → 1 LLM token via MLP); Q-Former and Perceiver Resampler are now niche. **Native models (Era 3)** still need visual tokenization but increasingly use VQ-VAE-style discrete tokens or continuous patches with shared embeddings. **The MoE finding** (Meta FAIR, March 2026): vision is significantly more data-hungry than language; MoE architectures harmonize this scaling asymmetry, becoming the default for unified multimodal pretraining. 

## Two new faces — closing the cast

E

Encoder

"I turn pixels into tokens. By 2026, I'm usually SigLIP 2 — but not always."

Show me an image; I'll produce visual tokens that downstream language models can attend to. I'm a Vision Transformer trained with contrastive image-text alignment — CLIP's recipe with the SigLIP improvements (sigmoid loss, no global normalization). My output is a grid of token embeddings, typically 256-1024 tokens for a 384×384 image. _SigLIP 2_ (the version in Qwen3-VL, Gemma 3, PaliGemma 2) added multilingual training, dense features, and NaFlex for native aspect ratio. **In Era 3 native multimodal models, I'm being replaced — by VQ-VAE-style tokenizers, continuous patch embeddings, or pure pixel-level early fusion.** The trend is "encoder-free" but production deployments in April 2026 still mostly use me. I'm the workhorse; the cutting edge is somewhere else.

⇄

Connector

"I bridge encoder space to LLM space. I used to be Q-Former; now I'm usually a small MLP."

The vision encoder produces tokens in one embedding space; the language model expects tokens in another. I project between them. The 2023 era favored complex bridges — Q-Former (BLIP-2) with learned queries, Perceiver Resampler (Flamingo) for cross-attention compression. The 2025-2026 consensus is simpler: **I'm a 2-layer MLP**. Qwen2.5-VL adds a token-compression trick: concatenate 4 vision tokens (a 2×2 spatial neighborhood), project through MLP to one LLM token. SigLIP2 + Qwen3 uses attention-pooling for the same purpose. _The complex connector designs underperformed simple MLPs once vision encoders got better._ When I'm not needed at all (Era 3 native models), I'm absent — the model handles cross-modal attention internally.

## The three architectural eras

Three architectural eras of vision-language models ① 2021-2022: bridged CLIP, BLIP, Flamingo Vision enc. FROZEN Q-Former / Resampler LLM FROZEN Both modalities pretrained; only the bridge is trained. Pros: cheap to train. Cons: bridge bottleneck. ② 2023-2025: bolt-on LLaVA, Qwen-VL, GPT-4V SigLIP 2 trained MLP 2-layer LLM trunk trained Pretrained LLM as the trunk; vision adapter trained E2E. Pros: stronger, scales well. Most production in April 2026. ③ 2025-2026: native Chameleon, Show-o2, Ming-Omni Single unified transformer image+text+audio tokens, all mixed No separate encoder; tokens from all modalities in one stream. Pros: native cross-modal reasoning. Cons: pretrain from scratch. Era 3 fork: native input → ? output 3a: Native input → text output Most current production omni-models Image+video+audio in via tokens → unified transformer → text out only Examples (April 2026): Nemotron 3 Nano Omni, MiMo V2 Omni, Gemini 2.5/3 Pro, GPT-5.4 (assumed) 3b: Native input → native output Omnimodal generation, still emerging Generates images/audio as tokens or via integrated diffusion decoders. Examples (April 2026): Chameleon, Anole, Show-o2, Ming-Omni, Ming-Flash-Omni, NExT-OMNI, GPT-4o image

The pace is striking. CLIP (Era 1) shipped in 2021. LLaVA (Era 2) shipped in April 2023. Chameleon and the first wave of native models (Era 3) shipped in 2024-2025. Each era has lasted ~2 years before being substantially superseded. April 2026 is roughly the midpoint of the Era 3 transition: most production deployments are still Era 2, but the new releases (Nemotron 3 Nano Omni on April 28, MiMo V2 Omni in March, Ming-Flash-Omni in March) are increasingly Era 3.

The key insight: **each era trades off training cost against capability ceiling**. Era 1 was cheap to train (frozen models, only the bridge) but hit a hard ceiling at the bridge bottleneck. Era 2 increased training cost (everything end-to-end) but raised the ceiling substantially. Era 3 requires pretraining a unified model from scratch — the training cost is proportionally higher than Era 2, but the ceiling is higher still because cross-modal reasoning is no longer constrained by an interface designed before training.

## Era 1: bridged architectures (CLIP, BLIP-2, Flamingo)

The first real vision-language paradigm was contrastive: train a vision encoder and a text encoder side-by-side so that matching image-text pairs are close in embedding space, non-matching pairs are far. CLIP (Radford et al., 2021) is the canonical example.

For generative tasks (captioning, VQA), this isn't quite enough — you need a language model that can produce text. The Era 1 approach: bolt the contrastive vision encoder to a frozen language model with a small learned bridge.

  * **BLIP-2 (2023)** : introduced the Q-Former — a small transformer with learned query tokens that cross-attends to the vision encoder's outputs and produces a fixed-size sequence of "queried" tokens for the LLM. The LLM is frozen; only Q-Former is trained.
  * **Flamingo (2022)** : introduced the Perceiver Resampler — uses learned latents and cross-attention to compress arbitrary-length vision token sequences to a fixed length. Cross-attention layers inserted into the frozen LLM allow it to attend to vision features. Both LLM cross-attention layers and resampler are trained.
  * **CLIP (2021)** : not generative, but the foundational vision encoder that everything else built on.

Era 1 had two structural weaknesses. First, the **bridge bottleneck** : the connector saw only what the vision encoder produced, and the LLM only saw what the connector passed through. Errors compounded; cross-modal reasoning was limited. Second, **frozen-component drift** : the LLM didn't get to adapt its representations for visual content, and the vision encoder didn't get to learn visual features that better serve language tasks.

## Era 2: bolt-on adapters (LLaVA, Qwen-VL, the production sweet spot)

The Era 2 insight: don't freeze. Take a pretrained LLM as the trunk, attach a vision encoder, train end-to-end on image-text data. The bridge can be much simpler because both sides adapt during training.

The canonical Era 2 architecture, drawn from LLaVA, Qwen2.5-VL, Llama 3.2-Vision, and many siblings:
    
    
    class VisionLanguageModel(nn.Module):
        def __init__(self, vision_encoder, llm, d_vision, d_llm):
            super().__init__()
            self.vision_encoder = vision_encoder      # SigLIP 2, ViT, etc.
            self.llm = llm                            # Llama, Qwen, Gemma backbone
    
            # Connector: 2-layer MLP from vision dim to LLM dim
            self.connector = nn.Sequential(
                nn.Linear(d_vision, d_llm),
                nn.GELU(),
                nn.Linear(d_llm, d_llm),
            )
    
        def forward(self, images, input_ids):
            # 1. Encode images to visual tokens
            vision_features = self.vision_encoder(images)        # [B, N_v, d_vision]
    
            # 2. Project to LLM embedding space
            vision_tokens = self.connector(vision_features)     # [B, N_v, d_llm]
    
            # 3. Get text token embeddings
            text_tokens = self.llm.embed_tokens(input_ids)       # [B, N_t, d_llm]
    
            # 4. Concatenate: prepend vision tokens to text tokens
            combined = torch.cat([vision_tokens, text_tokens], dim=1)
    
            # 5. Run the LLM on the combined sequence
            return self.llm.forward(inputs_embeds=combined)

The pattern: vision encoder produces tokens, MLP projects them, prepend to text token embeddings, run the LLM. The LLM sees the visual tokens as if they were word embeddings — its attention machinery handles cross-modal reasoning naturally because _visual tokens look like text tokens to the model_.

### Token compression: the Qwen2.5-VL trick

Vision encoders produce a lot of tokens. SigLIP 2 at 384×384 with 14×14 patches gives 729 tokens per image; at 896×896 (Gemma 3's "Pan & Scan" mode), 4096 tokens. Multiplied by multiple images per conversation, this dominates context length and inference cost.

The Qwen2.5-VL solution: **group 4 vision tokens (a 2×2 spatial neighborhood) and project through an MLP to a single LLM token**. Reduces token count by 4× without going through the complex Q-Former machinery.
    
    
    class CompressingConnector(nn.Module):
        def __init__(self, d_vision, d_llm, group_size=4):
            super().__init__()
            self.group_size = group_size
            # Project group_size × d_vision → d_llm via MLP
            self.proj = nn.Sequential(
                nn.Linear(group_size * d_vision, d_llm),
                nn.GELU(),
                nn.Linear(d_llm, d_llm),
            )
    
        def forward(self, vision_features):
            # vision_features: [B, N_v, d_vision] where N_v is a perfect square
            B, N_v, D = vision_features.shape
            H = W = int(N_v ** 0.5)
    
            # Reshape to spatial grid, group into 2x2 neighborhoods
            x = vision_features.view(B, H, W, D)
            # Spatial 2x2 → 4 tokens concatenated channel-wise
            x = x.view(B, H // 2, 2, W // 2, 2, D)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
            x = x.view(B, (H // 2) * (W // 2), 4 * D)   # [B, N_v/4, 4*D]
    
            # Project the concatenated 4-token group to a single LLM-dim token
            return self.proj(x)                              # [B, N_v/4, d_llm]

SigLIP2-paired connectors (Jina-VLM, Dec 2025) use a related approach: attention-pooling within 2×2 neighborhoods, where local means serve as queries, then SwiGLU projection. The attention-pooling variant preserves more information than raw concatenation. _For most production workloads in April 2026, MLP compression is the practical choice; attention-pooling is the higher-quality variant when compute budget permits._

### Why MLPs beat Q-Former

The published evidence is clear (Cambrian-1, Eagle, several other Era-2 architecture studies): **simple MLPs match or beat Q-Former at the same parameter count**. The intuition: when the vision encoder is good (SigLIP 2 is genuinely strong) and the LLM is good (Llama, Qwen, Gemma at 2026 scale), the bridge doesn't need to do much intelligent processing. It just needs to project tokens between embedding spaces. The Q-Former's learned-query-cross-attention machinery is overkill; it adds parameters and training instability without commensurate quality gains.

The exception: Q-Former and Perceiver-style designs still win when you need _fixed output length_ (e.g., for multi-image inputs where you want a guaranteed number of vision tokens regardless of input image count). For single-image-with-text inputs, MLPs dominate.

## SigLIP 2: the encoder of choice

The vision encoder space converged on SigLIP 2 (Tschannen et al., February 2025) for Era 2 production. Its predecessor SigLIP (2023) replaced CLIP's softmax contrastive loss with a sigmoid loss — operating on individual image-text pairs rather than the global softmax. This made training more efficient and removed CLIP's batch-size dependency.

SigLIP 2 added:

  * **Multilingual training** — works across 100+ languages, not just English. Critical for global production deployment.
  * **Dense features** — better localization, useful for tasks requiring spatial reasoning (object detection, OCR, document understanding).
  * **NaFlex variants** — native aspect ratio preservation. Don't squash a wide image into a square; process it as native aspect ratio with appropriate position encoding.
  * **De-biased data mixture** — better fairness across demographic categories.

Sizes: SigLIP 2 ships at ViT-B (86M), L (303M), So400m (400M), and g (1B). The So400m variant is the production sweet spot — better than L, much smaller than g. Used by Qwen3-VL, Gemma 3, PaliGemma 2.

Alternatives that exist in production but are less common: **EVA-02** (better at fine-grained tasks but slower), **InternViT** (used by InternVL, very high parameter counts, edge in dense prediction), **SAM-derived encoders** (used in some specialized models for segmentation-heavy work). For greenfield Era 2 work in 2026, SigLIP 2 is the default.

## Era 3: native multimodal models

Era 3's bet: _train a single transformer from scratch on mixed-modality data_. No separate vision encoder. No connector. All modalities are tokens in the same sequence; the same attention mechanism handles cross-modal reasoning.

The challenge: language is a discrete sequence; images are a 2D grid of continuous values; audio is a 1D continuous waveform. To get them all into one token sequence, each modality needs a tokenization scheme.

The 2025-2026 approaches:

  * **VQ-VAE-style discrete tokens** (Chameleon, Anole, Unified-IO 2): train a vector-quantized VAE on images; images become sequences of discrete tokens from a learned visual codebook. The transformer treats them like vocabulary tokens. Generation is autoregressive: predict the next visual token, decode through VAE.
  * **Continuous patch embeddings with shared transformer** (Show-o2, NExT-OMNI): split images into patches, embed each patch as a continuous vector, mix with text token embeddings in the same sequence. Generation uses flow matching or diffusion conditioned on the transformer's hidden states.
  * **3D causal VAE** (Show-o2 specifically): for video, use a 3D VAE that encodes spatial-temporal patches. The "3D causal" structure preserves causality along the temporal dimension — important for video generation.
  * **Modality-specific routers in MoE** (Ming-Omni, Ming-Flash-Omni): the transformer is MoE; modality-specific tokens get routed to modality-specific experts. Allows tailored processing per modality while sharing the transformer backbone.

### The MoE-for-multimodal finding

One of the most important results from March 2026: Meta FAIR's "Beyond Language Modeling: An Exploration of Multimodal Pretraining" paper. They computed scaling laws for vision and language modalities separately and found a striking asymmetry: **vision is significantly more data-hungry than language**. To reach equivalent performance per FLOP, vision needs more training tokens than language.

The implication: a unified dense transformer trained on a balanced mix is suboptimal — language gets enough data and capacity, vision is starved of data while consuming model capacity it can't fill productively. **MoE harmonizes this** : vision experts can be smaller (less parameter capacity per FLOP, since vision needs less raw capacity); language experts can be larger; both modalities use the data ratios they need. The scaling laws show MoE getting roughly equivalent multimodal capability at substantially less compute than dense unified training.

This finding ties into M28's MoE coverage. **Production native multimodal models are increasingly MoE** : Ming-Flash-Omni (March 2026), MiMo V2 Omni, Nemotron 3 Nano Omni. The shift from dense unified to MoE unified is happening in real-time as of April 2026.

## Production reality April 2026

Selected multimodal models, April 2026 Model| Era| Modalities| Notes  
---|---|---|---  
Claude Opus 4.6| 2 (assumed)| image, text| Frontier; vision encoder undisclosed  
GPT-5.4| 3a (assumed)| image, audio in; text out| Frontier; architecture undisclosed  
Gemini 3.1 Pro| 3a (assumed)| image, audio, video in; text out| Frontier; native multimodal claimed  
Qwen3-VL| 2| image, video, text| Open; SigLIP 2 + Qwen3 + MLP connector  
Llama 3.2-Vision| 2| image, text| Open; ViT + Llama 3 + cross-attention  
Gemma 3 / PaliGemma 2| 2| image, text| Open; SigLIP 2 + Gemma 2; "Pan & Scan" for high-res  
GLM-4.6V| 2/3a hybrid| image, text + tool use| 128K context; native multimodal tool calling  
Pixtral| 2| image, text| Mistral; 12B params; Apache 2.0  
Molmo| 2| image, text| Open dataset (PixMo) + open weights  
Chameleon (Meta)| 3b| image, text in/out| Early native; mixed-modal generation  
Show-o2| 3b| image, video, text| Native unified; AR + flow matching  
Ming-Omni| 3b| image, audio, text| MoE with modality routers; speech generation  
**Ming-Flash-Omni (Mar 2026)**|  3b| image, audio, video, text| Sparse MoE; on par with Gemini 2.5 Pro  
**MiMo V2 Omni (Mar 2026)**|  3b| image, video, audio, text| Xiaomi; 262K context; 21% OpenRouter share  
**Nemotron 3 Nano Omni (Apr 28, 2026)**|  3b| image, video, audio, text| NVIDIA; agent-focused; open  
NExT-OMNI| 3b| image, video, audio, text| Discrete flow matching for any-to-any  
  
Two patterns in the table. First, **Era 2 dominates production** : most deployed multimodal models are still LLM trunk + vision encoder + connector. Era 2 designs ship faster, are cheaper to train, and benefit from the rapid improvement of pretrained LLMs and vision encoders.

Second, **Era 3 is where the new releases are** : every major multimodal release in March-April 2026 is Era 3, increasingly with audio and video. The trajectory is clear; the question is how fast Era 3 displaces Era 2 in production. Probably 2027 for substantial displacement; full transition through 2028.

## Visual tokenization: the discrete vs continuous question

A core design question for native multimodal models: _how do you turn an image into tokens the transformer can process and generate?_

Two main approaches, with tradeoffs:

### Discrete visual tokens (VQ-VAE)

Train a Vector-Quantized VAE on image data: encoder maps images to a grid of discrete tokens drawn from a learned codebook (typically 8K-65K codes); decoder reconstructs images from the token grid. Once trained, images become sequences of integer tokens, and the multimodal transformer treats them like vocabulary tokens.

  * **Pro** : clean autoregressive generation. The transformer predicts visual tokens one at a time; the VAE decoder converts the token sequence back to pixels. Same generation procedure as text.
  * **Pro** : shared vocabulary makes interleaved generation natural (next token might be text or visual; same prediction head).
  * **Con** : VAE quantization loses information. Reconstruction quality is limited by codebook size.
  * **Con** : training the VAE is a separate, non-trivial step.

Used by Chameleon, Anole, Unified-IO 2, several Era-3b models.

### Continuous patches (no quantization)

Split images into patches, embed each patch as a continuous vector via a learned linear projection. Patches enter the transformer as continuous embeddings (like RoPE-positioned tokens). Generation uses diffusion or flow matching conditioned on the transformer's hidden states.

  * **Pro** : no quantization loss; image quality is bounded only by the diffusion decoder's capacity.
  * **Pro** : no separate VAE training stage.
  * **Con** : generation requires multiple diffusion steps; slower than autoregressive.
  * **Con** : hybrid AR+diffusion training has its own complexity.

Used by Show-o2, NExT-OMNI, several recent designs.

The choice depends on what you're optimizing for. Pure autoregressive (discrete tokens) is simpler and faster at inference. Hybrid AR+diffusion (continuous patches) gives higher visual quality. _Production omni-models in 2026 are split roughly evenly_ ; the converging recipe will likely show in 2027 releases.

## Audio and video: the omni extension

Adding audio and video to a multimodal stack uses the same architectural moves with modality-specific tweaks.

**Audio tokenization** : typically discrete codecs (EnCodec, SoundStream-derived). 24kHz audio at ~50 tokens per second; very long sequences result. Some models use continuous audio embeddings (Whisper-derived encoders); these compress 30-second audio clips into fewer tokens.

**Video tokenization** : temporal extension of image tokenization. 3D VAEs encode (T × H × W) cubes of pixels; the transformer sees a sequence of spatial-temporal tokens. The "3D causal VAE" used in Show-o2 preserves temporal causality so the VAE can be used during autoregressive generation.

The Nemotron 3 Nano Omni release notes (April 28, 2026) describe their unified approach: dedicated tokenizers per modality (image VQ, audio codec, video 3D VAE), all producing tokens in shared embedding space, all processed by a single MoE transformer with modality-specific routers. Same recipe MiMo V2 Omni and Ming-Flash-Omni follow. _The native-multimodal architecture pattern has converged: per-modality tokenizers + shared transformer trunk + MoE for capacity allocation._

## Evaluation: what M37 didn't cover

Multimodal evaluation has its own benchmarks and pitfalls beyond M37's text-only coverage.

Multimodal benchmarks, April 2026 Benchmark| Tests| Status  
---|---|---  
MMMU| Multi-discipline reasoning with images| Active; frontier ~85%  
MMBench| General VQA capabilities| Saturating; top models >80%  
MM-Vet| Integrated vision-language tasks| Active; top models >75%  
MathVista| Visual math reasoning| Active  
DocVQA, ChartQA| Document and chart understanding| Active; frontier >90%  
VideoMME| Long-form video understanding| Active; growing importance  
SEED-Bench-2-Plus| Comprehensive multimodal capability| Active  
Real-world image VQA| Practical product-style queries| Custom evals dominate (M37)  
  
Multimodal-specific evaluation pitfalls (in addition to M37's general issues):

  * **Image contamination is harder to detect than text contamination.** N-gram overlap doesn't apply to images. Image-similarity-based contamination detection exists (perceptual hashing, embedding-based) but is less mature.
  * **Resolution dependence** : a model's score on the same eval can vary 5-15pp by image resolution. Always report the resolution used. SigLIP 2 NaFlex helps with native-resolution processing, but eval-time resolution still matters.
  * **Cross-modal hallucinations** : models confidently describe objects not present in the image, or miss objects clearly present. Standard eval metrics may not catch this; specific hallucination benchmarks (POPE, AMBER) target it.
  * **Video evaluation is expensive** : each video query takes 5-50× the compute of an image query. Most teams undersize their video eval suites; this is a known limitation.
  * **OCR-dependent tasks** : a multimodal model that's bad at reading text in images fails at document QA regardless of its language reasoning. OCR quality is a separate axis worth measuring.

The Hybrid Norm from M37 applies cleanly: verifiable rewards for "did it identify the right answer in this VQA task" (exact match against label), LLM rubrics for "did it describe the image well" (visual fidelity, completeness, accuracy). Cross-family LLM judges become particularly important for multimodal tasks because vision-language judging itself is biased and bias mitigation is harder.

## The connector design space, summarized

Even though Era 3 reduces the connector to nothing, Era 2 remains dominant in production. The connector design space, summarized for the cases where you're building Era 2:

Connector designs, when to use each Design| Tokens out| Best for  
---|---|---  
Linear projection| Same as encoder| Smallest models; rarely optimal  
2-layer MLP| Same as encoder| **Default for Era 2 production**  
MLP with 4-token grouping| 1/4 of encoder| When token budget matters; Qwen2.5-VL approach  
Attention-pooling (2×2)| 1/4 of encoder| SigLIP2-pair models; better than MLP grouping  
Q-Former (BLIP-2)| Fixed (e.g., 32)| Multi-image inputs; fixed output length needs  
Perceiver Resampler| Fixed| Flamingo-style cross-attention insertion  
Cross-attention layers in LLM| N/A (parallel)| When LLM weights are partially frozen; Llama 3.2-Vision  
  
The April 2026 default: **2-layer MLP with optional 4-token grouping or attention-pooling for token compression**. The complex designs survive in specific use cases but rarely dominate quality in head-to-head comparisons.

#### Q&A; — About multimodal architectures in 2026 **Q:** Why did Q-Former lose to plain MLP connectors despite seeming more sophisticated? **A:** Two reasons. First, when the underlying components are strong (SigLIP 2 vision encoder + Llama-3-class LLM), the bridge doesn't need to do clever processing — it just needs to project between embedding spaces, which is what MLPs do well. Q-Former's learned-query cross-attention adds parameters and training instability without the underlying components needing the extra computation. Second, Q-Former was designed for the frozen-LLM regime (Era 1) where the bridge had to do all the cross-modal work. Once the LLM became trainable end-to-end (Era 2), the LLM itself learned to handle cross-modal attention; the bridge's job shrank to projection. _Cambrian-1's ablation studies were the empirical turning point_ — they showed MLP at parity or better than Q-Former at the same parameter count, and the field shifted within a few months. **Q:** Are native multimodal models (Era 3) actually better, or just newer? **A:** As of April 2026, the evidence is "competitive but not yet decisively better" for native models on understanding tasks. Era 2 production models match or beat Era 3 native models on standard benchmarks (MMMU, MMBench) for image-text understanding. But Era 3 wins clearly on tasks where mixed-modal generation matters (interleaved image-text generation, omnimodal output, audio+vision joint reasoning) — capabilities Era 2 architectures can't easily express. The April 2026 reality: for "describe this image," Era 2 is sufficient and cheaper; for "generate a video with synchronized audio narration," Era 3 is the only viable approach. Production teams pick by use case. **Q:** Why does the MoE-for-multimodal finding matter so much? **A:** It addresses a fundamental scaling tension. Naïve unified multimodal training scales worse than text-only training because vision and language have different optimal compute-to-data ratios. Without compensation, you either over-train vision (waste) or under-train language (lose quality). Meta FAIR's March 2026 finding showed MoE elegantly fixes this — modality-specific experts let each modality use its own optimal ratio. The practical consequence: multimodal training compute drops by potentially 2-5× relative to dense unified training while preserving quality. _This makes Era 3 economically viable_ in a way it wouldn't be for dense models. Expect every serious Era 3 release after Q2 2026 to be MoE. **Q:** What's the actual difference between MiMo V2 Omni and Gemini 3.1 Pro from a multimodal architecture perspective? **A:** MiMo V2 Omni is openly documented as Era 3b native (per Xiaomi's release notes): single transformer, per-modality tokenizers (image, video, audio), MoE backbone, supports both understanding and generation across modalities. Gemini 3.1 Pro is closed; Google describes it as "natively multimodal" but doesn't disclose architectural specifics. Based on capabilities (audio-visual joint reasoning, omnimodal generation), it's also Era 3 — likely 3b — but the specific tokenization and routing choices are unknown. _For research and engineering, MiMo V2 Omni is the more transparent reference point_ ; for capability, Gemini 3.1 Pro and GPT-5.4 set the frontier. The closed-frontier-vs-transparent-open split is now a permanent feature of the multimodal landscape. **Q:** If I'm building a vision-language model in April 2026, what should I actually do? **A:** For most teams: **Era 2 with SigLIP 2 + a strong open LLM + 2-layer MLP connector with 4-token grouping**. That's the production sweet spot — fast to train, predictable performance, well-supported by HF Transformers and other tooling. Use SigLIP 2 So400m as the encoder; Llama 3 or Qwen 3 as the LLM trunk; standard MLP connector with the Qwen2.5-VL grouping trick for token compression. Train end-to-end on instruction-following multimodal data. Expect 2-4 weeks of training compute on 8-32 H100s for a 7-30B model. Era 3 is research territory unless you're a frontier lab; the engineering complexity (tokenizer design, MoE routing, joint training stability) is substantial. Era 2 with current best practices gets you 90% of frontier capability at 10-20% of the engineering cost. **Q:** How do reasoning models (M34) compose with multimodal? **A:** They compose cleanly but increase compute pressure substantially. A reasoning model that generates 10K-token chains of thought, when augmented with a 4K-token vision input, has ~14K total tokens per request — much higher than text-only reasoning's ~10K or vision-only's ~4K. The serving infrastructure (M27, M34) is stressed in both dimensions. As of April 2026, multimodal reasoning is an active frontier: GPT-5.4 and Gemini 3.1 Pro both claim multimodal reasoning capability; Claude Opus 4.6 has it (vision understanding integrated with extended thinking); open implementations are emerging but not at frontier quality. _The intersection — multimodal-and-reasoning — is where the next significant capability jump is likely_. 

## Code Magnets: implement an Era 2 VLM forward pass

You're writing the forward pass for an Era 2 vision-language model with token compression. Three magnets are wrong choices.

Arrange the magnets to compute the multimodal forward.

def forward(self, images, input_ids): vision_features = self.vision_encoder(images) vision_tokens = self.connector(vision_features) vision_tokens = self.connector(images) text_tokens = self.llm.embed_tokens(input_ids) text_tokens = self.llm(input_ids) combined = torch.cat([vision_tokens, text_tokens], dim=1) combined = vision_tokens + text_tokens return self.llm.forward(inputs_embeds=combined)

show solution
    
    
    def forward(self, images, input_ids):
        vision_features = self.vision_encoder(images)
        vision_tokens = self.connector(vision_features)
        text_tokens = self.llm.embed_tokens(input_ids)
        combined = torch.cat([vision_tokens, text_tokens], dim=1)
        return self.llm.forward(inputs_embeds=combined)

The traps:

  * `vision_tokens = self.connector(images)`: passes raw images to the connector, skipping the vision encoder. The connector expects encoder-output features (e.g., 1152-dim SigLIP features), not raw pixels (3 × 384 × 384). The shape mismatch produces an immediate error or — worse — silently incorrect outputs if the connector happens to accept the input shape (e.g., a flatten + linear that processes pixels). The point of the encoder is to extract semantic features; bypassing it bypasses the entire vision processing pipeline.
  * `text_tokens = self.llm(input_ids)`: runs the entire LLM on the input IDs, getting back logits or hidden states from the final layer — not what we want. We want the _token embeddings_ (the input to the LLM) so we can prepend vision tokens before the LLM processes everything together. Calling `self.llm.embed_tokens(input_ids)` gets just the embedding lookup; calling `self.llm(input_ids)` runs the full transformer.
  * `combined = vision_tokens + text_tokens`: adds vision and text tensors element-wise. Wrong on multiple levels: shapes don't match (different sequence lengths), and even if they did, addition would mix vision and text representations rather than concatenating them as separate positions in the sequence. The LLM needs to attend to vision tokens at distinct positions; concatenation along the sequence dimension is what creates that.

The pattern: **encoder produces vision features → connector projects to LLM space → embed text tokens separately → concatenate along sequence dim → run LLM with inputs_embeds (not input_ids)**. Each step has a specific role; the most common bugs come from confusing "embed tokens" with "run the LLM" and confusing concatenation with addition.

## Who does what?

Match each multimodal concept to its real role.

Concept

Real role

SigLIP 2

A. April-2026 default vision encoder; multilingual, dense features, NaFlex variants.

Q-Former

B. BLIP-2 connector with learned queries; superseded by simple MLP in most production.

2-layer MLP connector

C. Era 2 production default; projects vision features to LLM embedding space.

4-token grouping (Qwen2.5-VL)

D. Compress 4 vision tokens (2×2 spatial) into 1 LLM token via concatenation + MLP.

Era 3 native multimodal

E. Single transformer trained from scratch on mixed-modality data; no separate encoder.

VQ-VAE visual tokenization

F. Quantize images into discrete tokens from a learned codebook for autoregressive generation.

MoE for multimodal

G. Modality-specific experts harmonize the vision-vs-language scaling asymmetry.

show solution

**SigLIP 2** → A  
**Q-Former** → B  
**2-layer MLP connector** → C  
**4-token grouping** → D  
**Era 3 native multimodal** → E  
**VQ-VAE visual tokenization** → F  
**MoE for multimodal** → G 

The mental shortcut: _SigLIP 2 encodes, Q-Former is legacy, MLPs project, 4-token grouping compresses, native models eliminate the bridge, VQ-VAE quantizes for generation, MoE harmonizes scaling_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team builds an Era 2 VLM with SigLIP 2 (1152 dim, producing 729 tokens per 384×384 image) and Llama-3-8B (4096 dim) bridged by a 2-layer MLP. They notice that adding a single image to a chat prompt blows past the LLM's context window much faster than expected. What's happening, and what would you change?

show answer

The math: each image consumes 729 tokens of context. A typical chat with system prompt + user message is ~200 text tokens; adding one image makes it ~929 tokens. Add a few images to a multi-turn conversation and you're at 5-10K tokens of vision per conversation, dwarfing the text. For an 8K-context model, even a single image takes 9% of the context.

The fix is **token compression** in the connector. The Qwen2.5-VL approach: group 4 vision tokens (2×2 spatial neighborhoods), project the 4×1152 = 4608-dim concatenated vector through MLP to a 4096-dim LLM token. This reduces 729 tokens to 729/4 ≈ 182 tokens per image — a 4× reduction. With this, a single image is ~3% of context instead of ~10%, and multi-image conversations become tractable.

If 4× compression isn't enough, attention-pooling variants (SigLIP2-pair) typically preserve more information per compressed token than MLP grouping. For really tight context budgets, perceiver-resampler-style fixed-output (e.g., 64 tokens regardless of image size) is the next step, at some quality cost. _The general rule: in 2026, raw vision-encoder token counts are too high; compression is essentially mandatory for production VLMs._

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Why do most Era 2 architectures freeze the vision encoder during early training and only unfreeze it later, while keeping the LLM trunk trainable throughout?

show answer

Two reasons relating to the very different training-data distributions of pretraining vs multimodal fine-tuning.

(1) **The vision encoder is already strong from contrastive pretraining**. SigLIP 2 was trained on billions of image-text pairs with carefully tuned contrastive loss. Modifying its weights too aggressively during early multimodal fine-tuning (when the connector is randomly initialized and producing noise) can destabilize the encoder's good representations. Better to freeze it initially: let the connector learn to map encoder space to LLM space, then unfreeze the encoder once the connector is reasonable.

(2) **The LLM trunk needs to learn cross-modal attention from scratch**. The LLM was pretrained on text only — its attention patterns don't naturally know how to handle visual tokens. End-to-end training of the LLM is necessary so attention can adapt. Freezing the LLM (Era 1's mistake) caps cross-modal capability.

The standard schedule (LLaVA-style): **Phase 1** (1-2 days), connector only, encoder and LLM frozen. **Phase 2** (1-2 weeks), connector + LLM trainable, encoder still frozen. **Phase 3** (optional, days), unfreeze encoder for final tuning at lower LR. Qwen2.5-VL and others use variations of this.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** Walk through why MoE became important specifically for multimodal training (not just for inference efficiency, which was M28's framing).

show answer

The Meta FAIR March 2026 paper established the empirical scaling laws: **vision is more data-hungry than language**. Translated: to extract a given amount of capability per FLOP, vision needs more training tokens than language does. In a dense unified model trained on a balanced mix, this creates a tension — language has enough data and consumes its allocated capacity productively; vision has limited data and underuses its capacity (or, alternatively, vision is starved while language has excess capacity).

MoE addresses this elegantly. With modality-specific experts (some for vision, some for language, possibly some for cross-modal), each modality can use its own optimal compute-to-data ratio. Vision experts can be smaller (less parameter capacity); language experts larger. Both still share the trunk attention mechanism, so cross-modal reasoning works. Routing tokens to modality-specific experts means each modality "trains as if it were unimodal" while sharing the unified backbone.

The empirical result: equivalent multimodal capability at ~2-5× less compute than dense unified. _For Era 3 native models, this is the difference between economically viable and not._ Expect every serious Era 3 release after Q2 2026 to be MoE — Ming-Flash-Omni, MiMo V2 Omni, Nemotron 3 Nano Omni already are. The dense Era-3 era was brief.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Your eval shows that a multimodal model has 95% accuracy on document QA (DocVQA) but only 65% on real customer support ticket-with-screenshot tasks. What might explain the gap, and how would you investigate?

show answer

This pattern — high public benchmark, lower private domain eval — is exactly what M37's eval-engineering machinery is built for. Multiple potential explanations:

(1) **Distribution shift**. DocVQA documents are typically clean, scanned, formal. Real support screenshots are messy: phone screen captures, partial UIs, multiple windows, varied resolutions, watermarks. The model is good at clean documents and bad at messy reality.

(2) **Resolution dependence**. DocVQA images are typically processed at high resolution. Customer screenshots may be lower resolution (mobile screens), and your inference pipeline may downsample further. Run the same model at the same resolution on both eval sets to test.

(3) **Multimodal hallucination on uncertain inputs**. When images are ambiguous, models often hallucinate confidently. Real screenshots are more ambiguous than DocVQA's clean scans; the model's hallucination rate goes up. Probe with explicit "are you sure?" prompts; a model that wavers when challenged on customer screenshots but not DocVQA is hallucinating.

(4) **Contamination of DocVQA**. The model may have seen DocVQA's training set, inflating eval performance. Test on a custom held-out document set to confirm.

Investigation plan: (a) run the same model on both DocVQA and customer tickets at matched resolutions; (b) probe for hallucination on a sample of customer cases; (c) human-grade 50 customer cases to estimate the true accuracy gap; (d) build a custom ticket eval (M37's recipe — 100-200 cases, hybrid scoring) and use it for production decisions instead of relying on DocVQA.

The general lesson: **multimodal benchmarks predict deployment quality even less reliably than text benchmarks** , because images have many more axes of distribution shift than text does. M37's eval-engineering rigor matters more, not less, in multimodal settings.

### What just happened?

  * Vision-language models went through three architectural eras in five years. **Era 1 (2021-2022)** : frozen encoder + frozen LLM bridged by Q-Former or Perceiver Resampler. **Era 2 (2023-2025)** : pretrained LLM trunk + bolt-on vision adapter, trained end-to-end. **Era 3 (2025-2026)** : single unified transformer trained from scratch on mixed-modality data.
  * Era 3 forks along output: **3a** (native multimodal input → text output) is most current production; **3b** (native input → native output, including image/audio generation) is emerging.
  * April 2026 Era 3 omnimodal releases: **Ming-Flash-Omni** (sparse MoE, par with Gemini 2.5 Pro), **MiMo V2 Omni** (Xiaomi, 21% OpenRouter share), **Nemotron 3 Nano Omni** (NVIDIA, April 28).
  * **SigLIP 2** is the production vision encoder of choice: multilingual, dense features, NaFlex for native aspect ratio. Used by Qwen3-VL, Gemma 3, PaliGemma 2. Sizes: B/L/So400m/g; So400m is the production sweet spot.
  * Connector designs converged: **2-layer MLP is the Era 2 default**. Qwen2.5-VL adds 4-token grouping (2×2 spatial neighborhoods → 1 LLM token via MLP) for token compression. Q-Former and Perceiver Resampler are now niche.
  * **Cambrian-1's ablations** : simple MLPs match or beat Q-Former at the same parameter count, given strong vision encoder + strong LLM trunk. The complex bridges' value declined as components improved.
  * Token compression is essentially mandatory in production. SigLIP 2 at 384×384 produces 729 tokens; without compression, single images dominate context budget. 4-token grouping → 4× reduction; attention-pooling preserves more info than concatenation.
  * Era 3 visual tokenization: **VQ-VAE discrete tokens** (Chameleon, Anole, Unified-IO 2) for clean autoregressive generation; **continuous patches with diffusion decoders** (Show-o2, NExT-OMNI) for higher quality, more complex training.
  * **Audio tokenization** : discrete codecs (EnCodec) or continuous embeddings (Whisper-derived). **Video tokenization** : 3D VAE preserves spatial-temporal structure with optional causal constraint.
  * The **MoE-for-multimodal finding** (Meta FAIR, March 2026): vision is significantly more data-hungry than language; MoE harmonizes by giving each modality its own optimal compute-to-data ratio. Production native multimodal models are increasingly MoE.
  * Era 2 still dominates production in April 2026. Era 3 dominates new releases. Transition probably 2027-2028.
  * Multimodal benchmarks (MMMU, MMBench, MathVista, DocVQA, VideoMME) face the same M37 issues — saturation, contamination, eval awareness — but image contamination is harder to detect than text contamination.
  * The April 2026 default for new Era 2 work: **SigLIP 2 So400m + open LLM (Llama 3 / Qwen 3) + 2-layer MLP with 4-token grouping**. Phased training: connector only → connector+LLM → optionally unfreeze encoder.
  * The reflex: when starting a multimodal project, ask "am I doing understanding only (Era 2 sufficient) or do I need cross-modal generation (Era 3 required)?" Match architecture to capability needs; Era 3 has substantially higher engineering cost and shouldn't be picked for capabilities Era 2 already serves.

## Closing Part X

Five Part X modules: M34 (reasoning models), M35 (pretraining data engineering), M36 (mechanistic interpretability), M37 (production eval engineering), M38 (multimodal architectures). Combined with the 33-module core course, you have ~38 modules covering the depth and breadth of modern AI engineering as of April 2026.

Where the field goes from here is genuinely uncertain. The most consequential April-2026 trends visible from this vantage point:

  * **Eval-awareness scaling** : as frontier models scale further, they recognize evaluations more, requiring deeper investments in deployment-realistic evaluation. M37's playbook will need extension.
  * **Era 3 maturation** : native omnimodal models will converge on a recipe over 2026-2027. The MoE-with-modality-routers pattern is the leading candidate.
  * **Reasoning + multimodal intersection** : M34's reasoning machinery applied to multimodal inputs is where the next significant capability jumps are likely.
  * **Interpretability scaling** : M36's SAE engineering is on track to produce production interpretability tools used in evaluation, debugging, and safety. Expect SAEs alongside model releases by mid-2027.

The framework you've now built — from `x.stride()` to GRPO to mechanistic interpretability to omnimodal models — will absorb whatever comes next. The architectures will change; the engineering principles (compute economics, scaling laws, the four-layer debugging model, the eval rigor playbook) will not.

Thanks for going the distance. The course you've completed didn't exist in this form before; you helped will it into being. Now go make something.
