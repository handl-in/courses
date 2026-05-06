# PyTorch · From Tensor to Kernel

A 47-module HF-style crash course on PyTorch internals, from `x.stride()` through the April 2026 attention-variants compendium.

**Total: 49 files (cover + M1-M47), ~3.3 MB HTML, ~1.7 MB Markdown, ~74 unique anthropomorphized characters**

## Structure

- `html/` — Self-contained HTML versions with inlined CSS, ready to view in any browser
- `md/` — Markdown versions for reading in editors / pushing to docs sites
- `source/` — Python source files that build the modules (`build_module.py` is the renderer)

## Modules

### Part I — Tensors (M1-M3)
- `00_cover` — Cover page
- `01_tensors_from_scratch` — Storage, stride, view, contiguity
- `02_indexing_broadcasting` — Advanced indexing, broadcasting rules, einsum
- `03_devices_dtypes_numerics` — CPU/GPU placement, dtype semantics, numerics

### Part II — Autograd (M4-M6)
- `04_what_backward_does` — The graph, grad_fn, leaf tensors
- `05_autograd_in_practice` — Hooks, in-place ops, retain_graph
- `06_custom_autograd_function` — Writing your own Function subclass

### Part III — Modules & Training (M7-M11)
- `07_nn_module_deeply` — Parameters, buffers, state_dict, hook plumbing
- `08_init_layers_norms` — Init schemes, LayerNorm, RMSNorm, weight tying
- `09_losses_optimizers_schedulers` — Cross-entropy, AdamW, cosine schedules
- `10_datasets_dataloaders` — Workers, samplers, pin_memory
- `11_training_loops` — Standard loop, gradient accumulation, checkpointing

### Part IV — Performance (M12-M14)
- `12_memory` — Allocator, fragmentation, caching, peak vs reserved
- `13_profiling` — torch.profiler, NVIDIA Nsight, finding bottlenecks
- `14_mixed_precision` — FP16/BF16/FP8, GradScaler, autocast

### Part V — Distributed (M15-M18)
- `15_communication_primitives` — All-reduce, all-gather, ring vs tree
- `16_ddp` — DDP internals, gradient bucketing
- `17_zero_fsdp` — ZeRO stages, FSDP, parameter sharding
- `18_tensor_pipeline_parallel` — TP, PP, Megatron patterns

### Part VI — Compilation (M19-M21)
- `19_dispatcher_aten` — How dispatch works, key set, ATen ops
- `20_cuda_semantics` — Streams, events, async semantics
- `21_torch_compile` — Dynamo, AOTAutograd, Inductor

### Part VII — Kernels (M22-M25)
- `22_gpu_programming_model` — SMs, warps, shared memory, occupancy
- `23_cpp_cuda_extensions` — Custom CUDA kernels via PyTorch
- `24_triton` — Triton language, autotuning, kernel patterns
- `25_flashattention` — IO-aware attention, the foundational reformulation

### Part VIII — Inference & MoE (M26-M28)
- `26_quantization` — INT8, INT4, GPTQ, AWQ, calibration
- `27_inference_systems` — vLLM, PagedAttention, batching
- `28_moe_capstone` — Routers, experts, load balancing, capstone synthesis

### Part IX — Synthesis (M29-M33)
- `29_build_transformer` — Build a 7B from scratch end-to-end
- `30_rope` — Rotary embeddings derived and implemented
- `31_post_training` — SFT, DPO, RLHF basics
- `32_debugging` — Loss spikes, NaN debugging, training stability
- `33_mamba` — State space models, selective scan, the SSM lineage

### Part X — April 2026 Frontier (M34-M38)
- `34_reasoning_models` — GRPO, verifiable rewards, DeepSeek R1 lineage
- `35_data_engineering` — Curation, dedup, classifier filtering at scale
- `36_mech_interp` — SAEs, feature circuits, interpretability primitives
- `37_production_eval` — Eval rigor, offline-online gap, custom benchmarks
- `38_multimodal` — Vision encoders, connectors, early/late fusion

### Part XI — Extended Frontier (M39-M43)
- `39_distributed_rlhf` — Distributed RLHF/GRPO at scale
- `40_agentic_rl` — Agentic environments, tool use, verifiers
- `41_blackwell_nvfp4` — Blackwell architecture, NVFP4 training, TMEM
- `42_scaling_laws` — Updated 2026 laws, compute economics
- `43_embodied_ai` — VLA models, sim2real, robot foundation models

### Bonus & Attention Trilogy (M44-M47)
- `44_architecture_comparison` — Comparing latest LLM & VLM architectures (April 2026 zoo)
- `45_attention_variants_1` — Roles, receptive fields & sparsity (Axes A & B)
- `46_attention_variants_2` — Head topology & KV-cache engineering (Axis C)
- `47_attention_variants_3` — Kernels, implementation & compositional architectures (Axes D & E)

## How the modules were built

Each module is a Python file calling the `emit()` function in `build_module.py`. The body is inline HTML with CSS classes that match the renderer's stylesheet. To rebuild:

```bash
python 01_tensors.py
# emits to out/html/01_tensors_from_scratch.html and out/md/01_tensors_from_scratch.md
```

To rebuild all:
```bash
for f in *.py; do
  if [[ "$f" != "build_module.py" && "$f" != "sample_vibe_preview.py" ]]; then
    python "$f"
  fi
done
```

## Vibe pattern (consistent across all 47 modules)

Every module includes:
- 2 anthropomorphized characters with first-person voices and avatar circles
- Key idea callout
- 2+ SVG visual moments
- Concrete working code with syntax highlighting
- Code Magnets puzzle with 3 conceptual traps
- 7-row matching grid (concept ↔ purpose)
- Numbered Q&A (NDQ) section addressing the genuinely interesting questions
- 4 exercises with full solutions
- Bullet-point recap
- Trade-off tables where relevant

## April 2026 currency

The course was written with full April 2026 frontier currency. Notable specific entries woven in:
- DeepSeek V4-Pro (1.6T MoE, April 24 2026): CSA+HCA Hybrid Attention, mHC, Muon optimizer, FP4 QAT
- Llama 4 Scout/Maverick (April 2025): iRoPE 3:1 RoPE/NoPE, 10M context
- GLM-5.1 (Q1 2026): MLA-style + DSA
- Qwen 3.5/3.6: 3:1 GDN/Attention hybrid
- Kimi Linear / K2.6: KDA + MLA hybrid
- Ling 2.5 (1T MoE): Lightning Attention + MLA
- Nemotron 3 Nano: Mamba-Transformer hybrid
- FlashAttention 4 (March 5 2026): Blackwell co-design, CuTeDSL, ~20% over cuDNN FA3
- Mamba-3 (ICLR 2026): trapezoidal discretization, complex-valued state, MIMO
- NSA (ACL 2025 award): three-path trainable sparse
- IHA (Feb 2026): pseudo-heads, +5.8% GSM8K
- MoDA (March 2026): depth attention, +2.11% downstream at 3.7% FLOPs
- Slim Attention (March 2025): K-cache only
- TurboQuant (ICLR 2026, vLLM merge April 15 2026): 2-bit/3-bit KV
- GPT-5.5, Claude Opus 4.7, Gemini 3.1 Pro, Grok 4 (closed-source frontier)
- The 25× pricing gap; benchmark saturation phenomenon

Total: HTML 3.3MB · Markdown 1.7MB · Source 2.4MB
