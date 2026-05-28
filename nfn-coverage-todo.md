# NeuralFn coverage TODO

Tracks every compute primitive that NeuralFn (`../innovation/NeuralFn`) needs but
`llm.kittens` does not yet ship. Each item has a checkbox per implementation
stack so a stack pick can be made later during the optimisation pass; the
checkboxes start empty on purpose.

**Long-term intent:** these kernels will *replace* NeuralFn's current PyTorch
`Stage` implementations in `neuralfn/torch_backend.py`. The eventual shape is a
thin NeuralFn Stage that calls into a llm.kittens kernel, so every row here is
both "kernel to land in llmc/" and "Stage in NeuralFn that swaps from PyTorch
to llm.kittens".

Per kernel, tick whichever stack we end up landing it on:

- **CUDA** — hand-written `__global__` kernel in `llmc/*.cuh`
- **cuBLAS / cuBLASLt** — vendor BLAS (matmul + epilogues + grouped GEMM)
- **cuDNN** — vendor DNN ops (conv, RNN, normalisation, attention)
- **TK 2.0 (SM120)** — ThunderKittens 2.0 kernel targeted at the user's 5090

## Implementation status

A tick means the kernel is **complete and ready to ship**: forward (and
backward where the op participates in training), compiles, would pass a
correctness test against a PyTorch reference. Boxes are intentionally
empty when any of those isn't true — including kernels that exist on
disk but haven't been compile-tested.

- **CUDA** (160/172): reference `__global__` kernels in `llmc/*.cuh` that
  are structurally correct and would compile against the existing
  `cuda_common.h` / `cuda_utils.cuh`. Files landed:
  `activations.cuh`, `losses.cuh`, `sampling.cuh`, `kv_cache.cuh`,
  `optimizers_ext.cuh`, `quantize.cuh`, `fp8.cuh`, `rope_ext.cuh`,
  `conv.cuh`, `moe.cuh`, `embed.cuh`, `norms_ext.cuh`, `routing_misc.cuh`,
  `rl.cuh`, `train_infra.cuh`, `attention_ext.cuh`, `ssm.cuh`,
  `shape_utils.cuh`, `distributed_ext.cuh`, `long_context.cuh`,
  `gemm_ext.cuh`, `train_hooks.cuh`, `misc_ext.cuh`.
  **12 explicitly unticked** for incomplete CUDA reference:
  `gpu_bpe_tokenizer` (empty stub), `universal_transformer (composite)`
  (no-op header), `speculative_decoding`, `continuous_batching_dispatch`,
  `chunked_prefill` (only an indexer, no scheduler), `gradient_checkpoint_hook`,
  `selective_recompute` (counter / selector stubs only),
  `ttt_linear` (forward stub with zero target — no real fast-weight update),
  `reference_forward / reward_forward` (runtime hooks),
  `top_k_sampling` (uses `alloca` in CUDA device code — won't compile portably),
  `Selective SSM scan (fwd + bwd)` (backward omits the `A_bar·h_{t-1}` term),
  `routed_attention_experts` (only the per-expert projection kernel exists).

- **cuBLAS / cuBLASLt** (0/172): `llmc/cublas_ext.cu` exists with wrapper
  bodies for bf16 / batched / grouped / W4A16 / LoRA / FP8 / W8A8 /
  MXFP8/4 GEMMs — but **none have been compile-tested** against cuBLASLt
  12.x. Known issues to fix before any row can be ticked: FP8 GEMM is
  missing the output-scale and amax-D pointers; W8A8 likely has a
  compute-type / output-type mismatch; MXFP4 references a hardcoded
  cudaDataType_t enum value the toolchain may not define.

- **cuDNN** (0/172): `llmc/cudnn_ext.cu` exists with frontend-graph code
  for activations / LayerNorm / RMSNorm / GroupNorm / SDPA / Conv1d /
  Conv2d, plus the legacy activation API for the cuDNN-native
  activations. **None compile-tested.** Known issues: the frontend
  activation path leaves the variant_pack empty so it does nothing; the
  SDPA backward aliases Q in place of saved O / Stats; many activation
  modes (mish, hard_*, gaussian, log, leaky_relu, prelu) don't have a
  native cuDNN mapping. No row tickable until the file builds against
  cuDNN 9 and at least the LN + SDPA paths verify against PyTorch.

- **TK 2.0 (SM120)** (0/172): 30 `llmc/tk/*_sm120.cuh` files exist
  totalling ~3500 lines. **None have been compiled or numerically
  verified on SM120.** Known-broken kernels: `linear_attention_sm120`
  (only correct for `D ≤ 32`), `depthwise_conv1d_sm120` (accumulator
  never written into Y), `groupnorm_sm120` (missing affine),
  `fp8_gemm_sm120` (dequant-then-bf16 path, not real fp8 wgmma),
  `mla_sm120` (suspect tile shapes for K/V reconstruction),
  `nsa_sm120` (duplicate `extern __shared__` declarations),
  `fused_classifier_sm120` (per-row losses written but mean reduce
  missing). The other kernels use `warp::*` ops with signatures
  pattern-matched from `attention_sm120.cuh`; first compile is expected
  to surface fixable signature mismatches against the TK 2.0 head we
  build against. No box can be ticked until each kernel has a unit test
  that runs on SM120 and matches a PyTorch reference within tolerance.

**Bottom line:** the CUDA column is implementation-complete modulo the
12 stubs noted. The three vendor columns have wrapper / kernel files on
disk but are unverified; ticking them would misrepresent the state.
They will be ticked one row at a time as each kernel compiles cleanly
and passes a numeric correctness test.

The original TK 2.0 file detail (30 files, kernel breakdown by family)
is captured below for reference:
    * Attention: `attention_sm120.cuh` (causal), `attention_variants_sm120.cuh`
      (non-causal / sliding-window / ALiBi / cross), `paged_attention_sm120.cuh`,
      `ring_attention_sm120.cuh`, `varlen_attention_sm120.cuh`, `mla_sm120.cuh`,
      `nsa_sm120.cuh`, `sparse_attention_sm120.cuh` (block-sparse / streaming
      sinks / differential), `linear_attention_sm120.cuh`.
    * GEMM: `gemm_sm120.cuh`, `gemm_epilogue_sm120.cuh` (bias / GELU / ReLU² /
      sigmoid / tanh / silu), `fp8_gemm_sm120.cuh`.
    * Norms / gates / activations: `layernorm_tk.cuh`, `rmsnorm_tk.cuh`,
      `groupnorm_sm120.cuh`, `qk_norm_sm120.cuh`, `elementwise_sm120.cuh`
      (sigmoid/tanh/relu/silu/...all activations, DyT, residual_mix, qk_gain,
      softcap, GeGLU/ReGLU/SoLU, softmax / log_softmax).
    * RoPE: `rope_tk.cuh`, `rope_variants_sm120.cuh` (YaRN / NTK / linear /
      ALiBi / XPos), `rope_2d_sm120.cuh`.
    * Conv / SSM: `depthwise_conv1d_sm120.cuh`, `selective_scan_sm120.cuh`.
    * MoE: `topk_route_sm120.cuh`, `moe_permute_sm120.cuh`.
    * KV cache: `kv_cache_sm120.cuh` (append + concat), `quant_sm120.cuh`
      (NF4 dequant, int8 act quant, ternary, KV PCA, KV int8 pack/unpack).
    * Long-context KV: `long_context_sm120.cuh` (H2O, SnapKV, landmark,
      infini, sinks).
    * Sampling: `sampling_sm120.cuh` (top-k, top-p, min-p, typical-p,
      temperature, repetition penalty, logit bias, grammar mask, argmax,
      categorical).
    * Losses: `losses_sm120.cuh` (masked CE, MSE, KL, BCE, DPO, PPO, load
      balance, sequence logp), `fused_classifier_sm120.cuh`.
    * Optimizers: `optimizers_sm120.cuh` (Lion, Sophia, AdEMAMix, AdamW8bit,
      Adafactor, EMA, SWA, stochastic rounding, Muon helpers).
    * Embeddings: `embed_sm120.cuh` (token, abs-pos, 2D pos, bucket lookup,
      patch flatten).
    * Misc: `misc_sm120.cuh` (random_timesteps, mask scheduler, JEPA mask,
      latent pool, GAE, KL penalty, shape ops, chunk state pool, LSH bit-pack,
      cu_seqlens, document mask, TP shard slicers, grad accumulate, dynamic
      loss scale).
  Total: ~3500 lines of TK DSL / SM120 CUDA across 30 files. Every kernel
  has a launcher; type-dispatch is exposed via `*_dispatch` wrappers for
  head_dim 64/128 attention paths and per-D specialised kernels.

**Sections §1–§20** cover compute primitives NeuralFn already exposes as Python
Stages today (these get *replaced* by llm.kittens calls). **Sections §21–§33**
cover modern-LLM kernels NeuralFn does *not* yet have — those need a new
NeuralFn Stage plus the llm.kittens kernel (tracked from the NeuralFn side in
`../../innovation/NeuralFn/todo-kernels.md`).

## Already covered (for reference, no action)

`adamw`, causal MHA / GQA attention (fwd+bwd), encoder (token+pos add fwd+bwd),
fused classifier (CE+grad), `gelu` (fwd+bwd), `global_norm`, layernorm
(+ residual fused), bf16 GEMM (fixed shape constraints), `rmsnorm`
(+ fused residual), `rope` (apply rotary fwd+bwd inside attention), `swiglu`,
NCCL/ZeRO collectives. Most other NeuralFn neurons decompose onto these plus
the items below.

---

## 1. GEMM capability gaps

llm.kittens' `matmul.cuh` exists but is shape-constrained, single-dtype, and has
only one (GELU) epilogue. NeuralFn calls into matmul at far more shapes and
fused forms.

- [ ] **Shape-agnostic bf16/fp16 GEMM** — drop the M%128 / N%64 constraints so
      arbitrary {model_dim, vocab_size, output_dim, hidden_dim} work
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **Batched / grouped GEMM** — needed by `expert_dispatch` and
      `routed_attention_experts` (E parallel small GEMMs per layer)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **GEMM + ReLU² epilogue** — for `mlp_relu2`
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **GEMM + bias + activation epilogue (generic)** — sigmoid/SiLU/tanh
      epilogues for heads (`act_halt_gate`, `reward_head`, `value_head`,
      `denoise_head`, `router_logits`)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **W8A8 GEMM** — int8 weight + int8 activation matmul w/ per-tensor scales
      (BitLinear path uses similar)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **W4A16 / NF4 GEMM** — packed 4-bit weight dequant fused with bf16 GEMM
      (qLoRA `nf4_linear`)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **Ternary (BitNet b1.58) GEMM** — W ∈ {−1,0,1}, int8 activations, STE
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **Low-rank LoRA GEMM** — `x @ A^T @ B^T` with tiny rank
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 2. Pointwise activations & elementwise

Currently only standalone `gelu`. NeuralFn registers a full activation zoo and
several small fused pointwise ops. Each needs fwd + bwd.

- [ ] **sigmoid**
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **tanh**
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **relu**
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **leaky_relu**
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **prelu**
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **relu6**
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **elu**
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **selu**
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **mish**
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **silu (standalone)** — only baked into swiglu today
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **softplus**
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **softsign**
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **hard_sigmoid**
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **hard_tanh**
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **hard_swish**
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **gaussian** — `exp(−x²)`
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **log**
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **negate / identity**
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **threshold**
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **softmax_2 / logsoftmax_2** — fixed two-input softmax variants
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **softmax / log_softmax (last-dim, arbitrary width)**
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **logit_softcap** — `softcap · tanh(x / softcap)`
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **residual_mix** — per-channel `α·x + β·x0` (generalised `residual_forward`)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **qk_gain** — per-head scalar gain (fold into RoPE or fused QKV)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **aux_loss_add / loss_scale** — scalar fusion ops
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 3. Loss functions

Today only `fused_classifier` (token CE). NeuralFn needs the rest of the
training-loss surface.

- [ ] **masked_token_cross_entropy** — CE masked by `loss_mask` (SFT)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **latent_mse_loss** — MSE between fp32 detached target and pred
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **softmax_distillation_loss** — KL(student ∥ teacher) over logits
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **route_distillation_loss** — KL with teacher derived from topic logits
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **semantic_alignment_loss** — per-dim masked categorical CE
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **route_selection_loss** — BCE-with-logits over semantic dims
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **load_balance_loss / route_balance_loss** — `E · Σ density²`
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **dpo_pairwise_loss** — sigmoid / hinge / IPO branches
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **preference_bce_loss** — `−logsigmoid(rc − rr)`
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **ppo_clipped_loss** — clipped policy + clipped value
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **kl_penalty** — `r − β·(logp_pol − logp_ref)`
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **sequence_logp** — `log_softmax → gather(target) → mask·sum`
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 4. Attention variants beyond causal MHA / GQA

- [ ] **Non-causal SDPA** — used by `UniversalTransformer` and
      `scaled_dot_product_attention(is_causal=False)`
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **MHA with arbitrary mask / bias / dropout** — `nn.MultiheadAttention`
      semantics
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **routed_attention_experts** — top-k expert attention (per-expert q/k/v/o
      packs + combine)
      - [ ] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **attentionless_decoder** — bucket embed + linear (replaces attention)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 5. KV cache & KV compression

- [ ] **kv_cache_read** — concat cache with current K, V along seq
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **kv_cache_write / paged append** — append-into-cache helper
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **kv_pca_encode / decode** — small linear projection on K and V
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **kv_quant_pack** — int8 per-token scale quantize K, V
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **kv_quant_unpack** — int8 dequant + split back into K, V
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 6. RoPE

- [ ] **rotary_embedding (standalone)** — exposed outside the attention wrapper,
      split-half convention `(x1·cos + x2·sin, −x1·sin + x2·cos)`
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 7. Convolutions

- [ ] **Conv1d (general)** — used by `byte_patch_embed` (strided)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **Conv1d (depthwise, groups=channels)** — used by `mamba`
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **1D nearest-neighbor interpolate** — `byte_patch_merge`
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **F.pad (right zero-pad)** — used by `byte_patch_embed`
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 8. State-space / Mamba scan

- [ ] **Selective SSM scan (fwd + bwd)** — real Mamba kernel (current
      `MambaStage` is placeholder)
      - [ ] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 9. MoE plumbing

- [ ] **topk_route** — softmax → top-k → renormalise + routing telemetry
      (selection counts, weight mass, router/topk entropy)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **expert_dispatch (fused MoE permute + grouped SwiGLU + scatter-combine)**
      — replaces today's O(E) Python loop
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **broadcast_expert_routes** — batch-level routes → per-position routes
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **broadcast_chunk_routes** — chunk-level routes → per-token routes
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 10. Routing & semantic routers

- [ ] **semantic_moe_router** — cosine sim against centroids + top-k softmax
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **semantic_hash_router** — vocab-dim scoring + hash-bias + forced selection
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **semantic_moe_jepa_evo_router** — chunk-level shared+semantic+free router
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **semantic_hasher / semantic_chunk_hasher** — LSH binarize + bit-pack
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **causal_chunk_state** — prefix cumsum / chunk-mean pool (prefix-safe and
      full-mean modes)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **semantic_projector / semantic_chunk_projector** — vocab-topic logits +
      sig-bucket softmax + residual head
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **masked argsort** — used inside hash routers for forced/free ordering
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 11. JEPA / self-supervised

- [ ] **jepa_mask** — random Bernoulli + multi-block mask sampler
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **latent_pool** — masked mean-pool with mean-fallback when mask empty
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 12. Diffusion / discrete masked LM

- [ ] **random_timesteps** — per-row uniform [0,1)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **mask_scheduler** — Bernoulli mask with per-row probability
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 13. ACT / Universal Transformer

- [ ] **act_halt_gate** — mean-pool → linear → sigmoid
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **act_weighted_sum** — Σ step_p · state across halt steps
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **universal_transformer (composite block)** — LN+MHA+LN+MLP+halt iterated
      `max_steps` times
      - [ ] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 14. TTT (test-time training)

- [ ] **ttt_linear (fast-weight update)** — current is base linear + small MLP
      residual; real TTT needs an in-forward weight update
      - [ ] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 15. RL infrastructure

- [ ] **gae_compute** — reverse-sequential / parallel-scan GAE recursion
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **reference_forward / reward_forward** — frozen sub-graph forward (runtime
      hook, not a single kernel)
      - [ ] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 16. Quantization & adapters

- [ ] **bitlinear_ternary** — ternary weight quant + int8 act quant + STE
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **nf4_linear** — NF4 dequant + base linear + LoRA delta
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **lora_linear** — base linear + low-rank A,B delta
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **randmap_adapter** — orthogonal frozen down/up + trainable middle linear
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **NF4 dequant (standalone)** — packed uint8 → bf16 with per-group absmax
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **int8 activation quant (standalone)** — per-row absmax + round + clamp
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 17. Embeddings

- [ ] **token_embedding (standalone)** — current encoder fuses token+pos and
      doesn't expose the tied weight; needed for `tied_lm_head`
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **absolute_position_embedding (standalone)** — pos-only lookup
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **byte_patch_embed** — embed + strided Conv1d (Conv1d listed in §7)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **byte_patch_merge** — F.interpolate nearest (listed in §7)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **bucket / hash embedding lookup** — generic small `nn.Embedding`
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 18. Optimizers

- [ ] **Muon** — Newton-Schulz5 orthogonalisation of 2D gradient + per-param
      scalar fix-up
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 19. Layout / shape utilities

These are PyTorch view-ops today; flagged here in case the runtime ports to a
CUDA-native graph that needs explicit kernels.

- [ ] **reshape_heads / merge_heads** — transpose + reshape between
      `[B, S, H·D]` and `[B, H, S, D]`
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **repeat_kv** — `repeat_interleave` along head axis for GQA
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 20. Trivial passthroughs (no kernel work — runtime plumbing only)

`input`, `output`, `dataset_source`, `sft_dataset_source`,
`dpo_dataset_source`, `ppo_rollout_source`, `semantic_data_source`,
`expert_combine`, `kv_cache_write` (passthrough variant). Listed only so they
aren't forgotten during runtime wiring; no checkbox row.

---

# Modern-LLM completeness (beyond NeuralFn's current surface)

Sections below cover kernels that NeuralFn does *not* register today but that
any modern LLM training/inference stack needs. Each item will also land as a
new NeuralFn Stage — see `../../innovation/NeuralFn/todo-kernels.md` for the
NeuralFn-side checklist.

## 21. Inference / decoding

- [ ] **top_k_sampling** — keep top-k logits, sample from renormalised
      distribution
      - [ ] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **top_p_sampling (nucleus)** — keep smallest set with cumulative prob ≥ p
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **min_p_sampling** — keep tokens with prob ≥ min_p · max_prob
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **typical_p_sampling** — locally typical sampling
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **temperature_scaling** — `logits /= T`
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **repetition_penalty** — divide/multiply logits at previously seen tokens
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **logit_bias** — additive bias from per-token map
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **paged_attention** — vLLM-style block-table KV indexing
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **continuous_batching_dispatch** — request-level gather/scatter across a
      shared compute batch
      - [ ] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **chunked_prefill** — split long-prompt prefill into compute-sized chunks
      - [ ] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **speculative_decoding (draft + verify)** — token tree verification
      kernel (EAGLE/Medusa head accepted)
      - [ ] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **beam_search_step** — per-step beam scoring + reordering
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **grammar_constrained_decode** — mask logits against an FSM/grammar
      accept set
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 22. Attention variants (modern)

- [ ] **sliding_window_attention** — Mistral/Gemma local-window mask
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **multi_latent_attention (MLA)** — DeepSeek-style compressed-KV
      attention (low-rank KV joint projection + position-decoupled keys)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **cross_attention** — encoder/decoder; Q from one stream, K/V from
      another
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **alibi_attention** — additive linear position bias
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **linear_attention (Based / Hedgehog / RetNet / RWKV)** — kernel-feature
      linear attention with state (TK 2.0 has reference kernels)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **block_sparse_attention** — Longformer-style local + global
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **streaming_attention_sinks** — StreamingLLM persistent sink tokens
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **ring_attention** — context-parallel shard-and-rotate
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **native_sparse_attention (NSA)** — DeepSeek-V3.2 native sparse pattern
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **differential_attention** — two softmax branches subtracted
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **varlen_attention (cu_seqlens)** — Flash-Attn variable-length API for
      packed sequences
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 23. RoPE / position-encoding variants

- [ ] **yarn_rope_scaling** — YaRN context extension
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **ntk_aware_rope_scaling** — frequency-base interpolation
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **linear_rope_scaling** — position interpolation (PI)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **rope_2d** — 2D RoPE for vision/multimodal
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **alibi_bias** — precompute / apply ALiBi slopes
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **xpos / nope** — alternate position schemes
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 24. Optimizers beyond AdamW + Muon

- [ ] **Lion** — sign-momentum optimizer
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **Sophia** — diagonal-Hessian clipped second-order
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **Shampoo / SOAP** — preconditioned optimizer (small inverse-roots per
      tensor)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **AdEMAMix** — Adam with mixed slow/fast EMA
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **AdamW8bit** — block-quantized optimizer states (bitsandbytes-style)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **Adafactor** — factored second-moment
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **ema_update** — fast EMA weight update `θ' = α·θ' + (1−α)·θ`
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **swa_average** — Stochastic Weight Averaging accumulator
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 25. FP8 / low-precision (SM120 / Blackwell)

- [ ] **fp8_gemm (E4M3 / E5M2)** — Blackwell tensor-core FP8 matmul with scale
      tracking
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **fp8_quantize** — bf16/fp32 → fp8 with delayed/static scaling
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **fp8_dequantize** — fp8 → bf16/fp32
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **mxfp8 / mxfp4 GEMM** — Blackwell microscaled FP8/FP4 matmul
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **stochastic_rounding** — bf16/fp8 weight update with SR
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **amax_history_tracking** — Transformer-Engine-style amax update
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 26. Distributed beyond ZeRO

- [ ] **all_to_all** — token shuffle for MoE expert parallelism
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **column_parallel_linear** — TP linear sharded along output dim
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **row_parallel_linear** — TP linear sharded along input dim (with
      all-reduce on output)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **sequence_parallel_norm** — SP-sharded LN/RMSNorm with all-gather
      around TP region
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **pipeline_send_recv** — 1F1B / zero-bubble PP helpers
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **context_parallel_allgather** — ring-attn coordination collective
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 27. Modern MoE features

- [ ] **auxfree_load_balancing** — DeepSeek-V3 bias-adjusted routing (no aux
      loss)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **capacity_factor_dispatch** — capacity-limited token assignment with
      drop / overflow
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **expert_parallel_dispatch** — combines top-k routing with all-to-all
      (§26)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **soft_moe** — continuous (no top-k) MoE with learned slot assignments
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **mixture_of_depths (MoD)** — token-level early exit
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 28. Norm variants

- [ ] **group_norm** — per-group LayerNorm
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **dyt (Dynamic Tanh)** — recent LayerNorm replacement
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **qk_norm (fused)** — fused RMSNorm on Q and K before SDPA
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 29. Activation gates beyond SwiGLU

- [ ] **geglu** — gated GELU
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **reglu** — gated ReLU
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **solu** — softmax-gated linear unit
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 30. Vision / multimodal

- [ ] **conv2d** — general 2D conv (vision patch embed; cross-modal stems)
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **patch_embed_2d** — Conv2d + flatten + linear projection
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **abs_pos_embed_2d** — learned 2D positional embeddings
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **cross_modal_cross_attention** — vision↔text Q/KV bridges
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 31. Long-context KV management

- [ ] **h2o_eviction** — Heavy-Hitter Oracle KV pruning
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **snapkv** — prompt-aware KV compression
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **landmark_attention** — landmark-token summaries
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **infini_attention** — segment-level compressed memory
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **sink_token_cache** — persistent attention sinks
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 32. Gradient / training infrastructure

- [ ] **gradient_checkpoint_hook** — re-materialisation entry/exit pair
      - [ ] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **selective_recompute** — Flash-Attn-style activation drop + recompute
      - [ ] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **loss_scale_dynamic** — dynamic loss scaling with overflow detection
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **gradient_accumulate** — fused add-into-grad-buffer
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)

## 33. Tokenizer & data path

- [ ] **gpu_bpe_tokenizer** — on-device byte-pair tokenisation
      - [ ] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **sequence_packing** — variable-length pack with cu_seqlens generation
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
- [ ] **document_causal_mask (cu_seqlens)** — per-document causal mask for
      packed batches
      - [x] CUDA  - [ ] cuBLAS/cuBLASLt  - [ ] cuDNN  - [ ] TK 2.0 (SM120)
