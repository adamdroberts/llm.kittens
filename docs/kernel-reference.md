# Kernel reference

Every operator in the training loop, where its implementation lives, and what
state it is in. Rows are listed in roughly the order they fire during a
forward pass.

The principle: **TK earns its keep on tile-MMA ops** (GEMM, MHA, LayerNorm).
Element-wise / gather-scatter / 1D-reduction kernels are kept verbatim from
`llm.c` — TK adds nothing there.

Hopper (`SM90`) uses the optimized TK H100 wrappers listed below. Blackwell
(`SM100`, `SM103`, `SM120`) is build-supported through ThunderKittens 2.0; the
Hopper-only GQA/RoPE wrappers use plain CUDA correctness fallbacks until
dedicated B200/GB200 kernels are ported. SM120 additionally has a cuBLASLt GEMM
fallback for GPT-2 and a custom warp-scope GPT MHA path.

## Mapping table

| Op | Wrapper | TK template | llm.c source kept identical | Status |
|---|---|---|---|---|
| Token + position embedding | [`llmc/encoder.cuh`](../llmc/encoder.cuh) | — | `llm.c/llmc/encoder.cuh:157,169` | ✅ verbatim |
| Matmul forward | [`llmc/matmul.cuh`](../llmc/matmul.cuh) | [`llmc/tk/gemm_h100.cuh`](../llmc/tk/gemm_h100.cuh) | signature parallels `llm.c/llmc/matmul.cuh:231` | ✅ TK-backed |
| Matmul backward (2 GEMMs + bias-grad reduce) | [`llmc/matmul.cuh`](../llmc/matmul.cuh) | [`llmc/tk/gemm_h100.cuh`](../llmc/tk/gemm_h100.cuh) | `llm.c/llmc/matmul.cuh:17,83,244` | ✅ M3 compile-wired — TK `A*B` dInp, TK `A^T*B` dWeight, verbatim bias reduction |
| Bias-grad reduction (`matmul_backward_bias_kernel9`, `reduce_add_sum_kernel`) | `llmc/matmul.cuh` | — | verbatim from `llm.c/llmc/matmul.cuh:17,83` | ✅ verbatim |
| Bias-add (default forward) | `llmc/matmul.cuh::add_bias_kernel` | — | new (cuBLASLt epilogue replacement) | ✅ |
| Bias+GELU epilogue (opt-in GPT-2 MLP up-projection) | `llmc/matmul.cuh::matmul_forward_gelu` | [`llmc/tk/gemm_h100.cuh`](../llmc/tk/gemm_h100.cuh) | parallels llm.c's fused GELU aux path | 🟡 compile-wired behind `-ge 1`; H100 runtime pending |
| Attention forward (MHA) | [`llmc/attention.cuh`](../llmc/attention.cuh) | [`llmc/tk/attention_h100.cuh`](../llmc/tk/attention_h100.cuh) | `llm.c/llmc/attention.cuh:195` | ✅ M2 compile-wired — TK fast path for `T % 192 == 0`; padded TK path for other lengths |
| Attention backward (MHA) | [`llmc/attention.cuh`](../llmc/attention.cuh) | [`llmc/tk/attention_h100.cuh`](../llmc/tk/attention_h100.cuh) | `llm.c/llmc/attention.cuh:239` | ✅ M3 compile-wired — TK `bwd_attend_prep_ker` / `bwd_attend_ker` fast path; slow CUDA fallback for unsupported shapes; runtime parity pending |
| QKV permute / unpermute | [`llmc/attention.cuh`](../llmc/attention.cuh) | — | verbatim from `llm.c/llmc/attention.cuh:14-83` | ✅ verbatim; bypassed by SM120 packed-QKV fast path |
| Attention forward (GQA + RoPE, Llama-3) | [`llmc/attention_gqa.cuh`](../llmc/attention_gqa.cuh) | [`llmc/tk/attention_gqa_h100.cuh`](../llmc/tk/attention_gqa_h100.cuh) | none — new kernel | 🟡 M6 — TK causal forward for supported H100 shapes; Q/K tile-load RoPE for supported shapes; CUDA fallback |
| Attention backward (GQA + RoPE, Llama-3) | [`llmc/attention_gqa.cuh`](../llmc/attention_gqa.cuh) | [`llmc/tk/attention_gqa_h100.cuh`](../llmc/tk/attention_gqa_h100.cuh) | none — new kernel | 🟡 M6 — TK supported-shape backward compile-wired with tile-load RoPE; CUDA fallback; runtime pending |
| LayerNorm forward | [`llmc/layernorm.cuh`](../llmc/layernorm.cuh) | [`llmc/tk/layernorm_tk.cuh`](../llmc/tk/layernorm_tk.cuh) | `llm.c/llmc/layernorm.cuh:433` | ✅ M2 compile-wired; `test_layernorm` compile-ready — TK fork for supported widths, CUDA fallback retained |
| LayerNorm backward | [`llmc/layernorm.cuh`](../llmc/layernorm.cuh) | TK `kittens::warp::sum` helper inside the hand-written CUDA accumulator | `llm.c/llmc/layernorm.cuh:233` (`layernorm_backward_kernel10`) | ✅ M3 compile-wired; `test_layernorm` compile-ready; runtime parity pending |
| Fused-residual + LN forward | [`llmc/layernorm.cuh::fused_residual_forward5`](../llmc/layernorm.cuh) | [`llmc/tk/layernorm_tk.cuh`](../llmc/tk/layernorm_tk.cuh) | `llm.c/llmc/layernorm.cuh:467` | ✅ M2 compile-wired; `test_layernorm` compile-ready — TK fused residual+LN for supported widths, CUDA fallback retained |
| RMSNorm forward / backward (Llama-3) | [`llmc/rmsnorm.cuh`](../llmc/rmsnorm.cuh) | [`llmc/tk/rmsnorm_tk.cuh`](../llmc/tk/rmsnorm_tk.cuh) | none in TK; mirrors layernorm minus mean and bias | ✅ M6 compile-wired; `test_rmsnorm` compile-ready; runtime pending |
| RoPE forward / backward (Llama-3) | [`llmc/rope.cuh`](../llmc/rope.cuh) | [`llmc/tk/rope_tk.cuh`](../llmc/tk/rope_tk.cuh) | wrap `ThunderKittens/kernels/rotary/rotary.cu`; backward = inverse rotation with `sin -> -sin` | ✅ M6 compile-wired; `test_rope` compile-ready; runtime pending |
| GELU forward / backward-inplace | [`llmc/gelu.cuh`](../llmc/gelu.cuh) | — | `llm.c/llmc/gelu.cuh:50,59` | ✅ verbatim |
| SwiGLU (Llama-3) | [`llmc/swiglu.cuh`](../llmc/swiglu.cuh) | — | none — new, plain CUDA elementwise fwd/bwd | ✅ M6 compile-wired; `test_swiglu` compile-ready; runtime pending |
| Cross-entropy + softmax + dlogits (fused) | [`llmc/fused_classifier.cuh`](../llmc/fused_classifier.cuh) | — | `llm.c/llmc/fused_classifier.cuh:140` | ✅ verbatim |
| AdamW step | [`llmc/adamw.cuh`](../llmc/adamw.cuh) | — | `llm.c/llmc/adamw.cuh:75,91` | ✅ verbatim |
| `init_from_master` | `llmc/adamw.cuh` | — | verbatim | ✅ |
| `global_norm_squared` (grad-clip norm) | [`llmc/global_norm.cuh`](../llmc/global_norm.cuh) | — | `llm.c/llmc/global_norm.cuh:69` | ✅ verbatim |
| ZeRO-0/1/2/3 + NCCL init (MPI/TCP/FS) | [`llmc/zero.cuh`](../llmc/zero.cuh) | — | based on `llm.c/llmc/zero.cuh` | ✅ ZeRO-0/1/2 wired; ZeRO-3 parameter-shard runtime path compile-wired; H100/NCCL runtime pending |
| Distributed dataloaders | [`llmc/dataloader.h`](../llmc/dataloader.h) | — | adapted from llm.c | ✅ GPT-2 uint16 + Llama-3 uint32 training/eval files dispatch by header magic |
| Tokenizer / sampler / schedulers / rand / mfu / outlier_detector / logger / utils | `llmc/{tokenizer,sampler,schedulers,rand,mfu,outlier_detector,logger,utils}.h` | — | verbatim | ✅ |

Legend: ✅ done · ⬜ not started · 🟡 partial.

## TK GEMM (`llmc/tk/gemm_h100.cuh`, `llmc/matmul.cuh`)

Two specialisations are exposed:

```cpp
namespace llmk::gemm {
    using matmul_default = matmul_template<2, 4, 8>;   // M_BLOCK=2, N_BLOCK=4, SUPER_M=8
    using matmul_small_n = matmul_template<2, 2, 8>;   // M_BLOCK=2, N_BLOCK=2, SUPER_M=8
    using matmul_default_nt_bias_gelu = matmul_template<2, 4, 8, false, true, true, true, true>;
}
```

The SM120-specific wrapper uses the same template surface but tunes its
Blackwell cp.async kernels separately: the shared `LLMK_SM120_SUPER_M` swizzle
defaults to `9` after RTX 5090 3-step validation. dInput uses a separate
`LLMK_SM120_DINP_SUPER_M=8` default, while dWeight keeps its separate
`LLMK_SM120_DWEIGHT_SUPER_M=2` default and routes supported TN dWeight shapes
through a 128x128 tile by default (`LLMK_SM120_DWEIGHT_N128=1`).

`matmul_template<M_BLOCK, N_BLOCK, SUPER_M, A_TRANSPOSED, B_TRANSPOSED,
APPLY_BIAS, APPLY_GELU, STORE_PRE_GELU>` is ported from
`ThunderKittens/kernels/gemm/bf16_h100/bf16_h100_gemm.cu` (lines 1-106). It uses
the LCF (Load-Compute-Finish) prototype with a persistent grid (132 SMs, the
H100 SM count), TMA async loads on the producer warpgroup, and WGMMA on the
consumer warpgroup. `B_TRANSPOSED=true` switches the consumer to `mma_ABt` and
loads B as row-major `(N, K)`, matching llm.c's dense weight layout `(OC, C)`.
`A_TRANSPOSED=true` switches the consumer to `mma_AtB` for dWeight. The opt-in
bias+GELU aliases add bias and GELU in the finish path after WGMMA and can TMA
store a pre-GELU auxiliary buffer for backward.

On Blackwell builds this Hopper WGMMA wrapper is not instantiated; the C-style
wrapper dispatches to a plain CUDA BF16 correctness kernel instead.

### Shape constraints

| Symbol | Constraint | Why |
|---|---|---|
| `M = B*T` | `M % (64 * M_BLOCK) == 0` (so `M % 128 == 0` at default) | TK MMA tile geometry |
| `N = OC` | `N % (64 * N_BLOCK) == 0` (256 default, 128 for `small_n`) | as above |
| `K = C` | `K % 64 == 0` | TK base tile is 64×64 |

`matmul_forward` `assert`s these at runtime. `OC % 256 == 0` selects the default
`A*B^T` specialization; `OC % 128 == 0` selects the small-N `A*B^T` fallback.
The GPT-2 124M LM head with `V_padded = 50304` is the canonical small-N case.

### Forward signature

```cpp
inline void matmul_forward(floatX* out, const floatX* inp, const floatX* weight,
                           const floatX* bias, int B, int T, int C, int OC,
                           cudaStream_t stream);
```

`out (B*T, OC) = inp (B*T, C) · weight(OC, C)^T + bias (OC)`.

The default path keeps bias and GELU separate: bias is added in
`add_bias_kernel`, and the trainer calls `gelu_forward` when needed. GPT-2's
MLP up-projection can opt into `matmul_forward_gelu(out, pre_gelu, ...)`, which
uses the TK finish path to apply bias, store the pre-GELU buffer, and write GELU
output. The trainer exposes this behind `-ge 1`, but the default remains `-ge 0`
on non-SM120 builds until H100 numerical validation and profiling are done.
SM120 builds default to `-ge 1` after RTX 5090 3-step validation showed it
faster than explicit CUDA GELU on the current pure-TK path.
The SM120 pure-TK forward dispatcher also enables `LLMK_SM120_FORWARD_N96=1`
by default so GPT-2 widths divisible by 96 use the 128x96 tile instead of the
older 256x64/128x64 choices; `LLMK_SM120_FORWARD_N96=0` remains an A/B escape
hatch. The GPT-2 LM-head huge-N forward route defaults to a 256x128 tile
(`LLMK_SM120_HUGE_N_M256=1`) after current RTX 5090 validation improved the
source-default pure-TK 3-step run; `LLMK_SM120_HUGE_N_M256=0` keeps the older
128x128 tile available for A/B tests. The huge-N/N128 K tile is now
`LLMK_SM120_HUGE_N_K_TILE=16` after the current dWeight N128 route made the
smaller K tile's dWeight gains outweigh the LM-head forward slowdown in the
3-step validation. The shared SM120
forward/dInput/huge-N tile swizzle now uses `LLMK_SM120_SUPER_M=9` by default;
the old `8` and `10` values remain useful A/B references.

### Backward signature (M3)

```cpp
inline void matmul_backward(floatX* dinp, floatX* dweight, floatX* dbias,
                            const floatX* dout, const floatX* inp, const floatX* weight,
                            float* dbias_buffer,
                            int B, int T, int C, int OC, cudaStream_t stream,
                            bool dweight_accumulate = true,
                            floatX* dweight_accum_scratch = nullptr,
                            size_t dweight_accum_scratch_elements = 0);
```

Current status:

1. `dinp (B*T, C) = dout (B*T, OC) · weight(OC, C)` — wired through the existing TK `A*B` GEMM.
   SM120 dInput uses `LLMK_SM120_DINP_SUPER_M=8` after a focused A/B on top
   of the K-tile 16 stack improved current 3-step timing.
   On SM120 pure-TK builds, `LLMK_SM120_FUSE_DGELU=1` is now the default when
   the trainer uses `-ge 1`, fusing the MLP GELU backward into the `fcproj`
   dInput GEMM. The fused SM120 path uses in-place register-layout swaps and
   an approximate dGELU tanh (`LLMK_SM120_APPROX_DGELU_TANH=1`) by default
   after RTX 5090 smoke and 3-step validation; both knobs remain explicit A/B
   fallbacks.
2. `dweight (OC, C) = doutᵀ (OC, B*T) · inp (B*T, C)` — wired through TK `A^T*B`. For accumulated micro-steps, the product lands in the caller-provided aligned scratch buffer and a small add kernel applies `dweight += scratch`. SM120 pure-TK defaults to `LLMK_SM120_DWEIGHT_SPLIT_K=8` after the reduced LM-head scratch regime made the lower qkv split faster in 3-step training; larger non-QKV dWeight shapes remain capped at 8-way split-K inside the wrapper because larger splits regressed them.
   Supported SM120 TN dWeight shapes now use the 128x128 tile
   (`LLMK_SM120_DWEIGHT_N128=1`) after the current pure-TK stack improved the
   TinyStories 3-step run below the supplied llm.c baseline. The trainer keeps
   qkv split-K scratch at the normal `LLMK_SM120_DWEIGHT_SPLIT_K` depth but
   defaults LM-head-sized dWeight scratch to one part
   (`LLMK_SM120_LARGE_DWEIGHT_SPLIT_K=1`); larger LM-head scratch fanout
   increased activation residency enough to dominate the pure-TK step time.
   The SM120 TN
   swizzle now defaults to `LLMK_SM120_DWEIGHT_SUPER_M=2` after the K-tile 16
   route made it faster in 3-step validation; `1` fails smoke, while `4` and
   higher tested values were slower or mixed. Eligible split-K dWeight rows are
   now started on their nonblocking part streams before the same
   `matmul_backward()` call launches dInput and bias-grad on the main stream,
   then reduced after those independent kernels are enqueued; this overlaps
   independent backward GEMM and reduction work
   without changing the external wrapper contract.
3. `dbias (OC) = column-sum of dout over B*T` — `matmul_backward_bias_kernel9` followed by `reduce_add_sum_kernel` when `dbias_buffer` is available. Both kernels are verbatim from llm.c. SM120 keeps the same kernels but uses a 512-thread launch block by default (`LLMK_SM120_BIAS_BLOCK_SIZE`) after RTX 5090 timing showed it faster than the H100-derived 768-thread choice.

The slow CUDA dWeight kernel remains only as a fallback for unsupported TK shapes
or missing scratch capacity.

## TK MHA (`llmc/attention.cuh`, `llmc/tk/attention_h100.cuh`, `llmc/tk/attention_sm120.cuh`) — M2/M3

Wrap `fwd_attend_ker`, `bwd_attend_prep_ker`, and `bwd_attend_ker` from
`ThunderKittens/kernels/attention/mha_h100/mha_h100.cu` on H100. SM120 uses
the local warp-scope FlashAttention-2 style implementation in
`llmc/tk/attention_sm120.cuh`.

On Blackwell builds the Hopper WGMMA/TMA attention kernels are not
instantiated. GPT-style SM120 attention dispatches to the custom warp-scope TK
forward path for `HS ∈ {64, 128}` and the custom TK backward path for `HS=64`;
unsupported backward shapes fall back to the existing recompute CUDA baseline.

### Constraints

- `head_dim ∈ {64, 128}` only. GPT-2 uses 64; Llama-3 uses 128. Anything else
  needs a new TK specialization. Confirmed adequate for everything in scope.
- The copied TK forward launch covers sequence lengths divisible by
  `3 * 16 * 4 = 192`. `llmc/attention.cuh` pads non-aligned sequence lengths
  into scratch, runs TK at `Tpad`, then unpads directly to the normal output.
- The copied TK backward launch covers sequence lengths divisible by
  `4 * 16 * 4 = 256`. GPT-2 `T=1024` is covered. Other backward shapes fall
  back to the slow CUDA recompute baseline until padded backward dispatch lands.
- The SM120 path uses independently tuned tile sizes: 32 rows for forward and
  16 rows for backward by default. `LLMK_SM120_ATTN_FWD_BLOCK`,
  `LLMK_SM120_ATTN_BWD_BLOCK`, or the shorthand `LLMK_SM120_ATTN_BLOCK` can be
  used for controlled experiments; 64-row tiles are correct but slower on RTX
  5090 GPT-2 training.
- SM120 attention backward prep uses a 3-warp CUDA launch
  (`LLMK_SM120_DPREP_WARPS=3`) after RTX 5090 validation found it faster than
  the earlier 4-warp default and the rejected 2-warp variant.
- BF16 only (same as the rest of v1).

### Layout glue

Generic and H100 GPT attention keep llm.c's `permute_kernel` and
`unpermute_kernel` (`llm.c/llmc/attention.cuh:14-83`) to reshape between the
QKV layout `(B, T, 3, NH, HS)` produced by the QKV matmul and the
`(B, NH, T, HS)` layout the TK MHA kernel expects. The SM120 GPT-2 trainer fast
path instead stores the QKV projection directly in the saved packed-QKV
activation slot and calls `attention_forward_packed_qkv` /
`attention_backward_packed_qkv`. That path loads packed Q/K/V directly, writes
the forward output as `(B, T, C)`, and writes packed dQ/dK/dV directly into the
QKV input-gradient buffer. Builds that disable SM120 TK backward or enable the
atomic-dQ experiment fall back to the generic permuted layout so the forward
and backward saved layouts stay consistent.

### Signatures

```cpp
inline void attention_forward(floatX* out, floatX* qkvr, floatX* att,
                              floatX* inp,
                              int B, int T, int C, int NH,
                              cudaStream_t stream);
inline void attention_backward(floatX* dinp, floatX* dqkvr,
                               floatX* datt, floatX* scratch,
                               const floatX* dout, const floatX* qkvr, const floatX* att,
                               int B, int T, int C, int NH,
                               cudaStream_t stream);
inline void attention_forward_packed_qkv(floatX* out, floatX* att,
                                         const floatX* qkv_packed,
                                         int B, int T, int C, int NH,
                                         cudaStream_t stream);
inline void attention_backward_packed_qkv(floatX* dinp, floatX* datt,
                                          const floatX* scratch, const floatX* dout,
                                          const floatX* qkv_packed, const floatX* att,
                                          int B, int T, int C, int NH,
                                          cudaStream_t stream);
```

`attention_forward` and `attention_backward` match
`llm.c/llmc/attention.cuh:195,239`. The packed-QKV signatures are SM120-only
GPT-2 trainer fast paths.

The M3 backward fast path reads the forward-saved LSE from `att`, permutes the
saved forward output from `scratch`, permutes `dout`, calls TK
`bwd_attend_prep_ker` and `bwd_attend_ker`, converts float Q/K/V gradients back
to bf16, then reuses the verbatim `permute_kernel_backward` glue. On SM120 the
forward-output/dout permutation and `D = rowsum(dO * O)` prep are fused into a
small CUDA helper before the TK backward launch. The default SM120 path also
uses the packed-QKV backward launcher, so the TK kernel reads packed Q/K/V and
writes packed bf16 gradients directly without the final
`permute_kernel_backward` glue. Unsupported head dimensions, sequence lengths,
or fallback build modes use the slow recompute baseline or the generic permuted
TK path.

Validation plan: [`dev/cuda/test_attention.cu`](../dev/cuda/test_attention.cu)
compile-wires an independent CPU-reference smoke harness. It covers direct TK
forward plus fallback backward at `T=192`, and padded TK forward plus
supported-shape TK backward at `T=256`, plus an SM120-only packed-QKV forward
and backward case at `T=256`. Runtime numerical validation still needs a
compatible H100 driver/runtime for the Hopper wrappers.

## TK LayerNorm (`llmc/layernorm.cuh`, `llmc/tk/layernorm_tk.cuh`) — M2/M3

### Forward (M2)

Fork `ThunderKittens/kernels/layernorm/layernorm.cu`. Three changes from the
upstream TK kernel:

1. Re-template the hidden width `D` over GPT-2 / Llama-3 widths
   `{768, 1024, 1280, 1600, 2048, 4096}`. Pre-instantiate all six.
2. Strip the dropout path. We never use it.
3. Remove the internal `cudaDeviceSynchronize` and add a `cudaStream_t` parameter.

Current implementation: `llmc/tk/layernorm_tk.cuh` is wired into
`llmc/layernorm.cuh` for the supported widths and saves `mean` / `rstd` for the
existing GPT-2 backward path. Runtime numerical validation is still gated on an
accessible H100 plus the GPT-2 debug-state files.

[`dev/cuda/test_layernorm.cu`](../dev/cuda/test_layernorm.cu) compile-wires a
CPU-reference smoke harness for forward, fused residual+LayerNorm forward,
saved `mean`/`rstd`, and backward `+=` accumulation into `dinp`, `dweight`, and
`dbias`.

### `fused_residual_forward5`

Match `llm.c/llmc/layernorm.cuh:467` — fuses `residual = x + skip; out = LN(residual)`
into one pass.

### Backward (M3)

TK does not ship a layernorm backward. Port `layernorm_backward_kernel10` from
`llm.c/llmc/layernorm.cuh:233` onto TK primitives:

- Replace warp-loop sums with `kittens::warp::sum`.
- Replace warp-loop row-max-style reductions with `kittens::warp::row_max` (where applicable).
- Keep the cross-block atomic-counter accumulator pattern unchanged.

This is the second-highest-risk kernel in the project (after Llama-3 GQA/RoPE) —
the partial-sum accumulator has subtle dependencies on which threads write
which counters.

## Llama-3 kernels — M6

Llama-3 introduces four new operators. Three are TK-backed; one is plain CUDA.

### RMSNorm (`llmc/rmsnorm.cuh`, `llmc/tk/rmsnorm_tk.cuh`)

Mirrors layernorm_tk minus the mean subtraction and bias term. Forward and
fused-residual forward dispatch through the TK fork for widths
`{768, 1024, 1280, 1600, 2048, 4096}`. Backward is a plain CUDA correctness
baseline that computes `dinp` per row and `dweight` per column.

Signatures:

```cpp
void rmsnorm_forward(floatX* out, float* rstd,
                     const floatX* inp, const floatX* weight,
                     int N, int C, float eps, cudaStream_t stream);
void fused_residual_rmsnorm_forward(floatX* residual, floatX* normed, float* rstd,
                                    const floatX* inp1, const floatX* inp2,
                                    const floatX* weight,
                                    int N, int C, float eps, cudaStream_t stream);
void rmsnorm_backward(floatX* dinp, floatX* dweight,
                      const floatX* dout, const floatX* inp,
                      const floatX* weight, const float* rstd,
                      int N, int C, cudaStream_t stream);
```

Compile-check: `nvcc -gencode arch=compute_90a,code=sm_90a -x cu -c
llmc/rmsnorm.cuh`. `dev/cuda/test_rmsnorm.cu` adds a CPU-reference smoke
harness for forward, fused-residual forward, `rstd`, `dinp`, and `dweight`.
Runtime numerical validation waits for a compatible H100 driver/runtime.

### RoPE (`llmc/rope.cuh`, `llmc/tk/rope_tk.cuh`)

Wrap `ThunderKittens/kernels/rotary/rotary.cu`. Backward is the same forward
op with `(sin, -cos)` swapped — no separate kernel needed.

On Blackwell builds the wrapper uses a plain CUDA RoPE fallback because the
current TK LCSF RoPE fork is Hopper-shaped.

`dev/cuda/test_rope.cu` adds a CPU-reference smoke harness for HS=64 and
HS=128 forward plus inverse-rotation backward. Runtime execution waits for a
compatible H100 driver/runtime.

### GQA + RoPE attention (`llmc/attention_gqa.cuh`, `llmc/tk/attention_gqa_h100.cuh`)

**Highest-risk kernel.** No full TK reference. The current wrapper permutes
packed Q/K/V and dispatches TK causal forward plus supported-shape TK backward
paths for H100 BF16 shapes. Where both TK forward and TK backward are available,
Q/K stay unrotated in the saved workspace and the TK kernels apply RoPE after
tile load before WGMMA. Unsupported shapes still use the slow plain-CUDA
correctness baseline with fused materialization/unpermute.

The adapted MHA template now covers:

- Accept `n_q_heads`, `n_kv_heads`, `n_rep = n_q / n_kv`.
- Forward: repeat KV across query groups inside the kernel (landed for supported shapes; no host-side replication).
- Backward: accumulate `dK`/`dV` into KV-head gradient buffers across the `n_rep` Q heads (landed for supported shapes; H100 numerical validation pending).
Remaining gate: runtime validation against the CPU/PyTorch references on H100.

Validation plan: `dev/cuda/test_attention_gqa.cu` compile-wires CPU-reference
smoke cases at B=1, head_dim=128. `T=128` covers TK forward plus CUDA fallback
backward with the materialized fallback path; `T=256` covers the supported-shape
TK backward path with tile-load RoPE. `dev/validate_attention_gqa_reference.py`
checks the CPU-only PyTorch equivalence for those same shapes by comparing
materialized-RoPE repeated-KV attention with grouped/tile-load-style RoPE,
including backward gradients into packed Q/K/V. The remaining gate is the H100
TK runtime comparison, exposed as `scripts/validate_goal_h100.sh gqa-runtime`.

### SwiGLU (`llmc/swiglu.cuh`)

Plain CUDA. Forward computes `out = silu(gate) * up`; backward writes `dgate`
and `dup` from the saved gate/up tensors. No TK benefit.
`dev/cuda/test_swiglu.cu` adds a CPU-reference smoke harness for forward,
`dgate`, and `dup`.

## Plain-CUDA kernels (kept verbatim)

These are imported from llm.c with no changes beyond `#include`s. The comment
header in each file points to the upstream source.

| File | Purpose | Notes |
|---|---|---|
| [`llmc/encoder.cuh`](../llmc/encoder.cuh) | Token + position embedding fwd/bwd | Sparse gather + bucketed deterministic scatter — wrong workload for TK |
| [`llmc/gelu.cuh`](../llmc/gelu.cuh) | GELU fwd + in-place backward | TK's flux GELU is fused with matmul, not standalone |
| [`llmc/fused_classifier.cuh`](../llmc/fused_classifier.cuh) | Cross-entropy + softmax + dlogits, fused | Block-wide online softmax over `V = 50304` (GPT-2) or `128256` (Llama-3). Optional v2 TK-ification. |
| [`llmc/adamw.cuh`](../llmc/adamw.cuh) | AdamW step + `init_from_master` | Element-wise |
| [`llmc/global_norm.cuh`](../llmc/global_norm.cuh) | Grad-clip norm | Element-wise reduction |
| [`llmc/zero.cuh`](../llmc/zero.cuh) | NCCL + ZeRO-0/1/2/3 + multi-node init; ZeRO-3 parameter shards all-gather into the full compute layout | NCCL operates on opaque buffers — compatible with TK kernels |
| [`llmc/dataloader.h`](../llmc/dataloader.h) | Distributed train/eval loaders | Dispatches GPT-2 uint16 and Llama-3 uint32 files on header magic |
| [`llmc/tokenizer.h`](../llmc/tokenizer.h) | GPT-2 BPE detokenizer | Used by sampling |
| [`llmc/sampler.h`](../llmc/sampler.h) | Top-k / top-p sampler | |
| [`llmc/schedulers.h`](../llmc/schedulers.h) | Cosine + linear LR schedules | |
| [`llmc/rand.h`](../llmc/rand.h) | Stateless RNG (Mersenne + Philox) | |
| [`llmc/mfu.h`](../llmc/mfu.h) | Model-FLOPs-utilization meter | |
| [`llmc/outlier_detector.h`](../llmc/outlier_detector.h) | Loss-outlier detection | |
| [`llmc/logger.h`](../llmc/logger.h) | Append-only training log | |
| [`llmc/utils.h`](../llmc/utils.h) | `fopenCheck`, `freadCheck`, `mallocCheck`, etc. | |

`llmc/cublas_common.h` is a stub kept only for symbol-name compatibility — no
cuBLAS / cuBLASLt symbol is referenced in v1.
