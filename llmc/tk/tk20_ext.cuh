/*
tk20_ext.cuh — ThunderKittens 2.0 (SM120) integration points for the wider
kernel surface.

Existing TK kernels in `llmc/tk/`:
  - attention_sm120.cuh   bf16 causal attention
  - attention_h100.cuh    H100 reference
  - attention_gqa_h100.cuh, attention_gqa_sm120 (NOT YET — declared as future)
  - gemm_sm120.cuh        bf16 GEMM
  - layernorm_tk.cuh, rmsnorm_tk.cuh, rope_tk.cuh

This header declares the remaining TK 2.0 entry points to be implemented in
follow-on files under `llmc/tk/`. The kernel surface listed here mirrors
nfn-coverage-todo.md TK 2.0 column.

Each function delegates to a kernel under `llmc/tk/*_sm120.cuh` once present.
*/
#pragma once

#include "cuda_common.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// ============================================================================
// Forward declarations only — TK 2.0 SM120 kernels live in llmc/tk/*_sm120.cuh
// and are written using the ThunderKittens DSL (../../../ThunderKittens/include).
// ============================================================================

// --- GEMM ---
void tk20_gemm_bf16(__nv_bfloat16* y, const __nv_bfloat16* x, const __nv_bfloat16* w,
                    int M, int N, int K, cudaStream_t stream);
void tk20_gemm_bf16_with_activation(__nv_bfloat16* y, const __nv_bfloat16* x, const __nv_bfloat16* w,
                                    int M, int N, int K, int activation_id, cudaStream_t stream);
void tk20_gemm_fp8_e4m3(__nv_bfloat16* y, const uint8_t* x, const uint8_t* w,
                        const float* x_scale, const float* w_scale,
                        int M, int N, int K, cudaStream_t stream);
void tk20_gemm_mxfp8(__nv_bfloat16* y, const uint8_t* x_data, const uint8_t* x_exps,
                     const uint8_t* w_data, const uint8_t* w_exps,
                     int M, int N, int K, cudaStream_t stream);
void tk20_gemm_grouped(__nv_bfloat16* y, const __nv_bfloat16* x, const __nv_bfloat16* w,
                       const int* expert_offsets, int E, int M_total, int N, int K, cudaStream_t stream);

// --- Attention ---
void tk20_attention_causal(__nv_bfloat16* out, const __nv_bfloat16* q,
                           const __nv_bfloat16* k, const __nv_bfloat16* v,
                           int B, int H, int S, int D, cudaStream_t stream);
void tk20_attention_gqa(__nv_bfloat16* out, const __nv_bfloat16* q,
                        const __nv_bfloat16* k, const __nv_bfloat16* v,
                        int B, int H_q, int H_k, int S, int D, cudaStream_t stream);
void tk20_attention_non_causal(__nv_bfloat16* out, const __nv_bfloat16* q,
                               const __nv_bfloat16* k, const __nv_bfloat16* v,
                               int B, int H, int S_q, int S_k, int D, cudaStream_t stream);
void tk20_attention_sliding_window(__nv_bfloat16* out, const __nv_bfloat16* q,
                                   const __nv_bfloat16* k, const __nv_bfloat16* v,
                                   int B, int H, int S, int D, int window, cudaStream_t stream);
void tk20_attention_alibi(__nv_bfloat16* out, const __nv_bfloat16* q,
                          const __nv_bfloat16* k, const __nv_bfloat16* v, const float* slopes,
                          int B, int H, int S, int D, cudaStream_t stream);
void tk20_attention_mla(__nv_bfloat16* out, const __nv_bfloat16* q,
                        const __nv_bfloat16* kv_compressed, const __nv_bfloat16* kv_up,
                        int B, int H, int S, int D, int comp_dim, cudaStream_t stream);
void tk20_attention_paged(__nv_bfloat16* out, const __nv_bfloat16* q,
                          const __nv_bfloat16* pages_k, const __nv_bfloat16* pages_v,
                          const int* block_table, const int* cache_len,
                          int B, int H, int Hk, int D, int page_size, int max_blocks,
                          cudaStream_t stream);
void tk20_attention_varlen(__nv_bfloat16* out, const __nv_bfloat16* q,
                           const __nv_bfloat16* k, const __nv_bfloat16* v,
                           const int* cu_seqlens, int H, int total_tokens, int D,
                           cudaStream_t stream);
void tk20_attention_native_sparse(__nv_bfloat16* out, const __nv_bfloat16* q,
                                  const __nv_bfloat16* k, const __nv_bfloat16* v,
                                  const int* sparse_indices,
                                  int B, int H, int S_q, int K_per_q, int D,
                                  cudaStream_t stream);
void tk20_attention_differential(__nv_bfloat16* out,
                                 const __nv_bfloat16* q1, const __nv_bfloat16* k1, const __nv_bfloat16* v1,
                                 const __nv_bfloat16* q2, const __nv_bfloat16* k2, const __nv_bfloat16* v2,
                                 float lambda, int B, int H, int S, int D, cudaStream_t stream);

// --- Linear / based / hedgehog (linear attention) ---
void tk20_linear_attention(__nv_bfloat16* out, const __nv_bfloat16* q,
                           const __nv_bfloat16* k, const __nv_bfloat16* v,
                           int B, int H, int S, int D, int feature_dim, cudaStream_t stream);

// --- SSM scan ---
void tk20_selective_scan(__nv_bfloat16* y, __nv_bfloat16* h_out,
                         const __nv_bfloat16* x, const __nv_bfloat16* delta,
                         const __nv_bfloat16* A, const __nv_bfloat16* B, const __nv_bfloat16* C,
                         const __nv_bfloat16* D,
                         int B_batch, int S, int d_inner, int d_state, cudaStream_t stream);

// --- Norms ---
void tk20_layernorm(__nv_bfloat16* y, float* mean, float* rstd,
                    const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                    int N, int C, float eps, cudaStream_t stream);
void tk20_rmsnorm(__nv_bfloat16* y, float* rstd,
                  const __nv_bfloat16* x, const __nv_bfloat16* weight,
                  int N, int C, float eps, cudaStream_t stream);
void tk20_groupnorm(__nv_bfloat16* y, float* mean, float* rstd,
                    const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                    int B, int C, int S, int groups, float eps, cudaStream_t stream);
void tk20_qk_norm(__nv_bfloat16* q_out, __nv_bfloat16* k_out,
                  const __nv_bfloat16* q, const __nv_bfloat16* k,
                  int B, int H_q, int H_k, int S, int D, float eps, cudaStream_t stream);

// --- RoPE ---
void tk20_rope_apply(__nv_bfloat16* q_out, __nv_bfloat16* k_out,
                     const __nv_bfloat16* q, const __nv_bfloat16* k,
                     const __nv_bfloat16* cos, const __nv_bfloat16* sin,
                     int B, int H_q, int H_k, int S, int D, cudaStream_t stream);
void tk20_rope_2d_apply(__nv_bfloat16* q_out, __nv_bfloat16* k_out,
                        const __nv_bfloat16* q, const __nv_bfloat16* k,
                        const __nv_bfloat16* cos2d, const __nv_bfloat16* sin2d,
                        int B, int H, int seq, int D, cudaStream_t stream);

// --- MoE permute + scatter ---
void tk20_moe_permute(__nv_bfloat16* permuted, int* sort_order, int* expert_offsets,
                      const __nv_bfloat16* tokens, const int* topk_indices,
                      int rows, int K, int dim, int E, cudaStream_t stream);
void tk20_moe_unpermute(__nv_bfloat16* out, const __nv_bfloat16* permuted_out,
                        const int* sort_order, const __nv_bfloat16* topk_weights,
                        int rows, int K, int dim, cudaStream_t stream);

// --- Long-context: paged kv, sinks, varlen ---
void tk20_kv_cache_append(__nv_bfloat16* cache_k, __nv_bfloat16* cache_v,
                          const __nv_bfloat16* current_k, const __nv_bfloat16* current_v,
                          int* cache_len, int B, int Hk, int S_cap, int S_new, int D,
                          cudaStream_t stream);
void tk20_kv_quant_pack(__nv_bfloat16* packed,
                        const __nv_bfloat16* k, const __nv_bfloat16* v, int rows, int D,
                        cudaStream_t stream);

// --- Activations (TK 2.0 only worth using when fused into a bigger kernel;
// standalone activations stay in activations.cuh). Provided for completeness ---
void tk20_act_geglu(__nv_bfloat16* out, const __nv_bfloat16* gate_out, const __nv_bfloat16* up_out,
                    int N, cudaStream_t stream);
void tk20_act_swiglu(__nv_bfloat16* out, const __nv_bfloat16* gate_out, const __nv_bfloat16* up_out,
                    int N, cudaStream_t stream);

// --- Losses (fused with the last GEMM ideally) ---
void tk20_fused_classifier_masked(__nv_bfloat16* dlogits, float* losses,
                                  const __nv_bfloat16* logits, const int* targets,
                                  const __nv_bfloat16* loss_mask, int rows, int vocab,
                                  cudaStream_t stream);
