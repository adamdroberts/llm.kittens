/*
cublas_ext.cuh — cuBLAS / cuBLASLt back-ends for GEMM-shaped kernels.

The bf16 standard GEMM lives in matmul.cuh (TK kernel). This header declares
cuBLASLt-backed variants for cases where the vendor path is preferred:
  - shape-agnostic bf16 GEMM (no padding restrictions)
  - fp8 GEMM (E4M3 / E5M2) via the cuBLASLt FP8 path
  - mxfp8 / mxfp4 GEMM (Blackwell)
  - grouped/batched GEMM (MoE / multi-expert)
  - GEMM + activation epilogues using cublasLtMatmul's pointwise epilogue ops
  - W8A8 int8 GEMM via cuBLASLt's int8 path
  - W4A16 GEMM (uses NF4 dequant kernel into a scratch then standard bf16)
  - sampling helpers route through cuBLAS only when stacked as matmul (e.g.,
    cosine-sim batched topk used by routers)

Implementations live in a matching cublas_ext.cu (set up the cublasLt handle,
allocate matmul descriptors and call cublasLtMatmul). Declarations only here.
*/
#pragma once

#include "cuda_common.h"
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// ============================================================================
// Shape-agnostic bf16 GEMM via cuBLASLt: y = x @ w^T (+ bias) (+ activation)
// ============================================================================

enum class GemmEpilogue : int {
    None      = 0,
    Bias      = 1,
    Gelu      = 2,
    ReluSq    = 3,
    Sigmoid   = 4,
    Tanh      = 5,
    Silu      = 6,
    BiasGelu  = 7,
    BiasReluSq= 8,
    BiasSilu  = 9,
};

void cublaslt_gemm_bf16(__nv_bfloat16* y, const __nv_bfloat16* x, const __nv_bfloat16* w,
                        const __nv_bfloat16* bias /*nullable*/,
                        int M, int N, int K, GemmEpilogue epilogue, cudaStream_t stream);

// FP8 GEMM via the cuBLASLt FP8 path. Scales come from amax history (fp8.cuh).
void cublaslt_gemm_fp8(__nv_bfloat16* y, const uint8_t* x_e4m3, const uint8_t* w_e4m3,
                       const float* x_scale, const float* w_scale,
                       int M, int N, int K, cudaStream_t stream);

// MX FP8 / FP4 GEMM (Blackwell native).
void cublaslt_gemm_mxfp8(__nv_bfloat16* y, const uint8_t* x_data, const uint8_t* x_exps,
                         const uint8_t* w_data, const uint8_t* w_exps,
                         int M, int N, int K, cudaStream_t stream);
void cublaslt_gemm_mxfp4(__nv_bfloat16* y, const uint8_t* x_packed, const uint8_t* x_exps,
                         const uint8_t* w_packed, const uint8_t* w_exps,
                         int M, int N, int K, cudaStream_t stream);

// W8A8 int8 GEMM (Tensor Cores int8 path).
void cublaslt_gemm_w8a8(__nv_bfloat16* y, const int8_t* x_q, const int8_t* w_q,
                        const float* x_scale, float w_scale,
                        int M, int N, int K, cudaStream_t stream);

// Grouped / batched bf16 GEMM.
void cublaslt_gemm_batched_bf16(__nv_bfloat16* y, const __nv_bfloat16* x, const __nv_bfloat16* w,
                                int batch_count, int M, int N, int K, cudaStream_t stream);
void cublaslt_gemm_grouped_bf16(__nv_bfloat16* y, const __nv_bfloat16* x, const __nv_bfloat16* w_packed,
                                const int* expert_offsets, int E,
                                int M_total, int N, int K, cudaStream_t stream);

// W4A16 NF4 GEMM: dequantizes NF4 weight into bf16 scratch then runs bf16 GEMM.
void cublaslt_gemm_w4a16(__nv_bfloat16* y, const __nv_bfloat16* x,
                         const uint8_t* w_packed, const float* w_absmax,
                         int M, int N, int K, int group_size, cudaStream_t stream);

// Low-rank LoRA GEMM:  y = x @ base^T + scaling * (x @ A^T) @ B^T.
void cublaslt_gemm_lora(__nv_bfloat16* y, const __nv_bfloat16* x,
                        const __nv_bfloat16* base_w,
                        const __nv_bfloat16* A, const __nv_bfloat16* B,
                        float scaling, int M, int N, int K, int rank, cudaStream_t stream);
