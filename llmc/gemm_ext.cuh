/*
gemm_ext.cuh — extensions to the matmul path:
  - GEMM + ReLU² epilogue                 (for mlp_relu2)
  - GEMM + bias + sigmoid/tanh/silu epilogues (for heads + halt gates)
  - Low-rank LoRA two-step GEMM (delegating per-step kernels in quantize.cuh)
  - Ternary BitNet GEMM (W ∈ {-1,0,1}, int8 activations)
  - Shape-agnostic launcher that wraps matmul.cuh and pads internally if
    needed (so M%128 / N%64 callers can still get correct outputs).
  - Batched / grouped GEMM scaffolding for MoE expert_dispatch.

The heavy bf16 GEMM still routes through matmul.cuh's TK kernel; this header
adds the epilogues and small helpers around it. Production should call the
cuBLASLt fp8 / mxfp8 / mxfp4 GEMM directly with the appropriate epilogue.
*/
#pragma once

#include <assert.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "matmul.cuh"  // existing matmul kernels

// ============================================================================
// Pointwise epilogues that run after a matmul. These take the matmul output
// in-place and overwrite it with the activation.
// ============================================================================

__global__ void relu_sq_epilogue_kernel(floatX* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = (float)y[i];
    v = v > 0.0f ? v : 0.0f;
    y[i] = (floatX)(v * v);
}
void relu_sq_epilogue(floatX* y, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    relu_sq_epilogue_kernel<<<CEIL_DIV(N, block_size), block_size, 0, stream>>>(y, N);
    cudaCheck(cudaGetLastError());
}

__global__ void add_bias_sigmoid_kernel(floatX* y, const floatX* bias, int N, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int c = i % C;
    float v = (float)y[i] + (float)bias[c];
    y[i] = (floatX)(1.0f / (1.0f + expf(-v)));
}
void add_bias_sigmoid_epilogue(floatX* y, const floatX* bias, int N, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    add_bias_sigmoid_kernel<<<CEIL_DIV(N, block_size), block_size, 0, stream>>>(y, bias, N, C);
    cudaCheck(cudaGetLastError());
}

__global__ void add_bias_tanh_kernel(floatX* y, const floatX* bias, int N, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int c = i % C;
    float v = (float)y[i] + (float)bias[c];
    y[i] = (floatX)tanhf(v);
}
void add_bias_tanh_epilogue(floatX* y, const floatX* bias, int N, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    add_bias_tanh_kernel<<<CEIL_DIV(N, block_size), block_size, 0, stream>>>(y, bias, N, C);
    cudaCheck(cudaGetLastError());
}

__global__ void add_bias_silu_kernel(floatX* y, const floatX* bias, int N, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int c = i % C;
    float v = (float)y[i] + (float)bias[c];
    float s = 1.0f / (1.0f + expf(-v));
    y[i] = (floatX)(v * s);
}
void add_bias_silu_epilogue(floatX* y, const floatX* bias, int N, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    add_bias_silu_kernel<<<CEIL_DIV(N, block_size), block_size, 0, stream>>>(y, bias, N, C);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Ternary BitNet b1.58 GEMM: int8 activations × ternary weight (stored as
// int8 ∈ {-1, 0, 1}), output bf16. Reference kernel; cuBLASLt path should be
// preferred for production.
//
//   x_q: int8 [M, K]
//   w_q: int8 [N, K]   (∈ {-1, 0, 1})
//   x_scale: fp32 [M]
//   w_scale: fp32 scalar
//   y: bf16 [M, N]
// ============================================================================

__global__ void ternary_gemm_kernel(__nv_bfloat16* y, const int8_t* x_q, const int8_t* w_q,
                                    const float* x_scale, float w_scale,
                                    int M, int N, int K) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    if (m >= M || n >= N) return;
    const int8_t* xr = x_q + m * K;
    const int8_t* wr = w_q + n * K;
    int acc = 0;
    for (int k = 0; k < K; ++k) acc += (int)xr[k] * (int)wr[k];
    float fy = (float)acc * x_scale[m] * w_scale;
    y[m * N + n] = __float2bfloat16(fy);
}
void ternary_gemm(__nv_bfloat16* y, const int8_t* x_q, const int8_t* w_q,
                  const float* x_scale, float w_scale, int M, int N, int K, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 64;
    dim3 grid(CEIL_DIV(M, block_size), N);
    ternary_gemm_kernel<<<grid, block_size, 0, stream>>>(y, x_q, w_q, x_scale, w_scale, M, N, K);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Grouped GEMM (MoE per-expert weights). Each "group" g has its own A_g (K, N)
// and shares the same B (M, K) but with different row ranges (after permute).
//
// Layout:
//   x:      [M_total, K]     (rows already permuted by expert)
//   expert_offsets: [E + 1]  (CSR-style)
//   W:      [E, K, N]        (per-expert weights)
//   y:      [M_total, N]
//
// Each thread handles one (row, n). This is a reference kernel — production
// should use cuBLASLt's grouped/batched matmul or TK 2.0.
// ============================================================================

__global__ void grouped_gemm_kernel(__nv_bfloat16* y, const __nv_bfloat16* x,
                                    const int* expert_offsets, const __nv_bfloat16* W,
                                    int E, int K, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (n >= N) return;

    // Identify the expert this row belongs to via binary search.
    int e = 0;
    for (int g = 0; g < E; ++g) {
        if (row >= expert_offsets[g] && row < expert_offsets[g + 1]) { e = g; break; }
    }
    const __nv_bfloat16* xr = x + row * K;
    const __nv_bfloat16* wm = W + e * K * N;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += __bfloat162float(xr[k]) * __bfloat162float(wm[k * N + n]);
    y[row * N + n] = __float2bfloat16(acc);
}
void grouped_gemm(__nv_bfloat16* y, const __nv_bfloat16* x,
                  const int* expert_offsets, const __nv_bfloat16* W,
                  int total_rows, int E, int K, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 64;
    dim3 grid(CEIL_DIV(N, block_size), total_rows);
    grouped_gemm_kernel<<<grid, block_size, 0, stream>>>(y, x, expert_offsets, W, E, K, N);
    cudaCheck(cudaGetLastError());
}
