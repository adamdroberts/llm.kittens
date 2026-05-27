/*
norms_ext.cuh — norm and gate variants beyond LayerNorm + RMSNorm + SwiGLU.

  - group_norm           per-group LayerNorm
  - dynamic_tanh (DyT)   `α · tanh(β · x) + γ`
  - qk_norm (fused)      RMSNorm over the last (head_dim) axis of Q and K
  - geglu, reglu, solu   gated activation variants
*/
#pragma once

#include <assert.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ============================================================================
// GroupNorm:
//   x:    [B, C, S]
//   out:  [B, C, S]
//   weight, bias: [C]    (affine, optional — pass nullptr to skip)
//   groups: divides C
//
// For each (b, g), compute mean / var over (C_per_group, S) and normalise.
// ============================================================================

__global__ void group_norm_forward_kernel(floatX* out, float* mean_out, float* rstd_out,
                                          const floatX* x, const floatX* weight, const floatX* bias,
                                          int B, int C, int S, int groups, float eps) {
    int b = blockIdx.y;
    int g = blockIdx.x;
    int C_per_group = C / groups;
    int N = C_per_group * S;

    // 1) mean
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        int c_local = i / S;
        int s = i % S;
        int c = g * C_per_group + c_local;
        local_sum += (float)x[(b * C + c) * S + s];
    }
    float sum = blockReduce<warpReduceSum>(local_sum);
    float mean = sum / (float)N;
    if (threadIdx.x == 0) mean_out[b * groups + g] = mean;

    // 2) var
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        int c_local = i / S;
        int s = i % S;
        int c = g * C_per_group + c_local;
        float v = (float)x[(b * C + c) * S + s] - mean;
        local_var += v * v;
    }
    float var = blockReduce<warpReduceSum>(local_var, true) / (float)N;
    float rstd = rsqrtf(var + eps);
    if (threadIdx.x == 0) rstd_out[b * groups + g] = rstd;

    // 3) normalise + affine
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        int c_local = i / S;
        int s = i % S;
        int c = g * C_per_group + c_local;
        float v = ((float)x[(b * C + c) * S + s] - mean) * rstd;
        if (weight) v *= (float)weight[c];
        if (bias)   v += (float)bias[c];
        out[(b * C + c) * S + s] = (floatX)v;
    }
}

void group_norm_forward(floatX* out, float* mean, float* rstd,
                        const floatX* x, const floatX* weight, const floatX* bias,
                        int B, int C, int S, int groups, float eps, cudaStream_t stream) {
    NVTX_RANGE_FN();
    dim3 grid(groups, B);
    group_norm_forward_kernel<<<grid, 128, 0, stream>>>(out, mean, rstd, x, weight, bias, B, C, S, groups, eps);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Dynamic Tanh (DyT):  out = α · tanh(β · x) + γ
//   α, β, γ: per-channel parameters of shape [C]   (γ optional)
// ============================================================================

__global__ void dyt_forward_kernel(floatX* out, const floatX* x,
                                   const floatX* alpha, const floatX* beta, const floatX* gamma,
                                   int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) return;
    int c = idx % C;
    float v = tanhf((float)beta[c] * (float)x[idx]);
    v = (float)alpha[c] * v;
    if (gamma) v += (float)gamma[c];
    out[idx] = (floatX)v;
}
void dyt_forward(floatX* out, const floatX* x,
                 const floatX* alpha, const floatX* beta, const floatX* gamma,
                 int N, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    int total = N * C;
    dyt_forward_kernel<<<CEIL_DIV(total, block_size), block_size, 0, stream>>>(out, x, alpha, beta, gamma, N, C);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// qk_norm (fused): RMSNorm applied to Q and K along their last axis (head_dim).
//
//   q, k: [B, H_q, S, D] / [B, H_k, S, D]
// ============================================================================

__global__ void qk_rmsnorm_kernel(floatX* out, const floatX* x, int D, int rows, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const floatX* xr = x + row * D;
    floatX* outr = out + row * D;
    // sum of squares
    float local = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = (float)xr[d];
        local += v * v;
    }
    float ss = blockReduce<warpReduceSum>(local, true);
    float rstd = rsqrtf(ss / (float)D + eps);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        outr[d] = (floatX)((float)xr[d] * rstd);
    }
}

void qk_norm_fused(floatX* q_out, floatX* k_out, const floatX* q, const floatX* k,
                   int B, int H_q, int H_k, int S, int D, float eps, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int rows_q = B * H_q * S;
    int rows_k = B * H_k * S;
    qk_rmsnorm_kernel<<<rows_q, 64, 0, stream>>>(q_out, q, D, rows_q, eps);
    cudaCheck(cudaGetLastError());
    qk_rmsnorm_kernel<<<rows_k, 64, 0, stream>>>(k_out, k, D, rows_k, eps);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// GeGLU / ReGLU / SoLU forward
//   x: input [B, S, model_dim]
//   gate, up: weights [hidden, model_dim]  applied separately
//
// In NeuralFn the gated MLP is two linears + activation:
//   h = act(gate(x)) * up(x);  out = w2(h)
// We provide the elementwise pointwise fuser for the (gate, up) → activated
// product step; the GEMMs go through matmul.cuh.
// ============================================================================

__global__ void geglu_pointwise_kernel(floatX* out, const floatX* gate_out, const floatX* up_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float g = (float)gate_out[idx];
    float u = (float)up_out[idx];
    // GELU approximation (same constants as gelu.cuh)
    const float scale = 0.7978845608028654f; // sqrt(2/pi)
    float cube = 0.044715f * g * g * g;
    float gelu = 0.5f * g * (1.0f + tanhf(scale * (g + cube)));
    out[idx] = (floatX)(gelu * u);
}
void geglu_pointwise(floatX* out, const floatX* gate_out, const floatX* up_out, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    geglu_pointwise_kernel<<<CEIL_DIV(N, block_size), block_size, 0, stream>>>(out, gate_out, up_out, N);
    cudaCheck(cudaGetLastError());
}

__global__ void reglu_pointwise_kernel(floatX* out, const floatX* gate_out, const floatX* up_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float g = (float)gate_out[idx];
    float u = (float)up_out[idx];
    out[idx] = (floatX)((g > 0.0f ? g : 0.0f) * u);
}
void reglu_pointwise(floatX* out, const floatX* gate_out, const floatX* up_out, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    reglu_pointwise_kernel<<<CEIL_DIV(N, block_size), block_size, 0, stream>>>(out, gate_out, up_out, N);
    cudaCheck(cudaGetLastError());
}

// SoLU: a softmax-gated linear unit. For a vector h of length D in each row:
//   solu(h) = h * softmax(h)
// Row-wise.
__global__ void solu_pointwise_kernel(floatX* out, const floatX* h, int D, int rows) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const floatX* hr = h + row * D;
    floatX* or_ = out + row * D;
    float local_max = -INFINITY;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = (float)hr[d];
        if (v > local_max) local_max = v;
    }
    float row_max = blockReduce<warpReduceMax>(local_max);
    float local_sumexp = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        local_sumexp += expf((float)hr[d] - row_max);
    }
    float sumexp = blockReduce<warpReduceSum>(local_sumexp, true);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float s = expf((float)hr[d] - row_max) / sumexp;
        or_[d] = (floatX)((float)hr[d] * s);
    }
}
void solu_pointwise(floatX* out, const floatX* h, int D, int rows, cudaStream_t stream) {
    NVTX_RANGE_FN();
    solu_pointwise_kernel<<<rows, 128, 0, stream>>>(out, h, D, rows);
    cudaCheck(cudaGetLastError());
}
