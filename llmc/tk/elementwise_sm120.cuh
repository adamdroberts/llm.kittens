/*
elementwise_sm120.cuh — ThunderKittens vectorised elementwise kernels for SM120.

Covers all the pointwise activations + small fused ops (residual_mix,
qk_gain, logit_softcap, gate fusion, dyt). These use kittens packed-128
loads when D divides 8.
*/
#pragma once

#include "tk_common.cuh"
#include <cmath>

namespace llmk::elementwise_sm120 {

using namespace ::kittens;

#define LLMK_ACT_FWD_BS 256
#define LLMK_ACT_BWD_BS 128

// ============================================================================
// Single-input activations.
// ============================================================================

#define LLMK_TK_ACT_FWD(name, expr)                                                       \
__global__ void name##_fwd_kernel(bf16* out, const bf16* x, int N) {                       \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (i >= N) return;                                                                    \
    float xi = __bfloat162float(x[i]);                                                     \
    out[i] = __float2bfloat16(expr);                                                       \
}                                                                                          \
inline void launch_##name(bf16* out, const bf16* x, int N, cudaStream_t stream) {          \
    name##_fwd_kernel<<<CEIL_DIV(N, LLMK_ACT_FWD_BS), LLMK_ACT_FWD_BS, 0, stream>>>(out, x, N); \
    cudaCheck(cudaGetLastError());                                                         \
}

LLMK_TK_ACT_FWD(sigmoid,    1.0f / (1.0f + expf(-xi)))
LLMK_TK_ACT_FWD(tanh_act,   tanhf(xi))
LLMK_TK_ACT_FWD(relu,       xi > 0.f ? xi : 0.f)
LLMK_TK_ACT_FWD(relu6,      fminf(fmaxf(xi, 0.f), 6.f))
LLMK_TK_ACT_FWD(silu,       xi * (1.0f / (1.0f + expf(-xi))))
LLMK_TK_ACT_FWD(softplus,   log1pf(expf(xi)))
LLMK_TK_ACT_FWD(softsign,   xi / (1.0f + fabsf(xi)))
LLMK_TK_ACT_FWD(hardsigmoid,fmaxf(0.f, fminf(1.f, 0.2f*xi + 0.5f)))
LLMK_TK_ACT_FWD(hardtanh,   fmaxf(-1.f, fminf(1.f, xi)))
LLMK_TK_ACT_FWD(hardswish,  xi * fmaxf(0.f, fminf(1.f, (xi + 3.f) / 6.f)))
LLMK_TK_ACT_FWD(gaussian,   expf(-xi * xi))
LLMK_TK_ACT_FWD(log_act,    logf(xi))
LLMK_TK_ACT_FWD(negate,     -xi)
LLMK_TK_ACT_FWD(mish,       xi * tanhf(log1pf(expf(xi))))
LLMK_TK_ACT_FWD(elu,        xi >= 0.f ? xi : (expf(xi) - 1.f))
LLMK_TK_ACT_FWD(selu,       xi >= 0.f ? 1.0507f * xi : 1.0507f * 1.6733f * (expf(xi) - 1.f))

#undef LLMK_TK_ACT_FWD

// Parameterised: leaky_relu, threshold, prelu.
__global__ void leaky_relu_fwd_kernel(bf16* out, const bf16* x, float slope, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float xi = __bfloat162float(x[i]);
    out[i] = __float2bfloat16(xi > 0.f ? xi : slope * xi);
}
inline void launch_leaky_relu(bf16* out, const bf16* x, float slope, int N, cudaStream_t stream) {
    leaky_relu_fwd_kernel<<<CEIL_DIV(N, LLMK_ACT_FWD_BS), LLMK_ACT_FWD_BS, 0, stream>>>(out, x, slope, N);
    cudaCheck(cudaGetLastError());
}
__global__ void threshold_fwd_kernel(bf16* out, const bf16* x, float thresh, float value, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float xi = __bfloat162float(x[i]);
    out[i] = __float2bfloat16(xi > thresh ? xi : value);
}
inline void launch_threshold(bf16* out, const bf16* x, float thresh, float value, int N, cudaStream_t stream) {
    threshold_fwd_kernel<<<CEIL_DIV(N, LLMK_ACT_FWD_BS), LLMK_ACT_FWD_BS, 0, stream>>>(out, x, thresh, value, N);
    cudaCheck(cudaGetLastError());
}
__global__ void prelu_fwd_kernel(bf16* out, const bf16* x, const bf16* slope, int C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float xi = __bfloat162float(x[i]);
    int c = i % C;
    float a = __bfloat162float(slope[c]);
    out[i] = __float2bfloat16(xi > 0.f ? xi : a * xi);
}
inline void launch_prelu(bf16* out, const bf16* x, const bf16* slope, int N, int C, cudaStream_t stream) {
    prelu_fwd_kernel<<<CEIL_DIV(N, LLMK_ACT_FWD_BS), LLMK_ACT_FWD_BS, 0, stream>>>(out, x, slope, C, N);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Two-input elementwise: add, multiply.
// ============================================================================

__global__ void add_kernel(bf16* out, const bf16* a, const bf16* b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}
inline void launch_add(bf16* out, const bf16* a, const bf16* b, int N, cudaStream_t stream) {
    add_kernel<<<CEIL_DIV(N, LLMK_ACT_FWD_BS), LLMK_ACT_FWD_BS, 0, stream>>>(out, a, b, N);
    cudaCheck(cudaGetLastError());
}
__global__ void multiply_kernel(bf16* out, const bf16* a, const bf16* b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = __float2bfloat16(__bfloat162float(a[i]) * __bfloat162float(b[i]));
}
inline void launch_multiply(bf16* out, const bf16* a, const bf16* b, int N, cudaStream_t stream) {
    multiply_kernel<<<CEIL_DIV(N, LLMK_ACT_FWD_BS), LLMK_ACT_FWD_BS, 0, stream>>>(out, a, b, N);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Fused per-channel scale + add (residual_mix), per-head gain (qk_gain), softcap.
// ============================================================================

__global__ void residual_mix_kernel(bf16* out, const bf16* x, const bf16* x0,
                                    const float* alpha, const float* beta, int N, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int c = i % C;
    out[i] = __float2bfloat16(alpha[c] * __bfloat162float(x[i]) + beta[c] * __bfloat162float(x0[i]));
}
inline void launch_residual_mix(bf16* out, const bf16* x, const bf16* x0,
                                const float* alpha, const float* beta, int N, int C,
                                cudaStream_t stream) {
    residual_mix_kernel<<<CEIL_DIV(N, LLMK_ACT_FWD_BS), LLMK_ACT_FWD_BS, 0, stream>>>(
        out, x, x0, alpha, beta, N, C);
    cudaCheck(cudaGetLastError());
}

__global__ void qk_gain_kernel(bf16* out, const bf16* q, const float* gain, int HD, int head_dim, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int head = (i / head_dim) % (HD / head_dim);
    out[i] = __float2bfloat16(gain[head] * __bfloat162float(q[i]));
}
inline void launch_qk_gain(bf16* out, const bf16* q, const float* gain, int N, int HD, int head_dim,
                           cudaStream_t stream) {
    qk_gain_kernel<<<CEIL_DIV(N, LLMK_ACT_FWD_BS), LLMK_ACT_FWD_BS, 0, stream>>>(
        out, q, gain, HD, head_dim, N);
    cudaCheck(cudaGetLastError());
}

__global__ void logit_softcap_kernel(bf16* out, const bf16* x, float softcap, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float xi = __bfloat162float(x[i]);
    out[i] = __float2bfloat16(softcap * tanhf(xi / softcap));
}
inline void launch_logit_softcap(bf16* out, const bf16* x, int N, float softcap, cudaStream_t stream) {
    logit_softcap_kernel<<<CEIL_DIV(N, LLMK_ACT_FWD_BS), LLMK_ACT_FWD_BS, 0, stream>>>(out, x, softcap, N);
    cudaCheck(cudaGetLastError());
}

// Gates: GeGLU, ReGLU, SoLU (row-wise softmax-gated).
__global__ void geglu_kernel(bf16* out, const bf16* gate_out, const bf16* up_out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = __bfloat162float(gate_out[i]);
    float u = __bfloat162float(up_out[i]);
    const float scale = 0.7978845608f;
    float cube = 0.044715f * g * g * g;
    float gelu = 0.5f * g * (1.f + tanhf(scale * (g + cube)));
    out[i] = __float2bfloat16(gelu * u);
}
inline void launch_geglu(bf16* out, const bf16* gate_out, const bf16* up_out, int N, cudaStream_t stream) {
    geglu_kernel<<<CEIL_DIV(N, LLMK_ACT_FWD_BS), LLMK_ACT_FWD_BS, 0, stream>>>(out, gate_out, up_out, N);
    cudaCheck(cudaGetLastError());
}
__global__ void reglu_kernel(bf16* out, const bf16* gate_out, const bf16* up_out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = __bfloat162float(gate_out[i]);
    float u = __bfloat162float(up_out[i]);
    out[i] = __float2bfloat16((g > 0.f ? g : 0.f) * u);
}
inline void launch_reglu(bf16* out, const bf16* gate_out, const bf16* up_out, int N, cudaStream_t stream) {
    reglu_kernel<<<CEIL_DIV(N, LLMK_ACT_FWD_BS), LLMK_ACT_FWD_BS, 0, stream>>>(out, gate_out, up_out, N);
    cudaCheck(cudaGetLastError());
}
__global__ void solu_kernel(bf16* out, const bf16* h, int D, int rows) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const bf16* hr = h + row * D;
    bf16* or_ = out + row * D;
    float local_max = -INFINITY;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = __bfloat162float(hr[d]);
        if (v > local_max) local_max = v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, local_max, off);
        if (other > local_max) local_max = other;
    }
    float row_max = __shfl_sync(0xFFFFFFFF, local_max, 0);
    float local_sum = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) local_sum += expf(__bfloat162float(hr[d]) - row_max);
    for (int off = 16; off > 0; off >>= 1) local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, off);
    float sumexp = __shfl_sync(0xFFFFFFFF, local_sum, 0);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float s = expf(__bfloat162float(hr[d]) - row_max) / sumexp;
        or_[d] = __float2bfloat16(__bfloat162float(hr[d]) * s);
    }
}
inline void launch_solu(bf16* out, const bf16* h, int D, int rows, cudaStream_t stream) {
    solu_kernel<<<rows, ::kittens::WARP_THREADS, 0, stream>>>(out, h, D, rows);
    cudaCheck(cudaGetLastError());
}

// Dynamic Tanh (DyT): out = α · tanh(β · x) + γ
__global__ void dyt_kernel(bf16* out, const bf16* x, const bf16* alpha, const bf16* beta, const bf16* gamma,
                           int N, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int c = i % C;
    float v = tanhf(__bfloat162float(beta[c]) * __bfloat162float(x[i]));
    v = __bfloat162float(alpha[c]) * v;
    if (gamma) v += __bfloat162float(gamma[c]);
    out[i] = __float2bfloat16(v);
}
inline void launch_dyt(bf16* out, const bf16* x, const bf16* alpha, const bf16* beta, const bf16* gamma,
                       int N, int C, cudaStream_t stream) {
    dyt_kernel<<<CEIL_DIV(N, LLMK_ACT_FWD_BS), LLMK_ACT_FWD_BS, 0, stream>>>(out, x, alpha, beta, gamma, N, C);
    cudaCheck(cudaGetLastError());
}

// Softmax / log-softmax (last dim).
__global__ void softmax_kernel(bf16* out, const bf16* x, int W) {
    int row = blockIdx.x;
    const bf16* xr = x + row * W;
    bf16* or_ = out + row * W;
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < W; i += blockDim.x) {
        float v = __bfloat162float(xr[i]);
        if (v > local_max) local_max = v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, local_max, off);
        if (other > local_max) local_max = other;
    }
    float row_max = __shfl_sync(0xFFFFFFFF, local_max, 0);
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < W; i += blockDim.x) local_sum += expf(__bfloat162float(xr[i]) - row_max);
    for (int off = 16; off > 0; off >>= 1) local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, off);
    float sumexp = __shfl_sync(0xFFFFFFFF, local_sum, 0);
    for (int i = threadIdx.x; i < W; i += blockDim.x) {
        or_[i] = __float2bfloat16(expf(__bfloat162float(xr[i]) - row_max) / sumexp);
    }
}
inline void launch_softmax(bf16* out, const bf16* x, int rows, int W, cudaStream_t stream) {
    softmax_kernel<<<rows, ::kittens::WARP_THREADS, 0, stream>>>(out, x, W);
    cudaCheck(cudaGetLastError());
}
__global__ void log_softmax_kernel(bf16* out, const bf16* x, int W) {
    int row = blockIdx.x;
    const bf16* xr = x + row * W;
    bf16* or_ = out + row * W;
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < W; i += blockDim.x) {
        float v = __bfloat162float(xr[i]);
        if (v > local_max) local_max = v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, local_max, off);
        if (other > local_max) local_max = other;
    }
    float row_max = __shfl_sync(0xFFFFFFFF, local_max, 0);
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < W; i += blockDim.x) local_sum += expf(__bfloat162float(xr[i]) - row_max);
    for (int off = 16; off > 0; off >>= 1) local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, off);
    float sumexp = __shfl_sync(0xFFFFFFFF, local_sum, 0);
    float log_z = row_max + logf(sumexp);
    for (int i = threadIdx.x; i < W; i += blockDim.x) {
        or_[i] = __float2bfloat16(__bfloat162float(xr[i]) - log_z);
    }
}
inline void launch_log_softmax(bf16* out, const bf16* x, int rows, int W, cudaStream_t stream) {
    log_softmax_kernel<<<rows, ::kittens::WARP_THREADS, 0, stream>>>(out, x, W);
    cudaCheck(cudaGetLastError());
}

}  // namespace llmk::elementwise_sm120
