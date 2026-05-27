/*
activations.cuh — bulk pointwise activation kernels.

Each activation has a forward (out = f(x)) and an in-place backward
(d_in_out *= f'(x)) launcher. Templated entry points share the same Packed128
load/store pattern as gelu.cuh so the bandwidth pattern is identical.

Layout per activation:

    void <name>_forward(floatX* out, const floatX* inp, int N, cudaStream_t);
    void <name>_backward_inplace(floatX* d_in_out, const floatX* inp,
                                 int N, cudaStream_t);

Some activations (relu, leaky_relu, hard_tanh, threshold) admit a "needs out
only" backward path; for uniformity we still pass `inp` and recompute.

Backward for prelu/elu/selu/threshold take their hyperparameter (alpha, slope)
as an extra argument; the launcher mirrors that. For the parameter-free ones
the backward only needs `inp`.

Two-argument elementwise ops:

    void add_forward(floatX* out, const floatX* a, const floatX* b, int N, ...);
    void multiply_forward(floatX* out, const floatX* a, const floatX* b, int N, ...);
    void negate_forward(floatX* out, const floatX* inp, int N, ...);

Reduction-style activations (softmax / log_softmax / softmax_2 / logit_softcap)
live below the elementwise group.

residual_mix and qk_gain (channel-/head-wise scaled add/multiply) are also
here since they're elementwise with broadcast.
*/
#pragma once

#include <assert.h>
#include <math.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ============================================================================
// Activation device-side ops (forward and derivative)
// ============================================================================

namespace llmk_act {

__device__ inline float act_sigmoid_fwd(float x)   { return 1.0f / (1.0f + expf(-x)); }
__device__ inline float act_sigmoid_grad(float x)  { float s = act_sigmoid_fwd(x); return s * (1.0f - s); }

__device__ inline float act_tanh_fwd(float x)      { return tanhf(x); }
__device__ inline float act_tanh_grad(float x)     { float t = tanhf(x); return 1.0f - t * t; }

__device__ inline float act_relu_fwd(float x)      { return x > 0.0f ? x : 0.0f; }
__device__ inline float act_relu_grad(float x)     { return x > 0.0f ? 1.0f : 0.0f; }

__device__ inline float act_leaky_relu_fwd(float x, float slope = 0.01f)  { return x > 0.0f ? x : slope * x; }
__device__ inline float act_leaky_relu_grad(float x, float slope = 0.01f) { return x > 0.0f ? 1.0f : slope; }

// PReLU uses a learnable slope (per-channel typically); reuse leaky_relu form.
__device__ inline float act_prelu_fwd(float x, float slope)  { return x > 0.0f ? x : slope * x; }
__device__ inline float act_prelu_grad_input(float x, float slope) { return x > 0.0f ? 1.0f : slope; }
__device__ inline float act_prelu_grad_slope(float x, float dout)  { return x > 0.0f ? 0.0f : x * dout; }

__device__ inline float act_relu6_fwd(float x)      { return fminf(fmaxf(x, 0.0f), 6.0f); }
__device__ inline float act_relu6_grad(float x)     { return (x > 0.0f && x < 6.0f) ? 1.0f : 0.0f; }

__device__ inline float act_elu_fwd(float x, float alpha = 1.0f)  {
    return x >= 0.0f ? x : alpha * (expf(x) - 1.0f);
}
__device__ inline float act_elu_grad(float x, float alpha = 1.0f) {
    return x >= 0.0f ? 1.0f : alpha * expf(x);
}

// SELU constants
#define SELU_ALPHA 1.6732632423543772848170429916717f
#define SELU_SCALE 1.0507009873554804934193349852946f
__device__ inline float act_selu_fwd(float x)  {
    return x >= 0.0f ? SELU_SCALE * x : SELU_SCALE * SELU_ALPHA * (expf(x) - 1.0f);
}
__device__ inline float act_selu_grad(float x) {
    return x >= 0.0f ? SELU_SCALE : SELU_SCALE * SELU_ALPHA * expf(x);
}

__device__ inline float act_silu_fwd(float x)  { float s = act_sigmoid_fwd(x); return x * s; }
__device__ inline float act_silu_grad(float x) {
    float s = act_sigmoid_fwd(x);
    return s + x * s * (1.0f - s);
}

__device__ inline float act_mish_fwd(float x)  {
    float sp = log1pf(expf(x));    // softplus
    return x * tanhf(sp);
}
__device__ inline float act_mish_grad(float x) {
    float sp  = log1pf(expf(x));
    float t   = tanhf(sp);
    float sig = act_sigmoid_fwd(x);
    return t + x * sig * (1.0f - t * t);
}

__device__ inline float act_softplus_fwd(float x)  { return log1pf(expf(x)); }
__device__ inline float act_softplus_grad(float x) { return act_sigmoid_fwd(x); }

__device__ inline float act_softsign_fwd(float x)  { return x / (1.0f + fabsf(x)); }
__device__ inline float act_softsign_grad(float x) { float d = 1.0f + fabsf(x); return 1.0f / (d * d); }

__device__ inline float act_hard_sigmoid_fwd(float x)  { return fmaxf(0.0f, fminf(1.0f, 0.2f * x + 0.5f)); }
__device__ inline float act_hard_sigmoid_grad(float x) { return (x > -2.5f && x < 2.5f) ? 0.2f : 0.0f; }

__device__ inline float act_hard_tanh_fwd(float x)  { return fmaxf(-1.0f, fminf(1.0f, x)); }
__device__ inline float act_hard_tanh_grad(float x) { return (x > -1.0f && x < 1.0f) ? 1.0f : 0.0f; }

__device__ inline float act_hard_swish_fwd(float x)  {
    return x * fmaxf(0.0f, fminf(1.0f, (x + 3.0f) / 6.0f));
}
__device__ inline float act_hard_swish_grad(float x) {
    if (x <= -3.0f) return 0.0f;
    if (x >=  3.0f) return 1.0f;
    return (2.0f * x + 3.0f) / 6.0f;
}

__device__ inline float act_gaussian_fwd(float x)  { return expf(-x * x); }
__device__ inline float act_gaussian_grad(float x) { return -2.0f * x * expf(-x * x); }

__device__ inline float act_log_fwd(float x)  { return logf(x); }
__device__ inline float act_log_grad(float x) { return 1.0f / x; }

__device__ inline float act_negate_fwd(float x)  { return -x; }
__device__ inline float act_negate_grad(float x) { return -1.0f; }

__device__ inline float act_threshold_fwd(float x, float thresh = 0.0f, float value = 0.0f) {
    return x > thresh ? x : value;
}
__device__ inline float act_threshold_grad(float x, float thresh = 0.0f) {
    return x > thresh ? 1.0f : 0.0f;
}

__device__ inline float act_logit_softcap_fwd(float x, float softcap) {
    return softcap * tanhf(x / softcap);
}
__device__ inline float act_logit_softcap_grad(float x, float softcap) {
    float t = tanhf(x / softcap);
    return 1.0f - t * t;
}

} // namespace llmk_act

// ============================================================================
// Generic dispatch macros: emit fwd + bwd kernels for a single-argument activation.
// ============================================================================

#define LLMK_ACT_DEFINE_FWD(NAME, FWD_EXPR)                                                       \
__global__ void NAME##_forward_kernel(floatX* out, const floatX* inp) {                            \
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;                                \
    x128 packed_out;                                                                               \
    x128 packed_inp = load128cs(inp + idx);                                                        \
    for (int k = 0; k < packed_inp.size; ++k) {                                                    \
        float xi = (float)packed_inp[k];                                                           \
        packed_out[k] = (floatX)(FWD_EXPR);                                                        \
    }                                                                                              \
    store128(out + idx, packed_out);                                                               \
}                                                                                                  \
void NAME##_forward(floatX* out, const floatX* inp, int N, cudaStream_t stream) {                  \
    NVTX_RANGE_FN();                                                                               \
    const int block_size = 512;                                                                    \
    assert(N % (block_size * x128::size) == 0);                                                    \
    const int grid_size = CEIL_DIV(N, block_size * x128::size);                                    \
    NAME##_forward_kernel<<<grid_size, block_size, 0, stream>>>(out, inp);                         \
    cudaCheck(cudaGetLastError());                                                                 \
}

#define LLMK_ACT_DEFINE_BWD(NAME, GRAD_EXPR)                                                       \
__global__ void NAME##_backward_inplace_kernel(floatX* d_in_out, const floatX* inp) {              \
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;                                \
    x128 packed_dinp;                                                                              \
    x128 packed_inp  = load128cs(inp + idx);                                                       \
    x128 packed_dout = load128(d_in_out + idx);                                                    \
    for (int k = 0; k < packed_inp.size; ++k) {                                                    \
        float xi = (float)packed_inp[k];                                                           \
        float grad = (GRAD_EXPR);                                                                  \
        packed_dinp[k] = (floatX)(grad * (float)packed_dout[k]);                                   \
    }                                                                                              \
    store128(d_in_out + idx, packed_dinp);                                                         \
}                                                                                                  \
void NAME##_backward_inplace(floatX* d_in_out, const floatX* inp, int N, cudaStream_t stream) {    \
    NVTX_RANGE_FN();                                                                               \
    const int block_size = 128;                                                                    \
    assert(N % (block_size * x128::size) == 0);                                                    \
    const int grid_size = CEIL_DIV(N, block_size * x128::size);                                    \
    NAME##_backward_inplace_kernel<<<grid_size, block_size, 0, stream>>>(d_in_out, inp);           \
    cudaCheck(cudaGetLastError());                                                                 \
}

#define LLMK_ACT_DEFINE(NAME, FWD_EXPR, GRAD_EXPR)                                                 \
    LLMK_ACT_DEFINE_FWD(NAME, FWD_EXPR)                                                            \
    LLMK_ACT_DEFINE_BWD(NAME, GRAD_EXPR)

// ============================================================================
// Standard single-argument activations (forward + backward)
// ============================================================================

LLMK_ACT_DEFINE(act_sigmoid,    llmk_act::act_sigmoid_fwd(xi),    llmk_act::act_sigmoid_grad(xi))
LLMK_ACT_DEFINE(act_tanh,       llmk_act::act_tanh_fwd(xi),       llmk_act::act_tanh_grad(xi))
LLMK_ACT_DEFINE(act_relu,       llmk_act::act_relu_fwd(xi),       llmk_act::act_relu_grad(xi))
LLMK_ACT_DEFINE(act_relu6,      llmk_act::act_relu6_fwd(xi),      llmk_act::act_relu6_grad(xi))
LLMK_ACT_DEFINE(act_selu,       llmk_act::act_selu_fwd(xi),       llmk_act::act_selu_grad(xi))
LLMK_ACT_DEFINE(act_silu,       llmk_act::act_silu_fwd(xi),       llmk_act::act_silu_grad(xi))
LLMK_ACT_DEFINE(act_mish,       llmk_act::act_mish_fwd(xi),       llmk_act::act_mish_grad(xi))
LLMK_ACT_DEFINE(act_softplus,   llmk_act::act_softplus_fwd(xi),   llmk_act::act_softplus_grad(xi))
LLMK_ACT_DEFINE(act_softsign,   llmk_act::act_softsign_fwd(xi),   llmk_act::act_softsign_grad(xi))
LLMK_ACT_DEFINE(act_hardsigmoid,llmk_act::act_hard_sigmoid_fwd(xi),llmk_act::act_hard_sigmoid_grad(xi))
LLMK_ACT_DEFINE(act_hardtanh,   llmk_act::act_hard_tanh_fwd(xi),  llmk_act::act_hard_tanh_grad(xi))
LLMK_ACT_DEFINE(act_hardswish,  llmk_act::act_hard_swish_fwd(xi), llmk_act::act_hard_swish_grad(xi))
LLMK_ACT_DEFINE(act_gaussian,   llmk_act::act_gaussian_fwd(xi),   llmk_act::act_gaussian_grad(xi))
LLMK_ACT_DEFINE(act_log,        llmk_act::act_log_fwd(xi),        llmk_act::act_log_grad(xi))
LLMK_ACT_DEFINE(act_negate,     llmk_act::act_negate_fwd(xi),     llmk_act::act_negate_grad(xi))

// ============================================================================
// Parameterised activations: leaky_relu(slope), elu(alpha), threshold(thresh,val).
// PReLU is identical to leaky_relu at forward; backward also writes a per-slope
// gradient and is grouped with elementwise ops.
// ============================================================================

__global__ void act_leaky_relu_forward_kernel(floatX* out, const floatX* inp, float slope) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    x128 packed_out;
    x128 packed_inp = load128cs(inp + idx);
    for (int k = 0; k < packed_inp.size; ++k) {
        float xi = (float)packed_inp[k];
        packed_out[k] = (floatX)llmk_act::act_leaky_relu_fwd(xi, slope);
    }
    store128(out + idx, packed_out);
}
void act_leaky_relu_forward(floatX* out, const floatX* inp, int N, float slope, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 512;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    act_leaky_relu_forward_kernel<<<grid_size, block_size, 0, stream>>>(out, inp, slope);
    cudaCheck(cudaGetLastError());
}
__global__ void act_leaky_relu_backward_inplace_kernel(floatX* d_in_out, const floatX* inp, float slope) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    x128 packed_dinp;
    x128 packed_inp  = load128cs(inp + idx);
    x128 packed_dout = load128(d_in_out + idx);
    for (int k = 0; k < packed_inp.size; ++k) {
        float xi   = (float)packed_inp[k];
        float grad = llmk_act::act_leaky_relu_grad(xi, slope);
        packed_dinp[k] = (floatX)(grad * (float)packed_dout[k]);
    }
    store128(d_in_out + idx, packed_dinp);
}
void act_leaky_relu_backward_inplace(floatX* d_in_out, const floatX* inp, int N, float slope, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    act_leaky_relu_backward_inplace_kernel<<<grid_size, block_size, 0, stream>>>(d_in_out, inp, slope);
    cudaCheck(cudaGetLastError());
}

__global__ void act_elu_forward_kernel(floatX* out, const floatX* inp, float alpha) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    x128 packed_out;
    x128 packed_inp = load128cs(inp + idx);
    for (int k = 0; k < packed_inp.size; ++k) {
        float xi = (float)packed_inp[k];
        packed_out[k] = (floatX)llmk_act::act_elu_fwd(xi, alpha);
    }
    store128(out + idx, packed_out);
}
void act_elu_forward(floatX* out, const floatX* inp, int N, float alpha, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 512;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    act_elu_forward_kernel<<<grid_size, block_size, 0, stream>>>(out, inp, alpha);
    cudaCheck(cudaGetLastError());
}
__global__ void act_elu_backward_inplace_kernel(floatX* d_in_out, const floatX* inp, float alpha) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    x128 packed_dinp;
    x128 packed_inp  = load128cs(inp + idx);
    x128 packed_dout = load128(d_in_out + idx);
    for (int k = 0; k < packed_inp.size; ++k) {
        float xi   = (float)packed_inp[k];
        float grad = llmk_act::act_elu_grad(xi, alpha);
        packed_dinp[k] = (floatX)(grad * (float)packed_dout[k]);
    }
    store128(d_in_out + idx, packed_dinp);
}
void act_elu_backward_inplace(floatX* d_in_out, const floatX* inp, int N, float alpha, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    act_elu_backward_inplace_kernel<<<grid_size, block_size, 0, stream>>>(d_in_out, inp, alpha);
    cudaCheck(cudaGetLastError());
}

__global__ void act_threshold_forward_kernel(floatX* out, const floatX* inp, float thresh, float value) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    x128 packed_out;
    x128 packed_inp = load128cs(inp + idx);
    for (int k = 0; k < packed_inp.size; ++k) {
        float xi = (float)packed_inp[k];
        packed_out[k] = (floatX)llmk_act::act_threshold_fwd(xi, thresh, value);
    }
    store128(out + idx, packed_out);
}
void act_threshold_forward(floatX* out, const floatX* inp, int N, float thresh, float value, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 512;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    act_threshold_forward_kernel<<<grid_size, block_size, 0, stream>>>(out, inp, thresh, value);
    cudaCheck(cudaGetLastError());
}
__global__ void act_threshold_backward_inplace_kernel(floatX* d_in_out, const floatX* inp, float thresh) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    x128 packed_dinp;
    x128 packed_inp  = load128cs(inp + idx);
    x128 packed_dout = load128(d_in_out + idx);
    for (int k = 0; k < packed_inp.size; ++k) {
        float xi   = (float)packed_inp[k];
        float grad = llmk_act::act_threshold_grad(xi, thresh);
        packed_dinp[k] = (floatX)(grad * (float)packed_dout[k]);
    }
    store128(d_in_out + idx, packed_dinp);
}
void act_threshold_backward_inplace(floatX* d_in_out, const floatX* inp, int N, float thresh, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    act_threshold_backward_inplace_kernel<<<grid_size, block_size, 0, stream>>>(d_in_out, inp, thresh);
    cudaCheck(cudaGetLastError());
}

// PReLU forward shares leaky_relu; backward also yields per-channel slope grad.
// `slope` is broadcast per channel; gradient is summed over batch/seq positions.
// The per-element activation slope is C-wide; we expect a [C] slope vector and
// an inp shape [N, C] in row-major. The reduction over per-channel grads is
// handled by `act_prelu_slope_grad_reduce_kernel` (block-stride along N).
__global__ void act_prelu_forward_kernel(floatX* out, const floatX* inp, const floatX* slope, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    x128 packed_out;
    x128 packed_inp = load128cs(inp + idx);
    for (int k = 0; k < packed_inp.size; ++k) {
        float xi = (float)packed_inp[k];
        int channel = ((idx + k) % C);
        float a = (float)slope[channel];
        packed_out[k] = (floatX)llmk_act::act_prelu_fwd(xi, a);
    }
    store128(out + idx, packed_out);
}
void act_prelu_forward(floatX* out, const floatX* inp, const floatX* slope, int N, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 512;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    act_prelu_forward_kernel<<<grid_size, block_size, 0, stream>>>(out, inp, slope, C);
    cudaCheck(cudaGetLastError());
}

__global__ void act_prelu_backward_input_kernel(floatX* d_in_out, const floatX* inp, const floatX* slope, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    x128 packed_dinp;
    x128 packed_inp  = load128cs(inp + idx);
    x128 packed_dout = load128(d_in_out + idx);
    for (int k = 0; k < packed_inp.size; ++k) {
        float xi   = (float)packed_inp[k];
        int channel = ((idx + k) % C);
        float a    = (float)slope[channel];
        float grad = llmk_act::act_prelu_grad_input(xi, a);
        packed_dinp[k] = (floatX)(grad * (float)packed_dout[k]);
    }
    store128(d_in_out + idx, packed_dinp);
}
void act_prelu_backward_input(floatX* d_in_out, const floatX* inp, const floatX* slope, int N, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    act_prelu_backward_input_kernel<<<grid_size, block_size, 0, stream>>>(d_in_out, inp, slope, C);
    cudaCheck(cudaGetLastError());
}

// Per-channel slope gradient reduce (sum_{n: x_n<0} x_n * d_n). One block per channel.
__global__ void act_prelu_backward_slope_kernel(float* dslope, const floatX* inp, const floatX* dout, int rows_per_channel, int C) {
    int channel = blockIdx.x;
    if (channel >= C) return;
    float partial = 0.0f;
    int total = rows_per_channel;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int linear = i * C + channel;
        float x  = (float)inp[linear];
        float dy = (float)dout[linear];
        partial += llmk_act::act_prelu_grad_slope(x, dy);
    }
    float sum = blockReduce<warpReduceSum>(partial);
    if (threadIdx.x == 0) dslope[channel] = sum;
}
void act_prelu_backward_slope(float* dslope, const floatX* inp, const floatX* dout, int rows_per_channel, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    act_prelu_backward_slope_kernel<<<C, block_size, 0, stream>>>(dslope, inp, dout, rows_per_channel, C);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// logit_softcap (elementwise, parameterised)
// ============================================================================

__global__ void logit_softcap_forward_kernel(floatX* out, const floatX* inp, float softcap) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    x128 packed_out;
    x128 packed_inp = load128cs(inp + idx);
    for (int k = 0; k < packed_inp.size; ++k) {
        float xi = (float)packed_inp[k];
        packed_out[k] = (floatX)llmk_act::act_logit_softcap_fwd(xi, softcap);
    }
    store128(out + idx, packed_out);
}
void logit_softcap_forward(floatX* out, const floatX* inp, int N, float softcap, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 512;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    logit_softcap_forward_kernel<<<grid_size, block_size, 0, stream>>>(out, inp, softcap);
    cudaCheck(cudaGetLastError());
}

__global__ void logit_softcap_backward_inplace_kernel(floatX* d_in_out, const floatX* inp, float softcap) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    x128 packed_dinp;
    x128 packed_inp  = load128cs(inp + idx);
    x128 packed_dout = load128(d_in_out + idx);
    for (int k = 0; k < packed_inp.size; ++k) {
        float xi   = (float)packed_inp[k];
        float grad = llmk_act::act_logit_softcap_grad(xi, softcap);
        packed_dinp[k] = (floatX)(grad * (float)packed_dout[k]);
    }
    store128(d_in_out + idx, packed_dinp);
}
void logit_softcap_backward_inplace(floatX* d_in_out, const floatX* inp, int N, float softcap, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    logit_softcap_backward_inplace_kernel<<<grid_size, block_size, 0, stream>>>(d_in_out, inp, softcap);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Identity (passthrough copy). Kept for completeness.
// ============================================================================

__global__ void act_identity_forward_kernel(floatX* out, const floatX* inp) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    x128 packed = load128cs(inp + idx);
    store128(out + idx, packed);
}
void act_identity_forward(floatX* out, const floatX* inp, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 512;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    act_identity_forward_kernel<<<grid_size, block_size, 0, stream>>>(out, inp);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Two-argument elementwise ops (add, multiply, sub, residual_mix)
// ============================================================================

__global__ void elem_add_kernel(floatX* out, const floatX* a, const floatX* b) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    x128 packed_a = load128cs(a + idx);
    x128 packed_b = load128cs(b + idx);
    x128 packed_out;
    for (int k = 0; k < packed_a.size; ++k) {
        packed_out[k] = (floatX)((float)packed_a[k] + (float)packed_b[k]);
    }
    store128(out + idx, packed_out);
}
void elem_add(floatX* out, const floatX* a, const floatX* b, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 512;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    elem_add_kernel<<<grid_size, block_size, 0, stream>>>(out, a, b);
    cudaCheck(cudaGetLastError());
}

__global__ void elem_multiply_kernel(floatX* out, const floatX* a, const floatX* b) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    x128 packed_a = load128cs(a + idx);
    x128 packed_b = load128cs(b + idx);
    x128 packed_out;
    for (int k = 0; k < packed_a.size; ++k) {
        packed_out[k] = (floatX)((float)packed_a[k] * (float)packed_b[k]);
    }
    store128(out + idx, packed_out);
}
void elem_multiply(floatX* out, const floatX* a, const floatX* b, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 512;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    elem_multiply_kernel<<<grid_size, block_size, 0, stream>>>(out, a, b);
    cudaCheck(cudaGetLastError());
}

// residual_mix: per-channel `alpha * x + beta * x0`, alpha/beta length C.
__global__ void residual_mix_kernel(floatX* out, const floatX* x, const floatX* x0,
                                    const float* alpha, const float* beta, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    x128 px  = load128cs(x  + idx);
    x128 px0 = load128cs(x0 + idx);
    x128 po;
    for (int k = 0; k < px.size; ++k) {
        int channel = (idx + k) % C;
        float a = alpha[channel];
        float b = beta[channel];
        po[k] = (floatX)(a * (float)px[k] + b * (float)px0[k]);
    }
    store128(out + idx, po);
}
void residual_mix_forward(floatX* out, const floatX* x, const floatX* x0,
                          const float* alpha, const float* beta,
                          int N, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 512;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    residual_mix_kernel<<<grid_size, block_size, 0, stream>>>(out, x, x0, alpha, beta, C);
    cudaCheck(cudaGetLastError());
}

// qk_gain: per-head scalar gain applied to Q ([B, H, S, D] view; pass H = num_heads,
// HD = H * head_dim so we can broadcast the head index from the flattened position).
__global__ void qk_gain_kernel(floatX* out, const floatX* q, const float* gain, int HD, int head_dim) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    x128 pq  = load128cs(q + idx);
    x128 po;
    for (int k = 0; k < pq.size; ++k) {
        int head = ((idx + k) / head_dim) % (HD / head_dim);
        po[k] = (floatX)(gain[head] * (float)pq[k]);
    }
    store128(out + idx, po);
}
void qk_gain_forward(floatX* out, const floatX* q, const float* gain,
                     int N, int HD, int head_dim, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 512;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    qk_gain_kernel<<<grid_size, block_size, 0, stream>>>(out, q, gain, HD, head_dim);
    cudaCheck(cudaGetLastError());
}

// Scalar fusion ops (aux_loss_add, loss_scale) — kept separate so they can act
// on a single fp32 loss tensor / scalar without going through Packed128.
__global__ void scalar_add_scaled_kernel(float* main_loss, const float* aux_loss, float coef) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *main_loss = *main_loss + coef * (*aux_loss);
    }
}
void aux_loss_add(float* main_loss, const float* aux_loss, float coef, cudaStream_t stream) {
    NVTX_RANGE_FN();
    scalar_add_scaled_kernel<<<1, 1, 0, stream>>>(main_loss, aux_loss, coef);
    cudaCheck(cudaGetLastError());
}

__global__ void scalar_scale_kernel(float* loss, float coef) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *loss = (*loss) * coef;
    }
}
void loss_scale(float* loss, float coef, cudaStream_t stream) {
    NVTX_RANGE_FN();
    scalar_scale_kernel<<<1, 1, 0, stream>>>(loss, coef);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Softmax / log_softmax along the last dim (general; arbitrary row width).
// Each block handles one row; threads cooperatively reduce.
// ============================================================================

__global__ void softmax_forward_kernel(floatX* out, const floatX* inp, int row_width) {
    int row = blockIdx.x;
    const floatX* row_in  = inp + row * row_width;
    floatX*       row_out = out + row * row_width;

    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < row_width; i += blockDim.x) {
        float xi = (float)row_in[i];
        if (xi > local_max) local_max = xi;
    }
    float row_max = blockReduce<warpReduceMax>(local_max);

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < row_width; i += blockDim.x) {
        float xi = (float)row_in[i];
        local_sum += expf(xi - row_max);
    }
    float row_sum = blockReduce<warpReduceSum>(local_sum);

    float inv = 1.0f / row_sum;
    for (int i = threadIdx.x; i < row_width; i += blockDim.x) {
        float xi = (float)row_in[i];
        row_out[i] = (floatX)(expf(xi - row_max) * inv);
    }
}
void softmax_forward(floatX* out, const floatX* inp, int rows, int row_width, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    softmax_forward_kernel<<<rows, block_size, 0, stream>>>(out, inp, row_width);
    cudaCheck(cudaGetLastError());
}

__global__ void log_softmax_forward_kernel(floatX* out, const floatX* inp, int row_width) {
    int row = blockIdx.x;
    const floatX* row_in  = inp + row * row_width;
    floatX*       row_out = out + row * row_width;

    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < row_width; i += blockDim.x) {
        float xi = (float)row_in[i];
        if (xi > local_max) local_max = xi;
    }
    float row_max = blockReduce<warpReduceMax>(local_max);

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < row_width; i += blockDim.x) {
        float xi = (float)row_in[i];
        local_sum += expf(xi - row_max);
    }
    float row_sum = blockReduce<warpReduceSum>(local_sum);
    float log_z = logf(row_sum);

    for (int i = threadIdx.x; i < row_width; i += blockDim.x) {
        float xi = (float)row_in[i];
        row_out[i] = (floatX)(xi - row_max - log_z);
    }
}
void log_softmax_forward(floatX* out, const floatX* inp, int rows, int row_width, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    log_softmax_forward_kernel<<<rows, block_size, 0, stream>>>(out, inp, row_width);
    cudaCheck(cudaGetLastError());
}

// softmax / log_softmax over exactly two inputs (fixed-arity 2).
// inp_a, inp_b are same-shape buffers; out_a, out_b receive the two outputs.
__global__ void softmax_two_kernel(floatX* out_a, floatX* out_b, const floatX* a, const floatX* b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float xa = (float)a[idx];
    float xb = (float)b[idx];
    float m = fmaxf(xa, xb);
    float ea = expf(xa - m);
    float eb = expf(xb - m);
    float z  = ea + eb;
    out_a[idx] = (floatX)(ea / z);
    out_b[idx] = (floatX)(eb / z);
}
void softmax_two_forward(floatX* out_a, floatX* out_b, const floatX* a, const floatX* b, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    const int grid_size  = CEIL_DIV(N, block_size);
    softmax_two_kernel<<<grid_size, block_size, 0, stream>>>(out_a, out_b, a, b);
    cudaCheck(cudaGetLastError());
}

__global__ void logsoftmax_two_kernel(floatX* out_a, floatX* out_b, const floatX* a, const floatX* b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float xa = (float)a[idx];
    float xb = (float)b[idx];
    float m = fmaxf(xa, xb);
    float ea = expf(xa - m);
    float eb = expf(xb - m);
    float log_z = m + logf(ea + eb);
    out_a[idx] = (floatX)(xa - log_z);
    out_b[idx] = (floatX)(xb - log_z);
}
void logsoftmax_two_forward(floatX* out_a, floatX* out_b, const floatX* a, const floatX* b, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    const int grid_size  = CEIL_DIV(N, block_size);
    logsoftmax_two_kernel<<<grid_size, block_size, 0, stream>>>(out_a, out_b, a, b);
    cudaCheck(cudaGetLastError());
}

#undef LLMK_ACT_DEFINE
#undef LLMK_ACT_DEFINE_FWD
#undef LLMK_ACT_DEFINE_BWD
