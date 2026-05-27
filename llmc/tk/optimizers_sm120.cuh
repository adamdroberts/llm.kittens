/*
optimizers_sm120.cuh — ThunderKittens optimizer updates for SM120.

Bulk pointwise / fused parameter updates: Lion, Sophia, Adafactor, AdEMAMix,
AdamW8bit, EMA, SWA, stochastic rounding helper, and the Muon Newton-Schulz
primitives (frobenius norm sq, in-place scale, combine).
*/
#pragma once

#include "tk_common.cuh"
#include <cmath>

namespace llmk::optimizers_sm120 {

using namespace ::kittens;

template <typename Tp, typename Tg>
__global__ void lion_kernel(Tp* params, float* master, Tg* grads, float* m, size_t N,
                            float lr, float beta1, float beta2, float weight_decay) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = (float)grads[i];
    float mm = m[i];
    float interp = beta2 * mm + (1.0f - beta2) * g;
    float sign_update = (interp > 0.0f) ? 1.0f : ((interp < 0.0f) ? -1.0f : 0.0f);
    float p = master ? master[i] : (float)params[i];
    p = p - lr * (sign_update + weight_decay * p);
    if (master) master[i] = p;
    params[i] = (Tp)p;
    m[i] = beta1 * mm + (1.0f - beta1) * g;
}
template <typename Tp, typename Tg>
inline void launch_lion(Tp* params, float* master, Tg* grads, float* m, size_t N,
                        float lr, float beta1, float beta2, float weight_decay, cudaStream_t stream) {
    const int bs = 256;
    lion_kernel<Tp, Tg><<<CEIL_DIV(N, bs), bs, 0, stream>>>(params, master, grads, m, N, lr, beta1, beta2, weight_decay);
    cudaCheck(cudaGetLastError());
}

template <typename Tp, typename Tg>
__global__ void sophia_kernel(Tp* params, float* master, Tg* grads, float* m, float* h, size_t N,
                              float lr, float beta1, float beta2, float gamma, float rho,
                              float eps, float weight_decay) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = (float)grads[i];
    float mn = beta1 * m[i] + (1.0f - beta1) * g;
    float hn = beta2 * h[i] + (1.0f - beta2) * g * g;
    m[i] = mn; h[i] = hn;
    float u = mn / fmaxf(gamma * hn, eps);
    u = fmaxf(-rho, fminf(rho, u));
    float p = master ? master[i] : (float)params[i];
    p = p - lr * (u + weight_decay * p);
    if (master) master[i] = p;
    params[i] = (Tp)p;
}
template <typename Tp, typename Tg>
inline void launch_sophia(Tp* params, float* master, Tg* grads, float* m, float* h, size_t N,
                          float lr, float beta1, float beta2, float gamma, float rho,
                          float eps, float weight_decay, cudaStream_t stream) {
    const int bs = 256;
    sophia_kernel<Tp, Tg><<<CEIL_DIV(N, bs), bs, 0, stream>>>(params, master, grads, m, h, N, lr, beta1, beta2, gamma, rho, eps, weight_decay);
    cudaCheck(cudaGetLastError());
}

template <typename Tp, typename Tg>
__global__ void ademamix_kernel(Tp* params, float* master, Tg* grads,
                                float* m1, float* m2, float* v, size_t N,
                                float lr, float beta1, float beta2, float beta3, float alpha,
                                float beta1_corr, float beta2_corr, float eps, float weight_decay) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = (float)grads[i];
    float m1n = beta1 * m1[i] + (1.0f - beta1) * g;
    float m2n = beta3 * m2[i] + (1.0f - beta3) * g;
    float vn  = beta2 * v[i]  + (1.0f - beta2) * g * g;
    m1[i] = m1n; m2[i] = m2n; v[i] = vn;
    float m1_hat = m1n / beta1_corr;
    float v_hat  = vn  / beta2_corr;
    float denom  = sqrtf(v_hat) + eps;
    float update = (m1_hat + alpha * m2n) / denom;
    float p = master ? master[i] : (float)params[i];
    p = p - lr * (update + weight_decay * p);
    if (master) master[i] = p;
    params[i] = (Tp)p;
}
template <typename Tp, typename Tg>
inline void launch_ademamix(Tp* params, float* master, Tg* grads,
                            float* m1, float* m2, float* v, size_t N,
                            float lr, float beta1, float beta2, float beta3, float alpha,
                            float beta1_corr, float beta2_corr, float eps, float weight_decay,
                            cudaStream_t stream) {
    const int bs = 256;
    ademamix_kernel<Tp, Tg><<<CEIL_DIV(N, bs), bs, 0, stream>>>(
        params, master, grads, m1, m2, v, N,
        lr, beta1, beta2, beta3, alpha, beta1_corr, beta2_corr, eps, weight_decay);
    cudaCheck(cudaGetLastError());
}

// EMA / SWA / stochastic rounding.
template <typename T>
__global__ void ema_kernel(T* target, const T* source, float decay, size_t N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    target[i] = (T)(decay * (float)target[i] + (1.0f - decay) * (float)source[i]);
}
template <typename T>
inline void launch_ema(T* target, const T* source, float decay, size_t N, cudaStream_t stream) {
    const int bs = 256;
    ema_kernel<T><<<CEIL_DIV(N, bs), bs, 0, stream>>>(target, source, decay, N);
    cudaCheck(cudaGetLastError());
}

template <typename T>
__global__ void swa_kernel(T* swa, const T* source, int n_avg, size_t N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    swa[i] = (T)(((float)swa[i] * (float)n_avg + (float)source[i]) / (float)(n_avg + 1));
}
template <typename T>
inline void launch_swa(T* swa, const T* source, int n_avg, size_t N, cudaStream_t stream) {
    const int bs = 256;
    swa_kernel<T><<<CEIL_DIV(N, bs), bs, 0, stream>>>(swa, source, n_avg, N);
    cudaCheck(cudaGetLastError());
}

__device__ inline uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state; x ^= x << 13; x ^= x >> 17; x ^= x << 5; *state = x; return x;
}
__global__ void stochastic_round_kernel(__nv_bfloat16* w, const float* update, uint32_t* rng, size_t N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    uint32_t r = xorshift32(&rng[i & 4095]);
    float jitter = (((float)(r >> 8) / (float)0xFFFFFF) - 0.5f) * (1.0f / 128.0f);
    float wv = __bfloat162float(w[i]) + update[i] + jitter;
    w[i] = __float2bfloat16_rn(wv);
}
inline void launch_stochastic_round(__nv_bfloat16* w, const float* update, uint32_t* rng, size_t N,
                                    cudaStream_t stream) {
    const int bs = 256;
    stochastic_round_kernel<<<CEIL_DIV(N, bs), bs, 0, stream>>>(w, update, rng, N);
    cudaCheck(cudaGetLastError());
}

// Muon helpers.
__global__ void frob_sq_kernel(float* out, const bf16* x, size_t N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float local = 0.f;
    for (size_t k = i; k < N; k += (size_t)blockDim.x * gridDim.x) {
        float v = __bfloat162float(x[k]);
        local += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) local += __shfl_xor_sync(0xFFFFFFFF, local, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, local);
}
inline void launch_frob_sq(float* out, const bf16* x, size_t N, cudaStream_t stream) {
    cudaCheck(cudaMemsetAsync(out, 0, sizeof(float), stream));
    const int bs = 256;
    int gs = std::min(1024, CEIL_DIV((int)N, bs));
    frob_sq_kernel<<<gs, bs, 0, stream>>>(out, x, N);
    cudaCheck(cudaGetLastError());
}
__global__ void scale_inplace_kernel(bf16* x, float scale, size_t N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    x[i] = __float2bfloat16(__bfloat162float(x[i]) * scale);
}
inline void launch_scale(bf16* x, float scale, size_t N, cudaStream_t stream) {
    const int bs = 256;
    scale_inplace_kernel<<<CEIL_DIV(N, bs), bs, 0, stream>>>(x, scale, N);
    cudaCheck(cudaGetLastError());
}

}  // namespace llmk::optimizers_sm120
