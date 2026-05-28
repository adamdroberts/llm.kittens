/*
rl.cuh — RL infrastructure kernels (GAE compute + helpers).

PPO clipped loss and KL penalty live in losses.cuh.
*/
#pragma once

#include <assert.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ============================================================================
// GAE compute (reverse-sequential).
//
//   rewards: [B, T]
//   values:  [B, T]
//   adv:     [B, T]  (output)
//   ret:     [B, T]  (= adv + values)
//
//   adv_t = δ_t + γλ·adv_{t+1}    with adv_{T-1}=δ_{T-1}, next_value padded to 0.
//   δ_t   = r_t + γ·next_value - v_t
//
// One block per batch; sequential along T inside the block.
// ============================================================================

__global__ void gae_kernel(floatX* adv, floatX* ret, const floatX* rewards, const floatX* values,
                          int T, float gamma, float lam) {
    int b = blockIdx.x;
    if (threadIdx.x != 0) return;
    const floatX* rb = rewards + b * T;
    const floatX* vb = values  + b * T;
    floatX* ab = adv + b * T;
    floatX* rb_out = ret + b * T;
    float next_adv = 0.0f;
    float next_value = 0.0f;
    for (int t = T - 1; t >= 0; --t) {
        float r = (float)rb[t];
        float v = (float)vb[t];
        float delta = r + gamma * next_value - v;
        next_adv = delta + gamma * lam * next_adv;
        ab[t] = (floatX)next_adv;
        rb_out[t] = (floatX)(next_adv + v);
        next_value = v;
    }
}

void gae_compute(floatX* adv, floatX* ret, const floatX* rewards, const floatX* values,
                 int B, int T, float gamma, float lam, cudaStream_t stream) {
    NVTX_RANGE_FN();
    gae_kernel<<<B, 1, 0, stream>>>(adv, ret, rewards, values, T, gamma, lam);
    cudaCheck(cudaGetLastError());
}

// Reverse-scan variant using a Blelloch-style scan over T (faster when T is large).
// We provide a simple parallel reverse-scan that's correct but not optimised;
// the sequential kernel above is faster for the small T (~ 64-1024) we expect.
// Kept for completeness; callers can pick either.
__global__ void gae_parallel_kernel(floatX* adv, const floatX* rewards, const floatX* values,
                                    int T, float gamma, float lam) {
    int b = blockIdx.x;
    // Each thread handles its own t; we do log2(T) doubling-passes.
    __shared__ float buf[1024];  // assumes T <= 1024
    if (T > 1024) return;
    const floatX* rb = rewards + b * T;
    const floatX* vb = values  + b * T;
    int t = threadIdx.x;
    if (t >= T) return;
    float v_t = (float)vb[t];
    float v_next = (t + 1 < T) ? (float)vb[t + 1] : 0.0f;
    float delta = (float)rb[t] + gamma * v_next - v_t;
    buf[t] = delta;
    __syncthreads();
    // Reverse scan: out[t] = sum_{k >= t} (γλ)^{k - t} * δ_k
    // We do log2(T) doubling passes.
    float gl = gamma * lam;
    for (int offset = 1; offset < T; offset <<= 1) {
        float add = 0.0f;
        if (t + offset < T) {
            float coef = 1.0f;
            for (int i = 0; i < offset; ++i) coef *= gl;
            add = coef * buf[t + offset];
        }
        __syncthreads();
        if (t + offset < T) buf[t] += add;
        __syncthreads();
    }
    adv[b * T + t] = (floatX)buf[t];
}
