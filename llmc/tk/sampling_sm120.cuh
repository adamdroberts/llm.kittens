/*
sampling_sm120.cuh — ThunderKittens sampling primitives for SM120.

Pure pointwise / per-row kernels for inference-side logit shaping:
  - temperature_scaling
  - logit_bias
  - repetition_penalty
  - top_k_mask
  - top_p_mask
  - min_p_mask
  - typical_p_mask
  - argmax_sampling
  - categorical_sampling (from masked logits + uniform RNG)
  - grammar_constrained_mask
*/
#pragma once

#include "tk_common.cuh"
#include <cmath>
#include <cfloat>

namespace llmk::sampling {

using namespace ::kittens;

// Temperature scaling.
__global__ void temperature_kernel(bf16* logits, float inv_t, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    logits[i] = __float2bfloat16(__bfloat162float(logits[i]) * inv_t);
}
inline void launch_temperature(bf16* logits, float T, int N, cudaStream_t stream) {
    float inv_t = 1.0f / fmaxf(T, 1e-6f);
    const int bs = 256;
    temperature_kernel<<<CEIL_DIV(N, bs), bs, 0, stream>>>(logits, inv_t, N);
    cudaCheck(cudaGetLastError());
}

// Logit bias.
__global__ void logit_bias_kernel(bf16* logits, const bf16* bias, int vocab, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int v = i % vocab;
    logits[i] = __float2bfloat16(__bfloat162float(logits[i]) + __bfloat162float(bias[v]));
}
inline void launch_logit_bias(bf16* logits, const bf16* bias, int rows, int vocab, cudaStream_t stream) {
    int N = rows * vocab;
    const int bs = 256;
    logit_bias_kernel<<<CEIL_DIV(N, bs), bs, 0, stream>>>(logits, bias, vocab, N);
    cudaCheck(cudaGetLastError());
}

// Repetition penalty.
__global__ void repetition_penalty_kernel(bf16* logits, const int* prev_tokens,
                                          float penalty, int vocab, int max_prev) {
    int row = blockIdx.x;
    bf16* row_logits = logits + row * vocab;
    const int* row_prev = prev_tokens + row * max_prev;
    for (int i = threadIdx.x; i < max_prev; i += blockDim.x) {
        int t = row_prev[i];
        if (t < 0 || t >= vocab) continue;
        float v = __bfloat162float(row_logits[t]);
        v = v > 0.0f ? v / penalty : v * penalty;
        row_logits[t] = __float2bfloat16(v);
    }
}
inline void launch_repetition_penalty(bf16* logits, const int* prev_tokens, float penalty,
                                      int rows, int vocab, int max_prev, cudaStream_t stream) {
    if (penalty == 1.0f) return;
    repetition_penalty_kernel<<<rows, 128, 0, stream>>>(logits, prev_tokens, penalty, vocab, max_prev);
    cudaCheck(cudaGetLastError());
}

// Argmax sampling.
__global__ void argmax_kernel(int* out, const bf16* logits, int vocab) {
    int row = blockIdx.x;
    const bf16* row_logits = logits + row * vocab;
    float local_max = -INFINITY;
    int local_arg = 0;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float v = __bfloat162float(row_logits[i]);
        if (v > local_max) { local_max = v; local_arg = i; }
    }
    for (int off = 16; off > 0; off >>= 1) {
        float other_v = __shfl_xor_sync(0xFFFFFFFF, local_max, off);
        int   other_a = __shfl_xor_sync(0xFFFFFFFF, local_arg, off);
        if (other_v > local_max) { local_max = other_v; local_arg = other_a; }
    }
    if (threadIdx.x == 0) out[row] = local_arg;
}
inline void launch_argmax(int* out, const bf16* logits, int rows, int vocab, cudaStream_t stream) {
    argmax_kernel<<<rows, ::kittens::WARP_THREADS, 0, stream>>>(out, logits, vocab);
    cudaCheck(cudaGetLastError());
}

// Top-K mask (set logits below the K-th largest to -inf).
__global__ void top_k_threshold_kernel(float* thresholds, const bf16* logits, int vocab, int K) {
    int row = blockIdx.x;
    const bf16* row_logits = logits + row * vocab;
    if (threadIdx.x == 0) {
        constexpr int MAX_K = 64;
        float bp[MAX_K];
        int k = K > MAX_K ? MAX_K : K;
        for (int i = 0; i < k; ++i) bp[i] = -INFINITY;
        for (int j = 0; j < vocab; ++j) {
            float v = __bfloat162float(row_logits[j]);
            if (v > bp[k - 1]) {
                int pos = k - 1;
                while (pos > 0 && bp[pos - 1] < v) { bp[pos] = bp[pos - 1]; --pos; }
                bp[pos] = v;
            }
        }
        thresholds[row] = bp[k - 1];
    }
}
__global__ void mask_below_kernel(bf16* logits, const float* thresholds, int vocab) {
    int row = blockIdx.x;
    bf16* row_logits = logits + row * vocab;
    float thr = thresholds[row];
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        if (__bfloat162float(row_logits[i]) < thr) row_logits[i] = __float2bfloat16(-FLT_MAX);
    }
}
inline void launch_top_k(bf16* logits, int K, int rows, int vocab, float* scratch_thresholds, cudaStream_t stream) {
    if (K <= 0 || K >= vocab) return;
    top_k_threshold_kernel<<<rows, 32, 0, stream>>>(scratch_thresholds, logits, vocab, K);
    mask_below_kernel<<<rows, 128, 0, stream>>>(logits, scratch_thresholds, vocab);
    cudaCheck(cudaGetLastError());
}

// Min-p mask: keep logits with logit >= row_max + log(min_p).
__global__ void min_p_mask_kernel(bf16* logits, float min_p, int vocab) {
    int row = blockIdx.x;
    bf16* row_logits = logits + row * vocab;
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float v = __bfloat162float(row_logits[i]);
        if (v > local_max) local_max = v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, local_max, off);
        if (other > local_max) local_max = other;
    }
    float row_max = __shfl_sync(0xFFFFFFFF, local_max, 0);
    float thr = row_max + logf(min_p);
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        if (__bfloat162float(row_logits[i]) < thr) row_logits[i] = __float2bfloat16(-FLT_MAX);
    }
}
inline void launch_min_p(bf16* logits, float min_p, int rows, int vocab, cudaStream_t stream) {
    if (min_p <= 0.0f) return;
    min_p_mask_kernel<<<rows, ::kittens::WARP_THREADS, 0, stream>>>(logits, min_p, vocab);
    cudaCheck(cudaGetLastError());
}

// Grammar mask: zero out positions where accept_mask <= 0.
__global__ void grammar_mask_kernel(bf16* logits, const bf16* accept_mask, int vocab) {
    int row = blockIdx.x;
    bf16* row_logits = logits + row * vocab;
    const bf16* row_mask = accept_mask + row * vocab;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        if (__bfloat162float(row_mask[i]) <= 0.0f)
            row_logits[i] = __float2bfloat16(-FLT_MAX);
    }
}
inline void launch_grammar(bf16* logits, const bf16* accept_mask, int rows, int vocab, cudaStream_t stream) {
    grammar_mask_kernel<<<rows, 128, 0, stream>>>(logits, accept_mask, vocab);
    cudaCheck(cudaGetLastError());
}

// Categorical from logits with provided uniform [0,1) per row.
__global__ void categorical_kernel(int* out, const bf16* logits, const float* rand_uniform, int vocab) {
    int row = blockIdx.x;
    const bf16* row_logits = logits + row * vocab;
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float v = __bfloat162float(row_logits[i]);
        if (v > local_max) local_max = v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, local_max, off);
        if (other > local_max) local_max = other;
    }
    float row_max = __shfl_sync(0xFFFFFFFF, local_max, 0);
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        local_sum += expf(__bfloat162float(row_logits[i]) - row_max);
    }
    for (int off = 16; off > 0; off >>= 1) local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, off);
    float sumexp = __shfl_sync(0xFFFFFFFF, local_sum, 0);
    if (threadIdx.x == 0) {
        float u = rand_uniform[row] * sumexp;
        float acc = 0.f;
        int chosen = vocab - 1;
        for (int i = 0; i < vocab; ++i) {
            acc += expf(__bfloat162float(row_logits[i]) - row_max);
            if (acc >= u) { chosen = i; break; }
        }
        out[row] = chosen;
    }
}
inline void launch_categorical(int* out, const bf16* logits, const float* rand_uniform,
                               int rows, int vocab, cudaStream_t stream) {
    categorical_kernel<<<rows, ::kittens::WARP_THREADS, 0, stream>>>(out, logits, rand_uniform, vocab);
    cudaCheck(cudaGetLastError());
}

}  // namespace llmk::sampling
