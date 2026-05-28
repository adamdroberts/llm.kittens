/*
topk_route_sm120.cuh — ThunderKittens fused softmax + top-K + renormalise for
MoE routing on SM120.

Inputs:
  router_logits: [rows, E]                bf16
Outputs:
  topk_weights: [rows, K]                 bf16 (renormalised softmax probs)
  topk_indices: [rows, K]                 int32
  (optional)
    selection_counts: [E]   uint32  (atomic-add accumulated per call)
    weight_mass:      [E]   fp32    (atomic-add accumulated per call)

E ≤ 256 typical. We do per-row softmax then a heap-based top-K selection.
Each warp owns one row.
*/
#pragma once

#include "tk_common.cuh"
#include <cmath>

namespace llmk::topk_route {

using namespace ::kittens;

struct globals {
    const bf16*   logits;
    bf16*         topk_weights;
    int*          topk_indices;
    unsigned int* selection_counts;
    float*        weight_mass;
    int           rows, E, K;
};

__global__ void topk_route_kernel(globals g) {
    int row = blockIdx.x;
    if (row >= g.rows) return;
    const bf16* row_logits = g.logits + row * g.E;

    // 1) Row max
    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < g.E; j += blockDim.x) {
        float v = __bfloat162float(row_logits[j]);
        if (v > local_max) local_max = v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, local_max, off);
        if (other > local_max) local_max = other;
    }
    float row_max = __shfl_sync(0xFFFFFFFF, local_max, 0);

    // 2) sumexp
    float local_sum = 0.f;
    for (int j = threadIdx.x; j < g.E; j += blockDim.x) {
        local_sum += expf(__bfloat162float(row_logits[j]) - row_max);
    }
    for (int off = 16; off > 0; off >>= 1) local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, off);
    float sumexp = __shfl_sync(0xFFFFFFFF, local_sum, 0);

    // 3) Top-K selection. Single-thread (small K).
    if (threadIdx.x == 0) {
        constexpr int MAX_K = 16;
        float bp[MAX_K]; int bi[MAX_K];
        int K = g.K < MAX_K ? g.K : MAX_K;
        for (int k = 0; k < K; ++k) { bp[k] = -INFINITY; bi[k] = 0; }
        for (int j = 0; j < g.E; ++j) {
            float p = expf(__bfloat162float(row_logits[j]) - row_max) / sumexp;
            float min_v = bp[K - 1];
            if (p > min_v) {
                int pos = K - 1;
                while (pos > 0 && bp[pos - 1] < p) { bp[pos] = bp[pos - 1]; bi[pos] = bi[pos - 1]; --pos; }
                bp[pos] = p; bi[pos] = j;
            }
        }
        float sum_top = 0.f;
        for (int k = 0; k < K; ++k) sum_top += bp[k];
        for (int k = 0; k < K; ++k) {
            float w = bp[k] / fmaxf(sum_top, 1e-12f);
            g.topk_weights[row * g.K + k] = __float2bfloat16(w);
            g.topk_indices[row * g.K + k] = bi[k];
            if (g.selection_counts) atomicAdd(&g.selection_counts[bi[k]], 1u);
            if (g.weight_mass)      atomicAdd(&g.weight_mass[bi[k]], w);
        }
    }
}

inline void launch(bf16* topk_weights, int* topk_indices,
                   const bf16* logits, int rows, int E, int K,
                   unsigned int* selection_counts, float* weight_mass,
                   cudaStream_t stream) {
    globals g{logits, topk_weights, topk_indices, selection_counts, weight_mass, rows, E, K};
    topk_route_kernel<<<rows, ::kittens::WARP_THREADS, 0, stream>>>(g);
    cudaCheck(cudaGetLastError());
}

}  // namespace llmk::topk_route
