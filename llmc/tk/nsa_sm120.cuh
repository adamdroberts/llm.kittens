/*
nsa_sm120.cuh — ThunderKittens native sparse attention (NSA, DeepSeek-V3.2)
forward for SM120.

NSA assigns each query a small set of high-affinity keys (selected by a
learned scoring head), then runs attention only against that sparse set.
The kernel here uses pre-computed `top_keys` indices (selected upstream by
`nsa_select_topk_keys`).

Inputs:
  q:         [B, H, S_q, D]
  k, v:      [B, H, S_k, D]
  top_keys:  [B, H, S_q, K_PER_Q]    int32 — indices into S_k axis

Each warp handles one (batch, head, q_pos). For each q_pos we gather the K
keys and V values referenced by top_keys[q_pos, :K_PER_Q], compute attention
scores against this small set, softmax, weighted-sum V.

K_PER_Q is small (16..64); the inner loop is fully unrolled.
*/
#pragma once

#include "tk_common.cuh"
#include <cmath>

namespace llmk::nsa {

using namespace ::kittens;

#ifndef LLMK_SM120_NSA_K_PER_Q
#define LLMK_SM120_NSA_K_PER_Q 32
#endif

template <int D>
struct globals {
    bf16* q;
    bf16* k;
    bf16* v;
    bf16* out;
    const int* top_keys;
    int H, S_q, S_k, K_per_q;
};

template <int D>
__global__ void nsa_fwd_kernel(globals<D> g) {
    int batch = blockIdx.z;
    int head  = blockIdx.y;
    int q_pos = blockIdx.x;

    // Each thread holds one of D dims of Q and O.
    int dim = threadIdx.x;
    if (dim >= D) return;

    // Load Q[batch, head, q_pos, :] — distributed across threads
    float q_val = __bfloat162float(g.q[((batch * g.H + head) * g.S_q + q_pos) * D + dim]);

    // Compute attention scores against top-K keys
    extern __shared__ float s_score[];      // length K_per_q
    extern __shared__ __align__(16) float s_v_dot[];  // length D (for weighted sum)
    float* scores = s_score;

    const float scale = 1.0f / sqrtf((float)D);

    for (int k_slot = 0; k_slot < g.K_per_q; ++k_slot) {
        int k_idx = g.top_keys[((batch * g.H + head) * g.S_q + q_pos) * g.K_per_q + k_slot];
        if (k_idx < 0 || k_idx >= g.S_k) {
            if (dim == 0) scores[k_slot] = -INFINITY;
            continue;
        }
        float k_val = __bfloat162float(g.k[((batch * g.H + head) * g.S_k + k_idx) * D + dim]);
        // Per-thread partial: q_val * k_val. Warp-reduce to get the full dot.
        float partial = q_val * k_val;
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_xor_sync(0xFFFFFFFF, partial, offset);
        }
        if (dim == 0) scores[k_slot] = partial * scale;
    }
    __syncwarp();

    // Softmax over scores
    if (dim == 0) {
        float row_max = -INFINITY;
        for (int s = 0; s < g.K_per_q; ++s) if (scores[s] > row_max) row_max = scores[s];
        float sumexp = 0.0f;
        for (int s = 0; s < g.K_per_q; ++s) { scores[s] = expf(scores[s] - row_max); sumexp += scores[s]; }
        for (int s = 0; s < g.K_per_q; ++s) scores[s] /= sumexp;
    }
    __syncwarp();

    // Weighted sum of V[top_keys[:K_per_q], dim]
    float acc = 0.0f;
    for (int k_slot = 0; k_slot < g.K_per_q; ++k_slot) {
        int k_idx = g.top_keys[((batch * g.H + head) * g.S_q + q_pos) * g.K_per_q + k_slot];
        if (k_idx < 0 || k_idx >= g.S_k) continue;
        float v_val = __bfloat162float(g.v[((batch * g.H + head) * g.S_k + k_idx) * D + dim]);
        acc += scores[k_slot] * v_val;
    }
    g.out[((batch * g.H + head) * g.S_q + q_pos) * D + dim] = __float2bfloat16(acc);
}

template <int D>
inline void launch_forward(bf16* out, const bf16* q, const bf16* k, const bf16* v,
                           const int* top_keys, int B, int H, int S_q, int S_k, int K_per_q,
                           cudaStream_t stream) {
    globals<D> g{const_cast<bf16*>(q), const_cast<bf16*>(k), const_cast<bf16*>(v),
                 out, top_keys, H, S_q, S_k, K_per_q};
    dim3 grid(S_q, H, B);
    int shmem = K_per_q * sizeof(float) + D * sizeof(float);
    nsa_fwd_kernel<D><<<grid, D, shmem, stream>>>(g);
    cudaCheck(cudaGetLastError());
}

inline void launch_forward_dispatch(bf16* out, const bf16* q, const bf16* k, const bf16* v,
                                    const int* top_keys, int B, int H, int S_q, int S_k, int K_per_q, int D,
                                    cudaStream_t stream) {
    if (D == 64)       launch_forward<64> (out, q, k, v, top_keys, B, H, S_q, S_k, K_per_q, stream);
    else if (D == 128) launch_forward<128>(out, q, k, v, top_keys, B, H, S_q, S_k, K_per_q, stream);
    else { fprintf(stderr, "nsa_sm120: D must be 64 or 128\n"); exit(EXIT_FAILURE); }
}

}  // namespace llmk::nsa
