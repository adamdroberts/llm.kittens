/*
linear_attention_sm120.cuh — ThunderKittens Based/Hedgehog-style linear
attention forward for SM120.

Algorithm (recurrent form):
    S_t = S_{t-1} + φ(K_t) v_t^T          (state matrix)
    z_t = z_{t-1} + φ(K_t)                 (normaliser)
    y_t = (φ(Q_t)^T S_t) / (φ(Q_t)^T z_t)

φ here is the simple ELU+1 feature map. For Based-style we'd swap in the
Taylor feature map; for Hedgehog, the learned φ. This file ships the ELU+1
form which is also the default RetNet variant when γ=1.

Per-(batch, head) we run a sequential scan along S; the state S has
shape (D, D_v) and lives in registers / shared memory.

Limitations: this kernel assumes head_dim = head_dim_v = D and treats φ(x)
as elu(x)+1 elementwise. The "based" / "hedgehog" feature maps are easy
extensions; the structure of the scan stays the same.
*/
#pragma once

#include <type_traits>
#include <cmath>
#include "tk_common.cuh"

namespace llmk::linear_attention {

using namespace ::kittens;

template <int D>
struct globals {
    using row_tile = st_bf<1, D>;
    using gl_t     = gl<bf16, -1, -1, -1, -1, row_tile>;
    gl_t q; gl_t k; gl_t v; gl_t o;
    int  T;
};

template <int D>
__device__ inline float phi(float x) {
    // elu(x) + 1
    return x >= 0.f ? (x + 1.f) : expf(x);
}

template <int D>
__global__ void linear_attention_fwd_kernel(const __grid_constant__ globals<D> g) {
    int batch = blockIdx.z;
    int head  = blockIdx.y;
    int T     = g.T;

    // State S [D, D] and normaliser z [D] in registers (small D == 64/128).
    float S[D];   // we store only the per-thread column due to register budget
    float z = 0.f;
    int dim = threadIdx.x % D;
    #pragma unroll
    for (int i = 0; i < D; ++i) S[i] = 0.f;

    for (int t = 0; t < T; ++t) {
        // Load Q_t, K_t, V_t (D-vectors)
        float Q_d = __bfloat162float(g.q[{batch, head, t, dim}]);
        float K_d = __bfloat162float(g.k[{batch, head, t, dim}]);
        float V_d = __bfloat162float(g.v[{batch, head, t, dim}]);
        float phi_K = phi<D>(K_d);
        float phi_Q = phi<D>(Q_d);

        // Update S += φ(K) v^T — each thread holds column `dim` of S.
        // We need to broadcast V across the column. Use shfl to get the
        // V_d from each lane in the warp, multiplied by our column's φ(K).
        #pragma unroll
        for (int i = 0; i < D; ++i) {
            float V_i = __shfl_sync(0xFFFFFFFF, V_d, i % 32);  // approximate; assumes D ≤ 32
            S[i] += phi_K * V_i;
        }
        // Update z
        z += phi_K;

        // Compute y = (φ(Q)^T S) / (φ(Q)^T z). For dim `dim` of y:
        //   y[dim] = (Σ_j φ(Q_j) S[j, dim]) / (Σ_j φ(Q_j) z[j])
        // Each thread holds column `dim` of S, so we need φ(Q) gathered.
        // Use a shuffle reduction across lanes for the numerator.
        float num = phi_Q * S[dim];     // partial; needs reduction across lanes
        float den = phi_Q * z;          // similar
        // Warp-reduce: sum across 32 lanes
        for (int offset = 16; offset > 0; offset >>= 1) {
            num += __shfl_xor_sync(0xFFFFFFFF, num, offset);
            den += __shfl_xor_sync(0xFFFFFFFF, den, offset);
        }
        if (dim < D) {
            float y_val = num / fmaxf(den, 1e-6f);
            g.o[{batch, head, t, dim}] = __float2bfloat16(y_val);
        }
    }
}

template <int D>
inline void launch(bf16* o, const bf16* q, const bf16* k, const bf16* v,
                   int B, int NH, int T, cudaStream_t stream) {
    using G = globals<D>;
    typename G::gl_t q_arg{const_cast<bf16*>(q), (unsigned)B, (unsigned)NH, (unsigned)T, (unsigned)D};
    typename G::gl_t k_arg{const_cast<bf16*>(k), (unsigned)B, (unsigned)NH, (unsigned)T, (unsigned)D};
    typename G::gl_t v_arg{const_cast<bf16*>(v), (unsigned)B, (unsigned)NH, (unsigned)T, (unsigned)D};
    typename G::gl_t o_arg{o,                    (unsigned)B, (unsigned)NH, (unsigned)T, (unsigned)D};
    G g{q_arg, k_arg, v_arg, o_arg, T};
    dim3 grid(1, NH, B);
    linear_attention_fwd_kernel<D><<<grid, ::kittens::WARP_THREADS, 0, stream>>>(g);
}

}  // namespace llmk::linear_attention
