/*
selective_scan_sm120.cuh — ThunderKittens selective SSM (Mamba) scan forward
for SM120.

Algorithm (per-channel, per-batch):
    h_t = A_bar(t) ⊙ h_{t-1} + B_bar(t) * x_t
    y_t = (C_bar(t)^T h_t) + D ⊙ x_t

with discretised parameters
    A_bar(t) = exp(Δ_t * A)
    B_bar(t) = Δ_t * B(x_t)

Parallelisation: one warp per (batch, channel). The state h ∈ R^{d_state}
lives in registers / shared memory across the sequential T loop.

This is the structurally-correct kernel suitable for Mamba's d_state ≤ 64
(typical: 16). For T parallel scan use cases consider a tree-reduced version
in a follow-up; the sequential form here is what Mamba's reference impl uses
inside `mamba_inner_fn`.
*/
#pragma once

#include "tk_common.cuh"
#include <cmath>

namespace llmk::selective_scan {

using namespace ::kittens;

#ifndef LLMK_SM120_SSM_BLOCK
#define LLMK_SM120_SSM_BLOCK 32  // channels per CTA
#endif

template <int D_STATE>
struct globals {
    bf16* x;          // [B, T, d_inner]
    bf16* delta;      // [B, T, d_inner]
    bf16* A;          // [d_inner, d_state]
    bf16* B;          // [B, T, d_state]
    bf16* C;          // [B, T, d_state]
    bf16* D;          // [d_inner] (optional, nullptr to skip)
    bf16* y;          // [B, T, d_inner]
    int   batch, T, d_inner;
};

template <int D_STATE>
__global__ void selective_scan_fwd_kernel(globals<D_STATE> g) {
    int batch   = blockIdx.y;
    int ch_blk  = blockIdx.x;
    int channel = ch_blk * LLMK_SM120_SSM_BLOCK + (threadIdx.x % LLMK_SM120_SSM_BLOCK);
    if (channel >= g.d_inner) return;

    // Per-thread state h[D_STATE] in registers.
    float h[D_STATE];
    #pragma unroll
    for (int n = 0; n < D_STATE; ++n) h[n] = 0.f;

    // Pre-load A[channel] (length D_STATE)
    float A_row[D_STATE];
    #pragma unroll
    for (int n = 0; n < D_STATE; ++n) {
        A_row[n] = __bfloat162float(g.A[channel * D_STATE + n]);
    }

    float D_skip = g.D ? __bfloat162float(g.D[channel]) : 0.0f;

    for (int t = 0; t < g.T; ++t) {
        float x_t     = __bfloat162float(g.x    [(batch * g.T + t) * g.d_inner + channel]);
        float delta_t = __bfloat162float(g.delta[(batch * g.T + t) * g.d_inner + channel]);

        float y_t = 0.0f;
        #pragma unroll
        for (int n = 0; n < D_STATE; ++n) {
            float B_t  = __bfloat162float(g.B[(batch * g.T + t) * D_STATE + n]);
            float C_t  = __bfloat162float(g.C[(batch * g.T + t) * D_STATE + n]);
            float A_bar = expf(delta_t * A_row[n]);
            float B_bar = delta_t * B_t;
            float h_new = A_bar * h[n] + B_bar * x_t;
            h[n] = h_new;
            y_t += C_t * h_new;
        }
        if (g.D) y_t += D_skip * x_t;
        g.y[(batch * g.T + t) * g.d_inner + channel] = __float2bfloat16(y_t);
    }
}

template <int D_STATE>
inline void launch_forward(bf16* y, const bf16* x, const bf16* delta,
                           const bf16* A, const bf16* B, const bf16* C, const bf16* D,
                           int batch, int T, int d_inner, cudaStream_t stream) {
    globals<D_STATE> g{
        const_cast<bf16*>(x), const_cast<bf16*>(delta), const_cast<bf16*>(A),
        const_cast<bf16*>(B), const_cast<bf16*>(C), const_cast<bf16*>(D),
        y, batch, T, d_inner,
    };
    int blocks_x = CEIL_DIV(d_inner, LLMK_SM120_SSM_BLOCK);
    dim3 grid(blocks_x, batch);
    selective_scan_fwd_kernel<D_STATE><<<grid, LLMK_SM120_SSM_BLOCK, 0, stream>>>(g);
    cudaCheck(cudaGetLastError());
}

inline void launch_forward_dispatch(bf16* y, const bf16* x, const bf16* delta,
                                    const bf16* A, const bf16* B, const bf16* C, const bf16* D,
                                    int batch, int T, int d_inner, int d_state, cudaStream_t stream) {
    if (d_state == 16) launch_forward<16>(y, x, delta, A, B, C, D, batch, T, d_inner, stream);
    else if (d_state == 32) launch_forward<32>(y, x, delta, A, B, C, D, batch, T, d_inner, stream);
    else if (d_state == 64) launch_forward<64>(y, x, delta, A, B, C, D, batch, T, d_inner, stream);
    else {
        fprintf(stderr, "selective_scan_sm120: unsupported d_state=%d (16/32/64 only)\n", d_state);
        exit(EXIT_FAILURE);
    }
}

}  // namespace llmk::selective_scan
