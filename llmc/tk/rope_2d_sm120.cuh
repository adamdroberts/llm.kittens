/*
rope_2d_sm120.cuh — ThunderKittens 2D RoPE (vision / multimodal) for SM120.

Treats head_dim as (H_part, W_part) — first quarter encodes the height
position, second quarter encodes the width position. The interleaved
half-split convention matches NeuralFn's apply_rotary_emb.

Inputs (cos/sin tables): [H_pos, W_pos, D/2]  bf16
Inputs (q, k): [B, H, H_pos * W_pos, D]
Outputs (q_out, k_out): same shape.
*/
#pragma once

#include "tk_common.cuh"

namespace llmk::rope_2d {

using namespace ::kittens;

#ifndef LLMK_SM120_ROPE2D_BLOCK
#define LLMK_SM120_ROPE2D_BLOCK 32
#endif

template <int D>
struct globals {
    bf16* q_in;
    bf16* k_in;
    bf16* q_out;
    bf16* k_out;
    bf16* cos2d;        // [H_pos, W_pos, D/2]
    bf16* sin2d;        // [H_pos, W_pos, D/2]
    int   B, H_q, H_k, H_pos, W_pos;
};

template <int D>
__global__ void rope_2d_apply_kernel(globals<D> g) {
    int b   = blockIdx.z;
    int seq = blockIdx.y * blockDim.y + threadIdx.y;   // 0 .. H_pos*W_pos - 1
    int d   = blockIdx.x * blockDim.x + threadIdx.x;   // 0 .. D - 1
    int total = g.H_pos * g.W_pos;
    if (seq >= total || d >= D) return;
    int half = D / 2;
    int d_half = d < half ? d : d - half;

    int h_pos_idx = seq / g.W_pos;
    int w_pos_idx = seq % g.W_pos;
    float c  = __bfloat162float(g.cos2d[(h_pos_idx * g.W_pos + w_pos_idx) * half + d_half]);
    float si = __bfloat162float(g.sin2d[(h_pos_idx * g.W_pos + w_pos_idx) * half + d_half]);

    for (int h = 0; h < g.H_q; ++h) {
        int off = ((b * g.H_q + h) * total + seq) * D + d;
        if (d < half) {
            float x1 = __bfloat162float(g.q_in[off]);
            float x2 = __bfloat162float(g.q_in[off + half]);
            g.q_out[off] = __float2bfloat16(x1 * c + x2 * si);
        } else {
            float x1 = __bfloat162float(g.q_in[off - half]);
            float x2 = __bfloat162float(g.q_in[off]);
            g.q_out[off] = __float2bfloat16(x1 * (-si) + x2 * c);
        }
    }
    for (int h = 0; h < g.H_k; ++h) {
        int off = ((b * g.H_k + h) * total + seq) * D + d;
        if (d < half) {
            float x1 = __bfloat162float(g.k_in[off]);
            float x2 = __bfloat162float(g.k_in[off + half]);
            g.k_out[off] = __float2bfloat16(x1 * c + x2 * si);
        } else {
            float x1 = __bfloat162float(g.k_in[off - half]);
            float x2 = __bfloat162float(g.k_in[off]);
            g.k_out[off] = __float2bfloat16(x1 * (-si) + x2 * c);
        }
    }
}

template <int D>
inline void launch(bf16* q_out, bf16* k_out,
                   const bf16* q, const bf16* k,
                   const bf16* cos2d, const bf16* sin2d,
                   int B, int H_q, int H_k, int H_pos, int W_pos,
                   cudaStream_t stream) {
    globals<D> g{const_cast<bf16*>(q), const_cast<bf16*>(k),
                 q_out, k_out,
                 const_cast<bf16*>(cos2d), const_cast<bf16*>(sin2d),
                 B, H_q, H_k, H_pos, W_pos};
    dim3 block(32, 8);
    dim3 grid(CEIL_DIV(D, 32), CEIL_DIV(H_pos * W_pos, 8), B);
    rope_2d_apply_kernel<D><<<grid, block, 0, stream>>>(g);
    cudaCheck(cudaGetLastError());
}

inline void launch_dispatch(bf16* q_out, bf16* k_out,
                            const bf16* q, const bf16* k,
                            const bf16* cos2d, const bf16* sin2d,
                            int B, int H_q, int H_k, int H_pos, int W_pos, int D,
                            cudaStream_t stream) {
    if (D == 64)       launch<64> (q_out, k_out, q, k, cos2d, sin2d, B, H_q, H_k, H_pos, W_pos, stream);
    else if (D == 128) launch<128>(q_out, k_out, q, k, cos2d, sin2d, B, H_q, H_k, H_pos, W_pos, stream);
    else { fprintf(stderr, "rope_2d_sm120: D must be 64 or 128\n"); exit(EXIT_FAILURE); }
}

}  // namespace llmk::rope_2d
