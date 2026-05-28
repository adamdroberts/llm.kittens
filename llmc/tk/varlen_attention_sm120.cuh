/*
varlen_attention_sm120.cuh — ThunderKittens variable-length attention for SM120.

Packed-sequence attention with cu_seqlens (FlashAttention varlen API). Each
"sequence" within the packed buffer has its own (start, length); attention
respects sequence boundaries (document-causal masking).

Inputs:
  q, k, v:     [total_tokens, H, D]
  cu_seqlens:  [B + 1]  (exclusive prefix sum of per-row lengths)
  max_seqlen:  int
  out:         [total_tokens, H, D]

We launch one warp per (sequence, head, q_block) and run FA-2 only across
the K-blocks within the same sequence.
*/
#pragma once

#include "tk_common.cuh"
#include <cmath>

namespace llmk::varlen_attention {

using namespace ::kittens;

#ifndef LLMK_SM120_VARLEN_BLOCK
#define LLMK_SM120_VARLEN_BLOCK 32
#endif

template <int D>
struct globals {
    using q_tile = st_bf<LLMK_SM120_VARLEN_BLOCK, D>;
    using kv_tile= st_bf<LLMK_SM120_VARLEN_BLOCK, D>;
    using o_tile = st_bf<LLMK_SM120_VARLEN_BLOCK, D>;
    using q_gl   = gl<bf16, -1, -1, -1, -1, q_tile>;
    using kv_gl  = gl<bf16, -1, -1, -1, -1, kv_tile>;
    using o_gl   = gl<bf16, -1, -1, -1, -1, o_tile>;
    q_gl q; kv_gl k; kv_gl v; o_gl o;
    const int* cu_seqlens;
    int H;
};

template <int D>
__global__ void varlen_fwd_kernel(const __grid_constant__ globals<D> g) {
    int seq    = blockIdx.z;
    int head   = blockIdx.y;
    int q_blk  = blockIdx.x;

    int seq_start = g.cu_seqlens[seq];
    int seq_end   = g.cu_seqlens[seq + 1];
    int seq_len   = seq_end - seq_start;

    int q_offset = q_blk * LLMK_SM120_VARLEN_BLOCK;
    if (q_offset >= seq_len) return;

    using q_rt    = rt_bf<LLMK_SM120_VARLEN_BLOCK, D, ducks::rt_layout::row>;
    using k_rt    = rt_bf<LLMK_SM120_VARLEN_BLOCK, D, ducks::rt_layout::row>;
    using v_rt    = rt_bf<LLMK_SM120_VARLEN_BLOCK, D, ducks::rt_layout::row>;
    using s_rt    = rt_fl<LLMK_SM120_VARLEN_BLOCK, LLMK_SM120_VARLEN_BLOCK, ducks::rt_layout::row>;
    using p_rt    = rt_bf<LLMK_SM120_VARLEN_BLOCK, LLMK_SM120_VARLEN_BLOCK, ducks::rt_layout::row>;
    using o_rt    = rt_fl<LLMK_SM120_VARLEN_BLOCK, D, ducks::rt_layout::row>;
    using row_cv  = typename s_rt::col_vec;

    // The Q tile for this q_block lives at absolute index (seq_start + q_offset).
    // We load via flat indexing: the GL layout treats axis 2 as a packed
    // sequence; we use the absolute row offset.
    int abs_q_row_block = (seq_start + q_offset) / LLMK_SM120_VARLEN_BLOCK;

    q_rt Q;
    ::kittens::warp::load(Q, g.q, {0, head, abs_q_row_block, 0});

    o_rt O;
    ::kittens::warp::zero(O);
    row_cv m, l_acc;
    ::kittens::warp::neg_infty(m);
    ::kittens::warp::zero(l_acc);

    const float scale = 1.0f / sqrtf((float)D);

    int num_k_blocks_in_seq = q_blk + 1;  // causal within sequence
    for (int kb = 0; kb < num_k_blocks_in_seq; ++kb) {
        int k_offset = kb * LLMK_SM120_VARLEN_BLOCK;
        if (k_offset >= seq_len) break;
        int abs_k_row_block = (seq_start + k_offset) / LLMK_SM120_VARLEN_BLOCK;

        k_rt K;
        v_rt V;
        ::kittens::warp::load(K, g.k, {0, head, abs_k_row_block, 0});
        ::kittens::warp::load(V, g.v, {0, head, abs_k_row_block, 0});

        s_rt S;
        ::kittens::warp::zero(S);
        ::kittens::warp::mma_ABt(S, Q, K, S);
        ::kittens::warp::mul(S, S, scale);

        if (kb == q_blk) {
            ::kittens::warp::make_causal(S, S, -INFINITY);
        }

        row_cv m_new;
        ::kittens::warp::row_max(m_new, S, m);
        row_cv rescale;
        ::kittens::warp::sub(rescale, m, m_new);
        ::kittens::warp::exp(rescale, rescale);
        ::kittens::warp::sub_row(S, S, m_new);
        ::kittens::warp::exp(S, S);
        row_cv block_sum;
        ::kittens::warp::row_sum(block_sum, S);
        ::kittens::warp::mul(l_acc, l_acc, rescale);
        ::kittens::warp::add(l_acc, l_acc, block_sum);
        p_rt P;
        ::kittens::warp::copy(P, S);
        ::kittens::warp::mul_row(O, O, rescale);
        auto& V_col = ::kittens::warp::swap_layout_inplace(V);
        ::kittens::warp::mma_AB(O, P, V_col, O);
        ::kittens::warp::copy(m, m_new);
    }
    ::kittens::warp::div_row(O, O, l_acc);
    ::kittens::warp::store(g.o, O, {0, head, abs_q_row_block, 0});
}

template <int D>
inline void launch_forward(bf16* out, const bf16* q, const bf16* k, const bf16* v,
                           const int* cu_seqlens, int B, int H, int total_tokens, int max_seqlen,
                           cudaStream_t stream) {
    assert(total_tokens % LLMK_SM120_VARLEN_BLOCK == 0);
    using G = globals<D>;
    typename G::q_gl q_arg {const_cast<bf16*>(q), 1u, (unsigned)H, (unsigned)total_tokens, (unsigned)D};
    typename G::kv_gl k_arg{const_cast<bf16*>(k), 1u, (unsigned)H, (unsigned)total_tokens, (unsigned)D};
    typename G::kv_gl v_arg{const_cast<bf16*>(v), 1u, (unsigned)H, (unsigned)total_tokens, (unsigned)D};
    typename G::o_gl o_arg {out,                   1u, (unsigned)H, (unsigned)total_tokens, (unsigned)D};
    G g{q_arg, k_arg, v_arg, o_arg, cu_seqlens, H};
    int q_blocks_per_seq = (max_seqlen + LLMK_SM120_VARLEN_BLOCK - 1) / LLMK_SM120_VARLEN_BLOCK;
    dim3 grid(q_blocks_per_seq, H, B);
    varlen_fwd_kernel<D><<<grid, ::kittens::WARP_THREADS, 0, stream>>>(g);
    cudaCheck(cudaGetLastError());
}

inline void launch_forward_dispatch(bf16* out, const bf16* q, const bf16* k, const bf16* v,
                                    const int* cu_seqlens, int B, int H, int total_tokens,
                                    int max_seqlen, int D, cudaStream_t stream) {
    if (D == 64)        launch_forward<64> (out, q, k, v, cu_seqlens, B, H, total_tokens, max_seqlen, stream);
    else if (D == 128)  launch_forward<128>(out, q, k, v, cu_seqlens, B, H, total_tokens, max_seqlen, stream);
    else { fprintf(stderr, "varlen_attention_sm120: D must be 64 or 128\n"); exit(EXIT_FAILURE); }
}

}  // namespace llmk::varlen_attention
