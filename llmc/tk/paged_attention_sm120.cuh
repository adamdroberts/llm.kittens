/*
paged_attention_sm120.cuh — ThunderKittens paged-KV decode attention for SM120.

vLLM-style paged KV cache: K, V are stored in fixed-size pages and a
per-request block_table maps logical positions to physical page IDs.

Decode form: Q is one token per batch row (shape [B, H, 1, D]). For each
(batch, head), iterate the active KV pages, fetch K/V tiles, run online
softmax accumulation, and write O. Backing layout:

  pages_k, pages_v: [num_pages, Hk, page_size, D]
  block_table:      [B, max_blocks]
  cache_len:        [B]
  out:              [B, H, D]
*/
#pragma once

#include "tk_common.cuh"
#include <cmath>

namespace llmk::paged_attention {

using namespace ::kittens;

#ifndef LLMK_SM120_PAGED_PAGE_BLOCK
#define LLMK_SM120_PAGED_PAGE_BLOCK 16   // K-tile rows per inner step
#endif

template <int D>
struct globals {
    using q_tile  = st_bf<1,  D>;
    using kv_tile = st_bf<LLMK_SM120_PAGED_PAGE_BLOCK, D>;
    using o_tile  = st_bf<1,  D>;
    using q_gl    = gl<bf16, -1, -1, -1, -1, q_tile>;
    using kv_gl   = gl<bf16, -1, -1, -1, -1, kv_tile>;
    using o_gl    = gl<bf16, -1, -1, -1, -1, o_tile>;

    q_gl  q;
    kv_gl pages_k;
    kv_gl pages_v;
    o_gl  out;
    const int* block_table;
    const int* cache_len;
    int H, Hk, page_size, max_blocks;
};

template <int D>
__global__ void paged_attention_decode_kernel(const __grid_constant__ globals<D> g) {
    static_assert(D == 64 || D == 128, "paged_attention_sm120: HS must be 64 or 128");
    int batch = blockIdx.z;
    int head  = blockIdx.y;
    int hk    = head % g.Hk;
    int len   = g.cache_len[batch];

    using q_rt   = rt_bf<1, D, ducks::rt_layout::row>;
    using kv_rt  = rt_bf<LLMK_SM120_PAGED_PAGE_BLOCK, D, ducks::rt_layout::row>;
    using s_rt   = rt_fl<1, LLMK_SM120_PAGED_PAGE_BLOCK, ducks::rt_layout::row>;
    using p_rt   = rt_bf<1, LLMK_SM120_PAGED_PAGE_BLOCK, ducks::rt_layout::row>;
    using o_rt   = rt_fl<1, D, ducks::rt_layout::row>;
    using row_cv = typename s_rt::col_vec;

    q_rt Q;
    ::kittens::warp::load(Q, g.q, {batch, head, 0, 0});

    o_rt O;
    ::kittens::warp::zero(O);
    row_cv m, l_acc;
    ::kittens::warp::neg_infty(m);
    ::kittens::warp::zero(l_acc);
    const float scale = 1.0f / sqrtf((float)D);

    int num_full_blocks = len / LLMK_SM120_PAGED_PAGE_BLOCK;
    for (int t_block = 0; t_block < num_full_blocks; ++t_block) {
        int logical_pos = t_block * LLMK_SM120_PAGED_PAGE_BLOCK;
        int page_idx    = logical_pos / g.page_size;
        int within_page = (logical_pos % g.page_size) / LLMK_SM120_PAGED_PAGE_BLOCK;
        int page_id     = g.block_table[batch * g.max_blocks + page_idx];
        if (page_id < 0) break;

        kv_rt K, V;
        ::kittens::warp::load(K, g.pages_k, {page_id, hk, within_page, 0});
        ::kittens::warp::load(V, g.pages_v, {page_id, hk, within_page, 0});

        s_rt S;
        ::kittens::warp::zero(S);
        ::kittens::warp::mma_ABt(S, Q, K, S);
        ::kittens::warp::mul(S, S, scale);

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
    ::kittens::warp::store(g.out, O, {batch, head, 0, 0});
}

template <int D>
inline void launch_decode(bf16* out, const bf16* q, const bf16* pages_k, const bf16* pages_v,
                          const int* block_table, const int* cache_len,
                          int B, int H, int Hk, int page_size, int max_blocks,
                          int num_pages, cudaStream_t stream) {
    using G = globals<D>;
    typename G::q_gl  q_arg  {const_cast<bf16*>(q),       (unsigned)B,   (unsigned)H,        1u, (unsigned)D};
    typename G::kv_gl pk_arg {const_cast<bf16*>(pages_k), (unsigned)num_pages, (unsigned)Hk, (unsigned)page_size, (unsigned)D};
    typename G::kv_gl pv_arg {const_cast<bf16*>(pages_v), (unsigned)num_pages, (unsigned)Hk, (unsigned)page_size, (unsigned)D};
    typename G::o_gl  o_arg  {out, (unsigned)B, (unsigned)H, 1u, (unsigned)D};
    G g{q_arg, pk_arg, pv_arg, o_arg, block_table, cache_len, H, Hk, page_size, max_blocks};
    dim3 grid(1, H, B);
    paged_attention_decode_kernel<D><<<grid, ::kittens::WARP_THREADS, 0, stream>>>(g);
    cudaCheck(cudaGetLastError());
}

inline void launch_decode_dispatch(bf16* out, const bf16* q, const bf16* pages_k, const bf16* pages_v,
                                   const int* block_table, const int* cache_len,
                                   int B, int H, int Hk, int D, int page_size, int max_blocks,
                                   int num_pages, cudaStream_t stream) {
    if (D == 64)       launch_decode<64> (out, q, pages_k, pages_v, block_table, cache_len, B, H, Hk, page_size, max_blocks, num_pages, stream);
    else if (D == 128) launch_decode<128>(out, q, pages_k, pages_v, block_table, cache_len, B, H, Hk, page_size, max_blocks, num_pages, stream);
    else {
        fprintf(stderr, "paged_attention_sm120: D must be 64 or 128 (got %d)\n", D);
        exit(EXIT_FAILURE);
    }
}

}  // namespace llmk::paged_attention
