/*
kv_cache_sm120.cuh — ThunderKittens KV-cache append/concat helpers for SM120.

These are pure data-movement kernels but written in the TK DSL so they sit
inside the same memory-pipeline as the attention kernels and can chain
without going through the host stack.

  - kv_cache_append: copy current K, V into cache_k, cache_v at per-batch
                     offsets cache_len, then bump cache_len.
  - kv_cache_concat: produce a contiguous concatenation of (cache, current)
                     into a fresh output.
*/
#pragma once

#include "tk_common.cuh"

namespace llmk::kv_cache_tk {

using namespace ::kittens;

#ifndef LLMK_SM120_KV_BLOCK
#define LLMK_SM120_KV_BLOCK 32
#endif
constexpr int KV_BLOCK = LLMK_SM120_KV_BLOCK;

template <int D>
struct append_globals {
    using cur_tile  = st_bf<KV_BLOCK, D>;
    using cache_tile= st_bf<KV_BLOCK, D>;
    using cur_gl    = gl<bf16, -1, -1, -1, -1, cur_tile>;
    using cache_gl  = gl<bf16, -1, -1, -1, -1, cache_tile>;
    cur_gl   cur_k;
    cur_gl   cur_v;
    cache_gl cache_k;
    cache_gl cache_v;
    int*     cache_len;
    int      S_new;
    int      Hk;
    int      S_cap;
};

template <int D>
__global__ void kv_cache_append_kernel(const __grid_constant__ append_globals<D> g) {
    int batch = blockIdx.z;
    int hk    = blockIdx.y;
    int sb    = blockIdx.x;   // block of S_new
    if (sb * KV_BLOCK >= g.S_new) return;

    using rt = rt_bf<KV_BLOCK, D, ducks::rt_layout::row>;
    rt K, V;
    ::kittens::warp::load(K, g.cur_k, {batch, hk, sb, 0});
    ::kittens::warp::load(V, g.cur_v, {batch, hk, sb, 0});

    int offset_blocks = g.cache_len[batch] / KV_BLOCK;
    int dst_block = offset_blocks + sb;
    if (dst_block * KV_BLOCK + KV_BLOCK > g.S_cap) return;

    ::kittens::warp::store(g.cache_k, K, {batch, hk, dst_block, 0});
    ::kittens::warp::store(g.cache_v, V, {batch, hk, dst_block, 0});

    if (threadIdx.x == 0 && hk == 0 && sb == 0) {
        atomicAdd(&g.cache_len[batch], g.S_new);
    }
}

template <int D>
inline void launch_append(bf16* cache_k, bf16* cache_v,
                          const bf16* current_k, const bf16* current_v,
                          int* cache_len, int B, int Hk, int S_cap, int S_new,
                          cudaStream_t stream) {
    assert(S_new % KV_BLOCK == 0);
    assert(S_cap % KV_BLOCK == 0);
    using G = append_globals<D>;
    typename G::cur_gl   cur_k_arg  {const_cast<bf16*>(current_k), (unsigned)B, (unsigned)Hk, (unsigned)S_new, (unsigned)D};
    typename G::cur_gl   cur_v_arg  {const_cast<bf16*>(current_v), (unsigned)B, (unsigned)Hk, (unsigned)S_new, (unsigned)D};
    typename G::cache_gl cache_k_arg{cache_k,                      (unsigned)B, (unsigned)Hk, (unsigned)S_cap, (unsigned)D};
    typename G::cache_gl cache_v_arg{cache_v,                      (unsigned)B, (unsigned)Hk, (unsigned)S_cap, (unsigned)D};
    G g{cur_k_arg, cur_v_arg, cache_k_arg, cache_v_arg, cache_len, S_new, Hk, S_cap};
    dim3 grid(S_new / KV_BLOCK, Hk, B);
    kv_cache_append_kernel<D><<<grid, ::kittens::WARP_THREADS, 0, stream>>>(g);
}

// ----------------------------------------------------------------------------
// Concatenate (cache, current) -> out. cache_len[b] is the prefix length per row.
// ----------------------------------------------------------------------------

template <int D>
struct concat_globals {
    using tile = st_bf<KV_BLOCK, D>;
    using gl_t = gl<bf16, -1, -1, -1, -1, tile>;
    gl_t cache_k; gl_t cache_v; gl_t cur_k; gl_t cur_v;
    gl_t out_k;  gl_t out_v;
    int* cache_len;
    int  S_new;
    int  S_cap;
};

template <int D>
__global__ void kv_cache_concat_kernel(const __grid_constant__ concat_globals<D> g) {
    int batch = blockIdx.z;
    int hk    = blockIdx.y;
    int sb    = blockIdx.x;

    using rt = rt_bf<KV_BLOCK, D, ducks::rt_layout::row>;
    int cache_blocks = g.cache_len[batch] / KV_BLOCK;
    if (sb < cache_blocks) {
        rt K, V;
        ::kittens::warp::load(K, g.cache_k, {batch, hk, sb, 0});
        ::kittens::warp::load(V, g.cache_v, {batch, hk, sb, 0});
        ::kittens::warp::store(g.out_k, K, {batch, hk, sb, 0});
        ::kittens::warp::store(g.out_v, V, {batch, hk, sb, 0});
    } else {
        int new_sb = sb - cache_blocks;
        if (new_sb * KV_BLOCK >= g.S_new) return;
        rt K, V;
        ::kittens::warp::load(K, g.cur_k, {batch, hk, new_sb, 0});
        ::kittens::warp::load(V, g.cur_v, {batch, hk, new_sb, 0});
        ::kittens::warp::store(g.out_k, K, {batch, hk, sb, 0});
        ::kittens::warp::store(g.out_v, V, {batch, hk, sb, 0});
    }
}

template <int D>
inline void launch_concat(bf16* out_k, bf16* out_v,
                          const bf16* cache_k, const bf16* cache_v,
                          const bf16* current_k, const bf16* current_v,
                          int* cache_len, int B, int Hk, int S_cap, int S_new,
                          cudaStream_t stream) {
    assert(S_new % KV_BLOCK == 0);
    assert(S_cap % KV_BLOCK == 0);
    using G = concat_globals<D>;
    typename G::gl_t cache_k_arg{const_cast<bf16*>(cache_k),   (unsigned)B, (unsigned)Hk, (unsigned)S_cap, (unsigned)D};
    typename G::gl_t cache_v_arg{const_cast<bf16*>(cache_v),   (unsigned)B, (unsigned)Hk, (unsigned)S_cap, (unsigned)D};
    typename G::gl_t cur_k_arg  {const_cast<bf16*>(current_k), (unsigned)B, (unsigned)Hk, (unsigned)S_new, (unsigned)D};
    typename G::gl_t cur_v_arg  {const_cast<bf16*>(current_v), (unsigned)B, (unsigned)Hk, (unsigned)S_new, (unsigned)D};
    typename G::gl_t out_k_arg  {out_k,                        (unsigned)B, (unsigned)Hk, (unsigned)(S_cap + S_new), (unsigned)D};
    typename G::gl_t out_v_arg  {out_v,                        (unsigned)B, (unsigned)Hk, (unsigned)(S_cap + S_new), (unsigned)D};
    G g{cache_k_arg, cache_v_arg, cur_k_arg, cur_v_arg, out_k_arg, out_v_arg, cache_len, S_new, S_cap};
    int total_blocks = (S_cap + S_new) / KV_BLOCK;
    dim3 grid(total_blocks, Hk, B);
    kv_cache_concat_kernel<D><<<grid, ::kittens::WARP_THREADS, 0, stream>>>(g);
}

}  // namespace llmk::kv_cache_tk
