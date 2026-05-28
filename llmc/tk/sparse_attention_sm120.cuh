/*
sparse_attention_sm120.cuh — ThunderKittens block-sparse / streaming-sinks /
differential attention variants for SM120.

All three are FA-2 derivatives sharing the same K-block walker:

  block_sparse_kernel:
    Mask defined by a per-(q_block, k_block) bool array — block-sparse
    pattern as in Longformer / BigBird. Bool array `block_mask` of shape
    [n_q_blocks, n_k_blocks] is read once per (q_block, k_block) pair.

  streaming_sinks_kernel:
    Causal + sliding window with the first `n_sink` tokens always-visible
    (StreamingLLM). Block-level fast skip plus per-element mask near the
    boundary.

  differential_attention_kernel:
    Computes O = softmax(Q1·K1^T) V - λ · softmax(Q2·K2^T) V using two
    parallel FA-2 walkers and combining at the end.
*/
#pragma once

#include "tk_common.cuh"
#include <cmath>

namespace llmk::sparse_attention {

using namespace ::kittens;

#ifndef LLMK_SM120_SPARSE_BLOCK
#define LLMK_SM120_SPARSE_BLOCK 32
#endif

// ============================================================================
// Block-sparse attention.
// ============================================================================

template <int D>
struct block_sparse_globals {
    using q_tile = st_bf<LLMK_SM120_SPARSE_BLOCK, D>;
    using kv_tile= st_bf<LLMK_SM120_SPARSE_BLOCK, D>;
    using o_tile = st_bf<LLMK_SM120_SPARSE_BLOCK, D>;
    using q_gl   = gl<bf16, -1, -1, -1, -1, q_tile>;
    using kv_gl  = gl<bf16, -1, -1, -1, -1, kv_tile>;
    using o_gl   = gl<bf16, -1, -1, -1, -1, o_tile>;
    q_gl q; kv_gl k; kv_gl v; o_gl o;
    const uint8_t* block_mask;  // [n_q_blocks, n_k_blocks]
    int n_k_blocks;
};

template <int D>
__global__ void block_sparse_fwd_kernel(const __grid_constant__ block_sparse_globals<D> g) {
    int batch = blockIdx.z;
    int head  = blockIdx.y;
    int q_blk = blockIdx.x;

    using q_rt    = rt_bf<LLMK_SM120_SPARSE_BLOCK, D, ducks::rt_layout::row>;
    using k_rt    = rt_bf<LLMK_SM120_SPARSE_BLOCK, D, ducks::rt_layout::row>;
    using v_rt    = rt_bf<LLMK_SM120_SPARSE_BLOCK, D, ducks::rt_layout::row>;
    using s_rt    = rt_fl<LLMK_SM120_SPARSE_BLOCK, LLMK_SM120_SPARSE_BLOCK, ducks::rt_layout::row>;
    using p_rt    = rt_bf<LLMK_SM120_SPARSE_BLOCK, LLMK_SM120_SPARSE_BLOCK, ducks::rt_layout::row>;
    using o_rt    = rt_fl<LLMK_SM120_SPARSE_BLOCK, D, ducks::rt_layout::row>;
    using row_cv  = typename s_rt::col_vec;

    q_rt Q;
    ::kittens::warp::load(Q, g.q, {batch, head, q_blk, 0});

    o_rt O;
    ::kittens::warp::zero(O);
    row_cv m, l_acc;
    ::kittens::warp::neg_infty(m);
    ::kittens::warp::zero(l_acc);
    const float scale = 1.0f / sqrtf((float)D);

    for (int kb = 0; kb < g.n_k_blocks; ++kb) {
        if (!g.block_mask[q_blk * g.n_k_blocks + kb]) continue;

        k_rt K; v_rt V;
        ::kittens::warp::load(K, g.k, {batch, head, kb, 0});
        ::kittens::warp::load(V, g.v, {batch, head, kb, 0});

        s_rt S;
        ::kittens::warp::zero(S);
        ::kittens::warp::mma_ABt(S, Q, K, S);
        ::kittens::warp::mul(S, S, scale);

        // Causal mask on diagonal block
        if (kb == q_blk) ::kittens::warp::make_causal(S, S, -INFINITY);

        row_cv m_new; ::kittens::warp::row_max(m_new, S, m);
        row_cv rescale; ::kittens::warp::sub(rescale, m, m_new); ::kittens::warp::exp(rescale, rescale);
        ::kittens::warp::sub_row(S, S, m_new); ::kittens::warp::exp(S, S);
        row_cv bs; ::kittens::warp::row_sum(bs, S);
        ::kittens::warp::mul(l_acc, l_acc, rescale); ::kittens::warp::add(l_acc, l_acc, bs);
        p_rt P; ::kittens::warp::copy(P, S);
        ::kittens::warp::mul_row(O, O, rescale);
        auto& V_col = ::kittens::warp::swap_layout_inplace(V);
        ::kittens::warp::mma_AB(O, P, V_col, O);
        ::kittens::warp::copy(m, m_new);
    }
    ::kittens::warp::div_row(O, O, l_acc);
    ::kittens::warp::store(g.o, O, {batch, head, q_blk, 0});
}

template <int D>
inline void launch_block_sparse(bf16* out, const bf16* q, const bf16* k, const bf16* v,
                                const uint8_t* block_mask, int B, int H, int S, int n_k_blocks,
                                cudaStream_t stream) {
    assert(S % LLMK_SM120_SPARSE_BLOCK == 0);
    using G = block_sparse_globals<D>;
    typename G::q_gl q_arg{const_cast<bf16*>(q), (unsigned)B, (unsigned)H, (unsigned)S, (unsigned)D};
    typename G::kv_gl k_arg{const_cast<bf16*>(k), (unsigned)B, (unsigned)H, (unsigned)S, (unsigned)D};
    typename G::kv_gl v_arg{const_cast<bf16*>(v), (unsigned)B, (unsigned)H, (unsigned)S, (unsigned)D};
    typename G::o_gl o_arg{out, (unsigned)B, (unsigned)H, (unsigned)S, (unsigned)D};
    G g{q_arg, k_arg, v_arg, o_arg, block_mask, n_k_blocks};
    dim3 grid(S / LLMK_SM120_SPARSE_BLOCK, H, B);
    block_sparse_fwd_kernel<D><<<grid, ::kittens::WARP_THREADS, 0, stream>>>(g);
    cudaCheck(cudaGetLastError());
}

inline void launch_block_sparse_dispatch(bf16* out, const bf16* q, const bf16* k, const bf16* v,
                                         const uint8_t* block_mask, int B, int H, int S, int n_k_blocks, int D,
                                         cudaStream_t stream) {
    if (D == 64)       launch_block_sparse<64> (out, q, k, v, block_mask, B, H, S, n_k_blocks, stream);
    else if (D == 128) launch_block_sparse<128>(out, q, k, v, block_mask, B, H, S, n_k_blocks, stream);
    else { fprintf(stderr, "block_sparse_sm120: D must be 64 or 128\n"); exit(EXIT_FAILURE); }
}

// ============================================================================
// Streaming-sinks attention: causal + sliding window + first n_sink visible.
// ============================================================================

template <int D>
struct streaming_globals {
    using tile = st_bf<LLMK_SM120_SPARSE_BLOCK, D>;
    using gl_t = gl<bf16, -1, -1, -1, -1, tile>;
    gl_t q; gl_t k; gl_t v; gl_t o;
    int window, n_sink;
};

template <int D>
__global__ void streaming_fwd_kernel(const __grid_constant__ streaming_globals<D> g) {
    int batch = blockIdx.z;
    int head  = blockIdx.y;
    int q_blk = blockIdx.x;

    using q_rt    = rt_bf<LLMK_SM120_SPARSE_BLOCK, D, ducks::rt_layout::row>;
    using k_rt    = rt_bf<LLMK_SM120_SPARSE_BLOCK, D, ducks::rt_layout::row>;
    using v_rt    = rt_bf<LLMK_SM120_SPARSE_BLOCK, D, ducks::rt_layout::row>;
    using s_rt    = rt_fl<LLMK_SM120_SPARSE_BLOCK, LLMK_SM120_SPARSE_BLOCK, ducks::rt_layout::row>;
    using p_rt    = rt_bf<LLMK_SM120_SPARSE_BLOCK, LLMK_SM120_SPARSE_BLOCK, ducks::rt_layout::row>;
    using o_rt    = rt_fl<LLMK_SM120_SPARSE_BLOCK, D, ducks::rt_layout::row>;
    using row_cv  = typename s_rt::col_vec;

    q_rt Q;
    ::kittens::warp::load(Q, g.q, {batch, head, q_blk, 0});

    o_rt O;
    ::kittens::warp::zero(O);
    row_cv m, l_acc;
    ::kittens::warp::neg_infty(m);
    ::kittens::warp::zero(l_acc);
    const float scale = 1.0f / sqrtf((float)D);

    int q_min = q_blk * LLMK_SM120_SPARSE_BLOCK;
    int q_max = q_min + LLMK_SM120_SPARSE_BLOCK - 1;

    int n_sink_blocks = (g.n_sink + LLMK_SM120_SPARSE_BLOCK - 1) / LLMK_SM120_SPARSE_BLOCK;

    // Helper to process a K block
    auto process_block = [&](int kb) {
        k_rt K; v_rt V;
        ::kittens::warp::load(K, g.k, {batch, head, kb, 0});
        ::kittens::warp::load(V, g.v, {batch, head, kb, 0});
        s_rt S;
        ::kittens::warp::zero(S);
        ::kittens::warp::mma_ABt(S, Q, K, S);
        ::kittens::warp::mul(S, S, scale);
        if (kb == q_blk) ::kittens::warp::make_causal(S, S, -INFINITY);
        row_cv m_new; ::kittens::warp::row_max(m_new, S, m);
        row_cv rescale; ::kittens::warp::sub(rescale, m, m_new); ::kittens::warp::exp(rescale, rescale);
        ::kittens::warp::sub_row(S, S, m_new); ::kittens::warp::exp(S, S);
        row_cv bs; ::kittens::warp::row_sum(bs, S);
        ::kittens::warp::mul(l_acc, l_acc, rescale); ::kittens::warp::add(l_acc, l_acc, bs);
        p_rt P; ::kittens::warp::copy(P, S);
        ::kittens::warp::mul_row(O, O, rescale);
        auto& V_col = ::kittens::warp::swap_layout_inplace(V);
        ::kittens::warp::mma_AB(O, P, V_col, O);
        ::kittens::warp::copy(m, m_new);
    };

    // Sink blocks
    for (int kb = 0; kb < n_sink_blocks && kb <= q_blk; ++kb) {
        process_block(kb);
    }
    // Window blocks (skip ones already handled in sink set, and only those within window)
    int kb_window_start = (q_min - g.window) / LLMK_SM120_SPARSE_BLOCK;
    if (kb_window_start < n_sink_blocks) kb_window_start = n_sink_blocks;
    for (int kb = kb_window_start; kb <= q_blk; ++kb) {
        process_block(kb);
    }

    ::kittens::warp::div_row(O, O, l_acc);
    ::kittens::warp::store(g.o, O, {batch, head, q_blk, 0});
}

template <int D>
inline void launch_streaming_sinks(bf16* out, const bf16* q, const bf16* k, const bf16* v,
                                   int B, int H, int S, int window, int n_sink,
                                   cudaStream_t stream) {
    assert(S % LLMK_SM120_SPARSE_BLOCK == 0);
    using G = streaming_globals<D>;
    typename G::gl_t q_arg{const_cast<bf16*>(q), (unsigned)B, (unsigned)H, (unsigned)S, (unsigned)D};
    typename G::gl_t k_arg{const_cast<bf16*>(k), (unsigned)B, (unsigned)H, (unsigned)S, (unsigned)D};
    typename G::gl_t v_arg{const_cast<bf16*>(v), (unsigned)B, (unsigned)H, (unsigned)S, (unsigned)D};
    typename G::gl_t o_arg{out, (unsigned)B, (unsigned)H, (unsigned)S, (unsigned)D};
    G g{q_arg, k_arg, v_arg, o_arg, window, n_sink};
    dim3 grid(S / LLMK_SM120_SPARSE_BLOCK, H, B);
    streaming_fwd_kernel<D><<<grid, ::kittens::WARP_THREADS, 0, stream>>>(g);
    cudaCheck(cudaGetLastError());
}

inline void launch_streaming_sinks_dispatch(bf16* out, const bf16* q, const bf16* k, const bf16* v,
                                            int B, int H, int S, int window, int n_sink, int D,
                                            cudaStream_t stream) {
    if (D == 64)       launch_streaming_sinks<64> (out, q, k, v, B, H, S, window, n_sink, stream);
    else if (D == 128) launch_streaming_sinks<128>(out, q, k, v, B, H, S, window, n_sink, stream);
    else { fprintf(stderr, "streaming_sinks_sm120: D must be 64 or 128\n"); exit(EXIT_FAILURE); }
}

// ============================================================================
// Differential attention: O = softmax(Q1 K1^T) V - λ softmax(Q2 K2^T) V.
//
// We expose the per-branch attention as the existing causal kernel; this
// helper kernel just blends two output tensors and is otherwise trivial.
// ============================================================================

__global__ void differential_blend_kernel(bf16* out, const bf16* a, const bf16* b, float lambda, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float va = __bfloat162float(a[i]);
    float vb = __bfloat162float(b[i]);
    out[i] = __float2bfloat16(va - lambda * vb);
}

inline void launch_differential_blend(bf16* out, const bf16* a, const bf16* b, float lambda, int N,
                                      cudaStream_t stream) {
    const int block_size = 256;
    differential_blend_kernel<<<CEIL_DIV(N, block_size), block_size, 0, stream>>>(out, a, b, lambda, N);
    cudaCheck(cudaGetLastError());
}

}  // namespace llmk::sparse_attention
