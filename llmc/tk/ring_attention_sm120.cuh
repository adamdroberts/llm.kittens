/*
ring_attention_sm120.cuh — ThunderKittens ring attention local step for SM120.

Context parallelism: the sequence is sharded across N ranks. Each rank holds
Q, K, V for its shard. To compute full attention each rank rotates its (K, V)
shard around the ring, performing local FA-2 against the current K, V chunk
each step. After N rotations every Q has seen every K, V.

This kernel implements one local step (the per-shard partial attention) with
log-sum-exp accumulation across steps. The host orchestrates the rotation
(via NCCL send/recv in `distributed_ext.cu::cp_ring_send_recv`).

Inputs:
  q_local:   [B, H, S_local, D]
  k_shard:   [B, H, S_shard, D]  (the currently-rotated K shard)
  v_shard:   [B, H, S_shard, D]
  causal_mask: optional bf16 [S_local, S_shard]
Outputs:
  o_acc:     [B, H, S_local, D] running output
  lse_acc:   [B, H, S_local] running log-sum-exp

For each step, the kernel merges the local partial result with the running
state using the log-sum-exp trick.
*/
#pragma once

#include "tk_common.cuh"
#include <cmath>

namespace llmk::ring_attention {

using namespace ::kittens;

#ifndef LLMK_SM120_RING_BLOCK
#define LLMK_SM120_RING_BLOCK 32
#endif

template <int D>
struct globals {
    using tile     = st_bf<LLMK_SM120_RING_BLOCK, D>;
    using lse_vec  = sv<float, LLMK_SM120_RING_BLOCK>;
    using gl_t     = gl<bf16, -1, -1, -1, -1, tile>;
    using lse_gl   = gl<float, -1, -1, -1, -1, lse_vec>;

    gl_t   q;
    gl_t   k_shard;
    gl_t   v_shard;
    gl_t   o_acc;
    lse_gl lse_acc;
    int    H, S_shard;
    bool   first_step;   // if true, initialise o_acc / lse_acc from scratch
};

template <int D>
__global__ void ring_local_step_kernel(const __grid_constant__ globals<D> g) {
    int batch = blockIdx.z;
    int head  = blockIdx.y;
    int q_blk = blockIdx.x;

    using q_rt    = rt_bf<LLMK_SM120_RING_BLOCK, D, ducks::rt_layout::row>;
    using k_rt    = rt_bf<LLMK_SM120_RING_BLOCK, D, ducks::rt_layout::row>;
    using v_rt    = rt_bf<LLMK_SM120_RING_BLOCK, D, ducks::rt_layout::row>;
    using s_rt    = rt_fl<LLMK_SM120_RING_BLOCK, LLMK_SM120_RING_BLOCK, ducks::rt_layout::row>;
    using p_rt    = rt_bf<LLMK_SM120_RING_BLOCK, LLMK_SM120_RING_BLOCK, ducks::rt_layout::row>;
    using o_rt    = rt_fl<LLMK_SM120_RING_BLOCK, D, ducks::rt_layout::row>;
    using row_cv  = typename s_rt::col_vec;
    using o_cv    = typename o_rt::col_vec;

    q_rt Q;
    ::kittens::warp::load(Q, g.q, {batch, head, q_blk, 0});

    o_rt O_local;
    ::kittens::warp::zero(O_local);
    row_cv m_local, l_local;
    ::kittens::warp::neg_infty(m_local);
    ::kittens::warp::zero(l_local);

    const float scale = 1.0f / sqrtf((float)D);
    int num_k_blocks = g.S_shard / LLMK_SM120_RING_BLOCK;

    // Compute local-shard attention (FA-2 walk over K shard).
    for (int kb = 0; kb < num_k_blocks; ++kb) {
        k_rt K; v_rt V;
        ::kittens::warp::load(K, g.k_shard, {batch, head, kb, 0});
        ::kittens::warp::load(V, g.v_shard, {batch, head, kb, 0});

        s_rt S;
        ::kittens::warp::zero(S);
        ::kittens::warp::mma_ABt(S, Q, K, S);
        ::kittens::warp::mul(S, S, scale);

        row_cv m_new; ::kittens::warp::row_max(m_new, S, m_local);
        row_cv rescale; ::kittens::warp::sub(rescale, m_local, m_new); ::kittens::warp::exp(rescale, rescale);
        ::kittens::warp::sub_row(S, S, m_new); ::kittens::warp::exp(S, S);
        row_cv bs; ::kittens::warp::row_sum(bs, S);
        ::kittens::warp::mul(l_local, l_local, rescale);
        ::kittens::warp::add(l_local, l_local, bs);

        p_rt P; ::kittens::warp::copy(P, S);
        ::kittens::warp::mul_row(O_local, O_local, rescale);
        auto& V_col = ::kittens::warp::swap_layout_inplace(V);
        ::kittens::warp::mma_AB(O_local, P, V_col, O_local);
        ::kittens::warp::copy(m_local, m_new);
    }

    // Merge with running (O_acc, lse_acc) using log-sum-exp combine.
    // lse_local = m_local + log(l_local).
    row_cv lse_local;
    ::kittens::warp::log(lse_local, l_local);
    ::kittens::warp::add(lse_local, lse_local, m_local);

    if (g.first_step) {
        // Initialise running state.
        ::kittens::warp::div_row(O_local, O_local, l_local);
        ::kittens::warp::store(g.o_acc, O_local, {batch, head, q_blk, 0});
        ::kittens::warp::store(g.lse_acc, lse_local, {batch, head, 0, q_blk});
        return;
    }

    // Load running output / lse and merge:
    //   lse_new = log(exp(lse_old) + exp(lse_local))
    //           = max(lse_old, lse_local) + log(1 + exp(-|lse_old - lse_local|))
    //   alpha   = exp(lse_old - lse_new)
    //   beta    = exp(lse_local - lse_new)
    //   O_new   = alpha * O_old + beta * (O_local / l_local)

    o_rt O_old;
    ::kittens::warp::load(O_old, g.o_acc, {batch, head, q_blk, 0});
    row_cv lse_old;
    ::kittens::warp::load(lse_old, g.lse_acc, {batch, head, 0, q_blk});

    row_cv lse_max;
    ::kittens::warp::max(lse_max, lse_old, lse_local);
    row_cv diff_old, diff_loc;
    ::kittens::warp::sub(diff_old, lse_old,   lse_max);
    ::kittens::warp::sub(diff_loc, lse_local, lse_max);
    ::kittens::warp::exp(diff_old, diff_old);
    ::kittens::warp::exp(diff_loc, diff_loc);
    row_cv sum_d;
    ::kittens::warp::add(sum_d, diff_old, diff_loc);
    row_cv log_sum;
    ::kittens::warp::log(log_sum, sum_d);
    row_cv lse_new;
    ::kittens::warp::add(lse_new, lse_max, log_sum);

    row_cv alpha, beta;
    ::kittens::warp::sub(alpha, lse_old,   lse_new); ::kittens::warp::exp(alpha, alpha);
    ::kittens::warp::sub(beta,  lse_local, lse_new); ::kittens::warp::exp(beta,  beta);

    // O_local_norm = O_local / l_local
    o_rt O_loc_norm;
    ::kittens::warp::copy(O_loc_norm, O_local);
    ::kittens::warp::div_row(O_loc_norm, O_loc_norm, l_local);

    ::kittens::warp::mul_row(O_old,      O_old,      alpha);
    ::kittens::warp::mul_row(O_loc_norm, O_loc_norm, beta);
    ::kittens::warp::add(O_old, O_old, O_loc_norm);

    ::kittens::warp::store(g.o_acc, O_old, {batch, head, q_blk, 0});
    ::kittens::warp::store(g.lse_acc, lse_new, {batch, head, 0, q_blk});
}

template <int D>
inline void launch_local_step(bf16* o_acc, float* lse_acc,
                              const bf16* q, const bf16* k_shard, const bf16* v_shard,
                              int B, int H, int S_local, int S_shard, bool first_step,
                              cudaStream_t stream) {
    assert(S_local % LLMK_SM120_RING_BLOCK == 0);
    assert(S_shard % LLMK_SM120_RING_BLOCK == 0);
    using G = globals<D>;
    typename G::gl_t q_arg  {const_cast<bf16*>(q),       (unsigned)B, (unsigned)H, (unsigned)S_local, (unsigned)D};
    typename G::gl_t k_arg  {const_cast<bf16*>(k_shard), (unsigned)B, (unsigned)H, (unsigned)S_shard, (unsigned)D};
    typename G::gl_t v_arg  {const_cast<bf16*>(v_shard), (unsigned)B, (unsigned)H, (unsigned)S_shard, (unsigned)D};
    typename G::gl_t o_arg  {o_acc,                       (unsigned)B, (unsigned)H, (unsigned)S_local, (unsigned)D};
    typename G::lse_gl l_arg{lse_acc,                     (unsigned)B, (unsigned)H, 1u, (unsigned)S_local};
    G g{q_arg, k_arg, v_arg, o_arg, l_arg, H, S_shard, first_step};
    dim3 grid(S_local / LLMK_SM120_RING_BLOCK, H, B);
    ring_local_step_kernel<D><<<grid, ::kittens::WARP_THREADS, 0, stream>>>(g);
    cudaCheck(cudaGetLastError());
}

inline void launch_local_step_dispatch(bf16* o_acc, float* lse_acc,
                                       const bf16* q, const bf16* k_shard, const bf16* v_shard,
                                       int B, int H, int S_local, int S_shard, int D, bool first_step,
                                       cudaStream_t stream) {
    if (D == 64)       launch_local_step<64> (o_acc, lse_acc, q, k_shard, v_shard, B, H, S_local, S_shard, first_step, stream);
    else if (D == 128) launch_local_step<128>(o_acc, lse_acc, q, k_shard, v_shard, B, H, S_local, S_shard, first_step, stream);
    else { fprintf(stderr, "ring_attention_sm120: D must be 64 or 128\n"); exit(EXIT_FAILURE); }
}

}  // namespace llmk::ring_attention
