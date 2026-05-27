/*
mla_sm120.cuh — ThunderKittens Multi-Latent Attention (DeepSeek-V2/V3) for SM120.

MLA stores K, V as a low-rank compressed representation kv_c ∈ R^{c_dim} per
token. At attention time we re-expand:
    K = kv_c @ W_K_up^T   (per head)
    V = kv_c @ W_V_up^T   (per head)

We expose two entry points:
  * mla_expand_kv   — reconstruct full K, V tiles from compressed kv_c.
  * mla_attention_decode — fused: expand on the fly, run FA-2 style decode.

The fused decode reads kv_c per-token, multiplies by the per-head up-projection
inside the attention loop, avoiding the materialised K/V tensors. This is the
core MLA inference win.
*/
#pragma once

#include "tk_common.cuh"
#include <cmath>

namespace llmk::mla {

using namespace ::kittens;

#ifndef LLMK_SM120_MLA_BLOCK
#define LLMK_SM120_MLA_BLOCK 16
#endif

// kv_c shape: [B, T, c_dim]
// w_k_up:    [H, D, c_dim]  -> per-head K up-projection weight
// w_v_up:    [H, D, c_dim]
// q:         [B, H, S_q, D]
// out:       [B, H, S_q, D]
template <int D, int C_DIM>
struct decode_globals {
    using q_tile  = st_bf<1, D>;
    using kvc_tile= st_bf<LLMK_SM120_MLA_BLOCK, C_DIM>;
    using wup_tile= st_bf<D, C_DIM>;
    using o_tile  = st_bf<1, D>;
    using q_gl    = gl<bf16, -1, -1, -1, -1, q_tile>;
    using kvc_gl  = gl<bf16, -1, -1, -1, -1, kvc_tile>;
    using w_gl    = gl<bf16, -1, -1, -1, -1, wup_tile>;
    using o_gl    = gl<bf16, -1, -1, -1, -1, o_tile>;

    q_gl   q;
    kvc_gl kv_c;
    w_gl   w_k_up;
    w_gl   w_v_up;
    o_gl   out;
    int    T;
    int    H;
};

template <int D, int C_DIM>
__global__ void mla_decode_kernel(const __grid_constant__ decode_globals<D, C_DIM> g) {
    int batch = blockIdx.z;
    int head  = blockIdx.y;
    int t     = threadIdx.x % g.T;  // simplified — caller pads T to warp

    using q_rt    = rt_bf<1, D, ducks::rt_layout::row>;
    using kvc_rt  = rt_bf<LLMK_SM120_MLA_BLOCK, C_DIM, ducks::rt_layout::row>;
    using wup_rt  = rt_bf<D, C_DIM, ducks::rt_layout::row>;

    q_rt Q;
    ::kittens::warp::load(Q, g.q, {batch, head, 0, 0});

    wup_rt WK, WV;
    ::kittens::warp::load(WK, g.w_k_up, {0, 0, head, 0});
    ::kittens::warp::load(WV, g.w_v_up, {0, 0, head, 0});

    using s_rt = rt_fl<1, LLMK_SM120_MLA_BLOCK, ducks::rt_layout::row>;
    using p_rt = rt_bf<1, LLMK_SM120_MLA_BLOCK, ducks::rt_layout::row>;
    using o_rt = rt_fl<1, D, ducks::rt_layout::row>;
    using row_cv = typename s_rt::col_vec;

    o_rt O;
    ::kittens::warp::zero(O);
    row_cv m, l_acc;
    ::kittens::warp::neg_infty(m);
    ::kittens::warp::zero(l_acc);

    float scale = 1.0f / sqrtf((float)D);

    int num_blocks = g.T / LLMK_SM120_MLA_BLOCK;
    for (int tb = 0; tb < num_blocks; ++tb) {
        kvc_rt KV_c;
        ::kittens::warp::load(KV_c, g.kv_c, {batch, 0, tb, 0});

        // Reconstruct K and V tiles by multiplying KV_c with the per-head
        // up-projection. K = KV_c @ WK^T (16, C) @ (C, D) -> (16, D)
        // We assemble K and V on the fly into bf16 register tiles.
        // Use mma_ABt: KV_c @ WK^T  (treating WK as (D, C)).
        using k_rt = rt_bf<LLMK_SM120_MLA_BLOCK, D, ducks::rt_layout::row>;
        using v_rt = rt_bf<LLMK_SM120_MLA_BLOCK, D, ducks::rt_layout::row>;
        using kfl  = rt_fl<LLMK_SM120_MLA_BLOCK, D, ducks::rt_layout::row>;
        kfl K_fl, V_fl;
        ::kittens::warp::zero(K_fl);
        ::kittens::warp::zero(V_fl);
        ::kittens::warp::mma_ABt(K_fl, KV_c, WK, K_fl);
        ::kittens::warp::mma_ABt(V_fl, KV_c, WV, V_fl);
        k_rt K; v_rt V;
        ::kittens::warp::copy(K, K_fl);
        ::kittens::warp::copy(V, V_fl);

        // S = Q · K^T
        s_rt S;
        ::kittens::warp::zero(S);
        ::kittens::warp::mma_ABt(S, Q, K, S);
        ::kittens::warp::mul(S, S, scale);

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
    ::kittens::warp::store(g.out, O, {batch, head, 0, 0});
}

template <int D, int C_DIM>
inline void launch_decode(bf16* out, const bf16* q, const bf16* kv_c,
                          const bf16* w_k_up, const bf16* w_v_up,
                          int B, int H, int T, cudaStream_t stream) {
    assert(T % LLMK_SM120_MLA_BLOCK == 0);
    using G = decode_globals<D, C_DIM>;
    typename G::q_gl  q_arg {const_cast<bf16*>(q),     (unsigned)B, (unsigned)H, 1u, (unsigned)D};
    typename G::kvc_gl kv_arg{const_cast<bf16*>(kv_c), (unsigned)B, 1u, (unsigned)T, (unsigned)C_DIM};
    typename G::w_gl  wk_arg{const_cast<bf16*>(w_k_up), 1u, 1u, (unsigned)H, (unsigned)(D*C_DIM)};
    typename G::w_gl  wv_arg{const_cast<bf16*>(w_v_up), 1u, 1u, (unsigned)H, (unsigned)(D*C_DIM)};
    typename G::o_gl  o_arg {out, (unsigned)B, (unsigned)H, 1u, (unsigned)D};
    G g{q_arg, kv_arg, wk_arg, wv_arg, o_arg, T, H};
    dim3 grid(1, H, B);
    mla_decode_kernel<D, C_DIM><<<grid, ::kittens::WARP_THREADS, 0, stream>>>(g);
    cudaCheck(cudaGetLastError());
}

inline void launch_decode_dispatch(bf16* out, const bf16* q, const bf16* kv_c,
                                   const bf16* w_k_up, const bf16* w_v_up,
                                   int B, int H, int T, int D, int C_DIM, cudaStream_t stream) {
    if      (D == 64  && C_DIM == 64)  launch_decode<64,  64> (out, q, kv_c, w_k_up, w_v_up, B, H, T, stream);
    else if (D == 64  && C_DIM == 128) launch_decode<64,  128>(out, q, kv_c, w_k_up, w_v_up, B, H, T, stream);
    else if (D == 128 && C_DIM == 64)  launch_decode<128, 64> (out, q, kv_c, w_k_up, w_v_up, B, H, T, stream);
    else if (D == 128 && C_DIM == 128) launch_decode<128, 128>(out, q, kv_c, w_k_up, w_v_up, B, H, T, stream);
    else {
        fprintf(stderr, "mla_sm120: unsupported (D=%d, C_DIM=%d)\n", D, C_DIM);
        exit(EXIT_FAILURE);
    }
}

}  // namespace llmk::mla
