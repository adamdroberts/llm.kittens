/*
attention_sm120.cuh — ThunderKittens causal multi-head attention forward for
consumer Blackwell (sm_120). Drop-in counterpart of attention_h100.cuh: same
`llmk::attention::launch_forward_causal(...)` signature and granularity
helpers, but using warp-scope MMAs only.

v1 covers FORWARD only. The backward path on sm_120 stays on the existing
scalar fallback in attention.cuh (which is correct, just slow). Adding a TK
backward is a separate follow-up.

Algorithm: FlashAttention-2 forward, single-warp granularity.
  * One warp per CTA.
  * Each warp owns one (batch, head, q_block) where q_block is 16 query rows.
  * Grid: dim3(T / 16, NH, B). Block: 32 threads (one warp).
  * Streaming K/V: load one (k, v) 16-row block at a time, compute partial
    attention with online softmax, accumulate O. Causal mask is the standard
    upper-triangular mask on the diagonal block (kb == q_block).

Supports HS == 64 and HS == 128. GPT-2 124M / 350M / 774M use HS = 64;
GPT-2 XL and Llama-style models use HS = 128. Both fit in the 256-register
per-thread budget on sm_120 for the forward (HS=128 forward sits around
~180 regs/thread).
*/
#pragma once

#include <type_traits>
#include <cmath>
#include "tk_common.cuh"

namespace llmk::attention {

using namespace ::kittens;

namespace sm120_detail {

// We tile Q in 16-row blocks. K and V are streamed 16 rows at a time.
constexpr int Q_BLOCK = 16;
constexpr int K_BLOCK = 16;

// Sequence granularity: T must be divisible by 16 so each (batch, head)
// has an integer number of Q/K blocks. GPT-2 with T=1024 satisfies this.
inline constexpr int fwd_sequence_granularity() { return Q_BLOCK; }
inline constexpr int bwd_sequence_granularity() { return K_BLOCK; }

template <int D>
struct fwd_globals {
    using q_tile     = st_bf<Q_BLOCK, D>;
    using kv_tile    = st_bf<K_BLOCK, D>;
    using o_tile     = st_bf<Q_BLOCK, D>;
    using lse_vec    = sv<float, Q_BLOCK>;

    using q_gl  = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl  = gl<bf16,  -1, -1, -1, -1, kv_tile>;
    using v_gl  = gl<bf16,  -1, -1, -1, -1, kv_tile>;
    using o_gl  = gl<bf16,  -1, -1, -1, -1, o_tile>;
    using l_gl  = gl<float, -1, -1, -1, -1, lse_vec>;

    q_gl q;
    k_gl k;
    v_gl v;
    o_gl o;
    l_gl l;
};

template <int D>
__global__ void fwd_kernel(const __grid_constant__ fwd_globals<D> g)
{
    static_assert(D == 64 || D == 128, "sm120 attention forward: HS must be 64 or 128");

    const int batch_idx   = blockIdx.z;
    const int q_head_idx  = blockIdx.y;
    const int q_block_idx = blockIdx.x;

    using q_rt    = rt_bf<Q_BLOCK, D,       ducks::rt_layout::row>;
    using k_rt    = rt_bf<K_BLOCK, D,       ducks::rt_layout::row>;
    using v_rt    = rt_bf<K_BLOCK, D,       ducks::rt_layout::row>;
    using s_rt    = rt_fl<Q_BLOCK, K_BLOCK, ducks::rt_layout::row>;
    using p_rt    = rt_bf<Q_BLOCK, K_BLOCK, ducks::rt_layout::row>;
    using o_rt    = rt_fl<Q_BLOCK, D,       ducks::rt_layout::row>;
    using row_cv  = typename s_rt::col_vec;  // 16 fp32 entries, one per query row
    using o_cv    = typename o_rt::col_vec;

    // Load Q for this q_block (stays in registers for the whole K/V sweep).
    q_rt Q;
    ::kittens::warp::load(Q, g.q, {batch_idx, q_head_idx, q_block_idx, 0});

    // Init online softmax state.
    o_rt O;
    ::kittens::warp::zero(O);
    row_cv m;
    ::kittens::warp::neg_infty(m);
    row_cv l_acc;
    ::kittens::warp::zero(l_acc);

    const float scale = 1.0f / sqrtf((float)D);

    // Iterate K/V blocks. Causal: only kb in [0, q_block_idx].
    #pragma unroll 1
    for (int kb = 0; kb <= q_block_idx; ++kb) {
        k_rt K;
        v_rt V;
        ::kittens::warp::load(K, g.k, {batch_idx, q_head_idx, kb, 0});
        ::kittens::warp::load(V, g.v, {batch_idx, q_head_idx, kb, 0});

        // S = Q · K^T  (16×16 fp32)
        s_rt S;
        ::kittens::warp::zero(S);
        ::kittens::warp::mma_ABt(S, Q, K, S);

        // Apply scale and (on the diagonal block) the causal mask.
        ::kittens::warp::mul(S, S, scale);
        if (kb == q_block_idx) {
            ::kittens::warp::make_causal(S, S, -INFINITY);
        }

        // Online softmax update:
        //   m_new = max(m, rowmax(S))
        //   rescale = exp(m - m_new)
        //   S = exp(S - m_new)           (in place)
        //   l_acc = l_acc * rescale + rowsum(S)
        //   O = O * rescale + S · V
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

        // Convert S → bf16 (P), then O = O*rescale + P·V.
        p_rt P;
        ::kittens::warp::copy(P, S);

        // O *= rescale (broadcast across columns of O).
        ::kittens::warp::mul_row(O, O, rescale);

        // V needs col-layout for mma_AB.
        auto& V_col = ::kittens::warp::swap_layout_inplace(V);
        ::kittens::warp::mma_AB(O, P, V_col, O);

        // Roll forward.
        ::kittens::warp::copy(m, m_new);
    }

    // Final normalization: O /= l_acc.
    ::kittens::warp::div_row(O, O, l_acc);

    // LSE = m + log(l_acc).
    row_cv lse;
    ::kittens::warp::log(lse, l_acc);
    ::kittens::warp::add(lse, lse, m);

    ::kittens::warp::store(g.o, O, {batch_idx, q_head_idx, q_block_idx, 0});
    ::kittens::warp::store(g.l, lse, {batch_idx, q_head_idx, 0, q_block_idx});
}

// =============================================================================
// Backward
// =============================================================================
//
// Two kernels follow the FA-2 backward layout:
//   1. bwd_prep_kernel computes the per-Q-row scalar  D[q] = sum_d (dO[q,d] * O[q,d])
//   2. bwd_main_kernel reads Q, K, V, dO, LSE, D and produces dQ, dK, dV.
//
// Parallelisation:
//   - prep:  1 warp per (batch, head, q_block_of_16). Grid (T/16, NH, B).
//   - main:  1 warp per (batch, head, k_block_of_16). Grid (T/16, NH, B).
//            Each warp owns dK_kb / dV_kb in registers across its inner Q-loop
//            (no cross-warp contention on these). dQ is updated via atomicAdd
//            on the fp32 qg buffer — multiple K-warps contribute to the same Q
//            rows, matching the H100 design (which uses tma::store_add_async).
//   - Causal mask: warp at K-block kb only loops over Q-blocks qb >= kb;
//                  inside the diagonal (qb == kb) we make S upper-triangular
//                  (-INFINITY on entries above the diagonal) before softmax.

template <int D>
struct bwd_globals {
    using q_tile   = st_bf<Q_BLOCK, D>;
    using kv_tile  = st_bf<K_BLOCK, D>;
    using qkv_fl_tile = st_fl<Q_BLOCK, D>;  // for dQ/dK/dV fp32 accumulators
    using vec16    = sv<float, Q_BLOCK>;

    using q_gl  = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl  = gl<bf16,  -1, -1, -1, -1, kv_tile>;
    using v_gl  = gl<bf16,  -1, -1, -1, -1, kv_tile>;
    using o_gl  = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using og_gl = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using l_gl  = gl<float, -1, -1, -1, -1, vec16>;
    using d_gl  = gl<float, -1, -1, -1, -1, vec16>;
    using grad_gl = gl<float, -1, -1, -1, -1, qkv_fl_tile>;

    q_gl    q;
    k_gl    k;
    v_gl    v;
    o_gl    o;
    og_gl   og;
    l_gl    l;
    d_gl    d;
    grad_gl qg;
    grad_gl kg;
    grad_gl vg;

    int T;
};

// ---- D = rowsum(dO ⊙ O) prep kernel ---------------------------------------

template <int D>
__global__ void bwd_prep_kernel(const __grid_constant__ bwd_globals<D> g)
{
    static_assert(D == 64 || D == 128, "sm120 attention bwd_prep: HS must be 64 or 128");
    const int batch_idx   = blockIdx.z;
    const int q_head_idx  = blockIdx.y;
    const int q_block_idx = blockIdx.x;

    using io_rt = rt_bf<Q_BLOCK, D, ducks::rt_layout::row>;
    using fl_rt = rt_fl<Q_BLOCK, D, ducks::rt_layout::row>;
    using row_cv = typename fl_rt::col_vec;

    io_rt O_bf, dO_bf;
    ::kittens::warp::load(O_bf,  g.o,  {batch_idx, q_head_idx, q_block_idx, 0});
    ::kittens::warp::load(dO_bf, g.og, {batch_idx, q_head_idx, q_block_idx, 0});

    fl_rt O_fl, dO_fl;
    ::kittens::warp::copy(O_fl,  O_bf);
    ::kittens::warp::copy(dO_fl, dO_bf);

    // O_fl := O * dO (elementwise)
    ::kittens::warp::mul(O_fl, O_fl, dO_fl);

    // D[q] = sum_d (O_fl[q, d])
    row_cv d_vec;
    ::kittens::warp::row_sum(d_vec, O_fl);

    ::kittens::warp::store(g.d, d_vec, {batch_idx, q_head_idx, 0, q_block_idx});
}

// ---- atomicAdd helper: warp-scope fp32 rt -> fp32 gl ----------------------
//
// Mirrors the addressing of kittens::warp::store for row-layout fp32 rt → fp32
// gl, but issues per-float atomicAdds instead of plain stores. Each thread does
// 8 atomicAdds per base tile (4 fragments × 2 floats). Output is the same
// flattened (b, h, row, col) coordinate as warp::store.
template <typename RT, typename GL, typename COORD>
__device__ static inline void warp_atomic_add(const GL& dst, const RT& src, const COORD& idx) {
    static_assert(::kittens::ducks::rt::row_layout<RT>);
    static_assert(std::is_same_v<typename GL::dtype, float>);
    static_assert(std::is_same_v<typename RT::T, float>);

    float* dst_ptr = (float*)&dst[idx.template unit_coord<2, 3>()];
    const int row_stride = dst.template stride<2>();
    constexpr int half_col = RT::tile_size_col / 2;
    const int laneid = threadIdx.x % ::kittens::WARP_THREADS;

    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
        const int row_base = i * RT::tile_size_row + (laneid / 4);
        #pragma unroll
        for (int j = 0; j < RT::width; ++j) {
            const int col_base = j * RT::tile_size_col + 2 * (laneid % 4);
            // data[0]: (row+0, col+0..1)
            atomicAdd(&dst_ptr[(row_base + 0) * row_stride + col_base + 0], src.tiles[i][j].data[0].x);
            atomicAdd(&dst_ptr[(row_base + 0) * row_stride + col_base + 1], src.tiles[i][j].data[0].y);
            // data[2]: (row+0, col+half_col..half_col+1)
            atomicAdd(&dst_ptr[(row_base + 0) * row_stride + col_base + half_col + 0], src.tiles[i][j].data[2].x);
            atomicAdd(&dst_ptr[(row_base + 0) * row_stride + col_base + half_col + 1], src.tiles[i][j].data[2].y);
            // data[1]: (row+8, col+0..1)
            atomicAdd(&dst_ptr[(row_base + 8) * row_stride + col_base + 0], src.tiles[i][j].data[1].x);
            atomicAdd(&dst_ptr[(row_base + 8) * row_stride + col_base + 1], src.tiles[i][j].data[1].y);
            // data[3]: (row+8, col+half_col..half_col+1)
            atomicAdd(&dst_ptr[(row_base + 8) * row_stride + col_base + half_col + 0], src.tiles[i][j].data[3].x);
            atomicAdd(&dst_ptr[(row_base + 8) * row_stride + col_base + half_col + 1], src.tiles[i][j].data[3].y);
        }
    }
}

// ---- Main backward kernel: dQ / dK / dV -----------------------------------

template <int D>
__global__ void bwd_main_kernel(const __grid_constant__ bwd_globals<D> g)
{
    static_assert(D == 64, "sm120 attention bwd: v1 supports HS==64 only "
                           "(HS=128 register pressure exceeds budget; follow-up).");

    const int batch_idx   = blockIdx.z;
    const int q_head_idx  = blockIdx.y;
    const int k_block_idx = blockIdx.x;
    const int num_q_blocks = g.T / Q_BLOCK;

    using q_rt    = rt_bf<Q_BLOCK, D, ducks::rt_layout::row>;
    using k_rt    = rt_bf<K_BLOCK, D, ducks::rt_layout::row>;
    using v_rt    = rt_bf<K_BLOCK, D, ducks::rt_layout::row>;
    using o_fl_rt = rt_fl<K_BLOCK, D, ducks::rt_layout::row>;  // dK / dV accumulators
    using s_rt    = rt_fl<Q_BLOCK, K_BLOCK, ducks::rt_layout::row>;
    using p_rt    = rt_bf<Q_BLOCK, K_BLOCK, ducks::rt_layout::row>;
    using dq_rt   = rt_fl<Q_BLOCK, D, ducks::rt_layout::row>;
    using row_cv  = typename s_rt::col_vec;

    const float scale = 1.0f / sqrtf((float)D);

    // Load K, V once and keep in registers for the whole Q-loop.
    k_rt K;
    v_rt V;
    ::kittens::warp::load(K, g.k, {batch_idx, q_head_idx, k_block_idx, 0});
    ::kittens::warp::load(V, g.v, {batch_idx, q_head_idx, k_block_idx, 0});

    // dK, dV accumulators for this warp's K-block.
    o_fl_rt dK, dV;
    ::kittens::warp::zero(dK);
    ::kittens::warp::zero(dV);

    #pragma unroll 1
    for (int qb = k_block_idx; qb < num_q_blocks; ++qb) {
        q_rt Q, dO;
        ::kittens::warp::load(Q,  g.q,  {batch_idx, q_head_idx, qb, 0});
        ::kittens::warp::load(dO, g.og, {batch_idx, q_head_idx, qb, 0});

        row_cv lse_vec, d_vec;
        ::kittens::warp::load(lse_vec, g.l, {batch_idx, q_head_idx, 0, qb});
        ::kittens::warp::load(d_vec,   g.d, {batch_idx, q_head_idx, 0, qb});

        // S = (Q · K^T) * scale ; causal mask on the diagonal block.
        s_rt S;
        ::kittens::warp::zero(S);
        ::kittens::warp::mma_ABt(S, Q, K, S);
        ::kittens::warp::mul(S, S, scale);
        if (qb == k_block_idx) {
            ::kittens::warp::make_causal(S, S, -INFINITY);
        }

        // P = exp(S - LSE_q)   (uses the LSE saved during forward).
        s_rt P;
        ::kittens::warp::sub_row(P, S, lse_vec);
        ::kittens::warp::exp(P, P);

        // dP = dO · V^T
        s_rt dP;
        ::kittens::warp::zero(dP);
        ::kittens::warp::mma_ABt(dP, dO, V, dP);

        // dS = P * (dP - D_q)
        s_rt dS;
        ::kittens::warp::sub_row(dS, dP, d_vec);
        ::kittens::warp::mul(dS, dS, P);

        // dV += P^T · dO   (in-place layout swap on P_bf and dO).
        p_rt P_bf;
        ::kittens::warp::copy(P_bf, P);
        auto& P_col  = ::kittens::warp::swap_layout_inplace(P_bf);
        auto& dO_col = ::kittens::warp::swap_layout_inplace(dO);
        ::kittens::warp::mma_AtB(dV, P_col, dO_col, dV);

        // dK += scale · dS^T · Q   (we fold scale into dS just before mma).
        ::kittens::warp::mul(dS, dS, scale);
        p_rt dS_bf;
        ::kittens::warp::copy(dS_bf, dS);
        auto& dS_col = ::kittens::warp::swap_layout_inplace(dS_bf);
        // Q was used in mma_ABt(S, Q, K, S) above as a row-layout tile; we
        // reload its col-layout view in place. mma_ABt left Q unchanged, but
        // we used dO above and swapped it in place — so reload dO would be
        // needed for further work. For mma_AtB(dK, dS_col, Q_col, dK), swap Q.
        auto& Q_col = ::kittens::warp::swap_layout_inplace(Q);
        ::kittens::warp::mma_AtB(dK, dS_col, Q_col, dK);

        // dQ_partial = dS_bf · K   (mma_AB needs B in col layout).
        // dS_bf is in col layout from the swap above — bring it back to row.
        auto& dS_row = ::kittens::warp::swap_layout_inplace(dS_col);
        // K is in row layout still (we never swapped K because the K-block
        // contents stay constant across qb iters; swapping in place would lose
        // them for the next iteration). Make a temporary col view by copying.
        rt_bf<K_BLOCK, D, ducks::rt_layout::col> K_col;
        ::kittens::warp::swap_layout(K_col, K);

        dq_rt dQ_partial;
        ::kittens::warp::zero(dQ_partial);
        ::kittens::warp::mma_AB(dQ_partial, dS_row, K_col, dQ_partial);

        // Atomic-add dQ_partial into the global dQ buffer for this Q-block.
        warp_atomic_add(g.qg, dQ_partial,
            ::kittens::coord<typename bwd_globals<D>::qkv_fl_tile>{batch_idx, q_head_idx, qb, 0});
    }

    // dK and dV are warp-private accumulators with no contention; plain store
    // through the fp32-tile gl<>s set up on the host.
    ::kittens::warp::store(g.kg, dK, {batch_idx, q_head_idx, k_block_idx, 0});
    ::kittens::warp::store(g.vg, dV, {batch_idx, q_head_idx, k_block_idx, 0});
}

} // namespace sm120_detail

// ---- granularity / capability helpers (mirrors attention_h100.cuh) -------

inline int fwd_sequence_granularity() { return sm120_detail::fwd_sequence_granularity(); }
inline int bwd_sequence_granularity() { return sm120_detail::bwd_sequence_granularity(); }
// v1 backward supports HS=64 only; HS=128 is a follow-up (register pressure).
inline bool bwd_supports_head_dim(int HS) { return HS == 64; }

// ---- forward launcher -----------------------------------------------------

template <int D>
inline void launch_forward_causal(bf16* q, bf16* k, bf16* v, float* l, bf16* o,
                                  int B, int NH, int T, cudaStream_t stream)
{
    static_assert(D == 64 || D == 128, "sm120 attention forward: HS must be 64 or 128");
    assert(T % sm120_detail::fwd_sequence_granularity() == 0 &&
           "sm120 attention: T must be a multiple of 16");

    using globals = sm120_detail::fwd_globals<D>;

    typename globals::q_gl q_arg{q, static_cast<unsigned int>(B), static_cast<unsigned int>(NH),
                                 static_cast<unsigned int>(T),  static_cast<unsigned int>(D)};
    typename globals::k_gl k_arg{k, static_cast<unsigned int>(B), static_cast<unsigned int>(NH),
                                 static_cast<unsigned int>(T),  static_cast<unsigned int>(D)};
    typename globals::v_gl v_arg{v, static_cast<unsigned int>(B), static_cast<unsigned int>(NH),
                                 static_cast<unsigned int>(T),  static_cast<unsigned int>(D)};
    typename globals::o_gl o_arg{o, static_cast<unsigned int>(B), static_cast<unsigned int>(NH),
                                 static_cast<unsigned int>(T),  static_cast<unsigned int>(D)};
    typename globals::l_gl l_arg{l, static_cast<unsigned int>(B), static_cast<unsigned int>(NH),
                                 1U, static_cast<unsigned int>(T)};

    globals g{q_arg, k_arg, v_arg, o_arg, l_arg};

    dim3 grid(T / sm120_detail::Q_BLOCK, NH, B);
    sm120_detail::fwd_kernel<D><<<grid, ::kittens::WARP_THREADS, 0, stream>>>(g);
}

inline void launch_forward_causal(bf16* q, bf16* k, bf16* v, float* l, bf16* o,
                                  int B, int NH, int T, int HS, cudaStream_t stream) {
    if (HS == 64) {
        launch_forward_causal<64>(q, k, v, l, o, B, NH, T, stream);
    } else if (HS == 128) {
        launch_forward_causal<128>(q, k, v, l, o, B, NH, T, stream);
    } else {
        fprintf(stderr, "attention_forward: sm120 TK MHA supports head_dim 64 or 128 (got %d)\n", HS);
        exit(EXIT_FAILURE);
    }
}

// ---- backward launcher ----------------------------------------------------
//
// Signature matches attention_h100.cuh's launch_backward_causal so the host
// wiring in llmc/attention.cuh is unchanged.

template <int D>
inline void launch_backward_causal(bf16* q, bf16* k, bf16* v, bf16* o,
                                   float* l, bf16* og,
                                   float* d, float* qg, float* kg, float* vg,
                                   int B, int NH, int T, cudaStream_t stream)
{
    static_assert(D == 64, "sm120 attention bwd: v1 supports HS==64 only");
    assert(T % sm120_detail::bwd_sequence_granularity() == 0 &&
           "sm120 attention bwd: T must be a multiple of 16");

    using globals = sm120_detail::bwd_globals<D>;
    const unsigned int Bu  = static_cast<unsigned int>(B);
    const unsigned int NHu = static_cast<unsigned int>(NH);
    const unsigned int Tu  = static_cast<unsigned int>(T);
    const unsigned int Du  = static_cast<unsigned int>(D);

    typename globals::q_gl    q_arg  {q,  Bu, NHu, Tu, Du};
    typename globals::k_gl    k_arg  {k,  Bu, NHu, Tu, Du};
    typename globals::v_gl    v_arg  {v,  Bu, NHu, Tu, Du};
    typename globals::o_gl    o_arg  {o,  Bu, NHu, Tu, Du};
    typename globals::og_gl   og_arg {og, Bu, NHu, Tu, Du};
    typename globals::l_gl    l_arg  {l,  Bu, NHu, 1U, Tu};
    typename globals::d_gl    d_arg  {d,  Bu, NHu, 1U, Tu};
    typename globals::grad_gl qg_arg {qg, Bu, NHu, Tu, Du};
    typename globals::grad_gl kg_arg {kg, Bu, NHu, Tu, Du};
    typename globals::grad_gl vg_arg {vg, Bu, NHu, Tu, Du};

    globals g{q_arg, k_arg, v_arg, o_arg, og_arg, l_arg, d_arg, qg_arg, kg_arg, vg_arg, T};

    dim3 grid(T / sm120_detail::Q_BLOCK, NH, B);
    // Prep computes D for every (b, h, q_block); main computes dQ/dK/dV.
    sm120_detail::bwd_prep_kernel<D><<<grid, ::kittens::WARP_THREADS, 0, stream>>>(g);
    sm120_detail::bwd_main_kernel<D><<<grid, ::kittens::WARP_THREADS, 0, stream>>>(g);
}

inline void launch_backward_causal(bf16* q, bf16* k, bf16* v, bf16* o,
                                   float* l, bf16* og,
                                   float* d, float* qg, float* kg, float* vg,
                                   int B, int NH, int T, int HS, cudaStream_t stream) {
    if (HS == 64) {
        launch_backward_causal<64>(q, k, v, o, l, og, d, qg, kg, vg, B, NH, T, stream);
    } else {
        // HS == 128 backward is a follow-up (register pressure exceeds 256/thread
        // without shared-memory staging for an accumulator). For now the caller
        // (attention.cuh) gates the TK bwd path on (HS == 64 || HS == 128); when
        // HS == 128 it'll hit this branch — handle by skipping back to scalar
        // bwd via assert. The host code already has the scalar fallback path
        // for arches without LLMK_USE_TK_MHA_BWD, but it's not reachable from
        // here. Document the gap rather than silently return wrong gradients.
        fprintf(stderr, "attention_backward: sm120 TK MHA backward supports head_dim 64 only "
                        "(got %d). HS=128 backward is on the follow-up list.\n", HS);
        exit(EXIT_FAILURE);
    }
}

} // namespace llmk::attention
