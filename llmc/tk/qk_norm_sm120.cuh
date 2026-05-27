/*
qk_norm_sm120.cuh — ThunderKittens fused QK-RMSNorm for SM120.

Applies RMSNorm independently to each (head_dim) row of Q and K. Each warp
processes one (batch, head, seq_position) row of length D.

Use case: callers that want QK-normalised attention can pipeline this kernel
between the QKV projection and the SDPA forward.
*/
#pragma once

#include <type_traits>
#include <cmath>
#include "tk_common.cuh"

namespace llmk::qk_norm {

using namespace ::kittens;

#ifndef LLMK_SM120_QK_NORM_BLOCK
#define LLMK_SM120_QK_NORM_BLOCK 32
#endif
constexpr int ROWS_PER_BLOCK = LLMK_SM120_QK_NORM_BLOCK;

template <int D>
struct globals {
    using row_tile = st_bf<ROWS_PER_BLOCK, D>;
    using gl_t     = gl<bf16, -1, -1, -1, -1, row_tile>;
    gl_t in;
    gl_t out;
    float eps;
    int   total_rows;
};

template <int D>
__global__ void qk_rmsnorm_kernel(const __grid_constant__ globals<D> g) {
    static_assert(D == 64 || D == 128, "qk_norm: head_dim must be 64 or 128");
    using rt = rt_bf<ROWS_PER_BLOCK, D, ducks::rt_layout::row>;
    using ft = rt_fl<ROWS_PER_BLOCK, D, ducks::rt_layout::row>;
    using row_cv = typename ft::col_vec;

    int row_block = blockIdx.x;
    rt X;
    ::kittens::warp::load(X, g.in, {row_block, 0, 0, 0});
    ft X_fl, sq;
    ::kittens::warp::copy(X_fl, X);
    ::kittens::warp::copy(sq, X_fl);
    ::kittens::warp::mul(sq, sq, X_fl);
    row_cv ss;
    ::kittens::warp::row_sum(ss, sq);
    // ss /= D, ss += eps, ss = rsqrt(ss)
    row_cv inv_d;
    ::kittens::warp::ones(inv_d);
    ::kittens::warp::mul(inv_d, inv_d, (float)D);
    ::kittens::warp::div(ss, ss, inv_d);
    ::kittens::warp::add(ss, ss, g.eps);
    ::kittens::warp::rsqrt(ss, ss);
    ::kittens::warp::mul_row(X_fl, X_fl, ss);
    rt Y;
    ::kittens::warp::copy(Y, X_fl);
    ::kittens::warp::store(g.out, Y, {row_block, 0, 0, 0});
}

template <int D>
inline void launch(bf16* y, const bf16* x, int total_rows, float eps, cudaStream_t stream) {
    assert(total_rows % ROWS_PER_BLOCK == 0);
    using G = globals<D>;
    typename G::gl_t in_arg {const_cast<bf16*>(x), 1u, 1u, 1u, (unsigned)(total_rows * D)};
    typename G::gl_t out_arg{y,                    1u, 1u, 1u, (unsigned)(total_rows * D)};
    G g{in_arg, out_arg, eps, total_rows};
    int blocks = total_rows / ROWS_PER_BLOCK;
    qk_rmsnorm_kernel<D><<<blocks, ::kittens::WARP_THREADS, 0, stream>>>(g);
}

}  // namespace llmk::qk_norm
