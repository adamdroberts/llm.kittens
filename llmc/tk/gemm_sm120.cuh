/*
gemm_sm120.cuh — ThunderKittens bf16 GEMM for consumer Blackwell (RTX 50-series,
sm_120). Drop-in replacement for gemm_h100.cuh, exposing the same
`llmk::gemm::launch<mmt>(...)` API and the same matmul_default_* / matmul_small_n_*
type aliases that matmul.cuh dispatches on.

Why a separate kernel from gemm_h100.cuh:
  * Hopper's WGMMA (`warpgroup::mma_*`) is gated on KITTENS_SM90.
  * Blackwell B200/B300's tcgen05 path is gated on KITTENS_SM10X.
  * Consumer Blackwell (sm_120) has neither — only warp-scope `mma.sync` via
    kittens::warp::mma_AB / mma_ABt / mma_AtB. We use those directly with no
    warpgroup pipelining and no TMA.

Design (kept deliberately simple for v1 correctness; perf headroom remains):
  * Each CTA computes a 64×64 fp32 output tile, written out as bf16.
  * 4 warps per CTA, 1-D partition: warp w owns rows [w*16, (w+1)*16) of the
    output tile.
  * K is iterated in chunks of 32. Per K step each warp does:
      - load its 16-row × 32-col slice of A through warp::load,
      - load the full 64-row × 32-col B tile through warp::load,
      - mma_ABt (or mma_AB / mma_AtB depending on the operation).
  * Loads go straight global → register (no shared staging). Redundant B-fetches
    across the 4 warps hit L1 after the first, so HBM bandwidth is not 4× — but
    L1→register traffic is. That's fine for getting llm.kittens unblocked on
    sm_120; a shared-staged version is a follow-up optimization.

Shape constraints (must hold at launch time):
  * M divisible by 64
  * N divisible by 64
  * K divisible by 32
GPT-2 124M's dense matmuls satisfy all three (M = B*T multiple of 128, N is
either 768, 2304, 3072, or 50304 — all multiples of 64, K = 768 multiple of 64).
*/
#pragma once

#include <type_traits>
#include "tk_common.cuh"

namespace llmk::gemm {

using namespace ::kittens;

namespace sm120_detail {

// ---- tunables -------------------------------------------------------------
constexpr int M_TILE     = 64;
constexpr int N_TILE     = 64;
constexpr int K_TILE     = 32;
constexpr int NUM_WARPS  = 4;
constexpr int WARP_M     = M_TILE / NUM_WARPS;  // 16
constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;

// ---- gl<> types & per-op globals ------------------------------------------
//
// gl<> objects own their pointer + shape; they must be constructed on the host
// (the constructor is a __host__ function) and passed to the kernel as a
// __grid_constant__ argument. We keep one globals struct per operation so the
// kernel signature is the minimum needed for that case.

using a_base_nt = st_bf<WARP_M, K_TILE>;
using b_base_nt = st_bf<N_TILE, K_TILE>;
using c_base    = st_bf<WARP_M, N_TILE>;

using a_base_nn = st_bf<WARP_M, K_TILE>;
using b_base_nn = st_bf<K_TILE, N_TILE>;

using a_base_tn = st_bf<K_TILE, WARP_M>;
using b_base_tn = st_bf<K_TILE, N_TILE>;

using a_gl_nt   = gl<bf16, 1, 1, -1, -1, a_base_nt>;
using b_gl_nt   = gl<bf16, 1, 1, -1, -1, b_base_nt>;
using c_gl      = gl<bf16, 1, 1, -1, -1, c_base>;
using bias_gl   = gl<bf16, 1, 1, 1,  -1, sv<bf16, N_TILE>>;

using a_gl_nn   = gl<bf16, 1, 1, -1, -1, a_base_nn>;
using b_gl_nn   = gl<bf16, 1, 1, -1, -1, b_base_nn>;

using a_gl_tn   = gl<bf16, 1, 1, -1, -1, a_base_tn>;
using b_gl_tn   = gl<bf16, 1, 1, -1, -1, b_base_tn>;

struct globals_nt {
    a_gl_nt A;
    b_gl_nt B;
    c_gl    C;
    c_gl    P;
    bias_gl bias;
    int     bias_present;
};

struct globals_nn {
    a_gl_nn A;
    b_gl_nn B;
    c_gl    C;
};

struct globals_tn {
    a_gl_tn A;
    b_gl_tn B;
    c_gl    C;
};

// ---- kernels --------------------------------------------------------------

template <bool APPLY_BIAS, bool APPLY_GELU, bool STORE_PRE_GELU>
__global__ void kernel_nt(const __grid_constant__ globals_nt g)
{
    using a_rt = rt_bf<WARP_M, K_TILE, ducks::rt_layout::row>;
    using b_rt = rt_bf<N_TILE, K_TILE, ducks::rt_layout::row>;
    using d_rt = rt_fl<WARP_M, N_TILE, ducks::rt_layout::row>;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int w  = threadIdx.x / ::kittens::WARP_THREADS;

    d_rt accum;
    ::kittens::warp::zero(accum);

    const int K = (int)g.A.cols();
    const int num_k_tiles = K / K_TILE;
    #pragma unroll 1
    for (int kk = 0; kk < num_k_tiles; ++kk) {
        a_rt a_reg;
        b_rt b_reg;
        ::kittens::warp::load(a_reg, g.A, {0, 0, by * NUM_WARPS + w, kk});
        ::kittens::warp::load(b_reg, g.B, {0, 0, bx, kk});
        ::kittens::warp::mma_ABt(accum, a_reg, b_reg, accum);
    }

    if constexpr (APPLY_BIAS) {
        if (g.bias_present) {
            typename d_rt::row_vec bias_vec;
            ::kittens::warp::load(bias_vec, g.bias, {0, 0, 0, bx});
            ::kittens::warp::add_col(accum, accum, bias_vec);
        }
    }

    if constexpr (STORE_PRE_GELU) {
        ::kittens::warp::store(g.P, accum, {0, 0, by * NUM_WARPS + w, bx});
    }

    if constexpr (APPLY_GELU) {
        constexpr float k0 = 0.7978845608028654f;
        constexpr float k1 = 0.044715f;
        #pragma unroll
        for (int i = 0; i < d_rt::height; ++i) {
            #pragma unroll
            for (int j = 0; j < d_rt::width; ++j) {
                auto& t = accum.tiles[i][j];
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float x0 = t.data[e].x;
                    float x1 = t.data[e].y;
                    float c0 = k1 * x0 * x0 * x0;
                    float c1 = k1 * x1 * x1 * x1;
                    t.data[e].x = 0.5f * x0 * (1.0f + tanhf(k0 * (x0 + c0)));
                    t.data[e].y = 0.5f * x1 * (1.0f + tanhf(k0 * (x1 + c1)));
                }
            }
        }
    }

    ::kittens::warp::store(g.C, accum, {0, 0, by * NUM_WARPS + w, bx});
}

__global__ void kernel_nn(const __grid_constant__ globals_nn g)
{
    using a_rt     = rt_bf<WARP_M, K_TILE, ducks::rt_layout::row>;
    using b_rt_row = rt_bf<K_TILE, N_TILE, ducks::rt_layout::row>;
    using b_rt_col = rt_bf<K_TILE, N_TILE, ducks::rt_layout::col>;
    using d_rt     = rt_fl<WARP_M, N_TILE, ducks::rt_layout::row>;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int w  = threadIdx.x / ::kittens::WARP_THREADS;

    d_rt accum;
    ::kittens::warp::zero(accum);

    const int K = (int)g.A.cols();
    const int num_k_tiles = K / K_TILE;
    #pragma unroll 1
    for (int kk = 0; kk < num_k_tiles; ++kk) {
        a_rt     a_reg;
        b_rt_row b_reg_row;
        b_rt_col b_reg_col;
        ::kittens::warp::load(a_reg, g.A, {0, 0, by * NUM_WARPS + w, kk});
        ::kittens::warp::load(b_reg_row, g.B, {0, 0, kk, bx});
        ::kittens::warp::swap_layout(b_reg_col, b_reg_row);
        ::kittens::warp::mma_AB(accum, a_reg, b_reg_col, accum);
    }

    ::kittens::warp::store(g.C, accum, {0, 0, by * NUM_WARPS + w, bx});
}

__global__ void kernel_tn(const __grid_constant__ globals_tn g)
{
    using a_rt_row = rt_bf<K_TILE, WARP_M, ducks::rt_layout::row>;
    using a_rt_col = rt_bf<K_TILE, WARP_M, ducks::rt_layout::col>;
    using b_rt_row = rt_bf<K_TILE, N_TILE, ducks::rt_layout::row>;
    using b_rt_col = rt_bf<K_TILE, N_TILE, ducks::rt_layout::col>;
    using d_rt     = rt_fl<WARP_M, N_TILE, ducks::rt_layout::row>;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int w  = threadIdx.x / ::kittens::WARP_THREADS;

    d_rt accum;
    ::kittens::warp::zero(accum);

    const int K = (int)g.A.rows();
    const int num_k_tiles = K / K_TILE;
    #pragma unroll 1
    for (int kk = 0; kk < num_k_tiles; ++kk) {
        a_rt_row a_reg_row;
        a_rt_col a_reg_col;
        b_rt_row b_reg_row;
        b_rt_col b_reg_col;
        ::kittens::warp::load(a_reg_row, g.A, {0, 0, kk, by * NUM_WARPS + w});
        ::kittens::warp::load(b_reg_row, g.B, {0, 0, kk, bx});
        ::kittens::warp::swap_layout(a_reg_col, a_reg_row);
        ::kittens::warp::swap_layout(b_reg_col, b_reg_row);
        ::kittens::warp::mma_AtB(accum, a_reg_col, b_reg_col, accum);
    }

    ::kittens::warp::store(g.C, accum, {0, 0, by * NUM_WARPS + w, bx});
}

} // namespace sm120_detail

// ---- mmt template-tag types -----------------------------------------------
//
// Mirrors the gemm_h100.cuh template-tag surface so matmul.cuh's dispatch
// (which references llmk::gemm::matmul_default_nt etc.) compiles unchanged.
// `M_BLOCK` / `N_BLOCK` are unused on sm_120 — the kernel uses a fixed 64×64
// CTA tile. We carry them as fields so the existing dispatch's static
// references resolve.

template <int _M_BLOCK = 2, int _N_BLOCK = 4, int _SUPER_M = 8,
          bool _A_TRANSPOSED = false, bool _B_TRANSPOSED = false,
          bool _APPLY_BIAS = false, bool _APPLY_GELU = false,
          bool _STORE_PRE_GELU = false>
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK;
    static constexpr int N_BLOCK = _N_BLOCK;
    static constexpr int SUPER_M = _SUPER_M;
    static constexpr bool A_TRANSPOSED  = _A_TRANSPOSED;
    static constexpr bool B_TRANSPOSED  = _B_TRANSPOSED;
    static constexpr bool APPLY_BIAS    = _APPLY_BIAS;
    static constexpr bool APPLY_GELU    = _APPLY_GELU;
    static constexpr bool STORE_PRE_GELU = _STORE_PRE_GELU;
};

// Forward (NT): C = A · B^T. Used for every dense matmul in the forward pass
// (weights stored as (OC, C)). Bias/GELU epilogues attach here.
using matmul_default_nt              = matmul_template<2, 4, 8, false, true,  false, false, false>;
using matmul_default_nt_bias         = matmul_template<2, 4, 8, false, true,  true,  false, false>;
using matmul_default_nt_bias_gelu    = matmul_template<2, 4, 8, false, true,  true,  true,  true>;
using matmul_small_n_nt              = matmul_template<2, 2, 8, false, true,  false, false, false>;
using matmul_small_n_nt_bias         = matmul_template<2, 2, 8, false, true,  true,  false, false>;
using matmul_small_n_nt_bias_gelu    = matmul_template<2, 2, 8, false, true,  true,  true,  true>;

// dInp (NN): C = A · B. dInp = dout · weight, weight stored as (OC, C) is
// treated as (K=OC, N=C) here.
using matmul_default                 = matmul_template<2, 4, 8, false, false, false, false, false>;
using matmul_small_n                 = matmul_template<2, 2, 8, false, false, false, false, false>;

// dWeight (TN): C = A^T · B. dweight = dout^T · inp.
using matmul_default_tn              = matmul_template<2, 4, 8, true,  false, false, false, false>;
using matmul_small_n_tn              = matmul_template<2, 2, 8, true,  false, false, false, false>;

// ---- launcher -------------------------------------------------------------

template <typename mmt>
inline void launch(bf16* d_A, bf16* d_B, bf16* d_C, int M, int N, int K,
                   cudaStream_t stream = 0, bf16* d_pre_gelu = nullptr,
                   const bf16* d_bias = nullptr) {
    using namespace sm120_detail;
    assert(M % M_TILE == 0 && "gemm_sm120: M must be a multiple of 64");
    assert(N % N_TILE == 0 && "gemm_sm120: N must be a multiple of 64");
    assert(K % K_TILE == 0 && "gemm_sm120: K must be a multiple of 32");

    const size_t M_ = static_cast<size_t>(M);
    const size_t N_ = static_cast<size_t>(N);
    const size_t K_ = static_cast<size_t>(K);

    dim3 grid(N / N_TILE, M / M_TILE);
    dim3 block(NUM_THREADS);

    if constexpr (mmt::A_TRANSPOSED && mmt::B_TRANSPOSED) {
        static_assert(!(mmt::A_TRANSPOSED && mmt::B_TRANSPOSED),
            "gemm_sm120: A_TRANSPOSED && B_TRANSPOSED not implemented (unused).");
    } else if constexpr (!mmt::A_TRANSPOSED && mmt::B_TRANSPOSED) {
        // C = A · B^T (forward)
        a_gl_nt Agl{d_A, nullptr, nullptr, M_, K_};
        b_gl_nt Bgl{d_B, nullptr, nullptr, N_, K_};
        c_gl    Cgl{d_C, nullptr, nullptr, M_, N_};
        c_gl    Pgl{d_pre_gelu == nullptr ? d_C : d_pre_gelu, nullptr, nullptr, M_, N_};
        // bias_gl needs a non-null pointer even when bias is unused; route to d_C as a safe stub.
        const bf16* bias_ptr = d_bias != nullptr ? d_bias : d_C;
        bias_gl Bias{const_cast<bf16*>(bias_ptr), nullptr, nullptr, nullptr, N_};
        globals_nt g{Agl, Bgl, Cgl, Pgl, Bias, d_bias != nullptr ? 1 : 0};

        auto kfn = kernel_nt<mmt::APPLY_BIAS, mmt::APPLY_GELU, mmt::STORE_PRE_GELU>;
        kfn<<<grid, block, 0, stream>>>(g);
    } else if constexpr (!mmt::A_TRANSPOSED && !mmt::B_TRANSPOSED) {
        static_assert(!mmt::APPLY_BIAS && !mmt::APPLY_GELU && !mmt::STORE_PRE_GELU,
            "gemm_sm120: bias/gelu epilogues only available on NT path.");
        a_gl_nn Agl{d_A, nullptr, nullptr, M_, K_};
        b_gl_nn Bgl{d_B, nullptr, nullptr, K_, N_};
        c_gl    Cgl{d_C, nullptr, nullptr, M_, N_};
        globals_nn g{Agl, Bgl, Cgl};
        kernel_nn<<<grid, block, 0, stream>>>(g);
    } else {
        static_assert(!mmt::APPLY_BIAS && !mmt::APPLY_GELU && !mmt::STORE_PRE_GELU,
            "gemm_sm120: bias/gelu epilogues only available on NT path.");
        a_gl_tn Agl{d_A, nullptr, nullptr, K_, M_};
        b_gl_tn Bgl{d_B, nullptr, nullptr, K_, N_};
        c_gl    Cgl{d_C, nullptr, nullptr, M_, N_};
        globals_tn g{Agl, Bgl, Cgl};
        kernel_tn<<<grid, block, 0, stream>>>(g);
    }
}

} // namespace llmk::gemm
