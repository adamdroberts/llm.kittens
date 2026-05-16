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

v2 design — shared staging + cp.async-pipelined K loop + 128×64 CTA tile.
  * Each CTA computes a 128×64 bf16 output tile (was 64×64 in v1). 4 warps split
    the M dim: warp w owns rows [w*32, (w+1)*32) of the output.
  * K is iterated in chunks of 32. A 2-stage shared-memory ring buffer holds
    {As[2], Bs[2]}. We use kittens::group<4>::load_async (cp.async.cg) to issue
    the next K-tile's loads while the current K-tile's MMAs run, then drain via
    load_async_wait<2> (≤ 2 commit groups outstanding) so the prior iter's
    loads have completed before we read them.
  * Per-warp register loads from shared use kittens::warp::load on a per-warp
    subtile of As[stage] (32×32 slice) and the full Bs[stage] tile.
  * No TMA on sm_120 (consumer Blackwell has only cp.async; TMA is sm_90+ and
    its consumer-Blackwell support is partial).

Shape constraints:
  * M divisible by 128
  * N divisible by 64
  * K divisible by 32
GPT-2 124M dense matmuls satisfy all three (M = B*T multiple of 128, N is one
of 768 / 2304 / 3072 / 50304 — all multiples of 64, K = 768 multiple of 32).
*/
#pragma once

#include <type_traits>
#include "tk_common.cuh"

namespace llmk::gemm {

using namespace ::kittens;

namespace sm120_detail {

// ---- tunables -------------------------------------------------------------
constexpr int M_TILE      = 128;
constexpr int N_TILE      = 64;
constexpr int K_TILE      = 32;
constexpr int NUM_WARPS   = 4;
constexpr int WARP_M      = M_TILE / NUM_WARPS;  // 32
constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
constexpr int PIPE_STAGES = 2;
// 4-warp `group<4>::load_async_wait<N>(bar_id)` will sync on this barrier.
constexpr int LOAD_BAR    = 0;

// ---- per-op shared-tile and gl<> types ------------------------------------

// NT: A is (M, K) row-major; B is (N, K) row-major. C = A·Bᵀ.
using a_smem_nt = st_bf<M_TILE, K_TILE>;
using b_smem_nt = st_bf<N_TILE, K_TILE>;
// NN: A is (M, K) row-major; B is (K, N) row-major. C = A·B.
using a_smem_nn = st_bf<M_TILE, K_TILE>;
using b_smem_nn = st_bf<K_TILE, N_TILE>;
// TN: A is (K, M) row-major; B is (K, N) row-major. C = Aᵀ·B.
using a_smem_tn = st_bf<K_TILE, M_TILE>;
using b_smem_tn = st_bf<K_TILE, N_TILE>;
// C is always (M, N) row-major; per-warp store granularity = 32×64.
using c_smem    = st_bf<WARP_M, N_TILE>;

using a_gl_nt   = gl<bf16, 1, 1, -1, -1, a_smem_nt>;
using b_gl_nt   = gl<bf16, 1, 1, -1, -1, b_smem_nt>;
using a_gl_nn   = gl<bf16, 1, 1, -1, -1, a_smem_nn>;
using b_gl_nn   = gl<bf16, 1, 1, -1, -1, b_smem_nn>;
using a_gl_tn   = gl<bf16, 1, 1, -1, -1, a_smem_tn>;
using b_gl_tn   = gl<bf16, 1, 1, -1, -1, b_smem_tn>;
using c_gl      = gl<bf16, 1, 1, -1, -1, c_smem>;
using bias_gl   = gl<bf16, 1, 1, 1,  -1, sv<bf16, N_TILE>>;

struct globals_nt {
    a_gl_nt A;
    b_gl_nt B;
    c_gl    C;
    c_gl    P;
    bias_gl bias;
    int     bias_present;
};
struct globals_nn { a_gl_nn A; b_gl_nn B; c_gl C; };
struct globals_tn { a_gl_tn A; b_gl_tn B; c_gl C; };

// ---- kernel_nt : C = A · Bᵀ -----------------------------------------------

template <bool APPLY_BIAS, bool APPLY_GELU, bool STORE_PRE_GELU>
__global__ __launch_bounds__(NUM_THREADS, 2)
void kernel_nt(const __grid_constant__ globals_nt g)
{
    using a_rt = rt_bf<WARP_M, K_TILE, ducks::rt_layout::row>;
    using b_rt = rt_bf<N_TILE, K_TILE, ducks::rt_layout::row>;
    using d_rt = rt_fl<WARP_M, N_TILE, ducks::rt_layout::row>;

    const int bx = blockIdx.x;          // output-col tile index (in N_TILE units)
    const int by = blockIdx.y;          // output-row tile index (in M_TILE units)
    const int w  = threadIdx.x / ::kittens::WARP_THREADS;

    __shared__ a_smem_nt As[PIPE_STAGES];
    __shared__ b_smem_nt Bs[PIPE_STAGES];

    d_rt accum;
    ::kittens::warp::zero(accum);

    const int num_k_tiles = (int)g.A.cols() / K_TILE;

    // Prologue: prefetch stage 0.
    ::kittens::group<NUM_WARPS>::load_async(As[0], g.A, {0, 0, by, 0});
    ::kittens::group<NUM_WARPS>::load_async(Bs[0], g.B, {0, 0, bx, 0});

    #pragma unroll 1
    for (int kk = 0; kk < num_k_tiles; ++kk) {
        const int stage = kk % PIPE_STAGES;

        if (kk + 1 < num_k_tiles) {
            const int next = (kk + 1) % PIPE_STAGES;
            ::kittens::group<NUM_WARPS>::load_async(As[next], g.A, {0, 0, by, kk + 1});
            ::kittens::group<NUM_WARPS>::load_async(Bs[next], g.B, {0, 0, bx, kk + 1});
            // Keep at most 2 commit groups outstanding — i.e. wait for THIS
            // iter's loads to finish; the next iter's are allowed to remain in
            // flight.
            ::kittens::group<NUM_WARPS>::template load_async_wait<2>(LOAD_BAR);
        } else {
            ::kittens::group<NUM_WARPS>::template load_async_wait<0>(LOAD_BAR);
        }

        // Each warp pulls its 32-row strip out of the just-loaded As tile.
        auto As_slice = As[stage].template subtile<WARP_M, K_TILE>({w, 0});
        a_rt a_reg;
        b_rt b_reg;
        ::kittens::warp::load(a_reg, As_slice);
        ::kittens::warp::load(b_reg, Bs[stage]);
        ::kittens::warp::mma_ABt(accum, a_reg, b_reg, accum);
    }

    // ---- epilogue (NT only): bias add, optional pre-GELU store, GELU ----
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

// ---- kernel_nn : C = A · B ------------------------------------------------

__global__ __launch_bounds__(NUM_THREADS, 2)
void kernel_nn(const __grid_constant__ globals_nn g)
{
    using a_rt     = rt_bf<WARP_M, K_TILE, ducks::rt_layout::row>;
    using b_rt_row = rt_bf<K_TILE, N_TILE, ducks::rt_layout::row>;
    using b_rt_col = rt_bf<K_TILE, N_TILE, ducks::rt_layout::col>;
    using d_rt     = rt_fl<WARP_M, N_TILE, ducks::rt_layout::row>;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int w  = threadIdx.x / ::kittens::WARP_THREADS;

    __shared__ a_smem_nn As[PIPE_STAGES];
    __shared__ b_smem_nn Bs[PIPE_STAGES];

    d_rt accum;
    ::kittens::warp::zero(accum);

    const int num_k_tiles = (int)g.A.cols() / K_TILE;

    ::kittens::group<NUM_WARPS>::load_async(As[0], g.A, {0, 0, by, 0});
    ::kittens::group<NUM_WARPS>::load_async(Bs[0], g.B, {0, 0, 0, bx});

    #pragma unroll 1
    for (int kk = 0; kk < num_k_tiles; ++kk) {
        const int stage = kk % PIPE_STAGES;

        if (kk + 1 < num_k_tiles) {
            const int next = (kk + 1) % PIPE_STAGES;
            ::kittens::group<NUM_WARPS>::load_async(As[next], g.A, {0, 0, by,     kk + 1});
            ::kittens::group<NUM_WARPS>::load_async(Bs[next], g.B, {0, 0, kk + 1, bx});
            ::kittens::group<NUM_WARPS>::template load_async_wait<2>(LOAD_BAR);
        } else {
            ::kittens::group<NUM_WARPS>::template load_async_wait<0>(LOAD_BAR);
        }

        auto As_slice = As[stage].template subtile<WARP_M, K_TILE>({w, 0});
        a_rt     a_reg;
        b_rt_row b_reg_row;
        b_rt_col b_reg_col;
        ::kittens::warp::load(a_reg, As_slice);
        ::kittens::warp::load(b_reg_row, Bs[stage]);
        ::kittens::warp::swap_layout(b_reg_col, b_reg_row);
        ::kittens::warp::mma_AB(accum, a_reg, b_reg_col, accum);
    }

    ::kittens::warp::store(g.C, accum, {0, 0, by * NUM_WARPS + w, bx});
}

// ---- kernel_tn : C = Aᵀ · B -----------------------------------------------

__global__ __launch_bounds__(NUM_THREADS, 2)
void kernel_tn(const __grid_constant__ globals_tn g)
{
    using a_rt_row = rt_bf<K_TILE, WARP_M, ducks::rt_layout::row>;
    using a_rt_col = rt_bf<K_TILE, WARP_M, ducks::rt_layout::col>;
    using b_rt_row = rt_bf<K_TILE, N_TILE, ducks::rt_layout::row>;
    using b_rt_col = rt_bf<K_TILE, N_TILE, ducks::rt_layout::col>;
    using d_rt     = rt_fl<WARP_M, N_TILE, ducks::rt_layout::row>;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int w  = threadIdx.x / ::kittens::WARP_THREADS;

    __shared__ a_smem_tn As[PIPE_STAGES];
    __shared__ b_smem_tn Bs[PIPE_STAGES];

    d_rt accum;
    ::kittens::warp::zero(accum);

    // For TN, A is stored as (K, M) row-major. The cooperative load grabs a
    // K_TILE × M_TILE shared tile; each warp takes a K_TILE × WARP_M slice
    // along the M axis (columns).
    const int num_k_tiles = (int)g.A.rows() / K_TILE;

    ::kittens::group<NUM_WARPS>::load_async(As[0], g.A, {0, 0, 0, by});
    ::kittens::group<NUM_WARPS>::load_async(Bs[0], g.B, {0, 0, 0, bx});

    #pragma unroll 1
    for (int kk = 0; kk < num_k_tiles; ++kk) {
        const int stage = kk % PIPE_STAGES;

        if (kk + 1 < num_k_tiles) {
            const int next = (kk + 1) % PIPE_STAGES;
            ::kittens::group<NUM_WARPS>::load_async(As[next], g.A, {0, 0, kk + 1, by});
            ::kittens::group<NUM_WARPS>::load_async(Bs[next], g.B, {0, 0, kk + 1, bx});
            ::kittens::group<NUM_WARPS>::template load_async_wait<2>(LOAD_BAR);
        } else {
            ::kittens::group<NUM_WARPS>::template load_async_wait<0>(LOAD_BAR);
        }

        // The shared As is K_TILE × M_TILE; warp w owns columns [w*WARP_M, (w+1)*WARP_M).
        auto As_slice = As[stage].template subtile<K_TILE, WARP_M>({0, w});
        a_rt_row a_reg_row;
        a_rt_col a_reg_col;
        b_rt_row b_reg_row;
        b_rt_col b_reg_col;
        ::kittens::warp::load(a_reg_row, As_slice);
        ::kittens::warp::load(b_reg_row, Bs[stage]);
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
// `M_BLOCK` / `N_BLOCK` / `SUPER_M` are unused on sm_120 — the kernel uses a
// fixed 128×64 CTA tile. We carry them as fields so the existing dispatch's
// static references resolve.

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

// Forward (NT): C = A · B^T.
using matmul_default_nt              = matmul_template<2, 4, 8, false, true,  false, false, false>;
using matmul_default_nt_bias         = matmul_template<2, 4, 8, false, true,  true,  false, false>;
using matmul_default_nt_bias_gelu    = matmul_template<2, 4, 8, false, true,  true,  true,  true>;
using matmul_small_n_nt              = matmul_template<2, 2, 8, false, true,  false, false, false>;
using matmul_small_n_nt_bias         = matmul_template<2, 2, 8, false, true,  true,  false, false>;
using matmul_small_n_nt_bias_gelu    = matmul_template<2, 2, 8, false, true,  true,  true,  true>;

// dInp (NN): C = A · B.
using matmul_default                 = matmul_template<2, 4, 8, false, false, false, false, false>;
using matmul_small_n                 = matmul_template<2, 2, 8, false, false, false, false, false>;

// dWeight (TN): C = A^T · B.
using matmul_default_tn              = matmul_template<2, 4, 8, true,  false, false, false, false>;
using matmul_small_n_tn              = matmul_template<2, 2, 8, true,  false, false, false, false>;

// ---- launcher -------------------------------------------------------------

template <typename mmt>
inline void launch(bf16* d_A, bf16* d_B, bf16* d_C, int M, int N, int K,
                   cudaStream_t stream = 0, bf16* d_pre_gelu = nullptr,
                   const bf16* d_bias = nullptr) {
    using namespace sm120_detail;
    assert(M % M_TILE == 0 && "gemm_sm120: M must be a multiple of 128");
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
        a_gl_nt Agl{d_A, nullptr, nullptr, M_, K_};
        b_gl_nt Bgl{d_B, nullptr, nullptr, N_, K_};
        c_gl    Cgl{d_C, nullptr, nullptr, M_, N_};
        c_gl    Pgl{d_pre_gelu == nullptr ? d_C : d_pre_gelu, nullptr, nullptr, M_, N_};
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
