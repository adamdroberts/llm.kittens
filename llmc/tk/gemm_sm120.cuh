/*
gemm_sm120.cuh — ThunderKittens bf16 GEMM for consumer Blackwell (RTX 50-series,
sm_120). Drop-in replacement for gemm_h100.cuh, exposing the same
`llmk::gemm::launch<mmt>(...)` API and the same matmul_default_* / matmul_small_n_*
type aliases that matmul.cuh dispatches on.

v3 design — shape-specialized via a `kernel_traits` bundle, with a SUPER_M
grid swizzle inside the kernels and a per-template tile-shape choice.
  * Each (M_TILE × N_TILE × K_TILE, NUM_WARPS) tuple lives in a
    `kernel_traits<>` instantiation. The three kernels (kernel_nt / _nn / _tn)
    are templated on traits + epilogue flags so multiple tile shapes coexist.
  * Four presets ship today:
      traits_128x64   — 128×64 ×32 with 4 warps  (baseline)
      traits_256x64   — 256×64 ×32 with 8 warps  (more M-reuse per CTA)
      traits_128x128  — 128×128×LLMK_SM120_HUGE_N_K_TILE with 4 warps
                        (bigger N tile for huge-N ops and SM120 dWeight N128)
      grad_*          — same M/N shapes as above, but K=64 for backward GEMMs
    See the public `matmul_default_* / matmul_wide_* / matmul_huge_n_*` alias
    families. The `matmul_template<>` struct carries a `_TRAITS` parameter so
    every existing call site keeps its current behavior (default = 128×64).
  * The K-loop is still the 2-stage cp.async ring buffer from v2 — shared A/B,
    `kittens::group<NUM_WARPS>::load_async` (which emits
    `cp.async.cg.shared.global.L2::128B`) followed by `load_async_wait<2>` so
    the next iter's loads can stay in flight while the current iter MMAs.
  * Inside each kernel we replace the naïve 2-D `(blockIdx.x = bx, blockIdx.y
    = by)` mapping with a SUPER_M grouped-M traversal. Consecutive linear
    blockIdx.x's iterate SUPER_M consecutive M-tiles for the same N-tile,
    re-using the same B (weight) tile in L2 across many A-tiles.

Shape constraints (per traits):
  * M divisible by T::M_TILE
  * N divisible by T::N_TILE
  * K divisible by T::K_TILE
GPT-2 124M dense matmuls satisfy this for all SM120 presets (see plan).

No TMA on sm_120 (consumer Blackwell has cp.async; TMA is sm_90+ with partial
sm_120 support that we don't lean on here).
*/
#pragma once

#include <type_traits>
#include "tk_common.cuh"

namespace llmk::gemm {

using namespace ::kittens;

namespace sm120_detail {

// 4-warp `group<NUM_WARPS>::load_async_wait<N>(bar_id)` syncs on this barrier.
constexpr int LOAD_BAR = 0;

#ifndef LLMK_SM120_K_TILE
#define LLMK_SM120_K_TILE 32
#endif

#ifndef LLMK_SM120_SUPER_M
#define LLMK_SM120_SUPER_M 7
#endif

#ifndef LLMK_SM120_HUGE_N_K_TILE
#define LLMK_SM120_HUGE_N_K_TILE 32
#endif

#ifndef LLMK_SM120_HUGE_N_M256
#define LLMK_SM120_HUGE_N_M256 1
#endif

#ifndef LLMK_SM120_GRAD_K_TILE
#define LLMK_SM120_GRAD_K_TILE 64
#endif

#ifndef LLMK_SM120_DINP_SUPER_M
#define LLMK_SM120_DINP_SUPER_M 8
#endif

#ifndef LLMK_SM120_DWEIGHT_SUPER_M
#define LLMK_SM120_DWEIGHT_SUPER_M 2
#endif

#ifndef LLMK_SM120_DWEIGHT_N128_K_TILE
#define LLMK_SM120_DWEIGHT_N128_K_TILE 16
#endif

#ifndef LLMK_SM120_DINP_DIRECT_BCOL_SUPER_M
#define LLMK_SM120_DINP_DIRECT_BCOL_SUPER_M LLMK_SM120_DINP_SUPER_M
#endif

#ifndef LLMK_SM120_INPLACE_LAYOUT_SWAP
#define LLMK_SM120_INPLACE_LAYOUT_SWAP 1
#endif

#ifndef LLMK_SM120_FAST_DGELU
#define LLMK_SM120_FAST_DGELU 1
#endif

#ifndef LLMK_SM120_APPROX_DGELU_TANH
#define LLMK_SM120_APPROX_DGELU_TANH 1
#endif

__device__ static inline float sm120_dgelu_tanh(float x) {
#if LLMK_SM120_APPROX_DGELU_TANH
    // Fast odd rational approximation, clamped to tanh's range.
    float x2 = x * x;
    float y = x * (27.0f + x2) / (27.0f + 9.0f * x2);
    return fminf(1.0f, fmaxf(-1.0f, y));
#else
    return tanhf(x);
#endif
}

// ---- kernel_traits bundle -------------------------------------------------
//
// Carries the per-CTA tile shape, the per-warp partition, and all derived
// shared-tile / gl<> / globals types. Each kernel template-parameter pack
// resolves to one instantiation per traits.

template <int _MT, int _NT, int _KT, int _NW>
struct kernel_traits {
    static constexpr int M_TILE      = _MT;
    static constexpr int N_TILE      = _NT;
    static constexpr int K_TILE      = _KT;
    static constexpr int NUM_WARPS   = _NW;
    static_assert(M_TILE % NUM_WARPS == 0, "M_TILE must divide evenly by NUM_WARPS");
    static constexpr int WARP_M      = M_TILE / NUM_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int PIPE_STAGES = 2;

    // NT: C = A · Bᵀ; A is (M, K) row, B is (N, K) row.
    using a_smem_nt = st_bf<M_TILE, K_TILE>;
    using b_smem_nt = st_bf<N_TILE, K_TILE>;
    // NN: C = A · B; A is (M, K) row, B is (K, N) row.
    using a_smem_nn = st_bf<M_TILE, K_TILE>;
    using b_smem_nn = st_bf<K_TILE, N_TILE>;
    // TN: C = Aᵀ · B; A is (K, M) row, B is (K, N) row.
    using a_smem_tn = st_bf<K_TILE, M_TILE>;
    using b_smem_tn = st_bf<K_TILE, N_TILE>;
    // C is (M, N) row; per-warp store granularity = WARP_M × N_TILE.
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
    struct globals_nn { a_gl_nn A; b_gl_nn B; c_gl C; c_gl P; };
    struct globals_tn { a_gl_tn A; b_gl_tn B; c_gl C; };
};

// Concrete presets used by the public mmt aliases.
using traits_128x64  = kernel_traits<128,  64, LLMK_SM120_K_TILE, 4>;
using traits_128x96  = kernel_traits<128,  96, LLMK_SM120_K_TILE, 4>;
using traits_256x64  = kernel_traits<256,  64, LLMK_SM120_K_TILE, 8>;
using traits_128x128 = kernel_traits<128, 128, LLMK_SM120_HUGE_N_K_TILE, 4>;
using traits_256x128 = kernel_traits<256, 128, LLMK_SM120_HUGE_N_K_TILE, 8>;
using traits_dweight_128x128 = kernel_traits<128, 128, LLMK_SM120_DWEIGHT_N128_K_TILE, 4>;
using traits_grad_128x64 = kernel_traits<128, 64, LLMK_SM120_GRAD_K_TILE, 4>;
using traits_grad_128x96 = kernel_traits<128, 96, LLMK_SM120_GRAD_K_TILE, 4>;
using traits_grad_256x64 = kernel_traits<256, 64, LLMK_SM120_GRAD_K_TILE, 8>;

// ---- SUPER_M grid swizzle -------------------------------------------------
//
// Linearizes the 2-D output-tile grid into a 1-D blockIdx.x and walks
// SUPER_M consecutive M-tiles for the same N-tile before stepping N. This
// keeps each B (weight) tile hot in L2 across SUPER_M A (activation) tiles.
//
// Returns false when the linear index has run off the end of the rounded grid.
__device__ static inline bool resolve_tile_coords(
    int  mtiles, int ntiles, int super_m,
    int& by /*M tile*/, int& bx /*N tile*/)
{
    int linear     = blockIdx.x;
    int super_size = super_m * ntiles;
    int super      = linear / super_size;
    int rem        = linear - super * super_size;
    bx             = rem / super_m;
    int local_m    = rem - bx * super_m;
    by             = super * super_m + local_m;
    return by < mtiles;
}

// Grid size that matches resolve_tile_coords() for given mtiles/ntiles/super_m.
__host__ static inline int swizzled_grid_size(int mtiles, int ntiles, int super_m) {
    int rounded_m = ((mtiles + super_m - 1) / super_m) * super_m;
    return ntiles * rounded_m;
}

// ---- kernel_nt : C = A · Bᵀ -----------------------------------------------

template <typename T, int SUPER_M, bool APPLY_BIAS, bool APPLY_GELU, bool STORE_PRE_GELU>
__global__ __launch_bounds__(T::NUM_THREADS, 1)
void kernel_nt(const __grid_constant__ typename T::globals_nt g)
{
    using a_rt = rt_bf<T::WARP_M, T::K_TILE, ducks::rt_layout::row>;
    using b_rt = rt_bf<T::N_TILE, T::K_TILE, ducks::rt_layout::row>;
    using d_rt = rt_fl<T::WARP_M, T::N_TILE, ducks::rt_layout::row>;

    const int mtiles = (int)g.A.rows() / T::M_TILE;
    const int ntiles = (int)g.C.cols() / T::N_TILE;
    int bx, by;
    if (!resolve_tile_coords(mtiles, ntiles, SUPER_M, by, bx)) return;
    const int warp_id = threadIdx.x / ::kittens::WARP_THREADS;
    const int w = (T::NUM_WARPS % 4 == 0)
        ? (warp_id / 4 + (warp_id % 4) * (T::NUM_WARPS / 4))
        : warp_id;

    __shared__ typename T::a_smem_nt As[T::PIPE_STAGES];
    __shared__ typename T::b_smem_nt Bs[T::PIPE_STAGES];

    d_rt accum;
    ::kittens::warp::zero(accum);

    const int num_k_tiles = (int)g.A.cols() / T::K_TILE;

    // Prologue: prefetch stage 0.
    ::kittens::group<T::NUM_WARPS>::load_async(As[0], g.A, {0, 0, by, 0});
    ::kittens::group<T::NUM_WARPS>::load_async(Bs[0], g.B, {0, 0, bx, 0});

    #pragma unroll 1
    for (int kk = 0; kk < num_k_tiles; ++kk) {
        const int stage = kk % T::PIPE_STAGES;

        if (kk + 1 < num_k_tiles) {
            const int next = (kk + 1) % T::PIPE_STAGES;
            ::kittens::group<T::NUM_WARPS>::load_async(As[next], g.A, {0, 0, by, kk + 1});
            ::kittens::group<T::NUM_WARPS>::load_async(Bs[next], g.B, {0, 0, bx, kk + 1});
            ::kittens::group<T::NUM_WARPS>::template load_async_wait<2>(LOAD_BAR);
        } else {
            ::kittens::group<T::NUM_WARPS>::template load_async_wait<0>(LOAD_BAR);
        }

        auto As_slice = As[stage].template subtile<T::WARP_M, T::K_TILE>({w, 0});
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
        ::kittens::warp::store(g.P, accum, {0, 0, by * T::NUM_WARPS + w, bx});
    }

    if constexpr (APPLY_GELU) {
        constexpr float k0 = 0.7978845608028654f;
        constexpr float k1 = 0.044715f;
        #pragma unroll
        for (int i = 0; i < d_rt::height; ++i) {
            #pragma unroll
            for (int j = 0; j < d_rt::width; ++j) {
                auto& tile = accum.tiles[i][j];
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float x0 = tile.data[e].x;
                    float x1 = tile.data[e].y;
                    float c0 = k1 * x0 * x0 * x0;
                    float c1 = k1 * x1 * x1 * x1;
                    tile.data[e].x = 0.5f * x0 * (1.0f + tanhf(k0 * (x0 + c0)));
                    tile.data[e].y = 0.5f * x1 * (1.0f + tanhf(k0 * (x1 + c1)));
                }
            }
        }
    }

    ::kittens::warp::store(g.C, accum, {0, 0, by * T::NUM_WARPS + w, bx});
}

// ---- kernel_nn : C = A · B ------------------------------------------------

template <typename T, int SUPER_M, bool APPLY_DGELU, bool DIRECT_B_COL>
__global__ __launch_bounds__(T::NUM_THREADS, 1)
void kernel_nn(const __grid_constant__ typename T::globals_nn g)
{
    using a_rt     = rt_bf<T::WARP_M, T::K_TILE, ducks::rt_layout::row>;
    using b_rt_row = rt_bf<T::K_TILE, T::N_TILE, ducks::rt_layout::row>;
    using b_rt_col = rt_bf<T::K_TILE, T::N_TILE, ducks::rt_layout::col>;
    using p_rt     = rt_bf<T::WARP_M, T::N_TILE, ducks::rt_layout::row>;
    using d_rt     = rt_fl<T::WARP_M, T::N_TILE, ducks::rt_layout::row>;

    const int mtiles = (int)g.A.rows() / T::M_TILE;
    const int ntiles = (int)g.C.cols() / T::N_TILE;
    int bx, by;
    if (!resolve_tile_coords(mtiles, ntiles, SUPER_M, by, bx)) return;
    const int warp_id = threadIdx.x / ::kittens::WARP_THREADS;
    const int w = (T::NUM_WARPS % 4 == 0)
        ? (warp_id / 4 + (warp_id % 4) * (T::NUM_WARPS / 4))
        : warp_id;

    __shared__ typename T::a_smem_nn As[T::PIPE_STAGES];
    __shared__ typename T::b_smem_nn Bs[T::PIPE_STAGES];

    d_rt accum;
    ::kittens::warp::zero(accum);

    const int num_k_tiles = (int)g.A.cols() / T::K_TILE;

    ::kittens::group<T::NUM_WARPS>::load_async(As[0], g.A, {0, 0, by, 0});
    ::kittens::group<T::NUM_WARPS>::load_async(Bs[0], g.B, {0, 0, 0,  bx});

    #pragma unroll 1
    for (int kk = 0; kk < num_k_tiles; ++kk) {
        const int stage = kk % T::PIPE_STAGES;

        if (kk + 1 < num_k_tiles) {
            const int next = (kk + 1) % T::PIPE_STAGES;
            ::kittens::group<T::NUM_WARPS>::load_async(As[next], g.A, {0, 0, by,     kk + 1});
            ::kittens::group<T::NUM_WARPS>::load_async(Bs[next], g.B, {0, 0, kk + 1, bx});
            ::kittens::group<T::NUM_WARPS>::template load_async_wait<2>(LOAD_BAR);
        } else {
            ::kittens::group<T::NUM_WARPS>::template load_async_wait<0>(LOAD_BAR);
        }

        auto As_slice = As[stage].template subtile<T::WARP_M, T::K_TILE>({w, 0});
        a_rt a_reg;
        ::kittens::warp::load(a_reg, As_slice);
        if constexpr (DIRECT_B_COL) {
            b_rt_col b_reg_col;
            ::kittens::warp::load(b_reg_col, Bs[stage]);
            ::kittens::warp::mma_AB(accum, a_reg, b_reg_col, accum);
        } else {
#if LLMK_SM120_INPLACE_LAYOUT_SWAP
            b_rt_row b_reg_row;
            ::kittens::warp::load(b_reg_row, Bs[stage]);
            auto& b_reg_col = ::kittens::warp::swap_layout_inplace(b_reg_row);
#else
            b_rt_col b_reg_col;
            b_rt_row b_reg_row;
            ::kittens::warp::load(b_reg_row, Bs[stage]);
            ::kittens::warp::swap_layout(b_reg_col, b_reg_row);
#endif
            ::kittens::warp::mma_AB(accum, a_reg, b_reg_col, accum);
        }
    }

    if constexpr (APPLY_DGELU) {
        p_rt pre;
        ::kittens::warp::load(pre, g.P, {0, 0, by * T::NUM_WARPS + w, bx});
        constexpr float k0 = 0.7978845608028654f;
        constexpr float k1 = 0.044715f;
        #pragma unroll
        for (int i = 0; i < d_rt::height; ++i) {
            #pragma unroll
            for (int j = 0; j < d_rt::width; ++j) {
                auto& acc_tile = accum.tiles[i][j];
                auto& pre_tile = pre.tiles[i][j];
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float x0 = (float)pre_tile.data[e].x;
                    float x1 = (float)pre_tile.data[e].y;
                    float x0_sq = x0 * x0;
                    float x1_sq = x1 * x1;
                    float tanh_arg0 = k0 * (x0 + k1 * x0 * x0_sq);
                    float tanh_arg1 = k0 * (x1 + k1 * x1 * x1_sq);
                    float tanh_out0 = sm120_dgelu_tanh(tanh_arg0);
                    float tanh_out1 = sm120_dgelu_tanh(tanh_arg1);
#if LLMK_SM120_FAST_DGELU
                    float sech0 = 1.0f - tanh_out0 * tanh_out0;
                    float sech1 = 1.0f - tanh_out1 * tanh_out1;
#else
                    float cosh_out0 = coshf(tanh_arg0);
                    float cosh_out1 = coshf(tanh_arg1);
                    float sech0 = 1.0f / (cosh_out0 * cosh_out0);
                    float sech1 = 1.0f / (cosh_out1 * cosh_out1);
#endif
                    float grad0 = 0.5f * (1.0f + tanh_out0)
                        + x0 * 0.5f * sech0 * k0 * (1.0f + 3.0f * k1 * x0_sq);
                    float grad1 = 0.5f * (1.0f + tanh_out1)
                        + x1 * 0.5f * sech1 * k0 * (1.0f + 3.0f * k1 * x1_sq);
                    acc_tile.data[e].x *= grad0;
                    acc_tile.data[e].y *= grad1;
                }
            }
        }
    }

    ::kittens::warp::store(g.C, accum, {0, 0, by * T::NUM_WARPS + w, bx});
}

// ---- kernel_tn : C = Aᵀ · B -----------------------------------------------

template <typename T, int SUPER_M, bool ACCUMULATE>
__global__ __launch_bounds__(T::NUM_THREADS, 1)
void kernel_tn(const __grid_constant__ typename T::globals_tn g)
{
    using a_rt_row = rt_bf<T::K_TILE, T::WARP_M, ducks::rt_layout::row>;
    using a_rt_col = rt_bf<T::K_TILE, T::WARP_M, ducks::rt_layout::col>;
    using b_rt_row = rt_bf<T::K_TILE, T::N_TILE, ducks::rt_layout::row>;
    using b_rt_col = rt_bf<T::K_TILE, T::N_TILE, ducks::rt_layout::col>;
    using c_rt     = rt_bf<T::WARP_M, T::N_TILE, ducks::rt_layout::row>;
    using d_rt     = rt_fl<T::WARP_M, T::N_TILE, ducks::rt_layout::row>;

    // For TN, output is (M, N) where M is the "A^T" rows. C.rows() == M_TILE-blocks.
    const int mtiles = (int)g.C.rows() / T::M_TILE;
    const int ntiles = (int)g.C.cols() / T::N_TILE;
    int bx, by;
    if (!resolve_tile_coords(mtiles, ntiles, SUPER_M, by, bx)) return;
    const int warp_id = threadIdx.x / ::kittens::WARP_THREADS;
    const int w = (T::NUM_WARPS % 4 == 0)
        ? (warp_id / 4 + (warp_id % 4) * (T::NUM_WARPS / 4))
        : warp_id;

    __shared__ typename T::a_smem_tn As[T::PIPE_STAGES];
    __shared__ typename T::b_smem_tn Bs[T::PIPE_STAGES];

    d_rt accum;
    ::kittens::warp::zero(accum);

    // For TN, A is (K, M) row-major; iterate K-tiles along rows.
    const int num_k_tiles = (int)g.A.rows() / T::K_TILE;

    ::kittens::group<T::NUM_WARPS>::load_async(As[0], g.A, {0, 0, 0, by});
    ::kittens::group<T::NUM_WARPS>::load_async(Bs[0], g.B, {0, 0, 0, bx});

    #pragma unroll 1
    for (int kk = 0; kk < num_k_tiles; ++kk) {
        const int stage = kk % T::PIPE_STAGES;

        if (kk + 1 < num_k_tiles) {
            const int next = (kk + 1) % T::PIPE_STAGES;
            ::kittens::group<T::NUM_WARPS>::load_async(As[next], g.A, {0, 0, kk + 1, by});
            ::kittens::group<T::NUM_WARPS>::load_async(Bs[next], g.B, {0, 0, kk + 1, bx});
            ::kittens::group<T::NUM_WARPS>::template load_async_wait<2>(LOAD_BAR);
        } else {
            ::kittens::group<T::NUM_WARPS>::template load_async_wait<0>(LOAD_BAR);
        }

        auto As_slice = As[stage].template subtile<T::K_TILE, T::WARP_M>({0, w});
        a_rt_row a_reg_row;
        b_rt_row b_reg_row;
        ::kittens::warp::load(a_reg_row, As_slice);
        ::kittens::warp::load(b_reg_row, Bs[stage]);
#if LLMK_SM120_INPLACE_LAYOUT_SWAP
        auto& a_reg_col = ::kittens::warp::swap_layout_inplace(a_reg_row);
        auto& b_reg_col = ::kittens::warp::swap_layout_inplace(b_reg_row);
#else
        a_rt_col a_reg_col;
        b_rt_col b_reg_col;
        ::kittens::warp::swap_layout(a_reg_col, a_reg_row);
        ::kittens::warp::swap_layout(b_reg_col, b_reg_row);
#endif
        ::kittens::warp::mma_AtB(accum, a_reg_col, b_reg_col, accum);
    }

    if constexpr (ACCUMULATE) {
        c_rt prior_bf;
        d_rt prior_fl;
        ::kittens::warp::load(prior_bf, g.C, {0, 0, by * T::NUM_WARPS + w, bx});
        ::kittens::warp::copy(prior_fl, prior_bf);
        accum += prior_fl;
    }
    ::kittens::warp::store(g.C, accum, {0, 0, by * T::NUM_WARPS + w, bx});
}

} // namespace sm120_detail

// ---- mmt template-tag types -----------------------------------------------
//
// Mirrors the gemm_h100.cuh template-tag surface so matmul.cuh's dispatch
// (which references llmk::gemm::matmul_default_nt etc.) compiles unchanged.
//
// `M_BLOCK` / `N_BLOCK` are unused on sm_120 — the kernel uses the tile shape
// from `_TRAITS`. `_SUPER_M` drives the grid swizzle and is honored.

template <int _M_BLOCK = 2, int _N_BLOCK = 4, int _SUPER_M = 8,
          bool _A_TRANSPOSED = false, bool _B_TRANSPOSED = false,
          bool _APPLY_BIAS = false, bool _APPLY_GELU = false,
          bool _STORE_PRE_GELU = false,
          typename _TRAITS = sm120_detail::traits_128x64,
          bool _DIRECT_B_COL = false>
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK;
    static constexpr int N_BLOCK = _N_BLOCK;
    static constexpr int SUPER_M = _SUPER_M;
    static constexpr bool A_TRANSPOSED   = _A_TRANSPOSED;
    static constexpr bool B_TRANSPOSED   = _B_TRANSPOSED;
    static constexpr bool APPLY_BIAS     = _APPLY_BIAS;
    static constexpr bool APPLY_GELU     = _APPLY_GELU;
    static constexpr bool STORE_PRE_GELU = _STORE_PRE_GELU;
    static constexpr bool DIRECT_B_COL   = _DIRECT_B_COL;
    using traits = _TRAITS;
};

// --- Forward (NT, C = A · Bᵀ) ---
// Default tile: 128×64. Wide variant: 256×64. Huge-N defaults to 256×128 on
// SM120, with the original 128×128 variant kept behind LLMK_SM120_HUGE_N_M256=0.

using matmul_default_nt              = matmul_template<2, 4, LLMK_SM120_SUPER_M, false, true,  false, false, false, sm120_detail::traits_128x64>;
using matmul_default_nt_bias         = matmul_template<2, 4, LLMK_SM120_SUPER_M, false, true,  true,  false, false, sm120_detail::traits_128x64>;
using matmul_default_nt_bias_gelu    = matmul_template<2, 4, LLMK_SM120_SUPER_M, false, true,  true,  true,  true,  sm120_detail::traits_128x64>;
using matmul_small_n_nt              = matmul_template<2, 2, LLMK_SM120_SUPER_M, false, true,  false, false, false, sm120_detail::traits_128x64>;
using matmul_small_n_nt_bias         = matmul_template<2, 2, LLMK_SM120_SUPER_M, false, true,  true,  false, false, sm120_detail::traits_128x64>;
using matmul_small_n_nt_bias_gelu    = matmul_template<2, 2, LLMK_SM120_SUPER_M, false, true,  true,  true,  true,  sm120_detail::traits_128x64>;

using matmul_wide_nt                 = matmul_template<2, 4, LLMK_SM120_SUPER_M, false, true,  false, false, false, sm120_detail::traits_256x64>;
using matmul_wide_nt_bias            = matmul_template<2, 4, LLMK_SM120_SUPER_M, false, true,  true,  false, false, sm120_detail::traits_256x64>;
using matmul_wide_nt_bias_gelu       = matmul_template<2, 4, LLMK_SM120_SUPER_M, false, true,  true,  true,  true,  sm120_detail::traits_256x64>;

using matmul_n96_nt                  = matmul_template<2, 4, LLMK_SM120_SUPER_M, false, true,  false, false, false, sm120_detail::traits_128x96>;
using matmul_n96_nt_bias             = matmul_template<2, 4, LLMK_SM120_SUPER_M, false, true,  true,  false, false, sm120_detail::traits_128x96>;
using matmul_n96_nt_bias_gelu        = matmul_template<2, 4, LLMK_SM120_SUPER_M, false, true,  true,  true,  true,  sm120_detail::traits_128x96>;

#if LLMK_SM120_HUGE_N_M256
using matmul_huge_n_nt               = matmul_template<2, 4, LLMK_SM120_SUPER_M, false, true,  false, false, false, sm120_detail::traits_256x128>;
using matmul_huge_n_nt_bias          = matmul_template<2, 4, LLMK_SM120_SUPER_M, false, true,  true,  false, false, sm120_detail::traits_256x128>;
using matmul_huge_n_nt_bias_gelu     = matmul_template<2, 4, LLMK_SM120_SUPER_M, false, true,  true,  true,  true,  sm120_detail::traits_256x128>;
#else
using matmul_huge_n_nt               = matmul_template<2, 4, LLMK_SM120_SUPER_M, false, true,  false, false, false, sm120_detail::traits_128x128>;
using matmul_huge_n_nt_bias          = matmul_template<2, 4, LLMK_SM120_SUPER_M, false, true,  true,  false, false, sm120_detail::traits_128x128>;
using matmul_huge_n_nt_bias_gelu     = matmul_template<2, 4, LLMK_SM120_SUPER_M, false, true,  true,  true,  true,  sm120_detail::traits_128x128>;
#endif

// --- dInp (NN, C = A · B) ---
using matmul_default                 = matmul_template<2, 4, LLMK_SM120_DINP_SUPER_M, false, false, false, false, false, sm120_detail::traits_grad_128x64>;
using matmul_small_n                 = matmul_template<2, 2, LLMK_SM120_DINP_SUPER_M, false, false, false, false, false, sm120_detail::traits_grad_128x64>;
using matmul_n96                     = matmul_template<2, 4, LLMK_SM120_DINP_SUPER_M, false, false, false, false, false, sm120_detail::traits_grad_128x96>;
using matmul_wide                    = matmul_template<2, 4, LLMK_SM120_DINP_SUPER_M, false, false, false, false, false, sm120_detail::traits_grad_256x64>;
using matmul_huge_n                  = matmul_template<2, 4, LLMK_SM120_SUPER_M, false, false, false, false, false, sm120_detail::traits_128x128>;
using matmul_default_direct_bcol     = matmul_template<2, 4, LLMK_SM120_DINP_DIRECT_BCOL_SUPER_M, false, false, false, false, false, sm120_detail::traits_grad_128x64, true>;
using matmul_default_dgelu           = matmul_template<2, 4, LLMK_SM120_DINP_SUPER_M, false, false, false, true,  false, sm120_detail::traits_grad_128x64>;
using matmul_small_n_dgelu           = matmul_template<2, 2, LLMK_SM120_DINP_SUPER_M, false, false, false, true,  false, sm120_detail::traits_grad_128x64>;
using matmul_n96_dgelu               = matmul_template<2, 4, LLMK_SM120_DINP_SUPER_M, false, false, false, true,  false, sm120_detail::traits_grad_128x96>;
using matmul_wide_dgelu              = matmul_template<2, 4, LLMK_SM120_DINP_SUPER_M, false, false, false, true,  false, sm120_detail::traits_grad_256x64>;
using matmul_huge_n_dgelu            = matmul_template<2, 4, LLMK_SM120_SUPER_M, false, false, false, true,  false, sm120_detail::traits_128x128>;

// --- dWeight (TN, C = Aᵀ · B) ---
using matmul_default_tn              = matmul_template<2, 4, LLMK_SM120_DWEIGHT_SUPER_M, true,  false, false, false, false, sm120_detail::traits_grad_128x64>;
using matmul_small_n_tn              = matmul_template<2, 2, LLMK_SM120_DWEIGHT_SUPER_M, true,  false, false, false, false, sm120_detail::traits_grad_128x64>;
using matmul_n96_tn                  = matmul_template<2, 4, LLMK_SM120_DWEIGHT_SUPER_M, true,  false, false, false, false, sm120_detail::traits_grad_128x96>;
using matmul_n128_tn                 = matmul_template<2, 4, LLMK_SM120_DWEIGHT_SUPER_M, true,  false, false, false, false, sm120_detail::traits_dweight_128x128>;
using matmul_wide_tn                 = matmul_template<2, 4, LLMK_SM120_DWEIGHT_SUPER_M, true,  false, false, false, false, sm120_detail::traits_grad_256x64>;
using matmul_huge_n_tn               = matmul_template<2, 4, LLMK_SM120_SUPER_M, true,  false, false, false, false, sm120_detail::traits_dweight_128x128>;

// ---- launcher -------------------------------------------------------------

template <typename mmt>
inline void launch(bf16* d_A, bf16* d_B, bf16* d_C, int M, int N, int K,
                   cudaStream_t stream = 0, bf16* d_pre_gelu = nullptr,
                   const bf16* d_bias = nullptr, bool accumulate = false) {
    using namespace sm120_detail;
    using T = typename mmt::traits;
    assert(M % T::M_TILE == 0 && "gemm_sm120: M must be a multiple of the kernel's M_TILE");
    assert(N % T::N_TILE == 0 && "gemm_sm120: N must be a multiple of the kernel's N_TILE");
    assert(K % T::K_TILE == 0 && "gemm_sm120: K must be a multiple of the kernel's K_TILE");

    const size_t M_ = static_cast<size_t>(M);
    const size_t N_ = static_cast<size_t>(N);
    const size_t K_ = static_cast<size_t>(K);

    const int mtiles = M / T::M_TILE;
    const int ntiles = N / T::N_TILE;
    dim3 grid(swizzled_grid_size(mtiles, ntiles, mmt::SUPER_M));
    dim3 block(T::NUM_THREADS);

    if constexpr (mmt::A_TRANSPOSED && mmt::B_TRANSPOSED) {
        static_assert(!(mmt::A_TRANSPOSED && mmt::B_TRANSPOSED),
            "gemm_sm120: A_TRANSPOSED && B_TRANSPOSED not implemented (unused).");
    } else if constexpr (!mmt::A_TRANSPOSED && mmt::B_TRANSPOSED) {
        typename T::a_gl_nt Agl{d_A, nullptr, nullptr, M_, K_};
        typename T::b_gl_nt Bgl{d_B, nullptr, nullptr, N_, K_};
        typename T::c_gl    Cgl{d_C, nullptr, nullptr, M_, N_};
        typename T::c_gl    Pgl{d_pre_gelu == nullptr ? d_C : d_pre_gelu, nullptr, nullptr, M_, N_};
        const bf16* bias_ptr = d_bias != nullptr ? d_bias : d_C;
        typename T::bias_gl Bias{const_cast<bf16*>(bias_ptr), nullptr, nullptr, nullptr, N_};
        typename T::globals_nt g{Agl, Bgl, Cgl, Pgl, Bias, d_bias != nullptr ? 1 : 0};

        auto kfn = kernel_nt<T, mmt::SUPER_M, mmt::APPLY_BIAS, mmt::APPLY_GELU, mmt::STORE_PRE_GELU>;
        kfn<<<grid, block, 0, stream>>>(g);
    } else if constexpr (!mmt::A_TRANSPOSED && !mmt::B_TRANSPOSED) {
        static_assert(!mmt::APPLY_BIAS && !mmt::STORE_PRE_GELU,
            "gemm_sm120: bias/pre-GELU epilogues only available on NT path.");
        typename T::a_gl_nn Agl{d_A, nullptr, nullptr, M_, K_};
        typename T::b_gl_nn Bgl{d_B, nullptr, nullptr, K_, N_};
        typename T::c_gl    Cgl{d_C, nullptr, nullptr, M_, N_};
        typename T::c_gl    Pgl{d_pre_gelu == nullptr ? d_C : d_pre_gelu, nullptr, nullptr, M_, N_};
        typename T::globals_nn g{Agl, Bgl, Cgl, Pgl};
        auto kfn = kernel_nn<T, mmt::SUPER_M, mmt::APPLY_GELU, mmt::DIRECT_B_COL>;
        kfn<<<grid, block, 0, stream>>>(g);
    } else {
        static_assert(!mmt::APPLY_BIAS && !mmt::APPLY_GELU && !mmt::STORE_PRE_GELU,
            "gemm_sm120: bias/gelu epilogues only available on NT path.");
        typename T::a_gl_tn Agl{d_A, nullptr, nullptr, K_, M_};
        typename T::b_gl_tn Bgl{d_B, nullptr, nullptr, K_, N_};
        typename T::c_gl    Cgl{d_C, nullptr, nullptr, M_, N_};
        typename T::globals_tn g{Agl, Bgl, Cgl};
        if (accumulate) {
            auto kfn = kernel_tn<T, mmt::SUPER_M, true>;
            kfn<<<grid, block, 0, stream>>>(g);
        } else {
            auto kfn = kernel_tn<T, mmt::SUPER_M, false>;
            kfn<<<grid, block, 0, stream>>>(g);
        }
    }
}

} // namespace llmk::gemm
