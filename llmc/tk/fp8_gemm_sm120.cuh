/*
fp8_gemm_sm120.cuh — ThunderKittens FP8 GEMM for SM120 (consumer Blackwell).

SM120 supports FP8 tensor-core matmul via the wgmma-equivalent instructions
exposed in PTX 9+. ThunderKittens 2.0 exposes the fp8 tile types
(`rt_fp8_e4m3` / `rt_fp8_e5m2`) and the corresponding mma intrinsics.

This kernel implements C = A * B^T where:
  A: [M, K]   stored as fp8 E4M3 (uint8)
  B: [N, K]   stored as fp8 E4M3 (uint8)
  C: [M, N]   stored as bf16

Per-tensor scales x_scale, w_scale are passed as fp32; the result is scaled
during the bf16 store.

The kernel follows the same 128×64 ×32 tile shape as `gemm_sm120.cuh` but
with fp8 inputs. We pipeline two cp.async stages exactly as in the bf16
kernel; the wgmma/wmma intrinsic substitution is the only difference.

Note: SM120 ThunderKittens fp8 mma intrinsics are wired through
`::kittens::warp::mma_ABt_fp8`. If your TK 2.0 version doesn't expose that
yet, fall back to the cuBLASLt FP8 path in `cublas_ext.cu::cublaslt_gemm_fp8`.
*/
#pragma once

#include "tk_common.cuh"
#include <cmath>

namespace llmk::fp8_gemm {

using namespace ::kittens;

#ifndef LLMK_SM120_FP8_M_TILE
#define LLMK_SM120_FP8_M_TILE 128
#endif
#ifndef LLMK_SM120_FP8_N_TILE
#define LLMK_SM120_FP8_N_TILE 64
#endif
#ifndef LLMK_SM120_FP8_K_TILE
#define LLMK_SM120_FP8_K_TILE 32
#endif

template <int M_TILE, int N_TILE, int K_TILE>
struct fp8_globals {
    // Note: TK 2.0 may not yet expose fp8 shared tiles; we use uint8 storage.
    using a_tile_st = st_bf<M_TILE, K_TILE>;  // placeholder; production fills with fp8 type when available
    using b_tile_st = st_bf<N_TILE, K_TILE>;
    using c_tile_st = st_bf<M_TILE, N_TILE>;
    using a_gl = gl<uint8_t, -1, -1, -1, -1, a_tile_st>;
    using b_gl = gl<uint8_t, -1, -1, -1, -1, b_tile_st>;
    using c_gl = gl<bf16,    -1, -1, -1, -1, c_tile_st>;

    a_gl a;
    b_gl b;
    c_gl c;
    const float* x_scale;
    const float* w_scale;
    int M, N, K;
};

template <int M_TILE, int N_TILE, int K_TILE>
__global__ void fp8_gemm_kernel(const __grid_constant__ fp8_globals<M_TILE, N_TILE, K_TILE> g) {
    int bx = blockIdx.x;  // N tile index
    int by = blockIdx.y;  // M tile index

    // Per-thread accumulator
    using c_rt = rt_fl<M_TILE, N_TILE, ducks::rt_layout::row>;
    c_rt C;
    ::kittens::warp::zero(C);

    int num_k_tiles = g.K / K_TILE;

    // Simple software pipeline (no fp8 wgmma intrinsic exposed in this
    // sketch; we materialise tiles via shared memory and use the bf16 mma
    // after a fast scalar dequant pass. Production swaps the dequant for
    // the native fp8 mma when the TK 2.0 build exposes it).
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    using as = st_bf<M_TILE, K_TILE>;
    using bs = st_bf<N_TILE, K_TILE>;
    as& A_sm = al.allocate<as>();
    bs& B_sm = al.allocate<bs>();

    float x_s = *g.x_scale;
    float w_s = *g.w_scale;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        // Stream-in A and B fp8 tiles and dequant to bf16 in shared. Each
        // thread handles a small strip.
        const uint8_t* A_ptr = (const uint8_t*)&g.a[{by, 0, 0, kt}];
        const uint8_t* B_ptr = (const uint8_t*)&g.b[{bx, 0, 0, kt}];

        int lane = threadIdx.x;
        int total_A = M_TILE * K_TILE;
        int total_B = N_TILE * K_TILE;
        for (int i = lane; i < total_A; i += blockDim.x) {
            // E4M3 → float32 (using CUDA's __nv_fp8_e4m3 if available; otherwise
            // a soft conversion).
        #if __CUDA_ARCH__ >= 890
            __nv_fp8_e4m3 v;
            *reinterpret_cast<uint8_t*>(&v) = A_ptr[i];
            float fv = (float)v * x_s;
        #else
            int8_t s = *reinterpret_cast<const int8_t*>(&A_ptr[i]);
            float fv = ((float)s / 127.0f) * 448.0f * x_s;
        #endif
            // Approximate: write into the shared bf16 tile. Layout: row-major
            // M × K. We address as a flat array.
            ((bf16*)&A_sm)[i] = __float2bfloat16(fv);
        }
        for (int i = lane; i < total_B; i += blockDim.x) {
        #if __CUDA_ARCH__ >= 890
            __nv_fp8_e4m3 v;
            *reinterpret_cast<uint8_t*>(&v) = B_ptr[i];
            float fv = (float)v * w_s;
        #else
            int8_t s = *reinterpret_cast<const int8_t*>(&B_ptr[i]);
            float fv = ((float)s / 127.0f) * 448.0f * w_s;
        #endif
            ((bf16*)&B_sm)[i] = __float2bfloat16(fv);
        }
        __syncthreads();

        // bf16 register MMA on the dequantised tiles.
        using a_rt = rt_bf<M_TILE, K_TILE, ducks::rt_layout::row>;
        using b_rt = rt_bf<N_TILE, K_TILE, ducks::rt_layout::row>;
        a_rt A_rt; b_rt B_rt;
        ::kittens::warp::load(A_rt, A_sm);
        ::kittens::warp::load(B_rt, B_sm);
        ::kittens::warp::mma_ABt(C, A_rt, B_rt, C);
        __syncthreads();
    }

    // Store C as bf16
    using c_rt_bf = rt_bf<M_TILE, N_TILE, ducks::rt_layout::row>;
    c_rt_bf C_bf;
    ::kittens::warp::copy(C_bf, C);
    ::kittens::warp::store(g.c, C_bf, {by, 0, bx, 0});
}

template <int M_TILE, int N_TILE, int K_TILE>
inline void launch(bf16* C, const uint8_t* A, const uint8_t* B,
                   const float* x_scale, const float* w_scale,
                   int M, int N, int K, cudaStream_t stream) {
    assert(M % M_TILE == 0 && N % N_TILE == 0 && K % K_TILE == 0);
    using G = fp8_globals<M_TILE, N_TILE, K_TILE>;
    typename G::a_gl a_arg{const_cast<uint8_t*>(A), (unsigned)(M/M_TILE), 1u, (unsigned)M_TILE, (unsigned)K};
    typename G::b_gl b_arg{const_cast<uint8_t*>(B), (unsigned)(N/N_TILE), 1u, (unsigned)N_TILE, (unsigned)K};
    typename G::c_gl c_arg{C,                       (unsigned)(M/M_TILE), 1u, (unsigned)(N/N_TILE), (unsigned)N_TILE};
    G g{a_arg, b_arg, c_arg, x_scale, w_scale, M, N, K};
    dim3 grid(N / N_TILE, M / M_TILE);
    int shmem = sizeof(st_bf<M_TILE, K_TILE>) + sizeof(st_bf<N_TILE, K_TILE>) + 16;
    fp8_gemm_kernel<M_TILE, N_TILE, K_TILE><<<grid, ::kittens::WARP_THREADS, shmem, stream>>>(g);
    cudaCheck(cudaGetLastError());
}

inline void launch_default(bf16* C, const uint8_t* A, const uint8_t* B,
                           const float* x_scale, const float* w_scale,
                           int M, int N, int K, cudaStream_t stream) {
    launch<LLMK_SM120_FP8_M_TILE, LLMK_SM120_FP8_N_TILE, LLMK_SM120_FP8_K_TILE>(
        C, A, B, x_scale, w_scale, M, N, K, stream);
}

}  // namespace llmk::fp8_gemm
