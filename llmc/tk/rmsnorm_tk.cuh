/*
rmsnorm_tk.cuh — ThunderKittens bf16 RMSNorm forward wrappers.

This mirrors layernorm_tk.cuh without mean subtraction or bias. Backward remains
in the plain CUDA wrapper because it is reduction-heavy and easier to validate
before deciding whether a TK rewrite is worth it.
*/
#pragma once

#include "tk_common.cuh"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

namespace llmk::rmsnorm {

using namespace ::kittens;

constexpr int NUM_WORKERS = 2;
constexpr int NUM_THREADS = NUM_WORKERS * ::kittens::WARP_THREADS;

template <int D>
struct rmsnorm_globals {
    using vec_smem_1xD = sv_bf<D>;

    using x_gl = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using residual_gl = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using out_gl = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using norm_weight_gl = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;

    x_gl x;
    residual_gl residual;
    out_gl out;
    out_gl out_residual;
    norm_weight_gl norm_weight;
    float* rstd;
    int rows;
    float eps;
};

template <int D, bool FusedResidual>
__global__ __launch_bounds__(NUM_THREADS, 1)
void rmsnorm_forward_ker(const __grid_constant__ rmsnorm_globals<D> g) {
    const int warp_id = kittens::warpid();
    const int lane = kittens::laneid();
    const int row = blockIdx.x * NUM_WORKERS + warp_id;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using vec_smem_1xD = sv_bf<D>;
    vec_smem_1xD (&x_s)[NUM_WORKERS] = al.allocate<vec_smem_1xD, NUM_WORKERS>();
    vec_smem_1xD (&residual_s)[NUM_WORKERS] = al.allocate<vec_smem_1xD, NUM_WORKERS>();
    vec_smem_1xD (&norm_weight_s) = al.allocate<vec_smem_1xD>();

    if (warp_id == 0) {
        warp::load(norm_weight_s, g.norm_weight, {0, 0, 0, 0});
    }
    __syncthreads();

    if (row >= g.rows) {
        return;
    }

    warp::load(x_s[warp_id], g.x, {0, 0, row, 0});
    __syncwarp();

    if constexpr (FusedResidual) {
        warp::load(residual_s[warp_id], g.residual, {0, 0, row, 0});
        __syncwarp();
        warp::add(x_s[warp_id], x_s[warp_id], residual_s[warp_id]);
        warp::store(g.out_residual, x_s[warp_id], {0, 0, row, 0});
        __syncwarp();
    }

    warp::mul(residual_s[warp_id], x_s[warp_id], x_s[warp_id]);
    bf16 sumsq_bf = __float2bfloat16(0.0f);
    warp::sum(sumsq_bf, residual_s[warp_id]);
    const float sumsq_f = __bfloat162float(sumsq_bf);
    const float rstd_f = rsqrtf(sumsq_f / static_cast<float>(D) + g.eps);
    const bf16 rstd_bf = __float2bfloat16(rstd_f);

    if (lane == 0 && g.rstd != nullptr) {
        g.rstd[row] = rstd_f;
    }

    warp::mul(x_s[warp_id], x_s[warp_id], rstd_bf);
    warp::mul(x_s[warp_id], x_s[warp_id], norm_weight_s);
    __syncwarp();

    warp::store(g.out, x_s[warp_id], {0, 0, row, 0});
}

template <int D, bool FusedResidual>
inline void launch_impl(bf16* out, bf16* out_residual, float* rstd,
                        const bf16* x, const bf16* residual,
                        const bf16* weight, int rows, float eps,
                        cudaStream_t stream) {
    static_assert(D == 768 || D == 1024 || D == 1280 || D == 1600 ||
                  D == 2048 || D == 4096);
    assert(rows >= 0);

    using globals = rmsnorm_globals<D>;
    using vec_smem_1xD = typename globals::vec_smem_1xD;
    using x_global = typename globals::x_gl;
    using residual_global = typename globals::residual_gl;
    using out_global = typename globals::out_gl;
    using weight_global = typename globals::norm_weight_gl;

    x_global x_arg{const_cast<bf16*>(x), 1U, 1U, static_cast<unsigned int>(rows),
                   static_cast<unsigned int>(D)};
    residual_global residual_arg{const_cast<bf16*>(residual), 1U, 1U,
                                 static_cast<unsigned int>(rows),
                                 static_cast<unsigned int>(D)};
    out_global out_arg{out, 1U, 1U, static_cast<unsigned int>(rows),
                       static_cast<unsigned int>(D)};
    out_global out_residual_arg{out_residual, 1U, 1U, static_cast<unsigned int>(rows),
                                static_cast<unsigned int>(D)};
    weight_global weight_arg{const_cast<bf16*>(weight), 1U, 1U, 1U,
                             static_cast<unsigned int>(D)};

    globals g{x_arg, residual_arg, out_arg, out_residual_arg, weight_arg, rstd, rows, eps};

    constexpr int vecs_per_block = 2 * NUM_WORKERS + 1;
    const unsigned long mem_size = sizeof(vec_smem_1xD) * vecs_per_block;

    static bool smem_attr_set = false;
    if (!smem_attr_set) {
        cudaCheck(cudaFuncSetAttribute(
            rmsnorm_forward_ker<D, FusedResidual>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size));
        smem_attr_set = true;
    }

    dim3 grid(CEIL_DIV(rows, NUM_WORKERS));
    rmsnorm_forward_ker<D, FusedResidual><<<grid, NUM_THREADS, mem_size, stream>>>(g);
}

template <int D>
inline void launch_forward(bf16* out, float* rstd, const bf16* x,
                           const bf16* weight, int rows, float eps,
                           cudaStream_t stream) {
    launch_impl<D, false>(out, nullptr, rstd, x, nullptr, weight, rows, eps, stream);
}

template <int D>
inline void launch_fused_residual_forward(bf16* residual_out, bf16* normed,
                                          float* rstd,
                                          const bf16* inp1, const bf16* inp2,
                                          const bf16* weight, int rows, float eps,
                                          cudaStream_t stream) {
    launch_impl<D, true>(normed, residual_out, rstd, inp1, inp2, weight, rows, eps, stream);
}

inline bool supports_width(int C) {
    return C == 768 || C == 1024 || C == 1280 || C == 1600 || C == 2048 || C == 4096;
}

inline void launch_forward(bf16* out, float* rstd, const bf16* x,
                           const bf16* weight, int rows, int C, float eps,
                           cudaStream_t stream) {
    switch (C) {
        case 768:  launch_forward<768>(out, rstd, x, weight, rows, eps, stream); break;
        case 1024: launch_forward<1024>(out, rstd, x, weight, rows, eps, stream); break;
        case 1280: launch_forward<1280>(out, rstd, x, weight, rows, eps, stream); break;
        case 1600: launch_forward<1600>(out, rstd, x, weight, rows, eps, stream); break;
        case 2048: launch_forward<2048>(out, rstd, x, weight, rows, eps, stream); break;
        case 4096: launch_forward<4096>(out, rstd, x, weight, rows, eps, stream); break;
        default:
            fprintf(stderr, "rmsnorm_forward: TK RMSNorm unsupported width C=%d\n", C);
            exit(EXIT_FAILURE);
    }
}

inline void launch_fused_residual_forward(bf16* residual_out, bf16* normed,
                                          float* rstd,
                                          const bf16* inp1, const bf16* inp2,
                                          const bf16* weight,
                                          int rows, int C, float eps,
                                          cudaStream_t stream) {
    switch (C) {
        case 768:  launch_fused_residual_forward<768>(residual_out, normed, rstd, inp1, inp2, weight, rows, eps, stream); break;
        case 1024: launch_fused_residual_forward<1024>(residual_out, normed, rstd, inp1, inp2, weight, rows, eps, stream); break;
        case 1280: launch_fused_residual_forward<1280>(residual_out, normed, rstd, inp1, inp2, weight, rows, eps, stream); break;
        case 1600: launch_fused_residual_forward<1600>(residual_out, normed, rstd, inp1, inp2, weight, rows, eps, stream); break;
        case 2048: launch_fused_residual_forward<2048>(residual_out, normed, rstd, inp1, inp2, weight, rows, eps, stream); break;
        case 4096: launch_fused_residual_forward<4096>(residual_out, normed, rstd, inp1, inp2, weight, rows, eps, stream); break;
        default:
            fprintf(stderr, "fused_residual_rmsnorm_forward: TK RMSNorm unsupported width C=%d\n", C);
            exit(EXIT_FAILURE);
    }
}

} // namespace llmk::rmsnorm
