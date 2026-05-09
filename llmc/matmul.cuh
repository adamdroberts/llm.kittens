/*
matmul.cuh — C-style wrapper around ThunderKittens H100 bf16 GEMM.

Replaces llm.c's cuBLASLt-backed matmul. The forward signature is
intentionally close to llm.c's `matmul_forward_cublaslt`, so train_gpt2.cu
can be ported with minimal changes. v1 differences:

  * The default forward path keeps bias as a separate pointwise pass. The GPT-2
    MLP up-projection can opt into a TK finish-path bias+GELU epilogue via
    `matmul_forward_gelu`, storing pre-GELU for backward while writing GELU
    output.
  * Forward uses the TK A*B^T variant because llm.c parameter files store
    dense weights as (OC, C), not as the GEMM-friendly (C, OC).
  * Backward dInp uses the existing TK A*B path. Backward dWeight uses TK A^T*B;
    accumulated micro-steps compute into caller scratch, then add into the
    gradient tensor. The bias-grad reduction kernels are ported verbatim from
    llm.c — TK gives no benefit on a column reduction.

Shape constraints (must hold; we do NOT pad on the fly):
  * M (= B*T) divisible by 128
  * N (= OC) divisible by 256 for the default kernel; 128 for the small-N
    fallback used by the GPT-2 LM-head projection
  * K (= C) divisible by  64

For GPT-2 124M with B=4 T=1024 C=768 these all hold; for the LM-head projection
M=4096 N=V_padded=50304 K=768 the small-N fallback is selected automatically.
*/
#pragma once

#include <assert.h>
#include <type_traits>
#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "tk/gemm_h100.cuh"

// ----------------------------------------------------------------------------
// Bias-grad reduction kernels — ported verbatim from llm.c/llmc/matmul.cuh
// (lines 17-102). These are plain CUDA; TK adds nothing.

template <typename OutFloat, bool UseAuxBuffer>
__global__ void matmul_backward_bias_kernel9(OutFloat* dbias, const floatX* dout,
                                             int B, int T, int OC,
                                             std::bool_constant<UseAuxBuffer>) {
    constexpr const int bdx = 4;
    constexpr const int bdy = WARP_SIZE / bdx;
    assert(blockDim.x == bdx);
    assert(blockDim.y == bdy);

    int warp_d  = (int)threadIdx.x;
    int warp_c  = (int)threadIdx.y;
    int block_d = (int)threadIdx.z;

    const int OC_per_warp = bdy * x128::size;

    int local_oc  = warp_c * x128::size;
    int global_oc = blockIdx.x * OC_per_warp + local_oc;

    int local_bt     = warp_d + bdx * block_d;
    int bt_per_block = bdx * blockDim.z;

    float accumulators[x128::size];
    for (int k = 0; k < x128::size; k++) accumulators[k] = 0.0f;

    if (global_oc < OC) {
        for (int idx = blockIdx.y * bt_per_block + local_bt; idx < B * T;
             idx += gridDim.y * bt_per_block) {
            x128 packed_dout = load128(dout + global_oc + idx * OC);
            for (int k = 0; k < x128::size; k++)
                accumulators[k] += (float)packed_dout[k];
        }
    }

    __shared__ float sub_results[x128::size][WARP_SIZE][bdy];

    for (int k = 0; k < x128::size; k++) {
        float v = accumulators[k];
        v += __shfl_down_sync(0xffffffff, v, 1, 4);
        v += __shfl_down_sync(0xffffffff, v, 2, 4);
        if (warp_d == 0) sub_results[k][block_d][warp_c] = v;
    }
    __syncthreads();

    for (int k = block_d; k < x128::size; k += blockDim.z) {
        float a = 0.f;
        for (int r = warp_d; r < blockDim.z; r += bdx) {
            float v = sub_results[k][r][warp_c];
            v += __shfl_down_sync(0xffffffff, v, 1, 4);
            v += __shfl_down_sync(0xffffffff, v, 2, 4);
            a += v;
        }
        if (warp_d == 0 && global_oc < OC) {
            if constexpr (!UseAuxBuffer) {
                dbias[global_oc + k] = (OutFloat)(a + (float)dbias[global_oc + k]);
            } else {
                dbias[global_oc + k + blockIdx.y * OC] = a;
            }
        }
    }
}

__global__ void reduce_add_sum_kernel(floatX* dst, const float* src, size_t n, size_t m) {
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * f128::size;
    assert(n % x128::size == 0);
    if (idx < n) {
        f128 acc;
        for (int k = 0; k < f128::size; ++k) acc[k] = 0.f;
        for (int l = 0; l < m; ++l) {
            f128 s = load128(src + idx + n * l);
            for (int k = 0; k < f128::size; ++k) acc[k] += s[k];
        }
        for (int k = 0; k < f128::size; ++k)
            dst[idx + k] = (floatX)((float)dst[idx + k] + acc[k]);
    }
}

// ----------------------------------------------------------------------------
// Bias-add kernel for the default forward path. The opt-in MLP up-projection
// path can fold bias into the TK finish stage via matmul_forward_gelu().

__global__ void add_bias_kernel(floatX* out, const floatX* bias, int N, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * OC) return;
    int oc = idx % OC;
    out[idx] = (floatX)((float)out[idx] + (float)bias[oc]);
}

inline void add_bias(floatX* out, const floatX* bias, int N, int OC, cudaStream_t stream) {
    if (bias == nullptr) return;
    const int block = 256;
    const int grid  = CEIL_DIV(N * OC, block);
    add_bias_kernel<<<grid, block, 0, stream>>>(out, bias, N, OC);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Forward.
//
// out (B*T, OC) = inp (B*T, C) · weight(OC, C)^T + bias (OC)
//
// Default forward keeps bias and GELU as separate passes. GPT-2 MLP up-projection
// can opt into a TK finish-path bias+GELU epilogue via matmul_forward_gelu(),
// which stores the pre-GELU buffer needed by backward.

inline void matmul_forward(floatX* out, const floatX* inp, const floatX* weight,
                           const floatX* bias, int B, int T, int C, int OC,
                           cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int M = B * T;
    const int N = OC;
    const int K = C;

    // Dispatch on shape constraints. Default kernel needs N % 256 == 0;
    // small-N fallback only needs N % 128 == 0.
    assert(M % 128 == 0 && "matmul_forward: B*T must be a multiple of 128");
    assert(K % 64  == 0 && "matmul_forward: C must be a multiple of 64");

    auto* A = llmk::to_bf16(const_cast<floatX*>(inp));
    auto* B_= llmk::to_bf16(const_cast<floatX*>(weight));
    auto* C_= llmk::to_bf16(out);

    if (N % 256 == 0) {
        llmk::gemm::launch<llmk::gemm::matmul_default_nt>(A, B_, C_, M, N, K, stream);
    } else {
        assert(N % 128 == 0 && "matmul_forward: OC must be a multiple of 128");
        llmk::gemm::launch<llmk::gemm::matmul_small_n_nt>(A, B_, C_, M, N, K, stream);
    }
    cudaCheck(cudaGetLastError());

    add_bias(out, bias, M, OC, stream);
}

// ----------------------------------------------------------------------------
// Backward CUDA fallback kernels.
//
// matmul_backward prefers TK for dInp and dWeight when the shape is supported.
// These slow kernels remain for unsupported shapes and for the small accumulation
// add after scratch-backed TK dWeight products.

__global__ void matmul_backward_dinp_kernel(floatX* dinp, const floatX* dout,
                                            const floatX* weight, int M, int C, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * C) return;
    int m = idx / C;
    int c = idx % C;
    float acc = 0.0f;
    for (int oc = 0; oc < OC; ++oc) {
        acc += (float)dout[m * OC + oc] * (float)weight[oc * C + c];
    }
    dinp[idx] = (floatX)acc;
}

__global__ void matmul_backward_dweight_kernel(floatX* dweight, const floatX* dout,
                                               const floatX* inp, int M, int C, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * OC) return;
    int oc = idx / C;
    int c = idx % C;
    float acc = 0.0f;
    for (int m = 0; m < M; ++m) {
        acc += (float)inp[m * C + c] * (float)dout[m * OC + oc];
    }
    dweight[idx] = (floatX)((float)dweight[idx] + acc);
}

__global__ void matmul_backward_dbias_kernel(floatX* dbias, const floatX* dout, int M, int OC) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= OC) return;
    float acc = 0.0f;
    for (int m = 0; m < M; ++m) {
        acc += (float)dout[m * OC + oc];
    }
    dbias[oc] = (floatX)((float)dbias[oc] + acc);
}

__global__ void matmul_add_inplace_kernel(floatX* dst, const floatX* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = (floatX)((float)dst[idx] + (float)src[idx]);
}

inline bool matmul_tk_shape_ok(int M, int N, int K) {
    return M % 128 == 0 && K % 64 == 0 && N % 128 == 0;
}

inline bool matmul_forward_gelu_supported(int B, int T, int C, int OC) {
    return matmul_tk_shape_ok(B * T, OC, C);
}

inline void matmul_dispatch_tk_ab(floatX* out, const floatX* a, const floatX* b,
                                  int M, int N, int K, cudaStream_t stream) {
    auto* A = llmk::to_bf16(const_cast<floatX*>(a));
    auto* B = llmk::to_bf16(const_cast<floatX*>(b));
    auto* C = llmk::to_bf16(out);
    if (N % 256 == 0) {
        llmk::gemm::launch<llmk::gemm::matmul_default>(A, B, C, M, N, K, stream);
    } else {
        llmk::gemm::launch<llmk::gemm::matmul_small_n>(A, B, C, M, N, K, stream);
    }
    cudaCheck(cudaGetLastError());
}

inline void matmul_dispatch_tk_atb(floatX* out, const floatX* a, const floatX* b,
                                   int M, int N, int K, cudaStream_t stream) {
    auto* A = llmk::to_bf16(const_cast<floatX*>(a));
    auto* B = llmk::to_bf16(const_cast<floatX*>(b));
    auto* C = llmk::to_bf16(out);
    if (N % 256 == 0) {
        llmk::gemm::launch<llmk::gemm::matmul_default_tn>(A, B, C, M, N, K, stream);
    } else {
        llmk::gemm::launch<llmk::gemm::matmul_small_n_tn>(A, B, C, M, N, K, stream);
    }
    cudaCheck(cudaGetLastError());
}

inline void matmul_forward_gelu(floatX* out, floatX* pre_gelu,
                                const floatX* inp, const floatX* weight,
                                const floatX* bias, int B, int T, int C, int OC,
                                cudaStream_t stream) {
    NVTX_RANGE_FN();
    assert(out != nullptr);
    assert(pre_gelu != nullptr);
    assert(bias != nullptr && "matmul_forward_gelu: fused path expects a bias vector");
    const int M = B * T;
    const int N = OC;
    const int K = C;
    assert(matmul_forward_gelu_supported(B, T, C, OC));

    auto* A = llmk::to_bf16(const_cast<floatX*>(inp));
    auto* B_= llmk::to_bf16(const_cast<floatX*>(weight));
    auto* C_= llmk::to_bf16(out);
    auto* P_= llmk::to_bf16(pre_gelu);
    auto* bias_ = llmk::to_bf16(const_cast<floatX*>(bias));

    if (N % 256 == 0) {
        llmk::gemm::launch<llmk::gemm::matmul_default_nt_bias_gelu>(
            A, B_, C_, M, N, K, stream, P_, bias_);
    } else {
        llmk::gemm::launch<llmk::gemm::matmul_small_n_nt_bias_gelu>(
            A, B_, C_, M, N, K, stream, P_, bias_);
    }
    cudaCheck(cudaGetLastError());
}

inline void matmul_backward_bias(floatX* dbias, const floatX* dout, float* dbias_buffer,
                                 int B, int T, int OC, cudaStream_t stream) {
    const int block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;
    dim3 block_dim = {4, 8, (unsigned)block_size / WARP_SIZE};
    const int OC_per_warp = block_dim.y * x128::size;
    const int grid_size_x = CEIL_DIV(OC, OC_per_warp);
    const int grid_size_y = max(1, deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount
                                   / (block_size * grid_size_x));

    if (grid_size_y == 1) {
        matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(
            dbias, dout, B, T, OC, std::bool_constant<false>{});
        cudaCheck(cudaGetLastError());
    } else if (dbias_buffer != nullptr) {
        matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(
            dbias_buffer, dout, B, T, OC, std::bool_constant<true>{});
        cudaCheck(cudaGetLastError());
        reduce_add_sum_kernel<<<CEIL_DIV(OC, 256 * f128::size), 256, 0, stream>>>(
            dbias, dbias_buffer, OC, grid_size_y);
        cudaCheck(cudaGetLastError());
    } else {
        const int block = 256;
        int grid = CEIL_DIV(OC, block);
        matmul_backward_dbias_kernel<<<grid, block, 0, stream>>>(dbias, dout, B * T, OC);
        cudaCheck(cudaGetLastError());
    }
}

inline void matmul_backward(floatX* dinp, floatX* dweight, floatX* dbias,
                            const floatX* dout, const floatX* inp,
                            const floatX* weight, float* dbias_buffer,
                            int B, int T, int C, int OC,
                            cudaStream_t stream,
                            bool dweight_accumulate = true,
                            floatX* dweight_accum_scratch = nullptr,
                            size_t dweight_accum_scratch_elements = 0) {
    NVTX_RANGE_FN();
    const int M = B * T;
    const int block = 256;

    if (dinp != nullptr) {
        if (matmul_tk_shape_ok(M, C, OC)) {
            matmul_dispatch_tk_ab(dinp, dout, weight, M, C, OC, stream);
        } else {
            int grid = CEIL_DIV(M * C, block);
            matmul_backward_dinp_kernel<<<grid, block, 0, stream>>>(dinp, dout, weight, M, C, OC);
            cudaCheck(cudaGetLastError());
        }
    }
    if (dweight != nullptr) {
        const size_t dweight_elements = (size_t)OC * C;
        if (matmul_tk_shape_ok(OC, C, M)) {
            if (!dweight_accumulate) {
                matmul_dispatch_tk_atb(dweight, dout, inp, OC, C, M, stream);
            } else if (dweight_accum_scratch != nullptr &&
                       dweight_accum_scratch_elements >= dweight_elements) {
                matmul_dispatch_tk_atb(dweight_accum_scratch, dout, inp, OC, C, M, stream);
                int grid = CEIL_DIV(dweight_elements, block);
                matmul_add_inplace_kernel<<<grid, block, 0, stream>>>(
                    dweight, dweight_accum_scratch, dweight_elements);
                cudaCheck(cudaGetLastError());
            } else {
                int grid = CEIL_DIV(dweight_elements, block);
                matmul_backward_dweight_kernel<<<grid, block, 0, stream>>>(
                    dweight, dout, inp, M, C, OC);
                cudaCheck(cudaGetLastError());
            }
        } else {
            int grid = CEIL_DIV(dweight_elements, block);
            matmul_backward_dweight_kernel<<<grid, block, 0, stream>>>(dweight, dout, inp, M, C, OC);
            cudaCheck(cudaGetLastError());
        }
    }
    if (dbias != nullptr) {
        matmul_backward_bias(dbias, dout, dbias_buffer, B, T, OC, stream);
    }
}
