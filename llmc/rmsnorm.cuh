/*
RMSNorm layer for Llama-3.

Forward and fused-residual forward dispatch to a ThunderKittens fork for
supported hidden widths. Backward is a plain CUDA correctness baseline.
*/
#pragma once

#include <assert.h>

#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "tk/rmsnorm_tk.cuh"

// ----------------------------------------------------------------------------
// CUDA fallback / backward kernels

__global__ void rmsnorm_forward_cuda_kernel(floatX* out, float* rstd,
                                            const floatX* inp, const floatX* weight,
                                            int N, int C, float eps) {
    assert(blockDim.x == WARP_SIZE);
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) { return; }

    inp += row * C;
    out += row * C;

    float sumsq = 0.0f;
    for (int c = threadIdx.x; c < C; c += WARP_SIZE) {
        float x = (float)inp[c];
        sumsq += x * x;
    }
    sumsq = warpReduceSum(sumsq);
    float s = rsqrtf(sumsq / C + eps);
    if (threadIdx.x == 0 && rstd != nullptr) {
        rstd[row] = s;
    }

    for (int c = threadIdx.x; c < C; c += WARP_SIZE) {
        out[c] = (floatX)((float)inp[c] * s * (float)weight[c]);
    }
}

__global__ void fused_residual_rmsnorm_forward_cuda_kernel(floatX* residual, floatX* normed, float* rstd,
                                                           const floatX* inp1, const floatX* inp2,
                                                           const floatX* weight,
                                                           int N, int C, float eps) {
    assert(blockDim.x == WARP_SIZE);
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) { return; }

    residual += row * C;
    normed += row * C;
    inp1 += row * C;
    inp2 += row * C;

    float sumsq = 0.0f;
    for (int c = threadIdx.x; c < C; c += WARP_SIZE) {
        float res = (float)inp1[c] + (float)inp2[c];
        residual[c] = (floatX)res;
        sumsq += res * res;
    }
    sumsq = warpReduceSum(sumsq);
    float s = rsqrtf(sumsq / C + eps);
    if (threadIdx.x == 0 && rstd != nullptr) {
        rstd[row] = s;
    }

    for (int c = threadIdx.x; c < C; c += WARP_SIZE) {
        normed[c] = (floatX)((float)residual[c] * s * (float)weight[c]);
    }
}

__global__ void rmsnorm_backward_dinp_kernel(floatX* dinp,
                                             const floatX* dout, const floatX* inp,
                                             const floatX* weight, const float* rstd,
                                             int N, int C) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= N) { return; }

    const floatX* row_dout = dout + row * C;
    const floatX* row_inp = inp + row * C;
    floatX* row_dinp = dinp + row * C;
    float s = rstd[row];

    extern __shared__ float shared[];
    float dot = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        dot += (float)row_dout[c] * (float)weight[c] * (float)row_inp[c];
    }
    shared[tid] = dot;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    dot = shared[0];
    float coeff = dot * s * s * s / C;

    for (int c = tid; c < C; c += blockDim.x) {
        float d = (float)row_dout[c] * (float)weight[c];
        float x = (float)row_inp[c];
        row_dinp[c] = (floatX)(d * s - x * coeff);
    }
}

__global__ void rmsnorm_backward_dweight_kernel(floatX* dweight,
                                                const floatX* dout, const floatX* inp,
                                                const float* rstd,
                                                int N, int C) {
    int c = blockIdx.x;
    int tid = threadIdx.x;
    if (c >= C) { return; }

    extern __shared__ float shared[];
    float sum = 0.0f;
    for (int row = tid; row < N; row += blockDim.x) {
        int idx = row * C + c;
        sum += (float)dout[idx] * (float)inp[idx] * rstd[row];
    }
    shared[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        dweight[c] = (floatX)((float)dweight[c] + shared[0]);
    }
}

// ----------------------------------------------------------------------------
// launchers

inline void rmsnorm_forward_cuda(floatX* out, float* rstd,
                                 const floatX* inp, const floatX* weight,
                                 int N, int C, float eps, cudaStream_t stream) {
    const int block_x = WARP_SIZE;
    const int block_y = 16;
    dim3 block(block_x, block_y);
    dim3 grid(CEIL_DIV(N, block_y));
    rmsnorm_forward_cuda_kernel<<<grid, block, 0, stream>>>(out, rstd, inp, weight, N, C, eps);
    cudaCheck(cudaGetLastError());
}

void rmsnorm_forward(floatX* out, float* rstd,
                     const floatX* inp, const floatX* weight,
                     int N, int C, float eps, cudaStream_t stream) {
    NVTX_RANGE_FN();
    if (llmk::rmsnorm::supports_width(C)) {
        llmk::rmsnorm::launch_forward(llmk::to_bf16(out), rstd, llmk::to_bf16(inp),
                                      llmk::to_bf16(weight), N, C, eps, stream);
        cudaCheck(cudaGetLastError());
        return;
    }
    rmsnorm_forward_cuda(out, rstd, inp, weight, N, C, eps, stream);
}

void fused_residual_rmsnorm_forward(floatX* residual, floatX* normed, float* rstd,
                                    const floatX* inp1, const floatX* inp2,
                                    const floatX* weight,
                                    int N, int C, float eps, cudaStream_t stream) {
    NVTX_RANGE_FN();
    if (llmk::rmsnorm::supports_width(C)) {
        llmk::rmsnorm::launch_fused_residual_forward(
            llmk::to_bf16(residual), llmk::to_bf16(normed), rstd,
            llmk::to_bf16(inp1), llmk::to_bf16(inp2), llmk::to_bf16(weight),
            N, C, eps, stream);
        cudaCheck(cudaGetLastError());
        return;
    }
    const int block_x = WARP_SIZE;
    const int block_y = 16;
    dim3 block(block_x, block_y);
    dim3 grid(CEIL_DIV(N, block_y));
    fused_residual_rmsnorm_forward_cuda_kernel<<<grid, block, 0, stream>>>(
        residual, normed, rstd, inp1, inp2, weight, N, C, eps);
    cudaCheck(cudaGetLastError());
}

void rmsnorm_backward(floatX* dinp, floatX* dweight,
                      const floatX* dout, const floatX* inp,
                      const floatX* weight, const float* rstd,
                      int N, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    rmsnorm_backward_dinp_kernel<<<N, block_size, block_size * sizeof(float), stream>>>(
        dinp, dout, inp, weight, rstd, N, C);
    cudaCheck(cudaGetLastError());
    rmsnorm_backward_dweight_kernel<<<C, block_size, block_size * sizeof(float), stream>>>(
        dweight, dout, inp, rstd, N, C);
    cudaCheck(cudaGetLastError());
}
