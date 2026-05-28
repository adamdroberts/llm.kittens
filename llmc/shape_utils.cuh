/*
shape_utils.cuh — layout transforms used by attention plumbing.

PyTorch view ops (`reshape`, `transpose(1, 2)`, `repeat_interleave`) are
free at the Python tensor level but require explicit kernels when calling
from CUDA-native code paths. These are simple permute/copy kernels.
*/
#pragma once

#include <assert.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ============================================================================
// reshape_heads:  [B, S, H*D]  ->  [B, H, S, D]    (transpose seq <-> heads)
// ============================================================================

__global__ void reshape_heads_kernel(floatX* out, const floatX* x, int B, int S, int H, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    int bh = blockIdx.z;
    int h  = bh % H;
    int b  = bh / H;
    if (d >= D) return;
    int src = ((b * S) + s) * (H * D) + h * D + d;
    int dst = (((b * H) + h) * S + s) * D + d;
    out[dst] = x[src];
}
void reshape_heads_forward(floatX* out, const floatX* x, int B, int S, int H, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 64;
    dim3 grid(CEIL_DIV(D, block_size), S, B * H);
    reshape_heads_kernel<<<grid, block_size, 0, stream>>>(out, x, B, S, H, D);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// merge_heads:  [B, H, S, D]  ->  [B, S, H*D]
// ============================================================================

__global__ void merge_heads_kernel(floatX* out, const floatX* x, int B, int H, int S, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    int bs = blockIdx.z;
    int s  = bs % S;
    int b  = bs / S;
    if (d >= D) return;
    int src = (((b * H) + h) * S + s) * D + d;
    int dst = ((b * S) + s) * (H * D) + h * D + d;
    out[dst] = x[src];
}
void merge_heads_forward(floatX* out, const floatX* x, int B, int H, int S, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 64;
    dim3 grid(CEIL_DIV(D, block_size), H, B * S);
    merge_heads_kernel<<<grid, block_size, 0, stream>>>(out, x, B, H, S, D);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// repeat_kv:  [B, H_kv, S, D]  ->  [B, H_q, S, D]
// where H_q = repeats * H_kv (GQA expansion).
// ============================================================================

__global__ void repeat_kv_kernel(floatX* out, const floatX* x, int B, int H_kv, int repeats, int S, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    int bh = blockIdx.z;
    int H_q = H_kv * repeats;
    int h_q = bh % H_q;
    int b   = bh / H_q;
    if (d >= D) return;
    int h_kv = h_q / repeats;
    int src = (((b * H_kv) + h_kv) * S + s) * D + d;
    int dst = (((b * H_q) + h_q) * S + s) * D + d;
    out[dst] = x[src];
}
void repeat_kv_forward(floatX* out, const floatX* x, int B, int H_kv, int repeats,
                       int S, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 64;
    int H_q = H_kv * repeats;
    dim3 grid(CEIL_DIV(D, block_size), S, B * H_q);
    repeat_kv_kernel<<<grid, block_size, 0, stream>>>(out, x, B, H_kv, repeats, S, D);
    cudaCheck(cudaGetLastError());
}
