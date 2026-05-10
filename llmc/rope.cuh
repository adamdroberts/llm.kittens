/*
RoPE wrapper for Llama-3.

The TK kernel expects x/out in `(B, H, T, HS)` and precomputed bf16 cos/sin in
`(T, HS/2)`. Backward is the inverse rotation, equivalent to applying RoPE with
`sin -> -sin`.
*/
#pragma once

#include <assert.h>

#include "cuda_common.h"
#include "cuda_utils.cuh"
#if defined(KITTENS_SM90)
#define LLMK_USE_TK_ROPE 1
#include "tk/rope_tk.cuh"
#else
#define LLMK_USE_TK_ROPE 0
namespace llmk::rope {
inline bool supports_head_dim(int head_dim) { return head_dim == 64 || head_dim == 128; }
} // namespace llmk::rope
#endif

__global__ void rope_cuda_kernel(floatX* out, const floatX* x,
                                 const floatX* cos, const floatX* sin,
                                 int total, int T, int head_dim, bool inverse) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int hs = idx % head_dim;
    int t = (idx / head_dim) % T;
    int half = head_dim / 2;
    int pair = hs % half;
    int row_base = idx - hs;

    float x1 = (float)x[row_base + pair];
    float x2 = (float)x[row_base + pair + half];
    float c = (float)cos[t * half + pair];
    float s = (float)sin[t * half + pair];
    if (inverse) {
        s = -s;
    }
    float rotated = (hs < half) ? (x1 * c - x2 * s) : (x2 * c + x1 * s);
    out[idx] = (floatX)rotated;
}

void rope_forward(floatX* out, const floatX* x, const floatX* cos, const floatX* sin,
                  int B, int H, int T, int head_dim, cudaStream_t stream) {
    NVTX_RANGE_FN();
    assert(llmk::rope::supports_head_dim(head_dim));
#if LLMK_USE_TK_ROPE
    assert(T % 16 == 0);
    llmk::rope::launch_forward(llmk::to_bf16(out), llmk::to_bf16(x),
                               llmk::to_bf16(cos), llmk::to_bf16(sin),
                               B, H, T, head_dim, stream);
#else
    int total = B * H * T * head_dim;
    const int block = 256;
    const int grid = CEIL_DIV(total, block);
    rope_cuda_kernel<<<grid, block, 0, stream>>>(out, x, cos, sin, total, T, head_dim, false);
    cudaCheck(cudaGetLastError());
#endif
}

void rope_backward(floatX* dx, const floatX* dout, const floatX* cos, const floatX* sin,
                   int B, int H, int T, int head_dim, cudaStream_t stream) {
    NVTX_RANGE_FN();
    assert(llmk::rope::supports_head_dim(head_dim));
#if LLMK_USE_TK_ROPE
    assert(T % 16 == 0);
    llmk::rope::launch_backward(llmk::to_bf16(dx), llmk::to_bf16(dout),
                                llmk::to_bf16(cos), llmk::to_bf16(sin),
                                B, H, T, head_dim, stream);
#else
    int total = B * H * T * head_dim;
    const int block = 256;
    const int grid = CEIL_DIV(total, block);
    rope_cuda_kernel<<<grid, block, 0, stream>>>(dx, dout, cos, sin, total, T, head_dim, true);
    cudaCheck(cudaGetLastError());
#endif
}
