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
#include "tk/rope_tk.cuh"

void rope_forward(floatX* out, const floatX* x, const floatX* cos, const floatX* sin,
                  int B, int H, int T, int head_dim, cudaStream_t stream) {
    NVTX_RANGE_FN();
    assert(llmk::rope::supports_head_dim(head_dim));
    assert(T % 16 == 0);
    llmk::rope::launch_forward(llmk::to_bf16(out), llmk::to_bf16(x),
                               llmk::to_bf16(cos), llmk::to_bf16(sin),
                               B, H, T, head_dim, stream);
}

void rope_backward(floatX* dx, const floatX* dout, const floatX* cos, const floatX* sin,
                   int B, int H, int T, int head_dim, cudaStream_t stream) {
    NVTX_RANGE_FN();
    assert(llmk::rope::supports_head_dim(head_dim));
    assert(T % 16 == 0);
    llmk::rope::launch_backward(llmk::to_bf16(dx), llmk::to_bf16(dout),
                                llmk::to_bf16(cos), llmk::to_bf16(sin),
                                B, H, T, head_dim, stream);
}
