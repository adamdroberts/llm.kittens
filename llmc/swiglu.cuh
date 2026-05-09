/*
SwiGLU activation used by Llama-3:

    out = silu(gate) * up

This is memory-bandwidth-bound elementwise work, so the v1 implementation keeps
it as plain CUDA rather than adding a ThunderKittens wrapper.
*/
#include <assert.h>

#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

__device__ inline float swiglu_silu(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return x * sig;
}

__device__ inline float swiglu_dsilu(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return sig * (1.0f + x * (1.0f - sig));
}

__global__ void swiglu_forward_kernel(floatX* out, const floatX* gate, const floatX* up) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

    x128 packed_out;
    x128 packed_gate = load128cs(gate + idx);
    x128 packed_up = load128cs(up + idx);
    for (int k = 0; k < x128::size; ++k) {
        float g = (float)packed_gate[k];
        float u = (float)packed_up[k];
        packed_out[k] = (floatX)(swiglu_silu(g) * u);
    }
    store128(out + idx, packed_out);
}

__global__ void swiglu_backward_kernel(
    floatX* dgate, floatX* dup,
    const floatX* dout, const floatX* gate, const floatX* up
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

    x128 packed_dgate;
    x128 packed_dup;
    x128 packed_dout = load128cs(dout + idx);
    x128 packed_gate = load128cs(gate + idx);
    x128 packed_up = load128cs(up + idx);
    for (int k = 0; k < x128::size; ++k) {
        float d = (float)packed_dout[k];
        float g = (float)packed_gate[k];
        float u = (float)packed_up[k];
        packed_dgate[k] = (floatX)(d * u * swiglu_dsilu(g));
        packed_dup[k] = (floatX)(d * swiglu_silu(g));
    }
    store128(dgate + idx, packed_dgate);
    store128(dup + idx, packed_dup);
}

// ----------------------------------------------------------------------------
// kernel launchers

void swiglu_forward(floatX* out, const floatX* gate, const floatX* up, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    swiglu_forward_kernel<<<grid_size, block_size, 0, stream>>>(out, gate, up);
    cudaCheck(cudaGetLastError());
}

void swiglu_backward(
    floatX* dgate, floatX* dup,
    const floatX* dout, const floatX* gate, const floatX* up,
    int N, cudaStream_t stream
) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    swiglu_backward_kernel<<<grid_size, block_size, 0, stream>>>(dgate, dup, dout, gate, up);
    cudaCheck(cudaGetLastError());
}
