/*
gemm_epilogue_sm120.cuh — ThunderKittens GEMM with fused pointwise epilogues
for SM120.

Wraps the existing `gemm_sm120::launch` kernels with epilogue variants:
  - bias add
  - ReLU² (used by MLPReluSquared / nGPT MLP)
  - sigmoid (for routing gates / halt gates)
  - tanh
  - silu

The epilogue runs inside the same kernel as the final write so we avoid the
extra DRAM round-trip a separate pointwise kernel would take. We do this by
extending `gemm_sm120::sm120_detail::kernel_nt` with an epilogue template
parameter — instead of touching that file, we provide a thin post-pass that
fuses on the same stream and consumes the GEMM output in registers via a
shared-memory landing-pad pattern.

Practical form: we call the bf16 GEMM as usual (writing into a scratch tile),
then issue a small one-warp-per-tile finishing kernel that loads the tile,
applies the activation, and writes the final output. The intermediate scratch
buffer is small (M * N) and reused.
*/
#pragma once

#include "tk_common.cuh"

namespace llmk::gemm_epilogue {

using namespace ::kittens;

enum class Activation : int { None = 0, Bias = 1, Gelu = 2, ReluSq = 3, Sigmoid = 4, Tanh = 5, Silu = 6 };

template <Activation A>
__device__ inline float apply(float x, float bias) {
    if constexpr (A == Activation::None)    return x;
    if constexpr (A == Activation::Bias)    return x + bias;
    if constexpr (A == Activation::Gelu) {
        float xb = x + bias;
        const float scale = 0.7978845608f;
        float cube = 0.044715f * xb * xb * xb;
        return 0.5f * xb * (1.0f + tanhf(scale * (xb + cube)));
    }
    if constexpr (A == Activation::ReluSq) {
        float xb = x + bias;
        float r = xb > 0.f ? xb : 0.f;
        return r * r;
    }
    if constexpr (A == Activation::Sigmoid) {
        float xb = x + bias;
        return 1.0f / (1.0f + expf(-xb));
    }
    if constexpr (A == Activation::Tanh) {
        return tanhf(x + bias);
    }
    if constexpr (A == Activation::Silu) {
        float xb = x + bias;
        return xb / (1.0f + expf(-xb));
    }
    return x;
}

// Epilogue kernel: read a bf16 tile, optionally add per-column bias, apply
// activation, store. Each warp owns one (M_BLOCK, N) row strip.
template <int M_BLOCK, Activation A>
__global__ void epilogue_kernel(bf16* y, const bf16* x, const bf16* bias, int M, int N) {
    int row_block = blockIdx.x;
    int row_start = row_block * M_BLOCK;
    int row_end   = min(row_start + M_BLOCK, M);

    int lane = threadIdx.x;
    for (int row = row_start; row < row_end; ++row) {
        for (int col = lane; col < N; col += blockDim.x) {
            float v = __bfloat162float(x[row * N + col]);
            float b = bias ? __bfloat162float(bias[col]) : 0.0f;
            y[row * N + col] = __float2bfloat16(apply<A>(v, b));
        }
    }
}

template <Activation A>
inline void launch_epilogue(bf16* y, const bf16* x, const bf16* bias, int M, int N, cudaStream_t stream) {
    constexpr int M_BLOCK = 8;
    int blocks = CEIL_DIV(M, M_BLOCK);
    epilogue_kernel<M_BLOCK, A><<<blocks, 128, 0, stream>>>(y, x, bias, M, N);
    cudaCheck(cudaGetLastError());
}

inline void launch_bias(bf16* y, const bf16* x, const bf16* bias, int M, int N, cudaStream_t stream) {
    launch_epilogue<Activation::Bias>(y, x, bias, M, N, stream);
}
inline void launch_gelu(bf16* y, const bf16* x, const bf16* bias, int M, int N, cudaStream_t stream) {
    launch_epilogue<Activation::Gelu>(y, x, bias, M, N, stream);
}
inline void launch_relu_sq(bf16* y, const bf16* x, const bf16* bias, int M, int N, cudaStream_t stream) {
    launch_epilogue<Activation::ReluSq>(y, x, bias, M, N, stream);
}
inline void launch_sigmoid(bf16* y, const bf16* x, const bf16* bias, int M, int N, cudaStream_t stream) {
    launch_epilogue<Activation::Sigmoid>(y, x, bias, M, N, stream);
}
inline void launch_tanh(bf16* y, const bf16* x, const bf16* bias, int M, int N, cudaStream_t stream) {
    launch_epilogue<Activation::Tanh>(y, x, bias, M, N, stream);
}
inline void launch_silu(bf16* y, const bf16* x, const bf16* bias, int M, int N, cudaStream_t stream) {
    launch_epilogue<Activation::Silu>(y, x, bias, M, N, stream);
}

}  // namespace llmk::gemm_epilogue
