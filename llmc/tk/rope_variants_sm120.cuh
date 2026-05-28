/*
rope_variants_sm120.cuh — ThunderKittens RoPE-scaling variants for SM120.

Generates cos/sin tables for YaRN, NTK-aware, linear (PI), XPos, and ALiBi
bias slopes. Application of cos/sin to Q/K still uses the standalone RoPE
apply kernel (rope_tk.cuh / rope_apply_qk in rope_ext.cuh).
*/
#pragma once

#include "tk_common.cuh"
#include <cmath>

namespace llmk::rope_variants {

using namespace ::kittens;

// Linear (Position Interpolation): pos /= scale.
__global__ void linear_rope_tables_kernel(bf16* cos, bf16* sin, float base, int S, int D, float scale) {
    int s = blockIdx.x;
    int d = threadIdx.x;
    int half = D / 2;
    if (d >= half) return;
    float inv_freq = 1.0f / powf(base, (float)(2 * d) / (float)D);
    float pos = (float)s / scale;
    float angle = pos * inv_freq;
    cos[s * half + d] = __float2bfloat16(cosf(angle));
    sin[s * half + d] = __float2bfloat16(sinf(angle));
}
inline void launch_linear(bf16* cos, bf16* sin, int S, int D, float base, float scale, cudaStream_t stream) {
    linear_rope_tables_kernel<<<S, D / 2, 0, stream>>>(cos, sin, base, S, D, scale);
    cudaCheck(cudaGetLastError());
}

// NTK-aware: base *= scale^(D/(D-2)).
__global__ void ntk_rope_tables_kernel(bf16* cos, bf16* sin, float new_base, int S, int D) {
    int s = blockIdx.x;
    int d = threadIdx.x;
    int half = D / 2;
    if (d >= half) return;
    float inv_freq = 1.0f / powf(new_base, (float)(2 * d) / (float)D);
    float angle = (float)s * inv_freq;
    cos[s * half + d] = __float2bfloat16(cosf(angle));
    sin[s * half + d] = __float2bfloat16(sinf(angle));
}
inline void launch_ntk(bf16* cos, bf16* sin, int S, int D, float base, float scale, cudaStream_t stream) {
    float new_base = base * powf(scale, (float)D / (float)(D - 2));
    ntk_rope_tables_kernel<<<S, D / 2, 0, stream>>>(cos, sin, new_base, S, D);
    cudaCheck(cudaGetLastError());
}

// YaRN.
__global__ void yarn_rope_tables_kernel(bf16* cos, bf16* sin, float base, int S, int D,
                                        float scale, int original_max_pos, float alpha, float beta,
                                        float attn_factor) {
    int s = blockIdx.x;
    int d = threadIdx.x;
    int half = D / 2;
    if (d >= half) return;
    float inv_freq_base = 1.0f / powf(base, (float)(2 * d) / (float)D);
    float wavelength = 2.0f * (float)M_PI / inv_freq_base;
    float r = (float)original_max_pos / wavelength;
    float ramp;
    if (r >= beta) ramp = 0.0f;
    else if (r <= alpha) ramp = 1.0f;
    else ramp = (beta - r) / (beta - alpha);
    float inv_freq_scaled = inv_freq_base / scale;
    float blended = ramp * inv_freq_scaled + (1.0f - ramp) * inv_freq_base;
    float angle = (float)s * blended;
    cos[s * half + d] = __float2bfloat16(cosf(angle) * attn_factor);
    sin[s * half + d] = __float2bfloat16(sinf(angle) * attn_factor);
}
inline void launch_yarn(bf16* cos, bf16* sin, int S, int D, float base, float scale,
                        int original_max_pos, float alpha, float beta, float attn_factor,
                        cudaStream_t stream) {
    yarn_rope_tables_kernel<<<S, D / 2, 0, stream>>>(cos, sin, base, S, D, scale,
                                                     original_max_pos, alpha, beta, attn_factor);
    cudaCheck(cudaGetLastError());
}

// XPos: cos/sin scaled by exp(-|s| * theta_d) — per-frequency decay.
__global__ void xpos_tables_kernel(bf16* cos, bf16* sin, float base, const float* theta, int S, int D) {
    int s = blockIdx.x;
    int d = threadIdx.x;
    int half = D / 2;
    if (d >= half) return;
    float inv_freq = 1.0f / powf(base, (float)(2 * d) / (float)D);
    float angle = (float)s * inv_freq;
    float decay = expf(-fabsf((float)s) * theta[d]);
    cos[s * half + d] = __float2bfloat16(cosf(angle) * decay);
    sin[s * half + d] = __float2bfloat16(sinf(angle) * decay);
}
inline void launch_xpos(bf16* cos, bf16* sin, const float* theta, int S, int D, float base, cudaStream_t stream) {
    xpos_tables_kernel<<<S, D / 2, 0, stream>>>(cos, sin, base, theta, S, D);
    cudaCheck(cudaGetLastError());
}

// ALiBi standard slopes: 2^{-8/H * (h+1)}  for h in [0, H).
__global__ void alibi_slopes_kernel(float* slopes, int H) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) return;
    float ratio = (float)(h + 1) * (8.0f / (float)H);
    slopes[h] = powf(2.0f, -ratio);
}
inline void launch_alibi_slopes(float* slopes, int H, cudaStream_t stream) {
    const int block_size = 32;
    alibi_slopes_kernel<<<CEIL_DIV(H, block_size), block_size, 0, stream>>>(slopes, H);
    cudaCheck(cudaGetLastError());
}

// ALiBi additive bias matrix: bias[h, q, k] = -|q - k| * slope[h].
__global__ void alibi_bias_kernel(bf16* bias, const float* slopes, int H, int S_q, int S_k, bool causal) {
    int h = blockIdx.z;
    int q = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= S_k) return;
    float v = -fabsf((float)(q - k)) * slopes[h];
    if (causal && k > q) v = -1e9f;
    bias[((h * S_q + q) * S_k) + k] = __float2bfloat16(v);
}
inline void launch_alibi_bias(bf16* bias, const float* slopes, int H, int S_q, int S_k, bool causal, cudaStream_t stream) {
    const int block_size = 128;
    dim3 grid(CEIL_DIV(S_k, block_size), S_q, H);
    alibi_bias_kernel<<<grid, block_size, 0, stream>>>(bias, slopes, H, S_q, S_k, causal);
    cudaCheck(cudaGetLastError());
}

}  // namespace llmk::rope_variants
