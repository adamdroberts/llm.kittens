/*
rope_ext.cuh — RoPE variants beyond the standard apply already shipped in
rope.cuh: standalone apply, YaRN / NTK-aware / linear-PI scaling, 2D RoPE,
ALiBi bias, XPos / NoPE.

The existing rope.cuh kernel applies (cos, sin) to a Q or K tensor with the
split-half convention used by NeuralFn:
    out[..., :half] = x[:, :half] * cos + x[..., half:] * sin
    out[..., half:] = x[:, :half] * -sin + x[..., half:] * cos

The kernels here generate the cos / sin tables for the scaling variants and
also provide a standalone single-call apply for Q and K together.
*/
#pragma once

#include <assert.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ============================================================================
// Standalone RoPE apply (Q and K together). Caller supplies cos, sin tables of
// shape [seq, head_dim/2]; we broadcast across batch/heads.
//
// q, k:       [B, H, S, D]
// cos, sin:   [S, D/2]
// ============================================================================

__global__ void rope_apply_qk_kernel(floatX* q_out, floatX* k_out,
                                     const floatX* q, const floatX* k,
                                     const floatX* cos, const floatX* sin,
                                     int B, int H_q, int H_k, int S, int D) {
    int half = D / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * D;
    if (idx >= total) return;
    int d   = idx % D;
    int s   = (idx / D) % S;
    int b   = idx / (D * S);
    int d_half = d < half ? d : d - half;
    float c = (float)cos[s * half + d_half];
    float si = (float)sin[s * half + d_half];

    for (int h = 0; h < H_q; ++h) {
        int q_off = ((b * H_q + h) * S + s) * D + d;
        float x1, x2;
        if (d < half) {
            x1 = (float)q[q_off];
            x2 = (float)q[q_off + half];
            q_out[q_off] = (floatX)(x1 * c + x2 * si);
        } else {
            x1 = (float)q[q_off - half];
            x2 = (float)q[q_off];
            q_out[q_off] = (floatX)(x1 * (-si) + x2 * c);
        }
    }
    for (int h = 0; h < H_k; ++h) {
        int k_off = ((b * H_k + h) * S + s) * D + d;
        float x1, x2;
        if (d < half) {
            x1 = (float)k[k_off];
            x2 = (float)k[k_off + half];
            k_out[k_off] = (floatX)(x1 * c + x2 * si);
        } else {
            x1 = (float)k[k_off - half];
            x2 = (float)k[k_off];
            k_out[k_off] = (floatX)(x1 * (-si) + x2 * c);
        }
    }
}
void rope_apply_qk(floatX* q_out, floatX* k_out,
                   const floatX* q, const floatX* k, const floatX* cos, const floatX* sin,
                   int B, int H_q, int H_k, int S, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int total = B * S * D;
    const int block_size = 128;
    rope_apply_qk_kernel<<<CEIL_DIV(total, block_size), block_size, 0, stream>>>(
        q_out, k_out, q, k, cos, sin, B, H_q, H_k, S, D);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// RoPE cos/sin generation with scaling variants.
//
//   freq[i] = 1 / base^(2i / D)         (standard)
//
// Variants:
//   * yarn_rope_scaling(D, S, base, scale, original_max_pos, alpha, beta):
//       blend low-/high-frequency scaling so high-freq components are barely
//       scaled and low-freq components get full positional interpolation.
//   * ntk_aware_rope_scaling(D, S, base, scale):  scale base by `scale^(D/(D-2))`
//   * linear_rope_scaling (PI)(D, S, base, scale): divide positions by `scale`
// ============================================================================

__global__ void rope_make_tables_kernel(floatX* cos, floatX* sin, float base, int S, int D, float pos_scale) {
    int s = blockIdx.x;
    int d = threadIdx.x;
    int half = D / 2;
    if (d >= half) return;
    float inv_freq = 1.0f / powf(base, (float)(2 * d) / (float)D);
    float pos = (float)s * pos_scale;
    float angle = pos * inv_freq;
    cos[s * half + d] = (floatX)cosf(angle);
    sin[s * half + d] = (floatX)sinf(angle);
}

void rope_make_tables(floatX* cos, floatX* sin, int S, int D, float base, float pos_scale, cudaStream_t stream) {
    NVTX_RANGE_FN();
    rope_make_tables_kernel<<<S, D / 2, 0, stream>>>(cos, sin, base, S, D, pos_scale);
    cudaCheck(cudaGetLastError());
}

// Linear (Position Interpolation): positions /= scale
void linear_rope_scaling(floatX* cos, floatX* sin, int S, int D, float base, float scale, cudaStream_t stream) {
    rope_make_tables(cos, sin, S, D, base, /*pos_scale=*/1.0f / scale, stream);
}

// NTK-aware: base *= scale^(D/(D-2))
void ntk_aware_rope_scaling(floatX* cos, floatX* sin, int S, int D, float base, float scale, cudaStream_t stream) {
    float new_base = base * powf(scale, (float)D / (float)(D - 2));
    rope_make_tables(cos, sin, S, D, new_base, /*pos_scale=*/1.0f, stream);
}

// YaRN: per-frequency blend between PI-scaled and identity, gated by smooth
// step over (rotations_per_token) bounds. Reference: Peng et al. 2023.
__global__ void yarn_rope_tables_kernel(floatX* cos, floatX* sin,
                                        float base, int S, int D, float scale,
                                        int original_max_pos, float alpha, float beta,
                                        float attn_factor) {
    int s = blockIdx.x;
    int d = threadIdx.x;
    int half = D / 2;
    if (d >= half) return;
    float inv_freq_base = 1.0f / powf(base, (float)(2 * d) / (float)D);
    float wavelength = 2.0f * (float)M_PI / inv_freq_base;
    float r = (float)original_max_pos / wavelength;
    // Smooth blend factor: 0 for r >= beta (high freq, no scaling), 1 for r <= alpha (low freq, full scaling).
    float ramp;
    if (r >= beta) ramp = 0.0f;
    else if (r <= alpha) ramp = 1.0f;
    else ramp = (beta - r) / (beta - alpha);
    float inv_freq_scaled = inv_freq_base / scale;
    float blended = ramp * inv_freq_scaled + (1.0f - ramp) * inv_freq_base;
    float angle = (float)s * blended;
    // YaRN attn-factor scales the cos/sin by mscale = 0.1 ln(s) + 1 typically.
    float mscale = attn_factor;
    cos[s * half + d] = (floatX)(cosf(angle) * mscale);
    sin[s * half + d] = (floatX)(sinf(angle) * mscale);
}

void yarn_rope_scaling(floatX* cos, floatX* sin,
                       int S, int D, float base, float scale,
                       int original_max_pos, float alpha, float beta, float attn_factor,
                       cudaStream_t stream) {
    NVTX_RANGE_FN();
    yarn_rope_tables_kernel<<<S, D / 2, 0, stream>>>(
        cos, sin, base, S, D, scale, original_max_pos, alpha, beta, attn_factor);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// 2D RoPE (for vision / multimodal): we treat the head_dim as two halves —
// one half for the H axis, the other for the W axis.
//
//   cos, sin:  [H_pos, W_pos, head_dim/2]
//   q, k:      [B, heads, H_pos * W_pos, head_dim]
// ============================================================================

__global__ void rope_2d_tables_kernel(floatX* cos, floatX* sin, float base,
                                      int H_pos, int W_pos, int D) {
    int h = blockIdx.y;
    int w = blockIdx.x;
    int d = threadIdx.x;
    int half = D / 2;
    if (d >= half) return;
    int quarter = D / 4;
    // First quarter encodes height, second quarter encodes width
    float inv_freq;
    float pos;
    if (d < quarter) {
        inv_freq = 1.0f / powf(base, (float)(2 * d) / (float)D);
        pos = (float)h;
    } else {
        int d_w = d - quarter;
        inv_freq = 1.0f / powf(base, (float)(2 * d_w) / (float)D);
        pos = (float)w;
    }
    float angle = pos * inv_freq;
    int idx = (h * W_pos + w) * half + d;
    cos[idx] = (floatX)cosf(angle);
    sin[idx] = (floatX)sinf(angle);
}
void rope_2d_tables(floatX* cos, floatX* sin, int H_pos, int W_pos, int D, float base, cudaStream_t stream) {
    NVTX_RANGE_FN();
    dim3 grid(W_pos, H_pos);
    rope_2d_tables_kernel<<<grid, D / 2, 0, stream>>>(cos, sin, base, H_pos, W_pos, D);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// ALiBi: additive linear position bias.
//
// slopes[h]: per-head slope, length [num_heads]
// bias[h, q, k] = slopes[h] * -|q - k|  (causal default; pass mask separately)
//
// Most attention kernels accept additive bias; we materialise it here.
// ============================================================================

__global__ void alibi_bias_kernel(floatX* bias, const float* slopes, int H, int S_q, int S_k, bool causal) {
    int h = blockIdx.z;
    int q = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= S_k) return;
    float slope = slopes[h];
    float v = -fabsf((float)(q - k)) * slope;
    if (causal && k > q) v = -INFINITY;
    bias[((h * S_q + q) * S_k) + k] = (floatX)v;
}
void alibi_bias_make(floatX* bias, const float* slopes, int H, int S_q, int S_k, bool causal, cudaStream_t stream) {
    NVTX_RANGE_FN();
    dim3 grid(CEIL_DIV(S_k, 128), S_q, H);
    alibi_bias_kernel<<<grid, 128, 0, stream>>>(bias, slopes, H, S_q, S_k, causal);
    cudaCheck(cudaGetLastError());
}

// ALiBi standard slopes: 2^{-8/H * (h+1)}  for h in [0, H).
__global__ void alibi_slopes_kernel(float* slopes, int H) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) return;
    float ratio = (float)(h + 1) * (8.0f / (float)H);
    slopes[h] = powf(2.0f, -ratio);
}
void alibi_slopes_make(float* slopes, int H, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 32;
    alibi_slopes_kernel<<<CEIL_DIV(H, block_size), block_size, 0, stream>>>(slopes, H);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// XPos: RoPE with an exponential decay applied to the rotation. Equivalent to
// scaling cos/sin by exp(-|t| * theta_d) where theta_d is an extra learnable
// (or scheduled) per-frequency rate.
//
// We provide a table generator that takes a `theta` vector of length D/2.
// ============================================================================

__global__ void xpos_tables_kernel(floatX* cos, floatX* sin, float base,
                                   const float* theta, int S, int D) {
    int s = blockIdx.x;
    int d = threadIdx.x;
    int half = D / 2;
    if (d >= half) return;
    float inv_freq = 1.0f / powf(base, (float)(2 * d) / (float)D);
    float angle = (float)s * inv_freq;
    float decay = expf(-fabsf((float)s) * theta[d]);
    cos[s * half + d] = (floatX)(cosf(angle) * decay);
    sin[s * half + d] = (floatX)(sinf(angle) * decay);
}
void xpos_tables(floatX* cos, floatX* sin, const float* theta, int S, int D, float base, cudaStream_t stream) {
    NVTX_RANGE_FN();
    xpos_tables_kernel<<<S, D / 2, 0, stream>>>(cos, sin, base, theta, S, D);
    cudaCheck(cudaGetLastError());
}

// NoPE: literally no position encoding — caller just skips the RoPE apply.
// Provided as a no-op for symmetry with the registry.
inline void nope_apply(floatX* /*q*/, floatX* /*k*/, int /*N*/, cudaStream_t /*stream*/) { /* no-op */ }
