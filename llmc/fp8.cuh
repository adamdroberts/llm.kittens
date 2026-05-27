/*
fp8.cuh — FP8 quantization, dequantization, amax tracking, and MX-style
microscaled FP8/FP4 helpers for SM120 (Blackwell).

For the GEMM itself we delegate to cuBLASLt's FP8 path or TK 2.0's tensor-core
fp8 GEMM; this file owns the *scaling* infrastructure that wraps it.

Notation:
  - E4M3 and E5M2 are the two FP8 formats. We use CUDA's __nv_fp8x4_e4m3 /
    __nv_fp8x4_e5m2 vector types where available.
  - amax history is kept on device: a small ring buffer per tensor that the
    delayed-scaling step samples.

This implementation assumes SM ≥ 89 for fp8 intrinsics. On older targets we
fall back to scalar conversion.
*/
#pragma once

#include <assert.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"

#if (__CUDA_ARCH__ >= 890) || !defined(__CUDA_ARCH__)
#include <cuda_fp8.h>
#define LLMK_HAS_FP8 1
#else
#define LLMK_HAS_FP8 0
#endif

// ============================================================================
// FP8 forward range constants
// ============================================================================

constexpr float FP8_E4M3_MAX  = 448.0f;
constexpr float FP8_E5M2_MAX  = 57344.0f;
constexpr float FP4_E2M1_MAX  = 6.0f;

// ============================================================================
// fp8_quantize: bf16/fp32 → fp8 with explicit fp32 scale.
//
// Saves uint8 values containing the FP8 bit-pattern (cast to / from __nv_fp8_e*).
//
// Two scaling strategies:
//   * static_scale:  use the supplied scale directly  (delayed scaling)
//   * dynamic_scale: compute amax on-the-fly per chunk (current-scale)
//
// Both write the updated amax back into `amax_buf` so the host can update its
// scale-history rolling window.
// ============================================================================

__device__ inline uint8_t fp32_to_e4m3(float x) {
#if LLMK_HAS_FP8
    __nv_fp8_e4m3 fp{x};
    uint8_t out;
    memcpy(&out, &fp, sizeof(uint8_t));
    return out;
#else
    float clamped = fmaxf(-FP8_E4M3_MAX, fminf(FP8_E4M3_MAX, x));
    return (uint8_t)((int)(clamped / FP8_E4M3_MAX * 127.0f) & 0xFF);
#endif
}

__device__ inline float e4m3_to_fp32(uint8_t v) {
#if LLMK_HAS_FP8
    __nv_fp8_e4m3 fp;
    memcpy(&fp, &v, sizeof(uint8_t));
    return (float)fp;
#else
    int8_t s = *reinterpret_cast<const int8_t*>(&v);
    return ((float)s / 127.0f) * FP8_E4M3_MAX;
#endif
}

__device__ inline uint8_t fp32_to_e5m2(float x) {
#if LLMK_HAS_FP8
    __nv_fp8_e5m2 fp{x};
    uint8_t out;
    memcpy(&out, &fp, sizeof(uint8_t));
    return out;
#else
    float clamped = fmaxf(-FP8_E5M2_MAX, fminf(FP8_E5M2_MAX, x));
    return (uint8_t)((int)(clamped / FP8_E5M2_MAX * 127.0f) & 0xFF);
#endif
}

__device__ inline float e5m2_to_fp32(uint8_t v) {
#if LLMK_HAS_FP8
    __nv_fp8_e5m2 fp;
    memcpy(&fp, &v, sizeof(uint8_t));
    return (float)fp;
#else
    int8_t s = *reinterpret_cast<const int8_t*>(&v);
    return ((float)s / 127.0f) * FP8_E5M2_MAX;
#endif
}

// Quantize bf16 → e4m3 with fp32 scale (out *= 1/scale). Static (delayed)
// scaling: scale is read from device memory.
__global__ void fp8_quantize_e4m3_kernel(uint8_t* out, const __nv_bfloat16* x,
                                         const float* scale, float* amax_out, size_t N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = *scale;
    float v = __bfloat162float(x[i]) / fmaxf(s, 1e-12f);
    out[i] = fp32_to_e4m3(v);
    if (amax_out) {
        float a = fabsf(__bfloat162float(x[i]));
        atomicMax((unsigned int*)amax_out, __float_as_uint(a));
    }
}

void fp8_quantize_e4m3(uint8_t* out, const __nv_bfloat16* x, const float* scale,
                       float* amax_out /*nullable*/, size_t N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    if (amax_out) cudaMemsetAsync(amax_out, 0, sizeof(float), stream);
    const int block_size = 256;
    fp8_quantize_e4m3_kernel<<<CEIL_DIV(N, block_size), block_size, 0, stream>>>(out, x, scale, amax_out, N);
    cudaCheck(cudaGetLastError());
}

// Dequantize e4m3 → bf16 (out *= scale).
__global__ void fp8_dequantize_e4m3_kernel(__nv_bfloat16* out, const uint8_t* x,
                                           const float* scale, size_t N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = *scale;
    out[i] = __float2bfloat16(e4m3_to_fp32(x[i]) * s);
}
void fp8_dequantize_e4m3(__nv_bfloat16* out, const uint8_t* x, const float* scale,
                         size_t N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    fp8_dequantize_e4m3_kernel<<<CEIL_DIV(N, block_size), block_size, 0, stream>>>(out, x, scale, N);
    cudaCheck(cudaGetLastError());
}

// E5M2 versions (wider dynamic range; used for gradients).
__global__ void fp8_quantize_e5m2_kernel(uint8_t* out, const __nv_bfloat16* x,
                                         const float* scale, float* amax_out, size_t N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = *scale;
    float v = __bfloat162float(x[i]) / fmaxf(s, 1e-12f);
    out[i] = fp32_to_e5m2(v);
    if (amax_out) {
        float a = fabsf(__bfloat162float(x[i]));
        atomicMax((unsigned int*)amax_out, __float_as_uint(a));
    }
}
void fp8_quantize_e5m2(uint8_t* out, const __nv_bfloat16* x, const float* scale,
                       float* amax_out, size_t N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    if (amax_out) cudaMemsetAsync(amax_out, 0, sizeof(float), stream);
    const int block_size = 256;
    fp8_quantize_e5m2_kernel<<<CEIL_DIV(N, block_size), block_size, 0, stream>>>(out, x, scale, amax_out, N);
    cudaCheck(cudaGetLastError());
}

__global__ void fp8_dequantize_e5m2_kernel(__nv_bfloat16* out, const uint8_t* x,
                                           const float* scale, size_t N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = *scale;
    out[i] = __float2bfloat16(e5m2_to_fp32(x[i]) * s);
}
void fp8_dequantize_e5m2(__nv_bfloat16* out, const uint8_t* x, const float* scale,
                         size_t N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    fp8_dequantize_e5m2_kernel<<<CEIL_DIV(N, block_size), block_size, 0, stream>>>(out, x, scale, N);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// amax history tracking (Transformer-Engine style):
//
//   amax_history: ring buffer of length H per tensor
//   scale_inv:    1 / (amax_history.max() / FP8_MAX)
//
// On every step we:
//   1) write the freshly measured amax into amax_history[step % H]
//   2) recompute scale_inv from max-over-history
//
// We provide kernels for both steps; the host orchestrates the step index.
// ============================================================================

__global__ void amax_history_write_kernel(float* amax_history, const float* amax_now,
                                          int step, int history_len) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        amax_history[step % history_len] = *amax_now;
    }
}
void amax_history_write(float* amax_history, const float* amax_now, int step, int history_len, cudaStream_t stream) {
    amax_history_write_kernel<<<1, 1, 0, stream>>>(amax_history, amax_now, step, history_len);
    cudaCheck(cudaGetLastError());
}

__global__ void amax_history_to_scale_kernel(float* scale, float* scale_inv,
                                             const float* amax_history, int history_len,
                                             float fp8_max) {
    float m = 0.0f;
    for (int i = threadIdx.x; i < history_len; i += blockDim.x) {
        m = fmaxf(m, amax_history[i]);
    }
    float row_max = blockReduce<warpReduceMax>(m);
    if (threadIdx.x == 0) {
        float s = fmaxf(row_max, 1e-12f) / fp8_max;
        *scale     = s;
        *scale_inv = 1.0f / s;
    }
}
void amax_history_to_scale(float* scale, float* scale_inv, const float* amax_history,
                           int history_len, float fp8_max, cudaStream_t stream) {
    amax_history_to_scale_kernel<<<1, 32, 0, stream>>>(scale, scale_inv, amax_history, history_len, fp8_max);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// MX (microscaled) FP8/FP4 — Blackwell native.
//
// A block of 32 (FP8) or 16 (FP4) values shares one E8M0 exponent. We emit
// (data_block, exp_byte) tiles. Here we ship the bf16→MXFP8 and the inverse
// for E4M3 blocks.
// ============================================================================

constexpr int MXFP8_BLOCK = 32;
constexpr int MXFP4_BLOCK = 16;

__device__ inline uint8_t pack_e8m0(int exp) {
    // E8M0 stores exponents biased by 127, like fp32 exponent field.
    int biased = exp + 127;
    if (biased < 0) biased = 0;
    if (biased > 255) biased = 255;
    return (uint8_t)biased;
}

__device__ inline int unpack_e8m0(uint8_t e) { return (int)e - 127; }

__global__ void mxfp8_quantize_e4m3_kernel(uint8_t* data, uint8_t* exps,
                                           const __nv_bfloat16* x, size_t num_blocks) {
    size_t blk = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks) return;
    const __nv_bfloat16* xb = x + blk * MXFP8_BLOCK;
    // Find block amax
    float amax = 0.0f;
    for (int i = 0; i < MXFP8_BLOCK; ++i) {
        float v = fabsf(__bfloat162float(xb[i]));
        if (v > amax) amax = v;
    }
    // exponent = floor(log2(amax / FP8_E4M3_MAX))
    int exp = (amax > 0.0f) ? (int)floorf(log2f(amax / FP8_E4M3_MAX)) : -127;
    float scale = ldexpf(1.0f, exp);
    exps[blk] = pack_e8m0(exp);
    uint8_t* db = data + blk * MXFP8_BLOCK;
    for (int i = 0; i < MXFP8_BLOCK; ++i) {
        float v = __bfloat162float(xb[i]) / scale;
        db[i] = fp32_to_e4m3(v);
    }
}

void mxfp8_quantize_e4m3(uint8_t* data, uint8_t* exps, const __nv_bfloat16* x, size_t N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    assert(N % MXFP8_BLOCK == 0);
    size_t num_blocks = N / MXFP8_BLOCK;
    const int block_size = 256;
    mxfp8_quantize_e4m3_kernel<<<CEIL_DIV(num_blocks, block_size), block_size, 0, stream>>>(data, exps, x, num_blocks);
    cudaCheck(cudaGetLastError());
}

__global__ void mxfp8_dequantize_e4m3_kernel(__nv_bfloat16* out, const uint8_t* data,
                                             const uint8_t* exps, size_t num_blocks) {
    size_t blk = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks) return;
    int exp = unpack_e8m0(exps[blk]);
    float scale = ldexpf(1.0f, exp);
    const uint8_t* db = data + blk * MXFP8_BLOCK;
    __nv_bfloat16* ob = out + blk * MXFP8_BLOCK;
    for (int i = 0; i < MXFP8_BLOCK; ++i) {
        ob[i] = __float2bfloat16(e4m3_to_fp32(db[i]) * scale);
    }
}
void mxfp8_dequantize_e4m3(__nv_bfloat16* out, const uint8_t* data, const uint8_t* exps,
                           size_t N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    assert(N % MXFP8_BLOCK == 0);
    size_t num_blocks = N / MXFP8_BLOCK;
    const int block_size = 256;
    mxfp8_dequantize_e4m3_kernel<<<CEIL_DIV(num_blocks, block_size), block_size, 0, stream>>>(out, data, exps, num_blocks);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// MXFP4 quantize/dequantize (16-element blocks, 4-bit data → packed nibbles).
// ============================================================================

constexpr float kFP4Codebook[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

__device__ inline uint8_t fp32_to_fp4(float x) {
    int sign = signbit(x) ? 1 : 0;
    float a = fabsf(x);
    int best = 0;
    float best_d = fabsf(a - kFP4Codebook[0]);
    for (int i = 1; i < 8; ++i) {
        float d = fabsf(a - kFP4Codebook[i]);
        if (d < best_d) { best_d = d; best = i; }
    }
    return (uint8_t)((sign << 3) | best);
}
__device__ inline float fp4_to_fp32(uint8_t v) {
    int sign = (v >> 3) & 1;
    int idx  = v & 0x07;
    float m  = kFP4Codebook[idx];
    return sign ? -m : m;
}

__global__ void mxfp4_quantize_kernel(uint8_t* packed, uint8_t* exps,
                                      const __nv_bfloat16* x, size_t num_blocks) {
    size_t blk = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks) return;
    const __nv_bfloat16* xb = x + blk * MXFP4_BLOCK;
    float amax = 0.0f;
    for (int i = 0; i < MXFP4_BLOCK; ++i) {
        float v = fabsf(__bfloat162float(xb[i]));
        if (v > amax) amax = v;
    }
    int exp = (amax > 0.0f) ? (int)floorf(log2f(amax / FP4_E2M1_MAX)) : -127;
    float scale = ldexpf(1.0f, exp);
    exps[blk] = pack_e8m0(exp);
    uint8_t* pb = packed + blk * (MXFP4_BLOCK / 2);
    for (int i = 0; i < MXFP4_BLOCK; i += 2) {
        uint8_t lo = fp32_to_fp4(__bfloat162float(xb[i])   / scale);
        uint8_t hi = fp32_to_fp4(__bfloat162float(xb[i+1]) / scale);
        pb[i / 2] = (lo & 0x0F) | ((hi & 0x0F) << 4);
    }
}
void mxfp4_quantize(uint8_t* packed, uint8_t* exps, const __nv_bfloat16* x, size_t N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    assert(N % MXFP4_BLOCK == 0);
    size_t num_blocks = N / MXFP4_BLOCK;
    const int block_size = 256;
    mxfp4_quantize_kernel<<<CEIL_DIV(num_blocks, block_size), block_size, 0, stream>>>(packed, exps, x, num_blocks);
    cudaCheck(cudaGetLastError());
}

__global__ void mxfp4_dequantize_kernel(__nv_bfloat16* out, const uint8_t* packed,
                                        const uint8_t* exps, size_t num_blocks) {
    size_t blk = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks) return;
    int exp = unpack_e8m0(exps[blk]);
    float scale = ldexpf(1.0f, exp);
    const uint8_t* pb = packed + blk * (MXFP4_BLOCK / 2);
    __nv_bfloat16* ob = out + blk * MXFP4_BLOCK;
    for (int i = 0; i < MXFP4_BLOCK; i += 2) {
        uint8_t byte = pb[i / 2];
        ob[i]   = __float2bfloat16(fp4_to_fp32(byte & 0x0F) * scale);
        ob[i+1] = __float2bfloat16(fp4_to_fp32((byte >> 4) & 0x0F) * scale);
    }
}
void mxfp4_dequantize(__nv_bfloat16* out, const uint8_t* packed, const uint8_t* exps,
                      size_t N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    assert(N % MXFP4_BLOCK == 0);
    size_t num_blocks = N / MXFP4_BLOCK;
    const int block_size = 256;
    mxfp4_dequantize_kernel<<<CEIL_DIV(num_blocks, block_size), block_size, 0, stream>>>(out, packed, exps, num_blocks);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// FP8 GEMM wrapper (delegates to cuBLASLt FP8 matmul).
//
// Declared here, defined in matmul_fp8.cu next to the existing matmul cuBLASLt
// setup. Header only declares the API.
// ============================================================================

void fp8_gemm_e4m3(__nv_bfloat16* out, const uint8_t* A, const uint8_t* B,
                   const float* scale_A, const float* scale_B,
                   int M, int N, int K, cudaStream_t stream);
