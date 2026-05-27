/*
quant_sm120.cuh — ThunderKittens quantization primitives for SM120.

  - nf4_dequantize          (packed uint8 → bf16 with per-group absmax)
  - int8_act_quantize       (per-row absmax → int8 + fp32 row scale)
  - int8_act_dequantize     (int8 + scale → bf16)
  - ternary_quantize        (bf16 W → int8 ∈ {-1,0,1} with per-tensor scale)
  - kv_pca_encode/decode    (small matmul on K, V along head_dim)
  - kv_quant_pack/unpack    (int8 per-token KV quant)

These are simple bulk pointwise kernels — TK 2.0 is overkill for most of
them, but for the dequant + GEMM fusion the TK kernel keeps the data hot in
registers between dequant and downstream attention/GEMM.
*/
#pragma once

#include "tk_common.cuh"

namespace llmk::quant_sm120 {

using namespace ::kittens;

__device__ __constant__ float kNF4_codebook_sm120[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

// nf4_dequantize
__global__ void nf4_dequant_kernel(bf16* out, const uint8_t* qweight, const float* absmax,
                                   int out_dim, int in_dim, int group_size, int packed_cols) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim || col >= in_dim) return;
    int packed_col = col / 2;
    bool high      = (col & 1) != 0;
    uint8_t byte   = qweight[row * packed_cols + packed_col];
    uint8_t code   = high ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
    int group      = col / group_size;
    int num_groups = (in_dim + group_size - 1) / group_size;
    float scale    = absmax[row * num_groups + group];
    float val      = kNF4_codebook_sm120[code] * scale;
    out[row * in_dim + col] = __float2bfloat16(val);
}
inline void launch_nf4_dequant(bf16* out, const uint8_t* qweight, const float* absmax,
                               int out_dim, int in_dim, int group_size, cudaStream_t stream) {
    int packed_cols = (in_dim + 1) / 2;
    dim3 block(128);
    dim3 grid(CEIL_DIV(in_dim, 128), out_dim);
    nf4_dequant_kernel<<<grid, block, 0, stream>>>(out, qweight, absmax, out_dim, in_dim, group_size, packed_cols);
    cudaCheck(cudaGetLastError());
}

// int8 activation quantize.
__global__ void int8_act_quant_kernel(int8_t* q, float* row_scale, const bf16* x, int rows, int in_dim) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const bf16* xr = x + row * in_dim;
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        float v = fabsf(__bfloat162float(xr[i]));
        if (v > local_max) local_max = v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, local_max, off);
        if (other > local_max) local_max = other;
    }
    float row_amax = __shfl_sync(0xFFFFFFFF, local_max, 0);
    float scale = row_amax / 127.0f + 1e-7f;
    if (threadIdx.x == 0) row_scale[row] = scale;
    float inv = 1.0f / scale;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        float v = __bfloat162float(xr[i]) * inv;
        v = fmaxf(-128.0f, fminf(127.0f, roundf(v)));
        q[row * in_dim + i] = (int8_t)v;
    }
}
inline void launch_int8_act_quant(int8_t* q, float* row_scale, const bf16* x,
                                  int rows, int in_dim, cudaStream_t stream) {
    int8_act_quant_kernel<<<rows, ::kittens::WARP_THREADS, 0, stream>>>(q, row_scale, x, rows, in_dim);
    cudaCheck(cudaGetLastError());
}

// Ternary BitNet quantize.
__global__ void ternary_reduce_kernel(float* amean_out, const bf16* w, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        local += fabsf(__bfloat162float(w[i]));
    }
    for (int off = 16; off > 0; off >>= 1) local += __shfl_xor_sync(0xFFFFFFFF, local, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(amean_out, local);
}
__global__ void ternary_finalize_kernel(float* amean, float* scale, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *scale = *amean / (float)N;
}
__global__ void ternary_apply_kernel(int8_t* w_q, const bf16* w, const float* scale, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = *scale;
    float v = __bfloat162float(w[i]) / fmaxf(s, 1e-7f);
    v = fmaxf(-1.0f, fminf(1.0f, roundf(v)));
    w_q[i] = (int8_t)v;
}
inline void launch_ternary_quant(int8_t* w_q, float* scale, const bf16* w, int N,
                                 float* scratch_amean, cudaStream_t stream) {
    cudaCheck(cudaMemsetAsync(scratch_amean, 0, sizeof(float), stream));
    const int bs = 256;
    int gs = std::min(1024, CEIL_DIV(N, bs));
    ternary_reduce_kernel<<<gs, bs, 0, stream>>>(scratch_amean, w, N);
    ternary_finalize_kernel<<<1, 1, 0, stream>>>(scratch_amean, scale, N);
    ternary_apply_kernel<<<CEIL_DIV(N, bs), bs, 0, stream>>>(w_q, w, scale, N);
    cudaCheck(cudaGetLastError());
}

// KV PCA encode/decode (small mat-vec).
__global__ void kv_pca_matvec_kernel(bf16* out, const bf16* x, const bf16* w, int D, int Cc, int N) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;
    int c = threadIdx.x;
    if (c >= Cc) return;
    const bf16* xr = x + row * D;
    const bf16* wr = w + c * D;
    float acc = 0.0f;
    for (int d = 0; d < D; ++d) acc += __bfloat162float(xr[d]) * __bfloat162float(wr[d]);
    out[row * Cc + c] = __float2bfloat16(acc);
}
inline void launch_kv_pca_encode(bf16* k_c, bf16* v_c, const bf16* k, const bf16* v,
                                 const bf16* k_proj, const bf16* v_proj, int N, int D, int Cc,
                                 cudaStream_t stream) {
    dim3 block(Cc, 8);
    int blocks = CEIL_DIV(N, 8);
    kv_pca_matvec_kernel<<<blocks, block, 0, stream>>>(k_c, k, k_proj, D, Cc, N);
    kv_pca_matvec_kernel<<<blocks, block, 0, stream>>>(v_c, v, v_proj, D, Cc, N);
    cudaCheck(cudaGetLastError());
}

__global__ void kv_pca_decode_matvec_kernel(bf16* out, const bf16* x, const bf16* w, int Cc, int D, int N) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;
    int d = threadIdx.x;
    if (d >= D) return;
    const bf16* xr = x + row * Cc;
    const bf16* wr = w + d * Cc;
    float acc = 0.0f;
    for (int c = 0; c < Cc; ++c) acc += __bfloat162float(xr[c]) * __bfloat162float(wr[c]);
    out[row * D + d] = __float2bfloat16(acc);
}
inline void launch_kv_pca_decode(bf16* k, bf16* v, const bf16* k_c, const bf16* v_c,
                                 const bf16* k_unproj, const bf16* v_unproj,
                                 int N, int Cc, int D, cudaStream_t stream) {
    dim3 block(D, 4);
    int blocks = CEIL_DIV(N, 4);
    kv_pca_decode_matvec_kernel<<<blocks, block, 0, stream>>>(k, k_c, k_unproj, Cc, D, N);
    kv_pca_decode_matvec_kernel<<<blocks, block, 0, stream>>>(v, v_c, v_unproj, Cc, D, N);
    cudaCheck(cudaGetLastError());
}

// KV quant pack/unpack (int8 per-token).
__global__ void kv_quant_pack_kernel(bf16* packed, const bf16* k, const bf16* v, int D, int rows) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const bf16* row_k = k + row * D;
    const bf16* row_v = v + row * D;
    bf16*       row_q = packed + row * (2 * D + 1);
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < 2 * D; i += blockDim.x) {
        float v_in = (i < D) ? fabsf(__bfloat162float(row_k[i])) : fabsf(__bfloat162float(row_v[i - D]));
        if (v_in > local_max) local_max = v_in;
    }
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, local_max, off);
        if (other > local_max) local_max = other;
    }
    float amax = __shfl_sync(0xFFFFFFFF, local_max, 0);
    float scale = fmaxf(amax / 127.0f, 1e-7f);
    for (int i = threadIdx.x; i < 2 * D; i += blockDim.x) {
        float val = (i < D) ? __bfloat162float(row_k[i]) : __bfloat162float(row_v[i - D]);
        float q = roundf(val / scale);
        q = fmaxf(-128.0f, fminf(127.0f, q));
        row_q[i] = __float2bfloat16(q);
    }
    if (threadIdx.x == 0) row_q[2 * D] = __float2bfloat16(scale);
}
inline void launch_kv_quant_pack(bf16* packed, const bf16* k, const bf16* v, int rows, int D,
                                 cudaStream_t stream) {
    kv_quant_pack_kernel<<<rows, ::kittens::WARP_THREADS, 0, stream>>>(packed, k, v, D, rows);
    cudaCheck(cudaGetLastError());
}

__global__ void kv_quant_unpack_kernel(bf16* k, bf16* v, const bf16* packed, int D) {
    int row = blockIdx.x;
    const bf16* row_q = packed + row * (2 * D + 1);
    bf16* row_k = k + row * D;
    bf16* row_v = v + row * D;
    float scale = __bfloat162float(row_q[2 * D]);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        row_k[i] = __float2bfloat16(__bfloat162float(row_q[i])       * scale);
        row_v[i] = __float2bfloat16(__bfloat162float(row_q[D + i])   * scale);
    }
}
inline void launch_kv_quant_unpack(bf16* k, bf16* v, const bf16* packed, int rows, int D,
                                   cudaStream_t stream) {
    kv_quant_unpack_kernel<<<rows, ::kittens::WARP_THREADS, 0, stream>>>(k, v, packed, D);
    cudaCheck(cudaGetLastError());
}

}  // namespace llmk::quant_sm120
