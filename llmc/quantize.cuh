/*
quantize.cuh — quantization helpers for adapters and quantized linears.

Covers:
  - NF4 dequantize (packed uint8 → bf16 with per-group absmax)
  - int8 activation per-row absmax quant + dequant
  - ternary BitNet b1.58 quant (W → {-1, 0, 1} with absmean scale + STE)
  - LoRA delta apply (low-rank `x @ A^T @ B^T` accumulated into base output)
  - randmap adapter forward (frozen orthogonal projections + trainable middle)
*/
#pragma once

#include <assert.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ============================================================================
// NF4 codebook
// ============================================================================

__device__ __constant__ float kNF4Codebook[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

// Dequantize NF4 packed weights:
//   qweight: [out_dim, packed_cols]   where packed_cols = (in_dim + 1) / 2
//   absmax:  [out_dim, num_groups]    where num_groups  = ceil(in_dim / group_size)
//   out:     [out_dim, in_dim]        bf16
__global__ void nf4_dequant_kernel(__nv_bfloat16* out, const uint8_t* qweight, const float* absmax,
                                   int out_dim, int in_dim, int group_size, int packed_cols) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim || col >= in_dim) return;
    int packed_col = col / 2;
    bool high      = (col & 1) != 0;
    uint8_t packed = qweight[row * packed_cols + packed_col];
    uint8_t code   = high ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
    int group  = col / group_size;
    float scale = absmax[row * ((in_dim + group_size - 1) / group_size) + group];
    float val   = kNF4Codebook[code] * scale;
    out[row * in_dim + col] = __float2bfloat16(val);
}

void nf4_dequantize(__nv_bfloat16* out, const uint8_t* qweight, const float* absmax,
                    int out_dim, int in_dim, int group_size, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int packed_cols = (in_dim + 1) / 2;
    dim3 block(128);
    dim3 grid(CEIL_DIV(in_dim, 128), out_dim);
    nf4_dequant_kernel<<<grid, block, 0, stream>>>(out, qweight, absmax, out_dim, in_dim, group_size, packed_cols);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// NF4 quantize (host-side helper called at load time; runs on GPU).
//   weight: [out_dim, in_dim] bf16
//   qweight: [out_dim, packed_cols]
//   absmax:  [out_dim, num_groups]
// ============================================================================

__device__ int find_closest_nf4_code(float v) {
    int best = 0;
    float best_dist = fabsf(v - kNF4Codebook[0]);
    for (int i = 1; i < 16; ++i) {
        float d = fabsf(v - kNF4Codebook[i]);
        if (d < best_dist) { best_dist = d; best = i; }
    }
    return best;
}

__global__ void nf4_quant_per_group_kernel(uint8_t* qweight, float* absmax,
                                           const __nv_bfloat16* weight,
                                           int out_dim, int in_dim, int group_size, int packed_cols) {
    int row = blockIdx.y;
    int g   = blockIdx.x;
    if (row >= out_dim) return;
    int start = g * group_size;
    int end   = min(start + group_size, in_dim);
    // absmax for the group
    float am = 0.0f;
    for (int c = start; c < end; ++c) {
        float v = __bfloat162float(weight[row * in_dim + c]);
        if (fabsf(v) > am) am = fabsf(v);
    }
    am = fmaxf(am, 1e-8f);
    int num_groups = (in_dim + group_size - 1) / group_size;
    absmax[row * num_groups + g] = am;
    // quantize each entry in the group
    for (int c = start; c < end; ++c) {
        float v = __bfloat162float(weight[row * in_dim + c]);
        int code = find_closest_nf4_code(v / am);
        int packed_col = c / 2;
        bool high      = (c & 1) != 0;
        uint8_t& byte = qweight[row * packed_cols + packed_col];
        if (high) {
            byte = (byte & 0x0F) | ((code & 0x0F) << 4);
        } else {
            byte = (byte & 0xF0) | (code & 0x0F);
        }
    }
}
void nf4_quantize(uint8_t* qweight, float* absmax, const __nv_bfloat16* weight,
                  int out_dim, int in_dim, int group_size, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int packed_cols = (in_dim + 1) / 2;
    int num_groups  = (in_dim + group_size - 1) / group_size;
    cudaMemsetAsync(qweight, 0, (size_t)out_dim * packed_cols, stream);
    dim3 grid(num_groups, out_dim);
    nf4_quant_per_group_kernel<<<grid, 1, 0, stream>>>(qweight, absmax, weight, out_dim, in_dim, group_size, packed_cols);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// int8 per-row activation quantize + dequantize.
//   x: [rows, in_dim] bf16
//   q: [rows, in_dim] int8
//   row_scale: [rows] fp32
// ============================================================================

__global__ void int8_activation_quant_kernel(int8_t* q, float* row_scale,
                                             const __nv_bfloat16* x, int rows, int in_dim) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16* row_x = x + row * in_dim;

    float local_max = 0.0f;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        float v = fabsf(__bfloat162float(row_x[i]));
        if (v > local_max) local_max = v;
    }
    float row_amax = blockReduce<warpReduceMax>(local_max);
    float scale = row_amax / 127.0f + 1e-7f;
    if (threadIdx.x == 0) row_scale[row] = scale;
    float inv = 1.0f / scale;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        float v = __bfloat162float(row_x[i]) * inv;
        v = fmaxf(-128.0f, fminf(127.0f, roundf(v)));
        q[row * in_dim + i] = (int8_t)v;
    }
}

void int8_activation_quantize(int8_t* q, float* row_scale, const __nv_bfloat16* x,
                              int rows, int in_dim, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int8_activation_quant_kernel<<<rows, 128, 0, stream>>>(q, row_scale, x, rows, in_dim);
    cudaCheck(cudaGetLastError());
}

__global__ void int8_activation_dequant_kernel(__nv_bfloat16* out, const int8_t* q,
                                               const float* row_scale, int rows, int in_dim) {
    int row = blockIdx.x;
    float s = row_scale[row];
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        out[row * in_dim + i] = __float2bfloat16((float)q[row * in_dim + i] * s);
    }
}
void int8_activation_dequantize(__nv_bfloat16* out, const int8_t* q, const float* row_scale,
                                int rows, int in_dim, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int8_activation_dequant_kernel<<<rows, 128, 0, stream>>>(out, q, row_scale, rows, in_dim);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Ternary BitNet b1.58 quantize (per-tensor): scale = mean(|W|);
// W_q = round(W/scale).clamp(-1, 1)
//
// Used for forward; backward uses STE so dW = dW_q.
// ============================================================================

__global__ void ternary_quant_reduce_amean_kernel(float* out, const __nv_bfloat16* w, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        local += fabsf(__bfloat162float(w[i]));
    }
    float sum = blockReduce<warpReduceSum>(local);
    if (threadIdx.x == 0) atomicAdd(out, sum);
}

__global__ void ternary_quant_apply_kernel(int8_t* w_q, const __nv_bfloat16* w, float scale, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = __bfloat162float(w[i]) / fmaxf(scale, 1e-7f);
    v = fmaxf(-1.0f, fminf(1.0f, roundf(v)));
    w_q[i] = (int8_t)v;
}

__global__ void ternary_finalize_scale_kernel(float* amean, float* scale, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *scale = *amean / (float)N;
}

// Apply kernel that reads the scale from device memory (so we can chain the
// finalize step on the same stream without host syncs).
__global__ void ternary_quant_apply_devscale_kernel(int8_t* w_q, const __nv_bfloat16* w,
                                                    const float* scale, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = *scale;
    float v = __bfloat162float(w[i]) / fmaxf(s, 1e-7f);
    v = fmaxf(-1.0f, fminf(1.0f, roundf(v)));
    w_q[i] = (int8_t)v;
}

void ternary_quantize(int8_t* w_q, float* scale, const __nv_bfloat16* w, int N,
                      float* scratch_amean, cudaStream_t stream) {
    NVTX_RANGE_FN();
    cudaMemsetAsync(scratch_amean, 0, sizeof(float), stream);
    const int block_size = 256;
    int grid_size = std::min(1024, CEIL_DIV(N, block_size));
    ternary_quant_reduce_amean_kernel<<<grid_size, block_size, 0, stream>>>(scratch_amean, w, N);
    cudaCheck(cudaGetLastError());
    ternary_finalize_scale_kernel<<<1, 1, 0, stream>>>(scratch_amean, scale, N);
    cudaCheck(cudaGetLastError());
    ternary_quant_apply_devscale_kernel<<<CEIL_DIV(N, block_size), block_size, 0, stream>>>(w_q, w, scale, N);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// LoRA delta apply: out += scaling * (x @ A^T) @ B^T
//   x: [N, in_dim]
//   A: [rank, in_dim]
//   B: [out_dim, rank]
//
// We compute hidden = x @ A^T  ([N, rank])  via per-row dot-products, then
// out_lora = hidden @ B^T ([N, out_dim]) accumulated into base output with
// scaling.
//
// (For larger sizes this should call the general matmul path; for small rank
// the inline kernel is faster.)
// ============================================================================

__global__ void lora_first_matmul_kernel(float* hidden, const __nv_bfloat16* x, const __nv_bfloat16* A,
                                         int N, int in_dim, int rank) {
    int row = blockIdx.y;
    int r   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || r >= rank) return;
    const __nv_bfloat16* xr = x + row * in_dim;
    const __nv_bfloat16* Ar = A + r * in_dim;
    float acc = 0.0f;
    for (int k = 0; k < in_dim; ++k) acc += __bfloat162float(xr[k]) * __bfloat162float(Ar[k]);
    hidden[row * rank + r] = acc;
}

__global__ void lora_second_matmul_add_kernel(__nv_bfloat16* out, const float* hidden,
                                              const __nv_bfloat16* B, float scaling,
                                              int N, int rank, int out_dim) {
    int row = blockIdx.y;
    int o   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || o >= out_dim) return;
    const __nv_bfloat16* Bo = B + o * rank;
    float acc = 0.0f;
    for (int r = 0; r < rank; ++r) acc += hidden[row * rank + r] * __bfloat162float(Bo[r]);
    float prev = __bfloat162float(out[row * out_dim + o]);
    out[row * out_dim + o] = __float2bfloat16(prev + scaling * acc);
}

void lora_apply(__nv_bfloat16* out, const __nv_bfloat16* x,
                const __nv_bfloat16* A, const __nv_bfloat16* B, float scaling,
                int N, int in_dim, int rank, int out_dim,
                float* scratch_hidden, cudaStream_t stream) {
    NVTX_RANGE_FN();
    dim3 block1(64);
    dim3 grid1(CEIL_DIV(rank, 64), N);
    lora_first_matmul_kernel<<<grid1, block1, 0, stream>>>(scratch_hidden, x, A, N, in_dim, rank);
    dim3 block2(64);
    dim3 grid2(CEIL_DIV(out_dim, 64), N);
    lora_second_matmul_add_kernel<<<grid2, block2, 0, stream>>>(out, scratch_hidden, B, scaling, N, rank, out_dim);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// RandMap adapter: x + scale * up(middle(down(x))) where down, up are frozen
// orthogonal random projections and middle is a trainable linear. This is
// just a sequence of three matmuls + add; provided here as a thin wrapper.
//
// Implementation note: at this size the project should call matmul.cuh; we
// don't add a bespoke kernel. The Stage glue lives in NeuralFn.
// ============================================================================
