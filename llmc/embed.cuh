/*
embed.cuh — standalone embedding kernels (token-only, position-only, generic
small-table lookup, 2D absolute position embedding, byte_patch_embed).

The existing encoder.cuh fuses token+positional addition (Llama-3 doesn't have
abs-pos, so we already use that path differently for Llama). NeuralFn needs
the embed layers exposed separately so callers can:
  * read the tied embedding weight for the lm_head,
  * use absolute position embedding standalone,
  * use the byte patch path for byte-level tokenisers,
  * generic small-table lookups for hash routers etc.

Backward kernels write into the embedding gradient table via atomicAdd
(scatter pattern); see token_embedding_backward.
*/
#pragma once

#include <assert.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ============================================================================
// token_embedding_forward: out[b, s, :] = wte[token_ids[b, s], :]
//
// Also returns a pointer to the wte buffer (for the lm_head tied path);
// callers just keep the wte argument around. No separate "return tied" kernel.
// ============================================================================

__global__ void token_embedding_forward_kernel(floatX* out, const int* tokens,
                                               const floatX* wte, int B, int S, int D) {
    int d   = blockIdx.x * blockDim.x + threadIdx.x;
    int s   = blockIdx.y;
    int b   = blockIdx.z;
    if (d >= D) return;
    int t = tokens[b * S + s];
    out[((b * S) + s) * D + d] = wte[t * D + d];
}

void token_embedding_forward(floatX* out, const int* tokens, const floatX* wte,
                             int B, int S, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    dim3 grid(CEIL_DIV(D, block_size), S, B);
    token_embedding_forward_kernel<<<grid, block_size, 0, stream>>>(out, tokens, wte, B, S, D);
    cudaCheck(cudaGetLastError());
}

// Backward: scatter-add into dwte by atomicAdd. Bf16 atomicAdd is unsupported
// on many archs, so we go through fp32 via a CAS loop on adjacent pairs.
__global__ void token_embedding_backward_kernel(floatX* dwte, const floatX* dout, const int* tokens,
                                                int B, int S, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    int b = blockIdx.z;
    int s = blockIdx.y;
    int t = tokens[b * S + s];
    float v = (float)dout[((b * S) + s) * D + d];

    // bf16 CAS-based atomic add on the dwte[t, d] slot (pair-aligned, so we
    // pack with the adjacent neighbour).
    unsigned int* addr = (unsigned int*)(dwte + t * D + (d & ~1));
    unsigned int old, assumed;
    do {
        old = *addr;
        __nv_bfloat162 packed;
        memcpy(&packed, &old, sizeof(packed));
        float lo = __bfloat162float(packed.x);
        float hi = __bfloat162float(packed.y);
        if ((d & 1) == 0) lo += v; else hi += v;
        packed.x = __float2bfloat16(lo);
        packed.y = __float2bfloat16(hi);
        unsigned int desired;
        memcpy(&desired, &packed, sizeof(desired));
        assumed = old;
        old = atomicCAS(addr, assumed, desired);
    } while (old != assumed);
}

void token_embedding_backward(floatX* dwte, const floatX* dout, const int* tokens,
                              int B, int S, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    dim3 grid(CEIL_DIV(D, block_size), S, B);
    token_embedding_backward_kernel<<<grid, block_size, 0, stream>>>(dwte, dout, tokens, B, S, D);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// absolute_position_embedding_forward: out[b, s, :] = wpe[s, :], broadcast over B.
// ============================================================================

__global__ void abs_pos_embedding_forward_kernel(floatX* out, const floatX* wpe, int B, int S, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    int b = blockIdx.z;
    if (d >= D) return;
    out[((b * S) + s) * D + d] = wpe[s * D + d];
}
void abs_pos_embedding_forward(floatX* out, const floatX* wpe, int B, int S, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    dim3 grid(CEIL_DIV(D, block_size), S, B);
    abs_pos_embedding_forward_kernel<<<grid, block_size, 0, stream>>>(out, wpe, B, S, D);
    cudaCheck(cudaGetLastError());
}

// Backward: dwpe[s, d] += sum_b dout[b, s, d]
__global__ void abs_pos_embedding_backward_kernel(floatX* dwpe, const floatX* dout, int B, int S, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    if (d >= D) return;
    float acc = 0.0f;
    for (int b = 0; b < B; ++b) {
        acc += (float)dout[((b * S) + s) * D + d];
    }
    dwpe[s * D + d] = (floatX)((float)dwpe[s * D + d] + acc);
}
void abs_pos_embedding_backward(floatX* dwpe, const floatX* dout, int B, int S, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    dim3 grid(CEIL_DIV(D, block_size), S);
    abs_pos_embedding_backward_kernel<<<grid, block_size, 0, stream>>>(dwpe, dout, B, S, D);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Generic small-table lookup: used for hash_embed, bucket_embed, etc.
//   table:    [N_table, D]
//   indices:  [rows]
//   out:      [rows, D]
// ============================================================================

__global__ void small_embed_lookup_kernel(floatX* out, const floatX* table, const int* indices,
                                          int rows, int D, int N_table) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (d >= D || row >= rows) return;
    int t = indices[row];
    if (t < 0 || t >= N_table) t = 0;
    out[row * D + d] = table[t * D + d];
}
void small_embed_lookup(floatX* out, const floatX* table, const int* indices,
                        int rows, int D, int N_table, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    dim3 grid(CEIL_DIV(D, block_size), rows);
    small_embed_lookup_kernel<<<grid, block_size, 0, stream>>>(out, table, indices, rows, D, N_table);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// 2D absolute position embedding (vision).
//   wpe_2d: [H_pos, W_pos, D]
//   out:    [B, H_pos*W_pos, D]   broadcast across B.
// ============================================================================

__global__ void abs_pos_embedding_2d_kernel(floatX* out, const floatX* wpe,
                                            int B, int H_pos, int W_pos, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int w = blockIdx.y % W_pos;
    int h = blockIdx.y / W_pos;
    int b = blockIdx.z;
    if (d >= D) return;
    int seq = h * W_pos + w;
    out[((b * H_pos * W_pos) + seq) * D + d] = wpe[(h * W_pos + w) * D + d];
}
void abs_pos_embedding_2d_forward(floatX* out, const floatX* wpe,
                                  int B, int H_pos, int W_pos, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    dim3 grid(CEIL_DIV(D, block_size), H_pos * W_pos, B);
    abs_pos_embedding_2d_kernel<<<grid, block_size, 0, stream>>>(out, wpe, B, H_pos, W_pos, D);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// patch_embed_2d: Conv2d (stride=patch_size, kernel=patch_size) + flatten +
// optional linear projection. We compose existing kernels: caller runs
// conv2d_general_forward from conv.cuh on the image, then reshapes / projects.
// This file provides the flatten step (HxW -> seq).
//
//   x:   [B, C_out, H_p, W_p]  (output of patch conv)
//   out: [B, H_p*W_p, C_out]
// ============================================================================

__global__ void patch_flatten_kernel(floatX* out, const floatX* x, int B, int C, int H_p, int W_p) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int sp = blockIdx.y;
    int b  = blockIdx.z;
    if (c >= C) return;
    int h = sp / W_p;
    int w = sp % W_p;
    int src = ((b * C + c) * H_p + h) * W_p + w;
    int dst = (b * H_p * W_p + sp) * C + c;
    out[dst] = x[src];
}

void patch_flatten(floatX* out, const floatX* x, int B, int C, int H_p, int W_p, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    dim3 grid(CEIL_DIV(C, block_size), H_p * W_p, B);
    patch_flatten_kernel<<<grid, block_size, 0, stream>>>(out, x, B, C, H_p, W_p);
    cudaCheck(cudaGetLastError());
}
