/*
kv_cache.cuh — KV-cache append, paged write, int8 quant/dequant, PCA project.

Layout notes:
  * KV tensors are [B, H_kv, S, D] (heads first after batch).
  * Append form: cache holds [B, H_kv, S_cap, D]; current K,V are [B, H_kv, S_new, D];
    `cache_len` is per-batch (length [B]) so each row can append independently.
  * The paged form takes a block_table [B, num_blocks] mapping logical block
    index to physical KV-page id; each page holds `page_size` tokens.
*/
#pragma once

#include <assert.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ============================================================================
// kv_cache_read: concatenate cache and current along sequence axis.
//
//   cache_k,v:   [B, Hk, S_cap, D]
//   current_k,v: [B, Hk, S_new, D]
//   out_k,v:     [B, Hk, S_cap + S_new, D]
//
// This is a pure data-movement kernel; appropriate when KV cache is grown
// contiguously and the consumer wants a single contiguous K and V tensor.
// ============================================================================

__global__ void kv_cache_concat_kernel(floatX* out_k, floatX* out_v,
                                       const floatX* cache_k, const floatX* cache_v,
                                       const floatX* current_k, const floatX* current_v,
                                       int B, int Hk, int S_cap, int S_new, int D) {
    int total = B * Hk * (S_cap + S_new) * D;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d = idx % D;
    int s = (idx / D) % (S_cap + S_new);
    int h = (idx / (D * (S_cap + S_new))) % Hk;
    int b = idx / (D * (S_cap + S_new) * Hk);
    if (s < S_cap) {
        int src = ((b * Hk + h) * S_cap + s) * D + d;
        out_k[idx] = cache_k[src];
        out_v[idx] = cache_v[src];
    } else {
        int src = ((b * Hk + h) * S_new + (s - S_cap)) * D + d;
        out_k[idx] = current_k[src];
        out_v[idx] = current_v[src];
    }
}
void kv_cache_concat(floatX* out_k, floatX* out_v,
                     const floatX* cache_k, const floatX* cache_v,
                     const floatX* current_k, const floatX* current_v,
                     int B, int Hk, int S_cap, int S_new, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int total = B * Hk * (S_cap + S_new) * D;
    const int block_size = 256;
    kv_cache_concat_kernel<<<CEIL_DIV(total, block_size), block_size, 0, stream>>>(
        out_k, out_v, cache_k, cache_v, current_k, current_v, B, Hk, S_cap, S_new, D);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// kv_cache_append (in-place): copy current_k,v into cache_k,v at per-batch
// offsets `cache_len` (length B), then update cache_len += S_new.
//
// cache_k,v:   [B, Hk, S_cap, D]
// current_k,v: [B, Hk, S_new, D]
// cache_len:   [B] int32 (in/out)
// ============================================================================

__global__ void kv_cache_append_kernel(floatX* cache_k, floatX* cache_v,
                                       const floatX* current_k, const floatX* current_v,
                                       int* cache_len, int Hk, int S_cap, int S_new, int D) {
    int b = blockIdx.x;
    int offset = cache_len[b];
    if (offset + S_new > S_cap) return;
    for (int h = 0; h < Hk; ++h) {
        for (int s = 0; s < S_new; ++s) {
            for (int d = threadIdx.x; d < D; d += blockDim.x) {
                int src = ((b * Hk + h) * S_new + s) * D + d;
                int dst = ((b * Hk + h) * S_cap + offset + s) * D + d;
                cache_k[dst] = current_k[src];
                cache_v[dst] = current_v[src];
            }
        }
    }
    if (threadIdx.x == 0) cache_len[b] = offset + S_new;
}
void kv_cache_append(floatX* cache_k, floatX* cache_v,
                     const floatX* current_k, const floatX* current_v,
                     int* cache_len, int B, int Hk, int S_cap, int S_new, int D,
                     cudaStream_t stream) {
    NVTX_RANGE_FN();
    kv_cache_append_kernel<<<B, 128, 0, stream>>>(
        cache_k, cache_v, current_k, current_v, cache_len, Hk, S_cap, S_new, D);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Paged KV write (vLLM-style):
//   pages_k,v:   [num_pages, Hk, page_size, D]
//   block_table: [B, max_blocks_per_seq] int32 (page ids; -1 = unused)
//   slot_index:  [B] int32 — next free slot within the active page for each row.
//
// We append `current_k,v` of shape [B, Hk, S_new, D] into the appropriate
// pages, allocating new ones as we cross page boundaries by using the next
// entry in block_table (already pre-allocated by host).
// ============================================================================

__global__ void paged_kv_append_kernel(floatX* pages_k, floatX* pages_v,
                                       const floatX* current_k, const floatX* current_v,
                                       const int* block_table, int* slot_index,
                                       int max_blocks, int page_size, int Hk, int D, int S_new) {
    int b = blockIdx.x;
    int slot = slot_index[b];
    for (int s = 0; s < S_new; ++s) {
        int block_idx = (slot + s) / page_size;
        int within    = (slot + s) % page_size;
        int page_id   = block_table[b * max_blocks + block_idx];
        if (page_id < 0) return; // out-of-capacity; host should pre-allocate
        for (int h = 0; h < Hk; ++h) {
            for (int d = threadIdx.x; d < D; d += blockDim.x) {
                int src = ((b * Hk + h) * S_new + s) * D + d;
                int dst = ((page_id * Hk + h) * page_size + within) * D + d;
                pages_k[dst] = current_k[src];
                pages_v[dst] = current_v[src];
            }
        }
    }
    if (threadIdx.x == 0) slot_index[b] = slot + S_new;
}

void paged_kv_append(floatX* pages_k, floatX* pages_v,
                     const floatX* current_k, const floatX* current_v,
                     const int* block_table, int* slot_index,
                     int B, int max_blocks, int page_size, int Hk, int D, int S_new,
                     cudaStream_t stream) {
    NVTX_RANGE_FN();
    paged_kv_append_kernel<<<B, 128, 0, stream>>>(
        pages_k, pages_v, current_k, current_v, block_table, slot_index,
        max_blocks, page_size, Hk, D, S_new);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// kv_pca_encode / decode: small linear projections on K and V across head_dim.
// These are just GEMMs but we ship a thin batched-tiny GEMM here for the case
// where compressed_dim is tiny (head_dim-sized loads).
//
//   k: [..., head_dim]    -> k_c: [..., compressed_dim]
//   weight (k_proj/v_proj): [compressed_dim, head_dim]   (stored as (Cc, D))
//
// We implement it as a per-row mat-vec; for production this should call the
// general GEMM path from matmul.cuh.
// ============================================================================

__global__ void kv_pca_matvec_kernel(floatX* out, const floatX* x, const floatX* w, int D, int Cc, int N) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;
    int c = threadIdx.x;
    if (c >= Cc) return;
    const floatX* x_row = x + row * D;
    const floatX* w_row = w + c * D;
    float acc = 0.0f;
    for (int d = 0; d < D; ++d) acc += (float)x_row[d] * (float)w_row[d];
    out[row * Cc + c] = (floatX)acc;
}
void kv_pca_encode(floatX* k_c, floatX* v_c, const floatX* k, const floatX* v,
                   const floatX* k_proj, const floatX* v_proj,
                   int N, int D, int Cc, cudaStream_t stream) {
    NVTX_RANGE_FN();
    // N rows × Cc outputs each
    dim3 block(Cc, 8);
    int blocks = CEIL_DIV(N, 8);
    kv_pca_matvec_kernel<<<blocks, block, 0, stream>>>(k_c, k, k_proj, D, Cc, N);
    kv_pca_matvec_kernel<<<blocks, block, 0, stream>>>(v_c, v, v_proj, D, Cc, N);
    cudaCheck(cudaGetLastError());
}

__global__ void kv_pca_decode_matvec_kernel(floatX* out, const floatX* x, const floatX* w, int Cc, int D, int N) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;
    int d = threadIdx.x;
    if (d >= D) return;
    const floatX* x_row = x + row * Cc;
    // w stored as (D, Cc): w[d, c]
    const floatX* w_row = w + d * Cc;
    float acc = 0.0f;
    for (int c = 0; c < Cc; ++c) acc += (float)x_row[c] * (float)w_row[c];
    out[row * D + d] = (floatX)acc;
}
void kv_pca_decode(floatX* k, floatX* v, const floatX* k_c, const floatX* v_c,
                   const floatX* k_unproj, const floatX* v_unproj,
                   int N, int Cc, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    dim3 block(D, 4);
    int blocks = CEIL_DIV(N, 4);
    kv_pca_decode_matvec_kernel<<<blocks, block, 0, stream>>>(k, k_c, k_unproj, Cc, D, N);
    kv_pca_decode_matvec_kernel<<<blocks, block, 0, stream>>>(v, v_c, v_unproj, Cc, D, N);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// kv_quant_pack: per-token int8 quantization of (k || v) with scale.
//
//   k,v: [..., D]
//   packed:  [..., 2D + 1]  — first 2D entries are int8 stored as floatX, last is fp scale.
//            (We store as floatX for kernel-call simplicity; production layout
//             uses int8 buffers separately.)
//
//   amax = max(|kv|) along last dim;  scale = amax/127;  q = round(kv/scale).
// ============================================================================

__global__ void kv_quant_pack_kernel(floatX* packed, const floatX* k, const floatX* v, int D, int rows) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const floatX* row_k = k + row * D;
    const floatX* row_v = v + row * D;
    floatX*       row_q = packed + row * (2 * D + 1);

    // 1) absmax over (k||v)
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < 2 * D; i += blockDim.x) {
        float v_in = (i < D) ? fabsf((float)row_k[i]) : fabsf((float)row_v[i - D]);
        if (v_in > local_max) local_max = v_in;
    }
    float row_amax = blockReduce<warpReduceMax>(local_max);
    float scale = fmaxf(row_amax / 127.0f, 1e-7f);

    // 2) write quantized values
    for (int i = threadIdx.x; i < 2 * D; i += blockDim.x) {
        float val = (i < D) ? (float)row_k[i] : (float)row_v[i - D];
        float q = roundf(val / scale);
        q = fmaxf(-128.0f, fminf(127.0f, q));
        row_q[i] = (floatX)q;
    }
    if (threadIdx.x == 0) row_q[2 * D] = (floatX)scale;
}
void kv_quant_pack(floatX* packed, const floatX* k, const floatX* v, int rows, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    kv_quant_pack_kernel<<<rows, 128, 0, stream>>>(packed, k, v, D, rows);
    cudaCheck(cudaGetLastError());
}

// kv_quant_unpack: inverse.
__global__ void kv_quant_unpack_kernel(floatX* k, floatX* v, const floatX* packed, int D) {
    int row = blockIdx.x;
    const floatX* row_q = packed + row * (2 * D + 1);
    floatX* row_k = k + row * D;
    floatX* row_v = v + row * D;
    float scale = (float)row_q[2 * D];
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        row_k[i] = (floatX)((float)row_q[i]       * scale);
        row_v[i] = (floatX)((float)row_q[D + i]   * scale);
    }
}
void kv_quant_unpack(floatX* k, floatX* v, const floatX* packed, int rows, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    kv_quant_unpack_kernel<<<rows, 128, 0, stream>>>(k, v, packed, D);
    cudaCheck(cudaGetLastError());
}
