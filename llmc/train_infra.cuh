/*
train_infra.cuh — training-loop infrastructure kernels.

  - gradient_accumulate (fused add-into-grad-buffer with scale)
  - loss_scale_dynamic (dynamic loss scale + overflow detect)
  - gradient_checkpoint helpers (no-op stubs; orchestration is host-side)
  - selective_recompute hooks (no-op stubs)

Also: data-path helpers
  - sequence_packing  (build cu_seqlens from per-row lengths)
  - document_causal_mask (cu_seqlens → block-causal mask)
*/
#pragma once

#include <assert.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ============================================================================
// gradient_accumulate: `dst += scale * src` (fp32 add, in-place).
//
//   dst, src: [N] fp32 buffers
// ============================================================================

__global__ void grad_accumulate_kernel(float* dst, const float* src, float scale, size_t N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dst[i] += scale * src[i];
}
void gradient_accumulate(float* dst, const float* src, float scale, size_t N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    grad_accumulate_kernel<<<CEIL_DIV(N, block_size), block_size, 0, stream>>>(dst, src, scale, N);
    cudaCheck(cudaGetLastError());
}

// bf16 → fp32 accumulate (used when accumulating gradients into the master fp32 buffer).
__global__ void grad_accumulate_bf16_kernel(float* dst, const __nv_bfloat16* src, float scale, size_t N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dst[i] += scale * __bfloat162float(src[i]);
}
void gradient_accumulate_bf16(float* dst, const __nv_bfloat16* src, float scale, size_t N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    grad_accumulate_bf16_kernel<<<CEIL_DIV(N, block_size), block_size, 0, stream>>>(dst, src, scale, N);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// loss_scale_dynamic: detect overflow in gradients (any non-finite element)
// and update a loss-scale value accordingly.
//
//   grads:    [N]    fp32 gradient buffer
//   scale:    *float (in/out)  -- current loss scale
//   has_inf:  *int   (out)     -- 1 if overflow detected, 0 otherwise
// ============================================================================

__global__ void check_inf_kernel(int* has_inf, const float* grads, size_t N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = grads[i];
    if (!isfinite(v)) atomicExch(has_inf, 1);
}

__global__ void loss_scale_step_kernel(float* scale, const int* has_inf, float growth_factor, float backoff_factor) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (*has_inf) {
            *scale = (*scale) * backoff_factor;
        } else {
            *scale = (*scale) * growth_factor;
        }
    }
}

void loss_scale_dynamic_update(float* scale, int* has_inf, const float* grads, size_t N,
                               float growth_factor, float backoff_factor, cudaStream_t stream) {
    NVTX_RANGE_FN();
    cudaMemsetAsync(has_inf, 0, sizeof(int), stream);
    const int block_size = 256;
    int grid_size = std::min(1024, CEIL_DIV((int)N, block_size));
    check_inf_kernel<<<grid_size, block_size, 0, stream>>>(has_inf, grads, N);
    cudaCheck(cudaGetLastError());
    loss_scale_step_kernel<<<1, 1, 0, stream>>>(scale, has_inf, growth_factor, backoff_factor);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// sequence_packing: build cu_seqlens from a per-row sequence-length array.
//
//   seq_lens: [batch] int32  (each row's actual sequence length)
//   cu_seqlens: [batch + 1] int32  (exclusive prefix sum; cu_seqlens[0] = 0)
//
// Single-block exclusive scan; batch is expected to be small.
// ============================================================================

__global__ void cu_seqlens_kernel(int* cu_seqlens, const int* seq_lens, int batch) {
    if (blockIdx.x != 0) return;
    if (threadIdx.x == 0) {
        cu_seqlens[0] = 0;
        for (int i = 0; i < batch; ++i) {
            cu_seqlens[i + 1] = cu_seqlens[i] + seq_lens[i];
        }
    }
}
void build_cu_seqlens(int* cu_seqlens, const int* seq_lens, int batch, cudaStream_t stream) {
    NVTX_RANGE_FN();
    cu_seqlens_kernel<<<1, 1, 0, stream>>>(cu_seqlens, seq_lens, batch);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// document_causal_mask: build [N, N] additive mask where positions in
// different documents see -inf, and positions later than themselves within
// the same document also see -inf.
//
//   cu_seqlens: [batch + 1]
//   total_len:  cu_seqlens[batch]
//   mask:       [total_len, total_len]  floatX (additive; -inf for masked,
//               0.0 elsewhere)
// ============================================================================

__global__ void document_causal_mask_kernel(floatX* mask, const int* cu_seqlens, int batch, int total_len) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    if (i >= total_len || j >= total_len) return;
    // Find the document id for i and j via binary search.
    int doc_i = 0;
    for (int d = 0; d < batch; ++d) {
        if (i < cu_seqlens[d + 1]) { doc_i = d; break; }
    }
    int doc_j = 0;
    for (int d = 0; d < batch; ++d) {
        if (j < cu_seqlens[d + 1]) { doc_j = d; break; }
    }
    bool same_doc = (doc_i == doc_j);
    bool causal = (j <= i);
    if (!(same_doc && causal)) {
        mask[i * total_len + j] = (floatX)(-1e9f);
    } else {
        mask[i * total_len + j] = (floatX)(0.0f);
    }
}
void document_causal_mask(floatX* mask, const int* cu_seqlens, int batch, int total_len, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 64;
    dim3 grid(CEIL_DIV(total_len, block_size), total_len);
    document_causal_mask_kernel<<<grid, block_size, 0, stream>>>(mask, cu_seqlens, batch, total_len);
    cudaCheck(cudaGetLastError());
}
