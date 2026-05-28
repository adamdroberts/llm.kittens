/*
moe.cuh — MoE routing and dispatch primitives.

Covered:
  - topk_route          softmax → topk → renormalise + routing telemetry
  - expert_permute      token-sort-by-expert (for grouped dispatch)
  - expert_unpermute    inverse: scatter weighted expert outputs back to tokens
  - broadcast_expert_routes
  - broadcast_chunk_routes
  - capacity_factor_dispatch
  - auxfree_load_balancing (bias-adjusted routing)
  - softmoe_dispatch
  - mixture_of_depths_select

The "expert_dispatch" stage in NeuralFn is `permute` + N grouped-GEMM steps
(one SwiGLU each) + `unpermute`. The grouped GEMM is delegated to the matmul
path (cublasLtMatmulBatched / TK 2.0); this file owns the permute/unpermute
plus the routing scoring.
*/
#pragma once

#include <assert.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ============================================================================
// topk_route: per-row softmax, take top-K, renormalise. Telemetry buffers
// (selection_counts, weight_mass) are accumulated.
//
//   router_logits: [rows, E]
//   topk_weights:  [rows, K]  (floatX)
//   topk_indices:  [rows, K]  (int32)
//   selection_counts: [E]     (uint32, accumulated)
//   weight_mass:      [E]     (fp32, accumulated)
//
// Caller resets selection_counts/weight_mass to 0 before the call if they
// want per-step stats.
// ============================================================================

__global__ void topk_route_kernel(floatX* topk_weights, int* topk_indices,
                                  const floatX* router_logits, int E, int K, int rows,
                                  unsigned int* selection_counts, float* weight_mass) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const floatX* row_logits = router_logits + row * E;

    // softmax → top-k with simple register-resident maintenance.
    // 1) row max
    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < E; j += blockDim.x) {
        float v = (float)row_logits[j];
        if (v > local_max) local_max = v;
    }
    float row_max = blockReduce<warpReduceMax>(local_max);
    // 2) softmax probabilities + topk maintenance (single-thread for simplicity)
    if (threadIdx.x == 0) {
        // stable softmax sum
        float sum_exp = 0.0f;
        for (int j = 0; j < E; ++j) sum_exp += expf((float)row_logits[j] - row_max);
        // top-k via selection
        constexpr int MAX_K = 16;
        float best_p[MAX_K];
        int   best_i[MAX_K];
        for (int k = 0; k < K; ++k) { best_p[k] = -INFINITY; best_i[k] = 0; }
        for (int j = 0; j < E; ++j) {
            float p = expf((float)row_logits[j] - row_max) / sum_exp;
            // insert into top-k
            float min_v = best_p[K - 1];
            if (p > min_v) {
                int pos = K - 1;
                while (pos > 0 && best_p[pos - 1] < p) {
                    best_p[pos] = best_p[pos - 1]; best_i[pos] = best_i[pos - 1];
                    --pos;
                }
                best_p[pos] = p; best_i[pos] = j;
            }
        }
        // renormalise
        float sum_top = 0.0f;
        for (int k = 0; k < K; ++k) sum_top += best_p[k];
        for (int k = 0; k < K; ++k) {
            float w = best_p[k] / fmaxf(sum_top, 1e-12f);
            topk_weights[row * K + k] = (floatX)w;
            topk_indices[row * K + k] = best_i[k];
            if (selection_counts) atomicAdd(&selection_counts[best_i[k]], 1u);
            if (weight_mass)      atomicAdd(&weight_mass[best_i[k]], w);
        }
    }
}

void topk_route(floatX* topk_weights, int* topk_indices,
                const floatX* router_logits, int rows, int E, int K,
                unsigned int* selection_counts, float* weight_mass,
                cudaStream_t stream) {
    NVTX_RANGE_FN();
    topk_route_kernel<<<rows, 64, 0, stream>>>(
        topk_weights, topk_indices, router_logits, E, K, rows,
        selection_counts, weight_mass);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// expert_permute: sort tokens by their (top-1) expert assignment so per-expert
// outputs are contiguous in memory.
//
// Inputs:
//   tokens:        [rows, dim] floatX
//   topk_indices:  [rows, K]   int32
//   topk_weights:  [rows, K]   floatX
//
// Outputs:
//   permuted_tokens: [rows*K, dim]  (token replicated per top-k assignment)
//   sort_order:      [rows*K]       (original (row, k) index)
//   expert_offsets:  [E + 1]        (CSR-style; expert e occupies
//                                    permuted_tokens[expert_offsets[e]:expert_offsets[e+1]])
//
// Implementation: count per-expert sizes, prefix-sum to get offsets, then
// scatter tokens into permuted layout. Three passes over [rows, K].
// ============================================================================

__global__ void moe_count_per_expert_kernel(int* counts, const int* topk_indices, int rows, int K, int E) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * K;
    if (idx >= total) return;
    int e = topk_indices[idx];
    if (e >= 0 && e < E) atomicAdd(&counts[e], 1);
}

__global__ void exclusive_scan_kernel(int* offsets, const int* counts, int E) {
    // E is small (<= 128 typically); single-block scan.
    if (blockIdx.x != 0) return;
    int i = threadIdx.x;
    if (i > E) return;
    int sum = 0;
    for (int e = 0; e < i; ++e) sum += counts[e];
    offsets[i] = sum;
}

__global__ void moe_scatter_permute_kernel(floatX* permuted, int* sort_order,
                                           const floatX* tokens, const int* topk_indices,
                                           const int* offsets, int* expert_cursors,
                                           int rows, int K, int dim) {
    int idx = blockIdx.x;
    int total = rows * K;
    if (idx >= total) return;
    int e = topk_indices[idx];
    int slot = atomicAdd(&expert_cursors[e], 1);
    int dst_row = offsets[e] + slot;
    int src_row = idx / K;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        permuted[dst_row * dim + d] = tokens[src_row * dim + d];
    }
    if (threadIdx.x == 0) sort_order[dst_row] = idx;
}

void expert_permute(floatX* permuted_tokens, int* sort_order, int* expert_offsets,
                    const floatX* tokens, const int* topk_indices,
                    int rows, int K, int dim, int E,
                    int* scratch_counts, int* scratch_cursors,
                    cudaStream_t stream) {
    NVTX_RANGE_FN();
    cudaMemsetAsync(scratch_counts, 0, sizeof(int) * E, stream);
    cudaMemsetAsync(expert_offsets, 0, sizeof(int) * (E + 1), stream);
    cudaMemsetAsync(scratch_cursors, 0, sizeof(int) * E, stream);

    const int block_size = 256;
    int total = rows * K;
    moe_count_per_expert_kernel<<<CEIL_DIV(total, block_size), block_size, 0, stream>>>(
        scratch_counts, topk_indices, rows, K, E);
    cudaCheck(cudaGetLastError());

    exclusive_scan_kernel<<<1, E + 1, 0, stream>>>(expert_offsets, scratch_counts, E);
    cudaCheck(cudaGetLastError());

    moe_scatter_permute_kernel<<<total, 64, 0, stream>>>(
        permuted_tokens, sort_order, tokens, topk_indices,
        expert_offsets, scratch_cursors, rows, K, dim);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// expert_unpermute: scatter-add weighted expert outputs back into the
// original token positions.
//
//   permuted_out:  [rows*K, dim]   (output from per-expert GEMMs)
//   sort_order:    [rows*K]        (from expert_permute)
//   topk_weights:  [rows, K]
//   out:           [rows, dim]     (accumulator; zero-init by caller)
// ============================================================================

__global__ void moe_unpermute_kernel(floatX* out, const floatX* permuted_out,
                                     const int* sort_order, const floatX* topk_weights,
                                     int rows, int K, int dim) {
    int slot = blockIdx.x;
    int total = rows * K;
    if (slot >= total) return;
    int orig = sort_order[slot];
    int row = orig / K;
    int k   = orig % K;
    float w = (float)topk_weights[row * K + k];
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = (float)permuted_out[slot * dim + d] * w;
        atomicAdd((float*)0, 0); // ensures atomic path is materialised
        // Use atomicAdd on bf16 via fp32 staging — caller must use fp32 out for safety.
        // For bf16 accumulation we use a CAS-based bf16 add (see helper below).
        // Simpler: read-modify-write per element (race ok because each token
        // index has top-k slots feeding it; this kernel is launched per slot
        // sequentially across blocks and atomics are required).
        // For correctness without atomics on bf16, callers should choose an
        // fp32 accumulator and convert at the end.
        unsigned int* addr = (unsigned int*)(out + (row * dim + (d & ~1)));
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
}

void expert_unpermute(floatX* out, const floatX* permuted_out,
                      const int* sort_order, const floatX* topk_weights,
                      int rows, int K, int dim, cudaStream_t stream) {
    NVTX_RANGE_FN();
    moe_unpermute_kernel<<<rows * K, 64, 0, stream>>>(
        out, permuted_out, sort_order, topk_weights, rows, K, dim);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// broadcast_expert_routes: batch-level routes -> per-position routes.
//
//   in_weights:  [B, K]  (or [B, 1, K])
//   in_indices:  [B, K]
//   out_weights: [B, S, K]
//   out_indices: [B, S, K]
// ============================================================================

__global__ void broadcast_expert_routes_kernel(floatX* out_w, int* out_i,
                                               const floatX* in_w, const int* in_i,
                                               int B, int S, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * K;
    if (idx >= total) return;
    int k = idx % K;
    int s = (idx / K) % S;
    int b = idx / (K * S);
    out_w[idx] = in_w[b * K + k];
    out_i[idx] = in_i[b * K + k];
    (void)s;
}
void broadcast_expert_routes(floatX* out_w, int* out_i, const floatX* in_w, const int* in_i,
                             int B, int S, int K, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int total = B * S * K;
    const int block_size = 256;
    broadcast_expert_routes_kernel<<<CEIL_DIV(total, block_size), block_size, 0, stream>>>(
        out_w, out_i, in_w, in_i, B, S, K);
    cudaCheck(cudaGetLastError());
}

// broadcast_chunk_routes: chunk-level routes -> per-token routes.
//   in_w:  [B, num_chunks, K]
//   in_i:  [B, num_chunks, K]
//   out_w: [B, S, K]
//   out_i: [B, S, K]
//
// token at sequence position s belongs to chunk (s / chunk_size), clamped to num_chunks-1.
__global__ void broadcast_chunk_routes_kernel(floatX* out_w, int* out_i,
                                              const floatX* in_w, const int* in_i,
                                              int B, int S, int K, int num_chunks, int chunk_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * K;
    if (idx >= total) return;
    int k = idx % K;
    int s = (idx / K) % S;
    int b = idx / (K * S);
    int chunk = min(s / chunk_size, num_chunks - 1);
    int src = (b * num_chunks + chunk) * K + k;
    out_w[idx] = in_w[src];
    out_i[idx] = in_i[src];
}
void broadcast_chunk_routes(floatX* out_w, int* out_i, const floatX* in_w, const int* in_i,
                            int B, int S, int K, int num_chunks, int chunk_size, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int total = B * S * K;
    const int block_size = 256;
    broadcast_chunk_routes_kernel<<<CEIL_DIV(total, block_size), block_size, 0, stream>>>(
        out_w, out_i, in_w, in_i, B, S, K, num_chunks, chunk_size);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// capacity_factor_dispatch: drop tokens that exceed per-expert capacity.
//   topk_indices: [rows, K]   (in/out, replaces over-capacity assignments with -1)
//   topk_weights: [rows, K]   (in/out, zeros over-capacity weights)
//   capacity:     int
//
// Single block per expert; uses atomic counters to admit the first `capacity`
// tokens.
// ============================================================================

__global__ void capacity_factor_kernel(int* topk_indices, floatX* topk_weights,
                                       int* counters, int rows, int K, int E, int capacity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * K;
    if (idx >= total) return;
    int e = topk_indices[idx];
    if (e < 0 || e >= E) return;
    int slot = atomicAdd(&counters[e], 1);
    if (slot >= capacity) {
        topk_indices[idx] = -1;
        topk_weights[idx] = (floatX)0.0f;
    }
}
void capacity_factor_dispatch(int* topk_indices, floatX* topk_weights,
                              int* scratch_counters, int rows, int K, int E, int capacity,
                              cudaStream_t stream) {
    NVTX_RANGE_FN();
    cudaMemsetAsync(scratch_counters, 0, sizeof(int) * E, stream);
    int total = rows * K;
    const int block_size = 256;
    capacity_factor_kernel<<<CEIL_DIV(total, block_size), block_size, 0, stream>>>(
        topk_indices, topk_weights, scratch_counters, rows, K, E, capacity);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// auxfree_load_balancing (DeepSeek-V3 style):
//
//   routing_score = sigmoid(W·x) + bias[e]
//   bias is updated each step by:  bias[e] -= γ * (load[e]/avg_load - 1)
//
// We provide the bias-update kernel; routing scoring is a regular linear +
// sigmoid + topk handled elsewhere.
// ============================================================================

__global__ void auxfree_bias_update_kernel(float* bias, const float* load, int E, float gamma) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    float sum = 0.0f;
    for (int i = 0; i < E; ++i) sum += load[i];
    float avg = sum / (float)E + 1e-12f;
    bias[e] -= gamma * (load[e] / avg - 1.0f);
}
void auxfree_bias_update(float* bias, const float* load, int E, float gamma, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 32;
    auxfree_bias_update_kernel<<<CEIL_DIV(E, block_size), block_size, 0, stream>>>(bias, load, E, gamma);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// softmoe_dispatch: continuous routing — every token contributes to every
// expert via a softmax over (token, slot) assignment. Tokens are mixed
// linearly via a learned dispatch matrix.
//
//   tokens:    [rows, dim]
//   dispatch:  [rows, E*slots]
//   slots_out: [E*slots, dim]
//
// Implementation: softmax(dispatch) along rows, then tokens.T @ probs.
// This is a small GEMM + softmax + GEMM; caller should call matmul for the
// GEMMs. We provide the softmax_along_rows helper.
// ============================================================================

__global__ void softmax_along_rows_kernel(floatX* probs, const floatX* logits, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    float local_max = -INFINITY;
    for (int r = 0; r < rows; ++r) {
        float v = (float)logits[r * cols + col];
        if (v > local_max) local_max = v;
    }
    float sumexp = 0.0f;
    for (int r = 0; r < rows; ++r) sumexp += expf((float)logits[r * cols + col] - local_max);
    for (int r = 0; r < rows; ++r) {
        probs[r * cols + col] = (floatX)(expf((float)logits[r * cols + col] - local_max) / sumexp);
    }
}
void softmax_along_rows(floatX* probs, const floatX* logits, int rows, int cols, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 64;
    softmax_along_rows_kernel<<<CEIL_DIV(cols, block_size), block_size, 0, stream>>>(probs, logits, rows, cols);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Mixture-of-Depths: per-token continue/skip mask. router_logits is per-token
// fp32; top-k tokens (sorted by score) are kept active for the block.
//
//   router_logits: [rows]
//   threshold:     scalar — pick top `keep` rows.
//   keep_mask:     [rows] uint8 (1=keep, 0=skip)
//
// Implementation: simple partial-sort (small `keep` typical).
// ============================================================================

__global__ void mod_threshold_kernel(uint8_t* keep_mask, const float* logits, int rows, int keep) {
    // Single-block partial selection (rows expected modest).
    if (blockIdx.x != 0) return;
    // Find the (rows-keep)-th smallest = threshold for keep set.
    if (threadIdx.x == 0) {
        // Brute force for clarity; production should use a fast top-k.
        // Find threshold via quickselect-like partial sort.
        // For now, a simple O(rows * keep) selection:
        // copy logits into a working array on stack — limited to small rows.
        const int MAX_ROWS = 4096;
        float wl[MAX_ROWS];
        if (rows > MAX_ROWS) return;
        for (int i = 0; i < rows; ++i) wl[i] = logits[i];
        for (int t = 0; t < keep; ++t) {
            float best = -INFINITY;
            int bi = 0;
            for (int i = 0; i < rows; ++i) {
                if (wl[i] > best) { best = wl[i]; bi = i; }
            }
            keep_mask[bi] = 1;
            wl[bi] = -INFINITY;
        }
    }
}
void mixture_of_depths_select(uint8_t* keep_mask, const float* logits, int rows, int keep, cudaStream_t stream) {
    NVTX_RANGE_FN();
    cudaMemsetAsync(keep_mask, 0, rows, stream);
    mod_threshold_kernel<<<1, 32, 0, stream>>>(keep_mask, logits, rows, keep);
    cudaCheck(cudaGetLastError());
}
