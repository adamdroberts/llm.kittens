/*
routing_misc.cuh — semantic routing, JEPA, diffusion, ACT, and chunk-pool
helpers that don't fit into moe.cuh.

Covered:
  - LSH binarize + bit-pack (semantic_hasher, semantic_chunk_hasher)
  - causal_chunk_state (prefix cumsum / chunk-mean pool)
  - jepa_mask (random + multi-block)
  - latent_pool (masked mean-pool with fallback)
  - random_timesteps (per-row uniform)
  - mask_scheduler (Bernoulli mask with per-row prob)
  - act_halt_gate (mean-pool + linear + sigmoid)
  - act_weighted_sum
  - masked_argsort (used in semantic routers)
*/
#pragma once

#include <assert.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ============================================================================
// LSH binarize + bit-pack
//
//   proj:    [tables, planes, dim]
//   sem_vec: [rows, dim]
//   buckets: [rows, tables]   (int32, ≤ planes ≤ 30 bits per bucket)
// ============================================================================

__global__ void lsh_bitpack_kernel(int* buckets, const floatX* sem_vec, const floatX* proj,
                                   int rows, int tables, int planes, int dim) {
    int row = blockIdx.x;
    int t   = blockIdx.y;
    if (row >= rows || t >= tables) return;

    int bucket = 0;
    for (int p = threadIdx.x; p < planes; p += blockDim.x) {
        const floatX* pl = proj + (t * planes + p) * dim;
        const floatX* sv = sem_vec + row * dim;
        float acc = 0.0f;
        for (int d = 0; d < dim; ++d) acc += (float)sv[d] * (float)pl[d];
        int bit = acc > 0.0f ? 1 : 0;
        atomicOr(&bucket, bit << p);
    }
    if (threadIdx.x == 0) buckets[row * tables + t] = bucket;
}
void lsh_bitpack(int* buckets, const floatX* sem_vec, const floatX* proj,
                 int rows, int tables, int planes, int dim, cudaStream_t stream) {
    NVTX_RANGE_FN();
    dim3 grid(rows, tables);
    lsh_bitpack_kernel<<<grid, 128, 0, stream>>>(buckets, sem_vec, proj, rows, tables, planes, dim);
    cudaCheck(cudaGetLastError());
}

// Chunk variant: same op with an outer chunks axis preserved.
__global__ void lsh_bitpack_chunk_kernel(int* buckets, const floatX* sem_vec, const floatX* proj,
                                         int B, int chunks, int tables, int planes, int dim) {
    int b      = blockIdx.z;
    int chunk  = blockIdx.y;
    int t      = blockIdx.x;
    int bucket = 0;
    for (int p = threadIdx.x; p < planes; p += blockDim.x) {
        const floatX* pl = proj + (t * planes + p) * dim;
        const floatX* sv = sem_vec + (b * chunks + chunk) * dim;
        float acc = 0.0f;
        for (int d = 0; d < dim; ++d) acc += (float)sv[d] * (float)pl[d];
        int bit = acc > 0.0f ? 1 : 0;
        atomicOr(&bucket, bit << p);
    }
    if (threadIdx.x == 0) buckets[((b * chunks) + chunk) * tables + t] = bucket;
}
void lsh_bitpack_chunk(int* buckets, const floatX* sem_vec, const floatX* proj,
                       int B, int chunks, int tables, int planes, int dim, cudaStream_t stream) {
    NVTX_RANGE_FN();
    dim3 grid(tables, chunks, B);
    lsh_bitpack_chunk_kernel<<<grid, 128, 0, stream>>>(buckets, sem_vec, proj, B, chunks, tables, planes, dim);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// causal_chunk_state — prefix-safe (cumsum)+gather or "mean" chunk pool.
//
//   hidden:  [B, S, D]
//   out:     [B, num_chunks, D]
//
// mode == "prefix":  out[b, c, d] = (cumsum_{0..min(S, (c+1)*chunk_size)-1}) / count
// mode == "mean":    out[b, c, d] = mean over the chunk window
// ============================================================================

__global__ void chunk_state_mean_kernel(floatX* out, const floatX* hidden,
                                        int B, int S, int D, int chunk_size, int num_chunks) {
    int d  = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y;
    int b  = blockIdx.z;
    if (d >= D || c >= num_chunks || b >= B) return;
    int start = c * chunk_size;
    int end   = min(start + chunk_size, S);
    int count = end - start;
    if (count <= 0) {
        out[((b * num_chunks) + c) * D + d] = (floatX)0.0f;
        return;
    }
    float acc = 0.0f;
    for (int s = start; s < end; ++s) {
        acc += (float)hidden[((b * S) + s) * D + d];
    }
    out[((b * num_chunks) + c) * D + d] = (floatX)(acc / (float)count);
}
void chunk_state_mean_forward(floatX* out, const floatX* hidden,
                              int B, int S, int D, int chunk_size, int num_chunks,
                              cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    dim3 grid(CEIL_DIV(D, block_size), num_chunks, B);
    chunk_state_mean_kernel<<<grid, block_size, 0, stream>>>(out, hidden, B, S, D, chunk_size, num_chunks);
    cudaCheck(cudaGetLastError());
}

__global__ void chunk_state_prefix_kernel(floatX* out, const floatX* hidden,
                                          int B, int S, int D, int chunk_size, int num_chunks) {
    int d  = blockIdx.x * blockDim.x + threadIdx.x;
    int c  = blockIdx.y;
    int b  = blockIdx.z;
    if (d >= D || c >= num_chunks || b >= B) return;
    int boundary = (c + 1) * chunk_size - 1;
    if (boundary >= S) boundary = S - 1;
    if (boundary < 0)  boundary = 0;
    float acc = 0.0f;
    for (int s = 0; s <= boundary; ++s) {
        acc += (float)hidden[((b * S) + s) * D + d];
    }
    out[((b * num_chunks) + c) * D + d] = (floatX)(acc / (float)(boundary + 1));
}
void chunk_state_prefix_forward(floatX* out, const floatX* hidden,
                                int B, int S, int D, int chunk_size, int num_chunks,
                                cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    dim3 grid(CEIL_DIV(D, block_size), num_chunks, B);
    chunk_state_prefix_kernel<<<grid, block_size, 0, stream>>>(out, hidden, B, S, D, chunk_size, num_chunks);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// jepa_mask: random Bernoulli + multi-block masking. RNG is xorshift32 seeded
// per (row, seed).
//
//   tokens:        [B, S]    int32
//   masked_tokens: [B, S]    int32 (output; copy of tokens with mask_token_id where selected)
//   mask:          [B, S]    floatX (1.0 where masked, 0.0 else)
//
// Strategy "random":   each (b, s) independently masked with probability `mask_ratio`.
// Strategy "block":    for each batch, sample `num_blocks` random spans of size
//                      `min_block` .. `max_block` and mask them.
// ============================================================================

__device__ inline uint32_t xorshift32_local(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}

__global__ void jepa_mask_random_kernel(int* masked_tokens, floatX* mask,
                                        const int* tokens, float mask_ratio,
                                        int mask_token_id, uint32_t seed,
                                        int B, int S) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (s >= S || b >= B) return;
    uint32_t state = seed ^ (uint32_t)(b * 1664525u + s * 1013904223u);
    float u = ((float)(xorshift32_local(&state) >> 8) / 16777216.0f);
    bool m = u < mask_ratio;
    masked_tokens[b * S + s] = m ? mask_token_id : tokens[b * S + s];
    mask[b * S + s] = (floatX)(m ? 1.0f : 0.0f);
}

void jepa_mask_random(int* masked_tokens, floatX* mask, const int* tokens,
                      float mask_ratio, int mask_token_id, uint32_t seed,
                      int B, int S, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    dim3 grid(CEIL_DIV(S, block_size), B);
    jepa_mask_random_kernel<<<grid, block_size, 0, stream>>>(
        masked_tokens, mask, tokens, mask_ratio, mask_token_id, seed, B, S);
    cudaCheck(cudaGetLastError());
}

__global__ void jepa_mask_block_kernel(int* masked_tokens, floatX* mask,
                                       const int* tokens,
                                       int num_blocks, int min_block, int max_block,
                                       int mask_token_id, uint32_t seed,
                                       int B, int S) {
    int b = blockIdx.x;
    if (b >= B) return;
    // initialise mask to 0 / passthrough tokens (single thread first)
    if (threadIdx.x == 0) {
        for (int s = 0; s < S; ++s) {
            mask[b * S + s] = (floatX)0.0f;
            masked_tokens[b * S + s] = tokens[b * S + s];
        }
        uint32_t state = seed ^ (uint32_t)(b * 2654435761u);
        for (int blk = 0; blk < num_blocks; ++blk) {
            int range = max_block - min_block + 1;
            int block_len = min_block + (int)(xorshift32_local(&state) % range);
            int max_start = S - block_len;
            if (max_start < 0) max_start = 0;
            int start = (int)(xorshift32_local(&state) % (max_start + 1));
            int end   = min(start + block_len, S);
            for (int s = start; s < end; ++s) {
                mask[b * S + s] = (floatX)1.0f;
                masked_tokens[b * S + s] = mask_token_id;
            }
        }
    }
}
void jepa_mask_block(int* masked_tokens, floatX* mask, const int* tokens,
                     int num_blocks, int min_block, int max_block,
                     int mask_token_id, uint32_t seed,
                     int B, int S, cudaStream_t stream) {
    NVTX_RANGE_FN();
    jepa_mask_block_kernel<<<B, 1, 0, stream>>>(masked_tokens, mask, tokens,
                                                 num_blocks, min_block, max_block,
                                                 mask_token_id, seed, B, S);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// latent_pool: masked mean-pool with mean-fallback when mask sum is zero.
//   x:    [B, S, D]
//   mask: [B, S]   floatX
//   out:  [B, D]
// ============================================================================

__global__ void latent_pool_kernel(floatX* out, const floatX* x, const floatX* mask,
                                   int B, int S, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (d >= D) return;
    float weight_sum = 0.0f;
    float weighted   = 0.0f;
    float mean       = 0.0f;
    for (int s = 0; s < S; ++s) {
        float m = (float)mask[b * S + s];
        float v = (float)x[((b * S) + s) * D + d];
        weight_sum += m;
        weighted   += m * v;
        mean       += v;
    }
    mean = mean / (float)S;
    float pool = (weight_sum > 0.0f) ? (weighted / weight_sum) : mean;
    out[b * D + d] = (floatX)pool;
}
void latent_pool_forward(floatX* out, const floatX* x, const floatX* mask,
                         int B, int S, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    dim3 grid(CEIL_DIV(D, block_size), B);
    latent_pool_kernel<<<grid, block_size, 0, stream>>>(out, x, mask, B, S, D);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// random_timesteps: per-row uniform [0,1)
// ============================================================================

__global__ void random_timesteps_kernel(float* out, uint32_t seed, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    uint32_t state = seed ^ (uint32_t)(b * 1664525u);
    out[b] = (float)(xorshift32_local(&state) >> 8) / 16777216.0f;
}
void random_timesteps(float* out, uint32_t seed, int B, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 64;
    random_timesteps_kernel<<<CEIL_DIV(B, block_size), block_size, 0, stream>>>(out, seed, B);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// mask_scheduler: Bernoulli mask with per-row probability.
//   tokens:     [B, S]
//   timesteps:  [B]   fp32
//   out_tokens: [B, S]   (= mask_token_id where masked)
// ============================================================================

__global__ void mask_scheduler_kernel(int* out_tokens, const int* tokens, const float* timesteps,
                                      int mask_token_id, uint32_t seed, int B, int S) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (s >= S || b >= B) return;
    float p = timesteps[b];
    uint32_t state = seed ^ (uint32_t)(b * 1664525u + s * 1013904223u);
    float u = (float)(xorshift32_local(&state) >> 8) / 16777216.0f;
    out_tokens[b * S + s] = (u < p) ? mask_token_id : tokens[b * S + s];
}
void mask_scheduler(int* out_tokens, const int* tokens, const float* timesteps,
                    int mask_token_id, uint32_t seed, int B, int S, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    dim3 grid(CEIL_DIV(S, block_size), B);
    mask_scheduler_kernel<<<grid, block_size, 0, stream>>>(
        out_tokens, tokens, timesteps, mask_token_id, seed, B, S);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// act_halt_gate: mean-pool along S → linear → sigmoid.
//
//   x: [B, S, D]
//   weight: [D]  (linear with 1 output)
//   bias:   scalar
//   out:   [B, 1]
//
// We pool first (mean over S) then dot with the weight and add bias and apply
// sigmoid in one kernel.
// ============================================================================

__global__ void act_halt_gate_kernel(floatX* out, const floatX* x, const floatX* weight, float bias,
                                     int B, int S, int D) {
    int b = blockIdx.x;
    // mean pool along S
    float pool_acc = 0.0f;
    // We collapse to a per-thread accumulator over D, summing dot-product as we go.
    float local = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float sum = 0.0f;
        for (int s = 0; s < S; ++s) {
            sum += (float)x[((b * S) + s) * D + d];
        }
        float pooled = sum / (float)S;
        local += pooled * (float)weight[d];
    }
    float total = blockReduce<warpReduceSum>(local);
    if (threadIdx.x == 0) {
        float gate = 1.0f / (1.0f + expf(-(total + bias)));
        out[b] = (floatX)gate;
    }
}
void act_halt_gate(floatX* out, const floatX* x, const floatX* weight, float bias,
                   int B, int S, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    act_halt_gate_kernel<<<B, 128, 0, stream>>>(out, x, weight, bias, B, S, D);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// act_weighted_sum: Σ step_p · state across step axis.
//
//   states:  [B, steps, S, D]
//   weights: [B, steps]
//   out:     [B, S, D]
// ============================================================================

__global__ void act_weighted_sum_kernel(floatX* out, const floatX* states, const floatX* weights,
                                        int B, int steps, int S, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    int b = blockIdx.z;
    if (d >= D) return;
    float acc = 0.0f;
    for (int t = 0; t < steps; ++t) {
        float w = (float)weights[b * steps + t];
        acc += w * (float)states[(((b * steps) + t) * S + s) * D + d];
    }
    out[((b * S) + s) * D + d] = (floatX)acc;
}
void act_weighted_sum(floatX* out, const floatX* states, const floatX* weights,
                      int B, int steps, int S, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    dim3 grid(CEIL_DIV(D, block_size), S, B);
    act_weighted_sum_kernel<<<grid, block_size, 0, stream>>>(out, states, weights, B, steps, S, D);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// masked_argsort: descending argsort per row, but treat masked entries as
// `-inf` so they sort to the back.
//
//   logits:   [rows, K]
//   valid:    [rows, K]  (0 = invalid, 1 = valid)
//   indices:  [rows, K]  (output)
//
// Single-thread per row; OK for the small K used by routers.
// ============================================================================

__global__ void masked_argsort_kernel(int* indices, const float* logits, const uint8_t* valid,
                                      int rows, int K) {
    int row = blockIdx.x;
    if (row >= rows) return;
    if (threadIdx.x != 0) return;
    const float* row_logits = logits + row * K;
    const uint8_t* row_valid = valid + row * K;
    int* row_out = indices + row * K;

    // Track which indices are still unselected by clearing logits to -inf as we go.
    float wl[256];
    if (K > 256) return; // safety
    for (int k = 0; k < K; ++k) {
        wl[k] = row_valid[k] ? row_logits[k] : -INFINITY;
    }
    for (int t = 0; t < K; ++t) {
        float best = -INFINITY;
        int bi = 0;
        for (int k = 0; k < K; ++k) {
            if (wl[k] > best) { best = wl[k]; bi = k; }
        }
        row_out[t] = bi;
        wl[bi] = -INFINITY;
    }
}
void masked_argsort(int* indices, const float* logits, const uint8_t* valid, int rows, int K,
                    cudaStream_t stream) {
    NVTX_RANGE_FN();
    masked_argsort_kernel<<<rows, 1, 0, stream>>>(indices, logits, valid, rows, K);
    cudaCheck(cudaGetLastError());
}
