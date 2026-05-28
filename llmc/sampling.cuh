/*
sampling.cuh — token sampling and logit-shaping kernels for inference.

Stages, indexed against §21 of nfn-coverage-todo.md:
  - temperature_scaling          (`logits /= T`)
  - logit_bias                   (additive per-token bias)
  - repetition_penalty           (multiply/divide previously seen tokens)
  - top_k_sampling               (mask logits to top-k, renormalise, sample)
  - top_p_sampling               (nucleus)
  - min_p_sampling               (prob ≥ min_p · max_prob)
  - typical_p_sampling           (locally typical)
  - grammar_constrained_decode   (mask logits against an accept-set bitmap)

Sampling reads a single uniform-[0,1) random number per row (caller supplies)
to keep RNG deterministic and avoid embedding cuRAND. For batched parallel
sampling pass `rand_uniform` of shape [rows]; this is the same scheme
inference engines like FlashInfer use.

For top-k / top-p / typical-p we use a one-block-per-row layout with up to
1024 threads in the block; each block does a parallel selection over the row.
*/
#pragma once

#include <assert.h>
#include <math.h>
#include <cfloat>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ============================================================================
// temperature_scaling: logits /= T (in-place ok). If T == 0, returns argmax —
// callers handle that case in the sampler directly.
// ============================================================================

__global__ void temperature_scaling_kernel(floatX* logits, float inv_t, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    logits[idx] = (floatX)((float)logits[idx] * inv_t);
}
void temperature_scaling_forward(floatX* logits, float temperature, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    float inv_t = 1.0f / fmaxf(temperature, 1e-6f);
    const int block_size = 256;
    temperature_scaling_kernel<<<CEIL_DIV(N, block_size), block_size, 0, stream>>>(logits, inv_t, N);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// logit_bias: per-token additive bias broadcast across rows.
//   logits: [rows, vocab]
//   bias:   [vocab]
// ============================================================================

__global__ void logit_bias_kernel(floatX* logits, const floatX* bias, int vocab, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int v = idx % vocab;
    logits[idx] = (floatX)((float)logits[idx] + (float)bias[v]);
}
void logit_bias_forward(floatX* logits, const floatX* bias, int rows, int vocab, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int N = rows * vocab;
    const int block_size = 256;
    logit_bias_kernel<<<CEIL_DIV(N, block_size), block_size, 0, stream>>>(logits, bias, vocab, N);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// repetition_penalty: for each row, walk a list of previous token ids and
// apply penalty: logit = logit > 0 ? logit / penalty : logit * penalty.
//   logits:        [rows, vocab]
//   prev_tokens:   [rows, max_prev]   (-1 padding to skip)
// ============================================================================

__global__ void repetition_penalty_kernel(floatX* logits, const int* prev_tokens,
                                          float penalty, int vocab, int max_prev) {
    int row = blockIdx.x;
    floatX* row_logits = logits + row * vocab;
    const int* row_prev = prev_tokens + row * max_prev;
    for (int i = threadIdx.x; i < max_prev; i += blockDim.x) {
        int t = row_prev[i];
        if (t < 0 || t >= vocab) continue;
        float v = (float)row_logits[t];
        v = v > 0.0f ? v / penalty : v * penalty;
        row_logits[t] = (floatX)v;
    }
}
void repetition_penalty_forward(floatX* logits, const int* prev_tokens,
                                float penalty, int rows, int vocab, int max_prev,
                                cudaStream_t stream) {
    NVTX_RANGE_FN();
    if (penalty == 1.0f) return;
    const int block_size = 128;
    repetition_penalty_kernel<<<rows, block_size, 0, stream>>>(logits, prev_tokens, penalty, vocab, max_prev);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// argmax_sampling: deterministic argmax over each row.
//   logits: [rows, vocab]
//   out:    [rows] int32
// ============================================================================

__global__ void argmax_kernel(int* out, const floatX* logits, int vocab) {
    int row = blockIdx.x;
    const floatX* row_logits = logits + row * vocab;
    float local_max = -INFINITY;
    int   local_arg = 0;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float v = (float)row_logits[i];
        if (v > local_max) { local_max = v; local_arg = i; }
    }
    // reduce across warps via shared memory
    __shared__ float s_max[32];
    __shared__ int   s_arg[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    // warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = __shfl_xor_sync(0xFFFFFFFF, local_max, offset);
        int   other_arg = __shfl_xor_sync(0xFFFFFFFF, local_arg, offset);
        if (other_max > local_max) { local_max = other_max; local_arg = other_arg; }
    }
    if (lane == 0) { s_max[warp] = local_max; s_arg[warp] = local_arg; }
    __syncthreads();
    if (warp == 0) {
        int num_warps = blockDim.x / 32;
        float m = (lane < num_warps) ? s_max[lane] : -INFINITY;
        int   a = (lane < num_warps) ? s_arg[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float om = __shfl_xor_sync(0xFFFFFFFF, m, offset);
            int   oa = __shfl_xor_sync(0xFFFFFFFF, a, offset);
            if (om > m) { m = om; a = oa; }
        }
        if (lane == 0) out[blockIdx.x] = a;
    }
}
void argmax_sampling(int* out_tokens, const floatX* logits, int rows, int vocab, cudaStream_t stream) {
    NVTX_RANGE_FN();
    argmax_kernel<<<rows, 256, 0, stream>>>(out_tokens, logits, vocab);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// categorical_sampling: given softmax probabilities and a uniform [0,1) per
// row, draw via inverse-CDF. Uses block-stride prefix-sum.
//
// (Reused by top-k / top-p / typical-p after they have already masked logits.)
// We compute softmax inside this kernel from logits directly (numerically
// stable max-subtraction).
//
//   logits:       [rows, vocab]
//   rand_uniform: [rows] (fp32 in [0,1))
//   out:          [rows] int32
// ============================================================================

__global__ void categorical_from_logits_kernel(int* out, const floatX* logits,
                                               const float* rand_uniform, int vocab) {
    int row = blockIdx.x;
    const floatX* row_logits = logits + row * vocab;

    // 1) row max
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float v = (float)row_logits[i];
        if (v > local_max) local_max = v;
    }
    float row_max = blockReduce<warpReduceMax>(local_max);

    // 2) sumexp
    float local_sumexp = 0.0f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        local_sumexp += expf((float)row_logits[i] - row_max);
    }
    float sumexp = blockReduce<warpReduceSum>(local_sumexp, true);

    // 3) inverse-CDF (single-thread; vocab is usually <= 200K so OK for first cut)
    if (threadIdx.x == 0) {
        float u = rand_uniform[row] * sumexp;
        float acc = 0.0f;
        int chosen = vocab - 1;
        for (int i = 0; i < vocab; ++i) {
            acc += expf((float)row_logits[i] - row_max);
            if (acc >= u) { chosen = i; break; }
        }
        out[row] = chosen;
    }
}
void categorical_sampling(int* out_tokens, const floatX* logits,
                          const float* rand_uniform, int rows, int vocab, cudaStream_t stream) {
    NVTX_RANGE_FN();
    categorical_from_logits_kernel<<<rows, 256, 0, stream>>>(out_tokens, logits, rand_uniform, vocab);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// top_k_sampling: mask out everything except top-k logits, then sample.
//
// Approach: per-row partial selection — find the k-th largest value, mask any
// logit < threshold to -inf, then call categorical sampler.
// O(vocab) per row; fine for inference at modest k.
// ============================================================================

__global__ void topk_threshold_kernel(float* thresholds, const floatX* logits, int vocab, int k) {
    int row = blockIdx.x;
    const floatX* row_logits = logits + row * vocab;
    // Simple selection: each thread keeps its own top-k list of (value), then
    // we merge across threads via the warp/block to find the smallest of the
    // global top-k. For modest k (<= 64) this fits in registers.
    constexpr int MAX_K = 64;
    float local_topk[MAX_K];
    for (int i = 0; i < MAX_K; ++i) local_topk[i] = -INFINITY;

    int k_clamped = k > MAX_K ? MAX_K : k;

    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float v = (float)row_logits[i];
        // insert into local top-k if greater than min
        float minv = local_topk[k_clamped - 1];
        if (v > minv) {
            int pos = k_clamped - 1;
            while (pos > 0 && local_topk[pos - 1] < v) {
                local_topk[pos] = local_topk[pos - 1];
                --pos;
            }
            local_topk[pos] = v;
        }
    }
    // Reduce across threads: shared mem of size blockDim*k floats — for k up to
    // 64 and 256 threads, that's 64KB which is fine in shared.
    __shared__ float s_topk[256 * MAX_K];
    int tid = threadIdx.x;
    for (int i = 0; i < k_clamped; ++i) s_topk[tid * MAX_K + i] = local_topk[i];
    __syncthreads();

    // Single-thread merge (acceptable for inference-time path).
    if (threadIdx.x == 0) {
        // Merge blockDim partial top-k lists into one global top-k.
        // Use a simple repeated max-over-all-heads selection.
        int num_threads = blockDim.x;
        // Initialise read indices.
        int* head_idx = (int*)alloca(sizeof(int) * num_threads);
        for (int t = 0; t < num_threads; ++t) head_idx[t] = 0;
        float global_topk[MAX_K];
        for (int i = 0; i < k_clamped; ++i) {
            float best = -INFINITY;
            int   best_t = 0;
            for (int t = 0; t < num_threads; ++t) {
                if (head_idx[t] < k_clamped) {
                    float v = s_topk[t * MAX_K + head_idx[t]];
                    if (v > best) { best = v; best_t = t; }
                }
            }
            global_topk[i] = best;
            head_idx[best_t] += 1;
        }
        thresholds[row] = global_topk[k_clamped - 1];
    }
}

__global__ void mask_below_threshold_kernel(floatX* logits, const float* thresholds, int vocab) {
    int row = blockIdx.x;
    floatX* row_logits = logits + row * vocab;
    float thr = thresholds[row];
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        if ((float)row_logits[i] < thr) row_logits[i] = (floatX)(-FLT_MAX);
    }
}

void top_k_mask(floatX* logits, int k, int rows, int vocab, float* scratch_thresholds, cudaStream_t stream) {
    NVTX_RANGE_FN();
    if (k <= 0 || k >= vocab) return;
    topk_threshold_kernel<<<rows, 256, 0, stream>>>(scratch_thresholds, logits, vocab, k);
    cudaCheck(cudaGetLastError());
    mask_below_threshold_kernel<<<rows, 256, 0, stream>>>(logits, scratch_thresholds, vocab);
    cudaCheck(cudaGetLastError());
}

void top_k_sampling(int* out_tokens, floatX* logits, int k,
                    const float* rand_uniform, int rows, int vocab,
                    float* scratch_thresholds, cudaStream_t stream) {
    top_k_mask(logits, k, rows, vocab, scratch_thresholds, stream);
    categorical_sampling(out_tokens, logits, rand_uniform, rows, vocab, stream);
}

// ============================================================================
// top_p_sampling (nucleus): mask all tokens outside the smallest cumulative
// probability mass >= p set. We sort descending per row, walk cumulative sum
// until it exceeds p, then mask everything else.
//
// Simple per-row sequential approach inside the block (acceptable at inference
// time). For high throughput / large vocab a parallel radix sort would be
// preferred (TODO TK 2.0 stack).
// ============================================================================

__global__ void top_p_mask_kernel(floatX* logits, float p, int vocab) {
    int row = blockIdx.x;
    floatX* row_logits = logits + row * vocab;

    // softmax (max-subtract + sumexp) so we can do a cumulative-prob threshold
    __shared__ float s_max, s_sumexp;
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float v = (float)row_logits[i];
        if (v > local_max) local_max = v;
    }
    float row_max = blockReduce<warpReduceMax>(local_max);
    if (threadIdx.x == 0) s_max = row_max;
    __syncthreads();

    float local_sumexp = 0.0f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        local_sumexp += expf((float)row_logits[i] - s_max);
    }
    float row_sumexp = blockReduce<warpReduceSum>(local_sumexp, true);
    if (threadIdx.x == 0) s_sumexp = row_sumexp;
    __syncthreads();

    // sequential per-row selection — one thread does the sorting+walking.
    if (threadIdx.x == 0) {
        // Build prob array on the fly. For very large vocab, a partitioning
        // scheme would be better; this is a correctness reference.
        // Find the threshold logit: descending sort + walk cumulative prob.
        // We do a simple repeated max-find since vocab is bounded.
        // For better perf use radix-sort or bitonic via TK 2.0 — flagged
        // there in the coverage doc.
        float kept_logit_min = -INFINITY;
        float cum_prob = 0.0f;
        // Track which indices have already been picked using a small bitmask
        // in dynamic shared memory — but we keep this simple: just walk by
        // repeatedly finding the next-max-not-yet-counted via a temporary.
        // Use a virtual sentinel by storing a "seen below" threshold approach:
        //   1) repeatedly find row_max value;
        //   2) accumulate exp(value-s_max)/s_sumexp;
        //   3) record threshold once cum >= p.
        // To avoid mutating logits, we collect "picked logits" in shared mem.
        // Cap picked entries at vocab; in practice this loop is short for
        // sane p.
        // For implementation simplicity, allocate up to 4096 picks.
        constexpr int MAX_PICKS = 4096;
        float picked[MAX_PICKS];
        int   n_picked = 0;
        float remaining_max = INFINITY;
        while (n_picked < MAX_PICKS && cum_prob < p) {
            float best = -INFINITY;
            for (int i = 0; i < vocab; ++i) {
                float v = (float)row_logits[i];
                if (v >= remaining_max) continue; // strictly less than prior max
                if (v > best) best = v;
            }
            if (!isfinite(best)) break;
            picked[n_picked++] = best;
            cum_prob += expf(best - s_max) / s_sumexp;
            remaining_max = best;
        }
        kept_logit_min = (n_picked > 0) ? picked[n_picked - 1] : -INFINITY;
        // store the threshold back into shared so the masking loop can read it
        s_max = kept_logit_min;  // reuse s_max as threshold scratch
    }
    __syncthreads();
    float thr = s_max;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        if ((float)row_logits[i] < thr) row_logits[i] = (floatX)(-FLT_MAX);
    }
}

void top_p_mask(floatX* logits, float p, int rows, int vocab, cudaStream_t stream) {
    NVTX_RANGE_FN();
    if (p >= 1.0f || p <= 0.0f) return;
    top_p_mask_kernel<<<rows, 256, 0, stream>>>(logits, p, vocab);
    cudaCheck(cudaGetLastError());
}

void top_p_sampling(int* out_tokens, floatX* logits, float p,
                    const float* rand_uniform, int rows, int vocab, cudaStream_t stream) {
    top_p_mask(logits, p, rows, vocab, stream);
    categorical_sampling(out_tokens, logits, rand_uniform, rows, vocab, stream);
}

// ============================================================================
// min_p_sampling: keep tokens with prob >= min_p * max_prob.
// Equivalent: keep logits with logit >= row_max + log(min_p).
// ============================================================================

__global__ void min_p_mask_kernel(floatX* logits, float min_p, int vocab) {
    int row = blockIdx.x;
    floatX* row_logits = logits + row * vocab;

    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float v = (float)row_logits[i];
        if (v > local_max) local_max = v;
    }
    float row_max = blockReduce<warpReduceMax>(local_max);
    float thr = row_max + logf(min_p);
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        if ((float)row_logits[i] < thr) row_logits[i] = (floatX)(-FLT_MAX);
    }
}
void min_p_mask(floatX* logits, float min_p, int rows, int vocab, cudaStream_t stream) {
    NVTX_RANGE_FN();
    if (min_p <= 0.0f) return;
    min_p_mask_kernel<<<rows, 256, 0, stream>>>(logits, min_p, vocab);
    cudaCheck(cudaGetLastError());
}
void min_p_sampling(int* out_tokens, floatX* logits, float min_p,
                    const float* rand_uniform, int rows, int vocab, cudaStream_t stream) {
    min_p_mask(logits, min_p, rows, vocab, stream);
    categorical_sampling(out_tokens, logits, rand_uniform, rows, vocab, stream);
}

// ============================================================================
// typical_p_sampling: keep tokens whose negative log-prob is within `eta` of
// the entropy. Approximation: drop tokens with |H + log p_i| > threshold so
// cumulative kept mass >= typical_p.
//
// Implementation: compute H over the row, sort tokens by |H + log p|
// ascending, walk cumulative prob until >= typical_p, mask the rest.
// ============================================================================

__global__ void typical_p_mask_kernel(floatX* logits, float typical_p, int vocab) {
    int row = blockIdx.x;
    floatX* row_logits = logits + row * vocab;

    __shared__ float s_max, s_sumexp, s_H;
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float v = (float)row_logits[i];
        if (v > local_max) local_max = v;
    }
    float row_max = blockReduce<warpReduceMax>(local_max);
    if (threadIdx.x == 0) s_max = row_max;
    __syncthreads();
    float local_sumexp = 0.0f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        local_sumexp += expf((float)row_logits[i] - s_max);
    }
    float row_sumexp = blockReduce<warpReduceSum>(local_sumexp, true);
    if (threadIdx.x == 0) s_sumexp = row_sumexp;
    __syncthreads();

    // Entropy
    float local_H = 0.0f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float p = expf((float)row_logits[i] - s_max) / s_sumexp;
        if (p > 0.0f) local_H -= p * logf(p);
    }
    float row_H = blockReduce<warpReduceSum>(local_H, true);
    if (threadIdx.x == 0) s_H = row_H;
    __syncthreads();

    // Walk by ascending |H + log p| via repeated min-find. Single-thread loop;
    // simple but correct.
    __shared__ float s_thr;
    if (threadIdx.x == 0) {
        float cum_prob = 0.0f;
        float last_dist = -INFINITY;
        float chosen_dist = INFINITY;
        constexpr int MAX_PICKS = 4096;
        int n_picked = 0;
        while (n_picked < MAX_PICKS && cum_prob < typical_p) {
            float best_dist = INFINITY;
            int   best_idx  = -1;
            for (int i = 0; i < vocab; ++i) {
                float p = expf((float)row_logits[i] - s_max) / s_sumexp;
                if (p == 0.0f) continue;
                float dist = fabsf(s_H + logf(p));
                if (dist <= last_dist) continue;
                if (dist < best_dist) { best_dist = dist; best_idx = i; }
            }
            if (best_idx < 0) break;
            float p = expf((float)row_logits[best_idx] - s_max) / s_sumexp;
            cum_prob += p;
            last_dist = best_dist;
            chosen_dist = best_dist;
            n_picked++;
        }
        s_thr = chosen_dist;
    }
    __syncthreads();

    float thr = s_thr;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float p = expf((float)row_logits[i] - s_max) / s_sumexp;
        float dist = p > 0.0f ? fabsf(s_H + logf(p)) : INFINITY;
        if (dist > thr) row_logits[i] = (floatX)(-FLT_MAX);
    }
}

void typical_p_mask(floatX* logits, float typical_p, int rows, int vocab, cudaStream_t stream) {
    NVTX_RANGE_FN();
    if (typical_p >= 1.0f || typical_p <= 0.0f) return;
    typical_p_mask_kernel<<<rows, 256, 0, stream>>>(logits, typical_p, vocab);
    cudaCheck(cudaGetLastError());
}
void typical_p_sampling(int* out_tokens, floatX* logits, float typical_p,
                        const float* rand_uniform, int rows, int vocab, cudaStream_t stream) {
    typical_p_mask(logits, typical_p, rows, vocab, stream);
    categorical_sampling(out_tokens, logits, rand_uniform, rows, vocab, stream);
}

// ============================================================================
// grammar_constrained_decode: mask logits against an accept-set bitmap.
//   accept_mask: [rows, vocab]  (1 = accept, 0 = reject; bf16 or fp16)
// Disallowed tokens get -FLT_MAX.
// ============================================================================

__global__ void grammar_mask_kernel(floatX* logits, const floatX* accept_mask, int vocab) {
    int row = blockIdx.x;
    floatX* row_logits = logits + row * vocab;
    const floatX* row_mask = accept_mask + row * vocab;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float m = (float)row_mask[i];
        if (m <= 0.0f) row_logits[i] = (floatX)(-FLT_MAX);
    }
}
void grammar_constrained_mask(floatX* logits, const floatX* accept_mask, int rows, int vocab, cudaStream_t stream) {
    NVTX_RANGE_FN();
    grammar_mask_kernel<<<rows, 256, 0, stream>>>(logits, accept_mask, vocab);
    cudaCheck(cudaGetLastError());
}
