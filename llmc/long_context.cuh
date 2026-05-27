/*
long_context.cuh — KV-cache compression for long contexts.

  - h2o_eviction        (Heavy-Hitter Oracle): keep the top-k K,V tokens by
                        attention-score history.
  - snapkv              prompt-aware KV compression: pick KV positions whose
                        attention to the most recent observation window is
                        highest.
  - landmark_attention  insert "landmark" tokens that summarise spans; keep
                        landmarks always-visible, evict their member tokens.
  - infini_attention    keep a compressed memory state alongside the KV
                        cache (segment-level summary).
  - sink_token_cache    keep the first n_sink K,V slots permanently.

These are KV-management ops (not attention themselves) — they all reshape /
prune the KV buffer based on score histories the caller maintains.
*/
#pragma once

#include <assert.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ============================================================================
// H2O eviction: pick top-k positions by accumulated attention score (history)
// and write a compact KV buffer.
//
//   scores_hist:  [S]          (per-position score; caller accumulates)
//   K_in, V_in:   [S, D]
//   K_out, V_out: [k_keep, D]
//   keep_idx:     [k_keep]     (which positions were kept; int32)
// ============================================================================

__global__ void h2o_select_kernel(int* keep_idx, const float* scores, int S, int k_keep) {
    // Single-thread top-k by score (S expected modest after prior prunes).
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    constexpr int MAX_KEEP = 4096;
    if (k_keep > MAX_KEEP) return;
    float ts[MAX_KEEP]; int ti[MAX_KEEP];
    for (int i = 0; i < k_keep; ++i) { ts[i] = -INFINITY; ti[i] = 0; }
    for (int p = 0; p < S; ++p) {
        float v = scores[p];
        if (v > ts[k_keep - 1]) {
            int pos = k_keep - 1;
            while (pos > 0 && ts[pos - 1] < v) { ts[pos] = ts[pos - 1]; ti[pos] = ti[pos - 1]; --pos; }
            ts[pos] = v; ti[pos] = p;
        }
    }
    // Sort indices ascending so K_out preserves order
    for (int i = 1; i < k_keep; ++i) {
        int v = ti[i]; int j = i;
        while (j > 0 && ti[j - 1] > v) { ti[j] = ti[j - 1]; --j; }
        ti[j] = v;
    }
    for (int i = 0; i < k_keep; ++i) keep_idx[i] = ti[i];
}

__global__ void gather_kv_kernel(floatX* K_out, floatX* V_out,
                                 const floatX* K_in, const floatX* V_in,
                                 const int* keep_idx, int k_keep, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (d >= D || t >= k_keep) return;
    int src = keep_idx[t];
    K_out[t * D + d] = K_in[src * D + d];
    V_out[t * D + d] = V_in[src * D + d];
}

void h2o_eviction(floatX* K_out, floatX* V_out, int* keep_idx,
                  const floatX* K_in, const floatX* V_in, const float* scores,
                  int S, int k_keep, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    h2o_select_kernel<<<1, 1, 0, stream>>>(keep_idx, scores, S, k_keep);
    cudaCheck(cudaGetLastError());
    const int block_size = 128;
    dim3 grid(CEIL_DIV(D, block_size), k_keep);
    gather_kv_kernel<<<grid, block_size, 0, stream>>>(K_out, V_out, K_in, V_in, keep_idx, k_keep, D);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// SnapKV: average attention score from a recent observation window onto each
// key position; keep the top-k.
//
//   obs_q:  [W, D]   (Q of the last W tokens)
//   K_in:   [S, D]
//   V_in:   [S, D]
//   K_out:  [k_keep, D]
//   V_out:  [k_keep, D]
// ============================================================================

__global__ void snapkv_score_kernel(float* scores, const floatX* obs_q, const floatX* K_in,
                                    int W, int S, int D) {
    int t = blockIdx.x;
    if (t >= S) return;
    if (threadIdx.x != 0) return;
    float acc = 0.0f;
    for (int w = 0; w < W; ++w) {
        float dot = 0.0f;
        const floatX* qr = obs_q + w * D;
        const floatX* kr = K_in + t * D;
        for (int d = 0; d < D; ++d) dot += (float)qr[d] * (float)kr[d];
        acc += dot;
    }
    scores[t] = acc / (float)W;
}

void snapkv_select(floatX* K_out, floatX* V_out, int* keep_idx,
                   const floatX* obs_q, const floatX* K_in, const floatX* V_in,
                   int W, int S, int k_keep, int D,
                   float* scratch_scores, cudaStream_t stream) {
    NVTX_RANGE_FN();
    snapkv_score_kernel<<<S, 1, 0, stream>>>(scratch_scores, obs_q, K_in, W, S, D);
    cudaCheck(cudaGetLastError());
    h2o_eviction(K_out, V_out, keep_idx, K_in, V_in, scratch_scores, S, k_keep, D, stream);
}

// ============================================================================
// Landmark attention: insert a landmark token every `span` positions; keep
// only landmarks plus the tail (most recent `tail` positions).
//
//   K_in, V_in: [S, D]
//   K_out, V_out: [k_out, D]   k_out = floor(S/span) + tail
//   indices:    [k_out]
// ============================================================================

__global__ void landmark_select_kernel(int* keep_idx, int S, int span, int tail) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_landmarks = S / span;
    int total = n_landmarks + tail;
    if (i >= total) return;
    if (i < n_landmarks) {
        keep_idx[i] = (i + 1) * span - 1;
    } else {
        int t = i - n_landmarks;
        keep_idx[i] = S - tail + t;
    }
}
void landmark_attention_select(floatX* K_out, floatX* V_out, int* keep_idx,
                               const floatX* K_in, const floatX* V_in,
                               int S, int D, int span, int tail, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int n_landmarks = S / span;
    int k_out = n_landmarks + tail;
    landmark_select_kernel<<<CEIL_DIV(k_out, 64), 64, 0, stream>>>(keep_idx, S, span, tail);
    cudaCheck(cudaGetLastError());
    const int block_size = 128;
    dim3 grid(CEIL_DIV(D, block_size), k_out);
    gather_kv_kernel<<<grid, block_size, 0, stream>>>(K_out, V_out, K_in, V_in, keep_idx, k_out, D);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Infini-attention memory update: accumulate compressed memory state per
// segment. The memory is a low-rank Σ k_iv_i^T accumulator updated with each
// new (K, V) segment, plus a normalizer Σ k_i.
//
//   K_seg, V_seg: [S, D]
//   mem:          [D, D]      (accumulator; in/out)
//   z:            [D]         (normalizer; in/out)
// ============================================================================

__global__ void infini_memory_update_kernel(floatX* mem, floatX* z,
                                            const floatX* K_seg, const floatX* V_seg, int S, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= D || j >= D) return;
    float acc = (float)mem[i * D + j];
    for (int t = 0; t < S; ++t) {
        acc += (float)K_seg[t * D + i] * (float)V_seg[t * D + j];
    }
    mem[i * D + j] = (floatX)acc;
    if (j == 0) {
        float zacc = (float)z[i];
        for (int t = 0; t < S; ++t) zacc += (float)K_seg[t * D + i];
        z[i] = (floatX)zacc;
    }
}
void infini_memory_update(floatX* mem, floatX* z, const floatX* K_seg, const floatX* V_seg,
                          int S, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    dim3 block(16, 16);
    dim3 grid(CEIL_DIV(D, 16), CEIL_DIV(D, 16));
    infini_memory_update_kernel<<<grid, block, 0, stream>>>(mem, z, K_seg, V_seg, S, D);
    cudaCheck(cudaGetLastError());
}

// Infini-attention retrieval: y = (q · mem) / (q · z + eps)
__global__ void infini_retrieve_kernel(floatX* y, const floatX* q, const floatX* mem, const floatX* z, int D) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= D) return;
    float qz = 0.0f;
    for (int i = 0; i < D; ++i) qz += (float)q[i] * (float)z[i];
    float qm = 0.0f;
    for (int i = 0; i < D; ++i) qm += (float)q[i] * (float)mem[i * D + o];
    y[o] = (floatX)(qm / (qz + 1e-6f));
}
void infini_retrieve(floatX* y, const floatX* q, const floatX* mem, const floatX* z, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 64;
    infini_retrieve_kernel<<<CEIL_DIV(D, block_size), block_size, 0, stream>>>(y, q, mem, z, D);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Sink token cache: keep the first n_sink KV slots permanently + a rolling
// window of size `window` from the tail. Equivalent to landmark_attention
// with span=∞ but with an explicit sink prefix.
// ============================================================================

__global__ void sink_token_select_kernel(int* keep_idx, int S, int n_sink, int window) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_sink + window;
    if (i >= total) return;
    if (i < n_sink) keep_idx[i] = i;
    else            keep_idx[i] = S - window + (i - n_sink);
}
void sink_token_cache_select(floatX* K_out, floatX* V_out, int* keep_idx,
                             const floatX* K_in, const floatX* V_in,
                             int S, int D, int n_sink, int window, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int k_out = n_sink + window;
    sink_token_select_kernel<<<CEIL_DIV(k_out, 64), 64, 0, stream>>>(keep_idx, S, n_sink, window);
    cudaCheck(cudaGetLastError());
    const int block_size = 128;
    dim3 grid(CEIL_DIV(D, block_size), k_out);
    gather_kv_kernel<<<grid, block_size, 0, stream>>>(K_out, V_out, K_in, V_in, keep_idx, k_out, D);
    cudaCheck(cudaGetLastError());
}
