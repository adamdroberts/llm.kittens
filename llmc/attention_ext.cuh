/*
attention_ext.cuh — attention variants beyond the bf16 causal MHA / GQA path.

Most of these are mask / bias generators that feed the existing SDPA kernel
plus a few small dedicated paths (attentionless_decoder, cross-attention shape
helpers, paged attention reference).

For the heavy lifters (sliding window, ALiBi-biased, block-sparse, ring attn,
NSA, differential attention) the recommended target is TK 2.0 fused attention
with the appropriate mask/bias plugin. The CUDA kernels here are correctness
references suitable for small shapes; production paths should swap them out.
*/
#pragma once

#include <assert.h>
#include <cfloat>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ============================================================================
// sliding_window_attention mask: bool/float mask for the (q, k) attention
// matrix. window=W keeps |q-k| <= W; combined with causal (k <= q).
// ============================================================================

__global__ void sliding_window_mask_kernel(floatX* mask, int S_q, int S_k, int window, bool causal) {
    int q = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= S_k || q >= S_q) return;
    int dist = q - k;
    bool keep = (dist >= 0) && (dist <= window);
    if (!causal) keep = (abs(dist) <= window);
    mask[q * S_k + k] = keep ? (floatX)0.0f : (floatX)(-1e9f);
}
void sliding_window_mask(floatX* mask, int S_q, int S_k, int window, bool causal, cudaStream_t stream) {
    NVTX_RANGE_FN();
    dim3 grid(CEIL_DIV(S_k, 128), S_q);
    sliding_window_mask_kernel<<<grid, 128, 0, stream>>>(mask, S_q, S_k, window, causal);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// ALiBi additive bias for a fully-additive attention path (use this when SDPA
// supports additive bias). See rope_ext.cuh for the slopes+bias generator;
// this header just declares the convenience launcher to apply bias to scores.
//
// We provide alibi_attention_scores: given Q, K and per-head slopes, compute
// q·k + alibi as a single pass for very small sizes.
// ============================================================================

__global__ void alibi_scores_kernel(floatX* scores, const floatX* q, const floatX* k,
                                    const float* slopes, int H, int S_q, int S_k, int D) {
    int kk = blockIdx.x * blockDim.x + threadIdx.x;
    int qq = blockIdx.y;
    int hh = blockIdx.z;
    if (kk >= S_k || qq >= S_q || hh >= H) return;
    const floatX* qr = q + (hh * S_q + qq) * D;
    const floatX* kr = k + (hh * S_k + kk) * D;
    float acc = 0.0f;
    for (int d = 0; d < D; ++d) acc += (float)qr[d] * (float)kr[d];
    float bias = -fabsf((float)(qq - kk)) * slopes[hh];
    scores[((hh * S_q + qq) * S_k) + kk] = (floatX)(acc / sqrtf((float)D) + bias);
}
void alibi_attention_scores(floatX* scores, const floatX* q, const floatX* k, const float* slopes,
                            int H, int S_q, int S_k, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    dim3 grid(CEIL_DIV(S_k, 128), S_q, H);
    alibi_scores_kernel<<<grid, 128, 0, stream>>>(scores, q, k, slopes, H, S_q, S_k, D);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Block-sparse mask: keep (q, k) if both q and k belong to the same window
// block, OR if k is in the global-attention set (passed as a bitmap).
//
//   global_mask: [S_k] uint8 (1 if k is global, 0 otherwise)
// ============================================================================

__global__ void block_sparse_mask_kernel(floatX* mask, const uint8_t* global_mask,
                                         int S_q, int S_k, int block, bool causal) {
    int q = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= S_k || q >= S_q) return;
    int qb = q / block;
    int kb = k / block;
    bool same_block = (qb == kb);
    bool global = global_mask[k] != 0;
    bool keep = same_block || global;
    if (causal && k > q) keep = false;
    mask[q * S_k + k] = keep ? (floatX)0.0f : (floatX)(-1e9f);
}
void block_sparse_mask(floatX* mask, const uint8_t* global_mask, int S_q, int S_k,
                       int block, bool causal, cudaStream_t stream) {
    NVTX_RANGE_FN();
    dim3 grid(CEIL_DIV(S_k, 128), S_q);
    block_sparse_mask_kernel<<<grid, 128, 0, stream>>>(mask, global_mask, S_q, S_k, block, causal);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Streaming attention sinks: same as causal but the first `n_sink` positions
// are always visible regardless of window. Mask builder.
// ============================================================================

__global__ void streaming_sinks_mask_kernel(floatX* mask, int S_q, int S_k, int window, int n_sink) {
    int q = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= S_k || q >= S_q) return;
    bool sink = (k < n_sink);
    bool causal = (k <= q);
    bool window_ok = (q - k <= window);
    bool keep = (sink && causal) || (causal && window_ok);
    mask[q * S_k + k] = keep ? (floatX)0.0f : (floatX)(-1e9f);
}
void streaming_sinks_mask(floatX* mask, int S_q, int S_k, int window, int n_sink, cudaStream_t stream) {
    NVTX_RANGE_FN();
    dim3 grid(CEIL_DIV(S_k, 128), S_q);
    streaming_sinks_mask_kernel<<<grid, 128, 0, stream>>>(mask, S_q, S_k, window, n_sink);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Differential attention: scores = softmax(QK1) - lambda * softmax(QK2)
// where QK1 and QK2 come from two heads-of-heads. We provide the
// pointwise blend: given two attention output buffers, blend them.
// ============================================================================

__global__ void differential_blend_kernel(floatX* out, const floatX* a, const floatX* b, float lambda, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = (floatX)((float)a[i] - lambda * (float)b[i]);
}
void differential_attention_blend(floatX* out, const floatX* a, const floatX* b, float lambda, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    differential_blend_kernel<<<CEIL_DIV(N, block_size), block_size, 0, stream>>>(out, a, b, lambda, N);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Cross attention: same as MHA but Q comes from one source and K, V from
// another. We provide a thin score+softmax+V kernel for small sizes; the
// production path should call the standard SDPA with the cross K/V.
//
//   q: [B, H, S_q, D]
//   k, v: [B, H, S_k, D]
//   out: [B, H, S_q, D]
// ============================================================================

__global__ void cross_attention_kernel(floatX* out, const floatX* q, const floatX* k, const floatX* v,
                                       int B, int H, int S_q, int S_k, int D) {
    int qq = blockIdx.x;
    int hh = blockIdx.y;
    int bb = blockIdx.z;
    int d  = threadIdx.x;
    if (d >= D) return;

    // 1) scores = q @ k^T  ->  vector of length S_k
    extern __shared__ float s_scores[];
    if (d == 0) {
        for (int kk = 0; kk < S_k; ++kk) {
            const floatX* qr = q + ((bb * H + hh) * S_q + qq) * D;
            const floatX* kr = k + ((bb * H + hh) * S_k + kk) * D;
            float acc = 0.0f;
            for (int dd = 0; dd < D; ++dd) acc += (float)qr[dd] * (float)kr[dd];
            s_scores[kk] = acc / sqrtf((float)D);
        }
        // 2) softmax over S_k
        float max_v = -INFINITY;
        for (int kk = 0; kk < S_k; ++kk) if (s_scores[kk] > max_v) max_v = s_scores[kk];
        float sumexp = 0.0f;
        for (int kk = 0; kk < S_k; ++kk) { s_scores[kk] = expf(s_scores[kk] - max_v); sumexp += s_scores[kk]; }
        for (int kk = 0; kk < S_k; ++kk) s_scores[kk] /= sumexp;
    }
    __syncthreads();
    // 3) out = scores @ v
    float acc = 0.0f;
    for (int kk = 0; kk < S_k; ++kk) {
        const floatX* vr = v + ((bb * H + hh) * S_k + kk) * D;
        acc += s_scores[kk] * (float)vr[d];
    }
    out[((bb * H + hh) * S_q + qq) * D + d] = (floatX)acc;
}

void cross_attention(floatX* out, const floatX* q, const floatX* k, const floatX* v,
                     int B, int H, int S_q, int S_k, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    dim3 grid(S_q, H, B);
    int shmem = S_k * sizeof(float);
    cross_attention_kernel<<<grid, D, shmem, stream>>>(out, q, k, v, B, H, S_q, S_k, D);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// attentionless_decoder: bucket_indices → embed lookup + add expert_output →
// linear projection to vocab.
//
//   bucket_indices: [B] int32  (uses first column only)
//   expert_output:  [B, residual_dim]   (already pooled by caller)
//   bucket_embed:   [n_buckets, residual_dim]
//   out_weight:     [vocab, residual_dim]
//   out:            [B, 1, vocab]
// ============================================================================

__global__ void attentionless_decoder_kernel(floatX* out, const int* bucket_indices,
                                             const floatX* bucket_embed, const floatX* out_weight,
                                             const floatX* expert_output,
                                             int B, int residual_dim, int n_buckets, int vocab) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (v >= vocab || b >= B) return;
    int bucket = bucket_indices[b] % n_buckets;
    if (bucket < 0) bucket += n_buckets;
    float acc = 0.0f;
    const floatX* be = bucket_embed + bucket * residual_dim;
    const floatX* eo = expert_output + b * residual_dim;
    const floatX* ow = out_weight + v * residual_dim;
    for (int d = 0; d < residual_dim; ++d) {
        float combined = (float)eo[d] + (float)be[d];
        acc += combined * (float)ow[d];
    }
    out[b * vocab + v] = (floatX)acc;
}
void attentionless_decoder_forward(floatX* out, const int* bucket_indices,
                                   const floatX* bucket_embed, const floatX* out_weight,
                                   const floatX* expert_output,
                                   int B, int residual_dim, int n_buckets, int vocab,
                                   cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    dim3 grid(CEIL_DIV(vocab, block_size), B);
    attentionless_decoder_kernel<<<grid, block_size, 0, stream>>>(
        out, bucket_indices, bucket_embed, out_weight, expert_output,
        B, residual_dim, n_buckets, vocab);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Paged attention reference kernel: each query position reads K, V from
// pages indexed by block_table.
//
//   q:           [B, H, 1, D]                            (decode: one query/token)
//   pages_k,v:   [num_pages, Hk, page_size, D]
//   block_table: [B, max_blocks]                          (page ids; -1 unused)
//   cache_len:   [B]                                      (active token count per row)
//   out:         [B, H, D]
//
// This is a correctness reference; production uses vLLM-style block kernels.
// ============================================================================

__global__ void paged_attention_decode_kernel(floatX* out, const floatX* q,
                                              const floatX* pages_k, const floatX* pages_v,
                                              const int* block_table, const int* cache_len,
                                              int H, int Hk, int D, int page_size, int max_blocks) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;
    if (d >= D) return;
    int hk = h % Hk;
    const floatX* qr = q + ((b * H) + h) * D;
    int total = cache_len[b];

    // Compute scores in fp32 via two passes: max then sumexp. Store the scores
    // in shared memory.
    extern __shared__ float s[];
    float* s_score = s;          // length total
    float local_max = -INFINITY;
    for (int t = 0; t < total; ++t) {
        int block_idx = t / page_size;
        int within    = t % page_size;
        int page_id   = block_table[b * max_blocks + block_idx];
        if (page_id < 0) break;
        const floatX* kr = pages_k + ((page_id * Hk) + hk) * page_size * D + within * D;
        float acc = 0.0f;
        if (d == 0) {
            for (int dd = 0; dd < D; ++dd) acc += (float)qr[dd] * (float)kr[dd];
            acc /= sqrtf((float)D);
            s_score[t] = acc;
        }
        __syncthreads();
        if (s_score[t] > local_max) local_max = s_score[t];
    }
    if (d == 0) {
        float sumexp = 0.0f;
        for (int t = 0; t < total; ++t) {
            s_score[t] = expf(s_score[t] - local_max);
            sumexp += s_score[t];
        }
        for (int t = 0; t < total; ++t) s_score[t] /= sumexp;
    }
    __syncthreads();

    float acc = 0.0f;
    for (int t = 0; t < total; ++t) {
        int block_idx = t / page_size;
        int within    = t % page_size;
        int page_id   = block_table[b * max_blocks + block_idx];
        if (page_id < 0) break;
        const floatX* vr = pages_v + ((page_id * Hk) + hk) * page_size * D + within * D;
        acc += s_score[t] * (float)vr[d];
    }
    out[(b * H + h) * D + d] = (floatX)acc;
}

void paged_attention_decode(floatX* out, const floatX* q,
                            const floatX* pages_k, const floatX* pages_v,
                            const int* block_table, const int* cache_len,
                            int B, int H, int Hk, int D, int page_size, int max_blocks,
                            int max_cache_len, cudaStream_t stream) {
    NVTX_RANGE_FN();
    dim3 grid(B, H);
    int shmem = max_cache_len * sizeof(float);
    paged_attention_decode_kernel<<<grid, D, shmem, stream>>>(
        out, q, pages_k, pages_v, block_table, cache_len,
        H, Hk, D, page_size, max_blocks);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Multi-Latent Attention (MLA, DeepSeek-V2 style): compressed-KV attention.
//
// At inference time, the K, V tensors are reconstructed from a low-rank
// compressed representation per token. We provide the K reconstruction step:
//   k_full[t] = k_compressed[t] @ kw_up^T   (per head)
// and similarly for V. Then standard SDPA / paged attn is used.
// ============================================================================

__global__ void mla_expand_kernel(floatX* k_full, const floatX* k_comp, const floatX* w_up,
                                  int N, int comp_dim, int full_dim) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (o >= full_dim || t >= N) return;
    const floatX* kc = k_comp + t * comp_dim;
    const floatX* wo = w_up + o * comp_dim;
    float acc = 0.0f;
    for (int c = 0; c < comp_dim; ++c) acc += (float)kc[c] * (float)wo[c];
    k_full[t * full_dim + o] = (floatX)acc;
}
void mla_expand(floatX* full, const floatX* compressed, const floatX* w_up,
                int N, int comp_dim, int full_dim, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 64;
    dim3 grid(CEIL_DIV(full_dim, block_size), N);
    mla_expand_kernel<<<grid, block_size, 0, stream>>>(full, compressed, w_up, N, comp_dim, full_dim);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Beam search step: given per-beam logits and accumulated logprobs, choose the
// top-K beams across all (beam, vocab) combinations.
//
//   logits:    [batch * beams, vocab]   (already at log-softmax stage)
//   cumlogp:   [batch * beams]          (current beam logprob)
//   out_beam:  [batch, K]               (which beam to extend)
//   out_token: [batch, K]               (which vocab token to emit)
//   out_score: [batch, K]               (new cumulative logprob)
// ============================================================================

__global__ void beam_search_step_kernel(int* out_beam, int* out_token, float* out_score,
                                        const floatX* logits, const float* cumlogp,
                                        int batch, int beams, int vocab, int K) {
    int b = blockIdx.x;
    if (threadIdx.x != 0) return;
    // Single-thread top-K over beams*vocab candidates (slow but correct).
    constexpr int MAX_K = 32;
    if (K > MAX_K) return;
    float top_score[MAX_K]; int top_beam[MAX_K]; int top_token[MAX_K];
    for (int k = 0; k < K; ++k) { top_score[k] = -INFINITY; top_beam[k] = 0; top_token[k] = 0; }
    for (int beam = 0; beam < beams; ++beam) {
        float base = cumlogp[b * beams + beam];
        const floatX* row = logits + (b * beams + beam) * vocab;
        for (int v = 0; v < vocab; ++v) {
            float cand = base + (float)row[v];
            float min_v = top_score[K - 1];
            if (cand > min_v) {
                int pos = K - 1;
                while (pos > 0 && top_score[pos - 1] < cand) {
                    top_score[pos] = top_score[pos - 1];
                    top_beam[pos]  = top_beam[pos - 1];
                    top_token[pos] = top_token[pos - 1];
                    --pos;
                }
                top_score[pos] = cand; top_beam[pos] = beam; top_token[pos] = v;
            }
        }
    }
    for (int k = 0; k < K; ++k) {
        out_beam[b * K + k]  = top_beam[k];
        out_token[b * K + k] = top_token[k];
        out_score[b * K + k] = top_score[k];
    }
}
void beam_search_step(int* out_beam, int* out_token, float* out_score,
                      const floatX* logits, const float* cumlogp,
                      int batch, int beams, int vocab, int K, cudaStream_t stream) {
    NVTX_RANGE_FN();
    beam_search_step_kernel<<<batch, 32, 0, stream>>>(
        out_beam, out_token, out_score, logits, cumlogp, batch, beams, vocab, K);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Chunked prefill scaffolding: nothing CUDA-specific — this is just a chunk
// indexer (per chunk: start_token, length) that the host scheduler uses.
// Provided here so the API surface is consistent.
// ============================================================================

__global__ void chunk_indexer_kernel(int* chunk_starts, int* chunk_lens,
                                     int total_tokens, int chunk_size, int num_chunks) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_chunks) return;
    int start = c * chunk_size;
    int len = min(chunk_size, total_tokens - start);
    chunk_starts[c] = start;
    chunk_lens[c] = len > 0 ? len : 0;
}
void chunked_prefill_indexer(int* chunk_starts, int* chunk_lens,
                             int total_tokens, int chunk_size, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int num_chunks = CEIL_DIV(total_tokens, chunk_size);
    const int block_size = 32;
    chunk_indexer_kernel<<<CEIL_DIV(num_chunks, block_size), block_size, 0, stream>>>(
        chunk_starts, chunk_lens, total_tokens, chunk_size, num_chunks);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Ring attention coordinate: each rank's local Q attends to a sliding window
// of K, V shards. We only ship the local-step kernel (one shard) — the host
// orchestrates the rotation. Re-uses cross_attention semantics with a `mask`
// argument so callers can enforce the causal cone across shards.
// ============================================================================

__global__ void ring_attention_local_kernel(floatX* out_acc, float* lse_acc,
                                            const floatX* q, const floatX* k, const floatX* v,
                                            const floatX* mask, int H, int S_q, int S_k, int D) {
    int qq = blockIdx.x;
    int hh = blockIdx.y;
    int d  = threadIdx.x;
    if (d >= D) return;
    extern __shared__ float s_score[];
    float local_max = -INFINITY;
    if (d == 0) {
        for (int kk = 0; kk < S_k; ++kk) {
            const floatX* qr = q + (hh * S_q + qq) * D;
            const floatX* kr = k + (hh * S_k + kk) * D;
            float acc = 0.0f;
            for (int dd = 0; dd < D; ++dd) acc += (float)qr[dd] * (float)kr[dd];
            acc /= sqrtf((float)D);
            if (mask) acc += (float)mask[qq * S_k + kk];
            s_score[kk] = acc;
            if (acc > local_max) local_max = acc;
        }
        float sumexp = 0.0f;
        for (int kk = 0; kk < S_k; ++kk) {
            s_score[kk] = expf(s_score[kk] - local_max);
            sumexp += s_score[kk];
        }
        lse_acc[hh * S_q + qq] = local_max + logf(sumexp);
    }
    __syncthreads();
    float acc = 0.0f;
    float sumexp = 0.0f;
    for (int kk = 0; kk < S_k; ++kk) {
        const floatX* vr = v + (hh * S_k + kk) * D;
        acc += s_score[kk] * (float)vr[d];
        sumexp += s_score[kk];
    }
    out_acc[(hh * S_q + qq) * D + d] = (floatX)(acc / sumexp);
}

void ring_attention_local_step(floatX* out_acc, float* lse_acc,
                               const floatX* q, const floatX* k, const floatX* v, const floatX* mask,
                               int H, int S_q, int S_k, int D, cudaStream_t stream) {
    NVTX_RANGE_FN();
    dim3 grid(S_q, H);
    int shmem = S_k * sizeof(float);
    ring_attention_local_kernel<<<grid, D, shmem, stream>>>(
        out_acc, lse_acc, q, k, v, mask, H, S_q, S_k, D);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Native Sparse Attention (NSA, DeepSeek-V3.2): masks tokens to a learned
// sparse pattern. We provide a "select top-k keys per query" helper kernel
// that picks the highest-affinity keys for each query in fp32 dot-product
// space; the resulting indices are used by a subsequent gathered attention
// kernel (callable as paged_attention with custom block table).
// ============================================================================

__global__ void nsa_select_topk_keys_kernel(int* top_keys, const floatX* q, const floatX* k,
                                            int H, int S_q, int S_k, int D, int K) {
    int qq = blockIdx.x;
    int hh = blockIdx.y;
    if (threadIdx.x != 0) return;
    const floatX* qr = q + (hh * S_q + qq) * D;
    constexpr int MAX_K = 64;
    float top_score[MAX_K]; int top_idx[MAX_K];
    int kK = K > MAX_K ? MAX_K : K;
    for (int i = 0; i < kK; ++i) { top_score[i] = -INFINITY; top_idx[i] = 0; }
    for (int kk = 0; kk < S_k; ++kk) {
        const floatX* kr = k + (hh * S_k + kk) * D;
        float acc = 0.0f;
        for (int dd = 0; dd < D; ++dd) acc += (float)qr[dd] * (float)kr[dd];
        float min_v = top_score[kK - 1];
        if (acc > min_v) {
            int pos = kK - 1;
            while (pos > 0 && top_score[pos - 1] < acc) {
                top_score[pos] = top_score[pos - 1];
                top_idx[pos]   = top_idx[pos - 1];
                --pos;
            }
            top_score[pos] = acc; top_idx[pos] = kk;
        }
    }
    for (int i = 0; i < kK; ++i) top_keys[(hh * S_q + qq) * K + i] = top_idx[i];
}
void nsa_select_topk_keys(int* top_keys, const floatX* q, const floatX* k,
                          int H, int S_q, int S_k, int D, int K, cudaStream_t stream) {
    NVTX_RANGE_FN();
    dim3 grid(S_q, H);
    nsa_select_topk_keys_kernel<<<grid, 32, 0, stream>>>(top_keys, q, k, H, S_q, S_k, D, K);
    cudaCheck(cudaGetLastError());
}
