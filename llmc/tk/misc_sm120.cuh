/*
misc_sm120.cuh — ThunderKittens kernels for remaining smaller ops on SM120.

  - random_timesteps          per-row uniform RNG
  - mask_scheduler            Bernoulli mask with per-row probability
  - jepa_mask_random          random Bernoulli mask
  - latent_pool               masked mean-pool
  - gae_compute               reverse-sequential GAE
  - kl_penalty                pointwise reward shaping
  - kl_div / mse / bce        loss kernels
  - lsh_bitpack               LSH binarize + bit-pack
  - causal_chunk_state        prefix-mean / chunk-mean pool
  - softmax_distillation      KL(student || teacher)
  - reshape_heads, merge_heads, repeat_kv   shape ops
  - all_to_all_local          local part of MoE all-to-all (the NCCL piece
                              still lives in distributed_ext.cu)
  - tp_column_slice / tp_row_slice  TP shard helpers
*/
#pragma once

#include "tk_common.cuh"
#include <cmath>

namespace llmk::misc_sm120 {

using namespace ::kittens;

__device__ inline uint32_t xs32(uint32_t* s) {
    uint32_t x = *s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; *s = x; return x;
}

// random_timesteps: per-row uniform [0,1)
__global__ void random_timesteps_kernel(float* out, uint32_t seed, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    uint32_t state = seed ^ (uint32_t)(b * 1664525u);
    out[b] = (float)(xs32(&state) >> 8) / 16777216.f;
}
inline void launch_random_timesteps(float* out, uint32_t seed, int B, cudaStream_t stream) {
    const int bs = 64;
    random_timesteps_kernel<<<CEIL_DIV(B, bs), bs, 0, stream>>>(out, seed, B);
    cudaCheck(cudaGetLastError());
}

// mask_scheduler: bernoulli mask with per-row probability
__global__ void mask_scheduler_kernel(int* out_tokens, const int* tokens, const float* timesteps,
                                      int mask_token_id, uint32_t seed, int B, int S) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (s >= S || b >= B) return;
    float p = timesteps[b];
    uint32_t state = seed ^ (uint32_t)(b * 1664525u + s * 1013904223u);
    float u = (float)(xs32(&state) >> 8) / 16777216.f;
    out_tokens[b * S + s] = (u < p) ? mask_token_id : tokens[b * S + s];
}
inline void launch_mask_scheduler(int* out_tokens, const int* tokens, const float* timesteps,
                                  int mask_token_id, uint32_t seed, int B, int S, cudaStream_t stream) {
    const int bs = 128;
    dim3 grid(CEIL_DIV(S, bs), B);
    mask_scheduler_kernel<<<grid, bs, 0, stream>>>(out_tokens, tokens, timesteps, mask_token_id, seed, B, S);
    cudaCheck(cudaGetLastError());
}

// JEPA random mask.
__global__ void jepa_random_mask_kernel(int* masked_tokens, bf16* mask, const int* tokens,
                                        float mask_ratio, int mask_token_id, uint32_t seed,
                                        int B, int S) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (s >= S || b >= B) return;
    uint32_t state = seed ^ (uint32_t)(b * 1664525u + s * 1013904223u);
    float u = (float)(xs32(&state) >> 8) / 16777216.f;
    bool m = u < mask_ratio;
    masked_tokens[b * S + s] = m ? mask_token_id : tokens[b * S + s];
    mask[b * S + s] = __float2bfloat16(m ? 1.f : 0.f);
}
inline void launch_jepa_random_mask(int* masked_tokens, bf16* mask, const int* tokens,
                                    float mask_ratio, int mask_token_id, uint32_t seed,
                                    int B, int S, cudaStream_t stream) {
    const int bs = 128;
    dim3 grid(CEIL_DIV(S, bs), B);
    jepa_random_mask_kernel<<<grid, bs, 0, stream>>>(masked_tokens, mask, tokens, mask_ratio,
                                                     mask_token_id, seed, B, S);
    cudaCheck(cudaGetLastError());
}

// latent_pool.
__global__ void latent_pool_kernel(bf16* out, const bf16* x, const bf16* mask,
                                   int B, int S, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (d >= D) return;
    float weight_sum = 0.f, weighted = 0.f, mean = 0.f;
    for (int s = 0; s < S; ++s) {
        float m = __bfloat162float(mask[b * S + s]);
        float v = __bfloat162float(x[((b * S) + s) * D + d]);
        weight_sum += m;
        weighted   += m * v;
        mean       += v;
    }
    mean = mean / (float)S;
    float pool = (weight_sum > 0.f) ? (weighted / weight_sum) : mean;
    out[b * D + d] = __float2bfloat16(pool);
}
inline void launch_latent_pool(bf16* out, const bf16* x, const bf16* mask,
                               int B, int S, int D, cudaStream_t stream) {
    const int bs = 128;
    dim3 grid(CEIL_DIV(D, bs), B);
    latent_pool_kernel<<<grid, bs, 0, stream>>>(out, x, mask, B, S, D);
    cudaCheck(cudaGetLastError());
}

// GAE (per-batch sequential).
__global__ void gae_kernel(bf16* adv, bf16* ret, const bf16* rewards, const bf16* values,
                           int T, float gamma, float lam) {
    int b = blockIdx.x;
    if (threadIdx.x != 0) return;
    float next_adv = 0.f, next_val = 0.f;
    for (int t = T - 1; t >= 0; --t) {
        float r = __bfloat162float(rewards[b * T + t]);
        float v = __bfloat162float(values [b * T + t]);
        float delta = r + gamma * next_val - v;
        next_adv = delta + gamma * lam * next_adv;
        adv[b * T + t] = __float2bfloat16(next_adv);
        ret[b * T + t] = __float2bfloat16(next_adv + v);
        next_val = v;
    }
}
inline void launch_gae(bf16* adv, bf16* ret, const bf16* rewards, const bf16* values,
                       int B, int T, float gamma, float lam, cudaStream_t stream) {
    gae_kernel<<<B, 1, 0, stream>>>(adv, ret, rewards, values, T, gamma, lam);
    cudaCheck(cudaGetLastError());
}

// KL penalty: rewards − β·(logp_pol − logp_ref)
__global__ void kl_penalty_kernel(bf16* out_rewards, const bf16* logp_pol, const bf16* logp_ref,
                                  const bf16* rewards, float kl_coef, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float kl = __bfloat162float(logp_pol[i]) - __bfloat162float(logp_ref[i]);
    out_rewards[i] = __float2bfloat16(__bfloat162float(rewards[i]) - kl_coef * kl);
}
inline void launch_kl_penalty(bf16* out_rewards, const bf16* logp_pol, const bf16* logp_ref,
                              const bf16* rewards, float kl_coef, int N, cudaStream_t stream) {
    const int bs = 256;
    kl_penalty_kernel<<<CEIL_DIV(N, bs), bs, 0, stream>>>(out_rewards, logp_pol, logp_ref, rewards, kl_coef, N);
    cudaCheck(cudaGetLastError());
}

// Shape ops.
__global__ void reshape_heads_kernel(bf16* out, const bf16* x, int B, int S, int H, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    int bh = blockIdx.z;
    int h = bh % H;
    int b = bh / H;
    if (d >= D) return;
    int src = ((b * S) + s) * (H * D) + h * D + d;
    int dst = (((b * H) + h) * S + s) * D + d;
    out[dst] = x[src];
}
inline void launch_reshape_heads(bf16* out, const bf16* x, int B, int S, int H, int D, cudaStream_t stream) {
    const int bs = 64;
    dim3 grid(CEIL_DIV(D, bs), S, B * H);
    reshape_heads_kernel<<<grid, bs, 0, stream>>>(out, x, B, S, H, D);
    cudaCheck(cudaGetLastError());
}
__global__ void merge_heads_kernel(bf16* out, const bf16* x, int B, int H, int S, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    int bs = blockIdx.z;
    int s = bs % S; int b = bs / S;
    if (d >= D) return;
    int src = (((b * H) + h) * S + s) * D + d;
    int dst = ((b * S) + s) * (H * D) + h * D + d;
    out[dst] = x[src];
}
inline void launch_merge_heads(bf16* out, const bf16* x, int B, int H, int S, int D, cudaStream_t stream) {
    const int bs = 64;
    dim3 grid(CEIL_DIV(D, bs), H, B * S);
    merge_heads_kernel<<<grid, bs, 0, stream>>>(out, x, B, H, S, D);
    cudaCheck(cudaGetLastError());
}
__global__ void repeat_kv_kernel(bf16* out, const bf16* x, int B, int H_kv, int reps, int S, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    int bh = blockIdx.z;
    int Hq = H_kv * reps;
    int h_q = bh % Hq;
    int b = bh / Hq;
    if (d >= D) return;
    int h_kv = h_q / reps;
    int src = (((b * H_kv) + h_kv) * S + s) * D + d;
    int dst = (((b * Hq) + h_q) * S + s) * D + d;
    out[dst] = x[src];
}
inline void launch_repeat_kv(bf16* out, const bf16* x, int B, int H_kv, int reps, int S, int D,
                             cudaStream_t stream) {
    const int bs = 64;
    int Hq = H_kv * reps;
    dim3 grid(CEIL_DIV(D, bs), S, B * Hq);
    repeat_kv_kernel<<<grid, bs, 0, stream>>>(out, x, B, H_kv, reps, S, D);
    cudaCheck(cudaGetLastError());
}

// Causal chunk-state pooling (cumsum or mean per chunk).
__global__ void chunk_state_mean_kernel(bf16* out, const bf16* hidden,
                                        int B, int S, int D, int chunk_size, int num_chunks) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y;
    int b = blockIdx.z;
    if (d >= D) return;
    int start = c * chunk_size;
    int end   = min(start + chunk_size, S);
    int count = end - start;
    if (count <= 0) { out[((b * num_chunks) + c) * D + d] = __float2bfloat16(0.f); return; }
    float acc = 0.f;
    for (int s = start; s < end; ++s) acc += __bfloat162float(hidden[((b * S) + s) * D + d]);
    out[((b * num_chunks) + c) * D + d] = __float2bfloat16(acc / (float)count);
}
inline void launch_chunk_state_mean(bf16* out, const bf16* hidden,
                                    int B, int S, int D, int chunk_size, int num_chunks,
                                    cudaStream_t stream) {
    const int bs = 128;
    dim3 grid(CEIL_DIV(D, bs), num_chunks, B);
    chunk_state_mean_kernel<<<grid, bs, 0, stream>>>(out, hidden, B, S, D, chunk_size, num_chunks);
    cudaCheck(cudaGetLastError());
}

// LSH bit-pack.
__global__ void lsh_bitpack_kernel(int* buckets, const bf16* sem_vec, const bf16* proj,
                                   int rows, int tables, int planes, int dim) {
    int row = blockIdx.x;
    int t   = blockIdx.y;
    if (row >= rows || t >= tables) return;
    int bucket = 0;
    for (int p = threadIdx.x; p < planes; p += blockDim.x) {
        const bf16* pl = proj + (t * planes + p) * dim;
        const bf16* sv = sem_vec + row * dim;
        float acc = 0.f;
        for (int d = 0; d < dim; ++d) acc += __bfloat162float(sv[d]) * __bfloat162float(pl[d]);
        int bit = acc > 0.f ? 1 : 0;
        atomicOr(&bucket, bit << p);
    }
    if (threadIdx.x == 0) buckets[row * tables + t] = bucket;
}
inline void launch_lsh_bitpack(int* buckets, const bf16* sem_vec, const bf16* proj,
                               int rows, int tables, int planes, int dim, cudaStream_t stream) {
    dim3 grid(rows, tables);
    lsh_bitpack_kernel<<<grid, 128, 0, stream>>>(buckets, sem_vec, proj, rows, tables, planes, dim);
    cudaCheck(cudaGetLastError());
}

// Cu_seqlens build + document-causal mask.
__global__ void cu_seqlens_kernel(int* cu_seqlens, const int* seq_lens, int B) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    cu_seqlens[0] = 0;
    for (int i = 0; i < B; ++i) cu_seqlens[i + 1] = cu_seqlens[i] + seq_lens[i];
}
inline void launch_cu_seqlens(int* cu_seqlens, const int* seq_lens, int B, cudaStream_t stream) {
    cu_seqlens_kernel<<<1, 1, 0, stream>>>(cu_seqlens, seq_lens, B);
    cudaCheck(cudaGetLastError());
}

__global__ void doc_causal_mask_kernel(bf16* mask, const int* cu_seqlens, int B, int total_len) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    if (i >= total_len || j >= total_len) return;
    int doc_i = 0, doc_j = 0;
    for (int d = 0; d < B; ++d) { if (i < cu_seqlens[d + 1]) { doc_i = d; break; } }
    for (int d = 0; d < B; ++d) { if (j < cu_seqlens[d + 1]) { doc_j = d; break; } }
    bool ok = (doc_i == doc_j) && (j <= i);
    mask[i * total_len + j] = __float2bfloat16(ok ? 0.f : -1e9f);
}
inline void launch_doc_causal_mask(bf16* mask, const int* cu_seqlens, int B, int total_len,
                                   cudaStream_t stream) {
    const int bs = 64;
    dim3 grid(CEIL_DIV(total_len, bs), total_len);
    doc_causal_mask_kernel<<<grid, bs, 0, stream>>>(mask, cu_seqlens, B, total_len);
    cudaCheck(cudaGetLastError());
}

// TP shard slicers.
__global__ void tp_col_slice_kernel(bf16* W_shard, const bf16* W_full, int out_local, int in_dim, int rank_offset) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (o >= out_local || i >= in_dim) return;
    W_shard[o * in_dim + i] = W_full[(rank_offset + o) * in_dim + i];
}
inline void launch_tp_col_slice(bf16* W_shard, const bf16* W_full, int out_local, int in_dim,
                                int rank_offset, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid(CEIL_DIV(out_local, 16), CEIL_DIV(in_dim, 16));
    tp_col_slice_kernel<<<grid, block, 0, stream>>>(W_shard, W_full, out_local, in_dim, rank_offset);
    cudaCheck(cudaGetLastError());
}
__global__ void tp_row_slice_kernel(bf16* W_shard, const bf16* W_full,
                                    int out_dim, int in_local, int rank_offset, int in_total) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (o >= out_dim || i >= in_local) return;
    W_shard[o * in_local + i] = W_full[o * in_total + (rank_offset + i)];
}
inline void launch_tp_row_slice(bf16* W_shard, const bf16* W_full,
                                int out_dim, int in_local, int rank_offset, int in_total,
                                cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid(CEIL_DIV(out_dim, 16), CEIL_DIV(in_local, 16));
    tp_row_slice_kernel<<<grid, block, 0, stream>>>(W_shard, W_full, out_dim, in_local, rank_offset, in_total);
    cudaCheck(cudaGetLastError());
}

// Gradient accumulate (fp32).
__global__ void grad_accum_kernel(float* dst, const float* src, float scale, size_t N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dst[i] += scale * src[i];
}
inline void launch_grad_accumulate(float* dst, const float* src, float scale, size_t N, cudaStream_t stream) {
    const int bs = 256;
    grad_accum_kernel<<<CEIL_DIV(N, bs), bs, 0, stream>>>(dst, src, scale, N);
    cudaCheck(cudaGetLastError());
}

// Loss-scale dynamic update.
__global__ void check_inf_kernel(int* has_inf, const float* g, size_t N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (!isfinite(g[i])) atomicExch(has_inf, 1);
}
__global__ void loss_scale_step_kernel(float* scale, const int* has_inf, float grow, float backoff) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *scale = (*scale) * (*has_inf ? backoff : grow);
    }
}
inline void launch_loss_scale_dynamic(float* scale, int* has_inf, const float* grads, size_t N,
                                      float grow, float backoff, cudaStream_t stream) {
    cudaCheck(cudaMemsetAsync(has_inf, 0, sizeof(int), stream));
    const int bs = 256;
    int gs = std::min(1024, CEIL_DIV((int)N, bs));
    check_inf_kernel<<<gs, bs, 0, stream>>>(has_inf, grads, N);
    loss_scale_step_kernel<<<1, 1, 0, stream>>>(scale, has_inf, grow, backoff);
    cudaCheck(cudaGetLastError());
}

}  // namespace llmk::misc_sm120
