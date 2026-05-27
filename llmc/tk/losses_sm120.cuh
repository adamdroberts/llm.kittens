/*
losses_sm120.cuh — ThunderKittens loss kernels for SM120.

  - masked_token_cross_entropy
  - latent_mse_loss
  - softmax_distillation_loss (KL)
  - bce_with_logits_loss
  - preference_bce_loss
  - dpo_pairwise_loss
  - ppo_clipped_loss
  - sequence_logp
  - load_balance_loss / route_balance_loss

All produce a single fp32 scalar in `out` unless otherwise noted. Targets are
int32 where natural.
*/
#pragma once

#include "tk_common.cuh"
#include <cmath>

namespace llmk::losses_sm120 {

using namespace ::kittens;

// masked_token_cross_entropy.
__global__ void masked_ce_per_row_kernel(float* per_row_loss, float* per_row_mask,
                                         const bf16* logits, const int* targets, const bf16* loss_mask,
                                         int vocab, int rows, int ignore_index) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int target = targets[row];
    float mask = __bfloat162float(loss_mask[row]);
    if (target == ignore_index || mask <= 0.f) {
        if (threadIdx.x == 0) { per_row_loss[row] = 0.f; per_row_mask[row] = 0.f; }
        return;
    }
    const bf16* row_logits = logits + row * vocab;
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float v = __bfloat162float(row_logits[i]);
        if (v > local_max) local_max = v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        float o = __shfl_xor_sync(0xFFFFFFFF, local_max, off);
        if (o > local_max) local_max = o;
    }
    float row_max = __shfl_sync(0xFFFFFFFF, local_max, 0);
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) local_sum += expf(__bfloat162float(row_logits[i]) - row_max);
    for (int off = 16; off > 0; off >>= 1) local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, off);
    float sumexp = __shfl_sync(0xFFFFFFFF, local_sum, 0);
    float log_z = row_max + logf(sumexp);
    if (threadIdx.x == 0) {
        float target_logit = __bfloat162float(row_logits[target]);
        per_row_loss[row] = (log_z - target_logit) * mask;
        per_row_mask[row] = mask;
    }
}
__global__ void reduce_mean_masked_kernel(float* out, const float* per_row_loss, const float* per_row_mask, int rows) {
    float pl = 0.f, pm = 0.f;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        pl += per_row_loss[i]; pm += per_row_mask[i];
    }
    for (int off = 16; off > 0; off >>= 1) {
        pl += __shfl_xor_sync(0xFFFFFFFF, pl, off);
        pm += __shfl_xor_sync(0xFFFFFFFF, pm, off);
    }
    if (threadIdx.x == 0) *out = pl / fmaxf(pm, 1.f);
}
inline void launch_masked_ce(float* out, const bf16* logits, const int* targets, const bf16* loss_mask,
                             int rows, int vocab, int ignore_index,
                             float* scratch_pl, float* scratch_pm, cudaStream_t stream) {
    masked_ce_per_row_kernel<<<rows, ::kittens::WARP_THREADS, 0, stream>>>(
        scratch_pl, scratch_pm, logits, targets, loss_mask, vocab, rows, ignore_index);
    reduce_mean_masked_kernel<<<1, ::kittens::WARP_THREADS, 0, stream>>>(out, scratch_pl, scratch_pm, rows);
    cudaCheck(cudaGetLastError());
}

// MSE.
__global__ void mse_kernel(float* sum_sq, const bf16* pred, const bf16* target, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float local = 0.f;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        float d = __bfloat162float(pred[i]) - __bfloat162float(target[i]);
        local += d * d;
    }
    for (int off = 16; off > 0; off >>= 1) local += __shfl_xor_sync(0xFFFFFFFF, local, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(sum_sq, local);
}
__global__ void mse_finalize_kernel(float* out, const float* sum_sq, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = *sum_sq / (float)N;
}
inline void launch_mse(float* out, const bf16* pred, const bf16* target, int N,
                       float* scratch_sum, cudaStream_t stream) {
    cudaCheck(cudaMemsetAsync(scratch_sum, 0, sizeof(float), stream));
    const int bs = 256;
    int gs = std::min(1024, CEIL_DIV(N, bs));
    mse_kernel<<<gs, bs, 0, stream>>>(scratch_sum, pred, target, N);
    mse_finalize_kernel<<<1, 1, 0, stream>>>(out, scratch_sum, N);
    cudaCheck(cudaGetLastError());
}

// Softmax distillation KL.
__global__ void kl_per_row_kernel(float* per_row, const bf16* student, const bf16* teacher, int K) {
    int row = blockIdx.x;
    const bf16* s = student + row * K;
    const bf16* t = teacher + row * K;
    float s_max = -INFINITY;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = __bfloat162float(s[i]);
        if (v > s_max) s_max = v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        float o = __shfl_xor_sync(0xFFFFFFFF, s_max, off);
        if (o > s_max) s_max = o;
    }
    s_max = __shfl_sync(0xFFFFFFFF, s_max, 0);
    float s_sum = 0.f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) s_sum += expf(__bfloat162float(s[i]) - s_max);
    for (int off = 16; off > 0; off >>= 1) s_sum += __shfl_xor_sync(0xFFFFFFFF, s_sum, off);
    s_sum = __shfl_sync(0xFFFFFFFF, s_sum, 0);
    float s_log_z = s_max + logf(s_sum);

    float t_max = -INFINITY;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = __bfloat162float(t[i]);
        if (v > t_max) t_max = v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        float o = __shfl_xor_sync(0xFFFFFFFF, t_max, off);
        if (o > t_max) t_max = o;
    }
    t_max = __shfl_sync(0xFFFFFFFF, t_max, 0);
    float t_sum = 0.f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) t_sum += expf(__bfloat162float(t[i]) - t_max);
    for (int off = 16; off > 0; off >>= 1) t_sum += __shfl_xor_sync(0xFFFFFFFF, t_sum, off);
    t_sum = __shfl_sync(0xFFFFFFFF, t_sum, 0);
    float inv_t_z = 1.f / t_sum;

    float local_kl = 0.f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float pt = expf(__bfloat162float(t[i]) - t_max) * inv_t_z;
        float log_ps = __bfloat162float(s[i]) - s_log_z;
        float log_pt = __bfloat162float(t[i]) - (t_max + logf(t_sum));
        local_kl += pt * (log_pt - log_ps);
    }
    for (int off = 16; off > 0; off >>= 1) local_kl += __shfl_xor_sync(0xFFFFFFFF, local_kl, off);
    if (threadIdx.x == 0) per_row[row] = local_kl;
}
__global__ void mean_kernel(float* out, const float* per_row, int rows) {
    float partial = 0.f;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) partial += per_row[i];
    for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xFFFFFFFF, partial, off);
    if (threadIdx.x == 0) *out = partial / (float)rows;
}
inline void launch_softmax_distill_kl(float* out, const bf16* student, const bf16* teacher,
                                      int rows, int K, float* scratch, cudaStream_t stream) {
    kl_per_row_kernel<<<rows, ::kittens::WARP_THREADS, 0, stream>>>(scratch, student, teacher, K);
    mean_kernel<<<1, ::kittens::WARP_THREADS, 0, stream>>>(out, scratch, rows);
    cudaCheck(cudaGetLastError());
}

// BCE-with-logits (mean over valid mask).
__global__ void bce_partial_kernel(float* sl, float* sm, const bf16* logits, const bf16* targets,
                                   const bf16* valid_mask, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ll = 0.f, lm = 0.f;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        float x = __bfloat162float(logits[i]);
        float y = __bfloat162float(targets[i]);
        float m = valid_mask ? __bfloat162float(valid_mask[i]) : 1.f;
        float loss = fmaxf(x, 0.f) - x * y + log1pf(expf(-fabsf(x)));
        ll += m * loss; lm += m;
    }
    for (int off = 16; off > 0; off >>= 1) { ll += __shfl_xor_sync(0xFFFFFFFF, ll, off); lm += __shfl_xor_sync(0xFFFFFFFF, lm, off); }
    if ((threadIdx.x & 31) == 0) { atomicAdd(sl, ll); atomicAdd(sm, lm); }
}
__global__ void bce_finalize_kernel(float* out, const float* sl, const float* sm) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = *sl / fmaxf(*sm, 1.f);
}
inline void launch_bce_with_logits(float* out, const bf16* logits, const bf16* targets,
                                   const bf16* valid_mask, int N,
                                   float* scratch_sl, float* scratch_sm, cudaStream_t stream) {
    cudaCheck(cudaMemsetAsync(scratch_sl, 0, sizeof(float), stream));
    cudaCheck(cudaMemsetAsync(scratch_sm, 0, sizeof(float), stream));
    const int bs = 256;
    int gs = std::min(1024, CEIL_DIV(N, bs));
    bce_partial_kernel<<<gs, bs, 0, stream>>>(scratch_sl, scratch_sm, logits, targets, valid_mask, N);
    bce_finalize_kernel<<<1, 1, 0, stream>>>(out, scratch_sl, scratch_sm);
    cudaCheck(cudaGetLastError());
}

// Preference BCE.
__global__ void pref_bce_partial_kernel(float* sl, const bf16* rc, const bf16* rr, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float local = 0.f;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        float d = __bfloat162float(rc[i]) - __bfloat162float(rr[i]);
        float val = -log1pf(expf(-fabsf(d))) - (d < 0.f ? -d : 0.f);
        local += -val;
    }
    for (int off = 16; off > 0; off >>= 1) local += __shfl_xor_sync(0xFFFFFFFF, local, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(sl, local);
}
__global__ void div_by_kernel(float* out, const float* sum, float denom) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = *sum / denom;
}
inline void launch_preference_bce(float* out, const bf16* rc, const bf16* rr, int N,
                                  float* scratch_sum, cudaStream_t stream) {
    cudaCheck(cudaMemsetAsync(scratch_sum, 0, sizeof(float), stream));
    const int bs = 256;
    int gs = std::min(1024, CEIL_DIV(N, bs));
    pref_bce_partial_kernel<<<gs, bs, 0, stream>>>(scratch_sum, rc, rr, N);
    div_by_kernel<<<1, 1, 0, stream>>>(out, scratch_sum, (float)N);
    cudaCheck(cudaGetLastError());
}

// DPO pairwise (sigmoid / hinge / IPO).
__global__ void dpo_partial_kernel(float* sl, float* cr, float* rr_,
                                   const bf16* lp_c, const bf16* lp_r,
                                   const bf16* rf_c, const bf16* rf_r,
                                   float beta, float ls, int loss_type, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float local = 0.f;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        float c_lr = __bfloat162float(lp_c[i]) - __bfloat162float(rf_c[i]);
        float r_lr = __bfloat162float(lp_r[i]) - __bfloat162float(rf_r[i]);
        float logits = beta * (c_lr - r_lr);
        float per;
        if (loss_type == 1) {
            per = fmaxf(0.f, 1.f - logits);
        } else if (loss_type == 2) {
            float target = 1.f / (2.f * fmaxf(beta, 1e-8f));
            per = (logits - target) * (logits - target);
        } else {
            float pos = -log1pf(expf(-fabsf( logits))) - (logits < 0.f ? -logits : 0.f);
            float neg = -log1pf(expf(-fabsf(-logits))) - ((-logits) < 0.f ?  logits : 0.f);
            per = -((1.f - ls) * pos + ls * neg);
        }
        local += per;
        cr[i]  = beta * c_lr;
        rr_[i] = beta * r_lr;
    }
    for (int off = 16; off > 0; off >>= 1) local += __shfl_xor_sync(0xFFFFFFFF, local, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(sl, local);
}
inline void launch_dpo(float* out, float* chosen_reward, float* rejected_reward,
                       const bf16* lp_c, const bf16* lp_r, const bf16* rf_c, const bf16* rf_r,
                       int N, float beta, float label_smoothing, int loss_type,
                       float* scratch_sum, cudaStream_t stream) {
    cudaCheck(cudaMemsetAsync(scratch_sum, 0, sizeof(float), stream));
    const int bs = 256;
    int gs = std::min(1024, CEIL_DIV(N, bs));
    dpo_partial_kernel<<<gs, bs, 0, stream>>>(scratch_sum, chosen_reward, rejected_reward,
                                              lp_c, lp_r, rf_c, rf_r, beta, label_smoothing, loss_type, N);
    div_by_kernel<<<1, 1, 0, stream>>>(out, scratch_sum, (float)N);
    cudaCheck(cudaGetLastError());
}

// PPO clipped.
__global__ void ppo_partial_kernel(float* sp, float* sv,
                                   const bf16* lp_new, const bf16* lp_old, const bf16* adv,
                                   const bf16* v_new, const bf16* v_old, const bf16* ret,
                                   float clip, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float lp = 0.f, lv = 0.f;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        float ratio = expf(__bfloat162float(lp_new[i]) - __bfloat162float(lp_old[i]));
        float a = __bfloat162float(adv[i]);
        float unc = ratio * a;
        float clp = fminf(fmaxf(ratio, 1.f - clip), 1.f + clip) * a;
        lp += -fminf(unc, clp);

        float vn = __bfloat162float(v_new[i]);
        float vo = __bfloat162float(v_old[i]);
        float rt = __bfloat162float(ret[i]);
        float vc = vo + fminf(fmaxf(vn - vo, -clip), clip);
        float d1 = (vn - rt); d1 *= d1;
        float d2 = (vc - rt); d2 *= d2;
        lv += 0.5f * fmaxf(d1, d2);
    }
    for (int off = 16; off > 0; off >>= 1) { lp += __shfl_xor_sync(0xFFFFFFFF, lp, off); lv += __shfl_xor_sync(0xFFFFFFFF, lv, off); }
    if ((threadIdx.x & 31) == 0) { atomicAdd(sp, lp); atomicAdd(sv, lv); }
}
__global__ void ppo_finalize_kernel(float* pl, float* vl, float* tot,
                                    const float* sp, const float* sv, float vf_coef, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float p = *sp / (float)N;
        float v = *sv / (float)N;
        *pl = p; *vl = v; *tot = p + vf_coef * v;
    }
}
inline void launch_ppo(float* pl, float* vl, float* tot,
                       const bf16* lp_new, const bf16* lp_old, const bf16* adv,
                       const bf16* v_new, const bf16* v_old, const bf16* ret,
                       int N, float clip, float vf_coef,
                       float* sp, float* sv, cudaStream_t stream) {
    cudaCheck(cudaMemsetAsync(sp, 0, sizeof(float), stream));
    cudaCheck(cudaMemsetAsync(sv, 0, sizeof(float), stream));
    const int bs = 256;
    int gs = std::min(1024, CEIL_DIV(N, bs));
    ppo_partial_kernel<<<gs, bs, 0, stream>>>(sp, sv, lp_new, lp_old, adv, v_new, v_old, ret, clip, N);
    ppo_finalize_kernel<<<1, 1, 0, stream>>>(pl, vl, tot, sp, sv, vf_coef, N);
    cudaCheck(cudaGetLastError());
}

// load_balance_loss / route_balance_loss.
__global__ void density_kernel(float* density, const bf16* router_logits, int rows, int E) {
    int e = blockIdx.x;
    if (e >= E) return;
    float partial = 0.f;
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
        const bf16* row = router_logits + r * E;
        float row_max = -INFINITY;
        for (int j = 0; j < E; ++j) { float v = __bfloat162float(row[j]); if (v > row_max) row_max = v; }
        float sumexp = 0.f;
        for (int j = 0; j < E; ++j) sumexp += expf(__bfloat162float(row[j]) - row_max);
        partial += expf(__bfloat162float(row[e]) - row_max) / sumexp;
    }
    for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xFFFFFFFF, partial, off);
    if (threadIdx.x == 0) density[e] = partial / (float)rows;
}
__global__ void balance_finalize_kernel(float* out, const float* density, int E) {
    float partial = 0.f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) partial += density[i] * density[i];
    for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xFFFFFFFF, partial, off);
    if (threadIdx.x == 0) *out = (float)E * partial;
}
inline void launch_load_balance(float* out, const bf16* router_logits, int rows, int E,
                                float* scratch_density, cudaStream_t stream) {
    density_kernel<<<E, ::kittens::WARP_THREADS, 0, stream>>>(scratch_density, router_logits, rows, E);
    balance_finalize_kernel<<<1, ::kittens::WARP_THREADS, 0, stream>>>(out, scratch_density, E);
    cudaCheck(cudaGetLastError());
}

// sequence_logp.
__global__ void seq_logp_kernel(float* out, const bf16* logits, const int* targets, const bf16* mask,
                                int seq, int vocab, int ignore_index) {
    int b = blockIdx.x;
    const bf16* row_logits = logits + b * seq * vocab;
    const int* row_targets = targets + b * seq;
    const bf16* row_mask = mask + b * seq;
    float partial = 0.f;
    for (int t = threadIdx.x; t < seq; t += blockDim.x) {
        int target = row_targets[t];
        float m = __bfloat162float(row_mask[t]);
        if (target == ignore_index || m <= 0.f) continue;
        const bf16* toks = row_logits + t * vocab;
        float local_max = -INFINITY;
        for (int v = 0; v < vocab; ++v) {
            float val = __bfloat162float(toks[v]);
            if (val > local_max) local_max = val;
        }
        float sumexp = 0.f;
        for (int v = 0; v < vocab; ++v) sumexp += expf(__bfloat162float(toks[v]) - local_max);
        float log_z = local_max + logf(sumexp);
        partial += m * (__bfloat162float(toks[target]) - log_z);
    }
    for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xFFFFFFFF, partial, off);
    if (threadIdx.x == 0) out[b] = partial;
}
inline void launch_sequence_logp(float* out, const bf16* logits, const int* targets, const bf16* mask,
                                 int batch, int seq, int vocab, int ignore_index, cudaStream_t stream) {
    seq_logp_kernel<<<batch, ::kittens::WARP_THREADS, 0, stream>>>(out, logits, targets, mask, seq, vocab, ignore_index);
    cudaCheck(cudaGetLastError());
}

}  // namespace llmk::losses_sm120
