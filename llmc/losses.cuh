/*
losses.cuh — pointwise / reduction losses beyond the existing fused_classifier.

All losses produce a single scalar in `*out` unless noted. Inputs follow the
NeuralFn `Stage` shapes (see neuralfn/torch_backend.py for reference).

Conventions:
  * floatX inputs are read in fp32 internally; the scalar result is fp32.
  * losses that take per-element targets accept `target_ids` as int32.
  * row reductions use blockReduce<warpReduceSum> (cuda_utils.cuh).
*/
#pragma once

#include <assert.h>
#include <math.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ============================================================================
// masked_token_cross_entropy: CE averaged over positions where loss_mask > 0.
//
//   logits:    [rows, vocab_size]  (floatX)
//   targets:   [rows]              (int32)
//   loss_mask: [rows]              (floatX)
//
// Output: a single fp32 scalar (mean per-token CE over valid positions).
// `ignore_index` rows are excluded (also forced mask=0).
// ============================================================================

__global__ void masked_ce_per_row_kernel(float* per_row_loss, float* per_row_mask,
                                         const floatX* logits, const int* targets,
                                         const floatX* loss_mask, int vocab_size, int rows,
                                         int ignore_index) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int target = targets[row];
    float mask = (float)loss_mask[row];
    if (target == ignore_index || mask <= 0.0f) {
        if (threadIdx.x == 0) {
            per_row_loss[row] = 0.0f;
            per_row_mask[row] = 0.0f;
        }
        return;
    }
    const floatX* row_logits = logits + row * vocab_size;

    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float v = (float)row_logits[i];
        if (v > local_max) local_max = v;
    }
    float row_max = blockReduce<warpReduceMax>(local_max);

    float local_sumexp = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        local_sumexp += expf((float)row_logits[i] - row_max);
    }
    float sumexp = blockReduce<warpReduceSum>(local_sumexp);
    float log_z  = row_max + logf(sumexp);

    if (threadIdx.x == 0) {
        float target_logit = (float)row_logits[target];
        float loss = log_z - target_logit;
        per_row_loss[row] = loss * mask;
        per_row_mask[row] = mask;
    }
}

__global__ void reduce_mean_masked_kernel(float* out, const float* per_row_loss, const float* per_row_mask, int rows) {
    float partial_loss = 0.0f;
    float partial_mask = 0.0f;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        partial_loss += per_row_loss[i];
        partial_mask += per_row_mask[i];
    }
    float loss_sum = blockReduce<warpReduceSum>(partial_loss);
    float mask_sum = blockReduce<warpReduceSum>(partial_mask, true);
    if (threadIdx.x == 0) {
        *out = loss_sum / fmaxf(mask_sum, 1.0f);
    }
}

void masked_token_cross_entropy_forward(float* out_loss,
                                        const floatX* logits, const int* targets,
                                        const floatX* loss_mask, int rows, int vocab_size,
                                        int ignore_index,
                                        float* scratch_per_row_loss,
                                        float* scratch_per_row_mask,
                                        cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    masked_ce_per_row_kernel<<<rows, block_size, 0, stream>>>(
        scratch_per_row_loss, scratch_per_row_mask,
        logits, targets, loss_mask, vocab_size, rows, ignore_index);
    cudaCheck(cudaGetLastError());
    reduce_mean_masked_kernel<<<1, 256, 0, stream>>>(out_loss, scratch_per_row_loss, scratch_per_row_mask, rows);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// latent_mse_loss: F.mse_loss(pred.float(), target.detach().float()).
//
//   pred, target: [N]
//   out: fp32 scalar.
// ============================================================================

__global__ void mse_partial_kernel(float* partials, const floatX* pred, const floatX* target, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        float d = (float)pred[i] - (float)target[i];
        local += d * d;
    }
    float sum = blockReduce<warpReduceSum>(local);
    if (threadIdx.x == 0) atomicAdd(partials, sum);
}

__global__ void mse_finalize_kernel(float* out, const float* sum_sq, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = *sum_sq / (float)N;
    }
}

void latent_mse_loss_forward(float* out, const floatX* pred, const floatX* target, int N,
                             float* scratch_sum_sq, cudaStream_t stream) {
    NVTX_RANGE_FN();
    cudaMemsetAsync(scratch_sum_sq, 0, sizeof(float), stream);
    const int block_size = 256;
    const int grid_size  = std::min(1024, CEIL_DIV(N, block_size));
    mse_partial_kernel<<<grid_size, block_size, 0, stream>>>(scratch_sum_sq, pred, target, N);
    cudaCheck(cudaGetLastError());
    mse_finalize_kernel<<<1, 1, 0, stream>>>(out, scratch_sum_sq, N);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// kl_div(log_student || softmax(teacher)) reduction=batchmean.
//
//   student_logits, teacher_logits: [rows, K]
//   out: fp32 scalar = mean over rows of Σ_k p_k * (logp_t - logp_s)
//
// We compute KL(teacher || student) — i.e. teacher is the target distribution.
// (Matches PyTorch's F.kl_div(student_log, teacher_probs) with reduction
// "batchmean": divides by rows.)
// ============================================================================

__global__ void kl_per_row_kernel(float* per_row, const floatX* student_logits, const floatX* teacher_logits, int K) {
    int row = blockIdx.x;
    const floatX* s = student_logits + row * K;
    const floatX* t = teacher_logits + row * K;

    // Student log-softmax
    float s_max_local = -INFINITY;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = (float)s[i];
        if (v > s_max_local) s_max_local = v;
    }
    float s_max = blockReduce<warpReduceMax>(s_max_local);
    float s_sumexp_local = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        s_sumexp_local += expf((float)s[i] - s_max);
    }
    float s_sumexp = blockReduce<warpReduceSum>(s_sumexp_local, true);
    float s_log_z = s_max + logf(s_sumexp);

    // Teacher softmax + KL accumulator
    float t_max_local = -INFINITY;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = (float)t[i];
        if (v > t_max_local) t_max_local = v;
    }
    float t_max = blockReduce<warpReduceMax>(t_max_local);
    float t_sumexp_local = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        t_sumexp_local += expf((float)t[i] - t_max);
    }
    float t_sumexp = blockReduce<warpReduceSum>(t_sumexp_local, true);
    float inv_z = 1.0f / t_sumexp;

    float local_kl = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float pt = expf((float)t[i] - t_max) * inv_z;
        float log_ps = (float)s[i] - s_log_z;
        float log_pt = (float)t[i] - (t_max + logf(t_sumexp));
        local_kl += pt * (log_pt - log_ps);
    }
    float row_kl = blockReduce<warpReduceSum>(local_kl);
    if (threadIdx.x == 0) per_row[row] = row_kl;
}

__global__ void reduce_mean_kernel(float* out, const float* per_row, int rows) {
    float partial = 0.0f;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) partial += per_row[i];
    float sum = blockReduce<warpReduceSum>(partial);
    if (threadIdx.x == 0) *out = sum / (float)rows;
}

void softmax_distillation_loss_forward(float* out,
                                       const floatX* student_logits, const floatX* teacher_logits,
                                       int rows, int K,
                                       float* scratch_per_row,
                                       cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    kl_per_row_kernel<<<rows, block_size, 0, stream>>>(scratch_per_row, student_logits, teacher_logits, K);
    cudaCheck(cudaGetLastError());
    reduce_mean_kernel<<<1, 256, 0, stream>>>(out, scratch_per_row, rows);
    cudaCheck(cudaGetLastError());
}

// route_distillation_loss is the same KL form between student route logits
// (rows x E) and a teacher distribution baked from topic scores; the prep is
// done host-side, then this kernel is called.
void route_distillation_loss_forward(float* out,
                                     const floatX* student_logits, const floatX* teacher_logits,
                                     int rows, int E,
                                     float* scratch_per_row,
                                     cudaStream_t stream) {
    softmax_distillation_loss_forward(out, student_logits, teacher_logits, rows, E, scratch_per_row, stream);
}

// ============================================================================
// Binary cross-entropy with logits (mean reduction over valid positions).
// Targets are floatX in [0,1].
//
//   logits, targets: [N]
//   valid_mask:      [N] (1.0 to include, 0.0 to drop) — optional, pass nullptr.
//
// Uses the numerically stable form
//   loss = max(x,0) - x*y + log(1 + exp(-|x|))
// ============================================================================

__global__ void bce_with_logits_partial_kernel(float* sum_loss, float* sum_mask,
                                               const floatX* logits, const floatX* targets,
                                               const floatX* valid_mask, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float local_loss = 0.0f;
    float local_mask = 0.0f;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        float x = (float)logits[i];
        float y = (float)targets[i];
        float m = valid_mask ? (float)valid_mask[i] : 1.0f;
        float loss = fmaxf(x, 0.0f) - x * y + log1pf(expf(-fabsf(x)));
        local_loss += m * loss;
        local_mask += m;
    }
    float l_sum = blockReduce<warpReduceSum>(local_loss);
    if (threadIdx.x == 0) atomicAdd(sum_loss, l_sum);
    float m_sum = blockReduce<warpReduceSum>(local_mask, true);
    if (threadIdx.x == 0) atomicAdd(sum_mask, m_sum);
}

__global__ void bce_finalize_kernel(float* out, const float* sum_loss, const float* sum_mask) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = *sum_loss / fmaxf(*sum_mask, 1.0f);
    }
}

void bce_with_logits_loss_forward(float* out, const floatX* logits, const floatX* targets,
                                  const floatX* valid_mask, int N,
                                  float* scratch_sum_loss, float* scratch_sum_mask,
                                  cudaStream_t stream) {
    NVTX_RANGE_FN();
    cudaMemsetAsync(scratch_sum_loss, 0, sizeof(float), stream);
    cudaMemsetAsync(scratch_sum_mask, 0, sizeof(float), stream);
    const int block_size = 256;
    const int grid_size  = std::min(1024, CEIL_DIV(N, block_size));
    bce_with_logits_partial_kernel<<<grid_size, block_size, 0, stream>>>(
        scratch_sum_loss, scratch_sum_mask, logits, targets, valid_mask, N);
    cudaCheck(cudaGetLastError());
    bce_finalize_kernel<<<1, 1, 0, stream>>>(out, scratch_sum_loss, scratch_sum_mask);
    cudaCheck(cudaGetLastError());
}

// route_selection_loss is BCE-with-logits over the semantic-dim slice; host
// constructs the slice + mask and calls bce_with_logits_loss_forward.

// ============================================================================
// preference_bce_loss: -mean(logsigmoid(rc - rr))
// Inputs are batched scalars (rewards) of shape [N].
// ============================================================================

__global__ void preference_bce_partial_kernel(float* sum_loss, const floatX* rc, const floatX* rr, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        float d = (float)rc[i] - (float)rr[i];
        // logsigmoid(d) = -softplus(-d) = -log(1+exp(-d)) numerically stable
        float val = -log1pf(expf(-fabsf(d))) - (d < 0.0f ? -d : 0.0f);
        local += -val;
    }
    float sum = blockReduce<warpReduceSum>(local);
    if (threadIdx.x == 0) atomicAdd(sum_loss, sum);
}
__global__ void scalar_div_kernel(float* out, const float* sum, float denom) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = *sum / denom;
}
void preference_bce_loss_forward(float* out, const floatX* rc, const floatX* rr, int N,
                                 float* scratch_sum, cudaStream_t stream) {
    NVTX_RANGE_FN();
    cudaMemsetAsync(scratch_sum, 0, sizeof(float), stream);
    const int block_size = 256;
    const int grid_size  = std::min(1024, CEIL_DIV(N, block_size));
    preference_bce_partial_kernel<<<grid_size, block_size, 0, stream>>>(scratch_sum, rc, rr, N);
    cudaCheck(cudaGetLastError());
    scalar_div_kernel<<<1, 1, 0, stream>>>(out, scratch_sum, (float)N);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// dpo_pairwise_loss
//   Variants:
//     "sigmoid":  -[(1-ls)·logsig(β·d) + ls·logsig(-β·d)]    (label smoothing)
//     "hinge":    relu(1 - β·d)
//     "ipo":      (β·d - 1/(2β))²
//   d = (logp_pol_chosen - logp_ref_chosen) - (logp_pol_rejected - logp_ref_rejected)
//   inputs are [N] each.
//   out_loss is a scalar; out_chosen_reward / out_rejected_reward are [N].
// ============================================================================

enum class DpoLossType : int { Sigmoid = 0, Hinge = 1, Ipo = 2 };

__global__ void dpo_pairwise_partial_kernel(float* sum_loss,
                                            float* chosen_reward, float* rejected_reward,
                                            const floatX* logp_chosen, const floatX* logp_rejected,
                                            const floatX* ref_chosen, const floatX* ref_rejected,
                                            float beta, float label_smoothing, int loss_type, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        float chosen_logratio   = (float)logp_chosen[i]   - (float)ref_chosen[i];
        float rejected_logratio = (float)logp_rejected[i] - (float)ref_rejected[i];
        float logits = beta * (chosen_logratio - rejected_logratio);
        float per;
        if (loss_type == (int)DpoLossType::Hinge) {
            per = fmaxf(0.0f, 1.0f - logits);
        } else if (loss_type == (int)DpoLossType::Ipo) {
            float target = 1.0f / (2.0f * fmaxf(beta, 1e-8f));
            per = (logits - target) * (logits - target);
        } else { // Sigmoid
            float pos = -log1pf(expf(-fabsf( logits))) - (logits < 0.0f ? -logits : 0.0f);
            float neg = -log1pf(expf(-fabsf(-logits))) - ((-logits) < 0.0f ?  logits : 0.0f);
            per = -((1.0f - label_smoothing) * pos + label_smoothing * neg);
        }
        local += per;
        // Detached reward outputs (per-row).
        chosen_reward[i]   = beta * chosen_logratio;
        rejected_reward[i] = beta * rejected_logratio;
    }
    float sum = blockReduce<warpReduceSum>(local);
    if (threadIdx.x == 0) atomicAdd(sum_loss, sum);
}
void dpo_pairwise_loss_forward(float* out_loss, float* chosen_reward, float* rejected_reward,
                               const floatX* logp_chosen, const floatX* logp_rejected,
                               const floatX* ref_chosen, const floatX* ref_rejected,
                               int N, float beta, float label_smoothing, int loss_type,
                               float* scratch_sum, cudaStream_t stream) {
    NVTX_RANGE_FN();
    cudaMemsetAsync(scratch_sum, 0, sizeof(float), stream);
    const int block_size = 256;
    const int grid_size  = std::min(1024, CEIL_DIV(N, block_size));
    dpo_pairwise_partial_kernel<<<grid_size, block_size, 0, stream>>>(
        scratch_sum, chosen_reward, rejected_reward,
        logp_chosen, logp_rejected, ref_chosen, ref_rejected,
        beta, label_smoothing, loss_type, N);
    cudaCheck(cudaGetLastError());
    scalar_div_kernel<<<1, 1, 0, stream>>>(out_loss, scratch_sum, (float)N);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// ppo_clipped_loss: clipped policy + clipped value loss; returns
// (policy_loss, value_loss, total_loss).
//
// All buffers are length N (per-token across rollout).
// ============================================================================

__global__ void ppo_clipped_partial_kernel(float* sum_pol, float* sum_val,
                                           const floatX* logp_new, const floatX* logp_old,
                                           const floatX* advantages,
                                           const floatX* value_new, const floatX* value_old,
                                           const floatX* returns,
                                           float clip_range, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float local_pol = 0.0f;
    float local_val = 0.0f;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        float ratio = expf((float)logp_new[i] - (float)logp_old[i]);
        float adv   = (float)advantages[i];
        float unclipped = ratio * adv;
        float clipped   = fminf(fmaxf(ratio, 1.0f - clip_range), 1.0f + clip_range) * adv;
        float policy = -fminf(unclipped, clipped);
        local_pol += policy;

        float vnew = (float)value_new[i];
        float vold = (float)value_old[i];
        float ret  = (float)returns[i];
        float vclipped = vold + fminf(fmaxf(vnew - vold, -clip_range), clip_range);
        float a = (vnew - ret); a = a * a;
        float b = (vclipped - ret); b = b * b;
        local_val += 0.5f * fmaxf(a, b);
    }
    float pol_sum = blockReduce<warpReduceSum>(local_pol);
    if (threadIdx.x == 0) atomicAdd(sum_pol, pol_sum);
    float val_sum = blockReduce<warpReduceSum>(local_val, true);
    if (threadIdx.x == 0) atomicAdd(sum_val, val_sum);
}

__global__ void ppo_finalize_kernel(float* policy_loss, float* value_loss, float* total_loss,
                                    const float* sum_pol, const float* sum_val,
                                    float vf_coef, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float pl = *sum_pol / (float)N;
        float vl = *sum_val / (float)N;
        *policy_loss = pl;
        *value_loss  = vl;
        *total_loss  = pl + vf_coef * vl;
    }
}

void ppo_clipped_loss_forward(float* policy_loss, float* value_loss, float* total_loss,
                              const floatX* logp_new, const floatX* logp_old,
                              const floatX* advantages,
                              const floatX* value_new, const floatX* value_old,
                              const floatX* returns,
                              int N, float clip_range, float vf_coef,
                              float* scratch_pol, float* scratch_val,
                              cudaStream_t stream) {
    NVTX_RANGE_FN();
    cudaMemsetAsync(scratch_pol, 0, sizeof(float), stream);
    cudaMemsetAsync(scratch_val, 0, sizeof(float), stream);
    const int block_size = 256;
    const int grid_size  = std::min(1024, CEIL_DIV(N, block_size));
    ppo_clipped_partial_kernel<<<grid_size, block_size, 0, stream>>>(
        scratch_pol, scratch_val,
        logp_new, logp_old, advantages, value_new, value_old, returns,
        clip_range, N);
    cudaCheck(cudaGetLastError());
    ppo_finalize_kernel<<<1, 1, 0, stream>>>(policy_loss, value_loss, total_loss, scratch_pol, scratch_val, vf_coef, N);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// kl_penalty: rewards - kl_coef * (logp_policy - logp_ref)   (per-token, in-place ok)
// ============================================================================

__global__ void kl_penalty_kernel(floatX* out_rewards, const floatX* logp_pol, const floatX* logp_ref,
                                  const floatX* rewards, float kl_coef, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float kl = (float)logp_pol[idx] - (float)logp_ref[idx];
    out_rewards[idx] = (floatX)((float)rewards[idx] - kl_coef * kl);
}
void kl_penalty_forward(floatX* out_rewards, const floatX* logp_pol, const floatX* logp_ref,
                        const floatX* rewards, float kl_coef, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    const int grid_size  = CEIL_DIV(N, block_size);
    kl_penalty_kernel<<<grid_size, block_size, 0, stream>>>(
        out_rewards, logp_pol, logp_ref, rewards, kl_coef, N);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// sequence_logp: per-sequence sum of token logprobs, masked.
//   logits:  [batch, seq, vocab]
//   targets: [batch, seq]
//   mask:    [batch, seq]
//   out:     [batch]  (fp32)
//
// Excludes target == ignore_index. Computes log_softmax over vocab inline.
// ============================================================================

__global__ void sequence_logp_kernel(float* out, const floatX* logits, const int* targets,
                                     const floatX* mask, int seq, int vocab, int ignore_index) {
    int b   = blockIdx.x;
    const floatX* row_logits = logits + b * seq * vocab;
    const int*    row_targets = targets + b * seq;
    const floatX* row_mask    = mask + b * seq;

    float partial = 0.0f;
    for (int t = threadIdx.x; t < seq; t += blockDim.x) {
        int target = row_targets[t];
        float m = (float)row_mask[t];
        if (target == ignore_index || m <= 0.0f) continue;
        const floatX* tok_logits = row_logits + t * vocab;
        // log_softmax compute on the per-token row.
        float local_max = -INFINITY;
        for (int v = 0; v < vocab; ++v) {
            float val = (float)tok_logits[v];
            if (val > local_max) local_max = val;
        }
        float sumexp = 0.0f;
        for (int v = 0; v < vocab; ++v) {
            sumexp += expf((float)tok_logits[v] - local_max);
        }
        float log_z = local_max + logf(sumexp);
        float log_p = (float)tok_logits[target] - log_z;
        partial += m * log_p;
    }
    float sum = blockReduce<warpReduceSum>(partial);
    if (threadIdx.x == 0) out[b] = sum;
}

void sequence_logp_forward(float* out, const floatX* logits, const int* targets,
                           const floatX* mask, int batch, int seq, int vocab,
                           int ignore_index, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128; // per-token vocab loop is serial, keep block small
    sequence_logp_kernel<<<batch, block_size, 0, stream>>>(out, logits, targets, mask, seq, vocab, ignore_index);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// load_balance_loss / route_balance_loss: E * Σ density² where density is
// mean over rows of softmax(router_logits).
//
// router_logits: [rows, E].
// Output: scalar.
// ============================================================================

__global__ void route_density_partial_kernel(float* density, const floatX* router_logits, int rows, int E) {
    int e = blockIdx.x;
    if (e >= E) return;
    float partial = 0.0f;
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
        const floatX* row = router_logits + r * E;
        float row_max = -INFINITY;
        for (int j = 0; j < E; ++j) {
            float v = (float)row[j];
            if (v > row_max) row_max = v;
        }
        float sumexp = 0.0f;
        for (int j = 0; j < E; ++j) {
            sumexp += expf((float)row[j] - row_max);
        }
        partial += expf((float)row[e] - row_max) / sumexp;
    }
    float sum = blockReduce<warpReduceSum>(partial);
    if (threadIdx.x == 0) density[e] = sum / (float)rows;
}

__global__ void route_balance_finalize_kernel(float* out, const float* density, int E) {
    float partial = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        partial += density[i] * density[i];
    }
    float sum = blockReduce<warpReduceSum>(partial);
    if (threadIdx.x == 0) *out = (float)E * sum;
}

void load_balance_loss_forward(float* out_loss, const floatX* router_logits, int rows, int E,
                               float* scratch_density, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    route_density_partial_kernel<<<E, block_size, 0, stream>>>(scratch_density, router_logits, rows, E);
    cudaCheck(cudaGetLastError());
    route_balance_finalize_kernel<<<1, 128, 0, stream>>>(out_loss, scratch_density, E);
    cudaCheck(cudaGetLastError());
}
// route_balance_loss is the same form over `route_logits`.
void route_balance_loss_forward(float* out_loss, const floatX* route_logits, int rows, int E,
                                float* scratch_density, cudaStream_t stream) {
    load_balance_loss_forward(out_loss, route_logits, rows, E, scratch_density, stream);
}

// ============================================================================
// semantic_alignment_loss: masked categorical CE summed over vocab dims
// (variable per-dim term counts). Host prepares contiguous logits and targets
// per dim and calls masked_token_cross_entropy_forward per dim. We provide a
// per-dim helper here for completeness.
// ============================================================================

void semantic_alignment_loss_per_dim_forward(float* out_loss,
                                             const floatX* logits, const int* targets,
                                             const floatX* loss_mask, int rows, int term_count,
                                             int ignore_index,
                                             float* scratch_per_row_loss,
                                             float* scratch_per_row_mask,
                                             cudaStream_t stream) {
    masked_token_cross_entropy_forward(out_loss, logits, targets, loss_mask,
                                       rows, term_count, ignore_index,
                                       scratch_per_row_loss, scratch_per_row_mask, stream);
}
