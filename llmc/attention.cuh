/*
attention.cuh — GPT-style causal attention wrapper.

Keeps llm.c's C-style attention_forward signature while routing supported
shapes through ThunderKittens H100 MHA. The QKV permute/unpermute kernels are
ported verbatim from llm.c/llmc/attention.cuh (lines 14-83).
*/
#pragma once

#include <assert.h>
#include <float.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "tk/attention_h100.cuh"

// ----------------------------------------------------------------------------
// QKV layout glue — ported verbatim from llm.c/llmc/attention.cuh.

__global__ void permute_kernel(floatX* q, floatX* k, floatX* v,
                               const floatX* inp,
                               int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
    q[idx] = __ldcs(&inp[inp_idx]);
    k[idx] = __ldcs(&inp[inp_idx + NH * d]);
    v[idx] = __ldcs(&inp[inp_idx + 2 * (NH * d)]);
}

__global__ void permute_kernel_backward(floatX* dinp,
                                        const floatX* dq, const floatX* dk, const floatX* dv,
                                        int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;

    int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
    dinp[inp_idx] = dq[idx];
    dinp[inp_idx + NH * d] = dk[idx];
    dinp[inp_idx + 2 * (NH * d)] = dv[idx];
}

__global__ void unpermute_kernel(floatX* inp, floatX *out, int B, int N, int NH, int d) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
    out[other_idx] = __ldcs(&inp[idx]);
}

__global__ void unpermute_kernel_backward(floatX* dinp, const floatX *dout, int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
    dinp[idx] = (floatX)dout[other_idx];
}

// ----------------------------------------------------------------------------
// Padding glue for sequence lengths not covered by TK's 192-token forward tile.
// The caller-provided scratch buffer must hold q/k/v/o at Tpad plus the float
// LSE vector; train_gpt2.cu sizes acts.output accordingly.

__global__ void pad_qkv_kernel(floatX* q_pad, floatX* k_pad, floatX* v_pad,
                               const floatX* q, const floatX* k, const floatX* v,
                               int B, int T, int Tpad, int NH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * Tpad * HS;
    if (idx >= total) return;

    int hs = idx % HS;
    int t = (idx / HS) % Tpad;
    int nh = (idx / (HS * Tpad)) % NH;
    int b = idx / (NH * Tpad * HS);

    if (t < T) {
        int src_idx = ((b * NH + nh) * T + t) * HS + hs;
        q_pad[idx] = q[src_idx];
        k_pad[idx] = k[src_idx];
        v_pad[idx] = v[src_idx];
    } else {
        q_pad[idx] = (floatX)0.0f;
        k_pad[idx] = (floatX)0.0f;
        v_pad[idx] = (floatX)0.0f;
    }
}

__global__ void unpermute_padded_kernel(const floatX* inp, floatX* out,
                                        int B, int T, int Tpad, int NH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * T * HS) return;

    int hs = idx % HS;
    int t = (idx / HS) % T;
    int nh = (idx / (HS * T)) % NH;
    int b = idx / (NH * T * HS);

    int src_idx = ((b * NH + nh) * Tpad + t) * HS + hs;
    int dst_idx = (b * T + t) * NH * HS + nh * HS + hs;
    out[dst_idx] = inp[src_idx];
}

__global__ void copy_lse_unpad_kernel(float* dst, const float* src,
                                      int B, int T, int Tpad, int NH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * T;
    if (idx >= total) return;

    int t = idx % T;
    int nh = (idx / T) % NH;
    int b = idx / (NH * T);
    dst[(b * NH + nh) * T + t] = src[(b * NH + nh) * Tpad + t];
}

__global__ void attention_float_grads_to_bf16_kernel(floatX* dq, floatX* dk, floatX* dv,
                                                     const float* qg, const float* kg, const float* vg,
                                                     int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    dq[idx] = (floatX)qg[idx];
    dk[idx] = (floatX)kg[idx];
    dv[idx] = (floatX)vg[idx];
}

inline void attention_forward(floatX* out, floatX* qkvr, floatX* att,
                              floatX* inp,
                              int B, int T, int C, int NH, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    const int HS = C / NH;
    assert(C % NH == 0);
    assert(HS == 64 || HS == 128);
    assert(att != nullptr && "attention_forward uses att storage for TK log-sum-exp scratch");

    floatX* q = qkvr + 0 * B * T * C;
    floatX* k = qkvr + 1 * B * T * C;
    floatX* v = qkvr + 2 * B * T * C;

    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size, 0, stream>>>(q, k, v, inp, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    const int granularity = llmk::attention::fwd_sequence_granularity();
    const int Tpad = CEIL_DIV(T, granularity) * granularity;

    if (Tpad == T) {
        floatX* vaccum = inp;
        float* lse = reinterpret_cast<float*>(att);
        llmk::attention::launch_forward_causal(
            llmk::to_bf16(q), llmk::to_bf16(k), llmk::to_bf16(v),
            lse, llmk::to_bf16(vaccum),
            B, NH, T, HS, stream);
        cudaCheck(cudaGetLastError());

        num_blocks = CEIL_DIV(B * T * C, block_size);
        unpermute_kernel<<<num_blocks, block_size, 0, stream>>>(vaccum, out, B, T, NH, HS);
    } else {
        size_t padded_elems = (size_t)B * Tpad * C;
        floatX* q_pad = inp;
        floatX* k_pad = q_pad + padded_elems;
        floatX* v_pad = k_pad + padded_elems;
        floatX* o_pad = v_pad + padded_elems;
        float* lse = reinterpret_cast<float*>(o_pad + padded_elems);

        int padded_threads = B * NH * Tpad * HS;
        int padded_blocks = CEIL_DIV(padded_threads, block_size);
        pad_qkv_kernel<<<padded_blocks, block_size, 0, stream>>>(
            q_pad, k_pad, v_pad, q, k, v, B, T, Tpad, NH, HS);
        cudaCheck(cudaGetLastError());

        llmk::attention::launch_forward_causal(
            llmk::to_bf16(q_pad), llmk::to_bf16(k_pad), llmk::to_bf16(v_pad),
            lse, llmk::to_bf16(o_pad),
            B, NH, Tpad, HS, stream);
        cudaCheck(cudaGetLastError());

        int lse_blocks = CEIL_DIV(B * NH * T, block_size);
        copy_lse_unpad_kernel<<<lse_blocks, block_size, 0, stream>>>(
            reinterpret_cast<float*>(att), lse, B, T, Tpad, NH);
        cudaCheck(cudaGetLastError());

        num_blocks = CEIL_DIV(B * T * C, block_size);
        unpermute_padded_kernel<<<num_blocks, block_size, 0, stream>>>(
            o_pad, out, B, T, Tpad, NH, HS);
    }
    cudaCheck(cudaGetLastError());
}

__device__ inline void attention_row_stats(float& maxval, float& denom,
                                           const floatX* q_row, const floatX* k_base,
                                           int bnh, int t, int T, int HS, float scale) {
    maxval = -FLT_MAX;
    for (int s = 0; s <= t; ++s) {
        const floatX* k_row = k_base + (bnh * T + s) * HS;
        float score = 0.0f;
        for (int d = 0; d < HS; ++d) {
            score += (float)q_row[d] * (float)k_row[d];
        }
        maxval = fmaxf(maxval, score * scale);
    }
    denom = 0.0f;
    for (int s = 0; s <= t; ++s) {
        const floatX* k_row = k_base + (bnh * T + s) * HS;
        float score = 0.0f;
        for (int d = 0; d < HS; ++d) {
            score += (float)q_row[d] * (float)k_row[d];
        }
        denom += expf(score * scale - maxval);
    }
}

__device__ inline float attention_probability(const floatX* q_row, const floatX* k_row,
                                              int HS, float scale, float maxval, float denom) {
    float score = 0.0f;
    for (int d = 0; d < HS; ++d) {
        score += (float)q_row[d] * (float)k_row[d];
    }
    return expf(score * scale - maxval) / denom;
}

__device__ inline float attention_dp_sum(const floatX* dout_row, const floatX* q_row,
                                         const floatX* k_base, const floatX* v_base,
                                         int bnh, int t, int T, int HS,
                                         float scale, float maxval, float denom) {
    float sum = 0.0f;
    for (int s = 0; s <= t; ++s) {
        const floatX* k_row = k_base + (bnh * T + s) * HS;
        const floatX* v_row = v_base + (bnh * T + s) * HS;
        float p = attention_probability(q_row, k_row, HS, scale, maxval, denom);
        float dp = 0.0f;
        for (int d = 0; d < HS; ++d) {
            dp += (float)dout_row[d] * (float)v_row[d];
        }
        sum += p * dp;
    }
    return sum;
}

__global__ void attention_backward_dq_kernel(floatX* dq, const floatX* dout,
                                             const floatX* q, const floatX* k, const floatX* v,
                                             int B, int T, int NH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * T * HS;
    if (idx >= total) return;

    int hs = idx % HS;
    int t = (idx / HS) % T;
    int bnh = idx / (T * HS);
    const float scale = rsqrtf((float)HS);
    const floatX* q_row = q + (bnh * T + t) * HS;
    const floatX* dout_row = dout + (bnh * T + t) * HS;

    float maxval, denom;
    attention_row_stats(maxval, denom, q_row, k, bnh, t, T, HS, scale);
    float dp_sum = attention_dp_sum(dout_row, q_row, k, v, bnh, t, T, HS, scale, maxval, denom);

    float acc = 0.0f;
    for (int s = 0; s <= t; ++s) {
        const floatX* k_row = k + (bnh * T + s) * HS;
        const floatX* v_row = v + (bnh * T + s) * HS;
        float p = attention_probability(q_row, k_row, HS, scale, maxval, denom);
        float dp = 0.0f;
        for (int d = 0; d < HS; ++d) {
            dp += (float)dout_row[d] * (float)v_row[d];
        }
        float ds = p * (dp - dp_sum) * scale;
        acc += ds * (float)k_row[hs];
    }
    dq[idx] = (floatX)acc;
}

__global__ void attention_backward_dk_dv_kernel(floatX* dk, floatX* dv, const floatX* dout,
                                                const floatX* q, const floatX* k, const floatX* v,
                                                int B, int T, int NH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * T * HS;
    if (idx >= total) return;

    int hs = idx % HS;
    int s = (idx / HS) % T;
    int bnh = idx / (T * HS);
    const float scale = rsqrtf((float)HS);

    float dk_acc = 0.0f;
    float dv_acc = 0.0f;
    for (int t = s; t < T; ++t) {
        const floatX* q_row = q + (bnh * T + t) * HS;
        const floatX* k_row = k + (bnh * T + s) * HS;
        const floatX* dout_row = dout + (bnh * T + t) * HS;

        float maxval, denom;
        attention_row_stats(maxval, denom, q_row, k, bnh, t, T, HS, scale);
        float p = attention_probability(q_row, k_row, HS, scale, maxval, denom);
        float dp_sum = attention_dp_sum(dout_row, q_row, k, v, bnh, t, T, HS, scale, maxval, denom);

        float dp = 0.0f;
        for (int d = 0; d < HS; ++d) {
            dp += (float)dout_row[d] * (float)v[(bnh * T + s) * HS + d];
        }
        float ds = p * (dp - dp_sum) * scale;
        dk_acc += ds * (float)q_row[hs];
        dv_acc += p * (float)dout_row[hs];
    }
    dk[idx] = (floatX)dk_acc;
    dv[idx] = (floatX)dv_acc;
}

inline void attention_backward(floatX* dinp, floatX* dqkvr, floatX* datt, floatX* scratch,
                               const floatX* dout,
                               const floatX* qkvr, const floatX* att,
                               int B, int T, int C, int NH, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    const int HS = C / NH;
    assert(C % NH == 0);

    const floatX* q = qkvr + 0 * B * T * C;
    const floatX* k = qkvr + 1 * B * T * C;
    const floatX* v = qkvr + 2 * B * T * C;
    floatX* dq = dqkvr + 0 * B * T * C;
    floatX* dk = dqkvr + 1 * B * T * C;
    floatX* dv = dqkvr + 2 * B * T * C;

    int total = B * T * C;
    int num_blocks = CEIL_DIV(total, block_size);

    if (datt != nullptr && att != nullptr &&
        (HS == 64 || HS == 128) &&
        T % llmk::attention::bwd_sequence_granularity() == 0) {
        const size_t qkv_elems = (size_t)B * NH * T * HS;
        floatX* o_perm = datt;
        floatX* og_perm = o_perm + qkv_elems;
        float* d_vec = reinterpret_cast<float*>(og_perm + qkv_elems);
        const size_t vec_elems = (size_t)B * NH * T;
        float* qg = d_vec + vec_elems;
        float* kg = qg + qkv_elems;
        float* vg = kg + qkv_elems;

        unpermute_kernel_backward<<<num_blocks, block_size, 0, stream>>>(
            o_perm, scratch, B, T, NH, HS);
        cudaCheck(cudaGetLastError());
        unpermute_kernel_backward<<<num_blocks, block_size, 0, stream>>>(
            og_perm, dout, B, T, NH, HS);
        cudaCheck(cudaGetLastError());

        cudaCheck(cudaMemsetAsync(qg, 0, 3 * qkv_elems * sizeof(float), stream));
        llmk::attention::launch_backward_causal(
            llmk::to_bf16(const_cast<floatX*>(q)),
            llmk::to_bf16(const_cast<floatX*>(k)),
            llmk::to_bf16(const_cast<floatX*>(v)),
            llmk::to_bf16(o_perm),
            reinterpret_cast<float*>(const_cast<floatX*>(att)),
            llmk::to_bf16(og_perm),
            d_vec, qg, kg, vg,
            B, NH, T, HS, stream);
        cudaCheck(cudaGetLastError());

        attention_float_grads_to_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
            dq, dk, dv, qg, kg, vg, total);
        cudaCheck(cudaGetLastError());
        permute_kernel_backward<<<num_blocks, block_size, 0, stream>>>(dinp, dq, dk, dv, B, T, NH, HS);
        cudaCheck(cudaGetLastError());
        return;
    }

    unpermute_kernel_backward<<<num_blocks, block_size, 0, stream>>>(scratch, dout, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    attention_backward_dq_kernel<<<num_blocks, block_size, 0, stream>>>(dq, scratch, q, k, v, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
    attention_backward_dk_dv_kernel<<<num_blocks, block_size, 0, stream>>>(dk, dv, scratch, q, k, v, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
    permute_kernel_backward<<<num_blocks, block_size, 0, stream>>>(dinp, dq, dk, dv, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}
