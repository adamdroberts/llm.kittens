/*
Grouped-query causal attention for Llama-3.

This exposes the Llama-3 GQA API, repeats KV logically across query groups, and
recomputes softmax statistics in backward. TK-supported forward+backward shapes
save unrotated Q/K and rotate inside the TK tile-load path; unsupported shapes
use the fused materialized-RoPE path and the plain-CUDA correctness baseline.
*/
#pragma once

#include <assert.h>
#include <float.h>

#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "tk/attention_gqa_h100.cuh"

inline bool attention_gqa_uses_tk_tile_rope(const floatX* cos, const floatX* sin,
                                            const floatX* tk_workspace,
                                            int T, int C, int NH, int NKVH) {
    if (cos == nullptr || sin == nullptr || tk_workspace == nullptr) { return false; }
    if (NH <= 0 || C % NH != 0) { return false; }
    return llmk::attention_gqa::has_tk_backward(T, C / NH, NH, NKVH);
}

__global__ void gqa_permute_q_kernel(floatX* q, const floatX* inp,
                                     int B, int T, int NH, int NKVH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * T * HS;
    if (idx >= total) { return; }

    int hs = idx % HS;
    int t = (idx / HS) % T;
    int qh = (idx / (HS * T)) % NH;
    int b = idx / (NH * T * HS);
    int qkv_width = (NH + 2 * NKVH) * HS;
    int src = (b * T + t) * qkv_width + qh * HS + hs;
    q[idx] = __ldcs(&inp[src]);
}

__global__ void gqa_permute_q_rope_kernel(floatX* q, const floatX* inp,
                                          const floatX* cos, const floatX* sin,
                                          int B, int T, int NH, int NKVH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * T * HS;
    if (idx >= total) { return; }

    int hs = idx % HS;
    int t = (idx / HS) % T;
    int qh = (idx / (HS * T)) % NH;
    int b = idx / (NH * T * HS);
    int half = HS / 2;
    int pair = hs % half;
    int qkv_width = (NH + 2 * NKVH) * HS;
    int base = (b * T + t) * qkv_width + qh * HS;

    float x1 = (float)__ldcs(&inp[base + pair]);
    float x2 = (float)__ldcs(&inp[base + pair + half]);
    float c = (float)__ldcs(&cos[t * half + pair]);
    float s = (float)__ldcs(&sin[t * half + pair]);

    float rotated = (hs < half) ? (x1 * c - x2 * s) : (x2 * c + x1 * s);
    q[idx] = (floatX)rotated;
}

__global__ void gqa_permute_kv_kernel(floatX* k, floatX* v, const floatX* inp,
                                      int B, int T, int NH, int NKVH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NKVH * T * HS;
    if (idx >= total) { return; }

    int hs = idx % HS;
    int t = (idx / HS) % T;
    int kvh = (idx / (HS * T)) % NKVH;
    int b = idx / (NKVH * T * HS);
    int qkv_width = (NH + 2 * NKVH) * HS;
    int base = (b * T + t) * qkv_width;
    k[idx] = __ldcs(&inp[base + (NH + kvh) * HS + hs]);
    v[idx] = __ldcs(&inp[base + (NH + NKVH + kvh) * HS + hs]);
}

__global__ void gqa_permute_kv_rope_kernel(floatX* k, floatX* v, const floatX* inp,
                                           const floatX* cos, const floatX* sin,
                                           int B, int T, int NH, int NKVH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NKVH * T * HS;
    if (idx >= total) { return; }

    int hs = idx % HS;
    int t = (idx / HS) % T;
    int kvh = (idx / (HS * T)) % NKVH;
    int b = idx / (NKVH * T * HS);
    int half = HS / 2;
    int pair = hs % half;
    int qkv_width = (NH + 2 * NKVH) * HS;
    int base = (b * T + t) * qkv_width;
    int k_base = base + (NH + kvh) * HS;

    float x1 = (float)__ldcs(&inp[k_base + pair]);
    float x2 = (float)__ldcs(&inp[k_base + pair + half]);
    float c = (float)__ldcs(&cos[t * half + pair]);
    float s = (float)__ldcs(&sin[t * half + pair]);

    float rotated = (hs < half) ? (x1 * c - x2 * s) : (x2 * c + x1 * s);
    k[idx] = (floatX)rotated;
    v[idx] = __ldcs(&inp[base + (NH + NKVH + kvh) * HS + hs]);
}

__global__ void gqa_unpermute_dout_kernel(floatX* dout_perm, const floatX* dout,
                                          int B, int T, int NH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * T * HS;
    if (idx >= total) { return; }

    int hs = idx % HS;
    int t = (idx / HS) % T;
    int qh = (idx / (HS * T)) % NH;
    int b = idx / (NH * T * HS);
    int src = (b * T + t) * NH * HS + qh * HS + hs;
    dout_perm[idx] = __ldcs(&dout[src]);
}

__global__ void gqa_unpermute_forward_kernel(floatX* out, const floatX* out_perm,
                                             int B, int T, int NH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * T * HS;
    if (idx >= total) { return; }

    int hs = idx % HS;
    int t = (idx / HS) % T;
    int qh = (idx / (HS * T)) % NH;
    int b = idx / (NH * T * HS);
    int dst = (b * T + t) * NH * HS + qh * HS + hs;
    out[dst] = __ldcs(&out_perm[idx]);
}

__global__ void gqa_permute_backward_kernel(floatX* dinp,
                                            const floatX* dq, const floatX* dk, const floatX* dv,
                                            int B, int T, int NH, int NKVH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int q_total = B * NH * T * HS;
    int kv_total = B * NKVH * T * HS;
    int qkv_width = (NH + 2 * NKVH) * HS;

    if (idx < q_total) {
        int hs = idx % HS;
        int t = (idx / HS) % T;
        int qh = (idx / (HS * T)) % NH;
        int b = idx / (NH * T * HS);
        dinp[(b * T + t) * qkv_width + qh * HS + hs] = dq[idx];
    }

    if (idx < kv_total) {
        int hs = idx % HS;
        int t = (idx / HS) % T;
        int kvh = (idx / (HS * T)) % NKVH;
        int b = idx / (NKVH * T * HS);
        int dst_base = (b * T + t) * qkv_width;
        dinp[dst_base + (NH + kvh) * HS + hs] = dk[idx];
        dinp[dst_base + (NH + NKVH + kvh) * HS + hs] = dv[idx];
    }
}

__global__ void gqa_permute_backward_rope_kernel(floatX* dinp,
                                                 const floatX* dq, const floatX* dk, const floatX* dv,
                                                 const floatX* cos, const floatX* sin,
                                                 int B, int T, int NH, int NKVH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int q_total = B * NH * T * HS;
    int kv_total = B * NKVH * T * HS;
    int qkv_width = (NH + 2 * NKVH) * HS;
    int half = HS / 2;

    if (idx < q_total) {
        int hs = idx % HS;
        int t = (idx / HS) % T;
        int qh = (idx / (HS * T)) % NH;
        int b = idx / (NH * T * HS);
        int pair = hs % half;
        size_t row = ((size_t)b * NH + qh) * T + t;
        size_t base = row * HS;
        float x1 = (float)__ldcs(&dq[base + pair]);
        float x2 = (float)__ldcs(&dq[base + pair + half]);
        float c = (float)__ldcs(&cos[t * half + pair]);
        float s = (float)__ldcs(&sin[t * half + pair]);
        float rotated = (hs < half) ? (x1 * c + x2 * s) : (x2 * c - x1 * s);
        dinp[(b * T + t) * qkv_width + qh * HS + hs] = (floatX)rotated;
    }

    if (idx < kv_total) {
        int hs = idx % HS;
        int t = (idx / HS) % T;
        int kvh = (idx / (HS * T)) % NKVH;
        int b = idx / (NKVH * T * HS);
        int pair = hs % half;
        size_t row = ((size_t)b * NKVH + kvh) * T + t;
        size_t base = row * HS;
        float x1 = (float)__ldcs(&dk[base + pair]);
        float x2 = (float)__ldcs(&dk[base + pair + half]);
        float c = (float)__ldcs(&cos[t * half + pair]);
        float s = (float)__ldcs(&sin[t * half + pair]);
        float rotated = (hs < half) ? (x1 * c + x2 * s) : (x2 * c - x1 * s);
        int dst_base = (b * T + t) * qkv_width;
        dinp[dst_base + (NH + kvh) * HS + hs] = (floatX)rotated;
        dinp[dst_base + (NH + NKVH + kvh) * HS + hs] = dv[idx];
    }
}

__global__ void gqa_float_grads_to_bf16_kernel(floatX* dq, floatX* dk, floatX* dv,
                                               const float* qg, const float* kg, const float* vg,
                                               size_t q_total, size_t kv_total) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < q_total) {
        dq[idx] = (floatX)qg[idx];
    }
    if (idx < kv_total) {
        dk[idx] = (floatX)kg[idx];
        dv[idx] = (floatX)vg[idx];
    }
}

__device__ inline const floatX* gqa_q_row(const floatX* q, int b, int qh, int t, int T, int NH, int HS) {
    return q + ((b * NH + qh) * T + t) * HS;
}

__device__ inline const floatX* gqa_kv_row(const floatX* kv, int b, int kvh, int t, int T, int NKVH, int HS) {
    return kv + ((b * NKVH + kvh) * T + t) * HS;
}

__device__ inline void gqa_row_stats(float& maxval, float& denom,
                                     const floatX* q_row, const floatX* k,
                                     int b, int kvh, int t, int T, int NKVH, int HS,
                                     float scale) {
    maxval = -FLT_MAX;
    for (int s = 0; s <= t; ++s) {
        const floatX* k_row = gqa_kv_row(k, b, kvh, s, T, NKVH, HS);
        float score = 0.0f;
        for (int d = 0; d < HS; ++d) {
            score += (float)q_row[d] * (float)k_row[d];
        }
        maxval = fmaxf(maxval, score * scale);
    }
    denom = 0.0f;
    for (int s = 0; s <= t; ++s) {
        const floatX* k_row = gqa_kv_row(k, b, kvh, s, T, NKVH, HS);
        float score = 0.0f;
        for (int d = 0; d < HS; ++d) {
            score += (float)q_row[d] * (float)k_row[d];
        }
        denom += expf(score * scale - maxval);
    }
}

__device__ inline float gqa_probability(const floatX* q_row, const floatX* k_row,
                                        int HS, float scale, float maxval, float denom) {
    float score = 0.0f;
    for (int d = 0; d < HS; ++d) {
        score += (float)q_row[d] * (float)k_row[d];
    }
    return expf(score * scale - maxval) / denom;
}

__device__ inline float gqa_dp_sum(const floatX* dout_row, const floatX* q_row,
                                  const floatX* k, const floatX* v,
                                  int b, int kvh, int t, int T, int NKVH, int HS,
                                  float scale, float maxval, float denom) {
    float sum = 0.0f;
    for (int s = 0; s <= t; ++s) {
        const floatX* k_row = gqa_kv_row(k, b, kvh, s, T, NKVH, HS);
        const floatX* v_row = gqa_kv_row(v, b, kvh, s, T, NKVH, HS);
        float p = gqa_probability(q_row, k_row, HS, scale, maxval, denom);
        float dp = 0.0f;
        for (int d = 0; d < HS; ++d) {
            dp += (float)dout_row[d] * (float)v_row[d];
        }
        sum += p * dp;
    }
    return sum;
}

__global__ void gqa_attention_forward_kernel(floatX* out, float* lse,
                                             const floatX* q, const floatX* k, const floatX* v,
                                             int B, int T, int NH, int NKVH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * T * HS;
    if (idx >= total) { return; }

    int hs = idx % HS;
    int t = (idx / HS) % T;
    int qh = (idx / (HS * T)) % NH;
    int b = idx / (NH * T * HS);
    int nrep = NH / NKVH;
    int kvh = qh / nrep;
    float scale = rsqrtf((float)HS);

    const floatX* q_row = gqa_q_row(q, b, qh, t, T, NH, HS);
    float maxval, denom;
    gqa_row_stats(maxval, denom, q_row, k, b, kvh, t, T, NKVH, HS, scale);

    float acc = 0.0f;
    for (int s = 0; s <= t; ++s) {
        const floatX* k_row = gqa_kv_row(k, b, kvh, s, T, NKVH, HS);
        const floatX* v_row = gqa_kv_row(v, b, kvh, s, T, NKVH, HS);
        float p = gqa_probability(q_row, k_row, HS, scale, maxval, denom);
        acc += p * (float)v_row[hs];
    }

    int out_idx = (b * T + t) * NH * HS + qh * HS + hs;
    out[out_idx] = (floatX)acc;
    if (lse != nullptr && hs == 0) {
        lse[(b * NH + qh) * T + t] = maxval + logf(denom);
    }
}

__global__ void gqa_attention_backward_dq_kernel(floatX* dq, const floatX* dout,
                                                 const floatX* q, const floatX* k, const floatX* v,
                                                 int B, int T, int NH, int NKVH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * T * HS;
    if (idx >= total) { return; }

    int hs = idx % HS;
    int t = (idx / HS) % T;
    int qh = (idx / (HS * T)) % NH;
    int b = idx / (NH * T * HS);
    int nrep = NH / NKVH;
    int kvh = qh / nrep;
    float scale = rsqrtf((float)HS);

    const floatX* q_row = gqa_q_row(q, b, qh, t, T, NH, HS);
    const floatX* dout_row = gqa_q_row(dout, b, qh, t, T, NH, HS);
    float maxval, denom;
    gqa_row_stats(maxval, denom, q_row, k, b, kvh, t, T, NKVH, HS, scale);
    float dp_norm = gqa_dp_sum(dout_row, q_row, k, v, b, kvh, t, T, NKVH, HS,
                               scale, maxval, denom);

    float acc = 0.0f;
    for (int s = 0; s <= t; ++s) {
        const floatX* k_row = gqa_kv_row(k, b, kvh, s, T, NKVH, HS);
        const floatX* v_row = gqa_kv_row(v, b, kvh, s, T, NKVH, HS);
        float p = gqa_probability(q_row, k_row, HS, scale, maxval, denom);
        float dp = 0.0f;
        for (int d = 0; d < HS; ++d) {
            dp += (float)dout_row[d] * (float)v_row[d];
        }
        float ds = p * (dp - dp_norm) * scale;
        acc += ds * (float)k_row[hs];
    }
    dq[idx] = (floatX)acc;
}

__global__ void gqa_attention_backward_dk_dv_kernel(floatX* dk, floatX* dv,
                                                    const floatX* dout,
                                                    const floatX* q, const floatX* k, const floatX* v,
                                                    int B, int T, int NH, int NKVH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NKVH * T * HS;
    if (idx >= total) { return; }

    int hs = idx % HS;
    int s = (idx / HS) % T;
    int kvh = (idx / (HS * T)) % NKVH;
    int b = idx / (NKVH * T * HS);
    int nrep = NH / NKVH;
    float scale = rsqrtf((float)HS);

    float dk_acc = 0.0f;
    float dv_acc = 0.0f;
    for (int qhi = 0; qhi < nrep; ++qhi) {
        int qh = kvh * nrep + qhi;
        for (int t = s; t < T; ++t) {
            const floatX* q_row = gqa_q_row(q, b, qh, t, T, NH, HS);
            const floatX* k_row = gqa_kv_row(k, b, kvh, s, T, NKVH, HS);
            const floatX* dout_row = gqa_q_row(dout, b, qh, t, T, NH, HS);

            float maxval, denom;
            gqa_row_stats(maxval, denom, q_row, k, b, kvh, t, T, NKVH, HS, scale);
            float p = gqa_probability(q_row, k_row, HS, scale, maxval, denom);
            float dp_norm = gqa_dp_sum(dout_row, q_row, k, v, b, kvh, t, T, NKVH, HS,
                                       scale, maxval, denom);

            float dp = 0.0f;
            for (int d = 0; d < HS; ++d) {
                dp += (float)dout_row[d] * (float)gqa_kv_row(v, b, kvh, s, T, NKVH, HS)[d];
            }
            float ds = p * (dp - dp_norm) * scale;
            dk_acc += ds * (float)q_row[hs];
            dv_acc += p * (float)dout_row[hs];
        }
    }
    dk[idx] = (floatX)dk_acc;
    dv[idx] = (floatX)dv_acc;
}

inline void attention_gqa_forward(floatX* out, floatX* qkvr, float* lse,
                                  const floatX* inp, const floatX* cos, const floatX* sin,
                                  int B, int T, int C, int NH, int NKVH,
                                  cudaStream_t stream, floatX* tk_workspace = nullptr) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    assert(C % NH == 0);
    assert(NH % NKVH == 0);
    int HS = C / NH;
    assert(HS == 64 || HS == 128);

    size_t q_elems = (size_t)B * NH * T * HS;
    size_t kv_elems = (size_t)B * NKVH * T * HS;
    floatX* q = qkvr;
    floatX* k = q + q_elems;
    floatX* v = k + kv_elems;
    int q_blocks = CEIL_DIV(q_elems, block_size);
    int kv_blocks = CEIL_DIV(kv_elems, block_size);
    bool tk_tile_rope = attention_gqa_uses_tk_tile_rope(cos, sin, tk_workspace, T, C, NH, NKVH);

    if (tk_tile_rope) {
        gqa_permute_q_kernel<<<q_blocks, block_size, 0, stream>>>(q, inp, B, T, NH, NKVH, HS);
        cudaCheck(cudaGetLastError());
        gqa_permute_kv_kernel<<<kv_blocks, block_size, 0, stream>>>(k, v, inp, B, T, NH, NKVH, HS);
        cudaCheck(cudaGetLastError());
    } else if (cos != nullptr && sin != nullptr) {
        gqa_permute_q_rope_kernel<<<q_blocks, block_size, 0, stream>>>(
            q, inp, cos, sin, B, T, NH, NKVH, HS);
        cudaCheck(cudaGetLastError());
        gqa_permute_kv_rope_kernel<<<kv_blocks, block_size, 0, stream>>>(
            k, v, inp, cos, sin, B, T, NH, NKVH, HS);
        cudaCheck(cudaGetLastError());
    } else {
        gqa_permute_q_kernel<<<q_blocks, block_size, 0, stream>>>(q, inp, B, T, NH, NKVH, HS);
        cudaCheck(cudaGetLastError());
        gqa_permute_kv_kernel<<<kv_blocks, block_size, 0, stream>>>(k, v, inp, B, T, NH, NKVH, HS);
        cudaCheck(cudaGetLastError());
    }

    if (tk_workspace != nullptr &&
        llmk::attention_gqa::has_tk_forward(T, HS, NH, NKVH)) {
        if (tk_tile_rope) {
            llmk::attention_gqa::launch_forward_causal_rope(
                llmk::to_bf16(q), llmk::to_bf16(k), llmk::to_bf16(v),
                llmk::to_bf16(const_cast<floatX*>(cos)),
                llmk::to_bf16(const_cast<floatX*>(sin)),
                lse, llmk::to_bf16(tk_workspace),
                B, NH, NKVH, T, HS, stream);
        } else {
            llmk::attention_gqa::launch_forward_causal(
                llmk::to_bf16(q), llmk::to_bf16(k), llmk::to_bf16(v),
                lse, llmk::to_bf16(tk_workspace),
                B, NH, NKVH, T, HS, stream);
        }
        cudaCheck(cudaGetLastError());
        gqa_unpermute_forward_kernel<<<q_blocks, block_size, 0, stream>>>(
            out, tk_workspace, B, T, NH, HS);
        cudaCheck(cudaGetLastError());
        return;
    }

    gqa_attention_forward_kernel<<<q_blocks, block_size, 0, stream>>>(
        out, lse, q, k, v, B, T, NH, NKVH, HS);
    cudaCheck(cudaGetLastError());
}

inline void attention_gqa_backward(floatX* dinp, floatX* dqkvr, float* datt, floatX* scratch,
                                   const floatX* dout, const floatX* out,
                                   const floatX* qkvr, const float* lse,
                                   const floatX* cos, const floatX* sin,
                                   int B, int T, int C, int NH, int NKVH,
                                   cudaStream_t stream,
                                   bool qkvr_uses_tk_tile_rope = false) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    assert(C % NH == 0);
    assert(NH % NKVH == 0);
    assert(scratch != nullptr);
    int HS = C / NH;

    size_t q_elems = (size_t)B * NH * T * HS;
    size_t kv_elems = (size_t)B * NKVH * T * HS;
    const floatX* q = qkvr;
    const floatX* k = q + q_elems;
    const floatX* v = k + kv_elems;
    floatX* dq = dqkvr;
    floatX* dk = dq + q_elems;
    floatX* dv = dk + kv_elems;

    int q_blocks = CEIL_DIV(q_elems, block_size);
    int kv_blocks = CEIL_DIV(kv_elems, block_size);

    if (out != nullptr && datt != nullptr && lse != nullptr &&
        llmk::attention_gqa::has_tk_backward(T, HS, NH, NKVH)) {
        floatX* o_perm = scratch;
        floatX* og_perm = o_perm + q_elems;
        size_t vec_elems = (size_t)B * NH * T;
        float* d_vec = datt;
        float* qg = d_vec + vec_elems;
        float* kg = qg + q_elems;
        float* vg = kg + kv_elems;

        gqa_unpermute_dout_kernel<<<q_blocks, block_size, 0, stream>>>(
            o_perm, out, B, T, NH, HS);
        cudaCheck(cudaGetLastError());
        gqa_unpermute_dout_kernel<<<q_blocks, block_size, 0, stream>>>(
            og_perm, dout, B, T, NH, HS);
        cudaCheck(cudaGetLastError());

        cudaCheck(cudaMemsetAsync(qg, 0, (q_elems + 2 * kv_elems) * sizeof(float), stream));
        llmk::attention_gqa::launch_backward_causal(
            llmk::to_bf16(const_cast<floatX*>(q)),
            llmk::to_bf16(const_cast<floatX*>(k)),
            llmk::to_bf16(const_cast<floatX*>(v)),
            llmk::to_bf16(o_perm),
            const_cast<float*>(lse),
            llmk::to_bf16(og_perm),
            d_vec, qg, kg, vg,
            B, NH, NKVH, T, HS, stream,
            qkvr_uses_tk_tile_rope ? llmk::to_bf16(const_cast<floatX*>(cos)) : nullptr,
            qkvr_uses_tk_tile_rope ? llmk::to_bf16(const_cast<floatX*>(sin)) : nullptr);
        cudaCheck(cudaGetLastError());

        size_t total = q_elems > kv_elems ? q_elems : kv_elems;
        int blocks = CEIL_DIV(total, (size_t)block_size);
        gqa_float_grads_to_bf16_kernel<<<blocks, block_size, 0, stream>>>(
            dq, dk, dv, qg, kg, vg, q_elems, kv_elems);
        cudaCheck(cudaGetLastError());
    } else {
        gqa_unpermute_dout_kernel<<<q_blocks, block_size, 0, stream>>>(
            scratch, dout, B, T, NH, HS);
        cudaCheck(cudaGetLastError());

        gqa_attention_backward_dq_kernel<<<q_blocks, block_size, 0, stream>>>(
            dq, scratch, q, k, v, B, T, NH, NKVH, HS);
        cudaCheck(cudaGetLastError());
        gqa_attention_backward_dk_dv_kernel<<<kv_blocks, block_size, 0, stream>>>(
            dk, dv, scratch, q, k, v, B, T, NH, NKVH, HS);
        cudaCheck(cudaGetLastError());
    }

    int total = (int)(q_elems > kv_elems ? q_elems : kv_elems);
    int blocks = CEIL_DIV(total, block_size);
    if (cos != nullptr && sin != nullptr) {
        gqa_permute_backward_rope_kernel<<<blocks, block_size, 0, stream>>>(
            dinp, dq, dk, dv, cos, sin, B, T, NH, NKVH, HS);
    } else {
        gqa_permute_backward_kernel<<<blocks, block_size, 0, stream>>>(
            dinp, dq, dk, dv, B, T, NH, NKVH, HS);
    }
    cudaCheck(cudaGetLastError());
}
