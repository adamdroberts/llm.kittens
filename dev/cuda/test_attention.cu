/*
test_attention.cu — smoke test for GPT-style causal MHA.

Generates packed GPT Q/K/V projections, runs llmc/attention.cuh, and compares
the forward output plus packed Q/K/V input gradients against an independent CPU
reference. The T=192 case covers a direct TK forward launch plus the CUDA
fallback backward path. The T=256 case covers padded TK forward plus the
supported-shape TK backward path.

Build via the Makefile target:

    make test_attention
*/
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

cudaDeviceProp deviceProp;

#include "llmc/attention.cuh"

static float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

static __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

static void fill_random_bf16(std::vector<__nv_bfloat16>& h, uint64_t seed,
                             float lo, float hi) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& v : h) {
        v = float_to_bf16(dist(rng));
    }
}

static inline size_t packed_idx(int b, int t, int qkv, int h, int d,
                                int T, int NH, int HS) {
    return ((size_t)b * T + t) * (3 * NH * HS) + (size_t)qkv * NH * HS +
           (size_t)h * HS + d;
}

static inline size_t perm_idx(int b, int h, int t, int d,
                              int NH, int T, int HS) {
    return (((size_t)b * NH + h) * T + t) * HS + d;
}

static inline size_t out_idx(int b, int t, int h, int d,
                             int T, int NH, int HS) {
    return (((size_t)b * T + t) * NH + h) * HS + d;
}

static void unpack_qkv(std::vector<float>& q,
                       std::vector<float>& k,
                       std::vector<float>& v,
                       const std::vector<__nv_bfloat16>& inp,
                       int B, int T, int NH, int HS) {
    q.assign((size_t)B * NH * T * HS, 0.0f);
    k.assign((size_t)B * NH * T * HS, 0.0f);
    v.assign((size_t)B * NH * T * HS, 0.0f);

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < NH; ++h) {
                for (int d = 0; d < HS; ++d) {
                    q[perm_idx(b, h, t, d, NH, T, HS)] =
                        bf16_to_float(inp[packed_idx(b, t, 0, h, d, T, NH, HS)]);
                    k[perm_idx(b, h, t, d, NH, T, HS)] =
                        bf16_to_float(inp[packed_idx(b, t, 1, h, d, T, NH, HS)]);
                    v[perm_idx(b, h, t, d, NH, T, HS)] =
                        bf16_to_float(inp[packed_idx(b, t, 2, h, d, T, NH, HS)]);
                }
            }
        }
    }
}

static void cpu_attention_forward(std::vector<float>& out,
                                  const std::vector<float>& q,
                                  const std::vector<float>& k,
                                  const std::vector<float>& v,
                                  int B, int T, int NH, int HS) {
    out.assign((size_t)B * T * NH * HS, 0.0f);
    float scale = 1.0f / sqrtf((float)HS);
    std::vector<float> scores(T);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < NH; ++h) {
            for (int t = 0; t < T; ++t) {
                float maxval = -INFINITY;
                for (int s = 0; s <= t; ++s) {
                    float score = 0.0f;
                    for (int d = 0; d < HS; ++d) {
                        score += q[perm_idx(b, h, t, d, NH, T, HS)] *
                                 k[perm_idx(b, h, s, d, NH, T, HS)];
                    }
                    score *= scale;
                    scores[s] = score;
                    maxval = fmaxf(maxval, score);
                }

                float denom = 0.0f;
                for (int s = 0; s <= t; ++s) {
                    scores[s] = expf(scores[s] - maxval);
                    denom += scores[s];
                }

                for (int d = 0; d < HS; ++d) {
                    float acc = 0.0f;
                    for (int s = 0; s <= t; ++s) {
                        float p = scores[s] / denom;
                        acc += p * v[perm_idx(b, h, s, d, NH, T, HS)];
                    }
                    out[out_idx(b, t, h, d, T, NH, HS)] = acc;
                }
            }
        }
    }
}

static void cpu_attention_backward_packed(std::vector<float>& dinp,
                                          const std::vector<float>& dout,
                                          const std::vector<float>& q,
                                          const std::vector<float>& k,
                                          const std::vector<float>& v,
                                          int B, int T, int NH, int HS) {
    std::vector<float> dq((size_t)B * NH * T * HS, 0.0f);
    std::vector<float> dk((size_t)B * NH * T * HS, 0.0f);
    std::vector<float> dv((size_t)B * NH * T * HS, 0.0f);
    std::vector<float> scores(T);
    std::vector<float> probs(T);

    float scale = 1.0f / sqrtf((float)HS);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < NH; ++h) {
            for (int t = 0; t < T; ++t) {
                float maxval = -INFINITY;
                for (int s = 0; s <= t; ++s) {
                    float score = 0.0f;
                    for (int d = 0; d < HS; ++d) {
                        score += q[perm_idx(b, h, t, d, NH, T, HS)] *
                                 k[perm_idx(b, h, s, d, NH, T, HS)];
                    }
                    score *= scale;
                    scores[s] = score;
                    maxval = fmaxf(maxval, score);
                }

                float denom = 0.0f;
                for (int s = 0; s <= t; ++s) {
                    probs[s] = expf(scores[s] - maxval);
                    denom += probs[s];
                }
                for (int s = 0; s <= t; ++s) {
                    probs[s] /= denom;
                }

                float dp_norm = 0.0f;
                for (int s = 0; s <= t; ++s) {
                    float dp = 0.0f;
                    for (int d = 0; d < HS; ++d) {
                        dp += dout[out_idx(b, t, h, d, T, NH, HS)] *
                              v[perm_idx(b, h, s, d, NH, T, HS)];
                    }
                    dp_norm += probs[s] * dp;
                }

                for (int s = 0; s <= t; ++s) {
                    float dp = 0.0f;
                    for (int d = 0; d < HS; ++d) {
                        dp += dout[out_idx(b, t, h, d, T, NH, HS)] *
                              v[perm_idx(b, h, s, d, NH, T, HS)];
                    }
                    float ds = probs[s] * (dp - dp_norm) * scale;
                    for (int d = 0; d < HS; ++d) {
                        dq[perm_idx(b, h, t, d, NH, T, HS)] +=
                            ds * k[perm_idx(b, h, s, d, NH, T, HS)];
                        dk[perm_idx(b, h, s, d, NH, T, HS)] +=
                            ds * q[perm_idx(b, h, t, d, NH, T, HS)];
                        dv[perm_idx(b, h, s, d, NH, T, HS)] +=
                            probs[s] * dout[out_idx(b, t, h, d, T, NH, HS)];
                    }
                }
            }
        }
    }

    dinp.assign((size_t)B * T * 3 * NH * HS, 0.0f);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < NH; ++h) {
                for (int d = 0; d < HS; ++d) {
                    dinp[packed_idx(b, t, 0, h, d, T, NH, HS)] =
                        dq[perm_idx(b, h, t, d, NH, T, HS)];
                    dinp[packed_idx(b, t, 1, h, d, T, NH, HS)] =
                        dk[perm_idx(b, h, t, d, NH, T, HS)];
                    dinp[packed_idx(b, t, 2, h, d, T, NH, HS)] =
                        dv[perm_idx(b, h, t, d, NH, T, HS)];
                }
            }
        }
    }
}

static double max_abs_diff_bf16_float(const std::vector<__nv_bfloat16>& actual,
                                      const std::vector<float>& expected) {
    double max_diff = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        double diff = std::abs((double)bf16_to_float(actual[i]) - (double)expected[i]);
        max_diff = std::max(max_diff, diff);
    }
    return max_diff;
}

static int run_case(int T, bool request_tk_backward, uint64_t seed_offset,
                    bool packed_qkv_sm120 = false) {
    constexpr int B = 1;
    constexpr int NH = 2;
    constexpr int HS = 64;
    constexpr int C = NH * HS;

    const int fwd_granularity = llmk::attention::fwd_sequence_granularity();
    const int bwd_granularity = llmk::attention::bwd_sequence_granularity();
    const int Tpad = ((T + fwd_granularity - 1) / fwd_granularity) * fwd_granularity;
    const bool padded_forward = Tpad != T;
    const bool tk_backward = request_tk_backward && (T % bwd_granularity == 0);

    printf("\nShape: B=%d T=%d NH=%d HS=%d\n", B, T, NH, HS);
    printf("TK forward padding: %s (Tpad=%d)\n", padded_forward ? "yes" : "no", Tpad);
    printf("TK backward requested/used: %s/%s\n",
           request_tk_backward ? "yes" : "no",
           tk_backward ? "yes" : "no");
#if defined(KITTENS_SM120)
    printf("SM120 packed-QKV fast path: %s\n", packed_qkv_sm120 ? "yes" : "no");
#else
    (void)packed_qkv_sm120;
#endif

    const size_t out_elems = (size_t)B * T * C;
    const size_t packed_elems = 3 * out_elems;
    const size_t out_bytes = out_elems * sizeof(__nv_bfloat16);
    const size_t packed_bytes = packed_elems * sizeof(__nv_bfloat16);
    const size_t padded_elems = (size_t)B * Tpad * C;
    const size_t fwd_workspace_bytes =
        4 * padded_elems * sizeof(__nv_bfloat16) +
        (size_t)B * NH * Tpad * sizeof(float);
    const size_t inp_workspace_bytes = std::max(packed_bytes, fwd_workspace_bytes);
    size_t att_bf16_elems = 2 * (size_t)B * NH * T;
    size_t datt_bf16_elems = 2 * out_elems;
    size_t datt_float_elems = (size_t)B * NH * T + 3 * out_elems;
#if defined(KITTENS_SM120)
    datt_bf16_elems = 2 * out_elems;
    datt_float_elems = (size_t)B * NH * T;
#if defined(LLMK_SM120_ATOMIC_DQ)
    datt_float_elems += out_elems;
#endif
#endif
    const size_t att_bytes = att_bf16_elems * sizeof(__nv_bfloat16);
    const size_t datt_bytes =
        datt_bf16_elems * sizeof(__nv_bfloat16) +
        datt_float_elems * sizeof(float);

    std::vector<__nv_bfloat16> h_inp(packed_elems);
    std::vector<__nv_bfloat16> h_dout(out_elems);
    fill_random_bf16(h_inp, 1234 + seed_offset, -0.35f, 0.35f);
    fill_random_bf16(h_dout, 5678 + seed_offset, -0.20f, 0.20f);

    std::vector<float> q, k, v, ref_out, ref_dinp;
    unpack_qkv(q, k, v, h_inp, B, T, NH, HS);
    cpu_attention_forward(ref_out, q, k, v, B, T, NH, HS);

    std::vector<float> h_dout_float(h_dout.size());
    for (size_t i = 0; i < h_dout.size(); ++i) {
        h_dout_float[i] = bf16_to_float(h_dout[i]);
    }
    cpu_attention_backward_packed(ref_dinp, h_dout_float, q, k, v, B, T, NH, HS);

    __nv_bfloat16* d_inp = nullptr;
    __nv_bfloat16* d_qkvr = nullptr;
    __nv_bfloat16* d_att = nullptr;
    __nv_bfloat16* d_out = nullptr;
    __nv_bfloat16* d_dout = nullptr;
    __nv_bfloat16* d_dinp = nullptr;
    __nv_bfloat16* d_dqkvr = nullptr;
    __nv_bfloat16* d_datt = nullptr;
    __nv_bfloat16* d_scratch = nullptr;

    cudaCheck(cudaMalloc(&d_inp, inp_workspace_bytes));
    cudaCheck(cudaMalloc(&d_qkvr, packed_bytes));
    cudaCheck(cudaMalloc(&d_att, att_bytes));
    cudaCheck(cudaMalloc(&d_out, out_bytes));
    cudaCheck(cudaMalloc(&d_dout, out_bytes));
    cudaCheck(cudaMalloc(&d_dinp, packed_bytes));
    cudaCheck(cudaMalloc(&d_dqkvr, packed_bytes));
    if (tk_backward) {
        cudaCheck(cudaMalloc(&d_datt, datt_bytes));
    } else {
        cudaCheck(cudaMalloc(&d_scratch, out_bytes));
    }

    cudaCheck(cudaMemset(d_inp, 0, inp_workspace_bytes));
    cudaCheck(cudaMemcpy(d_inp, h_inp.data(), packed_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, h_dout.data(), out_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_qkvr, 0, packed_bytes));
    cudaCheck(cudaMemset(d_att, 0, att_bytes));
    cudaCheck(cudaMemset(d_out, 0, out_bytes));
    cudaCheck(cudaMemset(d_dinp, 0, packed_bytes));
    cudaCheck(cudaMemset(d_dqkvr, 0, packed_bytes));
    if (d_datt != nullptr) {
        cudaCheck(cudaMemset(d_datt, 0, datt_bytes));
    }
    if (d_scratch != nullptr) {
        cudaCheck(cudaMemset(d_scratch, 0, out_bytes));
    }

#if defined(KITTENS_SM120) && LLMK_USE_TK_MHA
    if (packed_qkv_sm120) {
        attention_forward_packed_qkv(d_out, d_att, d_inp, B, T, C, NH, 0);
    } else
#endif
    {
        attention_forward(d_out, d_qkvr, d_att, d_inp, B, T, C, NH, 0);
    }
    cudaCheck(cudaDeviceSynchronize());

#if defined(KITTENS_SM120) && LLMK_USE_TK_MHA_BWD && !defined(LLMK_SM120_ATOMIC_DQ)
    if (packed_qkv_sm120 && tk_backward) {
        attention_backward_packed_qkv(d_dinp, d_datt, d_out,
                                      d_dout, d_inp, d_att, B, T, C, NH, 0);
    } else
#endif
    {
        attention_backward(d_dinp, d_dqkvr, d_datt, tk_backward ? d_out : d_scratch,
                           d_dout, d_qkvr, d_att, B, T, C, NH, 0);
    }
    cudaCheck(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_out(out_elems);
    std::vector<__nv_bfloat16> h_dinp(packed_elems);
    cudaCheck(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_dinp.data(), d_dinp, packed_bytes, cudaMemcpyDeviceToHost));

    double fwd_diff = max_abs_diff_bf16_float(h_out, ref_out);
    double bwd_diff = max_abs_diff_bf16_float(h_dinp, ref_dinp);
    double fwd_tol = 0.08;
    double bwd_tol = 0.20;
    printf("forward max abs diff  = %.6f (tol %.3f) %s\n",
           fwd_diff, fwd_tol, fwd_diff <= fwd_tol ? "PASS" : "FAIL");
    printf("backward max abs diff = %.6f (tol %.3f) %s\n",
           bwd_diff, bwd_tol, bwd_diff <= bwd_tol ? "PASS" : "FAIL");

    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_qkvr));
    cudaCheck(cudaFree(d_att));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_dqkvr));
    if (d_datt != nullptr) {
        cudaCheck(cudaFree(d_datt));
    }
    if (d_scratch != nullptr) {
        cudaCheck(cudaFree(d_scratch));
    }

    return (fwd_diff <= fwd_tol && bwd_diff <= bwd_tol) ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main() {
    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Device: %s (sm_%d%d)\n", deviceProp.name, deviceProp.major, deviceProp.minor);
    if (deviceProp.major != 9) {
        printf("warning: this smoke test targets H100 (sm_90a); continuing anyway\n");
    }

    int failures = 0;
    failures += run_case(192, false, 0);
    failures += run_case(256, true, 10000);
#if defined(KITTENS_SM120) && LLMK_USE_TK_MHA_BWD && !defined(LLMK_SM120_ATOMIC_DQ)
    failures += run_case(256, true, 20000, true);
#endif
    if (failures == 0) {
        printf("test_attention smoke OK\n");
    }
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
