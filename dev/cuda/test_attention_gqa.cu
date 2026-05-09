/*
test_attention_gqa.cu — smoke test for Llama-3 GQA + RoPE attention.

Generates small packed Llama Q/K/V projections, runs llmc/attention_gqa.cuh,
and compares forward output plus packed backward gradients against an
independent CPU reference. The T=128 case exercises TK GQA forward with the
CUDA backward fallback. The T=256 case also enables the supported-shape TK GQA
backward path.

Build via the Makefile target:

    make test_attention_gqa
*/
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

cudaDeviceProp deviceProp;

#include "llmc/attention_gqa.cuh"

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

static void fill_rope_cache(std::vector<__nv_bfloat16>& cos,
                            std::vector<__nv_bfloat16>& sin,
                            int T, int HS) {
    for (int t = 0; t < T; ++t) {
        for (int d = 0; d < HS / 2; ++d) {
            float angle = 0.0007f * (float)(t + 1) * (float)(d + 1);
            cos[t * (HS / 2) + d] = float_to_bf16(cosf(angle));
            sin[t * (HS / 2) + d] = float_to_bf16(sinf(angle));
        }
    }
}

static inline size_t packed_idx(int b, int t, int h, int d,
                                int T, int qkv_width, int HS) {
    return ((size_t)b * T + t) * qkv_width + (size_t)h * HS + d;
}

static inline size_t perm_idx(int b, int h, int t, int d,
                              int H, int T, int HS) {
    return (((size_t)b * H + h) * T + t) * HS + d;
}

static inline size_t out_idx(int b, int t, int h, int d,
                             int T, int H, int HS) {
    return (((size_t)b * T + t) * H + h) * HS + d;
}

static void rotate_pair(float& x1, float& x2, float c, float s, bool inverse) {
    float y1;
    float y2;
    if (inverse) {
        y1 = x1 * c + x2 * s;
        y2 = x2 * c - x1 * s;
    } else {
        y1 = x1 * c - x2 * s;
        y2 = x2 * c + x1 * s;
    }
    x1 = y1;
    x2 = y2;
}

static void unpack_qkv_with_rope(
    std::vector<float>& q, std::vector<float>& k, std::vector<float>& v,
    const std::vector<__nv_bfloat16>& inp,
    const std::vector<__nv_bfloat16>& cos,
    const std::vector<__nv_bfloat16>& sin,
    int B, int T, int NH, int NKVH, int HS
) {
    int qkv_width = (NH + 2 * NKVH) * HS;
    q.assign((size_t)B * NH * T * HS, 0.0f);
    k.assign((size_t)B * NKVH * T * HS, 0.0f);
    v.assign((size_t)B * NKVH * T * HS, 0.0f);

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < NH; ++h) {
                for (int d = 0; d < HS; ++d) {
                    q[perm_idx(b, h, t, d, NH, T, HS)] =
                        bf16_to_float(inp[packed_idx(b, t, h, d, T, qkv_width, HS)]);
                }
                for (int d = 0; d < HS / 2; ++d) {
                    float c = bf16_to_float(cos[t * (HS / 2) + d]);
                    float s = bf16_to_float(sin[t * (HS / 2) + d]);
                    float& x1 = q[perm_idx(b, h, t, d, NH, T, HS)];
                    float& x2 = q[perm_idx(b, h, t, d + HS / 2, NH, T, HS)];
                    rotate_pair(x1, x2, c, s, false);
                }
            }

            for (int h = 0; h < NKVH; ++h) {
                for (int d = 0; d < HS; ++d) {
                    k[perm_idx(b, h, t, d, NKVH, T, HS)] =
                        bf16_to_float(inp[packed_idx(b, t, NH + h, d, T, qkv_width, HS)]);
                    v[perm_idx(b, h, t, d, NKVH, T, HS)] =
                        bf16_to_float(inp[packed_idx(b, t, NH + NKVH + h, d, T, qkv_width, HS)]);
                }
                for (int d = 0; d < HS / 2; ++d) {
                    float c = bf16_to_float(cos[t * (HS / 2) + d]);
                    float s = bf16_to_float(sin[t * (HS / 2) + d]);
                    float& x1 = k[perm_idx(b, h, t, d, NKVH, T, HS)];
                    float& x2 = k[perm_idx(b, h, t, d + HS / 2, NKVH, T, HS)];
                    rotate_pair(x1, x2, c, s, false);
                }
            }
        }
    }
}

static void cpu_gqa_forward(
    std::vector<float>& out,
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    int B, int T, int NH, int NKVH, int HS
) {
    out.assign((size_t)B * T * NH * HS, 0.0f);
    int nrep = NH / NKVH;
    float scale = 1.0f / sqrtf((float)HS);
    std::vector<float> scores(T);

    for (int b = 0; b < B; ++b) {
        for (int qh = 0; qh < NH; ++qh) {
            int kvh = qh / nrep;
            for (int t = 0; t < T; ++t) {
                float maxval = -INFINITY;
                for (int s = 0; s <= t; ++s) {
                    float score = 0.0f;
                    for (int d = 0; d < HS; ++d) {
                        score += q[perm_idx(b, qh, t, d, NH, T, HS)] *
                                 k[perm_idx(b, kvh, s, d, NKVH, T, HS)];
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
                        acc += p * v[perm_idx(b, kvh, s, d, NKVH, T, HS)];
                    }
                    out[out_idx(b, t, qh, d, T, NH, HS)] = acc;
                }
            }
        }
    }
}

static void cpu_gqa_backward_packed(
    std::vector<float>& dinp,
    const std::vector<float>& dout,
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    const std::vector<__nv_bfloat16>& cos,
    const std::vector<__nv_bfloat16>& sin,
    int B, int T, int NH, int NKVH, int HS
) {
    std::vector<float> dq((size_t)B * NH * T * HS, 0.0f);
    std::vector<float> dk((size_t)B * NKVH * T * HS, 0.0f);
    std::vector<float> dv((size_t)B * NKVH * T * HS, 0.0f);
    std::vector<float> scores(T);
    std::vector<float> probs(T);

    int nrep = NH / NKVH;
    float scale = 1.0f / sqrtf((float)HS);

    for (int b = 0; b < B; ++b) {
        for (int qh = 0; qh < NH; ++qh) {
            int kvh = qh / nrep;
            for (int t = 0; t < T; ++t) {
                float maxval = -INFINITY;
                for (int s = 0; s <= t; ++s) {
                    float score = 0.0f;
                    for (int d = 0; d < HS; ++d) {
                        score += q[perm_idx(b, qh, t, d, NH, T, HS)] *
                                 k[perm_idx(b, kvh, s, d, NKVH, T, HS)];
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
                        dp += dout[out_idx(b, t, qh, d, T, NH, HS)] *
                              v[perm_idx(b, kvh, s, d, NKVH, T, HS)];
                    }
                    dp_norm += probs[s] * dp;
                }

                for (int s = 0; s <= t; ++s) {
                    float dp = 0.0f;
                    for (int d = 0; d < HS; ++d) {
                        dp += dout[out_idx(b, t, qh, d, T, NH, HS)] *
                              v[perm_idx(b, kvh, s, d, NKVH, T, HS)];
                    }
                    float ds = probs[s] * (dp - dp_norm) * scale;
                    for (int d = 0; d < HS; ++d) {
                        dq[perm_idx(b, qh, t, d, NH, T, HS)] +=
                            ds * k[perm_idx(b, kvh, s, d, NKVH, T, HS)];
                        dk[perm_idx(b, kvh, s, d, NKVH, T, HS)] +=
                            ds * q[perm_idx(b, qh, t, d, NH, T, HS)];
                        dv[perm_idx(b, kvh, s, d, NKVH, T, HS)] +=
                            probs[s] * dout[out_idx(b, t, qh, d, T, NH, HS)];
                    }
                }
            }
        }
    }

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < NH; ++h) {
                for (int d = 0; d < HS / 2; ++d) {
                    float c = bf16_to_float(cos[t * (HS / 2) + d]);
                    float s = bf16_to_float(sin[t * (HS / 2) + d]);
                    float& x1 = dq[perm_idx(b, h, t, d, NH, T, HS)];
                    float& x2 = dq[perm_idx(b, h, t, d + HS / 2, NH, T, HS)];
                    rotate_pair(x1, x2, c, s, true);
                }
            }
            for (int h = 0; h < NKVH; ++h) {
                for (int d = 0; d < HS / 2; ++d) {
                    float c = bf16_to_float(cos[t * (HS / 2) + d]);
                    float s = bf16_to_float(sin[t * (HS / 2) + d]);
                    float& x1 = dk[perm_idx(b, h, t, d, NKVH, T, HS)];
                    float& x2 = dk[perm_idx(b, h, t, d + HS / 2, NKVH, T, HS)];
                    rotate_pair(x1, x2, c, s, true);
                }
            }
        }
    }

    int qkv_width = (NH + 2 * NKVH) * HS;
    dinp.assign((size_t)B * T * qkv_width, 0.0f);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < NH; ++h) {
                for (int d = 0; d < HS; ++d) {
                    dinp[packed_idx(b, t, h, d, T, qkv_width, HS)] =
                        dq[perm_idx(b, h, t, d, NH, T, HS)];
                }
            }
            for (int h = 0; h < NKVH; ++h) {
                for (int d = 0; d < HS; ++d) {
                    dinp[packed_idx(b, t, NH + h, d, T, qkv_width, HS)] =
                        dk[perm_idx(b, h, t, d, NKVH, T, HS)];
                    dinp[packed_idx(b, t, NH + NKVH + h, d, T, qkv_width, HS)] =
                        dv[perm_idx(b, h, t, d, NKVH, T, HS)];
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

static int run_case(int T, bool request_tk_backward, uint64_t seed_offset) {
    constexpr int B = 1;
    constexpr int NH = 4;
    constexpr int NKVH = 2;
    constexpr int HS = 128;
    constexpr int C = NH * HS;
    constexpr int QKV_WIDTH = (NH + 2 * NKVH) * HS;

    bool tk_forward = llmk::attention_gqa::has_tk_forward(T, HS, NH, NKVH);
    bool tk_backward = request_tk_backward &&
                       llmk::attention_gqa::has_tk_backward(T, HS, NH, NKVH);
    printf("\nShape: B=%d T=%d NH=%d NKVH=%d HS=%d\n", B, T, NH, NKVH, HS);
    printf("TK forward supported: %s\n", tk_forward ? "yes" : "no");
    printf("TK backward requested/supported: %s/%s\n",
           request_tk_backward ? "yes" : "no",
           tk_backward ? "yes" : "no");

    std::vector<__nv_bfloat16> h_inp((size_t)B * T * QKV_WIDTH);
    std::vector<__nv_bfloat16> h_cos((size_t)T * (HS / 2));
    std::vector<__nv_bfloat16> h_sin((size_t)T * (HS / 2));
    std::vector<__nv_bfloat16> h_dout((size_t)B * T * C);
    fill_random_bf16(h_inp, 1234 + seed_offset, -0.35f, 0.35f);
    fill_random_bf16(h_dout, 5678 + seed_offset, -0.20f, 0.20f);
    fill_rope_cache(h_cos, h_sin, T, HS);

    std::vector<float> q, k, v, ref_out, ref_dinp;
    unpack_qkv_with_rope(q, k, v, h_inp, h_cos, h_sin, B, T, NH, NKVH, HS);
    cpu_gqa_forward(ref_out, q, k, v, B, T, NH, NKVH, HS);

    std::vector<float> h_dout_float(h_dout.size());
    for (size_t i = 0; i < h_dout.size(); ++i) {
        h_dout_float[i] = bf16_to_float(h_dout[i]);
    }
    cpu_gqa_backward_packed(ref_dinp, h_dout_float, q, k, v, h_cos, h_sin,
                            B, T, NH, NKVH, HS);

    size_t inp_bytes = h_inp.size() * sizeof(__nv_bfloat16);
    size_t cos_bytes = h_cos.size() * sizeof(__nv_bfloat16);
    size_t dout_bytes = h_dout.size() * sizeof(__nv_bfloat16);
    size_t qkvr_elems = (size_t)B * T * QKV_WIDTH;
    size_t out_elems = (size_t)B * T * C;
    size_t qkvr_bytes = qkvr_elems * sizeof(__nv_bfloat16);
    size_t out_bytes = out_elems * sizeof(__nv_bfloat16);
    size_t lse_bytes = (size_t)B * NH * T * sizeof(float);
    size_t kv_elems = (size_t)B * NKVH * T * HS;
    size_t bwd_scratch_elems = tk_backward ? 2 * out_elems : out_elems;
    size_t bwd_scratch_bytes = bwd_scratch_elems * sizeof(__nv_bfloat16);
    size_t bwd_float_elems = (size_t)B * NH * T + out_elems + 2 * kv_elems;
    size_t bwd_float_bytes = bwd_float_elems * sizeof(float);

    __nv_bfloat16* d_inp = nullptr;
    __nv_bfloat16* d_cos = nullptr;
    __nv_bfloat16* d_sin = nullptr;
    __nv_bfloat16* d_qkvr = nullptr;
    __nv_bfloat16* d_out = nullptr;
    __nv_bfloat16* d_tk_workspace = nullptr;
    __nv_bfloat16* d_dout = nullptr;
    __nv_bfloat16* d_dinp = nullptr;
    __nv_bfloat16* d_dqkvr = nullptr;
    __nv_bfloat16* d_scratch = nullptr;
    float* d_datt = nullptr;
    float* d_lse = nullptr;

    cudaCheck(cudaMalloc(&d_inp, inp_bytes));
    cudaCheck(cudaMalloc(&d_cos, cos_bytes));
    cudaCheck(cudaMalloc(&d_sin, cos_bytes));
    cudaCheck(cudaMalloc(&d_qkvr, qkvr_bytes));
    cudaCheck(cudaMalloc(&d_out, out_bytes));
    cudaCheck(cudaMalloc(&d_tk_workspace, out_bytes));
    cudaCheck(cudaMalloc(&d_dout, dout_bytes));
    cudaCheck(cudaMalloc(&d_dinp, inp_bytes));
    cudaCheck(cudaMalloc(&d_dqkvr, qkvr_bytes));
    cudaCheck(cudaMalloc(&d_scratch, bwd_scratch_bytes));
    if (tk_backward) {
        cudaCheck(cudaMalloc(&d_datt, bwd_float_bytes));
    }
    cudaCheck(cudaMalloc(&d_lse, lse_bytes));

    cudaCheck(cudaMemcpy(d_inp, h_inp.data(), inp_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_cos, h_cos.data(), cos_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_sin, h_sin.data(), cos_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, h_dout.data(), dout_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_qkvr, 0, qkvr_bytes));
    cudaCheck(cudaMemset(d_out, 0, out_bytes));
    cudaCheck(cudaMemset(d_tk_workspace, 0, out_bytes));
    cudaCheck(cudaMemset(d_dinp, 0, inp_bytes));
    cudaCheck(cudaMemset(d_dqkvr, 0, qkvr_bytes));
    cudaCheck(cudaMemset(d_scratch, 0, bwd_scratch_bytes));
    if (d_datt != nullptr) {
        cudaCheck(cudaMemset(d_datt, 0, bwd_float_bytes));
    }
    cudaCheck(cudaMemset(d_lse, 0, lse_bytes));

    attention_gqa_forward(d_out, d_qkvr, d_lse, d_inp, d_cos, d_sin,
                          B, T, C, NH, NKVH, 0, d_tk_workspace);
    cudaCheck(cudaDeviceSynchronize());

    attention_gqa_backward(d_dinp, d_dqkvr, d_datt, d_scratch, d_dout,
                           d_out, d_qkvr, d_lse, d_cos, d_sin,
                           B, T, C, NH, NKVH, 0,
                           attention_gqa_uses_tk_tile_rope(
                               d_cos, d_sin, d_tk_workspace, T, C, NH, NKVH));
    cudaCheck(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_out(out_elems);
    std::vector<__nv_bfloat16> h_dinp(qkvr_elems);
    cudaCheck(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_dinp.data(), d_dinp, inp_bytes, cudaMemcpyDeviceToHost));

    double fwd_diff = max_abs_diff_bf16_float(h_out, ref_out);
    double bwd_diff = max_abs_diff_bf16_float(h_dinp, ref_dinp);
    double fwd_tol = 0.08;
    double bwd_tol = 0.20;
    printf("forward max abs diff  = %.6f (tol %.3f) %s\n",
           fwd_diff, fwd_tol, fwd_diff <= fwd_tol ? "PASS" : "FAIL");
    printf("backward max abs diff = %.6f (tol %.3f) %s\n",
           bwd_diff, bwd_tol, bwd_diff <= bwd_tol ? "PASS" : "FAIL");

    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_cos));
    cudaCheck(cudaFree(d_sin));
    cudaCheck(cudaFree(d_qkvr));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_tk_workspace));
    cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_dqkvr));
    cudaCheck(cudaFree(d_scratch));
    if (d_datt != nullptr) {
        cudaCheck(cudaFree(d_datt));
    }
    cudaCheck(cudaFree(d_lse));

    if (fwd_diff > fwd_tol || bwd_diff > bwd_tol) {
        return EXIT_FAILURE;
    }
    printf("GQA case T=%d backward=%s OK\n", T, tk_backward ? "tk" : "fallback");
    return EXIT_SUCCESS;
}

int main() {
    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Device: %s (sm_%d%d)\n", deviceProp.name, deviceProp.major, deviceProp.minor);
    if (deviceProp.major != 9) {
        printf("warning: this smoke test targets H100 (sm_90a); continuing anyway\n");
    }

    int failures = 0;
    failures += run_case(128, false, 0);
    failures += run_case(256, true, 10000);
    if (failures == 0) {
        printf("test_attention_gqa smoke OK\n");
    }
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
