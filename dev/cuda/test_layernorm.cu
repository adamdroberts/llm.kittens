/*
test_layernorm.cu — smoke test for GPT-style LayerNorm forward/fused/backward.

Runs llmc/layernorm.cuh over a small GPT hidden width and compares LayerNorm
forward, fused residual+LayerNorm forward, and backward accumulation against
independent CPU references.

Build via the Makefile target:

    make test_layernorm
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

#include "llmc/layernorm.cuh"

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

static void cpu_layernorm_forward(std::vector<float>& out,
                                  std::vector<float>& mean,
                                  std::vector<float>& rstd,
                                  const std::vector<__nv_bfloat16>& inp,
                                  const std::vector<__nv_bfloat16>& weight,
                                  const std::vector<__nv_bfloat16>& bias,
                                  int N, int C) {
    out.assign((size_t)N * C, 0.0f);
    mean.assign(N, 0.0f);
    rstd.assign(N, 0.0f);

    for (int row = 0; row < N; ++row) {
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            sum += bf16_to_float(inp[(size_t)row * C + c]);
        }
        float m = sum / (float)C;

        float sumsq = 0.0f;
        for (int c = 0; c < C; ++c) {
            float x = bf16_to_float(inp[(size_t)row * C + c]);
            float d = x - m;
            sumsq += d * d;
        }
        float s = rsqrtf(sumsq / (float)C + 1e-5f);
        mean[row] = m;
        rstd[row] = s;

        for (int c = 0; c < C; ++c) {
            size_t idx = (size_t)row * C + c;
            float norm = (bf16_to_float(inp[idx]) - m) * s;
            out[idx] = norm * bf16_to_float(weight[c]) + bf16_to_float(bias[c]);
        }
    }
}

static void cpu_fused_residual_layernorm_forward(
    std::vector<float>& residual,
    std::vector<float>& normed,
    std::vector<float>& mean,
    std::vector<float>& rstd,
    const std::vector<__nv_bfloat16>& inp1,
    const std::vector<__nv_bfloat16>& inp2,
    const std::vector<__nv_bfloat16>& weight,
    const std::vector<__nv_bfloat16>& bias,
    int N, int C
) {
    residual.assign((size_t)N * C, 0.0f);
    std::vector<__nv_bfloat16> residual_bf16((size_t)N * C);
    for (int row = 0; row < N; ++row) {
        for (int c = 0; c < C; ++c) {
            size_t idx = (size_t)row * C + c;
            residual[idx] = bf16_to_float(inp1[idx]) + bf16_to_float(inp2[idx]);
            residual_bf16[idx] = float_to_bf16(residual[idx]);
        }
    }
    cpu_layernorm_forward(normed, mean, rstd, residual_bf16, weight, bias, N, C);
}

static void cpu_layernorm_backward(
    std::vector<float>& dinp,
    std::vector<float>& dweight,
    std::vector<float>& dbias,
    const std::vector<__nv_bfloat16>& base_dinp,
    const std::vector<__nv_bfloat16>& base_dweight,
    const std::vector<__nv_bfloat16>& base_dbias,
    const std::vector<__nv_bfloat16>& dout,
    const std::vector<__nv_bfloat16>& inp,
    const std::vector<__nv_bfloat16>& weight,
    const std::vector<float>& mean,
    const std::vector<float>& rstd,
    int N, int C
) {
    dinp.assign((size_t)N * C, 0.0f);
    dweight.assign(C, 0.0f);
    dbias.assign(C, 0.0f);

    for (size_t i = 0; i < dinp.size(); ++i) {
        dinp[i] = bf16_to_float(base_dinp[i]);
    }
    for (int c = 0; c < C; ++c) {
        dweight[c] = bf16_to_float(base_dweight[c]);
        dbias[c] = bf16_to_float(base_dbias[c]);
    }

    for (int row = 0; row < N; ++row) {
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int c = 0; c < C; ++c) {
            size_t idx = (size_t)row * C + c;
            float dnorm = bf16_to_float(dout[idx]) * bf16_to_float(weight[c]);
            float norm = (bf16_to_float(inp[idx]) - mean[row]) * rstd[row];
            dnorm_mean += dnorm;
            dnorm_norm_mean += dnorm * norm;
        }
        dnorm_mean /= (float)C;
        dnorm_norm_mean /= (float)C;

        for (int c = 0; c < C; ++c) {
            size_t idx = (size_t)row * C + c;
            float x = bf16_to_float(inp[idx]);
            float norm = (x - mean[row]) * rstd[row];
            float dnorm = bf16_to_float(dout[idx]) * bf16_to_float(weight[c]);
            float dval = (dnorm - dnorm_mean - norm * dnorm_norm_mean) * rstd[row];
            dinp[idx] += dval;
            dweight[c] += bf16_to_float(dout[idx]) * norm;
            dbias[c] += bf16_to_float(dout[idx]);
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

static double max_abs_diff_float(const std::vector<float>& actual,
                                 const std::vector<float>& expected) {
    double max_diff = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        double diff = std::abs((double)actual[i] - (double)expected[i]);
        max_diff = std::max(max_diff, diff);
    }
    return max_diff;
}

int main() {
    constexpr int N = 4;
    constexpr int C = 768;
    size_t elems = (size_t)N * C;
    size_t bytes = elems * sizeof(__nv_bfloat16);
    size_t row_bytes = (size_t)N * sizeof(float);
    size_t weight_bytes = (size_t)C * sizeof(__nv_bfloat16);

    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Device: %s (sm_%d%d)\n", deviceProp.name, deviceProp.major, deviceProp.minor);
    if (deviceProp.major != 9) {
        printf("warning: this smoke test targets H100 (sm_90a); continuing anyway\n");
    }
    printf("Shape: N=%d C=%d\n", N, C);

    std::vector<__nv_bfloat16> h_inp(elems);
    std::vector<__nv_bfloat16> h_skip(elems);
    std::vector<__nv_bfloat16> h_weight(C);
    std::vector<__nv_bfloat16> h_bias(C);
    std::vector<__nv_bfloat16> h_dout(elems);
    std::vector<__nv_bfloat16> h_base_dinp(elems);
    std::vector<__nv_bfloat16> h_base_dweight(C);
    std::vector<__nv_bfloat16> h_base_dbias(C);
    fill_random_bf16(h_inp, 11, -0.50f, 0.50f);
    fill_random_bf16(h_skip, 22, -0.30f, 0.30f);
    fill_random_bf16(h_weight, 33, 0.50f, 1.50f);
    fill_random_bf16(h_bias, 44, -0.20f, 0.20f);
    fill_random_bf16(h_dout, 55, -0.20f, 0.20f);
    fill_random_bf16(h_base_dinp, 66, -0.05f, 0.05f);
    fill_random_bf16(h_base_dweight, 77, -0.05f, 0.05f);
    fill_random_bf16(h_base_dbias, 88, -0.05f, 0.05f);

    std::vector<float> ref_out, ref_mean, ref_rstd;
    std::vector<float> ref_residual, ref_fused_out, ref_fused_mean, ref_fused_rstd;
    cpu_layernorm_forward(ref_out, ref_mean, ref_rstd, h_inp, h_weight, h_bias, N, C);
    cpu_fused_residual_layernorm_forward(ref_residual, ref_fused_out, ref_fused_mean, ref_fused_rstd,
                                         h_inp, h_skip, h_weight, h_bias, N, C);

    __nv_bfloat16* d_inp = nullptr;
    __nv_bfloat16* d_skip = nullptr;
    __nv_bfloat16* d_weight = nullptr;
    __nv_bfloat16* d_bias = nullptr;
    __nv_bfloat16* d_out = nullptr;
    __nv_bfloat16* d_residual = nullptr;
    __nv_bfloat16* d_fused_out = nullptr;
    __nv_bfloat16* d_dout = nullptr;
    __nv_bfloat16* d_dinp = nullptr;
    __nv_bfloat16* d_dweight = nullptr;
    __nv_bfloat16* d_dbias = nullptr;
    float* d_mean = nullptr;
    float* d_rstd = nullptr;
    float* d_fused_mean = nullptr;
    float* d_fused_rstd = nullptr;
    float* d_scratch = nullptr;

    size_t scratch_floats = 32 + 2 * (size_t)C * (2 * (size_t)deviceProp.multiProcessorCount);
    size_t scratch_bytes = scratch_floats * sizeof(float);

    cudaCheck(cudaMalloc(&d_inp, bytes));
    cudaCheck(cudaMalloc(&d_skip, bytes));
    cudaCheck(cudaMalloc(&d_weight, weight_bytes));
    cudaCheck(cudaMalloc(&d_bias, weight_bytes));
    cudaCheck(cudaMalloc(&d_out, bytes));
    cudaCheck(cudaMalloc(&d_residual, bytes));
    cudaCheck(cudaMalloc(&d_fused_out, bytes));
    cudaCheck(cudaMalloc(&d_dout, bytes));
    cudaCheck(cudaMalloc(&d_dinp, bytes));
    cudaCheck(cudaMalloc(&d_dweight, weight_bytes));
    cudaCheck(cudaMalloc(&d_dbias, weight_bytes));
    cudaCheck(cudaMalloc(&d_mean, row_bytes));
    cudaCheck(cudaMalloc(&d_rstd, row_bytes));
    cudaCheck(cudaMalloc(&d_fused_mean, row_bytes));
    cudaCheck(cudaMalloc(&d_fused_rstd, row_bytes));
    cudaCheck(cudaMalloc(&d_scratch, scratch_bytes));

    cudaCheck(cudaMemcpy(d_inp, h_inp.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_skip, h_skip.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, h_weight.data(), weight_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias, h_bias.data(), weight_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, h_dout.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dinp, h_base_dinp.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dweight, h_base_dweight.data(), weight_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dbias, h_base_dbias.data(), weight_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_out, 0, bytes));
    cudaCheck(cudaMemset(d_residual, 0, bytes));
    cudaCheck(cudaMemset(d_fused_out, 0, bytes));
    cudaCheck(cudaMemset(d_mean, 0, row_bytes));
    cudaCheck(cudaMemset(d_rstd, 0, row_bytes));
    cudaCheck(cudaMemset(d_fused_mean, 0, row_bytes));
    cudaCheck(cudaMemset(d_fused_rstd, 0, row_bytes));
    cudaCheck(cudaMemset(d_scratch, 0, scratch_bytes));

    layernorm_forward(d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, 1, N, C, 0);
    fused_residual_forward5(d_residual, d_fused_out, d_fused_mean, d_fused_rstd,
                            d_inp, d_skip, d_weight, d_bias, N, C, 0);
    layernorm_backward(d_dinp, d_dweight, d_dbias, d_scratch, d_dout,
                       d_inp, d_weight, d_mean, d_rstd, 1, N, C, 0);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_out(elems);
    std::vector<__nv_bfloat16> h_residual(elems);
    std::vector<__nv_bfloat16> h_fused_out(elems);
    std::vector<__nv_bfloat16> h_dinp(elems);
    std::vector<__nv_bfloat16> h_dweight(C);
    std::vector<__nv_bfloat16> h_dbias(C);
    std::vector<float> h_mean(N);
    std::vector<float> h_rstd(N);
    std::vector<float> h_fused_mean(N);
    std::vector<float> h_fused_rstd(N);
    cudaCheck(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_residual.data(), d_residual, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_fused_out.data(), d_fused_out, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_dinp.data(), d_dinp, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_dweight.data(), d_dweight, weight_bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_dbias.data(), d_dbias, weight_bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_mean.data(), d_mean, row_bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_rstd.data(), d_rstd, row_bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_fused_mean.data(), d_fused_mean, row_bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_fused_rstd.data(), d_fused_rstd, row_bytes, cudaMemcpyDeviceToHost));

    std::vector<float> ref_dinp, ref_dweight, ref_dbias;
    cpu_layernorm_backward(ref_dinp, ref_dweight, ref_dbias,
                           h_base_dinp, h_base_dweight, h_base_dbias,
                           h_dout, h_inp, h_weight, h_mean, h_rstd, N, C);

    double out_diff = max_abs_diff_bf16_float(h_out, ref_out);
    double mean_diff = max_abs_diff_float(h_mean, ref_mean);
    double rstd_diff = max_abs_diff_float(h_rstd, ref_rstd);
    double residual_diff = max_abs_diff_bf16_float(h_residual, ref_residual);
    double fused_out_diff = max_abs_diff_bf16_float(h_fused_out, ref_fused_out);
    double fused_mean_diff = max_abs_diff_float(h_fused_mean, ref_fused_mean);
    double fused_rstd_diff = max_abs_diff_float(h_fused_rstd, ref_fused_rstd);
    double dinp_diff = max_abs_diff_bf16_float(h_dinp, ref_dinp);
    double dweight_diff = max_abs_diff_bf16_float(h_dweight, ref_dweight);
    double dbias_diff = max_abs_diff_bf16_float(h_dbias, ref_dbias);

    double bf16_tol = 0.08;
    double stat_tol = 0.05;
    double grad_tol = 0.12;
    printf("forward out max abs diff       = %.6f (tol %.3f) %s\n",
           out_diff, bf16_tol, out_diff <= bf16_tol ? "PASS" : "FAIL");
    printf("forward mean max abs diff      = %.6f (tol %.3f) %s\n",
           mean_diff, stat_tol, mean_diff <= stat_tol ? "PASS" : "FAIL");
    printf("forward rstd max abs diff      = %.6f (tol %.3f) %s\n",
           rstd_diff, stat_tol, rstd_diff <= stat_tol ? "PASS" : "FAIL");
    printf("fused residual max abs diff    = %.6f (tol %.3f) %s\n",
           residual_diff, bf16_tol, residual_diff <= bf16_tol ? "PASS" : "FAIL");
    printf("fused out max abs diff         = %.6f (tol %.3f) %s\n",
           fused_out_diff, bf16_tol, fused_out_diff <= bf16_tol ? "PASS" : "FAIL");
    printf("fused mean max abs diff        = %.6f (tol %.3f) %s\n",
           fused_mean_diff, stat_tol, fused_mean_diff <= stat_tol ? "PASS" : "FAIL");
    printf("fused rstd max abs diff        = %.6f (tol %.3f) %s\n",
           fused_rstd_diff, stat_tol, fused_rstd_diff <= stat_tol ? "PASS" : "FAIL");
    printf("backward dinp max abs diff     = %.6f (tol %.3f) %s\n",
           dinp_diff, grad_tol, dinp_diff <= grad_tol ? "PASS" : "FAIL");
    printf("backward dweight max abs diff  = %.6f (tol %.3f) %s\n",
           dweight_diff, grad_tol, dweight_diff <= grad_tol ? "PASS" : "FAIL");
    printf("backward dbias max abs diff    = %.6f (tol %.3f) %s\n",
           dbias_diff, grad_tol, dbias_diff <= grad_tol ? "PASS" : "FAIL");

    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_skip));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_bias));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_residual));
    cudaCheck(cudaFree(d_fused_out));
    cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_dweight));
    cudaCheck(cudaFree(d_dbias));
    cudaCheck(cudaFree(d_mean));
    cudaCheck(cudaFree(d_rstd));
    cudaCheck(cudaFree(d_fused_mean));
    cudaCheck(cudaFree(d_fused_rstd));
    cudaCheck(cudaFree(d_scratch));

    bool ok = out_diff <= bf16_tol && mean_diff <= stat_tol &&
              rstd_diff <= stat_tol && residual_diff <= bf16_tol &&
              fused_out_diff <= bf16_tol && fused_mean_diff <= stat_tol &&
              fused_rstd_diff <= stat_tol && dinp_diff <= grad_tol &&
              dweight_diff <= grad_tol && dbias_diff <= grad_tol;
    if (ok) {
        printf("test_layernorm smoke OK\n");
    }
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
