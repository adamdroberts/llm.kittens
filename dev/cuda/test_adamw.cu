/*
test_adamw.cu — smoke test for llmc/adamw.cuh.

Initialises a small parameter buffer, gradient buffer, and optimizer state, then
runs adamw_update for a single step. Compares the post-step master_params (FP32,
bit-exact path), m, and v against a scalar AdamW CPU reference. The bf16
params_memory write goes through stochastic rounding so we don't compare it
strictly — we only require it to be within ~1 LSB of the master copy.

Build via the Makefile target:

    make test_adamw
*/
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

cudaDeviceProp deviceProp;

#include "llmc/adamw.cuh"

static float bf16_to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
static __nv_bfloat16 float_to_bf16(float x) { return __float2bfloat16(x); }

static double max_abs_diff_float(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = std::abs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main() {
    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Device: %s (sm_%d%d)\n", deviceProp.name, deviceProp.major, deviceProp.minor);

    constexpr size_t N = 4096;
    constexpr float lr = 1e-3f;
    constexpr float beta1 = 0.9f, beta2 = 0.95f;
    constexpr float eps = 1e-8f, wd = 0.1f;
    constexpr float grad_scale = 1.0f;
    constexpr int t = 1;
    constexpr unsigned int seed = 0xCAFEBABE;
    printf("Shape: N=%zu lr=%g beta1=%g beta2=%g eps=%g wd=%g t=%d\n",
           N, lr, beta1, beta2, eps, wd, t);

    std::mt19937_64 rng(99);
    std::uniform_real_distribution<float> wdist(-0.1f, 0.1f);
    std::uniform_real_distribution<float> gdist(-0.05f, 0.05f);

    std::vector<float> h_master(N), h_grad(N), h_m(N, 0.0f), h_v(N, 0.0f);
    std::vector<__nv_bfloat16> h_param(N), h_grad_bf16(N);
    for (size_t i = 0; i < N; ++i) {
        h_master[i] = wdist(rng);
        h_param[i] = float_to_bf16(h_master[i]);
        h_grad[i] = gdist(rng);
        h_grad_bf16[i] = float_to_bf16(h_grad[i]);
    }

    // CPU reference: scalar AdamW step, matching the CUDA kernel's math exactly
    // (see llmc::llmc_lerp; lerp(start, end, weight) = start + weight*(end-start),
    // and adamw_update calls lerp(grad, m, beta1) which yields beta1*m + (1-beta1)*grad).
    auto lerp = [](float start, float end, float w) { return start + w * (end - start); };
    float bc1 = 1.0f - powf(beta1, (float)t);
    float bc2 = 1.0f - powf(beta2, (float)t);
    std::vector<float> ref_master(N), ref_master_nomaster(N), ref_m(N), ref_v(N);
    for (size_t i = 0; i < N; ++i) {
        // GPU reads `grads_memory` as bf16 then casts to float; mirror that.
        float g = grad_scale * bf16_to_float(h_grad_bf16[i]);
        float mi = lerp(g, 0.0f, beta1);
        float vi = lerp(g * g, 0.0f, beta2);
        ref_m[i] = mi;
        ref_v[i] = vi;
        float mhat = mi / bc1;
        float vhat = vi / bc2;
        float old_p = h_master[i];
        ref_master[i] = old_p - (lr * (mhat / (sqrtf(vhat) + eps) + wd * old_p));
        float old_p_nomaster = bf16_to_float(h_param[i]);
        ref_master_nomaster[i] = old_p_nomaster - (lr * (mhat / (sqrtf(vhat) + eps) + wd * old_p_nomaster));
    }

    __nv_bfloat16* d_param = nullptr;
    float *d_master = nullptr, *d_m = nullptr, *d_v = nullptr;
    __nv_bfloat16* d_grad = nullptr;
    cudaCheck(cudaMalloc(&d_param, N * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_master, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_grad, N * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_m, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_v, N * sizeof(float)));

    cudaCheck(cudaMemcpy(d_param, h_param.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_master, h_master.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_grad, h_grad_bf16.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_m, h_m.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_v, h_v.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    adamw_update(d_param, d_master, d_grad, d_m, d_v, N,
                 /*w_stride=*/0, /*g_stride=*/0, /*s_stride=*/0,
                 /*num_slices=*/1, lr, beta1, beta2, t, eps, wd, grad_scale, seed, 0);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<float> g_master(N), g_m(N), g_v(N);
    std::vector<__nv_bfloat16> g_param(N);
    cudaCheck(cudaMemcpy(g_master.data(), d_master, N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(g_m.data(), d_m, N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(g_v.data(), d_v, N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(g_param.data(), d_param, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    double m_diff = max_abs_diff_float(g_m, ref_m);
    double v_diff = max_abs_diff_float(g_v, ref_v);
    double mp_diff = max_abs_diff_float(g_master, ref_master);

    // BF16 param: stochastic rounding can shift by ~1 ULP from the master value.
    // bf16 rel precision is ~3.9e-3; with masters in roughly [-0.1, 0.1] post-step,
    // a single ULP ≈ 7.8e-4. Use a generous 5e-3 absolute bound.
    double param_diff = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double d = std::abs((double)bf16_to_float(g_param[i]) - (double)g_master[i]);
        if (d > param_diff) param_diff = d;
    }

    double fp_tol = 1e-5;
    double sr_tol = 5e-3;
    printf("master max abs diff = %.3e (tol %.1e) %s\n", mp_diff, fp_tol, mp_diff <= fp_tol ? "PASS" : "FAIL");
    printf("m      max abs diff = %.3e (tol %.1e) %s\n", m_diff, fp_tol, m_diff <= fp_tol ? "PASS" : "FAIL");
    printf("v      max abs diff = %.3e (tol %.1e) %s\n", v_diff, fp_tol, v_diff <= fp_tol ? "PASS" : "FAIL");
    printf("bf16 param vs master max abs diff = %.3e (tol %.1e) %s\n",
           param_diff, sr_tol, param_diff <= sr_tol ? "PASS" : "FAIL");

    cudaCheck(cudaMemcpy(d_param, h_param.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_grad, h_grad_bf16.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_m, 0, N * sizeof(float)));
    cudaCheck(cudaMemset(d_v, 0, N * sizeof(float)));

    adamw_update(d_param, (float*)nullptr, d_grad, d_m, d_v, N,
                 /*w_stride=*/0, /*g_stride=*/0, /*s_stride=*/0,
                 /*num_slices=*/1, lr, beta1, beta2, t, eps, wd, grad_scale, seed, 0);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<float> g_m_nomaster(N), g_v_nomaster(N);
    std::vector<__nv_bfloat16> g_param_nomaster(N);
    cudaCheck(cudaMemcpy(g_m_nomaster.data(), d_m, N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(g_v_nomaster.data(), d_v, N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(g_param_nomaster.data(), d_param, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    double m_nomaster_diff = max_abs_diff_float(g_m_nomaster, ref_m);
    double v_nomaster_diff = max_abs_diff_float(g_v_nomaster, ref_v);
    double param_nomaster_diff = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double d = std::abs((double)bf16_to_float(g_param_nomaster[i]) - (double)ref_master_nomaster[i]);
        if (d > param_nomaster_diff) param_nomaster_diff = d;
    }

    printf("no-master m max abs diff = %.3e (tol %.1e) %s\n",
           m_nomaster_diff, fp_tol, m_nomaster_diff <= fp_tol ? "PASS" : "FAIL");
    printf("no-master v max abs diff = %.3e (tol %.1e) %s\n",
           v_nomaster_diff, fp_tol, v_nomaster_diff <= fp_tol ? "PASS" : "FAIL");
    printf("no-master bf16 param vs ref max abs diff = %.3e (tol %.1e) %s\n",
           param_nomaster_diff, sr_tol, param_nomaster_diff <= sr_tol ? "PASS" : "FAIL");

    cudaCheck(cudaFree(d_param));
    cudaCheck(cudaFree(d_master));
    cudaCheck(cudaFree(d_grad));
    cudaCheck(cudaFree(d_m));
    cudaCheck(cudaFree(d_v));

    bool ok = mp_diff <= fp_tol && m_diff <= fp_tol && v_diff <= fp_tol && param_diff <= sr_tol &&
              m_nomaster_diff <= fp_tol && v_nomaster_diff <= fp_tol && param_nomaster_diff <= sr_tol;
    if (ok) printf("test_adamw smoke OK\n");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
