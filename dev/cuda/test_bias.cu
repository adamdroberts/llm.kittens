/*
test_bias.cu - smoke test for standalone bias add and bias-gradient reduction.

Build via the Makefile target:

    make test_bias
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

#include "llmc/matmul.cuh"

static float bf16_to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
static __nv_bfloat16 float_to_bf16(float x) { return __float2bfloat16(x); }

static void fill_random_bf16(std::vector<__nv_bfloat16>& h, uint64_t seed,
                             float lo, float hi) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& v : h) v = float_to_bf16(dist(rng));
}

static double max_abs_diff_bf16_float(const std::vector<__nv_bfloat16>& actual,
                                      const std::vector<float>& expected) {
    double m = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        double d = std::abs((double)bf16_to_float(actual[i]) - (double)expected[i]);
        if (d > m) m = d;
    }
    return m;
}

static bool test_add_bias_shape(int N, int OC, const char* label) {
    printf("Bias add shape: N=%d OC=%d\n", N, OC);

    std::vector<__nv_bfloat16> h_out((size_t)N * OC);
    std::vector<__nv_bfloat16> h_bias(OC);
    fill_random_bf16(h_out, 1001, -1.0f, 1.0f);
    fill_random_bf16(h_bias, 1002, -0.5f, 0.5f);

    std::vector<float> ref((size_t)N * OC);
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < OC; ++oc) {
            ref[(size_t)n * OC + oc] = bf16_to_float(h_out[(size_t)n * OC + oc])
                                     + bf16_to_float(h_bias[oc]);
        }
    }

    __nv_bfloat16 *d_out = nullptr, *d_bias = nullptr;
    cudaCheck(cudaMalloc(&d_out, h_out.size() * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_bias, h_bias.size() * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMemcpy(d_out, h_out.data(), h_out.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    add_bias(d_out, d_bias, N, OC, 0);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> got(h_out.size());
    cudaCheck(cudaMemcpy(got.data(), d_out, got.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_bias));

    double diff = max_abs_diff_bf16_float(got, ref);
    double tol = 0.01;
    printf("bias add %s max abs diff = %.6f (tol %.3f) %s\n",
           label, diff, tol, diff <= tol ? "PASS" : "FAIL");
    return diff <= tol;
}

static bool test_add_bias() {
    bool ok = test_add_bias_shape(128, 768, "hidden aligned");
    ok = test_add_bias_shape(64, 3072, "mlp aligned") && ok;
    ok = test_add_bias_shape(17, 770, "unaligned fallback") && ok;
    return ok;
}

static bool test_bias_grad_reduce() {
    constexpr int B = 2;
    constexpr int T = 128;
    constexpr int OC = 768;
    constexpr int BT = B * T;
    printf("Bias grad shape: B=%d T=%d OC=%d\n", B, T, OC);

    std::vector<__nv_bfloat16> h_dout((size_t)BT * OC);
    fill_random_bf16(h_dout, 2001, -1.0f, 1.0f);

    std::vector<float> ref(OC, 0.0f);
    for (int i = 0; i < BT; ++i) {
        for (int oc = 0; oc < OC; ++oc) {
            ref[oc] += bf16_to_float(h_dout[(size_t)i * OC + oc]);
        }
    }

    __nv_bfloat16 *d_dout = nullptr, *d_dbias = nullptr;
    float* d_buffer = nullptr;
    cudaCheck(cudaMalloc(&d_dout, h_dout.size() * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_dbias, (size_t)OC * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_buffer, (size_t)OC * 1024 * sizeof(float)));
    cudaCheck(cudaMemcpy(d_dout, h_dout.data(), h_dout.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_dbias, 0, (size_t)OC * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMemset(d_buffer, 0, (size_t)OC * 1024 * sizeof(float)));

    matmul_backward_bias(d_dbias, d_dout, d_buffer, B, T, OC, 0);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> got(OC);
    cudaCheck(cudaMemcpy(got.data(), d_dbias, (size_t)OC * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_dbias));
    cudaCheck(cudaFree(d_buffer));

    double diff = max_abs_diff_bf16_float(got, ref);
    double tol = 0.25;
    printf("bias grad max abs diff = %.6f (tol %.2f) %s\n",
           diff, tol, diff <= tol ? "PASS" : "FAIL");
    return diff <= tol;
}

int main() {
    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Device: %s (sm_%d%d)\n", deviceProp.name, deviceProp.major, deviceProp.minor);

    bool ok = test_add_bias();
    ok = test_bias_grad_reduce() && ok;
    if (ok) printf("test_bias smoke OK\n");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
