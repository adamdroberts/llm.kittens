/*
test_gelu.cu — smoke test for llmc/gelu.cuh.

Runs gelu_forward and gelu_backward_inplace on a bf16 buffer that satisfies the
kernel launchers' alignment requirements (N % (512 * x128::size) == 0 for fwd,
N % (128 * x128::size) == 0 for bwd) and compares against an independent CPU
reference computed in float.

Build via the Makefile target:

    make test_gelu
*/
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

cudaDeviceProp deviceProp;

#include "llmc/gelu.cuh"

static float bf16_to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
static __nv_bfloat16 float_to_bf16(float x) { return __float2bfloat16(x); }

static void fill_random_bf16(std::vector<__nv_bfloat16>& h, uint64_t seed,
                             float lo, float hi) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& v : h) v = float_to_bf16(dist(rng));
}

// Matches gelu_forward_kernel2 / approximate-tanh GELU.
static float cpu_gelu(float x) {
    constexpr float k = 0.7978845608028654f;  // sqrt(2/pi)
    float cube = 0.044715f * x * x * x;
    return 0.5f * x * (1.0f + tanhf(k * (x + cube)));
}

// d/dx of approximate-tanh GELU, matching gelu_backward_inplace_kernel.
static float cpu_dgelu(float x) {
    constexpr float k = 0.7978845608028654f;  // sqrt(2/pi)
    float cube = 0.044715f * x * x * x;
    float arg = k * (x + cube);
    float t = tanhf(arg);
    float c = coshf(arg);
    float sech2 = 1.0f / (c * c);
    return 0.5f * (1.0f + t) + x * 0.5f * sech2 * k * (1.0f + 3.0f * 0.044715f * x * x);
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

int main() {
    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Device: %s (sm_%d%d)\n", deviceProp.name, deviceProp.major, deviceProp.minor);

    constexpr int N = 8192;  // multiple of 512*8 (fwd) and 128*8 (bwd) for bf16 x128
    size_t bytes = (size_t)N * sizeof(__nv_bfloat16);
    printf("Shape: N=%d\n", N);

    std::vector<__nv_bfloat16> h_inp(N);
    std::vector<__nv_bfloat16> h_dout(N);
    fill_random_bf16(h_inp, 11, -3.0f, 3.0f);
    fill_random_bf16(h_dout, 22, -1.0f, 1.0f);

    std::vector<float> ref_out(N), ref_dinp(N);
    for (int i = 0; i < N; ++i) {
        float x = bf16_to_float(h_inp[i]);
        ref_out[i] = cpu_gelu(x);
        ref_dinp[i] = cpu_dgelu(x) * bf16_to_float(h_dout[i]);
    }

    __nv_bfloat16 *d_inp = nullptr, *d_out = nullptr, *d_dinout = nullptr;
    cudaCheck(cudaMalloc(&d_inp, bytes));
    cudaCheck(cudaMalloc(&d_out, bytes));
    cudaCheck(cudaMalloc(&d_dinout, bytes));

    cudaCheck(cudaMemcpy(d_inp, h_inp.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_out, 0, bytes));
    cudaCheck(cudaMemcpy(d_dinout, h_dout.data(), bytes, cudaMemcpyHostToDevice));

    gelu_forward(d_out, d_inp, N, 0);
    gelu_backward_inplace(d_dinout, d_inp, N, 0);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_out(N), h_dinp(N);
    cudaCheck(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_dinp.data(), d_dinout, bytes, cudaMemcpyDeviceToHost));

    double fwd_diff = max_abs_diff_bf16_float(h_out, ref_out);
    double bwd_diff = max_abs_diff_bf16_float(h_dinp, ref_dinp);
    double tol = 0.02;  // tanh-approx GELU bf16; in [-3,3] this is plenty of slack
    printf("forward  max abs diff = %.6f (tol %.3f) %s\n", fwd_diff, tol, fwd_diff <= tol ? "PASS" : "FAIL");
    printf("backward max abs diff = %.6f (tol %.3f) %s\n", bwd_diff, tol, bwd_diff <= tol ? "PASS" : "FAIL");

    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_dinout));

    bool ok = fwd_diff <= tol && bwd_diff <= tol;
    if (ok) printf("test_gelu smoke OK\n");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
