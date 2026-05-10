/*
test_swiglu.cu — smoke test for Llama-3 SwiGLU forward/backward.

Runs llmc/swiglu.cuh over a small aligned bf16 vector and compares forward,
dgate, and dup against independent CPU references.

Build via the Makefile target:

    make test_swiglu
*/
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

cudaDeviceProp deviceProp;

#include "llmc/swiglu.cuh"

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

static float cpu_silu(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return x * sig;
}

static float cpu_dsilu(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return sig * (1.0f + x * (1.0f - sig));
}

static void cpu_swiglu_forward(std::vector<float>& out,
                               const std::vector<__nv_bfloat16>& gate,
                               const std::vector<__nv_bfloat16>& up) {
    out.assign(gate.size(), 0.0f);
    for (size_t i = 0; i < gate.size(); ++i) {
        out[i] = cpu_silu(bf16_to_float(gate[i])) * bf16_to_float(up[i]);
    }
}

static void cpu_swiglu_backward(std::vector<float>& dgate, std::vector<float>& dup,
                                const std::vector<__nv_bfloat16>& dout,
                                const std::vector<__nv_bfloat16>& gate,
                                const std::vector<__nv_bfloat16>& up) {
    dgate.assign(gate.size(), 0.0f);
    dup.assign(gate.size(), 0.0f);
    for (size_t i = 0; i < gate.size(); ++i) {
        float d = bf16_to_float(dout[i]);
        float g = bf16_to_float(gate[i]);
        float u = bf16_to_float(up[i]);
        dgate[i] = d * u * cpu_dsilu(g);
        dup[i] = d * cpu_silu(g);
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

int main() {
    constexpr int N = 4096;
    size_t bytes = (size_t)N * sizeof(__nv_bfloat16);

    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Device: %s (sm_%d%d)\n", deviceProp.name, deviceProp.major, deviceProp.minor);
    const bool blackwell = deviceProp.major == 10 && (deviceProp.minor == 0 || deviceProp.minor == 3);
    const bool rtx5090 = deviceProp.major == 12 && deviceProp.minor == 0
        && std::strstr(deviceProp.name, "RTX 5090") != nullptr;
    if (deviceProp.major != 9 && !blackwell && !rtx5090) {
        printf("warning: this plain CUDA smoke test is validated for H100 and Blackwell targets; continuing anyway\n");
    }
    printf("Shape: N=%d\n", N);

    std::vector<__nv_bfloat16> h_gate(N);
    std::vector<__nv_bfloat16> h_up(N);
    std::vector<__nv_bfloat16> h_dout(N);
    fill_random_bf16(h_gate, 123, -1.0f, 1.0f);
    fill_random_bf16(h_up, 456, -1.0f, 1.0f);
    fill_random_bf16(h_dout, 789, -0.5f, 0.5f);

    std::vector<float> ref_out, ref_dgate, ref_dup;
    cpu_swiglu_forward(ref_out, h_gate, h_up);
    cpu_swiglu_backward(ref_dgate, ref_dup, h_dout, h_gate, h_up);

    __nv_bfloat16* d_gate = nullptr;
    __nv_bfloat16* d_up = nullptr;
    __nv_bfloat16* d_dout = nullptr;
    __nv_bfloat16* d_out = nullptr;
    __nv_bfloat16* d_dgate = nullptr;
    __nv_bfloat16* d_dup = nullptr;
    cudaCheck(cudaMalloc(&d_gate, bytes));
    cudaCheck(cudaMalloc(&d_up, bytes));
    cudaCheck(cudaMalloc(&d_dout, bytes));
    cudaCheck(cudaMalloc(&d_out, bytes));
    cudaCheck(cudaMalloc(&d_dgate, bytes));
    cudaCheck(cudaMalloc(&d_dup, bytes));

    cudaCheck(cudaMemcpy(d_gate, h_gate.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_up, h_up.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, h_dout.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_out, 0, bytes));
    cudaCheck(cudaMemset(d_dgate, 0, bytes));
    cudaCheck(cudaMemset(d_dup, 0, bytes));

    swiglu_forward(d_out, d_gate, d_up, N, 0);
    swiglu_backward(d_dgate, d_dup, d_dout, d_gate, d_up, N, 0);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_out(N);
    std::vector<__nv_bfloat16> h_dgate(N);
    std::vector<__nv_bfloat16> h_dup(N);
    cudaCheck(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_dgate.data(), d_dgate, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_dup.data(), d_dup, bytes, cudaMemcpyDeviceToHost));

    double out_diff = max_abs_diff_bf16_float(h_out, ref_out);
    double dgate_diff = max_abs_diff_bf16_float(h_dgate, ref_dgate);
    double dup_diff = max_abs_diff_bf16_float(h_dup, ref_dup);
    double tolerance = 0.02;
    printf("forward max abs diff = %.6f (tol %.3f) %s\n",
           out_diff, tolerance, out_diff <= tolerance ? "PASS" : "FAIL");
    printf("dgate max abs diff   = %.6f (tol %.3f) %s\n",
           dgate_diff, tolerance, dgate_diff <= tolerance ? "PASS" : "FAIL");
    printf("dup max abs diff     = %.6f (tol %.3f) %s\n",
           dup_diff, tolerance, dup_diff <= tolerance ? "PASS" : "FAIL");

    cudaCheck(cudaFree(d_gate));
    cudaCheck(cudaFree(d_up));
    cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_dgate));
    cudaCheck(cudaFree(d_dup));

    bool ok = out_diff <= tolerance && dgate_diff <= tolerance && dup_diff <= tolerance;
    if (ok) {
        printf("test_swiglu smoke OK\n");
    }
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
