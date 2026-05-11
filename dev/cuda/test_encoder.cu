/*
test_encoder.cu — smoke test for llmc/encoder.cuh forward path.

Builds a small (B,T,C) embedding setup with random token IDs, calls
encoder_forward, and compares the gathered wte[ix] + wpe[t] sum against a CPU
reference. Backward (wte_backward + wpe_backward) is exercised by the larger
gpt2_validate / test_gpt2cu binaries; this smoke focuses on the forward path
since the backward bf16 stochastic-rounding write makes a strict per-element
comparison non-deterministic.

Build via the Makefile target:

    make test_encoder
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

#include "llmc/encoder.cuh"

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

int main() {
    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Device: %s (sm_%d%d)\n", deviceProp.name, deviceProp.major, deviceProp.minor);

    constexpr int B = 2, T = 64, C = 128, V = 256;  // C is multiple of x128::size=8
    printf("Shape: B=%d T=%d C=%d V=%d\n", B, T, C, V);

    std::vector<int> h_inp(B * T);
    {
        std::mt19937_64 rng(7);
        std::uniform_int_distribution<int> dist(0, V - 1);
        for (auto& v : h_inp) v = dist(rng);
    }

    std::vector<__nv_bfloat16> h_wte((size_t)V * C), h_wpe((size_t)T * C);
    fill_random_bf16(h_wte, 101, -0.1f, 0.1f);
    fill_random_bf16(h_wpe, 202, -0.05f, 0.05f);

    std::vector<float> ref_out((size_t)B * T * C, 0.0f);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int ix = h_inp[b * T + t];
            for (int c = 0; c < C; ++c) {
                ref_out[(size_t)b * T * C + (size_t)t * C + c] =
                    bf16_to_float(h_wte[(size_t)ix * C + c]) +
                    bf16_to_float(h_wpe[(size_t)t * C + c]);
            }
        }
    }

    int* d_inp = nullptr;
    __nv_bfloat16 *d_wte = nullptr, *d_wpe = nullptr, *d_out = nullptr;
    size_t out_bytes = (size_t)B * T * C * sizeof(__nv_bfloat16);
    cudaCheck(cudaMalloc(&d_inp, (size_t)B * T * sizeof(int)));
    cudaCheck(cudaMalloc(&d_wte, (size_t)V * C * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_wpe, (size_t)T * C * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_out, out_bytes));

    cudaCheck(cudaMemcpy(d_inp, h_inp.data(), (size_t)B * T * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_wte, h_wte.data(), (size_t)V * C * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_wpe, h_wpe.data(), (size_t)T * C * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_out, 0, out_bytes));

    encoder_forward(d_out, d_inp, d_wte, d_wpe, B, T, C, 0);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_out((size_t)B * T * C);
    cudaCheck(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

    double diff = max_abs_diff_bf16_float(h_out, ref_out);
    double tol = 0.01;  // single bf16 add of two small values; rounding only
    printf("forward max abs diff = %.6f (tol %.3f) %s\n", diff, tol, diff <= tol ? "PASS" : "FAIL");

    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_wte));
    cudaCheck(cudaFree(d_wpe));
    cudaCheck(cudaFree(d_out));

    bool ok = diff <= tol;
    if (ok) printf("test_encoder smoke OK\n");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
