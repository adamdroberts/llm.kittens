/*
test_rope.cu — smoke test for Llama-3 RoPE forward/backward.

Runs llmc/rope.cuh over small bf16 tensors and compares forward output plus
inverse-rotation backward output against an independent CPU reference.

Build via the Makefile target:

    make test_rope
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

#include "llmc/rope.cuh"

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
            float angle = 0.0011f * (float)(t + 1) * (float)(d + 1);
            cos[t * (HS / 2) + d] = float_to_bf16(cosf(angle));
            sin[t * (HS / 2) + d] = float_to_bf16(sinf(angle));
        }
    }
}

static inline size_t idx4(int b, int h, int t, int d, int H, int T, int HS) {
    return (((size_t)b * H + h) * T + t) * HS + d;
}

static void cpu_rope(std::vector<float>& out,
                     const std::vector<__nv_bfloat16>& x,
                     const std::vector<__nv_bfloat16>& cos,
                     const std::vector<__nv_bfloat16>& sin,
                     int B, int H, int T, int HS, bool inverse) {
    out.assign((size_t)B * H * T * HS, 0.0f);
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int t = 0; t < T; ++t) {
                for (int d = 0; d < HS / 2; ++d) {
                    float c = bf16_to_float(cos[t * (HS / 2) + d]);
                    float s = bf16_to_float(sin[t * (HS / 2) + d]);
                    float x1 = bf16_to_float(x[idx4(b, h, t, d, H, T, HS)]);
                    float x2 = bf16_to_float(x[idx4(b, h, t, d + HS / 2, H, T, HS)]);
                    float y1;
                    float y2;
                    if (inverse) {
                        y1 = x1 * c + x2 * s;
                        y2 = x2 * c - x1 * s;
                    } else {
                        y1 = x1 * c - x2 * s;
                        y2 = x2 * c + x1 * s;
                    }
                    out[idx4(b, h, t, d, H, T, HS)] = y1;
                    out[idx4(b, h, t, d + HS / 2, H, T, HS)] = y2;
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

static int run_case(int HS, uint64_t seed) {
    constexpr int B = 2;
    constexpr int H = 3;
    constexpr int T = 128;
    size_t elems = (size_t)B * H * T * HS;
    size_t bytes = elems * sizeof(__nv_bfloat16);
    size_t cache_bytes = (size_t)T * (HS / 2) * sizeof(__nv_bfloat16);

    printf("\nShape: B=%d H=%d T=%d HS=%d\n", B, H, T, HS);

    std::vector<__nv_bfloat16> h_x(elems);
    std::vector<__nv_bfloat16> h_y(elems);
    std::vector<__nv_bfloat16> h_back(elems);
    std::vector<__nv_bfloat16> h_cos((size_t)T * (HS / 2));
    std::vector<__nv_bfloat16> h_sin((size_t)T * (HS / 2));
    fill_random_bf16(h_x, seed, -0.50f, 0.50f);
    fill_rope_cache(h_cos, h_sin, T, HS);

    std::vector<float> ref_y;
    cpu_rope(ref_y, h_x, h_cos, h_sin, B, H, T, HS, false);

    __nv_bfloat16* d_x = nullptr;
    __nv_bfloat16* d_y = nullptr;
    __nv_bfloat16* d_back = nullptr;
    __nv_bfloat16* d_cos = nullptr;
    __nv_bfloat16* d_sin = nullptr;
    cudaCheck(cudaMalloc(&d_x, bytes));
    cudaCheck(cudaMalloc(&d_y, bytes));
    cudaCheck(cudaMalloc(&d_back, bytes));
    cudaCheck(cudaMalloc(&d_cos, cache_bytes));
    cudaCheck(cudaMalloc(&d_sin, cache_bytes));

    cudaCheck(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_cos, h_cos.data(), cache_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_sin, h_sin.data(), cache_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_y, 0, bytes));
    cudaCheck(cudaMemset(d_back, 0, bytes));

    rope_forward(d_y, d_x, d_cos, d_sin, B, H, T, HS, 0);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

    std::vector<float> ref_back;
    cpu_rope(ref_back, h_y, h_cos, h_sin, B, H, T, HS, true);

    rope_backward(d_back, d_y, d_cos, d_sin, B, H, T, HS, 0);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(h_back.data(), d_back, bytes, cudaMemcpyDeviceToHost));

    double fwd_diff = max_abs_diff_bf16_float(h_y, ref_y);
    double bwd_diff = max_abs_diff_bf16_float(h_back, ref_back);
    double tolerance = 0.02;
    printf("forward max abs diff  = %.6f (tol %.3f) %s\n",
           fwd_diff, tolerance, fwd_diff <= tolerance ? "PASS" : "FAIL");
    printf("backward max abs diff = %.6f (tol %.3f) %s\n",
           bwd_diff, tolerance, bwd_diff <= tolerance ? "PASS" : "FAIL");

    cudaCheck(cudaFree(d_x));
    cudaCheck(cudaFree(d_y));
    cudaCheck(cudaFree(d_back));
    cudaCheck(cudaFree(d_cos));
    cudaCheck(cudaFree(d_sin));

    return (fwd_diff <= tolerance && bwd_diff <= tolerance) ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main() {
    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Device: %s (sm_%d%d)\n", deviceProp.name, deviceProp.major, deviceProp.minor);
    if (deviceProp.major != 9) {
        printf("warning: this smoke test targets H100 (sm_90a); continuing anyway\n");
    }

    int failures = 0;
    failures += run_case(64, 101);
    failures += run_case(128, 202);
    if (failures == 0) {
        printf("test_rope smoke OK\n");
    }
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
