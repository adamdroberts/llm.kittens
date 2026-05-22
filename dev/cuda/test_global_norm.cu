/*
test_global_norm.cu — smoke test for llmc/global_norm.cuh.

Builds a couple of bf16 parameter "shards" of different lengths, calls
global_norm_squared() on each (accumulating into the same output buffer), then
launches the aggregate kernel to collapse the partial block sums into a single
sum-of-squares. Compares sqrt(out[0]) against a CPU reference.

Build via the Makefile target:

    make test_global_norm
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

#include "llmc/global_norm.cuh"

static float bf16_to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
static __nv_bfloat16 float_to_bf16(float x) { return __float2bfloat16(x); }

static void fill_random_bf16(std::vector<__nv_bfloat16>& h, uint64_t seed,
                             float lo, float hi) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& v : h) v = float_to_bf16(dist(rng));
}

int main() {
    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Device: %s (sm_%d%d)\n", deviceProp.name, deviceProp.major, deviceProp.minor);

    // Two "parameter shards" — different sizes, both contiguous (single slice each).
    constexpr int N1 = 4096 * 32;  // 131072 elements
    constexpr int N2 = 768 * 64;   // 49152 elements
    printf("Shape: shard1=%d shard2=%d (single-slice each)\n", N1, N2);

    std::vector<__nv_bfloat16> h_p1(N1), h_p2(N2);
    fill_random_bf16(h_p1, 31, -0.5f, 0.5f);
    fill_random_bf16(h_p2, 32, -0.25f, 0.25f);

    double cpu_sumsq = 0.0;
    for (auto& v : h_p1) { double f = bf16_to_float(v); cpu_sumsq += f * f; }
    for (auto& v : h_p2) { double f = bf16_to_float(v); cpu_sumsq += f * f; }
    double cpu_norm = std::sqrt(cpu_sumsq);

    __nv_bfloat16 *d_p1 = nullptr, *d_p2 = nullptr;
    cudaCheck(cudaMalloc(&d_p1, (size_t)N1 * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_p2, (size_t)N2 * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMemcpy(d_p1, h_p1.data(), (size_t)N1 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_p2, h_p2.data(), (size_t)N2 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    int num_slices_all[2] = {1, 1};
    int max_block_sums = get_max_num_block_sums(num_slices_all, 2);
    float* d_out = nullptr;
    cudaCheck(cudaMalloc(&d_out, max_block_sums * sizeof(float)));
    cudaCheck(cudaMemset(d_out, 0, max_block_sums * sizeof(float)));

    // First shard: reset=true clears the partial-sum buffer.
    global_norm_squared<__nv_bfloat16>(d_out, d_p1, N1, /*stride=*/0, /*num_slices=*/1,
                                       max_block_sums, /*reset=*/true, 0);
    // Second shard: accumulate without resetting.
    global_norm_squared<__nv_bfloat16>(d_out, d_p2, N2, /*stride=*/0, /*num_slices=*/1,
                                       max_block_sums, /*reset=*/false, 0);
    // Collapse all per-block partial sums into out[0].
    global_norm_aggregate_kernel<<<1, 1024, 0, 0>>>(d_out, max_block_sums);
    cudaCheck(cudaDeviceSynchronize());

    float gpu_sumsq = 0.0f;
    cudaCheck(cudaMemcpy(&gpu_sumsq, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    double gpu_norm = std::sqrt((double)gpu_sumsq);

    // Tolerance: bf16 inputs squared then summed in fp32. Relative error should be ~1%.
    double rel = std::abs(gpu_norm - cpu_norm) / std::max(cpu_norm, 1e-12);
    double tol = 0.01;
    printf("cpu norm = %.6f  gpu norm = %.6f  relative diff = %.6f (tol %.3f) %s\n",
           cpu_norm, gpu_norm, rel, tol, rel <= tol ? "PASS" : "FAIL");

    bool ok = rel <= tol;

    // Exercise the reset=true fallback where max_block_sums is larger than the
    // rows written by this call. The untouched tail must be cleared before the
    // deterministic aggregate reads the whole buffer.
    int mixed_slices[2] = {1, 4};
    int mixed_max_block_sums = get_max_num_block_sums(mixed_slices, 2);
    float* d_tail_out = nullptr;
    cudaCheck(cudaMalloc(&d_tail_out, (size_t)mixed_max_block_sums * sizeof(float)));
    std::vector<float> stale_tail((size_t)mixed_max_block_sums, 7.0f);
    cudaCheck(cudaMemcpy(d_tail_out, stale_tail.data(), (size_t)mixed_max_block_sums * sizeof(float), cudaMemcpyHostToDevice));
    global_norm_squared<__nv_bfloat16>(d_tail_out, d_p1, N1, /*stride=*/0, /*num_slices=*/1,
                                       mixed_max_block_sums, /*reset=*/true, 0);
    global_norm_aggregate_kernel<<<1, 1024, 0, 0>>>(d_tail_out, mixed_max_block_sums);
    cudaCheck(cudaDeviceSynchronize());
    float tail_gpu_sumsq = 0.0f;
    cudaCheck(cudaMemcpy(&tail_gpu_sumsq, d_tail_out, sizeof(float), cudaMemcpyDeviceToHost));
    double tail_gpu_norm = std::sqrt((double)tail_gpu_sumsq);
    double tail_cpu_sumsq = 0.0;
    for (auto& v : h_p1) { double f = bf16_to_float(v); tail_cpu_sumsq += f * f; }
    double tail_cpu_norm = std::sqrt(tail_cpu_sumsq);
    double tail_rel = std::abs(tail_gpu_norm - tail_cpu_norm) / std::max(tail_cpu_norm, 1e-12);
    printf("reset-tail cpu norm = %.6f  gpu norm = %.6f  relative diff = %.6f (tol %.3f) %s\n",
           tail_cpu_norm, tail_gpu_norm, tail_rel, tol, tail_rel <= tol ? "PASS" : "FAIL");
    cudaCheck(cudaFree(d_tail_out));
    ok = (tail_rel <= tol) && ok;

    cudaCheck(cudaFree(d_p1));
    cudaCheck(cudaFree(d_p2));
    cudaCheck(cudaFree(d_out));

    if (ok) printf("test_global_norm smoke OK\n");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
