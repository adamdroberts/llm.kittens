/*
test_fused_classifier.cu — smoke test for llmc/fused_classifier.cuh.

Builds a small bf16 logits tensor with shape (B*T, P) where V<=P (padded
vocab), random targets, and calls the fused kernel with WriteDLogits=true.
Compares the resulting per-row loss (float) and the in-place dlogits (bf16)
against an independent CPU reference computed with stable log-softmax in float.

Build via the Makefile target:

    make test_fused_classifier
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

#include "llmc/fused_classifier.cuh"

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

    constexpr int B = 2, T = 8;
    constexpr int V = 1003;     // unaligned vocab tail
    constexpr int P = 1024;     // padded multiple of x128::size=8
    constexpr float dloss = 1.0f / (float)(B * T);
    const int rows = B * T;
    printf("Shape: B=%d T=%d V=%d P=%d dloss=%.6f\n", B, T, V, P, dloss);

    // Random logits in (P) layout — only the first V columns are valid; the rest
    // are padding and the kernel ignores them when computing softmax/loss.
    std::vector<__nv_bfloat16> h_logits((size_t)rows * P);
    fill_random_bf16(h_logits, 555, -2.0f, 2.0f);

    std::vector<int> h_targets(rows);
    {
        std::mt19937_64 rng(666);
        std::uniform_int_distribution<int> dist(0, V - 1);
        for (auto& v : h_targets) v = dist(rng);
    }

    // CPU reference: stable log-softmax over the first V columns of each row.
    std::vector<float> ref_loss(rows, 0.0f);
    std::vector<float> ref_dlogits((size_t)rows * P, 0.0f);
    for (int r = 0; r < rows; ++r) {
        float maxv = -INFINITY;
        for (int j = 0; j < V; ++j) {
            float v = bf16_to_float(h_logits[(size_t)r * P + j]);
            if (v > maxv) maxv = v;
        }
        double sumexp = 0.0;
        for (int j = 0; j < V; ++j) {
            sumexp += std::exp((double)bf16_to_float(h_logits[(size_t)r * P + j]) - (double)maxv);
        }
        float log_sumexp = (float)std::log(sumexp) + maxv;
        int tgt = h_targets[r];
        float logit_tgt = bf16_to_float(h_logits[(size_t)r * P + tgt]);
        ref_loss[r] = -(logit_tgt - log_sumexp);
        for (int j = 0; j < V; ++j) {
            float v = bf16_to_float(h_logits[(size_t)r * P + j]);
            float prob = (float)std::exp((double)v - (double)log_sumexp);
            float indicator = (j == tgt) ? 1.0f : 0.0f;
            ref_dlogits[(size_t)r * P + j] = (prob - indicator) * dloss;
        }
        for (int j = V; j < P; ++j) {
            // The kernel leaves padding columns untouched. The trainer relies
            // on zero initialized padded embedding rows, so ignore this tail in
            // the classifier-local dlogits comparison.
            ref_dlogits[(size_t)r * P + j] = INFINITY;
        }
    }

    __nv_bfloat16* d_logits = nullptr;
    __nv_bfloat16* d_logits_loss_only = nullptr;
    int* d_targets = nullptr;
    float* d_losses = nullptr;
    float* d_losses_loss_only = nullptr;
    cudaCheck(cudaMalloc(&d_logits, (size_t)rows * P * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_logits_loss_only, (size_t)rows * P * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_targets, rows * sizeof(int)));
    cudaCheck(cudaMalloc(&d_losses, rows * sizeof(float)));
    cudaCheck(cudaMalloc(&d_losses_loss_only, rows * sizeof(float)));

    cudaCheck(cudaMemcpy(d_logits, h_logits.data(), (size_t)rows * P * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_logits_loss_only, h_logits.data(), (size_t)rows * P * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_targets, h_targets.data(), rows * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_losses, 0, rows * sizeof(float)));
    cudaCheck(cudaMemset(d_losses_loss_only, 0, rows * sizeof(float)));

    fused_classifier<__nv_bfloat16, false>(d_logits_loss_only, d_losses_loss_only, dloss, d_targets,
                                           B, T, V, P, False, 0);
    cudaCheck(cudaDeviceSynchronize());

    fused_classifier<__nv_bfloat16, true>(d_logits, d_losses, dloss, d_targets,
                                          B, T, V, P, True, 0);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<float> g_loss(rows);
    std::vector<float> g_loss_only(rows);
    std::vector<__nv_bfloat16> g_dlogits((size_t)rows * P);
    std::vector<__nv_bfloat16> g_logits_loss_only((size_t)rows * P);
    cudaCheck(cudaMemcpy(g_loss.data(), d_losses, rows * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(g_loss_only.data(), d_losses_loss_only, rows * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(g_dlogits.data(), d_logits, (size_t)rows * P * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(g_logits_loss_only.data(), d_logits_loss_only, (size_t)rows * P * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    double loss_diff = 0.0;
    double loss_only_diff = 0.0;
    for (int r = 0; r < rows; ++r) {
        double d = std::abs((double)g_loss[r] - (double)ref_loss[r]);
        if (d > loss_diff) loss_diff = d;
        double d_loss_only = std::abs((double)g_loss_only[r] - (double)ref_loss[r]);
        if (d_loss_only > loss_only_diff) loss_only_diff = d_loss_only;
    }
    double dlogits_diff = 0.0;
    double loss_only_logits_diff = 0.0;
    for (int r = 0; r < rows; ++r) {
        for (int j = 0; j < P; ++j) {
            float r_v = ref_dlogits[(size_t)r * P + j];
            if (std::isinf(r_v)) continue;
            double d = std::abs((double)bf16_to_float(g_dlogits[(size_t)r * P + j]) - (double)r_v);
            if (d > dlogits_diff) dlogits_diff = d;
        }
        for (int j = 0; j < P; ++j) {
            double d = std::abs((double)bf16_to_float(g_logits_loss_only[(size_t)r * P + j])
                              - (double)bf16_to_float(h_logits[(size_t)r * P + j]));
            if (d > loss_only_logits_diff) loss_only_logits_diff = d;
        }
    }

    double loss_tol = 5e-3;     // bf16 logits → fp32 loss; small V*max_log_softmax error
    double dlogits_tol = 1e-3;  // dlogits ≈ prob*dloss, small magnitudes
    double unchanged_tol = 0.0;
    printf("loss-only loss max abs diff = %.6f (tol %.4f) %s\n", loss_only_diff, loss_tol,
           loss_only_diff <= loss_tol ? "PASS" : "FAIL");
    printf("loss-only logits max abs diff = %.6f (tol %.4f) %s\n",
           loss_only_logits_diff, unchanged_tol,
           loss_only_logits_diff <= unchanged_tol ? "PASS" : "FAIL");
    printf("loss    max abs diff = %.6f (tol %.4f) %s\n", loss_diff, loss_tol,
           loss_diff <= loss_tol ? "PASS" : "FAIL");
    printf("dlogits max abs diff = %.6f (tol %.4f) %s\n", dlogits_diff, dlogits_tol,
           dlogits_diff <= dlogits_tol ? "PASS" : "FAIL");

    cudaCheck(cudaFree(d_logits));
    cudaCheck(cudaFree(d_logits_loss_only));
    cudaCheck(cudaFree(d_targets));
    cudaCheck(cudaFree(d_losses));
    cudaCheck(cudaFree(d_losses_loss_only));

    bool ok = loss_only_diff <= loss_tol
           && loss_only_logits_diff <= unchanged_tol
           && loss_diff <= loss_tol
           && dlogits_diff <= dlogits_tol;
    if (ok) printf("test_fused_classifier smoke OK\n");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
