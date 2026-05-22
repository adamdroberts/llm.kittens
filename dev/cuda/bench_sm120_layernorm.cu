#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

cudaDeviceProp deviceProp;

#include "llmc/layernorm.cuh"

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

static int bench_repeats() {
    const char* env = getenv("LLMK_BENCH_REPEATS");
    if (env == nullptr) {
        return 3;
    }
    int repeats = atoi(env);
    return repeats > 0 ? repeats : 3;
}

template <typename Fn>
static float median_event_us(cudaEvent_t start, cudaEvent_t stop, Fn&& fn) {
    std::vector<float> samples;
    int repeats = bench_repeats();
    samples.reserve(repeats);
    for (int repeat = 0; repeat < repeats; repeat++) {
        CHECK_CUDA(cudaEventRecord(start));
        fn();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(milliseconds * 10.0f);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void bench_layernorm(int B, int T, int C) {
    int N = B * T;
    size_t elems = (size_t)N * C;
    size_t bytes = elems * sizeof(__nv_bfloat16);
    size_t row_bytes = (size_t)N * sizeof(float);
    size_t weight_bytes = (size_t)C * sizeof(__nv_bfloat16);

    __nv_bfloat16 *d_inp, *d_skip, *d_out, *d_residual, *d_fused_out, *d_weight, *d_bias;
    float *d_mean, *d_rstd, *d_fused_mean, *d_fused_rstd;
    CHECK_CUDA(cudaMalloc(&d_inp, bytes));
    CHECK_CUDA(cudaMalloc(&d_skip, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMalloc(&d_residual, bytes));
    CHECK_CUDA(cudaMalloc(&d_fused_out, bytes));
    CHECK_CUDA(cudaMalloc(&d_weight, weight_bytes));
    CHECK_CUDA(cudaMalloc(&d_bias, weight_bytes));
    CHECK_CUDA(cudaMalloc(&d_mean, row_bytes));
    CHECK_CUDA(cudaMalloc(&d_rstd, row_bytes));
    CHECK_CUDA(cudaMalloc(&d_fused_mean, row_bytes));
    CHECK_CUDA(cudaMalloc(&d_fused_rstd, row_bytes));
    CHECK_CUDA(cudaMemset(d_inp, 1, bytes));
    CHECK_CUDA(cudaMemset(d_skip, 2, bytes));
    CHECK_CUDA(cudaMemset(d_out, 0, bytes));
    CHECK_CUDA(cudaMemset(d_residual, 0, bytes));
    CHECK_CUDA(cudaMemset(d_fused_out, 0, bytes));
    CHECK_CUDA(cudaMemset(d_weight, 3, weight_bytes));
    CHECK_CUDA(cudaMemset(d_bias, 4, weight_bytes));
    CHECK_CUDA(cudaMemset(d_mean, 0, row_bytes));
    CHECK_CUDA(cudaMemset(d_rstd, 0, row_bytes));
    CHECK_CUDA(cudaMemset(d_fused_mean, 0, row_bytes));
    CHECK_CUDA(cudaMemset(d_fused_rstd, 0, row_bytes));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Forward
    for(int i = 0; i < 10; i++) {
        layernorm_forward(d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, 0);
    }
    float microseconds = median_event_us(start, stop, [&]() {
        for(int i = 0; i < 100; i++) {
            layernorm_forward(d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, 0);
        }
    });
    printf("LayerNorm Forward (N=%d, C=%d): %.3f us\n", N, C, microseconds);

    // Fused residual add + LayerNorm forward.
    for(int i = 0; i < 10; i++) {
        fused_residual_forward5(d_residual, d_fused_out, d_fused_mean, d_fused_rstd,
                                d_inp, d_skip, d_weight, d_bias, N, C, 0);
    }
    microseconds = median_event_us(start, stop, [&]() {
        for(int i = 0; i < 100; i++) {
            fused_residual_forward5(d_residual, d_fused_out, d_fused_mean, d_fused_rstd,
                                    d_inp, d_skip, d_weight, d_bias, N, C, 0);
        }
    });
    printf("LayerNorm FusedResidualForward (N=%d, C=%d): %.3f us\n", N, C, microseconds);

    // Backward
    __nv_bfloat16 *d_dinp, *d_dweight, *d_dbias, *d_dout;
    float *d_scratch;
    CHECK_CUDA(cudaMalloc(&d_dinp, bytes));
    CHECK_CUDA(cudaMalloc(&d_dweight, weight_bytes));
    CHECK_CUDA(cudaMalloc(&d_dbias, weight_bytes));
    CHECK_CUDA(cudaMalloc(&d_dout, bytes));
    
    size_t scratch_floats = 32 + 2 * (size_t)C * (2 * (size_t)deviceProp.multiProcessorCount);
    CHECK_CUDA(cudaMalloc(&d_scratch, scratch_floats * sizeof(float)));

    for(int i = 0; i < 10; i++) {
        layernorm_backward(d_dinp, d_dweight, d_dbias, d_scratch, d_dout, d_inp, d_weight, d_mean, d_rstd, B, T, C, 0);
    }
    microseconds = median_event_us(start, stop, [&]() {
        for(int i = 0; i < 100; i++) {
            layernorm_backward(d_dinp, d_dweight, d_dbias, d_scratch, d_dout, d_inp, d_weight, d_mean, d_rstd, B, T, C, 0);
        }
    });
    printf("LayerNorm Backward (N=%d, C=%d): %.3f us\n", N, C, microseconds);

    CHECK_CUDA(cudaFree(d_inp));
    CHECK_CUDA(cudaFree(d_skip));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_residual));
    CHECK_CUDA(cudaFree(d_fused_out));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_mean));
    CHECK_CUDA(cudaFree(d_rstd));
    CHECK_CUDA(cudaFree(d_fused_mean));
    CHECK_CUDA(cudaFree(d_fused_rstd));
    CHECK_CUDA(cudaFree(d_dinp));
    CHECK_CUDA(cudaFree(d_dweight));
    CHECK_CUDA(cudaFree(d_dbias));
    CHECK_CUDA(cudaFree(d_dout));
    CHECK_CUDA(cudaFree(d_scratch));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Timing: median of %d event samples per row\n", bench_repeats());
    // GPT-2 124M hidden width and MLP width.
    bench_layernorm(64, 1024, 768);
    bench_layernorm(64, 1024, 3072);
    return 0;
}
