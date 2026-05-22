#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "llmc/attention.cuh"

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

cudaDeviceProp deviceProp;

static int bench_repeats() {
    const char* env = getenv("LLMK_BENCH_REPEATS");
    if (env == nullptr) {
        return 3;
    }
    int repeats = atoi(env);
    return repeats > 0 ? repeats : 3;
}

template <typename Fn>
static float median_event_us(cudaEvent_t start, cudaEvent_t stop, Fn&& fn, int iters) {
    std::vector<float> samples;
    int repeats = bench_repeats();
    samples.reserve(repeats);
    for (int repeat = 0; repeat < repeats; repeat++) {
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            fn();
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(milliseconds * 1000.0f / iters);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void bench_attention(int B, int T, int C, int NH) {
    int HS = C / NH;
    const size_t out_elems = (size_t)B * T * C;
    const size_t packed_elems = 3 * out_elems;
    const size_t out_bytes = out_elems * sizeof(__nv_bfloat16);
    const size_t packed_bytes = packed_elems * sizeof(__nv_bfloat16);

    size_t att_elems = (size_t)B * NH * T;
    size_t att_bytes = att_elems * sizeof(float);

    __nv_bfloat16 *d_qkv, *d_out;
    float *d_lse;
    CHECK_CUDA(cudaMalloc(&d_qkv, packed_bytes));
    CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
    CHECK_CUDA(cudaMalloc(&d_lse, att_bytes));

#if defined(LLMK_SM120_ATOMIC_DQ)
    __nv_bfloat16 *d_qkvr;
    CHECK_CUDA(cudaMalloc(&d_qkvr, packed_bytes));
#endif

    // Forward
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    for(int i = 0; i < 10; i++) {
#if defined(LLMK_SM120_ATOMIC_DQ)
        attention_forward(d_out, d_qkvr, (__nv_bfloat16*)d_lse, d_qkv, B, T, C, NH, 0);
#else
        attention_forward_packed_qkv(d_out, (__nv_bfloat16*)d_lse, d_qkv, B, T, C, NH, 0);
#endif
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    float microseconds = median_event_us(start, stop, [&]() {
#if defined(LLMK_SM120_ATOMIC_DQ)
        attention_forward(d_out, d_qkvr, (__nv_bfloat16*)d_lse, d_qkv, B, T, C, NH, 0);
#else
        attention_forward_packed_qkv(d_out, (__nv_bfloat16*)d_lse, d_qkv, B, T, C, NH, 0);
#endif
    }, 100);
    printf("Attention Forward (B=%d, T=%d, C=%d, NH=%d, HS=%d): %.3f us\n",
           B, T, C, NH, HS, microseconds);

    // Backward
    __nv_bfloat16 *d_dinp, *d_dout;

    CHECK_CUDA(cudaMalloc(&d_dinp, packed_bytes));
    CHECK_CUDA(cudaMalloc(&d_dout, out_bytes));

    void* d_datt_combined;
#if defined(LLMK_SM120_ATOMIC_DQ)
    __nv_bfloat16* d_dqkvr;
    CHECK_CUDA(cudaMalloc(&d_dqkvr, packed_bytes));
    size_t datt_combined_bytes = 2 * out_elems * sizeof(__nv_bfloat16)
                               + (B * NH * T + out_elems) * sizeof(float);
#else
    size_t datt_combined_bytes = 2 * out_elems * sizeof(__nv_bfloat16) + B * NH * T * sizeof(float);
#endif
    CHECK_CUDA(cudaMalloc(&d_datt_combined, datt_combined_bytes));

    // Warmup
    for(int i = 0; i < 10; i++) {
#if defined(LLMK_SM120_ATOMIC_DQ)
        attention_backward(d_dinp, d_dqkvr, (__nv_bfloat16*)d_datt_combined, d_out,
                           d_dout, d_qkvr, (__nv_bfloat16*)d_lse, B, T, C, NH, 0);
#else
        attention_backward_packed_qkv(d_dinp, (__nv_bfloat16*)d_datt_combined, d_out, d_dout, d_qkv, (__nv_bfloat16*)d_lse, B, T, C, NH, 0);
#endif
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    microseconds = median_event_us(start, stop, [&]() {
#if defined(LLMK_SM120_ATOMIC_DQ)
        attention_backward(d_dinp, d_dqkvr, (__nv_bfloat16*)d_datt_combined, d_out,
                           d_dout, d_qkvr, (__nv_bfloat16*)d_lse, B, T, C, NH, 0);
#else
        attention_backward_packed_qkv(d_dinp, (__nv_bfloat16*)d_datt_combined, d_out, d_dout, d_qkv, (__nv_bfloat16*)d_lse, B, T, C, NH, 0);
#endif
    }, 100);
    printf("Attention Backward (B=%d, T=%d, C=%d, NH=%d, HS=%d): %.3f us\n",
           B, T, C, NH, HS, microseconds);

    CHECK_CUDA(cudaFree(d_qkv));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_lse));
#if defined(LLMK_SM120_ATOMIC_DQ)
    CHECK_CUDA(cudaFree(d_qkvr));
    CHECK_CUDA(cudaFree(d_dqkvr));
#endif
    CHECK_CUDA(cudaFree(d_dinp));
    CHECK_CUDA(cudaFree(d_dout));
    CHECK_CUDA(cudaFree(d_datt_combined));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Timing: median of %d event samples per row\n", bench_repeats());
    // GPT-2 124M
    bench_attention(64, 1024, 768, 12);
    return 0;
}
