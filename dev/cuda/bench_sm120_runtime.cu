/*
SM120 GPT-2 runtime kernel microbenchmark.

Build with:
  make bench_sm120_runtime DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1

This covers the non-GEMM GPT-2 kernel families called out by optimise-goal.md:
bias add/reduction, GELU, classifier, AdamW, global norm, encoder, memsets, and
copies. These are plain CUDA baselines unless a future stack-specific candidate
is added.
*/
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <type_traits>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

cudaDeviceProp deviceProp;

#include "llmc/matmul.cuh"
#include "llmc/gelu.cuh"
#include "llmc/fused_classifier.cuh"
#include "llmc/adamw.cuh"
#include "llmc/global_norm.cuh"
#include "llmc/encoder.cuh"
#include "llmc/memory.cuh"

static int bench_repeats() {
    const char* env = getenv("LLMK_BENCH_REPEATS");
    if (env == nullptr) {
        return 3;
    }
    int repeats = atoi(env);
    return repeats > 0 ? repeats : 3;
}

template <typename Fn>
static float bench_us(Fn&& fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    cudaCheck(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    std::vector<float> samples;
    const int repeats = bench_repeats();
    samples.reserve(repeats);
    for (int repeat = 0; repeat < repeats; ++repeat) {
        cudaCheck(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i) {
            fn();
        }
        cudaCheck(cudaEventRecord(stop));
        cudaCheck(cudaEventSynchronize(stop));
        float ms = 0.0f;
        cudaCheck(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms * 1000.0f / iters);
    }

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));

    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

static void print_result(const char* name, const char* shape, const char* stack, float us) {
    printf("%-30s | %-28s | %-12s | %9.3f us\n", name, shape, stack, us);
}

static void bench_bias_and_gelu(cudaStream_t stream) {
    constexpr int B = 64;
    constexpr int T = 1024;
    constexpr int BT = B * T;
    constexpr int C = 768;
    constexpr int QKV = 3 * C;
    constexpr int FC = 3072;
    const size_t hidden_elems = (size_t)BT * C;
    const size_t qkv_elems = (size_t)BT * QKV;
    const size_t fc_elems = (size_t)BT * FC;

    floatX *hidden = nullptr, *hidden_aux = nullptr, *qkv = nullptr, *fc = nullptr, *fc_aux = nullptr;
    floatX *bias_c = nullptr, *bias_fc = nullptr, *dbias = nullptr;
    float* dbias_buffer = nullptr;
    cudaCheck(cudaMalloc(&hidden, hidden_elems * sizeof(floatX)));
    cudaCheck(cudaMalloc(&hidden_aux, hidden_elems * sizeof(floatX)));
    cudaCheck(cudaMalloc(&qkv, qkv_elems * sizeof(floatX)));
    cudaCheck(cudaMalloc(&fc, fc_elems * sizeof(floatX)));
    cudaCheck(cudaMalloc(&fc_aux, fc_elems * sizeof(floatX)));
    cudaCheck(cudaMalloc(&bias_c, C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&bias_fc, FC * sizeof(floatX)));
    cudaCheck(cudaMalloc(&dbias, FC * sizeof(floatX)));
    cudaCheck(cudaMalloc(&dbias_buffer, (size_t)FC * 1024 * sizeof(float)));

    cudaCheck(cudaMemset(hidden, 1, hidden_elems * sizeof(floatX)));
    cudaCheck(cudaMemset(hidden_aux, 2, hidden_elems * sizeof(floatX)));
    cudaCheck(cudaMemset(qkv, 1, qkv_elems * sizeof(floatX)));
    cudaCheck(cudaMemset(fc, 1, fc_elems * sizeof(floatX)));
    cudaCheck(cudaMemset(fc_aux, 2, fc_elems * sizeof(floatX)));
    cudaCheck(cudaMemset(bias_c, 3, C * sizeof(floatX)));
    cudaCheck(cudaMemset(bias_fc, 3, FC * sizeof(floatX)));
    cudaCheck(cudaMemset(dbias, 0, FC * sizeof(floatX)));
    cudaCheck(cudaMemset(dbias_buffer, 0, (size_t)FC * 1024 * sizeof(float)));

    print_result("bias_add", "BT=65536 OC=768", "CUDA", bench_us([&] {
        add_bias(hidden, bias_c, BT, C, stream);
    }, 5, 50));
    print_result("bias_add", "BT=65536 OC=3072", "CUDA", bench_us([&] {
        add_bias(fc, bias_fc, BT, FC, stream);
    }, 5, 25));
    print_result("gelu_forward", "BT=65536 C=3072", "CUDA", bench_us([&] {
        gelu_forward(fc, fc_aux, (int)fc_elems, stream);
    }, 5, 25));
    print_result("gelu_backward_inplace", "BT=65536 C=3072", "CUDA", bench_us([&] {
        gelu_backward_inplace(fc, fc_aux, (int)fc_elems, stream);
    }, 5, 25));
    print_result("bias_grad_reduce", "BT=65536 OC=768", "CUDA", bench_us([&] {
        matmul_backward_bias(dbias, hidden, dbias_buffer, B, T, C, stream);
    }, 3, 20));
    print_result("bias_grad_reduce", "BT=65536 OC=2304", "CUDA", bench_us([&] {
        matmul_backward_bias(dbias, qkv, dbias_buffer, B, T, QKV, stream);
    }, 3, 20));
    print_result("bias_grad_reduce", "BT=65536 OC=3072", "CUDA", bench_us([&] {
        matmul_backward_bias(dbias, fc, dbias_buffer, B, T, FC, stream);
    }, 3, 20));

    cudaCheck(cudaFree(hidden));
    cudaCheck(cudaFree(hidden_aux));
    cudaCheck(cudaFree(qkv));
    cudaCheck(cudaFree(fc));
    cudaCheck(cudaFree(fc_aux));
    cudaCheck(cudaFree(bias_c));
    cudaCheck(cudaFree(bias_fc));
    cudaCheck(cudaFree(dbias));
    cudaCheck(cudaFree(dbias_buffer));
}

static void bench_classifier(cudaStream_t stream) {
    constexpr int B = 64;
    constexpr int T = 1024;
    constexpr int V = 50257;
    constexpr int P = 50304;
    const size_t logits_elems = (size_t)B * T * P;

    floatX* logits = nullptr;
    floatX* logits_loss_only = nullptr;
    float* losses = nullptr;
    float* losses_loss_only = nullptr;
    int* targets = nullptr;
    cudaCheck(cudaMalloc(&logits, logits_elems * sizeof(floatX)));
    cudaCheck(cudaMalloc(&logits_loss_only, logits_elems * sizeof(floatX)));
    cudaCheck(cudaMalloc(&losses, (size_t)B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&losses_loss_only, (size_t)B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&targets, (size_t)B * T * sizeof(int)));
    cudaCheck(cudaMemset(logits, 0, logits_elems * sizeof(floatX)));
    cudaCheck(cudaMemset(logits_loss_only, 0, logits_elems * sizeof(floatX)));
    cudaCheck(cudaMemset(losses, 0, (size_t)B * T * sizeof(float)));
    cudaCheck(cudaMemset(losses_loss_only, 0, (size_t)B * T * sizeof(float)));
    cudaCheck(cudaMemset(targets, 0, (size_t)B * T * sizeof(int)));

    print_result("fused_classifier_loss", "B=64 T=1024 V=50257 P=50304", "CUDA", bench_us([&] {
        fused_classifier(logits_loss_only, losses_loss_only, 1.0f, targets, B, T, V, P,
                         std::bool_constant<false>{}, stream);
    }, 1, 5));
    print_result("fused_classifier", "B=64 T=1024 V=50257 P=50304", "CUDA", bench_us([&] {
        fused_classifier(logits, losses, 1.0f, targets, B, T, V, P,
                         std::bool_constant<true>{}, stream);
    }, 1, 5));
    print_result("cuda_memset", "logits_elems=3296722944", "CUDA runtime", bench_us([&] {
        cudaCheck(cudaMemsetAsync(logits_loss_only, 0, logits_elems * sizeof(floatX), stream));
    }, 1, 5));
    print_result("cuda_memset", "logits_elems=3296722944", "CUDA kernel", bench_us([&] {
        memory_zero_floatx(logits_loss_only, logits_elems, stream);
    }, 1, 5));
    print_result("cuda_copy_d2d", "logits_elems=3296722944", "CUDA runtime", bench_us([&] {
        cudaCheck(cudaMemcpyAsync(logits, logits_loss_only, logits_elems * sizeof(floatX),
                                  cudaMemcpyDeviceToDevice, stream));
    }, 1, 5));
    print_result("cuda_copy_d2d", "logits_elems=3296722944", "CUDA kernel", bench_us([&] {
        memory_copy_floatx(logits, logits_loss_only, logits_elems, stream);
    }, 1, 5));

    cudaCheck(cudaFree(logits));
    cudaCheck(cudaFree(logits_loss_only));
    cudaCheck(cudaFree(losses));
    cudaCheck(cudaFree(losses_loss_only));
    cudaCheck(cudaFree(targets));
}

static void bench_optimizer_and_norm(cudaStream_t stream) {
    constexpr size_t params_count = 124475904;

    floatX *params = nullptr, *grads = nullptr;
    float *m = nullptr, *v = nullptr, *norm = nullptr;
    cudaCheck(cudaMalloc(&params, params_count * sizeof(floatX)));
    cudaCheck(cudaMalloc(&grads, params_count * sizeof(floatX)));
    cudaCheck(cudaMalloc(&m, params_count * sizeof(float)));
    cudaCheck(cudaMalloc(&v, params_count * sizeof(float)));

    int num_slices = 1;
    int max_num_block_sums = get_max_num_block_sums(&num_slices, 1);
    cudaCheck(cudaMalloc(&norm, (size_t)max_num_block_sums * sizeof(float)));

    cudaCheck(cudaMemset(params, 1, params_count * sizeof(floatX)));
    cudaCheck(cudaMemset(grads, 1, params_count * sizeof(floatX)));
    cudaCheck(cudaMemset(m, 0, params_count * sizeof(float)));
    cudaCheck(cudaMemset(v, 0, params_count * sizeof(float)));
    cudaCheck(cudaMemset(norm, 0, (size_t)max_num_block_sums * sizeof(float)));

    print_result("cuda_memset", "grad_elems=124475904", "CUDA runtime", bench_us([&] {
        cudaCheck(cudaMemsetAsync(grads, 0, params_count * sizeof(floatX), stream));
    }, 3, 20));
    cudaCheck(cudaMemset(grads, 1, params_count * sizeof(floatX)));
    print_result("cuda_memset", "grad_elems=124475904", "CUDA kernel", bench_us([&] {
        memory_zero_floatx(grads, params_count, stream);
    }, 3, 20));
    cudaCheck(cudaMemset(grads, 1, params_count * sizeof(floatX)));

    print_result("global_norm_squared", "params=124475904", "CUDA", bench_us([&] {
        global_norm_squared(norm, grads, params_count, 0, 1, max_num_block_sums, true, stream);
        global_sum_deterministic(norm, norm, max_num_block_sums, stream);
    }, 3, 20));
    print_result("adamw_update", "params=124475904 no-master", "CUDA", bench_us([&] {
        adamw_update(params, (float*)nullptr, grads, m, v, params_count,
                     0, 0, 0, 1,
                     0.0006f, 0.9f, 0.95f, 1, 1.0e-8f, 0.1f,
                     1.0f, 1234u, stream);
    }, 2, 10));

    cudaCheck(cudaFree(params));
    cudaCheck(cudaFree(grads));
    cudaCheck(cudaFree(m));
    cudaCheck(cudaFree(v));
    cudaCheck(cudaFree(norm));
}

static void bench_encoder_and_memory(cudaStream_t stream) {
    constexpr int B = 64;
    constexpr int T = 1024;
    constexpr int C = 768;
    constexpr int P = 50304;
    const size_t hidden_elems = (size_t)B * T * C;

    floatX *out = nullptr, *copy_dst = nullptr, *wte = nullptr, *wpe = nullptr;
    int* tokens = nullptr;
    cudaCheck(cudaMalloc(&out, hidden_elems * sizeof(floatX)));
    cudaCheck(cudaMalloc(&copy_dst, hidden_elems * sizeof(floatX)));
    cudaCheck(cudaMalloc(&wte, (size_t)P * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&wpe, (size_t)T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&tokens, (size_t)B * T * sizeof(int)));
    cudaCheck(cudaMemset(out, 0, hidden_elems * sizeof(floatX)));
    cudaCheck(cudaMemset(copy_dst, 0, hidden_elems * sizeof(floatX)));
    cudaCheck(cudaMemset(wte, 1, (size_t)P * C * sizeof(floatX)));
    cudaCheck(cudaMemset(wpe, 1, (size_t)T * C * sizeof(floatX)));
    cudaCheck(cudaMemset(tokens, 0, (size_t)B * T * sizeof(int)));

    print_result("encoder_forward", "B=64 T=1024 C=768", "CUDA", bench_us([&] {
        encoder_forward(out, tokens, wte, wpe, B, T, C, stream);
    }, 5, 50));
    print_result("cuda_memset", "hidden_elems=50331648", "CUDA runtime", bench_us([&] {
        cudaCheck(cudaMemsetAsync(out, 0, hidden_elems * sizeof(floatX), stream));
    }, 5, 50));
    print_result("cuda_memset", "hidden_elems=50331648", "CUDA kernel", bench_us([&] {
        memory_zero_floatx(out, hidden_elems, stream);
    }, 5, 50));
    print_result("cuda_copy_d2d", "hidden_elems=50331648", "CUDA runtime", bench_us([&] {
        cudaCheck(cudaMemcpyAsync(copy_dst, out, hidden_elems * sizeof(floatX),
                                  cudaMemcpyDeviceToDevice, stream));
    }, 5, 50));
    print_result("cuda_copy_d2d", "hidden_elems=50331648", "CUDA kernel", bench_us([&] {
        memory_copy_floatx(copy_dst, out, hidden_elems, stream);
    }, 5, 50));

    cudaCheck(cudaFree(out));
    cudaCheck(cudaFree(copy_dst));
    cudaCheck(cudaFree(wte));
    cudaCheck(cudaFree(wpe));
    cudaCheck(cudaFree(tokens));
}

int main() {
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
    cudaStream_t stream = nullptr;
    printf("SM120 GPT-2 runtime kernel benchmark on %s\n", deviceProp.name);
    printf("Timing: median of %d event samples per row\n", bench_repeats());
    printf("%-30s | %-28s | %-12s | %12s\n", "Kernel", "Shape", "Stack", "Time");
    printf("%-30s-+-%-28s-+-%-12s-+-%12s\n",
           "------------------------------", "----------------------------",
           "------------", "------------");
    bench_bias_and_gelu(stream);
    bench_classifier(stream);
    bench_optimizer_and_norm(stream);
    bench_encoder_and_memory(stream);
    return 0;
}
