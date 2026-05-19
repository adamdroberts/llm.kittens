/*
SM120 GPT-2 matmul microbenchmark.

Build with:
  make bench_sm120_matmul DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1

This compares the current pure ThunderKittens SM120 GEMM launch path against the
SM120 cuBLASLt fallback for the GPT-2 124M shapes used by train_gpt2cu.
*/
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifndef LLMK_SM120_USE_CUBLASLT_GEMM
#define LLMK_SM120_USE_CUBLASLT_GEMM
#define LLMK_BENCH_DEFINED_CUBLASLT
#endif

cudaDeviceProp deviceProp;

#include "llmc/matmul.cuh"
#include "llmc/gelu.cuh"

#ifdef LLMK_BENCH_DEFINED_CUBLASLT
#undef LLMK_SM120_USE_CUBLASLT_GEMM
#undef LLMK_BENCH_DEFINED_CUBLASLT
#endif

#ifndef LLMK_SM120_DWEIGHT_SPLIT_K
#define LLMK_SM120_DWEIGHT_SPLIT_K 8
#endif

#ifndef LLMK_SM120_FUSE_BIAS
#define LLMK_SM120_FUSE_BIAS 1
#endif

struct Shape {
    const char* name;
    int M;
    int N;
    int K;
    bool bias;
    bool gelu;
};

static void fill_bytes(void* ptr, size_t bytes, int value) {
    cudaCheck(cudaMemset(ptr, value, bytes));
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
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    float ms = 0.0f;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    return ms * 1000.0f / iters;
}

static void tk_forward(floatX* out, const floatX* inp, const floatX* weight,
                       const floatX* bias, const Shape& s, cudaStream_t stream) {
    auto* A = llmk::to_bf16(const_cast<floatX*>(inp));
    auto* B = llmk::to_bf16(const_cast<floatX*>(weight));
    auto* C = llmk::to_bf16(out);
    const bool huge_n = matmul_sm120_use_huge_n_tile(s.N);
    const bool huge_n_wide = huge_n && (LLMK_SM120_HUGE_N_FORWARD_WIDE != 0) &&
                             (s.M % 256 == 0) && (s.N % 64 == 0);
    const bool wide = !huge_n && (s.M % 256 == 0) && (s.N % 64 == 0);
#if LLMK_SM120_FORWARD_N96
    const bool n96 = !huge_n && (s.N % 96 == 0);
#else
    const bool n96 = false;
#endif
#if LLMK_SM120_FUSE_BIAS
    auto* bias_bf = llmk::to_bf16(const_cast<floatX*>(bias));
    if (bias != nullptr && huge_n_wide) {
        llmk::gemm::launch<llmk::gemm::matmul_wide_nt_bias>(A, B, C, s.M, s.N, s.K, stream, nullptr, bias_bf);
    } else if (bias != nullptr && huge_n) {
        llmk::gemm::launch<llmk::gemm::matmul_huge_n_nt_bias>(A, B, C, s.M, s.N, s.K, stream, nullptr, bias_bf);
    } else if (bias != nullptr && n96) {
        llmk::gemm::launch<llmk::gemm::matmul_n96_nt_bias>(A, B, C, s.M, s.N, s.K, stream, nullptr, bias_bf);
    } else if (bias != nullptr && wide) {
        llmk::gemm::launch<llmk::gemm::matmul_wide_nt_bias>(A, B, C, s.M, s.N, s.K, stream, nullptr, bias_bf);
    } else if (bias != nullptr && s.N % 256 == 0) {
        llmk::gemm::launch<llmk::gemm::matmul_default_nt_bias>(A, B, C, s.M, s.N, s.K, stream, nullptr, bias_bf);
    } else if (bias != nullptr) {
        llmk::gemm::launch<llmk::gemm::matmul_small_n_nt_bias>(A, B, C, s.M, s.N, s.K, stream, nullptr, bias_bf);
    } else
#endif
    if (huge_n_wide) {
        llmk::gemm::launch<llmk::gemm::matmul_wide_nt>(A, B, C, s.M, s.N, s.K, stream);
    } else if (huge_n) {
        llmk::gemm::launch<llmk::gemm::matmul_huge_n_nt>(A, B, C, s.M, s.N, s.K, stream);
    } else if (n96) {
        llmk::gemm::launch<llmk::gemm::matmul_n96_nt>(A, B, C, s.M, s.N, s.K, stream);
    } else if (wide) {
        llmk::gemm::launch<llmk::gemm::matmul_wide_nt>(A, B, C, s.M, s.N, s.K, stream);
    } else if (s.N % 256 == 0) {
        llmk::gemm::launch<llmk::gemm::matmul_default_nt>(A, B, C, s.M, s.N, s.K, stream);
    } else {
        llmk::gemm::launch<llmk::gemm::matmul_small_n_nt>(A, B, C, s.M, s.N, s.K, stream);
    }
    cudaCheck(cudaGetLastError());
#if LLMK_SM120_FUSE_BIAS
    if (bias != nullptr) return;
#endif
    add_bias(out, bias, s.M, s.N, stream);
}

static void tk_forward_gelu(floatX* out, floatX* pre_gelu, const floatX* inp,
                            const floatX* weight, const floatX* bias,
                            const Shape& s, cudaStream_t stream) {
    auto* A = llmk::to_bf16(const_cast<floatX*>(inp));
    auto* B = llmk::to_bf16(const_cast<floatX*>(weight));
    auto* C = llmk::to_bf16(out);
    auto* P = llmk::to_bf16(pre_gelu);
    auto* bias_bf = llmk::to_bf16(const_cast<floatX*>(bias));
    const bool wide = (s.M % 256 == 0) && (s.N % 64 == 0);
#if LLMK_SM120_FORWARD_N96
    const bool n96 = s.N % 96 == 0;
#else
    const bool n96 = false;
#endif
    if (n96) {
        llmk::gemm::launch<llmk::gemm::matmul_n96_nt_bias_gelu>(A, B, C, s.M, s.N, s.K, stream, P, bias_bf);
    } else if (wide) {
        llmk::gemm::launch<llmk::gemm::matmul_wide_nt_bias_gelu>(A, B, C, s.M, s.N, s.K, stream, P, bias_bf);
    } else if (s.N % 256 == 0) {
        llmk::gemm::launch<llmk::gemm::matmul_default_nt_bias_gelu>(A, B, C, s.M, s.N, s.K, stream, P, bias_bf);
    } else {
        llmk::gemm::launch<llmk::gemm::matmul_small_n_nt_bias_gelu>(A, B, C, s.M, s.N, s.K, stream, P, bias_bf);
    }
    cudaCheck(cudaGetLastError());
}

static void tk_forward_explicit_gelu(floatX* out, floatX* pre_gelu, const floatX* inp,
                                     const floatX* weight, const floatX* bias,
                                     const Shape& s, cudaStream_t stream) {
    tk_forward(pre_gelu, inp, weight, bias, s, stream);
    gelu_forward(out, pre_gelu, s.M * s.N, stream);
}

static void cublaslt_forward(floatX* out, const floatX* inp, const floatX* weight,
                             const floatX* bias, const Shape& s, cudaStream_t stream) {
    llmk::cublaslt_sm120::matmul(
        out, weight, inp, bias, s.N, s.M, s.K, stream,
        /*transA=*/true, /*transB=*/false,
        /*batch_count=*/0, /*strideA=*/0, /*strideB=*/0, /*strideOut=*/0,
        /*accumulate=*/false, /*pre_gelu=*/nullptr, /*backward=*/false);
}

static void cublaslt_forward_gelu(floatX* out, floatX* pre_gelu, const floatX* inp,
                                  const floatX* weight, const floatX* bias,
                                  const Shape& s, cudaStream_t stream) {
    llmk::cublaslt_sm120::matmul(
        out, weight, inp, bias, s.N, s.M, s.K, stream,
        /*transA=*/true, /*transB=*/false,
        /*batch_count=*/0, /*strideA=*/0, /*strideB=*/0, /*strideOut=*/0,
        /*accumulate=*/false, /*pre_gelu=*/pre_gelu, /*backward=*/false);
}

static void tk_dinp(floatX* out, const floatX* dout, const floatX* weight,
                    const Shape& s, cudaStream_t stream) {
    matmul_dispatch_tk_ab(out, dout, weight, s.M, s.K, s.N, stream);
}

static void cublaslt_dinp(floatX* out, const floatX* dout, const floatX* weight,
                          const Shape& s, cudaStream_t stream) {
    llmk::cublaslt_sm120::matmul(
        out, weight, dout, nullptr, s.K, s.M, s.N, stream,
        /*transA=*/false, /*transB=*/false,
        /*batch_count=*/0, /*strideA=*/0, /*strideB=*/0, /*strideOut=*/0,
        /*accumulate=*/false, /*pre_gelu=*/nullptr, /*backward=*/true);
}

static void tk_dweight(floatX* out, const floatX* dout, const floatX* inp,
                       floatX* scratch, size_t scratch_elements,
                       const Shape& s, cudaStream_t stream) {
    if (!matmul_dispatch_tk_atb_splitk(
            out, dout, inp, s.N, s.K, s.M, stream,
            /*accumulate=*/false, scratch, scratch_elements)) {
        matmul_dispatch_tk_atb(out, dout, inp, s.N, s.K, s.M, stream);
    }
}

static void cublaslt_dweight(floatX* out, const floatX* dout, const floatX* inp,
                             const Shape& s, cudaStream_t stream) {
    llmk::cublaslt_sm120::matmul(
        out, inp, dout, nullptr, s.K, s.N, s.M, stream,
        /*transA=*/false, /*transB=*/true,
        /*batch_count=*/0, /*strideA=*/0, /*strideB=*/0, /*strideOut=*/0,
        /*accumulate=*/false, /*pre_gelu=*/nullptr, /*backward=*/true);
}

static void bench_shape(const Shape& s) {
    const size_t a_bytes = (size_t)s.M * s.K * sizeof(floatX);
    const size_t w_bytes = (size_t)s.N * s.K * sizeof(floatX);
    const size_t out_bytes = (size_t)s.M * s.N * sizeof(floatX);
    const size_t bias_bytes = (size_t)s.N * sizeof(floatX);
    const int warmup = s.N >= 8192 ? 1 : 3;
    const int iters = s.N >= 8192 ? 3 : 10;

    floatX* A = nullptr;
    floatX* W = nullptr;
    floatX* O = nullptr;
    floatX* Bias = nullptr;
    cudaCheck(cudaMalloc(&A, a_bytes));
    cudaCheck(cudaMalloc(&W, w_bytes));
    cudaCheck(cudaMalloc(&O, out_bytes));
    if (s.bias) cudaCheck(cudaMalloc(&Bias, bias_bytes));
    fill_bytes(A, a_bytes, 1);
    fill_bytes(W, w_bytes, 2);
    fill_bytes(O, out_bytes, 0);
    if (Bias != nullptr) fill_bytes(Bias, bias_bytes, 3);

    floatX* PreGelu = nullptr;
    if (s.gelu) {
        cudaCheck(cudaMalloc(&PreGelu, out_bytes));
        fill_bytes(PreGelu, out_bytes, 0);
    }

    printf("\n%-12s M=%d N=%d K=%d bias=%d gelu=%d\n", s.name, s.M, s.N, s.K,
           s.bias ? 1 : 0, s.gelu ? 1 : 0);
    if (s.gelu) {
        float tk_fused = bench_us([&] { tk_forward_gelu(O, PreGelu, A, W, Bias, s, 0); }, warmup, iters);
        float tk_explicit = bench_us([&] { tk_forward_explicit_gelu(O, PreGelu, A, W, Bias, s, 0); }, warmup, iters);
        float cb_fwd = bench_us([&] { cublaslt_forward_gelu(O, PreGelu, A, W, Bias, s, 0); }, warmup, iters);
        printf("  fwd+GeLU TK fused %9.2f us | TK explicit %9.2f us | cuBLASLt %9.2f us | explicit/cuBLASLt %.2fx\n",
               tk_fused, tk_explicit, cb_fwd, tk_explicit / cb_fwd);
    } else {
        float tk_fwd = bench_us([&] { tk_forward(O, A, W, Bias, s, 0); }, warmup, iters);
        float cb_fwd = bench_us([&] { cublaslt_forward(O, A, W, Bias, s, 0); }, warmup, iters);
        printf("  fwd      TK %9.2f us | cuBLASLt %9.2f us | TK/cuBLASLt %.2fx\n",
               tk_fwd, cb_fwd, tk_fwd / cb_fwd);
    }

    floatX* DInp = nullptr;
    cudaCheck(cudaMalloc(&DInp, a_bytes));
    fill_bytes(DInp, a_bytes, 0);
    float tk_di = bench_us([&] { tk_dinp(DInp, O, W, s, 0); }, warmup, iters);
    float cb_di = bench_us([&] { cublaslt_dinp(DInp, O, W, s, 0); }, warmup, iters);
    printf("  dInp   TK %9.2f us | cuBLASLt %9.2f us | TK/cuBLASLt %.2fx\n",
           tk_di, cb_di, tk_di / cb_di);
    cudaCheck(cudaFree(DInp));

    floatX* DW = nullptr;
    floatX* DWScratch = nullptr;
    cudaCheck(cudaMalloc(&DW, w_bytes));
    cudaCheck(cudaMalloc(&DWScratch, (size_t)LLMK_SM120_DWEIGHT_SPLIT_K * w_bytes));
    fill_bytes(DW, w_bytes, 0);
    fill_bytes(DWScratch, (size_t)LLMK_SM120_DWEIGHT_SPLIT_K * w_bytes, 0);
    float tk_dw = bench_us([&] {
        tk_dweight(DW, O, A, DWScratch,
                   (size_t)LLMK_SM120_DWEIGHT_SPLIT_K * s.N * s.K, s, 0);
    }, warmup, iters);
    float cb_dw = bench_us([&] { cublaslt_dweight(DW, O, A, s, 0); }, warmup, iters);
    printf("  dW     TK %9.2f us | cuBLASLt %9.2f us | TK/cuBLASLt %.2fx\n",
           tk_dw, cb_dw, tk_dw / cb_dw);
    cudaCheck(cudaFree(DW));
    cudaCheck(cudaFree(DWScratch));

    cudaCheck(cudaFree(A));
    cudaCheck(cudaFree(W));
    cudaCheck(cudaFree(O));
    if (PreGelu != nullptr) cudaCheck(cudaFree(PreGelu));
    if (Bias != nullptr) cudaCheck(cudaFree(Bias));
}

int main() {
    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Device: %s (sm_%d%d)\n", deviceProp.name, deviceProp.major, deviceProp.minor);
    if (deviceProp.major != 12) {
        fprintf(stderr, "bench_sm120_matmul expects an SM120-class GPU\n");
        return 1;
    }
    llmk::cublaslt_sm120::init();

    const int M = 64 * 1024;
    Shape shapes[] = {
        {"qkv", M, 3 * 768, 768, true, false},
        {"attproj", M, 768, 768, true, false},
        {"fc", M, 4 * 768, 768, true, true},
        {"fcproj", M, 768, 4 * 768, true, false},
        {"lmhead", M, 50304, 768, false, false},
    };
    for (const Shape& s : shapes) {
        bench_shape(s);
    }

    llmk::cublaslt_sm120::destroy();
    return 0;
}
