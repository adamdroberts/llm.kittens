/*
SM120 cuBLASLt epilogue algorithm probe.

Build with:
  make bench_sm120_cublaslt_epilogue_algos DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1

This enumerates the cuBLASLt heuristic results for the GPT-2 124M MLP fused
epilogue shapes used by train_gpt2cu. It is intentionally separate from
bench_sm120_matmul so a per-algorithm winner can be measured before any trainer
route is changed.
*/
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#ifndef LLMK_SM120_USE_CUBLASLT_GEMM
#define LLMK_SM120_USE_CUBLASLT_GEMM
#define LLMK_BENCH_DEFINED_CUBLASLT
#endif

cudaDeviceProp deviceProp;

#include "llmc/matmul.cuh"

#ifdef LLMK_BENCH_DEFINED_CUBLASLT
#undef LLMK_SM120_USE_CUBLASLT_GEMM
#undef LLMK_BENCH_DEFINED_CUBLASLT
#endif

static void cublaslt_check(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[cuBLASLt ERROR]: %d %s:%d\n", (int)status, file, line);
        exit(EXIT_FAILURE);
    }
}

#define CUBLASLT_CHECK(status) cublaslt_check((status), __FILE__, __LINE__)

struct ProbeShape {
    const char* name;
    const char* logical;
    int m;
    int n;
    int k;
    bool transA;
    bool transB;
    bool has_bias;
    bool backward;
};

struct ProbePlan {
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t ALayout = nullptr;
    cublasLtMatrixLayout_t BLayout = nullptr;
    cublasLtMatrixLayout_t CLayout = nullptr;
    cublasLtMatrixLayout_t DLayout = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    std::vector<cublasLtMatmulHeuristicResult_t> heuristics;
};

static int bench_repeats() {
    static int repeats = [] {
        const char* env = std::getenv("LLMK_BENCH_REPEATS");
        if (env == nullptr || env[0] == '\0') return 5;
        const int parsed = std::atoi(env);
        return parsed > 0 ? parsed : 1;
    }();
    return repeats;
}

static int bench_iters() {
    static int iters = [] {
        const char* env = std::getenv("LLMK_BENCH_ITERS");
        if (env == nullptr || env[0] == '\0') return 6;
        const int parsed = std::atoi(env);
        return parsed > 0 ? parsed : 1;
    }();
    return iters;
}

static void fill_bytes(void* ptr, size_t bytes, int value) {
    cudaCheck(cudaMemset(ptr, value, bytes));
}

static void destroy_plan(ProbePlan& plan) {
    if (plan.preference != nullptr) {
        CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(plan.preference));
        plan.preference = nullptr;
    }
    if (plan.operationDesc != nullptr) {
        CUBLASLT_CHECK(cublasLtMatmulDescDestroy(plan.operationDesc));
        plan.operationDesc = nullptr;
    }
    if (plan.ALayout != nullptr) {
        CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(plan.ALayout));
        plan.ALayout = nullptr;
    }
    if (plan.BLayout != nullptr) {
        CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(plan.BLayout));
        plan.BLayout = nullptr;
    }
    if (plan.CLayout != nullptr) {
        CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(plan.CLayout));
        plan.CLayout = nullptr;
    }
    if (plan.DLayout != nullptr) {
        CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(plan.DLayout));
        plan.DLayout = nullptr;
    }
}

static ProbePlan create_plan(const ProbeShape& s, const floatX* bias, const floatX* pre_gelu) {
    ProbePlan plan;
    const cublasOperation_t opTranspose = CUBLAS_OP_T;
    const cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(
        &plan.operationDesc, llmk::cublaslt_sm120::compute_type, CUDA_R_32F));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan.operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
        s.transA ? &opTranspose : &opNoTranspose, sizeof(opTranspose)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan.operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
        s.transB ? &opTranspose : &opNoTranspose, sizeof(opNoTranspose)));

    if (s.transA) {
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan.ALayout, CUDA_R_16BF, s.k, s.m, s.k));
    } else {
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan.ALayout, CUDA_R_16BF, s.m, s.k, s.m));
    }
    if (s.transB) {
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan.BLayout, CUDA_R_16BF, s.n, s.k, s.n));
    } else {
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan.BLayout, CUDA_R_16BF, s.k, s.n, s.k));
    }
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan.CLayout, CUDA_R_16BF, s.m, s.n, s.m));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan.DLayout, CUDA_R_16BF, s.m, s.n, s.m));

    int64_t gelu_ld = s.m;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan.operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &gelu_ld, sizeof(gelu_ld)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan.operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu, sizeof(pre_gelu)));

    cublasLtEpilogue_t epilogue = s.backward
                                ? CUBLASLT_EPILOGUE_DGELU
                                : CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan.operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    if (s.has_bias) {
        cublasDataType_t bias_data_type = CUDA_R_16BF;
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            plan.operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
            &bias_data_type, sizeof(bias_data_type)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            plan.operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }
    cublasDataType_t scale_type = CUDA_R_32F;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan.operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&plan.preference));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        plan.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &llmk::cublaslt_sm120::workspace_size, sizeof(llmk::cublaslt_sm120::workspace_size)));

    plan.heuristics.resize(LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS);
    int returnedResults = 0;
    CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        llmk::cublaslt_sm120::handle, plan.operationDesc,
        plan.ALayout, plan.BLayout, plan.CLayout, plan.DLayout,
        plan.preference, LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS,
        plan.heuristics.data(), &returnedResults));
    plan.heuristics.resize(returnedResults);
    if (plan.heuristics.empty()) {
        fprintf(stderr, "No cuBLASLt algorithms for %s: m=%d n=%d k=%d\n", s.name, s.m, s.n, s.k);
        exit(EXIT_FAILURE);
    }
    return plan;
}

static bool run_algo_once(const ProbePlan& plan, const cublasLtMatmulAlgo_t& algo,
                          const floatX* a, const floatX* b, floatX* d,
                          cudaStream_t stream) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasLtMatmul(
        llmk::cublaslt_sm120::handle, plan.operationDesc, &alpha,
        a, plan.ALayout, b, plan.BLayout, &beta, d, plan.CLayout, d, plan.DLayout,
        &algo, llmk::cublaslt_sm120::workspace, llmk::cublaslt_sm120::workspace_size, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

static bool bench_algo_us(const ProbePlan& plan, int heuristic_index,
                          const floatX* a, const floatX* b, floatX* d,
                          cudaStream_t stream, float* out_us) {
    const cublasLtMatmulAlgo_t& algo = plan.heuristics[heuristic_index].algo;
    for (int i = 0; i < 2; ++i) {
        if (!run_algo_once(plan, algo, a, b, d, stream)) return false;
    }
    cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) return false;

    std::vector<float> samples;
    samples.reserve(bench_repeats());
    for (int repeat = 0; repeat < bench_repeats(); ++repeat) {
        cudaEvent_t start, stop;
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&stop));
        cudaCheck(cudaEventRecord(start, stream));
        for (int i = 0; i < bench_iters(); ++i) {
            if (!run_algo_once(plan, algo, a, b, d, stream)) return false;
        }
        cudaCheck(cudaEventRecord(stop, stream));
        cudaCheck(cudaEventSynchronize(stop));
        float ms = 0.0f;
        cudaCheck(cudaEventElapsedTime(&ms, start, stop));
        cudaCheck(cudaEventDestroy(start));
        cudaCheck(cudaEventDestroy(stop));
        samples.push_back(ms * 1000.0f / bench_iters());
    }
    std::sort(samples.begin(), samples.end());
    *out_us = samples[samples.size() / 2];
    return true;
}

static int default_heuristic_index(const ProbePlan& plan) {
    int best = 0;
    for (int i = 1; i < (int)plan.heuristics.size(); ++i) {
        if (plan.heuristics[i].wavesCount < plan.heuristics[best].wavesCount) {
            best = i;
        }
    }
    return best;
}

static void probe_shape(const ProbeShape& s, const floatX* a, const floatX* b,
                        const floatX* bias, floatX* d, floatX* pre_gelu,
                        cudaStream_t stream) {
    ProbePlan plan = create_plan(s, bias, pre_gelu);
    const int default_index = default_heuristic_index(plan);
    printf("\n%s %s | m=%d n=%d k=%d transA=%d transB=%d bias=%d backward=%d | returned=%zu default_lowest_waves=%d\n",
           s.name, s.logical, s.m, s.n, s.k, s.transA ? 1 : 0, s.transB ? 1 : 0,
           s.has_bias ? 1 : 0, s.backward ? 1 : 0, plan.heuristics.size(), default_index);

    float best_us = 0.0f;
    int best_index = -1;
    for (int i = 0; i < (int)plan.heuristics.size(); ++i) {
        float us = 0.0f;
        const bool ok = bench_algo_us(plan, i, a, b, d, stream, &us);
        if (!ok) {
            printf("  idx %2d waves=%7.2f time=failed\n", i, plan.heuristics[i].wavesCount);
            continue;
        }
        if (best_index < 0 || us < best_us) {
            best_index = i;
            best_us = us;
        }
        printf("  idx %2d waves=%7.2f time=%9.3f us%s\n",
               i, plan.heuristics[i].wavesCount, us, i == default_index ? " default" : "");
    }
    if (best_index >= 0) {
        const float default_us = best_index == default_index ? best_us : [&] {
            float us = 0.0f;
            if (!bench_algo_us(plan, default_index, a, b, d, stream, &us)) return -1.0f;
            return us;
        }();
        if (default_us > 0.0f) {
            printf("  best idx %d %.3f us | default idx %d %.3f us | best/default %.4fx\n",
                   best_index, best_us, default_index, default_us, best_us / default_us);
        } else {
            printf("  best idx %d %.3f us | default idx %d failed on retest\n",
                   best_index, best_us, default_index);
        }
    }
    destroy_plan(plan);
}

int main() {
    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Device: %s (sm_%d%d)\n", deviceProp.name, deviceProp.major, deviceProp.minor);
    printf("Timing: median of %d event samples, %d matmuls/sample\n", bench_repeats(), bench_iters());
    if (deviceProp.major != 12) {
        fprintf(stderr, "bench_sm120_cublaslt_epilogue_algos expects an SM120-class GPU\n");
        return 1;
    }

    llmk::cublaslt_sm120::init();

    const int M = 64 * 1024;
    const int C = 768;
    const int F = 4 * C;
    const size_t hidden_bytes = (size_t)M * C * sizeof(floatX);
    const size_t wide_bytes = (size_t)M * F * sizeof(floatX);
    const size_t weight_bytes = (size_t)F * C * sizeof(floatX);
    const size_t bias_bytes = (size_t)F * sizeof(floatX);

    floatX* hidden = nullptr;
    floatX* wide = nullptr;
    floatX* weight = nullptr;
    floatX* bias = nullptr;
    floatX* out = nullptr;
    floatX* pre_gelu = nullptr;
    cudaCheck(cudaMalloc(&hidden, hidden_bytes));
    cudaCheck(cudaMalloc(&wide, wide_bytes));
    cudaCheck(cudaMalloc(&weight, weight_bytes));
    cudaCheck(cudaMalloc(&bias, bias_bytes));
    cudaCheck(cudaMalloc(&out, wide_bytes));
    cudaCheck(cudaMalloc(&pre_gelu, wide_bytes));
    fill_bytes(hidden, hidden_bytes, 1);
    fill_bytes(wide, wide_bytes, 2);
    fill_bytes(weight, weight_bytes, 3);
    fill_bytes(bias, bias_bytes, 4);
    fill_bytes(out, wide_bytes, 0);
    fill_bytes(pre_gelu, wide_bytes, 5);

    const ProbeShape fwd_gelu{
        "fc fwd+GeLU", "row-major out(M,4C)=inp(M,C)*weight(4C,C)^T + bias",
        F, M, C, true, false, true, false,
    };
    const ProbeShape dinp_dgelu{
        "fcproj dInp+dGeLU", "row-major dinp(M,4C)=dout(M,C)*weight(C,4C) with dGeLU",
        F, M, C, false, false, false, true,
    };

    probe_shape(fwd_gelu, weight, hidden, bias, out, pre_gelu, 0);
    probe_shape(dinp_dgelu, weight, hidden, nullptr, out, pre_gelu, 0);

    cudaCheck(cudaFree(hidden));
    cudaCheck(cudaFree(wide));
    cudaCheck(cudaFree(weight));
    cudaCheck(cudaFree(bias));
    cudaCheck(cudaFree(out));
    cudaCheck(cudaFree(pre_gelu));
    llmk::cublaslt_sm120::destroy();
    return 0;
}

#undef CUBLASLT_CHECK
