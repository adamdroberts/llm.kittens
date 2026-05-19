/*
matmul.cuh — C-style wrapper around ThunderKittens H100 bf16 GEMM.

Replaces llm.c's cuBLASLt-backed matmul. The forward signature is
intentionally close to llm.c's `matmul_forward_cublaslt`, so train_gpt2.cu
can be ported with minimal changes. v1 differences:

  * The default forward path keeps bias as a separate pointwise pass. The GPT-2
    MLP up-projection can opt into a TK finish-path bias+GELU epilogue via
    `matmul_forward_gelu`, storing pre-GELU for backward while writing GELU
    output.
  * Forward uses the TK A*B^T variant because llm.c parameter files store
    dense weights as (OC, C), not as the GEMM-friendly (C, OC).
  * Backward dInp uses the existing TK A*B path. Backward dWeight uses TK A^T*B;
    accumulated micro-steps compute into caller scratch, then add into the
    gradient tensor. The bias-grad reduction kernels are ported verbatim from
    llm.c — TK gives no benefit on a column reduction.

Shape constraints (must hold; we do NOT pad on the fly):
  * M (= B*T) divisible by 128
  * N (= OC) divisible by 256 for the default kernel; 128 for the small-N
    fallback used by the GPT-2 LM-head projection
  * K (= C) divisible by  64

For GPT-2 124M with B=4 T=1024 C=768 these all hold; for the LM-head projection
M=4096 N=V_padded=50304 K=768 the small-N fallback is selected automatically.
*/
#pragma once

#include <assert.h>
#include <type_traits>
#include "cuda_common.h"
#include "cuda_utils.cuh"
#if defined(KITTENS_SM120) && (defined(LLMK_SM120_USE_CUBLASLT_GEMM) || \
    LLMK_SM120_CUBLASLT_FORWARD_FALLBACK || LLMK_SM120_CUBLASLT_DINP_FALLBACK || \
    LLMK_SM120_CUBLASLT_DWEIGHT_FALLBACK)
#define LLMK_SM120_HAS_CUBLASLT_GEMM 1
#else
#define LLMK_SM120_HAS_CUBLASLT_GEMM 0
#endif
#if LLMK_SM120_HAS_CUBLASLT_GEMM
#include <cublasLt.h>
#include <cublas_v2.h>
#include <vector>
#endif
#if !defined(LLMK_DISABLE_TK_GEMM) && defined(KITTENS_SM90)
#define LLMK_USE_TK_GEMM 1
#include "tk/gemm_h100.cuh"
#elif !defined(LLMK_DISABLE_TK_GEMM) && defined(KITTENS_SM120)
#define LLMK_USE_TK_GEMM 1
#include "tk/gemm_sm120.cuh"
#else
#define LLMK_USE_TK_GEMM 0
#endif

#if LLMK_SM120_HAS_CUBLASLT_GEMM
namespace llmk::cublaslt_sm120 {

#ifndef LLMK_SM120_CUBLASLT_WORKSPACE_MB
#define LLMK_SM120_CUBLASLT_WORKSPACE_MB 128
#endif
static constexpr size_t workspace_size = (size_t)LLMK_SM120_CUBLASLT_WORKSPACE_MB * 1024 * 1024;
static void* workspace = nullptr;
static cublasLtHandle_t handle = nullptr;
static cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;

#ifndef LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS
#define LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS 1
#endif
static_assert(LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS >= 1 &&
              LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS <= 64,
              "LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS must be in [1, 64]");

struct PlanKey {
    int m;
    int n;
    int k;
    bool transA;
    bool transB;
    int batch_count;
    size_t strideA;
    size_t strideB;
    size_t strideOut;
    bool has_bias;
    bool has_gelu;
    bool backward;
};

struct MatmulPlan {
    PlanKey key;
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t ALayout = nullptr;
    cublasLtMatrixLayout_t BLayout = nullptr;
    cublasLtMatrixLayout_t CLayout = nullptr;
    cublasLtMatrixLayout_t DLayout = nullptr;
    cublasLtMatmulAlgo_t algo;
};

static std::vector<MatmulPlan> plans;

inline void check(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[cuBLASLt ERROR]: %d %s:%d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}

#define LLMK_CUBLASLT_CHECK(status) ::llmk::cublaslt_sm120::check((status), __FILE__, __LINE__)

inline void init() {
    if (handle != nullptr) return;
    LLMK_CUBLASLT_CHECK(cublasLtCreate(&handle));
    cudaCheck(cudaMalloc(&workspace, workspace_size));
}

inline void destroy() {
    for (MatmulPlan& plan : plans) {
        if (plan.operationDesc != nullptr) LLMK_CUBLASLT_CHECK(cublasLtMatmulDescDestroy(plan.operationDesc));
        if (plan.ALayout != nullptr) LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(plan.ALayout));
        if (plan.BLayout != nullptr) LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(plan.BLayout));
        if (plan.CLayout != nullptr) LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(plan.CLayout));
        if (plan.DLayout != nullptr) LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(plan.DLayout));
    }
    plans.clear();
    if (workspace != nullptr) {
        cudaCheck(cudaFree(workspace));
        workspace = nullptr;
    }
    if (handle != nullptr) {
        LLMK_CUBLASLT_CHECK(cublasLtDestroy(handle));
        handle = nullptr;
    }
}

inline bool same_key(const PlanKey& a, const PlanKey& b) {
    return a.m == b.m && a.n == b.n && a.k == b.k &&
           a.transA == b.transA && a.transB == b.transB &&
           a.batch_count == b.batch_count &&
           a.strideA == b.strideA && a.strideB == b.strideB && a.strideOut == b.strideOut &&
           a.has_bias == b.has_bias && a.has_gelu == b.has_gelu && a.backward == b.backward;
}

inline int select_heuristic(const cublasLtMatmulHeuristicResult_t* heuristics, int returnedResults) {
    int best = 0;
#if defined(LLMK_SM120_CUBLASLT_HEURISTIC_INDEX)
    (void)heuristics;
    best = LLMK_SM120_CUBLASLT_HEURISTIC_INDEX;
    if (best < 0) best = 0;
    if (best >= returnedResults) best = returnedResults - 1;
#elif defined(LLMK_SM120_CUBLASLT_SELECT_MIN_WAVES)
    for (int i = 1; i < returnedResults; ++i) {
        if (heuristics[i].wavesCount < heuristics[best].wavesCount) {
            best = i;
        }
    }
#elif defined(LLMK_SM120_CUBLASLT_SELECT_MAX_WAVES)
    for (int i = 1; i < returnedResults; ++i) {
        if (heuristics[i].wavesCount > heuristics[best].wavesCount) {
            best = i;
        }
    }
#else
    (void)heuristics;
    (void)returnedResults;
#endif
    return best;
}

inline MatmulPlan& get_plan(const PlanKey& key) {
    for (MatmulPlan& plan : plans) {
        if (same_key(plan.key, key)) return plan;
    }

    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue;
    if (key.has_gelu) {
        epilogue = key.backward
                 ? CUBLASLT_EPILOGUE_DGELU
                 : (key.has_bias ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS : CUBLASLT_EPILOGUE_GELU_AUX);
    } else if (key.has_bias) {
        epilogue = key.backward ? CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BIAS;
    } else {
        epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    }

    MatmulPlan plan;
    plan.key = key;
    LLMK_CUBLASLT_CHECK(cublasLtMatmulDescCreate(&plan.operationDesc, compute_type, CUDA_R_32F));
    LLMK_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan.operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
        key.transA ? &opTranspose : &opNoTranspose, sizeof(opTranspose)));
    LLMK_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan.operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
        key.transB ? &opTranspose : &opNoTranspose, sizeof(opNoTranspose)));

    if (key.transA) {
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan.ALayout, CUDA_R_16BF, key.k, key.m, key.k));
    } else {
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan.ALayout, CUDA_R_16BF, key.m, key.k, key.m));
    }
    if (key.transB) {
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan.BLayout, CUDA_R_16BF, key.n, key.k, key.n));
    } else {
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan.BLayout, CUDA_R_16BF, key.k, key.n, key.k));
    }
    LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan.CLayout, CUDA_R_16BF, key.m, key.n, key.m));
    LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan.DLayout, CUDA_R_16BF, key.m, key.n, key.m));

    if (key.batch_count) {
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            plan.ALayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &key.batch_count, sizeof(key.batch_count)));
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            plan.BLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &key.batch_count, sizeof(key.batch_count)));
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            plan.CLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &key.batch_count, sizeof(key.batch_count)));
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            plan.DLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &key.batch_count, sizeof(key.batch_count)));
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            plan.ALayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &key.strideA, sizeof(key.strideA)));
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            plan.BLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &key.strideB, sizeof(key.strideB)));
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            plan.CLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &key.strideOut, sizeof(key.strideOut)));
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            plan.DLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &key.strideOut, sizeof(key.strideOut)));
    }

    if (key.has_gelu) {
        int64_t gelu_ld = key.m;
        LLMK_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            plan.operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &gelu_ld, sizeof(gelu_ld)));
    }
    LLMK_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan.operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    if (key.has_bias) {
        cublasDataType_t bias_data_type = CUDA_R_16BF;
        LLMK_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            plan.operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
    }
    cublasDataType_t scale_type = CUDA_R_32F;
    LLMK_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan.operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

    cublasLtMatmulPreference_t preference;
    LLMK_CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    LLMK_CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size, sizeof(workspace_size)));

    cublasLtMatmulHeuristicResult_t heuristics[LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS];
    int returnedResults = 0;
    LLMK_CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        handle, plan.operationDesc, plan.ALayout, plan.BLayout, plan.CLayout, plan.DLayout,
        preference, LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS, heuristics, &returnedResults));
    if (returnedResults == 0) {
        fprintf(stderr, "No cuBLASLt algorithm: m=%d n=%d k=%d bias=%d\n",
                key.m, key.n, key.k, key.has_bias);
        exit(EXIT_FAILURE);
    }
    plan.algo = heuristics[select_heuristic(heuristics, returnedResults)].algo;
    LLMK_CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(preference));

    plans.push_back(plan);
    return plans.back();
}

inline void matmul(floatX* d, const floatX* a, const floatX* b, const floatX* bias,
                   int m, int n, int k, cudaStream_t stream,
                   bool transA, bool transB,
                   int batch_count = 0, size_t strideA = 0, size_t strideB = 0, size_t strideOut = 0,
                   bool accumulate = false, floatX* pre_gelu = nullptr, bool backward = false) {
    assert(handle != nullptr && "llmk::cublaslt_sm120::init() must run before matmul");

#if defined(LLMK_SM120_CACHE_CUBLASLT_PLANS)
    {
        const bool has_bias = bias != nullptr;
        const bool has_gelu = pre_gelu != nullptr;
        if (backward && has_bias && has_gelu) {
            fprintf(stderr, "cuBLASLt SM120 GEMM does not support simultaneous BGRADB and DGELU\n");
            exit(EXIT_FAILURE);
        }
        PlanKey key{m, n, k, transA, transB, batch_count, strideA, strideB, strideOut,
                    has_bias, has_gelu, backward};
        MatmulPlan& plan = get_plan(key);
        if (has_gelu) {
            LLMK_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
                plan.operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                &pre_gelu, sizeof(pre_gelu)));
        }
        if (has_bias) {
            LLMK_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
                plan.operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &bias, sizeof(bias)));
        }

        const float alpha = 1.0f;
        const float beta = accumulate ? 1.0f : 0.0f;
        LLMK_CUBLASLT_CHECK(cublasLtMatmul(
            handle, plan.operationDesc, &alpha,
            a, plan.ALayout, b, plan.BLayout, &beta, d, plan.CLayout, d, plan.DLayout,
            &plan.algo, workspace, workspace_size, stream));
        cudaCheck(cudaGetLastError());
        return;
    }
#endif

    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasLtMatmulDesc_t operationDesc;
    LLMK_CUBLASLT_CHECK(cublasLtMatmulDescCreate(&operationDesc, compute_type, CUDA_R_32F));
    LLMK_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
        transA ? &opTranspose : &opNoTranspose, sizeof(opTranspose)));
    LLMK_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
        transB ? &opTranspose : &opNoTranspose, sizeof(opNoTranspose)));

    cublasLtMatrixLayout_t ALayout;
    cublasLtMatrixLayout_t BLayout;
    cublasLtMatrixLayout_t CLayout;
    cublasLtMatrixLayout_t DLayout;
    if (transA) {
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&ALayout, CUDA_R_16BF, k, m, k));
    } else {
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&ALayout, CUDA_R_16BF, m, k, m));
    }
    if (transB) {
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&BLayout, CUDA_R_16BF, n, k, n));
    } else {
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&BLayout, CUDA_R_16BF, k, n, k));
    }
    LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&CLayout, CUDA_R_16BF, m, n, m));
    LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&DLayout, CUDA_R_16BF, m, n, m));

    if (batch_count) {
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            ALayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            BLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            CLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            DLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            ALayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            BLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            CLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut, sizeof(strideOut)));
        LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            DLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut, sizeof(strideOut)));
    }

    cublasLtMatmulPreference_t preference;
    LLMK_CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    LLMK_CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size, sizeof(workspace_size)));

    const bool has_bias = bias != nullptr;
    cublasLtEpilogue_t epilogue;
    if (pre_gelu != nullptr) {
        int64_t gelu_ld = m;
        LLMK_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &gelu_ld, sizeof(gelu_ld)));
        LLMK_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu, sizeof(pre_gelu)));
        if (backward) {
            assert(!has_bias);
            epilogue = CUBLASLT_EPILOGUE_DGELU;
        } else {
            epilogue = has_bias ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS : CUBLASLT_EPILOGUE_GELU_AUX;
        }
    } else if (has_bias) {
        epilogue = backward ? CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BIAS;
    } else {
        epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    }
    LLMK_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    if (has_bias) {
        cublasDataType_t bias_data_type = CUDA_R_16BF;
        LLMK_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        LLMK_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }
    cublasDataType_t scale_type = CUDA_R_32F;
    LLMK_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

    cublasLtMatmulHeuristicResult_t heuristics[LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS];
    int returnedResults = 0;
    LLMK_CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        handle, operationDesc, ALayout, BLayout, CLayout, DLayout,
        preference, LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS, heuristics, &returnedResults));
    if (returnedResults == 0) {
        fprintf(stderr, "No cuBLASLt algorithm: m=%d n=%d k=%d bias=%d\n", m, n, k, has_bias);
        exit(EXIT_FAILURE);
    }

    const float alpha = 1.0f;
    const float beta = accumulate ? 1.0f : 0.0f;
    LLMK_CUBLASLT_CHECK(cublasLtMatmul(
        handle, operationDesc, &alpha,
        a, ALayout, b, BLayout, &beta, d, CLayout, d, DLayout,
        &heuristics[select_heuristic(heuristics, returnedResults)].algo, workspace, workspace_size, stream));

    LLMK_CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    LLMK_CUBLASLT_CHECK(cublasLtMatmulDescDestroy(operationDesc));
    LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(ALayout));
    LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(BLayout));
    LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(CLayout));
    LLMK_CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(DLayout));
    cudaCheck(cudaGetLastError());
}

#undef LLMK_CUBLASLT_CHECK

} // namespace llmk::cublaslt_sm120
#endif

#if defined(KITTENS_SM120)
#ifndef LLMK_SM120_HUGE_N_THRESHOLD
#define LLMK_SM120_HUGE_N_THRESHOLD 8192
#endif
#ifndef LLMK_SM120_FORWARD_N96
#define LLMK_SM120_FORWARD_N96 1
#endif
#ifndef LLMK_SM120_BACKWARD_N96
#define LLMK_SM120_BACKWARD_N96 0
#endif
#ifndef LLMK_SM120_HUGE_N_FORWARD_WIDE
#define LLMK_SM120_HUGE_N_FORWARD_WIDE 0
#endif
#ifndef LLMK_SM120_DWEIGHT_N128
#define LLMK_SM120_DWEIGHT_N128 1
#endif
#ifndef LLMK_SM120_DWEIGHT_DIRECT_ACCUM
#define LLMK_SM120_DWEIGHT_DIRECT_ACCUM 1
#endif
inline bool matmul_sm120_use_huge_n_tile(int N) {
    return (N >= LLMK_SM120_HUGE_N_THRESHOLD) && (N % 128 == 0);
}
#endif

// ----------------------------------------------------------------------------
// Bias-grad reduction kernels — ported verbatim from llm.c/llmc/matmul.cuh
// (lines 17-102). These are plain CUDA; TK adds nothing.

template <typename OutFloat, bool UseAuxBuffer>
__global__ void matmul_backward_bias_kernel9(OutFloat* dbias, const floatX* dout,
                                             int B, int T, int OC,
                                             std::bool_constant<UseAuxBuffer>) {
    constexpr const int bdx = 4;
    constexpr const int bdy = WARP_SIZE / bdx;
    assert(blockDim.x == bdx);
    assert(blockDim.y == bdy);

    int warp_d  = (int)threadIdx.x;
    int warp_c  = (int)threadIdx.y;
    int block_d = (int)threadIdx.z;

    const int OC_per_warp = bdy * x128::size;

    int local_oc  = warp_c * x128::size;
    int global_oc = blockIdx.x * OC_per_warp + local_oc;

    int local_bt     = warp_d + bdx * block_d;
    int bt_per_block = bdx * blockDim.z;

    float accumulators[x128::size];
    for (int k = 0; k < x128::size; k++) accumulators[k] = 0.0f;

    if (global_oc < OC) {
        for (int idx = blockIdx.y * bt_per_block + local_bt; idx < B * T;
             idx += gridDim.y * bt_per_block) {
            x128 packed_dout = load128(dout + global_oc + idx * OC);
            for (int k = 0; k < x128::size; k++)
                accumulators[k] += (float)packed_dout[k];
        }
    }

    __shared__ float sub_results[x128::size][WARP_SIZE][bdy];

    for (int k = 0; k < x128::size; k++) {
        float v = accumulators[k];
        v += __shfl_down_sync(0xffffffff, v, 1, 4);
        v += __shfl_down_sync(0xffffffff, v, 2, 4);
        if (warp_d == 0) sub_results[k][block_d][warp_c] = v;
    }
    __syncthreads();

    for (int k = block_d; k < x128::size; k += blockDim.z) {
        float a = 0.f;
        for (int r = warp_d; r < blockDim.z; r += bdx) {
            float v = sub_results[k][r][warp_c];
            v += __shfl_down_sync(0xffffffff, v, 1, 4);
            v += __shfl_down_sync(0xffffffff, v, 2, 4);
            a += v;
        }
        if (warp_d == 0 && global_oc < OC) {
            if constexpr (!UseAuxBuffer) {
                dbias[global_oc + k] = (OutFloat)(a + (float)dbias[global_oc + k]);
            } else {
                dbias[global_oc + k + blockIdx.y * OC] = a;
            }
        }
    }
}

__global__ void reduce_add_sum_kernel(floatX* dst, const float* src, size_t n, size_t m) {
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * f128::size;
    assert(n % x128::size == 0);
    if (idx < n) {
        f128 acc;
        for (int k = 0; k < f128::size; ++k) acc[k] = 0.f;
        for (int l = 0; l < m; ++l) {
            f128 s = load128(src + idx + n * l);
            for (int k = 0; k < f128::size; ++k) acc[k] += s[k];
        }
        for (int k = 0; k < f128::size; ++k)
            dst[idx + k] = (floatX)((float)dst[idx + k] + acc[k]);
    }
}

// ----------------------------------------------------------------------------
// Bias-add kernel for the default forward path. The opt-in MLP up-projection
// path can fold bias into the TK finish stage via matmul_forward_gelu().

__global__ void add_bias_kernel(floatX* out, const floatX* bias, int N, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * OC) return;
    int oc = idx % OC;
    out[idx] = (floatX)((float)out[idx] + (float)bias[oc]);
}

__global__ void matmul_forward_cuda_kernel(floatX* out, const floatX* inp,
                                           const floatX* weight, const floatX* bias,
                                           int row_offset, int rows, int C, int OC) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    int local_m = blockIdx.y * blockDim.y + threadIdx.y;
    if (local_m >= rows || oc >= OC) return;
    int m = row_offset + local_m;
    float acc = 0.0f;
    for (int c = 0; c < C; ++c) {
        acc += (float)inp[m * C + c] * (float)weight[oc * C + c];
    }
    if (bias != nullptr) {
        acc += (float)bias[oc];
    }
    out[(size_t)m * OC + oc] = (floatX)acc;
}

__global__ void matmul_forward_gelu_cuda_kernel(floatX* out, floatX* pre_gelu,
                                                const floatX* inp, const floatX* weight,
                                                const floatX* bias,
                                                int row_offset, int rows, int C, int OC) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    int local_m = blockIdx.y * blockDim.y + threadIdx.y;
    if (local_m >= rows || oc >= OC) return;
    int m = row_offset + local_m;
    float acc = 0.0f;
    for (int c = 0; c < C; ++c) {
        acc += (float)inp[m * C + c] * (float)weight[oc * C + c];
    }
    acc += (float)bias[oc];
    const size_t idx = (size_t)m * OC + oc;
    pre_gelu[idx] = (floatX)acc;
    float cube = 0.044715f * acc * acc * acc;
    out[idx] = (floatX)(0.5f * acc * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (acc + cube))));
}

inline void add_bias(floatX* out, const floatX* bias, int N, int OC, cudaStream_t stream) {
    if (bias == nullptr) return;
    const int block = 256;
    const int grid  = CEIL_DIV(N * OC, block);
    add_bias_kernel<<<grid, block, 0, stream>>>(out, bias, N, OC);
    cudaCheck(cudaGetLastError());
}

inline void matmul_forward_cuda_launch(floatX* out, const floatX* inp,
                                       const floatX* weight, const floatX* bias,
                                       int M, int K, int N, cudaStream_t stream) {
    const dim3 block(16, 16);
    // Oversized GPT-2 LM-head projections exceed 2^31 output elements at
    // B=64,T=1024. Keep those fallback launches small enough for desktop
    // Blackwell while leaving ordinary hidden-size projections as one launch.
    const size_t total = (size_t)M * N;
    const int rows_per_launch = total > 2147483647ULL ? 128 : M;
    for (int row = 0; row < M; row += rows_per_launch) {
        const int rows = (row + rows_per_launch <= M) ? rows_per_launch : (M - row);
        const dim3 grid(CEIL_DIV(N, (int)block.x), CEIL_DIV(rows, (int)block.y));
        matmul_forward_cuda_kernel<<<grid, block, 0, stream>>>(out, inp, weight, bias, row, rows, K, N);
        cudaCheck(cudaGetLastError());
    }
}

inline void matmul_forward_gelu_cuda_launch(floatX* out, floatX* pre_gelu,
                                            const floatX* inp, const floatX* weight,
                                            const floatX* bias,
                                            int M, int K, int N, cudaStream_t stream) {
    const dim3 block(16, 16);
    const size_t total = (size_t)M * N;
    const int rows_per_launch = total > 2147483647ULL ? 128 : M;
    for (int row = 0; row < M; row += rows_per_launch) {
        const int rows = (row + rows_per_launch <= M) ? rows_per_launch : (M - row);
        const dim3 grid(CEIL_DIV(N, (int)block.x), CEIL_DIV(rows, (int)block.y));
        matmul_forward_gelu_cuda_kernel<<<grid, block, 0, stream>>>(
            out, pre_gelu, inp, weight, bias, row, rows, K, N);
        cudaCheck(cudaGetLastError());
    }
}

// ----------------------------------------------------------------------------
// Forward.
//
// out (B*T, OC) = inp (B*T, C) · weight(OC, C)^T + bias (OC)
//
// Default forward keeps bias and GELU as separate passes. GPT-2 MLP up-projection
// can opt into a TK finish-path bias+GELU epilogue via matmul_forward_gelu(),
// which stores the pre-GELU buffer needed by backward.

inline void matmul_forward(floatX* out, const floatX* inp, const floatX* weight,
                           const floatX* bias, int B, int T, int C, int OC,
                           cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int M = B * T;
    const int N = OC;
    const int K = C;

#if defined(KITTENS_SM120) && (defined(LLMK_SM120_USE_CUBLASLT_GEMM) || LLMK_SM120_CUBLASLT_FORWARD_FALLBACK)
    llmk::cublaslt_sm120::matmul(
        out, weight, inp, bias, N, M, K, stream,
        /*transA=*/true, /*transB=*/false,
        /*batch_count=*/0, /*strideA=*/0, /*strideB=*/0, /*strideOut=*/0,
        /*accumulate=*/false, /*pre_gelu=*/nullptr, /*backward=*/false);
    return;
#endif

#if LLMK_USE_TK_GEMM
    assert(M % 128 == 0 && "matmul_forward: B*T must be a multiple of 128");
    assert(K % 64  == 0 && "matmul_forward: C must be a multiple of 64");

    auto* A     = llmk::to_bf16(const_cast<floatX*>(inp));
    auto* B_    = llmk::to_bf16(const_cast<floatX*>(weight));
    auto* C_    = llmk::to_bf16(out);

#if defined(KITTENS_SM120)
#ifndef LLMK_SM120_FUSE_BIAS
#define LLMK_SM120_FUSE_BIAS 1
#endif
    // Shape-specialized SM120 dispatch:
    //   N >= 8192 && N % 128 == 0 → matmul_huge_n_*  (128×128 tile, fewer N-CTAs)
    //   M % 256 == 0 && N % 64 == 0 → matmul_wide_*  (256×64 tile, max M reuse)
    //   else                          → matmul_default_* / small_n_*
    //
    // The bias-present path uses the fused *_nt_bias kernel (epilogue is the
    // same matmul, with bias added inside the kernel) — no trailing
    // add_bias_kernel launch needed.
    //
    // A/B knob: define LLMK_SM120_FORCE_DEFAULT_TILE to disable shape
    // specialization and stick to traits_128x64 (still picks fused bias).
#ifdef LLMK_SM120_FORCE_DEFAULT_TILE
    const bool huge_n = false;
    const bool huge_n_wide = false;
    const bool wide   = false;
#else
    const bool huge_n = matmul_sm120_use_huge_n_tile(N);
    const bool huge_n_wide = huge_n && (LLMK_SM120_HUGE_N_FORWARD_WIDE != 0) &&
                             (M % 256 == 0) && (N % 64 == 0);
    const bool wide   = !huge_n && (M % 256 == 0) && (N % 64 == 0);
#endif
#if LLMK_SM120_FORWARD_N96
    const bool n96 = !huge_n && (N % 96 == 0);
#else
    const bool n96 = false;
#endif

#if LLMK_SM120_FUSE_BIAS
    if (bias != nullptr && huge_n_wide) {
        llmk::gemm::launch<llmk::gemm::matmul_wide_nt_bias>(A, B_, C_, M, N, K, stream, nullptr, llmk::to_bf16(const_cast<floatX*>(bias)));
    } else if (bias != nullptr && huge_n) {
        llmk::gemm::launch<llmk::gemm::matmul_huge_n_nt_bias>(A, B_, C_, M, N, K, stream, nullptr, llmk::to_bf16(const_cast<floatX*>(bias)));
    } else if (bias != nullptr && n96) {
        llmk::gemm::launch<llmk::gemm::matmul_n96_nt_bias>(A, B_, C_, M, N, K, stream, nullptr, llmk::to_bf16(const_cast<floatX*>(bias)));
    } else if (bias != nullptr && wide) {
        llmk::gemm::launch<llmk::gemm::matmul_wide_nt_bias>(A, B_, C_, M, N, K, stream, nullptr, llmk::to_bf16(const_cast<floatX*>(bias)));
    } else if (bias != nullptr && N % 256 == 0) {
        llmk::gemm::launch<llmk::gemm::matmul_default_nt_bias>(A, B_, C_, M, N, K, stream, nullptr, llmk::to_bf16(const_cast<floatX*>(bias)));
    } else if (bias != nullptr) {
        assert(N % 128 == 0 && "matmul_forward: OC must be a multiple of 128");
        llmk::gemm::launch<llmk::gemm::matmul_small_n_nt_bias>(A, B_, C_, M, N, K, stream, nullptr, llmk::to_bf16(const_cast<floatX*>(bias)));
    } else
#endif
    if (huge_n_wide) {
        llmk::gemm::launch<llmk::gemm::matmul_wide_nt>(A, B_, C_, M, N, K, stream);
    } else if (huge_n) {
        llmk::gemm::launch<llmk::gemm::matmul_huge_n_nt>(A, B_, C_, M, N, K, stream);
    } else if (n96) {
        llmk::gemm::launch<llmk::gemm::matmul_n96_nt>(A, B_, C_, M, N, K, stream);
    } else if (wide) {
        llmk::gemm::launch<llmk::gemm::matmul_wide_nt>(A, B_, C_, M, N, K, stream);
    } else if (N % 256 == 0) {
        llmk::gemm::launch<llmk::gemm::matmul_default_nt>(A, B_, C_, M, N, K, stream);
    } else {
        assert(N % 128 == 0 && "matmul_forward: OC must be a multiple of 128");
        llmk::gemm::launch<llmk::gemm::matmul_small_n_nt>(A, B_, C_, M, N, K, stream);
    }
    cudaCheck(cudaGetLastError());
#if LLMK_SM120_FUSE_BIAS
    if (bias == nullptr) return;
    return;
#endif
    add_bias(out, bias, M, OC, stream);
#else
    // SM90 (H100): keep the original binary dispatch; the gemm_h100.cuh path
    // does not expose wide/huge_n aliases.
    if (N % 256 == 0) {
        llmk::gemm::launch<llmk::gemm::matmul_default_nt>(A, B_, C_, M, N, K, stream);
    } else {
        assert(N % 128 == 0 && "matmul_forward: OC must be a multiple of 128");
        llmk::gemm::launch<llmk::gemm::matmul_small_n_nt>(A, B_, C_, M, N, K, stream);
    }
    cudaCheck(cudaGetLastError());
    add_bias(out, bias, M, OC, stream);
#endif
#else
    matmul_forward_cuda_launch(out, inp, weight, bias, M, K, N, stream);
#endif
}

// ----------------------------------------------------------------------------
// Backward CUDA fallback kernels.
//
// matmul_backward prefers TK for dInp and dWeight when the shape is supported.
// These slow kernels remain for unsupported shapes and for the small accumulation
// add after scratch-backed TK dWeight products.

__global__ void matmul_backward_dinp_kernel(floatX* dinp, const floatX* dout,
                                            const floatX* weight, int M, int C, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * C) return;
    int m = idx / C;
    int c = idx % C;
    float acc = 0.0f;
    for (int oc = 0; oc < OC; ++oc) {
        acc += (float)dout[m * OC + oc] * (float)weight[oc * C + c];
    }
    dinp[idx] = (floatX)acc;
}

__global__ void matmul_backward_dweight_kernel(floatX* dweight, const floatX* dout,
                                               const floatX* inp, int M, int C, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * OC) return;
    int oc = idx / C;
    int c = idx % C;
    float acc = 0.0f;
    for (int m = 0; m < M; ++m) {
        acc += (float)inp[m * C + c] * (float)dout[m * OC + oc];
    }
    dweight[idx] = (floatX)((float)dweight[idx] + acc);
}

__global__ void matmul_backward_dbias_kernel(floatX* dbias, const floatX* dout, int M, int OC) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= OC) return;
    float acc = 0.0f;
    for (int m = 0; m < M; ++m) {
        acc += (float)dout[m * OC + oc];
    }
    dbias[oc] = (floatX)((float)dbias[oc] + acc);
}

__global__ void matmul_add_inplace_kernel(floatX* dst, const floatX* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = (floatX)((float)dst[idx] + (float)src[idx]);
}

__global__ void matmul_reduce_bf16_partials_kernel(floatX* dst, const floatX* src,
                                                   size_t n, int parts, bool accumulate) {
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    assert(n % x128::size == 0);
    if (idx >= n) return;
    float acc[x128::size];
    for (int k = 0; k < x128::size; ++k) {
        acc[k] = accumulate ? (float)dst[idx + k] : 0.0f;
    }
    for (int p = 0; p < parts; ++p) {
        x128 vals = load128(src + (size_t)p * n + idx);
        for (int k = 0; k < x128::size; ++k) {
            acc[k] += (float)vals[k];
        }
    }
    x128 out;
    for (int k = 0; k < x128::size; ++k) {
        out[k] = (floatX)acc[k];
    }
    store128(dst + idx, out);
}

struct MatmulSplitKJob {
    bool active = false;
    floatX* out = nullptr;
    const floatX* partials = nullptr;
    const cudaEvent_t* done_events = nullptr;
    size_t out_elements = 0;
    int parts = 0;
    bool accumulate = false;
};

struct MatmulAsyncJob {
    bool active = false;
    cudaEvent_t done_event = nullptr;
};

inline bool matmul_tk_shape_ok(int M, int N, int K) {
#if LLMK_USE_TK_GEMM
    return M % 128 == 0 && K % 64 == 0 && N % 128 == 0;
#else
    (void)M;
    (void)N;
    (void)K;
    return false;
#endif
}

inline bool matmul_forward_gelu_supported(int B, int T, int C, int OC) {
#if defined(KITTENS_SM120) && !defined(LLMK_SM120_USE_CUBLASLT_GEMM)
#ifndef LLMK_SM120_FUSE_GELU
#define LLMK_SM120_FUSE_GELU 1
#endif
#if LLMK_SM120_FUSE_GELU
    return matmul_tk_shape_ok(B * T, OC, C);
#else
    (void)B;
    (void)T;
    (void)C;
    (void)OC;
    return false;
#endif
#else
    return matmul_tk_shape_ok(B * T, OC, C);
#endif
}

inline bool matmul_backward_gelu_fusion_supported() {
#if defined(KITTENS_SM120) && (defined(LLMK_SM120_USE_CUBLASLT_GEMM) || LLMK_SM120_CUBLASLT_DINP_FALLBACK)
    return true;
#elif defined(KITTENS_SM120)
#ifndef LLMK_SM120_FUSE_DGELU
#define LLMK_SM120_FUSE_DGELU 1
#endif
    return LLMK_SM120_FUSE_DGELU != 0;
#else
    return false;
#endif
}

inline void matmul_dispatch_tk_ab(floatX* out, const floatX* a, const floatX* b,
                                  int M, int N, int K, cudaStream_t stream,
                                  const floatX* pre_gelu = nullptr,
                                  bool apply_dgelu = false) {
#if LLMK_USE_TK_GEMM
    auto* A = llmk::to_bf16(const_cast<floatX*>(a));
    auto* B = llmk::to_bf16(const_cast<floatX*>(b));
    auto* C = llmk::to_bf16(out);
    auto* P = llmk::to_bf16(const_cast<floatX*>(pre_gelu));
#if defined(KITTENS_SM120)
#ifdef LLMK_SM120_FORCE_DEFAULT_TILE
    const bool huge_n = false;
    const bool wide   = false;
#else
    const bool huge_n = matmul_sm120_use_huge_n_tile(N);
    const bool wide   = apply_dgelu && !huge_n && (M % 256 == 0) && (N % 64 == 0);
#endif
#if LLMK_SM120_BACKWARD_N96
    const bool n96 = !huge_n && (N % 96 == 0);
#else
    const bool n96 = false;
#endif
#ifndef LLMK_SM120_DINP_DIRECT_BCOL_SMALLK
#define LLMK_SM120_DINP_DIRECT_BCOL_SMALLK 1
#endif
#ifndef LLMK_SM120_DINP_DIRECT_BCOL_K_CAP
#define LLMK_SM120_DINP_DIRECT_BCOL_K_CAP (3 * 768)
#endif
#ifndef LLMK_SM120_DINP_DIRECT_BCOL_LARGEK
#define LLMK_SM120_DINP_DIRECT_BCOL_LARGEK 0
#endif
#ifndef LLMK_SM120_DINP_DIRECT_BCOL_LARGEK_MIN
#define LLMK_SM120_DINP_DIRECT_BCOL_LARGEK_MIN 8192
#endif
    const bool direct_bcol_smallk = !apply_dgelu && (LLMK_SM120_DINP_DIRECT_BCOL_SMALLK != 0) &&
                                    !huge_n && !n96 && (N == 768) &&
                                    (K <= LLMK_SM120_DINP_DIRECT_BCOL_K_CAP);
    const bool direct_bcol_largek = !apply_dgelu && (LLMK_SM120_DINP_DIRECT_BCOL_LARGEK != 0) &&
                                    !huge_n && !n96 && (N == 768) &&
                                    (K >= LLMK_SM120_DINP_DIRECT_BCOL_LARGEK_MIN);
    if (apply_dgelu) {
        assert(P != nullptr && "matmul_dispatch_tk_ab: dGELU fusion requires pre_gelu");
        if (huge_n) {
            llmk::gemm::launch<llmk::gemm::matmul_huge_n_dgelu>(A, B, C, M, N, K, stream, P);
        } else if (n96) {
            llmk::gemm::launch<llmk::gemm::matmul_n96_dgelu>(A, B, C, M, N, K, stream, P);
        } else if (wide) {
            llmk::gemm::launch<llmk::gemm::matmul_wide_dgelu>(A, B, C, M, N, K, stream, P);
        } else if (N % 256 == 0) {
            llmk::gemm::launch<llmk::gemm::matmul_default_dgelu>(A, B, C, M, N, K, stream, P);
        } else {
            llmk::gemm::launch<llmk::gemm::matmul_small_n_dgelu>(A, B, C, M, N, K, stream, P);
        }
    } else if (huge_n) {
        llmk::gemm::launch<llmk::gemm::matmul_huge_n>(A, B, C, M, N, K, stream);
    } else if (n96) {
        llmk::gemm::launch<llmk::gemm::matmul_n96>(A, B, C, M, N, K, stream);
    } else if (direct_bcol_smallk || direct_bcol_largek) {
        llmk::gemm::launch<llmk::gemm::matmul_default_direct_bcol>(A, B, C, M, N, K, stream);
    } else if (wide) {
        llmk::gemm::launch<llmk::gemm::matmul_wide>(A, B, C, M, N, K, stream);
    } else if (N % 256 == 0) {
        llmk::gemm::launch<llmk::gemm::matmul_default>(A, B, C, M, N, K, stream);
    } else {
        llmk::gemm::launch<llmk::gemm::matmul_small_n>(A, B, C, M, N, K, stream);
    }
#else
    if (N % 256 == 0) {
        llmk::gemm::launch<llmk::gemm::matmul_default>(A, B, C, M, N, K, stream);
    } else {
        llmk::gemm::launch<llmk::gemm::matmul_small_n>(A, B, C, M, N, K, stream);
    }
#endif
    cudaCheck(cudaGetLastError());
#else
    (void)out;
    (void)a;
    (void)b;
    (void)M;
    (void)N;
    (void)K;
    (void)stream;
    (void)pre_gelu;
    (void)apply_dgelu;
    assert(false && "matmul_dispatch_tk_ab called without TK GEMM support");
#endif
}

inline void matmul_dispatch_tk_atb(floatX* out, const floatX* a, const floatX* b,
                                   int M, int N, int K, cudaStream_t stream,
                                   bool accumulate = false) {
#if LLMK_USE_TK_GEMM
    auto* A = llmk::to_bf16(const_cast<floatX*>(a));
    auto* B = llmk::to_bf16(const_cast<floatX*>(b));
    auto* C = llmk::to_bf16(out);
#if defined(KITTENS_SM120)
#ifdef LLMK_SM120_FORCE_DEFAULT_TILE
    const bool huge_n = false;
    const bool wide   = false;
#else
    const bool huge_n = matmul_sm120_use_huge_n_tile(N);
    const bool wide   = !huge_n && (M % 256 == 0) && (N % 64 == 0);
#endif
#if LLMK_SM120_BACKWARD_N96
    const bool n96 = !huge_n && (N % 96 == 0);
#else
    const bool n96 = false;
#endif
    if (huge_n) {
        llmk::gemm::launch<llmk::gemm::matmul_huge_n_tn>(A, B, C, M, N, K, stream, nullptr, nullptr, accumulate);
#if LLMK_SM120_DWEIGHT_N128
    } else if (N % 128 == 0) {
        llmk::gemm::launch<llmk::gemm::matmul_n128_tn>(A, B, C, M, N, K, stream, nullptr, nullptr, accumulate);
#endif
    } else if (n96) {
        llmk::gemm::launch<llmk::gemm::matmul_n96_tn>(A, B, C, M, N, K, stream, nullptr, nullptr, accumulate);
    } else if (wide) {
        llmk::gemm::launch<llmk::gemm::matmul_wide_tn>(A, B, C, M, N, K, stream, nullptr, nullptr, accumulate);
    } else if (N % 256 == 0) {
        llmk::gemm::launch<llmk::gemm::matmul_default_tn>(A, B, C, M, N, K, stream, nullptr, nullptr, accumulate);
    } else {
        llmk::gemm::launch<llmk::gemm::matmul_small_n_tn>(A, B, C, M, N, K, stream, nullptr, nullptr, accumulate);
    }
#else
    if (N % 256 == 0) {
        llmk::gemm::launch<llmk::gemm::matmul_default_tn>(A, B, C, M, N, K, stream);
    } else {
        llmk::gemm::launch<llmk::gemm::matmul_small_n_tn>(A, B, C, M, N, K, stream);
    }
#endif
    cudaCheck(cudaGetLastError());
#else
    (void)out;
    (void)a;
    (void)b;
    (void)M;
    (void)N;
    (void)K;
    (void)stream;
    (void)accumulate;
    assert(false && "matmul_dispatch_tk_atb called without TK GEMM support");
#endif
}

inline bool matmul_dispatch_tk_atb_splitk_start(MatmulSplitKJob* job,
                                                floatX* out, const floatX* a, const floatX* b,
                                                int M, int N, int K, cudaStream_t stream,
                                                bool accumulate,
                                                floatX* partials,
                                                size_t partial_elements);
inline void matmul_dispatch_tk_atb_splitk_finish(const MatmulSplitKJob& job, cudaStream_t stream);
inline bool matmul_dispatch_tk_atb_async_start(MatmulAsyncJob* job,
                                               floatX* out, const floatX* a, const floatX* b,
                                               int M, int N, int K, cudaStream_t stream,
                                               bool accumulate);
inline void matmul_dispatch_tk_atb_async_finish(const MatmulAsyncJob& job, cudaStream_t stream);

inline bool matmul_dispatch_tk_atb_splitk(floatX* out, const floatX* a, const floatX* b,
                                          int M, int N, int K, cudaStream_t stream,
                                          bool accumulate,
                                          floatX* partials,
                                          size_t partial_elements) {
    MatmulSplitKJob job;
    if (!matmul_dispatch_tk_atb_splitk_start(&job, out, a, b, M, N, K, stream,
                                             accumulate, partials, partial_elements)) {
        return false;
    }
    matmul_dispatch_tk_atb_splitk_finish(job, stream);
    return true;
}

inline bool matmul_dispatch_tk_atb_splitk_start(MatmulSplitKJob* job,
                                                floatX* out, const floatX* a, const floatX* b,
                                                int M, int N, int K, cudaStream_t stream,
                                                bool accumulate,
                                                floatX* partials,
                                                size_t partial_elements) {
#if LLMK_USE_TK_GEMM && defined(KITTENS_SM120)
#ifndef LLMK_SM120_DWEIGHT_SPLIT_K
#define LLMK_SM120_DWEIGHT_SPLIT_K 8
#endif
#ifndef LLMK_SM120_DWEIGHT_SPLIT_K_STREAMS
#define LLMK_SM120_DWEIGHT_SPLIT_K_STREAMS 1
#endif
    const size_t out_elements = (size_t)M * N;
    int requested_parts = LLMK_SM120_DWEIGHT_SPLIT_K;
    if (requested_parts > 8 && M != 3 * N) {
        requested_parts = 8;
    }
    int parts = (int)min((size_t)requested_parts, partial_elements / out_elements);
    while (parts > 1) {
        const int split_k = K / parts;
        if (K % parts == 0 && matmul_tk_shape_ok(M, N, split_k)) break;
        --parts;
    }

    if (partials == nullptr || parts <= 1) return false;

    const int split_k = K / parts;
    *job = {};
    job->active = true;
    job->out = out;
    job->partials = partials;
    job->out_elements = out_elements;
    job->parts = parts;
    job->accumulate = accumulate;
#if LLMK_SM120_DWEIGHT_SPLIT_K_STREAMS
    static bool initialized = false;
    static cudaStream_t part_streams[LLMK_SM120_DWEIGHT_SPLIT_K];
    static cudaEvent_t ready_event;
    static cudaEvent_t done_events[LLMK_SM120_DWEIGHT_SPLIT_K];
    if (!initialized) {
        cudaCheck(cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming));
        for (int part = 0; part < LLMK_SM120_DWEIGHT_SPLIT_K; ++part) {
            cudaCheck(cudaStreamCreateWithFlags(&part_streams[part], cudaStreamNonBlocking));
            cudaCheck(cudaEventCreateWithFlags(&done_events[part], cudaEventDisableTiming));
        }
        initialized = true;
    }
    cudaCheck(cudaEventRecord(ready_event, stream));
    for (int part = 0; part < parts; ++part) {
        cudaCheck(cudaStreamWaitEvent(part_streams[part], ready_event, 0));
        const int k0 = part * split_k;
        matmul_dispatch_tk_atb(
            partials + (size_t)part * out_elements,
            a + (size_t)k0 * M,
            b + (size_t)k0 * N,
            M, N, split_k, part_streams[part]);
        cudaCheck(cudaEventRecord(done_events[part], part_streams[part]));
    }
    job->done_events = done_events;
#else
    for (int part = 0; part < parts; ++part) {
        const int k0 = part * split_k;
        matmul_dispatch_tk_atb(
            partials + (size_t)part * out_elements,
            a + (size_t)k0 * M,
            b + (size_t)k0 * N,
            M, N, split_k, stream);
    }
#endif
    return true;
#else
    (void)job;
    (void)out;
    (void)a;
    (void)b;
    (void)M;
    (void)N;
    (void)K;
    (void)stream;
    (void)accumulate;
    (void)partials;
    (void)partial_elements;
    return false;
#endif
}

inline void matmul_dispatch_tk_atb_splitk_finish(const MatmulSplitKJob& job, cudaStream_t stream) {
#if LLMK_USE_TK_GEMM && defined(KITTENS_SM120)
    if (!job.active) return;
#if LLMK_SM120_DWEIGHT_SPLIT_K_STREAMS
    for (int part = 0; part < job.parts; ++part) {
        cudaCheck(cudaStreamWaitEvent(stream, job.done_events[part], 0));
    }
#endif
    const int block = 256;
    const int grid = CEIL_DIV(job.out_elements, block * x128::size);
    matmul_reduce_bf16_partials_kernel<<<grid, block, 0, stream>>>(
        job.out, job.partials, job.out_elements, job.parts, job.accumulate);
    cudaCheck(cudaGetLastError());
#else
    (void)job;
    (void)stream;
#endif
}

inline bool matmul_dispatch_tk_atb_async_start(MatmulAsyncJob* job,
                                               floatX* out, const floatX* a, const floatX* b,
                                               int M, int N, int K, cudaStream_t stream,
                                               bool accumulate) {
#if LLMK_USE_TK_GEMM && defined(KITTENS_SM120)
    static bool initialized = false;
    static cudaStream_t dweight_stream;
    static cudaEvent_t ready_event;
    static cudaEvent_t done_event;
    if (!initialized) {
        cudaCheck(cudaStreamCreateWithFlags(&dweight_stream, cudaStreamNonBlocking));
        cudaCheck(cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming));
        cudaCheck(cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming));
        initialized = true;
    }
    *job = {};
    cudaCheck(cudaEventRecord(ready_event, stream));
    cudaCheck(cudaStreamWaitEvent(dweight_stream, ready_event, 0));
    matmul_dispatch_tk_atb(out, a, b, M, N, K, dweight_stream, accumulate);
    cudaCheck(cudaEventRecord(done_event, dweight_stream));
    job->active = true;
    job->done_event = done_event;
    return true;
#else
    (void)job;
    (void)out;
    (void)a;
    (void)b;
    (void)M;
    (void)N;
    (void)K;
    (void)stream;
    (void)accumulate;
    return false;
#endif
}

inline void matmul_dispatch_tk_atb_async_finish(const MatmulAsyncJob& job, cudaStream_t stream) {
#if LLMK_USE_TK_GEMM && defined(KITTENS_SM120)
    if (!job.active) return;
    cudaCheck(cudaStreamWaitEvent(stream, job.done_event, 0));
#else
    (void)job;
    (void)stream;
#endif
}

inline void matmul_forward_gelu(floatX* out, floatX* pre_gelu,
                                const floatX* inp, const floatX* weight,
                                const floatX* bias, int B, int T, int C, int OC,
                                cudaStream_t stream) {
    NVTX_RANGE_FN();
    assert(out != nullptr);
    assert(pre_gelu != nullptr);
    assert(bias != nullptr && "matmul_forward_gelu: fused path expects a bias vector");
    const int M = B * T;
    const int N = OC;
    const int K = C;

#if defined(KITTENS_SM120) && (defined(LLMK_SM120_USE_CUBLASLT_GEMM) || LLMK_SM120_CUBLASLT_FORWARD_FALLBACK)
    llmk::cublaslt_sm120::matmul(
        out, weight, inp, bias, N, M, K, stream,
        /*transA=*/true, /*transB=*/false,
        /*batch_count=*/0, /*strideA=*/0, /*strideB=*/0, /*strideOut=*/0,
        /*accumulate=*/false, /*pre_gelu=*/pre_gelu, /*backward=*/false);
#elif LLMK_USE_TK_GEMM
    assert(matmul_forward_gelu_supported(B, T, C, OC));
    auto* A = llmk::to_bf16(const_cast<floatX*>(inp));
    auto* B_= llmk::to_bf16(const_cast<floatX*>(weight));
    auto* C_= llmk::to_bf16(out);
    auto* P_= llmk::to_bf16(pre_gelu);
    auto* bias_ = llmk::to_bf16(const_cast<floatX*>(bias));

#if defined(KITTENS_SM120)
#ifdef LLMK_SM120_FORCE_DEFAULT_TILE
    const bool wide = false;
#else
    const bool wide = (M % 256 == 0) && (N % 64 == 0);
#endif
#if LLMK_SM120_FORWARD_N96
    const bool n96 = N % 96 == 0;
#else
    const bool n96 = false;
#endif
    if (n96) {
        llmk::gemm::launch<llmk::gemm::matmul_n96_nt_bias_gelu>(
            A, B_, C_, M, N, K, stream, P_, bias_);
    } else if (wide) {
        llmk::gemm::launch<llmk::gemm::matmul_wide_nt_bias_gelu>(
            A, B_, C_, M, N, K, stream, P_, bias_);
    } else if (N % 256 == 0) {
        llmk::gemm::launch<llmk::gemm::matmul_default_nt_bias_gelu>(
            A, B_, C_, M, N, K, stream, P_, bias_);
    } else {
        llmk::gemm::launch<llmk::gemm::matmul_small_n_nt_bias_gelu>(
            A, B_, C_, M, N, K, stream, P_, bias_);
    }
#else
    if (N % 256 == 0) {
        llmk::gemm::launch<llmk::gemm::matmul_default_nt_bias_gelu>(
            A, B_, C_, M, N, K, stream, P_, bias_);
    } else {
        llmk::gemm::launch<llmk::gemm::matmul_small_n_nt_bias_gelu>(
            A, B_, C_, M, N, K, stream, P_, bias_);
    }
#endif
    cudaCheck(cudaGetLastError());
#else
    matmul_forward_gelu_cuda_launch(out, pre_gelu, inp, weight, bias, M, K, N, stream);
#endif
}

inline void matmul_backward_bias(floatX* dbias, const floatX* dout, float* dbias_buffer,
                                 int B, int T, int OC, cudaStream_t stream) {
#if defined(KITTENS_SM120)
#ifndef LLMK_SM120_BIAS_BLOCK_SIZE
#define LLMK_SM120_BIAS_BLOCK_SIZE 512
#endif
    const int block_size = LLMK_SM120_BIAS_BLOCK_SIZE;
#else
    const int block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;
#endif
    dim3 block_dim = {4, 8, (unsigned)block_size / WARP_SIZE};
    const int OC_per_warp = block_dim.y * x128::size;
    const int grid_size_x = CEIL_DIV(OC, OC_per_warp);
    const int grid_size_y = max(1, deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount
                                   / (block_size * grid_size_x));

    if (grid_size_y == 1) {
        matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(
            dbias, dout, B, T, OC, std::bool_constant<false>{});
        cudaCheck(cudaGetLastError());
    } else if (dbias_buffer != nullptr) {
        matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(
            dbias_buffer, dout, B, T, OC, std::bool_constant<true>{});
        cudaCheck(cudaGetLastError());
        reduce_add_sum_kernel<<<CEIL_DIV(OC, 256 * f128::size), 256, 0, stream>>>(
            dbias, dbias_buffer, OC, grid_size_y);
        cudaCheck(cudaGetLastError());
    } else {
        const int block = 256;
        int grid = CEIL_DIV(OC, block);
        matmul_backward_dbias_kernel<<<grid, block, 0, stream>>>(dbias, dout, B * T, OC);
        cudaCheck(cudaGetLastError());
    }
}

inline void matmul_backward(floatX* dinp, floatX* dweight, floatX* dbias,
                            const floatX* dout, const floatX* inp,
                            const floatX* weight, float* dbias_buffer,
                            int B, int T, int C, int OC,
                            cudaStream_t stream,
                            bool dweight_accumulate = true,
                            floatX* dweight_accum_scratch = nullptr,
                            size_t dweight_accum_scratch_elements = 0,
                            floatX* pre_gelu = nullptr,
                            bool fuse_backward_gelu = false) {
    NVTX_RANGE_FN();
    const int M = B * T;
    const int block = 256;
#if defined(LLMK_SM120_USE_CUBLASLT_GEMM) && defined(KITTENS_SM120)
    (void)block;
#endif

#if !(defined(LLMK_SM120_USE_CUBLASLT_GEMM) && defined(KITTENS_SM120)) && !LLMK_SM120_CUBLASLT_DWEIGHT_FALLBACK
    MatmulSplitKJob dweight_split_job;
    MatmulAsyncJob dweight_direct_job;
    bool dweight_split_started = false;
    bool dweight_direct_started = false;
    bool dweight_split_finish_pending = false;
    bool dweight_direct_finish_pending = false;
#endif
#if LLMK_USE_TK_GEMM && defined(KITTENS_SM120) && !defined(LLMK_SM120_USE_CUBLASLT_GEMM) && !LLMK_SM120_CUBLASLT_DWEIGHT_FALLBACK
#ifndef LLMK_SM120_OVERLAP_DINP_DWEIGHT
#define LLMK_SM120_OVERLAP_DINP_DWEIGHT 1
#endif
#ifndef LLMK_SM120_OVERLAP_DIRECT_DWEIGHT
#define LLMK_SM120_OVERLAP_DIRECT_DWEIGHT 1
#endif
#if LLMK_SM120_OVERLAP_DINP_DWEIGHT
    if (dweight != nullptr && matmul_tk_shape_ok(OC, C, M)) {
        dweight_split_started = matmul_dispatch_tk_atb_splitk_start(
            &dweight_split_job, dweight, dout, inp, OC, C, M, stream,
            dweight_accumulate, dweight_accum_scratch, dweight_accum_scratch_elements);
#if LLMK_SM120_OVERLAP_DIRECT_DWEIGHT
        if (!dweight_split_started) {
            dweight_direct_started = matmul_dispatch_tk_atb_async_start(
                &dweight_direct_job, dweight, dout, inp, OC, C, M, stream,
                dweight_accumulate);
        }
#endif
    }
#endif
#endif

    if (dinp != nullptr) {
#if defined(KITTENS_SM120) && (defined(LLMK_SM120_USE_CUBLASLT_GEMM) || LLMK_SM120_CUBLASLT_DINP_FALLBACK)
        llmk::cublaslt_sm120::matmul(
            dinp, weight, dout, nullptr, C, M, OC, stream,
            /*transA=*/false, /*transB=*/false,
            /*batch_count=*/0, /*strideA=*/0, /*strideB=*/0, /*strideOut=*/0,
            /*accumulate=*/false,
            /*pre_gelu=*/fuse_backward_gelu ? pre_gelu : nullptr,
            /*backward=*/true);
#else
        if (matmul_tk_shape_ok(M, C, OC)) {
            matmul_dispatch_tk_ab(
                dinp, dout, weight, M, C, OC, stream,
                fuse_backward_gelu ? pre_gelu : nullptr,
                fuse_backward_gelu);
        } else {
            assert(!fuse_backward_gelu && "matmul_backward: fused dGELU requested for unsupported TK shape");
            int grid = CEIL_DIV(M * C, block);
            matmul_backward_dinp_kernel<<<grid, block, 0, stream>>>(dinp, dout, weight, M, C, OC);
            cudaCheck(cudaGetLastError());
        }
#endif
    }
    if (dweight != nullptr) {
        const size_t dweight_elements = (size_t)OC * C;
#if defined(KITTENS_SM120) && (defined(LLMK_SM120_USE_CUBLASLT_GEMM) || LLMK_SM120_CUBLASLT_DWEIGHT_FALLBACK)
        (void)dweight_elements;
        llmk::cublaslt_sm120::matmul(
            dweight, inp, dout, nullptr, C, OC, M, stream,
            /*transA=*/false, /*transB=*/true,
            /*batch_count=*/0, /*strideA=*/0, /*strideB=*/0, /*strideOut=*/0,
            /*accumulate=*/dweight_accumulate, /*pre_gelu=*/nullptr, /*backward=*/true);
#else
        if (matmul_tk_shape_ok(OC, C, M)) {
            if (dweight_split_started) {
                dweight_split_finish_pending = true;
            } else if (dweight_direct_started) {
                dweight_direct_finish_pending = true;
            } else if (matmul_dispatch_tk_atb_splitk(
                    dweight, dout, inp, OC, C, M, stream, dweight_accumulate,
                    dweight_accum_scratch, dweight_accum_scratch_elements)) {
                // split-K path wrote or accumulated dweight directly
            } else if (!dweight_accumulate) {
                matmul_dispatch_tk_atb(dweight, dout, inp, OC, C, M, stream);
#if LLMK_SM120_DWEIGHT_DIRECT_ACCUM
            } else {
                matmul_dispatch_tk_atb(dweight, dout, inp, OC, C, M, stream,
                                       /*accumulate=*/true);
            }
#else
            } else if (dweight_accum_scratch != nullptr &&
                       dweight_accum_scratch_elements >= dweight_elements) {
                matmul_dispatch_tk_atb(dweight_accum_scratch, dout, inp, OC, C, M, stream);
                int grid = CEIL_DIV(dweight_elements, block);
                matmul_add_inplace_kernel<<<grid, block, 0, stream>>>(
                    dweight, dweight_accum_scratch, dweight_elements);
                cudaCheck(cudaGetLastError());
            } else {
                int grid = CEIL_DIV(dweight_elements, block);
                matmul_backward_dweight_kernel<<<grid, block, 0, stream>>>(
                    dweight, dout, inp, M, C, OC);
                cudaCheck(cudaGetLastError());
            }
#endif
        } else {
            int grid = CEIL_DIV(dweight_elements, block);
            matmul_backward_dweight_kernel<<<grid, block, 0, stream>>>(dweight, dout, inp, M, C, OC);
            cudaCheck(cudaGetLastError());
        }
#endif
    }
    if (dbias != nullptr) {
        matmul_backward_bias(dbias, dout, dbias_buffer, B, T, OC, stream);
    }
#if !(defined(LLMK_SM120_USE_CUBLASLT_GEMM) && defined(KITTENS_SM120)) && !LLMK_SM120_CUBLASLT_DWEIGHT_FALLBACK
    if (dweight_split_finish_pending) {
        matmul_dispatch_tk_atb_splitk_finish(dweight_split_job, stream);
    }
    if (dweight_direct_finish_pending) {
        matmul_dispatch_tk_atb_async_finish(dweight_direct_job, stream);
    }
#endif
}
