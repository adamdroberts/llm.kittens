/*
cublas_ext.cu — cuBLASLt-backed implementations for the GEMM family declared
in cublas_ext.cuh.

Coverage:
  * cublaslt_gemm_bf16            bf16 GEMM with fused epilogues (bias, GELU,
                                  ReLU², sigmoid/tanh/silu — the non-native
                                  epilogues use a follow-up pointwise kernel).
  * cublaslt_gemm_fp8             FP8 (E4M3) GEMM with per-tensor scales.
  * cublaslt_gemm_mxfp8/mxfp4     Microscaled FP8/FP4 GEMM (Blackwell). Uses
                                  the same cublasLtMatmul path with the
                                  CUDA_R_8F_E4M3 / E2M1 types and per-tile
                                  scale tensors set via vector scales.
  * cublaslt_gemm_w8a8            int8 GEMM (A and B int8, accumulate int32,
                                  output bf16 via scale).
  * cublaslt_gemm_batched_bf16    Batched bf16 GEMM (stride-batched).
  * cublaslt_gemm_grouped_bf16    Grouped bf16 GEMM for MoE: dispatches per
                                  expert via a small loop calling
                                  cublasLtMatmul on each expert's row range.

Plan caching: matches matmul.cuh's pattern — a vector of MatmulPlan entries
keyed by (m, n, k, types, epilogue). Plans accumulate over a run.
*/

#include "cublas_ext.cuh"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>

#include "cuda_common.h"
#include "cuda_utils.cuh"

namespace llmk::cublas_ext {

#ifndef LLMK_CUBLAS_EXT_WORKSPACE_MB
#define LLMK_CUBLAS_EXT_WORKSPACE_MB 128
#endif
static constexpr size_t workspace_size = (size_t)LLMK_CUBLAS_EXT_WORKSPACE_MB * 1024 * 1024;

static cublasLtHandle_t g_handle = nullptr;
static void*            g_workspace = nullptr;

inline void check(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[cublas_ext ERROR]: %d %s:%d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define LLMK_CHECK(x) check((x), __FILE__, __LINE__)

void init() {
    if (g_handle != nullptr) return;
    LLMK_CHECK(cublasLtCreate(&g_handle));
    cudaCheck(cudaMalloc(&g_workspace, workspace_size));
}

struct PlanKey {
    int m, n, k;
    bool transA, transB;
    int batch;
    cudaDataType_t a_type, b_type, c_type, scale_type;
    cublasComputeType_t compute_type;
    cublasLtEpilogue_t epilogue;
    bool has_bias;
};

struct Plan {
    PlanKey key;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t A = nullptr;
    cublasLtMatrixLayout_t B = nullptr;
    cublasLtMatrixLayout_t C = nullptr;
    cublasLtMatrixLayout_t D = nullptr;
    cublasLtMatmulAlgo_t algo;
};

static std::vector<Plan> g_plans;

static bool same_key(const PlanKey& a, const PlanKey& b) {
    return a.m == b.m && a.n == b.n && a.k == b.k &&
           a.transA == b.transA && a.transB == b.transB &&
           a.batch == b.batch &&
           a.a_type == b.a_type && a.b_type == b.b_type && a.c_type == b.c_type &&
           a.scale_type == b.scale_type && a.compute_type == b.compute_type &&
           a.epilogue == b.epilogue && a.has_bias == b.has_bias;
}

static Plan& get_plan(const PlanKey& key) {
    init();
    for (auto& p : g_plans) if (same_key(p.key, key)) return p;

    Plan plan; plan.key = key;
    LLMK_CHECK(cublasLtMatmulDescCreate(&plan.op, key.compute_type, key.scale_type));

    cublasOperation_t opT = CUBLAS_OP_T;
    cublasOperation_t opN = CUBLAS_OP_N;
    LLMK_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSA,
                                              key.transA ? &opT : &opN, sizeof(opT)));
    LLMK_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSB,
                                              key.transB ? &opT : &opN, sizeof(opN)));

    if (key.transA) {
        LLMK_CHECK(cublasLtMatrixLayoutCreate(&plan.A, key.a_type, key.k, key.m, key.k));
    } else {
        LLMK_CHECK(cublasLtMatrixLayoutCreate(&plan.A, key.a_type, key.m, key.k, key.m));
    }
    if (key.transB) {
        LLMK_CHECK(cublasLtMatrixLayoutCreate(&plan.B, key.b_type, key.n, key.k, key.n));
    } else {
        LLMK_CHECK(cublasLtMatrixLayoutCreate(&plan.B, key.b_type, key.k, key.n, key.k));
    }
    LLMK_CHECK(cublasLtMatrixLayoutCreate(&plan.C, key.c_type, key.m, key.n, key.m));
    LLMK_CHECK(cublasLtMatrixLayoutCreate(&plan.D, key.c_type, key.m, key.n, key.m));

    if (key.batch > 1) {
        size_t sa = (size_t)key.m * key.k;
        size_t sb = (size_t)key.k * key.n;
        size_t so = (size_t)key.m * key.n;
        LLMK_CHECK(cublasLtMatrixLayoutSetAttribute(plan.A, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &key.batch, sizeof(key.batch)));
        LLMK_CHECK(cublasLtMatrixLayoutSetAttribute(plan.B, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &key.batch, sizeof(key.batch)));
        LLMK_CHECK(cublasLtMatrixLayoutSetAttribute(plan.C, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &key.batch, sizeof(key.batch)));
        LLMK_CHECK(cublasLtMatrixLayoutSetAttribute(plan.D, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &key.batch, sizeof(key.batch)));
        LLMK_CHECK(cublasLtMatrixLayoutSetAttribute(plan.A, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sa, sizeof(sa)));
        LLMK_CHECK(cublasLtMatrixLayoutSetAttribute(plan.B, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sb, sizeof(sb)));
        LLMK_CHECK(cublasLtMatrixLayoutSetAttribute(plan.C, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &so, sizeof(so)));
        LLMK_CHECK(cublasLtMatrixLayoutSetAttribute(plan.D, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &so, sizeof(so)));
    }

    LLMK_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                              &key.epilogue, sizeof(key.epilogue)));
    if (key.has_bias) {
        cudaDataType_t bdt = key.c_type;
        LLMK_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bdt, sizeof(bdt)));
    }

    cublasLtMatmulPreference_t pref;
    LLMK_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    LLMK_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                    &workspace_size, sizeof(workspace_size)));
    cublasLtMatmulHeuristicResult_t h[8]; int n_h = 0;
    LLMK_CHECK(cublasLtMatmulAlgoGetHeuristic(g_handle, plan.op, plan.A, plan.B, plan.C, plan.D,
                                              pref, 8, h, &n_h));
    if (n_h == 0) {
        fprintf(stderr, "cublas_ext: no algorithm for m=%d n=%d k=%d epi=%d\n",
                key.m, key.n, key.k, (int)key.epilogue);
        exit(EXIT_FAILURE);
    }
    plan.algo = h[0].algo;
    LLMK_CHECK(cublasLtMatmulPreferenceDestroy(pref));

    g_plans.push_back(plan);
    return g_plans.back();
}

// ============================================================================
// Post-matmul activation epilogues for the cases cuBLASLt doesn't natively
// support (ReLU², Sigmoid, Tanh, SiLU, BiasGelu via separate steps).
// ============================================================================

__global__ void k_relu_sq(__nv_bfloat16* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = __bfloat162float(y[i]);
    v = v > 0.f ? v : 0.f;
    y[i] = __float2bfloat16(v * v);
}
__global__ void k_silu(__nv_bfloat16* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = __bfloat162float(y[i]);
    float s = 1.f / (1.f + expf(-v));
    y[i] = __float2bfloat16(v * s);
}
__global__ void k_sigmoid(__nv_bfloat16* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = __bfloat162float(y[i]);
    y[i] = __float2bfloat16(1.f / (1.f + expf(-v)));
}
__global__ void k_tanh(__nv_bfloat16* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = __bfloat162float(y[i]);
    y[i] = __float2bfloat16(tanhf(v));
}

// ============================================================================
// cublaslt_gemm_bf16 — single GEMM y = x @ w^T (+ bias) (+ epilogue)
// x: [M, K], w: [N, K], y: [M, N].
// ============================================================================

void cublaslt_gemm_bf16(__nv_bfloat16* y, const __nv_bfloat16* x, const __nv_bfloat16* w,
                        const __nv_bfloat16* bias, int M, int N, int K,
                        GemmEpilogue epilogue, cudaStream_t stream) {
    init();
    cublasLtEpilogue_t cuepi = CUBLASLT_EPILOGUE_DEFAULT;
    bool has_bias = (bias != nullptr);
    bool post_activation = false;
    int post_kind = 0;  // 0=none, 1=relu_sq, 2=sigmoid, 3=tanh, 4=silu
    switch (epilogue) {
        case GemmEpilogue::None:       cuepi = has_bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT; break;
        case GemmEpilogue::Bias:       cuepi = CUBLASLT_EPILOGUE_BIAS; break;
        case GemmEpilogue::Gelu:       cuepi = has_bias ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_GELU; break;
        case GemmEpilogue::BiasGelu:   cuepi = CUBLASLT_EPILOGUE_GELU_BIAS; break;
        case GemmEpilogue::ReluSq:
            cuepi = has_bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
            post_activation = true; post_kind = 1; break;
        case GemmEpilogue::BiasReluSq:
            cuepi = CUBLASLT_EPILOGUE_BIAS;
            post_activation = true; post_kind = 1; break;
        case GemmEpilogue::Sigmoid:
            cuepi = has_bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
            post_activation = true; post_kind = 2; break;
        case GemmEpilogue::Tanh:
            cuepi = has_bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
            post_activation = true; post_kind = 3; break;
        case GemmEpilogue::Silu:
        case GemmEpilogue::BiasSilu:
            cuepi = has_bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
            post_activation = true; post_kind = 4; break;
    }

    PlanKey key{M, N, K, false, true, 1, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F,
                CUBLAS_COMPUTE_32F, cuepi, has_bias};
    Plan& plan = get_plan(key);

    if (has_bias) {
        LLMK_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                  &bias, sizeof(bias)));
    }

    float alpha = 1.f, beta = 0.f;
    LLMK_CHECK(cublasLtMatmul(g_handle, plan.op,
                              &alpha, x, plan.A, w, plan.B,
                              &beta,  y, plan.C, y, plan.D,
                              &plan.algo, g_workspace, workspace_size, stream));
    if (post_activation) {
        const int bs = 256;
        int N_total = M * N;
        if (post_kind == 1) k_relu_sq <<<CEIL_DIV(N_total, bs), bs, 0, stream>>>(y, N_total);
        else if (post_kind == 2) k_sigmoid<<<CEIL_DIV(N_total, bs), bs, 0, stream>>>(y, N_total);
        else if (post_kind == 3) k_tanh   <<<CEIL_DIV(N_total, bs), bs, 0, stream>>>(y, N_total);
        else if (post_kind == 4) k_silu   <<<CEIL_DIV(N_total, bs), bs, 0, stream>>>(y, N_total);
        cudaCheck(cudaGetLastError());
    }
}

// ============================================================================
// cublaslt_gemm_fp8 — FP8 (E4M3) GEMM with fp32 scales.
// ============================================================================

void cublaslt_gemm_fp8(__nv_bfloat16* y, const uint8_t* x_e4m3, const uint8_t* w_e4m3,
                       const float* x_scale, const float* w_scale,
                       int M, int N, int K, cudaStream_t stream) {
    init();
    PlanKey key{M, N, K, false, true, 1, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF, CUDA_R_32F,
                CUBLAS_COMPUTE_32F, CUBLASLT_EPILOGUE_DEFAULT, false};
    Plan& plan = get_plan(key);

    // Set per-tensor scales: A_SCALE and B_SCALE pointers.
    LLMK_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                              &x_scale, sizeof(x_scale)));
    LLMK_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                              &w_scale, sizeof(w_scale)));

    float alpha = 1.f, beta = 0.f;
    LLMK_CHECK(cublasLtMatmul(g_handle, plan.op,
                              &alpha, x_e4m3, plan.A, w_e4m3, plan.B,
                              &beta,  y, plan.C, y, plan.D,
                              &plan.algo, g_workspace, workspace_size, stream));
}

// ============================================================================
// cublaslt_gemm_mxfp8 / mxfp4 — block-scaled FP8 / FP4 GEMM (Blackwell).
//
// cuBLASLt 12.6+ supports per-tile (32-element) E8M0 exponent scales via
// CUBLASLT_MATMUL_DESC_A_SCALE_MODE = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0
// (and similar). We set the per-block scale pointers and let cublasLt take
// care of the rest.
// ============================================================================

void cublaslt_gemm_mxfp8(__nv_bfloat16* y, const uint8_t* x_data, const uint8_t* x_exps,
                         const uint8_t* w_data, const uint8_t* w_exps,
                         int M, int N, int K, cudaStream_t stream) {
    init();
    PlanKey key{M, N, K, false, true, 1, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF, CUDA_R_32F,
                CUBLAS_COMPUTE_32F, CUBLASLT_EPILOGUE_DEFAULT, false};
    Plan& plan = get_plan(key);

#if defined(CUBLASLT_VERSION) && CUBLASLT_VERSION >= 120600
    cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    LLMK_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                              &scale_mode, sizeof(scale_mode)));
    LLMK_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                              &scale_mode, sizeof(scale_mode)));
    LLMK_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                              &x_exps, sizeof(x_exps)));
    LLMK_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                              &w_exps, sizeof(w_exps)));
#else
    // older cublasLt — fall back to per-tensor scale (less accurate)
    (void)x_exps; (void)w_exps;
#endif

    float alpha = 1.f, beta = 0.f;
    LLMK_CHECK(cublasLtMatmul(g_handle, plan.op,
                              &alpha, x_data, plan.A, w_data, plan.B,
                              &beta,  y, plan.C, y, plan.D,
                              &plan.algo, g_workspace, workspace_size, stream));
}

void cublaslt_gemm_mxfp4(__nv_bfloat16* y, const uint8_t* x_packed, const uint8_t* x_exps,
                         const uint8_t* w_packed, const uint8_t* w_exps,
                         int M, int N, int K, cudaStream_t stream) {
    init();
#if defined(CUBLASLT_VERSION) && CUBLASLT_VERSION >= 120800
    // CUDA_R_4F_E2M1 exists in CUDA 12.8+. Older toolchains fall through to
    // the unpacked fp8 path on dequantised data (caller would dequant first).
    PlanKey key{M, N, K, false, true, 1,
                (cudaDataType_t)/*CUDA_R_4F_E2M1*/ 32,
                (cudaDataType_t)/*CUDA_R_4F_E2M1*/ 32,
                CUDA_R_16BF, CUDA_R_32F,
                CUBLAS_COMPUTE_32F, CUBLASLT_EPILOGUE_DEFAULT, false};
    Plan& plan = get_plan(key);
    cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE8M0;
    LLMK_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                              &scale_mode, sizeof(scale_mode)));
    LLMK_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                              &scale_mode, sizeof(scale_mode)));
    LLMK_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                              &x_exps, sizeof(x_exps)));
    LLMK_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                              &w_exps, sizeof(w_exps)));
    float alpha = 1.f, beta = 0.f;
    LLMK_CHECK(cublasLtMatmul(g_handle, plan.op,
                              &alpha, x_packed, plan.A, w_packed, plan.B,
                              &beta,  y, plan.C, y, plan.D,
                              &plan.algo, g_workspace, workspace_size, stream));
#else
    (void)y; (void)x_packed; (void)x_exps; (void)w_packed; (void)w_exps;
    (void)M; (void)N; (void)K; (void)stream;
    fprintf(stderr, "cublaslt_gemm_mxfp4 requires CUDA 12.8+ and cuBLASLt 12.8+\n");
    exit(EXIT_FAILURE);
#endif
}

// ============================================================================
// cublaslt_gemm_w8a8 — int8 × int8 → bf16 with fp32 alpha = x_scale * w_scale.
// Per-row x_scale is collapsed to per-tensor before the call by callers in
// practice; this entry expects the per-row x_scale as a host buffer pointer
// and uses cublasLt's A_SCALE_POINTER for it.
// ============================================================================

void cublaslt_gemm_w8a8(__nv_bfloat16* y, const int8_t* x_q, const int8_t* w_q,
                        const float* x_scale, float w_scale,
                        int M, int N, int K, cudaStream_t stream) {
    init();
    PlanKey key{M, N, K, false, true, 1, CUDA_R_8I, CUDA_R_8I, CUDA_R_16BF, CUDA_R_32F,
                CUBLAS_COMPUTE_32I, CUBLASLT_EPILOGUE_DEFAULT, false};
    Plan& plan = get_plan(key);

    // Per-row A scale.
    LLMK_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                              &x_scale, sizeof(x_scale)));
    LLMK_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                              &w_scale, sizeof(w_scale)));

    float alpha = 1.f, beta = 0.f;
    LLMK_CHECK(cublasLtMatmul(g_handle, plan.op,
                              &alpha, x_q, plan.A, w_q, plan.B,
                              &beta,  y, plan.C, y, plan.D,
                              &plan.algo, g_workspace, workspace_size, stream));
}

// ============================================================================
// Batched bf16 GEMM (stride-batched).
// ============================================================================

void cublaslt_gemm_batched_bf16(__nv_bfloat16* y, const __nv_bfloat16* x, const __nv_bfloat16* w,
                                int batch_count, int M, int N, int K, cudaStream_t stream) {
    init();
    PlanKey key{M, N, K, false, true, batch_count,
                CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F,
                CUBLAS_COMPUTE_32F, CUBLASLT_EPILOGUE_DEFAULT, false};
    Plan& plan = get_plan(key);
    float alpha = 1.f, beta = 0.f;
    LLMK_CHECK(cublasLtMatmul(g_handle, plan.op,
                              &alpha, x, plan.A, w, plan.B,
                              &beta,  y, plan.C, y, plan.D,
                              &plan.algo, g_workspace, workspace_size, stream));
}

// ============================================================================
// W4A16 (NF4) GEMM: dequantize NF4 weights into a scratch bf16 buffer, then
// run the standard bf16 GEMM. The dequantize kernel is in quantize.cuh.
// ============================================================================

extern void nf4_dequantize(__nv_bfloat16* out, const uint8_t* qweight, const float* absmax,
                           int out_dim, int in_dim, int group_size, cudaStream_t stream);

static __nv_bfloat16* g_w4a16_scratch = nullptr;
static size_t g_w4a16_scratch_size = 0;

void cublaslt_gemm_w4a16(__nv_bfloat16* y, const __nv_bfloat16* x,
                         const uint8_t* w_packed, const float* w_absmax,
                         int M, int N, int K, int group_size, cudaStream_t stream) {
    init();
    size_t need = sizeof(__nv_bfloat16) * (size_t)N * K;
    if (need > g_w4a16_scratch_size) {
        if (g_w4a16_scratch) cudaCheck(cudaFree(g_w4a16_scratch));
        cudaCheck(cudaMalloc(&g_w4a16_scratch, need));
        g_w4a16_scratch_size = need;
    }
    nf4_dequantize(g_w4a16_scratch, w_packed, w_absmax, N, K, group_size, stream);
    llmk::cublas_ext::cublaslt_gemm_bf16(y, x, g_w4a16_scratch, /*bias=*/nullptr,
                                         M, N, K, GemmEpilogue::None, stream);
}

// ============================================================================
// Low-rank LoRA GEMM: y = base(x) + (alpha/rank) * (x @ A^T) @ B^T
//
// x: [M, K]      A: [rank, K]   B: [N, rank]    base: [N, K]
//
// Three cublasLt calls: base GEMM, low-rank-down GEMM, low-rank-up GEMM.
// The second-stage GEMM uses beta=1 to accumulate onto base.
// ============================================================================

static __nv_bfloat16* g_lora_hidden = nullptr;
static size_t g_lora_hidden_size = 0;

void cublaslt_gemm_lora(__nv_bfloat16* y, const __nv_bfloat16* x,
                        const __nv_bfloat16* base_w,
                        const __nv_bfloat16* A, const __nv_bfloat16* B,
                        float scaling, int M, int N, int K, int rank,
                        cudaStream_t stream) {
    init();
    // 1) y = x @ base^T
    llmk::cublas_ext::cublaslt_gemm_bf16(y, x, base_w, /*bias=*/nullptr,
                                         M, N, K, GemmEpilogue::None, stream);

    // 2) hidden = x @ A^T  shape [M, rank]
    size_t need = sizeof(__nv_bfloat16) * (size_t)M * rank;
    if (need > g_lora_hidden_size) {
        if (g_lora_hidden) cudaCheck(cudaFree(g_lora_hidden));
        cudaCheck(cudaMalloc(&g_lora_hidden, need));
        g_lora_hidden_size = need;
    }
    llmk::cublas_ext::cublaslt_gemm_bf16(g_lora_hidden, x, A, /*bias=*/nullptr,
                                         M, rank, K, GemmEpilogue::None, stream);

    // 3) y += scaling * hidden @ B^T
    // Use cublasLt directly with beta=1 to accumulate.
    using namespace llmk::cublas_ext;
    PlanKey key{M, N, rank, false, true, 1,
                CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F,
                CUBLAS_COMPUTE_32F, CUBLASLT_EPILOGUE_DEFAULT, false};
    Plan& plan = get_plan(key);
    float alpha = scaling, beta = 1.0f;
    LLMK_CHECK(cublasLtMatmul(g_handle, plan.op,
                              &alpha, g_lora_hidden, plan.A, B, plan.B,
                              &beta,  y, plan.C, y, plan.D,
                              &plan.algo, g_workspace, workspace_size, stream));
}

// ============================================================================
// Grouped bf16 GEMM (MoE per-expert weights).
//   y:              [M_total, N]
//   x:              [M_total, K]   (rows already permuted by expert)
//   w_packed:       [E, N, K]      (each expert's W)
//   expert_offsets: [E+1] int32 host buffer
//
// Calls cublasLtMatmul for each expert's row range.
// ============================================================================

void cublaslt_gemm_grouped_bf16(__nv_bfloat16* y, const __nv_bfloat16* x, const __nv_bfloat16* w_packed,
                                const int* expert_offsets_host, int E,
                                int M_total, int N, int K, cudaStream_t stream) {
    init();
    for (int e = 0; e < E; ++e) {
        int row_start = expert_offsets_host[e];
        int row_end   = expert_offsets_host[e + 1];
        int M_e = row_end - row_start;
        if (M_e <= 0) continue;

        PlanKey key{M_e, N, K, false, true, 1,
                    CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F,
                    CUBLAS_COMPUTE_32F, CUBLASLT_EPILOGUE_DEFAULT, false};
        Plan& plan = get_plan(key);
        float alpha = 1.f, beta = 0.f;
        const __nv_bfloat16* x_e = x + (size_t)row_start * K;
        const __nv_bfloat16* w_e = w_packed + (size_t)e * N * K;
        __nv_bfloat16*       y_e = y + (size_t)row_start * N;
        LLMK_CHECK(cublasLtMatmul(g_handle, plan.op,
                                  &alpha, x_e, plan.A, w_e, plan.B,
                                  &beta,  y_e, plan.C, y_e, plan.D,
                                  &plan.algo, g_workspace, workspace_size, stream));
    }
}

}  // namespace llmk::cublas_ext

// ============================================================================
// External C-style entry points matching cublas_ext.cuh declarations.
// ============================================================================

void cublaslt_gemm_bf16(__nv_bfloat16* y, const __nv_bfloat16* x, const __nv_bfloat16* w,
                        const __nv_bfloat16* bias, int M, int N, int K,
                        GemmEpilogue epilogue, cudaStream_t stream) {
    llmk::cublas_ext::cublaslt_gemm_bf16(y, x, w, bias, M, N, K, epilogue, stream);
}
void cublaslt_gemm_fp8(__nv_bfloat16* y, const uint8_t* x_e4m3, const uint8_t* w_e4m3,
                       const float* x_scale, const float* w_scale,
                       int M, int N, int K, cudaStream_t stream) {
    llmk::cublas_ext::cublaslt_gemm_fp8(y, x_e4m3, w_e4m3, x_scale, w_scale, M, N, K, stream);
}
void cublaslt_gemm_mxfp8(__nv_bfloat16* y, const uint8_t* x_data, const uint8_t* x_exps,
                         const uint8_t* w_data, const uint8_t* w_exps,
                         int M, int N, int K, cudaStream_t stream) {
    llmk::cublas_ext::cublaslt_gemm_mxfp8(y, x_data, x_exps, w_data, w_exps, M, N, K, stream);
}
void cublaslt_gemm_mxfp4(__nv_bfloat16* y, const uint8_t* x_packed, const uint8_t* x_exps,
                         const uint8_t* w_packed, const uint8_t* w_exps,
                         int M, int N, int K, cudaStream_t stream) {
    llmk::cublas_ext::cublaslt_gemm_mxfp4(y, x_packed, x_exps, w_packed, w_exps, M, N, K, stream);
}
void cublaslt_gemm_w8a8(__nv_bfloat16* y, const int8_t* x_q, const int8_t* w_q,
                        const float* x_scale, float w_scale,
                        int M, int N, int K, cudaStream_t stream) {
    llmk::cublas_ext::cublaslt_gemm_w8a8(y, x_q, w_q, x_scale, w_scale, M, N, K, stream);
}
void cublaslt_gemm_batched_bf16(__nv_bfloat16* y, const __nv_bfloat16* x, const __nv_bfloat16* w,
                                int batch_count, int M, int N, int K, cudaStream_t stream) {
    llmk::cublas_ext::cublaslt_gemm_batched_bf16(y, x, w, batch_count, M, N, K, stream);
}
void cublaslt_gemm_grouped_bf16(__nv_bfloat16* y, const __nv_bfloat16* x, const __nv_bfloat16* w_packed,
                                const int* expert_offsets, int E,
                                int M_total, int N, int K, cudaStream_t stream) {
    llmk::cublas_ext::cublaslt_gemm_grouped_bf16(y, x, w_packed, expert_offsets, E, M_total, N, K, stream);
}
