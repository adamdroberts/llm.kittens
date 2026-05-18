/*
test_matmul.cu — smoke test for llmc/matmul.cuh GEMM paths.

Generates random bf16 activations and llm.c-layout weights on device, runs
llmc::matmul_forward, computes a naive reference GEMM on device with FP32
accumulation, and compares. Also checks the opt-in fused bias+GELU epilogue
path for the GPT-2 MLP up-projection and both dWeight backward paths
(`dWeight = dOut^T * Inp` and `dWeight += dOut^T * Inp`) against naive
references.

Build (from llm.kittens/):
    nvcc -std=c++20 -O3 --use_fast_math \
         --expt-extended-lambda --expt-relaxed-constexpr \
         -forward-unknown-to-host-compiler \
         -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing \
         -DKITTENS_SM90 -gencode arch=compute_90a,code=sm_90a \
         -DENABLE_BF16 \
         -I$(TK_ROOT)/include -I$(TK_ROOT)/prototype \
         -I. \
         dev/cuda/test_matmul.cu -o dev/cuda/test_matmul \
         -lcudart -lcuda -lnvidia-ml

Or via the Makefile target: `make test_matmul` (added in the top-level Makefile).
*/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

cudaDeviceProp deviceProp;

#include "llmc/matmul.cuh"

// Reference: naive GEMM with FP32 accumulation. W is stored like llm.c model
// parameters: row-major (N, K), so the logical multiply is A * W^T.
__global__ void naive_gemm_ref(const __nv_bfloat16* A, const __nv_bfloat16* W,
                               __nv_bfloat16* C, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        acc += __bfloat162float(A[m * K + k]) * __bfloat162float(W[n * K + k]);
    }
    C[m * N + n] = __float2bfloat16(acc);
}

__global__ void naive_gemm_bias_gelu_ref(const __nv_bfloat16* A, const __nv_bfloat16* W,
                                         const __nv_bfloat16* bias,
                                         __nv_bfloat16* pre_gelu,
                                         __nv_bfloat16* gelu,
                                         int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        acc += __bfloat162float(A[m * K + k]) * __bfloat162float(W[n * K + k]);
    }
    acc += __bfloat162float(bias[n]);
    pre_gelu[m * N + n] = __float2bfloat16(acc);
    float cube = 0.044715f * acc * acc * acc;
    float out = 0.5f * acc * (1.0f + tanhf(0.7978845608028654f * (acc + cube)));
    gelu[m * N + n] = __float2bfloat16(out);
}

__global__ void naive_dweight_ref(const __nv_bfloat16* dout, const __nv_bfloat16* inp,
                                  __nv_bfloat16* dweight, int M, int C, int OC) {
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C || oc >= OC) return;
    float acc = 0.f;
    for (int m = 0; m < M; ++m) {
        acc += __bfloat162float(dout[m * OC + oc]) * __bfloat162float(inp[m * C + c]);
    }
    dweight[oc * C + c] = __float2bfloat16(acc);
}

__global__ void naive_dinp_ref(const __nv_bfloat16* dout, const __nv_bfloat16* weight,
                               __nv_bfloat16* dinp, int M, int C, int OC) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || c >= C) return;
    float acc = 0.f;
    for (int oc = 0; oc < OC; ++oc) {
        acc += __bfloat162float(dout[m * OC + oc]) * __bfloat162float(weight[oc * C + c]);
    }
    dinp[m * C + c] = __float2bfloat16(acc);
}

__global__ void naive_dweight_accum_ref(const __nv_bfloat16* dout, const __nv_bfloat16* inp,
                                        __nv_bfloat16* dweight, int M, int C, int OC) {
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C || oc >= OC) return;
    float acc = 0.f;
    for (int m = 0; m < M; ++m) {
        acc += __bfloat162float(dout[m * OC + oc]) * __bfloat162float(inp[m * C + c]);
    }
    int idx = oc * C + c;
    dweight[idx] = __float2bfloat16(__bfloat162float(dweight[idx]) + acc);
}

__global__ void naive_dinp_dgelu_ref(const __nv_bfloat16* dout, const __nv_bfloat16* weight,
                                     const __nv_bfloat16* pre_gelu,
                                     __nv_bfloat16* dinp,
                                     int M, int C, int OC) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || c >= C) return;
    float acc = 0.f;
    for (int oc = 0; oc < OC; ++oc) {
        acc += __bfloat162float(dout[m * OC + oc]) * __bfloat162float(weight[oc * C + c]);
    }
    float x = __bfloat162float(pre_gelu[m * C + c]);
    float cube = 0.044715f * x * x * x;
    float tanh_arg = sqrtf(2.0f / M_PI) * (x + cube);
    float tanh_out = tanhf(tanh_arg);
    float coshf_out = coshf(tanh_arg);
    float sech_out = 1.0f / (coshf_out * coshf_out);
    float local_grad = 0.5f * (1.0f + tanh_out)
        + x * 0.5f * sech_out * sqrtf(2.0f / M_PI) * (1.0f + 3.0f * 0.044715f * x * x);
    dinp[m * C + c] = __float2bfloat16(acc * local_grad);
}

static void fill_random_bf16(std::vector<__nv_bfloat16>& h, uint64_t seed, float lo, float hi) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& v : h) v = __float2bfloat16(dist(rng));
}

static double max_abs_diff(const std::vector<__nv_bfloat16>& a,
                           const std::vector<__nv_bfloat16>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = std::abs((double)__bfloat162float(a[i]) - (double)__bfloat162float(b[i]));
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char** argv) {
    int dev = 0;
    cudaCheck(cudaSetDevice(dev));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Device: %s (sm_%d%d)\n", deviceProp.name, deviceProp.major, deviceProp.minor);
    if (deviceProp.major != 9) {
        printf("⚠ This smoke test targets H100 (sm_90); detected sm_%d%d. Continuing anyway.\n",
               deviceProp.major, deviceProp.minor);
    }
#if LLMK_SM120_HAS_CUBLASLT_GEMM
    llmk::cublaslt_sm120::init();
#endif

    // Sweep three shapes that exercise the dispatch logic in matmul_forward:
    //   1. Small square — uses matmul_default<2,4,8>
    //   2. GPT-2 124M MLP up: M=4096 N=3072 K=768 — N%256==0
    //   3. GPT-2 124M LM head: M=4096 N=50304 K=768 — N%256!=0, falls back to small_n
    struct Shape { int M, N, K; const char* name; };
    Shape shapes[] = {
        { 1024, 1024, 1024, "1024^3 square (default)" },
        { 4096, 3072,  768, "GPT-2 124M MLP up (default)" },
        { 4096, 50304, 768, "GPT-2 124M LM head (small_n fallback)" },
    };

    int failures = 0;
    int total_tests = (int)(sizeof(shapes)/sizeof(*shapes));
    for (const auto& s : shapes) {
        printf("\n──── %s  M=%d N=%d K=%d ────\n", s.name, s.M, s.N, s.K);

        size_t a_bytes = (size_t)s.M * s.K * sizeof(__nv_bfloat16);
        size_t w_bytes = (size_t)s.N * s.K * sizeof(__nv_bfloat16);
        size_t c_bytes = (size_t)s.M * s.N * sizeof(__nv_bfloat16);

        std::vector<__nv_bfloat16> hA((size_t)s.M * s.K);
        std::vector<__nv_bfloat16> hW((size_t)s.N * s.K);
        std::vector<__nv_bfloat16> hC_tk((size_t)s.M * s.N);
        std::vector<__nv_bfloat16> hC_ref((size_t)s.M * s.N);

        fill_random_bf16(hA, 42,    -1.0f, 1.0f);
        fill_random_bf16(hW, 1337,  -1.0f, 1.0f);

        __nv_bfloat16 *dA = nullptr, *dW = nullptr, *dC_tk = nullptr, *dC_ref = nullptr;
        cudaCheck(cudaMalloc(&dA,    a_bytes));
        cudaCheck(cudaMalloc(&dW,    w_bytes));
        cudaCheck(cudaMalloc(&dC_tk, c_bytes));
        cudaCheck(cudaMalloc(&dC_ref, c_bytes));

        cudaCheck(cudaMemcpy(dA, hA.data(), a_bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dW, hW.data(), w_bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemset(dC_tk,  0, c_bytes));
        cudaCheck(cudaMemset(dC_ref, 0, c_bytes));

        // Pretend the matmul shape is laid out as (B*T, C) → (B*T, OC). The
        // wrapper expects M=B*T, N=OC, K=C, so we pick B*T=M arbitrarily.
        // No bias for this test.
        matmul_forward(dC_tk, dA, dW, /*bias=*/nullptr,
                       /*B=*/1, /*T=*/s.M, /*C=*/s.K, /*OC=*/s.N, /*stream=*/0);
        cudaCheck(cudaDeviceSynchronize());

        // Reference
        dim3 block(16, 16);
        dim3 grid(CEIL_DIV(s.N, 16), CEIL_DIV(s.M, 16));
        naive_gemm_ref<<<grid, block>>>(dA, dW, dC_ref, s.M, s.N, s.K);
        cudaCheck(cudaDeviceSynchronize());

        cudaCheck(cudaMemcpy(hC_tk.data(),  dC_tk,  c_bytes, cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(hC_ref.data(), dC_ref, c_bytes, cudaMemcpyDeviceToHost));

        double max_diff = max_abs_diff(hC_tk, hC_ref);
        // bf16 has ~3e-3 relative precision; for K=768 with values in [-1,1]
        // accumulation error is ~sqrt(K)*eps ~ 0.08. We use a loose 0.5 bound.
        double tolerance = 0.5;
        bool ok = max_diff < tolerance;
        printf("  max abs diff = %.4f  (tolerance %.2f)  %s\n",
               max_diff, tolerance, ok ? "PASS" : "FAIL");
        if (!ok) failures++;

        cudaCheck(cudaFree(dA));
        cudaCheck(cudaFree(dW));
        cudaCheck(cudaFree(dC_tk));
        cudaCheck(cudaFree(dC_ref));
    }

    {
        Shape s = {1024, 4096, 1024, "forward bias+GELU epilogue (default)"};
        printf("\n──── %s  M=%d N=%d K=%d ────\n", s.name, s.M, s.N, s.K);
        if (!matmul_forward_gelu_supported(/*B=*/1, /*T=*/s.M, /*C=*/s.K, /*OC=*/s.N)) {
            printf("  SKIP: fused bias+GELU is not supported by this build\n");
        } else {
        total_tests++;

        size_t a_bytes = (size_t)s.M * s.K * sizeof(__nv_bfloat16);
        size_t w_bytes = (size_t)s.N * s.K * sizeof(__nv_bfloat16);
        size_t b_bytes = (size_t)s.N * sizeof(__nv_bfloat16);
        size_t c_bytes = (size_t)s.M * s.N * sizeof(__nv_bfloat16);

        std::vector<__nv_bfloat16> hA((size_t)s.M * s.K);
        std::vector<__nv_bfloat16> hW((size_t)s.N * s.K);
        std::vector<__nv_bfloat16> hBias((size_t)s.N);
        std::vector<__nv_bfloat16> hPre_tk((size_t)s.M * s.N);
        std::vector<__nv_bfloat16> hGelu_tk((size_t)s.M * s.N);
        std::vector<__nv_bfloat16> hPre_ref((size_t)s.M * s.N);
        std::vector<__nv_bfloat16> hGelu_ref((size_t)s.M * s.N);

        fill_random_bf16(hA, 8080, -1.0f, 1.0f);
        fill_random_bf16(hW, 9090, -1.0f, 1.0f);
        fill_random_bf16(hBias, 10010, -0.1f, 0.1f);

        __nv_bfloat16 *dA = nullptr, *dW = nullptr, *dBias = nullptr;
        __nv_bfloat16 *dPre_tk = nullptr, *dGelu_tk = nullptr, *dPre_ref = nullptr, *dGelu_ref = nullptr;
        cudaCheck(cudaMalloc(&dA, a_bytes));
        cudaCheck(cudaMalloc(&dW, w_bytes));
        cudaCheck(cudaMalloc(&dBias, b_bytes));
        cudaCheck(cudaMalloc(&dPre_tk, c_bytes));
        cudaCheck(cudaMalloc(&dGelu_tk, c_bytes));
        cudaCheck(cudaMalloc(&dPre_ref, c_bytes));
        cudaCheck(cudaMalloc(&dGelu_ref, c_bytes));

        cudaCheck(cudaMemcpy(dA, hA.data(), a_bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dW, hW.data(), w_bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dBias, hBias.data(), b_bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemset(dPre_tk, 0, c_bytes));
        cudaCheck(cudaMemset(dGelu_tk, 0, c_bytes));
        cudaCheck(cudaMemset(dPre_ref, 0, c_bytes));
        cudaCheck(cudaMemset(dGelu_ref, 0, c_bytes));

        matmul_forward_gelu(dGelu_tk, dPre_tk, dA, dW, dBias,
                            /*B=*/1, /*T=*/s.M, /*C=*/s.K, /*OC=*/s.N, /*stream=*/0);
        cudaCheck(cudaDeviceSynchronize());

        dim3 block(16, 16);
        dim3 grid(CEIL_DIV(s.N, 16), CEIL_DIV(s.M, 16));
        naive_gemm_bias_gelu_ref<<<grid, block>>>(dA, dW, dBias, dPre_ref, dGelu_ref, s.M, s.N, s.K);
        cudaCheck(cudaDeviceSynchronize());

        cudaCheck(cudaMemcpy(hPre_tk.data(), dPre_tk, c_bytes, cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(hGelu_tk.data(), dGelu_tk, c_bytes, cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(hPre_ref.data(), dPre_ref, c_bytes, cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(hGelu_ref.data(), dGelu_ref, c_bytes, cudaMemcpyDeviceToHost));

        double pre_diff = max_abs_diff(hPre_tk, hPre_ref);
        double gelu_diff = max_abs_diff(hGelu_tk, hGelu_ref);
        double tolerance = 0.5;
        bool ok = pre_diff < tolerance && gelu_diff < tolerance;
        printf("  pre-GELU max abs diff = %.4f  GELU max abs diff = %.4f  (tolerance %.2f)  %s\n",
               pre_diff, gelu_diff, tolerance, ok ? "PASS" : "FAIL");
        if (!ok) failures++;

        cudaCheck(cudaFree(dA));
        cudaCheck(cudaFree(dW));
        cudaCheck(cudaFree(dBias));
        cudaCheck(cudaFree(dPre_tk));
        cudaCheck(cudaFree(dGelu_tk));
        cudaCheck(cudaFree(dPre_ref));
        cudaCheck(cudaFree(dGelu_ref));
        }
    }

    {
        total_tests++;
        Shape s = {1024, 1024, 1024, "dInp backward A*B (default)"};
        printf("\n──── %s  M=%d OC=%d C=%d ────\n", s.name, s.M, s.N, s.K);

        size_t dout_bytes = (size_t)s.M * s.N * sizeof(__nv_bfloat16);
        size_t w_bytes = (size_t)s.N * s.K * sizeof(__nv_bfloat16);
        size_t dinp_bytes = (size_t)s.M * s.K * sizeof(__nv_bfloat16);

        std::vector<__nv_bfloat16> hDout((size_t)s.M * s.N);
        std::vector<__nv_bfloat16> hW((size_t)s.N * s.K);
        std::vector<__nv_bfloat16> hDinp_tk((size_t)s.M * s.K);
        std::vector<__nv_bfloat16> hDinp_ref((size_t)s.M * s.K);

        fill_random_bf16(hDout, 50617, -0.75f, 0.75f);
        fill_random_bf16(hW, 50618, -0.75f, 0.75f);

        __nv_bfloat16 *dDout = nullptr, *dW = nullptr, *dDinp_tk = nullptr, *dDinp_ref = nullptr;
        cudaCheck(cudaMalloc(&dDout, dout_bytes));
        cudaCheck(cudaMalloc(&dW, w_bytes));
        cudaCheck(cudaMalloc(&dDinp_tk, dinp_bytes));
        cudaCheck(cudaMalloc(&dDinp_ref, dinp_bytes));

        cudaCheck(cudaMemcpy(dDout, hDout.data(), dout_bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dW, hW.data(), w_bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemset(dDinp_tk, 0, dinp_bytes));
        cudaCheck(cudaMemset(dDinp_ref, 0, dinp_bytes));

        matmul_backward(dDinp_tk, /*dweight=*/nullptr, /*dbias=*/nullptr,
                        dDout, /*inp=*/nullptr, dW, /*dbias_buffer=*/nullptr,
                        /*B=*/1, /*T=*/s.M, /*C=*/s.K, /*OC=*/s.N, /*stream=*/0,
                        /*dweight_accumulate=*/false);
        cudaCheck(cudaDeviceSynchronize());

        dim3 block(16, 16);
        dim3 grid(CEIL_DIV(s.K, 16), CEIL_DIV(s.M, 16));
        naive_dinp_ref<<<grid, block>>>(dDout, dW, dDinp_ref, s.M, s.K, s.N);
        cudaCheck(cudaDeviceSynchronize());

        cudaCheck(cudaMemcpy(hDinp_tk.data(), dDinp_tk, dinp_bytes, cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(hDinp_ref.data(), dDinp_ref, dinp_bytes, cudaMemcpyDeviceToHost));

        double max_diff = max_abs_diff(hDinp_tk, hDinp_ref);
        double tolerance = 0.5;
        bool ok = max_diff < tolerance;
        printf("  max abs diff = %.4f  (tolerance %.2f)  %s\n",
               max_diff, tolerance, ok ? "PASS" : "FAIL");
        if (!ok) failures++;

        cudaCheck(cudaFree(dDout));
        cudaCheck(cudaFree(dW));
        cudaCheck(cudaFree(dDinp_tk));
        cudaCheck(cudaFree(dDinp_ref));
    }

    {
        total_tests++;
        Shape s = {1024, 768, 768, "dInp backward A*B (SM120 small-K direct B-col)"};
        printf("\n──── %s  M=%d OC=%d C=%d ────\n", s.name, s.M, s.N, s.K);

        size_t dout_bytes = (size_t)s.M * s.N * sizeof(__nv_bfloat16);
        size_t w_bytes = (size_t)s.N * s.K * sizeof(__nv_bfloat16);
        size_t dinp_bytes = (size_t)s.M * s.K * sizeof(__nv_bfloat16);

        std::vector<__nv_bfloat16> hDout((size_t)s.M * s.N);
        std::vector<__nv_bfloat16> hW((size_t)s.N * s.K);
        std::vector<__nv_bfloat16> hDinp_tk((size_t)s.M * s.K);
        std::vector<__nv_bfloat16> hDinp_ref((size_t)s.M * s.K);

        fill_random_bf16(hDout, 51617, -0.75f, 0.75f);
        fill_random_bf16(hW, 51618, -0.75f, 0.75f);

        __nv_bfloat16 *dDout = nullptr, *dW = nullptr, *dDinp_tk = nullptr, *dDinp_ref = nullptr;
        cudaCheck(cudaMalloc(&dDout, dout_bytes));
        cudaCheck(cudaMalloc(&dW, w_bytes));
        cudaCheck(cudaMalloc(&dDinp_tk, dinp_bytes));
        cudaCheck(cudaMalloc(&dDinp_ref, dinp_bytes));

        cudaCheck(cudaMemcpy(dDout, hDout.data(), dout_bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dW, hW.data(), w_bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemset(dDinp_tk, 0, dinp_bytes));
        cudaCheck(cudaMemset(dDinp_ref, 0, dinp_bytes));

        matmul_backward(dDinp_tk, /*dweight=*/nullptr, /*dbias=*/nullptr,
                        dDout, /*inp=*/nullptr, dW, /*dbias_buffer=*/nullptr,
                        /*B=*/1, /*T=*/s.M, /*C=*/s.K, /*OC=*/s.N, /*stream=*/0,
                        /*dweight_accumulate=*/false);
        cudaCheck(cudaDeviceSynchronize());

        dim3 block(16, 16);
        dim3 grid(CEIL_DIV(s.K, 16), CEIL_DIV(s.M, 16));
        naive_dinp_ref<<<grid, block>>>(dDout, dW, dDinp_ref, s.M, s.K, s.N);
        cudaCheck(cudaDeviceSynchronize());

        cudaCheck(cudaMemcpy(hDinp_tk.data(), dDinp_tk, dinp_bytes, cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(hDinp_ref.data(), dDinp_ref, dinp_bytes, cudaMemcpyDeviceToHost));

        double max_diff = max_abs_diff(hDinp_tk, hDinp_ref);
        double tolerance = 0.5;
        bool ok = max_diff < tolerance;
        printf("  max abs diff = %.4f  (tolerance %.2f)  %s\n",
               max_diff, tolerance, ok ? "PASS" : "FAIL");
        if (!ok) failures++;

        cudaCheck(cudaFree(dDout));
        cudaCheck(cudaFree(dW));
        cudaCheck(cudaFree(dDinp_tk));
        cudaCheck(cudaFree(dDinp_ref));
    }

    {
        Shape s = {1024, 1024, 1024, "dInp backward fused dGELU"};
        printf("\n──── %s  M=%d OC=%d C=%d ────\n", s.name, s.M, s.N, s.K);
        if (!matmul_backward_gelu_fusion_supported()) {
            printf("  SKIP: fused dGELU backward is not supported by this build\n");
        } else {
        total_tests++;

        size_t dout_bytes = (size_t)s.M * s.N * sizeof(__nv_bfloat16);
        size_t w_bytes = (size_t)s.N * s.K * sizeof(__nv_bfloat16);
        size_t dinp_bytes = (size_t)s.M * s.K * sizeof(__nv_bfloat16);

        std::vector<__nv_bfloat16> hDout((size_t)s.M * s.N);
        std::vector<__nv_bfloat16> hW((size_t)s.N * s.K);
        std::vector<__nv_bfloat16> hPre((size_t)s.M * s.K);
        std::vector<__nv_bfloat16> hDinp_tk((size_t)s.M * s.K);
        std::vector<__nv_bfloat16> hDinp_ref((size_t)s.M * s.K);

        fill_random_bf16(hDout, 60617, -0.75f, 0.75f);
        fill_random_bf16(hW, 60618, -0.75f, 0.75f);
        fill_random_bf16(hPre, 60619, -1.5f, 1.5f);

        __nv_bfloat16 *dDout = nullptr, *dW = nullptr, *dPre = nullptr, *dDinp_tk = nullptr, *dDinp_ref = nullptr;
        cudaCheck(cudaMalloc(&dDout, dout_bytes));
        cudaCheck(cudaMalloc(&dW, w_bytes));
        cudaCheck(cudaMalloc(&dPre, dinp_bytes));
        cudaCheck(cudaMalloc(&dDinp_tk, dinp_bytes));
        cudaCheck(cudaMalloc(&dDinp_ref, dinp_bytes));

        cudaCheck(cudaMemcpy(dDout, hDout.data(), dout_bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dW, hW.data(), w_bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dPre, hPre.data(), dinp_bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemset(dDinp_tk, 0, dinp_bytes));
        cudaCheck(cudaMemset(dDinp_ref, 0, dinp_bytes));

        matmul_backward(dDinp_tk, /*dweight=*/nullptr, /*dbias=*/nullptr,
                        dDout, /*inp=*/nullptr, dW, /*dbias_buffer=*/nullptr,
                        /*B=*/1, /*T=*/s.M, /*C=*/s.K, /*OC=*/s.N, /*stream=*/0,
                        /*dweight_accumulate=*/false,
                        /*dweight_accum_scratch=*/nullptr,
                        /*dweight_accum_scratch_elements=*/0,
                        dPre, /*fuse_backward_gelu=*/true);
        cudaCheck(cudaDeviceSynchronize());

        dim3 block(16, 16);
        dim3 grid(CEIL_DIV(s.K, 16), CEIL_DIV(s.M, 16));
        naive_dinp_dgelu_ref<<<grid, block>>>(dDout, dW, dPre, dDinp_ref, s.M, s.K, s.N);
        cudaCheck(cudaDeviceSynchronize());

        cudaCheck(cudaMemcpy(hDinp_tk.data(), dDinp_tk, dinp_bytes, cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(hDinp_ref.data(), dDinp_ref, dinp_bytes, cudaMemcpyDeviceToHost));

        double max_diff = max_abs_diff(hDinp_tk, hDinp_ref);
        double tolerance = 0.5;
        bool ok = max_diff < tolerance;
        printf("  max abs diff = %.4f  (tolerance %.2f)  %s\n",
               max_diff, tolerance, ok ? "PASS" : "FAIL");
        if (!ok) failures++;

        cudaCheck(cudaFree(dDout));
        cudaCheck(cudaFree(dW));
        cudaCheck(cudaFree(dPre));
        cudaCheck(cudaFree(dDinp_tk));
        cudaCheck(cudaFree(dDinp_ref));
        }
    }

    {
        total_tests++;
        Shape s = {1024, 1024, 1024, "dWeight backward A^T*B (default)"};
        printf("\n──── %s  M=%d OC=%d C=%d ────\n", s.name, s.M, s.N, s.K);

        size_t dout_bytes = (size_t)s.M * s.N * sizeof(__nv_bfloat16);
        size_t inp_bytes = (size_t)s.M * s.K * sizeof(__nv_bfloat16);
        size_t dw_bytes = (size_t)s.N * s.K * sizeof(__nv_bfloat16);

        std::vector<__nv_bfloat16> hDout((size_t)s.M * s.N);
        std::vector<__nv_bfloat16> hInp((size_t)s.M * s.K);
        std::vector<__nv_bfloat16> hDW_tk((size_t)s.N * s.K);
        std::vector<__nv_bfloat16> hDW_ref((size_t)s.N * s.K);

        fill_random_bf16(hDout, 4242, -1.0f, 1.0f);
        fill_random_bf16(hInp, 9001, -1.0f, 1.0f);

        __nv_bfloat16 *dDout = nullptr, *dInp = nullptr, *dDW_tk = nullptr, *dDW_ref = nullptr, *dScratch = nullptr;
        cudaCheck(cudaMalloc(&dDout, dout_bytes));
        cudaCheck(cudaMalloc(&dInp, inp_bytes));
        cudaCheck(cudaMalloc(&dDW_tk, dw_bytes));
        cudaCheck(cudaMalloc(&dDW_ref, dw_bytes));
        cudaCheck(cudaMalloc(&dScratch, 8 * dw_bytes));

        cudaCheck(cudaMemcpy(dDout, hDout.data(), dout_bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dInp, hInp.data(), inp_bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemset(dDW_tk, 0, dw_bytes));
        cudaCheck(cudaMemset(dDW_ref, 0, dw_bytes));
        cudaCheck(cudaMemset(dScratch, 0, 8 * dw_bytes));

        matmul_backward(/*dinp=*/nullptr, dDW_tk, /*dbias=*/nullptr,
                        dDout, dInp, /*weight=*/nullptr, /*dbias_buffer=*/nullptr,
                        /*B=*/1, /*T=*/s.M, /*C=*/s.K, /*OC=*/s.N, /*stream=*/0,
                        /*dweight_accumulate=*/false,
                        /*dweight_accum_scratch=*/dScratch,
                        /*dweight_accum_scratch_elements=*/8 * (size_t)s.N * s.K);
        cudaCheck(cudaDeviceSynchronize());

        dim3 block(16, 16);
        dim3 grid(CEIL_DIV(s.N, 16), CEIL_DIV(s.K, 16));
        naive_dweight_ref<<<grid, block>>>(dDout, dInp, dDW_ref, s.M, s.K, s.N);
        cudaCheck(cudaDeviceSynchronize());

        cudaCheck(cudaMemcpy(hDW_tk.data(), dDW_tk, dw_bytes, cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(hDW_ref.data(), dDW_ref, dw_bytes, cudaMemcpyDeviceToHost));

        double max_diff = max_abs_diff(hDW_tk, hDW_ref);
        double tolerance = 0.5;
        bool ok = max_diff < tolerance;
        printf("  max abs diff = %.4f  (tolerance %.2f)  %s\n",
               max_diff, tolerance, ok ? "PASS" : "FAIL");
        if (!ok) failures++;

        cudaCheck(cudaFree(dDout));
        cudaCheck(cudaFree(dInp));
        cudaCheck(cudaFree(dDW_tk));
        cudaCheck(cudaFree(dDW_ref));
        cudaCheck(cudaFree(dScratch));
    }

    {
        total_tests++;
        Shape s = {1024, 1024, 1024, "accumulated dWeight backward A^T*B + add"};
        printf("\n──── %s  M=%d OC=%d C=%d ────\n", s.name, s.M, s.N, s.K);

        size_t dout_bytes = (size_t)s.M * s.N * sizeof(__nv_bfloat16);
        size_t inp_bytes = (size_t)s.M * s.K * sizeof(__nv_bfloat16);
        size_t dw_bytes = (size_t)s.N * s.K * sizeof(__nv_bfloat16);

        std::vector<__nv_bfloat16> hDout((size_t)s.M * s.N);
        std::vector<__nv_bfloat16> hInp((size_t)s.M * s.K);
        std::vector<__nv_bfloat16> hInitial((size_t)s.N * s.K);
        std::vector<__nv_bfloat16> hDW_tk((size_t)s.N * s.K);
        std::vector<__nv_bfloat16> hDW_ref((size_t)s.N * s.K);

        fill_random_bf16(hDout, 5150, -1.0f, 1.0f);
        fill_random_bf16(hInp, 6060, -1.0f, 1.0f);
        fill_random_bf16(hInitial, 7070, -0.25f, 0.25f);

        __nv_bfloat16 *dDout = nullptr, *dInp = nullptr, *dDW_tk = nullptr, *dDW_ref = nullptr, *dScratch = nullptr;
        cudaCheck(cudaMalloc(&dDout, dout_bytes));
        cudaCheck(cudaMalloc(&dInp, inp_bytes));
        cudaCheck(cudaMalloc(&dDW_tk, dw_bytes));
        cudaCheck(cudaMalloc(&dDW_ref, dw_bytes));
        cudaCheck(cudaMalloc(&dScratch, 8 * dw_bytes));

        cudaCheck(cudaMemcpy(dDout, hDout.data(), dout_bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dInp, hInp.data(), inp_bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dDW_tk, hInitial.data(), dw_bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dDW_ref, hInitial.data(), dw_bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemset(dScratch, 0, 8 * dw_bytes));

        matmul_backward(/*dinp=*/nullptr, dDW_tk, /*dbias=*/nullptr,
                        dDout, dInp, /*weight=*/nullptr, /*dbias_buffer=*/nullptr,
                        /*B=*/1, /*T=*/s.M, /*C=*/s.K, /*OC=*/s.N, /*stream=*/0,
                        /*dweight_accumulate=*/true,
                        /*dweight_accum_scratch=*/dScratch,
                        /*dweight_accum_scratch_elements=*/8 * (size_t)s.N * s.K);
        cudaCheck(cudaDeviceSynchronize());

        dim3 block(16, 16);
        dim3 grid(CEIL_DIV(s.N, 16), CEIL_DIV(s.K, 16));
        naive_dweight_accum_ref<<<grid, block>>>(dDout, dInp, dDW_ref, s.M, s.K, s.N);
        cudaCheck(cudaDeviceSynchronize());

        cudaCheck(cudaMemcpy(hDW_tk.data(), dDW_tk, dw_bytes, cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(hDW_ref.data(), dDW_ref, dw_bytes, cudaMemcpyDeviceToHost));

        double max_diff = max_abs_diff(hDW_tk, hDW_ref);
        double tolerance = 0.5;
        bool ok = max_diff < tolerance;
        printf("  max abs diff = %.4f  (tolerance %.2f)  %s\n",
               max_diff, tolerance, ok ? "PASS" : "FAIL");
        if (!ok) failures++;

        cudaCheck(cudaFree(dDout));
        cudaCheck(cudaFree(dInp));
        cudaCheck(cudaFree(dDW_tk));
        cudaCheck(cudaFree(dDW_ref));
        cudaCheck(cudaFree(dScratch));
    }

    printf("\n──── %d/%d passed ────\n", total_tests - failures, total_tests);
    if (failures == 0) {
        printf("test_matmul smoke OK\n");
    }
#if LLMK_SM120_HAS_CUBLASLT_GEMM
    llmk::cublaslt_sm120::destroy();
#endif
    return failures == 0 ? 0 : 1;
}
