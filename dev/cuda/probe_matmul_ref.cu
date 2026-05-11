// probe_matmul_ref.cu — runs llm.c's matmul_forward_cublaslt (cuBLASLt baseline).
// CLI: probe_matmul_ref <io_dir>
//   reads:  inp.npy (uint16 bf16, [B*T*C]), weight.npy (uint16 bf16, [OC*C]),
//           bias.npy (uint16 bf16, [OC]),
//           shape.npy (int32 [B, T, C, OC])
//   writes: ref/out.npy (uint16 bf16, [B*T*OC])
//
// Initialises the cuBLASLt handle + 32 MB workspace declared in
// llm.c/llmc/cublas_common.h. Each TU including that header gets its own
// instance, so creation lives here in main() rather than as a global ctor.
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

cudaDeviceProp deviceProp;

#include "llmc/matmul.cuh"

#include "npy/npy.h"

static std::string join(const std::string& a, const std::string& b) {
    return (!a.empty() && a.back() == '/') ? a + b : a + "/" + b;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <io_dir>\n", argv[0]); return 2; }
    std::string io = argv[1];
    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));

    // Initialise the cuBLASLt globals declared in llm.c/llmc/cublas_common.h.
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    auto h_inp = npy::load(join(io, "inp.npy"));
    auto h_w   = npy::load(join(io, "weight.npy"));
    auto h_b   = npy::load(join(io, "bias.npy"));
    auto h_sh  = npy::load(join(io, "shape.npy"));
    int32_t* sh = reinterpret_cast<int32_t*>(h_sh.data.data());
    int B = sh[0], T = sh[1], C = sh[2], OC = sh[3];

    size_t inp_n = (size_t)B * T * C;
    size_t w_n   = (size_t)OC * C;
    size_t b_n   = (size_t)OC;
    size_t out_n = (size_t)B * T * OC;

    __nv_bfloat16 *d_inp=nullptr, *d_w=nullptr, *d_b=nullptr, *d_out=nullptr;
    cudaCheck(cudaMalloc(&d_inp, inp_n * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_w,   w_n   * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_b,   b_n   * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_out, out_n * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMemcpy(d_inp, h_inp.data.data(), inp_n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_w,   h_w.data.data(),   w_n   * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b,   h_b.data.data(),   b_n   * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_out, 0, out_n * sizeof(__nv_bfloat16)));

    matmul_forward_cublaslt(d_out, d_inp, d_w, d_b, B, T, C, OC, /*stream=*/0,
                            /*pre_gelu=*/nullptr, /*gelu_fusion=*/0);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<uint16_t> h_out(out_n);
    cudaCheck(cudaMemcpy(h_out.data(), d_out, out_n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    std::string out_dir = join(io, "ref");
    if (system(("mkdir -p " + out_dir).c_str()) != 0) return 1;
    npy::save(join(out_dir, "out.npy"), npy::DType::U2, {out_n}, h_out.data());

    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_w));
    cudaCheck(cudaFree(d_b));
    cudaCheck(cudaFree(d_out));
    cublasCheck(cublasLtDestroy(cublaslt_handle));
    cudaCheck(cudaFree(cublaslt_workspace));
    printf("probe_matmul_ref OK\n");
    return 0;
}
