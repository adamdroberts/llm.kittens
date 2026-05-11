// probe_attention_ref.cu — runs llm.c's attention_forward (forward only).
// CLI: probe_attention_ref <io_dir>
//   reads:  inp.npy (uint16 bf16, [B*T*3*C]), shape.npy (int32 [B, T, C, NH])
//   writes: ref/out.npy (uint16 bf16, [B*T*C])
//
// Backward parity is deferred — the two repos' backward signatures and scratch
// layouts diverge, so a meaningful side-by-side requires more wrapping.
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

// llm.c's attention_forward internally calls matmul_cublaslt for QK^T and
// attn-V, so matmul.cuh must be visible and the cuBLASLt globals must be
// initialised before attention_forward runs.
#include "llmc/matmul.cuh"
#include "llmc/attention.cuh"

#include "npy/npy.h"

static std::string join(const std::string& a, const std::string& b) {
    return (!a.empty() && a.back() == '/') ? a + b : a + "/" + b;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <io_dir>\n", argv[0]); return 2; }
    std::string io = argv[1];
    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));

    // cuBLASLt globals from llm.c/llmc/cublas_common.h.
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    auto h_inp = npy::load(join(io, "inp.npy"));
    auto h_sh  = npy::load(join(io, "shape.npy"));
    int32_t* sh = reinterpret_cast<int32_t*>(h_sh.data.data());
    int B = sh[0], T = sh[1], C = sh[2], NH = sh[3];

    size_t inp_n = (size_t)B * T * 3 * C;
    size_t out_n = (size_t)B * T * C;
    size_t att_n = (size_t)B * NH * T * T;

    __nv_bfloat16 *d_inp=nullptr, *d_out=nullptr, *d_qkvr=nullptr, *d_att=nullptr;
    cudaCheck(cudaMalloc(&d_inp,  inp_n * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_out,  out_n * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_qkvr, inp_n * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_att,  att_n * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMemcpy(d_inp, h_inp.data.data(), inp_n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_out,  0, out_n * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMemset(d_qkvr, 0, inp_n * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMemset(d_att,  0, att_n * sizeof(__nv_bfloat16)));

    attention_forward(d_out, d_qkvr, d_att, d_inp, B, T, C, NH, 0);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<uint16_t> h_out(out_n);
    cudaCheck(cudaMemcpy(h_out.data(), d_out, out_n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    std::string out_dir = join(io, "ref");
    if (system(("mkdir -p " + out_dir).c_str()) != 0) return 1;
    npy::save(join(out_dir, "out.npy"), npy::DType::U2, {out_n}, h_out.data());

    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_qkvr));
    cudaCheck(cudaFree(d_att));
    cublasCheck(cublasLtDestroy(cublaslt_handle));
    cudaCheck(cudaFree(cublaslt_workspace));
    printf("probe_attention_ref OK\n");
    return 0;
}
