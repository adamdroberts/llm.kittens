// probe_attention_gqa.cu — runs llm.kittens's attention_gqa_forward (Llama-3
// GQA + RoPE + causal MHA). Family B: no llm.c counterpart; reference is in
// torch on the Python side.
// CLI: probe_attention_gqa <io_dir>
//   reads:  inp.npy (uint16 bf16, [B*T*(C+2*C_kv)] packed Q,K,V)
//           cos.npy, sin.npy (uint16 bf16, [T*HS/2])
//           shape.npy (int32 [B, T, C, NH, NKVH])
//   writes: tk/out.npy (uint16 bf16, [B*T*C])
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

cudaDeviceProp deviceProp;

#include "llmc/attention_gqa.cuh"

#include "npy/npy.h"

static std::string join(const std::string& a, const std::string& b) {
    return (!a.empty() && a.back() == '/') ? a + b : a + "/" + b;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <io_dir>\n", argv[0]); return 2; }
    std::string io = argv[1];
    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));

    auto h_inp = npy::load(join(io, "inp.npy"));
    auto h_cos = npy::load(join(io, "cos.npy"));
    auto h_sin = npy::load(join(io, "sin.npy"));
    auto h_sh  = npy::load(join(io, "shape.npy"));
    int32_t* sh = reinterpret_cast<int32_t*>(h_sh.data.data());
    int B = sh[0], T = sh[1], C = sh[2], NH = sh[3], NKVH = sh[4];
    int HS = C / NH;
    int C_kv = NKVH * HS;

    size_t inp_n = (size_t)B * T * (C + 2 * C_kv);
    size_t qkvr_n = (size_t)B * T * (C + 2 * C_kv);  // permuted Q + K + V
    size_t out_n = (size_t)B * T * C;
    size_t lse_n = (size_t)B * NH * T;
    size_t cs_n = (size_t)T * (HS / 2);

    __nv_bfloat16 *d_inp=nullptr, *d_cos=nullptr, *d_sin=nullptr;
    __nv_bfloat16 *d_out=nullptr, *d_qkvr=nullptr;
    float* d_lse=nullptr;
    cudaCheck(cudaMalloc(&d_inp,  inp_n  * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_cos,  cs_n   * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_sin,  cs_n   * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_out,  out_n  * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_qkvr, qkvr_n * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_lse,  lse_n  * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, h_inp.data.data(), inp_n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_cos, h_cos.data.data(), cs_n  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_sin, h_sin.data.data(), cs_n  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_out,  0, out_n  * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMemset(d_qkvr, 0, qkvr_n * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMemset(d_lse,  0, lse_n  * sizeof(float)));

    attention_gqa_forward(d_out, d_qkvr, d_lse, d_inp, d_cos, d_sin,
                          B, T, C, NH, NKVH, /*stream=*/0);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<uint16_t> h_out(out_n);
    cudaCheck(cudaMemcpy(h_out.data(), d_out, out_n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    std::string out_dir = join(io, "tk");
    if (system(("mkdir -p " + out_dir).c_str()) != 0) return 1;
    npy::save(join(out_dir, "out.npy"), npy::DType::U2, {out_n}, h_out.data());

    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_cos));
    cudaCheck(cudaFree(d_sin));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_qkvr));
    cudaCheck(cudaFree(d_lse));
    printf("probe_attention_gqa OK\n");
    return 0;
}
