// probe_layernorm_ref.cu — runs llm.c's layernorm_forward on bf16 inputs and
// dumps outputs. Paired with probe_layernorm_tk.cu (llm.kittens version) for a
// side-by-side parity test driven by tests/parity/test_parity_layernorm.py.
//
// CLI: probe_layernorm_ref <io_dir>
//   reads:  <io_dir>/{inp,weight,bias}.npy   (bf16 bits as uint16)
//           <io_dir>/shape.npy               (int32: [B, T, C])
//   writes: <io_dir>/ref/{out,mean,rstd}.npy
//
// Build: `make probe_layernorm_ref` (uses LLMC_REF_ROOT=../llm.c).
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

cudaDeviceProp deviceProp;

// Use llm.c's headers — they have the same #include guards as llm.kittens, but
// this TU only sees one set.
#include "llmc/layernorm.cuh"

#include "npy/npy.h"

static std::string join(const std::string& a, const std::string& b) {
    if (!a.empty() && a.back() == '/') return a + b;
    return a + "/" + b;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <io_dir>\n", argv[0]); return 2; }
    std::string io = argv[1];

    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));

    auto h_inp = npy::load(join(io, "inp.npy"));
    auto h_w   = npy::load(join(io, "weight.npy"));
    auto h_b   = npy::load(join(io, "bias.npy"));
    auto h_sh  = npy::load(join(io, "shape.npy"));
    if (h_sh.dtype != npy::DType::I4 || h_sh.numel() != 3)
        throw std::runtime_error("shape.npy must be int32[3]");
    int32_t* sh = reinterpret_cast<int32_t*>(h_sh.data.data());
    int B = sh[0], T = sh[1], C = sh[2];
    size_t N = (size_t)B * T;
    if (h_inp.numel() != N * (size_t)C) throw std::runtime_error("inp shape mismatch");
    if (h_w.numel() != (size_t)C) throw std::runtime_error("weight shape mismatch");
    if (h_b.numel() != (size_t)C) throw std::runtime_error("bias shape mismatch");

    __nv_bfloat16 *d_inp=nullptr, *d_w=nullptr, *d_b=nullptr, *d_out=nullptr;
    float *d_mean=nullptr, *d_rstd=nullptr;
    cudaCheck(cudaMalloc(&d_inp, N * C * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_w,   C * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_b,   C * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_out, N * C * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_mean, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_rstd, N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, h_inp.data.data(), N * C * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_w,   h_w.data.data(),   C * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b,   h_b.data.data(),   C * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    layernorm_forward(d_out, d_mean, d_rstd, d_inp, d_w, d_b, B, T, C, /*stream=*/0);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<uint16_t> h_out(N * C);
    std::vector<float> h_mean(N), h_rstd(N);
    cudaCheck(cudaMemcpy(h_out.data(),  d_out,  N * C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_mean.data(), d_mean, N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_rstd.data(), d_rstd, N * sizeof(float), cudaMemcpyDeviceToHost));

    std::string out_dir = join(io, "ref");
    // The Python driver creates the dir before calling us, but be defensive.
    std::string mkdir_cmd = "mkdir -p " + out_dir;
    if (system(mkdir_cmd.c_str()) != 0) { fprintf(stderr, "mkdir failed\n"); return 1; }
    npy::save(join(out_dir, "out.npy"),  npy::DType::U2, {N, (size_t)C}, h_out.data());
    npy::save(join(out_dir, "mean.npy"), npy::DType::F4, {N}, h_mean.data());
    npy::save(join(out_dir, "rstd.npy"), npy::DType::F4, {N}, h_rstd.data());

    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_w));
    cudaCheck(cudaFree(d_b));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_mean));
    cudaCheck(cudaFree(d_rstd));

    printf("probe_layernorm_ref OK\n");
    return 0;
}
