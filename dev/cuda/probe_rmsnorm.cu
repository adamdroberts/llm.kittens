// probe_rmsnorm.cu — runs llm.kittens's rmsnorm_forward. Family B: no llm.c
// counterpart; reference is computed in PyTorch on the Python side.
// CLI: probe_rmsnorm <io_dir>
//   reads:  inp.npy (uint16 bf16, [N*C]), weight.npy (uint16 bf16, [C]),
//           shape.npy (int32 [N, C])
//   writes: tk/{out, rstd}.npy
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

cudaDeviceProp deviceProp;

#include "llmc/rmsnorm.cuh"

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
    auto h_w   = npy::load(join(io, "weight.npy"));
    auto h_sh  = npy::load(join(io, "shape.npy"));
    int32_t* sh = reinterpret_cast<int32_t*>(h_sh.data.data());
    int N = sh[0], C = sh[1];
    constexpr float eps = 1e-5f;

    __nv_bfloat16 *d_inp=nullptr, *d_w=nullptr, *d_out=nullptr;
    float* d_rstd=nullptr;
    cudaCheck(cudaMalloc(&d_inp,  (size_t)N * C * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_w,    (size_t)C *     sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_out,  (size_t)N * C * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_rstd, (size_t)N *     sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, h_inp.data.data(), (size_t)N * C * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_w,   h_w.data.data(),   (size_t)C *     sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    rmsnorm_forward(d_out, d_rstd, d_inp, d_w, N, C, eps, 0);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<uint16_t> h_out((size_t)N * C);
    std::vector<float> h_rstd(N);
    cudaCheck(cudaMemcpy(h_out.data(),  d_out,  (size_t)N * C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_rstd.data(), d_rstd, (size_t)N *     sizeof(float), cudaMemcpyDeviceToHost));

    std::string out_dir = join(io, "tk");
    if (system(("mkdir -p " + out_dir).c_str()) != 0) return 1;
    npy::save(join(out_dir, "out.npy"),  npy::DType::U2, {(size_t)N, (size_t)C}, h_out.data());
    npy::save(join(out_dir, "rstd.npy"), npy::DType::F4, {(size_t)N}, h_rstd.data());

    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_w));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_rstd));
    printf("probe_rmsnorm OK\n");
    return 0;
}
