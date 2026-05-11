// probe_encoder_ref.cu — runs llm.c's encoder_forward (forward only; backward
// involves stochastic rounding which makes per-element parity nondeterministic).
// CLI: probe_encoder_ref <io_dir>
//   reads:  inp.npy (int32 [B*T] token ids), wte.npy, wpe.npy (uint16 bf16),
//           shape.npy (int32 [B, T, C, V])
//   writes: ref/out.npy (uint16 bf16, [B*T*C])
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

cudaDeviceProp deviceProp;

#include "llmc/encoder.cuh"

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
    auto h_wte = npy::load(join(io, "wte.npy"));
    auto h_wpe = npy::load(join(io, "wpe.npy"));
    auto h_sh  = npy::load(join(io, "shape.npy"));
    int32_t* sh = reinterpret_cast<int32_t*>(h_sh.data.data());
    int B = sh[0], T = sh[1], C = sh[2], V = sh[3];

    int* d_inp=nullptr;
    __nv_bfloat16 *d_wte=nullptr, *d_wpe=nullptr, *d_out=nullptr;
    cudaCheck(cudaMalloc(&d_inp, (size_t)B * T * sizeof(int)));
    cudaCheck(cudaMalloc(&d_wte, (size_t)V * C * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_wpe, (size_t)T * C * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_out, (size_t)B * T * C * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMemcpy(d_inp, h_inp.data.data(), (size_t)B * T * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_wte, h_wte.data.data(), (size_t)V * C * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_wpe, h_wpe.data.data(), (size_t)T * C * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    encoder_forward(d_out, d_inp, d_wte, d_wpe, B, T, C, 0);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<uint16_t> h_out((size_t)B * T * C);
    cudaCheck(cudaMemcpy(h_out.data(), d_out, (size_t)B * T * C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    std::string out_dir = join(io, "ref");
    if (system(("mkdir -p " + out_dir).c_str()) != 0) return 1;
    npy::save(join(out_dir, "out.npy"), npy::DType::U2, {(size_t)B * T * C}, h_out.data());

    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_wte));
    cudaCheck(cudaFree(d_wpe));
    cudaCheck(cudaFree(d_out));
    printf("probe_encoder_ref OK\n");
    return 0;
}
