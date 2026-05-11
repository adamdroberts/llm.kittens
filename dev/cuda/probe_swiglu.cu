// probe_swiglu.cu — runs llm.kittens's SwiGLU forward + backward.
// Family B: no llm.c counterpart, so the parity reference is computed in
// PyTorch on the Python side. CLI: probe_swiglu <io_dir>
//   reads:  gate.npy, up.npy, dout.npy (uint16 bf16-bits), shape.npy (int32 [N])
//   writes: tk/{out, dgate, dup}.npy
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

cudaDeviceProp deviceProp;

#include "llmc/swiglu.cuh"

#include "npy/npy.h"

static std::string join(const std::string& a, const std::string& b) {
    return (!a.empty() && a.back() == '/') ? a + b : a + "/" + b;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <io_dir>\n", argv[0]); return 2; }
    std::string io = argv[1];
    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));

    auto h_gate = npy::load(join(io, "gate.npy"));
    auto h_up   = npy::load(join(io, "up.npy"));
    auto h_dout = npy::load(join(io, "dout.npy"));
    auto h_sh   = npy::load(join(io, "shape.npy"));
    int N = reinterpret_cast<int32_t*>(h_sh.data.data())[0];

    __nv_bfloat16 *d_gate=nullptr, *d_up=nullptr, *d_dout=nullptr;
    __nv_bfloat16 *d_out=nullptr, *d_dgate=nullptr, *d_dup=nullptr;
    cudaCheck(cudaMalloc(&d_gate, N * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_up, N * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_dout, N * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_out, N * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_dgate, N * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_dup, N * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMemcpy(d_gate, h_gate.data.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_up,   h_up.data.data(),   N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, h_dout.data.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    swiglu_forward(d_out, d_gate, d_up, N, 0);
    swiglu_backward(d_dgate, d_dup, d_dout, d_gate, d_up, N, 0);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<uint16_t> h_out(N), h_dgate(N), h_dup(N);
    cudaCheck(cudaMemcpy(h_out.data(),   d_out,   N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_dgate.data(), d_dgate, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_dup.data(),   d_dup,   N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    std::string out_dir = join(io, "tk");
    if (system(("mkdir -p " + out_dir).c_str()) != 0) return 1;
    npy::save(join(out_dir, "out.npy"),   npy::DType::U2, {(size_t)N}, h_out.data());
    npy::save(join(out_dir, "dgate.npy"), npy::DType::U2, {(size_t)N}, h_dgate.data());
    npy::save(join(out_dir, "dup.npy"),   npy::DType::U2, {(size_t)N}, h_dup.data());

    cudaCheck(cudaFree(d_gate));
    cudaCheck(cudaFree(d_up));
    cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_dgate));
    cudaCheck(cudaFree(d_dup));
    printf("probe_swiglu OK\n");
    return 0;
}
