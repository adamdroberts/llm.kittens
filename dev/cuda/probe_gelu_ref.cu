// probe_gelu_ref.cu — runs llm.c's gelu_forward / gelu_backward_inplace.
// Paired with probe_gelu_tk.cu. CLI: probe_gelu_ref <io_dir>.
//   reads:  inp.npy, dout.npy (uint16 bf16-bits), shape.npy (int32 [N])
//   writes: ref/{out, dinp}.npy
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

cudaDeviceProp deviceProp;

#include "llmc/gelu.cuh"

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
    auto h_dout = npy::load(join(io, "dout.npy"));
    auto h_sh = npy::load(join(io, "shape.npy"));
    int N = reinterpret_cast<int32_t*>(h_sh.data.data())[0];

    __nv_bfloat16 *d_inp=nullptr, *d_out=nullptr, *d_dinout=nullptr;
    cudaCheck(cudaMalloc(&d_inp, N * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_out, N * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_dinout, N * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMemcpy(d_inp, h_inp.data.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dinout, h_dout.data.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    gelu_forward(d_out, d_inp, N, 0);
    gelu_backward_inplace(d_dinout, d_inp, N, 0);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<uint16_t> h_out(N), h_dinp(N);
    cudaCheck(cudaMemcpy(h_out.data(), d_out, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_dinp.data(), d_dinout, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    std::string out_dir = join(io, "ref");
    if (system(("mkdir -p " + out_dir).c_str()) != 0) return 1;
    npy::save(join(out_dir, "out.npy"),  npy::DType::U2, {(size_t)N}, h_out.data());
    npy::save(join(out_dir, "dinp.npy"), npy::DType::U2, {(size_t)N}, h_dinp.data());

    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_dinout));
    printf("probe_gelu_ref OK\n");
    return 0;
}
