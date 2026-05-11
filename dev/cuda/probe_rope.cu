// probe_rope.cu — runs llm.kittens's rope_forward + rope_backward. Family B:
// no llm.c counterpart; reference is computed in PyTorch on the Python side.
// CLI: probe_rope <io_dir>
//   reads:  x.npy (uint16 bf16, [B*H*T*HS]), cos.npy, sin.npy (uint16 bf16, [T*HS/2]),
//           dout.npy (uint16 bf16, [B*H*T*HS]),
//           shape.npy (int32 [B, H, T, HS])
//   writes: tk/{out, dx}.npy
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

cudaDeviceProp deviceProp;

#include "llmc/rope.cuh"

#include "npy/npy.h"

static std::string join(const std::string& a, const std::string& b) {
    return (!a.empty() && a.back() == '/') ? a + b : a + "/" + b;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <io_dir>\n", argv[0]); return 2; }
    std::string io = argv[1];
    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));

    auto h_x    = npy::load(join(io, "x.npy"));
    auto h_cos  = npy::load(join(io, "cos.npy"));
    auto h_sin  = npy::load(join(io, "sin.npy"));
    auto h_dout = npy::load(join(io, "dout.npy"));
    auto h_sh   = npy::load(join(io, "shape.npy"));
    int32_t* sh = reinterpret_cast<int32_t*>(h_sh.data.data());
    int B = sh[0], H = sh[1], T = sh[2], HS = sh[3];

    size_t total = (size_t)B * H * T * HS;
    size_t cs_n  = (size_t)T * (HS / 2);

    __nv_bfloat16 *d_x=nullptr, *d_cos=nullptr, *d_sin=nullptr;
    __nv_bfloat16 *d_out=nullptr, *d_dout=nullptr, *d_dx=nullptr;
    cudaCheck(cudaMalloc(&d_x,    total * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_cos,  cs_n  * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_sin,  cs_n  * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_out,  total * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_dout, total * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_dx,   total * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMemcpy(d_x,    h_x.data.data(),    total * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_cos,  h_cos.data.data(),  cs_n  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_sin,  h_sin.data.data(),  cs_n  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, h_dout.data.data(), total * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    rope_forward(d_out, d_x, d_cos, d_sin, B, H, T, HS, 0);
    rope_backward(d_dx, d_dout, d_cos, d_sin, B, H, T, HS, 0);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<uint16_t> h_out(total), h_dx(total);
    cudaCheck(cudaMemcpy(h_out.data(), d_out, total * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_dx.data(),  d_dx,  total * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    std::string out_dir = join(io, "tk");
    if (system(("mkdir -p " + out_dir).c_str()) != 0) return 1;
    npy::save(join(out_dir, "out.npy"), npy::DType::U2, {total}, h_out.data());
    npy::save(join(out_dir, "dx.npy"),  npy::DType::U2, {total}, h_dx.data());

    cudaCheck(cudaFree(d_x));
    cudaCheck(cudaFree(d_cos));
    cudaCheck(cudaFree(d_sin));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_dx));
    printf("probe_rope OK\n");
    return 0;
}
