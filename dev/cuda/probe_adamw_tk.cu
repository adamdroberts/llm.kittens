// probe_adamw_tk.cu — runs llm.kittens's adamw_update for one step.
// See probe_adamw_ref.cu for the protocol.
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

cudaDeviceProp deviceProp;

#include "llmc/adamw.cuh"

#include "npy/npy.h"

static std::string join(const std::string& a, const std::string& b) {
    return (!a.empty() && a.back() == '/') ? a + b : a + "/" + b;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <io_dir>\n", argv[0]); return 2; }
    std::string io = argv[1];
    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));

    auto h_param  = npy::load(join(io, "param.npy"));
    auto h_master = npy::load(join(io, "master.npy"));
    auto h_grad   = npy::load(join(io, "grad.npy"));
    auto h_m      = npy::load(join(io, "m.npy"));
    auto h_v      = npy::load(join(io, "v.npy"));
    auto h_sh     = npy::load(join(io, "shape.npy"));
    size_t N = (size_t)reinterpret_cast<int32_t*>(h_sh.data.data())[0];

    __nv_bfloat16 *d_param=nullptr, *d_grad=nullptr;
    float *d_master=nullptr, *d_m=nullptr, *d_v=nullptr;
    cudaCheck(cudaMalloc(&d_param,  N * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_grad,   N * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_master, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_m,      N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_v,      N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_param,  h_param.data.data(),  N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_grad,   h_grad.data.data(),   N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_master, h_master.data.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_m,      h_m.data.data(),      N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_v,      h_v.data.data(),      N * sizeof(float), cudaMemcpyHostToDevice));

    constexpr float lr = 1e-3f, beta1 = 0.9f, beta2 = 0.95f;
    constexpr float eps = 1e-8f, wd = 0.1f, grad_scale = 1.0f;
    constexpr int t = 1;
    constexpr unsigned int seed = 0xCAFEBABE;
    adamw_update(d_param, d_master, d_grad, d_m, d_v, N,
                 0, 0, 0, 1, lr, beta1, beta2, t, eps, wd, grad_scale, seed, 0);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<uint16_t> h_param_out(N);
    std::vector<float> h_master_out(N), h_m_out(N), h_v_out(N);
    cudaCheck(cudaMemcpy(h_param_out.data(),  d_param,  N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_master_out.data(), d_master, N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_m_out.data(),      d_m,      N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_v_out.data(),      d_v,      N * sizeof(float), cudaMemcpyDeviceToHost));

    std::string out_dir = join(io, "tk");
    if (system(("mkdir -p " + out_dir).c_str()) != 0) return 1;
    npy::save(join(out_dir, "param.npy"),  npy::DType::U2, {N}, h_param_out.data());
    npy::save(join(out_dir, "master.npy"), npy::DType::F4, {N}, h_master_out.data());
    npy::save(join(out_dir, "m.npy"),      npy::DType::F4, {N}, h_m_out.data());
    npy::save(join(out_dir, "v.npy"),      npy::DType::F4, {N}, h_v_out.data());

    cudaCheck(cudaFree(d_param));
    cudaCheck(cudaFree(d_grad));
    cudaCheck(cudaFree(d_master));
    cudaCheck(cudaFree(d_m));
    cudaCheck(cudaFree(d_v));
    printf("probe_adamw_tk OK\n");
    return 0;
}
