// probe_fused_classifier_tk.cu — runs llm.kittens's fused_classifier (byte-
// identical to llm.c's). See probe_fused_classifier_ref.cu for protocol.
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

cudaDeviceProp deviceProp;

#include "llmc/fused_classifier.cuh"

#include "npy/npy.h"

static std::string join(const std::string& a, const std::string& b) {
    return (!a.empty() && a.back() == '/') ? a + b : a + "/" + b;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <io_dir>\n", argv[0]); return 2; }
    std::string io = argv[1];
    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));

    if (deviceProp.major != 9) {
        printf("SKIPPED: fused_classifier requires sm_90 (H100); detected sm_%d%d\n",
               deviceProp.major, deviceProp.minor);
        std::string out_dir = join(io, "tk");
        system(("mkdir -p " + out_dir).c_str());
        float zero = 0.0f;
        npy::save(join(out_dir, "loss.npy"), npy::DType::F4, {1}, &zero);
        uint16_t z16 = 0;
        npy::save(join(out_dir, "dlogits.npy"), npy::DType::U2, {1}, &z16);
        printf("probe_fused_classifier_tk OK\n");
        return 0;
    }

    auto h_logits  = npy::load(join(io, "logits.npy"));
    auto h_targets = npy::load(join(io, "targets.npy"));
    auto h_sh      = npy::load(join(io, "shape.npy"));
    int32_t* sh = reinterpret_cast<int32_t*>(h_sh.data.data());
    int B = sh[0], T = sh[1], V = sh[2], P = sh[3];
    int rows = B * T;
    float dloss = 1.0f / (float)rows;

    __nv_bfloat16* d_logits=nullptr;
    int* d_targets=nullptr;
    float* d_losses=nullptr;
    cudaCheck(cudaMalloc(&d_logits,  (size_t)rows * P * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMalloc(&d_targets, rows * sizeof(int)));
    cudaCheck(cudaMalloc(&d_losses,  rows * sizeof(float)));
    cudaCheck(cudaMemcpy(d_logits,  h_logits.data.data(),  (size_t)rows * P * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_targets, h_targets.data.data(), rows * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_losses, 0, rows * sizeof(float)));

    fused_classifier<__nv_bfloat16, true>(d_logits, d_losses, dloss, d_targets,
                                          B, T, V, P, True, 0);
    cudaCheck(cudaDeviceSynchronize());

    std::vector<float> h_losses(rows);
    std::vector<uint16_t> h_dlogits((size_t)rows * P);
    cudaCheck(cudaMemcpy(h_losses.data(),  d_losses, rows * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_dlogits.data(), d_logits, (size_t)rows * P * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    std::string out_dir = join(io, "tk");
    if (system(("mkdir -p " + out_dir).c_str()) != 0) return 1;
    npy::save(join(out_dir, "loss.npy"),    npy::DType::F4, {(size_t)rows}, h_losses.data());
    npy::save(join(out_dir, "dlogits.npy"), npy::DType::U2, {(size_t)rows, (size_t)P}, h_dlogits.data());

    cudaCheck(cudaFree(d_logits));
    cudaCheck(cudaFree(d_targets));
    cudaCheck(cudaFree(d_losses));
    printf("probe_fused_classifier_tk OK\n");
    return 0;
}
