// probe_global_norm_ref.cu — runs llm.c's global_norm_squared (single shard).
// CLI: probe_global_norm_ref <io_dir>
//   reads:  vals.npy (uint16 bf16-bits), shape.npy (int32 [N])
//   writes: ref/norm.npy (float32 [1] — sqrt(sum(v^2)))
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

cudaDeviceProp deviceProp;

#include "llmc/global_norm.cuh"

#include "npy/npy.h"

static std::string join(const std::string& a, const std::string& b) {
    return (!a.empty() && a.back() == '/') ? a + b : a + "/" + b;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <io_dir>\n", argv[0]); return 2; }
    std::string io = argv[1];
    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));

    auto h_vals = npy::load(join(io, "vals.npy"));
    auto h_sh   = npy::load(join(io, "shape.npy"));
    int N = reinterpret_cast<int32_t*>(h_sh.data.data())[0];

    __nv_bfloat16* d_vals=nullptr;
    cudaCheck(cudaMalloc(&d_vals, N * sizeof(__nv_bfloat16)));
    cudaCheck(cudaMemcpy(d_vals, h_vals.data.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    int slices_all[1] = {1};
    int max_block_sums = get_max_num_block_sums(slices_all, 1);
    float* d_out = nullptr;
    cudaCheck(cudaMalloc(&d_out, max_block_sums * sizeof(float)));
    cudaCheck(cudaMemset(d_out, 0, max_block_sums * sizeof(float)));

    global_norm_squared<__nv_bfloat16>(d_out, d_vals, N, 0, 1, max_block_sums, true, 0);
    global_norm_aggregate_kernel<<<1, 1024, 0, 0>>>(d_out, max_block_sums);
    cudaCheck(cudaDeviceSynchronize());

    float sumsq = 0.0f;
    cudaCheck(cudaMemcpy(&sumsq, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    float norm = std::sqrt(sumsq);

    std::string out_dir = join(io, "ref");
    if (system(("mkdir -p " + out_dir).c_str()) != 0) return 1;
    npy::save(join(out_dir, "norm.npy"), npy::DType::F4, {1}, &norm);

    cudaCheck(cudaFree(d_vals));
    cudaCheck(cudaFree(d_out));
    printf("probe_global_norm_ref OK\n");
    return 0;
}
