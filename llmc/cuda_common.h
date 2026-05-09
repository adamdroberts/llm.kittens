/*
Common utilities for CUDA code in llm.kittens.

Diverges from llm.c/llmc/cuda_common.h in one important way: floatX is locked
to __nv_bfloat16. ThunderKittens H100 GEMM and MHA kernels are bf16-only, so
v1 of llm.kittens does not support FP16/FP32 activations. Master weights,
gradient accumulation, and AdamW state remain FP32 as in llm.c.
*/
#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <type_traits>      // std::bool_constant
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>
#include <cuda_profiler_api.h>
#include <cuda_bf16.h>

#include "utils.h"

// ----------------------------------------------------------------------------
// Global defines and settings

extern cudaDeviceProp deviceProp;

#define WARP_SIZE 32U

// try to make sure that 2 blocks fit on H100 to maximise latency tolerance
#if __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#else
#define MAX_1024_THREADS_BLOCKS 1
#endif

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

constexpr std::bool_constant<true> True;
constexpr std::bool_constant<false> False;

// ----------------------------------------------------------------------------
// Error checking

inline void cudaCheck_(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck_(err, __FILE__, __LINE__))

template<class T>
inline void cudaFreeCheck(T** ptr, const char *file, int line) {
    cudaError_t error = cudaFree(*ptr);
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    *ptr = nullptr;
}
#define cudaFreeCheck(ptr) (cudaFreeCheck(ptr, __FILE__, __LINE__))

// ----------------------------------------------------------------------------
// Precision settings — BF16-only in llm.kittens v1

enum PrecisionMode {
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16
};

#if defined(ENABLE_FP32) || defined(ENABLE_FP16)
#error "llm.kittens v1 is BF16-only. ThunderKittens H100 GEMM/MHA kernels are bf16. Build with PRECISION=BF16 (the default)."
#endif

typedef __nv_bfloat16 floatX;
#define PRECISION_MODE PRECISION_BF16

// ----------------------------------------------------------------------------
// Profiler utils

class NvtxRange {
 public:
    NvtxRange(const char* s) { nvtxRangePush(s); }
    NvtxRange(const std::string& base_str, int number) {
        std::string range_string = base_str + " " + std::to_string(number);
        nvtxRangePush(range_string.c_str());
    }
    ~NvtxRange() { nvtxRangePop(); }
};
#define NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)

// ----------------------------------------------------------------------------
// Utilities to Read & Write between CUDA memory <-> files

inline void device_to_file(FILE* dest, void* src, size_t num_bytes, size_t buffer_size, cudaStream_t stream) {
    char* buffer_space;
    cudaCheck(cudaMallocHost(&buffer_space, 2*buffer_size));
    void* read_buffer = buffer_space;
    void* write_buffer = buffer_space + buffer_size;

    char* gpu_read_ptr = (char*)src;
    size_t copy_amount = std::min(buffer_size, num_bytes);
    cudaCheck(cudaMemcpyAsync(read_buffer, gpu_read_ptr, copy_amount, cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    size_t rest_bytes = num_bytes - copy_amount;
    size_t write_buffer_size = copy_amount;
    gpu_read_ptr += copy_amount;

    std::swap(read_buffer, write_buffer);
    while(rest_bytes > 0) {
        copy_amount = std::min(buffer_size, rest_bytes);
        cudaCheck(cudaMemcpyAsync(read_buffer, gpu_read_ptr, copy_amount, cudaMemcpyDeviceToHost, stream));
        fwriteCheck(write_buffer, 1, write_buffer_size, dest);
        cudaCheck(cudaStreamSynchronize(stream));

        std::swap(read_buffer, write_buffer);
        rest_bytes -= copy_amount;
        write_buffer_size = copy_amount;
        gpu_read_ptr += copy_amount;
    }

    fwriteCheck(write_buffer, 1, write_buffer_size, dest);
    cudaCheck(cudaFreeHost(buffer_space));
}

inline void file_to_device(void* dest, FILE* src, size_t num_bytes, size_t buffer_size, cudaStream_t stream) {
    char* buffer_space;
    cudaCheck(cudaMallocHost(&buffer_space, 2*buffer_size, cudaHostAllocWriteCombined));
    void* read_buffer = buffer_space;
    void* write_buffer = buffer_space + buffer_size;

    char* gpu_write_ptr = (char*)dest;
    size_t copy_amount = std::min(buffer_size, num_bytes);
    freadCheck(read_buffer, 1, copy_amount, src);

    size_t rest_bytes = num_bytes - copy_amount;
    size_t write_buffer_size = copy_amount;
    std::swap(read_buffer, write_buffer);

    while(rest_bytes > 0) {
        copy_amount = std::min(buffer_size, rest_bytes);
        cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_buffer, write_buffer_size, cudaMemcpyHostToDevice, stream));
        gpu_write_ptr += write_buffer_size;
        freadCheck(read_buffer, 1, copy_amount, src);
        cudaCheck(cudaStreamSynchronize(stream));

        std::swap(read_buffer, write_buffer);
        rest_bytes -= copy_amount;
        write_buffer_size = copy_amount;
    }

    cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_buffer, write_buffer_size, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    cudaCheck(cudaFreeHost(buffer_space));
}

#endif // CUDA_COMMON_H
