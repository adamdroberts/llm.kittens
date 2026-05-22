/*
CUDA memory-operation candidates for SM120 optimization rounds.

These wrappers are trainer-callable C++/CUDA routes for exact BF16 activation
zero/copy rows. They intentionally do not replace cudaMemsetAsync or
cudaMemcpyAsync in the trainer until benchmark and TinyStories evidence proves
a win.
*/
#ifndef LLMK_MEMORY_CUH
#define LLMK_MEMORY_CUH

#include <algorithm>
#include <assert.h>
#include <stdint.h>

#include "cuda_common.h"
#include "cuda_utils.cuh"

#if defined(LLMK_SM120_USE_LIBTORCH_MEMORY)
extern "C" void* llmk_libtorch_wrap_bf16_cuda(void* ptr, int64_t elements);
extern "C" void llmk_libtorch_destroy_tensor(void* handle);
extern "C" void llmk_libtorch_zero_inplace(void* handle, cudaStream_t stream);

static inline void* memory_libtorch_wrap_floatx(floatX* ptr, size_t elements) {
    assert(elements <= (size_t)INT64_MAX);
    return llmk_libtorch_wrap_bf16_cuda((void*)ptr, (int64_t)elements);
}

static inline void memory_libtorch_zero_floatx(void* handle, cudaStream_t stream) {
    assert(handle != nullptr);
    llmk_libtorch_zero_inplace(handle, stream);
}

static inline void memory_libtorch_destroy(void** handle) {
    if (handle != nullptr && *handle != nullptr) {
        llmk_libtorch_destroy_tensor(*handle);
        *handle = nullptr;
    }
}
#else
static inline void memory_libtorch_destroy(void**) {}
#endif

#if defined(KITTENS_SM120) && !defined(LLMK_SM120_MEMORY_BLOCK_SIZE)
#define LLMK_SM120_MEMORY_BLOCK_SIZE 1024
#endif

#ifndef LLMK_SM120_MEMORY_LOAD_POLICY
#define LLMK_SM120_MEMORY_LOAD_POLICY 1
#endif

#ifndef LLMK_SM120_MEMORY_STORE_POLICY
#define LLMK_SM120_MEMORY_STORE_POLICY 1
#endif

__device__ inline x128 memory_load_floatx(const floatX* src) {
#if LLMK_SM120_MEMORY_LOAD_POLICY == 0
    return load128(src);
#elif LLMK_SM120_MEMORY_LOAD_POLICY == 1
    return load128cs(src);
#else
#error Unsupported LLMK_SM120_MEMORY_LOAD_POLICY
#endif
}

__device__ inline void memory_store_floatx(floatX* dst, x128 value) {
#if LLMK_SM120_MEMORY_STORE_POLICY == 0
    store128(dst, value);
#elif LLMK_SM120_MEMORY_STORE_POLICY == 1
    store128cs(dst, value);
#elif LLMK_SM120_MEMORY_STORE_POLICY == 2
    store128cg(dst, value);
#else
#error Unsupported LLMK_SM120_MEMORY_STORE_POLICY
#endif
}

__global__ void memory_zero_floatx_kernel(floatX* dst, size_t packed_count) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    const x128 zero = x128::zeros();
    for (size_t pack = idx; pack < packed_count; pack += stride) {
        memory_store_floatx(dst + pack * x128::size, zero);
    }
}

__global__ void memory_copy_floatx_kernel(floatX* dst, const floatX* src, size_t packed_count) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t pack = idx; pack < packed_count; pack += stride) {
        const x128 value = memory_load_floatx(src + pack * x128::size);
        memory_store_floatx(dst + pack * x128::size, value);
    }
}

static inline int memory_grid_size(size_t packed_count, int block_size) {
    const int occupancy_grid = deviceProp.multiProcessorCount * 8;
    const size_t required_grid = CEIL_DIV(packed_count, (size_t)block_size);
    return (int)std::min((size_t)occupancy_grid, required_grid);
}

void memory_zero_floatx(floatX* dst, size_t elements, cudaStream_t stream) {
    NVTX_RANGE_FN();
#if defined(KITTENS_SM120)
    const int block_size = LLMK_SM120_MEMORY_BLOCK_SIZE;
#else
    const int block_size = 256;
#endif
    assert(((uintptr_t)dst % sizeof(int4)) == 0);
    assert(elements % x128::size == 0);
    const size_t packed_count = elements / x128::size;
    if (packed_count == 0) {
        return;
    }
    const int grid_size = memory_grid_size(packed_count, block_size);
    memory_zero_floatx_kernel<<<grid_size, block_size, 0, stream>>>(dst, packed_count);
    cudaCheck(cudaGetLastError());
}

void memory_copy_floatx(floatX* dst, const floatX* src, size_t elements, cudaStream_t stream) {
    NVTX_RANGE_FN();
#if defined(KITTENS_SM120)
    const int block_size = LLMK_SM120_MEMORY_BLOCK_SIZE;
#else
    const int block_size = 256;
#endif
    assert(((uintptr_t)dst % sizeof(int4)) == 0);
    assert(((uintptr_t)src % sizeof(int4)) == 0);
    assert(elements % x128::size == 0);
    const size_t packed_count = elements / x128::size;
    if (packed_count == 0) {
        return;
    }
    const int grid_size = memory_grid_size(packed_count, block_size);
    memory_copy_floatx_kernel<<<grid_size, block_size, 0, stream>>>(dst, src, packed_count);
    cudaCheck(cudaGetLastError());
}

#endif
