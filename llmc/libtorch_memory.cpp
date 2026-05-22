#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

[[noreturn]] static void libtorch_memory_abort(const char* op, const char* message) {
    std::fprintf(stderr, "LibTorch memory route failed in %s: %s\n", op, message);
    std::abort();
}

extern "C" void* llmk_libtorch_wrap_bf16_cuda(void* ptr, std::int64_t elements) {
    try {
        int device = 0;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess) {
            libtorch_memory_abort("wrap_bf16_cuda", cudaGetErrorString(err));
        }
        auto options = at::TensorOptions()
            .device(c10::Device(c10::kCUDA, device))
            .dtype(at::kBFloat16);
        return new at::Tensor(at::from_blob(ptr, {elements}, options));
    } catch (const std::exception& exc) {
        libtorch_memory_abort("wrap_bf16_cuda", exc.what());
    }
}

extern "C" void llmk_libtorch_destroy_tensor(void* handle) {
    delete static_cast<at::Tensor*>(handle);
}

extern "C" void llmk_libtorch_zero_inplace(void* handle, cudaStream_t raw_stream) {
    try {
        at::Tensor* tensor = static_cast<at::Tensor*>(handle);
        const int device = tensor->device().index();
        if (raw_stream != nullptr) {
            at::cuda::CUDAStream stream = at::cuda::getStreamFromExternal(raw_stream, device);
            at::cuda::CUDAStreamGuard guard(stream);
            tensor->zero_();
        } else {
            c10::cuda::CUDAGuard guard(device);
            tensor->zero_();
        }
    } catch (const std::exception& exc) {
        libtorch_memory_abort("zero_inplace", exc.what());
    }
}
