/*
cuda_runtime_check.cu - lightweight CUDA driver/runtime preflight.

This target intentionally avoids model code and kernels. It verifies that the
CUDA runtime can see at least one device, set device 0, allocate memory, and
reports the driver/runtime versions before the heavier H100 validation phases.
*/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

static void cuda_check(cudaError_t status, const char* expr, const char* file, int line) {
    if (status != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] %s at %s:%d:\n%s\n",
                expr, file, line, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)

static bool device_name_contains(const char* name, const char* needle) {
    return std::strstr(name, needle) != nullptr;
}

int main() {
    int driver_version = 0;
    int runtime_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    printf("CUDA driver version: %d\n", driver_version);
    printf("CUDA runtime version: %d\n", runtime_version);
    fflush(stdout);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 1) {
        fprintf(stderr, "No CUDA devices visible to the runtime.\n");
        return EXIT_FAILURE;
    }
    printf("CUDA visible devices: %d\n", device_count);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("CUDA device 0: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    const char* allow_non_h100 = getenv("ALLOW_NON_H100");
    const bool allow_non_h100_debug =
        allow_non_h100 != nullptr && std::strcmp(allow_non_h100, "1") == 0;
    const bool sm90_class = prop.major == 9;
    const bool named_hopper =
        device_name_contains(prop.name, "H100") ||
        device_name_contains(prop.name, "H200") ||
        device_name_contains(prop.name, "GH200");
    if (!allow_non_h100_debug && !sm90_class && !named_hopper) {
        fprintf(stderr,
                "goal.md runtime gates require H100/sm_90-class GPUs; "
                "detected %s (sm_%d%d). "
                "Set ALLOW_NON_H100=1 only for dry debugging.\n",
                prop.name, prop.major, prop.minor);
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(0));
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, 1));
    CUDA_CHECK(cudaMemset(ptr, 0, 1));
    CUDA_CHECK(cudaFree(ptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("CUDA runtime check passed.\n");
    return EXIT_SUCCESS;
}
