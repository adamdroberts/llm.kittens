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

static bool target_is_rtx5090(const char* target) {
    return target != nullptr &&
           (std::strcmp(target, "rtx5090") == 0 ||
            std::strcmp(target, "rtx-5090") == 0 ||
            std::strcmp(target, "5090") == 0 ||
            std::strcmp(target, "sm120") == 0 ||
            std::strcmp(target, "sm_120") == 0);
}

static bool target_is_blackwell(const char* target) {
    return target != nullptr &&
           (std::strcmp(target, "blackwell") == 0 ||
            std::strcmp(target, "b200") == 0 ||
            std::strcmp(target, "gb200") == 0 ||
            std::strcmp(target, "sm100") == 0 ||
            std::strcmp(target, "sm_100") == 0 ||
            std::strcmp(target, "sm103") == 0 ||
            std::strcmp(target, "sm_103") == 0);
}

static bool target_is_h100(const char* target) {
    return target == nullptr ||
           std::strcmp(target, "") == 0 ||
           std::strcmp(target, "h100") == 0 ||
           std::strcmp(target, "hopper") == 0;
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
    const char* device_target = getenv("DEVICE_TEST_TARGET");
    const bool rtx5090_target = target_is_rtx5090(device_target);
    const bool blackwell_target = target_is_blackwell(device_target);
    const bool h100_target = target_is_h100(device_target);
    if (!rtx5090_target && !blackwell_target && !h100_target) {
        fprintf(stderr, "Unsupported DEVICE_TEST_TARGET=%s; use h100, blackwell, or rtx5090.\n", device_target);
        return EXIT_FAILURE;
    }
    printf("CUDA device target: %s\n", rtx5090_target ? "rtx5090" : (blackwell_target ? "blackwell" : "h100"));

    const char* allow_non_h100 = getenv("ALLOW_NON_H100");
    const bool allow_non_h100_debug =
        allow_non_h100 != nullptr && std::strcmp(allow_non_h100, "1") == 0;
    const bool sm90_class = prop.major == 9;
    const bool sm10x_class = prop.major == 10 && (prop.minor == 0 || prop.minor == 3);
    const bool sm120_class = prop.major == 12 && prop.minor == 0;
    const bool named_hopper =
        device_name_contains(prop.name, "H100") ||
        device_name_contains(prop.name, "H200") ||
        device_name_contains(prop.name, "GH200");
    const bool named_blackwell =
        device_name_contains(prop.name, "B200") ||
        device_name_contains(prop.name, "GB200") ||
        device_name_contains(prop.name, "Blackwell");
    const bool named_rtx5090 = device_name_contains(prop.name, "RTX 5090");
    if (!allow_non_h100_debug) {
        if (blackwell_target && !sm10x_class && !named_blackwell) {
            fprintf(stderr,
                    "device tests target Blackwell B200/GB200 sm_100/sm_103-class GPUs; "
                    "detected %s (sm_%d%d). "
                    "Set ALLOW_NON_H100=1 only for dry debugging.\n",
                    prop.name, prop.major, prop.minor);
            return EXIT_FAILURE;
        }
        if (rtx5090_target && !sm120_class && !named_rtx5090) {
            fprintf(stderr,
                    "device tests target RTX 5090/sm_120-class GPUs; "
                    "detected %s (sm_%d%d). "
                    "Set ALLOW_NON_H100=1 only for dry debugging.\n",
                    prop.name, prop.major, prop.minor);
            return EXIT_FAILURE;
        }
        if (h100_target && !sm90_class && !named_hopper) {
            fprintf(stderr,
                    "goal.md runtime gates require H100/sm_90-class GPUs; "
                    "detected %s (sm_%d%d). "
                    "Set ALLOW_NON_H100=1 only for dry debugging.\n",
                    prop.name, prop.major, prop.minor);
            return EXIT_FAILURE;
        }
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
