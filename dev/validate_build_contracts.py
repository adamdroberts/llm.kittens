#!/usr/bin/env python3
"""Source-level guards for llm.kittens build and precision contracts."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
CUDA_COMMON = ROOT / "llmc" / "cuda_common.h"
TK_COMMON = ROOT / "llmc" / "tk" / "tk_common.cuh"
CUBLAS_COMMON = ROOT / "llmc" / "cublas_common.h"
ATTENTION_H100 = ROOT / "llmc" / "tk" / "attention_h100.cuh"
ATTENTION_GQA_H100 = ROOT / "llmc" / "tk" / "attention_gqa_h100.cuh"
ROPE_TK = ROOT / "llmc" / "tk" / "rope_tk.cuh"
MFU = ROOT / "llmc" / "mfu.h"
HARNESS = ROOT / "scripts" / "validate_goal_h100.sh"

MAKEFILE_REQUIRED = [
    "--use_fast_math",
    "-std=c++20",
    "-I$(TK_ROOT)/include",
    "-I$(TK_ROOT)/prototype",
    "-DKITTENS_SM90",
    "-gencode arch=compute_90a,code=sm_90a",
    "-DENABLE_BF16",
]

MAKEFILE_FORBIDDEN = [
    "-DENABLE_FP16",
    "-DENABLE_FP32",
    "-lcublas",
    "-lcublasLt",
    "-lcudnn",
]

CUDA_COMMON_REQUIRED = [
    "#include <cuda_bf16.h>",
    "#if defined(ENABLE_FP32) || defined(ENABLE_FP16)",
    '#error "llm.kittens v1 is BF16-only.',
    "typedef __nv_bfloat16 floatX;",
    "#define PRECISION_MODE PRECISION_BF16",
]

TK_COMMON_REQUIRED = [
    "#include <kittens.cuh>",
    "#include <prototype.cuh>",
    "static_assert(std::is_same_v<floatX, __nv_bfloat16>",
    "#ifndef KITTENS_SM90",
    '#error "llm.kittens v1 targets H100 (sm_90a). Build with -DKITTENS_SM90."',
    "using bf16 = ::kittens::bf16;",
    "static_assert(sizeof(bf16) == sizeof(__nv_bfloat16)",
    "constexpr size_t TK_ALIGN = 128;",
    "tk_set_max_dynamic_smem",
    "cudaFuncAttributeMaxDynamicSharedMemorySize",
]

CUBLAS_COMMON_REQUIRED = [
    "cublas_common.h",
    "intentionally empty",
]

CUBLAS_COMMON_FORBIDDEN = [
    "#include <cublas",
    "#include <cudnn",
    "cublasHandle_t",
    "cublasLtHandle_t",
    "cudnnHandle_t",
    "cublasCreate",
    "cublasLtCreate",
    "cudnnCreate",
]

HARNESS_REQUIRED = [
    "if [ \"${ALLOW_NON_H100:-0}\" != \"1\" ]; then",
    "cap + 0 < 9.0 || cap + 0 >= 10.0",
    "goal.md runtime gates require H100/sm_90-class GPUs; set ALLOW_NON_H100=1 only for dry compile/debug runs",
]

TK_COORD_FORBIDDEN = [
    "{blockIdx",
    "blockIdx.x *",
    "blockIdx.y *",
    "blockIdx.z *",
]

MFU_REQUIRED = [
    "nvmlTemperature_t temperature = {};",
    "temperature.version = nvmlTemperature_v1;",
    "nvmlDeviceGetTemperatureV",
    "nvmlDeviceGetCurrentClocksEventReasons",
    "nvmlClocksEventReasonSwPowerCap",
    "nvmlClocksEventReasonSwThermalSlowdown",
]

MFU_FORBIDDEN = [
    "nvmlDeviceGetTemperature(device,",
    "nvmlDeviceGetCurrentClocksThrottleReasons",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def require_contains(text: str, needles: list[str], context: str, failures: list[str]) -> None:
    for needle in needles:
        if needle not in text:
            failures.append(f"{context} missing {needle!r}")


def require_absent(text: str, needles: list[str], context: str, failures: list[str]) -> None:
    for needle in needles:
        if needle in text:
            failures.append(f"{context} unexpectedly contains {needle!r}")


def main() -> None:
    makefile = MAKEFILE.read_text()
    cuda_common = CUDA_COMMON.read_text()
    tk_common = TK_COMMON.read_text()
    cublas_common = CUBLAS_COMMON.read_text()
    attention_h100 = ATTENTION_H100.read_text()
    attention_gqa_h100 = ATTENTION_GQA_H100.read_text()
    rope_tk = ROPE_TK.read_text()
    mfu = MFU.read_text()
    harness = HARNESS.read_text()
    failures: list[str] = []

    require_contains(makefile, MAKEFILE_REQUIRED, rel(MAKEFILE), failures)
    require_absent(makefile, MAKEFILE_FORBIDDEN, rel(MAKEFILE), failures)

    require_contains(cuda_common, CUDA_COMMON_REQUIRED, rel(CUDA_COMMON), failures)
    require_contains(tk_common, TK_COMMON_REQUIRED, rel(TK_COMMON), failures)

    require_contains(cublas_common, CUBLAS_COMMON_REQUIRED, rel(CUBLAS_COMMON), failures)
    require_absent(cublas_common, CUBLAS_COMMON_FORBIDDEN, rel(CUBLAS_COMMON), failures)
    require_contains(harness, HARNESS_REQUIRED, rel(HARNESS), failures)

    for path, text, required_grid_casts in [
        (ATTENTION_H100, attention_h100, [
            "static_cast<int>(blockIdx.x)",
            "static_cast<int>(blockIdx.y)",
            "static_cast<int>(blockIdx.z)",
        ]),
        (ATTENTION_GQA_H100, attention_gqa_h100, [
            "static_cast<int>(blockIdx.x)",
            "static_cast<int>(blockIdx.y)",
            "static_cast<int>(blockIdx.z)",
        ]),
        (ROPE_TK, rope_tk, [
            "static_cast<int>(blockIdx.x)",
            "static_cast<int>(blockIdx.y)",
        ]),
    ]:
        require_contains(text, required_grid_casts, rel(path), failures)
        require_absent(text, TK_COORD_FORBIDDEN, rel(path), failures)

    require_contains(mfu, MFU_REQUIRED, rel(MFU), failures)
    require_absent(mfu, MFU_FORBIDDEN, rel(MFU), failures)

    if failures:
        raise AssertionError("\n".join(failures))
    print("Build contract source guards OK")


if __name__ == "__main__":
    main()
