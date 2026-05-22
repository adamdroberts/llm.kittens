#!/usr/bin/env python3
"""Build and run a standalone LibTorch executable for trainer-link probing.

The Python LibTorch runtime benchmark proves operator timing, but a trainer
promotion also needs evidence that the C++ dependency shape links outside a
Python extension. This probe builds a small executable that links against
LibTorch without torch_python, wraps existing CUDA pointers with from_blob
tensors, and verifies zero/copy behavior.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from torch.utils.cpp_extension import CUDA_HOME, include_paths, library_paths


CPP_SOURCE = r"""
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <torch/torch.h>

static void check_cuda(cudaError_t err, const char* expr) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA runtime context init failed at %s: %s\n", expr, cudaGetErrorString(err));
        std::exit(2);
    }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr)

int main() {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        std::fprintf(stderr, "CUDA runtime context init failed in this process; rerun in the target GPU-visible context\n");
        return 2;
    }
    CHECK_CUDA(cudaSetDevice(0));

    const std::int64_t elements = 1 << 20;
    const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(std::uint16_t);
    void* lhs_ptr = nullptr;
    void* rhs_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&lhs_ptr, bytes));
    CHECK_CUDA(cudaMalloc(&rhs_ptr, bytes));

    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
    torch::Tensor lhs = torch::from_blob(lhs_ptr, {elements}, options);
    torch::Tensor rhs = torch::from_blob(rhs_ptr, {elements}, options);

    lhs.zero_();
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<std::uint16_t> host_lhs(elements);
    CHECK_CUDA(cudaMemcpy(host_lhs.data(), lhs_ptr, bytes, cudaMemcpyDeviceToHost));
    for (std::uint16_t value : host_lhs) {
        if (value != 0) {
            std::fprintf(stderr, "LibTorch trainer link probe zero parity failed\n");
            return 3;
        }
    }

    lhs.fill_(1.25);
    rhs.zero_();
    rhs.copy_(lhs);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<std::uint16_t> host_rhs(elements);
    CHECK_CUDA(cudaMemcpy(host_lhs.data(), lhs_ptr, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(host_rhs.data(), rhs_ptr, bytes, cudaMemcpyDeviceToHost));
    for (std::int64_t i = 0; i < elements; ++i) {
        if (host_lhs[static_cast<std::size_t>(i)] != host_rhs[static_cast<std::size_t>(i)]) {
            std::fprintf(stderr, "LibTorch trainer link probe copy parity failed at %lld\n", static_cast<long long>(i));
            return 4;
        }
    }

    CHECK_CUDA(cudaFree(lhs_ptr));
    CHECK_CUDA(cudaFree(rhs_ptr));
    std::printf("LibTorch trainer link runtime: PASS zero/copy from_blob executable\n");
    return 0;
}
"""


def build_probe(build_dir: Path) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    source_path = build_dir / "llmk_libtorch_trainer_link_probe.cpp"
    binary_path = build_dir / "llmk_libtorch_trainer_link_probe"
    source_path.write_text(CPP_SOURCE)

    torch_lib = library_paths()[0]
    cuda_home = Path(CUDA_HOME) if CUDA_HOME else Path("/usr/local/cuda")
    cuda_include = cuda_home / "include"
    cuda_lib64 = cuda_home / "lib64"
    include_args = [f"-I{path}" for path in include_paths()]
    if cuda_include.exists():
        include_args.append(f"-I{cuda_include}")
    link_args = [
        f"-L{torch_lib}",
        f"-Wl,-rpath,{torch_lib}",
        "-ltorch",
        "-ltorch_cpu",
        "-ltorch_cuda",
        "-lc10",
        "-lc10_cuda",
    ]
    if cuda_lib64.exists():
        link_args.extend([f"-L{cuda_lib64}", f"-Wl,-rpath,{cuda_lib64}"])
    link_args.append("-lcudart")

    cmd = [
        os.environ.get("CXX", "c++"),
        "-O2",
        "-std=c++17",
        f"-D_GLIBCXX_USE_CXX11_ABI={torch_cxx11_abi()}",
        str(source_path),
        "-o",
        str(binary_path),
        *include_args,
        *link_args,
    ]
    subprocess.run(cmd, check=True)
    return binary_path


def torch_cxx11_abi() -> int:
    import torch

    return int(torch._C._GLIBCXX_USE_CXX11_ABI)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate standalone LibTorch trainer link route")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(os.environ.get("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions"))
        / "llmk_libtorch_trainer_link_probe",
    )
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    print("LibTorch trainer link route: standalone executable without torch_python")
    binary_path = build_probe(args.build_dir)
    print(f"LibTorch trainer link compile: PASS {binary_path}")
    if args.compile_only:
        return 0
    proc = subprocess.run([str(binary_path)], text=True, capture_output=True, check=False)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        return proc.returncode
    print("LibTorch trainer link probe: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
