#!/usr/bin/env python3
"""LibTorch C++ API timing probe for SM120 GPT-2 dWeight matmul rows.

This is a focused feasibility benchmark for Torch matmul rows that can win as
Python operators but still need a C++/trainer-callable route before promotion.
It wraps existing CUDA buffers with cached LibTorch Tensor handles and times
dWeight GEMM through a standalone C++ shared library.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.cpp_extension import include_paths, library_paths


CXX_API_SOURCE = r"""
#include <cstdint>
#include <torch/torch.h>

extern "C" {

void* llmk_wrap_bf16_cuda_2d(std::uint64_t ptr, std::int64_t rows, std::int64_t cols) {
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
    return new torch::Tensor(torch::from_blob(reinterpret_cast<void*>(ptr), {rows, cols}, options));
}

void llmk_destroy_tensor(void* handle) {
    delete static_cast<torch::Tensor*>(handle);
}

void llmk_dw_out(void* dst, void* dout, void* inp) {
    auto* dst_tensor = static_cast<torch::Tensor*>(dst);
    auto* dout_tensor = static_cast<torch::Tensor*>(dout);
    auto* inp_tensor = static_cast<torch::Tensor*>(inp);
    at::mm_out(*dst_tensor, dout_tensor->transpose(0, 1), *inp_tensor);
}

void llmk_dw_accum_out(void* dst, void* dout, void* inp) {
    auto* dst_tensor = static_cast<torch::Tensor*>(dst);
    auto* dout_tensor = static_cast<torch::Tensor*>(dout);
    auto* inp_tensor = static_cast<torch::Tensor*>(inp);
    at::addmm_out(*dst_tensor, *dst_tensor, dout_tensor->transpose(0, 1), *inp_tensor, 1.0, 1.0);
}

}
"""

ROUTE_MARKER = "LibTorch matmul route: standalone C++ API cached from_blob handles over existing CUDA pointers"


@dataclass(frozen=True)
class Shape:
    name: str
    m: int
    n: int
    k: int
    bias: bool
    gelu: bool


SHAPES = (
    Shape("qkv", 64 * 1024, 3 * 768, 768, True, False),
    Shape("attproj", 64 * 1024, 768, 768, True, False),
    Shape("fc", 64 * 1024, 4 * 768, 768, True, True),
    Shape("fcproj", 64 * 1024, 768, 4 * 768, True, False),
    Shape("lmhead", 64 * 1024, 50304, 768, False, False),
)


def shape_label(shape: Shape) -> str:
    return (
        f"{shape.name} M={shape.m} N={shape.n} K={shape.k} "
        f"bias={1 if shape.bias else 0} gelu={1 if shape.gelu else 0}"
    )


class CxxApiMatmul:
    def __init__(self, lib: ctypes.CDLL):
        self.lib = lib
        self.lib.llmk_wrap_bf16_cuda_2d.argtypes = [ctypes.c_uint64, ctypes.c_int64, ctypes.c_int64]
        self.lib.llmk_wrap_bf16_cuda_2d.restype = ctypes.c_void_p
        self.lib.llmk_destroy_tensor.argtypes = [ctypes.c_void_p]
        self.lib.llmk_destroy_tensor.restype = None
        self.lib.llmk_dw_out.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.llmk_dw_out.restype = None
        self.lib.llmk_dw_accum_out.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.llmk_dw_accum_out.restype = None

    def wrap_bf16_cuda_2d(self, ptr: int, rows: int, cols: int) -> ctypes.c_void_p:
        handle = self.lib.llmk_wrap_bf16_cuda_2d(
            ctypes.c_uint64(ptr),
            ctypes.c_int64(rows),
            ctypes.c_int64(cols),
        )
        if not handle:
            raise RuntimeError("LibTorch C++ API wrapper returned null tensor handle")
        return ctypes.c_void_p(handle)

    def destroy(self, tensor: ctypes.c_void_p) -> None:
        self.lib.llmk_destroy_tensor(tensor)

    def dw_out(self, dst: ctypes.c_void_p, dout: ctypes.c_void_p, inp: ctypes.c_void_p) -> None:
        self.lib.llmk_dw_out(dst, dout, inp)

    def dw_accum_out(self, dst: ctypes.c_void_p, dout: ctypes.c_void_p, inp: ctypes.c_void_p) -> None:
        self.lib.llmk_dw_accum_out(dst, dout, inp)


def build_cxx_api_matmul() -> CxxApiMatmul:
    build_dir = Path(os.environ.get("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")) / "llmk_sm120_libtorch_matmul"
    build_dir.mkdir(parents=True, exist_ok=True)
    source_path = build_dir / "llmk_sm120_libtorch_matmul.cpp"
    source_path.write_text(CXX_API_SOURCE)
    output_path = build_dir / "libllmk_sm120_libtorch_matmul.so"
    torch_lib = library_paths()[0]
    include_args = [f"-I{path}" for path in include_paths()]
    link_args = [
        f"-L{torch_lib}",
        f"-Wl,-rpath,{torch_lib}",
        "-ltorch",
        "-ltorch_cpu",
        "-ltorch_cuda",
        "-lc10",
        "-lc10_cuda",
    ]
    cmd = [
        os.environ.get("CXX", "c++"),
        "-O3",
        "-shared",
        "-std=c++17",
        "-fPIC",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        str(source_path),
        "-o",
        str(output_path),
        *include_args,
        *link_args,
    ]
    subprocess.run(cmd, check=True)
    return CxxApiMatmul(ctypes.CDLL(str(output_path)))


def event_time_us(fn, *, warmup: int, repeats: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)
    return statistics.median(samples)


def max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs.float() - rhs.float()).abs().max().item())


def print_result(op: str, us: float) -> None:
    print(f"  {op:<8} Torch C++ {us:9.2f} us")


def bench_shape(runtime: CxxApiMatmul, shape: Shape, *, repeats: int, large_repeats: int, warmup: int) -> list[dict[str, object]]:
    iters = 1 if shape.n >= 8192 else 5
    repeats_for_shape = large_repeats if shape.n >= 8192 else repeats

    torch.manual_seed(2400 + shape.n + shape.k)
    inp = torch.randn((shape.m, shape.k), device="cuda", dtype=torch.bfloat16)
    dout = torch.randn((shape.m, shape.n), device="cuda", dtype=torch.bfloat16)
    dw = torch.empty((shape.n, shape.k), device="cuda", dtype=torch.bfloat16)
    dw_accum = torch.randn((shape.n, shape.k), device="cuda", dtype=torch.bfloat16)
    dw_accum_initial = dw_accum.clone()

    handles: list[ctypes.c_void_p] = []
    inp_handle = runtime.wrap_bf16_cuda_2d(inp.data_ptr(), shape.m, shape.k)
    dout_handle = runtime.wrap_bf16_cuda_2d(dout.data_ptr(), shape.m, shape.n)
    dw_handle = runtime.wrap_bf16_cuda_2d(dw.data_ptr(), shape.n, shape.k)
    dw_accum_handle = runtime.wrap_bf16_cuda_2d(dw_accum.data_ptr(), shape.n, shape.k)
    handles.extend([inp_handle, dout_handle, dw_handle, dw_accum_handle])

    rows: list[dict[str, object]] = []
    print(f"\n{shape_label(shape)}")
    try:
        ref_dw = dout.T @ inp
        runtime.dw_out(dw_handle, dout_handle, inp_handle)
        torch.cuda.synchronize()
        dw_diff = max_abs_diff(dw, ref_dw)
        if dw_diff > 0.0:
            raise RuntimeError(f"LibTorch matmul parity dW {shape.name} failed: max_abs={dw_diff}")
        print(f"LibTorch matmul parity dW {shape.name}: PASS max_abs={dw_diff:.6f}")

        ref_accum = torch.addmm(dw_accum_initial, dout.T, inp, beta=1.0, alpha=1.0)
        dw_accum.copy_(dw_accum_initial)
        runtime.dw_accum_out(dw_accum_handle, dout_handle, inp_handle)
        torch.cuda.synchronize()
        accum_diff = max_abs_diff(dw_accum, ref_accum)
        if accum_diff > 0.0:
            raise RuntimeError(f"LibTorch matmul parity dW+accum {shape.name} failed: max_abs={accum_diff}")
        print(f"LibTorch matmul parity dW+accum {shape.name}: PASS max_abs={accum_diff:.6f}")

        dw_us = event_time_us(
            lambda: runtime.dw_out(dw_handle, dout_handle, inp_handle),
            warmup=warmup,
            repeats=repeats_for_shape,
            iters=iters,
        )
        print_result("dW", dw_us)
        rows.append(
            {
                "suite": "matmul",
                "kernel": "dW",
                "shape_name": shape.name,
                "shape": shape_label(shape),
                "stack": "Torch C++",
                "time_us": dw_us,
                "parity_pass": True,
                "parity_max_abs": dw_diff,
                "timed_iters": iters,
                "timed_repeats": repeats_for_shape,
            }
        )
        accum_us = event_time_us(
            lambda: runtime.dw_accum_out(dw_accum_handle, dout_handle, inp_handle),
            warmup=warmup,
            repeats=repeats_for_shape,
            iters=iters,
        )
        print_result("dW+accum", accum_us)
        rows.append(
            {
                "suite": "matmul",
                "kernel": "dW+accum",
                "shape_name": shape.name,
                "shape": shape_label(shape),
                "stack": "Torch C++",
                "time_us": accum_us,
                "parity_pass": True,
                "parity_max_abs": accum_diff,
                "timed_iters": iters,
                "timed_repeats": repeats_for_shape,
            }
        )
        return rows
    finally:
        for handle in handles:
            runtime.destroy(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="LibTorch C++ SM120 GPT-2 dWeight matmul benchmark")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--large-repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--shape", choices=[shape.name for shape in SHAPES], action="append")
    parser.add_argument("--json-out", type=Path, help="Write structured LibTorch matmul timing/parity evidence")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(
            "PyTorch CUDA context is not initialized in this process; rerun inside the target benchmark context.",
            file=sys.stderr,
        )
        return 2

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"LibTorch matmul device: {device_name}; capability=sm_{capability[0]}{capability[1]}")
    print(ROUTE_MARKER)
    runtime = build_cxx_api_matmul()
    selected = set(args.shape) if args.shape else {shape.name for shape in SHAPES}
    rows: list[dict[str, object]] = []
    for shape in SHAPES:
        if shape.name in selected:
            rows.extend(bench_shape(runtime, shape, repeats=args.repeats, large_repeats=args.large_repeats, warmup=args.warmup))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "route_marker": ROUTE_MARKER,
                    "device_name": device_name,
                    "device_capability": f"sm_{capability[0]}{capability[1]}",
                    "selected_shapes": [shape.name for shape in SHAPES if shape.name in selected],
                    "repeats": args.repeats,
                    "large_repeats": args.large_repeats,
                    "warmup": args.warmup,
                    "argv": sys.argv,
                    "rows": rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
