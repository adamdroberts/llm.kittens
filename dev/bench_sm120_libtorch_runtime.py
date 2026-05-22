#!/usr/bin/env python3
"""LibTorch C++ API timing probe for SM120 runtime rows.

This is a focused feasibility benchmark for Torch rows that are faster as
Python operators but would need a C++/trainer-callable route before promotion.
It builds a tiny C++ extension that calls LibTorch Tensor APIs, then times exact
GPT-2 activation, logits memory, and GELU shapes with CUDA events. The default
route wraps existing device pointers with cached from_blob tensors, matching the
shape of a future trainer integration more closely than timing Python-owned
tensors directly.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import os
import subprocess
import statistics
import sys
import sysconfig
from pathlib import Path

import torch
from torch.utils.cpp_extension import include_paths, library_paths
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <cstdint>
#include <torch/extension.h>

torch::Tensor wrap_bf16_cuda(std::uint64_t ptr, std::int64_t elements) {
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
    return torch::from_blob(reinterpret_cast<void*>(ptr), {elements}, options);
}

void zero_inplace(torch::Tensor tensor) {
    tensor.zero_();
}

void copy_inplace(torch::Tensor dst, torch::Tensor src) {
    dst.copy_(src);
}

void gelu_out(torch::Tensor out, torch::Tensor inp) {
    at::gelu_out(out, inp, "tanh");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wrap_bf16_cuda", &wrap_bf16_cuda);
    m.def("zero_inplace", &zero_inplace);
    m.def("copy_inplace", &copy_inplace);
    m.def("gelu_out", &gelu_out);
}
"""


CXX_API_SOURCE = r"""
#include <cstdint>
#include <torch/torch.h>

extern "C" {

void* llmk_wrap_bf16_cuda(std::uint64_t ptr, std::int64_t elements) {
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
    return new torch::Tensor(torch::from_blob(reinterpret_cast<void*>(ptr), {elements}, options));
}

void llmk_destroy_tensor(void* handle) {
    delete static_cast<torch::Tensor*>(handle);
}

void llmk_zero_inplace(void* handle) {
    static_cast<torch::Tensor*>(handle)->zero_();
}

void llmk_copy_inplace(void* dst, void* src) {
    static_cast<torch::Tensor*>(dst)->copy_(*static_cast<torch::Tensor*>(src));
}

void llmk_gelu_out(void* out, void* inp) {
    at::gelu_out(*static_cast<torch::Tensor*>(out), *static_cast<torch::Tensor*>(inp), "tanh");
}

}
"""


def build_extension_manually(name: str, source: str):
    build_dir = Path(os.environ.get("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")) / name
    build_dir.mkdir(parents=True, exist_ok=True)
    source_path = build_dir / f"{name}.cpp"
    source_path.write_text(source)
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    output_path = build_dir / f"{name}{suffix}"
    torch_lib = library_paths()[0]
    py_include = sysconfig.get_paths()["include"]
    include_args = [f"-I{path}" for path in include_paths()]
    include_args.append(f"-I{py_include}")
    link_args = [
        f"-L{torch_lib}",
        f"-Wl,-rpath,{torch_lib}",
        "-ltorch_python",
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
        f"-DTORCH_EXTENSION_NAME={name}",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        str(source_path),
        "-o",
        str(output_path),
        *include_args,
        *link_args,
    ]
    subprocess.run(cmd, check=True)
    spec = importlib.util.spec_from_file_location(name, output_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load compiled extension at {output_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CxxApiRuntime:
    def __init__(self, lib: ctypes.CDLL):
        self.lib = lib
        self.lib.llmk_wrap_bf16_cuda.argtypes = [ctypes.c_uint64, ctypes.c_int64]
        self.lib.llmk_wrap_bf16_cuda.restype = ctypes.c_void_p
        self.lib.llmk_destroy_tensor.argtypes = [ctypes.c_void_p]
        self.lib.llmk_destroy_tensor.restype = None
        self.lib.llmk_zero_inplace.argtypes = [ctypes.c_void_p]
        self.lib.llmk_zero_inplace.restype = None
        self.lib.llmk_copy_inplace.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.llmk_copy_inplace.restype = None
        self.lib.llmk_gelu_out.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.llmk_gelu_out.restype = None

    def wrap_bf16_cuda(self, ptr: int, elements: int) -> ctypes.c_void_p:
        handle = self.lib.llmk_wrap_bf16_cuda(ctypes.c_uint64(ptr), ctypes.c_int64(elements))
        if not handle:
            raise RuntimeError("LibTorch C++ API wrapper returned null tensor handle")
        return ctypes.c_void_p(handle)

    def destroy(self, tensor: ctypes.c_void_p) -> None:
        self.lib.llmk_destroy_tensor(tensor)

    def zero_inplace(self, tensor: ctypes.c_void_p) -> None:
        self.lib.llmk_zero_inplace(tensor)

    def copy_inplace(self, dst: ctypes.c_void_p, src: ctypes.c_void_p) -> None:
        self.lib.llmk_copy_inplace(dst, src)

    def gelu_out(self, out: ctypes.c_void_p, inp: ctypes.c_void_p) -> None:
        self.lib.llmk_gelu_out(out, inp)


def build_cxx_api_runtime() -> CxxApiRuntime:
    build_dir = Path(os.environ.get("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")) / "llmk_sm120_libtorch_cxx_api"
    build_dir.mkdir(parents=True, exist_ok=True)
    source_path = build_dir / "llmk_sm120_libtorch_cxx_api.cpp"
    source_path.write_text(CXX_API_SOURCE)
    output_path = build_dir / "libllmk_sm120_libtorch_cxx_api.so"
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
    return CxxApiRuntime(ctypes.CDLL(str(output_path)))


def build_extension():
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
    name = "llmk_sm120_libtorch_runtime"
    try:
        return load_inline(
            name=name,
            cpp_sources=[CPP_SOURCE],
            with_cuda=False,
            extra_cflags=["-O3"],
            verbose=False,
        )
    except RuntimeError as exc:
        if "Ninja is required" not in str(exc):
            raise
        print("LibTorch C++ extension: ninja not found; falling back to direct c++ build.", file=sys.stderr)
        return build_extension_manually(name, CPP_SOURCE)


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


def print_result(name: str, shape: str, us: float) -> None:
    print(f"{name:<30} | {shape:<28} | {'Torch C++':<12} | {us:9.3f} us")


def verify_memory_ops(ext, tensor: torch.Tensor, tensor_copy: torch.Tensor, *, shape: str) -> None:
    tensor.fill_(1.0)
    ext.zero_inplace(tensor)
    torch.cuda.synchronize()
    nonzero = int(torch.count_nonzero(tensor).item())
    if nonzero != 0:
        raise RuntimeError(f"LibTorch parity cuda_memset {shape} failed: nonzero={nonzero}")
    print(f"LibTorch parity cuda_memset {shape}: PASS")

    tensor.fill_(1.25)
    tensor_copy.zero_()
    ext.copy_inplace(tensor_copy, tensor)
    torch.cuda.synchronize()
    if not torch.equal(tensor_copy, tensor):
        raise RuntimeError(f"LibTorch parity cuda_copy_d2d {shape} failed")
    print(f"LibTorch parity cuda_copy_d2d {shape}: PASS")


def verify_gelu(ext, out: torch.Tensor, inp: torch.Tensor, *, shape: str) -> None:
    ref = torch.nn.functional.gelu(inp, approximate="tanh")
    ext.gelu_out(out, inp)
    torch.cuda.synchronize()
    max_abs = float((out.float() - ref.float()).abs().max().item())
    if max_abs != 0.0:
        raise RuntimeError(f"LibTorch parity gelu_forward {shape} failed: max_abs={max_abs}")
    print(f"LibTorch parity gelu_forward {shape}: PASS max_abs={max_abs:.6f}")


def verify_memory_ops_cxx(
    runtime: CxxApiRuntime,
    tensor: torch.Tensor,
    tensor_copy: torch.Tensor,
    tensor_handle: ctypes.c_void_p,
    tensor_copy_handle: ctypes.c_void_p,
    *,
    shape: str,
) -> None:
    tensor.fill_(1.0)
    runtime.zero_inplace(tensor_handle)
    torch.cuda.synchronize()
    nonzero = int(torch.count_nonzero(tensor).item())
    if nonzero != 0:
        raise RuntimeError(f"LibTorch parity cuda_memset {shape} failed: nonzero={nonzero}")
    print(f"LibTorch parity cuda_memset {shape}: PASS")

    tensor.fill_(1.25)
    tensor_copy.zero_()
    runtime.copy_inplace(tensor_copy_handle, tensor_handle)
    torch.cuda.synchronize()
    if not torch.equal(tensor_copy, tensor):
        raise RuntimeError(f"LibTorch parity cuda_copy_d2d {shape} failed")
    print(f"LibTorch parity cuda_copy_d2d {shape}: PASS")


def verify_gelu_cxx(
    runtime: CxxApiRuntime,
    out: torch.Tensor,
    inp: torch.Tensor,
    out_handle: ctypes.c_void_p,
    inp_handle: ctypes.c_void_p,
    *,
    shape: str,
) -> None:
    ref = torch.nn.functional.gelu(inp, approximate="tanh")
    runtime.gelu_out(out_handle, inp_handle)
    torch.cuda.synchronize()
    max_abs = float((out.float() - ref.float()).abs().max().item())
    if max_abs != 0.0:
        raise RuntimeError(f"LibTorch parity gelu_forward {shape} failed: max_abs={max_abs}")
    print(f"LibTorch parity gelu_forward {shape}: PASS max_abs={max_abs:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="LibTorch C++ SM120 runtime benchmark")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--route",
        choices=("raw-pointer", "tensor", "cxx-api-raw-pointer"),
        default="raw-pointer",
        help=(
            "Use cached extension from_blob wrappers around existing CUDA pointers, "
            "ordinary tensors, or standalone LibTorch C++ API tensor handles with no torch_python link."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(
            "PyTorch CUDA context is not initialized in this process; rerun inside the target benchmark context.",
            file=sys.stderr,
        )
        return 2

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"LibTorch runtime device: {device_name}; capability=sm_{capability[0]}{capability[1]}")
    ext = None
    cxx_runtime: CxxApiRuntime | None = None
    if args.route == "cxx-api-raw-pointer":
        cxx_runtime = build_cxx_api_runtime()
    else:
        ext = build_extension()

    hidden_elems = 64 * 1024 * 768
    grad_elems = 124_475_904
    logits_elems = 64 * 1024 * 50304
    hidden = torch.empty((hidden_elems,), device="cuda", dtype=torch.bfloat16)
    hidden_copy = torch.empty_like(hidden)
    grad = torch.empty((grad_elems,), device="cuda", dtype=torch.bfloat16)
    logits = torch.empty((logits_elems,), device="cuda", dtype=torch.bfloat16)
    logits_copy = torch.empty_like(logits)
    gelu_shape = "BT=65536 C=3072"
    gelu_inp = torch.randn((64 * 1024, 3072), device="cuda", dtype=torch.bfloat16)
    gelu_out = torch.empty_like(gelu_inp)
    tensor_handles: list[ctypes.c_void_p] = []
    if args.route == "raw-pointer":
        assert ext is not None
        hidden_route = ext.wrap_bf16_cuda(hidden.data_ptr(), hidden.numel())
        hidden_copy_route = ext.wrap_bf16_cuda(hidden_copy.data_ptr(), hidden_copy.numel())
        grad_route = ext.wrap_bf16_cuda(grad.data_ptr(), grad.numel())
        logits_route = ext.wrap_bf16_cuda(logits.data_ptr(), logits.numel())
        logits_copy_route = ext.wrap_bf16_cuda(logits_copy.data_ptr(), logits_copy.numel())
        gelu_inp_route = ext.wrap_bf16_cuda(gelu_inp.data_ptr(), gelu_inp.numel())
        gelu_out_route = ext.wrap_bf16_cuda(gelu_out.data_ptr(), gelu_out.numel())
        print("LibTorch runtime route: cached from_blob wrappers over existing CUDA pointers")
    elif args.route == "cxx-api-raw-pointer":
        assert cxx_runtime is not None
        hidden_route = cxx_runtime.wrap_bf16_cuda(hidden.data_ptr(), hidden.numel())
        hidden_copy_route = cxx_runtime.wrap_bf16_cuda(hidden_copy.data_ptr(), hidden_copy.numel())
        grad_route = cxx_runtime.wrap_bf16_cuda(grad.data_ptr(), grad.numel())
        logits_route = cxx_runtime.wrap_bf16_cuda(logits.data_ptr(), logits.numel())
        logits_copy_route = cxx_runtime.wrap_bf16_cuda(logits_copy.data_ptr(), logits_copy.numel())
        gelu_inp_route = cxx_runtime.wrap_bf16_cuda(gelu_inp.data_ptr(), gelu_inp.numel())
        gelu_out_route = cxx_runtime.wrap_bf16_cuda(gelu_out.data_ptr(), gelu_out.numel())
        tensor_handles = [
            hidden_route,
            hidden_copy_route,
            grad_route,
            logits_route,
            logits_copy_route,
            gelu_inp_route,
            gelu_out_route,
        ]
        print("LibTorch runtime route: standalone C++ API cached from_blob handles over existing CUDA pointers")
    else:
        hidden_route = hidden
        hidden_copy_route = hidden_copy
        grad_route = grad
        logits_route = logits
        logits_copy_route = logits_copy
        gelu_inp_route = gelu_inp
        gelu_out_route = gelu_out
        print("LibTorch runtime route: direct Tensor API")

    try:
        if cxx_runtime is not None:
            verify_memory_ops_cxx(
                cxx_runtime,
                hidden,
                hidden_copy,
                hidden_route,
                hidden_copy_route,
                shape=f"hidden_elems={hidden_elems}",
            )
            verify_memory_ops_cxx(
                cxx_runtime,
                logits,
                logits_copy,
                logits_route,
                logits_copy_route,
                shape=f"logits_elems={logits_elems}",
            )
            grad.fill_(1.0)
            cxx_runtime.zero_inplace(grad_route)
            torch.cuda.synchronize()
            grad_nonzero = int(torch.count_nonzero(grad).item())
            if grad_nonzero != 0:
                raise RuntimeError(f"LibTorch parity cuda_memset grad_elems={grad_elems} failed: nonzero={grad_nonzero}")
            print(f"LibTorch parity cuda_memset grad_elems={grad_elems}: PASS")
            verify_gelu_cxx(
                cxx_runtime,
                gelu_out,
                gelu_inp,
                gelu_out_route,
                gelu_inp_route,
                shape=gelu_shape,
            )
            zero_hidden = lambda: cxx_runtime.zero_inplace(hidden_route)
            copy_hidden = lambda: cxx_runtime.copy_inplace(hidden_copy_route, hidden_route)
            zero_grad = lambda: cxx_runtime.zero_inplace(grad_route)
            zero_logits = lambda: cxx_runtime.zero_inplace(logits_route)
            copy_logits = lambda: cxx_runtime.copy_inplace(logits_copy_route, logits_route)
            gelu_forward = lambda: cxx_runtime.gelu_out(gelu_out_route, gelu_inp_route)
        else:
            assert ext is not None
            verify_memory_ops(ext, hidden_route, hidden_copy_route, shape=f"hidden_elems={hidden_elems}")
            verify_memory_ops(ext, logits_route, logits_copy_route, shape=f"logits_elems={logits_elems}")
            grad_route.fill_(1.0)
            ext.zero_inplace(grad_route)
            torch.cuda.synchronize()
            grad_nonzero = int(torch.count_nonzero(grad_route).item())
            if grad_nonzero != 0:
                raise RuntimeError(f"LibTorch parity cuda_memset grad_elems={grad_elems} failed: nonzero={grad_nonzero}")
            print(f"LibTorch parity cuda_memset grad_elems={grad_elems}: PASS")
            verify_gelu(ext, gelu_out_route, gelu_inp_route, shape=gelu_shape)
            zero_hidden = lambda: ext.zero_inplace(hidden_route)
            copy_hidden = lambda: ext.copy_inplace(hidden_copy_route, hidden_route)
            zero_grad = lambda: ext.zero_inplace(grad_route)
            zero_logits = lambda: ext.zero_inplace(logits_route)
            copy_logits = lambda: ext.copy_inplace(logits_copy_route, logits_route)
            gelu_forward = lambda: ext.gelu_out(gelu_out_route, gelu_inp_route)

        print_result(
            "cuda_memset",
            f"hidden_elems={hidden_elems}",
            event_time_us(zero_hidden, warmup=args.warmup, repeats=args.repeats, iters=20),
        )
        print_result(
            "cuda_copy_d2d",
            f"hidden_elems={hidden_elems}",
            event_time_us(copy_hidden, warmup=args.warmup, repeats=args.repeats, iters=20),
        )
        print_result(
            "cuda_memset",
            f"grad_elems={grad_elems}",
            event_time_us(zero_grad, warmup=args.warmup, repeats=args.repeats, iters=20),
        )
        print_result(
            "cuda_memset",
            f"logits_elems={logits_elems}",
            event_time_us(zero_logits, warmup=args.warmup, repeats=args.repeats, iters=1),
        )
        print_result(
            "cuda_copy_d2d",
            f"logits_elems={logits_elems}",
            event_time_us(copy_logits, warmup=args.warmup, repeats=args.repeats, iters=1),
        )
        print_result(
            "gelu_forward",
            gelu_shape,
            event_time_us(gelu_forward, warmup=args.warmup, repeats=args.repeats, iters=25),
        )
    finally:
        if cxx_runtime is not None:
            for handle in tensor_handles:
                cxx_runtime.destroy(handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
