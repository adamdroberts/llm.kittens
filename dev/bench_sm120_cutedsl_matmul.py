#!/usr/bin/env python3
"""CuTeDSL SM120 GEMM feasibility probe for GPT-2 matmul shapes.

The local nvidia-cutlass-dsl package can import on this machine, but import
availability is weaker evidence than a kernel compile. This script exercises the
installed CuTeDSL grouped-GEMM path on a small BF16 problem, then records whether
that route can be used as a timing provider for the GPT-2 shape matrix.
"""

from __future__ import annotations

import argparse
import importlib
import re
from dataclasses import dataclass

import cutlass
import cutlass.utils as cutlass_utils
import torch


@dataclass(frozen=True)
class Shape:
    name: str
    m: int
    n: int
    k: int
    bias: int
    gelu: int

    @property
    def label(self) -> str:
        return f"{self.name} M={self.m} N={self.n} K={self.k} bias={self.bias} gelu={self.gelu}"


GPT2_SHAPES = (
    Shape("qkv", 65536, 2304, 768, 1, 0),
    Shape("attproj", 65536, 768, 768, 1, 0),
    Shape("fc", 65536, 3072, 768, 1, 1),
    Shape("fcproj", 65536, 768, 3072, 1, 0),
    Shape("lmhead", 65536, 50304, 768, 0, 0),
)


def compact_error(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    if "expects arch to be one of" in text and "sm_120" in text:
        return "local CuTeDSL BF16 grouped-GEMM path rejects sm_120a"
    if "GPU is required" in text:
        return "CUDA runtime was not initialized in this process context"
    return text[:220]


def print_unavailable(reason: str) -> None:
    for shape in GPT2_SHAPES:
        print(f"{shape.label:<56} | CuTeDSL     | unavailable: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CuTeDSL SM120 GPT-2 GEMM feasibility probe")
    parser.add_argument("--smoke-m", type=int, default=128)
    parser.add_argument("--smoke-n", type=int, default=128)
    parser.add_argument("--smoke-k", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()

    print(f"CuTeDSL package: cutlass {getattr(cutlass, '__version__', 'unknown')}")
    print(f"CuTeDSL CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        print(f"CuTeDSL device: {torch.cuda.get_device_name()}; capability=sm_{capability[0]}{capability[1]}")

    try:
        module = importlib.import_module("torch._inductor.kernel.vendored_templates.cutedsl_grouped_gemm")
        elapsed_us = module.run(
            1,
            ((args.smoke_m, args.smoke_n, args.smoke_k, 1),),
            cutlass.BFloat16,
            cutlass.BFloat16,
            cutlass.Float32,
            "k",
            "k",
            "n",
            (128, 64),
            (1, 1),
            False,
            cutlass_utils.TensorMapUpdateMode.SMEM,
            0.1,
            args.warmup,
            args.iterations,
            True,
            False,
        )
    except Exception as exc:
        print_unavailable(compact_error(exc))
        return

    smoke = Shape("cutedsl_smoke", args.smoke_m, args.smoke_n, args.smoke_k, 0, 0)
    print(f"{smoke.label:<56} | CuTeDSL     | {elapsed_us:9.3f} us")
    print_unavailable("smoke shape compiled, exact GPT-2 shape timing not implemented yet")


if __name__ == "__main__":
    main()
