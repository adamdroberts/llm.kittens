#!/usr/bin/env python3
"""Torch SM120 GEMM timing prototype for GPT-2 shapes."""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Shape:
    name: str
    m: int
    n: int
    k: int
    bias: bool
    gelu: bool
    dgelu: bool


SHAPES = (
    Shape("qkv", 64 * 1024, 3 * 768, 768, True, False, False),
    Shape("attproj", 64 * 1024, 768, 768, True, False, False),
    Shape("fc", 64 * 1024, 4 * 768, 768, True, True, False),
    Shape("fcproj", 64 * 1024, 768, 4 * 768, True, False, True),
    Shape("lmhead", 64 * 1024, 50304, 768, False, False, False),
)


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


def gelu_backward_tanh(dout: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    x = inp.float()
    tanh_arg = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    tanh_out = torch.tanh(tanh_arg)
    sech2 = 1.0 - tanh_out * tanh_out
    local_grad = 0.5 * (1.0 + tanh_out) + 0.5 * x * sech2 * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)
    return (local_grad * dout.float()).to(torch.bfloat16)


def synchronize_delete(*tensors: torch.Tensor | None) -> None:
    torch.cuda.synchronize()
    del tensors
    gc.collect()
    torch.cuda.empty_cache()


def bench_shape(shape: Shape, *, repeats: int, large_repeats: int) -> None:
    warmup = 1 if shape.n >= 8192 else 3
    iters = 1 if shape.n >= 8192 else 5
    repeats_for_shape = large_repeats if shape.n >= 8192 else repeats

    torch.manual_seed(1200 + shape.n + shape.k)
    a = torch.randn((shape.m, shape.k), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((shape.n, shape.k), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((shape.m, shape.n), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((shape.n,), device="cuda", dtype=torch.bfloat16) if shape.bias else None

    print(f"\n{shape.name:<12} M={shape.m} N={shape.n} K={shape.k} bias={1 if shape.bias else 0} gelu={1 if shape.gelu else 0}")

    if shape.gelu:
        def fwd_gelu() -> torch.Tensor:
            pre = a @ w.T
            if bias is not None:
                pre = pre + bias
            return F.gelu(pre, approximate="tanh")

        us = event_time_us(fwd_gelu, warmup=warmup, repeats=repeats_for_shape, iters=iters)
        print(f"  fwd+GeLU Torch {us:9.2f} us")
    else:
        def fwd() -> torch.Tensor:
            result = a @ w.T
            if bias is not None:
                result = result + bias
            return result

        us = event_time_us(fwd, warmup=warmup, repeats=repeats_for_shape, iters=iters)
        print(f"  fwd      Torch {us:9.2f} us")

    def dinp() -> torch.Tensor:
        return out @ w

    us = event_time_us(dinp, warmup=warmup, repeats=repeats_for_shape, iters=iters)
    print(f"  dInp   Torch {us:9.2f} us")

    if shape.dgelu:
        pre_dgelu = torch.randn((shape.m, shape.k), device="cuda", dtype=torch.bfloat16)

        def dinp_dgelu() -> torch.Tensor:
            return gelu_backward_tanh(out @ w, pre_dgelu)

        us = event_time_us(dinp_dgelu, warmup=warmup, repeats=repeats_for_shape, iters=iters)
        print(f"  dInp+dGeLU Torch {us:9.2f} us")
        synchronize_delete(pre_dgelu)

    def dweight() -> torch.Tensor:
        return out.T @ a

    us = event_time_us(dweight, warmup=warmup, repeats=repeats_for_shape, iters=iters)
    print(f"  dW     Torch {us:9.2f} us")

    dw = torch.empty((shape.n, shape.k), device="cuda", dtype=torch.bfloat16)

    def dweight_accum() -> torch.Tensor:
        return torch.addmm(dw, out.T, a, beta=1.0, alpha=1.0)

    us = event_time_us(dweight_accum, warmup=warmup, repeats=repeats_for_shape, iters=iters)
    print(f"  dW+accum Torch {us:9.2f} us")

    synchronize_delete(a, w, out, bias, dw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Torch SM120 GPT-2 matmul benchmark prototype")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--large-repeats", type=int, default=3)
    parser.add_argument("--shape", choices=[shape.name for shape in SHAPES], action="append")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("PyTorch CUDA context is not initialized in this process; rerun inside the target benchmark context.", file=sys.stderr)
        return 2

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"Torch matmul device: {device_name}; capability=sm_{capability[0]}{capability[1]}")
    selected = set(args.shape) if args.shape else {shape.name for shape in SHAPES}
    for shape in SHAPES:
        if shape.name in selected:
            bench_shape(shape, repeats=args.repeats, large_repeats=args.large_repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
