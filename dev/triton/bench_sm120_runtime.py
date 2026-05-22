#!/usr/bin/env python3
"""Triton SM120 runtime-family timing prototype for GPT-2 pointwise rows."""

from __future__ import annotations

import argparse
import statistics
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def _tanh_approx(x):
    return 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0


@triton.jit
def _bias_add_kernel(x_ptr, bias_ptr, y_ptr, n_elements: tl.constexpr, cols: tl.constexpr, block_size: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    col_offsets = offsets % cols
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offsets, x + bias, mask=mask)


@triton.jit
def _gelu_forward_kernel(x_ptr, y_ptr, n_elements: tl.constexpr, block_size: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x3 = x * x * x
    tanh_arg = 0.7978845608028654 * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + _tanh_approx(tanh_arg))
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _gelu_backward_kernel(dout_ptr, x_ptr, dinp_ptr, n_elements: tl.constexpr, block_size: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    dout = tl.load(dout_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = x * x
    tanh_arg = 0.7978845608028654 * (x + 0.044715 * x * x2)
    tanh_out = _tanh_approx(tanh_arg)
    sech2 = 1.0 - tanh_out * tanh_out
    local_grad = 0.5 * (1.0 + tanh_out) + 0.5 * x * sech2 * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x2)
    tl.store(dinp_ptr + offsets, local_grad * dout, mask=mask)


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
    print(f"{name:<30} | {shape:<28} | {'Triton':<12} | {us:9.3f} us")


def print_unavailable(name: str, shape: str, reason: str) -> None:
    print(f"{name:<30} | {shape:<28} | {'Triton':<12} | unavailable: {reason}")


def launch_grid(n_elements: int, block_size: int) -> tuple[int]:
    return (triton.cdiv(n_elements, block_size),)


def bench_bias_add(*, rows: int, cols: int, repeats: int, warmup: int, output_tol: float) -> None:
    n_elements = rows * cols
    block_size = 256
    torch.manual_seed(120 + cols)
    x = torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((cols,), device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)

    def launch() -> None:
        _bias_add_kernel[launch_grid(n_elements, block_size)](x, bias, y, n_elements, cols, block_size)

    launch()
    ref = x + bias
    torch.cuda.synchronize()
    diff = (y.float() - ref.float()).abs().max().item()
    if diff > output_tol:
        raise AssertionError(f"Triton bias_add parity failed for cols={cols}: diff={diff:.6f}")
    print_result(
        "bias_add",
        f"BT={rows} OC={cols}",
        event_time_us(launch, warmup=warmup, repeats=repeats, iters=20),
    )


def bench_gelu(*, rows: int, cols: int, repeats: int, warmup: int, output_tol: float) -> None:
    n_elements = rows * cols
    block_size = 256
    torch.manual_seed(240 + cols)
    x = torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16)
    dout = torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)
    dinp = torch.empty_like(x)

    def forward() -> None:
        _gelu_forward_kernel[launch_grid(n_elements, block_size)](x, y, n_elements, block_size)

    def backward() -> None:
        _gelu_backward_kernel[launch_grid(n_elements, block_size)](dout, x, dinp, n_elements, block_size)

    forward()
    backward()
    ref_y = torch.nn.functional.gelu(x, approximate="tanh")
    x_fp32 = x.float()
    tanh_arg = 0.7978845608028654 * (x_fp32 + 0.044715 * x_fp32 * x_fp32 * x_fp32)
    tanh_out = torch.tanh(tanh_arg)
    sech2 = 1.0 - tanh_out * tanh_out
    local_grad = 0.5 * (1.0 + tanh_out) + 0.5 * x_fp32 * sech2 * 0.7978845608028654 * (
        1.0 + 3.0 * 0.044715 * x_fp32 * x_fp32
    )
    ref_dinp = (local_grad * dout.float()).to(torch.bfloat16)
    torch.cuda.synchronize()
    y_diff = (y.float() - ref_y.float()).abs().max().item()
    dinp_diff = (dinp.float() - ref_dinp.float()).abs().max().item()
    if y_diff > output_tol or dinp_diff > output_tol:
        raise AssertionError(
            f"Triton GELU parity failed: y_diff={y_diff:.6f}, dinp_diff={dinp_diff:.6f}"
        )
    print_result(
        "gelu_forward",
        f"BT={rows} C={cols}",
        event_time_us(forward, warmup=warmup, repeats=repeats, iters=20),
    )
    print_result(
        "gelu_backward_inplace",
        f"BT={rows} C={cols}",
        event_time_us(backward, warmup=warmup, repeats=repeats, iters=20),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Triton SM120 GPT-2 runtime-family benchmark prototype")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rows", type=int, default=64 * 1024)
    parser.add_argument("--output-tol", type=float, default=0.05)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("PyTorch CUDA context is not initialized in this process; rerun inside the target benchmark context.", file=sys.stderr)
        return 2

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"Triton runtime device: {device_name}; capability=sm_{capability[0]}{capability[1]}")
    bench_bias_add(rows=args.rows, cols=768, repeats=args.repeats, warmup=args.warmup, output_tol=args.output_tol)
    bench_bias_add(rows=args.rows, cols=3072, repeats=args.repeats, warmup=args.warmup, output_tol=args.output_tol)
    bench_gelu(rows=args.rows, cols=3072, repeats=args.repeats, warmup=args.warmup, output_tol=args.output_tol)
    print_unavailable("bias_grad_reduce", "BT=65536 OC=768", "not implemented in this Triton runtime prototype")
    print_unavailable("bias_grad_reduce", "BT=65536 OC=2304", "not implemented in this Triton runtime prototype")
    print_unavailable("bias_grad_reduce", "BT=65536 OC=3072", "not implemented in this Triton runtime prototype")
    print_unavailable("adamw_update", "params=124475904 no-master", "not implemented in this Triton runtime prototype")
    print_unavailable("global_norm_squared", "params=124475904", "not implemented in this Triton runtime prototype")
    print_unavailable("encoder_forward", "B=64 T=1024 C=768", "not implemented in this Triton runtime prototype")
    print_unavailable("cuda_memset", "hidden_elems=50331648", "not implemented in this Triton runtime prototype")
    print_unavailable("cuda_memset", "grad_elems=124475904", "not implemented in this Triton runtime prototype")
    print_unavailable("cuda_memset", "logits_elems=3296722944", "not implemented in this Triton runtime prototype")
    print_unavailable("cuda_copy_d2d", "hidden_elems=50331648", "not implemented in this Triton runtime prototype")
    print_unavailable("cuda_copy_d2d", "logits_elems=3296722944", "not implemented in this Triton runtime prototype")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
