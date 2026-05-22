#!/usr/bin/env python3
"""Triton SM120 GEMM feasibility probe for GPT-2 shapes."""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


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


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    pre_ptr,
    accum_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    k_total: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_pm: tl.constexpr,
    stride_pn: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    has_bias: tl.constexpr,
    gelu: tl.constexpr,
    dgelu: tl.constexpr,
    accumulate: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)
    offs_m_i64 = offs_m.to(tl.int64)
    offs_n_i64 = offs_n.to(tl.int64)

    acc = tl.zeros((block_m, block_n), tl.float32)
    for start_k in range(0, k_total, block_k):
        k_idxs = start_k + offs_k
        k_i64 = k_idxs.to(tl.int64)
        a = tl.load(
            a_ptr + offs_m_i64[:, None] * stride_am + k_i64[None, :] * stride_ak,
            mask=(offs_m[:, None] < m) & (k_idxs[None, :] < k_total),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k_i64[:, None] * stride_bk + offs_n_i64[None, :] * stride_bn,
            mask=(k_idxs[:, None] < k_total) & (offs_n[None, :] < n),
            other=0.0,
        )
        acc += tl.dot(a, b).to(tl.float32)

    if has_bias:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < n, other=0.0).to(tl.float32)
        acc += bias[None, :]

    if gelu:
        x = acc
        x3 = x * x * x
        inner = 0.7978845608028654 * (x + 0.044715 * x3)
        tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
        acc = 0.5 * x * (1.0 + tanh_inner)

    if dgelu:
        x = tl.load(
            pre_ptr + offs_m_i64[:, None] * stride_pm + offs_n_i64[None, :] * stride_pn,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
            other=0.0,
        ).to(tl.float32)
        tanh_arg = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        tanh_out = 2.0 * tl.sigmoid(2.0 * tanh_arg) - 1.0
        sech2 = 1.0 - tanh_out * tanh_out
        local_grad = 0.5 * (1.0 + tanh_out) + 0.5 * x * sech2 * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)
        acc *= local_grad

    if accumulate:
        old = tl.load(
            accum_ptr + offs_m_i64[:, None] * stride_xm + offs_n_i64[None, :] * stride_xn,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
            other=0.0,
        ).to(tl.float32)
        acc += old

    tl.store(
        c_ptr + offs_m_i64[:, None] * stride_cm + offs_n_i64[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
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


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x, approximate="tanh")


def gelu_backward_tanh(dout: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    x = inp.float()
    tanh_arg = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    tanh_out = torch.tanh(tanh_arg)
    sech2 = 1.0 - tanh_out * tanh_out
    local_grad = 0.5 * (1.0 + tanh_out) + 0.5 * x * sech2 * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)
    return (local_grad * dout.float()).to(torch.bfloat16)


def launch_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    pre: torch.Tensor | None = None,
    accum: torch.Tensor | None = None,
    accumulate: bool = False,
    gelu: bool = False,
    dgelu: bool = False,
    block_m: int,
    block_n: int,
    block_k: int,
) -> torch.Tensor:
    m, k_total = a.shape
    k_b, n = b.shape
    if k_b != k_total or c.shape != (m, n):
        raise ValueError("incompatible matmul shapes")
    if bias is None:
        bias = c
    if pre is None:
        pre = c
    if accum is None:
        accum = c
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _matmul_kernel[grid](
        a,
        b,
        c,
        bias,
        pre,
        accum,
        m,
        n,
        k_total,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        pre.stride(0),
        pre.stride(1),
        accum.stride(0),
        accum.stride(1),
        bias is not None and bias is not c,
        gelu,
        dgelu,
        accumulate,
        block_m,
        block_n,
        block_k,
        num_warps=4,
    )
    return c


def sample_diff(result: torch.Tensor, reference: torch.Tensor, *, rows: int, cols: int) -> tuple[float, float]:
    row_count = min(rows, result.shape[0])
    col_count = min(cols, result.shape[1])
    ref = reference[:row_count, :col_count].float()
    diff = (result[:row_count, :col_count].float() - ref).abs().max().item()
    ref_abs = ref.abs().max().item()
    return diff, ref_abs


def checked_timing(
    label: str,
    fn,
    ref_fn,
    *,
    repeats: int,
    warmup: int,
    iters: int,
    rows: int,
    cols: int,
    output_tol: float,
    output_rtol: float,
) -> None:
    result = fn()
    torch.cuda.synchronize()
    ref = ref_fn()
    diff, ref_abs = sample_diff(result, ref, rows=rows, cols=cols)
    del ref
    rel = diff / max(ref_abs, 1.0)
    if diff > output_tol and rel > output_rtol:
        raise AssertionError(f"{label} parity failed: diff={diff:.6f} rel={rel:.6f}")
    us = event_time_us(fn, warmup=warmup, repeats=repeats, iters=iters)
    print(f"  {label:<10} Triton {us:9.2f} us (diff={diff:.6f}, rel={rel:.6f})")


def synchronize_delete(*tensors: torch.Tensor | None) -> None:
    torch.cuda.synchronize()
    del tensors
    gc.collect()
    torch.cuda.empty_cache()


def bench_shape(
    shape: Shape,
    *,
    ops: set[str],
    repeats: int,
    large_repeats: int,
    block_m: int,
    block_n: int,
    block_k: int,
    output_tol: float,
    output_rtol: float,
    parity_rows: int,
    parity_cols: int,
) -> None:
    torch.manual_seed(2200 + shape.n + shape.k)
    a = torch.randn((shape.m, shape.k), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((shape.n, shape.k), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((shape.m, shape.n), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((shape.n,), device="cuda", dtype=torch.bfloat16) if shape.bias else None
    w_t = w.T

    print(f"\n{shape.name:<12} M={shape.m} N={shape.n} K={shape.k} bias={1 if shape.bias else 0} gelu={1 if shape.gelu else 0}")
    repeats_for_shape = large_repeats if shape.n >= 8192 else repeats
    warmup = 1 if shape.n >= 8192 else 2
    iters = 1 if shape.n >= 8192 else 5

    if shape.gelu and "fwd+GeLU" in ops:
        def run_fwd_gelu() -> torch.Tensor:
            return launch_matmul(a, w_t, out, bias=bias, gelu=True, block_m=block_m, block_n=block_n, block_k=block_k)

        def ref_fwd_gelu() -> torch.Tensor:
            ref = a[:parity_rows].float() @ w[:parity_cols].T.float()
            if bias is not None:
                ref += bias[:parity_cols].float()
            return gelu_tanh(ref).to(torch.bfloat16)

        checked_timing("fwd+GeLU", run_fwd_gelu, ref_fwd_gelu, repeats=repeats_for_shape, warmup=warmup, iters=iters, rows=parity_rows, cols=parity_cols, output_tol=output_tol, output_rtol=output_rtol)
    elif not shape.gelu and "fwd" in ops:
        def run_fwd() -> torch.Tensor:
            return launch_matmul(a, w_t, out, bias=bias, block_m=block_m, block_n=block_n, block_k=block_k)

        def ref_fwd() -> torch.Tensor:
            ref = a[:parity_rows].float() @ w[:parity_cols].T.float()
            if bias is not None:
                ref += bias[:parity_cols].float()
            return ref.to(torch.bfloat16)

        checked_timing("fwd", run_fwd, ref_fwd, repeats=repeats_for_shape, warmup=warmup, iters=iters, rows=parity_rows, cols=parity_cols, output_tol=output_tol, output_rtol=output_rtol)

    if "dInp" in ops:
        dinp_out = torch.empty((shape.m, shape.k), device="cuda", dtype=torch.bfloat16)

        def run_dinp() -> torch.Tensor:
            return launch_matmul(out, w, dinp_out, block_m=block_m, block_n=block_n, block_k=block_k)

        def ref_dinp() -> torch.Tensor:
            return (out[:parity_rows].float() @ w[:, :parity_cols].float()).to(torch.bfloat16)

        checked_timing("dInp", run_dinp, ref_dinp, repeats=repeats_for_shape, warmup=warmup, iters=iters, rows=parity_rows, cols=parity_cols, output_tol=output_tol, output_rtol=output_rtol)
        synchronize_delete(dinp_out)

    if shape.dgelu and "dInp+dGeLU" in ops:
        dinp_dgelu_out = torch.empty((shape.m, shape.k), device="cuda", dtype=torch.bfloat16)
        pre_dgelu = torch.randn((shape.m, shape.k), device="cuda", dtype=torch.bfloat16)

        def run_dinp_dgelu() -> torch.Tensor:
            return launch_matmul(out, w, dinp_dgelu_out, pre=pre_dgelu, dgelu=True, block_m=block_m, block_n=block_n, block_k=block_k)

        def ref_dinp_dgelu() -> torch.Tensor:
            mat = (out[:parity_rows].float() @ w[:, :parity_cols].float()).to(torch.bfloat16)
            return gelu_backward_tanh(mat, pre_dgelu[:parity_rows, :parity_cols])

        checked_timing("dInp+dGeLU", run_dinp_dgelu, ref_dinp_dgelu, repeats=repeats_for_shape, warmup=warmup, iters=iters, rows=parity_rows, cols=parity_cols, output_tol=output_tol, output_rtol=output_rtol)
        synchronize_delete(dinp_dgelu_out, pre_dgelu)

    if "dW" in ops:
        dw_out = torch.empty((shape.n, shape.k), device="cuda", dtype=torch.bfloat16)
        out_t = out.T

        def run_dw() -> torch.Tensor:
            return launch_matmul(out_t, a, dw_out, block_m=block_m, block_n=block_n, block_k=block_k)

        def ref_dw() -> torch.Tensor:
            return (out[:, :parity_rows].T.float() @ a[:, :parity_cols].float()).to(torch.bfloat16)

        checked_timing("dW", run_dw, ref_dw, repeats=repeats_for_shape, warmup=warmup, iters=iters, rows=parity_rows, cols=parity_cols, output_tol=output_tol, output_rtol=output_rtol)
        synchronize_delete(dw_out)

    if "dW+accum" in ops:
        dw_accum = torch.randn((shape.n, shape.k), device="cuda", dtype=torch.bfloat16)
        out_t = out.T

        def run_dw_accum() -> torch.Tensor:
            return launch_matmul(out_t, a, dw_accum, accum=dw_accum, accumulate=True, block_m=block_m, block_n=block_n, block_k=block_k)

        dw_base = dw_accum.clone()

        def ref_dw_accum() -> torch.Tensor:
            return (dw_base[:parity_rows, :parity_cols].float() + out[:, :parity_rows].T.float() @ a[:, :parity_cols].float()).to(torch.bfloat16)

        checked_timing("dW+accum", run_dw_accum, ref_dw_accum, repeats=repeats_for_shape, warmup=warmup, iters=iters, rows=parity_rows, cols=parity_cols, output_tol=output_tol, output_rtol=output_rtol)
        synchronize_delete(dw_accum, dw_base)

    synchronize_delete(a, w, out, bias)


def main() -> int:
    parser = argparse.ArgumentParser(description="Triton SM120 GPT-2 matmul benchmark prototype")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--large-repeats", type=int, default=2)
    parser.add_argument("--shape", choices=[shape.name for shape in SHAPES], action="append")
    parser.add_argument("--op", choices=["fwd", "fwd+GeLU", "dInp", "dInp+dGeLU", "dW", "dW+accum"], action="append")
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--parity-rows", type=int, default=16)
    parser.add_argument("--parity-cols", type=int, default=16)
    parser.add_argument("--output-tol", type=float, default=128.0)
    parser.add_argument("--output-rtol", type=float, default=0.02)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("PyTorch CUDA context is not initialized in this process; rerun inside the target benchmark context.", file=sys.stderr)
        return 2
    for value, name in ((args.block_m, "--block-m"), (args.block_n, "--block-n"), (args.block_k, "--block-k")):
        if value <= 0 or value & (value - 1):
            raise ValueError(f"{name} must be a positive power of two")

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"Triton matmul device: {device_name}; capability=sm_{capability[0]}{capability[1]}")
    selected_shapes = set(args.shape) if args.shape else {shape.name for shape in SHAPES}
    selected_ops = set(args.op) if args.op else {"fwd", "fwd+GeLU", "dInp", "dInp+dGeLU", "dW", "dW+accum"}
    for shape in SHAPES:
        if shape.name in selected_shapes:
            bench_shape(
                shape,
                ops=selected_ops,
                repeats=args.repeats,
                large_repeats=args.large_repeats,
                block_m=args.block_m,
                block_n=args.block_n,
                block_k=args.block_k,
                output_tol=args.output_tol,
                output_rtol=args.output_rtol,
                parity_rows=args.parity_rows,
                parity_cols=args.parity_cols,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
