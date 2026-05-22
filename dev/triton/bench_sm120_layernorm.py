#!/usr/bin/env python3
"""Triton SM120 LayerNorm parity and timing prototype."""

from __future__ import annotations

import argparse
import statistics
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_residual_layernorm_forward_kernel(
    x_ptr,
    skip_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    y_ptr,
    mean_ptr,
    rstd_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols
    base = row * n_cols + offsets

    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    skip = tl.load(skip_ptr + base, mask=mask, other=0.0).to(tl.float32)
    residual = x + skip
    tl.store(residual_ptr + base, residual, mask=mask)

    # Match the CUDA fused path: the residual stream is materialized as BF16
    # and that rounded value feeds the LayerNorm statistics.
    residual = tl.load(residual_ptr + base, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(residual, axis=0) / n_cols
    centered = tl.where(mask, residual - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    rstd = tl.rsqrt(variance + eps)

    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = centered * rstd * weight + bias

    tl.store(y_ptr + base, y, mask=mask)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _layernorm_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    mean_ptr,
    rstd_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols
    base = row * n_cols + offsets

    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    rstd = tl.rsqrt(variance + eps)

    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = centered * rstd * weight + bias

    tl.store(y_ptr + base, y, mask=mask)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _layernorm_backward_dinput_kernel(
    dout_ptr,
    x_ptr,
    weight_ptr,
    mean_ptr,
    rstd_ptr,
    dinp_ptr,
    n_cols: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols
    base = row * n_cols + offsets

    dout = tl.load(dout_ptr + base, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)

    norm = (x - mean) * rstd
    dnorm = dout * weight
    dnorm_mean = tl.sum(tl.where(mask, dnorm, 0.0), axis=0) / n_cols
    dnorm_norm_mean = tl.sum(tl.where(mask, dnorm * norm, 0.0), axis=0) / n_cols
    dinp = (dnorm - dnorm_mean - norm * dnorm_norm_mean) * rstd

    tl.store(dinp_ptr + base, dinp, mask=mask)


@triton.jit
def _layernorm_backward_grad_atomic_kernel(
    dout_ptr,
    x_ptr,
    mean_ptr,
    rstd_ptr,
    dweight_ptr,
    dbias_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    block_rows: tl.constexpr,
    block_cols: tl.constexpr,
):
    col_block = tl.program_id(0)
    row_block = tl.program_id(1)
    rows = row_block * block_rows + tl.arange(0, block_rows)
    cols = col_block * block_cols + tl.arange(0, block_cols)
    row_col_offsets = rows[:, None] * n_cols + cols[None, :]
    mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)

    x = tl.load(x_ptr + row_col_offsets, mask=mask, other=0.0).to(tl.float32)
    dout = tl.load(dout_ptr + row_col_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + rows, mask=rows < n_rows, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows, mask=rows < n_rows, other=1.0).to(tl.float32)
    norm = (x - mean[:, None]) * rstd[:, None]

    partial_dbias = tl.sum(tl.where(mask, dout, 0.0), axis=0)
    partial_dweight = tl.sum(tl.where(mask, dout * norm, 0.0), axis=0)
    tl.atomic_add(dbias_ptr + cols, partial_dbias, sem="relaxed", mask=cols < n_cols)
    tl.atomic_add(dweight_ptr + cols, partial_dweight, sem="relaxed", mask=cols < n_cols)


def _ceil_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _event_time_ms(fn, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _run_layernorm(x, weight, bias, eps: float):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    mean = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    block_size = _ceil_power_of_2(n_cols)
    _layernorm_forward_kernel[(n_rows,)](
        x,
        weight,
        bias,
        y,
        mean,
        rstd,
        n_cols,
        eps,
        block_size,
        num_warps=8,
    )
    return y, mean, rstd


def _run_layernorm_backward_dinput(dout, x, weight, mean, rstd):
    n_rows, n_cols = x.shape
    dinp = torch.empty_like(x)
    block_size = _ceil_power_of_2(n_cols)
    _layernorm_backward_dinput_kernel[(n_rows,)](
        dout,
        x,
        weight,
        mean,
        rstd,
        dinp,
        n_cols,
        block_size,
        num_warps=8,
    )
    return dinp


def _run_layernorm_backward_grad_atomic(dout, x, mean, rstd, dweight, dbias, *, block_rows: int, block_cols: int):
    n_rows, n_cols = x.shape
    grid = (triton.cdiv(n_cols, block_cols), triton.cdiv(n_rows, block_rows))
    _layernorm_backward_grad_atomic_kernel[grid](
        dout,
        x,
        mean,
        rstd,
        dweight,
        dbias,
        n_rows,
        n_cols,
        block_rows,
        block_cols,
        num_warps=4,
    )


def _run_layernorm_backward_atomic_fp32(dout, x, weight, mean, rstd, *, block_rows: int, block_cols: int):
    dinp = _run_layernorm_backward_dinput(dout, x, weight, mean, rstd)
    dweight = torch.zeros((x.shape[1],), device=x.device, dtype=torch.float32)
    dbias = torch.zeros((x.shape[1],), device=x.device, dtype=torch.float32)
    _run_layernorm_backward_grad_atomic(dout, x, mean, rstd, dweight, dbias, block_rows=block_rows, block_cols=block_cols)
    return dinp, dweight, dbias


def _run_torch_layernorm_native(x, weight, bias, eps: float):
    return F.layer_norm(x, (x.shape[1],), weight, bias, eps=eps)


def _run_torch_native_layernorm_with_stats(x, weight, bias, eps: float):
    return torch.ops.aten.native_layer_norm(x, [x.shape[1]], weight, bias, eps)


def _run_torch_layernorm_backward_native(dout, x, weight, bias, mean, rstd):
    return torch.ops.aten.native_layer_norm_backward(
        dout, x, [x.shape[1]], mean, rstd, weight, bias, [True, True, True]
    )


def _run_torch_layernorm_backward_dinput_native(dout, x, weight, bias, mean, rstd):
    dinp, _, _ = torch.ops.aten.native_layer_norm_backward(
        dout, x, [x.shape[1]], mean, rstd, weight, bias, [True, False, False]
    )
    return dinp


def _run_torch_layernorm_backward_dinput_native_plus_grads(dout, x, weight, bias, mean, rstd):
    dinp = _run_torch_layernorm_backward_dinput_native(dout, x, weight, bias, mean, rstd)
    mean_2d = mean.reshape(-1, 1)
    rstd_2d = rstd.reshape(-1, 1)
    norm = (x.float() - mean_2d) * rstd_2d
    dweight = (dout.float() * norm).sum(dim=0).to(torch.bfloat16)
    dbias = dout.float().sum(dim=0).to(torch.bfloat16)
    return dinp, dweight, dbias


def _run_torch_layernorm_with_stats(x, weight, bias, eps: float):
    x_fp32 = x.float()
    mean = x_fp32.mean(dim=1)
    variance = ((x_fp32 - mean[:, None]) ** 2).mean(dim=1)
    rstd = torch.rsqrt(variance + eps)
    y = ((x_fp32 - mean[:, None]) * rstd[:, None] * weight.float() + bias.float()).to(torch.bfloat16)
    return y, mean, rstd


def _run_fused_residual_layernorm(x, skip, weight, bias, eps: float):
    n_rows, n_cols = x.shape
    residual = torch.empty_like(x)
    y = torch.empty_like(x)
    mean = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    block_size = _ceil_power_of_2(n_cols)
    _fused_residual_layernorm_forward_kernel[(n_rows,)](
        x,
        skip,
        weight,
        bias,
        residual,
        y,
        mean,
        rstd,
        n_cols,
        eps,
        block_size,
        num_warps=8,
    )
    return residual, y, mean, rstd


def _run_torch_fused_residual_native(x, skip, weight, bias, eps: float):
    residual = x + skip
    return residual, F.layer_norm(residual, (x.shape[1],), weight, bias, eps=eps)


def _run_torch_fused_residual_with_stats(x, skip, weight, bias, eps: float):
    residual = (x.float() + skip.float()).to(torch.bfloat16)
    y, mean, rstd = _run_torch_layernorm_with_stats(residual, weight, bias, eps)
    return residual, y, mean, rstd


def _case(
    n_rows: int,
    n_cols: int,
    *,
    repeats: int,
    warmup: int,
    eps: float,
    output_tol: float,
    stat_tol: float,
    grad_tol: float,
) -> None:
    torch.manual_seed(1200 + n_cols)
    x = torch.randn((n_rows, n_cols), device="cuda", dtype=torch.bfloat16)
    skip = torch.randn((n_rows, n_cols), device="cuda", dtype=torch.bfloat16)
    dout = torch.randn((n_rows, n_cols), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n_cols,), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((n_cols,), device="cuda", dtype=torch.bfloat16)

    y, mean, rstd = _run_layernorm(x, weight, bias, eps)
    ref_mean = x.float().mean(dim=1)
    ref_var = ((x.float() - ref_mean[:, None]) ** 2).mean(dim=1)
    ref_rstd = torch.rsqrt(ref_var + eps)
    ref_y = ((x.float() - ref_mean[:, None]) * ref_rstd[:, None] * weight.float() + bias.float()).to(
        torch.bfloat16
    )

    torch.cuda.synchronize()
    y_diff = (y.float() - ref_y.float()).abs().max().item()
    mean_diff = (mean - ref_mean).abs().max().item()
    rstd_diff = (rstd - ref_rstd).abs().max().item()
    if y_diff > output_tol or mean_diff > stat_tol or rstd_diff > stat_tol:
        raise AssertionError(
            "Triton LayerNorm parity failed: "
            f"y_diff={y_diff:.6f}, mean_diff={mean_diff:.6f}, rstd_diff={rstd_diff:.6f}"
        )

    def launch() -> None:
        _run_layernorm(x, weight, bias, eps)

    ms = _event_time_ms(launch, warmup=warmup, repeats=repeats)
    print(
        "Triton LayerNorm Forward "
        f"(N={n_rows}, C={n_cols}): {ms * 1000.0:.3f} us "
        f"(y_diff={y_diff:.6f}, mean_diff={mean_diff:.6f}, rstd_diff={rstd_diff:.6f})"
    )

    torch_y = _run_torch_layernorm_native(x, weight, bias, eps)
    torch.cuda.synchronize()
    torch_y_diff = (torch_y.float() - ref_y.float()).abs().max().item()
    if torch_y_diff > output_tol:
        raise AssertionError(f"Torch native LayerNorm parity failed: y_diff={torch_y_diff:.6f}")

    def launch_torch_native() -> None:
        _run_torch_layernorm_native(x, weight, bias, eps)

    torch_native_ms = _event_time_ms(launch_torch_native, warmup=warmup, repeats=repeats)
    print(
        "Torch LayerNorm ForwardNative "
        f"(N={n_rows}, C={n_cols}): {torch_native_ms * 1000.0:.3f} us "
        f"(y_diff={torch_y_diff:.6f}; no saved mean/rstd)"
    )

    torch_stats_y, torch_mean, torch_rstd = _run_torch_layernorm_with_stats(x, weight, bias, eps)
    torch.cuda.synchronize()
    torch_stats_y_diff = (torch_stats_y.float() - ref_y.float()).abs().max().item()
    torch_mean_diff = (torch_mean - ref_mean).abs().max().item()
    torch_rstd_diff = (torch_rstd - ref_rstd).abs().max().item()
    if torch_stats_y_diff > output_tol or torch_mean_diff > stat_tol or torch_rstd_diff > stat_tol:
        raise AssertionError(
            "Torch stats LayerNorm parity failed: "
            f"y_diff={torch_stats_y_diff:.6f}, mean_diff={torch_mean_diff:.6f}, "
            f"rstd_diff={torch_rstd_diff:.6f}"
        )

    def launch_torch_stats() -> None:
        _run_torch_layernorm_with_stats(x, weight, bias, eps)

    torch_stats_ms = _event_time_ms(launch_torch_stats, warmup=warmup, repeats=repeats)
    print(
        "Torch LayerNorm ForwardWithStats "
        f"(N={n_rows}, C={n_cols}): {torch_stats_ms * 1000.0:.3f} us "
        f"(y_diff={torch_stats_y_diff:.6f}, mean_diff={torch_mean_diff:.6f}, "
        f"rstd_diff={torch_rstd_diff:.6f})"
    )

    norm = (x.float() - ref_mean[:, None]) * ref_rstd[:, None]
    dnorm = dout.float() * weight.float()
    ref_dinp = (
        (dnorm - dnorm.mean(dim=1)[:, None] - norm * (dnorm * norm).mean(dim=1)[:, None])
        * ref_rstd[:, None]
    ).to(torch.bfloat16)

    dinp = _run_layernorm_backward_dinput(dout, x, weight, mean, rstd)
    torch.cuda.synchronize()
    dinp_diff = (dinp.float() - ref_dinp.float()).abs().max().item()
    if dinp_diff > output_tol:
        raise AssertionError(f"Triton LayerNorm dInput parity failed: dinp_diff={dinp_diff:.6f}")

    def launch_backward_dinput() -> None:
        _run_layernorm_backward_dinput(dout, x, weight, mean, rstd)

    backward_dinput_ms = _event_time_ms(launch_backward_dinput, warmup=warmup, repeats=repeats)
    print(
        "Triton LayerNorm BackwardDInput "
        f"(N={n_rows}, C={n_cols}): {backward_dinput_ms * 1000.0:.3f} us "
        f"(dinp_diff={dinp_diff:.6f}; dweight/dbias not produced)"
    )

    _, torch_saved_mean, torch_saved_rstd = _run_torch_native_layernorm_with_stats(x, weight, bias, eps)
    torch_dinp_only = _run_torch_layernorm_backward_dinput_native(
        dout, x, weight, bias, torch_saved_mean, torch_saved_rstd
    )
    torch.cuda.synchronize()
    torch_dinp_only_diff = (torch_dinp_only.float() - ref_dinp.float()).abs().max().item()
    if torch_dinp_only_diff > output_tol:
        raise AssertionError(f"Torch native LayerNorm dInput parity failed: dinp_diff={torch_dinp_only_diff:.6f}")

    def launch_torch_backward_dinput() -> None:
        _run_torch_layernorm_backward_dinput_native(dout, x, weight, bias, torch_saved_mean, torch_saved_rstd)

    torch_backward_dinput_ms = _event_time_ms(launch_torch_backward_dinput, warmup=warmup, repeats=repeats)
    print(
        "Torch LayerNorm BackwardDInputNative "
        f"(N={n_rows}, C={n_cols}): {torch_backward_dinput_ms * 1000.0:.3f} us "
        f"(dinp_diff={torch_dinp_only_diff:.6f}; dweight/dbias not produced)"
    )

    torch_hybrid_dinp, torch_hybrid_dweight, torch_hybrid_dbias = (
        _run_torch_layernorm_backward_dinput_native_plus_grads(
            dout, x, weight, bias, torch_saved_mean, torch_saved_rstd
        )
    )
    torch.cuda.synchronize()
    ref_dweight = (dout.float() * norm).sum(dim=0)
    ref_dbias = dout.float().sum(dim=0)
    torch_hybrid_dinp_diff = (torch_hybrid_dinp.float() - ref_dinp.float()).abs().max().item()
    torch_hybrid_dweight_diff = (torch_hybrid_dweight.float() - ref_dweight).abs().max().item()
    torch_hybrid_dbias_diff = (torch_hybrid_dbias.float() - ref_dbias).abs().max().item()
    if torch_hybrid_dinp_diff > output_tol or torch_hybrid_dweight_diff > grad_tol or torch_hybrid_dbias_diff > grad_tol:
        raise AssertionError(
            "Torch hybrid LayerNorm backward parity failed: "
            f"dinp_diff={torch_hybrid_dinp_diff:.6f}, dweight_diff={torch_hybrid_dweight_diff:.6f}, "
            f"dbias_diff={torch_hybrid_dbias_diff:.6f}"
        )

    def launch_torch_backward_dinput_plus_grads() -> None:
        _run_torch_layernorm_backward_dinput_native_plus_grads(
            dout, x, weight, bias, torch_saved_mean, torch_saved_rstd
        )

    torch_backward_dinput_plus_grads_ms = _event_time_ms(
        launch_torch_backward_dinput_plus_grads, warmup=warmup, repeats=repeats
    )
    print(
        "Torch LayerNorm BackwardDInputNativePlusGrads "
        f"(N={n_rows}, C={n_cols}): {torch_backward_dinput_plus_grads_ms * 1000.0:.3f} us "
        f"(dinp_diff={torch_hybrid_dinp_diff:.6f}, dweight_diff={torch_hybrid_dweight_diff:.6f}, "
        f"dbias_diff={torch_hybrid_dbias_diff:.6f}; BF16 dweight/dbias)"
    )

    torch_dinp, torch_dweight, torch_dbias = _run_torch_layernorm_backward_native(
        dout, x, weight, bias, torch_saved_mean, torch_saved_rstd
    )
    torch.cuda.synchronize()
    torch_backward_dinp_diff = (torch_dinp.float() - ref_dinp.float()).abs().max().item()
    torch_dweight_diff = (torch_dweight.float() - ref_dweight).abs().max().item()
    torch_dbias_diff = (torch_dbias.float() - ref_dbias).abs().max().item()
    if torch_backward_dinp_diff > output_tol or torch_dweight_diff > grad_tol or torch_dbias_diff > grad_tol:
        raise AssertionError(
            "Torch native LayerNorm backward parity failed: "
            f"dinp_diff={torch_backward_dinp_diff:.6f}, dweight_diff={torch_dweight_diff:.6f}, "
            f"dbias_diff={torch_dbias_diff:.6f}"
        )

    def launch_torch_backward() -> None:
        _run_torch_layernorm_backward_native(dout, x, weight, bias, torch_saved_mean, torch_saved_rstd)

    torch_backward_ms = _event_time_ms(launch_torch_backward, warmup=warmup, repeats=repeats)
    print(
        "Torch LayerNorm BackwardNative "
        f"(N={n_rows}, C={n_cols}): {torch_backward_ms * 1000.0:.3f} us "
        f"(dinp_diff={torch_backward_dinp_diff:.6f}, dweight_diff={torch_dweight_diff:.6f}, "
        f"dbias_diff={torch_dbias_diff:.6f})"
    )

    triton_dinp, triton_dweight, triton_dbias = _run_layernorm_backward_atomic_fp32(
        dout,
        x,
        weight,
        mean,
        rstd,
        block_rows=8,
        block_cols=64,
    )
    torch.cuda.synchronize()
    triton_backward_dinp_diff = (triton_dinp.float() - ref_dinp.float()).abs().max().item()
    triton_dweight_diff = (triton_dweight - ref_dweight).abs().max().item()
    triton_dbias_diff = (triton_dbias - ref_dbias).abs().max().item()
    if triton_backward_dinp_diff > output_tol or triton_dweight_diff > grad_tol or triton_dbias_diff > grad_tol:
        raise AssertionError(
            "Triton atomic LayerNorm backward parity failed: "
            f"dinp_diff={triton_backward_dinp_diff:.6f}, dweight_diff={triton_dweight_diff:.6f}, "
            f"dbias_diff={triton_dbias_diff:.6f}"
        )

    def launch_triton_backward_atomic() -> None:
        _run_layernorm_backward_atomic_fp32(dout, x, weight, mean, rstd, block_rows=8, block_cols=64)

    triton_backward_ms = _event_time_ms(launch_triton_backward_atomic, warmup=warmup, repeats=repeats)
    print(
        "Triton LayerNorm BackwardAtomicFP32 "
        f"(N={n_rows}, C={n_cols}): {triton_backward_ms * 1000.0:.3f} us "
        f"(dinp_diff={triton_backward_dinp_diff:.6f}, dweight_diff={triton_dweight_diff:.6f}, "
        f"dbias_diff={triton_dbias_diff:.6f}; FP32 dweight/dbias)"
    )

    residual, fused_y, fused_mean, fused_rstd = _run_fused_residual_layernorm(x, skip, weight, bias, eps)
    ref_residual = (x.float() + skip.float()).to(torch.bfloat16)
    ref_fused_mean = ref_residual.float().mean(dim=1)
    ref_fused_var = ((ref_residual.float() - ref_fused_mean[:, None]) ** 2).mean(dim=1)
    ref_fused_rstd = torch.rsqrt(ref_fused_var + eps)
    ref_fused_y = (
        (ref_residual.float() - ref_fused_mean[:, None])
        * ref_fused_rstd[:, None]
        * weight.float()
        + bias.float()
    ).to(torch.bfloat16)

    torch.cuda.synchronize()
    residual_diff = (residual.float() - ref_residual.float()).abs().max().item()
    fused_y_diff = (fused_y.float() - ref_fused_y.float()).abs().max().item()
    fused_mean_diff = (fused_mean - ref_fused_mean).abs().max().item()
    fused_rstd_diff = (fused_rstd - ref_fused_rstd).abs().max().item()
    if (
        residual_diff > output_tol
        or fused_y_diff > output_tol
        or fused_mean_diff > stat_tol
        or fused_rstd_diff > stat_tol
    ):
        raise AssertionError(
            "Triton fused LayerNorm parity failed: "
            f"residual_diff={residual_diff:.6f}, y_diff={fused_y_diff:.6f}, "
            f"mean_diff={fused_mean_diff:.6f}, rstd_diff={fused_rstd_diff:.6f}"
        )

    def launch_fused() -> None:
        _run_fused_residual_layernorm(x, skip, weight, bias, eps)

    fused_ms = _event_time_ms(launch_fused, warmup=warmup, repeats=repeats)
    print(
        "Triton LayerNorm FusedResidualForward "
        f"(N={n_rows}, C={n_cols}): {fused_ms * 1000.0:.3f} us "
        f"(residual_diff={residual_diff:.6f}, y_diff={fused_y_diff:.6f}, "
        f"mean_diff={fused_mean_diff:.6f}, rstd_diff={fused_rstd_diff:.6f})"
    )

    torch_residual, torch_fused_y = _run_torch_fused_residual_native(x, skip, weight, bias, eps)
    torch.cuda.synchronize()
    torch_residual_diff = (torch_residual.float() - ref_residual.float()).abs().max().item()
    torch_fused_y_diff = (torch_fused_y.float() - ref_fused_y.float()).abs().max().item()
    if torch_residual_diff > output_tol or torch_fused_y_diff > output_tol:
        raise AssertionError(
            "Torch native fused LayerNorm parity failed: "
            f"residual_diff={torch_residual_diff:.6f}, y_diff={torch_fused_y_diff:.6f}"
        )

    def launch_torch_fused_native() -> None:
        _run_torch_fused_residual_native(x, skip, weight, bias, eps)

    torch_fused_native_ms = _event_time_ms(launch_torch_fused_native, warmup=warmup, repeats=repeats)
    print(
        "Torch LayerNorm FusedResidualForwardNative "
        f"(N={n_rows}, C={n_cols}): {torch_fused_native_ms * 1000.0:.3f} us "
        f"(residual_diff={torch_residual_diff:.6f}, y_diff={torch_fused_y_diff:.6f}; "
        "no saved mean/rstd)"
    )

    torch_residual_stats, torch_fused_stats_y, torch_fused_mean, torch_fused_rstd = (
        _run_torch_fused_residual_with_stats(x, skip, weight, bias, eps)
    )
    torch.cuda.synchronize()
    torch_residual_stats_diff = (torch_residual_stats.float() - ref_residual.float()).abs().max().item()
    torch_fused_stats_y_diff = (torch_fused_stats_y.float() - ref_fused_y.float()).abs().max().item()
    torch_fused_mean_diff = (torch_fused_mean - ref_fused_mean).abs().max().item()
    torch_fused_rstd_diff = (torch_fused_rstd - ref_fused_rstd).abs().max().item()
    if (
        torch_residual_stats_diff > output_tol
        or torch_fused_stats_y_diff > output_tol
        or torch_fused_mean_diff > stat_tol
        or torch_fused_rstd_diff > stat_tol
    ):
        raise AssertionError(
            "Torch stats fused LayerNorm parity failed: "
            f"residual_diff={torch_residual_stats_diff:.6f}, "
            f"y_diff={torch_fused_stats_y_diff:.6f}, "
            f"mean_diff={torch_fused_mean_diff:.6f}, rstd_diff={torch_fused_rstd_diff:.6f}"
        )

    def launch_torch_fused_stats() -> None:
        _run_torch_fused_residual_with_stats(x, skip, weight, bias, eps)

    torch_fused_stats_ms = _event_time_ms(launch_torch_fused_stats, warmup=warmup, repeats=repeats)
    print(
        "Torch LayerNorm FusedResidualForwardWithStats "
        f"(N={n_rows}, C={n_cols}): {torch_fused_stats_ms * 1000.0:.3f} us "
        f"(residual_diff={torch_residual_stats_diff:.6f}, "
        f"y_diff={torch_fused_stats_y_diff:.6f}, mean_diff={torch_fused_mean_diff:.6f}, "
        f"rstd_diff={torch_fused_rstd_diff:.6f})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Triton SM120 LayerNorm forward parity and timing prototype")
    parser.add_argument("--rows", type=int, default=65536)
    parser.add_argument("--cols", type=int, nargs="+", default=[768, 3072])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--output-tol", type=float, default=0.08)
    parser.add_argument("--stat-tol", type=float, default=0.05)
    parser.add_argument("--grad-tol", type=float, default=16.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("PyTorch CUDA context is not initialized in this process; rerun inside the target benchmark context.", file=sys.stderr)
        return 2

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"Triton LayerNorm device: {device_name}; capability=sm_{capability[0]}{capability[1]}")
    for n_cols in args.cols:
        _case(
            args.rows,
            n_cols,
            repeats=args.repeats,
            warmup=args.warmup,
            eps=args.eps,
            output_tol=args.output_tol,
            stat_tol=args.stat_tol,
            grad_tol=args.grad_tol,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
