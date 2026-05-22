#!/usr/bin/env python3
"""Triton SM120 classifier loss and dlogits feasibility probe for GPT-2."""

from __future__ import annotations

import argparse
import gc
import math
import re
import statistics
import sys
from collections.abc import Callable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _partial_max_kernel(
    logits_ptr,
    partial_max_ptr,
    rows: tl.constexpr,
    vocab: tl.constexpr,
    padded_vocab: tl.constexpr,
    blocks_per_row: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    cols = block * block_n + tl.arange(0, block_n).to(tl.int64)
    mask = (row < rows) & (cols < vocab)
    vals = tl.load(logits_ptr + row * padded_vocab + cols, mask=mask, other=-float("inf")).to(tl.float32)
    tl.store(partial_max_ptr + row * blocks_per_row + block, tl.max(vals, axis=0), mask=row < rows)


@triton.jit
def _row_max_kernel(
    partial_max_ptr,
    row_max_ptr,
    rows: tl.constexpr,
    blocks_per_row: tl.constexpr,
    reduce_blocks: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, reduce_blocks).to(tl.int64)
    mask = (row < rows) & (offsets < blocks_per_row)
    vals = tl.load(partial_max_ptr + row * blocks_per_row + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    tl.store(row_max_ptr + row, tl.max(vals, axis=0), mask=row < rows)


@triton.jit
def _partial_sum_kernel(
    logits_ptr,
    row_max_ptr,
    partial_sum_ptr,
    rows: tl.constexpr,
    vocab: tl.constexpr,
    padded_vocab: tl.constexpr,
    blocks_per_row: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    cols = block * block_n + tl.arange(0, block_n).to(tl.int64)
    mask = (row < rows) & (cols < vocab)
    row_max = tl.load(row_max_ptr + row, mask=row < rows, other=0.0).to(tl.float32)
    vals = tl.load(logits_ptr + row * padded_vocab + cols, mask=mask, other=-float("inf")).to(tl.float32)
    partial = tl.sum(tl.exp(vals - row_max), axis=0)
    tl.store(partial_sum_ptr + row * blocks_per_row + block, partial, mask=row < rows)


@triton.jit
def _loss_kernel(
    logits_ptr,
    targets_ptr,
    row_max_ptr,
    partial_sum_ptr,
    losses_ptr,
    rows: tl.constexpr,
    padded_vocab: tl.constexpr,
    blocks_per_row: tl.constexpr,
    reduce_blocks: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, reduce_blocks).to(tl.int64)
    mask = (row < rows) & (offsets < blocks_per_row)
    sums = tl.load(partial_sum_ptr + row * blocks_per_row + offsets, mask=mask, other=0.0).to(tl.float32)
    denom = tl.sum(sums, axis=0)
    row_max = tl.load(row_max_ptr + row, mask=row < rows, other=0.0).to(tl.float32)
    target = tl.load(targets_ptr + row, mask=row < rows, other=0).to(tl.int64)
    target_logit = tl.load(logits_ptr + row * padded_vocab + target, mask=row < rows, other=0.0).to(tl.float32)
    tl.store(losses_ptr + row, tl.log(denom) + row_max - target_logit, mask=row < rows)


@triton.jit
def _dlogits_kernel(
    logits_ptr,
    targets_ptr,
    row_max_ptr,
    partial_sum_ptr,
    dloss: tl.constexpr,
    rows: tl.constexpr,
    vocab: tl.constexpr,
    padded_vocab: tl.constexpr,
    blocks_per_row: tl.constexpr,
    reduce_blocks: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    cols = block * block_n + tl.arange(0, block_n).to(tl.int64)
    reduce_offsets = tl.arange(0, reduce_blocks).to(tl.int64)
    reduce_mask = (row < rows) & (reduce_offsets < blocks_per_row)
    sums = tl.load(partial_sum_ptr + row * blocks_per_row + reduce_offsets, mask=reduce_mask, other=0.0).to(
        tl.float32
    )
    denom = tl.sum(sums, axis=0)
    row_max = tl.load(row_max_ptr + row, mask=row < rows, other=0.0).to(tl.float32)
    target = tl.load(targets_ptr + row, mask=row < rows, other=0).to(tl.int64)
    mask = (row < rows) & (cols < vocab)
    logits = tl.load(logits_ptr + row * padded_vocab + cols, mask=mask, other=-float("inf")).to(tl.float32)
    prob = tl.exp(logits - row_max) / denom
    indicator = cols == target
    dlogit = (prob - indicator.to(tl.float32)) * dloss
    tl.store(logits_ptr + row * padded_vocab + cols, dlogit, mask=mask)


def compact_error(exc: BaseException) -> str:
    text = re.sub(r"\x1b\[[0-9;]*m", "", " ".join(str(exc).split()))
    return text[:220] if text else exc.__class__.__name__


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cudaerrormemoryallocation" in text


def event_time_us(fn: Callable[[], object], *, warmup: int, repeats: int, iters: int) -> float:
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


def release_cuda_cache() -> None:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def launch_loss_only(
    logits: torch.Tensor,
    targets: torch.Tensor,
    losses: torch.Tensor,
    partial_max: torch.Tensor,
    row_max: torch.Tensor,
    partial_sum: torch.Tensor,
    *,
    rows: int,
    vocab: int,
    padded_vocab: int,
    blocks_per_row: int,
    reduce_blocks: int,
    block_n: int,
) -> None:
    grid_blocks = (rows, blocks_per_row)
    _partial_max_kernel[grid_blocks](
        logits,
        partial_max,
        rows,
        vocab,
        padded_vocab,
        blocks_per_row,
        block_n,
    )
    _row_max_kernel[(rows,)](partial_max, row_max, rows, blocks_per_row, reduce_blocks)
    _partial_sum_kernel[grid_blocks](
        logits,
        row_max,
        partial_sum,
        rows,
        vocab,
        padded_vocab,
        blocks_per_row,
        block_n,
    )
    _loss_kernel[(rows,)](
        logits,
        targets,
        row_max,
        partial_sum,
        losses,
        rows,
        padded_vocab,
        blocks_per_row,
        reduce_blocks,
    )


def launch_classifier_full(
    logits: torch.Tensor,
    targets: torch.Tensor,
    losses: torch.Tensor,
    partial_max: torch.Tensor,
    row_max: torch.Tensor,
    partial_sum: torch.Tensor,
    *,
    rows: int,
    vocab: int,
    padded_vocab: int,
    blocks_per_row: int,
    reduce_blocks: int,
    block_n: int,
    dloss: float,
) -> None:
    launch_loss_only(
        logits,
        targets,
        losses,
        partial_max,
        row_max,
        partial_sum,
        rows=rows,
        vocab=vocab,
        padded_vocab=padded_vocab,
        blocks_per_row=blocks_per_row,
        reduce_blocks=reduce_blocks,
        block_n=block_n,
    )
    _dlogits_kernel[(rows, triton.cdiv(vocab, block_n))](
        logits,
        targets,
        row_max,
        partial_sum,
        dloss,
        rows,
        vocab,
        padded_vocab,
        blocks_per_row,
        reduce_blocks,
        block_n,
    )


def bench_classifier_loss(
    *,
    repeats: int,
    warmup: int,
    batch: int,
    seq: int,
    vocab: int,
    padded_vocab: int,
    block_n: int,
    dloss: float,
    output_tol: float,
    dlogits_tol: float,
) -> None:
    rows = batch * seq
    shape = f"B={batch} T={seq} V={vocab} P={padded_vocab}"
    blocks_per_row = triton.cdiv(vocab, block_n)
    reduce_blocks = triton.next_power_of_2(blocks_per_row)
    torch.manual_seed(120)
    try:
        logits = torch.randn((rows, padded_vocab), device="cuda", dtype=torch.bfloat16)
        logits_full = logits.clone()
        targets = torch.randint(0, vocab, (rows,), device="cuda", dtype=torch.long)
        losses = torch.empty((rows,), device="cuda", dtype=torch.float32)
        losses_full = torch.empty((rows,), device="cuda", dtype=torch.float32)
        partial_max = torch.empty((rows, blocks_per_row), device="cuda", dtype=torch.float32)
        row_max = torch.empty((rows,), device="cuda", dtype=torch.float32)
        partial_sum = torch.empty((rows, blocks_per_row), device="cuda", dtype=torch.float32)
        partial_max_full = torch.empty((rows, blocks_per_row), device="cuda", dtype=torch.float32)
        row_max_full = torch.empty((rows,), device="cuda", dtype=torch.float32)
        partial_sum_full = torch.empty((rows, blocks_per_row), device="cuda", dtype=torch.float32)
    except (RuntimeError, torch.AcceleratorError) as exc:
        if not is_cuda_oom(exc):
            raise
        print_unavailable("fused_classifier_loss", shape, "CUDA OOM at full GPT-2 padded-logits shape")
        return

    def launch() -> None:
        launch_loss_only(
            logits,
            targets,
            losses,
            partial_max,
            row_max,
            partial_sum,
            rows=rows,
            vocab=vocab,
            padded_vocab=padded_vocab,
            blocks_per_row=blocks_per_row,
            reduce_blocks=reduce_blocks,
            block_n=block_n,
        )

    try:
        launch()
    except Exception as exc:
        print_unavailable("fused_classifier_loss", shape, compact_error(exc))
        return
    torch.cuda.synchronize()

    if rows * vocab <= 32 * 1024 * 1024:
        ref = F.cross_entropy(logits[:, :vocab].float(), targets, reduction="none")
        diff = (losses - ref).abs().max().item()
        if diff > output_tol:
            raise AssertionError(f"Triton classifier loss parity failed: diff={diff:.6f}")
        del ref
    else:
        smoke_rows = min(rows, 128)
        ref = F.cross_entropy(logits[:smoke_rows, :vocab].float(), targets[:smoke_rows], reduction="none")
        diff = (losses[:smoke_rows] - ref).abs().max().item()
        if diff > output_tol:
            raise AssertionError(f"Triton classifier loss prefix parity failed: diff={diff:.6f}")
        del ref

    us = event_time_us(launch, warmup=warmup, repeats=repeats, iters=1)
    print_result("fused_classifier_loss", shape, us)

    if rows * vocab <= 32 * 1024 * 1024:
        ref_logits = logits_full.detach().float().requires_grad_(True)
        ref_losses = F.cross_entropy(ref_logits[:, :vocab], targets, reduction="none")
        ref_losses.backward(torch.full((rows,), dloss, device="cuda", dtype=torch.float32))
        ref_loss = ref_losses.detach()
        ref_dlogits = ref_logits.grad[:, :vocab]
        del ref_logits, ref_losses
    else:
        smoke_rows = min(rows, 128)
        ref_logits = logits_full[:smoke_rows].detach().float().requires_grad_(True)
        ref_losses = F.cross_entropy(ref_logits[:, :vocab], targets[:smoke_rows], reduction="none")
        ref_losses.backward(torch.full((smoke_rows,), dloss, device="cuda", dtype=torch.float32))
        ref_loss = ref_losses.detach()
        ref_dlogits = ref_logits.grad[:, :vocab]
        del ref_logits, ref_losses

    def launch_full() -> None:
        launch_classifier_full(
            logits_full,
            targets,
            losses_full,
            partial_max_full,
            row_max_full,
            partial_sum_full,
            rows=rows,
            vocab=vocab,
            padded_vocab=padded_vocab,
            blocks_per_row=blocks_per_row,
            reduce_blocks=reduce_blocks,
            block_n=block_n,
            dloss=dloss,
        )

    try:
        launch_full()
    except Exception as exc:
        print_unavailable("fused_classifier", shape, compact_error(exc))
        release_cuda_cache()
        return
    torch.cuda.synchronize()

    if rows * vocab <= 32 * 1024 * 1024:
        full_loss_diff = (losses_full - ref_loss).abs().max().item()
        dlogits_diff = (logits_full[:, :vocab].float() - ref_dlogits).abs().max().item()
    else:
        smoke_rows = min(rows, 128)
        full_loss_diff = (losses_full[:smoke_rows] - ref_loss).abs().max().item()
        dlogits_diff = (logits_full[:smoke_rows, :vocab].float() - ref_dlogits).abs().max().item()
    if full_loss_diff > output_tol or dlogits_diff > dlogits_tol:
        raise AssertionError(
            f"Triton classifier full parity failed: loss_diff={full_loss_diff:.6f}, "
            f"dlogits_diff={dlogits_diff:.6f}"
        )

    full_us = event_time_us(launch_full, warmup=warmup, repeats=repeats, iters=1)
    print_result("fused_classifier", shape, full_us)
    release_cuda_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description="Triton SM120 GPT-2 classifier benchmark prototype")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--vocab", type=int, default=50257)
    parser.add_argument("--padded-vocab", type=int, default=50304)
    parser.add_argument("--block-n", type=int, default=1024)
    parser.add_argument("--dloss", type=float, default=1.0)
    parser.add_argument("--output-tol", type=float, default=0.02)
    parser.add_argument("--dlogits-tol", type=float, default=0.002)
    args = parser.parse_args()

    if args.vocab > args.padded_vocab:
        raise ValueError("--vocab must be <= --padded-vocab")
    if args.block_n <= 0 or args.block_n & (args.block_n - 1):
        raise ValueError("--block-n must be a positive power of two")
    if not torch.cuda.is_available():
        print("PyTorch CUDA context is not initialized in this process; rerun inside the target benchmark context.", file=sys.stderr)
        return 2

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"Triton classifier device: {device_name}; capability=sm_{capability[0]}{capability[1]}")
    bench_classifier_loss(
        repeats=args.repeats,
        warmup=args.warmup,
        batch=args.batch,
        seq=args.seq,
        vocab=args.vocab,
        padded_vocab=args.padded_vocab,
        block_n=args.block_n,
        dloss=args.dloss,
        output_tol=args.output_tol,
        dlogits_tol=args.dlogits_tol,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
