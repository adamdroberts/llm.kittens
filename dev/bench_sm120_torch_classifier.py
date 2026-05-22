#!/usr/bin/env python3
"""Torch SM120 classifier timing prototype for GPT-2 padded-logits shape."""

from __future__ import annotations

import argparse
import gc
import statistics
import sys

import torch
import torch.nn.functional as F


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
    print(f"{name:<30} | {shape:<28} | {'Torch':<12} | {us:9.3f} us")


def print_unavailable(name: str, shape: str, reason: str) -> None:
    print(f"{name:<30} | {shape:<28} | {'Torch':<12} | unavailable: {reason}")


def release_cuda_cache() -> None:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cudaerrormemoryallocation" in text


def bench_classifier(
    *,
    repeats: int,
    warmup: int,
    batch: int,
    seq: int,
    vocab: int,
    padded_vocab: int,
) -> None:
    torch.manual_seed(120)
    bt = batch * seq
    logits = torch.randn((bt, padded_vocab), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    targets = torch.randint(0, vocab, (bt,), device="cuda", dtype=torch.long)
    dloss = torch.ones((bt,), device="cuda", dtype=torch.float32)
    shape = f"B={batch} T={seq} V={vocab} P={padded_vocab}"

    def active_logits() -> torch.Tensor:
        return logits[:, :vocab]

    def loss_only() -> torch.Tensor:
        return F.cross_entropy(active_logits(), targets, reduction="none")

    loss_us = event_time_us(loss_only, warmup=warmup, repeats=repeats, iters=1)
    print_result("fused_classifier_loss", shape, loss_us)
    release_cuda_cache()

    losses = loss_only()
    torch.cuda.synchronize()

    def dlogits() -> None:
        (grad,) = torch.autograd.grad(losses, logits, dloss, retain_graph=True)
        del grad

    try:
        dlogits_us = event_time_us(dlogits, warmup=warmup, repeats=repeats, iters=1)
    except (RuntimeError, torch.AcceleratorError) as exc:
        if not is_cuda_oom(exc):
            raise
        print_unavailable("fused_classifier", shape, "CUDA OOM at full GPT-2 padded-logits shape")
    else:
        print_result("fused_classifier", shape, dlogits_us)


def main() -> int:
    parser = argparse.ArgumentParser(description="Torch SM120 GPT-2 classifier benchmark prototype")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--vocab", type=int, default=50257)
    parser.add_argument("--padded-vocab", type=int, default=50304)
    args = parser.parse_args()

    if args.vocab > args.padded_vocab:
        raise ValueError("--vocab must be <= --padded-vocab")
    if not torch.cuda.is_available():
        print("PyTorch CUDA context is not initialized in this process; rerun inside the target benchmark context.", file=sys.stderr)
        return 2

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"Torch classifier device: {device_name}; capability=sm_{capability[0]}{capability[1]}")
    bench_classifier(
        repeats=args.repeats,
        warmup=args.warmup,
        batch=args.batch,
        seq=args.seq,
        vocab=args.vocab,
        padded_vocab=args.padded_vocab,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
