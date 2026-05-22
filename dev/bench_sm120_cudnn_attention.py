#!/usr/bin/env python3
"""cuDNN SDPA feasibility timing probe for the GPT-2 attention shape."""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from collections.abc import Callable

import torch
import torch.nn.functional as F


CUDNN_PACKED_BACKWARD_ROUTE = "cuDNNPacked Attention Backward route: saved-forward"


def compact_error(exc: BaseException) -> str:
    text = re.sub(r"\x1b\[[0-9;]*m", "", " ".join(str(exc).split()))
    if not text:
        return exc.__class__.__name__
    return text[:260]


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


def cudnn_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, scale: float):
    return torch.ops.aten._scaled_dot_product_cudnn_attention(
        q,
        k,
        v,
        None,
        True,
        0.0,
        True,
        False,
        scale=scale,
    )


def cudnn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    fwd,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.aten._scaled_dot_product_cudnn_attention_backward(
        dout,
        q,
        k,
        v,
        fwd[0],
        fwd[1],
        fwd[6],
        fwd[7],
        None,
        fwd[2],
        fwd[3],
        fwd[4],
        fwd[5],
        0.0,
        True,
        scale=scale,
    )


def compare_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor, *, scale: float) -> float:
    ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True, scale=scale)
    return (out.float() - ref.float()).abs().max().item()


def print_unavailable(stack: str, direction: str, batch: int, seq: int, channels: int, heads: int, reason: str) -> None:
    head_size = channels // heads
    print(
        f"{stack} Attention {direction} unavailable "
        f"(B={batch}, T={seq}, C={channels}, NH={heads}, HS={head_size}): {reason}"
    )


def bench_separated(*, repeats: int, warmup: int, batch: int, seq: int, channels: int, heads: int) -> None:
    head_size = channels // heads
    scale = 1.0 / math.sqrt(head_size)
    q = torch.randn((batch, heads, seq, head_size), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q)

    try:
        fwd = cudnn_forward(q, k, v, scale=scale)
        max_diff = compare_forward(q, k, v, fwd[0], scale=scale)
    except Exception as exc:
        reason = compact_error(exc)
        print_unavailable("cuDNN", "Forward", batch, seq, channels, heads, reason)
        print_unavailable("cuDNN", "Backward", batch, seq, channels, heads, "forward unavailable")
        return

    def forward() -> torch.Tensor:
        return cudnn_forward(q, k, v, scale=scale)[0]

    fwd_us = event_time_us(forward, warmup=warmup, repeats=repeats, iters=20)
    print(
        f"cuDNN Attention Forward (B={batch}, T={seq}, C={channels}, NH={heads}, HS={head_size}): "
        f"{fwd_us:.3f} us (max_diff={max_diff:.6f})"
    )

    try:
        cudnn_backward(dout, q, k, v, fwd, scale=scale)
    except Exception as exc:
        print_unavailable("cuDNN", "Backward", batch, seq, channels, heads, compact_error(exc))
        return

    def backward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return cudnn_backward(dout, q, k, v, fwd, scale=scale)

    bwd_us = event_time_us(backward, warmup=warmup, repeats=repeats, iters=10)
    print(f"cuDNN Attention Backward (B={batch}, T={seq}, C={channels}, NH={heads}, HS={head_size}): {bwd_us:.3f} us")


def bench_packed(*, repeats: int, warmup: int, batch: int, seq: int, channels: int, heads: int) -> None:
    head_size = channels // heads
    scale = 1.0 / math.sqrt(head_size)
    qkv = torch.randn((batch, seq, 3, heads, head_size), device="cuda", dtype=torch.bfloat16)
    out_btc = torch.empty((batch, seq, channels), device="cuda", dtype=torch.bfloat16)
    dout_btc = torch.randn_like(out_btc)
    dout = dout_btc.view(batch, seq, heads, head_size).permute(0, 2, 1, 3)
    dqkv = torch.empty_like(qkv)

    def qkv_views() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            qkv[:, :, 0].permute(0, 2, 1, 3),
            qkv[:, :, 1].permute(0, 2, 1, 3),
            qkv[:, :, 2].permute(0, 2, 1, 3),
        )

    def packed_forward() -> torch.Tensor:
        q, k, v = qkv_views()
        out = cudnn_forward(q, k, v, scale=scale)[0]
        out_btc.copy_(out.permute(0, 2, 1, 3).reshape(batch, seq, channels))
        return out_btc

    try:
        q, k, v = qkv_views()
        fwd = cudnn_forward(q, k, v, scale=scale)
        out_btc.copy_(fwd[0].permute(0, 2, 1, 3).reshape(batch, seq, channels))
    except Exception as exc:
        reason = compact_error(exc)
        print_unavailable("cuDNNPacked", "Forward", batch, seq, channels, heads, reason)
        print_unavailable("cuDNNPacked", "Backward", batch, seq, channels, heads, "forward unavailable")
        return

    packed_fwd_us = event_time_us(packed_forward, warmup=warmup, repeats=repeats, iters=20)
    print(
        f"cuDNNPacked Attention Forward (B={batch}, T={seq}, C={channels}, NH={heads}, HS={head_size}): "
        f"{packed_fwd_us:.3f} us"
    )

    # Match a trainer route: forward has already produced the cuDNN reserve
    # tensors, so backward timing should not include a fresh cuDNN forward.
    q, k, v = qkv_views()
    fwd = cudnn_forward(q, k, v, scale=scale)
    torch.cuda.synchronize()
    print(CUDNN_PACKED_BACKWARD_ROUTE)

    def packed_backward() -> torch.Tensor:
        dq, dk, dv = cudnn_backward(dout, q, k, v, fwd, scale=scale)
        dqkv[:, :, 0].copy_(dq.permute(0, 2, 1, 3))
        dqkv[:, :, 1].copy_(dk.permute(0, 2, 1, 3))
        dqkv[:, :, 2].copy_(dv.permute(0, 2, 1, 3))
        return dqkv

    try:
        packed_backward()
    except Exception as exc:
        print_unavailable("cuDNNPacked", "Backward", batch, seq, channels, heads, compact_error(exc))
        return

    packed_bwd_us = event_time_us(packed_backward, warmup=warmup, repeats=repeats, iters=10)
    print(
        f"cuDNNPacked Attention Backward (B={batch}, T={seq}, C={channels}, NH={heads}, HS={head_size}): "
        f"{packed_bwd_us:.3f} us"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="cuDNN SDPA benchmark prototype for GPT-2 attention")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    args = parser.parse_args()

    if args.channels % args.heads != 0:
        raise ValueError("--channels must be divisible by --heads")
    if not hasattr(torch.ops.aten, "_scaled_dot_product_cudnn_attention"):
        print("cuDNN Attention unavailable: PyTorch does not expose aten._scaled_dot_product_cudnn_attention")
        return 0
    if not torch.cuda.is_available():
        print("PyTorch CUDA context is not initialized in this process; rerun inside the target benchmark context.", file=sys.stderr)
        return 2

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"cuDNN attention device: {device_name}; capability=sm_{capability[0]}{capability[1]}")
    bench_separated(
        repeats=args.repeats,
        warmup=args.warmup,
        batch=args.batch,
        seq=args.seq,
        channels=args.channels,
        heads=args.heads,
    )
    bench_packed(
        repeats=args.repeats,
        warmup=args.warmup,
        batch=args.batch,
        seq=args.seq,
        channels=args.channels,
        heads=args.heads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
