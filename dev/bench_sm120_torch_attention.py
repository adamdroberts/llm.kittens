#!/usr/bin/env python3
"""Torch SM120 attention timing prototype for the GPT-2 packed-QKV shape."""

from __future__ import annotations

import argparse
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


def bench_attention(*, repeats: int, warmup: int, batch: int, seq: int, channels: int, heads: int) -> None:
    head_size = channels // heads
    torch.manual_seed(120)
    q = torch.randn((batch, heads, seq, head_size), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn((batch, heads, seq, head_size), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn((batch, heads, seq, head_size), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dout = torch.randn((batch, heads, seq, head_size), device="cuda", dtype=torch.bfloat16)

    def forward() -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)

    fwd_us = event_time_us(forward, warmup=warmup, repeats=repeats, iters=20)
    print(
        f"Torch Attention Forward (B={batch}, T={seq}, C={channels}, NH={heads}, HS={head_size}): "
        f"{fwd_us:.3f} us"
    )

    out = forward()
    torch.cuda.synchronize()

    def backward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grads = torch.autograd.grad(out, (q, k, v), dout, retain_graph=True)
        return grads

    bwd_us = event_time_us(backward, warmup=warmup, repeats=repeats, iters=10)
    print(
        f"Torch Attention Backward (B={batch}, T={seq}, C={channels}, NH={heads}, HS={head_size}): "
        f"{bwd_us:.3f} us"
    )

    qkv = torch.randn(
        (batch, seq, 3, heads, head_size),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    out_btc = torch.empty((batch, seq, channels), device="cuda", dtype=torch.bfloat16)
    dout_btc = torch.randn((batch, seq, channels), device="cuda", dtype=torch.bfloat16)
    dout_heads = dout_btc.view(batch, seq, heads, head_size).permute(0, 2, 1, 3)

    def packed_forward() -> torch.Tensor:
        q_view = qkv[:, :, 0].permute(0, 2, 1, 3)
        k_view = qkv[:, :, 1].permute(0, 2, 1, 3)
        v_view = qkv[:, :, 2].permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q_view, k_view, v_view, dropout_p=0.0, is_causal=True)
        out_btc.copy_(attn.permute(0, 2, 1, 3).reshape(batch, seq, channels))
        return out_btc

    packed_fwd_us = event_time_us(packed_forward, warmup=warmup, repeats=repeats, iters=20)
    print(
        f"TorchPacked Attention Forward (B={batch}, T={seq}, C={channels}, NH={heads}, HS={head_size}): "
        f"{packed_fwd_us:.3f} us"
    )

    q_view = qkv[:, :, 0].permute(0, 2, 1, 3)
    k_view = qkv[:, :, 1].permute(0, 2, 1, 3)
    v_view = qkv[:, :, 2].permute(0, 2, 1, 3)
    packed_out = F.scaled_dot_product_attention(q_view, k_view, v_view, dropout_p=0.0, is_causal=True)
    torch.cuda.synchronize()

    def packed_backward() -> torch.Tensor:
        (dqkv,) = torch.autograd.grad(packed_out, qkv, dout_heads, retain_graph=True)
        return dqkv

    packed_bwd_us = event_time_us(packed_backward, warmup=warmup, repeats=repeats, iters=10)
    print(
        f"TorchPacked Attention Backward (B={batch}, T={seq}, C={channels}, NH={heads}, HS={head_size}): "
        f"{packed_bwd_us:.3f} us"
    )

    def materialized_qkv_views() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            qkv[:, :, 0].permute(0, 2, 1, 3).contiguous(),
            qkv[:, :, 1].permute(0, 2, 1, 3).contiguous(),
            qkv[:, :, 2].permute(0, 2, 1, 3).contiguous(),
        )

    def materialized_packed_forward() -> torch.Tensor:
        with torch.no_grad():
            q_mat, k_mat, v_mat = materialized_qkv_views()
            attn = F.scaled_dot_product_attention(q_mat, k_mat, v_mat, dropout_p=0.0, is_causal=True)
            out_btc.copy_(attn.permute(0, 2, 1, 3).reshape(batch, seq, channels))
            return out_btc

    materialized_packed_fwd_us = event_time_us(
        materialized_packed_forward, warmup=warmup, repeats=repeats, iters=20
    )
    print(
        f"TorchMaterializedPacked Attention Forward (B={batch}, T={seq}, C={channels}, NH={heads}, HS={head_size}): "
        f"{materialized_packed_fwd_us:.3f} us"
    )

    q_mat, k_mat, v_mat = materialized_qkv_views()
    materialized_packed_out = F.scaled_dot_product_attention(
        q_mat, k_mat, v_mat, dropout_p=0.0, is_causal=True
    )
    torch.cuda.synchronize()

    def materialized_packed_backward() -> torch.Tensor:
        (dqkv,) = torch.autograd.grad(materialized_packed_out, qkv, dout_heads, retain_graph=True)
        return dqkv

    materialized_packed_bwd_us = event_time_us(
        materialized_packed_backward, warmup=warmup, repeats=repeats, iters=10
    )
    print(
        f"TorchMaterializedPacked Attention Backward (B={batch}, T={seq}, C={channels}, NH={heads}, HS={head_size}): "
        f"{materialized_packed_bwd_us:.3f} us"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Torch SM120 GPT-2 attention benchmark prototype")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    args = parser.parse_args()

    if args.channels % args.heads != 0:
        raise ValueError("--channels must be divisible by --heads")
    if not torch.cuda.is_available():
        print("PyTorch CUDA context is not initialized in this process; rerun inside the target benchmark context.", file=sys.stderr)
        return 2

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"Torch attention device: {device_name}; capability=sm_{capability[0]}{capability[1]}")
    bench_attention(
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
