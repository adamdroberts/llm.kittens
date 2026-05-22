#!/usr/bin/env python3
"""Triton SM120 attention forward feasibility probe for GPT-2."""

from __future__ import annotations

import argparse
import statistics
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _attention_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    seq: tl.constexpr,
    head_size: tl.constexpr,
    scale: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    rows = pid_m * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, head_size)
    q_offsets = ((pid_bh * seq + rows[:, None]) * head_size) + cols[None, :]
    q = tl.load(q_ptr + q_offsets, mask=rows[:, None] < seq, other=0.0)

    m_i = tl.full((block_m,), -float("inf"), tl.float32)
    l_i = tl.zeros((block_m,), tl.float32)
    acc = tl.zeros((block_m, head_size), tl.float32)

    for start_n in range(0, seq, block_n):
        keys = start_n + tl.arange(0, block_n)
        kv_offsets = ((pid_bh * seq + keys[:, None]) * head_size) + cols[None, :]
        k = tl.load(k_ptr + kv_offsets, mask=keys[:, None] < seq, other=0.0)
        v = tl.load(v_ptr + kv_offsets, mask=keys[:, None] < seq, other=0.0)
        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        causal_mask = keys[None, :] <= rows[:, None]
        valid_mask = (rows[:, None] < seq) & (keys[None, :] < seq)
        scores = tl.where(causal_mask & valid_mask, scores, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v).to(tl.float32)
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(out_ptr + q_offsets, acc, mask=rows[:, None] < seq)


@triton.jit
def _packed_attention_forward_kernel(
    qkv_ptr,
    out_ptr,
    seq: tl.constexpr,
    channels: tl.constexpr,
    heads: tl.constexpr,
    head_size: tl.constexpr,
    scale: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_idx = pid_bh // heads
    head_idx = pid_bh - batch_idx * heads
    rows = pid_m * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, head_size)

    q_offsets = ((((batch_idx * seq + rows[:, None]) * 3 + 0) * heads + head_idx) * head_size) + cols[None, :]
    q = tl.load(qkv_ptr + q_offsets, mask=rows[:, None] < seq, other=0.0)

    m_i = tl.full((block_m,), -float("inf"), tl.float32)
    l_i = tl.zeros((block_m,), tl.float32)
    acc = tl.zeros((block_m, head_size), tl.float32)

    for start_n in range(0, seq, block_n):
        keys = start_n + tl.arange(0, block_n)
        k_offsets = ((((batch_idx * seq + keys[:, None]) * 3 + 1) * heads + head_idx) * head_size) + cols[None, :]
        v_offsets = ((((batch_idx * seq + keys[:, None]) * 3 + 2) * heads + head_idx) * head_size) + cols[None, :]
        k = tl.load(qkv_ptr + k_offsets, mask=keys[:, None] < seq, other=0.0)
        v = tl.load(qkv_ptr + v_offsets, mask=keys[:, None] < seq, other=0.0)
        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        causal_mask = keys[None, :] <= rows[:, None]
        valid_mask = (rows[:, None] < seq) & (keys[None, :] < seq)
        scores = tl.where(causal_mask & valid_mask, scores, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v).to(tl.float32)
        m_i = m_ij

    acc = acc / l_i[:, None]
    out_offsets = ((batch_idx * seq + rows[:, None]) * channels) + head_idx * head_size + cols[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=rows[:, None] < seq)


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


def run_triton_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, block_m: int, block_n: int) -> torch.Tensor:
    batch, heads, seq, head_size = q.shape
    out = torch.empty_like(q)
    grid = (triton.cdiv(seq, block_m), batch * heads)
    _attention_forward_kernel[grid](
        q,
        k,
        v,
        out,
        seq,
        head_size,
        head_size ** -0.5,
        block_m,
        block_n,
        num_warps=4,
    )
    return out


def run_triton_packed_attention(qkv: torch.Tensor, *, block_m: int, block_n: int) -> torch.Tensor:
    batch, seq, _, heads, head_size = qkv.shape
    channels = heads * head_size
    out = torch.empty((batch, seq, channels), device=qkv.device, dtype=qkv.dtype)
    grid = (triton.cdiv(seq, block_m), batch * heads)
    _packed_attention_forward_kernel[grid](
        qkv,
        out,
        seq,
        channels,
        heads,
        head_size,
        head_size ** -0.5,
        block_m,
        block_n,
        num_warps=4,
    )
    return out


def bench_attention(
    *,
    repeats: int,
    warmup: int,
    batch: int,
    seq: int,
    channels: int,
    heads: int,
    block_m: int,
    block_n: int,
    output_tol: float,
) -> None:
    head_size = channels // heads
    torch.manual_seed(120)
    q = torch.randn((batch, heads, seq, head_size), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seq, head_size), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((batch, heads, seq, head_size), device="cuda", dtype=torch.bfloat16)

    out = run_triton_attention(q, k, v, block_m=block_m, block_n=block_n)
    torch.cuda.synchronize()

    if batch * heads * seq * seq <= 32 * 1024 * 1024:
        ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        diff = (out.float() - ref.float()).abs().max().item()
        del ref
    else:
        ref = F.scaled_dot_product_attention(q[:1, :1], k[:1, :1], v[:1, :1], dropout_p=0.0, is_causal=True)
        probe = run_triton_attention(q[:1, :1].contiguous(), k[:1, :1].contiguous(), v[:1, :1].contiguous(), block_m=block_m, block_n=block_n)
        torch.cuda.synchronize()
        diff = (probe.float() - ref.float()).abs().max().item()
        del ref, probe
    if diff > output_tol:
        raise AssertionError(f"Triton attention forward parity failed: diff={diff:.6f}")

    def forward() -> torch.Tensor:
        return run_triton_attention(q, k, v, block_m=block_m, block_n=block_n)

    fwd_us = event_time_us(forward, warmup=warmup, repeats=repeats, iters=20)
    print(
        f"Triton Attention Forward (B={batch}, T={seq}, C={channels}, NH={heads}, HS={head_size}): "
        f"{fwd_us:.3f} us (diff={diff:.6f})"
    )

    qkv = torch.randn((batch, seq, 3, heads, head_size), device="cuda", dtype=torch.bfloat16)
    packed_out = run_triton_packed_attention(qkv, block_m=block_m, block_n=block_n)
    torch.cuda.synchronize()
    if batch * heads * seq * seq <= 32 * 1024 * 1024:
        q_view = qkv[:, :, 0].permute(0, 2, 1, 3)
        k_view = qkv[:, :, 1].permute(0, 2, 1, 3)
        v_view = qkv[:, :, 2].permute(0, 2, 1, 3)
        ref = F.scaled_dot_product_attention(q_view, k_view, v_view, dropout_p=0.0, is_causal=True)
        ref = ref.permute(0, 2, 1, 3).reshape(batch, seq, channels)
        packed_diff = (packed_out.float() - ref.float()).abs().max().item()
        del ref
    else:
        qkv_probe = qkv[:1].contiguous()
        q_view = qkv_probe[:, :, 0].permute(0, 2, 1, 3)
        k_view = qkv_probe[:, :, 1].permute(0, 2, 1, 3)
        v_view = qkv_probe[:, :, 2].permute(0, 2, 1, 3)
        ref = F.scaled_dot_product_attention(q_view, k_view, v_view, dropout_p=0.0, is_causal=True)
        ref = ref.permute(0, 2, 1, 3).reshape(1, seq, channels)
        probe = run_triton_packed_attention(qkv_probe, block_m=block_m, block_n=block_n)
        torch.cuda.synchronize()
        packed_diff = (probe.float() - ref.float()).abs().max().item()
        del ref, probe
    if packed_diff > output_tol:
        raise AssertionError(f"Triton packed attention forward parity failed: diff={packed_diff:.6f}")

    def packed_forward() -> torch.Tensor:
        return run_triton_packed_attention(qkv, block_m=block_m, block_n=block_n)

    packed_fwd_us = event_time_us(packed_forward, warmup=warmup, repeats=repeats, iters=20)
    print(
        f"TritonPacked Attention Forward (B={batch}, T={seq}, C={channels}, NH={heads}, HS={head_size}): "
        f"{packed_fwd_us:.3f} us (diff={packed_diff:.6f})"
    )
    shape = f"B={batch} T={seq} C={channels} NH={heads} HS={head_size}"
    print(
        f"{shape:<40} | Triton       | unavailable: attention backward is not implemented in this Triton prototype"
    )
    print(
        f"{shape:<40} | TritonPacked | unavailable: packed attention backward is not implemented in this Triton prototype"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Triton SM120 GPT-2 attention benchmark prototype")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--output-tol", type=float, default=0.08)
    args = parser.parse_args()

    if args.channels % args.heads != 0:
        raise ValueError("--channels must be divisible by --heads")
    if args.block_m <= 0 or args.block_m & (args.block_m - 1):
        raise ValueError("--block-m must be a positive power of two")
    if args.block_n <= 0 or args.block_n & (args.block_n - 1):
        raise ValueError("--block-n must be a positive power of two")
    if not torch.cuda.is_available():
        print("PyTorch CUDA context is not initialized in this process; rerun inside the target benchmark context.", file=sys.stderr)
        return 2

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"Triton attention device: {device_name}; capability=sm_{capability[0]}{capability[1]}")
    bench_attention(
        repeats=args.repeats,
        warmup=args.warmup,
        batch=args.batch,
        seq=args.seq,
        channels=args.channels,
        heads=args.heads,
        block_m=args.block_m,
        block_n=args.block_n,
        output_tol=args.output_tol,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
