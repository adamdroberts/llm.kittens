#!/usr/bin/env python3
"""Torch attention layout-route probe for GPT-2 SM120 shapes.

This benchmark answers a narrower question than bench_sm120_torch_attention.py:
whether the faster separated-Q/K/V Torch SDPA row can survive the layout cost of
getting there from the trainer's hidden-state input and qkv projection weights.
It intentionally stays Python-side evidence until a native trainer route exists.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F


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


def release_cuda_cache() -> None:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def record_result(results: list[dict[str, object]], route: str, pass_name: str, shape: str, us: float) -> None:
    print(f"{route:<34} | {pass_name:<8} | {shape:<35} | {us:9.3f} us")
    results.append(
        {
            "route": route,
            "pass": pass_name,
            "shape": shape,
            "time_us": us,
        }
    )


def bench_attention_layouts(
    *,
    repeats: int,
    warmup: int,
    batch: int,
    seq: int,
    channels: int,
    heads: int,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    head_size = channels // heads
    bt = batch * seq
    shape = f"B={batch} T={seq} C={channels} NH={heads} HS={head_size}"
    torch.manual_seed(120)

    x = torch.randn((bt, channels), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    wqkv = torch.randn((3 * channels, channels), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bqkv = torch.randn((3 * channels,), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dout_btc = torch.randn((batch, seq, channels), device="cuda", dtype=torch.bfloat16)
    dout_flat = dout_btc.reshape(bt, channels)

    def single_packed_forward() -> torch.Tensor:
        qkv_flat = x @ wqkv.T + bqkv
        qkv = qkv_flat.view(batch, seq, 3, heads, head_size)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        return attn.permute(0, 2, 1, 3).reshape(bt, channels)

    fwd_us = event_time_us(single_packed_forward, warmup=warmup, repeats=repeats, iters=1)
    record_result(results, "TorchQKVSinglePacked", "forward", shape, fwd_us)
    packed_out = single_packed_forward()
    torch.cuda.synchronize()

    def single_packed_backward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.autograd.grad(packed_out, (x, wqkv, bqkv), dout_flat, retain_graph=True)

    bwd_us = event_time_us(single_packed_backward, warmup=warmup, repeats=repeats, iters=1)
    record_result(results, "TorchQKVSinglePacked", "backward", shape, bwd_us)
    del packed_out
    release_cuda_cache()

    wq, wk, wv = wqkv[:channels], wqkv[channels : 2 * channels], wqkv[2 * channels :]
    bq, bk, bv = bqkv[:channels], bqkv[channels : 2 * channels], bqkv[2 * channels :]

    def split_strided_forward() -> torch.Tensor:
        q_btc = x @ wq.T + bq
        k_btc = x @ wk.T + bk
        v_btc = x @ wv.T + bv
        q = q_btc.view(batch, seq, heads, head_size).permute(0, 2, 1, 3)
        k = k_btc.view(batch, seq, heads, head_size).permute(0, 2, 1, 3)
        v = v_btc.view(batch, seq, heads, head_size).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        return attn.permute(0, 2, 1, 3).reshape(bt, channels)

    fwd_us = event_time_us(split_strided_forward, warmup=warmup, repeats=repeats, iters=1)
    record_result(results, "TorchQKVSplitStrided", "forward", shape, fwd_us)
    split_strided_out = split_strided_forward()
    torch.cuda.synchronize()

    def split_strided_backward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.autograd.grad(split_strided_out, (x, wqkv, bqkv), dout_flat, retain_graph=True)

    bwd_us = event_time_us(split_strided_backward, warmup=warmup, repeats=repeats, iters=1)
    record_result(results, "TorchQKVSplitStrided", "backward", shape, bwd_us)
    del split_strided_out
    release_cuda_cache()

    def split_materialized_forward() -> torch.Tensor:
        q_btc = x @ wq.T + bq
        k_btc = x @ wk.T + bk
        v_btc = x @ wv.T + bv
        q = q_btc.view(batch, seq, heads, head_size).permute(0, 2, 1, 3).contiguous()
        k = k_btc.view(batch, seq, heads, head_size).permute(0, 2, 1, 3).contiguous()
        v = v_btc.view(batch, seq, heads, head_size).permute(0, 2, 1, 3).contiguous()
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        return attn.permute(0, 2, 1, 3).reshape(bt, channels)

    fwd_us = event_time_us(split_materialized_forward, warmup=warmup, repeats=repeats, iters=1)
    record_result(results, "TorchQKVSplitMaterialized", "forward", shape, fwd_us)
    split_materialized_out = split_materialized_forward()
    torch.cuda.synchronize()

    def split_materialized_backward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.autograd.grad(split_materialized_out, (x, wqkv, bqkv), dout_flat, retain_graph=True)

    bwd_us = event_time_us(split_materialized_backward, warmup=warmup, repeats=repeats, iters=1)
    record_result(results, "TorchQKVSplitMaterialized", "backward", shape, bwd_us)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Torch SM120 GPT-2 attention layout-route benchmark")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if args.channels % args.heads != 0:
        raise ValueError("--channels must be divisible by --heads")
    if not torch.cuda.is_available():
        print("PyTorch CUDA context is not initialized in this process; rerun inside the target benchmark context.", file=sys.stderr)
        return 2

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"Torch attention layout device: {device_name}; capability=sm_{capability[0]}{capability[1]}")
    results = bench_attention_layouts(
        repeats=args.repeats,
        warmup=args.warmup,
        batch=args.batch,
        seq=args.seq,
        channels=args.channels,
        heads=args.heads,
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "device_name": device_name,
            "capability": f"sm_{capability[0]}{capability[1]}",
            "repeats": args.repeats,
            "warmup": args.warmup,
            "results": results,
        }
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
