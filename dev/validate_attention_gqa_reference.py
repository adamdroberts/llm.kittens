#!/usr/bin/env python3
"""
CPU-only GQA + RoPE reference check for the Llama smoke shapes.

This does not validate the CUDA/TK kernels. It validates the reference math used
by the H100 smoke tests by comparing two independent PyTorch formulations:

1. materialize RoPE on Q/K, repeat KV heads, then run causal attention;
2. group query heads by KV head and apply RoPE inside each group, matching the
   TK tile-load-RoPE contract.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Shape:
    batch: int
    seq_len: int
    query_heads: int
    kv_heads: int
    head_dim: int

    @property
    def qkv_heads(self) -> int:
        return self.query_heads + 2 * self.kv_heads

    @property
    def channels(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def n_rep(self) -> int:
        if self.query_heads % self.kv_heads != 0:
            raise ValueError(f"query_heads={self.query_heads} must be divisible by kv_heads={self.kv_heads}")
        return self.query_heads // self.kv_heads


def parse_shape(value: str) -> Shape:
    parts = [int(p) for p in value.split(",")]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("shape must be B,T,NH,NKVH,HS")
    shape = Shape(*parts)
    if min(parts) <= 0:
        raise argparse.ArgumentTypeError("shape values must be positive")
    if shape.head_dim % 2 != 0:
        raise argparse.ArgumentTypeError("head_dim must be even for RoPE")
    if shape.query_heads % shape.kv_heads != 0:
        raise argparse.ArgumentTypeError("query_heads must be divisible by kv_heads")
    return shape


def rope_cache(seq_len: int, head_dim: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.arange(1, seq_len + 1, dtype=dtype).unsqueeze(1)
    d = torch.arange(1, head_dim // 2 + 1, dtype=dtype).unsqueeze(0)
    angles = 0.0007 * t * d
    return torch.cos(angles), torch.sin(angles)


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    c = cos.view(1, 1, cos.shape[0], half)
    s = sin.view(1, 1, sin.shape[0], half)
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)


def unpack_qkv(packed: torch.Tensor, shape: Shape) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # packed is (B, T, NH + 2*NKVH, HS), matching the Llama QKV projection layout.
    q = packed[:, :, : shape.query_heads, :].permute(0, 2, 1, 3).contiguous()
    k0 = shape.query_heads
    v0 = shape.query_heads + shape.kv_heads
    k = packed[:, :, k0:v0, :].permute(0, 2, 1, 3).contiguous()
    v = packed[:, :, v0:, :].permute(0, 2, 1, 3).contiguous()
    return q, k, v


def causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
    masked = torch.full((seq_len, seq_len), -torch.inf, dtype=dtype)
    return torch.where(mask, torch.zeros((), dtype=dtype), masked)


def materialized_repeat_attention(packed: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, shape: Shape) -> torch.Tensor:
    q, k, v = unpack_qkv(packed, shape)
    q = rotate(q, cos, sin)
    k = rotate(k, cos, sin)
    k_rep = k.repeat_interleave(shape.n_rep, dim=1)
    v_rep = v.repeat_interleave(shape.n_rep, dim=1)
    scores = torch.einsum("bhtd,bhsd->bhts", q, k_rep) / math.sqrt(shape.head_dim)
    scores = scores + causal_mask(shape.seq_len, scores.dtype).view(1, 1, shape.seq_len, shape.seq_len)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhts,bhsd->bhtd", probs, v_rep)
    return out.permute(0, 2, 1, 3).contiguous().view(shape.batch, shape.seq_len, shape.channels)


def grouped_tile_rope_attention(packed: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, shape: Shape) -> torch.Tensor:
    q, k, v = unpack_qkv(packed, shape)
    pieces: list[torch.Tensor] = []
    mask = causal_mask(shape.seq_len, packed.dtype).view(1, 1, shape.seq_len, shape.seq_len)
    for kvh in range(shape.kv_heads):
        q_group = q[:, kvh * shape.n_rep : (kvh + 1) * shape.n_rep, :, :]
        q_group = rotate(q_group, cos, sin)
        k_head = rotate(k[:, kvh : kvh + 1, :, :], cos, sin).squeeze(1)
        v_head = v[:, kvh, :, :]
        scores = torch.einsum("brtd,bsd->brts", q_group, k_head) / math.sqrt(shape.head_dim)
        probs = torch.softmax(scores + mask, dim=-1)
        pieces.append(torch.einsum("brts,bsd->brtd", probs, v_head))
    out = torch.cat(pieces, dim=1)
    return out.permute(0, 2, 1, 3).contiguous().view(shape.batch, shape.seq_len, shape.channels)


def run_shape(shape: Shape, dtype: torch.dtype, atol: float, rtol: float, seed: int) -> None:
    torch.manual_seed(seed + shape.seq_len)
    packed_a = torch.empty(shape.batch, shape.seq_len, shape.qkv_heads, shape.head_dim, dtype=dtype)
    packed_a.uniform_(-0.35, 0.35)
    packed_a.requires_grad_(True)
    packed_b = packed_a.detach().clone().requires_grad_(True)
    cos, sin = rope_cache(shape.seq_len, shape.head_dim, dtype)

    out_a = materialized_repeat_attention(packed_a, cos, sin, shape)
    out_b = grouped_tile_rope_attention(packed_b, cos, sin, shape)
    dout = torch.empty_like(out_a).uniform_(-0.2, 0.2)
    out_a.backward(dout)
    out_b.backward(dout)

    fwd_max = (out_a.detach() - out_b.detach()).abs().max().item()
    bwd_max = (packed_a.grad.detach() - packed_b.grad.detach()).abs().max().item()
    fwd_ok = torch.allclose(out_a, out_b, atol=atol, rtol=rtol)
    bwd_ok = torch.allclose(packed_a.grad, packed_b.grad, atol=atol, rtol=rtol)
    status = "OK" if fwd_ok and bwd_ok else "FAIL"
    print(
        f"{status} B={shape.batch} T={shape.seq_len} NH={shape.query_heads} "
        f"NKVH={shape.kv_heads} HS={shape.head_dim}: "
        f"forward_max={fwd_max:.3e} backward_max={bwd_max:.3e}"
    )
    if not (fwd_ok and bwd_ok):
        raise AssertionError(f"GQA reference mismatch for {shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CPU-only GQA + RoPE reference math")
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        default=None,
        help="Shape as B,T,NH,NKVH,HS. May be repeated.",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    shapes = args.shape or [
        Shape(batch=1, seq_len=128, query_heads=4, kv_heads=2, head_dim=128),
        Shape(batch=1, seq_len=256, query_heads=4, kv_heads=2, head_dim=128),
    ]
    for shape in shapes:
        run_shape(shape, dtype, args.atol, args.rtol, args.seed)
    print("GQA reference validation OK")


if __name__ == "__main__":
    main()
