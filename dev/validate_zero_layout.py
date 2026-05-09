#!/usr/bin/env python3
"""Host-only checks for ZeRO shard offset arithmetic."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    name: str
    param_sizes: tuple[int, ...]
    layer_ids: frozenset[int]
    num_layers: int


def gpt_layout(name: str, max_seq_len: int, num_layers: int, channels: int) -> Layout:
    vp = 50304
    c = channels
    max_t = max_seq_len
    l = num_layers
    return Layout(
        name=name,
        param_sizes=(
            vp * c,
            max_t * c,
            l * c,
            l * c,
            l * (3 * c) * c,
            l * (3 * c),
            l * c * c,
            l * c,
            l * c,
            l * c,
            l * (4 * c) * c,
            l * (4 * c),
            l * c * (4 * c),
            l * c,
            c,
            c,
        ),
        layer_ids=frozenset(range(2, 14)),
        num_layers=l,
    )


def llama_layout(
    name: str,
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    channels: int,
    hidden_dim: int,
) -> Layout:
    c = channels
    l = num_layers
    head_dim = c // num_heads
    qkv_width = (num_heads + 2 * num_kv_heads) * head_dim
    return Layout(
        name=name,
        param_sizes=(
            vocab_size * c,
            l * c,
            l * qkv_width * c,
            l * c * c,
            l * c,
            l * hidden_dim * c,
            l * hidden_dim * c,
            l * c * hidden_dim,
            c,
            vocab_size * c,
        ),
        layer_ids=frozenset(range(1, 8)),
        num_layers=l,
    )


def gpt3_layouts() -> list[Layout]:
    channel_depths = {
        384: 6,
        768: 12,
        1024: 24,
        1536: 24,
        2048: 24,
        2560: 32,
        4096: 32,
        5120: 40,
        12288: 96,
    }
    return [
        gpt_layout(f"gpt3:c{channels}", 2048, depth, channels)
        for channels, depth in channel_depths.items()
    ]


def tensor_intervals(layout: Layout, nproc: int) -> list[tuple[int, int, str]]:
    intervals: list[tuple[int, int, str]] = []
    tensor_offset = 0
    for tensor_id, tensor_elements in enumerate(layout.param_sizes):
        if tensor_id in layout.layer_ids:
            assert tensor_elements % layout.num_layers == 0
            per_layer = tensor_elements // layout.num_layers
            if per_layer % nproc != 0:
                raise AssertionError(
                    f"{layout.name}: tensor {tensor_id} per-layer size {per_layer} "
                    f"is not divisible by {nproc}"
                )
            if tensor_offset % nproc != 0:
                raise AssertionError(
                    f"{layout.name}: tensor {tensor_id} base offset {tensor_offset} "
                    f"is not divisible by {nproc}"
                )
            shard_size = per_layer // nproc
            shard_base = tensor_offset // nproc
            for layer in range(layout.num_layers):
                start = shard_base + layer * shard_size
                intervals.append((start, start + shard_size, f"tensor {tensor_id} layer {layer}"))
        else:
            if tensor_elements % nproc != 0:
                raise AssertionError(
                    f"{layout.name}: tensor {tensor_id} size {tensor_elements} "
                    f"is not divisible by {nproc}"
                )
            if tensor_offset % nproc != 0:
                raise AssertionError(
                    f"{layout.name}: tensor {tensor_id} base offset {tensor_offset} "
                    f"is not divisible by {nproc}"
                )
            shard_size = tensor_elements // nproc
            shard_base = tensor_offset // nproc
            intervals.append((shard_base, shard_base + shard_size, f"tensor {tensor_id}"))
        tensor_offset += tensor_elements
    total = sum(layout.param_sizes)
    if total % nproc != 0:
        raise AssertionError(f"{layout.name}: total parameters {total} not divisible by {nproc}")
    return intervals


def validate_layout(layout: Layout, nproc: int) -> None:
    intervals = sorted(tensor_intervals(layout, nproc))
    cursor = 0
    for start, end, label in intervals:
        if start != cursor:
            raise AssertionError(
                f"{layout.name}: local shard gap/overlap for {label} at nproc={nproc}: "
                f"expected start {cursor}, got {start}"
            )
        if end <= start:
            raise AssertionError(f"{layout.name}: empty interval for {label} at nproc={nproc}")
        cursor = end
    expected = sum(layout.param_sizes) // nproc
    if cursor != expected:
        raise AssertionError(
            f"{layout.name}: local shard coverage ended at {cursor}, expected {expected}"
        )


def main() -> None:
    layouts = [
        gpt_layout("gpt2:d12", 1024, 12, 768),
        *gpt3_layouts(),
        llama_layout("llama3:1B", 128256, 16, 16, 8, 2048, 8192),
        llama_layout("llama3:8B", 128256, 32, 32, 8, 4096, 14336),
        llama_layout("llama3.1:8B", 128256, 32, 32, 8, 4096, 14336),
    ]
    for layout in layouts:
        for nproc in (1, 2, 4, 8, 16):
            validate_layout(layout, nproc)
    print("ZeRO shard layout validation OK")


if __name__ == "__main__":
    main()
