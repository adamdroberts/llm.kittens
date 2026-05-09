#!/usr/bin/env python3
"""Host-only smoke test for train_llama3.py's checkpoint writer."""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
DEV_DIR = ROOT_DIR / "dev"
for path in (ROOT_DIR, DEV_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from train_llama3 import LLaMA, LlamaConfig, llama_hidden_dim, write_model
from dev.download_llama3 import LLAMA_HEADER_BYTES, validate_checkpoint


def bf16_bits(value: float) -> int:
    return int(torch.tensor([value], dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF


def fill_named_parameters(model: LLaMA) -> dict[str, float]:
    values: dict[str, float] = {}
    for index, (name, param) in enumerate(model.named_parameters(), start=1):
        value = 0.03125 * index
        param.data.fill_(value)
        values[name] = value
    return values


def expected_segments(model: LLaMA, values: dict[str, float]) -> list[tuple[str, int, int]]:
    cfg = model.config
    segments: list[tuple[str, int, int]] = []

    def add(name: str) -> None:
        tensor = dict(model.named_parameters())[name]
        segments.append((name, tensor.numel(), bf16_bits(values[name])))

    add("transformer.wte.weight")
    for i in range(cfg.n_layer):
        add(f"transformer.h.{i}.ln_1.weight")
    for i in range(cfg.n_layer):
        add(f"transformer.h.{i}.attn.c_attn.weight")
    for i in range(cfg.n_layer):
        add(f"transformer.h.{i}.attn.c_proj.weight")
    for i in range(cfg.n_layer):
        add(f"transformer.h.{i}.ln_2.weight")
    for i in range(cfg.n_layer):
        add(f"transformer.h.{i}.mlp.c_fc.weight")
    for i in range(cfg.n_layer):
        add(f"transformer.h.{i}.mlp.c_fc2.weight")
    for i in range(cfg.n_layer):
        add(f"transformer.h.{i}.mlp.c_proj.weight")
    add("transformer.ln_f.weight")
    add("lm_head.weight")
    return segments


def validate_header(path: Path, config: LlamaConfig) -> None:
    with path.open("rb") as f:
        header = list(struct.unpack("<256i", f.read(LLAMA_HEADER_BYTES)))
    expected = {
        0: 20240803,
        1: 5,
        2: config.block_size,
        3: config.vocab_size,
        4: config.n_layer,
        5: config.n_head,
        6: config.n_kv_head,
        7: config.n_embd,
        8: llama_hidden_dim(config),
        9: config.multiple_of,
        11: int(config.rope_theta),
        12: int(config.use_scaled_rope),
        13: config.max_gen_batch_size,
        14: int(config.version.split(".")[0]),
        15: int(config.version.split(".")[1]),
    }
    for index, value in expected.items():
        if header[index] != value:
            raise AssertionError(f"header[{index}]={header[index]}, expected {value}")


def validate_payload_order(path: Path, segments: list[tuple[str, int, int]]) -> None:
    payload = memoryview(path.read_bytes())[LLAMA_HEADER_BYTES:]
    offset = 0
    for name, numel, expected_bits in segments:
        nbytes = numel * 2
        chunk = payload[offset : offset + nbytes]
        if len(chunk) != nbytes:
            raise AssertionError(f"{name} payload segment is truncated")
        values = struct.unpack(f"<{numel}H", chunk)
        unique = set(values)
        if unique != {expected_bits}:
            sample = sorted(unique)[:5]
            raise AssertionError(f"{name} payload bits {sample}, expected only {expected_bits}")
        offset += nbytes
    if offset != len(payload):
        raise AssertionError(f"payload has {len(payload) - offset} trailing bytes")


def run_cpp_validation(path: Path, train_binary: Path) -> None:
    if not train_binary.exists():
        raise FileNotFoundError(f"{train_binary} does not exist; build it with `make train_llama3cu` first")
    subprocess.run([str(train_binary.resolve()), "-e", str(path.resolve()), "-x", "0", "-z", "2", "-pn", "8"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate tiny Llama write_model checkpoint output")
    parser.add_argument("--keep", action="store_true", help="Keep the generated checkpoint and print its path")
    parser.add_argument("--cpp-validate", action="store_true", help="Also dry-parse with train_llama3cu")
    parser.add_argument("--train-binary", default="./train_llama3cu", help="Path to train_llama3cu")
    args = parser.parse_args()

    config = LlamaConfig(
        version="3.1",
        block_size=16,
        vocab_size=32,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=16,
        multiple_of=8,
        max_gen_batch_size=4,
        use_kv=False,
        flash=False,
    )
    model = LLaMA(config)
    values = fill_named_parameters(model)

    with tempfile.TemporaryDirectory(prefix="llmkittens-llama-converter-") as tmp:
        path = Path(tmp) / "tiny_llama3_writer_bf16.bin"
        write_model(model, str(path), dtype="bfloat16")
        validate_header(path, config)
        validate_checkpoint(path)
        validate_payload_order(path, expected_segments(model, values))
        if args.cpp_validate:
            run_cpp_validation(path, Path(args.train_binary))
        if args.keep:
            kept = Path.cwd() / "tiny_llama3_writer_bf16.bin"
            kept.write_bytes(path.read_bytes())
            print(f"kept {kept}")

    print("Llama converter writer validation OK")


if __name__ == "__main__":
    main()
