#!/usr/bin/env python3
"""
Download a gated HuggingFace Llama-3 checkpoint and write llm.kittens' .bin.

Requires:
    huggingface-cli login
    python dev/download_llama3.py llama3.1:8B
"""

import argparse
import os
import struct
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from train_llama3 import LLaMA, write_model


MODEL_ALIASES = {
    "llama3.1:8B": "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B": "meta-llama/Meta-Llama-3.1-8B",
}

OUTPUT_NAMES = {
    "meta-llama/Meta-Llama-3.1-8B": "llama3.1_8B_bf16.bin",
}

LLAMA3_MAGIC = 20240803
LLAMA3_BF16_VERSION = 5
LLAMA_HEADER_INTS = 256
LLAMA_HEADER_BYTES = LLAMA_HEADER_INTS * 4
BF16_BYTES = 2
SYNTHETIC_CHUNK_BYTES = 1024 * 1024


def round_up_to_multiple(value, multiple):
    return multiple * ((value + multiple - 1) // multiple)


def hidden_dim_from_header(header):
    max_seq_len = header[2]
    vocab_size = header[3]
    num_layers = header[4]
    num_heads = header[5]
    num_kv_heads = header[6]
    channels = header[7]
    multiple_of = header[9] or 1024

    if max_seq_len <= 0 or vocab_size <= 0 or num_layers <= 0 or num_heads <= 0:
        raise ValueError(f"invalid Llama header values: {header[:16]}")
    if num_kv_heads <= 0 or channels <= 0:
        raise ValueError(f"invalid Llama attention/header values: {header[:16]}")
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}")
    if channels % num_heads != 0:
        raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")

    if channels == 2048 and num_layers == 16:
        return 8192
    if channels == 4096 and num_layers == 32:
        return 14336
    hidden = (4 * channels * 2) // 3
    hidden = (13 * hidden) // 10
    return round_up_to_multiple(hidden, multiple_of)


def expected_checkpoint_bytes(header):
    vocab_size = header[3]
    num_layers = header[4]
    num_heads = header[5]
    num_kv_heads = header[6]
    channels = header[7]
    head_size = channels // num_heads
    qkv_width = (num_heads + 2 * num_kv_heads) * head_size
    hidden = hidden_dim_from_header(header)

    elements = [
        vocab_size * channels,                  # wte
        num_layers * channels,                 # ln1w
        num_layers * qkv_width * channels,     # qkvw
        num_layers * channels * channels,      # attprojw
        num_layers * channels,                 # ln2w
        num_layers * hidden * channels,        # fcw_up
        num_layers * hidden * channels,        # fcw_gate
        num_layers * channels * hidden,        # fcprojw
        channels,                              # lnfw
        vocab_size * channels,                 # lm_head
    ]
    return LLAMA_HEADER_BYTES + sum(elements) * BF16_BYTES


def read_llama_header(path):
    with open(path, "rb") as f:
        data = f.read(LLAMA_HEADER_BYTES)
    if len(data) != LLAMA_HEADER_BYTES:
        raise ValueError(f"{path} is too small to contain a Llama checkpoint header")
    return list(struct.unpack("<256i", data))


def validate_checkpoint(path):
    path = Path(path)
    header = read_llama_header(path)
    if header[0] != LLAMA3_MAGIC:
        raise ValueError(f"bad magic for {path}: got {header[0]}, expected {LLAMA3_MAGIC}")
    if header[1] != LLAMA3_BF16_VERSION:
        raise ValueError(
            f"bad Llama checkpoint version for {path}: got {header[1]}, expected bf16 version {LLAMA3_BF16_VERSION}"
        )

    expected_bytes = expected_checkpoint_bytes(header)
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"bad checkpoint size for {path}: got {actual_bytes} bytes, expected {expected_bytes} bytes")

    hidden = hidden_dim_from_header(header)
    if header[8] not in (0, hidden):
        raise ValueError(f"unexpected hidden_dim header value for {path}: got {header[8]}, expected {hidden}")

    print(
        "validated "
        f"{path}: layers={header[4]} heads={header[5]}/{header[6]} "
        f"channels={header[7]} hidden={hidden} context={header[2]} bytes={actual_bytes}",
        flush=True,
    )
    return header


def synthetic_checkpoint_header():
    header = [0] * LLAMA_HEADER_INTS
    header[0] = LLAMA3_MAGIC
    header[1] = LLAMA3_BF16_VERSION
    header[2] = 128       # max sequence length
    header[3] = 128       # vocab size
    header[4] = 2         # layers
    header[5] = 4         # query heads
    header[6] = 2         # kv heads
    header[7] = 128       # channels
    header[8] = 448       # hidden dim
    header[9] = 64        # multiple_of
    header[11] = 500000   # rope theta
    header[12] = 0        # scaled RoPE disabled
    header[13] = 4        # max generation batch size
    header[14] = 3        # version major
    header[15] = 1        # version minor
    return header


def write_zero_payload(f, payload_bytes):
    chunk = b"\0" * min(SYNTHETIC_CHUNK_BYTES, payload_bytes)
    remaining = payload_bytes
    while remaining:
        n = min(len(chunk), remaining)
        f.write(chunk[:n])
        remaining -= n


def write_synthetic_checkpoint(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = synthetic_checkpoint_header()
    expected_bytes = expected_checkpoint_bytes(header)
    with path.open("wb") as f:
        f.write(struct.pack("<256i", *header))
        write_zero_payload(f, expected_bytes - LLAMA_HEADER_BYTES)
    validate_checkpoint(path)
    print(f"wrote synthetic Llama checkpoint {path}", flush=True)
    return path


def run_cpp_dry_validation(checkpoint_path, train_binary, zero_stage, num_processes):
    train_binary = Path(train_binary)
    if not train_binary.exists():
        raise FileNotFoundError(f"{train_binary} does not exist; build it with `make train_llama3cu` first")
    if num_processes < 1:
        raise ValueError(f"num_processes must be positive for C++ dry validation, got {num_processes}")
    subprocess.run(
        [
            str(train_binary.resolve()),
            "-e",
            str(Path(checkpoint_path).resolve()),
            "-x",
            "0",
            "-z",
            str(zero_stage),
            "-pn",
            str(num_processes),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Download and convert Llama-3 weights to llm.kittens .bin format")
    parser.add_argument("model", nargs="?", choices=sorted(MODEL_ALIASES), help="Llama model descriptor or HuggingFace model id")
    parser.add_argument("-o", "--output_dir", default=".", help="Directory to write the converted checkpoint")
    parser.add_argument("--validate-only", metavar="CHECKPOINT", help="Validate an existing llm.kittens Llama checkpoint and exit")
    parser.add_argument("--write-synthetic-checkpoint", metavar="CHECKPOINT", help="Write a tiny synthetic Llama checkpoint for parser smoke tests")
    parser.add_argument("--no-validate", action="store_true", help="Skip post-conversion header and payload-size validation")
    parser.add_argument("--cpp-validate", action="store_true", help="Also run `train_llama3cu -e CHECKPOINT -x 0` after Python validation")
    parser.add_argument(
        "--cpp-zero-stage",
        type=int,
        choices=range(0, 4),
        default=0,
        metavar="0-3",
        help="ZeRO stage for --cpp-validate dry-runs",
    )
    parser.add_argument(
        "--cpp-processes",
        type=int,
        default=1,
        help="Process count for --cpp-validate ZeRO layout dry-runs",
    )
    parser.add_argument("--train-binary", default="./train_llama3cu", help="Path to train_llama3cu for --cpp-validate")
    args = parser.parse_args()

    if args.validate_only:
        validate_checkpoint(args.validate_only)
        if args.cpp_validate:
            run_cpp_dry_validation(
                args.validate_only, args.train_binary, args.cpp_zero_stage, args.cpp_processes
            )
        return

    if args.write_synthetic_checkpoint:
        output_path = write_synthetic_checkpoint(args.write_synthetic_checkpoint)
        if args.cpp_validate:
            run_cpp_dry_validation(
                output_path, args.train_binary, args.cpp_zero_stage, args.cpp_processes
            )
        return

    if args.model is None:
        parser.error("model is required unless --validate-only or --write-synthetic-checkpoint is used")

    model_id = MODEL_ALIASES[args.model]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_NAMES[model_id]

    model = LLaMA.from_pretrained_llama3_hf(model_id)
    write_model(model, str(output_path), dtype="bfloat16")
    if not args.no_validate:
        validate_checkpoint(output_path)
        if args.cpp_validate:
            run_cpp_dry_validation(
                output_path, args.train_binary, args.cpp_zero_stage, args.cpp_processes
            )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
