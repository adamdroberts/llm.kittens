#!/usr/bin/env python3
"""Validate Llama checkpoint/resume artifacts without CUDA."""

from __future__ import annotations

import argparse
import struct
import tempfile
from pathlib import Path


LLAMA3_MAGIC = 20240803
LLAMA_MODEL_VERSION = 5
LLAMA_STATE_MAGIC = 20240804
LLAMA_STATE_VERSION = 1
LLAMA_HEADER_INTS = 256
LLAMA_HEADER_BYTES = LLAMA_HEADER_INTS * 4


def read_header(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint artifact: {path}")
    data = path.read_bytes()
    if len(data) < LLAMA_HEADER_BYTES:
        raise ValueError(f"{path} is too small for a Llama header: {len(data)} bytes")
    return list(struct.unpack("<256i", data[:LLAMA_HEADER_BYTES]))


def write_header(path: Path, header: list[int]) -> None:
    if len(header) != LLAMA_HEADER_INTS:
        raise ValueError(f"expected {LLAMA_HEADER_INTS} header ints, got {len(header)}")
    path.write_bytes(struct.pack("<256i", *header))


def validate_model(path: Path) -> None:
    header = read_header(path)
    if header[0] != LLAMA3_MAGIC:
        raise ValueError(f"{path} model magic {header[0]}, expected {LLAMA3_MAGIC}")
    if header[1] != LLAMA_MODEL_VERSION:
        raise ValueError(f"{path} model version {header[1]}, expected {LLAMA_MODEL_VERSION}")


def validate_state(path: Path, step: int, rank: int, num_processes: int) -> None:
    header = read_header(path)
    checks = {
        "state magic": (header[0], LLAMA_STATE_MAGIC),
        "state version": (header[1], LLAMA_STATE_VERSION),
        "num_processes": (header[2], num_processes),
        "rank": (header[3], rank),
        "step": (header[10], step),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"{path} {label} {actual}, expected {expected}")


def validate_step(output_dir: Path, step: int, rank: int, num_processes: int) -> None:
    step_tag = f"{step:08d}"
    rank_tag = f"{rank:05d}"
    done = output_dir / f"DONE_{step_tag}"
    if not done.exists():
        raise FileNotFoundError(f"missing checkpoint completion marker: {done}")
    validate_model(output_dir / f"model_{step_tag}.bin")
    validate_state(output_dir / f"state_{step_tag}_{rank_tag}.bin", step, rank, num_processes)


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="llmkittens_llama_artifacts_") as tmp:
        output_dir = Path(tmp)
        step = 7
        rank = 3
        num_processes = 8
        step_tag = f"{step:08d}"
        rank_tag = f"{rank:05d}"

        (output_dir / f"DONE_{step_tag}").touch()
        model_header = [0] * LLAMA_HEADER_INTS
        model_header[0] = LLAMA3_MAGIC
        model_header[1] = LLAMA_MODEL_VERSION
        write_header(output_dir / f"model_{step_tag}.bin", model_header)

        state_header = [0] * LLAMA_HEADER_INTS
        state_header[0] = LLAMA_STATE_MAGIC
        state_header[1] = LLAMA_STATE_VERSION
        state_header[2] = num_processes
        state_header[3] = rank
        state_header[10] = step
        state_path = output_dir / f"state_{step_tag}_{rank_tag}.bin"
        write_header(state_path, state_header)

        validate_step(output_dir, step, rank, num_processes)

        bad_header = list(state_header)
        bad_header[10] = step + 1
        write_header(state_path, bad_header)
        try:
            validate_step(output_dir, step, rank, num_processes)
        except ValueError as exc:
            if "step" not in str(exc):
                raise
        else:
            raise AssertionError("bad state step was accepted")

    print("Llama checkpoint artifact self-test OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Llama checkpoint artifact headers")
    parser.add_argument("--self-test", action="store_true", help="Run a synthetic pass/fail artifact validation smoke")
    parser.add_argument("--output-dir", type=Path, help="Llama output/checkpoint directory")
    parser.add_argument("--step", type=int, action="append", help="Checkpoint step to validate")
    parser.add_argument("--rank", type=int, default=0, help="State rank to validate")
    parser.add_argument("--num-processes", type=int, default=1, help="Expected process count in state header")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --self-test is set")
    if args.step is None:
        parser.error("--step is required unless --self-test is set")

    for step in args.step:
        if step < 0:
            raise ValueError(f"checkpoint step must be non-negative: {step}")
        validate_step(args.output_dir, step, args.rank, args.num_processes)

    steps = ", ".join(str(step) for step in args.step)
    print(f"Llama checkpoint artifacts OK: dir={args.output_dir}; steps={steps}; rank={args.rank}")


if __name__ == "__main__":
    main()
