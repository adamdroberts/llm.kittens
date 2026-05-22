#!/usr/bin/env python3
"""Write machine-readable metadata for an SM120 optimization round."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sm120_objective_contract import EXPECTED_MANIFEST_BINARIES

DEFAULT_BINARIES = EXPECTED_MANIFEST_BINARIES

CONFIG_KEYS = (
    "run_label",
    "artifact_dir",
    "train_out_dir",
    "max_steps",
    "train_zero_stage",
    "device_arch",
    "build_jobs",
    "no_multi_gpu",
    "no_use_mpi",
    "run_stack_probe",
    "run_correctness",
    "run_benchmarks",
    "run_python_stack_benchmarks",
    "cudnn_packed_backward_route",
    "libtorch_runtime_route",
    "libtorch_runtime_supplemental_shapes",
    "run_libtorch_trainer_link_probe",
    "run_libtorch_matmul_benchmarks",
    "libtorch_matmul_shapes",
    "sm120_use_libtorch_memory",
    "sm120_use_libtorch_grad_zero",
    "sm120_use_libtorch_dresidual_zero",
    "run_training",
    "keep_checkpoints",
)


def run_command(args: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(args, text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        return {
            "command": args,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "command": args,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
        }
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"])
    short_commit = run_command(["git", "rev-parse", "--short", "HEAD"])
    status = run_command(["git", "status", "--short"])
    status_lines = [line for line in status["stdout"].splitlines() if line.strip()]
    return {
        "commit": commit["stdout"] if commit["returncode"] == 0 else "unknown",
        "short_commit": short_commit["stdout"] if short_commit["returncode"] == 0 else "unknown",
        "status_count": len(status_lines),
        "status_path": "git-status.txt",
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    binaries = [binary_record(Path(raw)) for raw in args.binary]
    config = {key: getattr(args, key) for key in CONFIG_KEYS}
    env_keys = (
        "CUDA_VISIBLE_DEVICES",
        "CUDA_HOME",
        "LD_LIBRARY_PATH",
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        "VIRTUAL_ENV",
        "PYTHON_BIN",
        "SM120_USE_CUBLASLT_GEMM",
        "SM120_USE_LIBTORCH_MEMORY",
        "SM120_USE_LIBTORCH_GRAD_ZERO",
        "SM120_USE_LIBTORCH_DRESIDUAL_ZERO",
        "LLMK_SM120_CUBLASLT_HEURISTIC_RESULTS",
        "LLMK_SM120_CUBLASLT_SELECT_MAX_WAVES",
        "LLMK_SM120_CUBLASLT_HEURISTIC_INDEX",
        "LLMK_LIBTORCH_RUNTIME_ROUTE",
    )
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "git": git_metadata(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
        },
        "toolchain": {
            "nvcc": run_command(["nvcc", "--version"]),
            "nvidia_smi": run_command(["nvidia-smi"]),
        },
        "environment": {key: os.environ.get(key, "") for key in env_keys},
        "binaries": binaries,
    }


def write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    config = manifest["config"]
    git = manifest["git"]
    lines = [
        "# SM120 Round Manifest",
        "",
        f"- run label: `{config['run_label']}`",
        f"- artifact dir: `{config['artifact_dir']}`",
        f"- train output dir: `{config['train_out_dir']}`",
        f"- device arch: `{config['device_arch']}`",
        f"- max steps: `{config['max_steps']}`",
        f"- train zero stage: `{config['train_zero_stage']}`",
        f"- SM120 LibTorch grad-zero route: `{config['sm120_use_libtorch_grad_zero']}`",
        f"- SM120 LibTorch dresidual-zero route: `{config['sm120_use_libtorch_dresidual_zero']}`",
        f"- git commit: `{git['short_commit']}`",
        f"- changed paths: `{git['status_count']}`",
        "",
        "## Binaries",
        "",
        "| Path | Exists | Size bytes | SHA256 |",
        "|---|---:|---:|---|",
    ]
    for binary in manifest["binaries"]:
        lines.append(
            f"| `{binary['path']}` | `{binary['exists']}` | "
            f"`{binary.get('size_bytes', '')}` | `{binary.get('sha256', '')}` |"
        )
    lines.extend(["", "## Toolchain", ""])
    nvcc = manifest["toolchain"]["nvcc"]
    lines.extend(
        [
            f"- nvcc returncode: `{nvcc['returncode']}`",
            "```text",
            nvcc["stdout"] or nvcc["stderr"],
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write SM120 round manifest")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--train-out-dir", required=True)
    parser.add_argument("--max-steps", required=True)
    parser.add_argument("--train-zero-stage", required=True)
    parser.add_argument("--device-arch", required=True)
    parser.add_argument("--build-jobs", required=True)
    parser.add_argument("--no-multi-gpu", required=True)
    parser.add_argument("--no-use-mpi", required=True)
    parser.add_argument("--run-stack-probe", required=True)
    parser.add_argument("--run-correctness", required=True)
    parser.add_argument("--run-benchmarks", required=True)
    parser.add_argument("--run-python-stack-benchmarks", required=True)
    parser.add_argument("--cudnn-packed-backward-route", required=True)
    parser.add_argument("--libtorch-runtime-route", required=True)
    parser.add_argument("--libtorch-runtime-supplemental-shapes", required=True)
    parser.add_argument("--run-libtorch-trainer-link-probe", required=True)
    parser.add_argument("--run-libtorch-matmul-benchmarks", required=True)
    parser.add_argument("--libtorch-matmul-shapes", required=True)
    parser.add_argument("--sm120-use-libtorch-memory", required=True)
    parser.add_argument("--sm120-use-libtorch-grad-zero", required=True)
    parser.add_argument("--sm120-use-libtorch-dresidual-zero", required=True)
    parser.add_argument("--run-training", required=True)
    parser.add_argument("--keep-checkpoints", required=True)
    parser.add_argument("--binary", action="append", default=list(DEFAULT_BINARIES))
    args = parser.parse_args()

    manifest = build_manifest(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_markdown(args.markdown_out, manifest)

    existing = sum(1 for binary in manifest["binaries"] if binary["exists"])
    print(
        "SM120 round manifest OK: "
        f"json={args.json_out}; markdown={args.markdown_out}; "
        f"binaries={existing}/{len(manifest['binaries'])}"
    )


if __name__ == "__main__":
    main()
