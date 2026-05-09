#!/usr/bin/env python3
"""Host-only smoke test for profile_gpt2cu.py CSV parsing and threshold gates."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_row(kernel: str, time_ms: float, read_gib: float, write_gib: float, tensor_pct: float) -> list[str]:
    row = ["0"] * 18
    row[4] = kernel
    row[10] = "sm_90a"
    row[11] = f"{read_gib:.6f}"
    row[12] = f"{write_gib:.6f}"
    row[13] = f"{time_ms:.6f}"
    row[14] = "1024"
    row[15] = "512"
    row[16] = f"{tensor_pct:.6f}"
    row[17] = "1000000"
    return row


def write_csv(path: Path, tensor_pct: float) -> None:
    rows = [
        [f"h{i}" for i in range(18)],
        [f"u{i}" for i in range(18)],
        make_row("void encoder_forward_kernel(float*)", 1.0, 0.5, 0.2, tensor_pct),
        make_row("void matmul_forward_kernel(float*)", 2.0, 0.6, 0.3, tensor_pct),
        make_row("void layernorm_forward_kernel(float*)", 1.5, 0.4, 0.2, tensor_pct),
        make_row("void fused_classifier_kernel(float*)", 1.0, 0.2, 0.1, tensor_pct),
        make_row("void matmul_backward_kernel(float*)", 2.0, 0.5, 0.3, tensor_pct),
        make_row("void layernorm_backward_kernel(float*)", 1.0, 0.3, 0.2, tensor_pct),
        make_row("void adamw_kernel(float*)", 0.5, 0.1, 0.1, tensor_pct),
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def run_parser(csv_path: Path, min_tensor_util: float, gelu_fusion: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "profile_gpt2cu.py",
            "--csv-input",
            str(csv_path),
            "--min-tensor-util",
            str(min_tensor_util),
            "--gelu-fusion",
            str(gelu_fusion),
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="llmkittens-profile-parser-") as tmp:
        passing = Path(tmp) / "passing.csv"
        failing = Path(tmp) / "failing.csv"
        write_csv(passing, tensor_pct=82.0)
        write_csv(failing, tensor_pct=42.0)

        ok = run_parser(passing, 70.0, gelu_fusion=1)
        if ok.returncode != 0:
            print(ok.stdout)
            raise SystemExit("expected passing synthetic profile to satisfy threshold")
        if "Tensor-core utilization gate:" not in ok.stdout:
            print(ok.stdout)
            raise SystemExit("passing parser output did not include threshold gate")

        bad = run_parser(failing, 70.0, gelu_fusion=0)
        if bad.returncode == 0:
            print(bad.stdout)
            raise SystemExit("expected failing synthetic profile to trip threshold")
        if "below the required" not in bad.stdout:
            print(bad.stdout)
            raise SystemExit("failing parser output did not explain threshold failure")

    print("profile parser validation OK")


if __name__ == "__main__":
    main()
