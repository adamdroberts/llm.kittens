#!/usr/bin/env python3
"""Synthetic smoke checks for training-log validators."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_tool(args: list[str], *, should_pass: bool, marker: str | None = None) -> None:
    result = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if should_pass and result.returncode != 0:
        raise RuntimeError(
            f"{args[0]} failed unexpectedly with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if not should_pass and result.returncode == 0:
        raise RuntimeError(f"{args[0]} succeeded unexpectedly\nstdout:\n{result.stdout}")
    if marker is not None and marker not in result.stdout:
        raise RuntimeError(f"{args[0]} missing success marker {marker!r}\nstdout:\n{result.stdout}")


def write_log(path: Path, *, train_losses: list[float], val_loss: float, eval_acc: float) -> None:
    lines: list[str] = []
    for step, loss in enumerate(train_losses):
        lines.append(f"s:{step} trl:{loss:.6f} lr:0.000300 norm:1.250000")
    final_step = len(train_losses) - 1
    lines.append(f"s:{final_step} tel:{val_loss:.6f}")
    lines.append(f"s:{final_step} eval:{eval_acc:.6f}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="llmkittens_log_tools_") as tmp:
        tmpdir = Path(tmp)
        reference = tmpdir / "reference.log"
        candidate = tmpdir / "candidate.log"
        bad_loss = tmpdir / "bad_loss.log"
        bad_curve_reference = tmpdir / "bad_curve_reference.log"
        bad_curve_candidate = tmpdir / "bad_curve_candidate.log"

        write_log(reference, train_losses=[10.0, 8.0, 6.0], val_loss=4.0, eval_acc=0.25)
        write_log(candidate, train_losses=[10.05, 8.04, 6.03], val_loss=4.02, eval_acc=0.251)
        write_log(bad_loss, train_losses=[10.0, 11.0, 12.0], val_loss=5.0, eval_acc=0.10)
        write_log(bad_curve_reference, train_losses=[10.0, 10.5, 11.0], val_loss=5.0, eval_acc=0.10)
        write_log(bad_curve_candidate, train_losses=[10.0, 10.5, 11.0], val_loss=5.0, eval_acc=0.10)

        run_tool(
            [
                "dev/validate_training_log.py",
                "--log",
                str(reference),
                "--val-final-step",
                "2",
                "--eval-final-step",
                "2",
                "--train-final-step",
                "2",
                "--require-val",
                "--require-eval",
                "--require-train",
                "--require-train-loss-decrease",
                "--max-val-loss",
                "4.5",
                "--min-eval",
                "0.2",
            ],
            should_pass=True,
            marker="Training log validation OK",
        )
        run_tool(
            [
                "dev/validate_training_log.py",
                "--log",
                str(candidate),
                "--val-final-step",
                "2",
                "--eval-final-step",
                "2",
                "--require-val",
                "--require-eval",
                "--expected-val-loss",
                "4.0",
                "--expected-eval",
                "0.25",
                "--rel-tol",
                "0.01",
            ],
            should_pass=True,
            marker="Training log validation OK",
        )
        run_tool(
            [
                "dev/validate_training_log.py",
                "--log",
                str(bad_loss),
                "--require-train",
                "--require-train-loss-decrease",
            ],
            should_pass=False,
        )
        run_tool(
            [
                "dev/validate_training_log.py",
                "--log",
                str(reference),
                "--require-val",
                "--max-val-loss",
                "3.0",
            ],
            should_pass=False,
        )
        run_tool(
            [
                "dev/validate_training_log.py",
                "--log",
                str(reference),
                "--require-val",
                "--expected-val-loss",
                "4.5",
                "--rel-tol",
                "0.01",
            ],
            should_pass=False,
        )
        run_tool(
            [
                "dev/validate_training_log.py",
                "--log",
                str(reference),
                "--require-eval",
                "--expected-eval",
                "0.30",
                "--rel-tol",
                "0.01",
            ],
            should_pass=False,
        )
        run_tool(
            [
                "dev/validate_training_log.py",
                "--log",
                str(reference),
                "--require-val",
                "--val-final-step",
                "3",
            ],
            should_pass=False,
        )

        run_tool(
            [
                "dev/compare_training_logs.py",
                "--reference-log",
                str(reference),
                "--candidate-log",
                str(candidate),
                "--metric",
                "train",
                "--start-step",
                "0",
                "--steps",
                "3",
                "--rel-tol",
                "0.01",
                "--require-decrease",
            ],
            should_pass=True,
            marker="Training log comparison OK",
        )
        run_tool(
            [
                "dev/compare_training_logs.py",
                "--reference-log",
                str(reference),
                "--candidate-log",
                str(candidate),
                "--metric",
                "eval",
                "--start-step",
                "2",
                "--steps",
                "1",
                "--abs-tol",
                "0.01",
            ],
            should_pass=True,
            marker="Training log comparison OK",
        )
        run_tool(
            [
                "dev/compare_training_logs.py",
                "--reference-log",
                str(reference),
                "--candidate-log",
                str(candidate),
                "--metric",
                "train",
                "--start-step",
                "0",
                "--steps",
                "3",
                "--rel-tol",
                "0.001",
            ],
            should_pass=False,
        )
        run_tool(
            [
                "dev/compare_training_logs.py",
                "--reference-log",
                str(bad_curve_reference),
                "--candidate-log",
                str(bad_curve_candidate),
                "--metric",
                "train",
                "--start-step",
                "0",
                "--steps",
                "3",
                "--rel-tol",
                "0.01",
                "--require-decrease",
            ],
            should_pass=False,
        )

    print("Training log tool validation OK")


if __name__ == "__main__":
    main()
