#!/usr/bin/env python3
"""Compare rank-0 training-log loss curves for multi-node sanity gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dev.validate_training_log import Metric, TrainMetric, parse_log


def train_loss_by_step(metrics: list[TrainMetric]) -> dict[int, float]:
    return {metric.step: metric.loss for metric in metrics}


def metric_by_step(metrics: list[Metric]) -> dict[int, float]:
    return {metric.step: metric.value for metric in metrics}


def selected_metric(path: Path, metric: str) -> dict[int, float]:
    log = parse_log(path)
    if metric == "train":
        return train_loss_by_step(log.train)
    if metric == "val":
        return metric_by_step(log.val)
    if metric == "eval":
        return metric_by_step(log.eval)
    raise ValueError(f"unsupported metric: {metric}")


def require_tolerance(abs_tol: float | None, rel_tol: float | None) -> None:
    if abs_tol is None and rel_tol is None:
        raise ValueError("set at least one of --abs-tol or --rel-tol")
    if abs_tol is not None and abs_tol < 0:
        raise ValueError(f"--abs-tol must be non-negative: {abs_tol}")
    if rel_tol is not None and rel_tol < 0:
        raise ValueError(f"--rel-tol must be non-negative: {rel_tol}")


def check_step(step: int, reference: float, candidate: float, abs_tol: float | None, rel_tol: float | None) -> float:
    diff = abs(reference - candidate)
    allowed = 0.0
    if abs_tol is not None:
        allowed = max(allowed, abs_tol)
    if rel_tol is not None:
        allowed = max(allowed, abs(reference) * rel_tol)
    if diff > allowed:
        raise ValueError(
            f"step {step} differs: reference={reference:.8g}, candidate={candidate:.8g}, "
            f"diff={diff:.8g}, allowed={allowed:.8g}"
        )
    return diff


def require_decrease(label: str, values: list[tuple[int, float]]) -> None:
    if len(values) < 2:
        raise ValueError(f"{label} needs at least two points to check decrease")
    first_step, first_value = values[0]
    last_step, last_value = values[-1]
    if not last_value < first_value:
        raise ValueError(
            f"{label} did not decrease over compared window: "
            f"first s:{first_step}={first_value:.8g}, last s:{last_step}={last_value:.8g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two llm.kittens main.log metric curves")
    parser.add_argument("--reference-log", required=True, type=Path, help="Single-node/reference main.log")
    parser.add_argument("--candidate-log", required=True, type=Path, help="Two-node/candidate main.log")
    parser.add_argument("--metric", choices=("train", "val", "eval"), default="train", help="Metric curve to compare")
    parser.add_argument("--start-step", type=int, default=0, help="First step to compare")
    parser.add_argument("--steps", type=int, default=100, help="Number of consecutive steps to compare")
    parser.add_argument("--abs-tol", type=float, help="Absolute difference tolerance")
    parser.add_argument("--rel-tol", type=float, help="Relative difference tolerance")
    parser.add_argument(
        "--require-decrease",
        action="store_true",
        help="Require both compared curves to decrease over the selected window",
    )
    args = parser.parse_args()

    if args.start_step < 0:
        raise ValueError(f"--start-step must be non-negative: {args.start_step}")
    if args.steps < 1:
        raise ValueError(f"--steps must be at least 1: {args.steps}")
    require_tolerance(args.abs_tol, args.rel_tol)

    reference = selected_metric(args.reference_log, args.metric)
    candidate = selected_metric(args.candidate_log, args.metric)
    compared = 0
    max_diff = 0.0
    max_diff_step = args.start_step
    reference_window: list[tuple[int, float]] = []
    candidate_window: list[tuple[int, float]] = []
    for step in range(args.start_step, args.start_step + args.steps):
        if step not in reference:
            raise ValueError(f"reference log missing {args.metric} metric at step {step}: {args.reference_log}")
        if step not in candidate:
            raise ValueError(f"candidate log missing {args.metric} metric at step {step}: {args.candidate_log}")
        diff = check_step(step, reference[step], candidate[step], args.abs_tol, args.rel_tol)
        reference_window.append((step, reference[step]))
        candidate_window.append((step, candidate[step]))
        compared += 1
        if diff > max_diff:
            max_diff = diff
            max_diff_step = step
    if args.require_decrease:
        require_decrease("reference curve", reference_window)
        require_decrease("candidate curve", candidate_window)

    print(
        "Training log comparison OK: "
        f"metric={args.metric}; steps={args.start_step}-{args.start_step + args.steps - 1}; "
        f"pairs={compared}; max_diff={max_diff:.8g} at step {max_diff_step}"
    )


if __name__ == "__main__":
    main()
