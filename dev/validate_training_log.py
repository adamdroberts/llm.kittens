#!/usr/bin/env python3
"""Validate rank-0 training logs against goal-gate evidence requirements."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path


VAL_RE = re.compile(r"^s:(?P<step>\d+)\s+tel:(?P<value>[-+0-9.eE]+)\s*$")
EVAL_RE = re.compile(r"^s:(?P<step>\d+)\s+eval:(?P<value>[-+0-9.eE]+)\s*$")
TRAIN_RE = re.compile(
    r"^s:(?P<step>\d+)\s+trl:(?P<loss>[-+0-9.eE]+)\s+"
    r"lr:(?P<lr>[-+0-9.eE]+)\s+norm:(?P<norm>[-+0-9.eE]+)\s*$"
)


@dataclass(frozen=True)
class Metric:
    step: int
    value: float


@dataclass(frozen=True)
class TrainMetric:
    step: int
    loss: float
    lr: float
    norm: float


@dataclass
class TrainingLog:
    val: list[Metric]
    eval: list[Metric]
    train: list[TrainMetric]


def finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} is not finite: {value}")
    return value


def parse_log(path: Path) -> TrainingLog:
    if not path.exists():
        raise FileNotFoundError(f"training log not found: {path}")
    val: list[Metric] = []
    eval_metrics: list[Metric] = []
    train: list[TrainMetric] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        match = VAL_RE.match(line)
        if match:
            val.append(Metric(int(match.group("step")), finite(float(match.group("value")), f"{path}:{line_number} val")))
            continue
        match = EVAL_RE.match(line)
        if match:
            eval_metrics.append(
                Metric(int(match.group("step")), finite(float(match.group("value")), f"{path}:{line_number} eval"))
            )
            continue
        match = TRAIN_RE.match(line)
        if match:
            train.append(
                TrainMetric(
                    int(match.group("step")),
                    finite(float(match.group("loss")), f"{path}:{line_number} train loss"),
                    finite(float(match.group("lr")), f"{path}:{line_number} lr"),
                    finite(float(match.group("norm")), f"{path}:{line_number} norm"),
                )
            )
            continue
    return TrainingLog(val=val, eval=eval_metrics, train=train)


def latest(metrics: list[Metric], label: str) -> Metric:
    if not metrics:
        raise ValueError(f"no {label} metrics found")
    return max(metrics, key=lambda metric: metric.step)


def latest_train(metrics: list[TrainMetric]) -> TrainMetric:
    if not metrics:
        raise ValueError("no train metrics found")
    return max(metrics, key=lambda metric: metric.step)


def require_step(step: int | None, actual: int, label: str) -> None:
    if step is not None and actual < step:
        raise ValueError(f"{label} latest step {actual} is before required step {step}")


def within_expected(actual: float, expected: float, rel_tol: float, abs_tol: float, label: str) -> None:
    allowed = max(abs_tol, abs(expected) * rel_tol)
    diff = abs(actual - expected)
    if diff > allowed:
        raise ValueError(
            f"{label} {actual:.8g} is outside tolerance for expected {expected:.8g}; "
            f"diff={diff:.8g}, allowed={allowed:.8g}"
        )


def require_loss_decrease(train: list[TrainMetric]) -> None:
    if len(train) < 2:
        raise ValueError("need at least two train metrics to check loss decrease")
    ordered = sorted(train, key=lambda metric: metric.step)
    first = ordered[0]
    last = ordered[-1]
    if not last.loss < first.loss:
        raise ValueError(f"train loss did not decrease: first s:{first.step}={first.loss}, last s:{last.step}={last.loss}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate llm.kittens main.log metric evidence")
    parser.add_argument("--log", required=True, type=Path, help="Path to rank-0 main.log")
    parser.add_argument("--final-step", type=int, help="Default minimum final step expected in checked metrics")
    parser.add_argument("--val-final-step", type=int, help="Minimum final validation-loss step")
    parser.add_argument("--eval-final-step", type=int, help="Minimum final eval-accuracy step")
    parser.add_argument("--train-final-step", type=int, help="Minimum final train-loss step")
    parser.add_argument("--require-val", action="store_true", help="Require a validation loss metric")
    parser.add_argument("--require-eval", action="store_true", help="Require a HellaSwag/eval accuracy metric")
    parser.add_argument("--require-train", action="store_true", help="Require a train loss metric")
    parser.add_argument("--require-train-loss-decrease", action="store_true", help="Require final train loss < first train loss")
    parser.add_argument("--expected-val-loss", type=float, help="Published/reference validation loss")
    parser.add_argument("--expected-eval", type=float, help="Published/reference eval accuracy")
    parser.add_argument("--rel-tol", type=float, default=0.005, help="Relative tolerance for expected values")
    parser.add_argument("--abs-tol", type=float, default=0.0, help="Absolute tolerance floor for expected values")
    parser.add_argument("--max-val-loss", type=float, help="Maximum allowed final validation loss")
    parser.add_argument("--min-eval", type=float, help="Minimum allowed final eval accuracy")
    args = parser.parse_args()

    log = parse_log(args.log)

    final_val = None
    final_eval = None
    final_train = None
    if args.require_val or args.expected_val_loss is not None or args.max_val_loss is not None:
        final_val = latest(log.val, "validation loss")
        require_step(args.val_final_step if args.val_final_step is not None else args.final_step,
                     final_val.step, "validation loss")
        if args.expected_val_loss is not None:
            within_expected(final_val.value, args.expected_val_loss, args.rel_tol, args.abs_tol, "validation loss")
        if args.max_val_loss is not None and final_val.value > args.max_val_loss:
            raise ValueError(f"validation loss {final_val.value:.8g} exceeds max {args.max_val_loss:.8g}")

    if args.require_eval or args.expected_eval is not None or args.min_eval is not None:
        final_eval = latest(log.eval, "eval accuracy")
        require_step(args.eval_final_step if args.eval_final_step is not None else args.final_step,
                     final_eval.step, "eval accuracy")
        if args.expected_eval is not None:
            within_expected(final_eval.value, args.expected_eval, args.rel_tol, args.abs_tol, "eval accuracy")
        if args.min_eval is not None and final_eval.value < args.min_eval:
            raise ValueError(f"eval accuracy {final_eval.value:.8g} is below min {args.min_eval:.8g}")

    if args.require_train or args.require_train_loss_decrease:
        final_train = latest_train(log.train)
        require_step(args.train_final_step if args.train_final_step is not None else args.final_step,
                     final_train.step, "train loss")
        if args.require_train_loss_decrease:
            require_loss_decrease(log.train)

    summary_parts = [f"log={args.log}"]
    if final_train is not None:
        summary_parts.append(f"train s:{final_train.step} loss={final_train.loss:.8g}")
    if final_val is not None:
        summary_parts.append(f"val s:{final_val.step} loss={final_val.value:.8g}")
    if final_eval is not None:
        summary_parts.append(f"eval s:{final_eval.step} acc={final_eval.value:.8g}")
    print("Training log validation OK: " + "; ".join(summary_parts))


if __name__ == "__main__":
    main()
