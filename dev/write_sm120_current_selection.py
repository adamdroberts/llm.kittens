#!/usr/bin/env python3
"""Write a consolidated SM120 backend-selection summary."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from sm120_objective_contract import (
    CURRENT_NATIVE_SELECTION_ROUND,
    CURRENT_OPTIONAL_STACK_ROUND,
    LAYERNORM_SELECTION_SHAPE,
    expected_trainer_selection_keys,
)


DEFAULT_NATIVE_ROUND = Path(CURRENT_NATIVE_SELECTION_ROUND)
DEFAULT_OPTIONAL_ROUND = Path(CURRENT_OPTIONAL_STACK_ROUND)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def selected_payload(round_dir: Path) -> dict[str, Any]:
    path = round_dir / "selected-backends.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path)


def promotion_payload(round_dir: Path) -> dict[str, Any]:
    path = round_dir / "promotion-candidates.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path)


def native_training_evidence(round_dir: Path, allow_benchmark_only_native: bool) -> dict[str, Any]:
    manifest_path = round_dir / "round-manifest.json"
    train_log_path = round_dir / "train_gpt2cu.log"
    if allow_benchmark_only_native:
        return {
            "allow_benchmark_only_native": True,
            "evidence_note": (
                "native training evidence requirement bypassed explicitly; "
                "do not publish this as the stable trainer mix"
            ),
        }
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{manifest_path} lacks a config object")
    if str(config.get("run_training")) != "1":
        raise ValueError(
            "native selection round must include training evidence; "
            f"{manifest_path} records run_training={config.get('run_training')!r}"
        )
    try:
        max_steps = int(str(config.get("max_steps", "0")))
    except ValueError as exc:
        raise ValueError(f"{manifest_path} records invalid max_steps={config.get('max_steps')!r}") from exc
    if max_steps <= 0:
        raise ValueError(f"{manifest_path} records non-positive max_steps={max_steps}")
    if not train_log_path.exists() or train_log_path.stat().st_size == 0:
        raise FileNotFoundError(train_log_path)
    train_log = train_log_path.read_text(errors="replace")
    if "total average iteration time:" not in train_log:
        raise ValueError(f"{train_log_path} lacks total average iteration time evidence")
    return {
        "allow_benchmark_only_native": False,
        "manifest_path": str(manifest_path),
        "train_log_path": str(train_log_path),
        "max_steps": max_steps,
    }


def slim_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "suite",
        "kernel",
        "shape",
        "selected_stack",
        "selected_time_us",
        "next_stack",
        "next_time_us",
        "use_scope",
        "trainer_call_path_available",
        "trainer_call_path_kind",
        "decision_status",
        "decision_active",
        "decision_decision",
        "decision_evidence",
        "decision_note",
        "candidate_class",
        "promotion_gate",
        "priority",
        "speedup_vs_next_pct",
        "source_run_label",
        "source_artifact_dir",
        "source_git_commit",
        "source_run_config",
        "timing_log",
        "timing_log_path",
        "config_artifact_path",
        "stack_probe_artifact_path",
        "correctness_logs",
        "correctness_log_paths",
        "correctness_evidence_note",
    ]
    return {key: row[key] for key in keys if key in row}


def with_promotion_metadata(row: dict[str, Any], promotion_row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    registry_authoritative_keys = {
        "decision_note",
        "decision_decision",
        "decision_evidence",
    }
    for key in (
        "candidate_class",
        "promotion_gate",
        "priority",
        "speedup_vs_next_pct",
        "decision_note",
        "decision_decision",
        "decision_evidence",
    ):
        if key in promotion_row and (key in registry_authoritative_keys or key not in enriched):
            enriched[key] = promotion_row[key]
    return enriched


def decision_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("suite", "")),
        str(row.get("kernel", "")),
        str(row.get("shape", "")),
        str(row.get("selected_stack", "")),
    )


def trainer_selection_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("suite", "")),
        str(row.get("kernel", "")),
        str(row.get("shape", "")),
    )


def require_native_rows_trainer_callable(rows: list[dict[str, Any]]) -> None:
    bad_rows = [
        row
        for row in rows
        if not row.get("trainer_call_path_available")
    ]
    if bad_rows:
        labels = [
            f"{row.get('suite')}/{row.get('kernel')}/{row.get('shape')}/{row.get('selected_stack')}"
            for row in bad_rows[:5]
        ]
        raise ValueError(
            "native selection round contains non-trainer-callable rows: "
            + ", ".join(labels)
        )


def require_optional_non_trainer_decisions(
    optional_rows: list[dict[str, Any]],
    promotion_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    non_trainer_rows = [
        row
        for row in optional_rows
        if not row.get("trainer_call_path_available")
    ]
    promotion_by_key = {decision_key(row): row for row in promotion_rows}
    missing_promotion_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    for row in non_trainer_rows:
        promotion_row = promotion_by_key.get(decision_key(row))
        if promotion_row is None:
            missing_promotion_rows.append(row)
            promotion_row = {}
        decision_status = row.get("decision_status") or promotion_row.get("decision_status")
        decision_active = row.get("decision_active")
        if decision_active is None:
            decision_active = promotion_row.get("decision_active")
        if decision_status is None or decision_active is not False:
            unresolved_rows.append(row)
    if missing_promotion_rows:
        labels = [
            f"{row.get('suite')}/{row.get('kernel')}/{row.get('shape')}/{row.get('selected_stack')}"
            for row in missing_promotion_rows[:5]
        ]
        raise ValueError(
            "optional non-trainer selected rows lack matching promotion rows: "
            + ", ".join(labels)
        )
    if unresolved_rows:
        labels = [
            f"{row.get('suite')}/{row.get('kernel')}/{row.get('shape')}/{row.get('selected_stack')}"
            for row in unresolved_rows[:5]
        ]
        raise ValueError(
            "optional non-trainer selected rows lack inactive decisions: "
            + ", ".join(labels)
        )
    return non_trainer_rows, [promotion_by_key[decision_key(row)] for row in non_trainer_rows]


def effective_native_row(row: dict[str, Any]) -> dict[str, Any]:
    effective = slim_row(row)
    if row.get("decision_active") is False:
        next_stack = row.get("next_stack")
        next_time_us = row.get("next_time_us")
        if not next_stack or next_time_us is None:
            raise ValueError(
                "inactive selected native row lacks a next-stack fallback: "
                f"{row.get('suite')}/{row.get('kernel')}/{row.get('shape')}"
            )
        effective["rejected_selected_stack"] = row.get("selected_stack")
        effective["rejected_selected_time_us"] = row.get("selected_time_us")
        effective["selected_stack"] = next_stack
        effective["selected_time_us"] = next_time_us
        effective["effective_selection_reason"] = (
            "selected microbenchmark row is inactive in the decision registry; "
            "using the next stack as the current trainer default"
        )
    else:
        effective["effective_selection_reason"] = "selected row is trainer-callable and active"
    return effective


def has_inactive_decision(row: dict[str, Any]) -> bool:
    return row.get("decision_status") is not None and row.get("decision_active") is False


TRAINER_CALLABLE_DEBT_EVIDENCE_TERMS = (
    "x10",
    "tinystories",
    "trainer",
    "training smoke",
    "avg_ms",
    "total average iteration time",
    "stability",
)


def has_trainer_callable_debt_evidence(row: dict[str, Any]) -> bool:
    evidence = row.get("decision_evidence")
    if not isinstance(evidence, list) or not evidence:
        return False
    text = "\n".join(
        str(item)
        for item in (
            list(evidence)
            + [
                row.get("decision_decision", ""),
                row.get("decision_note", ""),
                row.get("promotion_gate", ""),
            ]
        )
    ).lower()
    return any(term in text for term in TRAINER_CALLABLE_DEBT_EVIDENCE_TERMS)


def project_extra_reason(row: dict[str, Any]) -> str:
    suite = str(row.get("suite", ""))
    kernel = str(row.get("kernel", ""))
    shape = str(row.get("shape", ""))
    use_scope = str(row.get("use_scope", ""))
    if suite == "layernorm" and shape != LAYERNORM_SELECTION_SHAPE:
        return "non-objective LayerNorm stress width; GPT-2 trainer LayerNorm uses C=768"
    if suite == "layernorm" and "partial backward" in use_scope:
        return "partial backward decomposition row; not the full trainer LayerNorm backward contract"
    if suite == "runtime" and kernel == "adamw_update" and "fp32-state" in shape:
        return "optimizer contract variant; current trainer objective row is no-master AdamW without this shape suffix"
    if suite == "runtime" and kernel == "adamw_update_bf16_state":
        return "non-equivalent BF16-state optimizer reference; current trainer objective uses FP32 moment buffers"
    if has_inactive_decision(row):
        return f"non-objective/reference row with inactive decision {row.get('decision_status')}"
    return "non-objective benchmark row retained as comparison evidence"


def faster_native_project_row(
    optional_row: dict[str, Any],
    native_row: dict[str, Any] | None,
) -> dict[str, Any]:
    row = slim_row(optional_row)
    if native_row is None:
        row["project_selection_source"] = "optional"
        return row
    optional_time = optional_row.get("selected_time_us")
    native_time = native_row.get("selected_time_us")
    if optional_time is None or native_time is None:
        row["project_selection_source"] = "optional"
        return row
    if float(native_time) <= float(optional_time):
        selected = dict(native_row)
        selected["project_selection_source"] = "native"
        selected["optional_compared_stack"] = optional_row.get("selected_stack")
        selected["optional_compared_time_us"] = optional_time
        return selected
    row["project_selection_source"] = "optional"
    row["native_compared_stack"] = native_row.get("selected_stack")
    row["native_compared_time_us"] = native_time
    return row


def build_summary(
    native_round: Path,
    optional_round: Path,
    *,
    allow_benchmark_only_native: bool = False,
) -> dict[str, Any]:
    native_selected = selected_payload(native_round)
    optional_selected = selected_payload(optional_round)
    optional_promotions = promotion_payload(optional_round)
    training_evidence = native_training_evidence(native_round, allow_benchmark_only_native)

    native_rows = native_selected.get("selected_backend_rows", [])
    optional_rows = optional_selected.get("selected_backend_rows", [])
    if not isinstance(native_rows, list) or not isinstance(optional_rows, list):
        raise ValueError("selected-backends payload has invalid selected_backend_rows")
    expected_native_keys = expected_trainer_selection_keys()
    native_trainer_rows = [
        row for row in native_rows if trainer_selection_key(row) in expected_native_keys
    ]
    extra_native_rows = [
        row for row in native_rows if trainer_selection_key(row) not in expected_native_keys
    ]
    missing_native_keys = expected_native_keys - {
        trainer_selection_key(row) for row in native_trainer_rows
    }
    if missing_native_keys:
        labels = [
            f"{suite}/{kernel}/{shape}"
            for suite, kernel, shape in sorted(missing_native_keys)[:5]
        ]
        raise ValueError(
            "native selection round is missing required trainer rows: "
            + ", ".join(labels)
        )
    require_native_rows_trainer_callable(native_trainer_rows)
    effective_native_rows = [effective_native_row(row) for row in native_trainer_rows]

    promotion_rows = optional_promotions.get("promotion_candidates", [])
    if not isinstance(promotion_rows, list):
        raise ValueError("promotion-candidates payload has invalid promotion_candidates")
    optional_non_trainer_rows, optional_non_trainer_promotion_rows = (
        require_optional_non_trainer_decisions(optional_rows, promotion_rows)
    )
    native_attention_route_rows = native_selected.get("attention_route_rows", [])
    optional_attention_route_rows = optional_selected.get("attention_route_rows", [])
    if not isinstance(native_attention_route_rows, list):
        raise ValueError("native selected-backends payload has invalid attention_route_rows")
    if not isinstance(optional_attention_route_rows, list):
        raise ValueError("optional selected-backends payload has invalid attention_route_rows")

    active_optional = optional_promotions.get("active_promotion_candidates", [])
    if active_optional:
        raise ValueError(
            f"optional-stack round still has {len(active_optional)} active promotion candidates"
        )

    promotion_by_key = {decision_key(row): row for row in promotion_rows}
    decision_rows = [
        row
        for row in optional_rows
        if row.get("decision_status") is not None
    ]
    promotion_decision_rows = [
        row
        for row in promotion_rows
        if row.get("decision_status") is not None
    ]
    resolved_decision_rows = [
        with_promotion_metadata(row, promotion_by_key.get(decision_key(row), {}))
        for row in decision_rows
    ]
    resolved_keys = {decision_key(row) for row in resolved_decision_rows}
    for row in promotion_decision_rows:
        key = decision_key(row)
        if key not in resolved_keys:
            resolved_decision_rows.append(row)
            resolved_keys.add(key)
    operator_decision_rows = [
        row
        for row in resolved_decision_rows
        if not row.get("trainer_call_path_available")
    ]
    trainer_decision_rows = [
        row
        for row in resolved_decision_rows
        if row.get("trainer_call_path_available")
    ]

    native_inactive_rows = [
        row for row in native_trainer_rows if row.get("decision_active") is False
    ]
    effective_native_by_key = {
        trainer_selection_key(row): row
        for row in effective_native_rows
    }
    project_fastest_rows = [
        faster_native_project_row(
            row,
            effective_native_by_key.get(trainer_selection_key(row))
            if trainer_selection_key(row) in expected_native_keys
            else None,
        )
        for row in optional_rows
    ]
    project_torch_fastest_rows = [
        row for row in project_fastest_rows if str(row.get("selected_stack", "")).startswith("Torch")
    ]
    project_trainer_callable_rows = [
        row for row in project_fastest_rows if row.get("trainer_call_path_available")
    ]
    project_fastest_extra_rows: list[dict[str, Any]] = []
    project_fastest_used_rows: list[dict[str, Any]] = []
    project_fastest_resolved_divergence_rows: list[dict[str, Any]] = []
    project_fastest_unresolved_objective_rows: list[dict[str, Any]] = []
    for row in project_fastest_rows:
        row_key = trainer_selection_key(row)
        native_row = effective_native_by_key.get(row_key)
        if row_key not in expected_native_keys:
            row["project_extra_reason"] = project_extra_reason(row)
            project_fastest_extra_rows.append(row)
        elif native_row is not None and native_row.get("selected_stack") == row.get("selected_stack"):
            project_fastest_used_rows.append(row)
        elif has_inactive_decision(row):
            project_fastest_resolved_divergence_rows.append(row)
        else:
            project_fastest_unresolved_objective_rows.append(row)
    if project_fastest_unresolved_objective_rows:
        labels = [
            f"{row.get('suite')}/{row.get('kernel')}/{row.get('shape')}/{row.get('selected_stack')}"
            for row in project_fastest_unresolved_objective_rows[:5]
        ]
        raise ValueError(
            "project-wide fastest objective rows are neither trainer-selected nor resolved: "
            + ", ".join(labels)
        )
    project_fastest_used_keys = {decision_key(row) for row in project_fastest_used_rows}
    project_fastest_resolved_keys = {
        decision_key(row) for row in project_fastest_resolved_divergence_rows
    }
    project_fastest_extra_keys = {decision_key(row) for row in project_fastest_extra_rows}
    project_torch_fastest_used_rows = [
        row for row in project_torch_fastest_rows if decision_key(row) in project_fastest_used_keys
    ]
    project_torch_fastest_resolved_rows = [
        row for row in project_torch_fastest_rows if decision_key(row) in project_fastest_resolved_keys
    ]
    project_torch_fastest_extra_rows = [
        row for row in project_torch_fastest_rows if decision_key(row) in project_fastest_extra_keys
    ]
    project_torch_fastest_partitioned_row_count = (
        len(project_torch_fastest_used_rows)
        + len(project_torch_fastest_resolved_rows)
        + len(project_torch_fastest_extra_rows)
    )
    resolved_decision_by_key = {decision_key(row): row for row in resolved_decision_rows}
    project_torch_fastest_disposition_rows: list[dict[str, Any]] = []
    for row in project_torch_fastest_rows:
        enriched = dict(row)
        key = decision_key(row)
        if key in project_fastest_used_keys:
            enriched["torch_disposition"] = "trainer_used"
            enriched["torch_action"] = "selected in the current trainer mix"
        elif key in project_fastest_resolved_keys:
            resolved_row = resolved_decision_by_key.get(key, {})
            enriched["torch_disposition"] = "resolved_away"
            for field in (
                "candidate_class",
                "promotion_gate",
                "priority",
                "decision_decision",
                "decision_evidence",
            ):
                if field in resolved_row and field not in enriched:
                    enriched[field] = resolved_row[field]
            enriched["torch_action"] = str(
                resolved_row.get("promotion_gate")
                or resolved_row.get("decision_decision")
                or row.get("decision_decision")
                or ""
            )
        elif key in project_fastest_extra_keys:
            enriched["torch_disposition"] = "extra_benchmark"
            enriched["torch_action"] = str(row.get("project_extra_reason", ""))
        else:
            enriched["torch_disposition"] = "unpartitioned"
            enriched["torch_action"] = ""
        project_torch_fastest_disposition_rows.append(enriched)
    project_torch_fastest_missing_disposition = [
        (
            f"{row.get('suite')}/{row.get('kernel')}/"
            f"{row.get('shape')}/{row.get('selected_stack')}"
        )
        for row in project_torch_fastest_disposition_rows
        if not row.get("torch_disposition")
        or row.get("torch_disposition") == "unpartitioned"
        or not row.get("torch_action")
    ]
    stack_counts = Counter(str(row.get("selected_stack", "unknown")) for row in effective_native_rows)
    suite_counts = Counter(str(row.get("suite", "unknown")) for row in effective_native_rows)
    optional_decision_status_counts = Counter(
        str(row.get("decision_status", "unknown")) for row in resolved_decision_rows
    )
    project_resolved_call_path_counts = Counter(
        str(row.get("trainer_call_path_kind", "unknown"))
        for row in project_fastest_resolved_divergence_rows
    )
    project_resolved_status_counts = Counter(
        str(row.get("decision_status", "unknown"))
        for row in project_fastest_resolved_divergence_rows
    )
    project_resolved_trainer_callable_rows = [
        row
        for row in project_fastest_resolved_divergence_rows
        if row.get("trainer_call_path_kind") == "trainer_or_cxx_route"
    ]
    project_resolved_trainer_callable_rows_with_evidence = [
        row
        for row in project_resolved_trainer_callable_rows
        if has_trainer_callable_debt_evidence(row)
    ]
    project_resolved_trainer_callable_rows_missing_evidence = [
        (
            f"{row.get('suite')}/{row.get('kernel')}/"
            f"{row.get('shape')}/{row.get('selected_stack')}"
        )
        for row in project_resolved_trainer_callable_rows
        if not has_trainer_callable_debt_evidence(row)
    ]
    project_resolved_missing_decision_rows = [
        row
        for row in project_fastest_resolved_divergence_rows
        if decision_key(row) not in resolved_decision_by_key
    ]
    project_resolved_non_trainer_rows = [
        row
        for row in project_fastest_resolved_divergence_rows
        if row.get("trainer_call_path_available") is False
    ]
    project_resolved_non_trainer_actionable_rows = [
        row
        for row in project_resolved_non_trainer_rows
        if (
            (resolved_row := resolved_decision_by_key.get(decision_key(row)))
            and resolved_row.get("candidate_class")
            and resolved_row.get("promotion_gate")
            and resolved_row.get("priority")
        )
    ]
    project_resolved_non_trainer_missing_action_rows = [
        (
            f"{row.get('suite')}/{row.get('kernel')}/"
            f"{row.get('shape')}/{row.get('selected_stack')}"
        )
        for row in project_resolved_non_trainer_rows
        if row not in project_resolved_non_trainer_actionable_rows
    ]

    return {
        "schema_version": 1,
        "native_selection_round": str(native_round),
        "optional_stack_round": str(optional_round),
        "native_training_evidence": training_evidence,
        "native_run_label": native_selected.get("run_label", "unknown"),
        "optional_run_label": optional_selected.get("run_label", "unknown"),
        "native_benchmark_row_count": native_selected.get("benchmark_row_count"),
        "optional_benchmark_row_count": optional_selected.get("benchmark_row_count"),
        "native_source_selected_row_count": len(native_rows),
        "native_selected_row_count": len(native_trainer_rows),
        "native_extra_selected_row_count": len(extra_native_rows),
        "native_inactive_selected_row_count": len(native_inactive_rows),
        "native_selected_stack_counts": dict(sorted(stack_counts.items())),
        "native_selected_suite_counts": dict(sorted(suite_counts.items())),
        "optional_selected_row_count": len(optional_rows),
        "optional_non_trainer_selected_row_count": len(optional_non_trainer_rows),
        "optional_non_trainer_promotion_row_count": len(optional_non_trainer_promotion_rows),
        "optional_decision_row_count": len(resolved_decision_rows),
        "optional_selected_decision_row_count": len(decision_rows),
        "optional_promotion_decision_row_count": len(promotion_decision_rows),
        "optional_operator_decision_row_count": len(operator_decision_rows),
        "optional_trainer_callable_decision_row_count": len(trainer_decision_rows),
        "optional_decision_status_counts": dict(sorted(optional_decision_status_counts.items())),
        "project_fastest_row_count": len(project_fastest_rows),
        "project_torch_fastest_row_count": len(project_torch_fastest_rows),
        "project_torch_fastest_used_row_count": len(project_torch_fastest_used_rows),
        "project_torch_fastest_resolved_divergence_row_count": len(project_torch_fastest_resolved_rows),
        "project_torch_fastest_extra_row_count": len(project_torch_fastest_extra_rows),
        "project_torch_fastest_partitioned_row_count": project_torch_fastest_partitioned_row_count,
        "project_torch_fastest_disposition_row_count": len(project_torch_fastest_disposition_rows),
        "project_torch_fastest_actionable_row_count": (
            len(project_torch_fastest_disposition_rows) - len(project_torch_fastest_missing_disposition)
        ),
        "project_torch_fastest_missing_disposition": project_torch_fastest_missing_disposition,
        "project_trainer_callable_row_count": len(project_trainer_callable_rows),
        "project_fastest_used_row_count": len(project_fastest_used_rows),
        "project_fastest_resolved_divergence_row_count": len(project_fastest_resolved_divergence_rows),
        "project_fastest_resolved_call_path_counts": dict(sorted(project_resolved_call_path_counts.items())),
        "project_fastest_resolved_status_counts": dict(sorted(project_resolved_status_counts.items())),
        "project_fastest_resolved_trainer_callable_row_count": len(project_resolved_trainer_callable_rows),
        "project_fastest_resolved_trainer_callable_evidence_count": (
            len(project_resolved_trainer_callable_rows_with_evidence)
        ),
        "project_fastest_resolved_trainer_callable_missing_evidence": (
            project_resolved_trainer_callable_rows_missing_evidence
        ),
        "project_fastest_resolved_decision_link_count": (
            len(project_fastest_resolved_divergence_rows)
            - len(project_resolved_missing_decision_rows)
        ),
        "project_fastest_resolved_missing_decision_links": [
            (
                f"{row.get('suite')}/{row.get('kernel')}/"
                f"{row.get('shape')}/{row.get('selected_stack')}"
            )
            for row in project_resolved_missing_decision_rows
        ],
        "project_fastest_resolved_non_trainer_row_count": len(project_resolved_non_trainer_rows),
        "project_fastest_resolved_non_trainer_actionable_count": len(
            project_resolved_non_trainer_actionable_rows
        ),
        "project_fastest_resolved_non_trainer_missing_action": (
            project_resolved_non_trainer_missing_action_rows
        ),
        "project_fastest_extra_row_count": len(project_fastest_extra_rows),
        "project_fastest_unresolved_objective_row_count": len(project_fastest_unresolved_objective_rows),
        "active_promotion_candidate_count": len(active_optional),
        "promotion_candidate_count": len(promotion_rows),
        "native_attention_route_row_count": len(native_attention_route_rows),
        "optional_attention_route_row_count": len(optional_attention_route_rows),
        "selection_policy": (
            "Use a native round with TinyStories training evidence as the current trainer backend mix. "
            "For each exact objective row, compare the optional-stack round against that native "
            "trainer row and publish the faster current observed row as the project-wide fastest "
            "selection, including Torch only where it still beats the current native evidence. "
            "Optional rows remain operator/reference evidence, or rejected trainer-callable "
            "microbench wins, unless a refreshed integration exposes a trainer call path and "
            "passes correctness plus TinyStories smoke gates. "
            "Every selected optional row without a trainer call path must have a matching "
            "inactive promotion decision before this artifact can be generated."
        ),
        "native_inactive_selected_rows": [slim_row(row) for row in native_inactive_rows],
        "native_extra_selected_rows": [slim_row(row) for row in extra_native_rows],
        "native_trainer_selection": effective_native_rows,
        "project_fastest_selection": project_fastest_rows,
        "project_torch_fastest_rows": project_torch_fastest_rows,
        "project_torch_fastest_disposition_rows": project_torch_fastest_disposition_rows,
        "project_fastest_used_rows": project_fastest_used_rows,
        "project_fastest_resolved_divergence_rows": project_fastest_resolved_divergence_rows,
        "project_fastest_extra_rows": project_fastest_extra_rows,
        "resolved_optional_stack_decisions": [slim_row(row) for row in resolved_decision_rows],
        "native_attention_route_rows": native_attention_route_rows,
        "optional_attention_route_rows": optional_attention_route_rows,
    }


def markdown_table(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    lines: list[str] = []
    header = rows[0]
    lines.append("| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(header)) + " |")
    lines.append("| " + " | ".join("-" * widths[index] for index in range(len(header))) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) + " |")
    return lines


def decision_evidence_summary(row: dict[str, Any]) -> str:
    evidence = row.get("decision_evidence")
    if not isinstance(evidence, list) or not evidence:
        return ""
    text = str(evidence[0]).replace("|", "/").replace("\n", " ")
    if len(evidence) > 1:
        text = f"{text} (+{len(evidence) - 1} more)"
    return text


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Current SM120 Backend Selection",
        "",
        f"- native selection round: `{summary['native_selection_round']}`",
        f"- optional-stack comparison round: `{summary['optional_stack_round']}`",
        f"- native training manifest: `{summary['native_training_evidence'].get('manifest_path', 'bypassed')}`",
        f"- native training log: `{summary['native_training_evidence'].get('train_log_path', 'bypassed')}`",
        f"- native selected rows: `{summary['native_selected_row_count']}`",
        f"- extra native benchmark-only selections: `{summary['native_extra_selected_row_count']}`",
        f"- inactive native microbench selections: `{summary['native_inactive_selected_row_count']}`",
        f"- optional non-trainer selected rows: `{summary['optional_non_trainer_selected_row_count']}`",
        f"- optional decision rows: `{summary['optional_decision_row_count']}`",
        f"- project-wide fastest rows: `{summary['project_fastest_row_count']}`",
        f"- project-wide Torch fastest rows: `{summary['project_torch_fastest_row_count']}`",
        (
            "- project-wide Torch fastest rows partition: "
            f"`{summary['project_torch_fastest_used_row_count']}` trainer-used, "
            f"`{summary['project_torch_fastest_resolved_divergence_row_count']}` resolved, "
            f"`{summary['project_torch_fastest_extra_row_count']}` extra"
        ),
        (
            "- project-wide Torch disposition rows with action/reason: "
            f"`{summary['project_torch_fastest_actionable_row_count']}`/"
            f"`{summary['project_torch_fastest_disposition_row_count']}`"
        ),
        f"- project-wide trainer-callable fastest rows: `{summary['project_trainer_callable_row_count']}`",
        f"- project-wide fastest rows used by trainer: `{summary['project_fastest_used_row_count']}`",
        f"- project-wide fastest rows resolved away from trainer: `{summary['project_fastest_resolved_divergence_row_count']}`",
        f"- project-wide extra benchmark rows: `{summary['project_fastest_extra_row_count']}`",
        f"- active promotion candidates: `{summary['active_promotion_candidate_count']}`",
        "",
        summary["selection_policy"],
        "",
        "## Native Trainer Mix",
        "",
    ]
    stack_rows = [["Stack", "Selected rows"]]
    for stack, count in summary["native_selected_stack_counts"].items():
        stack_rows.append([stack, str(count)])
    lines.extend(markdown_table(stack_rows))
    lines.extend(["", "## Optional-Stack Decisions", ""])
    decision_rows = [["Status", "Rows"]]
    for status, count in summary["optional_decision_status_counts"].items():
        decision_rows.append([status, str(count)])
    lines.extend(markdown_table(decision_rows))
    lines.extend(["", "## Fastest Rows Not Used By Trainer", ""])
    debt_rows = [["Call path", "Rows"]]
    for call_path, count in summary["project_fastest_resolved_call_path_counts"].items():
        debt_rows.append([call_path, str(count)])
    lines.extend(markdown_table(debt_rows))
    lines.extend(["", "### Decision Statuses", ""])
    debt_status_rows = [["Decision", "Rows"]]
    for status, count in summary["project_fastest_resolved_status_counts"].items():
        debt_status_rows.append([status, str(count)])
    lines.extend(markdown_table(debt_status_rows))
    lines.extend(
        [
            "",
            (
                "- trainer/C++ callable resolved rows with stability evidence: "
                f"`{summary['project_fastest_resolved_trainer_callable_evidence_count']}`/"
                f"`{summary['project_fastest_resolved_trainer_callable_row_count']}`"
            ),
            (
                "- resolved rows linked to decision table: "
                f"`{summary['project_fastest_resolved_decision_link_count']}`/"
                f"`{summary['project_fastest_resolved_divergence_row_count']}`"
            ),
            (
                "- non-trainer resolved rows with action metadata: "
                f"`{summary['project_fastest_resolved_non_trainer_actionable_count']}`/"
                f"`{summary['project_fastest_resolved_non_trainer_row_count']}`"
            ),
        ]
    )
    missing_trainer_callable_evidence = summary.get(
        "project_fastest_resolved_trainer_callable_missing_evidence",
        [],
    )
    if missing_trainer_callable_evidence:
        lines.append(
            "- trainer/C++ callable rows missing stability evidence: "
            + ", ".join(f"`{label}`" for label in missing_trainer_callable_evidence)
        )
    missing_decision_links = summary.get("project_fastest_resolved_missing_decision_links", [])
    if missing_decision_links:
        lines.append(
            "- resolved rows missing decision-table link: "
            + ", ".join(f"`{label}`" for label in missing_decision_links)
        )
    missing_action_rows = summary.get("project_fastest_resolved_non_trainer_missing_action", [])
    if missing_action_rows:
        lines.append(
            "- non-trainer resolved rows missing action metadata: "
            + ", ".join(f"`{label}`" for label in missing_action_rows)
        )
    lines.extend(["", "## Project-Wide Torch Fastest Rows", ""])
    torch_rows = [["Suite", "Kernel", "Shape", "Selected stack", "Time (us)", "Scope", "Call path"]]
    for row in summary["project_torch_fastest_rows"]:
        selected_time = row.get("selected_time_us")
        torch_rows.append(
            [
                str(row.get("suite", "")),
                str(row.get("kernel", "")),
                f"`{row.get('shape', '')}`",
                str(row.get("selected_stack", "")),
                f"{float(selected_time):.3f}" if selected_time is not None else "",
                str(row.get("use_scope", "")),
                str(row.get("trainer_call_path_kind", "unknown")),
            ]
        )
    lines.extend(markdown_table(torch_rows))
    lines.extend(["", "## Project-Wide Torch Fastest Row Disposition", ""])
    torch_disposition_rows = [
        ["Suite", "Kernel", "Shape", "Selected stack", "Disposition", "Class/Reason", "Action/Gate"]
    ]
    for row in summary["project_torch_fastest_disposition_rows"]:
        class_or_reason = str(row.get("candidate_class") or row.get("project_extra_reason") or "")
        torch_disposition_rows.append(
            [
                str(row.get("suite", "")),
                str(row.get("kernel", "")),
                f"`{row.get('shape', '')}`",
                str(row.get("selected_stack", "")),
                str(row.get("torch_disposition", "")),
                class_or_reason.replace("|", "/").replace("\n", " "),
                str(row.get("torch_action", "")).replace("|", "/").replace("\n", " "),
            ]
        )
    lines.extend(markdown_table(torch_disposition_rows))
    lines.extend(["", "## Project-Wide Fastest Rows", ""])
    fastest_rows = [["Suite", "Kernel", "Shape", "Selected stack", "Time (us)", "Scope", "Call path"]]
    for row in summary["project_fastest_selection"]:
        selected_time = row.get("selected_time_us")
        fastest_rows.append(
            [
                str(row.get("suite", "")),
                str(row.get("kernel", "")),
                f"`{row.get('shape', '')}`",
                str(row.get("selected_stack", "")),
                f"{float(selected_time):.3f}" if selected_time is not None else "",
                str(row.get("use_scope", "")),
                str(row.get("trainer_call_path_kind", "unknown")),
            ]
        )
    lines.extend(markdown_table(fastest_rows))
    lines.extend(["", "## Project-Wide Fastest Rows Used By Trainer", ""])
    used_rows = [["Suite", "Kernel", "Shape", "Selected stack", "Time (us)", "Scope", "Call path"]]
    for row in summary["project_fastest_used_rows"]:
        selected_time = row.get("selected_time_us")
        used_rows.append(
            [
                str(row.get("suite", "")),
                str(row.get("kernel", "")),
                f"`{row.get('shape', '')}`",
                str(row.get("selected_stack", "")),
                f"{float(selected_time):.3f}" if selected_time is not None else "",
                str(row.get("use_scope", "")),
                str(row.get("trainer_call_path_kind", "unknown")),
            ]
        )
    lines.extend(markdown_table(used_rows))
    lines.extend(["", "## Project-Wide Fastest Rows Resolved Away From Trainer", ""])
    divergence_rows = [["Suite", "Kernel", "Shape", "Selected stack", "Time (us)", "Call path", "Decision", "Reason", "Evidence"]]
    for row in summary["project_fastest_resolved_divergence_rows"]:
        selected_time = row.get("selected_time_us")
        divergence_rows.append(
            [
                str(row.get("suite", "")),
                str(row.get("kernel", "")),
                f"`{row.get('shape', '')}`",
                str(row.get("selected_stack", "")),
                f"{float(selected_time):.3f}" if selected_time is not None else "",
                str(row.get("trainer_call_path_kind", "unknown")),
                str(row.get("decision_status", "")),
                str(row.get("decision_decision", "")).replace("|", "/").replace("\n", " "),
                decision_evidence_summary(row),
            ]
        )
    lines.extend(markdown_table(divergence_rows))
    lines.extend(["", "## Extra Project-Wide Benchmark Rows", ""])
    extra_project_rows = [["Suite", "Kernel", "Shape", "Selected stack", "Time (us)", "Call path", "Reason"]]
    for row in summary["project_fastest_extra_rows"]:
        selected_time = row.get("selected_time_us")
        extra_project_rows.append(
            [
                str(row.get("suite", "")),
                str(row.get("kernel", "")),
                f"`{row.get('shape', '')}`",
                str(row.get("selected_stack", "")),
                f"{float(selected_time):.3f}" if selected_time is not None else "",
                str(row.get("trainer_call_path_kind", "unknown")),
                str(row.get("project_extra_reason", "")),
            ]
        )
    lines.extend(markdown_table(extra_project_rows))
    lines.extend(["", "## Resolved Optional-Stack Decisions", ""])
    resolved_rows = [
        ["Suite", "Kernel", "Shape", "Selected stack", "Time (us)", "Scope", "Call path", "Class", "Gate", "Decision"]
    ]
    for row in summary["resolved_optional_stack_decisions"]:
        selected_time = row.get("selected_time_us")
        resolved_rows.append(
            [
                str(row.get("suite", "")),
                str(row.get("kernel", "")),
                f"`{row.get('shape', '')}`",
                str(row.get("selected_stack", "")),
                f"{float(selected_time):.3f}" if selected_time is not None else "",
                str(row.get("use_scope", "")),
                str(row.get("trainer_call_path_kind", "unknown")),
                str(row.get("candidate_class", "")),
                str(row.get("promotion_gate", "")),
                str(row.get("decision_status", "")),
            ]
        )
    lines.extend(markdown_table(resolved_rows))
    if summary["native_inactive_selected_rows"]:
        lines.extend(["", "## Inactive Native Microbench Selections", ""])
        inactive_rows = [["Suite", "Kernel", "Shape", "Rejected stack", "Current stack", "Decision"]]
        effective_by_key = {
            (row.get("suite"), row.get("kernel"), row.get("shape"), row.get("rejected_selected_stack")): row
            for row in summary["native_trainer_selection"]
            if "rejected_selected_stack" in row
        }
        for row in summary["native_inactive_selected_rows"]:
            effective = effective_by_key[
                (row.get("suite"), row.get("kernel"), row.get("shape"), row.get("selected_stack"))
            ]
            inactive_rows.append(
                [
                    str(row.get("suite", "")),
                    str(row.get("kernel", "")),
                    f"`{row.get('shape', '')}`",
                    str(row.get("selected_stack", "")),
                    str(effective.get("selected_stack", "")),
                    str(row.get("decision_status", "")),
                ]
            )
        lines.extend(markdown_table(inactive_rows))
    if summary["native_attention_route_rows"] or summary["optional_attention_route_rows"]:
        lines.extend(["", "## Attention Route Totals", ""])
        route_rows = [
            [
                "Source",
                "Stack",
                "Shape",
                "Scope",
                "Trainer-layout",
                "Forward (us)",
                "Backward (us)",
                "Total (us)",
                "Complete",
                "Note",
            ]
        ]
        for source, routes in (
            ("native", summary["native_attention_route_rows"]),
            ("optional", summary["optional_attention_route_rows"]),
        ):
            for row in routes:
                route_rows.append(
                    [
                        source,
                        str(row.get("stack", "")),
                        f"`{row.get('shape', '')}`",
                        str(row.get("route_scope", "")),
                        str(bool(row.get("trainer_layout"))),
                        "-" if row.get("forward_us") is None else f"{float(row['forward_us']):.3f}",
                        "-" if row.get("backward_us") is None else f"{float(row['backward_us']):.3f}",
                        "-" if row.get("total_us") is None else f"{float(row['total_us']):.3f}",
                        str(bool(row.get("complete"))),
                        str(row.get("unavailable_reason") or ""),
                    ]
                )
        lines.extend(markdown_table(route_rows))
    if summary["native_extra_selected_rows"]:
        lines.extend(["", "## Extra Native Benchmark Selections", ""])
        extra_rows = [["Suite", "Kernel", "Shape", "Selected stack", "Time (us)", "Call path"]]
        for row in summary["native_extra_selected_rows"]:
            selected_time = row.get("selected_time_us")
            extra_rows.append(
                [
                    str(row.get("suite", "")),
                    str(row.get("kernel", "")),
                    f"`{row.get('shape', '')}`",
                    str(row.get("selected_stack", "")),
                    f"{float(selected_time):.3f}" if selected_time is not None else "",
                    str(row.get("trainer_call_path_kind", "unknown")),
                ]
            )
        lines.extend(markdown_table(extra_rows))
    lines.extend(["", "## Trainer Selection Rows", ""])
    rows = [["Suite", "Kernel", "Shape", "Current stack", "Time (us)", "Call path"]]
    for row in summary["native_trainer_selection"]:
        rows.append(
            [
                str(row.get("suite", "")),
                str(row.get("kernel", "")),
                f"`{row.get('shape', '')}`",
                str(row.get("selected_stack", "")),
                f"{float(row.get('selected_time_us', 0.0)):.3f}",
                str(row.get("trainer_call_path_kind", "unknown")),
            ]
        )
    lines.extend(markdown_table(rows))
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def write_selected_payload(round_dir: Path, rows: list[dict[str, Any]]) -> None:
    attention_route_rows: list[dict[str, Any]] = []
    grouped_attention: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("suite") != "attention":
            continue
        key = (str(row.get("shape", "")), str(row.get("selected_stack", "")))
        grouped_attention.setdefault(key, {})[str(row.get("kernel", ""))] = row.get("selected_time_us")
    for (shape, stack), timings in sorted(grouped_attention.items()):
        forward_us = timings.get("forward")
        backward_us = timings.get("backward")
        trainer_layout = stack in {"TK packed-QKV", "TorchPacked", "TorchMaterializedPacked", "cuDNNPacked"}
        attention_route_rows.append(
            {
                "shape": shape,
                "stack": stack,
                "route_scope": "packed trainer-layout route" if trainer_layout else "attention route",
                "trainer_layout": trainer_layout,
                "forward_us": forward_us,
                "backward_us": backward_us,
                "total_us": (forward_us + backward_us) if forward_us is not None and backward_us is not None else None,
                "complete": forward_us is not None and backward_us is not None,
                "unavailable_reason": None,
            }
        )
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "selected-backends.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_label": round_dir.name,
                "benchmark_row_count": len(rows),
                "selected_backend_rows": rows,
                "attention_route_rows": attention_route_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_manifest_payload(round_dir: Path, run_training: str = "1", max_steps: str = "10") -> None:
    (round_dir / "round-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config": {
                    "run_training": run_training,
                    "max_steps": max_steps,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_train_log(round_dir: Path) -> None:
    (round_dir / "train_gpt2cu.log").write_text(
        "step 1/10 | loss 1.0 | norm 1.0 | lr 0.1 | 1.0 ms | 0.0% bf16 MFU | 1 tok/s\n"
        "total average iteration time: 1.000 ms\n"
    )


def write_promotion_payload(round_dir: Path, active_count: int = 0) -> None:
    active = [
        {
            "suite": "runtime",
            "kernel": "synthetic",
            "shape": "synthetic",
            "selected_stack": "Triton",
        }
        for _ in range(active_count)
    ]
    (round_dir / "promotion-candidates.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "promotion_candidates": list(active),
                "active_promotion_candidates": active,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def synthetic_native_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite, kernel, shape in sorted(expected_trainer_selection_keys()):
        row: dict[str, Any] = {
            "suite": suite,
            "kernel": kernel,
            "shape": shape,
            "selected_stack": "CUDA",
            "selected_time_us": 1000.0,
            "next_stack": "cuBLASLt",
            "next_time_us": 1050.0,
            "use_scope": "C++ benchmark route",
            "trainer_call_path_available": True,
            "trainer_call_path_kind": "trainer_or_cxx_route",
        }
        if suite == "matmul" and kernel == "fwd" and str(shape).startswith("qkv "):
            row["selected_stack"] = "cuBLASLt"
            row["next_stack"] = "TK"
        if suite == "matmul" and kernel == "fwd" and str(shape).startswith("lmhead "):
            row.update(
                {
                    "selected_stack": "cuBLAS",
                    "selected_time_us": 22000.0,
                    "next_stack": "cuBLASLt",
                    "next_time_us": 22200.0,
                    "decision_status": "rejected_trainer_smoke",
                    "decision_active": False,
                    "decision_decision": "synthetic inactive row",
                    "decision_evidence": ["synthetic evidence"],
                }
            )
        if suite == "runtime" and (
            kernel == "cuda_copy_d2d"
            or (kernel == "cuda_memset" and shape == "logits_elems=3296722944")
        ):
            row["trainer_call_path_kind"] = "profiler_runtime_benchmark_only"
        rows.append(row)
    rows.append(
        {
            "suite": "matmul",
            "kernel": "synthetic_extra",
            "shape": "not an objective trainer row",
            "selected_stack": "CUDA",
            "selected_time_us": 1.0,
            "next_stack": None,
            "next_time_us": None,
            "use_scope": "C++ benchmark route",
            "trainer_call_path_available": True,
            "trainer_call_path_kind": "trainer_or_cxx_route",
        }
    )
    return rows


def synthetic_optional_rows() -> list[dict[str, Any]]:
    return [
        {
            "suite": "runtime",
            "kernel": "global_norm_squared",
            "shape": "params=124475904",
            "selected_stack": "CUDA",
            "selected_time_us": 990.0,
            "next_stack": "Torch",
            "next_time_us": 1100.0,
            "use_scope": "CUDA benchmark route",
            "trainer_call_path_available": True,
            "trainer_call_path_kind": "trainer_or_cxx_route",
        },
        {
            "suite": "runtime",
            "kernel": "gelu_forward",
            "shape": "BT=65536 C=3072",
            "selected_stack": "Torch",
            "selected_time_us": 530.0,
            "next_stack": "CUDA",
            "next_time_us": 535.0,
            "use_scope": "operator prototype",
            "trainer_call_path_available": False,
            "trainer_call_path_kind": "operator_or_reference_prototype",
            "decision_status": "rejected_same_session_refresh",
            "decision_active": False,
            "decision_decision": "synthetic resolved optional row",
            "decision_evidence": ["synthetic evidence"],
        }
    ]


def synthetic_promotion_rows() -> list[dict[str, Any]]:
    return [
        {
            "suite": "runtime",
            "kernel": "gelu_forward",
            "shape": "BT=65536 C=3072",
            "selected_stack": "Torch",
            "selected_time_us": 530.0,
            "next_stack": "CUDA",
            "next_time_us": 535.0,
            "use_scope": "operator prototype",
            "trainer_call_path_available": False,
            "trainer_call_path_kind": "operator_or_reference_prototype",
            "decision_status": "rejected_same_session_refresh",
            "decision_active": False,
            "decision_decision": "synthetic resolved optional row",
            "decision_evidence": ["synthetic evidence"],
            "decision_note": "synthetic decision note",
            "candidate_class": "library integration",
            "promotion_gate": "synthetic refresh plus smoke gate",
            "priority": "medium",
            "speedup_vs_next_pct": 1.0,
        },
        {
            "suite": "runtime",
            "kernel": "adamw_update_bf16_state",
            "shape": "params=124475904 no-master",
            "selected_stack": "Torch",
            "selected_time_us": 1200.0,
            "next_stack": None,
            "next_time_us": None,
            "use_scope": "non-equivalent BF16-state reference",
            "trainer_call_path_available": False,
            "trainer_call_path_kind": "operator_or_reference_prototype",
            "decision_status": "contract_mismatch",
            "decision_active": False,
            "decision_decision": "synthetic promotion-only decision",
            "decision_evidence": ["synthetic evidence"],
            "decision_note": "synthetic contract note",
            "candidate_class": "contract mismatch",
            "promotion_gate": "no promotion until contract matches trainer state",
            "priority": "low",
        }
    ]


def write_promotion_payload_with_rows(round_dir: Path, rows: list[dict[str, Any]]) -> None:
    (round_dir / "promotion-candidates.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "promotion_candidates": rows,
                "active_promotion_candidates": [
                    row for row in rows if row.get("decision_active") is not False
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="sm120_current_selection_") as tmp:
        root = Path(tmp)
        native_dir = root / "native"
        optional_dir = root / "optional"
        write_selected_payload(native_dir, synthetic_native_rows())
        write_manifest_payload(native_dir)
        write_train_log(native_dir)
        write_selected_payload(optional_dir, synthetic_optional_rows())
        write_promotion_payload_with_rows(optional_dir, synthetic_promotion_rows())

        summary = build_summary(native_dir, optional_dir)
        if summary["optional_decision_row_count"] != 2:
            raise RuntimeError("self-test did not merge selected and promotion-only decisions")
        if summary["optional_non_trainer_selected_row_count"] != 1:
            raise RuntimeError("self-test did not count optional non-trainer selected rows")
        if summary["optional_non_trainer_promotion_row_count"] != 1:
            raise RuntimeError("self-test did not match optional non-trainer promotion rows")
        if summary["project_torch_fastest_row_count"] != 1:
            raise RuntimeError("self-test did not preserve project-wide Torch fastest rows")
        if summary["project_torch_fastest_used_row_count"] != 0:
            raise RuntimeError("self-test unexpectedly counted a trainer-used Torch fastest row")
        if summary["project_torch_fastest_resolved_divergence_row_count"] != 1:
            raise RuntimeError("self-test did not count resolved Torch fastest rows")
        if summary["project_torch_fastest_extra_row_count"] != 0:
            raise RuntimeError("self-test unexpectedly counted an extra Torch fastest row")
        if summary["project_torch_fastest_partitioned_row_count"] != summary["project_torch_fastest_row_count"]:
            raise RuntimeError("self-test did not partition every Torch fastest row")
        if summary["project_torch_fastest_disposition_row_count"] != summary["project_torch_fastest_row_count"]:
            raise RuntimeError("self-test did not write a disposition row for every Torch fastest row")
        if summary["project_torch_fastest_actionable_row_count"] != summary["project_torch_fastest_row_count"]:
            raise RuntimeError("self-test did not give every Torch fastest row an action or reason")
        if summary["project_torch_fastest_missing_disposition"]:
            raise RuntimeError("self-test found missing Torch fastest row disposition data")
        if summary["project_fastest_resolved_divergence_row_count"] != 1:
            raise RuntimeError("self-test did not count resolved project-wide fastest divergence")
        if summary["project_fastest_resolved_call_path_counts"] != {"operator_or_reference_prototype": 1}:
            raise RuntimeError("self-test did not count resolved project-wide fastest rows by call path")
        if summary["project_fastest_resolved_status_counts"] != {"rejected_same_session_refresh": 1}:
            raise RuntimeError("self-test did not count resolved project-wide fastest rows by decision")
        if summary["project_fastest_resolved_decision_link_count"] != 1:
            raise RuntimeError("self-test did not link project resolved rows to decision rows")
        if summary["project_fastest_resolved_missing_decision_links"]:
            raise RuntimeError("self-test unexpectedly found missing decision links")
        if summary["project_fastest_resolved_non_trainer_row_count"] != 1:
            raise RuntimeError("self-test did not count non-trainer project resolved rows")
        if summary["project_fastest_resolved_non_trainer_actionable_count"] != 1:
            raise RuntimeError("self-test did not count actionable non-trainer project resolved rows")
        if summary["project_fastest_resolved_non_trainer_missing_action"]:
            raise RuntimeError("self-test unexpectedly found missing non-trainer action metadata")
        if summary["project_fastest_resolved_trainer_callable_row_count"] != 0:
            raise RuntimeError("self-test unexpectedly found trainer-callable fastest-row debt")
        if summary["project_fastest_resolved_trainer_callable_evidence_count"] != 0:
            raise RuntimeError("self-test unexpectedly found trainer-callable debt evidence")
        if summary["project_fastest_resolved_trainer_callable_missing_evidence"]:
            raise RuntimeError("self-test unexpectedly found missing trainer-callable debt evidence")
        if summary["project_fastest_used_row_count"] != 1:
            raise RuntimeError("self-test did not count project-wide fastest rows used by trainer")
        resolved_optional = summary["resolved_optional_stack_decisions"]
        if not all(row.get("candidate_class") and row.get("promotion_gate") for row in resolved_optional):
            raise RuntimeError("self-test did not preserve optional promotion metadata")
        if summary["native_inactive_selected_row_count"] != 1:
            raise RuntimeError("self-test did not count inactive native selected rows")
        if summary["active_promotion_candidate_count"] != 0:
            raise RuntimeError("self-test unexpectedly found active promotions")
        effective_lmhead = [
            row
            for row in summary["native_trainer_selection"]
            if row["kernel"] == "fwd" and str(row["shape"]).startswith("lmhead ")
        ][0]
        if effective_lmhead["selected_stack"] != "cuBLASLt":
            raise RuntimeError("self-test did not apply inactive native fallback")
        markdown_path = root / "current.md"
        write_markdown(markdown_path, summary)
        markdown_text = markdown_path.read_text()
        if "Inactive Native Microbench Selections" not in markdown_text:
            raise RuntimeError("self-test markdown did not include inactive native rows")
        if "Resolved Optional-Stack Decisions" not in markdown_text:
            raise RuntimeError("self-test markdown did not include resolved optional decisions")
        if "Project-Wide Torch Fastest Row Disposition" not in markdown_text:
            raise RuntimeError("self-test markdown did not include Torch fastest row dispositions")
        if "Fastest Rows Not Used By Trainer" not in markdown_text:
            raise RuntimeError("self-test markdown did not include fastest-row debt summary")
        if "Project-Wide Torch Fastest Rows" not in markdown_text:
            raise RuntimeError("self-test markdown did not include project-wide Torch rows")
        if "Project-Wide Fastest Rows" not in markdown_text:
            raise RuntimeError("self-test markdown did not include project-wide fastest rows")
        if "Project-Wide Fastest Rows Used By Trainer" not in markdown_text:
            raise RuntimeError("self-test markdown did not include project-wide fastest used rows")
        if "Project-Wide Fastest Rows Resolved Away From Trainer" not in markdown_text:
            raise RuntimeError("self-test markdown did not include resolved fastest divergence rows")
        if "synthetic evidence" not in markdown_text:
            raise RuntimeError("self-test markdown did not include resolved divergence evidence")
        if "Extra Project-Wide Benchmark Rows" not in markdown_text:
            raise RuntimeError("self-test markdown did not include extra project-wide rows")

        bad_native = root / "bad-native"
        rows = synthetic_native_rows()
        rows[0]["trainer_call_path_available"] = False
        write_selected_payload(bad_native, rows)
        write_manifest_payload(bad_native)
        write_train_log(bad_native)
        try:
            build_summary(bad_native, optional_dir)
        except ValueError as exc:
            if "non-trainer-callable" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted non-trainer-callable native row")

        bad_optional = root / "bad-optional"
        write_selected_payload(bad_optional, synthetic_optional_rows())
        write_promotion_payload_with_rows(
            bad_optional,
            synthetic_promotion_rows()
            + [
                {
                    "suite": "runtime",
                    "kernel": "synthetic",
                    "shape": "synthetic",
                    "selected_stack": "Triton",
                }
            ],
        )
        try:
            build_summary(native_dir, bad_optional)
        except ValueError as exc:
            if "active promotion" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted active optional promotion")

        bad_optional = root / "bad-optional-missing-promotion"
        write_selected_payload(bad_optional, synthetic_optional_rows())
        write_promotion_payload_with_rows(bad_optional, synthetic_promotion_rows()[1:])
        try:
            build_summary(native_dir, bad_optional)
        except ValueError as exc:
            if "lack matching promotion rows" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted optional row without promotion coverage")

        bad_optional = root / "bad-optional-unresolved"
        rows = synthetic_promotion_rows()
        rows[0].pop("decision_status")
        rows[0].pop("decision_active")
        selected_rows = synthetic_optional_rows()
        selected_decision_row = next(row for row in selected_rows if row.get("selected_stack") == "Torch")
        selected_decision_row.pop("decision_status")
        selected_decision_row.pop("decision_active")
        write_selected_payload(bad_optional, selected_rows)
        write_promotion_payload_with_rows(bad_optional, rows)
        try:
            build_summary(native_dir, bad_optional)
        except ValueError as exc:
            if "lack inactive decisions" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted unresolved optional row")

        bad_inactive = root / "bad-inactive"
        rows = synthetic_native_rows()
        for row in rows:
            if row.get("decision_active") is False:
                row.pop("next_stack")
                break
        write_selected_payload(bad_inactive, rows)
        write_manifest_payload(bad_inactive)
        write_train_log(bad_inactive)
        try:
            build_summary(bad_inactive, optional_dir)
        except ValueError as exc:
            if "lacks a next-stack fallback" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted inactive native row without fallback")

        benchmark_only_native = root / "benchmark-only-native"
        write_selected_payload(benchmark_only_native, synthetic_native_rows())
        write_manifest_payload(benchmark_only_native, run_training="0", max_steps="3")
        try:
            build_summary(benchmark_only_native, optional_dir)
        except ValueError as exc:
            if "must include training evidence" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted benchmark-only native round as current selection")
        bypassed_summary = build_summary(
            benchmark_only_native,
            optional_dir,
            allow_benchmark_only_native=True,
        )
        if not bypassed_summary["native_training_evidence"].get("allow_benchmark_only_native"):
            raise RuntimeError("self-test benchmark-only bypass did not record the bypass")

    print("SM120 current selection self-test OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run synthetic pass/fail selection checks")
    parser.add_argument("--native-round", type=Path, default=DEFAULT_NATIVE_ROUND)
    parser.add_argument("--optional-round", type=Path, default=DEFAULT_OPTIONAL_ROUND)
    parser.add_argument(
        "--allow-benchmark-only-native",
        action="store_true",
        help=(
            "Allow a native round without TinyStories training evidence. "
            "Use only for inspection artifacts, not the published current trainer mix."
        ),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.json_out is None or args.markdown_out is None:
        parser.error("--json-out and --markdown-out are required unless --self-test is set")

    summary = build_summary(
        args.native_round,
        args.optional_round,
        allow_benchmark_only_native=args.allow_benchmark_only_native,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(args.markdown_out, summary)
    print(
        "SM120 current selection OK: "
        f"native_rows={summary['native_selected_row_count']}; "
        f"inactive_native={summary['native_inactive_selected_row_count']}; "
        f"optional_non_trainer={summary['optional_non_trainer_selected_row_count']}; "
        f"optional_decisions={summary['optional_decision_row_count']}; "
        f"active_promotions={summary['active_promotion_candidate_count']}; "
        f"json={args.json_out}; markdown={args.markdown_out}"
    )


if __name__ == "__main__":
    main()
