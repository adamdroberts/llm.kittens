#!/usr/bin/env python3
"""Audit SM120 optimization artifacts against the mixed-backend goal."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from sm120_objective_contract import (
    ATTENTION_SELECTION_SHAPE,
    CURRENT_NATIVE_SELECTION_ROUND,
    CURRENT_OPTIONAL_STACK_ROUND,
    ENVIRONMENT_STACKS,
    EXPECTED_MANIFEST_BINARIES,
    LIBTORCH_RUNTIME_SHAPE_REQUIREMENTS,
    LIBTORCH_RUNTIME_SUPPLEMENTAL_SHAPE_REQUIREMENTS,
    LIBTORCH_TRAINER_LINK_LOG,
    MATMUL_SHAPE_REQUIREMENTS,
    MATMUL_SELECTION_SHAPES,
    OBJECTIVE_FAMILIES,
    OBJECTIVE_STACKS,
    PYTHON_STACK_BENCHMARK_LOGS,
    RUNTIME_SELECTION_SHAPES,
    expected_trainer_selection_keys,
)

STACK_STATUSES = {"available", "missing", "blocked", "unknown", "not_applicable"}
FAMILY_STACK_STATUSES = {"baseline", "candidate", "fallback", "missing", "blocked", "unknown", "not_applicable"}
CUTEDSL_RESULT_RE = re.compile(r"^(?P<shape>.+?)\s+\|\s*CuTeDSL\s+\|\s*(?P<result>.+?)\s*$")
ATTENTION_RESULT_RE = re.compile(
    r"^(?:(?P<stack>Torch|TorchPacked|TorchMaterializedPacked|cuDNN|cuDNNPacked|Triton|TritonPacked)\s+)?"
    r"Attention (?P<pass>Forward|Backward) "
    r"\(B=(?P<b>\d+), T=(?P<t>\d+), C=(?P<c>\d+), NH=(?P<nh>\d+), HS=(?P<hs>\d+)\): "
    r"(?P<us>[0-9]+(?:\.[0-9]+)?) us"
)
UNAVAILABLE_RESULT_RE = re.compile(r"^(?P<shape>[^|]+?)\s+\|\s+(?P<stack>[^|]+?)\s+\|\s+unavailable:\s*(?P<reason>.+)$")
RUNTIME_RESULT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_]+)\s+\|\s+(?P<shape>[^|]+?)\s+\|\s+"
    r"(?P<stack>[^|]+?)\s+\|\s+(?P<us>[0-9]+(?:\.[0-9]+)?) us$"
)
RUNTIME_UNAVAILABLE_RESULT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_]+)\s+\|\s+(?P<shape>[^|]+?)\s+\|\s+"
    r"(?P<stack>[^|]+?)\s+\|\s+unavailable:\s*(?P<reason>.+)$"
)
LIBTORCH_RAW_POINTER_ROUTE = "LibTorch runtime route: cached from_blob wrappers over existing CUDA pointers"
LIBTORCH_CXX_API_RAW_POINTER_ROUTE = (
    "LibTorch runtime route: standalone C++ API cached from_blob handles over existing CUDA pointers"
)
LIBTORCH_RUNTIME_ROUTE_MARKERS = {
    "raw-pointer": LIBTORCH_RAW_POINTER_ROUTE,
    "cxx-api-raw-pointer": LIBTORCH_CXX_API_RAW_POINTER_ROUTE,
}
LIBTORCH_MATMUL_ROUTE = "LibTorch matmul route: standalone C++ API cached from_blob handles over existing CUDA pointers"
LIBTORCH_ALL_SHAPE_MATMUL_LOG = Path(
    "scratch/sm120_rounds/libtorch_matmul_all_shapes_20260521/bench_sm120_libtorch_matmul.log"
)
LIBTORCH_ALL_SHAPE_MATMUL_JSON = Path(
    "scratch/sm120_rounds/libtorch_matmul_all_shapes_20260521/bench_sm120_libtorch_matmul.json"
)
LIBTORCH_SUPPLEMENTAL_RUNTIME_LOGS = (
    Path("scratch/sm120_rounds/libtorch_runtime_gelu_20260521/bench_sm120_libtorch_runtime.log"),
)
LIBTORCH_SUPPLEMENTAL_TRAINER_LINK_LOGS = (
    Path("scratch/sm120_rounds/libtorch_trainer_link_20260521") / LIBTORCH_TRAINER_LINK_LOG,
)
CUDNN_PACKED_BACKWARD_ROUTE = "cuDNNPacked Attention Backward route: saved-forward"
CUDNN_PACKED_BACKWARD_ROUTES = {
    "saved-forward": CUDNN_PACKED_BACKWARD_ROUTE,
}
MATMUL_SHAPE_RE = re.compile(
    r"^(?P<name>\S+)\s+M=(?P<m>\d+)\s+N=(?P<n>\d+)\s+K=(?P<k>\d+)\s+"
    r"bias=(?P<bias>[01])\s+gelu=(?P<gelu>[01])$"
)
TRITON_MATMUL_RESULT_RE = re.compile(
    r"^(?P<kernel>fwd|fwd\+GeLU|dInp|dInp\+dGeLU|dW|dW\+accum)\s+"
    r"Triton\s+(?P<us>[0-9]+(?:\.[0-9]+)?) us(?:\s+\(.*\))?$"
)


DEFAULT_SELECTION_JSON = Path("scratch/sm120_rounds/current-sm120-selection.json")
DEFAULT_SELECTION_MD = Path("scratch/sm120_rounds/current-sm120-selection.md")
DEFAULT_NATIVE_ROUND = Path(CURRENT_NATIVE_SELECTION_ROUND)
DEFAULT_OPTIONAL_ROUND = Path(CURRENT_OPTIONAL_STACK_ROUND)
class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, str]] = []

    def pass_(self, name: str, detail: str) -> None:
        self.checks.append({"name": name, "status": "pass", "detail": detail})

    def fail(self, name: str, detail: str) -> None:
        raise ValueError(f"{name}: {detail}")

    def require(self, name: str, condition: bool, detail: str) -> None:
        if not condition:
            self.fail(name, detail)
        self.pass_(name, detail)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def round_manifest_config(round_dir: Path) -> dict[str, Any]:
    manifest_path = round_dir / "round-manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = read_json(manifest_path)
    config = manifest.get("config", {}) if isinstance(manifest, dict) else {}
    return config if isinstance(config, dict) else {}


def round_manifest_flag(round_dir: Path, key: str) -> bool:
    return str(round_manifest_config(round_dir).get(key, "0")) == "1"


def round_libtorch_matmul_shapes(round_dir: Path) -> tuple[str, ...]:
    raw_shapes = str(round_manifest_config(round_dir).get("libtorch_matmul_shapes", "")).strip()
    return tuple(raw_shapes.split()) if raw_shapes else ()


def parse_libtorch_matmul_evidence(text: str) -> tuple[set[tuple[str, str]], set[tuple[str, str]], dict[tuple[str, str], float]]:
    current_shape = ""
    observed_rows: set[tuple[str, str]] = set()
    parity_rows: set[tuple[str, str]] = set()
    timing_rows: dict[tuple[str, str], float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        shape_match = MATMUL_SHAPE_RE.match(line)
        if shape_match:
            current_shape = (
                f"{shape_match.group('name')} M={shape_match.group('m')} "
                f"N={shape_match.group('n')} K={shape_match.group('k')} "
                f"bias={shape_match.group('bias')} gelu={shape_match.group('gelu')}"
            )
            continue
        parity_match = re.match(
            r"^LibTorch matmul parity (?P<op>dW|dW\+accum) (?P<shape>\S+): PASS",
            line,
        )
        if parity_match:
            shape = MATMUL_SELECTION_SHAPES.get(parity_match.group("shape"))
            if shape is not None:
                parity_rows.add((parity_match.group("op"), shape))
            continue
        result_match = re.match(
            r"^(?P<op>dW|dW\+accum)\s+Torch C\+\+\s+(?P<us>[0-9]+(?:\.[0-9]+)?) us$",
            line,
        )
        if current_shape and result_match:
            key = (result_match.group("op"), current_shape)
            observed_rows.add(key)
            timing_rows[key] = float(result_match.group("us"))
    return observed_rows, parity_rows, timing_rows


def parse_libtorch_matmul_json_evidence(path: Path) -> tuple[str, set[tuple[str, str]], set[tuple[str, str]], dict[tuple[str, str], float]]:
    payload = read_json(path)
    route_marker = str(payload.get("route_marker", ""))
    observed_rows: set[tuple[str, str]] = set()
    parity_rows: set[tuple[str, str]] = set()
    timing_rows: dict[tuple[str, str], float] = {}
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return route_marker, observed_rows, parity_rows, timing_rows
    for row in rows:
        if not isinstance(row, dict):
            continue
        op = str(row.get("kernel", ""))
        shape = str(row.get("shape", ""))
        stack = str(row.get("stack", ""))
        if op not in {"dW", "dW+accum"} or stack != "Torch C++" or not shape:
            continue
        key = (op, shape)
        if row.get("parity_pass") is True:
            parity_rows.add(key)
        try:
            timing_rows[key] = float(row["time_us"])
        except (KeyError, TypeError, ValueError):
            continue
        observed_rows.add(key)
    return route_marker, observed_rows, parity_rows, timing_rows


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("suite", "")),
        str(row.get("kernel", "")),
        str(row.get("shape", "")),
        str(row.get("selected_stack", "")),
    )


def base_row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("suite", "")),
        str(row.get("kernel", "")),
        str(row.get("shape", "")),
    )


def is_torch_stack(stack: object) -> bool:
    return str(stack).startswith("Torch")


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


ROW_RUN_CONFIG_REQUIRED_KEYS = (
    "device_arch",
    "run_python_stack_benchmarks",
)

PROJECT_FASTEST_SOURCE_FIELDS = (
    "suite",
    "kernel",
    "shape",
    "selected_stack",
    "selected_time_us",
    "next_stack",
    "next_time_us",
    "trainer_call_path_available",
    "trainer_call_path_kind",
    "use_scope",
    "timing_log_path",
    "config_artifact_path",
    "stack_probe_artifact_path",
    "correctness_log_paths",
    "source_run_label",
    "source_artifact_dir",
    "source_git_commit",
    "source_run_config",
)

NATIVE_INACTIVE_COMMON_SOURCE_FIELDS = (
    "suite",
    "kernel",
    "shape",
    "next_stack",
    "next_time_us",
    "decision_status",
    "decision_active",
    "trainer_call_path_available",
    "trainer_call_path_kind",
    "use_scope",
    "timing_log_path",
    "config_artifact_path",
    "stack_probe_artifact_path",
    "correctness_log_paths",
    "source_run_label",
    "source_artifact_dir",
    "source_git_commit",
    "source_run_config",
)

RESOLVED_OPTIONAL_SOURCE_FIELDS = PROJECT_FASTEST_SOURCE_FIELDS + (
    "decision_status",
    "decision_active",
    "decision_decision",
    "decision_evidence",
    "decision_note",
    "candidate_class",
    "promotion_gate",
    "priority",
    "speedup_vs_next_pct",
)


def source_field_mismatches(row: dict[str, Any], source_row: dict[str, Any]) -> list[str]:
    return [
        field
        for field in PROJECT_FASTEST_SOURCE_FIELDS
        if row.get(field) != source_row.get(field)
    ]


def native_selection_source_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    selected_stack = row.get("rejected_selected_stack", row.get("selected_stack", ""))
    return (
        str(row.get("suite", "")),
        str(row.get("kernel", "")),
        str(row.get("shape", "")),
        str(selected_stack),
    )


def native_selection_mismatches(row: dict[str, Any], source_row: dict[str, Any]) -> list[str]:
    if "rejected_selected_stack" not in row:
        return source_field_mismatches(row, source_row)
    mismatches: list[str] = []
    if row.get("rejected_selected_stack") != source_row.get("selected_stack"):
        mismatches.append("rejected_selected_stack")
    if row.get("rejected_selected_time_us") != source_row.get("selected_time_us"):
        mismatches.append("rejected_selected_time_us")
    if row.get("selected_stack") != source_row.get("next_stack"):
        mismatches.append("selected_stack:fallback")
    if row.get("selected_time_us") != source_row.get("next_time_us"):
        mismatches.append("selected_time_us:fallback")
    mismatches.extend(
        field
        for field in NATIVE_INACTIVE_COMMON_SOURCE_FIELDS
        if row.get(field) != source_row.get(field)
    )
    return mismatches


def resolved_optional_mismatches(row: dict[str, Any], source_row: dict[str, Any]) -> list[str]:
    return [
        field
        for field in RESOLVED_OPTIONAL_SOURCE_FIELDS
        if row.get(field) != source_row.get(field)
    ]


def partition_row_content_mismatches(
    rows: list[dict[str, Any]],
    source_by_key: dict[tuple[str, str, str, str], dict[str, Any]],
    label: str,
    fields: tuple[str, ...] | None = None,
) -> list[str]:
    mismatches: list[str] = []
    for row in rows:
        key = row_key(row)
        source_row = source_by_key.get(key)
        if source_row is None:
            mismatches.append(f"{label}:{'/'.join(key)}:missing_source")
        elif fields is None and row != source_row:
            mismatches.append(f"{label}:{'/'.join(key)}:content")
        elif fields is not None:
            drift = [field for field in fields if row.get(field) != source_row.get(field)]
            if drift:
                mismatches.append(f"{label}:{'/'.join(key)}:{','.join(drift)}")
    return mismatches


def attention_route_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("shape", "")),
        str(row.get("stack", "")),
        str(row.get("route_scope", "")),
    )


def attention_route_content_mismatches(
    rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    label: str,
) -> list[str]:
    source_by_key = {attention_route_key(row): row for row in source_rows}
    mismatches: list[str] = []
    for row in rows:
        key = attention_route_key(row)
        source_row = source_by_key.get(key)
        if source_row is None:
            mismatches.append(f"{label}:{'/'.join(key)}:missing_source")
        elif row != source_row:
                mismatches.append(f"{label}:{'/'.join(key)}:content")
    return mismatches


def markdown_cell(raw: str) -> str:
    value = raw.strip()
    if value.startswith("`") and value.endswith("`"):
        value = value[1:-1]
    return value


def parse_markdown_float(raw: str) -> float | None:
    value = markdown_cell(raw)
    if value == "-":
        return None
    return float(value)


def parse_selected_backend_scoreboard_rows(scoreboard_path: Path) -> list[dict[str, Any]]:
    if not scoreboard_path.exists():
        return []
    text = scoreboard_path.read_text()
    marker = "## Selected Backend Rows"
    if marker not in text:
        return []
    section = text.split(marker, 1)[1].split("\n## ", 1)[0]
    rows: list[dict[str, Any]] = []
    for raw in section.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [markdown_cell(cell) for cell in line.strip("|").split("|")]
        if not cells or cells[0] in {"Suite", "---"}:
            continue
        if len(cells) < 7:
            continue
        rows.append(
            {
                "suite": cells[0],
                "kernel": cells[1],
                "shape": cells[2],
                "selected_stack": cells[3],
                "selected_time_us": parse_markdown_float(cells[4]),
                "next_stack": None if cells[5] == "-" else cells[5],
                "next_time_us": parse_markdown_float(cells[6]),
            }
        )
    return rows


def selected_backend_scoreboard_mismatches(
    round_dir: Path,
    selected_rows: list[dict[str, Any]],
) -> list[str]:
    scoreboard_rows = parse_selected_backend_scoreboard_rows(round_dir / "scoreboard-candidates.md")
    scoreboard_by_key = {row_key(row): row for row in scoreboard_rows}
    selected_by_key = {row_key(row): row for row in selected_rows}
    mismatches: list[str] = []
    missing = sorted(set(selected_by_key) - set(scoreboard_by_key))
    extra = sorted(set(scoreboard_by_key) - set(selected_by_key))
    mismatches.extend(f"{'/'.join(key)}:missing_scoreboard" for key in missing)
    mismatches.extend(f"{'/'.join(key)}:extra_scoreboard" for key in extra)
    for key in sorted(set(selected_by_key) & set(scoreboard_by_key)):
        row = selected_by_key[key]
        scoreboard_row = scoreboard_by_key[key]
        drift: list[str] = []
        for field in ("selected_stack", "next_stack"):
            if row.get(field) != scoreboard_row.get(field):
                drift.append(field)
        for field in ("selected_time_us", "next_time_us"):
            left = row.get(field)
            right = scoreboard_row.get(field)
            if left is None or right is None:
                if left != right:
                    drift.append(field)
            elif round(float(left), 3) != round(float(right), 3):
                drift.append(field)
        if drift:
            mismatches.append(f"{'/'.join(key)}:{','.join(drift)}")
    if selected_rows and not scoreboard_rows:
        mismatches.append("selected_backend_rows:missing_scoreboard_section")
    return mismatches


def expected_libtorch_runtime_coverage_keys() -> set[str]:
    return {
        f"{kernel}:{shape}"
        for kernel, shapes in LIBTORCH_RUNTIME_SHAPE_REQUIREMENTS
        for shape in shapes
    }


def selected_backend_metadata_gaps(
    payload: dict[str, Any],
    round_dir: Path,
    selected_rows: list[dict[str, Any]],
    *,
    require_torch_coverage: bool,
    require_libtorch_coverage: bool,
) -> list[str]:
    gaps: list[str] = []
    if payload.get("schema_version") != 1:
        gaps.append("schema_version")
    manifest = read_json(round_dir / "round-manifest.json")
    config = manifest.get("config", {})
    git = manifest.get("git", {})
    if not isinstance(config, dict):
        config = {}
        gaps.append("manifest_config")
    if not isinstance(git, dict):
        git = {}
        gaps.append("manifest_git")
    expected_run_label = config.get("run_label")
    if str(payload.get("run_label")) != str(expected_run_label):
        gaps.append("run_label")
    if Path(str(payload.get("artifact_dir", ""))) != round_dir:
        gaps.append("artifact_dir")
    if str(payload.get("artifact_dir")) != str(config.get("artifact_dir")):
        gaps.append("artifact_dir:manifest")
    if str(payload.get("git_commit")) != str(git.get("short_commit")):
        gaps.append("git_commit")
    benchmark_count = payload.get("benchmark_row_count")
    if not isinstance(benchmark_count, int) or benchmark_count < len(selected_rows):
        gaps.append("benchmark_row_count")
    if any("trainer_call_path_kind" not in row for row in selected_rows):
        gaps.append("selected_backend_rows:trainer_call_path_kind")
    policy = payload.get("selection_policy")
    if not isinstance(policy, str) or "Fastest observed row" not in policy or "trainer call path" not in policy:
        gaps.append("selection_policy")
    torch_count = payload.get("torch_benchmark_row_count")
    torch_coverage = payload.get("torch_benchmark_coverage", {})
    if require_torch_coverage:
        expected_count = len(expected_trainer_selection_keys())
        if torch_count != expected_count:
            gaps.append("torch_benchmark_row_count")
        if (
            not isinstance(torch_coverage, dict)
            or len(torch_coverage) != expected_count
            or not all(value is True for value in torch_coverage.values())
        ):
            gaps.append("torch_benchmark_coverage")
    else:
        if torch_count not in (0, None):
            gaps.append("torch_benchmark_row_count")
        if torch_coverage not in ({}, None):
            gaps.append("torch_benchmark_coverage")
    expected_libtorch = expected_libtorch_runtime_coverage_keys()
    for field in ("libtorch_runtime_shape_coverage", "libtorch_runtime_parity_coverage"):
        coverage = payload.get(field, {})
        if require_libtorch_coverage:
            if (
                not isinstance(coverage, dict)
                or set(coverage) != expected_libtorch
                or not all(value is True for value in coverage.values())
            ):
                gaps.append(field)
        elif coverage not in ({}, None):
            gaps.append(field)
    route = payload.get("libtorch_runtime_raw_pointer_route")
    if require_libtorch_coverage:
        if route is not True:
            gaps.append("libtorch_runtime_raw_pointer_route")
    elif route not in (False, None):
        gaps.append("libtorch_runtime_raw_pointer_route")
    return gaps


def row_provenance_gaps(row: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    required = (
        "source_run_label",
        "source_artifact_dir",
        "source_git_commit",
        "timing_log_path",
        "config_artifact_path",
        "stack_probe_artifact_path",
    )
    for key in required:
        value = row.get(key)
        if not value:
            gaps.append(key)
            continue
        if key.endswith("_path") and not Path(str(value)).exists():
            gaps.append(f"{key}:missing")
    manifest_config: dict[str, Any] = {}
    manifest_git: dict[str, Any] = {}
    config_path = row.get("config_artifact_path")
    if config_path:
        try:
            manifest = read_json(Path(str(config_path)))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            manifest = {}
        config = manifest.get("config", {}) if isinstance(manifest, dict) else {}
        git = manifest.get("git", {}) if isinstance(manifest, dict) else {}
        if isinstance(config, dict):
            manifest_config = config
        else:
            gaps.append("config_artifact_path:manifest_config")
        if isinstance(git, dict):
            manifest_git = git
        else:
            gaps.append("config_artifact_path:manifest_git")
    if manifest_config:
        manifest_run_label = manifest_config.get("run_label")
        if manifest_run_label is not None and str(row.get("source_run_label")) != str(manifest_run_label):
            gaps.append("source_run_label:mismatch")
        manifest_artifact_dir = manifest_config.get("artifact_dir")
        if manifest_artifact_dir is not None and str(row.get("source_artifact_dir")) != str(manifest_artifact_dir):
            gaps.append("source_artifact_dir:mismatch")
    if manifest_git:
        manifest_short_commit = manifest_git.get("short_commit")
        if manifest_short_commit is not None and str(row.get("source_git_commit")) != str(manifest_short_commit):
            gaps.append("source_git_commit:mismatch")
    run_config = row.get("source_run_config")
    if not isinstance(run_config, dict):
        gaps.append("source_run_config")
    else:
        for key in ROW_RUN_CONFIG_REQUIRED_KEYS:
            if not run_config.get(key):
                gaps.append(f"source_run_config:{key}")
        if config_path and not manifest_config:
            gaps.append("source_run_config:manifest_config")
        for key, value in run_config.items():
            if key in manifest_config and str(manifest_config[key]) != str(value):
                gaps.append(f"source_run_config:{key}:mismatch")
                break
    correctness_paths = row.get("correctness_log_paths")
    if not isinstance(correctness_paths, list):
        gaps.append("correctness_log_paths")
    elif correctness_paths:
        for path in correctness_paths:
            if not Path(str(path)).exists():
                gaps.append("correctness_log_paths:missing")
                break
    elif not row.get("correctness_evidence_note"):
        gaps.append("correctness_evidence_note")
    return gaps


def row_objective_families(row: dict[str, Any]) -> set[str]:
    suite = str(row.get("suite", ""))
    kernel = str(row.get("kernel", ""))
    families: set[str] = set()
    if suite == "matmul":
        mapping = {
            "fwd": "gemm_forward",
            "fwd+gelu": "gemm_forward_fused_gelu",
            "dInp": "gemm_backward_dinput",
            "dInp+dGeLU": "gemm_backward_dinput_fused_dgelu",
            "dW": "gemm_backward_dweight",
            "dW+accum": "gemm_backward_dweight_accum",
        }
        family = mapping.get(kernel)
        if family:
            families.add(family)
    elif suite == "attention":
        mapping = {
            "forward": "attention_forward",
            "backward": "attention_backward",
        }
        family = mapping.get(kernel)
        if family:
            families.add(family)
    elif suite == "layernorm":
        mapping = {
            "forward": "layernorm_forward",
            "fused_residual_forward": "layernorm_fused_residual_forward",
            "backward": "layernorm_backward",
        }
        family = mapping.get(kernel)
        if family:
            families.add(family)
    elif suite == "runtime":
        mapping = {
            "bias_add": "bias_add",
            "bias_grad_reduce": "bias_gradient_reduce",
            "gelu_forward": "gelu_forward",
            "gelu_backward_inplace": "gelu_backward",
            "adamw_update": "adamw",
            "global_norm_squared": "global_norm",
            "encoder_forward": "encoder_forward",
            "cuda_memset": "cuda_memset",
            "cuda_copy_d2d": "cuda_copy_d2d",
        }
        family = mapping.get(kernel)
        if family:
            families.add(family)
        if kernel in {"fused_classifier", "fused_classifier_loss"}:
            families.add("classifier_softmax_cross_entropy_dlogits")
    return families


def trainer_family_coverage(rows: list[dict[str, Any]]) -> dict[str, bool]:
    covered: set[str] = set()
    for row in rows:
        covered.update(row_objective_families(row))
    return {family: family in covered for family in OBJECTIVE_FAMILIES}


def trainer_exact_row_coverage(rows: list[dict[str, Any]]) -> dict[str, bool]:
    present = {
        (
            str(row.get("suite", "")),
            str(row.get("kernel", "")),
            str(row.get("shape", "")),
        )
        for row in rows
    }
    return {
        f"{suite}/{kernel}/{shape}": (suite, kernel, shape) in present
        for suite, kernel, shape in sorted(expected_trainer_selection_keys())
    }


def training_summary(scoreboard_path: Path) -> dict[str, Any]:
    text = scoreboard_path.read_text()
    steps_match = re.search(r"training steps:\s*`?([0-9]+)`?", text)
    final_step_match = re.search(r"final step:\s*`?([0-9]+)/([0-9]+)`?", text)
    avg_match = re.search(r"total average iteration time:\s*`?([0-9.]+)\s*ms`?", text)
    steps = int(steps_match.group(1)) if steps_match else None
    if steps is None and final_step_match:
        steps = int(final_step_match.group(1))
    return {
        "steps": steps,
        "expected_steps": int(final_step_match.group(2)) if final_step_match else None,
        "avg_ms": float(avg_match.group(1)) if avg_match else None,
    }


def audit_round_manifest(
    audit: Audit,
    round_dir: Path,
    *,
    require_python_stack_benchmarks: bool,
) -> dict[str, Any]:
    manifest_path = round_dir / "round-manifest.json"
    manifest = read_json(manifest_path)
    config = manifest.get("config", {})
    git = manifest.get("git", {})
    toolchain = manifest.get("toolchain", {})
    binaries = manifest.get("binaries", [])

    audit.require(
        "round manifest schema",
        manifest.get("schema_version") == 1,
        f"{manifest_path} has schema_version={manifest.get('schema_version')}",
    )
    audit.require(
        "round manifest config",
        isinstance(config, dict)
        and Path(str(config.get("artifact_dir", ""))) == round_dir
        and config.get("device_arch") == "SM120",
        f"{manifest_path} records artifact_dir={config.get('artifact_dir')} device_arch={config.get('device_arch')}",
    )
    if require_python_stack_benchmarks:
        audit.require(
            "round manifest python-stack flag",
            config.get("run_python_stack_benchmarks") == "1",
            f"{manifest_path} records run_python_stack_benchmarks={config.get('run_python_stack_benchmarks')}",
        )
    audit.require(
        "round manifest git",
        isinstance(git, dict) and bool(git.get("short_commit")) and bool(git.get("status_path")),
        f"{manifest_path} records git short_commit={git.get('short_commit')}",
    )
    nvcc = toolchain.get("nvcc") if isinstance(toolchain, dict) else None
    audit.require(
        "round manifest toolchain",
        isinstance(nvcc, dict) and nvcc.get("returncode") == 0 and bool(nvcc.get("stdout")),
        f"{manifest_path} records nvcc returncode={nvcc.get('returncode') if isinstance(nvcc, dict) else 'missing'}",
    )
    binary_by_path = {
        str(row.get("path")): row
        for row in binaries
        if isinstance(row, dict) and row.get("path")
    } if isinstance(binaries, list) else {}
    missing_binaries = [
        binary
        for binary in EXPECTED_MANIFEST_BINARIES
        if binary not in binary_by_path or not binary_by_path[binary].get("exists")
    ]
    missing_sha = [
        path
        for path, row in binary_by_path.items()
        if row.get("exists") and not row.get("sha256")
    ]
    audit.require(
        "round manifest binaries",
        not missing_binaries and not missing_sha,
        f"{manifest_path} records {len(binary_by_path)} binaries with SHA256 evidence",
    )
    return {
        "binary_rows": len(binary_by_path),
        "git_short_commit": git.get("short_commit") if isinstance(git, dict) else None,
        "device_arch": config.get("device_arch") if isinstance(config, dict) else None,
        "run_python_stack_benchmarks": config.get("run_python_stack_benchmarks") if isinstance(config, dict) else None,
    }


def check_stack_matrix(audit: Audit, round_dir: Path) -> dict[str, Any]:
    stacks_payload = read_json(round_dir / "backend-stacks.json")
    stacks = {row.get("stack"): row for row in stacks_payload.get("stacks", [])}
    objective_stacks = tuple(stacks_payload.get("objective_stacks", []))
    objective_families = tuple(stacks_payload.get("objective_families", []))
    family_matrix = stacks_payload.get("family_matrix", [])

    audit.require(
        "objective stack list",
        objective_stacks == OBJECTIVE_STACKS,
        f"{round_dir} records {len(objective_stacks)} objective stacks",
    )
    audit.require(
        "objective family list",
        objective_families == OBJECTIVE_FAMILIES,
        f"{round_dir} records {len(objective_families)} objective families",
    )
    expected_rows = len(OBJECTIVE_STACKS) * len(OBJECTIVE_FAMILIES)
    audit.require(
        "family/stack matrix",
        len(family_matrix) == expected_rows,
        f"{round_dir} has {len(family_matrix)} family/stack rows",
    )
    required_stacks = set(OBJECTIVE_STACKS) | set(ENVIRONMENT_STACKS)
    missing_stacks = sorted(required_stacks - set(stacks))
    audit.require(
        "stack probe rows",
        not missing_stacks,
        f"{round_dir} records {len(stacks)} stack probe rows; missing={missing_stacks}",
    )
    bad_stack_rows: list[str] = []
    for stack_name in sorted(required_stacks):
        row = stacks.get(stack_name, {})
        evidence = row.get("evidence")
        candidate_use = row.get("candidate_use")
        next_action = row.get("next_action")
        if row.get("status") not in STACK_STATUSES:
            bad_stack_rows.append(f"{stack_name}:status")
        if not isinstance(evidence, list) or not evidence or not all(isinstance(item, str) and item for item in evidence):
            bad_stack_rows.append(f"{stack_name}:evidence")
        if not isinstance(candidate_use, str) or not candidate_use:
            bad_stack_rows.append(f"{stack_name}:candidate_use")
        if not isinstance(next_action, str) or not next_action:
            bad_stack_rows.append(f"{stack_name}:next_action")
    audit.require(
        "stack probe evidence",
        not bad_stack_rows,
        f"{round_dir} stack rows include status, evidence, candidate_use, and next_action",
    )
    matrix_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    bad_matrix_rows: list[str] = []
    for row in family_matrix:
        family = str(row.get("family", ""))
        stack = str(row.get("stack", ""))
        key = (family, stack)
        if key in matrix_by_key:
            bad_matrix_rows.append(f"{family}/{stack}:duplicate")
        matrix_by_key[key] = row
        status = row.get("status")
        reason = row.get("reason")
        next_action = row.get("next_action")
        if status not in FAMILY_STACK_STATUSES:
            bad_matrix_rows.append(f"{family}/{stack}:status")
        if not isinstance(reason, str) or not reason:
            bad_matrix_rows.append(f"{family}/{stack}:reason")
        if status == "not_applicable" and str(reason).lower() in {"n/a", "na", "none"}:
            bad_matrix_rows.append(f"{family}/{stack}:not_applicable_reason")
        if not isinstance(next_action, str) or not next_action:
            bad_matrix_rows.append(f"{family}/{stack}:next_action")
    expected_matrix = {
        (family, stack)
        for family in OBJECTIVE_FAMILIES
        for stack in OBJECTIVE_STACKS
    }
    missing_matrix_rows = sorted(expected_matrix - set(matrix_by_key))
    audit.require(
        "family/stack row evidence",
        not bad_matrix_rows and not missing_matrix_rows,
        f"{round_dir} family/stack rows include status, reason, and next_action; missing={len(missing_matrix_rows)} bad={len(bad_matrix_rows)}",
    )
    torch_stack = stacks.get("Torch")
    audit.require(
        "Torch stack probe",
        bool(torch_stack) and torch_stack.get("status") == "available",
        f"{round_dir} records Torch status={torch_stack.get('status') if torch_stack else 'missing'}",
    )
    torch_matrix_rows = [row for row in family_matrix if row.get("stack") == "Torch"]
    audit.require(
        "Torch family coverage",
        len(torch_matrix_rows) == len(OBJECTIVE_FAMILIES),
        f"{round_dir} records {len(torch_matrix_rows)} Torch family rows",
    )
    return {
        "stack_count": len(stacks_payload.get("stacks", [])),
        "family_stack_rows": len(family_matrix),
        "stack_probe_rows": len(stacks),
        "torch_status": torch_stack.get("status") if torch_stack else "missing",
    }


def audit_selection(
    audit: Audit,
    selection_path: Path,
    selection_md_path: Path,
    native_round: Path,
    optional_round: Path,
) -> dict[str, Any]:
    selection = read_json(selection_path)
    native_rows = selection.get("native_trainer_selection", [])
    resolved_rows = selection.get("resolved_optional_stack_decisions", [])
    project_fastest_rows = selection.get("project_fastest_selection", [])
    project_torch_rows = selection.get("project_torch_fastest_rows", [])
    project_torch_disposition_rows = selection.get("project_torch_fastest_disposition_rows", [])
    project_used_rows = selection.get("project_fastest_used_rows", [])
    project_resolved_rows = selection.get("project_fastest_resolved_divergence_rows", [])
    project_extra_artifact_rows = selection.get("project_fastest_extra_rows", [])
    native_extra_rows = selection.get("native_extra_selected_rows", [])
    native_attention_routes = selection.get("native_attention_route_rows", [])
    optional_attention_routes = selection.get("optional_attention_route_rows", [])
    if not isinstance(native_rows, list) or not isinstance(resolved_rows, list):
        raise ValueError("selection artifact has invalid row lists")
    if not isinstance(project_fastest_rows, list) or not isinstance(project_torch_rows, list):
        raise ValueError("selection artifact has invalid project-wide row lists")
    if not isinstance(project_torch_disposition_rows, list):
        raise ValueError("selection artifact has invalid project-wide Torch disposition row list")
    if not isinstance(project_used_rows, list):
        raise ValueError("selection artifact has invalid project-wide used row list")
    if not isinstance(project_resolved_rows, list) or not isinstance(project_extra_artifact_rows, list):
        raise ValueError("selection artifact has invalid project-wide partition row lists")
    if not isinstance(native_extra_rows, list):
        raise ValueError("selection artifact has invalid native extra selected row list")
    if not isinstance(native_attention_routes, list) or not isinstance(optional_attention_routes, list):
        raise ValueError("selection artifact has invalid attention route rows")
    native_selected = read_json(native_round / "selected-backends.json")
    native_source_rows = native_selected.get("selected_backend_rows", [])
    if not isinstance(native_source_rows, list):
        raise ValueError("native selected-backends artifact has invalid selected_backend_rows")
    native_scoreboard_mismatches = selected_backend_scoreboard_mismatches(native_round, native_source_rows)
    native_metadata_gaps = selected_backend_metadata_gaps(
        native_selected,
        native_round,
        native_source_rows,
        require_torch_coverage=False,
        require_libtorch_coverage=False,
    )
    native_source_attention_routes = native_selected.get("attention_route_rows", [])
    if not isinstance(native_source_attention_routes, list):
        raise ValueError("native selected-backends artifact has invalid attention_route_rows")
    optional_selected_for_routes = read_json(optional_round / "selected-backends.json")
    optional_scoreboard_source_rows = optional_selected_for_routes.get("selected_backend_rows", [])
    if not isinstance(optional_scoreboard_source_rows, list):
        raise ValueError("optional selected-backends artifact has invalid selected_backend_rows")
    optional_scoreboard_mismatches = selected_backend_scoreboard_mismatches(
        optional_round,
        optional_scoreboard_source_rows,
    )
    optional_metadata_gaps = selected_backend_metadata_gaps(
        optional_selected_for_routes,
        optional_round,
        optional_scoreboard_source_rows,
        require_torch_coverage=True,
        require_libtorch_coverage=True,
    )
    optional_source_attention_routes = optional_selected_for_routes.get("attention_route_rows", [])
    if not isinstance(optional_source_attention_routes, list):
        raise ValueError("optional selected-backends artifact has invalid attention_route_rows")
    native_route_content_mismatches = attention_route_content_mismatches(
        native_attention_routes,
        native_source_attention_routes,
        "native-route",
    )
    optional_route_content_mismatches = attention_route_content_mismatches(
        optional_attention_routes,
        optional_source_attention_routes,
        "optional-route",
    )
    native_source_by_key = {row_key(row): row for row in native_source_rows}
    expected_native_row_keys = expected_trainer_selection_keys()
    native_source_extra_rows = [
        row
        for row in native_source_rows
        if (
            str(row.get("suite", "")),
            str(row.get("kernel", "")),
            str(row.get("shape", "")),
        )
        not in expected_native_row_keys
    ]
    native_source_extra_by_key = {row_key(row): row for row in native_source_extra_rows}
    native_extra_keys = {row_key(row) for row in native_extra_rows}
    native_source_extra_keys = set(native_source_extra_by_key)
    native_extra_content_mismatches = partition_row_content_mismatches(
        native_extra_rows,
        native_source_extra_by_key,
        "native-extra",
        PROJECT_FASTEST_SOURCE_FIELDS,
    )
    native_source_content_mismatches: list[str] = []
    for row in native_rows:
        source_key = native_selection_source_key(row)
        source_row = native_source_by_key.get(source_key)
        if source_row is None:
            native_source_content_mismatches.append(f"{'/'.join(source_key)}:missing_source")
            continue
        mismatches = native_selection_mismatches(row, source_row)
        if mismatches:
            native_source_content_mismatches.append(
                f"{'/'.join(source_key)}:{','.join(mismatches)}"
            )
    active_promotions = selection.get("active_promotion_candidate_count")
    torch_trainer_rows = [
        row for row in native_rows if is_torch_stack(row.get("selected_stack"))
    ]
    project_torch_keys = {row_key(row) for row in project_torch_rows}
    project_fastest_keys = {row_key(row) for row in project_fastest_rows}
    missing_project_torch_rows = project_torch_keys - project_fastest_keys
    expected_keys = expected_trainer_selection_keys()
    native_by_selection_key = {
        (
            str(row.get("suite", "")),
            str(row.get("kernel", "")),
            str(row.get("shape", "")),
        ): row
        for row in native_rows
    }
    project_extra_rows = []
    project_computed_used_rows = []
    project_resolved_divergence_rows = []
    project_unresolved_objective_rows = []
    for row in project_fastest_rows:
        selection_key = (
            str(row.get("suite", "")),
            str(row.get("kernel", "")),
            str(row.get("shape", "")),
        )
        native_row = native_by_selection_key.get(selection_key)
        if selection_key not in expected_keys:
            project_extra_rows.append(row)
        elif native_row is not None and native_row.get("selected_stack") == row.get("selected_stack"):
            project_computed_used_rows.append(row)
            continue
        elif row.get("decision_status") is not None and row.get("decision_active") is False:
            project_resolved_divergence_rows.append(row)
        else:
            project_unresolved_objective_rows.append(row)
    project_computed_used_keys_for_torch = {row_key(row) for row in project_computed_used_rows}
    project_computed_resolved_keys_for_torch = {
        row_key(row) for row in project_resolved_divergence_rows
    }
    project_computed_extra_keys_for_torch = {row_key(row) for row in project_extra_rows}
    project_torch_used_rows = [
        row for row in project_torch_rows if row_key(row) in project_computed_used_keys_for_torch
    ]
    project_torch_resolved_rows = [
        row for row in project_torch_rows if row_key(row) in project_computed_resolved_keys_for_torch
    ]
    project_torch_extra_rows = [
        row for row in project_torch_rows if row_key(row) in project_computed_extra_keys_for_torch
    ]
    project_torch_partitioned_keys = (
        {row_key(row) for row in project_torch_used_rows}
        | {row_key(row) for row in project_torch_resolved_rows}
        | {row_key(row) for row in project_torch_extra_rows}
    )
    project_torch_partition_overlap = (
        len(project_torch_used_rows)
        + len(project_torch_resolved_rows)
        + len(project_torch_extra_rows)
        - len(project_torch_partitioned_keys)
    )
    project_torch_unpartitioned_keys = project_torch_keys - project_torch_partitioned_keys
    project_torch_disposition_by_key = {row_key(row): row for row in project_torch_disposition_rows}
    project_torch_disposition_keys = set(project_torch_disposition_by_key)
    project_torch_missing_disposition_keys = project_torch_keys - project_torch_disposition_keys
    project_torch_extra_disposition_keys = project_torch_disposition_keys - project_torch_keys
    project_torch_bad_disposition_rows = []
    project_torch_missing_action_rows = []
    for row in project_torch_rows:
        key = row_key(row)
        disposition_row = project_torch_disposition_by_key.get(key)
        if disposition_row is None:
            continue
        if key in project_computed_used_keys_for_torch:
            expected_disposition = "trainer_used"
        elif key in project_computed_resolved_keys_for_torch:
            expected_disposition = "resolved_away"
        elif key in project_computed_extra_keys_for_torch:
            expected_disposition = "extra_benchmark"
        else:
            expected_disposition = "unpartitioned"
        if disposition_row.get("torch_disposition") != expected_disposition:
            project_torch_bad_disposition_rows.append(disposition_row)
        if not disposition_row.get("torch_action"):
            project_torch_missing_action_rows.append(disposition_row)
    project_torch_missing_disposition_labels = [
        "/".join(key) for key in sorted(project_torch_missing_disposition_keys)
    ]
    project_torch_extra_disposition_labels = [
        "/".join(key) for key in sorted(project_torch_extra_disposition_keys)
    ]
    project_torch_bad_disposition_labels = [
        f"{row.get('suite')}/{row.get('kernel')}/{row.get('shape')}/{row.get('selected_stack')}"
        for row in project_torch_bad_disposition_rows
    ]
    project_torch_missing_action_labels = [
        f"{row.get('suite')}/{row.get('kernel')}/{row.get('shape')}/{row.get('selected_stack')}"
        for row in project_torch_missing_action_rows
    ]
    trainer_stack_counts = Counter(str(row.get("selected_stack", "unknown")) for row in native_rows)
    family_coverage = trainer_family_coverage(native_rows)
    exact_row_coverage = trainer_exact_row_coverage(native_rows)
    missing_families = [
        family
        for family, covered in family_coverage.items()
        if not covered
    ]
    missing_exact_rows = [
        key
        for key, covered in exact_row_coverage.items()
        if not covered
    ]

    selection_md_text = selection_md_path.read_text() if selection_md_path.exists() else ""
    audit.require(
        "selection markdown",
        selection_md_path.exists() and "Current SM120 Backend Selection" in selection_md_text,
        f"{selection_md_path} exists and has the current selection header",
    )
    audit.require(
        "selection optional decision table",
        "Resolved Optional-Stack Decisions" in selection_md_text,
        f"{selection_md_path} lists resolved optional-stack decisions",
    )
    audit.require(
        "selection project-wide Torch table",
        "Project-Wide Torch Fastest Rows" in selection_md_text,
        f"{selection_md_path} lists project-wide Torch fastest rows",
    )
    audit.require(
        "selection project-wide Torch disposition table",
        "Project-Wide Torch Fastest Row Disposition" in selection_md_text,
        f"{selection_md_path} lists disposition and next action for every project-wide Torch fastest row",
    )
    audit.require(
        "selection project-wide fastest table",
        "Project-Wide Fastest Rows" in selection_md_text,
        f"{selection_md_path} lists all project-wide fastest rows",
    )
    audit.require(
        "selection project-wide used table",
        "Project-Wide Fastest Rows Used By Trainer" in selection_md_text,
        f"{selection_md_path} lists project-wide fastest rows used by trainer",
    )
    audit.require(
        "selection project-wide divergence table",
        "Project-Wide Fastest Rows Resolved Away From Trainer" in selection_md_text,
        f"{selection_md_path} lists project-wide fastest rows resolved away from trainer",
    )
    audit.require(
        "selection fastest-row debt table",
        "Fastest Rows Not Used By Trainer" in selection_md_text,
        f"{selection_md_path} lists fastest rows not used by trainer by call-path kind",
    )
    audit.require(
        "selection extra project-wide table",
        "Extra Project-Wide Benchmark Rows" in selection_md_text,
        f"{selection_md_path} lists extra project-wide benchmark rows",
    )
    audit.require(
        "selection attention route table",
        "Attention Route Totals" in selection_md_text,
        f"{selection_md_path} lists attention route totals",
    )
    audit.require(
        "native round path",
        Path(str(selection.get("native_selection_round"))) == native_round,
        f"selection points to {selection.get('native_selection_round')}",
    )
    audit.require(
        "optional round path",
        Path(str(selection.get("optional_stack_round"))) == optional_round,
        f"selection points to {selection.get('optional_stack_round')}",
    )
    native_training_evidence = selection.get("native_training_evidence", {})
    native_manifest_path = native_round / "round-manifest.json"
    native_train_log_path = native_round / "train_gpt2cu.log"
    audit.require(
        "native training evidence",
        isinstance(native_training_evidence, dict)
        and native_training_evidence.get("allow_benchmark_only_native") is False
        and Path(str(native_training_evidence.get("manifest_path", ""))) == native_manifest_path
        and Path(str(native_training_evidence.get("train_log_path", ""))) == native_train_log_path
        and native_manifest_path.exists()
        and native_train_log_path.exists(),
        (
            "current selection records TinyStories training evidence at "
            f"{native_training_evidence.get('train_log_path')}"
        ),
    )
    audit.require(
        "active promotion count",
        active_promotions == 0,
        f"current selection has {active_promotions} active promotion candidates",
    )
    audit.require(
        "native trainer rows",
        len(native_rows) == len(expected_trainer_selection_keys()),
        (
            "current selection has "
            f"{len(native_rows)}/{len(expected_trainer_selection_keys())} required trainer rows"
        ),
    )
    audit.require(
        "native trainer source row content",
        not native_source_content_mismatches,
        (
            "native trainer rows preserve source selected-backends stack, time, fallback, "
            f"scope, and provenance fields; mismatches={len(native_source_content_mismatches)}"
        ),
    )
    audit.require(
        "source selected-backend scoreboard content",
        not native_scoreboard_mismatches and not optional_scoreboard_mismatches,
        (
            "source selected-backends rows match their generated scoreboard Selected Backend Rows; "
            f"native_mismatches={len(native_scoreboard_mismatches)} "
            f"optional_mismatches={len(optional_scoreboard_mismatches)}"
        ),
    )
    audit.require(
        "source selected-backend metadata",
        not native_metadata_gaps and not optional_metadata_gaps,
        (
            "source selected-backends metadata records schema, run identity, selection policy, "
            "benchmark counts, and Torch/LibTorch coverage; "
            f"native_gaps={native_metadata_gaps[:5]} optional_gaps={optional_metadata_gaps[:5]}"
        ),
    )
    audit.require(
        "native extra selected row content",
        selection.get("native_extra_selected_row_count") == len(native_source_extra_rows)
        and native_extra_keys == native_source_extra_keys
        and not native_extra_content_mismatches,
        (
            "current selection native extra benchmark rows match source selected-backends "
            f"non-objective rows; count={selection.get('native_extra_selected_row_count')} "
            f"expected={len(native_source_extra_rows)} "
            f"missing={len(native_source_extra_keys - native_extra_keys)} "
            f"extra={len(native_extra_keys - native_source_extra_keys)} "
            f"mismatches={len(native_extra_content_mismatches)}"
        ),
    )
    rejected_native = [
        row for row in native_rows if row.get("decision_active") is False and "rejected_selected_stack" not in row
    ]
    audit.require(
        "inactive native fallback mapping",
        not rejected_native,
        f"{selection.get('native_inactive_selected_row_count', 0)} inactive native rows are mapped to fallback stacks",
    )
    bad_torch_trainer = [
        row
        for row in torch_trainer_rows
        if not row.get("trainer_call_path_available") or row.get("decision_active") is False
    ]
    rows_without_provenance = [
        row for row in native_rows + resolved_rows + project_fastest_rows if row_provenance_gaps(row)
    ]
    profiler_runtime_rows_with_wrong_call_path = [
        row
        for row in native_rows + resolved_rows + project_fastest_rows + native_source_rows + optional_scoreboard_source_rows
        if row.get("suite") == "runtime"
        and (
            row.get("kernel") == "cuda_copy_d2d"
            or (
                row.get("kernel") == "cuda_memset"
                and row.get("shape") == "logits_elems=3296722944"
            )
        )
        and row.get("trainer_call_path_kind") != "profiler_runtime_benchmark_only"
    ]
    audit.require(
        "Torch trainer eligibility",
        not bad_torch_trainer,
        f"{len(torch_trainer_rows)} Torch rows are in the trainer mix and all are trainer-callable",
    )
    audit.require(
        "resolved optional decisions",
        bool(resolved_rows),
        f"selection records {len(resolved_rows)} resolved optional-stack decisions",
    )
    audit.require(
        "project-wide fastest rows",
        bool(project_fastest_rows),
        f"selection records {len(project_fastest_rows)} fastest observed rows from the optional-stack round",
    )
    audit.require(
        "project-wide Torch fastest rows",
        bool(project_torch_rows) and not missing_project_torch_rows,
        f"selection records {len(project_torch_rows)} Torch fastest rows; missing_from_project={len(missing_project_torch_rows)}",
    )
    native_by_base_key = {base_row_key(row): row for row in native_rows}
    optional_rows_superseded_by_native = []
    for row in project_fastest_rows:
        if row.get("project_selection_source") != "optional":
            continue
        native_row = native_by_base_key.get(base_row_key(row))
        if native_row is None:
            continue
        try:
            row_time = float(row.get("selected_time_us"))
            native_time = float(native_row.get("selected_time_us"))
        except (TypeError, ValueError):
            continue
        if native_time < row_time:
            optional_rows_superseded_by_native.append(row)
    audit.require(
        "project-wide native supersession",
        not optional_rows_superseded_by_native,
        (
            "project-wide fastest rows use native trainer evidence when it beats the optional row; "
            f"superseded={len(optional_rows_superseded_by_native)}"
        ),
    )
    audit.require(
        "project-wide Torch fastest partition",
        selection.get("project_torch_fastest_used_row_count") == len(project_torch_used_rows)
        and selection.get("project_torch_fastest_resolved_divergence_row_count")
        == len(project_torch_resolved_rows)
        and selection.get("project_torch_fastest_extra_row_count") == len(project_torch_extra_rows)
        and selection.get("project_torch_fastest_partitioned_row_count")
        == len(project_torch_rows)
        and not project_torch_unpartitioned_keys
        and project_torch_partition_overlap == 0,
        (
            "every Torch fastest row is accounted for as trainer-used, resolved, or extra; "
            f"used={len(project_torch_used_rows)} resolved={len(project_torch_resolved_rows)} "
            f"extra={len(project_torch_extra_rows)} unpartitioned={len(project_torch_unpartitioned_keys)} "
            f"overlap={project_torch_partition_overlap}"
        ),
    )
    audit.require(
        "project-wide Torch fastest disposition",
        selection.get("project_torch_fastest_disposition_row_count") == len(project_torch_rows)
        and selection.get("project_torch_fastest_actionable_row_count")
        == len(project_torch_rows) - len(project_torch_missing_action_rows)
        and selection.get("project_torch_fastest_missing_disposition")
        == project_torch_missing_disposition_labels + project_torch_bad_disposition_labels + project_torch_missing_action_labels
        and not project_torch_missing_disposition_keys
        and not project_torch_extra_disposition_keys
        and not project_torch_bad_disposition_rows
        and not project_torch_missing_action_rows,
        (
            "every Torch fastest row carries a trainer-used/resolved/extra disposition "
            "and an action or reason; "
            f"rows={len(project_torch_disposition_rows)}/{len(project_torch_rows)} "
            f"missing={len(project_torch_missing_disposition_keys)} "
            f"extra={len(project_torch_extra_disposition_keys)} "
            f"bad={len(project_torch_bad_disposition_rows)} "
            f"missing_action={len(project_torch_missing_action_rows)}"
        ),
    )
    audit.require(
        "project-wide unresolved objective rows",
        not project_unresolved_objective_rows
        and selection.get("project_fastest_unresolved_objective_row_count") == 0,
        (
            "every project-wide fastest objective row is trainer-selected or has an inactive decision; "
            f"unresolved={len(project_unresolved_objective_rows)}"
        ),
    )
    audit.require(
        "project-wide resolved divergence count",
        selection.get("project_fastest_resolved_divergence_row_count") == len(project_resolved_divergence_rows),
        (
            "current selection records "
            f"{selection.get('project_fastest_resolved_divergence_row_count')} resolved project-wide divergences"
        ),
    )
    project_computed_used_keys = {row_key(row) for row in project_computed_used_rows}
    project_used_keys = {row_key(row) for row in project_used_rows}
    project_resolved_keys = {row_key(row) for row in project_resolved_rows}
    project_computed_resolved_keys = {row_key(row) for row in project_resolved_divergence_rows}
    project_extra_keys = {row_key(row) for row in project_extra_artifact_rows}
    project_computed_extra_keys = {row_key(row) for row in project_extra_rows}
    project_computed_resolved_call_path_counts = dict(
        sorted(
            Counter(
                str(row.get("trainer_call_path_kind", "unknown"))
                for row in project_resolved_divergence_rows
            ).items()
        )
    )
    project_computed_resolved_status_counts = dict(
        sorted(
            Counter(
                str(row.get("decision_status", "unknown"))
                for row in project_resolved_divergence_rows
            ).items()
        )
    )
    project_computed_resolved_trainer_callable_rows = [
        row
        for row in project_resolved_divergence_rows
        if row.get("trainer_call_path_kind") == "trainer_or_cxx_route"
    ]
    project_computed_resolved_trainer_callable_rows_with_evidence = [
        row
        for row in project_computed_resolved_trainer_callable_rows
        if has_trainer_callable_debt_evidence(row)
    ]
    project_computed_resolved_trainer_callable_missing_evidence = [
        row
        for row in project_computed_resolved_trainer_callable_rows
        if not has_trainer_callable_debt_evidence(row)
    ]
    project_computed_resolved_trainer_callable_missing_labels = [
        f"{row.get('suite')}/{row.get('kernel')}/{row.get('shape')}/{row.get('selected_stack')}"
        for row in project_computed_resolved_trainer_callable_missing_evidence
    ]
    resolved_optional_by_key = {row_key(row): row for row in resolved_rows}
    project_resolved_missing_decision_links = [
        row
        for row in project_resolved_divergence_rows
        if row_key(row) not in resolved_optional_by_key
    ]
    project_resolved_missing_decision_link_labels = [
        f"{row.get('suite')}/{row.get('kernel')}/{row.get('shape')}/{row.get('selected_stack')}"
        for row in project_resolved_missing_decision_links
    ]
    project_resolved_non_trainer_rows = [
        row
        for row in project_resolved_divergence_rows
        if row.get("trainer_call_path_available") is False
    ]
    project_resolved_non_trainer_actionable_rows = [
        row
        for row in project_resolved_non_trainer_rows
        if (
            (resolved_row := resolved_optional_by_key.get(row_key(row)))
            and resolved_row.get("candidate_class")
            and resolved_row.get("promotion_gate")
            and resolved_row.get("priority")
        )
    ]
    project_resolved_non_trainer_missing_action = [
        row
        for row in project_resolved_non_trainer_rows
        if row not in project_resolved_non_trainer_actionable_rows
    ]
    project_resolved_non_trainer_missing_action_labels = [
        f"{row.get('suite')}/{row.get('kernel')}/{row.get('shape')}/{row.get('selected_stack')}"
        for row in project_resolved_non_trainer_missing_action
    ]
    project_fastest_by_key = {row_key(row): row for row in project_fastest_rows}
    project_partition_content_mismatches = (
        partition_row_content_mismatches(project_used_rows, project_fastest_by_key, "used")
        + partition_row_content_mismatches(project_resolved_rows, project_fastest_by_key, "resolved")
        + partition_row_content_mismatches(project_extra_artifact_rows, project_fastest_by_key, "extra")
    )
    audit.require(
        "project-wide used row identity",
        selection.get("project_fastest_used_row_count") == len(project_computed_used_rows)
        and project_used_keys == project_computed_used_keys,
        (
            "current selection lists the project-wide fastest rows that are used by trainer; "
            f"count={selection.get('project_fastest_used_row_count')} expected={len(project_computed_used_rows)} "
            f"missing={len(project_computed_used_keys - project_used_keys)} extra={len(project_used_keys - project_computed_used_keys)}"
        ),
    )
    audit.require(
        "project-wide resolved row identity",
        project_resolved_keys == project_computed_resolved_keys,
        (
            "current selection lists the project-wide fastest rows resolved away from trainer; "
            f"missing={len(project_computed_resolved_keys - project_resolved_keys)} "
            f"extra={len(project_resolved_keys - project_computed_resolved_keys)}"
        ),
    )
    audit.require(
        "project-wide resolved row debt counts",
        selection.get("project_fastest_resolved_call_path_counts") == project_computed_resolved_call_path_counts
        and selection.get("project_fastest_resolved_status_counts") == project_computed_resolved_status_counts,
        (
            "current selection summarizes fastest rows not used by trainer by call path and decision; "
            f"call_paths={selection.get('project_fastest_resolved_call_path_counts')} "
            f"expected={project_computed_resolved_call_path_counts} "
            f"statuses={selection.get('project_fastest_resolved_status_counts')} "
            f"expected_statuses={project_computed_resolved_status_counts}"
        ),
    )
    audit.require(
        "project-wide trainer-callable debt evidence",
        selection.get("project_fastest_resolved_trainer_callable_row_count")
        == len(project_computed_resolved_trainer_callable_rows)
        and selection.get("project_fastest_resolved_trainer_callable_evidence_count")
        == len(project_computed_resolved_trainer_callable_rows_with_evidence)
        and selection.get("project_fastest_resolved_trainer_callable_missing_evidence")
        == project_computed_resolved_trainer_callable_missing_labels
        and not project_computed_resolved_trainer_callable_missing_evidence,
        (
            "trainer/C++ callable fastest rows resolved away from trainer carry "
            "trainer-smoke or x10 stability evidence; "
            f"with_evidence={len(project_computed_resolved_trainer_callable_rows_with_evidence)}/"
            f"{len(project_computed_resolved_trainer_callable_rows)} "
            f"missing={len(project_computed_resolved_trainer_callable_missing_evidence)}"
        ),
    )
    audit.require(
        "project-wide resolved decision linkage",
        selection.get("project_fastest_resolved_decision_link_count")
        == len(project_resolved_divergence_rows) - len(project_resolved_missing_decision_links)
        and selection.get("project_fastest_resolved_missing_decision_links")
        == project_resolved_missing_decision_link_labels
        and selection.get("project_fastest_resolved_non_trainer_row_count")
        == len(project_resolved_non_trainer_rows)
        and selection.get("project_fastest_resolved_non_trainer_actionable_count")
        == len(project_resolved_non_trainer_actionable_rows)
        and selection.get("project_fastest_resolved_non_trainer_missing_action")
        == project_resolved_non_trainer_missing_action_labels
        and not project_resolved_missing_decision_links
        and not project_resolved_non_trainer_missing_action,
        (
            "project-wide fastest rows resolved away from trainer link to resolved "
            "decision rows, and non-trainer rows carry action metadata there; "
            f"linked={len(project_resolved_divergence_rows) - len(project_resolved_missing_decision_links)}/"
            f"{len(project_resolved_divergence_rows)} "
            f"actionable={len(project_resolved_non_trainer_actionable_rows)}/"
            f"{len(project_resolved_non_trainer_rows)}"
        ),
    )
    audit.require(
        "project-wide extra row identity",
        project_extra_keys == project_computed_extra_keys,
        (
            "current selection lists the extra project-wide benchmark rows; "
            f"missing={len(project_computed_extra_keys - project_extra_keys)} "
            f"extra={len(project_extra_keys - project_computed_extra_keys)}"
        ),
    )
    audit.require(
        "project-wide partition row content",
        not project_partition_content_mismatches,
        (
            "current selection project-wide used/resolved/extra rows preserve the full "
            f"source project-fastest row content; mismatches={len(project_partition_content_mismatches)}"
        ),
    )
    project_resolved_missing_decision_detail = [
        row
        for row in project_resolved_divergence_rows
        if not row.get("decision_decision") or not row.get("decision_evidence")
    ]
    audit.require(
        "project-wide resolved divergence detail",
        not project_resolved_missing_decision_detail,
        "project-wide fastest rows resolved away from trainer carry decision reason and evidence; "
        f"missing={len(project_resolved_missing_decision_detail)}",
    )
    torch_fc_dw_key = (
        "matmul",
        "dW",
        "fc M=65536 N=3072 K=768 bias=1 gelu=1",
        "Torch",
    )
    torch_fc_dw_row = resolved_optional_by_key.get(torch_fc_dw_key)
    libtorch_fc_log = Path("scratch/sm120_rounds/libtorch_matmul_fc_20260521/bench_sm120_libtorch_matmul.log")
    libtorch_fc_text = libtorch_fc_log.read_text(errors="replace") if libtorch_fc_log.exists() else ""
    torch_fc_dw_evidence = " ".join(str(item) for item in (torch_fc_dw_row or {}).get("decision_evidence", []))
    torch_fc_dw_decision = str((torch_fc_dw_row or {}).get("decision_decision", ""))
    torch_fc_dw_libtorch_proven = (
        torch_fc_dw_row is None
        or (
            torch_fc_dw_row.get("decision_active") is False
            and "standalone LibTorch C++" in torch_fc_dw_decision
            and "standalone LibTorch C++ raw-pointer probe" in torch_fc_dw_evidence
            and "dW 1355.76 us" in torch_fc_dw_evidence
            and "dW+accum 1372.33 us" in torch_fc_dw_evidence
            and "optional cuBLAS dW 1330.140 us" in torch_fc_dw_evidence
            and "native trainer-backed cuBLAS dW 1309.120 us" in torch_fc_dw_evidence
            and "LibTorch matmul parity dW fc: PASS" in libtorch_fc_text
            and "LibTorch matmul parity dW+accum fc: PASS" in libtorch_fc_text
            and "dW       Torch C++   1355.76 us" in libtorch_fc_text
            and "dW+accum Torch C++   1372.33 us" in libtorch_fc_text
        )
    )
    audit.require(
        "Torch fc dWeight LibTorch rejection evidence",
        torch_fc_dw_libtorch_proven,
        (
            "Torch fc dW resolved row carries standalone LibTorch C++ parity/timing "
            f"evidence from {libtorch_fc_log}"
        ),
    )
    current_all_shape_libtorch_log = optional_round / "bench_sm120_libtorch_matmul.log"
    current_all_shape_libtorch_json = optional_round / "bench_sm120_libtorch_matmul.json"
    all_shape_libtorch_log = (
        current_all_shape_libtorch_log
        if current_all_shape_libtorch_log.exists()
        else LIBTORCH_ALL_SHAPE_MATMUL_LOG
    )
    all_shape_libtorch_json = (
        current_all_shape_libtorch_json
        if current_all_shape_libtorch_json.exists()
        else LIBTORCH_ALL_SHAPE_MATMUL_JSON
    )
    all_shape_libtorch_text = (
        all_shape_libtorch_log.read_text(errors="replace")
        if all_shape_libtorch_log.exists()
        else ""
    )
    all_shape_required = {
        (op_name, shape)
        for shape in MATMUL_SELECTION_SHAPES.values()
        for op_name in ("dW", "dW+accum")
    }
    if all_shape_libtorch_json.exists():
        (
            all_shape_json_route,
            all_shape_rows,
            all_shape_parity,
            all_shape_timings,
        ) = parse_libtorch_matmul_json_evidence(all_shape_libtorch_json)
    else:
        all_shape_json_route = ""
        all_shape_rows, all_shape_parity, all_shape_timings = parse_libtorch_matmul_evidence(
            all_shape_libtorch_text
        )
    missing_all_shape_rows = sorted(all_shape_required - all_shape_rows)
    missing_all_shape_parity = sorted(all_shape_required - all_shape_parity)
    native_dweight_timings = {
        (str(row.get("kernel")), str(row.get("shape"))): float(row.get("selected_time_us", 0.0))
        for row in native_rows
        if row.get("suite") == "matmul" and row.get("kernel") in {"dW", "dW+accum"}
    }
    all_shape_not_faster = [
        key
        for key in sorted(all_shape_required)
        if key in all_shape_timings
        and key in native_dweight_timings
        and all_shape_timings[key] > native_dweight_timings[key]
    ]
    missing_native_comparisons = sorted(
        key
        for key in all_shape_required
        if key not in native_dweight_timings
    )
    audit.require(
        "LibTorch all-shape dWeight route marker",
        all_shape_json_route == LIBTORCH_MATMUL_ROUTE
        or LIBTORCH_MATMUL_ROUTE in all_shape_libtorch_text,
        (
            f"{all_shape_libtorch_json} or {all_shape_libtorch_log} "
            "records standalone C++ cached from_blob route evidence"
        ),
    )
    audit.require(
        "LibTorch all-shape dWeight structured evidence",
        native_round.name == "native" or all_shape_libtorch_json.exists(),
        f"{all_shape_libtorch_json} exists for machine-readable all-shape evidence",
    )
    audit.require(
        "LibTorch all-shape dWeight parity/timing coverage",
        not missing_all_shape_rows and not missing_all_shape_parity,
        (
            "supplemental LibTorch C++ dWeight probe covers every GPT-2 dW/dW+accum "
            f"shape; timing_missing={len(missing_all_shape_rows)} parity_missing={len(missing_all_shape_parity)}"
        ),
    )
    audit.require(
        "LibTorch all-shape dWeight rejection comparison",
        native_round.name == "native"
        or (len(all_shape_not_faster) == len(all_shape_required) and not missing_native_comparisons),
        (
            "supplemental LibTorch C++ dWeight rows are slower than the effective native "
            f"trainer rows for {len(all_shape_not_faster)}/{len(all_shape_required)} shapes; "
            f"missing_native={len(missing_native_comparisons)}"
        ),
    )
    torch_memory_rows = [
        row
        for row in project_resolved_divergence_rows
        if row.get("suite") == "runtime"
        and row.get("kernel") in {"cuda_memset", "cuda_copy_d2d"}
        and is_torch_stack(row.get("selected_stack"))
    ]
    libtorch_runtime_log = optional_round / "bench_sm120_libtorch_runtime.log"
    libtorch_runtime_text = libtorch_runtime_log.read_text(errors="replace") if libtorch_runtime_log.exists() else ""
    trainer_link_log = LIBTORCH_SUPPLEMENTAL_TRAINER_LINK_LOGS[0]
    trainer_link_text = trainer_link_log.read_text(errors="replace") if trainer_link_log.exists() else ""
    memory_evidence_missing: list[str] = []
    for row in torch_memory_rows:
        key = (str(row.get("kernel")), str(row.get("shape")))
        decision_status = str(row.get("decision_status", ""))
        evidence_text = " ".join(
            str(item)
            for item in (
                list(row.get("decision_evidence", []))
                + [
                    row.get("decision_decision", ""),
                    row.get("decision_note", ""),
                    row.get("promotion_gate", ""),
                ]
            )
        )
        parity_line = f"LibTorch parity {key[0]} {key[1]}: PASS"
        has_timing_line = any(
            line.startswith(f"{key[0]:<30} | {key[1]:<28} | Torch C++")
            for line in libtorch_runtime_text.splitlines()
        )
        missing = (
            "LibTorch" not in evidence_text
            or "full-row parity PASS" not in evidence_text
            or "CUDA runtime" not in evidence_text
            or parity_line not in libtorch_runtime_text
            or not has_timing_line
        )
        if decision_status == "profiler_only_runtime_row":
            missing = missing or "no " not in evidence_text or "call-site" not in evidence_text
        elif decision_status == "rejected_x10_trainer_route" and "x10" in evidence_text:
            missing = (
                parity_line not in libtorch_runtime_text
                or not has_timing_line
                or "trainer" not in evidence_text
                or "avg_ms" not in evidence_text
                or "LibTorch trainer link probe: PASS" not in trainer_link_text
            )
        else:
            missing = (
                missing
                or "standalone trainer-link probe" not in evidence_text
                or "zero/copy from_blob probe PASS" not in evidence_text
                or "LibTorch trainer link probe: PASS" not in trainer_link_text
            )
        if missing:
            memory_evidence_missing.append(
                f"{row.get('suite')}/{row.get('kernel')}/{row.get('shape')}/{row.get('selected_stack')}"
            )
    audit.require(
        "Torch runtime memory LibTorch rejection evidence",
        not memory_evidence_missing,
        (
            "Torch/Torch C++ runtime-memory rows resolved away from trainer carry "
            f"raw-pointer parity/timing evidence from {libtorch_runtime_log} and "
            f"trainer-link evidence from {trainer_link_log}; "
            f"rows={len(torch_memory_rows)} missing={len(memory_evidence_missing)}"
        ),
    )
    audit.require(
        "project-wide extra row count",
        selection.get("project_fastest_extra_row_count") == len(project_extra_rows),
        f"current selection records {selection.get('project_fastest_extra_row_count')} extra project-wide benchmark rows",
    )
    missing_extra_reasons = [row for row in project_extra_rows if not row.get("project_extra_reason")]
    resolved_optional_missing_promotion_metadata = [
        row
        for row in resolved_rows
        if row.get("trainer_call_path_available") is False
        and (
            not row.get("candidate_class")
            or not row.get("promotion_gate")
            or not row.get("priority")
        )
    ]
    resolved_optional_missing_decision_evidence = [
        row
        for row in resolved_rows
        if row.get("decision_status") is not None
        and not row.get("decision_evidence")
    ]
    native_complete_attention_routes = [
        row
        for row in native_attention_routes
        if row.get("complete") is True
        and row.get("trainer_layout") is True
        and row.get("stack") == "TK packed-QKV"
    ]
    optional_packed_attention_stacks = {
        str(row.get("stack"))
        for row in optional_attention_routes
        if row.get("trainer_layout") is True
    }
    optional_incomplete_triton_packed = [
        row
        for row in optional_attention_routes
        if row.get("stack") == "TritonPacked"
        and row.get("complete") is False
        and row.get("unavailable_reason")
    ]
    audit.require(
        "project-wide extra row reasons",
        not missing_extra_reasons,
        f"every extra project-wide benchmark row has an explicit reason; missing={len(missing_extra_reasons)}",
    )
    audit.require(
        "selection row provenance",
        not rows_without_provenance,
        f"native/resolved/project selection rows reference existing timing, config, stack-probe, and correctness evidence; missing={len(rows_without_provenance)}",
    )
    audit.require(
        "runtime profiler memory call-path kind",
        not profiler_runtime_rows_with_wrong_call_path,
        (
            "runtime cuda_copy_d2d rows and logits-sized memset rows are marked as "
            "profiler-runtime benchmark evidence, not trainer-callable replacements; "
            f"mismatches={len(profiler_runtime_rows_with_wrong_call_path)}"
        ),
    )
    audit.require(
        "resolved optional promotion metadata",
        not resolved_optional_missing_promotion_metadata,
        "resolved non-trainer optional decisions carry candidate class, priority, and promotion gate; "
        f"missing={len(resolved_optional_missing_promotion_metadata)}",
    )
    audit.require(
        "resolved optional decision evidence",
        not resolved_optional_missing_decision_evidence,
        "resolved optional decisions carry explicit evidence text; "
        f"missing={len(resolved_optional_missing_decision_evidence)}",
    )
    audit.require(
        "attention route totals",
        bool(native_complete_attention_routes)
        and {"TorchPacked", "TorchMaterializedPacked", "cuDNNPacked", "TritonPacked"}.issubset(optional_packed_attention_stacks)
        and bool(optional_incomplete_triton_packed),
        "current selection records native TK plus optional packed attention route totals, including incomplete TritonPacked evidence",
    )
    audit.require(
        "attention route source content",
        len(native_attention_routes) == len(native_source_attention_routes)
        and len(optional_attention_routes) == len(optional_source_attention_routes)
        and not native_route_content_mismatches
        and not optional_route_content_mismatches,
        (
            "current selection attention route rows match source selected-backends route totals; "
            f"native={len(native_attention_routes)}/{len(native_source_attention_routes)} "
            f"optional={len(optional_attention_routes)}/{len(optional_source_attention_routes)} "
            f"mismatches={len(native_route_content_mismatches) + len(optional_route_content_mismatches)}"
        ),
    )
    audit.require(
        "trainer objective family coverage",
        not missing_families,
        f"current trainer mix covers {len(family_coverage) - len(missing_families)}/{len(family_coverage)} objective families",
    )
    audit.require(
        "trainer exact row coverage",
        not missing_exact_rows,
        f"current trainer mix covers {len(exact_row_coverage) - len(missing_exact_rows)}/{len(exact_row_coverage)} required exact rows",
    )
    return {
        "native_trainer_rows": len(native_rows),
        "trainer_stack_counts": dict(sorted(trainer_stack_counts.items())),
        "trainer_family_coverage": family_coverage,
        "trainer_exact_row_coverage": exact_row_coverage,
        "torch_trainer_rows": len(torch_trainer_rows),
        "resolved_optional_decisions": len(resolved_rows),
        "project_fastest_rows": len(project_fastest_rows),
        "project_torch_fastest_rows": len(project_torch_rows),
        "project_torch_fastest_used_rows": len(project_torch_used_rows),
        "project_torch_fastest_resolved_rows": len(project_torch_resolved_rows),
        "project_torch_fastest_extra_rows": len(project_torch_extra_rows),
        "project_torch_fastest_unpartitioned_rows": len(project_torch_unpartitioned_keys),
        "project_torch_fastest_disposition_rows": len(project_torch_disposition_rows),
        "project_torch_fastest_rows_missing_disposition": len(project_torch_missing_disposition_keys),
        "project_torch_fastest_rows_with_bad_disposition": len(project_torch_bad_disposition_rows),
        "project_torch_fastest_rows_missing_action": len(project_torch_missing_action_rows),
        "project_fastest_used_rows": len(project_computed_used_rows),
        "project_fastest_resolved_divergence_rows": len(project_resolved_divergence_rows),
        "project_fastest_resolved_call_path_counts": project_computed_resolved_call_path_counts,
        "project_fastest_resolved_status_counts": project_computed_resolved_status_counts,
        "project_fastest_resolved_trainer_callable_rows": len(
            project_computed_resolved_trainer_callable_rows
        ),
        "project_fastest_resolved_trainer_callable_rows_with_evidence": len(
            project_computed_resolved_trainer_callable_rows_with_evidence
        ),
        "project_fastest_resolved_trainer_callable_rows_missing_evidence": len(
            project_computed_resolved_trainer_callable_missing_evidence
        ),
        "project_fastest_resolved_decision_link_rows": (
            len(project_resolved_divergence_rows) - len(project_resolved_missing_decision_links)
        ),
        "project_fastest_resolved_missing_decision_links": len(project_resolved_missing_decision_links),
        "project_fastest_resolved_non_trainer_rows": len(project_resolved_non_trainer_rows),
        "project_fastest_resolved_non_trainer_actionable_rows": len(
            project_resolved_non_trainer_actionable_rows
        ),
        "project_fastest_resolved_non_trainer_missing_action_rows": len(
            project_resolved_non_trainer_missing_action
        ),
        "project_fastest_extra_rows": len(project_extra_rows),
        "project_fastest_extra_rows_missing_reasons": len(missing_extra_reasons),
        "resolved_optional_rows_missing_promotion_metadata": len(resolved_optional_missing_promotion_metadata),
        "native_attention_route_rows": len(native_attention_routes),
        "optional_attention_route_rows": len(optional_attention_routes),
        "project_fastest_unresolved_objective_rows": len(project_unresolved_objective_rows),
        "project_fastest_keys": [row_key(row) for row in project_fastest_rows],
        "project_torch_fastest_keys": [row_key(row) for row in project_torch_rows],
        "project_fastest_row_records": project_fastest_rows,
        "project_torch_fastest_row_records": project_torch_rows,
        "selection_optional_non_trainer_selected_rows": selection.get(
            "optional_non_trainer_selected_row_count"
        ),
        "selection_optional_non_trainer_promotion_rows": selection.get(
            "optional_non_trainer_promotion_row_count"
        ),
        "resolved_optional_keys": [row_key(row) for row in resolved_rows],
        "resolved_optional_row_records": resolved_rows,
    }


def audit_optional_stack_rows(
    audit: Audit,
    optional_round: Path,
    resolved_optional_keys: list[tuple[str, str, str, str]],
    resolved_optional_row_records: object,
    selection_project_fastest_rows: object,
    selection_project_torch_fastest_rows: object,
    selection_project_fastest_row_records: object,
    selection_project_torch_fastest_row_records: object,
    selection_project_fastest_keys: list[tuple[str, str, str, str]],
    selection_project_torch_fastest_keys: list[tuple[str, str, str, str]],
    selection_optional_non_trainer_selected_rows: object,
    selection_optional_non_trainer_promotion_rows: object,
) -> dict[str, Any]:
    optional_selected = read_json(optional_round / "selected-backends.json")
    optional_promotions = read_json(optional_round / "promotion-candidates.json")
    selected_rows = optional_selected.get("selected_backend_rows", [])
    promotion_rows = optional_promotions.get("promotion_candidates", [])
    promotion_by_key = {row_key(row): row for row in promotion_rows}
    resolved_key_set = set(resolved_optional_keys)
    active_promotions = optional_promotions.get("active_promotion_candidates", [])
    torch_rows = [row for row in selected_rows if is_torch_stack(row.get("selected_stack"))]
    optional_torch_keys = {row_key(row) for row in torch_rows}
    optional_selected_base_keys = {base_row_key(row) for row in selected_rows}
    selection_fastest_key_set = set(selection_project_fastest_keys)
    selection_torch_key_set = set(selection_project_torch_fastest_keys)
    selection_fastest_base_key_set = {
        (suite, kernel, shape)
        for suite, kernel, shape, _stack in selection_fastest_key_set
    }
    selected_by_key = {row_key(row): row for row in selected_rows}
    torch_by_key = {row_key(row): row for row in torch_rows}
    resolved_optional_content_mismatches: list[str] = []
    if isinstance(resolved_optional_row_records, list):
        for row in resolved_optional_row_records:
            key = row_key(row)
            source_row = promotion_by_key.get(key) or selected_by_key.get(key)
            if source_row is None:
                resolved_optional_content_mismatches.append(f"{'/'.join(key)}:missing_source")
                continue
            mismatches = resolved_optional_mismatches(row, source_row)
            if mismatches:
                resolved_optional_content_mismatches.append(
                    f"{'/'.join(key)}:{','.join(mismatches)}"
                )
    else:
        resolved_optional_content_mismatches.append("resolved_optional_stack_decisions:not_list")
    missing_fastest_keys = optional_selected_base_keys - selection_fastest_base_key_set
    extra_fastest_keys = selection_fastest_base_key_set - optional_selected_base_keys
    non_optional_torch_keys = selection_torch_key_set - optional_torch_keys
    missing_torch_keys = optional_torch_keys - selection_torch_key_set
    fastest_content_mismatches: list[str] = []
    if isinstance(selection_project_fastest_row_records, list):
        for row in selection_project_fastest_row_records:
            if row.get("project_selection_source") == "native":
                continue
            source_row = selected_by_key.get(row_key(row))
            if source_row is None:
                continue
            mismatches = source_field_mismatches(row, source_row)
            if mismatches:
                fastest_content_mismatches.append(
                    f"{'/'.join(row_key(row))}:{','.join(mismatches)}"
                )
    else:
        fastest_content_mismatches.append("project_fastest_selection:not_list")
    torch_content_mismatches: list[str] = []
    if isinstance(selection_project_torch_fastest_row_records, list):
        for row in selection_project_torch_fastest_row_records:
            if row.get("project_selection_source") == "native":
                continue
            source_row = torch_by_key.get(row_key(row))
            if source_row is None:
                continue
            mismatches = source_field_mismatches(row, source_row)
            if mismatches:
                torch_content_mismatches.append(
                    f"{'/'.join(row_key(row))}:{','.join(mismatches)}"
                )
    else:
        torch_content_mismatches.append("project_torch_fastest_rows:not_list")
    optional_non_trainer_rows = [
        row for row in selected_rows if not row.get("trainer_call_path_available")
    ]
    missing_promotion_rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    missing_from_selection: list[dict[str, Any]] = []

    for row in optional_non_trainer_rows:
        promotion_row = promotion_by_key.get(row_key(row))
        if promotion_row is None:
            missing_promotion_rows.append(row)
            promotion_row = {}
        decision_status = row.get("decision_status") or promotion_row.get("decision_status")
        decision_active = row.get("decision_active")
        if decision_active is None:
            decision_active = promotion_row.get("decision_active")
        if decision_status is None or decision_active is not False:
            unresolved.append(row)
        elif row_key(row) not in resolved_key_set:
            missing_from_selection.append(row)

    audit.require(
        "Torch selected rows",
        bool(torch_rows),
        f"optional-stack round has {len(torch_rows)} Torch-selected benchmark rows",
    )
    audit.require(
        "optional non-trainer selected rows",
        bool(optional_non_trainer_rows),
        f"optional-stack round has {len(optional_non_trainer_rows)} non-trainer selected rows",
    )
    audit.require(
        "selection optional non-trainer count",
        selection_optional_non_trainer_selected_rows == len(optional_non_trainer_rows),
        "current selection records "
        f"{selection_optional_non_trainer_selected_rows} optional non-trainer rows",
    )
    audit.require(
        "selection optional promotion count",
        selection_optional_non_trainer_promotion_rows == len(optional_non_trainer_rows),
        "current selection records "
        f"{selection_optional_non_trainer_promotion_rows} optional non-trainer promotion rows",
    )
    audit.require(
        "selection project-wide fastest count",
        selection_project_fastest_rows == len(selected_rows),
        f"current selection records {selection_project_fastest_rows} project-wide fastest rows",
    )
    audit.require(
        "selection project-wide fastest identity",
        not missing_fastest_keys and not extra_fastest_keys,
        (
            "current selection project-wide fastest rows cover the optional selected-backends base rows; "
            f"missing={len(missing_fastest_keys)} extra={len(extra_fastest_keys)}"
        ),
    )
    audit.require(
        "selection project-wide fastest content",
        not fastest_content_mismatches,
        (
            "current selection project-wide fastest rows preserve source selected-backends stack, "
            f"time, scope, and provenance fields; mismatches={len(fastest_content_mismatches)}"
        ),
    )
    audit.require(
        "selection project-wide Torch fastest count",
        selection_project_torch_fastest_rows == len(selection_torch_key_set),
        (
            "current selection records "
            f"{selection_project_torch_fastest_rows} project-wide Torch fastest rows"
        ),
    )
    audit.require(
        "selection project-wide Torch fastest identity",
        not non_optional_torch_keys,
        (
            "current selection Torch fastest rows are selected from optional selected-backends Torch rows; "
            f"superseded_optional_torch={len(missing_torch_keys)} extra={len(non_optional_torch_keys)}"
        ),
    )
    audit.require(
        "selection project-wide Torch fastest content",
        not torch_content_mismatches,
        (
            "current selection Torch fastest rows preserve source selected-backends stack, "
            f"time, scope, and provenance fields; mismatches={len(torch_content_mismatches)}"
        ),
    )
    audit.require(
        "selection resolved optional content",
        not resolved_optional_content_mismatches,
        (
            "current selection resolved optional decisions preserve source promotion-candidate "
            "or selected-backend decision, timing, scope, and provenance fields; "
            f"mismatches={len(resolved_optional_content_mismatches)}"
        ),
    )
    audit.require(
        "optional promotion row coverage",
        not missing_promotion_rows,
        f"all non-trainer optional wins have matching promotion rows; missing={len(missing_promotion_rows)}",
    )
    audit.require(
        "optional decision coverage",
        not unresolved,
        f"all non-trainer optional wins are resolved or explicitly inactive; unresolved={len(unresolved)}",
    )
    audit.require(
        "optional consolidated decisions",
        not missing_from_selection,
        f"all resolved non-trainer optional wins are present in current selection; missing={len(missing_from_selection)}",
    )
    audit.require(
        "optional active promotions",
        not active_promotions,
        f"optional-stack promotion backlog has {len(active_promotions)} active candidates",
    )
    for log_name in PYTHON_STACK_BENCHMARK_LOGS:
        path = optional_round / log_name
        audit.require(
            f"Python stack log {log_name}",
            path.exists() and bool(path.read_text().strip()),
            f"{path} exists and is non-empty",
        )
    libtorch_matmul_log_count = 0
    libtorch_matmul_row_count = 0
    if round_manifest_flag(optional_round, "run_libtorch_matmul_benchmarks"):
        path = optional_round / "bench_sm120_libtorch_matmul.log"
        text = path.read_text() if path.exists() else ""
        shape_names = round_libtorch_matmul_shapes(optional_round) or ("fc",)
        unknown_shapes = sorted(set(shape_names) - set(MATMUL_SELECTION_SHAPES))
        required = {
            (op_name, MATMUL_SELECTION_SHAPES[shape_name])
            for shape_name in shape_names
            if shape_name in MATMUL_SELECTION_SHAPES
            for op_name in ("dW", "dW+accum")
        }
        observed, _, _ = parse_libtorch_matmul_evidence(text)
        missing_libtorch_matmul_rows = sorted(required - observed)
        libtorch_matmul_log_count = 1 if path.exists() and bool(text.strip()) else 0
        libtorch_matmul_row_count = len(observed & required)
        audit.require(
            "LibTorch matmul log",
            libtorch_matmul_log_count == 1,
            f"{path} exists and is non-empty when run_libtorch_matmul_benchmarks=1",
        )
        audit.require(
            "LibTorch matmul route marker",
            LIBTORCH_MATMUL_ROUTE in text,
            f"{path} records standalone C++ cached from_blob route evidence",
        )
        audit.require(
            "LibTorch matmul requested shapes",
            not unknown_shapes,
            f"optional manifest requested known LibTorch matmul shapes; unknown={unknown_shapes}",
        )
        audit.require(
            "LibTorch matmul dWeight coverage",
            not missing_libtorch_matmul_rows,
            (
                "optional round records Torch C++ dW/dW+accum rows for requested shapes; "
                f"covered={libtorch_matmul_row_count}/{len(required)} missing={len(missing_libtorch_matmul_rows)}"
            ),
        )
    return {
        "optional_non_trainer_selected_rows": len(optional_non_trainer_rows),
        "optional_resolved_non_trainer_rows": len(optional_non_trainer_rows) - len(missing_from_selection),
        "torch_selected_rows": len(torch_rows),
        "project_fastest_identity_missing": len(missing_fastest_keys),
        "project_fastest_identity_extra": len(extra_fastest_keys),
        "project_torch_identity_missing": len(missing_torch_keys),
        "project_torch_identity_extra": len(non_optional_torch_keys),
        "torch_promotion_rows": sum(1 for row in promotion_rows if is_torch_stack(row.get("selected_stack"))),
        "python_stack_log_count": len(PYTHON_STACK_BENCHMARK_LOGS),
        "libtorch_matmul_log_count": libtorch_matmul_log_count,
        "libtorch_matmul_rows": libtorch_matmul_row_count,
        "active_promotion_candidates": len(active_promotions),
    }


def audit_torch_objective_benchmark_coverage(audit: Audit, optional_round: Path) -> dict[str, Any]:
    scoreboard = optional_round / "scoreboard-candidates.md"
    text = scoreboard.read_text() if scoreboard.exists() else ""
    marker = "## Torch Objective Benchmark Coverage"
    audit.require(
        "Torch objective benchmark section",
        marker in text,
        f"{scoreboard} contains Torch objective benchmark coverage",
    )
    section = text.split(marker, 1)[1].split("\n## ", 1)[0]
    rows = [
        line
        for line in section.splitlines()
        if line.startswith("| `") and " | `" in line
    ]
    false_rows = [line for line in rows if line.rstrip().endswith("| `False` |")]
    expected_count = len(expected_trainer_selection_keys())
    audit.require(
        "Torch exact benchmark row coverage",
        len(rows) == expected_count and not false_rows,
        f"{scoreboard} records {len(rows)}/{expected_count} covered Torch objective benchmark rows; missing={len(false_rows)}",
    )
    return {
        "torch_objective_benchmark_rows": len(rows),
        "expected_torch_objective_benchmark_rows": expected_count,
        "missing_torch_objective_benchmark_rows": len(false_rows),
    }


def audit_cutedsl_gemm_feasibility(audit: Audit, optional_round: Path) -> dict[str, Any]:
    log_path = optional_round / "bench_sm120_cutedsl_matmul.log"
    text = log_path.read_text() if log_path.exists() else ""
    expected_shapes = set(MATMUL_SELECTION_SHAPES.values())
    row_results: dict[str, str] = {}
    unavailable_reasons: dict[str, str] = {}
    timed_shapes: set[str] = set()
    bad_rows: dict[str, str] = {}

    for raw in text.splitlines():
        match = CUTEDSL_RESULT_RE.match(raw.strip())
        if not match:
            continue
        shape = " ".join(match.group("shape").split())
        if shape not in expected_shapes:
            continue
        result = match.group("result").strip()
        row_results[shape] = result
        if result.startswith("unavailable:"):
            reason = result.split("unavailable:", 1)[1].strip()
            unavailable_reasons[shape] = reason
            if not reason:
                bad_rows[shape] = result
        elif re.match(r"^[0-9]+(?:\.[0-9]+)?\s+us$", result):
            timed_shapes.add(shape)
        else:
            bad_rows[shape] = result

    missing = sorted(expected_shapes - set(row_results))
    target_rejection_rows = [
        shape
        for shape, reason in unavailable_reasons.items()
        if "rejects sm_120" in reason or "rejects sm_120a" in reason
    ]
    audit.require(
        "CuTeDSL compile probe marker",
        "Running Blackwell Grouped GEMM test" in text or bool(timed_shapes),
        f"{log_path} records a CuTeDSL grouped-GEMM compile/probe attempt",
    )
    audit.require(
        "CuTeDSL exact GEMM feasibility rows",
        not missing and not bad_rows and (bool(timed_shapes) or len(target_rejection_rows) == len(expected_shapes)),
        (
            f"{log_path} records exact CuTeDSL rows for {len(row_results)}/{len(expected_shapes)} GPT-2 GEMM "
            f"shapes; timed={len(timed_shapes)} unavailable={len(unavailable_reasons)} "
            f"target_rejections={len(target_rejection_rows)} missing={len(missing)} bad={len(bad_rows)}"
        ),
    )
    return {
        "expected_cutedsl_gemm_rows": len(expected_shapes),
        "cutedsl_gemm_rows": len(row_results),
        "cutedsl_timed_rows": len(timed_shapes),
        "cutedsl_unavailable_rows": len(unavailable_reasons),
        "cutedsl_target_rejection_rows": len(target_rejection_rows),
        "missing_cutedsl_gemm_rows": missing,
        "bad_cutedsl_gemm_rows": bad_rows,
    }


def expected_matmul_operator_keys() -> set[tuple[str, str]]:
    return {
        (kernel, MATMUL_SELECTION_SHAPES[shape_name])
        for kernel, shape_names in MATMUL_SHAPE_REQUIREMENTS
        for shape_name in shape_names
    }


def audit_triton_matmul_coverage(audit: Audit, optional_round: Path) -> dict[str, Any]:
    log_path = optional_round / "bench_sm120_triton_matmul.log"
    text = log_path.read_text() if log_path.exists() else ""
    current_shape = ""
    timed_rows: set[tuple[str, str]] = set()
    bad_rows: list[str] = []
    kernel_names = {
        "fwd": "fwd",
        "fwd+GeLU": "fwd+gelu",
        "dInp": "dInp",
        "dInp+dGeLU": "dInp+dGeLU",
        "dW": "dW",
        "dW+accum": "dW+accum",
    }

    for raw in text.splitlines():
        line = raw.strip()
        shape_match = MATMUL_SHAPE_RE.match(line)
        if shape_match:
            current_shape = (
                f"{shape_match.group('name')} M={shape_match.group('m')} N={shape_match.group('n')} "
                f"K={shape_match.group('k')} bias={shape_match.group('bias')} gelu={shape_match.group('gelu')}"
            )
            continue
        if not current_shape:
            continue
        result_match = TRITON_MATMUL_RESULT_RE.match(line)
        if not result_match:
            continue
        kernel = kernel_names[result_match.group("kernel")]
        time_us = float(result_match.group("us"))
        if time_us <= 0.0:
            bad_rows.append(line)
        timed_rows.add((kernel, current_shape))

    expected = expected_matmul_operator_keys()
    missing = sorted(expected - timed_rows)
    audit.require(
        "Triton GEMM exact coverage",
        not missing and not bad_rows,
        (
            f"{log_path} records Triton timing rows for {len(timed_rows & expected)}/{len(expected)} "
            f"GPT-2 GEMM objective rows; missing={len(missing)} bad={len(bad_rows)}"
        ),
    )
    return {
        "expected_triton_matmul_rows": len(expected),
        "triton_matmul_rows": len(timed_rows & expected),
        "missing_triton_matmul_rows": missing,
        "bad_triton_matmul_rows": bad_rows,
    }


def attention_shape_from_match(match: re.Match[str]) -> str:
    return (
        f"B={match.group('b')} T={match.group('t')} C={match.group('c')} "
        f"NH={match.group('nh')} HS={match.group('hs')}"
    )


def audit_optional_attention_stack_coverage(audit: Audit, optional_round: Path) -> dict[str, Any]:
    timed_rows: set[tuple[str, str]] = set()
    unavailable_rows: dict[tuple[str, str], str] = {}
    bad_unavailable: dict[str, str] = {}
    config = round_manifest_config(optional_round)
    expected_cudnn_route = str(config.get("cudnn_packed_backward_route", "")).strip()
    cudnn_route_marker = CUDNN_PACKED_BACKWARD_ROUTES.get(expected_cudnn_route) if expected_cudnn_route else None
    if expected_cudnn_route and cudnn_route_marker is None:
        audit.fail("cuDNN packed attention backward route", f"unknown manifest route: {expected_cudnn_route}")
    cudnn_route_evidence = False

    for log_name in ("bench_sm120_cudnn_attention.log", "bench_sm120_triton_attention.log"):
        path = optional_round / log_name
        text = path.read_text() if path.exists() else ""
        if log_name == "bench_sm120_cudnn_attention.log" and cudnn_route_marker is not None:
            cudnn_route_evidence = cudnn_route_marker in text
        for raw in text.splitlines():
            line = raw.strip()
            match = ATTENTION_RESULT_RE.match(line)
            if match and attention_shape_from_match(match) == ATTENTION_SELECTION_SHAPE:
                stack = match.group("stack") or "TK packed-QKV"
                timed_rows.add((stack, match.group("pass").lower()))
                continue
            unavailable = UNAVAILABLE_RESULT_RE.match(line)
            if not unavailable:
                continue
            shape = " ".join(unavailable.group("shape").split())
            stack = unavailable.group("stack").strip()
            reason = unavailable.group("reason").strip()
            if shape != ATTENTION_SELECTION_SHAPE or stack not in {"Triton", "TritonPacked"}:
                continue
            key = (stack, "backward")
            unavailable_rows[key] = reason
            if not reason:
                bad_unavailable[f"{stack}/backward"] = line

    cudnn_required = {
        ("cuDNN", "forward"),
        ("cuDNN", "backward"),
        ("cuDNNPacked", "forward"),
        ("cuDNNPacked", "backward"),
    }
    triton_forward_required = {
        ("Triton", "forward"),
        ("TritonPacked", "forward"),
    }
    triton_backward_required = {
        ("Triton", "backward"),
        ("TritonPacked", "backward"),
    }
    triton_backward_covered = {
        key
        for key in triton_backward_required
        if key in timed_rows or key in unavailable_rows
    }
    missing_cudnn = sorted(cudnn_required - timed_rows)
    missing_triton_forward = sorted(triton_forward_required - timed_rows)
    missing_triton_backward = sorted(triton_backward_required - triton_backward_covered)

    audit.require(
        "cuDNN attention exact coverage",
        not missing_cudnn,
        f"optional round records cuDNN separated and packed forward/backward attention rows; missing={missing_cudnn}",
    )
    if cudnn_route_marker is not None:
        audit.require(
            "cuDNN packed attention backward route",
            cudnn_route_evidence,
            "optional round records saved-forward cuDNNPacked backward route evidence",
        )
    audit.require(
        "Triton attention exact coverage",
        not missing_triton_forward and not missing_triton_backward and not bad_unavailable,
        (
            "optional round records Triton separated/packed forward rows and explicit backward timing or "
            f"unavailable rows; missing_forward={missing_triton_forward} "
            f"missing_backward={missing_triton_backward} bad_unavailable={len(bad_unavailable)}"
        ),
    )
    return {
        "cudnn_attention_rows": len(cudnn_required - set(missing_cudnn)),
        "expected_cudnn_attention_rows": len(cudnn_required),
        "triton_attention_timed_rows": sum(1 for key in triton_forward_required | triton_backward_required if key in timed_rows),
        "triton_attention_unavailable_rows": sum(1 for key in triton_backward_required if key in unavailable_rows),
        "expected_triton_attention_rows": len(triton_forward_required | triton_backward_required),
        "cudnn_packed_backward_route": expected_cudnn_route,
        "cudnn_packed_backward_route_evidence": cudnn_route_evidence,
        "missing_cudnn_attention_rows": missing_cudnn,
        "missing_triton_attention_forward_rows": missing_triton_forward,
        "missing_triton_attention_backward_rows": missing_triton_backward,
    }


def runtime_objective_shape(kernel: str, shape: str, stack: str) -> str:
    if stack.startswith("Torch") and kernel == "adamw_update":
        return shape.replace(" fp32-state", "")
    return shape


def expected_runtime_operator_keys() -> set[tuple[str, str]]:
    return {
        (kernel, shape)
        for kernel, shapes in RUNTIME_SELECTION_SHAPES.items()
        for shape in shapes
    }


def expected_libtorch_runtime_keys() -> set[tuple[str, str]]:
    return {
        (kernel, shape)
        for kernel, shapes in LIBTORCH_RUNTIME_SHAPE_REQUIREMENTS
        for shape in shapes
    }


def expected_libtorch_supplemental_runtime_keys() -> set[tuple[str, str]]:
    return {
        (kernel, shape)
        for kernel, shapes in LIBTORCH_RUNTIME_SUPPLEMENTAL_SHAPE_REQUIREMENTS
        for shape in shapes
    }


def manifest_libtorch_runtime_route(optional_round: Path) -> object:
    manifest_path = optional_round / "round-manifest.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    config = manifest.get("config", {}) if isinstance(manifest, dict) else {}
    if not isinstance(config, dict):
        return None
    return config.get("libtorch_runtime_route")


def libtorch_runtime_route_status(optional_round: Path, text: str) -> tuple[bool, object, list[str]]:
    expected_route = manifest_libtorch_runtime_route(optional_round)
    observed_routes = sorted(
        route
        for route, marker in LIBTORCH_RUNTIME_ROUTE_MARKERS.items()
        if marker in text
    )
    if expected_route in (None, ""):
        return bool(observed_routes), expected_route, observed_routes
    if not isinstance(expected_route, str) or expected_route not in LIBTORCH_RUNTIME_ROUTE_MARKERS:
        return False, expected_route, observed_routes
    return expected_route in observed_routes, expected_route, observed_routes


def audit_libtorch_runtime_shape_coverage(audit: Audit, optional_round: Path) -> dict[str, Any]:
    path = optional_round / "bench_sm120_libtorch_runtime.log"
    timed: set[tuple[str, str]] = set()
    parity: set[tuple[str, str]] = set()
    text = path.read_text(errors="replace") if path.exists() else ""
    for raw in text.splitlines():
        line = raw.strip()
        match = RUNTIME_RESULT_RE.match(line)
        if not match:
            parity_match = re.match(
                r"^LibTorch parity (?P<name>cuda_memset|cuda_copy_d2d|gelu_forward) "
                r"(?P<shape>[^:]+): PASS(?: .*)?$",
                line,
            )
            if parity_match:
                parity.add((parity_match.group("name"), parity_match.group("shape").strip()))
            continue
        if match.group("stack").strip() != "Torch C++":
            continue
        timed.add((match.group("name").strip(), match.group("shape").strip()))

    supplemental_timed: set[tuple[str, str]] = set()
    supplemental_parity: set[tuple[str, str]] = set()
    supplemental_texts: list[str] = []
    present_supplemental_paths: list[str] = []
    for log_path in LIBTORCH_SUPPLEMENTAL_RUNTIME_LOGS:
        if not log_path.exists():
            continue
        supplemental_texts.append(log_path.read_text(errors="replace"))
        present_supplemental_paths.append(str(log_path))
    for raw in "\n".join(supplemental_texts).splitlines():
        line = raw.strip()
        match = RUNTIME_RESULT_RE.match(line)
        if match:
            if match.group("stack").strip() == "Torch C++":
                supplemental_timed.add((match.group("name").strip(), match.group("shape").strip()))
            continue
        parity_match = re.match(
            r"^LibTorch parity (?P<name>cuda_memset|cuda_copy_d2d|gelu_forward) "
            r"(?P<shape>[^:]+): PASS(?: .*)?$",
            line,
        )
        if parity_match:
            supplemental_parity.add((parity_match.group("name"), parity_match.group("shape").strip()))

    expected = expected_libtorch_runtime_keys()
    supplemental_expected = expected_libtorch_supplemental_runtime_keys()
    missing = sorted(expected - timed)
    missing_parity = sorted(expected - parity)
    supplemental_missing = sorted(supplemental_expected - supplemental_timed)
    supplemental_missing_parity = sorted(supplemental_expected - supplemental_parity)
    raw_pointer_route, expected_route, observed_routes = libtorch_runtime_route_status(optional_round, text)
    audit.require(
        "LibTorch runtime exact memory coverage",
        not missing,
        f"optional round records Torch C++ timing rows for {len(timed & expected)}/{len(expected)} memory rows; missing={len(missing)}",
    )
    audit.require(
        "LibTorch runtime parity coverage",
        not missing_parity,
        f"optional round records Torch C++ parity rows for {len(parity & expected)}/{len(expected)} memory rows; missing={len(missing_parity)}",
    )
    audit.require(
        "LibTorch runtime raw-pointer route",
        raw_pointer_route,
        "optional round records cached from_blob raw-pointer route evidence for LibTorch C++ memory rows; "
        f"manifest_route={expected_route or 'legacy-any'} observed={','.join(observed_routes) or 'none'}",
    )
    audit.require(
        "LibTorch supplemental runtime coverage",
        not supplemental_missing,
        "current supplemental logs record Torch C++ timing rows for "
        f"{len(supplemental_timed & supplemental_expected)}/{len(supplemental_expected)} supplemental runtime rows; "
        f"missing={len(supplemental_missing)}",
    )
    audit.require(
        "LibTorch supplemental runtime parity coverage",
        not supplemental_missing_parity,
        "current supplemental logs record Torch C++ parity rows for "
        f"{len(supplemental_parity & supplemental_expected)}/{len(supplemental_expected)} supplemental runtime rows; "
        f"missing={len(supplemental_missing_parity)}",
    )
    return {
        "expected_libtorch_runtime_rows": len(expected),
        "libtorch_runtime_rows": len(timed & expected),
        "libtorch_runtime_parity_rows": len(parity & expected),
        "expected_libtorch_supplemental_runtime_rows": len(supplemental_expected),
        "libtorch_supplemental_runtime_rows": len(supplemental_timed & supplemental_expected),
        "libtorch_supplemental_runtime_parity_rows": len(supplemental_parity & supplemental_expected),
        "libtorch_runtime_raw_pointer_route": raw_pointer_route,
        "libtorch_runtime_manifest_route": expected_route,
        "libtorch_runtime_observed_routes": observed_routes,
        "libtorch_runtime_evidence_log": str(path),
        "libtorch_supplemental_runtime_evidence_logs": present_supplemental_paths,
        "missing_libtorch_runtime_rows": missing,
        "missing_libtorch_runtime_parity_rows": missing_parity,
        "missing_libtorch_supplemental_runtime_rows": supplemental_missing,
        "missing_libtorch_supplemental_runtime_parity_rows": supplemental_missing_parity,
    }


def audit_libtorch_trainer_link_probe(audit: Audit) -> dict[str, Any]:
    required_markers = (
        "LibTorch trainer link route: standalone executable without torch_python",
        "LibTorch trainer link compile: PASS",
        "LibTorch trainer link runtime: PASS zero/copy from_blob executable",
        "LibTorch trainer link probe: PASS",
    )
    passed_logs: list[str] = []
    missing_logs: list[str] = []
    bad_logs: list[str] = []
    for path in LIBTORCH_SUPPLEMENTAL_TRAINER_LINK_LOGS:
        if not path.exists():
            missing_logs.append(str(path))
            continue
        text = path.read_text(errors="replace")
        if all(marker in text for marker in required_markers):
            passed_logs.append(str(path))
        else:
            bad_logs.append(str(path))
    audit.require(
        "LibTorch trainer link probe",
        len(passed_logs) == len(LIBTORCH_SUPPLEMENTAL_TRAINER_LINK_LOGS) and not bad_logs,
        "current supplemental logs prove standalone LibTorch executable link/runtime route; "
        f"passed={len(passed_logs)}/{len(LIBTORCH_SUPPLEMENTAL_TRAINER_LINK_LOGS)} "
        f"missing={len(missing_logs)} bad={len(bad_logs)}",
    )
    return {
        "expected_libtorch_trainer_link_logs": len(LIBTORCH_SUPPLEMENTAL_TRAINER_LINK_LOGS),
        "libtorch_trainer_link_passed_logs": len(passed_logs),
        "libtorch_trainer_link_evidence_logs": passed_logs,
        "missing_libtorch_trainer_link_logs": missing_logs,
        "bad_libtorch_trainer_link_logs": bad_logs,
    }


def audit_optional_runtime_stack_coverage(audit: Audit, optional_round: Path) -> dict[str, Any]:
    timed: dict[str, set[tuple[str, str]]] = {"Torch": set(), "Triton": set()}
    unavailable: dict[str, dict[tuple[str, str], str]] = {"Torch": {}, "Triton": {}}
    bad_unavailable: dict[str, str] = {}
    log_names = (
        "bench_sm120_torch_classifier.log",
        "bench_sm120_triton_classifier.log",
        "bench_sm120_torch_runtime.log",
        "bench_sm120_triton_runtime.log",
    )

    for log_name in log_names:
        path = optional_round / log_name
        text = path.read_text() if path.exists() else ""
        for raw in text.splitlines():
            line = raw.strip()
            match = RUNTIME_RESULT_RE.match(line)
            if match:
                stack = match.group("stack").strip()
                if stack not in timed:
                    continue
                kernel = match.group("name").strip()
                shape = runtime_objective_shape(kernel, match.group("shape").strip(), stack)
                timed[stack].add((kernel, shape))
                continue
            unavailable_match = RUNTIME_UNAVAILABLE_RESULT_RE.match(line)
            if not unavailable_match:
                continue
            stack = unavailable_match.group("stack").strip()
            if stack not in unavailable:
                continue
            kernel = unavailable_match.group("name").strip()
            shape = runtime_objective_shape(kernel, unavailable_match.group("shape").strip(), stack)
            reason = unavailable_match.group("reason").strip()
            unavailable[stack][(kernel, shape)] = reason
            if not reason:
                bad_unavailable[f"{stack}/{kernel}/{shape}"] = line

    expected = expected_runtime_operator_keys()
    torch_covered = timed["Torch"] | set(unavailable["Torch"])
    torch_missing = sorted(expected - torch_covered)
    triton_covered = timed["Triton"] | set(unavailable["Triton"])
    triton_missing = sorted(expected - triton_covered)

    audit.require(
        "Torch runtime/classifier exact coverage",
        not torch_missing and not bad_unavailable,
        (
            f"optional round records Torch timing or unavailable rows for {len(torch_covered & expected)}/{len(expected)} "
            f"runtime/classifier objective rows; timed={len(timed['Torch'] & expected)} "
            f"unavailable={len(set(unavailable['Torch']) & expected)} missing={len(torch_missing)} "
            f"bad_unavailable={len(bad_unavailable)}"
        ),
    )
    audit.require(
        "Triton runtime/classifier exact coverage",
        not triton_missing and not bad_unavailable,
        (
            f"optional round records Triton timing or unavailable rows for {len(triton_covered & expected)}/{len(expected)} "
            f"runtime/classifier objective rows; timed={len(timed['Triton'] & expected)} "
            f"unavailable={len(set(unavailable['Triton']) & expected)} missing={len(triton_missing)} "
            f"bad_unavailable={len(bad_unavailable)}"
        ),
    )
    return {
        "expected_runtime_operator_rows": len(expected),
        "torch_runtime_operator_rows": len(timed["Torch"] & expected),
        "torch_runtime_operator_unavailable_rows": len(set(unavailable["Torch"]) & expected),
        "triton_runtime_operator_timed_rows": len(timed["Triton"] & expected),
        "triton_runtime_operator_unavailable_rows": len(set(unavailable["Triton"]) & expected),
        "missing_torch_runtime_operator_rows": torch_missing,
        "missing_triton_runtime_operator_rows": triton_missing,
    }


def audit_training(audit: Audit, native_round: Path) -> dict[str, Any]:
    scoreboard = native_round / "scoreboard-candidates.md"
    summary = training_summary(scoreboard)
    audit.require(
        "native training smoke",
        summary["steps"] is not None and summary["steps"] >= 10,
        f"{scoreboard} records {summary['steps']} training steps",
    )
    audit.require(
        "native average step time",
        summary["avg_ms"] is not None and summary["avg_ms"] > 0.0,
        f"{scoreboard} records avg_ms={summary['avg_ms']}",
    )
    return summary


def audit_docs(audit: Audit, paths: tuple[Path, ...]) -> dict[str, Any]:
    required_terms = (
        "current-sm120-selection",
        "write_sm120_current_selection.py",
        "audit_sm120_optimization_goal.py",
        "sm120_objective_contract.py",
        "RUN_CURRENT_SELECTION_AUDIT=1",
        "SM120_SELECTION_NATIVE_ROUND",
        "SM120_SELECTION_OPTIONAL_ROUND",
        "DRY_RUN=1",
    )
    expected_torch_rows = len(expected_trainer_selection_keys())
    expected_torch_row_text = f"`{expected_torch_rows}/{expected_torch_rows}` Torch objective benchmark rows"
    stale_gpu_terms = (
        "GPU " + "not available",
        "GPU " + "un" + "available",
        "nvidia-smi/NVML metadata query did not " + "return device metadata",
        "target runtime "
        + "availability is proven by explicit SM120 correctness and benchmark runs",
    )
    missing: dict[str, list[str]] = {}
    stale_torch_counts: dict[str, list[str]] = {}
    stale_gpu_wording: dict[str, list[str]] = {}
    for path in paths:
        text = path.read_text() if path.exists() else ""
        missing[str(path)] = [term for term in required_terms if term not in text]
        stale_gpu_wording[str(path)] = [term for term in stale_gpu_terms if term in text]
        if path.name == "optimise-goal.md":
            stale_terms = []
            if "`42/42` Torch objective benchmark rows" in text:
                stale_terms.append("`42/42` Torch objective benchmark rows")
            if expected_torch_row_text not in text:
                stale_terms.append(expected_torch_row_text)
            stale_torch_counts[str(path)] = stale_terms
    probe_source = Path("dev/probe_sm120_backend_stacks.py")
    if probe_source.exists():
        text = probe_source.read_text()
        stale_gpu_wording[str(probe_source)] = [term for term in stale_gpu_terms if term in text]
    audit.require(
        "selection docs",
        all(not terms for terms in missing.values()),
        f"docs mention current selection replay contract terms; missing={missing}",
    )
    audit.require(
        "objective Torch row count docs",
        all(not terms for terms in stale_torch_counts.values()),
        (
            "optimise-goal.md names the current Torch objective benchmark count; "
            f"expected={expected_torch_row_text} stale_or_missing={stale_torch_counts}"
        ),
    )
    audit.require(
        "GPU runtime wording docs",
        all(not terms for terms in stale_gpu_wording.values()),
        (
            "GPU runtime wording avoids stale availability claims and raw NVML failure text; "
            f"stale_terms={stale_gpu_wording}"
        ),
    )
    return {
        "missing_terms": missing,
        "stale_torch_counts": stale_torch_counts,
        "stale_gpu_wording": stale_gpu_wording,
    }


def build_report(
    selection_json: Path,
    selection_md: Path,
    native_round: Path,
    optional_round: Path,
    doc_paths: tuple[Path, ...],
    check_docs: bool,
) -> dict[str, Any]:
    audit = Audit()
    selection_summary = audit_selection(audit, selection_json, selection_md, native_round, optional_round)
    native_manifest_summary = audit_round_manifest(
        audit,
        native_round,
        require_python_stack_benchmarks=False,
    )
    optional_manifest_summary = audit_round_manifest(
        audit,
        optional_round,
        require_python_stack_benchmarks=True,
    )
    native_stack_summary = check_stack_matrix(audit, native_round)
    optional_stack_summary = check_stack_matrix(audit, optional_round)
    optional_summary = audit_optional_stack_rows(
        audit,
        optional_round,
        selection_summary["resolved_optional_keys"],
        selection_summary["resolved_optional_row_records"],
        selection_summary["project_fastest_rows"],
        selection_summary["project_torch_fastest_rows"],
        selection_summary["project_fastest_row_records"],
        selection_summary["project_torch_fastest_row_records"],
        selection_summary["project_fastest_keys"],
        selection_summary["project_torch_fastest_keys"],
        selection_summary["selection_optional_non_trainer_selected_rows"],
        selection_summary["selection_optional_non_trainer_promotion_rows"],
    )
    torch_benchmark_summary = audit_torch_objective_benchmark_coverage(audit, optional_round)
    cutedsl_summary = audit_cutedsl_gemm_feasibility(audit, optional_round)
    triton_matmul_summary = audit_triton_matmul_coverage(audit, optional_round)
    optional_attention_summary = audit_optional_attention_stack_coverage(audit, optional_round)
    optional_runtime_summary = audit_optional_runtime_stack_coverage(audit, optional_round)
    libtorch_runtime_summary = audit_libtorch_runtime_shape_coverage(audit, optional_round)
    libtorch_trainer_link_summary = audit_libtorch_trainer_link_probe(audit)
    selection_summary = {
        key: value
        for key, value in selection_summary.items()
        if key
        not in {
            "resolved_optional_keys",
            "resolved_optional_row_records",
            "project_fastest_keys",
            "project_torch_fastest_keys",
            "project_fastest_row_records",
            "project_torch_fastest_row_records",
            "selection_optional_non_trainer_selected_rows",
            "selection_optional_non_trainer_promotion_rows",
        }
    }
    training = audit_training(audit, native_round)
    docs = audit_docs(audit, doc_paths) if check_docs else {}
    return {
        "schema_version": 1,
        "selection_json": str(selection_json),
        "selection_markdown": str(selection_md),
        "native_round": str(native_round),
        "optional_round": str(optional_round),
        "checks": audit.checks,
        "selection_summary": selection_summary,
        "native_manifest_summary": native_manifest_summary,
        "optional_manifest_summary": optional_manifest_summary,
        "native_stack_summary": native_stack_summary,
        "optional_stack_summary": optional_stack_summary,
        "optional_summary": optional_summary,
        "torch_benchmark_summary": torch_benchmark_summary,
        "cutedsl_summary": cutedsl_summary,
        "triton_matmul_summary": triton_matmul_summary,
        "optional_attention_summary": optional_attention_summary,
        "optional_runtime_summary": optional_runtime_summary,
        "libtorch_runtime_summary": libtorch_runtime_summary,
        "libtorch_trainer_link_summary": libtorch_trainer_link_summary,
        "training_summary": training,
        "doc_summary": docs,
    }


def markdown_table(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    lines = [
        "| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(rows[0])) + " |",
        "| " + " | ".join("-" * widths[index] for index in range(len(rows[0]))) + " |",
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) + " |")
    return lines


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# SM120 Optimization Goal Audit",
        "",
        f"- native round: `{report['native_round']}`",
        f"- optional-stack round: `{report['optional_round']}`",
        f"- trainer rows: `{report['selection_summary']['native_trainer_rows']}`",
        f"- optional non-trainer selected rows: `{report['optional_summary']['optional_non_trainer_selected_rows']}`",
        f"- Torch selected optional rows: `{report['optional_summary']['torch_selected_rows']}`",
        f"- project-wide fastest rows: `{report['selection_summary']['project_fastest_rows']}`",
        f"- project-wide Torch fastest rows: `{report['selection_summary']['project_torch_fastest_rows']}`",
        f"- project-wide Torch disposition rows: `{report['selection_summary']['project_torch_fastest_disposition_rows']}`",
        f"- project-wide resolved divergences: `{report['selection_summary']['project_fastest_resolved_divergence_rows']}`",
        f"- project-wide extra benchmark rows: `{report['selection_summary']['project_fastest_extra_rows']}`",
        f"- native attention route rows: `{report['selection_summary']['native_attention_route_rows']}`",
        f"- optional attention route rows: `{report['selection_summary']['optional_attention_route_rows']}`",
        f"- Torch objective benchmark rows: `{report['torch_benchmark_summary']['torch_objective_benchmark_rows']}`",
        f"- CuTeDSL exact GEMM feasibility rows: `{report['cutedsl_summary']['cutedsl_gemm_rows']}`",
        f"- CuTeDSL target rejection rows: `{report['cutedsl_summary']['cutedsl_target_rejection_rows']}`",
        f"- Triton GEMM rows: `{report['triton_matmul_summary']['triton_matmul_rows']}`",
        f"- cuDNN attention rows: `{report['optional_attention_summary']['cudnn_attention_rows']}`",
        f"- Triton attention unavailable rows: `{report['optional_attention_summary']['triton_attention_unavailable_rows']}`",
        f"- Torch runtime/classifier rows: `{report['optional_runtime_summary']['torch_runtime_operator_rows']}`",
        f"- LibTorch C++ runtime memory rows: `{report['libtorch_runtime_summary']['libtorch_runtime_rows']}`",
        f"- LibTorch C++ runtime parity rows: `{report['libtorch_runtime_summary']['libtorch_runtime_parity_rows']}`",
        f"- LibTorch C++ supplemental runtime rows: `{report['libtorch_runtime_summary']['libtorch_supplemental_runtime_rows']}`",
        f"- LibTorch C++ supplemental runtime parity rows: `{report['libtorch_runtime_summary']['libtorch_supplemental_runtime_parity_rows']}`",
        f"- LibTorch trainer link probe logs: `{report['libtorch_trainer_link_summary']['libtorch_trainer_link_passed_logs']}`",
        f"- Triton runtime/classifier unavailable rows: `{report['optional_runtime_summary']['triton_runtime_operator_unavailable_rows']}`",
        f"- Python-stack benchmark logs: `{report['optional_summary']['python_stack_log_count']}`",
        f"- active promotion candidates: `{report['optional_summary']['active_promotion_candidates']}`",
        f"- native manifest binaries: `{report['native_manifest_summary']['binary_rows']}`",
        f"- optional manifest binaries: `{report['optional_manifest_summary']['binary_rows']}`",
        f"- native training steps: `{report['training_summary']['steps']}`",
        f"- native avg step time: `{report['training_summary']['avg_ms']:.3f} ms`",
        "",
        "## Trainer Stack Mix",
        "",
    ]
    stack_rows = [["Stack", "Rows"]]
    for stack, count in report["selection_summary"]["trainer_stack_counts"].items():
        stack_rows.append([stack, str(count)])
    lines.extend(markdown_table(stack_rows))
    lines.extend(["", "## Checks", ""])
    check_rows = [["Check", "Status", "Detail"]]
    for check in report["checks"]:
        check_rows.append([check["name"], check["status"], check["detail"]])
    lines.extend(markdown_table(check_rows))
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def write_stack_probe(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "objective_stacks": list(OBJECTIVE_STACKS),
                "objective_families": list(OBJECTIVE_FAMILIES),
                "stacks": [
                    {
                        "stack": stack,
                        "status": "available",
                        "candidate_use": "synthetic candidate use",
                        "evidence": [f"synthetic {stack} evidence"],
                        "next_action": "synthetic next action",
                    }
                    for stack in tuple(OBJECTIVE_STACKS) + tuple(ENVIRONMENT_STACKS)
                ],
                "family_matrix": [
                    {
                        "family": family,
                        "stack": stack,
                        "status": "candidate",
                        "reason": "synthetic reason",
                        "next_action": "synthetic next action",
                    }
                    for family in OBJECTIVE_FAMILIES
                    for stack in OBJECTIVE_STACKS
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_manifest(path: Path, round_dir: Path, *, run_python_stack_benchmarks: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config": {
                    "artifact_dir": str(round_dir),
                    "build_jobs": "4",
                    "cudnn_packed_backward_route": "saved-forward",
                    "device_arch": "SM120",
                    "keep_checkpoints": "0",
                    "libtorch_runtime_route": "cxx-api-raw-pointer",
                    "max_steps": "10",
                    "no_multi_gpu": "1",
                    "no_use_mpi": "1",
                    "run_benchmarks": "1",
                    "run_correctness": "1",
                    "run_libtorch_matmul_benchmarks": run_python_stack_benchmarks,
                    "libtorch_matmul_shapes": "qkv attproj fc fcproj lmhead",
                    "run_python_stack_benchmarks": run_python_stack_benchmarks,
                    "run_label": round_dir.name,
                    "run_stack_probe": "1",
                    "run_training": "1",
                },
                "git": {
                    "short_commit": "abcdef0",
                    "status_path": "git-status.txt",
                },
                "toolchain": {
                    "nvcc": {
                        "returncode": 0,
                        "stdout": "synthetic nvcc",
                    }
                },
                "binaries": [
                    {
                        "path": binary,
                        "exists": True,
                        "sha256": "0" * 64,
                    }
                    for binary in EXPECTED_MANIFEST_BINARIES
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_self_test_scoreboard(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    prefix_lines: list[str] | None = None,
) -> None:
    lines = list(prefix_lines or [])
    if lines and lines[-1] != "":
        lines.append("")
    lines.extend(
        [
            "## Selected Backend Rows",
            "",
            "| Suite | Kernel | Shape | Selected stack | Time (us) | Next stack | Next time (us) | Use scope | Decision status | Decision note |",
            "|---|---|---|---|---:|---|---:|---|---|---|",
        ]
    )
    for row in rows:
        next_stack = row.get("next_stack")
        next_time = row.get("next_time_us")
        decision_status = row.get("decision_status", "-")
        lines.append(
            f"| {row.get('suite')} | {row.get('kernel')} | `{row.get('shape')}` | "
            f"{row.get('selected_stack')} | {float(row.get('selected_time_us', 0.0)):.3f} | "
            f"{'-' if next_stack is None else next_stack} | "
            f"{'-' if next_time is None else f'{float(next_time):.3f}'} | "
            f"{row.get('use_scope', 'synthetic benchmark route')} | "
            f"{decision_status} | {row.get('decision_note', 'synthetic decision note')} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def self_test_selected_backend_payload(
    round_dir: Path,
    rows: list[dict[str, Any]],
    attention_route_rows: list[dict[str, Any]],
    *,
    include_torch_coverage: bool = False,
    include_libtorch_coverage: bool = False,
) -> dict[str, Any]:
    torch_coverage = {}
    torch_count = 0
    if include_torch_coverage:
        torch_coverage = {
            f"{suite}/{kernel}/{shape}": True
            for suite, kernel, shape in sorted(expected_trainer_selection_keys())
        }
        torch_count = len(torch_coverage)
    libtorch_coverage = {
        key: True
        for key in sorted(expected_libtorch_runtime_coverage_keys())
    } if include_libtorch_coverage else {}
    return {
        "schema_version": 1,
        "run_label": round_dir.name,
        "artifact_dir": str(round_dir),
        "git_commit": "abcdef0",
        "benchmark_row_count": max(len(rows), 1),
        "torch_benchmark_row_count": torch_count,
        "torch_benchmark_coverage": torch_coverage,
        "libtorch_runtime_shape_coverage": libtorch_coverage,
        "libtorch_runtime_parity_coverage": libtorch_coverage,
        "libtorch_runtime_raw_pointer_route": include_libtorch_coverage,
        "selection_policy": (
            "Fastest observed row per exact suite/kernel/shape. Rows without an existing "
            "trainer call path remain reference/operator prototypes until route-specific "
            "correctness and TinyStories smoke evidence proves promotion."
        ),
        "selected_backend_rows": rows,
        "attention_route_rows": attention_route_rows,
    }


def write_self_test_fixture(root: Path) -> tuple[Path, Path, Path, Path, tuple[Path, ...]]:
    native = root / "native"
    optional = root / "optional"
    native.mkdir(parents=True)
    optional.mkdir(parents=True)
    write_stack_probe(native / "backend-stacks.json")
    write_stack_probe(optional / "backend-stacks.json")
    for round_dir, python_stack_flag in ((native, "0"), (optional, "1")):
        write_manifest(
            round_dir / "round-manifest.json",
            round_dir,
            run_python_stack_benchmarks=python_stack_flag,
        )
        (round_dir / "test_synthetic.log").write_text("synthetic PASS\n")
        (round_dir / "bench_sm120_runtime.log").write_text("synthetic timing\n")
        (round_dir / "bench_sm120_torch_runtime.log").write_text("Torch synthetic timing\n")
        (round_dir / "bench_sm120_triton_runtime.log").write_text("Triton synthetic timing\n")
    (native / "scoreboard-candidates.md").write_text(
        "\n".join(
            [
                "## Training Smoke",
                "",
                "- training steps: `10`",
                "- total average iteration time: `2495.443 ms`",
                "",
            ]
        )
    )
    (native / "train_gpt2cu.log").write_text(
        "step 10/10 | loss 1.0 | norm 1.0 | lr 0.1 | 2495.443 ms | 0.0% bf16 MFU | 1 tok/s\n"
        "total average iteration time: 2495.443 ms\n"
    )
    def add_provenance(row: dict[str, Any], round_dir: Path, log_name: str) -> dict[str, Any]:
        manifest = read_json(round_dir / "round-manifest.json")
        manifest_config = manifest.get("config", {})
        if not isinstance(manifest_config, dict):
            manifest_config = {}
        enriched = dict(row)
        if "trainer_call_path_kind" not in enriched:
            if enriched.get("suite") == "runtime" and (
                enriched.get("kernel") == "cuda_copy_d2d"
                or (
                    enriched.get("kernel") == "cuda_memset"
                    and enriched.get("shape") == "logits_elems=3296722944"
                )
            ):
                enriched["trainer_call_path_kind"] = "profiler_runtime_benchmark_only"
            else:
                enriched["trainer_call_path_kind"] = (
                    "trainer_or_cxx_route"
                    if enriched.get("trainer_call_path_available")
                    else "operator_or_reference_prototype"
                )
        enriched.update(
            {
                "source_run_label": str(round_dir.name),
                "source_artifact_dir": str(round_dir),
                "source_git_commit": "abcdef0",
                "source_run_config": {
                    key: str(manifest_config[key])
                    for key in (
                        "device_arch",
                        "build_jobs",
                        "no_multi_gpu",
                        "no_use_mpi",
                        "run_correctness",
                        "run_benchmarks",
                        "run_python_stack_benchmarks",
                        "run_stack_probe",
                        "run_training",
                        "max_steps",
                        "keep_checkpoints",
                    )
                    if key in manifest_config
                },
                "timing_log_path": str(round_dir / log_name),
                "config_artifact_path": str(round_dir / "round-manifest.json"),
                "stack_probe_artifact_path": str(round_dir / "backend-stacks.json"),
                "correctness_log_paths": [str(round_dir / "test_synthetic.log")],
            }
        )
        return enriched

    native_specs = [
        (suite, kernel, shape, "bench_sm120_runtime.log")
        for suite, kernel, shape in sorted(expected_trainer_selection_keys())
    ]
    native_rows = [
        add_provenance(
            {
                "suite": suite,
                "kernel": kernel,
                "shape": shape,
                "selected_stack": "CUDA",
                "trainer_call_path_available": True,
                "selected_time_us": 528.0,
            },
            native,
            log_name,
        )
        for suite, kernel, shape, log_name in native_specs
    ]
    native_attention_route_rows = [
        {
            "shape": "B=64 T=1024 C=768 NH=12 HS=64",
            "stack": "TK packed-QKV",
            "route_scope": "packed trainer-layout route",
            "trainer_layout": True,
            "forward_us": 800.0,
            "backward_us": 2700.0,
            "total_us": 3500.0,
            "complete": True,
            "unavailable_reason": None,
        }
    ]
    (native / "selected-backends.json").write_text(
        json.dumps(self_test_selected_backend_payload(native, native_rows, native_attention_route_rows), indent=2)
        + "\n"
    )
    write_self_test_scoreboard(
        native / "scoreboard-candidates.md",
        native_rows,
        prefix_lines=[
            "## Training Smoke",
            "",
            "- training steps: `10`",
            "- total average iteration time: `2495.443 ms`",
            "",
        ],
    )
    optional_rows = [
        add_provenance(
            {
                "suite": "runtime",
                "kernel": "global_norm_squared",
                "shape": "params=124475904",
                "selected_stack": "CUDA",
                "trainer_call_path_available": True,
                "selected_time_us": 184.0,
            },
            optional,
            "bench_sm120_runtime.log",
        ),
        add_provenance(
            {
                "suite": "runtime",
                "kernel": "gelu_forward",
                "shape": "BT=65536 C=3072",
                "selected_stack": "Torch",
                "trainer_call_path_available": False,
                "selected_time_us": 520.0,
                "decision_status": "library_integration_not_justified",
                "decision_active": False,
                "decision_decision": "synthetic Torch decision",
                "decision_evidence": ["synthetic Torch decision evidence"],
                "candidate_class": "library integration",
                "promotion_gate": "synthetic promotion gate",
                "priority": "medium",
            },
            optional,
            "bench_sm120_torch_runtime.log",
        ),
        add_provenance(
            {
                "suite": "runtime",
                "kernel": "adamw_update_bf16_state",
                "shape": "params=124475904 no-master",
                "selected_stack": "Torch",
                "trainer_call_path_available": False,
                "selected_time_us": 1200.0,
                "project_extra_reason": "synthetic non-objective reference row",
            },
            optional,
            "bench_sm120_torch_runtime.log",
        ),
        add_provenance(
            {
                "suite": "runtime",
                "kernel": "gelu_forward",
                "shape": "BT=65536 C=3072",
                "selected_stack": "Triton",
                "trainer_call_path_available": False,
                "selected_time_us": 480.0,
                "decision_status": "library_integration_not_justified",
                "decision_active": False,
                "decision_decision": "synthetic Triton decision",
                "decision_evidence": ["synthetic Triton decision evidence"],
            },
            optional,
            "bench_sm120_triton_runtime.log",
        ),
    ]
    promotion_rows = [
        add_provenance(
            {
                "suite": "runtime",
                "kernel": "gelu_forward",
                "shape": "BT=65536 C=3072",
                "selected_stack": "Torch",
                "trainer_call_path_available": False,
                "selected_time_us": 520.0,
                "decision_status": "library_integration_not_justified",
                "decision_active": False,
                "decision_decision": "synthetic Torch decision",
                "decision_evidence": ["synthetic Torch decision evidence"],
                "candidate_class": "library integration",
                "promotion_gate": "synthetic promotion gate",
                "priority": "medium",
            },
            optional,
            "bench_sm120_torch_runtime.log",
        ),
        add_provenance(
            {
                "suite": "runtime",
                "kernel": "adamw_update_bf16_state",
                "shape": "params=124475904 no-master",
                "selected_stack": "Torch",
                "trainer_call_path_available": False,
                "selected_time_us": 1200.0,
                "decision_status": "contract_mismatch",
                "decision_active": False,
                "decision_decision": "synthetic contract mismatch decision",
                "decision_evidence": ["synthetic contract mismatch evidence"],
                "candidate_class": "contract mismatch",
                "promotion_gate": "synthetic contract gate",
                "priority": "low",
            },
            optional,
            "bench_sm120_torch_runtime.log",
        ),
        add_provenance(
            {
                "suite": "runtime",
                "kernel": "gelu_forward",
                "shape": "BT=65536 C=3072",
                "selected_stack": "Triton",
                "trainer_call_path_available": False,
                "selected_time_us": 480.0,
                "decision_status": "library_integration_not_justified",
                "decision_active": False,
                "decision_decision": "synthetic Triton decision",
                "decision_evidence": ["synthetic Triton decision evidence"],
                "candidate_class": "native/codegen integration",
                "promotion_gate": "synthetic promotion gate",
                "priority": "medium",
            },
            optional,
            "bench_sm120_triton_runtime.log",
        ),
    ]
    optional_attention_route_rows = [
        {
            "shape": "B=64 T=1024 C=768 NH=12 HS=64",
            "stack": "TorchPacked",
            "route_scope": "packed trainer-layout route",
            "trainer_layout": True,
            "forward_us": 1100.0,
            "backward_us": 4100.0,
            "total_us": 5200.0,
            "complete": True,
            "unavailable_reason": None,
        },
        {
            "shape": "B=64 T=1024 C=768 NH=12 HS=64",
            "stack": "TorchMaterializedPacked",
            "route_scope": "packed trainer-layout route",
            "trainer_layout": True,
            "forward_us": 1200.0,
            "backward_us": 4200.0,
            "total_us": 5400.0,
            "complete": True,
            "unavailable_reason": None,
        },
        {
            "shape": "B=64 T=1024 C=768 NH=12 HS=64",
            "stack": "cuDNNPacked",
            "route_scope": "packed trainer-layout route",
            "trainer_layout": True,
            "forward_us": 780.0,
            "backward_us": 3460.0,
            "total_us": 4240.0,
            "complete": True,
            "unavailable_reason": None,
        },
        {
            "shape": "B=64 T=1024 C=768 NH=12 HS=64",
            "stack": "TritonPacked",
            "route_scope": "packed trainer-layout route",
            "trainer_layout": True,
            "forward_us": 1500.0,
            "backward_us": None,
            "total_us": None,
            "complete": False,
            "unavailable_reason": "packed attention backward is not implemented",
        },
    ]
    (optional / "selected-backends.json").write_text(
        json.dumps(
            self_test_selected_backend_payload(
                optional,
                optional_rows,
                optional_attention_route_rows,
                include_torch_coverage=True,
                include_libtorch_coverage=True,
            ),
            indent=2,
        )
        + "\n"
    )
    (optional / "promotion-candidates.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "promotion_candidates": promotion_rows,
                "active_promotion_candidates": [],
            },
            indent=2,
        )
        + "\n"
    )
    for log_name in PYTHON_STACK_BENCHMARK_LOGS:
        (optional / log_name).write_text(f"{log_name} synthetic benchmark row\n")
    (optional / "bench_sm120_libtorch_matmul.log").write_text(
        "\n".join(
            [
                "LibTorch matmul device: synthetic; capability=sm_120",
                LIBTORCH_MATMUL_ROUTE,
                "",
                "qkv M=65536 N=2304 K=768 bias=1 gelu=0",
                "LibTorch matmul parity dW qkv: PASS max_abs=0.000000",
                "LibTorch matmul parity dW+accum qkv: PASS max_abs=0.000000",
                "dW       Torch C++   1020.000 us",
                "dW+accum Torch C++   1030.000 us",
                "",
                "attproj M=65536 N=768 K=768 bias=1 gelu=0",
                "LibTorch matmul parity dW attproj: PASS max_abs=0.000000",
                "LibTorch matmul parity dW+accum attproj: PASS max_abs=0.000000",
                "dW       Torch C++    340.000 us",
                "dW+accum Torch C++    345.000 us",
                "",
                "fc M=65536 N=3072 K=768 bias=1 gelu=1",
                "LibTorch matmul parity dW fc: PASS max_abs=0.000000",
                "LibTorch matmul parity dW+accum fc: PASS max_abs=0.000000",
                "dW       Torch C++   1490.000 us",
                "dW+accum Torch C++   1500.000 us",
                "",
                "fcproj M=65536 N=768 K=3072 bias=1 gelu=0",
                "LibTorch matmul parity dW fcproj: PASS max_abs=0.000000",
                "LibTorch matmul parity dW+accum fcproj: PASS max_abs=0.000000",
                "dW       Torch C++   1380.000 us",
                "dW+accum Torch C++   1410.000 us",
                "",
                "lmhead M=65536 N=50304 K=768 bias=0 gelu=0",
                "LibTorch matmul parity dW lmhead: PASS max_abs=0.000000",
                "LibTorch matmul parity dW+accum lmhead: PASS max_abs=0.000000",
                "dW       Torch C++  22000.000 us",
                "dW+accum Torch C++  22900.000 us",
                "",
            ]
        )
    )
    triton_matmul_lines = ["Triton matmul device: synthetic; capability=sm_120", ""]
    triton_op_labels = {
        "fwd": "fwd",
        "fwd+gelu": "fwd+GeLU",
        "dInp": "dInp",
        "dInp+dGeLU": "dInp+dGeLU",
        "dW": "dW",
        "dW+accum": "dW+accum",
    }
    shape_to_ops: dict[str, list[str]] = {}
    for kernel, shape_names in MATMUL_SHAPE_REQUIREMENTS:
        for shape_name in shape_names:
            shape_to_ops.setdefault(shape_name, []).append(kernel)
    for shape_name, shape in MATMUL_SELECTION_SHAPES.items():
        triton_matmul_lines.append(shape)
        for kernel in shape_to_ops[shape_name]:
            triton_matmul_lines.append(f"  {triton_op_labels[kernel]:<11} Triton   1000.000 us (diff=0.000000)")
        triton_matmul_lines.append("")
    (optional / "bench_sm120_triton_matmul.log").write_text("\n".join(triton_matmul_lines) + "\n")
    cutedsl_lines = [
        "CuTeDSL package: cutlass 4.5.1",
        "CuTeDSL CUDA available: True",
        "CuTeDSL device: NVIDIA GeForce RTX 5090; capability=sm_120",
        "Running Blackwell Grouped GEMM test with:",
        "1 groups",
    ]
    for shape in MATMUL_SELECTION_SHAPES.values():
        cutedsl_lines.append(
            f"{shape:<56} | CuTeDSL     | unavailable: local CuTeDSL BF16 grouped-GEMM path rejects sm_120a"
        )
    (optional / "bench_sm120_cutedsl_matmul.log").write_text("\n".join(cutedsl_lines) + "\n")
    (optional / "bench_sm120_cudnn_attention.log").write_text(
        "\n".join(
            [
                "cuDNN attention device: synthetic; capability=sm_120",
                "cuDNN Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 668.000 us (max_diff=0.003906)",
                "cuDNN Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2385.000 us",
                "cuDNNPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 781.000 us",
                CUDNN_PACKED_BACKWARD_ROUTE,
                "cuDNNPacked Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 3468.000 us",
            ]
        )
        + "\n"
    )
    (optional / "bench_sm120_triton_attention.log").write_text(
        "\n".join(
            [
                "Triton attention device: synthetic; capability=sm_120",
                "Triton Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 1400.000 us (diff=0.031250)",
                "TritonPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 1500.000 us (diff=0.031250)",
                "B=64 T=1024 C=768 NH=12 HS=64     | Triton       | unavailable: attention backward is not implemented in this Triton prototype",
                "B=64 T=1024 C=768 NH=12 HS=64     | TritonPacked | unavailable: packed attention backward is not implemented in this Triton prototype",
            ]
        )
        + "\n"
    )
    torch_classifier_lines = [
        "fused_classifier_loss          | B=64 T=1024 V=50257 P=50304 | Torch        | 18000.000 us",
        "fused_classifier               | B=64 T=1024 V=50257 P=50304 | Torch        | unavailable: CUDA OOM at full GPT-2 padded-logits shape",
    ]
    triton_classifier_lines = [
        "fused_classifier_loss          | B=64 T=1024 V=50257 P=50304 | Triton       |  8200.000 us",
        "fused_classifier               | B=64 T=1024 V=50257 P=50304 | Triton       | 22200.000 us",
    ]
    torch_runtime_lines = [
        "bias_add                       | BT=65536 OC=768              | Torch        |   138.000 us",
        "bias_add                       | BT=65536 OC=3072             | Torch        |   539.000 us",
        "gelu_forward                   | BT=65536 C=3072              | Torch        |   547.000 us",
        "gelu_backward_inplace          | BT=65536 C=3072              | Torch        | 27328.000 us",
        "bias_grad_reduce               | BT=65536 OC=768              | Torch        |   325.000 us",
        "bias_grad_reduce               | BT=65536 OC=2304             | Torch        |  1018.000 us",
        "bias_grad_reduce               | BT=65536 OC=3072             | Torch        |  1359.000 us",
        "global_norm_squared            | params=124475904             | Torch        |  2367.000 us",
        "adamw_update                   | params=124475904 no-master fp32-state | Torch        |  7453.000 us",
        "encoder_forward                | B=64 T=1024 C=768            | Torch        |   203.000 us",
        "cuda_memset                    | hidden_elems=50331648        | Torch        |    64.000 us",
        "cuda_memset                    | grad_elems=124475904         | Torch        |   157.000 us",
        "cuda_copy_d2d                  | hidden_elems=50331648        | Torch        |   134.000 us",
        "cuda_memset                    | logits_elems=3296722944      | Torch        |  4397.000 us",
        "cuda_copy_d2d                  | logits_elems=3296722944      | Torch        |  9310.000 us",
    ]
    triton_runtime_lines = [
        "bias_add                       | BT=65536 OC=768              | Triton       |   100.000 us",
        "bias_add                       | BT=65536 OC=3072             | Triton       |   600.000 us",
        "gelu_forward                   | BT=65536 C=3072              | Triton       |   700.000 us",
        "gelu_backward_inplace          | BT=65536 C=3072              | Triton       |   900.000 us",
        "bias_grad_reduce               | BT=65536 OC=768              | Triton       | unavailable: not implemented in this Triton runtime prototype",
        "bias_grad_reduce               | BT=65536 OC=2304             | Triton       | unavailable: not implemented in this Triton runtime prototype",
        "bias_grad_reduce               | BT=65536 OC=3072             | Triton       | unavailable: not implemented in this Triton runtime prototype",
        "adamw_update                   | params=124475904 no-master   | Triton       | unavailable: not implemented in this Triton runtime prototype",
        "global_norm_squared            | params=124475904             | Triton       | unavailable: not implemented in this Triton runtime prototype",
        "encoder_forward                | B=64 T=1024 C=768            | Triton       | unavailable: not implemented in this Triton runtime prototype",
        "cuda_memset                    | hidden_elems=50331648        | Triton       | unavailable: not implemented in this Triton runtime prototype",
        "cuda_memset                    | grad_elems=124475904         | Triton       | unavailable: not implemented in this Triton runtime prototype",
        "cuda_memset                    | logits_elems=3296722944      | Triton       | unavailable: not implemented in this Triton runtime prototype",
        "cuda_copy_d2d                  | hidden_elems=50331648        | Triton       | unavailable: not implemented in this Triton runtime prototype",
        "cuda_copy_d2d                  | logits_elems=3296722944      | Triton       | unavailable: not implemented in this Triton runtime prototype",
    ]
    (optional / "bench_sm120_torch_classifier.log").write_text("\n".join(torch_classifier_lines) + "\n")
    (optional / "bench_sm120_triton_classifier.log").write_text("\n".join(triton_classifier_lines) + "\n")
    (optional / "bench_sm120_torch_runtime.log").write_text("\n".join(torch_runtime_lines) + "\n")
    (optional / "bench_sm120_libtorch_runtime.log").write_text(
        "\n".join(
            [
                LIBTORCH_CXX_API_RAW_POINTER_ROUTE,
                "LibTorch parity cuda_memset hidden_elems=50331648: PASS",
                "LibTorch parity cuda_copy_d2d hidden_elems=50331648: PASS",
                "LibTorch parity cuda_memset grad_elems=124475904: PASS",
                "LibTorch parity cuda_memset logits_elems=3296722944: PASS",
                "LibTorch parity cuda_copy_d2d logits_elems=3296722944: PASS",
                "LibTorch parity gelu_forward BT=65536 C=3072: PASS max_abs=0.000000",
                "cuda_memset                    | hidden_elems=50331648        | Torch C++    |    65.000 us",
                "cuda_copy_d2d                  | hidden_elems=50331648        | Torch C++    |   135.000 us",
                "cuda_memset                    | grad_elems=124475904         | Torch C++    |   158.000 us",
                "cuda_memset                    | logits_elems=3296722944      | Torch C++    |  4400.000 us",
                "cuda_copy_d2d                  | logits_elems=3296722944      | Torch C++    |  9320.000 us",
                "gelu_forward                   | BT=65536 C=3072             | Torch C++    |   550.000 us",
            ]
        )
        + "\n"
    )
    (optional / "bench_sm120_triton_runtime.log").write_text("\n".join(triton_runtime_lines) + "\n")
    coverage_lines = [
        "# SM120 Round Metrics - self_test",
        "",
        "## Torch Objective Benchmark Coverage",
        "",
        "| Suite | Kernel | Shape | Covered |",
        "|---|---|---|---:|",
    ]
    for suite, kernel, shape in sorted(expected_trainer_selection_keys()):
        coverage_lines.append(f"| `{suite}` | `{kernel}` | `{shape}` | `True` |")
    coverage_lines.append("")
    write_self_test_scoreboard(
        optional / "scoreboard-candidates.md",
        optional_rows,
        prefix_lines=coverage_lines,
    )

    selection_json = root / "current-sm120-selection.json"
    selection_md = root / "current-sm120-selection.md"
    selection_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "native_selection_round": str(native),
                "optional_stack_round": str(optional),
                "native_training_evidence": {
                    "allow_benchmark_only_native": False,
                    "manifest_path": str(native / "round-manifest.json"),
                    "train_log_path": str(native / "train_gpt2cu.log"),
                    "max_steps": 10,
                },
                "active_promotion_candidate_count": 0,
                "native_inactive_selected_row_count": 0,
                "native_extra_selected_row_count": 0,
                "native_extra_selected_rows": [],
                "native_trainer_selection": native_rows,
                "project_fastest_selection": optional_rows,
                "project_torch_fastest_rows": [optional_rows[1], optional_rows[2]],
                "project_torch_fastest_disposition_rows": [
                    {
                        **optional_rows[1],
                        "torch_disposition": "resolved_away",
                        "torch_action": "synthetic promotion gate",
                    },
                    {
                        **optional_rows[2],
                        "torch_disposition": "extra_benchmark",
                        "torch_action": "synthetic non-objective reference row",
                    },
                ],
                "project_fastest_used_rows": [optional_rows[0]],
                "project_fastest_row_count": 4,
                "project_torch_fastest_row_count": 2,
                "project_torch_fastest_used_row_count": 0,
                "project_torch_fastest_resolved_divergence_row_count": 1,
                "project_torch_fastest_extra_row_count": 1,
                "project_torch_fastest_partitioned_row_count": 2,
                "project_torch_fastest_disposition_row_count": 2,
                "project_torch_fastest_actionable_row_count": 2,
                "project_torch_fastest_missing_disposition": [],
                "project_fastest_used_row_count": 1,
                "project_fastest_resolved_divergence_row_count": 2,
                "project_fastest_resolved_call_path_counts": {
                    "operator_or_reference_prototype": 2,
                },
                "project_fastest_resolved_status_counts": {
                    "library_integration_not_justified": 2,
                },
                "project_fastest_resolved_trainer_callable_row_count": 0,
                "project_fastest_resolved_trainer_callable_evidence_count": 0,
                "project_fastest_resolved_trainer_callable_missing_evidence": [],
                "project_fastest_resolved_decision_link_count": 2,
                "project_fastest_resolved_missing_decision_links": [],
                "project_fastest_resolved_non_trainer_row_count": 2,
                "project_fastest_resolved_non_trainer_actionable_count": 2,
                "project_fastest_resolved_non_trainer_missing_action": [],
                "project_fastest_resolved_divergence_rows": [
                    optional_rows[1],
                    optional_rows[3],
                ],
                "project_fastest_extra_row_count": 1,
                "project_fastest_extra_rows": [optional_rows[2]],
                "project_fastest_unresolved_objective_row_count": 0,
                "optional_non_trainer_selected_row_count": 3,
                "optional_non_trainer_promotion_row_count": 3,
                "native_attention_route_rows": native_attention_route_rows,
                "optional_attention_route_rows": optional_attention_route_rows,
                "resolved_optional_stack_decisions": [
                    promotion_rows[0],
                    promotion_rows[1],
                    promotion_rows[2],
                ],
            },
            indent=2,
        )
        + "\n"
    )
    selection_md.write_text(
        "# Current SM120 Backend Selection\n\n"
        "## Project-Wide Torch Fastest Rows\n\n"
        "## Project-Wide Torch Fastest Row Disposition\n\n"
        "## Project-Wide Fastest Rows\n\n"
        "## Project-Wide Fastest Rows Used By Trainer\n\n"
        "## Project-Wide Fastest Rows Resolved Away From Trainer\n\n"
        "## Fastest Rows Not Used By Trainer\n\n"
        "## Extra Project-Wide Benchmark Rows\n\n"
        "## Attention Route Totals\n\n"
        "## Resolved Optional-Stack Decisions\n"
    )
    optimise_goal = root / "optimise-goal.md"
    best_runs = root / "best_runs.md"
    changelog = root / "CHANGELOG.md"
    for path in (optimise_goal, best_runs, changelog):
        path.write_text(
            "current-sm120-selection via dev/write_sm120_current_selection.py and "
            "dev/audit_sm120_optimization_goal.py; defaults live in "
            "dev/sm120_objective_contract.py; RUN_CURRENT_SELECTION_AUDIT=1 "
            "uses SM120_SELECTION_NATIVE_ROUND and SM120_SELECTION_OPTIONAL_ROUND "
            "overrides when set; DRY_RUN=1 prints the replay commands. "
            "The optional-stack scoreboard records all `43/43` Torch objective "
            "benchmark rows from dev/sm120_objective_contract.py.\n"
        )
    return selection_json, selection_md, native, optional, (optimise_goal, best_runs, changelog)


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="sm120_goal_audit_") as tmp:
        fixture = write_self_test_fixture(Path(tmp))
        report = build_report(*fixture, check_docs=True)
        if report["optional_summary"]["torch_selected_rows"] != 2:
            raise RuntimeError("self-test did not preserve Torch selected row count")
        if report["selection_summary"]["project_torch_fastest_used_rows"] != 0:
            raise RuntimeError("self-test unexpectedly counted trainer-used Torch fastest rows")
        if report["selection_summary"]["project_torch_fastest_resolved_rows"] != 1:
            raise RuntimeError("self-test did not count resolved Torch fastest rows")
        if report["selection_summary"]["project_torch_fastest_extra_rows"] != 1:
            raise RuntimeError("self-test did not count extra Torch fastest rows")
        if report["selection_summary"]["project_torch_fastest_unpartitioned_rows"] != 0:
            raise RuntimeError("self-test left Torch fastest rows unpartitioned")
        if report["selection_summary"]["project_torch_fastest_disposition_rows"] != 2:
            raise RuntimeError("self-test did not preserve Torch fastest disposition rows")
        if report["selection_summary"]["project_torch_fastest_rows_missing_disposition"] != 0:
            raise RuntimeError("self-test found missing Torch fastest dispositions")
        if report["selection_summary"]["project_torch_fastest_rows_with_bad_disposition"] != 0:
            raise RuntimeError("self-test found incorrect Torch fastest dispositions")
        if report["selection_summary"]["project_torch_fastest_rows_missing_action"] != 0:
            raise RuntimeError("self-test found Torch fastest rows missing actions")
        if report["selection_summary"]["project_fastest_resolved_decision_link_rows"] != 2:
            raise RuntimeError("self-test did not link project resolved rows to decision rows")
        if report["selection_summary"]["project_fastest_resolved_missing_decision_links"] != 0:
            raise RuntimeError("self-test unexpectedly found missing project resolved decision links")
        if report["selection_summary"]["project_fastest_resolved_non_trainer_actionable_rows"] != 2:
            raise RuntimeError("self-test did not count actionable project resolved non-trainer rows")
        if report["selection_summary"]["project_fastest_resolved_non_trainer_missing_action_rows"] != 0:
            raise RuntimeError("self-test unexpectedly found missing project resolved action metadata")
        if report["optional_summary"]["optional_non_trainer_selected_rows"] != 3:
            raise RuntimeError("self-test did not preserve optional non-trainer row count")
        if not report["optional_attention_summary"]["cudnn_packed_backward_route_evidence"]:
            raise RuntimeError("self-test did not preserve cuDNN packed saved-forward route evidence")
        if not all(report["selection_summary"]["trainer_family_coverage"].values()):
            raise RuntimeError("self-test fixture did not cover every objective family")
        bad_selection, selection_md, native, optional, docs = fixture
        payload = read_json(bad_selection)
        payload["active_promotion_candidate_count"] = 1
        bad_selection.write_text(json.dumps(payload) + "\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "active promotion" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted active promotion candidates")

        fixture = write_self_test_fixture(Path(tmp) / "missing-cudnn-packed-route")
        bad_selection, selection_md, native, optional, docs = fixture
        cudnn_log = optional / "bench_sm120_cudnn_attention.log"
        cudnn_log.write_text(cudnn_log.read_text().replace(f"{CUDNN_PACKED_BACKWARD_ROUTE}\n", ""))
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "cuDNN packed attention backward route" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted missing cuDNN packed backward route evidence")

        fixture = write_self_test_fixture(Path(tmp) / "bad-count")
        bad_selection, selection_md, native, optional, docs = fixture
        payload = read_json(bad_selection)
        payload["active_promotion_candidate_count"] = 0
        payload["optional_non_trainer_selected_row_count"] = 2
        bad_selection.write_text(json.dumps(payload) + "\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "selection optional non-trainer count" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted mismatched optional non-trainer count")

        fixture = write_self_test_fixture(Path(tmp) / "bad-native-content")
        bad_selection, selection_md, native, optional, docs = fixture
        payload = read_json(bad_selection)
        payload["native_trainer_selection"][0]["selected_time_us"] = 999999.0
        bad_selection.write_text(json.dumps(payload) + "\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "native trainer source row content" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted native trainer row content drift")

        fixture = write_self_test_fixture(Path(tmp) / "bad-native-inactive-content")
        bad_selection, selection_md, native, optional, docs = fixture
        source_payload = read_json(native / "selected-backends.json")
        source_row = source_payload["selected_backend_rows"][0]
        source_row["selected_stack"] = "cuBLAS"
        source_row["selected_time_us"] = 1.0
        source_row["next_stack"] = "CUDA"
        source_row["next_time_us"] = 528.0
        source_row["decision_status"] = "synthetic_rejected_native"
        source_row["decision_active"] = False
        (native / "selected-backends.json").write_text(json.dumps(source_payload) + "\n")
        write_self_test_scoreboard(native / "scoreboard-candidates.md", source_payload["selected_backend_rows"])
        payload = read_json(bad_selection)
        effective_row = payload["native_trainer_selection"][0]
        effective_row["rejected_selected_stack"] = "cuBLAS"
        effective_row["rejected_selected_time_us"] = 1.0
        effective_row["selected_stack"] = "CUDA runtime"
        effective_row["selected_time_us"] = 528.0
        effective_row["next_stack"] = "CUDA"
        effective_row["next_time_us"] = 528.0
        effective_row["decision_status"] = "synthetic_rejected_native"
        effective_row["decision_active"] = False
        bad_selection.write_text(json.dumps(payload) + "\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "native trainer source row content" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted inactive native fallback content drift")

        fixture = write_self_test_fixture(Path(tmp) / "bad-native-extra-content")
        bad_selection, selection_md, native, optional, docs = fixture
        source_payload = read_json(native / "selected-backends.json")
        extra_row = dict(source_payload["selected_backend_rows"][0])
        extra_row.update(
            {
                "suite": "layernorm",
                "kernel": "forward",
                "shape": "N=65536 C=3072",
                "selected_stack": "CUDA",
                "selected_time_us": 543.0,
                "use_scope": "CUDA benchmark route",
            }
        )
        source_payload["selected_backend_rows"].append(extra_row)
        source_payload["benchmark_row_count"] = len(source_payload["selected_backend_rows"])
        (native / "selected-backends.json").write_text(json.dumps(source_payload) + "\n")
        write_self_test_scoreboard(native / "scoreboard-candidates.md", source_payload["selected_backend_rows"])
        payload = read_json(bad_selection)
        payload["native_extra_selected_row_count"] = 1
        payload["native_extra_selected_rows"] = [dict(extra_row)]
        payload["native_extra_selected_rows"][0]["selected_time_us"] = 999999.0
        bad_selection.write_text(json.dumps(payload) + "\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "native extra selected row content" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted native extra row content drift")

        fixture = write_self_test_fixture(Path(tmp) / "bad-source-scoreboard-content")
        bad_selection, selection_md, native, optional, docs = fixture
        scoreboard = optional / "scoreboard-candidates.md"
        scoreboard.write_text(scoreboard.read_text().replace("520.000", "999999.000", 1))
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "source selected-backend scoreboard content" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted selected-backends scoreboard drift")

        fixture = write_self_test_fixture(Path(tmp) / "bad-source-metadata")
        bad_selection, selection_md, native, optional, docs = fixture
        source_payload = read_json(optional / "selected-backends.json")
        source_payload["selection_policy"] = "synthetic stale policy"
        (optional / "selected-backends.json").write_text(json.dumps(source_payload) + "\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "source selected-backend metadata" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted selected-backends metadata drift")

        fixture = write_self_test_fixture(Path(tmp) / "bad-attention-route-content")
        bad_selection, selection_md, native, optional, docs = fixture
        payload = read_json(bad_selection)
        payload["optional_attention_route_rows"][0]["total_us"] = 999999.0
        bad_selection.write_text(json.dumps(payload) + "\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "attention route source content" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted attention route content drift")

        fixture = write_self_test_fixture(Path(tmp) / "missing-promotion-metadata")
        bad_selection, selection_md, native, optional, docs = fixture
        payload = read_json(bad_selection)
        payload["resolved_optional_stack_decisions"][1].pop("promotion_gate")
        bad_selection.write_text(json.dumps(payload) + "\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "resolved optional promotion metadata" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted missing resolved optional promotion metadata")

        fixture = write_self_test_fixture(Path(tmp) / "missing-decision-evidence")
        bad_selection, selection_md, native, optional, docs = fixture
        payload = read_json(bad_selection)
        payload["resolved_optional_stack_decisions"][1]["decision_evidence"] = []
        bad_selection.write_text(json.dumps(payload) + "\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "resolved optional decision evidence" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted missing resolved optional decision evidence")

        fixture = write_self_test_fixture(Path(tmp) / "missing-divergence-reason")
        bad_selection, selection_md, native, optional, docs = fixture
        payload = read_json(bad_selection)
        payload["project_fastest_selection"][1].pop("decision_decision")
        payload["project_fastest_resolved_divergence_rows"][0].pop("decision_decision")
        bad_selection.write_text(json.dumps(payload) + "\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "project-wide resolved divergence detail" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted missing resolved divergence reason")

        fixture = write_self_test_fixture(Path(tmp) / "bad-resolved-optional-promotion-content")
        bad_selection, selection_md, native, optional, docs = fixture
        payload = read_json(bad_selection)
        payload["resolved_optional_stack_decisions"][0]["promotion_gate"] = "wrong gate"
        bad_selection.write_text(json.dumps(payload) + "\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "selection resolved optional content" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted resolved optional promotion content drift")

        fixture = write_self_test_fixture(Path(tmp) / "bad-resolved-optional-selected-content")
        bad_selection, selection_md, native, optional, docs = fixture
        promotion_payload = read_json(optional / "promotion-candidates.json")
        promotion_payload["promotion_candidates"] = promotion_payload["promotion_candidates"][1:]
        (optional / "promotion-candidates.json").write_text(json.dumps(promotion_payload) + "\n")
        selected_payload = read_json(optional / "selected-backends.json")
        selected_row = selected_payload["selected_backend_rows"][1]
        selected_row["decision_status"] = "synthetic_selected_only"
        selected_row["decision_active"] = False
        selected_row["decision_decision"] = "synthetic selected-only decision"
        selected_row["decision_evidence"] = ["synthetic selected-only evidence"]
        selected_row["candidate_class"] = "library integration"
        selected_row["promotion_gate"] = "synthetic selected-only gate"
        selected_row["priority"] = "medium"
        (optional / "selected-backends.json").write_text(json.dumps(selected_payload) + "\n")
        payload = read_json(bad_selection)
        payload["resolved_optional_stack_decisions"][0] = dict(selected_row)
        payload["resolved_optional_stack_decisions"][0]["decision_decision"] = "wrong selected-only decision"
        bad_selection.write_text(json.dumps(payload) + "\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "selection resolved optional content" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted resolved optional selected-backend content drift")

        fixture = write_self_test_fixture(Path(tmp) / "bad-used-fastest")
        bad_selection, selection_md, native, optional, docs = fixture
        payload = read_json(bad_selection)
        payload["project_fastest_used_rows"] = []
        bad_selection.write_text(json.dumps(payload) + "\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "project-wide used row identity" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted missing project-wide fastest used rows")

        fixture = write_self_test_fixture(Path(tmp) / "bad-used-fastest-content")
        bad_selection, selection_md, native, optional, docs = fixture
        payload = read_json(bad_selection)
        payload["project_fastest_used_rows"][0]["selected_time_us"] = 999999.0
        bad_selection.write_text(json.dumps(payload) + "\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "project-wide partition row content" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted project-wide partition row content drift")

        fixture = write_self_test_fixture(Path(tmp) / "bad-resolved-fastest")
        bad_selection, selection_md, native, optional, docs = fixture
        payload = read_json(bad_selection)
        payload["project_fastest_resolved_divergence_rows"] = []
        bad_selection.write_text(json.dumps(payload) + "\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "project-wide resolved row identity" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted missing project-wide resolved divergence rows")

        fixture = write_self_test_fixture(Path(tmp) / "bad-extra-fastest")
        bad_selection, selection_md, native, optional, docs = fixture
        payload = read_json(bad_selection)
        payload["project_fastest_extra_rows"] = []
        bad_selection.write_text(json.dumps(payload) + "\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "project-wide extra row identity" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted missing project-wide extra rows")

        fixture = write_self_test_fixture(Path(tmp) / "bad-fastest-content")
        bad_selection, selection_md, native, optional, docs = fixture
        payload = read_json(bad_selection)
        payload["project_fastest_selection"][0]["selected_time_us"] = 999999.0
        payload["project_fastest_used_rows"][0]["selected_time_us"] = 999999.0
        bad_selection.write_text(json.dumps(payload) + "\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "selection project-wide fastest content" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted project-wide fastest row content drift")

        fixture = write_self_test_fixture(Path(tmp) / "bad-torch-fastest-content")
        bad_selection, selection_md, native, optional, docs = fixture
        payload = read_json(bad_selection)
        payload["project_torch_fastest_rows"][0]["use_scope"] = "wrong scope"
        bad_selection.write_text(json.dumps(payload) + "\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "selection project-wide Torch fastest content" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted Torch fastest row content drift")

        fixture = write_self_test_fixture(Path(tmp) / "bad-row-manifest-provenance")
        bad_selection, selection_md, native, optional, docs = fixture
        payload = read_json(bad_selection)
        payload["project_fastest_selection"][0]["source_git_commit"] = "badc0de"
        payload["project_fastest_selection"][0]["source_artifact_dir"] = str(optional)
        payload["project_fastest_selection"][0]["source_run_label"] = "wrong-run"
        payload["project_fastest_used_rows"][0]["source_git_commit"] = "badc0de"
        payload["project_fastest_used_rows"][0]["source_artifact_dir"] = str(optional)
        payload["project_fastest_used_rows"][0]["source_run_label"] = "wrong-run"
        bad_selection.write_text(json.dumps(payload) + "\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "selection row provenance" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted row provenance that mismatches its manifest")

        fixture = write_self_test_fixture(Path(tmp) / "missing-doc-contract")
        bad_selection, selection_md, native, optional, docs = fixture
        docs[0].write_text("current-sm120-selection via dev/write_sm120_current_selection.py\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "selection docs" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted missing current-selection replay docs")

        fixture = write_self_test_fixture(Path(tmp) / "stale-gpu-wording")
        bad_selection, selection_md, native, optional, docs = fixture
        docs[0].write_text(
            docs[0].read_text()
            + "GPU "
            + "un"
            + "available because nvidia-smi/NVML metadata query did not "
            + "return device metadata\n"
        )
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "GPU runtime wording docs" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted stale GPU runtime wording")

        fixture = write_self_test_fixture(Path(tmp) / "missing-log")
        bad_selection, selection_md, native, optional, docs = fixture
        (optional / "bench_sm120_triton_matmul.log").unlink()
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "Python stack log bench_sm120_triton_matmul.log" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted a missing non-Torch Python stack log")

        fixture = write_self_test_fixture(Path(tmp) / "missing-libtorch-matmul-log")
        bad_selection, selection_md, native, optional, docs = fixture
        (optional / "bench_sm120_libtorch_matmul.log").unlink()
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "LibTorch matmul log" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted a missing LibTorch matmul log")

        fixture = write_self_test_fixture(Path(tmp) / "missing-libtorch-matmul-row")
        bad_selection, selection_md, native, optional, docs = fixture
        libtorch_log = optional / "bench_sm120_libtorch_matmul.log"
        libtorch_log.write_text(
            "\n".join(
                line
                for line in libtorch_log.read_text().splitlines()
                if not line.startswith("dW+accum")
            )
            + "\n"
        )
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "LibTorch matmul dWeight coverage" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted missing LibTorch matmul dWeight rows")

        fixture = write_self_test_fixture(Path(tmp) / "missing-libtorch-runtime-row")
        bad_selection, selection_md, native, optional, docs = fixture
        libtorch_log = optional / "bench_sm120_libtorch_runtime.log"
        libtorch_log.write_text(
            "\n".join(
                line
                for line in libtorch_log.read_text().splitlines()
                if "logits_elems=3296722944" not in line
            )
            + "\n"
        )
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "LibTorch runtime exact memory coverage" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted missing LibTorch runtime memory evidence")

        fixture = write_self_test_fixture(Path(tmp) / "missing-libtorch-runtime-parity")
        bad_selection, selection_md, native, optional, docs = fixture
        libtorch_log = optional / "bench_sm120_libtorch_runtime.log"
        libtorch_log.write_text(
            "\n".join(
                line
                for line in libtorch_log.read_text().splitlines()
                if "LibTorch parity cuda_copy_d2d logits_elems=3296722944" not in line
            )
            + "\n"
        )
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "LibTorch runtime parity coverage" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted missing LibTorch runtime parity evidence")

        fixture = write_self_test_fixture(Path(tmp) / "raw-pointer-libtorch-runtime-route")
        bad_selection, selection_md, native, optional, docs = fixture
        manifest_path = optional / "round-manifest.json"
        manifest = read_json(manifest_path)
        manifest["config"]["libtorch_runtime_route"] = "raw-pointer"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        libtorch_log = optional / "bench_sm120_libtorch_runtime.log"
        libtorch_log.write_text(
            libtorch_log.read_text().replace(
                LIBTORCH_CXX_API_RAW_POINTER_ROUTE,
                LIBTORCH_RAW_POINTER_ROUTE,
            )
        )
        build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)

        fixture = write_self_test_fixture(Path(tmp) / "mismatched-libtorch-runtime-route")
        bad_selection, selection_md, native, optional, docs = fixture
        libtorch_log = optional / "bench_sm120_libtorch_runtime.log"
        libtorch_log.write_text(
            libtorch_log.read_text().replace(
                LIBTORCH_CXX_API_RAW_POINTER_ROUTE,
                LIBTORCH_RAW_POINTER_ROUTE,
            )
        )
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "LibTorch runtime raw-pointer route" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted mismatched LibTorch raw-pointer route evidence")

        fixture = write_self_test_fixture(Path(tmp) / "unknown-libtorch-runtime-route")
        bad_selection, selection_md, native, optional, docs = fixture
        manifest_path = optional / "round-manifest.json"
        manifest = read_json(manifest_path)
        manifest["config"]["libtorch_runtime_route"] = "unexpected-route"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "LibTorch runtime raw-pointer route" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted unknown LibTorch route evidence")

        fixture = write_self_test_fixture(Path(tmp) / "missing-libtorch-runtime-route")
        bad_selection, selection_md, native, optional, docs = fixture
        libtorch_log = optional / "bench_sm120_libtorch_runtime.log"
        libtorch_log.write_text(
            "\n".join(
                line
                for line in libtorch_log.read_text().splitlines()
                if line != LIBTORCH_CXX_API_RAW_POINTER_ROUTE
            )
            + "\n"
        )
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "LibTorch runtime raw-pointer route" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted missing LibTorch raw-pointer route evidence")

        fixture = write_self_test_fixture(Path(tmp) / "missing-cutedsl-shape")
        bad_selection, selection_md, native, optional, docs = fixture
        cutedsl_log = optional / "bench_sm120_cutedsl_matmul.log"
        cutedsl_log.write_text(
            "\n".join(
                line
                for line in cutedsl_log.read_text().splitlines()
                if not line.startswith("lmhead M=65536")
            )
            + "\n"
        )
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "CuTeDSL exact GEMM feasibility rows" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted missing CuTeDSL exact-shape feasibility evidence")

        fixture = write_self_test_fixture(Path(tmp) / "missing-triton-matmul")
        bad_selection, selection_md, native, optional, docs = fixture
        triton_matmul_log = optional / "bench_sm120_triton_matmul.log"
        triton_matmul_log.write_text(
            "\n".join(
                line
                for line in triton_matmul_log.read_text().splitlines()
                if "dInp+dGeLU" not in line
            )
            + "\n"
        )
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "Triton GEMM exact coverage" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted missing Triton GEMM exact-row evidence")

        fixture = write_self_test_fixture(Path(tmp) / "missing-triton-attn-backward")
        bad_selection, selection_md, native, optional, docs = fixture
        triton_log = optional / "bench_sm120_triton_attention.log"
        triton_log.write_text(
            "\n".join(
                line
                for line in triton_log.read_text().splitlines()
                if "unavailable: attention backward" not in line
            )
            + "\n"
        )
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "Triton attention exact coverage" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted missing Triton attention backward evidence")

        fixture = write_self_test_fixture(Path(tmp) / "missing-triton-runtime")
        bad_selection, selection_md, native, optional, docs = fixture
        triton_runtime_log = optional / "bench_sm120_triton_runtime.log"
        triton_runtime_log.write_text(
            "\n".join(
                line
                for line in triton_runtime_log.read_text().splitlines()
                if "adamw_update" not in line
            )
            + "\n"
        )
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "Triton runtime/classifier exact coverage" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted missing Triton runtime unavailable evidence")

        fixture = write_self_test_fixture(Path(tmp) / "missing")
        bad_selection, selection_md, native, optional, docs = fixture
        payload = read_json(bad_selection)
        payload["active_promotion_candidate_count"] = 0
        payload["resolved_optional_stack_decisions"] = [
            row
            for row in payload["resolved_optional_stack_decisions"]
            if row.get("kernel") != "adamw_update_bf16_state"
        ]
        bad_selection.write_text(json.dumps(payload) + "\n")
        try:
            build_report(bad_selection, selection_md, native, optional, docs, check_docs=True)
        except ValueError as exc:
            if "optional consolidated decisions" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test accepted a missing consolidated optional decision")
    print("SM120 optimization goal audit self-test OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run synthetic audit checks")
    parser.add_argument("--selection-json", type=Path, default=DEFAULT_SELECTION_JSON)
    parser.add_argument("--selection-md", type=Path, default=DEFAULT_SELECTION_MD)
    parser.add_argument("--native-round", type=Path, default=DEFAULT_NATIVE_ROUND)
    parser.add_argument("--optional-round", type=Path, default=DEFAULT_OPTIONAL_ROUND)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument(
        "--doc-path",
        type=Path,
        action="append",
        default=[Path("optimise-goal.md"), Path("best_runs.md"), Path("CHANGELOG.md")],
        help="Doc path that must mention the current selection replay contract",
    )
    parser.add_argument("--skip-docs", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.json_out is None or args.markdown_out is None:
        parser.error("--json-out and --markdown-out are required unless --self-test is set")

    report = build_report(
        args.selection_json,
        args.selection_md,
        args.native_round,
        args.optional_round,
        tuple(args.doc_path),
        not args.skip_docs,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(args.markdown_out, report)
    print(
        "SM120 optimization goal audit OK: "
        f"checks={len(report['checks'])}; "
        f"trainer_rows={report['selection_summary']['native_trainer_rows']}; "
        f"optional_non_trainer={report['optional_summary']['optional_non_trainer_selected_rows']}; "
        f"optional_torch_selected={report['optional_summary']['torch_selected_rows']}; "
        f"project_torch_fastest={report['selection_summary']['project_torch_fastest_rows']}; "
        f"torch_objective_rows={report['torch_benchmark_summary']['torch_objective_benchmark_rows']}; "
        f"cutedsl_gemm_rows={report['cutedsl_summary']['cutedsl_gemm_rows']}; "
        f"triton_matmul_rows={report['triton_matmul_summary']['triton_matmul_rows']}; "
        f"cudnn_attention_rows={report['optional_attention_summary']['cudnn_attention_rows']}; "
        f"triton_attention_unavailable={report['optional_attention_summary']['triton_attention_unavailable_rows']}; "
        f"torch_runtime_rows={report['optional_runtime_summary']['torch_runtime_operator_rows']}; "
        f"libtorch_runtime_rows={report['libtorch_runtime_summary']['libtorch_runtime_rows']}; "
        f"libtorch_parity_rows={report['libtorch_runtime_summary']['libtorch_runtime_parity_rows']}; "
        f"libtorch_supplemental_runtime_rows={report['libtorch_runtime_summary']['libtorch_supplemental_runtime_rows']}; "
        f"libtorch_supplemental_parity_rows={report['libtorch_runtime_summary']['libtorch_supplemental_runtime_parity_rows']}; "
        f"libtorch_trainer_link_logs={report['libtorch_trainer_link_summary']['libtorch_trainer_link_passed_logs']}; "
        f"triton_runtime_unavailable={report['optional_runtime_summary']['triton_runtime_operator_unavailable_rows']}; "
        f"active_promotions={report['optional_summary']['active_promotion_candidates']}; "
        f"json={args.json_out}; markdown={args.markdown_out}"
    )


if __name__ == "__main__":
    main()
