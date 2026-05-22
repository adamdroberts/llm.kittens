#!/usr/bin/env python3
"""Validate and summarize SM120 optimization-round artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from sm120_objective_contract import (
    ATTENTION_SELECTION_SHAPE,
    CORRECTNESS_TESTS,
    ENVIRONMENT_STACKS,
    EXPECTED_MANIFEST_BINARIES,
    EXPECTED_RUNTIME_KERNELS,
    LIBTORCH_TRAINER_LINK_LOG,
    LIBTORCH_RUNTIME_SHAPE_REQUIREMENTS,
    LIBTORCH_RUNTIME_SUPPLEMENTAL_SHAPE_REQUIREMENTS,
    LAYERNORM_SELECTION_SHAPE,
    MATMUL_SELECTION_SHAPES,
    MATMUL_SHAPE_REQUIREMENTS,
    OBJECTIVE_FAMILIES,
    OBJECTIVE_STACKS,
    PYTHON_STACK_BENCHMARK_LOGS,
    RUNTIME_SELECTION_SHAPES,
    RUNTIME_SHAPE_REQUIREMENTS,
)


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

META_RE = re.compile(r"^-\s+(?P<key>[^:]+):\s+`(?P<value>[^`]*)`$")
MATMUL_SHAPE_RE = re.compile(
    r"^(?P<name>\S+)\s+M=(?P<m>\d+)\s+N=(?P<n>\d+)\s+K=(?P<k>\d+)\s+"
    r"bias=(?P<bias>[01])\s+gelu=(?P<gelu>[01])$"
)
MATMUL_FWD_RE = re.compile(
    rf"^fwd\s+TK\s+(?P<tk>{FLOAT_RE})\s+us\s+\|\s+"
    rf"cuBLASLt\s+(?P<cublaslt>{FLOAT_RE})\s+us"
    rf"(?:\s+\|\s+cuBLAS\s+(?P<cublas>{FLOAT_RE})\s+us)?\s+\|"
)
MATMUL_FWD_GELU_RE = re.compile(
    rf"^fwd\+GeLU\s+TK fused\s+(?P<tk_fused>{FLOAT_RE})\s+us\s+\|\s+"
    rf"TK explicit\s+(?P<tk_explicit>{FLOAT_RE})\s+us\s+\|\s+"
    rf"cuBLASLt\s+(?P<cublaslt>{FLOAT_RE})\s+us"
    rf"(?:\s+\|\s+cuBLAS explicit\s+(?P<cublas>{FLOAT_RE})\s+us)?\s+\|"
)
MATMUL_DINP_RE = re.compile(
    rf"^dInp\s+TK\s+(?P<tk>{FLOAT_RE})\s+us\s+\|\s+"
    rf"cuBLASLt\s+(?P<cublaslt>{FLOAT_RE})\s+us"
    rf"(?:\s+\|\s+cuBLAS\s+(?P<cublas>{FLOAT_RE})\s+us)?\s+\|"
)
MATMUL_DINP_DGELU_RE = re.compile(
    rf"^dInp\+dGeLU\s+TK\s+(?P<tk>{FLOAT_RE})\s+us\s+\|\s+"
    rf"cuBLASLt fused\s+(?P<cublas_fused>{FLOAT_RE})\s+us\s+\|\s+"
    rf"cuBLASLt explicit\s+(?P<cublas_explicit>{FLOAT_RE})\s+us"
    rf"(?:\s+\|\s+cuBLAS explicit\s+(?P<cublas>{FLOAT_RE})\s+us)?\s+\|"
)
MATMUL_DW_RE = re.compile(
    rf"^dW\s+TK\s+(?P<tk>{FLOAT_RE})\s+us\s+\|\s+"
    rf"cuBLASLt\s+(?P<cublaslt>{FLOAT_RE})\s+us"
    rf"(?:\s+\|\s+cuBLAS\s+(?P<cublas>{FLOAT_RE})\s+us)?\s+\|"
)
MATMUL_DW_ACCUM_RE = re.compile(
    rf"^dW\+accum\s+TK\s+(?P<tk>{FLOAT_RE})\s+us\s+\|\s+"
    rf"cuBLASLt\s+(?P<cublaslt>{FLOAT_RE})\s+us"
    rf"(?:\s+\|\s+cuBLAS\s+(?P<cublas>{FLOAT_RE})\s+us)?\s+\|"
)
PYTHON_MATMUL_RE = re.compile(
    rf"^(?P<op>fwd|fwd\+GeLU|dInp|dInp\+dGeLU|dW|dW\+accum)\s+"
    rf"(?P<stack>Torch C\+\+|Torch|Triton)\s+(?P<us>{FLOAT_RE})\s+us(?:\s+\(.*\))?$"
)
ATTENTION_RE = re.compile(
    rf"^(?:(?P<stack>Torch|TorchPacked|TorchMaterializedPacked|cuDNN|cuDNNPacked|Triton|TritonPacked)\s+)?Attention (?P<pass>Forward|Backward) "
    rf"\(B=(?P<b>\d+), T=(?P<t>\d+), C=(?P<c>\d+), NH=(?P<nh>\d+), HS=(?P<hs>\d+)\): "
    rf"(?P<us>{FLOAT_RE}) us(?:\s+\(.*\))?$"
)
LAYERNORM_RE = re.compile(
    rf"^(?:(?P<stack>Triton|Torch)\s+)?LayerNorm "
    rf"(?P<pass>Forward|ForwardNative|ForwardWithStats|FusedResidualForward|"
    rf"FusedResidualForwardNative|FusedResidualForwardWithStats|Backward|"
    rf"BackwardNative|BackwardDInput|BackwardDInputNative|"
    rf"BackwardDInputNativePlusGrads|BackwardAtomicFP32) "
    rf"\(N=(?P<n>\d+), C=(?P<c>\d+)\): (?P<us>{FLOAT_RE}) us"
    rf"(?:\s+\(.*\))?$"
)
RUNTIME_RE = re.compile(
    rf"^(?P<name>[A-Za-z0-9_]+)\s+\|\s+(?P<shape>[^|]+?)\s+\|\s+"
    rf"(?P<stack>[^|]+?)\s+\|\s+(?P<us>{FLOAT_RE}) us$"
)
RUNTIME_UNAVAILABLE_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_]+)\s+\|\s+(?P<shape>[^|]+?)\s+\|\s+"
    r"(?P<stack>[^|]+?)\s+\|\s+unavailable:\s+(?P<reason>.+)$"
)
LIBTORCH_PARITY_RE = re.compile(
    r"^LibTorch parity (?P<name>cuda_memset|cuda_copy_d2d|gelu_forward) "
    r"(?P<shape>[^:]+): PASS(?: .*)?$"
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
CUDNN_PACKED_BACKWARD_ROUTE = "cuDNNPacked Attention Backward route: saved-forward"
CUDNN_PACKED_BACKWARD_ROUTES = {
    "saved-forward": CUDNN_PACKED_BACKWARD_ROUTE,
}
UNAVAILABLE_RE = re.compile(
    r"^(?P<shape>[^|]+?)\s+\|\s+(?P<stack>[^|]+?)\s+\|\s+unavailable:\s+(?P<reason>.+)$"
)
SETTING_RE = re.compile(r"^\|\s*(?P<key>[^|]+?)\s*\|\s*(?P<value>[^|]+?)\s*\|$")
VAL_LOSS_RE = re.compile(rf"^val loss (?P<loss>{FLOAT_RE})$")
STEP_RE = re.compile(
    rf"^step\s+(?P<step>\d+)/(?P<total>\d+)\s+\|\s+"
    rf"loss\s+(?P<loss>{FLOAT_RE})\s+[^|]*\|\s+"
    rf"norm\s+(?P<norm>{FLOAT_RE})\s+[^|]*\|\s+"
    rf"lr\s+(?P<lr>{FLOAT_RE})\s+\|\s+"
    rf"(?P<ms>{FLOAT_RE})\s+ms\s+\|\s+"
    rf"(?P<mfu>{FLOAT_RE})% bf16 MFU\s+\|\s+"
    r"(?P<tok_s>\d+) tok/s$"
)
AVG_RE = re.compile(rf"^total average iteration time: (?P<ms>{FLOAT_RE}) ms$")


STACK_STATUSES = {"available", "missing", "blocked", "unknown", "not_applicable"}
FAMILY_STACK_STATUSES = {"baseline", "candidate", "fallback", "missing", "blocked", "unknown", "not_applicable"}

MATMUL_SHAPE_LABELS = {
    "qkv": "qkv",
    "attproj": "attention projection",
    "fc": "MLP up",
    "fcproj": "MLP projection",
    "lmhead": "LM-head",
}

MATMUL_PROVIDER_LABELS = {
    "tk": "ThunderKittens",
    "cublaslt": "cuBLASLt",
    "cublas": "cuBLAS",
}
FAMILY_METRIC_KEYS = {
    "gemm_forward": ("matmul", "fwd"),
    "gemm_forward_fused_gelu": ("matmul", "fwd+gelu"),
    "gemm_backward_dinput": ("matmul", "dInp"),
    "gemm_backward_dinput_fused_dgelu": ("matmul", "dInp+dGeLU"),
    "gemm_backward_dweight": ("matmul", "dW"),
    "gemm_backward_dweight_accum": ("matmul", "dW+accum"),
    "bias_add": ("runtime", "bias_add"),
    "bias_gradient_reduce": ("runtime", "bias_grad_reduce"),
    "gelu_forward": ("runtime", "gelu_forward"),
    "gelu_backward": ("runtime", "gelu_backward_inplace"),
    "attention_forward": ("attention", "forward"),
    "attention_backward": ("attention", "backward"),
    "layernorm_forward": ("layernorm", "forward"),
    "layernorm_fused_residual_forward": ("layernorm", "fused_residual_forward"),
    "layernorm_backward": ("layernorm", "backward"),
    "classifier_softmax_cross_entropy_dlogits": ("runtime", "fused_classifier"),
    "adamw": ("runtime", "adamw_update"),
    "global_norm": ("runtime", "global_norm_squared"),
    "encoder_forward": ("runtime", "encoder_forward"),
    "cuda_memset": ("runtime", "cuda_memset"),
    "cuda_copy_d2d": ("runtime", "cuda_copy_d2d"),
}


@dataclass(frozen=True)
class BenchmarkMetric:
    suite: str
    name: str
    shape: str
    stack: str
    time_us: float


@dataclass(frozen=True)
class UnavailableMetric:
    suite: str
    name: str
    shape: str
    stack: str
    reason: str


@dataclass(frozen=True)
class BackendSelection:
    suite: str
    name: str
    shape: str
    selected_stack: str
    selected_time_us: float
    next_stack: str
    next_time_us: float | None
    use_scope: str
    decision_note: str


@dataclass(frozen=True)
class PromotionCandidate:
    selection: BackendSelection
    speedup_vs_next_pct: float | None
    priority: str
    candidate_class: str
    promotion_gate: str


@dataclass(frozen=True)
class AttentionRouteSummary:
    shape: str
    stack: str
    route_scope: str
    trainer_layout: bool
    forward_us: float | None
    backward_us: float | None
    total_us: float | None
    complete: bool
    unavailable_reason: str | None


@dataclass(frozen=True)
class PromotionDecision:
    suite: str
    kernel: str
    shape: str
    selected_stack: str
    status: str
    active: bool
    decision: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class TrainStep:
    step: int
    total: int
    loss: float
    norm: float
    lr: float
    ms: float
    mfu: float
    tok_s: int


@dataclass
class RoundMetrics:
    metadata: dict[str, str]
    manifest: dict[str, object]
    backend_stacks: list[dict[str, object]]
    backend_family_matrix: list[dict[str, object]]
    benchmarks: list[BenchmarkMetric]
    coverage: dict[str, bool]
    matmul_shape_coverage: dict[str, bool]
    runtime_shape_coverage: dict[str, bool]
    libtorch_runtime_shape_coverage: dict[str, bool]
    libtorch_runtime_parity_coverage: dict[str, bool]
    libtorch_runtime_supplemental_shape_coverage: dict[str, bool]
    libtorch_runtime_supplemental_parity_coverage: dict[str, bool]
    libtorch_runtime_raw_pointer_route: bool
    libtorch_trainer_link_probe: bool
    libtorch_matmul_coverage: dict[str, bool]
    matmul_provider_coverage: dict[str, bool]
    baseline_provider_coverage: dict[str, bool]
    torch_benchmark_coverage: dict[str, bool]
    python_stack_log_coverage: dict[str, bool]
    backend_selections: list[BackendSelection]
    attention_route_summaries: list[AttentionRouteSummary]
    unavailable_rows: list[UnavailableMetric]
    settings: dict[str, str]
    val_losses: list[float]
    train_steps: list[TrainStep]
    total_average_ms: float | None


DEFAULT_DECISION_REGISTRY = Path(__file__).with_name("sm120_promotion_decisions.json")


def finite_float(raw: str, label: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{label} is not finite: {raw}")
    return value


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(errors="replace")
    if not text.strip():
        raise ValueError(f"{path} is empty")
    return text


def normalize_key(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_")


def parse_metadata(summary: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw in summary.splitlines():
        match = META_RE.match(raw.strip())
        if match:
            metadata[normalize_key(match.group("key"))] = match.group("value")
    required = ("run_label", "artifact_dir", "train_output_dir", "git_commit")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"summary metadata missing: {', '.join(missing)}")
    return metadata


def require_success_marker(log_name: str, text: str) -> None:
    if re.search(r"\bFAIL\b|failed", text, flags=re.IGNORECASE):
        raise ValueError(f"{log_name} contains a failure marker")
    if re.search(r"\bSKIPPED\b", text):
        raise ValueError(f"{log_name} was skipped and is not correctness evidence")
    if "smoke OK" not in text and "passed" not in text:
        raise ValueError(f"{log_name} is missing a smoke success marker")


def add_metric(metrics: list[BenchmarkMetric], suite: str, name: str, shape: str, stack: str, value: str) -> None:
    metrics.append(BenchmarkMetric(suite, name, shape, stack, finite_float(value, f"{suite} {name} {stack}")))


def parse_matmul(text: str) -> list[BenchmarkMetric]:
    metrics: list[BenchmarkMetric] = []
    current_shape = ""
    for raw in text.splitlines():
        line = raw.strip()
        match = MATMUL_SHAPE_RE.match(line)
        if match:
            current_shape = (
                f"{match.group('name')} M={match.group('m')} N={match.group('n')} K={match.group('k')} "
                f"bias={match.group('bias')} gelu={match.group('gelu')}"
            )
            continue
        if not current_shape:
            continue
        match = MATMUL_FWD_GELU_RE.match(line)
        if match:
            add_metric(metrics, "matmul", "fwd+gelu", current_shape, "TK fused", match.group("tk_fused"))
            add_metric(metrics, "matmul", "fwd+gelu", current_shape, "TK explicit", match.group("tk_explicit"))
            add_metric(metrics, "matmul", "fwd+gelu", current_shape, "cuBLASLt", match.group("cublaslt"))
            if match.group("cublas"):
                add_metric(metrics, "matmul", "fwd+gelu", current_shape, "cuBLAS explicit", match.group("cublas"))
            continue
        match = MATMUL_FWD_RE.match(line)
        if match:
            add_metric(metrics, "matmul", "fwd", current_shape, "TK", match.group("tk"))
            add_metric(metrics, "matmul", "fwd", current_shape, "cuBLASLt", match.group("cublaslt"))
            if match.group("cublas"):
                add_metric(metrics, "matmul", "fwd", current_shape, "cuBLAS", match.group("cublas"))
            continue
        match = MATMUL_DINP_RE.match(line)
        if match:
            add_metric(metrics, "matmul", "dInp", current_shape, "TK", match.group("tk"))
            add_metric(metrics, "matmul", "dInp", current_shape, "cuBLASLt", match.group("cublaslt"))
            if match.group("cublas"):
                add_metric(metrics, "matmul", "dInp", current_shape, "cuBLAS", match.group("cublas"))
            continue
        match = MATMUL_DINP_DGELU_RE.match(line)
        if match:
            add_metric(metrics, "matmul", "dInp+dGeLU", current_shape, "TK", match.group("tk"))
            add_metric(metrics, "matmul", "dInp+dGeLU", current_shape, "cuBLASLt fused", match.group("cublas_fused"))
            add_metric(
                metrics,
                "matmul",
                "dInp+dGeLU",
                current_shape,
                "cuBLASLt explicit",
                match.group("cublas_explicit"),
            )
            if match.group("cublas"):
                add_metric(metrics, "matmul", "dInp+dGeLU", current_shape, "cuBLAS explicit", match.group("cublas"))
            continue
        match = MATMUL_DW_RE.match(line)
        if match:
            add_metric(metrics, "matmul", "dW", current_shape, "TK", match.group("tk"))
            add_metric(metrics, "matmul", "dW", current_shape, "cuBLASLt", match.group("cublaslt"))
            if match.group("cublas"):
                add_metric(metrics, "matmul", "dW", current_shape, "cuBLAS", match.group("cublas"))
            continue
        match = MATMUL_DW_ACCUM_RE.match(line)
        if match:
            add_metric(metrics, "matmul", "dW+accum", current_shape, "TK", match.group("tk"))
            add_metric(metrics, "matmul", "dW+accum", current_shape, "cuBLASLt", match.group("cublaslt"))
            if match.group("cublas"):
                add_metric(metrics, "matmul", "dW+accum", current_shape, "cuBLAS", match.group("cublas"))
    if not metrics:
        raise ValueError("bench_sm120_matmul.log did not contain parseable timings")
    if not any(metric.name == "dInp+dGeLU" for metric in metrics):
        raise ValueError("bench_sm120_matmul.log did not contain fused dGELU dInput timings")
    if not any(metric.name == "dW+accum" for metric in metrics):
        raise ValueError("bench_sm120_matmul.log did not contain accumulated dWeight timings")
    return metrics


def parse_python_matmul(text: str, *, stack: str, log_name: str) -> list[BenchmarkMetric]:
    metrics: list[BenchmarkMetric] = []
    current_shape = ""
    op_names = {
        "fwd": "fwd",
        "fwd+GeLU": "fwd+gelu",
        "dInp": "dInp",
        "dInp+dGeLU": "dInp+dGeLU",
        "dW": "dW",
        "dW+accum": "dW+accum",
    }
    for raw in text.splitlines():
        line = raw.strip()
        match = MATMUL_SHAPE_RE.match(line)
        if match:
            current_shape = (
                f"{match.group('name')} M={match.group('m')} N={match.group('n')} K={match.group('k')} "
                f"bias={match.group('bias')} gelu={match.group('gelu')}"
            )
            continue
        if not current_shape:
            continue
        match = PYTHON_MATMUL_RE.match(line)
        if match and match.group("stack") == stack:
            add_metric(metrics, "matmul", op_names[match.group("op")], current_shape, stack, match.group("us"))
    if not metrics:
        raise ValueError(f"{log_name} did not contain parseable timings")
    return metrics


def parse_torch_matmul(text: str) -> list[BenchmarkMetric]:
    return parse_python_matmul(text, stack="Torch", log_name="bench_sm120_torch_matmul.log")


def parse_libtorch_matmul(text: str) -> list[BenchmarkMetric]:
    return parse_python_matmul(text, stack="Torch C++", log_name="bench_sm120_libtorch_matmul.log")


def parse_triton_matmul(text: str) -> list[BenchmarkMetric]:
    return parse_python_matmul(text, stack="Triton", log_name="bench_sm120_triton_matmul.log")


def parse_unavailable_rows(text: str, *, suite: str, name: str) -> list[UnavailableMetric]:
    rows: list[UnavailableMetric] = []
    for raw in text.splitlines():
        match = UNAVAILABLE_RE.match(raw.strip())
        if not match:
            continue
        rows.append(
            UnavailableMetric(
                suite=suite,
                name=name,
                shape=" ".join(match.group("shape").split()),
                stack=match.group("stack").strip(),
                reason=match.group("reason").strip(),
            )
        )
    return rows


def parse_runtime_unavailable_rows(text: str) -> list[UnavailableMetric]:
    rows: list[UnavailableMetric] = []
    for raw in text.splitlines():
        match = RUNTIME_UNAVAILABLE_RE.match(raw.strip())
        if not match:
            continue
        rows.append(
            UnavailableMetric(
                suite="runtime",
                name=match.group("name").strip(),
                shape=match.group("shape").strip(),
                stack=match.group("stack").strip(),
                reason=match.group("reason").strip(),
            )
        )
    return rows


def parse_attention(text: str, *, require_all: bool = True) -> list[BenchmarkMetric]:
    metrics: list[BenchmarkMetric] = []
    for raw in text.splitlines():
        match = ATTENTION_RE.match(raw.strip())
        if not match:
            continue
        shape = (
            f"B={match.group('b')} T={match.group('t')} C={match.group('c')} "
            f"NH={match.group('nh')} HS={match.group('hs')}"
        )
        stack = match.group("stack") or "TK packed-QKV"
        add_metric(metrics, "attention", match.group("pass").lower(), shape, stack, match.group("us"))
    passes = {metric.name for metric in metrics}
    if require_all and {"forward", "backward"} - passes:
        raise ValueError("bench_sm120_attention.log must contain forward and backward timings")
    if not metrics:
        raise ValueError("attention benchmark log did not contain parseable timings")
    return metrics


def has_parseable_attention(text: str) -> bool:
    return any(ATTENTION_RE.match(raw.strip()) for raw in text.splitlines())


def validate_cudnn_packed_backward_route(text: str, expected_route: object) -> bool:
    if expected_route in (None, ""):
        return False
    expected = str(expected_route)
    marker = CUDNN_PACKED_BACKWARD_ROUTES.get(expected)
    if marker is None:
        raise ValueError(f"cuDNN packed attention benchmark has unknown manifest route: {expected}")
    if marker not in text:
        raise ValueError("cuDNN packed attention benchmark missing saved-forward backward route evidence")
    return True


def parse_layernorm(text: str, *, require_all: bool = True) -> list[BenchmarkMetric]:
    metrics: list[BenchmarkMetric] = []
    pass_names = {
        "Forward": "forward",
        "ForwardNative": "forward",
        "ForwardWithStats": "forward",
        "FusedResidualForward": "fused_residual_forward",
        "FusedResidualForwardNative": "fused_residual_forward",
        "FusedResidualForwardWithStats": "fused_residual_forward",
        "Backward": "backward",
        "BackwardNative": "backward",
        "BackwardDInput": "backward_dinput",
        "BackwardDInputNative": "backward_dinput",
        "BackwardDInputNativePlusGrads": "backward",
        "BackwardAtomicFP32": "backward",
    }
    for raw in text.splitlines():
        match = LAYERNORM_RE.match(raw.strip())
        if not match:
            continue
        shape = f"N={match.group('n')} C={match.group('c')}"
        raw_stack = match.group("stack") or "CUDA"
        raw_pass = match.group("pass")
        stack = layernorm_stack_label(raw_stack, raw_pass)
        add_metric(metrics, "layernorm", pass_names[raw_pass], shape, stack, match.group("us"))
    passes = {metric.name for metric in metrics}
    if require_all and {"forward", "fused_residual_forward", "backward"} - passes:
        raise ValueError("bench_sm120_layernorm.log must contain forward, fused residual forward, and backward timings")
    if not metrics:
        raise ValueError("LayerNorm benchmark log did not contain parseable timings")
    return metrics


def layernorm_stack_label(stack: str, pass_name: str) -> str:
    if stack == "Triton" and pass_name == "BackwardDInput":
        return "Triton dInput-only"
    if stack == "Triton" and pass_name == "BackwardAtomicFP32":
        return "Triton atomic FP32-grad"
    if stack != "Torch":
        return stack
    if pass_name == "BackwardDInputNativePlusGrads":
        return "Torch native+BF16-grads"
    if pass_name in {"ForwardNative", "FusedResidualForwardNative", "BackwardNative", "BackwardDInputNative"}:
        return "Torch native"
    if pass_name in {"ForwardWithStats", "FusedResidualForwardWithStats"}:
        return "Torch stats"
    return stack


def parse_runtime(text: str, *, require_all: bool = True) -> list[BenchmarkMetric]:
    metrics: list[BenchmarkMetric] = []
    for raw in text.splitlines():
        match = RUNTIME_RE.match(raw.strip())
        if not match:
            continue
        add_metric(
            metrics,
            "runtime",
            match.group("name"),
            match.group("shape").strip(),
            match.group("stack").strip(),
            match.group("us"),
        )
    found = {metric.name for metric in metrics}
    missing = sorted(EXPECTED_RUNTIME_KERNELS - found)
    if require_all and missing:
        raise ValueError(f"bench_sm120_runtime.log missing timings: {', '.join(missing)}")
    if not metrics:
        raise ValueError("runtime benchmark log did not contain parseable timings")
    return metrics


def parse_training(text: str) -> tuple[dict[str, str], list[float], list[TrainStep], float | None]:
    settings: dict[str, str] = {}
    val_losses: list[float] = []
    steps: list[TrainStep] = []
    total_average_ms: float | None = None
    for raw in text.splitlines():
        line = raw.strip()
        match = SETTING_RE.match(line)
        if match:
            settings[normalize_key(match.group("key"))] = match.group("value").strip()
            continue
        match = VAL_LOSS_RE.match(line)
        if match:
            val_losses.append(finite_float(match.group("loss"), "val loss"))
            continue
        match = STEP_RE.match(line)
        if match:
            steps.append(
                TrainStep(
                    step=int(match.group("step")),
                    total=int(match.group("total")),
                    loss=finite_float(match.group("loss"), "train loss"),
                    norm=finite_float(match.group("norm"), "train norm"),
                    lr=finite_float(match.group("lr"), "learning rate"),
                    ms=finite_float(match.group("ms"), "step milliseconds"),
                    mfu=finite_float(match.group("mfu"), "MFU"),
                    tok_s=int(match.group("tok_s")),
                )
            )
            continue
        match = AVG_RE.match(line)
        if match:
            total_average_ms = finite_float(match.group("ms"), "total average iteration time")
    return settings, val_losses, steps, total_average_ms


def validate_training(settings: dict[str, str], val_losses: list[float], steps: list[TrainStep],
                      total_average_ms: float | None, check_sm120_defaults: bool) -> None:
    if not steps:
        raise ValueError("train_gpt2cu.log did not contain step timings")
    if total_average_ms is None:
        raise ValueError("train_gpt2cu.log did not contain total average iteration time")
    if not val_losses:
        raise ValueError("train_gpt2cu.log did not contain validation loss")
    if check_sm120_defaults:
        expected = {
            "use_master_weights": "disabled",
            "gelu_fusion": "1",
            "precision": "BF16",
        }
        for key, value in expected.items():
            actual = settings.get(key)
            if actual != value:
                raise ValueError(f"training setting {key}={actual!r}; expected {value!r}")


def validate_checkpoints(metadata: dict[str, str]) -> None:
    train_dir = Path(metadata["train_output_dir"])
    if not train_dir.exists():
        return
    leftovers = sorted(train_dir.glob("model_*.bin")) + sorted(train_dir.glob("state_*.bin"))
    if leftovers:
        rendered = ", ".join(str(path) for path in leftovers[:5])
        if len(leftovers) > 5:
            rendered += f", ... ({len(leftovers)} total)"
        raise ValueError(f"checkpoint cleanup left bulky files in {train_dir}: {rendered}")


def validate_manifest(round_dir: Path, metadata: dict[str, str]) -> dict[str, object]:
    path = round_dir / "round-manifest.json"
    payload = json.loads(read_text(path))
    if payload.get("schema_version") != 1:
        raise ValueError("round-manifest.json has unsupported schema_version")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("round-manifest.json missing config object")
    expected = {
        "run_label": metadata["run_label"],
        "artifact_dir": metadata["artifact_dir"],
        "train_out_dir": metadata["train_output_dir"],
        "device_arch": "SM120",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"round manifest config {key}={config.get(key)!r}; expected {value!r}")
    git = payload.get("git")
    if not isinstance(git, dict) or not git.get("short_commit"):
        raise ValueError("round-manifest.json missing git.short_commit")
    binaries = payload.get("binaries")
    if not isinstance(binaries, list) or not binaries:
        raise ValueError("round-manifest.json missing binaries list")
    binary_by_path: dict[str, dict[str, object]] = {}
    for index, binary in enumerate(binaries):
        if not isinstance(binary, dict):
            raise ValueError(f"round-manifest.json binaries[{index}] must be an object")
        path = binary.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"round-manifest.json binaries[{index}] missing path")
        if path in binary_by_path:
            raise ValueError(f"round-manifest.json has duplicate binary row: {path}")
        binary_by_path[path] = binary
    missing_expected = [
        path
        for path in EXPECTED_MANIFEST_BINARIES
        if path not in binary_by_path or not binary_by_path[path].get("exists")
    ]
    if missing_expected:
        raise ValueError(f"round manifest missing expected binaries: {', '.join(missing_expected)}")
    missing_sha = [
        path
        for path, binary in binary_by_path.items()
        if binary.get("exists") and not binary.get("sha256")
    ]
    if missing_sha:
        raise ValueError(f"round manifest binaries missing sha256: {', '.join(missing_sha)}")
    return payload


def manifest_flag(manifest: dict[str, object], key: str) -> bool:
    config = manifest.get("config", {})
    if not isinstance(config, dict):
        return False
    return str(config.get(key, "0")) == "1"


def validate_python_stack_benchmark_logs(round_dir: Path) -> dict[str, bool]:
    coverage: dict[str, bool] = {}
    missing: list[str] = []
    empty: list[str] = []
    for log_name in PYTHON_STACK_BENCHMARK_LOGS:
        path = round_dir / log_name
        exists = path.exists()
        coverage[log_name] = exists
        if not exists:
            missing.append(log_name)
            continue
        if not path.read_text().strip():
            empty.append(log_name)
    if missing:
        raise ValueError(
            "manifest requested Python stack benchmarks but logs are missing: "
            + ", ".join(missing)
        )
    if empty:
        raise ValueError(
            "manifest requested Python stack benchmarks but logs are empty: "
            + ", ".join(empty)
        )
    return coverage


def manifest_libtorch_matmul_shapes(manifest: dict[str, object]) -> tuple[str, ...]:
    config = manifest.get("config", {})
    if not isinstance(config, dict):
        return ()
    raw_shapes = str(config.get("libtorch_matmul_shapes", "")).strip()
    if not raw_shapes:
        return ()
    shapes = tuple(raw_shapes.split())
    unknown = sorted(set(shapes) - set(MATMUL_SELECTION_SHAPES))
    if unknown:
        raise ValueError("manifest requested unknown LibTorch matmul shapes: " + ", ".join(unknown))
    return shapes


def validate_libtorch_matmul_benchmark(round_dir: Path, manifest: dict[str, object], benchmarks: list[BenchmarkMetric]) -> dict[str, bool]:
    if not manifest_flag(manifest, "run_libtorch_matmul_benchmarks"):
        return {}
    path = round_dir / "bench_sm120_libtorch_matmul.log"
    if not path.exists():
        raise ValueError("manifest requested LibTorch matmul benchmarks but bench_sm120_libtorch_matmul.log is missing")
    text = path.read_text()
    if not text.strip():
        raise ValueError("manifest requested LibTorch matmul benchmarks but bench_sm120_libtorch_matmul.log is empty")
    if LIBTORCH_MATMUL_ROUTE not in text:
        raise ValueError("LibTorch matmul benchmark missing standalone C++ cached from_blob route evidence")
    shape_names = manifest_libtorch_matmul_shapes(manifest) or ("fc",)
    required = {
        (op_name, MATMUL_SELECTION_SHAPES[shape_name])
        for shape_name in shape_names
        for op_name in ("dW", "dW+accum")
    }
    present = {
        (metric.name, metric.shape)
        for metric in benchmarks
        if metric.suite == "matmul" and metric.stack == "Torch C++"
    }
    missing = sorted(required - present)
    if missing:
        rendered = ", ".join(f"{op} {shape}" for op, shape in missing)
        raise ValueError("LibTorch matmul benchmark missing exact dWeight rows: " + rendered)
    return {f"{op}:{shape}": (op, shape) in present for op, shape in sorted(required)}


def validate_backend_stacks(round_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    path = round_dir / "backend-stacks.json"
    payload = json.loads(read_text(path))
    stacks = payload.get("stacks")
    if not isinstance(stacks, list):
        raise ValueError(f"{path} must contain a stacks list")
    found: dict[str, dict[str, object]] = {}
    for index, raw_stack in enumerate(stacks):
        if not isinstance(raw_stack, dict):
            raise ValueError(f"{path} stacks[{index}] must be an object")
        stack = raw_stack.get("stack")
        status = raw_stack.get("status")
        evidence = raw_stack.get("evidence")
        candidate_use = raw_stack.get("candidate_use")
        next_action = raw_stack.get("next_action")
        if not isinstance(stack, str) or not stack:
            raise ValueError(f"{path} stacks[{index}] missing stack name")
        if stack in found:
            raise ValueError(f"{path} has duplicate stack row: {stack}")
        if status not in STACK_STATUSES:
            raise ValueError(f"{path} stack {stack} has invalid status: {status!r}")
        if not isinstance(evidence, list) or not evidence or not all(isinstance(item, str) and item for item in evidence):
            raise ValueError(f"{path} stack {stack} must include non-empty evidence strings")
        if not isinstance(candidate_use, str) or not candidate_use:
            raise ValueError(f"{path} stack {stack} missing candidate_use")
        if not isinstance(next_action, str) or not next_action:
            raise ValueError(f"{path} stack {stack} missing next_action")
        found[stack] = raw_stack
    required = set(OBJECTIVE_STACKS) | set(ENVIRONMENT_STACKS)
    missing = sorted(required - set(found))
    if missing:
        raise ValueError(f"{path} missing required stack probe rows: {', '.join(missing)}")
    family_matrix = payload.get("family_matrix")
    if not isinstance(family_matrix, list):
        raise ValueError(f"{path} must contain a family_matrix list")
    found_matrix: dict[tuple[str, str], dict[str, object]] = {}
    objective_families = set(OBJECTIVE_FAMILIES)
    objective_stacks = set(OBJECTIVE_STACKS)
    for index, raw_row in enumerate(family_matrix):
        if not isinstance(raw_row, dict):
            raise ValueError(f"{path} family_matrix[{index}] must be an object")
        family = raw_row.get("family")
        stack = raw_row.get("stack")
        status = raw_row.get("status")
        reason = raw_row.get("reason")
        next_action = raw_row.get("next_action")
        if not isinstance(family, str) or family not in objective_families:
            raise ValueError(f"{path} family_matrix[{index}] has invalid family: {family!r}")
        if not isinstance(stack, str) or stack not in objective_stacks:
            raise ValueError(f"{path} family_matrix[{index}] has invalid stack: {stack!r}")
        key = (family, stack)
        if key in found_matrix:
            raise ValueError(f"{path} has duplicate family_matrix row: {family}/{stack}")
        if status not in FAMILY_STACK_STATUSES:
            raise ValueError(f"{path} family_matrix {family}/{stack} has invalid status: {status!r}")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"{path} family_matrix {family}/{stack} missing reason")
        if status == "not_applicable" and reason.lower() in {"n/a", "na", "none"}:
            raise ValueError(f"{path} family_matrix {family}/{stack} needs a not_applicable reason")
        if not isinstance(next_action, str) or not next_action:
            raise ValueError(f"{path} family_matrix {family}/{stack} missing next_action")
        stack_probe_status = found[stack].get("status")
        if stack_probe_status in {"missing", "blocked"} and status in {"baseline", "candidate", "fallback"}:
            raise ValueError(
                f"{path} family_matrix {family}/{stack} status {status!r} conflicts with "
                f"stack probe status {stack_probe_status!r}"
            )
        found_matrix[key] = raw_row
    required_matrix = {(family, stack) for family in OBJECTIVE_FAMILIES for stack in OBJECTIVE_STACKS}
    missing_matrix = sorted(required_matrix - set(found_matrix))
    if missing_matrix:
        rendered = ", ".join(f"{family}/{stack}" for family, stack in missing_matrix[:10])
        if len(missing_matrix) > 10:
            rendered += f", ... ({len(missing_matrix)} total)"
        raise ValueError(f"{path} missing required family_matrix rows: {rendered}")
    return stacks, family_matrix


def benchmark_coverage(metrics: list[BenchmarkMetric]) -> dict[str, bool]:
    by_suite: dict[str, set[str]] = {}
    runtime_names: set[str] = set()
    for metric in metrics:
        by_suite.setdefault(metric.suite, set()).add(metric.name)
        if metric.suite == "runtime":
            runtime_names.add(metric.name)
    matmul = by_suite.get("matmul", set())
    attention = by_suite.get("attention", set())
    layernorm = by_suite.get("layernorm", set())
    return {
        "gemm_forward": "fwd" in matmul,
        "gemm_forward_fused_gelu": "fwd+gelu" in matmul,
        "gemm_backward_dinput": "dInp" in matmul,
        "gemm_backward_dinput_fused_dgelu": "dInp+dGeLU" in matmul,
        "gemm_backward_dweight": "dW" in matmul,
        "gemm_backward_dweight_accum": "dW+accum" in matmul,
        "bias_add": "bias_add" in runtime_names,
        "bias_gradient_reduce": "bias_grad_reduce" in runtime_names,
        "gelu_forward": "gelu_forward" in runtime_names,
        "gelu_backward": "gelu_backward_inplace" in runtime_names,
        "attention_forward": "forward" in attention,
        "attention_backward": "backward" in attention,
        "layernorm_forward": "forward" in layernorm,
        "layernorm_fused_residual_forward": "fused_residual_forward" in layernorm,
        "layernorm_backward": "backward" in layernorm,
        "classifier_softmax_cross_entropy_dlogits": "fused_classifier" in runtime_names,
        "adamw": "adamw_update" in runtime_names,
        "global_norm": "global_norm_squared" in runtime_names,
        "encoder_forward": "encoder_forward" in runtime_names,
        "cuda_memset": "cuda_memset" in runtime_names,
        "cuda_copy_d2d": "cuda_copy_d2d" in runtime_names,
    }


def matmul_shape_name(shape: str) -> str:
    return shape.split(maxsplit=1)[0] if shape else ""


def matmul_shape_coverage(metrics: list[BenchmarkMetric]) -> dict[str, bool]:
    present = {
        (metric.name, matmul_shape_name(metric.shape))
        for metric in metrics
        if metric.suite == "matmul"
    }
    coverage: dict[str, bool] = {}
    for op_name, shapes in MATMUL_SHAPE_REQUIREMENTS:
        for shape_name in shapes:
            coverage[f"{op_name}:{shape_name}"] = (op_name, shape_name) in present
    return coverage


def validate_libtorch_trainer_link_probe(round_dir: Path, manifest: dict[str, object]) -> bool:
    if not manifest_flag(manifest, "run_libtorch_trainer_link_probe"):
        return False
    path = round_dir / LIBTORCH_TRAINER_LINK_LOG
    if not path.exists():
        raise ValueError(
            f"manifest requested LibTorch trainer link probe but {LIBTORCH_TRAINER_LINK_LOG} is missing"
        )
    text = path.read_text()
    if not text.strip():
        raise ValueError(
            f"manifest requested LibTorch trainer link probe but {LIBTORCH_TRAINER_LINK_LOG} is empty"
        )
    required = (
        "LibTorch trainer link route: standalone executable without torch_python",
        "LibTorch trainer link compile: PASS",
        "LibTorch trainer link runtime: PASS zero/copy from_blob executable",
        "LibTorch trainer link probe: PASS",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise ValueError("LibTorch trainer link probe log missing markers: " + ", ".join(missing))
    return True


def runtime_shape_coverage(metrics: list[BenchmarkMetric]) -> dict[str, bool]:
    # Required runtime shape rows are baseline coverage gates. Optional stacks
    # such as Torch may add comparison rows for the same shape, but they must
    # not mask a missing CUDA/CUDA-runtime baseline row.
    present = {
        (metric.name, metric.shape)
        for metric in metrics
        if metric.suite == "runtime" and normalized_stack_key(metric.stack) == "cuda"
    }
    coverage: dict[str, bool] = {}
    for kernel_name, shapes in RUNTIME_SHAPE_REQUIREMENTS:
        for shape in shapes:
            coverage[f"{kernel_name}:{shape}"] = (kernel_name, shape) in present
    return coverage


def libtorch_runtime_shape_coverage(metrics: list[BenchmarkMetric]) -> dict[str, bool]:
    present = {
        (metric.name, metric.shape)
        for metric in metrics
        if metric.suite == "runtime" and metric.stack == "Torch C++"
    }
    coverage: dict[str, bool] = {}
    for kernel_name, shapes in LIBTORCH_RUNTIME_SHAPE_REQUIREMENTS:
        for shape in shapes:
            coverage[f"{kernel_name}:{shape}"] = (kernel_name, shape) in present
    return coverage


def libtorch_runtime_supplemental_shape_coverage(metrics: list[BenchmarkMetric]) -> dict[str, bool]:
    present = {
        (metric.name, metric.shape)
        for metric in metrics
        if metric.suite == "runtime" and metric.stack == "Torch C++"
    }
    coverage: dict[str, bool] = {}
    for kernel_name, shapes in LIBTORCH_RUNTIME_SUPPLEMENTAL_SHAPE_REQUIREMENTS:
        for shape in shapes:
            coverage[f"{kernel_name}:{shape}"] = (kernel_name, shape) in present
    return coverage


def libtorch_runtime_parity_coverage(text: str) -> dict[str, bool]:
    present: set[tuple[str, str]] = set()
    for raw in text.splitlines():
        match = LIBTORCH_PARITY_RE.match(raw.strip())
        if not match:
            continue
        present.add((match.group("name"), match.group("shape").strip()))
    coverage: dict[str, bool] = {}
    for kernel_name, shapes in LIBTORCH_RUNTIME_SHAPE_REQUIREMENTS:
        for shape in shapes:
            coverage[f"{kernel_name}:{shape}"] = (kernel_name, shape) in present
    return coverage


def libtorch_runtime_supplemental_parity_coverage(text: str) -> dict[str, bool]:
    present: set[tuple[str, str]] = set()
    for raw in text.splitlines():
        match = LIBTORCH_PARITY_RE.match(raw.strip())
        if not match:
            continue
        present.add((match.group("name"), match.group("shape").strip()))
    coverage: dict[str, bool] = {}
    for kernel_name, shapes in LIBTORCH_RUNTIME_SUPPLEMENTAL_SHAPE_REQUIREMENTS:
        for shape in shapes:
            coverage[f"{kernel_name}:{shape}"] = (kernel_name, shape) in present
    return coverage


def validate_libtorch_runtime_raw_pointer_route(text: str, expected_route: object = None) -> bool:
    observed_routes = [
        route
        for route, marker in LIBTORCH_RUNTIME_ROUTE_MARKERS.items()
        if marker in text
    ]
    if expected_route in (None, ""):
        if observed_routes:
            return True
        raise ValueError("LibTorch runtime benchmark missing cached from_blob raw-pointer route evidence")
    if not isinstance(expected_route, str) or expected_route not in LIBTORCH_RUNTIME_ROUTE_MARKERS:
        raise ValueError(f"LibTorch runtime benchmark has unknown manifest route {expected_route!r}")
    if expected_route not in observed_routes:
        observed = ", ".join(observed_routes) if observed_routes else "no raw-pointer route marker"
        raise ValueError(
            f"LibTorch runtime route mismatch: manifest requested {expected_route!r} but "
            f"bench_sm120_libtorch_runtime.log records {observed}"
        )
    return True


def matmul_provider_key(stack: str) -> str:
    if stack.startswith("cuBLASLt"):
        return "cublaslt"
    if stack.startswith("cuBLAS"):
        return "cublas"
    if stack.startswith("TK"):
        return "tk"
    return stack.lower().replace(" ", "_")


def normalized_stack_key(stack: str) -> str:
    if stack.startswith("ThunderKittens") or stack.startswith("TK"):
        return "tk"
    if stack.startswith("cuBLASLt"):
        return "cublaslt"
    if stack.startswith("cuBLAS"):
        return "cublas"
    if stack.startswith("Plain CUDA") or stack.startswith("CUDA"):
        return "cuda"
    if stack.startswith("cuDNN"):
        return "cudnn"
    if stack.startswith("Triton"):
        return "triton"
    if stack.startswith("Torch") or stack.startswith("LibTorch"):
        return "torch"
    if stack.startswith("CuTeDSL"):
        return "cutedsl"
    return stack.lower().replace(" ", "_")


def runtime_objective_shape(kernel: str, shape: str, stack: str) -> str:
    if stack.startswith("Torch") and kernel == "adamw_update":
        return shape.replace(" fp32-state", "")
    return shape


def expected_torch_benchmark_keys() -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for kernel, shape_names in MATMUL_SHAPE_REQUIREMENTS:
        for shape_name in shape_names:
            keys.add(("matmul", kernel, MATMUL_SELECTION_SHAPES[shape_name]))
    for kernel in ("forward", "backward"):
        keys.add(("attention", kernel, ATTENTION_SELECTION_SHAPE))
    for kernel in ("forward", "fused_residual_forward", "backward"):
        keys.add(("layernorm", kernel, LAYERNORM_SELECTION_SHAPE))
    for kernel, shapes in RUNTIME_SELECTION_SHAPES.items():
        for shape in shapes:
            keys.add(("runtime", kernel, shape))
    return keys


def torch_benchmark_key(metric: BenchmarkMetric) -> tuple[str, str, str] | None:
    if normalized_stack_key(metric.stack) != "torch":
        return None
    shape = runtime_objective_shape(metric.name, metric.shape, metric.stack)
    return (metric.suite, metric.name, shape)


def torch_unavailable_key(row: UnavailableMetric) -> tuple[str, str, str] | None:
    if normalized_stack_key(row.stack) != "torch":
        return None
    shape = runtime_objective_shape(row.name, row.shape, row.stack)
    return (row.suite, row.name, shape)


def torch_benchmark_coverage(
    metrics: list[BenchmarkMetric],
    unavailable_rows: list[UnavailableMetric] | None = None,
) -> dict[str, bool]:
    present = {
        key
        for metric in metrics
        for key in (torch_benchmark_key(metric),)
        if key is not None
    }
    if unavailable_rows is not None:
        present.update(
            key
            for row in unavailable_rows
            for key in (torch_unavailable_key(row),)
            if key is not None
        )
    return {
        f"{suite}/{kernel}/{shape}": (suite, kernel, shape) in present
        for suite, kernel, shape in sorted(expected_torch_benchmark_keys())
    }


def validate_torch_benchmark_coverage(
    metrics: list[BenchmarkMetric],
    unavailable_rows: list[UnavailableMetric],
) -> dict[str, bool]:
    coverage = torch_benchmark_coverage(metrics, unavailable_rows)
    missing = [name for name, covered in coverage.items() if not covered]
    if missing:
        rendered = ", ".join(missing[:10])
        if len(missing) > 10:
            rendered += f", ... ({len(missing)} total)"
        raise ValueError(f"Torch benchmark coverage missing objective rows: {rendered}")
    return coverage


def normalize_match_value(raw: object) -> str:
    return " ".join(str(raw).strip().split())


def load_promotion_decisions(path: Path | None) -> list[PromotionDecision]:
    if path is None or not path.exists():
        return []
    payload = json.loads(read_text(path))
    if payload.get("schema_version") != 1:
        raise ValueError(f"{path} has unsupported schema_version")
    rows = payload.get("decisions")
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a decisions list")
    decisions: list[PromotionDecision] = []
    seen: set[tuple[str, str, str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path} decisions[{index}] must be an object")
        match = row.get("match")
        if not isinstance(match, dict):
            raise ValueError(f"{path} decisions[{index}] missing match object")
        suite = normalize_match_value(match.get("suite", ""))
        kernel = normalize_match_value(match.get("kernel", ""))
        shape = normalize_match_value(match.get("shape", ""))
        selected_stack = normalize_match_value(match.get("selected_stack", ""))
        key = (suite, kernel, shape, selected_stack)
        if not all(key):
            raise ValueError(f"{path} decisions[{index}] has an incomplete match")
        if key in seen:
            raise ValueError(f"{path} has duplicate decision match: {key}")
        seen.add(key)
        status = normalize_match_value(row.get("status", ""))
        if not status:
            raise ValueError(f"{path} decisions[{index}] missing status")
        active = row.get("active", False)
        if not isinstance(active, bool):
            raise ValueError(f"{path} decisions[{index}] active must be a boolean")
        decision = normalize_match_value(row.get("decision", ""))
        if not decision:
            raise ValueError(f"{path} decisions[{index}] missing decision")
        evidence = row.get("evidence", [])
        if not isinstance(evidence, list) or not all(isinstance(item, str) and item for item in evidence):
            raise ValueError(f"{path} decisions[{index}] evidence must be a list of non-empty strings")
        decisions.append(
            PromotionDecision(
                suite=suite,
                kernel=kernel,
                shape=shape,
                selected_stack=selected_stack,
                status=status,
                active=active,
                decision=decision,
                evidence=tuple(evidence),
            )
        )
    return decisions


def matmul_provider_coverage(metrics: list[BenchmarkMetric]) -> dict[str, bool]:
    present = {
        (metric.name, matmul_shape_name(metric.shape), matmul_provider_key(metric.stack))
        for metric in metrics
        if metric.suite == "matmul"
    }
    coverage: dict[str, bool] = {}
    for op_name, shapes in MATMUL_SHAPE_REQUIREMENTS:
        for shape_name in shapes:
            for provider in MATMUL_PROVIDER_LABELS:
                coverage[f"{op_name}:{shape_name}:{provider}"] = (op_name, shape_name, provider) in present
    return coverage


def baseline_provider_coverage(
    metrics: list[BenchmarkMetric],
    family_matrix: list[dict[str, object]],
) -> dict[str, bool]:
    present = {
        (metric.suite, metric.name, normalized_stack_key(metric.stack))
        for metric in metrics
    }
    coverage: dict[str, bool] = {}
    for row in family_matrix:
        if row.get("status") != "baseline":
            continue
        family = row.get("family")
        stack = row.get("stack")
        if not isinstance(family, str) or not isinstance(stack, str):
            continue
        metric_key = FAMILY_METRIC_KEYS.get(family)
        if metric_key is None:
            continue
        suite, name = metric_key
        stack_key = normalized_stack_key(stack)
        coverage[f"{family}:{stack_key}"] = (suite, name, stack_key) in present
    return coverage


def validate_benchmark_coverage(metrics: list[BenchmarkMetric]) -> dict[str, bool]:
    coverage = benchmark_coverage(metrics)
    missing = [name for name, present in coverage.items() if not present]
    if missing:
        raise ValueError(f"benchmark coverage missing objective families: {', '.join(missing)}")
    return coverage


def validate_matmul_shape_coverage(metrics: list[BenchmarkMetric]) -> dict[str, bool]:
    coverage = matmul_shape_coverage(metrics)
    missing = [name for name, present in coverage.items() if not present]
    if missing:
        raise ValueError(f"benchmark matmul shape coverage missing objective rows: {', '.join(missing)}")
    return coverage


def validate_runtime_shape_coverage(metrics: list[BenchmarkMetric]) -> dict[str, bool]:
    coverage = runtime_shape_coverage(metrics)
    missing = [name for name, present in coverage.items() if not present]
    if missing:
        raise ValueError(f"benchmark runtime shape coverage missing objective rows: {', '.join(missing)}")
    return coverage


def validate_libtorch_runtime_shape_coverage(metrics: list[BenchmarkMetric]) -> dict[str, bool]:
    coverage = libtorch_runtime_shape_coverage(metrics)
    missing = [name for name, present in coverage.items() if not present]
    if missing:
        raise ValueError(f"LibTorch runtime benchmark missing exact memory rows: {', '.join(missing)}")
    return coverage


def validate_libtorch_runtime_parity_coverage(text: str) -> dict[str, bool]:
    coverage = libtorch_runtime_parity_coverage(text)
    missing = [name for name, present in coverage.items() if not present]
    if missing:
        raise ValueError(f"LibTorch runtime parity missing exact memory rows: {', '.join(missing)}")
    return coverage


def validate_libtorch_runtime_supplemental_shape_coverage(metrics: list[BenchmarkMetric]) -> dict[str, bool]:
    coverage = libtorch_runtime_supplemental_shape_coverage(metrics)
    missing = [name for name, present in coverage.items() if not present]
    if missing:
        raise ValueError(f"LibTorch runtime benchmark missing supplemental exact rows: {', '.join(missing)}")
    return coverage


def validate_libtorch_runtime_supplemental_parity_coverage(text: str) -> dict[str, bool]:
    coverage = libtorch_runtime_supplemental_parity_coverage(text)
    missing = [name for name, present in coverage.items() if not present]
    if missing:
        raise ValueError(f"LibTorch runtime parity missing supplemental exact rows: {', '.join(missing)}")
    return coverage


def validate_matmul_provider_coverage(metrics: list[BenchmarkMetric]) -> dict[str, bool]:
    coverage = matmul_provider_coverage(metrics)
    missing = [name for name, present in coverage.items() if not present]
    if missing:
        raise ValueError(f"benchmark matmul provider coverage missing objective stacks: {', '.join(missing)}")
    return coverage


def validate_baseline_provider_coverage(
    metrics: list[BenchmarkMetric],
    family_matrix: list[dict[str, object]],
) -> dict[str, bool]:
    coverage = baseline_provider_coverage(metrics, family_matrix)
    missing = [name for name, present in coverage.items() if not present]
    if missing:
        raise ValueError(f"benchmark baseline provider coverage missing rows: {', '.join(missing)}")
    return coverage


def selection_note(metric: BenchmarkMetric) -> tuple[str, str]:
    stack_key = normalized_stack_key(metric.stack)
    if metric.suite == "attention" and metric.stack == "Torch":
        return (
            "python separated-Q/K/V",
            "Use Torch SDPA for already-separated Q/K/V experiments; packed trainer selection must compare TorchPacked.",
        )
    if metric.suite == "attention" and metric.stack == "TorchPacked":
        return (
            "trainer packed-QKV prototype",
            "TorchPacked matches the trainer layout but still requires a trainer-callable integration gate before promotion.",
        )
    if metric.suite == "attention" and metric.stack == "TorchMaterializedPacked":
        return (
            "trainer packed-QKV materialized prototype",
            "TorchMaterializedPacked includes explicit Q/K/V materialization before SDPA; keep only if it beats packed TK.",
        )
    if metric.suite == "attention" and metric.stack == "cuDNN":
        return (
            "cuDNN separated-Q/K/V prototype",
            "cuDNN SDPA is a direct attention-library comparison row; trainer use still needs a packed route and build-gate decision.",
        )
    if metric.suite == "attention" and metric.stack == "cuDNNPacked":
        return (
            "cuDNN packed-QKV prototype",
            "cuDNNPacked includes trainer-layout view/copy overhead but remains dev-only until explicit cuDNN trainer linkage is scoped.",
        )
    if metric.suite == "matmul" and stack_key == "torch":
        if metric.stack == "Torch C++":
            return (
                "C++ API prototype",
                "LibTorch C++ uses cached from_blob handles over existing CUDA pointers; trainer use still needs call-site integration and TinyStories smoke.",
            )
        return (
            "operator prototype",
            "Torch is the fastest observed operator row here; trainer use still needs an explicit libtorch/C++ integration gate.",
        )
    if metric.suite == "layernorm" and metric.name == "backward" and metric.stack == "Torch native":
        return (
            "trainer-compatible prototype",
            "Native Torch backward consumes saved mean/rstd and produces dInput/dweight/dbias; trainer use still needs a libtorch/C++ route gate.",
        )
    if metric.suite == "layernorm" and metric.name == "backward_dinput" and metric.stack == "Torch native":
        return (
            "partial backward prototype",
            "Torch native dInput-only output-mask row is useful for backward decomposition work but needs dweight/dbias before trainer promotion.",
        )
    if metric.suite == "layernorm" and metric.stack == "Torch native":
        return (
            "reference only",
            "Native Torch is useful where saved mean/rstd are not needed; trainer LayerNorm still needs stats-compatible state.",
        )
    if metric.suite == "layernorm" and metric.stack == "Torch stats":
        return (
            "trainer-compatible prototype",
            "Stats-producing Torch composition exposes trainer state but must beat the CUDA/Triton row before promotion.",
        )
    if metric.suite == "layernorm" and metric.stack == "Triton dInput-only":
        return (
            "partial backward prototype",
            "Triton dInput-only is useful for backward decomposition work but needs dweight/dbias before trainer promotion.",
        )
    if metric.suite == "layernorm" and metric.stack == "Triton atomic FP32-grad":
        return (
            "non-equivalent FP32-grad prototype",
            "Triton atomic backward produces full gradients but stores dweight/dbias as FP32; trainer promotion needs a BF16-gradient or explicit contract change.",
        )
    if metric.suite == "runtime" and metric.name == "adamw_update_bf16_state":
        return (
            "non-equivalent BF16-state reference",
            "Torch fused AdamW is usable only for BF16 moment-state experiments; the trainer default uses FP32 moment buffers.",
        )
    if metric.suite == "runtime" and metric.stack == "Torch C++":
        return (
            "C++ API prototype",
            "LibTorch C++ API row proves a possible trainer-callable dependency path; promotion still needs call-site integration and TinyStories smoke.",
        )
    if stack_key == "torch":
        return (
            "operator prototype",
            "Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate.",
        )
    if stack_key == "triton":
        return (
            "operator prototype",
            "Use as a Triton comparison row until a trainer-callable integration beats the current provider.",
        )
    if stack_key == "cudnn":
        return (
            "operator prototype",
            "Use as a cuDNN comparison row until a trainer-callable cuDNN route beats the current provider.",
        )
    if metric.stack == "TK packed-QKV":
        return (
            "current packed trainer route",
            "Current C++ attention route; keep unless a packed-QKV candidate beats it and passes the TinyStories smoke gate.",
        )
    if stack_key in {"tk", "cublas", "cublaslt"}:
        return (
            "C++ benchmark route",
            "Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence.",
        )
    if stack_key == "cuda":
        return (
            "CUDA benchmark route",
            "Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence.",
        )
    return (
        "benchmark route",
        "Fastest observed row for this exact benchmark shape; check route-specific gates before promotion.",
    )


def select_fastest_backends(metrics: list[BenchmarkMetric]) -> list[BackendSelection]:
    grouped: dict[tuple[str, str, str], list[BenchmarkMetric]] = {}
    for metric in metrics:
        grouped.setdefault((metric.suite, metric.name, metric.shape), []).append(metric)

    selections: list[BackendSelection] = []
    for (suite, name, shape), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda metric: metric.time_us)
        selected = ordered[0]
        next_row = ordered[1] if len(ordered) > 1 else None
        use_scope, note = selection_note(selected)
        selections.append(
            BackendSelection(
                suite=suite,
                name=name,
                shape=shape,
                selected_stack=selected.stack,
                selected_time_us=selected.time_us,
                next_stack=next_row.stack if next_row else "-",
                next_time_us=next_row.time_us if next_row else None,
                use_scope=use_scope,
                decision_note=note,
            )
        )
    return selections


def attention_route_scope(stack: str) -> tuple[str, bool]:
    if stack in {"TK packed-QKV", "TorchPacked", "TorchMaterializedPacked", "cuDNNPacked", "TritonPacked"}:
        return "packed trainer-layout route", True
    if stack in {"Torch", "cuDNN", "Triton"}:
        return "separated Q/K/V reference route", False
    return "attention route", False


def summarize_attention_routes(
    metrics: list[BenchmarkMetric],
    unavailable_rows: list[UnavailableMetric],
) -> list[AttentionRouteSummary]:
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for metric in metrics:
        if metric.suite != "attention":
            continue
        grouped.setdefault((metric.shape, metric.stack), {})[metric.name] = metric.time_us

    unavailable_by_key: dict[tuple[str, str], str] = {}
    for row in unavailable_rows:
        if row.suite == "attention":
            unavailable_by_key[(row.shape, row.stack)] = row.reason

    summaries: list[AttentionRouteSummary] = []
    for shape, stack in sorted(set(grouped) | set(unavailable_by_key)):
        rows = grouped.get((shape, stack), {})
        forward_us = rows.get("forward")
        backward_us = rows.get("backward")
        complete = forward_us is not None and backward_us is not None
        scope, trainer_layout = attention_route_scope(stack)
        summaries.append(
            AttentionRouteSummary(
                shape=shape,
                stack=stack,
                route_scope=scope,
                trainer_layout=trainer_layout,
                forward_us=forward_us,
                backward_us=backward_us,
                total_us=(forward_us + backward_us) if complete else None,
                complete=complete,
                unavailable_reason=None if complete else unavailable_by_key.get((shape, stack)),
            )
        )
    return sorted(
        summaries,
        key=lambda row: (
            row.shape,
            not row.trainer_layout,
            row.total_us if row.total_us is not None else math.inf,
            row.stack,
        ),
    )


def validate_round(round_dir: Path, *, require_correctness: bool, require_benchmarks: bool,
                   require_training: bool, forbid_checkpoints: bool,
                   check_sm120_defaults: bool, require_stack_probe: bool,
                   require_manifest: bool) -> RoundMetrics:
    summary = read_text(round_dir / "summary.md")
    metadata = parse_metadata(summary)
    read_text(round_dir / "build.log")
    manifest: dict[str, object] = {}
    if (round_dir / "round-manifest.json").exists() or require_manifest:
        manifest = validate_manifest(round_dir, metadata)
    backend_stacks: list[dict[str, object]] = []
    backend_family_matrix: list[dict[str, object]] = []
    stack_probe_path = round_dir / "backend-stacks.json"
    if stack_probe_path.exists():
        backend_stacks, backend_family_matrix = validate_backend_stacks(round_dir)
    elif require_stack_probe:
        raise FileNotFoundError(stack_probe_path)

    if require_correctness:
        for name in CORRECTNESS_TESTS:
            require_success_marker(f"{name}.log", read_text(round_dir / f"{name}.log"))

    benchmarks: list[BenchmarkMetric] = []
    coverage: dict[str, bool] = {}
    shape_coverage: dict[str, bool] = {}
    runtime_shape_rows: dict[str, bool] = {}
    provider_coverage: dict[str, bool] = {}
    baseline_coverage: dict[str, bool] = {}
    torch_coverage: dict[str, bool] = {}
    libtorch_runtime_shape_rows: dict[str, bool] = {}
    libtorch_runtime_parity_rows: dict[str, bool] = {}
    libtorch_runtime_supplemental_shape_rows: dict[str, bool] = {}
    libtorch_runtime_supplemental_parity_rows: dict[str, bool] = {}
    libtorch_runtime_raw_pointer_route = False
    libtorch_trainer_link_probe = False
    libtorch_matmul_rows: dict[str, bool] = {}
    python_stack_log_coverage: dict[str, bool] = {}
    backend_selections: list[BackendSelection] = []
    attention_route_summaries: list[AttentionRouteSummary] = []
    unavailable_rows: list[UnavailableMetric] = []
    if require_benchmarks:
        if manifest_flag(manifest, "run_python_stack_benchmarks"):
            python_stack_log_coverage = validate_python_stack_benchmark_logs(round_dir)
        benchmarks.extend(parse_matmul(read_text(round_dir / "bench_sm120_matmul.log")))
        torch_matmul_path = round_dir / "bench_sm120_torch_matmul.log"
        if torch_matmul_path.exists():
            benchmarks.extend(parse_torch_matmul(read_text(torch_matmul_path)))
        libtorch_matmul_path = round_dir / "bench_sm120_libtorch_matmul.log"
        if libtorch_matmul_path.exists():
            benchmarks.extend(parse_libtorch_matmul(read_text(libtorch_matmul_path)))
        cutedsl_matmul_path = round_dir / "bench_sm120_cutedsl_matmul.log"
        if cutedsl_matmul_path.exists():
            unavailable_rows.extend(parse_unavailable_rows(read_text(cutedsl_matmul_path), suite="matmul", name="cutedsl_gemm"))
        triton_matmul_path = round_dir / "bench_sm120_triton_matmul.log"
        if triton_matmul_path.exists():
            benchmarks.extend(parse_triton_matmul(read_text(triton_matmul_path)))
        benchmarks.extend(parse_attention(read_text(round_dir / "bench_sm120_attention.log")))
        torch_attention_path = round_dir / "bench_sm120_torch_attention.log"
        if torch_attention_path.exists():
            benchmarks.extend(parse_attention(read_text(torch_attention_path), require_all=False))
        cudnn_attention_path = round_dir / "bench_sm120_cudnn_attention.log"
        if cudnn_attention_path.exists():
            cudnn_attention_text = read_text(cudnn_attention_path)
            manifest_config = manifest.get("config", {}) if isinstance(manifest, dict) else {}
            expected_cudnn_route = (
                manifest_config.get("cudnn_packed_backward_route")
                if isinstance(manifest_config, dict)
                else None
            )
            validate_cudnn_packed_backward_route(cudnn_attention_text, expected_cudnn_route)
            if has_parseable_attention(cudnn_attention_text):
                benchmarks.extend(parse_attention(cudnn_attention_text, require_all=False))
        triton_attention_path = round_dir / "bench_sm120_triton_attention.log"
        if triton_attention_path.exists():
            triton_attention_text = read_text(triton_attention_path)
            benchmarks.extend(parse_attention(triton_attention_text, require_all=False))
            unavailable_rows.extend(
                parse_unavailable_rows(triton_attention_text, suite="attention", name="backward")
            )
        torch_classifier_path = round_dir / "bench_sm120_torch_classifier.log"
        if torch_classifier_path.exists():
            torch_classifier_text = read_text(torch_classifier_path)
            benchmarks.extend(parse_runtime(torch_classifier_text, require_all=False))
            unavailable_rows.extend(parse_runtime_unavailable_rows(torch_classifier_text))
        triton_classifier_path = round_dir / "bench_sm120_triton_classifier.log"
        if triton_classifier_path.exists():
            benchmarks.extend(parse_runtime(read_text(triton_classifier_path), require_all=False))
        benchmarks.extend(parse_layernorm(read_text(round_dir / "bench_sm120_layernorm.log")))
        python_layernorm_path = round_dir / "bench_sm120_layernorm_python_stacks.log"
        if python_layernorm_path.exists():
            benchmarks.extend(parse_layernorm(read_text(python_layernorm_path), require_all=False))
        benchmarks.extend(parse_runtime(read_text(round_dir / "bench_sm120_runtime.log")))
        triton_runtime_path = round_dir / "bench_sm120_triton_runtime.log"
        if triton_runtime_path.exists():
            triton_runtime_text = read_text(triton_runtime_path)
            benchmarks.extend(parse_runtime(triton_runtime_text, require_all=False))
            unavailable_rows.extend(parse_runtime_unavailable_rows(triton_runtime_text))
        torch_runtime_path = round_dir / "bench_sm120_torch_runtime.log"
        if torch_runtime_path.exists():
            torch_runtime_text = read_text(torch_runtime_path)
            benchmarks.extend(parse_runtime(torch_runtime_text, require_all=False))
            unavailable_rows.extend(parse_runtime_unavailable_rows(torch_runtime_text))
        libtorch_runtime_path = round_dir / "bench_sm120_libtorch_runtime.log"
        if libtorch_runtime_path.exists():
            libtorch_runtime_text = read_text(libtorch_runtime_path)
            benchmarks.extend(parse_runtime(libtorch_runtime_text, require_all=False))
            libtorch_runtime_shape_rows = validate_libtorch_runtime_shape_coverage(benchmarks)
            libtorch_runtime_parity_rows = validate_libtorch_runtime_parity_coverage(libtorch_runtime_text)
            libtorch_runtime_supplemental_shape_rows = libtorch_runtime_supplemental_shape_coverage(benchmarks)
            libtorch_runtime_supplemental_parity_rows = libtorch_runtime_supplemental_parity_coverage(
                libtorch_runtime_text
            )
            manifest_config = manifest.get("config", {}) if isinstance(manifest, dict) else {}
            require_libtorch_supplemental_runtime = bool(
                isinstance(manifest_config, dict)
                and str(manifest_config.get("libtorch_runtime_supplemental_shapes", "")).strip()
            )
            if require_libtorch_supplemental_runtime:
                libtorch_runtime_supplemental_shape_rows = validate_libtorch_runtime_supplemental_shape_coverage(
                    benchmarks
                )
                libtorch_runtime_supplemental_parity_rows = validate_libtorch_runtime_supplemental_parity_coverage(
                    libtorch_runtime_text
                )
            expected_libtorch_runtime_route = (
                manifest_config.get("libtorch_runtime_route")
                if isinstance(manifest_config, dict)
                else None
            )
            libtorch_runtime_raw_pointer_route = validate_libtorch_runtime_raw_pointer_route(
                libtorch_runtime_text,
                expected_libtorch_runtime_route,
            )
        coverage = validate_benchmark_coverage(benchmarks)
        shape_coverage = validate_matmul_shape_coverage(benchmarks)
        runtime_shape_rows = validate_runtime_shape_coverage(benchmarks)
        provider_coverage = validate_matmul_provider_coverage(benchmarks)
        if backend_family_matrix:
            baseline_coverage = validate_baseline_provider_coverage(benchmarks, backend_family_matrix)
        if manifest_flag(manifest, "run_python_stack_benchmarks"):
            torch_coverage = validate_torch_benchmark_coverage(benchmarks, unavailable_rows)
            libtorch_matmul_rows = validate_libtorch_matmul_benchmark(round_dir, manifest, benchmarks)
            libtorch_trainer_link_probe = validate_libtorch_trainer_link_probe(round_dir, manifest)
        backend_selections = select_fastest_backends(benchmarks)
        attention_route_summaries = summarize_attention_routes(benchmarks, unavailable_rows)

    settings: dict[str, str] = {}
    val_losses: list[float] = []
    steps: list[TrainStep] = []
    total_average_ms: float | None = None
    if require_training:
        settings, val_losses, steps, total_average_ms = parse_training(read_text(round_dir / "train_gpt2cu.log"))
        validate_training(settings, val_losses, steps, total_average_ms, check_sm120_defaults)
        if forbid_checkpoints:
            validate_checkpoints(metadata)

    return RoundMetrics(
        metadata=metadata,
        manifest=manifest,
        backend_stacks=backend_stacks,
        backend_family_matrix=backend_family_matrix,
        benchmarks=benchmarks,
        coverage=coverage,
        matmul_shape_coverage=shape_coverage,
        runtime_shape_coverage=runtime_shape_rows,
        libtorch_runtime_shape_coverage=libtorch_runtime_shape_rows,
        libtorch_runtime_parity_coverage=libtorch_runtime_parity_rows,
        libtorch_runtime_supplemental_shape_coverage=libtorch_runtime_supplemental_shape_rows,
        libtorch_runtime_supplemental_parity_coverage=libtorch_runtime_supplemental_parity_rows,
        libtorch_runtime_raw_pointer_route=libtorch_runtime_raw_pointer_route,
        libtorch_trainer_link_probe=libtorch_trainer_link_probe,
        libtorch_matmul_coverage=libtorch_matmul_rows,
        matmul_provider_coverage=provider_coverage,
        baseline_provider_coverage=baseline_coverage,
        torch_benchmark_coverage=torch_coverage,
        python_stack_log_coverage=python_stack_log_coverage,
        backend_selections=backend_selections,
        attention_route_summaries=attention_route_summaries,
        unavailable_rows=unavailable_rows,
        settings=settings,
        val_losses=val_losses,
        train_steps=steps,
        total_average_ms=total_average_ms,
    )


def write_scoreboard(path: Path, metrics: RoundMetrics,
                     decisions: list[PromotionDecision] | None = None) -> None:
    lines = [
        f"# SM120 Round Metrics - {metrics.metadata['run_label']}",
        "",
        f"- artifact dir: `{metrics.metadata['artifact_dir']}`",
        f"- train output dir: `{metrics.metadata['train_output_dir']}`",
        f"- git commit: `{metrics.metadata['git_commit']}`",
        "",
    ]
    if metrics.backend_stacks:
        lines.extend(
            [
                "## Backend Stack Probe",
                "",
                "| Stack | Status | Candidate use | Next action |",
                "|---|---|---|---|",
            ]
        )
        for stack in metrics.backend_stacks:
            lines.append(
                f"| {stack.get('stack', 'unknown')} | {stack.get('status', 'unknown')} | "
                f"{stack.get('candidate_use', '')} | {stack.get('next_action', '')} |"
            )
        lines.append("")
    if metrics.backend_family_matrix:
        by_family: dict[str, dict[str, list[str]]] = {
            family: {
                "baseline": [],
                "candidate": [],
                "fallback": [],
                "missing_blocked": [],
                "not_applicable": [],
            }
            for family in OBJECTIVE_FAMILIES
        }
        for row in metrics.backend_family_matrix:
            family = str(row.get("family", ""))
            stack = str(row.get("stack", ""))
            status = str(row.get("status", "unknown"))
            if family not in by_family:
                continue
            if status in {"missing", "blocked", "unknown"}:
                by_family[family]["missing_blocked"].append(f"{stack} ({status})")
            elif status == "not_applicable":
                by_family[family]["not_applicable"].append(stack)
            else:
                by_family[family].setdefault(status, []).append(stack)
        lines.extend(
            [
                "## Backend Family-Stack Matrix",
                "",
                f"- detailed matrix: `{metrics.metadata['artifact_dir']}/backend-stacks.json`",
                "",
                "| Family | Baseline | Candidate | Fallback | Missing/blocked | Not applicable |",
                "|---|---|---|---|---|---|",
            ]
        )
        for family in OBJECTIVE_FAMILIES:
            groups = by_family[family]
            lines.append(
                f"| `{family}` | "
                f"{', '.join(groups['baseline']) or '-'} | "
                f"{', '.join(groups['candidate']) or '-'} | "
                f"{', '.join(groups['fallback']) or '-'} | "
                f"{', '.join(groups['missing_blocked']) or '-'} | "
                f"{', '.join(groups['not_applicable']) or '-'} |"
            )
        lines.append("")
    if metrics.manifest:
        config = metrics.manifest.get("config", {})
        git = metrics.manifest.get("git", {})
        if isinstance(config, dict) and isinstance(git, dict):
            lines.extend(
                [
                    "## Round Manifest",
                    "",
                    f"- manifest: `{metrics.metadata['artifact_dir']}/round-manifest.json`",
                    f"- device arch: `{config.get('device_arch', 'unknown')}`",
                    f"- build jobs: `{config.get('build_jobs', 'unknown')}`",
                    f"- changed paths: `{git.get('status_count', 'unknown')}`",
                    "",
                ]
            )
    if metrics.coverage:
        lines.extend(
            [
                "## Objective Coverage",
                "",
                "| Family | Covered |",
                "|---|---:|",
            ]
        )
        for family, covered in metrics.coverage.items():
            lines.append(f"| `{family}` | `{covered}` |")
        lines.append("")
    if metrics.matmul_shape_coverage:
        lines.extend(
            [
                "## GEMM Shape Coverage",
                "",
                "| Pass | Shape | Covered |",
                "|---|---|---:|",
            ]
        )
        for key, covered in metrics.matmul_shape_coverage.items():
            op_name, shape_name = key.split(":", 1)
            shape_label = MATMUL_SHAPE_LABELS.get(shape_name, shape_name)
            lines.append(f"| `{op_name}` | {shape_label} (`{shape_name}`) | `{covered}` |")
        lines.append("")
    if metrics.runtime_shape_coverage:
        lines.extend(
            [
                "## Runtime Shape Coverage",
                "",
                "| Kernel | Shape | Covered |",
                "|---|---|---:|",
            ]
        )
        for key, covered in metrics.runtime_shape_coverage.items():
            kernel_name, shape = key.split(":", 1)
            lines.append(f"| `{kernel_name}` | `{shape}` | `{covered}` |")
        lines.append("")
    if metrics.libtorch_runtime_shape_coverage:
        lines.extend(
            [
                "## LibTorch Runtime Shape Coverage",
                "",
                "| Kernel | Shape | Covered |",
                "|---|---|---:|",
            ]
        )
        for key, covered in metrics.libtorch_runtime_shape_coverage.items():
            kernel_name, shape = key.split(":", 1)
            lines.append(f"| `{kernel_name}` | `{shape}` | `{covered}` |")
        lines.append("")
    if metrics.libtorch_runtime_parity_coverage:
        lines.extend(
            [
                "## LibTorch Runtime Parity Coverage",
                "",
                "| Kernel | Shape | Covered |",
                "|---|---|---:|",
            ]
        )
        for key, covered in metrics.libtorch_runtime_parity_coverage.items():
            kernel_name, shape = key.split(":", 1)
            lines.append(f"| `{kernel_name}` | `{shape}` | `{covered}` |")
        lines.append("")
    if metrics.libtorch_runtime_supplemental_shape_coverage:
        lines.extend(
            [
                "## LibTorch Supplemental Runtime Shape Coverage",
                "",
                "| Kernel | Shape | Covered |",
                "|---|---|---:|",
            ]
        )
        for key, covered in metrics.libtorch_runtime_supplemental_shape_coverage.items():
            kernel_name, shape = key.split(":", 1)
            lines.append(f"| `{kernel_name}` | `{shape}` | `{covered}` |")
        lines.append("")
    if metrics.libtorch_runtime_supplemental_parity_coverage:
        lines.extend(
            [
                "## LibTorch Supplemental Runtime Parity Coverage",
                "",
                "| Kernel | Shape | Covered |",
                "|---|---|---:|",
            ]
        )
        for key, covered in metrics.libtorch_runtime_supplemental_parity_coverage.items():
            kernel_name, shape = key.split(":", 1)
            lines.append(f"| `{kernel_name}` | `{shape}` | `{covered}` |")
        lines.append("")
    if metrics.libtorch_trainer_link_probe:
        lines.extend(
            [
                "## LibTorch Trainer Link Probe",
                "",
                f"- log: `{metrics.metadata['artifact_dir']}/{LIBTORCH_TRAINER_LINK_LOG}`",
                "- status: `PASS`",
                "",
            ]
        )
    if metrics.matmul_provider_coverage:
        lines.extend(
            [
                "## GEMM Provider Coverage",
                "",
                "| Pass | Shape | Provider | Covered |",
                "|---|---|---|---:|",
            ]
        )
        for key, covered in metrics.matmul_provider_coverage.items():
            op_name, shape_name, provider = key.split(":", 2)
            shape_label = MATMUL_SHAPE_LABELS.get(shape_name, shape_name)
            provider_label = MATMUL_PROVIDER_LABELS.get(provider, provider)
            lines.append(f"| `{op_name}` | {shape_label} (`{shape_name}`) | {provider_label} | `{covered}` |")
        lines.append("")
    if metrics.baseline_provider_coverage:
        lines.extend(
            [
                "## Baseline Provider Coverage",
                "",
                "| Family | Baseline provider | Covered |",
                "|---|---|---:|",
            ]
        )
        for key, covered in metrics.baseline_provider_coverage.items():
            family, provider = key.split(":", 1)
            lines.append(f"| `{family}` | `{provider}` | `{covered}` |")
        lines.append("")
    if metrics.torch_benchmark_coverage:
        lines.extend(
            [
                "## Torch Objective Benchmark Coverage",
                "",
                "| Suite | Kernel | Shape | Covered |",
                "|---|---|---|---:|",
            ]
        )
        for key, covered in metrics.torch_benchmark_coverage.items():
            suite, kernel, shape = key.split("/", 2)
            lines.append(f"| `{suite}` | `{kernel}` | `{shape}` | `{covered}` |")
        lines.append("")
    if metrics.python_stack_log_coverage:
        lines.extend(
            [
                "## Python Stack Benchmark Logs",
                "",
                "| Log | Present |",
                "|---|---:|",
            ]
        )
        for log_name, present in metrics.python_stack_log_coverage.items():
            lines.append(f"| `{log_name}` | `{present}` |")
        lines.append("")
    if metrics.backend_selections:
        lines.extend(
            [
                "## Selected Backend Rows",
                "",
                "| Suite | Kernel | Shape | Selected stack | Time (us) | Next stack | Next time (us) | Use scope | Decision status | Decision note |",
                "|---|---|---|---|---:|---|---:|---|---|---|",
            ]
        )
        for selection in metrics.backend_selections:
            next_time = "-" if selection.next_time_us is None else f"{selection.next_time_us:.3f}"
            state = selection_decision_state(selection, decisions)
            decision_status = "-" if state["status"] is None else str(state["status"])
            lines.append(
                f"| {selection.suite} | {selection.name} | `{selection.shape}` | "
                f"{selection.selected_stack} | {selection.selected_time_us:.3f} | "
                f"{selection.next_stack} | {next_time} | {selection.use_scope} | "
                f"{decision_status} | {selection.decision_note} |"
            )
        lines.append("")
        resolved_selected = [
            selection
            for selection in metrics.backend_selections
            if selection_decision_state(selection, decisions)["status"] is not None
        ]
        if resolved_selected:
            lines.extend(
                [
                    "## Resolved Selected Backend Decisions",
                    "",
                    "| Suite | Kernel | Shape | Selected stack | Status | Decision |",
                    "|---|---|---|---|---|---|",
                ]
            )
            for selection in resolved_selected:
                state = selection_decision_state(selection, decisions)
                lines.append(
                    f"| {selection.suite} | {selection.name} | `{selection.shape}` | "
                    f"{selection.selected_stack} | {state['status']} | {state['decision']} |"
                )
            lines.append("")
        promotion_candidates = selected_promotion_candidates(metrics.backend_selections)
        active_candidates = [
            candidate
            for candidate in promotion_candidates
            if promotion_decision_state(candidate, decisions)["active"]
        ]
        if promotion_candidates:
            lines.extend(
                [
                    "## Promotion Backlog",
                    "",
                ]
            )
            if active_candidates:
                lines.extend(
                    [
                        "| Priority | Class | Suite | Kernel | Shape | Winning stack | Edge vs next | Status | Promotion gate |",
                        "|---|---|---|---|---|---|---:|---|---|",
                    ]
                )
                for candidate in active_candidates:
                    selection = candidate.selection
                    speedup = "-" if candidate.speedup_vs_next_pct is None else f"{candidate.speedup_vs_next_pct:.2f}%"
                    state = promotion_decision_state(candidate, decisions)
                    lines.append(
                        f"| {candidate.priority} | {candidate.candidate_class} | "
                        f"{selection.suite} | {selection.name} | `{selection.shape}` | "
                        f"{selection.selected_stack} | {speedup} | {state['status']} | {candidate.promotion_gate} |"
                    )
            else:
                lines.append(
                    "No active promotion candidates remain after applying "
                    "`dev/sm120_promotion_decisions.json`."
                )
            lines.append("")
            resolved_candidates = [
                candidate
                for candidate in promotion_candidates
                if not promotion_decision_state(candidate, decisions)["active"]
            ]
            if resolved_candidates:
                lines.extend(
                    [
                        "## Resolved Promotion Decisions",
                        "",
                        "| Class | Suite | Kernel | Shape | Winning stack | Status | Decision |",
                        "|---|---|---|---|---|---|---|",
                    ]
                )
                for candidate in resolved_candidates:
                    selection = candidate.selection
                    state = promotion_decision_state(candidate, decisions)
                    lines.append(
                        f"| {candidate.candidate_class} | {selection.suite} | {selection.name} | "
                        f"`{selection.shape}` | {selection.selected_stack} | {state['status']} | "
                        f"{state['decision']} |"
                )
            lines.append("")
    if metrics.attention_route_summaries:
        lines.extend(
            [
                "## Attention Route Totals",
                "",
                "| Shape | Stack | Scope | Trainer-layout | Forward (us) | Backward (us) | Total (us) | Complete | Note |",
                "|---|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in metrics.attention_route_summaries:
            forward = "-" if row.forward_us is None else f"{row.forward_us:.3f}"
            backward = "-" if row.backward_us is None else f"{row.backward_us:.3f}"
            total = "-" if row.total_us is None else f"{row.total_us:.3f}"
            note = row.unavailable_reason or ""
            lines.append(
                f"| `{row.shape}` | {row.stack} | {row.route_scope} | {row.trainer_layout} | "
                f"{forward} | {backward} | {total} | {row.complete} | {note} |"
            )
        lines.append("")
    if metrics.benchmarks:
        lines.extend(
            [
                "## Benchmark Candidates",
                "",
                "| Suite | Kernel | Shape | Stack | Time (us) |",
                "|---|---|---|---|---:|",
            ]
        )
        for metric in metrics.benchmarks:
            lines.append(
                f"| {metric.suite} | {metric.name} | `{metric.shape}` | {metric.stack} | {metric.time_us:.3f} |"
            )
        lines.append("")
    if metrics.unavailable_rows:
        lines.extend(
            [
                "## Unavailable Backend Rows",
                "",
                "| Suite | Kernel | Shape | Stack | Reason |",
                "|---|---|---|---|---|",
            ]
        )
        for row in metrics.unavailable_rows:
            lines.append(
                f"| {row.suite} | {row.name} | `{row.shape}` | {row.stack} | {row.reason} |"
            )
        lines.append("")
    if metrics.train_steps:
        final_step = metrics.train_steps[-1]
        lines.extend(
            [
                "## Training Smoke",
                "",
                f"- use_master_weights: `{metrics.settings.get('use_master_weights', 'unknown')}`",
                f"- gelu_fusion: `{metrics.settings.get('gelu_fusion', 'unknown')}`",
                f"- total average iteration time: `{metrics.total_average_ms:.3f} ms`",
                f"- final val loss: `{metrics.val_losses[-1]:.6f}`",
                f"- final step: `{final_step.step}/{final_step.total}`, loss `{final_step.loss:.6f}`, "
                f"`{final_step.ms:.2f} ms`, `{final_step.tok_s} tok/s`",
                "",
                "| Step | Loss | Norm | LR | Time (ms) | Tok/s |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for step in metrics.train_steps:
            lines.append(
                f"| {step.step} | {step.loss:.6f} | {step.norm:.4f} | {step.lr:.3g} | "
                f"{step.ms:.2f} | {step.tok_s} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def trainer_call_path_available(selection: BackendSelection) -> bool:
    return selection.use_scope in {
        "C++ benchmark route",
        "CUDA benchmark route",
        "current packed trainer route",
    }


def trainer_call_path_kind(selection: BackendSelection) -> str:
    if selection.suite == "runtime" and selection.name == "cuda_copy_d2d":
        return "profiler_runtime_benchmark_only"
    if (
        selection.suite == "runtime"
        and selection.name == "cuda_memset"
        and selection.shape == "logits_elems=3296722944"
    ):
        return "profiler_runtime_benchmark_only"
    if selection.use_scope == "C++ API prototype":
        return "libtorch_raw_pointer_prototype"
    if trainer_call_path_available(selection):
        return "trainer_or_cxx_route"
    return "operator_or_reference_prototype"


def promotion_gate(selection: BackendSelection) -> str:
    stack_key = normalized_stack_key(selection.selected_stack)
    if selection.suite == "layernorm" and "C=768" not in selection.shape:
        return "non-trainer GPT-2 LayerNorm shape; keep as operator evidence unless the trainer adds this shape"
    if selection.use_scope == "python separated-Q/K/V":
        return "refresh same-session baseline, add a packed-QKV or separated-Q/K/V trainer path, then compare against TK and run TinyStories"
    if selection.use_scope == "reference only":
        return "refresh same-session baseline, produce trainer-compatible saved state, then rerun parity and TinyStories"
    if selection.use_scope == "non-equivalent BF16-state reference":
        return "match the trainer optimizer state contract before considering promotion"
    if selection.use_scope == "non-equivalent FP32-grad prototype":
        return "match the trainer LayerNorm gradient-buffer contract before considering promotion"
    if selection.use_scope == "partial backward prototype":
        return "add dweight/dbias accumulation before considering trainer promotion"
    if selection.use_scope == "C++ API prototype":
        return "add trainer call-site integration, route parity, and TinyStories smoke before promotion"
    if stack_key == "torch":
        return "refresh same-session baseline, add an explicit libtorch/C++ or equivalent native route, then run route parity and TinyStories"
    if stack_key == "triton":
        return "refresh same-session baseline, add a trainer-callable Triton/C++ route, then run route parity and TinyStories"
    if stack_key == "cudnn":
        return "refresh same-session baseline, add an opt-in cuDNN trainer route and link gate, then run route parity and TinyStories"
    return "refresh same-session baseline, add a trainer-callable route, then run route parity and TinyStories"


def promotion_priority(speedup_vs_next_pct: float | None) -> str:
    if speedup_vs_next_pct is None:
        return "unknown"
    if speedup_vs_next_pct >= 5.0:
        return "high"
    if speedup_vs_next_pct >= 1.0:
        return "medium"
    return "low"


def promotion_candidate_class(selection: BackendSelection) -> str:
    stack_key = normalized_stack_key(selection.selected_stack)
    if selection.suite == "layernorm" and "C=768" not in selection.shape:
        return "non-trainer shape"
    if selection.use_scope == "python separated-Q/K/V":
        return "layout rewrite"
    if selection.use_scope == "reference only":
        return "reference/state gap"
    if selection.use_scope == "non-equivalent BF16-state reference":
        return "contract mismatch"
    if selection.use_scope == "non-equivalent FP32-grad prototype":
        return "contract mismatch"
    if selection.use_scope == "partial backward prototype":
        return "reference/state gap"
    if selection.use_scope == "C++ API prototype":
        return "library integration"
    if selection.use_scope in {"operator prototype", "trainer-compatible prototype"}:
        if stack_key == "torch":
            return "library integration"
        if stack_key == "triton":
            return "native/codegen integration"
        if stack_key == "cudnn":
            return "library integration"
        return "direct integration"
    return "integration"


def selected_promotion_candidates(selections: list[BackendSelection]) -> list[PromotionCandidate]:
    candidates: list[PromotionCandidate] = []
    for selection in selections:
        if trainer_call_path_available(selection):
            continue
        speedup: float | None = None
        if selection.next_time_us is not None and selection.next_time_us > 0:
            speedup = ((selection.next_time_us - selection.selected_time_us) / selection.next_time_us) * 100.0
        candidates.append(
            PromotionCandidate(
                selection=selection,
                speedup_vs_next_pct=speedup,
                priority=promotion_priority(speedup),
                candidate_class=promotion_candidate_class(selection),
                promotion_gate=promotion_gate(selection),
            )
        )
    class_order = {
        "direct integration": 0,
        "native/codegen integration": 1,
        "integration": 2,
        "library integration": 3,
        "layout rewrite": 4,
        "reference/state gap": 5,
        "non-trainer shape": 6,
        "contract mismatch": 7,
    }
    return sorted(
        candidates,
        key=lambda candidate: (
            class_order.get(candidate.candidate_class, 9),
            -1.0 if candidate.speedup_vs_next_pct is None else -candidate.speedup_vs_next_pct,
            candidate.selection.suite,
            candidate.selection.name,
            candidate.selection.shape,
        ),
    )


def find_selection_decision(
    selection: BackendSelection,
    decisions: list[PromotionDecision] | None,
) -> PromotionDecision | None:
    if not decisions:
        return None
    key = (
        normalize_match_value(selection.suite),
        normalize_match_value(selection.name),
        normalize_match_value(selection.shape),
        normalize_match_value(selection.selected_stack),
    )
    for decision in decisions:
        if (
            decision.suite,
            decision.kernel,
            decision.shape,
            decision.selected_stack,
        ) == key:
            return decision
    return None


def find_promotion_decision(
    candidate: PromotionCandidate,
    decisions: list[PromotionDecision] | None,
) -> PromotionDecision | None:
    return find_selection_decision(candidate.selection, decisions)


def selection_decision_state(
    selection: BackendSelection,
    decisions: list[PromotionDecision] | None = None,
) -> dict[str, object | None]:
    decision = find_selection_decision(selection, decisions)
    if decision is None:
        return {
            "status": None,
            "active": None,
            "decision": None,
            "evidence": [],
        }
    return {
        "status": decision.status,
        "active": decision.active,
        "decision": decision.decision,
        "evidence": list(decision.evidence),
    }


def promotion_decision_state(
    candidate: PromotionCandidate,
    decisions: list[PromotionDecision] | None = None,
) -> dict[str, object]:
    decision = find_promotion_decision(candidate, decisions)
    if decision is not None:
        return {
            "status": decision.status,
            "active": decision.active,
            "decision": decision.decision,
            "evidence": list(decision.evidence),
        }
    if candidate.candidate_class == "non-trainer shape":
        return {
            "status": "non_trainer_shape",
            "active": False,
            "decision": "Not an active trainer-promotion target for GPT-2 124M.",
            "evidence": [
                (
                    f"{candidate.selection.suite}/{candidate.selection.name} "
                    f"{candidate.selection.shape} is outside the current GPT-2 124M trainer objective rows."
                ),
            ],
        }
    if candidate.candidate_class == "contract mismatch":
        return {
            "status": "contract_mismatch",
            "active": False,
            "decision": "Not active until the candidate matches the trainer state contract.",
            "evidence": [
                (
                    f"{candidate.selection.suite}/{candidate.selection.name} "
                    f"{candidate.selection.shape} has use scope '{candidate.selection.use_scope}', "
                    "which does not match the current trainer state contract."
                ),
            ],
        }
    return {
        "status": "needs_refresh",
        "active": True,
        "decision": "Refresh the candidate and current trainer baseline in the same session before implementation.",
        "evidence": [],
    }


def selection_timing_log(selection: BackendSelection) -> str:
    stack_key = normalized_stack_key(selection.selected_stack)
    if selection.suite == "matmul":
        if stack_key == "torch":
            if selection.selected_stack == "Torch C++":
                return "bench_sm120_libtorch_matmul.log"
            return "bench_sm120_torch_matmul.log"
        if stack_key == "triton":
            return "bench_sm120_triton_matmul.log"
        return "bench_sm120_matmul.log"
    if selection.suite == "attention":
        if stack_key == "torch":
            return "bench_sm120_torch_attention.log"
        if stack_key == "cudnn":
            return "bench_sm120_cudnn_attention.log"
        if stack_key == "triton":
            return "bench_sm120_triton_attention.log"
        return "bench_sm120_attention.log"
    if selection.suite == "layernorm":
        if stack_key in {"torch", "triton"}:
            return "bench_sm120_layernorm_python_stacks.log"
        return "bench_sm120_layernorm.log"
    if selection.suite == "runtime":
        if selection.name in {"fused_classifier", "fused_classifier_loss"}:
            if stack_key == "torch":
                return "bench_sm120_torch_classifier.log"
            if stack_key == "triton":
                return "bench_sm120_triton_classifier.log"
        if stack_key == "torch":
            if selection.selected_stack == "Torch C++":
                return "bench_sm120_libtorch_runtime.log"
            return "bench_sm120_torch_runtime.log"
        if stack_key == "triton":
            return "bench_sm120_triton_runtime.log"
        return "bench_sm120_runtime.log"
    return "scoreboard-candidates.md"


def selection_correctness_logs(selection: BackendSelection) -> list[str]:
    if selection.suite == "matmul":
        if selection.selected_stack == "Torch C++":
            return ["bench_sm120_libtorch_matmul.log"]
        return ["test_matmul.log"]
    if selection.suite == "attention":
        return ["test_attention.log"]
    if selection.suite == "layernorm":
        return ["test_layernorm.log"]
    if selection.suite == "runtime":
        if selection.selected_stack == "Torch C++" and selection.name in {"cuda_memset", "cuda_copy_d2d", "gelu_forward"}:
            return ["bench_sm120_libtorch_runtime.log"]
        mapping = {
            "bias_add": "test_bias.log",
            "bias_grad_reduce": "test_bias.log",
            "gelu_forward": "test_gelu.log",
            "gelu_backward_inplace": "test_gelu.log",
            "fused_classifier": "test_fused_classifier.log",
            "fused_classifier_loss": "test_fused_classifier.log",
            "adamw_update": "test_adamw.log",
            "adamw_update_bf16_state": "test_adamw.log",
            "global_norm_squared": "test_global_norm.log",
            "encoder_forward": "test_encoder.log",
        }
        log_name = mapping.get(selection.name)
        if log_name:
            return [log_name]
    return []


def artifact_path(metrics: RoundMetrics | None, log_name: str) -> str:
    if metrics is None:
        return log_name
    artifact_dir = metrics.metadata.get("artifact_dir", "")
    if not artifact_dir:
        return log_name
    return str(Path(artifact_dir) / log_name)


ROW_RUN_CONFIG_KEYS = (
    "device_arch",
    "build_jobs",
    "no_multi_gpu",
    "no_use_mpi",
    "run_correctness",
    "run_benchmarks",
    "run_python_stack_benchmarks",
    "libtorch_runtime_route",
    "run_libtorch_matmul_benchmarks",
    "libtorch_matmul_shapes",
    "run_stack_probe",
    "run_training",
    "max_steps",
    "keep_checkpoints",
)


def row_run_config(metrics: RoundMetrics | None) -> dict[str, str]:
    if metrics is None:
        return {}
    config = metrics.manifest.get("config", {}) if metrics.manifest else {}
    if not isinstance(config, dict):
        return {}
    return {
        key: str(config[key])
        for key in ROW_RUN_CONFIG_KEYS
        if key in config
    }


def selection_to_dict(
    selection: BackendSelection,
    decisions: list[PromotionDecision] | None = None,
    metrics: RoundMetrics | None = None,
) -> dict[str, object]:
    timing_log = selection_timing_log(selection)
    correctness_logs = selection_correctness_logs(selection)
    row: dict[str, object] = {
        "suite": selection.suite,
        "kernel": selection.name,
        "shape": selection.shape,
        "selected_stack": selection.selected_stack,
        "selected_time_us": selection.selected_time_us,
        "next_stack": None if selection.next_stack == "-" else selection.next_stack,
        "next_time_us": selection.next_time_us,
        "use_scope": selection.use_scope,
        "decision_note": selection.decision_note,
        "trainer_call_path_available": trainer_call_path_available(selection),
        "trainer_call_path_kind": trainer_call_path_kind(selection),
        "source_run_label": metrics.metadata.get("run_label", "unknown") if metrics else "unknown",
        "source_artifact_dir": metrics.metadata.get("artifact_dir", "") if metrics else "",
        "source_git_commit": metrics.metadata.get("git_commit", "unknown") if metrics else "unknown",
        "source_run_config": row_run_config(metrics),
        "timing_log": timing_log,
        "timing_log_path": artifact_path(metrics, timing_log),
        "config_artifact": "round-manifest.json",
        "config_artifact_path": artifact_path(metrics, "round-manifest.json"),
        "stack_probe_artifact": "backend-stacks.json",
        "stack_probe_artifact_path": artifact_path(metrics, "backend-stacks.json"),
        "correctness_logs": correctness_logs,
        "correctness_log_paths": [artifact_path(metrics, log_name) for log_name in correctness_logs],
    }
    if not correctness_logs:
        row["correctness_evidence_note"] = (
            "No dedicated route-level correctness log is required for this runtime primitive; "
            "the timing log and CUDA runtime API checks are the available evidence."
        )
    state = selection_decision_state(selection, decisions)
    if state["status"] is not None:
        row.update(
            {
                f"decision_{key}": value
                for key, value in state.items()
            }
        )
    return row


def promotion_candidate_to_dict(
    candidate: PromotionCandidate,
    decisions: list[PromotionDecision] | None = None,
    metrics: RoundMetrics | None = None,
) -> dict[str, object]:
    row = selection_to_dict(candidate.selection, decisions, metrics)
    decision_state = promotion_decision_state(candidate, decisions)
    row["speedup_vs_next_pct"] = candidate.speedup_vs_next_pct
    row["priority"] = candidate.priority
    row["candidate_class"] = candidate.candidate_class
    row["promotion_gate"] = candidate.promotion_gate
    if decision_state["status"] == "profiler_only_runtime_row":
        row["candidate_class"] = "profiler-only runtime row"
        row["decision_note"] = "Keep as benchmark evidence only; no current trainer call-site maps to this row."
        row["promotion_gate"] = "none; profiler-only runtime row with no current trainer call-site"
    if decision_state["status"] == "rejected_x10_trainer_route":
        row["candidate_class"] = "trainer route rejected"
        row["decision_note"] = "Opt-in LibTorch trainer route passed correctness/smoke but x10 stability rejected promotion."
        row["promotion_gate"] = "none; opt-in trainer route passed but x10 stability rejected promotion"
    row.update(
        {
            f"decision_{key}": value
            for key, value in decision_state.items()
        }
    )
    return row


def attention_route_to_dict(row: AttentionRouteSummary) -> dict[str, object]:
    return {
        "shape": row.shape,
        "stack": row.stack,
        "route_scope": row.route_scope,
        "trainer_layout": row.trainer_layout,
        "forward_us": row.forward_us,
        "backward_us": row.backward_us,
        "total_us": row.total_us,
        "complete": row.complete,
        "unavailable_reason": row.unavailable_reason,
    }


def write_selected_backends(path: Path, metrics: RoundMetrics,
                            decisions: list[PromotionDecision] | None = None) -> None:
    promotion_candidates = selected_promotion_candidates(metrics.backend_selections)
    active_candidates = [
        candidate
        for candidate in promotion_candidates
        if promotion_decision_state(candidate, decisions)["active"]
    ]
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_label": metrics.metadata.get("run_label", "unknown"),
        "artifact_dir": metrics.metadata.get("artifact_dir", ""),
        "git_commit": metrics.metadata.get("git_commit", "unknown"),
        "benchmark_row_count": len(metrics.benchmarks),
        "torch_benchmark_row_count": sum(1 for covered in metrics.torch_benchmark_coverage.values() if covered),
        "torch_benchmark_coverage": metrics.torch_benchmark_coverage,
        "libtorch_runtime_shape_coverage": metrics.libtorch_runtime_shape_coverage,
        "libtorch_runtime_parity_coverage": metrics.libtorch_runtime_parity_coverage,
        "libtorch_runtime_supplemental_shape_coverage": metrics.libtorch_runtime_supplemental_shape_coverage,
        "libtorch_runtime_supplemental_parity_coverage": metrics.libtorch_runtime_supplemental_parity_coverage,
        "libtorch_runtime_raw_pointer_route": metrics.libtorch_runtime_raw_pointer_route,
        "libtorch_trainer_link_probe": metrics.libtorch_trainer_link_probe,
        "libtorch_matmul_coverage": metrics.libtorch_matmul_coverage,
        "selection_policy": (
            "Fastest observed row per exact suite/kernel/shape. Rows without an existing "
            "trainer call path remain reference/operator prototypes until route-specific "
            "correctness and TinyStories smoke evidence proves promotion. Repo-local "
            "decisions mark refreshed rejects and structurally non-promotable rows inactive "
            "without deleting their evidence."
        ),
        "selected_backend_rows": [
            selection_to_dict(selection, decisions, metrics)
            for selection in metrics.backend_selections
        ],
        "attention_route_rows": [
            attention_route_to_dict(row)
            for row in metrics.attention_route_summaries
        ],
        "promotion_candidates": [
            promotion_candidate_to_dict(candidate, decisions, metrics)
            for candidate in promotion_candidates
        ],
        "active_promotion_candidates": [
            promotion_candidate_to_dict(candidate, decisions, metrics)
            for candidate in active_candidates
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_promotion_candidates(path: Path, metrics: RoundMetrics,
                               decisions: list[PromotionDecision] | None = None) -> None:
    candidates = selected_promotion_candidates(metrics.backend_selections)
    active_candidates = [
        candidate
        for candidate in candidates
        if promotion_decision_state(candidate, decisions)["active"]
    ]
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_label": metrics.metadata.get("run_label", "unknown"),
        "artifact_dir": metrics.metadata.get("artifact_dir", ""),
        "git_commit": metrics.metadata.get("git_commit", "unknown"),
        "promotion_policy": (
            "Rows are selected benchmark winners that do not yet have a trainer call path. "
            "They are prioritized by speedup over the next observed stack when that comparison exists. "
            "Native/direct and codegen rows are listed before library integrations, layout rewrites, "
            "and reference/state gaps. "
            "Before implementation, rerun the selected row and current baseline in the same session."
        ),
        "decision_policy": (
            "Repo-local decisions in dev/sm120_promotion_decisions.json mark refreshed "
            "or structurally non-promotable rows as inactive without deleting their evidence."
        ),
        "promotion_candidates": [
            promotion_candidate_to_dict(candidate, decisions, metrics)
            for candidate in candidates
        ],
        "active_promotion_candidates": [
            promotion_candidate_to_dict(candidate, decisions, metrics)
            for candidate in active_candidates
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def synthetic_matmul_log(*, include_lmhead: bool = True, include_cublas: bool = True,
                         omit_cublas_shape: str | None = None) -> str:
    rows = [
        "Device: NVIDIA GeForce RTX 5090 (sm_120)",
        "",
        "qkv          M=65536 N=2304 K=768 bias=1 gelu=0",
        "  fwd      TK   1100.00 us | cuBLASLt   1000.00 us | cuBLAS   1010.00 us | TK/cuBLASLt 1.10x",
        "  dInp   TK   1200.00 us | cuBLASLt   1050.00 us | cuBLAS   1060.00 us | TK/cuBLASLt 1.14x",
        "  dW     TK   1500.00 us | cuBLASLt   1100.00 us | cuBLAS   1110.00 us | TK/cuBLASLt 1.36x",
        "  dW+accum TK   1520.00 us | cuBLASLt   1120.00 us | cuBLAS   1130.00 us | TK/cuBLASLt 1.36x",
        "",
        "attproj      M=65536 N=768 K=768 bias=1 gelu=0",
        "  fwd      TK    390.00 us | cuBLASLt    330.00 us | cuBLAS    335.00 us | TK/cuBLASLt 1.18x",
        "  dInp   TK    410.00 us | cuBLASLt    397.00 us | cuBLAS    400.00 us | TK/cuBLASLt 1.03x",
        "  dW     TK    580.00 us | cuBLASLt    327.00 us | cuBLAS    333.00 us | TK/cuBLASLt 1.77x",
        "  dW+accum TK    590.00 us | cuBLASLt    340.00 us | cuBLAS    344.00 us | TK/cuBLASLt 1.74x",
        "",
        "fc           M=65536 N=3072 K=768 bias=1 gelu=1",
        "  fwd+GeLU TK fused   1400.00 us | TK explicit   1600.00 us | cuBLASLt   1300.00 us | cuBLAS explicit   1320.00 us | explicit/cuBLASLt 1.23x",
        "  dInp   TK   1450.00 us | cuBLASLt   1350.00 us | cuBLAS   1360.00 us | TK/cuBLASLt 1.07x",
        "  dW     TK   1700.00 us | cuBLASLt   1370.00 us | cuBLAS   1380.00 us | TK/cuBLASLt 1.24x",
        "  dW+accum TK   1720.00 us | cuBLASLt   1390.00 us | cuBLAS   1400.00 us | TK/cuBLASLt 1.24x",
        "",
        "fcproj       M=65536 N=768 K=3072 bias=1 gelu=0",
        "  fwd      TK   1410.00 us | cuBLASLt   1438.00 us | cuBLAS   1440.00 us | TK/cuBLASLt 0.98x",
        "  dInp   TK   1469.00 us | cuBLASLt   1438.00 us | cuBLAS   1445.00 us | TK/cuBLASLt 1.02x",
        "  dInp+dGeLU TK   1460.00 us | cuBLASLt fused   1360.00 us | cuBLASLt explicit   1500.00 us | cuBLAS explicit   1510.00 us | explicit/fused 1.10x",
        "  dW     TK   1766.00 us | cuBLASLt   1309.00 us | cuBLAS   1320.00 us | TK/cuBLASLt 1.35x",
        "  dW+accum TK   1780.00 us | cuBLASLt   1330.00 us | cuBLAS   1340.00 us | TK/cuBLASLt 1.34x",
        "",
    ]
    if include_lmhead:
        rows.extend(
            [
                "lmhead       M=65536 N=50304 K=768 bias=0 gelu=0",
                "  fwd      TK  23500.00 us | cuBLASLt  20910.00 us | cuBLAS  21010.00 us | TK/cuBLASLt 1.12x",
                "  dInp   TK  23760.00 us | cuBLASLt  21130.00 us | cuBLAS  21230.00 us | TK/cuBLASLt 1.12x",
                "  dW     TK  26020.00 us | cuBLASLt  21002.00 us | cuBLAS  21102.00 us | TK/cuBLASLt 1.24x",
                "  dW+accum TK  26100.00 us | cuBLASLt  21110.00 us | cuBLAS  21210.00 us | TK/cuBLASLt 1.24x",
                "",
            ]
        )
    filtered_rows: list[str] = []
    current_shape = ""
    for row in rows:
        shape_match = MATMUL_SHAPE_RE.match(row.strip())
        if shape_match:
            current_shape = shape_match.group("name")
        if not include_cublas or (omit_cublas_shape is not None and current_shape == omit_cublas_shape):
            row = re.sub(r"\s+\|\s+cuBLAS(?: explicit)?\s+" + FLOAT_RE + r"\s+us", "", row)
        filtered_rows.append(row)
    return "\n".join(filtered_rows) + "\n"


def synthetic_family_matrix(*, omit_family: str | None = None, omit_stack: str | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    missing_stacks = {"cuDNN", "Triton", "CuTeDSL"}
    not_applicable = {
        "gemm": {"cuDNN"},
        "attention": {"cuBLAS", "cuBLASLt", "CuTeDSL"},
        "layernorm": {"cuBLAS", "cuBLASLt", "cuDNN", "CuTeDSL"},
        "runtime": {"ThunderKittens 2.0", "cuBLAS", "cuBLASLt", "cuDNN", "CuTeDSL"},
    }
    gemm_families = {
        "gemm_forward",
        "gemm_forward_fused_gelu",
        "gemm_backward_dinput",
        "gemm_backward_dinput_fused_dgelu",
        "gemm_backward_dweight",
        "gemm_backward_dweight_accum",
    }
    attention_families = {"attention_forward", "attention_backward"}
    layernorm_families = {"layernorm_forward", "layernorm_fused_residual_forward", "layernorm_backward"}
    for family in OBJECTIVE_FAMILIES:
        for stack in OBJECTIVE_STACKS:
            if family == omit_family and stack == omit_stack:
                continue
            if family in gemm_families:
                bucket = "gemm"
                baseline_stack = "cuBLASLt"
            elif family in attention_families:
                bucket = "attention"
                baseline_stack = "ThunderKittens 2.0"
            elif family in layernorm_families:
                bucket = "layernorm"
                baseline_stack = "Plain CUDA"
            else:
                bucket = "runtime"
                baseline_stack = "Plain CUDA"
            if stack in not_applicable[bucket]:
                status = "not_applicable"
                reason = f"{stack} is not a scoped provider for {family}"
                next_action = "none"
            elif stack in missing_stacks:
                status = "missing"
                reason = f"{stack} is missing; intended candidate for {family}"
                next_action = f"install {stack} before timing"
            elif stack == baseline_stack:
                status = "baseline"
                reason = f"{stack} is the current baseline for {family}"
                next_action = "compare focused candidates before changing defaults"
            elif stack == "Plain CUDA":
                status = "fallback"
                reason = f"{stack} is the fallback for {family}"
                next_action = "keep as safety path unless it wins focused timing"
            else:
                status = "candidate"
                reason = f"{stack} is a candidate provider for {family}"
                next_action = "run parity and benchmark before promotion"
            rows.append(
                {
                    "family": family,
                    "stack": stack,
                    "status": status,
                    "reason": reason,
                    "next_action": next_action,
                }
            )
    return rows


def synthetic_backend_stacks(*, include_cuteds: bool = True,
                             omit_matrix_family: str | None = None,
                             omit_matrix_stack: str | None = None) -> dict[str, object]:
    rows = [
        {
            "stack": "ThunderKittens 2.0",
            "status": "available",
            "evidence": ["synthetic TK_ROOT"],
            "candidate_use": "native TK kernels",
            "next_action": "benchmark by shape",
        },
        {
            "stack": "Plain CUDA",
            "status": "available",
            "evidence": ["synthetic nvcc"],
            "candidate_use": "plain CUDA baselines",
            "next_action": "run on RTX 5090",
        },
        {
            "stack": "GPU runtime",
            "status": "available",
            "evidence": ["synthetic RTX 5090"],
            "candidate_use": "runtime timing and correctness execution",
            "next_action": "capture fresh timings",
        },
        {
            "stack": "cuBLAS",
            "status": "available",
            "evidence": ["synthetic cublas_v2.h", "synthetic libcublas"],
            "candidate_use": "baseline GEMM comparison",
            "next_action": "add explicit benchmark rows before promotion",
        },
        {
            "stack": "cuBLASLt",
            "status": "available",
            "evidence": ["synthetic cublasLt.h", "synthetic libcublasLt"],
            "candidate_use": "current SM120 GEMM baseline",
            "next_action": "keep rows shape-specific",
        },
        {
            "stack": "cuDNN",
            "status": "missing",
            "evidence": ["synthetic cuDNN missing"],
            "candidate_use": "attention alternatives",
            "next_action": "install/prototype before benchmarking",
        },
        {
            "stack": "Triton",
            "status": "missing",
            "evidence": ["synthetic Triton missing"],
            "candidate_use": "optional fused candidates",
            "next_action": "install before benchmarking",
        },
        {
            "stack": "Torch",
            "status": "available",
            "evidence": ["synthetic torch"],
            "candidate_use": "PyTorch operator kernels",
            "next_action": "benchmark exact family shapes before promotion",
        },
    ]
    if include_cuteds:
        rows.append(
            {
                "stack": "CuTeDSL",
                "status": "missing",
                "evidence": ["synthetic CuTeDSL missing"],
                "candidate_use": "Blackwell GEMM and fused epilogue candidates",
                "next_action": "install before benchmarking",
            }
        )
    return {
        "schema_version": 1,
        "objective_stacks": list(OBJECTIVE_STACKS),
        "objective_families": list(OBJECTIVE_FAMILIES),
        "stacks": rows,
        "family_matrix": synthetic_family_matrix(
            omit_family=omit_matrix_family,
            omit_stack=omit_matrix_stack,
        ),
    }


def synthetic_binary_records(*, omit: str | None = None) -> list[dict[str, object]]:
    return [
        {
            "path": binary,
            "exists": True,
            "size_bytes": 1,
            "sha256": "0" * 64,
        }
        for binary in EXPECTED_MANIFEST_BINARIES
        if binary != omit
    ]


def set_synthetic_manifest_flag(round_dir: Path, key: str, value: str) -> None:
    manifest_path = round_dir / "round-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = manifest.setdefault("config", {})
    if not isinstance(config, dict):
        raise RuntimeError("synthetic manifest config is not an object")
    config[key] = value
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def write_synthetic_round(round_dir: Path) -> Path:
    train_dir = round_dir / "train-out"
    train_dir.mkdir(parents=True)
    (round_dir / "summary.md").write_text(
        "\n".join(
            [
                "# SM120 Optimization Round",
                "",
                "- run label: `self_test`",
                f"- artifact dir: `{round_dir}`",
                f"- train output dir: `{train_dir}`",
                "- max steps: `3`",
                "- git commit: `abcdef0`",
                "- working tree: `0` changed paths",
                "",
            ]
        )
    )
    (round_dir / "build.log").write_text("nvcc build complete\n")
    (round_dir / "backend-stacks.json").write_text(
        json.dumps(synthetic_backend_stacks(), indent=2)
        + "\n"
    )
    (round_dir / "round-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config": {
                    "run_label": "self_test",
                    "artifact_dir": str(round_dir),
                    "train_out_dir": str(train_dir),
                    "device_arch": "SM120",
                    "build_jobs": "4",
                    "run_python_stack_benchmarks": "1",
                    "cudnn_packed_backward_route": "saved-forward",
                    "libtorch_runtime_route": "cxx-api-raw-pointer",
                    "libtorch_runtime_supplemental_shapes": "gelu_forward",
                    "run_libtorch_trainer_link_probe": "1",
                    "run_libtorch_matmul_benchmarks": "1",
                    "libtorch_matmul_shapes": "qkv attproj fc fcproj lmhead",
                },
                "git": {
                    "short_commit": "abcdef0",
                    "status_count": 0,
                },
                "binaries": synthetic_binary_records(),
            },
            indent=2,
        )
        + "\n"
    )
    (round_dir / "test_matmul.log").write_text("8/8 passed\ntest_matmul smoke OK\n")
    (round_dir / "test_attention.log").write_text("forward PASS\nbackward PASS\ntest_attention smoke OK\n")
    (round_dir / "test_layernorm.log").write_text("forward PASS\nbackward PASS\ntest_layernorm smoke OK\n")
    (round_dir / "test_bias.log").write_text("bias add PASS\nbias grad PASS\ntest_bias smoke OK\n")
    (round_dir / "test_gelu.log").write_text("forward PASS\nbackward PASS\ntest_gelu smoke OK\n")
    (round_dir / "test_fused_classifier.log").write_text(
        "loss PASS\ndlogits PASS\ntest_fused_classifier smoke OK\n"
    )
    (round_dir / "test_encoder.log").write_text("forward PASS\ntest_encoder smoke OK\n")
    (round_dir / "test_adamw.log").write_text("master PASS\nm PASS\nv PASS\ntest_adamw smoke OK\n")
    (round_dir / "test_global_norm.log").write_text("relative diff PASS\ntest_global_norm smoke OK\n")
    (round_dir / "bench_sm120_matmul.log").write_text(synthetic_matmul_log())
    (round_dir / "bench_sm120_torch_matmul.log").write_text(
        "\n".join(
            [
                "qkv          M=65536 N=2304 K=768 bias=1 gelu=0",
                "  fwd        Torch   1200.00 us",
                "  dInp       Torch   1300.00 us",
                "  dW         Torch   1400.00 us",
                "  dW+accum   Torch   1420.00 us",
                "",
                "attproj      M=65536 N=768 K=768 bias=1 gelu=0",
                "  fwd        Torch    420.00 us",
                "  dInp       Torch    430.00 us",
                "  dW         Torch    440.00 us",
                "  dW+accum   Torch    450.00 us",
                "",
                "fc           M=65536 N=3072 K=768 bias=1 gelu=1",
                "  fwd+GeLU   Torch   1800.00 us",
                "  dInp       Torch   1500.00 us",
                "  dW         Torch   1510.00 us",
                "  dW+accum   Torch   1520.00 us",
                "",
                "fcproj       M=65536 N=768 K=3072 bias=1 gelu=0",
                "  fwd        Torch   1540.00 us",
                "  dInp       Torch   1550.00 us",
                "  dInp+dGeLU Torch   2100.00 us",
                "  dW         Torch   1560.00 us",
                "  dW+accum   Torch   1570.00 us",
                "",
                "lmhead       M=65536 N=50304 K=768 bias=0 gelu=0",
                "  fwd        Torch  22500.00 us",
                "  dInp       Torch  22600.00 us",
                "  dW         Torch  22700.00 us",
                "  dW+accum   Torch  22800.00 us",
                "",
            ]
        )
    )
    (round_dir / "bench_sm120_libtorch_matmul.log").write_text(
        "\n".join(
            [
                "LibTorch matmul device: NVIDIA GeForce RTX 5090; capability=sm_120",
                LIBTORCH_MATMUL_ROUTE,
                "",
                "qkv          M=65536 N=2304 K=768 bias=1 gelu=0",
                "LibTorch matmul parity dW qkv: PASS max_abs=0.000000",
                "LibTorch matmul parity dW+accum qkv: PASS max_abs=0.000000",
                "  dW       Torch C++   1020.00 us",
                "  dW+accum Torch C++   1030.00 us",
                "",
                "attproj      M=65536 N=768 K=768 bias=1 gelu=0",
                "LibTorch matmul parity dW attproj: PASS max_abs=0.000000",
                "LibTorch matmul parity dW+accum attproj: PASS max_abs=0.000000",
                "  dW       Torch C++    340.00 us",
                "  dW+accum Torch C++    345.00 us",
                "",
                "fc           M=65536 N=3072 K=768 bias=1 gelu=1",
                "LibTorch matmul parity dW fc: PASS max_abs=0.000000",
                "LibTorch matmul parity dW+accum fc: PASS max_abs=0.000000",
                "  dW       Torch C++   1490.00 us",
                "  dW+accum Torch C++   1500.00 us",
                "",
                "fcproj       M=65536 N=768 K=3072 bias=1 gelu=0",
                "LibTorch matmul parity dW fcproj: PASS max_abs=0.000000",
                "LibTorch matmul parity dW+accum fcproj: PASS max_abs=0.000000",
                "  dW       Torch C++   1380.00 us",
                "  dW+accum Torch C++   1410.00 us",
                "",
                "lmhead       M=65536 N=50304 K=768 bias=0 gelu=0",
                "LibTorch matmul parity dW lmhead: PASS max_abs=0.000000",
                "LibTorch matmul parity dW+accum lmhead: PASS max_abs=0.000000",
                "  dW       Torch C++  22000.00 us",
                "  dW+accum Torch C++  22900.00 us",
                "",
            ]
        )
    )
    (round_dir / "bench_sm120_cutedsl_matmul.log").write_text(
        "\n".join(
            [
                "CuTeDSL package: cutlass 4.5.1",
                "CuTeDSL CUDA available: True",
                "CuTeDSL device: NVIDIA GeForce RTX 5090; capability=sm_120",
                "qkv M=65536 N=2304 K=768 bias=1 gelu=0                   | CuTeDSL     | unavailable: local CuTeDSL BF16 grouped-GEMM path rejects sm_120a",
                "attproj M=65536 N=768 K=768 bias=1 gelu=0                | CuTeDSL     | unavailable: local CuTeDSL BF16 grouped-GEMM path rejects sm_120a",
                "fc M=65536 N=3072 K=768 bias=1 gelu=1                    | CuTeDSL     | unavailable: local CuTeDSL BF16 grouped-GEMM path rejects sm_120a",
                "fcproj M=65536 N=768 K=3072 bias=1 gelu=0                | CuTeDSL     | unavailable: local CuTeDSL BF16 grouped-GEMM path rejects sm_120a",
                "lmhead M=65536 N=50304 K=768 bias=0 gelu=0               | CuTeDSL     | unavailable: local CuTeDSL BF16 grouped-GEMM path rejects sm_120a",
                "",
            ]
        )
    )
    (round_dir / "bench_sm120_triton_matmul.log").write_text(
        "\n".join(
            [
                "Triton matmul device: NVIDIA GeForce RTX 5090; capability=sm_120",
                "",
                "qkv          M=65536 N=2304 K=768 bias=1 gelu=0",
                "  fwd        Triton   1500.00 us (diff=0.125000)",
                "  dInp       Triton   1600.00 us (diff=0.125000)",
                "  dW         Triton   1700.00 us (diff=0.125000)",
                "  dW+accum   Triton   1720.00 us (diff=0.125000)",
                "",
                "fc           M=65536 N=3072 K=768 bias=1 gelu=1",
                "  fwd+GeLU   Triton   1900.00 us (diff=0.125000)",
                "",
                "fcproj       M=65536 N=768 K=3072 bias=1 gelu=0",
                "  dInp+dGeLU Triton   2200.00 us (diff=0.125000)",
                "",
            ]
        )
    )
    (round_dir / "bench_sm120_attention.log").write_text(
        "\n".join(
            [
                "Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 771.000 us",
                "Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2679.000 us",
                "",
            ]
        )
    )
    (round_dir / "bench_sm120_torch_attention.log").write_text(
        "\n".join(
            [
                "Torch Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 572.000 us",
                "Torch Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2203.000 us",
                "TorchPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 1132.000 us",
                "TorchPacked Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 4093.000 us",
                "",
            ]
        )
    )
    (round_dir / "bench_sm120_cudnn_attention.log").write_text(
        "\n".join(
            [
                "cuDNN Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 668.000 us (max_diff=0.003906)",
                "cuDNN Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2385.000 us",
                "cuDNNPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 781.000 us",
                CUDNN_PACKED_BACKWARD_ROUTE,
                "cuDNNPacked Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 3468.000 us",
                "",
            ]
        )
    )
    (round_dir / "bench_sm120_triton_attention.log").write_text(
        "\n".join(
            [
                "Triton Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 1400.000 us (diff=0.031250)",
                "TritonPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 1500.000 us (diff=0.031250)",
                "B=64 T=1024 C=768 NH=12 HS=64     | Triton       | unavailable: attention backward is not implemented in this Triton prototype",
                "B=64 T=1024 C=768 NH=12 HS=64     | TritonPacked | unavailable: packed attention backward is not implemented in this Triton prototype",
                "",
            ]
        )
    )
    (round_dir / "bench_sm120_torch_classifier.log").write_text(
        "\n".join(
            [
                "fused_classifier_loss          | B=64 T=1024 V=50257 P=50304 | Torch        | 12000.000 us",
                "fused_classifier               | B=64 T=1024 V=50257 P=50304 | Torch        | unavailable: CUDA OOM at full GPT-2 padded-logits shape",
                "",
            ]
        )
    )
    (round_dir / "bench_sm120_triton_classifier.log").write_text(
        "\n".join(
            [
                "fused_classifier_loss          | B=64 T=1024 V=50257 P=50304 | Triton       | 14000.000 us",
                "fused_classifier               | B=64 T=1024 V=50257 P=50304 | Triton       | 26000.000 us",
                "",
            ]
        )
    )
    (round_dir / "bench_sm120_layernorm.log").write_text(
        "\n".join(
            [
                "LayerNorm Forward (N=65536, C=768): 137.000 us",
                "LayerNorm FusedResidualForward (N=65536, C=768): 184.000 us",
                "LayerNorm Backward (N=65536, C=768): 288.000 us",
                "",
            ]
        )
    )
    (round_dir / "bench_sm120_layernorm_python_stacks.log").write_text(
        "\n".join(
            [
                "Triton LayerNorm Forward (N=65536, C=768): 175.000 us",
                "Torch LayerNorm ForwardNative (N=65536, C=768): 153.000 us (no saved mean/rstd)",
                "Torch LayerNorm ForwardWithStats (N=65536, C=768): 2223.000 us",
                "Triton LayerNorm BackwardDInput (N=65536, C=768): 120.000 us (dweight/dbias not produced)",
                "Torch LayerNorm BackwardDInputNative (N=65536, C=768): 180.000 us (dweight/dbias not produced)",
                "Torch LayerNorm BackwardNative (N=65536, C=768): 360.000 us",
                "Triton LayerNorm BackwardAtomicFP32 (N=65536, C=768): 640.000 us (FP32 dweight/dbias)",
                "Triton LayerNorm FusedResidualForward (N=65536, C=768): 311.000 us",
                "Torch LayerNorm FusedResidualForwardNative (N=65536, C=768): 339.000 us (no saved mean/rstd)",
                "Torch LayerNorm FusedResidualForwardWithStats (N=65536, C=768): 3226.000 us",
                "",
            ]
        )
    )
    runtime_lines = [
        "SM120 GPT-2 runtime kernel benchmark on NVIDIA GeForce RTX 5090",
        "Kernel                         | Shape                        | Stack        |         Time",
    ]
    for name in sorted(EXPECTED_RUNTIME_KERNELS):
        stack = "CUDA runtime" if name.startswith("cuda_") else "CUDA"
        shapes = tuple(
            shape
            for kernel_name, required_shapes in RUNTIME_SHAPE_REQUIREMENTS
            if kernel_name == name
            for shape in required_shapes
        ) or ("BT=65536 C=768",)
        for shape in shapes:
            runtime_lines.append(f"{name:<30} | {shape:<28} | {stack:<12} |   42.000 us")
            if name.startswith("cuda_"):
                runtime_lines.append(f"{name:<30} | {shape:<28} | CUDA kernel  |   84.000 us")
    (round_dir / "bench_sm120_runtime.log").write_text("\n".join(runtime_lines) + "\n")
    (round_dir / "bench_sm120_torch_runtime.log").write_text(
        "\n".join(
            [
                "bias_add                       | BT=65536 OC=768              | Torch        |   138.000 us",
                "bias_add                       | BT=65536 OC=3072             | Torch        |   539.000 us",
                "gelu_forward                   | BT=65536 C=3072              | Torch        |   547.000 us",
                "gelu_backward_inplace          | BT=65536 C=3072              | Torch        | 27328.000 us",
                "bias_grad_reduce               | BT=65536 OC=768              | Torch        |   325.000 us",
                "bias_grad_reduce               | BT=65536 OC=2304             | Torch        |  1018.000 us",
                "bias_grad_reduce               | BT=65536 OC=3072             | Torch        |  1359.000 us",
                "global_norm_squared            | params=124475904             | Torch        |  2367.000 us",
                "adamw_update_bf16_state        | params=124475904 no-master   | Torch        |  1225.000 us",
                "adamw_update                   | params=124475904 no-master fp32-state | Torch        |  7453.000 us",
                "encoder_forward                | B=64 T=1024 C=768            | Torch        |   203.000 us",
                "cuda_memset                    | hidden_elems=50331648        | Torch        |    64.000 us",
                "cuda_memset                    | grad_elems=124475904         | Torch        |   157.000 us",
                "cuda_copy_d2d                  | hidden_elems=50331648        | Torch        |   134.000 us",
                "cuda_memset                    | logits_elems=3296722944      | Torch        |  4397.000 us",
                "cuda_copy_d2d                  | logits_elems=3296722944      | Torch        |  9310.000 us",
                "",
            ]
        )
    )
    (round_dir / "bench_sm120_libtorch_runtime.log").write_text(
        "\n".join(
            [
                "LibTorch runtime device: NVIDIA GeForce RTX 5090; capability=sm_120",
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
                "",
            ]
        )
    )
    (round_dir / LIBTORCH_TRAINER_LINK_LOG).write_text(
        "\n".join(
            [
                "LibTorch trainer link route: standalone executable without torch_python",
                "LibTorch trainer link compile: PASS /tmp/torch_extensions/llmk_libtorch_trainer_link_probe/llmk_libtorch_trainer_link_probe",
                "LibTorch trainer link runtime: PASS zero/copy from_blob executable",
                "LibTorch trainer link probe: PASS",
                "",
            ]
        )
    )
    (round_dir / "bench_sm120_triton_runtime.log").write_text(
        "\n".join(
            [
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
                "cuda_memset                    | logits_elems=3296722944      | Triton       | unavailable: not implemented in this Triton runtime prototype",
                "cuda_copy_d2d                  | hidden_elems=50331648        | Triton       | unavailable: not implemented in this Triton runtime prototype",
                "cuda_copy_d2d                  | logits_elems=3296722944      | Triton       | unavailable: not implemented in this Triton runtime prototype",
                "",
            ]
        )
    )
    (round_dir / "train_gpt2cu.log").write_text(
        "\n".join(
            [
                "| use_master_weights    | disabled                                           |",
                "| gelu_fusion           | 1                                                  |",
                "| precision             | BF16                                               |",
                "val loss 11.033152",
                "step    1/3 | loss 11.032356 (+nanz)| norm 22.1413 (+nanz)| lr 8.57e-07 | 2527.19 ms | 39.8% bf16 MFU | 207459 tok/s",
                "step    2/3 | loss 10.958524 (+nanz)| norm 22.0967 (+nanz)| lr 1.71e-06 | 2514.05 ms | 40.0% bf16 MFU | 208543 tok/s",
                "step    3/3 | loss 10.811322 (+nanz)| norm 21.1251 (+nanz)| lr 2.57e-06 | 2518.69 ms | 39.9% bf16 MFU | 208346 tok/s",
                "val loss 10.188138",
                "total average iteration time: 2521.277666 ms",
                "",
            ]
        )
    )
    return train_dir


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="llmkittens_sm120_round_") as tmp:
        round_dir = Path(tmp) / "round"
        round_dir.mkdir()
        write_synthetic_round(round_dir)
        metrics = validate_round(
            round_dir,
            require_correctness=True,
            require_benchmarks=True,
            require_training=True,
            forbid_checkpoints=True,
            check_sm120_defaults=True,
            require_stack_probe=True,
            require_manifest=True,
        )
        scoreboard = round_dir / "scoreboard-candidates.md"
        decision_registry = round_dir / "promotion-decisions.json"
        decision_registry.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "decisions": [
                        {
                            "match": {
                                "suite": "attention",
                                "kernel": "forward",
                                "shape": "B=64 T=1024 C=768 NH=12 HS=64",
                                "selected_stack": "Torch",
                            },
                            "status": "layout_rewrite_only",
                            "active": False,
                            "decision": "Synthetic separated-layout decision.",
                            "evidence": ["synthetic self-test evidence"],
                        },
                        {
                            "match": {
                                "suite": "matmul",
                                "kernel": "fwd",
                                "shape": "qkv M=65536 N=2304 K=768 bias=1 gelu=0",
                                "selected_stack": "cuBLASLt",
                            },
                            "status": "synthetic_trainer_route_rejected",
                            "active": False,
                            "decision": "Synthetic trainer-callable selected-row decision.",
                            "evidence": ["synthetic selected-row evidence"],
                        }
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        decisions = load_promotion_decisions(decision_registry)
        write_scoreboard(scoreboard, metrics, decisions)
        if "fused_classifier" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include runtime candidates")
        if "Objective Coverage" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include objective coverage")
        if "GEMM Shape Coverage" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include GEMM shape coverage")
        if "Runtime Shape Coverage" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include runtime shape coverage")
        if "GEMM Provider Coverage" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include GEMM provider coverage")
        if "Baseline Provider Coverage" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include baseline provider coverage")
        if "Torch Objective Benchmark Coverage" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include Torch objective benchmark coverage")
        if "Python Stack Benchmark Logs" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include Python stack benchmark log coverage")
        if "Selected Backend Rows" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include selected backend rows")
        if "Attention Route Totals" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include attention route totals")
        if "CuTeDSL" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include full stack matrix")
        if "Backend Family-Stack Matrix" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include family-stack matrix")
        if "TorchPacked" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include optional Torch attention rows")
        if "| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton |" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include optional Triton matmul rows")
        if "Unavailable Backend Rows" not in scoreboard.read_text() or "CuTeDSL" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include optional CuTeDSL unavailable rows")
        if "TritonPacked | packed attention backward is not implemented in this Triton prototype" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include optional Triton attention backward unavailable rows")
        if "runtime | adamw_update | `params=124475904 no-master` | Triton | not implemented in this Triton runtime prototype" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include optional Triton runtime unavailable rows")
        if "cuDNNPacked" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include optional cuDNN attention rows")
        if "| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Triton |" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include optional Triton attention rows")
        if "| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TritonPacked |" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include optional Triton packed attention rows")
        if "Use Torch SDPA for already-separated Q/K/V experiments" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not select Torch where the separated-Q/K/V row wins")
        if "Torch native" not in scoreboard.read_text() or "Torch stats" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not preserve Torch LayerNorm variant labels")
        if "Triton dInput-only" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not preserve partial Triton LayerNorm backward labels")
        if "Triton atomic FP32-grad" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not preserve Triton atomic LayerNorm backward labels")
        if "| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | Torch |" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include optional Torch classifier rows")
        if "| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | Triton |" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include optional Triton classifier rows")
        if "| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | Triton |" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include optional Triton classifier dlogits rows")
        if "adamw_update_bf16_state" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include optional Torch runtime rows")
        if "| runtime | cuda_memset | `hidden_elems=50331648` | CUDA kernel |" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include CUDA memory-kernel comparison rows")
        if "| runtime | gelu_forward | `BT=65536 C=3072` | Triton |" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include optional Triton runtime rows")
        selected_json = round_dir / "selected-backends.json"
        write_selected_backends(selected_json, metrics, decisions)
        selected_payload = json.loads(selected_json.read_text())
        rows = selected_payload.get("selected_backend_rows", [])
        if not rows:
            raise RuntimeError("selected backend JSON did not include rows")
        attention_route_rows = selected_payload.get("attention_route_rows", [])
        if not attention_route_rows:
            raise RuntimeError("selected backend JSON did not include attention route totals")
        packed_route_rows = [
            row for row in attention_route_rows
            if row.get("trainer_layout") is True
        ]
        if not any(row.get("stack") == "TK packed-QKV" and row.get("complete") for row in packed_route_rows):
            raise RuntimeError("attention route totals did not include complete TK packed trainer route")
        if not any(row.get("stack") == "TorchPacked" and row.get("complete") for row in packed_route_rows):
            raise RuntimeError("attention route totals did not include complete TorchPacked trainer-layout route")
        if not any(row.get("stack") == "TritonPacked" and not row.get("complete") for row in packed_route_rows):
            raise RuntimeError("attention route totals did not preserve incomplete TritonPacked route evidence")
        torch_attention_rows = [
            row for row in rows
            if row.get("suite") == "attention" and row.get("selected_stack") == "Torch"
        ]
        if not torch_attention_rows:
            raise RuntimeError("selected backend JSON did not preserve Torch attention wins")
        if any(row.get("trainer_call_path_available") for row in torch_attention_rows):
            raise RuntimeError("selected backend JSON incorrectly marked separated Torch SDPA as trainer-callable")
        cxx_rows = [
            row for row in rows
            if row.get("use_scope") in {"C++ benchmark route", "CUDA benchmark route"}
        ]
        if not cxx_rows or not all(row.get("trainer_call_path_available") for row in cxx_rows):
            raise RuntimeError("selected backend JSON did not mark C++/CUDA benchmark routes as trainer-callable")
        resolved_cxx_rows = [
            row for row in rows
            if row.get("suite") == "matmul"
            and row.get("kernel") == "fwd"
            and row.get("shape") == "qkv M=65536 N=2304 K=768 bias=1 gelu=0"
            and row.get("selected_stack") == "cuBLASLt"
        ]
        if not resolved_cxx_rows or resolved_cxx_rows[0].get("decision_status") != "synthetic_trainer_route_rejected":
            raise RuntimeError("selected backend JSON did not attach registry decisions to trainer-callable rows")
        promotion_rows = selected_payload.get("promotion_candidates", [])
        active_promotion_rows = selected_payload.get("active_promotion_candidates", [])
        if not promotion_rows:
            raise RuntimeError("selected backend JSON did not include promotion candidates")
        if not active_promotion_rows:
            raise RuntimeError("selected backend JSON did not include active promotion candidates")
        resolved_attention_rows = [
            row for row in promotion_rows
            if row.get("suite") == "attention"
            and row.get("kernel") == "forward"
            and row.get("selected_stack") == "Torch"
        ]
        if not resolved_attention_rows or resolved_attention_rows[0].get("decision_active"):
            raise RuntimeError("promotion decision registry did not mark the synthetic attention row inactive")
        if any(
            row.get("suite") == "attention"
            and row.get("kernel") == "forward"
            and row.get("selected_stack") == "Torch"
            for row in active_promotion_rows
        ):
            raise RuntimeError("active promotion candidates retained a registry-resolved row")
        if any(row.get("trainer_call_path_available") for row in promotion_rows):
            raise RuntimeError("promotion candidates included an already trainer-callable row")
        if not any(row.get("selected_stack") == "Torch" for row in promotion_rows):
            raise RuntimeError("promotion candidates did not preserve Torch selected winners")
        libtorch_selection = BackendSelection(
            suite="runtime",
            name="cuda_memset",
            shape="hidden_elems=50331648",
            selected_stack="Torch C++",
            selected_time_us=65.0,
            next_stack="CUDA runtime",
            next_time_us=66.0,
            use_scope="LibTorch C++ API route",
            decision_note="optional LibTorch memory row",
        )
        libtorch_selection_row = selection_to_dict(libtorch_selection, decisions, metrics)
        if libtorch_selection_row.get("correctness_logs") != ["bench_sm120_libtorch_runtime.log"]:
            raise RuntimeError("LibTorch C++ selected rows did not carry parity log provenance")
        if libtorch_selection_row.get("correctness_log_paths") != [
            str(round_dir / "bench_sm120_libtorch_runtime.log")
        ]:
            raise RuntimeError("LibTorch C++ selected rows did not carry parity log paths")
        libtorch_matmul_metrics = parse_libtorch_matmul(
            "\n".join(
                [
                    "fc           M=65536 N=3072 K=768 bias=1 gelu=1",
                    "  dW       Torch C++   1320.00 us",
                    "  dW+accum Torch C++   1330.00 us",
                    "",
                ]
            )
        )
        if len(libtorch_matmul_metrics) != 2 or any(metric.stack != "Torch C++" for metric in libtorch_matmul_metrics):
            raise RuntimeError("LibTorch C++ matmul parser did not preserve dWeight rows")
        libtorch_matmul_selection = BackendSelection(
            suite="matmul",
            name="dW",
            shape="fc M=65536 N=3072 K=768 bias=1 gelu=1",
            selected_stack="Torch C++",
            selected_time_us=1320.0,
            next_stack="cuBLAS",
            next_time_us=1330.0,
            use_scope="C++ API prototype",
            decision_note="optional LibTorch dWeight row",
        )
        if selection_timing_log(libtorch_matmul_selection) != "bench_sm120_libtorch_matmul.log":
            raise RuntimeError("LibTorch C++ matmul rows did not carry the LibTorch matmul timing log")
        if selection_correctness_logs(libtorch_matmul_selection) != ["bench_sm120_libtorch_matmul.log"]:
            raise RuntimeError("LibTorch C++ matmul rows did not carry parity log provenance")
        if not all(row.get("promotion_gate") for row in promotion_rows):
            raise RuntimeError("promotion candidates did not include promotion gates")
        if not all(row.get("candidate_class") for row in promotion_rows):
            raise RuntimeError("promotion candidates did not include candidate class")
        class_order = [row.get("candidate_class") for row in promotion_rows]
        native_index = next(
            (i for i, cls in enumerate(class_order) if cls in {"direct integration", "native/codegen integration"}),
            None,
        )
        library_index = next((i for i, cls in enumerate(class_order) if cls == "library integration"), None)
        layout_index = next((i for i, cls in enumerate(class_order) if cls == "layout rewrite"), None)
        if native_index is None or layout_index is None or native_index > layout_index:
            raise RuntimeError("promotion candidates did not prioritize native/direct integration before layout rewrites")
        if library_index is None or library_index > layout_index:
            raise RuntimeError("promotion candidates did not keep library integrations before layout rewrites")
        c3072_layernorm_rows = [
            row for row in promotion_rows
            if row.get("suite") == "layernorm" and "C=3072" in str(row.get("shape"))
        ]
        if c3072_layernorm_rows and any(row.get("candidate_class") != "non-trainer shape" for row in c3072_layernorm_rows):
            raise RuntimeError("promotion candidates did not mark C=3072 LayerNorm as non-trainer shape")
        if c3072_layernorm_rows and any(row.get("decision_active") for row in c3072_layernorm_rows):
            raise RuntimeError("promotion candidates did not mark C=3072 LayerNorm inactive")
        promotion_json = round_dir / "promotion-candidates.json"
        write_promotion_candidates(promotion_json, metrics, decisions)
        promotion_payload = json.loads(promotion_json.read_text())
        if promotion_payload.get("promotion_candidates") != promotion_rows:
            raise RuntimeError("standalone promotion candidates JSON diverged from selected backend payload")
        if promotion_payload.get("active_promotion_candidates") != active_promotion_rows:
            raise RuntimeError("standalone active promotion candidates JSON diverged from selected backend payload")
        if "Promotion Backlog" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include promotion backlog")
        if "Resolved Promotion Decisions" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include resolved promotion decisions")
        if "Resolved Selected Backend Decisions" not in scoreboard.read_text():
            raise RuntimeError("scoreboard output did not include resolved selected backend decisions")

        missing_cudnn_route_dir = Path(tmp) / "missing-cudnn-packed-route"
        missing_cudnn_route_dir.mkdir()
        write_synthetic_round(missing_cudnn_route_dir)
        cudnn_attention_text = (missing_cudnn_route_dir / "bench_sm120_cudnn_attention.log").read_text()
        cudnn_attention_text = cudnn_attention_text.replace(f"{CUDNN_PACKED_BACKWARD_ROUTE}\n", "")
        (missing_cudnn_route_dir / "bench_sm120_cudnn_attention.log").write_text(cudnn_attention_text)
        try:
            validate_round(
                missing_cudnn_route_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "cuDNN packed attention benchmark missing saved-forward backward route evidence" not in str(exc):
                raise
        else:
            raise RuntimeError("missing cuDNN packed backward route unexpectedly passed")

        missing_torch_coverage_dir = Path(tmp) / "missing-torch-coverage"
        missing_torch_coverage_dir.mkdir()
        write_synthetic_round(missing_torch_coverage_dir)
        torch_runtime_text = (missing_torch_coverage_dir / "bench_sm120_torch_runtime.log").read_text()
        torch_runtime_text = torch_runtime_text.replace(
            "encoder_forward                | B=64 T=1024 C=768            | Torch        |   203.000 us\n",
            "",
        )
        (missing_torch_coverage_dir / "bench_sm120_torch_runtime.log").write_text(torch_runtime_text)
        try:
            validate_round(
                missing_torch_coverage_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "Torch benchmark coverage missing objective rows" not in str(exc):
                raise
        else:
            raise RuntimeError("missing Torch objective benchmark row unexpectedly passed")

        missing_libtorch_dir = Path(tmp) / "missing-libtorch-coverage"
        missing_libtorch_dir.mkdir()
        write_synthetic_round(missing_libtorch_dir)
        libtorch_runtime_text = (missing_libtorch_dir / "bench_sm120_libtorch_runtime.log").read_text()
        libtorch_runtime_text = libtorch_runtime_text.replace(
            "cuda_copy_d2d                  | logits_elems=3296722944      | Torch C++    |  9320.000 us\n",
            "",
        )
        (missing_libtorch_dir / "bench_sm120_libtorch_runtime.log").write_text(libtorch_runtime_text)
        try:
            validate_round(
                missing_libtorch_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "LibTorch runtime benchmark missing exact memory rows" not in str(exc):
                raise
        else:
            raise RuntimeError("missing LibTorch runtime row unexpectedly passed")

        missing_libtorch_parity_dir = Path(tmp) / "missing-libtorch-parity"
        missing_libtorch_parity_dir.mkdir()
        write_synthetic_round(missing_libtorch_parity_dir)
        libtorch_runtime_text = (missing_libtorch_parity_dir / "bench_sm120_libtorch_runtime.log").read_text()
        libtorch_runtime_text = libtorch_runtime_text.replace(
            "LibTorch parity cuda_copy_d2d logits_elems=3296722944: PASS\n",
            "",
        )
        (missing_libtorch_parity_dir / "bench_sm120_libtorch_runtime.log").write_text(libtorch_runtime_text)
        try:
            validate_round(
                missing_libtorch_parity_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "LibTorch runtime parity missing exact memory rows" not in str(exc):
                raise
        else:
            raise RuntimeError("missing LibTorch parity row unexpectedly passed")

        missing_libtorch_supplemental_dir = Path(tmp) / "missing-libtorch-supplemental-coverage"
        missing_libtorch_supplemental_dir.mkdir()
        write_synthetic_round(missing_libtorch_supplemental_dir)
        libtorch_runtime_text = (missing_libtorch_supplemental_dir / "bench_sm120_libtorch_runtime.log").read_text()
        libtorch_runtime_text = libtorch_runtime_text.replace(
            "gelu_forward                   | BT=65536 C=3072             | Torch C++    |   550.000 us\n",
            "",
        )
        (missing_libtorch_supplemental_dir / "bench_sm120_libtorch_runtime.log").write_text(
            libtorch_runtime_text
        )
        try:
            validate_round(
                missing_libtorch_supplemental_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "LibTorch runtime benchmark missing supplemental exact rows" not in str(exc):
                raise
        else:
            raise RuntimeError("missing LibTorch supplemental runtime row unexpectedly passed")

        missing_libtorch_supplemental_parity_dir = Path(tmp) / "missing-libtorch-supplemental-parity"
        missing_libtorch_supplemental_parity_dir.mkdir()
        write_synthetic_round(missing_libtorch_supplemental_parity_dir)
        libtorch_runtime_text = (missing_libtorch_supplemental_parity_dir / "bench_sm120_libtorch_runtime.log").read_text()
        libtorch_runtime_text = libtorch_runtime_text.replace(
            "LibTorch parity gelu_forward BT=65536 C=3072: PASS max_abs=0.000000\n",
            "",
        )
        (missing_libtorch_supplemental_parity_dir / "bench_sm120_libtorch_runtime.log").write_text(
            libtorch_runtime_text
        )
        try:
            validate_round(
                missing_libtorch_supplemental_parity_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "LibTorch runtime parity missing supplemental exact rows" not in str(exc):
                raise
        else:
            raise RuntimeError("missing LibTorch supplemental runtime parity unexpectedly passed")

        missing_libtorch_trainer_link_dir = Path(tmp) / "missing-libtorch-trainer-link"
        missing_libtorch_trainer_link_dir.mkdir()
        write_synthetic_round(missing_libtorch_trainer_link_dir)
        (missing_libtorch_trainer_link_dir / LIBTORCH_TRAINER_LINK_LOG).unlink()
        try:
            validate_round(
                missing_libtorch_trainer_link_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "manifest requested LibTorch trainer link probe" not in str(exc):
                raise
        else:
            raise RuntimeError("missing LibTorch trainer link probe unexpectedly passed")

        bad_libtorch_trainer_link_dir = Path(tmp) / "bad-libtorch-trainer-link"
        bad_libtorch_trainer_link_dir.mkdir()
        write_synthetic_round(bad_libtorch_trainer_link_dir)
        trainer_link_log = bad_libtorch_trainer_link_dir / LIBTORCH_TRAINER_LINK_LOG
        trainer_link_log.write_text(
            trainer_link_log.read_text().replace("LibTorch trainer link probe: PASS\n", "")
        )
        try:
            validate_round(
                bad_libtorch_trainer_link_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "LibTorch trainer link probe log missing markers" not in str(exc):
                raise
        else:
            raise RuntimeError("incomplete LibTorch trainer link probe unexpectedly passed")

        raw_pointer_libtorch_route_dir = Path(tmp) / "raw-pointer-libtorch-route"
        raw_pointer_libtorch_route_dir.mkdir()
        write_synthetic_round(raw_pointer_libtorch_route_dir)
        manifest_path = raw_pointer_libtorch_route_dir / "round-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["config"]["libtorch_runtime_route"] = "raw-pointer"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        libtorch_runtime_text = (raw_pointer_libtorch_route_dir / "bench_sm120_libtorch_runtime.log").read_text()
        libtorch_runtime_text = libtorch_runtime_text.replace(
            LIBTORCH_CXX_API_RAW_POINTER_ROUTE,
            LIBTORCH_RAW_POINTER_ROUTE,
        )
        (raw_pointer_libtorch_route_dir / "bench_sm120_libtorch_runtime.log").write_text(libtorch_runtime_text)
        validate_round(
            raw_pointer_libtorch_route_dir,
            require_correctness=True,
            require_benchmarks=True,
            require_training=True,
            forbid_checkpoints=True,
            check_sm120_defaults=True,
            require_stack_probe=True,
            require_manifest=True,
        )

        mismatched_libtorch_route_dir = Path(tmp) / "mismatched-libtorch-route"
        mismatched_libtorch_route_dir.mkdir()
        write_synthetic_round(mismatched_libtorch_route_dir)
        libtorch_runtime_text = (mismatched_libtorch_route_dir / "bench_sm120_libtorch_runtime.log").read_text()
        libtorch_runtime_text = libtorch_runtime_text.replace(
            LIBTORCH_CXX_API_RAW_POINTER_ROUTE,
            LIBTORCH_RAW_POINTER_ROUTE,
        )
        (mismatched_libtorch_route_dir / "bench_sm120_libtorch_runtime.log").write_text(libtorch_runtime_text)
        try:
            validate_round(
                mismatched_libtorch_route_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "LibTorch runtime route mismatch" not in str(exc):
                raise
        else:
            raise RuntimeError("mismatched LibTorch route unexpectedly passed")

        unknown_libtorch_route_dir = Path(tmp) / "unknown-libtorch-route"
        unknown_libtorch_route_dir.mkdir()
        write_synthetic_round(unknown_libtorch_route_dir)
        manifest_path = unknown_libtorch_route_dir / "round-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["config"]["libtorch_runtime_route"] = "unexpected-route"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        try:
            validate_round(
                unknown_libtorch_route_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "unknown manifest route" not in str(exc):
                raise
        else:
            raise RuntimeError("unknown LibTorch route unexpectedly passed")

        missing_libtorch_route_dir = Path(tmp) / "missing-libtorch-route"
        missing_libtorch_route_dir.mkdir()
        write_synthetic_round(missing_libtorch_route_dir)
        libtorch_runtime_text = (missing_libtorch_route_dir / "bench_sm120_libtorch_runtime.log").read_text()
        libtorch_runtime_text = libtorch_runtime_text.replace(f"{LIBTORCH_CXX_API_RAW_POINTER_ROUTE}\n", "")
        (missing_libtorch_route_dir / "bench_sm120_libtorch_runtime.log").write_text(libtorch_runtime_text)
        try:
            validate_round(
                missing_libtorch_route_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "LibTorch runtime route mismatch" not in str(exc):
                raise
        else:
            raise RuntimeError("missing LibTorch raw-pointer route unexpectedly passed")

        bad_dir = Path(tmp) / "bad"
        bad_dir.mkdir()
        write_synthetic_round(bad_dir)
        (bad_dir / "bench_sm120_runtime.log").write_text("bias_add | BT=65536 OC=768 | CUDA | 1.000 us\n")
        try:
            validate_round(
                bad_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "missing timings" not in str(exc):
                raise
        else:
            raise RuntimeError("bad synthetic runtime log unexpectedly passed")

        missing_family_dir = Path(tmp) / "missing-family"
        missing_family_dir.mkdir()
        write_synthetic_round(missing_family_dir)
        set_synthetic_manifest_flag(missing_family_dir, "run_python_stack_benchmarks", "0")
        (missing_family_dir / "bench_sm120_triton_matmul.log").unlink()
        (missing_family_dir / "bench_sm120_matmul.log").write_text(
            "\n".join(
                [
                    "qkv          M=65536 N=2304 K=768 bias=1 gelu=0",
                    "  fwd      TK   1100.00 us | cuBLASLt   1000.00 us | TK/cuBLASLt 1.10x",
                    "  dInp   TK   1200.00 us | cuBLASLt   1050.00 us | TK/cuBLASLt 1.14x",
                    "  dW     TK   1500.00 us | cuBLASLt   1100.00 us | TK/cuBLASLt 1.36x",
                    "  dW+accum TK   1520.00 us | cuBLASLt   1120.00 us | TK/cuBLASLt 1.36x",
                    "fcproj       M=65536 N=768 K=3072 bias=1 gelu=0",
                    "  dInp+dGeLU TK   1460.00 us | cuBLASLt fused   1360.00 us | cuBLASLt explicit   1500.00 us | explicit/fused 1.10x",
                    "",
                ]
            )
        )
        try:
            validate_round(
                missing_family_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if (
                "benchmark coverage missing objective families" not in str(exc)
                and "benchmark matmul shape coverage missing objective rows" not in str(exc)
                and "benchmark matmul provider coverage missing objective stacks" not in str(exc)
            ):
                raise
        else:
            raise RuntimeError("missing benchmark family synthetic log unexpectedly passed")

        missing_python_stack_log_dir = Path(tmp) / "missing-python-stack-log"
        missing_python_stack_log_dir.mkdir()
        write_synthetic_round(missing_python_stack_log_dir)
        (missing_python_stack_log_dir / "bench_sm120_torch_matmul.log").unlink()
        try:
            validate_round(
                missing_python_stack_log_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "manifest requested Python stack benchmarks but logs are missing" not in str(exc):
                raise
        else:
            raise RuntimeError("missing Python stack benchmark log unexpectedly passed")

        missing_libtorch_matmul_log_dir = Path(tmp) / "missing-libtorch-matmul-log"
        missing_libtorch_matmul_log_dir.mkdir()
        write_synthetic_round(missing_libtorch_matmul_log_dir)
        (missing_libtorch_matmul_log_dir / "bench_sm120_libtorch_matmul.log").unlink()
        try:
            validate_round(
                missing_libtorch_matmul_log_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "manifest requested LibTorch matmul benchmarks" not in str(exc):
                raise
        else:
            raise RuntimeError("missing LibTorch matmul benchmark log unexpectedly passed")

        missing_libtorch_matmul_row_dir = Path(tmp) / "missing-libtorch-matmul-row"
        missing_libtorch_matmul_row_dir.mkdir()
        write_synthetic_round(missing_libtorch_matmul_row_dir)
        libtorch_matmul_text = (missing_libtorch_matmul_row_dir / "bench_sm120_libtorch_matmul.log").read_text()
        libtorch_matmul_text = libtorch_matmul_text.replace("  dW+accum Torch C++   1500.00 us\n", "")
        (missing_libtorch_matmul_row_dir / "bench_sm120_libtorch_matmul.log").write_text(libtorch_matmul_text)
        try:
            validate_round(
                missing_libtorch_matmul_row_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "LibTorch matmul benchmark missing exact dWeight rows" not in str(exc):
                raise
        else:
            raise RuntimeError("missing LibTorch matmul dWeight row unexpectedly passed")

        missing_shape_dir = Path(tmp) / "missing-shape"
        missing_shape_dir.mkdir()
        write_synthetic_round(missing_shape_dir)
        (missing_shape_dir / "bench_sm120_matmul.log").write_text(synthetic_matmul_log(include_lmhead=False))
        try:
            validate_round(
                missing_shape_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if (
                "benchmark matmul shape coverage missing objective rows" not in str(exc)
                and "benchmark matmul provider coverage missing objective stacks" not in str(exc)
            ):
                raise
        else:
            raise RuntimeError("missing matmul shape synthetic log unexpectedly passed")

        missing_runtime_shape_dir = Path(tmp) / "missing-runtime-shape"
        missing_runtime_shape_dir.mkdir()
        write_synthetic_round(missing_runtime_shape_dir)
        runtime_lines = [
            "SM120 GPT-2 runtime kernel benchmark on NVIDIA GeForce RTX 5090",
            "Kernel                         | Shape                        | Stack        |         Time",
        ]
        for name in sorted(EXPECTED_RUNTIME_KERNELS):
            stack = "CUDA runtime" if name.startswith("cuda_") else "CUDA"
            shapes = tuple(
                shape
                for kernel_name, required_shapes in RUNTIME_SHAPE_REQUIREMENTS
                if kernel_name == name
                for shape in required_shapes
                if not (kernel_name == "bias_grad_reduce" and shape == "BT=65536 OC=2304")
            ) or ("BT=65536 C=768",)
            for shape in shapes:
                runtime_lines.append(f"{name:<30} | {shape:<28} | {stack:<12} |   42.000 us")
        (missing_runtime_shape_dir / "bench_sm120_runtime.log").write_text("\n".join(runtime_lines) + "\n")
        try:
            validate_round(
                missing_runtime_shape_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "benchmark runtime shape coverage missing objective rows" not in str(exc):
                raise
        else:
            raise RuntimeError("missing runtime shape synthetic log unexpectedly passed")

        missing_provider_dir = Path(tmp) / "missing-provider"
        missing_provider_dir.mkdir()
        write_synthetic_round(missing_provider_dir)
        (missing_provider_dir / "bench_sm120_matmul.log").write_text(
            synthetic_matmul_log(omit_cublas_shape="lmhead")
        )
        try:
            validate_round(
                missing_provider_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "benchmark matmul provider coverage missing objective stacks" not in str(exc):
                raise
        else:
            raise RuntimeError("missing matmul provider synthetic log unexpectedly passed")

        missing_stack_dir = Path(tmp) / "missing-stack"
        missing_stack_dir.mkdir()
        write_synthetic_round(missing_stack_dir)
        (missing_stack_dir / "backend-stacks.json").write_text(
            json.dumps(synthetic_backend_stacks(include_cuteds=False), indent=2) + "\n"
        )
        try:
            validate_round(
                missing_stack_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "missing required stack probe rows" not in str(exc):
                raise
        else:
            raise RuntimeError("missing stack probe synthetic log unexpectedly passed")

        missing_binary_dir = Path(tmp) / "missing-manifest-binary"
        missing_binary_dir.mkdir()
        write_synthetic_round(missing_binary_dir)
        manifest = json.loads((missing_binary_dir / "round-manifest.json").read_text())
        manifest["binaries"] = synthetic_binary_records(omit="bench_sm120_runtime")
        (missing_binary_dir / "round-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        try:
            validate_round(
                missing_binary_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "round manifest missing expected binaries" not in str(exc):
                raise
        else:
            raise RuntimeError("missing manifest binary synthetic round unexpectedly passed")

        missing_matrix_dir = Path(tmp) / "missing-matrix-row"
        missing_matrix_dir.mkdir()
        write_synthetic_round(missing_matrix_dir)
        (missing_matrix_dir / "backend-stacks.json").write_text(
            json.dumps(
                synthetic_backend_stacks(
                    omit_matrix_family="attention_forward",
                    omit_matrix_stack="cuDNN",
                ),
                indent=2,
            )
            + "\n"
        )
        try:
            validate_round(
                missing_matrix_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "missing required family_matrix rows" not in str(exc):
                raise
        else:
            raise RuntimeError("missing family-stack matrix synthetic row unexpectedly passed")

        wrong_baseline_dir = Path(tmp) / "wrong-baseline-provider"
        wrong_baseline_dir.mkdir()
        write_synthetic_round(wrong_baseline_dir)
        runtime_lines = [
            "SM120 GPT-2 runtime kernel benchmark on NVIDIA GeForce RTX 5090",
            "Kernel                         | Shape                        | Stack        |         Time",
        ]
        for name in sorted(EXPECTED_RUNTIME_KERNELS):
            stack = "Triton" if name == "adamw_update" else ("CUDA runtime" if name.startswith("cuda_") else "CUDA")
            shapes = tuple(
                shape
                for kernel_name, required_shapes in RUNTIME_SHAPE_REQUIREMENTS
                if kernel_name == name
                for shape in required_shapes
            ) or ("BT=65536 C=768",)
            for shape in shapes:
                runtime_lines.append(f"{name:<30} | {shape:<28} | {stack:<12} |   42.000 us")
        (wrong_baseline_dir / "bench_sm120_runtime.log").write_text("\n".join(runtime_lines) + "\n")
        try:
            validate_round(
                wrong_baseline_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "benchmark baseline provider coverage missing rows" not in str(exc):
                raise
        else:
            raise RuntimeError("wrong baseline provider synthetic log unexpectedly passed")

        skipped_dir = Path(tmp) / "skipped"
        skipped_dir.mkdir()
        write_synthetic_round(skipped_dir)
        (skipped_dir / "test_fused_classifier.log").write_text(
            "SKIPPED: fused_classifier kernel requires sm_90\n"
            "test_fused_classifier smoke OK\n"
        )
        try:
            validate_round(
                skipped_dir,
                require_correctness=True,
                require_benchmarks=True,
                require_training=True,
                forbid_checkpoints=True,
                check_sm120_defaults=True,
                require_stack_probe=True,
                require_manifest=True,
            )
        except ValueError as exc:
            if "was skipped" not in str(exc):
                raise
        else:
            raise RuntimeError("skipped synthetic correctness log unexpectedly passed")
    print("SM120 round self-test OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SM120 optimization round artifacts")
    parser.add_argument("--round-dir", type=Path, help="Round artifact directory, e.g. scratch/sm120_rounds/<label>")
    parser.add_argument("--require-correctness", action="store_true", help="Require all SM120 correctness logs")
    parser.add_argument("--require-benchmarks", action="store_true", help="Require all SM120 benchmark logs")
    parser.add_argument("--require-training", action="store_true", help="Require train_gpt2cu.log metrics")
    parser.add_argument("--require-stack-probe", action="store_true", help="Require backend-stacks.json")
    parser.add_argument("--require-manifest", action="store_true", help="Require round-manifest.json")
    parser.add_argument("--forbid-checkpoints", action="store_true", help="Fail if model_*.bin/state_*.bin remain in train output dir")
    parser.add_argument("--no-sm120-default-checks", action="store_true", help="Do not require no-master-weights, gelu_fusion=1, BF16 training settings")
    parser.add_argument("--write-scoreboard", type=Path, help="Write parsed markdown rows for best_runs.md review")
    parser.add_argument("--write-selected-backends", type=Path, help="Write machine-readable selected backend rows")
    parser.add_argument("--write-promotion-candidates", type=Path, help="Write non-trainer-callable selected backend winners")
    parser.add_argument(
        "--promotion-decisions",
        type=Path,
        default=DEFAULT_DECISION_REGISTRY,
        help="Repo-local promotion decision registry; use an absent path to disable",
    )
    parser.add_argument("--self-test", action="store_true", help="Run synthetic pass/fail parser tests")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.round_dir is None:
        parser.error("--round-dir is required unless --self-test is set")

    metrics = validate_round(
        args.round_dir,
        require_correctness=args.require_correctness,
        require_benchmarks=args.require_benchmarks,
        require_training=args.require_training,
        forbid_checkpoints=args.forbid_checkpoints,
        check_sm120_defaults=not args.no_sm120_default_checks,
        require_stack_probe=args.require_stack_probe,
        require_manifest=args.require_manifest,
    )
    decisions = load_promotion_decisions(args.promotion_decisions)
    if args.write_scoreboard is not None:
        write_scoreboard(args.write_scoreboard, metrics, decisions)
    if args.write_selected_backends is not None:
        write_selected_backends(args.write_selected_backends, metrics, decisions)
    if args.write_promotion_candidates is not None:
        write_promotion_candidates(args.write_promotion_candidates, metrics, decisions)

    parts = [
        f"round={args.round_dir}",
        f"benchmarks={len(metrics.benchmarks)}",
        f"torch_objective_rows={sum(1 for covered in metrics.torch_benchmark_coverage.values() if covered)}",
        f"libtorch_runtime_rows={sum(1 for covered in metrics.libtorch_runtime_shape_coverage.values() if covered)}",
        f"libtorch_parity_rows={sum(1 for covered in metrics.libtorch_runtime_parity_coverage.values() if covered)}",
        f"libtorch_supplemental_runtime_rows={sum(1 for covered in metrics.libtorch_runtime_supplemental_shape_coverage.values() if covered)}",
        f"libtorch_supplemental_parity_rows={sum(1 for covered in metrics.libtorch_runtime_supplemental_parity_coverage.values() if covered)}",
        f"libtorch_trainer_link_probe={int(metrics.libtorch_trainer_link_probe)}",
        f"stacks={len(metrics.backend_stacks)}",
        f"family_stack_rows={len(metrics.backend_family_matrix)}",
        f"train_steps={len(metrics.train_steps)}",
    ]
    if metrics.total_average_ms is not None:
        parts.append(f"avg_ms={metrics.total_average_ms:.3f}")
    if args.write_scoreboard is not None:
        parts.append(f"scoreboard={args.write_scoreboard}")
    if args.write_selected_backends is not None:
        parts.append(f"selected_backends={args.write_selected_backends}")
    if args.write_promotion_candidates is not None:
        parts.append(f"promotion_candidates={args.write_promotion_candidates}")
    print("SM120 round validation OK: " + "; ".join(parts))


if __name__ == "__main__":
    main()
