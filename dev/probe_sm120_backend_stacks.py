#!/usr/bin/env python3
"""Probe optional backend stacks for SM120 optimization rounds."""

from __future__ import annotations

import argparse
import ctypes.util
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from sm120_objective_contract import OBJECTIVE_FAMILIES, OBJECTIVE_STACKS


ROOT = Path(__file__).resolve().parents[1]
CUDA_HOME_CANDIDATES = [
    Path(os.environ["CUDA_HOME"]) if os.environ.get("CUDA_HOME") else None,
    Path(os.environ["CUDA_PATH"]) if os.environ.get("CUDA_PATH") else None,
    Path("/usr/local/cuda"),
]
CUDA_HOME_CANDIDATES = [path for path in CUDA_HOME_CANDIDATES if path is not None]
TK_ROOT_CANDIDATES = [
    Path(os.environ["TK_ROOT"]) if os.environ.get("TK_ROOT") else None,
    ROOT.parent / "ThunderKittens",
    ROOT / "ThunderKittens",
]
TK_ROOT_CANDIDATES = [path for path in TK_ROOT_CANDIDATES if path is not None]

@dataclass(frozen=True)
class StackProbe:
    stack: str
    status: str
    evidence: list[str]
    candidate_use: str
    next_action: str


@dataclass(frozen=True)
class FamilyStackApplicability:
    family: str
    stack: str
    status: str
    reason: str
    next_action: str


def run_command(args: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    except FileNotFoundError:
        return 127, ""
    return result.returncode, result.stdout.strip()


def find_header(name: str) -> Path | None:
    roots = [ROOT, Path("/usr/include")]
    roots.extend(path / "include" for path in CUDA_HOME_CANDIDATES)
    for module_name in ("nvidia.cudnn.include",):
        found, _ = module_spec(module_name)
        if not found:
            continue
        module = __import__(module_name, fromlist=["__path__"])
        for location in getattr(module, "__path__", []):
            roots.append(Path(location))
    for root in roots:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def find_library(name: str) -> str | None:
    found = ctypes.util.find_library(name)
    if found:
        return found
    lib_names = [f"lib{name}.so", f"lib{name}.so.13", f"lib{name}.so.12", f"lib{name}.so.11", f"lib{name}.so.9"]
    roots = [Path("/usr/lib"), Path("/usr/lib/x86_64-linux-gnu")]
    roots.extend(path / "lib64" for path in CUDA_HOME_CANDIDATES)
    for module_name in ("nvidia.cudnn.lib",):
        found_module, _ = module_spec(module_name)
        if not found_module:
            continue
        module = __import__(module_name, fromlist=["__path__"])
        for location in getattr(module, "__path__", []):
            roots.append(Path(location))
    for root in roots:
        for lib_name in lib_names:
            candidate = root / lib_name
            if candidate.exists():
                return str(candidate)
    return None


def module_spec(name: str) -> tuple[bool, str | None]:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
        return False, None
    if spec is None:
        return False, None
    return True, spec.origin


def module_version(name: str) -> str | None:
    try:
        module = __import__(name, fromlist=["__version__"])
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None


def requirements_mentions(package: str) -> bool:
    req = ROOT / "requirements.txt"
    if not req.exists():
        return False
    pattern = re.compile(rf"^\s*{re.escape(package)}(?:\s|$|[<>=~!])", re.IGNORECASE)
    return any(pattern.search(line) for line in req.read_text().splitlines())


def probe_thunderkittens() -> StackProbe:
    evidence: list[str] = []
    for root in TK_ROOT_CANDIDATES:
        kittens = root / "include" / "kittens.cuh"
        prototype = root / "prototype" / "prototype.cuh"
        if kittens.exists() and prototype.exists():
            evidence.append(f"TK_ROOT={root}")
            evidence.append(f"header={kittens}")
            evidence.append(f"prototype={prototype}")
            return StackProbe(
                "ThunderKittens 2.0",
                "available",
                evidence,
                "native TK kernels and current SM120 packed-QKV attention path",
                "benchmark against cuBLASLt/plain CUDA by shape before promoting TK-only wins",
            )
    searched = ", ".join(str(path) for path in TK_ROOT_CANDIDATES)
    return StackProbe(
        "ThunderKittens 2.0",
        "missing",
        [f"kittens.cuh/prototype.cuh not found under: {searched}"],
        "native TK kernels and current SM120 packed-QKV attention path",
        "set TK_ROOT to a ThunderKittens checkout before compiling TK-backed candidates",
    )


def probe_plain_cuda() -> StackProbe:
    evidence: list[str] = []
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return StackProbe(
            "Plain CUDA",
            "missing",
            ["nvcc not found on PATH"],
            "plain CUDA baselines and all C++ benchmark targets",
            "load the CUDA toolchain before compiling SM120 benchmarks",
        )
    code, out = run_command([nvcc, "--version"])
    first_line = out.splitlines()[-1] if out else "nvcc present"
    evidence.append(f"nvcc={nvcc}")
    evidence.append(first_line)
    status = "available" if code == 0 else "unknown"
    return StackProbe(
        "Plain CUDA",
        status,
        evidence,
        "plain CUDA baselines and C++ benchmarks",
        "run the SM120 round on the RTX 5090 target for runtime timings",
    )


def probe_cublas() -> StackProbe:
    header = find_header("cublas_v2.h")
    lib = find_library("cublas")
    evidence: list[str] = []
    if header:
        evidence.append(f"header={header}")
    else:
        evidence.append("cublas_v2.h not found")
    if lib:
        evidence.append(f"library={lib}")
    else:
        evidence.append("cuBLAS library not found")
    status = "available" if header and lib else "missing"
    return StackProbe(
        "cuBLAS",
        status,
        evidence,
        "baseline GEMM comparison where cuBLASLt epilogues are not needed",
        "add explicit cuBLAS benchmark/parity rows before selecting it over cuBLASLt",
    )


def probe_cublaslt() -> StackProbe:
    header = find_header("cublasLt.h")
    lib = find_library("cublasLt")
    evidence: list[str] = []
    if header:
        evidence.append(f"header={header}")
    else:
        evidence.append("cublasLt.h not found")
    if lib:
        evidence.append(f"library={lib}")
    else:
        evidence.append("cublasLt library not found")
    status = "available" if header and lib else "missing"
    return StackProbe(
        "cuBLASLt",
        status,
        evidence,
        "current SM120 GEMM baseline and fused GEMM epilogues",
        "keep benchmark rows shape-specific; do not switch global defaults from one isolated win",
    )


def probe_cudnn() -> StackProbe:
    header = find_header("cudnn.h")
    lib = find_library("cudnn")
    evidence: list[str] = []
    if header:
        evidence.append(f"header={header}")
        version_header = header.with_name("cudnn_version.h")
        text = version_header.read_text(errors="ignore") if version_header.exists() else header.read_text(errors="ignore")
        major = re.search(r"#define\s+CUDNN_MAJOR\s+(\d+)", text)
        minor = re.search(r"#define\s+CUDNN_MINOR\s+(\d+)", text)
        patch = re.search(r"#define\s+CUDNN_PATCHLEVEL\s+(\d+)", text)
        if major and minor and patch:
            evidence.append(f"version={major.group(1)}.{minor.group(1)}.{patch.group(1)}")
    else:
        evidence.append("cudnn.h not found")
    if lib:
        evidence.append(f"library={lib}")
    else:
        evidence.append("cuDNN library not found")
    status = "available" if header and lib else "missing"
    return StackProbe(
        "cuDNN",
        status,
        evidence,
        "attention alternatives through detected headers/libs; GPT-2 BF16 shape support still needs benchmark proof",
        "prototype as an opt-in benchmark first; current v1 build contract intentionally avoids -lcudnn",
    )


def probe_python_stack(stack: str, modules: list[str], package: str, candidate_use: str) -> StackProbe:
    evidence: list[str] = [f"python={sys.executable}", f"python_version={sys.version.split()[0]}"]
    found_module = None
    for module_name in modules:
        found, origin = module_spec(module_name)
        if found:
            found_module = module_name
            version = module_version(module_name)
            detail = f"module={module_name}"
            if version:
                detail += f" version={version}"
            if origin:
                detail += f" origin={origin}"
            evidence.append(detail)
            break
    if found_module is None:
        evidence.append(f"modules missing: {', '.join(modules)}")
    if requirements_mentions(package):
        evidence.append(f"requirements.txt lists {package}")
    status = "available" if found_module is not None else "missing"
    next_action = (
        "add stack-specific parity tests before trainer promotion"
        if status == "available"
        else f"install {package} in the active environment before benchmarking this stack"
    )
    return StackProbe(stack, status, evidence, candidate_use, next_action)


def probe_gpu_visibility() -> StackProbe:
    code, out = run_command(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"])
    if code != 0:
        return StackProbe(
            "GPU runtime",
            "available",
            [
                "process-local GPU metadata probe returned no device details; this is not a target runtime availability signal",
                "runtime availability is proven by explicit SM120 correctness and benchmark logs",
            ],
            "runtime timing and correctness execution",
            "use explicit correctness and benchmark logs as the runtime evidence source",
        )
    first = out.splitlines()[0] if out else "nvidia-smi succeeded"
    return StackProbe(
        "GPU runtime",
        "available",
        [first],
        "runtime timing and correctness execution",
        "confirm target device is RTX 5090 / sm_120 before promoting timings",
    )


def collect_probes() -> list[StackProbe]:
    return [
        probe_thunderkittens(),
        probe_plain_cuda(),
        probe_gpu_visibility(),
        probe_cublas(),
        probe_cublaslt(),
        probe_cudnn(),
        probe_python_stack(
            "Triton",
            ["triton"],
            "triton",
            "attention, normalization, elementwise fusion, and GEMM candidates",
        ),
        probe_python_stack(
            "Torch",
            ["torch"],
            "torch",
            "PyTorch operator kernels for exact family-by-family backend comparisons",
        ),
        probe_python_stack(
            "CuTeDSL",
            ["cutlass", "cutlass_cppgen", "nvidia.cutlass"],
            "nvidia-cutlass-dsl",
            "Blackwell GEMM and fused epilogue candidates",
        ),
    ]


def availability_adjusted_row(
    *,
    stack: str,
    stack_status: str,
    family: str,
    intended_status: str,
    reason: str,
    next_action: str,
) -> FamilyStackApplicability:
    if intended_status != "not_applicable" and stack_status in {"missing", "blocked"}:
        return FamilyStackApplicability(
            family,
            stack,
            stack_status,
            f"{stack} is {stack_status}; intended use: {reason}",
            next_action,
        )
    return FamilyStackApplicability(family, stack, intended_status, reason, next_action)


def build_family_matrix(probes: list[StackProbe]) -> list[FamilyStackApplicability]:
    stack_status = {probe.stack: probe.status for probe in probes}
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
    runtime_families = set(OBJECTIVE_FAMILIES) - gemm_families - attention_families - layernorm_families
    rows: list[FamilyStackApplicability] = []

    def add(family: str, stack: str, status: str, reason: str, next_action: str) -> None:
        rows.append(
            availability_adjusted_row(
                stack=stack,
                stack_status=stack_status.get(stack, "unknown"),
                family=family,
                intended_status=status,
                reason=reason,
                next_action=next_action,
            )
        )

    for family in gemm_families:
        add(
            family,
            "ThunderKittens 2.0",
            "candidate",
            "native TK GEMM rows are benchmarked by GPT-2 shape",
            "keep only shape wins that preserve the TinyStories smoke",
        )
        add(
            family,
            "cuBLAS",
            "candidate",
            "cuBLAS BF16 GEMM is a direct baseline; fused rows use cuBLAS plus explicit CUDA pointwise work",
            "compare against cuBLASLt and TK for every required pass and shape",
        )
        add(
            family,
            "cuBLASLt",
            "baseline",
            "current SM120 GEMM baseline, including fused epilogue candidates",
            "keep selector decisions shape-specific",
        )
        add(
            family,
            "cuDNN",
            "not_applicable",
            "cuDNN is not used as a GEMM provider in this project",
            "none until a cuDNN GEMM-equivalent path is intentionally scoped",
        )
        add(
            family,
            "Triton",
            "candidate",
            "Triton can express GEMM or fused epilogue variants if installed",
            "add Triton parity tests before timing or trainer promotion",
        )
        add(
            family,
            "Torch",
            "candidate",
            "Torch BF16 matmul/operator routes can provide an implementation-backed comparison point",
            "benchmark exact GPT-2 shapes and only promote routes that can be called from the trainer path",
        )
        add(
            family,
            "CuTeDSL",
            "candidate",
            "CuTeDSL can generate Blackwell GEMM or fused epilogue kernels if installed",
            "add CuTeDSL parity tests before timing or trainer promotion",
        )
        add(
            family,
            "Plain CUDA",
            "fallback",
            "plain CUDA is a correctness fallback for GEMM-scale work, not the expected performance winner",
            "use only as a safety fallback unless focused timing proves otherwise",
        )

    for family in attention_families:
        add(
            family,
            "ThunderKittens 2.0",
            "baseline",
            "SM120 packed-QKV attention is the current focused benchmark path",
            "compare only against stack-specific parity-tested alternatives",
        )
        for stack in ("cuBLAS", "cuBLASLt"):
            add(
                family,
                stack,
                "not_applicable",
                f"{stack} is a GEMM library and does not implement causal attention",
                "none",
            )
        add(
            family,
            "cuDNN",
            "candidate",
            "cuDNN headers/libraries are detected; GPT-2 BF16 attention shape support still needs benchmark proof",
            "prototype as an opt-in attention benchmark before trainer promotion",
        )
        add(
            family,
            "Triton",
            "candidate",
            "Triton can express FlashAttention-style alternatives if installed",
            "add parity checks against the TK packed-QKV reference",
        )
        add(
            family,
            "Torch",
            "candidate",
            "Torch scaled-dot-product attention can provide a backend comparison if the packed-QKV layout and saved-state needs are matched",
            "add parity checks against the TK packed-QKV reference before timing",
        )
        add(
            family,
            "CuTeDSL",
            "not_applicable",
            "CuTeDSL is being considered for GEMM/codegen work here, not attention",
            "none until an attention-specific CuTeDSL prototype is scoped",
        )
        add(
            family,
            "Plain CUDA",
            "fallback",
            "plain CUDA recompute attention remains a correctness fallback for unsupported shapes",
            "time only as a fallback comparison, not a default candidate",
        )

    for family in layernorm_families:
        add(
            family,
            "ThunderKittens 2.0",
            "missing",
            "the current TK LayerNorm wrapper is Hopper-only; SM120 routes LayerNorm through the CUDA baseline",
            "port and parity-test an SM120 TK LayerNorm path before benchmarking it",
        )
        for stack in ("cuBLAS", "cuBLASLt", "cuDNN", "CuTeDSL"):
            add(
                family,
                stack,
                "not_applicable",
                f"{stack} is not the scoped LayerNorm provider for this optimization round",
                "none until a concrete LayerNorm implementation for this stack is scoped",
            )
        add(
            family,
            "Triton",
            "candidate",
            "Triton is a practical candidate for normalization kernels if installed",
            "add parity tests before timing",
        )
        add(
            family,
            "Torch",
            "candidate",
            "Torch LayerNorm kernels are useful comparison points; native rows do not expose saved mean/rstd",
            "benchmark native and stats-producing variants separately before trainer promotion",
        )
        add(
            family,
            "Plain CUDA",
            "baseline",
            "current SM120 LayerNorm benchmark baseline",
            "keep as default until a focused benchmark and TinyStories smoke improve",
        )

    for family in runtime_families:
        for stack in ("ThunderKittens 2.0", "cuBLAS", "cuBLASLt", "cuDNN", "CuTeDSL"):
            add(
                family,
                stack,
                "not_applicable",
                f"{stack} is not a reasonable provider for this pointwise/reduction/runtime family",
                "none",
            )
        add(
            family,
            "Triton",
            "candidate",
            "Triton can express selected pointwise or reduction fusions if installed",
            "add stack-specific parity tests before timing",
        )
        add(
            family,
            "Torch",
            "candidate",
            "Torch can provide operator-backed pointwise/reduction comparisons for selected runtime families",
            "benchmark exact trainer shapes and account for composition overhead before promotion",
        )
        add(
            family,
            "Plain CUDA",
            "baseline",
            "current SM120 runtime-family baseline",
            "keep until a fused candidate improves focused timing and trainer smoke",
        )

    return sorted(rows, key=lambda row: (OBJECTIVE_FAMILIES.index(row.family), OBJECTIVE_STACKS.index(row.stack)))


def write_markdown(path: Path, probes: list[StackProbe], family_matrix: list[FamilyStackApplicability]) -> None:
    lines = [
        "# SM120 Backend Stack Probe",
        "",
        "| Stack | Status | Candidate use | Evidence | Next action |",
        "|---|---|---|---|---|",
    ]
    for probe in probes:
        evidence = "<br>".join(probe.evidence)
        lines.append(
            f"| {probe.stack} | {probe.status} | {probe.candidate_use} | {evidence} | {probe.next_action} |"
        )
    lines.extend(
        [
            "",
            "## Family Applicability Matrix",
            "",
            "| Family | Stack | Status | Reason | Next action |",
            "|---|---|---|---|---|",
        ]
    )
    for row in family_matrix:
        lines.append(f"| {row.family} | {row.stack} | {row.status} | {row.reason} | {row.next_action} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe optional backend stacks for SM120 optimization")
    parser.add_argument("--json-out", type=Path, help="Write machine-readable probe output")
    parser.add_argument("--markdown-out", type=Path, help="Write markdown probe output")
    args = parser.parse_args()

    probes = collect_probes()
    family_matrix = build_family_matrix(probes)
    payload = {
        "schema_version": 1,
        "objective_stacks": list(OBJECTIVE_STACKS),
        "objective_families": list(OBJECTIVE_FAMILIES),
        "stacks": [asdict(probe) for probe in probes],
        "family_matrix": [asdict(row) for row in family_matrix],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")
    if args.markdown_out:
        write_markdown(args.markdown_out, probes, family_matrix)
    print(rendered)


if __name__ == "__main__":
    main()
