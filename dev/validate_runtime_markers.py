#!/usr/bin/env python3
"""Source-level guards for runtime success markers used by the H100 harness."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts" / "validate_goal_h100.sh"

SMOKE_TARGETS = {
    "test_matmul": ROOT / "dev" / "cuda" / "test_matmul.cu",
    "test_attention": ROOT / "dev" / "cuda" / "test_attention.cu",
    "test_layernorm": ROOT / "dev" / "cuda" / "test_layernorm.cu",
    "test_rope": ROOT / "dev" / "cuda" / "test_rope.cu",
    "test_rmsnorm": ROOT / "dev" / "cuda" / "test_rmsnorm.cu",
    "test_swiglu": ROOT / "dev" / "cuda" / "test_swiglu.cu",
    "test_attention_gqa": ROOT / "dev" / "cuda" / "test_attention_gqa.cu",
}

DIRECT_MARKERS = [
    (
        "cuda-runtime",
        ROOT / "dev" / "cuda" / "cuda_runtime_check.cu",
        "CUDA runtime check passed.",
        'run_contains "CUDA runtime check passed." ./cuda_runtime_check',
    ),
    (
        "gpt2 forward validation",
        ROOT / "dev" / "cuda" / "gpt2_validate.cu",
        "gpt2_validate OK",
        'run_contains "gpt2_validate OK" ./gpt2_validate',
    ),
    (
        "gpt2 parity",
        ROOT / "test_gpt2.cu",
        "test_gpt2cu OK",
        'run_contains "test_gpt2cu OK" ./test_gpt2cu',
    ),
]

GQA_MARKERS = [
    "GQA case T=128 backward=fallback OK",
    "GQA case T=256 backward=tk OK",
    "test_attention_gqa smoke OK",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def require_contains(text: str, needle: str, context: str, failures: list[str]) -> None:
    if needle not in text:
        failures.append(f"{context} missing {needle!r}")


def require_absent(text: str, needle: str, context: str, failures: list[str]) -> None:
    if needle in text:
        failures.append(f"{context} must not contain stale {needle!r}")


def validate_direct_markers(harness: str, failures: list[str]) -> None:
    for label, source, marker, harness_snippet in DIRECT_MARKERS:
        text = source.read_text()
        require_contains(text, marker, f"{rel(source)} ({label})", failures)
        require_contains(harness, harness_snippet, f"{rel(HARNESS)} ({label})", failures)
    for needle in [
        "CUDA_RUNTIME_VALIDATE_ONLY",
        "CUDA_RUNTIME_LOG",
        "GPT2_RUNTIME_VALIDATE_ONLY",
        "GPT2_VALIDATE_LOG",
        "GPT2_PARITY_LOG",
        'require_file_contains "CUDA runtime check passed."',
        'require_file_contains "gpt2_validate OK"',
        'require_file_contains "test_gpt2cu OK"',
    ]:
        require_contains(harness, needle, f"{rel(HARNESS)} runtime validate-only logs", failures)


def validate_cuda_runtime_contract(failures: list[str]) -> None:
    source = ROOT / "dev" / "cuda" / "cuda_runtime_check.cu"
    text = source.read_text()
    context = rel(source)
    for needle in [
        'std::strcmp(allow_non_h100, "1") == 0',
        "const bool sm90_class = prop.major == 9;",
        'device_name_contains(prop.name, "H100")',
        'device_name_contains(prop.name, "H200")',
        'device_name_contains(prop.name, "GH200")',
        "if (!allow_non_h100_debug && !sm90_class && !named_hopper)",
        "goal.md runtime gates require H100/sm_90-class GPUs;",
        "Set ALLOW_NON_H100=1 only for dry debugging.",
    ]:
        require_contains(text, needle, context, failures)
    require_absent(text, "prop.major < 9", context, failures)


def validate_smoke_markers(harness: str, failures: list[str]) -> None:
    for binary, source in SMOKE_TARGETS.items():
        marker = f"{binary} smoke OK"
        require_contains(source.read_text(), marker, f"{rel(source)} ({binary})", failures)
        require_contains(harness, binary, f"{rel(HARNESS)} smoke phase", failures)
    require_contains(
        harness,
        'run_contains "$bin smoke OK" "./$bin"',
        f"{rel(HARNESS)} smoke phase",
        failures,
    )
    for needle in [
        "SMOKE_VALIDATE_ONLY",
        "SMOKE_LOG_DIR",
        'require_file_contains "$bin smoke OK"',
    ]:
        require_contains(harness, needle, f"{rel(HARNESS)} smoke replay mode", failures)


def validate_gqa_runtime_markers(harness: str, failures: list[str]) -> None:
    source = SMOKE_TARGETS["test_attention_gqa"]
    text = source.read_text()
    require_contains(text, 'printf("GQA case T=%d backward=%s OK', rel(source), failures)
    require_contains(text, "run_case(128, false", rel(source), failures)
    require_contains(text, "run_case(256, true", rel(source), failures)
    for marker in GQA_MARKERS:
        require_contains(harness, marker, f"{rel(HARNESS)} gqa-runtime phase", failures)
    for needle in [
        "GQA_RUNTIME_VALIDATE_ONLY",
        "GQA_RUNTIME_LOG",
        "require_file_contains_all 3",
    ]:
        require_contains(harness, needle, f"{rel(HARNESS)} gqa replay mode", failures)


def main() -> None:
    harness = HARNESS.read_text()
    failures: list[str] = []
    validate_direct_markers(harness, failures)
    validate_cuda_runtime_contract(failures)
    validate_smoke_markers(harness, failures)
    validate_gqa_runtime_markers(harness, failures)
    if failures:
        raise AssertionError("\n".join(failures))
    print("Runtime marker source guards OK")


if __name__ == "__main__":
    main()
