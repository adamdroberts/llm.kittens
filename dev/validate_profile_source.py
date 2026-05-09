#!/usr/bin/env python3
"""Source-level guards for the Nsight Compute profiling gate."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCRIPT = ROOT / "profile_gpt2cu.py"
PROFILE_BINARY = ROOT / "profile_gpt2.cu"
PROFILE_PARSER_TEST = ROOT / "dev" / "validate_profile_parser.py"
HARNESS = ROOT / "scripts" / "validate_goal_h100.sh"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def extract_function(text: str, name: str) -> str:
    marker = f"{name}() {{"
    start = text.find(marker)
    if start == -1:
        raise AssertionError(f"missing function {name}")
    body_start = text.find("{", start)
    depth = 1
    i = body_start + 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        raise AssertionError(f"unterminated function {name}")
    return text[body_start + 1 : i - 1]


def require_contains(text: str, needle: str, context: str, failures: list[str]) -> None:
    if needle not in text:
        failures.append(f"{context} missing {needle!r}")


def require_all(text: str, needles: list[str], context: str, failures: list[str]) -> None:
    for needle in needles:
        require_contains(text, needle, context, failures)


def validate_profile_script(profile_script: str, failures: list[str]) -> None:
    require_all(
        profile_script,
        [
            'parser.add_argument("--binary", default="./profile_gpt2cu"',
            'parser.add_argument("--csv-input", default=None',
            'parser.add_argument("--min-tensor-util", type=float, default=70.0',
            'parser.add_argument("--gelu-fusion", type=int, choices=(0, 1), default=0',
            'subprocess.check_call(["make", "profile_gpt2cu", "NO_MULTI_GPU=1", "NO_USE_MPI=1"])',
            '"-f", args.binary, "--gelu-fusion", str(args.gelu_fusion),',
            '"gpu__time_duration.sum"',
            '"dram__bytes_read.sum"',
            '"dram__bytes_write.sum"',
            '"sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"',
            'cmd = [NCU, "-i", report_path, "--csv", "--page", "raw", "--metrics", ",".join(metrics)]',
            'print(f"Tensor-core utilization gate:',
            "if avg_tensor_util < min_tensor_util:",
            "average tensor-core utilization",
        ],
        rel(PROFILE_SCRIPT),
        failures,
    )


def validate_profile_binary(profile_binary: str, failures: list[str]) -> None:
    require_all(
        profile_binary,
        [
            '#define TESTING',
            '#include "train_gpt2.cu"',
            'gpt2_build_from_checkpoint(&model, "gpt2_124M_bf16.bin");',
            "model.config.num_layers = 1;",
            "model.gelu_fusion = gelu_fusion;",
            'printf("gelu fusion: %d\\n", gelu_fusion);',
            "set_zero_configs(&multi_gpu_config, 0, model.num_parameters);",
            "gpt2_forward(&model, x, B, T);",
            "gpt2_backward_and_reduce(&model, x, y, 1, 0);",
            "gpt2_calculate_grad_norm",
            "gpt2_update",
            "cudaCheck(cudaDeviceSynchronize());",
        ],
        rel(PROFILE_BINARY),
        failures,
    )


def validate_parser_test(parser_test: str, failures: list[str]) -> None:
    require_all(
        parser_test,
        [
            "write_csv(passing, tensor_pct=82.0)",
            "write_csv(failing, tensor_pct=42.0)",
            '"--min-tensor-util"',
            '"--gelu-fusion"',
            "gelu_fusion=1",
            "Tensor-core utilization gate:",
            "below the required",
            "profile parser validation OK",
        ],
        rel(PROFILE_PARSER_TEST),
        failures,
    )


def validate_harness(harness: str, failures: list[str]) -> None:
    phase = extract_function(harness, "phase_profile")
    require_all(
        phase,
        [
            'if [ "${PROFILE_VALIDATE_ONLY:-0}" != "1" ]; then',
            "require_cuda_tool ncu",
            "require_file gpt2_124M_bf16.bin",
            "for fusion in ${PROFILE_GELU_FUSIONS:-0 1}; do",
            "PROFILE_VALIDATE_ONLY",
            "PROFILE_REPORT_DIR",
            "PROFILE_CSV_DIR",
            'local csv="${PROFILE_CSV_DIR}/${output}.csv"',
            'args+=(--csv-input "$csv")',
            'require_file "$report"',
            "--skip-build --skip-run --report",
            '--gelu-fusion "$fusion"',
            '--output "$output"',
        ],
        f"{rel(HARNESS)} phase_profile",
        failures,
    )
    require_contains(harness, "PROFILE_MIN_TENSOR_UTIL=70", f"{rel(HARNESS)} usage", failures)
    require_contains(harness, 'PROFILE_GELU_FUSIONS="0 1"', f"{rel(HARNESS)} usage", failures)
    require_contains(harness, "PROFILE_VALIDATE_ONLY=0", f"{rel(HARNESS)} usage", failures)
    require_contains(harness, "PROFILE_REPORT_DIR=.", f"{rel(HARNESS)} usage", failures)
    require_contains(harness, "PROFILE_CSV_DIR=...", f"{rel(HARNESS)} usage", failures)
    require_contains(harness, "phase_profile_parser", f"{rel(HARNESS)} profile parser phase", failures)


def main() -> None:
    failures: list[str] = []
    validate_profile_script(PROFILE_SCRIPT.read_text(), failures)
    validate_profile_binary(PROFILE_BINARY.read_text(), failures)
    validate_parser_test(PROFILE_PARSER_TEST.read_text(), failures)
    validate_harness(HARNESS.read_text(), failures)
    if failures:
        raise AssertionError("\n".join(failures))
    print("Profile source guards OK")


if __name__ == "__main__":
    main()
