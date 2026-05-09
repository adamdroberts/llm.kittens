#!/usr/bin/env python3
"""Source-level guards for Llama-3.1 8B conversion and validation contracts."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD = ROOT / "dev" / "download_llama3.py"
CONVERTER_TEST = ROOT / "dev" / "validate_llama3_converter.py"
TRAIN_LLAMA_PY = ROOT / "train_llama3.py"
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


def validate_download_script(download: str, failures: list[str]) -> None:
    require_all(
        download,
        [
            '"llama3.1:8B": "meta-llama/Meta-Llama-3.1-8B"',
            '"meta-llama/Meta-Llama-3.1-8B": "llama3.1_8B_bf16.bin"',
            "LLAMA3_MAGIC = 20240803",
            "LLAMA3_BF16_VERSION = 5",
            "LLAMA_HEADER_INTS = 256",
            "def expected_checkpoint_bytes",
            "def validate_checkpoint",
            "if header[0] != LLAMA3_MAGIC",
            "if header[1] != LLAMA3_BF16_VERSION",
            "if actual_bytes != expected_bytes",
            "if header[8] not in (0, hidden)",
            "def write_synthetic_checkpoint",
            "def run_cpp_dry_validation",
            '"-e"',
            '"-x"',
            '"0"',
            '"-z"',
            '"-pn"',
            'parser.add_argument("--validate-only"',
            'parser.add_argument("--write-synthetic-checkpoint"',
            'parser.add_argument("--cpp-validate"',
            '"--cpp-zero-stage"',
            "choices=range(0, 4)",
            '"--cpp-processes"',
            'parser.add_argument("--train-binary", default="./train_llama3cu"',
            "model = LLaMA.from_pretrained_llama3_hf(model_id)",
            'write_model(model, str(output_path), dtype="bfloat16")',
        ],
        rel(DOWNLOAD),
        failures,
    )


def validate_train_converter(train_llama_py: str, failures: list[str]) -> None:
    require_all(
        train_llama_py,
        [
            'assert model_id == "meta-llama/Meta-Llama-3.1-8B"',
            "AutoModelForCausalLM.from_pretrained(model_id)",
            "AutoTokenizer.from_pretrained(model_id)",
            "def write_model",
            "header[0] = 20240803",
            '"bfloat16": 5',
            'assert dtype in {"float32", "bfloat16"}',
            "header[1] = version",
        ],
        rel(TRAIN_LLAMA_PY),
        failures,
    )


def validate_converter_test(converter_test: str, failures: list[str]) -> None:
    require_all(
        converter_test,
        [
            "from dev.download_llama3 import LLAMA_HEADER_BYTES, validate_checkpoint",
            "validate_checkpoint(path)",
            "validate_payload_order(path, expected_segments(model, values))",
            'subprocess.run([str(train_binary.resolve()), "-e", str(path.resolve()), "-x", "0", "-z", "2", "-pn", "8"], check=True)',
            "Llama converter writer validation OK",
        ],
        rel(CONVERTER_TEST),
        failures,
    )


def validate_harness(harness: str, failures: list[str]) -> None:
    phase = extract_function(harness, "phase_llama8b_convert")
    require_all(
        phase,
        [
            "require_file dev/download_llama3.py",
            "require_file ./train_llama3cu",
            'local output_dir="${LLAMA8B_OUTPUT_DIR:-.}"',
            'local checkpoint="${LLAMA8B_CHECKPOINT:-$output_dir/llama3.1_8B_bf16.bin}"',
            'local zero_stage="${LLAMA8B_CONVERT_ZERO_STAGE:-2}"',
            'local dry_processes="${LLAMA8B_CONVERT_PROCESSES:-16}"',
            'expected="train_llama3cu dry run: checkpoint/config parsed"',
            'expected="train_llama3cu dry run: ZeRO-${zero_stage} shard layout validated"',
            "LLAMA8B_CONVERT_VALIDATE_ONLY",
            'require_file "$checkpoint"',
            'if [ -f "$checkpoint" ]; then',
            '--validate-only "$checkpoint"',
            "--cpp-validate",
            '--cpp-zero-stage "$zero_stage"',
            '--cpp-processes "$dry_processes"',
            "--train-binary ./train_llama3cu",
            '"${LLAMA8B_MODEL:-llama3.1:8B}"',
            '--output_dir "$output_dir"',
        ],
        f"{rel(HARNESS)} phase_llama8b_convert",
        failures,
    )
    require_all(
        harness,
        [
            "LLAMA8B_MODEL=llama3.1:8B",
            "LLAMA8B_OUTPUT_DIR=.",
            "LLAMA8B_CHECKPOINT=./llama3.1_8B_bf16.bin",
            "LLAMA8B_CONVERT_VALIDATE_ONLY=0",
            "LLAMA8B_CONVERT_ZERO_STAGE=2",
            "LLAMA8B_CONVERT_PROCESSES=16",
            "llama8b-convert) phase_llama8b_convert ;;",
        ],
        f"{rel(HARNESS)} llama8b-convert usage/dispatcher",
        failures,
    )


def main() -> None:
    failures: list[str] = []
    validate_download_script(DOWNLOAD.read_text(), failures)
    validate_train_converter(TRAIN_LLAMA_PY.read_text(), failures)
    validate_converter_test(CONVERTER_TEST.read_text(), failures)
    validate_harness(HARNESS.read_text(), failures)
    if failures:
        raise AssertionError("\n".join(failures))
    print("Llama conversion source guards OK")


if __name__ == "__main__":
    main()
