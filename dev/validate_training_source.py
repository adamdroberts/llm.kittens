#!/usr/bin/env python3
"""Source-level guards for rank-0 training log evidence contracts."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGGER = ROOT / "llmc" / "logger.h"
GPT2 = ROOT / "train_gpt2.cu"
LLAMA = ROOT / "train_llama3.cu"
LOG_VALIDATOR = ROOT / "dev" / "validate_training_log.py"
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


def validate_logger(logger: str, failures: list[str]) -> None:
    require_all(
        logger,
        [
            "if (log_dir != NULL && process_rank == 0)",
            'snprintf(logger->output_log_file, 512, "%s/main.log", log_dir);',
            "if (resume == 0)",
            'fopenCheck(logger->output_log_file, "w")',
            'fopenCheck(logger->output_log_file, "a")',
            'fprintf(logfile, "s:%d eval:%.4f\\n", step, val);',
            'fprintf(logfile, "s:%d tel:%.4f\\n", step, val_loss);',
            'fprintf(logfile, "s:%d trl:%.4f lr:%.6f norm:%.2f\\n"',
        ],
        rel(LOGGER),
        failures,
    )


def validate_log_parser(parser: str, failures: list[str]) -> None:
    require_all(
        parser,
        [
            'VAL_RE = re.compile(r"^s:(?P<step>\\d+)\\s+tel:',
            'EVAL_RE = re.compile(r"^s:(?P<step>\\d+)\\s+eval:',
            'r"^s:(?P<step>\\d+)\\s+trl:',
            'r"lr:(?P<lr>[-+0-9.eE]+)\\s+norm:(?P<norm>[-+0-9.eE]+)\\s*$"',
            "Training log validation OK",
        ],
        rel(LOG_VALIDATOR),
        failures,
    )


def validate_trainer(source: str, path: Path, failures: list[str]) -> None:
    require_all(
        source,
        [
            '#include "llmc/logger.h"',
            "Logger logger;",
            "logger_init(&logger, output_log_dir, multi_gpu_config.process_rank, resuming);",
            "logger_log_val(&logger, step, val_loss);",
            "logger_log_eval(&logger, step, eval_acc_norm / eval_loader.num_examples);",
            "logger_log_train(&logger, step,",
        ],
        rel(path),
        failures,
    )


def validate_harness(harness: str, failures: list[str]) -> None:
    phase_requirements = {
        "phase_gpt2_smoke": [
            'local log_path="${GPT2_SMOKE_LOG:-$out_dir/main.log}"',
            "GPT2_SMOKE_VALIDATE_ONLY",
            'require_file "$log_path"',
            '--log "$log_path"',
            '--val-final-step "$steps"',
            '--train-final-step "$((steps - 1))"',
            "--require-val",
            "--require-train",
            "--require-train-loss-decrease",
        ],
        "phase_zero3_smoke": [
            'local log_path="${ZERO3_SMOKE_LOG:-$out_dir/main.log}"',
            'local run_log_path="${ZERO3_SMOKE_RUN_LOG:-$out_dir/run.log}"',
            'local zero3_marker="ZeRO Stage 3: parameter shards + runtime all-gather compute layout"',
            "ZERO3_SMOKE_VALIDATE_ONLY",
            "run_to_file_contains",
            'require_file_contains "$zero3_marker" "$run_log_path"',
            'require_file "$log_path"',
            '--log "$log_path"',
            '--val-final-step "$steps"',
            '--train-final-step "$((steps - 1))"',
            "--require-val",
            "--require-train",
            "--max-val-loss",
        ],
        "phase_llama_resume": [
            'local log_path="${LLAMA_RESUME_LOG:-$out/main.log}"',
            "LLAMA_RESUME_VALIDATE_ONLY",
            'require_file "$log_path"',
            '--log "$log_path"',
            '--val-final-step "$final_step"',
            '--train-final-step "$((final_step - 1))"',
            "--require-val",
            "--require-train",
        ],
        "phase_llama1b_stability": [
            'local log_path="${LLAMA1B_STABILITY_LOG:-$out_dir/main.log}"',
            "LLAMA1B_STABILITY_VALIDATE_ONLY",
            'require_file "$log_path"',
            '--log "$log_path"',
            '--val-final-step "$steps"',
            '--train-final-step "$((steps - 1))"',
            "--require-val",
            "--require-train",
            "--require-train-loss-decrease",
            '--eval-final-step "$steps"',
            "--require-eval",
        ],
        "phase_gpt2_full": [
            'local log_path="${GPT2_FULL_LOG:-$out_dir/main.log}"',
            'local run_log_path="${GPT2_FULL_RUN_LOG:-$out_dir/run.log}"',
            'require_gpt2_full_run_log "$run_log_path" "$final_step"',
            'require_gpt_checkpoint_step "$out_dir" "$final_step"',
            'require_file "$log_path"',
            '--log "$log_path"',
            '--val-final-step "$final_step"',
            '--eval-final-step "$final_step"',
            "--require-val",
            "--require-eval",
            "--expected-val-loss",
            "--expected-eval",
        ],
        "phase_llama1b_full": [
            'local log_path="${LLAMA1B_FULL_LOG:-$out_dir/main.log}"',
            'local run_log_path="${LLAMA1B_FULL_RUN_LOG:-$out_dir/run.log}"',
            'require_llama1b_full_run_log "$run_log_path" "$final_step"',
            'require_llama_checkpoint_step "$out_dir" "$final_step"',
            'require_file "$log_path"',
            '--log "$log_path"',
            '--val-final-step "$final_step"',
            '--eval-final-step "$final_step"',
            '--train-final-step "$((final_step - 1))"',
            "--require-val",
            "--require-eval",
            "--require-train",
            "--require-train-loss-decrease",
        ],
        "phase_llama8b_full": [
            'local run_log_path="${LLAMA8B_FULL_RUN_LOG:-$out_dir/run.log}"',
            'require_llama8b_full_run_log "$run_log_path" "$final_step" "$nproc"',
            '--log "$out_dir/main.log"',
            '--val-final-step "$final_step"',
            '--eval-final-step "$final_step"',
            '--train-final-step "$((final_step - 1))"',
            "--require-val",
            "--require-eval",
            "--require-train",
            "--require-train-loss-decrease",
        ],
    }
    for phase, needles in phase_requirements.items():
        body = extract_function(harness, phase)
        require_all(body, needles, f"{rel(HARNESS)} {phase}", failures)
        require_contains(
            body,
            'run_contains "Training log validation OK" python3 dev/validate_training_log.py "${args[@]}"',
            f"{rel(HARNESS)} {phase}",
            failures,
        )


def main() -> None:
    failures: list[str] = []
    validate_logger(LOGGER.read_text(), failures)
    validate_log_parser(LOG_VALIDATOR.read_text(), failures)
    validate_trainer(GPT2.read_text(), GPT2, failures)
    validate_trainer(LLAMA.read_text(), LLAMA, failures)
    validate_harness(HARNESS.read_text(), failures)
    if failures:
        raise AssertionError("\n".join(failures))
    print("Training evidence source guards OK")


if __name__ == "__main__":
    main()
