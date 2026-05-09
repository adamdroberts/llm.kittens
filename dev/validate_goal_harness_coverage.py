#!/usr/bin/env python3
"""Source-level coverage checks for the goal-complete validation harness."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts" / "validate_goal_h100.sh"
GOAL = ROOT / "goal.md"
MAKEFILE = ROOT / "Makefile"
GOAL_REPLAY = ROOT / "dev" / "validate_goal_replay.py"

COMPILE_TARGETS = [
    "test_matmul",
    "test_attention",
    "test_layernorm",
    "test_rope",
    "test_rmsnorm",
    "test_swiglu",
    "test_attention_gqa",
    "test_dataloader",
    "cuda_runtime_check",
    "train_gpt2cu",
    "test_gpt2cu",
    "gpt2_validate",
    "profile_gpt2cu",
    "train_llama3cu",
]

GOAL_CORE_PHASES = [
    "phase_preflight",
    "phase_compile",
    "phase_script_syntax",
    "phase_python_syntax",
    "phase_source_guards",
    "phase_data_artifacts",
    "phase_dataloader_smoke",
    "phase_gqa_reference",
    "phase_profile_parser",
    "phase_log_tools",
    "phase_goal_replay_smoke",
    "phase_llama_converter_smoke",
    "phase_cuda_runtime",
    "phase_starter_pack",
    "phase_smoke",
    "phase_gpt2",
    "phase_gpt_dry",
    "phase_llama_dry",
    "phase_llama_checkpoint_smoke",
    "phase_zero_guards",
]

GOAL_COMPLETE_PHASES = [
    "phase_goal_core",
    "phase_gpt2_smoke",
    "phase_zero3_smoke",
    "phase_llama_resume",
    "phase_gqa_runtime",
    "phase_llama1b_stability",
    "phase_profile",
    "phase_gpt2_full",
    "phase_gpt2_two_node",
    "phase_llama1b_full",
    "phase_llama8b_convert",
    "phase_llama8b_full",
]

REQUIRED_THRESHOLD_ENV = [
    "GPT2_SMOKE_MAX_VAL_LOSS",
    "ZERO3_SMOKE_MAX_VAL_LOSS",
    "LLAMA_RESUME_MAX_VAL_LOSS",
    "LLAMA1B_STABILITY_MAX_VAL_LOSS",
    "LLAMA1B_STABILITY_MIN_HELLASWAG",
    "GPT2_FULL_EXPECTED_VAL_LOSS",
    "GPT2_FULL_EXPECTED_HELLASWAG",
    "GPT2_TWO_NODE_REL_TOL",
    "LLAMA1B_FULL_MAX_VAL_LOSS",
    "LLAMA1B_FULL_MIN_HELLASWAG",
    "LLAMA8B_FULL_MAX_VAL_LOSS",
    "LLAMA8B_FULL_MIN_HELLASWAG",
]

GOAL_COMPLETE_REQUIRED = [
    "ALLOW_FULL_GOAL_RUN",
    "phase_goal_complete_prereqs",
    "LLAMA1B_STABILITY_HELLASWAG=1",
    "saved_llama1b_stability_hellaswag",
]

GOAL_COMPLETE_PREREQ_REQUIRED = [
    'if [ "${ALLOW_NON_H100:-0}" = "1" ]; then',
    "goal-complete requires real H100/sm_90-class runtime evidence; unset ALLOW_NON_H100",
    "require_goal_metric_thresholds",
    "require_cuda_tool ncu",
    "require_file gpt2_124M_bf16.bin",
    "PREFLIGHT_VALIDATE_ONLY",
    "PREFLIGHT_LOG",
    'require_file_contains "H100 preflight OK"',
    "CUDA_RUNTIME_VALIDATE_ONLY",
    "SMOKE_VALIDATE_ONLY",
    "GPT2_RUNTIME_VALIDATE_ONLY",
    "GQA_RUNTIME_VALIDATE_ONLY",
    "GPT2_SMOKE_VALIDATE_ONLY",
    "LLAMA_RESUME_VALIDATE_ONLY",
    "LLAMA1B_STABILITY_VALIDATE_ONLY",
    'require_file_contains "CUDA runtime check passed."',
    'require_file_contains "$bin smoke OK"',
    'require_file_contains "gpt2_validate OK"',
    'require_file_contains "test_gpt2cu OK"',
    "require_file_contains_all 3",
    "require_file \"$gpt2_smoke_log\"",
    "require_llama_checkpoint_step \"$llama_resume_out\" \"$llama_resume_step\"",
    "ZERO3_SMOKE_RUN_LOG",
    "ZeRO Stage 3: parameter shards + runtime all-gather compute layout",
    "require_file \"${LLAMA_RESUME_LOG:-$llama_resume_out/main.log}\"",
    "require_file \"${LLAMA1B_STABILITY_LOG:-$llama1b_stability_out/main.log}\"",
    "PROFILE_VALIDATE_ONLY",
    "PROFILE_CSV_DIR",
    "require_goal_complete_profile_evidence",
    "GPT2_FULL_VALIDATE_ONLY",
    "GPT2_TWO_NODE_VALIDATE_ONLY",
    "LLAMA1B_FULL_VALIDATE_ONLY",
    "LLAMA8B_CONVERT_VALIDATE_ONLY",
    "LLAMA8B_FULL_VALIDATE_ONLY",
    "require_cmd sbatch",
    "require_file \"$gpt2_full_log\"",
    "require_gpt2_full_run_log \"$gpt2_full_run_log\"",
    "require_gpt_checkpoint_step \"$gpt2_full_out\"",
    "require_file \"$reference_log\"",
    "require_file \"$candidate_log\"",
    "require_file \"$llama1b_log\"",
    "require_llama1b_full_run_log \"$llama1b_run_log\"",
    "require_llama_checkpoint_step \"$llama1b_out\"",
    "require_file \"$llama8b_checkpoint\"",
    "require_llama8b_full_run_log \"$run_log_path\"",
    "require_llama_checkpoint_step \"$out_dir\" \"$final_step\"",
    "require_file \"$out_dir/main.log\"",
]

GOAL_COMPLETE_PROFILE_REQUIRED = [
    "require_goal_complete_profile_evidence",
    "for fusion in 0 1; do",
    "PROFILE_CSV_DIR",
    'require_file "${PROFILE_CSV_DIR}/profile_ge${fusion}.csv"',
    "PROFILE_REPORT_DIR",
    'require_file "${PROFILE_REPORT_DIR:-.}/profile_ge${fusion}.ncu-rep"',
]

GOAL_THRESHOLD_HELPER_REQUIRED = [
    "require_metric_env",
    "GPT2_FULL_METRIC_REL_TOL",
    "must be numeric",
    "must be finite",
    "must be > 0",
    "must be in [0, 1]",
]

CASE_LABELS = [
    "goal-core",
    "goal-complete-prereqs",
    "goal-complete",
    "goal-replay-smoke",
    "gpt2-smoke",
    "llama-resume",
    "gqa-runtime",
    "llama1b-stability",
    "profile",
    "gpt2-full",
    "gpt2-two-node",
    "llama1b-full",
    "llama8b-convert",
    "llama8b-full",
]

GOAL_PHASE_REFERENCES = [
    "gpt2_validate",
    "test_gpt2cu",
    "gpt2-full",
    "gpt2-two-node",
    "gqa-runtime",
    "llama1b-stability",
    "llama8b-convert",
    "llama8b-full",
    "profile",
]

GOAL_CORE_USAGE_PHASES = [
    "preflight",
    "compile",
    "script-syntax",
    "python-syntax",
    "source-guards",
    "data-artifacts",
    "dataloader-smoke",
    "gqa-reference",
    "profile-parser",
    "log-tools",
    "goal-replay-smoke",
    "llama-converter-smoke",
    "cuda-runtime",
    "starter-pack",
    "smoke",
    "gpt2",
    "gpt-dry",
    "llama-dry",
    "llama-checkpoint-smoke",
    "zero-guards",
]

SOURCE_GUARD_REQUIRED = [
    "dev/validate_zero_layout.py",
    'run_contains "ZeRO shard layout validation OK" python3 dev/validate_zero_layout.py',
]

GOAL_REPLAY_REQUIRED = [
    "expect_harness_fail",
    "ALLOW_NON_H100",
    "goal-complete requires real H100/sm_90-class runtime evidence",
    "goal-complete requires explicit metric thresholds",
    "GPT2_FULL_EXPECTED_VAL_LOSS must be numeric",
    "LLAMA8B_FULL_MIN_HELLASWAG must be in [0, 1]",
    "ZERO3_SMOKE_MAX_VAL_LOSS",
    "ZERO3_SMOKE_RUN_LOG",
    "zero3_smoke_run_missing_stage.log",
    "LLAMA8B_FULL_MIN_HELLASWAG",
    "GPT2_FULL_RUN_LOG",
    "LLAMA1B_FULL_RUN_LOG",
    "LLAMA8B_FULL_RUN_LOG",
    "test_attention_gqa_missing_t256.log",
    "GQA case T=256 backward=tk OK",
    "missing_profile",
    "profile_ge0.csv",
    "missing_fused_profile",
    "profile_ge1.csv",
    "llama1b_stability_bad_eval_step.log",
    "eval accuracy latest step 1 is before required step 4",
    "gpt2_full_bad_run.log",
    "NPROC=8",
]

RUNTIME_EVIDENCE_CONTRACTS = [
    {
        "name": "M2 GPT-2 forward loss",
        "unchecked_anchor": "`./gpt2_validate` returns a sane forward loss",
        "goal": ["./gpt2_validate", "gpt2_validate"],
        "phase": "phase_gpt2",
        "phase_needles": [
            "require_file ./gpt2_validate",
            'run_contains "gpt2_validate OK" ./gpt2_validate',
            "GPT2_RUNTIME_VALIDATE_ONLY",
            'require_file_contains "gpt2_validate OK"',
        ],
    },
    {
        "name": "M3 GPT-2 backward parity",
        "unchecked_anchor": "`make test_gpt2cu && ./test_gpt2cu` passes on H100",
        "goal": ["make test_gpt2cu && ./test_gpt2cu", "test_gpt2cu"],
        "phase": "phase_gpt2",
        "phase_needles": [
            "require_file ./test_gpt2cu",
            'run_contains "test_gpt2cu OK" ./test_gpt2cu',
            "GPT2_RUNTIME_VALIDATE_ONLY",
            'require_file_contains "test_gpt2cu OK"',
        ],
    },
    {
        "name": "M4 GPT-2 full reproduction",
        "unchecked_anchor": "8×H100 end-to-end",
        "goal": ["8×H100 end-to-end", "GPT2_FULL_EXPECTED_VAL_LOSS", "GPT2_FULL_EXPECTED_HELLASWAG"],
        "phase": "phase_gpt2_full",
        "phase_needles": [
            "scripts/run_gpt2_124M.sh",
            "GPT2_FULL_VALIDATE_ONLY",
            "GPT2_FULL_RUN_LOG",
            "require_gpt2_full_run_log",
            "require_gpt_checkpoint_step",
            "require_file \"$log_path\"",
            "--expected-val-loss",
            "--expected-eval",
            'run_contains "Training log validation OK" python3 dev/validate_training_log.py "${args[@]}"',
        ],
    },
    {
        "name": "M5 two-node loss curve",
        "unchecked_anchor": "2-node sanity run",
        "goal": ["2-node sanity run", "GPT2_TWO_NODE_REL_TOL"],
        "phase": "phase_gpt2_two_node",
        "phase_needles": [
            "require_cmd sbatch",
            "scripts/multi_node/run_gpt2_124M_fs.sbatch",
            "--reference-log",
            "--candidate-log",
            "--rel-tol",
            "--require-decrease",
            'run_contains "Training log comparison OK" python3 dev/compare_training_logs.py "${args[@]}"',
        ],
    },
    {
        "name": "M5 GPT ZeRO-2 descriptor dry-runs",
        "unchecked_anchor": "ZeRO-2 and ZeRO-3 paths",
        "goal": ["ZeRO-2 and ZeRO-3 paths", "host-only GPT/Llama dry-runs", "ZeRO-2 tensor layout"],
        "phase": "phase_gpt_dry",
        "phase_needles": [
            "gpt3:c384",
            "gpt3:c5120",
            "gpt3:c12288",
            "-z 2",
            "GPT dry run: ZeRO-2 layout validated",
        ],
    },
    {
        "name": "M5 Llama ZeRO-2/3 dry-runs",
        "unchecked_anchor": "ZeRO-2 and ZeRO-3 paths",
        "goal": ["ZeRO-2 and ZeRO-3 paths", "host-only GPT/Llama dry-runs", "ZeRO-3 parameter-shard layout"],
        "phase": "phase_llama_dry",
        "phase_needles": [
            "llama3.1:8B",
            "-z 2",
            "-z 3",
            "train_llama3cu dry run: ZeRO-2 shard layout validated",
            "train_llama3cu dry run: ZeRO-3 shard layout validated",
        ],
    },
    {
        "name": "M5 ZeRO request guard diagnostics",
        "unchecked_anchor": "ZeRO-2 and ZeRO-3 paths",
        "goal": ["ZeRO-3 parameter-shard layout", "ZeRO request guard checks"],
        "phase": "phase_zero_guards",
        "phase_needles": [
            "GPT dry run: ZeRO-3 layout validated",
            "train_llama3cu dry run: ZeRO-3 shard layout validated",
            "supports only ZeRO-0, ZeRO-1, ZeRO-2, and ZeRO-3",
            "cannot be evenly partitioned across 5 processes",
            "cannot be evenly partitioned across 7 processes",
        ],
    },
    {
        "name": "M5 ZeRO-3 runtime smoke",
        "unchecked_anchor": "ZeRO-2 and ZeRO-3 paths",
        "goal": ["ZeRO-3 parameter-shard runtime path", "H100/NCCL end-to-end validation"],
        "phase": "phase_zero3_smoke",
        "phase_needles": [
            "ZERO3_SMOKE_VALIDATE_ONLY",
            "ZERO3_SMOKE_NPROC",
            "ZERO3_SMOKE_MAX_VAL_LOSS",
            "ZERO3_SMOKE_RUN_LOG",
            "run_to_file_contains",
            "ZeRO Stage 3: parameter shards + runtime all-gather compute layout",
            "-z 3",
            "--max-val-loss",
            "Training log validation OK",
        ],
    },
    {
        "name": "M6 GQA runtime numerics",
        "unchecked_anchor": "`llmc/attention_gqa.cuh` + `llmc/tk/attention_gqa_h100.cuh`",
        "goal": ["llmc/attention_gqa.cuh", "gqa-runtime", "Runtime numerical validation"],
        "phase": "phase_gqa_runtime",
        "phase_needles": [
            "phase_gqa_reference",
            "GQA_RUNTIME_VALIDATE_ONLY",
            "GQA_RUNTIME_LOG",
            "require_file_contains_all 3",
            "GQA case T=128 backward=fallback OK",
            "GQA case T=256 backward=tk OK",
            "test_attention_gqa smoke OK",
        ],
    },
    {
        "name": "M6 Llama-3 1B stability",
        "unchecked_anchor": "Forward+backward stable on FineWeb-edu",
        "goal": ["Forward+backward stable on FineWeb-edu", "llama1b-stability"],
        "phase": "phase_llama1b_stability",
        "phase_needles": [
            "LLAMA1B_STABILITY_TRAIN_PATTERN",
            "LLAMA1B_STABILITY_VAL_PATTERN",
            "LLAMA1B_STABILITY_VALIDATE_ONLY",
            "LLAMA1B_STABILITY_LOG",
            "require_file \"$log_path\"",
            "--require-train-loss-decrease",
            "--require-eval",
            "LLAMA1B_STABILITY_MIN_HELLASWAG",
            'run_contains "Training log validation OK" python3 dev/validate_training_log.py "${args[@]}"',
        ],
    },
    {
        "name": "M6 Llama-3 1B full run",
        "goal": ["llama1b-full", "LLAMA1B_FULL_MAX_VAL_LOSS", "LLAMA1B_FULL_MIN_HELLASWAG"],
        "phase": "phase_llama1b_full",
        "phase_needles": [
            "scripts/run_llama3_1B.sh",
            "LLAMA1B_FULL_VALIDATE_ONLY",
            "LLAMA1B_FULL_RUN_LOG",
            "require_llama1b_full_run_log",
            "require_llama_checkpoint_step",
            "require_file \"$log_path\"",
            "--require-train-loss-decrease",
            "--max-val-loss",
            "--min-eval",
            'run_contains "Training log validation OK" python3 dev/validate_training_log.py "${args[@]}"',
        ],
    },
    {
        "name": "M7 real Llama-3.1 8B conversion",
        "unchecked_anchor": "HF checkpoint converter validated end-to-end",
        "goal": ["HF checkpoint converter validated end-to-end", "llama8b-convert"],
        "phase": "phase_llama8b_convert",
        "phase_needles": [
            "dev/download_llama3.py",
            "LLAMA8B_CONVERT_VALIDATE_ONLY",
            "require_file \"$checkpoint\"",
            "--validate-only",
            "--cpp-validate",
            "--cpp-zero-stage",
            'run_contains "$expected"',
        ],
    },
    {
        "name": "M7 Llama-3 8B multi-node full run",
        "goal": ["scripts/multi_node/run_llama3_8B_fs.sbatch", "llama8b-full", "runtime awaits ZeRO-2"],
        "phase": "phase_llama8b_full",
        "phase_needles": [
            "scripts/multi_node/run_llama3_8B_fs.sbatch",
            "LLAMA8B_FULL_RUN_LOG",
            "require_llama8b_full_run_log",
            "require_llama_checkpoint_step",
            "dev/validate_llama_checkpoint_artifacts.py",
            "--require-train-loss-decrease",
            "LLAMA8B_FULL_MAX_VAL_LOSS",
            "LLAMA8B_FULL_MIN_HELLASWAG",
            'run_contains "Training log validation OK" python3 dev/validate_training_log.py "${args[@]}"',
        ],
    },
    {
        "name": "M8 ncu profile default and fused epilogue",
        "unchecked_anchor": "`scripts/validate_goal_h100.sh profile` on H100",
        "goal": ["scripts/validate_goal_h100.sh profile", "PROFILE_GELU_FUSIONS", "ncu"],
        "phase": "phase_profile",
        "phase_needles": [
            "require_cuda_tool ncu",
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
            'run python3 profile_gpt2cu.py "${args[@]}"',
        ],
    },
    {
        "name": "M8 optional bias+GELU epilogue",
        "unchecked_anchor": "**Optional v1.1**",
        "goal": ["Optional v1.1", "PROFILE_GELU_FUSIONS=\"0 1\"", "H100 numerical validation"],
        "phase": "phase_source_guards",
        "phase_needles": [
            "dev/validate_epilogue_source.py",
            'run_contains "GELU epilogue source guards OK" python3 dev/validate_epilogue_source.py',
        ],
    },
]


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


def require_all(text: str, needles: list[str], context: str, failures: list[str]) -> None:
    for needle in needles:
        if needle not in text:
            failures.append(f"{context} missing {needle}")


def require_phase_order(body: str, phases: list[str], context: str, failures: list[str]) -> None:
    cursor = -1
    for phase in phases:
        offset = body.find(phase)
        if offset == -1:
            failures.append(f"{context} missing {phase}")
            continue
        if offset <= cursor:
            failures.append(f"{context} phase order regression at {phase}")
        cursor = offset


def require_before(body: str, earlier: list[str], later: str, context: str, failures: list[str]) -> None:
    later_offset = body.find(later)
    if later_offset == -1:
        failures.append(f"{context} missing ordering anchor {later}")
        return
    for needle in earlier:
        offset = body.find(needle)
        if offset == -1:
            failures.append(f"{context} missing {needle}")
        elif offset >= later_offset:
            failures.append(f"{context} expected {needle} before {later}")


def validate_goal_core_usage(harness: str, failures: list[str]) -> None:
    usage_line = next((line for line in harness.splitlines() if line.strip().startswith("goal-core")), "")
    if not usage_line:
        failures.append("usage missing goal-core summary")
        return
    tokens = usage_line.split()[1:]
    if tokens != GOAL_CORE_USAGE_PHASES:
        failures.append(
            "goal-core usage summary mismatch: "
            f"expected {' '.join(GOAL_CORE_USAGE_PHASES)}, got {' '.join(tokens)}"
        )


def validate_runtime_evidence_map(goal: str, harness: str, failures: list[str]) -> None:
    for contract in RUNTIME_EVIDENCE_CONTRACTS:
        name = str(contract["name"])
        require_all(goal, contract["goal"], f"goal.md requirement map for {name}", failures)
        phase = extract_function(harness, str(contract["phase"]))
        require_all(phase, contract["phase_needles"], f"{contract['phase']} evidence for {name}", failures)


def validate_unchecked_goal_items_are_mapped(goal: str, failures: list[str]) -> None:
    unchecked_lines = [line for line in goal.splitlines() if line.startswith("- [ ]")]
    anchors = sorted(
        {str(contract["unchecked_anchor"]) for contract in RUNTIME_EVIDENCE_CONTRACTS if "unchecked_anchor" in contract}
    )
    for anchor in anchors:
        if not any(anchor in line for line in unchecked_lines):
            failures.append(f"unchecked goal anchor has no matching - [ ] item: {anchor}")
    for line in unchecked_lines:
        if not any(anchor in line for anchor in anchors):
            failures.append(f"unchecked goal item lacks runtime-evidence mapping: {line}")


def main() -> None:
    harness = HARNESS.read_text()
    goal = GOAL.read_text()
    makefile = MAKEFILE.read_text()
    goal_replay = GOAL_REPLAY.read_text()
    failures: list[str] = []

    require_phase_order(extract_function(harness, "phase_goal_core"), GOAL_CORE_PHASES, "phase_goal_core", failures)
    compile_body = extract_function(harness, "phase_compile")
    require_all(compile_body, COMPILE_TARGETS, "phase_compile target list", failures)
    require_all(makefile, [f"{target}:" for target in COMPILE_TARGETS], "Makefile target definitions", failures)
    complete_body = extract_function(harness, "phase_goal_complete")
    prereq_body = extract_function(harness, "phase_goal_complete_prereqs")
    source_guard_body = extract_function(harness, "phase_source_guards")
    require_phase_order(complete_body, GOAL_COMPLETE_PHASES, "phase_goal_complete", failures)
    require_all(complete_body, GOAL_COMPLETE_REQUIRED, "phase_goal_complete completion guard", failures)
    require_before(
        complete_body,
        ["phase_goal_complete_prereqs"],
        "phase_goal_core",
        "phase_goal_complete fail-fast guard",
        failures,
    )
    require_all(prereq_body, GOAL_COMPLETE_PREREQ_REQUIRED, "phase_goal_complete_prereqs completion guard", failures)
    require_all(harness, GOAL_COMPLETE_PROFILE_REQUIRED, "goal-complete profile evidence helper", failures)
    require_all(harness, GOAL_THRESHOLD_HELPER_REQUIRED, "goal metric threshold helper", failures)
    require_all(prereq_body, REQUIRED_THRESHOLD_ENV, "phase_goal_complete_prereqs threshold gate", failures)
    require_all(harness, [f"{label})" for label in CASE_LABELS], "phase dispatcher", failures)
    validate_goal_core_usage(harness, failures)
    require_all(source_guard_body, SOURCE_GUARD_REQUIRED, "phase_source_guards source guard list", failures)
    require_all(goal_replay, GOAL_REPLAY_REQUIRED, "goal replay negative evidence checks", failures)
    require_all(harness, REQUIRED_THRESHOLD_ENV, "harness usage/threshold docs", failures)
    require_all(goal, GOAL_PHASE_REFERENCES, "goal.md runtime phase references", failures)
    require_all(goal, REQUIRED_THRESHOLD_ENV, "goal.md explicit threshold references", failures)
    validate_runtime_evidence_map(goal, harness, failures)
    validate_unchecked_goal_items_are_mapped(goal, failures)

    if failures:
        raise AssertionError("\n".join(failures))
    print("Goal harness coverage OK")


if __name__ == "__main__":
    main()
