#!/usr/bin/env python3
"""Source-level guards for launch-script step-count contracts."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SCRIPT_CONTRACTS = {
    "scripts/run_gpt2_124M.sh": {
        "default": 'MAX_STEPS="${MAX_STEPS:-18865}"',
        "x_arg": '-x "$MAX_STEPS"',
        "done": 'DONE_FILE="${DONE_FILE:-$OUT_DIR/DONE_$done_tag}"',
        "run_log": 'RUN_LOG="${RUN_LOG:-$OUT_DIR/run.log}"',
        "launch_marker": "llm.kittens GPT-2 124M launch",
        "launch_shape": "B=64 T=1024 D=524288 ZERO_STAGE=1 RECOMPUTE=0 MODEL=d12",
    },
    "scripts/run_gpt2_350M.sh": {
        "default": 'MAX_STEPS="${MAX_STEPS:-60000}"',
        "x_arg": '-x "$MAX_STEPS"',
        "done": 'DONE_FILE="${DONE_FILE:-$OUT_DIR/DONE_$done_tag}"',
    },
    "scripts/run_gpt2_774M.sh": {
        "default": 'MAX_STEPS="${MAX_STEPS:-286102}"',
        "x_arg": '-x "$MAX_STEPS"',
        "done": 'DONE_FILE="${DONE_FILE:-$OUT_DIR/DONE_$done_tag}"',
    },
    "scripts/run_gpt2_1558M.sh": {
        "default": 'MAX_STEPS="${MAX_STEPS:-32000}"',
        "x_arg": '-x "$MAX_STEPS"',
        "done": 'DONE_FILE="${DONE_FILE:-$OUT_DIR/DONE_$done_tag}"',
    },
    "scripts/run_gpt3_125M.sh": {
        "default": 'MAX_STEPS="${MAX_STEPS:-572204}"',
        "x_arg": '-x "$MAX_STEPS"',
        "done": 'DONE_FILE="${DONE_FILE:-$OUT_DIR/DONE_$done_tag}"',
    },
    "scripts/run_llama3_1B.sh": {
        "default": 'MAX_STEPS="${MAX_STEPS:-57220}"',
        "x_arg": '-x "$MAX_STEPS"',
        "done": 'DONE_FILE="${DONE_FILE:-$OUT_DIR/DONE_$done_tag}"',
        "run_log": 'RUN_LOG="${RUN_LOG:-$OUT_DIR/run.log}"',
        "launch_marker": "llm.kittens Llama-3 1B launch",
        "launch_shape": "B=32 T=2048 D=524288 ZERO_STAGE=1 RECOMPUTE=0",
    },
    "scripts/multi_node/run_gpt2_124M_mpi.sh": {
        "default": 'MAX_STEPS="${MAX_STEPS:-18865}"',
        "x_arg": '-x "$MAX_STEPS"',
    },
    "scripts/multi_node/run_gpt2_124M_fs.sbatch": {
        "default": 'MAX_STEPS="${MAX_STEPS:-18865}"',
        "x_arg": "-x '$MAX_STEPS'",
    },
    "scripts/multi_node/run_gpt2_124M_tcp.sbatch": {
        "default": 'MAX_STEPS="${MAX_STEPS:-18865}"',
        "x_arg": "-x '$MAX_STEPS'",
    },
    "scripts/multi_node/run_llama3_8B_fs.sbatch": {
        "default": 'MAX_STEPS="${MAX_STEPS:-57220}"',
        "x_arg": "-x '$MAX_STEPS'",
        "run_log": 'RUN_LOG="${RUN_LOG:-$OUT_DIR/run.log}"',
        "launch_marker": "llm.kittens Llama-3 8B Slurm launch",
        "launch_shape": "B=$MICRO_BATCH T=$SEQ_LEN D=$TOTAL_BATCH ZERO_STAGE=$ZERO_STAGE RECOMPUTE=1 INIT=fs",
    },
    "scripts/pyrun_gpt2_124M.sh": {
        "default": 'MAX_STEPS="${MAX_STEPS:-18865}"',
        "num_iterations": '--num_iterations "$MAX_STEPS"',
    },
}


def require_contains(path: Path, text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{path.relative_to(ROOT)} missing {label}: {needle}")


def main() -> None:
    for rel_path, checks in SCRIPT_CONTRACTS.items():
        path = ROOT / rel_path
        if not path.exists():
            raise FileNotFoundError(f"missing launch script: {path}")
        text = path.read_text()
        for label, needle in checks.items():
            require_contains(path, text, needle, label)
        if "done" in checks:
            require_contains(path, text, 'printf -v done_tag "%08d" "$MAX_STEPS"', "DONE tag derivation")
    print("Launch script source guards OK")


if __name__ == "__main__":
    main()
