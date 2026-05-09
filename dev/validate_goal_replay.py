#!/usr/bin/env python3
"""Host-only smoke for validate-only goal evidence replay paths."""

from __future__ import annotations

import os
import csv
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts" / "validate_goal_h100.sh"
LLAMA_HEADER_INTS = 256
LLAMA3_MAGIC = 20240803
LLAMA_MODEL_VERSION = 5
LLAMA_STATE_MAGIC = 20240804
LLAMA_STATE_VERSION = 1


def run_harness(phase: str, env: dict[str, str], marker: str | None = None) -> None:
    merged_env = os.environ.copy()
    merged_env.update(env)
    result = subprocess.run(
        [str(HARNESS), phase],
        cwd=ROOT,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{phase} replay failed with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if marker is not None and marker not in result.stdout:
        raise RuntimeError(f"{phase} replay missing marker {marker!r}\nstdout:\n{result.stdout}")


def expect_harness_fail(phase: str, env: dict[str, str], expected: str) -> None:
    merged_env = os.environ.copy()
    merged_env.update(env)
    result = subprocess.run(
        [str(HARNESS), phase],
        cwd=ROOT,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode == 0:
        raise RuntimeError(f"{phase} unexpectedly accepted invalid replay evidence\nstdout:\n{result.stdout}")
    combined = result.stdout + result.stderr
    if expected not in combined:
        raise RuntimeError(
            f"{phase} failure did not mention {expected!r}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def write_training_log(path: Path, final_step: int, *, val_loss: float, eval_acc: float) -> Path:
    lines = [
        "s:0 trl:10.000000 lr:0.000300 norm:1.250000",
        f"s:{max(0, final_step - 1)} trl:5.000000 lr:0.000200 norm:1.000000",
        f"s:{final_step} tel:{val_loss:.6f}",
        f"s:{final_step} eval:{eval_acc:.6f}",
    ]
    return write(path, "\n".join(lines) + "\n")


def write_curve_log(path: Path, losses: list[float]) -> Path:
    lines = [f"s:{step} trl:{loss:.6f} lr:0.000300 norm:1.250000" for step, loss in enumerate(losses)]
    return write(path, "\n".join(lines) + "\n")


def make_profile_row(kernel: str, time_ms: float, read_gib: float, write_gib: float, tensor_pct: float) -> list[str]:
    row = ["0"] * 18
    row[4] = kernel
    row[10] = "sm_90a"
    row[11] = f"{read_gib:.6f}"
    row[12] = f"{write_gib:.6f}"
    row[13] = f"{time_ms:.6f}"
    row[14] = "1024"
    row[15] = "512"
    row[16] = f"{tensor_pct:.6f}"
    row[17] = "1000000"
    return row


def write_profile_csv(path: Path, tensor_pct: float = 82.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        [f"h{i}" for i in range(18)],
        [f"u{i}" for i in range(18)],
        make_profile_row("void encoder_forward_kernel(float*)", 1.0, 0.5, 0.2, tensor_pct),
        make_profile_row("void matmul_forward_kernel(float*)", 2.0, 0.6, 0.3, tensor_pct),
        make_profile_row("void layernorm_forward_kernel(float*)", 1.5, 0.4, 0.2, tensor_pct),
        make_profile_row("void fused_classifier_kernel(float*)", 1.0, 0.2, 0.1, tensor_pct),
        make_profile_row("void matmul_backward_kernel(float*)", 2.0, 0.5, 0.3, tensor_pct),
        make_profile_row("void layernorm_backward_kernel(float*)", 1.0, 0.3, 0.2, tensor_pct),
        make_profile_row("void adamw_kernel(float*)", 0.5, 0.1, 0.1, tensor_pct),
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return path


def write_llama_artifacts(output_dir: Path, step: int, *, rank: int = 0, num_processes: int = 1) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    step_tag = f"{step:08d}"
    rank_tag = f"{rank:05d}"
    (output_dir / f"DONE_{step_tag}").touch()

    model_header = [0] * LLAMA_HEADER_INTS
    model_header[0] = LLAMA3_MAGIC
    model_header[1] = LLAMA_MODEL_VERSION
    (output_dir / f"model_{step_tag}.bin").write_bytes(struct.pack("<256i", *model_header))

    state_header = [0] * LLAMA_HEADER_INTS
    state_header[0] = LLAMA_STATE_MAGIC
    state_header[1] = LLAMA_STATE_VERSION
    state_header[2] = num_processes
    state_header[3] = rank
    state_header[10] = step
    (output_dir / f"state_{step_tag}_{rank_tag}.bin").write_bytes(struct.pack("<256i", *state_header))


def write_gpt_artifacts(output_dir: Path, step: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    step_tag = f"{step:08d}"
    (output_dir / f"DONE_{step_tag}").touch()
    (output_dir / f"model_{step_tag}.bin").touch()
    (output_dir / f"state_{step_tag}_00000.bin").touch()


def write_gpt2_full_run_log(path: Path, final_step: int) -> Path:
    return write(
        path,
        "llm.kittens GPT-2 124M launch\n"
        "NPROC=8\n"
        "TRAIN_BIN=./train_gpt2cu\n"
        "OUT_DIR=log_gpt2_124M\n"
        f"MAX_STEPS={final_step}\n"
        "TRAIN_DATA_PATTERN=dev/data/fineweb10B/fineweb_train_*.bin\n"
        "VAL_DATA_PATTERN=dev/data/fineweb10B/fineweb_val_*.bin\n"
        "B=64 T=1024 D=524288 ZERO_STAGE=1 RECOMPUTE=0 MODEL=d12\n",
    )


def write_llama1b_full_run_log(path: Path, final_step: int) -> Path:
    return write(
        path,
        "llm.kittens Llama-3 1B launch\n"
        "NPROC=8\n"
        "TRAIN_BIN=./train_llama3cu\n"
        "MODEL_DESC=llama3:1B\n"
        "OUT_DIR=log_llama3_1B\n"
        f"MAX_STEPS={final_step}\n"
        "TRAIN_DATA_PATTERN=dev/data/edu_fineweb100B/edu_fineweb_train_*.bin\n"
        "VAL_DATA_PATTERN=dev/data/edu_fineweb100B/edu_fineweb_val_*.bin\n"
        "B=32 T=2048 D=524288 ZERO_STAGE=1 RECOMPUTE=0\n",
    )


def write_llama8b_full_run_log(path: Path, final_step: int, *, num_processes: int = 16) -> Path:
    return write(
        path,
        "llm.kittens Llama-3 8B Slurm launch\n"
        "SLURM_JOB_ID=12345\n"
        f"SLURM_NTASKS={num_processes}\n"
        "SLURM_JOB_NUM_NODES=2\n"
        "SLURM_NTASKS_PER_NODE=8\n"
        "MODEL_DESC=llama3:8B\n"
        "OUT_DIR=/ephemeral/data/fineweb/log_llama3_8B_multi\n"
        f"MAX_STEPS={final_step}\n"
        "TRAIN_DATA_PATH=/ephemeral/data/fineweb/bin_100B_llama3/fineweb_train_*.bin\n"
        "VAL_DATA_PATH=/ephemeral/data/fineweb/bin_100B_llama3/fineweb_val_*.bin\n"
        "SYNC_FS_PATH=/ephemeral/data/fineweb/log_llama3_8B_multi\n"
        "B=4 T=2048 D=524288 ZERO_STAGE=2 RECOMPUTE=1 INIT=fs\n",
    )


def write_synthetic_llama_checkpoint(path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "dev/download_llama3.py",
            "--write-synthetic-checkpoint",
            str(path),
            "--cpp-validate",
            "--cpp-zero-stage",
            "2",
            "--cpp-processes",
            "8",
            "--train-binary",
            "./train_llama3cu",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "synthetic Llama checkpoint creation failed\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="llmkittens_goal_replay_") as tmp:
        root = Path(tmp)
        run_harness(
            "preflight",
            {
                "PREFLIGHT_VALIDATE_ONLY": "1",
                "PREFLIGHT_LOG": str(write(root / "preflight.log", "NVIDIA H100, 9.0\nH100 preflight OK\n")),
            },
        )
        run_harness(
            "preflight",
            {
                "DEVICE_TEST_TARGET": "rtx5090",
                "PREFLIGHT_VALIDATE_ONLY": "1",
                "PREFLIGHT_LOG": str(
                    write(
                        root / "rtx5090_preflight.log",
                        "NVIDIA GeForce RTX 5090, 12.0\nRTX 5090 device preflight OK\n",
                    )
                ),
            },
        )
        expect_harness_fail(
            "preflight",
            {
                "DEVICE_TEST_TARGET": "rtx5090",
                "PREFLIGHT_VALIDATE_ONLY": "1",
                "PREFLIGHT_LOG": str(root / "preflight.log"),
            },
            "RTX 5090 device preflight OK",
        )
        run_harness(
            "cuda-runtime",
            {
                "CUDA_RUNTIME_VALIDATE_ONLY": "1",
                "CUDA_RUNTIME_LOG": str(write(root / "cuda_runtime_check.log", "CUDA runtime check passed.\n")),
            },
        )
        run_harness(
            "cuda-runtime",
            {
                "DEVICE_TEST_TARGET": "rtx5090",
                "CUDA_RUNTIME_VALIDATE_ONLY": "1",
                "CUDA_RUNTIME_LOG": str(
                    write(
                        root / "rtx5090_cuda_runtime_check.log",
                        "CUDA device target: rtx5090\nCUDA runtime check passed.\n",
                    )
                ),
            },
        )
        expect_harness_fail(
            "cuda-runtime",
            {
                "DEVICE_TEST_TARGET": "rtx5090",
                "CUDA_RUNTIME_VALIDATE_ONLY": "1",
                "CUDA_RUNTIME_LOG": str(root / "cuda_runtime_check.log"),
            },
            "CUDA device target: rtx5090",
        )

        smoke_dir = root / "smoke"
        for binary in [
            "test_matmul",
            "test_attention",
            "test_layernorm",
            "test_rope",
            "test_rmsnorm",
            "test_swiglu",
            "test_attention_gqa",
        ]:
            write(smoke_dir / f"{binary}.log", f"{binary} smoke OK\n")
        run_harness("smoke", {"SMOKE_VALIDATE_ONLY": "1", "SMOKE_LOG_DIR": str(smoke_dir)})

        run_harness(
            "gpt2",
            {
                "GPT2_RUNTIME_VALIDATE_ONLY": "1",
                "GPT2_VALIDATE_LOG": str(write(root / "gpt2_validate.log", "gpt2_validate OK\n")),
                "GPT2_PARITY_LOG": str(write(root / "test_gpt2cu.log", "test_gpt2cu OK\n")),
            },
        )
        run_harness(
            "gqa-runtime",
            {
                "GQA_RUNTIME_VALIDATE_ONLY": "1",
                "GQA_RUNTIME_LOG": str(
                    write(
                        root / "test_attention_gqa.log",
                        "GQA case T=128 backward=fallback OK\n"
                        "GQA case T=256 backward=tk OK\n"
                        "test_attention_gqa smoke OK\n",
                    )
                ),
            },
        )

        gpt2_smoke_log = write_training_log(root / "gpt2_smoke.log", 4, val_loss=4.0, eval_acc=0.25)
        run_harness(
            "gpt2-smoke",
            {
                "GPT2_SMOKE_VALIDATE_ONLY": "1",
                "GPT2_SMOKE_LOG": str(gpt2_smoke_log),
                "GPT2_SMOKE_STEPS": "4",
                "GPT2_SMOKE_MAX_VAL_LOSS": "4.5",
            },
        )
        zero3_smoke_log = write_training_log(root / "zero3_smoke.log", 1, val_loss=4.0, eval_acc=0.25)
        zero3_smoke_run_log = write(
            root / "zero3_smoke_run.log",
            "| ZeRO Stage 3: parameter shards + runtime all-gather compute layout        |\n",
        )
        run_harness(
            "zero3-smoke",
            {
                "ZERO3_SMOKE_VALIDATE_ONLY": "1",
                "ZERO3_SMOKE_LOG": str(zero3_smoke_log),
                "ZERO3_SMOKE_RUN_LOG": str(zero3_smoke_run_log),
                "ZERO3_SMOKE_STEPS": "1",
                "ZERO3_SMOKE_MAX_VAL_LOSS": "4.5",
            },
        )

        llama_resume_dir = root / "llama_resume"
        write_llama_artifacts(llama_resume_dir, 1)
        write_llama_artifacts(llama_resume_dir, 2)
        write_training_log(llama_resume_dir / "main.log", 2, val_loss=3.0, eval_acc=0.2)
        run_harness(
            "llama-resume",
            {
                "LLAMA_RESUME_VALIDATE_ONLY": "1",
                "LLAMA_RESUME_OUT": str(llama_resume_dir),
                "LLAMA_RESUME_STEPS": "2",
                "LLAMA_RESUME_MAX_VAL_LOSS": "3.5",
            },
        )

        llama1b_stability_log = write_training_log(root / "llama1b_stability.log", 4, val_loss=3.0, eval_acc=0.3)
        run_harness(
            "llama1b-stability",
            {
                "LLAMA1B_STABILITY_VALIDATE_ONLY": "1",
                "LLAMA1B_STABILITY_LOG": str(llama1b_stability_log),
                "LLAMA1B_STABILITY_STEPS": "4",
                "LLAMA1B_STABILITY_MAX_VAL_LOSS": "3.5",
                "LLAMA1B_STABILITY_MIN_HELLASWAG": "0.2",
            },
        )
        bad_llama1b_stability_eval_log = write(
            root / "llama1b_stability_bad_eval_step.log",
            "s:0 trl:10.000000 lr:0.000300 norm:1.250000\n"
            "s:3 trl:5.000000 lr:0.000200 norm:1.000000\n"
            "s:4 tel:3.000000\n"
            "s:1 eval:0.300000\n",
        )
        expect_harness_fail(
            "llama1b-stability",
            {
                "LLAMA1B_STABILITY_VALIDATE_ONLY": "1",
                "LLAMA1B_STABILITY_LOG": str(bad_llama1b_stability_eval_log),
                "LLAMA1B_STABILITY_STEPS": "4",
                "LLAMA1B_STABILITY_HELLASWAG": "0",
                "LLAMA1B_STABILITY_MAX_VAL_LOSS": "3.5",
                "LLAMA1B_STABILITY_MIN_HELLASWAG": "0.2",
            },
            "eval accuracy latest step 1 is before required step 4",
        )

        gpt2_full_dir = root / "gpt2_full"
        write_gpt_artifacts(gpt2_full_dir, 4)
        gpt2_full_log = write_training_log(gpt2_full_dir / "main.log", 4, val_loss=4.0, eval_acc=0.25)
        gpt2_full_run_log = write_gpt2_full_run_log(gpt2_full_dir / "run.log", 4)
        run_harness(
            "gpt2-full",
            {
                "GPT2_FULL_VALIDATE_ONLY": "1",
                "GPT2_FULL_OUT_DIR": str(gpt2_full_dir),
                "GPT2_FULL_LOG": str(gpt2_full_log),
                "GPT2_FULL_RUN_LOG": str(gpt2_full_run_log),
                "GPT2_FULL_FINAL_STEP": "4",
                "GPT2_FULL_EXPECTED_VAL_LOSS": "4.0",
                "GPT2_FULL_EXPECTED_HELLASWAG": "0.25",
            },
        )

        reference_log = write_curve_log(root / "single_node.log", [10.0, 8.0, 6.0])
        candidate_log = write_curve_log(root / "two_node.log", [10.01, 8.01, 6.01])
        run_harness(
            "gpt2-two-node",
            {
                "GPT2_TWO_NODE_VALIDATE_ONLY": "1",
                "GPT2_SINGLE_NODE_LOG": str(reference_log),
                "GPT2_TWO_NODE_LOG": str(candidate_log),
                "GPT2_TWO_NODE_STEPS": "3",
                "GPT2_TWO_NODE_REL_TOL": "0.01",
            },
        )

        llama1b_full_dir = root / "llama1b_full"
        write_llama_artifacts(llama1b_full_dir, 4)
        llama1b_full_log = write_training_log(llama1b_full_dir / "main.log", 4, val_loss=3.0, eval_acc=0.3)
        llama1b_full_run_log = write_llama1b_full_run_log(llama1b_full_dir / "run.log", 4)
        run_harness(
            "llama1b-full",
            {
                "LLAMA1B_FULL_VALIDATE_ONLY": "1",
                "LLAMA1B_FULL_OUT_DIR": str(llama1b_full_dir),
                "LLAMA1B_FULL_LOG": str(llama1b_full_log),
                "LLAMA1B_FULL_RUN_LOG": str(llama1b_full_run_log),
                "LLAMA1B_FULL_FINAL_STEP": "4",
                "LLAMA1B_FULL_MAX_VAL_LOSS": "3.5",
                "LLAMA1B_FULL_MIN_HELLASWAG": "0.2",
            },
        )

        llama8b_checkpoint = root / "llama3.1_8B_bf16.bin"
        write_synthetic_llama_checkpoint(llama8b_checkpoint)
        run_harness(
            "llama8b-convert",
            {
                "LLAMA8B_CONVERT_VALIDATE_ONLY": "1",
                "LLAMA8B_CHECKPOINT": str(llama8b_checkpoint),
                "LLAMA8B_CONVERT_PROCESSES": "8",
            },
        )

        llama8b_dir = root / "llama8b_full"
        write_llama_artifacts(llama8b_dir, 4, num_processes=16)
        write_training_log(llama8b_dir / "main.log", 4, val_loss=3.0, eval_acc=0.3)
        llama8b_full_run_log = write_llama8b_full_run_log(llama8b_dir / "run.log", 4)
        run_harness(
            "llama8b-full",
            {
                "LLAMA8B_FULL_VALIDATE_ONLY": "1",
                "LLAMA8B_FULL_OUT_DIR": str(llama8b_dir),
                "LLAMA8B_FULL_RUN_LOG": str(llama8b_full_run_log),
                "LLAMA8B_FULL_FINAL_STEP": "4",
                "LLAMA8B_FULL_NPROC": "16",
                "LLAMA8B_FULL_MAX_VAL_LOSS": "3.5",
                "LLAMA8B_FULL_MIN_HELLASWAG": "0.2",
            },
        )

        profile_dir = root / "profile"
        write_profile_csv(profile_dir / "profile_ge0.csv")
        write_profile_csv(profile_dir / "profile_ge1.csv")
        run_harness(
            "profile",
            {
                "PROFILE_VALIDATE_ONLY": "1",
                "PROFILE_GELU_FUSIONS": "0",
                "PROFILE_CSV_DIR": str(profile_dir),
            },
            marker="Tensor-core utilization gate:",
        )
        goal_prereq_env = {
            "PREFLIGHT_VALIDATE_ONLY": "1",
            "PREFLIGHT_LOG": str(root / "preflight.log"),
            "CUDA_RUNTIME_VALIDATE_ONLY": "1",
            "CUDA_RUNTIME_LOG": str(root / "cuda_runtime_check.log"),
            "SMOKE_VALIDATE_ONLY": "1",
            "SMOKE_LOG_DIR": str(smoke_dir),
            "GPT2_RUNTIME_VALIDATE_ONLY": "1",
            "GPT2_VALIDATE_LOG": str(root / "gpt2_validate.log"),
            "GPT2_PARITY_LOG": str(root / "test_gpt2cu.log"),
            "GQA_RUNTIME_VALIDATE_ONLY": "1",
            "GQA_RUNTIME_LOG": str(root / "test_attention_gqa.log"),
            "GPT2_SMOKE_VALIDATE_ONLY": "1",
            "GPT2_SMOKE_LOG": str(gpt2_smoke_log),
            "ZERO3_SMOKE_VALIDATE_ONLY": "1",
            "ZERO3_SMOKE_LOG": str(zero3_smoke_log),
            "ZERO3_SMOKE_RUN_LOG": str(zero3_smoke_run_log),
            "LLAMA_RESUME_VALIDATE_ONLY": "1",
            "LLAMA_RESUME_OUT": str(llama_resume_dir),
            "LLAMA_RESUME_STEPS": "2",
            "LLAMA1B_STABILITY_VALIDATE_ONLY": "1",
            "LLAMA1B_STABILITY_LOG": str(llama1b_stability_log),
            "PROFILE_VALIDATE_ONLY": "1",
            "PROFILE_GELU_FUSIONS": "0",
            "PROFILE_CSV_DIR": str(profile_dir),
            "GPT2_FULL_VALIDATE_ONLY": "1",
            "GPT2_FULL_OUT_DIR": str(gpt2_full_dir),
            "GPT2_FULL_LOG": str(gpt2_full_log),
            "GPT2_FULL_RUN_LOG": str(gpt2_full_run_log),
            "GPT2_FULL_FINAL_STEP": "4",
            "GPT2_TWO_NODE_VALIDATE_ONLY": "1",
            "GPT2_SINGLE_NODE_LOG": str(reference_log),
            "GPT2_TWO_NODE_LOG": str(candidate_log),
            "LLAMA1B_FULL_VALIDATE_ONLY": "1",
            "LLAMA1B_FULL_OUT_DIR": str(llama1b_full_dir),
            "LLAMA1B_FULL_LOG": str(llama1b_full_log),
            "LLAMA1B_FULL_RUN_LOG": str(llama1b_full_run_log),
            "LLAMA1B_FULL_FINAL_STEP": "4",
            "LLAMA8B_CONVERT_VALIDATE_ONLY": "1",
            "LLAMA8B_CHECKPOINT": str(llama8b_checkpoint),
            "LLAMA8B_FULL_VALIDATE_ONLY": "1",
            "LLAMA8B_FULL_OUT_DIR": str(llama8b_dir),
            "LLAMA8B_FULL_RUN_LOG": str(llama8b_full_run_log),
            "LLAMA8B_FULL_FINAL_STEP": "4",
            "LLAMA8B_FULL_NPROC": "16",
            "GPT2_SMOKE_MAX_VAL_LOSS": "4.5",
            "ZERO3_SMOKE_MAX_VAL_LOSS": "4.5",
            "LLAMA_RESUME_MAX_VAL_LOSS": "3.5",
            "LLAMA1B_STABILITY_MAX_VAL_LOSS": "3.5",
            "LLAMA1B_STABILITY_MIN_HELLASWAG": "0.2",
            "GPT2_FULL_EXPECTED_VAL_LOSS": "4.0",
            "GPT2_FULL_EXPECTED_HELLASWAG": "0.25",
            "GPT2_TWO_NODE_REL_TOL": "0.01",
            "LLAMA1B_FULL_MAX_VAL_LOSS": "3.5",
            "LLAMA1B_FULL_MIN_HELLASWAG": "0.2",
            "LLAMA8B_FULL_MAX_VAL_LOSS": "3.5",
            "LLAMA8B_FULL_MIN_HELLASWAG": "0.2",
        }
        run_harness("goal-complete-prereqs", goal_prereq_env)

        non_h100_env = dict(goal_prereq_env)
        non_h100_env["ALLOW_NON_H100"] = "1"
        expect_harness_fail(
            "goal-complete-prereqs",
            non_h100_env,
            "goal-complete requires real H100/sm_90-class runtime evidence",
        )

        missing_threshold_env = dict(goal_prereq_env)
        del missing_threshold_env["LLAMA8B_FULL_MIN_HELLASWAG"]
        expect_harness_fail(
            "goal-complete-prereqs",
            missing_threshold_env,
            "goal-complete requires explicit metric thresholds",
        )

        bad_numeric_threshold_env = dict(goal_prereq_env)
        bad_numeric_threshold_env["GPT2_FULL_EXPECTED_VAL_LOSS"] = "not-a-number"
        expect_harness_fail(
            "goal-complete-prereqs",
            bad_numeric_threshold_env,
            "GPT2_FULL_EXPECTED_VAL_LOSS must be numeric",
        )

        bad_fraction_threshold_env = dict(goal_prereq_env)
        bad_fraction_threshold_env["LLAMA8B_FULL_MIN_HELLASWAG"] = "1.5"
        expect_harness_fail(
            "goal-complete-prereqs",
            bad_fraction_threshold_env,
            "LLAMA8B_FULL_MIN_HELLASWAG must be in [0, 1]",
        )

        bad_gqa_log = write(
            root / "test_attention_gqa_missing_t256.log",
            "GQA case T=128 backward=fallback OK\n"
            "test_attention_gqa smoke OK\n",
        )
        bad_gqa_env = dict(goal_prereq_env)
        bad_gqa_env["GQA_RUNTIME_LOG"] = str(bad_gqa_log)
        expect_harness_fail("goal-complete-prereqs", bad_gqa_env, "GQA case T=256 backward=tk OK")

        bad_zero3_run_log = write(root / "zero3_smoke_run_missing_stage.log", "ZeRO Stage 1\n")
        bad_zero3_env = dict(goal_prereq_env)
        bad_zero3_env["ZERO3_SMOKE_RUN_LOG"] = str(bad_zero3_run_log)
        expect_harness_fail(
            "goal-complete-prereqs",
            bad_zero3_env,
            "ZeRO Stage 3: parameter shards + runtime all-gather compute layout",
        )

        missing_profile_env = dict(goal_prereq_env)
        missing_profile_env["PROFILE_CSV_DIR"] = str(root / "missing_profile")
        expect_harness_fail("goal-complete-prereqs", missing_profile_env, "profile_ge0.csv")

        missing_fused_profile_dir = root / "missing_fused_profile"
        write_profile_csv(missing_fused_profile_dir / "profile_ge0.csv")
        missing_fused_profile_env = dict(goal_prereq_env)
        missing_fused_profile_env["PROFILE_CSV_DIR"] = str(missing_fused_profile_dir)
        missing_fused_profile_env["PROFILE_GELU_FUSIONS"] = "0"
        expect_harness_fail("goal-complete-prereqs", missing_fused_profile_env, "profile_ge1.csv")

        bad_gpt2_full_run_log = write(root / "gpt2_full_bad_run.log", "llm.kittens GPT-2 124M launch\nNPROC=4\n")
        bad_gpt2_full_env = dict(goal_prereq_env)
        bad_gpt2_full_env["GPT2_FULL_RUN_LOG"] = str(bad_gpt2_full_run_log)
        expect_harness_fail("goal-complete-prereqs", bad_gpt2_full_env, "NPROC=8")

    print("Goal replay validation OK")


if __name__ == "__main__":
    main()
