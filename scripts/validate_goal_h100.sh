#!/usr/bin/env bash
set -euo pipefail

# Hardware validation harness for the unchecked gates in goal.md.
# This is intended for an H100 host with a compatible CUDA driver/runtime,
# starter-pack checkpoints, and optionally NCCL/MPI/ncu depending on the phase.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
    cat <<'USAGE'
Usage: scripts/validate_goal_h100.sh [phase ...]

Phases:
  preflight           Check H100/CUDA/NCCL/MPI prerequisites.
  compile             Build all compile-ready targets used by goal.md.
  script-syntax        Syntax-check launch/data shell scripts.
  python-syntax        Syntax-check Python data/converter/profile helpers.
  source-guards        Run source-level guards for brittle CUDA/NCCL/ZeRO/GQA/GELU-epilogue/profile/Llama-conversion/training-log/runtime-marker/build/layout contracts.
  data-artifacts       Validate prepared GPT-2/Llama train/eval data .bin metadata.
  dataloader-smoke     Run host-only C++ DataLoader/EvalLoader format smoke.
  gqa-reference        Validate CPU-only PyTorch GQA/RoPE reference math.
  gqa-runtime          Run CPU reference plus CUDA/TK GQA smoke on H100.
  profile-parser       Validate profile_gpt2cu.py CSV parser/threshold logic.
  log-tools            Validate training-log verifier pass/fail paths with synthetic logs.
  goal-replay-smoke    Validate captured-evidence replay paths with synthetic logs/artifacts.
  llama-converter-smoke
                      Validate train_llama3.py write_model with a tiny checkpoint.
  cuda-runtime        Run a tiny CUDA driver/runtime/device allocation probe.
  starter-pack        Check GPT-2 starter-pack files, checkpoint metadata, and debug-state payload.
  smoke               Run kernel smoke tests.
  gpt2                Run GPT-2 forward/parity gates: gpt2_validate + test_gpt2cu.
  gpt2-smoke          Run and log-validate a short GPT-2 tinyshakespeare smoke.
  zero3-smoke         Run and log-validate a short GPT-2 ZeRO-3 runtime smoke.
  gpt-dry             Run host-only GPT-2/GPT-3 descriptor dry-runs, including ZeRO-2/3 layouts.
  llama-dry           Run host-only Llama descriptor/checkpoint parser dry-runs, including ZeRO-2/3.
  llama-checkpoint-smoke
                      Write and validate a tiny synthetic Llama checkpoint.
  zero-guards         Check ZeRO-3 dry-runs and request diagnostics.
  llama-resume        Run and artifact/log-validate a short Llama checkpoint/resume smoke.
  llama1b-stability   Run bounded Llama-3 1B FineWeb-edu stability gate.
  profile             Run profile_gpt2cu through profile_gpt2cu.py / ncu.
  gpt2-full           Launch scripts/run_gpt2_124M.sh.
  gpt2-two-node       Compare first-100-step single-node vs two-node GPT-2 loss curves.
  llama1b-full        Launch scripts/run_llama3_1B.sh.
  llama8b-convert     Convert or validate the real gated HF Llama-3.1 8B checkpoint.
  llama8b-full        Run scripts/multi_node/run_llama3_8B_fs.sbatch via sbatch --wait, then validate final artifacts/logs.
  host-core           Run non-CUDA-runtime host-side gates using existing built binaries/artifacts.
  goal-core           preflight compile script-syntax python-syntax source-guards data-artifacts dataloader-smoke gqa-reference profile-parser log-tools goal-replay-smoke llama-converter-smoke cuda-runtime starter-pack smoke gpt2 gpt-dry llama-dry llama-checkpoint-smoke zero-guards
  goal-complete-prereqs
                      Check goal-complete thresholds, tooling, and validate-only evidence without launching jobs.
  goal-complete       Run goal-core plus ZeRO-3 smoke and all long runtime/profile/conversion/full-run gates.
  all-local           Compatibility alias for host-core.
  help                Show this message.

Environment:
  FORCE_NVCC_O=3              nvcc optimisation level for compile phase.
  MAKE_EXTRA="NO_MULTI_GPU=1" extra make arguments.
  REQUIRE_NCCL=1              require libnccl in preflight.
  REQUIRE_MPI=1               require mpirun/mpicc in preflight.
  REQUIRE_NCU=0               require ncu in preflight.
  NCCL_DIR=/path/to/nccl      custom NCCL prefix for Makefile/preflight.
  NCCL_INCLUDE_PATH=...       custom NCCL include directory.
  NCCL_LIB_PATH=...           custom NCCL library directory.
  ALLOW_NON_H100=0            allow non-H100 GPUs only for dry compile/debug phases.
  PREFLIGHT_VALIDATE_ONLY=0   Validate captured preflight stdout from PREFLIGHT_LOG instead of probing host.
  PREFLIGHT_LOG=...           Captured stdout/stderr from validate_goal_h100.sh preflight.
  CUDA_RUNTIME_VALIDATE_ONLY=0
                              Validate CUDA runtime stdout from CUDA_RUNTIME_LOG instead of running.
  CUDA_RUNTIME_LOG=...        Captured stdout/stderr from ./cuda_runtime_check.
  PROFILE_MIN_TENSOR_UTIL=70  minimum average tensor-core utilization for profile.
  PROFILE_GELU_FUSIONS="0 1"  GELU-fusion modes to profile; 1 exercises the opt-in bias+GELU epilogue.
  PROFILE_VALIDATE_ONLY=0     Validate existing profile_ge*.csv/.ncu-rep evidence instead of running ncu.
  PROFILE_REPORT_DIR=.        Directory containing profile_ge*.ncu-rep reports for validate-only mode.
  PROFILE_CSV_DIR=...          Directory containing profile_ge*.csv raw Nsight Compute exports for validate-only mode.
  DATA_ARTIFACT_ARGS=...      extra args for dev/validate_data_artifacts.py.
  GQA_RUNTIME_BINARY=...      override GQA runtime smoke binary.
  GQA_RUNTIME_VALIDATE_ONLY=0 Validate GQA runtime stdout from GQA_RUNTIME_LOG instead of running.
  GQA_RUNTIME_LOG=...         Captured stdout/stderr from test_attention_gqa.
  SMOKE_VALIDATE_ONLY=0       Validate kernel-smoke stdout logs instead of running smoke binaries.
  SMOKE_LOG_DIR=.             Directory containing test_matmul.log, test_attention.log, ...

  GPT2_TRAIN_PATTERN=...      GPT-2 train data pattern for gpt2-smoke.
  GPT2_VAL_PATTERN=...        GPT-2 val data pattern for gpt2-smoke.
  GPT2_RUNTIME_VALIDATE_ONLY=0
                              Validate gpt2_validate/test_gpt2cu stdout logs instead of running.
  GPT2_VALIDATE_LOG=...       Captured stdout/stderr from ./gpt2_validate.
  GPT2_PARITY_LOG=...         Captured stdout/stderr from ./test_gpt2cu.
  GPT_DRY_PROCESSES=8         Process count used for host-only GPT ZeRO dry-runs.
  GPT_DRY_CHECKPOINT=...      Optional GPT .bin checkpoint for header/payload dry-run.
  GPT2_SMOKE_STEPS=100        GPT-2 smoke step count.
  GPT2_SMOKE_VALIDATE_ONLY=0  Validate existing GPT-2 smoke main.log instead of launching.
  GPT2_SMOKE_LOG=...          Optional GPT-2 smoke main.log override.
  GPT2_SMOKE_MAX_VAL_LOSS=... Optional max final val loss for gpt2-smoke log check.
  ZERO3_SMOKE_NPROC=8         Process count for zero3-smoke.
  ZERO3_SMOKE_VALIDATE_ONLY=0 Validate existing ZeRO-3 smoke main.log instead of launching.
  ZERO3_SMOKE_LOG=...         Optional ZeRO-3 smoke main.log override.
  ZERO3_SMOKE_RUN_LOG=...     Captured stdout/stderr proving the ZeRO-3 stage banner.
  ZERO3_SMOKE_MAX_VAL_LOSS=... Optional max final val loss for zero3-smoke log check.
  GPT2_FULL_EXPECTED_VAL_LOSS=...
  GPT2_FULL_EXPECTED_HELLASWAG=...
  GPT2_FULL_METRIC_REL_TOL=0.005
  GPT2_FULL_VALIDATE_ONLY=0   Validate existing GPT-2 full-run log instead of launching.
  GPT2_FULL_RUN_LOG=...       Captured GPT-2 full-run launch metadata.
  GPT2_TWO_NODE_LOG=...       Optional candidate main.log override for gpt2-two-node.
  GPT2_SINGLE_NODE_LOG=...    Reference main.log; defaults to GPT2_FULL_LOG or log_gpt2_124M/main.log.
  GPT2_TWO_NODE_OUT_DIR=/ephemeral/data/fineweb/log_gpt2_124M_multi
  GPT2_TWO_NODE_VALIDATE_ONLY=0
  GPT2_TWO_NODE_SCRIPT=scripts/multi_node/run_gpt2_124M_fs.sbatch
  GPT2_TWO_NODE_SBATCH_ARGS=... Extra args passed to sbatch before the script path.
  GPT2_TWO_NODE_STEPS=100
  GPT2_TWO_NODE_REL_TOL=...   Required relative tolerance for gpt2-two-node.
  GPT2_TWO_NODE_ABS_TOL=...   Optional absolute tolerance for gpt2-two-node.

  LLAMA_TRAIN_PATTERN=...     Llama train data pattern for llama-resume.
  LLAMA_VAL_PATTERN=...       Llama val data pattern for llama-resume.
  LLAMA_DRY_PROCESSES=8       Process count used for host-only ZeRO dry-runs.
  LLAMA_DRY_CHECKPOINT=...    Optional Llama .bin checkpoint for header/payload dry-run.
  LLAMA_DRY_ZERO_STAGE=2      ZeRO stage used for optional Llama checkpoint dry-run.
  LLAMA_SYNTHETIC_CHECKPOINT=/tmp/llmkittens_synthetic_llama_bf16.bin
  LLAMA_RESUME_STEPS=2        Llama resume smoke final step count.
  LLAMA_RESUME_VALIDATE_ONLY=0
                              Validate existing Llama resume artifacts/main.log instead of launching.
  LLAMA_RESUME_LOG=...        Optional Llama resume main.log override.
  LLAMA_RESUME_MAX_VAL_LOSS=... Optional max final val loss for llama-resume.
  LLAMA1B_STABILITY_TRAIN_PATTERN=dev/data/edu_fineweb100B/edu_fineweb_train_*.bin
  LLAMA1B_STABILITY_VAL_PATTERN=dev/data/edu_fineweb100B/edu_fineweb_val_*.bin
  LLAMA1B_STABILITY_STEPS=1000
  LLAMA1B_STABILITY_NPROC=8
  LLAMA1B_STABILITY_VALIDATE_ONLY=0
                              Validate existing Llama-3 1B stability main.log instead of launching.
  LLAMA1B_STABILITY_LOG=...   Optional Llama-3 1B stability main.log override.
  LLAMA1B_STABILITY_MAX_VAL_LOSS=...
  LLAMA1B_STABILITY_MIN_HELLASWAG=...
  LLAMA1B_FULL_MAX_VAL_LOSS=...
  LLAMA1B_FULL_MIN_HELLASWAG=...
  LLAMA1B_FULL_VALIDATE_ONLY=0 Validate existing Llama-3 1B full-run log instead of launching.
  LLAMA1B_FULL_RUN_LOG=...    Captured Llama-3 1B full-run launch metadata.
  LLAMA8B_MODEL=llama3.1:8B
  LLAMA8B_OUTPUT_DIR=.
  LLAMA8B_CHECKPOINT=./llama3.1_8B_bf16.bin
  LLAMA8B_CONVERT_VALIDATE_ONLY=0
                              Require and validate LLAMA8B_CHECKPOINT instead of converting.
  LLAMA8B_CONVERT_ZERO_STAGE=2
  LLAMA8B_CONVERT_PROCESSES=16
  LLAMA8B_FULL_OUT_DIR=/ephemeral/data/fineweb/log_llama3_8B_multi
  LLAMA8B_FULL_FINAL_STEP=57220
  LLAMA8B_FULL_NPROC=16
  LLAMA8B_FULL_VALIDATE_ONLY=0
  LLAMA8B_FULL_RUN_LOG=...    Captured Llama-3 8B Slurm launch metadata.
  LLAMA8B_FULL_SBATCH_ARGS=... Extra args passed to sbatch before the script path.
  LLAMA8B_FULL_MAX_VAL_LOSS=...
  LLAMA8B_FULL_MIN_HELLASWAG=...
  ALLOW_FULL_GOAL_RUN=0       Must be 1 for goal-complete because it launches long jobs.
                              goal-complete also fail-fast checks ncu/sbatch when needed,
                              and all listed smoke/full expected metric thresholds.
USAGE
}

log() {
    printf '\n==> %s\n' "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

expect_fail() {
    local tmp status
    tmp="$(mktemp)"
    printf '+ !'
    printf ' %q' "$@"
    printf '\n'
    set +e
    "$@" >"$tmp" 2>&1
    status=$?
    set -e
    cat "$tmp"
    rm -f "$tmp"
    [ "$status" -ne 0 ] || die "command unexpectedly succeeded: $*"
}

expect_fail_contains() {
    local expected="$1"
    shift
    local tmp status
    tmp="$(mktemp)"
    printf '+ !'
    printf ' %q' "$@"
    printf '\n'
    set +e
    "$@" >"$tmp" 2>&1
    status=$?
    set -e
    cat "$tmp"
    if [ "$status" -eq 0 ]; then
        rm -f "$tmp"
        die "command unexpectedly succeeded: $*"
    fi
    if ! grep -Fq "$expected" "$tmp"; then
        rm -f "$tmp"
        die "command failed without expected text: $expected"
    fi
    rm -f "$tmp"
}

run_contains() {
    local expected="$1"
    shift
    local tmp status
    tmp="$(mktemp)"
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    set +e
    "$@" >"$tmp" 2>&1
    status=$?
    set -e
    cat "$tmp"
    if [ "$status" -ne 0 ]; then
        rm -f "$tmp"
        die "command failed: $*"
    fi
    if ! grep -Fq "$expected" "$tmp"; then
        rm -f "$tmp"
        die "command output lacked expected text: $expected"
    fi
    rm -f "$tmp"
}

run_contains_all() {
    local expected_count="$1"
    shift
    local expected=()
    local i
    for ((i = 0; i < expected_count; i++)); do
        expected+=("$1")
        shift
    done
    local tmp status text
    tmp="$(mktemp)"
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    set +e
    "$@" >"$tmp" 2>&1
    status=$?
    set -e
    cat "$tmp"
    if [ "$status" -ne 0 ]; then
        rm -f "$tmp"
        die "command failed: $*"
    fi
    for text in "${expected[@]}"; do
        if ! grep -Fq "$text" "$tmp"; then
            rm -f "$tmp"
            die "command output lacked expected text: $text"
        fi
    done
    rm -f "$tmp"
}

run_to_file_contains() {
    local expected="$1"
    local output="$2"
    shift 2
    mkdir -p "$(dirname "$output")"
    printf '+'
    printf ' %q' "$@"
    printf ' > %q\n' "$output"
    set +e
    "$@" >"$output" 2>&1
    local status=$?
    set -e
    cat "$output"
    if [ "$status" -ne 0 ]; then
        die "command failed: $*"
    fi
    if ! grep -Fq "$expected" "$output"; then
        die "command output lacked expected text: $expected"
    fi
}

have() {
    command -v "$1" >/dev/null 2>&1
}

require_cmd() {
    have "$1" || die "required command not found on PATH: $1"
}

require_cuda_tool() {
    have "$1" || [ -x "/usr/local/cuda/bin/$1" ] || die "required CUDA tool not found: $1"
}

require_file() {
    [ -f "$1" ] || die "required file is missing: $1"
}

require_file_contains() {
    local expected="$1"
    local path="$2"
    require_file "$path"
    if ! grep -Fq "$expected" "$path"; then
        die "file lacked expected text: $path :: $expected"
    fi
}

require_file_contains_all() {
    local expected_count="$1"
    shift
    local expected=()
    local i
    for ((i = 0; i < expected_count; i++)); do
        expected+=("$1")
        shift
    done
    local path="$1"
    require_file "$path"
    local needle
    for needle in "${expected[@]}"; do
        if ! grep -Fq "$needle" "$path"; then
            die "file lacked expected text: $path :: $needle"
        fi
    done
}

require_env_vars() {
    local missing=()
    local name
    for name in "$@"; do
        if [ -z "${!name:-}" ]; then
            missing+=("$name")
        fi
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        die "goal-complete requires explicit metric thresholds: ${missing[*]}"
    fi
}

require_metric_env() {
    local name="$1"
    local kind="$2"
    local value="${!name:-}"
    if ! python3 -c '
import math
import sys

name, value, kind = sys.argv[1:4]
try:
    x = float(value)
except ValueError:
    sys.stderr.write(f"{name} must be numeric, got {value!r}\n")
    sys.exit(1)
if not math.isfinite(x):
    sys.stderr.write(f"{name} must be finite, got {value!r}\n")
    sys.exit(1)
if kind == "positive" and not x > 0.0:
    sys.stderr.write(f"{name} must be > 0, got {value!r}\n")
    sys.exit(1)
if kind == "fraction" and not 0.0 <= x <= 1.0:
    sys.stderr.write(f"{name} must be in [0, 1], got {value!r}\n")
    sys.exit(1)
' "$name" "$value" "$kind"; then
        die "invalid metric threshold: $name=$value"
    fi
}

require_goal_metric_thresholds() {
    require_metric_env GPT2_SMOKE_MAX_VAL_LOSS positive
    require_metric_env ZERO3_SMOKE_MAX_VAL_LOSS positive
    require_metric_env LLAMA_RESUME_MAX_VAL_LOSS positive
    require_metric_env LLAMA1B_STABILITY_MAX_VAL_LOSS positive
    require_metric_env LLAMA1B_STABILITY_MIN_HELLASWAG fraction
    require_metric_env GPT2_FULL_EXPECTED_VAL_LOSS positive
    require_metric_env GPT2_FULL_EXPECTED_HELLASWAG fraction
    require_metric_env GPT2_TWO_NODE_REL_TOL positive
    require_metric_env LLAMA1B_FULL_MAX_VAL_LOSS positive
    require_metric_env LLAMA1B_FULL_MIN_HELLASWAG fraction
    require_metric_env LLAMA8B_FULL_MAX_VAL_LOSS positive
    require_metric_env LLAMA8B_FULL_MIN_HELLASWAG fraction
    if [ -n "${GPT2_FULL_METRIC_REL_TOL:-}" ]; then
        require_metric_env GPT2_FULL_METRIC_REL_TOL positive
    fi
}

require_glob() {
    compgen -G "$1" >/dev/null || die "required file pattern matched nothing: $1"
}

require_llama_checkpoint_step() {
    local out_dir="$1"
    local step="$2"
    local step_tag
    printf -v step_tag "%08d" "$step"
    require_file "$out_dir/DONE_${step_tag}"
    require_file "$out_dir/model_${step_tag}.bin"
    require_file "$out_dir/state_${step_tag}_00000.bin"
}

require_gpt_checkpoint_step() {
    local out_dir="$1"
    local step="$2"
    local step_tag
    printf -v step_tag "%08d" "$step"
    require_file "$out_dir/DONE_${step_tag}"
    require_file "$out_dir/model_${step_tag}.bin"
    require_file "$out_dir/state_${step_tag}_00000.bin"
}

require_gpt2_full_run_log() {
    local run_log="$1"
    local final_step="$2"
    require_file_contains_all 5 \
        "llm.kittens GPT-2 124M launch" \
        "NPROC=8" \
        "MAX_STEPS=$final_step" \
        "TRAIN_DATA_PATTERN=dev/data/fineweb10B/fineweb_train_*.bin" \
        "B=64 T=1024 D=524288 ZERO_STAGE=1 RECOMPUTE=0 MODEL=d12" \
        "$run_log"
}

require_llama1b_full_run_log() {
    local run_log="$1"
    local final_step="$2"
    require_file_contains_all 5 \
        "llm.kittens Llama-3 1B launch" \
        "NPROC=8" \
        "MODEL_DESC=llama3:1B" \
        "MAX_STEPS=$final_step" \
        "B=32 T=2048 D=524288 ZERO_STAGE=1 RECOMPUTE=0" \
        "$run_log"
}

require_llama8b_full_run_log() {
    local run_log="$1"
    local final_step="$2"
    local nproc="$3"
    require_file_contains_all 7 \
        "llm.kittens Llama-3 8B Slurm launch" \
        "SLURM_NTASKS=$nproc" \
        "SLURM_JOB_NUM_NODES=2" \
        "MODEL_DESC=llama3:8B" \
        "MAX_STEPS=$final_step" \
        "SYNC_FS_PATH=" \
        "B=4 T=2048 D=524288 ZERO_STAGE=2 RECOMPUTE=1 INIT=fs" \
        "$run_log"
}

require_goal_complete_profile_evidence() {
    local fusion
    for fusion in 0 1; do
        if [ -n "${PROFILE_CSV_DIR:-}" ]; then
            require_file "${PROFILE_CSV_DIR}/profile_ge${fusion}.csv"
        else
            require_file "${PROFILE_REPORT_DIR:-.}/profile_ge${fusion}.ncu-rep"
        fi
    done
}

require_nccl() {
    local inc="${NCCL_INCLUDE_PATH:-}"
    local lib="${NCCL_LIB_PATH:-}"

    if [ -n "${NCCL_DIR:-}" ]; then
        [ -n "$inc" ] || inc="${NCCL_DIR}/include"
        if [ -z "$lib" ]; then
            if [ -d "${NCCL_DIR}/lib64" ]; then
                lib="${NCCL_DIR}/lib64"
            else
                lib="${NCCL_DIR}/lib"
            fi
        fi
    fi

    if [ -n "$inc" ] || [ -n "$lib" ]; then
        [ -n "$inc" ] || die "NCCL include path is empty; set NCCL_INCLUDE_PATH or NCCL_DIR"
        [ -n "$lib" ] || die "NCCL library path is empty; set NCCL_LIB_PATH or NCCL_DIR"
        [ -f "$inc/nccl.h" ] || die "nccl.h not found at $inc/nccl.h"
        compgen -G "$lib/libnccl.so*" >/dev/null || die "libnccl.so not found under $lib"
        return
    fi

    ldconfig -p 2>/dev/null | grep -q 'libnccl' || die "libnccl not visible to ldconfig"
    [ -f /usr/include/nccl.h ] || [ -f /usr/local/cuda/include/nccl.h ] || die "nccl.h not found in /usr/include or /usr/local/cuda/include"
}

make_args() {
    local args=("FORCE_NVCC_O=${FORCE_NVCC_O:-3}")
    if [ -n "${MAKE_EXTRA:-}" ]; then
        # shellcheck disable=SC2206
        local extra=( ${MAKE_EXTRA} )
        args+=("${extra[@]}")
    fi
    printf '%s\n' "${args[@]}"
}

phase_preflight() {
    log "preflight"
    if [ "${PREFLIGHT_VALIDATE_ONLY:-0}" = "1" ]; then
        require_file_contains "H100 preflight OK" "${PREFLIGHT_LOG:-preflight.log}"
        return
    fi
    require_cmd make
    require_cmd python3
    require_cmd nvidia-smi
    require_cuda_tool nvcc

    local gpu_csv
    if ! gpu_csv="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>&1)"; then
        printf '%s\n' "$gpu_csv" >&2
        die "nvidia-smi could not query GPU inventory"
    fi
    printf '%s\n' "$gpu_csv"
    if [ "${ALLOW_NON_H100:-0}" != "1" ]; then
        printf '%s\n' "$gpu_csv" | awk -F, '
            {
                name=$1; cap=$2;
                gsub(/^[ \t]+|[ \t]+$/, "", name);
                gsub(/^[ \t]+|[ \t]+$/, "", cap);
                if (name !~ /(H100|H200|GH200)/ && (cap + 0 < 9.0 || cap + 0 >= 10.0)) bad=1;
            }
            END { exit bad ? 1 : 0 }
        ' || die "goal.md runtime gates require H100/sm_90-class GPUs; set ALLOW_NON_H100=1 only for dry compile/debug runs"
    fi

    if [ "${REQUIRE_NCCL:-1}" = "1" ]; then
        require_nccl
    fi
    if [ "${REQUIRE_MPI:-1}" = "1" ]; then
        require_cmd mpirun
        require_cmd mpicc
    fi
    if [ "${REQUIRE_NCU:-0}" = "1" ]; then
        require_cuda_tool ncu
    fi
    printf 'H100 preflight OK\n'
}

phase_compile() {
    log "compile"
    local args
    mapfile -t args < <(make_args)
    run make -B \
        test_matmul test_attention test_layernorm test_rope test_rmsnorm test_swiglu test_attention_gqa \
        test_dataloader cuda_runtime_check train_gpt2cu test_gpt2cu gpt2_validate profile_gpt2cu train_llama3cu \
        "${args[@]}"
}

phase_script_syntax() {
    log "script syntax"
    local scripts=(
        dev/download_starter_pack.sh
        dev/data/fineweb.sh
        dev/data/edu_fineweb.sh
        scripts/run_gpt2_124M.sh
        scripts/run_gpt2_350M.sh
        scripts/run_gpt2_774M.sh
        scripts/run_gpt2_1558M.sh
        scripts/run_gpt3_125M.sh
        scripts/run_llama3_1B.sh
        scripts/pyrun_gpt2_124M.sh
        scripts/multi_node/run_gpt2_124M_mpi.sh
        scripts/multi_node/run_gpt2_124M_fs.sbatch
        scripts/multi_node/run_gpt2_124M_tcp.sbatch
        scripts/multi_node/run_llama3_8B_fs.sbatch
    )
    for script in "${scripts[@]}"; do
        require_file "$script"
        run bash -n "$script"
    done
}

phase_python_syntax() {
    log "Python syntax"
    local modules=(
        train_gpt2.py
        train_llama3.py
        profile_gpt2cu.py
        dev/download_llama3.py
        dev/compare_training_logs.py
        dev/validate_launch_scripts.py
        dev/validate_log_tools.py
        dev/validate_goal_replay.py
        dev/validate_attention_gqa_reference.py
        dev/validate_build_contracts.py
        dev/validate_data_artifacts.py
        dev/validate_epilogue_source.py
        dev/validate_gpt2_starter_pack.py
        dev/validate_gqa_source.py
        dev/validate_goal_harness_coverage.py
        dev/validate_llama_conversion_source.py
        dev/validate_llama3_converter.py
        dev/validate_llama_checkpoint_artifacts.py
        dev/validate_nccl_source.py
        dev/validate_profile_source.py
        dev/validate_profile_parser.py
        dev/validate_runtime_markers.py
        dev/validate_training_source.py
        dev/validate_training_log.py
        dev/validate_zero_layout.py
        dev/data/data_common.py
        dev/data/fineweb.py
        dev/data/hellaswag.py
        dev/data/mmlu.py
        dev/data/tinyshakespeare.py
        dev/data/tinystories.py
    )
    for module in "${modules[@]}"; do
        require_file "$module"
    done
    run python3 -m py_compile "${modules[@]}"
}

phase_data_artifacts() {
    log "data artifacts"
    require_file dev/validate_data_artifacts.py
    run_contains "Data artifact self-test OK" python3 dev/validate_data_artifacts.py --self-test
    local args=()
    if [ -n "${DATA_ARTIFACT_ARGS:-}" ]; then
        # shellcheck disable=SC2206
        args=( ${DATA_ARTIFACT_ARGS} )
    fi
    run_contains "Data artifact metadata OK" python3 dev/validate_data_artifacts.py "${args[@]}"
}

phase_source_guards() {
    log "source guards"
    require_file dev/validate_nccl_source.py
    require_file dev/validate_launch_scripts.py
    require_file dev/validate_build_contracts.py
    require_file dev/validate_epilogue_source.py
    require_file dev/validate_gqa_source.py
    require_file dev/validate_runtime_markers.py
    require_file dev/validate_training_source.py
    require_file dev/validate_profile_source.py
    require_file dev/validate_llama_conversion_source.py
    require_file dev/validate_goal_harness_coverage.py
    require_file dev/validate_zero_layout.py
    run_contains "NCCL/ZeRO source guards OK" python3 dev/validate_nccl_source.py
    run_contains "Launch script source guards OK" python3 dev/validate_launch_scripts.py
    run_contains "Build contract source guards OK" python3 dev/validate_build_contracts.py
    run_contains "GELU epilogue source guards OK" python3 dev/validate_epilogue_source.py
    run_contains "GQA source guards OK" python3 dev/validate_gqa_source.py
    run_contains "Runtime marker source guards OK" python3 dev/validate_runtime_markers.py
    run_contains "Training evidence source guards OK" python3 dev/validate_training_source.py
    run_contains "Profile source guards OK" python3 dev/validate_profile_source.py
    run_contains "Llama conversion source guards OK" python3 dev/validate_llama_conversion_source.py
    run_contains "Goal harness coverage OK" python3 dev/validate_goal_harness_coverage.py
    run_contains "ZeRO shard layout validation OK" python3 dev/validate_zero_layout.py
}

phase_dataloader_smoke() {
    log "DataLoader smoke"
    require_file ./test_dataloader
    run_contains "DataLoader/EvalLoader smoke OK" ./test_dataloader
}

phase_gqa_reference() {
    log "GQA reference"
    require_file dev/validate_attention_gqa_reference.py
    run_contains "GQA reference validation OK" python3 dev/validate_attention_gqa_reference.py
}

phase_gqa_runtime() {
    log "GQA runtime"
    phase_gqa_reference
    local binary="${GQA_RUNTIME_BINARY:-./test_attention_gqa}"
    if [ "${GQA_RUNTIME_VALIDATE_ONLY:-0}" = "1" ]; then
        local log_path="${GQA_RUNTIME_LOG:-test_attention_gqa.log}"
        require_file_contains_all 3 \
            "GQA case T=128 backward=fallback OK" \
            "GQA case T=256 backward=tk OK" \
            "test_attention_gqa smoke OK" \
            "$log_path"
    else
        require_file "$binary"
        run_contains_all 3 \
            "GQA case T=128 backward=fallback OK" \
            "GQA case T=256 backward=tk OK" \
            "test_attention_gqa smoke OK" \
            "$binary"
    fi
}

phase_profile_parser() {
    log "profile parser"
    require_file dev/validate_profile_parser.py
    run_contains "profile parser validation OK" python3 dev/validate_profile_parser.py
}

phase_log_tools() {
    log "training log tools"
    require_file dev/validate_log_tools.py
    run_contains "Training log tool validation OK" python3 dev/validate_log_tools.py
}

phase_goal_replay_smoke() {
    log "goal replay smoke"
    require_file dev/validate_goal_replay.py
    require_file ./train_llama3cu
    run_contains "Goal replay validation OK" python3 dev/validate_goal_replay.py
}

phase_llama_converter_smoke() {
    log "Llama converter smoke"
    require_file dev/validate_llama3_converter.py
    require_file ./train_llama3cu
    run_contains "Llama converter writer validation OK" \
        python3 dev/validate_llama3_converter.py --cpp-validate --train-binary ./train_llama3cu
}

phase_cuda_runtime() {
    log "CUDA runtime"
    if [ "${CUDA_RUNTIME_VALIDATE_ONLY:-0}" = "1" ]; then
        require_file_contains "CUDA runtime check passed." "${CUDA_RUNTIME_LOG:-cuda_runtime_check.log}"
    else
        require_file ./cuda_runtime_check
        run_contains "CUDA runtime check passed." ./cuda_runtime_check
    fi
}

phase_starter_pack() {
    log "GPT-2 starter pack"
    require_file dev/validate_gpt2_starter_pack.py
    run_contains "GPT-2 starter-pack self-test OK" python3 dev/validate_gpt2_starter_pack.py --self-test
    require_file gpt2_tokenizer.bin
    require_file gpt2_124M.bin
    require_file gpt2_124M_bf16.bin
    require_file gpt2_124M_debug_state.bin
    require_file ./train_gpt2cu
    run_contains "GPT-2 starter-pack metadata OK" python3 dev/validate_gpt2_starter_pack.py
    local dry_processes="${GPT_DRY_PROCESSES:-8}"
    run_contains "GPT dry run: ZeRO-1 layout validated" \
        ./train_gpt2cu -e gpt2_124M_bf16.bin -x 0 -z 1 -pn "$dry_processes"
}

phase_smoke() {
    log "kernel smoke tests"
    for bin in test_matmul test_attention test_layernorm test_rope test_rmsnorm test_swiglu test_attention_gqa; do
        if [ "${SMOKE_VALIDATE_ONLY:-0}" = "1" ]; then
            require_file_contains "$bin smoke OK" "${SMOKE_LOG_DIR:-.}/$bin.log"
        else
            require_file "./$bin"
            run_contains "$bin smoke OK" "./$bin"
        fi
    done
}

phase_gpt2() {
    log "GPT-2 forward/parity gates"
    if [ "${GPT2_RUNTIME_VALIDATE_ONLY:-0}" = "1" ]; then
        require_file_contains "gpt2_validate OK" "${GPT2_VALIDATE_LOG:-gpt2_validate.log}"
        require_file_contains "test_gpt2cu OK" "${GPT2_PARITY_LOG:-test_gpt2cu.log}"
    else
        require_file gpt2_124M_debug_state.bin
        require_file ./gpt2_validate
        require_file ./test_gpt2cu
        run_contains "gpt2_validate OK" ./gpt2_validate
        run_contains "test_gpt2cu OK" ./test_gpt2cu
    fi
}

phase_gpt2_smoke() {
    log "GPT-2 short training smoke"
    require_file dev/validate_training_log.py
    local train_pattern="${GPT2_TRAIN_PATTERN:-dev/data/tinyshakespeare/tiny_shakespeare_train.bin}"
    local val_pattern="${GPT2_VAL_PATTERN:-dev/data/tinyshakespeare/tiny_shakespeare_val.bin}"
    local out_dir="${GPT2_SMOKE_OUT:-log_goal_gpt2_smoke}"
    local steps="${GPT2_SMOKE_STEPS:-100}"
    local log_path="${GPT2_SMOKE_LOG:-$out_dir/main.log}"
    if [ "${GPT2_SMOKE_VALIDATE_ONLY:-0}" != "1" ]; then
        require_file gpt2_124M_bf16.bin
        require_glob "$train_pattern"
        require_glob "$val_pattern"
        run ./train_gpt2cu \
            -i "$train_pattern" \
            -j "$val_pattern" \
            -e gpt2_124M_bf16.bin \
            -o "$out_dir" \
            -b "${GPT2_SMOKE_B:-1}" \
            -t "${GPT2_SMOKE_T:-128}" \
            -d "${GPT2_SMOKE_D:-128}" \
            -x "$steps" \
            -v "${GPT2_SMOKE_VAL_EVERY:-20}" \
            -m "${GPT2_SMOKE_VAL_STEPS:-2}" \
            -s 0 \
            -g 16 \
            -h 0 \
            -z 0 \
            -n 0
    fi
    require_file "$log_path"
    local args=(
        --log "$log_path"
        --val-final-step "$steps"
        --train-final-step "$((steps - 1))"
        --require-val
        --require-train
        --require-train-loss-decrease
    )
    if [ -n "${GPT2_SMOKE_MAX_VAL_LOSS:-}" ]; then
        args+=(--max-val-loss "$GPT2_SMOKE_MAX_VAL_LOSS")
    fi
    run_contains "Training log validation OK" python3 dev/validate_training_log.py "${args[@]}"
}

phase_zero3_smoke() {
    log "ZeRO-3 GPT-2 runtime smoke"
    require_file dev/validate_training_log.py
    local train_pattern="${ZERO3_SMOKE_TRAIN_PATTERN:-dev/data/tinyshakespeare/tiny_shakespeare_train.bin}"
    local val_pattern="${ZERO3_SMOKE_VAL_PATTERN:-dev/data/tinyshakespeare/tiny_shakespeare_val.bin}"
    local out_dir="${ZERO3_SMOKE_OUT:-log_goal_zero3_smoke}"
    local steps="${ZERO3_SMOKE_STEPS:-1}"
    local nproc="${ZERO3_SMOKE_NPROC:-8}"
    local B="${ZERO3_SMOKE_B:-1}"
    local T="${ZERO3_SMOKE_T:-128}"
    local D="${ZERO3_SMOKE_D:-$((B * T * nproc))}"
    local log_path="${ZERO3_SMOKE_LOG:-$out_dir/main.log}"
    local run_log_path="${ZERO3_SMOKE_RUN_LOG:-$out_dir/run.log}"
    local zero3_marker="ZeRO Stage 3: parameter shards + runtime all-gather compute layout"
    if [ "${ZERO3_SMOKE_VALIDATE_ONLY:-0}" != "1" ]; then
        require_file gpt2_124M_bf16.bin
        require_glob "$train_pattern"
        require_glob "$val_pattern"
        if [ "$nproc" = "1" ]; then
            run_to_file_contains "$zero3_marker" "$run_log_path" ./train_gpt2cu \
                -i "$train_pattern" \
                -j "$val_pattern" \
                -e gpt2_124M_bf16.bin \
                -o "$out_dir" \
                -b "$B" \
                -t "$T" \
                -d "$D" \
                -x "$steps" \
                -v "${ZERO3_SMOKE_VAL_EVERY:-1}" \
                -m "${ZERO3_SMOKE_VAL_STEPS:-1}" \
                -s 0 \
                -g 16 \
                -h 0 \
                -z 3 \
                -n 0
        else
            require_cmd "${ZERO3_SMOKE_MPIEXEC:-mpirun}"
            run_to_file_contains "$zero3_marker" "$run_log_path" \
                "${ZERO3_SMOKE_MPIEXEC:-mpirun}" -np "$nproc" ./train_gpt2cu \
                -i "$train_pattern" \
                -j "$val_pattern" \
                -e gpt2_124M_bf16.bin \
                -o "$out_dir" \
                -b "$B" \
                -t "$T" \
                -d "$D" \
                -x "$steps" \
                -v "${ZERO3_SMOKE_VAL_EVERY:-1}" \
                -m "${ZERO3_SMOKE_VAL_STEPS:-1}" \
                -s 0 \
                -g 16 \
                -h 0 \
                -z 3 \
                -n 0
        fi
    else
        require_file_contains "$zero3_marker" "$run_log_path"
    fi
    require_file "$log_path"
    local args=(
        --log "$log_path"
        --val-final-step "$steps"
        --train-final-step "$((steps - 1))"
        --require-val
        --require-train
    )
    if [ -n "${ZERO3_SMOKE_MAX_VAL_LOSS:-}" ]; then
        args+=(--max-val-loss "$ZERO3_SMOKE_MAX_VAL_LOSS")
    fi
    run_contains "Training log validation OK" python3 dev/validate_training_log.py "${args[@]}"
}

phase_gpt_dry() {
    log "GPT host-only parser dry-runs"
    require_file ./train_gpt2cu
    local dry_processes="${GPT_DRY_PROCESSES:-8}"
    run_contains "GPT dry run: ZeRO-1 layout validated" \
        ./train_gpt2cu -e gpt2:d12 -x 0 -z 1 -pn "$dry_processes"
    run_contains "GPT dry run: ZeRO-3 layout validated" \
        ./train_gpt2cu -e gpt2:d12 -x 0 -z 3 -pn "$dry_processes"
    if [ -n "${GPT_DRY_CHECKPOINT:-}" ]; then
        require_file "$GPT_DRY_CHECKPOINT"
        run_contains "GPT dry run: ZeRO-1 layout validated" \
            ./train_gpt2cu -e "$GPT_DRY_CHECKPOINT" -x 0 -z 1 -pn "$dry_processes"
    fi
    for desc in gpt3:c384 gpt3:c768 gpt3:c1024 gpt3:c1536 gpt3:c2048 gpt3:c2560 gpt3:c4096 gpt3:c5120 gpt3:c12288; do
        local channels="${desc#gpt3:c}"
        run_contains_all 3 \
            "GPT dry run: source=$desc" \
            "channels=$channels" \
            "GPT dry run: ZeRO-2 layout validated" \
            ./train_gpt2cu -e "$desc" -t 2048 -x 0 -z 2 -pn "$dry_processes"
    done
}

phase_llama_dry() {
    log "Llama host-only parser dry-runs"
    require_file ./train_llama3cu
    local dry_processes="${LLAMA_DRY_PROCESSES:-8}"
    run_contains "train_llama3cu dry run: ZeRO-1 shard layout validated" \
        ./train_llama3cu -e llama3:1B -x 0 -z 1 -pn "$dry_processes"
    run_contains_all 2 \
        "Llama config source   | llama3:8B" \
        "train_llama3cu dry run: ZeRO-1 shard layout validated" \
        ./train_llama3cu -e llama3:8B -x 0 -z 1 -pn "$dry_processes"
    run_contains_all 2 \
        "Llama config source   | llama3:8B" \
        "train_llama3cu dry run: ZeRO-2 shard layout validated" \
        ./train_llama3cu -e llama3:8B -x 0 -z 2 -pn "$dry_processes"
    run_contains_all 2 \
        "Llama config source   | llama3.1:8B" \
        "train_llama3cu dry run: ZeRO-2 shard layout validated" \
        ./train_llama3cu -e llama3.1:8B -x 0 -z 2 -pn "$dry_processes"
    run_contains "train_llama3cu dry run: ZeRO-3 shard layout validated" \
        ./train_llama3cu -e llama3.1:8B -x 0 -z 3 -pn "$dry_processes"
    if [ -n "${LLAMA_DRY_CHECKPOINT:-}" ]; then
        require_file "$LLAMA_DRY_CHECKPOINT"
        local expected="train_llama3cu dry run: checkpoint/config parsed"
        if [ "${LLAMA_DRY_ZERO_STAGE:-2}" != "0" ]; then
            expected="train_llama3cu dry run: ZeRO-${LLAMA_DRY_ZERO_STAGE:-2} shard layout validated"
        fi
        run_contains "$expected" \
            ./train_llama3cu \
            -e "$LLAMA_DRY_CHECKPOINT" \
            -x 0 \
            -z "${LLAMA_DRY_ZERO_STAGE:-2}" \
            -pn "$dry_processes"
    fi
}

phase_llama_checkpoint_smoke() {
    log "Llama synthetic checkpoint smoke"
    require_file dev/download_llama3.py
    require_file dev/validate_llama_checkpoint_artifacts.py
    require_file ./train_llama3cu
    local checkpoint="${LLAMA_SYNTHETIC_CHECKPOINT:-/tmp/llmkittens_synthetic_llama_bf16.bin}"
    local zero_stage="${LLAMA_DRY_ZERO_STAGE:-2}"
    local dry_processes="${LLAMA_DRY_PROCESSES:-8}"
    local expected="train_llama3cu dry run: checkpoint/config parsed"
    if [ "$zero_stage" != "0" ]; then
        expected="train_llama3cu dry run: ZeRO-${zero_stage} shard layout validated"
    fi
    run_contains "$expected" \
        python3 dev/download_llama3.py \
        --write-synthetic-checkpoint "$checkpoint" \
        --cpp-validate \
        --cpp-zero-stage "$zero_stage" \
        --cpp-processes "$dry_processes" \
        --train-binary ./train_llama3cu
    run_contains "Llama checkpoint artifact self-test OK" \
        python3 dev/validate_llama_checkpoint_artifacts.py --self-test
}

phase_zero_guards() {
    log "ZeRO dry-run and request guard checks"
    require_file ./train_gpt2cu
    require_file ./train_llama3cu
    run_contains "GPT dry run: ZeRO-3 layout validated" \
        ./train_gpt2cu -e gpt2:d12 -x 0 -z 3 -pn "${ZERO_GUARD_PROCESSES:-8}"
    run_contains "train_llama3cu dry run: ZeRO-3 shard layout validated" \
        ./train_llama3cu -e llama3:1B -x 0 -z 3 -pn "${ZERO_GUARD_PROCESSES:-8}"
    expect_fail_contains \
        "supports only ZeRO-0, ZeRO-1, ZeRO-2, and ZeRO-3" \
        ./train_gpt2cu -e gpt2:d12 -x 0 -z 4
    expect_fail_contains \
        "supports only ZeRO-0, ZeRO-1, ZeRO-2, and ZeRO-3" \
        ./train_llama3cu -e llama3:1B -x 0 -z 4
    expect_fail_contains \
        "cannot be evenly partitioned across 5 processes" \
        ./train_gpt2cu -e gpt2:d12 -x 0 -z 2 -pn 5
    expect_fail_contains \
        "cannot be evenly partitioned across 7 processes" \
        ./train_llama3cu -e llama3:1B -x 0 -z 2 -pn 7
}

phase_llama_resume() {
    log "Llama checkpoint/resume smoke"
    require_file dev/validate_training_log.py
    require_file dev/validate_llama_checkpoint_artifacts.py
    local train_pattern="${LLAMA_TRAIN_PATTERN:-dev/data/tinyshakespeare/tiny_shakespeare_train.bin}"
    local val_pattern="${LLAMA_VAL_PATTERN:-dev/data/tinyshakespeare/tiny_shakespeare_val.bin}"
    local out="${LLAMA_RESUME_OUT:-log_goal_llama_resume}"
    local final_step="${LLAMA_RESUME_STEPS:-2}"
    local log_path="${LLAMA_RESUME_LOG:-$out/main.log}"
    [ "$final_step" -ge 1 ] || die "LLAMA_RESUME_STEPS must be at least 1"
    if [ "${LLAMA_RESUME_VALIDATE_ONLY:-0}" != "1" ]; then
        require_glob "$train_pattern"
        require_glob "$val_pattern"
        run ./train_llama3cu \
            -i "$train_pattern" \
            -j "$val_pattern" \
            -o "$out" \
            -b "${LLAMA_RESUME_B:-1}" \
            -t "${LLAMA_RESUME_T:-128}" \
            -d "${LLAMA_RESUME_D:-128}" \
            -x 1 \
            -v 1 \
            -m 1 \
            -s 0 \
            -g 16 \
            -h 0 \
            -z 0 \
            -n 1 \
            -nk 2 \
            -y 0 \
            -e "${LLAMA_RESUME_MODEL:-llama3:1B}"
    fi
    require_llama_checkpoint_step "$out" 1
    run_contains "Llama checkpoint artifacts OK" \
        python3 dev/validate_llama_checkpoint_artifacts.py \
        --output-dir "$out" \
        --step 1 \
        --rank 0 \
        --num-processes 1
    if [ "${LLAMA_RESUME_VALIDATE_ONLY:-0}" != "1" ]; then
        run ./train_llama3cu \
            -i "$train_pattern" \
            -j "$val_pattern" \
            -o "$out" \
            -b "${LLAMA_RESUME_B:-1}" \
            -t "${LLAMA_RESUME_T:-128}" \
            -d "${LLAMA_RESUME_D:-128}" \
            -x "$final_step" \
            -v 1 \
            -m 1 \
            -s 0 \
            -g 16 \
            -h 0 \
            -z 0 \
            -n 1 \
            -nk 2 \
            -y 1 \
            -e "${LLAMA_RESUME_MODEL:-llama3:1B}"
    fi
    require_llama_checkpoint_step "$out" "$final_step"
    run_contains "Llama checkpoint artifacts OK" \
        python3 dev/validate_llama_checkpoint_artifacts.py \
        --output-dir "$out" \
        --step "$final_step" \
        --rank 0 \
        --num-processes 1
    require_file "$log_path"
    local args=(
        --log "$log_path"
        --val-final-step "$final_step"
        --train-final-step "$((final_step - 1))"
        --require-val
        --require-train
    )
    if [ -n "${LLAMA_RESUME_MAX_VAL_LOSS:-}" ]; then
        args+=(--max-val-loss "$LLAMA_RESUME_MAX_VAL_LOSS")
    fi
    run_contains "Training log validation OK" python3 dev/validate_training_log.py "${args[@]}"
}

phase_llama1b_stability() {
    log "Llama-3 1B FineWeb-edu stability"
    require_file dev/validate_training_log.py
    local train_pattern="${LLAMA1B_STABILITY_TRAIN_PATTERN:-dev/data/edu_fineweb100B/edu_fineweb_train_*.bin}"
    local val_pattern="${LLAMA1B_STABILITY_VAL_PATTERN:-dev/data/edu_fineweb100B/edu_fineweb_val_*.bin}"
    local out_dir="${LLAMA1B_STABILITY_OUT:-log_goal_llama1b_stability}"
    local steps="${LLAMA1B_STABILITY_STEPS:-1000}"
    local log_path="${LLAMA1B_STABILITY_LOG:-$out_dir/main.log}"

    if [ "${LLAMA1B_STABILITY_VALIDATE_ONLY:-0}" != "1" ]; then
        require_file ./train_llama3cu
        require_glob "$train_pattern"
        require_glob "$val_pattern"
        if [ "${LLAMA1B_STABILITY_HELLASWAG:-1}" = "1" ]; then
            require_file dev/data/hellaswag/hellaswag_val_llama3.bin
        fi

        local mpirun="${MPIRUN:-mpirun}"
        require_cmd "$mpirun"
        run "$mpirun" -np "${LLAMA1B_STABILITY_NPROC:-8}" ./train_llama3cu \
            -i "$train_pattern" \
            -j "$val_pattern" \
            -o "$out_dir" \
            -v "${LLAMA1B_STABILITY_VAL_EVERY:-250}" \
            -m "${LLAMA1B_STABILITY_VAL_STEPS:-20}" \
            -s "${LLAMA1B_STABILITY_SAMPLE_EVERY:-20000}" \
            -g "${LLAMA1B_STABILITY_GEN_TOKENS:-144}" \
            -h "${LLAMA1B_STABILITY_HELLASWAG:-1}" \
            -b "${LLAMA1B_STABILITY_B:-32}" \
            -t "${LLAMA1B_STABILITY_T:-2048}" \
            -d "${LLAMA1B_STABILITY_D:-524288}" \
            -r 0 \
            -z "${LLAMA1B_STABILITY_ZERO_STAGE:-1}" \
            -c 0.1 \
            -l "${LLAMA1B_STABILITY_LR:-0.0003}" \
            -q 0.1 \
            -u "${LLAMA1B_STABILITY_WARMUP:-2000}" \
            -n "${LLAMA1B_STABILITY_CHECKPOINT_EVERY:-5000}" \
            -nk 5 \
            -nm 50000 \
            -sl 7.0 \
            -sg 7.0 \
            -y "${LLAMA1B_STABILITY_RESUME:-0}" \
            -x "$steps" \
            -e "${LLAMA1B_STABILITY_MODEL:-llama3:1B}"
    fi
    require_file "$log_path"
    local args=(
        --log "$log_path"
        --val-final-step "$steps"
        --train-final-step "$((steps - 1))"
        --require-val
        --require-train
        --require-train-loss-decrease
    )
    if [ "${LLAMA1B_STABILITY_HELLASWAG:-1}" = "1" ] || [ -n "${LLAMA1B_STABILITY_MIN_HELLASWAG:-}" ]; then
        args+=(--eval-final-step "$steps" --require-eval)
    fi
    if [ -n "${LLAMA1B_STABILITY_MAX_VAL_LOSS:-}" ]; then
        args+=(--max-val-loss "$LLAMA1B_STABILITY_MAX_VAL_LOSS")
    fi
    if [ -n "${LLAMA1B_STABILITY_MIN_HELLASWAG:-}" ]; then
        args+=(--min-eval "$LLAMA1B_STABILITY_MIN_HELLASWAG")
    fi
    run_contains "Training log validation OK" python3 dev/validate_training_log.py "${args[@]}"
}

phase_profile() {
    log "ncu profiling"
    if [ "${PROFILE_VALIDATE_ONLY:-0}" != "1" ]; then
        require_cuda_tool ncu
        require_file gpt2_124M_bf16.bin
    elif [ -z "${PROFILE_CSV_DIR:-}" ]; then
        require_cuda_tool ncu
    fi
    local fusion
    local skip_build=0
    for fusion in ${PROFILE_GELU_FUSIONS:-0 1}; do
        case "$fusion" in
            0|1) ;;
            *) die "PROFILE_GELU_FUSIONS may contain only 0 or 1, got: $fusion" ;;
        esac
        local output="profile_ge${fusion}"
        local args=(
            --min-tensor-util "${PROFILE_MIN_TENSOR_UTIL:-70}"
            --gelu-fusion "$fusion"
            --output "$output"
        )
        if [ "${PROFILE_VALIDATE_ONLY:-0}" = "1" ]; then
            if [ -n "${PROFILE_CSV_DIR:-}" ]; then
                local csv="${PROFILE_CSV_DIR}/${output}.csv"
                require_file "$csv"
                args+=(--csv-input "$csv")
            else
                local report="${PROFILE_REPORT_DIR:-.}/${output}.ncu-rep"
                require_file "$report"
                args+=(--skip-build --skip-run --report "$report")
            fi
        elif [ "$skip_build" = "1" ]; then
            args+=(--skip-build)
        fi
        run python3 profile_gpt2cu.py "${args[@]}"
        skip_build=1
    done
}

phase_gpt2_full() {
    log "GPT-2 124M full reproduction"
    require_file dev/validate_training_log.py
    local out_dir="${GPT2_FULL_OUT_DIR:-${OUT_DIR:-log_gpt2_124M}}"
    local final_step="${GPT2_FULL_FINAL_STEP:-18865}"
    local log_path="${GPT2_FULL_LOG:-$out_dir/main.log}"
    local run_log_path="${GPT2_FULL_RUN_LOG:-$out_dir/run.log}"
    if [ "${GPT2_FULL_VALIDATE_ONLY:-0}" != "1" ]; then
        run env OUT_DIR="$out_dir" MAX_STEPS="$final_step" scripts/run_gpt2_124M.sh
    fi
    require_gpt2_full_run_log "$run_log_path" "$final_step"
    require_gpt_checkpoint_step "$out_dir" "$final_step"
    require_file "$log_path"
    local args=(
        --log "$log_path"
        --val-final-step "$final_step"
        --eval-final-step "$final_step"
        --require-val
        --require-eval
    )
    if [ -n "${GPT2_FULL_EXPECTED_VAL_LOSS:-}" ]; then
        args+=(--expected-val-loss "$GPT2_FULL_EXPECTED_VAL_LOSS")
    fi
    if [ -n "${GPT2_FULL_EXPECTED_HELLASWAG:-}" ]; then
        args+=(--expected-eval "$GPT2_FULL_EXPECTED_HELLASWAG")
    fi
    if [ -n "${GPT2_FULL_METRIC_REL_TOL:-}" ]; then
        args+=(--rel-tol "$GPT2_FULL_METRIC_REL_TOL")
    fi
    run_contains "Training log validation OK" python3 dev/validate_training_log.py "${args[@]}"
}

phase_gpt2_two_node() {
    log "GPT-2 two-node loss-curve sanity"
    require_file dev/compare_training_logs.py
    local reference_log="${GPT2_SINGLE_NODE_LOG:-${GPT2_FULL_LOG:-${GPT2_FULL_OUT_DIR:-${OUT_DIR:-log_gpt2_124M}}/main.log}}"
    local candidate_out="${GPT2_TWO_NODE_OUT_DIR:-/ephemeral/data/fineweb/log_gpt2_124M_multi}"
    local candidate_log="${GPT2_TWO_NODE_LOG:-$candidate_out/main.log}"
    [ -n "${GPT2_TWO_NODE_REL_TOL:-}" ] || die "GPT2_TWO_NODE_REL_TOL is required for gpt2-two-node"
    local start_step="${GPT2_TWO_NODE_START_STEP:-0}"
    local steps="${GPT2_TWO_NODE_STEPS:-100}"
    local max_steps=$((start_step + steps))
    if [ "${GPT2_TWO_NODE_VALIDATE_ONLY:-0}" != "1" ]; then
        require_cmd sbatch
        local script="${GPT2_TWO_NODE_SCRIPT:-scripts/multi_node/run_gpt2_124M_fs.sbatch}"
        require_file "$script"
        local sbatch_args=(--wait)
        if [ -n "${GPT2_TWO_NODE_SBATCH_ARGS:-}" ]; then
            # shellcheck disable=SC2206
            local extra_args=( ${GPT2_TWO_NODE_SBATCH_ARGS} )
            sbatch_args+=("${extra_args[@]}")
        fi
        run env OUT_DIR="$candidate_out" MAX_STEPS="$max_steps" sbatch "${sbatch_args[@]}" "$script"
    fi
    require_file "$reference_log"
    require_file "$candidate_log"
    local args=(
        --reference-log "$reference_log"
        --candidate-log "$candidate_log"
        --metric train
        --start-step "$start_step"
        --steps "$steps"
        --rel-tol "$GPT2_TWO_NODE_REL_TOL"
        --require-decrease
    )
    if [ -n "${GPT2_TWO_NODE_ABS_TOL:-}" ]; then
        args+=(--abs-tol "$GPT2_TWO_NODE_ABS_TOL")
    fi
    run_contains "Training log comparison OK" python3 dev/compare_training_logs.py "${args[@]}"
}

phase_llama1b_full() {
    log "Llama-3 1B full run"
    require_file dev/validate_training_log.py
    local out_dir="${LLAMA1B_FULL_OUT_DIR:-${OUT_DIR:-log_llama3_1B}}"
    local final_step="${LLAMA1B_FULL_FINAL_STEP:-57220}"
    [ "$final_step" -ge 1 ] || die "LLAMA1B_FULL_FINAL_STEP must be at least 1"
    local log_path="${LLAMA1B_FULL_LOG:-$out_dir/main.log}"
    local run_log_path="${LLAMA1B_FULL_RUN_LOG:-$out_dir/run.log}"
    if [ "${LLAMA1B_FULL_VALIDATE_ONLY:-0}" != "1" ]; then
        run env OUT_DIR="$out_dir" MAX_STEPS="$final_step" scripts/run_llama3_1B.sh
    fi
    require_llama1b_full_run_log "$run_log_path" "$final_step"
    require_llama_checkpoint_step "$out_dir" "$final_step"
    require_file "$log_path"
    local args=(
        --log "$log_path"
        --val-final-step "$final_step"
        --eval-final-step "$final_step"
        --train-final-step "$((final_step - 1))"
        --require-val
        --require-eval
        --require-train
        --require-train-loss-decrease
    )
    if [ -n "${LLAMA1B_FULL_MAX_VAL_LOSS:-}" ]; then
        args+=(--max-val-loss "$LLAMA1B_FULL_MAX_VAL_LOSS")
    fi
    if [ -n "${LLAMA1B_FULL_MIN_HELLASWAG:-}" ]; then
        args+=(--min-eval "$LLAMA1B_FULL_MIN_HELLASWAG")
    fi
    run_contains "Training log validation OK" python3 dev/validate_training_log.py "${args[@]}"
}

phase_llama8b_convert() {
    log "Llama-3.1 8B HF checkpoint conversion"
    require_file dev/download_llama3.py
    require_file ./train_llama3cu
    local output_dir="${LLAMA8B_OUTPUT_DIR:-.}"
    local checkpoint="${LLAMA8B_CHECKPOINT:-$output_dir/llama3.1_8B_bf16.bin}"
    local zero_stage="${LLAMA8B_CONVERT_ZERO_STAGE:-2}"
    local dry_processes="${LLAMA8B_CONVERT_PROCESSES:-16}"
    local expected="train_llama3cu dry run: checkpoint/config parsed"
    if [ "$zero_stage" != "0" ]; then
        expected="train_llama3cu dry run: ZeRO-${zero_stage} shard layout validated"
    fi

    if [ "${LLAMA8B_CONVERT_VALIDATE_ONLY:-0}" = "1" ]; then
        require_file "$checkpoint"
    fi

    if [ -f "$checkpoint" ]; then
        run_contains "$expected" \
            python3 dev/download_llama3.py \
            --validate-only "$checkpoint" \
            --cpp-validate \
            --cpp-zero-stage "$zero_stage" \
            --cpp-processes "$dry_processes" \
            --train-binary ./train_llama3cu
    else
        run_contains "$expected" \
            python3 dev/download_llama3.py \
            "${LLAMA8B_MODEL:-llama3.1:8B}" \
            --output_dir "$output_dir" \
            --cpp-validate \
            --cpp-zero-stage "$zero_stage" \
            --cpp-processes "$dry_processes" \
            --train-binary ./train_llama3cu
    fi
}

phase_llama8b_full() {
    log "Llama-3 8B multi-node run"
    require_file dev/validate_training_log.py
    require_file dev/validate_llama_checkpoint_artifacts.py
    local out_dir="${LLAMA8B_FULL_OUT_DIR:-/ephemeral/data/fineweb/log_llama3_8B_multi}"
    local final_step="${LLAMA8B_FULL_FINAL_STEP:-57220}"
    local nproc="${LLAMA8B_FULL_NPROC:-16}"
    local run_log_path="${LLAMA8B_FULL_RUN_LOG:-$out_dir/run.log}"
    [ "$final_step" -ge 1 ] || die "LLAMA8B_FULL_FINAL_STEP must be at least 1"
    [ "$nproc" -ge 1 ] || die "LLAMA8B_FULL_NPROC must be at least 1"

    if [ "${LLAMA8B_FULL_VALIDATE_ONLY:-0}" != "1" ]; then
        require_cmd sbatch
        local sbatch_args=(--wait)
        if [ -n "${LLAMA8B_FULL_SBATCH_ARGS:-}" ]; then
            # shellcheck disable=SC2206
            local extra_args=( ${LLAMA8B_FULL_SBATCH_ARGS} )
            sbatch_args+=("${extra_args[@]}")
        fi
        run env OUT_DIR="$out_dir" MAX_STEPS="$final_step" sbatch "${sbatch_args[@]}" \
            scripts/multi_node/run_llama3_8B_fs.sbatch
    fi

    require_llama8b_full_run_log "$run_log_path" "$final_step" "$nproc"
    require_llama_checkpoint_step "$out_dir" "$final_step"
    run_contains "Llama checkpoint artifacts OK" \
        python3 dev/validate_llama_checkpoint_artifacts.py \
        --output-dir "$out_dir" \
        --step "$final_step" \
        --rank 0 \
        --num-processes "$nproc"

    local args=(
        --log "$out_dir/main.log"
        --val-final-step "$final_step"
        --eval-final-step "$final_step"
        --train-final-step "$((final_step - 1))"
        --require-val
        --require-eval
        --require-train
        --require-train-loss-decrease
    )
    if [ -n "${LLAMA8B_FULL_MAX_VAL_LOSS:-}" ]; then
        args+=(--max-val-loss "$LLAMA8B_FULL_MAX_VAL_LOSS")
    fi
    if [ -n "${LLAMA8B_FULL_MIN_HELLASWAG:-}" ]; then
        args+=(--min-eval "$LLAMA8B_FULL_MIN_HELLASWAG")
    fi
    run_contains "Training log validation OK" python3 dev/validate_training_log.py "${args[@]}"
}

phase_goal_core() {
    phase_preflight
    phase_compile
    phase_script_syntax
    phase_python_syntax
    phase_source_guards
    phase_data_artifacts
    phase_dataloader_smoke
    phase_gqa_reference
    phase_profile_parser
    phase_log_tools
    phase_goal_replay_smoke
    phase_llama_converter_smoke
    phase_cuda_runtime
    phase_starter_pack
    phase_smoke
    phase_gpt2
    phase_gpt_dry
    phase_llama_dry
    phase_llama_checkpoint_smoke
    phase_zero_guards
}

phase_host_core() {
    phase_script_syntax
    phase_python_syntax
    phase_source_guards
    phase_data_artifacts
    phase_dataloader_smoke
    phase_gqa_reference
    phase_profile_parser
    phase_log_tools
    phase_goal_replay_smoke
    phase_llama_converter_smoke
    phase_starter_pack
    phase_gpt_dry
    phase_llama_dry
    phase_llama_checkpoint_smoke
    phase_zero_guards
}

phase_goal_complete_prereqs() {
    if [ "${ALLOW_NON_H100:-0}" = "1" ]; then
        die "goal-complete requires real H100/sm_90-class runtime evidence; unset ALLOW_NON_H100"
    fi
    require_env_vars \
        GPT2_SMOKE_MAX_VAL_LOSS \
        ZERO3_SMOKE_MAX_VAL_LOSS \
        LLAMA_RESUME_MAX_VAL_LOSS \
        LLAMA1B_STABILITY_MAX_VAL_LOSS \
        LLAMA1B_STABILITY_MIN_HELLASWAG \
        GPT2_FULL_EXPECTED_VAL_LOSS \
        GPT2_FULL_EXPECTED_HELLASWAG \
        GPT2_TWO_NODE_REL_TOL \
        LLAMA1B_FULL_MAX_VAL_LOSS \
        LLAMA1B_FULL_MIN_HELLASWAG \
        LLAMA8B_FULL_MAX_VAL_LOSS \
        LLAMA8B_FULL_MIN_HELLASWAG
    require_goal_metric_thresholds
    if [ "${PROFILE_VALIDATE_ONLY:-0}" != "1" ]; then
        require_cuda_tool ncu
    elif [ -z "${PROFILE_CSV_DIR:-}" ]; then
        require_cuda_tool ncu
    fi
    require_file gpt2_124M_bf16.bin
    if [ "${PREFLIGHT_VALIDATE_ONLY:-0}" = "1" ]; then
        require_file_contains "H100 preflight OK" "${PREFLIGHT_LOG:-preflight.log}"
    fi
    if [ "${CUDA_RUNTIME_VALIDATE_ONLY:-0}" = "1" ]; then
        require_file_contains "CUDA runtime check passed." "${CUDA_RUNTIME_LOG:-cuda_runtime_check.log}"
    fi
    if [ "${SMOKE_VALIDATE_ONLY:-0}" = "1" ]; then
        local bin
        for bin in test_matmul test_attention test_layernorm test_rope test_rmsnorm test_swiglu test_attention_gqa; do
            require_file_contains "$bin smoke OK" "${SMOKE_LOG_DIR:-.}/$bin.log"
        done
    fi
    if [ "${GPT2_RUNTIME_VALIDATE_ONLY:-0}" = "1" ]; then
        require_file_contains "gpt2_validate OK" "${GPT2_VALIDATE_LOG:-gpt2_validate.log}"
        require_file_contains "test_gpt2cu OK" "${GPT2_PARITY_LOG:-test_gpt2cu.log}"
    fi
    if [ "${GQA_RUNTIME_VALIDATE_ONLY:-0}" = "1" ]; then
        require_file_contains_all 3 \
            "GQA case T=128 backward=fallback OK" \
            "GQA case T=256 backward=tk OK" \
            "test_attention_gqa smoke OK" \
            "${GQA_RUNTIME_LOG:-test_attention_gqa.log}"
    fi
    if [ "${GPT2_SMOKE_VALIDATE_ONLY:-0}" = "1" ]; then
        local gpt2_smoke_out="${GPT2_SMOKE_OUT:-log_goal_gpt2_smoke}"
        local gpt2_smoke_log="${GPT2_SMOKE_LOG:-$gpt2_smoke_out/main.log}"
        require_file "$gpt2_smoke_log"
    fi
    if [ "${ZERO3_SMOKE_VALIDATE_ONLY:-0}" = "1" ]; then
        local zero3_smoke_out="${ZERO3_SMOKE_OUT:-log_goal_zero3_smoke}"
        local zero3_smoke_log="${ZERO3_SMOKE_LOG:-$zero3_smoke_out/main.log}"
        local zero3_smoke_run_log="${ZERO3_SMOKE_RUN_LOG:-$zero3_smoke_out/run.log}"
        require_file "$zero3_smoke_log"
        require_file_contains "ZeRO Stage 3: parameter shards + runtime all-gather compute layout" "$zero3_smoke_run_log"
    fi
    if [ "${LLAMA_RESUME_VALIDATE_ONLY:-0}" = "1" ]; then
        local llama_resume_out="${LLAMA_RESUME_OUT:-log_goal_llama_resume}"
        local llama_resume_step="${LLAMA_RESUME_STEPS:-2}"
        [ "$llama_resume_step" -ge 1 ] || die "LLAMA_RESUME_STEPS must be at least 1"
        require_llama_checkpoint_step "$llama_resume_out" 1
        require_llama_checkpoint_step "$llama_resume_out" "$llama_resume_step"
        require_file "${LLAMA_RESUME_LOG:-$llama_resume_out/main.log}"
    fi
    if [ "${LLAMA1B_STABILITY_VALIDATE_ONLY:-0}" = "1" ]; then
        local llama1b_stability_out="${LLAMA1B_STABILITY_OUT:-log_goal_llama1b_stability}"
        require_file "${LLAMA1B_STABILITY_LOG:-$llama1b_stability_out/main.log}"
    fi
    if [ "${PROFILE_VALIDATE_ONLY:-0}" = "1" ]; then
        require_goal_complete_profile_evidence
    fi
    if [ "${GPT2_TWO_NODE_VALIDATE_ONLY:-0}" != "1" ] || [ "${LLAMA8B_FULL_VALIDATE_ONLY:-0}" != "1" ]; then
        require_cmd sbatch
    fi
    if [ "${GPT2_FULL_VALIDATE_ONLY:-0}" = "1" ]; then
        local gpt2_full_out="${GPT2_FULL_OUT_DIR:-${OUT_DIR:-log_gpt2_124M}}"
        local gpt2_full_log="${GPT2_FULL_LOG:-$gpt2_full_out/main.log}"
        local gpt2_full_run_log="${GPT2_FULL_RUN_LOG:-$gpt2_full_out/run.log}"
        local gpt2_full_final_step="${GPT2_FULL_FINAL_STEP:-18865}"
        require_gpt2_full_run_log "$gpt2_full_run_log" "$gpt2_full_final_step"
        require_gpt_checkpoint_step "$gpt2_full_out" "$gpt2_full_final_step"
        require_file "$gpt2_full_log"
    fi
    if [ "${GPT2_TWO_NODE_VALIDATE_ONLY:-0}" = "1" ]; then
        local reference_log="${GPT2_SINGLE_NODE_LOG:-${GPT2_FULL_LOG:-${GPT2_FULL_OUT_DIR:-${OUT_DIR:-log_gpt2_124M}}/main.log}}"
        local candidate_out="${GPT2_TWO_NODE_OUT_DIR:-/ephemeral/data/fineweb/log_gpt2_124M_multi}"
        local candidate_log="${GPT2_TWO_NODE_LOG:-$candidate_out/main.log}"
        require_file "$reference_log"
        require_file "$candidate_log"
    fi
    if [ "${LLAMA1B_FULL_VALIDATE_ONLY:-0}" = "1" ]; then
        local llama1b_out="${LLAMA1B_FULL_OUT_DIR:-${OUT_DIR:-log_llama3_1B}}"
        local llama1b_log="${LLAMA1B_FULL_LOG:-$llama1b_out/main.log}"
        local llama1b_run_log="${LLAMA1B_FULL_RUN_LOG:-$llama1b_out/run.log}"
        local llama1b_final_step="${LLAMA1B_FULL_FINAL_STEP:-57220}"
        require_llama1b_full_run_log "$llama1b_run_log" "$llama1b_final_step"
        require_llama_checkpoint_step "$llama1b_out" "$llama1b_final_step"
        require_file "$llama1b_log"
    fi
    if [ "${LLAMA8B_CONVERT_VALIDATE_ONLY:-0}" = "1" ]; then
        local llama8b_output_dir="${LLAMA8B_OUTPUT_DIR:-.}"
        local llama8b_checkpoint="${LLAMA8B_CHECKPOINT:-$llama8b_output_dir/llama3.1_8B_bf16.bin}"
        require_file "$llama8b_checkpoint"
    fi
    if [ "${LLAMA8B_FULL_VALIDATE_ONLY:-0}" = "1" ]; then
        local out_dir="${LLAMA8B_FULL_OUT_DIR:-/ephemeral/data/fineweb/log_llama3_8B_multi}"
        local final_step="${LLAMA8B_FULL_FINAL_STEP:-57220}"
        local nproc="${LLAMA8B_FULL_NPROC:-16}"
        local run_log_path="${LLAMA8B_FULL_RUN_LOG:-$out_dir/run.log}"
        [ "$final_step" -ge 1 ] || die "LLAMA8B_FULL_FINAL_STEP must be at least 1"
        [ "$nproc" -ge 1 ] || die "LLAMA8B_FULL_NPROC must be at least 1"
        require_llama8b_full_run_log "$run_log_path" "$final_step" "$nproc"
        require_llama_checkpoint_step "$out_dir" "$final_step"
        require_file "$out_dir/main.log"
    fi
}

phase_goal_complete() {
    if [ "${ALLOW_FULL_GOAL_RUN:-0}" != "1" ]; then
        die "goal-complete launches long H100/NCCL/profile/conversion/full-run jobs; set ALLOW_FULL_GOAL_RUN=1 to run it intentionally"
    fi
    phase_goal_complete_prereqs
    phase_goal_core
    phase_gpt2_smoke
    phase_zero3_smoke
    phase_llama_resume
    phase_gqa_runtime
    local saved_llama1b_stability_hellaswag="${LLAMA1B_STABILITY_HELLASWAG:-}"
    LLAMA1B_STABILITY_HELLASWAG=1
    phase_llama1b_stability
    if [ -n "$saved_llama1b_stability_hellaswag" ]; then
        LLAMA1B_STABILITY_HELLASWAG="$saved_llama1b_stability_hellaswag"
    else
        unset LLAMA1B_STABILITY_HELLASWAG
    fi
    local saved_profile_gelu_fusions="${PROFILE_GELU_FUSIONS:-}"
    PROFILE_GELU_FUSIONS="0 1"
    phase_profile
    if [ -n "$saved_profile_gelu_fusions" ]; then
        PROFILE_GELU_FUSIONS="$saved_profile_gelu_fusions"
    else
        unset PROFILE_GELU_FUSIONS
    fi
    phase_gpt2_full
    phase_gpt2_two_node
    phase_llama1b_full
    phase_llama8b_convert
    phase_llama8b_full
}

if [ "$#" -eq 0 ]; then
    set -- goal-core
fi

for phase in "$@"; do
    case "$phase" in
        help|-h|--help) usage ;;
        goal-core)
            phase_goal_core
            ;;
        host-core|all-local) phase_host_core ;;
        goal-complete-prereqs) phase_goal_complete_prereqs ;;
        goal-complete) phase_goal_complete ;;
        preflight) phase_preflight ;;
        compile) phase_compile ;;
        script-syntax) phase_script_syntax ;;
        python-syntax) phase_python_syntax ;;
        source-guards) phase_source_guards ;;
        data-artifacts) phase_data_artifacts ;;
        dataloader-smoke) phase_dataloader_smoke ;;
        gqa-reference) phase_gqa_reference ;;
        gqa-runtime) phase_gqa_runtime ;;
        profile-parser) phase_profile_parser ;;
        log-tools) phase_log_tools ;;
        goal-replay-smoke) phase_goal_replay_smoke ;;
        llama-converter-smoke) phase_llama_converter_smoke ;;
        cuda-runtime) phase_cuda_runtime ;;
        starter-pack) phase_starter_pack ;;
        smoke) phase_smoke ;;
        gpt2) phase_gpt2 ;;
        gpt2-smoke) phase_gpt2_smoke ;;
        zero3-smoke) phase_zero3_smoke ;;
        gpt-dry) phase_gpt_dry ;;
        llama-dry) phase_llama_dry ;;
        llama-checkpoint-smoke) phase_llama_checkpoint_smoke ;;
        zero-guards) phase_zero_guards ;;
        llama-resume) phase_llama_resume ;;
        llama1b-stability) phase_llama1b_stability ;;
        profile) phase_profile ;;
        gpt2-full) phase_gpt2_full ;;
        gpt2-two-node) phase_gpt2_two_node ;;
        llama1b-full) phase_llama1b_full ;;
        llama8b-convert) phase_llama8b_convert ;;
        llama8b-full) phase_llama8b_full ;;
        *) die "unknown phase: $phase" ;;
    esac
done
