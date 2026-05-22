#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_LABEL="${RUN_LABEL:-$(date +%Y%m%d_%H%M%S)}"
ARTIFACT_DIR="${ARTIFACT_DIR:-scratch/sm120_rounds/$RUN_LABEL}"
TRAIN_OUT_DIR="${TRAIN_OUT_DIR:-log124M/5090_S_$RUN_LABEL}"
MAX_STEPS="${MAX_STEPS:-3}"
TRAIN_ZERO_STAGE="${TRAIN_ZERO_STAGE:-1}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
DEVICE_ARCH="${DEVICE_ARCH:-SM120}"
NO_MULTI_GPU="${NO_MULTI_GPU:-1}"
NO_USE_MPI="${NO_USE_MPI:-1}"
RUN_CORRECTNESS="${RUN_CORRECTNESS:-1}"
RUN_BENCHMARKS="${RUN_BENCHMARKS:-1}"
RUN_PYTHON_STACK_BENCHMARKS="${RUN_PYTHON_STACK_BENCHMARKS:-$RUN_BENCHMARKS}"
LLMK_CUDNN_PACKED_BACKWARD_ROUTE="${LLMK_CUDNN_PACKED_BACKWARD_ROUTE:-saved-forward}"
LLMK_LIBTORCH_RUNTIME_ROUTE="${LLMK_LIBTORCH_RUNTIME_ROUTE:-cxx-api-raw-pointer}"
LLMK_LIBTORCH_RUNTIME_SUPPLEMENTAL_SHAPES="${LLMK_LIBTORCH_RUNTIME_SUPPLEMENTAL_SHAPES:-gelu_forward}"
RUN_LIBTORCH_TRAINER_LINK_PROBE="${RUN_LIBTORCH_TRAINER_LINK_PROBE:-$RUN_PYTHON_STACK_BENCHMARKS}"
RUN_LIBTORCH_MATMUL_BENCHMARKS="${RUN_LIBTORCH_MATMUL_BENCHMARKS:-$RUN_PYTHON_STACK_BENCHMARKS}"
LLMK_LIBTORCH_MATMUL_SHAPES="${LLMK_LIBTORCH_MATMUL_SHAPES:-qkv attproj fc fcproj lmhead}"
SM120_USE_LIBTORCH_MEMORY="${SM120_USE_LIBTORCH_MEMORY:-0}"
SM120_USE_LIBTORCH_GRAD_ZERO="${SM120_USE_LIBTORCH_GRAD_ZERO:-$SM120_USE_LIBTORCH_MEMORY}"
SM120_USE_LIBTORCH_DRESIDUAL_ZERO="${SM120_USE_LIBTORCH_DRESIDUAL_ZERO:-0}"
RUN_TRAINING="${RUN_TRAINING:-1}"
RUN_STACK_PROBE="${RUN_STACK_PROBE:-1}"
RUN_ARTIFACT_VALIDATOR="${RUN_ARTIFACT_VALIDATOR:-1}"
RUN_CURRENT_SELECTION_AUDIT="${RUN_CURRENT_SELECTION_AUDIT:-0}"
SM120_SELECTION_NATIVE_ROUND="${SM120_SELECTION_NATIVE_ROUND:-}"
SM120_SELECTION_OPTIONAL_ROUND="${SM120_SELECTION_OPTIONAL_ROUND:-}"
SM120_SELECTION_JSON_OUT="${SM120_SELECTION_JSON_OUT:-scratch/sm120_rounds/current-sm120-selection.json}"
SM120_SELECTION_MD_OUT="${SM120_SELECTION_MD_OUT:-scratch/sm120_rounds/current-sm120-selection.md}"
SM120_AUDIT_JSON_OUT="${SM120_AUDIT_JSON_OUT:-scratch/sm120_rounds/current-sm120-audit.json}"
SM120_AUDIT_MD_OUT="${SM120_AUDIT_MD_OUT:-scratch/sm120_rounds/current-sm120-audit.md}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-0}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_BASE_LOG_DIR="${ALLOW_BASE_LOG_DIR:-0}"
if [[ -z "${PYTHON_BIN:-}" && -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if [[ "$DEVICE_ARCH" != "SM120" ]]; then
    echo "This harness is scoped to DEVICE_ARCH=SM120; got DEVICE_ARCH=$DEVICE_ARCH" >&2
    exit 2
fi

if [[ "$TRAIN_OUT_DIR" == "log124M/5090_S" && "$ALLOW_BASE_LOG_DIR" != "1" ]]; then
    echo "Refusing to write into log124M/5090_S without ALLOW_BASE_LOG_DIR=1" >&2
    exit 2
fi

mkdir -p "$ARTIFACT_DIR"

log_path() {
    printf '%s/%s.log' "$ARTIFACT_DIR" "$1"
}

run_logged() {
    local name="$1"
    shift
    local cmd=("$@")
    local logfile
    logfile="$(log_path "$name")"
    printf '\n## %s\n' "$name" | tee -a "$ARTIFACT_DIR/summary.md"
    printf 'Command: `%q' "${cmd[0]}" | tee -a "$ARTIFACT_DIR/summary.md"
    for arg in "${cmd[@]:1}"; do
        printf ' %q' "$arg" | tee -a "$ARTIFACT_DIR/summary.md"
    done
    printf '`\n\n' | tee -a "$ARTIFACT_DIR/summary.md"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run] see $logfile"
        return 0
    fi

    "${cmd[@]}" 2>&1 | tee "$logfile"
}

append_metadata() {
    {
        echo "# SM120 Optimization Round"
        echo
        echo "- run label: \`$RUN_LABEL\`"
        echo "- artifact dir: \`$ARTIFACT_DIR\`"
        echo "- train output dir: \`$TRAIN_OUT_DIR\`"
        echo "- max steps: \`$MAX_STEPS\`"
        echo "- train zero stage: \`$TRAIN_ZERO_STAGE\`"
        echo "- python: \`$PYTHON_BIN\`"
        echo "- cuDNN packed backward route: \`$LLMK_CUDNN_PACKED_BACKWARD_ROUTE\`"
        echo "- LibTorch runtime route: \`$LLMK_LIBTORCH_RUNTIME_ROUTE\`"
        echo "- LibTorch runtime supplemental shapes: \`$LLMK_LIBTORCH_RUNTIME_SUPPLEMENTAL_SHAPES\`"
        echo "- LibTorch trainer link probe: \`$RUN_LIBTORCH_TRAINER_LINK_PROBE\`"
        echo "- LibTorch matmul shapes: \`$LLMK_LIBTORCH_MATMUL_SHAPES\`"
        echo "- SM120 LibTorch trainer memory route: \`$SM120_USE_LIBTORCH_MEMORY\`"
        echo "- SM120 LibTorch grad-zero route: \`$SM120_USE_LIBTORCH_GRAD_ZERO\`"
        echo "- SM120 LibTorch dresidual-zero route: \`$SM120_USE_LIBTORCH_DRESIDUAL_ZERO\`"
        echo "- git commit: \`$(git rev-parse --short HEAD 2>/dev/null || echo unknown)\`"
        echo "- working tree: \`$(git status --short | wc -l | tr -d ' ')\` changed paths"
        echo
        echo "## Environment"
        echo
        echo '```text'
        nvcc --version 2>/dev/null || true
        nvidia-smi 2>/dev/null || true
        echo '```'
    } > "$ARTIFACT_DIR/summary.md"

    git status --short > "$ARTIFACT_DIR/git-status.txt"
}

summarize_log_lines() {
    local title="$1"
    local file="$2"
    local pattern="$3"

    {
        echo
        echo "## $title"
        echo
        echo '```text'
        if [[ -f "$file" ]]; then
            grep -E "$pattern" "$file" || true
        else
            echo "missing: $file"
        fi
        echo '```'
    } >> "$ARTIFACT_DIR/summary.md"
}

summarize_correctness_logs() {
    {
        echo
        echo "## Correctness Markers"
        echo
        echo '```text'
        for name in "$@"; do
            local file
            file="$(log_path "$name")"
            if [[ -f "$file" ]]; then
                echo "[$name]"
                grep -E 'PASS|FAIL|SKIPPED|test_|All|passed' "$file" || true
            else
                echo "[$name] missing: $file"
            fi
        done
        echo '```'
    } >> "$ARTIFACT_DIR/summary.md"
}

cleanup_checkpoints() {
    if [[ "$KEEP_CHECKPOINTS" == "1" || "$DRY_RUN" == "1" ]]; then
        return 0
    fi
    case "$TRAIN_OUT_DIR" in
        log124M/5090_S_*)
            find "$TRAIN_OUT_DIR" -maxdepth 1 -type f \( -name 'model_*.bin' -o -name 'state_*.bin' \) -print -delete \
                > "$ARTIFACT_DIR/removed-checkpoints.txt" 2>/dev/null || true
            ;;
        *)
            echo "Skipping checkpoint cleanup for non-round dir: $TRAIN_OUT_DIR" \
                > "$ARTIFACT_DIR/removed-checkpoints.txt"
            ;;
    esac
}

append_metadata

if [[ "$RUN_STACK_PROBE" == "1" ]]; then
    run_logged probe_sm120_backend_stacks "$PYTHON_BIN" dev/probe_sm120_backend_stacks.py \
        --json-out "$ARTIFACT_DIR/backend-stacks.json" \
        --markdown-out "$ARTIFACT_DIR/backend-stacks.md"
fi

COMMON_MAKE_ARGS=(
    DEVICE_ARCH="$DEVICE_ARCH"
    NO_MULTI_GPU="$NO_MULTI_GPU"
    NO_USE_MPI="$NO_USE_MPI"
    SM120_USE_LIBTORCH_MEMORY="$SM120_USE_LIBTORCH_MEMORY"
    SM120_USE_LIBTORCH_GRAD_ZERO="$SM120_USE_LIBTORCH_GRAD_ZERO"
    SM120_USE_LIBTORCH_DRESIDUAL_ZERO="$SM120_USE_LIBTORCH_DRESIDUAL_ZERO"
    PYTHON_BIN="$PYTHON_BIN"
)

BUILD_TARGETS=(
    test_matmul
    test_attention
    test_layernorm
    test_bias
    test_gelu
    test_fused_classifier
    test_encoder
    test_adamw
    test_global_norm
    bench_sm120_matmul
    bench_sm120_attention
    bench_sm120_layernorm
    bench_sm120_runtime
    train_gpt2cu
)

run_logged build make -j "$BUILD_JOBS" "${BUILD_TARGETS[@]}" "${COMMON_MAKE_ARGS[@]}"

run_logged write_sm120_round_manifest "$PYTHON_BIN" dev/write_sm120_round_manifest.py \
    --json-out "$ARTIFACT_DIR/round-manifest.json" \
    --markdown-out "$ARTIFACT_DIR/round-manifest.md" \
    --run-label "$RUN_LABEL" \
    --artifact-dir "$ARTIFACT_DIR" \
    --train-out-dir "$TRAIN_OUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --train-zero-stage "$TRAIN_ZERO_STAGE" \
    --device-arch "$DEVICE_ARCH" \
    --build-jobs "$BUILD_JOBS" \
    --no-multi-gpu "$NO_MULTI_GPU" \
    --no-use-mpi "$NO_USE_MPI" \
    --run-stack-probe "$RUN_STACK_PROBE" \
    --run-correctness "$RUN_CORRECTNESS" \
    --run-benchmarks "$RUN_BENCHMARKS" \
    --run-python-stack-benchmarks "$RUN_PYTHON_STACK_BENCHMARKS" \
    --cudnn-packed-backward-route "$LLMK_CUDNN_PACKED_BACKWARD_ROUTE" \
    --libtorch-runtime-route "$LLMK_LIBTORCH_RUNTIME_ROUTE" \
    --libtorch-runtime-supplemental-shapes "$LLMK_LIBTORCH_RUNTIME_SUPPLEMENTAL_SHAPES" \
    --run-libtorch-trainer-link-probe "$RUN_LIBTORCH_TRAINER_LINK_PROBE" \
    --run-libtorch-matmul-benchmarks "$RUN_LIBTORCH_MATMUL_BENCHMARKS" \
    --libtorch-matmul-shapes "$LLMK_LIBTORCH_MATMUL_SHAPES" \
    --sm120-use-libtorch-memory "$SM120_USE_LIBTORCH_MEMORY" \
    --sm120-use-libtorch-grad-zero "$SM120_USE_LIBTORCH_GRAD_ZERO" \
    --sm120-use-libtorch-dresidual-zero "$SM120_USE_LIBTORCH_DRESIDUAL_ZERO" \
    --run-training "$RUN_TRAINING" \
    --keep-checkpoints "$KEEP_CHECKPOINTS"

if [[ "$RUN_CORRECTNESS" == "1" ]]; then
    run_logged test_matmul ./test_matmul
    run_logged test_attention ./test_attention
    run_logged test_layernorm ./test_layernorm
    run_logged test_bias ./test_bias
    run_logged test_gelu ./test_gelu
    run_logged test_fused_classifier ./test_fused_classifier
    run_logged test_encoder ./test_encoder
    run_logged test_adamw ./test_adamw
    run_logged test_global_norm ./test_global_norm
fi

if [[ "$RUN_BENCHMARKS" == "1" ]]; then
    run_logged bench_sm120_matmul ./bench_sm120_matmul
    run_logged bench_sm120_attention ./bench_sm120_attention
    run_logged bench_sm120_layernorm ./bench_sm120_layernorm
    run_logged bench_sm120_runtime ./bench_sm120_runtime
fi

if [[ "$RUN_PYTHON_STACK_BENCHMARKS" == "1" ]]; then
    run_logged bench_sm120_torch_matmul "$PYTHON_BIN" dev/bench_sm120_torch_matmul.py \
        --repeats "${LLMK_BENCH_REPEATS:-7}" \
        --large-repeats "${LLMK_BENCH_LARGE_REPEATS:-3}"
    if [[ "$RUN_LIBTORCH_MATMUL_BENCHMARKS" == "1" ]]; then
        LIBTORCH_MATMUL_ARGS=(
            "$PYTHON_BIN" dev/bench_sm120_libtorch_matmul.py
            --repeats "${LLMK_BENCH_REPEATS:-7}"
            --large-repeats "${LLMK_BENCH_LARGE_REPEATS:-3}"
            --warmup 3
            --json-out "$ARTIFACT_DIR/bench_sm120_libtorch_matmul.json"
        )
        for shape in $LLMK_LIBTORCH_MATMUL_SHAPES; do
            LIBTORCH_MATMUL_ARGS+=(--shape "$shape")
        done
        run_logged bench_sm120_libtorch_matmul "${LIBTORCH_MATMUL_ARGS[@]}"
    fi
    run_logged bench_sm120_cutedsl_matmul "$PYTHON_BIN" dev/bench_sm120_cutedsl_matmul.py
    run_logged bench_sm120_triton_matmul "$PYTHON_BIN" dev/triton/bench_sm120_matmul.py \
        --repeats "${LLMK_BENCH_REPEATS:-5}" \
        --large-repeats "${LLMK_BENCH_LARGE_REPEATS:-2}"
    run_logged bench_sm120_torch_attention "$PYTHON_BIN" dev/bench_sm120_torch_attention.py \
        --repeats "${LLMK_BENCH_REPEATS:-7}" \
        --warmup 3
    run_logged bench_sm120_cudnn_attention "$PYTHON_BIN" dev/bench_sm120_cudnn_attention.py \
        --repeats "${LLMK_BENCH_REPEATS:-7}" \
        --warmup 3
    run_logged bench_sm120_triton_attention "$PYTHON_BIN" dev/triton/bench_sm120_attention.py \
        --repeats "${LLMK_BENCH_REPEATS:-7}" \
        --warmup 3
    run_logged bench_sm120_torch_classifier "$PYTHON_BIN" dev/bench_sm120_torch_classifier.py \
        --repeats "${LLMK_BENCH_REPEATS:-7}" \
        --warmup 3
    run_logged bench_sm120_triton_classifier "$PYTHON_BIN" dev/triton/bench_sm120_classifier.py \
        --repeats "${LLMK_BENCH_REPEATS:-7}" \
        --warmup 3
    run_logged bench_sm120_layernorm_python_stacks "$PYTHON_BIN" dev/triton/bench_sm120_layernorm.py \
        --rows 65536 \
        --cols 768 3072 \
        --repeats "${LLMK_BENCH_REPEATS:-7}" \
        --warmup 3
    run_logged bench_sm120_triton_runtime "$PYTHON_BIN" dev/triton/bench_sm120_runtime.py \
        --repeats "${LLMK_BENCH_REPEATS:-7}" \
        --warmup 3
    run_logged bench_sm120_torch_runtime "$PYTHON_BIN" dev/bench_sm120_torch_runtime.py \
        --repeats "${LLMK_BENCH_REPEATS:-7}" \
        --warmup 3
    run_logged bench_sm120_libtorch_runtime "$PYTHON_BIN" dev/bench_sm120_libtorch_runtime.py \
        --route "$LLMK_LIBTORCH_RUNTIME_ROUTE" \
        --repeats "${LLMK_BENCH_REPEATS:-7}" \
        --warmup 3
    if [[ "$RUN_LIBTORCH_TRAINER_LINK_PROBE" == "1" ]]; then
        run_logged validate_libtorch_trainer_link "$PYTHON_BIN" dev/validate_libtorch_trainer_link.py
    fi
fi

if [[ "$RUN_TRAINING" == "1" ]]; then
    run_logged train_gpt2cu ./train_gpt2cu \
        -i "dev/data/tinystories/TinyStories_train.bin" \
        -j "dev/data/tinystories/TinyStories_val.bin" \
        -o "$TRAIN_OUT_DIR" \
        -v 250 -s 20000 -g 144 \
        -h 0 \
        -b 64 -t 1024 -d 524288 \
        -r 0 \
        -z "$TRAIN_ZERO_STAGE" \
        -c 0.1 \
        -l 0.0006 -q 0.0 -u 700 -n 5000 \
        -y 0 \
        -e "d12" \
        -x "$MAX_STEPS"
    cleanup_checkpoints
fi

if [[ "$RUN_CORRECTNESS" == "1" ]]; then
    summarize_correctness_logs \
        test_matmul \
        test_attention \
        test_layernorm \
        test_bias \
        test_gelu \
        test_fused_classifier \
        test_encoder \
        test_adamw \
        test_global_norm
fi
summarize_log_lines "Matmul Benchmarks" "$(log_path bench_sm120_matmul)" 'shape|TK|cuBLASLt|Forward|dInp|dWeight|us|ms'
summarize_log_lines "Attention Benchmarks" "$(log_path bench_sm120_attention)" 'Attention'
summarize_log_lines "LayerNorm Benchmarks" "$(log_path bench_sm120_layernorm)" 'LayerNorm'
summarize_log_lines "Runtime Benchmarks" "$(log_path bench_sm120_runtime)" 'bias|gelu|classifier|adamw|global_norm|encoder|memset|copy'
summarize_log_lines "Torch Matmul Benchmarks" "$(log_path bench_sm120_torch_matmul)" 'Torch|M=|fwd|dInp|dW'
summarize_log_lines "LibTorch C++ Matmul Benchmarks" "$(log_path bench_sm120_libtorch_matmul)" 'LibTorch|Torch C\+\+|M=|dW'
summarize_log_lines "CuTeDSL Matmul Benchmarks" "$(log_path bench_sm120_cutedsl_matmul)" 'CuTeDSL|unavailable|M='
summarize_log_lines "Triton Matmul Benchmarks" "$(log_path bench_sm120_triton_matmul)" 'Triton|M=|fwd|dInp|dW'
summarize_log_lines "Torch Attention Benchmarks" "$(log_path bench_sm120_torch_attention)" 'Attention'
summarize_log_lines "cuDNN Attention Benchmarks" "$(log_path bench_sm120_cudnn_attention)" 'Attention|unavailable|PyTorch CUDA context'
summarize_log_lines "Triton Attention Benchmarks" "$(log_path bench_sm120_triton_attention)" 'Attention|unavailable|PyTorch CUDA context'
summarize_log_lines "Torch Classifier Benchmarks" "$(log_path bench_sm120_torch_classifier)" 'classifier|unavailable|OOM'
summarize_log_lines "Triton Classifier Benchmarks" "$(log_path bench_sm120_triton_classifier)" 'classifier|unavailable|OOM'
summarize_log_lines "Python Stack LayerNorm Benchmarks" "$(log_path bench_sm120_layernorm_python_stacks)" 'LayerNorm'
summarize_log_lines "Triton Runtime Benchmarks" "$(log_path bench_sm120_triton_runtime)" 'bias|gelu'
summarize_log_lines "Torch Runtime Benchmarks" "$(log_path bench_sm120_torch_runtime)" 'bias|gelu|classifier|adamw|global_norm|encoder|memset|copy'
summarize_log_lines "LibTorch C++ Runtime Benchmarks" "$(log_path bench_sm120_libtorch_runtime)" 'memset|copy|gelu|Torch C\+\+|LibTorch'
summarize_log_lines "LibTorch Trainer Link Probe" "$(log_path validate_libtorch_trainer_link)" 'LibTorch trainer link|PASS|CUDA runtime context'
summarize_log_lines "Training Steps" "$(log_path train_gpt2cu)" 'use_master_weights|gelu_fusion|val loss|step[[:space:]]+[0-9]+/|total average iteration time'

if [[ "$RUN_ARTIFACT_VALIDATOR" == "1" && "$DRY_RUN" != "1" ]]; then
    VALIDATOR_ARGS=(
        --round-dir "$ARTIFACT_DIR"
        --write-scoreboard "$ARTIFACT_DIR/scoreboard-candidates.md"
        --write-selected-backends "$ARTIFACT_DIR/selected-backends.json"
        --write-promotion-candidates "$ARTIFACT_DIR/promotion-candidates.json"
        --require-manifest
    )
    if [[ "$RUN_STACK_PROBE" == "1" ]]; then
        VALIDATOR_ARGS+=(--require-stack-probe)
    fi
    if [[ "$RUN_CORRECTNESS" == "1" ]]; then
        VALIDATOR_ARGS+=(--require-correctness)
    fi
    if [[ "$RUN_BENCHMARKS" == "1" ]]; then
        VALIDATOR_ARGS+=(--require-benchmarks)
    fi
    if [[ "$RUN_TRAINING" == "1" ]]; then
        VALIDATOR_ARGS+=(--require-training)
        if [[ "$KEEP_CHECKPOINTS" != "1" ]]; then
            VALIDATOR_ARGS+=(--forbid-checkpoints)
        fi
    fi
    run_logged validate_sm120_round "$PYTHON_BIN" dev/validate_sm120_round.py "${VALIDATOR_ARGS[@]}"
fi

if [[ "$RUN_CURRENT_SELECTION_AUDIT" == "1" ]]; then
    SELECTION_ARGS=()
    AUDIT_ROUND_ARGS=()
    if [[ -n "$SM120_SELECTION_NATIVE_ROUND" ]]; then
        SELECTION_ARGS+=(--native-round "$SM120_SELECTION_NATIVE_ROUND")
        AUDIT_ROUND_ARGS+=(--native-round "$SM120_SELECTION_NATIVE_ROUND")
    fi
    if [[ -n "$SM120_SELECTION_OPTIONAL_ROUND" ]]; then
        SELECTION_ARGS+=(--optional-round "$SM120_SELECTION_OPTIONAL_ROUND")
        AUDIT_ROUND_ARGS+=(--optional-round "$SM120_SELECTION_OPTIONAL_ROUND")
    fi
    run_logged write_sm120_current_selection "$PYTHON_BIN" dev/write_sm120_current_selection.py \
        "${SELECTION_ARGS[@]}" \
        --json-out "$SM120_SELECTION_JSON_OUT" \
        --markdown-out "$SM120_SELECTION_MD_OUT"
    run_logged audit_sm120_optimization_goal "$PYTHON_BIN" dev/audit_sm120_optimization_goal.py \
        --selection-json "$SM120_SELECTION_JSON_OUT" \
        --selection-md "$SM120_SELECTION_MD_OUT" \
        "${AUDIT_ROUND_ARGS[@]}" \
        --json-out "$SM120_AUDIT_JSON_OUT" \
        --markdown-out "$SM120_AUDIT_MD_OUT"
fi

echo "Artifacts written to $ARTIFACT_DIR"
