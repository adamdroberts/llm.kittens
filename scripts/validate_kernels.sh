#!/usr/bin/env bash
# Run the per-kernel pytest suite. Two tiers and three execution targets are
# available:
#
#   default                  → tests/kernels (CPU-reference smoke)
#   --parity                 → tests/parity  (llm.c / PyTorch parity)
#
#   no flag                  → local default DEVICE_ARCH=SM90 (H100 host expected)
#   --sm120                  → local DEVICE_ARCH=SM120 (RTX 5090 / Blackwell consumer)
#   USE_RUNPOD_FLASH=1 (env) → remote H100 via runpod-flash
#
# Examples:
#   scripts/validate_kernels.sh                               # CPU-ref smoke, local SM90
#   scripts/validate_kernels.sh --parity --sm120              # parity vs llm.c on local RTX 5090
#   USE_RUNPOD_FLASH=1 scripts/validate_kernels.sh --parity   # parity on remote H100 (authoritative)
#
# Note on local --sm120 + --parity: on RTX 5090 the TK fast paths fall back
# to the same CUDA kernels llm.c uses, so Family A tests pass with
# bit-exact diffs. This proves the harness plumbing end-to-end (build,
# probe execution, .npy roundtrip, max_abs_diff) without paying for a
# remote H100. The TK Hopper fast paths are only exercised by
# USE_RUNPOD_FLASH=1.
set -euo pipefail
cd "$(dirname "$0")/.."

TIER=tests/kernels
PASS_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --parity)
            TIER=tests/parity
            ;;
        --sm120)
            # Local DEVICE_ARCH=SM120 (compile for RTX 5090 / Blackwell).
            export DEVICE_ARCH=SM120
            ;;
        --remote-rtx5090)
            # Remote RTX 5090 via flash. Forces both env vars so the flash
            # runner picks ADA_32_PRO and the worker's `make` uses SM120.
            export USE_RUNPOD_FLASH=1
            export RUNPOD_GPU=ADA_32_PRO
            export DEVICE_ARCH=SM120
            ;;
        --remote-h100)
            # Remote H100 via flash — the primary validation target.
            # Exercises real TK Hopper kernels (WGMMA + TMA).
            export USE_RUNPOD_FLASH=1
            export RUNPOD_GPU=ADA_80_PRO
            export DEVICE_ARCH=SM90
            ;;
        *)
            PASS_ARGS+=("$arg")
            ;;
    esac
done

echo "==> tier=$TIER  DEVICE_ARCH=${DEVICE_ARCH:-SM90}  USE_RUNPOD_FLASH=${USE_RUNPOD_FLASH:-0}  RUNPOD_GPU=${RUNPOD_GPU:-}"
exec pytest "$TIER" -v --junitxml=.pytest_junit.xml "${PASS_ARGS[@]}"
