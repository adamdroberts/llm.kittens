#!/usr/bin/env bash
set -euo pipefail

# GPT-3 (125M) repro on FineWeb, ported from llm.c/scripts/run_gpt3_125M.sh.
# Context length 2048, 572,204 steps of 524,288 tokens/step.

make train_gpt2cu

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

MPIRUN="${MPIRUN:-mpirun}"
NPROC="${NPROC:-8}"
TRAIN_BIN="${TRAIN_BIN:-./train_gpt2cu}"
OUT_DIR="${OUT_DIR:-log_gpt3_125M}"
MAX_STEPS="${MAX_STEPS:-572204}"
printf -v done_tag "%08d" "$MAX_STEPS"
DONE_FILE="${DONE_FILE:-$OUT_DIR/DONE_$done_tag}"

while true; do
    if [ -f "$DONE_FILE" ]; then
        echo "File $DONE_FILE exists. Exiting the loop."
        break
    fi

    "$MPIRUN" -np "$NPROC" "$TRAIN_BIN" \
        -i "dev/data/fineweb100B/fineweb_train_*.bin" \
        -j "dev/data/fineweb100B/fineweb_val_*.bin" \
        -o "$OUT_DIR" \
        -v 250 -s 20000 -g 144 \
        -h 1 \
        -b 32 -t 2048 \
        -d 524288 \
        -r 0 \
        -z 1 \
        -c 0.1 \
        -l 0.0006 \
        -q 0.1 \
        -u 700 \
        -n 10000 \
        -nk 5 \
        -nm 50000 \
        -ge 1 \
        -sl 7.0 \
        -sg 7.0 \
        -y 1 \
        -x "$MAX_STEPS" \
        -e "gpt3:c768"

    sleep 1
done
