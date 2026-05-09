#!/usr/bin/env bash
set -euo pipefail

# GPT-2 (1558M) repro on FineWeb-EDU, ported from llm.c/scripts/run_gpt2_1558M.sh.
# 32,000 steps of 1,048,576 tokens/step. Target: 8xH100, ZeRO-1.

make train_gpt2cu

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

MPIRUN="${MPIRUN:-mpirun}"
NPROC="${NPROC:-8}"
TRAIN_BIN="${TRAIN_BIN:-./train_gpt2cu}"
OUT_DIR="${OUT_DIR:-log_gpt2_1558M}"
MAX_STEPS="${MAX_STEPS:-32000}"
printf -v done_tag "%08d" "$MAX_STEPS"
DONE_FILE="${DONE_FILE:-$OUT_DIR/DONE_$done_tag}"

while true; do
    if [ -f "$DONE_FILE" ]; then
        echo "File $DONE_FILE exists. Exiting the loop."
        break
    fi

    "$MPIRUN" -np "$NPROC" "$TRAIN_BIN" \
        -i "dev/data/edu_fineweb100B/edu_fineweb_train_*.bin" \
        -j "dev/data/edu_fineweb100B/edu_fineweb_val_*.bin" \
        -o "$OUT_DIR" \
        -v 250 -s 300000 -g 384 \
        -h 1 \
        -b 16 -t 1024 \
        -d 1048576 \
        -r 0 \
        -z 1 \
        -c 0.1 \
        -k "cosine" \
        -l 0.0006 \
        -q 0.1 \
        -u 700 \
        -n 2000 \
        -x "$MAX_STEPS" \
        -ge 1 \
        -y 1 \
        -e "d48"

    sleep 1
done
