#!/usr/bin/env bash
set -euo pipefail

# GPT-2 (124M) repro on FineWeb, ported from llm.c/scripts/run_gpt2_124M.sh.
# 18,865 steps of 524,288 tokens/step. Target: 8xH100, ZeRO-1.

make train_gpt2cu

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"      # H100 NVLink-SHARP.
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"             # Mellanox/NVIDIA IB HDR/NDR; tune per pod.
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"  # GPUDirect RDMA.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"        # Use InfiniBand when available.

MPIRUN="${MPIRUN:-mpirun}"
NPROC="${NPROC:-8}"
TRAIN_BIN="${TRAIN_BIN:-./train_gpt2cu}"
OUT_DIR="${OUT_DIR:-log_gpt2_124M}"
MAX_STEPS="${MAX_STEPS:-18865}"
RUN_LOG="${RUN_LOG:-$OUT_DIR/run.log}"
printf -v done_tag "%08d" "$MAX_STEPS"
DONE_FILE="${DONE_FILE:-$OUT_DIR/DONE_$done_tag}"

mkdir -p "$OUT_DIR"
{
    echo "llm.kittens GPT-2 124M launch"
    echo "NPROC=$NPROC"
    echo "TRAIN_BIN=$TRAIN_BIN"
    echo "OUT_DIR=$OUT_DIR"
    echo "MAX_STEPS=$MAX_STEPS"
    echo "TRAIN_DATA_PATTERN=dev/data/fineweb10B/fineweb_train_*.bin"
    echo "VAL_DATA_PATTERN=dev/data/fineweb10B/fineweb_val_*.bin"
    echo "B=64 T=1024 D=524288 ZERO_STAGE=1 RECOMPUTE=0 MODEL=d12"
} > "$RUN_LOG"

while true; do
    if [ -f "$DONE_FILE" ]; then
        echo "File $DONE_FILE exists. Exiting the loop."
        break
    fi

    # run python dev/data/fineweb.py --version 10B to prep data
    # run python dev/data/hellaswag.py to prep HellaSwag eval
    "$MPIRUN" -np "$NPROC" "$TRAIN_BIN" \
        -i "dev/data/fineweb10B/fineweb_train_*.bin" \
        -j "dev/data/fineweb10B/fineweb_val_*.bin" \
        -o "$OUT_DIR" \
        -v 250 -s 20000 -g 144 \
        -h 1 \
        -b 64 -t 1024 \
        -d 524288 \
        -r 0 \
        -z 1 \
        -c 0.1 \
        -l 0.0006 \
        -q 0.0 \
        -u 700 \
        -n 5000 \
        -y 1 \
        -x "$MAX_STEPS" \
        -e "d12"

    sleep 1
done
