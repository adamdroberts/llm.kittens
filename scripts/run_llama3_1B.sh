#!/usr/bin/env bash
set -euo pipefail

# Llama-3 1B single-node target for M6.
# Target: 8xH100, ZeRO-1, B=32, T=2048, 30B tokens at 524,288 tokens/step.
# Prep data with: python dev/data/fineweb.py --type edu --version 100B --model_desc llama-3
# Prep eval with: python dev/data/hellaswag.py --model_desc llama-3

make train_llama3cu

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"      # H100 NVLink-SHARP.
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"             # Mellanox/NVIDIA IB HDR/NDR; tune per pod.
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"  # GPUDirect RDMA.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"        # Use InfiniBand when available.

MPIRUN="${MPIRUN:-mpirun}"
NPROC="${NPROC:-8}"
TRAIN_BIN="${TRAIN_BIN:-./train_llama3cu}"
MODEL_DESC="${MODEL_DESC:-llama3:1B}"
OUT_DIR="${OUT_DIR:-log_llama3_1B}"
TRAIN_DATA_PATTERN="${TRAIN_DATA_PATTERN:-dev/data/edu_fineweb100B/edu_fineweb_train_*.bin}"
VAL_DATA_PATTERN="${VAL_DATA_PATTERN:-dev/data/edu_fineweb100B/edu_fineweb_val_*.bin}"
MAX_STEPS="${MAX_STEPS:-57220}"
RUN_LOG="${RUN_LOG:-$OUT_DIR/run.log}"
printf -v done_tag "%08d" "$MAX_STEPS"
DONE_FILE="${DONE_FILE:-$OUT_DIR/DONE_$done_tag}"

mkdir -p "$OUT_DIR"
{
    echo "llm.kittens Llama-3 1B launch"
    echo "NPROC=$NPROC"
    echo "TRAIN_BIN=$TRAIN_BIN"
    echo "MODEL_DESC=$MODEL_DESC"
    echo "OUT_DIR=$OUT_DIR"
    echo "MAX_STEPS=$MAX_STEPS"
    echo "TRAIN_DATA_PATTERN=$TRAIN_DATA_PATTERN"
    echo "VAL_DATA_PATTERN=$VAL_DATA_PATTERN"
    echo "B=32 T=2048 D=524288 ZERO_STAGE=1 RECOMPUTE=0"
} > "$RUN_LOG"

while true; do
    if [ -f "$DONE_FILE" ]; then
        echo "File $DONE_FILE exists. Exiting the loop."
        break
    fi

    "$MPIRUN" -np "$NPROC" "$TRAIN_BIN" \
        -i "$TRAIN_DATA_PATTERN" \
        -j "$VAL_DATA_PATTERN" \
        -o "$OUT_DIR" \
        -v 250 -m 20 -s 20000 -g 144 \
        -h 1 \
        -b 32 -t 2048 \
        -d 524288 \
        -r 0 \
        -z 1 \
        -c 0.1 \
        -l 0.0003 \
        -q 0.1 \
        -u 2000 \
        -n 5000 \
        -nk 5 \
        -nm 50000 \
        -sl 7.0 \
        -sg 7.0 \
        -y 1 \
        -x "$MAX_STEPS" \
        -e "$MODEL_DESC"

    sleep 1
done
