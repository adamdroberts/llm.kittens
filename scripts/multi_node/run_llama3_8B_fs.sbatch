#!/usr/bin/env bash
#SBATCH --job-name=llmk-llama3-8b-fs
#SBATCH --output=%x_%j_%t.log
#SBATCH --error=%x_%j_%t.err
#SBATCH --partition=llmc
#SBATCH --ntasks=16
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8

set -euo pipefail

# Two-node Llama-3 8B run using filesystem rendezvous for NCCL init.
# Run with: sbatch scripts/multi_node/run_llama3_8B_fs.sbatch
# Override MODEL_DESC=llama3.1:8B SEQ_LEN=8192 for the longer-context variant.

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Run this script inside Slurm with sbatch."
    exit 1
fi

make train_llama3cu NO_USE_MPI=1

BINARY_PATH="${BINARY_PATH:-$PWD/train_llama3cu}"
MODEL_DESC="${MODEL_DESC:-llama3:8B}"
OUT_DIR="${OUT_DIR:-/ephemeral/data/fineweb/log_llama3_8B_multi}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/ephemeral/data/fineweb/bin_100B_llama3/fineweb_train_*.bin}"
VAL_DATA_PATH="${VAL_DATA_PATH:-/ephemeral/data/fineweb/bin_100B_llama3/fineweb_val_*.bin}"
SYNC_FS_PATH="${SYNC_FS_PATH:-$OUT_DIR}"
MICRO_BATCH="${MICRO_BATCH:-4}"
SEQ_LEN="${SEQ_LEN:-2048}"
TOTAL_BATCH="${TOTAL_BATCH:-524288}"
MAX_STEPS="${MAX_STEPS:-57220}"
ZERO_STAGE="${ZERO_STAGE:-2}"
RUN_LOG="${RUN_LOG:-$OUT_DIR/run.log}"
SLURM_JOB_NODE_COUNT="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}}"

mkdir -p "$OUT_DIR"
{
    echo "llm.kittens Llama-3 8B Slurm launch"
    echo "SLURM_JOB_ID=$SLURM_JOB_ID"
    echo "SLURM_NTASKS=${SLURM_NTASKS:-}"
    echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NODE_COUNT"
    echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-}"
    echo "MODEL_DESC=$MODEL_DESC"
    echo "OUT_DIR=$OUT_DIR"
    echo "MAX_STEPS=$MAX_STEPS"
    echo "TRAIN_DATA_PATH=$TRAIN_DATA_PATH"
    echo "VAL_DATA_PATH=$VAL_DATA_PATH"
    echo "SYNC_FS_PATH=$SYNC_FS_PATH"
    echo "B=$MICRO_BATCH T=$SEQ_LEN D=$TOTAL_BATCH ZERO_STAGE=$ZERO_STAGE RECOMPUTE=1 INIT=fs"
} > "$RUN_LOG"

current_user="${USER:?USER must be set}"
hosts=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
current_host=$(hostname)
echo "HOSTS=$hosts" >> "$RUN_LOG"
if [ "${SKIP_BINARY_COPY:-0}" != "1" ]; then
    for host in $hosts; do
        if [ "$host" = "$current_host" ]; then
            continue
        fi
        echo "copying $BINARY_PATH to $current_user@$host"
        scp -r "$BINARY_PATH" "$current_user@$host:$BINARY_PATH"
    done
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"      # H100 NVLink-SHARP.
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"             # Tune to your IB HCA list.
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"  # GPUDirect RDMA.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"        # Use InfiniBand.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens17}"
export OMPI_MCA_btl_tcp_if_include="${OMPI_MCA_btl_tcp_if_include:-ens17}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-PXB}"

datestring=$(date "+%Y-%m-%dT%H:%M:%S")
echo "Running in Slurm job $SLURM_JOB_ID as $current_user"
echo "Running on hosts: $hosts"
echo "Model descriptor: $MODEL_DESC, B=$MICRO_BATCH, T=$SEQ_LEN, total batch=$TOTAL_BATCH, ZeRO=$ZERO_STAGE"
echo "$datestring"

srun -l -u bash -c "
    '$BINARY_PATH' \
    -i '$TRAIN_DATA_PATH' \
    -j '$VAL_DATA_PATH' \
    -o '$OUT_DIR' \
    -v 250 -m 20 -s 20000 -g 144 \
    -h 1 \
    -b '$MICRO_BATCH' -t '$SEQ_LEN' \
    -d '$TOTAL_BATCH' \
    -r 1 \
    -z '$ZERO_STAGE' \
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
    -x '$MAX_STEPS' \
    -e '$MODEL_DESC' \
    -pn \$SLURM_NTASKS \
    -pr \$SLURM_PROCID \
    -pg \$SLURM_NTASKS_PER_NODE \
    -pf '$SYNC_FS_PATH' \
    -pi fs
"

echo "$datestring"
