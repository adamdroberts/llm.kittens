#!/usr/bin/env bash
#SBATCH --job-name=llmk-gpt2-tcp
#SBATCH --output=%x_%j_%t.log
#SBATCH --error=%x_%j_%t.err
#SBATCH --partition=llmc
#SBATCH --ntasks=16
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8

set -euo pipefail

# Two-node GPT-2 124M run using TCP rendezvous for NCCL init.
# Set SERVER_IP to the rank-zero node IP for your pod.

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Run this script inside Slurm with sbatch."
    exit 1
fi

make train_gpt2cu NO_USE_MPI=1

BINARY_PATH="${BINARY_PATH:-$PWD/train_gpt2cu}"
OUT_DIR="${OUT_DIR:-/ephemeral/data/fineweb/log_gpt2_124M_multi}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/ephemeral/data/fineweb/bin_10B/fineweb_train_*.bin}"
VAL_DATA_PATH="${VAL_DATA_PATH:-/ephemeral/data/fineweb/bin_10B/fineweb_val_*.bin}"
SERVER_IP="${SERVER_IP:-10.0.1.220}"
MAX_STEPS="${MAX_STEPS:-18865}"

current_user="${USER:?USER must be set}"
hosts=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
current_host=$(hostname)
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
echo "$datestring"

srun -l -u bash -c "
    '$BINARY_PATH' \
    -i '$TRAIN_DATA_PATH' \
    -j '$VAL_DATA_PATH' \
    -o '$OUT_DIR' \
    -v 250 -s 20000 -g 144 \
    -h 1 \
    -b 64 -t 1024 \
    -d 2097152 \
    -r 0 \
    -z 1 \
    -c 0.1 \
    -l 0.0006 \
    -q 0.0 \
    -u 700 \
    -n 5000 \
    -y 1 \
    -x '$MAX_STEPS' \
    -e d12 \
    -pn \$SLURM_NTASKS \
    -pr \$SLURM_PROCID \
    -pg \$SLURM_NTASKS_PER_NODE \
    -ps '$SERVER_IP' \
    -pi tcp
"

echo "$datestring"
