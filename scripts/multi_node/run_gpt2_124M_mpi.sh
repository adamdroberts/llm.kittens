#!/usr/bin/env bash
set -euo pipefail

# Two-node GPT-2 124M run using MPI rendezvous for NCCL init.

make train_gpt2cu

BINARY_PATH="${BINARY_PATH:-$PWD/train_gpt2cu}"
OUT_DIR="${OUT_DIR:-/ephemeral/data/fineweb/log_gpt2_124M_multi}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/ephemeral/data/fineweb/bin_10B/fineweb_train_*.bin}"
VAL_DATA_PATH="${VAL_DATA_PATH:-/ephemeral/data/fineweb/bin_10B/fineweb_val_*.bin}"
HOST1="${HOST1:-h100-node-1-0}"
HOST2="${HOST2:-h100-node-1-1}"
HOSTS="${HOSTS:-$HOST1:8,$HOST2:8}"
NPROC="${NPROC:-16}"
MAX_STEPS="${MAX_STEPS:-18865}"

# If the filesystem is shared this is a no-op. Otherwise, copy the binary to workers.
if [ "${SKIP_BINARY_COPY:-0}" != "1" ]; then
    scp -r "$BINARY_PATH" "$USER@$HOST2:$BINARY_PATH"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"      # H100 NVLink-SHARP.
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"             # Tune to your IB HCA list.
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"  # GPUDirect RDMA.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"        # Use InfiniBand.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens17}"
export OMPI_MCA_btl_tcp_if_include="${OMPI_MCA_btl_tcp_if_include:-ens17}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-PXB}"

mpirun -np "$NPROC" --host "$HOSTS" \
    "$BINARY_PATH" \
    -i "$TRAIN_DATA_PATH" \
    -j "$VAL_DATA_PATH" \
    -o "$OUT_DIR" \
    -v 250 -s 20000 -g 144 \
    -h 1 \
    -b 64 -t 1024 \
    -d 2097152 \
    -r 0 \
    -z 1 \
    -c 0.1 \
    -l 0.0006 \
    -q 0.1 \
    -u 700 \
    -n 1000 \
    -y 0 \
    -x "$MAX_STEPS" \
    -e d12 \
    -pi mpi
