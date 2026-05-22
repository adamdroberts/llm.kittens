#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

./train_gpt2cu \
    -i "dev/data/tinystories/TinyStories_train.bin" \
    -j "dev/data/tinystories/TinyStories_val.bin" \
    -o "log124M/5090_S" \
    -v 250 -s 20000 -g 144 \
    -h 0 \
    -b 64 -t 1024 -d 524288 \
    -r 0 \
    -z 1 \
    -c 0.1 \
    -l 0.0006  -q 0.0 -u 700 -n 5000 \
    -y 0  \
    -e "d12" \
    -x 20000 
