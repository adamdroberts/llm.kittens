#!/bin/bash
# sweep_heuristics.sh - Sweep cuBLASLt heuristic indices for SM120.

export DEVICE_ARCH=SM120

for i in {0..7}; do
    echo "Testing Heuristic Index: $i"
    make clean > /dev/null 2>&1
    make train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_CACHE_CUBLASLT_PLANS -DLLMK_SM120_CUBLASLT_HEURISTIC_INDEX=$i" > build_h$i.log 2>&1
    if [ $? -ne 0 ]; then
        echo "Build failed for index $i. Check build_h$i.log"
        continue
    fi
    grep "NVCC arch" build_h$i.log
    ./train_gpt2cu -i "dev/data/tinystories/TinyStories_train.bin" -j "dev/data/tinystories/TinyStories_val.bin" -o "log124M/5090_S_h$i" -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5 -y 0 -e "d12" -x 5 > "log124M/5090_S_h$i.log" 2>&1
    avg=$(grep "^step" "log124M/5090_S_h$i.log" | head -n 4 | awk '{sum+=$13} END {if (NR>0) print sum/NR; else print "N/A"}')
    echo "Index $i Average (first 4 steps): $avg ms"
done

echo "Testing MIN_WAVES"
make clean > /dev/null 2>&1
make train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_CACHE_CUBLASLT_PLANS -DLLMK_SM120_CUBLASLT_SELECT_MIN_WAVES" > build_min.log 2>&1
grep "NVCC arch" build_min.log
./train_gpt2cu -i "dev/data/tinystories/TinyStories_train.bin" -j "dev/data/tinystories/TinyStories_val.bin" -o "log124M/5090_S_min" -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5 -y 0 -e "d12" -x 5 > "log124M/5090_S_min.log" 2>&1
avg=$(grep "^step" "log124M/5090_S_min.log" | head -n 4 | awk '{sum+=$13} END {if (NR>0) print sum/NR; else print "N/A"}')
echo "MIN_WAVES Average: $avg ms"

echo "Testing MAX_WAVES"
make clean > /dev/null 2>&1
make train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1 EXTRA_NVCC_FLAGS="-DLLMK_SM120_CACHE_CUBLASLT_PLANS -DLLMK_SM120_CUBLASLT_SELECT_MAX_WAVES" > build_max.log 2>&1
grep "NVCC arch" build_max.log
./train_gpt2cu -i "dev/data/tinystories/TinyStories_train.bin" -j "dev/data/tinystories/TinyStories_val.bin" -o "log124M/5090_S_max" -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5 -y 0 -e "d12" -x 5 > "log124M/5090_S_max.log" 2>&1
avg=$(grep "^step" "log124M/5090_S_max.log" | head -n 4 | awk '{sum+=$13} END {if (NR>0) print sum/NR; else print "N/A"}')
echo "MAX_WAVES Average: $avg ms"
