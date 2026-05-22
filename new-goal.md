optimise the kernels for performance starting with sm120, as that is locally testable.

You have the freedom to use any of kernel stacks:

thunderkittens 2.0 (TK)
cuBLAS / cuBLASLt / cuDNN
Triton
CuTeDSL

For each kernel that is used in the GPT-2 architecture, benchmark all of the kernel stacks seperately, then the figure out which combinations would give the best performance overall the attempt a training run

maintain a best runs markdown file for each kernel type: i.e attention any subtypes and all the stacks being tested:

thunderkittens
cuBLAS
cuBLASLt
cuDNN
Triton
CuTeDSL

The maintain a full list of architectures combinations that give the best results and any training runs tests

Frequently remove the model outputs, as they will fill the disk, but don't remove the log124M/5090_S folder

Use this GPT-2 training script:

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
    -x 3 

The baseline is llm.c 

step    1/1765 | loss 11.032000 (+nanz)| norm 22.1541 (+nanz)| lr 8.57e-07 | 3091.12 ms | -100.0% bf16 MFU | 169611 tok/s
step    2/1765 | loss 10.958870 (+nanz)| norm 22.0987 (+nanz)| lr 1.71e-06 | 2679.79 ms | -100.0% bf16 MFU | 195645 tok/s
step    3/1765 | loss 10.810900 (+nanz)| norm 21.1370 (+nanz)| lr 2.57e-06 | 2683.77 ms | -100.0% bf16 MFU | 195496 tok/s

and our current best run for this repo

step    1/10 | loss 11.032356 (+nanz)| norm 22.1413 (+nanz)| lr 8.57e-07 | 2469.31 ms | 40.7% bf16 MFU | 212321 tok/s
step    2/10 | loss 10.958524 (+nanz)| norm 22.0967 (+nanz)| lr 1.71e-06 | 2469.37 ms | 40.7% bf16 MFU | 212316 tok/s
step    3/10 | loss 10.811322 (+nanz)| norm 21.1251 (+nanz)| lr 2.57e-06 | 2470.85 ms | 40.7% bf16 MFU | 212251 tok/s
