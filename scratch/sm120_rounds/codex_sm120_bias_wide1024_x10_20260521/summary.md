# SM120 Optimization Round

- run label: `codex_sm120_bias_wide1024_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_bias_wide1024_x10_20260521`
- max steps: `10`
- python: `/home/adam/miniconda3/envs/llm-kittens/bin/python`
- cuDNN packed backward route: `saved-forward`
- LibTorch runtime route: `cxx-api-raw-pointer`
- LibTorch runtime supplemental shapes: `gelu_forward`
- LibTorch trainer link probe: `0`
- LibTorch matmul shapes: `qkv attproj fc fcproj lmhead`
- SM120 LibTorch trainer memory route: `0`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- working tree: `527` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Thu May 21 16:56:57 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0  On |                  N/A |
|100%   33C    P5             57W /  575W |    3103MiB /  32607MiB |      1%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1 SM120_USE_LIBTORCH_MEMORY=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0 PYTHON_BIN=/home/adam/miniconda3/envs/llm-kittens/bin/python`


## write_sm120_round_manifest
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521/round-manifest.md --run-label codex_sm120_bias_wide1024_x10_20260521 --artifact-dir scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521 --train-out-dir log124M/5090_S_codex_sm120_bias_wide1024_x10_20260521 --max-steps 10 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 0 --run-correctness 0 --run-benchmarks 0 --run-python-stack-benchmarks 0 --cudnn-packed-backward-route saved-forward --libtorch-runtime-route cxx-api-raw-pointer --libtorch-runtime-supplemental-shapes gelu_forward --run-libtorch-trainer-link-probe 0 --run-libtorch-matmul-benchmarks 0 --libtorch-matmul-shapes qkv\ attproj\ fc\ fcproj\ lmhead --sm120-use-libtorch-memory 0 --sm120-use-libtorch-grad-zero 0 --sm120-use-libtorch-dresidual-zero 0 --run-training 1 --keep-checkpoints 0`


## train_gpt2cu
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_bias_wide1024_x10_20260521 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 10`


## Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521/bench_sm120_matmul.log
```

## Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521/bench_sm120_attention.log
```

## LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521/bench_sm120_layernorm.log
```

## Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521/bench_sm120_runtime.log
```

## Torch Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521/bench_sm120_torch_matmul.log
```

## LibTorch C++ Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521/bench_sm120_libtorch_matmul.log
```

## CuTeDSL Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521/bench_sm120_cutedsl_matmul.log
```

## Triton Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521/bench_sm120_triton_matmul.log
```

## Torch Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521/bench_sm120_torch_attention.log
```

## cuDNN Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521/bench_sm120_cudnn_attention.log
```

## Triton Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521/bench_sm120_triton_attention.log
```

## Torch Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521/bench_sm120_torch_classifier.log
```

## Triton Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521/bench_sm120_triton_classifier.log
```

## Python Stack LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521/bench_sm120_layernorm_python_stacks.log
```

## Triton Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521/bench_sm120_triton_runtime.log
```

## Torch Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521/bench_sm120_torch_runtime.log
```

## LibTorch C++ Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521/bench_sm120_libtorch_runtime.log
```

## LibTorch Trainer Link Probe

```text
missing: scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521/validate_libtorch_trainer_link.log
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/10 | loss 11.032358 (+nanz)| norm 22.1414 (+nanz)| lr 8.57e-07 | 2500.65 ms | 40.2% bf16 MFU | 209661 tok/s
step    2/10 | loss 10.958507 (+nanz)| norm 22.0968 (+nanz)| lr 1.71e-06 | 2498.20 ms | 40.2% bf16 MFU | 209866 tok/s
step    3/10 | loss 10.811316 (+nanz)| norm 21.1251 (+nanz)| lr 2.57e-06 | 2500.02 ms | 40.2% bf16 MFU | 209788 tok/s
step    4/10 | loss 10.610130 (+nanz)| norm 18.7014 (+nanz)| lr 3.43e-06 | 2504.35 ms | 40.1% bf16 MFU | 209635 tok/s
step    5/10 | loss 10.392586 (+nanz)| norm 15.0184 (+nanz)| lr 4.29e-06 | 2508.12 ms | 40.1% bf16 MFU | 209473 tok/s
step    6/10 | loss 10.186255 (+nanz)| norm 12.0843 (+nanz)| lr 5.14e-06 | 2512.18 ms | 40.0% bf16 MFU | 209302 tok/s
step    7/10 | loss 10.010621 (+nanz)| norm 10.2002 (+nanz)| lr 6.00e-06 | 2513.50 ms | 40.0% bf16 MFU | 209167 tok/s
step    8/10 | loss 9.855870 (+nanz)| norm 8.7905 (+nanz)| lr 6.86e-06 | 2515.15 ms | 40.0% bf16 MFU | 209049 tok/s
step    9/10 | loss 9.719423 (+nanz)| norm 7.4665 (+nanz)| lr 7.71e-06 | 2517.50 ms | 39.9% bf16 MFU | 208931 tok/s
step   10/10 | loss 9.588612 (+nanz)| norm 6.3099 (+nanz)| lr 8.57e-06 | 2519.30 ms | 39.9% bf16 MFU | 208820 tok/s
val loss 9.483727
total average iteration time: 2509.813786 ms
```
