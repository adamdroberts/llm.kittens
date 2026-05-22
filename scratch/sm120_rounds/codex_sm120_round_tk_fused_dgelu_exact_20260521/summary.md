# SM120 Optimization Round

- run label: `codex_sm120_round_tk_fused_dgelu_exact_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_tk_fused_dgelu_exact_20260521`
- max steps: `3`
- python: `/home/adam/miniconda3/envs/llm-kittens/bin/python`
- git commit: `0f21747`
- working tree: `485` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Thu May 21 03:00:32 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0 Off |                  N/A |
|100%   32C    P8             47W /  575W |    1112MiB /  32607MiB |      1%      Default |
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

## probe_sm120_backend_stacks
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521/round-manifest.md --run-label codex_sm120_round_tk_fused_dgelu_exact_20260521 --artifact-dir scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521 --train-out-dir log124M/5090_S_codex_sm120_round_tk_fused_dgelu_exact_20260521 --max-steps 3 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 1 --run-benchmarks 1 --run-python-stack-benchmarks 0 --run-training 1 --keep-checkpoints 0`


## test_matmul
Command: `./test_matmul`


## test_attention
Command: `./test_attention`


## test_layernorm
Command: `./test_layernorm`


## test_bias
Command: `./test_bias`


## test_gelu
Command: `./test_gelu`


## test_fused_classifier
Command: `./test_fused_classifier`


## test_encoder
Command: `./test_encoder`


## test_adamw
Command: `./test_adamw`


## test_global_norm
Command: `./test_global_norm`


## bench_sm120_matmul
Command: `./bench_sm120_matmul`


## bench_sm120_attention
Command: `./bench_sm120_attention`


## bench_sm120_layernorm
Command: `./bench_sm120_layernorm`


## bench_sm120_runtime
Command: `./bench_sm120_runtime`


## train_gpt2cu
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_round_tk_fused_dgelu_exact_20260521 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 3`


## Correctness Markers

```text
[test_matmul]
  max abs diff = 0.1250  (tolerance 0.50)  PASS
  max abs diff = 0.2500  (tolerance 0.50)  PASS
  max abs diff = 0.2500  (tolerance 0.50)  PASS
  pre-GELU max abs diff = 0.2500  GELU max abs diff = 0.2500  (tolerance 0.50)  PASS
  max abs diff = 0.1250  (tolerance 0.50)  PASS
  max abs diff = 0.1250  (tolerance 0.50)  PASS
  max abs diff = 0.1250  (tolerance 0.50)  PASS
  max abs diff = 0.1250  (tolerance 0.50)  PASS
  max abs diff = 0.0312  (tolerance 0.50)  PASS
  max abs diff = 0.1250  (tolerance 0.50)  PASS
──── 10/10 passed ────
test_matmul smoke OK
[test_attention]
forward max abs diff  = 0.001027 (tol 0.080) PASS
backward max abs diff = 0.000969 (tol 0.200) PASS
forward max abs diff  = 0.001129 (tol 0.080) PASS
backward max abs diff = 0.000933 (tol 0.200) PASS
forward max abs diff  = 0.000877 (tol 0.080) PASS
backward max abs diff = 0.001126 (tol 0.200) PASS
test_attention smoke OK
[test_layernorm]
forward out max abs diff       = 0.007770 (tol 0.080) PASS
forward mean max abs diff      = 0.000000 (tol 0.050) PASS
forward rstd max abs diff      = 0.000001 (tol 0.050) PASS
fused residual max abs diff    = 0.001953 (tol 0.080) PASS
fused out max abs diff         = 0.007801 (tol 0.080) PASS
fused mean max abs diff        = 0.000000 (tol 0.050) PASS
fused rstd max abs diff        = 0.000001 (tol 0.050) PASS
backward dinp max abs diff     = 0.003643 (tol 0.120) PASS
backward dweight max abs diff  = 0.001824 (tol 0.120) PASS
backward dbias max abs diff    = 0.001953 (tol 0.120) PASS
test_layernorm smoke OK
[test_bias]
bias add hidden aligned max abs diff = 0.003906 (tol 0.010) PASS
bias add mlp aligned max abs diff = 0.003906 (tol 0.010) PASS
bias add unaligned fallback max abs diff = 0.003906 (tol 0.010) PASS
bias grad max abs diff = 0.072189 (tol 0.25) PASS
test_bias smoke OK
[test_gelu]
forward  max abs diff = 0.007755 (tol 0.020) PASS
backward max abs diff = 0.003887 (tol 0.020) PASS
test_gelu smoke OK
[test_fused_classifier]
loss-only loss max abs diff = 0.000001 (tol 0.0050) PASS
loss-only logits max abs diff = 0.000000 (tol 0.0000) PASS
loss    max abs diff = 0.000001 (tol 0.0050) PASS
dlogits max abs diff = 0.000173 (tol 0.0010) PASS
test_fused_classifier smoke OK
[test_encoder]
forward max abs diff = 0.000488 (tol 0.010) PASS
test_encoder smoke OK
[test_adamw]
master max abs diff = 7.451e-09 (tol 1.0e-05) PASS
m      max abs diff = 1.863e-09 (tol 1.0e-05) PASS
v      max abs diff = 7.276e-11 (tol 1.0e-05) PASS
bf16 param vs master max abs diff = 4.736e-04 (tol 5.0e-03) PASS
no-master m max abs diff = 1.863e-09 (tol 1.0e-05) PASS
no-master v max abs diff = 7.276e-11 (tol 1.0e-05) PASS
no-master bf16 param vs ref max abs diff = 4.744e-04 (tol 5.0e-03) PASS
test_adamw smoke OK
[test_global_norm]
cpu norm = 109.432192  gpu norm = 109.432190  relative diff = 0.000000 (tol 0.010) PASS
reset-tail cpu norm = 104.651146  gpu norm = 104.651150  relative diff = 0.000000 (tol 0.010) PASS
test_global_norm smoke OK
```

## Matmul Benchmarks

```text
  fwd      TK   1091.24 us | cuBLASLt   1039.59 us | cuBLAS   1429.97 us | TK/cuBLASLt 1.05x
  dInp   TK   1087.54 us | cuBLASLt   1010.35 us | cuBLAS   1030.59 us | TK/cuBLASLt 1.08x
  dW     TK   1456.68 us | cuBLASLt   1164.54 us | cuBLAS    993.49 us | TK/cuBLASLt 1.25x
  dW+accum TK   1462.91 us | cuBLASLt   1118.33 us | cuBLAS    995.23 us | TK/cuBLASLt 1.31x
  fwd      TK    375.97 us | cuBLASLt    369.67 us | cuBLAS    483.61 us | TK/cuBLASLt 1.02x
  dInp   TK    380.85 us | cuBLASLt    369.19 us | cuBLAS    365.60 us | TK/cuBLASLt 1.03x
  dW     TK    544.30 us | cuBLASLt    377.14 us | cuBLAS    327.26 us | TK/cuBLASLt 1.44x
  dW+accum TK    542.32 us | cuBLASLt    373.91 us | cuBLAS    331.40 us | TK/cuBLASLt 1.45x
  fwd+GeLU TK fused   1557.48 us | TK explicit   1989.04 us | cuBLASLt   1524.06 us | cuBLAS explicit   2456.63 us | explicit/cuBLASLt 1.31x
  dInp   TK   1495.19 us | cuBLASLt   1348.97 us | cuBLAS   1333.17 us | TK/cuBLASLt 1.11x
  dW     TK   1727.53 us | cuBLASLt   1519.17 us | cuBLAS   1308.35 us | TK/cuBLASLt 1.14x
  dW+accum TK   1743.96 us | cuBLASLt   1480.65 us | cuBLAS   1329.32 us | TK/cuBLASLt 1.18x
  fwd      TK   1419.51 us | cuBLASLt   1355.54 us | cuBLAS   1570.69 us | TK/cuBLASLt 1.05x
  dInp   TK   1496.76 us | cuBLASLt   1371.67 us | cuBLAS   1366.10 us | TK/cuBLASLt 1.09x
  dInp+dGeLU TK   1786.79 us | cuBLASLt fused   1815.83 us | cuBLASLt explicit   2180.73 us | cuBLAS explicit   2151.63 us | explicit/fused 1.20x
  dW     TK   1744.93 us | cuBLASLt   1478.04 us | cuBLAS   1324.97 us | TK/cuBLASLt 1.18x
  dW+accum TK   1714.08 us | cuBLASLt   1515.09 us | cuBLAS   1317.96 us | TK/cuBLASLt 1.13x
  fwd      TK  27637.97 us | cuBLASLt  22205.74 us | cuBLAS  22099.86 us | TK/cuBLASLt 1.24x
  dInp   TK  24057.64 us | cuBLASLt  21776.60 us | cuBLAS  21381.17 us | TK/cuBLASLt 1.10x
  dW     TK  26053.95 us | cuBLASLt  21015.52 us | cuBLAS  21099.50 us | TK/cuBLASLt 1.24x
  dW+accum TK  26231.28 us | cuBLASLt  20992.45 us | cuBLAS  21288.44 us | TK/cuBLASLt 1.25x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 786.661 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2727.509 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 135.586 us
LayerNorm FusedResidualForward (N=65536, C=768): 275.391 us
LayerNorm Backward (N=65536, C=768): 289.007 us
LayerNorm Forward (N=65536, C=3072): 543.676 us
LayerNorm FusedResidualForward (N=65536, C=3072): 1084.325 us
LayerNorm Backward (N=65536, C=3072): 1269.774 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    92.045 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   528.995 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   536.678 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   782.220 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    26.515 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   186.282 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   245.453 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  4002.809 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  8864.857 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  3954.963 us
cuda_memset                    | logits_elems=3296722944      | CUDA kernel  |  4118.752 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  8754.842 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA kernel  |  9216.826 us
global_norm_squared            | params=124475904             | CUDA         |   184.765 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1826.054 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    79.629 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    60.288 us
cuda_memset                    | hidden_elems=50331648        | CUDA kernel  |    62.630 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   131.585 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA kernel  |   136.015 us
```

## Torch Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521/bench_sm120_torch_matmul.log
```

## CuTeDSL Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521/bench_sm120_cutedsl_matmul.log
```

## Triton Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521/bench_sm120_triton_matmul.log
```

## Torch Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521/bench_sm120_torch_attention.log
```

## cuDNN Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521/bench_sm120_cudnn_attention.log
```

## Triton Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521/bench_sm120_triton_attention.log
```

## Torch Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521/bench_sm120_torch_classifier.log
```

## Triton Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521/bench_sm120_triton_classifier.log
```

## Python Stack LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521/bench_sm120_layernorm_python_stacks.log
```

## Triton Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521/bench_sm120_triton_runtime.log
```

## Torch Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521/bench_sm120_torch_runtime.log
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/3 | loss 11.032358 (+nanz)| norm 22.1414 (+nanz)| lr 8.57e-07 | 2490.15 ms | 40.4% bf16 MFU | 210545 tok/s
step    2/3 | loss 10.958515 (+nanz)| norm 22.0968 (+nanz)| lr 1.71e-06 | 2490.99 ms | 40.4% bf16 MFU | 210474 tok/s
step    3/3 | loss 10.811325 (+nanz)| norm 21.1250 (+nanz)| lr 2.57e-06 | 2493.05 ms | 40.3% bf16 MFU | 210385 tok/s
val loss 10.609930
total average iteration time: 2492.018461 ms
```

## validate_sm120_round
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521 --write-scoreboard scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521/scoreboard-candidates.md --write-selected-backends scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521/selected-backends.json --write-promotion-candidates scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521/promotion-candidates.json --require-manifest --require-stack-probe --require-correctness --require-benchmarks --require-training --forbid-checkpoints`

