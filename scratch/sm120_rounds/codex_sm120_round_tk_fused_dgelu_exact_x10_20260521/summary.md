# SM120 Optimization Round

- run label: `codex_sm120_round_tk_fused_dgelu_exact_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_tk_fused_dgelu_exact_x10_20260521`
- max steps: `10`
- python: `/home/adam/miniconda3/envs/llm-kittens/bin/python`
- git commit: `0f21747`
- working tree: `486` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Thu May 21 03:01:54 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0 Off |                  N/A |
|100%   34C    P8             48W /  575W |    1113MiB /  32607MiB |      1%      Default |
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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521/round-manifest.md --run-label codex_sm120_round_tk_fused_dgelu_exact_x10_20260521 --artifact-dir scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521 --train-out-dir log124M/5090_S_codex_sm120_round_tk_fused_dgelu_exact_x10_20260521 --max-steps 10 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 1 --run-benchmarks 1 --run-python-stack-benchmarks 0 --run-training 1 --keep-checkpoints 0`


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
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_round_tk_fused_dgelu_exact_x10_20260521 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 10`


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
  fwd      TK   1072.56 us | cuBLASLt   1042.71 us | cuBLAS   1406.45 us | TK/cuBLASLt 1.03x
  dInp   TK   1092.41 us | cuBLASLt   1010.84 us | cuBLAS   1024.20 us | TK/cuBLASLt 1.08x
  dW     TK   1453.78 us | cuBLASLt   1107.24 us | cuBLAS   1015.69 us | TK/cuBLASLt 1.31x
  dW+accum TK   1460.41 us | cuBLASLt   1108.59 us | cuBLAS   1000.91 us | TK/cuBLASLt 1.32x
  fwd      TK    376.68 us | cuBLASLt    371.61 us | cuBLAS    484.68 us | TK/cuBLASLt 1.01x
  dInp   TK    380.97 us | cuBLASLt    366.98 us | cuBLAS    365.53 us | TK/cuBLASLt 1.04x
  dW     TK    543.42 us | cuBLASLt    378.23 us | cuBLAS    329.32 us | TK/cuBLASLt 1.44x
  dW+accum TK    541.07 us | cuBLASLt    380.88 us | cuBLAS    331.72 us | TK/cuBLASLt 1.42x
  fwd+GeLU TK fused   1538.67 us | TK explicit   1993.48 us | cuBLASLt   1530.60 us | cuBLAS explicit   2433.96 us | explicit/cuBLASLt 1.30x
  dInp   TK   1495.08 us | cuBLASLt   1361.81 us | cuBLAS   1336.52 us | TK/cuBLASLt 1.10x
  dW     TK   1731.60 us | cuBLASLt   1516.38 us | cuBLAS   1348.97 us | TK/cuBLASLt 1.14x
  dW+accum TK   1746.69 us | cuBLASLt   1487.05 us | cuBLAS   1356.32 us | TK/cuBLASLt 1.17x
  fwd      TK   1443.70 us | cuBLASLt   1348.06 us | cuBLAS   1542.08 us | TK/cuBLASLt 1.07x
  dInp   TK   1475.64 us | cuBLASLt   1407.91 us | cuBLAS   1369.26 us | TK/cuBLASLt 1.05x
  dInp+dGeLU TK   1790.16 us | cuBLASLt fused   1851.91 us | cuBLASLt explicit   2157.34 us | cuBLAS explicit   2177.50 us | explicit/fused 1.16x
  dW     TK   1739.26 us | cuBLASLt   1489.27 us | cuBLAS   1351.92 us | TK/cuBLASLt 1.17x
  dW+accum TK   1710.58 us | cuBLASLt   1468.54 us | cuBLAS   1313.15 us | TK/cuBLASLt 1.16x
  fwd      TK  27642.85 us | cuBLASLt  22428.78 us | cuBLAS  22315.89 us | TK/cuBLASLt 1.23x
  dInp   TK  24045.45 us | cuBLASLt  21891.59 us | cuBLAS  21392.08 us | TK/cuBLASLt 1.10x
  dW     TK  26066.96 us | cuBLASLt  20806.73 us | cuBLAS  21236.62 us | TK/cuBLASLt 1.25x
  dW+accum TK  26123.31 us | cuBLASLt  21001.53 us | cuBLAS  21312.99 us | TK/cuBLASLt 1.24x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 783.589 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2727.581 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 135.884 us
LayerNorm FusedResidualForward (N=65536, C=768): 275.685 us
LayerNorm Backward (N=65536, C=768): 289.011 us
LayerNorm Forward (N=65536, C=3072): 545.395 us
LayerNorm FusedResidualForward (N=65536, C=3072): 1081.923 us
LayerNorm Backward (N=65536, C=3072): 1273.185 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    91.956 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   546.612 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   527.500 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   790.435 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    24.952 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   186.269 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   258.563 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  3897.875 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  8940.014 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  4008.787 us
cuda_memset                    | logits_elems=3296722944      | CUDA kernel  |  4106.649 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  8788.646 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA kernel  |  9121.588 us
global_norm_squared            | params=124475904             | CUDA         |   185.421 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1806.630 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    79.903 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    60.173 us
cuda_memset                    | hidden_elems=50331648        | CUDA kernel  |    62.234 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   131.737 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA kernel  |   137.770 us
```

## Torch Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521/bench_sm120_torch_matmul.log
```

## CuTeDSL Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521/bench_sm120_cutedsl_matmul.log
```

## Triton Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521/bench_sm120_triton_matmul.log
```

## Torch Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521/bench_sm120_torch_attention.log
```

## cuDNN Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521/bench_sm120_cudnn_attention.log
```

## Triton Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521/bench_sm120_triton_attention.log
```

## Torch Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521/bench_sm120_torch_classifier.log
```

## Triton Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521/bench_sm120_triton_classifier.log
```

## Python Stack LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521/bench_sm120_layernorm_python_stacks.log
```

## Triton Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521/bench_sm120_triton_runtime.log
```

## Torch Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521/bench_sm120_torch_runtime.log
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/10 | loss 11.032358 (+nanz)| norm 22.1414 (+nanz)| lr 8.57e-07 | 2494.44 ms | 40.3% bf16 MFU | 210183 tok/s
step    2/10 | loss 10.958509 (+nanz)| norm 22.0970 (+nanz)| lr 1.71e-06 | 2493.35 ms | 40.3% bf16 MFU | 210274 tok/s
step    3/10 | loss 10.811327 (+nanz)| norm 21.1252 (+nanz)| lr 2.57e-06 | 2495.94 ms | 40.3% bf16 MFU | 210163 tok/s
step    4/10 | loss 10.610130 (+nanz)| norm 18.7015 (+nanz)| lr 3.43e-06 | 2499.46 ms | 40.2% bf16 MFU | 210022 tok/s
step    5/10 | loss 10.392588 (+nanz)| norm 15.0185 (+nanz)| lr 4.29e-06 | 2499.74 ms | 40.2% bf16 MFU | 209945 tok/s
step    6/10 | loss 10.186265 (+nanz)| norm 12.0843 (+nanz)| lr 5.14e-06 | 2501.07 ms | 40.2% bf16 MFU | 209874 tok/s
step    7/10 | loss 10.010623 (+nanz)| norm 10.2002 (+nanz)| lr 6.00e-06 | 2504.04 ms | 40.2% bf16 MFU | 209781 tok/s
step    8/10 | loss 9.855854 (+nanz)| norm 8.7904 (+nanz)| lr 6.86e-06 | 2506.92 ms | 40.1% bf16 MFU | 209674 tok/s
step    9/10 | loss 9.719427 (+nanz)| norm 7.4664 (+nanz)| lr 7.71e-06 | 2509.35 ms | 40.1% bf16 MFU | 209564 tok/s
step   10/10 | loss 9.588631 (+nanz)| norm 6.3098 (+nanz)| lr 8.57e-06 | 2511.09 ms | 40.0% bf16 MFU | 209459 tok/s
val loss 9.483725
total average iteration time: 2502.328263 ms
```

## validate_sm120_round
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521 --write-scoreboard scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521/scoreboard-candidates.md --write-selected-backends scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521/selected-backends.json --write-promotion-candidates scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521/promotion-candidates.json --require-manifest --require-stack-probe --require-correctness --require-benchmarks --require-training --forbid-checkpoints`

