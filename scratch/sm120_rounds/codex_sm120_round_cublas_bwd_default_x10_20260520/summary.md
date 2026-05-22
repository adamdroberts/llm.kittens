# SM120 Optimization Round

- run label: `codex_sm120_round_cublas_bwd_default_x10_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_cublas_bwd_default_x10_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_cublas_bwd_default_x10_20260520`
- max steps: `10`
- git commit: `0f21747`
- working tree: `450` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Wed May 20 18:09:28 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0  On |                  N/A |
|100%   29C    P8             56W /  575W |    3604MiB /  32607MiB |      1%      Default |
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
Command: `python3 dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_round_cublas_bwd_default_x10_20260520/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_round_cublas_bwd_default_x10_20260520/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `python3 dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_round_cublas_bwd_default_x10_20260520/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_round_cublas_bwd_default_x10_20260520/round-manifest.md --run-label codex_sm120_round_cublas_bwd_default_x10_20260520 --artifact-dir scratch/sm120_rounds/codex_sm120_round_cublas_bwd_default_x10_20260520 --train-out-dir log124M/5090_S_codex_sm120_round_cublas_bwd_default_x10_20260520 --max-steps 10 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 1 --run-benchmarks 1 --run-training 1 --keep-checkpoints 0`


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
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_round_cublas_bwd_default_x10_20260520 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 10`


## Correctness Markers

```text
[test_matmul]
  max abs diff = 0.1250  (tolerance 0.50)  PASS
  max abs diff = 0.2500  (tolerance 0.50)  PASS
  max abs diff = 0.2500  (tolerance 0.50)  PASS
  pre-GELU max abs diff = 0.2500  GELU max abs diff = 0.2500  (tolerance 0.50)  PASS
  max abs diff = 0.1250  (tolerance 0.50)  PASS
  max abs diff = 0.0625  (tolerance 0.50)  PASS
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
bias add max abs diff = 0.003906 (tol 0.010) PASS
bias grad max abs diff = 0.072189 (tol 0.25) PASS
test_bias smoke OK
[test_gelu]
forward  max abs diff = 0.007755 (tol 0.020) PASS
backward max abs diff = 0.003887 (tol 0.020) PASS
test_gelu smoke OK
[test_fused_classifier]
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
test_global_norm smoke OK
```

## Matmul Benchmarks

```text
  fwd      TK   1068.45 us | cuBLASLt   1033.76 us | cuBLAS   1514.96 us | TK/cuBLASLt 1.03x
  dInp   TK   1120.78 us | cuBLASLt   1053.94 us | cuBLAS   1007.26 us | TK/cuBLASLt 1.06x
  dW     TK   1490.13 us | cuBLASLt   1103.70 us | cuBLAS   1052.23 us | TK/cuBLASLt 1.35x
  dW+accum TK   1458.66 us | cuBLASLt   1148.49 us | cuBLAS    990.38 us | TK/cuBLASLt 1.27x
  fwd      TK    374.21 us | cuBLASLt    413.55 us | cuBLAS    510.67 us | TK/cuBLASLt 0.90x
  dInp   TK    379.25 us | cuBLASLt    363.68 us | cuBLAS    363.52 us | TK/cuBLASLt 1.04x
  dW     TK    572.66 us | cuBLASLt    371.63 us | cuBLAS    326.55 us | TK/cuBLASLt 1.54x
  dW+accum TK    580.42 us | cuBLASLt    377.24 us | cuBLAS    331.92 us | TK/cuBLASLt 1.54x
  fwd+GeLU TK fused   1575.25 us | TK explicit   1994.45 us | cuBLASLt   1476.95 us | cuBLAS explicit   2512.62 us | explicit/cuBLASLt 1.35x
  dInp   TK   1489.15 us | cuBLASLt   1335.22 us | cuBLAS   1424.85 us | TK/cuBLASLt 1.12x
  dW     TK   1747.76 us | cuBLASLt   1471.06 us | cuBLAS   1350.03 us | TK/cuBLASLt 1.19x
  dW+accum TK   1710.24 us | cuBLASLt   1473.80 us | cuBLAS   1330.11 us | TK/cuBLASLt 1.16x
  fwd      TK   1437.05 us | cuBLASLt   1350.67 us | cuBLAS   1647.67 us | TK/cuBLASLt 1.06x
  dInp   TK   1475.60 us | cuBLASLt   1373.16 us | cuBLAS   1413.99 us | TK/cuBLASLt 1.07x
  dInp+dGeLU TK   1768.65 us | cuBLASLt fused   1828.48 us | cuBLASLt explicit   2183.50 us | cuBLAS explicit   2192.54 us | explicit/fused 1.19x
  dW     TK   1753.80 us | cuBLASLt   1467.01 us | cuBLAS   1306.03 us | TK/cuBLASLt 1.20x
  dW+accum TK   1749.00 us | cuBLASLt   1468.76 us | cuBLAS   1355.10 us | TK/cuBLASLt 1.19x
  fwd      TK  27692.13 us | cuBLASLt  22101.72 us | cuBLAS  22095.69 us | TK/cuBLASLt 1.25x
  dInp   TK  23729.57 us | cuBLASLt  21557.56 us | cuBLAS  21489.09 us | TK/cuBLASLt 1.10x
  dW     TK  26065.79 us | cuBLASLt  20715.34 us | cuBLAS  21191.52 us | TK/cuBLASLt 1.26x
  dW+accum TK  26218.39 us | cuBLASLt  20710.12 us | cuBLAS  21359.52 us | TK/cuBLASLt 1.27x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 787.306 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2722.195 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 139.524 us
LayerNorm FusedResidualForward (N=65536, C=768): 280.599 us
LayerNorm Backward (N=65536, C=768): 287.881 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    98.617 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   605.345 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   535.735 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   778.570 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   274.883 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  9013.396 us
global_norm_squared            | params=124475904             | CUDA         |   185.950 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1831.226 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    87.636 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    62.876 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   133.731 us
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/10 | loss 11.032358 (+nanz)| norm 22.1414 (+nanz)| lr 8.57e-07 | 2495.40 ms | 40.3% bf16 MFU | 210102 tok/s
step    2/10 | loss 10.958507 (+nanz)| norm 22.0968 (+nanz)| lr 1.71e-06 | 2488.70 ms | 40.4% bf16 MFU | 210668 tok/s
step    3/10 | loss 10.811316 (+nanz)| norm 21.1251 (+nanz)| lr 2.57e-06 | 2491.16 ms | 40.4% bf16 MFU | 210561 tok/s
step    4/10 | loss 10.610130 (+nanz)| norm 18.7014 (+nanz)| lr 3.43e-06 | 2493.62 ms | 40.3% bf16 MFU | 210453 tok/s
step    5/10 | loss 10.392586 (+nanz)| norm 15.0184 (+nanz)| lr 4.29e-06 | 2495.75 ms | 40.3% bf16 MFU | 210350 tok/s
step    6/10 | loss 10.186255 (+nanz)| norm 12.0843 (+nanz)| lr 5.14e-06 | 2495.22 ms | 40.3% bf16 MFU | 210299 tok/s
step    7/10 | loss 10.010621 (+nanz)| norm 10.2002 (+nanz)| lr 6.00e-06 | 2503.11 ms | 40.2% bf16 MFU | 210139 tok/s
step    8/10 | loss 9.855870 (+nanz)| norm 8.7905 (+nanz)| lr 6.86e-06 | 2499.24 ms | 40.2% bf16 MFU | 210080 tok/s
step    9/10 | loss 9.719423 (+nanz)| norm 7.4665 (+nanz)| lr 7.71e-06 | 2501.50 ms | 40.2% bf16 MFU | 210007 tok/s
step   10/10 | loss 9.588612 (+nanz)| norm 6.3099 (+nanz)| lr 8.57e-06 | 2506.93 ms | 40.1% bf16 MFU | 209889 tok/s
val loss 9.483727
total average iteration time: 2497.246398 ms
```

## validate_sm120_round
Command: `python3 dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_round_cublas_bwd_default_x10_20260520 --write-scoreboard scratch/sm120_rounds/codex_sm120_round_cublas_bwd_default_x10_20260520/scoreboard-candidates.md --require-manifest --require-stack-probe --require-correctness --require-benchmarks --require-training --forbid-checkpoints`

