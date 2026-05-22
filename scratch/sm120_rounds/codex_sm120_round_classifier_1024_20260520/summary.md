# SM120 Optimization Round

- run label: `codex_sm120_round_classifier_1024_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_classifier_1024_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_classifier_1024_20260520`
- max steps: `3`
- git commit: `0f21747`
- working tree: `434` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Wed May 20 17:33:20 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0  On |                  N/A |
|100%   30C    P5             56W /  575W |    3506MiB /  32607MiB |      1%      Default |
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
Command: `python3 dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_round_classifier_1024_20260520/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_round_classifier_1024_20260520/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `python3 dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_round_classifier_1024_20260520/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_round_classifier_1024_20260520/round-manifest.md --run-label codex_sm120_round_classifier_1024_20260520 --artifact-dir scratch/sm120_rounds/codex_sm120_round_classifier_1024_20260520 --train-out-dir log124M/5090_S_codex_sm120_round_classifier_1024_20260520 --max-steps 3 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 1 --run-benchmarks 1 --run-training 1 --keep-checkpoints 0`


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
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_round_classifier_1024_20260520 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 3`


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
──── 8/8 passed ────
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
test_adamw smoke OK
[test_global_norm]
cpu norm = 109.432192  gpu norm = 109.432190  relative diff = 0.000000 (tol 0.010) PASS
test_global_norm smoke OK
```

## Matmul Benchmarks

```text
  fwd      TK   1069.91 us | cuBLASLt   1033.17 us | cuBLAS   1514.28 us | TK/cuBLASLt 1.04x
  dInp   TK   1087.39 us | cuBLASLt   1054.61 us | cuBLAS   1006.93 us | TK/cuBLASLt 1.03x
  dW     TK   1492.18 us | cuBLASLt   1105.81 us | cuBLAS   1030.36 us | TK/cuBLASLt 1.35x
  dW+accum TK   1491.19 us | cuBLASLt   1168.39 us | cuBLAS    990.73 us | TK/cuBLASLt 1.28x
  fwd      TK    374.40 us | cuBLASLt    388.54 us | cuBLAS    513.88 us | TK/cuBLASLt 0.96x
  dInp   TK    379.74 us | cuBLASLt    363.71 us | cuBLAS    388.79 us | TK/cuBLASLt 1.04x
  dW     TK    584.68 us | cuBLASLt    372.89 us | cuBLAS    327.96 us | TK/cuBLASLt 1.57x
  dW+accum TK    550.64 us | cuBLASLt    373.95 us | cuBLAS    329.96 us | TK/cuBLASLt 1.47x
  fwd+GeLU TK fused   1581.96 us | TK explicit   1972.87 us | cuBLASLt   1498.27 us | cuBLAS explicit   2540.46 us | explicit/cuBLASLt 1.32x
  dInp   TK   1491.50 us | cuBLASLt   1335.98 us | cuBLAS   1423.47 us | TK/cuBLASLt 1.12x
  dW     TK   1759.53 us | cuBLASLt   1471.93 us | cuBLAS   1352.72 us | TK/cuBLASLt 1.20x
  dW+accum TK   1713.90 us | cuBLASLt   1504.61 us | cuBLAS   1336.83 us | TK/cuBLASLt 1.14x
  fwd      TK   1443.44 us | cuBLASLt   1343.19 us | cuBLAS   1621.06 us | TK/cuBLASLt 1.07x
  dInp   TK   1476.13 us | cuBLASLt   1365.04 us | cuBLAS   1442.20 us | TK/cuBLASLt 1.08x
  dInp+dGeLU TK   1786.41 us | cuBLASLt fused   1849.96 us | cuBLASLt explicit   2221.06 us | cuBLAS explicit   2191.10 us | explicit/fused 1.20x
  dW     TK   1790.20 us | cuBLASLt   1467.64 us | cuBLAS   1308.59 us | TK/cuBLASLt 1.22x
  dW+accum TK   1750.20 us | cuBLASLt   1470.36 us | cuBLAS   1356.12 us | TK/cuBLASLt 1.19x
  fwd      TK  27833.97 us | cuBLASLt  22294.62 us | cuBLAS  22204.33 us | TK/cuBLASLt 1.25x
  dInp   TK  23767.28 us | cuBLASLt  21581.93 us | cuBLAS  20952.98 us | TK/cuBLASLt 1.10x
  dW     TK  26090.79 us | cuBLASLt  21046.30 us | cuBLAS  21044.61 us | TK/cuBLASLt 1.24x
  dW+accum TK  25886.75 us | cuBLASLt  20893.26 us | cuBLAS  21164.43 us | TK/cuBLASLt 1.24x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 789.784 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2726.056 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 136.942 us
LayerNorm FusedResidualForward (N=65536, C=768): 281.060 us
LayerNorm Backward (N=65536, C=768): 293.201 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |   102.644 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   592.973 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   536.535 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   779.523 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   275.587 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  9086.464 us
global_norm_squared            | params=124475904             | CUDA         |   185.854 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1835.747 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    87.382 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    63.723 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   133.981 us
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/3 | loss 11.032358 (+nanz)| norm 22.1396 (+nanz)| lr 8.57e-07 | 2532.97 ms | 39.7% bf16 MFU | 206986 tok/s
step    2/3 | loss 10.958514 (+nanz)| norm 22.0950 (+nanz)| lr 1.71e-06 | 2530.08 ms | 39.7% bf16 MFU | 207222 tok/s
step    3/3 | loss 10.811321 (+nanz)| norm 21.1231 (+nanz)| lr 2.57e-06 | 2526.31 ms | 39.8% bf16 MFU | 207381 tok/s
val loss 10.609921
total average iteration time: 2528.192759 ms
```

## validate_sm120_round
Command: `python3 dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_round_classifier_1024_20260520 --write-scoreboard scratch/sm120_rounds/codex_sm120_round_classifier_1024_20260520/scoreboard-candidates.md --require-manifest --require-stack-probe --require-correctness --require-benchmarks --require-training --forbid-checkpoints`

