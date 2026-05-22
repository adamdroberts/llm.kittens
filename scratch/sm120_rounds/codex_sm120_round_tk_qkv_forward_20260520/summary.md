# SM120 Optimization Round

- run label: `codex_sm120_round_tk_qkv_forward_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_tk_qkv_forward_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_tk_qkv_forward_20260520`
- max steps: `3`
- git commit: `0f21747`
- working tree: `451` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Wed May 20 18:27:30 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0  On |                  N/A |
|100%   31C    P5             57W /  575W |    3823MiB /  32607MiB |      1%      Default |
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
Command: `python3 dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_round_tk_qkv_forward_20260520/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_round_tk_qkv_forward_20260520/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `python3 dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_round_tk_qkv_forward_20260520/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_round_tk_qkv_forward_20260520/round-manifest.md --run-label codex_sm120_round_tk_qkv_forward_20260520 --artifact-dir scratch/sm120_rounds/codex_sm120_round_tk_qkv_forward_20260520 --train-out-dir log124M/5090_S_codex_sm120_round_tk_qkv_forward_20260520 --max-steps 3 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 1 --run-benchmarks 1 --run-training 1 --keep-checkpoints 0`


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
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_round_tk_qkv_forward_20260520 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 3`


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
  fwd      TK   1073.24 us | cuBLASLt   1039.15 us | cuBLAS   1473.20 us | TK/cuBLASLt 1.03x
  dInp   TK   1119.59 us | cuBLASLt   1011.90 us | cuBLAS   1058.35 us | TK/cuBLASLt 1.11x
  dW     TK   1461.06 us | cuBLASLt   1109.20 us | cuBLAS    993.19 us | TK/cuBLASLt 1.32x
  dW+accum TK   1495.46 us | cuBLASLt   1141.00 us | cuBLAS    998.92 us | TK/cuBLASLt 1.31x
  fwd      TK    376.75 us | cuBLASLt    392.04 us | cuBLAS    512.84 us | TK/cuBLASLt 0.96x
  dInp   TK    380.53 us | cuBLASLt    365.66 us | cuBLAS    365.69 us | TK/cuBLASLt 1.04x
  dW     TK    558.17 us | cuBLASLt    374.33 us | cuBLAS    327.22 us | TK/cuBLASLt 1.49x
  dW+accum TK    541.95 us | cuBLASLt    377.13 us | cuBLAS    332.90 us | TK/cuBLASLt 1.44x
  fwd+GeLU TK fused   1559.96 us | TK explicit   1956.01 us | cuBLASLt   1501.90 us | cuBLAS explicit   2520.72 us | explicit/cuBLASLt 1.30x
  dInp   TK   1510.26 us | cuBLASLt   1394.17 us | cuBLAS   1360.22 us | TK/cuBLASLt 1.08x
  dW     TK   1796.25 us | cuBLASLt   1565.62 us | cuBLAS   1341.18 us | TK/cuBLASLt 1.15x
  dW+accum TK   1745.66 us | cuBLASLt   1494.06 us | cuBLAS   1315.31 us | TK/cuBLASLt 1.17x
  fwd      TK   1467.47 us | cuBLASLt   1390.81 us | cuBLAS   1575.83 us | TK/cuBLASLt 1.06x
  dInp   TK   1476.61 us | cuBLASLt   1372.41 us | cuBLAS   1366.92 us | TK/cuBLASLt 1.08x
  dInp+dGeLU TK   1816.49 us | cuBLASLt fused   1859.33 us | cuBLASLt explicit   2194.20 us | cuBLAS explicit   2179.37 us | explicit/fused 1.18x
  dW     TK   1750.94 us | cuBLASLt   1522.50 us | cuBLAS   1314.35 us | TK/cuBLASLt 1.15x
  dW+accum TK   1747.32 us | cuBLASLt   1517.41 us | cuBLAS   1340.92 us | TK/cuBLASLt 1.15x
  fwd      TK  30202.33 us | cuBLASLt  22318.97 us | cuBLAS  22315.28 us | TK/cuBLASLt 1.35x
  dInp   TK  24074.59 us | cuBLASLt  21858.43 us | cuBLAS  21248.01 us | TK/cuBLASLt 1.10x
  dW     TK  25852.70 us | cuBLASLt  20885.57 us | cuBLAS  21190.34 us | TK/cuBLASLt 1.24x
  dW+accum TK  26306.05 us | cuBLASLt  20982.02 us | cuBLAS  21281.18 us | TK/cuBLASLt 1.25x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 799.051 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2723.250 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 136.810 us
LayerNorm FusedResidualForward (N=65536, C=768): 282.996 us
LayerNorm Backward (N=65536, C=768): 289.273 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |   118.666 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   582.757 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   553.344 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   777.686 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   246.395 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  8999.084 us
global_norm_squared            | params=124475904             | CUDA         |   185.989 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1853.549 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    81.912 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    63.375 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   143.124 us
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033160
step    1/3 | loss 11.032354 (+nanz)| norm 22.1413 (+nanz)| lr 8.57e-07 | 2551.58 ms | 39.4% bf16 MFU | 205476 tok/s
step    2/3 | loss 10.958511 (+nanz)| norm 22.0969 (+nanz)| lr 1.71e-06 | 2634.86 ms | 38.2% bf16 MFU | 198981 tok/s
step    3/3 | loss 10.811319 (+nanz)| norm 21.1251 (+nanz)| lr 2.57e-06 | 2562.02 ms | 39.2% bf16 MFU | 201882 tok/s
val loss 10.609915
total average iteration time: 2598.442197 ms
```

## validate_sm120_round
Command: `python3 dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_round_tk_qkv_forward_20260520 --write-scoreboard scratch/sm120_rounds/codex_sm120_round_tk_qkv_forward_20260520/scoreboard-candidates.md --require-manifest --require-stack-probe --require-correctness --require-benchmarks --require-training --forbid-checkpoints`

