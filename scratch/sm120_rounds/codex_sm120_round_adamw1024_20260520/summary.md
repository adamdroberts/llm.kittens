# SM120 Optimization Round

- run label: `codex_sm120_round_adamw1024_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_adamw1024_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_adamw1024_20260520`
- max steps: `3`
- git commit: `0f21747`
- working tree: `438` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Wed May 20 17:46:41 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0  On |                  N/A |
|100%   30C    P5             64W /  575W |    3423MiB /  32607MiB |      1%      Default |
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
Command: `python3 dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_round_adamw1024_20260520/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_round_adamw1024_20260520/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `python3 dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_round_adamw1024_20260520/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_round_adamw1024_20260520/round-manifest.md --run-label codex_sm120_round_adamw1024_20260520 --artifact-dir scratch/sm120_rounds/codex_sm120_round_adamw1024_20260520 --train-out-dir log124M/5090_S_codex_sm120_round_adamw1024_20260520 --max-steps 3 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 1 --run-benchmarks 1 --run-training 1 --keep-checkpoints 0`


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
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_round_adamw1024_20260520 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 3`


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
bf16 param vs master max abs diff = 4.737e-04 (tol 5.0e-03) PASS
no-master m max abs diff = 1.863e-09 (tol 1.0e-05) PASS
no-master v max abs diff = 7.276e-11 (tol 1.0e-05) PASS
no-master bf16 param vs ref max abs diff = 7.071e-04 (tol 5.0e-03) PASS
test_adamw smoke OK
[test_global_norm]
cpu norm = 109.432192  gpu norm = 109.432190  relative diff = 0.000000 (tol 0.010) PASS
test_global_norm smoke OK
```

## Matmul Benchmarks

```text
  fwd      TK   1068.25 us | cuBLASLt   1084.87 us | cuBLAS   1570.32 us | TK/cuBLASLt 0.98x
  dInp   TK   1120.82 us | cuBLASLt   1156.73 us | cuBLAS   1190.99 us | TK/cuBLASLt 0.97x
  dW     TK   1653.70 us | cuBLASLt   1206.34 us | cuBLAS   1112.05 us | TK/cuBLASLt 1.37x
  dW+accum TK   1657.12 us | cuBLASLt   1245.32 us | cuBLAS   1160.80 us | TK/cuBLASLt 1.33x
  fwd      TK    436.13 us | cuBLASLt    392.10 us | cuBLAS    567.72 us | TK/cuBLASLt 1.11x
  dInp   TK    425.77 us | cuBLASLt    409.71 us | cuBLAS    439.93 us | TK/cuBLASLt 1.04x
  dW     TK    578.03 us | cuBLASLt    394.77 us | cuBLAS    350.59 us | TK/cuBLASLt 1.46x
  dW+accum TK    647.46 us | cuBLASLt    429.32 us | cuBLAS    384.95 us | TK/cuBLASLt 1.51x
  fwd+GeLU TK fused   1555.82 us | TK explicit   1998.00 us | cuBLASLt   1545.69 us | cuBLAS explicit   2573.86 us | explicit/cuBLASLt 1.29x
  dInp   TK   1499.61 us | cuBLASLt   1413.40 us | cuBLAS   1328.56 us | TK/cuBLASLt 1.06x
  dW     TK   1758.19 us | cuBLASLt   1575.71 us | cuBLAS   1335.24 us | TK/cuBLASLt 1.12x
  dW+accum TK   1839.82 us | cuBLASLt   1518.57 us | cuBLAS   1341.87 us | TK/cuBLASLt 1.21x
  fwd      TK   1417.26 us | cuBLASLt   1423.59 us | cuBLAS   1574.98 us | TK/cuBLASLt 1.00x
  dInp   TK   1470.15 us | cuBLASLt   1477.39 us | cuBLAS   1392.76 us | TK/cuBLASLt 1.00x
  dInp+dGeLU TK   1842.16 us | cuBLASLt fused   1846.35 us | cuBLASLt explicit   2181.84 us | cuBLAS explicit   2295.67 us | explicit/fused 1.18x
  dW     TK   1776.64 us | cuBLASLt   1466.40 us | cuBLAS   1409.31 us | TK/cuBLASLt 1.21x
  dW+accum TK   1759.93 us | cuBLASLt   1465.89 us | cuBLAS   1386.02 us | TK/cuBLASLt 1.20x
  fwd      TK  31901.99 us | cuBLASLt  24378.71 us | cuBLAS  25805.88 us | TK/cuBLASLt 1.31x
  dInp   TK  25532.75 us | cuBLASLt  23136.02 us | cuBLAS  22316.63 us | TK/cuBLASLt 1.10x
  dW     TK  26138.22 us | cuBLASLt  21003.94 us | cuBLAS  21131.97 us | TK/cuBLASLt 1.24x
  dW+accum TK  26008.62 us | cuBLASLt  21063.36 us | cuBLAS  21127.20 us | TK/cuBLASLt 1.23x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 864.433 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2779.795 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 145.115 us
LayerNorm FusedResidualForward (N=65536, C=768): 277.609 us
LayerNorm Backward (N=65536, C=768): 318.345 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |   118.169 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   639.775 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   559.858 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   831.945 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   261.781 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  9200.512 us
global_norm_squared            | params=124475904             | CUDA         |   185.510 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1887.088 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    87.558 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    65.023 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   134.227 us
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/3 | loss 11.032358 (+nanz)| norm 22.1396 (+nanz)| lr 8.57e-07 | 2570.05 ms | 39.1% bf16 MFU | 203999 tok/s
step    2/3 | loss 10.958622 (+nanz)| norm 22.0973 (+nanz)| lr 1.71e-06 | 2589.59 ms | 38.8% bf16 MFU | 202460 tok/s
step    3/3 | loss 10.811241 (+nanz)| norm 21.1267 (+nanz)| lr 2.57e-06 | 2545.46 ms | 39.5% bf16 MFU | 204260 tok/s
val loss 10.609804
total average iteration time: 2567.526102 ms
```

## validate_sm120_round
Command: `python3 dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_round_adamw1024_20260520 --write-scoreboard scratch/sm120_rounds/codex_sm120_round_adamw1024_20260520/scoreboard-candidates.md --require-manifest --require-stack-probe --require-correctness --require-benchmarks --require-training --forbid-checkpoints`

