# SM120 Optimization Round

- run label: `codex_sm120_round_cublas_bwd_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_cublas_bwd_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_cublas_bwd_20260520`
- max steps: `3`
- git commit: `0f21747`
- working tree: `447` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Wed May 20 18:02:40 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0  On |                  N/A |
|100%   30C    P5             56W /  575W |    3608MiB /  32607MiB |      1%      Default |
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
Command: `python3 dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_round_cublas_bwd_20260520/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_round_cublas_bwd_20260520/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `python3 dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_round_cublas_bwd_20260520/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_round_cublas_bwd_20260520/round-manifest.md --run-label codex_sm120_round_cublas_bwd_20260520 --artifact-dir scratch/sm120_rounds/codex_sm120_round_cublas_bwd_20260520 --train-out-dir log124M/5090_S_codex_sm120_round_cublas_bwd_20260520 --max-steps 3 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 1 --run-benchmarks 1 --run-training 1 --keep-checkpoints 0`


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
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_round_cublas_bwd_20260520 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 3`


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
  fwd      TK   1115.74 us | cuBLASLt   1036.31 us | cuBLAS   1512.60 us | TK/cuBLASLt 1.08x
  dInp   TK   1086.10 us | cuBLASLt   1028.09 us | cuBLAS   1007.03 us | TK/cuBLASLt 1.06x
  dW     TK   1492.11 us | cuBLASLt   1155.53 us | cuBLAS   1030.37 us | TK/cuBLASLt 1.29x
  dW+accum TK   1496.86 us | cuBLASLt   1104.88 us | cuBLAS   1086.89 us | TK/cuBLASLt 1.35x
  fwd      TK    374.68 us | cuBLASLt    369.16 us | cuBLAS    510.64 us | TK/cuBLASLt 1.01x
  dInp   TK    414.12 us | cuBLASLt    367.78 us | cuBLAS    363.48 us | TK/cuBLASLt 1.13x
  dW     TK    540.29 us | cuBLASLt    371.70 us | cuBLAS    330.04 us | TK/cuBLASLt 1.45x
  dW+accum TK    545.04 us | cuBLASLt    379.11 us | cuBLAS    332.29 us | TK/cuBLASLt 1.44x
  fwd+GeLU TK fused   1529.67 us | TK explicit   1949.17 us | cuBLASLt   1509.85 us | cuBLAS explicit   2570.41 us | explicit/cuBLASLt 1.29x
  dInp   TK   1440.95 us | cuBLASLt   1335.38 us | cuBLAS   1367.99 us | TK/cuBLASLt 1.08x
  dW     TK   1706.33 us | cuBLASLt   1590.70 us | cuBLAS   1308.58 us | TK/cuBLASLt 1.07x
  dW+accum TK   1709.20 us | cuBLASLt   1519.49 us | cuBLAS   1310.40 us | TK/cuBLASLt 1.12x
  fwd      TK   1419.72 us | cuBLASLt   1387.41 us | cuBLAS   1573.52 us | TK/cuBLASLt 1.02x
  dInp   TK   1476.28 us | cuBLASLt   1419.05 us | cuBLAS   1427.21 us | TK/cuBLASLt 1.04x
  dInp+dGeLU TK   1768.03 us | cuBLASLt fused   1826.38 us | cuBLASLt explicit   2250.00 us | cuBLAS explicit   2189.75 us | explicit/fused 1.23x
  dW     TK   1721.29 us | cuBLASLt   1466.51 us | cuBLAS   1350.98 us | TK/cuBLASLt 1.17x
  dW+accum TK   1714.57 us | cuBLASLt   1506.98 us | cuBLAS   1311.84 us | TK/cuBLASLt 1.14x
  fwd      TK  27673.19 us | cuBLASLt  22356.29 us | cuBLAS  22294.53 us | TK/cuBLASLt 1.24x
  dInp   TK  23896.67 us | cuBLASLt  21600.32 us | cuBLAS  21131.06 us | TK/cuBLASLt 1.11x
  dW     TK  25873.27 us | cuBLASLt  20834.63 us | cuBLAS  20968.83 us | TK/cuBLASLt 1.24x
  dW+accum TK  25738.51 us | cuBLASLt  20719.18 us | cuBLAS  21019.41 us | TK/cuBLASLt 1.24x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 791.075 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2720.411 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 136.462 us
LayerNorm FusedResidualForward (N=65536, C=768): 279.340 us
LayerNorm Backward (N=65536, C=768): 289.857 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |   109.300 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   582.765 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   553.711 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   814.171 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   247.560 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  9008.295 us
global_norm_squared            | params=124475904             | CUDA         |   185.125 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1809.910 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    81.990 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    63.631 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   134.268 us
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/3 | loss 11.032358 (+nanz)| norm 22.1414 (+nanz)| lr 8.57e-07 | 2498.67 ms | 40.2% bf16 MFU | 209827 tok/s
step    2/3 | loss 10.958507 (+nanz)| norm 22.0968 (+nanz)| lr 1.71e-06 | 2491.97 ms | 40.3% bf16 MFU | 210391 tok/s
step    3/3 | loss 10.811316 (+nanz)| norm 21.1251 (+nanz)| lr 2.57e-06 | 2495.51 ms | 40.3% bf16 MFU | 210238 tok/s
val loss 10.609911
total average iteration time: 2493.738174 ms
```

## validate_sm120_round
Command: `python3 dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_round_cublas_bwd_20260520 --write-scoreboard scratch/sm120_rounds/codex_sm120_round_cublas_bwd_20260520/scoreboard-candidates.md --require-manifest --require-stack-probe --require-correctness --require-benchmarks --require-training --forbid-checkpoints`

