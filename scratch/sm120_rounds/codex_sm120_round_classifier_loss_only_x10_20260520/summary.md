# SM120 Optimization Round

- run label: `codex_sm120_round_classifier_loss_only_x10_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_classifier_loss_only_x10_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_classifier_loss_only_x10_20260520`
- max steps: `10`
- git commit: `0f21747`
- working tree: `455` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Wed May 20 19:04:46 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0  On |                  N/A |
|100%   34C    P5             56W /  575W |    4037MiB /  32607MiB |      1%      Default |
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
Command: `python3 dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_round_classifier_loss_only_x10_20260520/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_round_classifier_loss_only_x10_20260520/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `python3 dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_round_classifier_loss_only_x10_20260520/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_round_classifier_loss_only_x10_20260520/round-manifest.md --run-label codex_sm120_round_classifier_loss_only_x10_20260520 --artifact-dir scratch/sm120_rounds/codex_sm120_round_classifier_loss_only_x10_20260520 --train-out-dir log124M/5090_S_codex_sm120_round_classifier_loss_only_x10_20260520 --max-steps 10 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 1 --run-benchmarks 1 --run-training 1 --keep-checkpoints 0`


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
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_round_classifier_loss_only_x10_20260520 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 10`


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
  max abs diff = 0.1250  (tolerance 0.50)  PASS
  max abs diff = 0.0312  (tolerance 0.50)  PASS
  max abs diff = 0.1250  (tolerance 0.50)  PASS
──── 11/11 passed ────
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
test_global_norm smoke OK
```

## Matmul Benchmarks

```text
  fwd      TK   1071.53 us | cuBLASLt   1064.70 us | cuBLAS   1467.12 us | TK/cuBLASLt 1.01x
  dInp   TK   1086.84 us | cuBLASLt   1014.75 us | cuBLAS   1012.34 us | TK/cuBLASLt 1.07x
  dW     TK   1456.78 us | cuBLASLt   1111.67 us | cuBLAS   1035.24 us | TK/cuBLASLt 1.31x
  dW+accum TK   1499.70 us | cuBLASLt   1110.98 us | cuBLAS   1005.77 us | TK/cuBLASLt 1.35x
  fwd      TK    376.36 us | cuBLASLt    369.19 us | cuBLAS    512.66 us | TK/cuBLASLt 1.02x
  dInp   TK    381.05 us | cuBLASLt    366.75 us | cuBLAS    366.00 us | TK/cuBLASLt 1.04x
  dW     TK    544.06 us | cuBLASLt    374.32 us | cuBLAS    330.76 us | TK/cuBLASLt 1.45x
  dW+accum TK    558.02 us | cuBLASLt    379.22 us | cuBLAS    331.96 us | TK/cuBLASLt 1.47x
  fwd+GeLU TK fused   1585.13 us | TK explicit   1977.22 us | cuBLASLt   1495.77 us | cuBLAS explicit   2528.40 us | explicit/cuBLASLt 1.32x
  dInp   TK   1449.35 us | cuBLASLt   1343.87 us | cuBLAS   1328.51 us | TK/cuBLASLt 1.08x
  dW     TK   1741.96 us | cuBLASLt   1486.30 us | cuBLAS   1308.79 us | TK/cuBLASLt 1.17x
  dW+accum TK   1748.07 us | cuBLASLt   1524.78 us | cuBLAS   1333.22 us | TK/cuBLASLt 1.15x
  fwd      TK   1418.45 us | cuBLASLt   1351.82 us | cuBLAS   1622.61 us | TK/cuBLASLt 1.05x
  dInp   TK   1475.45 us | cuBLASLt   1377.57 us | cuBLAS   1371.88 us | TK/cuBLASLt 1.07x
  dInp+dGeLU TK   1777.27 us | cuBLASLt fused   1849.39 us | cuBLASLt explicit   2190.27 us | cuBLAS explicit   2132.78 us | explicit/fused 1.18x
  dW     TK   1709.62 us | cuBLASLt   1463.83 us | cuBLAS   1309.34 us | TK/cuBLASLt 1.17x
  dW+accum TK   1749.19 us | cuBLASLt   1472.60 us | cuBLAS   1331.49 us | TK/cuBLASLt 1.19x
  fwd      TK  27767.84 us | cuBLASLt  22331.27 us | cuBLAS  22414.45 us | TK/cuBLASLt 1.24x
  dInp   TK  23917.10 us | cuBLASLt  21865.23 us | cuBLAS  21259.78 us | TK/cuBLASLt 1.09x
  dW     TK  26129.52 us | cuBLASLt  20939.07 us | cuBLAS  21184.39 us | TK/cuBLASLt 1.25x
  dW+accum TK  26020.97 us | cuBLASLt  20917.19 us | cuBLAS  21211.00 us | TK/cuBLASLt 1.24x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 788.432 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2718.345 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 139.323 us
LayerNorm FusedResidualForward (N=65536, C=768): 279.655 us
LayerNorm Backward (N=65536, C=768): 286.878 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |   102.794 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   602.185 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   535.167 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   795.615 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   247.509 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  3978.829 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  9090.682 us
global_norm_squared            | params=124475904             | CUDA         |   186.341 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1849.824 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    86.755 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    62.598 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   133.446 us
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/10 | loss 11.032358 (+nanz)| norm 22.1414 (+nanz)| lr 8.57e-07 | 2575.63 ms | 39.0% bf16 MFU | 203557 tok/s
step    2/10 | loss 10.958507 (+nanz)| norm 22.0968 (+nanz)| lr 1.71e-06 | 2536.03 ms | 39.6% bf16 MFU | 206736 tok/s
step    3/10 | loss 10.811316 (+nanz)| norm 21.1251 (+nanz)| lr 2.57e-06 | 2655.50 ms | 37.9% bf16 MFU | 201966 tok/s
step    4/10 | loss 10.610130 (+nanz)| norm 18.7014 (+nanz)| lr 3.43e-06 | 2494.63 ms | 40.3% bf16 MFU | 204841 tok/s
step    5/10 | loss 10.392586 (+nanz)| norm 15.0184 (+nanz)| lr 4.29e-06 | 2549.32 ms | 39.4% bf16 MFU | 205061 tok/s
step    6/10 | loss 10.186255 (+nanz)| norm 12.0843 (+nanz)| lr 5.14e-06 | 2522.63 ms | 39.9% bf16 MFU | 205674 tok/s
step    7/10 | loss 10.010621 (+nanz)| norm 10.2002 (+nanz)| lr 6.00e-06 | 2536.06 ms | 39.6% bf16 MFU | 205874 tok/s
step    8/10 | loss 9.855870 (+nanz)| norm 8.7905 (+nanz)| lr 6.86e-06 | 2524.39 ms | 39.8% bf16 MFU | 206175 tok/s
step    9/10 | loss 9.719423 (+nanz)| norm 7.4665 (+nanz)| lr 7.71e-06 | 2576.35 ms | 39.0% bf16 MFU | 205777 tok/s
step   10/10 | loss 9.588612 (+nanz)| norm 6.3099 (+nanz)| lr 8.57e-06 | 2566.64 ms | 39.2% bf16 MFU | 205574 tok/s
val loss 9.483727
total average iteration time: 2551.284048 ms
```

## validate_sm120_round
Command: `python3 dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_round_classifier_loss_only_x10_20260520 --write-scoreboard scratch/sm120_rounds/codex_sm120_round_classifier_loss_only_x10_20260520/scoreboard-candidates.md --require-manifest --require-stack-probe --require-correctness --require-benchmarks --require-training --forbid-checkpoints`

