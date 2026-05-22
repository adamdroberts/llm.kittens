# SM120 Optimization Round

- run label: `codex_sm120_round_current_default_x10_after_memory_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_current_default_x10_after_memory_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_current_default_x10_after_memory_20260520`
- max steps: `10`
- git commit: `0f21747`
- working tree: `460` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Wed May 20 20:19:22 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0  On |                  N/A |
|100%   28C    P5             55W /  575W |    3902MiB /  32607MiB |      0%      Default |
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
Command: `python3 dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_round_current_default_x10_after_memory_20260520/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_round_current_default_x10_after_memory_20260520/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `python3 dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_round_current_default_x10_after_memory_20260520/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_round_current_default_x10_after_memory_20260520/round-manifest.md --run-label codex_sm120_round_current_default_x10_after_memory_20260520 --artifact-dir scratch/sm120_rounds/codex_sm120_round_current_default_x10_after_memory_20260520 --train-out-dir log124M/5090_S_codex_sm120_round_current_default_x10_after_memory_20260520 --max-steps 10 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 1 --run-benchmarks 1 --run-training 1 --keep-checkpoints 0`


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
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_round_current_default_x10_after_memory_20260520 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 10`


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
  fwd      TK   1099.12 us | cuBLASLt   1037.31 us | cuBLAS   1461.54 us | TK/cuBLASLt 1.06x
  dInp   TK   1087.16 us | cuBLASLt   1007.09 us | cuBLAS   1007.40 us | TK/cuBLASLt 1.08x
  dW     TK   1454.91 us | cuBLASLt   1102.88 us | cuBLAS    988.49 us | TK/cuBLASLt 1.32x
  dW+accum TK   1463.74 us | cuBLASLt   1125.72 us | cuBLAS   1032.11 us | TK/cuBLASLt 1.30x
  fwd      TK    374.98 us | cuBLASLt    368.92 us | cuBLAS    503.21 us | TK/cuBLASLt 1.02x
  dInp   TK    379.48 us | cuBLASLt    365.62 us | cuBLAS    364.04 us | TK/cuBLASLt 1.04x
  dW     TK    540.93 us | cuBLASLt    374.90 us | cuBLAS    326.94 us | TK/cuBLASLt 1.44x
  dW+accum TK    546.88 us | cuBLASLt    376.76 us | cuBLAS    329.92 us | TK/cuBLASLt 1.45x
  fwd+GeLU TK fused   1578.70 us | TK explicit   2001.38 us | cuBLASLt   1489.73 us | cuBLAS explicit   2471.91 us | explicit/cuBLASLt 1.34x
  dInp   TK   1450.79 us | cuBLASLt   1355.49 us | cuBLAS   1327.69 us | TK/cuBLASLt 1.07x
  dW     TK   1750.61 us | cuBLASLt   1520.67 us | cuBLAS   1309.12 us | TK/cuBLASLt 1.15x
  dW+accum TK   1749.59 us | cuBLASLt   1507.31 us | cuBLAS   1317.57 us | TK/cuBLASLt 1.16x
  fwd      TK   1419.64 us | cuBLASLt   1390.71 us | cuBLAS   1548.33 us | TK/cuBLASLt 1.02x
  dInp   TK   1476.55 us | cuBLASLt   1396.66 us | cuBLAS   1431.50 us | TK/cuBLASLt 1.06x
  dInp+dGeLU TK   1833.46 us | cuBLASLt fused   1827.45 us | cuBLASLt explicit   2194.74 us | cuBLAS explicit   2183.36 us | explicit/fused 1.20x
  dW     TK   1758.20 us | cuBLASLt   1470.14 us | cuBLAS   1356.72 us | TK/cuBLASLt 1.20x
  dW+accum TK   1719.92 us | cuBLASLt   1495.99 us | cuBLAS   1316.88 us | TK/cuBLASLt 1.15x
  fwd      TK  27733.37 us | cuBLASLt  22336.23 us | cuBLAS  22169.92 us | TK/cuBLASLt 1.24x
  dInp   TK  23877.20 us | cuBLASLt  21871.53 us | cuBLAS  21250.51 us | TK/cuBLASLt 1.09x
  dW     TK  26115.62 us | cuBLASLt  20894.13 us | cuBLAS  21274.31 us | TK/cuBLASLt 1.25x
  dW+accum TK  26019.31 us | cuBLASLt  21000.52 us | cuBLAS  21212.98 us | TK/cuBLASLt 1.24x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 783.906 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2732.582 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 138.710 us
LayerNorm FusedResidualForward (N=65536, C=768): 279.820 us
LayerNorm Backward (N=65536, C=768): 286.407 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    82.143 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   554.136 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   535.077 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   799.103 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    24.464 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   187.960 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   247.230 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  3972.230 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  8985.197 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  4194.509 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  8867.174 us
global_norm_squared            | params=124475904             | CUDA         |   185.869 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1856.336 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    81.052 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    63.199 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   133.575 us
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/10 | loss 11.032358 (+nanz)| norm 22.1414 (+nanz)| lr 8.57e-07 | 2494.27 ms | 40.3% bf16 MFU | 210197 tok/s
step    2/10 | loss 10.958507 (+nanz)| norm 22.0968 (+nanz)| lr 1.71e-06 | 2486.98 ms | 40.4% bf16 MFU | 210813 tok/s
step    3/10 | loss 10.811316 (+nanz)| norm 21.1251 (+nanz)| lr 2.57e-06 | 2489.58 ms | 40.4% bf16 MFU | 210700 tok/s
step    4/10 | loss 10.610130 (+nanz)| norm 18.7014 (+nanz)| lr 3.43e-06 | 2491.65 ms | 40.4% bf16 MFU | 210601 tok/s
step    5/10 | loss 10.392586 (+nanz)| norm 15.0184 (+nanz)| lr 4.29e-06 | 2498.79 ms | 40.2% bf16 MFU | 210390 tok/s
step    6/10 | loss 10.186255 (+nanz)| norm 12.0843 (+nanz)| lr 5.14e-06 | 2495.76 ms | 40.3% bf16 MFU | 210319 tok/s
step    7/10 | loss 10.010621 (+nanz)| norm 10.2002 (+nanz)| lr 6.00e-06 | 2495.83 ms | 40.3% bf16 MFU | 210272 tok/s
step    8/10 | loss 9.855870 (+nanz)| norm 8.7905 (+nanz)| lr 6.86e-06 | 2500.51 ms | 40.2% bf16 MFU | 210172 tok/s
step    9/10 | loss 9.719423 (+nanz)| norm 7.4665 (+nanz)| lr 7.71e-06 | 2498.70 ms | 40.2% bf16 MFU | 210120 tok/s
step   10/10 | loss 9.588612 (+nanz)| norm 6.3099 (+nanz)| lr 8.57e-06 | 2501.19 ms | 40.2% bf16 MFU | 210052 tok/s
val loss 9.483727
total average iteration time: 2495.443185 ms
```

## validate_sm120_round
Command: `python3 dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_round_current_default_x10_after_memory_20260520 --write-scoreboard scratch/sm120_rounds/codex_sm120_round_current_default_x10_after_memory_20260520/scoreboard-candidates.md --require-manifest --require-stack-probe --require-correctness --require-benchmarks --require-training --forbid-checkpoints`

