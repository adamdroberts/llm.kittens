# SM120 Optimization Round

- run label: `codex_sm120_round_runtime_shape_validator_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_runtime_shape_validator_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_runtime_shape_validator_20260520`
- max steps: `3`
- git commit: `0f21747`
- working tree: `457` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Wed May 20 19:38:16 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0  On |                  N/A |
|100%   28C    P5             57W /  575W |    3980MiB /  32607MiB |      1%      Default |
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
Command: `python3 dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_round_runtime_shape_validator_20260520/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_round_runtime_shape_validator_20260520/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `python3 dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_round_runtime_shape_validator_20260520/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_round_runtime_shape_validator_20260520/round-manifest.md --run-label codex_sm120_round_runtime_shape_validator_20260520 --artifact-dir scratch/sm120_rounds/codex_sm120_round_runtime_shape_validator_20260520 --train-out-dir log124M/5090_S_codex_sm120_round_runtime_shape_validator_20260520 --max-steps 3 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 1 --run-benchmarks 1 --run-training 1 --keep-checkpoints 0`


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
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_round_runtime_shape_validator_20260520 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 3`


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
  fwd      TK   1069.90 us | cuBLASLt   1094.54 us | cuBLAS   1498.73 us | TK/cuBLASLt 0.98x
  dInp   TK   1174.57 us | cuBLASLt   1064.61 us | cuBLAS   1050.98 us | TK/cuBLASLt 1.10x
  dW     TK   1545.32 us | cuBLASLt   1180.76 us | cuBLAS   1067.67 us | TK/cuBLASLt 1.31x
  dW+accum TK   1518.64 us | cuBLASLt   1210.42 us | cuBLAS   1056.51 us | TK/cuBLASLt 1.25x
  fwd      TK    414.56 us | cuBLASLt    394.66 us | cuBLAS    507.84 us | TK/cuBLASLt 1.05x
  dInp   TK    380.01 us | cuBLASLt    365.02 us | cuBLAS    365.92 us | TK/cuBLASLt 1.04x
  dW     TK    547.99 us | cuBLASLt    375.08 us | cuBLAS    328.95 us | TK/cuBLASLt 1.46x
  dW+accum TK    546.72 us | cuBLASLt    377.70 us | cuBLAS    329.08 us | TK/cuBLASLt 1.45x
  fwd+GeLU TK fused   1581.16 us | TK explicit   2003.30 us | cuBLASLt   1472.59 us | cuBLAS explicit   2479.43 us | explicit/cuBLASLt 1.36x
  dInp   TK   1450.28 us | cuBLASLt   1370.52 us | cuBLAS   1332.77 us | TK/cuBLASLt 1.06x
  dW     TK   1732.35 us | cuBLASLt   1601.00 us | cuBLAS   1410.20 us | TK/cuBLASLt 1.08x
  dW+accum TK   1872.05 us | cuBLASLt   1577.99 us | cuBLAS   1410.85 us | TK/cuBLASLt 1.19x
  fwd      TK   1492.64 us | cuBLASLt   1370.41 us | cuBLAS   1575.06 us | TK/cuBLASLt 1.09x
  dInp   TK   1476.13 us | cuBLASLt   1382.28 us | cuBLAS   1366.46 us | TK/cuBLASLt 1.07x
  dInp+dGeLU TK   1836.63 us | cuBLASLt fused   1857.11 us | cuBLASLt explicit   2192.10 us | cuBLAS explicit   2163.79 us | explicit/fused 1.18x
  dW     TK   1735.08 us | cuBLASLt   1524.92 us | cuBLAS   1355.05 us | TK/cuBLASLt 1.14x
  dW+accum TK   1731.00 us | cuBLASLt   1483.95 us | cuBLAS   1357.61 us | TK/cuBLASLt 1.17x
  fwd      TK  27808.53 us | cuBLASLt  22427.17 us | cuBLAS  22480.34 us | TK/cuBLASLt 1.24x
  dInp   TK  24115.99 us | cuBLASLt  21981.90 us | cuBLAS  21187.21 us | TK/cuBLASLt 1.10x
  dW     TK  26066.11 us | cuBLASLt  22493.09 us | cuBLAS  22520.13 us | TK/cuBLASLt 1.16x
  dW+accum TK  26151.68 us | cuBLASLt  20922.18 us | cuBLAS  21216.49 us | TK/cuBLASLt 1.25x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 790.880 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2733.449 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 139.555 us
LayerNorm FusedResidualForward (N=65536, C=768): 280.100 us
LayerNorm Backward (N=65536, C=768): 289.997 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    97.122 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   538.396 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   554.926 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   790.955 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    29.499 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   188.219 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   247.298 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  4218.240 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  9657.062 us
global_norm_squared            | params=124475904             | CUDA         |   185.675 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1815.523 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    76.676 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    64.312 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   134.024 us
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/3 | loss 11.032358 (+nanz)| norm 22.1414 (+nanz)| lr 8.57e-07 | 2517.44 ms | 39.9% bf16 MFU | 208262 tok/s
step    2/3 | loss 10.958507 (+nanz)| norm 22.0968 (+nanz)| lr 1.71e-06 | 2507.23 ms | 40.1% bf16 MFU | 209110 tok/s
step    3/3 | loss 10.811316 (+nanz)| norm 21.1251 (+nanz)| lr 2.57e-06 | 2583.54 ms | 38.9% bf16 MFU | 205943 tok/s
val loss 10.609911
total average iteration time: 2545.383811 ms
```

## validate_sm120_round
Command: `python3 dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_round_runtime_shape_validator_20260520 --write-scoreboard scratch/sm120_rounds/codex_sm120_round_runtime_shape_validator_20260520/scoreboard-candidates.md --require-manifest --require-stack-probe --require-correctness --require-benchmarks --require-training --forbid-checkpoints`

