# SM120 Optimization Round

- run label: `codex_sm120_round_cublas_dinp_attproj_fc_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_cublas_dinp_attproj_fc_20260521`
- max steps: `3`
- python: `/home/adam/miniconda3/envs/llm-kittens/bin/python`
- git commit: `0f21747`
- working tree: `480` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Thu May 21 01:24:21 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0  On |                  N/A |
|100%   31C    P5             56W /  575W |    2467MiB /  32607MiB |      1%      Default |
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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521/round-manifest.md --run-label codex_sm120_round_cublas_dinp_attproj_fc_20260521 --artifact-dir scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521 --train-out-dir log124M/5090_S_codex_sm120_round_cublas_dinp_attproj_fc_20260521 --max-steps 3 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 1 --run-benchmarks 1 --run-python-stack-benchmarks 0 --run-training 1 --keep-checkpoints 0`


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
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_round_cublas_dinp_attproj_fc_20260521 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 3`


## Correctness Markers

```text
[test_matmul]
  max abs diff = 0.1250  (tolerance 0.50)  PASS
  max abs diff = 0.2500  (tolerance 0.50)  PASS
  max abs diff = 0.2500  (tolerance 0.50)  PASS
  pre-GELU max abs diff = 0.2500  GELU max abs diff = 0.2500  (tolerance 0.50)  PASS
  max abs diff = 0.1250  (tolerance 0.50)  PASS
  max abs diff = 0.0625  (tolerance 0.50)  PASS
  max abs diff = 0.2500  (tolerance 0.50)  PASS
  max abs diff = 0.0625  (tolerance 0.50)  PASS
  max abs diff = 0.1250  (tolerance 0.50)  PASS
  max abs diff = 0.1250  (tolerance 0.50)  PASS
  max abs diff = 0.1250  (tolerance 0.50)  PASS
  max abs diff = 0.0312  (tolerance 0.50)  PASS
  max abs diff = 0.1250  (tolerance 0.50)  PASS
──── 13/13 passed ────
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
  fwd      TK   1074.02 us | cuBLASLt   1102.50 us | cuBLAS   1436.61 us | TK/cuBLASLt 0.97x
  dInp   TK   1087.24 us | cuBLASLt   1007.14 us | cuBLAS   1007.51 us | TK/cuBLASLt 1.08x
  dW     TK   1482.29 us | cuBLASLt   1102.18 us | cuBLAS    988.24 us | TK/cuBLASLt 1.34x
  dW+accum TK   1482.94 us | cuBLASLt   1118.04 us | cuBLAS   1004.10 us | TK/cuBLASLt 1.33x
  fwd      TK    376.35 us | cuBLASLt    372.21 us | cuBLAS    487.64 us | TK/cuBLASLt 1.01x
  dInp   TK    380.85 us | cuBLASLt    365.83 us | cuBLAS    388.95 us | TK/cuBLASLt 1.04x
  dW     TK    547.80 us | cuBLASLt    380.61 us | cuBLAS    326.73 us | TK/cuBLASLt 1.44x
  dW+accum TK    541.32 us | cuBLASLt    409.23 us | cuBLAS    353.88 us | TK/cuBLASLt 1.32x
  fwd+GeLU TK fused   1534.16 us | TK explicit   2006.00 us | cuBLASLt   1494.77 us | cuBLAS explicit   2480.89 us | explicit/cuBLASLt 1.34x
  dInp   TK   1506.07 us | cuBLASLt   1342.03 us | cuBLAS   1328.38 us | TK/cuBLASLt 1.12x
  dW     TK   1735.23 us | cuBLASLt   1512.93 us | cuBLAS   1313.03 us | TK/cuBLASLt 1.15x
  dW+accum TK   1750.59 us | cuBLASLt   1476.52 us | cuBLAS   1316.64 us | TK/cuBLASLt 1.19x
  fwd      TK   1421.01 us | cuBLASLt   1396.96 us | cuBLAS   1548.46 us | TK/cuBLASLt 1.02x
  dInp   TK   1508.93 us | cuBLASLt   1372.53 us | cuBLAS   1419.24 us | TK/cuBLASLt 1.10x
  dInp+dGeLU TK   1815.84 us | cuBLASLt fused   1830.41 us | cuBLASLt explicit   2173.64 us | cuBLAS explicit   2184.55 us | explicit/fused 1.19x
  dW     TK   1724.97 us | cuBLASLt   1502.98 us | cuBLAS   1335.24 us | TK/cuBLASLt 1.15x
  dW+accum TK   1762.76 us | cuBLASLt   1496.68 us | cuBLAS   1341.54 us | TK/cuBLASLt 1.18x
  fwd      TK  27653.08 us | cuBLASLt  22112.01 us | cuBLAS  22131.91 us | TK/cuBLASLt 1.25x
  dInp   TK  23954.51 us | cuBLASLt  21640.67 us | cuBLAS  21049.08 us | TK/cuBLASLt 1.11x
  dW     TK  26273.60 us | cuBLASLt  20887.85 us | cuBLAS  21056.32 us | TK/cuBLASLt 1.26x
  dW+accum TK  26100.24 us | cuBLASLt  20997.01 us | cuBLAS  21003.70 us | TK/cuBLASLt 1.24x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 778.090 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2725.666 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 137.536 us
LayerNorm FusedResidualForward (N=65536, C=768): 279.937 us
LayerNorm Backward (N=65536, C=768): 286.596 us
LayerNorm Forward (N=65536, C=3072): 554.664 us
LayerNorm FusedResidualForward (N=65536, C=3072): 1106.651 us
LayerNorm Backward (N=65536, C=3072): 1283.459 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    83.485 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   546.761 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   536.877 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   805.097 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    22.995 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   187.827 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   247.376 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  4031.725 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  9129.952 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  4153.075 us
cuda_memset                    | logits_elems=3296722944      | CUDA kernel  |  4470.138 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  9026.247 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA kernel  |  9453.716 us
global_norm_squared            | params=124475904             | CUDA         |   185.670 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1837.488 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    86.633 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    64.858 us
cuda_memset                    | hidden_elems=50331648        | CUDA kernel  |    66.076 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   134.006 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA kernel  |   140.560 us
```

## Torch Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521/bench_sm120_torch_matmul.log
```

## CuTeDSL Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521/bench_sm120_cutedsl_matmul.log
```

## Triton Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521/bench_sm120_triton_matmul.log
```

## Torch Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521/bench_sm120_torch_attention.log
```

## cuDNN Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521/bench_sm120_cudnn_attention.log
```

## Triton Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521/bench_sm120_triton_attention.log
```

## Torch Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521/bench_sm120_torch_classifier.log
```

## Triton Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521/bench_sm120_triton_classifier.log
```

## Python Stack LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521/bench_sm120_layernorm_python_stacks.log
```

## Triton Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521/bench_sm120_triton_runtime.log
```

## Torch Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521/bench_sm120_torch_runtime.log
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/3 | loss 11.032358 (+nanz)| norm 22.1414 (+nanz)| lr 8.57e-07 | 2498.91 ms | 40.2% bf16 MFU | 209807 tok/s
step    2/3 | loss 10.958507 (+nanz)| norm 22.0968 (+nanz)| lr 1.71e-06 | 2493.17 ms | 40.3% bf16 MFU | 210289 tok/s
step    3/3 | loss 10.811316 (+nanz)| norm 21.1251 (+nanz)| lr 2.57e-06 | 2494.69 ms | 40.3% bf16 MFU | 210224 tok/s
val loss 10.609911
total average iteration time: 2493.930817 ms
```

## validate_sm120_round
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521 --write-scoreboard scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521/scoreboard-candidates.md --write-selected-backends scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521/selected-backends.json --write-promotion-candidates scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521/promotion-candidates.json --require-manifest --require-stack-probe --require-correctness --require-benchmarks --require-training --forbid-checkpoints`

