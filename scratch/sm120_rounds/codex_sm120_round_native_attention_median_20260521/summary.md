# SM120 Optimization Round

- run label: `codex_sm120_round_native_attention_median_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_native_attention_median_20260521`
- max steps: `3`
- python: `/home/adam/miniconda3/envs/llm-kittens/bin/python`
- git commit: `0f21747`
- working tree: `487` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Thu May 21 03:46:51 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0 Off |                  N/A |
|100%   29C    P8             46W /  575W |     819MiB /  32607MiB |      0%      Default |
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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521/round-manifest.md --run-label codex_sm120_round_native_attention_median_20260521 --artifact-dir scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521 --train-out-dir log124M/5090_S_codex_sm120_round_native_attention_median_20260521 --max-steps 3 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 1 --run-benchmarks 1 --run-python-stack-benchmarks 0 --run-training 0 --keep-checkpoints 0`


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
  fwd      TK   1073.00 us | cuBLASLt   1043.09 us | cuBLAS   1435.68 us | TK/cuBLASLt 1.03x
  dInp   TK   1093.69 us | cuBLASLt   1014.27 us | cuBLAS   1012.36 us | TK/cuBLASLt 1.08x
  dW     TK   1468.62 us | cuBLASLt   1113.88 us | cuBLAS    993.68 us | TK/cuBLASLt 1.32x
  dW+accum TK   1499.26 us | cuBLASLt   1115.10 us | cuBLAS    996.62 us | TK/cuBLASLt 1.34x
  fwd      TK    376.28 us | cuBLASLt    374.03 us | cuBLAS    482.51 us | TK/cuBLASLt 1.01x
  dInp   TK    381.19 us | cuBLASLt    367.37 us | cuBLAS    365.59 us | TK/cuBLASLt 1.04x
  dW     TK    546.74 us | cuBLASLt    375.23 us | cuBLAS    329.10 us | TK/cuBLASLt 1.46x
  dW+accum TK    543.32 us | cuBLASLt    379.44 us | cuBLAS    335.87 us | TK/cuBLASLt 1.43x
  fwd+GeLU TK fused   1587.45 us | TK explicit   2008.97 us | cuBLASLt   1492.05 us | cuBLAS explicit   2470.53 us | explicit/cuBLASLt 1.35x
  dInp   TK   1463.88 us | cuBLASLt   1375.35 us | cuBLAS   1366.13 us | TK/cuBLASLt 1.06x
  dW     TK   1758.30 us | cuBLASLt   1524.38 us | cuBLAS   1338.75 us | TK/cuBLASLt 1.15x
  dW+accum TK   1732.22 us | cuBLASLt   1508.84 us | cuBLAS   1316.63 us | TK/cuBLASLt 1.15x
  fwd      TK   1447.64 us | cuBLASLt   1365.11 us | cuBLAS   1581.96 us | TK/cuBLASLt 1.06x
  dInp   TK   1515.64 us | cuBLASLt   1404.94 us | cuBLAS   1405.70 us | TK/cuBLASLt 1.08x
  dInp+dGeLU TK   1807.37 us | cuBLASLt fused   1839.63 us | cuBLASLt explicit   2195.48 us | cuBLAS explicit   2190.12 us | explicit/fused 1.19x
  dW     TK   1757.35 us | cuBLASLt   1484.74 us | cuBLAS   1326.05 us | TK/cuBLASLt 1.18x
  dW+accum TK   1762.78 us | cuBLASLt   1528.82 us | cuBLAS   1315.10 us | TK/cuBLASLt 1.15x
  fwd      TK  27869.01 us | cuBLASLt  22412.14 us | cuBLAS  22334.93 us | TK/cuBLASLt 1.24x
  dInp   TK  24003.22 us | cuBLASLt  22074.02 us | cuBLAS  21416.19 us | TK/cuBLASLt 1.09x
  dW     TK  26173.65 us | cuBLASLt  20973.02 us | cuBLAS  21190.02 us | TK/cuBLASLt 1.25x
  dW+accum TK  26202.21 us | cuBLASLt  20984.58 us | cuBLAS  21226.04 us | TK/cuBLASLt 1.25x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 785.867 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2746.680 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 137.176 us
LayerNorm FusedResidualForward (N=65536, C=768): 275.449 us
LayerNorm Backward (N=65536, C=768): 288.253 us
LayerNorm Forward (N=65536, C=3072): 544.797 us
LayerNorm FusedResidualForward (N=65536, C=3072): 1085.474 us
LayerNorm Backward (N=65536, C=3072): 1267.071 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    91.607 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   537.034 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   528.018 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   780.614 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    24.941 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   186.702 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   245.163 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  3998.093 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  8917.556 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  3999.635 us
cuda_memset                    | logits_elems=3296722944      | CUDA kernel  |  4084.102 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  8796.077 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA kernel  |  9209.209 us
global_norm_squared            | params=124475904             | CUDA         |   185.090 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1806.457 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    84.854 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    60.232 us
cuda_memset                    | hidden_elems=50331648        | CUDA kernel  |    62.571 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   131.649 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA kernel  |   138.714 us
```

## Torch Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521/bench_sm120_torch_matmul.log
```

## CuTeDSL Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521/bench_sm120_cutedsl_matmul.log
```

## Triton Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521/bench_sm120_triton_matmul.log
```

## Torch Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521/bench_sm120_torch_attention.log
```

## cuDNN Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521/bench_sm120_cudnn_attention.log
```

## Triton Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521/bench_sm120_triton_attention.log
```

## Torch Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521/bench_sm120_torch_classifier.log
```

## Triton Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521/bench_sm120_triton_classifier.log
```

## Python Stack LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521/bench_sm120_layernorm_python_stacks.log
```

## Triton Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521/bench_sm120_triton_runtime.log
```

## Torch Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521/bench_sm120_torch_runtime.log
```

## Training Steps

```text
missing: scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521/train_gpt2cu.log
```

## validate_sm120_round
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521 --write-scoreboard scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521/scoreboard-candidates.md --write-selected-backends scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521/selected-backends.json --write-promotion-candidates scratch/sm120_rounds/codex_sm120_round_native_attention_median_20260521/promotion-candidates.json --require-manifest --require-stack-probe --require-correctness --require-benchmarks`

