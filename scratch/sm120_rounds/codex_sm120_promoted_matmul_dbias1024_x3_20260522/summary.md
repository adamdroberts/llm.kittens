# SM120 Optimization Round

- run label: `codex_sm120_promoted_matmul_dbias1024_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_matmul_dbias1024_x3_20260522`
- max steps: `3`
- train zero stage: `1`
- python: `/home/adam/miniconda3/envs/llm-kittens/bin/python`
- cuDNN packed backward route: `saved-forward`
- LibTorch runtime route: `cxx-api-raw-pointer`
- LibTorch runtime supplemental shapes: `gelu_forward`
- LibTorch trainer link probe: `0`
- LibTorch matmul shapes: `qkv attproj fc fcproj lmhead`
- SM120 LibTorch trainer memory route: `0`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- working tree: `663` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Fri May 22 06:07:02 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0 Off |                  N/A |
|100%   31C    P8             47W /  575W |    2649MiB /  32607MiB |      1%      Default |
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

## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1 SM120_USE_LIBTORCH_MEMORY=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 PYTHON_BIN=/home/adam/miniconda3/envs/llm-kittens/bin/python`


## write_sm120_round_manifest
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522/round-manifest.md --run-label codex_sm120_promoted_matmul_dbias1024_x3_20260522 --artifact-dir scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522 --train-out-dir log124M/5090_S_codex_sm120_promoted_matmul_dbias1024_x3_20260522 --max-steps 3 --train-zero-stage 1 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 0 --run-correctness 1 --run-benchmarks 1 --run-python-stack-benchmarks 0 --cudnn-packed-backward-route saved-forward --libtorch-runtime-route cxx-api-raw-pointer --libtorch-runtime-supplemental-shapes gelu_forward --run-libtorch-trainer-link-probe 0 --run-libtorch-matmul-benchmarks 0 --libtorch-matmul-shapes qkv\ attproj\ fc\ fcproj\ lmhead --sm120-use-libtorch-memory 0 --sm120-use-libtorch-grad-zero 0 --sm120-use-libtorch-dresidual-zero 1 --run-training 1 --keep-checkpoints 0`


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
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_promoted_matmul_dbias1024_x3_20260522 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 3`


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
  fwd      TK   1075.61 us | cuBLASLt   1042.57 us | cuBLAS   1430.95 us | TK/cuBLASLt 1.03x
  dInp   TK   1111.18 us | cuBLASLt   1017.71 us | cuBLAS   1012.57 us | TK/cuBLASLt 1.09x
  dW     TK   1949.22 us | cuBLASLt   1121.84 us | cuBLAS    993.08 us | TK/cuBLASLt 1.74x
  dW+accum TK   1978.96 us | cuBLASLt   1110.68 us | cuBLAS    996.06 us | TK/cuBLASLt 1.78x
  fwd      TK    376.70 us | cuBLASLt    370.02 us | cuBLAS    485.27 us | TK/cuBLASLt 1.02x
  dInp   TK    381.36 us | cuBLASLt    365.79 us | cuBLAS    365.94 us | TK/cuBLASLt 1.04x
  dW     TK   1907.00 us | cuBLASLt    372.23 us | cuBLAS    326.62 us | TK/cuBLASLt 5.12x
  dW+accum TK   1918.11 us | cuBLASLt    373.69 us | cuBLAS    333.27 us | TK/cuBLASLt 5.13x
  fwd+GeLU TK fused   1562.89 us | TK explicit   1993.42 us | cuBLASLt   1517.48 us | cuBLAS explicit   2458.88 us | explicit/cuBLASLt 1.31x
  dInp   TK   1495.19 us | cuBLASLt   1345.66 us | cuBLAS   1350.58 us | TK/cuBLASLt 1.11x
  dW     TK   1957.44 us | cuBLASLt   1522.54 us | cuBLAS   1330.85 us | TK/cuBLASLt 1.29x
  dW+accum TK   1981.60 us | cuBLASLt   1524.95 us | cuBLAS   1310.85 us | TK/cuBLASLt 1.30x
  fwd      TK   1465.72 us | cuBLASLt   1348.29 us | cuBLAS   1601.60 us | TK/cuBLASLt 1.09x
  dInp   TK   1500.40 us | cuBLASLt   1433.03 us | cuBLAS   1382.44 us | TK/cuBLASLt 1.05x
  dInp+dGeLU TK   1780.65 us | cuBLASLt fused   1855.73 us | cuBLASLt explicit   2175.32 us | cuBLAS explicit   2184.16 us | explicit/fused 1.17x
  dW     TK   1941.46 us | cuBLASLt   1472.67 us | cuBLAS   1335.59 us | TK/cuBLASLt 1.32x
  dW+accum TK   1971.54 us | cuBLASLt   1517.40 us | cuBLAS   1312.28 us | TK/cuBLASLt 1.30x
  fwd      TK  27650.79 us | cuBLASLt  22278.66 us | cuBLAS  22104.65 us | TK/cuBLASLt 1.24x
  dInp   TK  24037.26 us | cuBLASLt  21803.85 us | cuBLAS  21350.39 us | TK/cuBLASLt 1.10x
  dW     TK  26377.12 us | cuBLASLt  20901.95 us | cuBLAS  21190.44 us | TK/cuBLASLt 1.26x
  dW+accum TK  26507.52 us | cuBLASLt  21098.45 us | cuBLAS  21235.54 us | TK/cuBLASLt 1.26x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 783.716 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2741.899 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 140.045 us
LayerNorm FusedResidualForward (N=65536, C=768): 275.665 us
LayerNorm Backward (N=65536, C=768): 274.941 us
LayerNorm Forward (N=65536, C=3072): 545.285 us
LayerNorm FusedResidualForward (N=65536, C=3072): 1084.380 us
LayerNorm Backward (N=65536, C=3072): 1104.337 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    86.518 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   546.906 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   535.526 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   790.716 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    27.122 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   188.627 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   249.699 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  3998.266 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  8991.986 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  3998.323 us
cuda_memset                    | logits_elems=3296722944      | CUDA kernel  |  4031.155 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  8760.857 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA kernel  |  9036.576 us
cuda_memset                    | grad_elems=124475904         | CUDA runtime |   148.851 us
cuda_memset                    | grad_elems=124475904         | CUDA kernel  |   150.936 us
global_norm_squared            | params=124475904             | CUDA         |   184.893 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1825.568 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    85.148 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    61.562 us
cuda_memset                    | hidden_elems=50331648        | CUDA kernel  |    60.214 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   131.405 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA kernel  |   135.584 us
```

## Torch Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522/bench_sm120_torch_matmul.log
```

## LibTorch C++ Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522/bench_sm120_libtorch_matmul.log
```

## CuTeDSL Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522/bench_sm120_cutedsl_matmul.log
```

## Triton Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522/bench_sm120_triton_matmul.log
```

## Torch Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522/bench_sm120_torch_attention.log
```

## cuDNN Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522/bench_sm120_cudnn_attention.log
```

## Triton Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522/bench_sm120_triton_attention.log
```

## Torch Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522/bench_sm120_torch_classifier.log
```

## Triton Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522/bench_sm120_triton_classifier.log
```

## Python Stack LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522/bench_sm120_layernorm_python_stacks.log
```

## Triton Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522/bench_sm120_triton_runtime.log
```

## Torch Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522/bench_sm120_torch_runtime.log
```

## LibTorch C++ Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522/bench_sm120_libtorch_runtime.log
```

## LibTorch Trainer Link Probe

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522/validate_libtorch_trainer_link.log
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/3 | loss 11.032358 (+nanz)| norm 22.1414 (+nanz)| lr 8.57e-07 | 2494.12 ms | 40.3% bf16 MFU | 210210 tok/s
step    2/3 | loss 10.958508 (+nanz)| norm 22.0968 (+nanz)| lr 1.71e-06 | 2484.60 ms | 40.5% bf16 MFU | 211015 tok/s
step    3/3 | loss 10.811318 (+nanz)| norm 21.1251 (+nanz)| lr 2.57e-06 | 2489.08 ms | 40.4% bf16 MFU | 210820 tok/s
val loss 10.609922
total average iteration time: 2486.838818 ms
```

## validate_sm120_round
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522 --write-scoreboard scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522/scoreboard-candidates.md --write-selected-backends scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522/selected-backends.json --write-promotion-candidates scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522/promotion-candidates.json --require-manifest --require-correctness --require-benchmarks --require-training --forbid-checkpoints`

