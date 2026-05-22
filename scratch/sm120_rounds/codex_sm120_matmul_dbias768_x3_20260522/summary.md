# SM120 Optimization Round

- run label: `codex_sm120_matmul_dbias768_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_matmul_dbias768_x3_20260522`
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
- working tree: `678` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Fri May 22 07:55:05 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0 Off |                  N/A |
|100%   33C    P8             47W /  575W |    2711MiB /  32607MiB |      1%      Default |
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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522/round-manifest.md --run-label codex_sm120_matmul_dbias768_x3_20260522 --artifact-dir scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522 --train-out-dir log124M/5090_S_codex_sm120_matmul_dbias768_x3_20260522 --max-steps 3 --train-zero-stage 1 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 0 --run-correctness 1 --run-benchmarks 1 --run-python-stack-benchmarks 0 --cudnn-packed-backward-route saved-forward --libtorch-runtime-route cxx-api-raw-pointer --libtorch-runtime-supplemental-shapes gelu_forward --run-libtorch-trainer-link-probe 0 --run-libtorch-matmul-benchmarks 0 --libtorch-matmul-shapes qkv\ attproj\ fc\ fcproj\ lmhead --sm120-use-libtorch-memory 0 --sm120-use-libtorch-grad-zero 0 --sm120-use-libtorch-dresidual-zero 1 --run-training 1 --keep-checkpoints 0`


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
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_matmul_dbias768_x3_20260522 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 3`


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
  fwd      TK   1073.22 us | cuBLASLt   1060.18 us | cuBLAS   1454.36 us | TK/cuBLASLt 1.01x
  dInp   TK   1087.69 us | cuBLASLt   1036.02 us | cuBLAS   1012.37 us | TK/cuBLASLt 1.05x
  dW     TK   1969.76 us | cuBLASLt   1117.92 us | cuBLAS    992.46 us | TK/cuBLASLt 1.76x
  dW+accum TK   1947.13 us | cuBLASLt   1136.21 us | cuBLAS   1004.98 us | TK/cuBLASLt 1.71x
  fwd      TK    376.76 us | cuBLASLt    374.40 us | cuBLAS    484.07 us | TK/cuBLASLt 1.01x
  dInp   TK    380.48 us | cuBLASLt    365.70 us | cuBLAS    365.49 us | TK/cuBLASLt 1.04x
  dW     TK   1949.50 us | cuBLASLt    374.39 us | cuBLAS    329.30 us | TK/cuBLASLt 5.21x
  dW+accum TK   1970.27 us | cuBLASLt    377.86 us | cuBLAS    332.27 us | TK/cuBLASLt 5.21x
  fwd+GeLU TK fused   1588.34 us | TK explicit   1992.49 us | cuBLASLt   1472.50 us | cuBLAS explicit   2460.36 us | explicit/cuBLASLt 1.35x
  dInp   TK   1472.28 us | cuBLASLt   1358.43 us | cuBLAS   1333.92 us | TK/cuBLASLt 1.08x
  dW     TK   1984.75 us | cuBLASLt   1489.16 us | cuBLAS   1351.31 us | TK/cuBLASLt 1.33x
  dW+accum TK   2004.69 us | cuBLASLt   1489.12 us | cuBLAS   1313.81 us | TK/cuBLASLt 1.35x
  fwd      TK   1422.17 us | cuBLASLt   1384.21 us | cuBLAS   1548.49 us | TK/cuBLASLt 1.03x
  dInp   TK   1488.69 us | cuBLASLt   1404.43 us | cuBLAS   1376.02 us | TK/cuBLASLt 1.06x
  dInp+dGeLU TK   1822.47 us | cuBLASLt fused   1847.85 us | cuBLASLt explicit   2178.11 us | cuBLAS explicit   2164.30 us | explicit/fused 1.18x
  dW     TK   1982.30 us | cuBLASLt   1472.58 us | cuBLAS   1312.52 us | TK/cuBLASLt 1.35x
  dW+accum TK   1988.51 us | cuBLASLt   1498.30 us | cuBLAS   1349.87 us | TK/cuBLASLt 1.33x
  fwd      TK  27640.97 us | cuBLASLt  22480.76 us | cuBLAS  22310.17 us | TK/cuBLASLt 1.23x
  dInp   TK  24013.00 us | cuBLASLt  21917.60 us | cuBLAS  21319.91 us | TK/cuBLASLt 1.10x
  dW     TK  26552.17 us | cuBLASLt  20951.70 us | cuBLAS  21243.13 us | TK/cuBLASLt 1.27x
  dW+accum TK  26423.31 us | cuBLASLt  21130.50 us | cuBLAS  21280.53 us | TK/cuBLASLt 1.25x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 783.308 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2741.627 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 136.594 us
LayerNorm FusedResidualForward (N=65536, C=768): 276.261 us
LayerNorm Backward (N=65536, C=768): 270.834 us
LayerNorm Forward (N=65536, C=3072): 544.574 us
LayerNorm FusedResidualForward (N=65536, C=3072): 1082.835 us
LayerNorm Backward (N=65536, C=3072): 1101.246 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    80.534 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   528.595 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   535.508 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   793.189 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    22.790 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   200.814 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   245.667 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  3999.392 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  8900.455 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  3913.971 us
cuda_memset                    | logits_elems=3296722944      | CUDA kernel  |  4046.515 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  8740.832 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA kernel  |  9015.975 us
cuda_memset                    | grad_elems=124475904         | CUDA runtime |   149.253 us
cuda_memset                    | grad_elems=124475904         | CUDA kernel  |   149.958 us
global_norm_squared            | params=124475904             | CUDA         |   184.419 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1792.147 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    85.944 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    61.701 us
cuda_memset                    | hidden_elems=50331648        | CUDA kernel  |    60.759 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   131.768 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA kernel  |   135.840 us
```

## Torch Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522/bench_sm120_torch_matmul.log
```

## LibTorch C++ Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522/bench_sm120_libtorch_matmul.log
```

## CuTeDSL Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522/bench_sm120_cutedsl_matmul.log
```

## Triton Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522/bench_sm120_triton_matmul.log
```

## Torch Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522/bench_sm120_torch_attention.log
```

## cuDNN Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522/bench_sm120_cudnn_attention.log
```

## Triton Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522/bench_sm120_triton_attention.log
```

## Torch Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522/bench_sm120_torch_classifier.log
```

## Triton Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522/bench_sm120_triton_classifier.log
```

## Python Stack LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522/bench_sm120_layernorm_python_stacks.log
```

## Triton Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522/bench_sm120_triton_runtime.log
```

## Torch Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522/bench_sm120_torch_runtime.log
```

## LibTorch C++ Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522/bench_sm120_libtorch_runtime.log
```

## LibTorch Trainer Link Probe

```text
missing: scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522/validate_libtorch_trainer_link.log
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/3 | loss 11.032358 (+nanz)| norm 22.1414 (+nanz)| lr 8.57e-07 | 2499.60 ms | 40.2% bf16 MFU | 209749 tok/s
step    2/3 | loss 10.958508 (+nanz)| norm 22.0968 (+nanz)| lr 1.71e-06 | 2494.41 ms | 40.3% bf16 MFU | 210185 tok/s
step    3/3 | loss 10.811319 (+nanz)| norm 21.1249 (+nanz)| lr 2.57e-06 | 2501.38 ms | 40.2% bf16 MFU | 209885 tok/s
val loss 10.609926
total average iteration time: 2497.894287 ms
```

## validate_sm120_round
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522 --write-scoreboard scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522/scoreboard-candidates.md --write-selected-backends scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522/selected-backends.json --write-promotion-candidates scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522/promotion-candidates.json --require-manifest --require-correctness --require-benchmarks --require-training --forbid-checkpoints`

