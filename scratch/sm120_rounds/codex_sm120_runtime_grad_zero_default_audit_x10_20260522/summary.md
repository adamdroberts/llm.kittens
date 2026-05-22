# SM120 Optimization Round

- run label: `codex_sm120_runtime_grad_zero_default_audit_x10_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522`
- train output dir: `log124M/5090_S_codex_sm120_runtime_grad_zero_default_audit_x10_20260522`
- max steps: `10`
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
- working tree: `688` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Fri May 22 10:10:40 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0 Off |                  N/A |
|100%   31C    P8             40W /  575W |     876MiB /  32607MiB |      0%      Default |
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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1 SM120_USE_LIBTORCH_MEMORY=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=1 PYTHON_BIN=/home/adam/miniconda3/envs/llm-kittens/bin/python`


## write_sm120_round_manifest
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/round-manifest.md --run-label codex_sm120_runtime_grad_zero_default_audit_x10_20260522 --artifact-dir scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522 --train-out-dir log124M/5090_S_codex_sm120_runtime_grad_zero_default_audit_x10_20260522 --max-steps 10 --train-zero-stage 1 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 1 --run-benchmarks 1 --run-python-stack-benchmarks 0 --cudnn-packed-backward-route saved-forward --libtorch-runtime-route cxx-api-raw-pointer --libtorch-runtime-supplemental-shapes gelu_forward --run-libtorch-trainer-link-probe 0 --run-libtorch-matmul-benchmarks 0 --libtorch-matmul-shapes qkv\ attproj\ fc\ fcproj\ lmhead --sm120-use-libtorch-memory 0 --sm120-use-libtorch-grad-zero 0 --sm120-use-libtorch-dresidual-zero 1 --run-training 1 --keep-checkpoints 0`


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
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_runtime_grad_zero_default_audit_x10_20260522 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 10`


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
  fwd      TK   1073.30 us | cuBLASLt   1041.49 us | cuBLAS   1413.36 us | TK/cuBLASLt 1.03x
  dInp   TK   1086.76 us | cuBLASLt   1011.90 us | cuBLAS   1012.00 us | TK/cuBLASLt 1.07x
  dW     TK   1945.22 us | cuBLASLt   1111.85 us | cuBLAS    993.41 us | TK/cuBLASLt 1.75x
  dW+accum TK   1932.64 us | cuBLASLt   1112.51 us | cuBLAS    997.62 us | TK/cuBLASLt 1.74x
  fwd      TK    376.94 us | cuBLASLt    369.45 us | cuBLAS    483.36 us | TK/cuBLASLt 1.02x
  dInp   TK    381.07 us | cuBLASLt    366.96 us | cuBLAS    365.55 us | TK/cuBLASLt 1.04x
  dW     TK   1906.46 us | cuBLASLt    373.73 us | cuBLAS    326.89 us | TK/cuBLASLt 5.10x
  dW+accum TK   1902.76 us | cuBLASLt    383.92 us | cuBLAS    335.01 us | TK/cuBLASLt 4.96x
  fwd+GeLU TK fused   1532.28 us | TK explicit   1962.52 us | cuBLASLt   1474.56 us | cuBLAS explicit   2412.31 us | explicit/cuBLASLt 1.33x
  dInp   TK   1448.83 us | cuBLASLt   1352.17 us | cuBLAS   1335.40 us | TK/cuBLASLt 1.07x
  dW     TK   1946.34 us | cuBLASLt   1508.11 us | cuBLAS   1313.15 us | TK/cuBLASLt 1.29x
  dW+accum TK   1944.62 us | cuBLASLt   1492.88 us | cuBLAS   1313.41 us | TK/cuBLASLt 1.30x
  fwd      TK   1419.45 us | cuBLASLt   1384.76 us | cuBLAS   1543.66 us | TK/cuBLASLt 1.03x
  dInp   TK   1475.80 us | cuBLASLt   1383.94 us | cuBLAS   1375.98 us | TK/cuBLASLt 1.07x
  dInp+dGeLU TK   1795.72 us | cuBLASLt fused   1809.01 us | cuBLASLt explicit   2131.00 us | cuBLAS explicit   2130.28 us | explicit/fused 1.18x
  dW     TK   1942.04 us | cuBLASLt   1491.87 us | cuBLAS   1309.54 us | TK/cuBLASLt 1.30x
  dW+accum TK   1943.00 us | cuBLASLt   1469.78 us | cuBLAS   1319.49 us | TK/cuBLASLt 1.32x
  fwd      TK  27256.71 us | cuBLASLt  22073.87 us | cuBLAS  22056.10 us | TK/cuBLASLt 1.23x
  dInp   TK  23696.22 us | cuBLASLt  21558.43 us | cuBLAS  21043.99 us | TK/cuBLASLt 1.10x
  dW     TK  25977.76 us | cuBLASLt  20850.93 us | cuBLAS  20883.82 us | TK/cuBLASLt 1.25x
  dW+accum TK  26104.80 us | cuBLASLt  20706.13 us | cuBLAS  20996.98 us | TK/cuBLASLt 1.26x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 776.120 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2702.327 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 135.136 us
LayerNorm FusedResidualForward (N=65536, C=768): 274.999 us
LayerNorm Backward (N=65536, C=768): 265.737 us
LayerNorm Forward (N=65536, C=3072): 538.402 us
LayerNorm FusedResidualForward (N=65536, C=3072): 1071.240 us
LayerNorm Backward (N=65536, C=3072): 1091.596 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    91.817 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   537.901 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   527.468 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   770.103 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    25.413 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   186.765 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   245.147 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  3893.421 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  8749.869 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  3912.717 us
cuda_memset                    | logits_elems=3296722944      | CUDA kernel  |  3929.050 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  8698.899 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA kernel  |  8742.726 us
cuda_memset                    | grad_elems=124475904         | CUDA runtime |   149.947 us
cuda_memset                    | grad_elems=124475904         | CUDA kernel  |   150.096 us
global_norm_squared            | params=124475904             | CUDA         |   185.014 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1783.488 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    83.692 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    61.329 us
cuda_memset                    | hidden_elems=50331648        | CUDA kernel  |    60.666 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   131.491 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA kernel  |   132.475 us
```

## Torch Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/bench_sm120_torch_matmul.log
```

## LibTorch C++ Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/bench_sm120_libtorch_matmul.log
```

## CuTeDSL Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/bench_sm120_cutedsl_matmul.log
```

## Triton Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/bench_sm120_triton_matmul.log
```

## Torch Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/bench_sm120_torch_attention.log
```

## cuDNN Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/bench_sm120_cudnn_attention.log
```

## Triton Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/bench_sm120_triton_attention.log
```

## Torch Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/bench_sm120_torch_classifier.log
```

## Triton Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/bench_sm120_triton_classifier.log
```

## Python Stack LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/bench_sm120_layernorm_python_stacks.log
```

## Triton Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/bench_sm120_triton_runtime.log
```

## Torch Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/bench_sm120_torch_runtime.log
```

## LibTorch C++ Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/bench_sm120_libtorch_runtime.log
```

## LibTorch Trainer Link Probe

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/validate_libtorch_trainer_link.log
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/10 | loss 11.032358 (+nanz)| norm 22.1414 (+nanz)| lr 8.57e-07 | 2468.68 ms | 40.7% bf16 MFU | 212376 tok/s
step    2/10 | loss 10.958507 (+nanz)| norm 22.0968 (+nanz)| lr 1.71e-06 | 2461.91 ms | 40.8% bf16 MFU | 212960 tok/s
step    3/10 | loss 10.811316 (+nanz)| norm 21.1251 (+nanz)| lr 2.57e-06 | 2465.47 ms | 40.8% bf16 MFU | 212802 tok/s
step    4/10 | loss 10.610130 (+nanz)| norm 18.7014 (+nanz)| lr 3.43e-06 | 2467.56 ms | 40.7% bf16 MFU | 212687 tok/s
step    5/10 | loss 10.392586 (+nanz)| norm 15.0184 (+nanz)| lr 4.29e-06 | 2472.84 ms | 40.7% bf16 MFU | 212507 tok/s
step    6/10 | loss 10.186255 (+nanz)| norm 12.0843 (+nanz)| lr 5.14e-06 | 2475.15 ms | 40.6% bf16 MFU | 212355 tok/s
step    7/10 | loss 10.010621 (+nanz)| norm 10.2002 (+nanz)| lr 6.00e-06 | 2477.91 ms | 40.6% bf16 MFU | 212210 tok/s
step    8/10 | loss 9.855870 (+nanz)| norm 8.7905 (+nanz)| lr 6.86e-06 | 2478.44 ms | 40.6% bf16 MFU | 212098 tok/s
step    9/10 | loss 9.719423 (+nanz)| norm 7.4665 (+nanz)| lr 7.71e-06 | 2480.29 ms | 40.5% bf16 MFU | 211992 tok/s
step   10/10 | loss 9.588612 (+nanz)| norm 6.3099 (+nanz)| lr 8.57e-06 | 2480.50 ms | 40.5% bf16 MFU | 211907 tok/s
val loss 9.483727
total average iteration time: 2473.341915 ms
```

## validate_sm120_round
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522 --write-scoreboard scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/scoreboard-candidates.md --write-selected-backends scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/selected-backends.json --write-promotion-candidates scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/promotion-candidates.json --require-manifest --require-stack-probe --require-correctness --require-benchmarks --require-training --forbid-checkpoints`

