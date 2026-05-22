# SM120 Optimization Round

- run label: `codex_sm120_runtime_grad_zero_current2_x10_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522`
- train output dir: `log124M/5090_S_codex_sm120_runtime_grad_zero_current2_x10_20260522`
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
- working tree: `687` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Fri May 22 10:00:21 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0 Off |                  N/A |
|100%   32C    P8             43W /  575W |     742MiB /  32607MiB |      0%      Default |
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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522/round-manifest.md --run-label codex_sm120_runtime_grad_zero_current2_x10_20260522 --artifact-dir scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522 --train-out-dir log124M/5090_S_codex_sm120_runtime_grad_zero_current2_x10_20260522 --max-steps 10 --train-zero-stage 1 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 0 --run-correctness 1 --run-benchmarks 1 --run-python-stack-benchmarks 0 --cudnn-packed-backward-route saved-forward --libtorch-runtime-route cxx-api-raw-pointer --libtorch-runtime-supplemental-shapes gelu_forward --run-libtorch-trainer-link-probe 0 --run-libtorch-matmul-benchmarks 0 --libtorch-matmul-shapes qkv\ attproj\ fc\ fcproj\ lmhead --sm120-use-libtorch-memory 0 --sm120-use-libtorch-grad-zero 0 --sm120-use-libtorch-dresidual-zero 1 --run-training 1 --keep-checkpoints 0`


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
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_runtime_grad_zero_current2_x10_20260522 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 10`


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
  fwd      TK   1075.93 us | cuBLASLt   1039.05 us | cuBLAS   1407.14 us | TK/cuBLASLt 1.04x
  dInp   TK   1086.82 us | cuBLASLt   1015.19 us | cuBLAS   1012.75 us | TK/cuBLASLt 1.07x
  dW     TK   1924.16 us | cuBLASLt   1112.86 us | cuBLAS    993.13 us | TK/cuBLASLt 1.73x
  dW+accum TK   1931.48 us | cuBLASLt   1114.60 us | cuBLAS    999.60 us | TK/cuBLASLt 1.73x
  fwd      TK    376.25 us | cuBLASLt    370.99 us | cuBLAS    484.60 us | TK/cuBLASLt 1.01x
  dInp   TK    381.31 us | cuBLASLt    370.91 us | cuBLAS    364.85 us | TK/cuBLASLt 1.03x
  dW     TK   1907.08 us | cuBLASLt    376.20 us | cuBLAS    327.02 us | TK/cuBLASLt 5.07x
  dW+accum TK   1908.25 us | cuBLASLt    375.52 us | cuBLAS    329.82 us | TK/cuBLASLt 5.08x
  fwd+GeLU TK fused   1535.58 us | TK explicit   1970.43 us | cuBLASLt   1474.34 us | cuBLAS explicit   2412.91 us | explicit/cuBLASLt 1.34x
  dInp   TK   1448.79 us | cuBLASLt   1344.45 us | cuBLAS   1327.38 us | TK/cuBLASLt 1.08x
  dW     TK   1947.04 us | cuBLASLt   1485.80 us | cuBLAS   1313.27 us | TK/cuBLASLt 1.31x
  dW+accum TK   1975.15 us | cuBLASLt   1492.69 us | cuBLAS   1326.42 us | TK/cuBLASLt 1.32x
  fwd      TK   1419.75 us | cuBLASLt   1343.70 us | cuBLAS   1549.59 us | TK/cuBLASLt 1.06x
  dInp   TK   1476.39 us | cuBLASLt   1381.36 us | cuBLAS   1391.97 us | TK/cuBLASLt 1.07x
  dInp+dGeLU TK   1778.86 us | cuBLASLt fused   1818.57 us | cuBLASLt explicit   2138.85 us | cuBLAS explicit   2138.58 us | explicit/fused 1.18x
  dW     TK   1941.46 us | cuBLASLt   1486.82 us | cuBLAS   1312.28 us | TK/cuBLASLt 1.31x
  dW+accum TK   1949.14 us | cuBLASLt   1477.03 us | cuBLAS   1314.01 us | TK/cuBLASLt 1.32x
  fwd      TK  27291.82 us | cuBLASLt  22065.28 us | cuBLAS  21991.47 us | TK/cuBLASLt 1.24x
  dInp   TK  23728.47 us | cuBLASLt  21598.56 us | cuBLAS  21050.97 us | TK/cuBLASLt 1.10x
  dW     TK  26060.98 us | cuBLASLt  20765.26 us | cuBLAS  21017.13 us | TK/cuBLASLt 1.26x
  dW+accum TK  26106.03 us | cuBLASLt  20751.45 us | cuBLAS  20968.01 us | TK/cuBLASLt 1.26x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 774.759 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2701.580 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 136.911 us
LayerNorm FusedResidualForward (N=65536, C=768): 269.828 us
LayerNorm Backward (N=65536, C=768): 263.500 us
LayerNorm Forward (N=65536, C=3072): 537.373 us
LayerNorm FusedResidualForward (N=65536, C=3072): 1072.991 us
LayerNorm Backward (N=65536, C=3072): 1086.700 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    80.531 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   536.718 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   528.260 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   770.377 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    24.534 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   186.517 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   245.342 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  3899.392 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  8698.489 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  3960.262 us
cuda_memset                    | logits_elems=3296722944      | CUDA kernel  |  3928.999 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  8657.594 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA kernel  |  8820.704 us
cuda_memset                    | grad_elems=124475904         | CUDA runtime |   150.272 us
cuda_memset                    | grad_elems=124475904         | CUDA kernel  |   149.229 us
global_norm_squared            | params=124475904             | CUDA         |   185.013 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1784.950 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    84.049 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    59.683 us
cuda_memset                    | hidden_elems=50331648        | CUDA kernel  |    60.145 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   131.545 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA kernel  |   135.119 us
```

## Torch Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522/bench_sm120_torch_matmul.log
```

## LibTorch C++ Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522/bench_sm120_libtorch_matmul.log
```

## CuTeDSL Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522/bench_sm120_cutedsl_matmul.log
```

## Triton Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522/bench_sm120_triton_matmul.log
```

## Torch Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522/bench_sm120_torch_attention.log
```

## cuDNN Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522/bench_sm120_cudnn_attention.log
```

## Triton Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522/bench_sm120_triton_attention.log
```

## Torch Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522/bench_sm120_torch_classifier.log
```

## Triton Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522/bench_sm120_triton_classifier.log
```

## Python Stack LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522/bench_sm120_layernorm_python_stacks.log
```

## Triton Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522/bench_sm120_triton_runtime.log
```

## Torch Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522/bench_sm120_torch_runtime.log
```

## LibTorch C++ Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522/bench_sm120_libtorch_runtime.log
```

## LibTorch Trainer Link Probe

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522/validate_libtorch_trainer_link.log
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/10 | loss 11.032358 (+nanz)| norm 22.1414 (+nanz)| lr 8.57e-07 | 2470.01 ms | 40.7% bf16 MFU | 212262 tok/s
step    2/10 | loss 10.958507 (+nanz)| norm 22.0968 (+nanz)| lr 1.71e-06 | 2464.87 ms | 40.8% bf16 MFU | 212704 tok/s
step    3/10 | loss 10.811316 (+nanz)| norm 21.1251 (+nanz)| lr 2.57e-06 | 2468.19 ms | 40.7% bf16 MFU | 212557 tok/s
step    4/10 | loss 10.610130 (+nanz)| norm 18.7014 (+nanz)| lr 3.43e-06 | 2473.92 ms | 40.6% bf16 MFU | 212336 tok/s
step    5/10 | loss 10.392586 (+nanz)| norm 15.0184 (+nanz)| lr 4.29e-06 | 2474.82 ms | 40.6% bf16 MFU | 212205 tok/s
step    6/10 | loss 10.186255 (+nanz)| norm 12.0843 (+nanz)| lr 5.14e-06 | 2477.89 ms | 40.6% bf16 MFU | 212068 tok/s
step    7/10 | loss 10.010621 (+nanz)| norm 10.2002 (+nanz)| lr 6.00e-06 | 2479.35 ms | 40.6% bf16 MFU | 211953 tok/s
step    8/10 | loss 9.855870 (+nanz)| norm 8.7905 (+nanz)| lr 6.86e-06 | 2480.94 ms | 40.5% bf16 MFU | 211850 tok/s
step    9/10 | loss 9.719423 (+nanz)| norm 7.4665 (+nanz)| lr 7.71e-06 | 2483.13 ms | 40.5% bf16 MFU | 211744 tok/s
step   10/10 | loss 9.588612 (+nanz)| norm 6.3099 (+nanz)| lr 8.57e-06 | 2483.18 ms | 40.5% bf16 MFU | 211662 tok/s
val loss 9.483727
total average iteration time: 2476.255973 ms
```

## validate_sm120_round
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522 --write-scoreboard scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522/scoreboard-candidates.md --write-selected-backends scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522/selected-backends.json --write-promotion-candidates scratch/sm120_rounds/codex_sm120_runtime_grad_zero_current2_x10_20260522/promotion-candidates.json --require-manifest --require-correctness --require-benchmarks --require-training --forbid-checkpoints`

