# SM120 Optimization Round

- run label: `codex_sm120_promoted_attn_bwd32_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_attn_bwd32_x3_20260522`
- max steps: `3`
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
- working tree: `639` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Fri May 22 01:05:43 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0  On |                  N/A |
|100%   28C    P0             84W /  575W |    3463MiB /  32607MiB |      1%      Default |
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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522/round-manifest.md --run-label codex_sm120_promoted_attn_bwd32_x3_20260522 --artifact-dir scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522 --train-out-dir log124M/5090_S_codex_sm120_promoted_attn_bwd32_x3_20260522 --max-steps 3 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 0 --run-correctness 1 --run-benchmarks 1 --run-python-stack-benchmarks 0 --cudnn-packed-backward-route saved-forward --libtorch-runtime-route cxx-api-raw-pointer --libtorch-runtime-supplemental-shapes gelu_forward --run-libtorch-trainer-link-probe 0 --run-libtorch-matmul-benchmarks 0 --libtorch-matmul-shapes qkv\ attproj\ fc\ fcproj\ lmhead --sm120-use-libtorch-memory 0 --sm120-use-libtorch-grad-zero 0 --sm120-use-libtorch-dresidual-zero 1 --run-training 1 --keep-checkpoints 0`


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
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_promoted_attn_bwd32_x3_20260522 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 3`


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
  fwd      TK   1069.04 us | cuBLASLt   1041.30 us | cuBLAS   1486.11 us | TK/cuBLASLt 1.03x
  dInp   TK   1087.16 us | cuBLASLt   1012.33 us | cuBLAS   1007.24 us | TK/cuBLASLt 1.07x
  dW     TK   1961.85 us | cuBLASLt   1141.31 us | cuBLAS    995.84 us | TK/cuBLASLt 1.72x
  dW+accum TK   2120.41 us | cuBLASLt   1224.75 us | cuBLAS   1091.46 us | TK/cuBLASLt 1.73x
  fwd      TK    414.97 us | cuBLASLt    413.52 us | cuBLAS    549.13 us | TK/cuBLASLt 1.00x
  dInp   TK    379.47 us | cuBLASLt    365.88 us | cuBLAS    363.90 us | TK/cuBLASLt 1.04x
  dW     TK   1950.43 us | cuBLASLt    406.93 us | cuBLAS    347.45 us | TK/cuBLASLt 4.79x
  dW+accum TK   2105.37 us | cuBLASLt    400.44 us | cuBLAS    371.16 us | TK/cuBLASLt 5.26x
  fwd+GeLU TK fused   1749.90 us | TK explicit   2029.09 us | cuBLASLt   1641.66 us | cuBLAS explicit   2723.52 us | explicit/cuBLASLt 1.24x
  dInp   TK   1449.87 us | cuBLASLt   1385.53 us | cuBLAS   1479.67 us | TK/cuBLASLt 1.05x
  dW     TK   2222.78 us | cuBLASLt   1646.69 us | cuBLAS   1459.24 us | TK/cuBLASLt 1.35x
  dW+accum TK   1990.04 us | cuBLASLt   1538.84 us | cuBLAS   1353.59 us | TK/cuBLASLt 1.29x
  fwd      TK   1418.97 us | cuBLASLt   1348.54 us | cuBLAS   1568.95 us | TK/cuBLASLt 1.05x
  dInp   TK   1563.04 us | cuBLASLt   1429.35 us | cuBLAS   1447.32 us | TK/cuBLASLt 1.09x
  dInp+dGeLU TK   1905.48 us | cuBLASLt fused   1934.14 us | cuBLASLt explicit   2221.93 us | cuBLAS explicit   2188.79 us | explicit/fused 1.15x
  dW     TK   1984.78 us | cuBLASLt   1521.15 us | cuBLAS   1312.55 us | TK/cuBLASLt 1.30x
  dW+accum TK   1944.58 us | cuBLASLt   1523.13 us | cuBLAS   1458.45 us | TK/cuBLASLt 1.28x
  fwd      TK  27852.95 us | cuBLASLt  23125.81 us | cuBLAS  22331.83 us | TK/cuBLASLt 1.20x
  dInp   TK  24063.70 us | cuBLASLt  23188.75 us | cuBLAS  21521.24 us | TK/cuBLASLt 1.04x
  dW     TK  27340.76 us | cuBLASLt  23058.60 us | cuBLAS  21333.52 us | TK/cuBLASLt 1.19x
  dW+accum TK  29269.78 us | cuBLASLt  22178.35 us | cuBLAS  21620.37 us | TK/cuBLASLt 1.32x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 830.444 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 3074.490 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 155.549 us
LayerNorm FusedResidualForward (N=65536, C=768): 302.782 us
LayerNorm Backward (N=65536, C=768): 273.940 us
LayerNorm Forward (N=65536, C=3072): 555.996 us
LayerNorm FusedResidualForward (N=65536, C=3072): 1103.046 us
LayerNorm Backward (N=65536, C=3072): 1274.517 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    87.994 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   536.851 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   553.143 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   823.267 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    26.357 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   210.048 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   286.090 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  4279.667 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  9522.003 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  4430.848 us
cuda_memset                    | logits_elems=3296722944      | CUDA kernel  |  4580.313 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  9440.742 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA kernel  |  9271.314 us
cuda_memset                    | grad_elems=124475904         | CUDA runtime |   161.334 us
cuda_memset                    | grad_elems=124475904         | CUDA kernel  |   159.790 us
global_norm_squared            | params=124475904             | CUDA         |   185.248 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1860.979 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    86.625 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    70.118 us
cuda_memset                    | hidden_elems=50331648        | CUDA kernel  |    64.562 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   143.942 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA kernel  |   146.038 us
```

## Torch Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522/bench_sm120_torch_matmul.log
```

## LibTorch C++ Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522/bench_sm120_libtorch_matmul.log
```

## CuTeDSL Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522/bench_sm120_cutedsl_matmul.log
```

## Triton Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522/bench_sm120_triton_matmul.log
```

## Torch Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522/bench_sm120_torch_attention.log
```

## cuDNN Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522/bench_sm120_cudnn_attention.log
```

## Triton Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522/bench_sm120_triton_attention.log
```

## Torch Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522/bench_sm120_torch_classifier.log
```

## Triton Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522/bench_sm120_triton_classifier.log
```

## Python Stack LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522/bench_sm120_layernorm_python_stacks.log
```

## Triton Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522/bench_sm120_triton_runtime.log
```

## Torch Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522/bench_sm120_torch_runtime.log
```

## LibTorch C++ Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522/bench_sm120_libtorch_runtime.log
```

## LibTorch Trainer Link Probe

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_attn_bwd32_x3_20260522/validate_libtorch_trainer_link.log
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/3 | loss 11.032358 (+nanz)| norm 22.1414 (+nanz)| lr 8.57e-07 | 2511.08 ms | 40.0% bf16 MFU | 208790 tok/s
step    2/3 | loss 10.958507 (+nanz)| norm 22.0968 (+nanz)| lr 1.71e-06 | 2502.21 ms | 40.2% bf16 MFU | 209530 tok/s
step    3/3 | loss 10.811316 (+nanz)| norm 21.1251 (+nanz)| lr 2.57e-06 | 2502.79 ms | 40.2% bf16 MFU | 209505 tok/s
val loss 10.609911
total average iteration time: 2502.500892 ms
```
