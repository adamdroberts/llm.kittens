# SM120 Optimization Round

- run label: `codex_sm120_optional_refresh_current2_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_optional_refresh_current2_20260522`
- train output dir: `log124M/5090_S_codex_sm120_optional_refresh_current2_20260522`
- max steps: `3`
- train zero stage: `1`
- python: `/home/adam/miniconda3/envs/llm-kittens/bin/python`
- cuDNN packed backward route: `saved-forward`
- LibTorch runtime route: `cxx-api-raw-pointer`
- LibTorch runtime supplemental shapes: `gelu_forward`
- LibTorch trainer link probe: `1`
- LibTorch matmul shapes: `qkv attproj fc fcproj lmhead`
- SM120 LibTorch trainer memory route: `0`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- working tree: `687` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Fri May 22 09:52:24 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0 Off |                  N/A |
|100%   34C    P8             40W /  575W |    2069MiB /  32607MiB |      0%      Default |
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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_optional_refresh_current2_20260522/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_optional_refresh_current2_20260522/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1 SM120_USE_LIBTORCH_MEMORY=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0 PYTHON_BIN=/home/adam/miniconda3/envs/llm-kittens/bin/python`


## write_sm120_round_manifest
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_optional_refresh_current2_20260522/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_optional_refresh_current2_20260522/round-manifest.md --run-label codex_sm120_optional_refresh_current2_20260522 --artifact-dir scratch/sm120_rounds/codex_sm120_optional_refresh_current2_20260522 --train-out-dir log124M/5090_S_codex_sm120_optional_refresh_current2_20260522 --max-steps 3 --train-zero-stage 1 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 1 --run-benchmarks 1 --run-python-stack-benchmarks 1 --cudnn-packed-backward-route saved-forward --libtorch-runtime-route cxx-api-raw-pointer --libtorch-runtime-supplemental-shapes gelu_forward --run-libtorch-trainer-link-probe 1 --run-libtorch-matmul-benchmarks 1 --libtorch-matmul-shapes qkv\ attproj\ fc\ fcproj\ lmhead --sm120-use-libtorch-memory 0 --sm120-use-libtorch-grad-zero 0 --sm120-use-libtorch-dresidual-zero 0 --run-training 0 --keep-checkpoints 0`


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


## bench_sm120_torch_matmul
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_matmul.py --repeats 7 --large-repeats 3`


## bench_sm120_libtorch_matmul
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_libtorch_matmul.py --repeats 7 --large-repeats 3 --warmup 3 --json-out scratch/sm120_rounds/codex_sm120_optional_refresh_current2_20260522/bench_sm120_libtorch_matmul.json --shape qkv --shape attproj --shape fc --shape fcproj --shape lmhead`


## bench_sm120_cutedsl_matmul
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_cutedsl_matmul.py`


## bench_sm120_triton_matmul
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_matmul.py --repeats 5 --large-repeats 2`


## bench_sm120_torch_attention
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_attention.py --repeats 7 --warmup 3`


## bench_sm120_cudnn_attention
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_cudnn_attention.py --repeats 7 --warmup 3`


## bench_sm120_triton_attention
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_attention.py --repeats 7 --warmup 3`


## bench_sm120_torch_classifier
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_classifier.py --repeats 7 --warmup 3`


## bench_sm120_triton_classifier
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_classifier.py --repeats 7 --warmup 3`


## bench_sm120_layernorm_python_stacks
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_layernorm.py --rows 65536 --cols 768 3072 --repeats 7 --warmup 3`


## bench_sm120_triton_runtime
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_runtime.py --repeats 7 --warmup 3`


## bench_sm120_torch_runtime
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_runtime.py --repeats 7 --warmup 3`


## bench_sm120_libtorch_runtime
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_libtorch_runtime.py --route cxx-api-raw-pointer --repeats 7 --warmup 3`


## validate_libtorch_trainer_link
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_libtorch_trainer_link.py`


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
  fwd      TK   1073.03 us | cuBLASLt   1063.79 us | cuBLAS   1410.89 us | TK/cuBLASLt 1.01x
  dInp   TK   1092.69 us | cuBLASLt   1014.31 us | cuBLAS   1012.56 us | TK/cuBLASLt 1.08x
  dW     TK   1462.20 us | cuBLASLt   1117.85 us | cuBLAS    995.37 us | TK/cuBLASLt 1.31x
  dW+accum TK   1463.82 us | cuBLASLt   1112.65 us | cuBLAS    999.32 us | TK/cuBLASLt 1.32x
  fwd      TK    376.68 us | cuBLASLt    369.54 us | cuBLAS    483.46 us | TK/cuBLASLt 1.02x
  dInp   TK    381.35 us | cuBLASLt    367.66 us | cuBLAS    365.89 us | TK/cuBLASLt 1.04x
  dW     TK    549.47 us | cuBLASLt    376.59 us | cuBLAS    329.22 us | TK/cuBLASLt 1.46x
  dW+accum TK    546.97 us | cuBLASLt    381.32 us | cuBLAS    336.40 us | TK/cuBLASLt 1.43x
  fwd+GeLU TK fused   1537.88 us | TK explicit   1948.25 us | cuBLASLt   1471.71 us | cuBLAS explicit   2416.48 us | explicit/cuBLASLt 1.32x
  dInp   TK   1448.42 us | cuBLASLt   1345.59 us | cuBLAS   1328.43 us | TK/cuBLASLt 1.08x
  dW     TK   1734.77 us | cuBLASLt   1475.13 us | cuBLAS   1309.13 us | TK/cuBLASLt 1.18x
  dW+accum TK   1718.96 us | cuBLASLt   1475.70 us | cuBLAS   1309.48 us | TK/cuBLASLt 1.16x
  fwd      TK   1419.51 us | cuBLASLt   1343.64 us | cuBLAS   1542.16 us | TK/cuBLASLt 1.06x
  dInp   TK   1476.83 us | cuBLASLt   1380.78 us | cuBLAS   1390.68 us | TK/cuBLASLt 1.07x
  dInp+dGeLU TK   1781.45 us | cuBLASLt fused   1798.84 us | cuBLASLt explicit   2122.93 us | cuBLAS explicit   2147.82 us | explicit/fused 1.18x
  dW     TK   1730.13 us | cuBLASLt   1468.44 us | cuBLAS   1309.74 us | TK/cuBLASLt 1.18x
  dW+accum TK   1726.32 us | cuBLASLt   1499.29 us | cuBLAS   1315.31 us | TK/cuBLASLt 1.15x
  fwd      TK  27308.44 us | cuBLASLt  22152.85 us | cuBLAS  22140.63 us | TK/cuBLASLt 1.23x
  dInp   TK  23700.60 us | cuBLASLt  21618.55 us | cuBLAS  21018.67 us | TK/cuBLASLt 1.10x
  dW     TK  25822.77 us | cuBLASLt  20689.36 us | cuBLAS  21016.01 us | TK/cuBLASLt 1.25x
  dW+accum TK  25817.42 us | cuBLASLt  20747.89 us | cuBLAS  21137.21 us | TK/cuBLASLt 1.24x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 774.667 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2706.639 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 135.130 us
LayerNorm FusedResidualForward (N=65536, C=768): 271.036 us
LayerNorm Backward (N=65536, C=768): 267.484 us
LayerNorm Forward (N=65536, C=3072): 537.587 us
LayerNorm FusedResidualForward (N=65536, C=3072): 1072.468 us
LayerNorm Backward (N=65536, C=3072): 1095.961 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    67.964 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   528.467 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   527.517 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   770.513 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    24.514 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   186.488 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   244.925 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  3898.758 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  8793.012 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  3923.424 us
cuda_memset                    | logits_elems=3296722944      | CUDA kernel  |  3930.119 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  8698.957 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA kernel  |  8779.840 us
cuda_memset                    | grad_elems=124475904         | CUDA runtime |   148.589 us
cuda_memset                    | grad_elems=124475904         | CUDA kernel  |   149.846 us
global_norm_squared            | params=124475904             | CUDA         |   185.749 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1785.632 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    80.172 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    61.096 us
cuda_memset                    | hidden_elems=50331648        | CUDA kernel  |    60.321 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   131.588 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA kernel  |   133.514 us
```

## Torch Matmul Benchmarks

```text
Torch matmul device: NVIDIA GeForce RTX 5090; capability=sm_120
qkv          M=65536 N=2304 K=768 bias=1 gelu=0
  fwd      Torch   1465.78 us
  dInp   Torch   1023.99 us
  dW     Torch   1011.83 us
  dW+accum Torch   1016.74 us
attproj      M=65536 N=768 K=768 bias=1 gelu=0
  fwd      Torch    518.06 us
  dInp   Torch    373.52 us
  dW     Torch    338.69 us
  dW+accum Torch    346.96 us
fc           M=65536 N=3072 K=768 bias=1 gelu=1
  fwd+GeLU Torch   2444.31 us
  dInp   Torch   1368.51 us
  dW     Torch   1357.39 us
  dW+accum Torch   1369.93 us
fcproj       M=65536 N=768 K=3072 bias=1 gelu=0
  fwd      Torch   1595.62 us
  dInp   Torch   1405.41 us
  dInp+dGeLU Torch  27836.94 us
  dW     Torch   1377.93 us
  dW+accum Torch   1402.00 us
lmhead       M=65536 N=50304 K=768 bias=0 gelu=0
  fwd      Torch  22372.54 us
  dInp   Torch  21391.55 us
  dW     Torch  21094.30 us
  dW+accum Torch  21216.13 us
```

## LibTorch C++ Matmul Benchmarks

```text
LibTorch matmul device: NVIDIA GeForce RTX 5090; capability=sm_120
LibTorch matmul route: standalone C++ API cached from_blob handles over existing CUDA pointers
qkv M=65536 N=2304 K=768 bias=1 gelu=0
LibTorch matmul parity dW qkv: PASS max_abs=0.000000
LibTorch matmul parity dW+accum qkv: PASS max_abs=0.000000
  dW       Torch C++   1039.34 us
  dW+accum Torch C++   1042.80 us
attproj M=65536 N=768 K=768 bias=1 gelu=0
LibTorch matmul parity dW attproj: PASS max_abs=0.000000
LibTorch matmul parity dW+accum attproj: PASS max_abs=0.000000
  dW       Torch C++    352.31 us
  dW+accum Torch C++    351.25 us
fc M=65536 N=3072 K=768 bias=1 gelu=1
LibTorch matmul parity dW fc: PASS max_abs=0.000000
LibTorch matmul parity dW+accum fc: PASS max_abs=0.000000
  dW       Torch C++   1368.24 us
  dW+accum Torch C++   1402.28 us
fcproj M=65536 N=768 K=3072 bias=1 gelu=0
LibTorch matmul parity dW fcproj: PASS max_abs=0.000000
LibTorch matmul parity dW+accum fcproj: PASS max_abs=0.000000
  dW       Torch C++   1453.38 us
  dW+accum Torch C++   1452.45 us
lmhead M=65536 N=50304 K=768 bias=0 gelu=0
LibTorch matmul parity dW lmhead: PASS max_abs=0.000000
LibTorch matmul parity dW+accum lmhead: PASS max_abs=0.000000
  dW       Torch C++  22611.81 us
  dW+accum Torch C++  22820.35 us
```

## CuTeDSL Matmul Benchmarks

```text
CuTeDSL package: cutlass 4.5.1
CuTeDSL CUDA available: True
CuTeDSL device: NVIDIA GeForce RTX 5090; capability=sm_120
qkv M=65536 N=2304 K=768 bias=1 gelu=0                   | CuTeDSL     | unavailable: local CuTeDSL BF16 grouped-GEMM path rejects sm_120a
attproj M=65536 N=768 K=768 bias=1 gelu=0                | CuTeDSL     | unavailable: local CuTeDSL BF16 grouped-GEMM path rejects sm_120a
fc M=65536 N=3072 K=768 bias=1 gelu=1                    | CuTeDSL     | unavailable: local CuTeDSL BF16 grouped-GEMM path rejects sm_120a
fcproj M=65536 N=768 K=3072 bias=1 gelu=0                | CuTeDSL     | unavailable: local CuTeDSL BF16 grouped-GEMM path rejects sm_120a
lmhead M=65536 N=50304 K=768 bias=0 gelu=0               | CuTeDSL     | unavailable: local CuTeDSL BF16 grouped-GEMM path rejects sm_120a
```

## Triton Matmul Benchmarks

```text
Triton matmul device: NVIDIA GeForce RTX 5090; capability=sm_120
qkv          M=65536 N=2304 K=768 bias=1 gelu=0
  fwd        Triton   1981.75 us (diff=0.000000, rel=0.000000)
  dInp       Triton   2286.43 us (diff=16.000000, rel=0.002075)
  dW         Triton   2132.13 us (diff=1024.000000, rel=0.003817)
  dW+accum   Triton   2139.38 us (diff=1024.000000, rel=0.003817)
attproj      M=65536 N=768 K=768 bias=1 gelu=0
  fwd        Triton    663.79 us (diff=0.000000, rel=0.000000)
  dInp       Triton    686.04 us (diff=0.000000, rel=0.000000)
  dW         Triton    561.83 us (diff=512.000000, rel=0.001953)
  dW+accum   Triton    566.76 us (diff=0.000000, rel=0.000000)
fc           M=65536 N=3072 K=768 bias=1 gelu=1
  fwd+GeLU   Triton   2702.26 us (diff=0.000000, rel=0.000000)
  dInp       Triton   2981.13 us (diff=0.000000, rel=0.000000)
  dW         Triton   2236.88 us (diff=256.000000, rel=0.002825)
  dW+accum   Triton   2226.09 us (diff=256.000000, rel=0.002825)
fcproj       M=65536 N=768 K=3072 bias=1 gelu=0
  fwd        Triton   2973.55 us (diff=0.000000, rel=0.000000)
  dInp       Triton   2695.74 us (diff=0.000000, rel=0.000000)
  dInp+dGeLU Triton   3308.47 us (diff=32.000000, rel=0.006711)
  dW         Triton   2255.15 us (diff=1024.000000, rel=0.005025)
  dW+accum   Triton   2268.29 us (diff=1024.000000, rel=0.005025)
lmhead       M=65536 N=50304 K=768 bias=0 gelu=0
  fwd        Triton  43486.93 us (diff=0.000000, rel=0.000000)
  dInp       Triton  48587.84 us (diff=512.000000, rel=0.003067)
  dW         Triton  69211.60 us (diff=1024.000000, rel=0.005405)
  dW+accum   Triton  68848.64 us (diff=1024.000000, rel=0.005405)
```

## Torch Attention Benchmarks

```text
Torch Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 556.565 us
Torch Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2160.624 us
TorchPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 1142.946 us
TorchPacked Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 4002.704 us
TorchMaterializedPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 1247.822 us
TorchMaterializedPacked Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 4149.318 us
```

## cuDNN Attention Benchmarks

```text
cuDNN Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 675.282 us (max_diff=0.003906)
cuDNN Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2342.368 us
cuDNNPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 790.320 us
cuDNNPacked Attention Backward route: saved-forward
cuDNNPacked Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2765.123 us
```

## Triton Attention Benchmarks

```text
Triton Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 2079.432 us (diff=0.001953)
TritonPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 2186.347 us (diff=0.003906)
B=64 T=1024 C=768 NH=12 HS=64            | Triton       | unavailable: attention backward is not implemented in this Triton prototype
B=64 T=1024 C=768 NH=12 HS=64            | TritonPacked | unavailable: packed attention backward is not implemented in this Triton prototype
```

## Torch Classifier Benchmarks

```text
Torch classifier device: NVIDIA GeForce RTX 5090; capability=sm_120
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | Torch        | 17435.137 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | Torch        | unavailable: CUDA OOM at full GPT-2 padded-logits shape
```

## Triton Classifier Benchmarks

```text
Triton classifier device: NVIDIA GeForce RTX 5090; capability=sm_120
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | Triton       |  8226.144 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | Triton       | 21731.169 us
```

## Python Stack LayerNorm Benchmarks

```text
Triton LayerNorm device: NVIDIA GeForce RTX 5090; capability=sm_120
Triton LayerNorm Forward (N=65536, C=768): 177.120 us (y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm ForwardNative (N=65536, C=768): 153.088 us (y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm ForwardWithStats (N=65536, C=768): 2201.056 us (y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
Triton LayerNorm BackwardDInput (N=65536, C=768): 221.600 us (dinp_diff=0.015625; dweight/dbias not produced)
Torch LayerNorm BackwardDInputNative (N=65536, C=768): 216.416 us (dinp_diff=0.031250; dweight/dbias not produced)
Torch LayerNorm BackwardDInputNativePlusGrads (N=65536, C=768): 1973.600 us (dinp_diff=0.031250, dweight_diff=1.928711, dbias_diff=1.873962; BF16 dweight/dbias)
Torch LayerNorm BackwardNative (N=65536, C=768): 411.872 us (dinp_diff=0.031250, dweight_diff=1.928711, dbias_diff=1.873962)
Triton LayerNorm BackwardAtomicFP32 (N=65536, C=768): 364.224 us (dinp_diff=0.015625, dweight_diff=0.002136, dbias_diff=0.000549; FP32 dweight/dbias)
Triton LayerNorm FusedResidualForward (N=65536, C=768): 310.528 us (residual_diff=0.000000, y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm FusedResidualForwardNative (N=65536, C=768): 331.488 us (residual_diff=0.000000, y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm FusedResidualForwardWithStats (N=65536, C=768): 3176.512 us (residual_diff=0.000000, y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
Triton LayerNorm Forward (N=65536, C=3072): 574.240 us (y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm ForwardNative (N=65536, C=3072): 545.024 us (y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm ForwardWithStats (N=65536, C=3072): 8916.672 us (y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
Triton LayerNorm BackwardDInput (N=65536, C=3072): 799.040 us (dinp_diff=0.031250; dweight/dbias not produced)
Torch LayerNorm BackwardDInputNative (N=65536, C=3072): 828.576 us (dinp_diff=0.031250; dweight/dbias not produced)
Torch LayerNorm BackwardDInputNativePlusGrads (N=65536, C=3072): 7919.008 us (dinp_diff=0.031250, dweight_diff=1.991577, dbias_diff=1.993774; BF16 dweight/dbias)
Torch LayerNorm BackwardNative (N=65536, C=3072): 1390.688 us (dinp_diff=0.031250, dweight_diff=1.991577, dbias_diff=1.993774)
Triton LayerNorm BackwardAtomicFP32 (N=65536, C=3072): 1425.568 us (dinp_diff=0.031250, dweight_diff=0.003235, dbias_diff=0.000793; FP32 dweight/dbias)
Triton LayerNorm FusedResidualForward (N=65536, C=3072): 1104.608 us (residual_diff=0.000000, y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm FusedResidualForwardNative (N=65536, C=3072): 1306.048 us (residual_diff=0.000000, y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm FusedResidualForwardWithStats (N=65536, C=3072): 12951.232 us (residual_diff=0.000000, y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
```

## Triton Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | Triton       |   132.370 us
bias_add                       | BT=65536 OC=3072             | Triton       |   529.507 us
gelu_forward                   | BT=65536 C=3072              | Triton       |   529.829 us
gelu_backward_inplace          | BT=65536 C=3072              | Triton       |   770.482 us
bias_grad_reduce               | BT=65536 OC=768              | Triton       | unavailable: not implemented in this Triton runtime prototype
bias_grad_reduce               | BT=65536 OC=2304             | Triton       | unavailable: not implemented in this Triton runtime prototype
bias_grad_reduce               | BT=65536 OC=3072             | Triton       | unavailable: not implemented in this Triton runtime prototype
```

## Torch Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | Torch        |   135.883 us
bias_add                       | BT=65536 OC=3072             | Torch        |   530.372 us
gelu_forward                   | BT=65536 C=3072              | Torch        |   528.686 us
gelu_backward_inplace          | BT=65536 C=3072              | Torch        | 26475.491 us
bias_grad_reduce               | BT=65536 OC=768              | Torch        |   320.336 us
bias_grad_reduce               | BT=65536 OC=2304             | Torch        |   969.094 us
bias_grad_reduce               | BT=65536 OC=3072             | Torch        |  1304.864 us
cuda_memset                    | grad_elems=124475904         | Torch        |   148.379 us
global_norm_squared            | params=124475904             | Torch        |  2260.640 us
adamw_update_bf16_state        | params=124475904 no-master   | Torch        |  1198.336 us
adamw_update                   | params=124475904 no-master fp32-state | Torch        |  7284.800 us
encoder_forward                | B=64 T=1024 C=768            | Torch        |   199.632 us
cuda_memset                    | hidden_elems=50331648        | Torch        |    60.032 us
cuda_copy_d2d                  | hidden_elems=50331648        | Torch        |   131.720 us
cuda_memset                    | logits_elems=3296722944      | Torch        |  3953.312 us
cuda_copy_d2d                  | logits_elems=3296722944      | Torch        |  8637.824 us
```

## LibTorch C++ Runtime Benchmarks

```text
LibTorch runtime device: NVIDIA GeForce RTX 5090; capability=sm_120
LibTorch runtime route: standalone C++ API cached from_blob handles over existing CUDA pointers
LibTorch parity cuda_memset hidden_elems=50331648: PASS
LibTorch parity cuda_copy_d2d hidden_elems=50331648: PASS
LibTorch parity cuda_memset logits_elems=3296722944: PASS
LibTorch parity cuda_copy_d2d logits_elems=3296722944: PASS
LibTorch parity cuda_memset grad_elems=124475904: PASS
LibTorch parity gelu_forward BT=65536 C=3072: PASS max_abs=0.000000
cuda_memset                    | hidden_elems=50331648        | Torch C++    |    59.861 us
cuda_copy_d2d                  | hidden_elems=50331648        | Torch C++    |   131.829 us
cuda_memset                    | grad_elems=124475904         | Torch C++    |   148.206 us
cuda_memset                    | logits_elems=3296722944      | Torch C++    |  3911.808 us
cuda_copy_d2d                  | logits_elems=3296722944      | Torch C++    |  8633.024 us
gelu_forward                   | BT=65536 C=3072              | Torch C++    |   528.795 us
```

## LibTorch Trainer Link Probe

```text
LibTorch trainer link route: standalone executable without torch_python
LibTorch trainer link compile: PASS /tmp/torch_extensions/llmk_libtorch_trainer_link_probe/llmk_libtorch_trainer_link_probe
LibTorch trainer link runtime: PASS zero/copy from_blob executable
LibTorch trainer link probe: PASS
```

## Training Steps

```text
missing: scratch/sm120_rounds/codex_sm120_optional_refresh_current2_20260522/train_gpt2cu.log
```

## validate_sm120_round
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_optional_refresh_current2_20260522 --write-scoreboard scratch/sm120_rounds/codex_sm120_optional_refresh_current2_20260522/scoreboard-candidates.md --write-selected-backends scratch/sm120_rounds/codex_sm120_optional_refresh_current2_20260522/selected-backends.json --write-promotion-candidates scratch/sm120_rounds/codex_sm120_optional_refresh_current2_20260522/promotion-candidates.json --require-manifest --require-stack-probe --require-correctness --require-benchmarks`

