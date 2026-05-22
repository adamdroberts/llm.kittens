# SM120 Optimization Round

- run label: `codex_sm120_current_optional_refresh_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_current_optional_refresh_20260522`
- train output dir: `log124M/5090_S_codex_sm120_current_optional_refresh_20260522`
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
- working tree: `681` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Fri May 22 09:07:26 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0 Off |                  N/A |
|100%   34C    P8             48W /  575W |     798MiB /  32607MiB |      0%      Default |
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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_current_optional_refresh_20260522/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_current_optional_refresh_20260522/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1 SM120_USE_LIBTORCH_MEMORY=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0 PYTHON_BIN=/home/adam/miniconda3/envs/llm-kittens/bin/python`


## write_sm120_round_manifest
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_current_optional_refresh_20260522/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_current_optional_refresh_20260522/round-manifest.md --run-label codex_sm120_current_optional_refresh_20260522 --artifact-dir scratch/sm120_rounds/codex_sm120_current_optional_refresh_20260522 --train-out-dir log124M/5090_S_codex_sm120_current_optional_refresh_20260522 --max-steps 3 --train-zero-stage 1 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 1 --run-benchmarks 1 --run-python-stack-benchmarks 1 --cudnn-packed-backward-route saved-forward --libtorch-runtime-route cxx-api-raw-pointer --libtorch-runtime-supplemental-shapes gelu_forward --run-libtorch-trainer-link-probe 1 --run-libtorch-matmul-benchmarks 1 --libtorch-matmul-shapes qkv\ attproj\ fc\ fcproj\ lmhead --sm120-use-libtorch-memory 0 --sm120-use-libtorch-grad-zero 0 --sm120-use-libtorch-dresidual-zero 0 --run-training 0 --keep-checkpoints 0`


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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_libtorch_matmul.py --repeats 7 --large-repeats 3 --warmup 3 --json-out scratch/sm120_rounds/codex_sm120_current_optional_refresh_20260522/bench_sm120_libtorch_matmul.json --shape qkv --shape attproj --shape fc --shape fcproj --shape lmhead`


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
  fwd      TK   1073.56 us | cuBLASLt   1040.41 us | cuBLAS   1434.30 us | TK/cuBLASLt 1.03x
  dInp   TK   1092.21 us | cuBLASLt   1012.18 us | cuBLAS   1012.36 us | TK/cuBLASLt 1.08x
  dW     TK   1460.90 us | cuBLASLt   1116.21 us | cuBLAS    993.39 us | TK/cuBLASLt 1.31x
  dW+accum TK   1496.62 us | cuBLASLt   1112.70 us | cuBLAS   1000.45 us | TK/cuBLASLt 1.35x
  fwd      TK    376.70 us | cuBLASLt    369.82 us | cuBLAS    483.15 us | TK/cuBLASLt 1.02x
  dInp   TK    380.51 us | cuBLASLt    367.37 us | cuBLAS    365.77 us | TK/cuBLASLt 1.04x
  dW     TK    541.33 us | cuBLASLt    375.93 us | cuBLAS    328.50 us | TK/cuBLASLt 1.44x
  dW+accum TK    543.70 us | cuBLASLt    379.10 us | cuBLAS    332.13 us | TK/cuBLASLt 1.43x
  fwd+GeLU TK fused   1587.62 us | TK explicit   1969.45 us | cuBLASLt   1483.62 us | cuBLAS explicit   2466.56 us | explicit/cuBLASLt 1.33x
  dInp   TK   1448.58 us | cuBLASLt   1346.06 us | cuBLAS   1329.85 us | TK/cuBLASLt 1.08x
  dW     TK   1705.86 us | cuBLASLt   1481.81 us | cuBLAS   1312.64 us | TK/cuBLASLt 1.15x
  dW+accum TK   1764.14 us | cuBLASLt   1488.78 us | cuBLAS   1333.35 us | TK/cuBLASLt 1.18x
  fwd      TK   1462.40 us | cuBLASLt   1366.87 us | cuBLAS   1593.79 us | TK/cuBLASLt 1.07x
  dInp   TK   1487.42 us | cuBLASLt   1425.65 us | cuBLAS   1395.40 us | TK/cuBLASLt 1.04x
  dInp+dGeLU TK   1817.64 us | cuBLASLt fused   1850.49 us | cuBLASLt explicit   2173.26 us | cuBLAS explicit   2179.49 us | explicit/fused 1.17x
  dW     TK   1708.51 us | cuBLASLt   1511.77 us | cuBLAS   1312.60 us | TK/cuBLASLt 1.13x
  dW+accum TK   1748.42 us | cuBLASLt   1487.20 us | cuBLAS   1341.09 us | TK/cuBLASLt 1.18x
  fwd      TK  27754.26 us | cuBLASLt  22344.68 us | cuBLAS  22391.79 us | TK/cuBLASLt 1.24x
  dInp   TK  24037.77 us | cuBLASLt  21923.55 us | cuBLAS  21318.02 us | TK/cuBLASLt 1.10x
  dW     TK  25990.55 us | cuBLASLt  20939.03 us | cuBLAS  21235.84 us | TK/cuBLASLt 1.24x
  dW+accum TK  26207.72 us | cuBLASLt  20987.13 us | cuBLAS  21270.97 us | TK/cuBLASLt 1.25x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 784.749 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2738.776 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 140.567 us
LayerNorm FusedResidualForward (N=65536, C=768): 273.463 us
LayerNorm Backward (N=65536, C=768): 273.412 us
LayerNorm Forward (N=65536, C=3072): 545.719 us
LayerNorm FusedResidualForward (N=65536, C=3072): 1082.852 us
LayerNorm Backward (N=65536, C=3072): 1106.244 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    61.693 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   546.833 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   527.846 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   790.993 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    24.883 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   186.549 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   245.187 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  3899.872 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  8895.597 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  4007.981 us
cuda_memset                    | logits_elems=3296722944      | CUDA kernel  |  3975.066 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  8779.342 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA kernel  |  8942.599 us
cuda_memset                    | grad_elems=124475904         | CUDA runtime |   150.442 us
cuda_memset                    | grad_elems=124475904         | CUDA kernel  |   149.478 us
global_norm_squared            | params=124475904             | CUDA         |   184.930 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1830.045 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    84.120 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    59.980 us
cuda_memset                    | hidden_elems=50331648        | CUDA kernel  |    60.333 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   131.589 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA kernel  |   133.520 us
```

## Torch Matmul Benchmarks

```text
Torch matmul device: NVIDIA GeForce RTX 5090; capability=sm_120
qkv          M=65536 N=2304 K=768 bias=1 gelu=0
  fwd      Torch   1457.15 us
  dInp   Torch   1017.20 us
  dW     Torch   1002.57 us
  dW+accum Torch   1012.49 us
attproj      M=65536 N=768 K=768 bias=1 gelu=0
  fwd      Torch    516.14 us
  dInp   Torch    373.58 us
  dW     Torch    339.28 us
  dW+accum Torch    346.03 us
fc           M=65536 N=3072 K=768 bias=1 gelu=1
  fwd+GeLU Torch   2443.53 us
  dInp   Torch   1368.52 us
  dW     Torch   1359.99 us
  dW+accum Torch   1374.37 us
fcproj       M=65536 N=768 K=3072 bias=1 gelu=0
  fwd      Torch   1601.86 us
  dInp   Torch   1397.46 us
  dInp+dGeLU Torch  28229.93 us
  dW     Torch   1389.41 us
  dW+accum Torch   1407.90 us
lmhead       M=65536 N=50304 K=768 bias=0 gelu=0
  fwd      Torch  22929.34 us
  dInp   Torch  21676.86 us
  dW     Torch  21553.95 us
  dW+accum Torch  21373.09 us
```

## LibTorch C++ Matmul Benchmarks

```text
LibTorch matmul device: NVIDIA GeForce RTX 5090; capability=sm_120
LibTorch matmul route: standalone C++ API cached from_blob handles over existing CUDA pointers
qkv M=65536 N=2304 K=768 bias=1 gelu=0
LibTorch matmul parity dW qkv: PASS max_abs=0.000000
LibTorch matmul parity dW+accum qkv: PASS max_abs=0.000000
  dW       Torch C++   1037.50 us
  dW+accum Torch C++   1040.88 us
attproj M=65536 N=768 K=768 bias=1 gelu=0
LibTorch matmul parity dW attproj: PASS max_abs=0.000000
LibTorch matmul parity dW+accum attproj: PASS max_abs=0.000000
  dW       Torch C++    349.78 us
  dW+accum Torch C++    352.74 us
fc M=65536 N=3072 K=768 bias=1 gelu=1
LibTorch matmul parity dW fc: PASS max_abs=0.000000
LibTorch matmul parity dW+accum fc: PASS max_abs=0.000000
  dW       Torch C++   1386.69 us
  dW+accum Torch C++   1395.67 us
fcproj M=65536 N=768 K=3072 bias=1 gelu=0
LibTorch matmul parity dW fcproj: PASS max_abs=0.000000
LibTorch matmul parity dW+accum fcproj: PASS max_abs=0.000000
  dW       Torch C++   1445.97 us
  dW+accum Torch C++   1443.46 us
lmhead M=65536 N=50304 K=768 bias=0 gelu=0
LibTorch matmul parity dW lmhead: PASS max_abs=0.000000
LibTorch matmul parity dW+accum lmhead: PASS max_abs=0.000000
  dW       Torch C++  22974.24 us
  dW+accum Torch C++  23277.41 us
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
  fwd        Triton   2016.58 us (diff=0.000000, rel=0.000000)
  dInp       Triton   2304.81 us (diff=16.000000, rel=0.002075)
  dW         Triton   2186.00 us (diff=1024.000000, rel=0.003817)
  dW+accum   Triton   2140.17 us (diff=1024.000000, rel=0.003817)
attproj      M=65536 N=768 K=768 bias=1 gelu=0
  fwd        Triton    663.12 us (diff=0.000000, rel=0.000000)
  dInp       Triton    697.50 us (diff=0.000000, rel=0.000000)
  dW         Triton    563.55 us (diff=512.000000, rel=0.001953)
  dW+accum   Triton    563.19 us (diff=0.000000, rel=0.000000)
fc           M=65536 N=3072 K=768 bias=1 gelu=1
  fwd+GeLU   Triton   2677.78 us (diff=0.000000, rel=0.000000)
  dInp       Triton   3038.61 us (diff=0.000000, rel=0.000000)
  dW         Triton   2270.11 us (diff=256.000000, rel=0.002825)
  dW+accum   Triton   2325.06 us (diff=256.000000, rel=0.002825)
fcproj       M=65536 N=768 K=3072 bias=1 gelu=0
  fwd        Triton   3017.29 us (diff=0.000000, rel=0.000000)
  dInp       Triton   2779.39 us (diff=0.000000, rel=0.000000)
  dInp+dGeLU Triton   3317.04 us (diff=32.000000, rel=0.006711)
  dW         Triton   2234.39 us (diff=1024.000000, rel=0.005025)
  dW+accum   Triton   2319.79 us (diff=1024.000000, rel=0.005025)
lmhead       M=65536 N=50304 K=768 bias=0 gelu=0
  fwd        Triton  45589.86 us (diff=0.000000, rel=0.000000)
  dInp       Triton  49117.63 us (diff=512.000000, rel=0.003067)
  dW         Triton  70343.17 us (diff=1024.000000, rel=0.005405)
  dW+accum   Triton  70313.63 us (diff=1024.000000, rel=0.005405)
```

## Torch Attention Benchmarks

```text
Torch Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 552.902 us
Torch Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2200.397 us
TorchPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 1090.702 us
TorchPacked Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 4065.501 us
TorchMaterializedPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 1268.094 us
TorchMaterializedPacked Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 4190.842 us
```

## cuDNN Attention Benchmarks

```text
cuDNN Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 685.163 us (max_diff=0.003906)
cuDNN Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2394.672 us
cuDNNPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 813.355 us
cuDNNPacked Attention Backward route: saved-forward
cuDNNPacked Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2824.602 us
```

## Triton Attention Benchmarks

```text
Triton Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 2069.312 us (diff=0.001953)
TritonPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 2199.115 us (diff=0.003906)
B=64 T=1024 C=768 NH=12 HS=64            | Triton       | unavailable: attention backward is not implemented in this Triton prototype
B=64 T=1024 C=768 NH=12 HS=64            | TritonPacked | unavailable: packed attention backward is not implemented in this Triton prototype
```

## Torch Classifier Benchmarks

```text
Torch classifier device: NVIDIA GeForce RTX 5090; capability=sm_120
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | Torch        | 17858.240 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | Torch        | unavailable: CUDA OOM at full GPT-2 padded-logits shape
```

## Triton Classifier Benchmarks

```text
Triton classifier device: NVIDIA GeForce RTX 5090; capability=sm_120
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | Triton       |  8255.392 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | Triton       | 22262.625 us
```

## Python Stack LayerNorm Benchmarks

```text
Triton LayerNorm device: NVIDIA GeForce RTX 5090; capability=sm_120
Triton LayerNorm Forward (N=65536, C=768): 175.040 us (y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm ForwardNative (N=65536, C=768): 139.712 us (y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm ForwardWithStats (N=65536, C=768): 2183.296 us (y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
Triton LayerNorm BackwardDInput (N=65536, C=768): 224.320 us (dinp_diff=0.015625; dweight/dbias not produced)
Torch LayerNorm BackwardDInputNative (N=65536, C=768): 212.416 us (dinp_diff=0.031250; dweight/dbias not produced)
Torch LayerNorm BackwardDInputNativePlusGrads (N=65536, C=768): 1966.432 us (dinp_diff=0.031250, dweight_diff=1.928711, dbias_diff=1.873962; BF16 dweight/dbias)
Torch LayerNorm BackwardNative (N=65536, C=768): 423.232 us (dinp_diff=0.031250, dweight_diff=1.928711, dbias_diff=1.873962)
Triton LayerNorm BackwardAtomicFP32 (N=65536, C=768): 362.176 us (dinp_diff=0.015625, dweight_diff=0.002136, dbias_diff=0.000427; FP32 dweight/dbias)
Triton LayerNorm FusedResidualForward (N=65536, C=768): 309.088 us (residual_diff=0.000000, y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm FusedResidualForwardNative (N=65536, C=768): 338.080 us (residual_diff=0.000000, y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm FusedResidualForwardWithStats (N=65536, C=768): 3196.512 us (residual_diff=0.000000, y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
Triton LayerNorm Forward (N=65536, C=3072): 571.872 us (y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm ForwardNative (N=65536, C=3072): 549.888 us (y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm ForwardWithStats (N=65536, C=3072): 8999.488 us (y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
Triton LayerNorm BackwardDInput (N=65536, C=3072): 804.736 us (dinp_diff=0.031250; dweight/dbias not produced)
Torch LayerNorm BackwardDInputNative (N=65536, C=3072): 833.440 us (dinp_diff=0.031250; dweight/dbias not produced)
Torch LayerNorm BackwardDInputNativePlusGrads (N=65536, C=3072): 7908.384 us (dinp_diff=0.031250, dweight_diff=1.991577, dbias_diff=1.993774; BF16 dweight/dbias)
Torch LayerNorm BackwardNative (N=65536, C=3072): 1367.776 us (dinp_diff=0.031250, dweight_diff=1.991577, dbias_diff=1.993774)
Triton LayerNorm BackwardAtomicFP32 (N=65536, C=3072): 1414.368 us (dinp_diff=0.031250, dweight_diff=0.003174, dbias_diff=0.000793; FP32 dweight/dbias)
Triton LayerNorm FusedResidualForward (N=65536, C=3072): 1106.656 us (residual_diff=0.000000, y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm FusedResidualForwardNative (N=65536, C=3072): 1312.352 us (residual_diff=0.000000, y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm FusedResidualForwardWithStats (N=65536, C=3072): 12930.016 us (residual_diff=0.000000, y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
```

## Triton Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | Triton       |   132.038 us
bias_add                       | BT=65536 OC=3072             | Triton       |   528.474 us
gelu_forward                   | BT=65536 C=3072              | Triton       |   530.688 us
gelu_backward_inplace          | BT=65536 C=3072              | Triton       |   772.560 us
bias_grad_reduce               | BT=65536 OC=768              | Triton       | unavailable: not implemented in this Triton runtime prototype
bias_grad_reduce               | BT=65536 OC=2304             | Triton       | unavailable: not implemented in this Triton runtime prototype
bias_grad_reduce               | BT=65536 OC=3072             | Triton       | unavailable: not implemented in this Triton runtime prototype
```

## Torch Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | Torch        |   135.624 us
bias_add                       | BT=65536 OC=3072             | Torch        |   549.021 us
gelu_forward                   | BT=65536 C=3072              | Torch        |   530.216 us
gelu_backward_inplace          | BT=65536 C=3072              | Torch        | 26843.265 us
bias_grad_reduce               | BT=65536 OC=768              | Torch        |   317.805 us
bias_grad_reduce               | BT=65536 OC=2304             | Torch        |   967.293 us
bias_grad_reduce               | BT=65536 OC=3072             | Torch        |  1318.944 us
cuda_memset                    | grad_elems=124475904         | Torch        |   147.976 us
global_norm_squared            | params=124475904             | Torch        |  2368.902 us
adamw_update_bf16_state        | params=124475904 no-master   | Torch        |  1220.154 us
adamw_update                   | params=124475904 no-master fp32-state | Torch        |  7272.448 us
encoder_forward                | B=64 T=1024 C=768            | Torch        |   199.842 us
cuda_memset                    | hidden_elems=50331648        | Torch        |    60.022 us
cuda_copy_d2d                  | hidden_elems=50331648        | Torch        |   132.098 us
cuda_memset                    | logits_elems=3296722944      | Torch        |  3949.440 us
cuda_copy_d2d                  | logits_elems=3296722944      | Torch        |  8665.536 us
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
cuda_memset                    | hidden_elems=50331648        | Torch C++    |    59.938 us
cuda_copy_d2d                  | hidden_elems=50331648        | Torch C++    |   131.760 us
cuda_memset                    | grad_elems=124475904         | Torch C++    |   148.203 us
cuda_memset                    | logits_elems=3296722944      | Torch C++    |  3943.584 us
cuda_copy_d2d                  | logits_elems=3296722944      | Torch C++    |  8695.360 us
gelu_forward                   | BT=65536 C=3072              | Torch C++    |   529.755 us
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
missing: scratch/sm120_rounds/codex_sm120_current_optional_refresh_20260522/train_gpt2cu.log
```

## validate_sm120_round
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_current_optional_refresh_20260522 --write-scoreboard scratch/sm120_rounds/codex_sm120_current_optional_refresh_20260522/scoreboard-candidates.md --write-selected-backends scratch/sm120_rounds/codex_sm120_current_optional_refresh_20260522/selected-backends.json --write-promotion-candidates scratch/sm120_rounds/codex_sm120_current_optional_refresh_20260522/promotion-candidates.json --require-manifest --require-stack-probe --require-correctness --require-benchmarks`

