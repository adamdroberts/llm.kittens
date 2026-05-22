# SM120 Optimization Round

- run label: `codex_sm120_round_optional_refresh_correctness_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_optional_refresh_correctness_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_optional_refresh_correctness_20260521`
- max steps: `3`
- python: `/home/adam/miniconda3/envs/llm-kittens/bin/python`
- cuDNN packed backward route: `saved-forward`
- LibTorch runtime route: `cxx-api-raw-pointer`
- LibTorch matmul shapes: `fc`
- git commit: `0f21747`
- working tree: `497` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Thu May 21 09:54:46 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0 Off |                  N/A |
|100%   30C    P8             46W /  575W |     670MiB /  32607MiB |      1%      Default |
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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_round_optional_refresh_correctness_20260521/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_round_optional_refresh_correctness_20260521/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_round_optional_refresh_correctness_20260521/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_round_optional_refresh_correctness_20260521/round-manifest.md --run-label codex_sm120_round_optional_refresh_correctness_20260521 --artifact-dir scratch/sm120_rounds/codex_sm120_round_optional_refresh_correctness_20260521 --train-out-dir log124M/5090_S_codex_sm120_round_optional_refresh_correctness_20260521 --max-steps 3 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 1 --run-benchmarks 1 --run-python-stack-benchmarks 1 --cudnn-packed-backward-route saved-forward --libtorch-runtime-route cxx-api-raw-pointer --run-libtorch-matmul-benchmarks 1 --libtorch-matmul-shapes fc --run-training 0 --keep-checkpoints 0`


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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_libtorch_matmul.py --repeats 7 --large-repeats 3 --warmup 3 --shape fc`


## bench_sm120_cutedsl_matmul
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_cutedsl_matmul.py`


## bench_sm120_triton_matmul
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_matmul.py --repeats 7 --large-repeats 2`


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
  fwd      TK   1073.34 us | cuBLASLt   1089.66 us | cuBLAS   1426.42 us | TK/cuBLASLt 0.99x
  dInp   TK   1092.80 us | cuBLASLt   1014.28 us | cuBLAS   1012.52 us | TK/cuBLASLt 1.08x
  dW     TK   1475.52 us | cuBLASLt   1129.46 us | cuBLAS    997.80 us | TK/cuBLASLt 1.31x
  dW+accum TK   1483.56 us | cuBLASLt   1114.29 us | cuBLAS    999.25 us | TK/cuBLASLt 1.33x
  fwd      TK    376.28 us | cuBLASLt    371.14 us | cuBLAS    484.42 us | TK/cuBLASLt 1.01x
  dInp   TK    381.11 us | cuBLASLt    368.10 us | cuBLAS    365.72 us | TK/cuBLASLt 1.04x
  dW     TK    543.63 us | cuBLASLt    375.76 us | cuBLAS    328.84 us | TK/cuBLASLt 1.45x
  dW+accum TK    547.80 us | cuBLASLt    383.27 us | cuBLAS    336.52 us | TK/cuBLASLt 1.43x
  fwd+GeLU TK fused   1596.22 us | TK explicit   1969.83 us | cuBLASLt   1493.07 us | cuBLAS explicit   2464.02 us | explicit/cuBLASLt 1.32x
  dInp   TK   1463.55 us | cuBLASLt   1361.21 us | cuBLAS   1351.00 us | TK/cuBLASLt 1.08x
  dW     TK   1759.57 us | cuBLASLt   1497.03 us | cuBLAS   1327.30 us | TK/cuBLASLt 1.18x
  dW+accum TK   1759.76 us | cuBLASLt   1491.24 us | cuBLAS   1334.31 us | TK/cuBLASLt 1.18x
  fwd      TK   1434.36 us | cuBLASLt   1364.43 us | cuBLAS   1566.66 us | TK/cuBLASLt 1.05x
  dInp   TK   1491.39 us | cuBLASLt   1402.34 us | cuBLAS   1384.03 us | TK/cuBLASLt 1.06x
  dInp+dGeLU TK   1821.17 us | cuBLASLt fused   1836.88 us | cuBLASLt explicit   2167.99 us | cuBLAS explicit   2188.87 us | explicit/fused 1.18x
  dW     TK   1758.85 us | cuBLASLt   1488.90 us | cuBLAS   1307.97 us | TK/cuBLASLt 1.18x
  dW+accum TK   1743.27 us | cuBLASLt   1510.77 us | cuBLAS   1314.18 us | TK/cuBLASLt 1.15x
  fwd      TK  27873.86 us | cuBLASLt  22354.40 us | cuBLAS  22396.72 us | TK/cuBLASLt 1.25x
  dInp   TK  24037.57 us | cuBLASLt  21851.73 us | cuBLAS  21267.08 us | TK/cuBLASLt 1.10x
  dW     TK  26082.02 us | cuBLASLt  20960.23 us | cuBLAS  21187.56 us | TK/cuBLASLt 1.24x
  dW+accum TK  26197.94 us | cuBLASLt  21055.91 us | cuBLAS  21329.98 us | TK/cuBLASLt 1.24x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 788.745 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2744.542 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 136.413 us
LayerNorm FusedResidualForward (N=65536, C=768): 275.072 us
LayerNorm Backward (N=65536, C=768): 288.164 us
LayerNorm Forward (N=65536, C=3072): 545.169 us
LayerNorm FusedResidualForward (N=65536, C=3072): 1084.119 us
LayerNorm Backward (N=65536, C=3072): 1269.367 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    91.737 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   538.016 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   536.906 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   790.779 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    24.811 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   186.560 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   245.488 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  3999.507 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  8916.838 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  3955.744 us
cuda_memset                    | logits_elems=3296722944      | CUDA kernel  |  4123.994 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  8778.042 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA kernel  |  9208.922 us
global_norm_squared            | params=124475904             | CUDA         |   185.061 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1808.269 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    60.938 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    63.555 us
cuda_memset                    | hidden_elems=50331648        | CUDA kernel  |    63.484 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   131.677 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA kernel  |   138.511 us
```

## Torch Matmul Benchmarks

```text
Torch matmul device: NVIDIA GeForce RTX 5090; capability=sm_120
qkv          M=65536 N=2304 K=768 bias=1 gelu=0
  fwd      Torch   1443.42 us
  dInp   Torch   1017.91 us
  dW     Torch   1026.46 us
  dW+accum Torch   1015.64 us
attproj      M=65536 N=768 K=768 bias=1 gelu=0
  fwd      Torch    517.53 us
  dInp   Torch    371.86 us
  dW     Torch    338.66 us
  dW+accum Torch    346.85 us
fc           M=65536 N=3072 K=768 bias=1 gelu=1
  fwd+GeLU Torch   2451.92 us
  dInp   Torch   1366.60 us
  dW     Torch   1345.68 us
  dW+accum Torch   1366.92 us
fcproj       M=65536 N=768 K=3072 bias=1 gelu=0
  fwd      Torch   1599.67 us
  dInp   Torch   1400.44 us
  dInp+dGeLU Torch  28252.24 us
  dW     Torch   1408.28 us
  dW+accum Torch   1402.25 us
lmhead       M=65536 N=50304 K=768 bias=0 gelu=0
  fwd      Torch  22929.60 us
  dInp   Torch  21678.18 us
  dW     Torch  21303.65 us
  dW+accum Torch  21593.06 us
```

## LibTorch C++ Matmul Benchmarks

```text
LibTorch matmul device: NVIDIA GeForce RTX 5090; capability=sm_120
LibTorch matmul route: standalone C++ API cached from_blob handles over existing CUDA pointers
fc           M=65536 N=3072 K=768 bias=1 gelu=1
LibTorch matmul parity dW fc: PASS max_abs=0.000000
LibTorch matmul parity dW+accum fc: PASS max_abs=0.000000
  dW       Torch C++   1365.35 us
  dW+accum Torch C++   1372.57 us
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
  fwd        Triton   1952.47 us (diff=0.000000, rel=0.000000)
  dInp       Triton   2326.80 us (diff=16.000000, rel=0.002075)
  dW         Triton   2139.45 us (diff=1024.000000, rel=0.003817)
  dW+accum   Triton   2179.92 us (diff=1024.000000, rel=0.003817)
attproj      M=65536 N=768 K=768 bias=1 gelu=0
  fwd        Triton    663.10 us (diff=0.000000, rel=0.000000)
  dInp       Triton    657.10 us (diff=0.000000, rel=0.000000)
  dW         Triton    560.02 us (diff=512.000000, rel=0.001953)
  dW+accum   Triton    563.11 us (diff=0.000000, rel=0.000000)
fc           M=65536 N=3072 K=768 bias=1 gelu=1
  fwd+GeLU   Triton   2615.97 us (diff=0.000000, rel=0.000000)
  dInp       Triton   3078.47 us (diff=0.000000, rel=0.000000)
  dW         Triton   2271.58 us (diff=256.000000, rel=0.002825)
  dW+accum   Triton   2270.00 us (diff=256.000000, rel=0.002825)
fcproj       M=65536 N=768 K=3072 bias=1 gelu=0
  fwd        Triton   2976.16 us (diff=0.000000, rel=0.000000)
  dInp       Triton   2809.24 us (diff=0.000000, rel=0.000000)
  dInp+dGeLU Triton   3373.22 us (diff=32.000000, rel=0.006711)
  dW         Triton   2266.54 us (diff=1024.000000, rel=0.005025)
  dW+accum   Triton   2238.59 us (diff=1024.000000, rel=0.005025)
lmhead       M=65536 N=50304 K=768 bias=0 gelu=0
  fwd        Triton  44010.74 us (diff=0.000000, rel=0.000000)
  dInp       Triton  49021.49 us (diff=512.000000, rel=0.003067)
  dW         Triton  70116.51 us (diff=1024.000000, rel=0.005405)
  dW+accum   Triton  70271.55 us (diff=1024.000000, rel=0.005405)
```

## Torch Attention Benchmarks

```text
Torch Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 551.350 us
Torch Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2201.862 us
TorchPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 1090.229 us
TorchPacked Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 4074.912 us
TorchMaterializedPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 1260.011 us
TorchMaterializedPacked Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 4190.883 us
```

## cuDNN Attention Benchmarks

```text
cuDNN Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 694.430 us (max_diff=0.003906)
cuDNN Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2371.510 us
cuDNNPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 809.363 us
cuDNNPacked Attention Backward route: saved-forward
cuDNNPacked Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2818.074 us
```

## Triton Attention Benchmarks

```text
Triton Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 2070.654 us (diff=0.001953)
TritonPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 2199.478 us (diff=0.003906)
B=64 T=1024 C=768 NH=12 HS=64            | Triton       | unavailable: attention backward is not implemented in this Triton prototype
B=64 T=1024 C=768 NH=12 HS=64            | TritonPacked | unavailable: packed attention backward is not implemented in this Triton prototype
```

## Torch Classifier Benchmarks

```text
Torch classifier device: NVIDIA GeForce RTX 5090; capability=sm_120
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | Torch        | 17874.144 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | Torch        | 35105.568 us
```

## Triton Classifier Benchmarks

```text
Triton classifier device: NVIDIA GeForce RTX 5090; capability=sm_120
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | Triton       |  8195.616 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | Triton       | 22188.513 us
```

## Python Stack LayerNorm Benchmarks

```text
Triton LayerNorm device: NVIDIA GeForce RTX 5090; capability=sm_120
Triton LayerNorm Forward (N=65536, C=768): 176.704 us (y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm ForwardNative (N=65536, C=768): 155.104 us (y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm ForwardWithStats (N=65536, C=768): 2189.696 us (y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
Triton LayerNorm BackwardDInput (N=65536, C=768): 228.640 us (dinp_diff=0.015625; dweight/dbias not produced)
Torch LayerNorm BackwardDInputNative (N=65536, C=768): 220.352 us (dinp_diff=0.031250; dweight/dbias not produced)
Torch LayerNorm BackwardDInputNativePlusGrads (N=65536, C=768): 1968.192 us (dinp_diff=0.031250, dweight_diff=1.928711, dbias_diff=1.873962; BF16 dweight/dbias)
Torch LayerNorm BackwardNative (N=65536, C=768): 413.440 us (dinp_diff=0.031250, dweight_diff=1.928711, dbias_diff=1.873962)
Triton LayerNorm BackwardAtomicFP32 (N=65536, C=768): 364.320 us (dinp_diff=0.015625, dweight_diff=0.002014, dbias_diff=0.000549; FP32 dweight/dbias)
Triton LayerNorm FusedResidualForward (N=65536, C=768): 309.824 us (residual_diff=0.000000, y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm FusedResidualForwardNative (N=65536, C=768): 331.040 us (residual_diff=0.000000, y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm FusedResidualForwardWithStats (N=65536, C=768): 3187.104 us (residual_diff=0.000000, y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
Triton LayerNorm Forward (N=65536, C=3072): 569.056 us (y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm ForwardNative (N=65536, C=3072): 539.616 us (y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm ForwardWithStats (N=65536, C=3072): 8964.320 us (y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
Triton LayerNorm BackwardDInput (N=65536, C=3072): 797.120 us (dinp_diff=0.031250; dweight/dbias not produced)
Torch LayerNorm BackwardDInputNative (N=65536, C=3072): 826.528 us (dinp_diff=0.031250; dweight/dbias not produced)
Torch LayerNorm BackwardDInputNativePlusGrads (N=65536, C=3072): 8122.912 us (dinp_diff=0.031250, dweight_diff=1.991577, dbias_diff=1.993774; BF16 dweight/dbias)
Torch LayerNorm BackwardNative (N=65536, C=3072): 1385.376 us (dinp_diff=0.031250, dweight_diff=1.991577, dbias_diff=1.993774)
Triton LayerNorm BackwardAtomicFP32 (N=65536, C=3072): 1416.736 us (dinp_diff=0.031250, dweight_diff=0.002808, dbias_diff=0.000671; FP32 dweight/dbias)
Triton LayerNorm FusedResidualForward (N=65536, C=3072): 1101.376 us (residual_diff=0.000000, y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm FusedResidualForwardNative (N=65536, C=3072): 1307.456 us (residual_diff=0.000000, y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm FusedResidualForwardWithStats (N=65536, C=3072): 12952.704 us (residual_diff=0.000000, y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
```

## Triton Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | Triton       |   132.579 us
bias_add                       | BT=65536 OC=3072             | Triton       |   529.637 us
gelu_forward                   | BT=65536 C=3072              | Triton       |   529.843 us
gelu_backward_inplace          | BT=65536 C=3072              | Triton       |   777.470 us
bias_grad_reduce               | BT=65536 OC=768              | Triton       | unavailable: not implemented in this Triton runtime prototype
bias_grad_reduce               | BT=65536 OC=2304             | Triton       | unavailable: not implemented in this Triton runtime prototype
bias_grad_reduce               | BT=65536 OC=3072             | Triton       | unavailable: not implemented in this Triton runtime prototype
```

## Torch Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | Torch        |   135.996 us
bias_add                       | BT=65536 OC=3072             | Torch        |   540.000 us
gelu_forward                   | BT=65536 C=3072              | Torch        |   529.467 us
gelu_backward_inplace          | BT=65536 C=3072              | Torch        | 26842.850 us
bias_grad_reduce               | BT=65536 OC=768              | Torch        |   312.877 us
bias_grad_reduce               | BT=65536 OC=2304             | Torch        |   969.411 us
bias_grad_reduce               | BT=65536 OC=3072             | Torch        |  1300.707 us
global_norm_squared            | params=124475904             | Torch        |  2258.950 us
adamw_update_bf16_state        | params=124475904 no-master   | Torch        |  1209.933 us
adamw_update                   | params=124475904 no-master fp32-state | Torch        |  7264.192 us
encoder_forward                | B=64 T=1024 C=768            | Torch        |   200.806 us
cuda_memset                    | hidden_elems=50331648        | Torch        |    59.981 us
cuda_copy_d2d                  | hidden_elems=50331648        | Torch        |   131.722 us
cuda_memset                    | logits_elems=3296722944      | Torch        |  3941.984 us
cuda_copy_d2d                  | logits_elems=3296722944      | Torch        |  8707.392 us
```

## LibTorch C++ Runtime Benchmarks

```text
LibTorch runtime device: NVIDIA GeForce RTX 5090; capability=sm_120
LibTorch runtime route: standalone C++ API cached from_blob handles over existing CUDA pointers
LibTorch parity cuda_memset hidden_elems=50331648: PASS
LibTorch parity cuda_copy_d2d hidden_elems=50331648: PASS
LibTorch parity cuda_memset logits_elems=3296722944: PASS
LibTorch parity cuda_copy_d2d logits_elems=3296722944: PASS
cuda_memset                    | hidden_elems=50331648        | Torch C++    |    60.000 us
cuda_copy_d2d                  | hidden_elems=50331648        | Torch C++    |   131.867 us
cuda_memset                    | logits_elems=3296722944      | Torch C++    |  3923.616 us
cuda_copy_d2d                  | logits_elems=3296722944      | Torch C++    |  8669.632 us
```

## Training Steps

```text
missing: scratch/sm120_rounds/codex_sm120_round_optional_refresh_correctness_20260521/train_gpt2cu.log
```

## validate_sm120_round
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_round_optional_refresh_correctness_20260521 --write-scoreboard scratch/sm120_rounds/codex_sm120_round_optional_refresh_correctness_20260521/scoreboard-candidates.md --write-selected-backends scratch/sm120_rounds/codex_sm120_round_optional_refresh_correctness_20260521/selected-backends.json --write-promotion-candidates scratch/sm120_rounds/codex_sm120_round_optional_refresh_correctness_20260521/promotion-candidates.json --require-manifest --require-stack-probe --require-correctness --require-benchmarks`

