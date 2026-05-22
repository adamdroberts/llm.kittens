# SM120 Optimization Round

- run label: `codex_sm120_round_optional_refresh_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_optional_refresh_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_optional_refresh_20260521`
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
Thu May 21 09:48:36 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0 Off |                  N/A |
|100%   30C    P8             47W /  575W |    1171MiB /  32607MiB |      1%      Default |
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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_round_optional_refresh_20260521/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_round_optional_refresh_20260521/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_round_optional_refresh_20260521/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_round_optional_refresh_20260521/round-manifest.md --run-label codex_sm120_round_optional_refresh_20260521 --artifact-dir scratch/sm120_rounds/codex_sm120_round_optional_refresh_20260521 --train-out-dir log124M/5090_S_codex_sm120_round_optional_refresh_20260521 --max-steps 3 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 0 --run-benchmarks 1 --run-python-stack-benchmarks 1 --cudnn-packed-backward-route saved-forward --libtorch-runtime-route cxx-api-raw-pointer --run-libtorch-matmul-benchmarks 1 --libtorch-matmul-shapes fc --run-training 0 --keep-checkpoints 0`


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


## Matmul Benchmarks

```text
  fwd      TK   1096.17 us | cuBLASLt   1041.66 us | cuBLAS   1460.60 us | TK/cuBLASLt 1.05x
  dInp   TK   1093.14 us | cuBLASLt   1021.95 us | cuBLAS   1012.44 us | TK/cuBLASLt 1.07x
  dW     TK   1467.87 us | cuBLASLt   1114.81 us | cuBLAS    997.28 us | TK/cuBLASLt 1.32x
  dW+accum TK   1463.75 us | cuBLASLt   1130.74 us | cuBLAS    998.52 us | TK/cuBLASLt 1.29x
  fwd      TK    376.43 us | cuBLASLt    371.91 us | cuBLAS    483.55 us | TK/cuBLASLt 1.01x
  dInp   TK    381.19 us | cuBLASLt    367.38 us | cuBLAS    365.93 us | TK/cuBLASLt 1.04x
  dW     TK    565.64 us | cuBLASLt    375.78 us | cuBLAS    328.78 us | TK/cuBLASLt 1.51x
  dW+accum TK    544.37 us | cuBLASLt    378.92 us | cuBLAS    332.65 us | TK/cuBLASLt 1.44x
  fwd+GeLU TK fused   1563.54 us | TK explicit   1997.38 us | cuBLASLt   1471.13 us | cuBLAS explicit   2476.11 us | explicit/cuBLASLt 1.36x
  dInp   TK   1464.50 us | cuBLASLt   1384.88 us | cuBLAS   1364.90 us | TK/cuBLASLt 1.06x
  dW     TK   1746.17 us | cuBLASLt   1493.79 us | cuBLAS   1317.02 us | TK/cuBLASLt 1.17x
  dW+accum TK   1755.44 us | cuBLASLt   1499.30 us | cuBLAS   1315.25 us | TK/cuBLASLt 1.17x
  fwd      TK   1434.32 us | cuBLASLt   1379.88 us | cuBLAS   1555.30 us | TK/cuBLASLt 1.04x
  dInp   TK   1540.89 us | cuBLASLt   1386.98 us | cuBLAS   1380.01 us | TK/cuBLASLt 1.11x
  dInp+dGeLU TK   1822.44 us | cuBLASLt fused   1828.25 us | cuBLASLt explicit   2193.04 us | cuBLAS explicit   2187.03 us | explicit/fused 1.20x
  dW     TK   1748.65 us | cuBLASLt   1485.43 us | cuBLAS   1311.44 us | TK/cuBLASLt 1.18x
  dW+accum TK   1744.18 us | cuBLASLt   1496.18 us | cuBLAS   1318.57 us | TK/cuBLASLt 1.17x
  fwd      TK  27869.09 us | cuBLASLt  22396.62 us | cuBLAS  22411.32 us | TK/cuBLASLt 1.24x
  dInp   TK  24022.46 us | cuBLASLt  21792.80 us | cuBLAS  21272.17 us | TK/cuBLASLt 1.10x
  dW     TK  26145.54 us | cuBLASLt  20899.63 us | cuBLAS  21258.15 us | TK/cuBLASLt 1.25x
  dW+accum TK  26120.88 us | cuBLASLt  20986.71 us | cuBLAS  21346.56 us | TK/cuBLASLt 1.24x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 785.718 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2743.201 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 138.406 us
LayerNorm FusedResidualForward (N=65536, C=768): 275.342 us
LayerNorm Backward (N=65536, C=768): 288.147 us
LayerNorm Forward (N=65536, C=3072): 544.301 us
LayerNorm FusedResidualForward (N=65536, C=3072): 1082.597 us
LayerNorm Backward (N=65536, C=3072): 1272.855 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    80.025 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   548.129 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   528.059 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   791.201 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    24.630 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   186.802 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   245.528 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  3953.133 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  8942.509 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  3958.458 us
cuda_memset                    | logits_elems=3296722944      | CUDA kernel  |  4115.859 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  8777.522 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA kernel  |  9188.391 us
global_norm_squared            | params=124475904             | CUDA         |   185.069 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1809.206 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    79.286 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    60.164 us
cuda_memset                    | hidden_elems=50331648        | CUDA kernel  |    61.882 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   131.615 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA kernel  |   137.148 us
```

## Torch Matmul Benchmarks

```text
Torch matmul device: NVIDIA GeForce RTX 5090; capability=sm_120
qkv          M=65536 N=2304 K=768 bias=1 gelu=0
  fwd      Torch   1455.56 us
  dInp   Torch   1027.64 us
  dW     Torch   1011.02 us
  dW+accum Torch   1017.40 us
attproj      M=65536 N=768 K=768 bias=1 gelu=0
  fwd      Torch    516.91 us
  dInp   Torch    372.55 us
  dW     Torch    338.82 us
  dW+accum Torch    345.34 us
fc           M=65536 N=3072 K=768 bias=1 gelu=1
  fwd+GeLU Torch   2442.02 us
  dInp   Torch   1383.70 us
  dW     Torch   1352.33 us
  dW+accum Torch   1376.64 us
fcproj       M=65536 N=768 K=3072 bias=1 gelu=0
  fwd      Torch   1601.29 us
  dInp   Torch   1399.88 us
  dInp+dGeLU Torch  28230.19 us
  dW     Torch   1368.80 us
  dW+accum Torch   1411.15 us
lmhead       M=65536 N=50304 K=768 bias=0 gelu=0
  fwd      Torch  22768.19 us
  dInp   Torch  21444.64 us
  dW     Torch  21808.77 us
  dW+accum Torch  21527.58 us
```

## LibTorch C++ Matmul Benchmarks

```text
LibTorch matmul device: NVIDIA GeForce RTX 5090; capability=sm_120
LibTorch matmul route: standalone C++ API cached from_blob handles over existing CUDA pointers
fc           M=65536 N=3072 K=768 bias=1 gelu=1
LibTorch matmul parity dW fc: PASS max_abs=0.000000
LibTorch matmul parity dW+accum fc: PASS max_abs=0.000000
  dW       Torch C++   1353.16 us
  dW+accum Torch C++   1367.16 us
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
  fwd        Triton   2024.95 us (diff=0.000000, rel=0.000000)
  dInp       Triton   2281.49 us (diff=16.000000, rel=0.002075)
  dW         Triton   2123.94 us (diff=1024.000000, rel=0.003817)
  dW+accum   Triton   2142.69 us (diff=1024.000000, rel=0.003817)
attproj      M=65536 N=768 K=768 bias=1 gelu=0
  fwd        Triton    662.04 us (diff=0.000000, rel=0.000000)
  dInp       Triton    679.12 us (diff=0.000000, rel=0.000000)
  dW         Triton    558.29 us (diff=512.000000, rel=0.001953)
  dW+accum   Triton    561.24 us (diff=0.000000, rel=0.000000)
fc           M=65536 N=3072 K=768 bias=1 gelu=1
  fwd+GeLU   Triton   2601.84 us (diff=0.000000, rel=0.000000)
  dInp       Triton   3026.49 us (diff=0.000000, rel=0.000000)
  dW         Triton   2240.22 us (diff=256.000000, rel=0.002825)
  dW+accum   Triton   2251.60 us (diff=256.000000, rel=0.002825)
fcproj       M=65536 N=768 K=3072 bias=1 gelu=0
  fwd        Triton   3065.93 us (diff=0.000000, rel=0.000000)
  dInp       Triton   2661.39 us (diff=0.000000, rel=0.000000)
  dInp+dGeLU Triton   3202.09 us (diff=32.000000, rel=0.006711)
  dW         Triton   2259.92 us (diff=1024.000000, rel=0.005025)
  dW+accum   Triton   2257.02 us (diff=1024.000000, rel=0.005025)
lmhead       M=65536 N=50304 K=768 bias=0 gelu=0
  fwd        Triton  44015.87 us (diff=0.000000, rel=0.000000)
  dInp       Triton  49289.26 us (diff=512.000000, rel=0.003067)
  dW         Triton  69907.49 us (diff=1024.000000, rel=0.005405)
  dW+accum   Triton  70181.79 us (diff=1024.000000, rel=0.005405)
```

## Torch Attention Benchmarks

```text
Torch Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 556.573 us
Torch Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2196.240 us
TorchPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 1149.664 us
TorchPacked Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 4050.301 us
TorchMaterializedPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 1262.984 us
TorchMaterializedPacked Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 4215.184 us
```

## cuDNN Attention Benchmarks

```text
cuDNN Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 675.635 us (max_diff=0.003906)
cuDNN Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2404.397 us
cuDNNPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 805.997 us
cuDNNPacked Attention Backward route: saved-forward
cuDNNPacked Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2819.398 us
```

## Triton Attention Benchmarks

```text
Triton Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 2035.579 us (diff=0.001953)
TritonPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 2205.301 us (diff=0.003906)
B=64 T=1024 C=768 NH=12 HS=64            | Triton       | unavailable: attention backward is not implemented in this Triton prototype
B=64 T=1024 C=768 NH=12 HS=64            | TritonPacked | unavailable: packed attention backward is not implemented in this Triton prototype
```

## Torch Classifier Benchmarks

```text
Torch classifier device: NVIDIA GeForce RTX 5090; capability=sm_120
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | Torch        | 17713.633 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | Torch        | unavailable: CUDA OOM at full GPT-2 padded-logits shape
```

## Triton Classifier Benchmarks

```text
Triton classifier device: NVIDIA GeForce RTX 5090; capability=sm_120
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | Triton       |  8246.336 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | Triton       | 22228.512 us
```

## Python Stack LayerNorm Benchmarks

```text
Triton LayerNorm device: NVIDIA GeForce RTX 5090; capability=sm_120
Triton LayerNorm Forward (N=65536, C=768): 176.384 us (y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm ForwardNative (N=65536, C=768): 154.944 us (y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm ForwardWithStats (N=65536, C=768): 2181.408 us (y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
Triton LayerNorm BackwardDInput (N=65536, C=768): 223.328 us (dinp_diff=0.015625; dweight/dbias not produced)
Torch LayerNorm BackwardDInputNative (N=65536, C=768): 214.592 us (dinp_diff=0.031250; dweight/dbias not produced)
Torch LayerNorm BackwardDInputNativePlusGrads (N=65536, C=768): 1959.360 us (dinp_diff=0.031250, dweight_diff=1.928711, dbias_diff=1.873962; BF16 dweight/dbias)
Torch LayerNorm BackwardNative (N=65536, C=768): 415.360 us (dinp_diff=0.031250, dweight_diff=1.928711, dbias_diff=1.873962)
Triton LayerNorm BackwardAtomicFP32 (N=65536, C=768): 364.992 us (dinp_diff=0.015625, dweight_diff=0.002014, dbias_diff=0.000610; FP32 dweight/dbias)
Triton LayerNorm FusedResidualForward (N=65536, C=768): 322.912 us (residual_diff=0.000000, y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm FusedResidualForwardNative (N=65536, C=768): 331.648 us (residual_diff=0.000000, y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm FusedResidualForwardWithStats (N=65536, C=768): 3180.320 us (residual_diff=0.000000, y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
Triton LayerNorm Forward (N=65536, C=3072): 571.648 us (y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm ForwardNative (N=65536, C=3072): 547.456 us (y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm ForwardWithStats (N=65536, C=3072): 8947.072 us (y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
Triton LayerNorm BackwardDInput (N=65536, C=3072): 801.056 us (dinp_diff=0.031250; dweight/dbias not produced)
Torch LayerNorm BackwardDInputNative (N=65536, C=3072): 819.136 us (dinp_diff=0.031250; dweight/dbias not produced)
Torch LayerNorm BackwardDInputNativePlusGrads (N=65536, C=3072): 7936.800 us (dinp_diff=0.031250, dweight_diff=1.991577, dbias_diff=1.993774; BF16 dweight/dbias)
Torch LayerNorm BackwardNative (N=65536, C=3072): 1383.520 us (dinp_diff=0.031250, dweight_diff=1.991577, dbias_diff=1.993774)
Triton LayerNorm BackwardAtomicFP32 (N=65536, C=3072): 1418.048 us (dinp_diff=0.031250, dweight_diff=0.003296, dbias_diff=0.000549; FP32 dweight/dbias)
Triton LayerNorm FusedResidualForward (N=65536, C=3072): 1106.144 us (residual_diff=0.000000, y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm FusedResidualForwardNative (N=65536, C=3072): 1305.024 us (residual_diff=0.000000, y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm FusedResidualForwardWithStats (N=65536, C=3072): 13156.224 us (residual_diff=0.000000, y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
```

## Triton Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | Triton       |   133.016 us
bias_add                       | BT=65536 OC=3072             | Triton       |   542.899 us
gelu_forward                   | BT=65536 C=3072              | Triton       |   529.186 us
gelu_backward_inplace          | BT=65536 C=3072              | Triton       |   781.888 us
bias_grad_reduce               | BT=65536 OC=768              | Triton       | unavailable: not implemented in this Triton runtime prototype
bias_grad_reduce               | BT=65536 OC=2304             | Triton       | unavailable: not implemented in this Triton runtime prototype
bias_grad_reduce               | BT=65536 OC=3072             | Triton       | unavailable: not implemented in this Triton runtime prototype
```

## Torch Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | Torch        |   135.256 us
bias_add                       | BT=65536 OC=3072             | Torch        |   548.820 us
gelu_forward                   | BT=65536 C=3072              | Torch        |   539.764 us
gelu_backward_inplace          | BT=65536 C=3072              | Torch        | 26825.635 us
bias_grad_reduce               | BT=65536 OC=768              | Torch        |   320.944 us
bias_grad_reduce               | BT=65536 OC=2304             | Torch        |   971.398 us
bias_grad_reduce               | BT=65536 OC=3072             | Torch        |  1305.440 us
global_norm_squared            | params=124475904             | Torch        |  2262.349 us
adamw_update_bf16_state        | params=124475904 no-master   | Torch        |  1208.083 us
adamw_update                   | params=124475904 no-master fp32-state | Torch        |  7298.912 us
encoder_forward                | B=64 T=1024 C=768            | Torch        |   201.555 us
cuda_memset                    | hidden_elems=50331648        | Torch        |    59.912 us
cuda_copy_d2d                  | hidden_elems=50331648        | Torch        |   132.043 us
cuda_memset                    | logits_elems=3296722944      | Torch        |  3943.104 us
cuda_copy_d2d                  | logits_elems=3296722944      | Torch        |  8662.848 us
```

## LibTorch C++ Runtime Benchmarks

```text
LibTorch runtime device: NVIDIA GeForce RTX 5090; capability=sm_120
LibTorch runtime route: standalone C++ API cached from_blob handles over existing CUDA pointers
LibTorch parity cuda_memset hidden_elems=50331648: PASS
LibTorch parity cuda_copy_d2d hidden_elems=50331648: PASS
LibTorch parity cuda_memset logits_elems=3296722944: PASS
LibTorch parity cuda_copy_d2d logits_elems=3296722944: PASS
cuda_memset                    | hidden_elems=50331648        | Torch C++    |    59.875 us
cuda_copy_d2d                  | hidden_elems=50331648        | Torch C++    |   131.805 us
cuda_memset                    | logits_elems=3296722944      | Torch C++    |  3952.768 us
cuda_copy_d2d                  | logits_elems=3296722944      | Torch C++    |  8681.632 us
```

## Training Steps

```text
missing: scratch/sm120_rounds/codex_sm120_round_optional_refresh_20260521/train_gpt2cu.log
```

## validate_sm120_round
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_round_optional_refresh_20260521 --write-scoreboard scratch/sm120_rounds/codex_sm120_round_optional_refresh_20260521/scoreboard-candidates.md --write-selected-backends scratch/sm120_rounds/codex_sm120_round_optional_refresh_20260521/selected-backends.json --write-promotion-candidates scratch/sm120_rounds/codex_sm120_round_optional_refresh_20260521/promotion-candidates.json --require-manifest --require-stack-probe --require-benchmarks`

