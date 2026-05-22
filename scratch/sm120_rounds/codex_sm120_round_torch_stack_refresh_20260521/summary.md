# SM120 Optimization Round

- run label: `codex_sm120_round_torch_stack_refresh_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_torch_stack_refresh_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_torch_stack_refresh_20260521`
- max steps: `3`
- python: `/home/adam/miniconda3/envs/llm-kittens/bin/python`
- git commit: `0f21747`
- working tree: `482` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Thu May 21 01:33:22 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0 Off |                  N/A |
|100%   29C    P8             45W /  575W |    2658MiB /  32607MiB |      1%      Default |
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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_round_torch_stack_refresh_20260521/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_round_torch_stack_refresh_20260521/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_round_torch_stack_refresh_20260521/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_round_torch_stack_refresh_20260521/round-manifest.md --run-label codex_sm120_round_torch_stack_refresh_20260521 --artifact-dir scratch/sm120_rounds/codex_sm120_round_torch_stack_refresh_20260521 --train-out-dir log124M/5090_S_codex_sm120_round_torch_stack_refresh_20260521 --max-steps 3 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 1 --run-benchmarks 1 --run-python-stack-benchmarks 1 --run-training 0 --keep-checkpoints 0`


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
  fwd      TK   1068.68 us | cuBLASLt   1044.88 us | cuBLAS   1451.12 us | TK/cuBLASLt 1.02x
  dInp   TK   1088.51 us | cuBLASLt   1029.35 us | cuBLAS   1044.07 us | TK/cuBLASLt 1.06x
  dW     TK   1484.28 us | cuBLASLt   1112.62 us | cuBLAS    997.61 us | TK/cuBLASLt 1.33x
  dW+accum TK   1511.06 us | cuBLASLt   1116.57 us | cuBLAS   1000.58 us | TK/cuBLASLt 1.35x
  fwd      TK    376.40 us | cuBLASLt    393.06 us | cuBLAS    486.06 us | TK/cuBLASLt 0.96x
  dInp   TK    381.24 us | cuBLASLt    367.16 us | cuBLAS    365.28 us | TK/cuBLASLt 1.04x
  dW     TK    544.62 us | cuBLASLt    380.92 us | cuBLAS    328.25 us | TK/cuBLASLt 1.43x
  dW+accum TK    573.89 us | cuBLASLt    381.67 us | cuBLAS    334.40 us | TK/cuBLASLt 1.50x
  fwd+GeLU TK fused   1567.34 us | TK explicit   1956.01 us | cuBLASLt   1511.86 us | cuBLAS explicit   2521.48 us | explicit/cuBLASLt 1.29x
  dInp   TK   1486.86 us | cuBLASLt   1386.39 us | cuBLAS   1363.76 us | TK/cuBLASLt 1.07x
  dW     TK   1746.91 us | cuBLASLt   1523.08 us | cuBLAS   1353.38 us | TK/cuBLASLt 1.15x
  dW+accum TK   1741.71 us | cuBLASLt   1554.31 us | cuBLAS   1313.96 us | TK/cuBLASLt 1.12x
  fwd      TK   1457.28 us | cuBLASLt   1389.99 us | cuBLAS   1584.20 us | TK/cuBLASLt 1.05x
  dInp   TK   1566.32 us | cuBLASLt   1375.44 us | cuBLAS   1392.83 us | TK/cuBLASLt 1.14x
  dInp+dGeLU TK   1771.14 us | cuBLASLt fused   1820.23 us | cuBLASLt explicit   2173.55 us | cuBLAS explicit   2191.93 us | explicit/fused 1.19x
  dW     TK   1745.89 us | cuBLASLt   1515.53 us | cuBLAS   1338.06 us | TK/cuBLASLt 1.15x
  dW+accum TK   1757.29 us | cuBLASLt   1561.39 us | cuBLAS   1351.94 us | TK/cuBLASLt 1.13x
  fwd      TK  28150.22 us | cuBLASLt  22533.93 us | cuBLAS  22524.04 us | TK/cuBLASLt 1.25x
  dInp   TK  24395.39 us | cuBLASLt  21823.83 us | cuBLAS  21346.29 us | TK/cuBLASLt 1.12x
  dW     TK  26258.10 us | cuBLASLt  21177.73 us | cuBLAS  21566.84 us | TK/cuBLASLt 1.24x
  dW+accum TK  26311.21 us | cuBLASLt  21153.99 us | cuBLAS  21262.23 us | TK/cuBLASLt 1.24x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 792.863 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2753.120 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 142.649 us
LayerNorm FusedResidualForward (N=65536, C=768): 285.402 us
LayerNorm Backward (N=65536, C=768): 287.972 us
LayerNorm Forward (N=65536, C=3072): 563.255 us
LayerNorm FusedResidualForward (N=65536, C=3072): 1112.549 us
LayerNorm Backward (N=65536, C=3072): 1302.031 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    88.514 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   537.093 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   535.954 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   795.119 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    24.502 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   187.910 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   247.688 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  4064.608 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  9121.119 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  4312.973 us
cuda_memset                    | logits_elems=3296722944      | CUDA kernel  |  4434.816 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  8992.870 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA kernel  |  9604.640 us
global_norm_squared            | params=124475904             | CUDA         |   186.109 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1893.056 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    86.424 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    63.850 us
cuda_memset                    | hidden_elems=50331648        | CUDA kernel  |    66.008 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   133.843 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA kernel  |   139.068 us
```

## Torch Matmul Benchmarks

```text
Torch matmul device: NVIDIA GeForce RTX 5090; capability=sm_120
qkv          M=65536 N=2304 K=768 bias=1 gelu=0
  fwd      Torch   1452.33 us
  dInp   Torch   1038.48 us
  dW     Torch   1008.19 us
  dW+accum Torch   1023.40 us
attproj      M=65536 N=768 K=768 bias=1 gelu=0
  fwd      Torch    522.03 us
  dInp   Torch    372.38 us
  dW     Torch    339.98 us
  dW+accum Torch    346.45 us
fc           M=65536 N=3072 K=768 bias=1 gelu=1
  fwd+GeLU Torch   2459.78 us
  dInp   Torch   1346.50 us
  dW     Torch   1333.13 us
  dW+accum Torch   1354.60 us
fcproj       M=65536 N=768 K=3072 bias=1 gelu=0
  fwd      Torch   1590.64 us
  dInp   Torch   1398.87 us
  dInp+dGeLU Torch  29017.12 us
  dW     Torch   1368.92 us
  dW+accum Torch   1472.17 us
lmhead       M=65536 N=50304 K=768 bias=0 gelu=0
  fwd      Torch  22579.46 us
  dInp   Torch  21820.54 us
  dW     Torch  21888.16 us
  dW+accum Torch  21417.12 us
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
  fwd        Triton   2034.80 us (diff=0.000000, rel=0.000000)
  dInp       Triton   2353.11 us (diff=16.000000, rel=0.002075)
  dW         Triton   2187.09 us (diff=1024.000000, rel=0.003817)
  dW+accum   Triton   2139.88 us (diff=1024.000000, rel=0.003817)
attproj      M=65536 N=768 K=768 bias=1 gelu=0
  fwd        Triton    663.14 us (diff=0.000000, rel=0.000000)
  dInp       Triton    684.10 us (diff=0.000000, rel=0.000000)
  dW         Triton    562.16 us (diff=512.000000, rel=0.001953)
  dW+accum   Triton    562.57 us (diff=0.000000, rel=0.000000)
fc           M=65536 N=3072 K=768 bias=1 gelu=1
  fwd+GeLU   Triton   2646.04 us (diff=0.000000, rel=0.000000)
  dInp       Triton   3098.32 us (diff=0.000000, rel=0.000000)
  dW         Triton   2318.68 us (diff=256.000000, rel=0.002825)
  dW+accum   Triton   2364.97 us (diff=256.000000, rel=0.002825)
fcproj       M=65536 N=768 K=3072 bias=1 gelu=0
  fwd        Triton   3006.09 us (diff=0.000000, rel=0.000000)
  dInp       Triton   2827.80 us (diff=0.000000, rel=0.000000)
  dInp+dGeLU Triton   3371.89 us (diff=32.000000, rel=0.006711)
  dW         Triton   2305.33 us (diff=1024.000000, rel=0.005025)
  dW+accum   Triton   2298.36 us (diff=1024.000000, rel=0.005025)
lmhead       M=65536 N=50304 K=768 bias=0 gelu=0
  fwd        Triton  45438.74 us (diff=0.000000, rel=0.000000)
  dInp       Triton  50497.39 us (diff=512.000000, rel=0.003067)
  dW         Triton  71571.71 us (diff=1024.000000, rel=0.005405)
  dW+accum   Triton  71592.45 us (diff=1024.000000, rel=0.005405)
```

## Torch Attention Benchmarks

```text
Torch Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 567.918 us
Torch Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2223.190 us
TorchPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 1100.077 us
TorchPacked Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 4250.470 us
```

## cuDNN Attention Benchmarks

```text
cuDNN Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 683.973 us (max_diff=0.007812)
cuDNN Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2417.149 us
cuDNNPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 821.589 us
cuDNNPacked Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 3564.230 us
```

## Triton Attention Benchmarks

```text
Triton Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 2117.378 us (diff=0.001953)
TritonPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 2209.110 us (diff=0.003906)
```

## Torch Classifier Benchmarks

```text
Torch classifier device: NVIDIA GeForce RTX 5090; capability=sm_120
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | Torch        | 18319.103 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | Torch        | unavailable: CUDA OOM at full GPT-2 padded-logits shape
```

## Triton Classifier Benchmarks

```text
Triton classifier device: NVIDIA GeForce RTX 5090; capability=sm_120
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | Triton       |  8311.104 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | Triton       | 22694.304 us
```

## Python Stack LayerNorm Benchmarks

```text
Triton LayerNorm device: NVIDIA GeForce RTX 5090; capability=sm_120
Triton LayerNorm Forward (N=65536, C=768): 177.888 us (y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm ForwardNative (N=65536, C=768): 152.320 us (y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm ForwardWithStats (N=65536, C=768): 2221.440 us (y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
Triton LayerNorm BackwardDInput (N=65536, C=768): 224.736 us (dinp_diff=0.015625; dweight/dbias not produced)
Torch LayerNorm BackwardDInputNative (N=65536, C=768): 212.192 us (dinp_diff=0.031250; dweight/dbias not produced)
Torch LayerNorm BackwardNative (N=65536, C=768): 414.048 us (dinp_diff=0.031250, dweight_diff=1.928711, dbias_diff=1.873962)
Triton LayerNorm BackwardAtomicFP32 (N=65536, C=768): 364.032 us (dinp_diff=0.015625, dweight_diff=0.002319, dbias_diff=0.000580; FP32 dweight/dbias)
Triton LayerNorm FusedResidualForward (N=65536, C=768): 334.336 us (residual_diff=0.000000, y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm FusedResidualForwardNative (N=65536, C=768): 332.352 us (residual_diff=0.000000, y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm FusedResidualForwardWithStats (N=65536, C=768): 3241.952 us (residual_diff=0.000000, y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
Triton LayerNorm Forward (N=65536, C=3072): 590.368 us (y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm ForwardNative (N=65536, C=3072): 563.008 us (y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm ForwardWithStats (N=65536, C=3072): 8957.344 us (y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
Triton LayerNorm BackwardDInput (N=65536, C=3072): 817.120 us (dinp_diff=0.031250; dweight/dbias not produced)
Torch LayerNorm BackwardDInputNative (N=65536, C=3072): 845.856 us (dinp_diff=0.031250; dweight/dbias not produced)
Torch LayerNorm BackwardNative (N=65536, C=3072): 1451.424 us (dinp_diff=0.031250, dweight_diff=1.991577, dbias_diff=1.993774)
Triton LayerNorm BackwardAtomicFP32 (N=65536, C=3072): 1425.824 us (dinp_diff=0.031250, dweight_diff=0.003021, dbias_diff=0.000793; FP32 dweight/dbias)
Triton LayerNorm FusedResidualForward (N=65536, C=3072): 1108.544 us (residual_diff=0.000000, y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm FusedResidualForwardNative (N=65536, C=3072): 1365.536 us (residual_diff=0.000000, y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm FusedResidualForwardWithStats (N=65536, C=3072): 13293.056 us (residual_diff=0.000000, y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
```

## Triton Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | Triton       |   132.779 us
bias_add                       | BT=65536 OC=3072             | Triton       |   532.278 us
gelu_forward                   | BT=65536 C=3072              | Triton       |   531.483 us
gelu_backward_inplace          | BT=65536 C=3072              | Triton       |   793.576 us
```

## Torch Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | Torch        |   135.752 us
bias_add                       | BT=65536 OC=3072             | Torch        |   530.650 us
gelu_forward                   | BT=65536 C=3072              | Torch        |   552.707 us
gelu_backward_inplace          | BT=65536 C=3072              | Torch        | 26828.299 us
bias_grad_reduce               | BT=65536 OC=768              | Torch        |   317.315 us
bias_grad_reduce               | BT=65536 OC=2304             | Torch        |   969.171 us
bias_grad_reduce               | BT=65536 OC=3072             | Torch        |  1313.085 us
global_norm_squared            | params=124475904             | Torch        |  2268.339 us
adamw_update_bf16_state        | params=124475904 no-master   | Torch        |  1215.322 us
adamw_update                   | params=124475904 no-master fp32-state | Torch        |  7317.120 us
encoder_forward                | B=64 T=1024 C=768            | Torch        |   199.592 us
cuda_memset                    | hidden_elems=50331648        | Torch        |    59.968 us
cuda_copy_d2d                  | hidden_elems=50331648        | Torch        |   131.765 us
cuda_memset                    | logits_elems=3296722944      | Torch        |  3920.640 us
cuda_copy_d2d                  | logits_elems=3296722944      | Torch        |  8667.008 us
```

## Training Steps

```text
missing: scratch/sm120_rounds/codex_sm120_round_torch_stack_refresh_20260521/train_gpt2cu.log
```

## validate_sm120_round
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_round_torch_stack_refresh_20260521 --write-scoreboard scratch/sm120_rounds/codex_sm120_round_torch_stack_refresh_20260521/scoreboard-candidates.md --write-selected-backends scratch/sm120_rounds/codex_sm120_round_torch_stack_refresh_20260521/selected-backends.json --write-promotion-candidates scratch/sm120_rounds/codex_sm120_round_torch_stack_refresh_20260521/promotion-candidates.json --require-manifest --require-stack-probe --require-correctness --require-benchmarks`

