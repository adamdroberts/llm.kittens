# SM120 Optimization Round

- run label: `codex_sm120_round_torch_attention_materialized_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_torch_attention_materialized_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_torch_attention_materialized_20260521`
- max steps: `3`
- python: `/home/adam/miniconda3/envs/llm-kittens/bin/python`
- git commit: `0f21747`
- working tree: `487` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Thu May 21 03:13:48 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0 Off |                  N/A |
|100%   30C    P8             46W /  575W |     677MiB /  32607MiB |      1%      Default |
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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_round_torch_attention_materialized_20260521/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_round_torch_attention_materialized_20260521/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_round_torch_attention_materialized_20260521/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_round_torch_attention_materialized_20260521/round-manifest.md --run-label codex_sm120_round_torch_attention_materialized_20260521 --artifact-dir scratch/sm120_rounds/codex_sm120_round_torch_attention_materialized_20260521 --train-out-dir log124M/5090_S_codex_sm120_round_torch_attention_materialized_20260521 --max-steps 3 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 1 --run-benchmarks 1 --run-python-stack-benchmarks 1 --run-training 0 --keep-checkpoints 0`


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
  fwd      TK   1072.97 us | cuBLASLt   1044.71 us | cuBLAS   1456.83 us | TK/cuBLASLt 1.03x
  dInp   TK   1087.09 us | cuBLASLt   1007.11 us | cuBLAS   1007.10 us | TK/cuBLASLt 1.08x
  dW     TK   1487.36 us | cuBLASLt   1129.73 us | cuBLAS    993.81 us | TK/cuBLASLt 1.32x
  dW+accum TK   1484.35 us | cuBLASLt   1119.18 us | cuBLAS    996.07 us | TK/cuBLASLt 1.33x
  fwd      TK    375.77 us | cuBLASLt    375.37 us | cuBLAS    485.40 us | TK/cuBLASLt 1.00x
  dInp   TK    381.30 us | cuBLASLt    370.07 us | cuBLAS    365.35 us | TK/cuBLASLt 1.03x
  dW     TK    537.17 us | cuBLASLt    380.39 us | cuBLAS    328.70 us | TK/cuBLASLt 1.41x
  dW+accum TK    537.81 us | cuBLASLt    388.09 us | cuBLAS    332.00 us | TK/cuBLASLt 1.39x
  fwd+GeLU TK fused   1564.70 us | TK explicit   1995.76 us | cuBLASLt   1494.69 us | cuBLAS explicit   2467.54 us | explicit/cuBLASLt 1.34x
  dInp   TK   1448.88 us | cuBLASLt   1367.64 us | cuBLAS   1328.47 us | TK/cuBLASLt 1.06x
  dW     TK   1749.91 us | cuBLASLt   1493.23 us | cuBLAS   1330.14 us | TK/cuBLASLt 1.17x
  dW+accum TK   1750.58 us | cuBLASLt   1479.88 us | cuBLAS   1313.76 us | TK/cuBLASLt 1.18x
  fwd      TK   1444.56 us | cuBLASLt   1399.78 us | cuBLAS   1548.91 us | TK/cuBLASLt 1.03x
  dInp   TK   1476.48 us | cuBLASLt   1372.02 us | cuBLAS   1415.14 us | TK/cuBLASLt 1.08x
  dInp+dGeLU TK   1789.27 us | cuBLASLt fused   1815.34 us | cuBLASLt explicit   2197.96 us | cuBLAS explicit   2175.72 us | explicit/fused 1.21x
  dW     TK   1742.28 us | cuBLASLt   1470.47 us | cuBLAS   1354.62 us | TK/cuBLASLt 1.18x
  dW+accum TK   1725.65 us | cuBLASLt   1523.58 us | cuBLAS   1316.57 us | TK/cuBLASLt 1.13x
  fwd      TK  27766.07 us | cuBLASLt  22273.02 us | cuBLAS  22242.28 us | TK/cuBLASLt 1.25x
  dInp   TK  23961.37 us | cuBLASLt  21785.16 us | cuBLAS  21339.16 us | TK/cuBLASLt 1.10x
  dW     TK  26107.19 us | cuBLASLt  20944.17 us | cuBLAS  21245.57 us | TK/cuBLASLt 1.25x
  dW+accum TK  26135.64 us | cuBLASLt  20992.15 us | cuBLAS  21219.42 us | TK/cuBLASLt 1.25x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 778.909 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2728.503 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 139.810 us
LayerNorm FusedResidualForward (N=65536, C=768): 276.514 us
LayerNorm Backward (N=65536, C=768): 288.147 us
LayerNorm Forward (N=65536, C=3072): 543.194 us
LayerNorm FusedResidualForward (N=65536, C=3072): 1084.727 us
LayerNorm Backward (N=65536, C=3072): 1266.138 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    96.825 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   528.993 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   527.850 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   779.645 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    29.872 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   186.634 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   245.650 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  3948.493 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  8937.452 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  3913.728 us
cuda_memset                    | logits_elems=3296722944      | CUDA kernel  |  4121.190 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  8776.832 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA kernel  |  9152.672 us
global_norm_squared            | params=124475904             | CUDA         |   184.909 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1835.683 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    85.180 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    61.608 us
cuda_memset                    | hidden_elems=50331648        | CUDA kernel  |    62.200 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   131.723 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA kernel  |   136.210 us
```

## Torch Matmul Benchmarks

```text
Torch matmul device: NVIDIA GeForce RTX 5090; capability=sm_120
qkv          M=65536 N=2304 K=768 bias=1 gelu=0
  fwd      Torch   1455.71 us
  dInp   Torch   1026.83 us
  dW     Torch   1004.90 us
  dW+accum Torch   1017.32 us
attproj      M=65536 N=768 K=768 bias=1 gelu=0
  fwd      Torch    519.36 us
  dInp   Torch    372.62 us
  dW     Torch    337.22 us
  dW+accum Torch    347.01 us
fc           M=65536 N=3072 K=768 bias=1 gelu=1
  fwd+GeLU Torch   2486.11 us
  dInp   Torch   1356.87 us
  dW     Torch   1328.12 us
  dW+accum Torch   1366.06 us
fcproj       M=65536 N=768 K=3072 bias=1 gelu=0
  fwd      Torch   1601.02 us
  dInp   Torch   1405.25 us
  dInp+dGeLU Torch  28242.41 us
  dW     Torch   1368.78 us
  dW+accum Torch   1392.77 us
lmhead       M=65536 N=50304 K=768 bias=0 gelu=0
  fwd      Torch  22604.83 us
  dInp   Torch  21718.14 us
  dW     Torch  21530.62 us
  dW+accum Torch  21109.95 us
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
  fwd        Triton   1934.52 us (diff=0.000000, rel=0.000000)
  dInp       Triton   2271.06 us (diff=16.000000, rel=0.002075)
  dW         Triton   2127.65 us (diff=1024.000000, rel=0.003817)
  dW+accum   Triton   2137.71 us (diff=1024.000000, rel=0.003817)
attproj      M=65536 N=768 K=768 bias=1 gelu=0
  fwd        Triton    660.39 us (diff=0.000000, rel=0.000000)
  dInp       Triton    640.57 us (diff=0.000000, rel=0.000000)
  dW         Triton    559.89 us (diff=512.000000, rel=0.001953)
  dW+accum   Triton    561.00 us (diff=0.000000, rel=0.000000)
fc           M=65536 N=3072 K=768 bias=1 gelu=1
  fwd+GeLU   Triton   2635.40 us (diff=0.000000, rel=0.000000)
  dInp       Triton   3000.32 us (diff=0.000000, rel=0.000000)
  dW         Triton   2253.02 us (diff=256.000000, rel=0.002825)
  dW+accum   Triton   2261.29 us (diff=256.000000, rel=0.002825)
fcproj       M=65536 N=768 K=3072 bias=1 gelu=0
  fwd        Triton   3074.09 us (diff=0.000000, rel=0.000000)
  dInp       Triton   2724.38 us (diff=0.000000, rel=0.000000)
  dInp+dGeLU Triton   3303.80 us (diff=32.000000, rel=0.006711)
  dW         Triton   2263.01 us (diff=1024.000000, rel=0.005025)
  dW+accum   Triton   2255.76 us (diff=1024.000000, rel=0.005025)
lmhead       M=65536 N=50304 K=768 bias=0 gelu=0
  fwd        Triton  44101.74 us (diff=0.000000, rel=0.000000)
  dInp       Triton  48973.36 us (diff=512.000000, rel=0.003067)
  dW         Triton  69946.88 us (diff=1024.000000, rel=0.005405)
  dW+accum   Triton  69947.25 us (diff=1024.000000, rel=0.005405)
```

## Torch Attention Benchmarks

```text
Torch Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 555.498 us
Torch Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2195.661 us
TorchPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 1068.491 us
TorchPacked Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 4067.501 us
TorchMaterializedPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 1260.536 us
TorchMaterializedPacked Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 4195.885 us
```

## cuDNN Attention Benchmarks

```text
cuDNN Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 685.798 us (max_diff=0.007812)
cuDNN Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2375.994 us
cuDNNPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 802.810 us
cuDNNPacked Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 3514.362 us
```

## Triton Attention Benchmarks

```text
Triton Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 2074.990 us (diff=0.001953)
TritonPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 2196.746 us (diff=0.003906)
```

## Torch Classifier Benchmarks

```text
Torch classifier device: NVIDIA GeForce RTX 5090; capability=sm_120
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | Torch        | 17925.632 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | Torch        | 33300.255 us
```

## Triton Classifier Benchmarks

```text
Triton classifier device: NVIDIA GeForce RTX 5090; capability=sm_120
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | Triton       |  8248.928 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | Triton       | 22245.504 us
```

## Python Stack LayerNorm Benchmarks

```text
Triton LayerNorm device: NVIDIA GeForce RTX 5090; capability=sm_120
Triton LayerNorm Forward (N=65536, C=768): 177.568 us (y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm ForwardNative (N=65536, C=768): 154.880 us (y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm ForwardWithStats (N=65536, C=768): 2198.048 us (y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
Triton LayerNorm BackwardDInput (N=65536, C=768): 227.552 us (dinp_diff=0.015625; dweight/dbias not produced)
Torch LayerNorm BackwardDInputNative (N=65536, C=768): 222.112 us (dinp_diff=0.031250; dweight/dbias not produced)
Torch LayerNorm BackwardNative (N=65536, C=768): 417.984 us (dinp_diff=0.031250, dweight_diff=1.928711, dbias_diff=1.873962)
Triton LayerNorm BackwardAtomicFP32 (N=65536, C=768): 360.160 us (dinp_diff=0.015625, dweight_diff=0.001862, dbias_diff=0.000427; FP32 dweight/dbias)
Triton LayerNorm FusedResidualForward (N=65536, C=768): 312.320 us (residual_diff=0.000000, y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm FusedResidualForwardNative (N=65536, C=768): 329.632 us (residual_diff=0.000000, y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm FusedResidualForwardWithStats (N=65536, C=768): 3172.512 us (residual_diff=0.000000, y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
Triton LayerNorm Forward (N=65536, C=3072): 575.584 us (y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm ForwardNative (N=65536, C=3072): 554.592 us (y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm ForwardWithStats (N=65536, C=3072): 8947.552 us (y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
Triton LayerNorm BackwardDInput (N=65536, C=3072): 812.128 us (dinp_diff=0.031250; dweight/dbias not produced)
Torch LayerNorm BackwardDInputNative (N=65536, C=3072): 845.120 us (dinp_diff=0.031250; dweight/dbias not produced)
Torch LayerNorm BackwardNative (N=65536, C=3072): 1397.760 us (dinp_diff=0.031250, dweight_diff=1.991577, dbias_diff=1.993774)
Triton LayerNorm BackwardAtomicFP32 (N=65536, C=3072): 1424.608 us (dinp_diff=0.031250, dweight_diff=0.002930, dbias_diff=0.000793; FP32 dweight/dbias)
Triton LayerNorm FusedResidualForward (N=65536, C=3072): 1105.120 us (residual_diff=0.000000, y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm FusedResidualForwardNative (N=65536, C=3072): 1310.944 us (residual_diff=0.000000, y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm FusedResidualForwardWithStats (N=65536, C=3072): 12946.112 us (residual_diff=0.000000, y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
```

## Triton Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | Triton       |   132.389 us
bias_add                       | BT=65536 OC=3072             | Triton       |   527.930 us
gelu_forward                   | BT=65536 C=3072              | Triton       |   530.515 us
gelu_backward_inplace          | BT=65536 C=3072              | Triton       |   770.520 us
```

## Torch Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | Torch        |   136.147 us
bias_add                       | BT=65536 OC=3072             | Torch        |   532.366 us
gelu_forward                   | BT=65536 C=3072              | Torch        |   538.289 us
gelu_backward_inplace          | BT=65536 C=3072              | Torch        | 26831.989 us
bias_grad_reduce               | BT=65536 OC=768              | Torch        |   314.643 us
bias_grad_reduce               | BT=65536 OC=2304             | Torch        |   967.494 us
bias_grad_reduce               | BT=65536 OC=3072             | Torch        |  1315.942 us
global_norm_squared            | params=124475904             | Torch        |  2273.318 us
adamw_update_bf16_state        | params=124475904 no-master   | Torch        |  1196.518 us
adamw_update                   | params=124475904 no-master fp32-state | Torch        |  7315.776 us
encoder_forward                | B=64 T=1024 C=768            | Torch        |   199.050 us
cuda_memset                    | hidden_elems=50331648        | Torch        |    60.038 us
cuda_copy_d2d                  | hidden_elems=50331648        | Torch        |   131.984 us
cuda_memset                    | logits_elems=3296722944      | Torch        |  3976.288 us
cuda_copy_d2d                  | logits_elems=3296722944      | Torch        |  8665.152 us
```

## Training Steps

```text
missing: scratch/sm120_rounds/codex_sm120_round_torch_attention_materialized_20260521/train_gpt2cu.log
```

## validate_sm120_round
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_round_torch_attention_materialized_20260521 --write-scoreboard scratch/sm120_rounds/codex_sm120_round_torch_attention_materialized_20260521/scoreboard-candidates.md --write-selected-backends scratch/sm120_rounds/codex_sm120_round_torch_attention_materialized_20260521/selected-backends.json --write-promotion-candidates scratch/sm120_rounds/codex_sm120_round_torch_attention_materialized_20260521/promotion-candidates.json --require-manifest --require-stack-probe --require-correctness --require-benchmarks`

