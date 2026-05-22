# SM120 Optimization Round

- run label: `codex_sm120_promoted_disable_backward_stream_sync_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_disable_backward_stream_sync_x3_20260522`
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
- working tree: `647` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Fri May 22 02:03:49 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0  On |                  N/A |
|100%   28C    P8             55W /  575W |    3807MiB /  32607MiB |      1%      Default |
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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522/round-manifest.md --run-label codex_sm120_promoted_disable_backward_stream_sync_x3_20260522 --artifact-dir scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522 --train-out-dir log124M/5090_S_codex_sm120_promoted_disable_backward_stream_sync_x3_20260522 --max-steps 3 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 0 --run-correctness 1 --run-benchmarks 1 --run-python-stack-benchmarks 0 --cudnn-packed-backward-route saved-forward --libtorch-runtime-route cxx-api-raw-pointer --libtorch-runtime-supplemental-shapes gelu_forward --run-libtorch-trainer-link-probe 0 --run-libtorch-matmul-benchmarks 0 --libtorch-matmul-shapes qkv\ attproj\ fc\ fcproj\ lmhead --sm120-use-libtorch-memory 0 --sm120-use-libtorch-grad-zero 0 --sm120-use-libtorch-dresidual-zero 1 --run-training 1 --keep-checkpoints 0`


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
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_promoted_disable_backward_stream_sync_x3_20260522 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 3`


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
  fwd      TK   1067.73 us | cuBLASLt   1046.84 us | cuBLAS   1463.29 us | TK/cuBLASLt 1.02x
  dInp   TK   1087.57 us | cuBLASLt   1013.56 us | cuBLAS   1007.41 us | TK/cuBLASLt 1.07x
  dW     TK   1961.64 us | cuBLASLt   1112.44 us | cuBLAS    998.16 us | TK/cuBLASLt 1.76x
  dW+accum TK   1966.08 us | cuBLASLt   1111.36 us | cuBLAS    994.34 us | TK/cuBLASLt 1.77x
  fwd      TK    374.45 us | cuBLASLt    367.78 us | cuBLAS    482.26 us | TK/cuBLASLt 1.02x
  dInp   TK    380.13 us | cuBLASLt    371.93 us | cuBLAS    364.00 us | TK/cuBLASLt 1.02x
  dW     TK   1926.84 us | cuBLASLt    372.03 us | cuBLAS    327.88 us | TK/cuBLASLt 5.18x
  dW+accum TK   1928.85 us | cuBLASLt    375.30 us | cuBLAS    329.61 us | TK/cuBLASLt 5.14x
  fwd+GeLU TK fused   1599.74 us | TK explicit   2255.39 us | cuBLASLt   1714.76 us | cuBLAS explicit   2466.38 us | explicit/cuBLASLt 1.32x
  dInp   TK   1447.08 us | cuBLASLt   1441.92 us | cuBLAS   1458.61 us | TK/cuBLASLt 1.00x
  dW     TK   2151.03 us | cuBLASLt   1533.48 us | cuBLAS   1330.15 us | TK/cuBLASLt 1.40x
  dW+accum TK   1989.84 us | cuBLASLt   1481.40 us | cuBLAS   1309.92 us | TK/cuBLASLt 1.34x
  fwd      TK   1466.74 us | cuBLASLt   1485.74 us | cuBLAS   1732.51 us | TK/cuBLASLt 0.99x
  dInp   TK   1691.05 us | cuBLASLt   1536.44 us | cuBLAS   1575.39 us | TK/cuBLASLt 1.10x
  dInp+dGeLU TK   1991.25 us | cuBLASLt fused   2130.47 us | cuBLASLt explicit   2561.87 us | cuBLAS explicit   2411.68 us | explicit/fused 1.20x
  dW     TK   2070.35 us | cuBLASLt   1517.98 us | cuBLAS   1415.86 us | TK/cuBLASLt 1.36x
  dW+accum TK   2234.66 us | cuBLASLt   1489.67 us | cuBLAS   1309.85 us | TK/cuBLASLt 1.50x
  fwd      TK  27725.67 us | cuBLASLt  22239.39 us | cuBLAS  25494.74 us | TK/cuBLASLt 1.25x
  dInp   TK  25420.28 us | cuBLASLt  23029.78 us | cuBLAS  22534.18 us | TK/cuBLASLt 1.10x
  dW     TK  26873.24 us | cuBLASLt  22299.70 us | cuBLAS  23347.94 us | TK/cuBLASLt 1.21x
  dW+accum TK  28802.36 us | cuBLASLt  22366.23 us | cuBLAS  21172.97 us | TK/cuBLASLt 1.29x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 797.471 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2848.220 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 141.178 us
LayerNorm FusedResidualForward (N=65536, C=768): 284.532 us
LayerNorm Backward (N=65536, C=768): 275.974 us
LayerNorm Forward (N=65536, C=3072): 588.651 us
LayerNorm FusedResidualForward (N=65536, C=3072): 1175.116 us
LayerNorm Backward (N=65536, C=3072): 1146.245 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    95.194 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   558.796 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   566.258 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   813.951 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    26.082 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   187.664 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   247.734 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  4086.528 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  9117.165 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  4509.792 us
cuda_memset                    | logits_elems=3296722944      | CUDA kernel  |  4461.548 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  9115.469 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA kernel  |  9284.870 us
cuda_memset                    | grad_elems=124475904         | CUDA runtime |   160.965 us
cuda_memset                    | grad_elems=124475904         | CUDA kernel  |   163.107 us
global_norm_squared            | params=124475904             | CUDA         |   186.237 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1880.858 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    93.413 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    67.525 us
cuda_memset                    | hidden_elems=50331648        | CUDA kernel  |    68.830 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   134.648 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA kernel  |   145.631 us
```

## Torch Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522/bench_sm120_torch_matmul.log
```

## LibTorch C++ Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522/bench_sm120_libtorch_matmul.log
```

## CuTeDSL Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522/bench_sm120_cutedsl_matmul.log
```

## Triton Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522/bench_sm120_triton_matmul.log
```

## Torch Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522/bench_sm120_torch_attention.log
```

## cuDNN Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522/bench_sm120_cudnn_attention.log
```

## Triton Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522/bench_sm120_triton_attention.log
```

## Torch Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522/bench_sm120_torch_classifier.log
```

## Triton Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522/bench_sm120_triton_classifier.log
```

## Python Stack LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522/bench_sm120_layernorm_python_stacks.log
```

## Triton Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522/bench_sm120_triton_runtime.log
```

## Torch Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522/bench_sm120_torch_runtime.log
```

## LibTorch C++ Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522/bench_sm120_libtorch_runtime.log
```

## LibTorch Trainer Link Probe

```text
missing: scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522/validate_libtorch_trainer_link.log
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/3 | loss 11.032358 (+nanz)| norm 22.1414 (+nanz)| lr 8.57e-07 | 2589.33 ms | 38.8% bf16 MFU | 202480 tok/s
step    2/3 | loss 10.958507 (+nanz)| norm 22.0968 (+nanz)| lr 1.71e-06 | 2551.94 ms | 39.4% bf16 MFU | 205447 tok/s
step    3/3 | loss 10.811316 (+nanz)| norm 21.1251 (+nanz)| lr 2.57e-06 | 2518.34 ms | 39.9% bf16 MFU | 206852 tok/s
val loss 10.609911
total average iteration time: 2535.140514 ms
```
