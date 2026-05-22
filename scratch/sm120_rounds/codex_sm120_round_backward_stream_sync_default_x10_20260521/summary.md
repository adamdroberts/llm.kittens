# SM120 Optimization Round

- run label: `codex_sm120_round_backward_stream_sync_default_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_backward_stream_sync_default_x10_20260521`
- max steps: `10`
- python: `/home/adam/miniconda3/envs/llm-kittens/bin/python`
- git commit: `0f21747`
- working tree: `491` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Thu May 21 05:25:54 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0 Off |                  N/A |
|100%   29C    P8             46W /  575W |    1045MiB /  32607MiB |      1%      Default |
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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521/round-manifest.md --run-label codex_sm120_round_backward_stream_sync_default_x10_20260521 --artifact-dir scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521 --train-out-dir log124M/5090_S_codex_sm120_round_backward_stream_sync_default_x10_20260521 --max-steps 10 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 1 --run-benchmarks 1 --run-python-stack-benchmarks 0 --run-training 1 --keep-checkpoints 0`


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
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_round_backward_stream_sync_default_x10_20260521 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 10`


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
  fwd      TK   1070.02 us | cuBLASLt   1039.60 us | cuBLAS   1412.75 us | TK/cuBLASLt 1.03x
  dInp   TK   1087.27 us | cuBLASLt   1011.61 us | cuBLAS   1006.87 us | TK/cuBLASLt 1.07x
  dW     TK   1489.35 us | cuBLASLt   1105.39 us | cuBLAS   1010.95 us | TK/cuBLASLt 1.35x
  dW+accum TK   1498.29 us | cuBLASLt   1126.15 us | cuBLAS    999.97 us | TK/cuBLASLt 1.33x
  fwd      TK    376.86 us | cuBLASLt    370.91 us | cuBLAS    481.40 us | TK/cuBLASLt 1.02x
  dInp   TK    380.65 us | cuBLASLt    369.02 us | cuBLAS    365.51 us | TK/cuBLASLt 1.03x
  dW     TK    538.40 us | cuBLASLt    377.51 us | cuBLAS    328.19 us | TK/cuBLASLt 1.43x
  dW+accum TK    554.59 us | cuBLASLt    379.95 us | cuBLAS    332.00 us | TK/cuBLASLt 1.46x
  fwd+GeLU TK fused   1562.13 us | TK explicit   1948.78 us | cuBLASLt   1497.48 us | cuBLAS explicit   2462.76 us | explicit/cuBLASLt 1.30x
  dInp   TK   1473.78 us | cuBLASLt   1357.23 us | cuBLAS   1327.73 us | TK/cuBLASLt 1.09x
  dW     TK   1743.71 us | cuBLASLt   1500.86 us | cuBLAS   1309.12 us | TK/cuBLASLt 1.16x
  dW+accum TK   1725.80 us | cuBLASLt   1520.87 us | cuBLAS   1333.20 us | TK/cuBLASLt 1.13x
  fwd      TK   1468.60 us | cuBLASLt   1358.93 us | cuBLAS   1592.88 us | TK/cuBLASLt 1.08x
  dInp   TK   1475.35 us | cuBLASLt   1381.46 us | cuBLAS   1383.67 us | TK/cuBLASLt 1.07x
  dInp+dGeLU TK   1785.72 us | cuBLASLt fused   1841.30 us | cuBLASLt explicit   2187.36 us | cuBLAS explicit   2177.24 us | explicit/fused 1.19x
  dW     TK   1710.77 us | cuBLASLt   1514.35 us | cuBLAS   1308.93 us | TK/cuBLASLt 1.13x
  dW+accum TK   1748.55 us | cuBLASLt   1474.44 us | cuBLAS   1353.25 us | TK/cuBLASLt 1.19x
  fwd      TK  27852.15 us | cuBLASLt  22341.62 us | cuBLAS  22429.85 us | TK/cuBLASLt 1.25x
  dInp   TK  23885.27 us | cuBLASLt  21871.27 us | cuBLAS  21261.82 us | TK/cuBLASLt 1.09x
  dW     TK  26071.89 us | cuBLASLt  20951.49 us | cuBLAS  21181.56 us | TK/cuBLASLt 1.24x
  dW+accum TK  26043.79 us | cuBLASLt  20929.24 us | cuBLAS  21306.39 us | TK/cuBLASLt 1.24x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 784.691 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2743.724 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 136.995 us
LayerNorm FusedResidualForward (N=65536, C=768): 275.580 us
LayerNorm Backward (N=65536, C=768): 282.924 us
LayerNorm Forward (N=65536, C=3072): 543.091 us
LayerNorm FusedResidualForward (N=65536, C=3072): 1082.067 us
LayerNorm Backward (N=65536, C=3072): 1273.941 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    80.068 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   536.362 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   528.334 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   780.873 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    24.211 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   186.550 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   245.482 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  3997.862 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  8904.614 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  3948.186 us
cuda_memset                    | logits_elems=3296722944      | CUDA kernel  |  3992.384 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  8774.458 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA kernel  |  9146.873 us
global_norm_squared            | params=124475904             | CUDA         |   185.112 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1807.082 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    84.260 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    59.800 us
cuda_memset                    | hidden_elems=50331648        | CUDA kernel  |    63.130 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   131.734 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA kernel  |   138.625 us
```

## Torch Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521/bench_sm120_torch_matmul.log
```

## CuTeDSL Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521/bench_sm120_cutedsl_matmul.log
```

## Triton Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521/bench_sm120_triton_matmul.log
```

## Torch Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521/bench_sm120_torch_attention.log
```

## cuDNN Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521/bench_sm120_cudnn_attention.log
```

## Triton Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521/bench_sm120_triton_attention.log
```

## Torch Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521/bench_sm120_torch_classifier.log
```

## Triton Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521/bench_sm120_triton_classifier.log
```

## Python Stack LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521/bench_sm120_layernorm_python_stacks.log
```

## Triton Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521/bench_sm120_triton_runtime.log
```

## Torch Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521/bench_sm120_torch_runtime.log
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/10 | loss 11.032358 (+nanz)| norm 22.1414 (+nanz)| lr 8.57e-07 | 2486.74 ms | 40.4% bf16 MFU | 210834 tok/s
step    2/10 | loss 10.958507 (+nanz)| norm 22.0968 (+nanz)| lr 1.71e-06 | 2483.08 ms | 40.5% bf16 MFU | 211144 tok/s
step    3/10 | loss 10.811316 (+nanz)| norm 21.1251 (+nanz)| lr 2.57e-06 | 2484.88 ms | 40.5% bf16 MFU | 211066 tok/s
step    4/10 | loss 10.610130 (+nanz)| norm 18.7014 (+nanz)| lr 3.43e-06 | 2488.37 ms | 40.4% bf16 MFU | 210936 tok/s
step    5/10 | loss 10.392586 (+nanz)| norm 15.0184 (+nanz)| lr 4.29e-06 | 2489.89 ms | 40.4% bf16 MFU | 210836 tok/s
step    6/10 | loss 10.186255 (+nanz)| norm 12.0843 (+nanz)| lr 5.14e-06 | 2491.82 ms | 40.3% bf16 MFU | 210741 tok/s
step    7/10 | loss 10.010621 (+nanz)| norm 10.2002 (+nanz)| lr 6.00e-06 | 2495.18 ms | 40.3% bf16 MFU | 210624 tok/s
step    8/10 | loss 9.855870 (+nanz)| norm 8.7905 (+nanz)| lr 6.86e-06 | 2501.11 ms | 40.2% bf16 MFU | 210458 tok/s
step    9/10 | loss 9.719423 (+nanz)| norm 7.4665 (+nanz)| lr 7.71e-06 | 2502.01 ms | 40.2% bf16 MFU | 210322 tok/s
step   10/10 | loss 9.588612 (+nanz)| norm 6.3099 (+nanz)| lr 8.57e-06 | 2501.86 ms | 40.2% bf16 MFU | 210219 tok/s
val loss 9.483727
total average iteration time: 2493.132803 ms
```

## validate_sm120_round
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521 --write-scoreboard scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521/scoreboard-candidates.md --write-selected-backends scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521/selected-backends.json --write-promotion-candidates scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521/promotion-candidates.json --require-manifest --require-stack-probe --require-correctness --require-benchmarks --require-training --forbid-checkpoints`

