# SM120 Optimization Round

- run label: `codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522`
- train output dir: `log124M/5090_S_codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522`
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
- working tree: `686` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Fri May 22 09:42:12 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0  On |                  N/A |
|100%   35C    P3             71W /  575W |    1307MiB /  32607MiB |      1%      Default |
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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522/round-manifest.md --run-label codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522 --artifact-dir scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522 --train-out-dir log124M/5090_S_codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522 --max-steps 10 --train-zero-stage 1 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 0 --run-correctness 1 --run-benchmarks 1 --run-python-stack-benchmarks 0 --cudnn-packed-backward-route saved-forward --libtorch-runtime-route cxx-api-raw-pointer --libtorch-runtime-supplemental-shapes gelu_forward --run-libtorch-trainer-link-probe 0 --run-libtorch-matmul-benchmarks 0 --libtorch-matmul-shapes qkv\ attproj\ fc\ fcproj\ lmhead --sm120-use-libtorch-memory 0 --sm120-use-libtorch-grad-zero 0 --sm120-use-libtorch-dresidual-zero 1 --run-training 1 --keep-checkpoints 0`


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
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 10`


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
  fwd      TK   1073.20 us | cuBLASLt   1141.95 us | cuBLAS   1468.76 us | TK/cuBLASLt 0.94x
  dInp   TK   1116.22 us | cuBLASLt   1025.73 us | cuBLAS   1015.50 us | TK/cuBLASLt 1.09x
  dW     TK   1967.79 us | cuBLASLt   1111.63 us | cuBLAS   1045.58 us | TK/cuBLASLt 1.77x
  dW+accum TK   1969.90 us | cuBLASLt   1115.95 us | cuBLAS    998.14 us | TK/cuBLASLt 1.77x
  fwd      TK    375.94 us | cuBLASLt    373.58 us | cuBLAS    490.61 us | TK/cuBLASLt 1.01x
  dInp   TK    405.92 us | cuBLASLt    371.23 us | cuBLAS    365.85 us | TK/cuBLASLt 1.09x
  dW     TK   1926.38 us | cuBLASLt    376.69 us | cuBLAS    328.56 us | TK/cuBLASLt 5.11x
  dW+accum TK   1933.48 us | cuBLASLt    377.66 us | cuBLAS    336.51 us | TK/cuBLASLt 5.12x
  fwd+GeLU TK fused   1590.92 us | TK explicit   1983.82 us | cuBLASLt   1534.74 us | cuBLAS explicit   2457.49 us | explicit/cuBLASLt 1.29x
  dInp   TK   1522.04 us | cuBLASLt   1374.46 us | cuBLAS   1358.23 us | TK/cuBLASLt 1.11x
  dW     TK   1988.96 us | cuBLASLt   1478.78 us | cuBLAS   1309.96 us | TK/cuBLASLt 1.34x
  dW+accum TK   1990.50 us | cuBLASLt   1496.99 us | cuBLAS   1360.92 us | TK/cuBLASLt 1.33x
  fwd      TK   1426.17 us | cuBLASLt   1381.27 us | cuBLAS   1548.22 us | TK/cuBLASLt 1.03x
  dInp   TK   1480.32 us | cuBLASLt   1437.77 us | cuBLAS   1371.00 us | TK/cuBLASLt 1.03x
  dInp+dGeLU TK   1818.53 us | cuBLASLt fused   1831.64 us | cuBLASLt explicit   2161.45 us | cuBLAS explicit   2246.29 us | explicit/fused 1.18x
  dW     TK   1968.95 us | cuBLASLt   1507.32 us | cuBLAS   1317.82 us | TK/cuBLASLt 1.31x
  dW+accum TK   1971.70 us | cuBLASLt   1528.09 us | cuBLAS   1319.17 us | TK/cuBLASLt 1.29x
  fwd      TK  27960.96 us | cuBLASLt  22512.54 us | cuBLAS  22577.40 us | TK/cuBLASLt 1.24x
  dInp   TK  24133.08 us | cuBLASLt  21858.37 us | cuBLAS  21261.49 us | TK/cuBLASLt 1.10x
  dW     TK  26850.76 us | cuBLASLt  21124.49 us | cuBLAS  21275.05 us | TK/cuBLASLt 1.27x
  dW+accum TK  26515.94 us | cuBLASLt  21300.35 us | cuBLAS  21428.87 us | TK/cuBLASLt 1.24x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 794.987 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2753.765 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 143.694 us
LayerNorm FusedResidualForward (N=65536, C=768): 285.235 us
LayerNorm Backward (N=65536, C=768): 275.098 us
LayerNorm Forward (N=65536, C=3072): 558.391 us
LayerNorm FusedResidualForward (N=65536, C=3072): 1104.304 us
LayerNorm Backward (N=65536, C=3072): 1133.981 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    93.906 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   538.239 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   559.323 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   794.643 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    26.424 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   187.861 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   247.048 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  4122.796 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  9017.862 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  4216.352 us
cuda_memset                    | logits_elems=3296722944      | CUDA kernel  |  4488.973 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  8982.221 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA kernel  |  9238.956 us
cuda_memset                    | grad_elems=124475904         | CUDA runtime |   156.770 us
cuda_memset                    | grad_elems=124475904         | CUDA kernel  |   161.787 us
global_norm_squared            | params=124475904             | CUDA         |   185.651 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1864.371 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    83.792 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    63.692 us
cuda_memset                    | hidden_elems=50331648        | CUDA kernel  |    63.396 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   138.403 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA kernel  |   135.908 us
```

## Torch Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522/bench_sm120_torch_matmul.log
```

## LibTorch C++ Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522/bench_sm120_libtorch_matmul.log
```

## CuTeDSL Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522/bench_sm120_cutedsl_matmul.log
```

## Triton Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522/bench_sm120_triton_matmul.log
```

## Torch Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522/bench_sm120_torch_attention.log
```

## cuDNN Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522/bench_sm120_cudnn_attention.log
```

## Triton Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522/bench_sm120_triton_attention.log
```

## Torch Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522/bench_sm120_torch_classifier.log
```

## Triton Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522/bench_sm120_triton_classifier.log
```

## Python Stack LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522/bench_sm120_layernorm_python_stacks.log
```

## Triton Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522/bench_sm120_triton_runtime.log
```

## Torch Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522/bench_sm120_torch_runtime.log
```

## LibTorch C++ Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522/bench_sm120_libtorch_runtime.log
```

## LibTorch Trainer Link Probe

```text
missing: scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522/validate_libtorch_trainer_link.log
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/10 | loss 11.032358 (+nanz)| norm 22.1414 (+nanz)| lr 8.57e-07 | 2524.47 ms | 39.8% bf16 MFU | 207682 tok/s
step    2/10 | loss 10.958507 (+nanz)| norm 22.0968 (+nanz)| lr 1.71e-06 | 2517.24 ms | 39.9% bf16 MFU | 208279 tok/s
step    3/10 | loss 10.811316 (+nanz)| norm 21.1251 (+nanz)| lr 2.57e-06 | 2521.78 ms | 39.9% bf16 MFU | 208087 tok/s
step    4/10 | loss 10.610130 (+nanz)| norm 18.7014 (+nanz)| lr 3.43e-06 | 2525.43 ms | 39.8% bf16 MFU | 207917 tok/s
step    5/10 | loss 10.392586 (+nanz)| norm 15.0184 (+nanz)| lr 4.29e-06 | 2526.93 ms | 39.8% bf16 MFU | 207799 tok/s
step    6/10 | loss 10.186255 (+nanz)| norm 12.0843 (+nanz)| lr 5.14e-06 | 2531.79 ms | 39.7% bf16 MFU | 207641 tok/s
step    7/10 | loss 10.010621 (+nanz)| norm 10.2002 (+nanz)| lr 6.00e-06 | 2530.33 ms | 39.7% bf16 MFU | 207558 tok/s
step    8/10 | loss 9.855870 (+nanz)| norm 8.7905 (+nanz)| lr 6.86e-06 | 2535.65 ms | 39.6% bf16 MFU | 207427 tok/s
step    9/10 | loss 9.719423 (+nanz)| norm 7.4665 (+nanz)| lr 7.71e-06 | 2533.24 ms | 39.7% bf16 MFU | 207358 tok/s
step   10/10 | loss 9.588612 (+nanz)| norm 6.3099 (+nanz)| lr 8.57e-06 | 2538.55 ms | 39.6% bf16 MFU | 207246 tok/s
val loss 9.483727
total average iteration time: 2528.994269 ms
```

## validate_sm120_round
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522 --write-scoreboard scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522/scoreboard-candidates.md --write-selected-backends scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522/selected-backends.json --write-promotion-candidates scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_current_rerun_x10_20260522/promotion-candidates.json --require-manifest --require-correctness --require-benchmarks --require-training --forbid-checkpoints`

