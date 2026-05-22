# SM120 Optimization Round

- run label: `codex_sm120_runtime_grad_zero_dprep2_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_runtime_grad_zero_dprep2_x3_20260522`
- max steps: `3`
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
- working tree: `674` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Fri May 22 07:28:03 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0 Off |                  N/A |
|100%   32C    P8             47W /  575W |    2683MiB /  32607MiB |      1%      Default |
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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522/round-manifest.md --run-label codex_sm120_runtime_grad_zero_dprep2_x3_20260522 --artifact-dir scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522 --train-out-dir log124M/5090_S_codex_sm120_runtime_grad_zero_dprep2_x3_20260522 --max-steps 3 --train-zero-stage 1 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 0 --run-correctness 1 --run-benchmarks 1 --run-python-stack-benchmarks 0 --cudnn-packed-backward-route saved-forward --libtorch-runtime-route cxx-api-raw-pointer --libtorch-runtime-supplemental-shapes gelu_forward --run-libtorch-trainer-link-probe 0 --run-libtorch-matmul-benchmarks 0 --libtorch-matmul-shapes qkv\ attproj\ fc\ fcproj\ lmhead --sm120-use-libtorch-memory 0 --sm120-use-libtorch-grad-zero 0 --sm120-use-libtorch-dresidual-zero 1 --run-training 1 --keep-checkpoints 0`


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
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_runtime_grad_zero_dprep2_x3_20260522 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 3`


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
  fwd      TK   1077.88 us | cuBLASLt   1070.48 us | cuBLAS   1415.33 us | TK/cuBLASLt 1.01x
  dInp   TK   1090.44 us | cuBLASLt   1017.45 us | cuBLAS   1012.20 us | TK/cuBLASLt 1.07x
  dW     TK   1972.94 us | cuBLASLt   1110.88 us | cuBLAS   1014.16 us | TK/cuBLASLt 1.78x
  dW+accum TK   1942.02 us | cuBLASLt   1117.16 us | cuBLAS    995.69 us | TK/cuBLASLt 1.74x
  fwd      TK    376.76 us | cuBLASLt    373.16 us | cuBLAS    485.48 us | TK/cuBLASLt 1.01x
  dInp   TK    406.52 us | cuBLASLt    367.16 us | cuBLAS    365.36 us | TK/cuBLASLt 1.11x
  dW     TK   1928.00 us | cuBLASLt    376.08 us | cuBLAS    327.64 us | TK/cuBLASLt 5.13x
  dW+accum TK   1944.62 us | cuBLASLt    375.65 us | cuBLAS    332.29 us | TK/cuBLASLt 5.18x
  fwd+GeLU TK fused   1563.17 us | TK explicit   1995.70 us | cuBLASLt   1521.40 us | cuBLAS explicit   2460.46 us | explicit/cuBLASLt 1.31x
  dInp   TK   1498.67 us | cuBLASLt   1378.41 us | cuBLAS   1350.09 us | TK/cuBLASLt 1.09x
  dW     TK   1998.94 us | cuBLASLt   1498.29 us | cuBLAS   1313.32 us | TK/cuBLASLt 1.33x
  dW+accum TK   1959.27 us | cuBLASLt   1510.66 us | cuBLAS   1318.36 us | TK/cuBLASLt 1.30x
  fwd      TK   1433.00 us | cuBLASLt   1397.22 us | cuBLAS   1555.75 us | TK/cuBLASLt 1.03x
  dInp   TK   1524.64 us | cuBLASLt   1407.81 us | cuBLAS   1420.15 us | TK/cuBLASLt 1.08x
  dInp+dGeLU TK   1826.46 us | cuBLASLt fused   1839.35 us | cuBLASLt explicit   2176.21 us | cuBLAS explicit   2177.84 us | explicit/fused 1.18x
  dW     TK   1986.06 us | cuBLASLt   1513.87 us | cuBLAS   1308.57 us | TK/cuBLASLt 1.31x
  dW+accum TK   1988.61 us | cuBLASLt   1471.97 us | cuBLAS   1316.21 us | TK/cuBLASLt 1.35x
  fwd      TK  27722.72 us | cuBLASLt  22342.21 us | cuBLAS  22438.28 us | TK/cuBLASLt 1.24x
  dInp   TK  24015.31 us | cuBLASLt  21893.32 us | cuBLAS  21416.32 us | TK/cuBLASLt 1.10x
  dW     TK  26296.28 us | cuBLASLt  21069.36 us | cuBLAS  21247.63 us | TK/cuBLASLt 1.25x
  dW+accum TK  26571.44 us | cuBLASLt  21035.08 us | cuBLAS  21258.11 us | TK/cuBLASLt 1.26x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 783.837 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2744.174 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 136.619 us
LayerNorm FusedResidualForward (N=65536, C=768): 275.750 us
LayerNorm Backward (N=65536, C=768): 272.306 us
LayerNorm Forward (N=65536, C=3072): 541.065 us
LayerNorm FusedResidualForward (N=65536, C=3072): 1084.904 us
LayerNorm Backward (N=65536, C=3072): 1105.540 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    88.777 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   538.433 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   534.758 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   770.217 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    24.920 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   186.256 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   245.251 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  3942.925 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  8911.878 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  3916.339 us
cuda_memset                    | logits_elems=3296722944      | CUDA kernel  |  4083.443 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  8729.056 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA kernel  |  9012.762 us
cuda_memset                    | grad_elems=124475904         | CUDA runtime |   148.794 us
cuda_memset                    | grad_elems=124475904         | CUDA kernel  |   150.083 us
global_norm_squared            | params=124475904             | CUDA         |   185.085 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1805.322 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    73.967 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    62.769 us
cuda_memset                    | hidden_elems=50331648        | CUDA kernel  |    60.502 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   131.523 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA kernel  |   135.539 us
```

## Torch Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522/bench_sm120_torch_matmul.log
```

## LibTorch C++ Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522/bench_sm120_libtorch_matmul.log
```

## CuTeDSL Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522/bench_sm120_cutedsl_matmul.log
```

## Triton Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522/bench_sm120_triton_matmul.log
```

## Torch Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522/bench_sm120_torch_attention.log
```

## cuDNN Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522/bench_sm120_cudnn_attention.log
```

## Triton Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522/bench_sm120_triton_attention.log
```

## Torch Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522/bench_sm120_torch_classifier.log
```

## Triton Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522/bench_sm120_triton_classifier.log
```

## Python Stack LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522/bench_sm120_layernorm_python_stacks.log
```

## Triton Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522/bench_sm120_triton_runtime.log
```

## Torch Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522/bench_sm120_torch_runtime.log
```

## LibTorch C++ Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522/bench_sm120_libtorch_runtime.log
```

## LibTorch Trainer Link Probe

```text
missing: scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522/validate_libtorch_trainer_link.log
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/3 | loss 11.032358 (+nanz)| norm 22.1414 (+nanz)| lr 8.57e-07 | 2500.46 ms | 40.2% bf16 MFU | 209676 tok/s
step    2/3 | loss 10.958507 (+nanz)| norm 22.0968 (+nanz)| lr 1.71e-06 | 2494.79 ms | 40.3% bf16 MFU | 210153 tok/s
step    3/3 | loss 10.811316 (+nanz)| norm 21.1251 (+nanz)| lr 2.57e-06 | 2499.50 ms | 40.2% bf16 MFU | 209950 tok/s
val loss 10.609911
total average iteration time: 2497.146368 ms
```

## validate_sm120_round
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522 --write-scoreboard scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522/scoreboard-candidates.md --write-selected-backends scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522/selected-backends.json --write-promotion-candidates scratch/sm120_rounds/codex_sm120_runtime_grad_zero_dprep2_x3_20260522/promotion-candidates.json --require-manifest --require-correctness --require-benchmarks --require-training --forbid-checkpoints`

