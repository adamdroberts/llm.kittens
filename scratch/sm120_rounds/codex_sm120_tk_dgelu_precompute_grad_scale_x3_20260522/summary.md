# SM120 Optimization Round

- run label: `codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522`
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
- working tree: `670` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Fri May 22 06:52:19 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0 Off |                  N/A |
|100%   32C    P8             47W /  575W |    2679MiB /  32607MiB |      0%      Default |
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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522/round-manifest.md --run-label codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522 --artifact-dir scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522 --train-out-dir log124M/5090_S_codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522 --max-steps 3 --train-zero-stage 1 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 0 --run-correctness 1 --run-benchmarks 1 --run-python-stack-benchmarks 0 --cudnn-packed-backward-route saved-forward --libtorch-runtime-route cxx-api-raw-pointer --libtorch-runtime-supplemental-shapes gelu_forward --run-libtorch-trainer-link-probe 0 --run-libtorch-matmul-benchmarks 0 --libtorch-matmul-shapes qkv\ attproj\ fc\ fcproj\ lmhead --sm120-use-libtorch-memory 0 --sm120-use-libtorch-grad-zero 0 --sm120-use-libtorch-dresidual-zero 1 --run-training 1 --keep-checkpoints 0`


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
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 3`


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
  fwd      TK   1073.90 us | cuBLASLt   1042.05 us | cuBLAS   1452.81 us | TK/cuBLASLt 1.03x
  dInp   TK   1091.33 us | cuBLASLt   1012.04 us | cuBLAS   1018.79 us | TK/cuBLASLt 1.08x
  dW     TK   1990.44 us | cuBLASLt   1107.50 us | cuBLAS    996.50 us | TK/cuBLASLt 1.80x
  dW+accum TK   1995.06 us | cuBLASLt   1114.90 us | cuBLAS   1000.97 us | TK/cuBLASLt 1.79x
  fwd      TK    376.37 us | cuBLASLt    372.31 us | cuBLAS    488.66 us | TK/cuBLASLt 1.01x
  dInp   TK    380.78 us | cuBLASLt    365.59 us | cuBLAS    365.53 us | TK/cuBLASLt 1.04x
  dW     TK   1947.59 us | cuBLASLt    371.76 us | cuBLAS    326.79 us | TK/cuBLASLt 5.24x
  dW+accum TK   1920.33 us | cuBLASLt    375.75 us | cuBLAS    332.92 us | TK/cuBLASLt 5.11x
  fwd+GeLU TK fused   1583.53 us | TK explicit   1971.51 us | cuBLASLt   1474.48 us | cuBLAS explicit   2482.90 us | explicit/cuBLASLt 1.34x
  dInp   TK   1450.84 us | cuBLASLt   1389.18 us | cuBLAS   1350.81 us | TK/cuBLASLt 1.04x
  dW     TK   1988.09 us | cuBLASLt   1525.70 us | cuBLAS   1308.39 us | TK/cuBLASLt 1.30x
  dW+accum TK   1990.68 us | cuBLASLt   1493.85 us | cuBLAS   1355.42 us | TK/cuBLASLt 1.33x
  fwd      TK   1438.87 us | cuBLASLt   1382.66 us | cuBLAS   1541.89 us | TK/cuBLASLt 1.04x
  dInp   TK   1487.99 us | cuBLASLt   1389.97 us | cuBLAS   1380.22 us | TK/cuBLASLt 1.07x
  dInp+dGeLU TK   1790.50 us | cuBLASLt fused   1861.00 us | cuBLASLt explicit   2181.84 us | cuBLAS explicit   2179.14 us | explicit/fused 1.17x
  dW     TK   1940.92 us | cuBLASLt   1514.83 us | cuBLAS   1309.03 us | TK/cuBLASLt 1.28x
  dW+accum TK   1987.52 us | cuBLASLt   1513.52 us | cuBLAS   1316.93 us | TK/cuBLASLt 1.31x
  fwd      TK  27728.46 us | cuBLASLt  22416.07 us | cuBLAS  22224.23 us | TK/cuBLASLt 1.24x
  dInp   TK  24078.74 us | cuBLASLt  21821.53 us | cuBLAS  21381.90 us | TK/cuBLASLt 1.10x
  dW     TK  26383.01 us | cuBLASLt  21004.28 us | cuBLAS  21255.82 us | TK/cuBLASLt 1.26x
  dW+accum TK  26404.95 us | cuBLASLt  21041.89 us | cuBLAS  21313.26 us | TK/cuBLASLt 1.25x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 782.792 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2739.670 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 137.249 us
LayerNorm FusedResidualForward (N=65536, C=768): 275.846 us
LayerNorm Backward (N=65536, C=768): 270.832 us
LayerNorm Forward (N=65536, C=3072): 542.372 us
LayerNorm FusedResidualForward (N=65536, C=3072): 1082.442 us
LayerNorm Backward (N=65536, C=3072): 1106.328 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    88.732 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   528.599 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   545.190 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   779.113 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    23.605 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   186.493 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   245.134 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  3898.093 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  8895.303 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  4006.720 us
cuda_memset                    | logits_elems=3296722944      | CUDA kernel  |  4040.147 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  8786.003 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA kernel  |  8937.549 us
cuda_memset                    | grad_elems=124475904         | CUDA runtime |   148.923 us
cuda_memset                    | grad_elems=124475904         | CUDA kernel  |   149.744 us
global_norm_squared            | params=124475904             | CUDA         |   184.221 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1828.720 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    79.455 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    62.628 us
cuda_memset                    | hidden_elems=50331648        | CUDA kernel  |    60.292 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   131.588 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA kernel  |   135.215 us
```

## Torch Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522/bench_sm120_torch_matmul.log
```

## LibTorch C++ Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522/bench_sm120_libtorch_matmul.log
```

## CuTeDSL Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522/bench_sm120_cutedsl_matmul.log
```

## Triton Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522/bench_sm120_triton_matmul.log
```

## Torch Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522/bench_sm120_torch_attention.log
```

## cuDNN Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522/bench_sm120_cudnn_attention.log
```

## Triton Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522/bench_sm120_triton_attention.log
```

## Torch Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522/bench_sm120_torch_classifier.log
```

## Triton Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522/bench_sm120_triton_classifier.log
```

## Python Stack LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522/bench_sm120_layernorm_python_stacks.log
```

## Triton Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522/bench_sm120_triton_runtime.log
```

## Torch Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522/bench_sm120_torch_runtime.log
```

## LibTorch C++ Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522/bench_sm120_libtorch_runtime.log
```

## LibTorch Trainer Link Probe

```text
missing: scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522/validate_libtorch_trainer_link.log
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/3 | loss 11.032358 (+nanz)| norm 22.1414 (+nanz)| lr 8.57e-07 | 2497.61 ms | 40.3% bf16 MFU | 209916 tok/s
step    2/3 | loss 10.958512 (+nanz)| norm 22.0968 (+nanz)| lr 1.71e-06 | 2497.02 ms | 40.3% bf16 MFU | 209965 tok/s
step    3/3 | loss 10.811323 (+nanz)| norm 21.1250 (+nanz)| lr 2.57e-06 | 2500.84 ms | 40.2% bf16 MFU | 209801 tok/s
val loss 10.609920
total average iteration time: 2498.933315 ms
```

## validate_sm120_round
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522 --write-scoreboard scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522/scoreboard-candidates.md --write-selected-backends scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522/selected-backends.json --write-promotion-candidates scratch/sm120_rounds/codex_sm120_tk_dgelu_precompute_grad_scale_x3_20260522/promotion-candidates.json --require-manifest --require-correctness --require-benchmarks --require-training --forbid-checkpoints`

