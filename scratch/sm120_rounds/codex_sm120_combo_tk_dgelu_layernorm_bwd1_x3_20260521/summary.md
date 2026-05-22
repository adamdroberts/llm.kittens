# SM120 Optimization Round

- run label: `codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521`
- max steps: `3`
- python: `/home/adam/miniconda3/envs/llm-kittens/bin/python`
- cuDNN packed backward route: `saved-forward`
- LibTorch runtime route: `cxx-api-raw-pointer`
- LibTorch runtime supplemental shapes: `gelu_forward`
- LibTorch trainer link probe: `0`
- LibTorch matmul shapes: `qkv attproj fc fcproj lmhead`
- SM120 LibTorch trainer memory route: `0`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- working tree: `517` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Thu May 21 15:46:52 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0  On |                  N/A |
|100%   34C    P3             64W /  575W |    2313MiB /  32607MiB |      1%      Default |
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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1 SM120_USE_LIBTORCH_MEMORY=0 SM120_USE_LIBTORCH_GRAD_ZERO=0 SM120_USE_LIBTORCH_DRESIDUAL_ZERO=0 PYTHON_BIN=/home/adam/miniconda3/envs/llm-kittens/bin/python`


## write_sm120_round_manifest
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521/round-manifest.md --run-label codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521 --artifact-dir scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521 --train-out-dir log124M/5090_S_codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521 --max-steps 3 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 1 --run-benchmarks 1 --run-python-stack-benchmarks 0 --cudnn-packed-backward-route saved-forward --libtorch-runtime-route cxx-api-raw-pointer --libtorch-runtime-supplemental-shapes gelu_forward --run-libtorch-trainer-link-probe 0 --run-libtorch-matmul-benchmarks 0 --libtorch-matmul-shapes qkv\ attproj\ fc\ fcproj\ lmhead --sm120-use-libtorch-memory 0 --sm120-use-libtorch-grad-zero 0 --sm120-use-libtorch-dresidual-zero 0 --run-training 1 --keep-checkpoints 0`


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
Command: `./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin -j dev/data/tinystories/TinyStories_val.bin -o log124M/5090_S_codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521 -v 250 -s 20000 -g 144 -h 0 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 0 -e d12 -x 3`


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
  fwd      TK   1073.91 us | cuBLASLt   1048.49 us | cuBLAS   1462.05 us | TK/cuBLASLt 1.02x
  dInp   TK   1087.33 us | cuBLASLt   1013.58 us | cuBLAS   1012.12 us | TK/cuBLASLt 1.07x
  dW     TK   1482.63 us | cuBLASLt   1117.57 us | cuBLAS   1037.43 us | TK/cuBLASLt 1.33x
  dW+accum TK   1465.82 us | cuBLASLt   1146.41 us | cuBLAS   1001.20 us | TK/cuBLASLt 1.28x
  fwd      TK    376.54 us | cuBLASLt    369.45 us | cuBLAS    484.13 us | TK/cuBLASLt 1.02x
  dInp   TK    380.83 us | cuBLASLt    366.98 us | cuBLAS    365.66 us | TK/cuBLASLt 1.04x
  dW     TK    547.20 us | cuBLASLt    407.85 us | cuBLAS    329.08 us | TK/cuBLASLt 1.34x
  dW+accum TK    554.43 us | cuBLASLt    376.79 us | cuBLAS    334.60 us | TK/cuBLASLt 1.47x
  fwd+GeLU TK fused   1575.46 us | TK explicit   1982.60 us | cuBLASLt   1527.91 us | cuBLAS explicit   2448.89 us | explicit/cuBLASLt 1.30x
  dInp   TK   1495.96 us | cuBLASLt   1361.81 us | cuBLAS   1340.58 us | TK/cuBLASLt 1.10x
  dW     TK   1716.80 us | cuBLASLt   1518.19 us | cuBLAS   1327.33 us | TK/cuBLASLt 1.13x
  dW+accum TK   1730.02 us | cuBLASLt   1517.62 us | cuBLAS   1337.66 us | TK/cuBLASLt 1.14x
  fwd      TK   1447.91 us | cuBLASLt   1391.12 us | cuBLAS   1557.44 us | TK/cuBLASLt 1.04x
  dInp   TK   1512.47 us | cuBLASLt   1411.86 us | cuBLAS   1376.91 us | TK/cuBLASLt 1.07x
  dInp+dGeLU TK   1809.82 us | cuBLASLt fused   1860.84 us | cuBLASLt explicit   2190.63 us | cuBLAS explicit   2188.03 us | explicit/fused 1.18x
  dW     TK   1731.82 us | cuBLASLt   1504.68 us | cuBLAS   1309.48 us | TK/cuBLASLt 1.15x
  dW+accum TK   1810.39 us | cuBLASLt   1476.86 us | cuBLAS   1357.70 us | TK/cuBLASLt 1.23x
  fwd      TK  28030.27 us | cuBLASLt  22536.91 us | cuBLAS  22413.65 us | TK/cuBLASLt 1.24x
  dInp   TK  24188.88 us | cuBLASLt  22075.99 us | cuBLAS  21694.90 us | TK/cuBLASLt 1.10x
  dW     TK  26237.77 us | cuBLASLt  20940.67 us | cuBLAS  21397.89 us | TK/cuBLASLt 1.25x
  dW+accum TK  26325.87 us | cuBLASLt  21076.35 us | cuBLAS  23064.54 us | TK/cuBLASLt 1.25x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 854.264 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2777.959 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 141.735 us
LayerNorm FusedResidualForward (N=65536, C=768): 283.129 us
LayerNorm Backward (N=65536, C=768): 272.067 us
LayerNorm Forward (N=65536, C=3072): 579.365 us
LayerNorm FusedResidualForward (N=65536, C=3072): 1182.833 us
LayerNorm Backward (N=65536, C=3072): 1319.754 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |   100.156 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   582.300 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   547.629 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   800.616 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    25.693 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   188.082 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   247.454 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  4537.434 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  9685.095 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  4269.037 us
cuda_memset                    | logits_elems=3296722944      | CUDA kernel  |  4483.949 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  9706.381 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA kernel  | 10419.693 us
cuda_memset                    | grad_elems=124475904         | CUDA runtime |   172.981 us
cuda_memset                    | grad_elems=124475904         | CUDA kernel  |   178.491 us
global_norm_squared            | params=124475904             | CUDA         |   204.859 us
adamw_update                   | params=124475904 no-master   | CUDA         |  2065.447 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    84.467 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    70.589 us
cuda_memset                    | hidden_elems=50331648        | CUDA kernel  |    76.913 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   147.479 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA kernel  |   153.083 us
```

## Torch Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521/bench_sm120_torch_matmul.log
```

## LibTorch C++ Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521/bench_sm120_libtorch_matmul.log
```

## CuTeDSL Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521/bench_sm120_cutedsl_matmul.log
```

## Triton Matmul Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521/bench_sm120_triton_matmul.log
```

## Torch Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521/bench_sm120_torch_attention.log
```

## cuDNN Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521/bench_sm120_cudnn_attention.log
```

## Triton Attention Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521/bench_sm120_triton_attention.log
```

## Torch Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521/bench_sm120_torch_classifier.log
```

## Triton Classifier Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521/bench_sm120_triton_classifier.log
```

## Python Stack LayerNorm Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521/bench_sm120_layernorm_python_stacks.log
```

## Triton Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521/bench_sm120_triton_runtime.log
```

## Torch Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521/bench_sm120_torch_runtime.log
```

## LibTorch C++ Runtime Benchmarks

```text
missing: scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521/bench_sm120_libtorch_runtime.log
```

## LibTorch Trainer Link Probe

```text
missing: scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521/validate_libtorch_trainer_link.log
```

## Training Steps

```text
| use_master_weights    | disabled                                           |
| gelu_fusion           | 1                                                  |
val loss 11.033154
step    1/3 | loss 11.032358 (+nanz)| norm 22.1414 (+nanz)| lr 8.57e-07 | 2547.77 ms | 39.5% bf16 MFU | 205783 tok/s
step    2/3 | loss 10.958511 (+nanz)| norm 22.0969 (+nanz)| lr 1.71e-06 | 2521.33 ms | 39.9% bf16 MFU | 207941 tok/s
step    3/3 | loss 10.811323 (+nanz)| norm 21.1250 (+nanz)| lr 2.57e-06 | 2534.30 ms | 39.7% bf16 MFU | 207395 tok/s
val loss 10.609914
total average iteration time: 2527.814031 ms
```

## validate_sm120_round
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521 --write-scoreboard scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521/scoreboard-candidates.md --write-selected-backends scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521/selected-backends.json --write-promotion-candidates scratch/sm120_rounds/codex_sm120_combo_tk_dgelu_layernorm_bwd1_x3_20260521/promotion-candidates.json --require-manifest --require-stack-probe --require-correctness --require-benchmarks --require-training --forbid-checkpoints`

