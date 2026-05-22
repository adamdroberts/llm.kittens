# SM120 Optimization Round

- run label: `codex_sm120_round_python_stacks_selection_20260520_rerun`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun`
- train output dir: `log124M/5090_S_codex_sm120_round_python_stacks_selection_20260520_rerun`
- max steps: `3`
- python: `/home/adam/miniconda3/envs/llm-kittens/bin/python`
- git commit: `0f21747`
- working tree: `475` changed paths

## Environment

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
Wed May 20 22:26:03 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:E1:00.0  On |                  N/A |
|100%   27C    P5             56W /  575W |     687MiB /  32607MiB |      1%      Default |
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
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/probe_sm120_backend_stacks.py --json-out scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/backend-stacks.json --markdown-out scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/backend-stacks.md`


## build
Command: `make -j 4 test_matmul test_attention test_layernorm test_bias test_gelu test_fused_classifier test_encoder test_adamw test_global_norm bench_sm120_matmul bench_sm120_attention bench_sm120_layernorm bench_sm120_runtime train_gpt2cu DEVICE_ARCH=SM120 NO_MULTI_GPU=1 NO_USE_MPI=1`


## write_sm120_round_manifest
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/write_sm120_round_manifest.py --json-out scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/round-manifest.json --markdown-out scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/round-manifest.md --run-label codex_sm120_round_python_stacks_selection_20260520_rerun --artifact-dir scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun --train-out-dir log124M/5090_S_codex_sm120_round_python_stacks_selection_20260520_rerun --max-steps 3 --device-arch SM120 --build-jobs 4 --no-multi-gpu 1 --no-use-mpi 1 --run-stack-probe 1 --run-correctness 0 --run-benchmarks 1 --run-python-stack-benchmarks 1 --run-training 0 --keep-checkpoints 0`


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


## bench_sm120_torch_attention
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_attention.py --repeats 7 --warmup 3`


## bench_sm120_torch_classifier
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_classifier.py --repeats 7 --warmup 3`


## bench_sm120_layernorm_python_stacks
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_layernorm.py --rows 65536 --cols 768 3072 --repeats 7 --warmup 3`


## bench_sm120_triton_runtime
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/triton/bench_sm120_runtime.py --repeats 7 --warmup 3`


## bench_sm120_torch_runtime
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/bench_sm120_torch_runtime.py --repeats 7 --warmup 3`


## Correctness Markers

```text
[test_matmul] missing: scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/test_matmul.log
[test_attention] missing: scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/test_attention.log
[test_layernorm] missing: scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/test_layernorm.log
[test_bias] missing: scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/test_bias.log
[test_gelu] missing: scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/test_gelu.log
[test_fused_classifier] missing: scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/test_fused_classifier.log
[test_encoder] missing: scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/test_encoder.log
[test_adamw] missing: scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/test_adamw.log
[test_global_norm] missing: scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/test_global_norm.log
```

## Matmul Benchmarks

```text
  fwd      TK   1069.91 us | cuBLASLt   1063.34 us | cuBLAS   1413.42 us | TK/cuBLASLt 1.01x
  dInp   TK   1087.30 us | cuBLASLt   1031.03 us | cuBLAS   1011.63 us | TK/cuBLASLt 1.05x
  dW     TK   1455.71 us | cuBLASLt   1113.79 us | cuBLAS   1031.07 us | TK/cuBLASLt 1.31x
  dW+accum TK   1474.54 us | cuBLASLt   1108.04 us | cuBLAS   1033.19 us | TK/cuBLASLt 1.33x
  fwd      TK    374.62 us | cuBLASLt    369.76 us | cuBLAS    483.05 us | TK/cuBLASLt 1.01x
  dInp   TK    383.24 us | cuBLASLt    365.81 us | cuBLAS    363.40 us | TK/cuBLASLt 1.05x
  dW     TK    538.78 us | cuBLASLt    375.21 us | cuBLAS    326.79 us | TK/cuBLASLt 1.44x
  dW+accum TK    542.82 us | cuBLASLt    399.44 us | cuBLAS    330.08 us | TK/cuBLASLt 1.36x
  fwd+GeLU TK fused   1555.63 us | TK explicit   2006.49 us | cuBLASLt   1517.54 us | cuBLAS explicit   2450.27 us | explicit/cuBLASLt 1.32x
  dInp   TK   1476.58 us | cuBLASLt   1344.91 us | cuBLAS   1349.72 us | TK/cuBLASLt 1.10x
  dW     TK   1744.07 us | cuBLASLt   1538.38 us | cuBLAS   1309.68 us | TK/cuBLASLt 1.13x
  dW+accum TK   1776.37 us | cuBLASLt   1491.03 us | cuBLAS   1355.09 us | TK/cuBLASLt 1.19x
  fwd      TK   1431.67 us | cuBLASLt   1406.55 us | cuBLAS   1620.90 us | TK/cuBLASLt 1.02x
  dInp   TK   1476.49 us | cuBLASLt   1385.17 us | cuBLAS   1413.37 us | TK/cuBLASLt 1.07x
  dInp+dGeLU TK   1835.32 us | cuBLASLt fused   1843.00 us | cuBLASLt explicit   2189.76 us | cuBLAS explicit   2202.91 us | explicit/fused 1.19x
  dW     TK   1729.09 us | cuBLASLt   1513.04 us | cuBLAS   1309.74 us | TK/cuBLASLt 1.14x
  dW+accum TK   1780.47 us | cuBLASLt   1489.36 us | cuBLAS   1328.88 us | TK/cuBLASLt 1.20x
  fwd      TK  27889.37 us | cuBLASLt  25633.51 us | cuBLAS  24966.83 us | TK/cuBLASLt 1.09x
  dInp   TK  25379.78 us | cuBLASLt  23086.79 us | cuBLAS  22308.37 us | TK/cuBLASLt 1.10x
  dW     TK  27646.83 us | cuBLASLt  22131.21 us | cuBLAS  22432.05 us | TK/cuBLASLt 1.25x
  dW+accum TK  27758.51 us | cuBLASLt  22043.05 us | cuBLAS  22252.33 us | TK/cuBLASLt 1.26x
```

## Attention Benchmarks

```text
Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 825.886 us
Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2892.588 us
```

## LayerNorm Benchmarks

```text
LayerNorm Forward (N=65536, C=768): 148.137 us
LayerNorm FusedResidualForward (N=65536, C=768): 295.547 us
LayerNorm Backward (N=65536, C=768): 304.621 us
```

## Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | CUDA         |    86.808 us
bias_add                       | BT=65536 OC=3072             | CUDA         |   577.804 us
gelu_forward                   | BT=65536 C=3072              | CUDA         |   572.824 us
gelu_backward_inplace          | BT=65536 C=3072              | CUDA         |   854.545 us
bias_grad_reduce               | BT=65536 OC=768              | CUDA         |    24.507 us
bias_grad_reduce               | BT=65536 OC=2304             | CUDA         |   218.741 us
bias_grad_reduce               | BT=65536 OC=3072             | CUDA         |   261.328 us
fused_classifier_loss          | B=64 T=1024 V=50257 P=50304  | CUDA         |  4354.061 us
fused_classifier               | B=64 T=1024 V=50257 P=50304  | CUDA         |  9540.025 us
cuda_memset                    | logits_elems=3296722944      | CUDA runtime |  4455.706 us
cuda_copy_d2d                  | logits_elems=3296722944      | CUDA runtime |  9462.322 us
global_norm_squared            | params=124475904             | CUDA         |   198.059 us
adamw_update                   | params=124475904 no-master   | CUDA         |  1926.957 us
encoder_forward                | B=64 T=1024 C=768            | CUDA         |    91.999 us
cuda_memset                    | hidden_elems=50331648        | CUDA runtime |    67.214 us
cuda_copy_d2d                  | hidden_elems=50331648        | CUDA runtime |   143.239 us
```

## Torch Matmul Benchmarks

```text
Torch matmul device: NVIDIA GeForce RTX 5090; capability=sm_120
qkv          M=65536 N=2304 K=768 bias=1 gelu=0
  fwd      Torch   1468.31 us
  dInp   Torch   1023.35 us
  dW     Torch   1055.74 us
  dW+accum Torch   1017.82 us
attproj      M=65536 N=768 K=768 bias=1 gelu=0
  fwd      Torch    520.31 us
  dInp   Torch    373.38 us
  dW     Torch    340.49 us
  dW+accum Torch    348.06 us
fc           M=65536 N=3072 K=768 bias=1 gelu=1
  fwd+GeLU Torch   2498.55 us
  dInp   Torch   1491.92 us
  dW     Torch   1477.07 us
  dW+accum Torch   1483.65 us
fcproj       M=65536 N=768 K=3072 bias=1 gelu=0
  fwd      Torch   1748.62 us
  dInp   Torch   1534.72 us
  dInp+dGeLU Torch  28723.12 us
  dW     Torch   1366.90 us
  dW+accum Torch   1420.92 us
lmhead       M=65536 N=50304 K=768 bias=0 gelu=0
  fwd      Torch  22502.40 us
  dInp   Torch  21653.44 us
  dW     Torch  21231.94 us
  dW+accum Torch  21523.68 us
```

## Torch Attention Benchmarks

```text
Torch Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 584.058 us
Torch Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 2229.728 us
TorchPacked Attention Forward (B=64, T=1024, C=768, NH=12, HS=64): 1069.602 us
TorchPacked Attention Backward (B=64, T=1024, C=768, NH=12, HS=64): 4144.983 us
```

## Python Stack LayerNorm Benchmarks

```text
Triton LayerNorm device: NVIDIA GeForce RTX 5090; capability=sm_120
Triton LayerNorm Forward (N=65536, C=768): 187.200 us (y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm ForwardNative (N=65536, C=768): 155.392 us (y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm ForwardWithStats (N=65536, C=768): 2232.960 us (y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
Triton LayerNorm FusedResidualForward (N=65536, C=768): 316.288 us (residual_diff=0.000000, y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm FusedResidualForwardNative (N=65536, C=768): 331.584 us (residual_diff=0.000000, y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm FusedResidualForwardWithStats (N=65536, C=768): 3398.144 us (residual_diff=0.000000, y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
Triton LayerNorm Forward (N=65536, C=3072): 581.472 us (y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm ForwardNative (N=65536, C=3072): 557.504 us (y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm ForwardWithStats (N=65536, C=3072): 9103.712 us (y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
Triton LayerNorm FusedResidualForward (N=65536, C=3072): 1122.688 us (residual_diff=0.000000, y_diff=0.031250, mean_diff=0.000000, rstd_diff=0.000000)
Torch LayerNorm FusedResidualForwardNative (N=65536, C=3072): 1327.104 us (residual_diff=0.000000, y_diff=0.031250; no saved mean/rstd)
Torch LayerNorm FusedResidualForwardWithStats (N=65536, C=3072): 13410.368 us (residual_diff=0.000000, y_diff=0.000000, mean_diff=0.000000, rstd_diff=0.000000)
```

## Torch Runtime Benchmarks

```text
bias_add                       | BT=65536 OC=768              | Torch        |   137.793 us
bias_add                       | BT=65536 OC=3072             | Torch        |   538.298 us
gelu_forward                   | BT=65536 C=3072              | Torch        |   538.702 us
gelu_backward_inplace          | BT=65536 C=3072              | Torch        | 27275.003 us
bias_grad_reduce               | BT=65536 OC=768              | Torch        |   322.966 us
bias_grad_reduce               | BT=65536 OC=2304             | Torch        |  1040.090 us
bias_grad_reduce               | BT=65536 OC=3072             | Torch        |  1326.208 us
global_norm_squared            | params=124475904             | Torch        |  2348.851 us
adamw_update_bf16_state        | params=124475904 no-master   | Torch        |  1215.200 us
adamw_update                   | params=124475904 no-master fp32-state | Torch        |  7449.536 us
encoder_forward                | B=64 T=1024 C=768            | Torch        |   202.763 us
cuda_memset                    | hidden_elems=50331648        | Torch        |    63.554 us
cuda_copy_d2d                  | hidden_elems=50331648        | Torch        |   134.291 us
cuda_memset                    | logits_elems=3296722944      | Torch        |  4190.176 us
cuda_copy_d2d                  | logits_elems=3296722944      | Torch        |  8849.952 us
```

## Training Steps

```text
missing: scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/train_gpt2cu.log
```

## validate_sm120_round
Command: `/home/adam/miniconda3/envs/llm-kittens/bin/python dev/validate_sm120_round.py --round-dir scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun --write-scoreboard scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/scoreboard-candidates.md --require-manifest --require-stack-probe --require-benchmarks`

