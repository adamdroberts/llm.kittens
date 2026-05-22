# SM120 Round Metrics - codex_sm120_round_classifier_loss_only_x10_20260520

- artifact dir: `scratch/sm120_rounds/codex_sm120_round_classifier_loss_only_x10_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_classifier_loss_only_x10_20260520`
- git commit: `0f21747`

## Backend Stack Probe

| Stack | Status | Candidate use | Next action |
|---|---|---|---|
| ThunderKittens 2.0 | available | native TK kernels and current SM120 packed-QKV attention path | benchmark against cuBLASLt/plain CUDA by shape before promoting TK-only wins |
| Plain CUDA | available | plain CUDA baselines and C++ benchmarks | run the SM120 round on the RTX 5090 target for runtime timings |
| GPU runtime | available | runtime timing and correctness execution | confirm target device is RTX 5090 / sm_120 before promoting timings |
| cuBLAS | available | baseline GEMM comparison where cuBLASLt epilogues are not needed | add explicit cuBLAS benchmark/parity rows before selecting it over cuBLASLt |
| cuBLASLt | available | current SM120 GEMM baseline and fused GEMM epilogues | keep benchmark rows shape-specific; do not switch global defaults from one isolated win |
| cuDNN | missing | attention alternatives if local headers/libs and shape support line up | prototype as an opt-in benchmark first; current v1 build contract intentionally avoids -lcudnn |
| Triton | missing | attention, normalization, elementwise fusion, and GEMM candidates | install triton in the active environment before benchmarking this stack |
| CuTeDSL | missing | Blackwell GEMM and fused epilogue candidates | install nvidia-cutlass-dsl in the active environment before benchmarking this stack |

## Backend Family-Stack Matrix

- detailed matrix: `scratch/sm120_rounds/codex_sm120_round_classifier_loss_only_x10_20260520/backend-stacks.json`

| Family | Baseline | Candidate | Fallback | Missing/blocked | Not applicable |
|---|---|---|---|---|---|
| `gemm_forward` | cuBLASLt | ThunderKittens 2.0, cuBLAS | Plain CUDA | Triton (missing), CuTeDSL (missing) | cuDNN |
| `gemm_forward_fused_gelu` | cuBLASLt | ThunderKittens 2.0, cuBLAS | Plain CUDA | Triton (missing), CuTeDSL (missing) | cuDNN |
| `gemm_backward_dinput` | cuBLASLt | ThunderKittens 2.0, cuBLAS | Plain CUDA | Triton (missing), CuTeDSL (missing) | cuDNN |
| `gemm_backward_dinput_fused_dgelu` | cuBLASLt | ThunderKittens 2.0, cuBLAS | Plain CUDA | Triton (missing), CuTeDSL (missing) | cuDNN |
| `gemm_backward_dweight` | cuBLASLt | ThunderKittens 2.0, cuBLAS | Plain CUDA | Triton (missing), CuTeDSL (missing) | cuDNN |
| `gemm_backward_dweight_accum` | cuBLASLt | ThunderKittens 2.0, cuBLAS | Plain CUDA | Triton (missing), CuTeDSL (missing) | cuDNN |
| `bias_add` | Plain CUDA | - | - | Triton (missing) | ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `bias_gradient_reduce` | Plain CUDA | - | - | Triton (missing) | ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `gelu_forward` | Plain CUDA | - | - | Triton (missing) | ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `gelu_backward` | Plain CUDA | - | - | Triton (missing) | ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `attention_forward` | ThunderKittens 2.0 | - | Plain CUDA | cuDNN (missing), Triton (missing) | cuBLAS, cuBLASLt, CuTeDSL |
| `attention_backward` | ThunderKittens 2.0 | - | Plain CUDA | cuDNN (missing), Triton (missing) | cuBLAS, cuBLASLt, CuTeDSL |
| `layernorm_forward` | Plain CUDA | - | - | ThunderKittens 2.0 (missing), Triton (missing) | cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `layernorm_fused_residual_forward` | Plain CUDA | - | - | ThunderKittens 2.0 (missing), Triton (missing) | cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `layernorm_backward` | Plain CUDA | - | - | ThunderKittens 2.0 (missing), Triton (missing) | cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `classifier_softmax_cross_entropy_dlogits` | Plain CUDA | - | - | Triton (missing) | ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `adamw` | Plain CUDA | - | - | Triton (missing) | ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `global_norm` | Plain CUDA | - | - | Triton (missing) | ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `encoder_forward` | Plain CUDA | - | - | Triton (missing) | ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `cuda_memset` | Plain CUDA | - | - | Triton (missing) | ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `cuda_copy_d2d` | Plain CUDA | - | - | Triton (missing) | ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, CuTeDSL |

## Round Manifest

- manifest: `scratch/sm120_rounds/codex_sm120_round_classifier_loss_only_x10_20260520/round-manifest.json`
- device arch: `SM120`
- build jobs: `4`
- changed paths: `455`

## Objective Coverage

| Family | Covered |
|---|---:|
| `gemm_forward` | `True` |
| `gemm_forward_fused_gelu` | `True` |
| `gemm_backward_dinput` | `True` |
| `gemm_backward_dinput_fused_dgelu` | `True` |
| `gemm_backward_dweight` | `True` |
| `gemm_backward_dweight_accum` | `True` |
| `bias_add` | `True` |
| `bias_gradient_reduce` | `True` |
| `gelu_forward` | `True` |
| `gelu_backward` | `True` |
| `attention_forward` | `True` |
| `attention_backward` | `True` |
| `layernorm_forward` | `True` |
| `layernorm_fused_residual_forward` | `True` |
| `layernorm_backward` | `True` |
| `classifier_softmax_cross_entropy_dlogits` | `True` |
| `adamw` | `True` |
| `global_norm` | `True` |
| `encoder_forward` | `True` |
| `cuda_memset` | `True` |
| `cuda_copy_d2d` | `True` |

## GEMM Shape Coverage

| Pass | Shape | Covered |
|---|---|---:|
| `fwd` | qkv (`qkv`) | `True` |
| `fwd` | attention projection (`attproj`) | `True` |
| `fwd` | MLP projection (`fcproj`) | `True` |
| `fwd` | LM-head (`lmhead`) | `True` |
| `fwd+gelu` | MLP up (`fc`) | `True` |
| `dInp` | qkv (`qkv`) | `True` |
| `dInp` | attention projection (`attproj`) | `True` |
| `dInp` | MLP up (`fc`) | `True` |
| `dInp` | MLP projection (`fcproj`) | `True` |
| `dInp` | LM-head (`lmhead`) | `True` |
| `dInp+dGeLU` | MLP projection (`fcproj`) | `True` |
| `dW` | qkv (`qkv`) | `True` |
| `dW` | attention projection (`attproj`) | `True` |
| `dW` | MLP up (`fc`) | `True` |
| `dW` | MLP projection (`fcproj`) | `True` |
| `dW` | LM-head (`lmhead`) | `True` |
| `dW+accum` | qkv (`qkv`) | `True` |
| `dW+accum` | attention projection (`attproj`) | `True` |
| `dW+accum` | MLP up (`fc`) | `True` |
| `dW+accum` | MLP projection (`fcproj`) | `True` |
| `dW+accum` | LM-head (`lmhead`) | `True` |

## GEMM Provider Coverage

| Pass | Shape | Provider | Covered |
|---|---|---|---:|
| `fwd` | qkv (`qkv`) | ThunderKittens | `True` |
| `fwd` | qkv (`qkv`) | cuBLASLt | `True` |
| `fwd` | qkv (`qkv`) | cuBLAS | `True` |
| `fwd` | attention projection (`attproj`) | ThunderKittens | `True` |
| `fwd` | attention projection (`attproj`) | cuBLASLt | `True` |
| `fwd` | attention projection (`attproj`) | cuBLAS | `True` |
| `fwd` | MLP projection (`fcproj`) | ThunderKittens | `True` |
| `fwd` | MLP projection (`fcproj`) | cuBLASLt | `True` |
| `fwd` | MLP projection (`fcproj`) | cuBLAS | `True` |
| `fwd` | LM-head (`lmhead`) | ThunderKittens | `True` |
| `fwd` | LM-head (`lmhead`) | cuBLASLt | `True` |
| `fwd` | LM-head (`lmhead`) | cuBLAS | `True` |
| `fwd+gelu` | MLP up (`fc`) | ThunderKittens | `True` |
| `fwd+gelu` | MLP up (`fc`) | cuBLASLt | `True` |
| `fwd+gelu` | MLP up (`fc`) | cuBLAS | `True` |
| `dInp` | qkv (`qkv`) | ThunderKittens | `True` |
| `dInp` | qkv (`qkv`) | cuBLASLt | `True` |
| `dInp` | qkv (`qkv`) | cuBLAS | `True` |
| `dInp` | attention projection (`attproj`) | ThunderKittens | `True` |
| `dInp` | attention projection (`attproj`) | cuBLASLt | `True` |
| `dInp` | attention projection (`attproj`) | cuBLAS | `True` |
| `dInp` | MLP up (`fc`) | ThunderKittens | `True` |
| `dInp` | MLP up (`fc`) | cuBLASLt | `True` |
| `dInp` | MLP up (`fc`) | cuBLAS | `True` |
| `dInp` | MLP projection (`fcproj`) | ThunderKittens | `True` |
| `dInp` | MLP projection (`fcproj`) | cuBLASLt | `True` |
| `dInp` | MLP projection (`fcproj`) | cuBLAS | `True` |
| `dInp` | LM-head (`lmhead`) | ThunderKittens | `True` |
| `dInp` | LM-head (`lmhead`) | cuBLASLt | `True` |
| `dInp` | LM-head (`lmhead`) | cuBLAS | `True` |
| `dInp+dGeLU` | MLP projection (`fcproj`) | ThunderKittens | `True` |
| `dInp+dGeLU` | MLP projection (`fcproj`) | cuBLASLt | `True` |
| `dInp+dGeLU` | MLP projection (`fcproj`) | cuBLAS | `True` |
| `dW` | qkv (`qkv`) | ThunderKittens | `True` |
| `dW` | qkv (`qkv`) | cuBLASLt | `True` |
| `dW` | qkv (`qkv`) | cuBLAS | `True` |
| `dW` | attention projection (`attproj`) | ThunderKittens | `True` |
| `dW` | attention projection (`attproj`) | cuBLASLt | `True` |
| `dW` | attention projection (`attproj`) | cuBLAS | `True` |
| `dW` | MLP up (`fc`) | ThunderKittens | `True` |
| `dW` | MLP up (`fc`) | cuBLASLt | `True` |
| `dW` | MLP up (`fc`) | cuBLAS | `True` |
| `dW` | MLP projection (`fcproj`) | ThunderKittens | `True` |
| `dW` | MLP projection (`fcproj`) | cuBLASLt | `True` |
| `dW` | MLP projection (`fcproj`) | cuBLAS | `True` |
| `dW` | LM-head (`lmhead`) | ThunderKittens | `True` |
| `dW` | LM-head (`lmhead`) | cuBLASLt | `True` |
| `dW` | LM-head (`lmhead`) | cuBLAS | `True` |
| `dW+accum` | qkv (`qkv`) | ThunderKittens | `True` |
| `dW+accum` | qkv (`qkv`) | cuBLASLt | `True` |
| `dW+accum` | qkv (`qkv`) | cuBLAS | `True` |
| `dW+accum` | attention projection (`attproj`) | ThunderKittens | `True` |
| `dW+accum` | attention projection (`attproj`) | cuBLASLt | `True` |
| `dW+accum` | attention projection (`attproj`) | cuBLAS | `True` |
| `dW+accum` | MLP up (`fc`) | ThunderKittens | `True` |
| `dW+accum` | MLP up (`fc`) | cuBLASLt | `True` |
| `dW+accum` | MLP up (`fc`) | cuBLAS | `True` |
| `dW+accum` | MLP projection (`fcproj`) | ThunderKittens | `True` |
| `dW+accum` | MLP projection (`fcproj`) | cuBLASLt | `True` |
| `dW+accum` | MLP projection (`fcproj`) | cuBLAS | `True` |
| `dW+accum` | LM-head (`lmhead`) | ThunderKittens | `True` |
| `dW+accum` | LM-head (`lmhead`) | cuBLASLt | `True` |
| `dW+accum` | LM-head (`lmhead`) | cuBLAS | `True` |

## Baseline Provider Coverage

| Family | Baseline provider | Covered |
|---|---|---:|
| `gemm_forward` | `cublaslt` | `True` |
| `gemm_forward_fused_gelu` | `cublaslt` | `True` |
| `gemm_backward_dinput` | `cublaslt` | `True` |
| `gemm_backward_dinput_fused_dgelu` | `cublaslt` | `True` |
| `gemm_backward_dweight` | `cublaslt` | `True` |
| `gemm_backward_dweight_accum` | `cublaslt` | `True` |
| `bias_add` | `cuda` | `True` |
| `bias_gradient_reduce` | `cuda` | `True` |
| `gelu_forward` | `cuda` | `True` |
| `gelu_backward` | `cuda` | `True` |
| `attention_forward` | `tk` | `True` |
| `attention_backward` | `tk` | `True` |
| `layernorm_forward` | `cuda` | `True` |
| `layernorm_fused_residual_forward` | `cuda` | `True` |
| `layernorm_backward` | `cuda` | `True` |
| `classifier_softmax_cross_entropy_dlogits` | `cuda` | `True` |
| `adamw` | `cuda` | `True` |
| `global_norm` | `cuda` | `True` |
| `encoder_forward` | `cuda` | `True` |
| `cuda_memset` | `cuda` | `True` |
| `cuda_copy_d2d` | `cuda` | `True` |

## Benchmark Candidates

| Suite | Kernel | Shape | Stack | Time (us) |
|---|---|---|---|---:|
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1071.530 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1064.700 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1467.120 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1086.840 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1014.750 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1012.340 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1456.780 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1111.670 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1035.240 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1499.700 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1110.980 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1005.770 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 376.360 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 369.190 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 512.660 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 381.050 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 366.750 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 366.000 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 544.060 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 374.320 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 330.760 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 558.020 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 379.220 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 331.960 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK fused | 1585.130 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK explicit | 1977.220 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1495.770 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS explicit | 2528.400 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1449.350 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1343.870 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1328.510 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1741.960 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1486.300 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1308.790 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1748.070 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1524.780 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1333.220 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1418.450 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1351.820 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1622.610 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1475.450 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1377.570 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1371.880 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1777.270 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt fused | 1849.390 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt explicit | 2190.270 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS explicit | 2132.780 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1709.620 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1463.830 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1309.340 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1749.190 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1472.600 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1331.490 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 27767.840 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 22331.270 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22414.450 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 23917.100 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21865.230 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21259.780 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26129.520 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20939.070 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21184.390 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26020.970 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20917.190 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21211.000 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 788.432 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 2718.345 |
| layernorm | forward | `N=65536 C=768` | CUDA | 139.323 |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 279.655 |
| layernorm | backward | `N=65536 C=768` | CUDA | 286.878 |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 102.794 |
| runtime | bias_add | `BT=65536 OC=3072` | CUDA | 602.185 |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 535.167 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | CUDA | 795.615 |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 247.509 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 3978.829 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 9090.682 |
| runtime | global_norm_squared | `params=124475904` | CUDA | 186.341 |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1849.824 |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 86.755 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA runtime | 62.598 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 133.446 |

## Training Smoke

- use_master_weights: `disabled`
- gelu_fusion: `1`
- total average iteration time: `2551.284 ms`
- final val loss: `9.483727`
- final step: `10/10`, loss `9.588612`, `2566.64 ms`, `205574 tok/s`

| Step | Loss | Norm | LR | Time (ms) | Tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.032358 | 22.1414 | 8.57e-07 | 2575.63 | 203557 |
| 2 | 10.958507 | 22.0968 | 1.71e-06 | 2536.03 | 206736 |
| 3 | 10.811316 | 21.1251 | 2.57e-06 | 2655.50 | 201966 |
| 4 | 10.610130 | 18.7014 | 3.43e-06 | 2494.63 | 204841 |
| 5 | 10.392586 | 15.0184 | 4.29e-06 | 2549.32 | 205061 |
| 6 | 10.186255 | 12.0843 | 5.14e-06 | 2522.63 | 205674 |
| 7 | 10.010621 | 10.2002 | 6e-06 | 2536.06 | 205874 |
| 8 | 9.855870 | 8.7905 | 6.86e-06 | 2524.39 | 206175 |
| 9 | 9.719423 | 7.4665 | 7.71e-06 | 2576.35 | 205777 |
| 10 | 9.588612 | 6.3099 | 8.57e-06 | 2566.64 | 205574 |

