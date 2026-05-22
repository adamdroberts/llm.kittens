# SM120 Round Metrics - codex_sm120_round_adamw_nomaster_specialized_x10_20260520

- artifact dir: `scratch/sm120_rounds/codex_sm120_round_adamw_nomaster_specialized_x10_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_adamw_nomaster_specialized_x10_20260520`
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

- detailed matrix: `scratch/sm120_rounds/codex_sm120_round_adamw_nomaster_specialized_x10_20260520/backend-stacks.json`

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

- manifest: `scratch/sm120_rounds/codex_sm120_round_adamw_nomaster_specialized_x10_20260520/round-manifest.json`
- device arch: `SM120`
- build jobs: `4`
- changed paths: `467`

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

## Runtime Shape Coverage

| Kernel | Shape | Covered |
|---|---|---:|
| `bias_add` | `BT=65536 OC=768` | `True` |
| `bias_add` | `BT=65536 OC=3072` | `True` |
| `bias_grad_reduce` | `BT=65536 OC=768` | `True` |
| `bias_grad_reduce` | `BT=65536 OC=2304` | `True` |
| `bias_grad_reduce` | `BT=65536 OC=3072` | `True` |
| `cuda_memset` | `hidden_elems=50331648` | `True` |
| `cuda_memset` | `logits_elems=3296722944` | `True` |
| `cuda_copy_d2d` | `hidden_elems=50331648` | `True` |
| `cuda_copy_d2d` | `logits_elems=3296722944` | `True` |

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
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1115.350 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1041.940 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1413.380 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1088.010 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1010.160 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1072.140 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1489.790 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1145.700 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 989.700 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1469.210 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1113.410 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 997.820 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 399.560 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 369.390 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 486.030 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 380.880 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 366.990 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 365.820 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 550.150 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 374.040 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 326.700 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 545.700 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 377.970 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 330.250 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK fused | 1561.270 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK explicit | 1983.430 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1520.280 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS explicit | 2486.140 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1450.800 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1395.230 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1328.900 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1759.320 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1473.480 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1349.770 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1709.560 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1547.940 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1342.740 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1453.820 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1350.780 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1548.480 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1526.060 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1399.140 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1366.250 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1794.550 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt fused | 1815.350 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt explicit | 2158.170 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS explicit | 2191.290 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1743.400 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1466.230 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1309.150 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1726.360 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1516.340 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1314.040 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 27759.710 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 22349.750 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22150.560 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 24120.280 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21854.360 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21198.740 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26099.860 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20905.120 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21102.500 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26224.790 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21094.460 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21284.830 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 779.514 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 2741.231 |
| layernorm | forward | `N=65536 C=768` | CUDA | 139.285 |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 280.272 |
| layernorm | backward | `N=65536 C=768` | CUDA | 290.305 |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 103.867 |
| runtime | bias_add | `BT=65536 OC=3072` | CUDA | 547.238 |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 536.122 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | CUDA | 791.836 |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 29.325 |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 188.566 |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 247.074 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 4030.918 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 9119.930 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | 4225.831 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA runtime | 8888.729 |
| runtime | global_norm_squared | `params=124475904` | CUDA | 185.861 |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1855.027 |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 82.793 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA runtime | 64.911 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 133.931 |

## Training Smoke

- use_master_weights: `disabled`
- gelu_fusion: `1`
- total average iteration time: `2499.597 ms`
- final val loss: `9.483727`
- final step: `10/10`, loss `9.588612`, `2506.84 ms`, `209697 tok/s`

| Step | Loss | Norm | LR | Time (ms) | Tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.032358 | 22.1414 | 8.57e-07 | 2500.89 | 209640 |
| 2 | 10.958507 | 22.0968 | 1.71e-06 | 2491.90 | 210397 |
| 3 | 10.811316 | 21.1251 | 2.57e-06 | 2492.63 | 210365 |
| 4 | 10.610130 | 18.7014 | 3.43e-06 | 2496.46 | 210242 |
| 5 | 10.392586 | 15.0184 | 4.29e-06 | 2497.46 | 210157 |
| 6 | 10.186255 | 12.0843 | 5.14e-06 | 2501.77 | 210027 |
| 7 | 10.010621 | 10.2002 | 6e-06 | 2501.49 | 209944 |
| 8 | 9.855870 | 8.7905 | 6.86e-06 | 2504.14 | 209849 |
| 9 | 9.719423 | 7.4665 | 7.71e-06 | 2503.66 | 209783 |
| 10 | 9.588612 | 6.3099 | 8.57e-06 | 2506.84 | 209697 |

