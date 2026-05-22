# SM120 Round Metrics - codex_sm120_round_torch_attention_materialized_20260521

- artifact dir: `scratch/sm120_rounds/codex_sm120_round_torch_attention_materialized_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_torch_attention_materialized_20260521`
- git commit: `0f21747`

## Backend Stack Probe

| Stack | Status | Candidate use | Next action |
|---|---|---|---|
| ThunderKittens 2.0 | available | native TK kernels and current SM120 packed-QKV attention path | benchmark against cuBLASLt/plain CUDA by shape before promoting TK-only wins |
| Plain CUDA | available | plain CUDA baselines and C++ benchmarks | run the SM120 round on the RTX 5090 target for runtime timings |
| GPU runtime | available | runtime timing and correctness execution | confirm target device is RTX 5090 / sm_120 before promoting timings |
| cuBLAS | available | baseline GEMM comparison where cuBLASLt epilogues are not needed | add explicit cuBLAS benchmark/parity rows before selecting it over cuBLASLt |
| cuBLASLt | available | current SM120 GEMM baseline and fused GEMM epilogues | keep benchmark rows shape-specific; do not switch global defaults from one isolated win |
| cuDNN | available | attention alternatives through detected headers/libs; GPT-2 BF16 shape support still needs benchmark proof | prototype as an opt-in benchmark first; current v1 build contract intentionally avoids -lcudnn |
| Triton | available | attention, normalization, elementwise fusion, and GEMM candidates | add stack-specific parity tests before trainer promotion |
| Torch | available | PyTorch operator kernels for exact family-by-family backend comparisons | add stack-specific parity tests before trainer promotion |
| CuTeDSL | available | Blackwell GEMM and fused epilogue candidates | add stack-specific parity tests before trainer promotion |

## Backend Family-Stack Matrix

- detailed matrix: `scratch/sm120_rounds/codex_sm120_round_torch_attention_materialized_20260521/backend-stacks.json`

| Family | Baseline | Candidate | Fallback | Missing/blocked | Not applicable |
|---|---|---|---|---|---|
| `gemm_forward` | cuBLASLt | ThunderKittens 2.0, cuBLAS, Triton, Torch, CuTeDSL | Plain CUDA | - | cuDNN |
| `gemm_forward_fused_gelu` | cuBLASLt | ThunderKittens 2.0, cuBLAS, Triton, Torch, CuTeDSL | Plain CUDA | - | cuDNN |
| `gemm_backward_dinput` | cuBLASLt | ThunderKittens 2.0, cuBLAS, Triton, Torch, CuTeDSL | Plain CUDA | - | cuDNN |
| `gemm_backward_dinput_fused_dgelu` | cuBLASLt | ThunderKittens 2.0, cuBLAS, Triton, Torch, CuTeDSL | Plain CUDA | - | cuDNN |
| `gemm_backward_dweight` | cuBLASLt | ThunderKittens 2.0, cuBLAS, Triton, Torch, CuTeDSL | Plain CUDA | - | cuDNN |
| `gemm_backward_dweight_accum` | cuBLASLt | ThunderKittens 2.0, cuBLAS, Triton, Torch, CuTeDSL | Plain CUDA | - | cuDNN |
| `bias_add` | Plain CUDA | Triton, Torch | - | - | ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `bias_gradient_reduce` | Plain CUDA | Triton, Torch | - | - | ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `gelu_forward` | Plain CUDA | Triton, Torch | - | - | ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `gelu_backward` | Plain CUDA | Triton, Torch | - | - | ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `attention_forward` | ThunderKittens 2.0 | cuDNN, Triton, Torch | Plain CUDA | - | cuBLAS, cuBLASLt, CuTeDSL |
| `attention_backward` | ThunderKittens 2.0 | cuDNN, Triton, Torch | Plain CUDA | - | cuBLAS, cuBLASLt, CuTeDSL |
| `layernorm_forward` | Plain CUDA | Triton, Torch | - | ThunderKittens 2.0 (missing) | cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `layernorm_fused_residual_forward` | Plain CUDA | Triton, Torch | - | ThunderKittens 2.0 (missing) | cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `layernorm_backward` | Plain CUDA | Triton, Torch | - | ThunderKittens 2.0 (missing) | cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `classifier_softmax_cross_entropy_dlogits` | Plain CUDA | Triton, Torch | - | - | ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `adamw` | Plain CUDA | Triton, Torch | - | - | ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `global_norm` | Plain CUDA | Triton, Torch | - | - | ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `encoder_forward` | Plain CUDA | Triton, Torch | - | - | ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `cuda_memset` | Plain CUDA | Triton, Torch | - | - | ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, CuTeDSL |
| `cuda_copy_d2d` | Plain CUDA | Triton, Torch | - | - | ThunderKittens 2.0, cuBLAS, cuBLASLt, cuDNN, CuTeDSL |

## Round Manifest

- manifest: `scratch/sm120_rounds/codex_sm120_round_torch_attention_materialized_20260521/round-manifest.json`
- device arch: `SM120`
- build jobs: `4`
- changed paths: `487`

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

## LibTorch Runtime Shape Coverage

| Kernel | Shape | Covered |
|---|---|---:|
| `cuda_memset` | `hidden_elems=50331648` | `True` |
| `cuda_memset` | `logits_elems=3296722944` | `True` |
| `cuda_copy_d2d` | `hidden_elems=50331648` | `True` |
| `cuda_copy_d2d` | `logits_elems=3296722944` | `True` |

## LibTorch Runtime Parity Coverage

| Kernel | Shape | Covered |
|---|---|---:|
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

## Torch Objective Benchmark Coverage

| Suite | Kernel | Shape | Covered |
|---|---|---|---:|
| `attention` | `backward` | `B=64 T=1024 C=768 NH=12 HS=64` | `True` |
| `attention` | `forward` | `B=64 T=1024 C=768 NH=12 HS=64` | `True` |
| `layernorm` | `backward` | `N=65536 C=768` | `True` |
| `layernorm` | `forward` | `N=65536 C=768` | `True` |
| `layernorm` | `fused_residual_forward` | `N=65536 C=768` | `True` |
| `matmul` | `dInp` | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | `True` |
| `matmul` | `dInp` | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | `True` |
| `matmul` | `dInp` | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | `True` |
| `matmul` | `dInp` | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | `True` |
| `matmul` | `dInp` | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | `True` |
| `matmul` | `dInp+dGeLU` | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | `True` |
| `matmul` | `dW` | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | `True` |
| `matmul` | `dW` | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | `True` |
| `matmul` | `dW` | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | `True` |
| `matmul` | `dW` | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | `True` |
| `matmul` | `dW` | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | `True` |
| `matmul` | `dW+accum` | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | `True` |
| `matmul` | `dW+accum` | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | `True` |
| `matmul` | `dW+accum` | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | `True` |
| `matmul` | `dW+accum` | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | `True` |
| `matmul` | `dW+accum` | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | `True` |
| `matmul` | `fwd` | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | `True` |
| `matmul` | `fwd` | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | `True` |
| `matmul` | `fwd` | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | `True` |
| `matmul` | `fwd` | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | `True` |
| `matmul` | `fwd+gelu` | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | `True` |
| `runtime` | `adamw_update` | `params=124475904 no-master` | `True` |
| `runtime` | `bias_add` | `BT=65536 OC=3072` | `True` |
| `runtime` | `bias_add` | `BT=65536 OC=768` | `True` |
| `runtime` | `bias_grad_reduce` | `BT=65536 OC=2304` | `True` |
| `runtime` | `bias_grad_reduce` | `BT=65536 OC=3072` | `True` |
| `runtime` | `bias_grad_reduce` | `BT=65536 OC=768` | `True` |
| `runtime` | `cuda_copy_d2d` | `hidden_elems=50331648` | `True` |
| `runtime` | `cuda_copy_d2d` | `logits_elems=3296722944` | `True` |
| `runtime` | `cuda_memset` | `hidden_elems=50331648` | `True` |
| `runtime` | `cuda_memset` | `logits_elems=3296722944` | `True` |
| `runtime` | `encoder_forward` | `B=64 T=1024 C=768` | `True` |
| `runtime` | `fused_classifier` | `B=64 T=1024 V=50257 P=50304` | `True` |
| `runtime` | `fused_classifier_loss` | `B=64 T=1024 V=50257 P=50304` | `True` |
| `runtime` | `gelu_backward_inplace` | `BT=65536 C=3072` | `True` |
| `runtime` | `gelu_forward` | `BT=65536 C=3072` | `True` |
| `runtime` | `global_norm_squared` | `params=124475904` | `True` |

## Python Stack Benchmark Logs

| Log | Present |
|---|---:|
| `bench_sm120_torch_matmul.log` | `True` |
| `bench_sm120_cutedsl_matmul.log` | `True` |
| `bench_sm120_triton_matmul.log` | `True` |
| `bench_sm120_torch_attention.log` | `True` |
| `bench_sm120_cudnn_attention.log` | `True` |
| `bench_sm120_triton_attention.log` | `True` |
| `bench_sm120_torch_classifier.log` | `True` |
| `bench_sm120_triton_classifier.log` | `True` |
| `bench_sm120_layernorm_python_stacks.log` | `True` |
| `bench_sm120_triton_runtime.log` | `True` |
| `bench_sm120_torch_runtime.log` | `True` |
| `bench_sm120_libtorch_runtime.log` | `True` |

## Selected Backend Rows

| Suite | Kernel | Shape | Selected stack | Time (us) | Next stack | Next time (us) | Use scope | Decision status | Decision note |
|---|---|---|---|---:|---|---:|---|---|---|
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 2195.661 | cuDNN | 2375.994 | python separated-Q/K/V | layout_rewrite_only | Use Torch SDPA for already-separated Q/K/V experiments; packed trainer selection must compare TorchPacked. |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 555.498 | cuDNN | 685.798 | python separated-Q/K/V | layout_rewrite_only | Use Torch SDPA for already-separated Q/K/V experiments; packed trainer selection must compare TorchPacked. |
| layernorm | backward | `N=65536 C=3072` | CUDA | 1266.138 | Torch native | 1397.760 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | backward | `N=65536 C=768` | CUDA | 288.147 | Triton atomic FP32-grad | 360.160 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | backward_dinput | `N=65536 C=3072` | Triton dInput-only | 812.128 | Torch native | 845.120 | partial backward prototype | - | Triton dInput-only is useful for backward decomposition work but needs dweight/dbias before trainer promotion. |
| layernorm | backward_dinput | `N=65536 C=768` | Torch native | 222.112 | Triton dInput-only | 227.552 | partial backward prototype | partial_backward_only | Torch native dInput-only output-mask row is useful for backward decomposition work but needs dweight/dbias before trainer promotion. |
| layernorm | forward | `N=65536 C=3072` | CUDA | 543.194 | Torch native | 554.592 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | forward | `N=65536 C=768` | CUDA | 139.810 | Torch native | 154.880 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | fused_residual_forward | `N=65536 C=3072` | CUDA | 1084.727 | Triton | 1105.120 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 276.514 | Triton | 312.320 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 365.350 | cuBLASLt | 370.070 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1328.470 | Torch | 1356.870 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1372.020 | Torch | 1405.250 | C++ benchmark route | benchmark_context_flip | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21339.160 | Torch | 21718.140 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1007.100 | cuBLASLt | 1007.110 | C++ benchmark route | noise_floor_microbench_flip | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1789.270 | cuBLASLt fused | 1815.340 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 328.700 | Torch | 337.220 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1328.120 | cuBLAS | 1330.140 | operator prototype | rejected_same_session_refresh | Torch is the fastest observed operator row here; trainer use still needs an explicit libtorch/C++ integration gate. |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1354.620 | Torch | 1368.780 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20944.170 | cuBLAS | 21245.570 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 993.810 | Torch | 1004.900 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 332.000 | Torch | 347.010 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1313.760 | Torch | 1366.060 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1316.570 | Torch | 1392.770 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20992.150 | Torch | 21109.950 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 996.070 | Torch | 1017.320 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 375.370 | TK | 375.770 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1399.780 | TK | 1444.560 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22242.280 | cuBLASLt | 22273.020 | C++ benchmark route | rejected_trainer_smoke | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1044.710 | TK | 1072.970 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1494.690 | TK fused | 1564.700 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1835.683 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | 7315.776 | - | - | operator prototype | rejected_slower_than_trainer_baseline | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | adamw_update_bf16_state | `params=124475904 no-master` | Torch | 1196.518 | - | - | non-equivalent BF16-state reference | - | Torch fused AdamW is usable only for BF16 moment-state experiments; the trainer default uses FP32 moment buffers. |
| runtime | bias_add | `BT=65536 OC=3072` | Triton | 527.930 | CUDA | 528.993 | operator prototype | library_integration_not_justified | Use as a Triton comparison row until a trainer-callable integration beats the current provider. |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 96.825 | Triton | 132.389 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 186.634 | Torch | 967.494 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 245.650 | Torch | 1315.942 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 29.872 | Torch | 314.643 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | Torch C++ | 131.526 | CUDA runtime | 131.723 | C++ API prototype | library_integration_not_justified | LibTorch C++ API row proves a possible trainer-callable dependency path, but promotion still needs an explicit link gate and TinyStories smoke. |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch | 8665.152 | Torch C++ | 8686.496 | operator prototype | native_replacement_rejected | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch C++ | 60.011 | Torch | 60.038 | C++ API prototype | library_integration_not_justified | LibTorch C++ API row proves a possible trainer-callable dependency path, but promotion still needs an explicit link gate and TinyStories smoke. |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | 3913.728 | Torch | 3976.288 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 85.180 | Torch | 199.050 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 8937.452 | Triton | 22245.504 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 3948.493 | Triton | 8248.928 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | 770.520 | CUDA | 779.645 | operator prototype | rejected_same_session_refresh | Use as a Triton comparison row until a trainer-callable integration beats the current provider. |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 527.850 | Triton | 530.515 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | global_norm_squared | `params=124475904` | CUDA | 184.909 | Torch | 2273.318 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |

## Resolved Selected Backend Decisions

| Suite | Kernel | Shape | Selected stack | Status | Decision |
|---|---|---|---|---|---|
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| layernorm | backward_dinput | `N=65536 C=768` | Torch native | partial_backward_only | Do not promote the dInput-only Torch row. The full Torch native backward with dweight/dbias is slower than CUDA, the full Triton atomic prototype is also slower plus has an FP32 gradient-buffer contract mismatch, and the focused Torch dInput-plus-BF16-grads hybrid is much slower than the CUDA full-backward baseline. |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | rejected_x10_selector | Do not broaden the SM120 direct-cuBLAS dInput selector to attention projection; the x10 trainer stability round regressed, so the source default remains the huge-N LM-head-only direct-cuBLAS dInput route. |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | rejected_x10_selector | Do not broaden the SM120 direct-cuBLAS dInput selector to the GPT-2 MLP-up row. The microbench row can favor cuBLAS, but the broader direct-cuBLAS dInput selector regressed in the x10 TinyStories stability gate, while the stream-sync default keeps cuBLASLt for this row and improves the trainer smoke. |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | benchmark_context_flip | Keep the training-backed current native row on cuBLASLt for the GPT-2 MLP projection dInput path. The earlier current-source x10 native round selected cuBLAS, but the stream-sync x10 native round and optional-stack benchmark both select cuBLASLt; the prior flip is now resolved by the newer trainer-backed evidence. |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | noise_floor_microbench_flip | Do not promote the qkv dInput cuBLAS microbench flip as a trainer default without a trainer smoke. The refreshed benchmark-only round picked cuBLAS by about 0.2%, while the stable x10 selection artifact has cuBLASLt ahead for the same row. |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | rejected_x10_selector | Keep the cuBLASLt fused dGELU trainer route as the default. The opt-in TK exact-dGELU selector now passes correctness and has a focused row win, but its x10 TinyStories stability round regressed versus the current stable default. |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | rejected_same_session_refresh | Rejected for libtorch/native replacement work; focused uncontended and standalone LibTorch C++ timing showed current C++ cuBLAS was faster than Torch for MLP-up dWeight. |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | rejected_trainer_smoke | Keep LM-head forward on cuBLASLt; the opt-in direct-cuBLAS forward selector passed focused gates but regressed in TinyStories trainer timing. |
| runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | rejected_slower_than_trainer_baseline | Rejected for trainer promotion; Torch fp32-state AdamW is materially slower than the CUDA trainer route. |
| runtime | bias_add | `BT=65536 OC=3072` | Triton | library_integration_not_justified | Keep as operator evidence; the refreshed Triton edge is about 0.2% and not enough to justify a trainer-callable Triton route. |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | Torch C++ | library_integration_not_justified | Keep as C++ API feasibility evidence; the LibTorch route is tie-range and does not justify a trainer dependency. |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch | native_replacement_rejected | Keep as operator evidence; raw-pointer LibTorch preserves most of the copy edge, but Python Torch remains the fastest observed row and a linked trainer smoke is still required before promotion. |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch C++ | library_integration_not_justified | Keep as C++ API feasibility evidence; the LibTorch route is tie-range and does not justify a trainer dependency. |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | rejected_same_session_refresh | Rejected for trainer promotion; refreshed CUDA GELU backward was faster than Triton. |

## Promotion Backlog

No active promotion candidates remain after applying `dev/sm120_promotion_decisions.json`.

## Resolved Promotion Decisions

| Class | Suite | Kernel | Shape | Winning stack | Status | Decision |
|---|---|---|---|---|---|---|
| native/codegen integration | runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | rejected_same_session_refresh | Rejected for trainer promotion; refreshed CUDA GELU backward was faster than Triton. |
| native/codegen integration | runtime | bias_add | `BT=65536 OC=3072` | Triton | library_integration_not_justified | Keep as operator evidence; the refreshed Triton edge is about 0.2% and not enough to justify a trainer-callable Triton route. |
| library integration | runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | rejected_slower_than_trainer_baseline | Rejected for trainer promotion; Torch fp32-state AdamW is materially slower than the CUDA trainer route. |
| library integration | runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch | native_replacement_rejected | Keep as operator evidence; raw-pointer LibTorch preserves most of the copy edge, but Python Torch remains the fastest observed row and a linked trainer smoke is still required before promotion. |
| library integration | matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | rejected_same_session_refresh | Rejected for libtorch/native replacement work; focused uncontended and standalone LibTorch C++ timing showed current C++ cuBLAS was faster than Torch for MLP-up dWeight. |
| library integration | runtime | cuda_copy_d2d | `hidden_elems=50331648` | Torch C++ | library_integration_not_justified | Keep as C++ API feasibility evidence; the LibTorch route is tie-range and does not justify a trainer dependency. |
| library integration | runtime | cuda_memset | `hidden_elems=50331648` | Torch C++ | library_integration_not_justified | Keep as C++ API feasibility evidence; the LibTorch route is tie-range and does not justify a trainer dependency. |
| layout rewrite | attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| layout rewrite | attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| reference/state gap | layernorm | backward_dinput | `N=65536 C=768` | Torch native | partial_backward_only | Do not promote the dInput-only Torch row. The full Torch native backward with dweight/dbias is slower than CUDA, the full Triton atomic prototype is also slower plus has an FP32 gradient-buffer contract mismatch, and the focused Torch dInput-plus-BF16-grads hybrid is much slower than the CUDA full-backward baseline. |
| non-trainer shape | layernorm | backward_dinput | `N=65536 C=3072` | Triton dInput-only | non_trainer_shape | Not an active trainer-promotion target for GPT-2 124M. |
| contract mismatch | runtime | adamw_update_bf16_state | `params=124475904 no-master` | Torch | contract_mismatch | Not active until the candidate matches the trainer state contract. |

## Attention Route Totals

| Shape | Stack | Scope | Trainer-layout | Forward (us) | Backward (us) | Total (us) | Complete | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | packed trainer-layout route | True | 778.909 | 2728.503 | 3507.412 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | cuDNNPacked | packed trainer-layout route | True | 802.810 | 3514.362 | 4317.172 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | TorchPacked | packed trainer-layout route | True | 1068.491 | 4067.501 | 5135.992 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | TorchMaterializedPacked | packed trainer-layout route | True | 1260.536 | 4195.885 | 5456.421 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | TritonPacked | packed trainer-layout route | True | 2196.746 | - | - | False | packed attention backward is not implemented in this Triton prototype |
| `B=64 T=1024 C=768 NH=12 HS=64` | Torch | separated Q/K/V reference route | False | 555.498 | 2195.661 | 2751.159 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | cuDNN | separated Q/K/V reference route | False | 685.798 | 2375.994 | 3061.792 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | Triton | separated Q/K/V reference route | False | 2074.990 | - | - | False | attention backward is not implemented in this Triton prototype |

## Benchmark Candidates

| Suite | Kernel | Shape | Stack | Time (us) |
|---|---|---|---|---:|
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1072.970 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1044.710 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1456.830 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1087.090 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1007.110 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1007.100 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1487.360 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1129.730 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 993.810 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1484.350 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1119.180 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 996.070 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 375.770 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 375.370 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 485.400 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 381.300 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 370.070 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 365.350 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 537.170 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 380.390 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 328.700 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 537.810 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 388.090 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 332.000 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK fused | 1564.700 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK explicit | 1995.760 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1494.690 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS explicit | 2467.540 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1448.880 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1367.640 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1328.470 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1749.910 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1493.230 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1330.140 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1750.580 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1479.880 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1313.760 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1444.560 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1399.780 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1548.910 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1476.480 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1372.020 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1415.140 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1789.270 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt fused | 1815.340 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt explicit | 2197.960 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS explicit | 2175.720 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1742.280 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1470.470 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1354.620 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1725.650 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1523.580 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1316.570 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 27766.070 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 22273.020 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22242.280 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 23961.370 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21785.160 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21339.160 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26107.190 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20944.170 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21245.570 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26135.640 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20992.150 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21219.420 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1455.710 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1026.830 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1004.900 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1017.320 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 519.360 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 372.620 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 337.220 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 347.010 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 2486.110 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1356.870 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1328.120 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1366.060 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1601.020 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1405.250 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 28242.410 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1368.780 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1392.770 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 22604.830 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21718.140 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21530.620 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21109.950 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 1934.520 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2271.060 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2127.650 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2137.710 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 660.390 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 640.570 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 559.890 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 561.000 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 2635.400 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 3000.320 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 2253.020 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 2261.290 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 3074.090 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2724.380 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 3303.800 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2263.010 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2255.760 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 44101.740 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 48973.360 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 69946.880 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 69947.250 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 778.909 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 2728.503 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 555.498 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 2195.661 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchPacked | 1068.491 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchPacked | 4067.501 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchMaterializedPacked | 1260.536 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchMaterializedPacked | 4195.885 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNN | 685.798 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNN | 2375.994 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNNPacked | 802.810 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNNPacked | 3514.362 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Triton | 2074.990 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TritonPacked | 2196.746 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | Torch | 17925.632 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | Torch | 33300.255 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | Triton | 8248.928 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | Triton | 22245.504 |
| layernorm | forward | `N=65536 C=768` | CUDA | 139.810 |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 276.514 |
| layernorm | backward | `N=65536 C=768` | CUDA | 288.147 |
| layernorm | forward | `N=65536 C=3072` | CUDA | 543.194 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | CUDA | 1084.727 |
| layernorm | backward | `N=65536 C=3072` | CUDA | 1266.138 |
| layernorm | forward | `N=65536 C=768` | Triton | 177.568 |
| layernorm | forward | `N=65536 C=768` | Torch native | 154.880 |
| layernorm | forward | `N=65536 C=768` | Torch stats | 2198.048 |
| layernorm | backward_dinput | `N=65536 C=768` | Triton dInput-only | 227.552 |
| layernorm | backward_dinput | `N=65536 C=768` | Torch native | 222.112 |
| layernorm | backward | `N=65536 C=768` | Torch native | 417.984 |
| layernorm | backward | `N=65536 C=768` | Triton atomic FP32-grad | 360.160 |
| layernorm | fused_residual_forward | `N=65536 C=768` | Triton | 312.320 |
| layernorm | fused_residual_forward | `N=65536 C=768` | Torch native | 329.632 |
| layernorm | fused_residual_forward | `N=65536 C=768` | Torch stats | 3172.512 |
| layernorm | forward | `N=65536 C=3072` | Triton | 575.584 |
| layernorm | forward | `N=65536 C=3072` | Torch native | 554.592 |
| layernorm | forward | `N=65536 C=3072` | Torch stats | 8947.552 |
| layernorm | backward_dinput | `N=65536 C=3072` | Triton dInput-only | 812.128 |
| layernorm | backward_dinput | `N=65536 C=3072` | Torch native | 845.120 |
| layernorm | backward | `N=65536 C=3072` | Torch native | 1397.760 |
| layernorm | backward | `N=65536 C=3072` | Triton atomic FP32-grad | 1424.608 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Triton | 1105.120 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Torch native | 1310.944 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Torch stats | 12946.112 |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 96.825 |
| runtime | bias_add | `BT=65536 OC=3072` | CUDA | 528.993 |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 527.850 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | CUDA | 779.645 |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 29.872 |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 186.634 |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 245.650 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 3948.493 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 8937.452 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | 3913.728 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA kernel | 4121.190 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA runtime | 8776.832 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA kernel | 9152.672 |
| runtime | global_norm_squared | `params=124475904` | CUDA | 184.909 |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1835.683 |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 85.180 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA runtime | 61.608 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA kernel | 62.200 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 131.723 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA kernel | 136.210 |
| runtime | bias_add | `BT=65536 OC=768` | Triton | 132.389 |
| runtime | bias_add | `BT=65536 OC=3072` | Triton | 527.930 |
| runtime | gelu_forward | `BT=65536 C=3072` | Triton | 530.515 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | 770.520 |
| runtime | bias_add | `BT=65536 OC=768` | Torch | 136.147 |
| runtime | bias_add | `BT=65536 OC=3072` | Torch | 532.366 |
| runtime | gelu_forward | `BT=65536 C=3072` | Torch | 538.289 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Torch | 26831.989 |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | Torch | 314.643 |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | Torch | 967.494 |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | Torch | 1315.942 |
| runtime | global_norm_squared | `params=124475904` | Torch | 2273.318 |
| runtime | adamw_update_bf16_state | `params=124475904 no-master` | Torch | 1196.518 |
| runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | 7315.776 |
| runtime | encoder_forward | `B=64 T=1024 C=768` | Torch | 199.050 |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch | 60.038 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | Torch | 131.984 |
| runtime | cuda_memset | `logits_elems=3296722944` | Torch | 3976.288 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch | 8665.152 |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch C++ | 60.011 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | Torch C++ | 131.526 |
| runtime | cuda_memset | `logits_elems=3296722944` | Torch C++ | 3984.000 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch C++ | 8686.496 |

## Unavailable Backend Rows

| Suite | Kernel | Shape | Stack | Reason |
|---|---|---|---|---|
| matmul | cutedsl_gemm | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | CuTeDSL | local CuTeDSL BF16 grouped-GEMM path rejects sm_120a |
| matmul | cutedsl_gemm | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | CuTeDSL | local CuTeDSL BF16 grouped-GEMM path rejects sm_120a |
| matmul | cutedsl_gemm | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | CuTeDSL | local CuTeDSL BF16 grouped-GEMM path rejects sm_120a |
| matmul | cutedsl_gemm | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | CuTeDSL | local CuTeDSL BF16 grouped-GEMM path rejects sm_120a |
| matmul | cutedsl_gemm | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | CuTeDSL | local CuTeDSL BF16 grouped-GEMM path rejects sm_120a |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Triton | attention backward is not implemented in this Triton prototype |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TritonPacked | packed attention backward is not implemented in this Triton prototype |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | Triton | not implemented in this Triton runtime prototype |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | Triton | not implemented in this Triton runtime prototype |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | Triton | not implemented in this Triton runtime prototype |
| runtime | adamw_update | `params=124475904 no-master` | Triton | not implemented in this Triton runtime prototype |
| runtime | global_norm_squared | `params=124475904` | Triton | not implemented in this Triton runtime prototype |
| runtime | encoder_forward | `B=64 T=1024 C=768` | Triton | not implemented in this Triton runtime prototype |
| runtime | cuda_memset | `hidden_elems=50331648` | Triton | not implemented in this Triton runtime prototype |
| runtime | cuda_memset | `logits_elems=3296722944` | Triton | not implemented in this Triton runtime prototype |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | Triton | not implemented in this Triton runtime prototype |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Triton | not implemented in this Triton runtime prototype |

