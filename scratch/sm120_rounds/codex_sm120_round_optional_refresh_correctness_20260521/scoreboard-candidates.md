# SM120 Round Metrics - codex_sm120_round_optional_refresh_correctness_20260521

- artifact dir: `scratch/sm120_rounds/codex_sm120_round_optional_refresh_correctness_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_optional_refresh_correctness_20260521`
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

- detailed matrix: `scratch/sm120_rounds/codex_sm120_round_optional_refresh_correctness_20260521/backend-stacks.json`

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

- manifest: `scratch/sm120_rounds/codex_sm120_round_optional_refresh_correctness_20260521/round-manifest.json`
- device arch: `SM120`
- build jobs: `4`
- changed paths: `497`

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
| `cuda_memset` | `grad_elems=124475904` | `True` |
| `cuda_memset` | `logits_elems=3296722944` | `True` |
| `cuda_copy_d2d` | `hidden_elems=50331648` | `True` |
| `cuda_copy_d2d` | `logits_elems=3296722944` | `True` |

## LibTorch Runtime Shape Coverage

| Kernel | Shape | Covered |
|---|---|---:|
| `cuda_memset` | `hidden_elems=50331648` | `True` |
| `cuda_memset` | `grad_elems=124475904` | `True` |
| `cuda_memset` | `logits_elems=3296722944` | `True` |
| `cuda_copy_d2d` | `hidden_elems=50331648` | `True` |
| `cuda_copy_d2d` | `logits_elems=3296722944` | `True` |

## LibTorch Runtime Parity Coverage

| Kernel | Shape | Covered |
|---|---|---:|
| `cuda_memset` | `hidden_elems=50331648` | `True` |
| `cuda_memset` | `grad_elems=124475904` | `True` |
| `cuda_memset` | `logits_elems=3296722944` | `True` |
| `cuda_copy_d2d` | `hidden_elems=50331648` | `True` |
| `cuda_copy_d2d` | `logits_elems=3296722944` | `True` |

## LibTorch Supplemental Runtime Shape Coverage

| Kernel | Shape | Covered |
|---|---|---:|
| `gelu_forward` | `BT=65536 C=3072` | `True` |

## LibTorch Supplemental Runtime Parity Coverage

| Kernel | Shape | Covered |
|---|---|---:|
| `gelu_forward` | `BT=65536 C=3072` | `True` |

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
| `runtime` | `cuda_memset` | `grad_elems=124475904` | `True` |
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
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 2201.862 | cuDNN | 2371.510 | python separated-Q/K/V | layout_rewrite_only | Use Torch SDPA for already-separated Q/K/V experiments; packed trainer selection must compare TorchPacked. |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 551.350 | cuDNN | 694.430 | python separated-Q/K/V | layout_rewrite_only | Use Torch SDPA for already-separated Q/K/V experiments; packed trainer selection must compare TorchPacked. |
| layernorm | backward | `N=65536 C=3072` | CUDA | 1269.367 | Torch native | 1385.376 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | backward | `N=65536 C=768` | CUDA | 288.164 | Triton atomic FP32-grad | 364.320 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | backward_dinput | `N=65536 C=3072` | Triton dInput-only | 797.120 | Torch native | 826.528 | partial backward prototype | - | Triton dInput-only is useful for backward decomposition work but needs dweight/dbias before trainer promotion. |
| layernorm | backward_dinput | `N=65536 C=768` | Torch native | 220.352 | Triton dInput-only | 228.640 | partial backward prototype | partial_backward_only | Torch native dInput-only output-mask row is useful for backward decomposition work but needs dweight/dbias before trainer promotion. |
| layernorm | forward | `N=65536 C=3072` | Torch native | 539.616 | CUDA | 545.169 | reference only | non_trainer_shape | Native Torch is useful where saved mean/rstd are not needed; trainer LayerNorm still needs stats-compatible state. |
| layernorm | forward | `N=65536 C=768` | CUDA | 136.413 | Torch native | 155.104 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | fused_residual_forward | `N=65536 C=3072` | CUDA | 1084.119 | Triton | 1101.376 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 275.072 | Triton | 309.824 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 365.720 | cuBLASLt | 368.100 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1351.000 | cuBLASLt | 1361.210 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1384.030 | Torch | 1400.440 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21267.080 | Torch | 21678.180 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1012.520 | cuBLASLt | 1014.280 | C++ benchmark route | noise_floor_microbench_flip | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1821.170 | cuBLASLt fused | 1836.880 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 328.840 | Torch | 338.660 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1327.300 | Torch | 1345.680 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1307.970 | Torch | 1408.280 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20960.230 | cuBLAS | 21187.560 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 997.800 | Torch | 1026.460 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 336.520 | Torch | 346.850 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1334.310 | Torch | 1366.920 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1314.180 | Torch | 1402.250 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21055.910 | cuBLAS | 21329.980 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 999.250 | Torch | 1015.640 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 371.140 | TK | 376.280 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1364.430 | TK | 1434.360 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 22354.400 | cuBLAS | 22396.720 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1073.340 | cuBLASLt | 1089.660 | C++ benchmark route | benchmark_context_flip | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1493.070 | TK fused | 1596.220 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1807.536 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | 7316.448 | - | - | operator prototype | rejected_slower_than_trainer_baseline | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | adamw_update_bf16_state | `params=124475904 no-master` | Torch | 1195.731 | - | - | non-equivalent BF16-state reference | - | Torch fused AdamW is usable only for BF16 moment-state experiments; the trainer default uses FP32 moment buffers. |
| runtime | bias_add | `BT=65536 OC=3072` | Triton | 530.563 | CUDA | 537.004 | operator prototype | rejected_same_session_refresh | Use as a Triton comparison row until a trainer-callable integration beats the current provider. |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 79.551 | Triton | 132.664 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 186.680 | Torch | 1023.987 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 245.134 | Torch | 1328.042 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 24.107 | Torch | 320.022 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 131.673 | Torch C++ | 131.781 | CUDA benchmark route | profiler_only_runtime_row | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch C++ | 8659.680 | Torch | 8729.984 | C++ API prototype | profiler_only_runtime_row | LibTorch C++ API row proves a possible trainer-callable dependency path; promotion still needs call-site integration and TinyStories smoke. |
| runtime | cuda_memset | `grad_elems=124475904` | Torch C++ | 147.749 | Torch | 148.104 | C++ API prototype | rejected_x10_trainer_route | LibTorch C++ API row proves a possible trainer-callable dependency path; promotion still needs call-site integration and TinyStories smoke. |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch | 59.883 | CUDA runtime | 59.988 | operator prototype | rejected_x10_trainer_route | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | 3915.712 | Torch | 3917.664 | CUDA benchmark route | profiler_only_runtime_row | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 80.721 | Torch | 199.018 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 8855.699 | Triton | 22188.513 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 3996.774 | Triton | 8195.616 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | CUDA | 790.915 | Triton | 791.542 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | gelu_forward | `BT=65536 C=3072` | Triton | 529.843 | Torch C++ | 538.862 | operator prototype | rejected_same_session_refresh | Use as a Triton comparison row until a trainer-callable integration beats the current provider. |
| runtime | global_norm_squared | `params=124475904` | CUDA | 185.518 | Torch | 2264.032 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |

## Resolved Selected Backend Decisions

| Suite | Kernel | Shape | Selected stack | Status | Decision |
|---|---|---|---|---|---|
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| layernorm | backward_dinput | `N=65536 C=768` | Torch native | partial_backward_only | Do not promote the dInput-only Torch row. The full Torch native backward with dweight/dbias is slower than CUDA, the full Triton atomic prototype is also slower plus has an FP32 gradient-buffer contract mismatch, and the focused Torch dInput-plus-BF16-grads hybrid is much slower than the CUDA full-backward baseline. |
| layernorm | forward | `N=65536 C=3072` | Torch native | non_trainer_shape | Not a GPT-2 124M trainer LayerNorm width; keep as operator stress evidence. |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | rejected_x10_selector | Do not broaden the SM120 direct-cuBLAS dInput selector to attention projection; the x10 trainer stability round regressed, so the source default remains the huge-N LM-head-only direct-cuBLAS dInput route. |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | rejected_x10_selector | Do not broaden the SM120 direct-cuBLAS dInput selector to the GPT-2 MLP-up row. The microbench row can favor cuBLAS, but the broader direct-cuBLAS dInput selector regressed in the x10 TinyStories stability gate, while the stream-sync default keeps cuBLASLt for this row and improves the trainer smoke. |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | noise_floor_microbench_flip | Do not promote the qkv dInput cuBLAS microbench flip as a trainer default without a trainer smoke. The refreshed benchmark-only round picked cuBLAS by about 0.2%, while the stable x10 selection artifact has cuBLASLt ahead for the same row. |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | rejected_x10_selector | Keep the cuBLASLt fused dGELU trainer route as the default. The opt-in TK exact-dGELU selector now passes correctness and has a focused row win, but its x10 TinyStories stability round regressed versus the current stable default. |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | benchmark_context_flip | Keep qkv forward on the cuBLASLt trainer default. The correctness-enabled optional refresh picked TK by a small edge, but the current trainer-backed x10 native round selected cuBLASLt for the same row and no route-specific TinyStories smoke proves a switch. |
| runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | rejected_slower_than_trainer_baseline | Rejected for trainer promotion; Torch fp32-state AdamW is materially slower than the CUDA trainer route. |
| runtime | bias_add | `BT=65536 OC=3072` | Triton | rejected_same_session_refresh | Rejected for trainer promotion; focused same-session source-default CUDA vec2 timing beat Triton for the wide GPT-2 bias-add shape, and a wider vec4 CUDA candidate was slower. |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | profiler_only_runtime_row | Keep as benchmark evidence only. The refreshed optional round measured CUDA runtime as fastest for this profiler-only copy shape, but it is not a current trainer call path to promote. |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch C++ | profiler_only_runtime_row | Keep as profiler/runtime evidence only. The refreshed LibTorch row is the fastest observed logits-copy row, but the current GPT-2 trainer has no logits-sized device-to-device copy call-site to promote. |
| runtime | cuda_memset | `grad_elems=124475904` | Torch C++ | rejected_x10_trainer_route | Do not promote the LibTorch gradients-zero trainer route by default. The opt-in C++ call-site now exists and passes correctness plus TinyStories smoke, but its x10 stability round regressed versus the current native x10 trainer selection. |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch | rejected_x10_trainer_route | Do not promote the LibTorch dresidual-zero trainer route by default. The opt-in C++ call-site now exists and passes correctness plus TinyStories smoke, but its x10 stability round regressed versus the current native x10 trainer selection. |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | profiler_only_runtime_row | Keep as profiler/runtime evidence only. CUDA runtime remains the fastest current logits-sized memset row, but the current GPT-2 trainer has no logits-sized memset call-site to promote. |
| runtime | gelu_forward | `BT=65536 C=3072` | Triton | rejected_same_session_refresh | Rejected for trainer promotion; focused uncontended timing showed CUDA was faster than Triton for the GPT-2 MLP GELU-forward row. |

## Promotion Backlog

No active promotion candidates remain after applying `dev/sm120_promotion_decisions.json`.

## Resolved Promotion Decisions

| Class | Suite | Kernel | Shape | Winning stack | Status | Decision |
|---|---|---|---|---|---|---|
| native/codegen integration | runtime | gelu_forward | `BT=65536 C=3072` | Triton | rejected_same_session_refresh | Rejected for trainer promotion; focused uncontended timing showed CUDA was faster than Triton for the GPT-2 MLP GELU-forward row. |
| native/codegen integration | runtime | bias_add | `BT=65536 OC=3072` | Triton | rejected_same_session_refresh | Rejected for trainer promotion; focused same-session source-default CUDA vec2 timing beat Triton for the wide GPT-2 bias-add shape, and a wider vec4 CUDA candidate was slower. |
| library integration | runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | rejected_slower_than_trainer_baseline | Rejected for trainer promotion; Torch fp32-state AdamW is materially slower than the CUDA trainer route. |
| library integration | runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch C++ | profiler_only_runtime_row | Keep as profiler/runtime evidence only. The refreshed LibTorch row is the fastest observed logits-copy row, but the current GPT-2 trainer has no logits-sized device-to-device copy call-site to promote. |
| library integration | runtime | cuda_memset | `grad_elems=124475904` | Torch C++ | rejected_x10_trainer_route | Do not promote the LibTorch gradients-zero trainer route by default. The opt-in C++ call-site now exists and passes correctness plus TinyStories smoke, but its x10 stability round regressed versus the current native x10 trainer selection. |
| library integration | runtime | cuda_memset | `hidden_elems=50331648` | Torch | rejected_x10_trainer_route | Do not promote the LibTorch dresidual-zero trainer route by default. The opt-in C++ call-site now exists and passes correctness plus TinyStories smoke, but its x10 stability round regressed versus the current native x10 trainer selection. |
| layout rewrite | attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| layout rewrite | attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| reference/state gap | layernorm | backward_dinput | `N=65536 C=768` | Torch native | partial_backward_only | Do not promote the dInput-only Torch row. The full Torch native backward with dweight/dbias is slower than CUDA, the full Triton atomic prototype is also slower plus has an FP32 gradient-buffer contract mismatch, and the focused Torch dInput-plus-BF16-grads hybrid is much slower than the CUDA full-backward baseline. |
| non-trainer shape | layernorm | backward_dinput | `N=65536 C=3072` | Triton dInput-only | non_trainer_shape | Not an active trainer-promotion target for GPT-2 124M. |
| non-trainer shape | layernorm | forward | `N=65536 C=3072` | Torch native | non_trainer_shape | Not a GPT-2 124M trainer LayerNorm width; keep as operator stress evidence. |
| contract mismatch | runtime | adamw_update_bf16_state | `params=124475904 no-master` | Torch | contract_mismatch | Not active until the candidate matches the trainer state contract. |

## Attention Route Totals

| Shape | Stack | Scope | Trainer-layout | Forward (us) | Backward (us) | Total (us) | Complete | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | packed trainer-layout route | True | 788.745 | 2744.542 | 3533.287 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | cuDNNPacked | packed trainer-layout route | True | 809.363 | 2818.074 | 3627.437 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | TorchPacked | packed trainer-layout route | True | 1090.229 | 4074.912 | 5165.141 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | TorchMaterializedPacked | packed trainer-layout route | True | 1260.011 | 4190.883 | 5450.894 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | TritonPacked | packed trainer-layout route | True | 2199.478 | - | - | False | packed attention backward is not implemented in this Triton prototype |
| `B=64 T=1024 C=768 NH=12 HS=64` | Torch | separated Q/K/V reference route | False | 551.350 | 2201.862 | 2753.212 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | cuDNN | separated Q/K/V reference route | False | 694.430 | 2371.510 | 3065.940 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | Triton | separated Q/K/V reference route | False | 2070.654 | - | - | False | attention backward is not implemented in this Triton prototype |

## Benchmark Candidates

| Suite | Kernel | Shape | Stack | Time (us) |
|---|---|---|---|---:|
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1073.340 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1089.660 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1426.420 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1092.800 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1014.280 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1012.520 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1475.520 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1129.460 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 997.800 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1483.560 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1114.290 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 999.250 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 376.280 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 371.140 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 484.420 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 381.110 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 368.100 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 365.720 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 543.630 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 375.760 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 328.840 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 547.800 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 383.270 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 336.520 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK fused | 1596.220 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK explicit | 1969.830 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1493.070 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS explicit | 2464.020 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1463.550 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1361.210 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1351.000 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1759.570 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1497.030 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1327.300 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1759.760 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1491.240 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1334.310 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1434.360 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1364.430 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1566.660 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1491.390 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1402.340 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1384.030 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1821.170 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt fused | 1836.880 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt explicit | 2167.990 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS explicit | 2188.870 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1758.850 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1488.900 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1307.970 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1743.270 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1510.770 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1314.180 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 27873.860 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 22354.400 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22396.720 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 24037.570 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21851.730 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21267.080 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26082.020 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20960.230 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21187.560 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26197.940 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21055.910 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21329.980 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1443.420 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1017.910 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1026.460 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1015.640 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 517.530 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 371.860 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 338.660 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 346.850 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 2451.920 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1366.600 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1345.680 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1366.920 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1599.670 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1400.440 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 28252.240 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1408.280 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1402.250 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 22929.600 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21678.180 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21303.650 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21593.060 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch C++ | 1365.350 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch C++ | 1372.570 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 1952.470 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2326.800 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2139.450 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2179.920 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 663.100 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 657.100 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 560.020 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 563.110 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 2615.970 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 3078.470 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 2271.580 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 2270.000 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2976.160 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2809.240 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 3373.220 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2266.540 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2238.590 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 44010.740 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 49021.490 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 70116.510 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 70271.550 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 788.745 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 2744.542 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 551.350 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 2201.862 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchPacked | 1090.229 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchPacked | 4074.912 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchMaterializedPacked | 1260.011 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchMaterializedPacked | 4190.883 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNN | 694.430 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNN | 2371.510 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNNPacked | 809.363 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNNPacked | 2818.074 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Triton | 2070.654 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TritonPacked | 2199.478 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | Torch | 17874.144 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | Torch | 35105.568 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | Triton | 8195.616 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | Triton | 22188.513 |
| layernorm | forward | `N=65536 C=768` | CUDA | 136.413 |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 275.072 |
| layernorm | backward | `N=65536 C=768` | CUDA | 288.164 |
| layernorm | forward | `N=65536 C=3072` | CUDA | 545.169 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | CUDA | 1084.119 |
| layernorm | backward | `N=65536 C=3072` | CUDA | 1269.367 |
| layernorm | forward | `N=65536 C=768` | Triton | 176.704 |
| layernorm | forward | `N=65536 C=768` | Torch native | 155.104 |
| layernorm | forward | `N=65536 C=768` | Torch stats | 2189.696 |
| layernorm | backward_dinput | `N=65536 C=768` | Triton dInput-only | 228.640 |
| layernorm | backward_dinput | `N=65536 C=768` | Torch native | 220.352 |
| layernorm | backward | `N=65536 C=768` | Torch native+BF16-grads | 1968.192 |
| layernorm | backward | `N=65536 C=768` | Torch native | 413.440 |
| layernorm | backward | `N=65536 C=768` | Triton atomic FP32-grad | 364.320 |
| layernorm | fused_residual_forward | `N=65536 C=768` | Triton | 309.824 |
| layernorm | fused_residual_forward | `N=65536 C=768` | Torch native | 331.040 |
| layernorm | fused_residual_forward | `N=65536 C=768` | Torch stats | 3187.104 |
| layernorm | forward | `N=65536 C=3072` | Triton | 569.056 |
| layernorm | forward | `N=65536 C=3072` | Torch native | 539.616 |
| layernorm | forward | `N=65536 C=3072` | Torch stats | 8964.320 |
| layernorm | backward_dinput | `N=65536 C=3072` | Triton dInput-only | 797.120 |
| layernorm | backward_dinput | `N=65536 C=3072` | Torch native | 826.528 |
| layernorm | backward | `N=65536 C=3072` | Torch native+BF16-grads | 8122.912 |
| layernorm | backward | `N=65536 C=3072` | Torch native | 1385.376 |
| layernorm | backward | `N=65536 C=3072` | Triton atomic FP32-grad | 1416.736 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Triton | 1101.376 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Torch native | 1307.456 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Torch stats | 12952.704 |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 79.551 |
| runtime | bias_add | `BT=65536 OC=3072` | CUDA | 537.004 |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 544.852 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | CUDA | 790.915 |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 24.107 |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 186.680 |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 245.134 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 3996.774 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 8855.699 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | 3915.712 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA kernel | 4077.273 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA runtime | 8792.666 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA kernel | 9263.514 |
| runtime | cuda_memset | `grad_elems=124475904` | CUDA runtime | 148.470 |
| runtime | cuda_memset | `grad_elems=124475904` | CUDA kernel | 152.371 |
| runtime | global_norm_squared | `params=124475904` | CUDA | 185.518 |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1807.536 |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 80.721 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA runtime | 59.988 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA kernel | 62.268 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 131.673 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA kernel | 141.964 |
| runtime | bias_add | `BT=65536 OC=768` | Triton | 132.664 |
| runtime | bias_add | `BT=65536 OC=3072` | Triton | 530.563 |
| runtime | gelu_forward | `BT=65536 C=3072` | Triton | 529.843 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | 791.542 |
| runtime | bias_add | `BT=65536 OC=768` | Torch | 140.189 |
| runtime | bias_add | `BT=65536 OC=3072` | Torch | 538.940 |
| runtime | gelu_forward | `BT=65536 C=3072` | Torch | 540.131 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Torch | 26848.471 |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | Torch | 320.022 |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | Torch | 1023.987 |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | Torch | 1328.042 |
| runtime | cuda_memset | `grad_elems=124475904` | Torch | 148.104 |
| runtime | global_norm_squared | `params=124475904` | Torch | 2264.032 |
| runtime | adamw_update_bf16_state | `params=124475904 no-master` | Torch | 1195.731 |
| runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | 7316.448 |
| runtime | encoder_forward | `B=64 T=1024 C=768` | Torch | 199.018 |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch | 59.883 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | Torch | 131.874 |
| runtime | cuda_memset | `logits_elems=3296722944` | Torch | 3917.664 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch | 8729.984 |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch C++ | 60.203 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | Torch C++ | 131.781 |
| runtime | cuda_memset | `grad_elems=124475904` | Torch C++ | 147.749 |
| runtime | cuda_memset | `logits_elems=3296722944` | Torch C++ | 3946.688 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch C++ | 8659.680 |
| runtime | gelu_forward | `BT=65536 C=3072` | Torch C++ | 538.862 |

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
| runtime | cuda_memset | `grad_elems=124475904` | Triton | not implemented in this Triton runtime prototype |
| runtime | cuda_memset | `logits_elems=3296722944` | Triton | not implemented in this Triton runtime prototype |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | Triton | not implemented in this Triton runtime prototype |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Triton | not implemented in this Triton runtime prototype |

