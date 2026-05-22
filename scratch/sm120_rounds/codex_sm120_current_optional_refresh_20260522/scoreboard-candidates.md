# SM120 Round Metrics - codex_sm120_current_optional_refresh_20260522

- artifact dir: `scratch/sm120_rounds/codex_sm120_current_optional_refresh_20260522`
- train output dir: `log124M/5090_S_codex_sm120_current_optional_refresh_20260522`
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

- detailed matrix: `scratch/sm120_rounds/codex_sm120_current_optional_refresh_20260522/backend-stacks.json`

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

- manifest: `scratch/sm120_rounds/codex_sm120_current_optional_refresh_20260522/round-manifest.json`
- device arch: `SM120`
- build jobs: `4`
- changed paths: `681`

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

## LibTorch Trainer Link Probe

- log: `scratch/sm120_rounds/codex_sm120_current_optional_refresh_20260522/validate_libtorch_trainer_link.log`
- status: `PASS`

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
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 2200.397 | cuDNN | 2394.672 | python separated-Q/K/V | layout_rewrite_only | Use Torch SDPA for already-separated Q/K/V experiments; packed trainer selection must compare TorchPacked. |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 552.902 | cuDNN | 685.163 | python separated-Q/K/V | layout_rewrite_only | Use Torch SDPA for already-separated Q/K/V experiments; packed trainer selection must compare TorchPacked. |
| layernorm | backward | `N=65536 C=3072` | CUDA | 1106.244 | Torch native | 1367.776 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | backward | `N=65536 C=768` | CUDA | 273.412 | Triton atomic FP32-grad | 362.176 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | backward_dinput | `N=65536 C=3072` | Triton dInput-only | 804.736 | Torch native | 833.440 | partial backward prototype | - | Triton dInput-only is useful for backward decomposition work but needs dweight/dbias before trainer promotion. |
| layernorm | backward_dinput | `N=65536 C=768` | Torch native | 212.416 | Triton dInput-only | 224.320 | partial backward prototype | partial_backward_only | Torch native dInput-only output-mask row is useful for backward decomposition work but needs dweight/dbias before trainer promotion. |
| layernorm | forward | `N=65536 C=3072` | CUDA | 545.719 | Torch native | 549.888 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | forward | `N=65536 C=768` | Torch native | 139.712 | CUDA | 140.567 | reference only | stats_contract_mismatch | Native Torch is useful where saved mean/rstd are not needed; trainer LayerNorm still needs stats-compatible state. |
| layernorm | fused_residual_forward | `N=65536 C=3072` | CUDA | 1082.852 | Triton | 1106.656 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 273.463 | Triton | 309.088 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 365.770 | cuBLASLt | 367.370 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1329.850 | cuBLASLt | 1346.060 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1395.400 | Torch | 1397.460 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21318.020 | Torch | 21676.860 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1012.180 | cuBLAS | 1012.360 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1817.640 | cuBLASLt fused | 1850.490 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 328.500 | Torch | 339.280 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1312.640 | Torch | 1359.990 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1312.600 | Torch | 1389.410 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20939.030 | cuBLAS | 21235.840 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 993.390 | Torch | 1002.570 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 332.130 | Torch | 346.030 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1333.350 | Torch | 1374.370 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1341.090 | Torch | 1407.900 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20987.130 | cuBLAS | 21270.970 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1000.450 | Torch | 1012.490 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 369.820 | TK | 376.700 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1366.870 | TK | 1462.400 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 22344.680 | cuBLAS | 22391.790 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1040.410 | TK | 1073.560 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1483.620 | TK fused | 1587.620 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1830.045 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | 7272.448 | - | - | operator prototype | rejected_slower_than_trainer_baseline | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | adamw_update_bf16_state | `params=124475904 no-master` | Torch | 1220.154 | - | - | non-equivalent BF16-state reference | - | Torch fused AdamW is usable only for BF16 moment-state experiments; the trainer default uses FP32 moment buffers. |
| runtime | bias_add | `BT=65536 OC=3072` | Triton | 528.474 | CUDA | 546.833 | operator prototype | rejected_same_session_refresh | Use as a Triton comparison row until a trainer-callable integration beats the current provider. |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 61.693 | Triton | 132.038 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 186.549 | Torch | 967.293 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 245.187 | Torch | 1318.944 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 24.883 | Torch | 317.805 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 131.589 | Torch C++ | 131.760 | CUDA benchmark route | profiler_only_runtime_row | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch | 8665.536 | Torch C++ | 8695.360 | operator prototype | profiler_only_runtime_row | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | cuda_memset | `grad_elems=124475904` | Torch | 147.976 | Torch C++ | 148.203 | operator prototype | rejected_x10_trainer_route | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch C++ | 59.938 | CUDA runtime | 59.980 | C++ API prototype | rejected_x10_trainer_route | LibTorch C++ API row proves a possible trainer-callable dependency path; promotion still needs call-site integration and TinyStories smoke. |
| runtime | cuda_memset | `logits_elems=3296722944` | Torch C++ | 3943.584 | Torch | 3949.440 | C++ API prototype | profiler_only_runtime_row | LibTorch C++ API row proves a possible trainer-callable dependency path; promotion still needs call-site integration and TinyStories smoke. |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 84.120 | Torch | 199.842 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 8895.597 | Triton | 22262.625 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 3899.872 | Triton | 8255.392 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | 772.560 | CUDA | 790.993 | operator prototype | rejected_same_session_refresh | Use as a Triton comparison row until a trainer-callable integration beats the current provider. |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 527.846 | Torch C++ | 529.755 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | global_norm_squared | `params=124475904` | CUDA | 184.930 | Torch | 2368.902 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |

## Resolved Selected Backend Decisions

| Suite | Kernel | Shape | Selected stack | Status | Decision |
|---|---|---|---|---|---|
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| layernorm | backward_dinput | `N=65536 C=768` | Torch native | partial_backward_only | Do not promote the dInput-only Torch row. The full Torch native backward with dweight/dbias is slower than CUDA, the full Triton atomic prototype is also slower plus has an FP32 gradient-buffer contract mismatch, and the focused Torch dInput-plus-BF16-grads hybrid is much slower than the CUDA full-backward baseline. |
| layernorm | forward | `N=65536 C=768` | Torch native | stats_contract_mismatch | Keep Torch native LayerNorm forward as reference/operator evidence only. It does not produce the saved mean/rstd state required by the GPT-2 trainer backward contract; stats-compatible Torch compositions are much slower than CUDA. |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | rejected_x10_selector | Do not broaden the SM120 direct-cuBLAS dInput selector to attention projection. Both the bundled attproj/MLP-up selector and the later attproj-only selector regressed in x10 TinyStories stability gates, so the source default remains the huge-N LM-head-only direct-cuBLAS dInput route. |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | rejected_x10_selector | Do not promote the SM120 direct-cuBLAS dInput selector to the GPT-2 MLP-up row. The microbench row can favor cuBLAS, but both the broader attproj+MLP-up selector and the later FC-only selector regressed in x10 TinyStories stability gates, while the stream-sync default keeps cuBLASLt for this row and improves the trainer smoke. |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | rejected_x10_selector | Keep the cuBLASLt fused dGELU trainer route as the default. The opt-in TK exact-dGELU selector now passes correctness and has a focused row win, but its x10 TinyStories stability round regressed versus the current stable default. |
| runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | rejected_slower_than_trainer_baseline | Rejected for trainer promotion; Torch fp32-state AdamW is materially slower than the CUDA trainer route. |
| runtime | bias_add | `BT=65536 OC=3072` | Triton | rejected_same_session_refresh | Rejected for trainer promotion; focused same-session source-default CUDA vec2 timing beat Triton for the wide GPT-2 bias-add shape, and a wider vec4 CUDA candidate was slower. |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | profiler_only_runtime_row | Keep as benchmark evidence only. The refreshed optional round measured CUDA runtime as fastest for this profiler-only copy shape, but it is not a current trainer call path to promote. |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch | profiler_only_runtime_row | Keep as profiler/runtime evidence only. The older Python Torch logits-copy win is superseded by the refreshed Torch C++ row, but the current GPT-2 trainer has no logits-sized device-to-device copy call-site to promote. |
| runtime | cuda_memset | `grad_elems=124475904` | Torch | rejected_x10_trainer_route | Resolve the Python Torch grad-buffer memset operator row through the existing LibTorch C++ grad-zero trainer route, not a new trainer integration. The refreshed operator row is slightly faster in isolation, but the current LibTorch grad-zero x10 gate regressed, so keep CUDA-kernel grad-zero selected. |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch C++ | rejected_x10_trainer_route | Do not promote the LibTorch dresidual-zero trainer route by default. The C++ API feasibility row was tie-range, and the integrated trainer route regressed in the x10 TinyStories gate. |
| runtime | cuda_memset | `logits_elems=3296722944` | Torch C++ | profiler_only_runtime_row | Keep as profiler/runtime evidence only. The current GPT-2 trainer does not issue a logits-sized memset; this row measures large-buffer runtime behavior rather than a promotable trainer call-site. |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | rejected_same_session_refresh | Rejected for trainer promotion; refreshed CUDA GELU backward was faster than Triton. |

## Promotion Backlog

No active promotion candidates remain after applying `dev/sm120_promotion_decisions.json`.

## Resolved Promotion Decisions

| Class | Suite | Kernel | Shape | Winning stack | Status | Decision |
|---|---|---|---|---|---|---|
| native/codegen integration | runtime | bias_add | `BT=65536 OC=3072` | Triton | rejected_same_session_refresh | Rejected for trainer promotion; focused same-session source-default CUDA vec2 timing beat Triton for the wide GPT-2 bias-add shape, and a wider vec4 CUDA candidate was slower. |
| native/codegen integration | runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | rejected_same_session_refresh | Rejected for trainer promotion; refreshed CUDA GELU backward was faster than Triton. |
| library integration | runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | rejected_slower_than_trainer_baseline | Rejected for trainer promotion; Torch fp32-state AdamW is materially slower than the CUDA trainer route. |
| library integration | runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch | profiler_only_runtime_row | Keep as profiler/runtime evidence only. The older Python Torch logits-copy win is superseded by the refreshed Torch C++ row, but the current GPT-2 trainer has no logits-sized device-to-device copy call-site to promote. |
| library integration | runtime | cuda_memset | `grad_elems=124475904` | Torch | rejected_x10_trainer_route | Resolve the Python Torch grad-buffer memset operator row through the existing LibTorch C++ grad-zero trainer route, not a new trainer integration. The refreshed operator row is slightly faster in isolation, but the current LibTorch grad-zero x10 gate regressed, so keep CUDA-kernel grad-zero selected. |
| library integration | runtime | cuda_memset | `logits_elems=3296722944` | Torch C++ | profiler_only_runtime_row | Keep as profiler/runtime evidence only. The current GPT-2 trainer does not issue a logits-sized memset; this row measures large-buffer runtime behavior rather than a promotable trainer call-site. |
| library integration | runtime | cuda_memset | `hidden_elems=50331648` | Torch C++ | rejected_x10_trainer_route | Do not promote the LibTorch dresidual-zero trainer route by default. The C++ API feasibility row was tie-range, and the integrated trainer route regressed in the x10 TinyStories gate. |
| layout rewrite | attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| layout rewrite | attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| reference/state gap | layernorm | backward_dinput | `N=65536 C=768` | Torch native | partial_backward_only | Do not promote the dInput-only Torch row. The full Torch native backward with dweight/dbias is slower than CUDA, the full Triton atomic prototype is also slower plus has an FP32 gradient-buffer contract mismatch, and the focused Torch dInput-plus-BF16-grads hybrid is much slower than the CUDA full-backward baseline. |
| reference/state gap | layernorm | forward | `N=65536 C=768` | Torch native | stats_contract_mismatch | Keep Torch native LayerNorm forward as reference/operator evidence only. It does not produce the saved mean/rstd state required by the GPT-2 trainer backward contract; stats-compatible Torch compositions are much slower than CUDA. |
| non-trainer shape | layernorm | backward_dinput | `N=65536 C=3072` | Triton dInput-only | non_trainer_shape | Not an active trainer-promotion target for GPT-2 124M. |
| contract mismatch | runtime | adamw_update_bf16_state | `params=124475904 no-master` | Torch | contract_mismatch | Not active until the candidate matches the trainer state contract. |

## Attention Route Totals

| Shape | Stack | Scope | Trainer-layout | Forward (us) | Backward (us) | Total (us) | Complete | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | packed trainer-layout route | True | 784.749 | 2738.776 | 3523.525 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | cuDNNPacked | packed trainer-layout route | True | 813.355 | 2824.602 | 3637.957 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | TorchPacked | packed trainer-layout route | True | 1090.702 | 4065.501 | 5156.203 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | TorchMaterializedPacked | packed trainer-layout route | True | 1268.094 | 4190.842 | 5458.936 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | TritonPacked | packed trainer-layout route | True | 2199.115 | - | - | False | packed attention backward is not implemented in this Triton prototype |
| `B=64 T=1024 C=768 NH=12 HS=64` | Torch | separated Q/K/V reference route | False | 552.902 | 2200.397 | 2753.299 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | cuDNN | separated Q/K/V reference route | False | 685.163 | 2394.672 | 3079.835 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | Triton | separated Q/K/V reference route | False | 2069.312 | - | - | False | attention backward is not implemented in this Triton prototype |

## Benchmark Candidates

| Suite | Kernel | Shape | Stack | Time (us) |
|---|---|---|---|---:|
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1073.560 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1040.410 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1434.300 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1092.210 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1012.180 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1012.360 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1460.900 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1116.210 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 993.390 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1496.620 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1112.700 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1000.450 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 376.700 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 369.820 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 483.150 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 380.510 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 367.370 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 365.770 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 541.330 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 375.930 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 328.500 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 543.700 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 379.100 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 332.130 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK fused | 1587.620 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK explicit | 1969.450 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1483.620 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS explicit | 2466.560 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1448.580 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1346.060 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1329.850 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1705.860 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1481.810 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1312.640 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1764.140 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1488.780 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1333.350 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1462.400 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1366.870 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1593.790 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1487.420 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1425.650 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1395.400 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1817.640 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt fused | 1850.490 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt explicit | 2173.260 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS explicit | 2179.490 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1708.510 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1511.770 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1312.600 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1748.420 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1487.200 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1341.090 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 27754.260 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 22344.680 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22391.790 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 24037.770 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21923.550 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21318.020 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 25990.550 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20939.030 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21235.840 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26207.720 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20987.130 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21270.970 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1457.150 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1017.200 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1002.570 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1012.490 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 516.140 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 373.580 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 339.280 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 346.030 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 2443.530 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1368.520 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1359.990 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1374.370 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1601.860 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1397.460 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 28229.930 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1389.410 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1407.900 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 22929.340 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21676.860 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21553.950 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21373.090 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch C++ | 1037.500 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch C++ | 1040.880 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch C++ | 349.780 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch C++ | 352.740 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch C++ | 1386.690 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch C++ | 1395.670 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch C++ | 1445.970 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch C++ | 1443.460 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch C++ | 22974.240 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch C++ | 23277.410 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2016.580 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2304.810 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2186.000 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2140.170 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 663.120 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 697.500 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 563.550 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 563.190 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 2677.780 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 3038.610 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 2270.110 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 2325.060 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 3017.290 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2779.390 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 3317.040 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2234.390 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2319.790 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 45589.860 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 49117.630 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 70343.170 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 70313.630 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 784.749 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 2738.776 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 552.902 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 2200.397 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchPacked | 1090.702 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchPacked | 4065.501 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchMaterializedPacked | 1268.094 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchMaterializedPacked | 4190.842 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNN | 685.163 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNN | 2394.672 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNNPacked | 813.355 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNNPacked | 2824.602 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Triton | 2069.312 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TritonPacked | 2199.115 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | Torch | 17858.240 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | Triton | 8255.392 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | Triton | 22262.625 |
| layernorm | forward | `N=65536 C=768` | CUDA | 140.567 |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 273.463 |
| layernorm | backward | `N=65536 C=768` | CUDA | 273.412 |
| layernorm | forward | `N=65536 C=3072` | CUDA | 545.719 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | CUDA | 1082.852 |
| layernorm | backward | `N=65536 C=3072` | CUDA | 1106.244 |
| layernorm | forward | `N=65536 C=768` | Triton | 175.040 |
| layernorm | forward | `N=65536 C=768` | Torch native | 139.712 |
| layernorm | forward | `N=65536 C=768` | Torch stats | 2183.296 |
| layernorm | backward_dinput | `N=65536 C=768` | Triton dInput-only | 224.320 |
| layernorm | backward_dinput | `N=65536 C=768` | Torch native | 212.416 |
| layernorm | backward | `N=65536 C=768` | Torch native+BF16-grads | 1966.432 |
| layernorm | backward | `N=65536 C=768` | Torch native | 423.232 |
| layernorm | backward | `N=65536 C=768` | Triton atomic FP32-grad | 362.176 |
| layernorm | fused_residual_forward | `N=65536 C=768` | Triton | 309.088 |
| layernorm | fused_residual_forward | `N=65536 C=768` | Torch native | 338.080 |
| layernorm | fused_residual_forward | `N=65536 C=768` | Torch stats | 3196.512 |
| layernorm | forward | `N=65536 C=3072` | Triton | 571.872 |
| layernorm | forward | `N=65536 C=3072` | Torch native | 549.888 |
| layernorm | forward | `N=65536 C=3072` | Torch stats | 8999.488 |
| layernorm | backward_dinput | `N=65536 C=3072` | Triton dInput-only | 804.736 |
| layernorm | backward_dinput | `N=65536 C=3072` | Torch native | 833.440 |
| layernorm | backward | `N=65536 C=3072` | Torch native+BF16-grads | 7908.384 |
| layernorm | backward | `N=65536 C=3072` | Torch native | 1367.776 |
| layernorm | backward | `N=65536 C=3072` | Triton atomic FP32-grad | 1414.368 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Triton | 1106.656 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Torch native | 1312.352 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Torch stats | 12930.016 |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 61.693 |
| runtime | bias_add | `BT=65536 OC=3072` | CUDA | 546.833 |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 527.846 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | CUDA | 790.993 |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 24.883 |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 186.549 |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 245.187 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 3899.872 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 8895.597 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | 4007.981 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA kernel | 3975.066 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA runtime | 8779.342 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA kernel | 8942.599 |
| runtime | cuda_memset | `grad_elems=124475904` | CUDA runtime | 150.442 |
| runtime | cuda_memset | `grad_elems=124475904` | CUDA kernel | 149.478 |
| runtime | global_norm_squared | `params=124475904` | CUDA | 184.930 |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1830.045 |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 84.120 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA runtime | 59.980 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA kernel | 60.333 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 131.589 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA kernel | 133.520 |
| runtime | bias_add | `BT=65536 OC=768` | Triton | 132.038 |
| runtime | bias_add | `BT=65536 OC=3072` | Triton | 528.474 |
| runtime | gelu_forward | `BT=65536 C=3072` | Triton | 530.688 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | 772.560 |
| runtime | bias_add | `BT=65536 OC=768` | Torch | 135.624 |
| runtime | bias_add | `BT=65536 OC=3072` | Torch | 549.021 |
| runtime | gelu_forward | `BT=65536 C=3072` | Torch | 530.216 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Torch | 26843.265 |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | Torch | 317.805 |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | Torch | 967.293 |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | Torch | 1318.944 |
| runtime | cuda_memset | `grad_elems=124475904` | Torch | 147.976 |
| runtime | global_norm_squared | `params=124475904` | Torch | 2368.902 |
| runtime | adamw_update_bf16_state | `params=124475904 no-master` | Torch | 1220.154 |
| runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | 7272.448 |
| runtime | encoder_forward | `B=64 T=1024 C=768` | Torch | 199.842 |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch | 60.022 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | Torch | 132.098 |
| runtime | cuda_memset | `logits_elems=3296722944` | Torch | 3949.440 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch | 8665.536 |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch C++ | 59.938 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | Torch C++ | 131.760 |
| runtime | cuda_memset | `grad_elems=124475904` | Torch C++ | 148.203 |
| runtime | cuda_memset | `logits_elems=3296722944` | Torch C++ | 3943.584 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch C++ | 8695.360 |
| runtime | gelu_forward | `BT=65536 C=3072` | Torch C++ | 529.755 |

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
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | Torch | CUDA OOM at full GPT-2 padded-logits shape |
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

