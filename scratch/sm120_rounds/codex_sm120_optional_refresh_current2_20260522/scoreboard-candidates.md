# SM120 Round Metrics - codex_sm120_optional_refresh_current2_20260522

- artifact dir: `scratch/sm120_rounds/codex_sm120_optional_refresh_current2_20260522`
- train output dir: `log124M/5090_S_codex_sm120_optional_refresh_current2_20260522`
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

- detailed matrix: `scratch/sm120_rounds/codex_sm120_optional_refresh_current2_20260522/backend-stacks.json`

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

- manifest: `scratch/sm120_rounds/codex_sm120_optional_refresh_current2_20260522/round-manifest.json`
- device arch: `SM120`
- build jobs: `4`
- changed paths: `687`

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

- log: `scratch/sm120_rounds/codex_sm120_optional_refresh_current2_20260522/validate_libtorch_trainer_link.log`
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
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 2160.624 | cuDNN | 2342.368 | python separated-Q/K/V | layout_rewrite_only | Use Torch SDPA for already-separated Q/K/V experiments; packed trainer selection must compare TorchPacked. |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 556.565 | cuDNN | 675.282 | python separated-Q/K/V | layout_rewrite_only | Use Torch SDPA for already-separated Q/K/V experiments; packed trainer selection must compare TorchPacked. |
| layernorm | backward | `N=65536 C=3072` | CUDA | 1095.961 | Torch native | 1390.688 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | backward | `N=65536 C=768` | CUDA | 267.484 | Triton atomic FP32-grad | 364.224 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | backward_dinput | `N=65536 C=3072` | Triton dInput-only | 799.040 | Torch native | 828.576 | partial backward prototype | - | Triton dInput-only is useful for backward decomposition work but needs dweight/dbias before trainer promotion. |
| layernorm | backward_dinput | `N=65536 C=768` | Torch native | 216.416 | Triton dInput-only | 221.600 | partial backward prototype | partial_backward_only | Torch native dInput-only output-mask row is useful for backward decomposition work but needs dweight/dbias before trainer promotion. |
| layernorm | forward | `N=65536 C=3072` | CUDA | 537.587 | Torch native | 545.024 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | forward | `N=65536 C=768` | CUDA | 135.130 | Torch native | 153.088 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | fused_residual_forward | `N=65536 C=3072` | CUDA | 1072.468 | Triton | 1104.608 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 271.036 | Triton | 310.528 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 365.890 | cuBLASLt | 367.660 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1328.430 | cuBLASLt | 1345.590 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1380.780 | cuBLAS | 1390.680 | C++ benchmark route | benchmark_context_flip | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21018.670 | Torch | 21391.550 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1012.560 | cuBLASLt | 1014.310 | C++ benchmark route | noise_floor_microbench_flip | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1781.450 | cuBLASLt fused | 1798.840 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 329.220 | Torch | 338.690 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1309.130 | Torch | 1357.390 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1309.740 | Torch | 1377.930 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20689.360 | cuBLAS | 21016.010 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 995.370 | Torch | 1011.830 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 336.400 | Torch | 346.960 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1309.480 | Torch | 1369.930 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1315.310 | Torch | 1402.000 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20747.890 | cuBLAS | 21137.210 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 999.320 | Torch | 1016.740 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 369.540 | TK | 376.680 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1343.640 | TK | 1419.510 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22140.630 | cuBLASLt | 22152.850 | C++ benchmark route | rejected_trainer_smoke | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1063.790 | TK | 1073.030 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1471.710 | TK fused | 1537.880 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1785.632 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | 7284.800 | - | - | operator prototype | rejected_slower_than_trainer_baseline | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | adamw_update_bf16_state | `params=124475904 no-master` | Torch | 1198.336 | - | - | non-equivalent BF16-state reference | - | Torch fused AdamW is usable only for BF16 moment-state experiments; the trainer default uses FP32 moment buffers. |
| runtime | bias_add | `BT=65536 OC=3072` | CUDA | 528.467 | Triton | 529.507 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 67.964 | Triton | 132.370 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 186.488 | Torch | 969.094 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 244.925 | Torch | 1304.864 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 24.514 | Torch | 320.336 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 131.588 | Torch | 131.720 | CUDA benchmark route | profiler_only_runtime_row | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch C++ | 8633.024 | Torch | 8637.824 | C++ API prototype | profiler_only_runtime_row | LibTorch C++ API row proves a possible trainer-callable dependency path; promotion still needs call-site integration and TinyStories smoke. |
| runtime | cuda_memset | `grad_elems=124475904` | Torch C++ | 148.206 | Torch | 148.379 | C++ API prototype | rejected_x10_trainer_route | LibTorch C++ API row proves a possible trainer-callable dependency path; promotion still needs call-site integration and TinyStories smoke. |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch C++ | 59.861 | Torch | 60.032 | C++ API prototype | rejected_x10_trainer_route | LibTorch C++ API row proves a possible trainer-callable dependency path; promotion still needs call-site integration and TinyStories smoke. |
| runtime | cuda_memset | `logits_elems=3296722944` | Torch C++ | 3911.808 | CUDA runtime | 3923.424 | C++ API prototype | profiler_only_runtime_row | LibTorch C++ API row proves a possible trainer-callable dependency path; promotion still needs call-site integration and TinyStories smoke. |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 80.172 | Torch | 199.632 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 8793.012 | Triton | 21731.169 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 3898.758 | Triton | 8226.144 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | 770.482 | CUDA | 770.513 | operator prototype | rejected_same_session_refresh | Use as a Triton comparison row until a trainer-callable integration beats the current provider. |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 527.517 | Torch | 528.686 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | global_norm_squared | `params=124475904` | CUDA | 185.749 | Torch | 2260.640 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |

## Resolved Selected Backend Decisions

| Suite | Kernel | Shape | Selected stack | Status | Decision |
|---|---|---|---|---|---|
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| layernorm | backward_dinput | `N=65536 C=768` | Torch native | partial_backward_only | Do not promote the dInput-only Torch row. The full Torch native backward with dweight/dbias is slower than CUDA, the full Triton atomic prototype is also slower plus has an FP32 gradient-buffer contract mismatch, and the focused Torch dInput-plus-BF16-grads hybrid is much slower than the CUDA full-backward baseline. |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | rejected_x10_selector | Do not broaden the SM120 direct-cuBLAS dInput selector to attention projection. Both the bundled attproj/MLP-up selector and the later attproj-only selector regressed in x10 TinyStories stability gates, so the source default remains the huge-N LM-head-only direct-cuBLAS dInput route. |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | rejected_x10_selector | Do not promote the SM120 direct-cuBLAS dInput selector to the GPT-2 MLP-up row. The microbench row can favor cuBLAS, but both the broader attproj+MLP-up selector and the later FC-only selector regressed in x10 TinyStories stability gates, while the stream-sync default keeps cuBLASLt for this row and improves the trainer smoke. |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | benchmark_context_flip | Keep the training-backed current native row on cuBLASLt for the GPT-2 MLP projection dInput path. The earlier current-source x10 native round selected cuBLAS, but the stream-sync x10 native round and optional-stack benchmark both select cuBLASLt, and a narrow direct-cuBLAS trainer selector regressed badly in x3 TinyStories. |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | noise_floor_microbench_flip | Do not promote the qkv dInput cuBLAS microbench flip as a trainer default without a trainer smoke. The refreshed benchmark-only round picked cuBLAS by about 0.2%, while the stable x10 selection artifact has cuBLASLt ahead for the same row. |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | rejected_x10_selector | Keep the cuBLASLt fused dGELU trainer route as the default. The opt-in TK exact-dGELU selector now passes correctness and has a focused row win, but its x10 TinyStories stability round regressed versus the current stable default. |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | rejected_trainer_smoke | Keep LM-head forward on cuBLASLt; the opt-in direct-cuBLAS forward selector passed focused gates but regressed in TinyStories trainer timing. |
| runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | rejected_slower_than_trainer_baseline | Rejected for trainer promotion; Torch fp32-state AdamW is materially slower than the CUDA trainer route. |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | profiler_only_runtime_row | Keep as benchmark evidence only. The refreshed optional round measured CUDA runtime as fastest for this profiler-only copy shape, but it is not a current trainer call path to promote. |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch C++ | profiler_only_runtime_row | Keep as profiler/runtime evidence only. The refreshed LibTorch row is the fastest observed logits-copy row, but the current GPT-2 trainer has no logits-sized device-to-device copy call-site to promote. |
| runtime | cuda_memset | `grad_elems=124475904` | Torch C++ | rejected_x10_trainer_route | Do not promote the LibTorch gradients-zero trainer route by default. The opt-in C++ call-site now exists and passes correctness plus TinyStories smoke, but its x10 stability round regressed versus the current native x10 trainer selection. |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch C++ | rejected_x10_trainer_route | Do not promote the LibTorch dresidual-zero trainer route by default. The C++ API feasibility row was tie-range, and the integrated trainer route regressed in the x10 TinyStories gate. |
| runtime | cuda_memset | `logits_elems=3296722944` | Torch C++ | profiler_only_runtime_row | Keep as profiler/runtime evidence only. The current GPT-2 trainer does not issue a logits-sized memset; this row measures large-buffer runtime behavior rather than a promotable trainer call-site. |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | rejected_same_session_refresh | Rejected for trainer promotion; refreshed CUDA GELU backward was faster than Triton. |

## Promotion Backlog

No active promotion candidates remain after applying `dev/sm120_promotion_decisions.json`.

## Resolved Promotion Decisions

| Class | Suite | Kernel | Shape | Winning stack | Status | Decision |
|---|---|---|---|---|---|---|
| native/codegen integration | runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | rejected_same_session_refresh | Rejected for trainer promotion; refreshed CUDA GELU backward was faster than Triton. |
| library integration | runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | rejected_slower_than_trainer_baseline | Rejected for trainer promotion; Torch fp32-state AdamW is materially slower than the CUDA trainer route. |
| library integration | runtime | cuda_memset | `logits_elems=3296722944` | Torch C++ | profiler_only_runtime_row | Keep as profiler/runtime evidence only. The current GPT-2 trainer does not issue a logits-sized memset; this row measures large-buffer runtime behavior rather than a promotable trainer call-site. |
| library integration | runtime | cuda_memset | `hidden_elems=50331648` | Torch C++ | rejected_x10_trainer_route | Do not promote the LibTorch dresidual-zero trainer route by default. The C++ API feasibility row was tie-range, and the integrated trainer route regressed in the x10 TinyStories gate. |
| library integration | runtime | cuda_memset | `grad_elems=124475904` | Torch C++ | rejected_x10_trainer_route | Do not promote the LibTorch gradients-zero trainer route by default. The opt-in C++ call-site now exists and passes correctness plus TinyStories smoke, but its x10 stability round regressed versus the current native x10 trainer selection. |
| library integration | runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch C++ | profiler_only_runtime_row | Keep as profiler/runtime evidence only. The refreshed LibTorch row is the fastest observed logits-copy row, but the current GPT-2 trainer has no logits-sized device-to-device copy call-site to promote. |
| layout rewrite | attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| layout rewrite | attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| reference/state gap | layernorm | backward_dinput | `N=65536 C=768` | Torch native | partial_backward_only | Do not promote the dInput-only Torch row. The full Torch native backward with dweight/dbias is slower than CUDA, the full Triton atomic prototype is also slower plus has an FP32 gradient-buffer contract mismatch, and the focused Torch dInput-plus-BF16-grads hybrid is much slower than the CUDA full-backward baseline. |
| non-trainer shape | layernorm | backward_dinput | `N=65536 C=3072` | Triton dInput-only | non_trainer_shape | Not an active trainer-promotion target for GPT-2 124M. |
| contract mismatch | runtime | adamw_update_bf16_state | `params=124475904 no-master` | Torch | contract_mismatch | Not active until the candidate matches the trainer state contract. |

## Attention Route Totals

| Shape | Stack | Scope | Trainer-layout | Forward (us) | Backward (us) | Total (us) | Complete | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | packed trainer-layout route | True | 774.667 | 2706.639 | 3481.306 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | cuDNNPacked | packed trainer-layout route | True | 790.320 | 2765.123 | 3555.443 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | TorchPacked | packed trainer-layout route | True | 1142.946 | 4002.704 | 5145.650 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | TorchMaterializedPacked | packed trainer-layout route | True | 1247.822 | 4149.318 | 5397.140 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | TritonPacked | packed trainer-layout route | True | 2186.347 | - | - | False | packed attention backward is not implemented in this Triton prototype |
| `B=64 T=1024 C=768 NH=12 HS=64` | Torch | separated Q/K/V reference route | False | 556.565 | 2160.624 | 2717.189 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | cuDNN | separated Q/K/V reference route | False | 675.282 | 2342.368 | 3017.650 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | Triton | separated Q/K/V reference route | False | 2079.432 | - | - | False | attention backward is not implemented in this Triton prototype |

## Benchmark Candidates

| Suite | Kernel | Shape | Stack | Time (us) |
|---|---|---|---|---:|
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1073.030 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1063.790 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1410.890 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1092.690 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1014.310 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1012.560 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1462.200 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1117.850 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 995.370 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1463.820 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1112.650 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 999.320 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 376.680 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 369.540 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 483.460 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 381.350 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 367.660 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 365.890 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 549.470 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 376.590 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 329.220 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 546.970 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 381.320 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 336.400 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK fused | 1537.880 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK explicit | 1948.250 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1471.710 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS explicit | 2416.480 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1448.420 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1345.590 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1328.430 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1734.770 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1475.130 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1309.130 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1718.960 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1475.700 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1309.480 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1419.510 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1343.640 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1542.160 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1476.830 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1380.780 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1390.680 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1781.450 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt fused | 1798.840 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt explicit | 2122.930 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS explicit | 2147.820 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1730.130 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1468.440 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1309.740 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1726.320 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1499.290 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1315.310 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 27308.440 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 22152.850 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22140.630 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 23700.600 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21618.550 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21018.670 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 25822.770 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20689.360 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21016.010 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 25817.420 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20747.890 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21137.210 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1465.780 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1023.990 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1011.830 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1016.740 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 518.060 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 373.520 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 338.690 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 346.960 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 2444.310 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1368.510 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1357.390 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1369.930 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1595.620 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1405.410 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 27836.940 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1377.930 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1402.000 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 22372.540 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21391.550 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21094.300 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21216.130 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch C++ | 1039.340 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch C++ | 1042.800 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch C++ | 352.310 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch C++ | 351.250 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch C++ | 1368.240 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch C++ | 1402.280 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch C++ | 1453.380 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch C++ | 1452.450 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch C++ | 22611.810 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch C++ | 22820.350 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 1981.750 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2286.430 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2132.130 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2139.380 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 663.790 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 686.040 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 561.830 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 566.760 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 2702.260 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 2981.130 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 2236.880 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 2226.090 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2973.550 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2695.740 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 3308.470 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2255.150 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2268.290 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 43486.930 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 48587.840 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 69211.600 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 68848.640 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 774.667 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 2706.639 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 556.565 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 2160.624 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchPacked | 1142.946 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchPacked | 4002.704 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchMaterializedPacked | 1247.822 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchMaterializedPacked | 4149.318 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNN | 675.282 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNN | 2342.368 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNNPacked | 790.320 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNNPacked | 2765.123 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Triton | 2079.432 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TritonPacked | 2186.347 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | Torch | 17435.137 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | Triton | 8226.144 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | Triton | 21731.169 |
| layernorm | forward | `N=65536 C=768` | CUDA | 135.130 |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 271.036 |
| layernorm | backward | `N=65536 C=768` | CUDA | 267.484 |
| layernorm | forward | `N=65536 C=3072` | CUDA | 537.587 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | CUDA | 1072.468 |
| layernorm | backward | `N=65536 C=3072` | CUDA | 1095.961 |
| layernorm | forward | `N=65536 C=768` | Triton | 177.120 |
| layernorm | forward | `N=65536 C=768` | Torch native | 153.088 |
| layernorm | forward | `N=65536 C=768` | Torch stats | 2201.056 |
| layernorm | backward_dinput | `N=65536 C=768` | Triton dInput-only | 221.600 |
| layernorm | backward_dinput | `N=65536 C=768` | Torch native | 216.416 |
| layernorm | backward | `N=65536 C=768` | Torch native+BF16-grads | 1973.600 |
| layernorm | backward | `N=65536 C=768` | Torch native | 411.872 |
| layernorm | backward | `N=65536 C=768` | Triton atomic FP32-grad | 364.224 |
| layernorm | fused_residual_forward | `N=65536 C=768` | Triton | 310.528 |
| layernorm | fused_residual_forward | `N=65536 C=768` | Torch native | 331.488 |
| layernorm | fused_residual_forward | `N=65536 C=768` | Torch stats | 3176.512 |
| layernorm | forward | `N=65536 C=3072` | Triton | 574.240 |
| layernorm | forward | `N=65536 C=3072` | Torch native | 545.024 |
| layernorm | forward | `N=65536 C=3072` | Torch stats | 8916.672 |
| layernorm | backward_dinput | `N=65536 C=3072` | Triton dInput-only | 799.040 |
| layernorm | backward_dinput | `N=65536 C=3072` | Torch native | 828.576 |
| layernorm | backward | `N=65536 C=3072` | Torch native+BF16-grads | 7919.008 |
| layernorm | backward | `N=65536 C=3072` | Torch native | 1390.688 |
| layernorm | backward | `N=65536 C=3072` | Triton atomic FP32-grad | 1425.568 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Triton | 1104.608 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Torch native | 1306.048 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Torch stats | 12951.232 |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 67.964 |
| runtime | bias_add | `BT=65536 OC=3072` | CUDA | 528.467 |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 527.517 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | CUDA | 770.513 |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 24.514 |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 186.488 |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 244.925 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 3898.758 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 8793.012 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | 3923.424 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA kernel | 3930.119 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA runtime | 8698.957 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA kernel | 8779.840 |
| runtime | cuda_memset | `grad_elems=124475904` | CUDA runtime | 148.589 |
| runtime | cuda_memset | `grad_elems=124475904` | CUDA kernel | 149.846 |
| runtime | global_norm_squared | `params=124475904` | CUDA | 185.749 |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1785.632 |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 80.172 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA runtime | 61.096 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA kernel | 60.321 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 131.588 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA kernel | 133.514 |
| runtime | bias_add | `BT=65536 OC=768` | Triton | 132.370 |
| runtime | bias_add | `BT=65536 OC=3072` | Triton | 529.507 |
| runtime | gelu_forward | `BT=65536 C=3072` | Triton | 529.829 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | 770.482 |
| runtime | bias_add | `BT=65536 OC=768` | Torch | 135.883 |
| runtime | bias_add | `BT=65536 OC=3072` | Torch | 530.372 |
| runtime | gelu_forward | `BT=65536 C=3072` | Torch | 528.686 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Torch | 26475.491 |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | Torch | 320.336 |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | Torch | 969.094 |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | Torch | 1304.864 |
| runtime | cuda_memset | `grad_elems=124475904` | Torch | 148.379 |
| runtime | global_norm_squared | `params=124475904` | Torch | 2260.640 |
| runtime | adamw_update_bf16_state | `params=124475904 no-master` | Torch | 1198.336 |
| runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | 7284.800 |
| runtime | encoder_forward | `B=64 T=1024 C=768` | Torch | 199.632 |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch | 60.032 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | Torch | 131.720 |
| runtime | cuda_memset | `logits_elems=3296722944` | Torch | 3953.312 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch | 8637.824 |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch C++ | 59.861 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | Torch C++ | 131.829 |
| runtime | cuda_memset | `grad_elems=124475904` | Torch C++ | 148.206 |
| runtime | cuda_memset | `logits_elems=3296722944` | Torch C++ | 3911.808 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch C++ | 8633.024 |
| runtime | gelu_forward | `BT=65536 C=3072` | Torch C++ | 528.795 |

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

