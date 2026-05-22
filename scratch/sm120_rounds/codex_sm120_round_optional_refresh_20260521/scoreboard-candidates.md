# SM120 Round Metrics - codex_sm120_round_optional_refresh_20260521

- artifact dir: `scratch/sm120_rounds/codex_sm120_round_optional_refresh_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_optional_refresh_20260521`
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

- detailed matrix: `scratch/sm120_rounds/codex_sm120_round_optional_refresh_20260521/backend-stacks.json`

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

- manifest: `scratch/sm120_rounds/codex_sm120_round_optional_refresh_20260521/round-manifest.json`
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
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 2196.240 | cuDNN | 2404.397 | python separated-Q/K/V | layout_rewrite_only | Use Torch SDPA for already-separated Q/K/V experiments; packed trainer selection must compare TorchPacked. |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 556.573 | cuDNN | 675.635 | python separated-Q/K/V | layout_rewrite_only | Use Torch SDPA for already-separated Q/K/V experiments; packed trainer selection must compare TorchPacked. |
| layernorm | backward | `N=65536 C=3072` | CUDA | 1272.855 | Torch native | 1383.520 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | backward | `N=65536 C=768` | CUDA | 288.147 | Triton atomic FP32-grad | 364.992 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | backward_dinput | `N=65536 C=3072` | Triton dInput-only | 801.056 | Torch native | 819.136 | partial backward prototype | - | Triton dInput-only is useful for backward decomposition work but needs dweight/dbias before trainer promotion. |
| layernorm | backward_dinput | `N=65536 C=768` | Torch native | 214.592 | Triton dInput-only | 223.328 | partial backward prototype | partial_backward_only | Torch native dInput-only output-mask row is useful for backward decomposition work but needs dweight/dbias before trainer promotion. |
| layernorm | forward | `N=65536 C=3072` | CUDA | 544.301 | Torch native | 547.456 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | forward | `N=65536 C=768` | CUDA | 138.406 | Torch native | 154.944 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | fused_residual_forward | `N=65536 C=3072` | CUDA | 1082.597 | Triton | 1106.144 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 275.342 | Triton | 322.912 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 365.930 | cuBLASLt | 367.380 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1364.900 | Torch | 1383.700 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1380.010 | cuBLASLt | 1386.980 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21272.170 | Torch | 21444.640 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1012.440 | cuBLASLt | 1021.950 | C++ benchmark route | noise_floor_microbench_flip | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1822.440 | cuBLASLt fused | 1828.250 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 328.780 | Torch | 338.820 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1317.020 | Torch | 1352.330 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1311.440 | Torch | 1368.800 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20899.630 | cuBLAS | 21258.150 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 997.280 | Torch | 1011.020 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 332.650 | Torch | 345.340 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1315.250 | Torch C++ | 1367.160 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1318.570 | Torch | 1411.150 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20986.710 | cuBLAS | 21346.560 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 998.520 | Torch | 1017.400 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 371.910 | TK | 376.430 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1379.880 | TK | 1434.320 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 22396.620 | cuBLAS | 22411.320 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1041.660 | TK | 1096.170 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1471.130 | TK fused | 1563.540 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1809.206 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | 7298.912 | - | - | operator prototype | rejected_slower_than_trainer_baseline | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | adamw_update_bf16_state | `params=124475904 no-master` | Torch | 1208.083 | - | - | non-equivalent BF16-state reference | - | Torch fused AdamW is usable only for BF16 moment-state experiments; the trainer default uses FP32 moment buffers. |
| runtime | bias_add | `BT=65536 OC=3072` | Triton | 542.899 | CUDA | 548.129 | operator prototype | library_integration_not_justified | Use as a Triton comparison row until a trainer-callable integration beats the current provider. |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 80.025 | Triton | 133.016 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 186.802 | Torch | 971.398 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 245.528 | Torch | 1305.440 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 24.630 | Torch | 320.944 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 131.615 | Torch C++ | 131.805 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch | 8662.848 | Torch C++ | 8681.632 | operator prototype | native_replacement_rejected | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch C++ | 59.875 | Torch | 59.912 | C++ API prototype | library_integration_not_justified | LibTorch C++ API row proves a possible trainer-callable dependency path, but promotion still needs an explicit link gate and TinyStories smoke. |
| runtime | cuda_memset | `logits_elems=3296722944` | Torch | 3943.104 | Torch C++ | 3952.768 | operator prototype | native_replacement_rejected | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 79.286 | Torch | 201.555 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 8942.509 | Triton | 22228.512 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 3953.133 | Triton | 8246.336 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | 781.888 | CUDA | 791.201 | operator prototype | rejected_same_session_refresh | Use as a Triton comparison row until a trainer-callable integration beats the current provider. |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 528.059 | Triton | 529.186 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | global_norm_squared | `params=124475904` | CUDA | 185.069 | Torch | 2262.349 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |

## Resolved Selected Backend Decisions

| Suite | Kernel | Shape | Selected stack | Status | Decision |
|---|---|---|---|---|---|
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| layernorm | backward_dinput | `N=65536 C=768` | Torch native | partial_backward_only | Do not promote the dInput-only Torch row. The full Torch native backward with dweight/dbias is slower than CUDA, the full Triton atomic prototype is also slower plus has an FP32 gradient-buffer contract mismatch, and the focused Torch dInput-plus-BF16-grads hybrid is much slower than the CUDA full-backward baseline. |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | rejected_x10_selector | Do not broaden the SM120 direct-cuBLAS dInput selector to attention projection; the x10 trainer stability round regressed, so the source default remains the huge-N LM-head-only direct-cuBLAS dInput route. |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | rejected_x10_selector | Do not broaden the SM120 direct-cuBLAS dInput selector to the GPT-2 MLP-up row. The microbench row can favor cuBLAS, but the broader direct-cuBLAS dInput selector regressed in the x10 TinyStories stability gate, while the stream-sync default keeps cuBLASLt for this row and improves the trainer smoke. |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | noise_floor_microbench_flip | Do not promote the qkv dInput cuBLAS microbench flip as a trainer default without a trainer smoke. The refreshed benchmark-only round picked cuBLAS by about 0.2%, while the stable x10 selection artifact has cuBLASLt ahead for the same row. |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | rejected_x10_selector | Keep the cuBLASLt fused dGELU trainer route as the default. The opt-in TK exact-dGELU selector now passes correctness and has a focused row win, but its x10 TinyStories stability round regressed versus the current stable default. |
| runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | rejected_slower_than_trainer_baseline | Rejected for trainer promotion; Torch fp32-state AdamW is materially slower than the CUDA trainer route. |
| runtime | bias_add | `BT=65536 OC=3072` | Triton | library_integration_not_justified | Keep as operator evidence; the refreshed Triton edge is about 0.2% and not enough to justify a trainer-callable Triton route. |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch | native_replacement_rejected | Keep as operator evidence; raw-pointer LibTorch preserves most of the copy edge, but Python Torch remains the fastest observed row and a linked trainer smoke is still required before promotion. |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch C++ | library_integration_not_justified | Keep as C++ API feasibility evidence; the LibTorch route is tie-range and does not justify a trainer dependency. |
| runtime | cuda_memset | `logits_elems=3296722944` | Torch | native_replacement_rejected | Keep as operator evidence; native CUDA replacement was slower than CUDA runtime. |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | rejected_same_session_refresh | Rejected for trainer promotion; refreshed CUDA GELU backward was faster than Triton. |

## Promotion Backlog

No active promotion candidates remain after applying `dev/sm120_promotion_decisions.json`.

## Resolved Promotion Decisions

| Class | Suite | Kernel | Shape | Winning stack | Status | Decision |
|---|---|---|---|---|---|---|
| native/codegen integration | runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | rejected_same_session_refresh | Rejected for trainer promotion; refreshed CUDA GELU backward was faster than Triton. |
| native/codegen integration | runtime | bias_add | `BT=65536 OC=3072` | Triton | library_integration_not_justified | Keep as operator evidence; the refreshed Triton edge is about 0.2% and not enough to justify a trainer-callable Triton route. |
| library integration | runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | rejected_slower_than_trainer_baseline | Rejected for trainer promotion; Torch fp32-state AdamW is materially slower than the CUDA trainer route. |
| library integration | runtime | cuda_memset | `logits_elems=3296722944` | Torch | native_replacement_rejected | Keep as operator evidence; native CUDA replacement was slower than CUDA runtime. |
| library integration | runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch | native_replacement_rejected | Keep as operator evidence; raw-pointer LibTorch preserves most of the copy edge, but Python Torch remains the fastest observed row and a linked trainer smoke is still required before promotion. |
| library integration | runtime | cuda_memset | `hidden_elems=50331648` | Torch C++ | library_integration_not_justified | Keep as C++ API feasibility evidence; the LibTorch route is tie-range and does not justify a trainer dependency. |
| layout rewrite | attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| layout rewrite | attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| reference/state gap | layernorm | backward_dinput | `N=65536 C=768` | Torch native | partial_backward_only | Do not promote the dInput-only Torch row. The full Torch native backward with dweight/dbias is slower than CUDA, the full Triton atomic prototype is also slower plus has an FP32 gradient-buffer contract mismatch, and the focused Torch dInput-plus-BF16-grads hybrid is much slower than the CUDA full-backward baseline. |
| non-trainer shape | layernorm | backward_dinput | `N=65536 C=3072` | Triton dInput-only | non_trainer_shape | Not an active trainer-promotion target for GPT-2 124M. |
| contract mismatch | runtime | adamw_update_bf16_state | `params=124475904 no-master` | Torch | contract_mismatch | Not active until the candidate matches the trainer state contract. |

## Attention Route Totals

| Shape | Stack | Scope | Trainer-layout | Forward (us) | Backward (us) | Total (us) | Complete | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | packed trainer-layout route | True | 785.718 | 2743.201 | 3528.919 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | cuDNNPacked | packed trainer-layout route | True | 805.997 | 2819.398 | 3625.395 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | TorchPacked | packed trainer-layout route | True | 1149.664 | 4050.301 | 5199.965 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | TorchMaterializedPacked | packed trainer-layout route | True | 1262.984 | 4215.184 | 5478.168 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | TritonPacked | packed trainer-layout route | True | 2205.301 | - | - | False | packed attention backward is not implemented in this Triton prototype |
| `B=64 T=1024 C=768 NH=12 HS=64` | Torch | separated Q/K/V reference route | False | 556.573 | 2196.240 | 2752.813 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | cuDNN | separated Q/K/V reference route | False | 675.635 | 2404.397 | 3080.032 | True |  |
| `B=64 T=1024 C=768 NH=12 HS=64` | Triton | separated Q/K/V reference route | False | 2035.579 | - | - | False | attention backward is not implemented in this Triton prototype |

## Benchmark Candidates

| Suite | Kernel | Shape | Stack | Time (us) |
|---|---|---|---|---:|
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1096.170 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1041.660 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1460.600 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1093.140 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1021.950 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1012.440 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1467.870 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1114.810 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 997.280 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1463.750 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1130.740 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 998.520 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 376.430 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 371.910 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 483.550 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 381.190 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 367.380 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 365.930 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 565.640 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 375.780 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 328.780 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 544.370 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 378.920 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 332.650 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK fused | 1563.540 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK explicit | 1997.380 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1471.130 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS explicit | 2476.110 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1464.500 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1384.880 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1364.900 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1746.170 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1493.790 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1317.020 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1755.440 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1499.300 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1315.250 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1434.320 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1379.880 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1555.300 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1540.890 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1386.980 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1380.010 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1822.440 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt fused | 1828.250 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt explicit | 2193.040 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS explicit | 2187.030 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1748.650 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1485.430 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1311.440 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1744.180 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1496.180 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1318.570 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 27869.090 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 22396.620 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22411.320 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 24022.460 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21792.800 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21272.170 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26145.540 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20899.630 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21258.150 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26120.880 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20986.710 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21346.560 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1455.560 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1027.640 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1011.020 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1017.400 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 516.910 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 372.550 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 338.820 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 345.340 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 2442.020 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1383.700 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1352.330 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1376.640 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1601.290 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1399.880 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 28230.190 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1368.800 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1411.150 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 22768.190 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21444.640 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21808.770 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21527.580 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch C++ | 1353.160 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch C++ | 1367.160 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2024.950 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2281.490 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2123.940 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2142.690 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 662.040 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 679.120 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 558.290 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 561.240 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 2601.840 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 3026.490 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 2240.220 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 2251.600 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 3065.930 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2661.390 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 3202.090 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2259.920 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2257.020 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 44015.870 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 49289.260 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 69907.490 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 70181.790 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 785.718 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 2743.201 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 556.573 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 2196.240 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchPacked | 1149.664 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchPacked | 4050.301 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchMaterializedPacked | 1262.984 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchMaterializedPacked | 4215.184 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNN | 675.635 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNN | 2404.397 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNNPacked | 805.997 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNNPacked | 2819.398 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Triton | 2035.579 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TritonPacked | 2205.301 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | Torch | 17713.633 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | Triton | 8246.336 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | Triton | 22228.512 |
| layernorm | forward | `N=65536 C=768` | CUDA | 138.406 |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 275.342 |
| layernorm | backward | `N=65536 C=768` | CUDA | 288.147 |
| layernorm | forward | `N=65536 C=3072` | CUDA | 544.301 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | CUDA | 1082.597 |
| layernorm | backward | `N=65536 C=3072` | CUDA | 1272.855 |
| layernorm | forward | `N=65536 C=768` | Triton | 176.384 |
| layernorm | forward | `N=65536 C=768` | Torch native | 154.944 |
| layernorm | forward | `N=65536 C=768` | Torch stats | 2181.408 |
| layernorm | backward_dinput | `N=65536 C=768` | Triton dInput-only | 223.328 |
| layernorm | backward_dinput | `N=65536 C=768` | Torch native | 214.592 |
| layernorm | backward | `N=65536 C=768` | Torch native+BF16-grads | 1959.360 |
| layernorm | backward | `N=65536 C=768` | Torch native | 415.360 |
| layernorm | backward | `N=65536 C=768` | Triton atomic FP32-grad | 364.992 |
| layernorm | fused_residual_forward | `N=65536 C=768` | Triton | 322.912 |
| layernorm | fused_residual_forward | `N=65536 C=768` | Torch native | 331.648 |
| layernorm | fused_residual_forward | `N=65536 C=768` | Torch stats | 3180.320 |
| layernorm | forward | `N=65536 C=3072` | Triton | 571.648 |
| layernorm | forward | `N=65536 C=3072` | Torch native | 547.456 |
| layernorm | forward | `N=65536 C=3072` | Torch stats | 8947.072 |
| layernorm | backward_dinput | `N=65536 C=3072` | Triton dInput-only | 801.056 |
| layernorm | backward_dinput | `N=65536 C=3072` | Torch native | 819.136 |
| layernorm | backward | `N=65536 C=3072` | Torch native+BF16-grads | 7936.800 |
| layernorm | backward | `N=65536 C=3072` | Torch native | 1383.520 |
| layernorm | backward | `N=65536 C=3072` | Triton atomic FP32-grad | 1418.048 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Triton | 1106.144 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Torch native | 1305.024 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Torch stats | 13156.224 |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 80.025 |
| runtime | bias_add | `BT=65536 OC=3072` | CUDA | 548.129 |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 528.059 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | CUDA | 791.201 |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 24.630 |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 186.802 |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 245.528 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 3953.133 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 8942.509 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | 3958.458 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA kernel | 4115.859 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA runtime | 8777.522 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA kernel | 9188.391 |
| runtime | global_norm_squared | `params=124475904` | CUDA | 185.069 |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1809.206 |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 79.286 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA runtime | 60.164 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA kernel | 61.882 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 131.615 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA kernel | 137.148 |
| runtime | bias_add | `BT=65536 OC=768` | Triton | 133.016 |
| runtime | bias_add | `BT=65536 OC=3072` | Triton | 542.899 |
| runtime | gelu_forward | `BT=65536 C=3072` | Triton | 529.186 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | 781.888 |
| runtime | bias_add | `BT=65536 OC=768` | Torch | 135.256 |
| runtime | bias_add | `BT=65536 OC=3072` | Torch | 548.820 |
| runtime | gelu_forward | `BT=65536 C=3072` | Torch | 539.764 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Torch | 26825.635 |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | Torch | 320.944 |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | Torch | 971.398 |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | Torch | 1305.440 |
| runtime | global_norm_squared | `params=124475904` | Torch | 2262.349 |
| runtime | adamw_update_bf16_state | `params=124475904 no-master` | Torch | 1208.083 |
| runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | 7298.912 |
| runtime | encoder_forward | `B=64 T=1024 C=768` | Torch | 201.555 |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch | 59.912 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | Torch | 132.043 |
| runtime | cuda_memset | `logits_elems=3296722944` | Torch | 3943.104 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch | 8662.848 |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch C++ | 59.875 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | Torch C++ | 131.805 |
| runtime | cuda_memset | `logits_elems=3296722944` | Torch C++ | 3952.768 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch C++ | 8681.632 |

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
| runtime | cuda_memset | `logits_elems=3296722944` | Triton | not implemented in this Triton runtime prototype |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | Triton | not implemented in this Triton runtime prototype |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Triton | not implemented in this Triton runtime prototype |

