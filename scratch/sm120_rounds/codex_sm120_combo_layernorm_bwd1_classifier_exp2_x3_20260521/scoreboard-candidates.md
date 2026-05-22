# SM120 Round Metrics - codex_sm120_combo_layernorm_bwd1_classifier_exp2_x3_20260521

- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_layernorm_bwd1_classifier_exp2_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_layernorm_bwd1_classifier_exp2_x3_20260521`
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

- detailed matrix: `scratch/sm120_rounds/codex_sm120_combo_layernorm_bwd1_classifier_exp2_x3_20260521/backend-stacks.json`

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

- manifest: `scratch/sm120_rounds/codex_sm120_combo_layernorm_bwd1_classifier_exp2_x3_20260521/round-manifest.json`
- device arch: `SM120`
- build jobs: `4`
- changed paths: `515`

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

## Selected Backend Rows

| Suite | Kernel | Shape | Selected stack | Time (us) | Next stack | Next time (us) | Use scope | Decision status | Decision note |
|---|---|---|---|---:|---|---:|---|---|---|
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 2754.156 | - | - | current packed trainer route | - | Current C++ attention route; keep unless a packed-QKV candidate beats it and passes the TinyStories smoke gate. |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 786.649 | - | - | current packed trainer route | - | Current C++ attention route; keep unless a packed-QKV candidate beats it and passes the TinyStories smoke gate. |
| layernorm | backward | `N=65536 C=3072` | CUDA | 1124.940 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | backward | `N=65536 C=768` | CUDA | 276.144 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | forward | `N=65536 C=3072` | CUDA | 557.650 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | forward | `N=65536 C=768` | CUDA | 138.780 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | fused_residual_forward | `N=65536 C=3072` | CUDA | 1100.996 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 284.451 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 365.550 | cuBLASLt | 390.450 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1351.340 | cuBLASLt | 1384.560 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1377.780 | cuBLAS | 1397.970 | C++ benchmark route | benchmark_context_flip | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21528.470 | cuBLASLt | 21961.480 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1012.150 | cuBLAS | 1012.220 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt fused | 1831.180 | TK | 1839.820 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 329.180 | cuBLASLt | 376.230 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1352.540 | cuBLASLt | 1571.290 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1309.640 | cuBLASLt | 1493.950 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21049.320 | cuBLAS | 21372.790 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 993.060 | cuBLASLt | 1111.430 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 334.440 | cuBLASLt | 379.570 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1310.840 | cuBLASLt | 1503.620 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1343.110 | cuBLASLt | 1494.630 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20981.440 | cuBLAS | 21395.790 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 995.260 | cuBLASLt | 1131.850 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 369.630 | TK | 376.720 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1347.100 | TK | 1419.750 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 22346.090 | cuBLAS | 22441.920 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1073.890 | cuBLASLt | 1093.790 | C++ benchmark route | benchmark_context_flip | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1528.710 | TK fused | 1588.000 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1815.603 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_add | `BT=65536 OC=3072` | CUDA | 537.390 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 83.020 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 187.819 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 262.784 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 24.770 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 133.924 | CUDA kernel | 138.220 | CUDA benchmark route | profiler_only_runtime_row | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA runtime | 8894.637 | CUDA kernel | 9538.963 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_memset | `grad_elems=124475904` | CUDA runtime | 156.776 | CUDA kernel | 162.011 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA runtime | 64.043 | CUDA kernel | 67.705 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | 4252.678 | CUDA kernel | 4360.262 | CUDA benchmark route | profiler_only_runtime_row | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 87.000 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 9028.659 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 4052.775 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | CUDA | 794.246 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 536.776 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | global_norm_squared | `params=124475904` | CUDA | 186.296 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |

## Resolved Selected Backend Decisions

| Suite | Kernel | Shape | Selected stack | Status | Decision |
|---|---|---|---|---|---|
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | rejected_x10_selector | Do not broaden the SM120 direct-cuBLAS dInput selector to attention projection. Both the bundled attproj/MLP-up selector and the later attproj-only selector regressed in x10 TinyStories stability gates, so the source default remains the huge-N LM-head-only direct-cuBLAS dInput route. |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | rejected_x10_selector | Do not promote the SM120 direct-cuBLAS dInput selector to the GPT-2 MLP-up row. The microbench row can favor cuBLAS, but both the broader attproj+MLP-up selector and the later FC-only selector regressed in x10 TinyStories stability gates, while the stream-sync default keeps cuBLASLt for this row and improves the trainer smoke. |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | benchmark_context_flip | Keep the training-backed current native row on cuBLASLt for the GPT-2 MLP projection dInput path. The earlier current-source x10 native round selected cuBLAS, but the stream-sync x10 native round and optional-stack benchmark both select cuBLASLt; the prior flip is now resolved by the newer trainer-backed evidence. |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | benchmark_context_flip | Keep qkv forward on the cuBLASLt trainer default. The correctness-enabled optional refresh picked TK by a small edge, but the current trainer-backed x10 native round selected cuBLASLt for the same row and no route-specific TinyStories smoke proves a switch. |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | profiler_only_runtime_row | Keep as benchmark evidence only. The refreshed optional round measured CUDA runtime as fastest for this profiler-only copy shape, but it is not a current trainer call path to promote. |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | profiler_only_runtime_row | Keep as profiler/runtime evidence only. CUDA runtime remains the fastest current logits-sized memset row, but the current GPT-2 trainer has no logits-sized memset call-site to promote. |

## Attention Route Totals

| Shape | Stack | Scope | Trainer-layout | Forward (us) | Backward (us) | Total (us) | Complete | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | packed trainer-layout route | True | 786.649 | 2754.156 | 3540.805 | True |  |

## Benchmark Candidates

| Suite | Kernel | Shape | Stack | Time (us) |
|---|---|---|---|---:|
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1073.890 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1093.790 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1415.020 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1138.760 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1012.150 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1012.220 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1482.070 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1111.430 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 993.060 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1481.530 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1131.850 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 995.260 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 376.720 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 369.630 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 484.830 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 408.560 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 390.450 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 365.550 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 538.480 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 376.230 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 329.180 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 545.480 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 379.570 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 334.440 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK fused | 1588.000 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK explicit | 2004.470 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1528.710 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS explicit | 2479.370 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1448.900 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1384.560 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1351.340 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1723.440 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1571.290 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1352.540 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1714.960 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1503.620 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1310.840 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1419.750 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1347.100 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1592.220 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1500.300 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1377.780 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1397.970 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1839.820 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt fused | 1831.180 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt explicit | 2189.390 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS explicit | 2186.560 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1708.440 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1493.950 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1309.640 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1738.860 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1494.630 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1343.110 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 27639.020 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 22346.090 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22441.920 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 24236.960 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21961.480 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21528.470 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26095.330 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21049.320 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21372.790 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26195.770 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20981.440 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21395.790 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 786.649 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 2754.156 |
| layernorm | forward | `N=65536 C=768` | CUDA | 138.780 |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 284.451 |
| layernorm | backward | `N=65536 C=768` | CUDA | 276.144 |
| layernorm | forward | `N=65536 C=3072` | CUDA | 557.650 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | CUDA | 1100.996 |
| layernorm | backward | `N=65536 C=3072` | CUDA | 1124.940 |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 83.020 |
| runtime | bias_add | `BT=65536 OC=3072` | CUDA | 537.390 |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 536.776 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | CUDA | 794.246 |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 24.770 |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 187.819 |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 262.784 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 4052.775 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 9028.659 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | 4252.678 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA kernel | 4360.262 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA runtime | 8894.637 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA kernel | 9538.963 |
| runtime | cuda_memset | `grad_elems=124475904` | CUDA runtime | 156.776 |
| runtime | cuda_memset | `grad_elems=124475904` | CUDA kernel | 162.011 |
| runtime | global_norm_squared | `params=124475904` | CUDA | 186.296 |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1815.603 |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 87.000 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA runtime | 64.043 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA kernel | 67.705 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 133.924 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA kernel | 138.220 |

## Training Smoke

- use_master_weights: `disabled`
- gelu_fusion: `1`
- total average iteration time: `2504.913 ms`
- final val loss: `10.609911`
- final step: `3/3`, loss `10.811316`, `2507.87 ms`, `209298 tok/s`

| Step | Loss | Norm | LR | Time (ms) | Tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.032358 | 22.1414 | 8.57e-07 | 2510.32 | 208853 |
| 2 | 10.958507 | 22.0968 | 1.71e-06 | 2501.96 | 209551 |
| 3 | 10.811316 | 21.1251 | 2.57e-06 | 2507.87 | 209298 |

