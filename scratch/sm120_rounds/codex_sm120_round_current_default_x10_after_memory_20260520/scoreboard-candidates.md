# SM120 Round Metrics - codex_sm120_round_current_default_x10_after_memory_20260520

- artifact dir: `scratch/sm120_rounds/codex_sm120_round_current_default_x10_after_memory_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_current_default_x10_after_memory_20260520`
- git commit: `0f21747`

## Backend Stack Probe

| Stack | Status | Candidate use | Next action |
|---|---|---|---|
| ThunderKittens 2.0 | available | native TK kernels and current SM120 packed-QKV attention path | benchmark against cuBLASLt/plain CUDA by shape before promoting TK-only wins |
| Plain CUDA | available | plain CUDA baselines and C++ benchmarks | run the SM120 round on the RTX 5090 target for runtime timings |
| GPU runtime | available | runtime timing and correctness execution | use explicit correctness and benchmark logs as the runtime evidence source |
| cuBLAS | available | baseline GEMM comparison where cuBLASLt epilogues are not needed | add explicit cuBLAS benchmark/parity rows before selecting it over cuBLASLt |
| cuBLASLt | available | current SM120 GEMM baseline and fused GEMM epilogues | keep benchmark rows shape-specific; do not switch global defaults from one isolated win |
| cuDNN | available | attention alternatives through detected headers/libs; GPT-2 BF16 shape support still needs benchmark proof | prototype as an opt-in benchmark first; current v1 build contract intentionally avoids -lcudnn |
| Triton | available | attention, normalization, elementwise fusion, and GEMM candidates | add stack-specific parity tests before trainer promotion |
| Torch | available | PyTorch operator kernels for exact family-by-family backend comparisons | add stack-specific parity tests before trainer promotion |
| CuTeDSL | available | Blackwell GEMM and fused epilogue candidates | add stack-specific parity tests before trainer promotion |

## Backend Family-Stack Matrix

- detailed matrix: `scratch/sm120_rounds/codex_sm120_round_current_default_x10_after_memory_20260520/backend-stacks.json`

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

- manifest: `scratch/sm120_rounds/codex_sm120_round_current_default_x10_after_memory_20260520/round-manifest.json`
- device arch: `SM120`
- build jobs: `4`
- changed paths: `460`

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

## Selected Backend Rows

| Suite | Kernel | Shape | Selected stack | Time (us) | Next stack | Next time (us) | Use scope | Decision status | Decision note |
|---|---|---|---|---:|---|---:|---|---|---|
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 2732.582 | - | - | current packed trainer route | - | Current C++ attention route; keep unless a packed-QKV candidate beats it and passes the TinyStories smoke gate. |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 783.906 | - | - | current packed trainer route | - | Current C++ attention route; keep unless a packed-QKV candidate beats it and passes the TinyStories smoke gate. |
| layernorm | backward | `N=65536 C=768` | CUDA | 286.407 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | forward | `N=65536 C=768` | CUDA | 138.710 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 279.820 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 364.040 | cuBLASLt | 365.620 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1327.690 | cuBLASLt | 1355.490 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1396.660 | cuBLAS | 1431.500 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21250.510 | cuBLASLt | 21871.530 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1007.090 | cuBLAS | 1007.400 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt fused | 1827.450 | TK | 1833.460 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 326.940 | cuBLASLt | 374.900 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1309.120 | cuBLASLt | 1520.670 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1356.720 | cuBLASLt | 1470.140 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20894.130 | cuBLAS | 21274.310 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 988.490 | cuBLASLt | 1102.880 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 329.920 | cuBLASLt | 376.760 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1317.570 | cuBLASLt | 1507.310 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1316.880 | cuBLASLt | 1495.990 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21000.520 | cuBLAS | 21212.980 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1032.110 | cuBLASLt | 1125.720 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 368.920 | TK | 374.980 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1390.710 | TK | 1419.640 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22169.920 | cuBLASLt | 22336.230 | C++ benchmark route | rejected_trainer_smoke | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1037.310 | TK | 1099.120 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1489.730 | TK fused | 1578.700 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1856.336 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_add | `BT=65536 OC=3072` | CUDA | 554.136 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 82.143 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 187.960 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 247.230 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 24.464 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 133.575 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA runtime | 8867.174 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA runtime | 63.199 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | 4194.509 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 81.052 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 8985.197 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 3972.230 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | CUDA | 799.103 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 535.077 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | global_norm_squared | `params=124475904` | CUDA | 185.869 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |

## Resolved Selected Backend Decisions

| Suite | Kernel | Shape | Selected stack | Status | Decision |
|---|---|---|---|---|---|
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | rejected_x10_selector | Do not broaden the SM120 direct-cuBLAS dInput selector to attention projection; the x10 trainer stability round regressed, so the source default remains the huge-N LM-head-only direct-cuBLAS dInput route. |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | rejected_trainer_smoke | Keep LM-head forward on cuBLASLt; the opt-in direct-cuBLAS forward selector passed focused gates but regressed in TinyStories trainer timing. |

## Benchmark Candidates

| Suite | Kernel | Shape | Stack | Time (us) |
|---|---|---|---|---:|
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1099.120 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1037.310 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1461.540 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1087.160 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1007.090 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1007.400 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1454.910 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1102.880 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 988.490 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1463.740 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1125.720 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1032.110 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 374.980 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 368.920 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 503.210 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 379.480 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 365.620 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 364.040 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 540.930 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 374.900 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 326.940 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 546.880 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 376.760 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 329.920 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK fused | 1578.700 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK explicit | 2001.380 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1489.730 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS explicit | 2471.910 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1450.790 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1355.490 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1327.690 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1750.610 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1520.670 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1309.120 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1749.590 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1507.310 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1317.570 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1419.640 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1390.710 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1548.330 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1476.550 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1396.660 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1431.500 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1833.460 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt fused | 1827.450 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt explicit | 2194.740 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS explicit | 2183.360 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1758.200 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1470.140 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1356.720 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1719.920 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1495.990 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1316.880 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 27733.370 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 22336.230 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22169.920 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 23877.200 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21871.530 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21250.510 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26115.620 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20894.130 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21274.310 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26019.310 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21000.520 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21212.980 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 783.906 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 2732.582 |
| layernorm | forward | `N=65536 C=768` | CUDA | 138.710 |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 279.820 |
| layernorm | backward | `N=65536 C=768` | CUDA | 286.407 |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 82.143 |
| runtime | bias_add | `BT=65536 OC=3072` | CUDA | 554.136 |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 535.077 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | CUDA | 799.103 |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 24.464 |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 187.960 |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 247.230 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 3972.230 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 8985.197 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | 4194.509 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA runtime | 8867.174 |
| runtime | global_norm_squared | `params=124475904` | CUDA | 185.869 |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1856.336 |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 81.052 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA runtime | 63.199 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 133.575 |

## Training Smoke

- use_master_weights: `disabled`
- gelu_fusion: `1`
- total average iteration time: `2495.443 ms`
- final val loss: `9.483727`
- final step: `10/10`, loss `9.588612`, `2501.19 ms`, `210052 tok/s`

| Step | Loss | Norm | LR | Time (ms) | Tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.032358 | 22.1414 | 8.57e-07 | 2494.27 | 210197 |
| 2 | 10.958507 | 22.0968 | 1.71e-06 | 2486.98 | 210813 |
| 3 | 10.811316 | 21.1251 | 2.57e-06 | 2489.58 | 210700 |
| 4 | 10.610130 | 18.7014 | 3.43e-06 | 2491.65 | 210601 |
| 5 | 10.392586 | 15.0184 | 4.29e-06 | 2498.79 | 210390 |
| 6 | 10.186255 | 12.0843 | 5.14e-06 | 2495.76 | 210319 |
| 7 | 10.010621 | 10.2002 | 6e-06 | 2495.83 | 210272 |
| 8 | 9.855870 | 8.7905 | 6.86e-06 | 2500.51 | 210172 |
| 9 | 9.719423 | 7.4665 | 7.71e-06 | 2498.70 | 210120 |
| 10 | 9.588612 | 6.3099 | 8.57e-06 | 2501.19 | 210052 |

