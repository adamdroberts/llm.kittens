# SM120 Round Metrics - codex_sm120_round_python_stacks_selection_20260520_rerun

- artifact dir: `scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun`
- train output dir: `log124M/5090_S_codex_sm120_round_python_stacks_selection_20260520_rerun`
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

- detailed matrix: `scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/backend-stacks.json`

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

- manifest: `scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun/round-manifest.json`
- device arch: `SM120`
- build jobs: `4`
- changed paths: `475`

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

| Suite | Kernel | Shape | Selected stack | Time (us) | Next stack | Next time (us) | Use scope | Decision note |
|---|---|---|---|---:|---|---:|---|---|
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 2229.728 | cuDNN | 2385.046 | python separated-Q/K/V | Use Torch SDPA for already-separated Q/K/V experiments; packed trainer selection must compare TorchPacked. |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 584.058 | cuDNN | 667.986 | python separated-Q/K/V | Use Torch SDPA for already-separated Q/K/V experiments; packed trainer selection must compare TorchPacked. |
| layernorm | backward | `N=65536 C=3072` | Torch native | 1395.008 | Triton atomic FP32-grad | 1423.104 | trainer-compatible prototype | Native Torch backward consumes saved mean/rstd and produces dInput/dweight/dbias; trainer use still needs a libtorch/C++ route gate. |
| layernorm | backward | `N=65536 C=768` | CUDA | 304.621 | Triton atomic FP32-grad | 363.680 | CUDA benchmark route | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | backward_dinput | `N=65536 C=3072` | Triton dInput-only | 833.824 | Torch native | 838.336 | partial backward prototype | Triton dInput-only is useful for backward decomposition work but needs dweight/dbias before trainer promotion. |
| layernorm | backward_dinput | `N=65536 C=768` | Torch native | 217.568 | Triton dInput-only | 228.320 | partial backward prototype | Torch native dInput-only output-mask row is useful for backward decomposition work but needs dweight/dbias before trainer promotion. |
| layernorm | forward | `N=65536 C=3072` | Torch native | 556.352 | Triton | 575.584 | reference only | Native Torch is useful where saved mean/rstd are not needed; trainer LayerNorm still needs stats-compatible state. |
| layernorm | forward | `N=65536 C=768` | CUDA | 148.137 | Torch native | 153.024 | CUDA benchmark route | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Triton | 1119.424 | Torch native | 1324.096 | operator prototype | Use as a Triton comparison row until a trainer-callable integration beats the current provider. |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 295.547 | Triton | 310.464 | CUDA benchmark route | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 363.400 | cuBLASLt | 365.810 | C++ benchmark route | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1344.910 | cuBLAS | 1349.720 | C++ benchmark route | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1385.170 | cuBLAS | 1413.370 | C++ benchmark route | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21653.440 | cuBLAS | 22308.370 | operator prototype | Torch is the fastest observed operator row here; trainer use still needs an explicit libtorch/C++ integration gate. |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1011.630 | Torch | 1023.350 | C++ benchmark route | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1835.320 | cuBLASLt fused | 1843.000 | C++ benchmark route | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 326.790 | Torch | 340.490 | C++ benchmark route | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1309.680 | Torch | 1477.070 | C++ benchmark route | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1309.740 | Torch | 1366.900 | C++ benchmark route | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21231.940 | cuBLASLt | 22131.210 | operator prototype | Torch is the fastest observed operator row here; trainer use still needs an explicit libtorch/C++ integration gate. |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1031.070 | Torch | 1055.740 | C++ benchmark route | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 330.080 | Torch | 348.060 | C++ benchmark route | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1355.090 | Torch | 1483.650 | C++ benchmark route | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1328.880 | Torch | 1420.920 | C++ benchmark route | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21523.680 | cuBLASLt | 22043.050 | operator prototype | Torch is the fastest observed operator row here; trainer use still needs an explicit libtorch/C++ integration gate. |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1017.820 | cuBLAS | 1033.190 | operator prototype | Torch is the fastest observed operator row here; trainer use still needs an explicit libtorch/C++ integration gate. |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 369.760 | TK | 374.620 | C++ benchmark route | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1406.550 | TK | 1431.670 | C++ benchmark route | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 22502.400 | cuBLAS | 24966.830 | operator prototype | Torch is the fastest observed operator row here; trainer use still needs an explicit libtorch/C++ integration gate. |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1063.340 | TK | 1069.910 | C++ benchmark route | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1517.540 | TK fused | 1555.630 | C++ benchmark route | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1926.957 | - | - | CUDA benchmark route | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | 7449.536 | - | - | operator prototype | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | adamw_update_bf16_state | `params=124475904 no-master` | Torch | 1215.200 | - | - | non-equivalent BF16-state reference | Torch fused AdamW is usable only for BF16 moment-state experiments; the trainer default uses FP32 moment buffers. |
| runtime | bias_add | `BT=65536 OC=3072` | Torch | 538.298 | Triton | 538.710 | operator prototype | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 86.808 | Triton | 134.648 | CUDA benchmark route | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 218.741 | Torch | 1040.090 | CUDA benchmark route | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 261.328 | Torch | 1326.208 | CUDA benchmark route | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 24.507 | Torch | 322.966 | CUDA benchmark route | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | Torch | 134.291 | CUDA runtime | 143.239 | operator prototype | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch | 8849.952 | CUDA runtime | 9462.322 | operator prototype | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch | 63.554 | CUDA runtime | 67.214 | operator prototype | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | cuda_memset | `logits_elems=3296722944` | Torch | 4190.176 | CUDA runtime | 4455.706 | operator prototype | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 91.999 | Torch | 202.763 | CUDA benchmark route | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 9540.025 | Triton | 21981.632 | CUDA benchmark route | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 4354.061 | Triton | 8379.872 | CUDA benchmark route | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | 792.522 | CUDA | 854.545 | operator prototype | Use as a Triton comparison row until a trainer-callable integration beats the current provider. |
| runtime | gelu_forward | `BT=65536 C=3072` | Torch | 538.702 | Triton | 550.768 | operator prototype | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | global_norm_squared | `params=124475904` | CUDA | 198.059 | Torch | 2348.851 | CUDA benchmark route | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |

## Promotion Backlog

No active promotion candidates remain after applying `dev/sm120_promotion_decisions.json`.

## Resolved Promotion Decisions

| Class | Suite | Kernel | Shape | Winning stack | Status | Decision |
|---|---|---|---|---|---|---|
| native/codegen integration | runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | rejected_same_session_refresh | Rejected for trainer promotion; refreshed CUDA GELU backward was faster than Triton. |
| library integration | matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | rejected_same_session_refresh | Rejected for libtorch/native replacement work; refreshed C++ cuBLASLt was equal or faster. |
| library integration | runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch | native_replacement_rejected | Keep as operator evidence; native CUDA replacement was slower than CUDA runtime. |
| library integration | runtime | cuda_copy_d2d | `hidden_elems=50331648` | Torch | native_replacement_rejected | Keep as operator evidence; native CUDA replacement was slower than CUDA runtime. |
| library integration | runtime | cuda_memset | `logits_elems=3296722944` | Torch | native_replacement_rejected | Keep as operator evidence; native CUDA replacement was slower than CUDA runtime. |
| library integration | runtime | cuda_memset | `hidden_elems=50331648` | Torch | native_replacement_rejected | Keep as operator evidence; native CUDA replacement was slower than CUDA runtime. |
| library integration | matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | rejected_same_session_refresh | Rejected for libtorch/native replacement work; refreshed C++ cuBLASLt was faster. |
| library integration | matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | rejected_same_session_refresh | Rejected for libtorch/native replacement work; refreshed C++ cuBLAS was equal or faster. |
| library integration | matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | rejected_same_session_refresh | Rejected for libtorch/native replacement work; refreshed C++ cuBLASLt was faster. |
| library integration | runtime | gelu_forward | `BT=65536 C=3072` | Torch | library_integration_not_justified | Keep as operator evidence; the refreshed Torch win is too small to justify adding a libtorch trainer route, and native CUDA block-size retunes did not beat Torch. |
| library integration | matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | rejected_same_session_refresh | Rejected for trainer promotion; refreshed C++ cuBLAS was faster than Torch for qkv accumulated dWeight. |
| library integration | runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | rejected_slower_than_trainer_baseline | Rejected for trainer promotion; Torch fp32-state AdamW is materially slower than the CUDA trainer route. |
| library integration | runtime | bias_add | `BT=65536 OC=3072` | Torch | library_integration_not_justified | Keep as operator evidence; the refreshed Torch edge is sub-percent and not enough to justify a libtorch trainer route. |
| layout rewrite | attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| layout rewrite | attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| reference/state gap | layernorm | backward_dinput | `N=65536 C=768` | Torch native | partial_backward_only | Do not promote the dInput-only Torch row. The full Torch native backward with dweight/dbias is slower than CUDA, and the full Triton atomic prototype is also slower plus has an FP32 gradient-buffer contract mismatch. |
| non-trainer shape | layernorm | fused_residual_forward | `N=65536 C=3072` | Triton | non_trainer_shape | Not a GPT-2 124M trainer LayerNorm width, and refreshed CUDA was faster. |
| non-trainer shape | layernorm | forward | `N=65536 C=3072` | Torch native | non_trainer_shape | Not a GPT-2 124M trainer LayerNorm width; keep as operator stress evidence. |
| non-trainer shape | layernorm | backward | `N=65536 C=3072` | Torch native | non_trainer_shape | Not an active trainer-promotion target for GPT-2 124M. |
| non-trainer shape | layernorm | backward_dinput | `N=65536 C=3072` | Triton dInput-only | non_trainer_shape | Not an active trainer-promotion target for GPT-2 124M. |
| contract mismatch | runtime | adamw_update_bf16_state | `params=124475904 no-master` | Torch | contract_mismatch | Not active until the candidate matches the trainer state contract. |

## Benchmark Candidates

| Suite | Kernel | Shape | Stack | Time (us) |
|---|---|---|---|---:|
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1069.910 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1063.340 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1413.420 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1087.300 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1031.030 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1011.630 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1455.710 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1113.790 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1031.070 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1474.540 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1108.040 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1033.190 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 374.620 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 369.760 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 483.050 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 383.240 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 365.810 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 363.400 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 538.780 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 375.210 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 326.790 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 542.820 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 399.440 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 330.080 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK fused | 1555.630 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK explicit | 2006.490 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1517.540 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS explicit | 2450.270 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1476.580 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1344.910 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1349.720 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1744.070 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1538.380 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1309.680 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1776.370 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1491.030 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1355.090 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1431.670 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1406.550 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1620.900 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1476.490 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1385.170 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1413.370 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1835.320 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt fused | 1843.000 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt explicit | 2189.760 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS explicit | 2202.910 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1729.090 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1513.040 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1309.740 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1780.470 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1489.360 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1328.880 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 27889.370 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 25633.510 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 24966.830 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 25379.780 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 23086.790 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22308.370 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 27646.830 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 22131.210 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22432.050 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 27758.510 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 22043.050 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22252.330 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1468.310 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1023.350 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1055.740 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1017.820 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 520.310 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 373.380 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 340.490 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 348.060 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 2498.550 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1491.920 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1477.070 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1483.650 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1748.620 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1534.720 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 28723.120 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1366.900 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1420.920 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 22502.400 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21653.440 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21231.940 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21523.680 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2262.530 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2504.180 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2303.580 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2316.410 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 734.500 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 721.440 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 578.220 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 585.800 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 3169.320 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 3299.180 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 2471.110 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 2475.830 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 3285.190 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2969.000 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 3618.690 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2344.230 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2347.280 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 49372.510 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 52579.710 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 75083.970 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 76736.100 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 825.886 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 2892.588 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 584.058 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 2229.728 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchPacked | 1069.602 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchPacked | 4144.983 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNN | 667.986 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNN | 2385.046 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNNPacked | 780.912 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNNPacked | 3468.032 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Triton | 2113.485 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TritonPacked | 2234.986 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | Torch | 17959.841 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | Triton | 8379.872 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | Triton | 21981.632 |
| layernorm | forward | `N=65536 C=768` | CUDA | 148.137 |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 295.547 |
| layernorm | backward | `N=65536 C=768` | CUDA | 304.621 |
| layernorm | forward | `N=65536 C=768` | Triton | 175.488 |
| layernorm | forward | `N=65536 C=768` | Torch native | 153.024 |
| layernorm | forward | `N=65536 C=768` | Torch stats | 2219.296 |
| layernorm | backward_dinput | `N=65536 C=768` | Triton dInput-only | 228.320 |
| layernorm | backward_dinput | `N=65536 C=768` | Torch native | 217.568 |
| layernorm | backward | `N=65536 C=768` | Torch native | 416.224 |
| layernorm | backward | `N=65536 C=768` | Triton atomic FP32-grad | 363.680 |
| layernorm | fused_residual_forward | `N=65536 C=768` | Triton | 310.464 |
| layernorm | fused_residual_forward | `N=65536 C=768` | Torch native | 341.248 |
| layernorm | fused_residual_forward | `N=65536 C=768` | Torch stats | 3227.456 |
| layernorm | forward | `N=65536 C=3072` | Triton | 575.584 |
| layernorm | forward | `N=65536 C=3072` | Torch native | 556.352 |
| layernorm | forward | `N=65536 C=3072` | Torch stats | 9081.056 |
| layernorm | backward_dinput | `N=65536 C=3072` | Triton dInput-only | 833.824 |
| layernorm | backward_dinput | `N=65536 C=3072` | Torch native | 838.336 |
| layernorm | backward | `N=65536 C=3072` | Torch native | 1395.008 |
| layernorm | backward | `N=65536 C=3072` | Triton atomic FP32-grad | 1423.104 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Triton | 1119.424 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Torch native | 1324.096 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Torch stats | 13122.016 |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 86.808 |
| runtime | bias_add | `BT=65536 OC=3072` | CUDA | 577.804 |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 572.824 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | CUDA | 854.545 |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 24.507 |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 218.741 |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 261.328 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 4354.061 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 9540.025 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | 4455.706 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA runtime | 9462.322 |
| runtime | global_norm_squared | `params=124475904` | CUDA | 198.059 |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1926.957 |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 91.999 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA runtime | 67.214 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 143.239 |
| runtime | bias_add | `BT=65536 OC=768` | Triton | 134.648 |
| runtime | bias_add | `BT=65536 OC=3072` | Triton | 538.710 |
| runtime | gelu_forward | `BT=65536 C=3072` | Triton | 550.768 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | 792.522 |
| runtime | bias_add | `BT=65536 OC=768` | Torch | 137.793 |
| runtime | bias_add | `BT=65536 OC=3072` | Torch | 538.298 |
| runtime | gelu_forward | `BT=65536 C=3072` | Torch | 538.702 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Torch | 27275.003 |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | Torch | 322.966 |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | Torch | 1040.090 |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | Torch | 1326.208 |
| runtime | global_norm_squared | `params=124475904` | Torch | 2348.851 |
| runtime | adamw_update_bf16_state | `params=124475904 no-master` | Torch | 1215.200 |
| runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | 7449.536 |
| runtime | encoder_forward | `B=64 T=1024 C=768` | Torch | 202.763 |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch | 63.554 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | Torch | 134.291 |
| runtime | cuda_memset | `logits_elems=3296722944` | Torch | 4190.176 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch | 8849.952 |

## Unavailable Backend Rows

| Suite | Kernel | Shape | Stack | Reason |
|---|---|---|---|---|
| matmul | cutedsl_gemm | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | CuTeDSL | local CuTeDSL BF16 grouped-GEMM path rejects sm_120a |
| matmul | cutedsl_gemm | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | CuTeDSL | local CuTeDSL BF16 grouped-GEMM path rejects sm_120a |
| matmul | cutedsl_gemm | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | CuTeDSL | local CuTeDSL BF16 grouped-GEMM path rejects sm_120a |
| matmul | cutedsl_gemm | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | CuTeDSL | local CuTeDSL BF16 grouped-GEMM path rejects sm_120a |
| matmul | cutedsl_gemm | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | CuTeDSL | local CuTeDSL BF16 grouped-GEMM path rejects sm_120a |

