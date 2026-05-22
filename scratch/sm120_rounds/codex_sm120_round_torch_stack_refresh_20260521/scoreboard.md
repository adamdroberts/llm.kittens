# SM120 Round Metrics - codex_sm120_round_torch_stack_refresh_20260521

- artifact dir: `scratch/sm120_rounds/codex_sm120_round_torch_stack_refresh_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_torch_stack_refresh_20260521`
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

- detailed matrix: `scratch/sm120_rounds/codex_sm120_round_torch_stack_refresh_20260521/backend-stacks.json`

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

- manifest: `scratch/sm120_rounds/codex_sm120_round_torch_stack_refresh_20260521/round-manifest.json`
- device arch: `SM120`
- build jobs: `4`
- changed paths: `482`

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
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 2223.190 | cuDNN | 2417.149 | python separated-Q/K/V | layout_rewrite_only | Use Torch SDPA for already-separated Q/K/V experiments; packed trainer selection must compare TorchPacked. |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 567.918 | cuDNN | 683.973 | python separated-Q/K/V | layout_rewrite_only | Use Torch SDPA for already-separated Q/K/V experiments; packed trainer selection must compare TorchPacked. |
| layernorm | backward | `N=65536 C=3072` | CUDA | 1302.031 | Triton atomic FP32-grad | 1425.824 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | backward | `N=65536 C=768` | CUDA | 287.972 | Triton atomic FP32-grad | 364.032 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | backward_dinput | `N=65536 C=3072` | Triton dInput-only | 817.120 | Torch native | 845.856 | partial backward prototype | - | Triton dInput-only is useful for backward decomposition work but needs dweight/dbias before trainer promotion. |
| layernorm | backward_dinput | `N=65536 C=768` | Torch native | 212.192 | Triton dInput-only | 224.736 | partial backward prototype | partial_backward_only | Torch native dInput-only output-mask row is useful for backward decomposition work but needs dweight/dbias before trainer promotion. |
| layernorm | forward | `N=65536 C=3072` | Torch native | 563.008 | CUDA | 563.255 | reference only | non_trainer_shape | Native Torch is useful where saved mean/rstd are not needed; trainer LayerNorm still needs stats-compatible state. |
| layernorm | forward | `N=65536 C=768` | CUDA | 142.649 | Torch native | 152.320 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Triton | 1108.544 | CUDA | 1112.549 | operator prototype | non_trainer_shape | Use as a Triton comparison row until a trainer-callable integration beats the current provider. |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 285.402 | Torch native | 332.352 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 365.280 | cuBLASLt | 367.160 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1346.500 | cuBLAS | 1363.760 | operator prototype | rejected_same_session_refresh | Torch is the fastest observed operator row here; trainer use still needs an explicit libtorch/C++ integration gate. |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1375.440 | cuBLAS | 1392.830 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21346.290 | Torch | 21820.540 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1029.350 | Torch | 1038.480 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1771.140 | cuBLASLt fused | 1820.230 | C++ benchmark route | rejected_correctness | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 328.250 | Torch | 339.980 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1333.130 | cuBLAS | 1353.380 | operator prototype | rejected_same_session_refresh | Torch is the fastest observed operator row here; trainer use still needs an explicit libtorch/C++ integration gate. |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1338.060 | Torch | 1368.920 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21177.730 | cuBLAS | 21566.840 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 997.610 | Torch | 1008.190 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 334.400 | Torch | 346.450 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1313.960 | Torch | 1354.600 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1351.940 | Torch | 1472.170 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21153.990 | cuBLAS | 21262.230 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1000.580 | Torch | 1023.400 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 376.400 | cuBLASLt | 393.060 | C++ benchmark route | rejected_same_session_refresh | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1389.990 | TK | 1457.280 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22524.040 | cuBLASLt | 22533.930 | C++ benchmark route | rejected_trainer_smoke | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1044.880 | TK | 1068.680 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1511.860 | TK fused | 1567.340 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1893.056 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | 7317.120 | - | - | operator prototype | rejected_slower_than_trainer_baseline | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | adamw_update_bf16_state | `params=124475904 no-master` | Torch | 1215.322 | - | - | non-equivalent BF16-state reference | - | Torch fused AdamW is usable only for BF16 moment-state experiments; the trainer default uses FP32 moment buffers. |
| runtime | bias_add | `BT=65536 OC=3072` | Torch | 530.650 | Triton | 532.278 | operator prototype | library_integration_not_justified | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 88.514 | Triton | 132.779 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 187.910 | Torch | 969.171 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 247.688 | Torch | 1313.085 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 24.502 | Torch | 317.315 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | Torch | 131.765 | CUDA runtime | 133.843 | operator prototype | native_replacement_rejected | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch | 8667.008 | CUDA runtime | 8992.870 | operator prototype | native_replacement_rejected | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch | 59.968 | CUDA runtime | 63.850 | operator prototype | native_replacement_rejected | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | cuda_memset | `logits_elems=3296722944` | Torch | 3920.640 | CUDA runtime | 4312.973 | operator prototype | native_replacement_rejected | Use Torch for Python-side operator comparisons; trainer promotion needs a matching C++ call path and smoke gate. |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 86.424 | Torch | 199.592 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 9121.119 | Triton | 22694.304 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 4064.608 | Triton | 8311.104 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | 793.576 | CUDA | 795.119 | operator prototype | rejected_same_session_refresh | Use as a Triton comparison row until a trainer-callable integration beats the current provider. |
| runtime | gelu_forward | `BT=65536 C=3072` | Triton | 531.483 | CUDA | 535.954 | operator prototype | rejected_same_session_refresh | Use as a Triton comparison row until a trainer-callable integration beats the current provider. |
| runtime | global_norm_squared | `params=124475904` | CUDA | 186.109 | Torch | 2268.339 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |

## Resolved Selected Backend Decisions

| Suite | Kernel | Shape | Selected stack | Status | Decision |
|---|---|---|---|---|---|
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| layernorm | backward_dinput | `N=65536 C=768` | Torch native | partial_backward_only | Do not promote the dInput-only Torch row. The full Torch native backward with dweight/dbias is slower than CUDA, and the full Triton atomic prototype is also slower plus has an FP32 gradient-buffer contract mismatch. |
| layernorm | forward | `N=65536 C=3072` | Torch native | non_trainer_shape | Not a GPT-2 124M trainer LayerNorm width; keep as operator stress evidence. |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Triton | non_trainer_shape | Not a GPT-2 124M trainer LayerNorm width, and refreshed CUDA was faster. |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | rejected_x10_selector | Do not broaden the SM120 direct-cuBLAS dInput selector to attention projection; the x10 trainer stability round regressed, so the source default remains the huge-N LM-head-only direct-cuBLAS dInput route. |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | rejected_same_session_refresh | Rejected for libtorch/native replacement work; focused uncontended timing showed current C++ cuBLAS was faster than Torch for MLP-up dInput. |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | rejected_correctness | Keep the cuBLASLt fused dGELU trainer route; the TK fused dGELU row had a microbenchmark edge but failed the route-shaped correctness smoke. |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | rejected_same_session_refresh | Rejected for libtorch/native replacement work; focused uncontended timing showed current C++ cuBLAS was faster than Torch for MLP-up dWeight. |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | rejected_same_session_refresh | Keep attention-projection forward on the cuBLASLt trainer default; TK's full-round edge was not stable under focused uncontended timing. |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | rejected_trainer_smoke | Keep LM-head forward on cuBLASLt; the opt-in direct-cuBLAS forward selector passed focused gates but regressed in TinyStories trainer timing. |
| runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | rejected_slower_than_trainer_baseline | Rejected for trainer promotion; Torch fp32-state AdamW is materially slower than the CUDA trainer route. |
| runtime | bias_add | `BT=65536 OC=3072` | Torch | library_integration_not_justified | Keep as operator evidence; the refreshed Torch edge is sub-percent and not enough to justify a libtorch trainer route. |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | Torch | native_replacement_rejected | Keep as operator evidence; native CUDA replacement was slower than CUDA runtime. |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch | native_replacement_rejected | Keep as operator evidence; native CUDA replacement was slower than CUDA runtime. |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch | native_replacement_rejected | Keep as operator evidence; native CUDA replacement was slower than CUDA runtime. |
| runtime | cuda_memset | `logits_elems=3296722944` | Torch | native_replacement_rejected | Keep as operator evidence; native CUDA replacement was slower than CUDA runtime. |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | rejected_same_session_refresh | Rejected for trainer promotion; refreshed CUDA GELU backward was faster than Triton. |
| runtime | gelu_forward | `BT=65536 C=3072` | Triton | rejected_same_session_refresh | Rejected for trainer promotion; focused uncontended timing showed CUDA was faster than Triton for the GPT-2 MLP GELU-forward row. |

## Promotion Backlog

No active promotion candidates remain after applying `dev/sm120_promotion_decisions.json`.

## Resolved Promotion Decisions

| Class | Suite | Kernel | Shape | Winning stack | Status | Decision |
|---|---|---|---|---|---|---|
| native/codegen integration | runtime | gelu_forward | `BT=65536 C=3072` | Triton | rejected_same_session_refresh | Rejected for trainer promotion; focused uncontended timing showed CUDA was faster than Triton for the GPT-2 MLP GELU-forward row. |
| native/codegen integration | runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | rejected_same_session_refresh | Rejected for trainer promotion; refreshed CUDA GELU backward was faster than Triton. |
| library integration | runtime | cuda_memset | `logits_elems=3296722944` | Torch | native_replacement_rejected | Keep as operator evidence; native CUDA replacement was slower than CUDA runtime. |
| library integration | runtime | cuda_memset | `hidden_elems=50331648` | Torch | native_replacement_rejected | Keep as operator evidence; native CUDA replacement was slower than CUDA runtime. |
| library integration | runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch | native_replacement_rejected | Keep as operator evidence; native CUDA replacement was slower than CUDA runtime. |
| library integration | runtime | cuda_copy_d2d | `hidden_elems=50331648` | Torch | native_replacement_rejected | Keep as operator evidence; native CUDA replacement was slower than CUDA runtime. |
| library integration | matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | rejected_same_session_refresh | Rejected for libtorch/native replacement work; focused uncontended timing showed current C++ cuBLAS was faster than Torch for MLP-up dWeight. |
| library integration | matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | rejected_same_session_refresh | Rejected for libtorch/native replacement work; focused uncontended timing showed current C++ cuBLAS was faster than Torch for MLP-up dInput. |
| library integration | runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | rejected_slower_than_trainer_baseline | Rejected for trainer promotion; Torch fp32-state AdamW is materially slower than the CUDA trainer route. |
| library integration | runtime | bias_add | `BT=65536 OC=3072` | Torch | library_integration_not_justified | Keep as operator evidence; the refreshed Torch edge is sub-percent and not enough to justify a libtorch trainer route. |
| layout rewrite | attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| layout rewrite | attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | layout_rewrite_only | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK. |
| reference/state gap | layernorm | backward_dinput | `N=65536 C=768` | Torch native | partial_backward_only | Do not promote the dInput-only Torch row. The full Torch native backward with dweight/dbias is slower than CUDA, and the full Triton atomic prototype is also slower plus has an FP32 gradient-buffer contract mismatch. |
| non-trainer shape | layernorm | backward_dinput | `N=65536 C=3072` | Triton dInput-only | non_trainer_shape | Not an active trainer-promotion target for GPT-2 124M. |
| non-trainer shape | layernorm | fused_residual_forward | `N=65536 C=3072` | Triton | non_trainer_shape | Not a GPT-2 124M trainer LayerNorm width, and refreshed CUDA was faster. |
| non-trainer shape | layernorm | forward | `N=65536 C=3072` | Torch native | non_trainer_shape | Not a GPT-2 124M trainer LayerNorm width; keep as operator stress evidence. |
| contract mismatch | runtime | adamw_update_bf16_state | `params=124475904 no-master` | Torch | contract_mismatch | Not active until the candidate matches the trainer state contract. |

## Benchmark Candidates

| Suite | Kernel | Shape | Stack | Time (us) |
|---|---|---|---|---:|
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1068.680 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1044.880 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1451.120 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1088.510 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1029.350 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1044.070 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1484.280 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1112.620 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 997.610 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1511.060 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1116.570 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1000.580 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 376.400 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 393.060 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 486.060 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 381.240 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 367.160 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 365.280 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 544.620 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 380.920 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 328.250 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 573.890 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 381.670 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 334.400 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK fused | 1567.340 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK explicit | 1956.010 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1511.860 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS explicit | 2521.480 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1486.860 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1386.390 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1363.760 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1746.910 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1523.080 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1353.380 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1741.710 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1554.310 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1313.960 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1457.280 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1389.990 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1584.200 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1566.320 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1375.440 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1392.830 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1771.140 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt fused | 1820.230 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt explicit | 2173.550 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS explicit | 2191.930 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1745.890 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1515.530 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1338.060 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1757.290 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1561.390 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1351.940 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 28150.220 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 22533.930 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22524.040 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 24395.390 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21823.830 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21346.290 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26258.100 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21177.730 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21566.840 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26311.210 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21153.990 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21262.230 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1452.330 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1038.480 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1008.190 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Torch | 1023.400 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 522.030 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 372.380 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 339.980 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Torch | 346.450 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 2459.780 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1346.500 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1333.130 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Torch | 1354.600 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1590.640 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1398.870 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 29017.120 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1368.920 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Torch | 1472.170 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 22579.460 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21820.540 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21888.160 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Torch | 21417.120 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2034.800 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2353.110 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2187.090 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | Triton | 2139.880 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 663.140 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 684.100 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 562.160 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | Triton | 562.570 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 2646.040 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 3098.320 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 2318.680 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | Triton | 2364.970 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 3006.090 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2827.800 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 3371.890 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2305.330 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | Triton | 2298.360 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 45438.740 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 50497.390 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 71571.710 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | Triton | 71592.450 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 792.863 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 2753.120 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 567.918 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | Torch | 2223.190 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchPacked | 1100.077 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TorchPacked | 4250.470 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNN | 683.973 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNN | 2417.149 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNNPacked | 821.589 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | cuDNNPacked | 3564.230 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | Triton | 2117.378 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TritonPacked | 2209.110 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | Torch | 18319.103 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | Triton | 8311.104 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | Triton | 22694.304 |
| layernorm | forward | `N=65536 C=768` | CUDA | 142.649 |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 285.402 |
| layernorm | backward | `N=65536 C=768` | CUDA | 287.972 |
| layernorm | forward | `N=65536 C=3072` | CUDA | 563.255 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | CUDA | 1112.549 |
| layernorm | backward | `N=65536 C=3072` | CUDA | 1302.031 |
| layernorm | forward | `N=65536 C=768` | Triton | 177.888 |
| layernorm | forward | `N=65536 C=768` | Torch native | 152.320 |
| layernorm | forward | `N=65536 C=768` | Torch stats | 2221.440 |
| layernorm | backward_dinput | `N=65536 C=768` | Triton dInput-only | 224.736 |
| layernorm | backward_dinput | `N=65536 C=768` | Torch native | 212.192 |
| layernorm | backward | `N=65536 C=768` | Torch native | 414.048 |
| layernorm | backward | `N=65536 C=768` | Triton atomic FP32-grad | 364.032 |
| layernorm | fused_residual_forward | `N=65536 C=768` | Triton | 334.336 |
| layernorm | fused_residual_forward | `N=65536 C=768` | Torch native | 332.352 |
| layernorm | fused_residual_forward | `N=65536 C=768` | Torch stats | 3241.952 |
| layernorm | forward | `N=65536 C=3072` | Triton | 590.368 |
| layernorm | forward | `N=65536 C=3072` | Torch native | 563.008 |
| layernorm | forward | `N=65536 C=3072` | Torch stats | 8957.344 |
| layernorm | backward_dinput | `N=65536 C=3072` | Triton dInput-only | 817.120 |
| layernorm | backward_dinput | `N=65536 C=3072` | Torch native | 845.856 |
| layernorm | backward | `N=65536 C=3072` | Torch native | 1451.424 |
| layernorm | backward | `N=65536 C=3072` | Triton atomic FP32-grad | 1425.824 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Triton | 1108.544 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Torch native | 1365.536 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | Torch stats | 13293.056 |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 88.514 |
| runtime | bias_add | `BT=65536 OC=3072` | CUDA | 537.093 |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 535.954 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | CUDA | 795.119 |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 24.502 |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 187.910 |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 247.688 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 4064.608 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 9121.119 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | 4312.973 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA kernel | 4434.816 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA runtime | 8992.870 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA kernel | 9604.640 |
| runtime | global_norm_squared | `params=124475904` | CUDA | 186.109 |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1893.056 |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 86.424 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA runtime | 63.850 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA kernel | 66.008 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 133.843 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA kernel | 139.068 |
| runtime | bias_add | `BT=65536 OC=768` | Triton | 132.779 |
| runtime | bias_add | `BT=65536 OC=3072` | Triton | 532.278 |
| runtime | gelu_forward | `BT=65536 C=3072` | Triton | 531.483 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Triton | 793.576 |
| runtime | bias_add | `BT=65536 OC=768` | Torch | 135.752 |
| runtime | bias_add | `BT=65536 OC=3072` | Torch | 530.650 |
| runtime | gelu_forward | `BT=65536 C=3072` | Torch | 552.707 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | Torch | 26828.299 |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | Torch | 317.315 |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | Torch | 969.171 |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | Torch | 1313.085 |
| runtime | global_norm_squared | `params=124475904` | Torch | 2268.339 |
| runtime | adamw_update_bf16_state | `params=124475904 no-master` | Torch | 1215.322 |
| runtime | adamw_update | `params=124475904 no-master fp32-state` | Torch | 7317.120 |
| runtime | encoder_forward | `B=64 T=1024 C=768` | Torch | 199.592 |
| runtime | cuda_memset | `hidden_elems=50331648` | Torch | 59.968 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | Torch | 131.765 |
| runtime | cuda_memset | `logits_elems=3296722944` | Torch | 3920.640 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | Torch | 8667.008 |

## Unavailable Backend Rows

| Suite | Kernel | Shape | Stack | Reason |
|---|---|---|---|---|
| matmul | cutedsl_gemm | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | CuTeDSL | local CuTeDSL BF16 grouped-GEMM path rejects sm_120a |
| matmul | cutedsl_gemm | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | CuTeDSL | local CuTeDSL BF16 grouped-GEMM path rejects sm_120a |
| matmul | cutedsl_gemm | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | CuTeDSL | local CuTeDSL BF16 grouped-GEMM path rejects sm_120a |
| matmul | cutedsl_gemm | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | CuTeDSL | local CuTeDSL BF16 grouped-GEMM path rejects sm_120a |
| matmul | cutedsl_gemm | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | CuTeDSL | local CuTeDSL BF16 grouped-GEMM path rejects sm_120a |

