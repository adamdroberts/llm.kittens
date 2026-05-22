# SM120 Round Metrics - codex_sm120_combo_libtorch_grad_zero_all_native_winners_x10_20260522

- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_all_native_winners_x10_20260522`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_all_native_winners_x10_20260522`
- git commit: `0f21747`

## Round Manifest

- manifest: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_all_native_winners_x10_20260522/round-manifest.json`
- device arch: `SM120`
- build jobs: `4`
- changed paths: `685`

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

## Selected Backend Rows

| Suite | Kernel | Shape | Selected stack | Time (us) | Next stack | Next time (us) | Use scope | Decision status | Decision note |
|---|---|---|---|---:|---|---:|---|---|---|
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 2744.225 | - | - | current packed trainer route | - | Current C++ attention route; keep unless a packed-QKV candidate beats it and passes the TinyStories smoke gate. |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 785.530 | - | - | current packed trainer route | - | Current C++ attention route; keep unless a packed-QKV candidate beats it and passes the TinyStories smoke gate. |
| layernorm | backward | `N=65536 C=3072` | CUDA | 1105.586 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | backward | `N=65536 C=768` | CUDA | 269.623 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | forward | `N=65536 C=3072` | CUDA | 543.703 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | forward | `N=65536 C=768` | CUDA | 137.429 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | fused_residual_forward | `N=65536 C=3072` | CUDA | 1084.151 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 275.929 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 365.560 | cuBLASLt | 367.300 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1333.490 | cuBLASLt | 1350.420 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1390.520 | cuBLASLt | 1403.840 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21390.690 | cuBLASLt | 21763.400 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1017.150 | cuBLAS | 1036.610 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1741.990 | cuBLASLt fused | 1853.930 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 328.680 | cuBLASLt | 373.690 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1332.760 | cuBLASLt | 1481.890 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1309.530 | cuBLASLt | 1492.870 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20959.190 | cuBLAS | 21317.580 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 992.950 | cuBLASLt | 1122.080 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 331.570 | cuBLASLt | 380.170 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1333.040 | cuBLASLt | 1525.040 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1336.390 | cuBLASLt | 1471.900 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20993.410 | cuBLAS | 21301.890 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 997.900 | cuBLASLt | 1115.890 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 372.600 | TK | 377.240 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1374.230 | TK | 1418.490 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22324.910 | cuBLASLt | 22483.770 | C++ benchmark route | rejected_trainer_smoke | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1039.280 | TK | 1073.430 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1518.260 | TK fused | 1536.900 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1831.888 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_add | `BT=65536 OC=3072` | CUDA | 534.533 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 80.396 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 186.678 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 245.669 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 22.906 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 131.655 | CUDA kernel | 133.562 | CUDA benchmark route | profiler_only_runtime_row | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA runtime | 8821.900 | CUDA kernel | 8952.025 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_memset | `grad_elems=124475904` | CUDA runtime | 148.032 | CUDA kernel | 150.573 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA runtime | 59.782 | CUDA kernel | 60.301 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | 3954.695 | CUDA kernel | 4027.514 | CUDA benchmark route | profiler_only_runtime_row | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 83.885 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 8902.765 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 3944.019 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | CUDA | 791.812 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 527.754 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | global_norm_squared | `params=124475904` | CUDA | 184.429 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |

## Resolved Selected Backend Decisions

| Suite | Kernel | Shape | Selected stack | Status | Decision |
|---|---|---|---|---|---|
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | rejected_x10_selector | Do not broaden the SM120 direct-cuBLAS dInput selector to attention projection. Both the bundled attproj/MLP-up selector and the later attproj-only selector regressed in x10 TinyStories stability gates, so the source default remains the huge-N LM-head-only direct-cuBLAS dInput route. |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | rejected_x10_selector | Do not promote the SM120 direct-cuBLAS dInput selector to the GPT-2 MLP-up row. The microbench row can favor cuBLAS, but both the broader attproj+MLP-up selector and the later FC-only selector regressed in x10 TinyStories stability gates, while the stream-sync default keeps cuBLASLt for this row and improves the trainer smoke. |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | rejected_x10_selector | Keep the cuBLASLt fused dGELU trainer route as the default. The opt-in TK exact-dGELU selector now passes correctness and has a focused row win, but its x10 TinyStories stability round regressed versus the current stable default. |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | rejected_trainer_smoke | Keep LM-head forward on cuBLASLt; the opt-in direct-cuBLAS forward selector passed focused gates but regressed in TinyStories trainer timing. |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | profiler_only_runtime_row | Keep as benchmark evidence only. The refreshed optional round measured CUDA runtime as fastest for this profiler-only copy shape, but it is not a current trainer call path to promote. |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | profiler_only_runtime_row | Keep as profiler/runtime evidence only. CUDA runtime remains the fastest current logits-sized memset row, but the current GPT-2 trainer has no logits-sized memset call-site to promote. |

## Attention Route Totals

| Shape | Stack | Scope | Trainer-layout | Forward (us) | Backward (us) | Total (us) | Complete | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | packed trainer-layout route | True | 785.530 | 2744.225 | 3529.755 | True |  |

## Benchmark Candidates

| Suite | Kernel | Shape | Stack | Time (us) |
|---|---|---|---|---:|
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1073.430 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1039.280 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1406.860 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1116.270 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1017.150 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1036.610 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1955.540 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1122.080 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 992.950 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1968.510 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1115.890 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 997.900 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 377.240 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 372.600 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 482.580 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 381.140 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 367.300 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 365.560 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 1906.730 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 373.690 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 328.680 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 1928.930 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 380.170 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 331.570 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK fused | 1536.900 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK explicit | 1971.320 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1518.260 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS explicit | 2462.470 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1494.910 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1350.420 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1333.490 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1985.460 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1481.890 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1332.760 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1991.080 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1525.040 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1333.040 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1418.490 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1374.230 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1598.280 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1479.090 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1403.840 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1390.520 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1741.990 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt fused | 1853.930 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt explicit | 2178.150 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS explicit | 2176.270 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1982.460 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1492.870 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1309.530 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1943.740 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1471.900 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1336.390 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 27665.170 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 22483.770 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22324.910 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 24110.850 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21763.400 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21390.690 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26435.470 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20959.190 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21317.580 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26422.340 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20993.410 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21301.890 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 785.530 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 2744.225 |
| layernorm | forward | `N=65536 C=768` | CUDA | 137.429 |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 275.929 |
| layernorm | backward | `N=65536 C=768` | CUDA | 269.623 |
| layernorm | forward | `N=65536 C=3072` | CUDA | 543.703 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | CUDA | 1084.151 |
| layernorm | backward | `N=65536 C=3072` | CUDA | 1105.586 |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 80.396 |
| runtime | bias_add | `BT=65536 OC=3072` | CUDA | 534.533 |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 527.754 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | CUDA | 791.812 |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 22.906 |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 186.678 |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 245.669 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 3944.019 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 8902.765 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | 3954.695 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA kernel | 4027.514 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA runtime | 8821.900 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA kernel | 8952.025 |
| runtime | cuda_memset | `grad_elems=124475904` | CUDA runtime | 148.032 |
| runtime | cuda_memset | `grad_elems=124475904` | CUDA kernel | 150.573 |
| runtime | global_norm_squared | `params=124475904` | CUDA | 184.429 |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1831.888 |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 83.885 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA runtime | 59.782 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA kernel | 60.301 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 131.655 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA kernel | 133.562 |

## Training Smoke

- use_master_weights: `disabled`
- gelu_fusion: `1`
- total average iteration time: `2519.610 ms`
- final val loss: `9.483727`
- final step: `10/10`, loss `9.588621`, `2528.73 ms`, `208026 tok/s`

| Step | Loss | Norm | LR | Time (ms) | Tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.032358 | 22.1414 | 8.57e-07 | 2504.49 | 209339 |
| 2 | 10.958514 | 22.0969 | 1.71e-06 | 2508.37 | 209015 |
| 3 | 10.811329 | 21.1250 | 2.57e-06 | 2520.56 | 208497 |
| 4 | 10.610136 | 18.7016 | 3.43e-06 | 2513.75 | 208522 |
| 5 | 10.392597 | 15.0184 | 4.29e-06 | 2516.09 | 208482 |
| 6 | 10.186258 | 12.0843 | 5.14e-06 | 2518.59 | 208412 |
| 7 | 10.010626 | 10.2002 | 6e-06 | 2520.37 | 208338 |
| 8 | 9.855871 | 8.7905 | 6.86e-06 | 2523.73 | 208240 |
| 9 | 9.719436 | 7.4665 | 7.71e-06 | 2526.29 | 208135 |
| 10 | 9.588621 | 6.3098 | 8.57e-06 | 2528.73 | 208026 |

