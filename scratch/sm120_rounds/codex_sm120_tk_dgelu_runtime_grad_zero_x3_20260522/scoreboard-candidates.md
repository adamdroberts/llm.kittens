# SM120 Round Metrics - codex_sm120_tk_dgelu_runtime_grad_zero_x3_20260522

- artifact dir: `scratch/sm120_rounds/codex_sm120_tk_dgelu_runtime_grad_zero_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_tk_dgelu_runtime_grad_zero_x3_20260522`
- git commit: `0f21747`

## Round Manifest

- manifest: `scratch/sm120_rounds/codex_sm120_tk_dgelu_runtime_grad_zero_x3_20260522/round-manifest.json`
- device arch: `SM120`
- build jobs: `4`
- changed paths: `667`

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
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 2741.513 | - | - | current packed trainer route | - | Current C++ attention route; keep unless a packed-QKV candidate beats it and passes the TinyStories smoke gate. |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 783.432 | - | - | current packed trainer route | - | Current C++ attention route; keep unless a packed-QKV candidate beats it and passes the TinyStories smoke gate. |
| layernorm | backward | `N=65536 C=3072` | CUDA | 1104.796 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | backward | `N=65536 C=768` | CUDA | 271.323 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | forward | `N=65536 C=3072` | CUDA | 544.248 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | forward | `N=65536 C=768` | CUDA | 137.638 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | fused_residual_forward | `N=65536 C=3072` | CUDA | 1085.370 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 275.542 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 365.560 | cuBLASLt | 370.980 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1348.560 | cuBLAS | 1350.470 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1376.740 | cuBLASLt | 1394.520 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21383.540 | cuBLASLt | 21902.080 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1012.180 | cuBLAS | 1012.310 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1781.130 | cuBLASLt fused | 1812.510 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 327.100 | cuBLASLt | 380.770 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1308.240 | cuBLASLt | 1508.620 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1327.560 | cuBLASLt | 1509.960 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20986.550 | cuBLAS | 21203.670 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 992.840 | cuBLASLt | 1106.230 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 330.660 | cuBLASLt | 378.210 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1341.260 | cuBLASLt | 1530.500 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1334.670 | cuBLASLt | 1489.690 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21055.520 | cuBLAS | 21279.700 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 999.940 | cuBLASLt | 1120.220 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 376.200 | cuBLASLt | 405.210 | C++ benchmark route | rejected_same_session_refresh | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1366.390 | TK | 1418.640 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 22331.890 | cuBLAS | 22475.670 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1038.250 | TK | 1068.450 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1516.250 | TK fused | 1536.580 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1811.402 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_add | `BT=65536 OC=3072` | CUDA | 547.689 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 88.911 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 186.472 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 245.190 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 24.019 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA kernel | 135.331 | CUDA runtime | 136.299 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA runtime | 8765.056 | CUDA kernel | 8984.229 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_memset | `grad_elems=124475904` | CUDA runtime | 149.331 | CUDA kernel | 149.835 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA kernel | 60.190 | CUDA runtime | 62.604 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | 3997.773 | CUDA kernel | 4034.112 | CUDA benchmark route | profiler_only_runtime_row | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 79.697 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 8911.111 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 3996.902 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | CUDA | 790.487 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 527.433 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | global_norm_squared | `params=124475904` | CUDA | 184.699 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |

## Resolved Selected Backend Decisions

| Suite | Kernel | Shape | Selected stack | Status | Decision |
|---|---|---|---|---|---|
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | rejected_x10_selector | Do not broaden the SM120 direct-cuBLAS dInput selector to attention projection. Both the bundled attproj/MLP-up selector and the later attproj-only selector regressed in x10 TinyStories stability gates, so the source default remains the huge-N LM-head-only direct-cuBLAS dInput route. |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | rejected_x10_selector | Keep the cuBLASLt fused dGELU trainer route as the default. The opt-in TK exact-dGELU selector now passes correctness and has a focused row win, but its x10 TinyStories stability round regressed versus the current stable default. |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | rejected_same_session_refresh | Keep attention-projection forward on the cuBLASLt trainer default; TK's full-round edge was not stable under focused uncontended timing. |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | profiler_only_runtime_row | Keep as profiler/runtime evidence only. CUDA runtime remains the fastest current logits-sized memset row, but the current GPT-2 trainer has no logits-sized memset call-site to promote. |

## Attention Route Totals

| Shape | Stack | Scope | Trainer-layout | Forward (us) | Backward (us) | Total (us) | Complete | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | packed trainer-layout route | True | 783.432 | 2741.513 | 3524.945 | True |  |

## Benchmark Candidates

| Suite | Kernel | Shape | Stack | Time (us) |
|---|---|---|---|---:|
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1068.450 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1038.250 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1404.920 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1090.900 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1012.180 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1012.310 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1940.190 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1106.230 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 992.840 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1945.240 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1120.220 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 999.940 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 376.200 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 405.210 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 483.460 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 381.510 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 370.980 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 365.560 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 1906.350 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 380.770 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 327.100 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 1907.350 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 378.210 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 330.660 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK fused | 1536.580 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK explicit | 1996.350 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1516.250 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS explicit | 2462.840 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1493.820 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1348.560 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1350.470 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1951.660 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1508.620 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1308.240 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1983.800 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1530.500 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1341.260 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1418.640 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1366.390 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1598.790 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1525.440 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1394.520 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1376.740 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1781.130 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt fused | 1812.510 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt explicit | 2192.330 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS explicit | 2165.570 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1963.630 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1509.960 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1327.560 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1948.170 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1489.690 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1334.670 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 27791.220 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 22331.890 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22475.670 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 23999.290 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21902.080 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21383.540 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26456.820 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20986.550 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21203.670 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26518.510 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21055.520 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21279.700 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 783.432 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 2741.513 |
| layernorm | forward | `N=65536 C=768` | CUDA | 137.638 |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 275.542 |
| layernorm | backward | `N=65536 C=768` | CUDA | 271.323 |
| layernorm | forward | `N=65536 C=3072` | CUDA | 544.248 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | CUDA | 1085.370 |
| layernorm | backward | `N=65536 C=3072` | CUDA | 1104.796 |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 88.911 |
| runtime | bias_add | `BT=65536 OC=3072` | CUDA | 547.689 |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 527.433 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | CUDA | 790.487 |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 24.019 |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 186.472 |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 245.190 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 3996.902 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 8911.111 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | 3997.773 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA kernel | 4034.112 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA runtime | 8765.056 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA kernel | 8984.229 |
| runtime | cuda_memset | `grad_elems=124475904` | CUDA runtime | 149.331 |
| runtime | cuda_memset | `grad_elems=124475904` | CUDA kernel | 149.835 |
| runtime | global_norm_squared | `params=124475904` | CUDA | 184.699 |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1811.402 |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 79.697 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA runtime | 62.604 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA kernel | 60.190 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 136.299 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA kernel | 135.331 |

## Training Smoke

- use_master_weights: `disabled`
- gelu_fusion: `1`
- total average iteration time: `2496.712 ms`
- final val loss: `10.609915`
- final step: `3/3`, loss `10.811318`, `2499.48 ms`, `209986 tok/s`

| Step | Loss | Norm | LR | Time (ms) | Tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.032358 | 22.1414 | 8.57e-07 | 2497.46 | 209929 |
| 2 | 10.958508 | 22.0967 | 1.71e-06 | 2493.94 | 210225 |
| 3 | 10.811318 | 21.1250 | 2.57e-06 | 2499.48 | 209986 |

