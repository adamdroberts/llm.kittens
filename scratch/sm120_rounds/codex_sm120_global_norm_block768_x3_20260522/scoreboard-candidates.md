# SM120 Round Metrics - codex_sm120_global_norm_block768_x3_20260522

- artifact dir: `scratch/sm120_rounds/codex_sm120_global_norm_block768_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_global_norm_block768_x3_20260522`
- git commit: `0f21747`

## Round Manifest

- manifest: `scratch/sm120_rounds/codex_sm120_global_norm_block768_x3_20260522/round-manifest.json`
- device arch: `SM120`
- build jobs: `4`
- changed paths: `679`

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
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 2738.536 | - | - | current packed trainer route | - | Current C++ attention route; keep unless a packed-QKV candidate beats it and passes the TinyStories smoke gate. |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 785.353 | - | - | current packed trainer route | - | Current C++ attention route; keep unless a packed-QKV candidate beats it and passes the TinyStories smoke gate. |
| layernorm | backward | `N=65536 C=3072` | CUDA | 1103.108 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | backward | `N=65536 C=768` | CUDA | 272.723 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | forward | `N=65536 C=3072` | CUDA | 543.940 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | forward | `N=65536 C=768` | CUDA | 138.748 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | fused_residual_forward | `N=65536 C=3072` | CUDA | 1082.290 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 274.923 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 365.330 | cuBLASLt | 367.640 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1330.550 | cuBLASLt | 1346.500 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1368.720 | cuBLASLt | 1408.750 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21333.470 | cuBLASLt | 21850.440 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1012.160 | cuBLASLt | 1020.520 | C++ benchmark route | noise_floor_microbench_flip | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1786.310 | cuBLASLt fused | 1839.000 | C++ benchmark route | rejected_x10_selector | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 327.700 | cuBLASLt | 373.990 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1314.760 | cuBLASLt | 1487.600 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1351.240 | cuBLASLt | 1474.610 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20959.570 | cuBLAS | 21088.530 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1014.910 | cuBLASLt | 1113.340 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 331.820 | cuBLASLt | 378.800 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1313.660 | cuBLASLt | 1522.430 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1319.710 | cuBLASLt | 1474.470 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21067.710 | cuBLAS | 21282.200 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 997.020 | cuBLASLt | 1138.850 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 371.970 | TK | 376.260 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1345.020 | TK | 1441.980 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22402.950 | cuBLASLt | 22431.560 | C++ benchmark route | rejected_trainer_smoke | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1043.260 | TK | 1073.470 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1524.830 | TK fused | 1538.280 | C++ benchmark route | - | Trainer-callable route, but default promotion still requires route-specific correctness and TinyStories smoke evidence. |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1787.427 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_add | `BT=65536 OC=3072` | CUDA | 537.640 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 80.251 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 186.066 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 244.970 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 23.083 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 131.367 | CUDA kernel | 139.768 | CUDA benchmark route | profiler_only_runtime_row | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA runtime | 8795.629 | CUDA kernel | 8991.443 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_memset | `grad_elems=124475904` | CUDA runtime | 149.496 | CUDA kernel | 150.070 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA runtime | 59.690 | CUDA kernel | 60.424 | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | 3917.229 | CUDA kernel | 3951.066 | CUDA benchmark route | profiler_only_runtime_row | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 76.243 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 8894.617 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 3999.699 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | CUDA | 781.052 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 528.076 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |
| runtime | global_norm_squared | `params=124475904` | CUDA | 185.144 | - | - | CUDA benchmark route | - | Plain CUDA/CUDA-runtime route; eligible for trainer selection subject to correctness and TinyStories smoke evidence. |

## Resolved Selected Backend Decisions

| Suite | Kernel | Shape | Selected stack | Status | Decision |
|---|---|---|---|---|---|
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | rejected_x10_selector | Do not broaden the SM120 direct-cuBLAS dInput selector to attention projection. Both the bundled attproj/MLP-up selector and the later attproj-only selector regressed in x10 TinyStories stability gates, so the source default remains the huge-N LM-head-only direct-cuBLAS dInput route. |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | rejected_x10_selector | Do not promote the SM120 direct-cuBLAS dInput selector to the GPT-2 MLP-up row. The microbench row can favor cuBLAS, but both the broader attproj+MLP-up selector and the later FC-only selector regressed in x10 TinyStories stability gates, while the stream-sync default keeps cuBLASLt for this row and improves the trainer smoke. |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | noise_floor_microbench_flip | Do not promote the qkv dInput cuBLAS microbench flip as a trainer default without a trainer smoke. The refreshed benchmark-only round picked cuBLAS by about 0.2%, while the stable x10 selection artifact has cuBLASLt ahead for the same row. |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | rejected_x10_selector | Keep the cuBLASLt fused dGELU trainer route as the default. The opt-in TK exact-dGELU selector now passes correctness and has a focused row win, but its x10 TinyStories stability round regressed versus the current stable default. |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | rejected_trainer_smoke | Keep LM-head forward on cuBLASLt; the opt-in direct-cuBLAS forward selector passed focused gates but regressed in TinyStories trainer timing. |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | profiler_only_runtime_row | Keep as benchmark evidence only. The refreshed optional round measured CUDA runtime as fastest for this profiler-only copy shape, but it is not a current trainer call path to promote. |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | profiler_only_runtime_row | Keep as profiler/runtime evidence only. CUDA runtime remains the fastest current logits-sized memset row, but the current GPT-2 trainer has no logits-sized memset call-site to promote. |

## Attention Route Totals

| Shape | Stack | Scope | Trainer-layout | Forward (us) | Backward (us) | Total (us) | Complete | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | packed trainer-layout route | True | 785.353 | 2738.536 | 3523.889 | True |  |

## Benchmark Candidates

| Suite | Kernel | Shape | Stack | Time (us) |
|---|---|---|---|---:|
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1073.470 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1043.260 |
| matmul | fwd | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1413.410 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1103.970 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1020.520 |
| matmul | dInp | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1012.160 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1949.110 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1113.340 |
| matmul | dW | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 1014.910 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | TK | 1971.590 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLASLt | 1138.850 |
| matmul | dW+accum | `qkv M=65536 N=2304 K=768 bias=1 gelu=0` | cuBLAS | 997.020 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 376.260 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 371.970 |
| matmul | fwd | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 485.100 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 381.120 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 367.640 |
| matmul | dInp | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 365.330 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 1929.020 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 373.990 |
| matmul | dW | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 327.700 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | TK | 1969.790 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLASLt | 378.800 |
| matmul | dW+accum | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS | 331.820 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK fused | 1538.280 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK explicit | 1992.000 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1524.830 |
| matmul | fwd+gelu | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS explicit | 2455.000 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1496.920 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1346.500 |
| matmul | dInp | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1330.550 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1957.270 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1487.600 |
| matmul | dW | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1314.760 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | TK | 1989.720 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLASLt | 1522.430 |
| matmul | dW+accum | `fc M=65536 N=3072 K=768 bias=1 gelu=1` | cuBLAS | 1313.660 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1441.980 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1345.020 |
| matmul | fwd | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1594.830 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1497.560 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1408.750 |
| matmul | dInp | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1368.720 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1786.310 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt fused | 1839.000 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt explicit | 2177.310 |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS explicit | 2150.990 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1960.980 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1474.610 |
| matmul | dW | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1351.240 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK | 1964.610 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt | 1474.470 |
| matmul | dW+accum | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLAS | 1319.710 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 27713.180 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 22431.560 |
| matmul | fwd | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 22402.950 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 23950.390 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21850.440 |
| matmul | dInp | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21333.470 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26515.410 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 20959.570 |
| matmul | dW | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21088.530 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | TK | 26417.400 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt | 21067.710 |
| matmul | dW+accum | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS | 21282.200 |
| attention | forward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 785.353 |
| attention | backward | `B=64 T=1024 C=768 NH=12 HS=64` | TK packed-QKV | 2738.536 |
| layernorm | forward | `N=65536 C=768` | CUDA | 138.748 |
| layernorm | fused_residual_forward | `N=65536 C=768` | CUDA | 274.923 |
| layernorm | backward | `N=65536 C=768` | CUDA | 272.723 |
| layernorm | forward | `N=65536 C=3072` | CUDA | 543.940 |
| layernorm | fused_residual_forward | `N=65536 C=3072` | CUDA | 1082.290 |
| layernorm | backward | `N=65536 C=3072` | CUDA | 1103.108 |
| runtime | bias_add | `BT=65536 OC=768` | CUDA | 80.251 |
| runtime | bias_add | `BT=65536 OC=3072` | CUDA | 537.640 |
| runtime | gelu_forward | `BT=65536 C=3072` | CUDA | 528.076 |
| runtime | gelu_backward_inplace | `BT=65536 C=3072` | CUDA | 781.052 |
| runtime | bias_grad_reduce | `BT=65536 OC=768` | CUDA | 23.083 |
| runtime | bias_grad_reduce | `BT=65536 OC=2304` | CUDA | 186.066 |
| runtime | bias_grad_reduce | `BT=65536 OC=3072` | CUDA | 244.970 |
| runtime | fused_classifier_loss | `B=64 T=1024 V=50257 P=50304` | CUDA | 3999.699 |
| runtime | fused_classifier | `B=64 T=1024 V=50257 P=50304` | CUDA | 8894.617 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA runtime | 3917.229 |
| runtime | cuda_memset | `logits_elems=3296722944` | CUDA kernel | 3951.066 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA runtime | 8795.629 |
| runtime | cuda_copy_d2d | `logits_elems=3296722944` | CUDA kernel | 8991.443 |
| runtime | cuda_memset | `grad_elems=124475904` | CUDA runtime | 149.496 |
| runtime | cuda_memset | `grad_elems=124475904` | CUDA kernel | 150.070 |
| runtime | global_norm_squared | `params=124475904` | CUDA | 185.144 |
| runtime | adamw_update | `params=124475904 no-master` | CUDA | 1787.427 |
| runtime | encoder_forward | `B=64 T=1024 C=768` | CUDA | 76.243 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA runtime | 59.690 |
| runtime | cuda_memset | `hidden_elems=50331648` | CUDA kernel | 60.424 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA runtime | 131.367 |
| runtime | cuda_copy_d2d | `hidden_elems=50331648` | CUDA kernel | 139.768 |

## Training Smoke

- use_master_weights: `disabled`
- gelu_fusion: `1`
- total average iteration time: `2495.645 ms`
- final val loss: `10.609931`
- final step: `3/3`, loss `10.811321`, `2499.07 ms`, `210074 tok/s`

| Step | Loss | Norm | LR | Time (ms) | Tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.032358 | 22.1414 | 8.57e-07 | 2499.17 | 209785 |
| 2 | 10.958507 | 22.0968 | 1.71e-06 | 2492.22 | 210370 |
| 3 | 10.811321 | 21.1250 | 2.57e-06 | 2499.07 | 210074 |

