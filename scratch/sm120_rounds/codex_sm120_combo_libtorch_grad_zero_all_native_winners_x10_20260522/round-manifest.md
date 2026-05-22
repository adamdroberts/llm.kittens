# SM120 Round Manifest

- run label: `codex_sm120_combo_libtorch_grad_zero_all_native_winners_x10_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_all_native_winners_x10_20260522`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_all_native_winners_x10_20260522`
- device arch: `SM120`
- max steps: `10`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `1`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `685`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2190792` | `91e4a7822908ad83cfbc78f7bd919aa3c410ac784947ccb670762e938cb3b765` |
| `test_attention` | `True` | `1800528` | `9465761a0bc5b680c492114be0bbc36bcae958800974dd0b667e74e820aafddf` |
| `test_layernorm` | `True` | `1278296` | `f0e57e3a15c887d0438821560cddbf45111331bf796c5e01fbec7790cad3965f` |
| `test_bias` | `True` | `2052256` | `06112652d93c213fac7efe02c7f8db997a7339e7c01adac7a4989d9a0cc4c4a8` |
| `test_gelu` | `True` | `1179912` | `e2ea765394536bcfe5f14d5dc4bf0f7320444edd7b38baac51af34c369bb0ff0` |
| `test_fused_classifier` | `True` | `1208704` | `eada50780b79bb0e3af0d3694ec7a5bfc462a6d3e100583197fd11f7ba31676d` |
| `test_encoder` | `True` | `1210168` | `590448365534558f611ed5378c65eb89ac061a5b05272e1306d35167ad9a5a74` |
| `test_adamw` | `True` | `1183768` | `85739cb4e99381789c0a776150e3e0a1e1310b120625be9e9bc798a806c7abf2` |
| `test_global_norm` | `True` | `1179464` | `2111d52adb6f864c95faf698c27cbf73a5f2bd4b87e2f91287ae0c766547034b` |
| `bench_sm120_matmul` | `True` | `2369344` | `83e4ce9ea39ef4e4066cd26f81012955ea10da8f989e55d52b8903e7da8478f4` |
| `bench_sm120_attention` | `True` | `1768800` | `bec35357088a635bcd514aad6a8f69279cb741ec75eb7fad9b554197eb2400fe` |
| `bench_sm120_layernorm` | `True` | `1274232` | `3b0b78814fcee731938c960dec8d1b742f0f806ce89b8413e10cc2885ef93fbd` |
| `bench_sm120_runtime` | `True` | `2234256` | `120a1e3e31c6978270e4473a5e03ce4de001f23ad0a0efbaa79710c15362134c` |
| `train_gpt2cu` | `True` | `3107416` | `d4d3422e2dc207b59662e6cb81506f88738e9cfd57c2a27b56c89722e86c3162` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
