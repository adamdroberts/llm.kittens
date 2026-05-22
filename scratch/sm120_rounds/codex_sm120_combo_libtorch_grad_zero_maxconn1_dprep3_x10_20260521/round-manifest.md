# SM120 Round Manifest

- run label: `codex_sm120_combo_libtorch_grad_zero_maxconn1_dprep3_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_maxconn1_dprep3_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_maxconn1_dprep3_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `1`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `585`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `fed8e386cce42d66d7287b395d0d4e2c7e3af88fc9c1223d0256d1f1d943b805` |
| `test_attention` | `True` | `1800528` | `d092326b662427bc24b1bdc019ff5843ee936874fce395009d457617179f9d59` |
| `test_layernorm` | `True` | `1278296` | `85f1e018a792396a85cf33c1db09b72424da07480be013af37e63a0a4ad944df` |
| `test_bias` | `True` | `2089120` | `7663945adb3b895c0cb7f7ed75ab95faa69e14ba06747fc6075c2ec4778b2e84` |
| `test_gelu` | `True` | `1179912` | `492d32eec38229003379930b1996a71930e2ac27cbd1f754a917886e5656cbbc` |
| `test_fused_classifier` | `True` | `1208704` | `43535a0b53ce7539b2fece751f602fea82da5db22b09d523651c01a555730eb4` |
| `test_encoder` | `True` | `1210168` | `359dbda6d4d8f739e1191554e20190c1cd4cf3e93ba10c921657eea97b815a26` |
| `test_adamw` | `True` | `1183768` | `4bce88a0d7de36b9b7da64d92534218a5aa715b61dcc01ebf00adbd819444a56` |
| `test_global_norm` | `True` | `1179464` | `90752f62781f29573881b7096a921973f0f3927c89ce2f8c0ace769d561734db` |
| `bench_sm120_matmul` | `True` | `2410304` | `796999acccb97aaabaae77b6d45c23dc522346dc29d04932a85d99ac4f8953d6` |
| `bench_sm120_attention` | `True` | `1768800` | `6f712f20bf3e639f40e915b9e6fe1ef6f97eebcaa60b4f438734991bb3e47cd7` |
| `bench_sm120_layernorm` | `True` | `1274232` | `0ea975389633bdf4c9daaa28bbb524813b09bfe19f2845d345fc0a345bb6c20f` |
| `bench_sm120_runtime` | `True` | `2271168` | `93d89444676698fcfe2507e398770f7108a1ec9cc60035605b84237035b5ee41` |
| `train_gpt2cu` | `True` | `3105552` | `8cdf5d20d1bb9f9938ef66455fdfe4faf8f4af4cdf450d4f48a7cff68028b3cc` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
