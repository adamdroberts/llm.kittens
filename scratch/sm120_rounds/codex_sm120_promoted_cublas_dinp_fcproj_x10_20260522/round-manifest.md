# SM120 Round Manifest

- run label: `codex_sm120_promoted_cublas_dinp_fcproj_x10_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_fcproj_x10_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_cublas_dinp_fcproj_x10_20260522`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `627`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `f1b849ae73a03e6a10488a79d0df178406064db637e80ed325a54c19df443d83` |
| `test_attention` | `True` | `1800528` | `bd7b87cc89a1d697f68ce289e011df59edb718535df446213b64f7a5dce10acd` |
| `test_layernorm` | `True` | `1278296` | `385d315666f41b4f7d65354b89774340e74f1300a0c440fd00f2b58a101cb708` |
| `test_bias` | `True` | `2089120` | `3e3801f157e3e9d948ef7dbc1fafd3fbac8650f0cc68e0fa739ffd4eae678343` |
| `test_gelu` | `True` | `1179912` | `1937c17a0970fd86273b713dc0628b66994bc218dd1fc037167362c313105bbc` |
| `test_fused_classifier` | `True` | `1208704` | `f9fb1e7b6d8a018aed64ec554b8071ea895dccb908365bc7e715e76270178992` |
| `test_encoder` | `True` | `1210168` | `73e3636f2086effecdb9668ac393d3fb7084b36b0e49b771a50deae909a35d85` |
| `test_adamw` | `True` | `1183768` | `10b27c23fccb0f5dee7a5cf0ce64d5191a42006b1d1c1312e2e527ccea6b658f` |
| `test_global_norm` | `True` | `1179464` | `c3cd9497e57c0245ad8a2e0b975cb618c606e72513487a4b39fdba1d2ecd5261` |
| `bench_sm120_matmul` | `True` | `2410304` | `2452841224d4d75da409e36acd49a3f51729cdd13062176e1d5233317bcca308` |
| `bench_sm120_attention` | `True` | `1768800` | `d845a7e7c73575e5ad7128365372f0a6a9a8147a31d9fba61739602bd7cb0376` |
| `bench_sm120_layernorm` | `True` | `1274232` | `fb246d1dee0e3c04527e7a72db278e39c53530a51c1dd4bd3568e22e67423231` |
| `bench_sm120_runtime` | `True` | `2271168` | `0cdc2b902512ad1a21859e71d3987cd9182df15a8a23f1fe4a2cb16325757c9f` |
| `train_gpt2cu` | `True` | `3105552` | `8c11daffd3d3d0f97b3e72838964fe4fead484ad5e527dff68aa70bc46812662` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
