# SM120 Round Manifest

- run label: `codex_sm120_promoted_maxconn8_x10_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_maxconn8_x10_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_maxconn8_x10_20260522`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `634`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `1d872f3bc530fea9fe0e35a8645360a6aed6260f20e01ffd41b5a5f2e05a4fb1` |
| `test_attention` | `True` | `1800528` | `2ffcb7b97def51761cc1d5eac4fbb629984b3097a4f1b3c9636690374314ad8c` |
| `test_layernorm` | `True` | `1278296` | `8903329bf2ef54fc607741eb1e0087bcac1854bfcae429d9468e382d808f4b70` |
| `test_bias` | `True` | `2089120` | `0f513c26ba8ee8f689061b2c9e670f0136c9f3406dfcf47a27b858bbfc4cd173` |
| `test_gelu` | `True` | `1179912` | `ff044ea9eb49ac662284665b6cfe599b4e61a411760076977314974d58b6d617` |
| `test_fused_classifier` | `True` | `1208704` | `b811da7a11e4c9bb061420e9ae792e4fa72345eb824489a653c26ed7dd96d59f` |
| `test_encoder` | `True` | `1210168` | `39dd868cc63b7bfccdb6568fed4859011015011a5b5f9679a7352359f1c0da14` |
| `test_adamw` | `True` | `1183768` | `77d71daf1680243603291c09cfbd436a56bb9107050f95aaae5fefd7e2144684` |
| `test_global_norm` | `True` | `1179464` | `0b8f2953530760950131b9c9715ec5cda1d27b21339f3f43c64b8b30831752e5` |
| `bench_sm120_matmul` | `True` | `2410304` | `faf7ceb25ea289d29a80ac33cecba7b68dbc7c18ef9a6b2f1bf9589c85d5ac06` |
| `bench_sm120_attention` | `True` | `1768800` | `eef8469717fd678e43973b24b27e17063363875d26a0e74b44289d2218c27926` |
| `bench_sm120_layernorm` | `True` | `1274232` | `a09d96dcc3a79d9dc5865f3c135507b544753060e36a2b5a00812fb97a930d44` |
| `bench_sm120_runtime` | `True` | `2271168` | `c30254b0b15f054381a90a93cfd83472095efac5c12052f3f775350248080dca` |
| `train_gpt2cu` | `True` | `3105552` | `6d2b3a551cb1ec8e344668dedfbcfac11e0b7882184fa5ab28621ddcec36f707` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
