# SM120 Round Manifest

- run label: `codex_sm120_promoted_backward_n96_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_backward_n96_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_backward_n96_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `652`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `f9b1884c03b739ec3383deec2dcb92c12f7f9277d456bf2c98669da8dd762675` |
| `test_attention` | `True` | `1800528` | `db0366aad18064180b2c1ce32b18325c159fedf5ad52a1d7281d2627fa1fab1d` |
| `test_layernorm` | `True` | `1278296` | `6d00ec7735bd0d4d2ca04bb09fdfe42f8aa3fa5a559038015961910ac3c6f3e2` |
| `test_bias` | `True` | `2089120` | `8c9e5c66c637948b7d4fbcef14e603d54326c296ad7f1669fe8d29408de2c121` |
| `test_gelu` | `True` | `1179912` | `afad9bff8696996b2d8ae13f7e7b4b98fa5ed0f9209bbda2dfcfab231c62e49e` |
| `test_fused_classifier` | `True` | `1208704` | `e52e537b2a2d1f938452416213c02a32d3c8e541211403b18f1e25ed852caf4c` |
| `test_encoder` | `True` | `1210168` | `08af3b79a8fe2bb3b95e1bb587552e951c26391c40caa3b33a8940daa3de2c3a` |
| `test_adamw` | `True` | `1183768` | `82577534bb8e2db85769d3bdf35a5c15edd42b683a7f93277add0f31d23bdae0` |
| `test_global_norm` | `True` | `1179464` | `fe1f7b72b2027df14e922e4498bc9fcfccdfcaa489662a53f91ead8030e7aaf1` |
| `bench_sm120_matmul` | `True` | `2426000` | `0f1be29d84f48fcdfc0619e1478a30eeeccdde744d18566ed8e287cb6813afe1` |
| `bench_sm120_attention` | `True` | `1768800` | `1770ae9a98cdabe06a55fff442628956b9e539720a1de1dc4e409dfc5cdf50dd` |
| `bench_sm120_layernorm` | `True` | `1274232` | `4363b9fe2a807a8aaf03a4dd07ea6d35923571df29e05395b1ec6f832b6648d0` |
| `bench_sm120_runtime` | `True` | `2271168` | `6af8848a8fdeb748776cf23bad0e0634a9017d9beac7f5f97e24c61c82ca4a7d` |
| `train_gpt2cu` | `True` | `3105552` | `2319acc6c1190604bf753404dc2ea9b47b93c2b48cc8058132d438fd5c6108ab` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
