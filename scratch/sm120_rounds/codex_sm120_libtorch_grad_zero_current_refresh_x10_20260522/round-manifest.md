# SM120 Round Manifest

- run label: `codex_sm120_libtorch_grad_zero_current_refresh_x10_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_libtorch_grad_zero_current_refresh_x10_20260522`
- train output dir: `log124M/5090_S_codex_sm120_libtorch_grad_zero_current_refresh_x10_20260522`
- device arch: `SM120`
- max steps: `10`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `1`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `683`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `522d0dd314dfba46fe807ffe22fcff940533e3b0d6cea09b373bc3e6518175c8` |
| `test_attention` | `True` | `1800528` | `ae3753049bd9303344bd101220529e37b4e6e5da986b67d712f8713170f52b31` |
| `test_layernorm` | `True` | `1278296` | `0aee334f156eeb433b63877493829632991f6e28003b77f651f35fbbfb852653` |
| `test_bias` | `True` | `2089120` | `564ac837338fc085224e04abfc5ce4dc6efe04eb9d78e10494b47fb0d60720b7` |
| `test_gelu` | `True` | `1179912` | `f57ffbeb8737dfbadcea9c1b5edb38e121ce300cc2dbdb0897cb136242a8a2b8` |
| `test_fused_classifier` | `True` | `1208704` | `d33b4499e1ce18104bbfb98cbfe24df6e3e3dc5e516e068513b848a9f2f272f1` |
| `test_encoder` | `True` | `1210168` | `b1dca3a4163333fac61d3a808899c0dc8d62f10e8f0541933a91688406bd9d31` |
| `test_adamw` | `True` | `1183768` | `4fa70240266f085c69d0cf7d19b11e9a9482306327b95f56e10103653c138278` |
| `test_global_norm` | `True` | `1179464` | `1761fd91ccee949c9b748e17261e56cc4ac20ec828f8a0625db497a8243878a8` |
| `bench_sm120_matmul` | `True` | `2410304` | `91cb93516322573b6f2e9f5bf823fee4af48791df46ac3b10b6f413a577458e6` |
| `bench_sm120_attention` | `True` | `1768800` | `c492f5034c0cac0ab1f69aa31ee6fb7828e583703b3622c2be75948a0f9205f1` |
| `bench_sm120_layernorm` | `True` | `1274232` | `0e3b89288ab468845ec5203f1d1da1f5c2483ce08abb69fe23780528942b6e37` |
| `bench_sm120_runtime` | `True` | `2271168` | `f7e3f8c9ce89eb0ac8ad163690c98f4f9b4d57375f6cc45a43dbcc36ea07243f` |
| `train_gpt2cu` | `True` | `3105632` | `92845cb353040db15e54f08358a54b6f73c01444cfacc1ff221a2e239db3ba01` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
