# SM120 Round Manifest

- run label: `codex_sm120_round_cublas_bwd_default_x10_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_cublas_bwd_default_x10_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_cublas_bwd_default_x10_20260520`
- device arch: `SM120`
- max steps: `10`
- git commit: `0f21747`
- changed paths: `450`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2128288` | `cbd73e4bc8c4811fdee49b6d0d47d5328ec9b5d0874f7983169a863713c3730a` |
| `test_attention` | `True` | `1760032` | `27bdc0e2abb020fbcc59641f1273b597771511e8697cb80ac3556beddd87d5a4` |
| `test_layernorm` | `True` | `1237784` | `55c77338afdad92e476580156ca09f84e5aece58020849421ed8ee02a670a2a3` |
| `test_bias` | `True` | `2039504` | `91d980b0430b08dba3b82c0aa83d26c18718f03429ed34698ccd11e0f7dd6b54` |
| `test_gelu` | `True` | `1139336` | `aeeab94f77546016c22147089a7bb4d6f8a909680c6dd74fc78c04ac7353a9cb` |
| `test_fused_classifier` | `True` | `1146768` | `3f6190454ca34a1827ff968f9c04cc4f4cda1c436a30b71b686b95e1bf063bef` |
| `test_encoder` | `True` | `1165512` | `ffe4c72cfa0e63142649ea894d6a9ec673281a88ee1c5ea27587ba8c9b2917c5` |
| `test_adamw` | `True` | `1138408` | `dc87f02b2f1e8deeea9e09dba79da1ecf15260c6c049b873e2b6eca608a44b59` |
| `test_global_norm` | `True` | `1138816` | `8018d1718cb446c3e5c96dc7c2bcbc16ac7c9247a425e0ff2b6f453e2de36ae6` |
| `bench_sm120_matmul` | `True` | `2344440` | `b66df6cf20f25a8d7422fc9530ab120e0c4e3db64a9dc0723662762a04dcca39` |
| `bench_sm120_attention` | `True` | `1731352` | `c0ac1ee528d499ea2b44c38d0456c2cabfb29e32ee6f259f17f4a055b9d3867f` |
| `bench_sm120_layernorm` | `True` | `1229088` | `6cede0dfbcdba0a42f810449492c6b81089484cc525c35b57b6871d4c98328e9` |
| `bench_sm120_runtime` | `True` | `2168552` | `3717875f3a0f7993fd74d4e9916d1cc87663395e7d47d6136437ecb89bcb3f20` |
| `train_gpt2cu` | `True` | `3037016` | `dccc1e9ca2c43b69f9681e00a9bf1a9759be4f2d49b2757b14a0d2ac4b6c6237` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
