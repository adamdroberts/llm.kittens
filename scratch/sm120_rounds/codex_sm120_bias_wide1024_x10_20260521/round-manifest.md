# SM120 Round Manifest

- run label: `codex_sm120_bias_wide1024_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_bias_wide1024_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_bias_wide1024_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `527`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `f335cce26db1bb3825e76409eb6c7a42ffd83d7164fe6a268b13b549e1748a10` |
| `test_attention` | `True` | `1760032` | `f27235c45e51dd4d246c98bf3fb2b29567aa0a5fd845f8648ee1888c1ad4c268` |
| `test_layernorm` | `True` | `1237784` | `78ccad219384fa7163a4bfc32d73d5716962db0f1f218c7a0aa8b0b43bc60fda` |
| `test_bias` | `True` | `2048616` | `0e5ea4f317d5884acca184c5eda901fcff516e7adcd16ed867dcb934c0aaa726` |
| `test_gelu` | `True` | `1139336` | `a9356467ede8125666dad2270a88655986bafb7f24496b82ab182be5ac8c9a76` |
| `test_fused_classifier` | `True` | `1164032` | `82d583731ea950aa5b4dbf4c5f3e2c655f57e31c8f9501181dde4f14fe3fccc8` |
| `test_encoder` | `True` | `1165512` | `d778bdefaf3d3055e96c2ba02bd1b4668ce52ebb138d02e95d32b2b37a82f269` |
| `test_adamw` | `True` | `1138408` | `a6ad3b68da5617983c3d9f66f87207f6dc19a6e9f4f1b0d9f5d17f7ace1dcc76` |
| `test_global_norm` | `True` | `1138880` | `8ee97cedd9e28e1bfa07038bf2a7c3e7cfe6cb14b73bdb069d67c3deb73b3531` |
| `bench_sm120_matmul` | `True` | `2373912` | `39f9319a55ede8b4e19c5e0ae3aec5af3ec67651a2f539111e3b349aa4b16166` |
| `bench_sm120_attention` | `True` | `1728312` | `54f9b780f0088806417fd55ab4741638a64c4814e7bd0b1122b8d36cbf6f79be` |
| `bench_sm120_layernorm` | `True` | `1233728` | `b804a2715022bbf8952ae3ddd65b8e8dac57fa0d9eec62edd868f95c1a8702f8` |
| `bench_sm120_runtime` | `True` | `2221864` | `707ff37eff81f251f7140646b5f986080c820824d1204b51e5e8ecbc0196b306` |
| `train_gpt2cu` | `True` | `3060032` | `31dcb2d9c20664e0067425d344cb0bf508552bf03ba0eb16c1a00ec0c035f6f2` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
