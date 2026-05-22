# SM120 Round Manifest

- run label: `codex_sm120_promoted_maxconn4_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_maxconn4_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_maxconn4_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `632`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `ab9664080bf6e2b44c005aa0539acb70d6197882a050065c295b3ff6a8967b32` |
| `test_attention` | `True` | `1800528` | `7c50fba2eebb1a8d95165a0aaa88f0b973fcc5bc2f7a8df07f2b8648607d1e80` |
| `test_layernorm` | `True` | `1278296` | `c34a0248ccd3c2db8d5633f9ead833f17a72fe976ac694b1221b36c9a7832a73` |
| `test_bias` | `True` | `2089120` | `aef9eec090e1d242befbb319cb19ee8eaa5fcb035eb1bc0d2b199d9ef56de08a` |
| `test_gelu` | `True` | `1179912` | `e9983e33b042b30e6cac2b2b29ab666bad78b70382e3504960d3feb33ad3636b` |
| `test_fused_classifier` | `True` | `1208704` | `eee682482c0236799dd374960b4724261b5bdb980c6ac241a8540feeb42302dc` |
| `test_encoder` | `True` | `1210168` | `6f97af7a31dc9422264509e5a5002c1138028721494ff715092b8065d1f06f04` |
| `test_adamw` | `True` | `1183768` | `5520657f2837895118bd17e9df1ae52eb252de85c09025540c767d9a75bc7061` |
| `test_global_norm` | `True` | `1179464` | `d97962f183f0e9b92675366dad7a247a38498b0ac66dc9cabf34afbd1b1067cf` |
| `bench_sm120_matmul` | `True` | `2410304` | `38dfa6beb492bfcfe11480995bbe0db91ae63d9a75c828ede3a852f51e7e3b4d` |
| `bench_sm120_attention` | `True` | `1768800` | `36630cc00cabf7aeff3d62696f78133a4a0b0d00af8afbdcf90d4272cfce8863` |
| `bench_sm120_layernorm` | `True` | `1274232` | `a6a8a0a6739216a03ec0b84742380fa4533cbf03f73791f0be8f92cf3a5c6cee` |
| `bench_sm120_runtime` | `True` | `2271168` | `c23559ef6c4d878b96876351ea29b632116deb6865fef01c5afb9451589ba956` |
| `train_gpt2cu` | `True` | `3105552` | `3a7b18eb7720a58263e5f152f516e31bf3f9d9f0c7aa63450b175818ff07c48f` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
