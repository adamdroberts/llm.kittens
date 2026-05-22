# SM120 Round Manifest

- run label: `codex_sm120_round_cublas_dinp_attproj_only_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_only_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_cublas_dinp_attproj_only_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `508`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `54ad7f90df619ea947e4e00e210ac48dccc387820389d38f4246300461525daf` |
| `test_attention` | `True` | `1760032` | `0c45492d8fa3b5ac8a6bcf18b439ba062f0098d891c4558e80886f9c88a29d16` |
| `test_layernorm` | `True` | `1237784` | `a8b8b8eae08f84c59e5cdd54a285b6a9f4f35ef68b457c79205f58339dd32d15` |
| `test_bias` | `True` | `2048616` | `ebf2736da038ead888ced3ef64a2ea46072aa4b6617cb0131f5376df28a8dbe7` |
| `test_gelu` | `True` | `1139336` | `96a5ddf93c381baa602662bc16c4838fd00f82ac9e8080402688e2effde7bc6b` |
| `test_fused_classifier` | `True` | `1164032` | `679899509bc05760ff7c9084608ebfc81189d3bddb848dc7c31ff3d746814bf5` |
| `test_encoder` | `True` | `1165512` | `f33a92e023b6da9ef3ea17cb7795545260a6f33f9a0e2d40c5e401121d9dbd53` |
| `test_adamw` | `True` | `1138408` | `6493b065aecd525e960e68dc7e40cd397f6648fb5d2646e1d0b795bbc87bb13b` |
| `test_global_norm` | `True` | `1138880` | `3f5c01d2ef7ca173ed58841324ed81564c906794d7f79bae9c0030c883dc47eb` |
| `bench_sm120_matmul` | `True` | `2373912` | `bd0591f5da48beae89f36c83ca2cc384e1495d6a02cf02536cd185641b778290` |
| `bench_sm120_attention` | `True` | `1728312` | `63ab32eb3e0aa7894138646c3ca5644d4b82d7c1b45e87b5b3973f8a8b93a3dc` |
| `bench_sm120_layernorm` | `True` | `1233728` | `f92107854343400f586ff0ab5c9016d596193e4173925b632f81bf84e67f19d1` |
| `bench_sm120_runtime` | `True` | `2221864` | `2db557a648c1c2f41f1b380c0bfe70471dd59b8e2045b1151e634e2a690f10c8` |
| `train_gpt2cu` | `True` | `3060032` | `1b6e13597b93ad7f3b74de020dd27328ae6bd4c0a0edf0a0b3c99770bd5c582f` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
