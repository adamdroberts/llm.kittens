# SM120 Round Manifest

- run label: `codex_sm120_current_control_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_current_control_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_current_control_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `519`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `2a48b061bd947489b21a8c170ed33c8fa31198457909919a90b62b56029ce509` |
| `test_attention` | `True` | `1760032` | `acff467185e19f1d4ba279e62add7ebce8a5e5a1e185d386f1fca7cefb426dd8` |
| `test_layernorm` | `True` | `1237784` | `3cabadf0eedabf990dbadd4cb55e110ddd369fa300f9b1472d4c24e41baa8134` |
| `test_bias` | `True` | `2048616` | `639e6e1d9bbb1cb2ef9ae38a65eab70ac66f0e6493e6969547dfdf72c253cde8` |
| `test_gelu` | `True` | `1139336` | `121ab588e9b619dbc36141ec16f9fcef471400d0ed8cdb1ebcce68b869fbbe71` |
| `test_fused_classifier` | `True` | `1164032` | `057caed2cb070dede55fbb32ccaec2bbe4fdc775f3631253fc4693c617f90240` |
| `test_encoder` | `True` | `1165512` | `42491a7c957cc565bdf8c1e63aa5dde4eac98907b44fb2556bc4ccaed1d573c9` |
| `test_adamw` | `True` | `1138408` | `a86eb3e99c47014023f146f7991a750fe624d3f2450656900ee5f3ccb8e60a1e` |
| `test_global_norm` | `True` | `1138880` | `1cd939370c75a9bdd012b12bd2095ab7cef9df60a2a77cd557dfefaba8b1e6d7` |
| `bench_sm120_matmul` | `True` | `2373912` | `2cd791112adf2fbe65cc613c5cb6ab6bf5e9d03be6e21d55482cfdf775f697f5` |
| `bench_sm120_attention` | `True` | `1728312` | `9195cd3aedb306b6f5ef2affb18aac95924365640bae0c1f07f324e939196dc3` |
| `bench_sm120_layernorm` | `True` | `1233728` | `a2fe6960ec6e803071c1a9706b4bbc81d3fc97b7cfd3c5fc583b01cf35cb9e80` |
| `bench_sm120_runtime` | `True` | `2221864` | `ac12de8b9cea9025898f2bd464e8de5bf9b26b741b514a89a2b596b6971ae144` |
| `train_gpt2cu` | `True` | `3060032` | `1df684d4313a1edcd812bcf8a74294bf9af4f20d1cea7e49efd368ecaa12d9f9` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
