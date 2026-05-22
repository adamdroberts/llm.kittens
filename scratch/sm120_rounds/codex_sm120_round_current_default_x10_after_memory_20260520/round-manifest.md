# SM120 Round Manifest

- run label: `codex_sm120_round_current_default_x10_after_memory_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_current_default_x10_after_memory_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_current_default_x10_after_memory_20260520`
- device arch: `SM120`
- max steps: `10`
- git commit: `0f21747`
- changed paths: `460`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2137216` | `efab07dc9d856fe2be266db3a61b2c55d93772a5467331d7ccb1796514a9ef1d` |
| `test_attention` | `True` | `1760032` | `9ce1598184912f8241faf59b752e7acf4163bca9a04da34c9543732d9fa89633` |
| `test_layernorm` | `True` | `1237784` | `5c7d12d402611e61690ef04022bd3b652617d661515c709488c5fce011b0a46e` |
| `test_bias` | `True` | `2048616` | `8338261aae2069bc4b285c58770984ab0fbd99e34553901137b36c4a43795b32` |
| `test_gelu` | `True` | `1139336` | `b6a3da865f1b786057d4e3803c8a674ac22d685c04cb5c1dbeda64d54fb710e0` |
| `test_fused_classifier` | `True` | `1164032` | `94bac0426000ba77bca0a56f0423617dbd482b578d422c9d1922a30e7b85960b` |
| `test_encoder` | `True` | `1165512` | `3da2733405cd2938dccb897ad19a17476c19a42a87357b49fa071c4f785256a2` |
| `test_adamw` | `True` | `1138408` | `ddb32b814f21c4f0ec56a69f5a3b30962ecbd276c11cd133a5279f10d07d3d1f` |
| `test_global_norm` | `True` | `1138880` | `c110555e4bd3c61f25569de19823f8dd0d41a9821ea7f48f1ed34288484a365d` |
| `bench_sm120_matmul` | `True` | `2373912` | `81bc6ecf9dc819cfa5abfe771328b191d48da150960fa9cbfc751be2686d22f8` |
| `bench_sm120_attention` | `True` | `1731352` | `4936062ab3f94437d292bb5f960f84cd17ad66e2bb50d8741618a04ee0500897` |
| `bench_sm120_layernorm` | `True` | `1233728` | `24f7f9d682121746402c3f2599f68882aa4586b0248cbfe6c91fd77e3c3f15d4` |
| `bench_sm120_runtime` | `True` | `2199032` | `ba7a9a1f693d01a0c13789084ea2143330b56a098e6528287024334791ff7c29` |
| `train_gpt2cu` | `True` | `3045944` | `27d0fec7ac508e121a5b3d28bf2a01c4baddf170cbe324b346c6edccce5115cb` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
