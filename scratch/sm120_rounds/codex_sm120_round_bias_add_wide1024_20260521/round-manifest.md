# SM120 Round Manifest

- run label: `codex_sm120_round_bias_add_wide1024_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_bias_add_wide1024_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_bias_add_wide1024_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `505`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `ae1543d53b2a574a82c6fa7b9204a6763e16aa1d5fb6d80abff21917da93252a` |
| `test_attention` | `True` | `1760032` | `6a1e339a9cf20f42c6ddd38add6ee664586046082767e56cfd33a866ffe2457d` |
| `test_layernorm` | `True` | `1237784` | `20c1d29aadc0faa287c81a47461b8a58a02295f949a40563a6b7f150dc7735a1` |
| `test_bias` | `True` | `2048616` | `a28a639f8f2d35a43d1e2d1dd3891de1f073a5e784b85eeaaa3a1ce29bbc2b79` |
| `test_gelu` | `True` | `1139336` | `bb47f4dcb8886ddde02ceea68939ec4cb61ff088bb6e4308818a235a1ecf433b` |
| `test_fused_classifier` | `True` | `1164032` | `0e76a542f6c098873ecf966015e2e74f3b50118c9a304347e3026abae6494a0c` |
| `test_encoder` | `True` | `1165512` | `10d556a5348187206ab00bb8f59fc314ff4db15c526df68c40c1169f246188a6` |
| `test_adamw` | `True` | `1138408` | `24bfb110ba7dd4a8b6fbfad962ebda4a8fb17a028cbec2940b00e6784290fcba` |
| `test_global_norm` | `True` | `1138880` | `c0c3ca62db55f4baeab8ec8e72d73e7d36cca4aa6216a4ff9eaf0e7d5d915054` |
| `bench_sm120_matmul` | `True` | `2373912` | `a43824e5f21a6d066202c46bb6462500a30cfa10a8bcafef2eb025c0c1319fe9` |
| `bench_sm120_attention` | `True` | `1728312` | `20c3dd613563e1ec2b98572d30f42ad083eb1cd4e6e6d155bfd00f864223bead` |
| `bench_sm120_layernorm` | `True` | `1233728` | `dec3a5ac2cc458f3f197e2d433250c8bdf054bcecfc35221be905f8cecf258d7` |
| `bench_sm120_runtime` | `True` | `2221864` | `697454c2ca09ae64075f161fdebd997829af79aad50af252f198884989c16172` |
| `train_gpt2cu` | `True` | `3060032` | `1f85568b05c0d1e0dc30c3734d1f21567878460a773035e5d74bfc50f3e92f4c` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
