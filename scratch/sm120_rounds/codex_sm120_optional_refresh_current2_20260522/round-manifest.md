# SM120 Round Manifest

- run label: `codex_sm120_optional_refresh_current2_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_optional_refresh_current2_20260522`
- train output dir: `log124M/5090_S_codex_sm120_optional_refresh_current2_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `687`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `7d3cf289ee64fb29a6441427fdf2032e34146dcaebd6ce7d034c5059f0e8812a` |
| `test_attention` | `True` | `1764128` | `82acb5bfff19f1e8418110f2fd77d61e30dde603b27f35979d34a4fe0cd0641e` |
| `test_layernorm` | `True` | `1237784` | `2023d3fb7a01a3d1a396fa3ddadd28b9861c6703badbbd604275b5622d83aef7` |
| `test_bias` | `True` | `2048616` | `640b96463df3de13380d465be54b43598d825d034d5e24be685a5a2039552e0c` |
| `test_gelu` | `True` | `1139336` | `402f2afa64adb66f3d9ab1664468895435a9b8681b8488ef074971b8807a4a9c` |
| `test_fused_classifier` | `True` | `1164032` | `b72beefebeffd5863d88c5b5f1c4ef547d4cbc37c9523d870d09a8b3a2ed16e3` |
| `test_encoder` | `True` | `1165512` | `f7a5885558ac383d9ef18702613aa63fe947b46cab00a4d6f67a6c3ac9b46d79` |
| `test_adamw` | `True` | `1143192` | `1c1213991f15ac035b6adbc7e9467930952b150be38764003ffe8efc5127ecfa` |
| `test_global_norm` | `True` | `1138880` | `9bd4ceeabb797526fb4efef1ca0ac4c06b77e3fb65986f682dde458d46760a77` |
| `bench_sm120_matmul` | `True` | `2373912` | `871e59e8907097a16ad23eb40ab4e720c0be888b001426d5d679523923a9388f` |
| `bench_sm120_attention` | `True` | `1728312` | `3edceca079b7ccda6f6690c42884584a84d2d1c452ecb2e01e0043eb6f5dfa5f` |
| `bench_sm120_layernorm` | `True` | `1233728` | `ad50aaf27c415622dbf439d8d6f9edf9fcb426edba540763bda32c0cb6d3ceef` |
| `bench_sm120_runtime` | `True` | `2226576` | `3b42c503ae5a3d331abe402241fa8ea64086dbedb9e75cc191286ae1813c3046` |
| `train_gpt2cu` | `True` | `3064992` | `7dbdbe4929404b1de197a82a313e6bb2311d48fd8b0a0eac99d4dffc20314323` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
