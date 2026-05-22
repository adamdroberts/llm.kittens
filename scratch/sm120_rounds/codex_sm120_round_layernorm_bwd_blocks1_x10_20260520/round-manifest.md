# SM120 Round Manifest

- run label: `codex_sm120_round_layernorm_bwd_blocks1_x10_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_layernorm_bwd_blocks1_x10_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_layernorm_bwd_blocks1_x10_20260520`
- device arch: `SM120`
- max steps: `10`
- git commit: `0f21747`
- changed paths: `469`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2137216` | `20eae5507c9c61966a0bdfacab3836589b9cac4c7ea2da19b3f910429629c5b6` |
| `test_attention` | `True` | `1760032` | `d3731c0a392ce7b504d3c645c685e65f68e7a2c0e2c094a3352cff0fc9be5034` |
| `test_layernorm` | `True` | `1237784` | `3a6d1225b67a01803dd8c113396e835f8d6d34237f35bb05df169850797c676e` |
| `test_bias` | `True` | `2048616` | `8c0187c78efce6b9b40e5b4d3c98bd2bcbfad8335689736fcae7dc898a9b14b4` |
| `test_gelu` | `True` | `1139336` | `19274fb323c1c6bf3c27bbb25e486a1dac5e8f5ec6953bee906408d4df593c74` |
| `test_fused_classifier` | `True` | `1164032` | `28ca54c6b21372d6d68d57f3839e85c57ef17d16468dfa72b0c235ddf5e627b4` |
| `test_encoder` | `True` | `1165512` | `a84f5739aa9aa59c712e4c83be06a11556505bb63b67356c5269a963f0d18e1a` |
| `test_adamw` | `True` | `1138408` | `6be2d166b9e3c3e07cacaee39b9749047182cbb343ea38000d573306c5aa3643` |
| `test_global_norm` | `True` | `1138880` | `0b676341e3c6463969f98f50c8fb044b95ea98d2d252c77969381f2ed2143c13` |
| `bench_sm120_matmul` | `True` | `2373912` | `cbfe0a5b3590db51e60966aa2cc8acd36cdf47ff2b0f3c0e4836c09a26d45071` |
| `bench_sm120_attention` | `True` | `1731352` | `666e2a16bd49f82f9b44be102d07e1769fd6d7c071827fd73f42002f6b21a234` |
| `bench_sm120_layernorm` | `True` | `1233728` | `b12da31c28333e1e6c4efc22308323b99909e0734aa5b6761f10ae8dc8691ce6` |
| `bench_sm120_runtime` | `True` | `2199032` | `27f20517683fd2ae980aaddd42e547763e18c77ee5b5e80ae04e4af5d8baea2d` |
| `train_gpt2cu` | `True` | `3045944` | `90d2085ea12bb061ccce4048e0a0692c590cfb429bb20c8d54f3697c5b50b4b7` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
