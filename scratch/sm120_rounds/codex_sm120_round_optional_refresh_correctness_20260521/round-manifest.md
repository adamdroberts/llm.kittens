# SM120 Round Manifest

- run label: `codex_sm120_round_optional_refresh_correctness_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_optional_refresh_correctness_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_optional_refresh_correctness_20260521`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `497`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `9941f0f44a3a78b484fb9d5f943dd66f3edba5c22450dcf2a3e03049ee5936e0` |
| `test_attention` | `True` | `1760032` | `e08d521fa9af9c91bf3e4e0c916a04a0323680be6da1543084fa3f4baa9aec7c` |
| `test_layernorm` | `True` | `1237784` | `c05e6a01ebf228c4e25e83fd63b8951227e4f4fc29d099887692d498572d3b61` |
| `test_bias` | `True` | `2048616` | `331410abf734d803aedc3197616ca8ebce818f756afcb7f9863818759dd6a36c` |
| `test_gelu` | `True` | `1139336` | `badb23b579a23ae52ebbfd703fac33b64592f3916cfd27856e69846979484f7a` |
| `test_fused_classifier` | `True` | `1164032` | `4d688c704241060f9aab3caf4757947ad1a92d0ceef878571221413be7fb521a` |
| `test_encoder` | `True` | `1165512` | `114b9ee5bf294c691a5ceb29ddb2dfcb6b56126affdf5f7eaa981e4be16c9ed8` |
| `test_adamw` | `True` | `1138408` | `53e8a9d352ec807cf0f1cd1156a7383e6e1172decd7a3b0862b4ee8bdbe6c732` |
| `test_global_norm` | `True` | `1138880` | `06d527050277f2d1d7c790a222c270675690d5221492ebc984086d827375acd9` |
| `bench_sm120_matmul` | `True` | `2373912` | `066f8533f539c49938935d374f9f7aa013c3fdd4b90138dd3806794de73bfdf1` |
| `bench_sm120_attention` | `True` | `1728312` | `dcfd99f8ec0ba0261c2e1a0e43c6af602f49056c722c4099bfaf3e7128f967ba` |
| `bench_sm120_layernorm` | `True` | `1233728` | `9afe2a3bf04316910a2b345ffe833d6a027c162e75087e6cbd30c33876061a27` |
| `bench_sm120_runtime` | `True` | `2217576` | `71e922927bf582d8e52108487edadb02c4753a7876d74f4f0da0393f2e801b9e` |
| `train_gpt2cu` | `True` | `3045944` | `445ee85329929469427c20721857b8c2d405f6ef81a10dd3556d5390720c7519` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
