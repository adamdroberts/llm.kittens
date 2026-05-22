# SM120 Round Manifest

- run label: `codex_sm120_attention_atomic_dq_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_attention_atomic_dq_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_attention_atomic_dq_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `537`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `80c063b9212f5b31e062097015650c0a3a29d1afa25dbbd7fb491525ce391099` |
| `test_attention` | `True` | `1639512` | `431d0178a2a89c8601743885edd9804c8cc736ea107335ba216efb41b700b5be` |
| `test_layernorm` | `True` | `1237784` | `578c016ee35bd8d5443f50fdd67ae8608e9846793398c0e1ca42ec942a25ed8e` |
| `test_bias` | `True` | `2048616` | `0088b52886f3f06a4bf86e0e707d69a2011db85fdec6c5a4dfa69ad1cff62909` |
| `test_gelu` | `True` | `1139336` | `1f45cb234f280edc7b6ffa9fc415653069f6c42fd617e49be9aec8f8fc70895f` |
| `test_fused_classifier` | `True` | `1164032` | `6f98152af0cffddb40de2da6ce71eb48af47576993329f35910c63e4ddb3cbe5` |
| `test_encoder` | `True` | `1165512` | `c2aa71cc85b9599e1c6098cbdf8efa0fb75a36220f86efd47ac5568010ab679e` |
| `test_adamw` | `True` | `1138408` | `6855509e2044bd8b2a735378f98963cdfb70fbf92f368b9772c6d5f9b7794e06` |
| `test_global_norm` | `True` | `1138880` | `dcd9d4f80fc43a843f62de59c265512ab24d3ee552c9421981a1300b97c29012` |
| `bench_sm120_matmul` | `True` | `2373912` | `a9210c2e72dfc07a3f9ad76b56fcb41da584f24eba381cf55fdb4fd514b980b4` |
| `bench_sm120_attention` | `True` | `1626720` | `0378cc05f559df3c47b6791295d0235ea3b99abfb15f358a4f626516aeb26eb5` |
| `bench_sm120_layernorm` | `True` | `1233728` | `45dd5c270ecd737b7268d2898a3183c8159087c7a9a15894dec15b7e6436289d` |
| `bench_sm120_runtime` | `True` | `2221864` | `fb23cd1ff64c3c9021f1b8e75182040d9e741e5bc9952f7ebb31b1011fd55517` |
| `train_gpt2cu` | `True` | `2966784` | `e8e2f2ff04878317ff13c1022434b12164eb04075ae477e3a7a79e34bd9f1457` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
