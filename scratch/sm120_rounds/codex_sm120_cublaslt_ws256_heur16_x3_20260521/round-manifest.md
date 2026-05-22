# SM120 Round Manifest

- run label: `codex_sm120_cublaslt_ws256_heur16_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_cublaslt_ws256_heur16_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_cublaslt_ws256_heur16_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `581`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `810ecba9b8bb5d641b03e71d4b76e7c0254948e12c91c4ebf146a42d03ac17cd` |
| `test_attention` | `True` | `1760032` | `8d594773bed42cbfeff0688058daafeb7d84d442a8de9e2703bb8f077cf8584e` |
| `test_layernorm` | `True` | `1237784` | `da60c5952a9f049c620bca7bc2868440ac9951dd564bf78636acf506e740f8b7` |
| `test_bias` | `True` | `2048616` | `d50441aecba94fb69c7a5e85e5a0cf97c4b6489b994c26b62208d5d795a41d7a` |
| `test_gelu` | `True` | `1139336` | `c6dc32f7a7befb23ebf76ba6c40ec8d2cea70873eb9ab0b33556f3062c7ac656` |
| `test_fused_classifier` | `True` | `1164032` | `63230c0e7a342ca743f0e3cc0c79197548814512eb8950961f22ff0ead7cf968` |
| `test_encoder` | `True` | `1165512` | `cb13c3fe34a6d7515e6d040a3be9279cdf7a0fcfaae6b365583a187cb2518a53` |
| `test_adamw` | `True` | `1143192` | `8cbb5d69c154e4a5febe55d63933fb5ff79f1e8df4e42e9eee424ce1513f97f2` |
| `test_global_norm` | `True` | `1138880` | `2330d8fecd41f207599b58b7bfa31026ffced9643fff93496f8cd155fcabb741` |
| `bench_sm120_matmul` | `True` | `2373912` | `b10e4b1b592a3f4a5ab0a9227627ce2fc5db1d908eab4151d3ba16d0e2db17cb` |
| `bench_sm120_attention` | `True` | `1728312` | `7d95a3628f56fca8270d5e23f67363fb9bf6c8e490db4cc3a92c396ccab1ee0f` |
| `bench_sm120_layernorm` | `True` | `1233728` | `413922ce4d728afc3099599319b415bc4acc172fbd5d5599b008d870101cfdef` |
| `bench_sm120_runtime` | `True` | `2226576` | `d1de1078e37beae9f63392fb8758cb9cd7065192a5f50ad571bbf424bcf2405e` |
| `train_gpt2cu` | `True` | `3064992` | `77fa411177a4d1d693fde3483756777c31a9619c57ab698bd733f50f093eb0f7` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
