# SM120 Round Manifest

- run label: `codex_sm120_restore_promoted_default_after_precompute_zero0_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_precompute_zero0_20260522`
- train output dir: `log124M/5090_S_codex_sm120_restore_promoted_default_after_precompute_zero0_20260522`
- device arch: `SM120`
- max steps: `0`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `655`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `883451371f1958367dd33c7635d044b117bf525b207284354b09c5c06aa88e08` |
| `test_attention` | `True` | `1800528` | `1405289ee3d0c4db726d21f333fca49c4e8967967159f39acb0eac58a6282410` |
| `test_layernorm` | `True` | `1278296` | `a543a3a5ed1281e7268d3f6652176221321dd8fd35118917439a9a25a1351483` |
| `test_bias` | `True` | `2089120` | `f653ac0cdb1ddd59580cf639cc1d0b544fbd65cbe2ba26d5acd2665c19b15f9b` |
| `test_gelu` | `True` | `1179912` | `ce60e15c29e38933e1aeea91999a80bfe7c71a6531ce5185ebf17ab5c99974a4` |
| `test_fused_classifier` | `True` | `1208704` | `ed3ce99ebbd8c38800eab5acdfd248c4b1e3b3c64dfade853187c5d6f6405bff` |
| `test_encoder` | `True` | `1210168` | `e7cdaf90f330eb540cf583435abff672e1202fa913e53d250c45192cb2363de3` |
| `test_adamw` | `True` | `1183768` | `4129370d9ed6f92b3cc5a7809e59cda9c88402952b9d10e0f22cea1e3601e5bb` |
| `test_global_norm` | `True` | `1179464` | `4986164cd2ce63fabfa72edee9df765870d3fa69214076519cd4de1c24ff6c18` |
| `bench_sm120_matmul` | `True` | `2410304` | `f791c21f3e9e44ef2f02b673c21f9b1b7ccf45f9d35480dd5bd30275cda81dee` |
| `bench_sm120_attention` | `True` | `1768800` | `580ac037b801015bdfe4e9191060a91f386832d339b9b533aa0975e99bbbc5b3` |
| `bench_sm120_layernorm` | `True` | `1274232` | `f7233344de77139e0939399ac585903a3be3d1a5b02a5f753d798dc836c5a3bf` |
| `bench_sm120_runtime` | `True` | `2271168` | `3cb55cade56f94f5658436c152dc4d0ede89ab9f365e04fb450ac787b54a1364` |
| `train_gpt2cu` | `True` | `3105552` | `a55f2b0ed7bdc347fdd3f2bba301b9ebeb0b53587a323a2a3275b29d3b445303` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
