# SM120 Round Manifest

- run label: `codex_sm120_restore_promoted_default_after_heur1_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_heur1_20260522`
- train output dir: `log124M/5090_S_codex_sm120_restore_promoted_default_after_heur1_20260522`
- device arch: `SM120`
- max steps: `0`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `642`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `9e2290d7c91515a5b2e8e4827c142377129be1bc63f532721a1f3c593cc4a59b` |
| `test_attention` | `True` | `1800528` | `2a6807bfcc1381a00d92075e5a9e13f2470891009a6ae1c92011503e2681d334` |
| `test_layernorm` | `True` | `1278296` | `198469013dca4660e3ad3a9e333fb9a1f86db812fd807a5c84132e22727f1ca0` |
| `test_bias` | `True` | `2089120` | `5f3d4bcca581e345fe307438d3272b73a6b6084982e806b8ba45e49862c8ef25` |
| `test_gelu` | `True` | `1179912` | `6e9b249617b80b25d4116fcc2968ca5129ce38e0fc163e12eda93515e815e73c` |
| `test_fused_classifier` | `True` | `1208704` | `5873c211549ea0a9ded84bd3403e37a8cd3eafe67d0dbcfa96cfdb60350c4071` |
| `test_encoder` | `True` | `1210168` | `0a94b007e0b051d30ebd28f406660d09b5de2741bfb9d59f7c6027a97db0fbd2` |
| `test_adamw` | `True` | `1183768` | `64e8c4cda49c3546b2eb97257fc87408f420492cc53fc20731800c1763c9ef69` |
| `test_global_norm` | `True` | `1179464` | `01e6b0d1bab5078efb7a210880812ab4f41a2dcb1f31f24190ba23afe38eeade` |
| `bench_sm120_matmul` | `True` | `2410304` | `c8295d7c49bc873b539e08d2e61e63e65997e41eb79df49d0d84eed924571c29` |
| `bench_sm120_attention` | `True` | `1768800` | `9d15b80798611fec10cae3790639e86270e8249be5c453308fb5f77e40882c0e` |
| `bench_sm120_layernorm` | `True` | `1274232` | `6d139039baef379be54c451e32a1adab881e6d80d818e9730a89c8c162d0ec8b` |
| `bench_sm120_runtime` | `True` | `2271168` | `86afadf3efa492058492ee935c3693c43317b293c0bd2267101733e7b3cb5dd0` |
| `train_gpt2cu` | `True` | `3105552` | `240bb78c20ce1a3d1754ce2690350351e2dc6ebc79325f1a6b154ee002f4f278` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
