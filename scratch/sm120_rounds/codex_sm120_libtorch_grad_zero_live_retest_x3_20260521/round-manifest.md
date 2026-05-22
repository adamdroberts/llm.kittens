# SM120 Round Manifest

- run label: `codex_sm120_libtorch_grad_zero_live_retest_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_libtorch_grad_zero_live_retest_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_libtorch_grad_zero_live_retest_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `1`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `550`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `36f9d00b8e6c327116670e87ed72fa4253cb5af0047295bbdcc9fe8903a15383` |
| `test_attention` | `True` | `1800528` | `1c261ea5925a3cc21d8df5fa46927496f75d309d1d5c02e0057aafa929b91dd4` |
| `test_layernorm` | `True` | `1278296` | `c9f514129596b78da09f96637b256d1a9412a3e4d55127ab0069466e954761a9` |
| `test_bias` | `True` | `2089120` | `0a041493df3e85c90fc2cf08845125bdd5c450cdcbbe9936ca89be615bdf86c4` |
| `test_gelu` | `True` | `1179912` | `c2c1d84054c87c6775ff5391e32023debe1a8e05e55c1875e8e48d7dfccabc3a` |
| `test_fused_classifier` | `True` | `1208704` | `0ccc073c826a1a981a01d141c4c22e6521450260d2a98cfe1b81611bf4131244` |
| `test_encoder` | `True` | `1210168` | `2cf56c02afcce144d099122b936639e7fd16e8130513d251401e8d0573d187b4` |
| `test_adamw` | `True` | `1183768` | `68debd572fd375d40100c830c409094252491744a8db27076b23682e5af6dbfb` |
| `test_global_norm` | `True` | `1179464` | `27f1d597371561c13f037ab5bd2cad74450709f89093ee9a6385628e7a13776d` |
| `bench_sm120_matmul` | `True` | `2410304` | `562a8c3c356ca299fa1f81c9b7ec8a0fe580d379c76aa2363118f9ee3f847b65` |
| `bench_sm120_attention` | `True` | `1768800` | `cc1feda054564d9b57318253bf7f92e42edc16f38582c371f2ab97994c6f1407` |
| `bench_sm120_layernorm` | `True` | `1274232` | `9847f8eae8bc840a82b36f11bdf7a1e8015be6c35fd025696514006f7ee7bd4b` |
| `bench_sm120_runtime` | `True` | `2271168` | `7e14c566e562c429b2a9205cc8ae46f72a5370bddf27ba7623e4713206a0b374` |
| `train_gpt2cu` | `True` | `3105552` | `24e881d8991ce0123afeea1d2eaf1a64a4b8c22fb3dc85a2e2879d3189377b3a` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
