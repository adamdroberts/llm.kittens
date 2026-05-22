# SM120 Round Manifest

- run label: `codex_sm120_promoted_no_cublaslt_gemm_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_no_cublaslt_gemm_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_no_cublaslt_gemm_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `653`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2400784` | `ee0153ce4757a47fa1a91d8328d9b61e95a3195a38e219f1463c950c6dd17f3a` |
| `test_attention` | `True` | `1800528` | `ba595e507adc36bf4d7320c6138bf9ad5c9200708c8e02ebfd62b087a23b4151` |
| `test_layernorm` | `True` | `1278296` | `4cb1fb2ac230941c4879f7dabf7dc68544ad8e0571da21cca541262740990ba5` |
| `test_bias` | `True` | `2210720` | `4f43b62e3b252be291996abae01cf388d208e9e8278d8053bf192087d3820f5c` |
| `test_gelu` | `True` | `1179912` | `59e6fde6ec10df9fe05cf4b0e5081548f406614cdf7843429288ffa297b2ebc1` |
| `test_fused_classifier` | `True` | `1208704` | `7c96058b411817e125c3faa472c87d59309bc4b2b07d6cb9cd2f211b95d9f64a` |
| `test_encoder` | `True` | `1210168` | `a3c078960b6a19122da4ece8889eb11acd4bd0099850bdb43dfdd81e48848210` |
| `test_adamw` | `True` | `1183768` | `36832e9075959d0f11f480b104126580c1b960555399b8e12e6f1d0e28ba2486` |
| `test_global_norm` | `True` | `1179464` | `c72374c89712c66a50b634fe9b2003fbbde0c292f9ac10267389a814a2c6bbef` |
| `bench_sm120_matmul` | `True` | `2410304` | `b9c65e74a4b8c9be3b6449282242fab30aca8c5bb1dd22373ee9068a1f76ebc4` |
| `bench_sm120_attention` | `True` | `1768800` | `b313ced28a9221cad610dac2d73141a7e53eeb622740d4ebcd210fcb8990dd62` |
| `bench_sm120_layernorm` | `True` | `1274232` | `9084a5105157b75547e31235c9b7e9d946484b57f55c2f7499cc539040f3da3b` |
| `bench_sm120_runtime` | `True` | `2388632` | `311cbc9121a6ce559ccc6a50fbf931ff42d76ec6c1f9ac444e066ebb0489edb7` |
| `train_gpt2cu` | `True` | `3330528` | `816602d671bb0c73f0438c425e1976cd9b9b8436f5cb70d8ed53fff3be0d7313` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
