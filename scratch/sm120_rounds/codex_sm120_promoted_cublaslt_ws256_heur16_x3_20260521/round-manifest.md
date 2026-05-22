# SM120 Round Manifest

- run label: `codex_sm120_promoted_cublaslt_ws256_heur16_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_cublaslt_ws256_heur16_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_promoted_cublaslt_ws256_heur16_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `624`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `93cffe669a8eac17d3bd870271c4d73f83c5bc9f6ba11bd9e5ada2136804fe04` |
| `test_attention` | `True` | `1800528` | `d872cd83d543b76c101bcea452099dd35de43b6238cad1b09d123afcfe7a2c4a` |
| `test_layernorm` | `True` | `1278296` | `e519046482eea65e9a63d13cb4ff40b477be8bc5fd565990a688ad92efbbcadf` |
| `test_bias` | `True` | `2089120` | `0d1b3e55e4dfb47606e28758538d6329c5ef665bb1545b3e0d3f5457b7af2d73` |
| `test_gelu` | `True` | `1179912` | `1ff3c13d14d1ab3b9f801f3a52abde6d4c7699799012a6d1e715a16a69861589` |
| `test_fused_classifier` | `True` | `1208704` | `f22747017aa6ab8653f268b80c82474bdef3108c25d916b2e8427cae9a4bdae3` |
| `test_encoder` | `True` | `1210168` | `e7657a0c9ab5ac353bfadc22e93ee1682564d2bf100bdf01e2886fb3ea767583` |
| `test_adamw` | `True` | `1183768` | `648ac6100f11955a20c7581a80d7aab2eb0926029e4ed9183d81c406058edd51` |
| `test_global_norm` | `True` | `1179464` | `7463ba15adcb25e90ab313879953dfaa4f78dabe7928714ca81bf60b0df4629b` |
| `bench_sm120_matmul` | `True` | `2410304` | `7515b46c7d586c88618934936f578902f16e2ff21c94d1f419f686fe06a2a385` |
| `bench_sm120_attention` | `True` | `1768800` | `f2497a6cd5657561bc41e26d40121515aecaa964abc6360d07050e4c6aed817b` |
| `bench_sm120_layernorm` | `True` | `1274232` | `2a1684cc856a435544ebac971d21ddda15bc52dc0345df8f5a21322fd484713d` |
| `bench_sm120_runtime` | `True` | `2271168` | `887ca7613556326cfdad5d9691d745e95860a9f5d36971ad464b6347f8456501` |
| `train_gpt2cu` | `True` | `3105552` | `6dd93ebab9fa05243102d684fc8fc2e3675c8343ac2197e61b7278054e16495a` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
