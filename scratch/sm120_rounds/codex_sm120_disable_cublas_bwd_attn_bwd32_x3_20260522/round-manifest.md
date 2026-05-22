# SM120 Round Manifest

- run label: `codex_sm120_disable_cublas_bwd_attn_bwd32_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_disable_cublas_bwd_attn_bwd32_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_disable_cublas_bwd_attn_bwd32_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `644`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173408` | `85d512afdfc483c88df9a336e23cf9d0708b7a7cc7504e3bdac07b02f7f7309a` |
| `test_attention` | `True` | `1910920` | `68cf810952b5eb74311ad275b70a4b7ffc66a1bb20cc13dbf43f2e33f96c9bbe` |
| `test_layernorm` | `True` | `1278296` | `ed29d956fe52ec6b08e2018950c7477b07e7abdc01ddd497522fd1727691f38c` |
| `test_bias` | `True` | `2089120` | `1f857f43f98a18bc447a303ebdbc4ee74eaebcb26c569387d2fd3769192553a4` |
| `test_gelu` | `True` | `1179912` | `5139e46118751c89edb2d1705d9aabf5b4c9f5310189b936100ccffd9eb859f2` |
| `test_fused_classifier` | `True` | `1208704` | `fe11e499e7de4627838599f2502ccbe7f4d830910726ce24668ae8345c2436e8` |
| `test_encoder` | `True` | `1210168` | `e68546ec23485fef77c6ad2c734fea92e7479e6d7d80e62234179195b2251ec0` |
| `test_adamw` | `True` | `1183768` | `60e5655669adf22e7ef1c87c6f442ea96e06af710e180110b0f019d3a8ecc179` |
| `test_global_norm` | `True` | `1179464` | `c4f6b4b3633a3d742587ab4c86968a7bda7a14c9164d092497434e2b1a040ab7` |
| `bench_sm120_matmul` | `True` | `2410304` | `5acc3795597fb2f4d80045d57bf44949f2906129d83f1c232205411c47eecb46` |
| `bench_sm120_attention` | `True` | `1879736` | `dd5ed9d2937d98e10bc7de4cfed7d6a957df71d2e3c37b536696bf1e319a9966` |
| `bench_sm120_layernorm` | `True` | `1274232` | `a5548e6286bb46af4b8b68729a6b93c2badca95cdc348cc7a4a5b00ae845f87c` |
| `bench_sm120_runtime` | `True` | `2271168` | `b5861ccdacdebf4768deb6f4f0107d8ccd488a43ab1a9516456ea4bc251e3ed2` |
| `train_gpt2cu` | `True` | `3211664` | `29a84804f143a9d01eb3a64d0d6afe50bd2e5976990164c0a5ecdd856f514d8f` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
