# SM120 Round Manifest

- run label: `codex_sm120_zero0_disable_profiler_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_zero0_disable_profiler_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_zero0_disable_profiler_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `0`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `677`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `5d3ffd25108fd746f19376f0965adb0b0b638c159fc6b472e432b4ad6fd3cd63` |
| `test_attention` | `True` | `1800528` | `7205d58a44543f865cd8be5f821015c44d78fc42c01bc64288410cb82811f38e` |
| `test_layernorm` | `True` | `1278296` | `d32c0ce864bea7305092f582d2756981e10df50940feb32192cfd4d3898f2a66` |
| `test_bias` | `True` | `2089120` | `3b2f2810c0f4a4bcecf59f0cb7ff3aaf050c547766f8ee4a12e94ef12f6a5291` |
| `test_gelu` | `True` | `1179912` | `b6b1559e892c69b26de32e6c9ca9d41053a2382890727b6b0c781ca61fc0ee13` |
| `test_fused_classifier` | `True` | `1208704` | `3ddc0617b3ac52c26244f36afd92abb63fae0e47526000b717a2f50b63b08775` |
| `test_encoder` | `True` | `1210168` | `d4e081611760d15fc01e038d8616af329743576c66456e15d24b4c314b00f72f` |
| `test_adamw` | `True` | `1183768` | `83f00009a1d894ba91b75c0832b3d3109783bc3caf370c5580bd4de8504a486c` |
| `test_global_norm` | `True` | `1179464` | `9ec6fc5020fc52be3f8ad9cb87d393d1f4ad67e9b02dc4639ea0270f0cc34200` |
| `bench_sm120_matmul` | `True` | `2410304` | `fc8a71569b7dfeca0b255aae7a51a658dfa66aa1a372bacb962c8d6333b8bda3` |
| `bench_sm120_attention` | `True` | `1768800` | `f3644c7094235c20b261f05547e506a56bc387da5bc4ef60f7796b855d35a9b9` |
| `bench_sm120_layernorm` | `True` | `1274232` | `c55d36a8db75edbdd83152f4112e94f082d71cfbee7c755d58961ea9e8d066cb` |
| `bench_sm120_runtime` | `True` | `2271168` | `f8e47616ae2027daa0a73550bcdde8c4f9bf4fe624fb7bcbbd34a83915e78d04` |
| `train_gpt2cu` | `True` | `3105552` | `5499204f75f94630e7ca91435ce72c0cedf8414c23bab8d0aec64086f280aa4c` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
