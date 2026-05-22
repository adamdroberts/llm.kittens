# SM120 Round Manifest

- run label: `codex_sm120_promoted_maxconn16_x10_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_maxconn16_x10_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_maxconn16_x10_20260522`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `636`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `4e44995b81fcfb828ec19a652d875807cb39d1190ec627c534ddfa8a1786baa7` |
| `test_attention` | `True` | `1800528` | `fefb031cf70d8fbf784223b27c4ca2f8447231a7bee621153f1b3f2bfb04f527` |
| `test_layernorm` | `True` | `1278296` | `a03fcb5695019b7cc559607b68d7f2b915a12d90260b0b2e42b6a6c931f975ab` |
| `test_bias` | `True` | `2089120` | `bdc18087f937d5585eb95fd1f5560a6b53930816c904eebad6ada6b3984d052d` |
| `test_gelu` | `True` | `1179912` | `ea230dc296fea16ea353f83c0c70e7239cae1515b6fc273a6fde7325dab1a35c` |
| `test_fused_classifier` | `True` | `1208704` | `6cdca6779e72c6017dc67b638a5bd5c8432ec5d5799a99489106c2577329b0fc` |
| `test_encoder` | `True` | `1210168` | `c2164642c91374cba3f3b19884f9ffd7d5c011bcb10df4894c4fa1b84e38f4c3` |
| `test_adamw` | `True` | `1183768` | `4d4cffe96ae8b4ef318433f38b86a3f8a92787503a6833ff599970446382ae90` |
| `test_global_norm` | `True` | `1179464` | `b2607098e9ce8db4adb578249cddf6f277270093dabc5af60b82af3216cd87c9` |
| `bench_sm120_matmul` | `True` | `2410304` | `39cbc1d87e77193bdbcc089fb1e801818144883629aa9c2fdcdde4398de35153` |
| `bench_sm120_attention` | `True` | `1768800` | `6f2676381fea12bef3fd355c271acc23c405d04cde6b861442ff3a897f17c6a0` |
| `bench_sm120_layernorm` | `True` | `1274232` | `81bd64b462c1f3dd1a3f7fd8c258aaa8d0e0f15dd81db5f0ae46b54cf85673e6` |
| `bench_sm120_runtime` | `True` | `2271168` | `bf9f097d4ec56e4fe86801a51eae6a7dccca3d9d39f90c14e1db2275fc7fd35e` |
| `train_gpt2cu` | `True` | `3105552` | `f6e6aa878a03a8a9f23c52aa3e2f3c63408ab881f9a73a00b7e4223e1f7680e0` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
