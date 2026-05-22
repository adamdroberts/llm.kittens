# SM120 Round Manifest

- run label: `codex_sm120_attention_dprep3_current_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_attention_dprep3_current_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_attention_dprep3_current_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `579`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `4997e4eb80899d4f83b833a047eed8b5e895d2b97a697b20d5ff767f4e6e748a` |
| `test_attention` | `True` | `1764128` | `248751fd052c63d641fbdc9766af7a76bf93fb941d5a93402c178940bcb0088f` |
| `test_layernorm` | `True` | `1237784` | `f07dd0f73db0375cbfcd9301ef9dfa44f0a9ab89733a611607562ebbaaea393e` |
| `test_bias` | `True` | `2048616` | `3d695357d4e952f9779b6f1de68990a86acd90b91681c83d7198c64b6e1e1e4d` |
| `test_gelu` | `True` | `1139336` | `f2e6f7e9d726403de74e62a6c04b148b889f5f222875984d8686788a81a8f4cf` |
| `test_fused_classifier` | `True` | `1164032` | `63359bd276d72ccfb57c5d9b1dfeca72b1d01323e98060dccd514d9a8d80632f` |
| `test_encoder` | `True` | `1165512` | `2d89bd7ed35f3e3b0e665e7e4e343ab42ec5b762d559ebe8f3c9660883974e78` |
| `test_adamw` | `True` | `1143192` | `d30864f90750909e22f92dd909b1fbdf3b8fc4a48e05b3f8f5ff7f43d2a0c779` |
| `test_global_norm` | `True` | `1138880` | `01f1d9900c5a3388ade94e10fbbb50be31629b8d9d2288e90a0f5c4143c1166c` |
| `bench_sm120_matmul` | `True` | `2373912` | `23f86136b0d812231c85726c6ba8ab982f3425877d150181abeb0cecf57f86da` |
| `bench_sm120_attention` | `True` | `1728312` | `e3fdf5d643e6fcf58416d71c65f008aca8e2dada28896db8a7b31b01f6478acb` |
| `bench_sm120_layernorm` | `True` | `1233728` | `47b84b3f2150b8e78fda3f81d2ec3acd07194ddecdb3a76f9a0df7ebf3258f7e` |
| `bench_sm120_runtime` | `True` | `2226576` | `2d1fa6cc878b2c62e1cf4c3bd73aea2aeee9d64003638cfd95a681102deee47f` |
| `train_gpt2cu` | `True` | `3064992` | `6ec1cd3556a23c442d7fcb2c2dec6504e9809eba2f75b1596523c768c307b52d` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
