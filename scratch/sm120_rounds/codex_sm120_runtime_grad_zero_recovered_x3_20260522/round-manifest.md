# SM120 Round Manifest

- run label: `codex_sm120_runtime_grad_zero_recovered_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_recovered_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_runtime_grad_zero_recovered_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `657`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `03a13a7ccc11bb69b6855db5c782b74a4c105ac1c1cc53f98cb7009fd9cc8a3a` |
| `test_attention` | `True` | `1800528` | `6413aa2b9debe5675308beb3596926ee63c45e40d0423c69ae4c49ba4efb913d` |
| `test_layernorm` | `True` | `1278296` | `392400596bdc9561cad0bdb1bb8f02120e7038dabd8b520c3e6b466d54095ed4` |
| `test_bias` | `True` | `2089120` | `39ac6d68b2830feb572a117a0d5013a0dd7f6e06fa28ee57992f2082bdd7b351` |
| `test_gelu` | `True` | `1179912` | `238861c14a5a5c7f466ebe07e1206d34b81124f35843063056d708a9328afd8d` |
| `test_fused_classifier` | `True` | `1208704` | `3fc126a01b3a90b383adb911c8168a6a6112e9d45eae1579dd0022a769ad6bff` |
| `test_encoder` | `True` | `1210168` | `3eb728532c907ef4607c0b790fecdc95abe4275497aaf2605ee795f824c6dc0a` |
| `test_adamw` | `True` | `1183768` | `a86bb8e0b0b56f533fe49bfb34bf1a363432c3d69d2f00af80766b6636c2d8ec` |
| `test_global_norm` | `True` | `1179464` | `2efd166a9a3ab2d4bd1fbb99d1dbf30f0dcd1a288b314c3f3013667d182c5dc3` |
| `bench_sm120_matmul` | `True` | `2410304` | `6b82f78643d566db1abd2c2f6359d7c8ca6447cbf699c5c43a1de1f179ffeeaa` |
| `bench_sm120_attention` | `True` | `1768800` | `2c98237c9d1a590111a20f7e217650787a20f4f4770b5398ded8d8d35d832360` |
| `bench_sm120_layernorm` | `True` | `1274232` | `ff50b9063263da08d12a0263f4d3b1a7c227b9bada3e5de0ad4b6095582907e7` |
| `bench_sm120_runtime` | `True` | `2271168` | `97c23267e77575b3462e88a189c42fbbdba233c3f997e2b5561d8842647430b9` |
| `train_gpt2cu` | `True` | `3105552` | `37d4f87a862312bdb5cd3d86e464a506219a088f89fc3bae3f4ddd239c59ace8` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
