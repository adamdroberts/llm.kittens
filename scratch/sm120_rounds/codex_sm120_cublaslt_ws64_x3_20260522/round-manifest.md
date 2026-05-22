# SM120 Round Manifest

- run label: `codex_sm120_cublaslt_ws64_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_cublaslt_ws64_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_cublaslt_ws64_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `680`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `8e2091a9852e38fe203c67a56cd5850b77b040484c8c847732ffde38dacd1ebb` |
| `test_attention` | `True` | `1800528` | `298e05bbe4119a104e5dc3df23408bcdad31b7e1295afd8c2870da4b43304d5d` |
| `test_layernorm` | `True` | `1278296` | `900117dd03d8b132552bd5dd28a1058566a9162fbbac70f0a7eed76c5a4e0c45` |
| `test_bias` | `True` | `2089120` | `bc8590958eb3fa340ca6081097a384b4dd0d83797fc50c2c30ff774b62f8e223` |
| `test_gelu` | `True` | `1179912` | `2838d2ef25f7ca62138813db64633ab6e78c720a075309a7d8bfb7a1f8392909` |
| `test_fused_classifier` | `True` | `1208704` | `8c929f15e66da22e1110847f3a61433a3023d4371591dd965b94155ccdb56c60` |
| `test_encoder` | `True` | `1210168` | `3ca2a49e89e4b2eb999cf4bb7e2e6688aac1f2312b5ddb1d2bf6654a8a3bd1b4` |
| `test_adamw` | `True` | `1183768` | `cf4aef3404cf1fd13bee565dee7e5e686c3a98d03fdd9e1cd300f972633565b8` |
| `test_global_norm` | `True` | `1179464` | `c5a74016ff6e677318c8b0d8fbe057e36dd22d5e3ec244067d37d22bd629a1e7` |
| `bench_sm120_matmul` | `True` | `2410304` | `04164b41418ec6b561f5da89a5bbcf4e2ad715ef97de5d1630e6511a4a3f951b` |
| `bench_sm120_attention` | `True` | `1768800` | `f8398a9ef93850d454ec2821e4995ac01e2afb1ef8e6802fa18b3b65299a86b1` |
| `bench_sm120_layernorm` | `True` | `1274232` | `f0e65cd39b434c24cc169ab801afda930fa0c49e552a1102b6c96df25a4f77d6` |
| `bench_sm120_runtime` | `True` | `2271168` | `10f7eb8f598b574077d353d8f04dff0027e998f09b5c11d9c9aef913efb9ad82` |
| `train_gpt2cu` | `True` | `3105552` | `733f6c217b9b113e57f7871d0ad58bfc1299e54cf462e8702d996af8b9ae3f23` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
