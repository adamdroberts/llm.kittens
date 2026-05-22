# SM120 Round Manifest

- run label: `codex_sm120_promoted_maxconn8_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_maxconn8_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_maxconn8_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `633`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `d3ea81d2a55001dbb784859082cdcb1f43d054e49b6d4fd5667a3f5e097e086c` |
| `test_attention` | `True` | `1800528` | `7847d0972fcbd75b7ff9769f04a8a597b5ef68a23c8079375cab6603fe392c9c` |
| `test_layernorm` | `True` | `1278296` | `eb7b6165db63b6af91a6297a35ab5fd605b32422d497adc5dd699d3ce4545b6a` |
| `test_bias` | `True` | `2089120` | `5ac78b22c082131f288a17fbe50023fe02b6f2c284b0e2828d1237f4420c7212` |
| `test_gelu` | `True` | `1179912` | `e6117734f85803d2367522357dd3e6b88a8857cfc27570eaba338778a16e774f` |
| `test_fused_classifier` | `True` | `1208704` | `254d278b308a317ed0d3df3e0403efaf474ea35e90639531ace071d60d9ac687` |
| `test_encoder` | `True` | `1210168` | `c098f4fe19c61e58372c6d21816c587f83f7ce39b8706f1eba47e3391783c12d` |
| `test_adamw` | `True` | `1183768` | `b48868e0297e4ec97be466d0e4f18e9161e30f45991797e52bbb4606b2942bb8` |
| `test_global_norm` | `True` | `1179464` | `78af2b89718d948846e4e664e0bc216a6672eee9878564ec0ce1f9f7c473a39c` |
| `bench_sm120_matmul` | `True` | `2410304` | `b2e6a12085b756abd340b878e497c349d4351058c6740b3cc4ff5406b92dac35` |
| `bench_sm120_attention` | `True` | `1768800` | `efd63076c6c3d36d01f40c3fe5e3bd54f907e961d47cfc600b00b6b48d677cfb` |
| `bench_sm120_layernorm` | `True` | `1274232` | `0d298b700f1075c54bfdac04c3bd78af42ddd65757483b3ba6e5e69a6b5fb449` |
| `bench_sm120_runtime` | `True` | `2271168` | `83f993951a08336905dd843ec25c015c62b569a9b0525f6e857810f94847c7d3` |
| `train_gpt2cu` | `True` | `3105552` | `da37dd766e57ec27d0cd08414f7d2a27ee20ec2c2ed5077231c517612528b78d` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
