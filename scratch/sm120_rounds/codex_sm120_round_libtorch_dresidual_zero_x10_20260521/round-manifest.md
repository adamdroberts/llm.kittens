# SM120 Round Manifest

- run label: `codex_sm120_round_libtorch_dresidual_zero_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_libtorch_dresidual_zero_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_libtorch_dresidual_zero_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `504`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `09181bc45e64d216b08b470c40c4b6e7041a9954f3d38218bfd57b55282c7b65` |
| `test_attention` | `True` | `1800528` | `415f46422d71b22380c3c3b6f173a7a91c64d6d185d526b837e1f30931045b0b` |
| `test_layernorm` | `True` | `1278296` | `df8f63817b163758951961e06c33e3bdecc2953dd0418c1794dcebc8200493b8` |
| `test_bias` | `True` | `2089120` | `d1919e3d9b6d5549e049b78c90b4931c9952e758ef2ba7cae91471980974fbbe` |
| `test_gelu` | `True` | `1179912` | `2332be5e7bb37ad28c06d1cb28b8b11cb02f71c31db48f629a0487985f2e05b5` |
| `test_fused_classifier` | `True` | `1208704` | `1fa29ef088adb5b5e9a83c36aa2a920ae54a24d4ae4d3fe2ffaf7ef8b5b953c8` |
| `test_encoder` | `True` | `1210168` | `7151dfdf24e5074dafc23250168c6ebe2b8d2d632a4d0d63d01cbcf18ff48d34` |
| `test_adamw` | `True` | `1178984` | `29c471a92bab5b6f944a1b439c3e44409f11dd3f922c30556f3f043966ef690e` |
| `test_global_norm` | `True` | `1179464` | `70cc8216dcccd9e8f9aa9ecc16f5ccf596388875cd9eb9987456d648460075cf` |
| `bench_sm120_matmul` | `True` | `2410304` | `4c931de6d235aed9a533283235cec7bf80fe7c1efac314137e653eccc4122883` |
| `bench_sm120_attention` | `True` | `1768800` | `1c0b7255d6b15d80c4ef307053333b528679ef7933e9fd70f3acf834f1aaeecc` |
| `bench_sm120_layernorm` | `True` | `1274232` | `0e7dafc0358889eac8d3c54a5e53eb4b3dca1aefa63b2a5d58ea918cdcd7961c` |
| `bench_sm120_runtime` | `True` | `2270544` | `0ab39e99e8d3ef11d606515f93032a11c720dd764e3f9e760a489325b3c9b708` |
| `train_gpt2cu` | `True` | `3100592` | `46de3f49376e38ec0b394a4e2ba28ac9979b6540d630d4495607921b033d00f0` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
