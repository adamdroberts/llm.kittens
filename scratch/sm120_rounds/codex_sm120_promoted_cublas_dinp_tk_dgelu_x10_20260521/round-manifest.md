# SM120 Round Manifest

- run label: `codex_sm120_promoted_cublas_dinp_tk_dgelu_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_tk_dgelu_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_promoted_cublas_dinp_tk_dgelu_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `619`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2190792` | `8572e924aea7e1a0152314e7e5adb95138fd439dc4025f6ba2239fa735800cca` |
| `test_attention` | `True` | `1800528` | `36a00fcfb1bf098203dc2590d2e5eb9468d88984e055b9a7f72fa3d968c8c560` |
| `test_layernorm` | `True` | `1278296` | `7e47be5499205099a056e4fcb6b6cf90ed0e803396e326c0d537652766f25b7b` |
| `test_bias` | `True` | `2052256` | `d7ed25547873a812af4f52e6c8cc56fd8283bbbc955fc622210764ce6bb89376` |
| `test_gelu` | `True` | `1179912` | `928656fa5ab8b1c78e9a7f96f39737785f3347e8082191887c0d90a2fe7c2374` |
| `test_fused_classifier` | `True` | `1208704` | `a8962ad3625542c6933a64aa03cab69bf7e1070856ca5ca4abcfa92d2873e997` |
| `test_encoder` | `True` | `1210168` | `91aeb72e7f040ddff0e4b924fbe1a929758211658b899687cb72043d3ecc4ac5` |
| `test_adamw` | `True` | `1183768` | `d23f3678aa96add4a95b732648d5b357d2a871633c77dc083acd3b2e81324206` |
| `test_global_norm` | `True` | `1179464` | `0e9fa1193744c317a7a77bf2e25277a134269bdb7cfc8e5835593778955cedc4` |
| `bench_sm120_matmul` | `True` | `2369344` | `6fafc7b449fc01c2e43f98b898d378a420cef7a79c8e39e294ee527cdef991f9` |
| `bench_sm120_attention` | `True` | `1768800` | `493dc1804154463b2628516ad3d15bc008c272e3d09d73ea4d5aef3fe47d8ac7` |
| `bench_sm120_layernorm` | `True` | `1274232` | `82e1f7374986f34cba5eeb1bd422f14e2f30bf65c027a0c81ec44f662fee7c0c` |
| `bench_sm120_runtime` | `True` | `2234304` | `516cd2cd8169155530d97a49b681558bb194933e63b6e1c9da4ac5549e1447b8` |
| `train_gpt2cu` | `True` | `3107416` | `986e160c52d762ac91b720f9492bd2e9a9b59c89d9626b38479ff69fb9cd0062` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
