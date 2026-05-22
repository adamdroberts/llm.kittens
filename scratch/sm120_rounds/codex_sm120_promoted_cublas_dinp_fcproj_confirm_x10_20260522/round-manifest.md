# SM120 Round Manifest

- run label: `codex_sm120_promoted_cublas_dinp_fcproj_confirm_x10_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_fcproj_confirm_x10_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_cublas_dinp_fcproj_confirm_x10_20260522`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `629`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `144347204b28c439ab2c616d6a7daa1f488bf1448f48cf7b4315bf442bd4975a` |
| `test_attention` | `True` | `1800528` | `7e5ca986ebac14268e0b5ff56d38575b162e6bb376847629b8c9763c55d04b55` |
| `test_layernorm` | `True` | `1278296` | `818bd1e8ba6359459a13c4ffceff9f4af256c542311d12d7f0a4db32d8da573d` |
| `test_bias` | `True` | `2089120` | `313a34a0e8ef89e97acecfd711ee56b5eef64b0adf546254faccb3569a63e0d3` |
| `test_gelu` | `True` | `1179912` | `d2bff0e2d5c2f94ae417e0ea9235ccfc886a04005eb5fcb5c7c5ff6254f16128` |
| `test_fused_classifier` | `True` | `1208704` | `791d20978113453328755892cda6b19cffa0df883facfaa2b483e06113a3099a` |
| `test_encoder` | `True` | `1210168` | `57d3715a3cf3d4113e2d58e24092426c3c08658897d9b15b7f8a93b8a77bd353` |
| `test_adamw` | `True` | `1183768` | `7a3277222611250aa4f3039658f3cd8c3fdf827c084c8446bfb2d893feb2c0c5` |
| `test_global_norm` | `True` | `1179464` | `79f6ef31e388ff1c12821f30156fdf13cce0f72b8f31db1b64e2f9229d506176` |
| `bench_sm120_matmul` | `True` | `2410304` | `cd525d1dfe72f5227d88b922951205602b828ca6f71727202ee1da5eed709bf1` |
| `bench_sm120_attention` | `True` | `1768800` | `9fba0892a3db8b336112bd34ec83f5ad99a19abb8054ef94a8cdf6045b967015` |
| `bench_sm120_layernorm` | `True` | `1274232` | `af2083fdba320acafb3240223004d6b1e145d3689e10af16d5de541e9dc64eed` |
| `bench_sm120_runtime` | `True` | `2271168` | `c7447f8396f694548a2bd270b7883e3aedcf3af8cbf67c331695569b7096f5fb` |
| `train_gpt2cu` | `True` | `3105552` | `6be5f2aded0df76a7083c5ba918c05bb31e444b1310caa19a677ee4e9328abe7` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
