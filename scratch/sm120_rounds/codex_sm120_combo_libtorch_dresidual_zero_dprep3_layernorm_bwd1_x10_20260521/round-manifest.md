# SM120 Round Manifest

- run label: `codex_sm120_combo_libtorch_dresidual_zero_dprep3_layernorm_bwd1_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_dprep3_layernorm_bwd1_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_dresidual_zero_dprep3_layernorm_bwd1_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `593`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `0cf2be0caa135cc9df4bfd3183076c5bcfd386270bf4fb76e69b5acba013f311` |
| `test_attention` | `True` | `1800528` | `ae8c475a73374e2325c5f962bb2b668ca81ffec8cb677958105c71b7af809a07` |
| `test_layernorm` | `True` | `1278296` | `8d393649426b121c80d0b2c708bee7633b043f9d8d34a7cce5114bcc28d517c3` |
| `test_bias` | `True` | `2089120` | `0df0d1874be2b214cd614d0ce7a15475a75ad045b48d3681447a07be6ea9585d` |
| `test_gelu` | `True` | `1179912` | `39de5c1ef76dc434859b8cee1124fd511ce5cd6b47d20b7909198085c8763c96` |
| `test_fused_classifier` | `True` | `1208704` | `e4da2da8fe6abcdf9b2b4c1ced68fc09ee4bb007ab101d6f027ab8a29b5461a1` |
| `test_encoder` | `True` | `1210168` | `3d4ba61c1ecf2d02db894b9efe8d0962f330f3c2b28538550a519b01c4888a39` |
| `test_adamw` | `True` | `1183768` | `cbabc68839f4ebd5ae1558eeb2e45d233f25aba8d84b78283747d4164e555773` |
| `test_global_norm` | `True` | `1179464` | `8a19a1f4426c1382e52faff0d55831b570b0394e786de1cd48909c093bb313e8` |
| `bench_sm120_matmul` | `True` | `2410304` | `21b0d58d5d19351d6a5dbdf1cacdeefd6f63aa316eec7cfc99400f50216ed0fd` |
| `bench_sm120_attention` | `True` | `1768800` | `65b3f38c3b86509b3dbc321ad873fb69e365ad8188019c58b29809efe3f1f1de` |
| `bench_sm120_layernorm` | `True` | `1274232` | `0d4270b07faf58238f88a50449091f188ce7cd29319e1a3cae129610aac64705` |
| `bench_sm120_runtime` | `True` | `2271168` | `79039cfeb5dafbd67316ebb480942b2f822a5ee859044ea7585cd551c9ee2190` |
| `train_gpt2cu` | `True` | `3105552` | `13b87de83a1b6bd8930aab835b21ef7aa294f5a3ef5270add8de11b8dfc1c91a` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
