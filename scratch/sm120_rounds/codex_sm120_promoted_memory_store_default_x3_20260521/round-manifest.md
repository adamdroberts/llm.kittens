# SM120 Round Manifest

- run label: `codex_sm120_promoted_memory_store_default_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_memory_store_default_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_promoted_memory_store_default_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `621`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `ce59974396f51ba439e8ab2855867eae9d5fc9578aaf5d74893aeb4f9308e836` |
| `test_attention` | `True` | `1800528` | `62185a9a9182f4fec323b0e608b7461fde0c08cc7790b7cc5615df855a497509` |
| `test_layernorm` | `True` | `1278296` | `463e722309d5fa4e70f393eed7b5ac9bf3fcfe90c5e61b262dc3858522545392` |
| `test_bias` | `True` | `2089120` | `90ae61cd02d48c792e907fdad663639ca01c6c371f56ee56308afeb8642e568f` |
| `test_gelu` | `True` | `1179912` | `673ebae9a74e7522d135eed1f9649a959458b4602da832adf3047b32ae208a6f` |
| `test_fused_classifier` | `True` | `1208704` | `ac3627131386e48be21f42fecde77fc20116cf52363fc153b3587310352b3101` |
| `test_encoder` | `True` | `1210168` | `1f760ca43696ae4639590369a4ccded241e3046f5d17cb9d4b852d828ae18c26` |
| `test_adamw` | `True` | `1183768` | `7e05b50eaa4318f94ec6e4910a8b9271024889d4d53e4d1172ea2bb036cb5109` |
| `test_global_norm` | `True` | `1179464` | `2498f64eb9e293b876d2af2b6f1e6ad06092471e47cc1f8ef895fe3574fc5246` |
| `bench_sm120_matmul` | `True` | `2410304` | `6102c4a43d53463634a9f0ebbd34217fb79ba01d8ec63a46a6bc24ba72938555` |
| `bench_sm120_attention` | `True` | `1768800` | `b3a099e2497550a7301a43fcf18ce02b24558c321182419163c403b4e7e07708` |
| `bench_sm120_layernorm` | `True` | `1274232` | `6c45cf4f4c93d7352b78979d84b72e1715f7cb7d1fa6c5d55034a72d8ab0dcf2` |
| `bench_sm120_runtime` | `True` | `2271168` | `0ad6d16b485cbaa44a42e8c1208ff2c70102b586027b5173a8b817ec4db5f1d6` |
| `train_gpt2cu` | `True` | `3105552` | `961308d7118b107c34424009c5a673e590edfb1f1f9e03e4f445824522e38e9e` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
