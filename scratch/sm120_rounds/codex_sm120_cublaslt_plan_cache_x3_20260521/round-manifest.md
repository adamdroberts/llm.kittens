# SM120 Round Manifest

- run label: `codex_sm120_cublaslt_plan_cache_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_cublaslt_plan_cache_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_cublaslt_plan_cache_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `531`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133192` | `05a7f378c79eefefbd5b8fdc450b316a3cd5b5628445c8f57d049c5c01d41122` |
| `test_attention` | `True` | `1760032` | `6d8c61baf9d67b75e7881e5b41cf59d5904b6b826b9b894c0cd5a7def32952dd` |
| `test_layernorm` | `True` | `1237784` | `fe03f308a9635e4597b6a2b37d8593acf4a281c336a1bd24a7401ae156e640c0` |
| `test_bias` | `True` | `2048616` | `068ad860f80efc51045885ec10ba853e14dbffd5a02a55f3e2fe38ae30eb8cae` |
| `test_gelu` | `True` | `1139336` | `5a2c99872d5a00a033264a192f46ea56c303e31c7fc6adac752c3ec0a05de500` |
| `test_fused_classifier` | `True` | `1164032` | `0523dd8e5f758e1e16169b8ce5447bb94b19024d00b47971afd2dbda14a2e01f` |
| `test_encoder` | `True` | `1165512` | `a60b847a03408f6a11c4ae19ba662896692d76da61aa15b9716629ce73e9fbe8` |
| `test_adamw` | `True` | `1138408` | `184e646a70cb8347e16c86ef215c81e000410d49231a3f5349a099fb02b34975` |
| `test_global_norm` | `True` | `1138880` | `9edb97874cb82bbf71b1e1ce14a28a3100c18d80cbb101ef73cbefb0a77f5cfb` |
| `bench_sm120_matmul` | `True` | `2373776` | `729a888937d59971f06cba52e7714bad600a9f8c565ec0900c973ed1fe197f7c` |
| `bench_sm120_attention` | `True` | `1728312` | `bf454a45ec9e34c9a66fb7586a1dc66fde954209f48cb8bdc3e78871f26febf6` |
| `bench_sm120_layernorm` | `True` | `1233728` | `2e11523f1f73386fa5bba294227b7a9cdad546c948e5cd70f3956d522ec40ab2` |
| `bench_sm120_runtime` | `True` | `2221864` | `17fc6fdf4b3d3aa4f96c38f4b0da130135396f718255ebada6045c941e8de3dc` |
| `train_gpt2cu` | `True` | `3060176` | `1de5b597b2588bc013f8116dcde1c705cd1af62467d0851a7d788c483802cc7a` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
