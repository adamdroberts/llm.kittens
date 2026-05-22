# SM120 Round Manifest

- run label: `codex_sm120_promoted_disable_cublas_bwd_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_disable_cublas_bwd_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_disable_cublas_bwd_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `642`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173408` | `d129d7b3d230923395a9477aca023e44e8306ec9d36c9949b076167a698f8d1c` |
| `test_attention` | `True` | `1800528` | `2f32717390b9d9beffe03ad10666b9b32e59138fcc66cdaf03cce07c415b5866` |
| `test_layernorm` | `True` | `1278296` | `66fd8792d0a0c6701089de22d40375827a57e5d61ea011db136fc342f64767e1` |
| `test_bias` | `True` | `2089120` | `56ea220b9b0f5dcd8f5373c60c999b2095449da4e4acb07598fecf64454378fd` |
| `test_gelu` | `True` | `1179912` | `2e2658bf3b5c33a2de39cfdb7fc513ac8cd68cfeca43552425e26cc4280e661c` |
| `test_fused_classifier` | `True` | `1208704` | `ea46689e6ce7c8759b533d4968816ff846a5ff08e12b22f03151d9498174b07a` |
| `test_encoder` | `True` | `1210168` | `fff5e32b0d0f96cb60e1ed18fc664e8b1e1802eb9b7b7cfee1cf1c17adf3b296` |
| `test_adamw` | `True` | `1183768` | `0199758714079adab9ad93774c58b3d437254c9c16b88f2d8b9be433fcc4a636` |
| `test_global_norm` | `True` | `1179464` | `1416ba3a82caab5d577f3d857b5bce52626d10d0f0f2efbb0a25991bc4ac7247` |
| `bench_sm120_matmul` | `True` | `2410304` | `46ac2f6220881a1b3f0af76791e7c1bb2477579c0cb640f6716907c9db3f8222` |
| `bench_sm120_attention` | `True` | `1768800` | `0a63394bb8e9a3caa23bccd2198716bfb840f5765017a837a31ca8c868f81805` |
| `bench_sm120_layernorm` | `True` | `1274232` | `bf383f530e12cc5e71af96519a6ee1e56833784fd576c8341b7970cb205c5e42` |
| `bench_sm120_runtime` | `True` | `2271168` | `041e300c025666c465284e30228be24dc77ef6782984c8677b64dc27222c7b7d` |
| `train_gpt2cu` | `True` | `3105256` | `c4422f57aca3ef214bc43d839f15150bb0b06a6c2150f5a347958b430e55cfdf` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
