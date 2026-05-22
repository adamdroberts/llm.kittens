# SM120 Round Manifest

- run label: `codex_sm120_promoted_default_same_session_x10_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_default_same_session_x10_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_default_same_session_x10_20260522`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `628`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `9d279b6bdd45f044d7e8ff7cd9836c35ecb815ace0b3d59ef9e292aa240df4e1` |
| `test_attention` | `True` | `1800528` | `7f326c432c4345246f4570f9a753cda63f7f988a9cf68c390932009132bd24b9` |
| `test_layernorm` | `True` | `1278296` | `1196ca9d8caa508ce8474ac70d681bba61f3eb7873fd6c7d4c3c8e6ec29cbc79` |
| `test_bias` | `True` | `2089120` | `c99dca01e38a5bae4a18942782afaae1d6f152c1e6d9f4936f6240f872d324cf` |
| `test_gelu` | `True` | `1179912` | `9ff604b61b0d56186d486ab971298c45f4e496dab1bf41b6b0155a7e99db380c` |
| `test_fused_classifier` | `True` | `1208704` | `099c216fcb22d583a1c03994e5cde0223d892781a3bd4b9aae1614df4a8a1e14` |
| `test_encoder` | `True` | `1210168` | `e4854e72f18e014548b48031ff52cbe7589e61be0205f82ac81c736772d3d668` |
| `test_adamw` | `True` | `1183768` | `06a3eac69de9117007de5b1b7e3c9008fe16007b5358d457ce0aa7a82d5e8788` |
| `test_global_norm` | `True` | `1179464` | `6b67f8ca0b0e26ddac20e0c9fd74708a8d31cefe20d2f33cfe46c3835055bae1` |
| `bench_sm120_matmul` | `True` | `2410304` | `b463e20b294e124b29146a33a59f48f6487d3e66e5f7f683954b0350e588e312` |
| `bench_sm120_attention` | `True` | `1768800` | `f267bb06d83e171647d57ecf7bd7e3b9a92d070dd321d01066f58283adf6ec69` |
| `bench_sm120_layernorm` | `True` | `1274232` | `ea9ef3f53620ea4a924b880e49cc5c520aab19fd4cd9e7914a4d1b5c876c51ec` |
| `bench_sm120_runtime` | `True` | `2271168` | `3e633137f7c1773558b8b06d2fd9d5815c163a75b87f38b2b7da97ab1061593f` |
| `train_gpt2cu` | `True` | `3105552` | `a6ac04586625fa772ee8d052215717d6ddc1c8d05169ab1c4e16e8e33b305694` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
