# SM120 Round Manifest

- run label: `codex_sm120_disable_cublas_bwd_runtime_grad_zero_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_disable_cublas_bwd_runtime_grad_zero_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_disable_cublas_bwd_runtime_grad_zero_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `668`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173408` | `51f3ebcdf2d4552710c38f48fd717094fceea5c8240f8f59c507e6f86d537984` |
| `test_attention` | `True` | `1800528` | `899589410ff00dfa4f07a5eef0ac9dbca119d3e853c4453c51b9f77f94f9d45a` |
| `test_layernorm` | `True` | `1278296` | `ab94c86303bc1c2b32517aeed6acc4d25d8a52cc2752da87ba881407e7b6b6e0` |
| `test_bias` | `True` | `2089120` | `5100d6023f1e9b5a7fa9bb8c01ca1b101c403e36c1c154134fbabbd58b98b6fb` |
| `test_gelu` | `True` | `1179912` | `4e71101bcea471297486a157f70c41939fad60134abc05f04e725667a7bf4f97` |
| `test_fused_classifier` | `True` | `1208704` | `dd93bf56d8180e8ee39d698e76dedbfe7dc848dc85e5bfe919283bdb30b85166` |
| `test_encoder` | `True` | `1210168` | `2283d1a7b6adf05cc7d5c73f0f9ac391df556bee7d6aba905b2c2631f593ebeb` |
| `test_adamw` | `True` | `1183768` | `28bad80a831cf76547e9056afdd650afce808338bcf479576256083a54ff90f1` |
| `test_global_norm` | `True` | `1179464` | `849bca61785bc0132098bbe7cc8285cba78755187e38c8186fecc46d6212ee5e` |
| `bench_sm120_matmul` | `True` | `2410304` | `ff42b624e1fe592f591fd4de46a97ff79ac42e80c7c68867948c602050e42bc9` |
| `bench_sm120_attention` | `True` | `1768800` | `72e64ee05f0d2de6dc997d62d9432adba63a39764b1fdef558f726b7499ae2e1` |
| `bench_sm120_layernorm` | `True` | `1274232` | `4d7008e7a79057c3a4685ef75083209bc621e0ca6ed9d8a8c87d298246e96fb7` |
| `bench_sm120_runtime` | `True` | `2271168` | `d03956f0573fe844816d5c72c369465ac1529d3a3ed0161e3cb134bd648aa5eb` |
| `train_gpt2cu` | `True` | `3105200` | `32995287c01180f3f53db5b386c464ce43ffe61ea98db1002afc6532c8924867` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
