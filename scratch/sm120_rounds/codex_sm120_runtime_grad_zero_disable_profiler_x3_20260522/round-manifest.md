# SM120 Round Manifest

- run label: `codex_sm120_runtime_grad_zero_disable_profiler_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_disable_profiler_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_runtime_grad_zero_disable_profiler_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `675`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `7f766d15d751312e14095e8a809e5d266ee6feda38d8809a3cf18767b6de6814` |
| `test_attention` | `True` | `1800528` | `828c99dacbb29c64fe369688e4106dffdf9605ccd37663ab70b4c4874cb318db` |
| `test_layernorm` | `True` | `1278296` | `b2a7c899cac95e503c80dcfb4e245f80c0c9dc1025d93c8260fa9f2930f6988e` |
| `test_bias` | `True` | `2089120` | `75a975cb903dc620ed9169d313f62f6d122981e0d708ac6be316ea7abc0227b4` |
| `test_gelu` | `True` | `1179912` | `0e6c37d5af088b3ef97b352d8e4fe7ffcd28605dfce9ec2dbee86eb813ec1fea` |
| `test_fused_classifier` | `True` | `1208704` | `0b9d17d237f096640dace328ea410a59f9651506861dfd45bdae60bded002d84` |
| `test_encoder` | `True` | `1210168` | `1709c289185a3b59dda733f2069334323397cd5f3f6306cfdd9d6ed78b774e9b` |
| `test_adamw` | `True` | `1183768` | `60faa0ca750f169a9ccb8ba50cc9d341c7db6d2f3d0ddd454b0bfd2d10bad9a2` |
| `test_global_norm` | `True` | `1179464` | `45d35fda942522c1df2ac6affd493cfe6ba54cc5be5c463689c5876c76f666c6` |
| `bench_sm120_matmul` | `True` | `2410304` | `8e26974751887a1390c42a13e784a4dff6b8c016ae559a939dab7bc34d7e3748` |
| `bench_sm120_attention` | `True` | `1768800` | `18a45beb58da1857d7ffb4b13445099d38376c429c2cf233d4ae4c56fba953a6` |
| `bench_sm120_layernorm` | `True` | `1274232` | `e661766db0aa2a5002951a571ce7f972777a65a406b874360260303083c26244` |
| `bench_sm120_runtime` | `True` | `2271168` | `d05e197df2cae50655cc5f7d9c770ac91cb890cf824aacef4ce1d29cda10fcb3` |
| `train_gpt2cu` | `True` | `3105552` | `6a90766385fdbc0f58631538b0d98be858c806d2fb5e6667af4f177deb695a32` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
