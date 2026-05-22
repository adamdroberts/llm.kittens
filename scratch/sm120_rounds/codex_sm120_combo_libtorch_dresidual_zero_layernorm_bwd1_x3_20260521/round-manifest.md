# SM120 Round Manifest

- run label: `codex_sm120_combo_libtorch_dresidual_zero_layernorm_bwd1_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_layernorm_bwd1_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_dresidual_zero_layernorm_bwd1_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `538`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `0ab8e1ed91880ce7ce74b44ba88490d72beb2432566b5cdf4124e59bfc58bee5` |
| `test_attention` | `True` | `1800528` | `cccb7226ad2142d1d0cf9e60479c288626db8eff5b0b1150736b516321930d10` |
| `test_layernorm` | `True` | `1278296` | `83011bb161f8cae8d83fc6ba6e66566db7af0b3226ed9332c0dc13481ce8cdd9` |
| `test_bias` | `True` | `2089120` | `7c972e56ca51f5c9bc4ffc6a6520dc8dcfc1c31a3edc84d4aa5c0cec65b8db36` |
| `test_gelu` | `True` | `1179912` | `fa996329f712866ff1f9e62a29bebdd2e333a7b48d1f3f0610a9a446b3ac6b26` |
| `test_fused_classifier` | `True` | `1208704` | `083ae66441626ced2fcee10b16a4b31e190b3fe5ac5f77994f72e0220958e573` |
| `test_encoder` | `True` | `1210168` | `f5f1b31ddf3d6cfec48f04bb37b6b9d6959c36930b1f34aaf067b0e447036b7f` |
| `test_adamw` | `True` | `1178984` | `7368a1118be842dfedf81525eb9c1614d28eb4006b2a5ce08ddda8010129d018` |
| `test_global_norm` | `True` | `1179464` | `2d50cca243028414ac4015a8517e6a45725d579f68cfb6fc47dcca6fe841852a` |
| `bench_sm120_matmul` | `True` | `2410304` | `cc2c5793e3764ebc0e8fa87784ea0bb01a68fafc742f54742783ccf85391599f` |
| `bench_sm120_attention` | `True` | `1768800` | `de76ed0042668e53852840f39bcf8a3b0819ce8fc90772041ce2e1ce9aaee207` |
| `bench_sm120_layernorm` | `True` | `1274232` | `c25ea4a2b8c6086e3103ef7a03f9fd822a1622f9a5bc86bfa5833677f6d2620f` |
| `bench_sm120_runtime` | `True` | `2270544` | `80ce7f830cd0330399c7f389d0109d1ae4cadffbf59908f0337d7ec828adcc42` |
| `train_gpt2cu` | `True` | `3100592` | `ab6af35b640b72c4111bc244bddcceb76c85bc0fa144d1075ef63e70fbb5cf17` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
