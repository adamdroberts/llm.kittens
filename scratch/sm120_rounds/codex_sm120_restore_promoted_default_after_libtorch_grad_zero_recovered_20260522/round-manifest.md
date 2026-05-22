# SM120 Round Manifest

- run label: `codex_sm120_restore_promoted_default_after_libtorch_grad_zero_recovered_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_libtorch_grad_zero_recovered_20260522`
- train output dir: `log124M/5090_S_codex_sm120_restore_promoted_default_after_libtorch_grad_zero_recovered_20260522`
- device arch: `SM120`
- max steps: `0`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `654`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `da16d172e6cceae3ca8ed49327646e58319acdfc31c5cd733e9836b9bbc42d99` |
| `test_attention` | `True` | `1800528` | `420edadf8ff4db1c6274f3cc12d84cb3ff2e59d584230e7d357d289d29f6871f` |
| `test_layernorm` | `True` | `1278296` | `cc9a498ec1ca2d96450adf111f1e69a9cc1f6cb5e0693be9031b4ca656ccb132` |
| `test_bias` | `True` | `2089120` | `6cf7922682d4c512cd08a97c3bdd99b11d52d8a6cb1be2a01a5d19a279687a1c` |
| `test_gelu` | `True` | `1179912` | `2cf577928e1b8eadd682ea415c2b71dc99228522fd4261351911bb9d9b03365f` |
| `test_fused_classifier` | `True` | `1208704` | `bfc49956630bcc40678173337346b9990653e30d9028e9bfab0e1e4a1a478d99` |
| `test_encoder` | `True` | `1210168` | `61d3fe29a72b55d52d0e0dce6e7bff2fa7d20c74217d47ce985a6b9ed38f8d74` |
| `test_adamw` | `True` | `1183768` | `58f0bd4529c0d19ed9836baf6b9f6199347b4700651e4b97f6fee58898529e7d` |
| `test_global_norm` | `True` | `1179464` | `d848c2d163e1bc62cc7cb75c2342a2466e33e8967da7f7f147b768f4d811b292` |
| `bench_sm120_matmul` | `True` | `2410304` | `aa5b42df1a02c9caff13e40594663bfd4ce7edc788a2a35537658ef9e8a06386` |
| `bench_sm120_attention` | `True` | `1768800` | `be2ca96a9e2c750fe3c1db5512a947594a91a1e69b08362c45094f1dfbda523f` |
| `bench_sm120_layernorm` | `True` | `1274232` | `9df3574dd92985c1fb303645c47b14062762ab4533563e2dbcdbb8261a713e6b` |
| `bench_sm120_runtime` | `True` | `2271168` | `b79dd14418a80b95f4af7271c6ee007363f06419ddf84fa887b931414b71b413` |
| `train_gpt2cu` | `True` | `3105552` | `4a383d26106f39decf9952eb1283b2bf7e22f5088bcbcdf63cc36cfee7b35c81` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
