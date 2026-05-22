# SM120 Round Manifest

- run label: `codex_sm120_restore_promoted_default_after_precompute_recovered_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_precompute_recovered_20260522`
- train output dir: `log124M/5090_S_codex_sm120_restore_promoted_default_after_precompute_recovered_20260522`
- device arch: `SM120`
- max steps: `0`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `651`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `e5e0d9c113efe774553a09f9df1d4b742ffcf0899a7665a8ff0f0e1b741d1033` |
| `test_attention` | `True` | `1800528` | `7a9b112e515e6e5990b95549138e9c24173598dc489abfe5dfc11dc8ce8a8d64` |
| `test_layernorm` | `True` | `1278296` | `3328a72b35e27759455a9220d54dfb37644839f447f096c1a863e0dbf5d6e0ec` |
| `test_bias` | `True` | `2089120` | `9a96608d5e4f999ee1ef4b0ff713c8920431bcc81bbde03fdf6b99226fcc8030` |
| `test_gelu` | `True` | `1179912` | `7a84394400b06d6ae1b1804c3d597b779f6bf0ce6e67868c5a9a8c71ce546099` |
| `test_fused_classifier` | `True` | `1208704` | `4a071a4dcd3e4140ad96bc27c1fcd418baedf8705837812355d016c82649bc37` |
| `test_encoder` | `True` | `1210168` | `5cb14fd5572cf438a05480b63cc2f9c1386fcf17e6e887526ec8af18bd0851da` |
| `test_adamw` | `True` | `1183768` | `b104d6aafe2a68fe678abccfe250c7613d7fa2de6d12637b9b272c1e17750294` |
| `test_global_norm` | `True` | `1179464` | `a689b2e163ad45a1eadc1f7c318e0cb6d9363c5d3348d5a7052e255ebe8ab147` |
| `bench_sm120_matmul` | `True` | `2410304` | `769063bd4f29adc2b9ab676eed1ecc82fc4a08ce0076183a2dcc8c65d7aed28a` |
| `bench_sm120_attention` | `True` | `1768800` | `6bc6a86077c00fcaa5e9716c3be0ecf397aec43beba4d9e7dfa45d162f051690` |
| `bench_sm120_layernorm` | `True` | `1274232` | `52826089bcd7fffded041276c3d918167e4035f05d0dffcb9d1d16f7cd3db273` |
| `bench_sm120_runtime` | `True` | `2271168` | `1816e6876a15f9722b744a1e42e9d150a0062250b6b55a0a73c3e2064c61742e` |
| `train_gpt2cu` | `True` | `3105552` | `4f4ea367835deb19e415eb36db246873c392091edc73dd2050717de047148f61` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
