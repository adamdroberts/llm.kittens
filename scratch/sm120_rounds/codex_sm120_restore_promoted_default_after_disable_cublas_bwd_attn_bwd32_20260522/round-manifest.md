# SM120 Round Manifest

- run label: `codex_sm120_restore_promoted_default_after_disable_cublas_bwd_attn_bwd32_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_disable_cublas_bwd_attn_bwd32_20260522`
- train output dir: `log124M/5090_S_codex_sm120_restore_promoted_default_after_disable_cublas_bwd_attn_bwd32_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `645`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `d481b5b2c89e1c5f5a9340d75bd4de3f4bb4f4ebcb111a7cd9da4a68d7643a06` |
| `test_attention` | `True` | `1800528` | `38961d0cc90cd835e0aa03daed739878a11f54b6f2b2d935b245d14016ff71df` |
| `test_layernorm` | `True` | `1278296` | `36a7017b7ef7bec2187939f7fa10c7aca1db7fc5ea867da2e3337a19be4346e9` |
| `test_bias` | `True` | `2089120` | `6eb76e420c7269eea35ca597c7f78d409124aa734d7e606f6c81b36e2e2ca7a4` |
| `test_gelu` | `True` | `1179912` | `9cca01c9d1724003081da6fc3c178f9f9d18030b50a6a753a69561a92b822f5e` |
| `test_fused_classifier` | `True` | `1208704` | `866509fcd9b90c177af263f5c7c6ad88abc9cf79d7e5cd3c93f1b274d7936b7d` |
| `test_encoder` | `True` | `1210168` | `2354172439e563dcba23cb1f887b03ab2e2a161b7327c3532f48b484f83f3c5a` |
| `test_adamw` | `True` | `1183768` | `cd2fc1edb2c2fdc2ecb1b213f2e556f971d11600dd2a1a17ae13e7bbb52bbb15` |
| `test_global_norm` | `True` | `1179464` | `dc915d6d9bae4d3c6c4db3eeb775784285d2ee9a84f70f8ace5727096ede43bc` |
| `bench_sm120_matmul` | `True` | `2410304` | `54e40e08911dc4bee8f01172b5c91912c9c297f173d8771bc81c8cc09c18c185` |
| `bench_sm120_attention` | `True` | `1768800` | `36f47cb544eb4a2c1662af3338fcd275dbea1b0b059b5b76d7e91885814c864b` |
| `bench_sm120_layernorm` | `True` | `1274232` | `52e29ea14e6a2ec7598cc6dd29c04de54d6d6c72107ab0912e06b6f1f1cc5866` |
| `bench_sm120_runtime` | `True` | `2271168` | `f047e2c7d3c8f14d8b33d81d3c11275b9e4f065fe7b23b62355625065ebdc8ce` |
| `train_gpt2cu` | `True` | `3105552` | `267fdc471f8a924751e551ed09bc98811092e0b3a7865d05a866ef1f4443f35a` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
