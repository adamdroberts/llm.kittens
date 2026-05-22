# SM120 Round Manifest

- run label: `codex_sm120_promoted_disable_cuda_profiler_recovered_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_disable_cuda_profiler_recovered_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_disable_cuda_profiler_recovered_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `662`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `0dd15ebdf343d523430d9484dd397b841482bf16b7e160fb5e3532f0474dd63a` |
| `test_attention` | `True` | `1800528` | `62068711d42a703dd2d5ae86916ba075ca9aad6431251b4d0f8faa467f94b4c3` |
| `test_layernorm` | `True` | `1278296` | `fe35c512fdb7d9f954d944946ae5350d8f1b6a1ccccbc6790265c1fa395ec08e` |
| `test_bias` | `True` | `2089120` | `887b5dfeadcd734b79a5d7f637f37616c69586e68dc0b3767284063e0d695b14` |
| `test_gelu` | `True` | `1179912` | `cad73f6e99a5022805b33d4cbbdefc4f39e63f9821ca505fcf37f938ed445c75` |
| `test_fused_classifier` | `True` | `1208704` | `2f06f804fea34a2c31a507f4140f06273a58a84453979b66241c02d92aec15f4` |
| `test_encoder` | `True` | `1210168` | `c1fcb46e93d8a771d2814a49bd79d59a08535dc7fbfa2ca65453fccda8874874` |
| `test_adamw` | `True` | `1183768` | `ae0b7102bfc7e23e5d769bb4ba74cc7b79ec12ed42158da6627b2c00029e5284` |
| `test_global_norm` | `True` | `1179464` | `bdf0ceee08cb8152e034b1d67d9686543bf480452035b9c70366931e1af699d2` |
| `bench_sm120_matmul` | `True` | `2410304` | `a2724133915331b25b3d6d2a73a5e6c524ef230e8081475e2a6a643aa6ad8f96` |
| `bench_sm120_attention` | `True` | `1768800` | `083730bd3624fedfdb666e969eb4f34ffafc9f36373eb786b4b39658dd001bae` |
| `bench_sm120_layernorm` | `True` | `1274232` | `bf4f51de1817e0296ee39ab88f16e853944b4d5a2195f2b49769b218b8d5a85a` |
| `bench_sm120_runtime` | `True` | `2271168` | `8050d5c1234af05b5f10bbd6e43439942831441b5ab463a1047bf65d15b0d771` |
| `train_gpt2cu` | `True` | `3105552` | `105e95d7e7317561ea6b3e7e310d0c65200b8603a689367ad2e6334ea64e30fd` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
