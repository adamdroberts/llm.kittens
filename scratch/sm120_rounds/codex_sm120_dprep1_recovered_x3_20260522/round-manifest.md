# SM120 Round Manifest

- run label: `codex_sm120_dprep1_recovered_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_dprep1_recovered_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_dprep1_recovered_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `659`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `95bdf5e3705a0671fe48a05a2cb20b0874f1a6fbcdf314557b852d0401e6de8d` |
| `test_attention` | `True` | `1800528` | `bbd8e9dbdc6fc6d8e1506cf410859c4a83902273ed05a2f2e3ae76b3b1a11e7d` |
| `test_layernorm` | `True` | `1278296` | `896b48202fd26e057905972eb369dbf3d50eb78aa41365a5e509e78c252068a1` |
| `test_bias` | `True` | `2089120` | `63ddf7eace288d5f33fd6c405f3e67487778e742d189aacf0653fab4a3c92b33` |
| `test_gelu` | `True` | `1179912` | `6df65677740ab4c91e610ad7b14ebc8f2cb685bd670c34551ac1f4bdeeedb2d9` |
| `test_fused_classifier` | `True` | `1208704` | `8c2f98174ec340f4c2e4f2867fc92f602086e5b388d05dc68ed280fecadec274` |
| `test_encoder` | `True` | `1210168` | `2aed526313e1f3859bd2dfcb1a3f0ece3ffc41f50e0ca099a01eb77c17b5a7c0` |
| `test_adamw` | `True` | `1183768` | `f2b64ac50c0dbf37b5c68b854e5b2237ec87a093bb68c4c2083a91508e727f3a` |
| `test_global_norm` | `True` | `1179464` | `01ef2b96d2223c4fcb8a867a44ee32be3e8a5356e96c9da74c8897a76f655df9` |
| `bench_sm120_matmul` | `True` | `2410304` | `f5901ede6aa9d625a9221a334fab9a093c0cfe8a118b19b4eb3a4f859af18e0d` |
| `bench_sm120_attention` | `True` | `1768800` | `1a4f28405c7c35b68c39725a30828726b8f7329c407d63ad1099e938de5bef34` |
| `bench_sm120_layernorm` | `True` | `1274232` | `f5a341fcae3147ba894177e37d47dfb1ca9afc544246fe35e315226b6fc45684` |
| `bench_sm120_runtime` | `True` | `2271168` | `0bbc1566a13f73bf2bf5d77ee2f96f33a0d7a8a3bd98114aae6e22d355c39c58` |
| `train_gpt2cu` | `True` | `3105552` | `2d7bf7fa9a275f4ebc29608b2861d12fa7b698f65c4a86955baacb1531ce2d2e` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
