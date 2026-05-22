# SM120 Round Manifest

- run label: `codex_sm120_combo_libtorch_grad_zero_dprep3_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_dprep3_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_dprep3_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `1`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `587`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `22912ef065f8a6e4de73e07a6b6ab298df39931b931cf6f81bf348e1547eb42c` |
| `test_attention` | `True` | `1800528` | `4ddf696db021f323a606e9efaf6f66caa9c1e851149e58fa633193a7983e138b` |
| `test_layernorm` | `True` | `1278296` | `39e4c8a99460b2a7b0200bdd55000ae432edc9c87e11c5860bdfc6875a3c44bf` |
| `test_bias` | `True` | `2089120` | `512ebcd44644578536a90fcc1b42fdad7e2311f3ac4c053bf8c972b03f11f69f` |
| `test_gelu` | `True` | `1179912` | `d6929532212a72faf2148f40101d6f799aab3fd4505ab3fdccb378ccaf808572` |
| `test_fused_classifier` | `True` | `1208704` | `65f8aa55e18e2b377dda2c549d24042f8e60036af0cc1f0a865f680a51bfd141` |
| `test_encoder` | `True` | `1210168` | `171d9a24e8d4c62a4974f73ca37274c195a3ef1b3a1bcbf2b42ca010d9b9d679` |
| `test_adamw` | `True` | `1183768` | `406269c34eecb877bb61f7adb19310e0b68453b6cd3f842ca89f14326675059b` |
| `test_global_norm` | `True` | `1179464` | `63df2abeb93435ebf84124db0f4128fef62a8ccd4e42ceba4db2ee00ab5837e0` |
| `bench_sm120_matmul` | `True` | `2410304` | `b92acf72353d61ec8f82681382faf31cce55da37fd963b3ba56f96a0daa2952b` |
| `bench_sm120_attention` | `True` | `1768800` | `3ba83f999df984410449bb02cc5cfd5d312f34c08974297e89ba54e7010b2b2c` |
| `bench_sm120_layernorm` | `True` | `1274232` | `2acd83ee1cda9e15e47c3ee881246f61667b668cafcb23080497b0a0ecb09557` |
| `bench_sm120_runtime` | `True` | `2271168` | `8e911b95b5166918b779d3507156990698f43203671a5035250cab30651e0d8c` |
| `train_gpt2cu` | `True` | `3105552` | `7a4f77cdabc91130a5ca8ef58ebba782008678f151f618879ac95ecfc0fdfec9` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
