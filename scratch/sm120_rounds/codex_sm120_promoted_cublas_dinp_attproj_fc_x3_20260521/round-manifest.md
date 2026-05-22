# SM120 Round Manifest

- run label: `codex_sm120_promoted_cublas_dinp_attproj_fc_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_cublas_dinp_attproj_fc_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_promoted_cublas_dinp_attproj_fc_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `611`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `2c40de77bacbd74b9d040222be39029bb4a02a54e26e5cc765290062a2df7f03` |
| `test_attention` | `True` | `1800528` | `dce94db413d576a7acdcc076dcadeddf795cf3071516bba6ad432067b8d884b7` |
| `test_layernorm` | `True` | `1278296` | `ab5958f68f8542ed8d8497477d6fd64c3add2cc526d7a0652373e4acf0c5c727` |
| `test_bias` | `True` | `2089120` | `6438a2085e5afbc298bf846e7b3f81e75a8f70735239b19f84734030f3469c9a` |
| `test_gelu` | `True` | `1179912` | `def92e2764ff970f2259da113b110fc252ef8ae2729b96b33d7da7303b8a8a75` |
| `test_fused_classifier` | `True` | `1208704` | `36f08ccddb98014f95091c5381b17ef36e4b2fd54b2bc5d6c5187668368e1e78` |
| `test_encoder` | `True` | `1210168` | `f88d9bc36f2698a3022641bc8365af04e3709eff66adb93579852dacc6b1737f` |
| `test_adamw` | `True` | `1183768` | `b7652daed9949cb9584a88abe9169a464568979c20b79703cb9948b13e4b93d3` |
| `test_global_norm` | `True` | `1179464` | `734dc3027dcdfb5f60c26cbc4ec0562fbf43eff5b56273c50d86f365bcb946be` |
| `bench_sm120_matmul` | `True` | `2410304` | `a802fa520565934869b4ca20a006acd4059a2404fd90a723c1cb120114be8a50` |
| `bench_sm120_attention` | `True` | `1768800` | `0ecc23a7f3ff42ca7c3d46821eb78b57504412f5d9e64191a3a848e7abf360ab` |
| `bench_sm120_layernorm` | `True` | `1274232` | `f4d70a0f46a9ae5866302b4c9d4d90f057e47c18a590feb59309cd4c8fd4e6ef` |
| `bench_sm120_runtime` | `True` | `2271168` | `1a5b38cd02d6a4cbd14f56c99602d00052401523213f414a5cc66e67d1f6192f` |
| `train_gpt2cu` | `True` | `3105552` | `335b4c41ae66898f142843075e5151ab2561d72db618cd2ea4beb47dcaccd6a7` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
