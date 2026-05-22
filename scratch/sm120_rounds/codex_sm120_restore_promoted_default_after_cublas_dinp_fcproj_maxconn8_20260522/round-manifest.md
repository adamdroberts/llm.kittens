# SM120 Round Manifest

- run label: `codex_sm120_restore_promoted_default_after_cublas_dinp_fcproj_maxconn8_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_cublas_dinp_fcproj_maxconn8_20260522`
- train output dir: `log124M/5090_S_codex_sm120_restore_promoted_default_after_cublas_dinp_fcproj_maxconn8_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `647`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `3534532a287d02064e8f660f075f303a53480349d83fbdf78f888b2d393d173f` |
| `test_attention` | `True` | `1800528` | `c343583b8a02307ae42ef00e9da66d1832c5214bcdf8c4ce1a737e54f6be4cae` |
| `test_layernorm` | `True` | `1278296` | `9b3f5e9cb816ed5eee6f301237bfa5c3053773732bb1fcc9b71f6816e13ee91b` |
| `test_bias` | `True` | `2089120` | `ca3c04fe06bff7a5b47caba341f18c0bae0d82977c225735d01f987bd6d858ee` |
| `test_gelu` | `True` | `1179912` | `8d5ab842693a65a19e2f77bcf64010a982eed47eba94587a66b0d0331ea9d408` |
| `test_fused_classifier` | `True` | `1208704` | `be266110d9647f723dff16396f337e852e13f178ba3301a1d8aadd262e94b137` |
| `test_encoder` | `True` | `1210168` | `6c0b6f32d0ddd2218427dc8b39199d8daf764f433d180df7e464044bd2403329` |
| `test_adamw` | `True` | `1183768` | `cf94793e51d0e581702a26ae0a44353c4431157e3ea114a5f6ad02b302671710` |
| `test_global_norm` | `True` | `1179464` | `a86728c0f008640ed93cbc1cbeb7464be66d7f98f29a621ee6fb44740d599ece` |
| `bench_sm120_matmul` | `True` | `2410304` | `27dfc21dd29db7a379c0484e0f10db35006c64d4038a892b6cc8f418e66657a9` |
| `bench_sm120_attention` | `True` | `1768800` | `d16499a6a63ac363ed8025b7c8dc7c96ad054e9c5b6fd43420681eddb9b2fc30` |
| `bench_sm120_layernorm` | `True` | `1274232` | `50a311995c44919cf5f23bbe17a5e625e9fe8e008b50293bb0c58fafcf25fc1c` |
| `bench_sm120_runtime` | `True` | `2271168` | `ce7d32474943ccd37442c23c8ba9b804de51d34ba83340fc61b11e3abc5117b3` |
| `train_gpt2cu` | `True` | `3105552` | `960c820c780318f9e21e568d7fac227de02e3a9e65877f03e1edb996ca8415e4` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
