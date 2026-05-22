# SM120 Round Manifest

- run label: `codex_sm120_restore_promoted_default_after_disable_backward_stream_sync_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_disable_backward_stream_sync_20260522`
- train output dir: `log124M/5090_S_codex_sm120_restore_promoted_default_after_disable_backward_stream_sync_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `648`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `e1ac878b6c14d3eacaf6432651d168a4b93e62ca5a83dd41df9a0aed6d6f05ea` |
| `test_attention` | `True` | `1800528` | `60cb45964f1e691f8942611da06a3e6e9dcf0bf13d503a4783fffce2bf79bcf5` |
| `test_layernorm` | `True` | `1278296` | `d37dced586dcf24e2c182c6733f0adc59c1846c8421792023ceffdacace312d9` |
| `test_bias` | `True` | `2089120` | `7ac7a52deeb26277f5418c80b6607bd1507293a58a7c892647136870112b2d07` |
| `test_gelu` | `True` | `1179912` | `d8ee0a49a5e92ad84764587d817d65106aa59389a725ca8b59cc99bd96b5bcd9` |
| `test_fused_classifier` | `True` | `1208704` | `01bbeca93b90a8c01172e41931002f6262be2febb507136f61a34226ce68545e` |
| `test_encoder` | `True` | `1210168` | `4c49d50cc4fc704f498436f339af39c5ba497f8acbc823151ef11280181ca934` |
| `test_adamw` | `True` | `1183768` | `4efb8c33d5e7f07ba990d2944ae0ff084e0b173173a38a410c10dad15902cf60` |
| `test_global_norm` | `True` | `1179464` | `78678998f7719f4f9c05a4e76b2ccac12ff8e59669ef2eff227cf7bba0ff642d` |
| `bench_sm120_matmul` | `True` | `2410304` | `ebc6ade2f1fd719bb43c9310212badda8011b2f32d2843fe04373561c68ea923` |
| `bench_sm120_attention` | `True` | `1768800` | `cf1f484c96d3f7da3a2e9cdbd02efb07535fb073ba6675ebda0e51df30d5a8c3` |
| `bench_sm120_layernorm` | `True` | `1274232` | `5fa979e1b1401e356f2dd09f69f45b5899436cf66a4c9bc6c662d19dec61bd94` |
| `bench_sm120_runtime` | `True` | `2271168` | `20ab5ee0b6e7e96560245854ae956747da5ce464f228e53ecac53bbcb41158db` |
| `train_gpt2cu` | `True` | `3105552` | `c8934ab48aa399d4dfa218bd50f91768c526e3931d4226d95cf35764977f7d2c` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
