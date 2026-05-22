# SM120 Round Manifest

- run label: `codex_sm120_promoted_classifier_exp2_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_classifier_exp2_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_promoted_classifier_exp2_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `612`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `5335f79c2ef6756cff0ee5ea6914e2f60d1b46dc1fa0385ae02ad5b61c764253` |
| `test_attention` | `True` | `1800528` | `9a5dd1c91cd2bd85ce27c723e9ca69535d7c3808d0ab1bdcce6739727ded054b` |
| `test_layernorm` | `True` | `1278296` | `52af63df2f972bcf78ae0ad8c0a363bce8b0ea20c75388e0c005dfea75a913a0` |
| `test_bias` | `True` | `2089120` | `eb2b883eee76164959fa351bd28383e49ebc6324c305cedb3c7ce71db8870c93` |
| `test_gelu` | `True` | `1179912` | `96a7892ff43c19cde14ec86b1c3ecf1f8928ab2093d2f48bc6143ab37765651f` |
| `test_fused_classifier` | `True` | `1208704` | `e6e964abb786501bac2b2f57efba33aef1c0d4b433ad7bfec93a8e3df8ff5270` |
| `test_encoder` | `True` | `1210168` | `2a43f09ac43a5309f88feb1a5dbd11792b0d5edf700c2ce2a88ce38b3ae461f2` |
| `test_adamw` | `True` | `1183768` | `30ca71668c2938eb4acc6e49d4e0e8feeaf0b2a874ebd9c8d96e512893070cfb` |
| `test_global_norm` | `True` | `1179464` | `83ba638e4ab07c1241aa69506743a3b8fa0190e22e79fa5dc4230d25ec4a7741` |
| `bench_sm120_matmul` | `True` | `2410304` | `3d1bf04068ad32f2afcf1b71df637dc50af0cd024794f49d024749eab3b33837` |
| `bench_sm120_attention` | `True` | `1768800` | `c1ac2d26745bd509eaebb67523f3b5b4ec1ffd435f7ea62ee250c7544165286f` |
| `bench_sm120_layernorm` | `True` | `1274232` | `00caddb9457324876b891a4e6932b6e921dc123bbdfc5a8791f971616301cfc4` |
| `bench_sm120_runtime` | `True` | `2271168` | `064dc46b2c5fdfc81fd94144f79b51b1fd9c022e23ab4d6ebeef500ab0daaf9f` |
| `train_gpt2cu` | `True` | `3105552` | `b388ebb5e16055d01b6a88f063ca8e67bf91b190ef716ee5751494cc658f6b96` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
