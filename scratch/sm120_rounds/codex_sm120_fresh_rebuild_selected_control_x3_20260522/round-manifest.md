# SM120 Round Manifest

- run label: `codex_sm120_fresh_rebuild_selected_control_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_fresh_rebuild_selected_control_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_fresh_rebuild_selected_control_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `655`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `3a8036ff484290b4cfcbae70fb3cecc40d1c8e909d48afee6dd7dbeb9dc77888` |
| `test_attention` | `True` | `1800528` | `04d284c2036e967f0f5e30a60b958dd2dbc6a79fbd7f8aadaa7ccc251ed39fc3` |
| `test_layernorm` | `True` | `1278296` | `656f7c69af3a8ae5dbd08be3e09e297c22638f6403d12ee4cf5b620becb6c7ad` |
| `test_bias` | `True` | `2089120` | `0f804f8c6cec87ce43127c37227906ec8ae67c2b1672ae8f5802ca2c934d4a61` |
| `test_gelu` | `True` | `1179912` | `6f689bb88eee429a4bc18c022a2d4445d39ee9d781ed1dfabc312dd7068ddfb5` |
| `test_fused_classifier` | `True` | `1208704` | `bd8b0d1b5f699d8bc06ef88d00a9df8309bc79d4e478a4d483a04018055b1998` |
| `test_encoder` | `True` | `1210168` | `75c630140376e2d13847cdf9ff195f48d41f6e2ef4af4177f79a7a2ac5225d40` |
| `test_adamw` | `True` | `1183768` | `0f80968a17df8ef810e857e2d97a950557535632c5f7143e0cd2f145225cb69f` |
| `test_global_norm` | `True` | `1179464` | `063dc25de10a3382466c733bc1f72325a0f19749917b905947a70faf434f628b` |
| `bench_sm120_matmul` | `True` | `2410304` | `ef07f1bd262c3aeb5523c9aedacc14cafea6fdd669e745ed2661d2acbf2f4686` |
| `bench_sm120_attention` | `True` | `1768800` | `26828157ad4fc1de8811121bc140d3d7600743a98bdad7c67e30b2ea2924b756` |
| `bench_sm120_layernorm` | `True` | `1274232` | `1c9f9967fd7ed9994984d06cd437744cc5bf638fc4fc91fcc84d461bfa29e317` |
| `bench_sm120_runtime` | `True` | `2271168` | `5377baba02017e1620968c4daecd849b1ee1e5a7a70c30fb5d52233a1cc01de9` |
| `train_gpt2cu` | `True` | `3105552` | `814f797a276d0370208e6a84b33b41f674d139947bbde3cac50f460f887816c9` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
