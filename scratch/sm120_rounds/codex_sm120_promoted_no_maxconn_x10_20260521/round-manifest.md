# SM120 Round Manifest

- run label: `codex_sm120_promoted_no_maxconn_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_no_maxconn_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_promoted_no_maxconn_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `610`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `2993f9e9f77731ceec22044b7d0d02742f79f774b2aa6bafd714302438432a35` |
| `test_attention` | `True` | `1800528` | `dbf09a7a92221680b41d789727b5bc978b0e03d17ddf9af93418cf342b8c4b61` |
| `test_layernorm` | `True` | `1278296` | `ff4f68a30c1aace71b764e02398a4db68012f4604f36c012af81b39d01b5ac0b` |
| `test_bias` | `True` | `2089120` | `3ce962f4d79396df1b3cfc0a7a6d55bfb90e4e2a40ceab3005af6d9c8849100b` |
| `test_gelu` | `True` | `1179912` | `88b38731bba56fff4fe3de87a74ec3daa8b531f6ec45d5386ade40b8f6475b80` |
| `test_fused_classifier` | `True` | `1208704` | `04f774c2d693b32c55dbf62ecd11c70cfd58b52cfe09f419b8d34c2e1b7d7f74` |
| `test_encoder` | `True` | `1210168` | `50b0eb3cfc37679b56443291f057f3ff9c49fa04375fed24b54f8355747b0a2c` |
| `test_adamw` | `True` | `1183768` | `873bdd440779f4e19720fffe88303c523259821c35f35d0c5c2d1a4e4f732c01` |
| `test_global_norm` | `True` | `1179464` | `98e35184d3b6a6d42fea5c3366c54a772e1e913597b4b5c8762d883f6bdf682b` |
| `bench_sm120_matmul` | `True` | `2410304` | `f35e697436dffc199e985d6841ae1abf704e620775670286fe6054cbdeb43a4f` |
| `bench_sm120_attention` | `True` | `1768800` | `2160e9365306c684e068201c2a9b65ef52e22e4d18926a891a92c8384fd13c61` |
| `bench_sm120_layernorm` | `True` | `1274232` | `9b288bb48eeadf12d33b28f398ca66125ab40eed94e34500dc2b44e5adf6f7b5` |
| `bench_sm120_runtime` | `True` | `2271168` | `7ad1a60018236b7df8348dcdd6786673b37e5afd539c18d9e978e56026ef1bed` |
| `train_gpt2cu` | `True` | `3105552` | `2e10f52c198c1bbaa43f79dfa3cda487013a403e240deeca5072c5a16513f6cb` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
