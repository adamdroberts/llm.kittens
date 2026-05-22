# SM120 Round Manifest

- run label: `codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `603`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `4665594dd5ce86051c72a77fcdd753b3b8c3fbac74104d94c7cc7fc355d04227` |
| `test_attention` | `True` | `1800528` | `e678c4e79e3401b64eb5f313f2edb49278fed1eebdf2019d63dfb5f03d95ae8d` |
| `test_layernorm` | `True` | `1278296` | `7f60326ae7de9dfa8a8b8b03724504978425327d09ac22f7d2c829d7698c85f0` |
| `test_bias` | `True` | `2089120` | `1d86d3ef4dde73b020149c45c4508370b0ab99fa7ae6ff98bc4ef75ebb014cc5` |
| `test_gelu` | `True` | `1179912` | `70922b790fca348f262e1feb6bed90a72589acc4705e78d1932b37a551f4f5ca` |
| `test_fused_classifier` | `True` | `1208704` | `a27c243873a2f4c93ae3bf8baa88f74e42d6836f3572f63c0e67f10baff2b3fe` |
| `test_encoder` | `True` | `1210168` | `d207b72e9629a12ae31746316bf9f9a71eb97ba365933bf7822c3226c2cbcd07` |
| `test_adamw` | `True` | `1183768` | `a1524edbabf2697cdb17d6ab78b79d644799ea71e60ff1d7a813c7ab1a6e88fb` |
| `test_global_norm` | `True` | `1179464` | `27781fcbb12b538afa3d20b9f8d78036bd00c2023390a257ea6fe28d4d89c2ea` |
| `bench_sm120_matmul` | `True` | `2410304` | `5e944c3bb106ecf1dae8f89465a2b611ed48a514f09997d6d04a8aa86d6dc32d` |
| `bench_sm120_attention` | `True` | `1768800` | `8d1739d4b39eb140fc8840babe9aa7f9e157705ae359a2772965d000b755eaca` |
| `bench_sm120_layernorm` | `True` | `1274232` | `930f821d3a89b7f37269779bdabc3bd906710df21fa5a6241aceaf0691491d29` |
| `bench_sm120_runtime` | `True` | `2271168` | `8886584eb5f573391876a24513f1b2f1302b0c02f2ae898858e1d70e4cd6f17a` |
| `train_gpt2cu` | `True` | `3105552` | `799c99b2afd4ad4dd0b1dcba9f497be2bfbf0733487e6ef9c05599657972aa34` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
