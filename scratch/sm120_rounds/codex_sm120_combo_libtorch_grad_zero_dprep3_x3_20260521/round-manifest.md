# SM120 Round Manifest

- run label: `codex_sm120_combo_libtorch_grad_zero_dprep3_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_dprep3_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_dprep3_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `1`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `586`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `ea9391c0d23a8b253ac3a0a75206d4e25a0a31598c581d5cdb1029ff85c41ffd` |
| `test_attention` | `True` | `1800528` | `732d625deef261638915638dcc2ac670152caf16ec64eadf03603cef7e2355fd` |
| `test_layernorm` | `True` | `1278296` | `72192b6334eac42c84e52b0a8222ec6715737a93858c059e74896944ee722d40` |
| `test_bias` | `True` | `2089120` | `a60d3eba479614b4977e139328398bf8326854f88074beee233df489106391ca` |
| `test_gelu` | `True` | `1179912` | `25c5d5b4a2b7b44917b2249c28fededb08d5f62eca4d39baeeb6ba65ee5517f1` |
| `test_fused_classifier` | `True` | `1208704` | `e10e2eea66af5c9bfaf116030a84ec9ca6ee557257aec76d996f25f75e64ed6a` |
| `test_encoder` | `True` | `1210168` | `79d3f77b54d5210f5f625eee272e63711c1b71f16b7338d1f0e2e2ef32bd17ab` |
| `test_adamw` | `True` | `1183768` | `a5964de2a97c37f6d36b30490f62be01c308ca617199b81ae239fadc5eecb2b9` |
| `test_global_norm` | `True` | `1179464` | `97876926ac1e2b820aa4ebacf1b8d2e294fd357f8289585f88d5986af9541611` |
| `bench_sm120_matmul` | `True` | `2410304` | `0ec563d43b53528b95ee0732a9072c78ad9a1101e4df3b5c0027a99ef7ef6537` |
| `bench_sm120_attention` | `True` | `1768800` | `0ecac7db774646a1a02cddbe1f2fb2e4fb6a5710cad4ffebeac5f3d813932372` |
| `bench_sm120_layernorm` | `True` | `1274232` | `0f8c39c5d4f41f3d8a651e87c09947dc2f0d405f2787a1056eb0de20cc1e8fc8` |
| `bench_sm120_runtime` | `True` | `2271168` | `d3ae065c9856547941f70b7cdc4f82cc7a10ee62e38f96f083668a415e3b5e6f` |
| `train_gpt2cu` | `True` | `3105552` | `a85b2062c5a853b33e0ec3abff24136fa5077b38152c88847a4b7962d5bf0e73` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
