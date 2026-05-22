# SM120 Round Manifest

- run label: `codex_sm120_combo_cublas_fc_tk_dgelu_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_cublas_fc_tk_dgelu_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_cublas_fc_tk_dgelu_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `535`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2154392` | `4d8c133236cdaf2a0f54ee044c96637d6d11c12e93ba492ca171c6eff3eda6e3` |
| `test_attention` | `True` | `1760032` | `f4d9e2b58b3ddb68bb83da7bbb079104211e149ee5598ca559fe3f06443d7626` |
| `test_layernorm` | `True` | `1237784` | `3975a5adb3b5cdff3a5ef840d7572281dcff0e9614b6366839768b1dd0a2902d` |
| `test_bias` | `True` | `2007656` | `0bdf1ed51508b55001aac0c3a072e7372e86563f1d95ca1c3726671fec200ec0` |
| `test_gelu` | `True` | `1139336` | `111ad9523069bf7d6d7ad8eb3926373d59f666f8464040e4dc6d543eace04ec3` |
| `test_fused_classifier` | `True` | `1164032` | `ac738588343208b814923ff59da205b033ce3474b030af052ce23e03e8df4c08` |
| `test_encoder` | `True` | `1165512` | `694961291fe985699ede8855b64725cb793eaa6d3b79d5921ff3a689ca0bc262` |
| `test_adamw` | `True` | `1138408` | `df189e1339484a37555c8da1070c5a68ca6605bcea32fcb88c8fc025872a645f` |
| `test_global_norm` | `True` | `1138880` | `82bdd572c9aca1be1d761325adf775efe36e43bb5152319a76a88523a7064dda` |
| `bench_sm120_matmul` | `True` | `2332952` | `a84d3bdb304c3995d0838a6dfaf36031fc6f9e202334417d9abb381a7549c2f2` |
| `bench_sm120_attention` | `True` | `1728312` | `77012542e6b99e6cfc8c05463bf898e77bde2b0097cf380a92d052c0bf2be781` |
| `bench_sm120_layernorm` | `True` | `1233728` | `d0a7c926a0e31c8a2bc96aee58ea2f24cefb5ed16b373db2e0a7e9b3625e0a84` |
| `bench_sm120_runtime` | `True` | `2185000` | `c7cc01dc1b59c81e3bef05f390f22f4357fd1d4131e6954e22982c2456417114` |
| `train_gpt2cu` | `True` | `3054992` | `5513175c7e98ff69637bf1aabc5504149001ab95db23e6b14ee03556360b4d2d` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
