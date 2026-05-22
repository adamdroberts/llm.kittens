# SM120 Round Manifest

- run label: `codex_sm120_combo_project_fastest_dinp_tk_dgelu_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_project_fastest_dinp_tk_dgelu_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_project_fastest_dinp_tk_dgelu_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `561`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2154392` | `ad28e85fb0c39faa90206e37c4cced2d1cadeca6313a1de9519fe877107784af` |
| `test_attention` | `True` | `1760032` | `945cc91028297f25d887d9c0e49a4dc026e7857127edc9229d08f3cc74abb956` |
| `test_layernorm` | `True` | `1237784` | `577e7822f655c2750c8694c4187052b33fe40ddf8b1a8d74fd3219860b2d8b2b` |
| `test_bias` | `True` | `2007656` | `d57086fd591d092d2ad8b6ed39df8053b7b3c0f5ac04c73b8d5931d0b6daf787` |
| `test_gelu` | `True` | `1139336` | `090c89338b2b2ffe93b4adcf05476c9fcc372dcce9615c35b71848002decb016` |
| `test_fused_classifier` | `True` | `1164032` | `09f5b2b78ec1373497aaf8e0dd9c43aa88bed6dad55690ad0e3c522b1c43d566` |
| `test_encoder` | `True` | `1165512` | `0348585d3847f6ef5278f3ead1239ea2bc38a90ce2204eb8d4c982785cf40287` |
| `test_adamw` | `True` | `1143192` | `c0ee9a3b2befecee9b57dc220b4bf84e55a07886cabdbb4b04cb63f4e495cabe` |
| `test_global_norm` | `True` | `1138880` | `a56fd572eb7c802f1546e7864899a4678d0479e218cd29c5178fab439fed51d1` |
| `bench_sm120_matmul` | `True` | `2332952` | `fbe06457bbd76b7fab0ec59e836b51e3296aa11b34867973076684384ee10365` |
| `bench_sm120_attention` | `True` | `1728312` | `61c2708a2ac1aec4c80d1fd49b3715c71b33ac64b7b1c5a932d40d7a76e0b4a5` |
| `bench_sm120_layernorm` | `True` | `1233728` | `7b4c0df312e3894d9ff32f91144b256127a37dc02489dce67d1fa352c0b02145` |
| `bench_sm120_runtime` | `True` | `2185616` | `f00b51214427052b960fa22a9445a17bcf9f30fe0bea58e4bed87d4ea9b03911` |
| `train_gpt2cu` | `True` | `3062760` | `961dc27621eb2056f134e4c3cb55886362679b2f37ad8697d946db76e1565491` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
