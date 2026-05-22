# SM120 Round Manifest

- run label: `codex_sm120_combo_libtorch_dresidual_zero_fastest_dinp_tk_dgelu_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_fastest_dinp_tk_dgelu_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_dresidual_zero_fastest_dinp_tk_dgelu_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `572`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2190792` | `378c280907ef66101867fdb0b671a2acef52598ff795bf638e792b5bbc46ca5e` |
| `test_attention` | `True` | `1800528` | `4f99ff254b8251f5a4141555e5758bb33910b8921b950d376d674686e4b2afa1` |
| `test_layernorm` | `True` | `1278296` | `187737572fdc76b30876d61637d7a5725fc6f77e9907b33fff7ddca5b2659b33` |
| `test_bias` | `True` | `2052256` | `677261154288efd689d2e2a765f554aa712169e72fa09738472a2c7ccb18bce4` |
| `test_gelu` | `True` | `1179912` | `7bef511185e2a69b47017d55b665bd5b9703f465d3b9622276c7cb13552954b6` |
| `test_fused_classifier` | `True` | `1208704` | `b790ef052f9c8fd94acf93f0a7005ea3cbe226fa847a5390ee2da0b90fd56bc3` |
| `test_encoder` | `True` | `1210168` | `85055f5c52c60837864ce432dfc28e4c310bde0c9c28c0fe249a7e38a27a5815` |
| `test_adamw` | `True` | `1183768` | `aa8e680ce0c995e87560e3e4f880c6bc22e9be9423f42025d5e768cdeb51cd8c` |
| `test_global_norm` | `True` | `1179464` | `9a578b3ca022b7a49bfb76fe9b518e1ed3184794e049768e17c669bb86a1bed6` |
| `bench_sm120_matmul` | `True` | `2369344` | `d61aa2c1908923c5b6243a663e6df9ff4e89842fded7a2bbefe08a4a9e5d70e3` |
| `bench_sm120_attention` | `True` | `1768800` | `41b35aadf0e6a59e86fdb1a13e30af353984c76a1dbef7703cfe655001253c3b` |
| `bench_sm120_layernorm` | `True` | `1274232` | `bb38ed13034fc384a5f4e86499545f50ed565a0aee5ea0eb6f7845603b407a28` |
| `bench_sm120_runtime` | `True` | `2234304` | `2865e04ca6aad317a5f00be10a4f05458c3edcb0a6e71582ff9a71542d7a483c` |
| `train_gpt2cu` | `True` | `3107416` | `c0bc384dd02281dc9a4a6dc4e451b97b1304fed8124d20f9baed10cf7ff7e1ea` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
