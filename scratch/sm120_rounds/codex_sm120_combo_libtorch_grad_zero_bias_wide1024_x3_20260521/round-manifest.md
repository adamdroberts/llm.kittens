# SM120 Round Manifest

- run label: `codex_sm120_combo_libtorch_grad_zero_bias_wide1024_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_bias_wide1024_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_bias_wide1024_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `1`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `512`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `e678baab291d1d5545ac4a69c4defcd808b69f88d21b50e4a2d321507c987c72` |
| `test_attention` | `True` | `1800528` | `30d8c4f081dae74b90e8e911054ff5e7108a07e0ced466a3c34c5ca79f423956` |
| `test_layernorm` | `True` | `1278296` | `9c61b1f5dad49af7edc8db52ad2098ea264b44e219f9fc7c76fc79c9f2742b96` |
| `test_bias` | `True` | `2089120` | `5cd695e1a2232dcc97c1f138ac56a46e0ff69ddd69dc9ee25e79c9bd2a3ad982` |
| `test_gelu` | `True` | `1179912` | `8e7a57011f605f08b6d9ded7bbef20b28a94ef743a23786b5ae909777235dbb5` |
| `test_fused_classifier` | `True` | `1208704` | `6591648853cda2fe5c345509a25c7de695947644b7311b379fdef92e70211289` |
| `test_encoder` | `True` | `1210168` | `984672b3197a7adc8336904d38b710ce85ab00034ad6c99bc8c921f8c3c59100` |
| `test_adamw` | `True` | `1178984` | `b225714d2e85b8cc74c5ed89813c1515dce19f3d11bbe9ee1085a60fdb0c592e` |
| `test_global_norm` | `True` | `1179464` | `004cb7a0fbcb3990f575663b017acd20a9ddf6ead221da4ab5b69f76ebe6f22f` |
| `bench_sm120_matmul` | `True` | `2410304` | `40ed0552bb5d7b99348b805e5d13479632fafaaac15579002709105381f3ffa1` |
| `bench_sm120_attention` | `True` | `1768800` | `941a70f78361a45ed4ca5082036209d0c01f9dd428f18d7e7a05a28b01fd4b1b` |
| `bench_sm120_layernorm` | `True` | `1274232` | `51183d1cda398f2d257fae4eab19c1f7b171ad989e8586b30f78929a3519892f` |
| `bench_sm120_runtime` | `True` | `2270544` | `dcf247bd5d2aa36a06e8d9f98f5af2e2dda850a90bfd6b871cc6d61951cdc025` |
| `train_gpt2cu` | `True` | `3100592` | `065470840da9e9f632ce4d08a2a8c516521fd0ad22662637c3f997ea4829cef5` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
