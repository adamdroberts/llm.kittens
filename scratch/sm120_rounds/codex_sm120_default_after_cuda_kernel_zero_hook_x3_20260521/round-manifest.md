# SM120 Round Manifest

- run label: `codex_sm120_default_after_cuda_kernel_zero_hook_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_default_after_cuda_kernel_zero_hook_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_default_after_cuda_kernel_zero_hook_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `529`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `7cc05d4bcb3adaddd878c27739efccaac971f3bcd74ad02984802f467a866be7` |
| `test_attention` | `True` | `1760032` | `db3128dd7641cd4f50872a0b3794519d1eed45d9338578380a57b2f79d922ef2` |
| `test_layernorm` | `True` | `1237784` | `e1f4b33cf9dbf57620cfcf9581e25a6f8e6cc666dd853c815a5c10494312cbe6` |
| `test_bias` | `True` | `2048616` | `27fd068dc399434b42b08403d17cfb4b04388912c37488f85c598049e190e597` |
| `test_gelu` | `True` | `1139336` | `d8e8d240416b5e12874ceb16ea490d55cc32552793b02e98888bce54db8c9053` |
| `test_fused_classifier` | `True` | `1164032` | `8e6eb11ae208f9d4d96aa2e6aeb46384cdc26adb393a56c856761af101c6fe7d` |
| `test_encoder` | `True` | `1165512` | `1aab104d2e13e483f41f1b40470f8101be37eaa30f01d8763685348ffe1196af` |
| `test_adamw` | `True` | `1138408` | `b7b7eaaaed27a9b21362ed54d821d4445885c96e4f5df69002a24673ce54b01a` |
| `test_global_norm` | `True` | `1138880` | `e3642ad42554b2ec77c40df75950faecea9176203627171c0103e0e514cce57c` |
| `bench_sm120_matmul` | `True` | `2373912` | `41b69d7388a3f78dcf355d93440d3798561220386f9787b382d813880c19f3ce` |
| `bench_sm120_attention` | `True` | `1728312` | `b2415d4e16930d43d55f63ee176f3507123744ed18a09dee57c1ffe41be590d1` |
| `bench_sm120_layernorm` | `True` | `1233728` | `6d0bd3311eb66b1da0de06ff0f7b83c72436b14aa1d8dbee8affc8c564114270` |
| `bench_sm120_runtime` | `True` | `2221864` | `acbfcb1c835a8e0c62e760eaea6538eef3cb56a2c6f154c09ec0fcff0d32772e` |
| `train_gpt2cu` | `True` | `3060032` | `0ba6eac7b49f7e56fda3a8d523f17c659f3060fa67ad94f33ed9e1c253c3e839` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
