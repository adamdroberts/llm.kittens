# SM120 Round Manifest

- run label: `codex_sm120_combo_cublas_dinp_fc_bias_wide1024_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_cublas_dinp_fc_bias_wide1024_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_cublas_dinp_fc_bias_wide1024_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `513`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `c40471a401c20571d94b6b3d628a246c5688ddfd8f49dbdf4987893f4d543817` |
| `test_attention` | `True` | `1760032` | `ff43beda950964ef0442cb6d2619f7f58d1a962298964e7f63ea79eb3e5c09e5` |
| `test_layernorm` | `True` | `1237784` | `6776c14bd76132f70b65b16c54a2db5197bf7fa67e7cc672f7534cc33ff70f7a` |
| `test_bias` | `True` | `2048616` | `5bea7c59006fcae62de66e7781fb07bc2d8967f61e6c9ba96fa68c3015b6d3ef` |
| `test_gelu` | `True` | `1139336` | `a3635ccdd520e9c70039af455ddb1ae26a5d52e6e9e4faa8b39d53311af9f5fc` |
| `test_fused_classifier` | `True` | `1164032` | `ab482c2c853c57942cd84e48b8a4030d8e00c2437b946b9c2c8c5d685dca95ee` |
| `test_encoder` | `True` | `1165512` | `5535126e4ab1438bbb440f37aee1dd5eeb0c1ce047db332371acf87e2fbb5cca` |
| `test_adamw` | `True` | `1138408` | `28388403f0886bb7122c5e62f368a996d71eab722fcbf92314f5be6329fd419c` |
| `test_global_norm` | `True` | `1138880` | `f2170fbeb3b246e3e452b66beeee5bae7ee4abab3723171de184068bcc0b3750` |
| `bench_sm120_matmul` | `True` | `2373912` | `f3c98c237f34def72d8875537dd64923ccfa1b09785e1043163daaf91de644f6` |
| `bench_sm120_attention` | `True` | `1728312` | `e2061aac94bcc0442eadf9244f6234d09fa2e59d88f9d27de8a1daa71ec7b6e6` |
| `bench_sm120_layernorm` | `True` | `1233728` | `9f0825be06161557523284dae71dff706cf944b96641c86cf097c8877b8100e3` |
| `bench_sm120_runtime` | `True` | `2221864` | `9d73c6e702b014e7f35762b9c223eff01e06eca2f21f5d563eaf60c728d1c8fb` |
| `train_gpt2cu` | `True` | `3060032` | `87ea9661ce397806bf8a1be2fcb8d91734adc784e400d23f5215ae8f4cee3946` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
