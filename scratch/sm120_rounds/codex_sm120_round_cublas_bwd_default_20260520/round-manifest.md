# SM120 Round Manifest

- run label: `codex_sm120_round_cublas_bwd_default_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_cublas_bwd_default_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_cublas_bwd_default_20260520`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `448`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2128288` | `e7b619df3763e60e3fd0084735113f38b4915ed4e6be7e587fb81ed28ad73454` |
| `test_attention` | `True` | `1760032` | `f2274f0efd534141d7caa7b6259a37b7d44cd9cc4821f9dc1dc8e0b01e74e2e0` |
| `test_layernorm` | `True` | `1237784` | `57a2d563e1014f0919b934527723ba43b322bb700ee7484c7f162f86b7cf1cf4` |
| `test_bias` | `True` | `2039504` | `5c8e48359b16085760adcf356e9aaff1f30a267511190521f914003a4be979c5` |
| `test_gelu` | `True` | `1139336` | `101db2a43147f713f1fa00709ffa9eb9e72c23bc1322b6e7e8ec27eed685be13` |
| `test_fused_classifier` | `True` | `1146768` | `f5aba2ada19303bda8055e4349099b460197ebfad66835bef6ea17225d43f2ae` |
| `test_encoder` | `True` | `1165512` | `6112cf047e6bfebdfa8bca432f7316fb55e151eb6d0ce2903cf22e33e052f221` |
| `test_adamw` | `True` | `1138408` | `3eb0ba2d332b5bd9d09eee050a737282a16f8d80ece3bf9a43ddace6ef69628f` |
| `test_global_norm` | `True` | `1138816` | `2cd6c8cccbd507b8c423489357f22db89f59d21e7ebcd5de735a644a0428e6ea` |
| `bench_sm120_matmul` | `True` | `2344440` | `d70d5a2a7844075d0da936eeaf13a3e09290ecf9c85083db04a3e979606316e6` |
| `bench_sm120_attention` | `True` | `1731352` | `b0e7b6b4c2b8b9f585b080f0d5220d30dbb9808e624cf36d738b062f6954a560` |
| `bench_sm120_layernorm` | `True` | `1229088` | `1564863ead9415f128362ee11ba0acc636bd9f96b7a6de78cea263f62a353ada` |
| `bench_sm120_runtime` | `True` | `2168552` | `d8714360efb6035a9fa7ae7a76189a8a8763a41e003f4e0fbbd2f3e96ec779ed` |
| `train_gpt2cu` | `True` | `3037016` | `09e97f1edd86653e366b3e0f864058282658b6506f1ed8c4641ca4bebc518b5b` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
