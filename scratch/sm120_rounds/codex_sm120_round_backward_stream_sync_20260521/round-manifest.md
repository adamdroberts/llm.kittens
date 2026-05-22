# SM120 Round Manifest

- run label: `codex_sm120_round_backward_stream_sync_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_backward_stream_sync_20260521`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `489`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `de4347893d8b986e84c58292855e53fab1b310bcfc05fa5bb6ef7c519de20e33` |
| `test_attention` | `True` | `1760032` | `c45e2c32f0a44198382adfcc9c7c1a9b7a5eb5f22aa149876792804a37886663` |
| `test_layernorm` | `True` | `1237784` | `08aec408ce272f5b62e773bcbaf13bee462134b6e68290105185b38144905c14` |
| `test_bias` | `True` | `2048616` | `d8bcaceadc4822ed9fab3a189cefe185ad6203992aa6c6dee51aafb59a82a48a` |
| `test_gelu` | `True` | `1139336` | `9540466aa3cad8813f45973035278a68deac0914baf00e53a9a55f8c42823aac` |
| `test_fused_classifier` | `True` | `1164032` | `ff7dd22a2ddc91df8713aeee83b856e92c34d96c6ce8fba1a429b61aa9b4fe3c` |
| `test_encoder` | `True` | `1165512` | `a02abf5a854383421e5e880f52804914d7594562fe17777256069742d8259dac` |
| `test_adamw` | `True` | `1138408` | `df0a65ba5b5388e4278bbcb21991ef773cd37eda5b4391a8ad292aab14958da9` |
| `test_global_norm` | `True` | `1138880` | `7c7b9cbf0e18320f2333d3c5824223e9915724ab12e81dd3a421b024ea98284b` |
| `bench_sm120_matmul` | `True` | `2373912` | `fce7d9526a5d79b967f1dac8dab98684d8040503f68a65dd998089572dc204e7` |
| `bench_sm120_attention` | `True` | `1728312` | `75625b3f216c503dfc13bed7a699b5707029b5c993e19def2bf6ab209efce41d` |
| `bench_sm120_layernorm` | `True` | `1233728` | `47c56a744e6f6da1208b5433e203e76d52df19568e59be13e0416de2a03ced00` |
| `bench_sm120_runtime` | `True` | `2217576` | `a1c2e0d666f2dcf7b771bd03300d6c5b6d2f9660ab8a8fcb6bb1104567db8088` |
| `train_gpt2cu` | `True` | `3045944` | `a692aa92c8efc703c86ac1333db9d29f0382f5547ac6fe7227958933b90df570` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
