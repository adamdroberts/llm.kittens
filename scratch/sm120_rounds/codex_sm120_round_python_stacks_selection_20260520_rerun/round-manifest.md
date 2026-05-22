# SM120 Round Manifest

- run label: `codex_sm120_round_python_stacks_selection_20260520_rerun`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520_rerun`
- train output dir: `log124M/5090_S_codex_sm120_round_python_stacks_selection_20260520_rerun`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `475`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2137216` | `c8b1da7b015abf56345d60beb8a2757ec4cebfbb02911dc04cccd6c0820d7b2c` |
| `test_attention` | `True` | `1760032` | `70a98c37b79729a2d6d04c313bdfe8702b0fb0b5b190594467cca22027419182` |
| `test_layernorm` | `True` | `1237784` | `0e9ead31d64a60fe23aefa5813b49dcfaaf74e76e9b8c02f5c7940ef08dced8a` |
| `test_bias` | `True` | `2048616` | `83b5b485978b5bfbaf5afc2e922ce2338ba02819573c4c4d075d1100a3d2d7fc` |
| `test_gelu` | `True` | `1139336` | `7e239e7461407e5cd857e72ee6c10e5d976cd231a67f414b1f9eb8b20f0c0aeb` |
| `test_fused_classifier` | `True` | `1164032` | `d8ac7b0df0be66a3e56cacb0d74b9ead6fcad386b1c88cf77fc17e517c3bdf9f` |
| `test_encoder` | `True` | `1165512` | `aff7925b027d75c481661229a7c2eb7be645bdffc762c96e22cbcac6c79cbe4d` |
| `test_adamw` | `True` | `1138408` | `41a1ef1ca9c0350bccd79b63f0e0dd8f6427ee2b874c57ff387aae60e4065144` |
| `test_global_norm` | `True` | `1138880` | `0e706a8b4a26f43c70b36b365999e4cddd3fa99ffbadaa6b719515a0125279d9` |
| `bench_sm120_matmul` | `True` | `2373912` | `3254c638868f775825ce7b2502e53beaf3ed71aba517d081fba2fc233463a491` |
| `bench_sm120_attention` | `True` | `1731352` | `d61629f50e7b991d4a5d436303cd01aa03821a34c4280399790cddac7a393d4e` |
| `bench_sm120_layernorm` | `True` | `1233728` | `7be7952bbb71626e5fdddfa676612c49f7c5562ce23a44e3074ec87de65f64b5` |
| `bench_sm120_runtime` | `True` | `2199032` | `cd565726acd5d46b0a7ffa5bc5730b88344b28ab0ae2ea1361a84efacdffb393` |
| `train_gpt2cu` | `True` | `3045944` | `56fd1af46c8bfb4732913a1c10b380f37334104d24293af24b7922d6bd0e22f0` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
