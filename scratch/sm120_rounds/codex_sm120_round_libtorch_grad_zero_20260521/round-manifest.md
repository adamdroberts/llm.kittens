# SM120 Round Manifest

- run label: `codex_sm120_round_libtorch_grad_zero_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_libtorch_grad_zero_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_libtorch_grad_zero_20260521`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `501`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2185024` | `69929faef884994c3ce050da748e647fd776d004d4f861140b3352dc371ff788` |
| `test_attention` | `True` | `1807832` | `6d22f824ce3aaca35d9517eabcb675d4764833ab40b683c744cd55295d12d645` |
| `test_layernorm` | `True` | `1285600` | `ed893f2441ee29cc137583f436c1f637de501424160651adca60a384ab1b0548` |
| `test_bias` | `True` | `2096424` | `f5b7d6616f332e177ce7f5ae2c02984406de25ba6dbbcd9426094d26ea33c66b` |
| `test_gelu` | `True` | `1183120` | `db3743270d9934e4b70ebede6faee2a9c79e69ca9c0d23bb5030325ac9519e9b` |
| `test_fused_classifier` | `True` | `1207816` | `76f8266633128f9755ae7c8ae9a3780f39170424676392384515085be0974338` |
| `test_encoder` | `True` | `1213376` | `6f2b035460fbbd44823ee3d0b08aeb11ce8aacabe6f6327fb28def0c4ec76e19` |
| `test_adamw` | `True` | `1186288` | `9c61b4429b371b9db630babe7f5aa6a6d9f14d88087f3d03c164ba8e19eebeed` |
| `test_global_norm` | `True` | `1186760` | `8fef66eb295025aaad65f4c6eec3274a6828278ead58c12b0540a229b933c46c` |
| `bench_sm120_matmul` | `True` | `2421704` | `a5aac27754d41663b32f4327e465b9796b2555329e241e327bbe66ce814b08f4` |
| `bench_sm120_attention` | `True` | `1776104` | `fbbd21aa3e0ad6d0edc8166882bef62986cb6df4cc7b9001d3e198ecfbaa302b` |
| `bench_sm120_layernorm` | `True` | `1281536` | `f81c3f39f92496c9ad1fd42a3d77d8c68106053aa508cfe6cc61832410b26c5f` |
| `bench_sm120_runtime` | `True` | `2269656` | `3196632c3e06e1c7cfe555ba10e5151f601c9c8c0eefbbc9b3badb66cc77cd1e` |
| `train_gpt2cu` | `True` | `3111984` | `400eccda6fcb822a0e1bae02b81c9bfdeabea0b69cc884ad820a269cfb8135db` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
