# SM120 Round Manifest

- run label: `codex_sm120_round_python_stacks_selection_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_python_stacks_selection_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_python_stacks_selection_20260520`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `475`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2137216` | `d1487da47e1bc2a19ac0dc824febb522d013880ece2d180558bfe8b3a73ded86` |
| `test_attention` | `True` | `1760032` | `ce756582b6e24358ea72bb3e7f08d366d45b68d5a32d3af907b7e70b22df1e29` |
| `test_layernorm` | `True` | `1237784` | `8f10f2aea6dd0aa753d5150e9d4db1bd4f6c065be1a612645f3ec9fc8fcc9954` |
| `test_bias` | `True` | `2048616` | `7af82f8c381d181600dcf058de5036193f13ca66cc4a0e92a8f01af5dbe54ba7` |
| `test_gelu` | `True` | `1139336` | `370ccea991e3775afd93b02a197f02a801567479a091cc3e1d422c454d8cfef9` |
| `test_fused_classifier` | `True` | `1164032` | `1144b25ec2bbc36d7af113eaf24cd3cf8a9ce8854583ec4a6bb6bcfa16a2c4c9` |
| `test_encoder` | `True` | `1165512` | `1b49025754710535af19192089eeee21c518066204d1e1cdb8e4c5919bce9d3f` |
| `test_adamw` | `True` | `1138408` | `a57e7bb8d76dea7471137b16dafaa9f97e8d86009938796d549937182151bafc` |
| `test_global_norm` | `True` | `1138880` | `9900a4d53cbc2fea70f7a96f2967f4a25737d09f3572f63a9d129b2162102d47` |
| `bench_sm120_matmul` | `True` | `2373912` | `99fecb21670cae78380edf35e92ce96239bdbf743ff7cde39e2d72daeb4d2d9d` |
| `bench_sm120_attention` | `True` | `1731352` | `ef868c8893faa83c08782c79ec6ca24dde40a7e4e6aab5b603d3d8b0f2f83b4e` |
| `bench_sm120_layernorm` | `True` | `1233728` | `e6595e8fcbc28b813f5e2ddd3601561ff3b685ca83cdcf2ae9f83499877f7182` |
| `bench_sm120_runtime` | `True` | `2199032` | `67c36119600664f4ad6625b514dea92a3b9464a5e9f15d346cc7c1e39d8869b1` |
| `train_gpt2cu` | `True` | `3045944` | `a5aa7ab99c6897b19016e35e513239c80a2f55580024dd168a154075157ef139` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
