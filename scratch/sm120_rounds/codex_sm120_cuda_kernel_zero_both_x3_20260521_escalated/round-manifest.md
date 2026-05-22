# SM120 Round Manifest

- run label: `codex_sm120_cuda_kernel_zero_both_x3_20260521_escalated`
- artifact dir: `scratch/sm120_rounds/codex_sm120_cuda_kernel_zero_both_x3_20260521_escalated`
- train output dir: `log124M/5090_S_codex_sm120_cuda_kernel_zero_both_x3_20260521_escalated`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `528`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `70a597bafcc0508ddc531d8fa3dda28e87b8846f95c491f56792f85a99f7f8f0` |
| `test_attention` | `True` | `1760032` | `77ccd675758fbf39e34abef6b023560161f12141837a541bf1705fbe58919941` |
| `test_layernorm` | `True` | `1237784` | `270c5b91a092b36d272ca1891bb3444296804dedd04523d4bfb584634a60b523` |
| `test_bias` | `True` | `2048616` | `332fbcea43d06fc1b5f0b2062ce720563ce393043bf4d9840004da1da66220b9` |
| `test_gelu` | `True` | `1139336` | `132c05e522d4f23bb21f7b2db87c985c91b8fe9d288e694e6274c9aa4abe608f` |
| `test_fused_classifier` | `True` | `1164032` | `70d3887e5c7743a4d604b7f8d3fd00184db83b1aef807ef53fd5f4b38d6f4b7f` |
| `test_encoder` | `True` | `1165512` | `5be62b1b61ccc3d434535a66cd7684ca1564644682eac597ba57723c39954ead` |
| `test_adamw` | `True` | `1138408` | `f5e382a4750fe5b8dd3d5f190cb5f3076daf98122160a7c3d2ebd78060f0b407` |
| `test_global_norm` | `True` | `1138880` | `c15eb37e6dfde1f68a968731e0c6473f1cae20fa9404fe0f48068ca86ac76523` |
| `bench_sm120_matmul` | `True` | `2373912` | `0eed84b9f7b5adea05c3f6c42588905d446362d68350e31307b6d2408719928b` |
| `bench_sm120_attention` | `True` | `1728312` | `949c4b7377c99f158319b95176e4e8b8ad50e66eb9546ae5998d443dd0b06f5d` |
| `bench_sm120_layernorm` | `True` | `1233728` | `2e7c1f54519e91d19a79d3e0d1081d05c1dcc79136b12f7278aca360e3f2a470` |
| `bench_sm120_runtime` | `True` | `2221864` | `fc52fd010c469dc6535326814bf1f621f8fb45f27a181a0607c2f95ec89a70b8` |
| `train_gpt2cu` | `True` | `3060032` | `d4911249e54485fc59d545cc296c02244210e3f6202194ffb8077c57b8949a96` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
