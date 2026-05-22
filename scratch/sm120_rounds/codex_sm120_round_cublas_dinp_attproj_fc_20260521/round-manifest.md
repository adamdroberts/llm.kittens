# SM120 Round Manifest

- run label: `codex_sm120_round_cublas_dinp_attproj_fc_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_cublas_dinp_attproj_fc_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_cublas_dinp_attproj_fc_20260521`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `480`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2141112` | `846c20461fa983ebf03015c1bad810c9f6b2ccce5ab86986d6ef7b9775eb2bd2` |
| `test_attention` | `True` | `1760032` | `d6e5962c30f78e2237302819f319581d5475f6c5bd14b2cd5630de7c65b2f184` |
| `test_layernorm` | `True` | `1237784` | `7cd5728fdaf0c3ea77c2a5d1b6d79c181e49a59266e7052610bb6ead3f678138` |
| `test_bias` | `True` | `2048616` | `5b88d646b2417fb980a76429343ee0415ba932114ad78668a5154aef26eafa6f` |
| `test_gelu` | `True` | `1139336` | `59b59fd6c6c34330520bad1d0200f6d0ef8ee3d393710d338cf4e7c9ac373aa6` |
| `test_fused_classifier` | `True` | `1164032` | `8a08d3b9600d73f4095b4428874d051cf4416971a28bce16782ee0095c7438ad` |
| `test_encoder` | `True` | `1165512` | `59a3041caefd1e5343b7280e15f0de2cedb2d33eca533a7c08eda9d48d07b648` |
| `test_adamw` | `True` | `1138408` | `ee98653b2a57fda76450e81e5ab0b718589658462f0907df2f50b1e25c8dc88e` |
| `test_global_norm` | `True` | `1138880` | `e2abc079a22f50c88279c5f0a12db15239dbeb7f72aafc244ecb943e4240e73c` |
| `bench_sm120_matmul` | `True` | `2373912` | `d728e76238729dc01de77ac7e1831735836edb003abac6ffa20cb664b7d14906` |
| `bench_sm120_attention` | `True` | `1731352` | `7c0a184aec750b4d8c75c002ed8fdc1569ef5dce701eeb243b75301e4c719473` |
| `bench_sm120_layernorm` | `True` | `1233728` | `18cd936ee47476c17f21642726eac657a66b9f0fdc889769868b1a5d77655f71` |
| `bench_sm120_runtime` | `True` | `2217576` | `f2623653da847ad400aa2003aaada1f1e20bfe7fa6116c9381af904a9fc77034` |
| `train_gpt2cu` | `True` | `3045944` | `0f399f030f2fe4aa5b6c9a2faf3fd17d557e213e43bacff20249410f6a87be4f` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
