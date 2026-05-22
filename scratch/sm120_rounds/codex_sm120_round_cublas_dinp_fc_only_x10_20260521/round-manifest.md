# SM120 Round Manifest

- run label: `codex_sm120_round_cublas_dinp_fc_only_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_cublas_dinp_fc_only_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_cublas_dinp_fc_only_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `507`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `619e0a9e782c5a9010b27c6e5b66378e9ec2b7b5a97f93e1cbdbe13f6634f5fa` |
| `test_attention` | `True` | `1760032` | `c9c8edcea3dfce38f640bd371b7bf2ad5dfa309de91704a04abfceefa58f5b19` |
| `test_layernorm` | `True` | `1237784` | `63e306151b4c873f9cf493b947f5accbb26056b44c241e02f248b499876a4630` |
| `test_bias` | `True` | `2048616` | `b597d817add6a0ccc900e8ca86d86aa0774f0bb486488749c8835672bf40f3f6` |
| `test_gelu` | `True` | `1139336` | `a80fca3e88b2af5836812fc03d819cd779bd8531f27164f1172e750ad0a0e481` |
| `test_fused_classifier` | `True` | `1164032` | `0e07efc3c6d6c12175cd22ad265e01122bea229f4d144898e7b403c99b7239f0` |
| `test_encoder` | `True` | `1165512` | `a09ea1a633ec4c811f5e485b4d6c45d97816cf1359e204ef0a1ca6b82e77fa0c` |
| `test_adamw` | `True` | `1138408` | `ce57ad0b2e7ad61aa89bdff68923d178451bd99d23839ea8e1f242a0c349e582` |
| `test_global_norm` | `True` | `1138880` | `615b48015b9ee4dd889b1954fb5364aecef8eca170aa284c75d66ca990b8bb0b` |
| `bench_sm120_matmul` | `True` | `2373912` | `756128bbe612762da37c9a848f3c390c17ddc91cce560d26670010489973da5c` |
| `bench_sm120_attention` | `True` | `1728312` | `aff8900b87e355820244d7d8764a7b994fb185f9a27371c8c5528b1316369299` |
| `bench_sm120_layernorm` | `True` | `1233728` | `cc96f83154cadb5342884da604dbd2fd48f619bf146d954259938b88e0fc0632` |
| `bench_sm120_runtime` | `True` | `2221864` | `2722e4cb7aef271f5419125fc1cc6f9a6c33534bad8eea76472a2f6ea1372c54` |
| `train_gpt2cu` | `True` | `3060032` | `81ebc5f77653f22d5fb6a19b0016e5b87a1822f3c892513916decc4483cf4043` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
