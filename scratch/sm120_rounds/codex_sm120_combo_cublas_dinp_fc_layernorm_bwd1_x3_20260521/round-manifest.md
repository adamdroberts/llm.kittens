# SM120 Round Manifest

- run label: `codex_sm120_combo_cublas_dinp_fc_layernorm_bwd1_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_cublas_dinp_fc_layernorm_bwd1_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_cublas_dinp_fc_layernorm_bwd1_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `539`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `26b29b93284b52cf97cd69d7917613d3c81ac164368af92d0897111a84da2407` |
| `test_attention` | `True` | `1760032` | `a12fc1a6a31b2e779990b8b30530dd66e85dd09956591b6c875d1913874d5033` |
| `test_layernorm` | `True` | `1237784` | `50e40995c563d3d518e818cccf0c14e4e4fb325f579161029fa1b1e38d253630` |
| `test_bias` | `True` | `2048616` | `c594f5ad7677b67923b50e37367aa94da4ce9562d035cebf5d170bf46ba4873e` |
| `test_gelu` | `True` | `1139336` | `eb0f1d4e129c938f180fda2739099d57c73a209f98c4c87a3a5bc9f839852572` |
| `test_fused_classifier` | `True` | `1164032` | `f41ec5350c7ab03d0fd64a406e1599797d45095a839256a86b8bf3a98de5c4d9` |
| `test_encoder` | `True` | `1165512` | `2bf472d0e182564a7c28b58d4edd8b9f4ec45350224faa89e8909947f9c14729` |
| `test_adamw` | `True` | `1138408` | `7beabcaf3561beb423f5dbf5802c4a82b76680bb4ad6e9504ac55bda05a33147` |
| `test_global_norm` | `True` | `1138880` | `d79eb6cc929aa4527dcfa08484366d1a0f5f8adb5b3bd09b4409b32aa627ed45` |
| `bench_sm120_matmul` | `True` | `2373912` | `292e54a3aa24809d6088d5c7d00add794e9cb507477288b3a3ae96e1c1180d56` |
| `bench_sm120_attention` | `True` | `1728312` | `ce6faf041aabe5d010533b2275e8ad22dc8979a56b0878def53b9f2a57c19dd3` |
| `bench_sm120_layernorm` | `True` | `1233728` | `a3f576ed09b9ea4cbfad3645c3b101a6423c1056ae2813c905c776a9170c3df5` |
| `bench_sm120_runtime` | `True` | `2221864` | `a94e45f668638eec65b2d3062675ddd305d8ff54fdaa3e95d33bc523a16c6c35` |
| `train_gpt2cu` | `True` | `3060032` | `709805e8e9157bae3a5ec2ca5332230c22e2548baef5ebafb96ff45ff37fb764` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
