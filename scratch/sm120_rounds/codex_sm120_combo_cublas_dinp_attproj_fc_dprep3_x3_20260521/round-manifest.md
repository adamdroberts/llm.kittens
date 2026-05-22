# SM120 Round Manifest

- run label: `codex_sm120_combo_cublas_dinp_attproj_fc_dprep3_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_cublas_dinp_attproj_fc_dprep3_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_cublas_dinp_attproj_fc_dprep3_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `583`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `15c56788d4222b2be163e4091820f9668c5f1f13fba75f4c26a95827990452f6` |
| `test_attention` | `True` | `1764128` | `2ea533b2c3eb56be32c4ea867873832e977a22a951246f3fb186221726007280` |
| `test_layernorm` | `True` | `1237784` | `73a3cd4c789658e891b6f13bddde69bfb97f437fa84d82b69f365ff457087aad` |
| `test_bias` | `True` | `2048616` | `9491e379a5d2beee815d831138917d843a709d51ea3d929c8d307601ddbd89c5` |
| `test_gelu` | `True` | `1139336` | `a44b47cd7a3d29f510d9030068a1df0a2f8f4feef7836719eec40f5fce2c9c3e` |
| `test_fused_classifier` | `True` | `1164032` | `101963d23fc39ca13c15aa0c61e8df55fd5544fe74a1ad11cb3358b5946a4618` |
| `test_encoder` | `True` | `1165512` | `74a8489f6e3e14c27a4720054c8573b6f70118cdae854d04d62d0af422794e53` |
| `test_adamw` | `True` | `1143192` | `8c04111d8ba1cc2f5f9ac1e88c0760d3575fbcecbc09abaa8669d5c26bdb1819` |
| `test_global_norm` | `True` | `1138880` | `6e0ed8ec4079f9ded559415ff609b89535e78797e802f467b4a9844822b23b0f` |
| `bench_sm120_matmul` | `True` | `2373912` | `a860ba92a5823a9377a2cd76b706fc879dd5e9ad74fefac0a90f292bd3d130f4` |
| `bench_sm120_attention` | `True` | `1728312` | `1d5db51b6e039f92ca38f3148dbaf613e547567f178646084fdb71a662b383b5` |
| `bench_sm120_layernorm` | `True` | `1233728` | `bae65a294d2d3ff914966077f416123ac1bd3cc038cb89350fad7ea32b955aa4` |
| `bench_sm120_runtime` | `True` | `2226576` | `adf6c2e260d6ddb638a0ec89321537b3ad9c1f270536edf90826290aee98742d` |
| `train_gpt2cu` | `True` | `3064992` | `9ea79da33b882d1aafe889fed0a30c7ac5e665090bac1877986738f80cdc15d4` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
