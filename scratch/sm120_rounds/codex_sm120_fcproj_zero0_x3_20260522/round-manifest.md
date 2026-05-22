# SM120 Round Manifest

- run label: `codex_sm120_fcproj_zero0_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_fcproj_zero0_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_fcproj_zero0_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `0`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `669`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `20d10b4041076d4df7880d8dd7d587fb510d24a5f5ad3fa22329ff08e16d3892` |
| `test_attention` | `True` | `1800528` | `ea8ed1ea1b1c23efe4d6d24338ac0e9bc1229b0fea801cca3875e76ebafb1b78` |
| `test_layernorm` | `True` | `1278296` | `0c9436172b1d7c7ce706dfe8318e0875bd6f214ff0b12cd57057302f2ab42bc8` |
| `test_bias` | `True` | `2089120` | `044d466ab0215ada3b3607a8c295ec569fa0b8093f08f872bad9cb92fd2b1dfa` |
| `test_gelu` | `True` | `1179912` | `c7a2675235f504bb5d0bbba9db636ee2fd2c5ed27b366b0f97d9657eec8d0da9` |
| `test_fused_classifier` | `True` | `1208704` | `9aff4d15d806f82861be38944b215bf4a65c569630bb2886269135a47fe06a27` |
| `test_encoder` | `True` | `1210168` | `57ba9b27534facb8f1be4ef93a752eb17353b63a827e7ace56de4727c90d0c5e` |
| `test_adamw` | `True` | `1183768` | `f468223ac7b6591cb3ca18f80c628ea815a43fc4cd95f9d090167e926abd41fa` |
| `test_global_norm` | `True` | `1179464` | `612f2e0715d858d6632742b2325e5e847a3c1814520fca6c5f61a8a33ee4ebaa` |
| `bench_sm120_matmul` | `True` | `2410304` | `64e2037474d87f0a685662c478e4e862fe00ea7ea9df74895d67f2ce64db5e31` |
| `bench_sm120_attention` | `True` | `1768800` | `27b50b8fdb42312ddb9dc56a194ded4baf2009bab1b5f3c366910100c5c0bc66` |
| `bench_sm120_layernorm` | `True` | `1274232` | `283a6b8a421a5a0736cd1f43bf179a47134d3724b200da99243f304b0b3aa475` |
| `bench_sm120_runtime` | `True` | `2271168` | `041398cea7302a3ee7e74b6744314bb0629cef0ee5086649a8f31dd98ec90c63` |
| `train_gpt2cu` | `True` | `3105552` | `93d4c978602275b26caee28065a0e199b95a2f700dfef540463776e7100a7e51` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
