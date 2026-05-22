# SM120 Round Manifest

- run label: `codex_sm120_current_optional_refresh_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_current_optional_refresh_20260522`
- train output dir: `log124M/5090_S_codex_sm120_current_optional_refresh_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `681`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `23bc045010621e4ac01c82aae771a12e5b7668dc7cdc7454513a7dc96f526b4f` |
| `test_attention` | `True` | `1764128` | `ec5ddd80b6f75ece40c710174a2be360c956d0f7e00f6ab53b46ad9db1befb42` |
| `test_layernorm` | `True` | `1237784` | `9dcc4ccfa6975e8eabd308aa17c62eb30bc60152ff20bd4cbd598eac78b4689a` |
| `test_bias` | `True` | `2048616` | `1240dd77980ad34dd251f4e4d8f67b156a611e3d41307b5d5fd1c3efec75b2e9` |
| `test_gelu` | `True` | `1139336` | `4929ad2f65ec5ccc125e6e01d57ae84c38b9bfd68fafcbfa9511c4b3ad2f710f` |
| `test_fused_classifier` | `True` | `1164032` | `312c1c5a8fc192c093b43b45dad08cdbd21eda3b767c428b32ad514fcc7249a7` |
| `test_encoder` | `True` | `1165512` | `d8a7251fcb8656fe9e85758479d9d7abe61256d136828a8a764964412568a572` |
| `test_adamw` | `True` | `1143192` | `7788090d26c99767446c6892c95e0638398933731ba443cabb19cd5640f2e1bb` |
| `test_global_norm` | `True` | `1138880` | `aa99aa5c9176964c055d1935a1bef89bea5c2a822e4620005d2eea7989d67aab` |
| `bench_sm120_matmul` | `True` | `2373912` | `0528fb4e050608da51342dc692f81af0b33acc2e9ab7e865eec92a0905609787` |
| `bench_sm120_attention` | `True` | `1728312` | `67d258368c58e39b048ab37c3de8eb210b0f77545ab010d9c7238f4fb9484f00` |
| `bench_sm120_layernorm` | `True` | `1233728` | `6f5edd57e7df66078ce62e346c0e44acd04f05262bd94050e73252944d2b85da` |
| `bench_sm120_runtime` | `True` | `2226576` | `4789a233ab89fcb52175a0081e3580adbcf55817053a0f1b48c94e47cd0f81df` |
| `train_gpt2cu` | `True` | `3064992` | `fe5940e64b7cacce6b63f12170e4c378255799b4b6c6f95a60b79ee9ab2837d2` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
