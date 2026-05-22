# SM120 Round Manifest

- run label: `codex_sm120_libtorch_grad_zero_maxconn1_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_libtorch_grad_zero_maxconn1_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_libtorch_grad_zero_maxconn1_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `1`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `577`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `e4d1af6114d563ba7bf3732eebcec85770ffa8b48671cdcab3379d8541f2ef15` |
| `test_attention` | `True` | `1800528` | `88e28feec20dc71134975208b3ce3800bd632159fc1b31acf4f84856301e35d4` |
| `test_layernorm` | `True` | `1278296` | `048a43d35ea6402efec8a1a15132c5d5a8e2831853b5876451e722ae4f4a3d25` |
| `test_bias` | `True` | `2089120` | `f031f47055b918b5d9a51eb40179a723fd62c1a0c79c1311ca2d053dc945a9e2` |
| `test_gelu` | `True` | `1179912` | `f7916d7fd17cb7b03983f4874db2522c9dbb5a610cc3ad7aa292f537dc723cd9` |
| `test_fused_classifier` | `True` | `1208704` | `a0bfc2ab8c5ea916babc45b4a72e867ef5bae89caef4b444ce90e727b77f21e8` |
| `test_encoder` | `True` | `1210168` | `056c7f56c33b76abb8b37e05da192f4bcf71f4afcd0e9bfae6ba7535229a0071` |
| `test_adamw` | `True` | `1183768` | `44afd79f98a6c962b49fb2076f0277a0b63f8b1db6bf1c6527b4f786ea2a7459` |
| `test_global_norm` | `True` | `1179464` | `a0c903155bde6527bd08f844795a4e058d2e276c511dd9edcfddf30680290ecd` |
| `bench_sm120_matmul` | `True` | `2410304` | `16c7909cbf3054da1a2d386fe90789c251ef4da6632b1d033bb54e2a6d2dfbf9` |
| `bench_sm120_attention` | `True` | `1768800` | `537cbcf1e2aede34350fd382e3b8efb35e49e099375deb3c7c33af3e7a1f2ef5` |
| `bench_sm120_layernorm` | `True` | `1274232` | `1cb4afd5381859c2d14a25b664fd755daffce26438e29cd954b7f2ff685de6cc` |
| `bench_sm120_runtime` | `True` | `2271168` | `8f84ecfb2b6a71a64887134ab8768d58a7f7103befe37d4c40097a1e4340a4e3` |
| `train_gpt2cu` | `True` | `3105552` | `7bf7450eebff7c2a1c69a7caec468925682d72ca6586fea282462842b0ad1026` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
