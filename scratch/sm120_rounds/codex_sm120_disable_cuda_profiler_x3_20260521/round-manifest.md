# SM120 Round Manifest

- run label: `codex_sm120_disable_cuda_profiler_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_disable_cuda_profiler_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_disable_cuda_profiler_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `532`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `eca7708dea74ea591d8868100f153eea1aea0aaeef741aa17c96931ba804ca6a` |
| `test_attention` | `True` | `1760032` | `313c9975fdde2e4aa5cc8ce66706362a1262105619186914b07c5939366baa82` |
| `test_layernorm` | `True` | `1237784` | `41ebe1675138b4fad417a90116bc7af9f73059caeda77aa01a138f33857eeba6` |
| `test_bias` | `True` | `2048616` | `553e964564e6763dcdaebc92742372c014a44c17db1c030e16fcf3364c4905c1` |
| `test_gelu` | `True` | `1139336` | `c8f221291a9cb4fe577ce4bda546bb92199380c369538707c0178d21d0105653` |
| `test_fused_classifier` | `True` | `1164032` | `c05b050aafcb85d4a9deb52befdfbbe726b72f20662aed548209844016c00741` |
| `test_encoder` | `True` | `1165512` | `b949f2884f47954b7dc32fbae59e1855379ece68a23586c6cb6c0bce80b7cc2e` |
| `test_adamw` | `True` | `1138408` | `7bd3c0d2e1ac8282527bc86e6b4e70d4186e1381b85e4e85b82cbb9c0c7e1455` |
| `test_global_norm` | `True` | `1138880` | `d5a5e4b592dfd0d21281d31f2389e34b22d10029dfcd7cd4e8b8ac01b339ad08` |
| `bench_sm120_matmul` | `True` | `2373912` | `25190170ec1b20bdaf58530d05b156d40c97de4ed0e89c8677e18488c41e8d15` |
| `bench_sm120_attention` | `True` | `1728312` | `ef52139e1fe5616485b137d04d1c543687224469870cf098508a37ef7b9d5e59` |
| `bench_sm120_layernorm` | `True` | `1233728` | `19ca96429a23fbc048a1352b47bcb6205ccaa12ce5a6a82f3a49893a37fe4e43` |
| `bench_sm120_runtime` | `True` | `2221864` | `b1a4cba6c3663f16c458f5bd191e70f5dba42460b1d00f8a3fea9ec12fe5014b` |
| `train_gpt2cu` | `True` | `3060032` | `2e1ea067f8417886bd4feb10c2434993983c06795033e9e3c6566d2dfd539eef` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
