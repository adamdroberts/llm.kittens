# SM120 Round Manifest

- run label: `codex_sm120_round_cublas_dinp_fc_only_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_cublas_dinp_fc_only_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_cublas_dinp_fc_only_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `506`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `a910a548e039af563828ec2de5b5b2dcf57437f2fa0e56700032adbbaf98b157` |
| `test_attention` | `True` | `1760032` | `0a80e424244d9c5be9469ad05842a0c29c445dc19a8730d067a501834a388e92` |
| `test_layernorm` | `True` | `1237784` | `bf00e773d904fddf69ba14b16c8138a4a02225ca7e0c1a98adda5ff482ddc025` |
| `test_bias` | `True` | `2048616` | `9ae77b724f4436f55e5023375b6b17fb756ece976e55b69a1671c1f0f2915e54` |
| `test_gelu` | `True` | `1139336` | `d5ea5fa1074cf0b2147401f41036824bc70037e4a13bf5a6d947c7a2d7ba3626` |
| `test_fused_classifier` | `True` | `1164032` | `af20f7c5570c650dbd56d8dda4b0f7383cb9b6ed9e5af91096a7c3cf8762bda1` |
| `test_encoder` | `True` | `1165512` | `3b02402a445bb640b45bd8bff28ba7a4c57fc29592d7d0b131d7f98c24306549` |
| `test_adamw` | `True` | `1138408` | `74c47ead38aceae22438b9a125f33316e981a2ba6e2ed7d9ee0c4b9d9e7e7d85` |
| `test_global_norm` | `True` | `1138880` | `0fd97e80569924b655e2c512e18db7e9ba0f4a323bfb50cc73e4c8e2b0c4049b` |
| `bench_sm120_matmul` | `True` | `2373912` | `e89e73ba75740b186840ad35f9456ac7f00ac6141dc344e266a3aee05444100b` |
| `bench_sm120_attention` | `True` | `1728312` | `04162f8d9b2f87edf741b20975dd7db425bbd06f4895deb2cba7695fda61ae1c` |
| `bench_sm120_layernorm` | `True` | `1233728` | `d95761af38a705ea4f19d3d4834a87ed50f0c0e2356c0c46cb4decae2c9bd3eb` |
| `bench_sm120_runtime` | `True` | `2221864` | `097200c58146c6113e84bfaeaa40edafa31b7ef4eb672583f756ca41651b6d11` |
| `train_gpt2cu` | `True` | `3060032` | `33ac09760c73afa4a0da6e4ad3564f252f49357bf1827f6d3866aec4e1ddc539` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
