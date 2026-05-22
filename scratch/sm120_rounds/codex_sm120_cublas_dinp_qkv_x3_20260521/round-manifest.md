# SM120 Round Manifest

- run label: `codex_sm120_cublas_dinp_qkv_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_cublas_dinp_qkv_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_cublas_dinp_qkv_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `533`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `7a12e732c6677a8d87d4d19dfca00a359bb87e86db318d0e670f062f654218e6` |
| `test_attention` | `True` | `1760032` | `88180cbb266b7b9689d85405285c13ae95b5807b377ee26c4306118c9ab89325` |
| `test_layernorm` | `True` | `1237784` | `a35d4d719c818e976069f18c23550062c8b70749ccee16aeb05616e3e40d3d70` |
| `test_bias` | `True` | `2048616` | `c68bae155a8b5f65d1d0f55785b7acebac51fb8d3ceca40c1912f030f71e6cf2` |
| `test_gelu` | `True` | `1139336` | `996c80223cc1ca589c6c25eb0132be005869716abfb1f47f495a93d30b74bd3c` |
| `test_fused_classifier` | `True` | `1164032` | `42b9d8013a08cbb33fd1d06adae3c394824e9f57753016edc028de62b5371810` |
| `test_encoder` | `True` | `1165512` | `ab7d8fb92ac1aa123dae2661a9ed0975681a533956e637da4175f5a2eeffebab` |
| `test_adamw` | `True` | `1138408` | `b272d5d35451de8553cfaa619e3b25102f5f0b309883493af7233c3afe7e3cf6` |
| `test_global_norm` | `True` | `1138880` | `c7e2ecd7d0ddb67acb52a37c64aa2c362953f8e59c6549be9106911d3ed54b55` |
| `bench_sm120_matmul` | `True` | `2373912` | `fa04ad6724d830e0922e981d0202e3efc23a83a596b52b8d4dc8024a78010dd2` |
| `bench_sm120_attention` | `True` | `1728312` | `b827bddb0265da0d0b316c59a3eaef6c63cb6e5f84e7f301342913f8be9ce570` |
| `bench_sm120_layernorm` | `True` | `1233728` | `84f26a435048ba990ffcfacc0f2e85b84ab91bb9b277c7503f54dd2340c6339f` |
| `bench_sm120_runtime` | `True` | `2221864` | `e1d42498b58df8cfbcb95d72e7544b4beb8f7bdacab46e3554f425ed325f3fd6` |
| `train_gpt2cu` | `True` | `3060032` | `1e457c4d2bebe704fc3710549786281be773ba478711f7c9984b6009dba889d1` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
