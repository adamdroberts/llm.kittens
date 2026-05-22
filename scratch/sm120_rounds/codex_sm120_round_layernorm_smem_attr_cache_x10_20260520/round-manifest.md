# SM120 Round Manifest

- run label: `codex_sm120_round_layernorm_smem_attr_cache_x10_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_layernorm_smem_attr_cache_x10_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_layernorm_smem_attr_cache_x10_20260520`
- device arch: `SM120`
- max steps: `10`
- git commit: `0f21747`
- changed paths: `463`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2137216` | `2c8d93d6e186b09541acac89b7b3ad6cdc9e6f60f020463885455cfe56a22f51` |
| `test_attention` | `True` | `1760032` | `e6bc0e4ad660c40a50d85eca228ba7a9ec4aeee59ed6753bcf2add6ed231459e` |
| `test_layernorm` | `True` | `1238160` | `d77b654c8086cab9fd368b7791ec4e73a5b8595d3315f88838e63eb6bd69b529` |
| `test_bias` | `True` | `2048616` | `eedad7c74b983e77fbb22aa5fbea51e6792981a56aac5f9832a201f7df0f463a` |
| `test_gelu` | `True` | `1139336` | `b2bbd0217150daf243984991dc8d2b0da25102806d5dc1f9c9b44c8fabc7a23a` |
| `test_fused_classifier` | `True` | `1164032` | `e147c3a6076cb563d9a2e77496dcb5f5c7518899b802a6847cda38e6c799330e` |
| `test_encoder` | `True` | `1165512` | `30e596ac7727f88fb2109bb1542e4a26033e40126865064e398c210cc0a99e68` |
| `test_adamw` | `True` | `1138408` | `7a80cb53d6ef1e778d9b2899296290df062faeb2df3eb79f1f70e0a0edc6e26c` |
| `test_global_norm` | `True` | `1138880` | `6bcb4864ce81504df85034a3af24d57ed75c11630ca5290ba4fcffc16417554c` |
| `bench_sm120_matmul` | `True` | `2373912` | `8ba8e9a78945d193e8134a16064eeafe19fa6525310e9fa3b79af4005fe876f3` |
| `bench_sm120_attention` | `True` | `1731352` | `814f5bc3e90b614a72cd78dd66eee8278c0b1f98d9008af0b8e52ba9d561700f` |
| `bench_sm120_layernorm` | `True` | `1234096` | `0097a8666fafd84dd0d2499afcbd3ade2674bb9c9b62a99ab896aada4b02d410` |
| `bench_sm120_runtime` | `True` | `2199032` | `87c772aa9ff120575380607bdf6fbe0ffe27e45a7c95a2a8b4bc35b6e970ca2a` |
| `train_gpt2cu` | `True` | `3046312` | `a425d7023d4a6416022e313a6acdbc68ccee64d432b42e98bdac172eb4f3d39a` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
