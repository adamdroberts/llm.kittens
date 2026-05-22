# SM120 Round Manifest

- run label: `codex_sm120_round_after_lnfix_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_after_lnfix_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_after_lnfix_20260520`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `432`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2115600` | `139ebcd40f51a3ecde7cf909b4f512fd481e84e3a52b56af60c9094d88e97653` |
| `test_attention` | `True` | `1760032` | `090260a7063a8dc99c03915369dfa3d9a6a1f47177daec3f176448d215b0d510` |
| `test_layernorm` | `True` | `1237784` | `6aa46ac3ea8c9c32fa821a168b802d90b3a3496e6f9a52a86a5405bbf17f2c3b` |
| `test_bias` | `True` | `2039504` | `3231ce3a9498d58e5e3c36c36e48a8761f6d887ef4754549d95cecc8f741ec36` |
| `test_gelu` | `True` | `1139336` | `323879ff4abab1a92ea4c8f67722c508f07c16e253160049b98ca7cf32a1283f` |
| `test_fused_classifier` | `True` | `1146768` | `1e3cc05c42f5f5991d401894ff95c23136eed0809b0fd32d542eb90d793c68d5` |
| `test_encoder` | `True` | `1165512` | `fe298d7cedd5106b55c232c29ab236c4c66f9daf5672b5aac1fdae8bc55c76a4` |
| `test_adamw` | `True` | `1134168` | `54b15904c63f3814c8388bd46d4f64a01370b4509c780acf08c691f94de48182` |
| `test_global_norm` | `True` | `1138816` | `9285480ce10dc9e933473488622eb55f8c5d130bc39b3970d16826e350acd5bd` |
| `bench_sm120_matmul` | `True` | `2344376` | `e1df9151b152c50a77213eb3be3ad4d8f62a71f9df14d49a307aadfb79b7cd5a` |
| `bench_sm120_attention` | `True` | `1731352` | `8ae27a535f8b318429e1ee9ae624d1c6f24a5c917b65edd053e1a6050c0514f8` |
| `bench_sm120_layernorm` | `True` | `1229088` | `eb83f10ec18f4f9ab105ade82a2ccb3d610b48d2593cdca0869858b5bc37f7ef` |
| `bench_sm120_runtime` | `True` | `2168552` | `b11708e020a9ced7fdc050331b998fe6432e45d38feb457bd051ec2a67d0adfa` |
| `train_gpt2cu` | `True` | `3036536` | `d2eb493ceee749a1ad13402f15a4bad1757584b5b83f3b89d1fd324494aee15a` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
