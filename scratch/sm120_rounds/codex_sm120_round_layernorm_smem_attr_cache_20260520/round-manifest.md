# SM120 Round Manifest

- run label: `codex_sm120_round_layernorm_smem_attr_cache_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_layernorm_smem_attr_cache_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_layernorm_smem_attr_cache_20260520`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `462`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2137216` | `00c456360f733734d8a2ec786be75d60493fab3e42b24dab369cb1e6bbfe3a36` |
| `test_attention` | `True` | `1760032` | `40621880f099232d6fef22aa9be76e1683d88618be5e3fa61a527452f889545b` |
| `test_layernorm` | `True` | `1238160` | `8837ca96d8b409f72d95bca01b49ad8b75457d376bc3b9d7d7b3f5ec6cdb2b3e` |
| `test_bias` | `True` | `2048616` | `4626bebf048c556f6324b9cf4ed494bdfff66f678dfa57622e6c5572f62ba46f` |
| `test_gelu` | `True` | `1139336` | `83506969cb74750c9ac59b9f436381b63909cc64371493c3a3db0c484a1b013c` |
| `test_fused_classifier` | `True` | `1164032` | `828a9163b9477a2133155e0b72d466ca04af2c3b248aa267b3cf0225c6d0b40e` |
| `test_encoder` | `True` | `1165512` | `e09aa476110afd6fa23fe4fdc2e44fe961236464da327690a2ded5166c092492` |
| `test_adamw` | `True` | `1138408` | `e3ceb00df7c49667c6998b48913e7e2070e1905ac59b6be5a63a3c69b8e80880` |
| `test_global_norm` | `True` | `1138880` | `3a456d0e4a9124968dc9724bf6b77e45c4d39d835c718f75ab80723c83cd8611` |
| `bench_sm120_matmul` | `True` | `2373912` | `9ba871c21df676deacedb37c88a094144c14f4c33d30cfe24be2fdfd20d61b44` |
| `bench_sm120_attention` | `True` | `1731352` | `db87a840e31423c15424cd63d9bc45923147e27b9943f9c91395143b4cb33f87` |
| `bench_sm120_layernorm` | `True` | `1234096` | `351a991a9e6219053937ecfb282a3e6e4738b2d243faca1f3383aee05d1f6b16` |
| `bench_sm120_runtime` | `True` | `2199032` | `2f8326d7644d8b5c00bdcaa005e45ab0d8669dcb028a72e74600aa16de8852a7` |
| `train_gpt2cu` | `True` | `3046312` | `764ec3740ad87f7209b0cfdae8c3e516633a960f9cb7c364220611ba10eca4a1` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
