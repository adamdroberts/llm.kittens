# SM120 Round Manifest

- run label: `codex_sm120_promoted_precompute_grad_scale_recovered_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_precompute_grad_scale_recovered_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_precompute_grad_scale_recovered_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `650`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `846411ececf8e2b92fb557cb48d666cd8384fd8bae9f5839ebedf22dc90af18c` |
| `test_attention` | `True` | `1800528` | `d13939b36d7b3c9682f3e0c6f0b44f87da9791106ff7a6fc00c1aee8769cb5c7` |
| `test_layernorm` | `True` | `1278296` | `f727334d420680e9ec941906da2ceb54f0fe969cf33a01ce06c3bc057fad3d9c` |
| `test_bias` | `True` | `2089120` | `4e4eb25cd99b502a5b0b9227cd537a6a133beafb7d936575befb8d7bcf970e8e` |
| `test_gelu` | `True` | `1179912` | `05930b53c49ccb5f946cacc167899ee19a0b3eff6aab554b14347265eb49e080` |
| `test_fused_classifier` | `True` | `1208704` | `2c4b93c5fadcf3ed8c70d4b86a074676d897d371c730c587e658ba06c6062673` |
| `test_encoder` | `True` | `1210168` | `15ee6d02270fe266784b47299e1a6ccd574b5960678a702b7b695af4ae131a8e` |
| `test_adamw` | `True` | `1183768` | `f79067587148271b4046a1e62aa95ec1c32dade1cacda034b2fffeaa4ca02b82` |
| `test_global_norm` | `True` | `1179464` | `78f06c0e08879606f557b43014b319d3c85cae0b22c6c6b81e4bc6ef1707c519` |
| `bench_sm120_matmul` | `True` | `2410304` | `9bfb95a7c846dd5eb4d60077dd27e07af8700f2c960a759d0923a20f52d80895` |
| `bench_sm120_attention` | `True` | `1768800` | `9f359248a7d8a39c17a89419c6cf929a883afd591d852aaf565cdb24f377bc8d` |
| `bench_sm120_layernorm` | `True` | `1274232` | `ea56330374b2f132a537228536de2c06695d952089e4e85b357f51fdcf27b59b` |
| `bench_sm120_runtime` | `True` | `2271168` | `04bd6decbc7b0773a1e7a61ea4d8fe946996bb8e6bc2d1641a5b5f6d4aaa83cb` |
| `train_gpt2cu` | `True` | `3119216` | `f34d6230de8052468592cd9f8a0bcb85b82e2e48103d9417c9ce7e832e53f9af` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
