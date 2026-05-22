# SM120 Round Manifest

- run label: `codex_sm120_disable_cuda_profiler_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_disable_cuda_profiler_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_disable_cuda_profiler_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `536`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `00437ee158fdb1c531ef7f32169cc05e813a60e40690680722416c49d59a9239` |
| `test_attention` | `True` | `1760032` | `7048fa42c97cfea5bc06638a2a8de4266161368dc9c90c8ba2441293cc88e881` |
| `test_layernorm` | `True` | `1237784` | `ac4c7d9203827cadbe06a82cfe9db8747b87f4374a7901adbef5a6845880dac4` |
| `test_bias` | `True` | `2048616` | `113caa5b8a54a667b5f70626ad17b4dd0a194d9a54bf9973dfa4c110bc4fa599` |
| `test_gelu` | `True` | `1139336` | `f71437cd419b550e01933f3805df9cb3306ae3ece5d22a847e6bbc6035b244e9` |
| `test_fused_classifier` | `True` | `1164032` | `eca8cc71dd0da5e392c3dd399142fc003ba4ba9a9462de9f8286f6e41e93508f` |
| `test_encoder` | `True` | `1165512` | `18dbe0eed37d0ee07f7d004c0570079438086a706f95cc0789e99d12bbf3e08b` |
| `test_adamw` | `True` | `1138408` | `c8e92afcd6947a57f8867f68c901ea842d21004cb290abd084f02558fa0c7bb5` |
| `test_global_norm` | `True` | `1138880` | `acee66113ba9c5f09bb6dbd07bb5b33096c3811294916aa19cb94f00c495d5bd` |
| `bench_sm120_matmul` | `True` | `2373912` | `1a439f5861f336dbe9840e3fd1bd54d08812f1a46f494f32584eee546c381489` |
| `bench_sm120_attention` | `True` | `1728312` | `0e3256e24fe2da8c2798ab2a1d5ed20238b9d11ea3069e6980790c6cba0e6034` |
| `bench_sm120_layernorm` | `True` | `1233728` | `8bdb2951a47b313fc801e4fb6f240cf05e7f8d381e0eea30181c32802674749b` |
| `bench_sm120_runtime` | `True` | `2221864` | `b8739d59e58b4b54381f7b8e64b13576e0eb251e6818178cf1db18b2ddee047a` |
| `train_gpt2cu` | `True` | `3060032` | `4070585b0c7662041fd50d0702934bcc7e434f7ccbd529630a1a0353e916a811` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
