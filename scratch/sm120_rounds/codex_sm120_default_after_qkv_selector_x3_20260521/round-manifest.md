# SM120 Round Manifest

- run label: `codex_sm120_default_after_qkv_selector_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_default_after_qkv_selector_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_default_after_qkv_selector_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `534`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `1a09e843e711bae0428966a133997e476d85192b684d875a41e7815d58007a8d` |
| `test_attention` | `True` | `1760032` | `6656a7e2c8dcea724f23363ed6527d339f883dd3af297a570a3488daf583b194` |
| `test_layernorm` | `True` | `1237784` | `4c761b085b6a27a87766b495b42f1da1ea3b2bc82beb804e82a2ed3dfd0cadda` |
| `test_bias` | `True` | `2048616` | `30bbc93ddde87f306df7e98aa2b27fe41e64c9e1b167ac2286612c13d8008223` |
| `test_gelu` | `True` | `1139336` | `6b4ffa79176f0c5ed217ea37feb0b1afaab6d123cb7ac3bc081f1f7fabb0f85c` |
| `test_fused_classifier` | `True` | `1164032` | `401a5d6fe6578b4d542fe769f647c4ed237d5cc720fea3962b8c37b866181490` |
| `test_encoder` | `True` | `1165512` | `52466f7ff8cfb88d8a41b623523ca58eb8cd98bce3f38d376c6a1f2a6f776126` |
| `test_adamw` | `True` | `1138408` | `6cacce32e460f3a516770861dc2339de7324967531a2291629a7ebd555323a0f` |
| `test_global_norm` | `True` | `1138880` | `0aa0a48ab6cd601f09df3c2da7a2636b7e01bdd61a2ffd082c19637b4cae6ae0` |
| `bench_sm120_matmul` | `True` | `2373912` | `1a096f5bb3c51e869b89192ede26be12a731f3fae7b9d49ddd2430e0bae2935e` |
| `bench_sm120_attention` | `True` | `1728312` | `e5e1b1988b213cfdbed04db55c861588be07631bef8b7842a20da788fb84aec8` |
| `bench_sm120_layernorm` | `True` | `1233728` | `59f8e548f81a5c90c12c6f72690b2dfb4d707fe9073f8bc1714f53fe7a88256d` |
| `bench_sm120_runtime` | `True` | `2221864` | `74256d4227a47d790f91aa6099507d04b3791e48a32175a28658941a67ccf8e1` |
| `train_gpt2cu` | `True` | `3060032` | `0e6af006c46c6b3445d6706cec39b04eeba02e53933ae6fc6ea31ffbd73e9076` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
