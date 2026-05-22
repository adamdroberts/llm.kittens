# SM120 Round Manifest

- run label: `codex_sm120_async_grad_norm_copy_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_async_grad_norm_copy_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_async_grad_norm_copy_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `544`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `3404af205ca0a4ef8f884218be8a1aff87e21a18ca53214a424fe0d1dbcdcb2d` |
| `test_attention` | `True` | `1760032` | `9fc561a1d2c864bbee905f24c708392978f334c59614d930667d345ac068cd5b` |
| `test_layernorm` | `True` | `1237784` | `8d8392410da212975c0f73bd499a0c379c12509ac6d1fd5fc878b46edd16cf26` |
| `test_bias` | `True` | `2048616` | `a8e9725681e373ba80245e008146072f2d6d806b0d5c4f3dc9c0be584442eed5` |
| `test_gelu` | `True` | `1139336` | `97965713068431806c28709b9a24fe7b9c118a541427bdfce958eb81acfd98a2` |
| `test_fused_classifier` | `True` | `1164032` | `37170fad7b5b095bdf35a8c20bbb39a9e6aef7590b7a0cde0c90e2e71d900637` |
| `test_encoder` | `True` | `1165512` | `ef9483e4fe4b8cac14b0b89efff3b4d72c8e22c5fc93bc0ec0ac491be8d45cb9` |
| `test_adamw` | `True` | `1138408` | `62da3f68fce69d00c9d7f1237ded4983a67995fdf450e7381217dde1bc7f6b81` |
| `test_global_norm` | `True` | `1138880` | `738eb26e8df04c79ee88a8c2ca539b7321e34e70b97a9313adbdff009ce694a8` |
| `bench_sm120_matmul` | `True` | `2373912` | `7d69bcbfcdea1e4bea3282e1b31b4076b67dde8f6f8eed7fb858c09ac229e9cc` |
| `bench_sm120_attention` | `True` | `1728312` | `eeef6c2b713850ef37424fecec014b2921abf0c7500d67d854e67631efdfc359` |
| `bench_sm120_layernorm` | `True` | `1233728` | `a82c023a001e6b555b8fc591135ec9b410d1ebf69d881d506c6bc38edaa8711a` |
| `bench_sm120_runtime` | `True` | `2221864` | `fa502ab371ae4b6a4f0e74f7702d42c661377e3db95ac9042be24d70692bb187` |
| `train_gpt2cu` | `True` | `3060032` | `9e1de7a3de3d049c467223bc9f181a37f654fff36e11d04677a28a58f38a64e3` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
