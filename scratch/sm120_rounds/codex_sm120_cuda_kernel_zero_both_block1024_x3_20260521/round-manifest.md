# SM120 Round Manifest

- run label: `codex_sm120_cuda_kernel_zero_both_block1024_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_cuda_kernel_zero_both_block1024_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_cuda_kernel_zero_both_block1024_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `530`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `a010ee5d768d40035d69af24d0b8330ce597511bc91993a4313d7a90277402ad` |
| `test_attention` | `True` | `1760032` | `a28018cc37f1d58ec8926608cb86b485176d45bcf3549aa5a7e7245363f90d18` |
| `test_layernorm` | `True` | `1237784` | `645f5ae168986431c221e6a02118a99b1ca1e1931e8f07ad1929911723219930` |
| `test_bias` | `True` | `2048616` | `3c86bef33a92aca63c42905e5a9faf0af04082df444f8ccb9fdd29134bf0be32` |
| `test_gelu` | `True` | `1139336` | `edb150bc164de8761a177435cd27eebaa062e78c2ce42b48fcb05e0e43877c86` |
| `test_fused_classifier` | `True` | `1164032` | `4f26f7da4efcb63943eb054abffe28280420640de7471c214a26dfe5e2d0531c` |
| `test_encoder` | `True` | `1165512` | `c5cb814a1f0b60478f5a33d3ffac39c941d745da2865004bff41c602606d8fc3` |
| `test_adamw` | `True` | `1138408` | `571908d1e1ba8bce75a9a8b641935ba06d27e9f8ee1c49a65b54b76c7f156217` |
| `test_global_norm` | `True` | `1138880` | `cc1a2a2892677ab458d7f6c034d1e6b299388e2129a9b0d370155f47b28eb98c` |
| `bench_sm120_matmul` | `True` | `2373912` | `107965fcb5039230c512fb90d180facec801e0aaa5914937d7993a18fe3e1da2` |
| `bench_sm120_attention` | `True` | `1728312` | `a3b753fcf8dc0ddc793843a3c1f58b585cedbdbccdb7e1049315d0b458ef1b18` |
| `bench_sm120_layernorm` | `True` | `1233728` | `87ce2d5ad743581b57c3a8a2cc64ca908d6066a9e3deecdbd830e01c303b46e4` |
| `bench_sm120_runtime` | `True` | `2221864` | `8ab15813b9ac2112bd0409a20ad59c97992a06b3c1ced0d5c67ada9628d7c1db` |
| `train_gpt2cu` | `True` | `3060032` | `3d94ab172a10827a52ae71183d98ea8d208eb4297f555682b22a0d6df073a891` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
