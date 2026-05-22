# SM120 Round Manifest

- run label: `codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_block1024_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_block1024_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_block1024_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `590`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `c709834393482cb34818268008cb07e846b4eced4e657f44bb22edfcbc482016` |
| `test_attention` | `True` | `1800528` | `0a91cfe60814d64f7ce322689f0f44da71c62cbe5665d7d58ea2677490439b1d` |
| `test_layernorm` | `True` | `1278296` | `a37362d97cea8511c2a10fc704cc485f8b179751140da4ecb24bb3c6618864c3` |
| `test_bias` | `True` | `2089120` | `12da3c177fb903dd6d33661c9f1696177023048bde6d58f5b1334d5251101db9` |
| `test_gelu` | `True` | `1179912` | `fa4f0d948603cd044ebe6677288dfde256398ebbe75a167107f6e2118acabcf7` |
| `test_fused_classifier` | `True` | `1208704` | `8f9c3993cd8f3d86cde3ae3976b428adc5b57aee4eca30c452af9441baaaac08` |
| `test_encoder` | `True` | `1210168` | `29ecf51a172599d78a2de293112e24a1edd45c48f100e8189e649c681269765e` |
| `test_adamw` | `True` | `1183768` | `ca4dcb14da437d29669e593154e5b84827a8ffa545198abd7482ce167521e89b` |
| `test_global_norm` | `True` | `1179464` | `226b63c6b7d99340b63827027f39200737d0fb7100d9d840276f0559469a52ec` |
| `bench_sm120_matmul` | `True` | `2410304` | `286628e63dc82c84f952f9d56333ecc785c8e3a76120c23036a24b5902686b1e` |
| `bench_sm120_attention` | `True` | `1768800` | `4e2661482d4b85e06626bff96f332ae049878eb0e863bdcf9e08acbd5d0f0f78` |
| `bench_sm120_layernorm` | `True` | `1274232` | `b939d635c075071ea081edd56ab111aaf17936c3a6082060dc14dfd4c296d8ee` |
| `bench_sm120_runtime` | `True` | `2271168` | `8c5931156a138d2f955b3f9f78b754b1d3e9bbe993c55b25d792ff166a2427ef` |
| `train_gpt2cu` | `True` | `3105552` | `4cc977ab1a5f271b262aa14a384cd036b68dc953d524e3f90839d1e5f1a7098c` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
