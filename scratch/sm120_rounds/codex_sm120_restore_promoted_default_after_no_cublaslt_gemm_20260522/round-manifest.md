# SM120 Round Manifest

- run label: `codex_sm120_restore_promoted_default_after_no_cublaslt_gemm_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_no_cublaslt_gemm_20260522`
- train output dir: `log124M/5090_S_codex_sm120_restore_promoted_default_after_no_cublaslt_gemm_20260522`
- device arch: `SM120`
- max steps: `0`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `653`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `83b01b420ae3e3cfa24479edf08c6376761a20a92f92a6ab5a8ca388a759c939` |
| `test_attention` | `True` | `1800528` | `e4e43585fb10b71146403b73263b3d513b020fdb7d477b0a94f831386b1000e8` |
| `test_layernorm` | `True` | `1278296` | `45d9040abc3f0e340dd0948172caa3da6f093e979f42ffc32490ef81427d1430` |
| `test_bias` | `True` | `2089120` | `8b9aeca3c8fe80076d046431a4fb32edda7c48e5a6f0d6e1b27df9b3c721062e` |
| `test_gelu` | `True` | `1179912` | `b6e68b1c47984ac424992293b15d2e9129113e44503a3aa8e5eb5162c16fb49c` |
| `test_fused_classifier` | `True` | `1208704` | `0ea1de90820f0fe98518c1000251bf7e15fa50680760ec9dafd3b2e9ec9799a4` |
| `test_encoder` | `True` | `1210168` | `005edc4557800b71eaacba745418b2d454fa4790ae07c2a42c178eacaac6ab59` |
| `test_adamw` | `True` | `1183768` | `3c3a81d9622e2d973481be355b1b9eb13b69925c093f2467f2a7ac76e089007f` |
| `test_global_norm` | `True` | `1179464` | `f3ba89be077fce180d9079e696c242eec6bdbfc79f7dbd71324e509bb171a8aa` |
| `bench_sm120_matmul` | `True` | `2410304` | `bcb04e56677c54baa76b81f7bfc9dc00da14551aa13495e084002fed07a9d01e` |
| `bench_sm120_attention` | `True` | `1768800` | `410445705df4f15b6f463958a26c5cfc72f3f6a24e6ef569a1a927105ec15a68` |
| `bench_sm120_layernorm` | `True` | `1274232` | `635c1c23873120bb273ab0bfe2b6e9c2c359af90fa1ecfdc0c179f845efc5541` |
| `bench_sm120_runtime` | `True` | `2271168` | `658c71328cb0ba92f03c9c28d91dee0c85428378c5c8cde709b5f63ac0af41d4` |
| `train_gpt2cu` | `True` | `3105552` | `0817d8477161c7f1b829e1b42563f7038540b39844802f88f6570c7c6488b1fa` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
