# SM120 Round Manifest

- run label: `codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_maxconn1_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `602`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `209f2fc521a98e265c754513a73b1e0c63343f66a07c2aee1a2805f752644869` |
| `test_attention` | `True` | `1800528` | `c967abb83719830a162cbb8f558cef24a90898d537dd7f675737c62d01f89e84` |
| `test_layernorm` | `True` | `1278296` | `e15f33729adc5022c687b14d1dbee51550590cdb97ccd8aebf36f26d341ee63f` |
| `test_bias` | `True` | `2089120` | `bf5b50fa52ed52faceb5d5936b8dd84b54bcb65e075de977b9af1b33aad09ca7` |
| `test_gelu` | `True` | `1179912` | `1259fd920829a06274c2086372207ad0d7eeea1bfe5b88ab15992446e2e90d7f` |
| `test_fused_classifier` | `True` | `1208704` | `9e421343f4068f6253882a4cac0b62253a96e23cf8494f2a21b8c1de9e456b8d` |
| `test_encoder` | `True` | `1210168` | `72ff8f2c5c85d965bb29019520160c27762321b3e80c7dd3e0d5a0116069ef8b` |
| `test_adamw` | `True` | `1183768` | `79ea003deee3e9bcc3e6cdb81cc941a6805c74ea70978691e3878fbd8e138158` |
| `test_global_norm` | `True` | `1179464` | `1da7e8d5ec666e79706138900fd8097ad86ccee05f0b0e03ffe73632c8a6bad6` |
| `bench_sm120_matmul` | `True` | `2410304` | `24a018a226d851c6f174da11f44709c7945326bb8d2725544f6e90b7e1a8e5b1` |
| `bench_sm120_attention` | `True` | `1768800` | `c2acc190bef33bf0a4de48491811c9c8a3f4c18f3b2f61b4d7ac8346129d2578` |
| `bench_sm120_layernorm` | `True` | `1274232` | `f83eb3b9ac760bce9e997949d869ed2db89532860ac10a40332b5709418eac17` |
| `bench_sm120_runtime` | `True` | `2271168` | `eccf33afb0fb5c381535dfd41f6314e996dd03cd77339af935177b9b667aa116` |
| `train_gpt2cu` | `True` | `3105552` | `71ef30090893dd5d987b1473dec9701ce82f18bbd5dd878338ab983aa5131ea7` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
