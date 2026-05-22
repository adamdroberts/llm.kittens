# SM120 Round Manifest

- run label: `codex_sm120_promoted_tk_dgelu_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_tk_dgelu_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_promoted_tk_dgelu_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `613`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2190792` | `514a369ab80f0a45a73286aaf54627d5de749a4b2efaf5d4db978f0e6ad009a6` |
| `test_attention` | `True` | `1800528` | `5c96e4b78ed07b065cd2d72032794b92515c447c3ac24c86e805e79045ad3d9c` |
| `test_layernorm` | `True` | `1278296` | `241894f4464866510f96c8fa4f6026ffadad6947ce00a64a306a8e054aa4f172` |
| `test_bias` | `True` | `2052256` | `4edf6dce8a32908e74aacfb503dd04e5e0cc0bfd08a29a8e16eed99ed6897787` |
| `test_gelu` | `True` | `1179912` | `539a2da05a2a07cbe488ed9b37958ed601a1aba1bed7401745ccc2cd2fd31333` |
| `test_fused_classifier` | `True` | `1208704` | `d6a04339b2506e493e6ed64a8678a644baa17548e2448150752cdacece7d826d` |
| `test_encoder` | `True` | `1210168` | `c79005f86d91c98602047886a1a41b9ff05720e8a211a95c80ef1390c3c2db5b` |
| `test_adamw` | `True` | `1183768` | `6d3617c7abf31b755af18f2308c3a5a11bfd76bb518877e7da14adb779976a87` |
| `test_global_norm` | `True` | `1179464` | `31c2f903eebf527415d10cc7aedcc591ac3ee3961f9e946346942a328f5ff0a6` |
| `bench_sm120_matmul` | `True` | `2369344` | `98c4598b43b7a328473aa3f4161a0762f1663d38f46e33684683fd264edc2ee6` |
| `bench_sm120_attention` | `True` | `1768800` | `b5dd170c95042e714e743023d37eca51e378385ca18a424910573e0964ffa9b5` |
| `bench_sm120_layernorm` | `True` | `1274232` | `9ec284e9d8da355e2bff5b299032e75c8a44991b53a0fc491cde88047a21f15d` |
| `bench_sm120_runtime` | `True` | `2234304` | `80ef8f4ec9aeee4f2bf6214bf034ce3c9e32688bb192837589930ad53da5d585` |
| `train_gpt2cu` | `True` | `3104616` | `8d32b012b4cb1044a27347ba8a723bea51876315e0fde2957d924c60ad3e7274` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
