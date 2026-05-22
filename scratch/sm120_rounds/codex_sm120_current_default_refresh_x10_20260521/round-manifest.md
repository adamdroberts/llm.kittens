# SM120 Round Manifest

- run label: `codex_sm120_current_default_refresh_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_current_default_refresh_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_current_default_refresh_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `540`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `b170b56ce81b6a7fc4cc625703d1a233536792d01b8a194eb8e2a04f67e0a580` |
| `test_attention` | `True` | `1760032` | `d4c2eceeb2bd1434d25ba3ceacf6ddb2b61db0378748b707bf3c34d7dba63e91` |
| `test_layernorm` | `True` | `1237784` | `f9897100dc23576fde36d026a18b42d593171cbc7d8d6f9a75644868d39eed74` |
| `test_bias` | `True` | `2048616` | `c48330b5a592b12372df097ca14e4c62c72bafd66454f81054eca34fa026203d` |
| `test_gelu` | `True` | `1139336` | `4a9f7708c5f9361954a5090f28cab1230a8220bf56f1ded455b49f0f1c71f0b5` |
| `test_fused_classifier` | `True` | `1164032` | `ef49b3da01630265dcff98b560c6937a3bab3b9e551cd0f7b3d143e159b8f62c` |
| `test_encoder` | `True` | `1165512` | `b9d99a0bdba02a881f805511f7477f170450f65f3dbc8f53c230b0f54a71edee` |
| `test_adamw` | `True` | `1138408` | `a6a27f457871b8d3406667a5f0c3e745ae1e9ef423cbc1e2e71a04d9d67274d1` |
| `test_global_norm` | `True` | `1138880` | `558ad5ad0f685dd5998fda5cf00c588483ff1bd304ccbc53ab785f604f9f963f` |
| `bench_sm120_matmul` | `True` | `2373912` | `9e25032d9752ddd41bbe79c65ecb56fcce63de5039dab58a0ee9460eec5a5997` |
| `bench_sm120_attention` | `True` | `1728312` | `15ef794fe26f590505d8405cce489e9fc50798f803ed13da94ec002751997e23` |
| `bench_sm120_layernorm` | `True` | `1233728` | `f48cfb6b77744f788d2fc46e0838c8f01139508c68f23d515eb1dbfccf93b041` |
| `bench_sm120_runtime` | `True` | `2221864` | `d871d3866581d8de80e6ae9a97cfcc48600183ec455cfb4484bb84ae32ef86ef` |
| `train_gpt2cu` | `True` | `3060032` | `20979966114e4f6ac98511099115bb40f331ae844c07b7e6beb702a502167a4d` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
