# SM120 Round Manifest

- run label: `codex_sm120_combo_cuda_zero_block1024_layernorm_bwd1_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_cuda_zero_block1024_layernorm_bwd1_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_cuda_zero_block1024_layernorm_bwd1_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `548`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `15a6300e9b3ec727014586334faf8cff06559687675d06e3a2c4986034f88e3c` |
| `test_attention` | `True` | `1760032` | `9fe53894552a616b41c2cea555ff6325c491cb000c910527474413537a55115b` |
| `test_layernorm` | `True` | `1237784` | `185b25b78b056632508e821391bfb6a727c040a3c948f0bdcf322cdb14959a18` |
| `test_bias` | `True` | `2048616` | `c6a762db6593d6aaa778a5e324edf77f375a05a3b4f48ac203fe35a7ffd4162f` |
| `test_gelu` | `True` | `1139336` | `7304897f8acb63beb71a2acd0890ffaa49e030cad3f8125b328662c68914fdee` |
| `test_fused_classifier` | `True` | `1164032` | `e629f72e9777ba9d82309a2d4c00657e9821c0ee960296864eb15e6c4ccb76ac` |
| `test_encoder` | `True` | `1165512` | `757006d31706800c67507f72098dd2fe64eb4dab3090960b1229128c9954bf76` |
| `test_adamw` | `True` | `1143192` | `180a081a7377902e97c16a672c77c9bf38eb12d374e621df8592f67e8188f1be` |
| `test_global_norm` | `True` | `1138880` | `408477d97e9ae6d422530d9fc8bd3338e76979a4aae86f4ca3e927d6c09dab9c` |
| `bench_sm120_matmul` | `True` | `2373912` | `dd65a31132a890bf8ccefd894836f7aaccf028cbdab9847e9cb160539ca967dc` |
| `bench_sm120_attention` | `True` | `1728312` | `968b4654f223182ae71d16ad89d9cf621e0cc550551b06c10d2aca76999d4dd7` |
| `bench_sm120_layernorm` | `True` | `1233728` | `e3fc2f532f7dec0db1aa96f8b01b7bb840ad477d090f5aec2aee87cb417ba152` |
| `bench_sm120_runtime` | `True` | `2226576` | `b5012f481848f036299ff9e2eeef7b39a3a21559567a10ccda6ff3821da50cbc` |
| `train_gpt2cu` | `True` | `3064992` | `a4a0f01acdaf453daebd9ca2184393f0297ddcfdd5c6ad8b09afc4ded2324cde` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
