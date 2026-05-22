# SM120 Round Manifest

- run label: `codex_sm120_round_tk_qkv_forward_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_tk_qkv_forward_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_tk_qkv_forward_20260520`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `451`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2124040` | `d4b1273f2107f456a35b39408f0183c8db21a238d37a5278a7d0405fc2eaec88` |
| `test_attention` | `True` | `1760032` | `41eea6e1b301152393599e700fbd68a692ee12f80ce2eb8975bc72f15dbadaa6` |
| `test_layernorm` | `True` | `1237784` | `8a73070725686958451d81276c1b842c4b0369c0a88d390c220f4dbf6680b209` |
| `test_bias` | `True` | `2039504` | `6461b12db79ab30cd50fe88c7d15020bf71efae0c8b947d370a69f5e66da5734` |
| `test_gelu` | `True` | `1139336` | `96c176b6f73e48ed737bb7edb9729fbc39dcf9e1d1f82fb5c0278fdbf7947f83` |
| `test_fused_classifier` | `True` | `1146768` | `ea4b64bea245fbdfcbd194c8d31160db4dbc6cba6fe5cbbec77f5db3714e8a14` |
| `test_encoder` | `True` | `1165512` | `e33fe5174e2f3ce3ab62421b3852a3f3e85429b426a3281a6a33dd25c7d0a5bb` |
| `test_adamw` | `True` | `1138408` | `f6eef26bcdcee658be8ae434c5c1f7dff59154aff9a9104386878b6f0cf369b6` |
| `test_global_norm` | `True` | `1138816` | `1c7a4a20c746ea3094adeea37ab3adc97bbadb4000d61e3c0061b512e10c9d67` |
| `bench_sm120_matmul` | `True` | `2365064` | `6b679e48ef18c4dba3128dbc2195b98a53c4ab6c1192de6319a6daa21193d818` |
| `bench_sm120_attention` | `True` | `1731352` | `906406e53877e971c981fb07c14d40404b4d23e0b53ed8578891ed7a8534a6d6` |
| `bench_sm120_layernorm` | `True` | `1229088` | `a8dc3883c30e4f3cc591748521d45a0352e1680080d193549f2713df76a8d0c1` |
| `bench_sm120_runtime` | `True` | `2168552` | `0e9cfea092b3e789896dfa5ea01eac8ebd69c265927b10c9e29c626119deae7f` |
| `train_gpt2cu` | `True` | `3045632` | `1d06e1d85ae30d7ad2931a6c6c1831465166365a37a259b39fd8461fbbb98d7f` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
