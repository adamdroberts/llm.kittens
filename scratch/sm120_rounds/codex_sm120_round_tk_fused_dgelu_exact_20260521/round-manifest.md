# SM120 Round Manifest

- run label: `codex_sm120_round_tk_fused_dgelu_exact_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_tk_fused_dgelu_exact_20260521`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `485`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2154392` | `c91f645455f0548aca35b56868268c35c09c84433ac13b39d28bc3a9c2aa1d28` |
| `test_attention` | `True` | `1760032` | `a4337873c38948ebb0dadb8004e826415a7710c786e9bba91b568cc6cf28698c` |
| `test_layernorm` | `True` | `1237784` | `c6973f2d37a7a29683a8a4ffb380a75eac038fd3ddc59e526cdf442305827e74` |
| `test_bias` | `True` | `2007656` | `6b042157f1f6fab5f52b1e0fc8a3e7991d49a1f6db2cb5ec499dbd326085e78d` |
| `test_gelu` | `True` | `1139336` | `499346a84e6a4e5d6551567a4b4bb4995c2f29db2e251e005e2c33648ba74303` |
| `test_fused_classifier` | `True` | `1164032` | `99744400fc8f44e6a19c16f58118ae361df04724dbc66bc52d41f8ac75d609da` |
| `test_encoder` | `True` | `1165512` | `0c7fd63ee9d4998d7cc0f40d6930e94413156a1ed2dc2f206e15721921353b09` |
| `test_adamw` | `True` | `1138408` | `9784d4929b5021459cbca87095e1278f460e429e372f55f49017f97dd8e08212` |
| `test_global_norm` | `True` | `1138880` | `640bbac8ed039b8421343a4729941a1804bb8a9fc160f1e4c93184e1920e8227` |
| `bench_sm120_matmul` | `True` | `2332952` | `264be6c522773f4386d4b6f253a4046b02458b52de79ca175a258571322669f6` |
| `bench_sm120_attention` | `True` | `1731352` | `9e1233df2d805158976d4c47c63a9c5c7a7a0d96fbd57bb6ded4ce10ed749d08` |
| `bench_sm120_layernorm` | `True` | `1233728` | `f8137d9aba86367cece4f32d357284fbff5f0b6920e7658d9dbb45958b078445` |
| `bench_sm120_runtime` | `True` | `2180712` | `b6f078072312efb9622437158186f98aad98625969b4a38bc658204b4d9e24d4` |
| `train_gpt2cu` | `True` | `3040952` | `62324b8c19238e9308db4d273374ad47adf566f7009aabc0de5285c2c23059c6` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
