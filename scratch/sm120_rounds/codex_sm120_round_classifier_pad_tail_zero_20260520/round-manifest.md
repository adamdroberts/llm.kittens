# SM120 Round Manifest

- run label: `codex_sm120_round_classifier_pad_tail_zero_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_classifier_pad_tail_zero_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_classifier_pad_tail_zero_20260520`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `464`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2137216` | `1bf2e79093b4c0196ece8474c96ddb397b57c02e3e88eeceaf633fc94fdcd0de` |
| `test_attention` | `True` | `1760032` | `e1a8a1cf252a1257e52aac7033394b0db7ecf60461363594ad3c3bcd44eff82d` |
| `test_layernorm` | `True` | `1237784` | `e26251d638b414ec0a29465da04544360d376fffedb98fe430f3734752ecb247` |
| `test_bias` | `True` | `2048616` | `19bcf80ffad554acd496f3444bad42161281080cfca076dfe051c362fc521059` |
| `test_gelu` | `True` | `1139336` | `7b7d5438b86f1904bbc68d0017b90c5302c9e7f63e1991e3686ac1913665b568` |
| `test_fused_classifier` | `True` | `1164032` | `6c4d1f2fbd9a1f90d5237c69473efb4a9fd96239a7bf0ce88a08ff44bf299a4a` |
| `test_encoder` | `True` | `1165512` | `d69055becfcea94401db46eea2e4b41eea741537ac962aab2c40b70f116f8a80` |
| `test_adamw` | `True` | `1138408` | `df493758a0c4aeb14ebafcf6d2e6619a8ccb666a2c41af32d6cffacf1cee7956` |
| `test_global_norm` | `True` | `1138880` | `b14847044cc243944cabd3aa40fdbdea19e057918339c0fe4e4b16a49e8d5b5b` |
| `bench_sm120_matmul` | `True` | `2373912` | `810e1867b37ea42f9c08354106dbbcddfd4d0cbbefe51f1329a98c23660e467f` |
| `bench_sm120_attention` | `True` | `1731352` | `ad526a931bd4b41704387098fee519f1ad2b735c4b5393b1fd324ab69dea9961` |
| `bench_sm120_layernorm` | `True` | `1233728` | `4c251e1d3c65dedbbf90ff382296a94e1df3ca89df1c4e932c2882160ca78454` |
| `bench_sm120_runtime` | `True` | `2199032` | `e99edf5b4241b7571ad3b14a0b7f312c04eee6afc76c4747b13124c775d686cd` |
| `train_gpt2cu` | `True` | `3045944` | `5b9abb7520f66b9a743c82a094204c489bf4f1ed35b3054df0cf7b8a3e640b26` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
