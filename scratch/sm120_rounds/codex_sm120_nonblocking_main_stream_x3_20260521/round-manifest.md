# SM120 Round Manifest

- run label: `codex_sm120_nonblocking_main_stream_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_nonblocking_main_stream_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_nonblocking_main_stream_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `543`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `5d3a775517bb75785748e1eea0ec7fd061fbd78bcad8c8913cb5ad5ade902055` |
| `test_attention` | `True` | `1760032` | `6de5faebfdb0383e3259b3114ad66fd65993f13ae0911d64030d0b7fb0f657bc` |
| `test_layernorm` | `True` | `1237784` | `71055224661a3a692440b9d8986ec75f1f35466f00d823f218325f1215692191` |
| `test_bias` | `True` | `2048616` | `914caaef4ad2371c64af660809e66b63763d8c12f85e5057f9b500af33dd38a8` |
| `test_gelu` | `True` | `1139336` | `15119708ec8b5e95e4894fa41bab1437b51a7ce445438229b60c44079799c83b` |
| `test_fused_classifier` | `True` | `1164032` | `ab45b6b25dc78579fc250bfb3fe837d401d62f5da2f0d9c60f5c4e4d059d9d5f` |
| `test_encoder` | `True` | `1165512` | `6703189d1f2043859763d039545d0cb0587bb8365484fc9c13ed17fbc1ffd6f7` |
| `test_adamw` | `True` | `1138408` | `82334dd83c47f8ca7d2aa9620fd5df04cae9a925f954ddbb801e00e00e587c8f` |
| `test_global_norm` | `True` | `1138880` | `a50e2424d4bceb73efabaff60b936f0d876e124d2948112ac1b8b09f0d3c451c` |
| `bench_sm120_matmul` | `True` | `2373912` | `3221caa9777ecc154400a5218c449449af0f77657e6ed419b351267dd36086a0` |
| `bench_sm120_attention` | `True` | `1728312` | `1499f40cd206bed79738dab748c3487cb67e57950f28de5ce1df13a5529b7b9b` |
| `bench_sm120_layernorm` | `True` | `1233728` | `e9511295c63809604371e8ff8f5a614d86caa34fab2b043734c229ed6590613b` |
| `bench_sm120_runtime` | `True` | `2221864` | `3cb88dab57d561b303d3771fd711d5ae4dd73c5ce3e1a5730f2a1e033b786ac0` |
| `train_gpt2cu` | `True` | `3060032` | `a250c33834e6b6d61869edd00f66c02021b3b89d5a6a500ce04320933af4427e` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
