# SM120 Round Manifest

- run label: `codex_sm120_round_memory_shape_coverage_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_memory_shape_coverage_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_memory_shape_coverage_20260520`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `459`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2137216` | `aebc3d4271cdc7f2c209d7e0a74d1c5793a923d9c526f287e71b823183fc351c` |
| `test_attention` | `True` | `1760032` | `31b4e4e59224dd3b8a03027c837f38e55d0e1f302dde88fecdf19a8c6025c7f8` |
| `test_layernorm` | `True` | `1237784` | `e36884623f791a90918db7f923d947eeb08c77bf1fc034bf405f6ef81fb29e4d` |
| `test_bias` | `True` | `2048616` | `c4e51ac43c13762f6ca1fcd513658944e8c4a9ee5156b4db9e58bddeaa4439be` |
| `test_gelu` | `True` | `1139336` | `1d19f44acbf3f6fee61e29ebe4e37a3ce8ec993ce0d175b910b658b3598e3f03` |
| `test_fused_classifier` | `True` | `1164032` | `f00eee7cb2418dc2eef85e66e909d5bfda465211adba4b5c5c26c0bf31b1abb5` |
| `test_encoder` | `True` | `1165512` | `2b5860e230d9c64e947aa1d8ebca55c376cf2bbb341f723bf48326ee582032db` |
| `test_adamw` | `True` | `1138408` | `ca15c569acac20f726d603cae3d7cecae5a0d6a5a2a52ac007b6dd42d3c45589` |
| `test_global_norm` | `True` | `1138880` | `4ca5fc2e9d4b42760f6adb943b19f9705a3187ebe4234ff88538cc65cc95c0f3` |
| `bench_sm120_matmul` | `True` | `2373912` | `fcf2a436f55703be4ad268c7b164d64fbff4619e6e8fad2d5eeecc57d162508e` |
| `bench_sm120_attention` | `True` | `1731352` | `0585634fbf34c589c4ee52faf1fb8867f711f35b973c399597a9551b1f70f25d` |
| `bench_sm120_layernorm` | `True` | `1233728` | `510d3022e07af60f93d70294d63c9520b1c68f0b809d01b04c1079539d6cc4c2` |
| `bench_sm120_runtime` | `True` | `2199032` | `419f4cfb732cef5c39db3795ab90fc6dec0392f3b2b49bd4b080fd0640cc7c2f` |
| `train_gpt2cu` | `True` | `3045944` | `3143a9669c870c35d38ce6a16bfd40ed51100a3b99f7bd92a0fa8efee257cd20` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
