# SM120 Round Manifest

- run label: `codex_sm120_promoted_matmul_dbias256_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias256_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_matmul_dbias256_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `664`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `e7bedb6015e2a4e4902883b2c7125fb7de50494e4f9730768e88e5b709b54f12` |
| `test_attention` | `True` | `1800528` | `547d9ca9b21b6cd33f830f84d220abd6c807ecb8b1b1c66915bd77e06eb8cc34` |
| `test_layernorm` | `True` | `1278296` | `7b67cce13e4bc41bc358fa0024874211c8b6a65684e4b1d756cd8cc5408bada1` |
| `test_bias` | `True` | `2089120` | `8b7c24c4797ca59d3d8e3728d7b18dd0f7803daf0696001e9c45521b9a9311ef` |
| `test_gelu` | `True` | `1179912` | `84425413d07250a7323e5c54a02033e6404ce273927e7feb5afec1f02a286f4d` |
| `test_fused_classifier` | `True` | `1208704` | `42ae4eb956618577f501ee1ac35734bbc734d77551d04413e2e4c2816ad3ac42` |
| `test_encoder` | `True` | `1210168` | `c97ebd2303cd9ecf057124cc1f22672f26acd647213779a3db622834cb10cb44` |
| `test_adamw` | `True` | `1183768` | `23fab04b74c47b8a2ccca1d6dfb811ddcb309ca9e8f901b10d416c958265eb9c` |
| `test_global_norm` | `True` | `1179464` | `ab83327780ea162b9cdfd916e60f916c2e3c6f67ce1f803c2629cf2f68187e9c` |
| `bench_sm120_matmul` | `True` | `2410304` | `57a2934c7c8508edba34a5351a6415b3ced650ed80add6e196b2d26f83087a7a` |
| `bench_sm120_attention` | `True` | `1768800` | `386fce98f4741e680a97853c5c83308853621347512379658e9afde41568096a` |
| `bench_sm120_layernorm` | `True` | `1274232` | `c2d933e212184f4e12d0d587c88d2805028cbc0768f1c5db7f113b69f7d7e836` |
| `bench_sm120_runtime` | `True` | `2271168` | `826ba61a2a427ac7adaf0124284161b453ed4cb4cbeaf3f06c0d38d4d4d61358` |
| `train_gpt2cu` | `True` | `3105552` | `e8ccfa7fa34cbe175ffab1cd3f6c9525c3fccf4f59c703e3bd74c7cec626b6b5` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
