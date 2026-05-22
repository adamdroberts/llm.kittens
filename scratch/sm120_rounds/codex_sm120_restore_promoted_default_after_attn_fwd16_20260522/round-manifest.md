# SM120 Round Manifest

- run label: `codex_sm120_restore_promoted_default_after_attn_fwd16_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_attn_fwd16_20260522`
- train output dir: `log124M/5090_S_codex_sm120_restore_promoted_default_after_attn_fwd16_20260522`
- device arch: `SM120`
- max steps: `0`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `639`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `019e5a47b27ca2f7735654cac252fd5f85d582c23eed373b43a6472ee3a8bd9a` |
| `test_attention` | `True` | `1800528` | `1d1fd3a49d66bfd0251221ebbe9784327a33bac31ced37b97da4c737f6e93a2d` |
| `test_layernorm` | `True` | `1278296` | `7691f10641b3751bf510ff9e893b06af9aee33b553c9b6815ec74de6332510f2` |
| `test_bias` | `True` | `2089120` | `486192d9874d9cb1d956b52a933d4d1cfecf59bd0eb79cde1cfe2cdff2cde5cc` |
| `test_gelu` | `True` | `1179912` | `3f0c692555dfe00386d438c34bc48570a1846242377e3d2332a39f6d6ad9177e` |
| `test_fused_classifier` | `True` | `1208704` | `ac64d81185e7981767eef3573d179fd143248751446dff1295a126c020cb5325` |
| `test_encoder` | `True` | `1210168` | `c702e03e187980585af1e1658d6e0cf15d8aa332d6c4d214c323d9579252eb28` |
| `test_adamw` | `True` | `1183768` | `f2add334cfc67ed5f3918a687225a5189f232dfa57783090ee28ada87c9245a5` |
| `test_global_norm` | `True` | `1179464` | `6f846de783c9398fc2c71313c08806b864d10ede1f6b42c1946e2ee2c8d2219f` |
| `bench_sm120_matmul` | `True` | `2410304` | `03544dfffdf704a088ec2e69319f3fb94cbf4ea5c52318cf9b15f7eb63552396` |
| `bench_sm120_attention` | `True` | `1768800` | `1a42194645dbe5dbc87b3987170def6de2126ed8fcab53da0c00b898927132f9` |
| `bench_sm120_layernorm` | `True` | `1274232` | `b8d4953dbf84f85b8e38e979e636636be7bbd3de0b1001f2db90a8cd754bd420` |
| `bench_sm120_runtime` | `True` | `2271168` | `a7abf092f8d6fa3326104fc279070130d8d8e02b70824a9fa44b4952adf2aec1` |
| `train_gpt2cu` | `True` | `3105552` | `930321ac2faefca3e9300795756cd37b831923885ed36b551a181e51b2450ff2` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
