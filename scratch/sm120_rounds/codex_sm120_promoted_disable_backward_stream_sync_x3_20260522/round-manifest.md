# SM120 Round Manifest

- run label: `codex_sm120_promoted_disable_backward_stream_sync_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_disable_backward_stream_sync_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_disable_backward_stream_sync_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `647`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `cf47b58a9b6994c7d854c933e89cc9a5f415c9f183b6290a0ed15b9c8ab89c29` |
| `test_attention` | `True` | `1800528` | `06108fad472113cd51d6a33bb7493c66b4f14e8d19d5b8881f01b691885f27b4` |
| `test_layernorm` | `True` | `1278296` | `855dd665d6698d5950644096464964b7567bd7a1da8b9bd8cfd335f70a97d78e` |
| `test_bias` | `True` | `2089120` | `c60a72734f9d5a34e647e27b25ab6d4dc1ae20398989e2d69d12e213f3505e7b` |
| `test_gelu` | `True` | `1179912` | `6e6f7f3ab3de17fc2bf556216309264a903927027dc0a6cf25189de9f772581a` |
| `test_fused_classifier` | `True` | `1208704` | `692afae4324a6ffafbd228f2faf5afab231a405a796724275f21a0b085407b9d` |
| `test_encoder` | `True` | `1210168` | `eafa2076b2e8979c9b46fa4cc79c85cf52b05ce86a5cafde362278ad34dcb464` |
| `test_adamw` | `True` | `1183768` | `c7b7bdff465ea9ff35501744e5e2bedd482b887ca581bbe5f86085565a09b547` |
| `test_global_norm` | `True` | `1179464` | `1f3757c0f273646f68c9d4e7691c9266d3535445f7f20486be59f8483d3b0a87` |
| `bench_sm120_matmul` | `True` | `2410304` | `4cb7d36afd746cef5e8bad926088a693b96922cf16903656d3ad7db8d2b18000` |
| `bench_sm120_attention` | `True` | `1768800` | `2438489366d9de742d414fdd13cd416850ed83bb26b36f13af56f05ff4f71bb1` |
| `bench_sm120_layernorm` | `True` | `1274232` | `dbf77fe0a9f3da27d855362363fb6945dacc29caa47a62f6781ec306078bd80f` |
| `bench_sm120_runtime` | `True` | `2271168` | `c6ad4dca3253eddb021450f9956b0edb3e3fec8725dbf9362089dc7b86fd11dc` |
| `train_gpt2cu` | `True` | `3105552` | `02132d4e76f982f2e7112e870526c99321ae75993414964c589648644f617bc6` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
