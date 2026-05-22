# SM120 Round Manifest

- run label: `codex_sm120_disable_cublas_bwd_maxconn8_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_disable_cublas_bwd_maxconn8_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_disable_cublas_bwd_maxconn8_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `643`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173408` | `ccab78b3eb020f1baa62446fe3036328155f13dc1fe486bc60e843b5e400bdac` |
| `test_attention` | `True` | `1800528` | `c906d25e0976bd0af524c1027a511087e004e9d494fdb26bc9fa0b2219bd5107` |
| `test_layernorm` | `True` | `1278296` | `0e7120406836e54d3e83ca519d7e0a1dccb17e0817875fa67adc3a8a5badbd8a` |
| `test_bias` | `True` | `2089120` | `7c70a6d26a4a8541b105d83742b862e83feb489e6ea5f2cb88e17dae0a956680` |
| `test_gelu` | `True` | `1179912` | `558769af87b07d676dfeaceed03fbf03f7452c91c517b58395b0ef485fe42d4f` |
| `test_fused_classifier` | `True` | `1208704` | `1575e17e5fefccc29967768300776b4960a2b0ed2847215164b69e1c0dcf18bf` |
| `test_encoder` | `True` | `1210168` | `90583d0194bb8f62f5d2938f26f3d1a6de5594c437751e50de7d8948287328ce` |
| `test_adamw` | `True` | `1183768` | `3411538f793bc732d16f9102bb6b85718b71cdeefed6aa05598ebbf929547e88` |
| `test_global_norm` | `True` | `1179464` | `7587d5db0123d9b0840697b220037759eca60d55b3f6a613477b54718f06afda` |
| `bench_sm120_matmul` | `True` | `2410304` | `2323a5cf4d735e1546f0418cd7522057fb9342e4d21c75d14811455b10875eb5` |
| `bench_sm120_attention` | `True` | `1768800` | `a85122be57ed92f61da414a8ebe7aba085f1c9bdd3c9584d8c2e3f9e21fe2fb1` |
| `bench_sm120_layernorm` | `True` | `1274232` | `d6218a4ab1fcf03a27ebafdbcc02db8434d0bbdcb4a5c9499029639d78ccc765` |
| `bench_sm120_runtime` | `True` | `2271168` | `3f97e2178013baf6ed4966393b352319ee1e6b635cadcbfa9d30d7fe34313212` |
| `train_gpt2cu` | `True` | `3105256` | `3839f75c3d9c3427c4c1db075af41af009660f01a231bffdb264f78b0ed8c83a` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
