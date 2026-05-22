# SM120 Round Manifest

- run label: `codex_sm120_promoted_async_grad_norm_copy_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_async_grad_norm_copy_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_async_grad_norm_copy_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `630`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `c8322d24f21c23350e18438dadda8d771e8e3c6444f49aafa84338c567e83998` |
| `test_attention` | `True` | `1800528` | `eb1829db674a100dddfbc2f064b08e5fa1dcab836005e24b8cdeb90710ccf2ba` |
| `test_layernorm` | `True` | `1278296` | `50adff4186da2d669a6efabcc83bd8268e03f08dc0c7457df158f9caac6cfdf8` |
| `test_bias` | `True` | `2089120` | `72f3548b5e77dc68f283c6710dd21e9e84b28433c60ac7fba04843f29da83f17` |
| `test_gelu` | `True` | `1179912` | `e92578c395a3ff0a481c724ac255341a9b6a63935ee906b80faac582d1267772` |
| `test_fused_classifier` | `True` | `1208704` | `e3855c30e9cff384bfedcf0101b55b91779ffbcfb3b262cf6829e88d343dccbc` |
| `test_encoder` | `True` | `1210168` | `69c5c9d2650aa545193b349ff09a9e797d63bf2762864f1e59675ef3e4d903f7` |
| `test_adamw` | `True` | `1183768` | `64705b27443b85a71751279bfb479eff580a471e65d1ebcc45b3cdcc8dcb6835` |
| `test_global_norm` | `True` | `1179464` | `6a34993e85fbbbd562cf1cf9680888faee1a2f1e1df9b346685c16c15683f06c` |
| `bench_sm120_matmul` | `True` | `2410304` | `a7d135b14c75a37cd3651efaebfb713fafbc0524db260ca178c11859bb3bf41f` |
| `bench_sm120_attention` | `True` | `1768800` | `a6b0fee506f6d4494a9988badbdc92d9e7a2a616475b7d1b8f30ee89d4c927b7` |
| `bench_sm120_layernorm` | `True` | `1274232` | `4d09f714f49ea776551f6fe97682bbca1de441fad6b85320c57a998349be57f7` |
| `bench_sm120_runtime` | `True` | `2271168` | `1bc0ec47de44fc37f5f757c26650633fd040d1b22a8c6251f6682cf215993011` |
| `train_gpt2cu` | `True` | `3105552` | `a0e82a97006ae371b599ba6eea57fb1c01f7fcd876b860100307efbf05e7fd98` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
